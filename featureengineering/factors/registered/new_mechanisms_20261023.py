"""Four frozen 2026-10-23 mechanisms approved for production migration.

The research runner remains outside production.  This module contains only the
fixed candidate definitions and reads the declared upstream datasets through
FactorContext / its upstream reader; it never imports ``work/`` code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


NAMES = (
    "annual_report_lag_median3_v1",
    "pe_own_history_percentile_252_20",
    "toplist_multi_reason_breadth_20_v1",
    "corwin_schultz_overnight_adjusted_hl_proxy_20",
)


def _fixed_window_mean(values: np.ndarray, width: int, min_count: int) -> np.ndarray:
    """Ordered fixed-slot mean; missing slots stay in place."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or width < 1 or not 0 <= min_count <= width:
        raise ValueError("invalid fixed-window panel/parameters")
    t_count, c_count = x.shape
    out = np.full_like(x, np.nan)
    for t in range(width - 1, t_count):
        window = x[t - width + 1:t + 1]
        finite = np.isfinite(window)
        count = finite.sum(axis=0)
        sums = np.where(finite, window, 0.0).sum(axis=0)
        np.divide(sums, count, out=out[t], where=(count >= min_count) & (count > 0))
        if min_count == 0:
            out[t, count == 0] = 0.0
    return out


def _fixed_window_sum(values: np.ndarray, width: int, min_count: int) -> np.ndarray:
    """Ordered fixed-slot sum with a finite-observation threshold."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or width < 1 or not 0 <= min_count <= width:
        raise ValueError("invalid fixed-window panel/parameters")
    t_count, c_count = x.shape
    sums = np.zeros((t_count, c_count), dtype=np.float64)
    counts = np.zeros((t_count, c_count), dtype=np.int16)
    for lag in range(width):
        shifted = np.full_like(x, np.nan)
        if lag == 0:
            shifted[:] = x
        elif lag < t_count:
            shifted[lag:] = x[:-lag]
        finite = np.isfinite(shifted)
        sums += np.where(finite, shifted, 0.0)
        counts += finite.astype(np.int16)
    sums[counts < min_count] = np.nan
    return sums


def own_history_percentile(pe: np.ndarray, width: int = 252,
                           min_history: int = 180) -> np.ndarray:
    """Same-stock PE percentile on an inclusive fixed-slot history window."""
    x = np.asarray(pe, dtype=np.float64)
    if x.ndim != 2 or width < 1 or not 1 <= min_history <= width:
        raise ValueError("invalid PE panel/window")
    x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
    out = np.full_like(x, np.nan)
    for t in range(width - 1, len(x)):
        window = x[t - width + 1:t + 1]
        current = x[t]
        valid = np.isfinite(window)
        n = valid.sum(axis=0)
        less = (valid & (window < current[None, :])).sum(axis=0)
        equal = (valid & (window == current[None, :])).sum(axis=0)
        keep = np.isfinite(current) & (n >= min_history)
        out[t, keep] = (less[keep] + 0.5 * equal[keep]) / n[keep]
    return out


def _date_int(series: pd.Series) -> np.ndarray:
    text = series.astype("string").str.slice(0, 10)
    compact = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return (parsed.dt.year * 10000 + parsed.dt.month * 100 + parsed.dt.day).fillna(0).to_numpy(np.int32)


def _annual_report_lag(ctx) -> np.ndarray:
    """First-vintage annual filing-delay median, PIT-effective next session."""
    # The event table is read directly because the required ann/f_ann/end
    # dates are not Derivative fields; keep production dependency tracking exact.
    ctx.accessed.add("stock_income")
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    codes = np.asarray(ctx.panel.codes).astype(str)
    if dates.size == 0:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    code_pos = {code: i for i, code in enumerate(codes)}
    sessions = np.asarray(ctx.cal.days, dtype=np.int32)
    last_year = int(dates[-1]) // 10000
    source = Path(ctx.cfg.upstream)
    parts = []
    columns = ["stock_code", "ann_date", "f_ann_date", "end_date", "comp_type", "report_type"]
    for year in range(2010, last_year + 1):
        path = source / "stock_income" / f"year={year}" / "data.parquet"
        if not path.exists():
            continue
        # Annual income partitions are keyed by fiscal period end; only the
        # small frozen column set is materialized.
        frame = pd.read_parquet(path, columns=columns)
        if frame.empty:
            continue
        period = frame.end_date.astype("string")
        frame = frame[(frame.comp_type.astype(str) == "1") &
                      (frame.report_type.astype(str) == "1") &
                      period.str.endswith("12-31", na=False)]
        if frame.empty:
            continue
        frame = frame[frame.stock_code.astype(str).isin(code_pos)].copy()
        if frame.empty:
            continue
        frame["_end"] = _date_int(frame.end_date)
        frame["_ann"] = _date_int(frame.ann_date)
        frame["_fann"] = _date_int(frame.f_ann_date)
        valid_ann = frame["_ann"] >= 19900101
        valid_fann = frame["_fann"] >= 19900101
        frame = frame[(frame["_end"] > 0) & (valid_ann | valid_fann)].copy()
        if frame.empty:
            continue
        frame["_pit"] = np.maximum.reduce((
            np.where(frame["_ann"].to_numpy() >= 19900101, frame["_ann"].to_numpy(), 0),
            np.where(frame["_fann"].to_numpy() >= 19900101, frame["_fann"].to_numpy(), 0),
            frame["_end"].to_numpy(),
        ))
        frame = frame[frame["_pit"] <= int(dates[-1])]
        audit_cutoff = getattr(ctx.up, "audit_cutoff", None)
        if audit_cutoff is not None:
            frame = frame[frame["_pit"] <= int(audit_cutoff)]
        if not frame.empty:
            parts.append(frame[["stock_code", "_end", "_pit"]])
    out = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    if not parts:
        return out
    records = pd.concat(parts, ignore_index=True)
    # First disclosed vintage is immutable; stable tie order mirrors the
    # frozen runner's sort-by-PIT then drop-duplicate behavior.
    records = records.sort_values(["_pit"], kind="stable").drop_duplicates(
        ["stock_code", "_end"], keep="first")
    end_days = records["_end"].to_numpy(np.int32)
    pit_days = records["_pit"].to_numpy(np.int32)
    release_pos = np.searchsorted(sessions, pit_days, side="right")
    inside = release_pos < sessions.size
    release_days = np.zeros(len(records), dtype=np.int32)
    release_days[inside] = sessions[release_pos[inside]]
    code_ids = records.stock_code.astype(str).map(code_pos).to_numpy(np.int64)
    delay = (pd.to_datetime(pit_days.astype(str), format="%Y%m%d") -
             pd.to_datetime(end_days.astype(str), format="%Y%m%d")).days.to_numpy(np.int32)
    events = [(int(release_days[i]), int(code_ids[i]), int(end_days[i]), -int(delay[i]))
              for i in range(len(records)) if inside[i] and release_days[i] <= dates[-1]]
    events.sort(key=lambda item: (item[0], item[1], item[2]))
    history: list[list[tuple[int, int]]] = [[] for _ in codes]
    dpos = {int(day): i for i, day in enumerate(dates)}
    event_i = 0
    for day_i, day in enumerate(dates):
        while event_i < len(events) and events[event_i][0] <= int(day):
            _, code_i, end_i, score = events[event_i]
            if all(old_end != end_i for old_end, _ in history[code_i]):
                history[code_i].append((end_i, score))
                history[code_i].sort(key=lambda row: row[0])
            event_i += 1
        for code_i, values in enumerate(history):
            if len(values) >= 3:
                out[day_i, code_i] = float(np.median([v for _, v in values[-3:]]))
    traded = np.asarray(ctx.traded(), dtype=bool)
    return np.where(traded & np.asarray(ctx.universe, dtype=bool), out, np.nan)


def _exact_daily_panel(ctx, dataset: str, field: str) -> np.ndarray:
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    codes = np.asarray(ctx.panel.codes).astype(str)
    years = (int(dates[0]) // 10000, int(dates[-1]) // 10000)
    frame = ctx.dataset(dataset, columns=["stock_code", "trade_date", field], years=years)
    out = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    if frame.empty:
        return out
    day = _date_int(frame.trade_date)
    dpos = {int(value): i for i, value in enumerate(dates)}
    cpos = {str(value): i for i, value in enumerate(codes)}
    ti = pd.Series(day).map(dpos).to_numpy()
    ci = frame.stock_code.astype(str).map(cpos).to_numpy()
    keep = pd.notna(ti) & pd.notna(ci)
    if keep.any():
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(np.float64)
        out[ti[keep].astype(np.int64), ci[keep].astype(np.int64)] = values[keep]
    return out


def _pe_history_percentile(ctx) -> np.ndarray:
    pe = _exact_daily_panel(ctx, "stock_finance", "pe_ttm")
    pct = own_history_percentile(pe, width=252, min_history=180)
    return _fixed_window_mean(pct, width=20, min_count=15)


def _toplist_multi_reason(ctx) -> np.ndarray:
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    codes = np.asarray(ctx.panel.codes).astype(str)
    years = (int(dates[0]) // 10000, int(dates[-1]) // 10000)
    rows = ctx.dataset("stock_top_list", columns=["trade_date", "stock_code", "reason"], years=years)
    raw = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    if not rows.empty:
        day = _date_int(rows.trade_date)
        # A source date is covered when at least one row exists anywhere in
        # the upstream day's dataset.  On such dates, absent stock events are 0.
        covered = np.isin(dates, np.unique(day[day > 0]))
        raw[covered, :] = 0.0
        cpos = {str(value): i for i, value in enumerate(codes)}
        dpos = {int(value): i for i, value in enumerate(dates)}
        frame = rows.copy()
        frame["_day"] = day
        frame["_code"] = frame.stock_code.astype(str)
        frame["_reason"] = frame.reason.astype("string").str.strip()
        frame["_missing"] = frame.reason.isna() | frame["_reason"].eq("")
        # Preserve the frozen behavior: an invalid/empty reason row makes
        # that stock-day unknown even if another row is present.
        bad = frame.loc[frame["_missing"], ["_code", "_day"]].drop_duplicates()
        bad_keys = set(bad.itertuples(index=False, name=None))
        for code, day_value in bad_keys:
            ti, ci = dpos.get(int(day_value)), cpos.get(str(code))
            if ti is not None and ci is not None:
                raw[ti, ci] = np.nan
        good = frame.loc[~frame["_missing"]].drop_duplicates(["_code", "_day", "reason"])
        counts = good.groupby(["_code", "_day"], sort=False)["reason"].nunique()
        for (code, day_value), n in counts.items():
            ti, ci = dpos.get(int(day_value)), cpos.get(str(code))
            if ti is not None and ci is not None and (str(code), int(day_value)) not in bad_keys:
                raw[ti, ci] = float(n >= 2)
    # The frozen score divides by the fixed window width (20), not by the
    # number of covered source dates; 15 covered slots is only a validity gate.
    roll = _fixed_window_sum(raw, width=20, min_count=15) / 20.0
    out = np.full_like(roll, np.nan)
    if len(out) > 1:
        out[1:] = roll[:-1]
    return out


def _cs_pair_vectorized(hp, lp, cp, hn, ln):
    """Stock-axis form kept operation-for-operation with the frozen runner."""
    hp, lp, cp, hn, ln = [np.asarray(v, dtype=np.float64) for v in (hp, lp, cp, hn, ln)]
    valid = (np.isfinite(hp) & np.isfinite(lp) & np.isfinite(cp) & np.isfinite(hn) & np.isfinite(ln)
             & (hp > 0) & (lp > 0) & (cp > 0) & (hn > 0) & (ln > 0)
             & (hp >= lp) & (hn >= ln) & (cp >= lp) & (cp <= hp))
    up_gap = valid & (cp < ln)
    down_gap = valid & (cp > hn)
    up = ln - cp
    down = cp - hn
    hn_adj = np.where(up_gap, hn - up, np.where(down_gap, hn + down, hn))
    ln_adj = np.where(up_gap, cp, np.where(down_gap, ln + down, ln))
    valid &= (hn_adj > 0) & (ln_adj > 0) & (hn_adj >= ln_adj)
    out = np.full(np.broadcast_shapes(hp.shape, lp.shape, cp.shape, hn.shape, ln.shape), np.nan,
                  dtype=np.float64)
    if not valid.any():
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.log(hp / lp) ** 2 + np.log(hn_adj / ln_adj) ** 2
        gamma = np.log(np.maximum(hp, hn_adj) / np.minimum(lp, ln_adj)) ** 2
        k = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        out[valid] = 2.0 * np.tanh(np.maximum(alpha[valid], 0.0) / 2.0)
    return out


def corwin_schultz_overnight_adjusted_hl_proxy_20(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, traded: np.ndarray,
    min_pairs: int = 15,
) -> np.ndarray:
    """Twenty fixed ending-pair Corwin–Schultz estimates with gap correction."""
    h, lo, c = (np.asarray(v, dtype=np.float64) for v in (high, low, close))
    tr = np.asarray(traded, dtype=bool)
    if h.ndim != 2 or lo.shape != h.shape or c.shape != h.shape or tr.shape != h.shape:
        raise ValueError("high/low/close/traded must be aligned (T,C) panels")
    if not 1 <= min_pairs <= 20:
        raise ValueError("min_pairs must be in [1,20]")
    pair = np.full(h.shape, np.nan, dtype=np.float64)
    for t in range(1, len(h)):
        active = tr[t - 1] & tr[t]
        pair[t] = _cs_pair_vectorized(h[t - 1], lo[t - 1], c[t - 1], h[t], lo[t])
        pair[t, ~active] = np.nan
    out = np.full_like(pair, np.nan)
    for t in range(len(h)):
        start = max(1, t - 19)
        window = pair[start:t + 1]
        valid = np.isfinite(window)
        n = valid.sum(axis=0)
        sums = np.where(valid, window, 0.0).sum(axis=0)
        np.divide(sums, n, out=out[t], where=n >= min_pairs)
    return out


def _cs_signal(ctx) -> np.ndarray:
    high = np.asarray(ctx.hfq("high"), dtype=np.float64)
    low = np.asarray(ctx.hfq("low"), dtype=np.float64)
    close = np.asarray(ctx.hfq("close"), dtype=np.float64)
    result = corwin_schultz_overnight_adjusted_hl_proxy_20(
        high, low, close, np.asarray(ctx.traded(), dtype=bool), min_pairs=15)
    return np.where(np.asarray(ctx.universe, dtype=bool), result, np.nan)


_BUILDERS = {
    "annual_report_lag_median3_v1": _annual_report_lag,
    "pe_own_history_percentile_252_20": _pe_history_percentile,
    "toplist_multi_reason_breadth_20_v1": _toplist_multi_reason,
    "corwin_schultz_overnight_adjusted_hl_proxy_20": _cs_signal,
}


def build(parameters):
    """Explicitly build one frozen candidate by its exact name."""
    if (not isinstance(parameters, dict) or set(parameters) != {"name"} or
            parameters.get("name") not in _BUILDERS):
        raise ValueError(f"name must be one of {NAMES}; got {parameters!r}")
    function = _BUILDERS[parameters["name"]]
    return lambda ctx: function(ctx)
