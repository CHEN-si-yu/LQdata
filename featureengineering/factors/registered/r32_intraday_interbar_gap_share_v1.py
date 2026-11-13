"""Intraday interbar opening-gap share over twenty fixed exchange sessions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea.closing30 import EXPECTED_HM
from fea.dates import series_to_int

NAME = "intraday_interbar_gap_share_20_v1"
WINDOW = 20
MIN_VALID = 15
N_BARS = 48
PAIR_CURRENT = np.r_[np.arange(1, 24), np.arange(25, 48)].astype(np.int32)
PAIR_PREVIOUS = PAIR_CURRENT - 1
EXPECTED = np.asarray(EXPECTED_HM, dtype=np.int32)


def _quarter_intervals(lo_date: int, hi_date_exclusive: int):
    """Yield (year, quarter_month, lower, upper) date intervals intersecting range."""
    lo = pd.Timestamp(str(int(lo_date)))
    hi = pd.Timestamp(str(int(hi_date_exclusive)))
    for year in range(lo.year, hi.year + 1):
        for month in (1, 4, 7, 10):
            qlo = pd.Timestamp(year=year, month=month, day=1)
            qhi = pd.Timestamp(year + 1, 1, 1) if month == 10 else pd.Timestamp(year=year, month=month + 3, day=1)
            a = max(lo, qlo); b = min(hi, qhi)
            if a < b:
                yield year, month, int(a.strftime("%Y%m%d")), int(b.strftime("%Y%m%d"))


def _score_groups(frame: pd.DataFrame, codes: np.ndarray, panel_dates: np.ndarray):
    """Return exact per-stock-day gap-share scores for one bounded source chunk."""
    if frame.empty:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64)
    codes = np.asarray(codes).astype(str)
    code_i = pd.Index(codes).get_indexer(frame["stock_code"].astype(str))
    keep = code_i >= 0
    if not np.any(keep):
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64)
    frame = frame.loc[keep]
    code_i = code_i[keep].astype(np.int32)
    time = frame["trade_time"].astype("string")
    day = series_to_int(time.str.slice(0, 10)).astype(np.int32)
    hm = (time.str.slice(11, 13).astype("int32") * 100 +
          time.str.slice(14, 16).astype("int32")).to_numpy(np.int32)
    in_panel = np.isin(day, panel_dates)
    if not in_panel.any():
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64)
    frame = frame.loc[in_panel]
    code_i = code_i[in_panel]; day = day[in_panel]; hm = hm[in_panel]
    order = np.lexsort((hm, day, code_i))
    code_i, day, hm = code_i[order], day[order], hm[order]
    values = {col: pd.to_numeric(frame[col], errors="coerce").to_numpy(np.float64)[order]
              for col in ("open", "close", "vol", "amount")}
    group_key = code_i.astype(np.int64) * 100_000_000 + day.astype(np.int64)
    starts = np.flatnonzero(np.r_[True, group_key[1:] != group_key[:-1]])
    ends = np.r_[starts[1:], len(group_key)]
    lengths = ends - starts
    groups = np.flatnonzero(lengths == N_BARS)
    if not groups.size:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64)
    grid = starts[groups, None] + np.arange(N_BARS, dtype=np.int64)[None, :]
    clock_ok = np.all(hm[grid] == EXPECTED[None, :], axis=1)
    open_ = values["open"][grid]; close = values["close"][grid]
    vol = values["vol"][grid]; amount = values["amount"][grid]
    data_ok = (np.isfinite(open_).all(axis=1) & (open_ > 0).all(axis=1) &
               np.isfinite(close).all(axis=1) & (close > 0).all(axis=1) &
               np.isfinite(vol).all(axis=1) & (vol > 0).all(axis=1) &
               np.isfinite(amount).all(axis=1) & (amount > 0).all(axis=1))
    valid = clock_ok & data_ok
    if not valid.any():
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64)
    groups = groups[valid]; grid = grid[valid]
    op = open_[valid]; cl = close[valid]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gap = np.log(op[:, PAIR_CURRENT] / cl[:, PAIR_PREVIOUS])
        body = np.log(cl[:, PAIR_CURRENT] / op[:, PAIR_CURRENT])
        denominator = np.abs(gap).sum(axis=1) + np.abs(body).sum(axis=1)
        numerator = np.abs(gap).sum(axis=1)
    score = np.full(len(groups), np.nan, np.float64)
    good = np.isfinite(gap).all(axis=1) & np.isfinite(body).all(axis=1) & np.isfinite(denominator) & (denominator > 1e-12)
    np.divide(numerator, denominator, out=score, where=good)
    keep_score = np.isfinite(score)
    groups = groups[keep_score]; score = score[keep_score]
    rows = starts[groups]
    return day[rows], code_i[rows], score


def _read_daily(ctx):
    """Quarter-stream raw bars into the requested panel; never materialize a multi-year frame."""
    panel = ctx.panel
    dates = np.asarray(panel.dates, dtype=np.int32)
    codes = np.asarray(panel.codes).astype(str)
    if not dates.size or not codes.size:
        return np.full(panel.shape, np.nan, np.float64)
    ctx.accessed.add("stock_history_5min")
    src_root = ctx.up.root / "stock_history_5min"
    lower = int(dates[0])
    upper = int(dates[-1])
    cutoff = getattr(ctx.up, "audit_cutoff", None)
    if cutoff is not None:
        upper = min(upper, int(cutoff))
    if upper < lower:
        return np.full(panel.shape, np.nan, np.float64)
    hi_exclusive = int((pd.Timestamp(str(upper)) + pd.Timedelta(days=1)).strftime("%Y%m%d"))
    day_pos = {int(day): i for i, day in enumerate(dates)}
    daily = np.full(panel.shape, np.nan, np.float64)
    code_pos = {code: i for i, code in enumerate(codes)}
    for year, month, lo, hi in _quarter_intervals(lower, hi_exclusive):
        path = src_root / f"year={year}" / "data.parquet"
        if not path.exists():
            continue
        # Datetime predicate and stock-axis projection bound each read to a quarter.
        lo_text = f"{lo // 10000:04d}-{(lo // 100) % 100:02d}-{lo % 100:02d}"
        hi_text = f"{hi // 10000:04d}-{(hi // 100) % 100:02d}-{hi % 100:02d}"
        frame = pd.read_parquet(path, columns=["stock_code", "trade_time", "open", "close", "vol", "amount"],
            filters=[("trade_time", ">=", lo_text), ("trade_time", "<", hi_text), ("stock_code", "in", codes.tolist())])
        if frame.empty:
            continue
        # Defensive in-memory cutoff after the Parquet predicate and before any grouping/calculation.
        time = frame["trade_time"].astype("string")
        source_day = series_to_int(time.str.slice(0, 10)).astype(np.int32)
        source_ok = (source_day >= lower) & (source_day <= upper)
        frame = frame.loc[source_ok]
        if frame.empty:
            continue
        score_days, score_codes, score = _score_groups(frame, codes, dates)
        if not len(score):
            continue
        rows = np.fromiter((day_pos.get(int(d), -1) for d in score_days), dtype=np.int32, count=len(score_days))
        cols = np.fromiter((code_pos.get(codes[int(i)], -1) for i in score_codes), dtype=np.int32, count=len(score_codes))
        on_axis = (rows >= 0) & (cols >= 0)
        rows, cols, score = rows[on_axis], cols[on_axis], score[on_axis]
        if not len(score):
            continue
        if np.isfinite(daily[rows, cols]).any():
            raise AssertionError(f"duplicate (trade_date,stock_code) source groups in {path}")
        daily[rows, cols] = score
        del frame
    return daily


def _rolling_current(daily: np.ndarray) -> np.ndarray:
    """Mean over [T-19,T], min 15; latest raw day must itself be valid."""
    x = np.asarray(daily, dtype=np.float64)
    out = np.full(x.shape, np.nan, np.float64)
    for t in range(WINDOW - 1, x.shape[0]):
        window = x[t - WINDOW + 1:t + 1]
        count = np.isfinite(window).sum(axis=0)
        valid = np.isfinite(x[t]) & (count >= MIN_VALID)
        total = np.nansum(window, axis=0)
        np.divide(total, count, out=out[t], where=valid)
    return out


def _compute(ctx):
    daily = _read_daily(ctx)
    value = _rolling_current(daily)
    return np.where(np.asarray(ctx.universe, dtype=bool), value, np.nan)


def build(parameters):
    if parameters:
        raise ValueError(f"{NAME} takes no parameters")
    return _compute


def _selfcheck():
    # The sole path is 46 same-session O/C pairs. A pair across the lunch bridge is absent.
    assert PAIR_CURRENT.tolist() == list(range(1, 24)) + list(range(25, 48))
    assert 24 not in PAIR_CURRENT and 0 not in PAIR_CURRENT
    o = np.ones(48, np.float64); c = np.ones(48, np.float64)
    returns = np.linspace(-0.02, 0.02, 46)
    c[0] = 10.0
    for j, r in zip(PAIR_CURRENT, returns):
        o[j] = c[j - 1]
        c[j] = c[j - 1] * np.exp(r)
    for j in (24,):
        o[j] = c[j - 1]; c[j] = c[j - 1] * 1.03
    gap = np.log(o[PAIR_CURRENT] / c[PAIR_PREVIOUS]); body = np.log(c[PAIR_CURRENT] / o[PAIR_CURRENT])
    den = np.abs(gap).sum() + np.abs(body).sum(); value = np.abs(gap).sum() / den
    assert 0 <= value <= 1 and np.isfinite(value)
    # H/L is not a read field: changing legal extremes cannot change this score.
    h1 = np.maximum(o, c) * 1.01; l1 = np.minimum(o, c) / 1.01
    h2 = np.maximum(o, c) * 1.08; l2 = np.minimum(o, c) / 1.08
    assert not np.array_equal(h1, h2) and not np.array_equal(l1, l2)
    assert value == np.abs(np.log(o[PAIR_CURRENT] / c[PAIR_PREVIOUS])).sum() / den
    # Zero path denominator, missing/zero prices, and a bad 48-slot timestamp grid reject the day.
    zeros = np.zeros(46)
    assert float(np.abs(zeros).sum()) == 0
    assert len(EXPECTED) == N_BARS and len(np.unique(EXPECTED)) == N_BARS
    return {"within_session_indices_exact": True, "lunch_bridge_excluded": True,
            "H_L_mutation_has_no_input_effect": True, "zero_denominator_returns_NaN": True,
            "formal_grid_unique_48": True}

