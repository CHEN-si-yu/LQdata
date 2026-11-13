"""Public reported-name Dragon-Tiger gross concentration load.

The reported ``org_name`` values are disclosure labels. They do not guarantee a
unique physical brokerage seat or beneficial investor (notably for anonymous
labels such as ``机构专用``). This module therefore computes a reported-name
concentration proxy only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


NAME = "dragon_tiger_reported_name_gross_load_20"


def _reported_name_gross_hhi(org_names, buy_amount, sell_amount) -> float:
    """HHI over distinct disclosure labels, collapsing equal-gross mirrors.

    If rows for the same public label imply different gross amounts, identity
    is ambiguous at the source's available granularity and the whole event is
    unknown. The function never treats a label as a real seat identifier.
    """
    names = np.asarray(org_names, dtype=object)
    buy = np.asarray(buy_amount, dtype=np.float64)
    sell = np.asarray(sell_amount, dtype=np.float64)
    if len(names) == 0 or len(names) != len(buy) or len(names) != len(sell):
        return float("nan")
    if not (np.isfinite(buy).all() and np.isfinite(sell).all() and
            (buy >= 0).all() and (sell >= 0).all()):
        return float("nan")
    gross = buy + sell
    if not np.isfinite(gross).all():
        return float("nan")
    buckets = {}
    for name, value in zip(names, gross):
        if name is None or pd.isna(name) or not str(name).strip():
            return float("nan")
        buckets.setdefault(str(name), set()).add(float(value))
    if any(len(values) != 1 for values in buckets.values()):
        return float("nan")
    values = np.asarray([next(iter(v)) for v in buckets.values()], dtype=np.float64)
    values = values[values > 0]
    total = values.sum()
    if len(values) < 2 or not np.isfinite(total) or total <= 0:
        return float("nan")
    return float(np.square(values / total).sum())


def _fixed_trailing_mean(x: np.ndarray, window: int = 20,
                         min_valid: int = 15) -> np.ndarray:
    """Frozen fixed-slot trailing mean: row T consumes [T-window, T)."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or window != 20 or min_valid != 15:
        raise ValueError("public-name gross-load uses fixed window=20, min_valid=15")
    out = np.full_like(a, np.nan)
    for t in range(window, len(a)):
        z = a[t - window:t]
        n = np.isfinite(z).sum(axis=0)
        ok = n >= min_valid
        if ok.any():
            out[t, ok] = np.nanmean(z[:, ok], axis=0)
    return out


def _compute(ctx):
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    codes = np.asarray(ctx.panel.codes).astype(str)
    shape = (len(dates), len(codes))
    if not len(dates) or not len(codes):
        return np.full(shape, np.nan, dtype=np.float64)

    date_pos = {int(day): i for i, day in enumerate(dates)}
    code_pos = {str(code): i for i, code in enumerate(codes)}
    year_lo, year_hi = int(dates[0]) // 10000, int(dates[-1]) // 10000
    rows = ctx.dataset(
        "stock_dragon_tiger",
        columns=["trade_date", "stock_code", "org_name", "buy_amount",
                 "sell_amount", "reason"],
        years=(year_lo, year_hi),
    )

    daily_hhi = np.full(shape, np.nan, dtype=np.float64)
    any_event = np.zeros(shape, dtype=bool)
    source_complete = np.zeros(len(dates), dtype=bool)
    traded = np.asarray(ctx.px_raw("vol"), dtype=np.float64)
    traded = np.isfinite(traded) & (traded > 0)

    if rows is not None and not rows.empty:
        source_date = ctx.date_col(rows["trade_date"])
        in_panel = (source_date >= dates[0]) & (source_date <= dates[-1])

        # Coverage is determined from all market rows before restricting to the
        # frozen stock axis or dropping multi-day cumulative reasons.
        for day in np.unique(source_date[in_panel]):
            ti = date_pos.get(int(day))
            if ti is not None:
                source_complete[ti] = True

        code_i = rows["stock_code"].astype(str).map(code_pos).fillna(-1).to_numpy(np.int32)
        keep = in_panel & (code_i >= 0)
        f = rows.loc[keep].copy()
        f["date_i"] = source_date[keep]
        f["code_i"] = code_i[keep]
        f = f[~f["reason"].fillna("").astype(str).str.contains(
            r"连续|期间|个交易日", regex=True
        )]
        f["buy"] = pd.to_numeric(f["buy_amount"], errors="coerce")
        f["sell"] = pd.to_numeric(f["sell_amount"], errors="coerce")

        # One row per stock/event date after collapsing only exact same-gross
        # mirrors/reason duplicates. Conflicting amounts are left as one group
        # and the HHI kernel returns NaN for that stock-day.
        for (ci, day), g in f.groupby(["code_i", "date_i"], sort=True):
            ti = date_pos.get(int(day))
            if ti is None:
                continue
            ci = int(ci)
            any_event[ti, ci] = True
            daily_hhi[ti, ci] = _reported_name_gross_hhi(
                g["org_name"].to_numpy(),
                g["buy"].to_numpy(dtype=np.float64),
                g["sell"].to_numpy(dtype=np.float64),
            )

    known = np.broadcast_to(source_complete[:, None], shape) & traded
    daily_load = np.full(shape, np.nan, dtype=np.float64)
    no_event = known & ~any_event
    daily_load[no_event] = 0.0
    valid_event = known & any_event & np.isfinite(daily_hhi)
    daily_load[valid_event] = daily_hhi[valid_event]

    # Source day d remains on exact slot d. The rolling endpoint exclusion is
    # the one-session availability lag: output T consumes source slots T-20:T.
    value = _fixed_trailing_mean(daily_load, window=20, min_valid=15)
    return np.where(np.isfinite(value), value, np.nan)


def build(parameters):
    """Registered-module builder; the frozen candidate has no parameters."""
    if parameters:
        raise ValueError(f"{NAME} takes no parameters")
    return _compute


def selfcheck():
    # A buy/sell mirror with equal gross collapses to one label-level amount.
    assert np.isclose(_reported_name_gross_hhi(
        ["A", "A", "B"], [5, 0, 5], [0, 5, 5]), 5 / 9)
    # Same label but different implied gross is ambiguous, not double-counted.
    assert np.isnan(_reported_name_gross_hhi(
        ["A", "A", "B"], [5, 9, 5], [0, 0, 5]))
    # One positive disclosed name is not a concentration cross-section.
    assert np.isnan(_reported_name_gross_hhi(["A", "B"], [10, 0], [0, 0]))
    daily = np.full((24, 1), np.nan)
    daily[:20] = 0.0
    daily[10] = 0.4
    out = _fixed_trailing_mean(daily)
    assert np.isnan(out[19, 0])
    assert np.isclose(out[20, 0], 0.4 / 20)


if __name__ == "__main__":
    selfcheck()
