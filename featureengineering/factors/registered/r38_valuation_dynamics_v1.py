"""R38 valuation-dynamics factors: PB-ROE re-rating and multi-axis repair.

All inputs are daily post-close ``stock_finance`` snapshots.  Own-history
percentiles use a 252-slot window (minimum 180 valid values); repair is the
60-slot change.  Cross-sectional ranks are average-tie percentiles computed on
each date's common finite set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = (
    "vd_pb_roe_resid_change_60",
    "vd_pct_repair_consensus_60",
)
WINDOW = 252
MIN_VALID = 180
CHANGE = 60


def _place(ctx, frame, columns):
    idx = np.asarray(ctx.code_index(frame["stock_code"].to_numpy()), dtype=np.int64)
    day = np.asarray(ctx.date_col(frame["trade_date"]), dtype=np.int64)
    keep = idx >= 0
    out = {}
    for c in columns:
        vals = pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
        out[c] = ctx.panel.place(idx[keep], day[keep], vals[keep]).astype(np.float32)
    return out


def _rolling_pct_last(x):
    """Percentile of the last value in a trailing 252-slot window (card formula)."""
    df = pd.DataFrame(np.where(np.isfinite(x), x, np.nan))
    roll = df.rolling(WINDOW, min_periods=MIN_VALID)
    rank = roll.rank(pct=True, method="average")
    count = roll.count()
    return (rank - 0.5 / count.where(count > 0)).to_numpy(np.float64)


def _cs_rank(x, common):
    df = pd.DataFrame(np.where(common, x, np.nan))
    return df.rank(axis=1, pct=True, method="average").to_numpy(np.float64)


def _panels(ctx):
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    frame = ctx.dataset(
        "stock_finance",
        columns=["trade_date", "stock_code", "pe_ttm", "pb", "ps_ttm"],
        years=(int(dates[0]) // 10000, int(dates[-1]) // 10000),
    )
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    return _place(ctx, frame, ["pe_ttm", "pb", "ps_ttm"])


def _pb_roe(ctx):
    p = _panels(ctx)
    if p is None:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    pe = p["pe_ttm"].astype(np.float64)
    pb = p["pb"].astype(np.float64)
    valid = np.isfinite(pe) & (pe > 0.0) & np.isfinite(pb) & (pb > 0.0)
    implied = np.where(valid, pb / np.where(valid, pe, 1.0), np.nan)
    rank_pb = _cs_rank(pb, valid)
    rank_roe = _cs_rank(implied, valid)
    resid = rank_pb - rank_roe
    out = resid - np.roll(resid, CHANGE, axis=0)
    out[:CHANGE] = np.nan
    return out


def _consensus(ctx):
    p = _panels(ctx)
    if p is None:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    reps = []
    for col, positive in (("pe_ttm", True), ("pb", True), ("ps_ttm", True)):
        x = p[col]
        valid = np.isfinite(x) & ((x > 0.0) if positive else (x >= 0.0))
        pct = _rolling_pct_last(np.where(valid, x, np.nan))
        rep = pct - np.roll(pct, CHANGE, axis=0)
        rep[:CHANGE] = np.nan
        reps.append(rep)
    stack = np.stack(reps, axis=0)
    cnt = np.isfinite(stack).sum(axis=0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(stack, axis=0)
    return np.where(cnt >= 2, mean, np.nan)


def build(parameters):
    variant = (parameters or {}).get("variant")
    if variant not in NAMES:
        raise KeyError(f"unknown R38 valuation variant: {variant!r}")

    def compute(ctx):
        if variant == "vd_pb_roe_resid_change_60":
            return _pb_roe(ctx)
        return _consensus(ctx)

    return compute
