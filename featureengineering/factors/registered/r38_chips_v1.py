"""R38 chip-distribution and cost-asymmetry factors.

``stock_cyq_chips`` gives a daily price/percent histogram per stock.  Log
distance from the raw close is aggregated into overhang, local-congestion and
near/far supply shares.  ``stock_cyq_perf`` provides the 15/50/85 cost
percentiles for the upper-tail asymmetry.  All values use the same day's
post-close snapshot and are masked to real traded stock-days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = (
    "cs_overhang_pressure_v1",
    "cs_overhang_near_far_v1",
    "cs_local_congestion_v1",
    "cp_cost_asymmetry_v1",
)
LOG_102 = float(np.log(1.02))
LOG_110 = float(np.log(1.10))
LOG_130 = float(np.log(1.30))

def _traded(ctx):
    close = np.asarray(ctx.px_raw("close"), dtype=np.float64)
    vol = np.asarray(ctx.px_raw("vol"), dtype=np.float64)
    return close, np.isfinite(close) & np.isfinite(vol) & (close > 0) & (vol > 0)


def _chip_hist(ctx):
    from .chip_distribution_cache_v1 import field, LEGACY_FIELDS
    arrays = [field(ctx, name) for name in LEGACY_FIELDS]
    valid = np.isfinite(arrays[0])
    if not valid.any():
        return None
    ti, ci = np.nonzero(valid)
    return (ti, ci, *(x[ti, ci] for x in arrays))


def _chip_value(ctx, kind):
    close, traded = _traded(ctx)
    hist = _chip_hist(ctx)
    if hist is None:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    ti, ci, W, over, above, near, far, local = hist
    S, C = ctx.panel.shape
    out = np.full((S, C), np.nan, dtype=np.float64)
    if kind == "cs_overhang_pressure_v1":
        vals = np.where(W > 0, over / np.where(W > 0, W, np.nan), np.nan)
    elif kind == "cs_local_congestion_v1":
        vals = np.where(W > 0, local / np.where(W > 0, W, np.nan), np.nan)
    else:
        with np.errstate(all="ignore"):
            ratio = np.where(W > 0, above / np.where(W > 0, W, np.nan), np.nan)
            vals = np.where((W > 0) & (ratio >= 0.01),
                            (near - far) / np.where(above > 0, above, np.nan), np.nan)
        vals = np.where((W > 0) & (ratio < 0.01), 0.0, vals)
    out[ti, ci] = vals
    out[~traded] = np.nan
    return out


def _cost_asymmetry(ctx):
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    S, C = ctx.panel.shape
    lo = int(dates[0]) // 10000
    hi = int(dates[-1]) // 10000
    frame = ctx.dataset(
        "stock_cyq_perf",
        columns=["trade_date", "stock_code", "cost_15pct", "cost_50pct", "cost_85pct"],
        years=(lo, hi),
    )
    if frame is None or frame.empty:
        return np.full((S, C), np.nan, dtype=np.float64)
    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    ti = np.searchsorted(dates, ctx.date_col(frame["trade_date"]), side="left")
    ci = np.asarray(ctx.code_index(frame["stock_code"].to_numpy()), dtype=np.int64)
    ok = (ci >= 0) & (ti < S)
    src = ctx.date_col(frame["trade_date"])[ok]
    ti = ti[ok]
    exact = dates[np.clip(ti, 0, S - 1)] == src
    ti, ci = ti[exact], ci[ok][exact]
    c15 = pd.to_numeric(frame["cost_15pct"], errors="coerce").to_numpy(np.float64)[ok][exact]
    c50 = pd.to_numeric(frame["cost_50pct"], errors="coerce").to_numpy(np.float64)[ok][exact]
    c85 = pd.to_numeric(frame["cost_85pct"], errors="coerce").to_numpy(np.float64)[ok][exact]
    _, traded = _traded(ctx)
    out = np.full((S, C), np.nan, dtype=np.float64)
    with np.errstate(all="ignore"):
        d = c50 - c15
        vals = np.where(d > 1e-6, (c85 - c50) / np.where(d > 1e-6, d, np.nan), np.nan)
    out[ti, ci] = vals
    out[~traded] = np.nan
    return out


def build(parameters):
    variant = (parameters or {}).get("variant")
    if variant not in NAMES:
        raise KeyError(f"unknown R38 chips variant: {variant!r}")

    def compute(ctx):
        if variant == "cp_cost_asymmetry_v1":
            return _cost_asymmetry(ctx)
        return _chip_value(ctx, variant)

    return compute
