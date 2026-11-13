"""R38 valuation-dynamics reserves: implied ROE change and cross-axis repair.

These reuse the frozen daily ``stock_finance`` panel helpers from the first
valuation module.  Own-history percentiles use a 252-slot window (minimum 180
valid values) and repair is the 60-slot change; cross-sectional ranks are
average-tie percentiles on each date's common finite set.
"""
from __future__ import annotations

import numpy as np

from factors.registered.r38_valuation_dynamics_v1 import (
    CHANGE, _cs_rank, _panels, _rolling_pct_last,
)

NAMES = (
    "vd_implied_roe_change_60",
    "vd_pe_ps_margin_revision_60",
    "vd_ps_pb_turnover_revision_60",
)


def _repairs(ctx):
    p = _panels(ctx)
    if p is None:
        return None
    reps = []
    for col in ("pe_ttm", "pb", "ps_ttm"):
        x = p[col]
        valid = np.isfinite(x) & (x > 0.0)
        pct = _rolling_pct_last(np.where(valid, x, np.nan))
        rep = pct - np.roll(pct, CHANGE, axis=0)
        rep[:CHANGE] = np.nan
        reps.append(rep)
    return reps


def _implied(ctx):
    p = _panels(ctx)
    if p is None:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    pe = p["pe_ttm"].astype(np.float64)
    pb = p["pb"].astype(np.float64)
    valid = np.isfinite(pe) & (pe > 0.0) & np.isfinite(pb) & (pb > 0.0)
    implied = np.where(valid, pb / np.where(valid, pe, 1.0), np.nan)
    rank = _cs_rank(implied, valid)
    out = rank - np.roll(rank, CHANGE, axis=0)
    out[:CHANGE] = np.nan
    return out


def _diff(ctx, first, second):
    p = _panels(ctx)
    if p is None:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    reps = _repairs(ctx)
    return reps[first] - reps[second]


def build(parameters):
    variant = (parameters or {}).get("variant")
    if variant not in NAMES:
        raise KeyError(f"unknown R38 valuation reserve variant: {variant!r}")

    def compute(ctx):
        if variant == "vd_implied_roe_change_60":
            return _implied(ctx)
        if variant == "vd_pe_ps_margin_revision_60":
            return _diff(ctx, 2, 0)  # PS repair minus PE repair
        return _diff(ctx, 2, 1)      # PS repair minus PB repair

    return compute
