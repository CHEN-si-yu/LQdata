"""R2-I5 preregistered share-supply / dilution mechanisms; research only.

Frozen before any return or IC was computed (Lead RESEARCH_BRIEF section 3, R2-I5):

  1 float_unlock_shock ....................... float_share/total_share jump, higher_is_better=false
  2 total_share_dilution ..................... split-adjusted net total_share growth, false
  3 free_float_ratio_level ................... free_share/total_share level / own-history rank, false
  4 share_supply_pressure_interaction ........ supply shock x illiquidity, false

Six candidate identities are registered from these four mechanisms (two mechanisms get one
additional pre-declared window/expression variant each); no other window, threshold or
direction is searched.

Data facts measured before writing this file (see round2/supply/recon_*.json):
  * stock_finance is a true daily grid (one row per stock and trading day, unique key, no
    duplicates, 2115/2115 pool codes, 2018-2026).  total_share is *not* a constant snapshot:
    over 2018-01..2026-09 the pool median is 6 distinct values per stock and only 5.1% of
    stocks never change it.  It behaves as event-day update + carry forward, which is what
    makes jump detection meaningful.
  * 2799 cells have total_share(T)/total_share(T-1) > 1.05.  1083 of them coincide with a
    cumulative adjustment factor ratio > 1.05 within 2% -- those are capitalisation changes
    (song-zhuan), not new supply.  The remaining ones are real supply events.
  * 5795 cells have float_share up > 1% with total_share unchanged: pure unlocks.

Adjustment rule (frozen here, not tuned on IC): a same-day share increase is treated as a
mechanical capitalisation change when adj_factor(T)/adj_factor(T-1) >= 1.05 and
log(share ratio) <= log(adj ratio) + 0.005.  The one-sided tolerance is required because a
cash dividend inflates the adjustment-factor jump above the pure share multiplier.  Such
days contribute exactly zero net supply growth; every other day contributes log(share ratio).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea import mathx as mx

SIGNALS = (
    "float_unlock_decay60",
    "float_unlock_decay120",
    "total_share_dilution252",
    "free_float_ratio_level",
    "free_float_ratio_rank252",
    "supply_pressure_interaction",
)

SPLIT_ADJ_MIN = 1.05     # an adj-factor jump at or above this can be a capitalisation change
SPLIT_LOG_TOL = 0.005    # one-sided log tolerance (cash dividend makes the adj jump larger)
SHARE_NOISE = 1e-5       # relative share move below this is supplier rounding, not an event

DILUTION_WINDOW = 252    # one trading year of cumulative net share growth
SHOCK_WINDOWS = {"float_unlock_decay60": 60, "float_unlock_decay120": 120, "interaction": 20}
RATIO_WINDOW = 252       # own-history window for the free-float ratio percentile


def _divide(a, b, floor=1e-12):
    with np.errstate(all="ignore"):
        out = np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64)
    return np.where(np.isfinite(out) & (np.abs(b) > floor), out, np.nan)


def _finance(ctx, fields):
    """Daily share state on the (T, C) panel, as-of carried forward.

    Share counts are *state* variables (Panel.asof semantics), so a missing supplier row
    means "last known count still applies", never zero.  Non-positive values are supplier
    artefacts: they are dropped from the as-of source and the last valid count carries
    forward.  The number of such cells is reported in the family evidence, not hidden here.
    """
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    years = (int(days[0]) // 10000, int(days[-1]) // 10000)
    columns = ["stock_code", "trade_date"] + list(fields)
    frame = ctx.dataset("stock_finance", columns=columns, years=years)
    if frame.empty:
        return {name: ctx.panel.empty(dtype=np.float64) for name in fields}
    frame = frame[frame.stock_code.isin(ctx.panel.codes)]
    if frame.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate stock/day in stock_finance")
    codes = frame.stock_code.to_numpy()
    day = ctx.date_col(frame.trade_date)
    out = {}
    for name in fields:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        values = np.where(np.isfinite(values) & (values > 0), values, np.nan)
        out[name] = np.asarray(ctx.asof_daily(codes, day, values), dtype=np.float64)
    return out


def _adj_factor(ctx):
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    years = (int(days[0]) // 10000, int(days[-1]) // 10000)
    frame = ctx.dataset("stock_adj_factor", columns=["stock_code", "trade_date", "adj_factor"],
                        years=years)
    if frame.empty:
        return ctx.panel.empty(dtype=np.float64)
    frame = frame[frame.stock_code.isin(ctx.panel.codes)]
    if frame.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate stock/day in stock_adj_factor")
    values = pd.to_numeric(frame["adj_factor"], errors="coerce").to_numpy(dtype=np.float64)
    values = np.where(np.isfinite(values) & (values > 0), values, np.nan)
    return np.asarray(ctx.asof_daily(frame.stock_code.to_numpy(),
                                     ctx.date_col(frame.trade_date), values), dtype=np.float64)


def _net_share_log_growth(total_share, adj_factor):
    """log of the share ratio after removing capitalisation (song-zhuan) jumps."""
    prev_share = mx.shift(total_share, 1)
    prev_adj = mx.shift(adj_factor, 1)
    share_ratio = _divide(total_share, prev_share, floor=0.0)
    adj_ratio = _divide(adj_factor, prev_adj, floor=0.0)
    known = np.isfinite(share_ratio) & np.isfinite(adj_ratio) & (share_ratio > 0)
    with np.errstate(all="ignore"):
        log_share = np.log(np.where(share_ratio > 0, share_ratio, np.nan))
        log_adj = np.log(np.where(adj_ratio > 0, adj_ratio, np.nan))
    capitalisation = (known & (adj_ratio >= SPLIT_ADJ_MIN) & (share_ratio >= 1.0)
                      & (log_share <= log_adj + SPLIT_LOG_TOL))
    growth = np.where(np.abs(share_ratio - 1.0) <= SHARE_NOISE, 0.0, log_share)
    growth = np.where(capitalisation, 0.0, growth)
    return np.where(known, growth, np.nan)


def _float_ratio(float_share, total_share):
    return _divide(float_share, total_share, floor=0.0)


def _unlock_shock(float_share, total_share, window):
    """Linear-decay weighted sum of positive float-ratio jumps over the window.

    A capitalisation change moves float_share and total_share together, so the ratio is
    unchanged and this measure is automatically immune to song-zhuan without using
    adj_factor.  A placement moves total_share more than float_share, so it contributes
    nothing here (it is mechanism 2 job).
    """
    ratio = _float_ratio(float_share, total_share)
    delta = ratio - mx.shift(ratio, 1)
    positive = np.where(np.isfinite(delta), np.maximum(delta, 0.0), np.nan)
    return mx.decay_linear(positive, window)


def build(parameters):
    """One of the six preregistered share-supply expressions; no searched windows."""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("R2-I5 permits only the six preregistered signal identifiers")
    signal = parameters["signal"]

    def compute(ctx):
        traded = np.asarray(ctx.traded(), dtype=bool)
        universe = np.asarray(ctx.universe, dtype=bool)
        if signal == "float_unlock_decay60":
            share = _finance(ctx, ("float_share", "total_share"))
            value = _unlock_shock(share["float_share"], share["total_share"], SHOCK_WINDOWS[signal])
        elif signal == "float_unlock_decay120":
            share = _finance(ctx, ("float_share", "total_share"))
            value = _unlock_shock(share["float_share"], share["total_share"], SHOCK_WINDOWS[signal])
        elif signal == "total_share_dilution252":
            share = _finance(ctx, ("total_share",))
            growth = _net_share_log_growth(share["total_share"], _adj_factor(ctx))
            value = mx.roll_sum(growth, DILUTION_WINDOW)
        elif signal == "free_float_ratio_level":
            share = _finance(ctx, ("free_share", "total_share"))
            value = _float_ratio(share["free_share"], share["total_share"])
        elif signal == "free_float_ratio_rank252":
            share = _finance(ctx, ("free_share", "total_share"))
            value = mx.roll_rank(_float_ratio(share["free_share"], share["total_share"]),
                                 RATIO_WINDOW)
        else:
            share = _finance(ctx, ("float_share", "total_share"))
            shock = _unlock_shock(share["float_share"], share["total_share"],
                                  SHOCK_WINDOWS["interaction"])
            volume = np.asarray(ctx.px_raw("vol"), dtype=np.float64)
            active = traded & np.isfinite(volume) & (volume > 0)
            turnover = np.where(active, _divide(volume, share["float_share"], floor=0.0), np.nan)
            average = mx.roll_mean(turnover, 20)
            usable = universe & np.isfinite(average) & np.isfinite(shock)
            z_shock = mx.cs_zscore(mx.safe_log(1.0 + shock), mask=usable, min_count=30)
            z_illiquid = mx.cs_zscore(-mx.safe_log(average), mask=usable, min_count=30)
            value = z_shock * z_illiquid
        value = np.where(np.isfinite(value), value, np.nan)
        return np.where(universe & traded & np.isfinite(value), value, np.nan).astype(
            np.float64, copy=False)

    return compute
