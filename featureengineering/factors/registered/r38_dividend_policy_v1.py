"""R38 dividend-policy factors: capital-service split, DPS growth, large cut event.

Annual first-vintage financial rows are joined on the same fiscal-year PIT and
mapped to the first session strictly after the PIT.  The dividend proxy is
cash distributions minus interest expense; all three variants use that proxy
and a 550-calendar-day report-age cap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors.registered.r38_quarterly_w2_v1 import (
    Table, _joint_events, _load, _lookup, _map,
)
from fea.deriv import KEY_W, PERIOD_W

NAMES = (
    "dv_interest_split_fy_v1",
    "dv_dps_growth_fy_v1",
    "dv_cut_event_fy_v1",
)
MIN = 1_000_000.0


def _annual(table: Table) -> Table:
    m = (table.period % 4) == 3
    if not m.any():
        empty = np.empty(0, dtype=np.int64)
        return Table(table.dataset, table.fields, empty, empty, empty,
                     {f: np.empty(0, dtype=np.float64) for f in table.fields}, empty)
    code = table.code[m]
    period = table.period[m]
    pit = table.pit[m]
    keys = (code * PERIOD_W + period) * KEY_W + pit
    values = {f: table.values[f][m] for f in table.fields}
    return Table(table.dataset, table.fields, code, period, pit, values, keys)


def _annual_load(ctx, dataset, fields):
    return _annual(_load(ctx, dataset, tuple(fields)))


def _legs(ctx, with_bs):
    inc = _annual_load(ctx, "stock_income", ("fin_exp_int_exp", "n_income_attr_p"))
    cf = _annual_load(ctx, "stock_cashflow", ("c_pay_dist_dpcp_int_exp",))
    bs = _annual_load(ctx, "stock_balancesheet", ("total_share",)) if with_bs else None
    if len(inc.code) == 0 or len(cf.code) == 0 or (with_bs and len(bs.code) == 0):
        return None
    ev = _joint_events([inc, cf] + ([bs] if with_bs else []))
    if ev.empty:
        return None
    code = ev.code.to_numpy(np.int64)
    q = ev.q.to_numpy(np.int64)
    pit = ev.pit.to_numpy(np.int64)

    def lag(table, field, n):
        return _lookup(table, code, q - 4 * n, pit, field)

    dist = lag(cf, "c_pay_dist_dpcp_int_exp", 0)
    interest = lag(inc, "fin_exp_int_exp", 0)
    ni = lag(inc, "n_income_attr_p", 0)
    if with_bs:
        share0 = lag(bs, "total_share", 0)
        share1 = lag(bs, "total_share", 1)
    else:
        share0 = np.full(len(code), np.nan, dtype=np.float64)
        share1 = np.full(len(code), np.nan, dtype=np.float64)
    dist1 = lag(cf, "c_pay_dist_dpcp_int_exp", 1)
    interest1 = lag(inc, "fin_exp_int_exp", 1)
    d0 = dist - interest
    d1 = dist1 - interest1
    return ev, d0, d1, interest, ni, share0, share1


def _interest_split(ctx):
    legs = _legs(ctx, False)
    if legs is None:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    ev, d0, d1, interest, ni, share0, share1 = legs
    den = d0 + interest
    ok = (np.isfinite(d0) & (d0 >= 0.0) & np.isfinite(interest)
          & (interest > 0.0) & np.isfinite(den) & (den > MIN))
    val = np.where(ok, d0 / np.where(ok, den, 1.0), np.nan)
    return _map(ctx, ev, val)


def _dps_growth(ctx):
    legs = _legs(ctx, True)
    if legs is None:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    ev, d0, d1, interest, ni, share0, share1 = legs
    dps0 = np.where(np.isfinite(d0) & np.isfinite(share0) & (share0 > 0),
                    d0 / np.where(share0 > 0, share0, 1.0), np.nan)
    dps1 = np.where(np.isfinite(d1) & np.isfinite(share1) & (share1 > 0),
                    d1 / np.where(share1 > 0, share1, 1.0), np.nan)
    ok = (np.isfinite(dps0) & np.isfinite(dps1) & (dps0 > 0.0) & (dps1 > 0.0)
          & np.isfinite(d0) & (d0 >= 0.0) & np.isfinite(d1) & (d1 >= 0.0))
    val = np.where(ok, np.log(np.where(ok, dps0, 1.0) / np.where(ok, dps1, 1.0)), np.nan)
    return _map(ctx, ev, val)


def _cut_event(ctx):
    legs = _legs(ctx, True)
    if legs is None:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    ev, d0, d1, interest, ni, share0, share1 = legs
    dps0 = np.where(np.isfinite(d0) & np.isfinite(share0) & (share0 > 0),
                    d0 / np.where(share0 > 0, share0, 1.0), np.nan)
    dps1 = np.where(np.isfinite(d1) & np.isfinite(share1) & (share1 > 0),
                    d1 / np.where(share1 > 0, share1, 1.0), np.nan)
    ok = (np.isfinite(dps0) & np.isfinite(dps1) & (dps1 > 0.0) & (dps0 >= 0.0)
          & np.isfinite(d0) & (d0 >= 0.0) & np.isfinite(d1) & (d1 >= 0.0)
          & np.isfinite(ni) & (ni > MIN))
    event = np.where(ok, (dps0 < 0.5 * dps1).astype(np.float64), np.nan)
    return _map(ctx, ev, event)


def build(parameters):
    variant = (parameters or {}).get("variant")
    if variant not in NAMES:
        raise KeyError(f"unknown R38 dividend variant: {variant!r}")

    def compute(ctx):
        if variant == "dv_interest_split_fy_v1":
            return _interest_split(ctx)
        if variant == "dv_dps_growth_fy_v1":
            return _dps_growth(ctx)
        return _cut_event(ctx)

    return compute
