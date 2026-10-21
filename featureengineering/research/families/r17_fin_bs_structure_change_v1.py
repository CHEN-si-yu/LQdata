"""R17 seat_fin_bs batch 4: four-quarter changes in balance-sheet structure.

One parameterized builder for 18 GC descriptors. Each variant is the year-over-
year change of a composition share:

    share(q) - share(q - 4)

where q is the current report-period frontier for the stock and both shares
are evaluated from the versions visible at the *current* report PIT. The prior
year's value is therefore the as-known-then (PIT-capped) value, never a future
revision; no historical value is rewritten by later restatements.

The whole-row PIT pipeline is the helper set of r14f1_ccc_innovation_v2:

* comp_type == 1 and report_type == 1 before dedup/frontier;
* PIT = max(valid ann_date, valid f_ann_date, end_date);
* latest retained row per (stock_code, end_date, PIT), physical-last wins;
  a retained row with a missing amount clears older state;
* current report-period frontier per stock; next exchange session once;
* report age <= 550 natural days; fixed universe and actual traded mask.

Both the current and prior-year denominators must exceed 1e6 CNY and every
component must be finite, so no partial change is produced. Genuine zeros are
kept; signed source values (losses, net cash, working capital) are retained.
No missing-to-zero, no stale-report fallback, no clipping.
``bs_point(field)`` denotes this whole-row PIT value, not ``ctx.point``.
"""
from __future__ import annotations

import numpy as np

from research.families import r14f1_ccc_innovation_v2 as pit

WARMUP_DAYS = 1800
VERSION = 1
AMOUNT_FLOOR = 1_000_000.0
LAG_QUARTERS = 4

IBD = ("st_borr", "non_cur_liab_due_1y", "lt_borr", "bond_payable")

SPECS = {
    "chg_inventory_share_yoy": {
        "num_add": ("inventories",), "num_sub": (), "den": ("total_assets",),
    },
    "chg_receivables_share_yoy": {
        "num_add": ("accounts_receiv",), "num_sub": (), "den": ("total_assets",),
    },
    "chg_contract_assets_share_yoy": {
        "num_add": ("contract_assets",), "num_sub": (), "den": ("total_assets",),
    },
    "chg_cip_share_yoy": {
        "num_add": ("cip",), "num_sub": (), "den": ("total_assets",),
    },
    "chg_goodwill_share_yoy": {
        "num_add": ("goodwill",), "num_sub": (), "den": ("total_assets",),
    },
    "chg_current_assets_share_yoy": {
        "num_add": ("total_cur_assets",), "num_sub": (), "den": ("total_assets",),
    },
    "chg_trad_asset_share_yoy": {
        "num_add": ("trad_asset",), "num_sub": (), "den": ("total_assets",),
    },
    "chg_fixed_asset_share_yoy": {
        "num_add": ("fix_assets",), "num_sub": (), "den": ("total_assets",),
    },
    "chg_st_borr_share_yoy": {
        "num_add": ("st_borr",), "num_sub": (), "den": ("total_liab",),
    },
    "chg_contract_liab_share_yoy": {
        "num_add": ("contract_liab",), "num_sub": (), "den": ("total_liab",),
    },
    "chg_acct_payable_share_yoy": {
        "num_add": ("acct_payable",), "num_sub": (), "den": ("total_liab",),
    },
    "chg_ibd_share_assets_yoy": {
        "num_add": IBD, "num_sub": (), "den": ("total_assets",),
    },
    "chg_current_liab_share_yoy": {
        "num_add": ("total_cur_liab",), "num_sub": (), "den": ("total_liab",),
    },
    "chg_net_debt_equity_yoy": {
        "num_add": IBD, "num_sub": ("money_cap",),
        "den": ("total_hldr_eqy_inc_min_int",),
    },
    "chg_working_capital_assets_yoy": {
        "num_add": ("total_cur_assets",), "num_sub": ("total_cur_liab",),
        "den": ("total_assets",),
    },
    "chg_retained_earnings_assets_yoy": {
        "num_add": ("undistr_porfit",), "num_sub": (), "den": ("total_assets",),
    },
    "chg_cash_to_ibd_yoy": {
        "num_add": ("money_cap",), "num_sub": (), "den": IBD,
    },
    "chg_minority_share_yoy": {
        "num_add": ("minority_int",), "num_sub": (),
        "den": ("total_hldr_eqy_inc_min_int",),
    },
}

FAMILY_FIELDS = tuple(sorted({
    field
    for spec in SPECS.values()
    for field in (*spec["num_add"], *spec["num_sub"], *spec["den"])
}))
_TABLE_CACHE: dict[int, object] = {}


def _family_table(ctx):
    key = id(ctx)
    table = _TABLE_CACHE.get(key)
    if table is None:
        table = pit._report_table(ctx, "stock_balancesheet", FAMILY_FIELDS)
        _TABLE_CACHE[key] = table
    return table


def variant_values(data, variant):
    if variant not in SPECS:
        raise KeyError(variant)
    spec = SPECS[variant]
    fields = tuple(sorted(set((*spec["num_add"], *spec["num_sub"], *spec["den"]))))
    events = pit._frontier_events(data)
    versions = pit._sorted_versions(data, fields)
    code = events.code.to_numpy(dtype=np.int64)
    quarter = events.q.to_numpy(dtype=np.int64)
    cap = events.pit.to_numpy(dtype=np.int64)

    def summed_at(names, period):
        total = np.zeros(len(code), dtype=np.float64)
        ok = np.ones(len(code), dtype=bool)
        for field in names:
            value = pit._lookup(versions, code, period, cap, field)
            ok &= np.isfinite(value)
            total += np.where(np.isfinite(value), value, 0.0)
        return total, ok

    def share_at(period):
        num, ok = summed_at(spec["num_add"], period)
        if spec["num_sub"]:
            subtract, sub_ok = summed_at(spec["num_sub"], period)
            ok &= sub_ok
            num = num - subtract
        den, den_ok = summed_at(spec["den"], period)
        ok &= den_ok
        ok &= den > AMOUNT_FLOOR
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ok, num / den, np.nan), ok

    current, current_ok = share_at(quarter)
    prior, prior_ok = share_at(quarter - LAG_QUARTERS)
    ok = current_ok & prior_ok
    values = np.where(ok, current - prior, np.nan)
    return events, values


def build(parameters):
    params = dict(parameters)
    variant = params.get("variant")
    if variant not in SPECS or params.get("version") != VERSION:
        raise ValueError("Unknown frozen structure-change variant")

    def compute(ctx):
        data = _family_table(ctx)
        events, values = variant_values(data, variant)
        return pit._map_fresh(ctx, events, values)

    return compute
