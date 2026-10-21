"""R17 seat_fin_bs batch 3: equity composition and leverage/liquidity structure.

One parameterized builder for 18 GC descriptors. Every variant uses a single
whole-row PIT balance-sheet pipeline (the helper functions of
r14f1_ccc_innovation_v2):

* comp_type == 1 and report_type == 1 before dedup/frontier;
* PIT = max(valid ann_date, valid f_ann_date, end_date);
* latest retained row per (stock_code, end_date, PIT), physical-last wins;
  a retained row with a missing amount clears older state;
* current report-period frontier per stock; next exchange session once;
* report age <= 550 natural days; fixed universe and actual traded mask.

Six variants decompose total equity into capital reserve, surplus reserve,
undistributed profit, minority interest, treasury shares and other
comprehensive income.  Twelve variants measure interest-bearing debt, net
debt, current-liability structure, working capital, retained earnings, cash
coverage, tangible collateral and long-term asset coverage.

Missing/undisclosed is NaN; genuine zeros are kept; signed source values
(accumulated losses, negative OCI/minority, treasury shares, net debt, working
capital) are retained without clipping. No missing-to-zero, no stale-report
fallback. Every denominator exceeds 1e6 CNY. ``bs_point(field)`` denotes this
whole-row PIT value, not ``ctx.point``.

Note: the retained balance-sheet schema has no CNY paid-in-capital field
(``total_share`` is a share count, not par value), so paid-in capital is not
used in an equity share; capital reserve is the first equity component.
"""
from __future__ import annotations

import numpy as np

from research.families import r14f1_ccc_innovation_v2 as pit

WARMUP_DAYS = 1800
VERSION = 1
AMOUNT_FLOOR = 1_000_000.0

IBD = ("st_borr", "non_cur_liab_due_1y", "lt_borr", "bond_payable")

SPECS = {
    "eq_capital_reserve_share": {
        "num_add": ("cap_rese",), "num_sub": (), "den": ("total_hldr_eqy_inc_min_int",),
    },
    "eq_surplus_reserve_share": {
        "num_add": ("surplus_rese",), "num_sub": (), "den": ("total_hldr_eqy_inc_min_int",),
    },
    "eq_undistributed_profit_share": {
        "num_add": ("undistr_porfit",), "num_sub": (), "den": ("total_hldr_eqy_inc_min_int",),
    },
    "eq_minority_share": {
        "num_add": ("minority_int",), "num_sub": (), "den": ("total_hldr_eqy_inc_min_int",),
    },
    "eq_treasury_share": {
        "num_add": ("treasury_share",), "num_sub": (), "den": ("total_hldr_eqy_inc_min_int",),
    },
    "eq_oci_share": {
        "num_add": ("oth_comp_income",), "num_sub": (), "den": ("total_hldr_eqy_inc_min_int",),
    },
    "lev_ibd_to_assets": {
        "num_add": IBD, "num_sub": (), "den": ("total_assets",),
    },
    "lev_net_debt_to_equity": {
        "num_add": IBD, "num_sub": ("money_cap",), "den": ("total_hldr_eqy_inc_min_int",),
    },
    "lev_net_debt_to_assets": {
        "num_add": IBD, "num_sub": ("money_cap",), "den": ("total_assets",),
    },
    "lev_current_liab_share": {
        "num_add": ("total_cur_liab",), "num_sub": (), "den": ("total_liab",),
    },
    "lev_current_liab_to_assets": {
        "num_add": ("total_cur_liab",), "num_sub": (), "den": ("total_assets",),
    },
    "lev_noncurrent_liab_to_assets": {
        "num_add": ("total_ncl",), "num_sub": (), "den": ("total_assets",),
    },
    "lev_working_capital_to_assets": {
        "num_add": ("total_cur_assets",), "num_sub": ("total_cur_liab",), "den": ("total_assets",),
    },
    "lev_retained_earnings_to_assets": {
        "num_add": ("undistr_porfit",), "num_sub": (), "den": ("total_assets",),
    },
    "lev_cash_to_ibd": {
        "num_add": ("money_cap",), "num_sub": (), "den": IBD,
    },
    "lev_cash_to_liab": {
        "num_add": ("money_cap",), "num_sub": (), "den": ("total_liab",),
    },
    "lev_tangible_collateral_share": {
        "num_add": ("fix_assets", "cip", "inventories", "accounts_receiv"),
        "num_sub": (), "den": ("total_assets",),
    },
    "lev_ibd_share_liab": {
        "num_add": IBD, "num_sub": (), "den": ("total_liab",),
    },
    "lev_fixed_asset_coverage_equity": {
        "num_add": ("fix_assets",), "num_sub": (), "den": ("total_hldr_eqy_exc_min_int",),
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

    def summed(names):
        total = np.zeros(len(code), dtype=np.float64)
        ok = np.ones(len(code), dtype=bool)
        for field in names:
            value = pit._lookup(versions, code, quarter, cap, field)
            ok &= np.isfinite(value)
            total += np.where(np.isfinite(value), value, 0.0)
        return total, ok

    num, ok = summed(spec["num_add"])
    if spec["num_sub"]:
        subtract, sub_ok = summed(spec["num_sub"])
        ok &= sub_ok
        num = num - subtract
    den, den_ok = summed(spec["den"])
    ok &= den_ok
    ok &= den > AMOUNT_FLOOR
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(ok, num / den, np.nan)
    return events, values


def build(parameters):
    params = dict(parameters)
    variant = params.get("variant")
    if variant not in SPECS or params.get("version") != VERSION:
        raise ValueError("Unknown frozen equity/leverage variant")

    def compute(ctx):
        data = _family_table(ctx)
        events, values = variant_values(data, variant)
        return pit._map_fresh(ctx, events, values)

    return compute
