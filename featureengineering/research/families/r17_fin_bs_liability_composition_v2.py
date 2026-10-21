"""R17 seat_fin_bs batch 2: liability-side balance-sheet composition (GC descriptors).

One parameterized builder for 24 economically distinct liability-structure
descriptors. Every variant uses a single whole-row PIT balance-sheet pipeline
(the helper functions of r14f1_ccc_innovation_v2):

* comp_type == 1 and report_type == 1 before dedup/frontier;
* PIT = max(valid ann_date, valid f_ann_date, end_date);
* latest retained row per (stock_code, end_date, PIT), physical-last wins;
  a retained row with a missing amount clears older state;
* current report-period frontier per stock; next exchange session once;
* report age <= 550 natural days; fixed universe and actual traded mask.

Fifteen variants are single liability lines over total liabilities. Nine are
within declared liability pools: interest-bearing debt (three registered legs:
short-term borrowings, bonds, current portion), trade credit (one leg),
customer advances (one leg), payroll/tax accruals (one leg), long-term non-debt
obligations (two legs) and long-term debt instruments (one leg). No registered
pair sums to one except where both complementary legs are explicitly registered
(none here: only one leg is taken from every two-leg pool).

Genuine zeros are kept (no bond issuance, no notes payable, no estimated
liability are real zeros); missing/undisclosed is NaN. No missing-to-zero, no
stale-report fallback, no clipping. Every denominator exceeds 1e6 CNY.
Pool-share variants require every member leg finite and non-negative.
``bs_point(field)`` denotes this whole-row PIT value, not ``ctx.point``.
"""
from __future__ import annotations

import numpy as np

from research.families import r14f1_ccc_innovation_v2 as pit

WARMUP_DAYS = 1800
VERSION = 1
AMOUNT_FLOOR = 1_000_000.0

IBD = ("st_borr", "non_cur_liab_due_1y", "lt_borr", "bond_payable")
LT_OBLIG = ("long_pay_total", "defer_inc_non_cur_liab", "defer_tax_liab", "estimated_liab")
LT_DEBT = ("lt_borr", "bond_payable", "long_pay_total")

SPECS = {
    "liab_st_borr_to_liab": {
        "num_add": ("st_borr",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_notes_payable_to_liab": {
        "num_add": ("notes_payable",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_acct_payable_to_liab": {
        "num_add": ("acct_payable",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_adv_receipts_to_liab": {
        "num_add": ("adv_receipts",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_contract_liab_to_liab": {
        "num_add": ("contract_liab",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_payroll_to_liab": {
        "num_add": ("payroll_payable",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_taxes_to_liab": {
        "num_add": ("taxes_payable",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_oth_pay_to_liab": {
        "num_add": ("oth_pay_total",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_noncur_due1y_to_liab": {
        "num_add": ("non_cur_liab_due_1y",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_lt_borr_to_liab": {
        "num_add": ("lt_borr",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_bond_to_liab": {
        "num_add": ("bond_payable",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_long_pay_to_liab": {
        "num_add": ("long_pay_total",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_defer_income_to_liab": {
        "num_add": ("defer_inc_non_cur_liab",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_defer_tax_to_liab": {
        "num_add": ("defer_tax_liab",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_estimated_to_liab": {
        "num_add": ("estimated_liab",), "num_sub": (), "den": ("total_liab",), "pool": False,
    },
    "liab_st_borr_share_ibd": {
        "num_add": ("st_borr",), "num_sub": (), "den": IBD, "pool": True,
    },
    "liab_bond_share_ibd": {
        "num_add": ("bond_payable",), "num_sub": (), "den": IBD, "pool": True,
    },
    "liab_current_portion_share_ibd": {
        "num_add": ("non_cur_liab_due_1y",), "num_sub": (), "den": IBD, "pool": True,
    },
    "liab_acct_payable_share_trade": {
        "num_add": ("acct_payable",), "num_sub": (),
        "den": ("notes_payable", "acct_payable"), "pool": True,
    },
    "liab_contract_liab_share_customer_adv": {
        "num_add": ("contract_liab",), "num_sub": (),
        "den": ("adv_receipts", "contract_liab"), "pool": True,
    },
    "liab_payroll_share_accrual": {
        "num_add": ("payroll_payable",), "num_sub": (),
        "den": ("payroll_payable", "taxes_payable"), "pool": True,
    },
    "liab_defer_tax_share_lt_oblig": {
        "num_add": ("defer_tax_liab",), "num_sub": (), "den": LT_OBLIG, "pool": True,
    },
    "liab_estimated_share_lt_oblig": {
        "num_add": ("estimated_liab",), "num_sub": (), "den": LT_OBLIG, "pool": True,
    },
    "liab_long_pay_share_lt_debt": {
        "num_add": ("long_pay_total",), "num_sub": (), "den": LT_DEBT, "pool": True,
    },
}

FAMILY_FIELDS = tuple(sorted({
    field
    for spec in SPECS.values()
    for field in (*spec["num_add"], *spec["num_sub"], *spec["den"])
}))
_TABLE_CACHE: dict[int, object] = {}


def _family_table(ctx):
    key = (id(ctx.up), int(ctx.panel.dates[0]), int(ctx.panel.dates[-1]), int(ctx.panel.C))
    table = _TABLE_CACHE.get(key)
    if table is None:
        table = pit._report_table(ctx, "stock_balancesheet", FAMILY_FIELDS)
        _TABLE_CACHE[key] = table
        if len(_TABLE_CACHE) > 2:
            for old in list(_TABLE_CACHE)[:-1]:
                _TABLE_CACHE.pop(old, None)
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
    if spec["pool"]:
        for field in fields:
            leg = pit._lookup(versions, code, quarter, cap, field)
            ok &= np.isfinite(leg) & (leg >= 0.0)
    ok &= den > AMOUNT_FLOOR
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(ok, num / den, np.nan)
    return events, values


def build(parameters):
    params = dict(parameters)
    variant = params.get("variant")
    if variant not in SPECS or params.get("version") != VERSION:
        raise ValueError("Unknown frozen liability-composition variant")

    def compute(ctx):
        data = _family_table(ctx)
        events, values = variant_values(data, variant)
        return pit._map_fresh(ctx, events, values)

    return compute
