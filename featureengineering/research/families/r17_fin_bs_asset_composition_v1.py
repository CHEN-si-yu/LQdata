"""R17 seat_fin_bs batch 1: asset-side balance-sheet composition (GC descriptors).

One parameterized builder for 24 economically distinct asset-structure
descriptors. Every variant uses a single whole-row PIT balance-sheet pipeline
(the helper functions of r14f1_ccc_innovation_v2):

* keep comp_type == 1 and report_type == 1 before dedup/frontier;
* PIT = max(valid ann_date, valid f_ann_date, end_date);
* latest retained row per (stock_code, end_date, PIT), physical-last wins;
  a retained row with a missing amount clears older state (no stale fallback);
* current report-period frontier per stock (prefix max), so a late correction
  of an old period cannot displace a newer report;
* map PIT to the next exchange session exactly once, then as-of forward fill;
* report age <= 550 natural days, fixed universe and actual traded mask.

Numerator and denominator are always read from the same retained report row.
Genuine zeros are kept; missing/undisclosed is NaN; no missing-to-zero, no
partial filling, no clipping. Every denominator must exceed 1e6 CNY. Pool-share
variants require every member leg finite and non-negative; single-item shares
keep the finite source sign (an anomalous negative other-asset balance stays
negative and is disclosed, never clipped). Signed net-trade-working-capital
variant is explicitly signed.

``bs_point(field)`` in the formula strings means the whole-row PIT point value
implemented here, not ``ctx.point`` (which does not apply the comp/report row
filter or the 550-day report-age cap).
"""
from __future__ import annotations

import numpy as np

from research.families import r14f1_ccc_innovation_v2 as pit

WARMUP_DAYS = 1800
VERSION = 1
AMOUNT_FLOOR = 1_000_000.0

# variant -> {num_add, num_sub, den, pool}
# ``pool`` marks a share whose denominator is the sum of non-negative member
# legs printed here; every leg must be finite and >= 0. Only one leg per pool
# is registered, so no registered pair sums to one.
SPECS = {
    "asset_trad_asset_share": {
        "num_add": ("trad_asset",), "num_sub": (), "den": ("total_assets",), "pool": False,
    },
    "asset_notes_receiv_share": {
        "num_add": ("notes_receiv",), "num_sub": (), "den": ("total_assets",), "pool": False,
    },
    "asset_contract_assets_share": {
        "num_add": ("contract_assets",), "num_sub": (), "den": ("total_assets",), "pool": False,
    },
    "asset_oth_cur_share": {
        "num_add": ("oth_cur_assets",), "num_sub": (), "den": ("total_assets",), "pool": False,
    },
    "asset_oth_nca_share": {
        "num_add": ("oth_nca",), "num_sub": (), "den": ("total_assets",), "pool": False,
    },
    "asset_cip_share": {
        "num_add": ("cip",), "num_sub": (), "den": ("total_assets",), "pool": False,
    },
    "asset_goodwill_share": {
        "num_add": ("goodwill",), "num_sub": (), "den": ("total_assets",), "pool": False,
    },
    "asset_intan_share": {
        "num_add": ("intan_assets",), "num_sub": (), "den": ("total_assets",), "pool": False,
    },
    "asset_invest_property_share": {
        "num_add": ("invest_real_estate",), "num_sub": (), "den": ("total_assets",), "pool": False,
    },
    "asset_rnd_capital_share": {
        "num_add": ("r_and_d",), "num_sub": (), "den": ("total_assets",), "pool": False,
    },
    "asset_debt_invest_share": {
        "num_add": ("debt_invest", "oth_debt_invest"), "num_sub": (),
        "den": ("total_assets",), "pool": False,
    },
    "asset_nca_within1y_share": {
        "num_add": ("nca_within_1y",), "num_sub": (), "den": ("total_assets",), "pool": False,
    },
    "asset_current_share": {
        "num_add": ("total_cur_assets",), "num_sub": (), "den": ("total_assets",), "pool": False,
    },
    "asset_inventory_share": {
        "num_add": ("inventories",), "num_sub": (), "den": ("total_assets",), "pool": False,
    },
    "asset_notes_to_receivables": {
        "num_add": ("notes_receiv",), "num_sub": (),
        "den": ("notes_receiv", "accounts_receiv"), "pool": True,
    },
    "asset_cip_to_fixed": {
        "num_add": ("cip",), "num_sub": (),
        "den": ("cip", "fix_assets"), "pool": True,
    },
    "asset_goodwill_to_intangible": {
        "num_add": ("goodwill",), "num_sub": (),
        "den": ("goodwill", "intan_assets"), "pool": True,
    },
    "asset_inventory_to_payable": {
        "num_add": ("inventories",), "num_sub": (),
        "den": ("inventories", "acct_payable"), "pool": True,
    },
    "asset_prepay_to_customer_adv": {
        "num_add": ("prepayment",), "num_sub": (),
        "den": ("prepayment", "adv_receipts", "contract_liab"), "pool": True,
    },
    "asset_invest_property_to_fixed": {
        "num_add": ("invest_real_estate",), "num_sub": (),
        "den": ("invest_real_estate", "fix_assets"), "pool": True,
    },
    "asset_net_trade_wc_share": {
        "num_add": ("inventories", "accounts_receiv", "prepayment"),
        "num_sub": ("acct_payable", "adv_receipts", "contract_liab"),
        "den": ("total_assets",), "pool": False,
    },
    "asset_fix_to_nca": {
        "num_add": ("fix_assets",), "num_sub": (), "den": ("total_nca",), "pool": False,
    },
    "asset_lt_eqt_to_equity": {
        "num_add": ("lt_eqt_invest",), "num_sub": (),
        "den": ("total_hldr_eqy_exc_min_int",), "pool": False,
    },
    "asset_nca_within1y_to_current": {
        "num_add": ("nca_within_1y",), "num_sub": (), "den": ("total_cur_assets",), "pool": False,
    },
}

FAMILY_FIELDS = tuple(sorted({
    field
    for spec in SPECS.values()
    for field in (*spec["num_add"], *spec["num_sub"], *spec["den"])
}))
_TABLE_CACHE: dict[int, object] = {}


def _family_table(ctx):
    """Read the union field set once per build process / Context."""
    key = id(ctx)
    table = _TABLE_CACHE.get(key)
    if table is None:
        table = pit._report_table(ctx, "stock_balancesheet", FAMILY_FIELDS)
        _TABLE_CACHE[key] = table
    return table


def variant_values(data, variant):
    """Pure variant computation on a filtered whole-row PIT report table."""
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
        raise ValueError("Unknown frozen asset-composition variant")

    def compute(ctx):
        data = _family_table(ctx)
        events, values = variant_values(data, variant)
        return pit._map_fresh(ctx, events, values)

    return compute
