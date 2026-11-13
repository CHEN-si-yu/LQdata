"""R38 raw-field annual factors: OCI/equity, dividend intensity, opaque liabilities."""
from __future__ import annotations

import numpy as np

from factors.registered import r38_field_common as common

VERSION = 1
VARIANTS = {
    "oth_comp_income_to_equity_v1": {
        "dataset": "stock_balancesheet", "num": "oth_comp_income",
        "den": "total_hldr_eqy_exc_min_int", "sign": 1,
    },
    "dividend_paid_to_revenue_v1": {
        "dataset": "stock_cashflow", "num": "incl_dvd_profit_paid_sc_ms",
        "den_dataset": "stock_income", "den": "revenue", "sign": 1,
    },
    "oth_cur_liab_to_assets_v1": {
        "dataset": "stock_balancesheet", "num": "oth_cur_liab",
        "den": "total_assets", "sign": -1,
    },
    "oth_payable_to_assets_v1": {
        "dataset": "stock_balancesheet", "num": "oth_payable",
        "den": "total_assets", "sign": -1,
    },
    "oth_financing_cash_payment_share_v1": {
        "dataset": "stock_cashflow", "num": "oth_cashpay_ral_fnc_act",
        "den_dataset": "stock_income", "den": "revenue", "sign": -1,
    },
}
DEPS = {
    "oth_comp_income_to_equity_v1": ("stock_balancesheet",),
    "dividend_paid_to_revenue_v1": ("stock_cashflow", "stock_income"),
    "oth_cur_liab_to_assets_v1": ("stock_balancesheet",),
    "oth_payable_to_assets_v1": ("stock_balancesheet",),
    "oth_financing_cash_payment_share_v1": ("stock_cashflow", "stock_income"),
}


def build(parameters):
    params = dict(parameters or {})
    variant = params.get("variant")
    if variant not in VARIANTS or params.get("version", VERSION) != VERSION:
        raise ValueError(f"Unknown R38 field variant: {variant!r}")
    cfg = VARIANTS[variant]
    den_ds = cfg.get("den_dataset", cfg["dataset"])

    def compute(ctx):
        data, years = common.annual_panels(ctx, cfg["dataset"], (cfg["num"],))
        if den_ds == cfg["dataset"]:
            den_panel = common.annual_panels(ctx, den_ds, (cfg["den"],))[0][cfg["den"]]
            num = data[cfg["num"]]
        else:
            other, other_years = common.annual_panels(ctx, den_ds, (cfg["den"],))
            # Cross-statement ratios must use the same fiscal year on both legs.
            same = np.isfinite(years) & np.isfinite(other_years) & (years == other_years)
            num = np.where(same, data[cfg["num"]], np.nan)
            den_panel = np.where(same, other[cfg["den"]], np.nan)
        ratio = common.ratio(num, den_panel)
        return np.arcsinh(np.where(np.isfinite(ratio), ratio, np.nan))

    return compute
