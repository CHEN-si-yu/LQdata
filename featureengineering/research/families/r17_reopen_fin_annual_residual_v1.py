"""R17 seat_reopen batch 2: remaining annual financial-statement structure factors.

Second batch of reopened afx_bs_*/afx_is_*/afx_cf_*/afx_fi_* annual records.
Same frozen engineering contract as batch 1:
whole-row annual-report PIT, comp_type==1/report_type==1 for statements,
PIT=max(valid ann_date,valid f_ann_date,end_date), next-exchange-session
mapping exactly once, 550-day report-age cap, true zeros retained, no
missing-to-zero, ratio denominators >1e6 CNY, raw values (no asinh/rank).
All variants are GC descriptive accounting-structure descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from fea.deriv import KEY_W, PERIOD_BASE, PERIOD_W, period_of

Variant = Literal[
    "r17re_afx_bs_div_payable_v1",
    "r17re_afx_bs_div_payable_v1",
    "r17re_afx_bs_fix_assets_total_v1",
    "r17re_afx_bs_fix_assets_total_v1",
    "r17re_afx_bs_long_pay_total_v1",
    "r17re_afx_bs_long_pay_total_v1",
    "r17re_afx_bs_oth_comp_income_v1",
    "r17re_afx_bs_oth_comp_income_v1",
    "r17re_afx_bs_oth_cur_liab_v1",
    "r17re_afx_bs_oth_cur_liab_v1",
    "r17re_afx_bs_oth_nca_v1",
    "r17re_afx_bs_oth_nca_v1",
    "r17re_afx_bs_oth_pay_total_v1",
    "r17re_afx_bs_oth_pay_total_v1",
    "r17re_afx_bs_special_rese_v1",
    "r17re_afx_bs_special_rese_v1",
    "r17re_afx_bs_surplus_rese_v1",
    "r17re_afx_bs_surplus_rese_v1",
    "r17re_afx_is_compr_inc_attr_m_s_v1",
    "r17re_afx_is_compr_inc_attr_m_s_v1",
    "r17re_afx_cf_c_recp_return_invest_v1",
    "r17re_afx_cf_c_recp_return_invest_v1",
    "r17re_afx_cf_decr_def_inc_tax_assets_v1",
    "r17re_afx_cf_decr_def_inc_tax_assets_v1",
    "r17re_afx_cf_incr_def_inc_tax_liab_v1",
    "r17re_afx_cf_incr_def_inc_tax_liab_v1",
    "r17re_afx_cf_eff_fx_flu_cash_v1",
    "r17re_afx_cf_eff_fx_flu_cash_v1",
    "r17re_afx_cf_im_n_incr_cash_equ_v1",
    "r17re_afx_cf_im_n_incr_cash_equ_v1",
    "r17re_afx_cf_loss_fv_chg_v1",
    "r17re_afx_cf_loss_fv_chg_v1",
    "r17re_afx_cf_n_recp_disp_fiolta_v1",
    "r17re_afx_cf_n_recp_disp_fiolta_v1",
    "r17re_afx_cf_oth_cash_pay_oper_act_v1",
    "r17re_afx_cf_oth_cash_pay_oper_act_v1",
    "r17re_afx_cf_oth_cashpay_ral_fnc_act_v1",
    "r17re_afx_cf_oth_cashpay_ral_fnc_act_v1",
    "r17re_afx_fi_cash_to_liqdebt_v1",
    "r17re_afx_fi_cash_to_liqdebt_v1",
    "r17re_afx_fi_cash_to_liqdebt_withinterest_v1",
    "r17re_afx_fi_cash_to_liqdebt_withinterest_v1",
    "r17re_afx_fi_eqt_to_debt_v1",
    "r17re_afx_fi_eqt_to_debt_v1",
    "r17re_afx_fi_eqt_to_talcapital_v1",
    "r17re_afx_fi_eqt_to_talcapital_v1",
    "r17re_afx_fi_daa_v1",
    "r17re_afx_fi_daa_v1",
    "r17re_afx_fi_non_op_profit_v1",
    "r17re_afx_fi_non_op_profit_v1",
    "r17re_afx_fi_valuechange_income_v1",
    "r17re_afx_fi_valuechange_income_v1",
    "r17re_afx_fi_impai_ttm_v1",
    "r17re_afx_fi_impai_ttm_v1",
    "r17re_afx_fi_tax_to_ebt_v1",
    "r17re_afx_fi_tax_to_ebt_v1",
    "r17re_afx_fi_dtprofit_to_profit_v1",
    "r17re_afx_fi_dtprofit_to_profit_v1",
    "r17re_afx_fi_ebit_of_gr_v1",
    "r17re_afx_fi_ebit_of_gr_v1",
]
Kind = Literal["ratio", "delta"]

COMPANY_TYPE: Final = 1
REPORT_TYPE: Final = 1
MIN_DENOMINATOR: Final = 1_000_000.0
MAX_REPORT_AGE_DAYS: Final = 550
WARMUP_DAYS: Final = 1800


@dataclass(frozen=True)
class VariantDefinition:
    variant: Variant
    dataset: str
    fields: tuple[str, ...]
    kind: Kind
    numerator: str
    denominator: str | None = None
    dependencies: tuple[str, ...] = ("stock_daily",)


DEFINITIONS: Final[dict[Variant, VariantDefinition]] = {
    # ---- balance-sheet shares of total assets (total_liab_hldr_eqy) ----
    "r17re_afx_bs_div_payable_v1": VariantDefinition(
        "r17re_afx_bs_div_payable_v1", "stock_balancesheet",
        ("div_payable", "total_liab_hldr_eqy"), "ratio", "div_payable",
        "total_liab_hldr_eqy", ("stock_balancesheet", "stock_daily")),
    "r17re_afx_bs_fix_assets_total_v1": VariantDefinition(
        "r17re_afx_bs_fix_assets_total_v1", "stock_balancesheet",
        ("fix_assets_total", "total_liab_hldr_eqy"), "ratio",
        "fix_assets_total", "total_liab_hldr_eqy",
        ("stock_balancesheet", "stock_daily")),
    "r17re_afx_bs_long_pay_total_v1": VariantDefinition(
        "r17re_afx_bs_long_pay_total_v1", "stock_balancesheet",
        ("long_pay_total", "total_liab_hldr_eqy"), "ratio",
        "long_pay_total", "total_liab_hldr_eqy",
        ("stock_balancesheet", "stock_daily")),
    "r17re_afx_bs_oth_comp_income_v1": VariantDefinition(
        "r17re_afx_bs_oth_comp_income_v1", "stock_balancesheet",
        ("oth_comp_income", "total_liab_hldr_eqy"), "ratio",
        "oth_comp_income", "total_liab_hldr_eqy",
        ("stock_balancesheet", "stock_daily")),
    "r17re_afx_bs_oth_cur_liab_v1": VariantDefinition(
        "r17re_afx_bs_oth_cur_liab_v1", "stock_balancesheet",
        ("oth_cur_liab", "total_liab_hldr_eqy"), "ratio", "oth_cur_liab",
        "total_liab_hldr_eqy", ("stock_balancesheet", "stock_daily")),
    "r17re_afx_bs_oth_nca_v1": VariantDefinition(
        "r17re_afx_bs_oth_nca_v1", "stock_balancesheet",
        ("oth_nca", "total_liab_hldr_eqy"), "ratio", "oth_nca",
        "total_liab_hldr_eqy", ("stock_balancesheet", "stock_daily")),
    "r17re_afx_bs_oth_pay_total_v1": VariantDefinition(
        "r17re_afx_bs_oth_pay_total_v1", "stock_balancesheet",
        ("oth_pay_total", "total_liab_hldr_eqy"), "ratio", "oth_pay_total",
        "total_liab_hldr_eqy", ("stock_balancesheet", "stock_daily")),
    "r17re_afx_bs_special_rese_v1": VariantDefinition(
        "r17re_afx_bs_special_rese_v1", "stock_balancesheet",
        ("special_rese", "total_liab_hldr_eqy"), "ratio", "special_rese",
        "total_liab_hldr_eqy", ("stock_balancesheet", "stock_daily")),
    "r17re_afx_bs_surplus_rese_v1": VariantDefinition(
        "r17re_afx_bs_surplus_rese_v1", "stock_balancesheet",
        ("surplus_rese", "total_liab_hldr_eqy"), "ratio", "surplus_rese",
        "total_liab_hldr_eqy", ("stock_balancesheet", "stock_daily")),
    # ---- income-statement share of total revenue ----
    "r17re_afx_is_compr_inc_attr_m_s_v1": VariantDefinition(
        "r17re_afx_is_compr_inc_attr_m_s_v1", "stock_income",
        ("compr_inc_attr_m_s", "total_revenue"), "ratio",
        "compr_inc_attr_m_s", "total_revenue", ("stock_income", "stock_daily")),
    # ---- cash-flow shares of total operating cash inflow ----
    "r17re_afx_cf_c_recp_return_invest_v1": VariantDefinition(
        "r17re_afx_cf_c_recp_return_invest_v1", "stock_cashflow",
        ("c_recp_return_invest", "c_inf_fr_operate_a"), "ratio",
        "c_recp_return_invest", "c_inf_fr_operate_a",
        ("stock_cashflow", "stock_daily")),
    "r17re_afx_cf_decr_def_inc_tax_assets_v1": VariantDefinition(
        "r17re_afx_cf_decr_def_inc_tax_assets_v1", "stock_cashflow",
        ("decr_def_inc_tax_assets", "c_inf_fr_operate_a"), "ratio",
        "decr_def_inc_tax_assets", "c_inf_fr_operate_a",
        ("stock_cashflow", "stock_daily")),
    "r17re_afx_cf_incr_def_inc_tax_liab_v1": VariantDefinition(
        "r17re_afx_cf_incr_def_inc_tax_liab_v1", "stock_cashflow",
        ("incr_def_inc_tax_liab", "c_inf_fr_operate_a"), "ratio",
        "incr_def_inc_tax_liab", "c_inf_fr_operate_a",
        ("stock_cashflow", "stock_daily")),
    "r17re_afx_cf_eff_fx_flu_cash_v1": VariantDefinition(
        "r17re_afx_cf_eff_fx_flu_cash_v1", "stock_cashflow",
        ("eff_fx_flu_cash", "c_inf_fr_operate_a"), "ratio",
        "eff_fx_flu_cash", "c_inf_fr_operate_a",
        ("stock_cashflow", "stock_daily")),
    "r17re_afx_cf_im_n_incr_cash_equ_v1": VariantDefinition(
        "r17re_afx_cf_im_n_incr_cash_equ_v1", "stock_cashflow",
        ("im_n_incr_cash_equ", "c_inf_fr_operate_a"), "ratio",
        "im_n_incr_cash_equ", "c_inf_fr_operate_a",
        ("stock_cashflow", "stock_daily")),
    "r17re_afx_cf_loss_fv_chg_v1": VariantDefinition(
        "r17re_afx_cf_loss_fv_chg_v1", "stock_cashflow",
        ("loss_fv_chg", "c_inf_fr_operate_a"), "ratio", "loss_fv_chg",
        "c_inf_fr_operate_a", ("stock_cashflow", "stock_daily")),
    "r17re_afx_cf_n_recp_disp_fiolta_v1": VariantDefinition(
        "r17re_afx_cf_n_recp_disp_fiolta_v1", "stock_cashflow",
        ("n_recp_disp_fiolta", "c_inf_fr_operate_a"), "ratio",
        "n_recp_disp_fiolta", "c_inf_fr_operate_a",
        ("stock_cashflow", "stock_daily")),
    "r17re_afx_cf_oth_cash_pay_oper_act_v1": VariantDefinition(
        "r17re_afx_cf_oth_cash_pay_oper_act_v1", "stock_cashflow",
        ("oth_cash_pay_oper_act", "c_inf_fr_operate_a"), "ratio",
        "oth_cash_pay_oper_act", "c_inf_fr_operate_a",
        ("stock_cashflow", "stock_daily")),
    "r17re_afx_cf_oth_cashpay_ral_fnc_act_v1": VariantDefinition(
        "r17re_afx_cf_oth_cashpay_ral_fnc_act_v1", "stock_cashflow",
        ("oth_cashpay_ral_fnc_act", "c_inf_fr_operate_a"), "ratio",
        "oth_cashpay_ral_fnc_act", "c_inf_fr_operate_a",
        ("stock_cashflow", "stock_daily")),
    # ---- financial-indicator shares and annual deltas ----
    "r17re_afx_fi_cash_to_liqdebt_v1": VariantDefinition(
        "r17re_afx_fi_cash_to_liqdebt_v1", "stock_financial_indicator",
        ("cash_to_liqdebt",), "delta", "cash_to_liqdebt", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_cash_to_liqdebt_withinterest_v1": VariantDefinition(
        "r17re_afx_fi_cash_to_liqdebt_withinterest_v1",
        "stock_financial_indicator", ("cash_to_liqdebt_withinterest",),
        "delta", "cash_to_liqdebt_withinterest", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_eqt_to_debt_v1": VariantDefinition(
        "r17re_afx_fi_eqt_to_debt_v1", "stock_financial_indicator",
        ("eqt_to_debt",), "delta", "eqt_to_debt", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_eqt_to_talcapital_v1": VariantDefinition(
        "r17re_afx_fi_eqt_to_talcapital_v1", "stock_financial_indicator",
        ("eqt_to_talcapital",), "delta", "eqt_to_talcapital", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_daa_v1": VariantDefinition(
        "r17re_afx_fi_daa_v1", "stock_financial_indicator",
        ("daa", "invest_capital"), "ratio", "daa", "invest_capital",
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_non_op_profit_v1": VariantDefinition(
        "r17re_afx_fi_non_op_profit_v1", "stock_financial_indicator",
        ("non_op_profit", "invest_capital"), "ratio", "non_op_profit",
        "invest_capital", ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_valuechange_income_v1": VariantDefinition(
        "r17re_afx_fi_valuechange_income_v1", "stock_financial_indicator",
        ("valuechange_income", "invest_capital"), "ratio",
        "valuechange_income", "invest_capital",
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_impai_ttm_v1": VariantDefinition(
        "r17re_afx_fi_impai_ttm_v1", "stock_financial_indicator",
        ("impai_ttm",), "delta", "impai_ttm", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_tax_to_ebt_v1": VariantDefinition(
        "r17re_afx_fi_tax_to_ebt_v1", "stock_financial_indicator",
        ("tax_to_ebt",), "delta", "tax_to_ebt", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_dtprofit_to_profit_v1": VariantDefinition(
        "r17re_afx_fi_dtprofit_to_profit_v1", "stock_financial_indicator",
        ("dtprofit_to_profit",), "delta", "dtprofit_to_profit", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_ebit_of_gr_v1": VariantDefinition(
        "r17re_afx_fi_ebit_of_gr_v1", "stock_financial_indicator",
        ("ebit_of_gr",), "delta", "ebit_of_gr", None,
        ("stock_financial_indicator", "stock_daily")),
}

DATASET_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    dataset: tuple(dict.fromkeys(
        field for spec in DEFINITIONS.values() if spec.dataset == dataset
        for field in spec.fields))
    for dataset in {spec.dataset for spec in DEFINITIONS.values()}
}

PARAMETERS: Final = {
    name: {"variant": name, "definition_version": 1}
    for name in DEFINITIONS
}

def _valid_dates(ctx, values) -> np.ndarray:
    raw = np.asarray(ctx.date_col(values), dtype=np.int64)
    parsed = pd.to_datetime(raw.astype(str), format="%Y%m%d", errors="coerce")
    return np.where((raw >= 19900101) & ~np.asarray(pd.isna(parsed)), raw, 0)


def _source_years(ctx) -> tuple[int, int]:
    """Report-year partitions needed for the FY frontier and its FY-1 lag."""
    if not ctx.panel.T:
        return (PERIOD_BASE, PERIOD_BASE)
    lo = int(ctx.panel.dates[0]) // 10000
    hi = int(ctx.panel.dates[-1]) // 10000
    return max(PERIOD_BASE, lo - 5), hi


def _report_table(ctx, dataset: str) -> pd.DataFrame:
    cache = ctx.__dict__.setdefault("_r17re_annual_tables", {})
    if dataset in cache:
        return cache[dataset]
    fields = DATASET_FIELDS[dataset]
    is_indicator = dataset == "stock_financial_indicator"
    columns = ["stock_code", "ann_date", "end_date", *fields]
    if not is_indicator:
        columns = ["stock_code", "ann_date", "f_ann_date", "end_date",
                   "comp_type", "report_type", *fields]
    frame = ctx.dataset(dataset, columns=columns, years=_source_years(ctx))
    out_columns = ["code", "period", "pit", *fields]
    if frame is None or frame.empty:
        data = pd.DataFrame(columns=out_columns)
        cache[dataset] = data
        return data

    end = _valid_dates(ctx, frame["end_date"])
    ann = _valid_dates(ctx, frame["ann_date"])
    if is_indicator:
        fann = np.zeros_like(ann)
        known = ann >= 19900101
    else:
        fann = _valid_dates(ctx, frame["f_ann_date"])
        known = (ann >= 19900101) | (fann >= 19900101)
    pit = np.maximum(np.maximum(ann, fann), end)

    keep = (end % 10000) == 1231
    keep &= (end >= 19900101) & (pit >= end) & (pit > 0) & known
    if not is_indicator:
        comp = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy(
            dtype=np.float64)
        report = pd.to_numeric(frame["report_type"], errors="coerce").to_numpy(
            dtype=np.float64)
        keep &= (comp == COMPANY_TYPE) & (report == REPORT_TYPE)
    if ctx.panel.T:
        keep &= pit <= int(ctx.panel.dates[-1])

    codes = ctx.code_index(frame["stock_code"].to_numpy()[keep])
    data = pd.DataFrame({
        "code": codes,
        "period": period_of(end[keep]),
        "pit": pit[keep],
        **{field: pd.to_numeric(frame[field], errors="coerce").to_numpy(
            dtype=np.float64)[keep] for field in fields},
    })
    data = data.loc[data.code >= 0]
    data = data.drop_duplicates(["code", "period", "pit"], keep="last")
    cache[dataset] = data
    return data


def _sorted_versions(data: pd.DataFrame, fields: tuple[str, ...]) -> dict:
    if data.empty:
        return {"keys": np.empty(0, dtype=np.int64),
                "code": np.empty(0, dtype=np.int64),
                "period": np.empty(0, dtype=np.int64),
                **{field: np.empty(0, dtype=np.float64) for field in fields}}
    data = data.sort_values(["code", "period", "pit"], kind="stable")
    code = data.code.to_numpy(dtype=np.int64)
    period = data.period.to_numpy(dtype=np.int64)
    pit = data.pit.to_numpy(dtype=np.int64)
    return {"keys": (code * PERIOD_W + period) * KEY_W + pit,
            "code": code, "period": period,
            **{field: data[field].to_numpy(dtype=np.float64) for field in fields}}


def _lookup(vt: dict, code, period, pit, field: str) -> np.ndarray:
    code, period, pit = np.broadcast_arrays(code, period, pit)
    if not len(vt["keys"]):
        return np.full(code.shape, np.nan, dtype=np.float64)
    query = (code * PERIOD_W + period) * KEY_W + pit
    pos = np.searchsorted(vt["keys"], query, side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (vt["code"][safe] == code) & (vt["period"][safe] == period)
    return np.where(ok, vt[field][safe], np.nan).astype(np.float64, copy=False)


def _safe_ratio(numerator, denominator) -> np.ndarray:
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64))
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator)
    valid &= denominator > MIN_DENOMINATOR
    np.divide(numerator, denominator, out=out, where=valid)
    out[~np.isfinite(out)] = np.nan
    return out


def _frontier_events(data: pd.DataFrame) -> pd.DataFrame:
    """One PIT event at the prefix-maximum FY report period in this table."""
    if data.empty:
        return pd.DataFrame(columns=["code", "pit", "period"])
    present = data[["code", "period", "pit"]].drop_duplicates()
    present = present.sort_values(["code", "pit", "period"], kind="stable")
    present["period"] = present.groupby("code", sort=False).period.cummax()
    return present.drop_duplicates(["code", "pit"], keep="last")[
        ["code", "pit", "period"]].reset_index(drop=True)


def _compute_events(events: pd.DataFrame, vt: dict,
                    variant: Variant) -> np.ndarray:
    if events.empty:
        return np.empty(0, dtype=np.float64)
    spec = DEFINITIONS[variant]
    code = events.code.to_numpy(dtype=np.int64)
    period = events.period.to_numpy(dtype=np.int64)
    pit = events.pit.to_numpy(dtype=np.int64)
    if spec.kind == "ratio":
        numerator = _lookup(vt, code, period, pit, spec.numerator)
        denominator = _lookup(vt, code, period, pit, spec.denominator)
        return _safe_ratio(numerator, denominator)
    if spec.kind == "delta":
        current = _lookup(vt, code, period, pit, spec.numerator)
        previous = _lookup(vt, code, period - 4, pit, spec.numerator)
        out = np.full(current.shape, np.nan, dtype=np.float64)
        ok = np.isfinite(current) & np.isfinite(previous)
        np.subtract(current, previous, out=out, where=ok)
        out[~np.isfinite(out)] = np.nan
        return out
    raise AssertionError(f"Unknown frozen kind: {spec.kind}")


def _period_end(period) -> np.ndarray:
    period = np.asarray(period, dtype=np.int64)
    md = np.array([331, 630, 930, 1231], dtype=np.int64)[period % 4]
    return (PERIOD_BASE + period // 4) * 10000 + md


def _epoch_days(yyyymmdd) -> np.ndarray:
    return pd.to_datetime(np.asarray(yyyymmdd).astype(str), format="%Y%m%d",
                          errors="raise").to_numpy(dtype="datetime64[D]").astype(np.int64)


def _map_daily(ctx, events: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    empty = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    if events.empty or not ctx.panel.T:
        return empty
    pit = events.pit.to_numpy(dtype=np.int64)
    effective = np.asarray(ctx.next_trading_day(pit), dtype=np.int64)
    end = _period_end(events.period.to_numpy(dtype=np.int64))
    usable = (effective > pit) & (effective <= int(ctx.panel.dates[-1]))
    if not usable.any():
        return empty
    mapped = pd.DataFrame({
        "code": events.code.to_numpy(dtype=np.int64)[usable],
        "effective": effective[usable], "pit": pit[usable],
        "end_epoch": _epoch_days(end[usable]),
        "value": np.asarray(values, dtype=np.float64)[usable],
    })
    # Multiple releases mapped to one next session resolve to the last visible
    # source PIT for that stock/session; a NaN value is state-clearing.
    mapped = mapped.sort_values(["code", "effective", "pit"], kind="stable")
    mapped = mapped.drop_duplicates(["code", "effective"], keep="last")
    code = mapped.code.to_numpy(dtype=np.int64)
    effective = mapped.effective.to_numpy(dtype=np.int64)
    state = ctx.panel.asof(code, effective, mapped.value.to_numpy(dtype=np.float64),
                           dtype=np.float64)
    report_end = ctx.panel.asof(code, effective,
                                mapped.end_epoch.to_numpy(dtype=np.float64),
                                dtype=np.float64)
    age = _epoch_days(ctx.panel.dates)[:, None] - report_end
    valid = np.isfinite(state) & np.isfinite(age) & (age >= 0)
    valid &= age <= MAX_REPORT_AGE_DAYS
    valid &= np.asarray(ctx.universe, dtype=bool)
    valid &= np.asarray(ctx.traded(), dtype=bool)
    return np.where(valid, state, np.nan)


def _evaluate(ctx, variant: Variant) -> np.ndarray:
    spec = DEFINITIONS[variant]
    data = _report_table(ctx, spec.dataset)
    events = _frontier_events(data)
    vt = _sorted_versions(data, spec.fields)
    values = _compute_events(events, vt, variant)
    return _map_daily(ctx, events, values)


def build(parameters: dict):
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be one exact frozen variant dictionary")
    variant = parameters.get("variant")
    if variant not in PARAMETERS or parameters != PARAMETERS[variant]:
        raise ValueError("Only the 30 frozen R17 reopen batch-2 annual descriptors are allowed")
    return lambda ctx: _evaluate(ctx, variant)
