"""R17 seat_reopen batch 3: remaining field_indicator annual structure factors.

Third batch of reopened afx_fi_* annual records (non-q_*; q_* fields are excluded because missing Q4 values are encoded as 0.0).
Same frozen engineering contract as batch 1:
whole-row annual-report PIT, comp_type==1/report_type==1 for statements,
PIT=max(valid ann_date,valid f_ann_date,end_date), next-exchange-session
mapping exactly once, 550-day report-age cap, true zeros retained, no
missing-to-zero, ratio denominators >1e6 CNY for money ratios and >0.01 for per-share ratios, raw values (no asinh/rank).
All variants are GC descriptive accounting-structure descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from fea.deriv import KEY_W, PERIOD_BASE, PERIOD_W, period_of

Variant = Literal[
    "r17re_afx_fi_saleexp_to_gr_v1",
    "r17re_afx_fi_adminexp_of_gr_v1",
    "r17re_afx_fi_finaexp_of_gr_v1",
    "r17re_afx_fi_dp_assets_to_eqt_v1",
    "r17re_afx_fi_longdebt_to_workingcapital_v1",
    "r17re_afx_fi_tangibleasset_to_netdebt_v1",
    "r17re_afx_fi_ocf_to_or_v1",
    "r17re_afx_fi_networking_capital_v1",
    "r17re_afx_fi_invest_capital_v1",
    "r17re_afx_fi_ca_turn_v1",
    "r17re_afx_fi_total_fa_trun_v1",
    "r17re_afx_fi_diluted2_eps_v1",
    "r17re_afx_fi_retainedps_v1",
    "r17re_afx_fi_revenue_ps_v1",
    "r17re_afx_fi_surplus_rese_ps_v1",
    "r17re_afx_fi_bps_v1",
    "r17re_afx_fi_ar_turn_v1",
    "r17re_afx_fi_inv_turn_v1",
    "r17re_afx_fi_fa_turn_v1",
    "r17re_afx_fi_cogs_of_sales_v1",
    "r17re_afx_fi_ebit_to_interest_v1",
    "r17re_afx_fi_salescash_to_or_v1",
    "r17re_afx_fi_op_to_ebt_v1",
    "r17re_afx_fi_investincome_of_ebt_v1",
    "r17re_afx_fi_n_op_profit_of_ebt_v1",
    "r17re_afx_fi_roe_avg_v1",
    "r17re_afx_fi_basic_eps_yoy_v1",
    "r17re_afx_fi_cfps_yoy_v1",
    "r17re_afx_fi_opincome_of_ebt_v1",
    "r17re_afx_fi_equity_yoy_v1",
]
Kind = Literal["ratio", "delta", "growth"]

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
    min_den: float = MIN_DENOMINATOR


DEFINITIONS: Final[dict[Variant, VariantDefinition]] = {
    "r17re_afx_fi_saleexp_to_gr_v1": VariantDefinition(
        "r17re_afx_fi_saleexp_to_gr_v1", "stock_financial_indicator",
        ("saleexp_to_gr",), "delta", "saleexp_to_gr", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_adminexp_of_gr_v1": VariantDefinition(
        "r17re_afx_fi_adminexp_of_gr_v1", "stock_financial_indicator",
        ("adminexp_of_gr",), "delta", "adminexp_of_gr", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_finaexp_of_gr_v1": VariantDefinition(
        "r17re_afx_fi_finaexp_of_gr_v1", "stock_financial_indicator",
        ("finaexp_of_gr",), "delta", "finaexp_of_gr", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_dp_assets_to_eqt_v1": VariantDefinition(
        "r17re_afx_fi_dp_assets_to_eqt_v1", "stock_financial_indicator",
        ("dp_assets_to_eqt",), "delta", "dp_assets_to_eqt", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_longdebt_to_workingcapital_v1": VariantDefinition(
        "r17re_afx_fi_longdebt_to_workingcapital_v1",
        "stock_financial_indicator", ("longdebt_to_workingcapital",),
        "delta", "longdebt_to_workingcapital", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_tangibleasset_to_netdebt_v1": VariantDefinition(
        "r17re_afx_fi_tangibleasset_to_netdebt_v1",
        "stock_financial_indicator", ("tangibleasset_to_netdebt",),
        "delta", "tangibleasset_to_netdebt", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_ocf_to_or_v1": VariantDefinition(
        "r17re_afx_fi_ocf_to_or_v1", "stock_financial_indicator",
        ("ocf_to_or",), "delta", "ocf_to_or", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_networking_capital_v1": VariantDefinition(
        "r17re_afx_fi_networking_capital_v1", "stock_financial_indicator",
        ("networking_capital", "invest_capital"), "ratio",
        "networking_capital", "invest_capital",
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_invest_capital_v1": VariantDefinition(
        "r17re_afx_fi_invest_capital_v1", "stock_financial_indicator",
        ("invest_capital",), "growth", "invest_capital", None,
        ("stock_financial_indicator", "stock_daily"), 1e6),
    "r17re_afx_fi_ca_turn_v1": VariantDefinition(
        "r17re_afx_fi_ca_turn_v1", "stock_financial_indicator",
        ("ca_turn",), "delta", "ca_turn", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_total_fa_trun_v1": VariantDefinition(
        "r17re_afx_fi_total_fa_trun_v1", "stock_financial_indicator",
        ("total_fa_trun",), "delta", "total_fa_trun", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_diluted2_eps_v1": VariantDefinition(
        "r17re_afx_fi_diluted2_eps_v1", "stock_financial_indicator",
        ("diluted2_eps", "bps"), "ratio", "diluted2_eps", "bps",
        ("stock_financial_indicator", "stock_daily"), 0.01),
    "r17re_afx_fi_retainedps_v1": VariantDefinition(
        "r17re_afx_fi_retainedps_v1", "stock_financial_indicator",
        ("retainedps", "bps"), "ratio", "retainedps", "bps",
        ("stock_financial_indicator", "stock_daily"), 0.01),
    "r17re_afx_fi_revenue_ps_v1": VariantDefinition(
        "r17re_afx_fi_revenue_ps_v1", "stock_financial_indicator",
        ("revenue_ps", "bps"), "ratio", "revenue_ps", "bps",
        ("stock_financial_indicator", "stock_daily"), 0.01),
    "r17re_afx_fi_surplus_rese_ps_v1": VariantDefinition(
        "r17re_afx_fi_surplus_rese_ps_v1", "stock_financial_indicator",
        ("surplus_rese_ps", "bps"), "ratio", "surplus_rese_ps", "bps",
        ("stock_financial_indicator", "stock_daily"), 0.01),
    "r17re_afx_fi_bps_v1": VariantDefinition(
        "r17re_afx_fi_bps_v1", "stock_financial_indicator",
        ("bps",), "growth", "bps", None,
        ("stock_financial_indicator", "stock_daily"), 0.01),
    "r17re_afx_fi_ar_turn_v1": VariantDefinition(
        "r17re_afx_fi_ar_turn_v1", "stock_financial_indicator",
        ("ar_turn",), "delta", "ar_turn", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_inv_turn_v1": VariantDefinition(
        "r17re_afx_fi_inv_turn_v1", "stock_financial_indicator",
        ("inv_turn",), "delta", "inv_turn", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_fa_turn_v1": VariantDefinition(
        "r17re_afx_fi_fa_turn_v1", "stock_financial_indicator",
        ("fa_turn",), "delta", "fa_turn", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_cogs_of_sales_v1": VariantDefinition(
        "r17re_afx_fi_cogs_of_sales_v1", "stock_financial_indicator",
        ("cogs_of_sales",), "delta", "cogs_of_sales", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_ebit_to_interest_v1": VariantDefinition(
        "r17re_afx_fi_ebit_to_interest_v1", "stock_financial_indicator",
        ("ebit_to_interest",), "delta", "ebit_to_interest", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_salescash_to_or_v1": VariantDefinition(
        "r17re_afx_fi_salescash_to_or_v1", "stock_financial_indicator",
        ("salescash_to_or",), "delta", "salescash_to_or", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_op_to_ebt_v1": VariantDefinition(
        "r17re_afx_fi_op_to_ebt_v1", "stock_financial_indicator",
        ("op_to_ebt",), "delta", "op_to_ebt", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_investincome_of_ebt_v1": VariantDefinition(
        "r17re_afx_fi_investincome_of_ebt_v1", "stock_financial_indicator",
        ("investincome_of_ebt",), "delta", "investincome_of_ebt", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_n_op_profit_of_ebt_v1": VariantDefinition(
        "r17re_afx_fi_n_op_profit_of_ebt_v1", "stock_financial_indicator",
        ("n_op_profit_of_ebt",), "delta", "n_op_profit_of_ebt", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_roe_avg_v1": VariantDefinition(
        "r17re_afx_fi_roe_avg_v1", "stock_financial_indicator",
        ("roe_avg",), "delta", "roe_avg", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_basic_eps_yoy_v1": VariantDefinition(
        "r17re_afx_fi_basic_eps_yoy_v1", "stock_financial_indicator",
        ("basic_eps_yoy",), "delta", "basic_eps_yoy", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_cfps_yoy_v1": VariantDefinition(
        "r17re_afx_fi_cfps_yoy_v1", "stock_financial_indicator",
        ("cfps_yoy",), "delta", "cfps_yoy", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_opincome_of_ebt_v1": VariantDefinition(
        "r17re_afx_fi_opincome_of_ebt_v1", "stock_financial_indicator",
        ("opincome_of_ebt",), "delta", "opincome_of_ebt", None,
        ("stock_financial_indicator", "stock_daily")),
    "r17re_afx_fi_equity_yoy_v1": VariantDefinition(
        "r17re_afx_fi_equity_yoy_v1", "stock_financial_indicator",
        ("equity_yoy",), "delta", "equity_yoy", None,
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


def _safe_ratio(numerator, denominator, min_den: float = MIN_DENOMINATOR) -> np.ndarray:
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64))
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator)
    valid &= denominator > min_den
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
        return _safe_ratio(numerator, denominator, spec.min_den)
    if spec.kind == "delta":
        current = _lookup(vt, code, period, pit, spec.numerator)
        previous = _lookup(vt, code, period - 4, pit, spec.numerator)
        out = np.full(current.shape, np.nan, dtype=np.float64)
        ok = np.isfinite(current) & np.isfinite(previous)
        np.subtract(current, previous, out=out, where=ok)
        out[~np.isfinite(out)] = np.nan
        return out
    if spec.kind == "growth":
        current = _lookup(vt, code, period, pit, spec.numerator)
        previous = _lookup(vt, code, period - 4, pit, spec.numerator)
        out = np.full(current.shape, np.nan, dtype=np.float64)
        ok = np.isfinite(current) & np.isfinite(previous) & (np.abs(previous) > spec.min_den)
        np.subtract(current, previous, out=out, where=ok)
        np.divide(out, np.abs(previous), out=out, where=ok)
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
        raise ValueError("Only the 30 frozen R17 reopen batch-3 indicator descriptors are allowed")
    return lambda ctx: _evaluate(ctx, variant)
