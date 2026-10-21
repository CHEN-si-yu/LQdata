"""Artifact-only R13 six-descriptor financial GC draft (FI05 held separately).

No return labels, factor panels, or outcome statistics are read here. Every
executable descriptor reads one declared statement table, filters standard
report rows before whole-row vintage selection, uses PIT=max(ann,f_ann,end),
and maps the event to the project's next exchange session exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd


Variant = Literal[
    "FI01_selling_admin_mix",
    "FC01_payroll_vs_goods_cash_mix",
    "FC02_financial_investment_outflow_share",
    "FB05_deferred_tax_asset_share",
    "FB06_long_equity_investment_asset_mix",
    "FB07_supplier_prepayment_asset_share",
]
Kind = Literal["ttm_component_share", "ttm_ratio", "point_ratio"]

COMPANY_TYPE: Final = 1
REPORT_TYPE: Final = 1
MIN_DENOMINATOR: Final = 1_000_000.0
MAX_REPORT_AGE_DAYS: Final = 550
WARMUP_DAYS: Final = 1800
PERIOD_BASE: Final = 1990
PERIOD_W: Final = 100_000
KEY_W: Final = 100_000_000


@dataclass(frozen=True)
class VariantDefinition:
    variant: Variant
    dataset: str
    fields: tuple[str, ...]
    kind: Kind
    numerator: str
    denominator: str | None = None
    additive_denominator: str | None = None
    unit: str = "dimensionless"
    dependencies: tuple[str, ...] = ()


DEFINITIONS: Final[dict[Variant, VariantDefinition]] = {
    "FI01_selling_admin_mix": VariantDefinition(
        variant="FI01_selling_admin_mix",
        dataset="stock_income",
        fields=("sell_exp", "admin_exp"),
        kind="ttm_component_share",
        numerator="sell_exp",
        additive_denominator="admin_exp",
        dependencies=("stock_income", "stock_daily"),
    ),
    "FC01_payroll_vs_goods_cash_mix": VariantDefinition(
        variant="FC01_payroll_vs_goods_cash_mix",
        dataset="stock_cashflow",
        fields=("c_paid_to_for_empl", "c_paid_goods_s"),
        kind="ttm_component_share",
        numerator="c_paid_to_for_empl",
        additive_denominator="c_paid_goods_s",
        dependencies=("stock_cashflow", "stock_daily"),
    ),
    "FC02_financial_investment_outflow_share": VariantDefinition(
        variant="FC02_financial_investment_outflow_share",
        dataset="stock_cashflow",
        fields=("c_paid_invest", "stot_out_inv_act"),
        kind="ttm_ratio",
        numerator="c_paid_invest",
        denominator="stot_out_inv_act",
        dependencies=("stock_cashflow", "stock_daily"),
    ),
    "FB05_deferred_tax_asset_share": VariantDefinition(
        variant="FB05_deferred_tax_asset_share",
        dataset="stock_balancesheet",
        fields=("defer_tax_assets", "total_assets"),
        kind="point_ratio",
        numerator="defer_tax_assets",
        denominator="total_assets",
        dependencies=("stock_balancesheet", "stock_daily"),
    ),
    "FB06_long_equity_investment_asset_mix": VariantDefinition(
        variant="FB06_long_equity_investment_asset_mix",
        dataset="stock_balancesheet",
        fields=("lt_eqt_invest", "total_nca"),
        kind="point_ratio",
        numerator="lt_eqt_invest",
        denominator="total_nca",
        dependencies=("stock_balancesheet", "stock_daily"),
    ),
    "FB07_supplier_prepayment_asset_share": VariantDefinition(
        variant="FB07_supplier_prepayment_asset_share",
        dataset="stock_balancesheet",
        fields=("prepayment", "total_cur_assets"),
        kind="point_ratio",
        numerator="prepayment",
        denominator="total_cur_assets",
        dependencies=("stock_balancesheet", "stock_daily"),
    ),
}

PARAMETERS: Final = {
    name: {"variant": name, "definition_version": 1}
    for name in DEFINITIONS
}

# FI05_minority_earnings_share is deliberately held from executable definitions.
# Under a common income-table frontier and common YTD-to-TTM transform,
# minority_gain / n_income =
# (total_profit - income_tax - n_income_attr_p) /
# (total_profit - income_tax).
HOLD_DEFINITIONS: Final = {
    "FI05_minority_earnings_share": {
        "dataset": "stock_income",
        "fields": ("minority_gain", "n_income"),
        "expression": "TTM(minority_gain) / TTM(n_income)",
        "status": "HOLD_STRUCTURAL_RECONSTRUCTION",
        "reconstruction": (
            "If same-frontier provider identity n_income=total_profit-income_tax "
            "and net_income_attributable=n_income-minority_gain is respected, "
            "then the target is exactly (A-B-C)/(A-B), where A=TTM(total_profit)/TTM(revenue), "
            "B=TTM(income_tax)/TTM(revenue), and C=TTM(n_income_attr_p)/TTM(revenue); "
            "algebra requires common nonzero revenue and identical report vintages."
        ),
        "potential_source_gap": (
            "Any observed difference must be established as a source-contract, vintage, "
            "or coverage distinction before calling it new information; engineering "
            "clock differences alone do not establish an independent construct."
        ),
    }
}


def _empty_table(fields: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=["code", "period", "pit", *fields])


def _valid_dates(ctx, values) -> np.ndarray:
    raw = np.asarray(ctx.date_col(values), dtype=np.int64)
    parsed = pd.to_datetime(raw.astype(str), format="%Y%m%d", errors="coerce")
    valid = (raw >= 19900101) & ~np.asarray(pd.isna(parsed))
    return np.where(valid, raw, 0)


def _period_of(end_date) -> np.ndarray:
    end_date = np.asarray(end_date, dtype=np.int64)
    year = end_date // 10000
    month = (end_date // 100) % 100
    return ((year - PERIOD_BASE) * 4 + (month // 3 - 1)).astype(np.int64)


def _period_end(period) -> np.ndarray:
    period = np.asarray(period, dtype=np.int64)
    month_day = np.array([331, 630, 930, 1231], dtype=np.int64)[period % 4]
    year = PERIOD_BASE + period // 4
    return year * 10000 + month_day


def _yyyymmdd_to_days(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    return pd.to_datetime(values.astype(str), format="%Y%m%d", errors="raise") \
        .to_numpy(dtype="datetime64[D]").astype(np.int64)


def _report_table(ctx, dataset: str, fields: tuple[str, ...]) -> pd.DataFrame:
    """Project a single raw table; standard-filter then whole-row PIT dedup.

    A latest row with a missing/nonfinite amount stays in the vintages and
    clears the older amount. It is not removed before the union frontier.
    """
    columns = [
        "stock_code", "ann_date", "f_ann_date", "end_date",
        "comp_type", "report_type", *fields,
    ]
    frame = ctx.dataset(dataset, columns=columns)
    if frame is None or frame.empty:
        return _empty_table(fields)

    end = _valid_dates(ctx, frame["end_date"])
    ann = _valid_dates(ctx, frame["ann_date"])
    fann = _valid_dates(ctx, frame["f_ann_date"])
    pit = np.maximum(np.maximum(ann, fann), end)
    comp = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy(float)
    report = pd.to_numeric(frame["report_type"], errors="coerce").to_numpy(float)

    keep = np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= (end >= 19900101) & (pit >= end) & (pit > 0)
    keep &= (ann >= 19900101) | (fann >= 19900101)
    # The frozen contract filters report type before vintage selection.
    keep &= (comp == COMPANY_TYPE) & (report == REPORT_TYPE)
    keep &= pit <= int(ctx.panel.dates[-1]) if ctx.panel.T else False

    chosen = np.flatnonzero(keep)
    codes = ctx.code_index(frame["stock_code"].to_numpy()[chosen])
    data = pd.DataFrame({
        "code": codes,
        "period": _period_of(end[chosen]),
        "pit": pit[chosen],
        **{
            field: pd.to_numeric(frame[field], errors="coerce").to_numpy(
                dtype=np.float64
            )[chosen]
            for field in fields
        },
    })
    data = data.loc[data.code >= 0].copy()
    return data.drop_duplicates(["code", "period", "pit"], keep="last")


def _frontier_events(data: pd.DataFrame) -> pd.DataFrame:
    """One PIT event, at the prefix-maximum period on this table's frontier."""
    if data.empty:
        return pd.DataFrame(columns=["code", "pit", "period"])
    present = data[["code", "period", "pit"]].drop_duplicates()
    present = present.sort_values(["code", "pit", "period"], kind="stable")
    present["frontier"] = present.groupby("code", sort=False).period.cummax()
    present = present.drop_duplicates(["code", "pit"], keep="last")
    return present[["code", "pit", "frontier"]].rename(
        columns={"frontier": "period"}
    ).reset_index(drop=True)


def _sorted_versions(data: pd.DataFrame, fields: tuple[str, ...]) -> dict:
    data = data.sort_values(["code", "period", "pit"], kind="stable")
    code = data.code.to_numpy(dtype=np.int64)
    period = data.period.to_numpy(dtype=np.int64)
    pit = data.pit.to_numpy(dtype=np.int64)
    out = {
        "keys": (code * PERIOD_W + period) * KEY_W + pit,
        "code": code,
        "period": period,
    }
    for field in fields:
        out[field] = data[field].to_numpy(dtype=np.float64)
    return out


def _lookup(vt: dict, code, period, pit, field: str) -> np.ndarray:
    code, period, pit = np.broadcast_arrays(code, period, pit)
    shape = code.shape
    if not len(vt["keys"]):
        return np.full(shape, np.nan, dtype=np.float64)
    query = (code * PERIOD_W + period) * KEY_W + pit
    pos = np.searchsorted(vt["keys"], query, side="right") - 1
    safe = np.maximum(pos, 0)
    valid = pos >= 0
    valid &= vt["code"][safe] == code
    valid &= vt["period"][safe] == period
    values = vt[field][safe]
    return np.where(valid, values, np.nan).astype(np.float64, copy=False)


def _strict_ttm(vt: dict, code, period, pit, field: str) -> np.ndarray:
    """Strict three-row YTD TTM; exact report slots only, no partial fallback."""
    code, period, pit = np.broadcast_arrays(code, period, pit)
    own = _lookup(vt, code, period, pit, field)
    q4 = period % 4 == 3
    previous_fy_q4 = 4 * (period // 4) - 1
    previous_same_quarter = period - 4
    fy = _lookup(vt, code, previous_fy_q4, pit, field)
    prior_ytd = _lookup(vt, code, previous_same_quarter, pit, field)
    value = np.where(q4, own, own + fy - prior_ytd)
    valid = np.isfinite(own)
    valid &= q4 | (np.isfinite(fy) & np.isfinite(prior_ytd))
    valid &= np.isfinite(value)
    return np.where(valid, value, np.nan).astype(np.float64, copy=False)


def _safe_ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator)
    valid &= denominator > MIN_DENOMINATOR
    np.divide(numerator, denominator, out=out, where=valid)
    out[~np.isfinite(out)] = np.nan
    return out


def _value_at(vt: dict, code, period, pit, variant: Variant) -> np.ndarray:
    spec = DEFINITIONS[variant]
    if spec.kind == "ttm_component_share":
        a = _strict_ttm(vt, code, period, pit, spec.numerator)
        b = _strict_ttm(vt, code, period, pit, spec.additive_denominator)
        return _safe_ratio(a, a + b)
    if spec.kind == "ttm_ratio":
        numerator = _strict_ttm(vt, code, period, pit, spec.numerator)
        denominator = _strict_ttm(vt, code, period, pit, spec.denominator)
        return _safe_ratio(numerator, denominator)
    if spec.kind == "point_ratio":
        numerator = _lookup(vt, code, period, pit, spec.numerator)
        denominator = _lookup(vt, code, period, pit, spec.denominator)
        return _safe_ratio(numerator, denominator)
    raise AssertionError(f"Unknown frozen kind: {spec.kind}")


def _compute_events(events: pd.DataFrame, vt: dict, variant: Variant) -> np.ndarray:
    if events.empty:
        return np.empty(0, dtype=np.float64)
    return _value_at(
        vt,
        events.code.to_numpy(dtype=np.int64),
        events.period.to_numpy(dtype=np.int64),
        events.pit.to_numpy(dtype=np.int64),
        variant,
    )


def _map_daily(ctx, events: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    empty = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    if events.empty or not ctx.panel.T:
        return empty

    pit = events.pit.to_numpy(dtype=np.int64)
    effective = np.asarray(ctx.next_trading_day(pit), dtype=np.int64)
    mapped = pd.DataFrame({
        "code": events.code.to_numpy(dtype=np.int64),
        "pit": pit,
        "effective": effective,
        "period": events.period.to_numpy(dtype=np.int64),
        "value": np.asarray(values, dtype=np.float64),
    })
    mapped = mapped[(mapped.effective > mapped.pit)
                    & (mapped.effective <= int(ctx.panel.dates[-1]))]
    if mapped.empty:
        return empty
    mapped = mapped.sort_values(
        ["code", "effective", "pit"], kind="stable"
    ).drop_duplicates(["code", "effective"], keep="last")
    code = mapped.code.to_numpy(dtype=np.int64)
    effective = mapped.effective.to_numpy(dtype=np.int64)
    value = mapped.value.to_numpy(dtype=np.float64)
    report_end = _yyyymmdd_to_days(_period_end(mapped.period.to_numpy(dtype=np.int64)))
    state = ctx.panel.asof(code, effective, value, dtype=np.float64)
    end_state = ctx.panel.asof(
        code, effective, report_end.astype(np.float64), dtype=np.float64
    )
    age = _yyyymmdd_to_days(ctx.panel.dates)[:, None] - end_state
    valid = np.isfinite(state) & (age >= 0) & (age <= MAX_REPORT_AGE_DAYS)
    valid &= np.asarray(ctx.universe, dtype=bool)
    valid &= np.asarray(ctx.traded(), dtype=bool)
    return np.where(valid, state, np.nan)


def _evaluate(ctx, variant: Variant) -> np.ndarray:
    spec = DEFINITIONS[variant]
    data = _report_table(ctx, spec.dataset, spec.fields)
    events = _frontier_events(data)
    values = _compute_events(events, _sorted_versions(data, spec.fields), variant)
    return _map_daily(ctx, events, values)


def build(parameters: dict):
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be one exact frozen variant dictionary")
    variant = parameters.get("variant")
    if variant not in PARAMETERS or parameters != PARAMETERS[variant]:
        raise ValueError("Only the six frozen executable R13 GC descriptors are allowed")
    return lambda ctx: _evaluate(ctx, variant)
