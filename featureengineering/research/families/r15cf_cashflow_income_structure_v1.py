"""R15 seat_cashflow C1 builder: cash-flow / income-statement structure shares
and own-history seasonal-trend deviations.

Scope (R15 seat brief): cash-flow statement and income-statement *structure
shares* and *innovations relative to own history*. No balance-sheet, working
capital, or investment-efficiency object is read here.

All eleven frozen variants are GC descriptive accounting descriptors; no
future-return direction is declared or implied. Frozen conventions (identical
to the admitted r13gc financial-composition and r14f1/r13f innovation
builders):

* one declared statement table per variant; comp_type==1 and report_type==1
  are filtered **before** whole-row (code, period, PIT) vintage selection;
* PIT = max(valid ann_date, valid f_ann_date, end_date); a latest retained row
  with a missing/non-finite amount clears the older amount (no stale fallback);
* per-table prefix-maximum report-period frontier; one event per (code, PIT);
* strict three-slot YTD TTM: Q4 = own FY Q4 cumulative YTD; Q1-Q3 = current
  cumulative YTD + prior-FY Q4 cumulative YTD - prior-year same-quarter
  cumulative YTD; all three exact slots must be finite; no gap compression;
* ratio rule: denominator finite and strictly > 1e6 CNY; numerator finite and
  signed (no clipping, no zero fill); zero and undisclosed stay distinct;
* PIT maps to the next exchange session exactly once, last visible PIT per
  session wins, NaN clears state; report age > 550 natural days -> NaN;
* final mask = fixed-pool universe AND actual traded mask.

Deviation variants reuse the F001/r13f estimator (12 consecutive complete
training quarters, X=[1,t,3 seasonal dummies], forecast row [1,12,phase
dummies], z=(current-forecast)/training RMSE, RMSE floor 1e-4 in the target's
own units, df=7) applied to the declared level variant's own ratio. The RMSE
floor is expressed in the share's dimensionless units; extreme |z| values for
near-constant own histories are expected and are reported, not clipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from fea.deriv import KEY_W, PERIOD_BASE, PERIOD_W, period_of

Variant = Literal[
    "CF01_op_inflow_sales_collection_share",
    "CF02_op_outflow_tax_payment_share",
    "CF03_investing_outflow_capex_share",
    "CF04_financing_inflow_borrow_share",
    "CF05_financing_outflow_distribution_share",
    "IS01_selling_expense_ratio",
    "IS02_admin_expense_ratio",
    "IS03_financial_expense_ratio",
    "IS04_gross_profit_opex_coverage",
    "CF06_op_inflow_sales_collection_share_deviation",
    "IS05_selling_expense_ratio_deviation",
]
Kind = Literal["ttm_ratio", "ttm_gp_opex_coverage", "seasonal_deviation"]

COMPANY_TYPE: Final = 1
REPORT_TYPE: Final = 1
MIN_DENOMINATOR: Final = 1_000_000.0
MAX_REPORT_AGE_DAYS: Final = 550
WARMUP_DAYS: Final = 1800

WINDOW: Final = 12
FULL_RANK: Final = 5
RESIDUAL_DF: Final = WINDOW - FULL_RANK
RMSE_FLOOR: Final = 1e-4
EVENT_BATCH: Final = 2048


@dataclass(frozen=True)
class VariantDefinition:
    variant: Variant
    dataset: str
    fields: tuple[str, ...]
    kind: Kind
    numerator: str | None = None
    denominator: str | None = None
    shared_cost: str | None = None
    base: Variant | None = None
    unit: str = "dimensionless"
    dependencies: tuple[str, ...] = ("stock_daily",)


DEFINITIONS: Final[dict[Variant, VariantDefinition]] = {
    "CF01_op_inflow_sales_collection_share": VariantDefinition(
        variant="CF01_op_inflow_sales_collection_share",
        dataset="stock_cashflow",
        fields=("c_fr_sale_sg", "c_inf_fr_operate_a"),
        kind="ttm_ratio",
        numerator="c_fr_sale_sg",
        denominator="c_inf_fr_operate_a",
        dependencies=("stock_cashflow", "stock_daily"),
    ),
    "CF02_op_outflow_tax_payment_share": VariantDefinition(
        variant="CF02_op_outflow_tax_payment_share",
        dataset="stock_cashflow",
        fields=("c_paid_for_taxes", "st_cash_out_act"),
        kind="ttm_ratio",
        numerator="c_paid_for_taxes",
        denominator="st_cash_out_act",
        dependencies=("stock_cashflow", "stock_daily"),
    ),
    "CF03_investing_outflow_capex_share": VariantDefinition(
        variant="CF03_investing_outflow_capex_share",
        dataset="stock_cashflow",
        fields=("c_pay_acq_const_fiolta", "stot_out_inv_act"),
        kind="ttm_ratio",
        numerator="c_pay_acq_const_fiolta",
        denominator="stot_out_inv_act",
        dependencies=("stock_cashflow", "stock_daily"),
    ),
    "CF04_financing_inflow_borrow_share": VariantDefinition(
        variant="CF04_financing_inflow_borrow_share",
        dataset="stock_cashflow",
        fields=("c_recp_borrow", "stot_cash_in_fnc_act"),
        kind="ttm_ratio",
        numerator="c_recp_borrow",
        denominator="stot_cash_in_fnc_act",
        dependencies=("stock_cashflow", "stock_daily"),
    ),
    "CF05_financing_outflow_distribution_share": VariantDefinition(
        variant="CF05_financing_outflow_distribution_share",
        dataset="stock_cashflow",
        fields=("c_pay_dist_dpcp_int_exp", "stot_cashout_fnc_act"),
        kind="ttm_ratio",
        numerator="c_pay_dist_dpcp_int_exp",
        denominator="stot_cashout_fnc_act",
        dependencies=("stock_cashflow", "stock_daily"),
    ),
    "IS01_selling_expense_ratio": VariantDefinition(
        variant="IS01_selling_expense_ratio",
        dataset="stock_income",
        fields=("sell_exp", "revenue"),
        kind="ttm_ratio",
        numerator="sell_exp",
        denominator="revenue",
        dependencies=("stock_income", "stock_daily"),
    ),
    "IS02_admin_expense_ratio": VariantDefinition(
        variant="IS02_admin_expense_ratio",
        dataset="stock_income",
        fields=("admin_exp", "revenue"),
        kind="ttm_ratio",
        numerator="admin_exp",
        denominator="revenue",
        dependencies=("stock_income", "stock_daily"),
    ),
    "IS03_financial_expense_ratio": VariantDefinition(
        variant="IS03_financial_expense_ratio",
        dataset="stock_income",
        fields=("fin_exp", "revenue"),
        kind="ttm_ratio",
        numerator="fin_exp",
        denominator="revenue",
        dependencies=("stock_income", "stock_daily"),
    ),
    "IS04_gross_profit_opex_coverage": VariantDefinition(
        variant="IS04_gross_profit_opex_coverage",
        dataset="stock_income",
        fields=("revenue", "oper_cost", "total_cogs"),
        kind="ttm_gp_opex_coverage",
        numerator="revenue",
        denominator="total_cogs",
        shared_cost="oper_cost",
        dependencies=("stock_income", "stock_daily"),
    ),
    "CF06_op_inflow_sales_collection_share_deviation": VariantDefinition(
        variant="CF06_op_inflow_sales_collection_share_deviation",
        dataset="stock_cashflow",
        fields=("c_fr_sale_sg", "c_inf_fr_operate_a"),
        kind="seasonal_deviation",
        base="CF01_op_inflow_sales_collection_share",
        dependencies=("stock_cashflow", "stock_daily"),
    ),
    "IS05_selling_expense_ratio_deviation": VariantDefinition(
        variant="IS05_selling_expense_ratio_deviation",
        dataset="stock_income",
        fields=("sell_exp", "revenue"),
        kind="seasonal_deviation",
        base="IS01_selling_expense_ratio",
        dependencies=("stock_income", "stock_daily"),
    ),
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
    """Exact report-year partitions needed for the 16-quarter lookback."""
    if not ctx.panel.T:
        return (PERIOD_BASE, PERIOD_BASE)
    lo = int(ctx.panel.dates[0]) // 10000
    hi = int(ctx.panel.dates[-1]) // 10000
    return max(PERIOD_BASE, lo - 5), hi


def _report_table(ctx, dataset: str, fields: tuple[str, ...]) -> pd.DataFrame:
    """Project one raw table; standard-filter then whole-row PIT dedup.

    A latest standard row with a missing/non-finite amount stays in the
    vintages and clears the older amount; it is not removed before the
    frontier.
    """
    columns = ["stock_code", "ann_date", "f_ann_date", "end_date",
               "comp_type", "report_type", *fields]
    frame = ctx.dataset(dataset, columns=columns, years=_source_years(ctx))
    out_columns = ["code", "period", "pit", *fields]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=out_columns)

    end = _valid_dates(ctx, frame["end_date"])
    ann = _valid_dates(ctx, frame["ann_date"])
    fann = _valid_dates(ctx, frame["f_ann_date"])
    pit = np.maximum(np.maximum(ann, fann), end)
    comp = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy(dtype=np.float64)
    report = pd.to_numeric(frame["report_type"], errors="coerce").to_numpy(dtype=np.float64)

    keep = np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= (end >= 19900101) & (pit >= end) & (pit > 0)
    keep &= (ann >= 19900101) | (fann >= 19900101)
    keep &= (comp == COMPANY_TYPE) & (report == REPORT_TYPE)
    if ctx.panel.T:
        keep &= pit <= int(ctx.panel.dates[-1])

    codes = ctx.code_index(frame["stock_code"].to_numpy()[keep])
    data = pd.DataFrame({
        "code": codes,
        "period": period_of(end[keep]),
        "pit": pit[keep],
        **{
            field: pd.to_numeric(frame[field], errors="coerce").to_numpy(
                dtype=np.float64
            )[keep]
            for field in fields
        },
    })
    data = data.loc[data.code >= 0]
    return data.drop_duplicates(["code", "period", "pit"], keep="last")


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


def _strict_ttm(vt: dict, code, period, pit, field: str) -> np.ndarray:
    """Strict TTM from exact YTD slots as visible at `pit`; no partial fill."""
    code, period, pit = np.broadcast_arrays(code, period, pit)
    current = _lookup(vt, code, period, pit, field)
    phase = period % 4
    prior_fy_q4 = period - (phase + 1)
    prior_same_q = period - 4
    prior_q4 = _lookup(vt, code, prior_fy_q4, pit, field)
    prior_same = _lookup(vt, code, prior_same_q, pit, field)
    result = np.where(phase == 3, current, current + prior_q4 - prior_same)
    needed = np.isfinite(current) & np.where(
        phase == 3, True, np.isfinite(prior_q4) & np.isfinite(prior_same)
    )
    return np.where(needed & np.isfinite(result), result, np.nan)


def _safe_ratio(numerator, denominator) -> np.ndarray:
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


def _level_at(vt: dict, code, period, pit, variant: Variant) -> np.ndarray:
    """Level value of a ttm_ratio / ttm_gp_opex_coverage variant."""
    spec = DEFINITIONS[variant]
    numerator = _strict_ttm(vt, code, period, pit, spec.numerator)
    denominator = _strict_ttm(vt, code, period, pit, spec.denominator)
    if spec.shared_cost is not None:
        shared = _strict_ttm(vt, code, period, pit, spec.shared_cost)
        numerator = numerator - shared
        denominator = denominator - shared
    return _safe_ratio(numerator, denominator)


def _frontier_events(data: pd.DataFrame) -> pd.DataFrame:
    """One PIT event at the prefix-maximum report period on this frontier."""
    if data.empty:
        return pd.DataFrame(columns=["code", "pit", "period"])
    present = data[["code", "period", "pit"]].drop_duplicates()
    present = present.sort_values(["code", "pit", "period"], kind="stable")
    present["period"] = present.groupby("code", sort=False).period.cummax()
    return present.drop_duplicates(["code", "pit"], keep="last")[
        ["code", "pit", "period"]].reset_index(drop=True)


def _deviation_batch(events: pd.DataFrame, vt: dict,
                     variant: Variant) -> np.ndarray:
    """12-quarter seasonal/trend deviation of the declared base level.

    Training window q-12..q-1 must be twelve consecutive complete quarters at
    the event PIT; the current quarter is the frontier period q. Design matrix
    [1, t, 3 seasonal dummies] with the same 5-parameter rank check, RMSE
    floor 1e-4 in the base level's own units, residual df 7.
    """
    result = np.full(len(events), np.nan, dtype=np.float64)
    if events.empty:
        return result
    base = DEFINITIONS[variant].base
    for lo in range(0, len(events), EVENT_BATCH):
        hi = min(lo + EVENT_BATCH, len(events))
        block = events.iloc[lo:hi]
        code = block.code.to_numpy(dtype=np.int64)
        pit = block.pit.to_numpy(dtype=np.int64)
        q = block.period.to_numpy(dtype=np.int64)
        periods = q[:, None] - np.arange(WINDOW, -1, -1, dtype=np.int64)[None, :]
        code_grid = np.broadcast_to(code[:, None], periods.shape)
        pit_grid = np.broadcast_to(pit[:, None], periods.shape)
        levels = _level_at(vt, code_grid, periods, pit_grid, base)
        complete = np.isfinite(levels).all(axis=1)
        if not complete.any():
            continue
        train = levels[:, :WINDOW]
        current = levels[:, WINDOW]
        for phase in range(4):
            rows = np.flatnonzero(complete & (q % 4 == phase))
            if not len(rows):
                continue
            train_periods = periods[rows[0], :WINDOW]
            t = np.arange(WINDOW, dtype=np.float64)
            X = np.column_stack([
                np.ones(WINDOW, dtype=np.float64), t,
                (train_periods % 4 == 1).astype(np.float64),
                (train_periods % 4 == 2).astype(np.float64),
                (train_periods % 4 == 3).astype(np.float64),
            ])
            if np.linalg.matrix_rank(X) != FULL_RANK:
                continue
            try:
                beta, _, rank, _ = np.linalg.lstsq(X, train[rows].T, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if rank != FULL_RANK:
                continue
            residual = train[rows].T - X @ beta
            rmse = np.sqrt(np.sum(residual * residual, axis=0) / RESIDUAL_DF)
            forecast_row = np.array([1.0, 12.0,
                                     float(phase == 1), float(phase == 2),
                                     float(phase == 3)], dtype=np.float64)
            forecast = forecast_row @ beta
            ok = np.isfinite(rmse) & (rmse >= RMSE_FLOOR)
            ok &= np.isfinite(forecast) & np.isfinite(current[rows])
            # Reject zero/undersized RMSE before division (e.g. all-zero history).
            value = np.full(len(rows), np.nan, dtype=np.float64)
            with np.errstate(over="ignore"):
                np.subtract(current[rows], forecast, out=value, where=ok)
                np.divide(value, rmse, out=value, where=ok)
            ok &= np.isfinite(value)
            result[lo + rows[ok]] = value[ok]
    return result


def _compute_events(events: pd.DataFrame, vt: dict,
                    variant: Variant) -> np.ndarray:
    if events.empty:
        return np.empty(0, dtype=np.float64)
    spec = DEFINITIONS[variant]
    if spec.kind in ("ttm_ratio", "ttm_gp_opex_coverage"):
        return _level_at(
            vt,
            events.code.to_numpy(dtype=np.int64),
            events.period.to_numpy(dtype=np.int64),
            events.pit.to_numpy(dtype=np.int64),
            variant,
        )
    if spec.kind == "seasonal_deviation":
        return _deviation_batch(events, vt, variant)
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
    data = _report_table(ctx, spec.dataset, spec.fields)
    events = _frontier_events(data)
    vt = _sorted_versions(data, spec.fields)
    values = _compute_events(events, vt, variant)
    return _map_daily(ctx, events, values)


def build(parameters: dict):
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be one exact frozen variant dictionary")
    variant = parameters.get("variant")
    if variant not in PARAMETERS or parameters != PARAMETERS[variant]:
        raise ValueError("Only the eleven frozen R15 cashflow/income descriptors are allowed")
    return lambda ctx: _evaluate(ctx, variant)
