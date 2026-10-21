"""Standalone, unregistered R13 financial green-card descriptors.

The five frozen variants are source-row descriptors only. They carry no return
direction or alpha claim. All required accounting fields for a variant come
from one raw financial table, so each value is built on that table's own
standard-report frontier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


COMPANY_TYPE = 1
REPORT_TYPE = 1
MIN_DENOMINATOR = 1_000_000.0
MAX_REPORT_AGE_DAYS = 550
WARMUP_DAYS = 1800
PERIOD_BASE = 1990
PERIOD_W = 100_000
KEY_W = 100_000_000

FROZEN = {
    "total_profit_revenue_ttm_v2": {
        "candidate_name": "r13gc_financial_quality_total_profit_revenue_ttm_v2",
        "dataset": "stock_income",
        "fields": ("total_profit", "revenue"),
        "kind": "ttm_ratio",
        "numerator": "total_profit",
        "denominator": "revenue",
        "unit": "dimensionless",
    },
    "income_tax_revenue_ttm_v2": {
        "candidate_name": "r13gc_financial_quality_income_tax_revenue_ttm_v2",
        "dataset": "stock_income",
        "fields": ("income_tax", "revenue"),
        "kind": "ttm_ratio",
        "numerator": "income_tax",
        "denominator": "revenue",
        "unit": "dimensionless",
    },
    "non_oper_exp_revenue_ttm_v2": {
        "candidate_name": "r13gc_financial_quality_non_oper_exp_revenue_ttm_v2",
        "dataset": "stock_income",
        "fields": ("non_oper_exp", "revenue"),
        "kind": "ttm_ratio",
        "numerator": "non_oper_exp",
        "denominator": "revenue",
        "unit": "dimensionless",
    },
    "cash_tax_paid_asinh_ttm_v2": {
        "candidate_name": "r13gc_financial_quality_cash_tax_paid_asinh_ttm_v2",
        "dataset": "stock_cashflow",
        "fields": ("c_paid_for_taxes",),
        "kind": "ttm_asinh",
        "numerator": "c_paid_for_taxes",
        "unit": "asinh_yuan",
    },
    "minority_interest_assets_point_v2": {
        "candidate_name": "r13gc_financial_quality_minority_interest_assets_point_v2",
        "dataset": "stock_balancesheet",
        "fields": ("minority_int", "total_assets"),
        "kind": "point_ratio",
        "numerator": "minority_int",
        "denominator": "total_assets",
        "unit": "dimensionless",
    },
}

PARAMETERS = {
    variant: {"variant": variant, "definition_version": 2}
    for variant in FROZEN
}


def _empty_table(fields):
    return pd.DataFrame(columns=["code", "period", "pit", *fields])


def _valid_dates(ctx, values):
    raw = np.asarray(ctx.date_col(values), dtype=np.int64)
    parsed = pd.to_datetime(raw.astype(str), format="%Y%m%d", errors="coerce")
    valid = (raw >= 19900101) & ~np.asarray(pd.isna(parsed))
    return np.where(valid, raw, 0)


def _period_of(end_date):
    end_date = np.asarray(end_date, dtype=np.int64)
    year = end_date // 10000
    month = (end_date // 100) % 100
    return ((year - PERIOD_BASE) * 4 + (month // 3 - 1)).astype(np.int64)


def _period_end(period):
    period = np.asarray(period, dtype=np.int64)
    month_day = np.array([331, 630, 930, 1231], dtype=np.int64)[period % 4]
    year = PERIOD_BASE + period // 4
    return year * 10000 + month_day


def _yyyymmdd_to_days(values):
    values = np.asarray(values, dtype=np.int64)
    return pd.to_datetime(values.astype(str), format="%Y%m%d", errors="raise") \
        .to_numpy(dtype="datetime64[D]").astype(np.int64)


def _report_table(ctx, dataset, fields):
    """Read a projected raw source; type-filter before whole-row PIT dedup.

    A row with a missing or nonfinite amount is retained as a state-clearing
    revision. It is never removed before the report frontier is formed.
    """
    fields = tuple(fields)
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
    comp = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy(
        dtype=np.float64)
    report = pd.to_numeric(frame["report_type"], errors="coerce").to_numpy(
        dtype=np.float64)

    keep = np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= (end >= 19900101) & (pit >= end) & (pit > 0)
    keep &= (ann >= 19900101) | (fann >= 19900101)
    # Freeze the report scope before de-duplication and before frontier creation.
    keep &= (comp == COMPANY_TYPE) & (report == REPORT_TYPE)
    if ctx.panel.T:
        keep &= pit <= int(ctx.panel.dates[-1])
    else:
        keep &= False

    codes = ctx.code_index(frame["stock_code"].to_numpy()[keep])
    data = pd.DataFrame({
        "code": codes,
        "period": _period_of(end[keep]),
        "pit": pit[keep],
        **{
            field: pd.to_numeric(frame[field], errors="coerce").to_numpy(
                dtype=np.float64
            )[keep]
            for field in fields
        },
    })
    data = data.loc[data.code >= 0].copy()
    # Keep the physical last complete source row at a tied code/period/PIT.
    # Missing amounts remain on that row and therefore clear any old value.
    return data.drop_duplicates(["code", "period", "pit"], keep="last")


def _frontier_events(data):
    """One event per source PIT, carrying the prefix-max report period."""
    if data.empty:
        return pd.DataFrame(columns=["code", "pit", "period"])
    present = data[["code", "period", "pit"]].drop_duplicates()
    present = present.sort_values(["code", "pit", "period"], kind="stable")
    present["frontier"] = present.groupby("code", sort=False).period.cummax()
    present = present.drop_duplicates(["code", "pit"], keep="last")
    return present[["code", "pit", "frontier"]].rename(
        columns={"frontier": "period"}
    ).reset_index(drop=True)


def _sorted_versions(data, fields):
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


def _lookup(vt, code, period, pit, field):
    code, period, pit = np.broadcast_arrays(code, period, pit)
    shape = code.shape
    if not len(vt["keys"]):
        return np.full(shape, np.nan, dtype=np.float64)
    query = (code * PERIOD_W + period) * KEY_W + pit
    pos = np.searchsorted(vt["keys"], query, side="right") - 1
    safe = np.maximum(pos, 0)
    valid = (pos >= 0)
    valid &= (vt["code"][safe] == code)
    valid &= (vt["period"][safe] == period)
    values = vt[field][safe]
    # The whole-row latest version wins even when its amount is missing or inf.
    return np.where(valid, values, np.nan).astype(np.float64, copy=False)


def _strict_ttm(vt, code, period, pit, field):
    """Three-row TTM; exact quarters only, no fallback or partial windows."""
    code, period, pit = np.broadcast_arrays(code, period, pit)
    own = _lookup(vt, code, period, pit, field)
    q4 = period % 4 == 3
    previous_fy_q4 = 4 * (period // 4) - 1
    previous_same_quarter = period - 4
    fy = _lookup(vt, code, previous_fy_q4, pit, field)
    prior_ytd = _lookup(vt, code, previous_same_quarter, pit, field)
    values = np.where(q4, own, own + fy - prior_ytd)
    valid = np.isfinite(own)
    valid &= q4 | (np.isfinite(fy) & np.isfinite(prior_ytd))
    valid &= np.isfinite(values)
    return np.where(valid, values, np.nan).astype(np.float64, copy=False)


def _safe_positive_ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator)
    valid &= denominator >= MIN_DENOMINATOR
    np.divide(numerator, denominator, out=out, where=valid)
    out[~np.isfinite(out)] = np.nan
    return out


def _value_at(vt, code, period, pit, variant):
    spec = FROZEN[variant]
    if spec["kind"] == "ttm_ratio":
        numerator = _strict_ttm(vt, code, period, pit, spec["numerator"])
        denominator = _strict_ttm(vt, code, period, pit, spec["denominator"])
        return _safe_positive_ratio(numerator, denominator)
    if spec["kind"] == "ttm_asinh":
        amount = _strict_ttm(vt, code, period, pit, spec["numerator"])
        return np.where(np.isfinite(amount), np.arcsinh(amount), np.nan)
    if spec["kind"] == "point_ratio":
        numerator = _lookup(vt, code, period, pit, spec["numerator"])
        denominator = _lookup(vt, code, period, pit, spec["denominator"])
        return _safe_positive_ratio(numerator, denominator)
    raise AssertionError(f"Unknown frozen kind: {spec['kind']}")


def _compute_events(events, vt, variant):
    if events.empty:
        return np.empty(0, dtype=np.float64)
    code = events.code.to_numpy(dtype=np.int64)
    period = events.period.to_numpy(dtype=np.int64)
    pit = events.pit.to_numpy(dtype=np.int64)
    return _value_at(vt, code, period, pit, variant)


def _map_daily(ctx, events, values):
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
    # Multiple weekend/holiday announcements can map to one session; latest PIT
    # is the state known at that session, including a NaN revision.
    mapped = mapped.sort_values(
        ["code", "effective", "pit"], kind="stable"
    ).drop_duplicates(["code", "effective"], keep="last")
    code = mapped.code.to_numpy(dtype=np.int64)
    effective = mapped.effective.to_numpy(dtype=np.int64)
    value = mapped.value.to_numpy(dtype=np.float64)
    report_end_days = _yyyymmdd_to_days(
        _period_end(mapped.period.to_numpy(dtype=np.int64))
    )
    state = ctx.panel.asof(code, effective, value, dtype=np.float64)
    end_state = ctx.panel.asof(
        code, effective, report_end_days.astype(np.float64), dtype=np.float64
    )
    panel_days = _yyyymmdd_to_days(ctx.panel.dates)
    age = panel_days[:, None] - end_state
    valid = np.isfinite(state) & (age >= 0) & (age <= MAX_REPORT_AGE_DAYS)
    valid &= np.asarray(ctx.universe, dtype=bool)
    valid &= np.asarray(ctx.traded(), dtype=bool)
    return np.where(valid, state, np.nan)


def _evaluate(ctx, variant):
    spec = FROZEN[variant]
    data = _report_table(ctx, spec["dataset"], spec["fields"])
    events = _frontier_events(data)
    values = _compute_events(
        events, _sorted_versions(data, spec["fields"]), variant
    )
    return _map_daily(ctx, events, values)


def build(parameters):
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be the exact frozen variant dictionary")
    variant = parameters.get("variant")
    if variant not in PARAMETERS or parameters != PARAMETERS[variant]:
        raise ValueError("Only one of the five exact R13 financial GC v2 specs is allowed")
    return lambda ctx: _evaluate(ctx, variant)
