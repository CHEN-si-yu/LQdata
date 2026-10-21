"""R7 D: operating-profit change divided by preceding-year cash capex.
One frozen weak positive prior, not matched-project return or causal ROI.
Strict single quarters; latest jointly disclosed income/cashflow report frontier.
The G/L decomposition is a required control diagnostic, not extra candidates.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from fea.deriv import PERIOD_BASE, PERIOD_W, KEY_W, period_of

AMOUNT_FLOOR = 1e6
MAX_REPORT_AGE_DAYS = 550
INCOME_FIELDS = ("operate_profit",)
CASH_FIELDS = ("c_pay_acq_const_fiolta", "n_cashflow_act")
PARAMETERS = {"signal": "lagged_capex_profit_conversion", "window": 4, "lag_quarters": 4}

def _report_table(ctx, dataset, fields):
    frame = ctx.dataset(dataset, columns=["stock_code", "ann_date", "f_ann_date", "end_date", *fields])
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["code", "period", "pit", *fields])
    end = ctx.date_col(frame["end_date"])
    ann = ctx.date_col(frame["ann_date"])
    fann = ctx.date_col(frame["f_ann_date"])
    pit = np.maximum(np.maximum(ann, fann), end)
    keep = np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= ((ann >= 19900101) | (fann >= 19900101)) & (pit > 0)
    data = pd.DataFrame({"code": ctx.code_index(frame["stock_code"].to_numpy()[keep]),
                         "period": period_of(end[keep]), "pit": pit[keep]})
    for field in fields:
        data[field] = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)[keep]
    data = data.loc[data.code >= 0]
    # Whole report row wins: a newly missing component must not reuse an older value.
    return data.drop_duplicates(["code", "period", "pit"], keep="last")

def _sorted_versions(data, fields):
    data = data.sort_values(["code", "period", "pit"], kind="stable")
    code = data.code.to_numpy(dtype=np.int64)
    period = data.period.to_numpy(dtype=np.int64)
    pit = data.pit.to_numpy(dtype=np.int64)
    return {"keys": (code * PERIOD_W + period) * KEY_W + pit, "code": code,
            "period": period, **{f: data[f].to_numpy(dtype=np.float64) for f in fields}}

def _lookup(vt, code, period, pit, field):
    if len(vt["keys"]) == 0:
        return np.full(len(code), np.nan, dtype=np.float64)
    query = (code * PERIOD_W + period) * KEY_W + pit
    pos = np.searchsorted(vt["keys"], query, side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (vt["code"][safe] == code) & (vt["period"][safe] == period)
    return np.where(ok, vt[field][safe], np.nan)

def _point_window(vt, events, field, width=5, offset=0):
    code = events.code.to_numpy(dtype=np.int64)
    q = events.q.to_numpy(dtype=np.int64)
    pit = events.pit.to_numpy(dtype=np.int64)
    return np.column_stack([_lookup(vt, code, q-offset-j, pit, field) for j in range(width)])

def _joint_events(income, cashflow):
    if income.empty or cashflow.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    ia = income.groupby(["code", "period"], sort=False).pit.min().rename("income_pit")
    ca = cashflow.groupby(["code", "period"], sort=False).pit.min().rename("cash_pit")
    common = pd.concat([ia, ca], axis=1, join="inner").reset_index()
    if common.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    common["pit"] = np.maximum(common.income_pit, common.cash_pit).astype(np.int64)
    common = common.sort_values(["code", "pit", "period"], kind="stable")
    common["q"] = common.groupby("code", sort=False).period.cummax()
    common = common.drop_duplicates(["code", "pit"], keep="last")
    events = pd.concat([income[["code", "pit"]], cashflow[["code", "pit"]]], ignore_index=True)
    events = events.drop_duplicates().sort_values(["code", "pit"], kind="stable")
    source_code = common.code.to_numpy(dtype=np.int64)
    keys = source_code * KEY_W + common.pit.to_numpy(dtype=np.int64)
    event_code = events.code.to_numpy(dtype=np.int64)
    query = event_code * KEY_W + events.pit.to_numpy(dtype=np.int64)
    pos = np.searchsorted(keys, query, side="right") - 1
    safe = np.maximum(pos, 0)
    keep = (pos >= 0) & (source_code[safe] == event_code)
    events = events.loc[keep].copy()
    events["q"] = common.q.to_numpy(dtype=np.int64)[safe[keep]]
    return events.reset_index(drop=True)


def _epoch_days(yyyymmdd):
    parsed = pd.to_datetime(np.asarray(yyyymmdd).astype(str), format="%Y%m%d", errors="raise")
    return parsed.to_numpy(dtype="datetime64[D]").astype(np.int64)

def _map_fresh(ctx, events, values):
    if len(events) == 0:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    code = events.code.to_numpy(dtype=np.int64)
    pit = events.pit.to_numpy(dtype=np.int64)
    q = events.q.to_numpy(dtype=np.int64)
    end = (PERIOD_BASE + q // 4) * 10000 + np.array([331,630,930,1231])[q % 4]
    report_days = _epoch_days(end).astype(np.float64)
    state = ctx.panel.asof(code, pit, np.asarray(values, dtype=np.float64), dtype=np.float64)
    report_state = ctx.panel.asof(code, pit, report_days, dtype=np.float64)
    age = _epoch_days(ctx.panel.dates)[:, None] - report_state
    # Calendar age advances even when no new publication event arrives.
    return np.where((age >= 0) & (age <= MAX_REPORT_AGE_DAYS) & np.isfinite(state), state, np.nan)



def _single_window(vt, events, field, width, offset=0):
    cumulative = _point_window(vt, events, field, width=width+1, offset=offset)
    q = events.q.to_numpy(dtype=np.int64)
    first_quarter = ((q[:, None] - offset - np.arange(width)[None, :]) % 4) == 0
    # Q1 does not require previous fiscal-year Q4; other quarters require both.
    return np.where(first_quarter, cumulative[:, :-1],
                    cumulative[:, :-1] - cumulative[:, 1:])

def _conversion(op_singles, capex_singles, ocf_singles):
    """Inputs ordered newest first; all windows retain their calendar positions."""
    current = np.sum(op_singles[:, :4], axis=1)
    previous = np.sum(op_singles[:, 4:8], axis=1)
    capex = np.sum(capex_singles, axis=1)
    cash = np.sum(ocf_singles, axis=1)
    valid = np.isfinite(op_singles).all(axis=1)
    valid &= np.isfinite(capex_singles).all(axis=1) & (capex_singles >= 0).all(axis=1)
    valid &= np.isfinite(ocf_singles).all(axis=1)
    valid &= (current > AMOUNT_FLOOR) & (cash > 0) & (capex > AMOUNT_FLOOR)
    delta = current - previous
    base = np.maximum(np.abs(previous), AMOUNT_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw = delta / capex
        growth_leg = delta / base
        low_capex_leg = base / capex
    # Components share the candidate's eligibility, including negative delta.
    valid &= np.isfinite(raw) & np.isfinite(growth_leg) & np.isfinite(low_capex_leg)
    return tuple(np.where(valid, x, np.nan) for x in (raw, growth_leg, low_capex_leg))

def _event_components(ctx):
    income = _report_table(ctx, "stock_income", INCOME_FIELDS)
    cash = _report_table(ctx, "stock_cashflow", CASH_FIELDS)
    events = _joint_events(income, cash)
    if events.empty:
        empty = np.empty(0, dtype=np.float64)
        return events, (empty, empty, empty)
    iv = _sorted_versions(income, INCOME_FIELDS)
    cv = _sorted_versions(cash, CASH_FIELDS)
    components = _conversion(
        _single_window(iv, events, "operate_profit", width=8),
        _single_window(cv, events, "c_pay_acq_const_fiolta", width=4, offset=4),
        _single_window(cv, events, "n_cashflow_act", width=4))
    return events, components

def control_components(ctx):
    """Return G and L panel arrays on the identical PIT frontier and valid mask.
    B=max(abs(OP_previous),1e6); G=delta_OP/B; L=B/K_previous.
    This floor is only a diagnostic-control normalization, never the raw formula.
    """
    events, (_, growth, low_capex) = _event_components(ctx)
    return {"G": _map_fresh(ctx, events, growth), "L": _map_fresh(ctx, events, low_capex)}

def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only the single frozen R7D definition is permitted")
    def compute(ctx):
        events, (raw, _, _) = _event_components(ctx)
        return _map_fresh(ctx, events, raw)
    return compute
