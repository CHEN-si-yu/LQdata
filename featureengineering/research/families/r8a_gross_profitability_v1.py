"""R8 A: strict gross-profit TTM / same-quarter preceding-year total assets.
Weak positive prior, not literal Novy-Marx replication or new primitive data.
Versions are next-trading-day effective; latest joint invalid events clear old state.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from fea.deriv import PERIOD_BASE, PERIOD_W, KEY_W, period_of

AMOUNT_FLOOR = 1e6
MAX_REPORT_AGE_DAYS = 550
INCOME_FIELDS = ("revenue", "oper_cost", "n_income_attr_p")
BALANCE_FIELDS = ("total_assets",)
PARAMETERS = {"signal": "gross_profitability", "window": 4, "asset_lag_quarters": 4}

def _report_table(ctx, dataset, fields):
    frame = ctx.dataset(dataset, columns=["stock_code", "ann_date", "f_ann_date", "end_date", "comp_type", *fields])
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["code", "period", "pit", "comp_type", *fields])
    end = ctx.date_col(frame["end_date"])
    ann = ctx.date_col(frame["ann_date"])
    fann = ctx.date_col(frame["f_ann_date"])
    ann = np.where(ann >= 19900101, ann, 0)
    fann = np.where(fann >= 19900101, fann, 0)
    pit = np.maximum(ann, fann)
    keep = np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= (end >= 19900101) & (pit >= end) & (pit > 0)
    data = pd.DataFrame({"code": ctx.code_index(frame["stock_code"].to_numpy()[keep]),
                         "period": period_of(end[keep]), "pit": pit[keep]})
    data["comp_type"] = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy(dtype=np.float64)[keep]
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
            "period": period, **{f: data[f].to_numpy(dtype=np.float64) for f in (*fields, "comp_type")}}

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

def _joint_events(income, balance):
    if income.empty or balance.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    ia = income.groupby(["code", "period"], sort=False).pit.min().rename("income_pit")
    ca = balance.groupby(["code", "period"], sort=False).pit.min().rename("balance_pit")
    common = pd.concat([ia, ca], axis=1, join="inner").reset_index()
    if common.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    common["pit"] = np.maximum(common.income_pit, common.balance_pit).astype(np.int64)
    common = common.sort_values(["code", "pit", "period"], kind="stable")
    common["q"] = common.groupby("code", sort=False).period.cummax()
    common = common.drop_duplicates(["code", "pit"], keep="last")
    events = pd.concat([income[["code", "pit"]], balance[["code", "pit"]]], ignore_index=True)
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
    effective = np.asarray(ctx.next_trading_day(pit), dtype=np.int64)
    keep = effective > pit
    state = ctx.panel.asof(code[keep], effective[keep], np.asarray(values, dtype=np.float64)[keep], dtype=np.float64)
    report_state = ctx.panel.asof(code[keep], effective[keep], report_days[keep], dtype=np.float64)
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


def _profitability(revenue_singles, cost_singles, lag_assets, general):
    revenue = np.sum(revenue_singles, axis=1)
    costs = np.sum(cost_singles, axis=1)
    valid = np.isfinite(revenue_singles).all(axis=1)
    valid &= np.isfinite(cost_singles).all(axis=1)
    valid &= np.isfinite(lag_assets) & (lag_assets > AMOUNT_FLOOR) & general
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw = (revenue - costs) / lag_assets
        margin = (revenue - costs) / revenue
        turnover = revenue / lag_assets
    valid &= np.isfinite(raw)
    raw = np.where(valid, raw, np.nan)
    # A zero-revenue report can have valid raw. M has no defined value there;
    # report the extra M/U common-sample loss rather than silently rejecting raw.
    leg_valid = valid & (revenue != 0) & np.isfinite(margin) & np.isfinite(turnover)
    return raw, np.where(leg_valid, margin, np.nan), np.where(leg_valid, turnover, np.nan)

def _event_components(ctx, include_controls=False):
    fields = INCOME_FIELDS if include_controls else INCOME_FIELDS[:2]
    income = _report_table(ctx, "stock_income", fields)
    balance = _report_table(ctx, "stock_balancesheet", BALANCE_FIELDS)
    events = _joint_events(income, balance)
    if events.empty:
        empty = np.empty(0, dtype=np.float64)
        return events, (empty, empty, empty, empty)
    iv = _sorted_versions(income, fields)
    bv = _sorted_versions(balance, BALANCE_FIELDS)
    current_income_type = _point_window(iv, events, "comp_type", width=1)[:, 0]
    current_balance_type = _point_window(bv, events, "comp_type", width=1)[:, 0]
    general = (current_income_type == 1) & (current_balance_type == 1)
    assets = _point_window(bv, events, "total_assets", width=1, offset=4)[:, 0]
    raw, margin, turnover = _profitability(
        _single_window(iv, events, "revenue", width=4),
        _single_window(iv, events, "oper_cost", width=4), assets, general)
    ni = np.full(len(events), np.nan)
    if include_controls:
        singles = _single_window(iv, events, "n_income_attr_p", width=4)
        with np.errstate(all="ignore"):
            ni = np.sum(singles, axis=1) / assets
        ni = np.where(np.isfinite(raw) & np.isfinite(singles).all(axis=1) & np.isfinite(ni), ni, np.nan)
    return events, (raw, margin, turnover, ni)

def control_components(ctx):
    """M/U follow raw PIT and validity; NI uses the identical lagged-asset base.
    NI or undefined M can add diagnostic missingness, explicitly reported.
    They never tighten the raw candidate's eligibility.
    """
    events, (_, margin, turnover, ni) = _event_components(ctx, include_controls=True)
    return {"M": _map_fresh(ctx, events, margin),
            "U": _map_fresh(ctx, events, turnover),
            "NI_same_assets": _map_fresh(ctx, events, ni)}

def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only the single frozen R8A definition is permitted")
    def compute(ctx):
        events, (raw, _, _, _) = _event_components(ctx)
        return _map_fresh(ctx, events, raw)
    return compute
