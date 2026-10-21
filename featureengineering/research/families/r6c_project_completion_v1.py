"""Round6 C: cash-supported project completion accounting co-movement proxy.

Single preregistered candidate; direction true is a mechanism prior, unvalidated.
Two reports share the latest actually jointly disclosed Q. All report versions
are re-read as of each publication event; invalid new events overwrite old state.
Strict four-single-quarter operating cashflow and daily 550-calendar-day expiry.
Net fixed assets and CIP movements cannot identify actual transfer transactions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fea.deriv import PERIOD_BASE, PERIOD_W, KEY_W, period_of

WINDOW = 4
AMOUNT_FLOOR = 1e6
MAX_REPORT_AGE_DAYS = 550

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

def _point_window(vt, events, field, width=5):
    code = events.code.to_numpy(dtype=np.int64)
    q = events.q.to_numpy(dtype=np.int64)
    pit = events.pit.to_numpy(dtype=np.int64)
    return np.column_stack([_lookup(vt, code, q-j, pit, field) for j in range(width)])

def _strict_cashflow(vt, events):
    cumulative = _point_window(vt, events, "n_cashflow_act", width=5)
    q = events.q.to_numpy(dtype=np.int64)
    q1 = ((q[:, None] - np.arange(WINDOW)[None, :]) % 4) == 0
    singles = np.where(q1, cumulative[:, :-1], cumulative[:, :-1] - cumulative[:, 1:])
    # Finite continuous single quarters are mandatory; no three-row TTM fallback.
    valid = np.isfinite(singles).all(axis=1)
    return np.where(valid, np.sum(singles, axis=1), np.nan)

def _joint_events(balance, cashflow):
    if balance.empty or cashflow.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    ia = balance.groupby(["code", "period"], sort=False).pit.min().rename("balance_pit")
    ca = cashflow.groupby(["code", "period"], sort=False).pit.min().rename("cash_pit")
    common = pd.concat([ia, ca], axis=1, join="inner").reset_index()
    if common.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    common["pit"] = np.maximum(common.balance_pit, common.cash_pit).astype(np.int64)
    common = common.sort_values(["code", "pit", "period"], kind="stable")
    common["q"] = common.groupby("code", sort=False).period.cummax()
    common = common.drop_duplicates(["code", "pit"], keep="last")
    events = pd.concat([balance[["code", "pit"]], cashflow[["code", "pit"]]], ignore_index=True)
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


BALANCE_FIELDS = ("cip_total", "fix_assets")

def _completion(cip, fixed, cash_ttm):
    valid = np.isfinite(cip).all(axis=1) & np.isfinite(fixed).all(axis=1)
    valid &= (cip >= 0).all(axis=1) & (fixed >= 0).all(axis=1)
    denominator = np.mean(cip[:, 1:], axis=1)
    valid &= (denominator > AMOUNT_FLOOR) & np.isfinite(cash_ttm) & (cash_ttm > 0)
    matched = np.minimum(np.maximum(cip[:, 1:] - cip[:, :-1], 0),
                         np.maximum(fixed[:, :-1] - fixed[:, 1:], 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.sum(matched, axis=1) / denominator
    return np.where(valid & np.isfinite(raw), raw, np.nan)

def build(parameters):
    if parameters != {"signal": "cash_supported_project_completion", "window": 4}:
        raise ValueError("Only the single frozen R6C definition is permitted")
    def compute(ctx):
        balance = _report_table(ctx, "stock_balancesheet", BALANCE_FIELDS)
        cash = _report_table(ctx, "stock_cashflow", ("n_cashflow_act",))
        events = _joint_events(balance, cash)
        if events.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        bv = _sorted_versions(balance, BALANCE_FIELDS)
        cv = _sorted_versions(cash, ("n_cashflow_act",))
        cip = _point_window(bv, events, "cip_total")
        fixed = _point_window(bv, events, "fix_assets")
        values = _completion(cip, fixed, _strict_cashflow(cv, events))
        return _map_fresh(ctx, events, values)
    return compute
