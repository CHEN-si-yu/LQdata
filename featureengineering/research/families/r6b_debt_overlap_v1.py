"""R6B frozen borrowing/repayment overlap divided by average reported debt.

min(strict four-quarter borrowing cash, strict four-quarter principal repayment
cash) / mean(Debt_Q,Debt_(Q-4)). This is a cash-amount turnover proxy, not observed
matched refinancing transactions. Both reports share the latest common PIT
period; all later revisions refetch every component under the same event date.
"""
from __future__ import annotations

import datetime as _dt
import numpy as np
import pandas as pd
from fea.deriv import PERIOD_W, KEY_W, PERIOD_BASE, period_of

AMOUNT_FLOOR = 1e6
STALE_DAYS = 550

def _table(ctx, dataset, fields):
    frame = ctx.dataset(dataset, columns=["stock_code", "ann_date", "f_ann_date", "end_date", *fields])
    columns = ["code", "period", "pit", *fields]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    end = ctx.date_col(frame["end_date"]).astype(np.int64)
    ann = ctx.date_col(frame["ann_date"]).astype(np.int64)
    fann = ctx.date_col(frame["f_ann_date"]).astype(np.int64)
    pit = np.maximum(np.maximum(ann, fann), end)
    keep = np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= ((ann >= 19900101) | (fann >= 19900101))
    keep &= (pit > 0) & (pit <= int(ctx.panel.dates[-1]))
    code = ctx.code_index(frame["stock_code"].to_numpy()[keep])
    data = {"code": code, "period": period_of(end[keep]), "pit": pit[keep]}
    for field in fields:
        vals = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)[keep]
        data[field] = np.where(np.isfinite(vals), vals, np.nan)
    out = pd.DataFrame(data)
    return out.loc[out.code >= 0].drop_duplicates(["code", "period", "pit"], keep="last")

def _versions(data, field):
    data = data.sort_values(["code", "period", "pit"], kind="stable")
    code = data.code.to_numpy(dtype=np.int64)
    period = data.period.to_numpy(dtype=np.int64)
    pit = data.pit.to_numpy(dtype=np.int64)
    return {"keys": (code * PERIOD_W + period) * KEY_W + pit,
            "code": code, "period": period, "val": data[field].to_numpy(dtype=np.float64)}

def _lookup(vt, code, period, pit):
    if len(vt["keys"]) == 0:
        return np.full(len(code), np.nan, dtype=np.float64)
    query = (code * PERIOD_W + period) * KEY_W + pit
    pos = np.searchsorted(vt["keys"], query, side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (vt["code"][safe] == code) & (vt["period"][safe] == period)
    return np.where(ok, vt["val"][safe], np.nan)

def _single(vt, events, width):
    code = events.code.to_numpy(dtype=np.int64)
    q = events.q.to_numpy(dtype=np.int64)
    pit = events.pit.to_numpy(dtype=np.int64)
    cum = np.column_stack([_lookup(vt, code, q-j, pit) for j in range(width+1)])
    q1 = ((q[:, None] - np.arange(width)[None, :]) % 4) == 0
    result = np.where(q1, cum[:, :-1], cum[:, :-1] - cum[:, 1:])
    return np.where(np.isfinite(result), result, np.nan)

def _map(ctx, events, values):
    if events.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    return ctx.panel.asof(events.code.to_numpy(dtype=np.int64),
                          events.pit.to_numpy(dtype=np.int64),
                          np.asarray(values, dtype=np.float64), dtype=np.float64)

def _ordinals(days):
    return np.asarray([_dt.date(int(d)//10000, (int(d)//100)%100, int(d)%100).toordinal()
                       for d in np.asarray(days).ravel()], dtype=np.int64)

def _with_age(ctx, events, values):
    # Source report-period end, not latest revision date, determines freshness.
    q = events.q.to_numpy(dtype=np.int64)
    ends = (PERIOD_BASE + q//4)*10000 + np.asarray([331, 630, 930, 1231])[q % 4]
    end_ord = _map(ctx, events, _ordinals(ends))
    grid = _map(ctx, events, values)
    age = _ordinals(ctx.panel.dates)[:, None] - end_ord
    return np.where(np.isfinite(grid) & np.isfinite(age) & (age <= STALE_DAYS), grid, np.nan)

CASH_FIELDS = ("c_recp_borrow", "c_prepay_amt_borr")
DEBT_FIELDS = ("st_borr", "non_cur_liab_due_1y", "lt_borr", "bond_payable")

def _joint_events(cash, balance):
    if cash.empty or balance.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    ca = cash.groupby(["code", "period"], sort=False).pit.min().rename("cash_pit")
    ba = balance.groupby(["code", "period"], sort=False).pit.min().rename("balance_pit")
    common = pd.concat([ca, ba], axis=1, join="inner").reset_index()
    if common.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    common["pit"] = np.maximum(common.cash_pit, common.balance_pit).astype(np.int64)
    common = common.sort_values(["code", "pit", "period"], kind="stable")
    common["q"] = common.groupby("code", sort=False).period.cummax()
    common = common.drop_duplicates(["code", "pit"], keep="last")
    events = pd.concat([cash[["code", "pit"]], balance[["code", "pit"]]], ignore_index=True)
    events = events.drop_duplicates().sort_values(["code", "pit"], kind="stable")
    source_code = common.code.to_numpy(dtype=np.int64)
    keys = source_code * KEY_W + common.pit.to_numpy(dtype=np.int64)
    event_code = events.code.to_numpy(dtype=np.int64)
    query = event_code * KEY_W + events.pit.to_numpy(dtype=np.int64)
    pos = np.searchsorted(keys, query, side="right")-1
    safe = np.maximum(pos, 0)
    keep = (pos >= 0) & (source_code[safe] == event_code)
    events = events.loc[keep].copy()
    events["q"] = common.q.to_numpy(dtype=np.int64)[safe[keep]]
    return events.reset_index(drop=True)

def _overlap(borrow_quarters, repayment_quarters, debt_now, debt_lag):
    valid = np.isfinite(borrow_quarters).all(axis=1) & np.isfinite(repayment_quarters).all(axis=1)
    valid &= np.isfinite(debt_now).all(axis=1) & np.isfinite(debt_lag).all(axis=1)
    valid &= (debt_now >= 0).all(axis=1) & (debt_lag >= 0).all(axis=1)
    borrow = borrow_quarters.sum(axis=1)
    repay = repayment_quarters.sum(axis=1)
    debt_avg = (debt_now.sum(axis=1)+debt_lag.sum(axis=1))/2
    valid &= (borrow >= 0) & (repay >= 0) & (debt_avg > AMOUNT_FLOOR)
    with np.errstate(all="ignore"):
        value = np.minimum(borrow, repay) / np.where(valid, debt_avg, np.nan)
    return np.where(valid & np.isfinite(value), value, np.nan)

def build(parameters):
    if parameters != {"signal": "borrowing_repayment_overlap_turnover", "quarters": 4}:
        raise ValueError("Only the single frozen R6B definition is permitted")
    def compute(ctx):
        cash = _table(ctx, "stock_cashflow", CASH_FIELDS)
        balance = _table(ctx, "stock_balancesheet", DEBT_FIELDS)
        events = _joint_events(cash, balance)
        if events.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        borrow = _single(_versions(cash, CASH_FIELDS[0]), events, 4)
        repay = _single(_versions(cash, CASH_FIELDS[1]), events, 4)
        code = events.code.to_numpy(dtype=np.int64)
        q = events.q.to_numpy(dtype=np.int64)
        pit = events.pit.to_numpy(dtype=np.int64)
        now = np.column_stack([_lookup(_versions(balance, f), code, q, pit) for f in DEBT_FIELDS])
        lag = np.column_stack([_lookup(_versions(balance, f), code, q-4, pit) for f in DEBT_FIELDS])
        values = _overlap(borrow, repay, now, lag)
        return _with_age(ctx, events, values)
    return compute

