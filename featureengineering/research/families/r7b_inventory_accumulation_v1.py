"""R7B frozen sales-adjusted inventory accumulation, direction false.

An operating-imbalance decomposition within accruals, not a wholly new source.
Inventory is measured at cost; revenue also varies with price/margin. Frozen
diagnostics therefore control gross-margin level/change as well as accruals,
turnover and growth. Real zero inventory retained, missing never filled zero.
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
    valid = np.isfinite(grid) & np.isfinite(age) & (age <= STALE_DAYS)
    valid &= np.asarray(ctx.universe, dtype=bool) & np.asarray(ctx.traded(), dtype=bool)
    return np.where(valid, grid, np.nan)


SIGNAL = "sales_adjusted_inventory_accumulation"

def _joint_events(income, balance):
    if income.empty or balance.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    ia = income.groupby(["code", "period"], sort=False).pit.min().rename("income_pit")
    ba = balance.groupby(["code", "period"], sort=False).pit.min().rename("balance_pit")
    common = pd.concat([ia, ba], axis=1, join="inner").reset_index()
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
    pos = np.searchsorted(keys, query, side="right")-1
    safe = np.maximum(pos, 0)
    keep = (pos >= 0) & (source_code[safe] == event_code)
    events = events.loc[keep].copy()
    events["q"] = common.q.to_numpy(dtype=np.int64)[safe[keep]]
    return events.reset_index(drop=True)


def _inventory_accumulation(inv_now, inv_lag, revenue_quarters, assets_lag):
    valid = np.isfinite(inv_now) & np.isfinite(inv_lag) & np.isfinite(assets_lag)
    valid &= (inv_now >= 0) & (inv_lag >= 0) & (assets_lag > AMOUNT_FLOOR)
    valid &= np.isfinite(revenue_quarters).all(axis=1)
    revenue_now = revenue_quarters[:, :4].sum(axis=1)
    revenue_lag = revenue_quarters[:, 4:8].sum(axis=1)
    valid &= (revenue_now > AMOUNT_FLOOR) & (revenue_lag > AMOUNT_FLOOR)
    with np.errstate(all="ignore"):
        value = (inv_now-inv_lag*revenue_now/np.where(valid, revenue_lag, np.nan))/np.where(valid, assets_lag, np.nan)
    return np.where(valid & np.isfinite(value), value, np.nan)

def build(parameters):
    if parameters != {"signal": SIGNAL}:
        raise ValueError("Only the single frozen R7B definition is permitted")
    def compute(ctx):
        income = _table(ctx, "stock_income", ("revenue",))
        balance = _table(ctx, "stock_balancesheet", ("inventories", "total_assets"))
        events = _joint_events(income, balance)
        if events.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        revenue = _single(_versions(income, "revenue"), events, 8)
        code = events.code.to_numpy(dtype=np.int64)
        q = events.q.to_numpy(dtype=np.int64)
        pit = events.pit.to_numpy(dtype=np.int64)
        inv = _versions(balance, "inventories")
        now = _lookup(inv, code, q, pit)
        lag = _lookup(inv, code, q-4, pit)
        assets = _lookup(_versions(balance, "total_assets"), code, q-4, pit)
        return _with_age(ctx, events, _inventory_accumulation(now, lag, revenue, assets))
    return compute
