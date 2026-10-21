"""R5A frozen customer-cash promptness: 16 seasonal innovations, higher is better.

All operations are at publication events, not daily independently filled quarter
grids. Anchor Q is the latest actually common reported period as of each event.
The frontier is built from max(first income PIT, first cashflow PIT) per period.
Every later revision event from either table refetches the complete 21-quarter
sales and 20-quarter receipts windows under its own as-of PIT. Missing windows
are NaN events and deliberately replace the previous valid state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fea.deriv import PERIOD_W, KEY_W, period_of

WINDOW = 16
REL_VAR_FLOOR = 1e-12
AMOUNT_FLOOR = 1e6

def _quarter_table(ctx, dataset, field):
    frame = ctx.dataset(dataset, columns=["stock_code", "ann_date", "f_ann_date", "end_date", field])
    if frame is None or frame.empty or field not in frame:
        return pd.DataFrame(columns=["code", "period", "pit", "val"])
    end = ctx.date_col(frame["end_date"])
    ann = ctx.date_col(frame["ann_date"])
    fann = ctx.date_col(frame["f_ann_date"])
    pit = np.maximum(np.maximum(ann, fann), end)
    keep = np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= ((ann >= 19900101) | (fann >= 19900101)) & (pit > 0)
    codes = ctx.code_index(frame["stock_code"].to_numpy()[keep])
    vals = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)[keep]
    periods = period_of(end[keep])
    data = pd.DataFrame({"code": codes, "period": periods, "pit": pit[keep], "val": vals})
    data = data.loc[data.code >= 0]
    return data.drop_duplicates(["code", "period", "pit"], keep="last")

def _sorted_versions(data):
    data = data.sort_values(["code", "period", "pit"], kind="stable")
    code = data.code.to_numpy(dtype=np.int64)
    period = data.period.to_numpy(dtype=np.int64)
    pit = data.pit.to_numpy(dtype=np.int64)
    return {"keys": (code * PERIOD_W + period) * KEY_W + pit, "code": code,
            "period": period, "val": data.val.to_numpy(dtype=np.float64)}

def _lookup(vt, code, period, pit):
    if len(vt["keys"]) == 0:
        return np.full(len(code), np.nan, dtype=np.float64)
    query = (code * PERIOD_W + period) * KEY_W + pit
    pos = np.searchsorted(vt["keys"], query, side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (vt["code"][safe] == code) & (vt["period"][safe] == period)
    return np.where(ok, vt["val"][safe], np.nan)

def _single_matrix(vt, events, width):
    code = events.code.to_numpy(dtype=np.int64)
    q = events.q.to_numpy(dtype=np.int64)
    pit = events.pit.to_numpy(dtype=np.int64)
    cumulative = np.column_stack([_lookup(vt, code, q - j, pit) for j in range(width + 1)])
    offsets = np.arange(width, dtype=np.int64)
    q1 = ((q[:, None] - offsets[None, :]) % 4) == 0
    # Q1 is cumulative itself; other quarters require the preceding cumulative report.
    values = np.where(q1, cumulative[:, :-1], cumulative[:, :-1] - cumulative[:, 1:])
    return np.where(np.isfinite(values), values, np.nan)

def _map(ctx, events, values):
    if len(events) == 0:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    return ctx.panel.asof(events.code.to_numpy(dtype=np.int64),
                          events.pit.to_numpy(dtype=np.int64),
                          np.asarray(values, dtype=np.float64), dtype=np.float64)

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

def _partial_corr(x, y, z):
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1) & np.isfinite(z).all(axis=1)
    def centered_scaled(a):
        scale = np.mean(np.abs(a), axis=1, keepdims=True)
        scale = np.where(scale > 0, scale, np.nan)
        a = a / scale
        return a - np.mean(a, axis=1, keepdims=True)
    x, y, z = centered_scaled(x), centered_scaled(y), centered_scaled(z)
    zz = np.mean(z * z, axis=1)
    valid &= zz > REL_VAR_FLOOR
    safe_zz = np.where(valid, zz, np.nan)
    x = x - (np.mean(x * z, axis=1) / safe_zz)[:, None] * z
    y = y - (np.mean(y * z, axis=1) / safe_zz)[:, None] * z
    xx, yy = np.mean(x * x, axis=1), np.mean(y * y, axis=1)
    valid &= (xx > REL_VAR_FLOOR) & (yy > REL_VAR_FLOOR)
    denominator = np.sqrt(np.where(valid, xx * yy, np.nan))
    # Clip only floating-point correlation roundoff; this is not economic winsorization.
    result = np.clip(np.mean(x * y, axis=1) / denominator, -1.0, 1.0)
    return np.where(valid & np.isfinite(result), result, np.nan)

def build(parameters):
    if parameters != {"signal": "cash_collection_promptness", "window": 16}:
        raise ValueError("Only the single frozen R5A definition is permitted")
    def compute(ctx):
        income = _quarter_table(ctx, "stock_income", "revenue")
        cashflow = _quarter_table(ctx, "stock_cashflow", "c_fr_sale_sg")
        events = _joint_events(income, cashflow)
        if events.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        sales = _single_matrix(_sorted_versions(income), events, 21)
        cash = _single_matrix(_sorted_versions(cashflow), events, 20)
        valid = np.isfinite(sales).all(axis=1) & np.isfinite(cash).all(axis=1)
        valid &= (sales > 0).all(axis=1) & (cash >= 0).all(axis=1)
        valid &= np.mean(np.abs(sales), axis=1) > AMOUNT_FLOOR
        x = sales[:, :16] - sales[:, 4:20]
        y = cash[:, :16] - cash[:, 4:20]
        z = sales[:, 1:17] - sales[:, 5:21]
        values = _partial_corr(x, y, z)
        return _map(ctx, events, np.where(valid, values, np.nan))
    return compute
