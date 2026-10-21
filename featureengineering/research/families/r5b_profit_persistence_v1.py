"""R5B frozen detrended-profit persistence, weak positive quality prior.

Persistence can also persistently describe weak or loss-making earnings. This
definition deliberately does not multiply by latest shock or positive-profit
state. Full-window net profit is projected on intercept, deterministic trend and
fiscal-quarter dummies; then measure residual lag-one serial correlation.
Each historical version is available only at max(ann_date,f_ann_date,end_date).
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

def _income_events(data):
    if data.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    data = data.sort_values(["code", "pit", "period"], kind="stable")
    data = data.assign(q=data.groupby("code", sort=False).period.cummax())
    return data.drop_duplicates(["code", "pit"], keep="last")[["code", "pit", "q"]].reset_index(drop=True)

def _residual_persistence(profit):
    valid = np.isfinite(profit).all(axis=1)
    scale = np.mean(np.abs(profit), axis=1)
    valid &= scale > AMOUNT_FLOOR
    normalized = profit / np.where(valid, scale, np.nan)[:, None]
    j = np.arange(WINDOW, dtype=np.float64)
    # Reverse chronological order is used throughout. Rotating fiscal-quarter
    # labels changes dummy names, not their span, so this matrix covers every Q.
    X = np.column_stack([np.ones(WINDOW), j - j.mean()] +
                        [(j.astype(np.int64) % 4 == phase).astype(np.float64) for phase in [1, 2, 3]])
    projection = np.eye(WINDOW) - X @ np.linalg.pinv(X)
    residual = normalized @ projection.T
    now, lag = residual[:, :-1], residual[:, 1:]
    now = now - now.mean(axis=1, keepdims=True)
    lag = lag - lag.mean(axis=1, keepdims=True)
    nn, ll = np.mean(now * now, axis=1), np.mean(lag * lag, axis=1)
    valid &= (nn > REL_VAR_FLOOR) & (ll > REL_VAR_FLOOR)
    denominator = np.sqrt(np.where(valid, nn * ll, np.nan))
    value = np.clip(np.mean(now * lag, axis=1) / denominator, -1.0, 1.0)
    return np.where(valid & np.isfinite(value), value, np.nan)

def build(parameters):
    if parameters != {"signal": "detrended_profit_persistence", "window": 16}:
        raise ValueError("Only the single frozen R5B definition is permitted")
    def compute(ctx):
        income = _quarter_table(ctx, "stock_income", "n_income_attr_p")
        events = _income_events(income)
        if events.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        profit = _single_matrix(_sorted_versions(income), events, 16)
        return _map(ctx, events, _residual_persistence(profit))
    return compute
