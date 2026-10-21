"""R6A frozen SG&A downside stickiness: one 20-quarter candidate, false.

Quarterly YTD differences are looked up together at each publication/revision
event. Twenty fixed report slots admit at least 16 valid YoY pairs, with four
strict up and four strict down observations. Missing/degenerate new windows
produce NaN events replacing old state. No return-dependent parameter choices.
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

WINDOW = 20
MIN_PAIRS = 16
MIN_SIDE = 4
EIG_FLOOR = 1e-8
FIELDS = ("revenue", "sell_exp", "admin_exp")

def _events(data):
    if data.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    data = data.sort_values(["code", "pit", "period"], kind="stable")
    data = data.assign(q=data.groupby("code", sort=False).period.cummax())
    return data.drop_duplicates(["code", "pit"], keep="last")[["code", "pit", "q"]].reset_index(drop=True)

def _stickiness(sales, expense):
    # Matrices are reverse chronological: column j is Q-j. Twenty fixed
    # chronological slot labels remain fixed even when some pairs are absent.
    s0, s4 = sales[:, :WINDOW], sales[:, 4:WINDOW+4]
    c0, c4 = expense[:, :WINDOW], expense[:, 4:WINDOW+4]
    valid = np.isfinite(s0) & np.isfinite(s4) & np.isfinite(c0) & np.isfinite(c4)
    valid &= (s0 > AMOUNT_FLOOR) & (s4 > AMOUNT_FLOOR)
    valid &= (c0 > AMOUNT_FLOOR) & (c4 > AMOUNT_FLOOR)
    with np.errstate(all="ignore"):
        x = np.log(np.where(valid, s0/s4, np.nan))
        y = np.log(np.where(valid, c0/c4, np.nan))
    valid &= np.isfinite(x) & np.isfinite(y)
    out = np.full(sales.shape[0], np.nan, dtype=np.float64)
    trend = (WINDOW-1-np.arange(WINDOW, dtype=np.float64)) - (WINDOW-1)/2
    for i in range(sales.shape[0]):
        use = valid[i]
        if use.sum() < MIN_PAIRS:
            continue
        xx, yy = x[i, use], y[i, use]
        if (xx < 0).sum() < MIN_SIDE or (xx > 0).sum() < MIN_SIDE:
            continue
        if np.std(xx) <= 0 or np.std(yy) <= 0:
            continue
        raw = np.column_stack([xx, np.where(xx < 0, xx, 0.0), trend[use]])
        scale = np.std(raw, axis=0)
        if not np.isfinite(scale).all() or (scale <= 0).any():
            continue
        z = (raw-raw.mean(axis=0)) / scale
        gram = z.T @ z / len(xx)
        if np.linalg.eigvalsh(gram)[0] <= EIG_FLOOR:
            continue
        design = np.column_stack([np.ones(len(xx)), z])
        coef = np.linalg.lstsq(design, yy, rcond=None)[0]
        # Undo only numerical design scaling; economic delta is unstandardized.
        value = -coef[2]/scale[1]
        if np.isfinite(value):
            out[i] = value
    return out

def build(parameters):
    if parameters != {"signal": "sga_downside_stickiness", "window": 20}:
        raise ValueError("Only the single frozen R6A definition is permitted")
    def compute(ctx):
        data = _table(ctx, "stock_income", FIELDS)
        events = _events(data)
        if events.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        sales = _single(_versions(data, "revenue"), events, WINDOW+4)
        sell = _single(_versions(data, "sell_exp"), events, WINDOW+4)
        admin = _single(_versions(data, "admin_exp"), events, WINDOW+4)
        values = _stickiness(sales, sell+admin)
        return _with_age(ctx, events, values)
    return compute

