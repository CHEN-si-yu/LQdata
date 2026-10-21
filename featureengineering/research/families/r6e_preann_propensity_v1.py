"""R6E frozen completed-announcement turnover propensity, weak negative prior.

Last four continuous independent visible-date events, not four selected valid
events. Every 5+60 day source window must be complete; all first PIT dates must
be within 730 calendar days. New invalid events replace old valid state with
NaN. Event anchor is strictly before PIT; release is strictly next trading day.
"""
from __future__ import annotations

import datetime as _dt
import numpy as np
import pandas as pd

FAST = 5
BASE = 60
EVENTS = 4
MAX_AGE = 730
BASE_FLOOR = 1e-6

def _ordinals(days):
    return np.asarray([_dt.date(int(d)//10000, (int(d)//100)%100, int(d)%100).toordinal()
                       for d in np.asarray(days).ravel()], dtype=np.int64)

def _events(ctx):
    frame = ctx.dataset("stock_income", columns=["stock_code", "ann_date", "f_ann_date", "end_date"])
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["code", "end", "pit", "visible", "anchor"])
    end = ctx.date_col(frame["end_date"]).astype(np.int64)
    ann = ctx.date_col(frame["ann_date"]).astype(np.int64)
    fann = ctx.date_col(frame["f_ann_date"]).astype(np.int64)
    pit = np.maximum(np.maximum(ann, fann), end)
    keep = np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= ((ann >= 19900101) | (fann >= 19900101))
    keep &= (pit > 0) & (pit <= int(ctx.panel.dates[-1]))
    code = ctx.code_index(frame["stock_code"].to_numpy()[keep])
    rows = pd.DataFrame({"code": code, "end": end[keep], "pit": pit[keep]})
    rows = rows.loc[rows.code >= 0].sort_values(["code", "end", "pit"], kind="stable")
    rows = rows.drop_duplicates(["code", "end"], keep="first")
    cal = np.asarray(ctx.cal.days, dtype=np.int64)
    pit = rows.pit.to_numpy(dtype=np.int64)
    vp = np.searchsorted(cal, pit, side="right")
    ap = np.searchsorted(cal, pit, side="left")-1
    valid = (vp < len(cal)) & (ap >= 0)
    rows = rows.loc[valid].copy()
    rows["visible"] = cal[vp[valid]]
    rows["anchor"] = cal[ap[valid]]
    rows = rows.loc[rows.visible <= int(ctx.panel.dates[-1])]
    # Earliest PIT, then earliest report end, is deterministic when a weekend
    # groups several different report dates into one next-trading-day release.
    return rows.sort_values(["code", "visible", "pit", "end"], kind="stable").drop_duplicates(
        ["code", "visible"], keep="first").reset_index(drop=True)

def _event_features(ctx, events, turnover, traded):
    out = np.full(len(events), np.nan, dtype=np.float64)
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    cal = np.asarray(ctx.cal.days, dtype=np.int64)
    for i, ev in enumerate(events.itertuples(index=False)):
        row = int(np.searchsorted(days, ev.anchor, side="left"))
        cp = int(np.searchsorted(cal, ev.anchor, side="left"))
        if row < FAST+BASE-1 or row >= len(days) or days[row] != ev.anchor or cp < FAST+BASE-1:
            continue
        hist = days[row-FAST-BASE+1:row+1]
        if not np.array_equal(hist, cal[cp-FAST-BASE+1:cp+1]):
            continue
        u = turnover[row-FAST-BASE+1:row+1, int(ev.code)]
        tr = traded[row-FAST-BASE+1:row+1, int(ev.code)]
        if not tr.all() or not np.isfinite(u).all() or not (u > 0).all():
            continue
        baseline, fast = u[:-FAST].mean(), u[-FAST:].mean()
        if baseline > BASE_FLOOR and fast > 0:
            value = np.log(fast/baseline)
            if np.isfinite(value):
                out[i] = value
    return out

def _map(ctx, events, values):
    if events.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    return ctx.panel.asof(events.code.to_numpy(dtype=np.int64),
                          events.visible.to_numpy(dtype=np.int64),
                          np.asarray(values, dtype=np.float64), dtype=np.float64)

def _aggregate(ctx, events, z):
    values = np.full(len(events), np.nan, dtype=np.float64)
    oldest = np.full(len(events), np.nan, dtype=np.float64)
    pit_ord = _ordinals(events.pit.to_numpy(dtype=np.int64))
    for _, group in events.groupby("code", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        for j in range(EVENTS-1, len(indices)):
            selected = indices[j-EVENTS+1:j+1]
            i = int(indices[j])
            # Keep every independent event in the queue, including invalid ones.
            oldest[i] = pit_ord[selected].min()
            if np.isfinite(z[selected]).all():
                values[i] = np.median(z[selected])
    grid = _map(ctx, events, values)
    age = _ordinals(ctx.panel.dates)[:, None] - _map(ctx, events, oldest)
    return np.where(np.isfinite(grid) & np.isfinite(age) & (age <= MAX_AGE), grid, np.nan)

def build(parameters):
    if parameters != {"signal": "completed_preann_turnover_propensity", "events": 4,
                      "max_age_days": 730, "fast": 5, "base": 60}:
        raise ValueError("Only the single frozen R6E definition is permitted")
    def compute(ctx):
        events = _events(ctx)
        if events.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        turnover = np.asarray(ctx.px("turnover"), dtype=np.float64)
        traded = np.asarray(ctx.traded(), dtype=bool)
        z = _event_features(ctx, events, turnover, traded)
        grid = _aggregate(ctx, events, z)
        return np.where(traded & np.isfinite(grid), grid, np.nan)
    return compute

