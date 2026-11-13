"""R38 forecast-event factors: forward run-rate, type transition, midpoint momentum.

The ``stock_forecast`` table is visible at ``ann_date`` only.  A forecast state
is effective on the next exchange session and survives for 250 calendar days;
a later announcement supersedes it.  Financial TTM/assets use the production
PIT layers.  No label, return or evaluation file is read here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = (
    "fc_mid_vs_runrate_250d",
    "fc_type_transition_250d",
    "fc_mid_momentum_250d",
)
CARRY_DAYS = 250
START_YEAR = 2014
TYPE_SCORE = {
    "预增": 3.0, "略增": 2.0, "扭亏": 2.0, "续盈": 1.0, "不确定": 0.0,
    "其他": 0.0, "略减": -1.0, "预减": -2.0, "续亏": -2.0, "首亏": -3.0,
    "减亏": -1.0, "增亏": -2.0,
}


def _date_int(series) -> np.ndarray:
    raw = pd.to_datetime(series, errors="coerce")
    return np.where(
        raw.isna(), 0,
        raw.dt.strftime("%Y%m%d").astype("Int64").fillna(0).astype(np.int64).to_numpy(),
    ).astype(np.int64)


def _load_forecast(ctx):
    end_year = int(ctx.panel.dates[-1]) // 10000
    frame = ctx.dataset(
        "stock_forecast",
        columns=["stock_code", "ann_date", "end_date", "type",
                 "p_change_min", "p_change_max", "net_profit_min", "net_profit_max"],
        years=(START_YEAR, end_year),
    )
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    codes = np.asarray(ctx.panel.codes).astype(str)
    frame = frame[frame["stock_code"].isin(set(codes.tolist()))]
    frame["ann"] = _date_int(frame["ann_date"])
    frame["end"] = _date_int(frame["end_date"])
    md = (frame["end"] // 100) % 100
    day = frame["end"] % 100
    frame = frame[np.isin(md * 100 + day, [331, 630, 930, 1231]) & (frame["ann"] > 0)]
    if frame.empty:
        return None
    frame = frame.sort_values(["stock_code", "ann", "end"], kind="stable").reset_index(drop=True)
    return frame


def _row_features(frame):
    pmin = pd.to_numeric(frame["p_change_min"], errors="coerce").to_numpy(np.float64)
    pmax = pd.to_numeric(frame["p_change_max"], errors="coerce").to_numpy(np.float64)
    npmin = pd.to_numeric(frame["net_profit_min"], errors="coerce").to_numpy(np.float64)
    npmax = pd.to_numeric(frame["net_profit_max"], errors="coerce").to_numpy(np.float64)
    mid = (pmin + pmax) / 2.0
    tnum = frame["type"].map(TYPE_SCORE).astype(float).to_numpy(np.float64)
    end = frame["end"].to_numpy(np.int64)
    ann = frame["ann"].to_numpy(np.int64)
    n = len(frame)
    mid_prev_diff = np.full(n, np.nan)
    trans = np.full(n, np.nan)

    codes = frame["stock_code"].to_numpy()
    change = np.r_[True, codes[1:] != codes[:-1]] if n else np.array([], dtype=bool)
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n] if n else np.array([], dtype=int)
    for gi in range(len(starts)):
        lo, hi = int(starts[gi]), int(ends[gi])
        latest = {}
        for k in range(lo, hi):
            e = int(end[k])
            prior = sorted((q for q in latest if q < e), reverse=True)
            if prior:
                pm, pt = latest[prior[0]]
                if np.isfinite(pm) and np.isfinite(mid[k]):
                    mid_prev_diff[k] = float(np.arcsinh(mid[k] / 100.0)
                                             - np.arcsinh(pm / 100.0))
                if np.isfinite(pt) and np.isfinite(tnum[k]):
                    trans[k] = float(tnum[k] - pt)
            latest[e] = (mid[k], tnum[k])

    q = ((end.astype(np.float64) // 100.0) % 100.0) // 3.0
    q_ok = np.isfinite(q) & (q >= 1.0) & (q <= 4.0)
    ann_mid = (npmin + npmax) / 2.0 * 1e4 * 4.0 / np.where(q_ok, q, 1.0)
    ann_mid = np.where(q_ok & np.isfinite(npmin) & np.isfinite(npmax), ann_mid, np.nan)
    return ann_mid, mid_prev_diff, trans, ann


def _state(ctx, frame, ann, values):
    """Assign event values to next session through ann+250 days; later ann wins."""
    out = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    codes = np.asarray(ctx.panel.codes).astype(str)
    cpos = {c: i for i, c in enumerate(codes)}
    ci = np.asarray([cpos.get(c, -1) for c in frame["stock_code"].to_numpy()], dtype=np.int64)
    eff = np.searchsorted(days, ann, side="right").astype(np.int64)
    expiry = (pd.to_datetime(pd.Series(ann).astype(str), format="%Y%m%d", errors="coerce")
              + pd.Timedelta(days=CARRY_DAYS))
    exp_i = expiry.dt.strftime("%Y%m%d").astype("Int64").fillna(0).astype(np.int64).to_numpy()
    stop = np.searchsorted(days, exp_i, side="right") - 1
    stop = np.minimum(stop, len(days) - 1)
    order = np.lexsort((ann, ci))
    ci, eff, stop, values = ci[order], eff[order], stop[order], np.asarray(values)[order]
    for k in range(len(ci)):
        c = int(ci[k])
        e = int(eff[k])
        s = int(stop[k])
        if c < 0 or e >= len(days) or s < e:
            continue
        out[e:s + 1, c] = values[k]
    return out


def build(parameters):
    variant = (parameters or {}).get("variant")
    if variant not in NAMES:
        raise KeyError(f"unknown R38 forecast-event variant: {variant!r}")

    def compute(ctx):
        frame = _load_forecast(ctx)
        if frame is None:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        ann_mid, mid_diff, trans, ann = _row_features(frame)
        if variant == "fc_mid_vs_runrate_250d":
            state = _state(ctx, frame, ann, ann_mid)
            ttm_ni = ctx.ttm("n_income_attr_p")
            assets = ctx.point("total_assets")
            out = np.arcsinh(ctx.safe_div(state - ttm_ni, assets, 1e8))
        elif variant == "fc_type_transition_250d":
            out = _state(ctx, frame, ann, trans)
        else:
            out = _state(ctx, frame, ann, mid_diff)
        return np.where(ctx.universe, out, np.nan)

    return compute
