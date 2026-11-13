"""R38 margin-flow factor: short-selling volume as a share of total volume.

The source ``stock_margin_detail`` is published with a one-session delay, so
the output at T uses only source dates in [T-5, T-1].  The factor is the mean
short-selling-volume / total-volume ratio over those five source sessions
(at least four valid days), masked to a real trading row at T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "mf_short_sell_volume_share_5_v1"
ALT_NAME = "mf_fin_buy_rate_anomaly_60_v1"
WINDOW = 5
MIN_VALID = 4
BASE_WINDOW = 60
BASE_MIN = 40


def _panel(ctx, frame, column):
    idx = np.asarray(ctx.code_index(frame["stock_code"].to_numpy()), dtype=np.int64)
    day = np.asarray(ctx.date_col(frame["trade_date"]), dtype=np.int64)
    keep = idx >= 0
    val = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    return ctx.panel.place(idx[keep], day[keep], val[keep])


def _short_sell_volume_share(ctx):
    end_year = int(ctx.panel.dates[-1]) // 10000
    margin = ctx.dataset(
        "stock_margin_detail",
        columns=["trade_date", "stock_code", "rqmcl"],
        years=(int(ctx.panel.dates[0]) // 10000, end_year),
    )
    if margin is None or margin.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    margin = margin.copy()
    margin["stock_code"] = margin["stock_code"].astype(str)
    rqmcl = _panel(ctx, margin, "rqmcl")
    seen = np.isfinite(rqmcl)
    vol = ctx.px_raw("vol")
    traded = ctx.traded()
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(
            seen & np.isfinite(rqmcl) & (rqmcl >= 0.0)
            & np.isfinite(vol) & (vol > 0.0) & traded,
            rqmcl / np.where(np.isfinite(vol) & (vol > 0.0), vol, 1.0),
            np.nan,
        )
    shifted = ctx.shift(x, 1)
    mean = ctx.roll_mean(shifted, WINDOW, min_count=MIN_VALID)
    return np.where(traded, mean, np.nan)


def _fin_buy_rate_anomaly(ctx):
    end_year = int(ctx.panel.dates[-1]) // 10000
    margin = ctx.dataset(
        "stock_margin_detail",
        columns=["trade_date", "stock_code", "rzmre", "rzye"],
        years=(int(ctx.panel.dates[0]) // 10000, end_year),
    )
    if margin is None or margin.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    margin = margin.copy()
    margin["stock_code"] = margin["stock_code"].astype(str)
    rzmre = _panel(ctx, margin, "rzmre")
    rzye = _panel(ctx, margin, "rzye")
    seen = np.isfinite(rzmre) | np.isfinite(rzye)
    seen_prev = ctx.shift(seen.astype(np.float64), 1) > 0.0
    rzye_prev = ctx.shift(rzye, 1)
    traded = ctx.traded()
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(
            seen & seen_prev & np.isfinite(rzmre) & (rzmre >= 0.0)
            & np.isfinite(rzye_prev) & (rzye_prev > 0.0),
            np.log1p(rzmre / np.where(rzye_prev > 0.0, rzye_prev, 1.0)),
            np.nan,
        )
    base = ctx.roll_mean(ctx.shift(b, 1), BASE_WINDOW, min_count=BASE_MIN)
    anomaly = b - base
    signal = ctx.roll_mean(ctx.shift(anomaly, 1), WINDOW, min_count=MIN_VALID)
    return np.where(traded, signal, np.nan)


def build(parameters):
    variant = (parameters or {}).get("variant", NAME)
    if variant == NAME:
        return _short_sell_volume_share
    if variant == ALT_NAME:
        return _fin_buy_rate_anomaly
    raise KeyError(f"unknown R38 margin-flow variant: {variant!r}")
