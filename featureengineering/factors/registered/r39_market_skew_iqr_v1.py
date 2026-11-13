"""R39 market cross-sectional intraday realized-skewness dispersion."""
from __future__ import annotations

import importlib
import numpy as np

NAME = "mkt_intraday_realized_skewness_iqr_5_v1"
WINDOW = 5
MIN_DAILY_STOCKS = 500
RET_SLOTS = np.r_[np.arange(1, 24, dtype=np.int64), np.arange(25, 48, dtype=np.int64)]


def _daily_components(ctx):
    minute = importlib.import_module("factors.registered.minute_structure_v1")
    layer = minute._layer(ctx)
    scalar, slot_arrays = layer.panel_all(ctx.panel)
    all_returns = np.asarray(slot_arrays[0], dtype=np.float64)
    if all_returns.shape != (*ctx.panel.shape, 48):
        raise RuntimeError(f"minute_structure_v1 return axes mismatch: {all_returns.shape}")

    returns = all_returns[:, :, RET_SLOTS]
    rv = np.sum(returns * returns, axis=2)
    third = np.sum(returns * returns * returns, axis=2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rs = np.sqrt(46.0) * third / np.power(rv, 1.5)

    traded = np.asarray(ctx.traded(), dtype=bool)
    universe = np.asarray(ctx.universe, dtype=bool)
    grid = (np.asarray(scalar["n_bars"]) == 48.0) & (np.asarray(scalar["grid_ok"]) == 1.0)
    valid = (universe & traded & grid & np.isfinite(returns).all(axis=2)
             & np.isfinite(rv) & (rv > 0.0) & np.isfinite(rs))

    daily_iqr = np.full(ctx.panel.T, np.nan, dtype=np.float64)
    daily_mean_rs = np.full(ctx.panel.T, np.nan, dtype=np.float64)
    daily_n = np.full(ctx.panel.T, np.nan, dtype=np.float64)
    for t in range(ctx.panel.T):
        values = rs[t, valid[t]]
        n = int(values.size)
        if n < MIN_DAILY_STOCKS:
            continue
        q25, q75 = np.quantile(values, [0.25, 0.75], method="linear")
        daily_iqr[t] = float(q75 - q25)
        daily_mean_rs[t] = float(np.mean(values))
        daily_n[t] = float(n)
    return daily_iqr, daily_mean_rs, daily_n


def _fixed_five_slot_mean(daily):
    x = np.asarray(daily, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for t in range(WINDOW - 1, len(x)):
        w = x[t - WINDOW + 1:t + 1]
        if np.isfinite(w).all():
            out[t] = float(np.mean(w))
    return out


def compute(ctx):
    """Return one scalar per panel date; T uses only same-session inputs."""
    daily_iqr, _, _ = _daily_components(ctx)
    return _fixed_five_slot_mean(daily_iqr)


def build(parameters):
    if parameters:
        raise ValueError(f"{NAME} takes no parameters")
    return compute
