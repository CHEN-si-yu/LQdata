"""Roll-style five-minute price-bounce proxy using the shared minute structure cache."""
from __future__ import annotations

import importlib

import numpy as np

MINUTE_LAYER_MODULE = "factors.registered.minute_structure_v1"
N_BARS = 48
WINDOW = 20
MIN_DAYS = 16


def _fixed_window_mean(values, width, min_count=None):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"fixed_window_mean expects (T,C), got {x.shape}")
    width = int(width)
    if width <= 0:
        raise ValueError("width must be positive")
    need = width if min_count is None else int(min_count)
    if need < 0 or need > width:
        raise ValueError("min_count must be in [0,width]")
    t, c = x.shape
    sums = np.zeros((t, c), dtype=np.float64)
    count = np.zeros((t, c), dtype=np.int16)
    for lag in range(width):
        shifted = np.full((t, c), np.nan, dtype=np.float64)
        if lag == 0:
            shifted[:] = x
        elif lag < t:
            shifted[lag:] = x[:-lag]
        finite = np.isfinite(shifted)
        sums += np.where(finite, shifted, 0.0)
        count += finite.astype(np.int16)
    out = np.full((t, c), np.nan, dtype=np.float64)
    good = count >= need
    np.divide(sums, count, out=out, where=good & (count > 0))
    if need == 0:
        out[count == 0] = 0.0
    return out


def _population_lag_cov(x, y):
    """Pairwise row covariance after centering within a continuous session."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(axis=2)
    xx = np.where(ok, x, 0.0)
    yy = np.where(ok, y, 0.0)
    den = np.maximum(n, 1)
    mx = xx.sum(axis=2) / den
    my = yy.sum(axis=2) / den
    xc = np.where(ok, x - mx[:, :, None], 0.0)
    yc = np.where(ok, y - my[:, :, None], 0.0)
    cov = np.sum(xc * yc, axis=2) / den
    return np.where(n >= 16, cov, np.nan)


def _compute(ctx):
    minute_layer = importlib.import_module(MINUTE_LAYER_MODULE)
    layer = minute_layer._layer(ctx)
    scalar, slots = layer.panel_all(ctx.panel)
    rets = slots[0]
    if rets.shape[2] != N_BARS:
        raise RuntimeError(f"expected {N_BARS} cached return slots, got {rets.shape[2]}")

    # Slot 24 spans the lunch closure and is excluded. AM pairs are 2..23
    # against 1..22; PM pairs are 26..47 against 25..46.
    cov_am = _population_lag_cov(rets[:, :, 2:24], rets[:, :, 1:23])
    cov_pm = _population_lag_cov(rets[:, :, 26:48], rets[:, :, 25:47])
    cov = 0.5 * (cov_am + cov_pm)
    daily = 2.0 * np.sqrt(np.maximum(-cov, 0.0))
    daily = np.where(np.isfinite(cov), daily, np.nan)
    valid_day = scalar["grid_ok"] == 1.0
    daily = np.where(valid_day, daily, np.nan)
    value = _fixed_window_mean(daily, WINDOW, min_count=MIN_DAYS)
    value = np.where(valid_day & np.isfinite(value), value, np.nan)
    traded = np.asarray(ctx.traded(), dtype=bool)
    good = np.asarray(ctx.universe, dtype=bool) & traded & valid_day & np.isfinite(value)
    return np.where(good, value, np.nan)


def build(parameters):
    if parameters:
        raise ValueError("roll_bounce_proxy takes no parameters")
    return _compute
