"""R28 approved intraday peak-volatility relaxation factor.

The formula consumes the shared minute_structure_v1 cache and preserves the
frozen R28 candidate calculation, including its within-session event selection,
zero/NaN rules, and fixed 20-session mean.
"""
from __future__ import annotations

import importlib

import numpy as np


NAME = "intraday_peak_volatility_relaxation_20_v1"
WINDOW = 20
MIN_VALID = 15
N_BARS = 48
J = np.r_[np.arange(1, 24, dtype=np.int64), np.arange(25, 48, dtype=np.int64)]
SEGMENTS = (np.arange(23), np.arange(23, 46))


def _event_room(before: int, after: int) -> np.ndarray:
    room = np.zeros(46, dtype=bool)
    for seg in SEGMENTS:
        room[seg[before:len(seg) - after]] = True
    return room


def _max_eligible_slot(r: np.ndarray, before: int = 3, after: int = 3):
    room = _event_room(before, after)
    complete = np.isfinite(r).all(axis=2)
    score = np.where(room[None, None, :], np.abs(r), -np.inf)
    q = np.argmax(score, axis=2)
    ok = complete & (np.max(score, axis=2) > 0.0)
    return q, ok


def _daily_value(returns: np.ndarray) -> np.ndarray:
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != N_BARS:
        raise ValueError(f"expected (T,C,{N_BARS}) return slots, got {x.shape}")
    r = x[:, :, J]
    q, ok = _max_eligible_slot(r, before=3, after=3)
    t, c = np.indices(q.shape)
    prior = np.zeros(q.shape, dtype=np.float64)
    post = np.zeros(q.shape, dtype=np.float64)
    for k in (1, 2, 3):
        prior += np.square(r[t, c, np.maximum(q - k, 0)])
        post += np.square(r[t, c, np.minimum(q + k, 45)])
    daily = np.full(q.shape, np.nan, dtype=np.float64)
    valid = ok & (prior > 0.0) & (post > 0.0)
    daily[valid] = np.log(post[valid] / prior[valid])
    return daily


def _fixed_window_mean(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        w = x[max(0, t - WINDOW + 1):t + 1]
        n = np.isfinite(w).sum(axis=0)
        good = n >= MIN_VALID
        total = np.nansum(w, axis=0)
        out[t, good] = total[good] / n[good]
    return out


def _compute(ctx) -> np.ndarray:
    minute = importlib.import_module("factors.registered.minute_structure_v1")
    layer = minute._layer(ctx)
    scalar, slots = layer.panel_all(ctx.panel)
    returns = slots[0]
    if returns.shape != (*ctx.panel.shape, N_BARS):
        raise RuntimeError(f"minute cache return axes mismatch: {returns.shape}")
    daily = _daily_value(returns)
    value = _fixed_window_mean(daily)
    # FactorContext.universe is the ordinary output mask applied by Engine too;
    # retaining it here follows the project convention while preserving the
    # frozen candidate's NaN/rolling behavior on otherwise eligible dates.
    return np.where(np.asarray(ctx.universe, dtype=bool), value, np.nan)


def build(parameters: dict):
    if parameters:
        raise ValueError(f"{NAME} has no tunable parameters")
    return _compute
