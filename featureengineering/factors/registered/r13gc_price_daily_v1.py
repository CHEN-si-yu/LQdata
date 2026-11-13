"""R13 GC daily descriptive drafts; no engine registration or data IO.

The family has exactly three frozen (signal, window, min_count) parameter
triples. All outputs are raw descriptive values: higher_is_better is only the
project's declared=raw convention and makes no predictive claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 63
MIN_COUNT = 50
CHUNK_COLS = 128
SQRT_4LOG2 = float(np.sqrt(4.0 * np.log(2.0)))

FROZEN = {
    ("return_quantile_skew", 63, 50): "r13gc_daily_return_quantile_skew_63_v1",
    ("daily_turnover_ac1", 63, 50): "r13gc_daily_turnover_ac1_63_v1",
    ("daily_range_tail_spread", 63, 50): "r13gc_daily_range_tail_spread_63_v1",
}
PARAM_KEYS = {"signal", "window", "min_count"}


def _panel2d(x, name):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape (T,C)")
    return x


def _mask_grid(value, universe, traded):
    value = _panel2d(value, "value")
    universe = np.asarray(universe, dtype=bool)
    traded = np.asarray(traded, dtype=bool)
    if universe.shape != value.shape or traded.shape != value.shape:
        raise ValueError("universe/traded must match value shape")
    return np.where(universe & traded & np.isfinite(value), value, np.nan)


def rolling_quantile(panel, q, window=WINDOW, min_count=MIN_COUNT,
                     chunk_cols=CHUNK_COLS):
    """Trailing linear quantile, chunked over codes to bound temporary memory."""
    x = _panel2d(panel, "panel")
    if window != WINDOW or min_count != MIN_COUNT:
        raise ValueError("only the frozen 63/50 rolling-quantile contract is allowed")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0,1]")
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for left in range(0, x.shape[1], chunk_cols):
        right = min(left + chunk_cols, x.shape[1])
        frame = pd.DataFrame(x[:, left:right], copy=False)
        result = frame.rolling(window=window, min_periods=min_count).quantile(
            q, interpolation="linear")
        out[:, left:right] = result.to_numpy(dtype=np.float64, copy=False)
    return out


def return_quantile_skew(returns, universe, traded):
    """(q95 + q05 - 2*q50)/(q95-q05), over 63 rows incl. current, min 50."""
    r = _panel2d(returns, "returns")
    u = np.asarray(universe, dtype=bool)
    tr = np.asarray(traded, dtype=bool)
    if u.shape != r.shape or tr.shape != r.shape:
        raise ValueError("universe/traded must match returns shape")
    r = np.where(tr & np.isfinite(r), r, np.nan)
    q05 = rolling_quantile(r, .05)
    q50 = rolling_quantile(r, .50)
    q95 = rolling_quantile(r, .95)
    den = q95 - q05
    with np.errstate(all="ignore"):
        value = (q95 + q05 - 2.0 * q50) / den
    value = np.where(np.isfinite(den) & (den > 0.0), value, np.nan)
    return _mask_grid(value, u, tr)


def _roll_sum_full(x, window):
    """Cumulative-sum trailing window, requiring a complete row window."""
    x = _panel2d(x, "rolling input")
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] < window:
        return out
    cs = np.vstack((np.zeros((1, x.shape[1]), dtype=np.float64),
                    np.cumsum(x, axis=0, dtype=np.float64)))
    out[window - 1:] = cs[window:] - cs[:-window]
    return out


def daily_turnover_ac1(turnover, universe, traded):
    """Pearson AC1 of log(turnover) over the 63 adjacent calendar-row pairs.

    Positive turnover and traded status are required at both endpoints. At least
    50 of 63 adjacent pairs must be usable; gaps are not bridged.
    """
    to = _panel2d(turnover, "turnover")
    u = np.asarray(universe, dtype=bool)
    tr = np.asarray(traded, dtype=bool)
    if u.shape != to.shape or tr.shape != to.shape:
        raise ValueError("universe/traded must match turnover shape")
    valid = tr & np.isfinite(to) & (to > 0.0)
    log_to = np.full(to.shape, np.nan, dtype=np.float64)
    log_to[valid] = np.log(to[valid])
    prev = np.full(to.shape, np.nan, dtype=np.float64)
    prev[1:] = log_to[:-1]
    prev_valid = np.zeros(to.shape, dtype=bool)
    prev_valid[1:] = valid[:-1]
    pair = valid & prev_valid & np.isfinite(log_to) & np.isfinite(prev)
    x = np.where(pair, prev, 0.0)
    y = np.where(pair, log_to, 0.0)
    w = pair.astype(np.float64)
    n = _roll_sum_full(w, WINDOW)
    sx = _roll_sum_full(x, WINDOW)
    sy = _roll_sum_full(y, WINDOW)
    sxx = _roll_sum_full(x * x, WINDOW)
    syy = _roll_sum_full(y * y, WINDOW)
    sxy = _roll_sum_full(x * y, WINDOW)
    with np.errstate(all="ignore"):
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        cov = sxy - sx * sy / n
        rho = cov / np.sqrt(vx * vy)
        tol_x = 64.0 * np.finfo(np.float64).eps * np.maximum.reduce(
            [np.abs(sxx), np.abs(sx * sx / n), np.ones_like(sxx)])
        tol_y = 64.0 * np.finfo(np.float64).eps * np.maximum.reduce(
            [np.abs(syy), np.abs(sy * sy / n), np.ones_like(syy)])
    good = (n >= MIN_COUNT) & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(cov)
    # A complete 63-transition horizon spans 64 calendar-row observations;
    # do not emit an artificially shortened first window just because min_count=50.
    good &= (np.arange(to.shape[0])[:, None] >= WINDOW)
    good &= (vx > tol_x) & (vy > tol_y)
    rho = np.where(good, np.clip(rho, -1.0, 1.0), np.nan)
    return _mask_grid(rho, u, tr)


def daily_range_tail_spread(high, low, universe, traded):
    """Q90/Q50 of daily Parkinson range over 63 rows; valid zero-ranges stay in."""
    h = _panel2d(high, "high")
    l = _panel2d(low, "low")
    u = np.asarray(universe, dtype=bool)
    tr = np.asarray(traded, dtype=bool)
    if l.shape != h.shape or u.shape != h.shape or tr.shape != h.shape:
        raise ValueError("high/low/universe/traded must have equal (T,C) shape")
    valid = tr & np.isfinite(h) & np.isfinite(l) & (l > 0.0) & (h >= l)
    p = np.full(h.shape, np.nan, dtype=np.float64)
    p[valid] = np.log(h[valid] / l[valid]) / SQRT_4LOG2
    q50 = rolling_quantile(p, .50)
    q90 = rolling_quantile(p, .90)
    with np.errstate(all="ignore"):
        value = q90 / q50
    value = np.where(np.isfinite(q50) & (q50 > 0.0), value, np.nan)
    return _mask_grid(value, u, tr)


def build(parameters):
    """Only exact frozen 3-parameter triples are accepted; no window search."""
    if not isinstance(parameters, dict) or set(parameters) != PARAM_KEYS:
        raise ValueError("parameters must be exactly {signal, window, min_count}")
    if (not isinstance(parameters["signal"], str) or
            type(parameters["window"]) is not int or
            type(parameters["min_count"]) is not int):
        raise ValueError("signal must be str and window/min_count must be exact integers")
    key = (parameters["signal"], parameters["window"], parameters["min_count"])
    if key not in FROZEN:
        raise ValueError(f"Only the three frozen R13 GC variants are allowed: {sorted(FROZEN)}")
    signal = key[0]

    def compute(ctx):
        universe = np.asarray(ctx.universe, dtype=bool)
        traded = np.asarray(ctx.traded(), dtype=bool)
        if signal == "return_quantile_skew":
            returns = np.asarray(ctx.ret_clean(1), dtype=np.float64)
            value = return_quantile_skew(returns, universe, traded)
        elif signal == "daily_turnover_ac1":
            turnover = np.asarray(ctx.px("turnover"), dtype=np.float64)
            value = daily_turnover_ac1(turnover, universe, traded)
        else:
            high = np.asarray(ctx.px_raw("high"), dtype=np.float64)
            low = np.asarray(ctx.px_raw("low"), dtype=np.float64)
            value = daily_range_tail_spread(high, low, universe, traded)
        return np.ascontiguousarray(value, dtype=np.float64)

    return compute
