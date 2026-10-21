"""Bounded examples. These hypotheses have not established profitability."""
from __future__ import annotations

import math
import warnings
import numpy as np


def volume_price_exhaustion(parameters):
    """Past-only volume acceleration / positive-price deceleration hypothesis."""
    from fea import mathx as mx
    window = int(parameters["window"])
    expression = parameters["expression"]
    valid_fraction = float(parameters.get("valid_fraction", 0.8))
    if window not in (5, 10, 20) or expression not in ("continuous", "event") or valid_fraction != 0.8:
        raise ValueError("Example family preregisters only six variants: windows5/10/20 x continuous/event, valid_fraction0.8")

    def compute(ctx):
        close = np.asarray(ctx.hfq("close"), dtype=np.float64)
        volume = np.asarray(ctx.px_raw("vol"), dtype=np.float64)
        traded = np.asarray(ctx.traded(), dtype=bool)
        with np.errstate(all="ignore"):
            log_price = np.where(close > 0, np.log(close), np.nan)
            log_volume = np.where(traded & (volume > 0), np.log(volume), np.nan)
        r_a = log_price - mx.shift(log_price, window)
        r_b = mx.shift(log_price, window) - mx.shift(log_price, 2 * window)
        need = int(math.ceil(window * valid_fraction))
        v_a = mx.roll_mean(log_volume, window, min_count=need)
        v_b = mx.shift(v_a, window)
        volume_accel = v_a - v_b
        price_decel = r_b - r_a
        valid = traded & np.isfinite(r_a) & np.isfinite(r_b) & np.isfinite(v_a) & np.isfinite(v_b)
        if expression == "event":
            result = ((r_a > 0) & (price_decel > 0) & (volume_accel > 0)).astype(np.float64)
        else:
            # Entirely unavailable warmup rows are expected and remain NaN.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice", category=RuntimeWarning)
                z_v = mx.cs_zscore(volume_accel)
                z_p = mx.cs_zscore(price_decel)
            result = np.where(r_a > 0, np.maximum(z_v, 0.0) * np.maximum(z_p, 0.0), 0.0)
        return np.where(valid, result, np.nan)
    return compute
