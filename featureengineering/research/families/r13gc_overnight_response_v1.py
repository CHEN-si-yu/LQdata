"""Artifact-only GC draft: daily overnight-gap to intraday-response slope.

This file is intentionally outside ``research/`` and is not registered.  It
describes the frozen LE04 expression for independent review only.
"""
from __future__ import annotations

import numpy as np


PARAMETERS = {"signal": "overnight_gap_intraday_response_beta_60"}
WINDOW = 60
MIN_PAIRS = 45
MIN_OVERNIGHT_VARIANCE = 1e-12


def build(parameters):
    """Return the sole predeclared builder; window and cutoff are not tunable."""
    if parameters != PARAMETERS:
        raise ValueError("Only the frozen LE04 60-slot response-slope signal is allowed")

    def compute(ctx):
        # Strict raw endpoints prevent PriceLayer's level forward fill from
        # making a suspended day look like a valid open/close observation.
        raw_open = np.asarray(ctx.px_raw("open"), dtype=np.float64)
        raw_close = np.asarray(ctx.px_raw("close"), dtype=np.float64)
        # hfq() supplies the PIT cumulative adjustment factor. Its level
        # forward-fill is safe here only because every raw price endpoint is
        # independently required below to exist on an actually traded row.
        hfq_open = np.asarray(ctx.hfq("open"), dtype=np.float64)
        hfq_close = np.asarray(ctx.hfq("close"), dtype=np.float64)
        traded = np.asarray(ctx.traded(), dtype=bool)
        universe = np.asarray(ctx.universe, dtype=bool)
        clean_close_return = np.asarray(ctx.ret_clean(1), dtype=np.float64)
        if (raw_open.ndim != 2 or raw_close.shape != raw_open.shape or
                hfq_open.shape != raw_open.shape or hfq_close.shape != raw_open.shape or
                traded.shape != raw_open.shape or universe.shape != raw_open.shape or
                clean_close_return.shape != raw_open.shape):
            raise ValueError("LE04 requires matching date-by-stock panels")

        prev_close = np.asarray(ctx.shift(raw_close, 1), dtype=np.float64)
        prev_hfq_close = np.asarray(ctx.shift(hfq_close, 1), dtype=np.float64)
        prev_traded = np.zeros_like(traded, dtype=bool)
        if traded.shape[0] > 1:
            prev_traded[1:] = traded[:-1]

        valid = traded & prev_traded & np.isfinite(clean_close_return)
        valid &= np.isfinite(raw_open) & np.isfinite(raw_close) & np.isfinite(prev_close)
        valid &= np.isfinite(hfq_open) & np.isfinite(prev_hfq_close)
        valid &= (raw_open > 0.0) & (raw_close > 0.0) & (prev_close > 0.0)
        valid &= (hfq_open > 0.0) & (prev_hfq_close > 0.0)

        # hfq endpoints are exactly raw_price * PIT cumulative adj_factor;
        # raw endpoint checks above forbid stale forward-filled prices.
        overnight = np.full(raw_open.shape, np.nan, dtype=np.float64)
        intraday = np.full(raw_open.shape, np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            overnight[valid] = hfq_open[valid] / prev_hfq_close[valid] - 1.0
            intraday[valid] = raw_close[valid] / raw_open[valid] - 1.0
        valid &= np.isfinite(overnight) & np.isfinite(intraday)

        # Neutral zeros are used only inside sufficient-statistic reductions;
        # the companion count keeps missing pairs out of the OLS sample.
        x = np.where(valid, overnight, 0.0)
        y = np.where(valid, intraday, 0.0)
        v = valid.astype(np.float64)
        n = np.asarray(ctx.roll_sum(v, WINDOW, min_count=0), dtype=np.float64)
        sx = np.asarray(ctx.roll_sum(x, WINDOW, min_count=0), dtype=np.float64)
        sy = np.asarray(ctx.roll_sum(y, WINDOW, min_count=0), dtype=np.float64)
        sxx = np.asarray(ctx.roll_sum(x * x, WINDOW, min_count=0), dtype=np.float64)
        sxy = np.asarray(ctx.roll_sum(x * y, WINDOW, min_count=0), dtype=np.float64)

        n_safe = np.where(n > 0.0, n, 1.0)
        centered_xx = sxx - sx * sx / n_safe
        centered_xy = sxy - sx * sy / n_safe
        variance_x = centered_xx / n_safe
        beta = np.full(raw_open.shape, np.nan, dtype=np.float64)
        fit_ok = (n >= MIN_PAIRS) & np.isfinite(variance_x)
        fit_ok &= variance_x > MIN_OVERNIGHT_VARIANCE
        fit_ok &= np.isfinite(centered_xy) & np.isfinite(centered_xx) & (centered_xx > 0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            beta[fit_ok] = centered_xy[fit_ok] / centered_xx[fit_ok]

        # Require a mature 60-row calendar window even if 45 valid pairs
        # happen to occur in an incomplete leading prefix.
        mature = np.arange(raw_open.shape[0]) >= WINDOW - 1
        output_ok = mature[:, None] & universe & traded & fit_ok & np.isfinite(beta)
        out = np.where(output_ok, beta, np.nan).astype(np.float64, copy=False)
        out[~np.isfinite(out)] = np.nan
        return np.ascontiguousarray(out)

    return compute
