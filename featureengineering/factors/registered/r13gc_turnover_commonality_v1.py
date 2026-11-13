"""Isolated, non-registered implementation draft for LE05.

Input matrices are [trading-session, fixed-pool-stock]. `volume` and
`float_share` must use the same share unit. Missing/halted/zero-volume rows
remain invalid; valid turnover is 100 * volume / float_share. The signal is a
raw descriptive OLS beta, not a return-direction alpha.
"""

from __future__ import annotations

import numpy as np


WINDOW_PAIRS = 126
MIN_PAIRS = 95
MIN_PEERS = 100
MIN_MARKET_VARIANCE = 1e-12
PARAMETERS = {"signal": "turnover_commonality_beta_126"}


def _leave_one_out_medians(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Daily cross-sectional medians of log1p(turnover), excluding each stock.

    Uses one sort per date and rank deletion, O(C log C), avoiding an O(C^2)
    stock-by-stock peer expansion.
    """
    t_count, c_count = z.shape
    out = np.full((t_count, c_count), np.nan, dtype=np.float64)
    for t in range(t_count):
        cols = np.flatnonzero(valid[t] & np.isfinite(z[t]))
        n = cols.size
        if n - 1 < MIN_PEERS:
            continue
        ordered = np.sort(z[t, cols])
        ranks = np.searchsorted(ordered, z[t, cols], side="left")
        # If one item at rank r is removed, new order-statistic k maps to
        # original k when k < r, otherwise to k+1. This covers odd and even
        # sample sizes and preserves the usual midpoint median convention.
        n_after = n - 1
        k_lo = (n_after - 1) // 2
        k_hi = n_after // 2
        idx_lo = k_lo + (ranks <= k_lo)
        idx_hi = k_hi + (ranks <= k_hi)
        med = 0.5 * (ordered[idx_lo] + ordered[idx_hi])
        out[t, cols] = med
    return out


def turnover_commonality_beta(
    volume: np.ndarray,
    float_share: np.ndarray,
    actually_traded: np.ndarray,
    current_universe: np.ndarray | None = None,
) -> np.ndarray:
    """Compute fixed-window leave-one-out turnover commonality beta.

    Formula at stock i/date T:
      z_i,t = log1p(100 * volume_i,t / float_share_i,t)
      M_-i,t = median_j!=i(z_j,t) over actually traded valid peers
      dx_i,t = z_i,t - z_i,t-1; dm_i,t = M_-i,t - M_-i,t-1
      beta_i,T = Cov(dx_i, dm_i) / Var(dm_i), with an OLS intercept,
                 over the trailing 126 calendar slots ending T.

    At least 95 valid adjacent pairs are required, every peer median needs at
    least 100 other valid stocks, and the population variance of dm must exceed
    1e-12. Only current-universe and actually-traded stocks receive a value.
    Missing, suspended, invalid-share, and zero-volume rows are never converted
    to zero. Array inputs and outputs are float64; nonfinite results are NaN.
    """
    volume = np.asarray(volume, dtype=np.float64)
    float_share = np.asarray(float_share, dtype=np.float64)
    actually_traded = np.asarray(actually_traded, dtype=bool)
    if volume.ndim != 2 or float_share.shape != volume.shape:
        raise ValueError("volume and float_share must be same-shape 2D arrays")
    if actually_traded.shape != volume.shape:
        raise ValueError("actually_traded must match input matrix shape")
    if current_universe is None:
        current_universe = np.ones(volume.shape, dtype=bool)
    else:
        current_universe = np.asarray(current_universe, dtype=bool)
        if current_universe.shape != volume.shape:
            raise ValueError("current_universe must match input matrix shape")

    t_count, c_count = volume.shape
    result = np.full((t_count, c_count), np.nan, dtype=np.float64)
    if t_count <= WINDOW_PAIRS or c_count <= MIN_PEERS:
        return result

    valid = (
        actually_traded
        & np.isfinite(volume)
        & (volume > 0.0)
        & np.isfinite(float_share)
        & (float_share > 0.0)
    )
    z = np.full(volume.shape, np.nan, dtype=np.float64)
    np.divide(volume, float_share, out=z, where=valid)
    z *= 100.0
    valid &= np.isfinite(z) & (z >= 0.0)
    np.log1p(z, out=z, where=valid)
    z[~valid] = np.nan

    peer_median = _leave_one_out_medians(z, valid)

    end = np.arange(t_count, dtype=np.int64)
    start = end - WINDOW_PAIRS + 1
    mature = start >= 1  # 126 differences require 127 level slots.
    safe_start = np.maximum(start, 0)

    # Process columns one at a time to avoid large temporary T x C cumulative
    # arrays on the full fixed pool. Prefix sums give O(T*C) rolling OLS work.
    for c in range(c_count):
        xv = np.full(t_count, np.nan, dtype=np.float64)
        yv = np.full(t_count, np.nan, dtype=np.float64)
        xv[1:] = z[1:, c] - z[:-1, c]
        yv[1:] = peer_median[1:, c] - peer_median[:-1, c]
        pair = np.isfinite(xv) & np.isfinite(yv)
        if not pair.any():
            continue
        x0 = np.where(pair, xv, 0.0)
        y0 = np.where(pair, yv, 0.0)
        p_count = np.concatenate(([0.0], np.cumsum(pair, dtype=np.float64)))
        p_x = np.concatenate(([0.0], np.cumsum(x0, dtype=np.float64)))
        p_y = np.concatenate(([0.0], np.cumsum(y0, dtype=np.float64)))
        p_xx = np.concatenate(([0.0], np.cumsum(x0 * x0, dtype=np.float64)))
        p_yy = np.concatenate(([0.0], np.cumsum(y0 * y0, dtype=np.float64)))
        p_xy = np.concatenate(([0.0], np.cumsum(x0 * y0, dtype=np.float64)))

        hi = end + 1
        n = p_count[hi] - p_count[safe_start]
        sx = p_x[hi] - p_x[safe_start]
        sy = p_y[hi] - p_y[safe_start]
        sxx = p_xx[hi] - p_xx[safe_start]
        syy = p_yy[hi] - p_yy[safe_start]
        sxy = p_xy[hi] - p_xy[safe_start]
        var_y_sum = syy - sy * sy / np.maximum(n, 1.0)
        cov_xy_sum = sxy - sx * sy / np.maximum(n, 1.0)
        var_y = var_y_sum / np.maximum(n, 1.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            beta = cov_xy_sum / var_y_sum
        eligible = (
            mature
            & (n >= MIN_PAIRS)
            & np.isfinite(var_y)
            & (var_y > MIN_MARKET_VARIANCE)
            & current_universe[:, c]
            & valid[:, c]
            & np.isfinite(beta)
        )
        result[eligible, c] = beta[eligible]

    result[~np.isfinite(result)] = np.nan
    return result


def build(parameters):
    """Build the one frozen FactorContext wrapper for the isolated LE05 draft.

    Historical input arrays come only through PriceLayer: volume is a flow
    (halted rows stay NaN), float_share is the PIT state field, and traded is
    PriceLayer's finite positive-volume mask. No source DataFrame is read here.
    """
    if parameters != PARAMETERS:
        raise ValueError("Only the frozen LE05 126-pair beta expression is allowed")

    def compute(ctx):
        volume = np.asarray(ctx.px("vol"), dtype=np.float64)
        float_share = np.asarray(ctx.px("float_share"), dtype=np.float64)
        actually_traded = np.asarray(ctx.traded(), dtype=bool)
        current_universe = np.asarray(ctx.universe, dtype=bool)
        value = turnover_commonality_beta(
            volume, float_share, actually_traded, current_universe
        )
        return np.ascontiguousarray(value, dtype=np.float64)

    return compute
