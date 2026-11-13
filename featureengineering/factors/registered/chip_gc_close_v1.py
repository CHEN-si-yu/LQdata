"""Descriptive chip-distribution metrics relative to the same-day close.

Inputs are one stock-day's cleaned, duplicate-merged chip histogram.  Values
are vendor-estimated cost levels and masses, not observed holder lots or flows.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit

NAMES = (
    "gain_mean_cond", "loss_mean_cond", "gain_sd_cond", "loss_sd_cond",
    "gain_cv_cond", "loss_cv_cond", "above_close_semidev", "below_close_semidev",
    "gain_loss_mean_ratio", "downside_skew_cond", "upside_skew_cond",
    "downside_entropy", "upside_entropy", "downside_hhi", "upside_hhi",
    "gain_cvar10", "loss_cvar10", "cvar_asymmetry", "loss_tail_to_mean",
    "gain_tail_to_mean", "nearest_upper_gap", "nearest_lower_gap",
    "upper_adjacent_mass", "lower_adjacent_mass", "upper_nearest_share",
    "lower_nearest_share", "upper_local_density", "lower_local_density",
    "local_density_asymmetry", "upper_spacing_ratio", "lower_spacing_ratio",
    "adjacent_mass_center", "adjacent_mass_imbalance", "mean_abs_log_gap",
    "median_abs_log_gap", "abs_gap_mad", "abs_gap_to_sd",
)

_N = len(NAMES)


@njit(cache=True)
def _moments(x, w):
    sw = float(np.sum(w))
    if sw <= 0.0:
        return (math.nan, math.nan, math.nan, math.nan)
    mu = float(np.dot(x, w) / sw)
    d = x - mu
    v = float(np.dot(d * d, w) / sw)
    sd = math.sqrt(max(v, 0.0))
    skew = float(np.dot(d * d * d, w) / sw / (sd * sd * sd)) if sd > 0 else math.nan
    cv = sd / abs(mu) if mu != 0.0 else math.nan
    return mu, sd, cv, skew


@njit(cache=True)
def _entropy_hhi(w):
    sw = float(np.sum(w))
    if sw <= 0.0:
        return math.nan, math.nan
    q = w / sw
    h = 0.0
    hhi = 0.0
    for z in q:
        if z > 0.0:
            h -= float(z) * math.log(float(z))
            hhi += float(z) * float(z)
    return h, hhi


@njit(cache=True)
def _weighted_quantile(x, w, q):
    sw = float(np.sum(w))
    if sw <= 0.0:
        return math.nan
    order = np.argsort(x)
    xx, ww = x[order], w[order]
    target = q * sw
    acc = 0.0
    for i in range(len(xx)):
        acc += float(ww[i])
        if acc >= target:
            return float(xx[i])
    return float(xx[-1])


@njit(cache=True)
def _cvar_positive(loss, w, tail=0.10):
    """Mean of the worst ``tail`` probability mass of a nonnegative loss."""
    order = np.argsort(loss)[::-1]
    remaining = tail
    total = 0.0
    for i in order:
        take = min(float(w[i]), remaining)
        if take > 0.0:
            total += take * float(loss[i])
            remaining -= take
        if remaining <= 1e-15:
            break
    # If positive losses have less than tail mass, the unfilled part is zero.
    return total / tail


@njit(cache=True)
def _side_geometry(x, w, indices):
    """Density and spacing of the first three occupied support levels by distance."""
    if indices.size < 2:
        return math.nan, math.nan
    ordered = indices[np.argsort(np.abs(x[indices]))]
    d1 = abs(float(x[ordered[0]]))
    d2 = abs(float(x[ordered[1]]))
    width = d2 - d1
    density = float(w[ordered[0]]) / width if width > 0.0 else math.nan
    if ordered.size >= 3:
        d3 = abs(float(x[ordered[2]]))
        spacing = (d3 - d2) / width if width > 0.0 else math.nan
    else:
        spacing = math.nan
    return density, spacing


@njit(cache=True)
def summarize(p, w, close) -> np.ndarray:
    """Return one float64 vector in ``NAMES`` order.

    The parent owns cleaning, duplicate-price aggregation, sorting, mass
    normalization, and real-trade masking.  Invalid close yields all NaNs.
    """
    out = np.full(_N, np.nan, dtype=np.float64)
    c = float(close)
    if not math.isfinite(c) or c <= 0.0:
        return out
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if p.ndim != 1 or w.ndim != 1 or len(p) != len(w) or len(p) == 0:
        return out
    if not np.all(np.isfinite(p)) or not np.all(p > 0.0):
        return out
    if not np.all(np.isfinite(w)) or not np.all(w >= 0.0):
        return out
    sw = float(np.sum(w))
    if sw <= 0.0:
        return out
    w = w / sw
    x = np.log(p / c)
    below = x < 0.0
    above = x > 0.0
    wg, wl = w[below], w[above]
    gains, losses = -x[below], x[above]
    gm, gs, gcv, gsk = _moments(gains, wg)
    lm, ls, lcv, lsk = _moments(losses, wl)
    # Price-axis semi-deviations: explicitly separate them from holder P/L.
    above_semi = math.sqrt(max(float(np.dot(np.maximum(x, 0.0) ** 2, w)), 0.0))
    below_semi = math.sqrt(max(float(np.dot(np.maximum(-x, 0.0) ** 2, w)), 0.0))
    gh, ghhi = _entropy_hhi(wg)
    lh, lhhi = _entropy_hhi(wl)
    gcvar = _cvar_positive(np.maximum(-x, 0.0), w)
    lcvar = _cvar_positive(np.maximum(x, 0.0), w)
    q_abs = np.abs(x)
    med_abs = _weighted_quantile(q_abs, w, 0.5)
    abs_mad = float(np.dot(np.abs(q_abs - med_abs), w)) if math.isfinite(med_abs) else math.nan
    abs_mean = float(np.dot(q_abs, w))
    abs_sd = math.sqrt(max(float(np.dot((q_abs - abs_mean) ** 2, w)), 0.0))

    # Exact nearest occupied price levels on each side; no arbitrary price band.
    ni = np.flatnonzero(below & (w > 0.0))
    pi = np.flatnonzero(above & (w > 0.0))
    nearest_lower = int(ni[np.argmax(x[ni])]) if ni.size else -1
    nearest_upper = int(pi[np.argmin(x[pi])]) if pi.size else -1
    lower_mass = float(w[nearest_lower]) if nearest_lower >= 0 else 0.0
    upper_mass = float(w[nearest_upper]) if nearest_upper >= 0 else 0.0
    lower_total, upper_total = float(np.sum(wg)), float(np.sum(wl))
    lower_gap = float(-x[nearest_lower]) if nearest_lower >= 0 else math.nan
    upper_gap = float(x[nearest_upper]) if nearest_upper >= 0 else math.nan

    ld, lr = _side_geometry(x, w, ni)
    ud, ur = _side_geometry(x, w, pi)
    density_asym = ((ud - ld) / (ud + ld)
                    if math.isfinite(ud) and math.isfinite(ld) and ud + ld > 0 else math.nan)
    adj_mass = lower_mass + upper_mass
    adj_num = 0.0
    if nearest_upper >= 0:
        adj_num += upper_mass * upper_gap
    if nearest_lower >= 0:
        adj_num -= lower_mass * lower_gap
    adj_center = adj_num / adj_mass if adj_mass > 0.0 else math.nan
    adj_imbalance = ((upper_mass - lower_mass) / adj_mass if adj_mass > 0 else math.nan)
    gain_loss = gm / lm if math.isfinite(gm) and math.isfinite(lm) and lm > 0 else math.nan
    cvar_asym = ((gcvar - lcvar) / (gcvar + lcvar)
                 if gcvar + lcvar > 0 else math.nan)

    out[0] = gm
    out[1] = lm
    out[2] = gs
    out[3] = ls
    out[4] = gcv
    out[5] = lcv
    out[6] = above_semi
    out[7] = below_semi
    out[8] = gain_loss
    out[9] = lsk
    out[10] = gsk
    out[11] = lh
    out[12] = gh
    out[13] = lhhi
    out[14] = ghhi
    out[15] = gcvar
    out[16] = lcvar
    out[17] = cvar_asym
    out[18] = lcvar / lm if math.isfinite(lm) and lm > 0 else math.nan
    out[19] = gcvar / gm if math.isfinite(gm) and gm > 0 else math.nan
    out[20] = upper_gap
    out[21] = lower_gap
    out[22] = upper_mass
    out[23] = lower_mass
    out[24] = upper_mass / upper_total if upper_total > 0 else math.nan
    out[25] = lower_mass / lower_total if lower_total > 0 else math.nan
    out[26] = ud
    out[27] = ld
    out[28] = density_asym
    out[29] = ur
    out[30] = lr
    out[31] = adj_center
    out[32] = adj_imbalance
    out[33] = abs_mean
    out[34] = med_abs
    out[35] = abs_mad
    out[36] = abs_mean / abs_sd if abs_sd > 0 else math.nan
    return out
