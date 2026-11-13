"""Descriptive stock-CYQ distribution temporal features.

API for integration: compute(ctx, getter, variant) -> float64 (T,C).
`getter(ctx, field)` returns the new shared annual metric panel. Fields are
absolute-log-price moments/quantiles, HHI, raw entropy, max_mass, n_peaks.
All temporal estimates use the single frozen 20-session window (15 valid min).
No return labels or future data are read.
"""
from __future__ import annotations
import numpy as np
WINDOW = 20
MIN_COUNT = 15
QFIELDS = tuple((f'q{q:02d}' for q in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95)))
LEVEL_FIELDS = {'log_mean', 'log_q10', 'log_q25', 'log_q50', 'log_q75', 'log_q90', *QFIELDS}
NAMES = (
    'cyq_log_median_drift_20_v1',
    'cyq_log_median_acceleration_20_v1',
    'cyq_log_median_path_efficiency_20_v1',
    'cyq_log_median_reversal_20_v1',
    'cyq_log_iqr_expansion_20_v1',
    'cyq_log_upper_tail_buildup_20_v1',
    'cyq_log_lower_tail_buildup_20_v1',
    'cyq_log_tail_imbalance_change_20_v1',
    'cyq_log_dispersion_drift_20_v1',
    'cyq_log_mean_median_gap_drift_20_v1',
    'cyq_log_skew_migration_20_v1',
    'cyq_log_tail_weight_migration_20_v1',
    'cyq_hhi_concentration_drift_20_v1',
    'cyq_entropy_reorganization_20_v1',
    'cyq_peak_mass_release_20_v1',
    'cyq_peak_fragmentation_20_v1',
    'cyq_peak_topology_churn_20_v1',
    'cyq_cost_shape_transport_20_v1',
    'cyq_cost_shape_directional_transport_20_v1',
    'cyq_width_change_persistence_20_v1',
    'cyq_shape_expansion_concentration_20_v1',
    'cyq_median_change_autocorr_20_v1',
    'cyq_width_change_autocorr_20_v1',
    'cyq_hhi_change_autocorr_20_v1',
)

def _shift(x, k=1, axis=0):
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan)
    if axis == 0:
        if k < x.shape[0]:
            out[k:] = x[:-k]
    elif axis == 1:
        if k < x.shape[1]:
            out[:, k:] = x[:, :-k]
    else:
        raise ValueError(axis)
    return out

def _roll(ctx, x):
    return ctx.roll_mean(np.asarray(x, dtype=np.float64), WINDOW, min_count=MIN_COUNT)

def _get(ctx, getter, field, log_adj=None):
    x = np.asarray(getter(ctx, field), dtype=np.float64)
    if x.shape != ctx.panel.shape:
        raise ValueError(f'{field} shape {x.shape} != {ctx.panel.shape}')
    if field in LEVEL_FIELDS:
        if log_adj is None:
            log_adj = _adjustment(ctx)
        x = x + log_adj
    return x

def _adjustment(ctx):
    adj = np.asarray(ctx.px_raw('adj_factor'), dtype=np.float64)
    if adj.shape != ctx.panel.shape:
        raise ValueError('adj_factor shape differs from panel')
    valid = np.isfinite(adj) & (adj > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(valid, np.log(np.where(valid, adj, 1.0)), np.nan)

def _nanmean_axis0(x):
    x = np.asarray(x, dtype=np.float64)
    good = np.isfinite(x)
    n = np.sum(good, axis=0)
    return np.divide(np.sum(np.where(good, x, 0.0), axis=0), n, out=np.full(x.shape[1:], np.nan, dtype=np.float64), where=n > 0)

def _path_efficiency(ctx, change):
    num = ctx.roll_sum(change, WINDOW, min_count=MIN_COUNT)
    den = ctx.roll_sum(np.abs(change), WINDOW, min_count=MIN_COUNT)
    valid = np.isfinite(num) & np.isfinite(den)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.divide(num, den, out=np.full(np.broadcast(num, den).shape, np.nan), where=valid & (den > 1e-12))
    return np.where(valid & (den <= 1e-12), 0.0, out)

def _ratio(a, b):
    with np.errstate(invalid='ignore', divide='ignore'):
        valid = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
        return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=valid)

def _rolling_corr(ctx, x, y):
    return ctx.roll_corr(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), WINDOW, min_count=MIN_COUNT)

def _impl(ctx, getter):
    # A raw-panel request performs source-to-grid placement. All levels in a
    # single pack use the same contemporaneous adjustment snapshot, so place
    # it once rather than repeating the large join for every cost quantile.
    log_adj = _adjustment(ctx)
    g = lambda f: _get(ctx, getter, f, log_adj)
    q = {v: g(f'q{v:02d}' if v in (5, 95) else f'log_q{v:02d}') for v in (5, 10, 25, 50, 75, 90, 95)}
    med = q[50]
    mean = g('log_mean')
    sd = g('log_std')
    skew = g('log_skew')
    kurt = g('log_kurt')
    hhi = g('hhi')
    ent = g('entropy')
    maxm = g('max_mass')
    peaks = g('n_peaks')
    upper = q[90] - med
    lower = med - q[10]
    iqr = q[75] - q[25]
    qfull = np.stack([g(f) for f in QFIELDS], axis=0)
    dq = qfull - _shift(qfull, 1, axis=1)
    transport = _nanmean_axis0(np.abs(dq))
    signed_transport = _nanmean_axis0(dq)
    med_change = med - _shift(med)
    width = iqr - _shift(iqr)
    daily = {
        'cyq_log_median_drift_20_v1': _roll(ctx, med_change),
        'cyq_log_median_acceleration_20_v1': _roll(ctx, med_change) - _roll(ctx, _shift(med_change, 5)),
        'cyq_log_median_path_efficiency_20_v1': _path_efficiency(ctx, med_change),
        'cyq_log_median_reversal_20_v1': _ratio(_roll(ctx, med_change) - med_change, sd),
        'cyq_log_iqr_expansion_20_v1': _roll(ctx, width),
        'cyq_log_upper_tail_buildup_20_v1': _roll(ctx, upper - _shift(upper, 1)),
        'cyq_log_lower_tail_buildup_20_v1': _roll(ctx, lower - _shift(lower, 1)),
        'cyq_log_tail_imbalance_change_20_v1': _roll(ctx, upper - lower - _shift(upper - lower, 1)),
        'cyq_log_dispersion_drift_20_v1': _roll(ctx, sd - _shift(sd, 1)),
        'cyq_log_mean_median_gap_drift_20_v1': _roll(ctx, mean - med - _shift(mean - med, 1)),
        'cyq_log_skew_migration_20_v1': _roll(ctx, skew - _shift(skew, 1)),
        'cyq_log_tail_weight_migration_20_v1': _roll(ctx, kurt - _shift(kurt, 1)),
        'cyq_hhi_concentration_drift_20_v1': _roll(ctx, hhi - _shift(hhi, 1)),
        'cyq_entropy_reorganization_20_v1': _roll(ctx, ent - _shift(ent, 1)),
        'cyq_peak_mass_release_20_v1': -_roll(ctx, maxm - _shift(maxm, 1)),
        'cyq_peak_fragmentation_20_v1': _roll(ctx, peaks - _shift(peaks, 1)),
        'cyq_peak_topology_churn_20_v1': _roll(ctx, np.abs(peaks - _shift(peaks, 1))),
        'cyq_cost_shape_transport_20_v1': _roll(ctx, transport),
        'cyq_cost_shape_directional_transport_20_v1': _roll(ctx, signed_transport),
        'cyq_width_change_persistence_20_v1': _roll(ctx, np.sign(width)),
        'cyq_shape_expansion_concentration_20_v1': _rolling_corr(ctx, sd - _shift(sd, 1), hhi - _shift(hhi, 1)),
        'cyq_median_change_autocorr_20_v1': _rolling_corr(ctx, med_change, _shift(med_change, 1)),
        'cyq_width_change_autocorr_20_v1': _rolling_corr(ctx, width, _shift(width, 1)),
        'cyq_hhi_change_autocorr_20_v1': _rolling_corr(ctx, hhi - _shift(hhi, 1), _shift(hhi - _shift(hhi, 1), 1)),
    }
    return daily

def compute(ctx, getter, variant: str) -> np.ndarray:
    if variant not in NAMES:
        raise KeyError(variant)
    result = _impl(ctx, getter)[variant]
    return np.asarray(result, dtype=np.float64)
