"""P013 frozen design: restricted (capped) first-passage gain/loss waiting time.

GC descriptive factor; no future-return ordering claimed. Formula frozen in
artifacts/research/round14_20261007/seat1_price/directions_batch3.json (P013)
plus batch3_probe_contract_amendment.json; probe reference implementation
probe_p013_p014_v1.py::p13. R15 full-axis contract:
artifacts/research/round15_20261007/seat_price/p013_fullaxis_contract_v1.json.
Capping at 60 sessions measures restricted waiting time; an uncrossed barrier
is not an actual crossing.
"""
import numpy as np

PARAMETERS = {
    'barrier_gain': 0.05,
    'barrier_loss': -0.05,
    'max_wait': 60,
    'cohorts': 252,
    'first_start_lag': 311,
    'last_start_lag': 60,
    'min_valid_cohorts': 200,
    'endpoint_rule': 'traded_and_raw_close_finite_positive',
    'signal': 'mean_valid_tau_plus_minus_tau_minus',
    'censoring': 'capped_at_60_not_infinite_wait',
}


def compute_full(hfq_close, raw_close, traded, universe, rows):
    """Return (values, diagnostics) on grid `rows`; identical core to the probe.

    values: mean over valid cohorts of (tau_plus - tau_minus), NaN when fewer
    than min_valid_cohorts cohorts are valid or t is not traded/universe.
    diagnostics: per-output-cell capped shares and valid-cohort counts.
    """
    pg = int(PARAMETERS['cohorts'])
    max_wait = int(PARAMETERS['max_wait'])
    first_lag = int(PARAMETERS['first_start_lag'])
    last_lag = int(PARAMETERS['last_start_lag'])
    min_valid = int(PARAMETERS['min_valid_cohorts'])
    p = np.log(np.where(hfq_close > 0, hfq_close, np.nan))
    n, k = hfq_close.shape
    valid = np.ones((n - max_wait, k), dtype=bool)
    raw_ok = np.isfinite(raw_close) & (raw_close > 0)
    for lag in range(max_wait + 1):
        valid &= (traded[lag:n - max_wait + lag]
                  & np.isfinite(p[lag:n - max_wait + lag])
                  & raw_ok[lag:n - max_wait + lag])
    tp = np.full((n - max_wait, k), float(max_wait))
    tm = np.full((n - max_wait, k), float(max_wait))
    hitp = np.zeros((n - max_wait, k), dtype=bool)
    hitm = np.zeros((n - max_wait, k), dtype=bool)
    for lag in range(1, max_wait + 1):
        d = p[lag:n - max_wait + lag] - p[:n - max_wait]
        hp = d >= PARAMETERS['barrier_gain']
        hm = d <= PARAMETERS['barrier_loss']
        tp[hp & ~hitp] = lag
        tm[hm & ~hitm] = lag
        hitp |= hp
        hitm |= hm
    diff = tp - tm
    out = np.full((len(rows), k), np.nan)
    capped_plus = np.full((len(rows), k), np.nan)
    capped_minus = np.full((len(rows), k), np.nan)
    valid_counts = np.full((len(rows), k), np.nan)
    for i, t in enumerate(rows):
        sl = slice(int(t) - first_lag, int(t) - last_lag + 1)
        g = valid[sl]
        ct = g.sum(0)
        out[i] = np.where(ct >= min_valid,
                          np.sum(np.where(g, diff[sl], 0.0), axis=0) / np.maximum(ct, 1),
                          np.nan)
        denom = np.maximum(ct, 1)
        capped_plus[i] = np.where(ct > 0, np.sum(g & ~hitp[sl], 0) / denom, np.nan)
        capped_minus[i] = np.where(ct > 0, np.sum(g & ~hitm[sl], 0) / denom, np.nan)
        valid_counts[i] = ct
    out[~traded[rows]] = np.nan
    out[~universe[rows]] = np.nan
    return out, {'capped_plus_share': capped_plus,
                 'capped_minus_share': capped_minus,
                 'valid_cohorts': valid_counts}


def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only frozen P013 parameters accepted")
    def compute(ctx):
        days = np.asarray(ctx.panel.dates, dtype=np.int64)
        n = len(days)
        out = np.full(ctx.panel.shape, np.nan)
        rows = np.arange(int(PARAMETERS['first_start_lag']), n)
        if len(rows):
            hfq = np.asarray(ctx.hfq("close"), dtype=np.float64)
            raw = np.asarray(ctx.px_raw("close"), dtype=np.float64)
            tr = np.asarray(ctx.traded(), dtype=bool)
            u = np.asarray(ctx.universe, dtype=bool)
            values, _ = compute_full(hfq, raw, tr, u, rows)
            out[rows] = values
        return out
    return compute
