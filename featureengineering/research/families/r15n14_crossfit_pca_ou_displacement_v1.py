"""N014 cross-fitted monthly PCA-OU residual equilibrium displacement (pure builder).

Frozen definition: artifacts/research/round14_20261007/seat5_network/directions_batch3.json#N014
Frozen implementation copied operation-for-operation from
artifacts/research/round14_20261007/seat5_network/probe_n014_pca_ou_v4.py
(sha256 a4b7d409240158daf645e4b8e79c7728037277c2161e9b72a28d857231208991) and its
full-axis driver build_n014_full_v41.py (sha256 5a485b83173b4ef0e6bca6a39994d0205d84ed3782dc753047c569e310307e02).
Only the input adapter changes: all inputs arrive through ctx (no disk/cache reads).
Every return requires traded[d-1] AND traded[d] (frozen return amendment); no
compression of suspended intervals, no zero fill, output axis fixed 2115 codes.
"""
import numpy as np

PARAMETERS = {'signal': 'crossfit_pca_ou_displacement', 'first_month': 201801, 'warmup_days': 600}


def ou_panel(e, current):
    x = np.vstack([np.zeros(e.shape[1]), np.cumsum(e, axis=0)]); xp = x[:-1]; yp = x[1:]
    xm = xp.mean(0); ym = yp.mean(0); dx = xp - xm; dy = yp - ym
    with np.errstate(all='ignore'):
        b = (dx * dy).sum(0) / (dx * dx).sum(0); a = ym - b * xm
        v = ((yp - a - b * xp) ** 2).sum(0) / 58; tau = -1 / np.log(b); m = a / (1 - b); disp = m - x[-1] - current; score = disp / np.sqrt(v / (1 - b * b))
    good = (b > 0) & (b < 1) & (tau < 30) & (v > 1e-16) & np.isfinite(score)
    return np.where(good, score, np.nan), np.where(good, disp, np.nan), np.where(good, tau, np.nan)


def panel_values(r, univ, days, codes, target_ids, months):
    """Operation order copied verbatim from the frozen probe; returns (value, controls, qa)."""
    T, N = r.shape; values = np.full((T, N), np.nan, np.float32); disp = values.copy(); tau = values.copy(); control = values.copy(); own60 = values.copy(); qa = []
    order = np.argsort(codes, kind='stable'); fold = np.empty(N, int); fold[order] = np.arange(N) % 5
    sums = np.nansum(r, axis=1); counts = np.isfinite(r).sum(axis=1)
    for month in months:
        ds = np.flatnonzero(days // 100 == month); u = int(ds[0]) - 1
        for f in range(5):
            target = np.array([i for i in target_ids if fold[i] == f], int)
            if not len(target): continue
            # Require all 252 prior sessions before slicing: negative indices
            # can produce an empty window or wrap into later panel data.
            if u < 251:
                qa.append({'month': int(month), 'fold': f, 'donors': 0})
                continue
            train = r[u - 251:u + 1]; good = (fold != f) & univ[u] & np.isfinite(train).all(0) & (np.std(train, axis=0) > 0)
            donors = np.flatnonzero(good); qa.append({'month': int(month), 'fold': f, 'donors': len(donors)})
            if len(donors) < 500: continue
            y = train[:, donors]; mu = y.mean(0); sd = y.std(0, ddof=1); z = (y - mu) / sd
            vals, vec = np.linalg.eigh(z @ z.T); ev = vals[-5:]
            if not np.isfinite(ev).all() or ev[0] <= 1e-12: continue
            L = z.T @ vec[:, -5:] / np.sqrt(ev)[None, :]
            start = int(ds[0]) - 60; end = int(ds[-1]) + 1; F = np.full((end - start, 5), np.nan)
            for d in range(start, end):
                now = (r[d, donors] - mu) / sd; ok = np.isfinite(now)
                if ok.sum() < max(500, int(np.ceil(.9 * len(donors)))): continue
                fit, _, rank, _ = np.linalg.lstsq(L[ok], now[ok], rcond=None)
                if rank == 5: F[d - start] = fit
            for d in ds:
                pos = d - start; fh = F[pos - 60:pos]; fc = F[pos]
                if not np.isfinite(fh).all() or not np.isfinite(fc).all(): continue
                hist = r[d - 60:d, target]; valid = np.isfinite(hist).all(0) & np.isfinite(r[d, target]) & univ[d, target]
                ids = target[valid]
                if not len(ids): continue
                A = np.column_stack([np.ones(60), fh]); Y = hist[:, valid]
                beta, _, rank, _ = np.linalg.lstsq(A, Y, rcond=None)
                if rank != 6: continue
                e = Y - A @ beta; current = r[d, ids] - np.r_[1., fc] @ beta
                score, dd, tt = ou_panel(e, current); values[d, ids] = score; disp[d, ids] = dd; tau[d, ids] = tt
                own60[d, ids] = np.prod(1 + r[d - 59:d + 1, ids], axis=0) - 1
                # Same OU kernel and 60-day estimation with one LOO market factor.
                M = (sums[d - 60:d, None] - Y) / (counts[d - 60:d, None] - 1)
                mc = (sums[d] - r[d, ids]) / (counts[d] - 1)
                if np.min(counts[d - 60:d + 1] - 1) < 100: continue
                dm = M - M.mean(0); dy = Y - Y.mean(0)
                with np.errstate(all='ignore'):
                    b = (dm * dy).sum(0) / (dm * dm).sum(0); a = Y.mean(0) - b * M.mean(0)
                ee = Y - a - b * M; cur = r[d, ids] - a - b * mc
                control[d, ids] = ou_panel(ee, cur)[0]
    return values, {'unscaled_displacement': disp, 'tau': tau, 'own60return': own60, 'single_market_ou': control}, qa


def build(parameters):
    """Candidate entry point: returns compute(ctx) -> (T, C) float32 panel."""
    if parameters != PARAMETERS:
        raise ValueError('Only the frozen N014 parameters')
    def compute(ctx):
        days = np.asarray(ctx.panel.dates, dtype=np.int64)
        codes = np.asarray(ctx.panel.codes).astype(str)
        if len(codes) != 2115:
            raise ValueError('Frozen 2115 stock axis required')
        r = np.array(ctx.ret_clean(1), dtype=float, copy=True)
        tr = np.asarray(ctx.traded(), dtype=bool)
        adj = tr.copy(); adj[0] = False; adj[1:] &= tr[:-1]; r[~adj] = np.nan
        univ = np.asarray(ctx.universe, dtype=bool)
        if univ.shape != r.shape:
            raise ValueError('Universe shape mismatch')
        months = np.unique(days[days >= PARAMETERS['first_month'] * 100 + 1] // 100)
        values = panel_values(r, univ, days, codes, np.arange(len(codes)), months)[0]
        if np.isinf(values).any():
            raise ValueError('Infinite N014 value')
        return values
    return compute
