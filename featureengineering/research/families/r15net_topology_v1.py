"""R15 peer-network topology description family (seat_network batch B3). GC channel.

Four DESCRIPTIVE structural objects of the monthly-frozen return-correlation
network.  No forward returns are read anywhere in this module and no predictive
direction is declared: these are ex-ante economic description factors (GC), per
SKILL.md §5 the GC channel grades quality (G2/G3) and structure/economy, not
predictive significance.

Objects
-------
mc_neutral_density    Size-neutralized network density: the stock's mean |corr|
                      to its correlation-top-k peers, residualized cross-
                      sectionally on log market cap at the freeze date.
degree_centrality     Degree centrality: share of eligible peers with
                      |corr| >= DEGREE_THRESHOLD (breadth of strong links).
leadlag_net_ratio     Net lead-lag structural ratio over the full eligible
                      graph: (own-leads sum - follows sum) / (sum), built from
                      the window lead-lag correlations L[j, i] = corr(r_j[s],
                      r_i[s+1]) with positive parts only.  This is the directed
                      STRUCTURE of the correlation network; it makes no return
                      prediction (contrast: the closed B2 leadlag_spillover /
                      leadership directions used returns of leaders).
corr_stability        Temporal stability of the correlation profile: Pearson
                      correlation, over common eligible peers, between the
                      pairwise correlations estimated on the first and second
                      halves of the window.

All networks are frozen at u = the last session strictly before the output
month (window = the corr_window sessions ending at u); every descriptor is a
constant per month.  Only information available at u is used.
"""
from __future__ import annotations

import numpy as np

SIGNALS = (
    'mc_neutral_density',
    'degree_centrality',
    'leadlag_net_ratio',
    'corr_stability',
)

BASE_PARAMETERS = {
    'first_month': 201801,
    'warmup_days': 600,
    'corr_window': 126,
    'min_obs': 100,
    'lag_min_obs': 80,
    'top_k': 20,
    'min_stocks': 100,
}


def variant(signal: str) -> dict:
    if signal not in SIGNALS:
        raise ValueError('Unknown signal ' + signal)
    p = dict(BASE_PARAMETERS)
    p['signal'] = signal
    return p


ALLOWED_VARIANTS = [variant(s) for s in SIGNALS]

# Frozen structural constants (no parameterization).
DENSITY_MIN_VALID = 10        # density: valid top-k peer slots required
DEGREE_THRESHOLD = 0.5        # degree centrality: |corr| >= threshold edge
DEGREE_MIN_VALID = 100        # degree: valid peers required for the denominator
LEADLAG_MIN_VALID = 10        # leadlag: finite lead legs and lag legs required
LEADLAG_MIN_TOTAL = 1e-12     # leadlag: total (lead+lag) floor for the ratio
STABILITY_HALF_OBS = 50       # stability: per-half window observations per stock
STABILITY_MIN_VALID = 50      # stability: common finite peers required
NEUTRALIZE_MIN_STOCKS = 100   # density neutralization: cross-section size floor

# ---------------------------------------------------------------------------
# Shared helpers.  Deliberately duplicated (not imported) from
# r15net_diffusion_v1.py so this module's frozen source hash fully determines
# its behaviour even if the B2 module ever changes.
# ---------------------------------------------------------------------------


def _pairwise(X: np.ndarray, Y: np.ndarray, min_n: int) -> np.ndarray:
    """Pairwise-complete Pearson correlation.

    Returns corr[j, i] = corr(X[:, j], Y[:, i]) over rows where both are finite;
    NaN where the common count < min_n or either variance is non-positive.
    """
    mx = np.isfinite(X)
    my = np.isfinite(Y)
    X0 = np.where(mx, X, 0.0)
    Y0 = np.where(my, Y, 0.0)
    P = mx.astype(np.float64)
    Q = my.astype(np.float64)
    n = P.T @ Q
    sx = X0.T @ Q
    sy = P.T @ Y0
    sxx = (X0 * X0).T @ Q
    syy = P.T @ (Y0 * Y0)
    sxy = X0.T @ Y0
    with np.errstate(divide='ignore', invalid='ignore'):
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        cov = sxy - sx * sy / n
        corr = cov / np.sqrt(vx * vy)
    corr[(n < min_n) | ~(vx > 0) | ~(vy > 0)] = np.nan
    return corr


def _topk(mat: np.ndarray, k: int):
    """Row-wise top-k of mat (NaN treated as -inf).  Returns (idx, values)."""
    N = mat.shape[0]
    kk = min(int(k), N - 1)
    filled = np.where(np.isfinite(mat), mat, -np.inf)
    idx = np.argpartition(-filled, kk - 1, axis=1)[:, :kk]
    vals = np.take_along_axis(mat, idx, axis=1)
    return idx.astype(np.int32), vals


def _month_runs(days: np.ndarray):
    """[(tag, d0, end)] for each contiguous run of equal YYYYMM tag."""
    tags = np.asarray(days, dtype=np.int64) // 100
    out = []
    start = 0
    for i in range(1, len(tags) + 1):
        if i == len(tags) or tags[i] != tags[start]:
            out.append((int(tags[start]), start, i))
            start = i
    return out


# ---------------------------------------------------------------------------
# Structural objects (each a pure function of the frozen window, or of halves)
# ---------------------------------------------------------------------------


def _density(absC: np.ndarray, sel: np.ndarray, top_k: int, min_valid: int) -> np.ndarray:
    """Mean |corr| over the top-k peers (raw, before size neutralization)."""
    _idx, vals = _topk(absC, top_k)
    ok = np.isfinite(vals)
    cnt = ok.sum(axis=1)
    s = np.where(ok, np.abs(vals), 0.0).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(cnt >= min_valid, s / np.maximum(cnt, 1), np.nan)
    out = np.where(sel, out, np.nan)
    return out


def _neutralize(raw: np.ndarray, logmv_u: np.ndarray, sel: np.ndarray,
                min_stocks: int) -> np.ndarray:
    """Cross-sectional OLS residual of raw on [1, logmv_u] at the freeze date.

    Regression sample: sel & both inputs finite & count >= min_stocks.
    Stocks outside the sample get NaN.  Deterministic (normal equations solved
    by lstsq).
    """
    m = sel & np.isfinite(raw) & np.isfinite(logmv_u)
    out = np.full(raw.shape, np.nan, dtype=np.float64)
    if int(m.sum()) < min_stocks:
        return out
    x = logmv_u[m]
    y = raw[m]
    X = np.column_stack([np.ones_like(x), x])
    beta, _res, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    out[m] = y - X @ beta
    return out


def _degree(absC: np.ndarray, sel: np.ndarray, threshold: float,
            min_valid: int) -> np.ndarray:
    """Share of eligible peers with |corr| >= threshold."""
    valid = np.isfinite(absC) & sel[:, None] & sel[None, :]
    strong = valid & (absC >= threshold)
    nv = valid.sum(axis=1)
    ns = strong.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(nv >= min_valid, ns / np.maximum(nv, 1), np.nan)
    out = np.where(sel, out, np.nan)
    return out


def _leadlag(L: np.ndarray, sel: np.ndarray, min_valid: int,
             min_total: float) -> np.ndarray:
    """Full-graph net lead-lag ratio from L[j, i] = corr(r_j[s], r_i[s+1]).

    own leads j: L[i, j] with L > 0 (row sum);  j leads own: L[j, i] with L > 0
    (column sum).  Requires >= min_valid finite legs on each side; NaN if the
    total is below min_total (degenerate).
    """
    fin = np.isfinite(L)
    pos = np.where(fin & (L > 0), L, 0.0)
    lead = pos.sum(axis=1)
    lag = pos.sum(axis=0)
    rows_ok = fin.sum(axis=1) >= min_valid
    cols_ok = fin.sum(axis=0) >= min_valid
    total = lead + lag
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (lead - lag) / total
    out[~(total > min_total)] = np.nan
    out[~(rows_ok & cols_ok)] = np.nan
    out = np.where(sel, out, np.nan)
    return out


def _halves_corr(r: np.ndarray, u_idx: int, corr_window: int,
                 half_obs: int):
    """Half-window pairwise correlations plus eligibility."""
    half = corr_window // 2
    lo = u_idx - corr_window + 1
    H1 = r[lo:lo + half]
    H2 = r[lo + half:u_idx + 1]
    C1 = _pairwise(H1, H1, half_obs)
    C2 = _pairwise(H2, H2, half_obs)
    obs1 = np.isfinite(H1).sum(axis=0)
    obs2 = np.isfinite(H2).sum(axis=0)
    return C1, C2, obs1, obs2


def _stability(C1: np.ndarray, C2: np.ndarray, sel: np.ndarray,
               min_valid: int) -> np.ndarray:
    """Row-wise Pearson corr of the two halves' correlation profiles."""
    m = sel[:, None] & sel[None, :] & np.isfinite(C1) & np.isfinite(C2)
    np.fill_diagonal(m, False)
    n = m.sum(axis=1)
    A = np.where(m, C1, 0.0)
    B = np.where(m, C2, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        ma = A.sum(axis=1) / np.maximum(n, 1)
        mb = B.sum(axis=1) / np.maximum(n, 1)
        Za = np.where(m, C1 - ma[:, None], 0.0)
        Zb = np.where(m, C2 - mb[:, None], 0.0)
        cov = (Za * Zb).sum(axis=1) / np.maximum(n, 1)
        va = (Za * Za).sum(axis=1) / np.maximum(n, 1)
        vb = (Zb * Zb).sum(axis=1) / np.maximum(n, 1)
        out = cov / np.sqrt(va * vb)
    out[~(n >= min_valid)] = np.nan
    out[~(va > 0) | ~(vb > 0)] = np.nan
    out = np.where(sel, out, np.nan)
    return out


def _network(r, univ, logmv, u_idx, parameters, need):
    """Monthly-frozen topology objects at u_idx; only `need` pieces computed."""
    cw = int(parameters['corr_window'])
    min_obs = int(parameters['min_obs'])
    lag_min_obs = int(parameters['lag_min_obs'])
    top_k = int(parameters['top_k'])
    min_stocks = int(parameters['min_stocks'])
    lo = u_idx - cw + 1
    Rw = r[lo:u_idx + 1]
    obs = np.isfinite(Rw).sum(axis=0)
    sel = np.asarray(univ[u_idx], dtype=bool) & (obs >= min_obs)

    out = {'sel': sel}
    for name in need:
        if name in ('mc_neutral_density', 'degree_centrality'):
            C = _pairwise(Rw, Rw, min_obs)
            C = np.where(sel[:, None] & sel[None, :], C, np.nan)
            np.fill_diagonal(C, np.nan)
            absC = np.abs(C)
            if name == 'mc_neutral_density':
                raw = _density(absC, sel, top_k, DENSITY_MIN_VALID)
                out['density_raw'] = raw
                out['mc_neutral_density'] = _neutralize(
                    raw, np.asarray(logmv[u_idx], dtype=np.float64), sel,
                    NEUTRALIZE_MIN_STOCKS)
            else:
                out['degree_centrality'] = _degree(
                    absC, sel, DEGREE_THRESHOLD, DEGREE_MIN_VALID)
        elif name == 'leadlag_net_ratio':
            Xl = r[lo:u_idx]
            Yl = r[lo + 1:u_idx + 1]
            L = _pairwise(Xl, Yl, lag_min_obs)
            L = np.where(sel[:, None] & sel[None, :], L, np.nan)
            np.fill_diagonal(L, np.nan)
            out['leadlag_net_ratio'] = _leadlag(
                L, sel, LEADLAG_MIN_VALID, LEADLAG_MIN_TOTAL)
        elif name == 'corr_stability':
            C1, C2, obs1, obs2 = _halves_corr(
                r, u_idx, cw, STABILITY_HALF_OBS)
            selS = np.asarray(univ[u_idx], dtype=bool) & \
                (obs1 >= STABILITY_HALF_OBS) & (obs2 >= STABILITY_HALF_OBS)
            out['corr_stability'] = _stability(
                C1, C2, selS & sel, STABILITY_MIN_VALID)
        else:
            raise ValueError('Unknown piece ' + name)
    return out


def panel_values(r, univ, logmv, days, codes, months, parameters):
    """Compute one object's (T, C) float32 panel plus a small qa record."""
    if parameters not in ALLOWED_VARIANTS:
        raise ValueError('Parameters are not a frozen allowed variant')
    signal = parameters['signal']
    if signal == 'mc_neutral_density' and logmv is None:
        raise ValueError('log market cap required for mc_neutral_density')
    first = int(parameters['first_month'])
    cw = int(parameters['corr_window'])

    r = np.asarray(r, dtype=np.float64)
    T, N = r.shape
    if len(days) != T or len(codes) != N:
        raise ValueError('days/codes shape mismatch')
    if r.shape != np.asarray(univ).shape:
        raise ValueError('universe shape mismatch')

    values = np.full((T, N), np.nan, dtype=np.float64)
    qa = {'months': 0, 'donor_mean': 0.0, 'finite_total': 0}
    for tag, d0, end in _month_runs(days):
        if tag < first:
            continue
        u_idx = d0 - 1
        if u_idx < cw - 1:
            continue
        net = _network(r, univ, logmv, u_idx, parameters, (signal,))
        val = net[signal]
        sel = net['sel']
        values[d0:end] = np.where(sel[None, :], val[None, :], np.nan)
        qa['months'] += 1
        qa['finite_total'] += int(np.isfinite(val).sum())
    qa['donor_mean'] = (qa['finite_total'] / qa['months'] if qa['months'] else 0.0)
    return values.astype(np.float32), qa


# ---------------------------------------------------------------------------


def build(parameters):
    """Candidate entry point: returns compute(ctx) -> (T, C) float32 panel."""
    if parameters not in ALLOWED_VARIANTS:
        raise ValueError('Only frozen B3 variants are allowed')
    signal = parameters['signal']

    def compute(ctx):
        days = np.asarray(ctx.panel.dates, dtype=np.int64)
        codes = np.asarray(ctx.panel.codes).astype(str)
        if len(codes) != 2115:
            raise ValueError('Frozen 2115 stock axis required')
        r = np.array(ctx.ret_clean(1), dtype=np.float64, copy=True)
        tr = np.asarray(ctx.traded(), dtype=bool)
        adj = tr.copy()
        adj[0] = False
        adj[1:] &= tr[:-1]
        r[~adj] = np.nan
        univ = np.asarray(ctx.universe, dtype=bool)
        if univ.shape != r.shape:
            raise ValueError('Universe shape mismatch')
        logmv = None
        if signal == 'mc_neutral_density':
            close = np.asarray(ctx.px('close'), dtype=np.float64)
            tshare = np.asarray(ctx.px('total_share'), dtype=np.float64)
            if close.shape != r.shape or tshare.shape != r.shape:
                raise ValueError('Price layer shape mismatch')
            with np.errstate(divide='ignore', invalid='ignore'):
                mv = close * tshare
                logmv = np.log(np.where(mv > 0, mv, np.nan))
        months = np.unique(days[days >= parameters['first_month'] * 100 + 1] // 100)
        values, _ = panel_values(r, univ, logmv, days, codes, months, parameters)
        if np.isinf(values).any():
            raise ValueError('Infinite topology factor value')
        return values

    return compute
