"""R15 cross-stock information diffusion / network family (seat_network batch B2).

One parameterized builder, five economically independent signals over a shared
monthly-frozen return-correlation network.  All network estimation is frozen at
the last session u strictly before the output month (window = the 126 sessions
ending at u); every factor leg on day t uses only returns/turnover up to t-1.
No forward returns are read anywhere in this module.

Signals
-------
leadlag_spillover        Directed lead-lag diffusion: leaders' past 5-session
                         return weighted by their positive day-ahead predictive
                         correlation -> own continuation (declared positive).
peer_relative_reversal   Own trailing 20-session mean return versus the
                         correlation-top-k peer average -> laggard catch-up
                         (declared positive on peer minus own).
peer_flow_diffusion      Peer signed-turnover flow diffusion; the own-stock
                         flow leg is subtracted to separate own event intensity
                         from peer propagation (declared positive).
leadership_continuation  Lead-lag asymmetry index (own leads minus own lags)
                         interacted with own 5-session return; information
                         leaders' moves persist (declared positive).
centrality_growth        Three-month growth of mean absolute correlation to the
                         centrality-top-100 peers; rising co-movement = crowding
                         -> declared NEGATIVE (higher_is_better=False).
"""
from __future__ import annotations

import numpy as np

SIGNALS = (
    'leadlag_spillover',
    'peer_relative_reversal',
    'peer_flow_diffusion',
    'leadership_continuation',
    'centrality_growth',
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
LEAD_MIN_VALID = 5          # leadlag_spillover: valid leaders per day
PEER_MIN_VALID = 10         # peer_relative_reversal: valid peers per day
FLOW_MIN_VALID = 5          # peer_flow_diffusion: valid peers per day
LEAD_POOL = 40              # leadership index pool size by |L|+|L^T|
CENTRALITY_POOL = 100       # centrality peer pool
CENTRALITY_MIN_VALID = 50   # centrality minimum valid peers
CENTRALITY_LAG_MONTHS = 3   # centrality change horizon in months
RET5_MIN = 4                # trailing 5-session mean minimum count |own
RET20_MIN = 16              # trailing 20-session mean minimum count


def _minus_months(tag: int, n: int) -> int:
    year, month = divmod(int(tag), 100)
    total = year * 12 + (month - 1) - n
    return (total // 12) * 100 + (total % 12) + 1


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
    """Row-wise top-k of mat (NaN treated as -inf).  Returns (idx, values).

    Selection uses argpartition (order inside the k chosen columns is
    unspecified but deterministic for identical input); downstream uses only
    sets and summed weights, never positional order.
    """
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


def _trailing_mean(r: np.ndarray, window: int, min_count: int) -> np.ndarray:
    """Mean of r over [t-window, t-1] (exclusive of t), NaN if count < min_count."""
    T, N = r.shape
    fin = np.isfinite(r)
    v = np.where(fin, r, 0.0)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(v, axis=0)])
    cn = np.vstack([np.zeros((1, N)), np.cumsum(fin, axis=0)])
    rows = np.arange(T)
    lo = np.maximum(rows - window, 0)
    s = cs[rows] - cs[lo]
    c = cn[rows] - cn[lo]
    with np.errstate(invalid='ignore', divide='ignore'):
        out = s / c
    out[c < min_count] = np.nan
    return out


def _flow_z(r: np.ndarray, turn: np.ndarray, min_stocks: int) -> np.ndarray:
    """Daily cross-sectional z of signed turnover s = sign(r)*turnover."""
    s = np.sign(r) * turn
    T, N = r.shape
    z = np.full((T, N), np.nan)
    for t in range(T):
        row = s[t]
        fin = np.isfinite(row)
        if int(fin.sum()) < min_stocks:
            continue
        mu = row[fin].mean()
        sd = row[fin].std(ddof=0)
        if not sd > 0:
            continue
        z[t, fin] = (row[fin] - mu) / sd
    return z


def _network(r, univ, u_idx, corr_window, min_obs, lag_min_obs, top_k,
             centrality_pool, centrality_min_valid, lead_pool):
    """Monthly-frozen network estimated on the 126 sessions ending at u_idx."""
    lo = u_idx - corr_window + 1
    Rw = r[lo:u_idx + 1]
    obs = np.isfinite(Rw).sum(axis=0)
    sel = np.asarray(univ[u_idx], dtype=bool) & (obs >= min_obs)

    C = _pairwise(Rw, Rw, min_obs)
    C = np.where(sel[:, None] & sel[None, :], C, np.nan)
    np.fill_diagonal(C, np.nan)
    absC = np.abs(C)
    top_idx, top_val = _topk(absC, top_k)
    cent_idx, cent_val = _topk(absC, centrality_pool)
    cent_cnt = np.isfinite(cent_val).sum(axis=1)
    cent_sum = np.nansum(np.abs(cent_val), axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cmean = np.where(cent_cnt >= centrality_min_valid,
                         cent_sum / np.maximum(cent_cnt, 1), np.nan)

    Xl = r[lo:u_idx]
    Yl = r[lo + 1:u_idx + 1]
    L = _pairwise(Xl, Yl, lag_min_obs)          # L[leader j, follower i]
    L = np.where(sel[:, None] & sel[None, :], L, np.nan)
    np.fill_diagonal(L, np.nan)
    Lf = L.T                                     # Lf[follower i, leader j]
    lead_pos = np.where(np.isfinite(Lf) & (Lf > 0), Lf, np.nan)
    lead_idx, lead_val = _topk(lead_pos, top_k)

    # Own-leads-peer for stock i over pool member m is L[i, m] (i at s, m at s+1);
    # peer-leads-own is Lf[i, m] = L[m, i].  The pool ranks by the larger magnitude
    # of the two directions.  NOTE: an earlier revision gathered L.T for BOTH legs
    # (L.T is exactly Lf), which made lead_sum == lag_sum and the index identically
    # zero; fixed here to gather L and Lf (implementation bug vs the frozen formula).
    with np.errstate(invalid='ignore'):
        pair = np.maximum(np.abs(L), np.abs(Lf))
    pool_idx, pool_val = _topk(pair, lead_pool)
    pool_ok = np.isfinite(pool_val)
    g1 = np.where(pool_ok, np.take_along_axis(L, pool_idx, axis=1), np.nan)
    g2 = np.where(pool_ok, np.take_along_axis(Lf, pool_idx, axis=1), np.nan)
    lead_sum = np.nansum(np.maximum(np.where(np.isfinite(g1), g1, 0.0), 0.0), axis=1)
    lag_sum = np.nansum(np.maximum(np.where(np.isfinite(g2), g2, 0.0), 0.0), axis=1)
    total = lead_sum + lag_sum
    with np.errstate(divide='ignore', invalid='ignore'):
        leadership = (lead_sum - lag_sum) / total
    leadership[total < 0.5] = np.nan

    return {'sel': sel, 'top_idx': top_idx, 'top_val': top_val,
            'lead_idx': lead_idx, 'lead_val': lead_val,
            'leadership': leadership, 'cmean': cmean}


def _gather(vec: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """vec (N,) gathered along idx (N, k); invalid (-1) slots give NaN."""
    valid = idx >= 0
    g = vec[np.where(valid, idx, 0)]
    return np.where(valid, g, np.nan)


def _wmean(g: np.ndarray, w: np.ndarray):
    """Row-wise weighted mean with weights zeroed on non-finite g."""
    ww = np.where(np.isfinite(g), w, 0.0)
    tot = ww.sum(axis=1)
    cnt = np.isfinite(g).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (np.where(np.isfinite(g), g, 0.0) * ww).sum(axis=1) / tot
    out[~(tot > 0)] = np.nan
    return out, cnt


def panel_values(r, univ, turn, days, codes, months, parameters):
    """Compute one signal's (T, C) float32 panel plus a small qa record.

    r: (T, C) float64 traded-masked returns; turn only for peer_flow_diffusion.
    Network for each output month is frozen at the last session of the previous
    month; every signal leg is trailing (ends at t-1).
    """
    if parameters not in ALLOWED_VARIANTS:
        raise ValueError('Parameters are not a frozen allowed variant')
    signal = parameters['signal']
    cw = int(parameters['corr_window'])
    min_obs = int(parameters['min_obs'])
    lag_min_obs = int(parameters['lag_min_obs'])
    top_k = int(parameters['top_k'])
    min_stocks = int(parameters['min_stocks'])
    first = int(parameters['first_month'])

    r = np.asarray(r, dtype=np.float64)
    T, N = r.shape
    if len(days) != T or len(codes) != N:
        raise ValueError('days/codes shape mismatch')
    if r.shape != np.asarray(univ).shape:
        raise ValueError('universe shape mismatch')

    need_flow = signal == 'peer_flow_diffusion'
    if signal == 'centrality_growth':
        c_hist = {}
    values = np.full((T, N), np.nan, dtype=np.float64)

    R5 = _trailing_mean(r, 5, RET5_MIN)
    R20 = _trailing_mean(r, 20, RET20_MIN) if signal == 'peer_relative_reversal' else None
    if need_flow:
        if turn is None:
            raise ValueError('turnover required for peer_flow_diffusion')
        Z5 = _trailing_mean(_flow_z(r, np.asarray(turn, dtype=np.float64), min_stocks),
                            5, RET5_MIN)

    qa = {'months': 0, 'months_requested': 0, 'donor_mean': 0.0}
    for tag, d0, end in _month_runs(days):
        u_idx = d0 - 1
        if u_idx < cw - 1:
            continue
        net = _network(r, univ, u_idx, cw, min_obs, lag_min_obs, top_k,
                       CENTRALITY_POOL, CENTRALITY_MIN_VALID, LEAD_POOL)
        qa['months'] += 1
        if tag < first:
            if signal == 'centrality_growth':
                c_hist[tag] = net['cmean']
            continue
        qa['months_requested'] += 1
        qa['donor_mean'] += float(net['sel'].sum())
        rows = np.arange(d0, end)
        if signal == 'leadlag_spillover':
            lw = np.where(np.isfinite(net['lead_val']) & (net['lead_val'] > 0),
                          net['lead_val'], 0.0)
            for t in rows:
                g = _gather(R5[t], net['lead_idx'])
                g = np.where(np.isfinite(net['lead_val']), g, np.nan)
                out, cnt = _wmean(g, lw)
                out[cnt < LEAD_MIN_VALID] = np.nan
                values[t] = np.where(net['sel'], out, np.nan)
        elif signal == 'peer_relative_reversal':
            uw = np.where(np.isfinite(net['top_val']), np.abs(net['top_val']), 0.0)
            for t in rows:
                g = _gather(R20[t], net['top_idx'])
                g = np.where(np.isfinite(net['top_val']), g, np.nan)
                peer, cnt = _wmean(g, uw)
                out = peer - R20[t]
                out[cnt < PEER_MIN_VALID] = np.nan
                values[t] = np.where(net['sel'], out, np.nan)
        elif signal == 'peer_flow_diffusion':
            uw = np.where(np.isfinite(net['top_val']), np.abs(net['top_val']), 0.0)
            for t in rows:
                g = _gather(Z5[t], net['top_idx'])
                g = np.where(np.isfinite(net['top_val']), g, np.nan)
                peer, cnt = _wmean(g, uw)
                out = peer - Z5[t]
                out[cnt < FLOW_MIN_VALID] = np.nan
                values[t] = np.where(net['sel'], out, np.nan)
        elif signal == 'leadership_continuation':
            for t in rows:
                out = net['leadership'] * R5[t]
                values[t] = np.where(net['sel'], out, np.nan)
        elif signal == 'centrality_growth':
            c_hist[tag] = net['cmean']
            prev = c_hist.get(_minus_months(tag, CENTRALITY_LAG_MONTHS))
            if prev is None:
                continue
            delta = net['cmean'] - prev
            values[rows] = np.where(net['sel'][None, :], delta[None, :], np.nan)
    qa['donor_mean'] = (qa['donor_mean'] / qa['months_requested']
                        if qa['months_requested'] else 0.0)
    return values.astype(np.float32), qa


def _higher_is_better(signal: str) -> bool:
    return signal != 'centrality_growth'


def build(parameters):
    """Candidate entry point: returns compute(ctx) -> (T, C) float32 panel."""
    if parameters not in ALLOWED_VARIANTS:
        raise ValueError('Only frozen B2 variants are allowed')
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
        turn = None
        if signal == 'peer_flow_diffusion':
            turn = np.array(ctx.px('turnover'), dtype=np.float64, copy=True)
            if turn.shape != r.shape:
                raise ValueError('Turnover shape mismatch')
        months = np.unique(days[days >= parameters['first_month'] * 100 + 1] // 100)
        values, _ = panel_values(r, univ, turn, days, codes, months, parameters)
        if np.isinf(values).any():
            raise ValueError('Infinite network factor value')
        return values

    return compute
