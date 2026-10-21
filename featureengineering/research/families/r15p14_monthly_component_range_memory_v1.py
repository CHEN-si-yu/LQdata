"""P014 frozen design: HAR-style monthly partial-memory coefficient of daily
range variance (daily HFQ high/low range proxy, NOT high-frequency HAR-RV).

GC descriptive factor; no future-return ordering. Formula frozen in
artifacts/research/round14_20261007/seat1_price/directions_batch3.json (P014)
plus batch3_probe_contract_amendment.json; probe reference implementation
probe_p013_p014_v1.py (design/fit). R15 full-axis contract:
artifacts/research/round15_20261007/seat_price/p014_fullaxis_contract_v1.json.
Estimation boundary: R14 probe showed finite-sample bias/uncertainty (true .35
recovered median .2979, 10-90pct .1693-.4515); not exact recovery.

Numerics note: the batched fast path uses normal equations in standardized
coordinates (prefix sums). On near-constant windows the covariance of the sums
cancels at the roundoff floor and the batched algebra loses precision, so any
cell whose window variance sits at the cancellation floor, whose rank test
fails, or whose condition exceeds _FALLBACK_COND is routed per cell through
_frozen_cell, a verbatim port of the frozen probe per-cell fit. The routing
thresholds are implementation details, not formula parameters; fallback
outcomes carry the same status labels (insufficient/rank_deficient/collinear/
valid) as the frozen reference.
"""
import numpy as np

_LOG2_4 = 4.0 * np.log(2.0)
_FALLBACK_COND = 100.0
_VARIANCE_FLOOR_K = 65536.0

PARAMETERS = {
    'proxy': 'daily_log_range_variance_hfq_high_low',
    'design_windows': [1, 5, 22],
    'fit_window': 504,
    'min_pairs': 400,
    'max_condition': 100000000.0,
    'signal': 'coefficient_on_log_22day_mean',
    'endpoint_rule': 'hfq_high_gt_low_gt_0_and_raw_high_gt_raw_low_gt_0_on_traded_days',
}


def range_variance(hfq_high, hfq_low, raw_high, raw_low, traded):
    """q_d frozen proxy; NaN unless traded with valid adjusted and raw ranges."""
    good = (traded & (hfq_high > hfq_low) & (hfq_low > 0)
            & np.isfinite(hfq_high) & np.isfinite(hfq_low)
            & np.isfinite(raw_high) & np.isfinite(raw_low)
            & (raw_high > raw_low) & (raw_low > 0))
    q = np.full(hfq_high.shape, np.nan)
    q[good] = np.log(hfq_high[good] / hfq_low[good]) ** 2 / _LOG2_4
    return q


def _frozen_cell(q_col, t):
    """Frozen per-cell reference (verbatim probe fit() core).

    Raw-design OLS via lstsq, standardized-design rank + condition gate.
    Returns (value, status, condition).
    """
    win = int(PARAMETERS['fit_window'])
    min_pairs = int(PARAMETERS['min_pairs'])
    max_cond = float(PARAMETERS['max_condition'])
    lq = np.log(q_col)
    xs, ys = [], []
    for d in range(t - win, t):
        a = q_col[d - 21:d + 1]
        if np.isfinite(a).all() and (a > 0).all():
            xs.append([1.0, np.log(a[-1]), np.log(np.mean(a[-5:])), np.log(np.mean(a))])
        else:
            xs.append([np.nan] * 4)
        ys.append(lq[d + 1])
    X = np.asarray(xs)
    y = np.asarray(ys)
    g = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[g], y[g]
    n_ = len(y)
    if n_ < min_pairs:
        return np.nan, 'insufficient', np.nan
    sd = X[:, 1:].std(0)
    if (sd == 0).any():
        return np.nan, 'rank_deficient', np.nan
    Z = np.column_stack((np.ones(n_), (X[:, 1:] - X[:, 1:].mean(0)) / sd))
    cond = float(np.linalg.cond(Z))
    if np.linalg.matrix_rank(Z) < 4 or cond >= max_cond:
        return np.nan, 'collinear', cond
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(b[3]), 'valid', cond


def estimate(q, traded, universe, rows):
    """Batched frozen OLS; returns (values, diagnostics)."""
    win = int(PARAMETERS['fit_window'])
    min_pairs = int(PARAMETERS['min_pairs'])
    n, k = q.shape
    good = np.isfinite(q)
    lq = np.full((n, k), np.nan)
    lq[good] = np.log(q[good])
    y = np.full((n, k), np.nan)
    y[:-1] = lq[1:]
    g = good.astype(np.float64)
    qv = np.where(good, q, 0.0)

    def prefix(a):
        P = np.zeros((n + 1, k))
        np.cumsum(a, axis=0, out=P[1:])
        return P

    def win_sum(P, w):
        out = np.full((n, k), np.nan)
        out[w - 1:] = P[w:] - P[:n - w + 1]
        return out

    Pg = prefix(g)
    Pq = prefix(qv)
    cnt5 = win_sum(Pg, 5)
    cnt22 = win_sum(Pg, 22)
    with np.errstate(invalid='ignore', divide='ignore'):
        x2 = np.where(cnt5 == 5, np.log(win_sum(Pq, 5) / 5.0), np.nan)
        x3 = np.where(cnt22 == 22, np.log(win_sum(Pq, 22) / 22.0), np.nan)
    x1 = lq
    valid = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(x3) & np.isfinite(y)
    vf = valid.astype(np.float64)
    vx1 = np.where(valid, x1, 0.0)
    vx2 = np.where(valid, x2, 0.0)
    vx3 = np.where(valid, x3, 0.0)
    vy = np.where(valid, y, 0.0)
    P = {
        'w': prefix(vf), 'x1': prefix(vx1), 'x2': prefix(vx2), 'x3': prefix(vx3),
        'x1x1': prefix(vx1 * vx1), 'x2x2': prefix(vx2 * vx2), 'x3x3': prefix(vx3 * vx3),
        'x1x2': prefix(vx1 * vx2), 'x1x3': prefix(vx1 * vx3), 'x2x3': prefix(vx2 * vx3),
        'y': prefix(vy), 'x1y': prefix(vx1 * vy), 'x2y': prefix(vx2 * vy),
        'x3y': prefix(vx3 * vy),
    }
    out = np.full((len(rows), k), np.nan)
    conds = np.full((len(rows), k), np.nan)
    npairs = np.full((len(rows), k), np.nan)
    status = {'insufficient': 0, 'rank_deficient': 0, 'collinear': 0, 'valid': 0,
              'fallback_cells': 0}
    eps = np.finfo(np.float64).eps
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        for i, t in enumerate(rows):
            a = int(t) - win
            b = int(t) - 1
            S = {key: arr[b + 1] - arr[a] for key, arr in P.items()}
            nn = S['w']
            npairs[i] = nn
            elig = np.isfinite(nn) & (nn >= min_pairs)
            status['insufficient'] += int((~elig).sum())
            if not elig.any():
                continue
            idx = np.flatnonzero(elig)
            nnv = nn[idx]
            m1 = S['x1'][idx] / nnv
            m2 = S['x2'][idx] / nnv
            m3 = S['x3'][idx] / nnv
            c11 = S['x1x1'][idx] - nnv * m1 * m1
            c22 = S['x2x2'][idx] - nnv * m2 * m2
            c33 = S['x3x3'][idx] - nnv * m3 * m3
            # variance above the cancellation floor of the sum-of-squares
            floor_scale = _VARIANCE_FLOOR_K * nnv * eps
            healthy = ((c11 > floor_scale * (m1 * m1 + 1.0))
                       & (c22 > floor_scale * (m2 * m2 + 1.0))
                       & (c33 > floor_scale * (m3 * m3 + 1.0)))
            fb_idx = idx[~healthy]
            if healthy.any():
                h = healthy
                nn2 = nnv[h]
                m1h, m2h, m3h = m1[h], m2[h], m3[h]
                c11h, c22h, c33h = c11[h], c22[h], c33[h]
                sd1 = np.sqrt(c11h / nn2)
                sd2 = np.sqrt(c22h / nn2)
                sd3 = np.sqrt(c33h / nn2)
                r12 = (S['x1x2'][idx][h] - nn2 * m1h * m2h) / np.sqrt(c11h * c22h)
                r13 = (S['x1x3'][idx][h] - nn2 * m1h * m3h) / np.sqrt(c11h * c33h)
                r23 = (S['x2x3'][idx][h] - nn2 * m2h * m3h) / np.sqrt(c22h * c33h)
                R = np.empty((len(nn2), 3, 3))
                R[:, 0, 0] = R[:, 1, 1] = R[:, 2, 2] = 1.0
                R[:, 0, 1] = R[:, 1, 0] = r12
                R[:, 0, 2] = R[:, 2, 0] = r13
                R[:, 1, 2] = R[:, 2, 1] = r23
                lam = np.sort(np.linalg.eigvalsh(R), axis=1)
                cond = np.sqrt(lam[:, 2] / np.maximum(lam[:, 0], 0.0))
                tol = np.sqrt(np.maximum(lam[:, 2], 0.0)) * np.maximum(nn2, 4.0) * eps
                rank3 = lam[:, 0] > tol * tol
                fast = rank3 & (cond < _FALLBACK_COND)
                conds[i, idx[h]] = cond
                fb_idx = np.concatenate([fb_idx, idx[h][~fast]])
                if fast.any():
                    # solve in standardized coordinates (R well-conditioned,
                    # unlike the raw normal equations whose conditioning is
                    # squared). With z_i=(x_i-m_i)/sd_i, Z'Z = n R and
                    # Z'y_c = C_yi/sd_i, so b3 = (R^-1 rhs)_3 / (n sd3).
                    nn3 = nn2[fast]
                    my = S['y'][idx[h][fast]] / nn3
                    c_y1 = S['x1y'][idx[h][fast]] - nn3 * m1h[fast] * my
                    c_y2 = S['x2y'][idx[h][fast]] - nn3 * m2h[fast] * my
                    c_y3 = S['x3y'][idx[h][fast]] - nn3 * m3h[fast] * my
                    rhs = np.stack([c_y1 / sd1[fast], c_y2 / sd2[fast],
                                    c_y3 / sd3[fast]], axis=1)
                    coef = np.linalg.solve(R[fast], rhs[:, :, None])[:, :, 0]
                    b3 = coef[:, 2] / (nn3 * sd3[fast])
                    finite = np.isfinite(b3)
                    out[i, idx[h][fast][finite]] = b3[finite]
                    status['valid'] += int(finite.sum())
            if len(fb_idx):
                # near-degenerate / ill-conditioned cells: exact frozen
                # per-cell semantics instead of the batched algebra
                for j in fb_idx:
                    v, st, cd = _frozen_cell(q[:, j], int(t))
                    out[i, j] = v
                    conds[i, j] = cd
                    status[st] += 1
                    status['fallback_cells'] += 1
    out[~traded[rows]] = np.nan
    out[~universe[rows]] = np.nan
    return out, {'condition': conds, 'pairs': npairs, 'status_counts': status}


def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only frozen P014 parameters accepted")
    def compute(ctx):
        days = np.asarray(ctx.panel.dates, dtype=np.int64)
        n = len(days)
        out = np.full(ctx.panel.shape, np.nan)
        rows = np.arange(525, n)
        if len(rows):
            hi = np.asarray(ctx.hfq("high"), dtype=np.float64)
            lo = np.asarray(ctx.hfq("low"), dtype=np.float64)
            rhi = np.asarray(ctx.px_raw("high"), dtype=np.float64)
            rlo = np.asarray(ctx.px_raw("low"), dtype=np.float64)
            tr = np.asarray(ctx.traded(), dtype=bool)
            u = np.asarray(ctx.universe, dtype=bool)
            q = range_variance(hi, lo, rhi, rlo, tr)
            values, _ = estimate(q, tr, u, rows)
            out[rows] = values
        return out
    return compute
