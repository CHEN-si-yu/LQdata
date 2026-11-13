"""R17 seat_cross batch 1: returns-based peer-network family (C + GC).

Economic object
---------------
A stock's *economic peer group* is inferred from realised co-movement only:
at each output month the graph is frozen on the 126 traded sessions ending at
u = the last session strictly before the month.  Edges are pairwise-complete
Pearson correlations of strict two-endpoint traded returns.  Each stock's
donor set is its top-20 peers by |corr| (self always excluded by the NaN
diagonal), with a minimum of 10 valid donors per day.  No current industry /
theme snapshot, no full-sample PCA, no full-sample standardisation and no
forward return is read anywhere in this module.

Two classes of objects
----------------------
C (declared direction and primary horizon frozen before any return contact):
  net_peer_mom_20              corr-weighted peer 20d mean return.
  net_peer_mom_breadth_20      corr-weighted share of donors with R20 > 0.
  net_peer_resid_reversal_5    own R5 minus freeze-beta times donor R5.
  net_peer_dispersion_20       corr-weighted cross-donor std of R20.
  net_peer_lg_flow_5           donor large-order net-flow share, 5d mean.
  net_peer_flow_surprise_20    donor total net-flow share (20d-60d)/60d sd.
  net_peer_small_flow_5        donor small-order net-flow share, 5d mean.
  net_peer_attention_turnover_5 donor turnover ratio to own 60d mean, 5d.
  net_peer_limit_up_share_5    share of donor-days with r >= 9.5%, 5d.
  net_peer_downside_beta_asym  freeze down-donor beta minus up-donor beta.
  net_peer_down_tail_coincidence_20  joint lower-tail coincidence, 20d.
  net_peer_return_skew_63      corr-weighted donor 63d return skewness.
  net_cluster_tightness        mean pairwise |corr| among the donors.
  net_two_hop_density          corr-weighted mean of donors' own density.
  net_eigen_centrality_resid   abs-graph eigenvector centrality, size residual.
  net_peer_size_tilt           own log size minus corr-weighted donor log size.
  net_peer_value_gap           own EP minus corr-weighted donor EP.
  net_corr_stability_change    profile-stability change vs previous freeze.
  net_coupling_innovation_20   trailing 20d own-donor corr minus freeze corr.
  net_peer_lead_return_1       corr-weighted donor return at t-1.
  net_peer_downside_vol_20     corr-weighted donor downside semideviation.

GC (ex-ante descriptive purpose, no predictive claim):
  net_edge_weight_hhi          concentration of the top-k edge weights.
  net_peer_turnover_dispersion  cross-donor dispersion of turnover ratios.
  net_peer_size_dispersion     cross-donor dispersion of log market cap.
  net_peer_corr_dispersion     cross-donor dispersion of signed correlations.

Timing
------
The graph at month m uses returns dated <= u only.  Every dynamic leg is a
trailing statistic ending at t-1 (explicit one-session shift), so the factor
at t uses information available at the previous close.  Static legs are
constants of the freeze date applied to the month.  The actual traded mask and
the frozen universe are applied last.  Missing sources stay NaN; nothing is
filled with zero.
"""
from __future__ import annotations

import gc

import numpy as np

WARMUP_DAYS = 800
FIRST_MONTH = 201801
CORR_WINDOW = 126
MIN_OBS = 100
TOPK = 20
MIN_PEERS = 10
MIN_STOCKS = 200
HALF_OBS = 50
MIN_COMMON = 50
EIG_ITERS = 40
STABILITY_FIRST = 201712  # month before FIRST_MONTH; needed by the change leg
BETA_ABS_MAX = 10.0
FLOW = "stock_main_fund_flow"
TIERS = ("sm", "md", "lg", "elg")

C_VARIANTS = (
    "net_peer_mom_20",
    "net_peer_mom_breadth_20",
    "net_peer_resid_reversal_5",
    "net_peer_dispersion_20",
    "net_peer_lg_flow_5",
    "net_peer_flow_surprise_20",
    "net_peer_small_flow_5",
    "net_peer_attention_turnover_5",
    "net_peer_limit_up_share_5",
    "net_peer_downside_beta_asym",
    "net_peer_down_tail_coincidence_20",
    "net_peer_return_skew_63",
    "net_cluster_tightness",
    "net_two_hop_density",
    "net_eigen_centrality_resid",
    "net_peer_size_tilt",
    "net_peer_value_gap",
    "net_corr_stability_change",
    "net_coupling_innovation_20",
    "net_peer_lead_return_1",
    "net_peer_downside_vol_20",
)
GC_VARIANTS = (
    "net_edge_weight_hhi",
    "net_peer_turnover_dispersion",
    "net_peer_size_dispersion",
    "net_peer_corr_dispersion",
)
PARAMETERS = {name: {"variant": name} for name in C_VARIANTS + GC_VARIANTS}
ALLOWED_VARIANTS = tuple(PARAMETERS[name] for name in PARAMETERS)

_CACHE_KEY = "_r17_cross_network_v2"
_MISS = object()


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------
def _pairwise(X: np.ndarray, Y: np.ndarray, min_n: int) -> np.ndarray:
    """Pairwise-complete Pearson correlation (r15net pattern, duplicated)."""
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
    with np.errstate(divide="ignore", invalid="ignore"):
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        cov = sxy - sx * sy / n
        corr = cov / np.sqrt(vx * vy)
    corr[(n < min_n) | ~(vx > 0) | ~(vy > 0)] = np.nan
    return corr


def _topk(mat: np.ndarray, k: int):
    N = mat.shape[0]
    kk = min(int(k), N - 1)
    filled = np.where(np.isfinite(mat), mat, -np.inf)
    idx = np.argpartition(-filled, kk - 1, axis=1)[:, :kk]
    vals = np.take_along_axis(mat, idx, axis=1)
    valid = np.isfinite(vals)
    idx = np.where(valid, idx, -1).astype(np.int32)
    return idx, vals, valid


def _month_runs(days: np.ndarray):
    tags = np.asarray(days, dtype=np.int64) // 100
    out = []
    start = 0
    for i in range(1, len(tags) + 1):
        if i == len(tags) or tags[i] != tags[start]:
            out.append((int(tags[start]), start, i))
            start = i
    return out


def _shift1(a: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float64)
    if a.shape[0] > 1:
        out[1:] = a[:-1]
    return out


def _ratio_pos(num, den, floor=1e-9):
    d = np.asarray(den, dtype=np.float64)
    n = np.asarray(num, dtype=np.float64)
    out = np.full(np.broadcast(n, d).shape, np.nan, dtype=np.float64)
    ok = np.isfinite(d) & (d > floor) & np.isfinite(n)
    np.divide(n, d, out=out, where=ok)
    return out


def _gather_rows(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """X[d0:d1] gathered over donor slots -> (D, N, k); invalid slots are NaN."""
    valid = idx >= 0
    safe = np.where(valid, idx, 0)
    G = X[:, safe]
    return np.where(valid[None, :, :], G, np.nan)


def _peer_stats(X, idx, w, mode="mean", min_count=MIN_PEERS):
    """Donor aggregate for a (D, N) slice.  X may contain NaN."""
    G = _gather_rows(X, idx)
    valid = np.isfinite(G) & np.isfinite(w)[None, :, :]
    ww = np.where(valid, w[None, :, :], 0.0)
    den = ww.sum(axis=2)
    cnt = valid.sum(axis=2)
    num = np.where(valid, G, 0.0) * ww
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = num.sum(axis=2) / den
    ok = (den > 0) & (cnt >= min_count)
    if mode == "mean":
        out = mean
    elif mode == "share_pos":
        pos = np.where(valid & (G > 0), w[None, :, :], 0.0).sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = pos / den
    elif mode == "std":
        dev = np.where(valid, G - mean[:, :, None], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (dev * dev * ww).sum(axis=2) / den
        out = np.sqrt(np.maximum(var, 0.0))
    else:
        raise ValueError(mode)
    out = np.where(ok, out, np.nan)
    return out


def _peer_static(vec, idx, w, mode="mean", min_count=MIN_PEERS):
    """Donor aggregate of a single freeze-date vector (N,)."""
    N = idx.shape[0]
    valid = idx >= 0
    safe = np.where(valid, idx, 0)
    g = vec[safe]
    vv = valid & np.isfinite(g) & np.isfinite(w)
    ww = np.where(vv, w, 0.0)
    den = ww.sum(axis=1)
    cnt = vv.sum(axis=1)
    num = np.where(vv, g, 0.0) * ww
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = num.sum(axis=1) / den
    ok = (den > 0) & (cnt >= min_count)
    if mode == "mean":
        out = mean
    elif mode == "std":
        dev = np.where(vv, g - mean[:, None], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (dev * dev * ww).sum(axis=1) / den
        out = np.sqrt(np.maximum(var, 0.0))
    else:
        raise ValueError(mode)
    return np.where(ok, out, np.nan)


def _peer_slice_stats(mat, w, mode="mean", min_count=MIN_PEERS):
    """Donor aggregate of an already-gathered (N, k) matrix (e.g. signed corr)."""
    vv = np.isfinite(mat) & np.isfinite(w)
    ww = np.where(vv, w, 0.0)
    den = ww.sum(axis=1)
    cnt = vv.sum(axis=1)
    num = np.where(vv, mat, 0.0) * ww
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = num.sum(axis=1) / den
    if mode == "mean":
        out = mean
    elif mode == "std":
        dev = np.where(vv, mat - mean[:, None], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (dev * dev * ww).sum(axis=1) / den
        out = np.sqrt(np.maximum(var, 0.0))
    else:
        raise ValueError(mode)
    return np.where((den > 0) & (cnt >= min_count), out, np.nan)


def _roll_quantile(mat, n, q, min_count):
    """Exact linear-interpolation rolling quantile (same result as nanquantile).

    Numerical implementation only: sort the finite window values, interpolate
    between floor/ceil positions with the finite count, and mask windows with
    fewer than min_count finite observations.  Chunked to bound memory.
    """
    mat = np.asarray(mat, dtype=np.float64)
    T, N = mat.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if T < n:
        return out
    x = np.where(np.isfinite(mat), mat, np.nan)
    step = 32
    for s in range(0, T - n + 1, step):
        e = min(T - n + 1, s + step)
        w = np.lib.stride_tricks.sliding_window_view(x[s:e + n - 1], n, axis=0)
        cnt = np.isfinite(w).sum(axis=-1)
        y = np.where(np.isfinite(w), w, np.inf)
        y.sort(axis=-1)
        pos = q * np.maximum(cnt - 1, 0)
        lo = np.clip(np.floor(pos).astype(np.int64), 0, n - 1)
        hi = np.clip(np.ceil(pos).astype(np.int64), 0, n - 1)
        vlo = np.take_along_axis(y, lo[..., None], axis=-1)[..., 0]
        vhi = np.take_along_axis(y, hi[..., None], axis=-1)[..., 0]
        with np.errstate(invalid="ignore"):
            val = vlo + (vhi - vlo) * (pos - lo)
        val[cnt < max(int(min_count), 1)] = np.nan
        out[s + n - 1:e + n - 1] = val
    return out


def _neutralize(raw, x, mask, min_stocks=MIN_STOCKS):
    m = mask & np.isfinite(raw) & np.isfinite(x)
    out = np.full(raw.shape, np.nan, dtype=np.float64)
    if int(m.sum()) < min_stocks:
        return out
    X = np.column_stack([np.ones(int(m.sum())), x[m]])
    y = raw[m]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out[m] = y - X @ beta
    return out


def _cluster_tight(C, idx, w, sel, min_pairs=20, chunk=128):
    N = idx.shape[0]
    out = np.full(N, np.nan, dtype=np.float64)
    for s in range(0, N, chunk):
        rows = idx[s:s + chunk]
        vr = rows >= 0
        safe = np.where(vr, rows, 0)
        c_, k_ = rows.shape
        sub = C[safe]                                   # (c,k,N)
        cols = np.broadcast_to(safe[:, None, :], (c_, k_, k_))   # (c,k,k)
        sub = np.take_along_axis(sub, cols, axis=2)     # (c,k,k)
        m = vr[:, :, None] & vr[:, None, :]
        a = np.where(m, np.abs(sub), np.nan)
        cnt = np.isfinite(a).sum(axis=(1, 2))
        val = np.nansum(a, axis=(1, 2)) / np.maximum(cnt, 1)
        take = sel[s:s + chunk] & (cnt >= min_pairs)
        out[s:s + chunk] = np.where(take, val, np.nan)
    return out


def _stability(r, lo, u, sel):
    half = CORR_WINDOW // 2
    C1 = _pairwise(r[lo:lo + half], r[lo:lo + half], HALF_OBS)
    C2 = _pairwise(r[lo + half:u + 1], r[lo + half:u + 1], HALF_OBS)
    m = sel[:, None] & sel[None, :] & np.isfinite(C1) & np.isfinite(C2)
    np.fill_diagonal(m, False)
    n = m.sum(axis=1)
    A = np.where(m, C1, 0.0)
    B = np.where(m, C2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = A.sum(axis=1) / np.maximum(n, 1)
        mb = B.sum(axis=1) / np.maximum(n, 1)
        Za = np.where(m, C1 - ma[:, None], 0.0)
        Zb = np.where(m, C2 - mb[:, None], 0.0)
        cov = (Za * Zb).sum(axis=1) / np.maximum(n, 1)
        va = (Za * Za).sum(axis=1) / np.maximum(n, 1)
        vb = (Zb * Zb).sum(axis=1) / np.maximum(n, 1)
        out = cov / np.sqrt(va * vb)
    out[(n < MIN_COMMON) | ~(va > 0) | ~(vb > 0)] = np.nan
    return np.where(sel, out, np.nan)


def _eigen_centrality(absC, sel):
    A = np.where(np.isfinite(absC), absC, 0.0)
    np.fill_diagonal(A, 0.0)
    v = np.where(sel, 1.0, 0.0)
    nrm = np.linalg.norm(v)
    if nrm <= 0:
        return np.full(sel.shape, np.nan)
    v /= nrm
    for _ in range(EIG_ITERS):
        v = A @ v
        nrm = np.linalg.norm(v)
        if nrm <= 1e-15:
            return np.full(sel.shape, np.nan)
        v /= nrm
    v = np.where(sel, v, np.nan)
    # sign is arbitrary for an eigenvector; fix it deterministically.
    if np.nansum(v) < 0:
        v = -v
    return v


# ---------------------------------------------------------------------------
# flow loader (stock_main_fund_flow -> three net shares)
# ---------------------------------------------------------------------------
def _load_flow(ctx, tr):
    T, N = tr.shape
    net_all = np.zeros((T, N), dtype=np.float64)
    net_lg = np.zeros((T, N), dtype=np.float64)
    net_sm = np.zeros((T, N), dtype=np.float64)
    turnf = np.zeros((T, N), dtype=np.float64)
    d0, d1 = int(ctx.panel.dates[0]), int(ctx.panel.dates[-1])
    for year in range(d0 // 10000, d1 // 10000 + 1):
        df = ctx.dataset(FLOW, columns=[
            "stock_code", "trade_date",
            *[f"buy_{t}_amount" for t in TIERS],
            *[f"sell_{t}_amount" for t in TIERS],
        ], years=(year, year))
        if df is None or df.empty:
            continue
        codes = df["stock_code"].to_numpy()
        days = ctx.date_col(df["trade_date"])
        buy = {t: df[f"buy_{t}_amount"].to_numpy(np.float64) for t in TIERS}
        sell = {t: df[f"sell_{t}_amount"].to_numpy(np.float64) for t in TIERS}
        turn = 0.5 * (sum(buy.values()) + sum(sell.values()))
        net = sum(buy.values()) - sum(sell.values())
        net_lg_y = ((buy["lg"] + buy["elg"]) - (sell["lg"] + sell["elg"]))
        net_sm_y = buy["sm"] - sell["sm"]
        turnf += ctx.event_grid(codes, days, turn).astype(np.float64)
        net_all += ctx.event_grid(codes, days, net).astype(np.float64)
        net_lg += ctx.event_grid(codes, days, net_lg_y).astype(np.float64)
        net_sm += ctx.event_grid(codes, days, net_sm_y).astype(np.float64)
        del df, buy, sell, turn, net, net_lg_y, net_sm_y
        for key in list(getattr(ctx.up, "_cache", {})):
            if key and key[0] == FLOW and (len(key) < 3 or key[2] == (year, year)):
                ctx.up._cache.pop(key, None)
        gc.collect()
    out = {}
    for name, raw in (("flow", net_all), ("lg", net_lg), ("sm", net_sm)):
        val = _ratio_pos(raw, turnf, 1e-9)
        out[name] = np.where(tr, val, np.nan)
    return out


# ---------------------------------------------------------------------------
# cached monthly graph + global donor panels
# ---------------------------------------------------------------------------
class _Net:
    def __init__(self, ctx):
        cache = ctx.__dict__.setdefault(_CACHE_KEY, {})
        if cache.get("ready"):
            self.__dict__.update(cache["obj"])
            self.cache = cache
            return
        t0 = ctx.panel.dates
        days = np.asarray(t0, dtype=np.int64)
        r = np.array(ctx.ret_clean(1), dtype=np.float64, copy=True)
        tr = np.asarray(ctx.traded(), dtype=bool)
        adj = tr.copy()
        adj[0] = False
        adj[1:] &= tr[:-1]
        r[~adj] = np.nan
        univ = np.asarray(ctx.universe, dtype=bool)
        if r.shape != univ.shape:
            raise ValueError("universe shape mismatch")

        self.ctx = ctx
        self.days = days
        self.r = r
        self.tr = tr
        self.univ = univ
        self.cache = {}

        # turnover ratio to own 60d baseline (trailing through t-1)
        turn = np.array(ctx.px("turnover"), dtype=np.float64, copy=True)
        turn[~tr] = np.nan
        base = ctx.shift(ctx.roll_mean(turn, 60, 40), 1)
        self.turn_ratio = _ratio_pos(turn, base, 1e-9)

        # fund-flow net shares
        flow = _load_flow(ctx, tr)
        self.flow = flow["flow"]
        self.flow_lg = flow["lg"]
        self.flow_sm = flow["sm"]

        # market cap / EP (freeze-time cross-sectional uses only)
        close = np.asarray(ctx.px("close"), dtype=np.float64)
        tshare = np.asarray(ctx.px("total_share"), dtype=np.float64)
        mv = close * tshare
        self.logmv = np.log(np.where(mv > 0, mv, np.nan))
        np_ = np.asarray(ctx.ttm("n_income_attr_p"), dtype=np.float64)
        self.ep = ctx.safe_div(np_, mv, min_abs_den=1e6)

        # per-stock 63d skew (used only at the freeze date)
        self.skew63 = ctx.roll_skew(r, 63, 50)

        graphs = []
        runs = _month_runs(days)
        for tag, d0, end in runs:
            u = d0 - 1
            if u < CORR_WINDOW - 1:
                continue
            lo = u - CORR_WINDOW + 1
            Rw = r[lo:u + 1]
            obs = np.isfinite(Rw).sum(axis=0)
            sel = univ[u] & (obs >= MIN_OBS)
            g = {"tag": tag, "d0": d0, "end": end, "u": u, "lo": lo,
                 "sel": sel, "skip": False}
            if int(sel.sum()) < MIN_STOCKS:
                g["skip"] = True
                graphs.append(g)
                continue
            C = _pairwise(Rw, Rw, MIN_OBS)
            C = np.where(sel[:, None] & sel[None, :], C, np.nan)
            np.fill_diagonal(C, np.nan)
            absC = np.abs(C)
            idx, vals, valid = _topk(absC, TOPK)
            w = np.where(valid, np.abs(vals), np.nan)
            signed = np.where(valid, np.take_along_axis(
                C, np.where(valid, idx, 0), axis=1), np.nan)
            g["idx"] = idx
            g["w"] = w
            g["signed"] = signed

            # descriptor vectors that need the full correlation matrix
            cnt = valid.sum(axis=1)
            dens = np.nansum(w, axis=1) / np.maximum(cnt, 1)
            g["density"] = np.where(sel & (cnt >= MIN_PEERS), dens, np.nan)
            if tag >= STABILITY_FIRST:
                g["stability"] = _stability(r, lo, u, sel)
            if tag >= FIRST_MONTH:
                g["tight"] = _cluster_tight(C, idx, w, sel)
                g["eig"] = _eigen_centrality(absC, sel)
                g["hhi"] = _peer_hhi(w, sel)
                g["corr_disp"] = _peer_slice_stats(signed, w, "std", MIN_PEERS)
                logmv_u = self.logmv[u]
                ep_u = self.ep[u]
                skew_u = self.skew63[u]
                g["size_tilt"] = logmv_u - _peer_static(
                    logmv_u, idx, w, "mean", MIN_PEERS)
                g["value_gap"] = ep_u - _peer_static(
                    ep_u, idx, w, "mean", MIN_PEERS)
                g["peer_skew"] = _peer_static(skew_u, idx, w, "mean", MIN_PEERS)
                g["size_disp"] = _peer_static(logmv_u, idx, w, "std", MIN_PEERS)
            graphs.append(g)

        # second chronological pass: daily donor panels and freeze-relative legs
        peer_avg = np.full(r.shape, np.nan)
        peer_flow = np.full(r.shape, np.nan)
        peer_lg = np.full(r.shape, np.nan)
        peer_sm = np.full(r.shape, np.nan)
        peer_turn = np.full(r.shape, np.nan)
        peer_turn_disp = np.full(r.shape, np.nan)
        peer_limit = np.full(r.shape, np.nan)
        limit_ind = np.where(np.isfinite(r), (r >= 0.095).astype(np.float64), np.nan)
        for g in graphs:
            if g["skip"]:
                continue
            d0, end, u, lo = g["d0"], g["end"], g["u"], g["lo"]
            idx, w, sel = g["idx"], g["w"], g["sel"]
            peer_avg[d0:end] = _peer_stats(r[d0:end], idx, w, "mean", MIN_PEERS)
            if g["tag"] >= FIRST_MONTH:
                peer_flow[d0:end] = _peer_stats(
                    self.flow[d0:end], idx, w, "mean", 8)
                peer_lg[d0:end] = _peer_stats(
                    self.flow_lg[d0:end], idx, w, "mean", 8)
                peer_sm[d0:end] = _peer_stats(
                    self.flow_sm[d0:end], idx, w, "mean", 8)
                peer_turn[d0:end] = _peer_stats(
                    self.turn_ratio[d0:end], idx, w, "mean", MIN_PEERS)
                peer_turn_disp[d0:end] = _peer_stats(
                    self.turn_ratio[d0:end], idx, w, "std", MIN_PEERS)
                peer_limit[d0:end] = _peer_stats(
                    limit_ind[d0:end], idx, w, "mean", MIN_PEERS)

            # freeze-relative beta/coupling on the current donor portfolio
            Xw = r[lo:u + 1]
            Yw = _peer_stats(Xw, idx, w, "mean", MIN_PEERS)
            g["beta"], g["beta_down"], g["beta_up"], g["coupling"] = \
                _beta_block(Xw, Yw, sel)

        self.graphs = graphs
        self.peer_avg = peer_avg
        self.peer_flow = peer_flow
        self.peer_lg = peer_lg
        self.peer_sm = peer_sm
        self.peer_turn = peer_turn
        self.peer_turn_disp = peer_turn_disp
        self.peer_limit = peer_limit
        self.limit_ind = limit_ind

        cache["obj"] = {k: v for k, v in self.__dict__.items() if k != "cache"}
        cache["ready"] = True

    # -- panels ----------------------------------------------------------
    def trail(self, key, n, min_count):
        ck = ("trail", key)
        if ck not in self.cache:
            self.cache[ck] = _shift1(self.ctx.roll_mean(self.r, n, min_count))
        return self.cache[ck]

    def agg(self, key, X, mode="mean", min_count=MIN_PEERS):
        ck = ("agg", key, mode, min_count)
        if ck not in self.cache:
            out = np.full(self.r.shape, np.nan)
            for g in self.graphs:
                if g["skip"]:
                    continue
                d0, end = g["d0"], g["end"]
                out[d0:end] = _peer_stats(
                    X[d0:end], g["idx"], g["w"], mode, min_count)
            self.cache[ck] = out
        return self.cache[ck]

    def static_panel(self, key):
        ck = ("static", key)
        if ck not in self.cache:
            out = np.full(self.r.shape, np.nan)
            for g in self.graphs:
                if g["skip"] or key not in g:
                    continue
                out[g["d0"]:g["end"]] = np.where(
                    g["sel"][None, :], g[key][None, :], np.nan)
            self.cache[ck] = out
        return self.cache[ck]

    def stability_change(self):
        ck = ("stability_change",)
        if ck not in self.cache:
            out = np.full(self.r.shape, np.nan)
            prev_tag = None
            prev = None
            prev_map = {}
            for g in self.graphs:
                if g["skip"] or "stability" not in g:
                    continue
                prev_map[g["tag"]] = g["stability"]
            for g in self.graphs:
                if g["skip"] or "stability" not in g:
                    continue
                p = _minus_months(g["tag"], 1)
                if p in prev_map:
                    val = g["stability"] - prev_map[p]
                    out[g["d0"]:g["end"]] = np.where(
                        g["sel"][None, :], val[None, :], np.nan)
            self.cache[ck] = out
        return self.cache[ck]


def _peer_hhi(w, sel):
    """HHI of the valid top-k |corr| weights (descriptive concentration)."""
    valid = np.isfinite(w)
    ww = np.where(valid, w, 0.0)
    den = ww.sum(axis=1)
    p = np.where(valid, ww / np.maximum(den[:, None], 1e-12), 0.0)
    hhi = (p * p).sum(axis=1)
    cnt = valid.sum(axis=1)
    return np.where(sel & (cnt >= MIN_PEERS) & (den > 0), hhi, np.nan)


def _minus_months(tag: int, n: int) -> int:
    year, month = divmod(int(tag), 100)
    total = year * 12 + (month - 1) - n
    return (total // 12) * 100 + (total % 12) + 1


def _beta_block(Xw, Yw, sel):
    """Per-stock regression of own daily return on the freeze donor portfolio."""
    N = Xw.shape[1]
    mx = np.isfinite(Xw) & np.isfinite(Yw)
    X = np.where(mx, Xw, 0.0)
    Y = np.where(mx, Yw, 0.0)
    M = mx.astype(np.float64)
    n = M.sum(axis=0)
    sx = X.sum(axis=0)
    sy = Y.sum(axis=0)
    sxx = (X * X).sum(axis=0)
    syy = (Y * Y).sum(axis=0)
    sxy = (X * Y).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        cov = sxy - sx * sy / n
        beta = cov / vx
        coupling = cov / np.sqrt(vx * vy)
    ok = sel & (n >= MIN_OBS) & (vx > 1e-10) & (vy > 1e-10)
    beta = np.where(ok & (np.abs(beta) <= BETA_ABS_MAX), beta, np.nan)
    coupling = np.where(ok, coupling, np.nan)

    bdown = np.full(N, np.nan)
    bup = np.full(N, np.nan)
    for mask, target, need in ((mx & (Yw < 0), bdown, 20),
                               (mx & (Yw > 0), bup, 20)):
        mm = mask.astype(np.float64)
        nn = mm.sum(axis=0)
        sxm = (X * mm).sum(axis=0)
        sym = (Y * mm).sum(axis=0)
        sxxm = (X * X * mm).sum(axis=0)
        sxym = (X * Y * mm).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            vxm = sxxm - sxm * sxm / nn
            covm = sxym - sxm * sym / nn
            bb = covm / vxm
        good = sel & (nn >= need) & (vxm > 1e-10) & \
            (np.abs(bb) <= BETA_ABS_MAX)
        target[good] = bb[good]
    asym = bdown - bup
    return beta, bdown, bup, np.where(sel, coupling, np.nan)


# ---------------------------------------------------------------------------
# variant computation
# ---------------------------------------------------------------------------
def _mask(b, out):
    return np.where(b.tr & b.univ, out, np.nan)


def _compute(ctx, variant):
    b = _Net(ctx)
    if variant == "net_peer_mom_20":
        R20 = _shift1(ctx.roll_mean(b.r, 20, 16))
        return _mask(b, b.agg("R20", R20, "mean", MIN_PEERS))

    if variant == "net_peer_mom_breadth_20":
        R20 = _shift1(ctx.roll_mean(b.r, 20, 16))
        return _mask(b, b.agg("R20_breadth", R20, "share_pos", MIN_PEERS))

    if variant == "net_peer_resid_reversal_5":
        R5 = _shift1(ctx.roll_mean(b.r, 5, 4))
        peerR5 = b.agg("R5", R5, "mean", MIN_PEERS)
        beta = b.static_panel("beta")
        return _mask(b, R5 - beta * peerR5)

    if variant == "net_peer_dispersion_20":
        R20 = _shift1(ctx.roll_mean(b.r, 20, 16))
        return _mask(b, b.agg("R20_disp", R20, "std", MIN_PEERS))

    if variant == "net_peer_lg_flow_5":
        x = _shift1(ctx.roll_mean(b.peer_lg, 5, 3))
        return _mask(b, x)

    if variant == "net_peer_flow_surprise_20":
        f = b.peer_flow
        m20 = ctx.shift(ctx.roll_mean(f, 20, 15), 1)
        m60 = ctx.shift(ctx.roll_mean(f, 60, 40), 1)
        sd60 = ctx.shift(ctx.roll_std(f, 60, 40), 1)
        out = _ratio_pos(m20 - m60, sd60, 1e-8)
        return _mask(b, out)

    if variant == "net_peer_small_flow_5":
        x = _shift1(ctx.roll_mean(b.peer_sm, 5, 3))
        return _mask(b, x)

    if variant == "net_peer_attention_turnover_5":
        x = _shift1(ctx.roll_mean(b.peer_turn, 5, 3))
        return _mask(b, x)

    if variant == "net_peer_limit_up_share_5":
        x = _shift1(ctx.roll_mean(b.peer_limit, 5, 3))
        return _mask(b, x)

    if variant == "net_peer_downside_beta_asym":
        bd = b.static_panel("beta_down")
        bu = b.static_panel("beta_up")
        return _mask(b, bd - bu)

    if variant == "net_peer_down_tail_coincidence_20":
        q_own = _roll_quantile(b.r, 126, 0.2, 100)
        q_peer = _roll_quantile(b.peer_avg, 126, 0.2, 100)
        with np.errstate(invalid="ignore"):
            ind_own = np.where(np.isfinite(b.r) & np.isfinite(q_own),
                               (b.r <= q_own).astype(np.float64), np.nan)
            ind_peer = np.where(np.isfinite(b.peer_avg) & np.isfinite(q_peer),
                                (b.peer_avg <= q_peer).astype(np.float64), np.nan)
        both = ind_own * ind_peer
        p_both = _shift1(ctx.roll_mean(both, 20, 15))
        p_own = _shift1(ctx.roll_mean(ind_own, 20, 15))
        p_peer = _shift1(ctx.roll_mean(ind_peer, 20, 15))
        return _mask(b, p_both - p_own * p_peer)

    if variant == "net_peer_return_skew_63":
        return _mask(b, b.static_panel("peer_skew"))

    if variant == "net_cluster_tightness":
        return _mask(b, b.static_panel("tight"))

    if variant == "net_two_hop_density":
        # donor-of-donor density, aggregated over the same top-k donors
        out = np.full(b.r.shape, np.nan)
        for g in b.graphs:
            if g["skip"] or "density" not in g:
                continue
            val = _peer_static(g["density"], g["idx"], g["w"], "mean", MIN_PEERS)
            out[g["d0"]:g["end"]] = np.where(g["sel"][None, :], val[None, :], np.nan)
        return _mask(b, out)

    if variant == "net_eigen_centrality_resid":
        out = np.full(b.r.shape, np.nan)
        for g in b.graphs:
            if g["skip"] or "eig" not in g:
                continue
            u = g["u"]
            raw = _neutralize(g["eig"], b.logmv[u], g["sel"], MIN_STOCKS)
            out[g["d0"]:g["end"]] = np.where(g["sel"][None, :], raw[None, :], np.nan)
        return _mask(b, out)

    if variant == "net_peer_size_tilt":
        return _mask(b, b.static_panel("size_tilt"))

    if variant == "net_peer_value_gap":
        return _mask(b, b.static_panel("value_gap"))

    if variant == "net_corr_stability_change":
        return _mask(b, b.stability_change())

    if variant == "net_coupling_innovation_20":
        cr = _shift1(ctx.roll_corr(b.r, b.peer_avg, 20, 15))
        return _mask(b, cr - b.static_panel("coupling"))

    if variant == "net_peer_lead_return_1":
        return _mask(b, _shift1(b.peer_avg))

    if variant == "net_peer_downside_vol_20":
        neg = np.where(np.isfinite(b.r), (b.r < 0).astype(np.float64), np.nan)
        sq = np.where(np.isfinite(b.r), b.r * b.r, np.nan) * neg
        dv = np.sqrt(np.maximum(_shift1(ctx.roll_mean(sq, 20, 15)), 0.0))
        return _mask(b, b.agg("downvol", dv, "mean", MIN_PEERS))

    if variant == "net_edge_weight_hhi":
        return _mask(b, b.static_panel("hhi"))

    if variant == "net_peer_turnover_dispersion":
        x = _shift1(ctx.roll_mean(b.peer_turn_disp, 20, 15))
        return _mask(b, x)

    if variant == "net_peer_size_dispersion":
        return _mask(b, b.static_panel("size_disp"))

    if variant == "net_peer_corr_dispersion":
        return _mask(b, b.static_panel("corr_disp"))

    raise KeyError(variant)


def build(parameters):
    """Candidate entry point: returns compute(ctx) -> (T, C) float64 panel."""
    if parameters not in ALLOWED_VARIANTS:
        raise ValueError("Only frozen batch-1 cross-network variants are allowed")
    variant = parameters["variant"]

    def compute(ctx):
        values = _compute(ctx, variant)
        if np.isinf(values).any():
            raise ValueError("Infinite cross-network factor value")
        return values

    return compute
