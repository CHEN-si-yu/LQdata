"""R17 seat_cross batch 2: breadth / market-state exposures (C + GC).

Market-state series (all point-in-time, no future rows)
-------------------------------------------------------
From `stock_market_distribution_history` at the 15:00 close row of each
calendar-confirmed trading day (a day is used only if the vendor minute table
has at least 200 rows for it and up+down+flat > 0; a missing/partial day stays
NaN, never zero):
  up_share      = up_count / total
  strong_share  = (up_5_to_7 + up_7_to_10 + up_over_10) / total
  panic_share   = (down_5_to_7 + down_7_to_10 + down_over_10) / total
  limit_up_share, limit_down_share: vendor limit counts / total.  The vendor
      columns are identically zero before 2020-01-02 (no limit list coverage),
      so the limit series are set to NaN before that date and the two limit
      candidates start 2020-01-02.

From the frozen 2115-stock panel (equal-weight, traded mask, strict two-endpoint
returns):
  disp        = cross-sectional std of ret_clean(1) across >=100 valid stocks
  newhigh     = share of traded stocks at their 250-session hfq-close high
  newlow      = share at their 250-session hfq-close low
  above_ma    = share above their 200-session hfq-close mean
  upvol       = amount-weighted share of stocks with positive return

These are market-level series; the factors are stock-level exposures to them
(beta, asymmetric-state beta) or conditional return spreads.  Market-level
levels are deliberately not registered: they have zero cross-sectional
dispersion.

Directions (frozen before any return contact)
---------------------------------------------
C legs are listed in C_VARIANTS with the sign in each candidate spec.  The
economic view is that high-dispersion exposure is fragility (negative),
new-high/above-MA/up-volume breadth exposure is trend participation
(positive), new-low exposure is fear sensitivity (negative), and limit-up
breadth exposure is sentiment crowding (negative).  GC legs describe the share
of own variance explained by dispersion and the dispersion / |return| coupling.
"""
from __future__ import annotations

import numpy as np

WARMUP_DAYS = 800
FIRST_MONTH = None
LIMIT_START = 20200102
BW_TBL = "stock_market_distribution_history"
ATTR = ("up_count", "down_count", "flat_count",
        "up_5_to_7", "up_7_to_10", "up_over_10",
        "down_5_to_7", "down_7_to_10", "down_over_10",
        "limit_up_count", "limit_down_count")

C_VARIANTS = (
    "mk_disp_beta_60",
    "mk_disp_capture_252",
    "mk_disp_downside_capture_252",
    "mk_disp_beta_asym_252",
    "mk_newhigh_beta_60",
    "mk_newhigh_capture_60",
    "mk_newhigh_beta_asym_120",
    "mk_newlow_beta_60",
    "mk_newlow_capture_60",
    "mk_above_ma_beta_60",
    "mk_above_ma_capture_60",
    "mk_above_ma_beta_asym_120",
    "mk_upvol_beta_60",
    "mk_upvol_capture_60",
    "mk_limitup_beta_60",
    "mk_limitdown_capture_60",
    "mk_strong_capture_60",
    "mk_breadth_disp_divergence_capture_120",
)
GC_VARIANTS = (
    "mk_disp_r2_60",
    "mk_disp_absret_corr_60",
)
PARAMETERS = {name: {"variant": name} for name in C_VARIANTS + GC_VARIANTS}
ALLOWED_VARIANTS = tuple(PARAMETERS[name] for name in PARAMETERS)

_CACHE_KEY = "_r17_cross_breadth_v1"


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------
def _cs(a):
    return np.vstack([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])


def _roll_sum(a, n, min_count):
    """Rolling sum over the n rows ending at t; NaN+count policy as mathx."""
    T = a.shape[0]
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if T < n:
        return out
    fin = np.isfinite(a)
    x = np.where(fin, a, 0.0)
    cs = _cs(x)
    cn = _cs(fin.astype(np.float64))
    s = cs[n:] - cs[:-n]
    c = cn[n:] - cn[:-n]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[n - 1:] = np.where(c >= max(int(min_count), 1), s, np.nan)
    return out


def _roll_quantile(mat, n, q, min_count):
    """Exact linear-interpolation rolling quantile (sort + interpolate)."""
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


def _roll_beta(y, x1d, n, min_count):
    """Per-column rolling beta of y on a single market series x1d (T,)."""
    x = np.asarray(x1d, dtype=np.float64)[:, None]
    ok = np.isfinite(y) & np.isfinite(x)
    Y = np.where(ok, y, 0.0)
    X = np.where(ok, x, 0.0)
    cnt = _roll_sum(ok.astype(np.float64), n, min_count)
    sy = _roll_sum(Y, n, min_count)
    sx = _roll_sum(X, n, min_count)
    syy = _roll_sum(Y * Y, n, min_count)
    sxx = _roll_sum(X * X, n, min_count)
    sxy = _roll_sum(Y * X, n, min_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        my = sy / cnt
        mx = sx / cnt
        cov = sxy / cnt - mx * my
        var = sxx / cnt - mx * mx
        out = cov / var
    out[~(np.isfinite(out)) | (var <= 1e-8) | (np.abs(out) > 10.0)] = np.nan
    return out


def _roll_r2(y, x1d, n, min_count):
    """Per-column rolling R^2 of y on a single market series x1d."""
    x = np.asarray(x1d, dtype=np.float64)[:, None]
    ok = np.isfinite(y) & np.isfinite(x)
    Y = np.where(ok, y, 0.0)
    X = np.where(ok, x, 0.0)
    cnt = _roll_sum(ok.astype(np.float64), n, min_count)
    sy = _roll_sum(Y, n, min_count)
    sx = _roll_sum(X, n, min_count)
    syy = _roll_sum(Y * Y, n, min_count)
    sxx = _roll_sum(X * X, n, min_count)
    sxy = _roll_sum(Y * X, n, min_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        my = sy / cnt
        mx = sx / cnt
        cov = sxy / cnt - mx * my
        vy = syy / cnt - my * my
        vx = sxx / cnt - mx * mx
        out = cov * cov / (vy * vx)
    out[~(np.isfinite(out)) | (vy <= 1e-14) | (vx <= 1e-14)] = np.nan
    return out


def _roll_corr_mkt(y, x1d, n, min_count):
    """Per-column rolling correlation of y with a single market series x1d."""
    x = np.asarray(x1d, dtype=np.float64)[:, None]
    ok = np.isfinite(y) & np.isfinite(x)
    Y = np.where(ok, y, 0.0)
    X = np.where(ok, x, 0.0)
    cnt = _roll_sum(ok.astype(np.float64), n, min_count)
    sy = _roll_sum(Y, n, min_count)
    sx = _roll_sum(X, n, min_count)
    syy = _roll_sum(Y * Y, n, min_count)
    sxx = _roll_sum(X * X, n, min_count)
    sxy = _roll_sum(Y * X, n, min_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        my = sy / cnt
        mx = sx / cnt
        cov = sxy / cnt - mx * my
        vy = syy / cnt - my * my
        vx = sxx / cnt - mx * mx
        out = cov / np.sqrt(vy * vx)
    out[~(np.isfinite(out)) | (vy <= 1e-14) | (vx <= 1e-14)] = np.nan
    return out


def _roll_beta_state(y, x1d, state, n, min_state):
    """Beta of y on x1d computed on days where state > 0.5."""
    x = np.asarray(x1d, dtype=np.float64)[:, None]
    state = np.asarray(state, dtype=np.float64)
    if state.ndim == 1:
        state = state[:, None]
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(state)
    use = ok & (state > 0.5)
    Y = np.where(use, y, np.nan)
    X = np.where(use, x, np.nan)
    I = np.where(use, 1.0, np.nan)
    cnt = _roll_sum(I, n, min_state)
    sy = _roll_sum(Y, n, min_state)
    sx = _roll_sum(X, n, min_state)
    syy = _roll_sum(Y * Y, n, min_state)
    sxx = _roll_sum(X * X, n, min_state)
    sxy = _roll_sum(Y * X, n, min_state)
    with np.errstate(invalid="ignore", divide="ignore"):
        my = sy / cnt
        mx = sx / cnt
        cov = sxy / cnt - mx * my
        var = sxx / cnt - mx * mx
        out = cov / var
    out[~(np.isfinite(out)) | (var <= 1e-8) | (np.abs(out) > 10.0)] = np.nan
    return out


def _capture(y, state, n, min_state):
    """Mean y on state>0.5 days minus mean y on state<0.5 days."""
    state = np.asarray(state, dtype=np.float64)
    if state.ndim == 1:
        state = state[:, None]
    ok = np.isfinite(y) & np.isfinite(state)
    hi = ok & (state > 0.5)
    lo = ok & (state < 0.5)
    h = np.where(hi, 1.0, np.nan)
    l = np.where(lo, 1.0, np.nan)
    yh = np.where(hi, y, np.nan)
    yl = np.where(lo, y, np.nan)
    ch = _roll_sum(h, n, min_state)
    cl = _roll_sum(l, n, min_state)
    sh = _roll_sum(yh, n, min_state)
    sl = _roll_sum(yl, n, min_state)
    with np.errstate(invalid="ignore", divide="ignore"):
        mh = sh / ch
        ml = sl / cl
        out = mh - ml
    out[~(np.isfinite(out))] = np.nan
    return out


def _state_high(series, n=500, min_count=200):
    med = _roll_quantile(np.asarray(series, dtype=np.float64)[:, None],
                         n, 0.5, min_count)[:, 0]
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(series) & np.isfinite(med),
                        (series > med).astype(np.float64), np.nan)


def _state_low(series, n=500, min_count=200):
    med = _roll_quantile(np.asarray(series, dtype=np.float64)[:, None],
                         n, 0.5, min_count)[:, 0]
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(series) & np.isfinite(med),
                        (series < med).astype(np.float64), np.nan)


# ---------------------------------------------------------------------------
# market-state panel
# ---------------------------------------------------------------------------
class _Mkt:
    def __init__(self, ctx):
        cache = ctx.__dict__.setdefault(_CACHE_KEY, {})
        if cache.get("ready"):
            self.__dict__.update(cache["obj"])
            self.cache = cache
            return
        self.ctx = ctx
        days = np.asarray(ctx.panel.dates, dtype=np.int64)
        self.days = days
        self.cache = {}
        bk = _breadth_close(ctx)
        self.up_share = bk["up_share"]
        self.strong_share = bk["strong_share"]
        self.panic_share = bk["panic_share"]
        self.limit_up = bk["limit_up"]
        self.limit_down = bk["limit_down"]

        r = np.array(ctx.ret_clean(1), dtype=np.float64, copy=True)
        tr = np.asarray(ctx.traded(), dtype=bool)
        adj = tr.copy()
        adj[0] = False
        adj[1:] &= tr[:-1]
        r[~adj] = np.nan
        univ = np.asarray(ctx.universe, dtype=bool)
        self.r = r
        self.tr = tr
        self.univ = univ

        valid = tr & univ & np.isfinite(r)
        cnt = valid.sum(axis=1)
        safe_cnt = np.maximum(cnt, 1)
        m1 = np.where(valid, r, 0.0).sum(axis=1) / safe_cnt
        m2 = np.where(valid, r * r, 0.0).sum(axis=1) / safe_cnt
        disp = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
        self.disp = np.where(cnt >= 100, disp, np.nan)

        amount = np.asarray(ctx.px("amount"), dtype=np.float64)
        ape = valid & np.isfinite(amount) & (amount > 0)
        tot = np.where(ape, amount, 0.0).sum(axis=1)
        up_amt = np.where(ape & (r > 0), amount, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            vshare = up_amt / tot
        self.upvol = np.where((cnt >= 100) & (tot > 0), vshare, np.nan)

        hfq = np.asarray(ctx.hfq("close"), dtype=np.float64)
        hvalid = tr & univ & np.isfinite(hfq)
        hcnt = hvalid.sum(axis=1)
        rmax = ctx.roll_max(hfq, 250, 200)
        rmin = ctx.roll_min(hfq, 250, 200)
        ma = ctx.roll_mean(hfq, 200, 150)
        hi = hvalid & np.isfinite(rmax) & (hfq >= rmax - 1e-12)
        lo = hvalid & np.isfinite(rmin) & (hfq <= rmin + 1e-12)
        ab = hvalid & np.isfinite(ma) & (hfq > ma)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.newhigh = np.where(hcnt >= 100,
                                    hi.sum(axis=1) / np.maximum(hcnt, 1), np.nan)
            self.newlow = np.where(hcnt >= 100,
                                   lo.sum(axis=1) / np.maximum(hcnt, 1), np.nan)
            self.above_ma = np.where(hcnt >= 100,
                                     ab.sum(axis=1) / np.maximum(hcnt, 1), np.nan)
        self.mkt_ret = _index_ret(ctx, days)

        cache["obj"] = {k: v for k, v in self.__dict__.items() if k != "cache"}
        cache["ready"] = True

    def out(self, values):
        return np.where(self.tr & self.univ, values, np.nan)


def _breadth_close(ctx):
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    T = len(days)
    names = ("up_share", "strong_share", "panic_share", "limit_up", "limit_down")
    out = {k: np.full(T, np.nan) for k in names}
    y0, y1 = int(days[0]) // 10000, int(days[-1]) // 10000
    df = ctx.dataset(BW_TBL, columns=["trade_time", *ATTR], years=(y0, y1))
    if df is None or df.empty:
        return out
    tt = df["trade_time"].astype("string")
    day = ctx.date_col(tt.str.slice(0, 10)).astype(np.int64)
    hm = (tt.str.slice(11, 13).astype("int32") * 100
          + tt.str.slice(14, 16).astype("int32")).to_numpy(dtype=np.int64)
    num = lambda c: np.asarray(df[c], dtype=np.float64)
    up, dn, flat = num("up_count"), num("down_count"), num("flat_count")
    tot = up + dn + flat
    with np.errstate(invalid="ignore", divide="ignore"):
        up_share = up / tot
        strong = (num("up_5_to_7") + num("up_7_to_10") + num("up_over_10")) / tot
        panic = (num("down_5_to_7") + num("down_7_to_10") + num("down_over_10")) / tot
        lu = num("limit_up_count") / tot
        ld = num("limit_down_count") / tot
    order = np.lexsort((hm, day))
    day, hm = day[order], hm[order]
    vals = {k: v[order] for k, v in
            (("up_share", up_share), ("strong_share", strong),
             ("panic_share", panic), ("limit_up", lu), ("limit_down", ld),
             ("total", tot))}
    st = np.flatnonzero(np.concatenate(([True], day[1:] != day[:-1])))
    en = np.concatenate((st[1:], [day.size]))
    dday = day[st]
    rowcnt = en - st
    last = en - 1
    ok = (rowcnt >= 200) & (vals["total"][last] > 0)
    for k in names:
        v = np.where(ok, vals[k][last], np.nan)
        out[k] = ctx.panel.place(np.zeros(dday.size, dtype=np.int64),
                                 dday, v)[:, 0]
    for k in ("limit_up", "limit_down"):
        out[k] = np.where(days >= LIMIT_START, out[k], np.nan)
    return out


def _index_ret(ctx, days):
    out = np.full(len(days), np.nan)
    y0, y1 = int(days[0]) // 10000, int(days[-1]) // 10000
    df = ctx.dataset("index_daily", columns=["ts_code", "trade_date", "pct_chg"],
                     years=(y0, y1))
    if df is None or df.empty:
        return out
    df = df.loc[df["ts_code"] == "000300.SH"]
    if df.empty:
        return out
    d = ctx.date_col(df["trade_date"])
    v = np.asarray(df["pct_chg"], dtype=np.float64) / 100.0
    pos = np.searchsorted(days, d)
    ok = (pos < len(days)) & (days[np.minimum(pos, len(days) - 1)] == d)
    out[pos[ok]] = v[ok]
    return out


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------
def _compute(ctx, variant):
    m = _Mkt(ctx)
    r = m.r

    if variant == "mk_disp_beta_60":
        out = _roll_beta(r, m.disp, 60, 40)
    elif variant == "mk_disp_capture_252":
        out = _capture(r, _state_high(m.disp), 252, 20)
    elif variant == "mk_disp_downside_capture_252":
        sh = _state_high(m.disp)
        sl = _state_low(m.disp)
        down = np.isfinite(m.mkt_ret) & (m.mkt_ret < 0)
        state = np.where(down & np.isfinite(sh), sh, np.nan)
        low = np.where(down & np.isfinite(sl), sl, np.nan)
        # high-state vs low-state capture on market-down days
        out = _capture_two(r, state, low, 252, 10)
    elif variant == "mk_disp_beta_asym_252":
        s = _state_high(m.disp)
        out = _roll_beta_state(r, m.disp, s, 252, 30) - \
            _roll_beta_state(r, m.disp, 1.0 - s, 252, 30)
    elif variant == "mk_newhigh_beta_60":
        out = _roll_beta(r, m.newhigh, 60, 40)
    elif variant == "mk_newhigh_capture_60":
        out = _capture(r, _state_high(m.newhigh), 60, 12)
    elif variant == "mk_newhigh_beta_asym_120":
        s = _state_high(m.newhigh)
        out = _roll_beta_state(r, m.newhigh, s, 120, 20) - \
            _roll_beta_state(r, m.newhigh, 1.0 - s, 120, 20)
    elif variant == "mk_newlow_beta_60":
        out = _roll_beta(r, m.newlow, 60, 40)
    elif variant == "mk_newlow_capture_60":
        out = _capture(r, _state_high(m.newlow), 60, 12)
    elif variant == "mk_above_ma_beta_60":
        out = _roll_beta(r, m.above_ma, 60, 40)
    elif variant == "mk_above_ma_capture_60":
        out = _capture(r, _state_high(m.above_ma), 60, 12)
    elif variant == "mk_above_ma_beta_asym_120":
        s = _state_high(m.above_ma)
        out = _roll_beta_state(r, m.above_ma, s, 120, 20) - \
            _roll_beta_state(r, m.above_ma, 1.0 - s, 120, 20)
    elif variant == "mk_upvol_beta_60":
        out = _roll_beta(r, m.upvol, 60, 40)
    elif variant == "mk_upvol_capture_60":
        out = _capture(r, _state_high(m.upvol), 60, 12)
    elif variant == "mk_limitup_beta_60":
        out = _roll_beta(r, m.limit_up, 60, 40)
    elif variant == "mk_limitdown_capture_60":
        out = _capture(r, _state_high(m.limit_down, 250, 120), 60, 12)
    elif variant == "mk_strong_capture_60":
        out = _capture(r, _state_high(m.strong_share), 60, 12)
    elif variant == "mk_breadth_disp_divergence_capture_120":
        good = np.where((_state_high(m.up_share) > 0.5) &
                        (_state_low(m.disp) > 0.5), 1.0, np.nan)
        bad = np.where((_state_low(m.up_share) > 0.5) &
                       (_state_high(m.disp) > 0.5), 1.0, np.nan)
        out = _capture_two(r, good, bad, 120, 8)
    elif variant == "mk_disp_r2_60":
        out = _roll_r2(r, m.disp, 60, 40)
    elif variant == "mk_disp_absret_corr_60":
        out = _roll_corr_mkt(np.abs(r), m.disp, 60, 40)
    else:
        raise KeyError(variant)
    return m.out(out)


def _capture_two(y, hi_state, lo_state, n, min_state):
    """Mean y on hi_state>0.5 days minus mean y on lo_state>0.5 days."""
    hi_state = np.asarray(hi_state, dtype=np.float64)
    lo_state = np.asarray(lo_state, dtype=np.float64)
    if hi_state.ndim == 1:
        hi_state = hi_state[:, None]
    if lo_state.ndim == 1:
        lo_state = lo_state[:, None]
    ok = np.isfinite(y)
    hi = ok & np.isfinite(hi_state) & (hi_state > 0.5)
    lo = ok & np.isfinite(lo_state) & (lo_state > 0.5)
    h = np.where(hi, 1.0, np.nan)
    l = np.where(lo, 1.0, np.nan)
    yh = np.where(hi, y, np.nan)
    yl = np.where(lo, y, np.nan)
    ch = _roll_sum(h, n, min_state)
    cl = _roll_sum(l, n, min_state)
    sh = _roll_sum(yh, n, min_state)
    sl = _roll_sum(yl, n, min_state)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sh / ch - sl / cl
    out[~(np.isfinite(out))] = np.nan
    return out


def build(parameters):
    if parameters not in ALLOWED_VARIANTS:
        raise ValueError("Only frozen batch-2 breadth variants are allowed")
    variant = parameters["variant"]

    def compute(ctx):
        values = _compute(ctx, variant)
        if np.isinf(values).any():
            raise ValueError("Infinite breadth factor value")
        return values

    return compute
