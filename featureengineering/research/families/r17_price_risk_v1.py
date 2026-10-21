"""R17 seat_price batch 4: cross-sectional risk and co-movement with the market.

Pure builder for `r17p_d_risk_comove_*`.  Inputs: stock_daily + stock_adj_factor.
The market is the fixed-pool **leave-one-out equal-weight return** built from
`ctx.ret_clean(1)` and the current `ctx.universe`.  It is entirely point-in-time
(row t uses only day-t cross-section and past windows) and uses no index table,
no industry snapshot and no labels.

All regressions are rolling OLS on paired valid traded days with effective
counts.  A window with fewer than `min_count` paired days is NaN.  No full-
sample standardisation, no PCA, no fill.
"""
from __future__ import annotations

import numpy as np

WARMUP_DAYS = 1200

PARAMETERS = {
    "downside_beta_126": {"variant": "downside_beta_126"},
    "coskew_126": {"variant": "coskew_126"},
    "idio_skew_126": {"variant": "idio_skew_126"},
    "idio_vol_innov_21_252": {"variant": "idio_vol_innov_21_252"},
    "corr_market_126": {"variant": "corr_market_126"},
    "lower_tail_dependence_126": {"variant": "lower_tail_dependence_126"},
    "downside_coskew_126": {"variant": "downside_coskew_126"},
    "vol_beta_market_126": {"variant": "vol_beta_market_126"},
    "beta_dispersion_252": {"variant": "beta_dispersion_252"},
    "idio_vol_share_126": {"variant": "idio_vol_share_126"},
    "downside_idio_vol_ratio_126": {"variant": "downside_idio_vol_ratio_126"},
    "beta_innovation_21_252": {"variant": "beta_innovation_21_252"},
    "tail_beta_asymmetry_252": {"variant": "tail_beta_asymmetry_252"},
    "idio_vol_beta_market_vol_126": {"variant": "idio_vol_beta_market_vol_126"},
    "market_r2_126": {"variant": "market_r2_126"},
    "downside_beta_dispersion_252": {"variant": "downside_beta_dispersion_252"},
    "upper_tail_dependence_126": {"variant": "upper_tail_dependence_126"},
    "idio_return_autocorr_126": {"variant": "idio_return_autocorr_126"},
    "market_beta_downside_share_126": {"variant": "market_beta_downside_share_126"},
    "idio_vol_trend_63": {"variant": "idio_vol_trend_63"},
}


# --------------------------------------------------------------------- cache
def _cache(ctx, key, fn):
    store = ctx.__dict__.setdefault("_r17p_cache", {})
    if key not in store:
        store[key] = fn()
    return store[key]


def _r(ctx):
    return _cache(ctx, "ret_clean_1", lambda: ctx.ret_clean(1))


def _traded(ctx):
    return _cache(ctx, "traded", lambda: ctx.traded())


def _final(ctx, value):
    return np.where(_traded(ctx) & ctx.universe, value, np.nan)


def _market_loo(ctx):
    """Leave-one-out equal-weight market return per stock-day."""
    def _build():
        r = _r(ctx)
        valid = np.isfinite(r) & ctx.universe
        cnt = valid.sum(axis=1).astype(np.float64)
        tot = np.where(valid, r, 0.0).sum(axis=1)
        den = cnt[:, None] - valid.astype(np.float64)
        with np.errstate(all="ignore"):
            m = (tot[:, None] - np.where(valid, r, 0.0)) / np.maximum(den, 1.0)
        ok = valid & (cnt[:, None] >= 100.0)
        return np.where(ok, m, np.nan)
    return _cache(ctx, "market_loo", _build)


def _masked_moments(ctx, y, x, n, mc, extra=None):
    ok = np.isfinite(y) & np.isfinite(x)
    if extra is not None:
        ok = ok & extra
    ym = np.where(ok, y, np.nan)
    xm = np.where(ok, x, np.nan)
    my = ctx.roll_mean(ym, n, mc)
    mx = ctx.roll_mean(xm, n, mc)
    cov = ctx.roll_mean(ym * xm, n, mc) - mx * my
    vy = ctx.roll_mean(ym * ym, n, mc) - my * my
    vx = ctx.roll_mean(xm * xm, n, mc) - mx * mx
    return my, mx, cov, vy, vx


def _masked_beta(ctx, y, x, n, mc, extra=None):
    _, _, cov, _, vx = _masked_moments(ctx, y, x, n, mc, extra)
    with np.errstate(all="ignore"):
        return np.where(vx > 1e-10, cov / vx, np.nan)


def _masked_corr(ctx, y, x, n, mc, extra=None):
    _, _, cov, vy, vx = _masked_moments(ctx, y, x, n, mc, extra)
    with np.errstate(all="ignore"):
        corr = cov / np.sqrt(np.maximum(vy * vx, 0.0))
    return np.where(np.isfinite(corr) & (np.abs(corr) <= 1.0), corr, np.nan)


def _coskew(ctx, r, m, n, mc, extra=None):
    ok = np.isfinite(r) & np.isfinite(m)
    if extra is not None:
        ok = ok & extra
    rr = np.where(ok, r, np.nan)
    mm = np.where(ok, m, np.nan)
    mr = ctx.roll_mean(rr, n, mc)
    mmu = ctx.roll_mean(mm, n, mc)
    m2 = ctx.roll_mean(mm * mm, n, mc)
    m_rm = ctx.roll_mean(rr * mm, n, mc)
    m_rm2 = ctx.roll_mean(rr * mm * mm, n, mc)
    vr = ctx.roll_mean(rr * rr, n, mc) - mr * mr
    vm = m2 - mmu * mmu
    cov = m_rm2 - 2.0 * mmu * m_rm - mr * m2 + 2.0 * mr * mmu * mmu
    with np.errstate(all="ignore"):
        out = np.where((vr > 1e-12) & (vm > 1e-12),
                       cov / (np.sqrt(vr) * vm), np.nan)
    return out


def _skew(ctx, x, n, mc):
    mu = ctx.roll_mean(x, n, mc)
    m2 = ctx.roll_mean(x * x, n, mc) - mu * mu
    m3 = ctx.roll_mean(x * x * x, n, mc) - 3.0 * mu * m2 - mu ** 3
    with np.errstate(all="ignore"):
        out = np.where(m2 > 1e-12, m3 / np.power(m2, 1.5), np.nan)
    return np.where(np.isfinite(out) & (np.abs(out) <= 100.0), out, np.nan)


def _roll_quantile(ctx, x, n, q, min_count):
    """Chunked sort-based rolling linear-interpolation quantile (fast path)."""
    T, C = x.shape
    out = np.full((T, C), np.nan, dtype=np.float64)
    if T < n or n <= 0:
        return out
    Tn = T - n + 1
    block = 64
    for s in range(0, Tn, block):
        e = min(Tn, s + block)
        win = np.lib.stride_tricks.sliding_window_view(x[s:e + n - 1], n, axis=0)
        valid = np.isfinite(win)
        cnt = valid.sum(axis=2)
        arr = np.where(valid, win, np.inf).copy()
        arr.sort(axis=2)
        r = q * np.maximum(cnt - 1, 0).astype(np.float64)
        k0 = np.floor(r).astype(np.int64)
        k1 = np.minimum(k0 + 1, np.maximum(cnt - 1, 0))
        w = r - k0
        lo = np.take_along_axis(arr, k0[:, :, None], axis=2)[:, :, 0]
        hi = np.take_along_axis(arr, k1[:, :, None], axis=2)[:, :, 0]
        with np.errstate(all="ignore"):
            val = lo + w * (hi - lo)
        out[s + n - 1:e + n - 1] = np.where(cnt >= min_count, val, np.nan)
    return out


def _trend_slope(ctx, x, n, min_count):
    T, C = x.shape
    out = np.full((T, C), np.nan, dtype=np.float64)
    if T < n:
        return out
    k = np.arange(n, dtype=np.float64)
    block = 64
    Tn = T - n + 1
    for s in range(0, Tn, block):
        e = min(Tn, s + block)
        win = np.lib.stride_tricks.sliding_window_view(x[s:e + n - 1], n, axis=0)
        valid = np.isfinite(win)
        cnt = valid.sum(axis=2)
        wv = np.where(valid, 1.0, 0.0)
        mx = np.einsum("bcn,n->bc", wv, k) / np.maximum(cnt, 1)
        my = np.einsum("bcn,bcn->bc", np.where(valid, win, 0.0), wv) / np.maximum(cnt, 1)
        dx = k[None, None, :] - mx[:, :, None]
        dy = np.where(valid, win - my[:, :, None], 0.0)
        cov = np.einsum("bcn,bcn->bc", dx, dy) / np.maximum(cnt, 1)
        var = np.einsum("bcn,bcn->bc", dx, dx) / np.maximum(cnt, 1)
        with np.errstate(all="ignore"):
            slope = np.where(var > 1e-12, cov / var, np.nan)
        ok = (cnt >= min_count) & np.isfinite(slope)
        out[s + n - 1:e + n - 1] = np.where(ok, slope, np.nan)
    return out


def _beta(ctx, window):
    return _cache(ctx, f"beta_{window}", lambda: _masked_beta(ctx, _r(ctx), _market_loo(ctx), window, max(10, int(0.8 * window))))


def _resid(ctx, window):
    def _build():
        r = _r(ctx)
        m = _market_loo(ctx)
        return r - _beta(ctx, window) * m
    return _cache(ctx, f"resid_{window}", _build)


# --------------------------------------------------------------------- build
def build(parameters):
    variant = parameters["variant"]

    def fn(ctx):
        r = _r(ctx)
        m = _market_loo(ctx)

        if variant == "downside_beta_126":
            out = _masked_beta(ctx, r, m, 126, 20, extra=(m < 0.0))
            return _final(ctx, out)

        if variant == "coskew_126":
            return _final(ctx, _coskew(ctx, r, m, 126, 100))

        if variant == "idio_skew_126":
            return _final(ctx, _skew(ctx, _resid(ctx, 126), 126, 100))

        if variant == "idio_vol_innov_21_252":
            e21 = r - _beta(ctx, 21) * m
            e252 = _resid(ctx, 252)
            v21 = ctx.roll_std(e21, 21, 16)
            v252 = ctx.roll_std(e252, 252, 200)
            with np.errstate(all="ignore"):
                out = np.where(v252 > 1e-8, v21 / v252 - 1.0, np.nan)
            return _final(ctx, out)

        if variant == "corr_market_126":
            return _final(ctx, ctx.roll_corr(r, m, 126, 100))

        if variant == "lower_tail_dependence_126":
            q_i = _roll_quantile(ctx, ctx.shift(r, 1), 252, 0.10, 200)
            q_m = _roll_quantile(ctx, ctx.shift(m, 1), 252, 0.10, 200)
            pair = np.isfinite(r) & np.isfinite(m)
            low_i = (r <= q_i) & pair
            low_m = (m <= q_m) & pair
            joint = low_i & low_m
            num = ctx.roll_mean(np.where(pair, joint.astype(np.float64), np.nan), 126, 80)
            den = ctx.roll_mean(np.where(pair, low_m.astype(np.float64), np.nan), 126, 80)
            with np.errstate(all="ignore"):
                out = np.where(den > 0.01, num / den, np.nan)
            return _final(ctx, out)

        if variant == "downside_coskew_126":
            return _final(ctx, _coskew(ctx, r, m, 126, 20, extra=(m < 0.0)))

        if variant == "vol_beta_market_126":
            out = _masked_beta(ctx, np.abs(r), np.abs(m), 126, 100)
            return _final(ctx, out)

        if variant == "beta_dispersion_252":
            out = ctx.roll_std(_beta(ctx, 63), 252, 200)
            return _final(ctx, out)

        if variant == "idio_vol_share_126":
            e = _resid(ctx, 126)
            vr = ctx.roll_var(r, 126, 100)
            ve = ctx.roll_var(e, 126, 100)
            with np.errstate(all="ignore"):
                out = np.where(vr > 1e-12, ve / vr, np.nan)
            out = np.where(np.isfinite(out) & (out >= 0.0) & (out <= 2.0), out, np.nan)
            return _final(ctx, out)

        if variant == "downside_idio_vol_ratio_126":
            e = _resid(ctx, 126)
            down = np.where(e < 0.0, e * e, np.nan)
            up = np.where(e > 0.0, e * e, np.nan)
            md = ctx.roll_mean(down, 126, 20)
            mu = ctx.roll_mean(up, 126, 20)
            with np.errstate(all="ignore"):
                out = np.where(mu > 1e-14, np.sqrt(np.maximum(md, 0.0) / mu), np.nan)
            out = np.where(np.isfinite(out) & (out >= 0.0) & (out <= 100.0), out, np.nan)
            return _final(ctx, out)

        if variant == "beta_innovation_21_252":
            out = _beta(ctx, 21) - _beta(ctx, 252)
            return _final(ctx, out)

        if variant == "tail_beta_asymmetry_252":
            q10 = _roll_quantile(ctx, ctx.shift(m, 1), 252, 0.10, 200)
            q90 = _roll_quantile(ctx, ctx.shift(m, 1), 252, 0.90, 200)
            bd = _masked_beta(ctx, r, m, 252, 10, extra=(m <= q10))
            bu = _masked_beta(ctx, r, m, 252, 10, extra=(m >= q90))
            return _final(ctx, bd - bu)

        if variant == "idio_vol_beta_market_vol_126":
            e = _resid(ctx, 126)
            out = _masked_beta(ctx, np.abs(e), np.abs(m), 126, 100)
            return _final(ctx, out)

        if variant == "market_r2_126":
            c = ctx.roll_corr(r, m, 126, 100)
            return _final(ctx, c * c)

        if variant == "downside_beta_dispersion_252":
            bd = _masked_beta(ctx, r, m, 63, 10, extra=(m < 0.0))
            out = ctx.roll_std(bd, 252, 200)
            return _final(ctx, out)

        if variant == "upper_tail_dependence_126":
            q_i = _roll_quantile(ctx, ctx.shift(r, 1), 252, 0.90, 200)
            q_m = _roll_quantile(ctx, ctx.shift(m, 1), 252, 0.90, 200)
            pair = np.isfinite(r) & np.isfinite(m)
            high_i = (r >= q_i) & pair
            high_m = (m >= q_m) & pair
            joint = high_i & high_m
            num = ctx.roll_mean(np.where(pair, joint.astype(np.float64), np.nan), 126, 80)
            den = ctx.roll_mean(np.where(pair, high_m.astype(np.float64), np.nan), 126, 80)
            with np.errstate(all="ignore"):
                out = np.where(den > 0.01, num / den, np.nan)
            return _final(ctx, out)

        if variant == "idio_return_autocorr_126":
            e = _resid(ctx, 126)
            out = ctx.roll_corr(e, ctx.shift(e, 1), 126, 100)
            return _final(ctx, out)

        if variant == "market_beta_downside_share_126":
            cd = _masked_corr(ctx, r, m, 126, 20, extra=(m < 0.0))
            ca = ctx.roll_corr(r, m, 126, 100)
            with np.errstate(all="ignore"):
                out = cd * cd - ca * ca
            out = np.where(np.isfinite(out) & (out >= -1.0) & (out <= 1.0), out, np.nan)
            return _final(ctx, out)

        if variant == "idio_vol_trend_63":
            e = _resid(ctx, 126)
            a = np.abs(e)
            mu = ctx.roll_mean(a, 63, 50)
            slope = _trend_slope(ctx, a, 63, 50)
            with np.errstate(all="ignore"):
                out = np.where(mu > 1e-8, slope * 62.0 / mu, np.nan)
            out = np.where(np.isfinite(out) & (np.abs(out) <= 100.0), out, np.nan)
            return _final(ctx, out)

        raise KeyError(variant)

    return fn
