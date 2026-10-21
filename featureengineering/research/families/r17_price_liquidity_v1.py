"""R17 seat_price batch 3: daily liquidity, price impact and volume structure.

Pure builder for `r17p_c_liquidity_*`.  Inputs: stock_daily (vol/amount/close/
high/low), stock_adj_factor (hfq returns), and stock_finance.float_share only
for turnover-based candidates (declared in their deps).

Units and semantics:
* amount is CNY, vol is shares, turnover = vol / float_share * 100 (percent).
* Amihud = |ret_clean(1)| / amount (1/CNY), no 1e8 rescaling.
* All flow fields are NaN on suspension, never zero.  Rolling means use the
  effective valid count.  No fill, no winsor.
* VWAP = amount / vol is the same-day share-weighted average price; the
  deviation uses the *unadjusted* close (both stock_daily fields) so no
  adjustment mismatch enters.
"""
from __future__ import annotations

import numpy as np

WARMUP_DAYS = 1200

PARAMETERS = {
    "amihud_asymmetry_63": {"variant": "amihud_asymmetry_63"},
    "amihud_innovation_21_252": {"variant": "amihud_innovation_21_252"},
    "amihud_tail_concentration_63": {"variant": "amihud_tail_concentration_63"},
    "kyle_lambda_63": {"variant": "kyle_lambda_63"},
    "volume_price_elasticity_63": {"variant": "volume_price_elasticity_63"},
    "turnover_hhi_63": {"variant": "turnover_hhi_63"},
    "nontraded_share_126": {"variant": "nontraded_share_126"},
    "clv_63": {"variant": "clv_63"},
    "chaikin_money_flow_63": {"variant": "chaikin_money_flow_63"},
    "force_index_13": {"variant": "force_index_13"},
    "signed_volume_ratio_63": {"variant": "signed_volume_ratio_63"},
    "vwap_deviation_5": {"variant": "vwap_deviation_5"},
    "high_volume_down_day_share_63": {"variant": "high_volume_down_day_share_63"},
    "downside_turnover_asymmetry_63": {"variant": "downside_turnover_asymmetry_63"},
    "volume_impact_quantile_ratio_63": {"variant": "volume_impact_quantile_ratio_63"},
    "turnover_trend_63": {"variant": "turnover_trend_63"},
    "volume_return_corr_63": {"variant": "volume_return_corr_63"},
    "corwin_schultz_spread_63": {"variant": "corwin_schultz_spread_63"},
    "turnover_volatility_asymmetry_63": {"variant": "turnover_volatility_asymmetry_63"},
    "range_volume_corr_63": {"variant": "range_volume_corr_63"},
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


def _amihud(ctx):
    def _build():
        r = _r(ctx)
        amt = ctx.px("amount")
        with np.errstate(all="ignore"):
            out = np.where(np.isfinite(r) & np.isfinite(amt) & (amt > 0.0),
                           np.abs(r) / amt, np.nan)
        return out
    return _cache(ctx, "amihud", _build)


def _clv(ctx):
    def _build():
        tr = _traded(ctx)
        hi = ctx.hfq("high")
        lo = ctx.hfq("low")
        cl = ctx.hfq("close")
        width = hi - lo
        valid = tr & (width > 1e-12) & np.isfinite(hi) & np.isfinite(lo) & np.isfinite(cl)
        with np.errstate(all="ignore"):
            out = (2.0 * cl - hi - lo) / width
        return np.where(valid, out, np.nan)
    return _cache(ctx, "clv", _build)


def _roll_quantile(ctx, x, n, q, min_count):
    """Chunked sort-based rolling linear-interpolation quantile.

    Same values as `np.nanquantile` on the valid prefix but ~50x faster than
    the generic masked path on 63d windows.  NaN is mapped to +inf so the
    valid observations occupy the sorted prefix; the interpolation indices are
    bounded by the per-cell valid count.
    """
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
    """OLS slope of x on time (0..n-1), effective valid indices per cell."""
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


# --------------------------------------------------------------------- build
def build(parameters):
    variant = parameters["variant"]

    def fn(ctx):
        r = _r(ctx)

        if variant == "amihud_asymmetry_63":
            a = _amihud(ctx)
            down = np.where(r < 0.0, a, np.nan)
            up = np.where(r > 0.0, a, np.nan)
            md = ctx.roll_mean(down, 63, 10)
            mu = ctx.roll_mean(up, 63, 10)
            with np.errstate(all="ignore"):
                out = np.where(mu > 0.0, md / mu - 1.0, np.nan)
            return _final(ctx, out)

        if variant == "amihud_innovation_21_252":
            a = _amihud(ctx)
            a21 = ctx.roll_mean(a, 21, 15)
            a252 = ctx.roll_mean(a, 252, 200)
            with np.errstate(all="ignore"):
                out = np.where(a252 > 0.0, a21 / a252 - 1.0, np.nan)
            return _final(ctx, out)

        if variant == "amihud_tail_concentration_63":
            a = _amihud(ctx)
            q90 = _roll_quantile(ctx, a, 63, 0.90, 40)
            tail = np.where(a >= q90, a, np.nan)
            mt = ctx.roll_mean(tail, 63, 5)
            mall = ctx.roll_mean(a, 63, 50)
            with np.errstate(all="ignore"):
                out = np.where(mall > 0.0, mt / mall - 1.0, np.nan)
            return _final(ctx, out)

        if variant == "kyle_lambda_63":
            amt = ctx.px("amount")
            x = np.where(np.isfinite(amt) & (amt > 0.0), np.sqrt(amt / 1e8), np.nan)
            y = np.abs(r)
            cov = ctx.roll_cov(y, x, 63, 50)
            var = ctx.roll_var(x, 63, 50)
            with np.errstate(all="ignore"):
                out = np.where(var > 1e-12, cov / var, np.nan)
            return _final(ctx, out)

        if variant == "volume_price_elasticity_63":
            to = ctx.px("turnover")
            x = np.where((np.abs(r) > 1e-6) & (to > 0.0),
                         np.log(np.abs(r)), np.nan)
            y = np.where((np.abs(r) > 1e-6) & (to > 0.0), np.log(to), np.nan)
            cov = ctx.roll_cov(y, x, 63, 40)
            var = ctx.roll_var(x, 63, 40)
            with np.errstate(all="ignore"):
                out = np.where(var > 1e-8, cov / var, np.nan)
            return _final(ctx, out)

        if variant == "turnover_hhi_63":
            to = ctx.px("turnover")
            e1 = ctx.roll_mean(to, 63, 50)
            e2 = ctx.roll_mean(to * to, 63, 50)
            cnt = ctx.roll_count(to, 63)
            with np.errstate(all="ignore"):
                out = np.where(e1 > 0.0, e2 / (cnt * e1 * e1), np.nan)
            out = np.where(np.isfinite(out) & (out >= 0.0) & (out <= 1.0), out, np.nan)
            return _final(ctx, out)

        if variant == "nontraded_share_126":
            uni = ctx.universe.astype(np.float64)
            nt = (ctx.universe & (~_traded(ctx))).astype(np.float64)
            den = ctx.roll_sum(uni, 126)
            num = ctx.roll_sum(nt, 126)
            with np.errstate(all="ignore"):
                out = np.where(den >= 100.0, num / den, np.nan)
            return _final(ctx, out)

        if variant == "clv_63":
            c = _clv(ctx)
            return _final(ctx, ctx.roll_mean(c, 63, 50))

        if variant == "chaikin_money_flow_63":
            c = _clv(ctx)
            amt = ctx.px("amount")
            valid = np.isfinite(c) & np.isfinite(amt) & (amt > 0.0)
            flow = np.where(valid, c * amt, np.nan)
            den = np.where(valid, amt, np.nan)
            num_s = ctx.roll_sum(flow, 63, 50)
            den_s = ctx.roll_sum(den, 63, 50)
            with np.errstate(all="ignore"):
                out = np.where(den_s > 0.0, num_s / den_s, np.nan)
            return _final(ctx, out)

        if variant == "force_index_13":
            px = ctx.hfq("close")
            vol = ctx.px("vol")
            prev = ctx.shift(px, 1)
            mean_vol = ctx.roll_mean(vol, 63, 50)
            with np.errstate(all="ignore"):
                chg = ctx.safe_div(px - prev, np.abs(prev), min_abs_den=1e-12)
                fi = chg * vol / mean_vol
            fi = np.where(np.isfinite(fi), fi, np.nan)
            out = ctx.ewm_mean(fi, span=13, min_count=10)
            return _final(ctx, out)

        if variant == "signed_volume_ratio_63":
            vol = ctx.px("vol")
            valid = np.isfinite(r) & np.isfinite(vol) & (vol > 0.0)
            v = np.where(valid, vol, np.nan)
            up = np.where(valid & (r > 0.0), v, 0.0)
            dn = np.where(valid & (r < 0.0), v, 0.0)
            up = np.where(valid, up, np.nan)
            dn = np.where(valid, dn, np.nan)
            num = ctx.roll_sum(up - dn, 63, 50)
            den = ctx.roll_sum(v, 63, 50)
            with np.errstate(all="ignore"):
                out = np.where(den > 0.0, num / den, np.nan)
            return _final(ctx, out)

        if variant == "vwap_deviation_5":
            amt = ctx.px("amount")
            vol = ctx.px("vol")
            close = ctx.px("close")
            with np.errstate(all="ignore"):
                vwap = np.where((vol > 0.0) & np.isfinite(amt), amt / vol, np.nan)
                dev = np.where(np.isfinite(vwap) & (vwap > 0.0) & np.isfinite(close),
                               close / vwap - 1.0, np.nan)
            out = ctx.roll_mean(dev, 5, 4)
            return _final(ctx, out)

        if variant == "high_volume_down_day_share_63":
            vol = ctx.px("vol")
            q90 = _roll_quantile(ctx, vol, 63, 0.90, 50)
            top = (vol >= q90) & np.isfinite(vol)
            down = top & (r < 0.0)
            top_f = np.where(np.isfinite(vol), top.astype(np.float64), np.nan)
            down_f = np.where(np.isfinite(vol), down.astype(np.float64), np.nan)
            num = ctx.roll_mean(down_f, 63, 10)
            den = ctx.roll_mean(top_f, 63, 10)
            with np.errstate(all="ignore"):
                out = np.where(den > 0.0, num / den, np.nan)
            return _final(ctx, out)

        if variant == "downside_turnover_asymmetry_63":
            to = ctx.px("turnover")
            down = np.where(r < 0.0, to, np.nan)
            up = np.where(r > 0.0, to, np.nan)
            md = ctx.roll_mean(down, 63, 10)
            mu = ctx.roll_mean(up, 63, 10)
            with np.errstate(all="ignore"):
                out = np.where(mu > 0.0, md / mu - 1.0, np.nan)
            return _final(ctx, out)

        if variant == "volume_impact_quantile_ratio_63":
            vol = ctx.px("vol")
            a = np.abs(r)
            q75 = _roll_quantile(ctx, vol, 63, 0.75, 50)
            q25 = _roll_quantile(ctx, vol, 63, 0.25, 50)
            hi = (vol >= q75) & np.isfinite(vol) & np.isfinite(a)
            lo = (vol <= q25) & np.isfinite(vol) & np.isfinite(a)
            hi_m = ctx.roll_mean(np.where(hi, a, np.nan), 63, 5)
            lo_m = ctx.roll_mean(np.where(lo, a, np.nan), 63, 5)
            with np.errstate(all="ignore"):
                out = np.where(lo_m > 1e-8, hi_m / lo_m - 1.0, np.nan)
            return _final(ctx, out)

        if variant == "turnover_trend_63":
            to = ctx.px("turnover")
            x = np.where(to > 0.0, np.log(to), np.nan)
            slope = _trend_slope(ctx, x, 63, 50)
            out = np.expm1(slope * 62.0)
            out = np.where(np.isfinite(out) & (np.abs(out) <= 1e6), out, np.nan)
            return _final(ctx, out)

        if variant == "volume_return_corr_63":
            vol = ctx.px("vol")
            x = np.where((vol > 0.0) & np.isfinite(vol), np.log(vol), np.nan)
            out = ctx.roll_corr(r, x, 63, 50)
            return _final(ctx, out)

        if variant == "corwin_schultz_spread_63":
            hi = ctx.hfq("high")
            lo = ctx.hfq("low")
            tr = _traded(ctx)
            prev_tr = ctx.shift(tr.astype(np.float64), 1) > 0.5
            hi1 = ctx.shift(hi, 1)
            lo1 = ctx.shift(lo, 1)
            valid = tr & prev_tr & (hi > lo) & (hi1 > lo1) & (lo > 0) & (lo1 > 0)
            with np.errstate(all="ignore"):
                beta = np.log(hi / lo) ** 2 + np.log(hi1 / lo1) ** 2
                hmax = np.maximum(hi, hi1)
                lmin = np.minimum(lo, lo1)
                gamma = np.log(hmax / lmin) ** 2
                k = 3.0 - 2.0 * np.sqrt(2.0)
                alpha = ((np.sqrt(2.0 * beta) - np.sqrt(beta)) / k
                         - np.sqrt(gamma / k))
                spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
            spread = np.where(valid & np.isfinite(spread), np.maximum(spread, 0.0), np.nan)
            out = ctx.roll_mean(spread, 63, 50)
            return _final(ctx, out)

        if variant == "turnover_volatility_asymmetry_63":
            to = ctx.px("turnover")
            down = np.where(r < 0.0, to, np.nan)
            up = np.where(r > 0.0, to, np.nan)
            sd = ctx.roll_std(down, 63, 10)
            su = ctx.roll_std(up, 63, 10)
            with np.errstate(all="ignore"):
                out = np.where(su > 1e-8, sd / su - 1.0, np.nan)
            return _final(ctx, out)

        if variant == "range_volume_corr_63":
            hi = ctx.hfq("high")
            lo = ctx.hfq("low")
            vol = ctx.px("vol")
            with np.errstate(all="ignore"):
                rng = np.where((hi > lo) & (lo > 0), np.log(hi / lo), np.nan)
            lv = np.where((vol > 0.0) & np.isfinite(vol), np.log(vol), np.nan)
            out = ctx.roll_corr(rng, lv, 63, 50)
            return _final(ctx, out)

        raise KeyError(variant)

    return fn
