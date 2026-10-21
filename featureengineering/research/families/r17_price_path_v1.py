"""R17 seat_price batch 2: path, trend, memory and time allocation.

Pure builder for `r17p_b_path_memory_*`.  Inputs: stock_daily + stock_adj_factor.

Return conventions:
* daily return for path/variance work is `ctx.ret_clean(1)` (suspension = NaN,
  not a synthetic zero).  For cumulative k-day momentum/efficiency the product
  of `1 + ctx.ret(1)` is used (the project's `_ret_k` convention): suspensions
  contribute a zero daily return, and a window with no trade at all is NaN.
* A window never reads T+1.  All statistics use rows <= T.
* No full-sample standardisation, no labels, no qfq.
"""
from __future__ import annotations

import numpy as np

WARMUP_DAYS = 1200

PARAMETERS = {
    "variance_ratio_5_126": {"variant": "variance_ratio_5_126"},
    "return_autocorr_1d_63": {"variant": "return_autocorr_1d_63"},
    "sign_entropy_63": {"variant": "sign_entropy_63"},
    "trend_efficiency_63": {"variant": "trend_efficiency_63"},
    "hurst_rs_126": {"variant": "hurst_rs_126"},
    "price_path_area_63": {"variant": "price_path_area_63"},
    "momentum_quality_126": {"variant": "momentum_quality_126"},
    "variance_time_concentration_63": {"variant": "variance_time_concentration_63"},
    "multi_horizon_trend_agreement_126": {"variant": "multi_horizon_trend_agreement_126"},
    "time_since_trough_126": {"variant": "time_since_trough_126"},
    "drawdown_recovery_days_126": {"variant": "drawdown_recovery_days_126"},
    "peak_timing_63": {"variant": "peak_timing_63"},
    "close_location_autocorr_63": {"variant": "close_location_autocorr_63"},
    "drawdown_velocity_63": {"variant": "drawdown_velocity_63"},
    "trend_age_126": {"variant": "trend_age_126"},
    "time_between_extremes_63": {"variant": "time_between_extremes_63"},
    "upside_path_share_63": {"variant": "upside_path_share_63"},
    "volume_weighted_momentum_63": {"variant": "volume_weighted_momentum_63"},
    "trend_strength_60": {"variant": "trend_strength_60"},
    "overnight_autocorr_63": {"variant": "overnight_autocorr_63"},
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


def _ret_k(ctx, k):
    """Product of clean daily returns with the project's zero-trade guard.

    Uses `ctx.ret(1)` (suspension contributes 1.0); a window in which the stock
    never traded is NaN rather than a manufactured zero return.
    """
    r1 = ctx.ret(1)
    with np.errstate(all="ignore"):
        out = np.expm1(ctx.roll_sum(np.log1p(r1), int(k)))
    ntrade = ctx.roll_count(_traded(ctx).astype(np.float64), int(k))
    return np.where(ntrade > 0.0, out, np.nan)


def _overnight(ctx):
    def _build():
        tr = _traded(ctx)
        prev_tr = ctx.shift(tr.astype(np.float64), 1) > 0.5
        op = ctx.hfq("open")
        prev_cl = ctx.shift(ctx.hfq("close"), 1)
        with np.errstate(all="ignore"):
            out = ctx.safe_div(op - prev_cl, np.abs(prev_cl), min_abs_den=1e-12)
        return np.where(tr & prev_tr & np.isfinite(out), out, np.nan)
    return _cache(ctx, "overnight_ret", _build)


def _clv(ctx):
    """Daily close location value on traded days with a non-degenerate range."""
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


# ------------------------------------------------------------- trend helpers
def _trend_r2(ctx, px, n, min_traded):
    """R^2 of log price on a linear time index inside each trailing window."""
    T, C = px.shape
    out = np.full((T, C), np.nan, dtype=np.float64)
    if T < n:
        return out
    x = ctx.safe_log(px)
    ramp = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    var_ramp = float(np.mean(ramp * ramp))
    block = 64
    Tn = T - n + 1
    for s in range(0, Tn, block):
        e = min(Tn, s + block)
        win = np.lib.stride_tricks.sliding_window_view(x[s:e + n - 1], n, axis=0)
        valid = np.isfinite(win).all(axis=2)
        mean = win.mean(axis=2)
        cov = np.einsum("bcn,n->bc", win, ramp) / n
        var = win.var(axis=2)
        with np.errstate(all="ignore"):
            corr = cov / np.sqrt(np.maximum(var * var_ramp, 0.0))
        r2 = np.where(valid & np.isfinite(corr), corr * corr, np.nan)
        out[s + n - 1:e + n - 1] = r2
    tc = ctx.roll_count(_traded(ctx).astype(np.float64), n)
    return np.where(tc >= min_traded, out, np.nan)


def _trend_slope(ctx, x, n, min_count):
    """OLS slope of x on time (0..n-1) with per-cell valid indices."""
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


def _rolling_recovery_days(ctx, px, n, min_traded=100, block=32):
    """Days from the window trough to the first full recovery of the prior peak.

    For each trailing window: trough = argmin price; peak = max price before
    (and including) the trough; recovery = first later row with price >= peak.
    Returns recovery duration in sessions; NaN if the peak is not recovered
    inside the window.  Full finite positive window plus traded-count guard.
    """
    T, C = px.shape
    out = np.full((T, C), np.nan, dtype=np.float64)
    if T < n:
        return out
    tc = ctx.roll_count(_traded(ctx).astype(np.float64), n)
    Tn = T - n + 1
    for s in range(0, Tn, block):
        e = min(Tn, s + block)
        win = np.lib.stride_tricks.sliding_window_view(px[s:e + n - 1], n, axis=0)
        b = e - s
        valid = np.isfinite(win).all(axis=2) & (win > 0).all(axis=2)
        trough_pos = np.argmin(win, axis=2)
        peak_before = np.zeros((b, C), dtype=np.float64)
        best = np.full((b, C), -np.inf, dtype=np.float64)
        for k in range(n):
            x = win[:, :, k]
            best = np.maximum(best, x)
            # only keep the running max up to each cell's trough
            peak_before = np.where(k <= trough_pos, best, peak_before)
        target = np.take_along_axis(win, trough_pos[:, :, None], axis=2)[:, :, 0]
        peak = peak_before
        rec_day = np.full((b, C), np.nan, dtype=np.float64)
        found = np.zeros((b, C), dtype=bool)
        for k in range(n):
            x = win[:, :, k]
            hit = (k > trough_pos) & (x >= peak) & (~found) & np.isfinite(peak)
            rec_day = np.where(hit, float(k), rec_day)
            found = found | hit
        dur = rec_day - trough_pos.astype(np.float64)
        ok = valid & found & (tc[s + n - 1:e + n - 1] >= min_traded)
        out[s + n - 1:e + n - 1] = np.where(ok, dur, np.nan)
    return out


def _rolling_maxdd(ctx, px, n, min_traded=100, block=64):
    """Maximum drawdown magnitude and duration, exact per trailing window."""
    T, C = px.shape
    out_dd = np.full((T, C), np.nan, dtype=np.float64)
    out_dur = np.full((T, C), np.nan, dtype=np.float64)
    if T < n:
        return out_dd, out_dur
    tc = ctx.roll_count(_traded(ctx).astype(np.float64), n)
    Tn = T - n + 1
    for s in range(0, Tn, block):
        e = min(Tn, s + block)
        win = np.lib.stride_tricks.sliding_window_view(px[s:e + n - 1], n, axis=0)
        b = e - s
        valid = np.isfinite(win).all(axis=2) & (win > 0).all(axis=2)
        best = np.full((b, C), -np.inf, dtype=np.float64)
        best_idx = np.zeros((b, C), dtype=np.int64)
        min_dd = np.zeros((b, C), dtype=np.float64)
        dur = np.zeros((b, C), dtype=np.float64)
        for k in range(n):
            x = win[:, :, k]
            upd = x > best
            best = np.where(upd, x, best)
            best_idx = np.where(upd, k, best_idx)
            dd = x / best - 1.0
            newmin = dd < min_dd
            min_dd = np.where(newmin, dd, min_dd)
            dur = np.where(newmin, (k - best_idx).astype(np.float64), dur)
        ok = valid & (tc[s + n - 1:e + n - 1] >= min_traded)
        out_dd[s + n - 1:e + n - 1] = np.where(ok, -min_dd, np.nan)
        out_dur[s + n - 1:e + n - 1] = np.where(ok, dur, np.nan)
    return out_dd, out_dur


def _run_length(ctx, sign):
    """Consecutive-day run length of the current sign, vectorised over stocks."""
    T, C = sign.shape
    run = np.zeros((T, C), dtype=np.float64)
    prev = np.full((T, C), np.nan, dtype=np.float64)
    prev[1:] = sign[:-1]
    prev[0] = np.nan
    for t in range(T):
        same = np.isfinite(sign[t]) & np.isfinite(prev[t]) & (sign[t] == prev[t])
        if t == 0:
            run[t] = np.where(np.isfinite(sign[t]), 1.0, np.nan)
        else:
            run[t] = np.where(np.isfinite(sign[t]),
                              np.where(same, run[t - 1] + 1.0, 1.0), np.nan)
    return run


# --------------------------------------------------------------------- build
def build(parameters):
    variant = parameters["variant"]

    def fn(ctx):
        r = _r(ctx)

        if variant == "variance_ratio_5_126":
            r1 = ctx.ret(1)
            with np.errstate(all="ignore"):
                r5 = np.expm1(ctx.roll_sum(np.log1p(r1), 5, min_count=4))
            v5 = ctx.roll_var(r5, 126, 100)
            v1 = ctx.roll_var(r, 126, 100)
            with np.errstate(all="ignore"):
                out = np.where(v1 > 1e-14, v5 / (5.0 * v1), np.nan)
            out = np.where(np.isfinite(out) & (out >= 0.0) & (out <= 100.0), out, np.nan)
            return _final(ctx, out)

        if variant == "return_autocorr_1d_63":
            out = ctx.roll_corr(r, ctx.shift(r, 1), 63, 50)
            return _final(ctx, out)

        if variant == "sign_entropy_63":
            up = (r > 0.0).astype(np.float64)
            dn = (r < 0.0).astype(np.float64)
            zr = (r == 0.0).astype(np.float64)
            valid = np.where(np.isfinite(r), 1.0, np.nan)
            up = np.where(np.isfinite(r), up, np.nan)
            dn = np.where(np.isfinite(r), dn, np.nan)
            zr = np.where(np.isfinite(r), zr, np.nan)
            p_up = ctx.roll_mean(up, 63, 50)
            p_dn = ctx.roll_mean(dn, 63, 50)
            p_zr = ctx.roll_mean(zr, 63, 50)
            n = ctx.roll_count(valid, 63)
            ent = np.zeros_like(p_up)
            for p in (p_up, p_dn, p_zr):
                with np.errstate(all="ignore"):
                    ent = ent - np.where(p > 0.0, p * np.log(p), 0.0)
            ent = np.where(np.isfinite(ent), ent, np.nan)
            # Miller-Madow first-order bias correction for K=3 categories
            with np.errstate(all="ignore"):
                ent = ent + 2.0 / (2.0 * np.maximum(n, 1.0))
            out = ent / np.log(3.0)
            out = np.where(np.isfinite(out) & (out >= 0.0) & (out <= 1.5), out, np.nan)
            return _final(ctx, out)

        if variant == "trend_efficiency_63":
            net = np.abs(_ret_k(ctx, 63))
            gross = ctx.roll_sum(np.abs(ctx.ret(1)), 63, min_count=50)
            with np.errstate(all="ignore"):
                out = np.where(gross > 1e-8, net / gross, np.nan)
            out = np.where(np.isfinite(out) & (out >= 0.0) & (out <= 1.0), out, np.nan)
            return _final(ctx, out)

        if variant == "hurst_rs_126":
            T, C = r.shape
            n = 126
            out = np.full((T, C), np.nan, dtype=np.float64)
            scales = [7, 14, 21, 42, 63]
            ls = np.log(np.asarray(scales, dtype=np.float64))
            lc = ls - ls.mean()
            denom = float(np.sum(lc * lc))
            for t in range(n - 1, T):
                w = r[t - n + 1:t + 1]
                idx = np.flatnonzero(np.isfinite(w).all(axis=0))
                if idx.size == 0:
                    continue
                ww = w[:, idx]
                logs = np.full((len(scales), idx.size), np.nan)
                for j, s in enumerate(scales):
                    m = n // s
                    sub = ww[:m * s].reshape(m, s, idx.size)
                    mu = sub.mean(axis=1)
                    dev = sub - mu[:, None, :]
                    cum = np.cumsum(dev, axis=1)
                    R = cum.max(axis=1) - cum.min(axis=1)
                    S = sub.std(axis=1, ddof=1)
                    with np.errstate(all="ignore"):
                        rs = np.where(S > 1e-12, R / S, np.nan)
                    rs_mean = rs.mean(axis=0)
                    with np.errstate(all="ignore"):
                        logs[j] = np.log(rs_mean)
                ok = np.isfinite(logs).all(axis=0)
                if not ok.any():
                    continue
                lr = logs - logs.mean(axis=0, keepdims=True)
                H = (lc[:, None] * lr).sum(axis=0) / denom
                out[t, idx[ok]] = H[ok]
            out = np.where(np.isfinite(out) & (out >= -1.0) & (out <= 2.0), out, np.nan)
            return _final(ctx, out)

        if variant == "price_path_area_63":
            x = ctx.safe_log(ctx.hfq("close"))
            mean_x = ctx.roll_mean(x, 63, 50)
            x0 = ctx.shift(x, 62)
            chord_mean = (x0 + x) / 2.0
            area = chord_mean - mean_x
            scale = ctx.roll_std(x, 63, 50)
            with np.errstate(all="ignore"):
                out = np.where(scale > 1e-8, area / scale, np.nan)
            for arr in (out,):
                arr[~np.isfinite(arr)] = np.nan
            return _final(ctx, out)

        if variant == "momentum_quality_126":
            mu = ctx.roll_mean(r, 126, 100)
            sd = ctx.roll_std(r, 126, 100)
            with np.errstate(all="ignore"):
                out = np.where(sd > 1e-8, mu / sd * np.sqrt(126.0), np.nan)
            return _final(ctx, out)

        if variant == "variance_time_concentration_63":
            e2 = ctx.roll_mean(r * r, 63, 50)
            e4 = ctx.roll_mean(r ** 4, 63, 50)
            cnt = ctx.roll_count(r, 63)
            with np.errstate(all="ignore"):
                out = np.where(e2 > 1e-14, e4 / (cnt * e2 * e2), np.nan)
            out = np.where(np.isfinite(out) & (out >= 0.0) & (out <= 1.0), out, np.nan)
            return _final(ctx, out)

        if variant == "multi_horizon_trend_agreement_126":
            s21 = np.sign(_ret_k(ctx, 21))
            s63 = np.sign(_ret_k(ctx, 63))
            s126 = np.sign(_ret_k(ctx, 126))
            out = (s21 + s63 + s126) / 3.0
            out = np.where(np.isfinite(out), out, np.nan)
            return _final(ctx, out)

        if variant == "time_since_trough_126":
            pos = ctx.roll_argmin(ctx.hfq("close"), 126)
            out = pos / 125.0
            tc = ctx.roll_count(_traded(ctx).astype(np.float64), 126)
            out = np.where(tc >= 100, out, np.nan)
            return _final(ctx, out)

        if variant == "drawdown_recovery_days_126":
            out = _rolling_recovery_days(ctx, ctx.hfq("close"), 126, min_traded=100)
            return _final(ctx, out)

        if variant == "peak_timing_63":
            ago = ctx.roll_argmax(ctx.hfq("close"), 63)
            out = (62.0 - ago) / 62.0
            tc = ctx.roll_count(_traded(ctx).astype(np.float64), 63)
            out = np.where(tc >= 50, out, np.nan)
            return _final(ctx, out)

        if variant == "close_location_autocorr_63":
            c = _clv(ctx)
            out = ctx.roll_corr(c, ctx.shift(c, 1), 63, 40)
            return _final(ctx, out)

        if variant == "drawdown_velocity_63":
            dd, dur = _rolling_maxdd(ctx, ctx.hfq("close"), 63, min_traded=50)
            with np.errstate(all="ignore"):
                out = np.where(dur >= 1.0, dd / dur, np.nan)
            return _final(ctx, out)

        if variant == "trend_age_126":
            ma20 = ctx.roll_mean(ctx.hfq("close"), 20, min_count=15)
            slope = np.sign(ma20 - ctx.shift(ma20, 1))
            run = _run_length(ctx, slope)
            out = slope * np.minimum(run, 126.0) / 126.0
            out = np.where(np.isfinite(out), out, np.nan)
            return _final(ctx, out)

        if variant == "time_between_extremes_63":
            ago_max = ctx.roll_argmax(ctx.hfq("close"), 63)
            ago_min = ctx.roll_argmin(ctx.hfq("close"), 63)
            out = np.abs(ago_max - ago_min) / 62.0
            tc = ctx.roll_count(_traded(ctx).astype(np.float64), 63)
            out = np.where(tc >= 50, out, np.nan)
            return _final(ctx, out)

        if variant == "upside_path_share_63":
            pos = np.where(r > 0.0, r, 0.0)
            pos = np.where(np.isfinite(r), pos, np.nan)
            num = ctx.roll_sum(pos, 63, min_count=50)
            den = ctx.roll_sum(np.abs(r), 63, min_count=50)
            with np.errstate(all="ignore"):
                out = np.where(den > 1e-8, num / den, np.nan)
            out = np.where(np.isfinite(out) & (out >= 0.0) & (out <= 1.0), out, np.nan)
            return _final(ctx, out)

        if variant == "volume_weighted_momentum_63":
            vol = ctx.px("vol")
            v = np.where(np.isfinite(r) & np.isfinite(vol) & (vol > 0.0), vol, np.nan)
            num = ctx.roll_sum(np.where(np.isfinite(v), r * v, np.nan), 63, 50)
            den = ctx.roll_sum(v, 63, 50)
            with np.errstate(all="ignore"):
                out = np.where(den > 0.0, num / den, np.nan)
            return _final(ctx, out)

        if variant == "trend_strength_60":
            out = _trend_r2(ctx, ctx.hfq("close"), 60, min_traded=40)
            return _final(ctx, out)

        if variant == "overnight_autocorr_63":
            ron = _overnight(ctx)
            out = ctx.roll_corr(ron, ctx.shift(ron, 1), 63, 50)
            return _final(ctx, out)

        raise KeyError(variant)

    return fn
