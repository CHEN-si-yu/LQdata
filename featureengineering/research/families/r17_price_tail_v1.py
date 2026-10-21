"""R17 seat_price batch 1: daily return distribution shape / tail / drawdown.

Pure builder for the preregistered family `r17p_a_tail_shape_*`.
All inputs are `stock_daily` + `stock_adj_factor`; no index, no labels, no qfq.
Semantics used throughout:

* `r = ctx.ret_clean(1)` -> posterior-adjusted single-day return with
  `traded(t) & traded(t-1)` endpoints implicitly enforced by the price layer's
  daily return; `ctx.ret_clean` additionally sets non-traded cells to NaN.
* Windows count *traded sessions with a finite return*.  `roll_mean` /
  `roll_std` / `roll_cov` / `roll_corr` in fea.mathx use the effective valid
  count, so min_count windows are not silently diluted by suspension days.
* Rolling skewness / kurtosis are computed from central moments built from
  `roll_mean` of powers (effective-count correct), not from
  `ctx.roll_skew/roll_kurt` whose cumulant path divides by the fixed window
  length even when NaNs are present.
* hfq close is forward-filled on suspension by the price layer.  Drawdown
  constructs intentionally use that state series (the last traded price stays
  the marking price) but require at least 100 traded sessions per window.
* Every output is finally masked by `universe & traded` at T; no winsor, no
  zero fill.
"""
from __future__ import annotations

import numpy as np

WARMUP_DAYS = 1200

PARAMETERS = {
    "ret_kurt_63": {"variant": "ret_kurt_63"},
    "downside_vol_63": {"variant": "downside_vol_63"},
    "downside_var_share_63": {"variant": "downside_var_share_63"},
    "tail_ratio_63": {"variant": "tail_ratio_63"},
    "max_drawdown_126": {"variant": "max_drawdown_126"},
    "ulcer_index_63": {"variant": "ulcer_index_63"},
    "drawdown_entries_126": {"variant": "drawdown_entries_126"},
    "gain_loss_asymmetry_63": {"variant": "gain_loss_asymmetry_63"},
    "extreme_gain_concentration_63": {"variant": "extreme_gain_concentration_63"},
    "max_min_asymmetry_63": {"variant": "max_min_asymmetry_63"},
    "expected_shortfall_5_63": {"variant": "expected_shortfall_5_63"},
    "left_tail_hill_126": {"variant": "left_tail_hill_126"},
    "overnight_skew_63": {"variant": "overnight_skew_63"},
    "overnight_kurt_63": {"variant": "overnight_kurt_63"},
    "skew_instability_252": {"variant": "skew_instability_252"},
    "return_range_ratio_63": {"variant": "return_range_ratio_63"},
    "downside_vol_innov_21_252": {"variant": "downside_vol_innov_21_252"},
    "jump_frequency_63": {"variant": "jump_frequency_63"},
    "max_drawdown_duration_126": {"variant": "max_drawdown_duration_126"},
    "overnight_downside_vol_63": {"variant": "overnight_downside_vol_63"},
}

MIN_RET_VOL = 1e-14


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


def _universe(ctx):
    return ctx.universe


def _final(ctx, value):
    """Mask the value at T only; rolling windows already use point-in-time data."""
    return np.where(_traded(ctx) & _universe(ctx), value, np.nan)


# ------------------------------------------------------------------- moments
def _moments(ctx, x, n, min_count):
    """Central moments (2,3,4) with effective valid counts.

    Uses E[x^k] over the window's valid observations; the fixed-n cumulant
    implementation in fea.mathx is biased when suspensions create NaNs.
    """
    mu = ctx.roll_mean(x, n, min_count)
    m2 = ctx.roll_mean(x * x, n, min_count) - mu * mu
    m3 = ctx.roll_mean(x * x * x, n, min_count) - 3.0 * mu * m2 - mu ** 3
    m4 = (ctx.roll_mean(x ** 4, n, min_count) - 4.0 * mu * m3
          - 6.0 * mu * mu * m2 - mu ** 4)
    return mu, m2, m3, m4


def _skew(ctx, x, n, min_count):
    _, m2, m3, _ = _moments(ctx, x, n, min_count)
    with np.errstate(all="ignore"):
        out = np.where(m2 > MIN_RET_VOL, m3 / np.power(m2, 1.5), np.nan)
    return np.where(np.isfinite(out) & (np.abs(out) <= 100.0), out, np.nan)


def _kurt(ctx, x, n, min_count):
    _, m2, _, m4 = _moments(ctx, x, n, min_count)
    with np.errstate(all="ignore"):
        out = np.where(m2 > MIN_RET_VOL, m4 / (m2 * m2) - 3.0, np.nan)
    return np.where(np.isfinite(out) & (np.abs(out) <= 1e4), out, np.nan)


def _downside_moment(ctx, r, power, n, min_count):
    """E[r^power * 1(r<0)] over valid traded returns in the window."""
    base = np.where(r < 0.0, np.power(np.abs(r), power), 0.0)
    base = np.where(np.isfinite(r), base, np.nan)
    return ctx.roll_mean(base, n, min_count)


def _roll_quantile(ctx, x, n, q, min_count):
    """Rolling linear-interpolation quantile, chunked sort implementation.

    `np.nanquantile` over sliding windows is prohibitively slow here (the
    generic masked path re-sorts each cell and took ~150 s for one 63d
    quantile on a 996x2115 probe).  Sorting a block of windows with NaN mapped
    to +inf gives identical linear-interpolation quantiles for the valid
    prefix and runs ~50x faster.
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
        ok = cnt >= min_count
        out[s + n - 1:e + n - 1] = np.where(ok, val, np.nan)
    return out


def _rolling_maxdd(ctx, px, traded, n, min_traded=100, block=64):
    """Exact maximum peak-to-trough drawdown and its duration per trailing window.

    For each window the running maximum is tracked in window order; the
    deepest drawdown is `min(px_k / running_max_k - 1)` and its duration is
    `k - argmax(running_max)` at that point.  Returns positive magnitudes.
    Full finite positive price window required; also needs min_traded.
    """
    T, C = px.shape
    out_dd = np.full((T, C), np.nan, dtype=np.float64)
    out_dur = np.full((T, C), np.nan, dtype=np.float64)
    if T < n:
        return out_dd, out_dur
    traded_count = ctx.roll_count(traded.astype(np.float64), n)
    Tn = T - n + 1
    for s in range(0, Tn, block):
        e = min(Tn, s + block)
        w = px[s:e + n - 1]
        win = np.lib.stride_tricks.sliding_window_view(w, n, axis=0)
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
        dd_pos = -min_dd
        ok = valid & (traded_count[s + n - 1:e + n - 1] >= min_traded)
        out_dd[s + n - 1:e + n - 1] = np.where(ok, dd_pos, np.nan)
        out_dur[s + n - 1:e + n - 1] = np.where(ok, dur, np.nan)
    return out_dd, out_dur


def _overnight(ctx):
    """Overnight return with real trade endpoints on t and t-1."""
    def _build():
        tr = _traded(ctx)
        prev_tr = ctx.shift(tr.astype(np.float64), 1) > 0.5
        op = ctx.hfq("open")
        cl = ctx.hfq("close")
        prev_cl = ctx.shift(cl, 1)
        with np.errstate(all="ignore"):
            out = ctx.safe_div(op - prev_cl, np.abs(prev_cl), min_abs_den=1e-12)
        return np.where(tr & prev_tr & np.isfinite(out), out, np.nan)
    return _cache(ctx, "overnight_ret", _build)


# --------------------------------------------------------------------- build
def build(parameters):
    variant = parameters["variant"]

    def fn(ctx):
        r = _r(ctx)

        if variant == "ret_kurt_63":
            return _final(ctx, _kurt(ctx, r, 63, 50))

        if variant == "downside_vol_63":
            d2 = _downside_moment(ctx, r, 2.0, 63, 50)
            return _final(ctx, np.sqrt(np.maximum(d2, 0.0)))

        if variant == "downside_var_share_63":
            d2 = _downside_moment(ctx, r, 2.0, 63, 50)
            tot = ctx.roll_mean(r * r, 63, 50)
            with np.errstate(all="ignore"):
                out = np.where(tot > MIN_RET_VOL, d2 / tot, np.nan)
            return _final(ctx, out)

        if variant == "tail_ratio_63":
            q95 = _roll_quantile(ctx, r, 63, 0.95, 50)
            q05 = _roll_quantile(ctx, r, 63, 0.05, 50)
            with np.errstate(all="ignore"):
                out = np.where((-q05) > 1e-6, q95 / (-q05), np.nan)
            out = np.where(np.isfinite(out) & (out >= 0.0) & (out <= 1e4), out, np.nan)
            return _final(ctx, out)

        if variant == "max_drawdown_126":
            px = ctx.hfq("close")
            dd, _ = _rolling_maxdd(ctx, px, _traded(ctx), 126, min_traded=100)
            return _final(ctx, dd)

        if variant == "max_drawdown_duration_126":
            px = ctx.hfq("close")
            _, dur = _rolling_maxdd(ctx, px, _traded(ctx), 126, min_traded=100)
            return _final(ctx, dur)

        if variant == "ulcer_index_63":
            px = ctx.hfq("close")
            hi = ctx.roll_max(px, 63, min_count=50)
            with np.errstate(all="ignore"):
                dd = np.where(hi > 0, px / hi - 1.0, np.nan)
            ui = np.sqrt(np.maximum(ctx.roll_mean(dd * dd, 63, 50), 0.0))
            ok = ctx.roll_count(_traded(ctx).astype(np.float64), 63) >= 50
            return _final(ctx, np.where(ok, ui, np.nan))

        if variant == "drawdown_entries_126":
            px = ctx.hfq("close")
            peak = np.fmax.accumulate(px, axis=0)
            with np.errstate(all="ignore"):
                dd = np.where(peak > 0, px / peak - 1.0, np.nan)
            prev = ctx.shift(dd, 1)
            entry = (dd < -0.05) & np.isfinite(dd) & (~(prev < -0.05))
            entry = entry & _traded(ctx)
            out = ctx.roll_sum(entry.astype(np.float64), 126, min_count=100)
            return _final(ctx, out)

        if variant == "gain_loss_asymmetry_63":
            pos = np.where(r > 0.0, r, np.nan)
            neg = np.where(r < 0.0, -r, np.nan)
            mg = ctx.roll_mean(pos, 63, 10)
            ml = ctx.roll_mean(neg, 63, 10)
            with np.errstate(all="ignore"):
                out = np.where(ml > 1e-8, np.log(np.maximum(mg, 1e-12) / ml), np.nan)
            out = np.where(np.isfinite(out), out, np.nan)
            return _final(ctx, out)

        if variant == "extreme_gain_concentration_63":
            q90 = _roll_quantile(ctx, r, 63, 0.90, 50)
            tail = (r >= q90) & np.isfinite(r) & np.isfinite(q90)
            r2 = np.where(np.isfinite(r), r * r, np.nan)
            num = ctx.roll_sum(np.where(tail, r2, np.where(np.isfinite(r2), 0.0, np.nan)), 63, 50)
            den = ctx.roll_sum(r2, 63, 50)
            with np.errstate(all="ignore"):
                out = np.where(den > MIN_RET_VOL, num / den, np.nan)
            out = np.where(np.isfinite(out) & (out >= 0.0) & (out <= 1.0), out, np.nan)
            return _final(ctx, out)

        if variant == "max_min_asymmetry_63":
            q99 = _roll_quantile(ctx, r, 63, 0.99, 50)
            q50 = _roll_quantile(ctx, r, 63, 0.50, 50)
            q01 = _roll_quantile(ctx, r, 63, 0.01, 50)
            with np.errstate(all="ignore"):
                out = np.where((q50 - q01) > 1e-6, (q99 - q50) / (q50 - q01), np.nan)
            out = np.where(np.isfinite(out) & (out >= 0.0) & (out <= 1e4), out, np.nan)
            return _final(ctx, out)

        if variant == "expected_shortfall_5_63":
            q05 = _roll_quantile(ctx, r, 63, 0.05, 50)
            tail = r <= q05
            mean_tail = ctx.roll_mean(np.where(tail, r, np.nan), 63, 3)
            return _final(ctx, -mean_tail)

        if variant == "left_tail_hill_126":
            loss = np.where(r < 0.0, -r, np.nan)
            u = _roll_quantile(ctx, loss, 126, 0.90, 40)
            with np.errstate(all="ignore"):
                exceed = np.where(loss >= u, np.log(loss / u), np.nan)
            gamma = ctx.roll_mean(exceed, 126, 5)
            with np.errstate(all="ignore"):
                out = np.where(gamma > 1e-6, 1.0 / gamma, np.nan)
            out = np.where(np.isfinite(out) & (out > 0.0) & (out <= 100.0), out, np.nan)
            return _final(ctx, out)

        if variant == "overnight_skew_63":
            return _final(ctx, _skew(ctx, _overnight(ctx), 63, 50))

        if variant == "overnight_kurt_63":
            return _final(ctx, _kurt(ctx, _overnight(ctx), 63, 50))

        if variant == "skew_instability_252":
            s20 = _skew(ctx, r, 20, 15)
            out = ctx.roll_std(s20, 252, 200)
            return _final(ctx, out)

        if variant == "return_range_ratio_63":
            q75 = _roll_quantile(ctx, r, 63, 0.75, 50)
            q25 = _roll_quantile(ctx, r, 63, 0.25, 50)
            q90 = _roll_quantile(ctx, r, 63, 0.90, 50)
            q10 = _roll_quantile(ctx, r, 63, 0.10, 50)
            with np.errstate(all="ignore"):
                out = np.where((q90 - q10) > 1e-8, (q75 - q25) / (q90 - q10), np.nan)
            out = np.where(np.isfinite(out) & (out >= 0.0) & (out <= 1.0), out, np.nan)
            return _final(ctx, out)

        if variant == "downside_vol_innov_21_252":
            d21 = np.sqrt(np.maximum(_downside_moment(ctx, r, 2.0, 21, 18), 0.0))
            d252 = np.sqrt(np.maximum(_downside_moment(ctx, r, 2.0, 252, 200), 0.0))
            with np.errstate(all="ignore"):
                out = np.where(d252 > 1e-8, d21 / d252 - 1.0, np.nan)
            return _final(ctx, out)

        if variant == "jump_frequency_63":
            sigma = ctx.shift(ctx.roll_std(r, 252, 200), 1)
            jump = (np.abs(r) > 3.0 * sigma) & np.isfinite(sigma)
            out = ctx.roll_mean(jump.astype(np.float64), 63, 50)
            return _final(ctx, out)

        if variant == "overnight_downside_vol_63":
            ron = _overnight(ctx)
            base = np.where(ron < 0.0, ron * ron, 0.0)
            base = np.where(np.isfinite(ron), base, np.nan)
            out = np.sqrt(np.maximum(ctx.roll_mean(base, 63, 50), 0.0))
            return _final(ctx, out)

        raise KeyError(variant)

    return fn
