"""R17 seat_intraday batch 3: intraday volatility / range structure.

Economic object
---------------
Intraday variance is not one number.  Range-based estimators (Parkinson,
Garman-Klass, Rogers-Satchell) use different parts of the OHLC path; session
ranges split the day into morning/afternoon volatility; the drift share
separates directional displacement from bar-to-bar noise; semideviations split
upside from downside; the jump-share proxy separates squared variation
explained by the daily range from variation outside it.  These are distinct
volatility descriptors, not levels of the same RV series.

Conventions
-----------
* The layer's bar returns are SIMPLE returns; `ret2_sum` is a sum of squared
  simple returns and `ret_sum` is the sum of simple returns over the 47
  measurable bars (the first bar has no previous close in the segment).
* Only completed 48-bar days with finite positive OHLC/VWAP/amount/volume and
  the traded/universe masks.  Incomplete days are NaN, never zero.
* High/low are expanded with the day's own open/close before any range ratio.
* `jump_share_proxy` and `drift_variance_share` are explicitly proxies, not
  bipower-variation jump tests: the layer has no lag-1 cross products.
* Volumes are not used as levels, so the hand/share switch is irrelevant.

C directions/horizons are frozen before labels.
"""
from __future__ import annotations

import numpy as np

WARMUP_DAYS = 80

PARAMETERS = {
    "vol_gc_garman_klass": {"variant": "vol_gc_garman_klass"},
    "vol_gc_rogers_satchell": {"variant": "vol_gc_rogers_satchell"},
    "vol_gc_am_pm_parkinson_ratio": {"variant": "vol_gc_am_pm_parkinson_ratio"},
    "vol_gc_pm_range_share": {"variant": "vol_gc_pm_range_share"},
    "vol_gc_close30_upper_range": {"variant": "vol_gc_close30_upper_range"},
    "vol_gc_drift_variance_share": {"variant": "vol_gc_drift_variance_share"},
    "vol_gc_upside_semideviation": {"variant": "vol_gc_upside_semideviation"},
    "vol_gc_am_pm_squared_return_ratio": {"variant": "vol_gc_am_pm_squared_return_ratio"},
    "vol_c_jump_share_proxy": {"variant": "vol_c_jump_share_proxy"},
    "vol_c_downside_semideviation": {"variant": "vol_c_downside_semideviation"},
    "vol_c_range_expansion_20": {"variant": "vol_c_range_expansion_20"},
    "vol_c_close30_range_share": {"variant": "vol_c_close30_range_share"},
}

LOG2 = np.log(2.0)


def _cache(ctx):
    box = ctx.__dict__.get("_r17_vol_cache")
    if box is None:
        box = {}
        ctx.__dict__["_r17_vol_cache"] = box
    return box


def _traded(ctx):
    box = _cache(ctx)
    if "traded" not in box:
        box["traded"] = np.asarray(ctx.traded(), dtype=bool)
    return box["traded"]


def _base(ctx):
    box = _cache(ctx)
    if "base" in box:
        return box["base"]

    def f(name):
        return np.asarray(ctx.intraday_field(name), dtype=np.float64)

    n = f("n_bars")
    o, cl, hi, lo = f("open5"), f("close5"), f("hi5"), f("lo5")
    vwap, amt, vol = f("vwap"), f("amt"), f("vol")
    ret_sum, ret2 = f("ret_sum"), f("ret2_sum")
    pos2, neg2 = f("posret2_sum"), f("negret2_sum")
    amo, amc, amhi, amlo = f("am_open5"), f("am_close5"), f("am_hi"), f("am_lo")
    pmo, pmc, pmhi, pmlo = f("pm_open5"), f("pm_close5"), f("pm_hi"), f("pm_lo")

    hi_e = np.maximum(hi, np.fmax(o, cl))
    lo_e = np.minimum(lo, np.fmin(o, cl))
    amhi_e = np.maximum(amhi, np.fmax(amo, amc))
    amlo_e = np.minimum(amlo, np.fmin(amo, amc))
    pmhi_e = np.maximum(pmhi, np.fmax(pmo, pmc))
    pmlo_e = np.minimum(pmlo, np.fmin(pmo, pmc))

    ok = (n == 48) & np.asarray(ctx.universe, dtype=bool) & _traded(ctx)
    for arr in (o, cl, hi_e, lo_e, vwap, amt, vol, ret_sum, ret2, pos2, neg2,
                amo, amc, amhi_e, amlo_e, pmo, pmc, pmhi_e, pmlo_e):
        ok &= np.isfinite(arr)
    ok &= (o > 0) & (cl > 0) & (vwap > 0) & (amt > 0) & (vol > 0)
    ok &= (amo > 0) & (amc > 0) & (pmo > 0) & (pmc > 0)

    with np.errstate(all="ignore"):
        rng = np.log(hi_e / lo_e)
        rng_am = np.log(amhi_e / amlo_e)
        rng_pm = np.log(pmhi_e / pmlo_e)
        r_am = np.log(amc / amo)
        r_pm = np.log(pmc / pmo)
        gk = 0.5 * rng ** 2 - (2.0 * LOG2 - 1.0) * np.log(cl / o) ** 2
        rs = (np.log(hi_e / cl) * np.log(hi_e / o)
              + np.log(lo_e / cl) * np.log(lo_e / o))
        park_var = rng ** 2 / (4.0 * LOG2)
        drift_share = ret_sum ** 2 / ((n - 1.0) * ret2)

    base = {
        "ok": ok, "n": n, "o": o, "cl": cl, "hi_e": hi_e, "lo_e": lo_e,
        "rng": rng, "rng_am": rng_am, "rng_pm": rng_pm,
        "r_am": r_am, "r_pm": r_pm,
        "gk": gk, "rs": rs, "park_var": park_var, "ret2": ret2,
        "pos2": pos2, "neg2": neg2, "drift_share": drift_share,
    }
    box["base"] = base
    return base


def _close30(ctx):
    box = _cache(ctx)
    if "close30" not in box:
        g = lambda k: np.asarray(ctx.closing30_field(k), dtype=np.float64)
        grid = g("grid_ok")
        ot, ht, ct = g("open_tail30"), g("high_tail30"), g("close_tail30")
        b = _base(ctx)
        ok = (b["ok"] & (grid == 1) & np.isfinite(ot) & np.isfinite(ht) & np.isfinite(ct)
              & (ot > 0) & (ht > 0) & (ct > 0))
        ht_e = np.maximum(ht, np.fmax(ot, ct))
        box["close30"] = {"ok": ok, "open_tail30": ot, "high_tail30": ht_e,
                          "close_tail30": ct}
    return box["close30"]


def _mask(x, ok):
    return np.where(ok, x, np.nan)


def build(parameters):
    variant = parameters["variant"]
    if variant not in {v["variant"] for v in PARAMETERS.values()}:
        raise KeyError(variant)

    def fn(ctx):
        b = _base(ctx)
        ok = b["ok"]

        # ---------------- GC: range-based variance estimators --------------
        if variant == "vol_gc_garman_klass":
            # Daily GK variance estimate; negative values are estimator noise
            # when the open-close term dominates and are reported, not clipped.
            return _mask(b["gk"], ok)

        if variant == "vol_gc_rogers_satchell":
            return _mask(b["rs"], ok)

        # ---------------- GC: session range allocation ----------------------
        if variant == "vol_gc_am_pm_parkinson_ratio":
            out = np.where(b["rng_pm"] > 0,
                           b["rng_am"] / np.where(b["rng_pm"] > 0, b["rng_pm"], np.nan),
                           np.nan)
            return _mask(out, ok)

        if variant == "vol_gc_pm_range_share":
            out = np.where(b["rng"] > 0,
                           b["rng_pm"] / np.where(b["rng"] > 0, b["rng"], np.nan),
                           np.nan)
            # real vendor OHLC can make the PM session range marginally exceed
            # the full-day range; a share is bounded [0,1] by construction.
            return _mask(np.clip(out, 0.0, 1.0), ok)

        if variant == "vol_gc_am_pm_squared_return_ratio":
            out = np.where((b["r_am"] != 0) & (b["r_pm"] != 0),
                           2.0 * (np.log(np.abs(b["r_am"])) - np.log(np.abs(b["r_pm"]))),
                           np.nan)
            return _mask(out, ok)

        # ---------------- GC: closing-window range --------------------------
        if variant == "vol_gc_close30_upper_range":
            c3 = _close30(ctx)
            out = c3["high_tail30"] / c3["open_tail30"] - 1.0
            return _mask(out, c3["ok"])

        # ---------------- GC: drift vs noise --------------------------------
        if variant == "vol_gc_drift_variance_share":
            out = np.where(b["ret2"] > 0, b["drift_share"], np.nan)
            return _mask(np.clip(out, 0.0, 1.0), ok)

        # ---------------- GC: semideviation levels --------------------------
        if variant == "vol_gc_upside_semideviation":
            return _mask(np.sqrt(np.maximum(b["pos2"], 0.0)), ok)

        # ---------------- C: variance decomposition / shocks ----------------
        if variant == "vol_c_jump_share_proxy":
            # 1 - range-implied variance / squared-bar-return variance.
            # Both are daily-variance units under independent bars; this is a
            # range-vs-RV proxy, not a bipower-variation jump test.
            out = np.where(b["ret2"] > 0,
                           1.0 - b["park_var"] / np.where(b["ret2"] > 0, b["ret2"], np.nan),
                           np.nan)
            return _mask(np.clip(out, 0.0, 1.0), ok)

        if variant == "vol_c_downside_semideviation":
            return _mask(np.sqrt(np.maximum(b["neg2"], 0.0)), ok)

        if variant == "vol_c_range_expansion_20":
            # Volatility shock relative to a robust 20-day baseline. This is a
            # shock state, not a level; the doc's pseudo-standardization warning
            # is acknowledged and the 20-day median is the ex-ante expectation.
            x = b["rng"]
            med = ctx.roll_quantile(x, 20, 0.5, min_count=10)
            out = np.where(np.isfinite(x) & np.isfinite(med) & (med > 0),
                           x / np.where(med > 0, med, np.nan) - 1.0, np.nan)
            return _mask(out, ok)

        if variant == "vol_c_close30_range_share":
            c3 = _close30(ctx)
            den = b["hi_e"] - b["lo_e"]
            num = c3["high_tail30"] - c3["open_tail30"]
            out = np.where(den > 0, num / np.where(den > 0, den, np.nan), np.nan)
            return _mask(np.clip(out, 0.0, 1.0), c3["ok"])

        raise KeyError(variant)

    return fn
