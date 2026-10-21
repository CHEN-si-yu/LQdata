"""R17 seat_intraday batch 2: intraday path / order statistics.

Economic object
---------------
The same day return and volatility can be produced by very different intraday
paths: a smooth trend, a V-shaped reversal, a single jump, or a choppy walk.
The 5-minute layer pre-aggregates path statistics that are not recoverable from
daily OHLC: running-peak drawdown, running-trough run-up, the time positions of
the high and low, gross path length, up/down/zero bar counts and the sign
partition of squared bar returns.  This family turns those into descriptors and
a small set of ex-ante directional candidates.

Data contract
-------------
* Only completed 48-bar days (`n_bars == 48`) with finite positive OHLC, VWAP,
  amount and volume; traded and universe masks applied.  Incomplete days
  (including the 107 mixed-unit 49-bar 2026 rows) are NaN, never zero.
* `hi5/lo5` are expanded with the day's own open/close before any range ratio
  because vendor high/low do not always bracket their own bar (0.075% of days).
* `n_zero` includes one artificial zero for the first bar (no previous close in
  the segment); every zero-count statistic therefore subtracts 1.  `n_up` does
  not include it.
* The layer freezes at 2026-09-11; later evaluation days are NaN.
* Volumes are never used as levels here, so the hand/share switch is
  irrelevant (only counts and close-price path statistics are used).

Variant directions/horizons for C candidates are frozen before labels.
"""
from __future__ import annotations

import numpy as np

WARMUP_DAYS = 60

PARAMETERS = {
    # ---- GC descriptors -------------------------------------------------
    "path_gc_lo_pos": {"variant": "path_gc_lo_pos"},
    "path_gc_efficiency_abs": {"variant": "path_gc_efficiency_abs"},
    "path_gc_zero_bar_share": {"variant": "path_gc_zero_bar_share"},
    "path_gc_chop_ratio": {"variant": "path_gc_chop_ratio"},
    "path_gc_vwap_range_position": {"variant": "path_gc_vwap_range_position"},
    "path_gc_downside_variance_share": {"variant": "path_gc_downside_variance_share"},
    "path_gc_runup_path_share": {"variant": "path_gc_runup_path_share"},
    "path_gc_upper_shadow_share": {"variant": "path_gc_upper_shadow_share"},
    "path_gc_lower_shadow_share": {"variant": "path_gc_lower_shadow_share"},
    "path_gc_open_position": {"variant": "path_gc_open_position"},
    "path_gc_vwap_vs_open": {"variant": "path_gc_vwap_vs_open"},
    # ---- C candidates ---------------------------------------------------
    "path_c_max_drawdown": {"variant": "path_c_max_drawdown"},
    "path_c_max_runup": {"variant": "path_c_max_runup"},
    "path_c_signed_efficiency": {"variant": "path_c_signed_efficiency"},
    "path_c_hi_pos": {"variant": "path_c_hi_pos"},
    "path_c_up_bar_share": {"variant": "path_c_up_bar_share"},
    "path_c_gross_path_length": {"variant": "path_c_gross_path_length"},
    "path_c_vwap_deviation": {"variant": "path_c_vwap_deviation"},
}


def _cache(ctx):
    box = ctx.__dict__.get("_r17_path_cache")
    if box is None:
        box = {}
        ctx.__dict__["_r17_path_cache"] = box
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
    ret_sum, ret2, absret = f("ret_sum"), f("ret2_sum"), f("absret_sum")
    pos2, neg2 = f("posret2_sum"), f("negret2_sum")
    n_up, n_zero = f("n_up"), f("n_zero")
    max_dd, max_ru = f("max_dd"), f("max_ru")
    hi_pos, lo_pos = f("hi_pos"), f("lo_pos")

    hi_e = np.maximum(hi, np.fmax(o, cl))
    lo_e = np.minimum(lo, np.fmin(o, cl))

    ok = (n == 48) & np.asarray(ctx.universe, dtype=bool) & _traded(ctx)
    for arr in (o, cl, hi_e, lo_e, vwap, amt, vol, ret_sum, ret2, absret, pos2, neg2,
                n_up, n_zero, max_dd, max_ru, hi_pos, lo_pos):
        ok &= np.isfinite(arr)
    ok &= (o > 0) & (cl > 0) & (vwap > 0) & (amt > 0) & (vol > 0)

    with np.errstate(all="ignore"):
        day_ret = np.log(cl / o)
        rng = np.log(hi_e / lo_e)
        rng_simple = hi_e / lo_e - 1.0
        body = np.abs(cl - o)
        upper = hi_e - np.maximum(o, cl)
        lower = np.minimum(o, cl) - lo_e
        n_down = n - n_up - n_zero                 # 47 real returns minus up minus true zeros
        denom_up = n_up + n_down
        n_true_zero = np.maximum(n_zero - 1.0, 0.0)
        var_den = pos2 + neg2
        ru_dd_den = max_ru - max_dd

    base = {
        "ok": ok, "n": n, "o": o, "cl": cl, "hi_e": hi_e, "lo_e": lo_e,
        "vwap": vwap, "ret_sum": ret_sum, "absret": absret, "ret2": ret2,
        "pos2": pos2, "neg2": neg2,
        "n_up": n_up, "n_zero": n_zero, "n_down": n_down,
        "max_dd": max_dd, "max_ru": max_ru, "hi_pos": hi_pos, "lo_pos": lo_pos,
        "day_ret": day_ret, "rng": rng, "rng_simple": rng_simple,
        "body": body, "upper": upper, "lower": lower,
        "denom_up": denom_up, "n_true_zero": n_true_zero, "var_den": var_den,
        "ru_dd_den": ru_dd_den,
    }
    box["base"] = base
    return base


def _mask(x, ok):
    return np.where(ok, x, np.nan)


def build(parameters):
    variant = parameters["variant"]
    if variant not in {v["variant"] for v in PARAMETERS.values()}:
        raise KeyError(variant)

    def fn(ctx):
        b = _base(ctx)
        ok = b["ok"]

        # ---------------- GC: timing / shape descriptors --------------------
        if variant == "path_gc_lo_pos":
            return _mask(b["lo_pos"], ok)

        if variant == "path_gc_efficiency_abs":
            den = b["hi_e"] - b["lo_e"]
            out = np.where(den > 0, b["body"] / np.where(den > 0, den, np.nan), np.nan)
            return _mask(np.clip(out, 0.0, 1.0), ok)

        if variant == "path_gc_zero_bar_share":
            out = b["n_true_zero"] / (b["n"] - 1.0)
            return _mask(np.clip(out, 0.0, 1.0), ok)

        if variant == "path_gc_chop_ratio":
            den = b["rng_simple"]
            out = np.where(den > 0, b["absret"] / np.where(den > 0, den, np.nan), np.nan)
            return _mask(out, ok)

        if variant == "path_gc_vwap_range_position":
            den = b["hi_e"] - b["lo_e"]
            out = np.where(den > 0, (b["vwap"] - b["lo_e"]) / np.where(den > 0, den, np.nan), np.nan)
            return _mask(out, ok)

        if variant == "path_gc_open_position":
            den = b["hi_e"] - b["lo_e"]
            out = np.where(den > 0, (b["o"] - b["lo_e"]) / np.where(den > 0, den, np.nan), np.nan)
            return _mask(np.clip(out, 0.0, 1.0), ok)

        if variant == "path_gc_vwap_vs_open":
            return _mask(b["vwap"] / b["o"] - 1.0, ok)

        if variant == "path_gc_downside_variance_share":
            den = b["var_den"]
            out = np.where(den > 0, b["neg2"] / np.where(den > 0, den, np.nan), np.nan)
            return _mask(np.clip(out, 0.0, 1.0), ok)

        if variant == "path_gc_runup_path_share":
            den = b["ru_dd_den"]
            out = np.where(den > 0, b["max_ru"] / np.where(den > 0, den, np.nan), np.nan)
            return _mask(np.clip(out, 0.0, 1.0), ok)

        if variant == "path_gc_upper_shadow_share":
            den = b["hi_e"] - b["lo_e"]
            out = np.where(den > 0, b["upper"] / np.where(den > 0, den, np.nan), np.nan)
            return _mask(np.clip(out, 0.0, 1.0), ok)

        if variant == "path_gc_lower_shadow_share":
            den = b["hi_e"] - b["lo_e"]
            out = np.where(den > 0, b["lower"] / np.where(den > 0, den, np.nan), np.nan)
            return _mask(np.clip(out, 0.0, 1.0), ok)

        # ---------------- C candidates --------------------------------------
        if variant == "path_c_max_drawdown":
            return _mask(b["max_dd"], ok)

        if variant == "path_c_max_runup":
            return _mask(b["max_ru"], ok)

        if variant == "path_c_signed_efficiency":
            # Both legs are the layer's simple bar-return sums over the 47
            # measurable bars; the first bar is excluded by the layer.
            out = np.where(b["absret"] > 0,
                           b["ret_sum"] / np.where(b["absret"] > 0, b["absret"], np.nan),
                           np.nan)
            return _mask(np.clip(out, -1.0, 1.0), ok)

        if variant == "path_c_hi_pos":
            return _mask(b["hi_pos"], ok)

        if variant == "path_c_up_bar_share":
            out = b["n_up"] / (b["n"] - 1.0)
            return _mask(np.clip(out, 0.0, 1.0), ok)

        if variant == "path_c_gross_path_length":
            return _mask(b["absret"], ok)

        if variant == "path_c_vwap_deviation":
            return _mask(b["cl"] / b["vwap"] - 1.0, ok)

        raise KeyError(variant)

    return fn
