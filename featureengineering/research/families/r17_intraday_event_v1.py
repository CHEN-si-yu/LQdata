"""R17 seat_intraday batch 5: event-conditioned intraday reactions.

Economic object
---------------
How does the intraday path respond to a prior shock?  We condition on (i) a
non-trivial opening gap (|overnight gap| >= 1%) and (ii) a prior-day extreme
move (|close-to-close return at T-1| >= 5%).  Responses are measured inside the
same day T: morning-session continuation/retracement of the gap, lunch-break
response, full-intraday reversal of the prior extreme, closing-30m continuation
of the earlier intraday move, and closing-30m response to the gap or to the
prior extreme.  This is a state-conditioned path family, not a re-labeling of
the unconditional session returns.

Timing / endpoint rules
-----------------------
* Gap = log(hfq_open(T) / hfq_close(T-1)) with traded(T) and traded(T-1).
* Prior-day return = ctx.ret_clean(1) shifted one row, additionally requiring
  traded(T-1) and traded(T-2), so a suspension-resumption return cannot become
  the "prior extreme".
* All response legs are same-day raw-price ratios (no adj factor).
* Only 48-bar days with finite positive prices/amounts; event-conditioned
  candidates are NaN outside the event, never zero.
* Closing-30m legs require closing30 grid_ok==1.
* The layer freezes at 2026-09-11; later days are NaN.

C directions/horizons are frozen before labels.
"""
from __future__ import annotations

import numpy as np

WARMUP_DAYS = 80

GAP_THRESHOLD = 0.01
EXTREME_THRESHOLD = 0.05
RATIO_CAP = 3.0

PARAMETERS = {
    "evt_gc_gap_morning_ratio": {"variant": "evt_gc_gap_morning_ratio"},
    "evt_gc_gap_lunch_ratio": {"variant": "evt_gc_gap_lunch_ratio"},
    "evt_gc_prior_extreme_intraday_reversal": {"variant": "evt_gc_prior_extreme_intraday_reversal"},
    "evt_gc_prior_extreme_close30_response": {"variant": "evt_gc_prior_extreme_close30_response"},
    "evt_c_close30_continuation": {"variant": "evt_c_close30_continuation"},
    "evt_c_gap_close30_response": {"variant": "evt_c_gap_close30_response"},
    "evt_c_lunch_post_extreme_response": {"variant": "evt_c_lunch_post_extreme_response"},
}


def _cache(ctx):
    box = ctx.__dict__.get("_r17_event_cache")
    if box is None:
        box = {}
        ctx.__dict__["_r17_event_cache"] = box
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
    amt, vol, vwap = f("amt"), f("vol"), f("vwap")
    amo, amc = f("am_open5"), f("am_close5")
    pmo, pmc = f("pm_open5"), f("pm_close5")

    hi_e = np.maximum(hi, np.fmax(o, cl))
    lo_e = np.minimum(lo, np.fmin(o, cl))
    ok = (n == 48) & np.asarray(ctx.universe, dtype=bool) & _traded(ctx)
    for arr in (o, cl, hi_e, lo_e, amt, vol, vwap, amo, amc, pmo, pmc):
        ok &= np.isfinite(arr)
    ok &= (o > 0) & (cl > 0) & (amt > 0) & (vol > 0) & (vwap > 0)
    ok &= (amo > 0) & (amc > 0) & (pmo > 0) & (pmc > 0)

    with np.errstate(all="ignore"):
        r_am = np.log(amc / amo)
        r_pm = np.log(pmc / pmo)
        r_lunch = np.log(pmo / amc)
        day_ret = np.log(cl / o)

    # opening gap with strict adjacent-trade endpoints
    tr = _traded(ctx)
    tr_prev = ctx.shift(tr.astype(np.float64), 1) > 0.5
    tr_prev2 = ctx.shift(tr.astype(np.float64), 2) > 0.5
    op = np.asarray(ctx.hfq("open"), dtype=np.float64)
    pc = ctx.shift(np.asarray(ctx.hfq("close"), dtype=np.float64), 1)
    with np.errstate(all="ignore"):
        gap = np.log(op / pc)
    gap_ok = ok & tr_prev & np.isfinite(op) & np.isfinite(pc) & (op > 0) & (pc > 0)

    # prior-day close-to-close return, strict endpoints at T-1 and T-2
    ret1 = np.asarray(ctx.ret_clean(1), dtype=np.float64)
    prior_ret = ctx.shift(ret1, 1)
    prior_ok = ok & tr_prev & tr_prev2 & np.isfinite(prior_ret)

    base = {
        "ok": ok, "o": o, "cl": cl, "hi_e": hi_e, "lo_e": lo_e,
        "r_am": r_am, "r_pm": r_pm, "r_lunch": r_lunch, "day_ret": day_ret,
        "gap": gap, "gap_ok": gap_ok,
        "prior_ret": prior_ret, "prior_ok": prior_ok,
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
        with np.errstate(all="ignore"):
            tail_ret = np.log(ct / ot)
            rest_ret = np.log(ot / b["o"])
        box["close30"] = {"ok": ok, "open_tail30": ot, "high_tail30": ht_e,
                          "close_tail30": ct, "tail_ret": tail_ret, "rest_ret": rest_ret}
    return box["close30"]


def _mask(x, ok):
    return np.where(ok, x, np.nan)


def build(parameters):
    variant = parameters["variant"]
    if variant not in {v["variant"] for v in PARAMETERS.values()}:
        raise KeyError(variant)

    def fn(ctx):
        b = _base(ctx)
        big_gap = b["gap_ok"] & (np.abs(b["gap"]) >= GAP_THRESHOLD)
        extreme = b["prior_ok"] & (np.abs(b["prior_ret"]) >= EXTREME_THRESHOLD)
        sign_gap = np.sign(b["gap"])
        sign_prior = np.sign(b["prior_ret"])

        if variant == "evt_gc_gap_morning_ratio":
            out = np.where(big_gap, b["r_am"] / np.where(big_gap, b["gap"], np.nan), np.nan)
            return _mask(np.clip(out, -RATIO_CAP, RATIO_CAP), big_gap)

        if variant == "evt_gc_gap_lunch_ratio":
            out = np.where(big_gap, b["r_lunch"] / np.where(big_gap, b["gap"], np.nan), np.nan)
            return _mask(np.clip(out, -RATIO_CAP, RATIO_CAP), big_gap)

        if variant == "evt_gc_prior_extreme_intraday_reversal":
            out = -sign_prior * b["day_ret"]
            return _mask(out, extreme)

        if variant == "evt_gc_prior_extreme_close30_response":
            c3 = _close30(ctx)
            out = sign_prior * c3["tail_ret"]
            return _mask(out, extreme & c3["ok"])

        if variant == "evt_c_close30_continuation":
            c3 = _close30(ctx)
            cond = c3["ok"] & (np.abs(c3["rest_ret"]) >= GAP_THRESHOLD)
            out = np.sign(c3["rest_ret"]) * c3["tail_ret"]
            return _mask(out, cond)

        if variant == "evt_c_gap_close30_response":
            c3 = _close30(ctx)
            cond = big_gap & c3["ok"]
            out = sign_gap * c3["tail_ret"]
            return _mask(out, cond)

        if variant == "evt_c_lunch_post_extreme_response":
            out = sign_prior * b["r_lunch"]
            return _mask(out, extreme)

        raise KeyError(variant)

    return fn
