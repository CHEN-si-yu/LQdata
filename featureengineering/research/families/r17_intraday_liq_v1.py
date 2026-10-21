"""R17 seat_intraday batch 4: intraday liquidity / price impact.

Economic object
---------------
How much price movement does a yuan of trading buy, and when?  The 5-minute
layer gives same-day amount, session amounts, VWAPs and closing-window amounts.
Combining them yields intraday impact measures (return / realized variance /
range per yuan), a closing-vs-rest illiquidity asymmetry, an opening-gap
impact per opening yuan, VWAP-based session price drift, and session-specific
liquidity percentiles relative to each stock's own 20-day history.

Conventions
-----------
* Amount is yuan in every year (verified against stock_daily.amount to <0.1%),
  so the 2025-12 hand/share vol switch cannot affect any level used here.
  VWAPs are amount/volume ratios; the per-(stock,day) unit multiplier cancels.
* Only completed 48-bar days with finite positive prices/amounts and the
  traded/universe masks.  Incomplete days are NaN, never zero.
* The opening-gap impact uses only actual adjacent trades: hfq_open(T) and
  hfq_close(T-1) are required, and traded(T-1)&traded(T) must both hold, so a
  forward-filled suspension price cannot create a synthetic gap.
* Closing-window fields require closing30 grid_ok==1.
* Rolling percentiles use a 20-day window with at least 10 valid days and an
  explicit current-value finite guard, so a stale rank cannot survive a
  suspension or the 2026-09-11 source freeze.
"""
from __future__ import annotations

import numpy as np

WARMUP_DAYS = 80


def _variants():
    gc = [
        "liq_gc_intraday_amihud",
        "liq_gc_intraday_rv_per_amount",
        "liq_gc_intraday_range_per_amount",
        "liq_gc_close30_impact",
        "liq_gc_tail_vs_rest_illiquidity",
        "liq_gc_open5_gap_impact",
        "liq_gc_am_pm_vwap_spread",
        "liq_gc_vwap_close_divergence",
        "liq_gc_signed_intraday_amihud",
        "liq_gc_open5_amount_ts_rank_20",
        "liq_gc_tail30_amount_ts_rank_20",
        "liq_gc_am_amount_ts_rank_20",
    ]
    return {v: {"variant": v} for v in gc}


PARAMETERS = _variants()


def _cache(ctx):
    box = ctx.__dict__.get("_r17_liq_cache")
    if box is None:
        box = {}
        ctx.__dict__["_r17_liq_cache"] = box
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
    amo, amc, amhi, amlo = f("am_open5"), f("am_close5"), f("am_hi"), f("am_lo")
    amvol, amamt = f("am_vol"), f("am_amt")
    pmo, pmc, pmhi, pmlo = f("pm_open5"), f("pm_close5"), f("pm_hi"), f("pm_lo")
    pmvol, pmamt = f("pm_vol"), f("pm_amt")

    hi_e = np.maximum(hi, np.fmax(o, cl))
    lo_e = np.minimum(lo, np.fmin(o, cl))

    ok = (n == 48) & np.asarray(ctx.universe, dtype=bool) & _traded(ctx)
    for arr in (o, cl, hi_e, lo_e, vwap, amt, vol, ret_sum, ret2,
                amo, amc, amhi, amlo, amvol, amamt,
                pmo, pmc, pmhi, pmlo, pmvol, pmamt):
        ok &= np.isfinite(arr)
    ok &= (o > 0) & (cl > 0) & (vwap > 0) & (amt > 0) & (vol > 0)
    ok &= (amo > 0) & (amc > 0) & (amvol > 0) & (amamt > 0)
    ok &= (pmo > 0) & (pmc > 0) & (pmvol > 0) & (pmamt > 0)

    with np.errstate(all="ignore"):
        day_ret = cl / o - 1.0
        rv = np.sqrt(ret2)
        rng = np.log(hi_e / lo_e)
        am_vwap = amamt / amvol
        pm_vwap = pmamt / pmvol
        am_vwap_spread = pm_vwap / am_vwap - 1.0
        close_spread = pmc / amc - 1.0

    base = {
        "ok": ok, "o": o, "cl": cl, "hi_e": hi_e, "lo_e": lo_e,
        "amt": amt, "rv": rv, "rng": rng, "day_ret": day_ret,
        "am_vwap": am_vwap, "pm_vwap": pm_vwap,
        "am_vwap_spread": am_vwap_spread, "close_spread": close_spread,
        "amamt": amamt, "pmamt": pmamt, "vol": vol,
    }
    box["base"] = base
    return base


def _close30(ctx):
    box = _cache(ctx)
    if "close30" not in box:
        g = lambda k: np.asarray(ctx.closing30_field(k), dtype=np.float64)
        grid = g("grid_ok")
        ad, at = g("amt_day"), g("amt_tail30")
        ot, ht, ct = g("open_tail30"), g("high_tail30"), g("close_tail30")
        b = _base(ctx)
        ok = (b["ok"] & (grid == 1)
              & np.isfinite(ad) & np.isfinite(at) & np.isfinite(ot)
              & np.isfinite(ht) & np.isfinite(ct)
              & (ad > 0) & (at > 0) & (ot > 0) & (ht > 0) & (ct > 0))
        ht_e = np.maximum(ht, np.fmax(ot, ct))
        box["close30"] = {"ok": ok, "amt_day": ad, "amt_tail30": at,
                          "open_tail30": ot, "high_tail30": ht_e, "close_tail30": ct}
    return box["close30"]


def _open5(ctx):
    box = _cache(ctx)
    if "open5" not in box:
        g = lambda k: np.asarray(ctx.open5_field(k), dtype=np.float64)
        nb, fh = g("n_bars"), g("first_hm")
        ao, ad = g("amt_open5"), g("amt_day")
        b = _base(ctx)
        ok = (b["ok"] & (nb == 48) & (fh == 935.0)
              & np.isfinite(ao) & np.isfinite(ad) & (ao > 0) & (ad > 0))
        box["open5"] = {"ok": ok, "amt_open5": ao, "amt_day": ad}
    return box["open5"]


def _ts_rank(ctx, x, min_count=10):
    rank = ctx.roll_rank(x, 20, min_count=min_count)
    return np.where(np.isfinite(x), rank, np.nan)


def _mask(x, ok):
    return np.where(ok, x, np.nan)


def build(parameters):
    variant = parameters["variant"]
    if variant not in {v["variant"] for v in PARAMETERS.values()}:
        raise KeyError(variant)

    def fn(ctx):
        b = _base(ctx)
        ok = b["ok"]

        # ---------------- impact per yuan (same-day) ------------------------
        if variant == "liq_gc_intraday_amihud":
            return _mask(np.abs(b["day_ret"]) / (b["amt"] / 1e8), ok)

        if variant == "liq_gc_intraday_rv_per_amount":
            return _mask(b["rv"] / (b["amt"] / 1e8), ok)

        if variant == "liq_gc_intraday_range_per_amount":
            return _mask(b["rng"] / (b["amt"] / 1e8), ok)

        if variant == "liq_gc_signed_intraday_amihud":
            return _mask(b["day_ret"] / (b["amt"] / 1e8), ok)

        # ---------------- closing-window impact -----------------------------
        if variant == "liq_gc_close30_impact":
            c3 = _close30(ctx)
            tail_ret = c3["close_tail30"] / c3["open_tail30"] - 1.0
            return _mask(np.abs(tail_ret) / (c3["amt_tail30"] / 1e8), c3["ok"])

        if variant == "liq_gc_tail_vs_rest_illiquidity":
            c3 = _close30(ctx)
            with np.errstate(all="ignore"):
                tail_ret = c3["close_tail30"] / c3["open_tail30"] - 1.0
                rest_ret = c3["open_tail30"] / b["o"] - 1.0
                rest_amt = c3["amt_day"] - c3["amt_tail30"]
                i_tail = np.log1p(np.abs(tail_ret) * 1e8 / c3["amt_tail30"])
                i_rest = np.where(rest_amt > 0,
                                  np.log1p(np.abs(rest_ret) * 1e8 / np.where(rest_amt > 0, rest_amt, np.nan)),
                                  np.nan)
                out = i_tail - i_rest
            return _mask(out, c3["ok"])

        # ---------------- opening-gap impact per opening yuan ---------------
        if variant == "liq_gc_open5_gap_impact":
            o5 = _open5(ctx)
            op = np.asarray(ctx.hfq("open"), dtype=np.float64)
            pc = ctx.shift(np.asarray(ctx.hfq("close"), dtype=np.float64), 1)
            tr = _traded(ctx)
            tr_prev = ctx.shift(tr.astype(np.float64), 1) > 0.5
            with np.errstate(all="ignore"):
                gap = np.log(op / pc)
                out = np.abs(gap) / (o5["amt_open5"] / 1e8)
            good = o5["ok"] & tr & tr_prev & np.isfinite(op) & np.isfinite(pc) & (op > 0) & (pc > 0)
            return _mask(out, good)

        # ---------------- VWAP-based session drift --------------------------
        if variant == "liq_gc_am_pm_vwap_spread":
            return _mask(b["am_vwap_spread"], ok)

        if variant == "liq_gc_vwap_close_divergence":
            return _mask(b["am_vwap_spread"] - b["close_spread"], ok)

        # ---------------- session-specific liquidity percentiles ------------
        if variant == "liq_gc_open5_amount_ts_rank_20":
            o5 = _open5(ctx)
            return _mask(_ts_rank(ctx, o5["amt_open5"]), o5["ok"])

        if variant == "liq_gc_tail30_amount_ts_rank_20":
            c3 = _close30(ctx)
            return _mask(_ts_rank(ctx, c3["amt_tail30"]), c3["ok"])

        if variant == "liq_gc_am_amount_ts_rank_20":
            return _mask(_ts_rank(ctx, b["amamt"]), ok)

        raise KeyError(variant)

    return fn
