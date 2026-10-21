"""R17 seat_intraday batch 1: session allocation (morning / lunch / afternoon).

Economic object
---------------
Where inside the trading day does the day's price movement, liquidity and range
arrive?  The 5-minute grid gives four mutually exclusive price windows
(previous close -> 09:30 is *overnight*, 09:30-11:30 is the morning session,
11:30-13:00 is the *lunch break*, 13:00-15:00 is the afternoon session).  The
overnight leg is already covered by production factors (`idt_overnight_*`,
`r15t3_overnight_variance_share`); this family therefore allocates *within* the
intraday part and uses the lunch break as a separate information channel.

Data contract (verified by `probe_intraday_grid.py`, no labels)
---------------------------------------------------------------
* `stock_history_5min` bar timestamps are right endpoints; a complete day is
  exactly the 48-slot grid 09:35..11:30 + 13:05..15:00, first_hm=935,
  last_hm=1500.  Sampled 2016..2026 stock-days are 100% exact-grid.
* In 2026-07/08 a 49th 13:00 auction bar appears for 107 (stock, day) rows.
  Those rows are rejected by ``n_bars == 48``; they are also the only mixed
  hand/share unit days.  `stock_history_5min` freezes at 2026-09-11, so the
  last three evaluation weeks are legitimately NaN.
* The `vol` unit is hand before the 2025-12 switch and share afterwards, but
  the switch is **per stock and multi-phase** in 2026 (e.g. 167 of 2112 frozen
  stocks still share-unit on 2026-01-15).  `fea/intraday.py` normalizes each
  (stock, day) by `amt/vol/close`; this probe verified the normalized `vol`
  matches `stock_daily.vol` and `amount` matches `stock_daily.amount` to
  <0.1% in every sampled year/month.  The layer's normalization is
  authoritative.  This family nevertheless prefers **amount (yuan)** for all
  liquidity shares, and uses volume only through the unit-free argmax
  `vol_peak_pos`, so the unit switch cannot leak into a level.
* Incomplete days are NaN, never zero: the shared mask requires n_bars==48 and
  finite positive prices/amounts in both sessions plus the closing30/open5
  grid checks where those layers are used.
* All price ratios are same-day ratios (no cross-day comparison), so the
  unadjusted intraday prices are used directly and no adj_factor is needed.

Variants are frozen before any label is read.  C variants carry an explicit
ex-ante direction and h1/h5; GC variants are pure descriptors.
"""
from __future__ import annotations

import numpy as np

WARMUP_DAYS = 60

# variant -> parameters (parameters must be preregistered verbatim)
PARAMETERS = {
    # ---- GC descriptors -------------------------------------------------
    "sess_gc_am_return_share": {"variant": "sess_gc_am_return_share"},
    "sess_gc_lunch_abs_share": {"variant": "sess_gc_lunch_abs_share"},
    "sess_gc_session_alignment": {"variant": "sess_gc_session_alignment"},
    "sess_gc_am_amount_share": {"variant": "sess_gc_am_amount_share"},
    "sess_gc_open5_amount_share": {"variant": "sess_gc_open5_amount_share"},
    "sess_gc_tail30_amount_share": {"variant": "sess_gc_tail30_amount_share"},
    "sess_gc_volume_peak_timing": {"variant": "sess_gc_volume_peak_timing"},
    "sess_gc_vol_peak_x_direction": {"variant": "sess_gc_vol_peak_x_direction"},
    "sess_gc_am_range_share": {"variant": "sess_gc_am_range_share"},
    "sess_gc_close_location": {"variant": "sess_gc_close_location"},
    "sess_gc_am_efficiency": {"variant": "sess_gc_am_efficiency"},
    "sess_gc_close30_ret_share": {"variant": "sess_gc_close30_ret_share"},
    "sess_gc_amount_weighted_return": {"variant": "sess_gc_amount_weighted_return"},
    "sess_gc_morning_close_position": {"variant": "sess_gc_morning_close_position"},
    # ---- C candidates (declared direction fixed ex ante) ----------------
    "sess_c_lunch_ret": {"variant": "sess_c_lunch_ret"},
    "sess_c_pm_ret": {"variant": "sess_c_pm_ret"},
    "sess_c_am_ret": {"variant": "sess_c_am_ret"},
    "sess_c_am_signed_volume": {"variant": "sess_c_am_signed_volume"},
    "sess_c_pm_signed_volume": {"variant": "sess_c_pm_signed_volume"},
    "sess_c_close_location_pm": {"variant": "sess_c_close_location_pm"},
    "sess_c_tail30_fade": {"variant": "sess_c_tail30_fade"},
    "sess_c_tail30_high_proximity": {"variant": "sess_c_tail30_high_proximity"},
}


def _cache(ctx):
    box = ctx.__dict__.get("_r17_session_cache")
    if box is None:
        box = {}
        ctx.__dict__["_r17_session_cache"] = box
    return box


def _traded(ctx):
    box = _cache(ctx)
    if "traded" not in box:
        box["traded"] = np.asarray(ctx.traded(), dtype=bool)
    return box["traded"]


def _base(ctx):
    """Read and align every 5-minute field once; return a dict of arrays."""
    box = _cache(ctx)
    if "base" in box:
        return box["base"]

    def f(name):
        return np.asarray(ctx.intraday_field(name), dtype=np.float64)

    n = f("n_bars")
    o, cl, hi, lo = f("open5"), f("close5"), f("hi5"), f("lo5")
    vol, amt = f("vol"), f("amt")
    amo, amc, amhi, amlo = f("am_open5"), f("am_close5"), f("am_hi"), f("am_lo")
    amvol, amamt = f("am_vol"), f("am_amt")
    pmo, pmc, pmhi, pmlo = f("pm_open5"), f("pm_close5"), f("pm_hi"), f("pm_lo")
    pmvol, pmamt = f("pm_vol"), f("pm_amt")

    # high/low do not always bracket their own bar (0.075% of stock-days);
    # expand each segment with its own open/close before taking logs.
    hi_e = np.maximum(hi, np.fmax(o, cl))
    lo_e = np.minimum(lo, np.fmin(o, cl))
    amhi_e = np.maximum(amhi, np.fmax(amo, amc))
    amlo_e = np.minimum(amlo, np.fmin(amo, amc))
    pmhi_e = np.maximum(pmhi, np.fmax(pmo, pmc))
    pmlo_e = np.minimum(pmlo, np.fmin(pmo, pmc))

    ok = (n == 48) & np.asarray(ctx.universe, dtype=bool) & _traded(ctx)
    for arr in (o, cl, hi_e, lo_e, amt, vol, amo, amc, amhi_e, amlo_e, amvol,
                amamt, pmo, pmc, pmhi_e, pmlo_e, pmvol, pmamt):
        ok &= np.isfinite(arr)
    ok &= (o > 0) & (cl > 0) & (amo > 0) & (amc > 0) & (pmo > 0) & (pmc > 0)
    ok &= (amt > 0) & (amamt > 0) & (pmamt > 0) & (vol > 0) & (amvol > 0) & (pmvol > 0)

    with np.errstate(all="ignore"):
        r_am = np.log(amc / amo)
        r_pm = np.log(pmc / pmo)
        r_lunch = np.log(pmo / amc)
        day_ret = np.log(cl / o)
        rng_full = np.log(hi_e / lo_e)
        rng_am = np.log(amhi_e / amlo_e)
        rng_pm = np.log(pmhi_e / pmlo_e)
        abs_total = np.abs(r_am) + np.abs(r_lunch) + np.abs(r_pm)
        am_share_amt = amamt / amt
        pm_share_amt = pmamt / amt

    base = {
        "ok": ok,
        "r_am": r_am, "r_pm": r_pm, "r_lunch": r_lunch, "day_ret": day_ret,
        "rng_full": rng_full, "rng_am": rng_am, "rng_pm": rng_pm,
        "abs_total": abs_total,
        "am_share_amt": am_share_amt, "pm_share_amt": pm_share_amt,
        "o": o, "cl": cl, "hi_e": hi_e, "lo_e": lo_e,
        "amc": amc, "amo": amo, "amhi_e": amhi_e, "amlo_e": amlo_e,
        "pmc": pmc, "pmo": pmo, "pmhi_e": pmhi_e, "pmlo_e": pmlo_e,
        "amt": amt, "vol": vol,
    }
    box["base"] = base
    return base


def _mask(x, ok):
    return np.where(ok, x, np.nan)


def _close30(ctx):
    """Closing-30m fields aligned to the same-day base mask."""
    box = _cache(ctx)
    if "close30" not in box:
        g = lambda k: np.asarray(ctx.closing30_field(k), dtype=np.float64)
        grid = g("grid_ok")
        ad, at = g("amt_day"), g("amt_tail30")
        ot, ht, ct = g("open_tail30"), g("high_tail30"), g("close_tail30")
        b = _base(ctx)
        ok = (b["ok"] & (grid == 1) & np.isfinite(ad) & np.isfinite(at)
              & np.isfinite(ot) & np.isfinite(ht) & np.isfinite(ct)
              & (ad > 0) & (at > 0) & (ot > 0) & (ht > 0) & (ct > 0))
        # vendor tail high may sit below its own open/close; expand it.
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


def build(parameters):
    variant = parameters["variant"]
    if variant not in {v["variant"] for v in PARAMETERS.values()}:
        raise KeyError(variant)

    def fn(ctx):
        b = _base(ctx)
        ok = b["ok"]

        # ---------------- GC: where does the movement happen? ---------------
        if variant == "sess_gc_am_return_share":
            den = np.abs(b["r_am"]) + np.abs(b["r_pm"])
            out = np.where(den > 0, np.abs(b["r_am"]) / np.where(den > 0, den, np.nan), np.nan)
            return _mask(out, ok)

        if variant == "sess_gc_lunch_abs_share":
            den = b["abs_total"]
            out = np.where(den > 0, np.abs(b["r_lunch"]) / np.where(den > 0, den, np.nan), np.nan)
            return _mask(out, ok)

        if variant == "sess_gc_session_alignment":
            # positive = both sessions same direction (trend day), negative = reversal
            return _mask(1e4 * b["r_am"] * b["r_pm"], ok)

        if variant == "sess_gc_am_amount_share":
            return _mask(b["am_share_amt"], ok)

        # ---------------- GC: opening / closing liquidity share -------------
        if variant == "sess_gc_open5_amount_share":
            o5 = _open5(ctx)
            return _mask(o5["amt_open5"] / o5["amt_day"], o5["ok"])

        if variant == "sess_gc_tail30_amount_share":
            c3 = _close30(ctx)
            return _mask(c3["amt_tail30"] / c3["amt_day"], c3["ok"])

        # ---------------- GC: timing of the volume peak ---------------------
        if variant == "sess_gc_volume_peak_timing":
            return _mask(np.asarray(ctx.intraday_field("vol_peak_pos"), dtype=np.float64), ok)

        if variant == "sess_gc_vol_peak_x_direction":
            pos = np.asarray(ctx.intraday_field("vol_peak_pos"), dtype=np.float64)
            direction = np.sign(b["day_ret"])          # +1 up day, -1 down day, 0 flat
            return _mask(direction * (pos - 0.5), ok)

        # ---------------- GC: range allocation ------------------------------
        if variant == "sess_gc_am_range_share":
            out = np.where(b["rng_full"] > 0,
                           b["rng_am"] / np.where(b["rng_full"] > 0, b["rng_full"], np.nan),
                           np.nan)
            return _mask(out, ok)

        # ---------------- GC: close position within the range ---------------
        if variant == "sess_gc_close_location":
            den = b["hi_e"] - b["lo_e"]
            out = np.where(den > 0, (b["cl"] - b["lo_e"]) / np.where(den > 0, den, np.nan), np.nan)
            return _mask(np.clip(out, 0.0, 1.0), ok)

        if variant == "sess_gc_morning_close_position":
            den = b["amhi_e"] - b["amlo_e"]
            out = np.where(den > 0, (b["amc"] - b["amlo_e"]) / np.where(den > 0, den, np.nan), np.nan)
            return _mask(np.clip(out, 0.0, 1.0), ok)

        if variant == "sess_gc_am_efficiency":
            out = np.where(b["rng_am"] > 0,
                           b["r_am"] / np.where(b["rng_am"] > 0, b["rng_am"], np.nan),
                           np.nan)
            return _mask(np.clip(out, -1.0, 1.0), ok)

        # ---------------- GC: last-30m share of the day's movement ----------
        if variant == "sess_gc_close30_ret_share":
            c3 = _close30(ctx)
            with np.errstate(all="ignore"):
                tail = np.log(c3["close_tail30"] / c3["open_tail30"])
                rest = np.log(c3["open_tail30"] / b["o"])
                den = np.abs(tail) + np.abs(rest)
                out = np.where(den > 0, np.abs(tail) / np.where(den > 0, den, np.nan), np.nan)
            return _mask(out, c3["ok"])

        # ---------------- GC: amount-weighted continuous-session return -----
        if variant == "sess_gc_amount_weighted_return":
            with np.errstate(all="ignore"):
                out = (b["am_share_amt"] * b["r_am"] + b["pm_share_amt"] * b["r_pm"])
            return _mask(out, ok)

        # ---------------- C: lunch-break information (+ h1) -----------------
        if variant == "sess_c_lunch_ret":
            return _mask(b["r_lunch"], ok)

        # ---------------- C: afternoon session return (- h1) ----------------
        if variant == "sess_c_pm_ret":
            return _mask(b["r_pm"], ok)

        # ---------------- C: morning session return (+ h5) ------------------
        if variant == "sess_c_am_ret":
            return _mask(b["r_am"], ok)

        # ---------------- C: signed session turnover shares -----------------
        if variant == "sess_c_am_signed_volume":
            out = np.sign(b["r_am"]) * b["am_share_amt"]
            return _mask(out, ok)

        if variant == "sess_c_pm_signed_volume":
            out = np.sign(b["r_pm"]) * b["pm_share_amt"]
            return _mask(out, ok)

        # ---------------- C: close position inside the afternoon range ------
        if variant == "sess_c_close_location_pm":
            den = b["pmhi_e"] - b["pmlo_e"]
            out = np.where(den > 0, (b["pmc"] - b["pmlo_e"]) / np.where(den > 0, den, np.nan), np.nan)
            return _mask(np.clip(out, 0.0, 1.0), ok)

        # ---------------- C: closing-30m path --------------------------------
        if variant == "sess_c_tail30_fade":
            c3 = _close30(ctx)
            out = c3["close_tail30"] / c3["high_tail30"] - 1.0
            return _mask(np.minimum(out, 0.0), c3["ok"])

        if variant == "sess_c_tail30_high_proximity":
            c3 = _close30(ctx)
            den = np.maximum(b["hi_e"], c3["high_tail30"])
            out = c3["high_tail30"] / den - 1.0
            return _mask(np.minimum(out, 0.0), c3["ok"])

        raise KeyError(variant)

    return fn
