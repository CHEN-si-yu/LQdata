"""R17 seat chip_event — batch 4 (design): cross-domain chip state x limit event.

Six variants that condition a chip-distribution state on a public event, or
measure how the chip state changes around informed flow:
  * post-limit drift vs winner rate (winner_rate at the latest limit-up, 5d carry)
  * post-limit overhead supply in days of turnover
  * dragon-tiger appearance days: chip Gini vs the stock's own 60d mean
  * limit-up days: chip entropy vs the stock's own 60d mean
  * post-suspension resumption: chip median-cost gap (state)
  * limit-down days: winner rate vs the stock's own 60d mean

These are not products of two factors; each is an event-conditional state or a
state-relative innovation, so they are not the "multiply known factors" pattern
the design guide rejects.  Sources: stock_cyq_chips via ctx.chip, stock_cyq_perf
via exact daily placement, stock_limit_up / stock_limit_list / stock_suspension
event dates, stock_daily.  Dragon/top-list lag 1 day; limit/suspension same-day.
All outputs masked by traded & universe; event-conditional means require >=1
event in the window, else NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea.mathx import nan_fill_ffill

WARMUP_DAYS = 400

_C = ("post_limit_winner_rate_5", "post_limit_overhead_days_5", "limit_down_winner_gap_60")
_GC = ("dt_appearance_gini_gap_60", "limit_up_entropy_gap_60", "post_susp_chip_gap_250")
PARAMETERS = {n: {"variant": n} for n in (*_C, *_GC)}

_CACHE: dict[tuple, dict] = {}
_PERF_COLUMNS = ("winner_rate",)


def _years(ctx):
    return int(str(ctx.panel.dates[0])[:4]), int(str(ctx.panel.dates[-1])[:4])


def _tables(ctx):
    key = (id(ctx.up), int(ctx.panel.dates[0]), int(ctx.panel.dates[-1]), ctx.panel.C)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    years = _years(ctx)
    out = {}
    lu = ctx.dataset("stock_limit_up",
                     columns=["trade_date", "stock_code", "is_limit_up"], years=years)
    if lu is None or lu.empty:
        out["lu"] = None
    else:
        lu = lu[pd.to_numeric(lu["is_limit_up"], errors="coerce").fillna(0) == 1]
        out["lu"] = {"codes": lu["stock_code"].astype(str).to_numpy(),
                     "days": ctx.date_col(lu["trade_date"])}
    ll = ctx.dataset("stock_limit_list",
                     columns=["trade_date", "stock_code", "limit"], years=years)
    if ll is None or ll.empty:
        out["ll"] = None
    else:
        lim = ll["limit"].astype(str).to_numpy()
        out["ll"] = {"codes": ll["stock_code"].astype(str).to_numpy(),
                     "days": ctx.date_col(ll["trade_date"]),
                     "u": lim == "U", "d": lim == "D"}
    su = ctx.dataset("stock_suspension", columns=["stock_code", "suspend_date"], years=years)
    out["susp"] = None if su is None or su.empty else {
        "codes": su["stock_code"].astype(str).to_numpy(),
        "days": ctx.date_col(su["suspend_date"])}
    dt = ctx.dataset("stock_dragon_tiger",
                     columns=["trade_date", "stock_code", "org_name", "buy_amount",
                              "sell_amount", "net_buy_amount"], years=years)
    if dt is None or dt.empty:
        out["dt"] = None
    else:
        dt = dt.copy()
        dt["stock_code"] = dt["stock_code"].astype(str)
        dt["absnet"] = pd.to_numeric(dt["net_buy_amount"], errors="coerce").abs().round(2)
        dt = dt.dropna(subset=["absnet"]).drop_duplicates(
            ["trade_date", "stock_code", "org_name", "absnet"])
        per = dt.drop_duplicates(["trade_date", "stock_code"])
        out["dt"] = {"codes": per["stock_code"].to_numpy(),
                     "days": ctx.date_col(per["trade_date"])}
    perf = ctx.dataset("stock_cyq_perf",
                       columns=["trade_date", "stock_code", "winner_rate"], years=years)
    if perf is None or perf.empty:
        out["perf"] = None
    else:
        idx = ctx.code_index(perf["stock_code"].astype(str).to_numpy())
        keep = idx >= 0
        out["perf"] = {"grid": ctx.panel.place(
            idx[keep].astype(np.int64), ctx.date_col(perf["trade_date"])[keep],
            pd.to_numeric(perf["winner_rate"], errors="coerce").to_numpy(np.float64)[keep])}
    _CACHE[key] = out
    if len(_CACHE) > 2:
        for old in list(_CACHE)[:-1]:
            _CACHE.pop(old, None)
    return out


def _grid(ctx, t, mask=None):
    if t is None:
        return ctx.panel.empty()
    c, d = t["codes"], t["days"]
    if mask is not None:
        m = np.asarray(mask, bool)
        c, d = c[m], d[m]
    return ctx.event_grid(c, d)


def _at_events(ctx, t, arr):
    idx = ctx.code_index(t["codes"])
    rows = np.searchsorted(ctx.panel.dates, t["days"])
    clipped = np.clip(rows, 0, ctx.panel.T - 1)
    on = (rows < ctx.panel.T) & (ctx.panel.dates[clipped] == t["days"])
    keep = (idx >= 0) & on
    out = np.full(len(t["days"]), np.nan)
    out[keep] = arr[clipped[keep], idx[keep]]
    return out


def _evmean2(ctx, t, vals, mask, n):
    v = np.asarray(vals, np.float64)
    m = np.asarray(mask, bool) & np.isfinite(v)
    c, d = t["codes"][m], t["days"][m]
    num = ctx.roll_sum(ctx.event_grid(c, d, v[m]), n, min_count=1)
    den = ctx.roll_sum(ctx.event_grid(c, d), n, min_count=1)
    return ctx.safe_div(num, den, min_abs_den=1e-12), den


def _last_event_state(ctx, t, mask, vals, recent_n):
    c, d = t["codes"][mask], t["days"][mask]
    v = np.asarray(vals, np.float64)[mask]
    ok = np.isfinite(v)
    idx = ctx.code_index(c[ok])
    keep = idx >= 0
    latest = nan_fill_ffill(ctx.panel.place(idx[keep].astype(np.int64), d[ok][keep], v[ok][keep]))
    recent = ctx.roll_sum(ctx.event_grid(t["codes"][mask], t["days"][mask]), recent_n,
                          min_count=1) > 0
    return np.where(recent, latest, np.nan)


def _mask(ctx, out):
    return np.where(ctx.traded() & ctx.universe, out, np.nan)


def build(parameters):
    variant = parameters["variant"]
    if variant not in PARAMETERS:
        raise KeyError(variant)

    def fn(ctx):
        tab = _tables(ctx)
        lu, ll, su, dt, perf = tab["lu"], tab["ll"], tab["susp"], tab["dt"], tab["perf"]
        if variant in _C and variant.startswith("post_limit_"):
            if lu is None or perf is None:
                return _mask(ctx, ctx.panel.empty())
            wr = _at_events(ctx, lu, perf["grid"])
            if variant == "post_limit_winner_rate_5":
                out = _last_event_state(ctx, lu, np.ones(len(lu["days"]), bool), wr, 5)
            else:
                above = ctx.chip("above_close")
                turn = ctx.px("turnover")
                to20 = ctx.roll_mean(turn, 20, min_count=10) / 100.0
                days = ctx.safe_div(above, to20, min_abs_den=0.001)
                dvals = _at_events(ctx, lu, days)
                out = _last_event_state(ctx, lu, np.ones(len(lu["days"]), bool), dvals, 5)
            return _mask(ctx, out)

        if variant == "limit_down_winner_gap_60":
            if ll is None:
                return _mask(ctx, ctx.panel.empty())
            wr = _at_events(ctx, ll, perf["grid"])
            m = ll["d"]
            ev, _ = _evmean2(ctx, ll, wr, m, 60)
            base = ctx.roll_mean(perf["grid"], 60, min_count=30)
            return _mask(ctx, ev - base)

        if variant == "limit_up_entropy_gap_60":
            if lu is None:
                return _mask(ctx, ctx.panel.empty())
            ent = ctx.chip("entropy")
            ev, _ = _evmean2(ctx, lu, _at_events(ctx, lu, ent),
                             np.ones(len(lu["days"]), bool), 60)
            base = ctx.roll_mean(ent, 60, min_count=30)
            return _mask(ctx, ev - base)

        if variant == "dt_appearance_gini_gap_60":
            if dt is None:
                return _mask(ctx, ctx.panel.empty())
            gini = ctx.chip("gini")
            ev, _ = _evmean2(ctx, dt, _at_events(ctx, dt, gini),
                             np.ones(len(dt["days"]), bool), 60)
            base = ctx.roll_mean(gini, 60, min_count=30)
            out = ev - base
            return _mask(ctx, ctx.lag_grid(out, 1))

        if variant == "post_susp_chip_gap_250":
            if su is None:
                return _mask(ctx, ctx.panel.empty())
            S = _grid(ctx, su)
            end_prev = (ctx.shift(S, 1) > 0) & (S <= 0)
            gap = ctx.safe_log(ctx.safe_div(ctx.chip("p50"), ctx.px("close"), min_abs_den=1e-8))
            vals = np.where(end_prev, gap, np.nan)
            num = ctx.roll_sum(np.where(np.isfinite(vals), vals, 0.0), 250, min_count=1)
            den = ctx.roll_sum(np.isfinite(vals).astype(np.float64), 250, min_count=1)
            return _mask(ctx, ctx.safe_div(num, den, min_abs_den=1e-9))

        raise KeyError(variant)

    return fn
