"""R17 seat chip_event — batch 3: dragon-tiger / top-list participation + suspension.

Sources and semantics
---------------------
* `stock_dragon_tiger` (2013+, sparse, multiple seat rows per stock-day).
  Raw rows are deduplicated by (trade_date, stock_code, org_name, round(|net_buy_amount|,2)):
  the table contains exact duplicate rows and buy/sell-column-swapped duplicates with the
  same |net| in several years; this rule was validated against `stock_top_list` (per
  stock-day sum of net_buy_amount reproduces top_list net_amount for ~92% of stock-days
  within 1%).  Seat names are raw; institution = org_name contains '机构专用', northbound
  = contains '沪股通专用'/'深股通专用'.  No name standardisation is claimed.
* `stock_top_list` (2010+, sparse, multiple reason rows per stock-day).  Per stock-day
  dedup rule: keep the row with the largest `amount_rate` (participation intensity) for
  amount_rate / net_rate / l_buy / l_sell; reason-group flags are computed over **all**
  rows of that stock-day (`any`) so no reason is lost.
* Delay: `stock_dragon_tiger` / `stock_top_list` are known to arrive one trading day late
  (spec.py comment; frequency.yaml says 0 -> conflict resolved conservatively).  Every
  dt/tl factor is passed through `ctx.lag_grid(out, 1)` and candidates declare
  `lagged_ok` for those tables.  `stock_suspension` is same-day (delay 0).
* Suspension: `suspend_date` rows are per suspended trading day (event dates).  An episode
  = consecutive suspended trading days.  Resumption return uses `ret_clean(1)` on the
  first traded day after the episode; the last pre-suspension close is ffilled, so this is
  the actual resumption-day return.  Frozen universe excludes delisted names; only
  survivors' suspension histories are observable.
* All outputs masked with `ctx.traded() & ctx.universe`; event-conditional means require
  >=1 finite event in the window, else NaN (never 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea.mathx import nan_fill_ffill

WARMUP_DAYS = 400

_GC = (
    "dt_hhi_absnet_60",
    "dt_pos_seat_share_60",
    "dt_seat_dispersion_60",
    "dt_inst_presence_60",
    "dt_north_presence_60",
    "dt_repeat_seat_60",
    "dt_first_appearance_60",
    "tl_amount_rate_60",
    "tl_amount_rate_disp_60",
    "tl_up_reason_share_60",
    "tl_down_reason_share_60",
    "tl_turnover_reason_share_60",
    "susp_days_250",
    "susp_episodes_250",
    "susp_max_len_250",
    "susp_avg_len_250",
    "susp_recency_250",
    "post_susp_first_ret_250",
)
_C = (
    "dt_inst_breadth_60",
    "dt_net_persistence_60",
    "dt_net_innovation",
    "dt_turnover_excess_60",
    "tl_net_rate_persistence_60",
    "tl_net_rate_innovation",
    "tl_l_buy_share_60",
)

PARAMETERS = {name: {"variant": name} for name in (*_GC, *_C)}

_TABLE_CACHE: dict[tuple, dict] = {}


def _years(ctx):
    return int(str(ctx.panel.dates[0])[:4]), int(str(ctx.panel.dates[-1])[:4])


def _tables(ctx) -> dict:
    key = (id(ctx.up), int(ctx.panel.dates[0]), int(ctx.panel.dates[-1]), ctx.panel.C)
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    years = _years(ctx)
    out: dict = {}

    # ---------------- dragon-tiger seat table -> per stock-day metrics ------
    dt = ctx.dataset("stock_dragon_tiger",
                     columns=["trade_date", "stock_code", "org_name", "buy_amount",
                              "sell_amount", "net_buy_amount"], years=years)
    if dt is None or dt.empty:
        out["dt"] = None
    else:
        dt = dt.copy()
        dt["stock_code"] = dt["stock_code"].astype(str)
        dt["org_name"] = dt["org_name"].astype(str)
        dt["absnet"] = pd.to_numeric(dt["net_buy_amount"], errors="coerce").abs().round(2)
        dt["net"] = pd.to_numeric(dt["net_buy_amount"], errors="coerce")
        dt["gross"] = (pd.to_numeric(dt["buy_amount"], errors="coerce")
                       + pd.to_numeric(dt["sell_amount"], errors="coerce"))
        dt = dt.dropna(subset=["net", "absnet"])
        dt = dt.drop_duplicates(["trade_date", "stock_code", "org_name", "absnet"])
        dt["is_inst"] = dt["org_name"].str.contains("机构专用", regex=False)
        dt["is_north"] = dt["org_name"].str.contains("沪股通专用|深股通专用", regex=True)
        abs_sum = dt.groupby(["trade_date", "stock_code"], sort=False)["absnet"].transform("sum")
        dt["abs_share"] = np.where(abs_sum > 0, dt["absnet"] / abs_sum, np.nan)
        dt["signed_share"] = np.where(abs_sum > 0, dt["net"] / abs_sum, np.nan)
        dt["share_sq"] = dt["abs_share"] ** 2
        dt["pos_i"] = (dt["net"] > 0).astype(np.float64)
        dt["inst_pos_i"] = ((dt["net"] > 0) & dt["is_inst"]).astype(np.float64)
        dt["td"] = np.searchsorted(ctx.panel.dates,
                                   ctx.date_col(dt["trade_date"]).astype(np.int64))
        gb = dt.groupby(["trade_date", "stock_code"], sort=False)
        # repeat-seat: same raw org appeared on this stock within prior 20 trading days
        seat = dt[["trade_date", "stock_code", "org_name", "td"]].sort_values(
            ["stock_code", "org_name", "td"])
        seat["prev_td"] = seat.groupby(["stock_code", "org_name"], sort=False)["td"].shift(1)
        gap = seat["td"] - seat["prev_td"]
        seat["repeat"] = (gap > 0) & (gap <= 20)
        rep = seat.groupby(["trade_date", "stock_code"], sort=False)["repeat"].mean()
        per = gb.agg(net=("net", "sum"), gross=("gross", "sum"), seats=("net", "size"),
                     inst=("is_inst", "any"), north=("is_north", "any"),
                     hhi=("share_sq", "sum"), pos=("pos_i", "sum"),
                     inst_pos=("inst_pos_i", "sum"), inst_n=("is_inst", "sum"),
                     td=("td", "first"))
        per["pos_share"] = per["pos"] / per["seats"]
        per["disp"] = gb["signed_share"].std(ddof=0)
        per["inst_breadth"] = np.where(per["inst_n"] > 0,
                                       per["inst_pos"] / per["inst_n"], np.nan)
        per = per.join(rep.rename("repeat_share"))
        # first appearance of the stock in the prior 60 trading days
        per2 = per.reset_index().sort_values(["stock_code", "td"])
        per2["prev_td"] = per2.groupby("stock_code", sort=False)["td"].shift(1)
        per2["first"] = per2["prev_td"].isna() | ((per2["td"] - per2["prev_td"]) > 60)
        per = per2.set_index(["trade_date", "stock_code"])
        out["dt"] = {c: per[c].to_numpy(np.float64) for c in
                     ("net", "gross", "hhi", "pos_share", "disp", "inst_breadth",
                      "repeat_share", "first")}
        out["dt"].update({
            "codes": per.index.get_level_values(1).to_numpy(),
            "days": ctx.date_col(pd.Series(per.index.get_level_values(0))),
            "is_inst": per["inst"].to_numpy(bool),
            "is_north": per["north"].to_numpy(bool),
        })

    # ---------------- top-list: dedup daily row + reason flags --------------
    tl = ctx.dataset("stock_top_list",
                     columns=["trade_date", "stock_code", "amount_rate", "net_rate",
                              "l_buy", "l_sell", "reason"], years=years)
    if tl is None or tl.empty:
        out["tl"] = None
    else:
        tl = tl.copy()
        tl["stock_code"] = tl["stock_code"].astype(str)
        r = tl["reason"].astype(str)
        tl["r_up"] = r.str.contains("涨幅偏离", regex=False)
        tl["r_down"] = r.str.contains("跌幅偏离", regex=False)
        tl["r_turn"] = r.str.contains("换手率", regex=False)
        tl["r_amp"] = r.str.contains("振幅", regex=False)
        flags = tl.groupby(["trade_date", "stock_code"], sort=False)[
            ["r_up", "r_down", "r_turn", "r_amp"]].any()
        tl["amount_rate"] = pd.to_numeric(tl["amount_rate"], errors="coerce")
        tl["net_rate"] = pd.to_numeric(tl["net_rate"], errors="coerce")
        tl["l_buy"] = pd.to_numeric(tl["l_buy"], errors="coerce")
        tl["l_sell"] = pd.to_numeric(tl["l_sell"], errors="coerce")
        main = (tl.sort_values(["trade_date", "stock_code", "amount_rate"])
                .drop_duplicates(["trade_date", "stock_code"], keep="last")
                .drop(columns=["r_up", "r_down", "r_turn", "r_amp"])
                .set_index(["trade_date", "stock_code"]))
        main = main.join(flags)
        out["tl"] = {
            "codes": main.index.get_level_values(1).to_numpy(),
            "days": ctx.date_col(pd.Series(main.index.get_level_values(0))),
            "amount_rate": main["amount_rate"].to_numpy(np.float64),
            "net_rate": main["net_rate"].to_numpy(np.float64),
            "l_buy": main["l_buy"].to_numpy(np.float64),
            "l_sell": main["l_sell"].to_numpy(np.float64),
            "r_up": main["r_up"].to_numpy(bool), "r_down": main["r_down"].to_numpy(bool),
            "r_turn": main["r_turn"].to_numpy(bool), "r_amp": main["r_amp"].to_numpy(bool),
        }

    # ---------------- suspension event dates ------------------------------
    su = ctx.dataset("stock_suspension",
                     columns=["stock_code", "suspend_date"], years=years)
    if su is None or su.empty:
        out["susp"] = None
    else:
        su = su.copy()
        su["stock_code"] = su["stock_code"].astype(str)
        out["susp"] = {"codes": su["stock_code"].to_numpy(),
                       "days": ctx.date_col(su["suspend_date"])}
    _TABLE_CACHE[key] = out
    if len(_TABLE_CACHE) > 2:
        for old in list(_TABLE_CACHE)[:-1]:
            _TABLE_CACHE.pop(old, None)
    return out


# --------------------------------------------------------------------------- #
def _grid(ctx, t, mask=None, values=None):
    if t is None:
        return ctx.panel.empty()
    c, d = t["codes"], t["days"]
    if mask is not None:
        m = np.asarray(mask, bool)
        c, d = c[m], d[m]
        if values is not None:
            values = np.asarray(values, np.float64)[m]
    if values is None:
        return ctx.event_grid(c, d)
    v = np.asarray(values, np.float64)
    ok = np.isfinite(v)
    return ctx.event_grid(c[ok], d[ok], v[ok])


def _evmean(ctx, t, mask, values, n):
    v = np.asarray(values, np.float64)
    m = np.asarray(mask, bool) & np.isfinite(v)
    num = ctx.roll_sum(_grid(ctx, t, m, v), n, min_count=1)
    den = ctx.roll_sum(_grid(ctx, t, m), n, min_count=1)
    return ctx.safe_div(num, den, min_abs_den=1e-12), den


def _evstd(ctx, t, mask, values, n):
    v = np.asarray(values, np.float64)
    m = np.asarray(mask, bool) & np.isfinite(v)
    s1 = ctx.roll_sum(_grid(ctx, t, m, v), n, min_count=1)
    s2 = ctx.roll_sum(_grid(ctx, t, m, v * v), n, min_count=1)
    den = ctx.roll_sum(_grid(ctx, t, m), n, min_count=1)
    mean = ctx.safe_div(s1, den, min_abs_den=1e-12)
    var = np.clip(ctx.safe_div(s2, den, min_abs_den=1e-12) - mean * mean, 0.0, None)
    out = np.sqrt(var)
    out[den < 3] = np.nan
    return out, den


def _share(ctx, t, mask, n):
    num = ctx.roll_sum(_grid(ctx, t, mask), n, min_count=1)
    den = ctx.roll_sum(_grid(ctx, t, None), n, min_count=1)
    return ctx.safe_div(num, den, min_abs_den=1e-12), den


def _at_events(ctx, t, arr):
    idx = ctx.code_index(t["codes"])
    rows = np.searchsorted(ctx.panel.dates, t["days"])
    clipped = np.clip(rows, 0, ctx.panel.T - 1)
    on = (rows < ctx.panel.T) & (ctx.panel.dates[clipped] == t["days"])
    keep = (idx >= 0) & on
    out = np.full(len(t["days"]), np.nan)
    out[keep] = arr[clipped[keep], idx[keep]]
    return out


def _mask(ctx, out):
    return np.where(ctx.traded() & ctx.universe, out, np.nan)


def _lag(ctx, out):
    """Conservative one-trading-day availability lag for dragon/top-list."""
    return ctx.lag_grid(out, 1)


def _run_len(x):
    t, c = x.shape
    pos = np.arange(t, dtype=np.float64)[:, None] * np.ones((1, c), dtype=np.float64)
    last0 = np.maximum.accumulate(np.where(x > 0, -1.0, pos), axis=0)
    return np.where(x > 0, pos - last0, 0.0)


def build(parameters):
    variant = parameters["variant"]
    if variant not in PARAMETERS:
        raise KeyError(variant)

    def fn(ctx):
        tab = _tables(ctx)
        dt, tl, su = tab["dt"], tab["tl"], tab["susp"]

        # ================= dragon-tiger =====================================
        if variant.startswith("dt_"):
            if dt is None:
                return _mask(ctx, ctx.panel.empty())
            allm = np.ones(len(dt["days"]), bool)
            if variant == "dt_hhi_absnet_60":
                out, _ = _evmean(ctx, dt, allm, dt["hhi"], 60)
            elif variant == "dt_pos_seat_share_60":
                out, _ = _evmean(ctx, dt, allm, dt["pos_share"], 60)
            elif variant == "dt_seat_dispersion_60":
                out, _ = _evmean(ctx, dt, allm, dt["disp"], 60)
            elif variant == "dt_inst_presence_60":
                out, _ = _share(ctx, dt, dt["is_inst"], 60)
            elif variant == "dt_north_presence_60":
                out, _ = _share(ctx, dt, dt["is_north"], 60)
            elif variant == "dt_repeat_seat_60":
                out, _ = _evmean(ctx, dt, allm, dt["repeat_share"], 60)
            elif variant == "dt_first_appearance_60":
                out, _ = _evmean(ctx, dt, allm, dt["first"], 60)
            elif variant == "dt_inst_breadth_60":
                m = np.isfinite(dt["inst_breadth"])
                out, _ = _evmean(ctx, dt, m, dt["inst_breadth"], 60)
            elif variant == "dt_net_persistence_60":
                out, _ = _share(ctx, dt, dt["net"] > 0, 60)
            elif variant == "dt_net_innovation":
                ratio = np.where(dt["gross"] > 0, dt["net"] / dt["gross"], np.nan)
                idx = ctx.code_index(dt["codes"])
                keep = idx >= 0
                latest = nan_fill_ffill(ctx.panel.place(
                    idx[keep].astype(np.int64), dt["days"][keep], ratio[keep]))
                hist, _ = _evmean(ctx, dt, allm, ratio, 60)
                out = latest - hist
            elif variant == "dt_turnover_excess_60":
                turn = ctx.px("turnover")
                tev = _at_events(ctx, dt, turn)
                m, _ = _evmean(ctx, dt, allm, tev, 60)
                out = m - ctx.roll_mean(turn, 60, min_count=30)
            else:
                raise KeyError(variant)
            return _mask(ctx, _lag(ctx, out))

        # ================= top-list ========================================
        if variant.startswith("tl_"):
            if tl is None:
                return _mask(ctx, ctx.panel.empty())
            allm = np.ones(len(tl["days"]), bool)
            if variant == "tl_amount_rate_60":
                out, _ = _evmean(ctx, tl, allm, tl["amount_rate"], 60)
            elif variant == "tl_amount_rate_disp_60":
                out, _ = _evstd(ctx, tl, allm, tl["amount_rate"], 60)
            elif variant == "tl_net_rate_persistence_60":
                out, _ = _share(ctx, tl, tl["net_rate"] > 0, 60)
            elif variant == "tl_net_rate_innovation":
                idx = ctx.code_index(tl["codes"])
                keep = idx >= 0
                latest = nan_fill_ffill(ctx.panel.place(
                    idx[keep].astype(np.int64), tl["days"][keep], tl["net_rate"][keep]))
                hist, _ = _evmean(ctx, tl, allm, tl["net_rate"], 60)
                out = latest - hist
            elif variant == "tl_up_reason_share_60":
                out, _ = _share(ctx, tl, tl["r_up"], 60)
            elif variant == "tl_down_reason_share_60":
                out, _ = _share(ctx, tl, tl["r_down"], 60)
            elif variant == "tl_turnover_reason_share_60":
                out, _ = _share(ctx, tl, tl["r_turn"], 60)
            elif variant == "tl_l_buy_share_60":
                with np.errstate(invalid="ignore", divide="ignore"):
                    share = np.where((tl["l_buy"] + tl["l_sell"]) > 0,
                                     tl["l_buy"] / (tl["l_buy"] + tl["l_sell"]), np.nan)
                out, _ = _evmean(ctx, tl, allm, share, 60)
            else:
                raise KeyError(variant)
            return _mask(ctx, _lag(ctx, out))

        # ================= suspension ======================================
        if su is None:
            return _mask(ctx, ctx.panel.empty())
        S = _grid(ctx, su)
        if variant in ("susp_days_250", "susp_episodes_250"):
            if variant == "susp_days_250":
                out = ctx.roll_sum(S, 250, min_count=1)
            else:
                start = S * (ctx.shift(S, 1) <= 0)
                out = ctx.roll_sum(start, 250, min_count=1)
            return _mask(ctx, out)

        rl = _run_len(S)
        end_prev = (ctx.shift(S, 1) > 0) & (S <= 0)        # episode ended yesterday
        ep_len = np.where(end_prev, ctx.shift(rl, 1), np.nan)
        tix = np.arange(ctx.panel.T, dtype=np.float64)[:, None] * np.ones(
            (1, ctx.panel.C), dtype=np.float64)
        full = tix >= 249.0                                 # complete 250-row window
        if variant == "susp_max_len_250":
            out = ctx.roll_max(ep_len, 250, min_count=1)
            out = np.where(np.isfinite(out), out, np.where(full, 0.0, np.nan))
        elif variant == "susp_avg_len_250":
            num = ctx.roll_sum(np.where(np.isfinite(ep_len), ep_len, 0.0), 250, min_count=1)
            den = ctx.roll_sum(np.isfinite(ep_len).astype(np.float64), 250, min_count=1)
            out = ctx.safe_div(num, den, min_abs_den=1e-12)
            out = np.where(np.isfinite(out), out, np.where(full, 0.0, np.nan))
        elif variant == "susp_recency_250":
            last = nan_fill_ffill(np.where(S > 0, tix, np.nan))
            rec = tix - last
            has = ctx.roll_sum(S, 250, min_count=1) > 0
            # censored at 250: no suspension in the window -> 250 (lower bound)
            out = np.where(has, np.minimum(rec, 250.0), np.where(full, 250.0, np.nan))
        elif variant == "post_susp_first_ret_250":
            r1 = ctx.ret_clean(1)
            valid = end_prev & np.isfinite(r1)
            num = ctx.roll_sum(np.where(valid, r1, 0.0), 250, min_count=1)
            den = ctx.roll_sum(valid.astype(np.float64), 250, min_count=1)
            out = ctx.safe_div(num, den, min_abs_den=0.5)
        else:
            raise KeyError(variant)
        return _mask(ctx, out)

    return fn
