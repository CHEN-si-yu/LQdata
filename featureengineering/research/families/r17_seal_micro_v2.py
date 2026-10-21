"""R17 seat chip_event — batch 2: limit-up / limit-list seal microstructure.

Design constraint (important)
-----------------------------
The removed-object index already contains a `field_events` family of 60-day
*event-mean* levels on almost every raw seal field (seal_float_strength_20,
seal_reopen_pressure_20, seal_turnover_strength_20, efx_limit_first,
efx_limit_streak, efx_limit_turnover, efx_lu_seal_amount, efx_seal_float, ...),
plus `tradability` limit_open_pressure_20 / limit_seal_to_float_20 /
limit_streak_max_60.  Those were plain `mean_60(raw field | event)` variants.
This batch deliberately does NOT re-add them.  It builds **composition shares,
term structure (recent vs history), conditional contrasts, transition/repair
probabilities, post-event drift and relative-to-own-turnover states**, none of
which appear in the removed list.

Data semantics
--------------
* `stock_limit_up` (2015+, one row per stock-day, `is_limit_up==1`): sealed_*
  fields ~99.9% complete from 2018 (89% missing in 2015 -> start 2018).  No row
  on a stock-day means "no sealed limit-up that day" = 0 within the covered
  years.  `open_count` counts intraday reopenings that were resealed by close;
  `consecutive_days` is the board height at close.
* `stock_limit_list` (2020+, one row per stock-day, limit in U/Z/D): event
  table; no row = no touch.  Starts 2020-01-02, so 60-day-window candidates
  start 2020-04-01 to avoid windows contaminated by pre-coverage zeros.
* `stock_limit_up` / `stock_limit_list` / `stock_suspension` are same-day
  available (frequency.yaml delay 0, confirmed in factors/event3.py).  This
  batch does not use dragon/top-list.
* Timestamps: `first_limit_time` / `final_limit_time` / `first_time` /
  `last_time` appear as `HH:MM:SS`, `HHMMSS`, or full `YYYY-MM-DD HH:MM:SS`
  depending on the row; parsed robustly and kept only in [09:15, 15:05],
  normalised to (minutes-09:30)/240 in [0,1].
* Event-conditional means use the count of **finite-valued events** as the
  denominator; zero events in the window -> NaN (unknown), never 0.  Shares use
  all events with a defined category as denominator.  No forward fill of
  events across the source-start boundary; spec.start truncates output.

All outputs are masked with `ctx.traded() & ctx.universe`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea.mathx import nan_fill_ffill

WARMUP_DAYS = 120

_GC = (
    "t_shape_share_60",
    "streak_share_60",
    "board_height_std_60",
    "strength_disp_60",
    "first_time_disp_60",
    "first_board_vs_streak",
    "next_ret_Z_60",
    "next_ret_D_60",
    "down_up_turnover_asym_60",
)
_C = (
    "oneword_share_60",
    "volume_type_share_60",
    "strength_term_10_60",
    "open_term_10_60",
    "latest_vs_history",
    "broken_repair_rate_60",
    "repair_speed_60",
    "limit_down_recovery_rate_60",
    "limit_down_to_up_rate_60",
    "down_limit_share_60",
    "next_ret_U_60",
    "queue_amount_ratio_60",
    "turnover_excess_60",
    "up_stat_gap_60",
)

PARAMETERS = {name: {"variant": name} for name in (*_GC, *_C)}

_LU_COLUMNS = ("first_limit_time", "final_limit_time", "open_count", "consecutive_days",
               "limit_type", "sealed_flow_ratio", "sealed_turnover_ratio", "is_limit_up")
_LL_COLUMNS = ("limit", "fd_amount", "amount", "turnover_ratio", "open_times",
               "limit_times", "up_stat", "first_time", "last_time")
_TABLE_CACHE: dict[tuple, dict] = {}


def _years(ctx) -> tuple[int, int]:
    return int(str(ctx.panel.dates[0])[:4]), int(str(ctx.panel.dates[-1])[:4])


def _minutes(value) -> float:
    """Robust limit timestamp -> minutes since 00:00; NaN if unparseable."""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return np.nan
    if " " in s:
        s = s.split()[-1]
    s = s.replace(":", "").zfill(6)
    if len(s) != 6 or not s.isdigit():
        return np.nan
    h, m, sec = int(s[:2]), int(s[2:4]), int(s[4:6])
    if h > 23 or m > 59 or sec > 59:
        return np.nan
    return h * 60.0 + m + sec / 60.0


def _norm_time(series) -> np.ndarray:
    t = series.map(_minutes).to_numpy(np.float64)
    out = (np.clip(t, 570.0, 900.0) - 570.0) / 240.0
    out[(t < 555.0) | (t > 905.0)] = np.nan
    return out


def _tables(ctx) -> dict:
    key = (id(ctx.up), int(ctx.panel.dates[0]), int(ctx.panel.dates[-1]), ctx.panel.C)
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    years = _years(ctx)
    lu = ctx.dataset("stock_limit_up",
                     columns=["trade_date", "stock_code", *_LU_COLUMNS], years=years)
    ll = ctx.dataset("stock_limit_list",
                     columns=["trade_date", "stock_code", *_LL_COLUMNS], years=years)
    out: dict = {}
    if lu is not None and not lu.empty:
        lu = lu[pd.to_numeric(lu["is_limit_up"], errors="coerce").fillna(0) == 1].copy()
        out["lu"] = {
            "codes": lu["stock_code"].astype(str).to_numpy(),
            "days": ctx.date_col(lu["trade_date"]),
            "first": _norm_time(lu["first_limit_time"]),
            "final": _norm_time(lu["final_limit_time"]),
            "open": pd.to_numeric(lu["open_count"], errors="coerce").to_numpy(np.float64),
            "cd": pd.to_numeric(lu["consecutive_days"], errors="coerce").to_numpy(np.float64),
            "type": lu["limit_type"].astype(str).to_numpy(),
            "flow": pd.to_numeric(lu["sealed_flow_ratio"], errors="coerce").to_numpy(np.float64),
            "stop": pd.to_numeric(lu["sealed_turnover_ratio"], errors="coerce").to_numpy(np.float64),
        }
    else:
        out["lu"] = None
    if ll is not None and not ll.empty:
        lim = ll["limit"].astype(str).to_numpy()
        out["ll"] = {
            "codes": ll["stock_code"].astype(str).to_numpy(),
            "days": ctx.date_col(ll["trade_date"]),
            "is_u": lim == "U",
            "is_z": lim == "Z",
            "is_d": lim == "D",
            "fd": pd.to_numeric(ll["fd_amount"], errors="coerce").to_numpy(np.float64),
            "amount": pd.to_numeric(ll["amount"], errors="coerce").to_numpy(np.float64),
            "turn": pd.to_numeric(ll["turnover_ratio"], errors="coerce").to_numpy(np.float64),
            "up_stat": ll["up_stat"].astype(str).to_numpy(),
        }
    else:
        out["ll"] = None
    _TABLE_CACHE[key] = out
    if len(_TABLE_CACHE) > 2:
        for old in list(_TABLE_CACHE)[:-1]:
            _TABLE_CACHE.pop(old, None)
    return out


# --------------------------------------------------------------------------- #
# event-grid helpers (sparse -> (T,C); no-event within coverage = 0)
# --------------------------------------------------------------------------- #
def _grid(ctx, t: dict, mask=None, values=None) -> np.ndarray:
    if t is None:
        return ctx.panel.empty()
    c, d = t["codes"], t["days"]
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        c, d = c[m], d[m]
        if values is not None:
            values = np.asarray(values, dtype=np.float64)[m]
    if values is None:
        return ctx.event_grid(c, d)
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    return ctx.event_grid(c[ok], d[ok], v[ok])


def _evmean(ctx, t, mask, values, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Event-conditional mean over the last n rows; denominator = finite events."""
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    m = np.asarray(mask, bool) & finite
    num = ctx.roll_sum(_grid(ctx, t, m, v), n, min_count=1)
    den = ctx.roll_sum(_grid(ctx, t, m), n, min_count=1)
    return ctx.safe_div(num, den, min_abs_den=1e-9), den


def _evstd(ctx, t, mask, values, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Event-conditional std over the last n rows; NaN if fewer than 3 events."""
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, bool) & np.isfinite(v)
    num1 = ctx.roll_sum(_grid(ctx, t, m, v), n, min_count=1)
    num2 = ctx.roll_sum(_grid(ctx, t, m, v * v), n, min_count=1)
    den = ctx.roll_sum(_grid(ctx, t, m), n, min_count=1)
    mean = ctx.safe_div(num1, den, min_abs_den=1e-9)
    var = np.clip(ctx.safe_div(num2, den, min_abs_den=1e-9) - mean * mean, 0.0, None)
    out = np.sqrt(var)
    out[den < 3] = np.nan
    return out, den


def _at_events(ctx, t, arr: np.ndarray) -> np.ndarray:
    """Panel (T,C) values sampled at each event's (day, code); NaN when not pool/off-grid."""
    idx = ctx.code_index(t["codes"])
    rows = np.searchsorted(ctx.panel.dates, t["days"])
    clipped = np.clip(rows, 0, ctx.panel.T - 1)
    on = (rows < ctx.panel.T) & (ctx.panel.dates[clipped] == t["days"])
    keep = (idx >= 0) & on
    out = np.full(len(t["days"]), np.nan, dtype=np.float64)
    out[keep] = arr[clipped[keep], idx[keep]]
    return out


def _share(ctx, t, mask, n: int) -> tuple[np.ndarray, np.ndarray]:
    num = ctx.roll_sum(_grid(ctx, t, mask), n, min_count=1)
    den = ctx.roll_sum(_grid(ctx, t, None), n, min_count=1)
    return ctx.safe_div(num, den, min_abs_den=1e-9), den


def _mask(ctx, out):
    return np.where(ctx.traded() & ctx.universe, out, np.nan)


# --------------------------------------------------------------------------- #
def build(parameters):
    variant = parameters["variant"]
    if variant not in PARAMETERS:
        raise KeyError(variant)

    def fn(ctx):
        tab = _tables(ctx)
        lu, ll = tab["lu"], tab["ll"]

        # ================= stock_limit_up (2018+) =========================
        if variant in ("oneword_share_60", "t_shape_share_60", "volume_type_share_60"):
            if lu is None:
                return _mask(ctx, ctx.panel.empty())
            typ = lu["type"]
            if variant == "oneword_share_60":
                prop = np.char.find(typ.astype(str), "一字") >= 0
            elif variant == "t_shape_share_60":
                prop = np.char.find(typ.astype(str), "T字") >= 0
            else:
                prop = np.char.find(typ.astype(str), "放量") >= 0
            out, _ = _share(ctx, lu, prop, 60)
            return _mask(ctx, out)

        if variant == "streak_share_60":
            if lu is None:
                return _mask(ctx, ctx.panel.empty())
            out, _ = _share(ctx, lu, lu["cd"] >= 2, 60)
            return _mask(ctx, out)

        if variant == "board_height_std_60":
            if lu is None:
                return _mask(ctx, ctx.panel.empty())
            out, _ = _evstd(ctx, lu, np.ones(len(lu["days"]), bool), lu["cd"], 60)
            return _mask(ctx, out)

        if variant == "strength_disp_60":
            if lu is None:
                return _mask(ctx, ctx.panel.empty())
            out, _ = _evstd(ctx, lu, np.ones(len(lu["days"]), bool), lu["flow"], 60)
            return _mask(ctx, out)

        if variant == "first_time_disp_60":
            if lu is None:
                return _mask(ctx, ctx.panel.empty())
            out, _ = _evstd(ctx, lu, np.ones(len(lu["days"]), bool), lu["first"], 60)
            return _mask(ctx, out)

        if variant == "first_board_vs_streak":
            if lu is None:
                return _mask(ctx, ctx.panel.empty())
            cd = lu["cd"]
            m1, _ = _evmean(ctx, lu, cd == 1, lu["flow"], 60)
            m2, _ = _evmean(ctx, lu, cd >= 2, lu["flow"], 60)
            return _mask(ctx, m2 - m1)

        if variant == "strength_term_10_60":
            if lu is None:
                return _mask(ctx, ctx.panel.empty())
            allm = np.ones(len(lu["days"]), bool)
            m10, _ = _evmean(ctx, lu, allm, lu["flow"], 10)
            m60, _ = _evmean(ctx, lu, allm, lu["flow"], 60)
            return _mask(ctx, m10 - m60)

        if variant == "open_term_10_60":
            if lu is None:
                return _mask(ctx, ctx.panel.empty())
            v = np.log1p(np.clip(lu["open"], 0.0, None))
            allm = np.ones(len(lu["days"]), bool)
            m10, _ = _evmean(ctx, lu, allm, v, 10)
            m60, _ = _evmean(ctx, lu, allm, v, 60)
            return _mask(ctx, m10 - m60)

        if variant == "latest_vs_history":
            if lu is None:
                return _mask(ctx, ctx.panel.empty())
            allm = np.ones(len(lu["days"]), bool)
            # exact place (NaN on non-event days) so ffill gives the LAST event value
            idx = ctx.code_index(lu["codes"])
            keep = idx >= 0
            latest_grid = ctx.panel.place(idx[keep].astype(np.int64), lu["days"][keep],
                                          lu["flow"][keep])
            latest = nan_fill_ffill(latest_grid)
            hist, _ = _evmean(ctx, lu, allm, lu["flow"], 60)
            return _mask(ctx, latest - hist)

        if variant == "next_ret_U_60":
            if lu is None:
                return _mask(ctx, ctx.panel.empty())
            e1 = ctx.shift(_grid(ctx, lu), 1) > 0
            r1 = ctx.ret_clean(1)
            valid = e1 & np.isfinite(r1)
            num = ctx.roll_sum(np.where(valid, r1, 0.0), 60, min_count=1)
            den = ctx.roll_sum(valid.astype(np.float64), 60, min_count=1)
            return _mask(ctx, ctx.safe_div(num, den, min_abs_den=0.5))

        if variant == "turnover_excess_60":
            if lu is None:
                return _mask(ctx, ctx.panel.empty())
            turn = ctx.px("turnover")
            turn_ev = _at_events(ctx, lu, turn)
            m, _ = _evmean(ctx, lu, np.ones(len(lu["days"]), bool), turn_ev, 60)
            normal = ctx.roll_mean(turn, 60, min_count=30)
            return _mask(ctx, m - normal)

        # ================= stock_limit_list (2020+) =======================
        if ll is None:
            return _mask(ctx, ctx.panel.empty())
        u, z, d = ll["is_u"], ll["is_z"], ll["is_d"]

        if variant == "queue_amount_ratio_60":
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(ll["amount"] > 0, ll["fd"] / ll["amount"], np.nan)
            out, _ = _evmean(ctx, ll, u, ratio, 60)
            return _mask(ctx, out)

        if variant == "up_stat_gap_60":
            s = pd.Series(ll["up_stat"]).astype(str).str.replace(" ", "", regex=False)
            parts = s.str.extract(r"^(\d+)/(\d+)$")
            x = pd.to_numeric(parts[0], errors="coerce").to_numpy(np.float64)
            y = pd.to_numeric(parts[1], errors="coerce").to_numpy(np.float64)
            gap = np.where(np.isfinite(x) & np.isfinite(y), y - x, np.nan)
            out, _ = _evmean(ctx, ll, u, gap, 60)
            return _mask(ctx, out)

        if variant == "down_limit_share_60":
            nu = ctx.roll_sum(_grid(ctx, ll, u), 60, min_count=1)
            nd = ctx.roll_sum(_grid(ctx, ll, d), 60, min_count=1)
            return _mask(ctx, ctx.safe_div(nd, nu + nd, min_abs_den=1e-9))

        if variant == "next_ret_Z_60":
            e1 = ctx.shift(_grid(ctx, ll, z), 1) > 0
            r1 = ctx.ret_clean(1)
            valid = e1 & np.isfinite(r1)
            num = ctx.roll_sum(np.where(valid, r1, 0.0), 60, min_count=1)
            den = ctx.roll_sum(valid.astype(np.float64), 60, min_count=1)
            return _mask(ctx, ctx.safe_div(num, den, min_abs_den=0.5))

        if variant == "next_ret_D_60":
            e1 = ctx.shift(_grid(ctx, ll, d), 1) > 0
            r1 = ctx.ret_clean(1)
            valid = e1 & np.isfinite(r1)
            num = ctx.roll_sum(np.where(valid, r1, 0.0), 60, min_count=1)
            den = ctx.roll_sum(valid.astype(np.float64), 60, min_count=1)
            return _mask(ctx, ctx.safe_div(num, den, min_abs_den=0.5))

        if variant == "down_up_turnover_asym_60":
            turn = ctx.px("turnover")
            turn_ev = _at_events(ctx, ll, turn)
            mu, _ = _evmean(ctx, ll, u, turn_ev, 60)
            md, _ = _evmean(ctx, ll, d, turn_ev, 60)
            return _mask(ctx, md - mu)

        if variant in ("broken_repair_rate_60", "repair_speed_60"):
            # event at e is observed at t=e+5 (5-day repair window)
            base5 = ctx.shift(_grid(ctx, ll, z), 5)
            tgts = [ctx.shift(_grid(ctx, ll, u), 4 - j) for j in range(5)]  # t-4..t
            hit = np.zeros_like(base5)
            first = np.full(base5.shape, np.nan, dtype=np.float64)
            for j, w in enumerate(tgts, start=1):
                new = (hit == 0) & (w > 0)
                first = np.where(new, float(j), first)
                hit = np.where(w > 0, 1.0, hit)
            if variant == "broken_repair_rate_60":
                num = ctx.roll_sum(np.where(base5 > 0, hit, 0.0), 60, min_count=1)
                den = ctx.roll_sum(base5, 60, min_count=1)
                return _mask(ctx, ctx.safe_div(num, den, min_abs_den=1e-9))
            valid = (base5 > 0) & np.isfinite(first)
            num = ctx.roll_sum(np.where(valid, first, 0.0), 60, min_count=1)
            den = ctx.roll_sum(valid.astype(np.float64), 60, min_count=1)
            return _mask(ctx, ctx.safe_div(num, den, min_abs_den=1e-9))

        if variant == "limit_down_recovery_rate_60":
            # D at e, next day e+1 is observed (traded) and not D -> recovered
            base1 = ctx.shift(_grid(ctx, ll, d), 1)
            ok = ctx.traded() & ctx.universe
            not_d = (1.0 - _grid(ctx, ll, d)) * ok
            rec = (base1 > 0) & ok
            num = ctx.roll_sum(np.where(rec, not_d, 0.0), 60, min_count=1)
            den = ctx.roll_sum(rec.astype(np.float64), 60, min_count=1)
            return _mask(ctx, ctx.safe_div(num, den, min_abs_den=1e-9))

        if variant == "limit_down_to_up_rate_60":
            base5 = ctx.shift(_grid(ctx, ll, d), 5)
            u_recent = ctx.roll_sum(_grid(ctx, ll, u), 5, min_count=1)
            rep = base5 > 0
            num = ctx.roll_sum(np.where(rep, (u_recent > 0).astype(np.float64), 0.0), 60, min_count=1)
            den = ctx.roll_sum(rep.astype(np.float64), 60, min_count=1)
            return _mask(ctx, ctx.safe_div(num, den, min_abs_den=1e-9))

        raise KeyError(variant)

    return fn
