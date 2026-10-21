"""R17 seat_flow_event batch 2: margin structure, including the short side.

Source
------
`stock_margin_detail` (2011-01-04 onward; 1-trading-day availability lag per
`conf/frequency.yaml`, verified in `fea/delay.py`):
  state  rzye (financing balance, 元), rqye (short balance, 元),
         rqyl (short balance, 股)
  flow   rzmre (financing buy, 元), rzche (financing repay, 元),
         rqmcl (short sell volume, 股)
Float market cap uses the PIT unadjusted close x current float shares
(`ctx.px("close") * ctx.px("float_share")`, 元).  Daily amount (元) and volume
(股) come from `ctx.px`.

PIT and missing semantics
-------------------------
Every factor is computed on the raw grid and the FINAL step is
`ctx.lag_grid(value, LAGGED_DATASETS["stock_margin_detail"]=1)`, so value(T)
only uses records with trade_date <= T-1.  State variables use as-of forward
fill with a 25-trading-day freshness mask: a stock that leaves the margin list
becomes NaN instead of carrying its last balance forever.  Flow variables use
an appearance-indicator grid: a stock outside the margin list is NaN, not zero;
true zeros inside a row (no short selling, suspension-day rzmre=0) are kept.
Rolling windows use explicit min_count (5->3, 20->10); ratios use positive
denominator floors.  Final mask = actual traded & universe on the evaluation
day (after the lag).

Batch 2 variants (16: 11 C + 5 GC)
----------------------------------
C (declared direction, primary horizon h5):
  fin_balance_to_float_mv_20   financing balance / float cap, 20d mean  (-)
  fin_balance_chg_20           20d financing balance growth              (+)
  fin_net_buy_to_balance_5     5d mean (buy-repay)/balance              (+)
  fin_buy_intensity_20         20d mean financing buy / float cap       (+)
  fin_balance_volatility_20    20d sd of daily balance growth           (-)
  short_balance_to_float_mv_20 short balance / float cap, 20d mean      (-)
  short_balance_chg_20         20d short balance growth                 (-)
  short_sell_volume_ratio_20   20d mean short-sell volume / stock volume(-)
  margin_price_divergence_20   cs-z(20d ret) - cs-z(20d balance growth) (-)
  margin_balance_to_amount_20  20d mean balance / daily amount (days)   (-)
  short_cover_intensity_5      implied 5d short covering / initial short (+ )
GC (descriptive only):
  fin_turnover_20              20d mean (buy+repay)/balance
  fin_net_buy_share_20         20d sum(buy-repay) / sum(buy+repay)
  short_sell_to_balance_5      5d mean short-sell volume / short balance
  fin_balance_drawdown_20      balance vs its own 20d peak
  short_balance_volatility_20  20d sd of daily short-balance growth
"""
from __future__ import annotations

import gc

import numpy as np

from fea.spec import LAGGED_DATASETS

WARMUP_DAYS = 260

MARGIN = "stock_margin_detail"
DAILY = "stock_daily"
FINANCE = "stock_finance"
ADJ = "stock_adj_factor"

MIN5, MIN20 = 3, 10
FRESH_N = 25  # longest state window (20) + buffer

_C = "_r17mg1_cache"

C_VARIANTS = (
    "fin_balance_to_float_mv_20",
    "fin_balance_chg_20",
    "fin_net_buy_to_balance_5",
    "fin_buy_intensity_20",
    "fin_balance_volatility_20",
    "short_balance_to_float_mv_20",
    "short_balance_chg_20",
    "short_sell_volume_ratio_20",
    "margin_price_divergence_20",
    "margin_balance_to_amount_20",
    "short_cover_intensity_5",
)
GC_VARIANTS = (
    "fin_turnover_20",
    "fin_net_buy_share_20",
    "short_sell_to_balance_5",
    "fin_balance_drawdown_20",
    "short_balance_volatility_20",
)
PARAMETERS = {name: {"variant": name} for name in C_VARIANTS + GC_VARIANTS}


def _ratio(num, den, floor=1e-6):
    d = np.asarray(den, dtype=np.float64)
    n = np.asarray(num, dtype=np.float64)
    out = np.full(np.broadcast(n, d).shape, np.nan, dtype=np.float64)
    ok = np.isfinite(d) & (d > floor) & np.isfinite(n)
    np.divide(n, d, out=out, where=ok)
    return out


def _mg_state(ctx, df, day, col, fresh_n=FRESH_N):
    """State variable -> as-of daily grid, NaN after `fresh_n` sessions without a row."""
    code = df["stock_code"].to_numpy()
    grid = np.asarray(ctx.asof_daily(code, day, df[col].to_numpy(np.float64)),
                      dtype=np.float64)
    last = np.asarray(ctx.asof_daily(code, day, np.asarray(day, dtype=np.float64)),
                      dtype=np.float64)
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    row = np.arange(dates.size, dtype=np.int64)[:, None]
    pos = np.searchsorted(dates, last)
    return np.where((row - pos) <= fresh_n, grid, np.nan)


def _mg_flow(ctx, df, day, col):
    """Flow variable -> exact grid; no row = NaN, true zeros preserved."""
    v = df[col].to_numpy(np.float64)
    code = df["stock_code"].to_numpy()
    got = ctx.event_grid(code, day, np.ones(v.size, dtype=np.float64))
    raw = ctx.event_grid(code, day, v)
    return _ratio(raw, got, 0.5)


class _MG1:
    def __init__(self, ctx):
        cache = ctx.__dict__.setdefault(_C, {})
        if "mg" in cache:
            self.__dict__.update(cache["mg"])
            return
        self.ctx = ctx
        tr = ctx.__dict__.get("_r17mg1_traded")
        if tr is None:
            tr = ctx.traded()
            ctx.__dict__["_r17mg1_traded"] = tr
        self.tr = tr
        d0, d1 = int(ctx.panel.dates[0]), int(ctx.panel.dates[-1])
        df = ctx.dataset(MARGIN, columns=[
            "stock_code", "trade_date", "rzye", "rzmre", "rzche",
            "rqye", "rqmcl", "rzrqye", "rqyl",
        ], years=(d0 // 10000, d1 // 10000))
        if df is None or df.empty:
            raise RuntimeError("stock_margin_detail returned no rows in the panel range")
        day = ctx.date_col(df["trade_date"])
        g = {
            "B": _mg_state(ctx, df, day, "rzye"),
            "S": _mg_state(ctx, df, day, "rqye"),
            "SQ": _mg_state(ctx, df, day, "rqyl"),
            "NB": _mg_flow(ctx, df, day, "rzmre"),
            "RP": _mg_flow(ctx, df, day, "rzche"),
            "SS": _mg_flow(ctx, df, day, "rqmcl"),
        }
        del df
        # Release the upstream partition cache before building the price grids.
        for key in list(getattr(ctx.up, "_cache", {})):
            if key and key[0] == MARGIN:
                ctx.up._cache.pop(key, None)
        gc.collect()
        g["MV"] = ctx.px("close") * ctx.px("float_share")
        g["AMT"] = ctx.px("amount")
        g["VOL"] = ctx.px("vol")
        cache["mg"] = {"g": g, "tr": tr}
        self.g = g
        ctx.__dict__[_C] = cache


def _compute(ctx, variant):
    m = _MG1(ctx)
    g = m.g
    tr = m.tr
    uni = ctx.universe
    B, S, SQ = g["B"], g["S"], g["SQ"]
    NB, RP, SS = g["NB"], g["RP"], g["SS"]
    MV, AMT, VOL = g["MV"], g["AMT"], g["VOL"]

    # Market-price-based ratios are only observed on traded days: the price
    # level and float shares are forward-filled through suspensions, so using
    # them on a suspended day would pair a stale market cap with a live margin
    # balance.  Pure margin ratios do not need this mask.
    def _traded(x):
        return np.where(tr, x, np.nan)

    if variant == "fin_balance_to_float_mv_20":
        x = _traded(_ratio(B, MV, 1e6))
        out = ctx.roll_mean(x, 20, MIN20)

    elif variant == "fin_balance_chg_20":
        out = ctx.pct_change(B, 20, min_abs_den=1e6)

    elif variant == "fin_net_buy_to_balance_5":
        x = _ratio(NB - RP, B, 1e6)
        out = ctx.roll_mean(x, 5, MIN5)

    elif variant == "fin_buy_intensity_20":
        x = _traded(_ratio(NB, MV, 1e6))
        out = ctx.roll_mean(x, 20, MIN20)

    elif variant == "fin_balance_volatility_20":
        out = ctx.roll_std(ctx.pct_change(B, 1, min_abs_den=1e6), 20, MIN20)

    elif variant == "short_balance_to_float_mv_20":
        x = _traded(_ratio(S, MV, 1e6))
        out = ctx.roll_mean(x, 20, MIN20)

    elif variant == "short_balance_chg_20":
        out = ctx.pct_change(S, 20, min_abs_den=1e4)

    elif variant == "short_sell_volume_ratio_20":
        x = _traded(_ratio(SS, VOL, 1.0))
        out = ctx.roll_mean(x, 20, MIN20)

    elif variant == "margin_price_divergence_20":
        r20 = ctx.ret_clean(20)
        g20 = ctx.pct_change(B, 20, min_abs_den=1e6)
        zr = ctx.cs_zscore(r20, mask=uni)
        zg = ctx.cs_zscore(g20, mask=uni)
        out = zr - zg

    elif variant == "margin_balance_to_amount_20":
        x = _traded(_ratio(B, AMT, 1e4))
        out = ctx.roll_mean(x, 20, MIN20)

    elif variant == "short_cover_intensity_5":
        sq0 = ctx.shift(SQ, 5)
        gross5 = ctx.roll_sum(SS, 5, 5)
        cover = gross5 - (SQ - sq0)
        out = _ratio(cover, sq0, 1.0)

    # ---- GC descriptors -------------------------------------------------
    elif variant == "fin_turnover_20":
        x = _ratio(NB + RP, B, 1e6)
        out = ctx.roll_mean(x, 20, MIN20)

    elif variant == "fin_net_buy_share_20":
        num = ctx.roll_sum(NB - RP, 20, MIN20)
        den = ctx.roll_sum(NB + RP, 20, MIN20)
        out = _ratio(num, den, 1e4)

    elif variant == "short_sell_to_balance_5":
        x = _ratio(SS, SQ, 1.0)
        out = ctx.roll_mean(x, 5, MIN5)

    elif variant == "fin_balance_drawdown_20":
        mx = ctx.roll_max(B, 20, MIN20)
        out = _ratio(B - mx, mx, 1e6)

    elif variant == "short_balance_volatility_20":
        out = ctx.roll_std(ctx.pct_change(S, 1, min_abs_den=1e4), 20, MIN20)

    else:
        raise KeyError(f"unknown variant {variant!r}")

    lag = int(LAGGED_DATASETS.get(MARGIN, 1))
    out = ctx.lag_grid(out, lag)
    return np.where(tr & uni, out, np.nan)


def build(parameters):
    variant = parameters["variant"]
    if variant not in PARAMETERS:
        raise KeyError(variant)

    def fn(ctx):
        return _compute(ctx, variant)

    return fn
