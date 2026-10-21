"""R17 seat_flow_event batch 1: fund-flow structure and dynamics.

Source
------
`stock_main_fund_flow` (2010-01-04 onward), 4 order-size tiers
sm / md / lg / elg, each recorded on the buy and the sell side (double entry).
Amount unit is 万元, volume unit is 手.  The vendor double-entry identity
`sum_t buy_t == sum_t sell_t` holds only approximately: 2020+ p99 relative
gap ~1e-5, but 2018 contains dirty rows with percent-level gaps.  Therefore
every net share uses the symmetric one-side turnover

    turn = 0.5 * (sum_t buy_t + sum_t sell_t)

as denominator, and gross shares use `gtot = sum_t (buy_t + sell_t)`
(which sums to one by construction).

Missing / zero semantics
------------------------
The table is a daily event table.  A (day, stock) with no row is **NaN**, not
zero.  Rows that exist on suspended days are masked out by `ctx.traded()`.
Rolling windows always pass an explicit `min_count` (5->3, 10->5, 20->10,
60->30), the same relaxation convention as `factors/fundflow.py`; a window
with too few observations is NaN, never zero.

Price/PIT
---------
All signals use same-day or trailing data only; `ctx.ret_clean(k)` masks
suspension days and uses hfq returns (stock_daily + stock_adj_factor).  The
frozen stock axis and actual traded mask are applied last.  No label, no qfq,
no external mapping, no future fit.

Family construction (batch 1, 23 variants)
------------------------------------------
C channel (declared direction, primary horizon h5):
  elg_lg_net_spread_10        within-main net-flow composition, XL minus LG
  elg_sm_pressure_spread_10   within-tier buy pressure, XL minus SM
  elg_lg_pressure_spread_10   within-tier buy pressure, XL minus LG
  main_net_accel_5_20         main net-flow 5d block acceleration / own 60d sd
  main_net_volatility_20      20d sd of daily main net flow share
  flow_price_corr_20          20d corr(main net flow, daily return)
  flow_feedback_20            20d corr(main net flow(t), return(t-1))
  extreme_inflow_z_20         max daily inflow vs own 60d mean / sd
  down_day_flow_bias_20       flow on down days minus flow on up days
  flow_price_divergence_10    cs-z(10d return) - cs-z(10d main net flow)
  flow_consistency_20         |sum flow| / sum |flow| over 20d
  elg_buy_sell_price_gap_5    within-ELG buy VWAP / sell VWAP, 5d mean

GC channel (descriptive purpose only, no predictive claim):
  tier_hhi_gross_20           concentration of turnover across size tiers
  elg_gross_share_dev_20      XL gross share vs own 60d history
  small_mid_gross_split_20    small / (small+mid) gross turnover split
  tier_net_dispersion_10      L1 dispersion of tier net-flow shares
  net_hhi_abs_10              concentration of the imbalance across tiers
  flow_skew_20                skewness of daily main net-flow share
  flow_commonality_20         corr with equal-weight cross-sectional mean flow
  high_volume_flow_share_20   share of |net flow| on above-median turnover days
  avg_trade_size_dev_20       log(avg trade size) vs own 60d median
  flow_impact_lambda_20       sum(ret*flow) / sum(flow^2), Kyle-lambda proxy
  main_share_of_abs_net_20    main-tier share of total absolute tier imbalance
"""
from __future__ import annotations

import gc

import numpy as np

WARMUP_DAYS = 260  # 60 trading-day windows + 5-day shifts + suspension gaps

FLOW = "stock_main_fund_flow"
TIERS = ("sm", "md", "lg", "elg")

# Explicit min_count for rolling windows: (window -> minimum valid observations).
MIN5, MIN10, MIN20, MIN60 = 3, 5, 10, 30

_C = "_r17ff1_cache"

C_VARIANTS = (
    "elg_lg_net_spread_10",
    "elg_sm_pressure_spread_10",
    "elg_lg_pressure_spread_10",
    "main_net_accel_5_20",
    "main_net_volatility_20",
    "flow_price_corr_20",
    "flow_feedback_20",
    "extreme_inflow_z_20",
    "down_day_flow_bias_20",
    "flow_price_divergence_10",
    "flow_consistency_20",
    "elg_buy_sell_price_gap_5",
)
GC_VARIANTS = (
    "tier_hhi_gross_20",
    "elg_gross_share_dev_20",
    "small_mid_gross_split_20",
    "tier_net_dispersion_10",
    "net_hhi_abs_10",
    "flow_skew_20",
    "flow_commonality_20",
    "high_volume_flow_share_20",
    "avg_trade_size_dev_20",
    "flow_impact_lambda_20",
    "main_share_of_abs_net_20",
)
PARAMETERS = {name: {"variant": name} for name in C_VARIANTS + GC_VARIANTS}


def _ratio(num, den, floor=1e-6):
    """num/den with a *positive* denominator floor; NaN outside the positive domain."""
    d = np.asarray(den, dtype=np.float64)
    n = np.asarray(num, dtype=np.float64)
    out = np.full(np.broadcast(n, d).shape, np.nan, dtype=np.float64)
    ok = np.isfinite(d) & (d > floor) & np.isfinite(n)
    np.divide(n, d, out=out, where=ok)
    return out


class _FF1:
    """Year-by-year fund-flow loader into (T, C) grids, cached on the context.

    Peak memory is one yearly partition plus the accumulated float64 grids.
    Every grid is divided by an appearance-count grid (missing row -> NaN)
    and then masked by the actual traded mask.
    """

    def __init__(self, ctx):
        cache = ctx.__dict__.setdefault(_C, {})
        if "ff" in cache:
            self.__dict__.update(cache["ff"])
            return
        self.ctx = ctx
        tr = ctx.__dict__.get("_r17ff1_traded")
        if tr is None:
            tr = ctx.traded()
            ctx.__dict__["_r17ff1_traded"] = tr
        self.tr = tr
        shape = ctx.panel.shape
        keys = (
            [f"net_{t}" for t in TIERS]
            + [f"gross_{t}" for t in TIERS]
            + ["turn", "turnvol", "buy_elg_vol", "sell_elg_vol"]
        )
        raw = {k: np.zeros(shape, dtype=np.float64) for k in keys}
        got = np.zeros(shape, dtype=np.float64)
        d0, d1 = int(ctx.panel.dates[0]), int(ctx.panel.dates[-1])
        for year in range(d0 // 10000, d1 // 10000 + 1):
            df = ctx.dataset(FLOW, columns=[
                "stock_code", "trade_date",
                *[f"buy_{t}_amount" for t in TIERS],
                *[f"sell_{t}_amount" for t in TIERS],
                *[f"buy_{t}_vol" for t in TIERS],
                *[f"sell_{t}_vol" for t in TIERS],
            ], years=(year, year))
            if df is None or df.empty:
                continue
            codes = df["stock_code"].to_numpy()
            days = ctx.date_col(df["trade_date"])
            buy = {t: df[f"buy_{t}_amount"].to_numpy(np.float64) for t in TIERS}
            sell = {t: df[f"sell_{t}_amount"].to_numpy(np.float64) for t in TIERS}
            buyv = {t: df[f"buy_{t}_vol"].to_numpy(np.float64) for t in TIERS}
            sellv = {t: df[f"sell_{t}_vol"].to_numpy(np.float64) for t in TIERS}
            turn = 0.5 * (sum(buy.values()) + sum(sell.values()))
            turnvol = 0.5 * (sum(buyv.values()) + sum(sellv.values()))
            got += ctx.event_grid(codes, days, np.ones(len(df), dtype=np.float64))
            for t in TIERS:
                raw[f"net_{t}"] += ctx.event_grid(
                    codes, days, buy[t] - sell[t]).astype(np.float64)
                raw[f"gross_{t}"] += ctx.event_grid(
                    codes, days, buy[t] + sell[t]).astype(np.float64)
            raw["turn"] += ctx.event_grid(codes, days, turn).astype(np.float64)
            raw["turnvol"] += ctx.event_grid(codes, days, turnvol).astype(np.float64)
            raw["buy_elg_vol"] += ctx.event_grid(
                codes, days, buyv["elg"]).astype(np.float64)
            raw["sell_elg_vol"] += ctx.event_grid(
                codes, days, sellv["elg"]).astype(np.float64)
            del df, buy, sell, buyv, sellv, turn, turnvol
            # Upstream.read memoizes every partition it touches; drop this year's
            # entry so a 10-year build does not hold all raw partitions at once.
            for key in list(getattr(ctx.up, "_cache", {})):
                if key and key[0] == FLOW and (len(key) < 3 or key[2] == (year, year)):
                    ctx.up._cache.pop(key, None)
            gc.collect()
        have = got > 0.5
        for k in raw:
            raw[k] = np.where(have, raw[k] / np.where(have, got, 1.0), np.nan)
            raw[k] = np.where(self.tr, raw[k], np.nan)
        cache["ff"] = {"g": raw, "tr": tr}
        self.g = raw
        ctx.__dict__[_C] = cache

    # ---- primitives -----------------------------------------------------
    def net(self, tier):
        return self.g[f"net_{tier}"]

    def gross(self, tier):
        return self.g[f"gross_{tier}"]

    def turn(self):
        return self.g["turn"]

    def gtot(self):
        return sum(self.g[f"gross_{t}"] for t in TIERS)

    def gross_share(self, tier):
        return _ratio(self.gross(tier), self.gtot())

    def net_share(self, tier):
        return _ratio(self.net(tier), self.turn())

    def pressure(self, tier):
        """Within-tier buy pressure (buy-sell)/(buy+sell), dimensionless."""
        return _ratio(self.net(tier), self.gross(tier))

    def main_net(self):
        return self.net("lg") + self.net("elg")

    def main_net_share(self):
        return _ratio(self.main_net(), self.turn())


# ---------------------------------------------------------------------------
# variant implementations
# ---------------------------------------------------------------------------
def _compute(ctx, variant):
    f = _FF1(ctx)
    tr = f.tr
    uni = ctx.universe

    if variant == "elg_lg_net_spread_10":
        x = _ratio(f.net("elg") - f.net("lg"), f.turn())
        out = ctx.roll_mean(x, 10, MIN10)

    elif variant == "elg_sm_pressure_spread_10":
        x = f.pressure("elg") - f.pressure("sm")
        out = ctx.roll_mean(x, 10, MIN10)

    elif variant == "elg_lg_pressure_spread_10":
        x = f.pressure("elg") - f.pressure("lg")
        out = ctx.roll_mean(x, 10, MIN10)

    elif variant == "main_net_accel_5_20":
        flow = f.main_net_share()
        d5 = ctx.roll_sum(flow, 5, MIN5)
        accel = d5 - ctx.shift(d5, 5)
        scale = ctx.roll_std(flow, 60, MIN60) * np.sqrt(5.0)
        out = _ratio(accel, scale, 1e-8)

    elif variant == "main_net_volatility_20":
        out = ctx.roll_std(f.main_net_share(), 20, MIN20)

    elif variant == "flow_price_corr_20":
        out = ctx.roll_corr(f.main_net_share(), ctx.ret_clean(1), 20, MIN20)

    elif variant == "flow_feedback_20":
        out = ctx.roll_corr(f.main_net_share(), ctx.shift(ctx.ret_clean(1), 1),
                            20, MIN20)

    elif variant == "extreme_inflow_z_20":
        flow = f.main_net_share()
        mx = ctx.roll_max(flow, 20, MIN20)
        mu = ctx.shift(ctx.roll_mean(flow, 60, MIN60), 1)
        sd = ctx.shift(ctx.roll_std(flow, 60, MIN60), 1)
        out = _ratio(mx - mu, sd, 1e-8)

    elif variant == "down_day_flow_bias_20":
        flow = f.main_net_share()
        r = ctx.ret_clean(1)
        valid = np.isfinite(r)
        neg = np.where(valid, (r < 0).astype(np.float64), np.nan)
        pos = np.where(valid, (r > 0).astype(np.float64), np.nan)
        cnt_neg = ctx.roll_sum(neg, 20, 20)
        cnt_pos = ctx.roll_sum(pos, 20, 20)
        sum_neg = ctx.roll_sum(np.where(neg == 1.0, flow, 0.0), 20, 20)
        sum_pos = ctx.roll_sum(np.where(pos == 1.0, flow, 0.0), 20, 20)
        mean_neg = _ratio(sum_neg, cnt_neg, 0.5)
        mean_pos = _ratio(sum_pos, cnt_pos, 0.5)
        out = np.where((cnt_neg >= 5) & (cnt_pos >= 5), mean_neg - mean_pos,
                       np.nan)

    elif variant == "flow_price_divergence_10":
        r10 = ctx.ret_clean(10)
        f10 = ctx.roll_sum(f.main_net_share(), 10, MIN10)
        zr = ctx.cs_zscore(r10, mask=uni)
        zf = ctx.cs_zscore(f10, mask=uni)
        out = zr - zf

    elif variant == "flow_consistency_20":
        flow = f.main_net_share()
        s = ctx.roll_sum(flow, 20, MIN20)
        a = ctx.roll_sum(np.abs(flow), 20, MIN20)
        out = _ratio(np.abs(s), a, 1e-8)

    elif variant == "elg_buy_sell_price_gap_5":
        g = f.gross("elg")
        n = f.net("elg")
        b_amt = 0.5 * (g + n)
        s_amt = 0.5 * (g - n)
        bp = _ratio(b_amt, f.g["buy_elg_vol"], 1e-9)
        sp = _ratio(s_amt, f.g["sell_elg_vol"], 1e-9)
        lr = ctx.safe_log(_ratio(bp, sp, 1e-9))
        out = ctx.roll_mean(lr, 5, MIN5)

    # ---- GC descriptors -------------------------------------------------
    elif variant == "tier_hhi_gross_20":
        gs = [f.gross_share(t) for t in TIERS]
        hhi = sum(x * x for x in gs)
        out = ctx.roll_mean(hhi, 20, MIN20)

    elif variant == "elg_gross_share_dev_20":
        gs = f.gross_share("elg")
        mu = ctx.shift(ctx.roll_mean(gs, 60, MIN60), 1)
        sd = ctx.shift(ctx.roll_std(gs, 60, MIN60), 1)
        out = _ratio(gs - mu, sd, 1e-6)

    elif variant == "small_mid_gross_split_20":
        x = _ratio(f.gross("sm"), f.gross("sm") + f.gross("md"))
        out = ctx.roll_mean(x, 20, MIN20)

    elif variant == "tier_net_dispersion_10":
        l1 = sum(np.abs(f.net(t)) for t in TIERS)
        x = _ratio(l1, 2.0 * f.turn())
        out = ctx.roll_mean(x, 10, MIN10)

    elif variant == "net_hhi_abs_10":
        a = [np.abs(f.net(t)) for t in TIERS]
        s = sum(a)
        hhi = sum(_ratio(x, s) ** 2 for x in a)
        out = ctx.roll_mean(hhi, 10, MIN10)

    elif variant == "flow_skew_20":
        out = ctx.roll_skew(f.main_net_share(), 20, MIN20)

    elif variant == "flow_commonality_20":
        flow = f.main_net_share()
        valid = np.isfinite(flow) & uni
        cnt = valid.sum(axis=1)
        tot = np.where(valid, flow, 0.0).sum(axis=1)
        mean = tot / np.maximum(cnt, 1)
        mkt = np.where((cnt >= 100)[:, None], mean[:, None], np.nan)
        mkt = np.broadcast_to(mkt, flow.shape)
        out = ctx.roll_corr(flow, mkt, 20, MIN20)

    elif variant == "high_volume_flow_share_20":
        abs_main = np.abs(f.main_net())
        med = ctx.roll_quantile(f.turn(), 20, 0.5, MIN20)
        hi = np.isfinite(f.turn()) & np.isfinite(med) & (f.turn() > med)
        num = ctx.roll_sum(np.where(hi, abs_main, 0.0), 20, MIN20)
        den = ctx.roll_sum(abs_main, 20, MIN20)
        out = _ratio(num, den, 1e-8)

    elif variant == "avg_trade_size_dev_20":
        avg = _ratio(f.turn(), f.g["turnvol"], 1e-9)
        la = ctx.safe_log(avg)
        med = ctx.shift(ctx.roll_quantile(la, 60, 0.5, MIN60), 1)
        out = la - med

    elif variant == "flow_impact_lambda_20":
        flow = f.main_net_share()
        r = ctx.ret_clean(1)
        pair = np.isfinite(flow) & np.isfinite(r)
        sxy = ctx.roll_sum(np.where(pair, flow * r, np.nan), 20, MIN20)
        sxx = ctx.roll_sum(np.where(pair, flow * flow, np.nan), 20, MIN20)
        out = _ratio(sxy, sxx, 1e-8)

    elif variant == "main_share_of_abs_net_20":
        a = [np.abs(f.net(t)) for t in TIERS]
        x = _ratio(np.abs(f.main_net()), sum(a))
        out = ctx.roll_mean(x, 20, MIN20)

    else:
        raise KeyError(f"unknown variant {variant!r}")

    return np.where(tr & uni, out, np.nan)


def build(parameters):
    variant = parameters["variant"]
    if variant not in PARAMETERS:
        raise KeyError(variant)

    def fn(ctx):
        return _compute(ctx, variant)

    return fn
