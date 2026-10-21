"""R17 seat_flow_event batch 5: cross-source interactions (flow x margin x price).

All legs are PIT: current-day flow/price and margin records lagged one trading
day.  A margin-based variant computes on the raw grids and shifts the final
factor by one panel row, so value(T) uses margin records <= T-1.  Forecast and
holder legs are used at their own announcement date; when they are combined with
a margin leg, the final one-row shift makes the whole product conservative.

Sources: stock_main_fund_flow (tier net flow), stock_margin_detail (financing
balance and net buy), stock_forecast (type score), stock_holder_number (holder
count), stock_daily/stock_adj_factor (returns, vol), stock_finance (float
shares).  No label is read; final mask = traded & universe.

Batch 5 variants (8: 6 C + 2 GC)
--------------------------------
C:
  flow_margin_alignment_20     20d corr(main net flow, financing net buy/MV)   (-)
  flow_highvol_flow_60         main flow on high-vol minus low-vol days (60d)  (+)
  margin_dip_buying_20         financing net buy on 20d-down minus up states   (-)
  flow_forecast_confirmation_20 20d corr(main flow, latest forecast score)     (+)
  margin_holder_concentration_20  z(fin balance/MV) x z(log shares per holder) (-)
  crowding_triple_20           z(flow 20d) x z(margin balance growth) x z(ret20)(-)
GC:
  flow_margin_divergence_20    z(flow 20d) - z(margin balance growth 20d)
  margin_flow_price_timing_20  flow-weighted return of financing net buying
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd

WARMUP_DAYS = 400

FLOW = "stock_main_fund_flow"
MARGIN = "stock_margin_detail"
FC = "stock_forecast"
HN = "stock_holder_number"
TIERS = ("sm", "md", "lg", "elg")
MIN5, MIN20, MIN60 = 3, 10, 30
FRESH_N = 25

TYPE_SCORE = {
    "预增": 2.0, "扭亏": 2.0, "略增": 1.0, "续盈": 1.0, "不确定": 0.0,
    "略减": -1.0, "预减": -2.0, "首亏": -2.0, "续亏": -2.0,
}

_C = "_r17x1_cache"

C_VARIANTS = (
    "flow_margin_alignment_20",
    "flow_highvol_flow_60",
    "margin_dip_buying_20",
    "flow_forecast_confirmation_20",
    "margin_holder_concentration_20",
    "crowding_triple_20",
)
GC_VARIANTS = (
    "flow_margin_divergence_20",
    "margin_flow_price_timing_20",
)
PARAMETERS = {name: {"variant": name} for name in C_VARIANTS + GC_VARIANTS}


def _ratio(num, den, floor=1e-6):
    d = np.asarray(den, dtype=np.float64)
    n = np.asarray(num, dtype=np.float64)
    out = np.full(np.broadcast(n, d).shape, np.nan, dtype=np.float64)
    ok = np.isfinite(d) & (d > floor) & np.isfinite(n)
    np.divide(n, d, out=out, where=ok)
    return out


def _ymd(s):
    return pd.to_numeric(
        pd.to_datetime(s, errors="coerce").dt.strftime("%Y%m%d"),
        errors="coerce").astype("float64")


def _mg_state(ctx, df, day, col, fresh_n=FRESH_N):
    code = df["stock_code"].to_numpy()
    grid = np.asarray(ctx.asof_daily(code, day, df[col].to_numpy(np.float64)), dtype=np.float64)
    last = np.asarray(ctx.asof_daily(code, day, np.asarray(day, dtype=np.float64)), dtype=np.float64)
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    row = np.arange(dates.size, dtype=np.int64)[:, None]
    pos = np.searchsorted(dates, last)
    return np.where((row - pos) <= fresh_n, grid, np.nan)


def _mg_flow(ctx, df, day, col):
    v = df[col].to_numpy(np.float64)
    code = df["stock_code"].to_numpy()
    got = ctx.event_grid(code, day, np.ones(v.size, dtype=np.float64))
    raw = ctx.event_grid(code, day, v)
    return _ratio(raw, got, 0.5)


class _X1:
    def __new__(cls, ctx):
        cache = ctx.__dict__.setdefault(_C, {})
        if "x" in cache:
            return cache["x"]
        self = super().__new__(cls)
        cache["x"] = self
        return self

    def __init__(self, ctx):
        if getattr(self, "_ready", False):
            return
        self.ctx = ctx
        tr = ctx.__dict__.get("_r17x1_traded")
        if tr is None:
            tr = ctx.traded()
            ctx.__dict__["_r17x1_traded"] = tr
        self.tr = tr
        self._flow = None
        self._mg = None
        self._fc = None
        self._hn = None
        self._ready = True

    # ---- fund flow ------------------------------------------------------
    @property
    def flow(self):
        if self._flow is not None:
            return self._flow
        ctx = self.ctx
        d0, d1 = int(ctx.panel.dates[0]), int(ctx.panel.dates[-1])
        shape = ctx.panel.shape
        raw_net = np.zeros(shape)
        raw_turn = np.zeros(shape)
        got = np.zeros(shape)
        for year in range(d0 // 10000, d1 // 10000 + 1):
            df = ctx.dataset(FLOW, columns=[
                "stock_code", "trade_date",
                *[f"buy_{t}_amount" for t in TIERS],
                *[f"sell_{t}_amount" for t in TIERS],
            ], years=(year, year))
            if df is None or df.empty:
                continue
            codes = df["stock_code"].to_numpy()
            days = ctx.date_col(df["trade_date"])
            buy = sum(df[f"buy_{t}_amount"].to_numpy(np.float64) for t in TIERS)
            sell = sum(df[f"sell_{t}_amount"].to_numpy(np.float64) for t in TIERS)
            main = (df["buy_lg_amount"].to_numpy(np.float64)
                    - df["sell_lg_amount"].to_numpy(np.float64)
                    + df["buy_elg_amount"].to_numpy(np.float64)
                    - df["sell_elg_amount"].to_numpy(np.float64))
            turn = 0.5 * (buy + sell)
            got += ctx.event_grid(codes, days, np.ones(len(df)))
            raw_net += ctx.event_grid(codes, days, main).astype(np.float64)
            raw_turn += ctx.event_grid(codes, days, turn).astype(np.float64)
            del df, buy, sell, main, turn
            for key in list(getattr(ctx.up, "_cache", {})):
                if key and key[0] == FLOW and (len(key) < 3 or key[2] == (year, year)):
                    ctx.up._cache.pop(key, None)
            gc.collect()
        have = got > 0.5
        net = np.where(have, raw_net / np.where(have, got, 1.0), np.nan)
        turn = np.where(have, raw_turn / np.where(have, got, 1.0), np.nan)
        self._flow = {"net": np.where(self.tr, net, np.nan),
                      "turn": np.where(self.tr, turn, np.nan)}
        return self._flow

    def flow_share(self):
        return _ratio(self.flow["net"], self.flow["turn"])

    # ---- margin ---------------------------------------------------------
    @property
    def mg(self):
        if self._mg is not None:
            return self._mg
        ctx = self.ctx
        d0, d1 = int(ctx.panel.dates[0]), int(ctx.panel.dates[-1])
        df = ctx.dataset(MARGIN, columns=[
            "stock_code", "trade_date", "rzye", "rzmre", "rzche"], years=(d0 // 10000, d1 // 10000))
        day = ctx.date_col(df["trade_date"])
        g = {
            "B": _mg_state(ctx, df, day, "rzye"),
            "NB": _mg_flow(ctx, df, day, "rzmre"),
            "RP": _mg_flow(ctx, df, day, "rzche"),
        }
        del df
        for key in list(getattr(ctx.up, "_cache", {})):
            if key and key[0] == MARGIN:
                ctx.up._cache.pop(key, None)
        gc.collect()
        g["MV"] = ctx.px("close") * ctx.px("float_share")
        self._mg = g
        return g

    def margin_net_intensity(self):
        g = self.mg
        x = _ratio(g["NB"] - g["RP"], g["MV"], 1e6)
        return np.where(self.tr, x, np.nan)

    # ---- forecast score -------------------------------------------------
    @property
    def fc(self):
        if self._fc is not None:
            return self._fc
        ctx = self.ctx
        d1 = int(ctx.panel.dates[-1])
        df = ctx.dataset(FC, columns=["stock_code", "ann_date", "end_date", "type"],
                         years=(2015, d1 // 10000))
        df = df.copy()
        df["ann"] = _ymd(df["ann_date"])
        df["end"] = _ymd(df["end_date"])
        df["score"] = df["type"].astype(str).map(TYPE_SCORE).astype(np.float64)
        df = df[np.isfinite(df["ann"]) & np.isfinite(df["end"]) & np.isfinite(df["score"])]
        df = df.sort_values(["stock_code", "ann", "end"], kind="stable")
        df = df.drop_duplicates(["stock_code", "ann"], keep="last")
        grid = np.asarray(ctx.asof_daily(
            df["stock_code"].to_numpy(), df["ann"].to_numpy(),
            df["score"].to_numpy(np.float64)), dtype=np.float64)
        for key in list(getattr(ctx.up, "_cache", {})):
            if key and key[0] == FC:
                ctx.up._cache.pop(key, None)
        self._fc = grid
        return grid

    # ---- latest holder count -------------------------------------------
    @property
    def hn(self):
        if self._hn is not None:
            return self._hn
        ctx = self.ctx
        d1 = int(ctx.panel.dates[-1])
        df = ctx.dataset(HN, columns=["stock_code", "ann_date", "end_date", "holder_num"],
                         years=(2015, d1 // 10000))
        df = df.copy()
        df["ann"] = _ymd(df["ann_date"])
        df["end"] = _ymd(df["end_date"])
        df["n"] = pd.to_numeric(df["holder_num"], errors="coerce")
        df = df[np.isfinite(df["ann"]) & np.isfinite(df["end"]) & (df["n"] > 0)]
        df["ann"] = df["ann"].astype(np.int64)
        df["end"] = df["end"].astype(np.int64)
        df["end_month"] = df["end"] // 100
        df = df.sort_values(["stock_code", "end", "ann"], kind="stable")
        df = df.drop_duplicates(["stock_code", "end_month"], keep="last")
        df = df.sort_values(["stock_code", "ann", "end"], kind="stable")
        df = df.drop_duplicates(["stock_code", "ann"], keep="last")
        grid = np.asarray(ctx.asof_daily(
            df["stock_code"].to_numpy(), df["ann"].to_numpy(),
            df["n"].to_numpy(np.float64)), dtype=np.float64)
        for key in list(getattr(ctx.up, "_cache", {})):
            if key and key[0] == HN:
                ctx.up._cache.pop(key, None)
        self._hn = grid
        return grid


def _cond_mean_diff(ctx, value, state, n, min_each):
    """mean(value | state=1) - mean(value | state=0) over n sessions.

    Only state/value pairs where both are finite enter either mean; a window
    needs at least `min_each` pairs on each side.
    """
    s = np.asarray(state, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(v)
    one = ok & (s == 1.0)
    zero = ok & (s == 0.0)
    c1 = ctx.roll_sum(np.where(one, 1.0, np.nan), n, min_each)
    c0 = ctx.roll_sum(np.where(zero, 1.0, np.nan), n, min_each)
    m1 = _ratio(ctx.roll_sum(np.where(one, v, np.nan), n, min_each), c1, 0.5)
    m0 = _ratio(ctx.roll_sum(np.where(zero, v, np.nan), n, min_each), c0, 0.5)
    return np.where((c1 >= min_each) & (c0 >= min_each), m1 - m0, np.nan)


def _compute(ctx, variant):
    x = _X1(ctx)
    tr, uni = x.tr, ctx.universe
    flow = x.flow_share()
    ret1 = ctx.ret_clean(1)

    if variant == "flow_margin_alignment_20":
        out = ctx.roll_corr(flow, x.margin_net_intensity(), 20, MIN20)
        out = ctx.lag_grid(out, 1)

    elif variant == "flow_highvol_flow_60":
        vol = ctx.roll_std(ret1, 20, MIN20)
        med = ctx.roll_quantile(vol, 60, 0.5, MIN60)
        state = np.where(np.isfinite(vol) & np.isfinite(med), vol > med, np.nan)
        out = _cond_mean_diff(ctx, flow, state, 60, 10)

    elif variant == "margin_dip_buying_20":
        # Use the 20-day price state, not the daily sign.
        r20 = ctx.ret_clean(20)
        state = np.where(np.isfinite(r20), r20 < 0, np.nan)
        out = _cond_mean_diff(ctx, x.margin_net_intensity(), state, 20, 5)
        out = ctx.lag_grid(out, 1)

    elif variant == "flow_forecast_confirmation_20":
        out = ctx.roll_corr(flow, x.fc, 20, MIN20)

    elif variant == "margin_holder_concentration_20":
        g = x.mg
        lev = ctx.roll_mean(np.where(tr, _ratio(g["B"], g["MV"], 1e6), np.nan), 20, MIN20)
        avg = ctx.safe_log(_ratio(ctx.px("float_share"), x.hn, 1.0))
        z1 = ctx.cs_zscore(lev, mask=uni)
        z2 = ctx.cs_zscore(avg, mask=uni)
        out = ctx.lag_grid(z1 * z2, 1)

    elif variant == "crowding_triple_20":
        g = x.mg
        zf = ctx.cs_zscore(ctx.roll_sum(flow, 20, MIN20), mask=uni)
        zm = ctx.cs_zscore(ctx.pct_change(g["B"], 20, min_abs_den=1e6), mask=uni)
        zp = ctx.cs_zscore(ctx.ret_clean(20), mask=uni)
        out = ctx.lag_grid(zf * zm * zp, 1)

    elif variant == "flow_margin_divergence_20":
        g = x.mg
        zf = ctx.cs_zscore(ctx.roll_sum(flow, 20, MIN20), mask=uni)
        zm = ctx.cs_zscore(ctx.pct_change(g["B"], 20, min_abs_den=1e6), mask=uni)
        out = ctx.lag_grid(zf - zm, 1)

    elif variant == "margin_flow_price_timing_20":
        m = x.margin_net_intensity()
        num = ctx.roll_sum(np.where(np.isfinite(m) & np.isfinite(ret1), m * ret1, np.nan),
                           20, MIN20)
        den = ctx.roll_sum(np.abs(m), 20, MIN20)
        out = ctx.lag_grid(_ratio(num, den, 1e-8), 1)

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
