"""R17 seat_flow_event batch 4: holder-number and pledge ownership structure.

Sources
-------
`stock_holder_number` (2016 onward; irregular).  A row is a holder-count
observation as of `end_date`, disclosed on `ann_date` (delay 0 per
conf/frequency.yaml).  holder_num <= 0 is invalid -> NaN.  Multiple rows can
share an end month (month-end date variants); they are collapsed to one period
per (stock, YYYYMM) keeping the latest ann_date, so a 1-day-apart duplicate does
not create a fake "change".

`stock_pledge_stat` (2014 onward; weekly in the actual data, although
conf/frequency.yaml calls it quarterly).  Rows carry only `end_date`, no
announcement date.  We apply a documented **15-trading-day conservative lag**
(`PLEDGE_LAG = 15`) as the final step; this is the established convention in the
r4s/r9d pledge candidates.  LIMITATION: 15 sessions is a lower bound on the
disclosure delay, not a verified PIT date.  Any residual look-ahead would bias
the factor; pledge candidates are therefore evidence-limited and the chief must
treat their G1/PIT status as not fully proven.  `unrest_pledge`/`rest_pledge` and
`total_share` are 万股; pledge_ratio is percent = pledged/total_share*100
(verified on 2024: median absolute identity error 0.0024pp).  pledged shares are
converted to 股 by *1e4.

All per-period history uses only earlier periods.  Missing values are NaN, never
zero.  Final mask = actual traded & universe.

Batch 4 variants (20: 10 C + 10 GC)
-----------------------------------
Holder C (ann_date, no lag):
  change_accel             second difference of log holder count        (-)
  seasonal_innovation      latest change vs same-quarter prior changes  (-)
  accumulation_divergence  cs-z(-change) - cs-z(period price return)    (+)
  delay_own_history        disclosure delay minus own prior median      (-)
  change_volatility        sd of the last 6 log changes                 (-)
Holder GC:
  log_level                log(holder_num), latest                      (descriptor)
  avg_holding              log(float_share / holder_num)                (descriptor)
  report_count_250         disclosures in the last 250 sessions        (descriptor)
  abs_change               |latest log change|                          (descriptor)
  staleness                sessions since the latest period end         (descriptor)
Pledge C (15-session lag):
  ratio_level              pledge_ratio (%)                             (-)
  ratio_to_float           pledged shares / float shares                (-)
  count_activity           log((count+1)/(13w median+1))                (-)
  ratio_innovation_52w     ratio vs own 52w median / sd                 (-)
  pressure_price_drawdown  (ratio/100) * max(0, 1 - close/52w high)     (-)
  ratio_x_holder_disp      cs-z(lagged ratio) * cs-z(holder change)     (-)
Pledge GC:
  rest_share               restricted / total pledged shares            (descriptor)
  count_per_pledged        pledge_count / pledged shares (万股)         (descriptor)
  ratio_volatility_26w     sd of weekly ratio changes                   (descriptor)
  staleness                sessions since the latest period end         (descriptor)
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd

WARMUP_DAYS = 400  # holder seasonal history (3 years) + pledge 52 weeks

HN = "stock_holder_number"
PL = "stock_pledge_stat"
DAILY = "stock_daily"
FINANCE = "stock_finance"
ADJ = "stock_adj_factor"

PLEDGE_LAG = 15  # documented conservative lower bound; see module docstring

_C = "_r17own1_cache"

C_VARIANTS = (
    "hn_change_accel",
    "hn_seasonal_innovation",
    "hn_accumulation_divergence",
    "hn_delay_own_history",
    "hn_change_volatility",
    "pl_ratio_level",
    "pl_ratio_to_float",
    "pl_count_activity",
    "pl_ratio_innovation_52w",
    "pl_pressure_price_drawdown",
    "pl_ratio_x_holder_disp",
)
GC_VARIANTS = (
    "hn_log_level",
    "hn_avg_holding",
    "hn_report_count_250",
    "hn_abs_change",
    "hn_staleness",
    "pl_rest_share",
    "pl_count_per_pledged",
    "pl_ratio_volatility_26w",
    "pl_staleness",
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


def _dates_from_ymd(s):
    """YYYYMMDD (int/float, NaN allowed) -> datetime64 series."""
    return pd.to_datetime(pd.Series(s).astype("Int64").astype(str),
                          format="%Y%m%d", errors="coerce")


def _row_index(ctx, codes):
    """Unique code -> column index, then expand to rows (avoids a 1M-row dict loop)."""
    uniq, inv = np.unique(codes, return_inverse=True)
    ci = np.asarray(ctx.code_index(uniq), dtype=np.int64)
    return ci[inv]


def _price_at_rows(ctx, codes, days, grid):
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    ci = _row_index(ctx, codes)
    d = np.asarray(days, dtype=np.int64)
    pos = np.searchsorted(dates, d, side="right") - 1
    ok = (pos >= 0) & (ci >= 0) & np.isfinite(d)
    p = np.where(ok, pos, 0)
    c = np.where(ok, ci, 0)
    out = grid[p, c]
    return np.where(ok, out, np.nan)


class _HN:
    """Holder-number period table (collapsed to one row per stock-month)."""

    def __init__(self, ctx):
        self.ctx = ctx
        d0, d1 = int(ctx.panel.dates[0]), int(ctx.panel.dates[-1])
        df = ctx.dataset(HN, columns=["stock_code", "ann_date", "end_date", "holder_num"],
                         years=(2015, d1 // 10000))
        if df is None or df.empty:
            raise RuntimeError("stock_holder_number returned no rows")
        df = df.copy()
        df["ann"] = _ymd(df["ann_date"])
        df["end"] = _ymd(df["end_date"])
        df["n"] = pd.to_numeric(df["holder_num"], errors="coerce")
        df = df[np.isfinite(df["ann"]) & np.isfinite(df["end"]) & (df["n"] > 0)].copy()
        df["ann"] = df["ann"].astype(np.int64)
        df["end"] = df["end"].astype(np.int64)
        df["end_month"] = df["end"] // 100
        df = df.sort_values(["stock_code", "end", "ann"], kind="stable")
        df = df.drop_duplicates(["stock_code", "end_month"], keep="last")
        df = df.sort_values(["stock_code", "end"], kind="stable").reset_index(drop=True)
        df["log_n"] = np.log(df["n"])
        g = df.groupby("stock_code", sort=False)
        df["prev_end"] = g["end"].shift(1)
        df["prev_log_n"] = df["log_n"].shift(1)
        dt = (_dates_from_ymd(df["end"]) - _dates_from_ymd(df["prev_end"])).dt.days
        df["dt"] = dt.astype("float64")
        valid = (df["dt"] > 0) & (df["dt"] <= 200)
        df["g"] = np.where(valid, df["log_n"] - df["prev_log_n"], np.nan)
        df["delay"] = (_dates_from_ymd(df["ann"])
                       - _dates_from_ymd(df["end"])).dt.days.astype("float64")
        q = (df["end"] // 100) % 100
        df["q"] = ((q - 1) // 3 + 1).astype("int64")
        # same-quarter prior changes (exclude current), min 2 prior years
        gq = df.groupby(["stock_code", "q"], sort=False)["g"]
        mu = gq.transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
        sd = gq.transform(lambda s: s.shift(1).rolling(3, min_periods=2).std(ddof=0))
        df["seasonal_mu"] = mu
        df["seasonal_sd"] = sd
        # acceleration
        df["g_prev"] = df.groupby("stock_code", sort=False)["g"].shift(1)
        # volatility of last 6 changes
        df["g_vol"] = df.groupby("stock_code", sort=False)["g"].transform(
            lambda s: s.rolling(6, min_periods=3).std(ddof=0))
        # delay innovation vs own prior 8 disclosures
        df["delay_prior"] = df.groupby("stock_code", sort=False)["delay"].transform(
            lambda s: s.shift(1).rolling(8, min_periods=3).median())
        self.df = df
        del df
        for key in list(getattr(ctx.up, "_cache", {})):
            if key and key[0] == HN:
                ctx.up._cache.pop(key, None)
        gc.collect()
        self._grid_cache = {}

    def asof(self, frame, value):
        f = frame[["stock_code", "ann", "end", value]]
        f = f.sort_values(["stock_code", "ann", "end"], kind="stable")
        f = f.drop_duplicates(["stock_code", "ann"], keep="last")
        return np.asarray(self.ctx.asof_daily(
            f["stock_code"].to_numpy(), f["ann"].to_numpy(),
            f[value].to_numpy(np.float64)), dtype=np.float64)


class _PL:
    """Pledge table (weekly).  End-date as-of only; the 15-session lag is applied last."""

    def __init__(self, ctx):
        self.ctx = ctx
        d1 = int(ctx.panel.dates[-1])
        df = ctx.dataset(PL, columns=[
            "stock_code", "end_date", "pledge_count", "unrest_pledge",
            "rest_pledge", "total_share", "pledge_ratio"], years=(2014, d1 // 10000))
        if df is None or df.empty:
            raise RuntimeError("stock_pledge_stat returned no rows")
        df = df.copy()
        df["end"] = _ymd(df["end_date"])
        df = df[np.isfinite(df["end"])].copy()
        df["end"] = df["end"].astype(np.int64)
        for c in ("pledge_count", "unrest_pledge", "rest_pledge", "total_share", "pledge_ratio"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["pledged"] = df["unrest_pledge"] + df["rest_pledge"]  # 万股
        df["ratio"] = df["pledge_ratio"]  # percent
        df = df.sort_values(["stock_code", "end"], kind="stable").reset_index(drop=True)
        g = df.groupby("stock_code", sort=False)
        df["ratio_med52"] = g["ratio"].transform(
            lambda s: s.rolling(52, min_periods=26).median())
        df["ratio_sd52"] = g["ratio"].transform(
            lambda s: s.rolling(52, min_periods=26).std(ddof=0))
        df["count_med13"] = g["pledge_count"].transform(
            lambda s: s.rolling(13, min_periods=7).median())
        df["ratio_prev"] = g["ratio"].shift(1)
        df["ratio_vol26"] = g["ratio"].transform(
            lambda s: s.diff().rolling(26, min_periods=13).std(ddof=0))
        self.df = df
        del df
        for key in list(getattr(ctx.up, "_cache", {})):
            if key and key[0] == PL:
                ctx.up._cache.pop(key, None)
        gc.collect()


class _Own1:
    def __new__(cls, ctx):
        cache = ctx.__dict__.setdefault(_C, {})
        if "own" in cache:
            return cache["own"]
        self = super().__new__(cls)
        cache["own"] = self
        return self

    def __init__(self, ctx):
        if getattr(self, "_ready", False):
            return
        self.ctx = ctx
        tr = ctx.__dict__.get("_r17own1_traded")
        if tr is None:
            tr = ctx.traded()
            ctx.__dict__["_r17own1_traded"] = tr
        self.tr = tr
        self.hfq = ctx.hfq("close")
        self.fs = ctx.px("float_share")
        self.rmax = ctx.roll_max(self.hfq, 250, 200)
        self._hn = None
        self._pl = None
        self._ready = True

    @property
    def hn(self):
        if self._hn is None:
            self._hn = _HN(self.ctx)
        return self._hn

    @property
    def pl(self):
        if self._pl is None:
            self._pl = _PL(self.ctx)
        return self._pl

    def price_at(self, codes, days, grid):
        return _price_at_rows(self.ctx, codes, days, grid)


def _period_return(own, codes, end_prev, end_cur):
    p0 = own.price_at(codes, end_prev, own.hfq)
    p1 = own.price_at(codes, end_cur, own.hfq)
    return _ratio(p1 - p0, p0, 1e-6)


def _hn_grid(own, value):
    return own.hn.asof(own.hn.df, value)


def _pl_asof_frame(own, frame, value):
    f = frame[["stock_code", "end", value]].dropna(subset=[value])
    return np.asarray(own.ctx.asof_daily(
        f["stock_code"].to_numpy(), f["end"].to_numpy(),
        f[value].to_numpy(np.float64)), dtype=np.float64)


def _pl_asof(own, value):
    return _pl_asof_frame(own, own.pl.df, value)


def _compute(ctx, variant):
    own = _Own1(ctx)
    hn, pl = own.hn, own.pl
    dfh, dfp = hn.df, pl.df
    tr, uni = own.tr, ctx.universe

    if variant == "hn_change_accel":
        d = dfh.copy()
        d["value"] = d["g"] - d["g_prev"]
        out = hn.asof(d, "value")

    elif variant == "hn_seasonal_innovation":
        d = dfh.copy()
        d["value"] = _ratio(d["g"] - d["seasonal_mu"], d["seasonal_sd"], 1e-8)
        out = hn.asof(d, "value")

    elif variant == "hn_accumulation_divergence":
        d = dfh.copy()
        d["ret"] = _period_return(own, d["stock_code"].to_numpy(),
                                  d["prev_end"].to_numpy(), d["end"].to_numpy())
        g = hn.asof(d, "g")
        r = hn.asof(d, "ret")
        zc = ctx.cs_zscore(-g, mask=uni)
        zr = ctx.cs_zscore(r, mask=uni)
        out = zc - zr

    elif variant == "hn_delay_own_history":
        d = dfh.copy()
        d["value"] = d["delay"] - d["delay_prior"]
        out = hn.asof(d, "value")

    elif variant == "hn_change_volatility":
        out = hn.asof(dfh, "g_vol")

    # ---- holder GC ------------------------------------------------------
    elif variant == "hn_log_level":
        d = dfh.copy()
        d["value"] = d["log_n"]
        out = hn.asof(d, "value")

    elif variant == "hn_avg_holding":
        n = hn.asof(dfh, "n")
        out = np.where(tr, ctx.safe_log(_ratio(own.fs, n, 1.0)), np.nan)

    elif variant == "hn_report_count_250":
        ev = np.asarray(ctx.event_grid(
            dfh["stock_code"].to_numpy(), dfh["ann"].to_numpy(),
            np.ones(len(dfh), dtype=np.float64)), dtype=np.float64)
        out = ctx.roll_sum(ev, 250, None)

    elif variant == "hn_abs_change":
        d = dfh.copy()
        d["value"] = np.abs(d["g"])
        out = hn.asof(d, "value")

    elif variant == "hn_staleness":
        d = dfh.copy()
        d["value"] = d["end"].astype(np.float64)
        end_grid = hn.asof(d, "value")
        dates = np.asarray(ctx.panel.dates, dtype=np.int64)
        pos = np.searchsorted(dates, end_grid)
        row = np.arange(dates.size, dtype=np.int64)[:, None]
        out = np.where(np.isfinite(end_grid), (row - pos).astype(np.float64), np.nan)

    # ---- pledge C -------------------------------------------------------
    elif variant == "pl_ratio_level":
        out = _pl_asof(own, "ratio")

    elif variant == "pl_ratio_to_float":
        fs = own.price_at(dfp["stock_code"].to_numpy(), dfp["end"].to_numpy(), own.fs)
        d = dfp.copy()
        d["value"] = _ratio(d["pledged"].to_numpy() * 1e4, fs, 1e6)
        out = _pl_asof_frame(own, d, "value")

    elif variant == "pl_count_activity":
        d = dfp.copy()
        d["value"] = np.log((d["pledge_count"] + 1.0) / (d["count_med13"] + 1.0))
        out = _pl_asof_frame(own, d, "value")

    elif variant == "pl_ratio_innovation_52w":
        d = dfp.copy()
        d["value"] = _ratio(d["ratio"] - d["ratio_med52"], d["ratio_sd52"], 1e-6)
        out = _pl_asof_frame(own, d, "value")

    elif variant == "pl_pressure_price_drawdown":
        px = own.price_at(dfp["stock_code"].to_numpy(), dfp["end"].to_numpy(), own.hfq)
        hi = own.price_at(dfp["stock_code"].to_numpy(), dfp["end"].to_numpy(), own.rmax)
        dd = np.maximum(0.0, 1.0 - _ratio(px, hi, 1e-6))
        d = dfp.copy()
        d["value"] = d["ratio"] / 100.0 * dd
        out = _pl_asof_frame(own, d, "value")

    elif variant == "pl_ratio_x_holder_disp":
        # Only the pledge leg is lagged; the holder leg uses its own ann_date.
        ratio = _pl_asof(own, "ratio")
        ratio_lag = ctx.lag_grid(ratio, PLEDGE_LAG)
        hd = dfh.copy()
        hd["value"] = hd["g"]
        holder_g = hn.asof(hd, "value")
        z1 = ctx.cs_zscore(ratio_lag, mask=uni)
        z2 = ctx.cs_zscore(holder_g, mask=uni)
        out = z1 * z2

    # ---- pledge GC ------------------------------------------------------
    elif variant == "pl_rest_share":
        d = dfp.copy()
        d["value"] = _ratio(d["rest_pledge"], d["pledged"], 1e-6)
        out = _pl_asof_frame(own, d, "value")

    elif variant == "pl_count_per_pledged":
        d = dfp.copy()
        d["value"] = _ratio(d["pledge_count"], d["pledged"], 1e-6)
        out = _pl_asof_frame(own, d, "value")

    elif variant == "pl_ratio_volatility_26w":
        out = _pl_asof(own, "ratio_vol26")

    elif variant == "pl_staleness":
        d = dfp.copy()
        d["value"] = d["end"].astype(np.float64)
        end_grid = _pl_asof_frame(own, d, "value")
        end_lag = ctx.lag_grid(end_grid, PLEDGE_LAG)
        dates = np.asarray(ctx.panel.dates, dtype=np.int64)
        pos = np.searchsorted(dates, end_lag)
        row = np.arange(dates.size, dtype=np.int64)[:, None]
        out = np.where(np.isfinite(end_lag), (row - pos).astype(np.float64), np.nan)

    else:
        raise KeyError(f"unknown variant {variant!r}")

    # Pledge-based variants get the documented 15-session conservative lag.
    if variant.startswith("pl_") and variant != "pl_ratio_x_holder_disp":
        out = ctx.lag_grid(out, PLEDGE_LAG)
    return np.where(tr & uni, out, np.nan)


def build(parameters):
    variant = parameters["variant"]
    if variant not in PARAMETERS:
        raise KeyError(variant)

    def fn(ctx):
        return _compute(ctx, variant)

    return fn
