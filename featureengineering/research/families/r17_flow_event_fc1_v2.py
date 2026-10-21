"""R17 seat_flow_event batch 3 v2: forecast / earnings-event structure.

v2 is a pure engineering fix of v1: the per-context cache now carries `ctx`,
so variants after the first no longer lose the Context. Formulas, directions,
windows and missing semantics are byte-for-byte unchanged from v1; v1 stays
registered and its first-variant output is retained as the failed record.

Source
------
`stock_forecast` (2010 onward; irregular; `ann_date` is the availability date,
delay 0 per conf/frequency.yaml).  Columns used: ann_date, end_date, type,
p_change_min, p_change_max, net_profit_min, net_profit_max, last_parent_net,
first_ann_date.  Profit amounts are 万元, percentage changes are percent.
`total_assets`/`revenue` come from the strict PIT financial layer
(stock_balancesheet / stock_income), and market values use PIT close x shares.

PIT / missing semantics
-----------------------
Each row becomes an event at its `ann_date`; a value is forward-filled only
from that date (as-of), never backfilled.  Per-row history (delay innovation,
type persistence, width dispersion, entropy) uses only rows with an earlier
ann_date.  Missing p_change/net_profit rows are NaN, never zero.  A missing
forecast date is a confirmed "no forecast row", which is the correct zero only
for the event-count/decay variants.  No label is read.  Final mask is the
actual traded & universe mask on the evaluation day.

Batch 3 variants (14: 10 C + 4 GC)
----------------------------------
C (declared direction, primary horizon h5):
  width                  (p_change_max-p_change_min)/100, latest forecast  (-)
  mid_to_assets          forecast net-profit midpoint / total assets       (+)
  implied_earnings_yield midpoint / total market cap                       (+)
  type_transition        type score last - first for the same period       (+)
  delay_own_history      announcement delay minus own prior median         (-)
  width_narrowing        first width - last width for the same period      (+)
  event_decay            60d exponential decay of the type score           (+)
  mid_change_to_assets   (midpoint - last-year profit) / total assets      (+)
  forecast_margin        annualised midpoint / TTM revenue                 (+)
  type_persistence       mean type score over the last 4 report periods    (+)
GC (descriptive only):
  coverage_count_60      number of forecast rows in the last 60 sessions
  width_volatility       sd of the last 8 forecast widths
  type_entropy           entropy of the last 8 forecast type categories
  revision_count         number of revisions for the latest period
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd

WARMUP_DAYS = 260

FC = "stock_forecast"
DAILY = "stock_daily"
FINANCE = "stock_finance"
ADJ = "stock_adj_factor"
BS = "stock_balancesheet"
IS = "stock_income"

MIN20 = 10

# Frozen ordinal map: +2 strong positive, +1 mild positive, 0 unclear,
# -1 mild negative, -2 strong negative.  Unknown types are NaN (not zero).
TYPE_SCORE = {
    "预增": 2.0, "扭亏": 2.0,
    "略增": 1.0, "续盈": 1.0,
    "不确定": 0.0,
    "略减": -1.0,
    "预减": -2.0, "首亏": -2.0, "续亏": -2.0,
}
TYPE_LIST = tuple(TYPE_SCORE)

# Annualisation of a period forecast from its end_date month.
PERIOD_FRACTION = {3: 0.25, 6: 0.5, 9: 0.75, 12: 1.0}

_C = "_r17fc1_cache"

C_VARIANTS = (
    "width",
    "mid_to_assets",
    "implied_earnings_yield",
    "type_transition",
    "delay_own_history",
    "width_narrowing",
    "event_decay",
    "mid_change_to_assets",
    "forecast_margin",
    "type_persistence",
)
GC_VARIANTS = (
    "coverage_count_60",
    "width_volatility",
    "type_entropy",
    "revision_count",
)
PARAMETERS = {name: {"variant": name} for name in C_VARIANTS + GC_VARIANTS}


def _ratio(num, den, floor=1e-6):
    d = np.asarray(den, dtype=np.float64)
    n = np.asarray(num, dtype=np.float64)
    out = np.full(np.broadcast(n, d).shape, np.nan, dtype=np.float64)
    ok = np.isfinite(d) & (d > floor) & np.isfinite(n)
    np.divide(n, d, out=out, where=ok)
    return out


def _ymd(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        pd.to_datetime(s, errors="coerce").dt.strftime("%Y%m%d"),
        errors="coerce").astype("float64")


class _FC1:
    """Forecast table + per-row derived metrics, cached on the context."""

    def __init__(self, ctx):
        cache = ctx.__dict__.setdefault(_C, {})
        if "fc" in cache:
            self.__dict__.update(cache["fc"])
            return
        self.ctx = ctx
        tr = ctx.__dict__.get("_r17fc1_traded")
        if tr is None:
            tr = ctx.traded()
            ctx.__dict__["_r17fc1_traded"] = tr
        self.tr = tr
        d0, d1 = int(ctx.panel.dates[0]), int(ctx.panel.dates[-1])
        df = ctx.dataset(FC, columns=[
            "stock_code", "ann_date", "end_date", "type",
            "p_change_min", "p_change_max", "net_profit_min", "net_profit_max",
            "last_parent_net", "first_ann_date",
        ], years=(d0 // 10000, d1 // 10000))
        if df is None or df.empty:
            raise RuntimeError("stock_forecast returned no rows in the panel range")
        df = df.copy()
        df["ann"] = _ymd(df["ann_date"])
        df["end"] = _ymd(df["end_date"])
        df["first_ann"] = _ymd(df["first_ann_date"])
        for c in ("p_change_min", "p_change_max", "net_profit_min",
                  "net_profit_max", "last_parent_net"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["type"] = df["type"].astype(str)
        df = df[np.isfinite(df["ann"]) & np.isfinite(df["end"])].copy()
        df["ann"] = df["ann"].astype(np.int64)
        df["end"] = df["end"].astype(np.int64)
        df["score"] = df["type"].map(TYPE_SCORE).astype(np.float64)
        df["mid"] = 0.5 * (df["net_profit_min"] + df["net_profit_max"])
        pmax, pmin = df["p_change_max"], df["p_change_min"]
        df["width"] = np.where(pmax >= pmin, (pmax - pmin) / 100.0, np.nan)
        d_ann = pd.to_datetime(df["ann"].astype(str), format="%Y%m%d", errors="coerce")
        d_end = pd.to_datetime(df["end"].astype(str), format="%Y%m%d", errors="coerce")
        df["delay"] = (d_ann - d_end).dt.days.astype(np.float64)
        df["month"] = (df["end"] // 100) % 100
        df["fraction"] = df["month"].map(PERIOD_FRACTION)
        df = df.sort_values(["stock_code", "ann", "end"], kind="stable")
        self.df = df.reset_index(drop=True)
        del df
        for key in list(getattr(ctx.up, "_cache", {})):
            if key and key[0] == FC:
                ctx.up._cache.pop(key, None)
        gc.collect()
        cache["fc"] = {"df": self.df, "tr": tr, "ctx": ctx}
        ctx.__dict__[_C] = cache

    # ---- helpers --------------------------------------------------------
    def latest_per_day(self, frame, value, dropna=True):
        """One value per (stock, ann_date): the row with the largest end_date.

        `dropna=False` keeps a missing value as an explicit NaN at that
        announcement date, so the as-of grid clears an older finite value
        instead of silently carrying it forward.
        """
        f = frame[["stock_code", "ann", "end", value]]
        if dropna:
            f = f.dropna(subset=[value])
        f = f.sort_values(["stock_code", "ann", "end"], kind="stable")
        return f.drop_duplicates(["stock_code", "ann"], keep="last")

    def asof(self, frame, value):
        f = self.latest_per_day(frame, value, dropna=False)
        return ctx_asof(self.ctx, f["stock_code"].to_numpy(), f["ann"].to_numpy(),
                        f[value].to_numpy(np.float64))

    def event(self, frame, value):
        f = self.latest_per_day(frame, value, dropna=True)
        return ctx_event(self.ctx, f["stock_code"].to_numpy(), f["ann"].to_numpy(),
                         f[value].to_numpy(np.float64))


def ctx_asof(ctx, code, day, val):
    return np.asarray(ctx.asof_daily(code, day, val), dtype=np.float64)


def ctx_event(ctx, code, day, val):
    return np.asarray(ctx.event_grid(code, day, val), dtype=np.float64)


def _period_pairs(df, value):
    """first/last row per (stock, end_date) for a per-row value."""
    f = df[["stock_code", "end", "ann", value]].dropna(subset=[value])
    g = f.sort_values(["stock_code", "end", "ann"], kind="stable").groupby(
        ["stock_code", "end"], sort=False)
    first = g[value].first().rename("first")
    last = g[value].last().rename("last")
    last_ann = g["ann"].last().rename("ann")
    cnt = g[value].size().rename("count")
    out = pd.concat([first, last, last_ann, cnt], axis=1).reset_index()
    return out


def _compute(ctx, variant):
    f = _FC1(ctx)
    df = f.df
    tr = f.tr
    uni = ctx.universe

    if variant == "width":
        out = f.asof(df, "width")

    elif variant == "mid_to_assets":
        mid = f.asof(df, "mid") * 1e4
        out = _ratio(mid, ctx.point("total_assets"), 1e6)

    elif variant == "implied_earnings_yield":
        mid = f.asof(df, "mid") * 1e4
        mv = ctx.px("close") * ctx.px("total_share")
        out = np.where(tr, _ratio(mid, mv, 1e6), np.nan)

    elif variant == "type_transition":
        p = _period_pairs(df, "score")
        p["value"] = p["last"] - p["first"]
        p = p[np.isfinite(p["value"])]
        out = f.asof(p, "value")

    elif variant == "delay_own_history":
        g = df.sort_values(["stock_code", "ann"], kind="stable").groupby("stock_code")
        prior = g["delay"].transform(
            lambda s: s.shift(1).rolling(8, min_periods=3).median())
        d = df.copy()
        d["value"] = d["delay"] - prior
        out = f.asof(d, "value")

    elif variant == "width_narrowing":
        p = _period_pairs(df, "width")
        p["value"] = p["first"] - p["last"]
        p = p[np.isfinite(p["value"]) & (p["count"] >= 2)]
        out = f.asof(p, "value")

    elif variant == "event_decay":
        ev = f.event(df, "score")
        out = np.zeros(ctx.panel.shape, dtype=np.float64)
        for k in range(60):
            sh = ctx.shift(ev, k)
            out += np.where(np.isfinite(sh), sh, 0.0) * float(0.5 ** (k / 20.0))

    elif variant == "mid_change_to_assets":
        mid = f.asof(df, "mid") * 1e4
        last = f.asof(df, "last_parent_net") * 1e4
        out = _ratio(mid - last, ctx.point("total_assets"), 1e6)

    elif variant == "forecast_margin":
        mid = f.asof(df, "mid") * 1e4
        # Annualise the period midpoint, then scale by current TTM revenue.
        latest = f.latest_per_day(df, "mid").copy()
        latest["per"] = latest["end"] // 100 % 100
        latest["frac"] = latest["per"].map(PERIOD_FRACTION)
        latest["ann_mid"] = latest["mid"] / latest["frac"]
        ann_mid = f.asof(latest, "ann_mid") * 1e4
        out = _ratio(ann_mid, ctx.ttm("revenue"), 1e6)

    elif variant == "type_persistence":
        # Latest row per report period; mean score over the last 4 periods.
        f2 = df.dropna(subset=["score"]).sort_values(
            ["stock_code", "end", "ann"], kind="stable")
        f2 = f2.drop_duplicates(["stock_code", "end"], keep="last")
        f2 = f2.sort_values(["stock_code", "end"], kind="stable")
        f2["value"] = f2.groupby("stock_code")["score"].transform(
            lambda s: s.rolling(4, min_periods=2).mean())
        out = f.asof(f2, "value")

    # ---- GC descriptors -------------------------------------------------
    elif variant == "coverage_count_60":
        ones = np.ones(len(df), dtype=np.float64)
        ev = ctx_event(ctx, df["stock_code"].to_numpy(), df["ann"].to_numpy(), ones)
        # No row in the window is a confirmed zero count, not unknown.
        out = ctx.roll_sum(ev, 60, None)

    elif variant == "width_volatility":
        d = df.sort_values(["stock_code", "ann"], kind="stable").copy()
        d["value"] = d.groupby("stock_code")["width"].transform(
            lambda s: s.rolling(8, min_periods=3).std(ddof=0))
        out = f.asof(d, "value")

    elif variant == "type_entropy":
        d = df.sort_values(["stock_code", "ann"], kind="stable").copy()
        d = d.dropna(subset=["score"])
        ent = pd.Series(np.nan, index=d.index, dtype=np.float64)
        for t in TYPE_LIST:
            d[f"is_{t}"] = (d["type"] == t).astype(np.float64)
        cols = [f"is_{t}" for t in TYPE_LIST]
        rolled = d.groupby("stock_code")[cols].transform(
            lambda s: s.rolling(8, min_periods=3).mean())
        arr = rolled.to_numpy()
        valid = np.isfinite(arr).all(axis=1)
        with np.errstate(all="ignore"):
            p = np.where(arr > 0, arr, 1.0)
            h = -(arr * np.log(p)).sum(axis=1) / np.log(len(TYPE_LIST))
        d["value"] = np.where(valid, h, np.nan)
        out = f.asof(d, "value")

    elif variant == "revision_count":
        p = _period_pairs(df, "score")
        p["value"] = p["count"].astype(np.float64)
        out = f.asof(p, "value")

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
