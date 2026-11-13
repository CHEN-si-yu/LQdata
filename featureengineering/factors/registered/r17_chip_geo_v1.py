"""R17 seat chip_event — batch 1: chip cost-distribution geometry (25 variants).

Sources
-------
* `stock_cyq_chips` via the pre-aggregated `fea.chips.ChipLayer` (ctx.chip):
  per (stock, trade_date) weighted quantiles / moments / concentration / peaks
  of the holder cost distribution.  The layer normalises each stock-day's
  `percent` weights by their sum (vendor sums are not exactly 100; p99 ~183 in
  recent years).  Unadjusted prices; therefore every price level enters only
  as `ln(level / close)` (same-day unadjusted close), an adjustment-invariant
  transform.
* `stock_cyq_perf` via exact daily placement (`stock_cyq_perf` rows are one per
  stock-day, 2018+): vendor `winner_rate` (0-100, NOT normalised) and
  `cost_5pct/cost_95pct/his_low/his_high` used for historical-range geometry.
* `stock_daily` close / traded; `stock_finance` float shares to build the
  project-standard `turnover` (percent) for the overhang-days variant.

Masks
-----
Every output is masked with `ctx.traded() & ctx.universe`; suspension days are
NaN, never zero.  Missing chip rows are NaN (ChipLayer.place semantics), not
forward-filled.  No zero fill anywhere.

Direction / horizons were frozen in the seat configs BEFORE any label access.
GC variants are descriptive states (no alpha direction claimed); C variants
carry an ex-ante direction and h5 primary horizon.

This builder is screening-only; formal registration/evaluation is the chief's.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WARMUP_DAYS = 120

_GC = (
    "chip_gini",
    "chip_cr3",
    "chip_skew",
    "chip_peak_gap",
    "peak_vs_top3",
    "mode_vs_median",
    "mid_tier_mass",
    "mean_gap_chg5",
    "mode_gap_chg20",
    "p90_gap_chg20",
    "skew_chg20",
    "n_peaks_chg20",
    "p90_hist_pos",
    "cost_range_vs_hist",
    "chip_n_levels",
    "cost_width_80",
)
_C = (
    "overhang_days",
    "winner_rate",
    "cost_p50_gap",
    "cost_p10_gap",
    "cost_p90_gap",
    "winner_rate_chg20",
    "gini_chg20",
    "chip_entropy",
    "chip_n_peaks",
)

PARAMETERS = {name: {"variant": name} for name in (*_GC, *_C)}

_PERF_COLUMNS = ("cost_5pct", "cost_95pct", "his_low", "his_high", "winner_rate")
_PERF_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _years(ctx) -> tuple[int, int]:
    return int(str(ctx.panel.dates[0])[:4]), int(str(ctx.panel.dates[-1])[:4])


def _perf(ctx) -> dict[str, np.ndarray]:
    """Exact daily placement of stock_cyq_perf fields (no forward fill).

    The table is dense (one row per stock-day since 2018) and is only ~1M
    rows/year; it is read once per (upstream, panel window) and shared by all
    variants of this family in the same process.
    """
    key = (id(ctx.up), int(ctx.panel.dates[0]), int(ctx.panel.dates[-1]), ctx.panel.C)
    hit = _PERF_CACHE.get(key)
    if hit is not None:
        return hit
    df = ctx.dataset(
        "stock_cyq_perf",
        columns=["trade_date", "stock_code", *_PERF_COLUMNS],
        years=_years(ctx),
    )
    if df is None or df.empty:
        raise RuntimeError("stock_cyq_perf unavailable")
    idx = ctx.code_index(df["stock_code"].to_numpy())
    keep = idx >= 0
    days = ctx.date_col(df["trade_date"])
    out: dict[str, np.ndarray] = {}
    for col in _PERF_COLUMNS:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
        out[col] = ctx.panel.place(
            idx[keep].astype(np.int64), days[keep], vals[keep]
        )
    _PERF_CACHE[key] = out
    if len(_PERF_CACHE) > 2:  # bounded like tradability.py; variants are sequential
        for old in list(_PERF_CACHE)[:-1]:
            _PERF_CACHE.pop(old, None)
    return out


def _mask(ctx, out: np.ndarray) -> np.ndarray:
    return np.where(ctx.traded() & ctx.universe, out, np.nan)


def _log_ratio(ctx, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """ln(num/den) with a positive-denominator guard; NaN propagates."""
    return ctx.safe_log(ctx.safe_div(num, den, min_abs_den=1e-8))


def _close(ctx) -> np.ndarray:
    return ctx.px("close")


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def build(parameters):
    variant = parameters["variant"]
    if variant not in PARAMETERS:
        raise KeyError(variant)

    def fn(ctx):
        # ---- chip-layer level / shape / concentration states ---------------
        if variant == "chip_gini":
            return _mask(ctx, ctx.chip("gini"))

        if variant == "chip_cr3":
            return _mask(ctx, ctx.chip("cr3"))

        if variant == "chip_skew":
            return _mask(ctx, ctx.chip("skew"))

        if variant == "chip_peak_gap":
            return _mask(ctx, ctx.chip("peak_gap"))

        if variant == "peak_vs_top3":
            pp = ctx.chip("peak_purity")
            cr3 = ctx.chip("cr3")
            return _mask(ctx, ctx.safe_div(pp, cr3, min_abs_den=1e-6))

        if variant == "mode_vs_median":
            return _mask(ctx, _log_ratio(ctx, ctx.chip("mode"), ctx.chip("p50")))

        if variant == "mid_tier_mass":
            return _mask(ctx, ctx.chip("top10_share") - ctx.chip("cr3"))

        if variant == "chip_n_levels":
            return _mask(ctx, ctx.chip("n_levels"))

        if variant == "cost_width_80":
            p10 = ctx.chip("p10")
            p90 = ctx.chip("p90")
            p50 = ctx.chip("p50")
            return _mask(ctx, ctx.safe_div(p90 - p10, p50, min_abs_den=1e-8))

        if variant == "chip_entropy":
            return _mask(ctx, ctx.chip("entropy"))

        if variant == "chip_n_peaks":
            return _mask(ctx, ctx.chip("n_peaks"))

        # ---- close-vs-cost level gaps (log, adjustment invariant) ----------
        if variant == "cost_p50_gap":
            return _mask(ctx, _log_ratio(ctx, ctx.chip("p50"), _close(ctx)))

        if variant == "cost_p10_gap":
            return _mask(ctx, _log_ratio(ctx, ctx.chip("p10"), _close(ctx)))

        if variant == "cost_p90_gap":
            return _mask(ctx, _log_ratio(ctx, ctx.chip("p90"), _close(ctx)))

        # ---- 5/20-day migration of distribution anchors --------------------
        if variant == "mean_gap_chg5":
            g = _log_ratio(ctx, ctx.chip("mean"), _close(ctx))
            return _mask(ctx, ctx.diff(g, 5))

        if variant == "mode_gap_chg20":
            g = _log_ratio(ctx, ctx.chip("mode"), _close(ctx))
            return _mask(ctx, ctx.diff(g, 20))

        if variant == "p90_gap_chg20":
            g = _log_ratio(ctx, ctx.chip("p90"), _close(ctx))
            return _mask(ctx, ctx.diff(g, 20))

        if variant == "skew_chg20":
            return _mask(ctx, ctx.diff(ctx.chip("skew"), 20))

        if variant == "n_peaks_chg20":
            return _mask(ctx, ctx.diff(ctx.chip("n_peaks"), 20))

        if variant == "gini_chg20":
            return _mask(ctx, ctx.diff(ctx.chip("gini"), 20))

        # ---- vendor perf: winner rate and historical-range geometry --------
        if variant == "winner_rate":
            return _mask(ctx, _perf(ctx)["winner_rate"])

        if variant == "winner_rate_chg20":
            return _mask(ctx, ctx.diff(_perf(ctx)["winner_rate"], 20))

        if variant == "p90_hist_pos":
            p = _perf(ctx)
            num = p["cost_95pct"] - p["his_low"]
            den = p["his_high"] - p["his_low"]
            return _mask(ctx, ctx.safe_div(num, den, min_abs_den=1e-8))

        if variant == "cost_range_vs_hist":
            p = _perf(ctx)
            num = p["cost_95pct"] - p["cost_5pct"]
            den = p["his_high"] - p["his_low"]
            return _mask(ctx, ctx.safe_div(num, den, min_abs_den=1e-8))

        # ---- overhead supply scaled by liquidity ---------------------------
        if variant == "overhang_days":
            above = ctx.chip("above_close")
            turn = ctx.px("turnover")                       # percent of float
            to20 = ctx.roll_mean(turn, 20, min_count=10) / 100.0
            return _mask(ctx, ctx.safe_div(above, to20, min_abs_den=0.001))

        raise KeyError(variant)

    return fn
