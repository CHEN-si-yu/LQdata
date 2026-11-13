"""R38 flow/value production factors.

Three independent mechanisms selected from the frozen flow_value research batch:

* ``fv_netflow_to_float_mv_20``  -- 20-session vendor net inflow / current float
  market value (daily flow absorption relative to tradable supply).
* ``fv_pe_pct_repair_60``        -- 60-session change in the stock's own 252-slot
  PE percentile (valuation repair speed).
* ``fv_dividend_stability_252``  -- coefficient of variation of daily ``dv_ttm``
  over 252 sessions (dividend-policy stability).

All inputs are daily post-close snapshots.  The formulas reproduce the frozen
research cards exactly; no label, return or evaluation file is read here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = (
    "fv_netflow_to_float_mv_20",
    "fv_pe_pct_repair_60",
    "fv_dividend_stability_252",
)

NETFLOW_WINDOW = 20
NETFLOW_MIN_VALID = 15
PE_WINDOW = 252
PE_MIN_VALID = 180
PE_REPAIR_LAG = 60
DIV_WINDOW = 252
DIV_MIN_VALID = 180


def _panel_from_long(ctx, frame, columns):
    """Place selected daily columns on the context panel (exact date/code match)."""
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    codes = np.asarray(ctx.panel.codes).astype(str)
    idx = np.asarray(ctx.code_index(frame["stock_code"].to_numpy()), dtype=np.int64)
    day = np.asarray(ctx.date_col(frame["trade_date"]), dtype=np.int64)
    keep = idx >= 0
    out = {}
    for col in columns:
        vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(np.float64)
        out[col] = ctx.panel.place(idx[keep], day[keep], vals[keep])
    return out


def _netflow_to_float_mv(ctx):
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    years = (int(dates[0]) // 10000, int(dates[-1]) // 10000)
    frame = ctx.dataset(
        "stock_main_fund_flow",
        columns=["trade_date", "stock_code", "net_mf_amount"],
        years=years,
    )
    if frame is None or frame.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)

    net = _panel_from_long(ctx, frame, ["net_mf_amount"])["net_mf_amount"]
    traded = ctx.traded()
    net = np.where(traded & np.isfinite(net), net, np.nan)
    cum = ctx.roll_sum(net, NETFLOW_WINDOW, min_count=NETFLOW_MIN_VALID) * 1e4

    close = ctx.px_raw("close")
    fshare = ctx.px_raw("float_share")
    fmv = close * fshare
    valid = traded & np.isfinite(fmv) & (fmv > 0.0) & np.isfinite(cum)
    return np.where(valid, cum / np.where(valid, fmv, 1.0), np.nan)


def _rolling_pe_percentile(ctx):
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    years = (int(dates[0]) // 10000, int(dates[-1]) // 10000)
    frame = ctx.dataset(
        "stock_finance",
        columns=["trade_date", "stock_code", "pe_ttm"],
        years=years,
    )
    if frame is None or frame.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)

    pe = _panel_from_long(ctx, frame, ["pe_ttm"])["pe_ttm"]
    # pandas rolling rank ignores NaN and uses average ties; the card's
    # (count(<)+0.5*count(==))/count is rank_pct - 0.5/count.
    df = pd.DataFrame(np.where(pe > 0.0, pe, np.nan))
    roll = df.rolling(PE_WINDOW, min_periods=PE_MIN_VALID)
    rank_pct = roll.rank(pct=True, method="average")
    count = roll.count()
    pct = rank_pct - 0.5 / count.where(count > 0)
    out = pct - pct.shift(PE_REPAIR_LAG)
    return out.to_numpy(np.float64)


def _dividend_stability(ctx):
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    years = (int(dates[0]) // 10000, int(dates[-1]) // 10000)
    frame = ctx.dataset(
        "stock_finance",
        columns=["trade_date", "stock_code", "dv_ttm"],
        years=years,
    )
    if frame is None or frame.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)

    dv = _panel_from_long(ctx, frame, ["dv_ttm"])["dv_ttm"]
    dv = np.where(np.isfinite(dv) & (dv >= 0.0), dv, np.nan)
    df = pd.DataFrame(dv)
    roll = df.rolling(DIV_WINDOW, min_periods=DIV_MIN_VALID)
    mean = roll.mean()
    std = roll.std(ddof=1)
    out = std / mean.where(mean > 0.0)
    return out.to_numpy(np.float64)


def build(parameters):
    variant = parameters.get("variant")
    if variant not in NAMES:
        raise KeyError(f"unknown R38 flow/value variant: {variant!r}")

    def compute(ctx):
        if variant == "fv_netflow_to_float_mv_20":
            return _netflow_to_float_mv(ctx)
        if variant == "fv_pe_pct_repair_60":
            return _rolling_pe_percentile(ctx)
        return _dividend_stability(ctx)

    return compute
