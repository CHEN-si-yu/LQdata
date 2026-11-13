"""R24 calendar-gap sensitivity of idiosyncratic overnight variance.

Measures how market-adjusted stock opening-gap residual variance changes with
the full natural-day closure interval. This is a descriptive risk association,
not a claim of risk compensation or causality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


NAME = "calendar_gap_idio_variance_slope_60"
WINDOW = 60
MIN_PAIRS = 45
MIN_MARKET_VARIANCE = 1e-8
MIN_CLOSED_DAY_VARIANCE = 1e-8
MIN_RESIDUAL_VARIANCE = 1e-8
MARKET_CODE = "000300.SH"


def calendar_gap_idio_variance_slope_60(
    stock_gap: np.ndarray,
    market_gap: np.ndarray,
    closed_calendar_days: np.ndarray,
) -> np.ndarray:
    """Frozen 60-session normalized slope; keeps missing sessions in place."""
    y = np.asarray(stock_gap, dtype=np.float64)
    market = np.asarray(market_gap, dtype=np.float64)
    closed = np.asarray(closed_calendar_days, dtype=np.float64)
    if y.ndim != 2 or market.shape != (len(y),) or closed.shape != (len(y),):
        raise ValueError("stock_gap must be (T,C); market/closed days must be (T,)")
    t_count, n_codes = y.shape
    out = np.full_like(y, np.nan)
    for t in range(WINDOW - 1, t_count):
        sl = slice(t - WINDOW + 1, t + 1)
        yy, mm, cc = y[sl], market[sl], closed[sl]
        good = np.isfinite(yy) & np.isfinite(mm[:, None]) & np.isfinite(cc[:, None])
        n = good.sum(axis=0)
        use = n >= MIN_PAIRS
        if not use.any():
            continue
        nn = np.maximum(n, 1)
        xx = np.where(good, mm[:, None], 0.0)
        yy0 = np.where(good, yy, 0.0)
        cc0 = np.where(good, cc[:, None], 0.0)
        xm = xx.sum(axis=0) / nn
        ym = yy0.sum(axis=0) / nn
        cm = cc0.sum(axis=0) / nn
        xc = np.where(good, xx - xm, 0.0)
        yc = np.where(good, yy0 - ym, 0.0)
        cc_centered = np.where(good, cc0 - cm, 0.0)
        market_ss = np.square(xc).sum(axis=0)
        beta = np.divide((xc * yc).sum(axis=0), market_ss,
                         out=np.zeros(n_codes), where=market_ss > 0)
        eps = np.where(good, yc - beta[None, :] * xc, 0.0)
        eps2 = eps * eps
        residual_var = eps2.sum(axis=0) / nn
        market_var = market_ss / nn
        closed_var = np.square(cc_centered).sum(axis=0) / nn
        covariance = (eps2 * cc_centered).sum(axis=0) / nn
        valid = (use & (market_var > MIN_MARKET_VARIANCE) &
                 (closed_var > MIN_CLOSED_DAY_VARIANCE) &
                 (residual_var > MIN_RESIDUAL_VARIANCE))
        out[t, valid] = covariance[valid] / (closed_var[valid] * residual_var[valid])
    return out


def _closed_calendar_days(ctx) -> np.ndarray:
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    sessions = np.asarray(ctx.cal.days, dtype=np.int32)
    pos = np.searchsorted(sessions, dates)
    if np.any(pos >= len(sessions)) or not np.array_equal(sessions[pos], dates):
        raise ValueError("panel dates are not a subset of the injected exchange calendar")
    previous_pos = pos - 1
    out = np.full(len(dates), np.nan, dtype=np.float64)
    valid = previous_pos >= 0
    if valid.any():
        current_dates = pd.to_datetime(dates[valid].astype(str), format="%Y%m%d")
        previous_dates = pd.to_datetime(sessions[previous_pos[valid]].astype(str), format="%Y%m%d")
        out[valid] = (current_dates - previous_dates).days.to_numpy(dtype=np.float64) - 1.0
    return out


def _stock_open_gaps(ctx) -> np.ndarray:
    open_hfq = np.asarray(ctx.hfq("open"), dtype=np.float64)
    close_hfq = np.asarray(ctx.hfq("close"), dtype=np.float64)
    traded = np.asarray(ctx.traded(), dtype=bool)
    previous_close = np.full_like(close_hfq, np.nan)
    previous_traded = np.zeros_like(traded)
    if len(traded) > 1:
        previous_close[1:] = close_hfq[:-1]
        previous_traded[1:] = traded[:-1]
    valid = (traded & previous_traded & np.isfinite(open_hfq) & (open_hfq > 0) &
             np.isfinite(previous_close) & (previous_close > 0))
    ratio = np.full_like(open_hfq, np.nan)
    np.divide(open_hfq, previous_close, out=ratio, where=valid)
    out = np.full_like(open_hfq, np.nan)
    good = valid & np.isfinite(ratio) & (ratio > 0)
    out[good] = np.log(ratio[good])
    return out


def _market_open_gaps(ctx) -> np.ndarray:
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    if dates.size == 0:
        return np.full(0, np.nan, dtype=np.float64)
    years = (int(dates[0]) // 10000, int(dates[-1]) // 10000)
    frame = ctx.dataset("index_daily", columns=["ts_code", "trade_date", "open", "close"], years=years)
    open_px = np.full(len(dates), np.nan, dtype=np.float64)
    close_px = np.full(len(dates), np.nan, dtype=np.float64)
    if not frame.empty:
        frame = frame.loc[frame.ts_code.astype(str) == MARKET_CODE]
        if frame.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError(f"duplicate {MARKET_CODE} index_daily date")
        days = np.asarray(ctx.date_col(frame.trade_date), dtype=np.int32)
        dpos = {int(day): i for i, day in enumerate(dates)}
        ti = pd.Series(days).map(dpos).to_numpy()
        keep = pd.notna(ti)
        if keep.any():
            open_values = pd.to_numeric(frame.open, errors="coerce").to_numpy(np.float64)
            close_values = pd.to_numeric(frame.close, errors="coerce").to_numpy(np.float64)
            open_px[ti[keep].astype(np.int64)] = open_values[keep]
            close_px[ti[keep].astype(np.int64)] = close_values[keep]
    out = np.full(len(dates), np.nan, dtype=np.float64)
    if len(dates) > 1:
        valid = (np.isfinite(open_px[1:]) & (open_px[1:] > 0) &
                 np.isfinite(close_px[:-1]) & (close_px[:-1] > 0))
        ratio = open_px[1:] / close_px[:-1]
        valid &= np.isfinite(ratio) & (ratio > 0)
        out[1:][valid] = np.log(ratio[valid])
    return out


def build(parameters):
    if (not isinstance(parameters, dict) or set(parameters) != {"name"} or
            parameters.get("name") != NAME):
        raise ValueError(f"name must be {NAME!r}; got {parameters!r}")

    def compute(ctx):
        stock_gap = _stock_open_gaps(ctx)
        market_gap = _market_open_gaps(ctx)
        closed_days = _closed_calendar_days(ctx)
        value = calendar_gap_idio_variance_slope_60(stock_gap, market_gap, closed_days)
        return np.where(np.asarray(ctx.universe, dtype=bool), value, np.nan)

    return compute
