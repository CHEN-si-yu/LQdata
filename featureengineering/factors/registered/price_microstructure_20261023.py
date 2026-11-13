"""Two frozen R24 price microstructure proxies admitted with weak evidence.

These implementations retain the R24 fixed-slot definitions.  In particular,
HFQ close is reconstructed from the exact day's raw close and adjustment factor;
missing values are not carried forward into either endpoint of a return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_DELAY = "price_delay_r2_60"
TICK_CLUSTERING = "raw_price_tick_clustering_20"
MARKET_CODE = "000300.SH"


def _price_delay_r2_60(stock_return: np.ndarray, market_return: np.ndarray,
                       window: int = 60, max_lag: int = 5,
                       min_pairs: int = 45,
                       minimum_full_r2: float = 1e-6) -> np.ndarray:
    """Frozen R24 nested-regression delay share on fixed exchange slots."""
    y = np.asarray(stock_return, dtype=np.float64)
    m = np.asarray(market_return, dtype=np.float64)
    if y.ndim != 2 or m.shape != (y.shape[0],):
        raise ValueError("stock_return must be (T,C); market_return must align on T")
    if window < 1 or max_lag < 1 or not 2 <= min_pairs <= window:
        raise ValueError("invalid rolling parameters")
    out = np.full_like(y, np.nan)
    t_count, c_count = y.shape
    for t in range(window + max_lag, t_count):
        slot_idx = np.arange(t - window, t)
        reg_idx = slot_idx[:, None] - np.arange(max_lag + 1)[None, :]
        market_x = m[reg_idx]
        base_good = np.all(np.isfinite(market_x), axis=1)
        y_window = y[slot_idx]
        good_matrix = np.isfinite(y_window) & base_good[:, None]
        groups: dict[bytes, tuple[np.ndarray, list[int]]] = {}
        for c in range(c_count):
            pattern = good_matrix[:, c]
            if int(pattern.sum()) < min_pairs:
                continue
            key = np.packbits(pattern).tobytes()
            if key not in groups:
                groups[key] = (pattern, [])
            groups[key][1].append(c)
        for pattern, columns in groups.values():
            cols = np.asarray(columns, dtype=np.int64)
            yy = y_window[pattern][:, cols]
            xx = market_x[pattern]
            centered = yy - yy.mean(axis=0, keepdims=True)
            sst = np.square(centered).sum(axis=0)
            valid_y = np.isfinite(sst) & (sst > 0)
            if not valid_y.any():
                continue
            yy = yy[:, valid_y]
            sst = sst[valid_y]
            full_design = np.column_stack((np.ones(len(yy)), xx))
            restricted_design = np.column_stack((np.ones(len(yy)), xx[:, 0]))
            full_fit = np.linalg.lstsq(full_design, yy, rcond=None)[0]
            restricted_fit = np.linalg.lstsq(restricted_design, yy, rcond=None)[0]
            full_resid = yy - full_design @ full_fit
            restricted_resid = yy - restricted_design @ restricted_fit
            full_r2 = 1.0 - np.square(full_resid).sum(axis=0) / sst
            restricted_r2 = 1.0 - np.square(restricted_resid).sum(axis=0) / sst
            keep = np.isfinite(full_r2) & (full_r2 > minimum_full_r2)
            output_cols = cols[valid_y]
            values = np.clip((full_r2 - restricted_r2) / full_r2, 0.0, 1.0)
            out[t, output_cols[keep]] = values[keep]
    return out


def _raw_price_tick_clustering_20(raw_close: np.ndarray, traded: np.ndarray,
                                  width: int = 20, min_count: int = 15) -> np.ndarray:
    """Frozen R24 excess frequency of traded raw closes on 0.05-yuan ticks."""
    close = np.asarray(raw_close, dtype=np.float64)
    is_traded = np.asarray(traded, dtype=bool)
    if close.ndim != 2 or is_traded.shape != close.shape:
        raise ValueError("raw_close and traded must share shape (T,C)")
    if width < 1 or not 1 <= min_count <= width:
        raise ValueError("invalid rolling parameters")
    out = np.full_like(close, np.nan)
    cents = close * 100.0
    aligned = (is_traded & np.isfinite(close) & (close > 0) & np.isfinite(cents) &
               (np.abs(cents - np.rint(cents)) <= 1e-4))
    cluster = np.zeros_like(close, dtype=np.float64)
    cluster[aligned] = np.isin(np.rint(cents[aligned]).astype(np.int64) % 10, (0, 5))
    for t in range(width, len(close)):
        sl = slice(t - width, t)
        ok = aligned[sl]
        count = ok.sum(axis=0)
        total = np.where(ok, cluster[sl], 0.0).sum(axis=0)
        valid_cols = count >= min_count
        out[t, valid_cols] = total[valid_cols] / count[valid_cols] - 0.20
    return out


def _market_close(ctx) -> np.ndarray:
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    result = np.full(len(dates), np.nan, dtype=np.float64)
    if not len(dates):
        return result
    frame = ctx.dataset("index_daily", columns=["ts_code", "trade_date", "close"],
                        years=(int(dates[0]) // 10000, int(dates[-1]) // 10000))
    if frame is None or frame.empty:
        return result
    frame = frame.loc[frame.ts_code.astype(str) == MARKET_CODE]
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError(f"duplicate {MARKET_CODE} index_daily date")
    days = np.asarray(ctx.date_col(frame.trade_date), dtype=np.int32)
    dpos = {int(day): i for i, day in enumerate(dates)}
    ti = pd.Series(days).map(dpos).to_numpy()
    keep = pd.notna(ti)
    if keep.any():
        values = pd.to_numeric(frame.close, errors="coerce").to_numpy(np.float64)
        result[ti[keep].astype(np.int64)] = values[keep]
    return result


def _source_panels(ctx):
    raw_close = np.asarray(ctx.px_raw("close"), dtype=np.float64)
    raw_volume = np.asarray(ctx.px_raw("vol"), dtype=np.float64)
    adj_factor = np.asarray(ctx.px_raw("adj_factor"), dtype=np.float64)
    traded = np.isfinite(raw_volume) & (raw_volume > 0)
    # Do not use ctx.hfq(): its daily level panel carries prices across gaps.
    hfq_close = np.full_like(raw_close, np.nan)
    valid = (traded & np.isfinite(raw_close) & (raw_close > 0) &
             np.isfinite(adj_factor) & (adj_factor > 0))
    hfq_close[valid] = raw_close[valid] * adj_factor[valid]
    stock_return = np.full_like(hfq_close, np.nan)
    if len(hfq_close) > 1:
        good = (traded[1:] & traded[:-1] & np.isfinite(hfq_close[1:]) &
                (hfq_close[1:] > 0) & np.isfinite(hfq_close[:-1]) &
                (hfq_close[:-1] > 0))
        ratio = hfq_close[1:] / hfq_close[:-1]
        good &= np.isfinite(ratio) & (ratio > 0)
        stock_return[1:][good] = ratio[good] - 1.0
    market_close = _market_close(ctx)
    market_return = np.full(len(market_close), np.nan, dtype=np.float64)
    if len(market_close) > 1:
        good = (np.isfinite(market_close[1:]) & (market_close[1:] > 0) &
                np.isfinite(market_close[:-1]) & (market_close[:-1] > 0))
        ratio = market_close[1:] / market_close[:-1]
        good &= np.isfinite(ratio) & (ratio > 0)
        market_return[1:][good] = ratio[good] - 1.0
    return raw_close, traded, stock_return, market_return


def build(parameters):
    if (not isinstance(parameters, dict) or set(parameters) != {"name"} or
            parameters.get("name") not in (PRICE_DELAY, TICK_CLUSTERING)):
        raise ValueError(f"name must be one of {(PRICE_DELAY, TICK_CLUSTERING)!r}; got {parameters!r}")
    name = parameters["name"]

    def compute(ctx):
        if name == PRICE_DELAY:
            raw_close, traded, stock_return, market_return = _source_panels(ctx)
            value = _price_delay_r2_60(stock_return, market_return)
        else:
            raw_close = np.asarray(ctx.px_raw("close"), dtype=np.float64)
            volume = np.asarray(ctx.px_raw("vol"), dtype=np.float64)
            traded = np.isfinite(volume) & (volume > 0)
            value = _raw_price_tick_clustering_20(raw_close, traded)
        return np.where(np.asarray(ctx.universe, dtype=bool), value, np.nan)

    return compute
