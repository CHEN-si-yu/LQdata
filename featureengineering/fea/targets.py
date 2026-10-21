"""共享目标合同。未来成交掩码只供离线评价，不能作为信号日输入。"""
from __future__ import annotations
import numpy as np

HORIZONS = (1, 3, 5, 10, 20)
LABEL_CONTRACT = "open_hfq_t+1_to_t+1+h_traded_endpoints_v3"


def open_to_open_labels(hfq_open, traded, horizon: int) -> np.ndarray:
    """T+1 开盘买、T+1+h 开盘卖；两端成交、价格有限，尾部未成熟保持 NaN。"""
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be positive")
    prices, trade = np.asarray(hfq_open, dtype=np.float64), np.asarray(traded, dtype=bool)
    if prices.ndim != 2 or trade.shape != prices.shape:
        raise ValueError("prices/traded must have equal (T,C) shapes")
    result = np.full(prices.shape, np.nan, dtype=np.float64)
    n = prices.shape[0] - h - 1
    if n <= 0:
        return result
    buy, sell = prices[1:n+1], prices[h+1:h+1+n]
    valid = (trade[1:n+1] & trade[h+1:h+1+n] & np.isfinite(buy)
             & np.isfinite(sell) & (np.abs(buy) > 1e-12))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw = sell / buy - 1.0
    result[:n] = np.where(valid & np.isfinite(raw), raw, np.nan)
    return result


def last_mature_signal_date(dates, traded, horizon: int):
    """市场末端成熟界限，不代表每只个股的端点可评价。"""
    dates, trade = np.asarray(dates), np.asarray(traded, dtype=bool)
    if trade.ndim != 2 or trade.shape[0] != dates.size:
        raise ValueError("dates/traded mismatch")
    observed = np.flatnonzero(trade.any(axis=1))
    pos = int(observed[-1]) - int(horizon) - 1 if observed.size else -1
    return dates[pos].item() if pos >= 0 else None
