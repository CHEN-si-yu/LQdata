"""R4-F：有效价差微结构（Roll 1984 / Corwin–Schultz 2012）——Lead 预登记 → 独立执行。

方向（预登记，B2 §1，看结果前固定）：两个核心机制与两个窗口变体均 higher_is_better=True
（有效价差大 = 流动性差 → 未来补偿收益高）。`roll_spread_20_nan` 是机制 1 的
**口径敏感性对照**（cov≥0 取 NaN 而非 0），非新机制主张，只用于报告敏感性。

估计量（只用信号日收盘及以前数据；hfq 复权因子截至当日，PIT 安全）：
  r_t       单日后复权简单收益（`ctx.ret_clean(1)`：traded 掩码 + 价格层单日
            |r|>60% 脏数据清洗）；并额外要求 t−1 也真实成交 —— 停牌日不入窗口。
            hfq_close 停牌日被前向填充，若不用掩码会把停牌伪装成 0 收益配对、
            系统性污染一阶自协方差（Roll 模型假设连续交易）。
  Roll      `2*sqrt(max(−cov(r_t, r_{t−1}), 0))`；cov 为尾部 W 个**配对**的滚动协方差
            （W=20, min_count=15；W=60, min_count=45 —— 均取 75% 窗口）。
            cov≥0 的处置**事前定死**：主候选（roll_spread_20/60）取 0.0
            （「未检出价差」边界，与 CS 论文的截零惯例一致）；敏感性变体
            （roll_spread_20_nan）取 NaN（「不可估计」）。两组如实并报。
  CS        Corwin–Schultz(2012)：相邻两日 high/low 重叠的隐含价差。
            beta  = ln(H_a/L_a)² + ln(H_b/L_b)²；
            gamma = ln(max(H_a,H_b)/min(L_a,L_b))²；
            alpha = (√(2β)−√β)/(3−2√2) − √(γ/(3−2√2))；pair = 2(e^α−1)/(1+e^α)。
            负估计截 0（论文口径）。日度序列 = 0.5·(pair_t + pair_{t−1})：
            论文 S̄_t 取覆盖第 t 日的两个两日估计 [f(t−1,t)+f(t,t+1)]/2 —— 其中
            f(t,t+1) 要用 t+1（未来）；本实现用 [f(t−2,t−1)+f(t−1,t)]/2，
            等价于把论文级数**整体后移一日**（不使用未来），在 t 收盘可得。
            窗口均值 W=20/60，min_count=15/45。
            停牌日 high/low 被前向填充 → 必须用 traded 掩码挡掉（否则停牌日
            伪装成 H=L 的零振幅日）。相邻两日都必须真实成交，pair 才有效；
            日度序列进一步要求相邻两个 pair 都有效（严格均值，缺一即 NaN）。

数据：stock_daily + stock_adj_factor（经 PriceLayer 的 hfq 口径）；只通过 ctx 读。
不读前复权（QFQ）。本模块不注册生产因子（research 候选专用）。
"""
from __future__ import annotations

import numpy as np

from fea import mathx as mx

#: 预登记信号标识（candidate.yaml 的 parameters 必须精确落在集合内）
SIGNALS = (
    "roll_spread_20",
    "corwin_schultz_spread_20",
    "roll_spread_60",
    "corwin_schultz_spread_60",
    "roll_spread_20_nan",
)
_ROLL_WINDOW = {"roll_spread_20": 20, "roll_spread_60": 60, "roll_spread_20_nan": 20}
_CS_WINDOW = {"corwin_schultz_spread_20": 20, "corwin_schultz_spread_60": 60}
_MIN_COUNT = {20: 15, 60: 45}          # 75% 窗口；事前定死
_CS_CONST = 3.0 - 2.0 * np.sqrt(2.0)   # (√2−1)² ≈ 0.171573


def _shift_bool(x: np.ndarray) -> np.ndarray:
    """沿交易日轴下移一行（上一日），布尔版；首行 False。"""
    out = np.zeros_like(x, dtype=bool)
    out[1:] = x[:-1]
    return out


def _roll_spread(ctx, window: int, cov_nonneg_nan: bool) -> np.ndarray:
    traded = np.asarray(ctx.traded(), dtype=bool)
    raw = np.asarray(ctx.ret_clean(1), dtype=np.float64)
    prev_traded = _shift_bool(traded)
    r = np.where(traded & prev_traded & np.isfinite(raw), raw, np.nan)
    cov = mx.roll_cov(r, mx.shift(r, 1), window, min_count=_MIN_COUNT[window])
    with np.errstate(all="ignore"):
        spread = 2.0 * np.sqrt(np.maximum(-cov, 0.0))
    if cov_nonneg_nan:
        return np.where(cov < 0.0, spread, np.nan)
    return np.where(cov >= 0.0, 0.0, spread)


def _cs_daily(ctx) -> np.ndarray:
    traded = np.asarray(ctx.traded(), dtype=bool)
    high = np.asarray(ctx.hfq("high"), dtype=np.float64)
    low = np.asarray(ctx.hfq("low"), dtype=np.float64)
    valid = traded & np.isfinite(high) & np.isfinite(low) & (high > 0.0) & (high >= low)
    h = np.where(valid, high, np.nan)
    l = np.where(valid, low, np.nan)
    h0, l0 = mx.shift(h, 1), mx.shift(l, 1)
    both = valid & _shift_bool(valid)
    with np.errstate(all="ignore"):
        beta = np.log(h / l) ** 2 + np.log(h0 / l0) ** 2
        gamma = np.log(np.maximum(h, h0) / np.minimum(l, l0)) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_CONST - np.sqrt(gamma / _CS_CONST)
        pair = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    pair = np.where(both & np.isfinite(pair), pair, np.nan)
    pair = np.maximum(pair, 0.0)                 # 论文口径：负估计截 0（NaN 保持 NaN）
    return 0.5 * (pair + mx.shift(pair, 1))      # 因果化日度序列（论文级数滞后一日）


def build(parameters):
    """Family 只允许预登记的五个信号标识；不搜索窗口、不搜索方向。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("R4-F 只允许预登记的五个信号标识")
    signal = parameters["signal"]

    def compute(ctx):
        if signal in _ROLL_WINDOW:
            value = _roll_spread(ctx, _ROLL_WINDOW[signal], cov_nonneg_nan=signal.endswith("_nan"))
        else:
            daily = _cs_daily(ctx)
            window = _CS_WINDOW[signal]
            value = mx.roll_mean(daily, window, min_count=_MIN_COUNT[window])
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
