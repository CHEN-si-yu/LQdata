"""Round 4 批 1 · U4 —— 股票信号 × 市场状态交互（耦合）。

预登记来源：`artifacts/research/round4_20261004/RESEARCH_BRIEF_B1.md` §4-U4。
机制集合（方向冻结为 true，见各候选 yaml）：rev5_x_mktvol / mom120_x_trendstate /
lowvol_x_mktvol / illiq_x_dispstate。本模块只有这四个机制，无搜索网格。

冻结口径（任务书事前定死，本实现逐字执行）：
  · 股票腿 = 截面 winsor（cfg.winsor = 1/99）+ 截面 zscore；
  · 市场状态 = 252 日滚动 z（min_periods=120，禁全样本标准化）；
  · 交互值 = 两腿乘积（缺失传播，不零填充）。

关于市场腿的读取路径（如实披露，非最终结论）：
  任务书要求把市场正式因子写进 `deps`（runner 会一并复算）。但框架**不允许**
  股票因子通过 `ctx.load_factor` 读市场标量：`fea/factors_io.py:62` 对
  `spec.is_market=True` 的父因子直接抛「市场标量不能通过股票父因子接口读取」，
  且耦合读取器只扫 `cfg.factors_dir`（市场因子落在 `data/market_factors/`）。
  因此市场腿在候选内部**调用正式市场因子的同一实现函数**
  （`factors.market._base` / `factors.market._index`，与生产因子逐行同源）
  在当前面板上复算，而不是从盘上读值。父因子名仍按任务书进 `deps`：
  一让 runner 真实复算它们（进 run 证据），二让交付侧能把「候选内部复算值」
  与「run 命名空间里父因子实际落盘值」做逐值对照（见 delivery/evidence）。

股票腿的实现是对四个正式单腿因子的**逐行复刻**（同文件同函数语义）：
  rev5   ← factors/momentum.py: short_term_reversal_5 = −_ret_k(ctx, 5)
  mom120 ← factors/momentum.py: momentum_120 = _ret_k(ctx, 120)
  lowvol ← factors/volatility.py: vol_120 = roll_std(_ret(ctx), 120, min_count=60)
  illiq  ← factors/liquidity.py: amihud_daily_5 = roll_mean(_amihud(ctx), 5, min_count=2)
"""
from __future__ import annotations

import numpy as np

from fea import mathx as mx

SIGNALS = ("rev5_x_mktvol", "mom120_x_trendstate", "lowvol_x_mktvol", "illiq_x_dispstate")

#: 机制 → 正式市场父因子名（进 deps，runner 会一并复算）
MARKET_PARENT = {
    "rev5_x_mktvol": "mkt_idx_csi300_volatility20",
    "mom120_x_trendstate": "mkt_above_ma60_ratio",
    "lowvol_x_mktvol": "mkt_idx_csi300_volatility20",
    "illiq_x_dispstate": "mkt_dispersion",
}
#: 机制 → 在 factors.market 输出字典里的键
MARKET_KEY = {
    "rev5_x_mktvol": "csi300_volatility20",
    "mom120_x_trendstate": "above_ma60_ratio",
    "lowvol_x_mktvol": "csi300_volatility20",
    "illiq_x_dispstate": "dispersion",
}

STATE_WINDOW, STATE_MIN_PERIODS = 252, 120


# ══════════════════════════════════════════════════════════════════════════
# 股票腿 —— 正式单腿因子的逐行复刻（注释给出出处文件）
# ══════════════════════════════════════════════════════════════════════════

def _ret_traded(ctx) -> np.ndarray:
    """factors/volatility.py `_ret`（= factors/liquidity.py `_ret`）：停牌日 NaN。"""
    return np.where(ctx.traded(), ctx.ret(1), np.nan)


def _ret_k(ctx, k: int) -> np.ndarray:
    """factors/momentum.py `_ret_k` + `_zero_trade_mask`：k 日累计收益，
    单日脏数据毒化整窗；窗口内零成交 ⇒ NaN。"""
    r1 = ctx.ret(1)
    r = np.expm1(ctx.roll_sum(np.log1p(r1), int(k)))
    zero_trade = ctx.roll_sum(np.asarray(ctx.traded(), dtype=np.float64), int(k)) <= 0.0
    return np.where(zero_trade, np.nan, r)


def _leg_rev5(ctx) -> np.ndarray:
    """short_term_reversal_5 = −momentum_5（值大 = 近 5 日超跌更多）。"""
    return -_ret_k(ctx, 5)


def _leg_mom120(ctx) -> np.ndarray:
    """momentum_120 = _ret_k(ctx, 120)（值大 = 半年趋势更强）。"""
    return _ret_k(ctx, 120)


def _leg_lowvol(ctx) -> np.ndarray:
    """−vol_120 = −roll_std(_ret_traded, 120, min_count=60)（值大 = 低波动）。"""
    return -ctx.roll_std(_ret_traded(ctx), 120, min_count=60)


def _leg_illiq(ctx) -> np.ndarray:
    """amihud_daily_5 = roll_mean(|ret|/amount, 5, min_count=2)（值大 = 更不流动）。"""
    amihud = ctx.safe_div(np.abs(_ret_traded(ctx)), ctx.px("amount"), min_abs_den=1.0)
    return ctx.roll_mean(amihud, 5, min_count=2)


STOCK_LEG = {
    "rev5_x_mktvol": _leg_rev5,
    "mom120_x_trendstate": _leg_mom120,
    "lowvol_x_mktvol": _leg_lowvol,
    "illiq_x_dispstate": _leg_illiq,
}


# ══════════════════════════════════════════════════════════════════════════
# 市场状态腿 —— 调用正式市场因子的同一实现（factors/market.py），当前面板复算
# ══════════════════════════════════════════════════════════════════════════

def _market_series(ctx, signal: str) -> np.ndarray:
    """正式市场父因子在**当前面板**上的 (T,) 取值。

    调用 `factors.market._base`（池内横截面标量：above_ma60_ratio / dispersion）
    或 `factors.market._index`（指数标量：csi300_volatility20）——与生产因子
    完全同一函数，无转写。读取的数据源都由 deps 声明（stock_daily /
    stock_adj_factor / index_daily），`ctx.accessed` 由本候选的 deps 覆盖。
    """
    from factors.market import _base, _index
    key = MARKET_KEY[signal]
    source = _index(ctx) if key == "csi300_volatility20" else _base(ctx)
    return np.asarray(source[key], dtype=np.float64)


def _time_z(ctx, series: np.ndarray) -> np.ndarray:
    """252 日滚动 z（min_periods=120）：只用当日及以前的窗口，禁全样本标准化。

    `mx.roll_mean/roll_std(min_count=120)` 的毒化语义 = 窗口内至少 120 个有限值，
    均值/标准差按窗口内有效值计算（与 pandas rolling(252, min_periods=120) 同口径）。
    返回 (T, 1) 以便与股票腿广播相乘。
    """
    x = np.asarray(series, dtype=np.float64)[:, None]
    mu = mx.roll_mean(x, STATE_WINDOW, min_count=STATE_MIN_PERIODS)
    sd = mx.roll_std(x, STATE_WINDOW, min_count=STATE_MIN_PERIODS)
    with np.errstate(all="ignore"):
        z = (x - mu) / sd
    return np.where(np.isfinite(z), z, np.nan)


# ══════════════════════════════════════════════════════════════════════════
# 装配
# ══════════════════════════════════════════════════════════════════════════

def _standardize(ctx, leg: np.ndarray) -> np.ndarray:
    """股票腿：截面 winsor（cfg.winsor）+ 截面 zscore，mask=固定池。"""
    lo, hi = (float(ctx.cfg.winsor[0]), float(ctx.cfg.winsor[1]))
    clipped = ctx.cs_winsor(np.asarray(leg, dtype=np.float64), lo, hi, mask=ctx.universe)
    return np.asarray(ctx.cs_zscore(clipped, mask=ctx.universe), dtype=np.float64)


def build(parameters):
    """唯一入口：四选一的预登记机制，无窗口/方向搜索。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("U4 permits only the four preregistered interaction signals")
    signal = parameters["signal"]

    def compute(ctx):
        leg = STOCK_LEG[signal](ctx)
        state = _time_z(ctx, _market_series(ctx, signal))
        value = _standardize(ctx, leg) * state
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
