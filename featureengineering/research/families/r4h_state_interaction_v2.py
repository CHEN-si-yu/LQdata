"""Round 4 批 2 · U8 —— 状态依赖交互 v2（第二批腿组合）。

预登记来源：`artifacts/research/round4_20261004/RESEARCH_BRIEF_B2.md` §3-U8。
机制集合（方向由任务书冻结，见各候选 yaml）：
  · rev5_x_illiqstate     = z_cs(−ret5)    × z_time(mkt_illiquidity_median)  [true]
  · mom120_x_dispstate    = z_cs(mom120)   × z_time(mkt_dispersion)          [false]
  · lowvol_x_breadthstate = z_cs(−vol_120) × z_time(mkt_above_ma60_ratio)    [true]
变体（事前登记为「窗口变体」，标注变体性质、不单列新机制主张）：
  · rev10_x_illiqstate    = z_cs(−ret10)   × z_time(mkt_illiquidity_median)  [true]
  · mom60_x_dispstate     = z_cs(mom60)    × z_time(mkt_dispersion)          [false]
本模块只有这五个机制，无窗口/方向搜索；与 U4(v1) 的腿组合无重复配对。

冻结口径（任务书事前定死，本实现逐字执行）：
  · 股票腿 = 截面 winsor（cfg.winsor = 1/99）+ 截面 zscore；
  · 市场状态 = 252 日滚动 z（min_periods=120，禁全样本标准化）；
  · 交互值 = 两腿乘积（缺失传播，不零填充）。

市场腿读取路径（如实披露）：框架不允许股票因子经 `ctx.load_factor` 读 is_market
标量（`fea/factors_io.py:62` 守卫：「市场标量不能通过股票父因子接口读取」），
市场腿在候选内部调用正式市场因子的**同一实现**（`factors.market._base`，与生产
逐行同源）在当前面板复算；父因子名仍按任务书进 `deps`，一让 runner 在 run 命名
空间内一并复算（供交付侧把「候选内部复算值」与「父因子落盘值」逐值对照），
二让水位失效传播成立。本单元三个父因子（illiquidity_median / dispersion /
above_ma60_ratio）全部来自 `_base`（固定池日频标量），不涉及 `_index` / `_am`。

股票腿的实现是对正式单腿因子的**逐行复刻**（同文件同函数语义）：
  rev5   ← factors/momentum.py: short_term_reversal_5 = −_ret_k(ctx, 5)
  mom120 ← factors/momentum.py: momentum_120 = _ret_k(ctx, 120)
  mom60  ← factors/momentum.py: momentum_60 = _ret_k(ctx, 60)
  lowvol ← factors/volatility.py: vol_120 = roll_std(_ret(ctx), 120, min_count=60)
  rev10  = −_ret_k(ctx, 10)（无对应正式因子的窗口变体，仅此一处不在生产库中）
"""
from __future__ import annotations

import numpy as np

from fea import mathx as mx

SIGNALS = (
    "rev5_x_illiqstate", "mom120_x_dispstate", "lowvol_x_breadthstate",
    "rev10_x_illiqstate", "mom60_x_dispstate",
)
#: 机制/变体 → 正式市场父因子名（进 deps，runner 会一并复算）
MARKET_PARENT = {
    "rev5_x_illiqstate": "mkt_illiquidity_median",
    "mom120_x_dispstate": "mkt_dispersion",
    "lowvol_x_breadthstate": "mkt_above_ma60_ratio",
    "rev10_x_illiqstate": "mkt_illiquidity_median",
    "mom60_x_dispstate": "mkt_dispersion",
}
#: 机制/变体 → 在 factors.market._base 输出字典里的键
MARKET_KEY = {
    "rev5_x_illiqstate": "illiquidity_median",
    "mom120_x_dispstate": "dispersion",
    "lowvol_x_breadthstate": "above_ma60_ratio",
    "rev10_x_illiqstate": "illiquidity_median",
    "mom60_x_dispstate": "dispersion",
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


def _leg_rev10(ctx) -> np.ndarray:
    """−momentum_10 = −_ret_k(ctx, 10)（变体腿；值大 = 近 10 日超跌更多）。"""
    return -_ret_k(ctx, 10)


def _leg_mom120(ctx) -> np.ndarray:
    """momentum_120 = _ret_k(ctx, 120)（值大 = 半年趋势更强）。"""
    return _ret_k(ctx, 120)


def _leg_mom60(ctx) -> np.ndarray:
    """momentum_60 = _ret_k(ctx, 60)（变体腿；值大 = 季度趋势更强）。"""
    return _ret_k(ctx, 60)


def _leg_lowvol(ctx) -> np.ndarray:
    """−vol_120 = −roll_std(_ret_traded, 120, min_count=60)（值大 = 低波动）。"""
    return -ctx.roll_std(_ret_traded(ctx), 120, min_count=60)


STOCK_LEG = {
    "rev5_x_illiqstate": _leg_rev5,
    "mom120_x_dispstate": _leg_mom120,
    "lowvol_x_breadthstate": _leg_lowvol,
    "rev10_x_illiqstate": _leg_rev10,
    "mom60_x_dispstate": _leg_mom60,
}


# ══════════════════════════════════════════════════════════════════════════
# 市场状态腿 —— 调用正式市场因子的同一实现（factors/market.py _base），当前面板复算
# ══════════════════════════════════════════════════════════════════════════

def _market_series(ctx, signal: str) -> np.ndarray:
    """正式市场父因子在**当前面板**上的 (T,) 取值。

    调用 `factors.market._base`（池内固定 2115 只主板的日频横截面标量，与生产
    因子完全同一函数，无转写）。三个父因子的数据源由 deps 声明
    （stock_daily / stock_adj_factor），`ctx.accessed` 由本候选的 deps 覆盖。
    """
    from factors.market import _base
    return np.asarray(_base(ctx)[MARKET_KEY[signal]], dtype=np.float64)


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
    """唯一入口：五选一的预登记机制/变体，无窗口/方向搜索。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("U8 permits only the five preregistered interaction signals")
    signal = parameters["signal"]

    def compute(ctx):
        leg = STOCK_LEG[signal](ctx)
        state = _time_z(ctx, _market_series(ctx, signal))
        value = _standardize(ctx, leg) * state
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
