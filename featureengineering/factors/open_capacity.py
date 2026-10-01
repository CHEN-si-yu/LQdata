"""T 日收盘估计 T+1 开盘的可成交容量（执行信号，非收益 alpha）。

仅使用截至 T 日的首根 5 分钟成交额。`amt_open5` 来自 `fea/open5.py` 的
`stock_history_5min` 日频派生层，单位为元；T+1 的价格、成交额和状态均不读取。

两个量的分工：`open5_capacity_p20_log` 是保守的绝对金额基准，可按交易参与率
换算单笔金额上限；`open5_capacity_floor_ratio_20` 是下尾/中位之比，表示开盘
池子的稳定度。它们是待检验的执行特征，不能直接解释为实盘成交保证。

缺失：窗口至少 15 个有真实成交的日子；当日停牌、首根缺失、金额非正或仅有
一根分钟棒时为 NaN。窗口内停牌日不当作零元成交。没有 T+1 状态资料，故不把
T 日封板直接推定为 T+1 无法成交；封板风险应在模型执行侧另行检验。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register


WINDOW = 20
MIN_COUNT = 15
WARMUP_DAYS = 56  # 20 个交易日按框架换算为 20*1.8+20 个日历日
DEPS = ("stock_history_5min", "stock_daily")


def _open_amount(ctx) -> np.ndarray:
    """只保留 T 日实际可观测到的首根金额，停牌不补零。"""
    amount = np.asarray(ctx.open5_field("amt_open5"), dtype=np.float64)
    n_bars = np.asarray(ctx.open5_field("n_bars"), dtype=np.float64)
    traded = np.asarray(ctx.traded(), dtype=bool)
    valid = traded & np.isfinite(amount) & (amount > 0) & np.isfinite(n_bars) & (n_bars >= 2)
    return np.where(valid, amount, np.nan)


def _history_quantile(ctx, amount: np.ndarray, q: float) -> np.ndarray:
    value = ctx.roll_quantile(amount, WINDOW, q, min_count=MIN_COUNT)
    # T 日没有真实交易时，历史量不能冒充当天可用的买入容量估计。
    return np.where(np.isfinite(amount), value, np.nan)


@register(FactorSpec(
    name="open5_capacity_p20_log",
    group="intraday",
    desc="截至 T 日的近 20 个交易日首根成交额 20% 分位对数，估 T+1 开盘保守容量",
    formula="log(Quantile_0.20(amt_open5[T-19:T]))，窗口至少 15 个有效日",
    deps=DEPS,
    warmup_days=WARMUP_DAYS,
    higher_is_better=True,
    version=1,
    note=("执行容量口径，金额单位元。T 日收盘可得；不读取 T+1 数据。"
          "可按 exp(value) 还原历史开盘池子下分位，再乘交易参与率；"
          "这是容量基准而非成交保证。停牌和无效分钟棒为 NaN，窗口内不补零。"),
))
def open5_capacity_p20_log(ctx):
    amount = _open_amount(ctx)
    p20 = _history_quantile(ctx, amount, 0.20)
    return ctx.safe_log(np.where(p20 > 0, p20, np.nan))


@register(FactorSpec(
    name="open5_capacity_floor_ratio_20",
    group="intraday",
    desc="近 20 日首根成交额下分位与中位数之比，衡量开盘池子的下尾稳定度",
    formula="Quantile_0.20(amt_open5[T-19:T])/Median(amt_open5[T-19:T])，至少 15 个有效日",
    deps=DEPS,
    warmup_days=WARMUP_DAYS,
    higher_is_better=True,
    version=1,
    note=("执行稳定度口径，值越低表示历史开盘池子更容易塌缩。"
          "与绝对容量对数分工：同样大金额的两只股票可有不同下尾风险。"
          "分子分母同为首根成交额，单位相消；停牌和无效分钟棒为 NaN。"),
))
def open5_capacity_floor_ratio_20(ctx):
    amount = _open_amount(ctx)
    p20 = _history_quantile(ctx, amount, 0.20)
    median = _history_quantile(ctx, amount, 0.50)
    return ctx.safe_div(p20, median, min_abs_den=1.0)
