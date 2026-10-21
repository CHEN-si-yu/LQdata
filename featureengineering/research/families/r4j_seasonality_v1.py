"""R4-J · U10 — 日历季节性（Heston–Sadka 同月效应）。

机制集合（B2 §5 U10 预登记，预算 ≤4；方向与实现口径在看任何结果前冻结）：

  1. same_month_mean_ret     过去最多 6 个历年同月月收益的**均值**（min 2 个历史同月）。方向 true。
  2. same_month_consistency  同月历史月收益**为正的比例**（min 2）。方向 true。
  变体（登记前冻结，标注变体性）：
  A. same_month_min4         机制 1 的样本充足性变体：最少 4 个历史同月。方向 true。
  B. same_month_prev_year    同月口径退化为「去年同期月」单点（min 1）。方向 true。

──────────────────────────────────────────────────────────────────────────────
实现冻结（与任务书逐条对应，字段口径以代码为唯一准绳）：

  * 只用**当年本月之前**已实现的历史同月：t 所在日历月 (Y, m) 的值只由
    (Y−1, m) … (Y−6, m) 六个已完结日历月的月收益决定；**当月内不得回填本月收益**，
    不得使用 (Y, m) 自身的任何数据。月内为常数（步进函数，每次跨年更新）。
  * min 2 个历史同月才出值（min4 变体 = 4，prev_year 变体 = 1 个点本身必须存在）。
  * 值前 NaN 前缀逐股对齐：样本不足的股票为 NaN；分年有效票数表实测见 report.md（G2）。
  * 月收益 = 当月所有面板交易日的单日后复权收益累乘 − 1：`Π(1+r₁) − 1`。
    单日收益 r₁ 用价格层 `ctx.ret(1)`：|r|>60% 的复权因子脏数据已置 NaN、
    停牌日 r₁=0（ffill 语义，价格层既定口径）。以下三种情形该月月收益 = NaN：
      - 当月「一个成交日都没有」（整月停牌）→ NaN（沿用价格层 k>1 的守卫口径，
        避免凭空造出 0 收益；部分停牌不挡）；
      - 当月任一日 r₁ = NaN（未上市 / 脏数据）→ 整月 NaN；
      - 该股当月无任何价格（未上市）→ 整月 NaN。
  * 月序列按**面板日历月**切段（面板日期轴 = 连续交易日，月段序列 = 连续日历月），
    同月 = 段下标相差 12 的整数倍，即 `R[k−12], …, R[k−72]`，k = t 所在月的段。
    因此整个值为纯 PIT：所用月份段全部早于 t 所在月，任何面板截断（增量复算）下逐值一致。
  * 一致性口径：严格 `R > 0` 记为 1（R = 0 不算正）；分母 = 有限样本个数。
  * 输出 float64，无 Inf；不做的处理：不 winsor、不截面标准化（交给评价层）、不零填充。

数据/工程事实（2026-10-04 实测，详见 delivery/r4j_seasonality_v1/）：

  * 上游 `stock_daily` / `stock_adj_factor` 分区自 2010 起；本因子只依赖价格层
    （轻源），deps = [stock_daily, stock_adj_factor]，两者都无可得性滞后。
  * warmup_days = 2300 日历天：2018 年任务最深需要 2011-12 月收盘（为计算 2012-01
    同月收益），面板下界 = 2018-01-01 − 2300 天 ≈ 2011-09-16，留 100+ 天余量。
  * 朴素对照：momentum_20 / momentum_120 / short_term_reversal_5；定向对照 price_to_52w_high。

防重蹈检查（B2 §0，2026-10-04 执行）：`doc/因子清理记录_20261003.md`（374 个已删股票因子）
与 `doc/市场因子清理记录_20261004.md`（27 个已删市场因子）grep「同月 / 季节 / 日历 /
monthly / season / calendar / same-month」均 0 命中；现存正式因子（data/factors 119 个）
无同月/季节性类。最近亲缘为动量/反转族（作为对照报告，非同构念先行者）。
"""
from __future__ import annotations

import numpy as np

SIGNALS = (
    "same_month_mean_ret",
    "same_month_consistency",
    "same_month_min4",
    "same_month_prev_year",
)

SAME_MONTH_MAX = 6      # 最多回看 6 个历年同月
MIN_SAMPLES = {         # 每个信号出值所需的最少历史同月样本数（预登记）
    "same_month_mean_ret": 2,
    "same_month_consistency": 2,
    "same_month_min4": 4,
    "same_month_prev_year": 1,
}


def _month_segments(days: np.ndarray) -> np.ndarray:
    """面板日历月切段的段首行下标（days 已按升序）。"""
    mkey = np.asarray(days, dtype=np.int64) // 100          # YYYYMM
    return np.flatnonzero(np.r_[True, mkey[1:] != mkey[:-1]])


def _month_returns(r1: np.ndarray, traded: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """(S, C) 月收益：段内 Π(1+r₁) − 1；整月无成交 / 段内任一 NaN → NaN。

    multiply.reduceat 对 NaN 天然毒化（段内任一 NaN → 整段 NaN），与冻结口径一致。
    """
    x = 1.0 + r1
    gross = np.multiply.reduceat(x, starts, axis=0)
    out = gross - 1.0
    n_tr = np.add.reduceat(traded.astype(np.float64), starts, axis=0)
    return np.where(n_tr >= 1.0, out, np.nan)


def _lag_months(mat: np.ndarray, lags: int = 12) -> np.ndarray:
    """整体下移 lags 行（= lags//12 年前的同一日历月），首部 NaN。"""
    out = np.full_like(mat, np.nan)
    if lags < mat.shape[0]:
        out[lags:] = mat[:-lags]
    return out


def build(parameters):
    """四个预登记信号之一；无搜索窗口、无搜索方向。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("U10 只允许四个预登记信号（见 B2 §5 U10）")
    signal = parameters["signal"]
    minimum = MIN_SAMPLES[signal]

    def compute(ctx):
        r1 = np.asarray(ctx.ret(1), dtype=np.float64)         # 单日后复权收益（已清洗）
        traded = np.asarray(ctx.traded(), dtype=bool)
        days = np.asarray(ctx.panel.dates, dtype=np.int64)

        starts = _month_segments(days)
        month_ret = _month_returns(r1, traded, starts)        # (S, C)

        # 同月历史段：j = 1..6 年前的同月（段下标差 12j）
        lags = [_lag_months(month_ret, 12 * j) for j in range(1, SAME_MONTH_MAX + 1)]
        stack = np.stack(lags, axis=0)                        # (6, S, C)
        finite = np.isfinite(stack)
        count = finite.sum(axis=0)

        if signal == "same_month_prev_year":
            seg_value = lags[0]                               # 去年同期月单点（min 1）
        elif signal == "same_month_consistency":
            pos = np.sum(np.where(finite, stack > 0, 0), axis=0)
            seg_value = np.where(count >= minimum, pos / np.maximum(count, 1), np.nan)
        else:  # same_month_mean_ret / same_month_min4
            total = np.sum(np.where(finite, stack, 0.0), axis=0)
            seg_value = np.where(count >= minimum, total / np.maximum(count, 1), np.nan)

        # 月内广播：t 所在月的段 = 最后一个段首 ≤ t 的段
        seg_of_row = np.searchsorted(starts, np.arange(days.size), side="right") - 1
        value = seg_value[seg_of_row]                         # (T, C)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
