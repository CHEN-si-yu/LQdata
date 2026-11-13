"""R10 绿卡基础因子批 · 收益波动与振幅期限结构（Tushare Risk 族）。

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-05 裁定：三项基础门槛 + 市场绿卡「透明观测描述量」规则）
════════════════════════════════════════════════════════════════════════
本模块全部 6 个对象都是**单条原始价格序列的标准变换**：
  · `return_std_n` = Std(日收益, n)              —— 单序列滚动标准差（volatility）
  · `high_low_n`   = Max(hfq_close,n)/Min(hfq_close,n) —— 单序列滚动极值位置（rolling extreme / amplitude）
没有跨序列复合、没有滚动相关/回归、没有矩比（mean/std）、没有事件构造。

★ 与 Tushare 原文的对应：`high_low_*d` 原文写
  `NetValue_t = CumulativeProduct(1+DailyReturn_i) from a fixed start point;
   Factor = Max(NetValue_{t-w+1:t}) / Min(NetValue_{t-w+1:t})`。
  窗口内 max 与 min 共享同一个累乘基座 ⇒ 基座精确相消
  ⇒ 恒等于 `Max(hfq_close)/Min(hfq_close)` 在**同一窗口**上。故直接实现为后复权收盘价的
  窗口极值之比（不引入任何近似）。这是**振幅**观测，不是「现价相对极值的位置」
  （后者是既有 `price_to_52w_high` / `price_distance_from_52w_low` 两个生产因子的轴）。

★ 明确**不在**本模块（Lead 裁定 OUT，规则：排除滚动相关/回归类构造与矩比构造）：
  `beta_*_000300/000001`、`sigma_1320d_*`、`beta_consistency_1320d_*`、`volume_beta_120d_*`、
  `sharpe_60d/750d`、`adjusted_sharpe_750d`。它们虽在 Tushare 的 Risk 类里，但不属于绿卡通道。

★★ 本批对象是**绿卡候选（只主张三项基础门槛；G4–G7 未主张）**。
   `higher_is_better=true` 只表示 declared=raw（不做任何反号），**不代表任何预测方向**。

════════════════════════════════════════════════════════════════════════
二、口径
════════════════════════════════════════════════════════════════════════
· 收益率用 `ctx.ret_clean(1)`：**停牌日置 NaN**。`ctx.ret(1)` 的 hfq_close 是前向填充的，
  停牌日算出精确 0.0，直接进滚动窗口会系统性**低估**波动（`fea/context.py::ret_clean`
  的原文警告）。在产 `vol_120` 用 `np.where(ctx.traded(), ctx.ret(1), np.nan)`，语义相同。
· 极值用 `ctx.hfq("close")`：后复权 = PIT 安全（历史值不随未来分红改变），且水平量在停牌日
  前向填充 ⇒ 不会造出假的极值。
· 窗口 21/63/126/252 个**交易日**（Tushare 原文用 `month × 21`，1/2/3/6/12 个月；
  本模块取 21/63/252 = 短/中/长三点，42/126 已按 IN 判定但为限定批次规模未登记）。
· `min_count = n//2`，与项目既有滚动因子同口径（停牌多的股票窗口内有效值少，但不到一半即 NaN）。
· 全部为因果滚动窗口；模块内**没有** `shift` / 前向填充 / 全样本统计量。
"""
from __future__ import annotations

import numpy as np

FROZEN = {
    ("ret_std", 21): "r10gc_ret_std_21_v1",
    ("ret_std", 63): "r10gc_ret_std_63_v1",
    ("ret_std", 252): "r10gc_ret_std_252_v1",
    ("high_low", 21): "r10gc_high_low_21_v1",
    ("high_low", 63): "r10gc_high_low_63_v1",
    ("high_low", 252): "r10gc_high_low_252_v1",
}


def _frozen_key(parameters):
    if not isinstance(parameters, dict):
        return None
    return (parameters.get("kind"), parameters.get("window"))


def _roll_std(ctx, mat, n, min_count):
    """带 min_count 的滚动总体标准差（ddof=0）；公式与 `factors/liquidity.py::_roll_std` 一致，
    理由（cumsum 版的缺失值分母 bug）见该处注释与 `factors/volatility.py` 的说明。"""
    mu = ctx.roll_mean(mat, n, min_count)
    var = ctx.roll_mean(mat * mat, n, min_count) - mu * mu
    return np.sqrt(np.maximum(var, 0.0))


def build(parameters):
    key = _frozen_key(parameters)
    if key not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))
    kind, n = key

    def compute(ctx):
        if kind == "ret_std":
            r = np.asarray(ctx.ret_clean(1), dtype=np.float64)
            return _roll_std(ctx, r, n, n // 2)
        px = np.asarray(ctx.hfq("close"), dtype=np.float64)
        hi = ctx.roll_max(px, n, n // 2)
        lo = ctx.roll_min(px, n, n // 2)
        # 比值型：分母是窗口最低价，恒 > 0；地板 1e-12 只挡 0/NaN。
        return ctx.safe_div(hi, lo, min_abs_den=1e-12)

    return compute
