"""R10 绿卡基础因子批 · 换手率期限结构（Tushare Liquidity 族）。

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-05 裁定：三项基础门槛 + 市场绿卡「透明观测描述量」规则）
════════════════════════════════════════════════════════════════════════
本模块全部 10 个对象都是**单条原始日频序列 `turnover` 的标准变换**：
  · 换手率水平 `MA(turnover, n)`           —— 单序列滚动均值（"means"）
  · 换手率波动 `Std(turnover, n)`          —— 单序列滚动标准差（"standard deviations"）
  · 均线偏离 `MA_s/MA_l − 1`               —— 同序列短/长均值偏离（"moving-average deviation"）
  · 波动偏离 `Std_s/Std_l − 1`             —— 同序列短/长标准差之比（★ 边界案例，见下）
没有跨序列复合、没有对已聚合比率的再运算、没有事件/条件构造、没有 HHI、
没有滚动相关/回归、没有同字段净额构造、没有联想式解读。

`turnover = vol / float_share × 100`（`ctx.px("turnover")`）—— 两个**原始字段**之比
（"ratios"），与在产 `turnover_std_20`/`turnover_anomaly_20` 同一口径，截面可比。

★ 边界案例（唯一一个，已上报 Lead）：`bias_std_turn_*` 是「同序列两个滚动标准差之比」。
  规则明文列 IN 的是「均线偏离」；标准差之比是同一族的期限结构读法，且与
  `MA_s/MA_l − 1` 结构同构（都是同一条原始序列的短/长端比较），故判 IN；
  但它在字面上不是「均线」，记录在案，若 Lead 复核改判，它是全批第一个应撤的对象。

★★ 本批对象是**绿卡候选（只主张三项基础门槛；G4–G7 未主张）**。
   `higher_is_better=true` 只表示 declared=raw（不做任何反号），
   **不代表任何预测方向**，也不构成「已验证信号」。

════════════════════════════════════════════════════════════════════════
二、窗口口径（★ Tushare 原文自相矛盾，必须明示解释）
════════════════════════════════════════════════════════════════════════
原文写作 `avg_turnover_21d` → w=121、`avg_turnover_252d` → w=1221、`bias_*_504d` → w=2421。
121/221/321/621/1221/2421 是「月数拼接 21」的损坏渲染（1|21、2|21、3|21、6|21、12|21、24|21），
照抄会得到 1221/2421 个交易日的窗口，远超样本长度。
**本模块按名称解释：1 个月 = 21 个交易日**，故窗口 = 21/42/63/126/252（与 Tushare 自己的
中文描述「过去 1 个月 / 2 个月 / 3 个月 / 6 个月 / 12 个月」一致）。
24 个月（504）的网格本轮未登记（见 family.yaml 的 bound 说明）。

════════════════════════════════════════════════════════════════════════
三、PIT 合同
════════════════════════════════════════════════════════════════════════
输入只有 `stock_daily`（vol）与 `stock_finance`（float_share），两者在
`conf/frequency.yaml` 均声明 `delay_days: 0`（T 日收盘后可得），**不需要任何滞后位移**。
`ctx.px("turnover")` 的 vol 是**流量**（停牌日 NaN，不前向填充），float_share 是**状态量**
（前向填充）—— 这正是「T 日只用到 ≤T 的量」的口径。
本模块内**没有任何** `shift` / 前向填充 / 全样本统计量；全部为因果滚动窗口
（`roll_*` 只看 ≤t 的格子）。`min_count = n//2` 与项目既有换手率因子同口径。
"""
from __future__ import annotations

import numpy as np

# ══════════════════════════════════════════════════════════════════════
# 事前冻结的变体表 —— build() 拒绝任何未列出的参数组合（不存在被搜索的窗口）
# ══════════════════════════════════════════════════════════════════════
FROZEN = {
    ("lvl", 21): "r10gc_turn_lvl_21_v1",
    ("lvl", 63): "r10gc_turn_lvl_63_v1",
    ("lvl", 252): "r10gc_turn_lvl_252_v1",
    ("std", 42): "r10gc_turn_std_42_v1",
    ("std", 126): "r10gc_turn_std_126_v1",
    ("std", 252): "r10gc_turn_std_252_v1",
    ("bias_turn", 21, 252): "r10gc_bias_turn_21_252_v1",
    ("bias_turn", 63, 252): "r10gc_bias_turn_63_252_v1",
    ("bias_turn", 126, 252): "r10gc_bias_turn_126_252_v1",
    ("bias_std_turn", 21, 252): "r10gc_bias_std_turn_21_252_v1",
}


def _frozen_key(parameters):
    if not isinstance(parameters, dict):
        return None
    kind = parameters.get("kind")
    if kind in ("lvl", "std"):
        return (kind, parameters.get("window"))
    if kind in ("bias_turn", "bias_std_turn"):
        return (kind, parameters.get("short"), parameters.get("long"))
    return None


def _roll_std(ctx, mat, n, min_count):
    """带 min_count 的滚动总体标准差（ddof=0）。

    ★ 不能用 `ctx.roll_std`：`mathx.roll_var` 走 cumsum，缺失值按 0 计入、分母固定取 n，
    min_count < n 时结果系统性偏小（`factors/liquidity.py` 的实测：1.125 vs 正确 1.219），
    而缺多少格与停牌频率相关 ⇒ 偏差自带截面结构。
    改用 E[x²] − E[x]²，两个期望都走 `roll_mean`（在有效值上取均值），与
    pandas `rolling(n, min_periods=k).std(ddof=0)` 逐格一致。与在产 `turnover_std_20`
    使用的 `_roll_std` 是同一个公式。
    """
    mu = ctx.roll_mean(mat, n, min_count)
    var = ctx.roll_mean(mat * mat, n, min_count) - mu * mu
    return np.sqrt(np.maximum(var, 0.0))


def build(parameters):
    key = _frozen_key(parameters)
    if key not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))
    kind = key[0]

    def compute(ctx):
        t = np.asarray(ctx.px("turnover"), dtype=np.float64)
        if kind == "lvl":
            n = key[1]
            return ctx.roll_mean(t, n, n // 2)
        if kind == "std":
            n = key[1]
            return _roll_std(ctx, t, n, n // 2)
        short, long = key[1], key[2]
        if kind == "bias_turn":
            s = ctx.roll_mean(t, short, short // 2)
            l = ctx.roll_mean(t, long, long // 2)
            # 分母是换手率水平（百分数，恒 >= 0）；地板 1e-9 只挡 0/NaN，不制造假值。
            return ctx.safe_div(s, l, min_abs_den=1e-9) - 1.0
        s = _roll_std(ctx, t, short, short // 2)
        l = _roll_std(ctx, t, long, long // 2)
        return ctx.safe_div(s, l, min_abs_den=1e-12) - 1.0

    return compute
