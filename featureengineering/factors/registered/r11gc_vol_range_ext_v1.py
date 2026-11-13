"""R11 绿卡批 2 · 收益波动与振幅期限结构的补窗（Tushare Risk 族；family `r11gc_vol_range_ext_v1`）。

════════════════════════════════════════════════════════════════════════
一、与批 1 的关系（为什么需要新模块，而不是新 family 就够）
════════════════════════════════════════════════════════════════════════
批 1 = `research/families/r10gc_vol_range_v1.py`，登记 21/63/252 三点 × 两种度量（6 个对象）。
它的 `FROZEN` 是**模块层常量**，`build()` 直接对 `(kind, window)` 查表并 raise，
**不从 family YAML 读取允许集** ⇒ 新 family 无法复用批 1 模块。本模块是**薄扩展**：
同一口径的独立实现，`FROZEN` 只含批 1 未登记的四个窗口。
**批 1 模块一行未改**（其源码 hash 参与已注册候选的定义，改动会作废既有证据）。

`build()` 只接收候选自己的 `parameters` 字典（`research/catalog.py::build_spec`
→ `fn = builder(dict(definition["parameters"]))`），**family id 不参与**，
故「新 family + 新模块」是自由的组合，没有隐藏耦合。

════════════════════════════════════════════════════════════════════════
二、★ 窗口语义：Tushare 原文的数字是「月数 × 21」被拼接后的产物，不得照抄
════════════════════════════════════════════════════════════════════════
Tushare 文本里的 `w=121 / 221 / 321 / 621 / 1221 / 2421` **不是交易日数**：
    1 个月 → 21 交易日   被写成 121     （"1" 个月 + "21" 交易日）
    2 个月 → 42 交易日   被写成 221
    3 个月 → 63 交易日   被写成 321
    6 个月 → 126 交易日  被写成 621
   12 个月 → 252 交易日  被写成 1221
   24 个月 → 504 交易日  被写成 2421
同类错误在别的族里同样出现（`avg_turnover_21d` → `w=121`、`bias_*_504d` → `w=2421`）。
**照抄字面数字会得到 221 / 621 个交易日的窗口**：在 2018-01 起的输出区间里，
621 个交易日 ≈ 2.5 年，窗口内大量格子只有一半有效值甚至不足，闸门照样能过（NaN 被 min_count 处理），
但语义已经不是「2 个月 / 6 个月的近期水平」了 —— 能通过所有闸门却毫无意义。

**本模块一律按「交易日」实现**：
    ret_std_42  = 过去 **42 个交易日**（2 个月 × 21）日收益的总体标准差
    ret_std_126 = 过去 **126 个交易日**（6 个月 × 21）日收益的总体标准差
    high_low_42 / high_low_126 同理（42 / 126 个交易日）
批 1 的 docstring 独立记录了同一判定（「42/126 已按 IN 判定但为限定批次规模未登记」），
两处一致。

════════════════════════════════════════════════════════════════════════
三、资格（用户 2026-10-05 裁定：三项基础门槛 + 市场绿卡「透明观测描述量」规则）
════════════════════════════════════════════════════════════════════════
四个对象都是**单条原始价格/收益序列的标准变换**：
  · `ret_std_n`  = Std(日收益, n)                        —— 单序列滚动标准差（volatility）
  · `high_low_n` = Max(hfq_close, n) / Min(hfq_close, n) —— 单序列滚动极值之比（振幅）
没有跨序列复合、没有滚动相关/回归、没有矩比（mean/std）、没有事件构造、没有截面排名。
Alpha101（31 个多因子 Rank/Correlation 复合式）**不在本通道内**。

★★ 本批对象是**绿卡候选（只主张三项基础门槛成立；G4–G7 未主张）**。
   `higher_is_better=true` **只表示 declared=raw**（机制层不做任何反号），
   **不代表任何预测方向，不主张预测有效性**，任何报告里都不得写成「已验证信号」。

════════════════════════════════════════════════════════════════════════
四、口径（与批 1 逐条一致）
════════════════════════════════════════════════════════════════════════
· 收益率 `ctx.ret_clean(1)`：**停牌日置 NaN**。`ctx.ret(1)` 的 hfq_close 被前向填充，
  停牌日算出精确 0.0，直接进滚动窗口会系统性**低估**波动。
· 极值 `ctx.hfq("close")`：后复权 = PIT 安全（历史值不随未来分红改变），
  水平量在停牌日前向填充 ⇒ 不会造出假的极值。
· `min_count = n // 2`（与批 1 及项目既有滚动因子同口径）。
· 标准差为**总体口径 ddof=0**；`_roll_std` 用 `mean(x²) − mean(x)²` 自算，
  因为 `ctx.roll_mean` 的分母是窗口内**有效值个数**，两个窗口有效集相同 ⇒ 该式在缺失下仍严格等于
  ddof=0 的总体标准差（绕开 `ctx.roll_std` 在 min_count<n 时的已知口径坑）。
· 极值比的分母（窗口最低价）恒 > 0；`ctx.safe_div(min_abs_den=1e-12)` 只挡 0/NaN。
· 全部为因果滚动窗口；模块内**没有** `shift` / 前向填充 / 全样本标准化。
· 输入只有 `stock_daily` 与 `stock_adj_factor`，两者 `delay_days: 0`（`conf/frequency.yaml` 原文）。
"""
from __future__ import annotations

import numpy as np

# 批 1 未登记的四个窗口；键 = (kind, window)，与批 1 的 `_frozen_key` 同构。
FROZEN = {
    ("ret_std", 42): "r11gc_ret_std_42_v1",
    ("ret_std", 126): "r11gc_ret_std_126_v1",
    ("high_low", 42): "r11gc_high_low_42_v1",
    ("high_low", 126): "r11gc_high_low_126_v1",
}


def _frozen_key(parameters):
    if not isinstance(parameters, dict):
        return None
    return (parameters.get("kind"), parameters.get("window"))


def _roll_std(ctx, mat, n, min_count):
    """带 min_count 的滚动总体标准差（ddof=0）；与批 1 `_roll_std`、`factors/liquidity.py` 同式。"""
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
        return ctx.safe_div(hi, lo, min_abs_den=1e-12)

    return compute
