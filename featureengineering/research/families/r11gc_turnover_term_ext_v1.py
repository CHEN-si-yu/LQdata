"""R11 绿卡批 2 · 换手率期限结构补窗（Tushare Liquidity 族 IN-D 余量；family `r11gc_turnover_term_ext_v1`）。

════════════════════════════════════════════════════════════════════════
一、与批 1 的关系
════════════════════════════════════════════════════════════════════════
批 1 = `research/families/r10gc_turnover_term_v1.py`（10 个对象：lvl 21/63/252、std 42/126/252、
bias_turn 21/63/126_252、bias_std_turn 21_252）。其 `FROZEN` 是**模块层常量**，
`build()` 直接查表并 raise，**不从 family YAML 读取允许集** ⇒ 新 family 无法复用批 1 模块。
本模块是**薄扩展**：口径逐条与批 1 一致，`FROZEN` 只含批 1 未登记的 19 个变体。
**批 1 模块一行未改**（其 hash 参与已注册候选的定义）。

`build()` 只接收候选自己的 `parameters`（`research/catalog.py::build_spec`
→ `fn = builder(dict(definition["parameters"]))`），**family id 不参与**，故无隐藏耦合。

════════════════════════════════════════════════════════════════════════
二、★ 窗口语义（与批 1 §二 同一判定；Tushare 原文的数字是损坏渲染）
════════════════════════════════════════════════════════════════════════
原文 `avg_turnover_21d → w=121`、`bias_*_504d → w=2421`：121/221/321/621/1221/2421 是
「月数拼接 21」（1|21、2|21、3|21、6|21、12|21、24|21）。**照抄会得到 1221/2421 个交易日的窗口**，
远超样本可得区间，闸门照样过（min_count 把缺失吃掉）却毫无意义。
本模块按名称解释：**1 个月 = 21 个交易日** ⇒ 窗口取值 5/10/20/21/42/63/126/252/504 个**交易日**。
本轮首次登记 24 个月 = **504 个交易日**的长端（批 1 的 bound 说明里明确留给后续批次）。
warmup_days = 800 日历天（≈550 交易日）以保证 504 日长端在 2018-01-02 就有完整历史。

════════════════════════════════════════════════════════════════════════
三、资格（用户 2026-10-05 裁定：三项基础门槛 + 市场绿卡「透明观测描述量」规则）
════════════════════════════════════════════════════════════════════════
全部 19 个对象都是**单条原始日频序列的标准变换**：
  · 换手率水平    MA(turnover, n)                  —— 单序列滚动均值
  · 换手率波动    Std(turnover, n)                 —— 单序列滚动标准差
  · 均线偏离      MA_s / MA_l − 1                  —— 同序列短/长均值偏离
  · 波动偏离      Std_s / Std_l − 1                —— 同序列短/长标准差之比（★ 边界案例，同批 1）
  · 成交额水平    MA(amount, 20)                   —— 单序列滚动均值
没有跨序列复合、没有对已聚合比率的再运算、没有事件/条件构造、没有 HHI、
没有滚动相关/回归、没有同字段净额构造。Alpha101 不在本通道。

`turnover = vol / float_share × 100`（`ctx.px("turnover")`）—— 两个**原始字段**之比，
与在产 `turnover_std_20` / `turnover_anomaly_20` 同一口径，截面可比。

★★ 本批对象是**绿卡候选（只主张三项基础门槛；G4–G7 未主张）**。
   `higher_is_better=true` 只表示 declared=raw（不做任何反号），
   **不代表任何预测方向，不主张预测有效性**，不得写成「已验证信号」。

════════════════════════════════════════════════════════════════════════
四、口径与 PIT
════════════════════════════════════════════════════════════════════════
· 输入 `stock_daily`（vol / amount）与 `stock_finance`（float_share），`conf/frequency.yaml`
  均声明 `delay_days: 0`（T 日收盘后可得），**不需要滞后位移**。
· `ctx.px("turnover")` 的 vol 是**流量**（停牌日 NaN，不前向填充）；float_share 是**状态量**
  （前向填充）—— 即「T 日只用到 ≤T 的量」。`amount` 同为流量，停牌日 NaN。
· `_roll_std` 用 `E[x²] − E[x]²` 自算（两个期望都走 `ctx.roll_mean`，在有效值上取均值），
  与批 1 及在产 `turnover_std_20` 同一公式；**不用 `ctx.roll_std`**（其 cumsum 版把缺失当 0、
  分母固定取 n，min_count<n 时系统性偏小，且偏差与停牌频率相关）。
· `min_count = n // 2`；标准差为总体口径 ddof=0。
· 模块内**没有** `shift` / 前向填充 / 全样本统计量；全部为因果滚动窗口。
"""
from __future__ import annotations

import numpy as np

FROZEN = {
    ("lvl", 5): "r11gc_turn_lvl_5_v1",
    ("lvl", 10): "r11gc_turn_lvl_10_v1",
    ("lvl", 20): "r11gc_turn_lvl_20_v1",
    ("lvl", 42): "r11gc_turn_lvl_42_v1",
    ("lvl", 126): "r11gc_turn_lvl_126_v1",
    ("std", 63): "r11gc_turn_std_63_v1",
    ("bias_turn", 21, 504): "r11gc_bias_turn_21_504_v1",
    ("bias_turn", 42, 252): "r11gc_bias_turn_42_252_v1",
    ("bias_turn", 42, 504): "r11gc_bias_turn_42_504_v1",
    ("bias_turn", 63, 504): "r11gc_bias_turn_63_504_v1",
    ("bias_turn", 126, 504): "r11gc_bias_turn_126_504_v1",
    ("bias_std_turn", 42, 252): "r11gc_bias_std_turn_42_252_v1",
    ("bias_std_turn", 63, 252): "r11gc_bias_std_turn_63_252_v1",
    ("bias_std_turn", 126, 252): "r11gc_bias_std_turn_126_252_v1",
    ("bias_std_turn", 21, 504): "r11gc_bias_std_turn_21_504_v1",
    ("bias_std_turn", 42, 504): "r11gc_bias_std_turn_42_504_v1",
    ("bias_std_turn", 63, 504): "r11gc_bias_std_turn_63_504_v1",
    ("bias_std_turn", 126, 504): "r11gc_bias_std_turn_126_504_v1",
    ("amount_ma", 20): "r11gc_amount_ma_20_v1",
}


def _frozen_key(parameters):
    if not isinstance(parameters, dict):
        return None
    kind = parameters.get("kind")
    if kind in ("lvl", "std", "amount_ma"):
        return (kind, parameters.get("window"))
    if kind in ("bias_turn", "bias_std_turn"):
        return (kind, parameters.get("short"), parameters.get("long"))
    return None


def _roll_std(ctx, mat, n, min_count):
    """带 min_count 的滚动总体标准差（ddof=0）——与批 1 `_roll_std` 同一公式。"""
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
        if kind == "amount_ma":
            n = key[1]
            a = np.asarray(ctx.px("amount"), dtype=np.float64)
            return ctx.roll_mean(a, n, n // 2)
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
            return ctx.safe_div(s, l, min_abs_den=1e-9) - 1.0
        s = _roll_std(ctx, t, short, short // 2)
        l = _roll_std(ctx, t, long, long // 2)
        return ctx.safe_div(s, l, min_abs_den=1e-12) - 1.0

    return compute
