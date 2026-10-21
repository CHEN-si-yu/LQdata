"""R12 绿卡批 A · 规模与价格四件套（family `r12gc_size_price_v1`）。

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-06 绿卡通道扩容裁定：三项基础门槛）
════════════════════════════════════════════════════════════════════════
  log_price              对数收盘价        ln(ClosePrice)
  float_size             流通市值负对数    −ln(FloatShares × ClosePrice / 1e6)
  market_value_leverage  市值杠杆         (MarketCap − 非流动负债合计) / MarketCap
  return_42              42 日动量         Π(1 + DailyReturn, 42) − 1

前三个是**价格/股本状态量的一条序列变换或一次比率**；return_42 是单条后复权价格
序列的标准滚动乘积（规则明文 IN 的「对单序列原始字段的标准变换」）。
★ `ma_20d` 不在本模块：与 log_price 同为「未复权 close 单序列」的变换，
  池化 |ρ| 预期 ≥0.98，已按规则 1 F3 近邻同构裁 OUT（README §3/§4-5）。

★★ 本批对象是**绿卡候选（只主张三项基础门槛；G4–G7 未主张）**。
   `higher_is_better=true` 只表示 declared=raw（不做任何反号），不代表任何预测方向。

════════════════════════════════════════════════════════════════════════
二、口径（与同批 YAML 逐条对应）
════════════════════════════════════════════════════════════════════════
· 自算市值约定与在产 `log_mv` 完全同源：**未复权 close（状态量，停牌前向填充）
  × 当期已披露股本（`*_share`，状态量）**；不用供应商日频 total_mv 快照（会事后重算）。
· float_size 保留供应商的 1e6 常数：−ln(市值/1e6)，对截面秩无影响，逐字对上原文。
· market_value_leverage 的 NonCurrentLiability 映射白名单 `total_ncl`（BS 时点）。
  非流动负债为零的公司取值为 1（真实值堆叠在右端，预检 B 看分位）。
· return_42 用 `r12gc_common.ret_k`（= 在产 momentum_* 家族的 `_ret_k`）：
  后复权 + 停牌 ffill 语义；**不用 ctx.ret(42)** —— 它把「单日 ±60%」阈值套在
  42 日累计上，会把涨得最多的股票误杀；单日 |ret|>60% 由价格层置 NaN 并毒化整窗；
  窗口内零成交 ⇒ NaN（不是凭空 0）；min_count 语义由 roll_sum 的 NaN 毒化保证（满 42 日）。
  窗口 42 = 2×21 交易日（供应商参数行「默认值 21」为笔误，名称与正文均为 42）。

════════════════════════════════════════════════════════════════════════
三、PIT / 无泄露
════════════════════════════════════════════════════════════════════════
· price 层 `conf/frequency.yaml` 声明 delay_days: 0（T 日收盘后可得）；
  `close`/`total_share`/`float_share` 均为**状态量**（停牌前向填充），
  `vol`/`amount` 流量字段不参与本模块。
· 财务侧（market_value_leverage 的 total_ncl）走版本表：T 日只用已披露最新版本。
· 模块内没有 shift / 全样本统计量；return_42 是纯因果滚动窗。

★ 依赖 `research/families/r12gc_common.py`（mktcap / ret_k）。
  catalog 只对**本文件**取 sha256，公共文件改动等价于定义变更 —— 须重新注册。
"""
from __future__ import annotations

import numpy as np

from research.families.r12gc_common import mktcap, ret_k

FROZEN = {
    "log_price": "r12gc_log_price_v1",
    "float_size": "r12gc_float_size_v1",
    "market_value_leverage": "r12gc_market_value_leverage_v1",
    "return_42": "r12gc_return_42_v1",
}

MIN_ABS_DEN = 1e6  # 市值（元）地板


def build(parameters):
    variant = parameters.get("variant") if isinstance(parameters, dict) else None
    if variant not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))

    def compute(ctx):
        if variant == "log_price":
            return ctx.safe_log(np.asarray(ctx.px("close"), dtype=np.float64))
        if variant == "float_size":
            cap = np.asarray(ctx.px("close"), dtype=np.float64) * \
                np.asarray(ctx.px("float_share"), dtype=np.float64)
            return -ctx.safe_log(cap / 1e6)
        if variant == "market_value_leverage":
            cap = mktcap(ctx, "total_share")
            ncl = np.asarray(ctx.point("total_ncl"), dtype=np.float64)
            return ctx.safe_div(cap - ncl, cap, MIN_ABS_DEN)
        # return_42
        return ret_k(ctx, 42)

    return compute
