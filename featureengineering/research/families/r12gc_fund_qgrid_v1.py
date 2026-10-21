"""R12 绿卡批 A · 单季口径六件套（family `r12gc_fund_qgrid_v1`）。

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-06 绿卡通道扩容裁定：三项基础门槛）
════════════════════════════════════════════════════════════════════════
  eps_q            单季每股收益     S(归母净利) / total_share（时点）
  gpm_q            单季毛利率       (S(revenue) − S(oper_cost)) / S(revenue)
  npm_q            单季净利率       S(归母净利) / S(revenue)
  roa_q            单季总资产收益率 S(归母净利) / total_assets（时点）
  opt_tpro         单季营业利润占比 S(operate_profit) / S(total_profit)
  sales_to_market  单季营收市值比   S(revenue) / 自算总市值

六者 = 单季水平量（一次差分得到）与白名单时点/价格分量的**一次比率**。
供应商原文 _q 行与 _ttm/_y 行文本雷同（渲染损坏），口径按后缀语义 + 同表兄弟
条目的显式 _Q 写法裁定（npm_q_qoq 的 NetProfit_Q/OperatingRevenue_Q、eaa/eap 的 EPS_Q）。

★★ 本批对象是**绿卡候选（只主张三项基础门槛；G4–G7 未主张）**。
   `higher_is_better=true` 只表示 declared=raw（不做任何反号），不代表任何预测方向。

════════════════════════════════════════════════════════════════════════
二、单季网格（自建差分；先例 = 在产 r4c_accrual_ratio_q_v1）
════════════════════════════════════════════════════════════════════════
ctx 的 `ttm()/point()` **没有任何**单季水平入口（白名单只有 5 个单季比率，没有
单季净利/营收水平量），故按 `research/families/r4c_accruals_v1.py` 自建网格：

    S(q) = cum(q)                       若 q 为 Q1（q % 4 == 0）
         = cum(q) − cum(q−1)            否则（同一财年内差分）

**财年重置**：累计值每年 Q1 归零，Q1 的单季值就是累计本身，绝不拿 Q1 去减上一年年报。
q 与 q−1 两个累计值都可考时才出值，否则 NaN（不做 partial-sum 回退）。
季末过滤 / ann_date 可考 / pit = max(ann, f_ann, end) / 保留修订取最新版 —— 全部
细节见 `r12gc_common.py`。分母：eps_q 用 point(total_share)，roa_q 用 point(total_assets)，
sales_to_market 用自算市值（口径与在产 sp_ttm/log_mv 同源）；一律 safe_div(min 1e6)。

· 单季口径有季节性（四个季度系统性起伏）—— 这是 _q 频率的固有属性，非缺陷。
· opt_tpro 分母利润总额近零（盈亏平衡附近）会放大到极端值：safe_div floor=1e6
  只挡精确近零，两侧尾部交引擎 1%/99% 缩尾（不另设 clip，供应商原文无截幅）。

════════════════════════════════════════════════════════════════════════
三、PIT / 无泄露
════════════════════════════════════════════════════════════════════════
单季网格按 ann_date 锚定（旧期修订在其公告时点才影响当期单季值，与 ttm() 对修订的
响应一致）；落格走 `ctx.panel.asof(..., dtype=float64)`（绕开 float32 精度）；
市值/股本走价格层状态量。模块内没有 shift / 前向填充 / 全样本统计量。

★ 依赖 `research/families/r12gc_common.py`（网格）。
  catalog 只对**本文件**取 sha256，公共文件改动等价于定义变更 —— 须重新注册。
"""
from __future__ import annotations

import numpy as np

from research.families.r12gc_common import mktcap, single_quarter, table

FROZEN = {
    "eps_q": "r12gc_eps_q_v1",
    "gpm_q": "r12gc_gpm_q_v1",
    "npm_q": "r12gc_npm_q_v1",
    "roa_q": "r12gc_roa_q_v1",
    "opt_tpro": "r12gc_opt_tpro_v1",
    "sales_to_market": "r12gc_sales_to_market_v1",
}

MIN_ABS_DEN = 1e6


def build(parameters):
    variant = parameters.get("variant") if isinstance(parameters, dict) else None
    if variant not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))

    def compute(ctx):
        cache = {}
        if variant == "eps_q":
            s = single_quarter(ctx, table(ctx, cache, "stock_income", "n_income_attr_p"), 0)
            return ctx.safe_div(s, np.asarray(ctx.point("total_share"), dtype=np.float64), MIN_ABS_DEN)
        if variant == "gpm_q":
            rev = single_quarter(ctx, table(ctx, cache, "stock_income", "revenue"), 0)
            cost = single_quarter(ctx, table(ctx, cache, "stock_income", "oper_cost"), 0)
            return ctx.safe_div(rev - cost, rev, MIN_ABS_DEN)
        if variant == "npm_q":
            npf = single_quarter(ctx, table(ctx, cache, "stock_income", "n_income_attr_p"), 0)
            rev = single_quarter(ctx, table(ctx, cache, "stock_income", "revenue"), 0)
            return ctx.safe_div(npf, rev, MIN_ABS_DEN)
        if variant == "roa_q":
            npf = single_quarter(ctx, table(ctx, cache, "stock_income", "n_income_attr_p"), 0)
            return ctx.safe_div(npf, np.asarray(ctx.point("total_assets"), dtype=np.float64), MIN_ABS_DEN)
        if variant == "opt_tpro":
            op = single_quarter(ctx, table(ctx, cache, "stock_income", "operate_profit"), 0)
            tp = single_quarter(ctx, table(ctx, cache, "stock_income", "total_profit"), 0)
            return ctx.safe_div(op, tp, MIN_ABS_DEN)
        # sales_to_market
        rev = single_quarter(ctx, table(ctx, cache, "stock_income", "revenue"), 0)
        return ctx.safe_div(rev, mktcap(ctx, "total_share"), MIN_ABS_DEN)

    return compute
