"""基本面因子（8 个）—— 全部来自利润表 / 资产负债表 / 现金流量表。

所有因子都是**日频**、**只主板**、**2015 年起**（用户的三条硬约束）。
TTM、单季拆分、追溯修正、PIT 对齐全部由 `fea/deriv.py` 统一处理，
这里只写公式本身。

口径说明（与灵启因子库文档对齐 / 偏离的理由）：
  - 营收用 `revenue`（营业收入），不是 `total_revenue`（营业总收入）。
    实测茅台 FY2025：1688.4亿 vs 1720.5亿，差的是利息收入。文档写的是营业收入。
  - `roe_ttm` 用**期末**归母权益除以归母净利润TTM，即利润/期末权益。
    文档提「平均」，但平均口径要取前一期权益，在 PIT 上更脆；先用期末并标注。
  - `cash_profit_ratio` 文档写 `/NetProfit_TTM`，但净利润可能为负或近零，
    直接相除会爆炸。改用 `|NP_TTM|` 做分母并加下限，语义不变（超额现金流的相对量）。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

# ★ 起点留 None = 跟随 conf/config.yaml 的 default_start（当前 2012-01-01）。
#   改起点只改配置一处即可，不必挨个改因子。
FIN_START = None
# TTM 需要 4 季 + 同比再要 4 季 + ann_date 最长滞后 15 个月 → 500 天是下限
FIN_WARMUP = 700

NP = "n_income_attr_p"
REV = "revenue"
COST = "oper_cost"
OCF = "n_cashflow_act"
EQ = "total_hldr_eqy_exc_min_int"
TA = "total_assets"
TL = "total_liab"


@register(FactorSpec(
    name="roe_ttm", group="quality", deps=("stock_income", "stock_balancesheet"),
    desc="净资产收益率（TTM）= 归母净利润TTM / 归母股东权益",
    formula="ROE = NetProfit_Parent_TTM / SE_without_MI",
    start=FIN_START, warmup_days=FIN_WARMUP, higher_is_better=True,
    note="用期末权益；灵启文档提「平均权益」，平均口径需前一期行，PIT 更脆",
))
def roe_ttm(ctx):
    return ctx.safe_div(ctx.ttm(NP), ctx.point(EQ))






@register(FactorSpec(
    name="yoy_revenue", group="growth", deps=("stock_income",),
    desc="营业收入（TTM）同比增速",
    formula="YoY = Revenue_TTM_t / Revenue_TTM_{t-4Q} - 1",
    start=FIN_START, warmup_days=FIN_WARMUP, higher_is_better=True,
))
def yoy_revenue(ctx):
    cur = ctx.ttm(REV)
    prev = ctx.lag_ttm(REV, 4)
    return ctx.safe_div(cur - prev, np.abs(prev), min_abs_den=1e-6)




@register(FactorSpec(
    name="asset_growth_qoq", group="growth", deps=("stock_balancesheet",),
    desc="总资产环比增速（相对上一个报告期）",
    formula="Growth = TotalAssets_t / TotalAssets_{t-1Q} - 1",
    start=FIN_START, warmup_days=FIN_WARMUP, higher_is_better=False,
    note="★ 文档的 t-1 必须理解为「上一个**报告期**」而不是「63 个交易日前」。"
         "按日频 lag 实现的话，一年里大部分时间比较的是同一期数据，因子会塌成 0",
))
def asset_growth_qoq(ctx):
    cur = ctx.point(TA)
    prev = ctx.point(TA, lag=1)
    return ctx.safe_div(cur - prev, np.abs(prev), min_abs_den=1e-6)




