"""R12 绿卡批 A · 年度口径六件套（family `r12gc_fund_annual_v1`）。

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-06 绿卡通道扩容裁定：三项基础门槛）
════════════════════════════════════════════════════════════════════════
  eps_y  年度每股收益     ctx.annual('stock_income','basic_eps',0)（已声明年度别名直取）
  gpm_y  年度毛利率       (AN(revenue) − AN(oper_cost)) / AN(revenue)
  npm_y  年度净利率       AN(归母净利) / AN(revenue)
  opm_y  年度营业利润率   AN(operate_profit) / AN(revenue)
  roa_y  年度总资产收益率 AN(归母净利) / AN(total_assets)
  roe_y  年度净资产收益率 AN(归母净利) / AN(归母股东权益)

AN(f) = 该股**当时已公告的最新 12/31 年报**的 FY 值。年度快照在两次年报之间为常数。

★★ 本批对象是**绿卡候选（只主张三项基础门槛；G4–G7 未主张）**。
   `higher_is_better=true` 只表示 declared=raw（不做任何反号），不代表任何预测方向。

════════════════════════════════════════════════════════════════════════
二、两条实现渠道（README 边界案例 2）
════════════════════════════════════════════════════════════════════════
· **eps_y 走已声明别名**：`annual__stock_income__basic_eps` 已在
  `conf/field_expansion.json`（afx_is_basic_eps 同源），`ctx.annual` 逐字给供应商
  BasicEPS 的年度报告披露值（含加权平均股本与追溯重述）⇒ **零 conf 变更**。
· **其余五名走自建年度网格**：`r12gc_common.quarter_table(..., annual_only=True)`
  只保留 12/31 年报（ann_date 锚定、保留修订、前缀最大报告期保证季度披露不会把
  年报推走），`annual_value()` 取当时最新年报的 FY 值 ⇒ 同样**零 conf 变更**。
  替代路线（在 conf/field_expansion.json 新增 6 条年度源字段声明后走 ctx.annual）
  留待主裁判裁决；若要求 eps 三后缀完全同构（attr_p ÷ 期末 total_share），
  eps_y 可切换为网格路线。
· 分子分母同取**同一份年报**（roa_y/roe_y 的资产负债表分母 = 该 12/31 报告的
  时点值），不用「最新季度时点」混合口径。分母一律 safe_div(min 1e6)。
· 净利口径 = 归母 `n_income_attr_p`；roe_y 分母取**归母权益**
  `total_hldr_eqy_exc_min_int`（与在产 roe_ttm 完全同对：ttm(归母NP)/point(归母权益)）；
  与 F1 的 de/equity_turnover（分母=所有者权益合计）的口径差已列入 README 不确定项。
  期末 vs 平均权益：供应商原文无「平均」字样 ⇒ 取期末（平均版本列 README 不确定项）。

════════════════════════════════════════════════════════════════════════
三、PIT / 无泄露
════════════════════════════════════════════════════════════════════════
年报只从实际公告时点（ann_date/f_ann_date 版本表）生效；追溯重述不改写历史生效日；
「最新年报」用前缀最大报告期选择，旧年报的迟到修订不会把新年度数据顶掉。
模块内没有 shift / 前向填充 / 全样本统计量。

★ 依赖 `research/families/r12gc_common.py`（年度网格）。
  catalog 只对**本文件**取 sha256，公共文件改动等价于定义变更 —— 须重新注册。
"""
from __future__ import annotations

import numpy as np

from research.families.r12gc_common import annual_value, table

FROZEN = {
    "eps_y": "r12gc_eps_y_v1",
    "gpm_y": "r12gc_gpm_y_v1",
    "npm_y": "r12gc_npm_y_v1",
    "opm_y": "r12gc_opm_y_v1",
    "roa_y": "r12gc_roa_y_v1",
    "roe_y": "r12gc_roe_y_v1",
}

MIN_ABS_DEN = 1e6


def build(parameters):
    variant = parameters.get("variant") if isinstance(parameters, dict) else None
    if variant not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))

    def compute(ctx):
        if variant == "eps_y":
            # 已声明年度别名（conf/field_expansion.json 的 afx_is_basic_eps 同源）。
            return np.asarray(ctx.annual("stock_income", "basic_eps"), dtype=np.float64)
        cache = {}

        def an(dataset, field):
            return annual_value(ctx, table(ctx, cache, dataset, field, annual_only=True))

        if variant == "gpm_y":
            rev = an("stock_income", "revenue")
            cost = an("stock_income", "oper_cost")
            return ctx.safe_div(rev - cost, rev, MIN_ABS_DEN)
        if variant == "npm_y":
            return ctx.safe_div(an("stock_income", "n_income_attr_p"),
                                an("stock_income", "revenue"), MIN_ABS_DEN)
        if variant == "opm_y":
            return ctx.safe_div(an("stock_income", "operate_profit"),
                                an("stock_income", "revenue"), MIN_ABS_DEN)
        if variant == "roa_y":
            return ctx.safe_div(an("stock_income", "n_income_attr_p"),
                                an("stock_balancesheet", "total_assets"), MIN_ABS_DEN)
        # roe_y
        return ctx.safe_div(an("stock_income", "n_income_attr_p"),
                            an("stock_balancesheet", "total_hldr_eqy_exc_min_int"), MIN_ABS_DEN)

    return compute
