"""R12 绿卡批 A · 财务 TTM 比率九件套（family `r12gc_fund_ratio_ttm_v1`）。

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-06 绿卡通道扩容裁定：三项基础门槛）
════════════════════════════════════════════════════════════════════════
九个对象都是**两条白名单原始科目序列 TTM/时点值的一次比率**（周转率、利润率、
每股收益、负债权益比）—— 规则明文 IN 的「对单序列原始字段的标准变换 / 两原始
量的比率」，没有对已聚合比率的再运算、没有同比/环比、没有事件构造。

  asset_turnover        总资产周转率   revenue_TTM / total_assets（时点）
  equity_turnover       股东权益周转率 revenue_TTM / 所有者权益合计（时点）
  receivable_turnover   应收账款周转率 revenue_TTM / accounts_receiv（时点）
  fixed_asset_turnover  固定资产周转率 revenue_TTM / fix_assets（时点）
  gpm_ttm               毛利率         (revenue_TTM − oper_cost_TTM) / revenue_TTM
  npm_ttm               净利率         n_income_attr_p_TTM / revenue_TTM
  opm_ttm               营业利润率     operate_profit_TTM / revenue_TTM
  eps_ttm               每股收益(TTM)  n_income_attr_p_TTM / total_share（时点）
  de                    负债权益比     total_liab（时点）/ 所有者权益合计（时点）

★★ 本批对象是**绿卡候选（只主张三项基础门槛：非空 / 非常数 / 无泄露；G4–G7 未主张）**。
   `higher_is_better=true` 只表示 declared=raw（不做任何反号），**不代表任何预测方向**。

════════════════════════════════════════════════════════════════════════
二、口径选择（与同批 YAML 逐条对应；争议点已列 README 边界案例）
════════════════════════════════════════════════════════════════════════
· 分子分母配对统一用 `revenue`（营业收入）而非 `total_revenue`（营业总收入）：
  `fea/deriv.py` 注 3 明确「灵启文档写的是『营业收入』，所以统一用 revenue」
  （实测差利息收入 ~2%）。oper_cost 与之成对；银行/券商的 `oper_cost` 恒为精确 0
  已被 POSITIVE_ONLY 置 NaN，不会出现「毛利率 100%」的金融股假值。
· 净利口径 = **归母** `n_income_attr_p`（三条论据：在产 roa_ttm/roe_ttm 用 attr_p；
  已删 opm_npm_spread 用 attr_p；族内一致）。含少数版 `n_income` 未采用。
· 权益分母：`equity_turnover`/`de` 用**所有者权益合计** `total_hldr_eqy_inc_min_int`
  （供应商在需要指「归母」时明确写 SEWithoutMI，故 TotalShareholderEquity = 合计；
  与 financial_leverage 同分母口径）；`eps_ttm` 分母取**期末** total_share
  （point 时点值，PIT 可复现；供应商 BasicEPS 实为加权平均股本，偏差已披露）。
· 分母一律 `ctx.safe_div(..., min_abs_den=1e6)`：近零分母 → NaN，不产生 ±1e6 假值。
  `revenue`/`oper_cost` 在 POSITIVE_ONLY（<=0 → NaN）；total_hldr_eqy_inc_min_int 亦在
  POSITIVE_ONLY ⇒ 负权益公司缺席（与在产 bp 同处理）。

════════════════════════════════════════════════════════════════════════
三、PIT / 无泄露
════════════════════════════════════════════════════════════════════════
`ctx.ttm(field)` / `ctx.point(field, lag=0)` 走 `fea/deriv.py::Derivative` 的**版本表**：
T 日只用 `pit = max(ann_date, f_ann_date, end_date) <= T` 的最新已披露版本，
追溯修订不改写历史生效日；TTM 按四季滚动（中间季度缺失不影响三行式）。
模块内**没有** shift / 前向填充 / 全样本统计量。
本模块不依赖 `r12gc_common.py`（纯 ttm/point 比率，无需网格）。
"""
from __future__ import annotations

import numpy as np

FROZEN = {
    "asset_turnover": "r12gc_asset_turnover_v1",
    "equity_turnover": "r12gc_equity_turnover_v1",
    "receivable_turnover": "r12gc_receivable_turnover_v1",
    "fixed_asset_turnover": "r12gc_fixed_asset_turnover_v1",
    "gpm_ttm": "r12gc_gpm_ttm_v1",
    "npm_ttm": "r12gc_npm_ttm_v1",
    "opm_ttm": "r12gc_opm_ttm_v1",
    "eps_ttm": "r12gc_eps_ttm_v1",
    "de": "r12gc_de_v1",
}

MIN_ABS_DEN = 1e6  # 分母地板：金额（元）/ 股本（股）都必须大于 1e6 才有意义


def build(parameters):
    variant = parameters.get("variant") if isinstance(parameters, dict) else None
    if variant not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))

    def compute(ctx):
        if variant == "asset_turnover":
            num, den = ctx.ttm("revenue"), ctx.point("total_assets")
        elif variant == "equity_turnover":
            num, den = ctx.ttm("revenue"), ctx.point("total_hldr_eqy_inc_min_int")
        elif variant == "receivable_turnover":
            num, den = ctx.ttm("revenue"), ctx.point("accounts_receiv")
        elif variant == "fixed_asset_turnover":
            num, den = ctx.ttm("revenue"), ctx.point("fix_assets")
        elif variant == "gpm_ttm":
            rev = np.asarray(ctx.ttm("revenue"), dtype=np.float64)
            return ctx.safe_div(rev - np.asarray(ctx.ttm("oper_cost"), dtype=np.float64), rev, MIN_ABS_DEN)
        elif variant == "npm_ttm":
            num, den = ctx.ttm("n_income_attr_p"), ctx.ttm("revenue")
        elif variant == "opm_ttm":
            num, den = ctx.ttm("operate_profit"), ctx.ttm("revenue")
        elif variant == "eps_ttm":
            num, den = ctx.ttm("n_income_attr_p"), ctx.point("total_share")
        else:  # de
            num, den = ctx.point("total_liab"), ctx.point("total_hldr_eqy_inc_min_int")
        return ctx.safe_div(np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64), MIN_ABS_DEN)

    return compute
