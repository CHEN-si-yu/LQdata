"""R12 绿卡批 D · 现金流量表白名单枚举

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-06 绿卡通道扩容裁定：三项基础门槛）
════════════════════════════════════════════════════════════════════════
本模块是 **批 D 财务白名单系统枚举**（worklist 381 行 → IN 46 行）的 r12gd_cashflow_v1 族实现。
每一个变体都是**白名单原始科目序列的标准变换**或**两原始量的比率**：
· `lvl_*`   —— 水平量，`arcsinh` 单调压缩（金额/比率水平；单调保序、无未来信息）；
· `ratio`   —— 分子 ÷ point(total_assets) 或 ttm(revenue) 的比率，分母 `ctx.safe_div(..., 1e6)` 近零保护；
· `ind`/`annual_*` —— 供应商指标表安全字段 / 已公告年报值（见下）。

逐名资格依据、差异论证与边界案例裁决：`../worklist.json` 同名行 + `../README.md` §7。
本批对象是**绿卡候选（只主张三项基础门槛：非空 / 非常数 / 无泄露；G4–G7 未主张）**。
`higher_is_better=true` 只表示 declared=raw（不做任何反号），不代表任何预测方向。

════════════════════════════════════════════════════════════════════════
二、口径与 PIT / 无泄露
════════════════════════════════════════════════════════════════════════
· 日频分子/分母走 `ctx.ttm()` / `ctx.point()` 的 fea/deriv.py **版本表**：
  T 日只用 `pit = max(ann_date, f_ann_date, end_date) <= T` 的最新已披露版本，
  追溯修订不改写历史生效日；TTM 按四季滚动（三行式，中间季度缺失不影响）。
  POSITIVE_ONLY 字段（total_assets 等）≤0 ⇒ NaN 由版本表处理。
· 分母一律 `ctx.safe_div(..., min_abs_den=1e6)`：近零分母 ⇒ NaN，不产生巨值假值。
· 模块内**没有** shift / 前向填充 / 全样本统计量；缺失一律 NaN，不补零。

· 现金流科目（经营/投资/筹资净额及其分项）可为负，水平量 arcsinh 保留符号。

★ 哈希口径提醒：`research/catalog.py::implementation` 只对**本模块文件**取 sha256；
  修改本文件等价于定义变更，必须重新注册并留痕。
"""
from __future__ import annotations

import numpy as np

MIN_ABS_DEN = 1e6

FROZEN = {
    # ---- r12gd_cashflow_v1：46 变体（逐条对应 candidate_*.yaml 的 variant）
    'amort_intang_assets__lvl': ('lvl_ttm', 'amort_intang_assets'),  # asinh(ttm(amort_intang_assets))
    'amort_intang_assets__ta': ('ratio', 'ttm', 'amort_intang_assets', 'point', 'total_assets'),  # safe_div(ttm(amort_intang_assets), point(total_assets), 1e6)
    'amort_intang_assets__tr': ('ratio', 'ttm', 'amort_intang_assets', 'ttm', 'revenue'),  # safe_div(ttm(amort_intang_assets), ttm(revenue), 1e6)
    'c_fr_sale_sg__lvl': ('lvl_ttm', 'c_fr_sale_sg'),  # asinh(ttm(c_fr_sale_sg))
    'c_fr_sale_sg__ta': ('ratio', 'ttm', 'c_fr_sale_sg', 'point', 'total_assets'),  # safe_div(ttm(c_fr_sale_sg), point(total_assets), 1e6)
    'c_paid_for_taxes__lvl': ('lvl_ttm', 'c_paid_for_taxes'),  # asinh(ttm(c_paid_for_taxes))
    'c_paid_for_taxes__ta': ('ratio', 'ttm', 'c_paid_for_taxes', 'point', 'total_assets'),  # safe_div(ttm(c_paid_for_taxes), point(total_assets), 1e6)
    'c_paid_goods_s__lvl': ('lvl_ttm', 'c_paid_goods_s'),  # asinh(ttm(c_paid_goods_s))
    'c_paid_goods_s__ta': ('ratio', 'ttm', 'c_paid_goods_s', 'point', 'total_assets'),  # safe_div(ttm(c_paid_goods_s), point(total_assets), 1e6)
    'c_paid_to_for_empl__lvl': ('lvl_ttm', 'c_paid_to_for_empl'),  # asinh(ttm(c_paid_to_for_empl))
    'c_paid_to_for_empl__ta': ('ratio', 'ttm', 'c_paid_to_for_empl', 'point', 'total_assets'),  # safe_div(ttm(c_paid_to_for_empl), point(total_assets), 1e6)
    'c_paid_to_for_empl__tr': ('ratio', 'ttm', 'c_paid_to_for_empl', 'ttm', 'revenue'),  # safe_div(ttm(c_paid_to_for_empl), ttm(revenue), 1e6)
    'c_pay_acq_const_fiolta__lvl': ('lvl_ttm', 'c_pay_acq_const_fiolta'),  # asinh(ttm(c_pay_acq_const_fiolta))
    'c_pay_acq_const_fiolta__ta': ('ratio', 'ttm', 'c_pay_acq_const_fiolta', 'point', 'total_assets'),  # safe_div(ttm(c_pay_acq_const_fiolta), point(total_assets), 1e6)
    'c_pay_acq_const_fiolta__tr': ('ratio', 'ttm', 'c_pay_acq_const_fiolta', 'ttm', 'revenue'),  # safe_div(ttm(c_pay_acq_const_fiolta), ttm(revenue), 1e6)
    'c_pay_dist_dpcp_int_exp__lvl': ('lvl_ttm', 'c_pay_dist_dpcp_int_exp'),  # asinh(ttm(c_pay_dist_dpcp_int_exp))
    'c_pay_dist_dpcp_int_exp__ta': ('ratio', 'ttm', 'c_pay_dist_dpcp_int_exp', 'point', 'total_assets'),  # safe_div(ttm(c_pay_dist_dpcp_int_exp), point(total_assets), 1e6)
    'c_pay_dist_dpcp_int_exp__tr': ('ratio', 'ttm', 'c_pay_dist_dpcp_int_exp', 'ttm', 'revenue'),  # safe_div(ttm(c_pay_dist_dpcp_int_exp), ttm(revenue), 1e6)
    'c_prepay_amt_borr__lvl': ('lvl_ttm', 'c_prepay_amt_borr'),  # asinh(ttm(c_prepay_amt_borr))
    'c_prepay_amt_borr__ta': ('ratio', 'ttm', 'c_prepay_amt_borr', 'point', 'total_assets'),  # safe_div(ttm(c_prepay_amt_borr), point(total_assets), 1e6)
    'c_recp_borrow__lvl': ('lvl_ttm', 'c_recp_borrow'),  # asinh(ttm(c_recp_borrow))
    'c_recp_borrow__ta': ('ratio', 'ttm', 'c_recp_borrow', 'point', 'total_assets'),  # safe_div(ttm(c_recp_borrow), point(total_assets), 1e6)
    'credit_impa_loss__lvl': ('lvl_ttm', 'credit_impa_loss'),  # asinh(ttm(credit_impa_loss))
    'credit_impa_loss__ta': ('ratio', 'ttm', 'credit_impa_loss', 'point', 'total_assets'),  # safe_div(ttm(credit_impa_loss), point(total_assets), 1e6)
    'credit_impa_loss__tr': ('ratio', 'ttm', 'credit_impa_loss', 'ttm', 'revenue'),  # safe_div(ttm(credit_impa_loss), ttm(revenue), 1e6)
    'decr_inventories__lvl': ('lvl_ttm', 'decr_inventories'),  # asinh(ttm(decr_inventories))
    'decr_inventories__ta': ('ratio', 'ttm', 'decr_inventories', 'point', 'total_assets'),  # safe_div(ttm(decr_inventories), point(total_assets), 1e6)
    'depr_fa_coga_dpba__lvl': ('lvl_ttm', 'depr_fa_coga_dpba'),  # asinh(ttm(depr_fa_coga_dpba))
    'depr_fa_coga_dpba__ta': ('ratio', 'ttm', 'depr_fa_coga_dpba', 'point', 'total_assets'),  # safe_div(ttm(depr_fa_coga_dpba), point(total_assets), 1e6)
    'depr_fa_coga_dpba__tr': ('ratio', 'ttm', 'depr_fa_coga_dpba', 'ttm', 'revenue'),  # safe_div(ttm(depr_fa_coga_dpba), ttm(revenue), 1e6)
    'lt_amort_deferred_exp__lvl': ('lvl_ttm', 'lt_amort_deferred_exp'),  # asinh(ttm(lt_amort_deferred_exp))
    'lt_amort_deferred_exp__ta': ('ratio', 'ttm', 'lt_amort_deferred_exp', 'point', 'total_assets'),  # safe_div(ttm(lt_amort_deferred_exp), point(total_assets), 1e6)
    'lt_amort_deferred_exp__tr': ('ratio', 'ttm', 'lt_amort_deferred_exp', 'ttm', 'revenue'),  # safe_div(ttm(lt_amort_deferred_exp), ttm(revenue), 1e6)
    'n_cash_flows_fnc_act__lvl': ('lvl_ttm', 'n_cash_flows_fnc_act'),  # asinh(ttm(n_cash_flows_fnc_act))
    'n_cash_flows_fnc_act__ta': ('ratio', 'ttm', 'n_cash_flows_fnc_act', 'point', 'total_assets'),  # safe_div(ttm(n_cash_flows_fnc_act), point(total_assets), 1e6)
    'n_cashflow_act__lvl': ('lvl_ttm', 'n_cashflow_act'),  # asinh(ttm(n_cashflow_act))
    'n_cashflow_act__ta': ('ratio', 'ttm', 'n_cashflow_act', 'point', 'total_assets'),  # safe_div(ttm(n_cashflow_act), point(total_assets), 1e6)
    'n_cashflow_inv_act__lvl': ('lvl_ttm', 'n_cashflow_inv_act'),  # asinh(ttm(n_cashflow_inv_act))
    'n_cashflow_inv_act__ta': ('ratio', 'ttm', 'n_cashflow_inv_act', 'point', 'total_assets'),  # safe_div(ttm(n_cashflow_inv_act), point(total_assets), 1e6)
    'n_incr_cash_cash_equ__lvl': ('lvl_ttm', 'n_incr_cash_cash_equ'),  # asinh(ttm(n_incr_cash_cash_equ))
    'n_incr_cash_cash_equ__ta': ('ratio', 'ttm', 'n_incr_cash_cash_equ', 'point', 'total_assets'),  # safe_div(ttm(n_incr_cash_cash_equ), point(total_assets), 1e6)
    'prov_depr_assets__lvl': ('lvl_ttm', 'prov_depr_assets'),  # asinh(ttm(prov_depr_assets))
    'prov_depr_assets__ta': ('ratio', 'ttm', 'prov_depr_assets', 'point', 'total_assets'),  # safe_div(ttm(prov_depr_assets), point(total_assets), 1e6)
    'prov_depr_assets__tr': ('ratio', 'ttm', 'prov_depr_assets', 'ttm', 'revenue'),  # safe_div(ttm(prov_depr_assets), ttm(revenue), 1e6)
    'recp_tax_rends__lvl': ('lvl_ttm', 'recp_tax_rends'),  # asinh(ttm(recp_tax_rends))
    'recp_tax_rends__ta': ('ratio', 'ttm', 'recp_tax_rends', 'point', 'total_assets'),  # safe_div(ttm(recp_tax_rends), point(total_assets), 1e6)
}


def build(parameters):
    variant = parameters.get("variant") if isinstance(parameters, dict) else None
    if variant not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))
    spec = FROZEN[variant]

    def compute(ctx):
        kind = spec[0]
        if kind == "lvl_ttm":
            return np.arcsinh(np.asarray(ctx.ttm(spec[1]), dtype=np.float64))
        if kind == "lvl_point":
            return np.arcsinh(np.asarray(ctx.point(spec[1]), dtype=np.float64))
        if kind == "ratio":
            _, nm, nf, dm, df = spec
            num = np.asarray(ctx.ttm(nf) if nm == "ttm" else ctx.point(nf), dtype=np.float64)
            den = np.asarray(ctx.ttm(df) if dm == "ttm" else ctx.point(df), dtype=np.float64)
            return ctx.safe_div(num, den, MIN_ABS_DEN)
        if kind == "ind":
            return np.asarray(ctx.ind(spec[1]), dtype=np.float64)
        raise AssertionError("unreachable spec kind: %r" % (spec,))

    return compute
