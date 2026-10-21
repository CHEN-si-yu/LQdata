"""R12 绿卡批 D · 利润表白名单枚举

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-06 绿卡通道扩容裁定：三项基础门槛）
════════════════════════════════════════════════════════════════════════
本模块是 **批 D 财务白名单系统枚举**（worklist 381 行 → IN 49 行）的 r12gd_income_v1 族实现。
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

年报专用口径见 annual 族；本族无 annual/ind 变体。

★ 哈希口径提醒：`research/catalog.py::implementation` 只对**本模块文件**取 sha256；
  修改本文件等价于定义变更，必须重新注册并留痕。
"""
from __future__ import annotations

import numpy as np

MIN_ABS_DEN = 1e6

FROZEN = {
    # ---- r12gd_income_v1：49 变体（逐条对应 candidate_*.yaml 的 variant）
    'admin_exp__lvl': ('lvl_ttm', 'admin_exp'),  # asinh(ttm(admin_exp))
    'admin_exp__ta': ('ratio', 'ttm', 'admin_exp', 'point', 'total_assets'),  # safe_div(ttm(admin_exp), point(total_assets), 1e6)
    'admin_exp__tr': ('ratio', 'ttm', 'admin_exp', 'ttm', 'revenue'),  # safe_div(ttm(admin_exp), ttm(revenue), 1e6)
    'assets_impair_loss__lvl': ('lvl_ttm', 'assets_impair_loss'),  # asinh(ttm(assets_impair_loss))
    'assets_impair_loss__ta': ('ratio', 'ttm', 'assets_impair_loss', 'point', 'total_assets'),  # safe_div(ttm(assets_impair_loss), point(total_assets), 1e6)
    'assets_impair_loss__tr': ('ratio', 'ttm', 'assets_impair_loss', 'ttm', 'revenue'),  # safe_div(ttm(assets_impair_loss), ttm(revenue), 1e6)
    'ebit__lvl': ('lvl_ttm', 'ebit'),  # asinh(ttm(ebit))
    'ebit__ta': ('ratio', 'ttm', 'ebit', 'point', 'total_assets'),  # safe_div(ttm(ebit), point(total_assets), 1e6)
    'ebit__tr': ('ratio', 'ttm', 'ebit', 'ttm', 'revenue'),  # safe_div(ttm(ebit), ttm(revenue), 1e6)
    'fin_exp__lvl': ('lvl_ttm', 'fin_exp'),  # asinh(ttm(fin_exp))
    'fin_exp__ta': ('ratio', 'ttm', 'fin_exp', 'point', 'total_assets'),  # safe_div(ttm(fin_exp), point(total_assets), 1e6)
    'fin_exp__tr': ('ratio', 'ttm', 'fin_exp', 'ttm', 'revenue'),  # safe_div(ttm(fin_exp), ttm(revenue), 1e6)
    'fin_exp_int_exp__lvl': ('lvl_ttm', 'fin_exp_int_exp'),  # asinh(ttm(fin_exp_int_exp))
    'fin_exp_int_exp__ta': ('ratio', 'ttm', 'fin_exp_int_exp', 'point', 'total_assets'),  # safe_div(ttm(fin_exp_int_exp), point(total_assets), 1e6)
    'fin_exp_int_exp__tr': ('ratio', 'ttm', 'fin_exp_int_exp', 'ttm', 'revenue'),  # safe_div(ttm(fin_exp_int_exp), ttm(revenue), 1e6)
    'fv_value_chg_gain__lvl': ('lvl_ttm', 'fv_value_chg_gain'),  # asinh(ttm(fv_value_chg_gain))
    'fv_value_chg_gain__ta': ('ratio', 'ttm', 'fv_value_chg_gain', 'point', 'total_assets'),  # safe_div(ttm(fv_value_chg_gain), point(total_assets), 1e6)
    'fv_value_chg_gain__tr': ('ratio', 'ttm', 'fv_value_chg_gain', 'ttm', 'revenue'),  # safe_div(ttm(fv_value_chg_gain), ttm(revenue), 1e6)
    'income_tax__lvl': ('lvl_ttm', 'income_tax'),  # asinh(ttm(income_tax))
    'income_tax__ta': ('ratio', 'ttm', 'income_tax', 'point', 'total_assets'),  # safe_div(ttm(income_tax), point(total_assets), 1e6)
    'income_tax__tr': ('ratio', 'ttm', 'income_tax', 'ttm', 'revenue'),  # safe_div(ttm(income_tax), ttm(revenue), 1e6)
    'invest_income__lvl': ('lvl_ttm', 'invest_income'),  # asinh(ttm(invest_income))
    'invest_income__ta': ('ratio', 'ttm', 'invest_income', 'point', 'total_assets'),  # safe_div(ttm(invest_income), point(total_assets), 1e6)
    'invest_income__tr': ('ratio', 'ttm', 'invest_income', 'ttm', 'revenue'),  # safe_div(ttm(invest_income), ttm(revenue), 1e6)
    'minority_gain__lvl': ('lvl_ttm', 'minority_gain'),  # asinh(ttm(minority_gain))
    'minority_gain__ta': ('ratio', 'ttm', 'minority_gain', 'point', 'total_assets'),  # safe_div(ttm(minority_gain), point(total_assets), 1e6)
    'n_income_attr_p__lvl': ('lvl_ttm', 'n_income_attr_p'),  # asinh(ttm(n_income_attr_p))
    'non_oper_exp__lvl': ('lvl_ttm', 'non_oper_exp'),  # asinh(ttm(non_oper_exp))
    'non_oper_exp__ta': ('ratio', 'ttm', 'non_oper_exp', 'point', 'total_assets'),  # safe_div(ttm(non_oper_exp), point(total_assets), 1e6)
    'non_oper_exp__tr': ('ratio', 'ttm', 'non_oper_exp', 'ttm', 'revenue'),  # safe_div(ttm(non_oper_exp), ttm(revenue), 1e6)
    'non_oper_income__lvl': ('lvl_ttm', 'non_oper_income'),  # asinh(ttm(non_oper_income))
    'non_oper_income__ta': ('ratio', 'ttm', 'non_oper_income', 'point', 'total_assets'),  # safe_div(ttm(non_oper_income), point(total_assets), 1e6)
    'non_oper_income__tr': ('ratio', 'ttm', 'non_oper_income', 'ttm', 'revenue'),  # safe_div(ttm(non_oper_income), ttm(revenue), 1e6)
    'oper_cost__lvl': ('lvl_ttm', 'oper_cost'),  # asinh(ttm(oper_cost))
    'oper_cost__ta': ('ratio', 'ttm', 'oper_cost', 'point', 'total_assets'),  # safe_div(ttm(oper_cost), point(total_assets), 1e6)
    'operate_profit__lvl': ('lvl_ttm', 'operate_profit'),  # asinh(ttm(operate_profit))
    'operate_profit__ta': ('ratio', 'ttm', 'operate_profit', 'point', 'total_assets'),  # safe_div(ttm(operate_profit), point(total_assets), 1e6)
    'rd_exp__lvl': ('lvl_ttm', 'rd_exp'),  # asinh(ttm(rd_exp))
    'rd_exp__ta': ('ratio', 'ttm', 'rd_exp', 'point', 'total_assets'),  # safe_div(ttm(rd_exp), point(total_assets), 1e6)
    'revenue__lvl': ('lvl_ttm', 'revenue'),  # asinh(ttm(revenue))
    'sell_exp__lvl': ('lvl_ttm', 'sell_exp'),  # asinh(ttm(sell_exp))
    'sell_exp__ta': ('ratio', 'ttm', 'sell_exp', 'point', 'total_assets'),  # safe_div(ttm(sell_exp), point(total_assets), 1e6)
    'sell_exp__tr': ('ratio', 'ttm', 'sell_exp', 'ttm', 'revenue'),  # safe_div(ttm(sell_exp), ttm(revenue), 1e6)
    'total_cogs__lvl': ('lvl_ttm', 'total_cogs'),  # asinh(ttm(total_cogs))
    'total_cogs__ta': ('ratio', 'ttm', 'total_cogs', 'point', 'total_assets'),  # safe_div(ttm(total_cogs), point(total_assets), 1e6)
    'total_cogs__tr': ('ratio', 'ttm', 'total_cogs', 'ttm', 'revenue'),  # safe_div(ttm(total_cogs), ttm(revenue), 1e6)
    'total_profit__lvl': ('lvl_ttm', 'total_profit'),  # asinh(ttm(total_profit))
    'total_profit__ta': ('ratio', 'ttm', 'total_profit', 'point', 'total_assets'),  # safe_div(ttm(total_profit), point(total_assets), 1e6)
    'total_profit__tr': ('ratio', 'ttm', 'total_profit', 'ttm', 'revenue'),  # safe_div(ttm(total_profit), ttm(revenue), 1e6)
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
