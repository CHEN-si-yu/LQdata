"""R12 绿卡批 D · 资产负债表白名单枚举

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-06 绿卡通道扩容裁定：三项基础门槛）
════════════════════════════════════════════════════════════════════════
本模块是 **批 D 财务白名单系统枚举**（worklist 381 行 → IN 71 行）的 r12gd_bs_v1 族实现。
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

· 资产负债表科目取**期末时点值**（point），不做 TTM 累计。

★ 哈希口径提醒：`research/catalog.py::implementation` 只对**本模块文件**取 sha256；
  修改本文件等价于定义变更，必须重新注册并留痕。
"""
from __future__ import annotations

import numpy as np

MIN_ABS_DEN = 1e6

FROZEN = {
    # ---- r12gd_bs_v1：71 变体（逐条对应 candidate_*.yaml 的 variant）
    'accounts_receiv__lvl': ('lvl_point', 'accounts_receiv'),  # asinh(point(accounts_receiv))
    'accounts_receiv__ta': ('ratio', 'point', 'accounts_receiv', 'point', 'total_assets'),  # safe_div(point(accounts_receiv), point(total_assets), 1e6)
    'acct_payable__lvl': ('lvl_point', 'acct_payable'),  # asinh(point(acct_payable))
    'acct_payable__ta': ('ratio', 'point', 'acct_payable', 'point', 'total_assets'),  # safe_div(point(acct_payable), point(total_assets), 1e6)
    'acct_payable__tr': ('ratio', 'point', 'acct_payable', 'ttm', 'revenue'),  # safe_div(point(acct_payable), ttm(revenue), 1e6)
    'adv_receipts__lvl': ('lvl_point', 'adv_receipts'),  # asinh(point(adv_receipts))
    'adv_receipts__ta': ('ratio', 'point', 'adv_receipts', 'point', 'total_assets'),  # safe_div(point(adv_receipts), point(total_assets), 1e6)
    'bond_payable__lvl': ('lvl_point', 'bond_payable'),  # asinh(point(bond_payable))
    'bond_payable__ta': ('ratio', 'point', 'bond_payable', 'point', 'total_assets'),  # safe_div(point(bond_payable), point(total_assets), 1e6)
    'cip_total__lvl': ('lvl_point', 'cip_total'),  # asinh(point(cip_total))
    'cip_total__ta': ('ratio', 'point', 'cip_total', 'point', 'total_assets'),  # safe_div(point(cip_total), point(total_assets), 1e6)
    'contract_assets__lvl': ('lvl_point', 'contract_assets'),  # asinh(point(contract_assets))
    'contract_assets__ta': ('ratio', 'point', 'contract_assets', 'point', 'total_assets'),  # safe_div(point(contract_assets), point(total_assets), 1e6)
    'contract_assets__tr': ('ratio', 'point', 'contract_assets', 'ttm', 'revenue'),  # safe_div(point(contract_assets), ttm(revenue), 1e6)
    'contract_liab__lvl': ('lvl_point', 'contract_liab'),  # asinh(point(contract_liab))
    'contract_liab__ta': ('ratio', 'point', 'contract_liab', 'point', 'total_assets'),  # safe_div(point(contract_liab), point(total_assets), 1e6)
    'defer_inc_non_cur_liab__lvl': ('lvl_point', 'defer_inc_non_cur_liab'),  # asinh(point(defer_inc_non_cur_liab))
    'defer_inc_non_cur_liab__ta': ('ratio', 'point', 'defer_inc_non_cur_liab', 'point', 'total_assets'),  # safe_div(point(defer_inc_non_cur_liab), point(total_assets), 1e6)
    'defer_tax_assets__lvl': ('lvl_point', 'defer_tax_assets'),  # asinh(point(defer_tax_assets))
    'defer_tax_assets__ta': ('ratio', 'point', 'defer_tax_assets', 'point', 'total_assets'),  # safe_div(point(defer_tax_assets), point(total_assets), 1e6)
    'defer_tax_liab__lvl': ('lvl_point', 'defer_tax_liab'),  # asinh(point(defer_tax_liab))
    'defer_tax_liab__ta': ('ratio', 'point', 'defer_tax_liab', 'point', 'total_assets'),  # safe_div(point(defer_tax_liab), point(total_assets), 1e6)
    'fix_assets__lvl': ('lvl_point', 'fix_assets'),  # asinh(point(fix_assets))
    'fix_assets__ta': ('ratio', 'point', 'fix_assets', 'point', 'total_assets'),  # safe_div(point(fix_assets), point(total_assets), 1e6)
    'goodwill__lvl': ('lvl_point', 'goodwill'),  # asinh(point(goodwill))
    'goodwill__ta': ('ratio', 'point', 'goodwill', 'point', 'total_assets'),  # safe_div(point(goodwill), point(total_assets), 1e6)
    'goodwill__tr': ('ratio', 'point', 'goodwill', 'ttm', 'revenue'),  # safe_div(point(goodwill), ttm(revenue), 1e6)
    'intan_assets__lvl': ('lvl_point', 'intan_assets'),  # asinh(point(intan_assets))
    'inventories__lvl': ('lvl_point', 'inventories'),  # asinh(point(inventories))
    'inventories__ta': ('ratio', 'point', 'inventories', 'point', 'total_assets'),  # safe_div(point(inventories), point(total_assets), 1e6)
    'inventories__tr': ('ratio', 'point', 'inventories', 'ttm', 'revenue'),  # safe_div(point(inventories), ttm(revenue), 1e6)
    'lt_borr__lvl': ('lvl_point', 'lt_borr'),  # asinh(point(lt_borr))
    'lt_borr__ta': ('ratio', 'point', 'lt_borr', 'point', 'total_assets'),  # safe_div(point(lt_borr), point(total_assets), 1e6)
    'minority_int__lvl': ('lvl_point', 'minority_int'),  # asinh(point(minority_int))
    'minority_int__ta': ('ratio', 'point', 'minority_int', 'point', 'total_assets'),  # safe_div(point(minority_int), point(total_assets), 1e6)
    'money_cap__lvl': ('lvl_point', 'money_cap'),  # asinh(point(money_cap))
    'money_cap__ta': ('ratio', 'point', 'money_cap', 'point', 'total_assets'),  # safe_div(point(money_cap), point(total_assets), 1e6)
    'money_cap__tr': ('ratio', 'point', 'money_cap', 'ttm', 'revenue'),  # safe_div(point(money_cap), ttm(revenue), 1e6)
    'non_cur_liab_due_1y__lvl': ('lvl_point', 'non_cur_liab_due_1y'),  # asinh(point(non_cur_liab_due_1y))
    'non_cur_liab_due_1y__ta': ('ratio', 'point', 'non_cur_liab_due_1y', 'point', 'total_assets'),  # safe_div(point(non_cur_liab_due_1y), point(total_assets), 1e6)
    'notes_payable__lvl': ('lvl_point', 'notes_payable'),  # asinh(point(notes_payable))
    'notes_payable__ta': ('ratio', 'point', 'notes_payable', 'point', 'total_assets'),  # safe_div(point(notes_payable), point(total_assets), 1e6)
    'notes_payable__tr': ('ratio', 'point', 'notes_payable', 'ttm', 'revenue'),  # safe_div(point(notes_payable), ttm(revenue), 1e6)
    'notes_receiv__lvl': ('lvl_point', 'notes_receiv'),  # asinh(point(notes_receiv))
    'notes_receiv__ta': ('ratio', 'point', 'notes_receiv', 'point', 'total_assets'),  # safe_div(point(notes_receiv), point(total_assets), 1e6)
    'notes_receiv__tr': ('ratio', 'point', 'notes_receiv', 'ttm', 'revenue'),  # safe_div(point(notes_receiv), ttm(revenue), 1e6)
    'oth_receiv__lvl': ('lvl_point', 'oth_receiv'),  # asinh(point(oth_receiv))
    'prepayment__lvl': ('lvl_point', 'prepayment'),  # asinh(point(prepayment))
    'prepayment__ta': ('ratio', 'point', 'prepayment', 'point', 'total_assets'),  # safe_div(point(prepayment), point(total_assets), 1e6)
    'r_and_d__lvl': ('lvl_point', 'r_and_d'),  # asinh(point(r_and_d))
    'r_and_d__ta': ('ratio', 'point', 'r_and_d', 'point', 'total_assets'),  # safe_div(point(r_and_d), point(total_assets), 1e6)
    'st_borr__lvl': ('lvl_point', 'st_borr'),  # asinh(point(st_borr))
    'st_borr__ta': ('ratio', 'point', 'st_borr', 'point', 'total_assets'),  # safe_div(point(st_borr), point(total_assets), 1e6)
    'total_assets__lvl': ('lvl_point', 'total_assets'),  # asinh(point(total_assets))
    'total_cur_assets__lvl': ('lvl_point', 'total_cur_assets'),  # asinh(point(total_cur_assets))
    'total_cur_assets__ta': ('ratio', 'point', 'total_cur_assets', 'point', 'total_assets'),  # safe_div(point(total_cur_assets), point(total_assets), 1e6)
    'total_cur_liab__lvl': ('lvl_point', 'total_cur_liab'),  # asinh(point(total_cur_liab))
    'total_cur_liab__ta': ('ratio', 'point', 'total_cur_liab', 'point', 'total_assets'),  # safe_div(point(total_cur_liab), point(total_assets), 1e6)
    'total_hldr_eqy_exc_min_int__lvl': ('lvl_point', 'total_hldr_eqy_exc_min_int'),  # asinh(point(total_hldr_eqy_exc_min_int))
    'total_hldr_eqy_exc_min_int__ta': ('ratio', 'point', 'total_hldr_eqy_exc_min_int', 'point', 'total_assets'),  # safe_div(point(total_hldr_eqy_exc_min_int), point(total_assets), 1e6)
    'total_liab__lvl': ('lvl_point', 'total_liab'),  # asinh(point(total_liab))
    'total_liab__ta': ('ratio', 'point', 'total_liab', 'point', 'total_assets'),  # safe_div(point(total_liab), point(total_assets), 1e6)
    'total_nca__lvl': ('lvl_point', 'total_nca'),  # asinh(point(total_nca))
    'total_nca__ta': ('ratio', 'point', 'total_nca', 'point', 'total_assets'),  # safe_div(point(total_nca), point(total_assets), 1e6)
    'total_ncl__lvl': ('lvl_point', 'total_ncl'),  # asinh(point(total_ncl))
    'total_ncl__ta': ('ratio', 'point', 'total_ncl', 'point', 'total_assets'),  # safe_div(point(total_ncl), point(total_assets), 1e6)
    'total_share__lvl': ('lvl_point', 'total_share'),  # asinh(point(total_share))
    'trad_asset__lvl': ('lvl_point', 'trad_asset'),  # asinh(point(trad_asset))
    'trad_asset__ta': ('ratio', 'point', 'trad_asset', 'point', 'total_assets'),  # safe_div(point(trad_asset), point(total_assets), 1e6)
    'treasury_share__lvl': ('lvl_point', 'treasury_share'),  # asinh(point(treasury_share))
    'treasury_share__ta': ('ratio', 'point', 'treasury_share', 'point', 'total_assets'),  # safe_div(point(treasury_share), point(total_assets), 1e6)
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
