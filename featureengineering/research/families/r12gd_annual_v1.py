"""R12 绿卡批 D · 年报口径白名单枚举

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-06 绿卡通道扩容裁定：三项基础门槛）
════════════════════════════════════════════════════════════════════════
本模块是 **批 D 财务白名单系统枚举**（worklist 381 行 → IN 15 行）的 r12gd_annual_v1 族实现。
每一个变体都是**白名单原始科目序列的标准变换**或**两原始量的比率**：
· `lvl_*`   —— 水平量，`arcsinh` 单调压缩（金额/比率水平；单调保序、无未来信息）；
· `ratio`   —— 分子 ÷ annual_value 网格（total_assets，无别名） 的比率，分母 `ctx.safe_div(..., 1e6)` 近零保护；
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
· `ctx.annual(dataset, field, 0)` = 该股**当时已公告的最新 12/31 年报**值
  （ann_date 锚定、保留修订、前缀最大报告期）；年报别名已声明于
  conf/field_expansion.json（annual__* 渠道）。比率型字段（ocf_to_* 等）直取不压缩。
· annual_ratio 的**分母** total_assets 无年度别名 ⇒ 走 `r12gc_common` 的
  年度网格 `annual_value(annual_only=True)`（与批 A 的 roa_y/roe_y 同源码）。

★ 哈希口径提醒：`research/catalog.py::implementation` 只对**本模块文件**取 sha256；
  修改本文件等价于定义变更，必须重新注册并留痕。
"""
from __future__ import annotations

import numpy as np

from research.families.r12gc_common import annual_value, table

MIN_ABS_DEN = 1e6

FROZEN = {
    # ---- r12gd_annual_v1：15 变体（逐条对应 candidate_*.yaml 的 variant）
    'bps__lvl': ('annual_lvl', 'stock_financial_indicator', 'bps'),  # asinh(ctx.annual('stock_financial_indicator','bps',0))
    'ebitda__lvl': ('annual_lvl', 'stock_income', 'ebitda'),  # asinh(ctx.annual('stock_income','ebitda',0))
    'ebitda__ta': ('annual_ratio', 'stock_income', 'ebitda'),  # safe_div(ctx.annual('stock_income','ebitda',0), AN(total_assets), 1e6)
    'fcfe__lvl': ('annual_lvl', 'stock_financial_indicator', 'fcfe'),  # asinh(ctx.annual('stock_financial_indicator','fcfe',0))
    'fcfe__ta': ('annual_ratio', 'stock_financial_indicator', 'fcfe'),  # safe_div(ctx.annual('stock_financial_indicator','fcfe',0), AN(total_assets), 1e6)
    'fcff__lvl': ('annual_lvl', 'stock_financial_indicator', 'fcff'),  # asinh(ctx.annual('stock_financial_indicator','fcff',0))
    'fcff__ta': ('annual_ratio', 'stock_financial_indicator', 'fcff'),  # safe_div(ctx.annual('stock_financial_indicator','fcff',0), AN(total_assets), 1e6)
    'invest_capital__lvl': ('annual_lvl', 'stock_financial_indicator', 'invest_capital'),  # asinh(ctx.annual('stock_financial_indicator','invest_capital',0))
    'invest_capital__ta': ('annual_ratio', 'stock_financial_indicator', 'invest_capital'),  # safe_div(ctx.annual('stock_financial_indicator','invest_capital',0), AN(total_assets), 1e6)
    'ocf_to_interestdebt__lvl': ('annual_raw', 'stock_financial_indicator', 'ocf_to_interestdebt'),  # ctx.annual('stock_financial_indicator','ocf_to_interestdebt',0)
    'ocf_to_opincome__lvl': ('annual_raw', 'stock_financial_indicator', 'ocf_to_opincome'),  # ctx.annual('stock_financial_indicator','ocf_to_opincome',0)
    'ocf_to_profit__lvl': ('annual_raw', 'stock_financial_indicator', 'ocf_to_profit'),  # ctx.annual('stock_financial_indicator','ocf_to_profit',0)
    'ocfps__lvl': ('annual_lvl', 'stock_financial_indicator', 'ocfps'),  # asinh(ctx.annual('stock_financial_indicator','ocfps',0))
    'q_ocf_to_or__lvl': ('annual_raw', 'stock_financial_indicator', 'q_ocf_to_or'),  # ctx.annual('stock_financial_indicator','q_ocf_to_or',0)
    'total_revenue__lvl': ('annual_lvl', 'stock_income', 'total_revenue'),  # asinh(ctx.annual('stock_income','total_revenue',0))
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
        if kind == "annual_lvl":
            return np.arcsinh(np.asarray(ctx.annual(spec[1], spec[2], 0), dtype=np.float64))
        if kind == "annual_raw":
            return np.asarray(ctx.annual(spec[1], spec[2], 0), dtype=np.float64)
        if kind == "annual_ratio":
            cache = {}
            den = annual_value(ctx, table(ctx, cache, "stock_balancesheet", "total_assets",
                                          annual_only=True))
            num = np.asarray(ctx.annual(spec[1], spec[2], 0), dtype=np.float64)
            return ctx.safe_div(num, np.asarray(den, dtype=np.float64), MIN_ABS_DEN)
        if kind == "ind":
            return np.asarray(ctx.ind(spec[1]), dtype=np.float64)
        raise AssertionError("unreachable spec kind: %r" % (spec,))

    return compute
