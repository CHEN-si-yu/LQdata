"""R12 绿卡批 D · 供应商财务指标表安全字段枚举

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-06 绿卡通道扩容裁定：三项基础门槛）
════════════════════════════════════════════════════════════════════════
本模块是 **批 D 财务白名单系统枚举**（worklist 381 行 → IN 10 行）的 r12gd_ind_level_v1 族实现。
每一个变体都是**白名单原始科目序列的标准变换**或**两原始量的比率**：
· `lvl_*`   —— 水平量，`arcsinh` 单调压缩（金额/比率水平；单调保序、无未来信息）；
· `ratio`   —— 分子 ÷ （无比率变体） 的比率，分母 `ctx.safe_div(..., 1e6)` 近零保护；
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
· `ctx.ind()` 只取 `fea/deriv.py::IND_SAFE` 白名单字段（时点比率、
  同期同比及明确白名单的单季度比率），框架内强制拦截累计 YTD 锯齿字段。
· 量纲 = 供应商原始百分数，不 /100 重标（三项门槛只看截面秩）。

★ 哈希口径提醒：`research/catalog.py::implementation` 只对**本模块文件**取 sha256；
  修改本文件等价于定义变更，必须重新注册并留痕。
"""
from __future__ import annotations

import numpy as np

MIN_ABS_DEN = 1e6

FROZEN = {
    # ---- r12gd_ind_level_v1：10 变体（逐条对应 candidate_*.yaml 的 variant）
    'current_ratio__lvl': ('ind', 'current_ratio'),  # ctx.ind('current_ratio')
    'ebitda_to_debt__lvl': ('ind', 'ebitda_to_debt'),  # ctx.ind('ebitda_to_debt')
    'ebt_yoy__lvl': ('ind', 'ebt_yoy'),  # ctx.ind('ebt_yoy')
    'op_yoy__lvl': ('ind', 'op_yoy'),  # ctx.ind('op_yoy')
    'q_dt_roe__lvl': ('ind', 'q_dt_roe'),  # ctx.ind('q_dt_roe')
    'q_npta__lvl': ('ind', 'q_npta'),  # ctx.ind('q_npta')
    'q_ocf_to_sales__lvl': ('ind', 'q_ocf_to_sales'),  # ctx.ind('q_ocf_to_sales')
    'q_roe__lvl': ('ind', 'q_roe'),  # ctx.ind('q_roe')
    'tbassets_to_totalassets__lvl': ('ind', 'tbassets_to_totalassets'),  # ctx.ind('tbassets_to_totalassets')
    'turn_days__lvl': ('ind', 'turn_days'),  # ctx.ind('turn_days')
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
