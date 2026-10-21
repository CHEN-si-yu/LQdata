"""R10 绿卡基础因子批 · 单科目同比（Tushare Growth/Quality 的 `*_yoy` 干净子集）。

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-05 裁定：三项基础门槛 + 市场绿卡「透明观测描述量」规则）
════════════════════════════════════════════════════════════════════════
本模块 5 个对象的操作数都是**单条原始会计科目序列**，变换都是**同比比率**
（`Value_t / Value_{t−4Q} − 1`）—— 规则明文 IN 的「a difference or year-on-year ratio
of a single fundamental series / 对单序列原始字段的标准变换」：

  yoy_net_profit     stock_income.n_income_attr_p      TTM   归母净利润
  yoy_ocf            stock_cashflow.n_cashflow_act     TTM   经营活动现金流净额
  yoy_total_asset    stock_balancesheet.total_assets   时点  总资产
  yoy_net_asset      stock_balancesheet.total_hldr_eqy_exc_min_int 时点 归母股东权益
  income_tax_yoy     stock_income.income_tax           TTM   所得税费用

★ 明确**不在**本模块（Lead 裁定边界，本执行者按「边界案例从严」同样判 OUT，已单独上报）：
  · `delta_*`（delta_roe/roa/gpm/npm/opm/de/quick_ratio/current_ratio/asset_turnover/
    inventory_turnover/cash_ratio）—— 被差分的对象本身是**两个聚合量之比**，
    取变化率即规则排除的「re-operations on already-aggregated quantities（变化率 of ratio）」；
  · `np_to_*_yoy`、`expenses_to_equity_yoy`、`yoy_roa`、`yoy_roe` —— 同比的操作数是比率，同上；
  · `sa`/`pa`/`eaa`/`eap` —— 变化的再变化（加速度），对已聚合量的再运算；
  · `gross_margin_qoq`、`np_ttm_qoq`、`gpm_qoq`、`npm_q_qoq`、`npm_ttm_qoq` —— 比率/聚合量的环比。
  以上 28 个已在给 Lead 的逐名表中列为 IN-B（待裁决），**本批不登记、不计算**。

★★ 本批对象是**绿卡候选（只主张三项基础门槛；G4–G7 未主张）**。
   `higher_is_better=true` 只表示 declared=raw（不做任何反号），**不代表任何预测方向**，
   尤其**不主张**「利润增长 → 未来收益高」。

════════════════════════════════════════════════════════════════════════
二、PIT 合同（ann_date 锚定，无跨报告前向填充）
════════════════════════════════════════════════════════════════════════
`ctx.ttm(field)` / `ctx.point(field, lag)` 走 `fea/deriv.py::Derivative` 的**版本表**：
每条记录按 `ann_date`/`f_ann_date` 排序，T 日只看到 **ann_date ≤ T** 的最新已披露版本；
追溯修订不会改写历史生效日。lag 的单位是**报告期**（`lag=4` = 四个季度前），
不是交易日位移 —— 与在产 `yoy_revenue`（`ctx.ttm` + `ctx.lag_ttm(REV, 4)`）和
`asset_growth_qoq`（`ctx.point(TA)` + `ctx.point(TA, lag=1)`）完全同一套机制。
本模块内**没有** `shift` / `ffill` / 全样本统计量。
"""

from __future__ import annotations

import numpy as np

# 事前冻结：(变体名) -> (数据集, 财务字段, 口径)
FROZEN = {
    "yoy_net_profit": ("stock_income", "n_income_attr_p", "ttm"),
    "yoy_ocf": ("stock_cashflow", "n_cashflow_act", "ttm"),
    "yoy_total_asset": ("stock_balancesheet", "total_assets", "point"),
    "yoy_net_asset": ("stock_balancesheet", "total_hldr_eqy_exc_min_int", "point"),
    "income_tax_yoy": ("stock_income", "income_tax", "ttm"),
}


def _frozen_key(parameters):
    if not isinstance(parameters, dict):
        return None
    return parameters.get("variant")


def build(parameters):
    variant = _frozen_key(parameters)
    if variant not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))
    dataset, field, mode = FROZEN[variant]

    def compute(ctx):
        if mode == "ttm":
            cur = np.asarray(ctx.ttm(field), dtype=np.float64)
            prev = np.asarray(ctx.lag_ttm(field, 4), dtype=np.float64)
        else:
            cur = np.asarray(ctx.point(field, 0), dtype=np.float64)
            prev = np.asarray(ctx.point(field, 4), dtype=np.float64)
        # 与在产 yoy_revenue 同口径：分母取 |prev|（保留符号语义），地板 1e-6 挡 0/NaN。
        # 归母净利润可跨零 ⇒ 可能出现大绝对值；不在此处截断，交由引擎 1%/99% 缩尾
        # （与在产 yoy_revenue 的处理一致）。prev 缺失 ⇒ NaN，不补值、不前向填充。
        return ctx.safe_div(cur - prev, np.abs(prev), min_abs_den=1e-6)

    return compute
