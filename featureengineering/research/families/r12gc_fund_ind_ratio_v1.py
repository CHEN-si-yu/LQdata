"""R12 绿卡批 A · 供应商财务指标表时点比率两件套（family `r12gc_fund_ind_ratio_v1`）。

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-06 绿卡通道扩容裁定：三项基础门槛）
════════════════════════════════════════════════════════════════════════
  cash_ratio   现金比率 = (货币资金 + 交易性金融资产) / 流动负债合计
  quick_ratio  速动比率 = (流动资产 − 存货) / 流动负债合计

两个对象是 `stock_financial_indicator` 的**时点比率**白名单字段（`fea/deriv.py`
IND_SAFE），`ctx.ind()` 明文允许直取 —— 时点比率不随累计口径变化、无跨季锯齿。
供应商原文注明两者「直接使用预计算字段」（fin_*_y 系）。比率**水平**，不是
delta_cash_ratio / delta_quick_ratio（比率的同比差，属比率再运算，已按 R10 批量裁 OUT）。

★★ 本批对象是**绿卡候选（只主张三项基础门槛；G4–G7 未主张）**。
   `higher_is_better=true` 只表示 declared=raw（不做任何反号），不代表任何预测方向。

════════════════════════════════════════════════════════════════════════
二、口径
════════════════════════════════════════════════════════════════════════
· 量纲 = 供应商**原始百分数**，不做 /100 重标（与在产 `ind_int_to_talcap` 同一处理；
  三项基础门槛只关心截面秩）。
· 语义 = 当日已知的**最新已披露报告期**（ctx.ind 按 ann_date 锚定）的供应商值，
  报告期之间保持常数。供应商标注 _y（年报口径）与「最新报告期」的差异列入
  README 边界案例 9；若主裁判要求严格年报口径需改走年频渠道。
· 报告期缺失 ⇒ NaN（不跨报告期前向填充、不补零）；池外股票 ⇒ NaN。

════════════════════════════════════════════════════════════════════════
三、PIT / 无泄露
════════════════════════════════════════════════════════════════════════
`ctx.ind(field, lag=0)` 走版本表（ann_date 锚定，T 日只用已披露最新版本）。
模块内仅一次直取调用，**没有** shift / 前向填充 / 全样本统计量。
本模块不依赖 `r12gc_common`（直取字段，无需网格）。
"""
from __future__ import annotations

import numpy as np

FROZEN = {
    "cash_ratio": "r12gc_cash_ratio_v1",
    "quick_ratio": "r12gc_quick_ratio_v1",
}


def build(parameters):
    variant = parameters.get("variant") if isinstance(parameters, dict) else None
    if variant not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))

    def compute(ctx):
        return np.asarray(ctx.ind(variant), dtype=np.float64)

    return compute
