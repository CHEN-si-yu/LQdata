"""R12 绿卡批 A · 现金流市值比两件套（family `r12gc_cashflow_value_v1`）。

════════════════════════════════════════════════════════════════════════
一、资格（用户 2026-10-06 绿卡通道扩容裁定：三项基础门槛）
════════════════════════════════════════════════════════════════════════
  ncf_to_market  净现金流市值比  (经营 + 投资 + 筹资净现金流 TTM) / 总市值
  fcf_to_market  自由现金流市值比 (经营净现金流 TTM − 投资现金流出 TTM) / 总市值

两个对象 = 现金流量表原始净额科目（TTM）与自算市值的一次比率。三条净额科目均在
TTM_SOURCES 白名单，逐条对应供应商原文列出的分量。

★★ 本批对象是**绿卡候选（只主张三项基础门槛；G4–G7 未主张）**。
   `higher_is_better=true` 只表示 declared=raw（不做任何反号），不代表任何预测方向。

════════════════════════════════════════════════════════════════════════
二、口径（与同批 YAML 逐条对应）
════════════════════════════════════════════════════════════════════════
· ncf_to_market 按供应商原文逐项求和：OCF + ICF + FCF_fin（`n_cashflow_act` +
  `n_cashflow_inv_act` + `n_cash_flows_fnc_act`）。三项之和 ≡ 现金及现金等价物净
  增加额（n_incr_cash_cash_equ，白名单内）在汇率影响项为 0 时严格相等 —— 按原文
  逐项求和便于逐项复核。净额可为负（大额投资/偿债年份），不设地板。
· fcf_to_market 的 SICO（投资现金流出**小计**）**不在本库 127 字段内**，故以
  capex `c_pay_acq_const_fiolta` 代理 —— 即最常见定义「经营净流入 − 资本开支」，
  与供应商括注「经营性现金净流入 − 投资性现金流出」在资本开支主导时一致；
  对金融资产投资大的公司会**高估** FCF。★ 字段替换 vs 字段缺失属 IN-C 先例的
  边界，已列 README 边界案例 3 待主裁判裁决（若裁 OUT，本族剩 ncf_to_market）。
  替代方案 ttm(free_cashflow)（供应商自算企业自由现金流）因本地词典未载口径未采用。
· 市值 = `r12gc_common.mktcap`（未复权 close × total_share，与在产 cfp_ttm/sp_ttm
  同源）；分母 safe_div(min_abs_den=1e6)；分子为净额、可为负，不设地板。
· 与在产 cfp_ttm 的差异：cfp_ttm 只含经营净流量（本批含投资/筹资分量或减投资流出）。

════════════════════════════════════════════════════════════════════════
三、PIT / 无泄露
════════════════════════════════════════════════════════════════════════
TTM 走版本表（ann_date 锚定）；市值走价格层状态量；模块内没有 shift / 前向填充 /
全样本统计量。

★ 依赖 `research/families/r12gc_common.py`（mktcap）。
  catalog 只对**本文件**取 sha256，公共文件改动等价于定义变更 —— 须重新注册。
"""
from __future__ import annotations

import numpy as np

from research.families.r12gc_common import mktcap

FROZEN = {
    "ncf_to_market": "r12gc_ncf_to_market_v1",
    "fcf_to_market": "r12gc_fcf_to_market_v1",
}

MIN_ABS_DEN = 1e6


def build(parameters):
    variant = parameters.get("variant") if isinstance(parameters, dict) else None
    if variant not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))

    def compute(ctx):
        cap = mktcap(ctx, "total_share")
        if variant == "ncf_to_market":
            num = (np.asarray(ctx.ttm("n_cashflow_act"), dtype=np.float64)
                   + np.asarray(ctx.ttm("n_cashflow_inv_act"), dtype=np.float64)
                   + np.asarray(ctx.ttm("n_cash_flows_fnc_act"), dtype=np.float64))
        else:  # fcf_to_market
            num = (np.asarray(ctx.ttm("n_cashflow_act"), dtype=np.float64)
                   - np.asarray(ctx.ttm("c_pay_acq_const_fiolta"), dtype=np.float64))
        return ctx.safe_div(num, cap, MIN_ABS_DEN)

    return compute
