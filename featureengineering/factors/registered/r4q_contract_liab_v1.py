"""U-V2 · r4q_contract_liab_v1 —— 合同负债前瞻 / 订单先行指标（Round 4 批 3）。

五个预登记信号（3 机制 + 2 变体，方向全部 true，任务书 B3 §V2 事前固定）：

    contract_liab_growth_yoy   true  合同负债同比增速 = CL(p)/CL(p−4季) − 1（订单增长）
    contract_liab_to_revenue   true  合同负债 / 营收TTM（订单覆盖率）
    contract_liab_accel        true  增速的二阶：g(p) − g(p−1)（增速环比加速）
    adv_recv_cl_growth_yoy     true  变体：预收款项+合同负债 合计口径的同比增速
    adv_recv_cl_to_revenue     true  变体：合计口径 / 营收TTM

构念差异（防重蹈，B3 §V2 要求写明）：
    已删因子 `bs_net_contract_to_assets` 是**合同资产净额/总资产**——合同【资产】
    是「已履约、待收款」的应收性资产（收款权），高=回款风险；本单元是合同
    【负债】与预收款——「已收款、待履约」的订单储备（履约义务），两类的会计
    方向、经济含义与预期收益方向均不同（前者风险侧 false / 后者订单前瞻 true）。

字段语义实测（delivery/r4q_contract_liab_v1/report.md 数据事实节，2026-10-04）：
    · `contract_liab` = 合同负债（负债侧）：2018–2019 仅早采用者 >0（106/126 家，
      万科 2018 年报即 5047 亿）；2020 新收入准则全面启用后 2037+/2115 家 >0；
      2015–2017 全池为显式 0.0（科目未启用）。0 是供应商的「无此科目」表达，
      不是「订单覆盖为零」，故水平类因子要求 CL>0（严格正），0 → NaN。
    · `prepayment` = **预付款项（资产侧，预付款）**——实测排除：茅台 2015–2025
      prepayment 仅 15→0.1 亿且不随 2020 准则切换跳变，而同日历 adv_receipts
      137.4 亿 → 0、contract_liab 0 → 133.2 亿（典型重分类签名）。故**合计口径
      采用 `adv_receipts`（预收款项，负债侧）+ `contract_liab`，不含 prepayment**。
    · `adv_receipts` 2020 起大幅归零（重分类进合同负债，>0 家数 1969→772），
      合计口径在 2019→2020 切换处连续（comb20/comb19 中位数 1.13）。

PIT/可得性：全部经财务层 `ctx.point`（时点科目，pit=max(ann_date, f_ann_date,
end_date)，保留修订版本、取公告时点已知的最新报告期）与 `ctx.ttm`；同比 =
同一公告时点上的 p 与 p−4 期（报告期口径，**不是**日频减 252 天）。无未来信息。

已知数据危害（报告披露，不做事后修剪）：2023 年 IFRS17 保险合同负债并入——
601318/601601/601336 合同负债 0 → 4.16/1.87/1.15 万亿（YoY 分母 0 → NaN，次年
增速与水平项将被这一重分类放大）。全池保留（含金融股），敏感性在报告中给出。
"""
from __future__ import annotations

import numpy as np

SIGNALS = (
    "contract_liab_growth_yoy",
    "contract_liab_to_revenue",
    "contract_liab_accel",
    "adv_recv_cl_growth_yoy",
    "adv_recv_cl_to_revenue",
)
COMBINED = ("adv_recv_cl_growth_yoy", "adv_recv_cl_to_revenue")

POINT = "contract_liab"        # 合同负债（负债侧）
ADV = "adv_receipts"           # 预收款项（负债侧；旧准则科目）
MIN_ABS_DEN = 1e6              # 分母地板：金额科目 > 100 万元才参与比率
                               # （与 r4c_accruals_v1 同一约定）


def _yoy_growth(cur: np.ndarray, base: np.ndarray) -> np.ndarray:
    """(cur − base) / base；仅当 base > MIN_ABS_DEN（严格正）时有定义。

    base ≤ 0 的情况（科目未启用期的 0、实测仅 3 行的微小负值）一律 NaN——
    增速的分母必须是真实存在的订单存量；cur=0 而 base>0 是真实的 −100%（保留）。
    """
    cur = np.asarray(cur, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    ok = np.isfinite(cur) & np.isfinite(base) & (base > MIN_ABS_DEN)
    out = np.full(cur.shape, np.nan, dtype=np.float64)
    np.divide(cur - base, base, out=out, where=ok)
    return out


def _coverage(level: np.ndarray, revenue: np.ndarray) -> np.ndarray:
    """合同负债（或合计）/ 营收TTM；仅当 level > 0 且 revenue > MIN_ABS_DEN。

    level > 0（严格）把「科目未启用/无此业务」的 0 与真实覆盖率区分开
    （B1 §2.1：缺失与零必须区分，只有明确用零表达「无」才可零填充；
    此处 0 实证为「科目未启用」，非「订单为零」）。
    """
    level = np.asarray(level, dtype=np.float64)
    revenue = np.asarray(revenue, dtype=np.float64)
    ok = np.isfinite(level) & (level > 0) & np.isfinite(revenue) & (revenue > MIN_ABS_DEN)
    out = np.full(level.shape, np.nan, dtype=np.float64)
    np.divide(level, revenue, out=out, where=ok)
    return out


def build(parameters):
    """五个固定机制之一；不允许搜窗口、改方向或增删机制。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("r4q_contract_liab_v1 permits only the five preregistered signal identifiers")
    signal = parameters["signal"]
    combined = signal in COMBINED

    def _level(ctx, lag=0):
        """合同负债（或合计口径）在滞后 lag 个报告期上的时点值网格，(T,C) float64。

        `ctx.point` 返回 float32（Panel.asof 默认）；财务金额 ~1e11 量级下
        float32 只有 7 位有效数字，故此处一律先转 float64 再参与差/商。
        """
        cl = np.asarray(ctx.point(POINT, lag=lag), dtype=np.float64)
        if combined:
            cl = cl + np.asarray(ctx.point(ADV, lag=lag), dtype=np.float64)
        return cl

    def compute(ctx):
        if signal in ("contract_liab_growth_yoy", "adv_recv_cl_growth_yoy"):
            value = _yoy_growth(_level(ctx, 0), _level(ctx, 4))
        elif signal in ("contract_liab_to_revenue", "adv_recv_cl_to_revenue"):
            value = _coverage(_level(ctx, 0), np.asarray(ctx.ttm("revenue"), dtype=np.float64))
        else:  # contract_liab_accel：g(p) − g(p−1)，四个时点全部取同一公告时点
            g_now = _yoy_growth(_level(ctx, 0), _level(ctx, 4))
            g_prev = _yoy_growth(_level(ctx, 1), _level(ctx, 5))
            value = g_now - g_prev
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
