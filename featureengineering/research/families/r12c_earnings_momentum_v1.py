"""R12 批 C · C3 盈利改善速度复合 ΔROE+ΔROA+Δ毛利率（family `r12c_earnings_momentum_v1`；**alpha_v2（B）通道**）。

════════════════════════════════════════════════════════════════════════
一、机制与方向预注册（H2）
════════════════════════════════════════════════════════════════════════
市场对盈利**水平**定价较充分、对**改善方向**定价不足（增量信息在变化量里，idea_bank
r12u2_015；原文提示高极值组合胜率高）。三个盈利比率各自取**同比**与**环比**变化：

    ROE(TTM) = n_income_attr_p_TTM / total_hldr_eqy_exc_min_int        （归母净利/归母权益）
    ROA(TTM) = n_income_attr_p_TTM / total_assets                      （归母净利/总资产）
    GPM(TTM) = (revenue_TTM − oper_cost_TTM) / revenue_TTM             （毛利率）

    Δ 同比 leg = X(t) − X(t−4 个报告期)      Δ 环比 leg = X(t) − X(t−1 个报告期)
    6 条 leg 各自做**当日截面** z-score（`ctx.cs_zscore`，只用当日有效值；无全样本统计），
    factor = 6 条 z 的**等权均值**（任一 leg 缺失 ⇒ 当日该股 NaN，不做部分腿回退）
    变体 `ems_yoy` = 只用 3 条同比 leg 的等权均值（同比 vs 同比+环比的期限结构对照）。

**方向先验（事前冻结，不得反号）**：改善速度高 → 未来 h=1/h=5 收益正向
（`higher_is_better=True`，declared = raw）。
**证伪点**：h=1 或 h=5 的 declared RankIC HAC95 下界 ≤ 0、或正向年份 < 6/9（政策 v2 G4）
⇒ 先验被证伪，按 H2 不反号、直接否决。

**实现口径与 idea 原文的偏离（如实记录，README 同步）**：idea r12u2_015 写的是
“Value_t / Value_{t−4Q} − 1”（对**比率**再取变化率）；`plan.md` C3 写的是
“ΔROE+ΔROA+Δ毛利率 … 同比+环比”。本模块按 `plan.md` 的 Δ 记号实现为**水平差**
（百分点），不用比率变化率：ROE/ROA 可跨零，比率变化率在近零分母上会爆炸成 ±10^3
量级的假极值（在产 `factor_growth` 族只对**单序列原始科目**用 `分母取 |prev| + 1e-6 地板`
的变化率；对已聚合比率再取变化率正是 R10 §1.6 批量裁 OUT 的形式）。若后续要试
原文的比率变化率口径，须另立 family、另占预算。

差异论证（H3/H4）：在库 229 项只有盈利**水平**（`roe_ttm`/`roa_ttm`/`gpm_ttm`）
与单指标同比（`yoy_revenue`/`qf_core_roe_yoy_change` 等，均为**供应商单季字段**口径）；
已删 374 项中变化量族仅 `np_to_opex_yoy`/`share_issuance_yoy`（费用/股本变化），
与本对象（三比率改善速度合成）不同构。须测量与 `roe_ttm`、`roa_ttm`、`gpm_ttm`、
`qf_*_yoy_change` 的 pooled |ρ| 并报告（G5；>0.85 需论证）。

════════════════════════════════════════════════════════════════════════
二、PIT（H5，ann_date 锚定，无跨报告前向填充）
════════════════════════════════════════════════════════════════════════
`ctx.ttm(field)` / `ctx.lag_ttm(field, k)` / `ctx.point(field, lag)` 走 `fea/deriv.py`
版本表：T 日只看到 ann_date/f_ann_date ≤ T 的最新已披露版本（追溯修订不改写历史生效日）；
lag 的单位是**报告期**（lag=4 = 四个报告期前），不是交易日位移。分母的时点量用
`ctx.point(..., lag=k)` 与分子的 `lag_ttm(field, k)` **同期对齐**（分子分母同报告期）。
模块内无 shift / 前向填充 / 全样本统计量；季度值经版本表天然日频 as-of。

════════════════════════════════════════════════════════════════════════
三、口径与缺失
════════════════════════════════════════════════════════════════════════
· 分母保护：`ctx.safe_div(..., min_abs_den=1e6)`（元）；`total_assets` /
  `total_hldr_eqy_exc_min_int` / `revenue` / `oper_cost` 在 `fea/deriv.py` 的
  POSITIVE_ONLY 中（≤0 记缺失），因此**银行/保险**（`oper_cost = 精确 0.0`）的
  GPM 恒为 NaN ⇒ 该批股票三比率腿缺一 ⇒ **整股当日 NaN**（覆盖率损失如实上报；
  这是把「金融股没有毛利率」与「毛利率改善」机制隔离的保守处置）。
· 任一 leg 非有限 ⇒ NaN（不补值）；池外/未上市/ST ⇒ NaN。
· Δ 可能出现大值（近零净利的公司），**不在模块内截断**，交由引擎输出端 1%/99%
  缩尾（与在产 `yoy_revenue` 处理一致）。
· `lag_ttm` 依赖更早的报告期 ⇒ warmup_days=700（与 gcA 财务族同口径）。
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("research.families.r12c_earnings_momentum_v1")

NP = "n_income_attr_p"                      # 归母净利润（累计制 → TTM）
EQ = "total_hldr_eqy_exc_min_int"           # 归母股东权益（时点）
TA = "total_assets"                         # 总资产（时点）
REV = "revenue"                             # 营业收入（累计制 → TTM）
OC = "oper_cost"                            # 营业成本（累计制 → TTM）
MIN_ABS_DEN = 1e6                           # 分母地板：元
# 事前冻结：变体名 -> 是否加入环比腿（期限结构式，不扫网格）
FROZEN = {
    "ems_yoy_qoq": True,     # 3 比率 ×（同比 + 环比）= 6 条 z 腿
    "ems_yoy": False,        # 3 比率 × 同比 = 3 条 z 腿
}


def _frozen_key(parameters):
    if not isinstance(parameters, dict):
        return None
    return parameters.get("variant")


def _ratio(ctx, num_field, den_mode, lag):
    """X(t − lag 个报告期)：分子 lag_ttm，分母同期 point（点量取 lag 期前的已披露值）。"""
    num = ctx.lag_ttm(num_field, lag) if lag else ctx.ttm(num_field)
    den = ctx.point(den_mode, lag) if lag else ctx.point(den_mode)
    return ctx.safe_div(num, den, min_abs_den=MIN_ABS_DEN)


def _gpm(ctx, lag):
    rev = ctx.lag_ttm(REV, lag) if lag else ctx.ttm(REV)
    oc = ctx.lag_ttm(OC, lag) if lag else ctx.ttm(OC)
    return ctx.safe_div(rev - oc, rev, min_abs_den=MIN_ABS_DEN)


def build(parameters):
    """只允许预登记的两个变体；不搜索窗口、不搜索方向。"""
    variant = _frozen_key(parameters)
    if variant not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))
    with_qoq = FROZEN[variant]

    def compute(ctx):
        univ = np.asarray(ctx.universe, dtype=bool)
        legs = []
        for lag in ([4, 1] if with_qoq else [4]):
            roe0 = np.asarray(_ratio(ctx, NP, EQ, 0), dtype=np.float64)
            roa0 = np.asarray(_ratio(ctx, NP, TA, 0), dtype=np.float64)
            gpm0 = np.asarray(_gpm(ctx, 0), dtype=np.float64)
            roe_k = np.asarray(_ratio(ctx, NP, EQ, lag), dtype=np.float64)
            roa_k = np.asarray(_ratio(ctx, NP, TA, lag), dtype=np.float64)
            gpm_k = np.asarray(_gpm(ctx, lag), dtype=np.float64)
            for cur, prev in ((roe0, roe_k), (roa0, roa_k), (gpm0, gpm_k)):
                leg = cur - prev
                legs.append(np.asarray(ctx.cs_zscore(leg, mask=univ), dtype=np.float64))
        stacked = np.stack(legs)                      # (L, T, C)
        ok = np.isfinite(stacked).all(axis=0)         # 全腿有效才出值（不做部分腿回退）
        value = np.where(ok, stacked.sum(axis=0) / len(legs), np.nan)
        grid = np.where(univ, value, np.nan)
        log.info("r12c_%s panel_finite=%d", variant, int(np.isfinite(grid).sum()))
        return np.ascontiguousarray(grid, dtype=np.float64)

    return compute
