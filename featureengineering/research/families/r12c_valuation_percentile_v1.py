"""R12 批 C · C5 估值时序分位（family `r12c_valuation_percentile_v1`；**alpha_v2（B）通道**）。

════════════════════════════════════════════════════════════════════════
一、机制与方向预注册（H2）
════════════════════════════════════════════════════════════════════════
估值的绝对水平受行业与规模结构污染；个股**自身历史分位**控制了个股估值中枢差异，
估值在自身历史低位时的均值回复力度更强（时序价值效应；idea_bank r12u2_029）。

    EP(t) = n_income_attr_p_TTM / (未复权 close × 当期已披露 total_share)     （与在产 `ep_ttm` 同口径）
    BP(t) = total_hldr_eqy_exc_min_int / 同上市值                            （与在产 `bp` 同口径）
    factor(t) = EP(t)（或 BP(t)）在过去 **750 个交易日**窗口内的时间序列分位（∈[0,1]）

分位定义 = `ctx.roll_rank`（WorldQuant Ts_Rank 口径）：窗口内**严格小于**当前值的样本
占比 `less / (cnt − 1)`（`cnt` = 窗内有效值个数，≥2）。**高 = 当前估值相对自身历史便宜**。

**方向先验（事前冻结，不得反号）**：时序分位高（自身历史便宜）→ 未来 h=1/h=5 收益正向
（`higher_is_better=True`，declared = raw）。
**证伪点**：h=1 或 h=5 的 declared RankIC HAC95 下界 ≤ 0、或正向年份 < 6/9（政策 v2 G4）
⇒ 先验被证伪，按 H2 不反号、直接否决。

两个预登记变体：EP-750 与 BP-750（估值度量的期限结构对照；idea 原文另提“500 日”期限，
本族预算 2 已由 EP/BP 用尽，**不登记** 500 日窗，列入 README 不确定项）。

差异论证（H3/H4）·重点（供应商 `pe_ttm_percentile` 的处置）：
· 在库 229 项中与“历史锚定”最近的是 `price_to_52w_high` / `price_distance_from_52w_low` /
  `r10gc_high_low_*`（**价格**相对自身 1 年高低点），与本对象（**估值比率**相对自身
  3 年滚动分位）不同构：被分位的对象、窗口（252 vs 750）与语义（价格位置 vs 估值位置）均不同。
· 供应商 `stock_finance` **存在** `pe_ttm_percentile` 字段（[0,1]，2026 年池内无缺失）。本模块
  **不读**该字段：其窗口口径（上市以来？固定窗？）与可用时点不可审计，无法证明无前视
  （H5）。本对象的分子/分母均自算、分位窗为**因果滚动 750 日**，PIT 可复现。
  ⇒ G5 差异测量时**必须**报告与供应商 `pe_ttm_percentile` 的相关性（若 eval 侧可读该字段）；
  若 |ρ| 过高（软带 >0.85），以「口径可审计 + PIT 可证」为由论证，或按 G5 处理，不得静默。
· 与在产 `ep_ttm` / `bp` 的相关性必须报告（分位化后应显著降低；这是本构造的核心增量主张）。

════════════════════════════════════════════════════════════════════════
二、PIT 与口径
════════════════════════════════════════════════════════════════════════
· 分子走 `ctx.ttm` / `ctx.point`（ann_date 锚定的版本表，T 日只看到 ≤T 的已披露版本）；
  市值 = 未复权 `close`（状态量，停牌前向填充）× `total_share`（状态量）—— 与在产
  `log_mv`/`ep_ttm`/`bp` 同一自算市值约定（不用供应商 `total_mv`，日频快照不可审计）。
· 分位窗是**交易日窗口**（750 行），`ctx.roll_rank` 只用窗内历史值，天然因果。
· 缺失：窗内有效值 < MIN_COUNT(600) ⇒ NaN（80% 下限；上市不足 ~600 交易日的股票
  天然为 NaN，覆盖率如实上报）；`roll_rank` 内部要求 cnt ≥ 2；池外/未上市/ST ⇒ NaN。
· `stock_daily` / `stock_finance` / 三张财报表 `delay_days: 0`，不需要 `lag_grid`；
  模块内无 shift / 前向填充 / 全样本统计量（分位窗滚动、逐日重算）。
· warmup_days=1370（750 交易日 ≈ 1350 日历天 + 余量），保证 2018-01-02 起即有多数
  股票具备完整分位历史。
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("research.families.r12c_valuation_percentile_v1")

NP = "n_income_attr_p"                      # 归母净利润（累计制 → TTM）
EQ = "total_hldr_eqy_exc_min_int"           # 归母股东权益（时点；POSITIVE_ONLY）
MIN_ABS_DEN = 1e6                           # 分母地板：元
MIN_COUNT = 600                             # 750 窗内有效值下限（80%）
# 事前冻结：(kind, window) -> 候选名；只登记 EP/BP × 750 日两变体
FROZEN = {
    ("ep", 750): "r12c_ep_pct750_v1",
    ("bp", 750): "r12c_bp_pct750_v1",
}


def _frozen_key(parameters):
    if not isinstance(parameters, dict):
        return None
    return (parameters.get("kind"), parameters.get("window"))


def _mktcap(ctx):
    """自算市值（元）= 未复权 close（状态量）× total_share（状态量）；与在产估值族同口径。"""
    close = np.asarray(ctx.px("close"), dtype=np.float64)
    share = np.asarray(ctx.px("total_share"), dtype=np.float64)
    return close * share


def build(parameters):
    """只允许预登记的两个变体；不搜索窗口、不搜索方向。"""
    key = _frozen_key(parameters)
    if key not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))
    kind, window = key

    def compute(ctx):
        if kind == "ep":
            ratio = ctx.safe_div(ctx.ttm(NP), _mktcap(ctx), min_abs_den=MIN_ABS_DEN)
        else:
            ratio = ctx.safe_div(ctx.point(EQ), _mktcap(ctx), min_abs_den=MIN_ABS_DEN)
        pct = np.asarray(ctx.roll_rank(ratio, window, MIN_COUNT), dtype=np.float64)
        grid = np.where(np.asarray(ctx.universe, dtype=bool), pct, np.nan)
        log.info("r12c_vp kind=%s window=%d panel_finite=%d", kind, window,
                 int(np.isfinite(grid).sum()))
        return np.ascontiguousarray(grid, dtype=np.float64)

    return compute
