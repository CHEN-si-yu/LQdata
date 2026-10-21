"""R12 批 C · C4 残差动量（family `r12c_residual_momentum_v1`；**alpha_v2（B）通道**）。

════════════════════════════════════════════════════════════════════════
一、机制与方向预注册（H2）
════════════════════════════════════════════════════════════════════════
传统动量含系统性风险暴露；剔除市场 β 后的**标准化残差**仍具持续性（Blitz/Huij/Martens
残差动量范式；idea_bank r12u2_055）。原文排序用**标准化残差累计**而非 alpha 或原始残差
（残差波动差异大，标准化后可横截面比较）。估计与观察都在市场模型上完成：

  市场基准 r_m(t) = 池内等权日收益（`ctx.ret_clean(1)` 的有效值逐日等权均值；
  有效票数 < cfg.min_cross_section 的交易日 ⇒ NaN —— 与生产市场族 `_ew_return` 同口径；
  idea 允许「沪深300 或池内等权」，本模块选**池内等权**：自包含、PIT 可控、无需第三方指数口径）。

  估计窗 W = 24 周（= 120 个交易日）或 12 周（= 60 个交易日），**结束于 t − S**（S = 跳过日数）：
     r_i(d) = α + β · r_m(d) + e_i(d)        （OLS；同一窗内残差方差估计 σ）
  观察窗 O = 3 周（= 15 个交易日），同样**结束于 t − S**（即窗口为 [t−S−O+1, t−S]，
  与原文“观察 3 周、跳过最近 1 周”一致）：

     factor(t) = Σ_{d∈观察窗} (r_i(d) − α − β·r_m(d)) / (σ · √n_obs)
                 其中 σ = 估计窗内残差标准差（OLS 分解式 SSR/(n−2) 开方，ddof=2）

**方向先验（事前冻结，不得反号）**：残差动量高（剔除市场后仍强）→ 未来 h=1/h=5 正向
（`higher_is_better=True`，declared = raw）。
**证伪点**：h=1 或 h=5 的 declared RankIC HAC95 下界 ≤ 0、或正向年份 < 6/9（政策 v2 G4）
⇒ 先验被证伪，按 H2 不反号、直接否决。

三个预登记变体（期限结构式；不扫网格）：24 周窗/跳 1 周、24 周窗/跳 2 周、12 周窗/跳 1 周
（观察窗恒为 3 周；跳 2 周 = 对照“反转污染是否更彻底”，12 周 = 估计窗长度对照）。

差异论证（H3/H4）：在库动量 `momentum_20/60/120/250` 为**原始收益**、`idio_vol_60` 为
残差波动，均非「标准化残差累计 + 间隔期」；已删 374 项中 `rel_mom_ind_*` 为个股−行业差额
（行业归属口径）、无市场模型标准化残差构造。须测量与 `momentum_20/60/120`、`idio_vol_60`、
`short_term_reversal_5` 的 pooled |ρ| 并报告（G5；>0.85 需论证）。

════════════════════════════════════════════════════════════════════════
二、实现口径（全向量化；无 Python 循环、无逐股回归）
════════════════════════════════════════════════════════════════════════
用**共享有效配对掩码** pair = universe ∧ traded ∧ 双方有限 把 r 与 r_m 同时掩码后，
逐日滚动的 n / Σx / Σy / Σxx / Σxy / Σyy 六项充分统计量给出滚动 OLS：

    β = (n·Σxy − Σx·Σy) / (n·Σxx − (Σx)²)，  α = (Σy − β·Σx) / n
    SSR = Σyy − α·Σy − β·Σxy        （OLS 正规方程给出的残差平方和）
    σ  = √(SSR / (n − 2))

随后把 (α, β, σ, n) 整体 `shift(S)`（= 从「结束于 t−S」的窗取参数），观察窗的
Σy_O / Σx_O / n_O 同样 `shift(S)`，得到 factor(t)。**全部统计量只用 ≤ t 的数据**
（shift 只向过去取），无全样本统计、无前向填充。

**与原文的偏离（README 同步）**：原文为**周频**再平衡（24 个周收益回归、观察 3 周收益）；
本模块在**日频**上取等价结构（周数 × 5 个交易日），原因：引擎面板是日频、eval 在日频
截面；周频会引入「周五效应」与对齐歧义。近似误差 = 用日收益估计 β/σ 比周频噪声更低
（β 更稳），观察窗用日残差累计而非周残差累计（3 周日残差累计 ≈ 3 周累计收益的分解，
但不由周收益复利得到）。列为 README 不确定项。

════════════════════════════════════════════════════════════════════════
三、口径与缺失
════════════════════════════════════════════════════════════════════════
· `r` 用 `ctx.ret_clean(1)`（停牌日 NaN，**不是** 0 —— 与 `ctx.ret` 的停牌贡献 0 语义
  刻意区分，回归不能把停牌日当「0 收益观测」）。
· 最小有效观测：估计窗 n ≥ 80%·W（24 周 = 96，12 周 = 48），观察窗 n_O ≥ 80%·15 = 12；
  不足 ⇒ NaN（不 partial 回退）。分母 σ 需 > 0（退化时 NaN）。
· `stock_daily` / `stock_adj_factor` 均 `delay_days: 0`，不需要 `lag_grid`。
· 停牌/未上市/ST/池外 ⇒ NaN（universe 掩码）。
· warmup_days：24 周变体 300、12 周变体 200（日历天，覆盖 W+S+O 的交易日窗口）。
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("research.families.r12c_residual_momentum_v1")

TRADING_PER_WEEK = 5
# 事前冻结：(est_weeks, skip_weeks, obs_weeks) -> 候选名；观察窗恒 3 周
FROZEN = {
    ("rmom", 24, 1, 3): "r12c_rmom_24_1_3_v1",
    ("rmom", 24, 2, 3): "r12c_rmom_24_2_3_v1",
    ("rmom", 12, 1, 3): "r12c_rmom_12_1_3_v1",
}
MIN_PAIR_FRAC = 0.8      # 估计/观察窗有效观测下限比例


def _frozen_key(parameters):
    if not isinstance(parameters, dict):
        return None
    if parameters.get("kind") != "rmom":
        return None
    return ("rmom", parameters.get("est_weeks"), parameters.get("skip_weeks"),
            parameters.get("obs_weeks"))


def _ew_market(ctx, r, univ):
    """池内等权日收益（有效票数不足的交易日 -> NaN；与生产市场族 `_ew_return` 同口径）。"""
    valid = univ & np.isfinite(r)
    n = valid.sum(axis=1).astype(np.float64)
    tot = np.where(valid, r, 0.0).sum(axis=1)
    return np.where(n >= float(ctx.cfg.min_cross_section), tot / np.maximum(n, 1.0), np.nan)


def build(parameters):
    """只允许预登记的三个变体；不搜索窗口、不搜索跳数、不搜索方向。"""
    key = _frozen_key(parameters)
    if key not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))
    est = key[1] * TRADING_PER_WEEK
    skip = key[2] * TRADING_PER_WEEK
    obs = key[3] * TRADING_PER_WEEK
    min_est = int(MIN_PAIR_FRAC * est)
    min_obs = int(MIN_PAIR_FRAC * obs)

    def compute(ctx):
        r = np.asarray(ctx.ret_clean(1), dtype=np.float64)
        univ = np.asarray(ctx.universe, dtype=bool)
        rm = _ew_market(ctx, r, univ)
        pair = univ & np.isfinite(r) & np.isfinite(rm)[:, None]
        pairf = pair.astype(np.float64)
        rv = np.where(pair, r, np.nan)
        rmv = np.where(pair, rm[:, None], np.nan)

        n = ctx.roll_count(pairf, est, min_count=0)
        sx = ctx.roll_sum(rmv, est, min_count=1)
        sy = ctx.roll_sum(rv, est, min_count=1)
        sxx = ctx.roll_sum(rmv * rmv, est, min_count=1)
        sxy = ctx.roll_sum(rmv * rv, est, min_count=1)
        syy = ctx.roll_sum(rv * rv, est, min_count=1)

        den = n * sxx - sx * sx
        beta = ctx.safe_div(n * sxy - sx * sy, den, min_abs_den=1e-12)
        alpha = ctx.safe_div(sy - beta * sx, n, min_abs_den=1e-9)
        ssr = syy - alpha * sy - beta * sxy
        sig = np.sqrt(np.maximum(ctx.safe_div(ssr, n - 2.0, min_abs_den=1e-18), 0.0))

        n_o = ctx.shift(ctx.roll_count(pairf, obs, min_count=0), skip)
        sx_o = ctx.shift(ctx.roll_sum(rmv, obs, min_count=1), skip)
        sy_o = ctx.shift(ctx.roll_sum(rv, obs, min_count=1), skip)
        a_e = ctx.shift(alpha, skip)
        b_e = ctx.shift(beta, skip)
        s_e = ctx.shift(sig, skip)
        n_e = ctx.shift(n, skip)

        num = sy_o - a_e * n_o - b_e * sx_o
        ok = (n_e >= min_est) & (n_o >= min_obs) & (s_e > 0.0)
        ok &= np.isfinite(num) & np.isfinite(a_e) & np.isfinite(b_e) & np.isfinite(s_e) & univ
        with np.errstate(all="ignore"):
            value = num / (s_e * np.sqrt(n_o))
        value = np.where(ok & np.isfinite(value), value, np.nan)
        log.info("r12c_rmom est=%d skip=%d obs=%d panel_finite=%d",
                 est, skip, obs, int(np.isfinite(value).sum()))
        return np.ascontiguousarray(value, dtype=np.float64)

    return compute
