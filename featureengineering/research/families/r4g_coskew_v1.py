"""Round 4 · 单元 U7：协同风险分解（coskew / downside beta）——只读价格层 + 自算池等权市场收益。

机制集合（任务书 B2 §2 事前冻结，4 个，方向不可改）：
  1. `coskew_60`         60 日窗口：r 对 [r_m, r_m²] 的**标准化偏回归** r_m² 系数（方向 false）
  2. `downside_beta_60`  60 日窗口：仅 r_m<0 日的市场 beta（方向 true）
  3. `coskew_120`        窗口变体（标注变体性质，方向 false）
  4. `downside_beta_120` 窗口变体（标注变体性质，方向 true）

口径（事前冻结，报告披露）：
  · 日收益 r = `ctx.ret_clean(1)` = 后复权日收益，停牌/未上市/当日无成交 → NaN
    （**不按 0 收益参与窗口**），|r|>60% 的复权脏数据由价格层置 NaN；再按
    `ctx.universe` 掩码收口（主板 + 上市窗口）。
  · 市场 `mkt_ew[T]` = 冻结池内**当日**等权平均日收益：对 `ctx.universe & isfinite(r)`
    的格子逐日求均值（要求 ≥ max(100, cfg.min_cross_section) 只，池 2115 只下从不触发），
    只用同日已实现收益，无未来信息。与 U5 (r4e) 完全同口径。
    ★ 与正式因子 `beta_60` 的「沪深300」基准**不同**：本单元按任务书固定为等权池均值；
    正式库的 `bw_beta_*`（宽度变化）/ `ind_beta_60`（行业指数）等已删因子回归量互不相同
    （见 report.md 防重蹈一节）。自包含偏差：池均值含自身权重 1/n≈0.05%，忽略。
  · `coskew` = 标准化偏回归系数 b2（z_r ~ [z_m, z_m²] 窗口内 OLS 的 z_m² 系数），闭式：
      b2 = (S_y2 − γ·ρ) / (Var(z_m²) − γ²)，S_y2 = E[z_r·z_m²]、γ = E[z_m³]（窗口偏度）、
      ρ = 窗口相关、Var(z_m²) = κ − 1（κ = E[z_m⁴]）。
    ★ 为什么用标准化版而不是原始量纲 OLS 系数：原始系数 ≈ b2·σ_r/σ_m²，在横截面上与
    个股波动水平成正比（机械共线），会把因子变成波动代理；标准化版去量纲，是同一回归
    的尺度重参数化（任务书允许的两种表达之一，事前定死为标准化版）。
    全部窗口矩在**成对有效日**上计算（min_count = 45 / 90 = ceil(0.75n)）。
  · `downside_beta` = 窗口内仅 `r_m<0` 且成对有效的日子上 r 对 r_m 的 OLS 斜率
    （60/120 日窗，min 8 / 16 个下行日；任务书固定）。
  · 数值守卫（退化窗口 → NaN，不 clip）：σ² 守卫 1e-10；coskew 偏回归分母守卫 1e-4
    （分母 = Var(z_m²) − γ² ≥ 0 恒成立，仅两点分布市场退化到 0）；
    downside beta 分母守卫 1e-12（= n_down·Var(r_m | r_m<0) > 0）。
"""
from __future__ import annotations

import numpy as np

SIGNALS = ("coskew_60", "downside_beta_60", "coskew_120", "downside_beta_120")

# 事前冻结的窗口参数（不随结果调整）
_WINDOW = {"coskew_60": 60, "downside_beta_60": 60, "coskew_120": 120, "downside_beta_120": 120}
_COSKEW_MIN = {60: 45, 120: 90}     # ceil(0.75 × n)，与库内 ret_skew_20（15/20）同比例
_DOWN_MIN = {60: 8, 120: 16}        # 任务书固定：60 日窗 min 8 个下行日
_MKT_MIN_NAMES = 100                # 池内等权均值的健康度下限（池 2115 只，从不触发）
_VAR_FLOOR = 1e-10                  # σ² 退化守卫
_COSKEW_DENOM_FLOOR = 1e-4          # Var(z_m²) − γ² 退化守卫（真实窗口 ≈ O(1)）
_DOWN_DENOM_FLOOR = 1e-12           # n_down·Var(r_m|down) 退化守卫

__all__ = ["SIGNALS", "build"]


def _market_ew_return(ctx, r: np.ndarray) -> np.ndarray:
    """冻结池内当日等权平均日收益 (T,)（与 U5 r4e 同口径）。"""
    valid = np.asarray(ctx.universe, dtype=bool) & np.isfinite(r)
    count = valid.sum(axis=1)
    total = np.where(valid, r, 0.0).sum(axis=1)
    mean = np.full(r.shape[0], np.nan, dtype=np.float64)
    ok = count >= max(_MKT_MIN_NAMES, int(ctx.cfg.min_cross_section))
    mean[ok] = total[ok] / count[ok]
    return mean


def _coskew(ctx, r: np.ndarray, mkt: np.ndarray, n: int, mc: int) -> np.ndarray:
    """60/120 日标准化协同偏度：z_r ~ [z_m, z_m²] 的 z_m² 偏回归系数（闭式，见模块 docstring）。"""
    rm = np.where(np.isfinite(r), mkt[:, None], np.nan)   # 成对掩码：仅在 r 有效日计市场矩
    rm2 = rm * rm
    m1 = ctx.roll_mean(rm, n, min_count=mc)
    m2 = ctx.roll_mean(rm2, n, min_count=mc)
    m3 = ctx.roll_mean(rm2 * rm, n, min_count=mc)
    m4 = ctx.roll_mean(rm2 * rm2, n, min_count=mc)
    r1 = ctx.roll_mean(r, n, min_count=mc)
    r2 = ctx.roll_mean(r * r, n, min_count=mc)
    c1 = ctx.roll_mean(r * rm, n, min_count=mc)
    c2 = ctx.roll_mean(r * rm2, n, min_count=mc)
    var_m = np.maximum(m2 - m1 * m1, 0.0)
    var_r = np.maximum(r2 - r1 * r1, 0.0)
    with np.errstate(all="ignore"):
        sig_m = np.sqrt(var_m)
        sig_r = np.sqrt(var_r)
        cov = c1 - r1 * m1
        # E[(r−μr)(rm−μm)²] = (E[r·rm²] − μr·E[rm²]) − 2μm·Cov(r, rm)
        cross2 = (c2 - r1 * m2) - 2.0 * m1 * cov
        gamma = (m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3) / (sig_m ** 3)
        kurt = (m4 - 4.0 * m1 * m3 + 6.0 * m1 ** 2 * m2 - 3.0 * m1 ** 4) / (var_m * var_m)
        rho = cov / (sig_r * sig_m)
        s_y2 = cross2 / (sig_r * var_m)
        denom = (kurt - 1.0) - gamma * gamma
        value = (s_y2 - gamma * rho) / denom
    ok = (var_m > _VAR_FLOOR) & (var_r > _VAR_FLOOR) & (denom > _COSKEW_DENOM_FLOOR)
    return np.where(ok, value, np.nan)


def _downside_beta(ctx, r: np.ndarray, mkt: np.ndarray, n: int, mc: int) -> np.ndarray:
    """60/120 日下行贝塔：仅 r_m<0（且 r 有效）的日子上 r 对 r_m 的 OLS 斜率。"""
    market = mkt[:, None]
    down = np.isfinite(r) & (market < 0.0)
    rd = np.where(down, r, np.nan)
    md = np.where(down, market, np.nan)
    cnt = ctx.roll_count(rd, n, min_count=mc)
    s_r = ctx.roll_sum(rd, n, min_count=mc)
    s_m = ctx.roll_sum(md, n, min_count=mc)
    s_rm = ctx.roll_sum(rd * md, n, min_count=mc)
    s_mm = ctx.roll_sum(md * md, n, min_count=mc)
    with np.errstate(all="ignore"):
        denom = s_mm - s_m * s_m / cnt
        value = (s_rm - s_r * s_m / cnt) / denom
    ok = np.isfinite(cnt) & (denom > _DOWN_DENOM_FLOOR)
    return np.where(ok, value, np.nan)


def build(parameters):
    """四个预登记机制之一；不搜索窗口、不搜索方向。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError(f"Batch permits only the four preregistered signal identifiers: {SIGNALS}")
    signal = parameters["signal"]
    n = _WINDOW[signal]
    coskew = signal.startswith("coskew")
    mc = _COSKEW_MIN[n] if coskew else _DOWN_MIN[n]

    def compute(ctx):
        universe = np.asarray(ctx.universe, dtype=bool)
        r = np.asarray(ctx.ret_clean(1), dtype=np.float64)
        r = np.where(universe, r, np.nan)
        mkt = _market_ew_return(ctx, r)
        value = _coskew(ctx, r, mkt, n, mc) if coskew else _downside_beta(ctx, r, mkt, n, mc)
        # 输出按池掩码与有限性收口；NaN 语义 = 不知道（不填 0）。
        return np.where(universe & np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
