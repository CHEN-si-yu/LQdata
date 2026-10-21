"""Round 11 · 单元 U3 · 市场状态门控的**个股形状**条件因子（family `r11u3_market_gated_v1`）。

前置事实（决定了本单元的形状）
══════════════════════════════════════════════════════════════════════
1. `research/evaluation.py::evaluate_run` 对 `spec.is_market=True` 有硬分支：只输出
   `evaluation_type="market_context_quality_only"`、`rankic=None, ic=None`
   （实测源码 L45–L55 / L68–L73）⇒ **市场标量在框架内不可能产出 RankIC**。
   所以本单元的交付物是**股票形状的 (T, C) 交互候选**（`is_market` 保持默认 False），
   市场量只作**有符号状态**在 `compute(ctx)` 内部构造。
2. 项目内市场标量的预测性通道已被系统性检验：`doc/市场因子清理记录_20261004.md` 记录
   27 个构造型市场因子 **M5 预测检验 0/27**，全数删除。⇒ 不再登记任何独立市场标量。

★ 结构性恒等式（本单元全部推理的基石，R9-F 已实测；Lead 2026-10-05 已就此裁定）
──────────────────────────────────────────────────────────────────────
引擎逐日 winsor → z-score，而 z 对正尺度不变，故对 `factor = state_t × leg_{i,t}`
（state 为 `(T,)` 标量序列）有

        z_t(factor) = sign(state_t) · z_t(leg)          （逐日**精确**成立）

⇒ (a) 候选的逐日 RankIC `IC_cand(t) = sign(state_t) · IC_leg(t)`；
   (b) 候选的**逐日截面秩恒等于腿的秩**（只差当日符号）。

★★ **这是本候选的头号结构性局限，必须与结论一起读（Lead 裁定原文口径）**：
   候选**每一天的截面排序都不携带超出模块内自算腿的任何信息**；全部增量信息在
   **一个日度符号的时间序列**上 —— 策略等价于「市场参与度高于其近一年常态时持腿多头、
   低于时持空头」。这是一个**有真实预登记机制的条件/择时构造**，但**不是新的截面排序**；
   它的 G4 RankIC 在算术上**就是 P3 的斜率项**。因此本单元**以 P3 斜率 t 为唯一证据列**，
   G4 通过**不得**被表述为新因子的证据。

★ 裁定要点（区分两种「腿复制」）：
   · R9 的 F1/F3 被否 —— 它们的腿**本身就是在产在册因子**（`turnover_std_20`、
     `short_term_reversal_5`），逐日 ±1 的带符号副本 = 对一个已在册对象的**纯重复** ⇒ 不合格。
   · 本候选**不是**这种情况：腿是模块内自算的 120 日**成交额参与 β**，库内没有该对象
     （P1 实测：对全部 125 个在产股票因子，逐日 |ρ|>0.99 的天数占比 ≈ 0）。
     「相对**自己的私有腿**没有独立截面变化」**本身不构成取消资格**。
   · 但必须**如实标注**：对**自己的腿**，`frac_days_abs_rho_gt_0.99 = 1.0000`
     （精确带符号置换），并同时报 `daily_abs_mean` 与 dedup 合并 ρ
     （合并 ρ ≈ 状态为正的天数占比 − 状态为负的天数占比），以免后来者误判为 F1 式重复。

预登记的五个假设（方向全部先于任何结果统计冻结；冻结件 sha256 = b04f2560dc571a4f）
══════════════════════════════════════════════════════════════════════
状态（全部 z252：`(x − roll_mean252(x)) / roll_var252(x)^0.5`，`min_count=120`、`ddof=0`，
沿用 R9-F / `r4n_market_b2_v1` 的已冻结市场腿口径；**有符号**是交互能成立的前提）：

* `S_AMT`    = `z252(log(median over pool of amount))`
               （口径同在生产 `r4o_mkt_amount_median_v1_fix1_36ab6152c0703e8c`，多一层 log）
* `S_DISP`   = `z252(std over pool of ret_clean(1), ddof=0)`（口径同在生产 `mkt_dispersion`）
* `S_MFB`    = `z252(count(lg+elg 净额>0)/count(有效))`
               （口径同在生产 `mkt_mf_large_breadth`，源 `stock_main_fund_flow`）

腿（全部 120 日窗，min_count=60；半 β 用 min_count=25 个同类日）：

* `L_AMTB`   = `roll_cov(log amount_i, log median-pool-amount, 120, 60) / roll_var(·, 120, 60)`
               —— 个股成交额对全市场成交额的**敏感度**（资金参与 β）。
* `L_DBETA`  = 仅在池内等权收益 `m < 0` 的日子上估
               `roll_cov(r_i, m, 120, 25) / roll_var(m, 120, 25)` —— **下行半 β**。
* `L_ILB`    = `roll_cov(log illiq_i, log median-pool-illiq, 120, 60) / roll_var(·, 120, 60)`，
               `illiq = |ret_clean(1)| / amount` —— **流动性风险 β**。

五个预登记假设与**冻结方向**（`higher_is_better`）：

| # | signal | 状态 × 腿 | 冻结方向 | 机制（一句话） |
|---|---|---|---|---|
| H1 | `mktliq_x_amountbeta`   | S_AMT  × L_AMTB  | **false** | 参与度高于常态 ⇒ 增量资金把高参与 β 推过均衡价后回吐 |
| H2 | `mktvol_x_downbeta`     | S_DISP × L_DBETA | false     | 高波动 ⇒ 融资约束收紧、高下行 β 被去杠杆式抛售 |
| H3 | `mktflow_x_illiqbeta`   | S_MFB  × L_ILB   | false     | 大单净买入面广 ⇒ 高流动性风险 β 被资金吸纳后回吐 |
| H4 | `mktvol_x_illiqbeta`    | S_DISP × L_ILB   | true      | 高波动 ⇒ 总量流动性风险被重新定价（正补偿） |
| H5 | `mktliq_x_downbeta`     | S_AMT  × L_DBETA | false     | 参与度高 = 风险偏好高 ⇒ 下行风险被低估、随后跑输 |

★★ **强制预筛结果（本模块写死只准入通过者）** ★★
（`/tmp/r11u3/ps/*.json`，全区间 2018-01-02~2026-09-30、2115 只；离线预筛口径无 warmup，
故有效日 1,870，比框架正式 run 少约 250 日）

* **P2（腿无独立信号，|HAC t| < 2，h=1 与 h=5）**
  `L_AMTB` **t=+0.92 / +0.68 PASS**；`L_DBETA` t=−0.29 / +0.49 PASS；`L_ILB` **t=+3.14 / +2.88 FAIL**；
  参照腿 `beta_120`（R9-F 的 F2 腿）本管线复算 t=−1.36，框架实测 −1.60 ⇒ 管线自校验通过。
  另测 `beta_up_120`(−1.51)、`beta_asym_120`(−1.20)、`coskew_120`(**−6.55 FAIL**) 均未用。
* **P3（`IC_leg ~ a + b·sign(state)`，HAC |t(b)| ≥ 3，且两状态条件 IC 反号各自显著）**
  H1 **斜率 t = −4.09 (h=1) / −3.82 (h=5) PASS**（状态+ IC=−0.00862 vs 状态− IC=+0.01748，反号）；
  H2 t=−0.93/+0.75 **FAIL**（且两期限斜率异号）；H5 t=+0.46/+0.63 **FAIL**（且与冻结方向反号）；
  H3/H4 因腿 P2 失败而终止。
* **P1（与任一在产股票因子逐日 |ρ|>0.99 ≈ 0）** 见 family/report 与
  `artifacts/research/round11_20261005/u3_market_gated/`。
* ⇒ **只有 H1 通过全部三条，`ADMITTED` 只含 H1。** 其余四个假设以 not-registered / 证伪记录，
  **不占候选、不反号重登**。下面的 `_SIGNALS` 保留全部五个实现，是为了让预登记可审计；
  `build()` 只放行 `ADMITTED`。

§8.8 强制重蹈检查（`doc/因子清理记录_20261003.md`，374 个已删股票因子 + 125 在产）
──────────────────────────────────────────────────────────────────────
* 已删 `ret_autocorr_1d_20` / `ret_efficiency_20` / `trend_strength_60`（自相关 / 路径效率 / 趋势强度）
  —— 本模块**没有**任何自相关或路径类腿。
* 已删 `downside_upside_vol_60` / `downside_vol_ratio_20` / `sortino_ratio_60` / `gain_loss_asymmetry_60`
  全是**个股自身波动**的上下行之比；`L_DBETA` 是**对市场收益的半协方差**，不含自身波动，不同轴。
* 已删 `bw_beta_60` / `bw_beta_asym_20` 的回归量是**宽度变化**、被解释量是**收益**；
  `L_AMTB` 是**成交额对成交额**、`L_ILB` 是**非流动性对非流动性**，回归量与被解释量都不同。
* 已删 `liquidity_shock_20`（自身流动性**水平变化**）、`amihud_asymmetry_20`（自身涨跌侧**比值**）
  都是自身量；在产 `s26_liq_up_fragility_v1` 是"上涨日收益-成交额弹性"。`L_ILB` 是**对市场的协动**。
* 已删 `ind_beta_60` 是**行业** β；本模块的 β 全部对**池内等权市场**。
* 在产 `beta_60` 对 **index_daily** 的 60 日 β（deps 含 index_daily）；`L_DBETA` 是对池内等权收益的
  **120 日下行半 β**。P1 实测两者逐日 |ρ| 远低于 0.99。

PIT（可得性）合同
-----------------
* 状态与腿只用到 `stock_daily`（T 日收盘后定稿）与 `stock_adj_factor`；`ctx.ret_clean(1)` 把停牌日
  置 NaN；`z252` 只用 ≤T 的历史。T 日取值、用于 T+1 开盘起的标签，**无前视**。
* `amount` 是 T 日**盘中累计**成交额，T 日收盘即定稿（生产 `amihud_daily_5`、
  `r4o_mkt_amount_median_v1_fix1` 同口径同可得性）。
* 状态是有符号 z252：若状态恒正，交互项在**每一天**的截面秩恒等于腿本身（结构性同构），
  逐日 IC 与腿完全相同 —— 有符号 z 才让「状态是否调制了腿的收益关系」可被检验。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 预登记的全部五个 signal（保留实现以便审计）
PREREGISTERED = (
    "mktliq_x_amountbeta",
    "mktvol_x_downbeta",
    "mktflow_x_illiqbeta",
    "mktvol_x_illiqbeta",
    "mktliq_x_downbeta",
)
# ★ 只有通过 P1/P2/P3 全部门槛的假设才被放行（见模块 docstring 的预筛结果）
ADMITTED = ("mktliq_x_amountbeta",)

PARAMETERS = {"signal": "mktliq_x_amountbeta"}

STATE_WINDOW = 252
STATE_MIN_COUNT = 120
BETA_WINDOW = 120
BETA_MIN_COUNT = 60
SEMI_MIN_COUNT = 25
MIN_CROSS = 100
ILLIQ_EPS = 1e-12

FF_COLS = ["trade_date", "stock_code", "buy_lg_amount", "sell_lg_amount",
           "buy_elg_amount", "sell_elg_amount"]


# ──────────────────────────────────────────────────────────── 池 / 聚合工具
def _valid(ctx) -> np.ndarray:
    """池掩码：冻结票池 ∩ 当日真有成交（停牌 / 未上市 / 已退市为 False）。"""
    return np.asarray(ctx.universe, dtype=bool) & np.asarray(ctx.traded(), dtype=bool)


def _pool_mean(ctx, mat: np.ndarray) -> np.ndarray:
    """逐日池内均值（有效票 < MIN_CROSS 时判缺失，不拿稀疏均值冒充市场状态）。"""
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=1)
    with np.errstate(all="ignore"):
        m = np.nanmean(np.where(ok, mat, np.nan), axis=1)
    return np.where(cnt >= MIN_CROSS, m, np.nan)


def _pool_std(ctx, mat: np.ndarray) -> np.ndarray:
    """逐日池内总体标准差（ddof=0，口径同生产 `mkt_dispersion`）。"""
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=1).astype(np.float64)
    with np.errstate(all="ignore"):
        m = np.nanmean(np.where(ok, mat, np.nan), axis=1)
        dev = np.where(ok, mat - m[:, None], 0.0)
        var = (dev * dev).sum(axis=1) / np.maximum(cnt, 1.0)
    return np.where(cnt >= MIN_CROSS, np.sqrt(np.maximum(var, 0.0)), np.nan)


def _pool_median(ctx, mat: np.ndarray) -> np.ndarray:
    """逐日池内中位数（口径同生产 `r4o_mkt_amount_median_v1_fix1` / `mkt_illiquidity_median`）。"""
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=1)
    out = np.full(mat.shape[0], np.nan)
    for t in range(mat.shape[0]):
        if cnt[t] >= MIN_CROSS:
            row = mat[t][ok[t]]
            if row.size:
                out[t] = float(np.median(row))
    return out


def _z252(ctx, series: np.ndarray) -> np.ndarray:
    """252 日滚动 z（min_count=120，ddof=0）—— 与 r4n / R9-F 冻结口径逐位一致。"""
    col = np.asarray(series, dtype=np.float64)[:, None]
    mean = ctx.roll_mean(col, STATE_WINDOW, min_count=STATE_MIN_COUNT)
    var = ctx.roll_var(col, STATE_WINDOW, min_count=STATE_MIN_COUNT, ddof=0)
    sd = np.sqrt(np.where(var > 0, var, np.nan))
    with np.errstate(all="ignore"):
        z = (col - mean) / sd
    return np.where(np.isfinite(z[:, 0]), z[:, 0], np.nan)


def _col_to_grid(ctx, frame: pd.DataFrame, col: str) -> np.ndarray:
    """把长表的一列精确落格到面板 (T, C)（缺失补 NaN，不做前向填充）。"""
    idx = ctx.code_index(frame["stock_code"].astype(str).to_numpy())
    day = ctx.date_col(frame["trade_date"])
    val = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
    return ctx.panel.place(idx, day, val)


# ──────────────────────────────────────────────────────────── 市场状态（(T,) 有符号）
def _state_amount(ctx) -> np.ndarray:
    """S_AMT = z252(log(池内成交额中位数))：市场参与度 / 流动性状态。"""
    a = np.asarray(ctx.px("amount"), dtype=np.float64)
    valid = _valid(ctx) & np.isfinite(a) & (a > 0)
    med = _pool_median(ctx, np.where(valid, a, np.nan))
    with np.errstate(all="ignore"):
        log_med = np.log(np.where(med > 0, med, np.nan))
    return _z252(ctx, log_med)


def _state_disp(ctx) -> np.ndarray:
    """S_DISP = z252(池内日收益截面标准差)：市场波动 / 离散度状态（同 mkt_dispersion 口径）。"""
    r = np.asarray(ctx.ret_clean(1), dtype=np.float64)
    valid = _valid(ctx) & np.isfinite(r)
    return _z252(ctx, _pool_std(ctx, np.where(valid, r, np.nan)))


def _state_flow(ctx) -> np.ndarray:
    """S_MFB = z252(大单净流入家数占比)：市场资金流 / 情绪状态（同 mkt_mf_large_breadth 口径）。"""
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    y0, y1 = int(dates[0]) // 10000, int(dates[-1]) // 10000
    frame = ctx.dataset("stock_main_fund_flow", columns=list(FF_COLS), years=(y0, y1))
    if frame is None or frame.empty:
        return np.full(dates.size, np.nan)
    buy = _col_to_grid(ctx, frame, "buy_lg_amount") + _col_to_grid(ctx, frame, "buy_elg_amount")
    sell = _col_to_grid(ctx, frame, "sell_lg_amount") + _col_to_grid(ctx, frame, "sell_elg_amount")
    with np.errstate(all="ignore"):
        net = buy - sell
    valid = _valid(ctx) & np.isfinite(net)
    cnt = valid.sum(axis=1)
    num = (valid & (net > 0)).sum(axis=1)
    with np.errstate(all="ignore"):
        breadth = np.where(cnt >= MIN_CROSS, num / np.maximum(cnt, 1), np.nan)
    return _z252(ctx, breadth)


# ──────────────────────────────────────────────────────────── 个股腿（(T, C) 自算量）
def _leg_amount_beta(ctx) -> np.ndarray:
    """L_AMTB：个股成交额对全市场成交额中位数的 120 日敏感度（资金参与 β）。"""
    a = np.asarray(ctx.px("amount"), dtype=np.float64)
    valid = _valid(ctx) & np.isfinite(a) & (a > 0)
    with np.errstate(all="ignore"):
        la = np.log(np.where(valid, a, np.nan))
    med = _pool_median(ctx, la)
    mkt = np.repeat(med[:, None], la.shape[1], axis=1)
    cov = ctx.roll_cov(la, mkt, BETA_WINDOW, BETA_MIN_COUNT)
    var = ctx.roll_var(mkt, BETA_WINDOW, BETA_MIN_COUNT)
    with np.errstate(all="ignore"):
        beta = np.where(var > 0, cov / var, np.nan)
    return np.where(valid, beta, np.nan)


def _leg_down_beta(ctx) -> np.ndarray:
    """L_DBETA：仅在池内等权收益为负的日子上估的 120 日下行半 β。"""
    r = np.asarray(ctx.ret_clean(1), dtype=np.float64)
    valid = _valid(ctx) & np.isfinite(r)
    mkt = _pool_mean(ctx, np.where(valid, r, np.nan))
    down = np.isfinite(mkt) & (mkt < 0)
    mask = np.where(down, 1.0, np.nan)[:, None]
    r_d = r * mask
    mkt2 = np.repeat(mkt[:, None], r.shape[1], axis=1)
    m_d = mkt2 * mask
    cov = ctx.roll_cov(r_d, m_d, BETA_WINDOW, SEMI_MIN_COUNT)
    var = ctx.roll_var(m_d, BETA_WINDOW, SEMI_MIN_COUNT)
    with np.errstate(all="ignore"):
        beta = np.where(var > 0, cov / var, np.nan)
    return np.where(valid, beta, np.nan)


def _leg_illiq_beta(ctx) -> np.ndarray:
    """L_ILB：个股 log 非流动性对市场 log 非流动性的 120 日敏感度（流动性风险 β）。"""
    r = np.asarray(ctx.ret_clean(1), dtype=np.float64)
    a = np.asarray(ctx.px("amount"), dtype=np.float64)
    valid = _valid(ctx) & np.isfinite(r) & np.isfinite(a) & (a > 0)
    with np.errstate(all="ignore"):
        illiq = np.abs(r) / np.where(a > 0, a, np.nan)
        li = np.log(np.where(illiq > 0, illiq, np.nan) + ILLIQ_EPS)
    li = np.where(valid, li, np.nan)
    med = _pool_median(ctx, li)
    mkt = np.repeat(med[:, None], li.shape[1], axis=1)
    cov = ctx.roll_cov(li, mkt, BETA_WINDOW, BETA_MIN_COUNT)
    var = ctx.roll_var(mkt, BETA_WINDOW, BETA_MIN_COUNT)
    with np.errstate(all="ignore"):
        beta = np.where(var > 0, cov / var, np.nan)
    return np.where(valid, beta, np.nan)


def _interaction(ctx, state_fn, leg_fn):
    state = state_fn(ctx)
    if state is None:
        return None
    leg = leg_fn(ctx)
    if leg is None:
        return None
    with np.errstate(all="ignore"):
        grid = np.asarray(state, dtype=np.float64)[:, None] * leg
    return np.where(np.isfinite(state)[:, None] & np.isfinite(leg), grid, np.nan)


_SIGNALS = {
    "mktliq_x_amountbeta": lambda ctx: _interaction(ctx, _state_amount, _leg_amount_beta),
    "mktvol_x_downbeta": lambda ctx: _interaction(ctx, _state_disp, _leg_down_beta),
    "mktflow_x_illiqbeta": lambda ctx: _interaction(ctx, _state_flow, _leg_illiq_beta),
    "mktvol_x_illiqbeta": lambda ctx: _interaction(ctx, _state_disp, _leg_illiq_beta),
    "mktliq_x_downbeta": lambda ctx: _interaction(ctx, _state_amount, _leg_down_beta),
}


def build(parameters):
    """builder：只放行通过 P1/P2/P3 全部门槛的预登记 signal（family allowed_variants 逐项冻结）。"""
    params = dict(parameters)
    signal = params.get("signal")
    if signal not in ADMITTED:
        raise ValueError(
            "Family admits only pre-screened signals passing P1/P2/P3: "
            f"{sorted(ADMITTED)}; got {signal!r} "
            f"(pre-registered but not admitted: {sorted(set(PREREGISTERED) - set(ADMITTED))})")
    if set(params) != {"signal"}:
        raise ValueError(f"Unregistered parameters: {sorted(set(params) - {'signal'})}")
    fn = _SIGNALS[str(signal)]

    def compute(ctx):
        value = fn(ctx)
        if value is None:
            value = ctx.panel.empty(dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    compute.__name__ = "r11u3_" + str(signal)
    return compute
