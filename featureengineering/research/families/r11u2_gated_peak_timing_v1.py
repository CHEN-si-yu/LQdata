"""Round 11 · 单元 U2（股票）· **条件门控日内结构**：成交量峰值时点 × 会话状态。

family `r11u2_gated_peak_timing_v1`
════════════════════════════════════════════════════════════════════════════
一、为什么是「成交量峰值**时点**」这条腿
────────────────────────────────────────
`stock_history_5min` 在产库已有 11 个 `intraday` + 3 个 `research_intraday` 因子，
但**全是无条件量**（水平/比值/波动/集中度）。本项目实测（`tools/prescreen.py`，
2018-01-02~2026-09-30，2,123 交易日 × 2,115 只）横扫 26 条日内形状腿 × 2 个期限后：

| 腿 | 日度 RankIC h=1 | HAC t |
|---|---|---|
| `max_ru`（日内最大上冲） | −0.04226 | −13.72 |
| `amt_conc`（成交额集中度） | −0.02251 | −10.65 |
| `intraday_ret`（日内收益） | −0.03104 | −8.81 |
| …（其余 23 条全部 \|t\| ≥ 2.6）… | | |
| **`vol_peak_pos`（放量时点）** | **+0.00086** | **+0.60** |

**26 条腿里只有一条自身没有无条件信号**：`vol_peak_pos` = 当日最大成交量那根
5 分钟棒的归一化时点（0 = 09:35 那根，1 = 15:00 那根）。
其余 25 条的无条件 IC 都显著 —— 按 P2 判据它们只是既有波动/收益因子的投影。

二、状态（P3 实测）
──────────────────
状态 = 当日**池内等权午后收益**的 z252（`pm_ret = pm_close5/pm_open5 − 1`）。
横扫 26 腿 × 25 状态（650 组）后，**唯一**同时满足
「P2 腿不显著 + P3 斜率 HAC |t| ≥ 3 + 两状态腿 IC 符号相反且各自 |t| ≥ 2」的组合是：

    腿 `vol_peak_pos` × 状态 `z252(市场午后收益)`
    IC(state>0) = +0.00580 (t=+2.73, n=1060)
    IC(state<0) = −0.00457 (t=−2.12, n=942)
    斜率（IC_leg ~ sign(state)）HAC t = +3.32

三、★ 结构约束（Round 9 教训的正面应用）
────────────────────────────────────────
任何「(T,) 标量状态 × 单一 (T,C) 腿」的乘积，在引擎逐日 winsor→z-score 之后
**逐日截面秩恒等于该腿的秩 × 当日符号**（z-score 对正尺度不变 ⇒ |state| 被抹掉）。
本项目实测：纯交互 `sign(state)×(vol_peak_pos−0.5)` 与腿的逐日 |Spearman| **恒为 1.0000**
（2,123/2,123 天），子夜离散取值只有 48 个 ⇒ 它是腿的**逐日秩置换**，
不构成新的截面信息，**本 family 不登记纯交互**。

因此本 family 的每个变体都必须携带**逐股可变的门控量**，使当日截面排序**不是**腿的
单调变换。两个变体分别用两种不同的门控：

* 变体 A `mkt_pm_state_x_shape_gate` —— 门控是**同日个股形状一致度**
  `g = ½(z_cs(vol_peak_pos − hi_pos) + z_cs(am_ret − pm_ret))`，
  即「放量是在价格高点**之前**还是**之后**」与「收益路径是前重还是后重」。
* 变体 B `own_pm_rel_state` —— 门控是**个股自身相对市场的午后收益符号**
  `sign(pm_ret_i − mean_j pm_ret_j)`；状态本身下放到逐股，截面天然异质。

实测（同窗口）：A 的逐日 |ρ| vs 腿 = 0.547（max 0.872，frac>0.99 = 0），
B 的逐日 |ρ| vs 腿 = 0.127（max 0.603，frac>0.99 = 0）⇒ 两者都**不是**秩置换。

四、方向（先于任何候选级结果冻结）
──────────────────────────────────
`higher_is_better = true`。理由只来自上面的 P3 预筛：
状态为正时腿的 IC 为正（+0.00580），状态为负时腿的 IC 为负（−0.00457）；
`sign(state)×leg` 使 IC 在两个状态下都为正，**故 declared 方向为正**。
登记后不看 G4/G7 改方向、不反号、不注册 v2 扳方向。

五、数据边界与口径
──────────────────
* `stock_history_5min` 冻结在 **2026-09-11**（`factors/intraday.py` §8 实测），
  之后的日子日内层没有行 ⇒ 因子为 NaN，**不前向填充**。
* 用 `ctx.intraday_field(...)`（`data/derived/intraday`：48 根/日预聚合的日频宽表），
  停牌/未上市当日**没有行** ⇒ 精确落格为 NaN。
* `vol_peak_pos / hi_pos` 已是 [0,1] 归一化时点（`fea/intraday.py`：argmax ÷ (n_bars−1)），
  **不再除 n_bars**。
* 市场状态用池内有效值等权均值（`nanmean` 语义），不做前向填充。
"""
from __future__ import annotations

import numpy as np

PARAMETERS = {"signal": "mkt_pm_state_x_shape_gate"}

SIGNALS = ("mkt_pm_state_x_shape_gate", "own_pm_rel_state")

STATE_WINDOW = 252       # z252（沿用 r9f / r4n 冻结的市场腿口径）
STATE_MIN_COUNT = 120
GATE_LAMBDA = 1.0
LEG_CENTER = 0.5         # vol_peak_pos 的中位量级；减常数不改变逐日秩，只让门控可比


# ────────────────────────────────────────────────────────────────────
# 日内层字段
# ────────────────────────────────────────────────────────────────────
def _field(ctx, name):
    v = np.asarray(ctx.intraday_field(name), dtype=np.float64)
    return np.where(np.isfinite(v), v, np.nan)


def _leg(ctx):
    """腿：当日最大成交量那根 5 分钟棒的归一化时点 ∈ [0,1]。"""
    return _field(ctx, "vol_peak_pos")


def _seg_ret(ctx, o_name, c_name):
    o = _field(ctx, o_name)
    c = _field(ctx, c_name)
    ok = np.isfinite(o) & np.isfinite(c) & (o > 0) & (c > 0)
    with np.errstate(all="ignore"):
        r = c / o - 1.0
    return np.where(ok, r, np.nan)


def _am_pm_diff(ctx):
    """上午段收益 − 下午段收益（同日内两段，段内比值不需复权）。"""
    am = _seg_ret(ctx, "am_open5", "am_close5")
    pm = _seg_ret(ctx, "pm_open5", "pm_close5")
    ok = np.isfinite(am) & np.isfinite(pm)
    return np.where(ok, am - pm, np.nan)


def _pm_ret(ctx):
    return _seg_ret(ctx, "pm_open5", "pm_close5")


def _cs_z(mat):
    """逐日截面 z（ddof=1，NaN 保留）；只用于让门控与常数 1 可比。"""
    P = np.asarray(mat, dtype=np.float64)
    fin = np.isfinite(P)
    X = np.where(fin, P, 0.0)
    n = fin.sum(axis=1)
    mu = X.sum(axis=1) / np.maximum(n, 1)
    D = np.where(fin, X - mu[:, None], 0.0)
    sd = np.sqrt((D ** 2).sum(axis=1) / np.maximum(n - 1, 1))
    Z = D / np.where(sd > 0, sd, 1.0)[:, None]
    return np.where(fin, Z, np.nan)


def _z252(ctx, series):
    """252 日滚动 z（min_count=120，ddof=0）—— 与 r9f 冻结的市场腿口径逐位一致。"""
    col = np.asarray(series, dtype=np.float64)[:, None]
    mean = ctx.roll_mean(col, STATE_WINDOW, min_count=STATE_MIN_COUNT)
    mean_sq = ctx.roll_mean(col * col, STATE_WINDOW, min_count=STATE_MIN_COUNT)
    var = mean_sq - mean * mean
    sd = np.sqrt(np.where(var > 0, var, np.nan))
    with np.errstate(all="ignore"):
        z = (col - mean) / sd
    return np.where(np.isfinite(z[:, 0]), z[:, 0], np.nan)


def _market_mean(ctx, mat):
    """池内有效值等权均值（= pandas nanmean 语义）。"""
    valid = np.isfinite(mat)
    cnt = valid.sum(axis=1)
    tot = np.where(valid, np.asarray(mat, dtype=np.float64), 0.0).sum(axis=1)
    with np.errstate(all="ignore"):
        m = tot / np.maximum(cnt, 1)
    return np.where(cnt > 0, m, np.nan)


# ────────────────────────────────────────────────────────────────────
# 两个变体
# ────────────────────────────────────────────────────────────────────
def _signal_a(ctx):
    """A：市场午后状态 × 放量时点，门控 = 同日个股形状一致度（非秩置换）。"""
    leg = _leg(ctx)
    state = _z252(ctx, _market_mean(ctx, _pm_ret(ctx)))
    vphi = leg - _field(ctx, "hi_pos")          # 放量在价格高点之前(<0)/之后(>0)
    apm = _am_pm_diff(ctx)                      # 收益路径前重(>0)/后重(<0)
    ok = (np.isfinite(leg) & np.isfinite(state)[:, None]
          & np.isfinite(vphi) & np.isfinite(apm))
    gate = 0.5 * (_cs_z(vphi) + _cs_z(apm))
    gate = np.where(np.isfinite(gate), gate, 0.0)
    with np.errstate(all="ignore"):
        value = np.sign(state)[:, None] * (leg - LEG_CENTER) * (1.0 + GATE_LAMBDA * gate)
    return np.where(ok, value, np.nan)


def _signal_b(ctx):
    """B：状态下放到逐股（自身午后收益 − 市场均值）的符号，截面天然异质。"""
    leg = _leg(ctx)
    pm = _pm_ret(ctx)
    mkt = _market_mean(ctx, pm)
    ok = np.isfinite(leg) & np.isfinite(pm) & np.isfinite(mkt)[:, None]
    with np.errstate(all="ignore"):
        rel = pm - mkt[:, None]
    return np.where(ok, np.sign(rel) * (leg - LEG_CENTER), np.nan)


_SIGNALS = {
    "mkt_pm_state_x_shape_gate": _signal_a,
    "own_pm_rel_state": _signal_b,
}


def build(parameters):
    """builder：参数只允许预登记的 signal 名（family allowed_variants 逐项冻结）。"""
    params = dict(parameters)
    signal = params.get("signal")
    if signal not in SIGNALS:
        raise ValueError(f"Family permits only preregistered signals: {sorted(SIGNALS)}")
    if set(params) != {"signal"}:
        raise ValueError(f"Unregistered parameters: {sorted(set(params) - {'signal'})}")
    fn = _SIGNALS[str(signal)]

    def compute(ctx):
        value = fn(ctx)
        if value is None:
            value = ctx.panel.empty(dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    compute.__name__ = "r11u2_" + str(signal)
    return compute
