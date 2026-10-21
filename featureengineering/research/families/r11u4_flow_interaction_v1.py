"""Round 11 · 单元 U4 · 字段级条件交互：**同族两字段**的条件关系（stock-varying state）。

════════════════════════════════════════════════════════════════════════
一、与 U1 的分工（Lead 明确要求，不得重复）
════════════════════════════════════════════════════════════════════════
U1 = **跨族**（不同族的状态 × 腿）；本单元 = **同一族内两个字段**的条件关系。
本候选两腿**同属 `fundflow` 族**：
  state = `mf_net_inflow_5d`（资金流强度/方向）
  leg   = `mf_net_vol_surprise_20`（资金流量的意外度 = 持续性/波动维度）
⇒ 构念是"资金流的**强度**是否调节资金流**意外度**的收益关系"，不是跨族组合。
U1 的 `B quality_x_net_inflow` 用的是 `roa_ttm × mf_net_inflow_5d`（质量族 × 资金族），
与本候选的两腿组合不同；本报告在 §6 给出与 U1 已登记候选的实测相关。

════════════════════════════════════════════════════════════════════════
二、为什么用 stock-varying state（附录 K 的直接应用）
════════════════════════════════════════════════════════════════════════
引擎逐日 winsor→z-score 对**正尺度**不变。若状态是 (T,) 广播标量，则
    z_t(state_t · leg) = sign(state_t) · z_t(leg)    逐日**精确**成立
⇒ 候选当日截面秩恒等于腿的秩（只差符号），`frac_days_abs_rho_gt_0.99` 对自身腿 = **1.0000**，
且 `IC_cand(t) = sign(state_t)·IC_leg(t)` ⇒ **G4 退化为 P3 的斜率项、几乎无信息量**。
本候选的状态是**股票级**的：`X(t,i) = z_cs(state)(t,i) · z_cs(leg)(t,i)` 在每个交易日内
都不是腿的单调变换。**实测**（本执行者预筛，2123 天）：
  `corr_t(X, leg)` 的 daily_abs_mean = **0.1312**、`frac |ρ_t| > 0.99` = **0.0000**
  `corr_t(X, state)` 的 daily_abs_mean = **0.0671**、`frac |ρ_t| > 0.99` = **0.0000**
⇒ 附录 K 的广播态陷阱**不适用**，本候选不是自身腿的有符号置换。

════════════════════════════════════════════════════════════════════════
三、预筛结果（登记之前的强制判据，全部实测；详见 u4_report.md §1–§3）
════════════════════════════════════════════════════════════════════════
**P2（腿自身无条件 IC 不显著）**：`mf_net_vol_surprise_20` 的 h1 无条件 IC t = **−0.02**
  （全库 125 因子的 |t| 中位数 5.10、p25 2.65 ⇒ 弱腿是少数，本腿是最弱的一个）。
**P3（状态显著翻转腿的条件 IC，最强判据）**：按 state 的**逐日截面中位数**劈分，
  IC+(state 高) = **−0.00521**（t = **−2.08**）、IC−(state 低) = **+0.00646**（t = **+2.42**）、
  **差值序列 HAC t = −7.42**，两状态**符号相反**且各自显著 ⇒ **P3 PASS**
  （对照基准：已准入 `r9f_breadth_intraday_slope_x_beta_v1` 斜率 t = −4.38）。
**P1（附录 I 预检 A：是否为在库因子的每日秩置换）**：与全部 **125** 个在库因子逐日比对，
  **FLAGGED = 0**；`frac |ρ_t| > 0.99` 与 `exact_perm` 占比在**所有 125 个因子上均为 0.000000**；
  最大 pooled |ρ| = **0.1963**（`volume_ratio`），且**没有任何因子 |ρ| > 0.5**。
**预检 B（值分布）**：p0/p1/p25/p50/p75/p99/p99.9/p100 = −12.97 / −2.19 / −0.150 / +0.052 /
  +0.480 / +4.95 / +8.41 / +18.88；`|max|/|p99| = 3.82` ⇒ **未触发**尾部计数要求。
**unique/day**：min **1,969** / median 2,106；**0 天**唯一值 ≤20 ⇒ 连续量、非阶梯（D2 不适用）。

════════════════════════════════════════════════════════════════════════
四、方向（事前固定，不因任何 IC/收益结果改变）
════════════════════════════════════════════════════════════════════════
机制：`mf_net_vol_surprise_20` 是成交量的**意外度**（需求冲击的规模维度），
`mf_net_inflow_5d` 是资金净流向的**强度/方向**。两者同号且都高时＝"大额且方向明确的意外流入"，
是拥挤/情绪驱动的需求冲击，随后回吐；两者异号时该"意外"缺少方向确认，信息含量低。
⇒ **`higher_is_better = false`**：乘积越大（同号且幅度大），预期收益越低。
本模块只返回值，**不做任何反号**。

════════════════════════════════════════════════════════════════════════
五、实现约定（沿用 U1 的既有约定，未新增自由度）
════════════════════════════════════════════════════════════════════════
* 状态与腿都经 `ctx.load_factor(name)` 读在产因子的落盘 `value`（精确落格、缺失 NaN，
  不 asof 前向填充）。`deps` 写**因子名**，引擎据此做水位失效传播。
* 标准化一律 `ctx.cs_zscore(x, mask=ctx.universe)`：只用当日截面，无全样本统计量（前视）。
* 缺失传播：任一腿 NaN → 该格 NaN；不零填充、不插值（**本候选无缺失约定自由度**：
  两腿都是稠密在产因子，实测缺失率 0.61% / 0.55%，交集覆盖率 0.9935 ⇒ 附录 J 规则 4
  的"两口径"问题**不适用**，因为没有可选的填充约定）。
* PIT：读到的都是 T 日收盘后已可得的在产因子值；T 日取值用于 T+1 开盘起的标签。
"""
from __future__ import annotations

import numpy as np

SIGNALS = ("flow_intensity_x_flow_surprise",)
PARAMETERS = {"signal": "flow_intensity_x_flow_surprise"}

# (状态因子名, 腿因子名) —— 两者都是**在产因子**，由 spec.deps 声明；同属 fundflow 族。
PAIRS = {
    "flow_intensity_x_flow_surprise": ("mf_net_inflow_5d", "mf_net_vol_surprise_20"),
}


def build(parameters):
    """builder：参数只允许预登记的 signal 名；无可搜参数、无窗口/阈值自由度。"""
    params = dict(parameters)
    signal = params.get("signal")
    if signal not in SIGNALS:
        raise ValueError(f"Family permits only preregistered signals: {sorted(SIGNALS)}")
    if set(params) != {"signal"}:
        raise ValueError(f"Unregistered parameters: {sorted(set(params) - {'signal'})}")
    state_name, leg_name = PAIRS[str(signal)]

    def compute(ctx):
        state = np.asarray(ctx.load_factor(state_name), dtype=np.float64)
        leg = np.asarray(ctx.load_factor(leg_name), dtype=np.float64)
        if state.shape != leg.shape:
            raise ValueError(f"state/leg grid mismatch: {state.shape} vs {leg.shape}")
        zs = np.asarray(ctx.cs_zscore(state, mask=ctx.universe), dtype=np.float64)
        zl = np.asarray(ctx.cs_zscore(leg, mask=ctx.universe), dtype=np.float64)
        with np.errstate(all="ignore"):
            value = zs * zl
        value = np.where(np.isfinite(zs) & np.isfinite(zl), value, np.nan)
        value = value.astype(np.float64, copy=False)
        if value.shape != state.shape:
            raise ValueError(f"wrong shape {value.shape}, expected {state.shape}")
        return value

    compute.__name__ = "r11u4_" + str(signal)
    return compute
