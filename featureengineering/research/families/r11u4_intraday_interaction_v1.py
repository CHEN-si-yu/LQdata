"""Round 11 · 单元 U4 · 字段级条件交互（第二存活配对）：**日内形态 × 日内成交量集中度**。

════════════════════════════════════════════════════════════════════════
一、来源：强制预筛（954 配对，19 族）的两个双筛存活者之一
════════════════════════════════════════════════════════════════════════
**P2（腿自身无条件 IC 不显著）**：`intraday_vol_ratio_5d` 的 h1 无条件 IC t = **+1.69**
  （< 2；对照 F2 参照腿 t = −1.60）。
**P3（状态显著翻转腿的条件 IC）**：按 state 的**逐日截面中位数**劈分，
  IC+(state 高) = **+0.00517**（t = **+3.37**）、IC−(state 低) = **−0.00480**（t = **−2.67**）、
  差值 HAC t = **+5.94**，两状态**符号相反且各自显著** ⇒ PASS
  （**高于**已准入 `r9f_breadth_intraday_slope_x_beta_v1` 的斜率 t = −4.38）。
**P1 / 预检 B / unique-per-day / coverage**：见 u4_report.md §4（与第一个候选同口径实测）。

════════════════════════════════════════════════════════════════════════
二、与 U1 的分工、与附录 K 的关系
════════════════════════════════════════════════════════════════════════
两腿**同属日内族**（`pattern` 组的 `intraday_vol_ratio_5d` × `research_intraday` 组的
`r4b_intraday_volume_hhi_20`）⇒ 符合 U4「同族两字段」，不是 U1 的跨族设计。
状态是**股票级**（逐股变化），故候选在每个交易日内都不是腿的单调变换：
  `z_t(state_t · leg) = sign(state_t)·z_t(leg)` 只对**广播态**成立 ⇒ 附录 K 的陷阱不适用。
实测（报告中给出）`frac_days_abs_rho_gt_0.99` 对自身腿与对自身状态均为 0。

════════════════════════════════════════════════════════════════════════
三、方向（事前固定）
════════════════════════════════════════════════════════════════════════
机制：`intraday_vol_ratio_5d`（日内/隔夜波动比）刻画价格发现在**哪里**发生；
`r4b_intraday_volume_hhi_20`（日内成交量 HHI）刻画成交量在**时间上有多集中**。
两者都高＝"日内既承担价格发现、成交又高度集中在少数时段"，是信息驱动但**流动性脆弱**的状态，
随后反转；成交量分散时日内波动更可能是噪声，无方向性 ⇒ 乘积越大，预期收益越低。
⇒ **`higher_is_better = false`**。本模块只返回值，不做反号。
"""
from __future__ import annotations

import numpy as np

SIGNALS = ("intraday_shape_x_volume_concentration",)
PARAMETERS = {"signal": "intraday_shape_x_volume_concentration"}

# (状态因子名, 腿因子名) —— 两者都是在产因子，同属日内族。
PAIRS = {
    "intraday_shape_x_volume_concentration": (
        "r4b_intraday_volume_hhi_20_2462d9ca21d9adf8",
        "intraday_vol_ratio_5d",
    ),
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
