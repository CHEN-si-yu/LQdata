"""Round 11 · 单元 U1 · 跨族交互 v2：**P2 筛后的**基本面状态 × 交易行为腿。

预登记（先于任何交互统计量）：
  * idea `r11u1b_p2screened_v1`，`registered_at = 2026-10-05T09:30:06Z`
    （`artifacts/research/catalog/ideas/r11u1b_p2screened_v1.json`）。
  * 被淘汰的第一版：idea `r11u1_crossfamily_v1`（09:22:13Z）的 3 个机制、6 条腿**全部**被
    强制 P2 判据（|t(IC_1d)| ≤ 2.0）淘汰，实测 |t| = 11.84 / 11.84 / 12.10 / 6.90 / 11.06 / 10.82。

════════════════════════════════════════════════════════════════════════
一、为什么构造是「股票级状态 × 股票级腿」而不是 R9-F 的「市场标量 × 腿」
════════════════════════════════════════════════════════════════════════
引擎逐日 winsor→z-score 对**正尺度**不变。若状态是 (T,) 标量，
    z_t(state_t · leg) = sign(state_t) · z_t(leg)      逐日**精确**成立
⇒ 候选的当日截面秩恒等于腿的秩（只差符号），「控制两腿主效应后的增量」在同一日内
**共线不可识别**（R9-F 的结构性事实）。
本单元状态是**股票级**的，`X(t,i) = z_cs(S)(t,i)·z_cs(L)(t,i)` 在每个交易日内都不是腿的
单调变换，`corr_t(X, L)` 是 (−1,1) 内的连续量 ⇒ 交互项在两腿主效应进入控制后**可识别**。
实测：`frac_days_abs_rho(X, L)_gt_0.99 = 0.0000`（v1 的 6 个配对，2123 天）。
在产同构先例：`cp_momentum_lowvol_20 = z(momentum_20) × z(−idio_vol_60)`。

════════════════════════════════════════════════════════════════════════
二、预登记机制（三个，方向全部先于结果冻结为 True）
════════════════════════════════════════════════════════════════════════
A `valuation_x_flow_surprise`  状态 `ep_ttm` × 腿 `mf_net_vol_surprise_20`
   意外的大单净买入是需求冲击。Miller(1977) 分歧 + 卖空约束下，**贵**股票（低盈利收益率）
   里的正向需求冲击更可能是情绪/拥挤交易、随后回吐；**便宜**股票里更可能是价值知情的累积、
   倾向延续 ⇒ E[ret] 随 (ep_ttm × 流入意外度) 递增 ⇒ hib = True。
B `quality_x_net_inflow`  状态 `roa_ttm` × 腿 `mf_net_inflow_5d`
   持续 5 日净流入＝机构累积。机构信息优势只在基本面可分析的公司里转化为收益；
   低 ROA 公司里的净流入更可能是散户/注意力驱动、随后回吐 ⇒ E[ret] 随 (roa_ttm × 净流入)
   递增 ⇒ hib = True。★ 两腿无条件 IC 都不显著（t=+0.95 / −0.62），是 R9-F 通过参照的形态。
C `growth_x_intraday_overnight_vol`  状态 `revenue_cagr_3y` × 腿 `intraday_vol_ratio_5d`
   日内/隔夜波动比刻画价格发现在哪里发生（Lou-Polk-Skouras 2019 的隔夜—日内拔河）。
   高成长公司估值不确定性最大，日内波动更多由信息驱动、倾向延续；成熟公司更多是噪声
   ⇒ E[ret] 随 (成长 × 日内隔夜波动比) 递增 ⇒ hib = True。

★ 方向纪律：三者全部 `higher_is_better = True`。登记后不反号、不改方向、不换版本找方向。

════════════════════════════════════════════════════════════════════════
三、实现约定
════════════════════════════════════════════════════════════════════════
* 状态与腿**都**经 `ctx.load_factor(name)` 读在产因子的落盘 `value`（精确落格、缺失 NaN，
  不 asof 前向填充）。`deps` 因此写的是**因子名**，引擎据此做水位失效传播并把父因子历史
  物化进 run 命名空间（`research/runner.py::dependency_output_starts`）。
* 标准化一律 `ctx.cs_zscore(x, mask=ctx.universe)`：只用当日截面，无全样本统计量（前视）。
  因子内不做 rank（契约 §2.3）。
* 缺失传播：任一腿 NaN → 该格 NaN；不零填充、不插值。
* PIT：读到的都是 T 日收盘后已可得的在产因子值；T 日取值用于 T+1 开盘起的标签。
* 覆盖 = 两腿同时有值的交集；逐候选实测报 coverage 与 `constant_cross_section`。
"""
from __future__ import annotations

import numpy as np

SIGNALS = ("valuation_x_flow_surprise", "quality_x_net_inflow", "growth_x_intraday_overnight_vol")
PARAMETERS = {"signal": "valuation_x_flow_surprise"}

# (状态因子名, 腿因子名) —— 二者都必须是在产因子（REGISTRY 内），由 spec.deps 声明。
PAIRS = {
    "valuation_x_flow_surprise": ("ep_ttm", "mf_net_vol_surprise_20"),
    "quality_x_net_inflow": ("roa_ttm", "mf_net_inflow_5d"),
    "growth_x_intraday_overnight_vol": ("revenue_cagr_3y", "intraday_vol_ratio_5d"),
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

    compute.__name__ = "r11u1_" + str(signal)
    return compute
