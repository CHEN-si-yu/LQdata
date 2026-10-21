"""V1 · r4p_goodwill_v1 —— 商誉与减值风险（Round 4 批 3，机制/方向事前冻结）。

四个预登记信号（2 机制 + 2 winsor 敏感性变体；方向全部 false）：
    goodwill_to_equity        false  商誉 / 归母净资产（时点）
    goodwill_growth_yoy       false  商誉同比增速 = g(最新报告期) / g(去年同季) − 1
    goodwill_to_equity_win    false  变体：min(g/eq, 1.0) 固定上帽
    goodwill_growth_yoy_win   false  变体：clip(增速, −1.0, 2.0) 固定双帽
帽值（1.0 / [−1.0, 2.0]）在**未查看任何数据分布之前**冻结（先于首个 run），
变体即「winsor 敏感性」：检验结论是否由极端样本驱动（比率>100%净资产；
增速>+200% 的并购冲刺）。

★ 机制 2 `goodwill_to_assets`（商誉/总资产）**未实现、未登记**：
  防重蹈检查（任务书 B2 §0，本批强制）在 `factors/fundamental3.py` 头部发现
  **同名已删先行者**——2026-09-15/17 两轮「去糟粕」删除的财务因子清单中逐字含
  `goodwill_to_assets`，死因 =「慢变量水平在一年期检验里没有统计功效」（ac1≈1、
  换手≈0）。对象（全池含金融股）/口径（商誉/总资产）/机制（减值风险）与先行者
  逐项相同、说不出任何差异 → 按 B2 §0.1 停止该机制并如实记录（不占预算、不算
  失败）。完整证据链（含同清单 19 个名字在现役库全部无存活的可靠性验证）见
  `artifacts/research/round4_20261004/delivery/r4p_goodwill_v1/report.md`「防重蹈
  检查」节。机制 1 与 2 的关系按任务书 §V1 原文「分母改口径（净资产 vs 总资产
  互换算作不同机制）」处理，且 M1 无同名先行者，故 M1 照跑、在报告中披露
  先行者死因为检验功效（非机制证伪）所提示的家族级风险。

口径冻结（报告披露）：
  · 商誉 = `ctx.point("goodwill")`，时点值、经财务层 PIT（pit=max(ann_date,
    f_ann_date, end_date) ≤ 当日的最新可考版本，含历次修订；公告不可考的行在
    财务层丢弃）。商誉**不在** POSITIVE_ONLY——`fea/deriv.py` 的注释逐字点名
    「商誉占比」（0 是真实值：无并购的公司商誉=0 合法），故分子 0 保留。
  · 分母 = `ctx.point("total_hldr_eqy_exc_min_int")`（归母净资产）。该字段**在**
    POSITIVE_ONLY（0 与负数 → NaN）：净资产 ≤0 的公司（多为商誉风险最高的困境
    公司）在比率口径中整体缺失——这是财务层除外口径，报告如实披露。
  · 分母保护统一 `min_abs_den=1e6` 元（safe_div）：归母净资产 ≤1e6 元的比率记
    NaN；商誉同比的分母（去年同季商誉）也须 >1e6 元——**基期商誉归零/近零的
    新增并购样本记 NaN 而非 +∞**，增速口径只覆盖「有可测基期」的公司。
  · 同比 = `ctx.point("goodwill", lag=4)`：同一公告 pit 上「最新报告期 − 4 期」
    的 as-of 值 = 严格同季对比（不跨报告期边界；见 deriv.Derivative.lag 注释）。
    增速数学下界 = −1（本期商誉归 0）；无上界（基期近 1e6 元时可达 1e4 量级），
    固定上帽变体即为此。
  · 商誉是**存量时点科目**，不做 TTM 口径（ttm 对存量科目无意义）。
  · 输出统一 float64，非有限值 → NaN；不零填充、不向旧报告回退。
"""
from __future__ import annotations

import numpy as np

SIGNALS = (
    "goodwill_to_equity", "goodwill_growth_yoy",
    "goodwill_to_equity_win", "goodwill_growth_yoy_win",
)

MIN_ABS_DEN = 1e6   # 分母地板：归母净资产 / 去年同季商誉都必须大于 1e6 元
EQ_CAP = 1.0        # 变体1 上帽：商誉/归母净资产（事前固定）
GR_LO, GR_HI = -1.0, 2.0   # 变体2 双帽：商誉同比增速（下界即数学下界）

__all__ = ["SIGNALS", "build"]


def build(parameters):
    """四个预登记信号之一；没有可搜索的窗口、帽值或方向。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError(f"Batch permits only the preregistered signal identifiers: {SIGNALS}")
    signal = parameters["signal"]

    def compute(ctx):
        if signal in ("goodwill_to_equity", "goodwill_to_equity_win"):
            num = np.asarray(ctx.point("goodwill"), dtype=np.float64)
            den = np.asarray(ctx.point("total_hldr_eqy_exc_min_int"), dtype=np.float64)
            val = ctx.safe_div(num, den, min_abs_den=MIN_ABS_DEN)
            if signal == "goodwill_to_equity_win":
                val = np.minimum(val, EQ_CAP)
        else:  # goodwill_growth_yoy / goodwill_growth_yoy_win
            cur = np.asarray(ctx.point("goodwill"), dtype=np.float64)
            base = np.asarray(ctx.point("goodwill", lag=4), dtype=np.float64)
            val = ctx.safe_div(cur, base, min_abs_den=MIN_ABS_DEN) - 1.0
            if signal == "goodwill_growth_yoy_win":
                val = np.clip(val, GR_LO, GR_HI)
        # NaN 语义 = 不知道（不填 0）；非有限值统一收口。
        return np.where(np.isfinite(val), val, np.nan).astype(np.float64, copy=False)

    return compute
