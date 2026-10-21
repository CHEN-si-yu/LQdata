"""R9-C 股东户数分散度：异常分散位移（时间序列标准化的一期户数变化）。

数据：`stock_holder_number`（stock_code, ann_date, end_date, holder_num）。

════════════════════════════════════════════════════════════════════════
一、立项与重蹈审计（在写第一行因子代码之前完成，结论写进交付报告）
════════════════════════════════════════════════════════════════════════
本单元被指派的三个机制中，C2/C3 经事前去重审计判为**重蹈**，**未登记、未计算**：

· C2「股权分散度水平」log(holder_num / float_share) —— 重蹈 R2-I3 机制 1a。
  R2 的 `r2own_v2_per_capita_value_level` 已经测过同一个 (holder_num, float_share)
  水平轴（户均持股市值 = circ_mv / holder_num）。实测恒等式
  **circ_mv ≡ close × float_share**（见 tools/audit_proof.py 的 identity 段）
  ⇒ log(circ_mv/holder_num) = log(close) − log(holder_num/float_share)
  ⇒ **C2 = log(close) − (R2 M1a)**，即同一轴的「符号反转 + 价格项」再参数化。
  R2 M1a 的实测结论：declared RankIC h1 = −0.00638（3/6 年为正）、h5 = −0.01220
  （3/6），五个期限的 HAC95 区间全部跨 0 ⇒ 无稳定信号。故放弃。

· C3「公告时滞不对称」—— 重蹈 r7c。
  `r7c_self_disclosure_delay_v1` 已在 `stock_income` 上测过**同一机制**
  （L_p − median(前 3 个同季 L)）；Round7 记录原文：「C 披露延迟均 G4 失败」。
  换一张表不构成新机制。故放弃。

存活机制（唯一）：**C1 异常分散位移**。

════════════════════════════════════════════════════════════════════════
二、机制陈述与差异论证（G5 / BRIEF §8.6 结构性同构自检）
════════════════════════════════════════════════════════════════════════
定义 d_j = log(n_j) − log(n_{j−1})，是**该股自身**按报告期排序的第 j 个户数对数变化。
C1 = z_j = (d_0 − mean(d_{−1..−K})) / std(d_{−1..−K}, ddof=1)，
K = 8（主口径）/ 4（事前登记的窗口稳健性变体），最少需要 4 / 3 个历史变化。

经济含义：**这次户数变动在该股自身的历史波动尺度下是否异常**（异常分散位移），
而不是「变动本身」。既有构造全部不是这个：
  · `holder_number_chg`（生产因子）= 单期变化率本身，**无历史基准、无波动率归一**；
  · `r2own_v2_trend_persistence_v1`（生产，t=+4.74/+4.53）= 最近 4 期下降的
    方向 × 幅度均值，**无波动率归一**，是趋势强度而非标准化意外；
  · R2 `dispersion_residual`（已被 R2 判重蹈）= 变化率对 log 市值/换手的
    **截面** OLS 残差，实测与 `holder_number_chg` |rho| = 0.9534。
C1 是**时间序列**标准化（除以该股自身历史标准差），横截面含义是「自身历史上
户数很稳、本期却大幅变动」——既不是 M4 的截面中性化，也不是 M2 的方向持续性。
预期两两 |rho| 在报告 §6 实测呈现，不做事前断言。

方向（Lead 事前固定，不因任何 IC/收益结果改变）：
  **higher_is_better = false**。z 越大 = 户数异常增加 = 分散度异常上升 = 预期收益越低。
  与 `holder_number_chg` 的 hb=false 及 R2 M2 的「户数下降 = 好」同一经济先验。
  本模块只返回值，不做任何反号。

════════════════════════════════════════════════════════════════════════
三、PIT 合同（唯一可用时点 = ann_date）
════════════════════════════════════════════════════════════════════════
· ann_date（公告日）是唯一可用时点。end_date（报告期截止日）只用于确定
  「哪一期更新」，**从不**作为可见日期、不参与任何 as-of 键。
· 实测：428,878 行中有 **27,744 行 (stock_code, end_date) 重复**（同一报告期被
  不同 ann_date 重述）。状态机按 ann_date 推进，T 日只看到 ann_date <= T 的记录，
  因此新公告**不会改写历史生效日**（这正是生产因子 holder_number_chg 2026-09-17
  修过的那个 PIT 违规）。
· 同一个 ann_date 上的多条记录（同日可披露 1~19 个报告期）在该日**一次性全部生效**。
· 无记录 != 未发生：披露后的状态一直有效，用 as-of 前向沿用；从未披露 -> NaN。
· 上游审计按 ann_date 截断（`fea/upstream.py::_audit_columns` 的 announcement 分支），
  本表日期列在白名单内，**不需要**任何框架绕行。

缺失 / 无效（事前冻结）：
· holder_num <= 0 视为无效（全表实测 20,434 行为 0，且 (0,100) 内有 1,901 行，
  故阈值取 > 0 即可，不做额外的下界猜测）。
· 相邻报告期 end_date 间隔必须落在 (0, 200] **真实日历天**（PERIOD_GAP_MAX），
  否则该配对不产生变化 —— 与 R2-I3 及生产因子 holder_number_chg 的间隔保护同口径
  （1~12 月都有披露，间隔过长等于拿两个不同年份的观察值相比）。
· 历史样本不足 Z_MIN_PRIOR，或历史标准差 <= 1e-12（无离散度）-> z 未定义 -> NaN，
  且**不发射事件**（沿用最近一次可计算的 z；从未可计算 -> NaN）。
  理由：无离散度时 z 是 0/0，不是「0」；把它写成 0 会伪造一个观测。
· 完整输入下 z 恰为 0 是观测到的 0，保留。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── 事前冻结的常量（不因任何结果调整）──────────────────────────────────
PERIOD_GAP_MAX = 200     # 相邻报告期 end_date 的最大间隔（真实日历天）
STD_FLOOR = 1e-12        # 历史标准差下限；<= 此值视为无离散度 -> z 未定义
MIN_PERIODS = 2          # 至少要有 2 个报告期才谈得上「变化」

# 允许的 (signal, z_lookback, z_min_prior) 组合。build() 拒绝任何其它取值，
# 因此定义不可变；扩口径必须换新模块与新登记（工程约定）。
FROZEN = {
    ("abnormal_dispersion_shift", 8, 4): "main",
    ("abnormal_dispersion_shift_k4", 4, 3): "declared_lookback_variant",
}

_COLUMNS = ("stock_code", "ann_date", "end_date", "holder_num")


def _log_ratio(num_new: np.ndarray, num_old: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        out = np.log(num_new / num_old)
    return np.where(np.isfinite(out), out, np.nan)


def _zscore_from_state(known, lookback, min_prior, day_of):
    """给定「截至当前已生效」的报告期状态，算异常分散位移 z。

    known: dict end_date(int YYYYMMDD) -> holder_num(float > 0)
    返回 float（z）或 None（无法定义）。
    """
    if len(known) < MIN_PERIODS:
        return None
    periods = sorted(known, reverse=True)          # 最新报告期在前
    pairs = []
    for k in range(len(periods) - 1):
        e_new, e_old = periods[k], periods[k + 1]
        gap = day_of[e_new] - day_of[e_old]
        if 0 < gap <= PERIOD_GAP_MAX:
            pairs.append(np.log(known[e_new]) - np.log(known[e_old]))
    if not pairs:
        return None
    hist = np.asarray(pairs[1:1 + lookback], dtype=np.float64)
    if hist.size < min_prior:
        return None
    sd = float(hist.std(ddof=1))
    if not np.isfinite(sd) or sd <= STD_FLOOR:
        return None
    z = (pairs[0] - float(hist.mean())) / sd
    return float(z) if np.isfinite(z) else None


def _events(ctx, lookback, min_prior):
    """按 ann_date 逐日推进状态机，返回 (codes, days, z) 三个平行数组。

    只在 z 可计算时发射事件；as-of 前向沿用最近一次可计算值。
    """
    frame = ctx.dataset("stock_holder_number", columns=list(_COLUMNS), years=None)
    if frame is None or frame.empty:
        return (np.array([], dtype=object), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float64))

    code = frame["stock_code"].astype(str).to_numpy()
    ann = np.asarray(ctx.date_col(frame["ann_date"]), dtype=np.int64)
    end = np.asarray(ctx.date_col(frame["end_date"]), dtype=np.int64)
    num = pd.to_numeric(frame["holder_num"], errors="coerce").to_numpy(dtype=np.float64)

    keep = (num > 0) & (ann >= 19000101) & (end >= 19000101)
    if not keep.any():
        return (np.array([], dtype=object), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float64))
    code, ann, end, num = code[keep], ann[keep], end[keep], num[keep]

    # 日历天序号：一次性算好，避免在 10 万级的状态推进循环里反复调 pandas
    uniq_end = np.unique(end)
    parsed = pd.to_datetime(pd.Series(uniq_end).astype(str), format="%Y%m%d",
                            errors="coerce")
    day_ord = (parsed - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    day_of = {int(e): int(d) for e, d in zip(uniq_end, day_ord)
              if np.isfinite(d)}

    # 排序：先 code，再 ann_date，再 end_date（同日多期一次性生效）
    _, code_id = np.unique(code, return_inverse=True)
    order = np.lexsort((end, ann, code_id))
    code, ann, end, num = code[order], ann[order], end[order], num[order]

    boundary = np.empty(code.size, dtype=bool)
    boundary[0] = True
    if code.size > 1:
        boundary[1:] = (code[1:] != code[:-1]) | (ann[1:] != ann[:-1])
    starts = np.flatnonzero(boundary)
    stops = np.append(starts[1:], code.size)

    out_code, out_day, out_val = [], [], []
    known = {}
    current = None
    for start, stop in zip(starts, stops):
        group_code = code[start]
        if group_code != current:
            current, known = group_code, {}
        for j in range(start, stop):
            known[int(end[j])] = float(num[j])     # 同日多条全部生效
        z = _zscore_from_state(known, lookback, min_prior, day_of)
        if z is not None:
            out_code.append(group_code)
            out_day.append(int(ann[start]))
            out_val.append(z)

    return (np.asarray(out_code, dtype=object),
            np.asarray(out_day, dtype=np.int64),
            np.asarray(out_val, dtype=np.float64))


def build(parameters):
    """R9-C 唯一存活机制的两个事前登记口径；不搜索窗口、不按结果翻转方向。"""
    key = (parameters.get("signal"), parameters.get("z_lookback"),
           parameters.get("z_min_prior"))
    if set(parameters) != {"signal", "z_lookback", "z_min_prior"} or key not in FROZEN:
        raise ValueError(
            "r9c_holder_dispersion_v1 只允许两个事前登记口径："
            "abnormal_dispersion_shift(K=8,min=4) / abnormal_dispersion_shift_k4(K=4,min=3)")
    lookback, min_prior = int(parameters["z_lookback"]), int(parameters["z_min_prior"])

    def compute(ctx):
        codes, days, values = _events(ctx, lookback, min_prior)
        if codes.size == 0:
            return ctx.panel.empty(dtype=np.float64)
        grid = np.asarray(ctx.asof_daily(codes, days, values), dtype=np.float64)
        universe = np.asarray(ctx.universe, dtype=bool)
        value = np.where(universe, grid, np.nan)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
