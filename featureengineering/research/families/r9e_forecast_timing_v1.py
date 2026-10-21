"""Round 9 · 单元 E · 业绩预告「同类同伴相对披露时点」（family `r9e_forecast_timing_v1`）。

预登记（先于任何结果统计冻结；登记后不得因结果改方向或改口径）
══════════════════════════════════════════════════════════════════════
唯一机制 `peer_lateness` —— 方向 `higher_is_better=False`（越晚越差）。

机制陈述（为什么它与现有/已删因子不同构）
----------------------------------------
在**同一报告期（end_date）、同一预告类型（type）**的公告队列内做**截至该公告日的
扩张中位数**归一化，得到「相对同类同伴的披露时点」`rel = late − peer_median`，
其中 `late = ann_date − end_date`（日历天）。条件在 `type` 上，就把「消息本身的好
坏档位」固定住了，剩下的只是**披露时机**：管理层对坏消息拖延、对好消息抢先
（Skinner 1994；Kothari/Shu/Wysocki 2009），拖延本身携带「后续还有坏消息要消化」
的信息，与消息幅度是两个维度。方向取负：相对同类更晚披露 → 未来 1/5 日收益偏低。

§8.8 强制重蹈检查（`doc/因子清理记录_20261003.md`，374 个已删股票因子）
--------------------------------------------------------------------
* `disclosure_detail` 族已覆盖并已删：
  - `forecast_profit_midpoint_change` = ((np_min+np_max)/2 − last_parent_net)/|last_parent_net|
    → **预告幅度/惊喜轴**（r4a 的 `forecast_implied_surprise` 正因此被 Lead 撤销）；
  - `forecast_profit_range_uncertainty` = (np_max−np_min)/|last_parent_net|
    → **预告区间宽度/不确定性轴**（本单元任务书 E1 的原始想法，已判为重蹈，未登记）。
* `event` 族已覆盖并已删：`forecast_type_score`（预增/扭亏=+2 … 预减/首亏/续亏=−2，
  按 ann_date 前向填充）→ **类型分类轴**（本单元任务书 E1 的第二个原始想法，已判重蹈）。
* r4a（`research_earnings`，在库）已覆盖：SUE / 惊喜加速度 / PEAD 漂移 /
  落地偏差（`r4a_forecast_landing_gap_v3`，在库）。
* r4t（`research_preann`）已覆盖：以 `stock_income` 公告时点定义窗口的**公告前量价抢跑**。
* ⇒ **披露时点（ann_date − end_date，且条件在 type 上做队列归一化）不在以上任何一项中**：
  它既不是幅度、也不是区间宽度、也不是类型打分，也不是公告前量价。已删因子里没有任何
  一个用到 `ann_date − end_date` 或 `type` 的分组中位数。

PIT（可得性）合同
-----------------
* `stock_forecast` 的日期列 `ann_date`（`conf/frequency.yaml` 声明 delay_days=0），
  行自 `ann_date` 起可见；同一交易日的公告在 T 日收盘后（含当晚）全部可得，
  因子在 T 日取值、用于 T+1 开盘起的标签（`open_hfq(t+1+h)/open_hfq(t+1)-1`），无前视。
* 队列中位数只用**公告日 ≤ 本公告日**的同队列成员（`expanding(min_periods=5)`），
  因此 T 日的值只用到 ≤T 的信息，**不含任何未来公告**。
* 截断自证：`artifacts/research/round9_20261005/e_forecast_margin/probe_truncation.txt`
  （`fea.upstream.Upstream._audit_frame` 直接实测：cutoff 之后的行全部消失、
  cutoff 及以前的行全部保留）。

实测数据事实（2026-10-05，本单元 probe，全池口径）
--------------------------------------------------
* `stock_forecast` 全表 **110,918 行**（17 个年分区），池内 **52,245 行**，
  覆盖池内 2,099/2,115 只股票（与 r4a 模块 docstring 记录的 110,918 / 52,245 一致）。
  ★ 任务书/手册里的「约 5,195 行」实为 **year=2026 单分区**的行数，不是全表。
* `type` 取值实测：预增 30,252 / 略增 18,259 / 预减 12,606 / 首亏 12,287 /
  续亏 12,217 / 扭亏 9,935 / 略减 8,499 / 续盈 5,109 / 不确定 1,283 /
  增亏 217 / 减亏 199 / 其他 55。本机制排除 `不确定`、`其他`（无方向含义）。
* `late = ann_date − end_date`（池内）分位：p1 −74 / p25 −61 / p50 +10 / p75 +15 /
  p95 +31 / p99 +34 日历天；负值是「报告期还没结束就预告」的合法早披露，保留。
* 同 (code, end_date) 有 >1 条记录的组 3,760 个（修订/多报告期同日）；
  `first_ann_date < ann_date`（修订过）占 9.77%。
* 队列规模 (end_date, type) 分位：p10 = 4、p25 = 18、p50 = 55、p75 = 116、p90 = 212。
* 真实值起点与可覆盖性：本机制按 `ann_date` as-of 前向填充（口径与 r4a 一致），
  实测池内「至少有一条可用预告」的交易日占比 **2,110/2,123 = 99.39%**，
  每日可用票数中位数 1,910（p1 也有 649）——**G2 的 ≥100 只有效票在 ≥60% 交易日
  这一条对本机制不构成瓶颈**（任务书预判「G2 很可能失败」不成立，见 report.md）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PARAMETERS = {"signal": "peer_lateness"}

MIN_PEERS = 5                 # 同 (报告期, 类型) 队列中先于或等于本公告的成员数下限
MIN_ANN = 19900101            # 哨兵过滤（与 r4a 同口径）
FORECAST_COLS = ["stock_code", "ann_date", "end_date", "type"]
EXCLUDED_TYPES = ("不确定", "其他")


def _int_dates(series: pd.Series) -> pd.Series:
    """把供应商标注的日期列（'YYYY-MM-DD' 字符串或整数）统一成 YYYYMMDD 整数。"""
    return pd.to_datetime(series.astype(str).str.strip(), errors="coerce")


def _peer_lateness(ctx):
    """同类同伴相对披露时点：`late − expanding_median(late | 同 (end_date, type), ann ≤ 本公告)`。"""
    frame = ctx.dataset("stock_forecast", columns=list(FORECAST_COLS))
    if frame is None or frame.empty:
        return None
    frame = frame[frame["stock_code"].isin(ctx.panel.codes)]
    if frame.empty:
        return None

    ann_dt = _int_dates(frame["ann_date"])
    end_dt = _int_dates(frame["end_date"])
    ann = (ann_dt.dt.year * 10000 + ann_dt.dt.month * 100 + ann_dt.dt.day)
    end = (end_dt.dt.year * 10000 + end_dt.dt.month * 100 + end_dt.dt.day)
    typ = frame["type"].astype(str).to_numpy()
    late = (ann_dt - end_dt).dt.days

    ann_i = ann.to_numpy(dtype=np.float64)
    end_i = end.to_numpy(dtype=np.float64)
    late_v = late.to_numpy(dtype=np.float64)
    keep = (np.isfinite(ann_i) & (ann_i >= MIN_ANN)
            & np.isfinite(end_i) & (end_i > 0)
            & np.isfinite(late_v)
            & ~np.isin(typ, np.asarray(EXCLUDED_TYPES)))
    if not keep.any():
        return None

    rows = pd.DataFrame({
        "code": frame["stock_code"].to_numpy()[keep],
        "ann": ann_i[keep].astype(np.int64),
        "late": late_v[keep],
        "cohort": np.char.add(np.char.add(end_i[keep].astype(np.int64).astype(str), "|"),
                              typ[keep].astype(str)),
    })
    # 队列内按公告日升序，扩张中位数（含本公告）——只用已经公开的同队列成员。
    rows = rows.sort_values(["cohort", "ann"], kind="mergesort")
    peer = rows.groupby("cohort", sort=False)["late"].transform(
        lambda s: s.expanding(MIN_PEERS).median())
    rows["rel"] = rows["late"] - peer
    rows = rows[np.isfinite(rows["rel"].to_numpy(dtype=np.float64))]
    if rows.empty:
        return None
    # 同 (股票, 公告日) 多条（同日多报告期/多类型）取中位数，与既有因子口径一致。
    agg = rows.groupby(["code", "ann"], as_index=False)["rel"].median()
    return ctx.asof_daily(agg["code"].to_numpy(),
                          agg["ann"].to_numpy(dtype=np.int64),
                          agg["rel"].to_numpy(dtype=np.float64))


_SIGNALS = {"peer_lateness": _peer_lateness}


def build(parameters):
    """builder：参数只允许预登记的 signal 名（family allowed_variants 逐项冻结）。"""
    params = dict(parameters)
    if params != PARAMETERS:
        raise ValueError(f"Only the single frozen definition is permitted: {PARAMETERS}")
    signal = str(params["signal"])
    fn = _SIGNALS[signal]

    def compute(ctx):
        value = fn(ctx)
        if value is None:
            value = ctx.panel.empty(dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    compute.__name__ = "r9e_" + signal
    return compute
