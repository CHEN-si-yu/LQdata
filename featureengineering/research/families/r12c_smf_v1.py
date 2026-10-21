"""R12 批 C · C2 聪明钱因子 SMF（family `r12c_smf_v1`；**alpha_v2（B）通道**）。

════════════════════════════════════════════════════════════════════════
一、机制与方向预注册（H2）
════════════════════════════════════════════════════════════════════════
单位成交量推动的价格变动越大（|涨跌幅|/√量 越大）越可能来自知情交易者。
逐 (股票, 日)：

    zdf_t  = |close_t / open_t − 1|                （棒内涨跌幅）
    smart_t = zdf_t / √vol_t                       （信息强度）
    按 smart **降序**累加 vol，直到累计达到当日可用棒总 vol 的 20%（从大到小取，
    累计**严格小于** 20% 的棒入选；至少入选强度最高的一根）——即“高信息量成交子集”
    VWAP_sel = Σ(vol × close)_sel / Σ vol_sel      （与原文一致：子集用棒 close 加权）
    VWAP_day = Σ amount / Σ vol                    （全日 VWAP）
    SMF(d)   = VWAP_sel / VWAP_day − 1
    factor   = 过去 N 个交易日的 SMF 均值（N = 5 / 20，两个期限结构变体）

SMF > 0 表示高信息强度的那批成交**价格中枢高于**全日中枢 ⇒ 知情资金偏买。

**方向先验（事前冻结，不得反号）**：SMF 高 → 未来 h=1/h=5 收益正向
（`higher_is_better=True`，declared = raw）。
**证伪点**：h=1 或 h=5 的 declared RankIC HAC95 下界 ≤ 0、或正向年份 < 6/9（政策 v2 G4）
⇒ 先验被证伪，按 H2 不反号、直接否决。

差异论证（H3/H4）：在库 229 项与已删 374 项中**没有** SMF / 量价子集 VWAP 类构造
（最近邻 = 已删 `mf_large_order_avg_price` 厂商大单均价，分档口径与数据源均不同；
在库 `mf_*` 为供应商单笔金额分档口径）。须在报告中测量与 `mf_*`、`turnover_*`、
`momentum_*` 的 pooled |ρ|（policy §三/G5）。

════════════════════════════════════════════════════════════════════════
二、口径的窗口读法与两个变体（预登记，README 亦记录）
════════════════════════════════════════════════════════════════════════
原文（1min 约 230 棒）是**逐日**选子集、逐日算 SMF；本模块按 idea_bank r12u2_002
的逐字构造做**逐日**计算，两个登记变体的差异只在**最终平滑窗**：5 日均 与 20 日均。
⇒ `plan.md` C2 的“20 日窗内取强度前 20% 棒”若被读成“跨日池化选子集”则是**另一条**
构造（VWAP_day 的分母会变成跨日池），本模块**不登记**该读法（变体预算 2 用尽；
列入 README 不确定项）。原文为 1min 口径（约 230 棒），本项目用 5min（约 48 棒）
近似，**近似误差 = 子集颗粒度变粗**（每棒内部的价格路径信息被压成一根 bar），
无法在本项目内量化消除，如实记录。

════════════════════════════════════════════════════════════════════════
三、数据、PIT 与单位口径
════════════════════════════════════════════════════════════════════════
· 只经 `ctx.dataset("stock_history_5min", ...)` 读原始 5min 年分区；逐 (股票, 日) 段
  聚合后走 `ctx.panel.place`（精确落格、缺格 NaN；停牌日 NaN，绝不沿用旧值）。
· 5min 分区在 `conf/frequency.yaml` 中 `delay_days: 0`，不需要 `lag_grid`。
· **vol 手/股翻转免疫（本机制的关键设计点）**：`stock_history_5min.vol` 的单位
  分段翻转且不稳定（2025-12-01 前后 手↔股；见 fea/intraday.py 与 gcC/README 实测）。
  本机制对 vol 的两次使用都免疫：
    (i) 强度排序 smart ∝ 1/√vol：同一日内 vol 的整体单位因子（×100）不改变排序；
    (ii) VWAP_sel / VWAP_day：分子分母都在**同一日、同一单位**下计算，比值无量纲；
        最终跨日的 5/20 日均是对**无量纲比值**求均值。
  模块内**没有任何**跨日绝对量比较。
· 分段 = 「日期变化 | 时段回跳」（r3/r4b 范式），段内 (股票, 日) 唯一性抽样复核，
  不通过抛错；午休棒不剔除（本机制不做收益累乘）。
· 缺失：当日可用棒（o>0、c>0、vol>0、amount≥0 全有限）< MIN_BARS(40) ⇒ NaN；
  池外/未上市/ST（`ctx.universe` False）⇒ NaN；窗口均值 min_count = 5→4 / 20→15
  （75% 有效日，r4b 口径），不足 ⇒ NaN。
· **阈值 20% 的量基准** = 当日**可用棒**的 vol 合计（零成交/不可用棒不进入分母）。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

log = logging.getLogger("research.families.r12c_smf_v1")

SOURCE = "stock_history_5min"
MIN_BARS = 40            # 当日可用棒下限（与 R3-B / r4b 同口径）
FRAC = 0.20              # 高信息量成交子集的量份额（原文 20%）
AUDIT_SEGMENTS = 2000
# 事前冻结：(kind, window) -> 候选名；窗口只登记 5 / 20 两个期限（期限结构式，不扫网格）
FROZEN = {
    ("smf", 5): "r12c_smf_5_v1",
    ("smf", 20): "r12c_smf_20_v1",
}
MIN_VALID = {5: 4, 20: 15}


def _frozen_key(parameters):
    if not isinstance(parameters, dict):
        return None
    if parameters.get("kind") != "smf":
        return None
    return ("smf", parameters.get("window"))


def build(parameters):
    """只允许预登记的两个期限变体；不搜索窗口、不搜索方向。"""
    key = _frozen_key(parameters)
    if key not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))
    window = key[1]
    min_valid = MIN_VALID[window]

    def compute(ctx):
        daily = _smf_daily(ctx)
        value = ctx.roll_mean(daily, window, min_valid)
        value = np.where(np.isfinite(value), value, np.nan)
        grid = np.where(np.asarray(ctx.universe, dtype=bool), value, np.nan)
        log.info("r12c_smf window=%d panel_finite=%d", window, int(np.isfinite(grid).sum()))
        return np.ascontiguousarray(grid, dtype=np.float64)

    return compute


# ---------------------------------------------------------------- 分段工具
# 与 r12c_net_turnover_v1 / r4b_intraday_shape_v1 同一套（上游 = r3_intraday_moments_v2）。
def _arrow(series):
    arr = getattr(series, "array", None)
    pa_arr = getattr(arr, "_pa_array", None)
    if pa_arr is None:
        pa_arr = pa.array(series)
    return pa_arr


def _segments(frame):
    """切 (股票, 日) 段；返回 (starts, lengths, lunch_mask)。"""
    tt = _arrow(frame["trade_time"])
    n = len(tt)
    if n < 2:
        raise ValueError("stock_history_5min 年分区行数 < 2")
    t16 = pc.utf8_slice_codeunits(tt, 11, 16)
    back = pc.fill_null(pc.less(t16.slice(1, n - 1), t16.slice(0, n - 1)), False)
    back = np.asarray(back.to_numpy(zero_copy_only=False), dtype=bool)
    d10 = pc.utf8_slice_codeunits(tt, 0, 10)
    daychg = pc.not_equal(d10.slice(1, n - 1), d10.slice(0, n - 1))
    daychg = np.asarray(pc.fill_null(daychg, False).to_numpy(zero_copy_only=False), dtype=bool)
    boundary = back | daychg

    lunch = np.zeros(n, dtype=bool)
    at1130 = pc.equal(t16.slice(0, n - 1), pa.scalar("11:30"))
    at1130 = np.asarray(pc.fill_null(at1130, False).to_numpy(zero_copy_only=False), dtype=bool)
    cand = np.flatnonzero(at1130 & (~boundary))
    if cand.size:
        nxt = pc.take(t16, pa.array((cand + 1).astype(np.int64)))
        is_pm = np.asarray(pc.or_(pc.equal(nxt, pa.scalar("13:00")),
                                  pc.equal(nxt, pa.scalar("13:05"))).to_numpy(zero_copy_only=False),
                           dtype=bool)
        lunch[cand[is_pm] + 1] = True

    starts = np.flatnonzero(np.concatenate((np.array([True]), boundary)))
    lengths = np.diff(np.concatenate((starts, np.array([n], dtype=starts.dtype))))
    return starts.astype(np.int64), lengths.astype(np.int64), lunch


def _audit_segments(ctx, frame, starts, lengths, sample=AUDIT_SEGMENTS):
    if starts.size == 0:
        return
    sel = np.unique(np.linspace(0, starts.size - 1, min(sample, starts.size)).astype(np.int64))
    picks, gids = [], []
    for g, k in enumerate(sel):
        s, ln = int(starts[k]), int(lengths[k])
        take = np.unique(np.linspace(s, s + ln - 1, min(ln, 8)).astype(np.int64))
        picks.append(take)
        gids.append(np.full(take.size, g, dtype=np.int64))
    rows = np.concatenate(picks)
    gid = np.concatenate(gids)
    codes = frame["stock_code"].iloc[rows].to_numpy()
    days = np.asarray(ctx.date_col(frame["trade_time"].iloc[rows].str.slice(0, 10)), dtype=np.int64)
    chk = pd.DataFrame({"g": gid, "c": codes, "d": days}).groupby("g", sort=False).nunique()
    bad = chk[(chk["c"] != 1) | (chk["d"] != 1)]
    if len(bad):
        raise ValueError("分段完整性复核失败：%d 个抽样段内出现多个股票代码或日期，"
                         "说明段边界被漏切（拒绝继续）" % len(bad))


def _year_facts(ctx, frame, starts, lengths):
    _audit_segments(ctx, frame, starts, lengths)
    code_seg = frame["stock_code"].iloc[starts].to_numpy()
    day_seg = np.asarray(ctx.date_col(frame["trade_time"].iloc[starts].str.slice(0, 10)),
                         dtype=np.int64)
    if not np.all(day_seg > 0):
        raise ValueError("trade_time 不能全部解析为合法日期")
    pos_seg = np.asarray(ctx.code_index(code_seg), dtype=np.int64)
    keep = np.flatnonzero((pos_seg >= 0) & (lengths >= MIN_BARS))
    if keep.size == 0:
        return None
    key = pos_seg[keep] * np.int64(100_000_000) + day_seg[keep]
    if np.unique(key).size != key.size:
        raise ValueError("(股票, 日) 段不唯一：年分区存在重复股票日，拒绝继续")
    return pos_seg, day_seg, keep


def _gather(frame, starts, lengths, keep, cols):
    lens = lengths[keep]
    total = int(lens.sum())
    cum = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lens)))
    rows = np.repeat(starts[keep], lens) + (np.arange(total, dtype=np.int64)
                                            - np.repeat(cum[:-1], lens))
    out = {}
    for c in cols:
        allv = frame[c].to_numpy(np.float64)
        out[c] = allv[rows]
        del allv
    return out, rows, lens


# ---------------------------------------------------------------- C2 日频值
def _smf_daily(ctx):
    """逐 (股票, 日) 的当日 SMF（未滚动）铺进面板；池外与不足 MIN_BARS 记 NaN。"""
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    y0, y1 = int(days[0]) // 10000, int(days[-1]) // 10000
    cols = ["stock_code", "trade_time", "open", "close", "vol", "amount"]
    daily = None
    for year in range(y0, y1 + 1):
        frame = ctx.dataset(SOURCE, columns=cols, years=(year, year))
        if frame is None or frame.empty:
            continue
        starts, lengths, _lunch = _segments(frame)
        facts = _year_facts(ctx, frame, starts, lengths)
        if facts is None:
            continue
        pos_seg, day_seg, keep = facts
        data, rows, lens = _gather(frame, starts, lengths, keep,
                                   ["open", "close", "vol", "amount"])
        seg_id = np.repeat(np.arange(lens.size, dtype=np.int64), lens)
        seg_starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lens)[:-1]))
        value = _daily_value(data, seg_starts, lens, seg_id)
        g = ctx.panel.place(pos_seg[keep], day_seg[keep], value)
        daily = g if daily is None else np.where(np.isfinite(g), g, daily)
        log.info("r12c_smf year=%d segs=%d kept=%d bars=%d finite_day=%d",
                 year, int(starts.size), int(keep.size), int(lens.sum()),
                 int(np.isfinite(value).sum()))
        del frame, data, rows, seg_id
    if daily is None:
        return ctx.panel.empty(dtype=np.float64)
    return daily


def _daily_value(data, seg_starts, lens, seg_id):
    """每个保留段的当日 SMF = VWAP_sel / VWAP_day − 1。

    · 强度 smart = |close/open − 1| / √vol；桶内排序用 `np.lexsort`（主键段号、
      次键 −smart；lexsort 稳定 ⇒ 同强度的棒保持原始时间序，结果可复现）。
    · 子集累计：段内 vol 前缀和（`cumsum` 减去段基点）严格小于 20%×段内可用 vol 合计
      ⇒ 强度最高的棒必然入选，子集量略小于 20%（颗粒度 = 一根棒）。
    · 子集 VWAP 用棒 `close` 加权（原文口径）；全日 VWAP = Σamount/Σvol（可用棒）。
    """
    o, c, v, a = data["open"], data["close"], data["vol"], data["amount"]
    usable = (np.isfinite(o) & (o > 0.0) & np.isfinite(c) & (c > 0.0)
              & np.isfinite(v) & (v > 0.0) & np.isfinite(a) & (a >= 0.0))
    with np.errstate(all="ignore"):
        zdf = np.abs(c / np.where(usable, o, np.nan) - 1.0)
        smart = zdf / np.sqrt(np.where(usable, v, np.nan))
    neg = np.where(usable, -smart, np.inf)          # 不可用棒排到段尾
    order = np.lexsort((neg, seg_id.astype(np.int32)))

    usable_s = usable[order]
    v_s = np.where(usable_s, v[order], 0.0)
    vc_s = np.where(usable_s, v_s * c[order], 0.0)
    a_s = np.where(usable_s, a[order], 0.0)

    cs = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(v_s)))
    base = cs[seg_starts]
    cum_prev = cs[:-1] - np.repeat(base, lens)          # 段内、当前棒之前的累计量
    total = cs[seg_starts + lens] - base
    keep = cum_prev < FRAC * np.repeat(total, lens)

    num = np.add.reduceat(np.where(keep, vc_s, 0.0), seg_starts)
    den = np.add.reduceat(np.where(keep, v_s, 0.0), seg_starts)
    day_v = np.add.reduceat(v_s, seg_starts)
    day_a = np.add.reduceat(a_s, seg_starts)
    n_ok = np.add.reduceat(usable_s.astype(np.float64), seg_starts)

    vwap_sel = np.where(den > 0.0, num / np.where(den > 0.0, den, np.nan), np.nan)
    vwap_day = np.where(day_v > 0.0, day_a / np.where(day_v > 0.0, day_v, np.nan), np.nan)
    with np.errstate(all="ignore"):
        smf = vwap_sel / vwap_day - 1.0
    good = (n_ok >= MIN_BARS) & np.isfinite(smf) & (vwap_day > 0.0)
    return np.where(good, smf, np.nan)
