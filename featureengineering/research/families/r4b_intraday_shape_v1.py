"""R4-B：日内成交/路径形状（Lead 预登记 → 独立执行；不写生产）。

族 `r4b_intraday_shape_v1`；机制、方向、期限、试验预算由
`artifacts/research/round4_20261004/RESEARCH_BRIEF_B1.md` §4-U2 事前冻结：

  1. intraday_volume_hhi_20        日内 48 bar 成交量份额 HHI 的 20 日均        false
  2. intraday_path_inefficiency_20 1 − |当日净收益| / Σ|bar 收益| 的 20 日均    false
  3. close30_volume_share_20       尾盘 14:30–15:00 成交额份额的 20 日均        false
  4. upbar_volume_share_20         上涨 bar 的成交量占比的 20 日均               true
  5. close_location_value_20       (close−low)/(high−low) 的 20 日均（日线口径） true

实现范式沿用 `research/families/r3_intraday_moments_v2.py`（已入库、verify 通过）：

  * 分段 = 「日期变化 | 时段回跳」（不依赖 48 网格；兼容 2026 分区的不规整：
    实测有 47/49 根 bar 的股票日）。段内「(股票,日) 唯一、HH:MM 单调不减」抽样复核，
    不通过直接报错。
  * 午休跨段收益剔除：同一段内 `"11:30"` 之后紧跟 `"13:00"/"13:05"` 的那一根
    bar 的收益置 NaN —— 沿用 R3-B 口径。只影响机制 2（收益路径）；份额类机制
    （HHI/尾盘/上涨占比）用当日全部 bar，不受影响。
  * 只经 `ctx.dataset("stock_history_5min", ...)` 读原始 5min（截断审计管得到）；
    价格未复权 —— 本族只做日内比较与同日比值，不跨日比价，无需 adj_factor。
  * 每 (机制, 年) 任务把面板跨越的所有年分区都读进来（面板含 warmup 前缀），
    否则年初首值的 20 日窗会缺 warmup 数据。

单位口径（2025-12-01 手/股翻转的历史坑）：机制 1/2/4 一律用**份额或比值**
（Σv²/V²、占比），对整体单位翻转严格不变；机制 3 用 `amount`（元，全程一致）。
不做任何绝对量的跨日比较。

每日值 → 20 日均：`ctx.roll_mean(daily, 20, min_count=15)`（20 日窗内至少 15 个
有效日，否则 NaN）。warmup_days=45（日历天，≈30 个交易日 > 20 日窗）。

已冻结的规则细节（事前定死，见 registry 登记）：
  * 上涨 bar 方向 = bar 内 close vs open；**持平 bar 记 0.5 权重**（方向不可判定
    的 bar 同样 0.5 权重）；分母 = 当日 vol 有限且 ≥0 的 bar 之和。
  * 尾盘 30 分钟 = **右端标签** {14:35, 14:40, …, 15:00} 六根（厂商时间戳为区间
    右端，与 fea/closing30.py 一致：14:35 的 open 是 14:30 价格）。
  * 净收益口径 = 段内可用 bar 收益的**算术和**（与分母同为可用集合，午休跨段收益
    在分子分母中一致剔除）；由三角不等式保证 PI∈[0,1]。
  * `close_location_value`：`high > low` 且价格有限为正才定义；`high == low`
    （涨跌停锁死等）记 NaN。

已知局限（如实记录，不影响方向）：
  * 每日 bar 数 != 48 时不做网格补齐：份额类按当日实际 bar 计算
    （HHI = Σv²/V² 与 bar 数无关，定义一致）；尾盘段按右端标签选 bar，
    缺标签当日按实际存在的子集求和。
  * 2026 少量股票日的重复行（49 根 bar）会被当作两根 bar 计入份额（占比 <0.03%）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

log = logging.getLogger("research.families.r4b_intraday_shape_v1")

SOURCE = "stock_history_5min"
WINDOW = 20          # 滚动窗口（交易日）：20 日均
MIN_VALID = 15       # 20 日窗内有效日下限
MIN_BARS = 40        # 当日可用 bar 下限（与 R3-B 的 MIN_RETURNS=40 同口径）
AUDIT_SEGMENTS = 2000
TAIL_HM = (1435, 1440, 1445, 1450, 1455, 1500)   # 右端标签：尾盘 14:30–15:00 六根
SIGNALS = ("intraday_volume_hhi_20", "intraday_path_inefficiency_20",
           "close30_volume_share_20", "upbar_volume_share_20",
           "close_location_value_20")
_NEEDS = {
    "intraday_volume_hhi_20": ("vol",),
    "intraday_path_inefficiency_20": ("close",),
    "close30_volume_share_20": ("amount",),
    "upbar_volume_share_20": ("open", "close", "vol"),
}


def build(parameters):
    """Family 只允许预登记的五个信号标识；不搜索窗口、不搜索方向。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("R4-B 只允许预登记的五个信号标识")
    signal = parameters["signal"]

    def compute(ctx):
        if signal == "close_location_value_20":
            daily = _clv_daily(ctx)
        else:
            daily = _intraday_daily(ctx, signal)
        value = ctx.roll_mean(daily, WINDOW, MIN_VALID)
        value = np.where(np.isfinite(value), value, np.nan)
        grid = np.where(np.asarray(ctx.universe, dtype=bool), value, np.nan)
        log.info("r4b %s panel_finite=%d", signal, int(np.isfinite(grid).sum()))
        return np.ascontiguousarray(grid, dtype=np.float64)

    return compute


# ---------------------------------------------------------------- 分段工具
def _arrow(series):
    """pandas 的 arrow-backed 字符串列 -> pyarrow 数组（零拷贝优先）。"""
    arr = getattr(series, "array", None)
    pa_arr = getattr(arr, "_pa_array", None)
    if pa_arr is None:
        pa_arr = pa.array(series)
    return pa_arr


def _segments(frame):
    """切 (股票, 日) 段 + 标出午休跨段收益；返回 (starts, lengths, lunch_mask)。

    与 r3_intraday_moments_v2 完全一致：段边界 = 日期变化 或 时段回跳。
    """
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
    """抽样复核：每个段内的股票代码与日期必须唯一（能发现被误合并的段）。"""
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
    """段头取股票位与日期；唯一性守卫（同 R3-B）。"""
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
    """把保留段的行按段连续聚拢；逐列取出（控制峰值内存）。"""
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


def _hm_of(frame, rows):
    """取出行的 HH:MM 数值标签（如 1435）。"""
    tt = _arrow(frame["trade_time"].iloc[rows])
    h = np.asarray(pc.utf8_slice_codeunits(tt, 11, 13).to_numpy(zero_copy_only=False))
    m = np.asarray(pc.utf8_slice_codeunits(tt, 14, 16).to_numpy(zero_copy_only=False))
    return (h.astype(np.int32) * 100 + m.astype(np.int32))


# ---------------------------------------------------------------- 日内机制
def _intraday_daily(ctx, signal):
    """机制 1–4：逐 (股票, 日) 的当日值（未滚动）铺进面板。"""
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    y0, y1 = int(days[0]) // 10000, int(days[-1]) // 10000
    cols = ["stock_code", "trade_time", *sorted(_NEEDS[signal])]
    daily = None
    for year in range(y0, y1 + 1):
        frame = ctx.dataset(SOURCE, columns=cols, years=(year, year))
        if frame is None or frame.empty:
            continue
        starts, lengths, lunch = _segments(frame)
        facts = _year_facts(ctx, frame, starts, lengths)
        if facts is None:
            continue
        pos_seg, day_seg, keep = facts
        n_irregular = int(np.count_nonzero(lengths[keep] != 48))
        need = _NEEDS[signal]          # trade_time 走 _hm_of 直取，不进 float64 聚拢
        data, rows, lens = _gather(frame, starts, lengths, keep, list(need))
        seg_starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lens)[:-1]))
        value = _daily_value(signal, frame, rows, seg_starts, lens, data,
                             lunch[rows], keep, lengths)
        g = ctx.panel.place(pos_seg[keep], day_seg[keep], value)
        daily = g if daily is None else np.where(np.isfinite(g), g, daily)
        log.info("r4b %s year=%d segs=%d kept=%d bars=%d irregular=%d finite_day=%d",
                 signal, year, int(starts.size), int(keep.size), int(lens.sum()),
                 n_irregular, int(np.isfinite(value).sum()))
        del frame, data, rows
    if daily is None:
        return ctx.panel.empty(dtype=np.float64)
    return daily


def _daily_value(signal, frame, rows, seg_starts, lens, data, lunch_rows, keep, lengths):
    """按机制算每个保留段的当日值；bar 数不足 MIN_BARS 或退化 -> NaN。"""
    if signal == "intraday_volume_hhi_20":
        v = data["vol"]
        ok = np.isfinite(v) & (v >= 0.0)
        n_ok = np.add.reduceat(ok.astype(np.float64), seg_starts)
        v0 = np.where(ok, v, 0.0)
        tot = np.add.reduceat(v0, seg_starts)
        sq = np.add.reduceat(v0 * v0, seg_starts)
        with np.errstate(all="ignore"):
            hhi = sq / np.where(tot > 0.0, tot * tot, np.nan)
        good = (n_ok >= MIN_BARS) & (tot > 0.0) & np.isfinite(hhi)
        return np.where(good, hhi, np.nan)

    if signal == "intraday_path_inefficiency_20":
        c = data["close"]
        prev = np.empty_like(c)
        prev[0] = np.nan
        prev[1:] = c[:-1]
        prev[seg_starts] = np.nan
        with np.errstate(all="ignore"):
            r = c / prev - 1.0
        usable = np.isfinite(r) & (c > 0.0) & (prev > 0.0)
        r = np.where(usable, r, np.nan)
        r[lunch_rows] = np.nan                    # 午休跨段收益剔除（R3-B 口径）
        usable = np.isfinite(r)
        del prev
        r0 = np.where(usable, r, 0.0)
        n_ret = np.add.reduceat(usable.astype(np.float64), seg_starts)
        s_abs = np.add.reduceat(np.abs(r0), seg_starts)
        s_net = np.add.reduceat(r0, seg_starts)
        with np.errstate(all="ignore"):
            pi = 1.0 - np.abs(s_net) / np.where(s_abs > 0.0, s_abs, np.nan)
        good = (n_ret >= MIN_BARS) & (s_abs > 0.0) & np.isfinite(pi)
        return np.where(good, pi, np.nan)

    if signal == "close30_volume_share_20":
        a = data["amount"]
        ok = np.isfinite(a) & (a >= 0.0)
        n_ok = np.add.reduceat(ok.astype(np.float64), seg_starts)
        a0 = np.where(ok, a, 0.0)
        tot = np.add.reduceat(a0, seg_starts)
        hm = _hm_of(frame, rows)
        tail = np.zeros(hm.shape, dtype=bool)
        for lab in TAIL_HM:
            tail |= hm == lab
        tail_a = np.add.reduceat(np.where(tail, a0, 0.0), seg_starts)
        with np.errstate(all="ignore"):
            share = tail_a / np.where(tot > 0.0, tot, np.nan)
        good = (n_ok >= MIN_BARS) & (tot > 0.0) & np.isfinite(share)
        return np.where(good, share, np.nan)

    # upbar_volume_share_20
    o, c, v = data["open"], data["close"], data["vol"]
    dv = np.isfinite(o) & (o > 0.0) & np.isfinite(c) & (c > 0.0)
    up = dv & (c > o)
    dn = dv & (c < o)
    w = np.where(up, 1.0, np.where(dn, 0.0, 0.5))   # 持平/方向不可判定 = 0.5 权重
    ok = np.isfinite(v) & (v >= 0.0)
    n_ok = np.add.reduceat(ok.astype(np.float64), seg_starts)
    v0 = np.where(ok, v, 0.0)
    tot = np.add.reduceat(v0, seg_starts)
    num = np.add.reduceat(v0 * w, seg_starts)
    with np.errstate(all="ignore"):
        share = num / np.where(tot > 0.0, tot, np.nan)
    good = (n_ok >= MIN_BARS) & (tot > 0.0) & np.isfinite(share)
    return np.where(good, share, np.nan)


# ---------------------------------------------------------------- 日线机制
def _clv_daily(ctx):
    """close_location_value：日线口径 (close−low)/(high−low)，同日比值。"""
    high = ctx.px_raw("high")
    low = ctx.px_raw("low")
    close = ctx.px_raw("close")
    with np.errstate(all="ignore"):
        clv = (close - low) / (high - low)
    ok = (np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
          & (high > low) & (low > 0.0))
    return np.where(ok & np.isfinite(clv), clv, np.nan)
