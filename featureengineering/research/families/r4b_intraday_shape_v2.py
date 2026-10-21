"""R4-B v2：日内成交/路径形状（Lead 14:37 纠偏后的替换机制；不写生产）。

族 `r4b_intraday_shape_v2`。v1（`r4b_intraday_shape_v1.py`）的机制 3/5 撞已删因子被撤销
（close30_volume_share_20 ≙ 已删 close30_amt_share；close_location_value_20 ≙ 已删
idt_close_position / id2_am_close_position），替换为本模块的两个机制（方向由 Lead 冻结）：

  新 3 intraday_rv_kurtosis_20   日内 5min 收益已实现峰度（±6σ_bpv 截尾）的 20 日均   false
  新 5 vol_lead_ret_corr_20      corr(Δlog vol_t, r_{t+1})（当日 bar 内相邻）20 日均  true

本模块**自包含**（不 import v1），其源文件哈希即候选定义哈希；v1 模块零改动。

口径全部沿用 R3-B `r3_intraday_moments_v2.py` 家族既有约定（保持同族一致）：

  * 分段 = 「日期变化 | 时段回跳」；段内 (股票,日) 唯一性抽样复核，不通过直接报错。
  * 午休跨段收益剔除：段内 `"11:30"` 后紧跟 `"13:00"/"13:05"` 的那一根 bar 收益置 NaN。
    （只剔价格收益；量变化 Δlog vol 不属收益、跨午休保留。）
  * rv_kurtosis：同 rv_skewness 的截尾口径 —— 当日 BPV = (π/2)Σ|r_j||r_{j-1}|（相邻皆可用），
    scale = √(BPV/n)，逐 bar 截尾 |r|≤6σ_bpv；s2=Σtrunc²、s4=Σtrunc⁴；
    值 = n_used × s4 / s2²（非去均值的标准化 4 阶矩；对全部 |x| 相等的序列值=1，故定义域 ≥1；
    与 rv_skewness 的 √n·s3/s2^1.5 同一累加口径）。当日可用收益 <40 → NaN；s2=0 → NaN。
  * vol_lead_ret_corr：x_t = log vol_t − log vol_{t−1}（vol>0 才可用），y_t = r_{t+1}
    （close_{t+1}/close_t − 1，午休与段首自动为 NaN）；仅当日 bar 内相邻 t→t+1；
    对有效对算 Pearson 相关；有效对 <40 或 x/y 零方差 → 当日 NaN。
    相关对 vol 的手/股单位翻转不变（log 差平移不变）。
  * 每 (机制, 年) 任务读面板跨越的所有年分区（warmup 前缀需要）。

每日值 → 20 日均：`ctx.roll_mean(daily, 20, min_count=15)`。warmup_days=45。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

log = logging.getLogger("research.families.r4b_intraday_shape_v2")

SOURCE = "stock_history_5min"
WINDOW = 20          # 滚动窗口（交易日）：20 日均
MIN_VALID = 15       # 20 日窗内有效日下限
MIN_BARS = 40        # 当日可用 bar/有效对下限（与 R3-B MIN_RETURNS=40 同口径）
AUDIT_SEGMENTS = 2000
TRUNC_SIGMA = 6.0    # 截尾倍数（相对当日 BPV 尺度，同 rv_skewness）
SIGNALS = ("intraday_rv_kurtosis_20", "vol_lead_ret_corr_20")
_NEEDS = {
    "intraday_rv_kurtosis_20": ("close",),
    "vol_lead_ret_corr_20": ("close", "vol"),
}


def build(parameters):
    """Family 只允许预登记的两个信号标识；不搜索窗口、不搜索方向。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("R4-B v2 只允许预登记的两个信号标识")
    signal = parameters["signal"]

    def compute(ctx):
        daily = _intraday_daily(ctx, signal)
        value = ctx.roll_mean(daily, WINDOW, MIN_VALID)
        value = np.where(np.isfinite(value), value, np.nan)
        grid = np.where(np.asarray(ctx.universe, dtype=bool), value, np.nan)
        log.info("r4b-v2 %s panel_finite=%d", signal, int(np.isfinite(grid).sum()))
        return np.ascontiguousarray(grid, dtype=np.float64)

    return compute


# ---------------------------------------------------------------- 分段工具（与 v1 逐字一致）
def _arrow(series):
    """pandas 的 arrow-backed 字符串列 -> pyarrow 数组（零拷贝优先）。"""
    arr = getattr(series, "array", None)
    pa_arr = getattr(arr, "_pa_array", None)
    if pa_arr is None:
        pa_arr = pa.array(series)
    return pa_arr


def _segments(frame):
    """切 (股票, 日) 段 + 标出午休跨段收益；返回 (starts, lengths, lunch_mask)。"""
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


# ---------------------------------------------------------------- 日内机制
def _intraday_daily(ctx, signal):
    """逐 (股票, 日) 的当日值（未滚动）铺进面板。"""
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
        need = _NEEDS[signal]
        data, rows, lens = _gather(frame, starts, lengths, keep, list(need))
        seg_starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lens)[:-1]))
        value = _daily_value(signal, seg_starts, lens, data, lunch[rows])
        g = ctx.panel.place(pos_seg[keep], day_seg[keep], value)
        daily = g if daily is None else np.where(np.isfinite(g), g, daily)
        log.info("r4b-v2 %s year=%d segs=%d kept=%d bars=%d irregular=%d finite_day=%d",
                 signal, year, int(starts.size), int(keep.size), int(lens.sum()),
                 n_irregular, int(np.isfinite(value).sum()))
        del frame, data, rows
    if daily is None:
        return ctx.panel.empty(dtype=np.float64)
    return daily


def _returns_with_lunch(seg_starts, n_rows, data, lunch_rows):
    """段内 close-to-close 收益；段首/午休跨段/非正价格 -> NaN。返回 (r_nan, usable)。"""
    c = data["close"]
    prev = np.empty_like(c)
    prev[0] = np.nan
    prev[1:] = c[:-1]
    prev[seg_starts] = np.nan
    with np.errstate(all="ignore"):
        r = c / prev - 1.0
    usable = np.isfinite(r) & (c > 0.0) & (prev > 0.0)
    r = np.where(usable, r, np.nan)
    r[lunch_rows] = np.nan
    usable = np.isfinite(r)
    del prev
    return r, usable


def _daily_value(signal, seg_starts, lens, data, lunch_rows):
    """按机制算每个保留段的当日值；bar/有效对不足或退化 -> NaN。"""
    n_rows = int(lens.sum())
    if signal == "intraday_rv_kurtosis_20":
        r, usable = _returns_with_lunch(seg_starts, n_rows, data, lunch_rows)
        r0 = np.where(usable, r, 0.0)
        n_used = np.add.reduceat(usable.astype(np.float64), seg_starts)
        absr = np.abs(r0)
        pair = np.zeros(n_rows, dtype=np.float64)
        pair[1:] = np.where(usable[1:] & usable[:-1], absr[1:] * absr[:-1], 0.0)
        bpv = np.add.reduceat(pair, seg_starts) * (np.pi / 2.0)
        del pair, absr
        scale = np.sqrt(np.maximum(bpv / np.maximum(n_used, 1.0), 0.0))
        limit_row = np.repeat(TRUNC_SIGMA * scale, lens)
        trunc = np.where(usable, np.clip(r0, -limit_row, limit_row), 0.0)
        del limit_row, r0, r
        sq = trunc * trunc
        s2t = np.add.reduceat(sq, seg_starts)
        s4t = np.add.reduceat(sq * sq, seg_starts)
        del sq, trunc
        with np.errstate(all="ignore"):
            kurt = n_used * s4t / np.where(s2t > 0.0, s2t * s2t, np.nan)
        good = (n_used >= MIN_BARS) & (s2t > 0.0) & np.isfinite(kurt)
        return np.where(good, kurt, np.nan)

    # vol_lead_ret_corr_20
    r, _ = _returns_with_lunch(seg_starts, n_rows, data, lunch_rows)
    v = data["vol"]
    okv = np.isfinite(v) & (v > 0.0)
    lv = np.where(okv, np.log(np.where(okv, v, 1.0)), np.nan)
    del okv
    dlv = np.empty_like(lv)
    dlv[0] = np.nan
    dlv[1:] = lv[1:] - lv[:-1]
    dlv[seg_starts] = np.nan
    del lv
    p_x = dlv
    p_y = np.empty_like(r)
    p_y[-1] = np.nan
    p_y[:-1] = r[1:]          # 段首与午休的 r 已为 NaN -> 边界/跨段自动剔除
    del r
    valid = np.isfinite(p_x) & np.isfinite(p_y)
    x0 = np.where(valid, p_x, 0.0)
    y0 = np.where(valid, p_y, 0.0)
    del p_x, p_y, dlv
    n_p = np.add.reduceat(valid.astype(np.float64), seg_starts)
    sx = np.add.reduceat(x0, seg_starts)
    sy = np.add.reduceat(y0, seg_starts)
    sxx = np.add.reduceat(x0 * x0, seg_starts)
    syy = np.add.reduceat(y0 * y0, seg_starts)
    sxy = np.add.reduceat(x0 * y0, seg_starts)
    del x0, y0
    with np.errstate(all="ignore"):
        dxx = n_p * sxx - sx * sx
        dyy = n_p * syy - sy * sy
        corr = (n_p * sxy - sx * sy) / np.sqrt(dxx * dyy)
    good = (n_p >= MIN_BARS) & (dxx > 0.0) & (dyy > 0.0) & np.isfinite(corr)
    out = np.where(good, np.clip(corr, -1.0, 1.0), np.nan)
    return out
