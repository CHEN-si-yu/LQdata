"""保留迁移（清理复审，2026-10-03）：仅 rv_skewness。

来源：research/families/r3_intraday_moments_v2.py（FILE-SHA 封存，不改动原件）。
_year 的共享预计算逐字保留；被拒信号的判定分支与 vol_price_diffusion 专用路径删去。
参数与方向不变；新身份须重新走原生 G1 与旧生产值精确 parity 后才批准。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc


SIGNALS = ("rv_skewness",)
KEPT_SIGNALS = SIGNALS

log = logging.getLogger("research.families.r3_intraday_moments_v2")


SOURCE = "stock_history_5min"


TRUNC_SIGMA = 6.0         # 截尾倍数（相对当日 BPV 尺度）


MIN_RETURNS = 40          # 当日可用 bar 收益下限


AUDIT_SEGMENTS = 2000     # 抽样完整性复核的段数


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


def _read_year(ctx, year, need_volume):
    columns = ["stock_code", "trade_time", "close"] + (["vol"] if need_volume else [])
    frame = ctx.dataset(SOURCE, columns=columns, years=(year, year))
    if frame is None or frame.empty:
        raise ValueError("stock_history_5min year=%d 无可读行" % year)
    starts, lengths, lunch = _segments(frame)
    _audit_segments(ctx, frame, starts, lengths)
    code_seg = frame["stock_code"].iloc[starts].to_numpy()
    day_seg = np.asarray(ctx.date_col(frame["trade_time"].iloc[starts].str.slice(0, 10)), dtype=np.int64)
    if not np.all(day_seg > 0):
        raise ValueError("trade_time 不能全部解析为合法日期")
    pos_seg = np.asarray(ctx.code_index(code_seg), dtype=np.int64)
    keep = np.flatnonzero((pos_seg >= 0) & (lengths >= MIN_RETURNS))
    if keep.size == 0:
        return None
    key = pos_seg[keep] * np.int64(100_000_000) + day_seg[keep]
    if np.unique(key).size != key.size:
        raise ValueError("(股票, 日) 段不唯一：年分区存在重复股票日，拒绝继续")
    return frame, starts, lengths, lunch, pos_seg, day_seg, keep


def _gather(frame, starts, lengths, lunch, keep, need_volume):
    lens = lengths[keep]
    total = int(lens.sum())
    cum = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lens)))
    rows = np.repeat(starts[keep], lens) + (np.arange(total, dtype=np.int64) - np.repeat(cum[:-1], lens))
    close_all = frame["close"].to_numpy(np.float64)
    close = close_all[rows]
    vol = frame["vol"].to_numpy(np.float64)[rows] if need_volume else None
    del close_all
    lunch_rows = lunch[rows]
    return close, vol, lunch_rows, cum[:-1]


def build(parameters):
    """保留迁移：仅 rv_skewness（参数与方向与原登记一致）。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in KEPT_SIGNALS:
        raise ValueError("retained family permits only the preserved preregistered signal")
    signal = parameters["signal"]

    def compute(ctx):
        return _year(ctx, signal)

    return compute


def _year(ctx, signal):
    """一个输出年度的全部候选值（同一份 5min 分区只读一次）。"""
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    year = int(str(int(days[-1]))[:4])
    need_volume = False  # 保留迁移：rv_skewness 不需要量
    loaded = _read_year(ctx, year, need_volume)
    if loaded is None:
        return ctx.panel.empty(dtype=np.float64)
    frame, starts, lengths, lunch, pos_seg, day_seg, keep = loaded
    close, vol, lunch_rows, seg_starts = _gather(frame, starts, lengths, lunch, keep, need_volume)
    n_seg_total = int(starts.size)
    n_irregular = int(np.count_nonzero(lengths[keep] != 48))
    del frame, lunch, starts, lengths

    n_rows = int(close.size)
    prev = np.empty(n_rows, dtype=np.float64)
    prev[0] = np.nan
    prev[1:] = close[:-1]
    prev[seg_starts] = np.nan
    with np.errstate(all="ignore"):
        ret = close / prev - 1.0
    usable = np.isfinite(ret) & (close > 0.0) & (prev > 0.0)
    ret = np.where(usable, ret, np.nan)
    ret[lunch_rows] = np.nan
    usable = np.isfinite(ret)
    del prev, close, lunch_rows
    r0 = np.where(usable, ret, 0.0)
    del ret

    n_used = np.add.reduceat(usable.astype(np.float64), seg_starts)
    rv = np.add.reduceat(r0 * r0, seg_starts)
    absr = np.abs(r0)
    pair = np.zeros(n_rows, dtype=np.float64)
    pair[1:] = np.where(usable[1:] & usable[:-1], absr[1:] * absr[:-1], 0.0)
    bpv = np.add.reduceat(pair, seg_starts) * (np.pi / 2.0)
    del pair

    lens = np.diff(np.concatenate((seg_starts, np.array([n_rows], dtype=seg_starts.dtype))))
    scale = np.sqrt(np.maximum(bpv / np.maximum(n_used, 1.0), 0.0))
    limit_row = np.repeat(TRUNC_SIGMA * scale, lens)
    trunc = np.where(usable, np.clip(r0, -limit_row, limit_row), 0.0)
    del limit_row
    sq = trunc * trunc
    s2t = np.add.reduceat(sq, seg_starts)
    s3t = np.add.reduceat(sq * trunc, seg_starts)
    up2 = np.add.reduceat(np.where(r0 > 0.0, sq, 0.0), seg_starts)
    dn2 = np.add.reduceat(np.where(r0 < 0.0, sq, 0.0), seg_starts)
    del sq, trunc, r0

    full = n_used >= MIN_RETURNS
    if signal == "rv_skewness":
        with np.errstate(all="ignore"):
            rs = np.sqrt(n_used) * s3t / np.power(s2t, 1.5)
        value = np.where(full & (s2t > 0.0) & np.isfinite(rs), -rs, np.nan)
    else:
        raise ValueError("unregistered signal")
    value = np.where(np.isfinite(value), value, np.nan)
    grid = ctx.panel.place(pos_seg[keep], day_seg[keep], value)
    grid = np.where(np.asarray(ctx.universe, dtype=bool), grid, np.nan)
    log.info("r3b-v2 %s year=%d segs=%d kept=%d bars=%d irregular_seg=%d finite=%d panel_finite=%d",
             signal, year, n_seg_total, int(keep.size), n_rows, n_irregular,
             int(np.isfinite(value).sum()), int(np.isfinite(grid).sum()))
    return np.ascontiguousarray(grid, dtype=np.float64)
