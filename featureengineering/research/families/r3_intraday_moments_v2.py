"""R3-B v2：日内已实现高阶矩与跳跃（Lead 预登记 → 独立执行；不写生产）。

与 v1 的差异只有一处、且是**实现缺陷修复**，不涉及任何机制/方向/参数变化：
v1 假设「年分区按 (stock_code, trade_time) 排序，且全局第 48k 行一定是某个 (股票,日) 的首根 bar」。
v1 的冒烟测试（2026-08-01..2026-09-30）在这个断言上直接报错：
  2026 年分区 rows%48 = 15，有 10 处同股票块内日期非递增；
  例如 000503.SZ 的 2026-08-03 有 49 根 bar（重复行）、2026-08-05 只有 47 根；
  全池 share_48 = 0.99972（382,447 个池内股票日中约 107 个不是 48 根）。
2018-2025 年分区完全规整（rows%48=0、head 行数与股票数一致、无异常），但 v1 因此对 2026 不可用。
v2 改为**完全不依赖 48 网格**，只依赖两条实测性质：
  ① 分区内同一 (股票,日) 的行连续；
  ② 同一段内 bar 的时段 HH:MM 单调不减，跨日/换股票时时段回跳（15:00 -> 09:35）。
段边界 = 日期变化 或 时段回跳；午休跨段收益 = 同一段内 "11:30" 之后紧跟 "13:00"/"13:05" 的那一根。
另外对最多 2000 个抽样段做「段内股票代码与日期必须唯一」的完整性复核，不通过就报错。

数据：stock_history_5min（stock_code, trade_time, close, vol），只经 ctx.dataset 读取（不做直接
文件 IO，research verify 的截断审计才管得到）。池：conf/universe_frozen.tsv 的 2115 只主板股；
区间 2018-01-02 … 2026-09-30。

实测数据事实（dsh_scratch/round3/intraday/recon_coverage.json、diag_grid.json）
  * 2018-2025：rows%48 == 0，每个 (股票,交易日) 恰好 48 根 bar，分区按 (stock_code, trade_time) 升序；
  * 2026：rows%48 = 15，约 107 个池内股票日 bar 数 != 48（49 根=重复行；47 根；最少 4 根）；
  * bar 时点集合为 09:35…11:30 与 13:00…15:00（49 个候选时刻，单日取其中 48 个）；
  * 价格未复权；本模块只做日内比较、不跨日比价，因此不需要 adj_factor；
  * vol 在 2025-12-01 前是「手」、之后是「股」（fea/intraday.py 实测）。量价回归只用日内份额
    v_i / mean(v)，对整体单位翻转严格不变。

估计量（全部同日、无跨日状态、无滚动窗口；warmup_days=1 只用于框架锚定年份）
  r_i        段内 bar 收益 close_i/close_{i-1}-1（段首无收益）；午休那一根置 NaN
  RV         求和 r_i^2
  BPV        (pi/2) * 求和 |r_i||r_{i-1}|（相邻日内对，BNS 抗噪跳跃检验的连续部分）
  sigma      sqrt(BPV/N)；r~_i = r_i 在 ±6*sigma 截尾
  RS         sqrt(N)*求和 r~^3 / (求和 r~^2)^1.5
因子值（符号即 higher_is_better 声明的方向）
  rv_daily            RV（朴素基线：已有 idt_rv_daily 的直译）
  rv_skewness         -RS（负偏度=崩盘风险溢价，越高越好）
  jump_share          max(RV-BPV,0)/RV（越高越差）
  signed_jump         (RV+ - RV-)/sum(r~^2)，取值 [-1,1]（越高越好）
  vol_price_diffusion 日内 OLS 斜率 slope(|r| 水平 ~ 成交量份额)，无量纲（越高越差）

已知局限（如实记录）：若某股票某日最后一根 bar 恰为 11:30、紧随其后同一日期的下一只股票首根
bar 恰为 13:00/13:05，则这两段会被判成「午休」而合并。抽样完整性复核正是为了在真实数据里发现
这种情况（复核不通过会直接报错）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

log = logging.getLogger("research.families.r3_intraday_moments_v2")

SOURCE = "stock_history_5min"
TRUNC_SIGMA = 6.0         # 截尾倍数（相对当日 BPV 尺度）
MIN_RETURNS = 40          # 当日可用 bar 收益下限
X_CAP = 8.0               # 量价回归：成交量份额截尾
Y_CAP = 8.0               # 量价回归：|收益| 相对水平截尾
AUDIT_SEGMENTS = 2000     # 抽样完整性复核的段数
SIGNALS = ("rv_daily", "rv_skewness", "jump_share", "signed_jump", "vol_price_diffusion")


def build(parameters):
    """Family 只允许预登记的五个信号标识；不搜索窗口、不搜索方向。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("R3-B 只允许预登记的五个信号标识")
    signal = parameters["signal"]

    def compute(ctx):
        return _year(ctx, signal)

    return compute


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


def _volume_price_slope(vol, usable, absr, seg_starts, full):
    """无量纲日内量价斜率：x=v/mean(v)、y=|r|/mean(|r|) 后做带截距 OLS。

    只用当日成交量份额，因此对 stock_history_5min.vol 在 2025-12-01 的手/股单位翻转
    严格不变（整体乘数在份额里约掉）。
    """
    vok = usable & np.isfinite(vol) & (vol > 0.0)
    n_v = np.add.reduceat(vok.astype(np.float64), seg_starts)
    v_use = np.where(vok, vol, 0.0)
    a_use = np.where(vok, absr, 0.0)
    sum_v = np.add.reduceat(v_use, seg_starts)
    sum_a = np.add.reduceat(a_use, seg_starts)
    with np.errstate(all="ignore"):
        mean_v = sum_v / np.where(n_v > 0.0, n_v, np.nan)
        mean_a = sum_a / np.where(n_v > 0.0, n_v, np.nan)
    lens = np.diff(np.concatenate((seg_starts, np.array([usable.size], dtype=seg_starts.dtype))))
    mean_v_row = np.repeat(mean_v, lens)
    mean_a_row = np.repeat(mean_a, lens)
    x = np.clip(np.where(vok, v_use / mean_v_row, 0.0), 0.0, X_CAP)
    y = np.clip(np.where(vok, a_use / mean_a_row, 0.0), 0.0, Y_CAP)
    del v_use, a_use, mean_v_row, mean_a_row
    s_x = np.add.reduceat(x, seg_starts)
    s_y = np.add.reduceat(y, seg_starts)
    s_xx = np.add.reduceat(x * x, seg_starts)
    s_xy = np.add.reduceat(x * y, seg_starts)
    del x, y
    n = np.maximum(n_v, 1.0)
    with np.errstate(all="ignore"):
        spread = s_xx - s_x * s_x / n
        cov = s_xy - s_x * s_y / n
        slope = cov / np.where(spread > 1e-12, spread, np.nan)
    ok = full & (n_v >= MIN_RETURNS) & (spread > 1e-12) & np.isfinite(slope)
    return np.where(ok, slope, np.nan)


def _year(ctx, signal):
    """一个输出年度的全部候选值（同一份 5min 分区只读一次）。"""
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    year = int(str(int(days[-1]))[:4])
    need_volume = signal == "vol_price_diffusion"
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
    if signal == "rv_daily":
        value = np.where(full & (rv > 0.0), rv, np.nan)
    elif signal == "rv_skewness":
        with np.errstate(all="ignore"):
            rs = np.sqrt(n_used) * s3t / np.power(s2t, 1.5)
        value = np.where(full & (s2t > 0.0) & np.isfinite(rs), -rs, np.nan)
    elif signal == "jump_share":
        with np.errstate(all="ignore"):
            share = np.maximum(rv - bpv, 0.0) / np.where(rv > 0.0, rv, np.nan)
        value = np.where(full & (rv > 0.0) & np.isfinite(share), share, np.nan)
    elif signal == "signed_jump":
        with np.errstate(all="ignore"):
            signed = (up2 - dn2) / np.where(s2t > 0.0, s2t, np.nan)
        value = np.where(full & (s2t > 0.0) & np.isfinite(signed), signed, np.nan)
    else:
        value = _volume_price_slope(vol, usable, absr, seg_starts, full)

    value = np.where(np.isfinite(value), value, np.nan)
    grid = ctx.panel.place(pos_seg[keep], day_seg[keep], value)
    grid = np.where(np.asarray(ctx.universe, dtype=bool), grid, np.nan)
    log.info("r3b-v2 %s year=%d segs=%d kept=%d bars=%d irregular_seg=%d finite=%d panel_finite=%d",
             signal, year, n_seg_total, int(keep.size), n_rows, n_irregular,
             int(np.isfinite(value).sum()), int(np.isfinite(grid).sum()))
    return np.ascontiguousarray(grid, dtype=np.float64)
