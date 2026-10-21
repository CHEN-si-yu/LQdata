"""R3-B：日内已实现高阶矩与跳跃（Lead 预登记 → 独立执行；不写生产）。

机制集合、预期方向、期限、试验预算均由 Lead 在 conf/research/round3_20261002/
intraday/idea.yaml 与 family.yaml 中事前固定；本模块只做具体表达与实现。

数据：stock_history_5min（stock_code, trade_time, close, vol），只经 ctx.dataset 读取
（不做直接文件 IO，research verify 的截断审计才管得到）。池：conf/universe_frozen.tsv
的 2115 只主板股；区间 2018-01-02 … 2026-09-30。

实测数据事实（见 dsh_scratch/round3/intraday/recon_coverage.json 与 report.md 第 2 节）
  * 2018-2026 每个 (股票, 交易日) 恰好 48 根 5min bar；bar 时点为 09:35…11:30（24 根）
    与 13:05…15:00（24 根）；
  * 年分区按 (stock_code, trade_time) 升序，每只股票是连续块、块内行数为 48 的整数倍
    —— 因此全局第 48k 行一定是某个 (股票, 日) 的首根 bar。本模块用这条结构做 O(1) 分段，
    并在每次调用时做两项廉价断言（块内日期严格递增 / 股票块连续），不成立就报错而不是静默错位；
  * 价格未复权；本模块只做日内比较、不跨日比价，因此不需要 adj_factor；
  * vol 在 2025-12-01 前是「手」、之后是「股」（fea/intraday.py 的实测结论）。量价回归只用
    日内份额 v_i / mean(v)，对整体单位翻转严格不变，所以不需要额外归一。

估计量（全部同日、无跨日状态、无滚动窗口；warmup_days=1 只用于框架锚定年份）
  r_i        段内 bar 收益 close_i/close_{i-1}-1，i=1..47；剔除 i=24（11:30→13:05 午休跨段收益）
  RV         求和 r_i^2
  BPV        (pi/2) * 求和 |r_i||r_{i-1}|（相邻日内对，BNS 抗噪跳跃检验的连续部分）
  sigma      sqrt(BPV/N)（双幂尺度，只用于截尾）
  r~_i       r_i 在 ±6*sigma 截尾
  RS         sqrt(N)*求和 r~^3 / (求和 r~^2)^1.5（已实现偏度）
  RV±        分别对 r~>0 与 r~<0 求和 r~^2
因子值（符号即 higher_is_better 声明的方向）
  rv_daily            RV（朴素基线：已有 idt_rv_daily 的直译，仅此处剔除午休收益）
  rv_skewness         -RS（负偏度=崩盘风险溢价，越高越好）
  jump_share          max(RV-BPV,0)/RV（越高越差）
  signed_jump         (RV+ - RV-)/sum(r~^2)，取值 [-1,1]（越高越好）
  vol_price_diffusion 日内 OLS 斜率 slope(|r| 水平 ~ 成交量份额)，无量纲（越高越差）
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("research.families.r3_intraday_moments_v1")

SOURCE = "stock_history_5min"
BARS = 48                 # 实测：每个 (股票, 交易日) 恰好 48 根 5min bar（2018-2026 全部年份）
LUNCH_ROW = 24            # 段内第 24 根 = 13:05 bar，其收益跨越 11:30 -> 13:05 午休
TRUNC_SIGMA = 6.0         # 截尾倍数（相对当日 BPV 尺度）
MIN_RETURNS = 40          # 当日可用 bar 收益下限（满日最多 46 个可用收益）
X_CAP = 8.0               # 量价回归：成交量份额截尾
Y_CAP = 8.0               # 量价回归：|收益| 相对水平截尾
SIGNALS = ("rv_daily", "rv_skewness", "jump_share", "signed_jump", "vol_price_diffusion")


def build(parameters):
    """Family 只允许预登记的五个信号标识；不搜索窗口、不搜索方向。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("R3-B 只允许预登记的五个信号标识")
    signal = parameters["signal"]

    def compute(ctx):
        return _year(ctx, signal)

    return compute


def _read_head(ctx, year, need_volume):
    """读本年度分区并做结构断言，返回 (frame, nseg, code_head, day_head, tail_rows)。"""
    columns = ["stock_code", "trade_time", "close"] + (["vol"] if need_volume else [])
    frame = ctx.dataset(SOURCE, columns=columns, years=(year, year))
    if frame is None or frame.empty:
        raise ValueError("stock_history_5min year=%d 无可读行" % year)
    rows_total = int(frame.shape[0])
    nseg = rows_total // BARS
    if nseg < 1:
        raise ValueError("stock_history_5min year=%d 行数 %d 不足一个股票日" % (year, rows_total))
    head = frame.iloc[: nseg * BARS : BARS]
    code_head = head["stock_code"].to_numpy()
    day_head = np.asarray(ctx.date_col(head["trade_time"].str.slice(0, 10)), dtype=np.int64)
    if not np.all(day_head > 0):
        raise ValueError("trade_time 不能全部解析为合法日期")
    same = code_head[1:] == code_head[:-1]
    if np.any(same & (np.diff(day_head) <= 0)):
        raise ValueError("同一股票的 48 行块内日期非严格递增："
                         "「48 行 = 一个股票日」的结构假设不成立，拒绝继续")
    if (1 + int(np.count_nonzero(~same))) != int(pd.unique(code_head).size):
        raise ValueError("同一股票在年分区中不是连续块：分区未按 (stock_code, trade_time) 排序")
    return frame, nseg, code_head, day_head, rows_total - nseg * BARS


def _volume_price_slope(vol, usable, absr, starts, full):
    """无量纲日内量价斜率：x=v/mean(v)、y=|r|/mean(|r|) 后做带截距 OLS。

    只用当日成交量份额，因此对 stock_history_5min.vol 在 2025-12-01 的手/股单位翻转
    严格不变（整体乘数在份额里约掉）。
    """
    vok = usable & np.isfinite(vol) & (vol > 0.0)
    n_v = np.add.reduceat(vok.astype(np.float64), starts)
    v_use = np.where(vok, vol, 0.0)
    a_use = np.where(vok, absr, 0.0)
    sum_v = np.add.reduceat(v_use, starts)
    sum_a = np.add.reduceat(a_use, starts)
    with np.errstate(all="ignore"):
        mean_v = sum_v / np.where(n_v > 0.0, n_v, np.nan)
        mean_a = sum_a / np.where(n_v > 0.0, n_v, np.nan)
    mean_v_row = np.repeat(mean_v, BARS)
    mean_a_row = np.repeat(mean_a, BARS)
    x = np.clip(np.where(vok, v_use / mean_v_row, 0.0), 0.0, X_CAP)
    y = np.clip(np.where(vok, a_use / mean_a_row, 0.0), 0.0, Y_CAP)
    del v_use, a_use, mean_v_row, mean_a_row
    s_x = np.add.reduceat(x, starts)
    s_y = np.add.reduceat(y, starts)
    s_xx = np.add.reduceat(x * x, starts)
    s_xy = np.add.reduceat(x * y, starts)
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
    frame, nseg, code_head, day_head, tail_rows = _read_head(ctx, year, need_volume)
    pos_head = np.asarray(ctx.code_index(code_head), dtype=np.int64)
    keep = np.flatnonzero(pos_head >= 0)
    if keep.size == 0:
        return ctx.panel.empty(dtype=np.float64)
    offsets = (keep[:, None] * BARS + np.arange(BARS, dtype=np.int64)[None, :]).ravel()
    close_all = frame["close"].to_numpy(np.float64)[: nseg * BARS]
    close = close_all[offsets]
    vol = None
    if need_volume:
        vol_all = frame["vol"].to_numpy(np.float64)[: nseg * BARS]
        vol = vol_all[offsets]
        del vol_all
    del close_all, offsets

    n_rows = int(close.size)
    prev = np.empty(n_rows, dtype=np.float64)
    prev[0] = np.nan
    prev[1:] = close[:-1]
    prev[::BARS] = np.nan
    with np.errstate(all="ignore"):
        ret = close / prev - 1.0
    usable = np.isfinite(ret) & (close > 0.0) & (prev > 0.0)
    ret = np.where(usable, ret, np.nan)
    ret[LUNCH_ROW::BARS] = np.nan
    usable = np.isfinite(ret)
    del prev, close
    r0 = np.where(usable, ret, 0.0)
    del ret

    starts = np.arange(int(keep.size), dtype=np.int64) * BARS
    n_used = np.add.reduceat(usable.astype(np.float64), starts)
    rv = np.add.reduceat(r0 * r0, starts)
    absr = np.abs(r0)
    pair = np.zeros(n_rows, dtype=np.float64)
    pair[1:] = np.where(usable[1:] & usable[:-1], absr[1:] * absr[:-1], 0.0)
    bpv = np.add.reduceat(pair, starts) * (np.pi / 2.0)
    del pair

    scale = np.sqrt(np.maximum(bpv / np.maximum(n_used, 1.0), 0.0))
    limit_row = np.repeat(TRUNC_SIGMA * scale, BARS)
    trunc = np.where(usable, np.clip(r0, -limit_row, limit_row), 0.0)
    del limit_row
    sq = trunc * trunc
    s2t = np.add.reduceat(sq, starts)
    s3t = np.add.reduceat(sq * trunc, starts)
    up2 = np.add.reduceat(np.where(r0 > 0.0, sq, 0.0), starts)
    dn2 = np.add.reduceat(np.where(r0 < 0.0, sq, 0.0), starts)
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
        value = _volume_price_slope(vol, usable, absr, starts, full)

    value = np.where(np.isfinite(value), value, np.nan)
    grid = ctx.panel.place(pos_head[keep], day_head[keep], value)
    grid = np.where(np.asarray(ctx.universe, dtype=bool), grid, np.nan)
    log.info("r3-intraday %s year=%d bars=%d segs=%d finite=%d panel_finite=%d tail_rows=%d",
             signal, year, n_rows, int(keep.size), int(np.isfinite(value).sum()),
             int(np.isfinite(grid).sum()), int(tail_rows))
    return np.ascontiguousarray(grid, dtype=np.float64)
