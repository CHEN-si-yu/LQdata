"""R3-C 筹码分布形态（CYQ distribution shape）—— 五个预登记候选（v2）。

★ v1 -> v2 的唯一差异（缺陷披露，v1 已按不可变原则保留在 r3_chipshape_v1.py）：
  v1 的 compute() 把**逐行**（(股票,日,档位) 三级）的 panel 日/列下标直接喂给
  Panel.place，而 place 要求三个入参同为**逐组**（(股票,日)）长度，于是第一个年块就抛
  IndexError（size of axis is 399348 but boolean axis is 41644688）。该缺陷由**冒烟运行**
  （2026-09 单月窗口）在计算任何 IC 之前捕获：v1 的 5 个候选从未产生评价、从未消耗试验预算。
  v2 只把返回的定位数组改成逐组（rows[starts] / cols[starts]），机制、公式、常数、
  窗口、方向全部逐字不变。

★ 数据语义：`stock_cyq_chips` 是**供应商估算**的持仓成本分布（逐价位 price/percent），
  不是真实逐账户持仓。本模块的任何字段都不得解释为真实投资者结构。

★ 为什么自建「对数相对价位直方图」，而不是直接用 `fea/chips.py` 的派生字段：
  1) 派生层的 `entropy` 按供应商**自己的档位数**归一化（H/ln(n_levels)），而供应商
     档位是**价格线性**的：实测同一只股票不同年份档位分辨率都会变（000001.SZ 2018 档宽
     0.1 元、2025 档宽 0.2 元；600519.SH 档宽 23.3 元）。该熵因此混入了「档位分辨率」，
     不是纯粹的形态分散度。
  2) 派生层没有「相对现价的局部密度」（斜率不对称）与「重心」字段，且 `mean` 是未复权价，
     跨日相减会被除权污染。
  本模块把每个股票日的整条档位表搬到**统一的无量纲坐标** x = ln(price / close_t)，
  用固定窗宽重采样成 80 格直方图 —— 所有股票、所有年份共用同一套格子。

★ 复权口径（Lead 指定必须论证）：`stock_cyq_chips.price` 是**未复权**价。本模块所有
  形态量都只用 `x = ln(price / close_raw_t)`。同一交易日内整条档位表与当日 close 共享
  同一个累计复权因子 a_t，于是
      ln(p_raw / close_raw) == ln((p_raw·a_t) / (close_raw·a_t)) == ln(p_hfq / close_hfq)
  即 x 本身就是**后复权口径**，跨日、跨年可比；重心迁移量因此不需要额外乘 adj_factor。
  诊断脚本用 `ctx.hfq("close") / ctx.px_raw("close")` 显式重建 a_t 做了等价性复核
  （残差见交付报告第 2 节），不是仅有口头论证。

★ 五个候选（四个机制 + 一个预登记窗口变体），方向由 Lead 事前固定，本模块不含任何反号逻辑：
    chip_entropy              归一化 Shannon 熵（分散度）           higher_is_better=False
    chip_multimodality        双峰分裂度 = 谷深 × 次峰侧质量占比     higher_is_better=False
    chip_slope_asymmetry      现价上方等对数带宽质量占比 A/(A+B)    higher_is_better=False
    chip_migration_speed      重心对数相对价位的 |5 日变化|          higher_is_better=True
    chip_migration_speed_w20  同机制 20 日窗口变体                   higher_is_better=True
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea import mathx as mx

SIGNALS = (
    "chip_entropy",
    "chip_multimodality",
    "chip_slope_asymmetry",
    "chip_migration_speed",
    "chip_migration_speed_w20",
)

# ---- 事前固定的形态表示常数（预登记；未按任何 IC / 收益结果调整） ----
LOG_HALF = 1.60            # 对数窗 [-1.60, +1.60]，对应现价的 [0.20x, 4.95x]
BIN_WIDTH = 0.040          # 对数窗宽 -> 恰 80 个等宽格子
SMOOTH_SIGMA = 1.5         # 多峰检测前的核平滑带宽（单位：格）
NEAR_BAND = 0.10           # 斜率不对称的等对数带宽（现价 ±10%）
MIN_LEVELS = 12            # 单股票日最少档位数
MIN_WINDOW_MASS = 0.90     # 落在窗内的归一化筹码质量下限
MIN_NEAR_MASS = 0.05       # 现价 ±10% 带内质量下限（斜率不对称专用）
MIGRATION_FAST = 5         # 主表达的迁移窗口（交易日）
MIGRATION_SLOW = 20        # 预登记窗口变体
N_BINS = int(round(2.0 * LOG_HALF / BIN_WIDTH))

_COLUMNS = ("trade_date", "stock_code", "price", "percent")


def _kernel(sigma=SMOOTH_SIGMA):
    half = int(np.ceil(4.0 * sigma))
    grid = np.arange(-half, half + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (grid / sigma) ** 2)
    return weights / weights.sum()


_KERNEL = _kernel()
_KERNEL_HALF = _KERNEL.size // 2
_CENTERS = -LOG_HALF + BIN_WIDTH * (np.arange(N_BINS, dtype=np.float64) + 0.5)
_ABOVE = (_CENTERS > 0.0) & (_CENTERS <= NEAR_BAND)
_BELOW = (_CENTERS < 0.0) & (_CENTERS >= -NEAR_BAND)
_LOG_BINS = float(np.log(N_BINS))


def _smooth_rows(hist):
    padded = np.pad(hist, ((0, 0), (_KERNEL_HALF, _KERNEL_HALF)))
    return np.apply_along_axis(
        lambda row: np.convolve(row, _KERNEL, mode="valid"), 1, padded)


def _shape_stats(hist):
    """hist:(G, N_BINS) 每行是该股票日的归一化筹码质量。

    返回 (entropy, multimodality, slope, centroid, near_mass, window_mass)。
    """
    groups = hist.shape[0]
    window_mass = hist.sum(axis=1)
    mass_ok = window_mass > 0.0
    normalised = np.divide(hist, window_mass[:, None], out=np.zeros_like(hist),
                           where=mass_ok[:, None])
    positive = normalised > 0.0
    safe = np.where(positive, normalised, 1.0)
    entropy = -(np.where(positive, normalised * np.log(safe), 0.0)).sum(axis=1) / _LOG_BINS
    centroid = (normalised * _CENTERS[None, :]).sum(axis=1)
    above = (normalised * _ABOVE[None, :]).sum(axis=1)
    below = (normalised * _BELOW[None, :]).sum(axis=1)
    near_mass = above + below
    slope = np.divide(above, near_mass, out=np.full(groups, np.nan),
                      where=near_mass > 0.0)

    smoothed = _smooth_rows(normalised)
    left = np.empty_like(smoothed)
    right = np.empty_like(smoothed)
    left[:, 1:] = smoothed[:, :-1]
    left[:, 0] = -1.0
    right[:, :-1] = smoothed[:, 1:]
    right[:, -1] = -1.0
    is_peak = (smoothed > left) & (smoothed > right)
    multimodality = np.zeros(groups, dtype=np.float64)
    for index in range(groups):
        peaks = np.flatnonzero(is_peak[index])
        if peaks.size < 2:
            continue
        ranked = peaks[np.argsort(-smoothed[index, peaks])]
        first, second = int(ranked[0]), int(ranked[1])
        low, high = (first, second) if first < second else (second, first)
        if high - low < 3:
            continue
        valley = low + int(np.argmin(smoothed[index, low:high + 1]))
        taller = max(min(smoothed[index, first], smoothed[index, second]), 1e-12)
        depth = 1.0 - smoothed[index, valley] / taller
        share = (float(normalised[index, valley + 1:].sum()) if second > first
                 else float(normalised[index, :valley].sum()))
        multimodality[index] = max(depth, 0.0) * share

    valid = mass_ok & (window_mass >= MIN_WINDOW_MASS)
    for array in (entropy, multimodality, slope, centroid):
        array[~valid] = np.nan
    slope = np.where(near_mass >= MIN_NEAR_MASS, slope, np.nan)
    return entropy, multimodality, slope, centroid, near_mass, window_mass


def _chip_shape(ctx):
    """逐价位表 -> 每 (股票日) 一行的对数相对价位直方图与形态量。"""
    panel = ctx.panel
    days = np.asarray(panel.dates, dtype=np.int64)
    codes = np.asarray(panel.codes, dtype=object)
    close = np.asarray(ctx.px_raw("close"), dtype=np.float64)
    year0 = int(str(int(days[0]))[:4])
    year1 = int(str(int(days[-1]))[:4])
    frame = ctx.dataset("stock_cyq_chips", columns=list(_COLUMNS),
                        years=(year0, year1))
    if frame.empty:
        return None
    price = pd.to_numeric(frame["price"], errors="coerce").to_numpy(np.float64)
    weight = pd.to_numeric(frame["percent"], errors="coerce").to_numpy(np.float64)
    code_id, code_unique = pd.factorize(frame["stock_code"])
    day_id, day_unique = pd.factorize(frame["trade_date"])
    del frame
    code_pos = pd.Series(np.arange(codes.size, dtype=np.int64),
                         index=codes).reindex(code_unique).to_numpy(dtype=np.float64)
    day_ints = np.array([int(t[0:4]) * 10000 + int(t[5:7]) * 100 + int(t[8:10])
                         for t in day_unique], dtype=np.int64)
    day_pos = pd.Series(np.arange(days.size, dtype=np.int64),
                        index=days).reindex(day_ints).to_numpy(dtype=np.float64)
    rows = day_pos[day_id]
    cols = code_pos[code_id]
    del code_id, day_id, code_unique, day_unique, day_pos, code_pos
    keep = (np.isfinite(rows) & np.isfinite(cols) & np.isfinite(price)
            & (price > 0.0) & np.isfinite(weight) & (weight >= 0.0))
    rows = rows[keep].astype(np.int64)
    cols = cols[keep].astype(np.int64)
    price = price[keep]
    weight = weight[keep]
    close_t = close[rows, cols]
    good = np.isfinite(close_t) & (close_t > 0.0)
    rows, cols = rows[good], cols[good]
    price, weight, close_t = price[good], weight[good], close_t[good]
    if rows.size == 0:
        return None

    key = cols * 100000 + rows
    order = np.argsort(key, kind="stable")
    key = key[order]
    rows, cols = rows[order], cols[order]
    price, weight, close_t = price[order], weight[order], close_t[order]
    starts = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
    ends = np.concatenate((starts[1:], [key.size]))
    groups = starts.size
    level_count = (ends - starts).astype(np.float64)
    group_id = np.repeat(np.arange(groups, dtype=np.int64), ends - starts)

    total = np.bincount(group_id, weights=weight, minlength=groups)
    norm_w = np.divide(weight, total[group_id], out=np.zeros_like(weight),
                       where=total[group_id] > 0.0)
    log_rel = np.log(price) - np.log(close_t)
    bins = np.clip(((log_rel + LOG_HALF) / BIN_WIDTH).astype(np.int64), 0, N_BINS - 1)
    hist = np.bincount(group_id * N_BINS + bins, weights=norm_w,
                       minlength=groups * N_BINS).reshape(groups, N_BINS)

    entropy, multimodality, slope, centroid, near_mass, window_mass = _shape_stats(hist)
    usable = (level_count >= MIN_LEVELS) & (total > 0.0) & np.isfinite(entropy)
    entropy = np.where(usable, entropy, np.nan)
    multimodality = np.where(usable, multimodality, np.nan)
    slope = np.where(usable, slope, np.nan)
    centroid = np.where(usable, centroid, np.nan)
    # 每组一行：place() 要求 code/day/value 三个入参同为逐组长度。
    return {"group_rows": rows[starts], "group_cols": cols[starts],
            "entropy": entropy, "multimodality": multimodality, "slope": slope,
            "centroid": centroid, "level_count": level_count,
            "window_mass": window_mass, "near_mass": near_mass,
            "groups": groups, "usable": usable}


def build(parameters):
    """五个预登记机制之一；不接受窗口/阈值搜索。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("本 Family 只允许五个预登记的 signal 标识符")
    signal = parameters["signal"]

    def compute(ctx):
        shape = _chip_shape(ctx)
        if shape is None:
            return ctx.panel.empty(dtype=np.float64)
        panel = ctx.panel
        days = np.asarray(panel.dates, dtype=np.int64)
        cols = shape["group_cols"]
        dates = days[shape["group_rows"]]
        if signal == "chip_entropy":
            value = panel.place(cols, dates, shape["entropy"])
        elif signal == "chip_multimodality":
            value = panel.place(cols, dates, shape["multimodality"])
        elif signal == "chip_slope_asymmetry":
            value = panel.place(cols, dates, shape["slope"])
        else:
            lag = MIGRATION_FAST if signal == "chip_migration_speed" else MIGRATION_SLOW
            grid = panel.place(cols, dates, shape["centroid"])
            prior = mx.shift(grid, lag)
            value = np.abs(grid - prior)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute