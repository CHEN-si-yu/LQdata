"""保留迁移（清理复审，2026-10-03）：仅 trend_persistence_k4。

来源：research/families/r2_ownership_v2.py（FILE-SHA 封存，不改动原件）。被拒候选的
实现不随迁；保留实现的函数体逐字复制，参数与方向不变。新身份须重新走原生 G1 与
旧生产值精确 parity 后才批准。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


SIGNALS = ("trend_persistence_k4",)
KEPT_SIGNALS = SIGNALS

PERIOD_COUNT = 4          # M2：最近 4 个报告期 = 3 个连续变化


PERIOD_GAP_MAX = 200      # 相邻报告期 end_date 的最大间隔（日历天）


def _day_numbers(values):
    """YYYYMMDD 整数 -> 真实日历天序号（差值即天数，不是 YYYYMMDD 差值）。

    v1 的缺陷就在这里：20260630 - 20260331 = 299，看起来"没有超过 200"以外的含义，
    实际区间只有 91 天。用日期序号做差才与生产因子 holder_number_chg 的
    pd.to_datetime(...).dt.days 同口径。
    """
    flat = np.asarray(values, dtype=np.int64).ravel()
    parsed = pd.to_datetime(pd.Series(flat).astype(str), format="%Y%m%d", errors="coerce")
    days = (parsed - pd.Timestamp("1970-01-01")).dt.days.to_numpy(dtype=np.float64)
    return days.reshape(np.asarray(values).shape)


def _holder_events(ctx):
    """按 ann_date 逐日推进的状态机；返回逐公告日事件表。

    两组事件：
      all_*   —— 每个不同的 (stock_code, ann_date)（M3 的披露事件，同日多期算一次）
      state_* —— 只有"最近 PERIOD_COUNT 期状态"真正变化的那天才发一行
                 （M1/M2/M4 用它，避免同一状态的重复公告制造人造 0 变化）
    """
    frame = ctx.dataset("stock_holder_number",
                        columns=["stock_code", "ann_date", "end_date", "holder_num"],
                        years=None)
    blank = {
        "all_codes": np.array([], dtype=object), "all_days": np.zeros(0, dtype=np.int64),
        "state_codes": np.array([], dtype=object), "state_days": np.zeros(0, dtype=np.int64),
        "period_end": np.zeros((0, PERIOD_COUNT), dtype=np.int64),
        "period_num": np.zeros((0, PERIOD_COUNT), dtype=np.float64),
        "invalid_rows": 0,
    }
    if frame is None or frame.empty:
        return blank
    code = frame["stock_code"].astype(str).to_numpy()
    ann = np.asarray(ctx.date_col(frame["ann_date"]), dtype=np.int64)
    end = np.asarray(ctx.date_col(frame["end_date"]), dtype=np.int64)
    num = pd.to_numeric(frame["holder_num"], errors="coerce").to_numpy(dtype=np.float64)
    keep = (num > 0) & (ann >= 19900101) & (end >= 19900101)
    blank["invalid_rows"] = int((~keep).sum())
    if not keep.any():
        return blank
    code, ann, end, num = code[keep], ann[keep], end[keep], num[keep]
    _, code_id = np.unique(code, return_inverse=True)
    order = np.lexsort((end, ann, code_id))
    code, ann, end, num = code[order], ann[order], end[order], num[order]
    # (code, ann) 分组：同一天披露的多个报告期一次性生效
    boundary = np.empty(code.size, dtype=bool)
    boundary[0] = True
    if code.size > 1:
        boundary[1:] = (code[1:] != code[:-1]) | (ann[1:] != ann[:-1])
    starts = np.flatnonzero(boundary)
    stops = np.append(starts[1:], code.size)

    all_codes, all_days = [], []
    state_codes, state_days, period_end, period_num = [], [], [], []
    known = {}
    signature = None
    current = None
    for start, stop in zip(starts, stops):
        group_code = code[start]
        if group_code != current:
            current, known, signature = group_code, {}, None
        for j in range(start, stop):
            known[int(end[j])] = float(num[j])            # 同日多条全部生效
        all_codes.append(group_code)
        all_days.append(int(ann[start]))
        top = sorted(known, reverse=True)[:PERIOD_COUNT]
        newest = tuple((int(e), float(known[e])) for e in top)
        if newest == signature:
            continue
        signature = newest
        state_codes.append(group_code)
        state_days.append(int(ann[start]))
        row_end = np.zeros(PERIOD_COUNT, dtype=np.int64)
        row_num = np.full(PERIOD_COUNT, np.nan, dtype=np.float64)
        for k, e in enumerate(top):
            row_end[k] = int(e)
            row_num[k] = float(known[e])
        period_end.append(row_end)
        period_num.append(row_num)
    return {
        "all_codes": np.asarray(all_codes, dtype=object),
        "all_days": np.asarray(all_days, dtype=np.int64),
        "state_codes": np.asarray(state_codes, dtype=object),
        "state_days": np.asarray(state_days, dtype=np.int64),
        "period_end": (np.stack(period_end) if period_end
                       else np.zeros((0, PERIOD_COUNT), dtype=np.int64)),
        "period_num": (np.stack(period_num) if period_num
                       else np.zeros((0, PERIOD_COUNT), dtype=np.float64)),
        "invalid_rows": 0,
    }


def _state_asof(ctx, event, values):
    return ctx.asof_daily(event["state_codes"],
                          np.asarray(event["state_days"], dtype=np.int64),
                          np.asarray(values, dtype=np.float64))


def _persistence(event):
    """M2：最近 PERIOD_COUNT 期的"下降持续性 x 趋势强度"。

    对 j = 1..K-1：g_j = log(n_j) - log(n_{j+1})（> 0 表示户数增加）。
    value = mean_j(-g_j) * (下降期数 / (K-1))；任一间隔越界或期数不足 -> NaN。
    """
    end, num = event["period_end"], event["period_num"]
    k = PERIOD_COUNT
    gaps = _day_numbers(end)[:, :k - 1] - _day_numbers(end)[:, 1:k]
    with np.errstate(all="ignore"):
        growth = np.log(num[:, :k - 1]) - np.log(num[:, 1:k])
    ok = np.isfinite(growth) & (gaps > 0) & (gaps <= PERIOD_GAP_MAX)
    complete = ok.all(axis=1)
    decline = growth < 0
    strength = np.where(complete, -growth.mean(axis=1), np.nan)
    share = np.where(complete, decline.mean(axis=1), np.nan)
    return strength * share


def build(parameters):
    """保留迁移：仅 trend_persistence_k4（参数与方向与原登记一致）。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in KEPT_SIGNALS:
        raise ValueError("retained family permits only the preserved preregistered signal")
    signal = parameters["signal"]

    def compute(ctx):
        event = _holder_events(ctx)
        universe = np.asarray(ctx.universe, dtype=bool)
        if event["state_days"].size == 0:
            return ctx.panel.empty(dtype=np.float64)
        value = _state_asof(ctx, event, _persistence(event))
        value = np.where(universe, value, np.nan)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
