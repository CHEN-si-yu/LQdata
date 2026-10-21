"""R2-I3 股东户数与户均持股结构：四个事前固定机制 + 两个预登记窗口变体。

数据：stock_holder_number（stock_code, ann_date, end_date, holder_num）
加 stock_finance（circ_mv / float_share，日频）做户均市值与截面中性化。

====================== PIT 合同（唯一可用时点 = ann_date） ======================
  · ann_date（公告日）是本模块唯一的可用时点。end_date（报告期）只用来判断
    "哪一期更新"，从不作为可见日期，也不参与任何 as-of 键。
  · 同一个 ann_date 上的多条记录（实测同日可披露 1~19 个报告期）在该日全部生效，
    之后才取状态 —— 同日多期不会被拆成多个不同时点。
  · 同一个 end_date 被后续公告重述时，新值只从它自己的 ann_date 起生效
    （状态机逐日推进，历史生效日不会被新公告改写）。
  · 上游审计按 ann_date 截断（fea/upstream.py::_audit_columns -> announcement）。
  · 无记录 != 未发生：披露后状态一直有效，用 as-of 前向沿用；从未披露过 -> NaN。

============================ 缺失 / 无效 / 零的判定 ============================
  · holder_num <= 0 视为无效。池内实测 8222 行取值为 0（占 4.4%），且 (0,100)
    区间内没有任何取值 -> 阈值取 > 0 即可，不需要额外的下界猜测。
  · 相邻报告期 end_date 间隔必须落在 (0, 200] **真实日历天**，否则该"变化率"置 NaN。
    这与生产因子 holder_number_chg 的间隔保护同口径：1~12 月都有披露，
    间隔过长等于拿两个不同年份的观察值相比。
  · 连续两次状态变更公告的间隔必须落在 (0, 400] 日历天，否则该变化置 NaN。
  · circ_mv 缺失或非正 -> NaN；停牌日的成交额/收益不被前向填充（价格层 FLOW 语义），
    所以依赖真实成交的机制在停牌日自然为 NaN。
  · 完整输入但机制给出 0 是观测到的 0（例如户数不变时一期变化率恰为 0），保留。

方向：六个候选全部 higher_is_better = true（Lead 事前固定，未按结果翻转）。

============================== v2 修订（实现缺陷修复） ==============================
v1 用 YYYYMMDD 整数相减当作"天数"（20260630 - 20260331 = 299），于是 200 天的报告期
间隔保护把**正常的季度配对**全部拒掉，M2/M4 的有效覆盖被压到约 28%。v2 改成真实的
日历天差（与生产因子 holder_number_chg 的 pd.to_datetime(...).dt.days 同口径）。
本次修订**不改变机制集合、窗口、方向与任何阈值**（PERIOD_GAP_MAX 仍是 200 天，
EVENT_GAP_MAX 仍是 400 天），只把单位算对；修订发生在任何 IC/收益评价之前。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = (
    "per_capita_value_level",
    "per_capita_value_change",
    "trend_persistence_k4",
    "announcement_drift_d5",
    "announcement_drift_d10",
    "dispersion_residual",
)
PERIOD_COUNT = 4          # M2：最近 4 个报告期 = 3 个连续变化
PERIOD_GAP_MAX = 200      # 相邻报告期 end_date 的最大间隔（日历天）
EVENT_GAP_MAX = 400       # 连续两次状态变更公告的最大间隔（日历天）
DRIFT_DAYS = {"announcement_drift_d5": 5, "announcement_drift_d10": 10}
MIN_CROSS = 100           # 截面回归/中性的最少有效票数（= 协议 min_cross_section）


def _divide(a, b, floor=1e-12):
    with np.errstate(all="ignore"):
        result = np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64)
    return np.where(np.isfinite(result) & (np.abs(b) > floor), result, np.nan)


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


def _pool_relative(ctx, values):
    """用当日固定池（不是未来标签资格）做截面去均值。"""
    valid = np.asarray(ctx.universe, dtype=bool) & np.isfinite(values)
    count = valid.sum(axis=1)
    minimum = max(3, int(ctx.cfg.min_cross_section))
    mean = _divide(np.where(valid, values, 0.).sum(axis=1), count, floor=0)
    mean[count < minimum] = np.nan
    return np.where(valid, values - mean[:, None], np.nan)


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


def _circ_mv(ctx):
    """stock_finance 的日频 circ_mv -> (T, C) 网格。缺失/非正为 NaN。"""
    days = ctx.panel.dates
    frame = ctx.dataset("stock_finance", columns=["stock_code", "trade_date", "circ_mv"],
                        years=(int(days[0]) // 10000, int(days[-1]) // 10000))
    if frame is None or frame.empty:
        return ctx.panel.empty(dtype=np.float64)
    frame = frame[frame["stock_code"].isin(ctx.panel.codes)]
    value = pd.to_numeric(frame["circ_mv"], errors="coerce").to_numpy(dtype=np.float64)
    value = np.where(np.isfinite(value) & (value > 0), value, np.nan)
    grid = ctx.panel.place(ctx.code_index(frame["stock_code"].to_numpy()),
                           np.asarray(ctx.date_col(frame["trade_date"]), dtype=np.int64),
                           value)
    return np.where(np.isfinite(grid) & (grid > 0), grid, np.nan)


def _state_asof(ctx, event, values):
    return ctx.asof_daily(event["state_codes"],
                          np.asarray(event["state_days"], dtype=np.int64),
                          np.asarray(values, dtype=np.float64))


def _per_capita_at_event(ctx, event, circ_mv):
    """返回 (当日户均市值对数网格, 逐状态事件公告日的户均市值对数)。"""
    log_mv = np.log(np.where(circ_mv > 0, circ_mv, np.nan))
    log_holder = np.log(event["period_num"][:, 0])
    holder_asof = _state_asof(ctx, event, log_holder)
    level = log_mv - holder_asof
    days = np.asarray(event["state_days"], dtype=np.int64)
    pos = np.clip(np.searchsorted(ctx.panel.dates, days), 0, ctx.panel.T - 1)
    col = np.asarray(ctx.code_index(event["state_codes"]), dtype=np.int64)
    on_grid = (ctx.panel.dates[pos] == days) & (col >= 0)
    at_event = np.where(on_grid, level[pos, np.clip(col, 0, None)], np.nan)
    return level, at_event


def _sequential_change(event, at_event):
    """把逐事件值变成"相对上一次状态变更公告"的变化；间隔越界置 NaN。"""
    n = at_event.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    days = np.asarray(event["state_days"], dtype=np.int64)
    codes = np.asarray(event["state_codes"], dtype=object)
    prev = np.full(n, np.nan, dtype=np.float64)
    prev_day = np.zeros(n, dtype=np.int64)
    if n > 1:
        prev[1:] = at_event[:-1]
        prev_day[1:] = days[:-1]
    first = np.zeros(n, dtype=bool)
    first[0] = True
    if n > 1:
        first[1:] = codes[1:] != codes[:-1]
    prev[first] = np.nan
    gap = _day_numbers(days) - _day_numbers(prev_day)
    ok = (~first) & np.isfinite(prev) & np.isfinite(at_event) & (gap > 0) & (gap <= EVENT_GAP_MAX)
    return np.where(ok, at_event - prev, np.nan)


def _period_change(event):
    """一期户数变化 log(n1/n2)，含间隔保护。返回 (变化, 有效掩码)。"""
    end, num = event["period_end"], event["period_num"]
    gap = (_day_numbers(end)[:, 0] - _day_numbers(end)[:, 1])
    ratio = _divide(num[:, 0], num[:, 1], floor=0)
    ok = (gap > 0) & (gap <= PERIOD_GAP_MAX) & np.isfinite(ratio) & (ratio > 0)
    with np.errstate(all="ignore"):
        change = np.where(ok, np.log(ratio), np.nan)
    return change, ok


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


def _dispersion_residual(change, circ_mv, turnover, mask):
    """M4：户数变化对 log 市值 与 log 换手的逐日截面 OLS 残差（只用当日截面）。"""
    with np.errstate(all="ignore"):
        control_mv = np.where(np.isfinite(circ_mv) & (circ_mv > 0), np.log(circ_mv), np.nan)
        control_to = np.where(np.isfinite(turnover) & (turnover > 0), np.log(turnover), np.nan)
    T, C = change.shape
    out = np.full((T, C), np.nan, dtype=np.float64)
    for t in range(T):
        row = mask[t]
        if int(row.sum()) < MIN_CROSS:
            continue
        idx = np.flatnonzero(row)
        y = change[t, idx]
        z1 = control_mv[t, idx]
        z2 = control_to[t, idx]
        good = np.isfinite(y) & np.isfinite(z1) & np.isfinite(z2)
        if int(good.sum()) < MIN_CROSS:
            continue
        design = np.column_stack([np.ones(int(good.sum())), z1[good], z2[good]])
        target = y[good]
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        out[t, idx[good]] = target - design @ coef
    return out


def _drift(ctx, event, days_after):
    """M3：公告日之后的池内相对漂移（T 日只用 <= T 的收盘价）。"""
    panel = ctx.panel
    days = np.asarray(event["all_days"], dtype=np.int64)
    codes = np.asarray(event["all_codes"], dtype=object)
    if days.size == 0 or panel.T == 0:
        return panel.empty(dtype=np.float64)
    pos = np.searchsorted(panel.dates, days)
    # 面板窗口之前发生的公告不参与：它们的生效日会被硬压到窗口首日，
    # 从而在热身区制造一段假的"公告后漂移"。
    inside = (pos < panel.T) & (days >= int(panel.dates[0]))
    pos = np.clip(pos, 0, panel.T - 1)
    eff = panel.dates[pos]
    col = np.asarray(ctx.code_index(codes), dtype=np.int64)
    keep = inside & (col >= 0)
    eff, col, pos = eff[keep], col[keep], pos[keep]
    if eff.size == 0:
        return panel.empty(dtype=np.float64)
    # 同一 (股票, 生效日) 只留一条，避免 asof 的重复键歧义
    key = col.astype(np.int64) * 100000000 + eff.astype(np.int64)
    _, unique = np.unique(key, return_index=True)
    unique = np.sort(unique)
    eff, col, pos = eff[unique], col[unique], pos[unique]
    close = np.asarray(ctx.hfq("close"), dtype=np.float64)
    anchor = ctx.asof_daily(codes[keep][unique], eff, close[pos, col])
    relative = _pool_relative(ctx, _divide(close, anchor) - 1.)
    grid = ctx.event_grid(codes[keep][unique], eff)
    index = np.arange(panel.T, dtype=np.int64)[:, None]
    last = np.maximum.accumulate(np.where(grid > 0, index, -1), axis=0)
    since = np.where(last >= 0, index - last, -1)
    valid = (since >= 1) & (since <= days_after)
    return np.where(valid, relative, np.nan)


def build(parameters):
    """R2-I3 的六个事前固定机制；不搜索窗口、不按结果翻转方向。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("R2-I3 只登记六个事前固定的机制标识")
    signal = parameters["signal"]

    def compute(ctx):
        event = _holder_events(ctx)
        universe = np.asarray(ctx.universe, dtype=bool)
        if signal in DRIFT_DAYS:
            value = _drift(ctx, event, DRIFT_DAYS[signal])
            traded = np.asarray(ctx.traded(), dtype=bool)
            value = np.where(universe & traded, value, np.nan)
            return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)
        if event["state_days"].size == 0:
            return ctx.panel.empty(dtype=np.float64)
        if signal == "trend_persistence_k4":
            value = _state_asof(ctx, event, _persistence(event))
            value = np.where(universe, value, np.nan)
            return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)
        circ_mv = _circ_mv(ctx)
        if signal == "per_capita_value_level":
            level, _ = _per_capita_at_event(ctx, event, circ_mv)
            value = np.where(universe, level, np.nan)
            return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)
        if signal == "per_capita_value_change":
            _, at_event = _per_capita_at_event(ctx, event, circ_mv)
            value = _state_asof(ctx, event, _sequential_change(event, at_event))
            value = np.where(universe, value, np.nan)
            return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)
        change, ok = _period_change(event)
        change = np.where(ok, -change, np.nan)          # 户数下降 = 正
        grid = _state_asof(ctx, event, change)
        turnover = np.asarray(ctx.px("turnover"), dtype=np.float64)
        value = _dispersion_residual(grid, circ_mv, turnover, universe & np.isfinite(grid))
        value = np.where(universe, value, np.nan)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute