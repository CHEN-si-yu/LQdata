"""R4-A · U1 — 盈余惊喜与公告后漂移（PEAD）。

五个预登记机制 + 两个预登记变体（方向与机制集合由
`artifacts/research/round4_20261004/RESEARCH_BRIEF_B1.md` §4-U1 事前冻结，执行时不得更改）：

  1. sue_seasonal_q            单季归母净利同比惊喜的标准化：d0 / std(过去 8 个同类同比差, min 6)。方向 true。
  2. sue_accel                 本期 SUE − 上一期 SUE（惊喜加速度）。方向 true。
  3. pead_drift_60             最近一次定期报告公告后 0–59 个交易日的累计池内超额收益。方向 true。
  4. forecast_implied_surprise 预告净利中点相对去年同期 last_parent_net 的惊喜（arcsinh 压缩）。方向 true。
  5. forecast_landing_gap      实际定期报告净利相对同报告期预告中点的落地偏差（arcsinh 压缩）。方向 true。

变体（登记前冻结）：
  A. pead_drift_60 + skip=5    跳过公告后前 5 个交易日，累计区间第 5–59 个交易日（隔离即时反应）。方向 true。
  B. sue_seasonal_nostd        机制 1 去掉个股 σ 标准化（直接给 d0，截面标准化交给评价层）。方向 true。

──────────────────────────────────────────────────────────────────────────────
数据实测事实（2026-10-04，脚本 delivery/r4a_earnings_surprise_v1/tools/probe_data.py、
probe_pairs.py，全池口径）：

* `stock_income`（归母净利，累计 YTD 制）：
  - 全表 281,265 行；池内 129,447 行；end_date 全部是季末（实测脏日期 0 行）；
  - ann_date / f_ann_date 当前版本**没有** <1990 哨兵（0 行；1970 哨兵在
    `stock_financial_indicator`，防御性过滤仍保留）；f_ann_date < ann_date 5 行；
  - 池内同一 (stock_code, end_date) 只有 4 组存在 >1 个公告时点版本（追溯修正极罕见）；
  - 2017 起每年 ≥2113 只池内股票有报告公告；2018-01-01 前 ≥9 个报告期的股票 2055/2115。
* `stock_forecast`：
  - 全表 110,918 行；池内 52,245 行；ann_date 无哨兵；first_ann_date 有 80 行哨兵（本模块不用该字段）；
  - `net_profit_min/max`、`last_parent_net` 单位是**万元**；`stock_income.n_income_attr_p`
    单位是**元** —— 落地偏差必须先 ×1e4 换算（实测 actual/(mid×1e4) 中位数 = 1.001）；
  - last_parent_net ≤ 0 占池内 16%（全表 20.5%）；实测厂商对负基数的 p_change 就是
    `(mid-base)/|base|`（中位差 2.8e-5），本模块沿用同一口径，避免丢样；
  - 预告事件每年覆盖约 1,200–1,600 只池内股票（池的 60–68%）；同一 (code, end_date)
    的多版本（修订）占 7.8%，同一天多报告期 81 组。
* PIT 合同（与 `fea/deriv.py::_load_one` 和 `research/families/r2_disclosure_v1.py` 一致）：
  - 财报行自 `pit = max(ann_date, f_ann_date, end_date)` 起可见，ann/fann 必须 ≥1990-01-01；
  - 预告行自 `ann_date` 起可见，ann_date 必须 ≥1990-01-01；
  - 单季同比差不用供应商单季字段、也不做日频差分，而是用财务层三行式 TTM 的恒等式
    `d_j = lag_ttm(NP, j) − lag_ttm(NP, j+1) = Q(p−j) − Q(p−j−4)`（对含跨年在内的所有季度成立）；
  - 缺预告 ≠ 预告为零：forecast 缺行/字段缺失一律 NaN，绝不零填充。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea import mathx as mx

NP = "n_income_attr_p"

SIGNALS = (
    "sue_seasonal_q",
    "sue_accel",
    "pead_drift_60",
    "forecast_implied_surprise",
    "forecast_landing_gap",
    "sue_seasonal_nostd",
)
SUE_PAST_COUNT = 8      # 分母窗口：过去 8 个同类同比差
SUE_MIN_PAST = 6        # 其中至少 6 个非 NaN 才出值
SUE_DEPTH = 10          # 取 lag_ttm 0..10，供 d_0..d_9（accel 需要 d_1 及其分母窗口）
DRIFT_DAYS = 60         # 公告后第 0..59 个交易日（= 60 个交易日）
DRIFT_SKIP = 5          # 变体 A：跳过前 5 个交易日
WAN_TO_YUAN = 1e4       # 预告万元 → 元（与 stock_income 对齐；实测校验过）
INCOME_COLS = ["stock_code", "ann_date", "f_ann_date", "end_date"]
FORECAST_COLS = ["stock_code", "ann_date", "end_date",
                 "net_profit_min", "net_profit_max", "last_parent_net"]


def _divide(a, b, floor=1e-12):
    with np.errstate(all="ignore"):
        result = np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64)
    return np.where(np.isfinite(result) & (np.abs(b) > floor), result, np.nan)


# ══════════════════════════════════════════════════════════════════════
# 取数与 PIT
# ══════════════════════════════════════════════════════════════════════

def _pool_frame(ctx, name, columns):
    frame = ctx.dataset(name, columns=list(columns))
    if frame is None or frame.empty:
        return None
    frame = frame[frame["stock_code"].isin(ctx.panel.codes)]
    if frame.empty:
        return None
    return frame


def _income_rows(ctx, with_value=False):
    """定期报告公告行 → (stock_code, end_i, pit[, value])；可用时点 = max(ann, fann, end)。"""
    frame = _pool_frame(ctx, "stock_income", INCOME_COLS + ([NP] if with_value else []))
    if frame is None:
        return None
    ann = ctx.date_col(frame["ann_date"]).astype(np.int64)
    fann = ctx.date_col(frame["f_ann_date"]).astype(np.int64)
    end = ctx.date_col(frame["end_date"]).astype(np.int64)
    keep = ((end % 10000).isin([331, 630, 930, 1231])) & ((ann >= 19900101) | (fann >= 19900101))
    if not keep.any():
        return None
    pit = np.maximum(np.maximum(ann, fann), end)[keep]
    out = pd.DataFrame({"stock_code": frame["stock_code"].to_numpy()[keep],
                        "end_i": end[keep], "pit": pit})
    if with_value:
        out["value"] = pd.to_numeric(frame[NP], errors="coerce").to_numpy(dtype=np.float64)[keep]
    return out


def _forecast_rows(ctx):
    """业绩预告公告行 → (stock_code, ann, end_i, mid, base)，金额单位 = 万元（供应商原值）。"""
    frame = _pool_frame(ctx, "stock_forecast", FORECAST_COLS)
    if frame is None:
        return None
    ann = ctx.date_col(frame["ann_date"]).astype(np.int64)
    keep = ann >= 19900101
    if not keep.any():
        return None
    lo = pd.to_numeric(frame["net_profit_min"], errors="coerce").to_numpy(dtype=np.float64)
    hi = pd.to_numeric(frame["net_profit_max"], errors="coerce").to_numpy(dtype=np.float64)
    base = pd.to_numeric(frame["last_parent_net"], errors="coerce").to_numpy(dtype=np.float64)
    return pd.DataFrame({"stock_code": frame["stock_code"].to_numpy()[keep],
                         "ann": ann[keep],
                         "end_i": ctx.date_col(frame["end_date"]).astype(np.int64)[keep],
                         "mid": ((lo + hi) / 2.0)[keep], "base": base[keep]})


# ══════════════════════════════════════════════════════════════════════
# 机制 1/2：单季同比惊喜
# ══════════════════════════════════════════════════════════════════════

def _seasonal_diffs(ctx, depth=SUE_DEPTH):
    """d_j = Q(p−j) − Q(p−j−4)：单季归母净利的同类同比差（j 个报告期前）。

    恒等式（三行式 TTM，含跨年季度成立）：
        lag_ttm(NP, j) − lag_ttm(NP, j+1) = Q(p−j) − Q(p−j−4)。
    取数全部走财务层（ctx.lag_ttm），PIT / 追溯修正 / 哨兵过滤由 fea/deriv.py 统一处理。
    """
    levels = [np.asarray(ctx.lag_ttm(NP, k), dtype=np.float64) for k in range(depth + 1)]
    return [levels[j] - levels[j + 1] for j in range(depth)]


def _sue_at(diffs, current, past=SUE_PAST_COUNT):
    """d_current / std(d_{current+1..current+past})，要求 ≥SUE_MIN_PAST 个有效历史差。

    分母是**严格过去**的同类差（不含当期，避免自指）；标准差用总体口径（ddof=0，
    与 fea/mathx 的 roll_std 默认一致）。sd == 0（恒定的同比差序列）→ NaN，不做编造。
    """
    num = diffs[current]
    window = np.stack(diffs[current + 1: current + 1 + past], axis=0)
    mask = np.isfinite(window)
    count = mask.sum(axis=0)
    filled = np.where(mask, window, 0.0)
    mean = filled.sum(axis=0) / np.maximum(count, 1)
    dev = np.where(mask, window - mean, 0.0)
    sd = np.sqrt((dev * dev).sum(axis=0) / np.maximum(count, 1))
    ok = (count >= SUE_MIN_PAST) & np.isfinite(num) & (sd > 0)
    with np.errstate(all="ignore"):
        value = num / np.where(sd > 0, sd, np.nan)
    return np.where(ok, value, np.nan)


def _sue(ctx, standardized=True):
    diffs = _seasonal_diffs(ctx)
    return _sue_at(diffs, 0) if standardized else diffs[0]


def _sue_accel(ctx):
    diffs = _seasonal_diffs(ctx)
    return _sue_at(diffs, 0) - _sue_at(diffs, 1)


# ══════════════════════════════════════════════════════════════════════
# 机制 3：公告后漂移
# ══════════════════════════════════════════════════════════════════════

def _pool_relative_returns(ctx):
    """日频后复权收益 − 池内已成交股票的截面均值。

    停牌日按项目口径贡献 0 收益（hfq 收盘价前向填充；`ctx.ret` 的既定语义），
    均值只用已成交股票计算。未上市/无价格 → NaN（不伪造成交）。
    """
    returns = np.asarray(ctx.ret(1), dtype=np.float64)
    listed = np.asarray(ctx.universe, dtype=bool)
    traded = np.asarray(ctx.traded(), dtype=bool)
    valid = listed & traded & np.isfinite(returns)
    count = valid.sum(axis=1)
    minimum = max(3, int(ctx.cfg.min_cross_section))
    mean = _divide(np.where(valid, returns, 0.0).sum(axis=1), count, floor=0)
    mean[count < minimum] = np.nan
    return np.where(listed & np.isfinite(returns), returns - mean[:, None], np.nan)


def _pead_drift(ctx, skip=0):
    """最近一次定期报告公告后的累计池内超额收益（公告后第 skip..59 个交易日）。

    锚点 = 公告可得日 pit 的**下一个交易日** t0（公告通常盘后发布，保守从次日起算）；
    t0 必须是本面板内的交易日（窗口仅 60 个交易日，warmup 700 日远大于它，
    面板起点前的锚点不可能落入输出窗口）。同一公告日多报告期时取最新报告期。
    累计区间：`s = t − t0`，s ∈ [skip, 59]；超出 → NaN，由下一次公告重置。
    """
    rows = _income_rows(ctx, with_value=False)
    if rows is None:
        return None
    rows = rows.sort_values(["stock_code", "end_i", "pit"], kind="mergesort")
    rows = rows.drop_duplicates(["stock_code", "pit"], keep="last")
    t0 = np.asarray(ctx.next_trading_day(rows["pit"].to_numpy(dtype=np.int64)), dtype=np.int64)
    days = ctx.panel.dates
    T = days.size
    pos = np.searchsorted(days, t0)
    inside = (pos < T) & (days[np.clip(pos, 0, T - 1)] == t0)
    if not inside.any():
        return ctx.panel.empty(dtype=np.float64)
    codes = rows["stock_code"].to_numpy()[inside]
    anchors = pos[inside].astype(np.float64)
    grid = ctx.asof_daily(codes, t0[inside], anchors)
    anchor = np.asarray(grid, dtype=np.float64)
    since = np.arange(T, dtype=np.float64)[:, None] - anchor
    relative = _pool_relative_returns(ctx)
    total = np.zeros((T, ctx.panel.C), dtype=np.float64)
    for k in range(DRIFT_DAYS):
        usable = since >= (k + skip)
        total += np.where(usable, mx.shift(relative, k), 0.0)
    in_window = np.isfinite(anchor) & (since >= skip) & (since <= DRIFT_DAYS - 1)
    return np.where(in_window, total, np.nan)


# ══════════════════════════════════════════════════════════════════════
# 机制 4/5：预告惊喜与落地偏差
# ══════════════════════════════════════════════════════════════════════

def _compressed(rel):
    """arcsinh 压缩：对 |rel| 小值近似线性、大值对数化；单调、无未来信息。"""
    return np.arcsinh(rel)


def _implied_surprise(ctx):
    """预告净利中点相对去年同期的惊喜：arcsinh((mid−base)/|base|)。

    口径说明：分母用 |base|（实测与厂商对负基数 p_change 的口径一致，见模块 docstring），
    base == 0 → NaN。取「最近一次预告公告」的版本（同日多报告期取较新报告期），
    最新版本字段缺失时值为 NaN（不退回旧版本）。缺预告 → NaN（不零填充）。
    """
    rows = _forecast_rows(ctx)
    if rows is None:
        return None
    rel = np.full(len(rows), np.nan, dtype=np.float64)
    mid = rows["mid"].to_numpy(dtype=np.float64)
    base = rows["base"].to_numpy(dtype=np.float64)
    ok = np.isfinite(mid) & np.isfinite(base) & (base != 0)
    rel[ok] = (mid[ok] - base[ok]) / np.abs(base[ok])
    value = np.where(ok, _compressed(rel), np.nan)
    return ctx.asof_daily(rows["stock_code"].to_numpy(), rows["ann"].to_numpy(dtype=np.int64), value)


def _landing_gap(ctx):
    """实际定期报告 vs 同报告期预告中点的落地偏差：arcsinh((actual−mid)/|mid|)。

    - actual = 该 (code, end_date) **首次披露**版本（追溯修正不改变落地日的信息）；
      单位从元换算到与预告一致的万元后再比较（WAN_TO_YUAN）。
    - mid = 落地日前最后一次预告修订（ann ≤ 落地 pit）的中点，同样 ×1e4。
    - 事件网格铺在**每一个**定期报告公告 pit 上：该期有配对预告则给落地偏差值，
      无配对则为 NaN —— 即下一次报告落地时旧值自动失效（不做无限期陈旧延续）。
    """
    inc = _income_rows(ctx, with_value=True)
    fc = _forecast_rows(ctx)
    if inc is None or fc is None:
        return None
    act = inc.sort_values(["stock_code", "end_i", "pit"], kind="mergesort")
    act = act.drop_duplicates(["stock_code", "end_i"], keep="first")
    fc_l = fc.sort_values(["stock_code", "end_i", "ann"], kind="mergesort")[
        ["stock_code", "end_i", "ann", "mid"]].rename(columns={"ann": "fc_ann"})
    pairs = pd.merge_asof(act.sort_values("pit", kind="mergesort"),
                          fc_l.sort_values("fc_ann", kind="mergesort"),
                          left_on="pit", right_on="fc_ann",
                          by=["stock_code", "end_i"], direction="backward")
    mid = pairs["mid"].to_numpy(dtype=np.float64) * WAN_TO_YUAN
    actual = pairs["value"].to_numpy(dtype=np.float64)
    ok = (pairs["fc_ann"].notna().to_numpy() & np.isfinite(mid)
          & np.isfinite(actual) & (mid != 0))
    rel = np.full(len(pairs), np.nan, dtype=np.float64)
    rel[ok] = (actual[ok] - mid[ok]) / np.abs(mid[ok])
    gap = np.where(ok, _compressed(rel), np.nan)
    per_period = pairs[["stock_code", "end_i"]].copy()
    per_period["gap"] = gap
    events = inc[["stock_code", "end_i", "pit"]].merge(per_period,
                                                       on=["stock_code", "end_i"], how="left")
    events = events.sort_values(["stock_code", "end_i", "pit"], kind="mergesort")
    events = events.drop_duplicates(["stock_code", "pit"], keep="last")
    return ctx.asof_daily(events["stock_code"].to_numpy(),
                          events["pit"].to_numpy(dtype=np.int64),
                          events["gap"].to_numpy(dtype=np.float64))


# ══════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════

def build(parameters):
    """One of the preregistered U1 signals; directions fixed before any result was seen."""
    params = dict(parameters)
    signal = params.pop("signal", None)
    if signal not in SIGNALS:
        raise ValueError(f"Family permits only preregistered signals: {sorted(SIGNALS)}")
    skip = int(params.pop("skip", 0))
    if params:
        raise ValueError(f"Unregistered parameters: {sorted(params)}")
    if signal == "pead_drift_60":
        if skip not in (0, DRIFT_SKIP):
            raise ValueError("pead_drift_60 permits only the preregistered skip in {0, 5}")
    elif skip:
        raise ValueError("skip is only defined for pead_drift_60")

    def compute(ctx):
        if signal == "sue_seasonal_q":
            value = _sue(ctx, standardized=True)
        elif signal == "sue_seasonal_nostd":
            value = _sue(ctx, standardized=False)
        elif signal == "sue_accel":
            value = _sue_accel(ctx)
        elif signal == "pead_drift_60":
            value = _pead_drift(ctx, skip=skip)
        elif signal == "forecast_implied_surprise":
            value = _implied_surprise(ctx)
        else:
            value = _landing_gap(ctx)
        if value is None:
            value = ctx.panel.empty(dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
