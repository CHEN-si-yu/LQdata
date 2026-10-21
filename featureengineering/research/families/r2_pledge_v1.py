"""R2-I1 大股东质押压力：六个事前登记候选的构造器。

数据：上游只读表 `stock_pledge_stat`（`stock_code, end_date, pledge_count,
unrest_pledge, rest_pledge, total_share, pledge_ratio`）加价格层
（`stock_daily` / `stock_adj_factor`，仅压力机制用）。

════════════════════════════════════════════════════════════════════════
一、本表的真实形状（实测，见交付报告 §2；不是文档推断）
════════════════════════════════════════════════════════════════════════
1) `end_date` 是**周频截止日**：2015—2026 每年 50—55 个取值，93% 落在周五。
2) 这张表**不是**每周都给全截面：
   · 「整截面日」（池内 ≥2000 只，2018+ 共 83 个，多为月末/季末）给 2114 只，
     含大量显式 0 行；
   · 其余周（池内中位数约 1050 只）**只报有正质押余额的股票**。
   实测证据：给定「本周为正质押」的股票，下一周仍被报告的比例为
   **0.9964**（2018 年 0.9993，2020 年 0.9941，2025 年 0.9942，逐年 0.994—0.999）；
   而「两个整截面日均显式为 0」的股票在中间周被报告的比例约 0.001—0.01。
   非整截面日上的显式 0 行只有 821 行，整截面日上有 18394 行。
3) 结论：**某只股票在某周缺席 ⇔ 该周没有存续质押**。因此本模块把
   (股票 × 周) 建成稠密网格，缺席记 0；这是有字段行为证据的零填充，
   不是「无记录当无事件」的默认推断。
4) `pledge_ratio` 是**百分数、且被厂商四舍五入到 0.01**（实测
   `|pledge_ratio − 100·(unrest+rest)/total_share|` 最大 0.0050、均值 0.0024）。
   本模块改用 float64 的 `(unrest_pledge + rest_pledge) / total_share` 重算，
   避免 0.01% 的量化台阶。

════════════════════════════════════════════════════════════════════════
二、可用性（PIT）——本 Idea 最关键的风险
════════════════════════════════════════════════════════════════════════
`end_date` 是**截止日**，不是公开时刻；`conf/frequency.yaml` 把它标成
`delay_days: 0`，但那是没有实测支撑的默认值，本项目里该表 `gap_checkable: false`。
可核对的观测：2026-09-30 22:17 的上游落盘里最新 `end_date` 是 **2026-09-11**
（相差 13 个交易日）。因此本模块采用**保守发布滞后 LAG_DEFAULT = 15 个交易日**
（> 直接观测到的最大陈旧度 13 个交易日），做法是
`ctx.asof_daily`（只前向填充、绝不用未来值回填）之后再
`ctx.lag_grid(..., 15)`。第 6 个候选 `pledge_level_lag5` 用 5 个交易日，
只作**滞后敏感性对照**（若信号只在短滞后下成立，则是 PIT 泄漏的警示）。

════════════════════════════════════════════════════════════════════════
三、方向（Lead 事前固定，不因任何 IC 结果改变）
════════════════════════════════════════════════════════════════════════
四个机制全部 higher_is_better = false。本模块只返回值，不做反号。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea import mathx as mx

SIGNALS = (
    "pledge_level",
    "pledge_change",
    "pledge_change_fast",
    "pledge_price_stress",
    "pledge_structure",
    "pledge_level_lag5",
)

LAG_DEFAULT = 15       # 保守发布滞后：> 实测最大陈旧度 13 个交易日
LAG_FAST = 5           # 仅用于 PIT 敏感性对照
QUARTER_STEP = 65      # 13 周 ≈ 一个季度的质押比例变化
MONTH_STEP = 25        # 5 周 ≈ 一个月的质押比例变化
HIGH_DAYS = 250        # 「相对近期高点」的窗口（约一年）
HIGH_MIN = 200         # 上述窗口的最少有效样本
MIN_CROSS = 100        # 截面标准化/排名的有效样本下限

_COLUMNS = ("stock_code", "end_date", "pledge_count", "unrest_pledge",
            "rest_pledge", "total_share")


def _divide(a, b, floor=1e-12):
    with np.errstate(all="ignore"):
        out = (np.asarray(a, dtype=np.float64) /
               np.asarray(b, dtype=np.float64))
    return np.where(np.isfinite(out) & (np.abs(b) > floor), out, np.nan)


def _cs_pct_rank(x, mask):
    """池内截面百分位排名（平均并列），返回 (0,1]；被 mask 掉的格子为 NaN。

    用 pandas 的 rank 而不是 argsort，是为了让并列值拿平均秩 —— 质押比例
    在整截面日上有大量精确 0（约 1000 只），拿名次而不平均并列会把
    「同样没质押」的股票人为排序。
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x)
    work = np.where(m, x, np.nan)
    ranked = pd.DataFrame(work).rank(axis=1, pct=True, na_option="keep")
    return np.where(m, ranked.to_numpy(dtype=np.float64), np.nan)


def _weekly_dense(ctx, frame):
    """(股票 × 周) 稠密网格：缺席 = 0；返回 level 与 rest_share 两张网格。

    level     = (unrest_pledge + rest_pledge) / total_share，无量纲比例
    rest_share= rest_pledge / (unrest_pledge + rest_pledge)，无质押时为 0
                （没有质押敞口 ⇒ 没有限售质押敞口，这是机制定义而非缺省填充）
    """
    codes = np.asarray(ctx.panel.codes).astype(str)
    code = frame["stock_code"].astype(str).to_numpy()
    days = ctx.date_col(frame["end_date"])

    unrest = pd.to_numeric(frame["unrest_pledge"], errors="coerce").to_numpy(np.float64)
    rest = pd.to_numeric(frame["rest_pledge"], errors="coerce").to_numpy(np.float64)
    total = pd.to_numeric(frame["total_share"], errors="coerce").to_numpy(np.float64)

    valid = (np.isfinite(unrest) & np.isfinite(rest) & np.isfinite(total)
             & (unrest >= 0) & (rest >= 0) & (total > 0) & (days >= 19000101))
    pledged = np.where(valid, unrest + rest, np.nan)
    level = np.where(valid, _divide(pledged, total), np.nan)
    rest_share = np.where(valid & (pledged > 0), _divide(rest, pledged), np.nan)
    # 有记录但明确记为 0（整截面日上很常见）：rest_share 定义为 0
    rest_share = np.where(valid & (pledged <= 0), 0.0, rest_share)

    obs = pd.DataFrame({"code": code, "day": days.astype(np.int64),
                        "level": level, "rest_share": rest_share})
    obs = obs[obs["code"].isin(set(codes.tolist()))]
    if obs.duplicated(["code", "day"]).any():
        raise ValueError("stock_pledge_stat 在 (stock_code, end_date) 上有重复键")

    weeks = np.unique(days[(days >= 19000101)]).astype(np.int64)
    grid = pd.MultiIndex.from_product([codes, weeks], names=["code", "day"])
    dense = obs.set_index(["code", "day"]).reindex(grid)
    # 缺席 = 该周没有存续质押 = 0（见模块头 §1 的实测证据）
    dense["level"] = dense["level"].fillna(0.0)
    dense["rest_share"] = dense["rest_share"].fillna(0.0)

    dcode = dense.index.get_level_values("code").to_numpy().astype(str)
    dday = dense.index.get_level_values("day").to_numpy().astype(np.int32)
    level_grid = np.asarray(
        ctx.asof_daily(dcode, dday, dense["level"].to_numpy(np.float64)),
        dtype=np.float64)
    rest_grid = np.asarray(
        ctx.asof_daily(dcode, dday, dense["rest_share"].to_numpy(np.float64)),
        dtype=np.float64)
    return level_grid, rest_grid


def _pledge_grids(ctx):
    days = ctx.panel.dates
    years = (int(days[0]) // 10000, int(days[-1]) // 10000)
    frame = ctx.dataset("stock_pledge_stat", columns=list(_COLUMNS), years=years)
    if frame is None or len(frame) == 0:
        raise RuntimeError("stock_pledge_stat 在请求年份内没有可见行")
    return _weekly_dense(ctx, frame)


def _price_valid(ctx):
    traded = np.asarray(ctx.traded(), dtype=bool)
    close = np.asarray(ctx.hfq("close"), dtype=np.float64)
    valid = traded & np.isfinite(close) & (close > 0)
    return np.where(valid, close, np.nan)


def build(parameters):
    """六个事前登记信号之一；不含任何被搜索的窗口或方向。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("R2-I1 只允许六个事前登记的 signal 标识")
    signal = parameters["signal"]

    def compute(ctx):
        level_grid, rest_grid = _pledge_grids(ctx)
        lag = LAG_FAST if signal == "pledge_level_lag5" else LAG_DEFAULT

        if signal in ("pledge_level", "pledge_level_lag5"):
            value = np.log1p(100.0 * ctx.lag_grid(level_grid, lag))
        elif signal in ("pledge_change", "pledge_change_fast"):
            step = QUARTER_STEP if signal == "pledge_change" else MONTH_STEP
            lagged = ctx.lag_grid(level_grid, lag)
            # 两条腿用同一个 15 交易日滞后，差值即「过去 step 个交易日的质押比例变化」
            value = 100.0 * (lagged - mx.shift(lagged, step))
        elif signal == "pledge_structure":
            value = ctx.lag_grid(rest_grid, lag)
        else:
            lagged = ctx.lag_grid(level_grid, lag)
            close = _price_valid(ctx)
            peak = mx.roll_max(close, HIGH_DAYS, min_count=HIGH_MIN)
            with np.errstate(all="ignore"):
                ratio = close / peak
            depth = np.where(np.isfinite(ratio),
                             np.clip(1.0 - ratio, 0.0, 1.0), np.nan)
            mask = (np.asarray(ctx.universe, dtype=bool)
                    & np.isfinite(lagged) & np.isfinite(depth))
            pressure = _cs_pct_rank(np.log1p(100.0 * lagged), mask)
            weakness = _cs_pct_rank(depth, mask)
            valid = mask & np.isfinite(pressure) & np.isfinite(weakness)
            value = np.where(valid, pressure * weakness, np.nan)
            counts = np.sum(mask, axis=1)
            value = np.where(counts[:, None] >= MIN_CROSS, value, np.nan)

        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
