"""涨跌停生态（5 个股票因子）—— 填补库里的空白。

## 为什么挑这两块

按名字扫过 `factorcatalog`：`mf_*` 资金流已有 28 个、涨跌停/封单类已有 32 个
（大多出自 `stock_limit_up`/`stock_limit_list`），而
`stock_limit_list` 的**炸板态（limit=Z）**需要与涨停态分开统计。所以这批只做涨跌停生态，
不往已经饱和的资金流里再塞。

★ 停牌族**试过又撤了**：`stock_suspension` 确实一个因子都没有，但实测做出来的四个量
（近 60 日停牌天数 99.0% 是 0、近 250 日停牌次数同样、距今距离 95% 是 NaN、
归一化的 250 日占比 91.1% 是 0）全部落在"截面排序被并列的 0 支配"的形态里，
正是本工程质检会判删的那一类。**自己会判删的因子不该往里加**，所以停牌这块
留到上游能提供**停牌区间**（起止时间，现在那两个字段是全空列）之后再做。

## 口径

1. **事件表 vs 状态表**：`stock_limit_list` / `stock_suspension` 是**事件表**，
   只登记"发生了的股票" ⇒ 表覆盖范围内没记录的交易日是**计数 0**，不是缺失；
   但表**覆盖范围之外**（`stock_limit_list` 2020 之前）必须保持 NaN，
   否则 2018 年会满屏"涨停 0 次"这种假事实 —— 由 `start="2020-01-01"` 直接截断。
2. ★ **加权事件铺网格前必须先滤掉非有限权重**：`Panel.scatter` 用的是
   `np.bincount(idx, weights=w)`，NaN 权重会**污染整个格子**（0 + NaN = NaN），
   而滚动窗口「窗口内出现 NaN 则整窗 NaN」⇒ 一个缺失的开板次数会让后面 20 天全部变空。
   实测踩过：开板次数网格不滤 NaN 时，比率因子的非空率会大幅下降。
3. 不做全样本标准化、不用未来值回填；缺失保持 NaN。
4. ★ **不做零膨胀的计数型因子**。这一批本来还写了「近 20 日炸板次数 / 跌停次数 /
   (涨停−跌停) 次数 / 涨停日成交额占比」四个，实测零占比 78%~91%；停牌族的四个量
   （停牌天数/次数、距今距离、占比）零占比 91%~99% —— 那正是本工程的质检会把
   「波动性低 / 截面排序被并列值支配」判为待删的形态。**自己会判删的因子不该往里加**，
   因此本模块保留比率型、距今距离型与连板最高值摘要，计数为零的事件窗口按事件表口径处理。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fea.spec import FactorSpec, register

LIMIT_DEPS = ("stock_limit_list", "stock_daily", "stock_adj_factor")
START_LIMIT = "2020-01-01"       # stock_limit_list 实测最早 2020-01-02

_LIMIT_CACHE: dict[tuple, dict] = {}


def _ev(ctx, codes, days, values=None) -> np.ndarray:
    """把稀疏事件铺成 `(T, C)`：同格累加，窗口外的格子补 0。

    ★ 带 `values` 时**先滤掉非有限值**再铺 —— 见模块 docstring 第 2 条。
    """
    if values is None:
        return ctx.event_grid(codes, days)
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    return ctx.event_grid(codes[ok], days[ok], v[ok])


def _years(ctx):
    return (int(ctx.panel.dates[0]) // 10000, int(ctx.panel.dates[-1]) // 10000)


def _limit_events(ctx) -> dict:
    key = (id(ctx.up), int(ctx.panel.dates[0]), int(ctx.panel.dates[-1]))
    hit = _LIMIT_CACHE.get(key)
    if hit is not None:
        return hit
    df = ctx.dataset("stock_limit_list",
                     columns=["trade_date", "stock_code", "limit", "open_times",
                              "limit_times", "fd_amount", "float_mv"],
                     years=_years(ctx))
    if df is None or df.empty:
        raise RuntimeError("stock_limit_list 缺源")
    codes = df["stock_code"].astype(str).to_numpy()
    days = (df["trade_date"].astype(str).str.replace("-", "", regex=False)
            .to_numpy(dtype=np.int32))
    lim = df["limit"].astype(str).to_numpy()
    open_times = pd.to_numeric(df["open_times"], errors="coerce").to_numpy(dtype=np.float64)
    limit_times = pd.to_numeric(df["limit_times"], errors="coerce").to_numpy(dtype=np.float64)
    fd_amount = pd.to_numeric(df["fd_amount"], errors="coerce").to_numpy(dtype=np.float64)
    float_mv = pd.to_numeric(df["float_mv"], errors="coerce").to_numpy(dtype=np.float64)
    # ★★ 这三个量只对**涨停（U）**行有意义，必须按 is_u 掩码后再铺网格。
    #   不掩码的后果是静默的：`stock_limit_list` 同一天对同一只股票也可能有一条
    #   **炸板（Z）**行，其 `open_times` 是炸板过程中打开的次数 —— 混进来会让
    #   「近 20 日涨停的日均开板次数」把炸板次数也算进去。
    #   实测抓法：独立复算（另一条代码路径）在 002010.SZ / 2021-06-15 上得到 0.5，
    #   产物给 1.5 —— 差额正是窗口里那条 Z 行的 open_times=2。
    is_u = np.asarray(lim) == "U"
    out = {
        "up": _ev(ctx, codes, days, is_u.astype(np.float64)),
        "broken": _ev(ctx, codes, days, (lim == "Z").astype(np.float64)),
        "open_times": _ev(ctx, codes, days, np.where(is_u, open_times, np.nan)),
        "seal_to_float": _ev(ctx, codes, days,
                              np.where(is_u, _ratio(fd_amount, float_mv), np.nan)),
        "streak": _ev(ctx, codes, days, np.where(is_u, limit_times, np.nan)),
    }
    _LIMIT_CACHE[key] = out
    if len(_LIMIT_CACHE) > 2:
        for k in list(_LIMIT_CACHE)[:-1]:
            _LIMIT_CACHE.pop(k, None)
    return out


def _days_since(grid: np.ndarray) -> np.ndarray:
    """距今多少个交易日（当日有事件记 0）；从未发生过记 NaN。"""
    T = grid.shape[0]
    idx = np.repeat(np.arange(T, dtype=np.float64)[:, None], grid.shape[1], axis=1)
    last = np.where(np.isfinite(grid) & (grid > 0), idx, np.nan)
    filled = pd.DataFrame(last).ffill().to_numpy(dtype=np.float64)
    return idx - filled


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(den) & (den > 0), num / den, np.nan)


# ══════════════════════════════════════════════════════════════════════
# 涨跌停生态（5）
# ══════════════════════════════════════════════════════════════════════
def limit_broken_ratio_20(ctx):
    ev = _limit_events(ctx)
    touch = ev["broken"] + ev["up"]
    return _ratio(ctx.roll_sum(ev["broken"], 20), ctx.roll_sum(touch, 20))


def limit_days_since_up_20(ctx):
    d = _days_since(_limit_events(ctx)["up"])
    return np.where(d <= 20, d, np.nan)


def limit_open_pressure_20(ctx):
    ev = _limit_events(ctx)
    # 同上：分母是涨停日数
    return _ratio(ctx.roll_sum(ev["open_times"], 20), ctx.roll_sum(ev["up"], 20))


def limit_seal_to_float_20(ctx):
    ev = _limit_events(ctx)
    return _ratio(ctx.roll_sum(ev["seal_to_float"], 20), ctx.roll_sum(ev["up"], 20))


def limit_streak_max_60(ctx):
    return ctx.roll_max(_limit_events(ctx)["streak"], 60)


# ══════════════════════════════════════════════════════════════════════
# 注册
# ══════════════════════════════════════════════════════════════════════
def _register(name: str, desc: str, formula: str, deps: tuple, fn, *,
              start: str | None, warmup: int, note: str = "") -> None:
    def compute(ctx, _fn=fn):
        return _fn(ctx)
    compute.__name__ = name
    register(FactorSpec(
        name=name, group="tradability", desc=desc, formula=formula, deps=deps,
        start=start, warmup_days=warmup, version=1,
        note="池外股票不产出；缺失保持 NaN，不用未来值回填。" + note))(compute)


_LIMIT_NOTE = ("事件表口径：`stock_limit_list` 只登记发生了事件的股票，"
               "表覆盖范围内没有记录的交易日计 0；覆盖范围之外（2020 之前）由 start 截断为不产出。")

for _n, _d, _f, _w, _fn in (
    ("limit_broken_ratio_20", "近 20 日炸板占触板次数比",
     "sum(Z,20)/(sum(Z,20)+sum(U,20))", 60, limit_broken_ratio_20),
    ("limit_days_since_up_20", "距最近一次涨停的交易日数", "t-last(limit=U)，超过 20 日记 NaN",
     60, limit_days_since_up_20),
    ("limit_open_pressure_20", "近 20 日涨停的日均开板次数", "mean(open_times | limit=U, 20日)",
     60, limit_open_pressure_20),
    ("limit_seal_to_float_20", "近 20 日涨停日封单额占流通市值比均值",
     "mean(fd_amount/float_mv | limit=U, 20日)", 60, limit_seal_to_float_20),
    ("limit_streak_max_60", "近 60 日最高连板数",
     "max(limit_times | limit=U, 60日)", 120, limit_streak_max_60),
):
    _register(_n, _d, _f, LIMIT_DEPS, _fn, start=START_LIMIT, warmup=_w, note=_LIMIT_NOTE)
