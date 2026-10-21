"""第二批市场/情绪标量（12 个；2026-10-04 市场因子复审清理后，原注册 31 个）—— 涨跌停生态 · 资金流结构 · 两融杠杆 · 板块轮动 · 指数情绪 · 量能 · 可交易性。

## 与 `factors/market.py`（第一批 22 个）的关系

第一批只用了 `stock_daily` / `stock_adj_factor` / `stock_history_5min` / `index_daily`
（且 `index_daily` 库里有一万多个指数码，只用掉了 4 个）。本模块专门开**上游里一直没人读过**的
那批表：全市场涨跌分布、涨跌停明细、大小单资金流分层、两融余额、行业板块指数、停牌。

两批**互不共享状态**，只从 `market.py` 借几个纯函数（`_div` / `_ratio` / `_gt` 这些）——
「分母保护」「带容差的阈值比较」这类口径抄两遍迟早会走样（第一批为「恰好 5% 算不算涨停」
专门加过 1e-12 量级的容差）。

## 口径（每条都是实测出来的，不是默认值）

1. **样本范围分两类，逐个因子在 `note` 里写明**：
   - **固定池口径**（多数）：只用冻结的 2115 只主板股票；缺测/停牌不计入分母；
     有效样本少于 `min_cross_section` 记 NaN。与第一批一致。
   - **全市场口径**：`stock_market_distribution_history` 上游给的就是**已聚合好的全市场家数**
     （没有个股明细，还原不回 2115 池）。这类因子的名字统一加 `mkt_full_` 前缀，
     `note` 里逐条写明，避免和固定池口径的数字混着用。
2. **可得性**：`stock_margin_detail` 实测滞后 1 个交易日（`fea/delay.py` 生效表里唯一一张），
   两融族每个因子都声明 `lagged_ok` 并把网格整体下移一行 —— 不做的话 T 日的值会在
   数据到达后**静默变化**（当天看着完全正确）。
3. **资金流金额单位是万元**（实测：`stock_main_fund_flow` 四档买卖金额之和恰好等于
   `stock_daily.amount ÷ 1e4`，逐行对齐）。所以只用**同表内部**的比率，不跨表相除。
4. `net_mf_amount` 字段**不用**：实测它不等于四档买卖之差
   （2025-01-02 000001.SZ：四档净额 ≈ 0，该字段 −64322.99），语义无法从源表自证。
   净额一律由四档原始列自行构造，可复核。
5. **行业板块候选**沿用 `factors/sector.py::_verified_industry_boards` 的口径：
   TDX 0 类里**名称与 dc_blocks「行业板块」精确一致**的子集。TDX 0 类混着总市值、
   涨跌家数这类统计序列，不交叉核验就会把它们当行业算进横截面统计。
6. 不做截面 rank、每天恰好一行；缺失保持 NaN，不用未来值回填。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fea.spec import FactorSpec, register

from .market import _div, _gt

# 各族输出起点：由数据源**实测**起点决定，不是估计值
START_LIMIT_LIST = "2020-01-01"   # stock_limit_list 实测最早 2020-01-02
START_DEFAULT = "2018-01-01"      # 其余源都早于它

_LAG1 = "stock_margin_detail 实测滞后 1 个交易日，网格整体下移一行后使用。"


# ══════════════════════════════════════════════════════════════════════
# 通用工具
# ══════════════════════════════════════════════════════════════════════
def _rows(panel_days: np.ndarray, days, values, agg: str = "sum") -> np.ndarray:
    """把「(交易日, 值)」折叠成与面板等长的 `(T,)`；同一天多条时按 `agg` 聚合。

    上游有几张表是**一天多行**的（涨跌分布是分钟级）。折叠口径必须显式写出来，
    不能靠下游默认 —— 默认会把它们当成一天一行静默算错。
    """
    d = np.asarray(days, dtype=np.int64)
    s = pd.Series(np.asarray(values, dtype=np.float64), index=d)
    g = s.groupby(level=0, sort=True).agg(agg)
    return g.reindex(np.asarray(panel_days, dtype=np.int64)).to_numpy(dtype=np.float64)


def _count_by_day(panel_days: np.ndarray, days, weights=None, cover=None) -> np.ndarray:
    """事件计数折叠成 `(T,)`。

    ★ 与 `_rows` 的语义差别是**关键**，不能混用：
      · `stock_limit_list` / `stock_suspension` 这类**事件表**只登记"发生了事件的股票"，
        所以表覆盖范围内「这一天一条记录都没有」= **计数 0**，不是缺失。
        把两者混成一个 NaN，实测让 `limit_net_ratio` 白丢 17% 的交易日。
      · 但事件表**覆盖范围之外**（`stock_limit_list` 2020 之前）必须仍是 NaN，
        否则 2018 年会满屏"涨停 0 家"这种假事实。`cover` 传表自身的交易日集合来划这条线。
      · `cover=None` 时**无法区分**上面两种情况 ⇒ 保守地保持 NaN（不猜成 0）。
    """
    pd_days = np.asarray(panel_days, dtype=np.int64)
    d = np.asarray(days, dtype=np.int64)
    w = (np.ones(d.size, dtype=np.float64) if weights is None
         else np.asarray(weights, dtype=np.float64))
    g = pd.Series(w, index=d).groupby(level=0, sort=True).sum()
    # ★ copy=True 不能省：reindex 恰好是恒等映射时 pandas 会返回底层缓冲的**只读视图**，
    #   后面的布尔赋值会抛 `ValueError: assignment destination is read-only`。
    #   （`fea/panel.py::cs_rank` 里记过同一个坑；这里比那里更隐蔽 ——
    #    它只在"面板日期正好等于事件日期"时才触发，平时看着完全正常。）
    out = np.array(g.reindex(pd_days).to_numpy(dtype=np.float64), copy=True)
    if cover is None:
        return out
    c = np.asarray(cover, dtype=np.int64)
    if c.size == 0:
        return out
    inside = (pd_days >= c.min()) & (pd_days <= c.max())
    out[inside & ~np.isfinite(out)] = 0.0        # 覆盖范围内的"没记录" = 计数 0
    out[~inside] = np.nan                        # 覆盖范围外 = 真的没有数据
    return out


def _lag2d(M: np.ndarray, k: int) -> np.ndarray:
    a = np.asarray(M, dtype=np.float64)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if k < a.shape[0]:
        out[k:] = a[:a.shape[0] - k]
    return out


def _roll_sum_nan(M: np.ndarray, n: int) -> np.ndarray:
    """`(T, K)` 的 nan-安全滚动和：窗口内出现 NaN 则结果 NaN（不把缺失当 0 累加）。"""
    a = np.asarray(M, dtype=np.float64)
    T, K = a.shape
    fin = np.isfinite(a)
    x = np.where(fin, a, 0.0)
    cs = np.vstack([np.zeros((1, K)), np.cumsum(x, axis=0)])
    cb = np.vstack([np.zeros((1, K)), np.cumsum(~fin, axis=0)])
    out = np.full((T, K), np.nan, dtype=np.float64)
    if T >= n:
        out[n - 1:] = cs[n:] - cs[:-n]
        out[n - 1:][(cb[n:] - cb[:-n]) > 0] = np.nan
    return out


def _cs(M: np.ndarray, kind: str, need: int) -> np.ndarray:
    """逐交易日横截面统计（行向量化）。`M` 为 `(T, K)`，NaN 不计入分母。"""
    x = np.asarray(M, dtype=np.float64)
    fin = np.isfinite(x)
    n = fin.sum(axis=1)
    z = np.where(fin, x, 0.0)
    mu = _div(z.sum(axis=1), n)
    if kind == "mean":
        out = mu
    elif kind == "std":
        out = np.sqrt(np.maximum(_div((np.where(fin, x * x, 0.0)).sum(axis=1), n) - mu * mu, 0.0))
    elif kind == "sum":
        out = z.sum(axis=1)
    elif kind == "up_ratio":
        out = _div(_gt(x, 0.0).sum(axis=1), n)
    else:
        raise ValueError(f"未知横截面统计：{kind}")
    return np.where(n >= need, out, np.nan)


def _ratio_all(ctx, test: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """池内「满足条件 / 有效样本」占比；有效样本不足 `min_cross_section` 记 NaN。"""
    n = valid.sum(axis=1)
    hit = (test & valid).sum(axis=1).astype(np.float64)
    return np.where(n >= ctx.cfg.min_cross_section, _div(hit, n.astype(np.float64)), np.nan)


def _sum_ok(ctx, x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """按有效样本求和；有效样本不足 `min_cross_section` 记 NaN。"""
    v = valid & np.isfinite(x)
    n = v.sum(axis=1)
    return np.where(n >= ctx.cfg.min_cross_section, np.where(v, x, 0.0).sum(axis=1), np.nan)


_CACHE: dict[tuple, dict] = {}


def _pkey(ctx, tag: str):
    return (tag, int(ctx.panel.dates[0]), int(ctx.panel.dates[-1]), int(ctx.panel.codes.size))


def _cache_get(key):
    return _CACHE.get(key)


def _cache_put(key, val):
    _CACHE[key] = val
    # 只留最近两个面板指纹：长跑进程里按年累积会把内存吃光（同 `fea` 的 trim 精神）
    if len(_CACHE) > 4:
        for k in list(_CACHE)[:-2]:
            _CACHE.pop(k, None)


def _read_table(ctx, dataset: str, code_col: str, value_cols, suffix: str | None = None,
                keep=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读「(代码, 交易日) → 多列值」的上游表，返回 `(dpos, codes, cpos, values, ok)`。

    ★ **不在这里做轴对齐**：上游表的「股票代码全集」是**全市场**（两融表实测 4140 个码），
      而面板只有冻结池的 2115 列，且两者顺序不同。谁用谁负责把它落到自己那根轴上 ——
      直接拿上游的码当列下标会**静默错位**（形状对不上时还能报错，一旦恰好相等就完全无声）。
    """
    p = ctx.panel
    years = (int(p.dates[0]) // 10000, int(p.dates[-1]) // 10000)
    names = list(value_cols) if isinstance(value_cols, (list, tuple)) else [value_cols]
    df = ctx.dataset(dataset, columns=[code_col, "trade_date", *names], years=years)
    if df is None or df.empty:
        # ★ 缺源必须**明确失败**：静默返回全 NaN 会让「数据没到」看起来像「因子算出来是空的」，
        #   而后者会被下游当成正常的稀疏因子放过去。
        raise RuntimeError(f"{dataset} 在 {years} 区间没有数据（缺源）")
    if suffix:
        df = df.assign(**{code_col: df[code_col].astype(str).str.replace(suffix, "", regex=False)})
    if keep is not None:
        df = df[df[code_col].isin(set(keep))]
    day_i = (df["trade_date"].astype(str).str.replace("-", "", regex=False)
             .to_numpy(dtype=np.int64))
    codes = np.sort(df[code_col].astype(str).unique())
    ci = pd.Series(np.arange(codes.size), index=codes)
    cpos = ci.reindex(df[code_col].astype(str)).to_numpy()
    di = pd.Series(np.arange(p.dates.size), index=np.asarray(p.dates, dtype=np.int64))
    dpos = di.reindex(day_i).to_numpy()
    ok = np.isfinite(cpos) & np.isfinite(dpos)
    vals = np.stack([pd.to_numeric(df[nm], errors="coerce").to_numpy(dtype=np.float64)
                     for nm in names], axis=1)
    return dpos, codes, cpos, vals, ok


def _load_panel(ctx, dataset: str, value_cols) -> np.ndarray:
    """读个股级上游表，**直接落到面板的列轴**上，返回 `(T, C, ncol)`。

    池外的股票被丢弃、池内该表没有的股票整列 NaN —— 只有这样，
    `ctx.universe` / `ctx.traded()` 这些 (T, C) 掩码才能和它逐格对齐。
    """
    p = ctx.panel
    names = list(value_cols) if isinstance(value_cols, (list, tuple)) else [value_cols]
    dpos, codes, cpos, vals, ok = _read_table(ctx, dataset, "stock_code", names)
    col = ctx.code_index(codes)                       # 上游码 → 面板列下标（池外 −1）
    out = np.full((p.dates.size, p.codes.size, len(names)), np.nan, dtype=np.float64)
    m = ok & (col[cpos] >= 0)
    if m.any():
        rows = dpos[m].astype(np.int64)
        cols = col[cpos[m]].astype(np.int64)
        for k in range(len(names)):
            out[rows, cols, k] = vals[m, k]
    return out


def _load_axis(ctx, dataset: str, code_col: str, value_cols, suffix: str | None = None,
               keep=None):
    """读「自成一根轴」的上游表（板块/指数），返回 `(codes[K], V[T,K,ncol])`。"""
    p = ctx.panel
    names = list(value_cols) if isinstance(value_cols, (list, tuple)) else [value_cols]
    dpos, codes, cpos, vals, ok = _read_table(ctx, dataset, code_col, names,
                                              suffix=suffix, keep=keep)
    out = np.full((p.dates.size, codes.size, len(names)), np.nan, dtype=np.float64)
    if ok.any():
        rows = dpos[ok].astype(np.int64)
        cols = cpos[ok].astype(np.int64)
        for k in range(len(names)):
            out[rows, cols, k] = vals[ok, k]
    return codes, out


# ══════════════════════════════════════════════════════════════════════
# A 涨跌停生态（3；2026-10-04 复审清理后）
# ══════════════════════════════════════════════════════════════════════
LIMIT_DEPS_DIST = ("stock_market_distribution_history",)
LIMIT_DEPS_LIST = ("stock_limit_list", "stock_daily", "stock_adj_factor")


def _limit_breadth(ctx) -> dict:
    """全市场涨跌**家数**（上游已聚合的分钟级分布表，取每日 15:00 那一行）。

    ★ 全市场口径：这张表按**全市**聚合，没有个股明细，还原不到 2115 池 ⇒
      名字统一带 `mkt_full_`，与固定池口径的数字不可混用。
    ★ 取 15:00 行而不是「当天最后一行」：源表极少数日期缺尾行，取最后一行会把
      14:5x 的中间状态当成收盘状态。15:00 缺失就记 NaN，宁可缺也不要错。
    ★★ **只用左尾可靠的列**。实测（2018~2026 逐年统计「15:00 值为 0 的天数」）：
      · `up_count` / `down_count` / `flat_count` / `up_over_10` / `up_7_to_10` /
        `up_5_to_7` / `up_3_to_5` / `up_0_to_3` —— **从未出现 0**，可用；
      · `limit_up_count` / `limit_down_count` —— **2018、2019 两年全年为 0**，
        2020 起仍偶发（2026-09-23 全市场记 0，而同日 `stock_limit_list` 有 51 只 U、13 只 D）；
      · `down_over_10` / `down_7_to_10` / `down_5_to_7` —— 有数百/数十天为 0，右尾不稳。
      ⇒ 涨停跌停家数一律改用 `stock_limit_list`（见 `_limit_eco`，2020 起），
        本函数只保留左尾那几列。把不可靠的列做成因子，等于往库里塞一颗随时会响的雷。
    """
    key = _pkey(ctx, "limit_dist")
    hit = _cache_get(key)
    if hit is not None:
        return hit
    p = ctx.panel
    years = (int(p.dates[0]) // 10000, int(p.dates[-1]) // 10000)
    df = ctx.dataset("stock_market_distribution_history",
                     columns=["trade_time", "up_count", "down_count", "flat_count",
                              "up_over_10", "up_7_to_10", "up_5_to_7", "up_3_to_5"],
                     years=years)
    if df is None or df.empty:
        raise RuntimeError("stock_market_distribution_history 缺源")
    tt = df["trade_time"].astype(str)
    is_close = tt.str.endswith(" 15:00:00").to_numpy()
    df = df.loc[is_close]
    d = (tt[is_close].str.slice(0, 10).str.replace("-", "", regex=False)
         .to_numpy(dtype=np.int64))

    def num(col):
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)

    def fold(values, agg="sum"):
        return _rows(p.dates, d, values, agg=agg)

    up, dn, fl = num("up_count"), num("down_count"), num("flat_count")
    total = fold(up + dn + fl)
    extreme = fold(num("up_over_10") + num("up_7_to_10"))
    same_day = np.isfinite(total) & (total > 0)
    out = {
                "full_flat_ratio": _div(fold(fl), total),
    }
    out = {k: np.where(same_day, v, np.nan) for k, v in out.items()}
    _cache_put(key, out)
    return out


def _limit_eco(ctx) -> dict:
    """涨跌停**家数**（`stock_limit_list`，实测 2020 起）：池内涨停/跌停家数占比。

    `limit` 三态 U/D/Z 中只取 U、D；Z（炸板）/连板/封单族于 2026-10-04 复审清理删除。
    """
    key = _pkey(ctx, "limit_eco")
    hit = _cache_get(key)
    if hit is not None:
        return hit
    p = ctx.panel
    years = (int(p.dates[0]) // 10000, int(p.dates[-1]) // 10000)
    df = ctx.dataset("stock_limit_list",
                     columns=["trade_date", "stock_code", "limit"],
                     years=years)
    if df is None or df.empty:
        raise RuntimeError("stock_limit_list 缺源")
    cover = (df["trade_date"].astype(str).str.replace("-", "", regex=False)
             .to_numpy(dtype=np.int64))
    ci = ctx.code_index(df["stock_code"].astype(str))
    df = df.loc[ci >= 0]
    d = df["trade_date"].astype(str).str.replace("-", "", regex=False).to_numpy(dtype=np.int64)

    lim = df["limit"].astype(str)
    is_u = (lim == "U").to_numpy()

    n_pool = ctx.universe.sum(axis=1).astype(np.float64)

    def count(mask):
        m = np.asarray(mask) & np.isfinite(d)
        return _count_by_day(p.dates, d[m], cover=cover)

    is_d = (lim == "D").to_numpy()
    n_u, n_d = count(is_u), count(is_d)
    out = {
        "limit_up_ratio": _div(n_u, n_pool),
        "limit_down_ratio": _div(n_d, n_pool),
    }
    _cache_put(key, out)
    return out


# ══════════════════════════════════════════════════════════════════════
# B 资金流结构（2；2026-10-04 复审清理后）
# ══════════════════════════════════════════════════════════════════════
MF_DEPS = ("stock_main_fund_flow", "stock_daily", "stock_adj_factor")
_TIERS = ("sm", "md", "lg", "elg")


def _money_flow(ctx) -> dict:
    key = _pkey(ctx, "mf")
    hit = _cache_get(key)
    if hit is not None:
        return hit
    names = [f"buy_{t}_amount" for t in _TIERS] + [f"sell_{t}_amount" for t in _TIERS]
    V = _load_panel(ctx, "stock_main_fund_flow", names)
    n = len(_TIERS)
    # 四档齐全才算有效：只到一半的记录算出来的净额是残的
    fin = np.isfinite(V).all(axis=2)
    net = np.zeros(fin.shape, dtype=np.float64)
    gross = np.zeros(fin.shape, dtype=np.float64)
    big = np.zeros(fin.shape, dtype=np.float64)
    sml = np.zeros(fin.shape, dtype=np.float64)
    elg = np.zeros(fin.shape, dtype=np.float64)
    for k, t in enumerate(_TIERS):
        b, s = V[:, :, k], V[:, :, k + n]
        nt = b - s
        net += np.where(fin, nt, 0.0)
        gross += np.where(fin, b + s, 0.0)
        if t in ("lg", "elg"):
            big += np.where(fin, nt, 0.0)
        else:
            sml += np.where(fin, nt, 0.0)
        if t == "elg":
            elg = np.where(fin, b + s, 0.0)
    net = np.where(fin, net, np.nan)
    big = np.where(fin, big, np.nan)
    sml = np.where(fin, sml, np.nan)
    elg = np.where(fin, elg, np.nan)
    gross = np.where(fin & (gross > 0), gross, np.nan)
    pool = ctx.universe & ctx.traded() & fin

    den = _sum_ok(ctx, gross, pool)
    sml_r = _div(_sum_ok(ctx, sml, pool), den)
    # ★ 只留「大单净流入率」一个净额口径，理由是**恒等式**（不是经验观察）：
    #   每笔成交的买方与卖方金额相同 ⇒ Σ四档买 = Σ四档卖 ⇒
    #   ① 全档净额恒等于 0（实测相对量级 5e-8，符号由浮点舍入决定）；
    #   ② lg+elg 净额与 sm+md 净额严格互为相反数 ⇒
    #      「小单净流入率」「大减小」与「大单净流入率」在数学上共线（|ρ|=1）。
    #   所以「全档净流入家数占比」这类量是在数**舍入误差的符号**，没有经济含义 ——
    #   实测它与独立复算对不上（0.29 vs 0.44），正是"结果由求和顺序决定"的证据。
    #   独立复算把这个因子挑出来后已从本模块删除。
    out = {
        "mf_large_breadth": _ratio_all(ctx, _gt(big, 0.0), pool),
        "mf_elg_turnover_share": _div(_sum_ok(ctx, elg, pool), den),
    }
    _cache_put(key, out)
    return out


# ══════════════════════════════════════════════════════════════════════
# C 两融杠杆情绪（1；2026-10-04 复审清理后）—— 全库唯一的滞后源
# ══════════════════════════════════════════════════════════════════════
MARGIN_DEPS = ("stock_margin_detail", "stock_daily", "stock_adj_factor")


def _margin(ctx) -> dict:
    key = _pkey(ctx, "margin")
    hit = _cache_get(key)
    if hit is not None:
        return hit
    V = _load_panel(ctx, "stock_margin_detail", ["rzye", "rzmre", "rzche", "rqye"])
    # ★ 滞后 1 个交易日：这张表 T 日的数据 T+1 才拿得到（`fea/delay.py` 生效表里的唯一一张）。
    #   用平台统一的 `ctx.lag_grid`（逐列下移一行）≡ 「T 日的因子只用 trade_date ≤ T−1 的记录」。
    #   ★ 这里必须用 `lag_grid` 这个**名字**而不是自己 np.concatenate：
    #     `main.py audit-pit` 的静态扫描按源码字面量找位移，换写法它看不见（见本文件末尾注释）。
    V = np.stack([ctx.lag_grid(V[:, :, k], 1) for k in range(V.shape[2])], axis=2)
    _ = ctx  # lag_grid 已用到；保留这行是为了让"位移确实发生在这里"在源码上一目了然
    fin = np.isfinite(V).all(axis=2)
    pool = ctx.universe & ctx.traded() & fin
    out = {
        # ★ 原口径是「融资买入额>0 的家数占比」，实测恒在 99.2%~100%（std=0.001）——
        #   融资买入几乎是全市场行为，那个比率没有截面信息。改成**净买入**家数占比
        #   （rzmre>rzche），衡量的是"加杠杆 vs 还杠杆"的分歧，才有区分度。
        "margin_breadth": _ratio_all(ctx, _gt(V[:, :, 1] - V[:, :, 2], 0.0), pool),
    }
    _cache_put(key, out)
    return out


# ══════════════════════════════════════════════════════════════════════
# D 板块/行业轮动（2；2026-10-04 复审清理后）—— 只做板块指数层，不需要成分股映射
# ══════════════════════════════════════════════════════════════════════
SECTOR_DEPS = ("tdx_daily", "tdx_blocks", "dc_blocks")


def _industry_boards(ctx) -> np.ndarray:
    """交叉核验过的行业板块代码（已剥掉 `.TDX`）—— 口径与 `factors/sector.py` 一致。"""
    key = _pkey(ctx, "boards")
    hit = _cache_get(key)
    if hit is not None:
        return hit["boards"]
    tdx = ctx.dataset("tdx_blocks", columns=["block_code", "block_name", "block_type"])
    dc = ctx.dataset("dc_blocks", columns=["block_code", "block_name", "block_type"])
    if tdx is None or dc is None or tdx.empty or dc.empty:
        raise RuntimeError("行业分类字典缺失或字段不完整")
    industries = set(dc.loc[dc["block_type"].astype("string").eq("行业板块"), "block_name"]
                     .astype("string").dropna().str.strip())
    industries.discard("")
    keep = (pd.to_numeric(tdx["block_type"], errors="coerce").eq(0)
            & tdx["block_name"].astype("string").str.strip().isin(industries))
    codes = (tdx.loc[keep, "block_code"].astype("string").dropna()
             .str.replace(".TDX", "", regex=False).str.strip().unique().astype(str))
    codes = np.unique(codes[codes != ""])
    if codes.size == 0:
        raise RuntimeError("行业分类交叉核验没有候选指数，拒绝回退到混有统计序列的 TDX 0 类")
    _cache_put(key, {"boards": codes})
    return codes


def _sector_cs(ctx) -> dict:
    key = _pkey(ctx, "sector")
    hit = _cache_get(key)
    if hit is not None:
        return hit
    boards = _industry_boards(ctx)
    _, V = _load_axis(ctx, "tdx_daily", "board_code", ["close"],
                      suffix=".TDX", keep=boards)
    close = V[:, :, 0]
    ret = _div(close, _lag2d(close, 1)) - 1.0
    mom20 = np.expm1(_roll_sum_nan(np.log1p(np.where(np.isfinite(ret), ret, np.nan)), 20))
    need = 30                      # 核验后的行业数有数百个，30 个起算即视为截面够用
    out = {
        "sector_dispersion": _cs(ret, "std", need),
                "sector_mom20_dispersion": _cs(mom20, "std", need),
    }
    _cache_put(key, out)
    return out


# ══════════════════════════════════════════════════════════════════════
# E 指数情绪（3；2026-10-04 复审清理后）
# ══════════════════════════════════════════════════════════════════════
IDX_DEPS = ("index_daily", "stock_daily", "stock_adj_factor")
_BROAD = ("000300.SH", "000905.SH", "000852.SH", "000001.SH", "399001.SZ",
          "399006.SZ", "000016.SH", "000688.SH", "000985.CSI")


def _index_sentiment(ctx) -> dict:
    key = _pkey(ctx, "idx2")
    hit = _cache_get(key)
    if hit is not None:
        return hit
    p = ctx.panel
    years = (int(p.dates[0]) // 10000, int(p.dates[-1]) // 10000)
    want = set(_BROAD)
    df = ctx.dataset("index_daily",
                     columns=["ts_code", "trade_date", "close", "high", "low",
                              "pre_close"],
                     years=years)
    if df is None or df.empty:
        raise RuntimeError("index_daily 缺源")
    df = df[df["ts_code"].isin(want)]
    if df.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("index_daily 含重复主键，拒绝继续（重复主键会让 pivot 静默取错行）")
    day_i = df["trade_date"].astype(str).str.replace("-", "", regex=False).to_numpy(dtype=np.int64)
    di = pd.Series(np.arange(p.dates.size), index=np.asarray(p.dates, dtype=np.int64))
    dpos = di.reindex(day_i).to_numpy()
    codes = np.sort(df["ts_code"].astype(str).unique())
    ci = pd.Series(np.arange(codes.size), index=codes)
    cpos = ci.reindex(df["ts_code"].astype(str)).to_numpy()
    ok = np.isfinite(dpos) & np.isfinite(cpos)
    col = {c: i for i, c in enumerate(codes)}
    if "000300.SH" not in col:
        raise RuntimeError("index_daily 缺沪深300，指数类市场因子无法计算")

    def wide(c):
        out = np.full((p.dates.size, codes.size), np.nan, dtype=np.float64)
        out[dpos[ok].astype(np.int64), cpos[ok].astype(np.int64)] = \
            pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)[ok]
        return out

    close, high, low, pre = (wide(c) for c in
                             ("close", "high", "low", "pre_close"))
    ret = _div(close, _lag2d(close, 1)) - 1.0
    j = col["000300.SH"]
    c3 = close[:, j]

    hi250 = pd.Series(c3).rolling(250, min_periods=125).max().to_numpy()
    lo250 = pd.Series(c3).rolling(250, min_periods=125).min().to_numpy()
    broad = [col[c] for c in _BROAD if c in col]
    n_b = np.isfinite(ret[:, broad]).sum(axis=1)
    out = {
        "idx_csi300_amplitude20": (pd.Series(_div(high - low, pre)[:, j])
                                   .rolling(20, min_periods=10).mean().to_numpy()),
        # ★ 原来是「9 个宽基里有几个上涨」—— 一个截面只有 10 个可能取值（卡片化）。
        #   换成**跨宽基指数的收益离散**：连续、且衡量的是风格/规模之间的分化强度，
        #   与已有的 `mkt_dispersion`（个股横截面离散）是两个层次的东西。
        "idx_broad_dispersion": np.where(
            n_b >= 3, _cs(ret[:, broad], "std", 3), np.nan),
        "idx_csi300_range_position250": _div(c3 - lo250, hi250 - lo250),
        "idx_csi300_ret_skew20": (pd.Series(ret[:, j]).rolling(20, min_periods=10)
                                  .skew().to_numpy()),
            }
    _cache_put(key, out)
    return out


# ══════════════════════════════════════════════════════════════════════
# F 量能与可交易性（1；2026-10-04 复审清理后）
# ══════════════════════════════════════════════════════════════════════
LIQ_DEPS = ("stock_daily", "stock_adj_factor")


def _liquidity(ctx) -> dict:
    key = _pkey(ctx, "liq")
    hit = _cache_get(key)
    if hit is not None:
        return hit
    n_trade = (ctx.universe & ctx.traded()).sum(axis=1).astype(np.float64)
    n_pool = ctx.universe.sum(axis=1).astype(np.float64)
    out = {
        "no_trade_ratio": 1.0 - _div(n_trade, np.where(n_pool > 0, n_pool, np.nan)),
    }
    _cache_put(key, out)
    return out


# ══════════════════════════════════════════════════════════════════════
# 注册
# ══════════════════════════════════════════════════════════════════════
def _register(name: str, desc: str, formula: str, deps: tuple, compute, *,
              start: str = START_DEFAULT, warmup: int = 80, lagged: tuple = (),
              note: str = "", align_fill: float = float("nan")) -> None:
    key = name[len("mkt_"):] if name.startswith("mkt_") else name

    def fn(ctx):
        return compute(ctx)[key]
    fn.__name__ = name
    register(FactorSpec(
        name=name, group="market", is_market=True, desc=desc, formula=formula,
        deps=deps, start=start, warmup_days=warmup, version=2, lagged_ok=lagged,
        align_fill=align_fill,     # 见 FactorSpec.align_fill（时间轴对齐）
        note=("市场标量，每天恰好一行、无股票列、无横截面 rank；未做收益有效性承诺。"
              + note + " 缺失保持 NaN，不用未来值回填。")))(fn)


_FULL_NOTE = ("★ 全市场口径：源表是上游按全市聚合好的家数，没有个股明细，还原不到 2115 池；"
              "名字带 mkt_full_ 前缀以示区别，与固定池口径的数字不可直接比较。")

for _n, _d, _f in (
    ("mkt_full_flat_ratio", "全市场平盘家数占比", "flat_count/(up+down+flat)"),
):
    _register(_n, _d, _f + "；取每日 15:00 那一行", LIMIT_DEPS_DIST, _limit_breadth,
              warmup=40, note=_FULL_NOTE)

for _n, _d, _f in (
    ("mkt_limit_up_ratio", "池内涨停家数占比", "count(limit=U)/count(pool)"),
    ("mkt_limit_down_ratio", "池内跌停家数占比", "count(limit=D)/count(pool)"),
):
    _register(_n, _d, _f, LIMIT_DEPS_LIST, _limit_eco, start=START_LIMIT_LIST, warmup=40,
              note="只统计冻结池内的股票；limit 三态 U/D/Z 中只取 U涨停/D跌停。")

for _n, _d, _f in (
    ("mkt_mf_large_breadth", "池内大单净流入家数占比", "count(lg+elg净额>0)/count(有效)"),
    ("mkt_mf_elg_turnover_share", "池内超大单成交额占四档比", "sum(elg买+卖)/sum(四档买+卖)"),
):
    _register(_n, _d, _f, MF_DEPS, _money_flow, warmup=80,
              note="资金流金额单位是万元，只做同表内部比率、不跨表相除；"
                   "不使用 net_mf_amount 字段（实测其语义无法从源表自证），净额由四档原始列构造。")

for _n, _d, _f in (
    ("mkt_margin_breadth", "池内融资净买入家数占比", "count(rzmre>rzche)/count(有效)"),
):
    _register(_n, _d, _f, MARGIN_DEPS, _margin, warmup=80,
              lagged=("stock_margin_detail",), note=_LAG1)

for _n, _d, _f in (
    ("mkt_sector_dispersion", "行业板块当日收益的横截面离散", "std(ret_ind)"),
    ("mkt_sector_mom20_dispersion", "行业板块 20 日收益离散", "std(ret20_ind)"),
):
    _register(_n, _d, _f + "；行业候选经 dc_blocks「行业板块」名称交叉核验",
              SECTOR_DEPS, _sector_cs, warmup=80,
              note="板块指数层口径（不需要成分股映射，本地没有历史成分表）；"
                   "收益由板块指数点位现算；不使用总市值/涨跌家数这类统计序列当行业。")

for _n, _d, _f in (
    ("mkt_idx_csi300_amplitude20", "沪深300 20 日平均振幅", "mean((high-low)/pre_close,20)"),
    ("mkt_idx_broad_dispersion", "宽基指数间收益离散", "std(ret1) 于 9 个宽基指数"),
    ("mkt_idx_csi300_range_position250", "沪深300 在 250 日区间中的位置",
     "(close-min250)/(max250-min250)"),
):
    _register(_n, _d, _f, IDX_DEPS, _index_sentiment, warmup=300,
              note="指数按官方编制范围，不限制为固定池；均为价格指数，收益不含分红。")

for _n, _d, _f in (
    ("mkt_no_trade_ratio", "池内无真实成交股票占比", "1-count(vol>0)/count(pool)"),
):
    _register(_n, _d, _f, LIQ_DEPS, _liquidity, warmup=80)
