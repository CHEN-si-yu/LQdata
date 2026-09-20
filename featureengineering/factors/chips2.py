"""筹码分布深挖（chip distribution deep-dive）—— 18 个因子，全部 group="chip"。

本文件是 `factors/chips.py`（注册 16 个 chip 因子；其 docstring 里的任务清单预算
写的是 24，本文件不引用那个数）的**第二批**：不重复它已用的构造套路
（获利盘水平的 20 日变化、宽度的 20 日变化、现价/均价比值…），只补
「还没被度量过的维度」——成本分布的**偏度/偏度比**、三类成本中枢的**距离**、
主峰的**动态**、以及**尾部与深套**结构。

──────────────────────────────────────────────────────────────────────────
★ 口径（与 chips.py 完全一致，前三条不重复论证、只重申结论，第 4 条是本批新增）
──────────────────────────────────────────────────────────────────────────
1. **未复权 close**。本摘要表的 mean / p10..p90 / below_close 全部建立在**未复权**
   价上（chips.py 的模块 docstring 有实测证明：below_close≈0.5 时 close/p50≈1.01）。
   所有「现价 vs 筹码价」的比较一律用 `ctx.px("close")`，**绝不用** `ctx.hfq(...)`。
   参考库用 `_close_adj_basis(daily)`（复权）是因为**它们的** cyq 成本价是复权口径；
   本项目照抄会把口径弄反（实测会把比值放大两个数量级），这是最重要的一条偏离。
2. **分位点映射**（本表只有 5 个加权分位点，按最近分位映射）：
       cost_5pct→p10   cost_15pct→p25   cost_50pct→p50
       cost_85pct→p75  cost_95pct→p90   weight_avg→mean   cost_median→p50
   宽度类因子覆盖的是 **80% 筹码区间**（参考库 90%），数值略小、截面排序同构。
   参考库的分位点是 `searchsorted`（取档位价），本表是 `np.interp`（线性插值），
   两者有零点几个百分点的差异，不影响截面排序。
3. **起点 `start="2018-01-02"`**（`stock_cyq_chips` 真实起点），全部显式写死；
   截面小：本家族只覆盖**主板**（筹码表 2026 实测日均 3187 只，本批因子日均落格
   3047 只 ≈ 筹码表自身的 96%，其余被引擎的停牌/上市窗口挡掉），非空率天然低于
   其它家族，是**数据源覆盖**造成的，不是实现缺陷。
4. **值的符号约定**（与 chips.py 一致）：**水平类**因子返回原始量，方向由
   `higher_is_better` 标注（引擎的 rank 始终是「值大在前」）。**变化类**分两种：
   ① 「收敛 / 尾部风险」型取相反数，让**正 = 分布改善**（`cost_convergence_signal`、
   `chip_tail_risk_change`，与 chips.py 的 `chip_concentration_change_20d`
   = −diff(width) 同一约定）；② 「动量」型返回**原始差分**（`chip_peak_growing`
   = diff(peak_purity,5)、`chip_below_momentum` = diff(below_close,5)）。
   ⚠️ 另有 2 个（`chip_peak_ratio`、`chip_below_momentum`）的 `higher_is_better`
   与参考库的 rank 符号**相反**：改按本地 chip 家族对「获利盘」这一信号的既有立场
   标注（同一列的 `winner_rate`、同信号二阶的 `winner_rate_acceleration` 都是
   False = 高获利盘是兑现压力）。这是**纯文档口径**偏离，不改变任何数值。

──────────────────────────────────────────────────────────────────────────
★ 本批 18 个里，有 4 个与 `chips.py` 已注册因子是**同一列/代数等价**（实测 ρ=±1.0000）
──────────────────────────────────────────────────────────────────────────
用户点名要这 18 个，故全部实现；但下游做共线性筛选/因子数压缩时，
**这 4 个应当直接删掉、保留 `chips.py` 里的那个**（同一列信息，多存一份）：

    cost_displacement        ≡ −avg_cost_premium     （同一分子分母，只差负号）
    cost_distribution_width  ≡  chip_concentration   （就是 width 字段本身）
    chip_mean_distance       ≡  avg_cost_premium 的严格单调变换
                                （(close−mean)/close = 1 − 1/(1+premium)）
    chip_peak_ratio          ≡  winner_rate         （就是 below_close 字段本身）

另有 2 个是第一家族的**近似**重复（保留与否由主 Agent 定，见 §实测共线性）：
    chip_median_distance     ~ avg_cost_premium      ρ = +0.9150
    cost_convergence_signal  ~ chip_concentration_change_20d  ρ = +0.9241

★ 反过来，有 3 个用的是**首批 16 个因子从未碰过的字段/构造**（不是重复列）：
    `cost_distribution_skew`（摘要表的 `skew` 字段，本地无人用过）
    `chip_tail_risk` / `chip_tail_risk_change`（摘要表的 `kurt` 字段）
    `cost_skew_ratio`（p10/p50/p90 的**不对称**比，首批全是 (p90−p10) 或 (p75−p25) 对称口径）

──────────────────────────────────────────────────────────────────────────
★ 4 个因子必须回读原始筹码档位（本文件唯一的架构偏离，务必看）
──────────────────────────────────────────────────────────────────────────
`chip_deep_trap_ratio` / `chip_high_float_ratio` / `chip_win_peak_frac` /
`chip_loss_peak_frac` 需要**任意价格点的 CDF 或「现价上方/下方的最大单档占比」**，
而 `data/derived/chips` 摘要表只有 5 个分位点 + 上下方**总量**，插值不可行 ——
实测 2026 全年 54.2 万格（摘要表与 stock_daily 的 close 逐格对齐）：

  · 只有 **6.4%** 的格子能让 0.9×close 与 1.1×close **同时**落在 [p10, p90] 内；
  · **70.8%** 的格子 0.9×close 已经跌破 p10（下探针必须外推）；
  · **47.4%** 的格子 1.1×close 已经高过 p90（上探针必须外推）；
  · close 本身高于 p75 的格占 22.7%、高于 p90 的占 9.9%、低于 p25 的占 29.1%。

即：两个阈值探针在九成以上的格子上都落在摘要表覆盖区间之外，插值/外推等于凭空
捏造，**不能近似**。

因此这 4 个走 `ctx.dataset("stock_cyq_chips", ...)` 自己扫一遍原始分区，
一次扫描同时产出 6 个聚合量（下方/上方的 sum 与 max、0.9/1.1 倍现价的 CDF、
以及归一化分母），4 个因子共享同一份结果（模块级 `_AGG` 缓存，按年份键）：

  · 实测 1 个年份分区（2026：**5553 万行 × 4 列**）：读 **3.7s** + 聚合 **44.1s**
    （单进程，含 categorical 编码、复合键、一次 argsort、bincount/reduceat）；
  · 峰值 RSS：基线 0.20 GB（引擎+面板）→ 读表后 3.83 GB → 聚合 **8.27 GB**
    （峰值来自分组期的几个 440 MB 级 int64/float64 临时量）；
  · 聚合结果只有 54.2 万行 × 4 列（几十 MB），按年缓存后可反复使用；
  · **扫完立刻把 3.83 GB 的原始表从 `Upstream` 缓存里摘掉**
    （不摘的话 9 个年份分区会在一个 worker 里累积 ~34 GB，
     4 个 worker 就是 ~137 GB → 顶穿 100 GB 上限）。

★ **正确的做法是引擎扩展**（本 Agent 无权改 `fea/**`）：在 `fea/chips.py` 的
  `build_year` 里补 4 列 `below_90pct_close / above_110pct_close /
  below_peak_frac / above_peak_frac`（都是一次遍历里顺手算出来的量），
  这 4 个因子就退化成和首批 16 个一样的 O(1) 读取，本文件的整个 `_AGG` 段
  可以直接删掉。见汇报的「需要引擎扩展的地方」。

──────────────────────────────────────────────────────────────────────────
★ PIT / 未来信息
──────────────────────────────────────────────────────────────────────────
· 筹码层是**当日快照**（当日收盘后的持仓成本分布），天然 PIT；
· `stock_cyq_chips` **不在** `fea/delay.py` 的滞后表名单里 → 不需要 `lag_grid`；
· 所有 `ctx.diff(...)` 的 k 均为正（只看过去）；没有用到任何全样本统计量；
· 面板右端就是 T，未来一行都没读（`forward_days=0`）。

──────────────────────────────────────────────────────────────────────────
★ 实测共线性（2026-01-05 ~ 2026-09-14，170 个交易日，逐日截面 Spearman 取中位）
──────────────────────────────────────────────────────────────────────────
对手方 = 本地**生产因子** `data/factors/<name>/year=2026`，或摘要表**字段本身**
（当该列/构造没有对应生产因子时，标「字段」）。逐日计算、只取当日可比格 ≥ 100 的
交易日、再取中位；全部为本次实跑数字，无估计值。

  |ρ| ≥ 0.9（同一列或近重复，下游必看）：
    cost_displacement        ~ avg_cost_premium                −1.0000  ← 同一列（差负号）
    cost_distribution_width  ~ chip_concentration              +1.0000  ← 同一列 width
    chip_mean_distance       ~ avg_cost_premium                +1.0000  ← 单调变换
    chip_peak_ratio          ~ winner_rate                     +1.0000  ← 同一列 below_close
    chip_median_distance     ~ chip_support_distance           +0.9300
    cost_convergence_signal  ~ chip_concentration_change_20d   +0.9241
    chip_median_distance     ~ avg_cost_premium                +0.9150
    chip_deep_trap_ratio     ~ chip_resistance_distance        +0.9025
    chip_deep_trap_ratio     ~ avg_cost_premium                −0.8978  ← 略低于 0.9，同源
  0.6 ~ 0.9：
    chip_deep_trap_ratio     ~ chip_position                   −0.8429
    cost_skew_ratio          ~ cost_distribution_skew          +0.8125
    chip_below_momentum      ~ winner_rate_acceleration        +0.7781
    chip_deep_trap_ratio     ~ above_close 字段                +0.7533（= −winner_rate）
    chip_loss_peak_frac      ~ chip_cr3_factor                 +0.7284
    chip_deep_trap_ratio     ~ chip_support_distance           −0.7022
    chip_win_peak_frac       ~ peak_purity 字段                +0.6942
    chip_win_peak_frac       ~ chip_cr3_factor                 +0.6847
    chip_high_float_ratio    ~ chip_support_distance           +0.6531
    chip_peak_distance       ~ avg_cost_premium                +0.6439
    chip_high_float_ratio    ~ avg_cost_premium                +0.6315
  < 0.6（本批新增维度主要落在这一档）：
    chip_high_float_ratio ~ winner_rate        +0.5612 ｜ chip_high_float_ratio ~ below_close 字段 +0.5612
    chip_win_peak_frac    ~ below_close 字段   −0.5300 ｜ chip_loss_peak_frac ~ chip_deep_trap_ratio −0.5112
    chip_loss_peak_frac   ~ above_close 字段   −0.4604 ｜ chip_below_momentum  ~ below_close 字段  +0.4565
    chip_win_peak_frac    ~ chip_deep_trap_ratio +0.4252 ｜ chip_deep_trap_ratio ~ chip_high_float_ratio −0.4155
    cost_convergence_signal ~ width 字段       −0.3785 ｜ cost_skew_ratio ~ skew 字段 −0.2731
    chip_peak_growing     ~ peak_purity 字段   +0.2082 ｜ chip_tail_risk_change ~ chip_tail_risk −0.2030
    cost_distribution_skew ~ skew 字段         −0.1528 ｜ chip_above_below_ratio ~ winner_rate +0.1150
    chip_tail_risk        ~ chip_cost_kurtosis_20d −0.0970 ｜ chip_median_distance ~ p50 字段 −0.0805
    chip_peak_distance    ~ mode 字段          −0.0684 ｜ chip_below_momentum ~ chip_peak_growing +0.0252
（`chip_tail_risk_change` 的值已取「正 = 峰度下降」，故与 `chip_tail_risk` 的相关是
  **负**号；若按原始 diff(kurt) 看，符号反过来。表中「4 个原始档位因子 vs 首批 16 个」
  的 64 对是逐对全测的，其余按需测。）

★ 这张表要读的两件事：
 1. 与首批的 4 个重复列（上表 ρ=±1.0000）**建议下游直接删这 4 个**；
 2. 本批 4 个原始档位因子里，`chip_high_float_ratio` / `chip_loss_peak_frac` /
    `chip_win_peak_frac` 对首批 16 个的最大 |ρ| 分别只有 0.6531 / 0.7284 / 0.6847，
    确实是新维度；**但 `chip_deep_trap_ratio` 不是** —— 它 ~chip_resistance_distance
    +0.9025、~avg_cost_premium −0.8978，因为「现价上方 10% 以外的筹码占比」本质上
    就是「现价相对筹码成本有多低」的另一个说法。要压因子数就删它。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fea.spec import FactorSpec, register

# ── 起点：stock_cyq_chips 的真实起点（本家族硬约束，见模块 docstring）
CHIP_START = "2018-01-02"

# ── 依赖：读摘要表的因子只依赖筹码层；用到现价的因子同时声明 stock_daily
DEPS = ("stock_cyq_chips",)
DEPS_PX = ("stock_cyq_chips", "stock_daily")

# ── warmup_days：窗口 N 个交易日 → N×1.8+20（与 chips.py 同一套口径）
W_FIELD = 20      # 无窗口（纯当日字段）
W5 = 30           # 5 日窗口

# ── 分母地板：价格类分母（元）。见 chips.py 的「分母保护」
MIN_PRICE = 1e-6


# ══════════════════════════════════════════════════════════════════════════
# 一、内部工具（摘要表侧）
# ══════════════════════════════════════════════════════════════════════════
def _close(ctx) -> np.ndarray:
    """**未复权** close（与筹码档位同口径）。停牌日自动前向填充，但那天筹码行缺失，
    相乘/相除后仍是 NaN。"""
    return np.asarray(ctx.px("close"), dtype=np.float64)


def _wr(ctx) -> np.ndarray:
    """获利盘比例 = below_close ∈ [0,1]（只挡浮点噪声）。"""
    return np.clip(np.asarray(ctx.chip("below_close"), dtype=np.float64), 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════
# 二、原始筹码档位的按需聚合（4 个 CDF / 峰占比因子专用）
#
#     产出（每 (交易日, 股票) 一行，只保留主板且在面板里的股票）：
#       below_90        成本 < 0.9×close 的筹码占比   = F(0.9·close)
#       upper_110       成本 > 1.1×close 的筹码占比   = 1 − F(1.1·close)
#       win_peak_frac   现价**下方**筹码中最大单档占比（获利筹码的峰集中度）
#       loss_peak_frac  现价**上方**筹码中最大单档占比（套牢筹码的峰集中度）
#
#     ★ 与摘要表的一致性：分母用当日 Σpercent（原始和，未归一），
#       与 fea/chips.py 一样按当日总量归一 → 口径与 below_close 完全一致。
#     ★ 现价用 `stock_daily` 的**未复权 close**（与 fea/chips.py 的 close_map 同源）：
#       不用 `ctx.px("close")` 是因为后者停牌日前向填充，会在「有筹码行但当天没
#       成交」的格子上凭空造出值，而摘要表那里是 NaN。
# ══════════════════════════════════════════════════════════════════════════
_RAW_COLS = ("trade_date", "stock_code", "price", "percent")
_CODE_W = 100_000                      # 复合键位宽：code 下标 < 10 万（主板 ~3500）
_AGG: dict[int, dict | None] = {}      # year -> 聚合结果（None = 该年无分区）


def _ymd(s) -> int:
    """"2026-09-14" -> 20260914"""
    return int(s[:4]) * 10000 + int(s[5:7]) * 100 + int(s[8:10])


def _float_col(df: pd.DataFrame, name: str) -> np.ndarray:
    """数值列 → float64（上游 schema 是 double，直接 view；异常时退回 to_numeric）。"""
    try:
        return np.asarray(df[name], dtype=np.float64)
    except (TypeError, ValueError):
        return pd.to_numeric(df[name], errors="coerce").to_numpy(np.float64)


def _close_lookup(ctx, year: int, key_g: np.ndarray) -> np.ndarray:
    """按 (day, code) 复合键取**未复权 close**（stock_daily 当年分区）。"""
    sd = ctx.dataset("stock_daily", columns=["stock_code", "trade_date", "close"],
                     years=(year, year))
    if sd.empty:
        return np.full(key_g.shape, np.nan)
    day = ctx.date_col(sd["trade_date"])
    ci = ctx.code_index(sd["stock_code"].to_numpy())
    cl = _float_col(sd, "close")
    ok = (ci >= 0) & np.isfinite(cl) & (day > 0)
    if not ok.any():
        return np.full(key_g.shape, np.nan)
    k = day[ok].astype(np.int64) * _CODE_W + ci[ok]
    uk, first = np.unique(k, return_index=True)
    uv = cl[ok][first]
    pos = np.searchsorted(uk, key_g)
    pos_c = np.clip(pos, 0, uk.size - 1)
    hit = uk[pos_c] == key_g
    return np.where(hit, uv[pos_c], np.nan)


def _aggregate(df: pd.DataFrame, ctx, year: int) -> dict:
    """把一个年份分区的原始档位聚合成小表（全向量化：categorical + 单次 argsort
    + bincount/reduceat）。见模块 docstring 的耗时/内存实测。"""
    panel = ctx.panel
    # 日期 / 代码 → 小整数编码。★ 用 `astype("category")`（Arrow 字典编码，实测
    # 4.4s/3.7s）而不是 `pd.Categorical(..., categories=<5300 个字符串>)`（26s）——
    # 后者会对 5500 万行做 Python 级的字符串匹配。
    dc = df["trade_date"].astype("category")
    cc = df["stock_code"].astype("category")
    day_cat = np.array([_ymd(s) for s in dc.cat.categories], dtype=np.int32)
    day = day_cat[dc.cat.codes.to_numpy(np.int32)]
    cidx_cat = ctx.code_index(np.asarray(cc.cat.categories, dtype=object))
    cidx = cidx_cat[cc.cat.codes.to_numpy(np.int32)]
    price = _float_col(df, "price")
    pct = _float_col(df, "percent")

    ok = (np.isfinite(price) & (price > 0) & np.isfinite(pct) & (pct >= 0)
          & (cidx >= 0))                     # cidx<0 = 不在面板（非主板）
    day, cidx, price, pct = day[ok], cidx[ok], price[ok], pct[ok]
    if day.size == 0:
        return {"day": np.zeros(0, np.int32), "code": np.zeros(0, np.int32),
                **{f: np.zeros(0) for f in _AGG_FIELDS}}

    # 分段：按 (日期, 股票) 排序后找边界。复合键 = day×1e5 + code 下标。
    key = day.astype(np.int64) * _CODE_W + cidx
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.concatenate(([True], ks[1:] != ks[:-1])))
    ends = np.concatenate((starts[1:], [ks.size]))
    G = starts.size
    gid = np.repeat(np.arange(G, dtype=np.int64), ends - starts)

    day_g = day[order][starts]
    code_g = cidx[order][starts].astype(np.int32)
    close_g = _close_lookup(ctx, year, ks[starts])
    close_r = close_g[gid]                       # 逐行的现价（未复权）
    pm = pct[order]                              # 重排后的权重
    pr = price[order]

    below = pr <= close_r
    above = pr > close_r
    m90 = pr < 0.9 * close_r
    m110 = pr > 1.1 * close_r

    tot = np.bincount(gid, weights=pm, minlength=G)
    bsum = np.bincount(gid, weights=pm * below, minlength=G)
    asum = np.bincount(gid, weights=pm * above, minlength=G)
    b90 = np.bincount(gid, weights=pm * m90, minlength=G)
    a110 = np.bincount(gid, weights=pm * m110, minlength=G)
    neg = -np.inf
    bmax = np.maximum.reduceat(np.where(below, pm, neg), starts)
    amax = np.maximum.reduceat(np.where(above, pm, neg), starts)

    bad = ~np.isfinite(close_g) | (tot <= 0.0)   # 没现价 / 该股当日无有效档位
    safe_tot = np.where(tot > 0, tot, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "below_90": np.where(bad, np.nan, b90 / safe_tot),
            "upper_110": np.where(bad, np.nan, a110 / safe_tot),
            "win_peak_frac": np.where((bsum > 0) & ~bad, bmax / bsum, np.nan),
            "loss_peak_frac": np.where((asum > 0) & ~bad, amax / asum, np.nan),
        }
    return {"day": day_g.astype(np.int32), "code": code_g, **out}


_AGG_FIELDS = ("below_90", "upper_110", "win_peak_frac", "loss_peak_frac")


def _year_aggregate(ctx, year: int) -> dict | None:
    """按年缓存的聚合结果（None = 该年没有分区）。"""
    if year in _AGG:
        return _AGG[year]
    df = ctx.dataset("stock_cyq_chips", columns=list(_RAW_COLS), years=(year, year))
    agg = None
    if not df.empty:
        agg = _aggregate(df, ctx, year)
    del df
    # ★ 立刻把原始大表从 Upstream 缓存里摘掉：不摘的话 worker 进程会按年份累积
    #   （9 个分区 × 3.83 GB ≈ 34 GB/worker，实测；4 个 worker 就是 ~137 GB）。
    #   键的格式与 fea/upstream.py 的 `read` 一致。
    try:
        ctx.up._cache.pop(("stock_cyq_chips", tuple(_RAW_COLS), (year, year)), None)
    except Exception:                                   # 缓存实现变了也不该让因子挂掉
        pass
    if len(_AGG) > 12:
        _AGG.clear()
    _AGG[year] = agg
    return agg


_GRID: dict[tuple, np.ndarray] = {}


def _raw_grid(ctx, field: str) -> np.ndarray:
    """把某年的聚合结果铺到当前面板 (T, C) 上（精确落格：缺则 NaN）。"""
    panel = ctx.panel
    key = (int(panel.dates[0]), int(panel.dates[-1]), panel.C, field)
    hit = _GRID.get(key)
    if hit is not None:
        return hit
    T, C = panel.shape
    out = np.full((T, C), np.nan, dtype=np.float64)
    y0, y1 = int(str(panel.dates[0])[:4]), int(str(panel.dates[-1])[:4])
    for y in range(y0, y1 + 1):
        agg = _year_aggregate(ctx, y)
        if agg is None or agg["day"].size == 0:
            continue
        pos = np.searchsorted(panel.dates, agg["day"])
        ok = (pos < T) & (panel.dates[np.clip(pos, 0, T - 1)] == agg["day"])
        if not ok.any():
            continue
        out[pos[ok], agg["code"][ok]] = agg[field][ok]
    if len(_GRID) > 8:
        _GRID.clear()
    _GRID[key] = out
    return out




@register(FactorSpec(
    name="cost_distribution_skew", group="chip", deps=DEPS,
    desc="成本分布偏度（分位点口径）= (p50 − p25) / (p75 − p50)（<1 = 上方尾部更长/右偏）",
    formula='cyq = context.load("cyq_perf.parquet"); left_tail = cyq["cost_50pct"] - cyq["cost_15pct"]; '
            'right_tail = cyq["cost_85pct"] - cyq["cost_50pct"]; '
            'skew = left_tail / right_tail.replace(0, np.nan); return cross_sectional_rank(skew)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=True,
    note="★ 参考库 `factors/chip_deep.py` 的 cost_distribution_skew = (cost_50−cost_15)/(cost_85−cost_50)，"
         "1:1 照搬（分位映射 15/50/85 → p25/p50/p75）。"
         "⚠️ 本地**没有** `chip_cost_skew` / `chip_cost_asymmetry` 这两个因子（首批 16 个"
         "都没注册）→ **不是重复列**，对本地是新信息。"
         "★ 与摘要表的 `skew` 字段**不是一回事**：`skew` 是加权三阶标准矩"
         "（2026 实测：中位 1.2765、p1 −4.63、p99 11.98、min −8.85、max 16.40），"
         "本因子只用两个分位差（对离群价稳健）—— 实测两者 ρ = −0.1528，确实不是同一量。"
         "★ 值域无界：p75−p50 → 0 时发散（2026 实测 [0, 76.79]）→ 截面 winsor 由引擎做。",
))
def cost_distribution_skew(ctx):
    lower = ctx.chip("p50") - ctx.chip("p25")
    upper = ctx.chip("p75") - ctx.chip("p50")
    return ctx.safe_div(lower, upper, min_abs_den=MIN_PRICE)




@register(FactorSpec(
    name="cost_skew_ratio", group="chip", deps=DEPS,
    desc="成本偏度比率 = (p50 − p10) / (p90 − p50)（>1 = 下方尾部更长/左偏）",
    formula='cyq = context.load("cyq_perf.parquet"); lower_range = cyq["cost_50pct"] - cyq["cost_5pct"]; '
            'upper_range = cyq["cost_95pct"] - cyq["cost_50pct"]; skew = safe_divide(lower_range, upper_range + 1e-10); '
            'skew = skew.clip(0.1, 10); return cross_sectional_rank(skew)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=True,
    note="★ 参考库 `factors/chip_cost_extended.py` 公式 = (cost_50−cost_5)/(cost_95−cost_50)，"
         "1:1 照搬（分位映射 5/50/95 → p10/p50/p90）。"
         "⚠️ 本地**没有** `chip_cost_asymmetry` / `chip_cost_skew`（首批 16 个未注册）→ 不是重复列。"
         "★ 与 cost_distribution_skew 的区别在分位点：本因子用 5/50/95（覆盖 90% 区间），"
         "那个用 15/50/85（覆盖 70% 区间），实测两者 ρ = +0.8125（高度相关但不等价）。"
         "★ 参考库的 `clip(0.1, 10)` 未保留（winsor 由引擎统一做）；"
         "2026 实测值域 [0, 64.67]（分母 p90−p50 → 0 时发散）。",
))
def cost_skew_ratio(ctx):
    lower = ctx.chip("p50") - ctx.chip("p10")
    upper = ctx.chip("p90") - ctx.chip("p50")
    return ctx.safe_div(lower, upper, min_abs_den=MIN_PRICE)


@register(FactorSpec(
    name="cost_convergence_signal", group="chip", deps=DEPS,
    desc="成本收敛信号 = (p90 − p10) 的 20 个交易日**变化率**（收敛=筹码向成本中枢凝聚）",
    formula='cyq = context.load("cyq_perf.parquet"); width = cyq["cost_95pct"] - cyq["cost_5pct"]; '
            'chg = width.groupby(level="Code").transform(lambda s: s.pct_change(20, fill_method=None)); '
            'chg = chg.clip(-1, 1); return cross_sectional_rank(-chg)',
    start=CHIP_START, warmup_days=60, higher_is_better=True,
    note="★ 出处：参考库 `factors/chip_cost_extended.py`。"
         "值是**收敛量**（= 参考库的 −chg，正 = 宽度收窄），与 chips.py 的 "
         "`chip_concentration_change_20d`（= −diff(width)）同一约定 —— 参考库 rank(−chg) "
         "「分布收窄=筹码集中排前」的方向由 higher_is_better=True 表达，数值不取反两次。"
         "★ 与 `chip_concentration_change_20d` 的差别是**差分口径**：本因子是**绝对价差** "
         "(p90−p10) 的 20 日**变化率**（pct_change，无量纲、自动按股价水平归一），"
         "那个是**归一化宽度** width=(p90−p10)/p50 的 20 日**差分**（带 p50 量纲）。"
         "两者都度量「宽度在收窄」，实测逐日截面 ρ = +0.9241 —— **高度共线**，"
         "下游若要压因子数，这两个里留一个即可（本因子对低价股的绝对价差变动更敏感、"
         "对股价水平不敏感，那个相反）。"
         "★ 参考库的 `clip(-1, 1)` 未保留（winsor 由引擎统一做）：去掉后值域是 "
         "「宽度放大 >100% 时为负」——2026 实测 value ∈ [−11.65, +0.937]、中位 0.0037、"
         "5.24% 的格子 < −1（那 5% 恰好是参考库 clip 掉的部分，截面排序不受影响，"
         "因为 rank 只关心顺序）。"
         "★ 2018 年开头约 20 个交易日为 NaN（20 日窗口要读到 2017 年，上游无数据）。",
))
def cost_convergence_signal(ctx):
    spread = ctx.chip("p90") - ctx.chip("p10")
    return -ctx.pct_change(spread, 20, min_abs_den=MIN_PRICE)


# ══════════════════════════════════════════════════════════════════════════
# 四、距离族（3 个）—— 现价对三类成本中枢的**相对距离**，全部用未复权 close
#     参考库对三者都用 rank(+distance)（价格在成本中枢上方 = 多数持仓者盈利）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="chip_median_distance", group="chip", deps=DEPS_PX,
    desc="中位数成本距离 = (现价 − p50) / 现价（正=过半持仓者盈利）",
    formula='median_series = _compute_chip_factor(..., "chip_median_price"); '
            'close = daily_panel["close"]; common = close.index.intersection(median_series.index); '
            'distance = (close.loc[common] - median_series.loc[common]) / close.loc[common].replace(0, np.nan); '
            'return cross_sectional_rank(distance)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=True,
    note="★ 出处：参考库 `factors/chip_deep.py`。口径偏离：参考库用复权 close 与复权口径的 chip_median_price，本因子两边都用"
         "**未复权**（摘要表的 p50 就是未复权价，见模块 docstring 口径 1）。"
         "★ 与 avg_cost_premium 的关系：都是「现价 vs 成本中枢」的归一，但中枢不同"
         "（p50 vs 加权 mean）、分母不同（close vs mean）→ **没有任何代数关系**。"
         "实测逐日截面 ρ(·, avg_cost_premium) = +0.9150、ρ(·, chip_support_distance) = +0.9300"
         "（chips.py 2019 年实测 0.974 / 0.979，同一量级）→ **高度共线**。"
         "chips.py 曾以 0.974 为由排除它（预算耗尽），本批按用户清单实现；"
         "下游若压因子数，这个与 avg_cost_premium 二选一。",
))
def chip_median_distance(ctx):
    close = _close(ctx)
    return ctx.safe_div(close - ctx.chip("p50"), close, min_abs_den=MIN_PRICE)




@register(FactorSpec(
    name="chip_peak_distance", group="chip", deps=DEPS_PX,
    desc="主峰距离 = (现价 − 众数价) / 现价（正=现价在最大筹码峰上方=有支撑）",
    formula='peak_series = _compute_chip_factor(..., "chip_peak_price"); '
            'close_adj = _close_adj_basis(daily_panel); common = close_adj.index.intersection(peak_series.index); '
            'distance = (close_adj.loc[common] - peak_series.loc[common]) / close_adj.loc[common].replace(0, np.nan); '
            'return cross_sectional_rank(distance)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=True,
    note="★ 出处：参考库 `factors/chip_deep.py`。`chip_peak_price` = **percent 最大的那一档的价格**，对应摘要表的 `mode`"
         "（同一个定义，见 fea/chips.py 的 build_year）。★ 本因子是三个「距离」里"
         "**唯一**与 avg_cost_premium 共线度低的一个（实测 ρ = +0.6439；另两个是 "
         "+0.9150 / +1.0000）：众数价是分布的**局部**特征（对档宽与单档权重敏感），"
         "均值/中位数是**整体**特征。实测 ρ(·, mode 字段) = −0.0684（该字段本身没被"
         "任何因子用过，这里只作为定义核对）。"
         "★ 口径：未复权 close（同 chip_median_distance）。",
))
def chip_peak_distance(ctx):
    close = _close(ctx)
    return ctx.safe_div(close - ctx.chip("mode"), close, min_abs_den=MIN_PRICE)




@register(FactorSpec(
    name="chip_peak_growing", group="chip", deps=DEPS,
    desc="主峰增强 = 主峰纯度（peak_purity）的 5 个交易日变化（升=筹码向核心价位凝聚）",
    formula='s = _compute_chip_factor(..., "chip_peak_dominance"); '
            'chg = s.groupby(level="Code").transform(lambda x: x.diff(5)); '
            'return cross_sectional_rank(chg)',
    start=CHIP_START, warmup_days=W5, higher_is_better=True,
    note="★ 出处：参考库 `factors/chip_deep.py`。`chip_peak_dominance` = 最大单档权重占比，对应摘要表的 `peak_purity` 字段"
         "（参考库另有 `chip_peak_purity` = 该量的**水平**；本地首批 16 个因子没有注册它，"
         "`peak_purity` / `skew` / `kurt` 这几个字段都是本批第一次用）。"
         "★ 主峰纯度的水平受**股价水平**影响（档宽固定 0.1/0.01 元时，股价越高单档占比越小），"
         "做 5 日**差分**正好把这个个体固定效应消掉：实测 ρ(·, peak_purity 字段) = +0.2082"
         "（2026 逐日截面中位）→ 与水平近乎正交，是比水平更干净的一版。"
         "★ 契约的 NaN 策略：窗口内出现 NaN 即 NaN（停牌日筹码行缺失 → 该日 NaN）。",
))
def chip_peak_growing(ctx):
    return ctx.diff(ctx.chip("peak_purity"), 5)




@register(FactorSpec(
    name="chip_above_below_ratio", group="chip", deps=DEPS_PX,
    desc="筹码压力比 = (现价 − p75) / (p25 − 现价)（正 ⟺ 现价落在 p25~p75 带内，绝对值越大越靠近 p25）",
    formula='close_adj = _close_adj_basis(daily); cost_85 = cyq["cost_85pct"]; cost_15 = cyq["cost_15pct"]; '
            'above = close_adj.loc[common] - cost_85.loc[common]; '
            'below = cost_15.loc[common] - close_adj.loc[common]; '
            'ratio = safe_divide(above, below); return cross_sectional_rank(-ratio)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=False,
    note="★ 出处：参考库 `factors/chip_extended.py`（Class 1，基于 cyq_perf）。"
         "定义辨析（重要）：这是**距离比**（现价距上方成本带 vs 距下方成本带），"
         "**不是**筹码质量比。实测 ρ(·, winner_rate) = +0.1150（几乎正交）——"
         "所谓「上方套牢>下方获利」的说法在这个公式里并不成立，它**不是**获利盘因子的重述。"
         "chips.py 当年因预算耗尽（清单第 29 位）没实现它，本批补上。cost_15/85pct → p25/p75。"
         "★ **符号的真实几何**（2026 全年 54.2 万格逐格验证，一致率 100.0000%）："
         "值 > 0 ⟺ 现价落在 [p25, p75] 带内；带内时值随现价单调**递减**"
         "（在 p75 处 → 0，越靠近 p25 越大，p25 上发散）；值 < 0 ⟺ 现价已**突破 p75**"
         "（上方无近端筹码）或已**跌破 p25**（下方筹码全部套牢）。"
         "2026 实测：48.2% 的格为正、51.8% 为负。"
         "★ 正因为有符号翻转 + p25 处的极点，原始值**无界**（2026 实测 [−1.1e5, +2.2e5]），"
         "参考库取 `rank(−ratio)` 也压不住 —— 本因子只靠引擎的截面 winsor；"
         "下游若直接用原始值做线性合成，先看它被 winsor 后的分布。"
         "★ 参考库对 close 用复权价，本因子用未复权（口径 1）；"
         "方向标注按参考库 `rank(−ratio)`（值大=压力大排后）→ higher_is_better=False。",
))
def chip_above_below_ratio(ctx):
    close = _close(ctx)
    above = close - ctx.chip("p75")
    below = ctx.chip("p25") - close
    return ctx.safe_div(above, below, min_abs_den=MIN_PRICE)


# ══════════════════════════════════════════════════════════════════════════
# 六、尾部 / 深套族（6 个）
#     前 2 个读摘要表（kurt 字段），后 4 个扫原始档位（见模块 docstring）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="chip_tail_risk", group="chip", deps=DEPS,
    desc="筹码尾部风险 = 成本分布的超额峰度（高=极端价位筹码堆积=肥尾）",
    formula='s = _compute_chip_factor(..., "chip_kurtosis"); return cross_sectional_rank(-s)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=False,
    note="★ 出处：参考库 `factors/chip_deep.py`（Class 2，读 cyq_chips 原始档位）。"
         "摘要表的 `kurt` 字段 = 加权四阶中心矩 − 3（超额峰度），与参考库的 "
         "`chip_kurtosis` 逐字同口径（都是 Σw·(p−μ)^4/σ^4 − 3）——**本批之前没有任何"
         "因子用过这个字段**。2026 实测：中位 13.44、p99 169.05、min −1.9529、"
         "max 418.34，恒 > −3（chips.py 2019 年实测中位数 25、p99 164，量级一致 ✓）。"
         "★ 注意与 chips.py 的 `chip_cost_kurtosis_20d` 区分：那个是分位点代理"
         "(p75−p25)/(p90−p10)，是**有界**的比值；本因子是四阶矩，量纲为 1 但值域极宽"
         "（2026 实测 [−1.95, 418]）→ 引擎的 1%/99% 截面 winsor 后仍有少数极端值，"
         "这是**原口径**（参考库也没截断）。实测两者 ρ = −0.0970（几乎正交，不是同一个量）。"
         "★ 参考库的 `chip_kurtosis` 在 std≈0 时取 −3，本表同情形给 NaN（fea/chips.py "
         "在 sd>0 时才写 kurt）—— 2026 实测摘要表 `n_levels` 最小 6、中位 101，"
         "单档筹码的格子 **0 个**，无影响。",
))
def chip_tail_risk(ctx):
    return ctx.chip("kurt")




@register(FactorSpec(
    name="chip_deep_trap_ratio", group="chip", deps=DEPS_PX,
    desc="深套筹码占比 = 成本 > 1.1×现价 的筹码比例（高=上方深度套牢盘重）",
    formula='s = _chip_series(context, "chip_upper_110", need_close=True); '
            'return cross_sectional_rank(-s)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=False,
    note="★ 出处：参考库 `factors/chip_deep_extra.py`。摘要表算不出（需要任意价格点的 CDF），本因子扫原始档位精确计算："
         "Σpercent(price > 1.1×close)，分母为当日 Σpercent（与 below_close 同口径归一）。"
         "★ 为什么不能插值：摘要表只有 p10..p90 五个点，而 **1.1×close 在 47.4% 的格子上"
         "已经高过 p90**（0.9×close 更糟：70.8% 的格子跌破 p10；两个探针同时落在区间内的"
         "只有 6.4%，见模块 docstring 的探针几何）→ 插值／外推等于凭空捏造。"
         "chips.py 的 docstring 第 6 条也是用「摘要表算不出任意价格点的 CDF」为由"
         "排除了同类因子。"
         "★ 与 `above_close`（**全部**上方筹码占比，摘要表字段）的区别：本因子聚焦"
         "**深度**套牢区（>10%），实测 ρ(·, above_close 字段) = +0.7533、"
         "ρ(·, winner_rate) = −0.7533 —— 高度相关但不等价，深套区才是"
         "「解套抛压最重」的那部分。"
         "⚠️ 但它**不是**新维度：实测对首批 16 个的最大 |ρ| 达 0.9025"
         "（~chip_resistance_distance +0.9025、~avg_cost_premium −0.8978、"
         "~chip_position −0.8429、~chip_support_distance −0.7022）——"
         "「现价上方 10% 以外的筹码占比」本质上就是「现价相对成本有多低」的另一种说法，"
         "下游压因子数时应删掉本因子。"
         "★ 现价用 stock_daily 的未复权 close（与 fea/chips.py 的 close_map 同源），"
         "不用 ctx.px('close')（后者停牌前向填充，会造出摘要表里是 NaN 的格子）。",
))
def chip_deep_trap_ratio(ctx):
    return _raw_grid(ctx, "upper_110")


@register(FactorSpec(
    name="chip_high_float_ratio", group="chip", deps=DEPS_PX,
    desc="高浮盈筹码占比 = 成本 < 0.9×现价 的筹码比例（高=获利丰厚、兑现压力大）",
    formula='s = _chip_series(context, "chip_below_90", need_close=True); '
            'return cross_sectional_rank(-s)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=False,
    note="★ 出处：参考库 `factors/chip_deep_extra.py`。与 chip_deep_trap_ratio 同一次扫描产出（见模块 docstring）。"
         "Σpercent(price < 0.9×close)/Σpercent。"
         "★ 它**不是** winner_rate 的重述：winner_rate 是 F(close)（阈值恰好是现价），"
         "本因子是 F(0.9×close)（阈值下移 10%），只统计「深度获利」的那部分。"
         "实测 ρ(·, winner_rate) = +0.5612、ρ(·, avg_cost_premium) = +0.6315、"
         "ρ(·, chip_deep_trap_ratio) = −0.4155 —— 与获利盘**总量**只算中度相关；"
         "对首批 16 个因子的最大 |ρ| = 0.6531（~chip_support_distance）→ 确实是新维度。"
         "⚠️ chips.py 当年用 5 点插值**近似**算过同类量，实测 ρ(·, winner_rate) = 0.907，"
         "那是它排除该因子的理由；本因子是**精确版**（原始档位逐档比较），"
         "共线度掉到 0.56 —— 说明当年那个 0.907 里有一大半是插值误差造成的假共线。"
         "★ 2026 实测值域 [0, 0.9925]，中位 0.0138（绝大多数格子的 0.9×close 都在 p10 之下，"
         "见模块 docstring 的探针几何）→ 它是个**右偏**的稀疏量，靠引擎 winsor。"
         "★ 两条阈值（0.9 / 1.1）不对称是参考库的原始口径，保留。",
))
def chip_high_float_ratio(ctx):
    return _raw_grid(ctx, "below_90")




@register(FactorSpec(
    name="chip_win_peak_frac", group="chip", deps=DEPS_PX,
    desc="获利筹码峰集中度 = 现价下方最大单档占比 / 下方筹码总量（高=获利盘锁筹集中）",
    formula='s = _chip_series(context, "chip_win_peak_frac", need_close=True); '
            'return cross_sectional_rank(s)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=True,
    note="★ 出处：参考库 `factors/chip_deep_extra.py`。参考库定义：max(percents[prices ≤ close]) / Σpercents[prices ≤ close]。"
         "★ 与 `peak_purity`（**全分布**最大单档权重，摘要表字段）的区别：本因子把分母"
         "限制在**现价下方**（只统计获利盘），因此「下方集中」与「全分布集中」是两件事 —— "
         "实测 ρ(·, peak_purity 字段) = +0.6942（中度相关，不是重述）、"
         "ρ(·, below_close 字段) = −0.5300（获利盘越多、下方峰反而越分散："
         "获利筹码被摊到更宽的价位上）、ρ(·, chip_deep_trap_ratio) = +0.4252；"
         "对首批 16 个因子的最大 |ρ| = 0.6847（~chip_cr3_factor，同为「筹码向少数价位"
         "集中」的度量，但本因子的分母只含现价下方）→ 新维度。"
         "2026 实测值域 [0.0117, 1]、中位 0.3138。"
         "★ 语义：获利筹码集中成峰=主力成本密集单一（吸筹完成/锁筹），"
         "分散=浮筹多、涨时兑现压力大。是「吸筹 vs 出货」的形态维度。",
))
def chip_win_peak_frac(ctx):
    return _raw_grid(ctx, "win_peak_frac")
