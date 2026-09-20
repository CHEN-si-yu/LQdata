"""两融（融资融券）因子（19 个）—— 数据源 `stock_margin_detail`（2011-01-04 起，678 万行）。

参考库：`学习资料/factors.md` 类别 fund_flow 的 `margin_*` / `short_*` / `total_leverage_ratio`
（17 个 margin_* + 4 个 short_* + 1 个 total_leverage_ratio）。

★★★ 本族是**滞后表守卫（§4 上游 delay）的样板**，两件事缺一不可 ★★★

    `stock_margin_detail` 的 `trade_date = d` 行，要到 **d 的下一个交易日**才拿得到。
    ① `FactorSpec(lagged_ok=("stock_margin_detail",))` —— 不写，`register()` 直接抛错
       （守卫本身也写在 `fea/spec.py` 的 LAGGED_DATASETS 里）；
    ② **在原始网格上把因子算完，最后一步整体下移一个交易日**：
       `return ctx.lag_grid(grid, 1)`，等价于「value(T) 只用到 trade_date <= T-1 的记录」。

    ⚠️ 位移必须是**最后一步**。在中间步骤位移（比如先位移 rzye 再算 20 日变化率）
       会把「T-1 相对于 T-21 的变化」错写成「T-1 相对于 T-20」，语义错位一格，
       而且看起来完全正常。本文件的模式统一是：算完 → `ctx.lag_grid(grid, 1)`。

    为什么这条重要：不做位移时，同一个 T 的因子值「今天跑」用 T-1 的数据、
    「明天跑」用 T 的数据 —— **静默改变，且没有任何报错**（当天算出来还完全正确）。

    上游实测佐证（2026-09-14 抓的数据）：`stock_margin_detail` 最后一天 = **2026-09-11**，
    而 `stock_daily` 已到 **2026-09-14**；最近 14 个交易日每天稳定 4435~4445 行
    （不是「涨停家数」那种每天波动的表）。即：最近一个交易日的两融数据确实还没出。
    参考库自己也是这么处理的（factors.md: 「margin 数据已 shift(1)」「PIT 平移后」），
    本文件与该口径一致。

## 状态量 vs 流量（本族最容易做错的地方）

| 列 | 语义 | 本文件处置 |
|:--|:--|:--|
| `rzye`(融资余额) `rqye`(融券余额) `rzrqye`(两融余额) `rqyl`(融券余量) | **状态量** | `ctx.asof_daily` —— 稀疏/缺失值 → 日频前向填充 |
| `rzmre`(融资买入额) `rzche`(融资偿还额) `rqmcl`(融券卖出量) | **流量** | 精确落格 + **覆盖范围外 NaN**（`event_grid` + 出现指示，见 `_flow`）|

★ 用 `event_grid` 直接铺余额（= 缺失补 0）是**错的**：余额不是流量，补 0 等于说
「今天杠杆资金全部平仓了」。本文件只在**流量列**上用 `event_grid`，且同时铺一张
「当日有没有记录」的指示网格把覆盖范围外的格子还原成 NaN —— 非两融标的
「不是 0 交易，而是根本没有这项业务」，混进截面会变成一大坨并列的 0，
把排名挤到一边（参考库同样是「NaN 由覆盖范围决定」）。

**停牌日的处置（实测，与直觉相反，必须看）**：上游在停牌日**仍然有行**，
而且余额在继续变化 —— 实测 `600777.SH`（2015 年只有 27 个有行情的交易日，
`stock_daily` 里 217 天没有行）在两融表里有完整的 244 行：停牌期间 `rzmre = 0`
（买不进）、`rzche > 0`（融资盘在停牌中继续还款）、`rzye` 逐日下降。
所以（a）「停牌日没有行」这个假设对本表**不成立**，余额本身就带当日真值；
（b）用 `asof` 前向填充既不会把停牌当 0，也不会在停牌期造出假 0；
（c）流量列的 0（`rzmre = 0`）是**上游真值**，不是我们补的 0，不会被当成缺失。

**覆盖范围**：非两融标的在上游**没有行**（实测 2011~2015：961 只有过两融数据的股票里，
923 只在覆盖期内零缺口，38 只有缺口，缺口多为 149~396 个交易日 —— 是「退出两融名单
后又回来」，缺口期间状态量会被 `asof` 沿用最后一个余额、流量列直接是 NaN、比率因子
因此自动变 NaN）。**两融标的数量随扩容逐年变化**（沙箱实测的**每日有效股票数中位数**）：
2012 年 **277**、2013 年 **543**、2014 年 **704**、2015 年 **856**；
2026-09-09 全市场 4443 只、其中**主板 1954 只** ——
本族 2012~2015 的**截面天然很小**（两融标的还没扩容完），这是数据属性，不是因子坏了。
★★ 因此契约自查里的「每日有效股票数中位数 1500~3300」这一条**对本族的历史区间不适用**
（那是个**全市场**口径的阈值）；本族的截面大小**就是两融名单的大小**，
唯一正确的判据是「当日有值的股数 ≈ 当日两融名单股数」。按此判据实测全部吻合
（2015 年 856 vs 名单 921，差额是主板过滤 + 停牌）；2026 年主板两融名单 1954 只
**正好落在 1500~3300 区间内**，即这条阈值在框架默认起点之后是自然满足的。
下游建模若要求全截面覆盖，两融族只能作为「子样本因子」使用。

## 单位（全部实测核对过，不要凭直觉改）

- `rzye` / `rzmre` / `rzche` / `rqye` / `rzrqye` 单位是**元**（样例 `4,659,469,716`）；
- `rqmcl`（融券卖出量）/ `rqyl`（融券余量）单位是**股**；`ctx.px("vol")` 也是**股**
  → `rqmcl / vol` 可直接相除。实测 2026-09-10：中位数 5.2e-5、p99 7.0e-3、
  约 48% 的两融标的当日融券卖出量为 0 —— 与「A 股融券占比极小」一致；
- `ctx.px("amount")` 是**元**（与 `rzmre`/`rzche` 同量纲，可直接相除）；
- 市值口径按用户约束用**流通市值 = `ctx.px("close")`（未复权）× `ctx.px("float_share")`（股）**
  = 元（PIT 红线：价格水平类用未复权价 × 当期已披露股本）。实测 `rzrqye / 流通市值`
  p50 = 0.0377、p99 = 0.111 —— 与「小百分比」的预期一致；
- **`exchange_id` 一律不用**：两套词表混用（`SSE`/`SZSE` 与另一套），
  且 2026-05-13 起全部是空串，用它切沪深市场会静默错。

## 与参考库的统一偏离（逐条给理由）

1. **不实现参考库的 `clip(...)`**。引擎统一做截面 1%/99% 缩尾 + 截面 rank（契约 §2 铁律 3）。
   clip 只是同一件事的粗糙版本，而且会**掩盖单位错误** —— 实测参考库的
   `clip(0, 0.5)`（`margin_balance_volatility_20d` 的 CV）、`clip(0, 2)`
   （`short_interest_volatility_20d`）恰好落在各自 p99 附近，说明它们本来就是
   「手工缩尾」；`clip(0, 10)`（`short_squeeze_risk`）同理。引擎的 1%/99% 缩尾
   覆盖同一批极端值，且口径统一。
2. **不实现 `cross_sectional_rank(...)`**：引擎统一产出 `rank` 列，因子返回原始值。
3. **方向符号不写进因子值**：参考库把「取反」写进公式（`rank(-shock)`），
   本文件返回**原始量**并用 `higher_is_better=False` 标注方向（与
   `factors/volatility.py` 的既有约定一致），这样 `value` 列仍是可解释的量纲。
4. **价格一律 `ctx.hfq`（后复权）**，参考库用的 `daily_adj.parquet` 是前复权
   （见 `fea/spec.py` 的 PIT 红线）；但**市值类**用未复权 `close` × 当期股本
   （价格水平不给未来分红改写历史）。
5. **滚动实现的边界差异**：`mathx` 的滚动函数是 cumsum 差分实现，
   面板前 `n-1` 行**恒为 NaN**（要求满 n 行历史），比 pandas 的
   `rolling(n, min_periods=k)` 略晚出值。这段差异落在 warmup 区、不进输出窗口，
   已用 `warmup_days` 覆盖（本文件每个因子都按契约 §2 的 `N×1.8+20` 再加 1 天位移损失给）。
6. **`min_count` 对齐参考库的 `min_periods`**（20 日窗 → 10、10 日窗 → 5、5 日窗 → 3），
   语义是「窗口内至少这么多个有效观测」，不是「把缺失当 0」。

## 已实测的数值锚（2026-09-10 截面，用于自查「因子是不是坏了」）

| 因子 | p1 / p50 / p99 |
|:--|:--|
| `margin_buy_pressure` = rzmre/rzye | 0.0032 / 0.038 / 0.61 |
| `total_leverage_ratio` = rzrqye/流通市值 | 0.0031 / 0.038 / 0.111 |
| `short_sell_volume_ratio` = rqmcl/vol | 0 / 5.2e-5 / 7.0e-3 |
| `margin_balance_20d` = rzye 20 日变化率 | −0.45 / −0.026 / +1.20 |
| `margin_velocity` = (rzmre+rzche)/rzye | 0.009 / 0.081 / 1.21 |

以上用**上游原始表直接重算**得到（不经过引擎的 universe 掩码）。同三个锚在
**沙箱 2012~2015 落盘值**上的复核（契约要求的三个 sanity anchor）：

| 锚 | 期望 | 沙箱实测（p1 / p50 / p99） |
|:--|:--|:--|
| `margin_buy_pressure` | 0.01~0.3 的小正数 | 0 / **0.063** / 0.388 ✓ |
| `total_leverage_ratio` | 小百分比 0~0.2 | 0.0011 / **0.0388** / 0.166 ✓（最大 0.393）|
| `short_sell_volume_ratio` | 多数极小 | 0 / **0.0038** / 0.081 ✓（约 48% 为 0）|
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

MARGIN = "stock_margin_detail"
DAILY = "stock_daily"
FINANCE = "stock_finance"

# ★ 起点留 None = 跟随 conf/config.yaml 的 default_start（2012-01-01）。
#   上游 stock_margin_detail 实测 2011-01-04 起，不构成限制。
MAR_START = None

# warmup_days = 「计算窗口要往前多读多少**日历天**」= N×1.8 + 20 + 1（位移损失一天）。
W_D = 30     # 无滚动窗口的当日比率（21 天足矣，给一个月冗余）
W5 = 30      # 5 交易日窗口（5×1.8+20+1 = 30）
W10 = 40     # 10 交易日窗口（10×1.8+20+1 = 39）
W20 = 60     # 20 交易日窗口（20×1.8+20+1 = 57）

# 分母地板：余额/余量的量纲是元/股，1.0 足够挡掉「余额被清零」的格子，
# 又不会误伤真实的极小余额（两家交易所的余额都是 ≥ 0 的实数）。
DEN = 1.0

MARGIN_COLS = ("rzye", "rzmre", "rzche", "rqye", "rqmcl", "rzrqye", "rqyl")


# ══════════════════════════════════════════════════════════════════════
# 数据接入
# ══════════════════════════════════════════════════════════════════════

def _load(ctx):
    """读上游两融明细，年份裁剪到**面板范围**（含 warmup 区）。

    与财报类不同，这张表每个交易日都有行，所以**不需要往前多读一年**：
    面板本身已经被 `warmup_days` 往前推过，窗口内的历史都在面板里。

    ⚠️ 落在面板外的行由 `asof` / `event_grid` 自己丢掉（不会串行到别的格子）。
    """
    d0 = int(ctx.panel.dates[0]) // 10000
    d1 = int(ctx.panel.dates[-1]) // 10000
    df = ctx.dataset(MARGIN, columns=["stock_code", "trade_date", *MARGIN_COLS],
                     years=(d0, d1))
    if df is None or df.empty:
        return None, None
    return df, ctx.date_col(df["trade_date"])


def _state(ctx, df, day, col: str) -> np.ndarray:
    """**状态量**（余额、余量）→ 日频，as-of 前向填充。

    停牌日上游仍有行（余额在继续变化），所以这里的 ffill 只对
    「退出两融名单」这类真缺口起作用：缺口期间沿用最后一个余额。
    """
    v = df[col].to_numpy(dtype=np.float64)
    grid = ctx.asof_daily(df["stock_code"].to_numpy(), day, v)
    return np.asarray(grid, dtype=np.float64)


def _flow(ctx, df, day, col: str) -> np.ndarray:
    """**流量**（当日买入额/偿还额/卖出量）→ 日频精确落格，覆盖范围外 **NaN**。

    ★ `ctx.event_grid` 默认「缺失补 0」，直接用会把「这只股票不是两融标的」
      当成「今天净买入 0」。这里同时铺一张「出现指示」网格（权重全 1），
      用 `safe_div` 把没有记录的日子还原成 NaN —— 与参考库
      「.fea 仅覆盖融资标的，NaN 由覆盖范围决定」同口径。
    ★ 上游真值里的 0（停牌日的 `rzmre`、无融券股票的 `rqmcl`）**保持 0**，
      它们是有效观测，不会被误判成缺失。
    """
    v = df[col].to_numpy(dtype=np.float64)
    got = ctx.event_grid(df["stock_code"].to_numpy(), day,
                         np.ones(v.size, dtype=np.float64))
    raw = ctx.event_grid(df["stock_code"].to_numpy(), day, v)
    return ctx.safe_div(raw, got, min_abs_den=0.5)


def _mv(ctx) -> np.ndarray:
    """流通市值（元）= 未复权 close × float_share。

    PIT 红线（`fea/spec.py`）：价格**水平**类用未复权价 × **当期已披露**股本；
    两个乘数都是状态量（`fea/prices.py` 会对价格/股本做前向填充），
    所以停牌日仍有定义。参考库用的是 `finance.parquet` 的 `total_mv`（总市值），
    用户口径要求**流通**市值，见各因子的 note。
    """
    return ctx.px("close") * ctx.px("float_share")


def _roll_sum(ctx, mat: np.ndarray, n: int, min_count: int) -> np.ndarray:
    """带 `min_count` 的滚动求和。

    ★ `ctx.roll_sum` 没有 min_count 参数（`fea/context.py` 里有两处同名定义，
      后一处 `Panel.roll_sum` 覆盖前一处）。用 `roll_mean × roll_count` 复现：
      `roll_mean` 的分子就是「Σ有效值 / 有效个数」，乘回有效个数即 Σ有效值。
    """
    return ctx.roll_mean(mat, n, min_count) * ctx.roll_count(mat, n)


def _cv(ctx, mat: np.ndarray, n: int, min_count: int, floor: float) -> np.ndarray:
    """变异系数 CV = 滚动标准差 / 滚动均值（参考库口径）。"""
    return ctx.safe_div(ctx.roll_std(mat, n, min_count),
                        ctx.roll_mean(mat, n, min_count), min_abs_den=floor)


def _empty(ctx) -> np.ndarray:
    return np.full(ctx.panel.shape, np.nan, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════
# 一、融资余额的水平与变化（杠杆资金趋势）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="margin_balance_20d", group="margin", deps=(MARGIN,),
    desc="融资余额 20 日变化率（中期杠杆资金趋势）",
    formula='chg = m["rzye"].groupby(level="Code").transform('
            'lambda s: s.pct_change(20, fill_method=None)); '
            'chg = chg.clip(-0.5, 1.0); return cross_sectional_rank(chg)'
            '【本实现：clip/rank 交给引擎；结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W20, higher_is_better=True,
    lagged_ok=(MARGIN,),
    note="★ 滞后表：`lagged_ok` 声明 + `ctx.lag_grid(grid, 1)` 是**最后一步**。"
         "分子分母都是同一列、间隔 20 个交易日，分母用 |rzye(T-20)| 并设 1 元地板"
         "（余额被清零的股票直接给 NaN，而不是 ±1e6 的假变化率）。"
         "上游实测 2026-09-10 截面 p1/p50/p99 = -0.45 / -0.026 / +1.20 —— "
         "「刚纳入两融名单」的股票会出现 +100% 以上的变化率（真实事件，"
         "引擎 1%/99% 缩尾会处理）；参考库的 clip(-0.5, 1.0) 同理，未实现。"
         "停牌日：余额是状态量，asof 前向填充；实测上游停牌日仍有行且余额在变。",
))
def margin_balance_20d(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    grid = ctx.pct_change(_state(ctx, df, day, "rzye"), 20, min_abs_den=DEN)
    return ctx.lag_grid(grid, 1)


@register(FactorSpec(
    name="margin_balance_5d", group="margin", deps=(MARGIN,),
    desc="融资余额 5 日变化率（短期杠杆资金进出速度）",
    formula='chg = m["rzye"].groupby(level="Code").transform('
            'lambda s: s.pct_change(5, fill_method=None)); '
            'chg = chg.clip(-0.5, 1.0); return cross_sectional_rank(chg)'
            '【本实现：clip/rank 交给引擎；结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W5, higher_is_better=True,
    lagged_ok=(MARGIN,),
    note="与 `margin_balance_20d` 同口径，窗口 5 日（短期边际变化）。"
         "上游实测 2026-09-10 p1/p50/p99 = -0.30 / -0.0079 / +0.59。"
         "两个窗口**不重复**：沙箱实测（2012~2015）日内 rank 相关中位数 0.516"
         "（5 日抓拐点、20 日抓趋势），保留两者。",
))
def margin_balance_5d(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    grid = ctx.pct_change(_state(ctx, df, day, "rzye"), 5, min_abs_den=DEN)
    return ctx.lag_grid(grid, 1)








@register(FactorSpec(
    name="margin_chg_rel_5d", group="margin", deps=(MARGIN,),
    desc="融资余额变化率之加速度 = 5 日变化率 − 20 日变化率",
    formula='chg5 = m["rzye"].groupby(level="Code").transform('
            'lambda s: s.pct_change(5, fill_method=None)); '
            'chg20 = m["rzye"].groupby(level="Code").transform('
            'lambda s: s.pct_change(20, fill_method=None)); '
            'return cross_sectional_rank(chg5 - chg20)   '
            '★ 参考库原条目为 margin_chg_rel_ind_5d（融资余额5日变化率**行业相对**）'
            '【本实现：行业表在本框架不可得（见 note），改为「相对自身中期趋势」；'
            '结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W20, higher_is_better=True,
    lagged_ok=(MARGIN,),
    note="★★ 偏离参考库（口径级，必须看）：factors.md 对应条目是 `margin_chg_rel_ind_5d`"
         " —— 「融资余额5日变化率**减去行业等权均值**」。本框架**没有行业表**"
         "（`ctx` 无行业字段，`fea/**` 也不提供），硬做只能去读 `stock_list.industry`，"
         "而那是**当前时点快照**、用在 2012 年的因子上属于前视（本族的全部意义就是"
         " PIT 正确，不为此破例）。同时，「减去**全市场**当日等权均值」是**每日常数平移**，"
         "不改变截面排名 —— 那样写出来会和 `margin_balance_5d` 的 rank 完全重复。"
         "故改为**相对自身中期趋势**：chg5 − chg20，即「短期杠杆资金流入相对中期趋势的"
         "加速度」。它既保留了「剥离整体杠杆环境」的原意（短期 vs 中期用的是同一只股票），"
         "又不是另两个变化率因子的复制品 —— 沙箱实测（2012~2015）日内 rank 相关中位数："
         "与 `margin_balance_5d` **−0.149**、与 `margin_balance_20d` **−0.886**。"
         "★ 与 20 日变化率的强负相关是「加速度」定义的固有性质（chg5 − chg20 里 chg20 是"
         "主导项，符号相反），不是错误；但它意味着下游若已经在用 `margin_balance_20d`，"
         "本因子主要提供的是「反向 + 5 日增量」，建议与 20 日变化率二选一或做正交化。"
         "若后续框架补上**时点化**的行业表，本因子应改写为行业相对口径（version +1）。",
))
def margin_chg_rel_5d(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    rzye = _state(ctx, df, day, "rzye")
    grid = (ctx.pct_change(rzye, 5, min_abs_den=DEN)
            - ctx.pct_change(rzye, 20, min_abs_den=DEN))
    return ctx.lag_grid(grid, 1)


# ══════════════════════════════════════════════════════════════════════
# 二、杠杆水平（相对市值）与总杠杆趋势
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="margin_leverage_change_20d", group="margin",
    deps=(MARGIN, DAILY, FINANCE),
    desc="融资杠杆变化 =（融资余额 / 流通市值）的 20 个交易日变化",
    formula='total_mv_m = fin["total_mv"].reindex(m.index); '
            'leverage = safe_divide(m["rzye"], total_mv_m); '
            'change = leverage.groupby(level="Code").diff(20); '
            'return cross_sectional_rank(change)'
            '【本实现：total_mv → 流通市值 = 未复权 close × float_share（用户口径）；'
            'diff(20) 即 20 个交易日之差；结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W20, higher_is_better=True,
    lagged_ok=(MARGIN,),
    note="★ 分母是**流通市值**（用户口径），参考库用 `finance.total_mv`（总市值）—— "
         "两者对同一只股票的截面排名有系统性差异（总市值含未流通部分，"
         "银行/次新股的差异最大），本文件统一用流通市值，与 `total_leverage_ratio` /"
         " `short_balance_ratio_change_20d` 保持一致。"
         "★ `diff` 而不是 `pct_change`：占比本身已经是标准化量，"
         "参考库用的就是 `.diff(20)`，照抄（占比的绝对值变化）。"
         "★ 滞后的配合：市值用**原始网格当日**的 close（也就是下移后的 T-1 日收盘），"
         "与参考库「Date=T 使用可获得的 T-1 融资数据与 T 日市值」不同 —— "
         "参考库那句话自相矛盾（它自己的 note 又说 close 逐股 shift(1)）。"
         "本实现把整个网格一起下移，分子分母**严格同日**（都是 T-1），"
         "这才是无未来函数且口径自洽的做法。"
         "★ 与 `total_leverage_ratio`（同一个比值的**水平**）实测日内 rank 相关中位数只有"
         " **0.268**（沙箱 2012~2015）—— 「变化」与「水平」基本正交，"
         "两个都保留是有增量的，不要因为名字像就当成重复因子。",
))
def margin_leverage_change_20d(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    lev = ctx.safe_div(_state(ctx, df, day, "rzye"), _mv(ctx), min_abs_den=DEN)
    return ctx.lag_grid(ctx.diff(lev, 20), 1)




@register(FactorSpec(
    name="total_leverage_ratio", group="margin", deps=(MARGIN, DAILY, FINANCE),
    desc="总杠杆率 = 融资融券余额 / 流通市值（高杠杆=平仓风险大，低者优）",
    formula='mv_t = f["total_mv"].reindex(m.index); '
            'ratio = safe_divide(m["rzrqye"], mv_t); ratio = ratio.clip(0, 0.5); '
            'return cross_sectional_rank(-ratio)'
            '【本实现：total_mv → 流通市值 = 未复权 close × float_share（用户口径）；'
            '方向用 higher_is_better=False 标注而不是写进值；'
            '结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W_D, higher_is_better=False,
    lagged_ok=(MARGIN,),
    note="★ 分母是**流通市值**（用户口径，参考库用总市值）。本因子的值就是"
         "「两融余额占流通市值的百分比」原值（不取反），方向由 metadata 标注。"
         "实测 2026-09-10 p1/p50/p99 = 0.0031 / **0.0377** / 0.111 —— "
         "与「小百分比（0~0.2）」的预期一致。参考库的 clip(0, 0.5) 在本平台是空操作"
         "（p99 只有 0.11），未实现。"
         "停牌日：分子是状态量、分母的 close 与 float_share 也都是状态量（前向填充），"
         "所以停牌日仍有值 —— 这是有意的（停牌期间杠杆并没有消失）。",
))
def total_leverage_ratio(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    grid = ctx.safe_div(_state(ctx, df, day, "rzrqye"), _mv(ctx), min_abs_den=DEN)
    return ctx.lag_grid(grid, 1)


# ══════════════════════════════════════════════════════════════════════
# 三、融资偿还与周转（杠杆资金的行为面）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="margin_repay_deceleration", group="margin", deps=(MARGIN,),
    desc="融资偿还额 5 日变化率（偿还减速=看空力量减弱，低者优）",
    formula='chg = m["rzche"].groupby(level="Code").transform('
            'lambda s: s.pct_change(5, fill_method=None)); '
            'chg = chg.clip(-0.5, 0.5); return cross_sectional_rank(-chg)'
            '【本实现：不取反（方向交给 higher_is_better=False）；'
            '结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W5, higher_is_better=False,
    lagged_ok=(MARGIN,),
    note="分子的 `rzche` 是**流量**（覆盖外 NaN），分母是 5 个交易日前的同一列，"
         "地板 1 元 —— 实测 96.8% 的格子有值（只有 3.2% 因为「5 天前偿还额恰好为 0」"
         "被地板挡成 NaN，这是正确的：从 0 到任意值的「变化率」没有定义）。"
         "实测 2026-09-10 p1/p50/p99 = -0.87 / -0.042 / +5.96，是重尾的正数分布。"
         "★ 重尾的实测厚度（沙箱 2012~2015）：|值| > 10 只占 **1.02%**、> 100 占 0.11%、"
         "> 1000 占 0.013%，最大 1.19e6（来自「5 天前偿还额≈1 元 → 今天正常偿还」的"
         "个股，是真实数据不是脏值）。契约上限 1e8 内，且引擎的 `cs_rank` 会做 "
         "(0.01, 0.99) winsor，故不影响排序；但**下游若要用原始值（不是 rank）请自行截尾**。",
))
def margin_repay_deceleration(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    grid = ctx.pct_change(_flow(ctx, df, day, "rzche"), 5, min_abs_den=DEN)
    return ctx.lag_grid(grid, 1)


@register(FactorSpec(
    name="margin_repay_shock", group="margin", deps=(MARGIN,),
    desc="融资偿还冲击 = 当日偿还额 / 20 日均偿还额（突然放大=恐慌平仓，低者优）",
    formula='repay_ma20 = rzche.groupby(level="Code").transform('
            'lambda s: s.rolling(20, min_periods=10).mean()); '
            'shock = safe_divide(rzche, repay_ma20); shock = shock.clip(0, 5); '
            'return cross_sectional_rank(-shock)'
            '【本实现：不取反（方向交给 higher_is_better=False）；'
            '结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W20, higher_is_better=False,
    lagged_ok=(MARGIN,),
    note="参考库 `min_periods=10` → 本实现 `min_count=10`（窗口内至少 10 个有效观测）。"
         "★ 分母均值的有效个数口径由 `ctx.roll_mean` 保证（它在有效值上取均值），"
         "不是「把缺失当 0 除以 20」。实测 2026-09-10 p1/p50/p99 = "
         "0.055 / 0.855 / 3.29（参考库 clip(0,5) 在 p99 之外，未实现）。"
         "停牌日 `rzche` 仍是上游真值（实测停牌期间融资盘会继续还款），不会变 NaN。",
))
def margin_repay_shock(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    rzche = _flow(ctx, df, day, "rzche")
    grid = ctx.safe_div(rzche, ctx.roll_mean(rzche, 20, 10), min_abs_den=DEN)
    return ctx.lag_grid(grid, 1)


@register(FactorSpec(
    name="margin_flow_asymmetry_10d", group="margin", deps=(MARGIN,),
    desc="10 日累计融资净买入 / 10 日累计融资交易额（方向持续性）",
    formula='net = m["rzmre"] - m["rzche"]; total = m["rzmre"] + m["rzche"]; '
            'net_10d = net.groupby(level="Code").transform('
            'lambda s: s.rolling(10, min_periods=5).sum()); '
            'total_10d = total.groupby(level="Code").transform('
            'lambda s: s.rolling(10, min_periods=5).sum()); '
            'asymmetry = safe_divide(net_10d, total_10d); asymmetry = asymmetry.clip(-1, 1); '
            'return cross_sectional_rank(asymmetry)'
            '【本实现：clip/rank 交给引擎；结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W10, higher_is_better=True,
    lagged_ok=(MARGIN,),
    note="两个滚动和都用 `min_count=5`（对齐参考库 `min_periods=5`），"
         "分子分母的**有效观测完全同步**（两列来自同一行、缺一起缺），"
         "所以比值天然落在 [-1, 1]。"
         "★ 这里必须用 `_roll_sum`（= `roll_mean × roll_count`）：`ctx.roll_sum` "
         "没有 min_count 参数，窗口里缺一天就整体变 NaN，10 日窗会大面积作废。",
))
def margin_flow_asymmetry_10d(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    rzmre = _flow(ctx, df, day, "rzmre")
    rzche = _flow(ctx, df, day, "rzche")
    net = _roll_sum(ctx, rzmre - rzche, 10, 5)
    tot = _roll_sum(ctx, rzmre + rzche, 10, 5)
    return ctx.lag_grid(ctx.safe_div(net, tot, min_abs_den=DEN), 1)


@register(FactorSpec(
    name="margin_velocity", group="margin", deps=(MARGIN,),
    desc="融资周转速度 =（融资买入 + 融资偿还）/ 融资余额",
    formula='total_flow = m["rzmre"] + m["rzche"]; velocity = safe_divide(total_flow, m["rzye"]); '
            'velocity = velocity.clip(0, 2); return cross_sectional_rank(velocity)'
            '【本实现：clip/rank 交给引擎；结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W_D, higher_is_better=True,
    lagged_ok=(MARGIN,),
    note="周转速度 = 当日融资交易额 / 存量余额，量纲 1/日。实测 2026-09-10 "
         "p1/p50/p99 = 0.0093 / 0.081 / 1.21（参考库 clip(0,2) 在 p99 之外，未实现）。"
         "与 `margin_buy_pressure` / `margin_net_flow_ratio` 共用分母但含义不同："
         "本因子是**双边**成交强度（周转），那两个分别是**单边买入**与**净买入**。"
         "⚠️ 但**与 `margin_buy_pressure` 实测高度相关**：沙箱（2012~2015）日内 rank "
         "相关中位数 **0.944**（上游原始表同日截面 0.966）—— 因为 A 股融资盘"
         "买入/偿还额量级接近，两项之和近似是买入项的 2 倍常数缩放。"
         "**下游用法建议：与 `margin_buy_pressure` 二选一**；若都要留，"
         "本因子的增量信息主要在「偿还端活跃度」，可考虑改用 (rzche−rzmre)/rzye 取正交残差。"
         "（此处不改定义是为了忠于参考库的公式原文。）",
))
def margin_velocity(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    tot = _flow(ctx, df, day, "rzmre") + _flow(ctx, df, day, "rzche")
    grid = ctx.safe_div(tot, _state(ctx, df, day, "rzye"), min_abs_den=DEN)
    return ctx.lag_grid(grid, 1)


@register(FactorSpec(
    name="margin_balance_volatility_20d", group="margin", deps=(MARGIN,),
    desc="融资余额 20 日变异系数 CV=std/mean（杠杆资金稳定性，低者优）",
    formula='roll_std = rzye.groupby(level="Code").transform('
            'lambda s: s.rolling(20, min_periods=10).std()); '
            'roll_mean = rzye.groupby(level="Code").transform('
            'lambda s: s.rolling(20, min_periods=10).mean()); '
            'cv = safe_divide(roll_std, roll_mean); cv = cv.clip(0, 0.5); '
            'return cross_sectional_rank(-cv)'
            '【本实现：不取反（方向交给 higher_is_better=False）；clip 交给引擎缩尾；'
            '结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W20, higher_is_better=False,
    lagged_ok=(MARGIN,),
    note="参考库用 pandas `.std()`（ddof=1），本框架 `ctx.roll_std` 是总体口径 ddof=0；"
         "在 20 日窗上两者差一个 n/(n-1) 的常数因子，**截面排名完全不受影响**"
         "（同一个 n 对所有股票相同），故不另做修正。"
         "实测 2026-09-10 CV p1/p50/p99 = 0.0076 / 0.049 / 0.455 —— "
         "参考库的 clip(0, 0.5) 恰好落在 p99 上，说明它本来就是手工缩尾，"
         "引擎的 1%/99% 缩尾覆盖同一批极端值。"
         "分母（余额均值）设 1 元地板：余额接近 0 的股票直接 NaN，而不是 ±1e6 的假 CV。",
))
def margin_balance_volatility_20d(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    grid = _cv(ctx, _state(ctx, df, day, "rzye"), 20, 10, DEN)
    return ctx.lag_grid(grid, 1)


@register(FactorSpec(
    name="margin_buyer_avg_cost_premium", group="margin",
    deps=(MARGIN, DAILY, "stock_adj_factor"),
    desc="融资盘成本溢价 = 现价 / 近 20 日融资买入加权平均成本 − 1",
    formula='cost = _margin_weighted_cost(margin, daily); '
            'close = daily["close"].reindex(cost.index); '
            'raw = safe_divide(close, cost) - 1.0; return cross_sectional_rank(raw)'
            '【本实现：加权成本 = Σ(rzmre × hfq_close) / Σ(rzmre) over 20 日（原函数未公开，'
            '见 note）；配对价格用后复权 close；结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W20, higher_is_better=True,
    lagged_ok=(MARGIN,),
    note="★ 参考库的 `_margin_weighted_cost` 源码未公开（factors.md 只给了调用），"
         "本实现按「近 20 日融资买入额加权的成交价」落地："
         "cost = Σ(rzmre × close) / Σ(rzmre)，min_count=10（20 日窗的半数，与同族一致）。"
         "★ 价格用**后复权** `ctx.hfq(\"close\")`（分子分母同一口径）："
         "如果用未复权 close，除权日会凭空造出 −10% 的「融资盘被套」，"
         "而后复权把分红还原成收益，得到的才是真正的「浮盈/被套」；"
         "且后复权是 PIT 安全的（历史值不被未来分红改写）。"
         "★ 参考库 note 写「margin 数据已 shift(1)，配对价格用 close 逐股 shift(1)」—— "
         "本实现等价：整个网格一起下移一格，T 日的值用 T-1 的融资买入、T-1 的收盘价。"
         "实测 2026-09-10 该因子 p1/p50/p99 ≈ **−0.15 / −0.006 / +0.15**"
         "（p25/p75 = −0.04 / +0.02，标准差 0.057；融资盘加权成本紧贴现价，"
         "符合「融资盘平均持仓期很短」的事实 —— 若这个因子某天开始出现 ±0.5 以上的"
         "截面离散度，先怀疑成本端用错了复权口径）。"
         "该统计用上游原始表直接重算（20 日窗、min_count=10、仅两融标的），"
         "与沙箱落盘值的差异来自 universe 掩码与停牌处置，量级一致。",
))
def margin_buyer_avg_cost_premium(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    rzmre = _flow(ctx, df, day, "rzmre")
    px = ctx.hfq("close")                       # 后复权（状态量，停牌日沿用最后价）
    cost = ctx.safe_div(_roll_sum(ctx, rzmre * px, 20, 10),
                        _roll_sum(ctx, rzmre, 20, 10), min_abs_den=DEN)
    return ctx.lag_grid(ctx.safe_div(px, cost, min_abs_den=DEN) - 1.0, 1)


# ══════════════════════════════════════════════════════════════════════
# 四、融券 / 做空（存量、流量与逼空）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="short_sell_volume_ratio", group="margin", deps=(MARGIN, DAILY),
    desc="融券卖出占比 = 融券卖出量 / 当日成交量（活跃做空）",
    formula='vol_t = d["vol"].reindex(m.index); ratio = safe_divide(m["rqmcl"], vol_t); '
            'ratio = ratio.clip(0, 1); return cross_sectional_rank(ratio)'
            '【本实现：分子分母取**同一个交易日**（见 note），再整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W_D, higher_is_better=True,
    lagged_ok=(MARGIN,),
    note="★ 单位已实测核对：`rqmcl`（融券卖出量）与 `ctx.px(\"vol\")` **都是股**，"
         "可以直接相除。实测 2026-09-10 中位数 5.2e-5、p99 7.0e-3、"
         "约 48% 的两融标的当日融券卖出量为 0 —— 与「A 股融券占成交比重极小」一致。"
         "★ 偏离参考库的一句话（口径级）：参考库 note 写「Date=T 使用当时可获得的 "
         "T-1 融券卖出量和 T 日总成交量」，即**跨日配对**；本实现按框架 §4 的强制口径，"
         "在**同一交易日**网格上算 rqmcl/vol、再把整个网格下移一格 —— "
         "这样 T 日的因子用的是 T-1 的融券卖出量 **和 T-1 的成交量**，"
         "分子分母严格同日（跨日配对会把两天的信息混在一起，反而更难解释）。"
         "停牌日 `vol` 是 NaN（流量语义）→ 因子 NaN，不会把停牌当成「零做空」。",
))
def short_sell_volume_ratio(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    grid = ctx.safe_div(_flow(ctx, df, day, "rqmcl"), ctx.px("vol"), min_abs_den=DEN)
    return ctx.lag_grid(grid, 1)


@register(FactorSpec(
    name="short_balance_ratio_change_20d", group="margin",
    deps=(MARGIN, DAILY, FINANCE),
    desc="融券余额占比变化 =（融券余额 / 流通市值）的 20 个交易日变化（空头加仓，低者优）",
    formula='total_mv_m = fin["total_mv"].reindex(m.index); '
            'short_ratio = safe_divide(m["rqye"], total_mv_m); '
            'change = short_ratio.groupby(level="Code").diff(20); '
            'return cross_sectional_rank(-change)'
            '【本实现：total_mv → 流通市值 = 未复权 close × float_share（用户口径）；'
            '不取反（方向交给 higher_is_better=False）；结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W20, higher_is_better=False,
    lagged_ok=(MARGIN,),
    note="与 `margin_leverage_change_20d` 完全同口径（同一个分母、同一个 diff(20)、"
         "同一天下移），只把分子从融资余额换成**融券**余额，"
         "所以两个因子可以直接对比「多头杠杆 vs 空头杠杆」的边际变化。"
         "占比为 0 的股票（约 6% 的行 `rqye = 0`）变化也是 0 —— 参考库明确指出这是"
         "**有效信息**（空头仓位本来就是 0），不置 NaN。"
         "★ 融券余额占流通市值实测只有万分之几（`rqye/rzye` 中位数 0.0021），"
         "所以本因子的绝对量级很小，靠截面排名取信息。"
         "★ 「多空杠杆变化的对比」这个用法实测是成立的：与 `margin_leverage_change_20d`"
         " 的日内 rank 相关中位数 **−0.073**（沙箱 2012~2015），几乎正交 —— "
         "融资盘的加杠杆与融券盘的加仓在截面上是两件独立的事。",
))
def short_balance_ratio_change_20d(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    ratio = ctx.safe_div(_state(ctx, df, day, "rqye"), _mv(ctx), min_abs_den=DEN)
    return ctx.lag_grid(ctx.diff(ratio, 20), 1)


@register(FactorSpec(
    name="short_squeeze_risk", group="margin", deps=(MARGIN,),
    desc="逼空风险 = 融券余量 / 融资余额（做空拥挤，高者优=潜在逼空反转）",
    formula='squeeze = safe_divide(m["rqyl"], m["rzye"]); squeeze = squeeze.clip(0, 10); '
            'return cross_sectional_rank(squeeze)'
            '【本实现：clip/rank 交给引擎；结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W_D, higher_is_better=True,
    lagged_ok=(MARGIN,),
    note="★ 参考库原式是 `rqyl(股) / rzye(元)`，**量纲混合**（股/元），"
         "这里**照抄参考库**保持可比性：它仍然是一个有效的截面排序量，"
         "但会带上一点股价水平的倾斜（同样经济含义下，高价股的比值更小）。"
         "实测 2026-09-10 该比值 ×100 后 p1/p50/p99 = 0 / 0.0104 / 10.04 —— "
         "参考库 clip(0, 10) 恰好压在 p99 上（再次说明它的 clip 就是手工缩尾）。"
         "★ 若下游希望去掉价格水平的影响，等价的「金额口径」是 `rqye / rzye`"
         "（两列都是元，实测 ×100 后 p50 = 0.213、p99 = 22.6）—— "
         "本文件未改口径，因为参考库与任务书都写的是 `rqyl / rzye`；"
         "要换口径请改 version 并在 FACTORS.md 里登记。",
))
def short_squeeze_risk(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    grid = ctx.safe_div(_state(ctx, df, day, "rqyl"), _state(ctx, df, day, "rzye"),
                        min_abs_den=DEN)
    return ctx.lag_grid(grid, 1)


@register(FactorSpec(
    name="short_interest_volatility_20d", group="margin", deps=(MARGIN,),
    desc="融券余量 20 日变异系数 CV=std/mean（空头仓位稳定性，低者优）",
    formula='roll_std = rqyl.groupby(level="Code").transform('
            'lambda s: s.rolling(20, min_periods=10).std()); '
            'roll_mean = rqyl.groupby(level="Code").transform('
            'lambda s: s.rolling(20, min_periods=10).mean()); '
            'cv = safe_divide(roll_std, roll_mean); cv = cv.clip(0, 2); '
            'return cross_sectional_rank(-cv)'
            '【本实现：不取反（方向交给 higher_is_better=False）；clip 交给引擎缩尾；'
            '结果整体下移 1 个交易日】',
    start=MAR_START, warmup_days=W20, higher_is_better=False,
    lagged_ok=(MARGIN,),
    note="与 `margin_balance_volatility_20d` 同结构，只把余额换成**融券余量**。"
         "实测 2026-09-10 CV p1/p50/p99 = 0 / 0.220 / 1.835 —— "
         "参考库 clip(0, 2) 也在 p99 附近（手工缩尾）。"
         "★ 融券余量大量为 0（约 6% 的格子 `rqyl = 0`）：窗口内全 0 时 std 与 mean 都是 0，"
         "`safe_div` 会给出 NaN（而不是 0/0 的 inf）——这是对的，"
         "「从来没有融券」不构成「空头仓位稳定」的证据。",
))
def short_interest_volatility_20d(ctx):
    df, day = _load(ctx)
    if df is None:
        return _empty(ctx)
    grid = _cv(ctx, _state(ctx, df, day, "rqyl"), 20, 10, DEN)
    return ctx.lag_grid(grid, 1)
