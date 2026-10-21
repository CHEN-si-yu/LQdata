"""两融（融资融券）因子（19 个）—— 数据源 `stock_margin_detail`（2011-01-04 起，678 万行）。

参考库：`学习资料/factors.md` 类别 fund_flow 的 `margin_*` / `short_*` / `total_leverage_ratio`
（17 个 margin_* + 4 个 short_* + 1 个 total_leverage_ratio）。

★★★ 本族是**滞后表守卫（§4 上游 delay）的样板**，两件事缺一不可 ★★★

    `stock_margin_detail` 的 `trade_date = d` 行，要到 **d 的下一个交易日**才拿得到。
    ① `FactorSpec(lagged_ok=("stock_margin_detail",))` —— 不写，`register()` 直接抛错
       （守卫本身也写在 `fea/spec.py` 的 LAGGED_DATASETS 里）；
    ② **在原始网格上把因子算完，最后一步整体下移一个交易日**：
       `return ctx.lag_grid(grid, LAGGED_DATASETS.get(MARGIN, 1))`，等价于「value(T) 只用到 trade_date <= T-1 的记录」。

    ⚠️ 位移必须是**最后一步**。在中间步骤位移（比如先位移 rzye 再算 20 日变化率）
       会把「T-1 相对于 T-21 的变化」错写成「T-1 相对于 T-20」，语义错位一格，
       而且看起来完全正常。本文件的模式统一是：算完 → `ctx.lag_grid(grid, LAGGED_DATASETS.get(MARGIN, 1))`。

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

from fea.spec import FactorSpec, register, LAGGED_DATASETS

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








def _empty(ctx) -> np.ndarray:
    return np.full(ctx.panel.shape, np.nan, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════
# 一、融资余额的水平与变化（杠杆资金趋势）
# ══════════════════════════════════════════════════════════════════════













# ══════════════════════════════════════════════════════════════════════
# 二、杠杆水平（相对市值）与总杠杆趋势
# ══════════════════════════════════════════════════════════════════════







# ══════════════════════════════════════════════════════════════════════
# 三、融资偿还与周转（杠杆资金的行为面）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="margin_repay_deceleration", version=2, group="margin", deps=(MARGIN,),
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
    return ctx.lag_grid(grid, LAGGED_DATASETS.get(MARGIN, 1))


@register(FactorSpec(
    name="margin_repay_shock", version=2, group="margin", deps=(MARGIN,),
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
    return ctx.lag_grid(grid, LAGGED_DATASETS.get(MARGIN, 1))










# ══════════════════════════════════════════════════════════════════════
# 四、融券 / 做空（存量、流量与逼空）
# ══════════════════════════════════════════════════════════════════════







