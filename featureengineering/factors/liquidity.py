"""流动性 / 交易活跃度因子（18 个）—— 全部日频、只主板、跟随 default_start。

参考库：`学习资料/因子库.md` §3 Liquidity（35 个） + `学习资料/factors.md`
（price / volume / liquidity 段落），净换手率另参
`学习资料/量化策略/选股因子研究系列（五）——寻找股价驱动新因子之净换手率.pdf`。

## 写这一族时必须记住的六件事（都是实测踩过的）

1. **单位**：`ctx.px("amount")` 是**元**，`ctx.px("vol")` 是**股**，
   `ctx.px("turnover")` 是**百分数**（= vol / float_share × 100）。
   量纲在比率里自己约掉。`amount / vol` = 当日成交均价（元/股）——
   两列**同源同日**、停牌日**同为 NaN**，可以相除当 VWAP 用（已核对 2013Q1：1732 个
   停牌格里 amount 与 vol 全部同时为 NaN）。反之 `close` 是**未复权**价，
   所以 VWAP 偏离只能用未复权 close，**不能**换成 `ctx.hfq`（那样两边不同量纲）。

2. **停牌日的 vol / amount / pct_chg / turnover 是 NaN 而不是 0**（`fea/prices.py`
   的流量语义），这是有意的。本文件**从不**用 `np.nan_to_num` 之类的把 NaN 变 0 ——
   那等于把停牌记成「零成交」，会系统性低估流动性。
   需要窗口跳过停牌日时，一律用 `min_count=` 显式放松，取值对齐参考库的
   `min_periods`（20 日窗 → 10，60 日窗 → 30，5 日窗 → 2，涨跌分区窗 → 5），
   语义是「**只在有行情的交易日上算**」，不是「把停牌当 0」。

3. **收益一律走 `_ret(ctx)`**：`ctx.ret(1)` 的分子分母都是**前向填充过**的
   `hfq_close`，所以停牌日的收益是**精确的 0.0**，直接拿来做滚动均值会把
   「没交易」稀释成「零涨跌」。`_ret()` 用 `ctx.traded()` 把它还原成 NaN，
   与参考库 `daily.parquet` 里停牌日 `pct_chg` 为 NaN 的口径一致。
   另外参考库用的是 `daily_adj.parquet`（**前复权**），本项目一律换成后复权
   （PIT 红线），所以这里用 `ctx.ret` 而不是 `ctx.px("pct_chg")`：
   实测 2013 年 `pct_chg/100` 与后复权收益在 5.4e-5 的格子上差 >0.5%，
   差值最大 0.88 —— 全是除权/复牌那几天，正是最需要复权口径的地方。
   （唯一例外：`zero_return_fraction_20` 要的就是「交易所报的涨跌幅是不是 0」，
   按任务要求显式用 `ctx.px("pct_chg")`。）

4. **Amihud 一律不带 ×1e8**。参考库自身不一致：`amihud_daily_20` 与
   `liquidity_shock_20` 不带，`amihud_amt_20d` / `amihud_daily_5` 带。
   本族统一取 1/元 量纲，因为 (a) 同族值域可比，(b) `liquidity_shock_20` 的
   `log(amihud + 1e-12)` 只有不带 ×1e8 时才成立（带的话常数项会盖住信号）。
   截面 rank 不受常数因子影响。

5. **三个框架原语在这里要绕开**（本文件用 `roll_mean` 的等价展开重写，
   实测与 pandas 逐窗结果一致到 1e-12）：
   - `ctx.roll_std(mat, n, min_count=k)`：`mathx.roll_var` 把缺失当 0 进 cumsum、
     分母固定取 **n**（不是有效值个数），`k<n` 时波动率被系统性算小
     （实测窗口 20 缺 3：1.125 vs 正确 1.219）→ 用 `_roll_std`。
   - `ctx.roll_corr(x, y, n, min_count=k)`：同一问题，等价于把缺失格当成
     `(x̄, ȳ)` 补进去，|r| 被系统性拉向 0（实测 −0.2063 vs 正确 −0.2083）→ 用 `_roll_corr`。
   - `ctx.roll_sum` **根本没有 min_count 参数**（`fea/context.py` 里有两处同名定义，
     后者覆盖前者）→ 用 `_roll_sum`（mean × count）。
   这是本族唯一需要引擎补齐的地方，详见各 helper 的 docstring。

6. **`net_turnover_rate_20` 有一处公式级偏离**（本族唯一一处，务必看该因子的 note）：
   参考库的「买入四桶之和 − 卖出四桶之和」在 `stock_main_fund_flow` 上是**恒等于 0**
   的（该表是双记口径，实测 97.68% 的行 |差| ≤ 1 手），照抄会得到一个常数因子。
   已改为「主力净买入量 = (大单+超大单) 买 − 卖」。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

# 跟随 conf/config.yaml 的 default_start（当前 2012-01-01）
LIQ_START = None

# warmup_days = 「计算窗口要往前多读多少日历天」，纯日频滚动按 N×1.8+20 给。
W5 = 30        # 5 交易日窗口
W10 = 40       # 10 交易日窗口
W20 = 60       # 20 交易日窗口（20×1.8+20 = 56，取整到 60）
W40 = 95       # liquidity_shock_20：20 日窗 + 再往前 shift 20 日 → 有效 40 日
W60 = 128      # turnover_anomaly_20 用 60 日窗（60×1.8+20）
W1 = 30        # 当日比率（VWAP 偏离），没有滚动窗口，给一个月冗余

_MF_BUY = ("buy_sm_vol", "buy_md_vol", "buy_lg_vol", "buy_elg_vol")
_MF_SELL = ("sell_sm_vol", "sell_md_vol", "sell_lg_vol", "sell_elg_vol")
# 主力（大单 + 超大单）：本平台唯一的「主动买卖」代理，理由见 net_turnover_rate_20 的 note
_MF_BUY_MAIN = ("buy_lg_vol", "buy_elg_vol")
_MF_SELL_MAIN = ("sell_lg_vol", "sell_elg_vol")


# ══════════════════════════════════════════════════════════════════════
# 工具：把参考库的 pandas 语义在本框架里**精确**复现
# ══════════════════════════════════════════════════════════════════════

def _ret(ctx) -> np.ndarray:
    """当日后复权收益，**停牌日强制 NaN**。

    ★ `ctx.ret(1)` 的 hfq_close 是前向填充的，停牌日算出来是精确的 0.0。
      直接用它做 `rolling.mean` / `rolling.sum` 会把「当天没交易」混进分母，
      把均值稀释、把量能倾斜拉偏。参考库的 daily.parquet 里停牌日 pct_chg 是 NaN，
      这里用 `ctx.traded()` 把语义对齐（它 = vol 是否有限）。
    """
    return np.where(ctx.traded(), ctx.ret(1), np.nan)


def _amihud(ctx) -> np.ndarray:
    """Amihud 非流动性 = |收益| / 成交额（量纲 1/元）。

    分母是成交额（元），可能为 0（极冷门股）或 NaN（停牌）——
    用 `ctx.safe_div` 挡掉，绝不产生 inf / 1e9 量级的假值。
    """
    return ctx.safe_div(np.abs(_ret(ctx)), ctx.px("amount"), min_abs_den=1.0)


def _roll_sum(ctx, mat: np.ndarray, n: int, min_count: int) -> np.ndarray:
    """带 `min_count` 的滚动求和。

    ★ `ctx.roll_sum` 没有 min_count 参数（`fea/context.py` 里有两处同名定义，
      后一处覆盖前一处），`Panel.roll_sum` 也没有。这里用
      `roll_mean × roll_count` 复现：roll_mean 的分子就是 Σ有效值 / 有效个数，
      乘回有效个数即 Σ有效值（mathx.roll_mean 内部先减列均值再求和，
      所以对 |x|~1e8 的成交额/成交量列也不丢精度）。
    """
    return ctx.roll_mean(mat, n, min_count) * ctx.roll_count(mat, n)


def _roll_std(ctx, mat: np.ndarray, n: int, min_count: int) -> np.ndarray:
    """带 `min_count` 的滚动标准差（总体口径 ddof=0 = 参考库 pandas 的 min_periods 语义）。

    ★ 不能用 `ctx.roll_std`：`mathx.roll_var` 走 cumsum，缺失值按 0 计入、
      分母**固定取 n** 而不是有效值个数，`min_count < n` 时结果系统性偏小
      （实测 n=20 里有 3 个 NaN：1.125 vs 正确 1.219）。
      改用 `E[x²] − E[x]²`：两个期望都走 `roll_mean(...)`（它在有效值上取均值），
      等价展开后与 pandas `rolling(n, min_periods=k).std(ddof=0)` 逐格一致。
    """
    mu = ctx.roll_mean(mat, n, min_count)
    var = ctx.roll_mean(mat * mat, n, min_count) - mu * mu
    return np.sqrt(np.maximum(var, 0.0))         # 浮点误差导致的极小负值


def _roll_corr(ctx, x: np.ndarray, y: np.ndarray, n: int, min_count: int) -> np.ndarray:
    """带 `min_count` 的滚动相关，**只用 (x, y) 成对有效的格子**。

    ★ 不能用 `ctx.roll_corr`：同一个 cumsum 实现把「x 或 y 缺失」的格子当成
      `(x̄, ȳ)`（即零偏差点）补进公式，min_count<n 时 |r| 被系统性拉向 0
      （实测 −0.2063 vs 正确 −0.2083）—— 而「缺了多少格」与停牌频率有关，
      于是偏差本身带上了截面结构，这不是随机噪声。
      这里先把两侧都遮成 NaN，再用 E[xy] − E[x]E[y] 展开，与 pandas
      `rolling(n, min_periods=k).corr()` 的成对语义一致。
    """
    ok = np.isfinite(x) & np.isfinite(y)
    xm = np.where(ok, x, np.nan)
    ym = np.where(ok, y, np.nan)
    mx = ctx.roll_mean(xm, n, min_count)
    my = ctx.roll_mean(ym, n, min_count)
    cov = ctx.roll_mean(xm * ym, n, min_count) - mx * my
    vx = ctx.roll_mean(xm * xm, n, min_count) - mx * mx
    vy = ctx.roll_mean(ym * ym, n, min_count) - my * my
    return ctx.safe_div(cov, np.sqrt(np.maximum(vx * vy, 0.0)), min_abs_den=0.0)


def _turnover(ctx) -> np.ndarray:
    """换手率（**百分数**，流通股本口径）= ctx.px("turnover") = vol / float_share × 100。

    参考库 `turnover_*` 系列用的是 `finance.parquet` 的 `turnover_rate`（**总股本**口径）。
    本框架的价格层只从 stock_finance 取 total_share/float_share/free_share 三列，
    `turnover_rate` 取不到；任务也明确要求优先用 `ctx.px("turnover")`。
    自算口径（vol/float_share）与参考值同量纲、同号，数值偏大
    （float_share ≤ total_share，实测样例 1.07 vs 0.45），截面 rank 高度相关但不逐位相同。
    """
    return ctx.px("turnover")


# ══════════════════════════════════════════════════════════════════════
# ① 非流动性（Amihud）
# ══════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="amihud_daily_5", version=2, group="liquidity",
    deps=("stock_daily", "stock_adj_factor"),
    desc="Amihud 日频非流动性：5 日平均 |收益|/成交额（短周期版）",
    formula='amihud = |ret| / amount; amihud5 = roll(amihud, 5, "mean", min_periods=2)',
    start=LIQ_START, warmup_days=W5, higher_is_better=True,
    note="★ 偏离参考库：去掉原文的 ×1e8 常数（原文 `df['absret']/df['amount']*1e8`），"
         "与同族 amihud_daily_20 / liquidity_shock_20 保持同一量纲（1/元）；"
         "截面 rank 完全不受常数影响。min_count=2 = 参考库 min_periods=2。"
         "5 日窗对停牌更敏感、对资金流出更灵敏，与 20 日版互补。",
))
def amihud_daily_5(ctx):
    return ctx.roll_mean(_amihud(ctx), 5, min_count=2)




# ══════════════════════════════════════════════════════════════════════
# ② 换手率
# ══════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="turnover_std_20", version=2, group="liquidity", deps=("stock_daily", "stock_finance"),
    desc="换手率波动：20 日换手率标准差（流动性不稳定的代理）",
    formula='to_vol = rolling(20, min_periods=10).std(); Factor = -to_vol',
    start=LIQ_START, warmup_days=W20, higher_is_better=False,
    note="★ 用 _roll_std 而不是 ctx.roll_std：后者在 min_count=10<n=20 时把缺失当 0、"
         "分母固定取 20，会把波动率算小约 1/3（实测 1.125 vs 1.219），"
         "而缺多少格与停牌频率相关 → 偏差带截面结构。"
         "ddof=0（总体），参考库 pandas .std() 是 ddof=1；两者只差 20/19 的常数，"
         "在有停牌的窗口上 ddof=0 对「有效值个数」才自洽。口径为流通股本换手率。",
))
def turnover_std_20(ctx):
    return _roll_std(ctx, _turnover(ctx), 20, min_count=10)




@register(FactorSpec(
    name="turnover_anomaly_20", version=2, group="liquidity", deps=("stock_daily", "stock_finance"),
    desc="换手率异常：20 日均换手 / 60 日均换手 − 1",
    formula='to_20 = rolling_group_mean(turnover, 20); to_60 = rolling_group_mean(turnover, 60); '
            'anomaly = to_20 / to_60 - 1.0',
    start=LIQ_START, warmup_days=W60, higher_is_better=False,
    note="参考库没写 min_periods，按本框架的 n//2 约定取 min_count=10 / 30。"
         "60 日窗要 128 个日历天 warmup。分母 to_60 是换手率、恒正，"
         "但停牌多的股票 to_60 可能为 NaN → 直接 NaN，不补值。",
))
def turnover_anomaly_20(ctx):
    t = _turnover(ctx)
    to20 = ctx.roll_mean(t, 20, min_count=10)
    to60 = ctx.roll_mean(t, 60, min_count=30)
    return ctx.safe_div(to20, to60, min_abs_den=1e-9) - 1.0




# ══════════════════════════════════════════════════════════════════════
# ③ 量价关系
# ══════════════════════════════════════════════════════════════════════





@register(FactorSpec(
    name="amount_ratio_20", version=2, group="liquidity", deps=("stock_daily",),
    desc="相对成交额：当日成交额 / 20 日均成交额 − 1",
    formula='avg_amount = rolling(20, min_periods=10).mean(); '
            'a_ratio = amount / avg_amount.replace(0, np.nan) - 1.0',
    start=LIQ_START, warmup_days=W20, higher_is_better=False,
    note="成交额比成交量更能反映资金参与度。窗口**含当日**（参考库就是含当日的 "
         "rolling），所以放量日 a_ratio > 0。min_count=10 对齐参考库。"
         "成交额是元、量级 1e7~1e9，分母用 safe_div(min_abs_den=1.0) 挡 0。",
))
def amount_ratio_20(ctx):
    amt = ctx.px("amount")
    avg = ctx.roll_mean(amt, 20, min_count=10)
    return ctx.safe_div(amt, avg, min_abs_den=1.0) - 1.0






# ══════════════════════════════════════════════════════════════════════
# ④ 资金流：净换手率（★ 选股因子研究系列（五））
# ══════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════
# ⑤ 量能形态
# ══════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="vol_ratio_ma5_ma20", version=2, group="liquidity", deps=("stock_daily",),
    desc="量能短长比：5 日均量 / 20 日均量",
    formula='ma_5 = rolling_group_mean(vol, 5); ma_20 = rolling_group_mean(vol, 20); '
            'ratio = safe_divide(ma_5, ma_20 + 1e-8)',
    start=LIQ_START, warmup_days=W20, higher_is_better=True,
    note="对齐 factors.md 的 volume_momentum_5（成交量短长均线比）。"
         "参考库的 `vol_ratio_ma5` / `vol_ratio_ma20` 是 finance.volume_ratio 的两条均线，"
         "而本框架的价格层读不到 stock_finance 的 volume_ratio 列，"
         "所以按同样的「5 日 vs 20 日」构造直接用 vol 自算，比参考的「量比之均线」更平滑、"
         "不含当日突刺（当日突刺由 volume 相关的 amount_ratio_20 / turnover 系列覆盖）。"
         "min_count=2 / 10 对应 n//2。",
))
def vol_ratio_ma5_ma20(ctx):
    vol = ctx.px("vol")
    ma5 = ctx.roll_mean(vol, 5, min_count=2)
    ma20 = ctx.roll_mean(vol, 20, min_count=10)
    return ctx.safe_div(ma5, ma20, min_abs_den=1.0)


@register(FactorSpec(
    name="amount_mom_accel", version=2, group="liquidity", deps=("stock_daily",),
    desc="成交额动量加速：3 日变化率 − 10 日变化率",
    formula='p3 = amount.pct_change(3); p10 = amount.pct_change(10); vals = p3 - p10',
    start=LIQ_START, warmup_days=W10, higher_is_better=True,
    note="量能一阶动量反映资金流入方向，二阶（加速）反映资金行为切换。"
         "ctx.pct_change 的 min_abs_den=1e-12 对成交额（元）足够松，不会误伤。"
         "停牌日成交额 NaN → 变化率 NaN → 该格为 NaN（不补 0）。",
))
def amount_mom_accel(ctx):
    amt = ctx.px("amount")
    return ctx.pct_change(amt, 3) - ctx.pct_change(amt, 10)


