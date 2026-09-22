"""波动率 / 风险因子（26 个）—— 全部由**后复权**日收益与后复权价格推导。

## 口径（本家族唯一允许的两条输入）

  · 收益：`_ret(ctx)` = `ctx.ret(1)` 再叠加 `ctx.traded()` 掩码 —— 后复权、
    |收益|>60% 的复权脏数据置 NaN，**停牌日也置 NaN**（见 `_ret` 的 docstring：
    裸 `ctx.ret` 在停牌日给的是 0.0，与契约 §3.1 的说法不符，会系统性低估停牌股的风险）。
  · 价格：`ctx.hfq("close")` —— 后复权价（停牌期间前向填充，未上市为 NaN）。
    回撤类因子**故意**用前向填充价：停牌期间「最后一个成交价」仍然是有效价格，
    回撤就该继续对着它算，不该断成 NaN。

**绝不用 `close.pct_change()`**：未复权收盘价在除权日会跳水，造出 −50% 的假暴跌，
直接毁掉全部波动 / 回撤 / 尾部因子，而且**不报错**（看起来只是「因子有点噪」）。

## 三条窗口约定（每个因子在 note 里各写一遍）

  1. **收益类滚动窗口** N 个交易日，`min_count` 抄参考库的 `min_periods`
     （契约 §3.4 明写：「参考库里的 `min_periods=n//2` 对应 `min_count=n//2`」）。
     ⚠️ 框架语义实测：`roll_var/roll_std` 带 `min_count=k` 时分母仍是**满窗 n**，
     缺失格子按「逐列全局均值」补 → 窗口内缺得越多，波动率被压得越低
     （k=45/60 时约 −14%）。收益类因子的正常格子 88% 是满窗（实测 2012-2015 主板），
     影响集中在长期停牌股；已在 note 里标注，并在汇报里作为「需要引擎扩展」提出。
  2. **回撤类** 用 `hfq_close / roll_max(hfq_close, N) − 1`，即「当前价相对**最近 N 个
     交易日（含当日）**最高价」的回撤，取值 ≤ 0。这**不是**教科书式的「窗口内峰谷最大
     回撤」（后者允许峰值早于窗口起点、需要窗口内的运行最大值，一次 `roll_max` 表达不了）；
     选它的理由见 `max_drawdown_120` 的 note。
  3. **市场基准** = 沪深300（`000300.SH`，来自 `index_daily`），
     全池等权市场收益（参考库的 `_market_proxy`）**没有**采用，理由见 `beta_60` 的 note。

## 依赖的框架能力缺口（已在汇报中提出）

  · `fea/mathx.py` **有** `roll_quantile`，但 `FactorContext` **没有**转发它
    （VaR/CVaR 需要）。本文件按 `factors/labels.py` 的先例直接 `from fea import mathx`
    调用，未改任何框架代码。
  · `ctx.panel.place` 未在契约 §3 的能力表里，但它是「精确落格」语义的唯一入口
    （指数收益不能前向填充），本文件使用它。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fea import mathx as mx
from fea.spec import FactorSpec, register

# 全部因子只依赖后复权价格；三个市场类因子额外依赖指数日线。
PRICE_DEPS = ("stock_daily", "stock_adj_factor")
MKT_DEPS = ("stock_daily", "stock_adj_factor", "index_daily")
MKT_INDEX = "000300.SH"

# warmup：窗口 N 个交易日 -> N×1.8+20 日历天；N=120 的按契约取 240。
W20, W60, W120 = 56, 128, 240


# --------------------------------------------------------------------- 工具
def _ret(ctx) -> np.ndarray:
    """后复权日收益，并**显式挡掉停牌/未上市日**。

    ★ 为什么不能直接用 `ctx.ret(1)`（实测，2014 年主板）：
      `ctx.ret` 是在**前向填充过**的 hfq 价上做差分的，停牌日的价格 = 上一交易日价格
      → 收益恰好是 **0.0 而不是 NaN**（2014 年 367032 个停牌格子里有 35408 个 ret==0.0，
      占全部收益的 6.02%）。契约 §3.1 与价格层的 docstring 都写着「已挡掉停牌」，
      实际行为不是 —— 停牌股会因此被算成**低波动/低风险**，正是本家族最怕的假信号。
      参考库的 `_ret_wide(daily)` 是 pivot 出来的，缺失格就是 NaN，所以这里补上
      `ctx.traded()` 才是与参考库同口径。停牌期间的分红会体现在复牌当日的收益里（保留）。
      （已在汇报里作为「引擎与契约不一致」提出，由主 Agent 决定是否改价格层。）

    另注：停牌后复牌当天的收益 = (复牌价 − 停牌前价)/停牌前价，一次并入当日 ✓。
    """
    return np.where(ctx.traded(), ctx.ret(1), np.nan)


def _valid_count(ctx, ret: np.ndarray, n: int) -> np.ndarray:
    """窗口内「真有成交」的收益个数（停牌 / 复权脏数据日是 NaN，不计入）。"""
    return ctx.roll_count(np.where(np.isfinite(ret), 1.0, np.nan), n)


def _run_length(flag: np.ndarray) -> np.ndarray:
    """连续 True 的天数（含当日）；当日 False 则为 0。

    全向量化：`last_false(t) = max{ s <= t : flag(s) == False }`，
     streak = t − last_false（True 的位置），首行之前没有 False 时用 −1 哨兵
     保证「从面板第一行就开始的连续段」计数正确。O(T·C)，无 Python 逐日循环。
    """
    T = flag.shape[0]
    ar = np.arange(T, dtype=np.float64)[:, None]
    last_false = np.maximum.accumulate(np.where(flag, -1.0, ar), axis=0)
    return np.where(flag, ar - last_false, 0.0)


def _degenerate_guard(x: np.ndarray, lim: float) -> np.ndarray:
    """退化窗口的数值守卫：|值| > lim 或非有限 → NaN。

    ★ 为什么必需（实测）：一字板 / 长期停牌后价格不动 的股票，窗口内日收益**几乎全等**
      → 窗口中心二阶矩 m2 → 0 → skew = m3/m2^1.5、kurt = m4/m2² 数值爆炸。
      2012 年截面（26 个因子全跑）实测：偏度呈**完美双峰** —— 真实值 |skew| < 10，
      垃圾值 |skew| > 1000，中间 [10,1000] **一个格子都没有**；峰度的垃圾值全都 > 1e8。
      所以 lim 取 100（偏度）/ 1e4（峰度）两边都留 10 倍余量：
      20 日样本偏度的代数上界只有 √n ≈ 4.5、峰度现实上界约 20。
      顺带满足契约 §7 的第 3 条（|value| 不得 > 1e8 —— inf 会被引擎的 check 判为值域异常）。
    """
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x) & (np.abs(x) <= lim), x, np.nan)


def _market_ret(ctx) -> np.ndarray:
    """(T,) 沪深300 日收益，与面板日期**精确对齐**（网格外为 NaN）。

    用 `panel.place`（精确落格）而**不是** `asof_daily`（前向填充）：
    指数少一天就应该是 NaN，前向填充会造出「0 收益日」，把 beta 与相关系数静默拉低。
    实测 `index_daily` 的 000300.SH 交易日与 `basic_calendar` 的开市日
    完全一致（2010-01-04 起 3674/3674 天），正常窗口内不会出现 NaN。
    """
    panel = ctx.panel
    y0 = int(str(panel.dates[0])[:4])
    y1 = int(str(panel.dates[-1])[:4])
    df = ctx.dataset("index_daily",
                     columns=["ts_code", "trade_date", "pct_chg"], years=(y0, y1))
    if df is None or df.empty:
        return np.full(panel.T, np.nan)
    df = df[df["ts_code"] == MKT_INDEX]
    if df.empty:
        return np.full(panel.T, np.nan)
    days = ctx.date_col(df["trade_date"])
    val = pd.to_numeric(df["pct_chg"], errors="coerce").to_numpy(dtype=np.float64) / 100.0
    # 借第 0 列落格：place 要求 code 在 [0, C)，指数的「伪代码」统一用 0
    grid = panel.place(np.zeros(days.size, dtype=np.int64), days, val)
    return np.ascontiguousarray(grid[:, 0])


def _beta(ctx, ret: np.ndarray, mkt: np.ndarray, n: int, mc: int) -> np.ndarray:
    """对沪深300 的滚动 beta = Cov(ret, mkt) / Var(mkt)。mkt 传 (T,1) 自动广播。

    `roll_var` 只吃二维，所以把市场收益写成 (T,1) 而不是 (T,C) —— 结果完全一样，
    但省掉 3483 列重复计算。
    """
    return ctx.safe_div(ctx.roll_cov(ret, mkt, n, mc),
                        ctx.roll_var(mkt, n, mc), min_abs_den=1e-12)






@register(FactorSpec(
    name="vol_120", group="risk", deps=PRICE_DEPS,
    desc="120 个交易日已实现波动率（日收益滚动标准差）",
    formula="ReturnStd = StdDev(DailyReturn) over 6*21 trading days",
    start=None, warmup_days=W120, higher_is_better=False,
    note="参考库 risk 类只有 21/42/63/126/252 日的 return_std_*（因子库.md 7、Risk），"
         "本因子按家族文档取 120 个交易日窗口，公式形式逐字同 return_std_126d。"
         "min_count=60 = n//2。★ 窗口内缺 60 天时框架的方差分母仍是 120，会把长期停牌股的"
         "波动率压低约 30%，这批股票同时是低波因子会挑出来的——下游用它做风险控制时需注意。",
))
def vol_120(ctx):
    return ctx.roll_std(_ret(ctx), 120, min_count=60)


# ══════════════════════════════════════════════════════════════ 下行 / 不对称
@register(FactorSpec(
    name="downside_vol_ratio_20", group="risk", deps=PRICE_DEPS,
    desc="下行波动占比 = 20 日下行半波动 / 总波动（反向）",
    formula="neg = wide.clip(upper=0.0); var_all = (wide ** 2).rolling(20, min_periods=15).mean(); "
            "var_neg = (neg ** 2).rolling(20, min_periods=15).mean(); "
            "ratio = var_neg.pow(0.5) / var_all.pow(0.5).replace(0, np.nan)",
    start=None, warmup_days=W20, higher_is_better=False,
    note="逐字抄自 Class1 risk / downside_vol_ratio_20。口径注意：参考的 clip(upper=0) 把**上涨日**"
         "置 0（不是 NaN），所以分母是「窗口内所有交易日」而不是「下跌日个数」——即标准半方差比，"
         "不是「下跌日条件波动」。min_count=15 = min_periods；停牌日 ret 是 NaN，会随 r² 一起毒化窗口。",
))
def downside_vol_ratio_20(ctx):
    r = _ret(ctx)
    neg = np.minimum(r, 0.0)
    var_all = ctx.roll_mean(r * r, 20, min_count=15)
    var_neg = ctx.roll_mean(neg * neg, 20, min_count=15)
    ratio = ctx.safe_div(ctx.safe_sqrt(var_neg), ctx.safe_sqrt(var_all), min_abs_den=1e-12)
    # ★ clip 到 [0,1]：两个窗口的 NaN 模式完全相同（neg*neg 与 r*r 同时为 NaN），
    #   所以数学上 var_neg ≤ var_all、比值必在 [0,1]。实测 2012 年有 141/495019 个格子
    #   因两条 cumsum 路径的浮点残差略微 >1（最大 1.38），clip 不会伤到任何真实值。
    return np.clip(ratio, 0.0, 1.0)


@register(FactorSpec(
    name="downside_upside_vol_60", group="risk", deps=PRICE_DEPS,
    desc="下行/上行波动比 = 60 日负收益标准差 / 正收益标准差（反向）",
    formula="down = ret.where(ret < 0); up = ret.where(ret > 0); "
            "down_std = down.groupby(level=\"Code\").transform(lambda s: s.rolling(60, min_periods=5).std()); "
            "up_std = up.groupby(level=\"Code\").transform(lambda s: s.rolling(60, min_periods=5).std()); "
            "ratio = safe_divide(down_std, up_std + 1e-10)",
    start=None, warmup_days=W60, higher_is_better=False,
    note="逐字抄自 Class1 risk / downside_upside_vol_60。min_count=5 是**必须**的放松：正/负收益各自"
         "稀疏（60 日里约各 28 天），参考库自己也写「正/负样本各自稀疏:60日窗口内 min_periods=5 即可估计」。"
         "★ 框架语义：min_count 下分母仍是满窗 60，而分子分母同缩放，比值本身仍然成立。",
))
def downside_upside_vol_60(ctx):
    r = _ret(ctx)
    down = ctx.roll_std(np.where(r < 0, r, np.nan), 60, min_count=5)
    up = ctx.roll_std(np.where(r > 0, r, np.nan), 60, min_count=5)
    return ctx.safe_div(down, up, min_abs_den=1e-12)


@register(FactorSpec(
    name="gain_loss_asymmetry_60", group="risk", deps=PRICE_DEPS,
    desc="涨跌幅度不对称 = 60 日平均涨幅 / |平均跌幅|（涨多跌少排前）",
    formula="mean_up = ret_w.where(ret_w > 0).rolling(60, min_periods=10).mean(); "
            "mean_down = ret_w.where(ret_w < 0).rolling(60, min_periods=10).mean(); "
            "asym = safe_divide(mean_up, mean_down.abs() + 1e-10)",
    start=None, warmup_days=W60, higher_is_better=True,
    note="逐字抄自 Class1 risk / gain_loss_asymmetry_60。min_count=10 = 参考库 min_periods"
         "（涨/跌样本各自稀疏）。与 downside_upside_vol_60 互补：这个是**一阶矩**比，那个是二阶矩比。",
))
def gain_loss_asymmetry_60(ctx):
    r = _ret(ctx)
    up = ctx.roll_mean(np.where(r > 0, r, np.nan), 60, min_count=10)
    down = ctx.roll_mean(np.where(r < 0, r, np.nan), 60, min_count=10)
    return ctx.safe_div(up, np.abs(down), min_abs_den=1e-12)


# ══════════════════════════════════════════════════════════════ 回撤 / 尾部
@register(FactorSpec(
    name="max_drawdown_120", group="risk", deps=PRICE_DEPS,
    desc="120 日回撤深度 = 后复权价 / 最近 120 日最高价 − 1（≤0）",
    formula="peak = adj.groupby(level=\"Code\").transform(lambda s: s.rolling(120, min_periods=60).max()); "
            "drawdown = adj / peak.replace(0, np.nan) - 1.0  # ≤ 0",
    start=None, warmup_days=W120, higher_is_better=True,
    note="★ 窗口口径（想清楚后写在这里）：dd = hfq_close / roll_max(hfq_close,120) − 1，"
         "即「当前价相对**含当日在内的最近 120 个交易日**最高价」的回撤，恒 ≤0（roll_max 的窗口含当日，"
         "故比值 ≤1；停牌期间 hfq 前向填充，不会造假回撤）。"
         "**不用**教科书式的「窗口内峰谷最大回撤」：那个要窗口内的运行最大值（峰值可早于窗口起点），"
         "一次 roll_max 表达不了，得物化 (T,C,120) 张量；而且参考库明说 drawdown_120 是"
         "「8.10 删除的 max_drawdown_120 的合规重建」，口径就是本文这个。min_count=60 = min_periods。",
))
def max_drawdown_120(ctx):
    px = ctx.hfq("close")
    peak = ctx.roll_max(px, 120, min_count=60)
    return ctx.safe_div(px, peak, min_abs_den=1e-12) - 1.0


@register(FactorSpec(
    name="max_drawdown_60", group="risk", deps=PRICE_DEPS,
    desc="60 日回撤深度 = 后复权价 / 最近 60 日最高价 − 1（≤0）",
    formula="peak = adj.groupby(level=\"Code\").transform(lambda s: s.rolling(60, min_periods=30).max()); "
            "drawdown = adj / peak.replace(0, np.nan) - 1.0  # ≤ 0",
    start=None, warmup_days=W60, higher_is_better=True,
    note="逐字抄自 Class1 risk / drawdown_60（即被删除的 max_drawdown_60 的合规重建），"
         "窗口口径同 max_drawdown_120。min_count=30 = min_periods。",
))
def max_drawdown_60(ctx):
    px = ctx.hfq("close")
    peak = ctx.roll_max(px, 60, min_count=30)
    return ctx.safe_div(px, peak, min_abs_den=1e-12) - 1.0


@register(FactorSpec(
    name="drawdown_duration_120", group="risk", deps=PRICE_DEPS,
    desc="回撤持续期 = 价格低于 120 日滚动前高的连续天数（上限 120）",
    formula="rolling_high = adj.groupby(level=\"Code\").transform(lambda s: s.rolling(120, min_periods=1).max()); "
            "in_dd = adj.lt(rolling_high * 0.999); duration = _consecutive_count(in_dd).clip(upper=120)",
    start=None, warmup_days=W120, higher_is_better=False,
    note="逐字抄自 Class1 risk / drawdown_duration_120。★ 偏离任务书的一句话描述：任务书写「px < 历史最高」，"
         "本实现按参考库用 **120 日滚动前高**（roll_max(px,120)）——全历史最高需要全历史面板，"
         "增量重算无法给出「与全量一致」的 warmup。"
         "0.999 的容忍带、上限 120 都照抄参考库；min_count=1 = 参考 min_periods=1（上市首日即有前高）。"
         "连续段用 maximum.accumulate 向量化，没有逐日 Python 循环。",
))
def drawdown_duration_120(ctx):
    px = ctx.hfq("close")
    high = ctx.roll_max(px, 120, min_count=1)
    in_dd = np.isfinite(px) & (px < high * 0.999)
    dur = np.minimum(_run_length(in_dd), 120.0)
    return np.where(np.isfinite(px), dur, np.nan)


@register(FactorSpec(
    name="ulcer_index_20", group="risk", deps=PRICE_DEPS,
    desc="溃疡指数 = 20 日窗口内回撤平方均值的平方根（回撤面积，反向）",
    formula="peak = adj.groupby(level=\"Code\").transform(lambda s: s.rolling(20, min_periods=10).max()); "
            "dd = adj / peak.replace(0, np.nan) - 1.0; "
            "dd_sq = (dd ** 2).groupby(level=\"Code\").transform(lambda s: s.rolling(20, min_periods=10).mean()); "
            "ulcer = dd_sq.pow(0.5)",
    start=None, warmup_days=92, higher_is_better=False,
    note="逐字抄自 Class1 risk / ulcer_index_20。回撤口径同 max_drawdown_*（对 20 日滚动前高）。"
         "嵌套窗口：前高 20 日 + 回撤均值 20 日 = 40 个交易日，故 warmup=40×1.8+20≈92（比单窗口的大）。"
         "min_count=10 = 参考 min_periods。",
))
def ulcer_index_20(ctx):
    px = ctx.hfq("close")
    dd = ctx.safe_div(px, ctx.roll_max(px, 20, min_count=10), min_abs_den=1e-12) - 1.0
    return ctx.safe_sqrt(ctx.roll_mean(dd * dd, 20, min_count=10))




@register(FactorSpec(
    name="var_95_20", group="risk", deps=PRICE_DEPS,
    desc="20 日 VaR(95%) = 日收益的 5% 历史分位（≤0，越负尾部越厚）",
    formula="var = _ret(daily).groupby(level=\"Code\").transform("
            "lambda s: s.rolling(20, min_periods=10).quantile(0.05))",
    start=None, warmup_days=W20, higher_is_better=True,
    note="逐字抄自 Class1 risk / var_95_20。★ 用**历史分位数**（线性插值，与 pandas rolling.quantile 同口径），"
         "不假设正态。min_count=10 = 参考 min_periods。"
         "实现走 `fea.mathx.roll_quantile`（mathx 里有、但 FactorContext 没转发，见模块 docstring）；"
         "它对窗口内的 NaN 是**忽略**而不是毒化，与 pandas 的 min_periods 语义一致。",
))
def var_95_20(ctx):
    return mx.roll_quantile(_ret(ctx), 20, 0.05, min_count=10)


@register(FactorSpec(
    name="cvar_95_120", group="risk", deps=PRICE_DEPS,
    desc="120 日 CVaR(95%) = 窗口内 ≤5% 分位那部分收益的均值（尾部期望损失）",
    formula="quantiles = [ret_w.rolling(120, min_periods=60).quantile(q) for q in (0.01, 0.02, 0.03, 0.04, 0.05)]; "
            "cvar_w = sum(quantiles) / len(quantiles)",
    start=None, warmup_days=W120, higher_is_better=True,
    note="公式行逐字抄自 Class1 risk / cvar_95_120。★ 偏离参考库**实现**：参考用 5 个分位数取平均"
         "做黎曼近似（它自己的注释说明这是为了向量化），本实现按任务书口径取「窗口内 ≤5% 分位那部分的"
         "**均值**」= 精确历史 CVaR（参考库「意义」里写的也是「取最坏 5% 日收益的均值」）。"
         "120 日窗口下尾部期望 6 个样本，两种算法差异很小。min_count=60 = 参考 min_periods。"
         "实现：roll_quantile 取 q05（NaN 忽略）→ 掩码 r<=q05 → 窗口内尾部求和 / 尾部天数（都是 cumsum 类，无逐日循环）。",
))
def cvar_95_120(ctx):
    r = _ret(ctx)
    q05 = mx.roll_quantile(r, 120, 0.05, min_count=60)
    tail = np.isfinite(r) & (r <= q05)
    s = ctx.roll_sum(np.where(tail, r, 0.0), 120)
    n = ctx.roll_count(np.where(tail, 1.0, np.nan), 120)
    return ctx.safe_div(s, n, min_abs_den=0.5)      # 每个窗口至少 1 个尾部样本


@register(FactorSpec(
    name="tail_risk_pct_60", group="risk", deps=PRICE_DEPS,
    desc="尾风险频率 = 60 日内 |z|>2 的极端收益占比（反向）",
    formula="z = safe_divide(ret - mean60, std60 + 1e-10); "
            "freq = z.abs().gt(2).groupby(level=\"Code\").transform(lambda s: s.rolling(60, min_periods=30).mean())",
    start=None, warmup_days=W120, higher_is_better=False,
    note="逐字抄自 Class1 risk / tail_risk_pct_60。嵌套窗口：60 日标准化 + 60 日频率 = 120 个交易日，"
         "故 warmup=240（同 N=120 的口径）。min_count=30 = 参考 min_periods。"
         "★ 偏离：参考库把 |z| 算不出来的日子（停牌）当 False 计进分母（pandas 的 NaN.gt() → False），"
         "会把长期停牌股的尾风险稀释掉；本实现额外要求窗口内 ≥30 个真实交易日，否则置 NaN。",
))
def tail_risk_pct_60(ctx):
    r = _ret(ctx)
    z = ctx.safe_div(r - ctx.roll_mean(r, 60, min_count=30),
                     ctx.roll_std(r, 60, min_count=30), min_abs_den=1e-12)
    freq = ctx.roll_mean((np.abs(z) > 2).astype(np.float64), 60, min_count=30)
    # 同 loss_probability_20：概率必须落在 [0,1]，挡掉 cumsum 差分的 −1e−16
    return np.where(_valid_count(ctx, r, 60) >= 30, np.clip(freq, 0.0, 1.0), np.nan)


@register(FactorSpec(
    name="ret_skew_20", group="risk", deps=PRICE_DEPS,
    desc="20 日收益偏度（日收益滚动三阶矩，反向）",
    formula="skew = _ret_wide(daily).rolling(20, min_periods=15).skew()",
    start=None, warmup_days=W20, higher_is_better=False,
    note="逐字抄自 Class1 risk / ret_skew_20。min_count=15 = 参考 min_periods。"
         "用 `ctx.roll_skew`（与 pandas `.rolling(n).skew()` 同口径：有偏偏度再乘 sqrt(n(n-1))/(n-2) 去偏）。"
         "★ 数值守卫 |skew| > 100 → NaN：一字板/长期停牌后价格不动会让 m2→0、m3/m2^1.5 爆炸，"
         "实测截面双峰（真实值 <10、垃圾值 >1000、中间 0 个格子）。",
))
def ret_skew_20(ctx):
    return _degenerate_guard(ctx.roll_skew(_ret(ctx), 20, min_count=15), 100.0)


@register(FactorSpec(
    name="ret_kurt_20", group="risk", deps=PRICE_DEPS,
    desc="20 日收益峰度（日收益滚动四阶矩，反向）",
    formula="kurt = _ret_wide(daily).rolling(20, min_periods=15).kurt()",
    start=None, warmup_days=W20, higher_is_better=False,
    note="逐字抄自 Class1 risk / ret_kurt_20。min_count=15 = 参考 min_periods。"
         "★ 返回的是**超额**峰度（减 3 之后，正态 ≈ 0），与 pandas `.kurt()` 同口径；"
         "不是原始的四阶矩比。"
         "★ 数值守卫 |kurt| > 1e4 → NaN：同 ret_skew_20 的退化窗口（m2→0 → m4/m2² 爆炸），"
         "实测垃圾值全部 >1e8（含 ±inf），真实值 <1e3，中间同样是空的。",
))
def ret_kurt_20(ctx):
    return _degenerate_guard(ctx.roll_kurt(_ret(ctx), 20, min_count=15), 1e4)




# ══════════════════════════════════════════════════════════════ 市场 / 系统性
@register(FactorSpec(
    name="beta_60", group="risk", deps=MKT_DEPS,
    desc="60 日市场贝塔（对沪深300，反向）",
    formula="beta = _rolling_beta(wide, mkt, 60, 30)  # "
            "Beta = Cov(StockReturn, IndexReturn) / Var(IndexReturn) over a rolling window",
    start=None, warmup_days=W60, higher_is_better=False, version=2,
    note="★ 基准偏离：参考库的 `_market_proxy(wide)` 是**全池等权**市场收益，本实现按要求改用"
         "**沪深300（000300.SH）**——它是可投资的、有真实指数数据的基准，且与下游基准一致。"
         "min_count=30 = 参考 `_rolling_beta(..., 60, 30)`。"
         "实现：市场收益取 (T,1) 广播进 roll_cov/roll_var（**两者都按各自窗口内的有效对数/有效天数"
         "做分母**，与 pandas 的 `rolling().cov()/var()` 同口径）。"
         "停牌日 ret 为 NaN → 该窗口若有效天数不足 30 则为 NaN。"
         "★★ v2（2026-09-17）：`mathx.roll_cov` 修了分母用固定 n 的 bug（见该函数注释）——"
         "本因子是它的下游，历史值整体重算过；修前在「窗口内有停牌/缺失」的股票上是失真的值。",
))
def beta_60(ctx):
    return _beta(ctx, _ret(ctx), _market_ret(ctx)[:, None], 60, 30)




@register(FactorSpec(
    name="idio_vol_60", group="risk", deps=MKT_DEPS,
    desc="60 日特质波动率 = 剔除市场暴露后残差的标准差（反向）",
    formula="beta = _rolling_beta(wide, mkt, 60, 30); resid = wide - beta.multiply(mkt, axis=0); "
            "idio = resid.rolling(60, min_periods=30).std()",
    start=None, warmup_days=W60, higher_is_better=False, version=2,
    note="逐字抄自 Class1 risk / idio_vol_60；基准同 beta_60（沪深300）。"
         "min_count=30 = 参考 min_periods。残差 = ret − beta×mkt，beta 用同一 60 日窗口的滚动估计"
         "（与参考的 _rolling_beta 一致，不做全样本回归，故是严格 PIT 的）。"
         "★★ v2（2026-09-17）：随 `mathx.roll_cov` 的分母 bug 修复整体重算"
         "（残差里含 beta，beta 错则残差也错）。",
))
def idio_vol_60(ctx):
    r = _ret(ctx)
    mkt = _market_ret(ctx)[:, None]
    resid = r - _beta(ctx, r, mkt, 60, 30) * mkt
    return ctx.roll_std(resid, 60, min_count=30)


# ══════════════════════════════════════════════════════════════ 波动结构
@register(FactorSpec(
    name="vol_of_vol_20", group="risk", deps=PRICE_DEPS,
    desc="波动之波动 = |日收益| 的 20 日标准差（反向）",
    formula="roll(df, \"absret\", 20, \"std\", min_periods=5)",
    start=None, warmup_days=W20, higher_is_better=False,
    note="逐字抄自 Class1 risk / vol_of_vol_20d。口径就是参考库的「|日收益| 的 20 日 std」"
         "（不是「波动率序列的波动」——那个在参考库里叫 vol_of_vol_60，本家族未选）。"
         "min_count=5 = 参考 min_periods。",
))
def vol_of_vol_20(ctx):
    return ctx.roll_std(np.abs(_ret(ctx)), 20, min_count=5)


@register(FactorSpec(
    name="vol_clustering_20", group="risk", deps=PRICE_DEPS,
    desc="波动聚集 = |日收益| 的 20 日 lag-1 自相关（反向）",
    formula="abs_ret = _ret(daily).abs().unstack(\"Code\"); lag = abs_ret.shift(1); "
            "ac = abs_ret.rolling(20, min_periods=10).corr(lag)",
    start=None, warmup_days=60, higher_is_better=False,
    note="逐字抄自 Class1 risk / vol_clustering_20。min_count=10 = 参考 min_periods。"
         "shift(1) 在面板上就是「上一交易日」（面板本身即交易日历），不会跨股错位；"
         "warmup 比 W20 多 4 天是为了覆盖这个 lag。"
         "★ 数值守卫 |corr| > 1 → NaN：相关系数在数学上不可能越界，越界只可能是"
         "「窗口内 |收益| 几乎恒定 → 分母方差 ≈ 0」的退化窗口（实测 296/497093 个格子，"
         "最大 4.0），此时真实相关系未定义，置 NaN 而不是 clip（clip 会造出假的 ±1）。",
))
def vol_clustering_20(ctx):
    a = np.abs(_ret(ctx))
    c = ctx.roll_corr(a, ctx.shift(a, 1), 20, min_count=10)
    return np.where(np.abs(c) <= 1.0, c, np.nan)


# ══════════════════════════════════════════════════════════════ 风险调整收益
@register(FactorSpec(
    name="sortino_ratio_60", group="risk", deps=PRICE_DEPS,
    desc="60 日 Sortino 比率 = 均收益 / 下行波动（正向）",
    formula="mean60 = ret.groupby(level=\"Code\").transform(lambda s: s.rolling(60, min_periods=30).mean()); "
            "down = ret.where(ret < 0); "
            "down_std = down.groupby(level=\"Code\").transform(lambda s: s.rolling(60, min_periods=5).std()); "
            "sortino = safe_divide(mean60, down_std + 1e-10)",
    start=None, warmup_days=W60, higher_is_better=True,
    note="逐字抄自 Class1 risk / sortino_ratio_60。★ 偏离任务书的一句话描述：任务书写「60 日**超额**收益均值」，"
         "参考库用的是**原始**均收益（不减无风险利率，A 股日频也没有合适的无风险日利率口径），照抄参考库。"
         "分母 = 负收益子样本的 60 日 roll_std，min_count=5（必须的放松：60 日里负收益约 28 天）；"
         "分子 min_count=30 = 参考 min_periods。",
))
def sortino_ratio_60(ctx):
    r = _ret(ctx)
    mean60 = ctx.roll_mean(r, 60, min_count=30)
    down_std = ctx.roll_std(np.where(r < 0, r, np.nan), 60, min_count=5)
    return ctx.safe_div(mean60, down_std, min_abs_den=1e-12)


# ══════════════════════════════════════════════════════════════ 行为 / 频率
@register(FactorSpec(
    name="panic_selling_ratio_60", group="risk", deps=PRICE_DEPS,
    desc="放量下跌占比 = 60 日内「放量且下跌」天数 / 放量天数（恐慌抛售排前）",
    formula="ma20 = vol.groupby(level=\"Code\").transform(lambda s: s.rolling(20, min_periods=10).mean()); "
            "big = vol.gt(1.5 * ma20); down = ret.lt(0); panic = (big & down).astype(float); "
            "n_big = big.astype(float)...rolling(60, min_periods=1).sum(); "
            "n_panic = panic...rolling(60, min_periods=1).sum(); ratio = safe_divide(n_panic, n_big)",
    start=None, warmup_days=164, higher_is_better=True,
    note="逐字抄自 Class1 risk / panic_selling_ratio_60。嵌套窗口：20 日均量 + 60 日统计 = 80 个交易日，"
         "故 warmup=80×1.8+20≈164。min_count：均量 10（=参考 min_periods）、两个 sum 用 1"
         "（参考的 min_periods=1：只要有 ≥1 个放量日就能估计占比，0 个放量日 → 0/0 → NaN）。"
         "★ 用 `ctx.px(\"vol\")`（流量，停牌日为 NaN）—— 补 0 会把停牌当成「缩量日」。"
         "放量日 = vol > 1.5×20 日均量，停牌日与均量为 NaN 的头部都不算放量日（与 pandas 的 NaN.gt() → False 一致）。",
))
def panic_selling_ratio_60(ctx):
    r = _ret(ctx)
    vol = ctx.px("vol")
    big = (vol > 1.5 * ctx.roll_mean(vol, 20, min_count=10)).astype(np.float64)
    panic = big * (r < 0).astype(np.float64)
    # ctx.roll_sum 不转发 min_count（见汇报的引擎扩展请求），这里直接用 mathx
    n_big = mx.roll_sum(big, 60, min_count=1)
    n_panic = mx.roll_sum(panic, 60, min_count=1)
    return ctx.safe_div(n_panic, n_big, min_abs_den=0.5)   # 分母 <1 无意义（0 个放量日）




