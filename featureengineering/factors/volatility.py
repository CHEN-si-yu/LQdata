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
    paired_market = np.where(np.isfinite(ret) & np.isfinite(mkt), mkt, np.nan)
    return ctx.safe_div(ctx.roll_cov(ret, paired_market, n, mc),
                        ctx.roll_var(paired_market, n, mc), min_abs_den=1e-12)






@register(FactorSpec(
    name="vol_120", version=2, group="risk", deps=PRICE_DEPS,
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






# ══════════════════════════════════════════════════════════════ 回撤 / 尾部
















@register(FactorSpec(
    name="ret_skew_20", version=2, group="risk", deps=PRICE_DEPS,
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






# ══════════════════════════════════════════════════════════════ 市场 / 系统性
@register(FactorSpec(
    name="beta_60", group="risk", deps=MKT_DEPS,
    desc="60 日市场贝塔（对沪深300，反向）",
    formula="beta = _rolling_beta(wide, mkt, 60, 30)  # "
            "Beta = Cov(StockReturn, IndexReturn) / Var(IndexReturn) over a rolling window",
    start=None, warmup_days=W60, higher_is_better=False, version=4,
    note='Cov与Var均使用股票收益、指数收益成对有效的同一窗口，min_count=30。',
))
def beta_60(ctx):
    return _beta(ctx, _ret(ctx), _market_ret(ctx)[:, None], 60, 30)




@register(FactorSpec(
    name="idio_vol_60", group="risk", deps=MKT_DEPS,
    desc="60 日特质波动率 = 剔除市场暴露后残差的标准差（反向）",
    formula="beta = _rolling_beta(wide, mkt, 60, 30); resid = wide - beta.multiply(mkt, axis=0); "
            "idio = resid.rolling(60, min_periods=30).std()",
    start=None, warmup_days=260, higher_is_better=False, version=4,
    note='滚动60日beta残差再取60日标准差；119日有效预热。Beta使用成对样本。',
))
def idio_vol_60(ctx):
    r = _ret(ctx)
    mkt = _market_ret(ctx)[:, None]
    resid = r - _beta(ctx, r, mkt, 60, 30) * mkt
    return ctx.roll_std(resid, 60, min_count=30)




@register(FactorSpec(
    name="vol_clustering_20", version=2, group="risk", deps=PRICE_DEPS,
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




# ══════════════════════════════════════════════════════════════ 行为 / 频率







