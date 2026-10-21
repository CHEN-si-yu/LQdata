"""日内微观结构·深化（24 个）—— 全部日频产出、只主板、跟随 `default_start`。

数据源：`data/derived/intraday/`（`fea/intraday.py` 把 `stock_history_5min`
预聚合成 34 个日频字段）+ 价格层。

本文件**只新增**因子，不改 `factors/intraday.py`（那 9 个 `idt_*` 一字不动）。
分工：`intraday.py` 挖的是**隔夜腿 / 全天聚合比**；本文件挖的是
**会话切分（上半天 vs 下半天）、极差型波动估计量、收益路径的集中度、
跨会话区间结构、放量时点**——五个体量都不小、此前**零覆盖**的维度。

═══════════════════════════════════════════════════════════════════════════
一、★★ 日内层的五条硬事实（全部实测，写在这里免得后人再踩一次）
═══════════════════════════════════════════════════════════════════════════

1. **`path_len ≡ absret_sum`**。层里两者是同一个表达式 `rsum(np.abs(r0))`
   （`fea/intraday.py` 的 `out` 字典）。所以参考库那种
   「路径效率 = absret_sum / path_len」在这里**恒等于 1.0**，是个常数因子。
   真正有意义的写法是 `|ret_sum| / absret_sum`（净行程 / 总行程），
   但那个名字（`idt_path_efficiency`）在 2026-09-17 的「去糟粕」里已被删，
   且 20 日尺度的同类思想已有注册资本 `ret_efficiency_20`（Kaufman ER）
   ⇒ **本文件不做路径效率**，理由记在此处，避免下一轮再试。

2. **`open5 ≡ am_open5`、`close5 ≡ pm_close5`**。上午段的**第一根**就是全天第一根
   （`out[f"{tag}_open5"][r] = o[s2]`，`s2` 是段首）；下午段的**最后一根**就是全天最后一根
   （`out[f"{tag}_close5"][r] = c[e2_ - 1]`）。所以「首根/末根 K 线动量」这类
   参考库因子**造不出来**——两者逐位相同。
   `am_close5`（11:30）与 `pm_open5`（13:05）才是层里**唯一新增的两个价格点**。

3. **`n_zero` 多算一根**。段首那根棒的 `prev_c = NaN → r = NaN → r0 = 0`
   （`r0 = np.nan_to_num(r, nan=0.0)`），于是每个 (股票, 日) 段的 `n_zero`
   都**恒含一个假的零收益**。任何「零收益棒占比」必须写
   `(n_zero − 1) / (n_bars − 1)`。（对称的那个坑 `n_up` 已经在
   `idt_up_minutes_ratio` 的 note 里记过。）

4. **`hi5/lo5` 不一定包住 `open5/close5`**（上游 `high/low` 不总是包住自己那根棒）。
   实测 2012-2016 有 0.075% 的 (股票,日) `close5 ∉ [lo5, hi5]`。
   于是 `ln(hi/lo)` 会变负、`(hi−lo)` 会变负。**取对数/开方前必须做区间扩张**
   —— 用本文件的 `_rng()`（与 `factors/intraday.py::_seg_range` 同一套
   `np.maximum(hi, np.fmax(o,c))` / `np.minimum(lo, np.fmin(o,c))`）。

5. **`min_count = N // 2`**（与 `intraday.py` §7 一致，也是参考库自己的
   `min_periods`）。日内层的缺失**不是随机的**（停牌、上市/退市窗口），
   要求窗口全有效会让绝大多数股票整窗 NaN。但**光有 min_count 不够**：
   窗口只在最近几天缺值时，滚动值会一直吐出**陈旧**的值，把长期停牌悄悄藏起来
   ⇒ 本文件一律用 `_roll()`，它**额外汇总「当日输入必须有限」**。

═══════════════════════════════════════════════════════════════════════════
二、明确做不到的（写下来，别再试）
═══════════════════════════════════════════════════════════════════════════

层里只存 `ret_sum / ret2_sum / absret_sum / posret2_sum / negret2_sum`，
**没有逐根棒的数据**，所以下面这些一概做不了：

  · 双幂变差 `bv_daily`、跳跃 `rjump_*`、四次变差 `rq_intraday`
    —— 需要**相邻棒乘积** `Σ r_i·r_{i+1}`（层里没有 lag-1 交叉项）
  · 成交量偏度/峰度、量能 U 形分数 —— 需要 `Σv³/Σv⁴` 或 48 根的量能路径
  · 首/末 30 分钟切片（`open_30_momentum` / `last30_ret` / `last30_vol_share`）
  · 逐分钟收益自相关、Kyle λ、已实现价差、真假 `am_pm_rv_ratio`
    —— 需要逐 bar 收益/成交量

解锁它们都要**扩 `fea/intraday.py::FIELDS`**（并 `IntradayLayer.version +1`
⇒ 派生层整层重建 + 已有 9 个 `idt_*` 全部重算）。用户 2026-09-18 明确
**本轮不扩**。清单留在 `QUANT_PLATFORM.md` §52 与
`factors/intraday.py` 的附录里，属下一轮的 P1。

═══════════════════════════════════════════════════════════════════════════
三、与 `intraday.py` 的关系（避免读者以为重复）
═══════════════════════════════════════════════════════════════════════════

`intraday.py` 已有的 9 个：隔夜跳空、隔夜/日内分解、`rv_daily`、`rv_term_structure_slope`、
`intraday_max_drawdown`、`up_minutes_ratio`、`vol_stability`、`lo_pos`、`intraday_ma_{5,20,60}d`。
本文件**一个都不重做**：`idt_rv_daily = sqrt(ret2_sum)` 是全天 RV，本文件的
`id2_parkinson_vol` 是**极差型**估计量（不同统计量）；
`idt_intraday_max_drawdown` 是回撤侧，本文件补反弹侧与两者之比。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

# 跟随 conf/config.yaml 的 default_start（当前 2012-01-01）。日内层覆盖 2010 起。
ID2_START = None

# warmup_days = 「计算窗口要往前多读多少**日历天**」。滚动窗口 N 按 N×1.8+20 给。
W_D = 60      # 纯当日量（层已聚合完，无滚动）——给两个月冗余，跨年/停牌都够
W_20 = 56     # 20 交易日窗：20×1.8+20（≈37 个交易日 > 20）

# 每个吃日内层的因子的 note 都要带这句（本文件 24 个全部是 5min 口径）
_DEG5 = ("★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 "
         "stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。")


# ══════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════

def _f(ctx, field: str) -> np.ndarray:
    """日内层字段（当日口径、未复权）。停牌/未上市 -> NaN（精确落格，不 ffill）。"""
    return np.asarray(ctx.intraday_field(field), dtype=np.float64)


def _rng(ctx, hi_f: str, lo_f: str, op_f: str, cl_f: str):
    """一段（全日或半日）的 [最低, 最高]，用**本段自己的开/收盘扩张**。见 docstring §一.4。"""
    o = _f(ctx, op_f)
    c = _f(ctx, cl_f)
    hi = np.maximum(_f(ctx, hi_f), np.fmax(o, c))
    lo = np.minimum(_f(ctx, lo_f), np.fmin(o, c))
    return hi, lo


def _day_range(ctx):
    return _rng(ctx, "hi5", "lo5", "open5", "close5")


def _roll(ctx, x: np.ndarray, n: int, mc: int) -> np.ndarray:
    """滚动均值 + 「当日输入必须有限」守卫（见 docstring §一.5）。"""
    return np.where(np.isfinite(x), ctx.roll_mean(x, n, mc), np.nan)












def _pm_vwap(ctx) -> np.ndarray:
    return ctx.safe_div(_f(ctx, "pm_amt"), _f(ctx, "pm_vol"), 1e-6)


# ══════════════════════════════════════════════════════════════════════
# 1. 极差型波动估计量（2 个）
#    —— 与已注册的 `idt_rv_daily`（收盘价已实现波动）是**不同的统计量**：
#       用的是 high/low（Parkinson）或四价（Garman-Klass），不依赖逐棒数据。
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="id2_parkinson_vol", version=2,
    group="intraday",
    deps=("stock_history_5min",),
    desc="Parkinson 极差波动（当日）= sqrt( ln(hi/lo)^2 / (4 ln2) )",
    formula="parkinson_vol = sqrt(log(high/low)**2 / (4*log(2)))",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=False,
    note=(_DEG5 + " 逐字抄参考库 Class3 `parkinson_vol`（high/low 用本日值，"
          "不需复权因子）。★ 必做区间扩张：上游 high/low 不总包住 close/open"
          "（实测 0.075% 越界），不扩张会让 ln(hi/lo) 变负、开方出 NaN 或假值。"
          "⚠ 与已注册的 `amihud_parkinson_ratio` 有关联但**不是**同一个量："
          "那个是 20 日 Parkinson 均值与 Amihud 之比，本因子是单日水平。"
          "方向取负（波动越低越好），与全库波动类口径一致。"),
))
def id2_parkinson_vol(ctx):
    hi, lo = _day_range(ctx)
    with np.errstate(all="ignore"):
        x = ctx.safe_div(hi, lo, 1e-12)
        pk = np.log(x) ** 2 / (4.0 * np.log(2.0))
    return ctx.safe_sqrt(pk)






# ══════════════════════════════════════════════════════════════════════
# 2. 收益路径结构（2 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="id2_ret_concentration", version=2,
    group="intraday",
    deps=("stock_history_5min",),
    desc="日内收益路径的集中度 = n·Σr² / (Σ|r|)²（反参与比）",
    formula="concentration = n_bars * ret2_sum / absret_sum**2",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=False,
    note=(_DEG5 + " **本文件新造**（参考库没有；它靠逐根棒的数据直接算更高阶矩，"
          "本项目只有聚合量）。"
          "数学性质：由柯西-施瓦茨，`1 ≤ n·Σr²/(Σ|r|)² ≤ n`。"
          "取 1 ⇒ **单根棒扛下了整条路径**（其余 47 根不动）= 跳跃/瞬时重定价；"
          "取 n ⇒ 每根棒幅度相同 = 均匀推进。"
          "**这是在没有逐棒数据的前提下，唯一还能拿到的「路径形状」高阶量**"
          "（名字取自逆参与比 / inverse participation ratio）。"
          "方向取负：集中度高 = 单点驱动，与「平滑趋势」相比更可能是噪声。"),
))
def id2_ret_concentration(ctx):
    n = _f(ctx, "n_bars")
    s = _f(ctx, "absret_sum")
    c = ctx.safe_div(n * _f(ctx, "ret2_sum"), s ** 2, 1e-12)
    return np.where((n >= 2) & np.isfinite(c), c, np.nan)


@register(FactorSpec(
    name="id2_hi_lo_pos_gap", version=2,
    group="intraday",
    deps=("stock_history_5min",),
    desc="当日最高点与最低点的**时点**间隔（日内在时间轴上的铺开程度）",
    formula="hi_pos - lo_pos   （两者都是段内 argmax/argmin 归一化到 [0,1]）",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=(_DEG5 + " **本文件新造**。`hi_pos`/`lo_pos` 层里已经归一化到 [0,1]"
          "（= 段内 argmin/argmax ÷ (n_bars−1)），**不要再除 n_bars**。"
          "`|hi_pos − lo_pos|` 大 ⇒ 极值分居上下午，日内走了单边；"
          "小 ⇒ 高低点挨在一起，V 形/倒 V 反转。"
          "与 `idt_lo_pos` 不重复：那个只看**最低点在哪**（位置水平），"
          "本因子看**两个极值的时间距离**（位置差的绝对值，与整体早晚无关）。"
          "取绝对值后再给方向：铺得开 = 日内趋势清晰。"),
))
def id2_hi_lo_pos_gap(ctx):
    return np.abs(_f(ctx, "hi_pos") - _f(ctx, "lo_pos"))


# ══════════════════════════════════════════════════════════════════════
# 3. 上/下午会话切分（8 个）
#    —— 参考库 Class3 的 `am_*` / `pm_*` / `lunch_break_*` 一族。
#       层里的 am_/pm_ 前缀字段**此前只被 0 个因子消费过**。
# ══════════════════════════════════════════════════════════════════════















# ══════════════════════════════════════════════════════════════════════
# 4. 会话 VWAP 与收盘定价（3 个）
# ══════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="id2_close_vs_pm_vwap", version=2,
    group="intraday",
    deps=("stock_history_5min",),
    desc="收盘价相对下午 VWAP 的偏离（收盘集合竞价的定价压力）",
    formula="return cross_sectional_rank(-_metric(context, 'vwap_dev'))",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=False,
    note=(_DEG5 + " 源自参考库 Class3 `vwap_dev`（方向负）。"
          "★ 与已注册的 `vwap_daily_deviation` **不是**同一个量：那个是"
          "**全天 VWAP** 与收盘的偏离，本因子把 VWAP 收窄到**下午段**"
          "（13:05–15:00），从而把「上午的成交重心」从基准里剔掉 —— "
          "剩下的是纯粹的**尾盘定价**：A 股 14:57–15:00 是收盘集合竞价，"
          "收盘 > 下午均价 ⇒ 尾盘有主动买盘承接。"
          "A 股实施收盘集合竞价制度后（2018-08 起深市、2018-08-20 起沪市），"
          "这个量的信息含量前后**不同质** —— 下游若按年切分训练需注意。"
          "方向取负（参考库口径：偏离越大排后）。"),
))
def id2_close_vs_pm_vwap(ctx):
    pv = _pm_vwap(ctx)
    return ctx.safe_div(_f(ctx, "close5") - pv, pv, 1e-8)


@register(FactorSpec(
    name="id2_close_vs_pm_vwap_20", version=2,
    group="intraday",
    deps=("stock_history_5min",),
    desc="收盘相对下午 VWAP 偏离的 20 日水平（持续性尾盘溢价/折价）",
    formula="roll_mean(close5/pm_vwap - 1, 20, min_count=10)",
    start=ID2_START,
    warmup_days=W_20,
    higher_is_better=False,
    note=(_DEG5 + " `id2_close_vs_pm_vwap` 的 20 日滚动均值。"
          "**为什么值得单独留**：单日的尾盘偏离有相当一部分是买卖盘随机冲击，"
          "但「这只股票**长期**收盘都高于下午均价」是一个持续性的"
          "资金行为特征（机构按收盘价建仓/指数化资金），"
          "在截面上与单日值的相关性不高（单日噪声占主导）。"
          "★ 2018-08 收盘集合竞价制度变更前，这个量的均值结构不同 —— "
          "跨年比较时留意（与单日版同一条 note）。"),
))
def id2_close_vs_pm_vwap_20(ctx):
    return _roll(ctx, id2_close_vs_pm_vwap(ctx), 20, 10)


# ══════════════════════════════════════════════════════════════════════
# 5. 日内路径的极值 / 放量时点（4 个）
# ══════════════════════════════════════════════════════════════════════









# ══════════════════════════════════════════════════════════════════════
# 6. 成交结构的日频代理（3 个）
# ══════════════════════════════════════════════════════════════════════







# ══════════════════════════════════════════════════════════════════════
# 7. 20 日滚动：会话一致性与日内流动性冲击（2 个）
# ══════════════════════════════════════════════════════════════════════



