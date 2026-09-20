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


def _am(ctx) -> tuple:
    return _rng(ctx, "am_hi", "am_lo", "am_open5", "am_close5")


def _pm(ctx) -> tuple:
    return _rng(ctx, "pm_hi", "pm_lo", "pm_open5", "pm_close5")


def _am_ret(ctx) -> np.ndarray:
    """上午段收益 = am_close5(11:30) / am_open5(09:35) − 1。同日比值，无需复权。"""
    return ctx.safe_div(_f(ctx, "am_close5"), _f(ctx, "am_open5"), 1e-8) - 1.0


def _pm_ret(ctx) -> np.ndarray:
    """下午段收益 = close5(15:00) / pm_open5(13:05) − 1。同日比值，无需复权。"""
    return ctx.safe_div(_f(ctx, "close5"), _f(ctx, "pm_open5"), 1e-8) - 1.0


def _am_vwap(ctx) -> np.ndarray:
    return ctx.safe_div(_f(ctx, "am_amt"), _f(ctx, "am_vol"), 1e-6)


def _pm_vwap(ctx) -> np.ndarray:
    return ctx.safe_div(_f(ctx, "pm_amt"), _f(ctx, "pm_vol"), 1e-6)


# ══════════════════════════════════════════════════════════════════════
# 1. 极差型波动估计量（2 个）
#    —— 与已注册的 `idt_rv_daily`（收盘价已实现波动）是**不同的统计量**：
#       用的是 high/low（Parkinson）或四价（Garman-Klass），不依赖逐棒数据。
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="id2_parkinson_vol",
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


@register(FactorSpec(
    name="id2_rv_parkinson_gap",
    group="intraday",
    deps=("stock_history_5min",),
    desc="连续 vs 跳跃诊断 = ln(已实现波动 / Parkinson 极差波动)",
    formula="efficiency = safe_divide(rv_5min, parkinson_vol + 1e-10); rank(-efficiency)",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=False,
    note=(_DEG5 + " 源自参考库 Class5 增强 `microstructure_efficiency`"
          "（`rv_5min / parkinson_vol`，方向取负）。原式是比值，本因子取 ln —— "
          "比值在截面上的分布极度右偏（分母可以极小），ln 后可比性更好，"
          "且与排名的单调关系不变。"
          "**经济含义**：Parkinson 只用当日极差，看不到路径；RV 用逐棒收益平方和。"
          "两者相等 ⇒ 价格走的是**单调趋势**（每根棒都在同一方向积累）；"
          "RV 远大于 Parkinson ⇒ 路径**来回震荡**；RV 远小于 Parkinson ⇒ "
          "**一根大棒 + 其余不动**（跳空/集合竞价式重定价）。"
          "参考库的方向是负（「效率高」排后），此处沿用。"),
))
def id2_rv_parkinson_gap(ctx):
    hi, lo = _day_range(ctx)
    with np.errstate(all="ignore"):
        rv = ctx.safe_sqrt(_f(ctx, "ret2_sum"))
        pk = ctx.safe_sqrt(ctx.safe_div(np.log(ctx.safe_div(hi, lo, 1e-12)) ** 2,
                                       4.0 * np.log(2.0)))
        return ctx.safe_log(ctx.safe_div(rv, pk, 1e-12))


@register(FactorSpec(
    name="id2_rv_parkinson_gap_20",
    group="intraday",
    deps=("stock_history_5min",),
    desc="连续 vs 跳跃诊断的 20 日水平（单日太噪，20 日均值才稳）",
    formula="roll_mean(ln(rv_5min/parkinson_vol), 20, min_count=10)",
    start=ID2_START,
    warmup_days=W_20,
    higher_is_better=False,
    note=(_DEG5 + " `id2_rv_parkinson_gap` 的 20 日滚动均值。"
          "★ 为什么必须平均：单日的 `rv/parkinson` 分母可以极小"
          "（一字板日极差趋 0），单日值的截面分布有长尾；20 日均值把它压成"
          "「这只股票**一贯**是趋势型还是震荡型」的稳定描述。"
          "min_count=10 = N//2，见模块 docstring §一.5；`_roll()` 另外要求当日输入有限。"
          "与单日版**不是**同一维度的重标定（一个是当日状态、一个是风格），"
          "两者并存由 `dedup` 定量裁决。"),
))
def id2_rv_parkinson_gap_20(ctx):
    return _roll(ctx, id2_rv_parkinson_gap(ctx), 20, 10)


# ══════════════════════════════════════════════════════════════════════
# 2. 收益路径结构（2 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="id2_ret_concentration",
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
    name="id2_hi_lo_pos_gap",
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

@register(FactorSpec(
    name="id2_am_ret",
    group="intraday",
    deps=("stock_history_5min",),
    desc="上午段收益 = 11:30 价 / 09:35 价 − 1",
    formula="am = _compute_intraday_factor(context, 'am_momentum')",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=(_DEG5 + " 抄参考库 Class3 `am_momentum`。"
          "★ 与已注册的三个同类**都不重复**：`overnight_intraday_ratio_20d` 是"
          "隔夜腿 vs 全天日内腿的占比；`intraday_ret_momentum` 是"
          "**全天** close/open−1；`intraday_ma_{5,20,60}d` 是全天日内收益的"
          "多日滚动均值。本因子是**上午这半天单独**的收益，"
          "与下午段正交（A 股的上午段承载了大部分信息发布后的即时反应）。"
          "同日比值，价格未复权也无妨（同一天内复权因子是常数）。"
          "★ 09:35 是 5min 首根棒的结束时刻，不是开盘价 —— 这不是偏离，"
          "是 5min 口径的固有粒度（参考库 1min 版从 09:31 起）。"),
))
def id2_am_ret(ctx):
    return _am_ret(ctx)


@register(FactorSpec(
    name="id2_pm_ret",
    group="intraday",
    deps=("stock_history_5min",),
    desc="下午段收益 = 收盘价 / 13:05 价 − 1",
    formula="pm = _compute_intraday_factor(context, 'pm_momentum')",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=(_DEG5 + " 抄参考库 Class3 `pm_momentum`。下午段 = 13:05 开盘到 15:00 收盘，"
          "含尾盘集合竞价（A 股 14:57-15:00）。"
          "与上午段**不是**互补关系（两者之间还夹着午间跳空，见 `id2_lunch_gap`），"
          "所以 `am + pm + 午间` 才等于全天。同日比值，无需复权。"),
))
def id2_pm_ret(ctx):
    return _pm_ret(ctx)


@register(FactorSpec(
    name="id2_am_pm_ret_gap",
    group="intraday",
    deps=("stock_history_5min",),
    desc="上午段收益 − 下午段收益（会话动量的时间差）",
    formula="am_ret - pm_ret",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=(_DEG5 + " ★ **故意偏离参考库**：参考库 Class3 是 `am_pm_return_ratio`"
          "（比值），但它因分母 `pm_ret → 0` 时会爆出 ±1e3 量级的假值，"
          "已于 2026-09-17 以 `idt_am_pm_return_ratio` 之名被删（去糟粕）。"
          "本因子用**差**：有界、不会被小分母放大，而且在截面上"
          "「am−pm」与「am/pm」**不是**单调同序关系（比值对小分母敏感、"
          "差值不敏感），所以它不是一个改头换面的同义因子。"
          "经济含义：>0 ⇒ 上午强、下午回吐（隔夜信息驱动）；"
          "<0 ⇒ 下午拉升（日内资金持续流入）。"),
))
def id2_am_pm_ret_gap(ctx):
    return _am_ret(ctx) - _pm_ret(ctx)


@register(FactorSpec(
    name="id2_lunch_gap",
    group="intraday",
    deps=("stock_history_5min",),
    desc="午间跳空 = 13:05 价 / 11:30 价 − 1",
    formula="lb = _compute_intraday_factor(context, 'lunch_break_ret')",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=(_DEG5 + " 抄参考库 Class3 `lunch_break_ret`。"
          "★ 这是**日线层完全看不见**的量：日线只有 昨收/开/高/低/收，"
          "而 11:30→13:05 这条断口把「上午收盘」和「下午开盘」连接起来，"
          "期间是午间公告、午间舆情、以及 12:00 前后发布的宏观数据的反应窗口。"
          "同日比值、未复权，量级通常极小（中位数 < 0.2%），"
          "但截面上的**符号与大小**承载了午间信息的即时定价。"),
))
def id2_lunch_gap(ctx):
    return ctx.safe_div(_f(ctx, "pm_open5"), _f(ctx, "am_close5"), 1e-8) - 1.0


@register(FactorSpec(
    name="id2_am_vol_share",
    group="intraday",
    deps=("stock_history_5min",),
    desc="上午段成交量占全天的比重",
    formula="avs = _compute_intraday_factor(context, 'am_vol_share'); rank(avs)",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=(_DEG5 + " 抄参考库 Class3 `am_vol_share`。"
          "★ 单位自动约掉：日内层已逐 (股票,日) 判过 `vol` 是「手」还是「股」"
          "并 ×100 归一（`fea/intraday.py` 的 `vol_mult`），`am_vol` 与 `vol` "
          "用的是**同一个** `mult`，所以比值在任何年份都成立 —— "
          "跨 2025-11/12 的单位翻转边界**没有** 100 倍假跳变。"
          "取值域 [0,1]。★ 不做 `pm_vol/vol`：那是本因子的**精确补集**"
          "（1 − x），截面排名会完全反向，纯属重复。"
          "高 ⇒ 交易集中在上午（信息驱动）；低 ⇒ 尾盘/下午放量（资金驱动）。"),
))
def id2_am_vol_share(ctx):
    return ctx.safe_div(_f(ctx, "am_vol"), _f(ctx, "vol"), 0.0)


@register(FactorSpec(
    name="id2_am_pm_range_ratio",
    group="intraday",
    deps=("stock_history_5min",),
    desc="上午振幅 / 下午振幅（取 ln，会话波动的时间分配）",
    formula="return cross_sectional_rank(-_metric(context, 'am_pm_hl_range_ratio'))",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=False,
    note=(_DEG5 + " 源自参考库 Class3 `am_pm_hl_range_ratio`（方向负）。"
          "★ 与已被删的 `idt_am_pm_rv_ratio` 是**同一个构念、不同的估计量**："
          "那个用两段各自的 `ret2_sum`（真实已实现波动，需要逐 session 的二阶量，"
          "本项目层里**没有**分段的 ret2_sum），本因子退一步用 Parkinson 式"
          "「极差」做段内波动代理。参考库只此一个版本，"
          "而「上下午波动比」是 A 股日内择时文献里的标准量，故保留。"
          "★ 必须区间扩张，否则极差可能 ≤0 使 ln 出 NaN。"),
))
def id2_am_pm_range_ratio(ctx):
    ah, al = _am(ctx)
    ph, pl = _pm(ctx)
    with np.errstate(all="ignore"):
        ra = np.log(ctx.safe_div(ah, al, 1e-12))
        rp = np.log(ctx.safe_div(ph, pl, 1e-12))
    return ctx.safe_div(ra, rp, 1e-12)


@register(FactorSpec(
    name="id2_session_range_overlap",
    group="intraday",
    deps=("stock_history_5min",),
    desc="跨会话区间结构 = (上午振幅 + 下午振幅) / 全天振幅 ∈ [1,2]",
    formula="(am_range + pm_range) / day_range",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=False,
    note=(_DEG5 + " **本文件新造**（参考库没有跨会话区间结构的任何因子）。"
          "数学性质：上午与下午是两个不相交的时间段，"
          "`max(am_hi,pm_hi) − min(am_lo,pm_lo) ≥ max(段内振幅)`，"
          "而全天振幅由区间扩张后的 hi/lo 给出，故比值 ∈ [1,2]。"
          "= 1 ⇒ **下午从未走出上午的区间**（盘整、方向未变）；"
          "= 2 ⇒ 两个会话探索了**互不相交**的价格区间（重定价、方向翻转）。"
          "与 `id2_am_pm_range_ratio` 的关键区别：那个是两段**振幅之比**"
          "（只看大小），本因子是**区间是否错开**（看位置），"
          "两个会话可以振幅相同但完全错开（比值=1，本因子=2）。"
          "分母全天振幅趋 0（一字板）时由 `safe_div` 给 NaN —— 正确语义：无信息。"),
))
def id2_session_range_overlap(ctx):
    ah, al = _am(ctx)
    ph, pl = _pm(ctx)
    dh, dl = _day_range(ctx)
    ra = ah - al
    rp = ph - pl
    rd = dh - dl
    return ctx.safe_div(ra + rp, rd, 1e-12)


# ══════════════════════════════════════════════════════════════════════
# 4. 会话 VWAP 与收盘定价（3 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="id2_am_pm_vwap_gap",
    group="intraday",
    deps=("stock_history_5min",),
    desc="上午 VWAP / 下午 VWAP − 1（会话成交均价的时间位移）",
    formula="return cross_sectional_rank(_metric(context, 'vwap_am_pm_gap'))",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=(_DEG5 + " 抄参考库 Class3 `vwap_am_pm_gap`。"
          "★ 会话 VWAP 是层里**新出现**的对象（`am_amt/am_vol`、`pm_amt/pm_vol`），"
          "已注册的 `vwap_daily_deviation` 只覆盖**全天** VWAP 与收盘的偏离，"
          "不区分会话。同日比值、同单位约掉，未复权无妨。"
          "与 `id2_am_pm_ret_gap` 的区别：那个比较两个会话的**端点价差**"
          "（首尾两点），本因子比较两个会话的**成交均价**（整段重心）。"
          "均价位移 > 端点位移 ⇒ 该段成交集中在高位（放量拉抬）。"),
))
def id2_am_pm_vwap_gap(ctx):
    return ctx.safe_div(_am_vwap(ctx), _pm_vwap(ctx), 1e-8) - 1.0


@register(FactorSpec(
    name="id2_close_vs_pm_vwap",
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
    name="id2_close_vs_pm_vwap_20",
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

@register(FactorSpec(
    name="id2_intraday_max_runup",
    group="intraday",
    deps=("stock_history_5min",),
    desc="日内最大反弹（从段内低点到其后高点的最大涨幅）",
    formula="ru = _compute_intraday_factor(context, 'intraday_max_runup'); rank(ru)",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=(_DEG5 + " 抄参考库 Class3 `intraday_max_runup`。"
          "★ **层里早就算好了 `max_ru`，但此前 0 个因子用它** —— "
          "已注册的 `idt_intraday_max_drawdown` 只做了回撤那一侧"
          "（层里 `_path_stats` 同时产出 `max_dd` 与 `max_ru`，"
          "参考库也是两个都发，本项目只建了回撤侧）。"
          "注意口径：这是**日内路径**上的最大反弹，不是「从昨日低点反弹」，"
          "与 `price_distance_from_52w_low` / `rebound_from_low_20` 无关。"
          "≥ 0。"),
))
def id2_intraday_max_runup(ctx):
    return _f(ctx, "max_ru")


@register(FactorSpec(
    name="id2_dd_ru_asym",
    group="intraday",
    deps=("stock_history_5min",),
    desc="日内行程的不对称 = |最大回撤| / (|最大回撤| + 最大反弹) ∈ [0,1]",
    formula="abs(max_dd) / (abs(max_dd) + max_ru)",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=False,
    note=(_DEG5 + " **本文件新造**。层里 `max_dd ≤ 0`（= min(seg/run_max − 1)，"
          "回撤是负的）而 `max_ru ≥ 0`（反弹是正的），所以取绝对值后相加"
          "是「这天所有极端行程的总量」，比值就是**向下的占比**。"
          "= 0.5 ⇒ 上下对称；> 0.5 ⇒ 向下行程占主导（日内抛压）。"
          "与已注册的 `gain_loss_asymmetry_60` 的区别：那个是 60 日**逐日收益**"
          "的正负半方差比（跨日、低频），本因子是**单日之内**的路径极值比"
          "（同日、路径形状）—— 两者衡量的时间尺度完全不同。"
          "分母趋 0（全天几乎不动）时由 `safe_div` 给 NaN，语义正确。"),
))
def id2_dd_ru_asym(ctx):
    dd = np.abs(_f(ctx, "max_dd"))
    ru = _f(ctx, "max_ru")
    return ctx.safe_div(dd, dd + ru, 1e-12)


@register(FactorSpec(
    name="id2_vol_peak_pos",
    group="intraday",
    deps=("stock_history_5min",),
    desc="当日成交量的峰值时点（层已归一化到 [0,1]）",
    formula="vpt = _compute_intraday_factor(context, 'volume_peak_time'); rank(-vpt)",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=False,
    note=(_DEG5 + " 抄参考库 Class3 `volume_peak_time`（方向负：放量越晚排越后）。"
          "★ 层里 `vol_peak_pos = argmax(vol) / (n_bars − 1)` **已经是 [0,1] "
          "的归一化时点**，不要再除 `n_bars`。"
          "与已注册的 `idt_vol_stability` 不重复：那个是 `Σv²/(Σv)²`"
          "（成交量的**离散度**），本因子是**峰值在哪**（位置的**一阶**统计量）——"
          "同样的离散度可以对应早盘放量或尾盘放量。"
          "值小 ⇒ 开盘/早盘放量（隔夜信息消化）；值大 ⇒ 尾盘放量"
          "（资金驱动、可能的收盘操纵）。"),
))
def id2_vol_peak_pos(ctx):
    return _f(ctx, "vol_peak_pos")


@register(FactorSpec(
    name="id2_vol_peak_std_20",
    group="intraday",
    deps=("stock_history_5min",),
    desc="放量时点的 20 日标准差（日内流动性节奏稳不稳）",
    formula="roll_std(vol_peak_pos, 20, min_count=10)",
    start=ID2_START,
    warmup_days=W_20,
    higher_is_better=False,
    note=(_DEG5 + " **本文件新造**。`id2_vol_peak_pos` 看「今天几点放量」，"
          "本因子看「**每天几点放量这件事稳不稳**」。"
          "稳定的日内流动性节奏（标准差小）= 有固定的做市/被动资金，"
          "峰位乱跳 = 交易由事件驱动、不可预期。"
          "与 `vol_of_vol_20` / `vol_clustering_20` 的区别：那两个算的是"
          "**收益率波动**的波动，本因子算的是**成交量时点**的波动 —— "
          "一个是价格维、一个是时间维。"),
))
def id2_vol_peak_std_20(ctx):
    return ctx.roll_std(_f(ctx, "vol_peak_pos"), 20, 10)


# ══════════════════════════════════════════════════════════════════════
# 6. 成交结构的日频代理（3 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="id2_zero_bar_share",
    group="intraday",
    deps=("stock_history_5min",),
    desc="零收益 5min 棒占比 = (n_zero − 1) / (n_bars − 1)",
    formula="(n_zero - 1) / (n_bars - 1)",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=False,
    note=(_DEG5 + " **本文件新造**（已注册的 `zero_return_fraction_20` 是"
          "**日频**的 |pct_chg| < 0.1% 占比，20 日窗口；本因子是**单日之内**"
          "在 5min 粒度上的零收益棒占比，灵敏度高一个数量级）。"
          "★★ `−1` 是**强制**的：层里每个 (股票,日) 段的**第一根棒** "
          "`prev_c = NaN → r = NaN → r0 = 0`，于是 `n_zero` 恒含一个假零。"
          "不减去它，全市场每只股票每天都凭空多一根零收益棒"
          "（48 根里多 1 根 = 2.1% 的系统性高估），而且**不报错**。"
          "分母同时减 1（第一根本来就没有收益可比）。"
          "经济含义：一字板 / 极度不活跃 / 长时间无成交。"),
))
def id2_zero_bar_share(ctx):
    n = _f(ctx, "n_bars")
    z = _f(ctx, "n_zero")
    return ctx.safe_div(z - 1.0, np.maximum(n - 1.0, 0.0), 1e-12)


@register(FactorSpec(
    name="id2_vol_amt_hhi_gap",
    group="intraday",
    deps=("stock_history_5min",),
    desc="成交**额**集中度 − 成交**量**集中度（大单是否集中在高价区）",
    formula="n*amt2_sum/amt**2 - n*vol2_sum/vol**2",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=(_DEG5 + " **本文件新造**。`n·Σx²/(Σx)²` 是逆参与比形式的集中度"
          "（与 `id2_ret_concentration` 同一族，但用在成交量/额上）∈ [1, n]。"
          "两项之差 > 0 ⇒ 成交**额**比成交**量**更集中 ⇒ "
          "放量的那几根棒价格**高于**当日均价（大单成交在高位）；"
          "< 0 ⇒ 放量集中在低价区（承接/吸筹）。"
          "★ 用到的 `amt2_sum` **此前 0 个因子消费过**。"
          "★ 单位：`vol2_sum` 层里已按 `mult²` 修正（`fea/intraday.py` 的注释），"
          "所以本比值跨 2025-11/12 单位翻转边界**量纲一致**，无需在因子内补偿。"
          "⚠ 与 `idt_vol_stability` 的关系：那个是**单个** HHI 的水平，"
          "本因子是**两个不同物理量**的 HHI 之差，不是它的重标定。"),
))
def id2_vol_amt_hhi_gap(ctx):
    n = _f(ctx, "n_bars")
    amt = _f(ctx, "amt")
    vol = _f(ctx, "vol")
    with np.errstate(all="ignore"):
        h_amt = ctx.safe_div(n * _f(ctx, "amt2_sum"), amt ** 2, 1e-12)
        h_vol = ctx.safe_div(n * _f(ctx, "vol2_sum"), vol ** 2, 1e-12)
    return h_amt - h_vol


@register(FactorSpec(
    name="id2_am_close_position",
    group="intraday",
    deps=("stock_history_5min",),
    desc="上午收盘价在上午区间中的位置 ∈ [0,1]",
    formula="return cross_sectional_rank(_metric(context, 'am_hl_position'))",
    start=ID2_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=(_DEG5 + " 抄参考库 Class3 `am_hl_position`（上午收盘在上午高低区间的位置）。"
          "★ 与已被删的 `idt_close_position` **构念相同、对象不同**："
          "那个是**全天**收盘在全天区间的位置，本因子是**上午 11:30** 在"
          "**上午区间**的位置。保留它的理由是本文件 docstring §一.2 ——"
          "`am_close5`（11:30）是日内层**唯一新增的两个价格点之一**，"
          "日线层与已注册因子都拿不到它。"
          "若 `dedup` 实测与全天版 |ρ| ≥ 0.95 则砍掉。"),
))
def id2_am_close_position(ctx):
    ah, al = _am(ctx)
    return ctx.safe_div(_f(ctx, "am_close5") - al, ah - al, 1e-12)


# ══════════════════════════════════════════════════════════════════════
# 7. 20 日滚动：会话一致性与日内流动性冲击（2 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="id2_session_sign_agreement_20",
    group="intraday",
    deps=("stock_history_5min",),
    desc="上午与下午同向的频率（20 日）∈ [0,1]",
    formula="roll_mean(sign(am_ret) == sign(pm_ret), 20, min_count=10)",
    start=ID2_START,
    warmup_days=W_20,
    higher_is_better=True,
    note=(_DEG5 + " **本文件新造**。上午与下午的**符号**是否一致 —— "
          "高 ⇒ 日内方向持续（趋势型股票），低 ⇒ 日内反转频繁（震荡型）。"
          "★ 用 `==` 比较**有限值掩码后**的符号，不用 `np.sign(a) != np.sign(b)`："
          "后者在任一侧为 NaN 时 `np.sign(NaN) = NaN`，`NaN != NaN` 为 **True**，"
          "会把缺失日静默计成「方向不一致」（这类 bug 已在 "
          "`mf_flow_stability_20d` 的 note 里记录过一次）。"
          "★★ **截面取值卡片化警告**：20 日均值只有 21 个可能取值"
          "（0, 1/20, …, 1），且大多数股票会挤在两端。"
          "`fea/eval.py` 的退化判据是「单值占比 > 75%」，本因子可能触发 ——"
          "沙箱自检阶段必看每日截面唯一值数，不足就地砍。"),
))
def id2_session_sign_agreement_20(ctx):
    am = _am_ret(ctx)
    pm = _pm_ret(ctx)
    both = np.isfinite(am) & np.isfinite(pm)
    agree = np.where(both, (np.sign(am) == np.sign(pm)).astype(np.float64), np.nan)
    return ctx.roll_mean(agree, 20, 10)


@register(FactorSpec(
    name="id2_amihud_intraday_20",
    group="intraday",
    deps=("stock_history_5min",),
    desc="日内 Amihud 非流动性（20 日）：|日内收益| / 成交额",
    formula="amihud_intraday = abs(ret_sum) / amt; 20 日均值",
    start=ID2_START,
    warmup_days=W_20,
    higher_is_better=False,
    note=(_DEG5 + " 源自参考库 Class3 `amihud_5min`。"
          "★ **与已注册的两个 Amihud 因子的唯一区别是「哪条腿」**："
          "`amihud_daily_5` 用 close-to-close（含隔夜跳空），"
          "`amihud_asymmetry_20` 是涨跌日的不对称，本因子用**日内腿**"
          "（`ret_sum` = 48 根棒收益之和，剔除了隔夜跳空）。"
          "日内腿的 Amihud 衡量的是「**盘中**每元成交额推动的价格变化」。"
          "⚠ **本文件置信度最低的一行**：`amihud_daily_20` 曾在 2026-09-17 "
          "的去糟粕清单里（虽未注册），与已注册的两个 Amihud 家族的相关性"
          "未实测。**`dedup` 阶段与 `amihud_daily_5` / `amihud_asymmetry_20` "
          "逐一比 |ρ|，超过 0.95 就地删除。**"
          "分子是单位无关的收益、分母是元 —— 同日截面内一致，可跨日比。"),
))
def id2_amihud_intraday_20(ctx):
    x = ctx.safe_div(np.abs(_f(ctx, "ret_sum")), _f(ctx, "amt"), 1e-6)
    return _roll(ctx, x, 20, 10)
