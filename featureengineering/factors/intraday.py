"""日内 / 微结构因子（18 个）—— 全部日频产出、只主板、跟随 default_start。

数据源：`data/derived/intraday/`（`fea/intraday.py` 把 `stock_history_5min`
**6.86 亿行**预聚合成约 35 个日频字段）+ 价格层（`fea/prices.py`）。

## 写这一族必须记住的九件事（全部实测，不是推理）

1. **日内层的价格一律未复权，绝不跨日相除**。`open5/close5/hi5/lo5/am_*/pm_*`
   都是**当日**未复权价；不同交易日的复权因子不同（分红送转），
   `open5(T)/close5(T-1)-1` 会在除权日造出 −30% 量级的**假跳空**（而且不报错）。
   → 隔夜跳空一律 `ctx.hfq("open") / ctx.shift(ctx.hfq("close"), 1) - 1`。
   这与 `stock_daily.pre_close` 是同一个口径（交易所的前收 = 昨收 × 除权调整），
   所以与参考库的 `open/pre_close-1` 等价，但走的是本项目唯一允许的
   「未复权价 × 累计复权因子」路径（PIT 安全，见 `fea/spec.py` 的红线）。

2. **停牌日 `ctx.hfq(...)` 是前向填充的**（价格层对状态量 ffill，对流量留 NaN）。
   于是停牌日 `hfq_open(T)/hfq_close(T-1) - 1` **不是 0**，而是
   「上一个成交日**自己**的日内涨跌」——一个凭空的假跳空。
   → 三个跨日因子一律用 `ctx.traded()` 掩码。实测 2015 年每天只有
   1361~2584 只股票有行情，不掩码等于给停牌股编收益。
   （日内层的因子不需要这一层掩码：停牌日在日内层**根本没有行**。）

3. **日内层缺失 = NaN，不是 0**。停牌 / 未上市 / 已退市当天在日内层没有行，
   `intraday_field` 用**精确落格**（`panel.place`）而非 `asof` 前向填充，
   所以空洞直接暴露成 NaN。本文件**从不**用 `nan_to_num` 之类把 NaN 变 0：
   那等于把停牌记成「零成交 / 零波动」，会系统性低估波动。

4. **`vol` 的单位分段翻转**（`fea/intraday.py` 的模块 docstring）：
   2010-01-04~2025-11-28 是**手**，2025-12-01 起是**股**；日内层已逐
   (股票, 日) 用 `amt/vol ≈ close` 判过并 ×100 归一。本文件用的 `vol`
   是**归一后**的，跨 2025-11/12 边界没有 100 倍假跳变。
   ★ **但归一化漏了一次乘法**：`out["vol2_sum"] *= mult` 只乘了一次 `mult`，
   而 `vol2_sum = Σ v_raw²` 是**平方量**，正确值要乘 `mult²`。
   （`vol` 自己只乘一次是对的，所以 `vol2_sum/vol²` 在 2010~2025-11 与
   2025-12 之后**量纲不一致**。）本文件在 `idt_vol_stability` 里用
   `vol2_sum * vol_mult / vol²` 显式补回——比值 Σv²/(Σv)² 与原单位无关，
   补回后严格满足柯西下界 1/n_bars（实测 2025 年违反率从 90.5% → 0.000%）。
   **这是框架侧的一处 bug；本文件没有、也不允许修改 `fea/**`。**

5. **`hi_pos / lo_pos / vol_peak_pos` 已经是 [0,1] 的归一化时点**
   （= 段内 argmax / (n_bars − 1)），**不要再除 n_bars**。本文件直接透传。

6. **`rv_daily = sqrt(ret2_sum)` 是「日」已实现波动率，不是年化**。
   量纲就是当日收益的标准差（实测中位数 ≈ 0.02，即 2%）。要年化需 ×√244，
   本文件**不**年化（参考库也不年化），下游需要自己乘。
   同理 `rv_semi_up/down` 是日频半波动；`rv_term_structure_slope` 是
   两个日频 RV 之比，口径自动约掉，可以跨 5min/1min 比较**相对**水平。

7. **`min_count = N//2` 的理由**：日内层的缺失**不是随机的**——停牌
   （2015 年 9.89% 的收益格是停牌日）、上市/退市窗口、以及第 8 条的
   数据冻结。要求「窗口满 N 且全有效」（`min_count=None`）会让
   **绝大多数股票**在 20 日窗里撞上 1~2 天停牌就整窗 NaN；
   完全不加 min_count 又会让「窗口里只有 1 天有数据」也产出值。
   取 `N//2`：这正是参考库自己的 `min_periods`（`overnight_return_share_20`
   写的就是 `_roll_sum(x, 20, 10)`），两边口径一致。
   ★ 另外每个滚动因子**额外要求当日输入有限**：否则窗口只在最近几天缺值时，
   会一直吐出**陈旧**的值，把长期停牌和数据冻结悄悄藏起来。

8. **★ 数据边界：`stock_history_5min` 冻结在 2026-09-11。**
   2026-09-14 只有 48 只股票（2026 分区第一个 row group 的尾巴），
   2026-09-15 起完全没有。所以**吃日内层的 15 个因子在 2026-09-11 之后是 NaN**。
   这是**预期的、不是 bug**，本文件不做任何前向填充去掩盖它。
   三个跨日因子（`idt_overnight_gap` / `idt_overnight_minus_intraday` /
   `idt_overnight_return_share_20`）只吃价格层，**不受这个冻结影响**
   （各自的 note 里都注明了）。本次实测边界（2026-09）：
   15 个吃日内层的因子在 2026-09-11 还是整截面（3,180~3,190 只），
   2026-09-14 只剩 **48 只**（= 2026 分区第一个 row group 的尾巴），
   2026-09-15 起为 0；3 个价格层因子 2026-09-14 仍是整截面（3,187 只）。

9. **★ 上游 5min 的 `high/low` 不总是包住自己那根棒的 open/close，区间类因子
   必须先修区间**。实测 2012-2016 共 246.2 万 (股票, 日) 行：
   `close5 ∉ [lo5,hi5]` 1,836 行（0.075%）、`open5 ∉ [lo5,hi5]` 1,970 行、
   `hi5 < lo5` 117 行、am/pm 段 3,846/5,210 行。典型例子 `600317.SH 2012-01-09`：
   `high` 从 14:30 起一直停在 4.00，而收盘是 4.05（当天 9 根棒 close > high）。
   后果：`(close−lo)/(hi−lo)` 跑出 [0,1]（**实测最差 −3 / +5**，
   自测的合法性锚点就是这样抓到的），`ln(hi/lo)` 变负、把半日振幅比翻符号。
   → 本文件统一走 `_seg_range()`：`hi = max(hi5, open5, close5)`、
   `lo = min(lo5, open5, close5)`（半日段同理），保证 `o,c ∈ [lo,hi]` 后再算。
   只影响约 0.1% 的格子，其余格子**逐位不变**；这是分母构造的一致性修复，
   不是对结果做 winsor。

## 与参考库的偏离（`学习资料/factors.md` Class 1 / Class 3）

参考库用 `history_1min`（240 根/日），本项目只有 5 分钟（48 根/日）：

* **收益平方和类**（rv_daily / rv_semi_* / rv_term_structure_slope）：
  5min 收益比 1min 更接近正态、少掉一部分微结构噪声（买卖价差跳动），
  RV **系统性偏小一档**；但它是**每只股票都偏小**的截面近似常数，
  截面排序不变。绝对水平不要与参考库对齐。
* **时点类**（hi_pos / lo_pos）：分辨率只有 48 档（不是 240 档），
  同一个时点会被量化到 5min 网格上，**会有大量并列值**。
* **分钟占比类**：5min 口径下的「正收益区间占比」在参考库里叫
  `intra_trend`（定义逐字为「正收益 5 分钟区间占比」），与
  `up_minutes_ratio`（1 分钟正收益分钟数/总分钟数）同义不同分辨率。
* **不可计算的 4 类 6 个名字**（预聚合表里没有原料，需引擎扩字段，见文件末尾附录）：
  `open_30_momentum` / `close_30_momentum` / `last30_ret`（要首/尾 30 分钟切片）、
  `volume_profile_skew`（要分钟内量三阶矩）、`intraday_reversal_intensity`
  （要分钟收益的一阶自协方差）、`vol_concentration`（参考库定义是
  「(开盘 30 分 + 收盘 30 分) 成交量 / 全日成交量」）。
  其中最后一维用**同族、同为 5 分钟口径**的 `vol_stability`
  （5 分钟成交量 std/mean）替代：二者互为单调变换
  （CV = √(n·HHI − 1)），是同一维度的等价刻画，详见 `idt_vol_stability`。
* **代理式偏离 1 处**：`am_pm_rv_ratio` 用 Parkinson 对数振幅
  `ln(hi5/lo5)` 代理半日 RV（预聚合表没有分时段的 `ret2_sum`），
  同量纲、同方向，详见该因子 note。
* **公式级偏离 1 处**：`idt_price_impact_intraday` 的分子比参考库多除了一个
  开盘价（价格水平），理由见该因子 note。

## 命名

`idt_` 前缀（本文件独占），避让 `turnover_*`/`amihud_*`/`volume_*`（liquidity.py）、
`vol_*`/`drawdown_*`（volatility.py）、`momentum_*`/`rs_*`（momentum.py）、
`rsi_*`/`macd_*`/`bollinger_*`（technical.py）。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

# 跟随 conf/config.yaml 的 default_start（当前 2012-01-01）。
# 日内层覆盖 2010 起，不受上游起点限制。
IDT_START = None

# warmup_days = 「计算窗口要往前多读多少**日历天**」，滚动窗口 N 按 N×1.8+20 给。
W_D = 40      # 纯当日量（没有滚动窗口）——给一个月冗余，跨年/停牌都够
W_20 = 56     # 20 交易日窗：20×1.8+20（≈37 个交易日 > 20）
W_60 = 128    # 60 交易日窗：60×1.8+20（≈85 个交易日 > 60）

# 每个吃日内层的因子的 note 都要带这句（见模块 docstring 的「偏离」一节）
_DEG5 = ("★ 5min 口径（参考库 history_1min 240 根/日 → 本项目 stock_history_5min "
         "48 根/日）：")


# ══════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════

def _f(ctx, field: str) -> np.ndarray:
    """日内层字段（当日口径、未复权）。

    停牌 / 未上市 / 未落格 -> NaN（精确落格，不是前向填充）。**不填 0**。
    """
    return np.asarray(ctx.intraday_field(field), dtype=np.float64)


def _hfq(ctx, field: str) -> np.ndarray:
    """后复权价（本项目唯一允许的价格口径）。停牌日是前向填充的旧值。"""
    return np.asarray(ctx.hfq(field), dtype=np.float64)


def _overnight_gap(ctx) -> np.ndarray:
    """隔夜跳空 = hfq_open(T) / hfq_close(T−1) − 1，**停牌日 NaN**。

    两个必须（模块 docstring 第 1、2 条）：
      (a) 用后复权价——未复权价跨日相除在除权日会造出假跳空；
      (b) 用 `ctx.traded()` 掩码——`hfq_close` 是前向填充的，停牌日算出来的
          是「上一个成交日自己的日内涨跌」，不是跳空。
    ★ 不用 `ctx.px("pre_close")`：那列的除权调整依赖供应商口径，
      而后复权比值 `hfq_open(T)/hfq_close(T−1)` 与交易所前收**数学等价**
      （前收 ≡ 昨收 × 除权调整），且完全走本项目已验证的复权路径。
    """
    op = _hfq(ctx, "open")
    pc = ctx.shift(_hfq(ctx, "close"), 1)
    g = ctx.safe_div(op, pc, min_abs_den=1e-8) - 1.0
    return np.where(ctx.traded(), g, np.nan)


def _intraday_ret(ctx) -> np.ndarray:
    """当日日内收益 = hfq_close / hfq_open − 1，停牌日 NaN。

    只吃**价格层**，所以不受 `stock_history_5min` 冻结（2026-09-11）影响。
    为什么不用日内层的 `close5/open5`：隔夜腿与日内腿必须**同源同日**，
    两条腿都来自 `stock_daily` + `adj_factor` 时 NaN 掩码完全一致，
    占比类因子不会出现「隔夜有效、日内缺失」的半残格；
    且两腿的复权口径统一（日内层的比值本身也是同日的，二者数值只差
    首个 5 分钟棒是否计入，实测差异 ~1e-4 量级）。
    """
    op = _hfq(ctx, "open")
    cl = _hfq(ctx, "close")
    r = ctx.safe_div(cl, op, min_abs_den=1e-8) - 1.0
    return np.where(ctx.traded(), r, np.nan)


def _seg_range(ctx, hi_f: str, lo_f: str, op_f: str, cl_f: str):
    """一段（全日或半日）的 [最低, 最高]，并用**本段自己的开/收盘扩张**。

    ★★ 为什么必须扩张：上游 `stock_history_5min` 的 `high/low` **不总是包住
      自己那根棒的 close/open**。实测 2012-2016 共 246.2 万 (股票, 日) 行：

          close5 ∉ [lo5, hi5]  1,836 行 (0.075%)
          open5  ∉ [lo5, hi5]  1,970 行 (0.080%)
          hi5 < lo5              117 行
          am 段 / pm 段越界     3,846 / 5,210 行

      典型案例 `600317.SH 2012-01-09`：`high` 从 14:30 起一直停在 4.00，
      而 14:35 之后的价格一路到 4.05（当天 9 根棒的 close > high），
      于是 `close5=4.05 > hi5=4.00`。
      不修的话：(close−lo)/(hi−lo) 会跑出 [0,1]（实测最差 −3 / +5），
      `ln(hi/lo)` 会变负、把「上/下午振幅比」翻成负值 —— 都是**静默**的错误值。

      扩张后保证 `hi ≥ max(o,c) ≥ min(o,c) ≥ lo`，三个区间类因子的取值域
      恢复有界（close_position / path_efficiency ∈[0,1]，am_pm_rv_ratio ≥ 0）。
      修正只触及约 0.1% 的格子，其余格子与原口径**逐位相同**
      （`np.maximum(hi5, max(o,c))` 在 hi5 本来就更大时不改变任何值）。
      这是**数据一致性修复**，不是 winsor —— 只动分母的构造，不动结果分布。

    NaN 语义：hi5/lo5 缺失（整段没有有效高低价）时保持 NaN；
    o/c 有一侧缺失时用另一侧（`np.fmax/np.fmin` 忽略 NaN），不会因为
    某一根棒的开盘价缺失而丢掉整格。
    """
    o = _f(ctx, op_f)
    c = _f(ctx, cl_f)
    hi = np.maximum(_f(ctx, hi_f), np.fmax(o, c))
    lo = np.minimum(_f(ctx, lo_f), np.fmin(o, c))
    return hi, lo


# ══════════════════════════════════════════════════════════════════════
# 1. 隔夜 / 跨日（3 个）—— 只吃价格层，不受 5min 冻结影响
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="idt_overnight_gap",
    group="intraday",
    desc="隔夜跳空：后复权开盘 / 上一交易日后复权收盘 − 1",
    formula="gap = open / pre_close - 1  （参考库 rank(-gap)）",
    deps=("stock_daily", "stock_adj_factor"),
    start=IDT_START,
    warmup_days=W_D,
    higher_is_better=False,
    note=("参考库 Class1 `overnight_gap`（alternative.py）逐字为 "
          "`-(open-pre_close)/pre_close` 截面排名 —— 高开=反转信号排后，故本因子"
          "（有符号的跳空）方向取负。"
          "★ 口径：`pre_close` 换成 `hfq_open(T)/hfq_close(T-1)-1`，两者数学等价"
          "（交易所前收 = 昨收 × 除权调整），但走的是 PIT 安全的「未复权价 × "
          "累计复权因子」路径，不依赖供应商的 pre_close 列。"
          "★ **不用日内层的 open5/close5**：那是未复权价，跨日相除在除权日会造出"
          " −30% 量级的假跳空且不报错（模块 docstring 第 1 条）。"
          "★ 停牌日（`ctx.traded()` 为 False）置 NaN：价格层对 open/close 做了"
          "前向填充，不掩码的话停牌日会产出「上一个成交日自己的日内涨跌」。"
          "★ 本因子**不依赖** `stock_history_5min`，所以 2026-09-11 之后仍有值"
          "（数据冻结只影响吃日内层的因子）。"),
))
def idt_overnight_gap(ctx):
    return _overnight_gap(ctx)


@register(FactorSpec(
    name="idt_overnight_minus_intraday",
    group="intraday",
    desc="隔夜 − 日内收益差：隔夜强于日内＝信息在开盘被消化",
    formula="overnight - intraday  (= open/pre_close-1 与 close/open-1 之差)",
    deps=("stock_daily", "stock_adj_factor"),
    start=IDT_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=("参考库 Class1 `overnight_minus_intraday`（fac_cand_daily.py）逐字为 "
          "`df['overnight'] - df['intraday']`，其中 overnight=open/pre_close-1、"
          "intraday=close/open-1；文档记录 V8 筛查 **meanIC 0.030 / ICIR 0.20**，"
          "是 27 个日频候选里最强的，方向为正。"
          "★ 两条腿都走 `ctx.hfq`（后复权）且都用 `ctx.traded()` 掩码，"
          "保证「隔夜强于日内」不会被停牌日的假收益污染。"
          "★ 本因子**不依赖** `stock_history_5min`，2026-09-11 之后仍有值。"),
))
def idt_overnight_minus_intraday(ctx):
    return _overnight_gap(ctx) - _intraday_ret(ctx)


@register(FactorSpec(
    name="idt_overnight_return_share_20",
    group="intraday",
    desc="隔夜收益占比：20 日 |跳空| / Σ(|跳空|+|日内|)",
    formula="roll_sum(|gap|, 20, 10) / roll_sum(|gap|+|intra|, 20, 10)",
    deps=("stock_daily", "stock_adj_factor"),
    start=IDT_START,
    warmup_days=W_20,
    higher_is_better=True,
    note=("参考库 Class1 `overnight_return_share_20`（momentum_structure.py）逐字为 "
          "`num=_roll_sum(abs_gap,20,10); den=_roll_sum(abs_gap+abs_intra,20,10); "
          "share=safe_divide(num,den)` —— 参考库自己就用 **min_count=10 = 20//2**，"
          "本文件与它口径一致（模块 docstring 第 7 条）。"
          "★ 分子分母同量纲（都是 |收益| 之和），占比 ∈[0,1]。"
          "★ 停牌日两腿都 NaN，被 min_count 跳过（不是当 0 计入，"
          "否则停牌多的股票占比会被稀释）。"
          "★ 本因子**不依赖** `stock_history_5min`，2026-09-11 之后仍有值"
          "（分母的分母是价格层）。"),
))
def idt_overnight_return_share_20(ctx):
    g = np.abs(_overnight_gap(ctx))
    it = np.abs(_intraday_ret(ctx))
    num = ctx.roll_sum(g, 20, min_count=10)
    den = ctx.roll_sum(g + it, 20, min_count=10)
    return ctx.safe_div(num, den, min_abs_den=1e-8)


# ══════════════════════════════════════════════════════════════════════
# 2. 上/下午不对称（2 个）
# ══════════════════════════════════════════════════════════════════════





# ══════════════════════════════════════════════════════════════════════
# 3. 已实现波动率（4 个）—— 日频口径，非年化（模块 docstring 第 6 条）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="idt_rv_daily",
    group="intraday",
    desc="日已实现波动率：当日 5min 收益平方和开根（**日频口径，未年化**）",
    formula="rv_daily = sqrt(sum(r_5min^2)) = sqrt(ret2_sum)",
    deps=("stock_history_5min",),
    start=IDT_START,
    warmup_days=W_D,
    higher_is_better=False,
    note=(_DEG5 + "参考库 `rv_daily` 用 240 个 1min 收益，本因子用 48 个 5min 收益，"
          "**RV 系统性偏小一档**（5min 棒吃掉了棒内的价格路径，也少掉了买卖价差"
          "跳动带来的正偏差）；这是对**所有股票**方向的同向压缩，截面排序不受影响，"
          "但绝对水平不要与参考库对齐。"
          "★ **这是「日」波动率不是年化**：中位数 ≈ 0.02（2%）。要年化自己 ×√244。"
          "参考库方向为负（低波排前）。"
          "★ 停牌日 NaN（日内层无行），不是 0 —— 填 0 会造出「零波动日」。"),
))
def idt_rv_daily(ctx):
    return ctx.safe_sqrt(_f(ctx, "ret2_sum"))






@register(FactorSpec(
    name="idt_rv_term_structure_slope",
    group="intraday",
    desc="波动率期限结构斜率：rv_5日/rv_60日 − 1（陡峭＝短期波动高）",
    formula="rv_5min / rv_60min - 1,  rv_N = sqrt(roll_mean(ret2_sum, N))",
    deps=("stock_history_5min",),
    start=IDT_START,
    warmup_days=W_60,
    higher_is_better=False,
    note=(_DEG5 + "参考库 `rv_term_structure_slope` = 「rv_5min/rv_60min-1」，"
          "指的是**5 分钟窗 / 60 分钟窗**的 RV 之比（注意它的名字里 5min/60min "
          "是**窗口长度**、不是采样频率）—— 本项目按「5 日 / 60 日」两档实现，"
          "语义即「短期波动 / 长期波动」，与参考库的「短窗/长窗」一致，只是把"
          "窗口从分钟级拉到日级（本项目产物必须日频）。"
          "★ 分母是 60 日窗：`min_count=30 = 60//2`（模块 docstring 第 7 条）；"
          "分子 `min_count=2 = 5//2`。两边都开根后再相比，所以是「波动率之比」"
          "而不是「方差之比」（与参考库一致）。"
          "★ **额外要求当日 `ret2_sum` 有限**：否则窗口缺最近几天时，min_count "
          "会放行一个**陈旧**的斜率，把长期停牌与 2026-09-11 的数据冻结藏起来。"
          "★ 停牌日 NaN 而非 0（roll_mean 只在有效日上取均值）。"),
))
def idt_rv_term_structure_slope(ctx):
    var = _f(ctx, "ret2_sum")
    rv_s = ctx.safe_sqrt(ctx.roll_mean(var, 5, min_count=2))
    rv_l = ctx.safe_sqrt(ctx.roll_mean(var, 60, min_count=30))
    slope = ctx.safe_div(rv_s, rv_l, min_abs_den=1e-8) - 1.0
    return np.where(np.isfinite(var), slope, np.nan)


# ══════════════════════════════════════════════════════════════════════
# 4. 日内价格路径（5 个）
# ══════════════════════════════════════════════════════════════════════





@register(FactorSpec(
    name="idt_intraday_max_drawdown",
    group="intraday",
    desc="日内最大回撤：从盘内高点到后续低点的最大跌幅（≤0）",
    formula="min(close5_t / cummax(close5) - 1)  （= 日内层 max_dd 字段）",
    deps=("stock_history_5min",),
    start=IDT_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=(_DEG5 + "参考库 Class3 `intraday_max_drawdown` 的**定义**是「日内从最高点到"
          "后续最低点的最大跌幅」，公式却写 `rank(-dd.abs())`——`max_dd` 本身 ≤0，"
          "所以 `-|dd| ≡ dd`，**参考库的排序等价于「回撤浅的排前」**，本因子方向"
          "取 True（= 参考库的实际方向）。"
          "★ 取值 ∈[−1,0]（价格恒正），实测中位数 ≈ −0.02。"
          "★ 日内层用**收盘价序列**的 running max（不是 bar 的 high）——"
          "与参考库「从最高点到**后续**最低点」的口径一致（顺序敏感，"
          "不含「先跌后涨」式的回撤）。"
          "★ 5min 棒数从 240 降到 48，会**轻微低估**回撤深度（棒内极值被抹平），"
          "方向为对所有股票的同向压缩。"),
))
def idt_intraday_max_drawdown(ctx):
    return _f(ctx, "max_dd")




@register(FactorSpec(
    name="idt_lo_pos",
    group="intraday",
    desc="日内最低价时点（归一化到 [0,1]）：越晚见低＝尾盘走弱（方向为负）",
    formula="argmin(close5) / (n_bars - 1)  （日内层 lo_pos，已在 [0,1]）",
    deps=("stock_history_5min",),
    start=IDT_START,
    warmup_days=W_D,
    higher_is_better=False,
    note=(_DEG5 + "★★ **本因子在参考库里没有同名条目**：Class3 只有 "
          "`intraday_high_time`（最高价时点），没有 `intraday_low_time`。"
          "这里按镜像构造（同一字段族、同一归一化口径），"
          "方向取负 = 「最低点越晚出现越差」（尾盘才见低 = 尾盘抛压）。"
          "**方向是本次推断的，未经 V8 独立筛查，下游首次使用前应用样本外 IC 复核。**"
          "★ 与 `idt_hi_pos` 同为 [0,1] 的归一化时点，直接透传字段、不再除 n_bars。"
          "★ 与 hi_pos 的分析口径一致：分辨率 48 档，大量并列。"),
))
def idt_lo_pos(ctx):
    return _f(ctx, "lo_pos")


# ══════════════════════════════════════════════════════════════════════
# 5. 量能结构（3 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="idt_vol_stability",
    group="intraday",
    desc="成交量稳定性：日内 5min 量的变异系数 std/mean（越大越不稳定）",
    formula="std(vol_5min) / mean(vol_5min) = sqrt(n_bars * vol2_sum*vol_mult / vol^2 - 1)",
    deps=("stock_history_5min",),
    start=IDT_START,
    warmup_days=W_D,
    higher_is_better=False,
    note=(_DEG5 + "★ 参考库选型说明：父任务清单里的 `vol_concentration` 在参考库中的"
          "定义是「(开盘 30 分 + 收盘 30 分) 成交量 / 全日成交量」，"
          "**预聚合表没有首/尾 30 分钟切片，无法复现**。改用参考库 Class3 "
          "`vol_stability`：「5 分钟成交量 std/均值截面排名（取负向=不稳定排后）」——"
          "它**本身就是 5 分钟口径**，本项目可以**零退化**复现；"
          "而且它与「量能时间集中度」互为单调变换"
          "（CV = √(n·HHI − 1)，HHI = vol2_sum/vol²），是同一维度的等价刻画。"
          "★ 用**总体**标准差（ddof=0）与参考库 pandas 的 `std` 默认口径一致；"
          "ddof=1 只差常数因子 √(n/(n−1))，截面 rank 完全不变。"
          "★ 量单位为「手 vs 股」的翻转在本公式里自动约掉（比值），"
          "所以不受 2025-11/12 边界影响。"
          "★ `vol=0`（当天只有停牌级别的异常）→ NaN。"),
))
def idt_vol_stability(ctx):
    vol = _f(ctx, "vol")
    # ★ `vol2_sum` 的单位归一（乘手→股）由 `fea/intraday.py` 在层内完成
    #   （那个"平方量只乘了一次 mult"的 bug 已修复，因子侧不再需要补偿；
    #    再乘一次 `vol_mult` 会双重放大）。
    v2 = _f(ctx, "vol2_sum")
    n = _f(ctx, "n_bars")
    hhi = ctx.safe_div(v2, vol * vol, min_abs_den=0.0)
    cv2 = n * hhi - 1.0                      # 柯西下界保证 ≥ 0（浮点误差下可能 −1e-16）
    return ctx.safe_sqrt(np.maximum(cv2, 0.0))






# ══════════════════════════════════════════════════════════════════════
# 6. 分钟级方向占比（1 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="idt_up_minutes_ratio",
    group="intraday",
    desc="上涨棒占比：n_up / n_bars，买盘持续主导",
    formula="n_up / n_bars   （5min 口径 = 参考库 intra_trend）",
    deps=("stock_history_5min",),
    start=IDT_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=(_DEG5 + "★ 参考库 `up_minutes_ratio` 是「1 分钟正收益分钟数/总分钟数」；"
          "本项目是 5 分钟棒，所以本因子**精确等于**参考库同族的 `intra_trend`"
          "（其定义逐字为「正收益 5 分钟区间占比」）。二者同义不同分辨率，"
          "方向均为正（买盘主导排前）。"
          "★ 取值 ∈[0,1]：分子 `n_up` 只数 `r>0` 的棒。"
          "★ 日内层的每根棒的首棒相对上一棒，**当日第一根棒没有前收**（段首 diff=NaN→0）"
          "所以在所有口径下都不计入 `n_up`，与参考库 1min 口径一致"
          "（240 根里最新一根也没有下一根）。副作用：全天一字涨停（所有棒收益为 0）"
          "会得到 `n_up=0` → 本因子 0.0 —— 这是**真的**（没有一根棒在涨），"
          "不是缺失；如果下游想区分「一字板」与「全天阴跌」，"
          "请配 `idt_rv_daily`（一字板 RV=0）或 `idt_close_position`（NaN）。"
          "★ 停牌日 NaN（日内层无行）。"),
))
def idt_up_minutes_ratio(ctx):
    return ctx.safe_div(_f(ctx, "n_up"), _f(ctx, "n_bars"), min_abs_den=0.0)


# ══════════════════════════════════════════════════════════════════════
# 附录：本族想做但**当前框架做不到**的因子（需引擎扩字段，不是本文件能解决的）
# ══════════════════════════════════════════════════════════════════════
#
# 1. `open_30_momentum` / `close_30_momentum` / `last30_ret`
#    （参考库 Class3：开盘 30 分钟收益 / 收盘 30 分钟收益 / 最后 30 分钟收益）
#    → 需要 (09:35-10:00) 与 (14:35-15:00) 两个切片的价格。
#      预聚合表只有 `am_*`（09:35-11:30，24 根）与 `pm_*`（13:01-15:00，24 根）
#      两段，拿不出首/尾 6 根棒。建议在 `fea/intraday.py` 的 `for tag, sid in
#      (("am",0),("pm",1))` 后面再加两个 tag（`open30` = hm ≤ 1000、
#      `close30` = hm ≥ 1435，各自 open5/close5/hi5/lo5/vol/amt），
#      成本几乎为零（同一套 range_reduce 已经在那儿了），加完这 3 个因子各 3 行。
#
# 2. `volume_profile_skew`（父任务清单里的名字；参考库对应 `volume_profile_kurt`，
#    源文件 intraday_extra.py，需要分钟成交量的三阶/四阶矩）
#    → 预聚合表只有 `vol2_sum`（二阶）与 `vol_peak_pos`。建议加
#      `vol3_sum`（Σv³），随后 CV 偏度 = (Σv³/n − 3μσ² − μ³)/σ³ 一行可得。
#
# 3. `intraday_reversal_intensity`（父任务清单里的名字，Class3 无同名条目，
#    语义为「日内分钟收益的反转强度」，需要分钟收益的**一阶自协方差** Σ r_i·r_{i+1}）
#    → 预聚合表没有 lag-1 交叉项。建议加 `ret_ac1_sum`（段内 Σ r_i*r_{i+1}），
#      一行 reduceat 的事。
#
# 4. `am_pm_rv_ratio` 的真 RV 版本：需要分时段的 `am_ret2_sum` / `pm_ret2_sum`
#    （现在用的是 Parkinson 代理，见该因子 note）。
#
# 5. `vol_concentration`（参考库定义 = (开盘30分+收盘30分)量/全日量）
#    → 与第 1 条同一套切片，加 `open30_vol` / `close30_vol` 即可。
#      当前用 `idt_vol_stability` 替代（同为量能集中度维度的等价刻画）。
#
# 6. `DEP_PIT_COL`（`fea/engine.py`）里没有 `stock_history_5min: "trade_time"`
#    → 水位探测因为该表没有 `trade_date` 列而退化成**全量读**（实测 5.3s/次）。
#      L2 失效判定靠行数变化仍能工作，但每次 plan() 都会白读一遍 8.4 GB 分区的元数据
#      之外的统计。建议补一行。
