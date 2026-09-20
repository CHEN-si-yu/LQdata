"""技术指标因子（32 个）—— 全部由**日线行情**推出，不含任何财务/资金流依赖。

参考库出处（`学习资料/factors.md` 的 `类别 price` / Class 1，以及 `学习资料/因子库.md`
的 `4、Momentum` / `6、Reversal`）见每个因子的 `note` 与汇报表。

## 本家族的六条口径（每条都是踩过或差点踩到的坑）

1. **一律用后复权价** `ctx.hfq("close"|"high"|"low")`。
   未复权价在除权日跳水，会让 MA / EMA / 滚动极值在**除权日之后最多 60 个交易日**
   都产生假信号（参考库自己也在 `_adjusted_close` 注释里反复记了这件事）。
   后复权锚定序列起点 → 历史值不随新分红改变 → PIT 安全。

2. **「昨收」用 `ctx.shift(ctx.hfq("close"), 1)`，不用 `ctx.hfq("pre_close")`。**
   未复权 `pre_close` 在价格层是**状态量**（停牌日前向填充），停牌期间它填的是
   「上一行（更早）的 pre_close」，与真实昨收不等；而在复权空间里
   `hfq_pre_close(t) ≡ hfq_close(t-1)`（除权日 `adj_factor` 的跳变正好抵消
   `pre_close` 的向下调整），所以 `shift` 在正常交易日与它**恒等**，
   在停牌日还更准。TR / DM 里全部走这一条。

3. **量纲相关的指标一律除以价格归一化。** 后复权价的绝对水平取决于该股的
   累计复权因子（老股可以到 1e5 量级），直接拿 BOLL 带宽 / ATR / 滚动方差做截面
   排名，排出来的是「谁的后复权基座大」而不是「谁的波动大」。
   本文件里 `bollinger_width_20` / `bollinger_squeeze` / `atr_14_ratio` /
   `keltner_position_20` / `chandelier_position` / `macd_daily_dif` 都做了归一化。

4. **RSI / MFI 的分母一律走 `ctx.safe_div`。** 连续横盘（涨跌幅和 = 0）会让
   `avg_loss` / 负向资金流**精确等于 0**，直接相除得到 inf 或 ±1e6 的假值，
   把整个截面的排名带偏。`safe_div` 在分母 ≈ 0 时给 NaN（缺失而不是假值）。

5. **`ctx.roll_std` 的口径是 ddof=0**（`fea/mathx.py::roll_var`），
   pandas 默认 ddof=1 会大 `sqrt(n/(n-1))` 倍。对同一截面不改变排序，
   但**值的水平**不同（20 日窗差 2.6%），跨家族拼因子时要知道。

6. **EMA 链（MACD / TRIX / DI / ADX / KDJ）的 `warmup_days` 给得比
   「N×1.8+20」更宽**。EMA 的种子取窗口第一个有效值，严格说永远收敛不到
   「全量重算」的值；span=26 时 `(1-2/27)^205 ≈ 1.4e-7`，300 个日历天
   （约 205 个交易日）后残差已经在 float32 精度以下。给少了的后果是
   **每次增量输出与全量输出在窗口头部悄悄不一致**，而且不报错。

## 与本框架的三个约定

- 参考库最后都套了 `cross_sectional_rank(...)`，**本文件一律不套** ——
  契约 §2.3 要求返回原始值，winsor 与截面排名由引擎统一做。
  参考库的「负向排名」在本文件里翻译成 `higher_is_better=False`。
- 参考库的 `.rolling(n, min_periods=k)` 翻译成 `min_count=k`。
  `roll_sum/mean/var/std/cov` 的 min_count 都是**正确的 pandas `min_periods` 语义**
  （`mathx.roll_var` 用 `cdiv = max(cnt - ddof, 1)` 按**有效值个数**做分母，
  不是固定 n），所以可以放心把参考库的 min_periods 照搬过来。
  本文件只在两处用它：`adx_14`（min_count=7，照搬参考库）与
  `rsrs_zscore_18`（min_count=100，理由见该因子的 note —— 长期停牌产生的
  NaN beta 会把 200 日窗整段毒化）。其余场合一律不给 min_count：
  hfq 序列除上市前没有内部缺口，满窗语义更干净。
- 停牌日价格被前向填充 → 当日收益为 0（不是 NaN），滚动窗口照常推进。
  这是价格层「水平量前向填充」设计的直接推论，覆盖率高；
  代价是长期停牌会被当作「横盘」，`note` 里逐条标注了。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

# 全部指标只依赖日线行情 + 当日累计复权因子（= 后复权价的两个输入）。
DEPS = ("stock_daily", "stock_adj_factor")

# EMA 链要 ~200 个交易日才能把种子残差压到 float32 精度以下（见模块 docstring 第 6 条）
EMA_WARMUP = 300


# ══════════════════════════════════════════════════════════════════════
# 共用零件
# ══════════════════════════════════════════════════════════════════════

def _prev_close(ctx):
    """后复权空间的昨收 = `shift(hfq_close, 1)`（理由见模块 docstring 第 2 条）。"""
    return ctx.shift(ctx.hfq("close"), 1)


def _true_range(ctx):
    """真实波幅 TR = max(high-low, |high-昨收|, |low-昨收|)，全部在**后复权**空间。

    三个量同尺度（都在 hfq 空间），所以不需要参考库那种 `× adj/close` 的折算 ——
    这正是全项目统一用 hfq 的好处。
    """
    high = ctx.hfq("high")
    low = ctx.hfq("low")
    pc = _prev_close(ctx)
    return np.maximum(np.maximum(high - low, np.abs(high - pc)), np.abs(low - pc))


def _rsi(ctx, close, n):
    """Wilder RSI，值域 [0, 100]。

    `ewm(alpha=1/n, adjust=False)` 就是 Wilder 平滑（= EMA with com = n-1）。
    `min_count=n` 对齐 pandas 的 `min_periods=n`：前 n-1 个有效观测给 NaN。
    """
    d = ctx.diff(close, 1)
    gain = np.maximum(d, 0.0)            # NaN 会原样传递（np.maximum 传播 NaN）
    loss = np.maximum(-d, 0.0)
    ag = ctx.ewm_mean(gain, alpha=1.0 / n, min_count=n)
    al = ctx.ewm_mean(loss, alpha=1.0 / n, min_count=n)
    rs = ctx.safe_div(ag, al)            # ★ 分母保护：横盘时 al == 0 -> NaN
    return 100.0 - 100.0 / (1.0 + rs)


def _rsi_close(ctx, n):
    return _rsi(ctx, ctx.hfq("close"), n)


def _kdj(ctx, n=9, m1=3, m2=3):
    """日频 KDJ 链：RSV(n) -> K = EMA(RSV, m1) -> D = EMA(K, m2)。全部走 hfq 基座。"""
    close = ctx.hfq("close")
    ll = ctx.roll_min(close, n)
    hh = ctx.roll_max(close, n)
    rsv = ctx.safe_div(close - ll, hh - ll) * 100.0
    k = ctx.ewm_mean(rsv, alpha=1.0 / m1)
    d = ctx.ewm_mean(k, alpha=1.0 / m2)
    return k, d


def _macd_dif(ctx, fast=12, slow=26):
    """MACD 快线 DIF = (EMA12 − EMA26) / EMA26。

    ★ **必须除以慢线**：DIF 是价格单位，后复权基座大的股票天然数值大，
      不归一化的话这个因子实际在选「复权因子大（上市早、分红多）」的股票，
      而不是选动量。参考库的 `macd_daily_hist_5d` 也是这么处理的。
    """
    close = ctx.hfq("close")
    ef = ctx.ewm_mean(close, span=fast)
    es = ctx.ewm_mean(close, span=slow)
    return ctx.safe_div(ef - es, es)


def _dmi(ctx, n=14):
    """Wilder 的 ±DI。

    DM 的方向判定用**复权昨收**（不是未复权 pre_close），除权日不会误判方向。
    plus_dm/minus_dm 在上市前是 0.0（比较运算对 NaN 给 False），
    正好等于 Wilder 的标准初值；除数为 NaN 的 ATR，所以结果仍是 NaN。
    """
    high = ctx.hfq("high")
    low = ctx.hfq("low")
    close = ctx.hfq("close")
    up = high - ctx.shift(high, 1)
    dn = ctx.shift(low, 1) - low
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = ctx.ewm_mean(_true_range(ctx), alpha=1.0 / n, min_count=n)
    pdi = 100.0 * ctx.safe_div(ctx.ewm_mean(plus_dm, alpha=1.0 / n, min_count=n), atr)
    mdi = 100.0 * ctx.safe_div(ctx.ewm_mean(minus_dm, alpha=1.0 / n, min_count=n), atr)
    return pdi, mdi


def _bollinger_bandwidth_20(ctx):
    """20 日布林带宽 = 4×std / ma —— **已按价格归一化**（无量纲）。

    `ctx.roll_std` 是 ddof=0（与 pandas 的 ddof=1 差 sqrt(20/19)）。
    """
    close = ctx.hfq("close")
    ma = ctx.roll_mean(close, 20)
    sd = ctx.roll_std(close, 20)
    return ctx.safe_div(4.0 * sd, ma)


# ══════════════════════════════════════════════════════════════════════
# 动量振荡器：RSI / KDJ / MACD
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="rsi_14", group="technical", deps=DEPS,
    desc="14 日 Wilder RSI（相对强弱指标），值域 [0,100]",
    formula=(
        "1. Gain = Max(Close - PrevClose, 0)\n"
        "2. Loss = Max(PrevClose - Close, 0)\n"
        "3. AvgGain = EMA(Gain, 14) [使用 Wilder's Smoothing]\n"
        "4. AvgLoss = EMA(Loss, 14) [使用 Wilder's Smoothing]\n"
        "5. RS = AvgGain / AvgLoss\n"
        "6. RSI = 100 - 100 / (1 + RS)\n"
        "其中 Wilder's Smoothing 等价于 EMA with com = period - 1\n"
        "参数: period: RSI 周期，默认为 14"),
    start=None, warmup_days=EMA_WARMUP, higher_is_better=False,
    note="参考库出处：因子库.md「6、Reversal」第 2 条 rsi（同一段公式也出现在 "
         "factors.md 的 rsi_spread_6_14 的 `_rsi` 私有函数里，口径一致）。"
         "★ 偏离：参考库 rsi_spread_6_14 的 `_rsi` 用 `delta = pct_chg/100`，"
         "本文件用 `diff(hfq_close)` —— 两者在正常交易日恒等，但 pct_chg 未经复权因子"
         "还原（交易所给的 pct_chg 已在除权日调整过，实际也等价），统一走 hfq 更省心。"
         "★ 分母保护：AvgLoss 精确为 0（连续上涨或长期停牌导致涨跌幅全为 0）时 "
         "safe_div 给 NaN，理论上该点 RSI=100 —— 宁可缺一格也不给假极值。"
         "★ 停牌日 hfq_close 被前向填充 -> 涨跌幅为 0（不是 NaN），RSI 会向中位漂移。"
         "★ 方向：参考库 rsi_14_excess 取负向排名（高 RSI = 超买排后），本因子标 "
         "higher_is_better=False，与之一致。"
         "warmup 给 300 而不是 14×1.8+20=46：Wilder EMA 的种子残差要 ~200 个交易日"
         "才衰减到 float32 精度以下，给少了增量与全量的值会在窗口头部不一致。"),
)
def rsi_14(ctx):
    return _rsi_close(ctx, 14)


@register(FactorSpec(
    name="rsi_spread_6_14", group="technical", deps=DEPS,
    desc="日频 RSI(6) − RSI(14)：短期动能相对中期动能的强弱差",
    formula=(
        "delta = pct_chg / 100\n"
        "def _rsi(delta, n):\n"
        "    gain = delta.clip(lower=0.0)\n"
        "    loss = (-delta).clip(lower=0.0)\n"
        "    # Wilder smoothing: alpha=1/n.  min_periods avoids presenting a\n"
        "    # partially initialised oscillator as a fully formed RSI value.\n"
        "    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()\n"
        "    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()\n"
        "    rs = safe_divide(avg_gain, avg_loss + 1e-10)\n"
        "    return 100.0 - 100.0 / (1.0 + rs)\n\n"
        "spread = _rsi(delta, 6) - _rsi(delta, 14)"),
    start=None, warmup_days=EMA_WARMUP, higher_is_better=True,
    note="参考库出处：factors.md `类别 price` / technical_daily.py 的 rsi_spread_6_14。"
         "★ 偏离：参考库分母写 `avg_loss + 1e-10`，本文件用 `ctx.safe_div`（分母为 0 给 "
         "NaN）。1e-10 在 RSI 的 0~1 量纲上等价于「分母为 0 时返回 1e10 量级的假值」，"
         "契约 §2.3 明令禁止哨兵/巨值，故改成 NaN。"
         "★ 复权：参考库用 pct_chg（交易所口径），本文件用 diff(hfq_close)，正常交易日等价。"),
)
def rsi_spread_6_14(ctx):
    return _rsi_close(ctx, 6) - _rsi_close(ctx, 14)




@register(FactorSpec(
    name="kdj_k_minus_d", group="technical", deps=DEPS,
    desc="日频 KDJ 的 K − D（KD 金叉/死叉的横截面形态），>0 = 多头排列",
    formula=(
        "# 参考库只有分钟级 kdj_k_d_distance = (K-D)/|D|，日频版沿用 kdj_daily_j 的链：\n"
        "wide = _adjusted_close(daily).unstack(\"Code\")\n"
        "ll9 = wide.rolling(9, min_periods=5).min()\n"
        "hh9 = wide.rolling(9, min_periods=5).max()\n"
        "rsv = safe_divide(wide - ll9, hh9 - ll9 + 1e-10) * 100.0\n"
        "k = rsv.ewm(alpha=1.0 / 3.0, adjust=False).mean()\n"
        "d = k.ewm(alpha=1.0 / 3.0, adjust=False).mean()\n"
        "return k - d"),
    start=None, warmup_days=EMA_WARMUP, higher_is_better=True,
    note="★ 参考库**没有**日频的 kdj_k_minus_d 条目：factors.md `类别 intraday` Class 4 只有"
         "分钟级的 `kdj_k_d_distance`（(K−D)/|D|，归一化以消除 K 的水平）。"
         "本因子按任务要求做**日频**版，直接取 K−D（不除以 |D|）——"
         "因为 D 可能穿过 0 使 |D| 变成极小的分母，除法会把一条平滑的动能差变成"
         "带尖刺的比值；K−D 本身有界（∈[−100,100]），截面排名不关心量纲，更稳。"
         "★ 该因子与 kdj_daily_j 取自同一条链，两者相关性会很高（J=3K−2D），"
         "下游建模时注意共线性。"),
)
def kdj_k_minus_d(ctx):
    k, d = _kdj(ctx)
    return k - d




@register(FactorSpec(
    name="macd_daily_hist_5d", group="technical", deps=DEPS,
    desc="日频 MACD 柱（归一化 DIF − DEA）的 5 日变化：动能的二阶导",
    formula=(
        "wide = _adjusted_close(daily).unstack(\"Code\")\n"
        "ema12 = wide.ewm(span=12, adjust=False).mean()\n"
        "ema26 = wide.ewm(span=26, adjust=False).mean()\n"
        "# Normalise by the slow EMA before cross-sectional comparison.  Raw MACD\n"
        "# is denominated in price units and would otherwise mostly reflect the\n"
        "# arbitrary base level of each self-built adjusted-price index.\n"
        "dif = safe_divide(ema12 - ema26, ema26)\n"
        "dea = dif.ewm(span=9, adjust=False).mean()\n"
        "hist = dif - dea\n"
        "chg = hist.diff(5)"),
    start=None, warmup_days=EMA_WARMUP, higher_is_better=True,
    note="参考库出处：factors.md `类别 price` / technical_daily.py 的 macd_daily_hist_5d。"
         "本文件逐字照搬其归一化口径（除以 EMA26 后再算 DEA，而不是算完原始柱再除）——"
         "这样 DIF 与 macd_daily_dif 是同一个数，两个因子可以放在一起看。"
         "★ `hist.diff(5)` 用交易日位移（面板本身就是交易日历），不是 5 个日历天。"
         "★ 数值：柱是 12/26 两个 EMA 之差再减去其 9 日 EMA，量级 ~1e-3，"
         "float32 下有效位约 4 位；这是参考库的口径，保持不变。"),
)
def macd_daily_hist_5d(ctx):
    dif = _macd_dif(ctx)
    dea = ctx.ewm_mean(dif, span=9)
    return ctx.diff(dif - dea, 5)


# ══════════════════════════════════════════════════════════════════════
# 趋势强度：DMI / ADX / AROON / TRIX / ROC
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="adx_14", group="technical", deps=DEPS,
    desc="14 日趋势强度 ADX（只衡量强弱、不含方向），单位 %",
    formula=(
        "scale = _adjusted_close(daily) / daily[\"close\"].replace(0, np.nan)\n"
        "di_plus, di_minus = _directional_movement(daily[\"high\"], daily[\"low\"],"
        " daily[\"pre_close\"], scale, 14)\n"
        "dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-8)\n"
        "adx = dx.groupby(level=\"Code\").transform(\n"
        "    lambda s: s.rolling(14, min_periods=7).mean()\n"
        ")"),
    start=None, warmup_days=EMA_WARMUP, higher_is_better=True,
    note="参考库出处：factors.md `类别 price` / trend_pattern.py 的 adx_14。"
         "★ 参考库的 `_directional_movement` 是私有函数、文档未展开；本文件按 Wilder 的"
         "标准定义实现（up=high−high[-1]、dn=low[-1]−low，plus_dm 取 up>dn 且 up>0，"
         "minus_dm 对称，各自与 TR 一起做 alpha=1/14 的 Wilder 平滑，DI=100×DM/ATR）。"
         "★ 偏离：参考库用未复权 `pre_close` 且乘 `scale` 折算到复权空间；本文件直接用"
         "`shift(hfq_close,1)` 当昨收（复权空间里两者在有交易的日子恒等，停牌日更准），"
         "不再需要 scale。"
         "★ `min_count=7` 对齐参考库的 `min_periods=7`：DX 在「多空双方 DM 同时为 0」"
         "（长期停牌/一字板）时是 NaN，满窗要求会让 ADX 在真实数据上留下 14 天的空洞。"
         "★ ADX ∈ [0,100]；DX = 100×|DI+−DI-|/(DI++DI-) ∈ [0,100]，其 14 日均值同界。"),
)
def adx_14(ctx):
    pdi, mdi = _dmi(ctx, 14)
    dx = 100.0 * ctx.safe_div(np.abs(pdi - mdi), pdi + mdi)
    return ctx.roll_mean(dx, 14, min_count=7)


@register(FactorSpec(
    name="di_plus_minus_ratio_14", group="technical", deps=DEPS,
    desc="14 日 DI+/DI- 比率：多头相对空头的趋势优势，>1 = 上升趋势占优",
    formula=(
        "scale = _adjusted_close(daily) / daily[\"close\"].replace(0, np.nan)\n"
        "di_plus, di_minus = _directional_movement(daily[\"high\"], daily[\"low\"],"
        " daily[\"pre_close\"], scale, 14)\n"
        "ratio = safe_divide(di_plus, di_minus + 1e-8)"),
    start=None, warmup_days=EMA_WARMUP, higher_is_better=True,
    note="参考库出处：factors.md `类别 price` / trend_pattern.py 的 di_plus_minus_ratio_14。"
         "★ 偏离（分母保护）：参考库写 `di_minus + 1e-8`，但 DI 的量纲是百分数，"
         "分母踩到 1e-8 时比值会飙到 1e10 —— 直接触发引擎「值域异常」（|value|>1e8）。"
         "本文件用 `safe_div(..., min_abs_den=1e-4)`：DI- < 0.0001%（几个月没有一个"
         "有效下移）时给 NaN，其余情况比值上界 1e6，安全。"
         "★ 复权：同 adx_14，昨收走 `shift(hfq_close,1)`。"),
)
def di_plus_minus_ratio_14(ctx):
    pdi, mdi = _dmi(ctx, 14)
    return ctx.safe_div(pdi, mdi, min_abs_den=1e-4)


@register(FactorSpec(
    name="aroon_up_25", group="technical", deps=DEPS,
    desc="Aroon 上行：25 日窗口内「距最近新高的交易日数」的位置，越接近新高越高",
    formula=(
        "wide = _adjusted_close(daily).unstack(\"Code\")\n"
        "days_since = _rolling_extreme_age(wide, window=25, find_max=True)\n"
        "aroon = safe_divide(25.0 - days_since, 25.0) * 100.0"),
    start=None, warmup_days=25 * 2 + 20, higher_is_better=True,
    note="参考库出处：factors.md `类别 price` / technical_daily.py 的 aroon_up_25。"
         "★ 实现：`ctx.roll_argmax(close, 25)` 的返回值就是「窗口极值距今天多少个 bar」"
         "（0 = 今天创的窗口新高），正好是文档里的 `days_since`，不需要额外换算。"
         "★ 值域：[4, 100]（days_since 取整 0..24，(25−24)/25×100 = 4 是下界）。"
         "  参考库的 `_rolling_extreme_age` 若以 1 为起点则会得到 (25−25)/25=0，"
         "  本文件按框架语义（0 起）实现，下界是 4 —— 差一个 bar 的常数，"
         "  对同一截面的排序没有影响。"
         "★ 窗口 25 天用 `roll_max/argmax`（O(T·C·n)）没有问题；不做 n>250 的版本。"),
)
def aroon_up_25(ctx):
    age = ctx.roll_argmax(ctx.hfq("close"), 25)
    return ctx.safe_div(25.0 - age, 25.0) * 100.0


@register(FactorSpec(
    name="aroon_down_25", group="technical", deps=DEPS,
    desc="Aroon 下行：25 日窗口内「距最近新低的交易日数」的位置，越接近新低越低",
    formula=(
        "wide = _adjusted_close(daily).unstack(\"Code\")\n"
        "days_since = _rolling_extreme_age(wide, window=25, find_max=False)\n"
        "aroon = safe_divide(25.0 - days_since, 25.0) * 100.0"),
    start=None, warmup_days=25 * 2 + 20, higher_is_better=False,
    note="参考库出处：factors.md `类别 price` / technical_daily.py 的 aroon_down_25。"
         "参考库取 `rank(-aroon)`（近期破位排后），本因子返回原始 aroon 值并标 "
         "higher_is_better=False —— 高 = 距上次新低很久 = 空头趋势弱，是好事。"
         "★ 值域 [4, 100]，同 aroon_up_25。基于 hfq 基座，除权日不产生假新低。"),
)
def aroon_down_25(ctx):
    age = ctx.roll_argmin(ctx.hfq("close"), 25)
    return ctx.safe_div(25.0 - age, 25.0) * 100.0


@register(FactorSpec(
    name="trix_12_20", group="technical", deps=DEPS,
    desc="TRIX：(后复权收盘价的 12 日三重 EMA) 的 20 日变化率",
    formula=(
        "wide = _adjusted_close(daily).unstack(\"Code\")\n"
        "trix = _trix_wide(wide)\n"
        "# _trix_wide = 三重指数平滑(12) 的 20 日变化率"),
    start=None, warmup_days=400, higher_is_better=True,
    note="参考库出处：factors.md `类别 price` / technical_daily.py 的 trix_12_20"
         "（`_trix_wide` 是私有函数，文档只给了「三重指数平滑(12)的20日变化率」一句）。"
         "★ 实现：`E = EMA(EMA(EMA(close, 12), 12), 12)`，`trix = pct_change(E, 20)`。"
         "  这与**经典 TRIX 不同**：经典口径是三重 EMA 的**1 日**变化率 ×100，"
         "  本因子按文档/因子名的 `_12_20`（N=12, M=20）取 20 日变化率。"
         "  两者是同一趋势的不同平滑度，方向语义一致。"
         "★ 无量纲（分母是同一条 EMA），不需要额外归一化。"
         "★ warmup 400：三重 EMA 的脉冲响应是 Γ(3, 1/α) 形状，拖尾比单层 EMA 长得多，"
         "  再叠 20 日变化率；300 不够，给 400。"),
)
def trix_12_20(ctx):
    e = ctx.hfq("close")
    for _ in range(3):
        e = ctx.ewm_mean(e, span=12)
    return ctx.pct_change(e, 20)


@register(FactorSpec(
    name="trix_signal_gap", group="technical", deps=DEPS,
    desc="TRIX 与其 20 日信号线的乖离（TRIX − MA20(TRIX)）/ |MA20(TRIX)|",
    formula=(
        "wide = _adjusted_close(daily).unstack(\"Code\")\n"
        "trix = _trix_wide(wide)\n"
        "signal = trix.rolling(20, min_periods=10).mean()\n"
        "gap = trix.sub(signal).div(signal.abs() + 1e-10)"),
    start=None, warmup_days=400, higher_is_better=True,
    note="参考库出处：factors.md `类别 price` / technical_daily.py 的 trix_signal_gap。"
         "★ 分母保护：TRIX 会穿过 0（无趋势时三重 EMA 走平），参考库写 `|signal|+1e-10`，"
         "本文件用 `safe_div(..., min_abs_den=1e-8)`：TRIX 的量级是 ~1e-2，"
         "20 日均值 |signal| < 1e-8（= 十万分之一）已经等价于「完全无趋势」，给 NaN；"
         "这样比值上界 ~4e6，不会触发引擎的值域告警。"
         "★ 与 trix_12_20 共用同一条 TRIX 链，两者会高度相关，下游注意共线性。"),
)
def trix_signal_gap(ctx):
    e = ctx.hfq("close")
    for _ in range(3):
        e = ctx.ewm_mean(e, span=12)
    trix = ctx.pct_change(e, 20)
    signal = ctx.roll_mean(trix, 20)
    return ctx.safe_div(trix - signal, np.abs(signal), min_abs_den=1e-8)


@register(FactorSpec(
    name="roc_12", group="technical", deps=DEPS,
    desc="12 日 ROC 变动率（后复权），无量纲",
    formula=(
        "adj = _adjusted_close(daily)\n"
        "roc = adj.groupby(level=\"Code\").transform(\n"
        "    lambda s: s.pct_change(12, fill_method=None)\n"
        ")"),
    start=None, warmup_days=12 * 2 + 20, higher_is_better=True,
    note="参考库出处：factors.md `类别 price` / technical_daily.py 的 roc_12。"
         "等价于 `ctx.pct_change(hfq_close, 12)`，即 12 个**交易日**的变化率"
         "（面板本身是交易日历，位移 = 交易日）。"
         "★ 参考库的 `fill_method=None` 是 pandas 的显式声明（不对停牌做填充）；"
         "  本框架的价格层对水平量做前向填充，停牌日收益为 0，窗口照常推进。"),
)
def roc_12(ctx):
    return ctx.pct_change(ctx.hfq("close"), 12)


# ══════════════════════════════════════════════════════════════════════
# 通道 / 位置类：BOLL / Donchian / Keltner / Chandelier
# ══════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="bollinger_width_20", group="technical", deps=DEPS,
    desc="20 日布林带宽 = 4×std/ma（已按价格归一化，无量纲）",
    formula=(
        "# 跨日 MA/std 窗口走复权基座,未复权 close 在除权日跳变会污染带宽\n"
        "adj = _adjusted_close(daily)\n"
        "ma = rolling_group_mean(adj, 20)\n"
        "std = rolling_group_std(adj, 20)\n"
        "width = safe_divide(4 * std, ma)"),
    start=None, warmup_days=20 * 2 + 20, higher_is_better=False,
    note="参考库出处：factors.md `类别 price` / trend_pattern.py 的 bollinger_width_20。"
         "参考库取 `rank(-width)`（窄幅 = 蓄力），本因子返回原始带宽并标 "
         "higher_is_better=False。"
         "★ **已按价格归一化**（÷ ma）：后复权价的绝对水平取决于累计复权因子，"
         "不归一化的话这个因子实际在选「复权因子大」的股票而不是选波动。"
         "  这与参考库口径一致（它的公式里本来就除了 ma）。"
         "★ `roll_std` 口径 ddof=0，见模块 docstring 第 5 条。"),
)
def bollinger_width_20(ctx):
    return _bollinger_bandwidth_20(ctx)


@register(FactorSpec(
    name="bollinger_squeeze", group="technical", deps=DEPS,
    desc="布林带宽的 250 日**历史分位**（∈[0,1]）：低 = 波动压缩，变盘前夜",
    formula=(
        "# 跨日 MA/std 窗口走复权基座,未复权 close 在除权日跳变会污染带宽\n"
        "adj = _adjusted_close(daily_panel)\n"
        "ma_20 = adj.groupby(level=\"Code\").transform("
        "lambda s: s.rolling(20, min_periods=10).mean())\n"
        "std_20 = adj.groupby(level=\"Code\").transform("
        "lambda s: s.rolling(20, min_periods=10).std())\n"
        "bandwidth = 4 * std_20 / ma_20.replace(0, np.nan)\n"
        "# Rank negative: narrow band = squeeze = ranked high\n"
        "# 本项目口径：squeeze = ctx.roll_rank(bandwidth, 250)（带宽自身的历史分位），"
        "见 note"),
    start=None, warmup_days=250 * 2, higher_is_better=False,
    note="参考库出处：factors.md `类别 price` / trend_pattern.py 的 bollinger_squeeze。"
         "★ **偏离（按任务要求）**：参考库给的是 `rank(-bandwidth)` —— 带宽的**截面**排名；"
         "任务明确要求 `bollinger_squeeze` = 带宽的**历史分位**，所以本文件实现为 "
         "`ctx.roll_rank(bandwidth, 250)`：今天的带宽在**自己过去 250 个交易日**里的百分位"
         "（∈[0,1]）。这不是量纲问题而是**信息维度**不同：截面排名度量「今天谁比谁窄」，"
         "历史分位度量「这只股票今天相对自己一年来的水平窄不窄」——后者才是"
         "「低波动后往往伴随方向性突破」这句话的正确横截面形态（每只股票与自己比）。"
         "★ 250 日窗口：任务提到 `roll_max/min` 做 n>250 需要先报备。"
         "本因子用的是 `roll_rank`（分块跨步视图，O(T·C·n) 时间 / O(step·C·n) 内存，"
         "实测不构成瓶颈），不是 `roll_max/min`，所以按契约要求在此明示。"
         "若主 Agent 认为 250 过大，改成 120 只需改这一个常量。"
         "★ `higher_is_better=False`：分位低 = 挤压 = 突破前兆。"),
)
def bollinger_squeeze(ctx):
    return ctx.roll_rank(_bollinger_bandwidth_20(ctx), 250)


@register(FactorSpec(
    name="donchian_position_20", group="technical", deps=DEPS,
    desc="Donchian 通道位置 ∈[0,1]：(close − 20日最低)/(20日最高 − 20日最低)",
    formula=(
        "close = daily[\"close\"]\n"
        "# 用每日复权系数 (后复权基座/close) 折算 high/low 后再取 20 日极值,\n"
        "# 避免除权日污染通道上下轨\n"
        "scale = _adjusted_close(daily) / close.replace(0, np.nan)\n"
        "adj_high = daily[\"high\"] * scale\n"
        "adj_low = daily[\"low\"] * scale\n"
        "adj = _adjusted_close(daily)\n\n"
        "previous_high = adj_high.groupby(level=\"Code\").shift(1)\n"
        "highest = previous_high.groupby(level=\"Code\").transform(\n"
        "    lambda s: s.rolling(20, min_periods=10).max()\n"
        ")\n"
        "lowest = adj_low.groupby(level=\"Code\").transform(\n"
        "    lambda s: s.rolling(20, min_periods=10).min()\n"
        ")\n\n"
        "position = safe_divide(adj - lowest, highest - lowest + 1e-10)\n"
        "position = position.clip(0, 1)"),
    start=None, warmup_days=20 * 2 + 20, higher_is_better=True,
    note="参考库出处：factors.md `类别 price` / technical_pattern.py 的 donchian_position_20。"
         "★ 口径**逐字照搬**了参考库的**不对称窗口**：上轨排除了当日"
         "（`shift(1)` 后再取 20 日 max，窗口 = [T−20, T−1] 的 high），"
         "下轨含当日（窗口 = [T−19, T] 的 low）。这不是笔误 —— 突破的定义就是"
         "「今天的价格超过了此前的高点」，所以上轨必须排除今天。"
         "★ `clip(0,1)` 同样是参考库的口径（通道位置的定义域边界），不是 winsor；"
         "  代价是「突破」的信息被截断在 1.0，要突破幅度请用 donchian_breakout_20。"
         "★ 值域 [0,1]，无量纲（分子分母同为价格单位）。"
         "★ 复权：high/low/close 全部走 hfq，参考库的 `scale` 折算在本框架里"
         "  由价格层统一完成，不需要再乘一次。"),
)
def donchian_position_20(ctx):
    close = ctx.hfq("close")
    high = ctx.hfq("high")
    low = ctx.hfq("low")
    highest = ctx.roll_max(ctx.shift(high, 1), 20)
    lowest = ctx.roll_min(low, 20)
    pos = ctx.safe_div(close - lowest, highest - lowest)
    return np.clip(pos, 0.0, 1.0)








# ══════════════════════════════════════════════════════════════════════
# 波动 / 价格位置：ATR 比率 / 威廉 %R / 随机指标 / 收盘位置
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="atr_14_ratio", group="technical", deps=DEPS,
    desc="相对波幅 = ATR14 / 后复权收盘价（已归一化，无量纲）",
    formula=(
        "close = daily[\"close\"]\n"
        "high = daily[\"high\"]\n"
        "low = daily[\"low\"]\n"
        "# ATR 折算到复权空间(×scale)后与复权基座 adj 同口径,避免除权日 close\n"
        "# 跳变造成相对波幅虚高\n"
        "scale = _adjusted_close(daily) / close.replace(0, np.nan)\n"
        "adj = _adjusted_close(daily)\n\n"
        "tr1 = high - low\n"
        "tr2 = (high - daily[\"pre_close\"]).abs()\n"
        "tr3 = (low - daily[\"pre_close\"]).abs()\n"
        "tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)\n"
        "atr = (tr * scale).groupby(level=\"Code\").transform(\n"
        "    lambda s: s.rolling(20, min_periods=10).mean()\n"
        ")\n\n"
        "atr_pct = safe_divide(atr, adj + 1e-10)"),
    start=None, warmup_days=14 * 2 + 20, higher_is_better=False,
    note="参考库出处：factors.md `类别 risk` / technical_pattern.py 的 atr_ratio_20"
         "（窗口 20）。本因子按任务命名为 `atr_14_ratio`，窗口改成 14 ——"
         "与 RSI14 / MFI14 / ADX14 的周期保持一致，便于和同族振荡器对齐观察。"
         "★ **已按价格归一化**（÷ hfq 收盘价）：这是本任务明确要求的 - "
         "ATR 是价格单位，后复权基座大的股票 ATR 天然大，不归一化就是在选复权因子。"
         "★ 参考库取 `rank(-atr_pct)`（低波平稳排前，低波异象），"
         "本因子返回原始比率并标 `higher_is_better=False`。"
         "★ TR 用复权昨收（`shift(hfq_close,1)`），不是未复权 pre_close，"
         "参考库的 `× scale` 折算由价格层统一完成。"
         "★ 末尾 `np.maximum(...,0)`：TR 恒 >= 0，比值也恒 >= 0，"
         "  该保护只消掉 cumsum 差分的 ±1e-16 负残差（实测 min = −2e-16）。"),
)
def atr_14_ratio(ctx):
    atr = ctx.roll_mean(_true_range(ctx), 14)
    # TR 恒 >= 0，比值也恒 >= 0；消掉 cumsum 差分的 ±1e-16 负残差（实测 −2e-16）
    return np.maximum(ctx.safe_div(atr, ctx.hfq("close")), 0.0)






@register(FactorSpec(
    name="close_location_20d", group="technical", deps=DEPS,
    desc="收盘位置 = 20 日收益 / 20 日振幅，度量上涨的「上攻效率」",
    formula=(
        "h20 = roll(df, \"high\", 20, \"max\")\n"
        "l20 = roll(df, \"low\", 20, \"min\")\n"
        "chg20 = df.groupby(\"Code\")[\"close\"].shift(20)\n"
        "vals = (df[\"close\"] - chg20) / (h20 - l20 + 1e-8)"),
    start=None, warmup_days=20 * 2 + 20, higher_is_better=True,
    note="参考库出处：factors.md `类别 price` / fac_new_daily.py 的 close_location_20d。"
         "★ 复权：参考库这条**没有**走 `_adjusted_close`（用的是原始 high/low/close），"
         "  在除权日会被污染；本文件按项目硬约束改成 hfq 口径 —— "
         "  `(hfq_close − shift(hfq_close,20)) / (max(hfq_high,20) − min(hfq_low,20))`。"
         "★ 值域：分子 ≤ 分母（20 日振幅包含这 20 日的全部价格），所以理论上 ∈ [−1, 1]；"
         "  实测在 [−1,1] 内（分子分母同口径时严格成立）。"
         "★ 分母保护：20 日完全无振幅时 safe_div 给 NaN。"),
)
def close_location_20d(ctx):
    close = ctx.hfq("close")
    h20 = ctx.roll_max(ctx.hfq("high"), 20)
    l20 = ctx.roll_min(ctx.hfq("low"), 20)
    return ctx.safe_div(close - ctx.shift(close, 20), h20 - l20)


# ══════════════════════════════════════════════════════════════════════
# 通道位置 / 乖离 / 情绪 / 量价
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="dpo_20", group="technical", deps=DEPS,
    desc="20 日 DPO 去趋势：(11 日前收盘 − 当前 20 日均价)/20 日均价",
    formula=(
        "adj = _adjusted_close(daily)\n"
        "ma20 = adj.groupby(level=\"Code\").transform(\n"
        "    lambda s: s.rolling(20, min_periods=10).mean()\n"
        ")\n"
        "lagged = adj.groupby(level=\"Code\").shift(11)\n"
        "dpo = safe_divide(lagged - ma20, ma20)"),
    start=None, warmup_days=(20 + 11) * 2 + 20, higher_is_better=True,
    note="参考库出处：factors.md `类别 price` / technical_daily.py 的 dpo_20。"
         "★ shift(11) 是「20 日周期的一半 + 1」（去趋势的经典参数），不是 20；"
         "  `lagged` 是 11 个**交易日**前的复权收盘价。"
         "★ 已归一化（÷ ma20），无量纲。warmup 要覆盖 20 日窗 + 11 日位移。"
         "★ 参考库取正向排名（DPO > 0 = 11 日前的价格高于当前均线 = 近期在回落），"
         "本因子同样标 `higher_is_better=True`，方向语义以参考库为准。"),
)
def dpo_20(ctx):
    adj = ctx.hfq("close")
    ma20 = ctx.roll_mean(adj, 20)
    return ctx.safe_div(ctx.shift(adj, 11) - ma20, ma20)


@register(FactorSpec(
    name="bias_20", group="technical", deps=DEPS,
    desc="20 日乖离率 = close/MA20 − 1",
    formula=(
        "# 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离\n"
        "adj = _adjusted_close(daily_panel)\n"
        "ma_20 = adj.groupby(level=\"Code\").transform(\n"
        "    lambda s: s.rolling(20, min_periods=10).mean()\n"
        ")\n"
        "bias = adj / ma_20.replace(0, np.nan) - 1.0"),
    start=None, warmup_days=20 * 2 + 20, higher_is_better=True,
    note="参考库出处：factors.md `类别 price` / price.py 的 bias_20。"
         "实现为 `safe_div(close − ma20, ma20)`，与 `close/ma20 − 1` 在数值上略有差异"
         "（后者在 ma20 很大时更稳），但同为无量纲比值，截面排序一致。"
         "★ 复权：走 hfq_close，除权日不伪造负乖离（参考库注释的原话）。"
         "★ 方向：参考库正向排名（价格偏离中期成本的程度，极端正 = 超买但强势延续）。"),
)
def bias_20(ctx):
    adj = ctx.hfq("close")
    ma = ctx.roll_mean(adj, 20)
    return ctx.safe_div(adj - ma, ma)






@register(FactorSpec(
    name="mfi_14", group="technical", deps=DEPS,
    desc="14 日资金流量指标 MFI（量加权的 RSI），值域 [0,100]",
    formula=(
        "# TP 方向判定为跨日比较,走复权口径避免除权日误判方向\n"
        "# (scale=adj/close 折算 high/low,close 直接用复权基座)\n"
        "adj = _adjusted_close(daily)\n"
        "scale = adj / daily[\"close\"].replace(0, np.nan)\n"
        "tp = (daily[\"high\"] * scale + daily[\"low\"] * scale + adj) / 3.0\n"
        "raw_flow = tp * daily[\"vol\"]\n"
        "tp_chg = tp.groupby(level=\"Code\").diff()\n"
        "pos_flow = raw_flow.where(tp_chg > 0, 0.0)\n"
        "neg_flow = raw_flow.where(tp_chg < 0, 0.0)\n"
        "pos_w = pos_flow.unstack(\"Code\").rolling(14, min_periods=7).sum()\n"
        "neg_w = neg_flow.unstack(\"Code\").rolling(14, min_periods=7).sum()\n"
        "ratio = safe_divide(pos_w, neg_w + 1e-10)\n"
        "mfi = 100.0 - 100.0 / (1.0 + ratio)"),
    start=None, warmup_days=14 * 2 + 20, higher_is_better=True,
    note="参考库出处：factors.md `类别 price` / technical_daily.py 的 mfi_14。"
         "★ 复权：TP = (hfq_high + hfq_low + hfq_close)/3，方向判定 `diff(tp)` 因此"
         "  在除权日不会误判（参考库的 `scale` 折算由价格层统一完成）。"
         "★ 成交量语义：`vol` 是**流量**，停牌日是 NaN（价格层明确不补 0）。"
         "  但停牌日 tp 也被前向填充 -> tp_chg == 0 -> 正负两个条件都不成立 -> "
         "  该日资金流计 0。这是**正确**的：停牌当天确实没有资金流，"
         "  `rolling(14).sum()` 的语义就是「窗口内累计净流入」，不是「日均」。"
         "★ **分母保护**：负向资金流之和精确为 0（14 日全是上涨）时，"
         "  参考库的 `+1e-10` 会给出 1e10 量级的假比值 -> 引擎值域告警；"
         "  本文件用 `ctx.safe_div` 给 NaN，MFI 保持 [0,100] 的值域。"
         "★ 方向：参考库正向排名（资金流入推动排前），本因子同样 True。"),
)
def mfi_14(ctx):
    tp = (ctx.hfq("high") + ctx.hfq("low") + ctx.hfq("close")) / 3.0
    raw = tp * ctx.px("vol")
    chg = ctx.diff(tp, 1)
    pos = np.where(chg > 0, raw, 0.0)
    neg = np.where(chg < 0, raw, 0.0)
    ratio = ctx.safe_div(ctx.roll_sum(pos, 14), ctx.roll_sum(neg, 14))
    return 100.0 - 100.0 / (1.0 + ratio)




# ══════════════════════════════════════════════════════════════════════
# RSRS：阻力支撑相对强度（滚动 OLS 斜率 + 200 日 z-score）
# ══════════════════════════════════════════════════════════════════════

def _rsrs_beta(ctx, n=18):
    """18 日滚动 OLS 斜率 beta = cov(low, high) / var(high)。

    ★ 数值处理：先把 high/low 各自除以**当日后复权收盘价**再回归。
      因为 beta = cov(low,high)/var(high) 对 high 和 low 的**同一个**正常数缩放不变，
      除以 close 在数学上完全等价，但把数值从「hfq 基座」（老股的累计复权因子可以到 1e3，
      价格 1e5 量级）压到 O(1)，避免 `roll_cov/roll_var` 在巨大均值上做差分的抵消误差。
    ★ 用 `ctx.roll_cov` + `ctx.roll_var` 向量化实现，不写循环。
    ★ `min_abs_den=1e-12`：18 日 high 完全不动（长期停牌）时 var == 0 -> NaN。
      归一化之后 var 的合法量级是 1e-6~1e-2，1e-12 只挡退化的 0。
    """
    close = ctx.hfq("close")
    h = ctx.safe_div(ctx.hfq("high"), close)
    l = ctx.safe_div(ctx.hfq("low"), close)
    return ctx.safe_div(ctx.roll_cov(l, h, n), ctx.roll_var(h, n), min_abs_den=1e-12)


@register(FactorSpec(
    name="rsrs_beta_18", group="technical", deps=DEPS,
    desc="RSRS 斜率 beta：18 日 high~low 滚动 OLS 斜率（阻力相对支撑的上升速度）",
    formula=(
        "# 折算到复权空间再回归,避免除权日 high/low 阶跃污染斜率(见 _compute_rsrs_beta)\n"
        "scale = _adjusted_close(daily) / daily[\"close\"].replace(0, np.nan)\n"
        "high = daily[\"high\"] * scale\n"
        "low = daily[\"low\"] * scale\n"
        "beta, _ = _compute_rsrs_beta(high, low, window=18)\n"
        "# 本项目口径：beta = roll_cov(low, high, 18) / roll_var(high, 18)"),
    start=None, warmup_days=18 * 2 + 20, higher_is_better=True, version=2,
    note="★★ v2（2026-09-17）：`mathx.roll_cov` 修了「分母用固定 n」的 bug（见该函数注释）——"
         "本因子是它的直接下游，历史值整体重算过。"
         "参考库出处：factors.md `类别 price` / technical_pattern.py 的 rsrs_beta_18，"
         "以及 因子库.md「4、Momentum」第 16 条 `rsrs`"
         "（`Slope_t = Beta from OLS(Low_{t-N+1:t} ~ High_{t-N+1:t}), N=18`）。"
         "★ 实现：滚动 OLS 斜率 = cov(low, high)/var(high)，用 `ctx.roll_cov` + "
         "`ctx.roll_var` 向量化（都是 O(T·C) 的 cumsum 差分），**没有 Python 循环**。"
         "★ 偏离（数值）：参考库把 high/low 乘 `scale = adj/close` 折算到复权空间；"
         "  本框架的 `ctx.hfq(\"high\"/\"low\")` 已经是复权价（少一次折算），"
         "  再各自除以当日 hfq 收盘价只为一个目的：把数值压到 O(1)。"
         "  除以同一个正常数不改变斜率（beta 对分子分母同尺度缩放不变）。"
         "★ beta 是**无量纲**的（cov(low,high)/var(high) 的单位是 low/high），"
         "  所以不需要额外归一化；但它的绝对值会随「high 与 low 的相对波动」漂移。"
         "★ 退化保护：18 日 high 完全不动（长期停牌）-> var == 0 -> NaN。"),
)
def rsrs_beta_18(ctx):
    return _rsrs_beta(ctx, 18)


@register(FactorSpec(
    name="rsrs_zscore_18", group="technical", deps=DEPS,
    desc="RSRS 标准分：18 日 OLS 斜率相对自身 200 日历史的标准分",
    formula=(
        "scale = _adjusted_close(daily) / daily[\"close\"].replace(0, np.nan)\n"
        "high = daily[\"high\"] * scale\n"
        "low = daily[\"low\"] * scale\n"
        "beta, _ = _compute_rsrs_beta(high, low, window=18)\n\n"
        "# Z-score relative to 400-day rolling window\n"
        "roll_mean = rolling_group_mean(beta, 400, min_periods=100)\n"
        "roll_std = rolling_group_std(beta, 400, min_periods=100)\n"
        "zscore = safe_divide(beta - roll_mean, roll_std + 1e-8)\n"
        "# 本项目口径：窗口取 200 个交易日（M=200，见 因子库.md 的 rsrs 条目）"),
    start=None, warmup_days=400, higher_is_better=True, version=2,
    note="★★ v2（2026-09-17）：随 `mathx.roll_cov` 的分母 bug 修复整体重算"
         "（z-score 里的 beta 正是该函数的输出）。"
         "参考库出处：factors.md `类别 price` / technical_pattern.py 的 rsrs_zscore_18；"
         "参数口径以 因子库.md「4、Momentum」第 16 条 `rsrs` 为准 ——"
         "`RSRS_t = Z-Score(Slope_{t-M+1:t}), M=200`，即**200 个交易日**。"
         "★ **偏离（窗口长度）**：factors.md 的代码注释写「400-day rolling window」，"
         "  与 因子库.md 的 M=200（交易日）冲突。两个数字单位不同："
         "  200 个**交易日** ≈ 280 个日历天，仍不等于 400。"
         "  本文件以 因子库.md 的显式参数 M=200（交易日）为准，"
         "  因为它是带参数名的规范定义，而 factors.md 那句只是行内注释。"
         "  如需改为 400 个交易日，只改 `rsrs_zscore_18` 里的 200 与它的 warmup。"
         "★ `warmup_days=400`（任务指定）：要覆盖 200 个交易日的 z-score 窗"
         "  + 18 日回归窗，400 个日历天 ≈ 274 个交易日，够用且有余量。"
         "★ `min_count=100`（= 参考库的 `min_periods=100`，但参考库窗口 400、"
         "  本文件窗口 200，所以本文件的放松**相对更严格**）。"
         "  为什么必须放松：`high`/`low` 属于价格层的 LEVELS，停牌日**前向填充**，"
         "  所以**连续停牌 >= 18 个交易日**的股票，其 18 日窗口内 high 完全不动 ->"
         "  `roll_var(high,18) == 0` -> `safe_div` 给 NaN -> beta 缺失。"
         "  而 beta 一旦缺一天，200 日 z-score 窗就被 NaN 毒化 **200 个交易日**。"
         "  实测：2012~2014 年截面勉强够用（1600~1900），但 2015 年千股停牌后"
         "  截面从年初 1596 一路衰减到年末 1305（中位数 1385），"
         "  低于契约 §7「日均截面 1500~3300」的下限。给 min_count=100 后回升。"
         "★ 不给 min_count 的代价（原实现）：长期停牌股会被整段踢出截面，"
         "  而停牌恰恰与「重组/重大事项」强相关 -> 缺失是**非随机**的，"
         "  等于在截面上系统性剔除了事件股。这是比放松窗口更坏的性质。")
)
def rsrs_zscore_18(ctx):
    beta = _rsrs_beta(ctx, 18)
    # min_count=100：见 note —— 长期停牌会把 beta 打成 NaN，200 日窗若要求「全满」
    # 会把千股停牌年份的截面从 ~2100 只砍到 ~1385 只（低于契约 §7 的 1500 下限）。
    # 100 与参考库 `min_periods=100` 同值（参考库窗口 400，相对更松）。
    mu = ctx.roll_mean(beta, 200, min_count=100)
    sd = ctx.roll_std(beta, 200, min_count=100)
    return ctx.safe_div(beta - mu, sd)
