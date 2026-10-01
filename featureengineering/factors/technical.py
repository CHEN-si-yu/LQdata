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
    total = ag + al
    return np.where(np.isfinite(total) & (total == 0), 50., 100. * ctx.safe_div(ag, total))


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
    note='Wilder平滑，RSI=100*平均涨幅/(平均涨幅+平均跌幅)；全涨100、全跌0、全平50。',version=2,),
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
         "★ 复权：参考库用 pct_chg（交易所口径），本文件用 diff(hfq_close)，正常交易日等价。",version=2,),
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
    note='保持后复权典型价格乘成交量的既有加权流量口径；MFI=100*正流量/(正流量+负流量)，全正100、全负0、全平50。它是复权加权资金流，非实际人民币净流入。',version=2,),
)
def mfi_14(ctx):
    tp = (ctx.hfq("high") + ctx.hfq("low") + ctx.hfq("close")) / 3.0
    raw = tp * ctx.px("vol")
    chg = ctx.diff(tp, 1)
    pos = np.where(chg > 0, raw, 0.0)
    neg = np.where(chg < 0, raw, 0.0)
    p = ctx.roll_sum(pos, 14); n = ctx.roll_sum(neg, 14)
    total = p + n
    return np.where(np.isfinite(total) & (total == 0), 50., 100. * ctx.safe_div(p, total))




# ══════════════════════════════════════════════════════════════════════
# RSRS：阻力支撑相对强度（滚动 OLS 斜率 + 200 日 z-score）
# ══════════════════════════════════════════════════════════════════════

def _rsrs_beta(ctx, n=18):
    """18日 low 对 high 的OLS斜率；使用同一后复权价格空间。"""
    h = ctx.hfq("high")
    l = ctx.hfq("low")
    return ctx.safe_div(ctx.roll_cov(l, h, n), ctx.roll_var(h, n), min_abs_den=1e-12)


@register(FactorSpec(
    name="rsrs_beta_18", group="technical", deps=DEPS,
    desc='18日后复权low对high的滚动OLS斜率',
    formula=(
        'cov(hfq_low,hfq_high,18)/var(hfq_high,18)'),
    start=None, warmup_days=18 * 2 + 20, higher_is_better=True, version=3,
    note='沿用本项目low~high的方向；直接回归后复权高低价。逐日除以不同收盘价并非共同常数缩放，会改变OLS斜率，已移除。'),
)
def rsrs_beta_18(ctx):
    return _rsrs_beta(ctx, 18)


