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
    note='Wilder平滑，RSI=100*平均涨幅/(平均涨幅+平均跌幅)；全涨100、全跌0、全平50。',version=3,),
)
def rsi_14(ctx):
    return _rsi_close(ctx, 14)












# ══════════════════════════════════════════════════════════════════════
# 趋势强度：DMI / ADX / AROON / TRIX / ROC
# ══════════════════════════════════════════════════════════════════════













@register(FactorSpec(
    name="roc_12", version=2, group="technical", deps=DEPS,
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















# ══════════════════════════════════════════════════════════════════════
# 波动 / 价格位置：ATR 比率 / 威廉 %R / 随机指标 / 收盘位置
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="atr_14_ratio", version=2, group="technical", deps=DEPS,
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
    name="dpo_20", version=2, group="technical", deps=DEPS,
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
    start=None, warmup_days=18 * 2 + 20, higher_is_better=True, version=4,
    note='沿用本项目low~high的方向；直接回归后复权高低价。逐日除以不同收盘价并非共同常数缩放，会改变OLS斜率，已移除。'),
)
def rsrs_beta_18(ctx):
    return _rsrs_beta(ctx, 18)


