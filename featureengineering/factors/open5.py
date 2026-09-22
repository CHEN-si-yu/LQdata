"""开盘首段流动性因子（3 个）—— 全部日频产出、只主板、跟随 default_start。

数据源：`data/derived/open5/`（`fea/open5.py` 把 `stock_history_5min`
**6.87 亿行**预聚合成 4 个日频字段）+ 本文件。

## ★ 这三个因子是「执行 / 容量」口径，不是 alpha 声称

它们回答的是「**按 T+1 开盘价下的单，开盘那几分钟有没有足够的池子吃下它**」，
而不是「哪只票明天会涨」。`higher_is_better=True` 只是给模型侧的符号统一用的
（占用高 = 开盘流动性好 = 更容易成交）；**不要据此推断收益方向**。
下游要榨 alpha 请自己另立假设 —— 本文件不给。

## 为什么把「首 5 分钟占比」当信号：机制

全日成交额 ≠ 开盘可成交额。一笔按开盘价成交的单，面对的是**开盘那几分钟的流动性池**
（A 股还有 09:15–09:25 的集合竞价，成交量并进第一根 bar）。
项目原先只在执行层用「信号日成交额的 1%」限容量（`V63/analysis.py:272-273`），
那是**全天**口径 —— 对开盘时段的真实承载力是**系统性高估**。

实测（`open5` 层，冻结池内）：

    首 5 分钟占全日成交额的比例     p1 ≈ 2.3% · 中位 ≈ 8.1% · p99 ≈ 28%
    而首 5 分钟只占交易时间的        5 / 240 ≈ 2.1%

⇒ 开盘时段的成交密度是全天均值的 **~4 倍**（中位），且**个股之间差 12 倍以上** ——
这个横截面差异就是容量约束该用的那个自由度。

★ 另有**长期漂移**要处理：占比中位数 2010 ≈ 2.6% → 2018 ≈ 3.4% → 2022 ≈ 6.7% →
2025 ≈ 8.0%。所以除了当期占比，另给一个**过去 20 日时序分位**版本 —— 它把
「这只票自己的历史常态」当基准，漂移与个股规模都被约掉。

## 口径（与 `fea/open5.py` 一致，这里只重复最要紧的三条）

1. **「首 5 分钟」= 每个 (股票, 日) 的第一根 bar**。厂商时间戳是区间**右端**，
   实测最早一根恒为 `09:35`、每天恒为 48 根（2010–2026 全部年份实测如此）。
2. 只落**金额**（元），不碰 `vol` —— 绕开 `vol` 在 2025-11-28/12-01 从「手」翻「股」
   的单位陷阱（见 `fea/intraday.py` 的模块 docstring）。
3. 停牌 / 未上市 / 已退市当天**没有行** → 精确落格成 NaN，**不填 0**
   （填 0 等于把停牌记成「零成交」，会被 rank 排到最低）。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

# 日内层覆盖 2010 起；输出起点跟随 conf/config.yaml 的 default_start
O5_START = None

# warmup：滚动 20 日窗按 20×1.8+20 给（与 factors/intraday.py 同一换算）
W_D = 40
W_20 = 56


def _o5(ctx, field: str) -> np.ndarray:
    """开盘首段层字段（当日口径）。停牌 / 未落格 -> NaN，**不填 0**。"""
    return np.asarray(ctx.open5_field(field), dtype=np.float64)


def _share(ctx) -> np.ndarray:
    """首 5 分钟成交额 / 全日成交额。分母用 `min_abs_den` 兜底防爆。

    另加两道退化保护（正常情况下都是 no-op）：

    · `n_bars < 2` → NaN。只有一根 bar 的"整日"会让占比恒等于 1，
      在截面 rank 里被顶到最前，是个纯噪声的吸引子。
      实测 2010–2026 各抽样年份 `n_bars` **恒为 48**（p1 就是 48），所以这道闸
      在真实数据上从不触发 —— 它是防厂商改 bar 对齐方式的，不是过滤器。
    · `amt_day <= 0` → NaN。停牌半天只有零星成交的日子，占比没有意义。
    """
    a5 = _o5(ctx, "amt_open5")
    ad = _o5(ctx, "amt_day")
    nb = _o5(ctx, "n_bars")
    sh = ctx.safe_div(a5, ad, min_abs_den=1e3)          # 全日成交额 < 1000 元视为无效
    bad = ~np.isfinite(nb) | (nb < 2) | ~np.isfinite(ad) | (ad <= 0)
    return np.where(bad, np.nan, sh)


@register(FactorSpec(
    name="open5_amt_share",
    group="intraday",
    desc="开盘首 5 分钟成交额 / 全日成交额（执行容量口径，非 alpha）",
    formula="amt_open5 / amt_day",
    deps=("stock_history_5min",),
    start=O5_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=("★ 执行/容量口径：高 = 开盘时段流动性池厚，按开盘价成交的冲击更小、"
          "可容纳的资金更多。**不是**收益方向的声称。"
          "口径见 fea/open5.py：首 5 分钟 = 当日第一根 bar（厂商时间戳 09:35，区间右端）。"
          "实测全池中位 ≈ 8.1%（2026Q1）、且随年份单调上行（2010 约 2.6% → 2025 约 8.0%）——"
          "做时序比较请用本文件的 _pct20 版本，或只依赖逐日截面 rank。"
          "另：项目原有执行层只用「信号日全日成交额的 1%」限容量（V63/analysis.py:272-273），"
          "本因子是给它换成**开盘时段**口径的原料。"),
))
def open5_amt_share(ctx):
    return _share(ctx)

@register(FactorSpec(
    name="open5_amt_share_pct20",
    group="intraday",
    desc="开盘首 5 分钟成交额占比的 20 日时序分位（去掉长期漂移与个股规模）",
    formula="Ts_Rank(amt_open5/amt_day, 20)",
    deps=("stock_history_5min",),
    start=O5_START,
    warmup_days=W_20,
    higher_is_better=True,
    note=("★ 执行/容量口径，不是 alpha 声称。"
          "为什么要这一版：占比本身有**长期漂移**（2010 中位 2.6% → 2025 中位 8.0%，"
          "见 fea/open5.py 与 README 的交付记录），直接跨年比较会把「全市场都在变」"
          "误读成「这只票变了」。时序分位把每只票自己的 20 日历史当基准，"
          "漂移与个股规模一起约掉。"
          "`ctx.roll_rank` 是 WorldQuant 的 Ts_Rank（窗口最后一个值在窗口内的百分位，"
          "返回 [0,1]），min_count=10 = 20//2，与参考库 `_roll_sum(x,20,10)` 的 "
          "`min_periods` 约定一致。"
          "★ 与 open5_amt_share 高度相关（同一分子分母，只差一次时序排名）——"
          "下游若做冗余筛除，这两个应当**视作一簇**。"),
))
def open5_amt_share_pct20(ctx):
    return ctx.roll_rank(_share(ctx), 20, min_count=10)

@register(FactorSpec(
    name="open5_amt_log",
    group="intraday",
    desc="开盘首 5 分钟成交额的对数（绝对开盘容量，供执行层估单笔上限）",
    formula="log(amt_open5)",
    deps=("stock_history_5min",),
    start=O5_START,
    warmup_days=W_D,
    higher_is_better=True,
    note=("★ 执行/容量口径，不是 alpha 声称。"
          "与 share 版的区别：share 是**相对**结构（这只票自己开盘占全天的比重），"
          "log 版是**绝对**池子大小 —— 执行层要估「这笔单子最多下多少钱」"
          "用的是后者（`单笔上限 ≈ 开盘 5 分钟成交额 × 参与率`），"
          "而 share 版回答的是「同规模下谁的开盘更厚」。两者不可互相替代。"
          "取对数是因为原始金额跨 4~5 个数量级，截面 rank 虽然能吃掉量级、"
          "但下游若要做线性回归或分层，对数尺度更稳。"
          "停牌 / 未落格 -> NaN（不填 0；填 0 的 log 是 −inf）。"),
))
def open5_amt_log(ctx):
    a5 = _o5(ctx, "amt_open5")
    # 先挡掉非正数再取对数：safe_log 对 <=0 返回 NaN，但 min_abs_den 更直白
    return ctx.safe_log(np.where(a5 > 0, a5, np.nan))
