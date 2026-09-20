"""因子耦合（20 个）—— 把两个**已有因子**的信息交叉起来，或自算微观结构比率。

═══════════════════════════════════════════════════════════════════════════
一、为什么「耦合」值得单独做一族
═══════════════════════════════════════════════════════════════════════════

单因子回答的是「这个维度上谁高谁低」，而很多**有效信号只在条件里出现**：
低波动本身没有方向性，但「低波动 + 强动量」是经典的防守型动量；
筹码获利盘本身弱，但「获利盘高 + 近期超跌」是另一回事。
参考库里这一族有 147 个，但其中约 34 个是 `rank(A) × rank(B)` 的**同构堆砌**
（与母因子共线），23 个是单调变换（`log`/`sqrt`/`zscore` —— 因为下游只用 rank 列，
单调变换**逐位相同、零信息**），本文件按「信息论上真的有增量」筛出 20 个。

═══════════════════════════════════════════════════════════════════════════
二、实现约定（本族特有，别照抄到别的家族）
═══════════════════════════════════════════════════════════════════════════

1. **父因子用 `ctx.load_factor(name)` 读**，它读的是兄弟因子的**产物文件**（精确落格，
   缺失 NaN），所以：
   · 必须把父因子名写进 `deps` —— 引擎靠它做**水位失效传播**
     （父因子更新 → 本因子自动重算，见 `fea/engine.py::_factor_watermark`）；
   · 父因子必须**先算完**：`main.py run` 会分两趟执行（第一趟非耦合、第二趟耦合），
     跑在父因子前面会静默拿到全 NaN。
2. **标准化一律用 `ctx.cs_zscore(x, mask=ctx.universe)`**（逐交易日的截面标准化）。
   ★ 绝不能用全样本均值/标准差 —— 那是前视（用了未来的分布）。契约 §2.3 也禁止在因子里做 rank。
3. **乘积型耦合的方向**：`z(A) × z(B)` 的符号语义是「两个维度同时高/同时低」。
   为了让「值大 = 信号强」可解释，需要**同向**的两个维度相乘；若某个维度天然是「低优」
   （如波动率、非流动性），先取负号再相乘（本文件统一用 `_nz()` 明确表达这件事）。
4. **乘法与减法各管一件事**：乘法 = 共振（两维度同时高），减法 = 错配（一个高一个低）。
   参考库里 `rank(A) − rank(B)` 型有几十个，本文件只保留 3 个分歧型。

## 覆盖与真值约定

- 耦合因子的非空率 = **两个父因子同时有值**的交集。若某个父因子覆盖率低
  （如 `margin_*` 只有两融标的 ≈50%、`limit_up_count_20` 稀疏），交集会更低 ——
  每个因子在 note 里给出由父因子推出的**预期覆盖率**。
- 缺失一律 NaN（不用 0 填充）：`cs_zscore` 的 NaN 会传播，乘法自然得到 NaN。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

GROUP = "coupling"

# 短窗耦合（父因子窗口 ≤ 20 交易日）：200 天够（含父因子 warmup + 冗余）
W_SHORT = 200
# 长窗耦合（父因子用到 60~120 日动量/波动）：给足，宁多勿少
W_LONG = 320


def _z(ctx, mat: np.ndarray) -> np.ndarray:
    """逐日截面 z-score（**只用当日截面**，绝无前视）。缺失保持 NaN。"""
    return np.asarray(ctx.cs_zscore(mat, mask=ctx.universe), dtype=np.float64)


def _f(ctx, name: str) -> np.ndarray:
    """读一个父因子的值（精确对齐到当前面板）。"""
    return np.asarray(ctx.load_factor(name), dtype=np.float64)


def _nz(x: np.ndarray) -> np.ndarray:
    """取负号 —— 把「低优」维度翻成「高优」，让乘积的符号可解释。"""
    return -x


def _mul(ctx, a: str, b: str, flip_a: bool = False, flip_b: bool = False) -> np.ndarray:
    """两个父因子的标准化乘积（共振信号）。"""
    za = _z(ctx, _f(ctx, a))
    zb = _z(ctx, _f(ctx, b))
    if flip_a:
        za = _nz(za)
    if flip_b:
        zb = _nz(zb)
    return za * zb


def _diff(ctx, a: str, b: str) -> np.ndarray:
    """两个父因子的标准化之差（分歧/错配信号）。"""
    return _z(ctx, _f(ctx, a)) - _z(ctx, _f(ctx, b))


# ══════════════════════════════════════════════════════════════════════════
# A. 价格 × 风险（3 个）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="cp_momentum_lowvol_20", group=GROUP,
    deps=("momentum_20", "idio_vol_60"),
    desc="防守型动量 = z(20日动量) × z(−特质波动)",
    formula="CP = z(momentum_20) × z(−idio_vol_60)",
    start=None, warmup_days=W_LONG, higher_is_better=True,
    note="★ 低波动本身没有方向（低波动异象的方向是「低波动跑赢」），乘以动量后表达的是"
         "「强势但波动小」的股票。父因子 `momentum_20` 与 `idio_vol_60` 实测在 2026 单年"
         "|RankIC| 分别 0.0424（负向）与 0.0752（负向），两者都强且相关性不高 → 乘积有增量。"
         "预期非空率 ≈ 两父因子交集（两者都是价格类，接近 100%）。",
))
def cp_momentum_lowvol_20(ctx):
    return _mul(ctx, "momentum_20", "idio_vol_60", flip_b=True)


@register(FactorSpec(
    name="cp_momentum_highvol_60", group=GROUP,
    deps=("momentum_60", "vol_120"),
    desc="进攻型动量 = z(60日动量) × z(120日波动)",
    formula="CP = z(momentum_60) × z(vol_120)",
    start=None, warmup_days=W_LONG, higher_is_better=True,
    note="与 `cp_momentum_lowvol_20` 是**对照组**（同样两个维度、方向相反）。"
         "两个都留着是有意的：A 股在不同区间对「高波动动量」的定价方向会翻，"
         "下游回归可以各自取值。预期非空率 ≈ 100%。",
))
def cp_momentum_highvol_60(ctx):
    return _mul(ctx, "momentum_60", "vol_120")




# ══════════════════════════════════════════════════════════════════════════
# B. 价值 × 质量 × 成长 × 资金结构（4 个）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="cp_value_quality", group=GROUP, deps=("bp", "roe_ttm"),
    desc="价值质量共振 = z(账面市值比 BP) × z(ROE TTM)",
    formula="CP = z(bp) × z(roe_ttm)",
    start=None, warmup_days=700, higher_is_better=True,
    note="经典的「便宜且好」（格雷厄姆式）。★ 两个父因子的 ac1 都 ≈0.999（季度才变一次），"
         "所以本因子**也是慢变量**：2026 单年只有约 4 个独立样本，IC 不可信；"
         "它的价值在于作为**风格暴露**进模型，而不是独立 alpha。",
))
def cp_value_quality(ctx):
    return _mul(ctx, "bp", "roe_ttm")


@register(FactorSpec(
    name="cp_value_momentum_div", group=GROUP, deps=("bp", "momentum_20"),
    desc="价值−动量分歧 = z(BP) − z(20日动量)",
    formula="CP = z(bp) − z(momentum_20)",
    start=None, warmup_days=W_LONG, higher_is_better=True,
    note="★ 用**差**而不是积：表达的是「便宜但近期没涨」（错配），与「又便宜又强」（共振）"
         "是两种不同的信号。参考库里这种 `rank(A) − rank(B)` 型有几十个，"
         "本文件只保留 3 个分歧型（价值-动量、两融-趋势、筹码-换手各一）。",
))
def cp_value_momentum_div(ctx):
    return _diff(ctx, "bp", "momentum_20")


@register(FactorSpec(
    name="cp_quality_momentum", group=GROUP, deps=("roe_ttm", "momentum_60"),
    desc="质量动量共振 = z(ROE TTM) × z(60日动量)",
    formula="CP = z(roe_ttm) × z(momentum_60)",
    start=None, warmup_days=700, higher_is_better=True,
    note="质量动量（QMJ 的 A 股版本）。慢变量（roe_ttm 季度才变） × 快变量（动量）→ "
         "整体换手由动量部分决定。预期非空率 ≈ 100%。",
))
def cp_quality_momentum(ctx):
    return _mul(ctx, "roe_ttm", "momentum_60")


@register(FactorSpec(
    name="cp_moneyflow_momentum_res", group=GROUP,
    deps=("mf_order_size_entropy", "momentum_20"),
    desc="资金流结构 × 动量 = z(订单规模熵) × z(20日动量)",
    formula="CP = z(mf_order_size_entropy) × z(momentum_20)",
    start=None, warmup_days=W_LONG, higher_is_better=False,
    note="`mf_order_size_entropy` 实测是 fundflow 族里最单调的因子（分层单调性 −0.99）："
         "熵高 = 订单规模分散 = 散户主导。乘以动量后表达「散户主导的上涨」。"
         "预期非空率 ≈ 100%（两张表都是全市场日频）。",
))
def cp_moneyflow_momentum_res(ctx):
    return _mul(ctx, "mf_order_size_entropy", "momentum_20")




@register(FactorSpec(
    name="cp_rsi_moneyflow_res", group=GROUP,
    deps=("rsi_14", "mf_net_inflow_ratio"),
    desc="RSI × 主力净流入 = z(RSI14) × z(净流入占比)",
    formula="CP = z(rsi_14) × z(mf_net_inflow_ratio)",
    start=None, warmup_days=W_SHORT, higher_is_better=False,
    note="技术面超买 × 资金面流入同时高 = 情绪与资金共振，A 股里通常是反向信号。"
         "⚠️ `mf_net_inflow_ratio` 的 ac1 ≈ 0.04（日频全换手），本因子换手也很高，"
         "必须按**成本后**收益复核。预期非空率 ≈ 100%。",
))
def cp_rsi_moneyflow_res(ctx):
    return _mul(ctx, "rsi_14", "mf_net_inflow_ratio")


@register(FactorSpec(
    name="cp_margin_trend_div", group=GROUP,
    deps=("margin_velocity", "trend_strength_60"),
    desc="两融加速 − 趋势强度 = z(融资余额速度) − z(60日趋势强度)",
    formula="CP = z(margin_velocity) − z(trend_strength_60)",
    start="2011-01-01", warmup_days=W_LONG, higher_is_better=True,
    note="分歧型：杠杆资金在加速、但价格趋势还没走出来（或反之）。"
         "★ 父因子 `margin_velocity` 是**滞后表**（stock_margin_detail 晚 1 个交易日）做出来的，"
         "它自己已经 `lag_grid(1)` 过了，所以本因子**不需要**再位移 —— 父因子的滞后处置会被继承，"
         "重复位移等于白丢一天信息。预期非空率 ≈ 50%（只有两融标的）。",
))
def cp_margin_trend_div(ctx):
    return _diff(ctx, "margin_velocity", "trend_strength_60")


@register(FactorSpec(
    name="cp_bigflow_margin_20", group=GROUP,
    deps=("mf_large_order_avg_price", "margin_velocity"),
    desc="大单成本 × 两融加速 = z(大单成交均价偏离) × z(融资余额速度)",
    formula="CP = z(mf_large_order_avg_price) × z(margin_velocity)",
    start="2011-01-01", warmup_days=W_LONG, higher_is_better=True,
    note="两个「聪明钱」维度：大单成交价（成本端）与杠杆资金加速（资金端）。"
         "预期非空率 ≈ 50%（受 margin 覆盖限制）。",
))
def cp_bigflow_margin_20(ctx):
    return _mul(ctx, "mf_large_order_avg_price", "margin_velocity")


# ══════════════════════════════════════════════════════════════════════════
# D. 筹码 × 价格/量（3 个）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="cp_chip_support_reversal_5", group=GROUP,
    deps=("chip_position", "short_term_reversal_5"),
    desc="筹码位置 × 短期反转 = z(收盘相对筹码分布位置) × z(5日反转)",
    formula="CP = z(chip_position) × z(short_term_reversal_5)",
    start="2018-01-02", warmup_days=W_SHORT, higher_is_better=True,
    note="筹码位置高 = 收盘价高于大部分持仓成本（获利盘多）；叠加超跌 → "
         "「获利盘被套」的反转机会。★ 起点受上游 `stock_cyq_chips` 限制（2018-01-02 起）。"
         "预期非空率 ≈ 90%（筹码层覆盖主板，与股票池一致）。",
))
def cp_chip_support_reversal_5(ctx):
    return _mul(ctx, "chip_position", "short_term_reversal_5")


@register(FactorSpec(
    name="cp_chip_turnover", group=GROUP, deps=("chip_concentration", "turnover_std_20"),
    desc="筹码集中 × 换手波动 = z(筹码集中度) × z(换手率波动)",
    formula="CP = z(chip_concentration) × z(turnover_std_20)",
    start="2018-01-02", warmup_days=W_SHORT, higher_is_better=False,
    note="筹码集中 + 换手剧烈 = 典型的「变盘前夜」。预期非空率 ≈ 90%。",
))
def cp_chip_turnover(ctx):
    return _mul(ctx, "chip_concentration", "turnover_std_20")






# ══════════════════════════════════════════════════════════════════════════
# F. 自算微观结构（5 个）—— 不依赖父因子，直接由价格/成交额派生
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="rv_term_structure", group=GROUP, deps=("stock_daily", "stock_adj_factor"),
    desc="波动率期限结构 = 20 日 RV / 60 日 RV（短端相对长端的高低）",
    formula="RV_n = std(ret(1), n);  TermStructure = RV_20 / RV_60",
    start=None, warmup_days=W_LONG, higher_is_better=False,
    note="参考库的 `rv_term_structure_slope` 用的是 5min 已实现波动；本实现用**日频收益的"
         "滚动标准差**做同一件事（日频口径更稳、且不受 5min 数据起点限制）。"
         "值 >1 = 短期波动放大（不安定期），<1 = 波动回落。"
         "⚠️ 与已删除的 `vol_20`/`vol_60` 不同：那两个是**水平**，这个是**比率**（不含水平信息）。",
))
def rv_term_structure(ctx):
    r = ctx.ret(1)
    short = ctx.roll_std(r, 20)
    long = ctx.roll_std(r, 60)
    return ctx.safe_div(short, long, min_abs_den=1e-12)


@register(FactorSpec(
    name="vol_of_rv", group=GROUP, deps=("stock_daily", "stock_adj_factor"),
    desc="波动的波动 = 20 日 RV 在 60 日窗口内的标准差（不稳定度）",
    formula="VoV = std( roll_std(ret(1), 20), 60 )",
    start=None, warmup_days=W_LONG, higher_is_better=False,
    note="表达「波动率本身有多不稳定」。★ 与已删除的 `vol_of_vol_20` 不同源："
         "那个是「波动水平的窗口比较」（与 vol_20 ρ=0.968，已删）；本口径是**嵌套滚动**"
         "（先算 20 日 RV，再取它 60 日的标准差），衡量的是波动率的**变化节奏**。",
))
def vol_of_rv(ctx):
    rv20 = ctx.roll_std(ctx.ret(1), 20)
    return ctx.roll_std(rv20, 60)




@register(FactorSpec(
    name="amihud_parkinson_ratio", group=GROUP, deps=("stock_daily", "stock_adj_factor"), version=2,
    desc="非流动性 / 波动率 = 20 日 Amihud ÷ 20 日 Parkinson 波动",
    formula="Amihud = roll_mean(|ret| / amount, 20);\n"
            "Parkinson = sqrt( roll_mean( ln(high/low)^2, 20) / (4 ln 2) );\n"
            "Ratio = Amihud / Parkinson",
    start=None, warmup_days=W_SHORT, higher_is_better=False,
    note="参考库同名因子的思路：**单位波动带来的价格冲击**。分母用 Parkinson 波动"
         "（只用日内高低价、对成交稀疏不敏感），比用收盘价标准差更干净。"
         "⚠️ 单位：`stock_daily.amount` 是元，Amihud 保留元级量纲（不乘 1e8）——"
         "这里只取**比率**，量纲自己约掉，乘不乘常数不影响截面排序。"
         "⚠️ high/low 用**未复权**价：Parkinson 是日内比值，除权日会有一个交易日的失真，"
         "但 20 日均值把它摊薄到可忽略（与参考库一致）。",
))
def amihud_parkinson_ratio(ctx):
    r = np.abs(ctx.ret(1))
    amt = ctx.px("amount")
    amihud = ctx.roll_mean(ctx.safe_div(r, amt, min_abs_den=1.0), 20)
    hi, lo = ctx.px("high"), ctx.px("low")
    with np.errstate(invalid="ignore", divide="ignore"):
        hl = np.log(hi / lo)
    park = np.sqrt(ctx.roll_mean(hl * hl, 20) / (4.0 * np.log(2.0)))
    return ctx.safe_div(amihud, park, min_abs_den=1e-12)


@register(FactorSpec(
    name="intraday_ret_share_20", group=GROUP, deps=("stock_daily", "stock_adj_factor"),
    desc="日内收益占比 = 20 日 Σ(日内收益) / 20 日 Σ(|隔夜| + |日内|)",
    formula="intraday(t) = hfq_close(t)/hfq_open(t) − 1;  overnight(t) = hfq_open(t)/hfq_close(t−1) − 1;\n"
            "Share = roll_sum(intraday, 20) / roll_sum(|overnight| + |intraday|, 20)",
    start=None, warmup_days=W_SHORT, higher_is_better=False,
    note="与 P0-2 家族的 `overnight_*` 是**同一现象的两个视角**（那边是绝对量、这边是占比）——"
         "保留两者是因为占比口径对停牌/低流动性股票更稳（分母自带缩放）。"
         "★ 用 `ctx.hfq`（后复权）算两条腿，停牌日用 `ctx.traded()` 掩码挡掉。",
))
def intraday_ret_share_20(ctx):
    o, c = ctx.hfq("open"), ctx.hfq("close")
    prev_c = ctx.shift(c, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        intr = np.asarray(c / o - 1.0, dtype=np.float64)
        over = np.asarray(o / prev_c - 1.0, dtype=np.float64)
    traded = np.asarray(ctx.traded(), dtype=bool)
    intr = np.where(traded, intr, np.nan)
    over = np.where(traded, over, np.nan)
    num = ctx.roll_sum(intr, 20)
    den = ctx.roll_sum(np.abs(over) + np.abs(intr), 20)
    return ctx.safe_div(num, den, min_abs_den=1e-12)
