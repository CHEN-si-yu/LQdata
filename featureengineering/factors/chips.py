"""筹码峰（chip distribution）因子 —— 24 个，全部读 `ctx.chip(...)`。

数据源：上游 `stock_cyq_chips`（6.24 亿行逐股筹码分布）由 `fea/chips.py`
预聚合成每 (股票, 日) 一行的摘要表（`data/derived/chips/`），本家族只读摘要表，
不再触碰原始表。字段语义见 `fea/chips.py` 的模块 docstring。

──────────────────────────────────────────────────────────────────────────
★ 口径一：筹码档位是**未复权价**，比较时两边必须同口径
──────────────────────────────────────────────────────────────────────────
实测（2019 全年，66 万行）：当 `below_close ≈ 0.5` 时 `close / p50 ≈ 1.01`，
且 `below_close` 与「用 5 个分位点插值出的 CDF(close)」相关系数 0.98 ——
证明摘要表里的 `mean/p10..p90` 与 `below_close` 都建立在**未复权 close** 上。

所以：凡是「现价 vs 筹码均价 / 分位价」的比值（avg_cost_premium、chip_position、
chip_profit_loss_ratio、chip_resistance_distance、chip_support_distance），
一律用 `ctx.px("close")`（**未复权**，停牌日自动前向填充），
**绝不**用 `ctx.hfq("close")`（后复权）—— 混用会在除权日造出 ±50% 的假信号
（实测 2019 年 adj_factor 量级 108~142，混用等于把比值放大两个数量级）。

⚠️ 参考库（`factors/chip*.py`）用的是 `_close_adj_basis(daily)`（复权口径），
那是因为**它们的** cyq 成本价是复权口径。本项目的筹码表是未复权口径，
照抄参考库的复权折算会把口径弄反，故全部改为未复权——这是本家族最重要的一条偏离。

⚠️ 本口径的固有代价：除权日未复权 close 会跳变（10 派 5 → close 掉 33%），
而筹码成本不变，当天比值会出现一次真实的数据跳变。这是未复权口径的物理属性
（上游摘要表本身就是这个口径，无法在因子层消除），下游按日横截面排序时
影响限于当天一格。选择它是因为：**同口径 > 平滑**，混口径会长期造出假信号。

──────────────────────────────────────────────────────────────────────────
★ 口径二：分位点映射（本表只有 5 个加权分位点）
──────────────────────────────────────────────────────────────────────────
参考库用 `cyq_perf.parquet` 的 cost_5/15/50/85/95pct。本摘要表提供
p10/p25/p50/p75/p90，按**最近分位**映射：

    cost_5pct  → p10        cost_15pct → p25
    cost_50pct → p50        cost_85pct → p75
    cost_95pct → p90

后果：宽度类因子（chip_concentration / chip_range_normalized /
chip_p90_p10_factor / chip_cost_kurtosis_20d）覆盖的是 **80% 筹码区间**而不是
参考库的 90%；两侧各窄 5%，数值略小，但**截面排序不变**（同一映射对所有股票一致）。

──────────────────────────────────────────────────────────────────────────
★ 口径三：起点与覆盖率
──────────────────────────────────────────────────────────────────────────
1. 起点硬约束 `start="2018-01-02"`（`stock_cyq_chips` 的真实起点，2018 年前无数据）。
2. 上游只覆盖**主板**（`stock_cyq_chips` 全表 3484 只，注意这是**跨年份累计**的只数，
   不是某一天的截面只数）。实测 2019：摘要表当日只有 2746 只、全市场 `stock_daily`
   当日 3795 只 ⇒ 本家族截面 ~2650 只 ≈ 筹码表自身的 **96.7%**（少数 ST/停牌/上市窗口外的
   股票被引擎挡掉）、全市场的 **70%** —— 所以本家族非空率天然低于其它家族，
   这是**数据源覆盖**造成的，不是实现缺陷。
3. `ctx.chip()` 是**精确落格**（缺则 NaN，不做前向填充）。因此：
   · 停牌日筹码行缺失 → 滚动窗口被毒化（契约的 NaN 策略），不是「沿用昨天」；
   · 2018 年开头的 20/5 日变化因子有一段 NaN 头（窗口要读到 2017 年），
     这是上游起点的必然结果，不做补偿。

──────────────────────────────────────────────────────────────────────────
★ 分母保护
──────────────────────────────────────────────────────────────────────────
实测 2019：mean 最小 0.71 元、p50 最小 0.65 元，均为正；但分位**之差**
（p75−p50、p90−p10 等）理论上可以到 0，一律走 `ctx.safe_div(..., min_abs_den=1e-6)`。
1e-6 元在 A 股价格尺度下经济意义为零，作地板不会改变任何有效样本的排序。

──────────────────────────────────────────────────────────────────────────
★ 与任务清单的偏离：清单列了 33 个名字（预算 24 个），本文件的选取规则是
   **「按清单顺序取前 24 个（a）能从摘要表精确算出、（b）不是已有因子的代数重复」**。
   据此排除的 9 个 —— 注意：只有前 6 条是真正的「重复/不可算」，后 3 条是**预算耗尽**
   造成的（它们是清单第 29/30/33 位，不是坏因子，主 Agent 若要扩到 27 个可直接加回）：

   1. `cost_distribution_width` —— **精确重复**。参考库 = (cost_95−cost_5)/cost_50，
      映射后 ≡ 摘要表的 width ≡ chip_concentration。实测 2019 逐日中位 ρ = **1.0000**。
   2. `chip_dispersion`         —— **精确重复**，参考库 chip_dispersion_width 与上一条
      同一公式（(p90−p10)/p50），实测 ρ(chip_dispersion, chip_concentration) = **1.0000**。
      （⚠️ 早先的草稿把这条写成「std 的绝对值受股价水平主导」，那是另一个量，已更正。）
   3. `cost_displacement`       —— **精确重复**。参考库 = (close−weight_avg)/weight_avg，
      与 avg_cost_premium 只差一个负号，实测 ρ = **1.0000**。
   4. `chip_mean_distance`      —— **精确重复**。参考库 = (close−mean)/close，
      是 avg_cost_premium 的单调变换（分母 close 换成 mean），实测 ρ = **1.0000**。
   5. `chip_median_distance`    —— **近似重复**。参考库 = (close−p50)/close；
      实测 ρ(·, avg_cost_premium) = **0.974**、ρ(·, chip_support_distance) = **0.979**。
   6. `chip_high_float_ratio`   —— **摘要表算不出**。参考库 = F(0.9×close)（成本低于
      0.9×现价的筹码占比），需要任意价格点的 CDF；摘要表只给 5 个分位点 + 上下方总占比。
      用 5 点线性插值近似后实测 ρ(·, winner_rate) = **0.907**、ρ(·, avg_cost_premium) = **0.922**
      —— 即便算出来也基本是获利盘因子的重述。★ 若要精确实现需 `fea/chips.py` 扩字段
      （如在 build_year 里补一列 below_90pct_close），属引擎扩展，见汇报。
   7. `chip_above_below_ratio`  —— **预算耗尽**（清单第 29 位），不是重复。参考库 =
      rank(−(close−cost_85)/(cost_15−close)) = **距离比**，与「质量比」winner_rate 无关：
      实测 ρ(·, winner_rate) = **0.59**、ρ(·, chip_profit_loss_ratio) = **0.05**，
      对全集 33 个候选的最大 |ρ| = 0.79。
      （⚠️ 早先的草稿声称「above_close ≡ 1 − below_close ⇒ 与 winner_rate 秩相关恒为 −1」，
      这句**错了**：恒等式本身成立（实测 Σ|below+above−1| = 1.7e-15），但参考库这个因子
      用的不是质量比而是距离比。已更正。）
   8. `chip_support_strength`   —— **预算耗尽**（清单第 30 位）。参考库 = [1/|cost_15−cost_5|] /
      |close−cost_15|（底部筹码密度 × 支撑距离倒数），用 p25−p10 与 |close−p25| 可精确还原。
      实测对全集最大 |ρ| = **0.724**（与 avg_cost_premium，负号），是清单里信息最独立但
      优先级最低的一个。（⚠️ 早先的草稿称「与 chip_cost_skew 逐字同式」，**错了**，已更正。）
   9. `chip_mode_mean_gap`      —— **预算耗尽**（清单第 33 位）。参考库 = |mode−mean|/mean，
      实测对全集最大 |ρ| = **0.70**（与 chip_range_normalized），ρ(·, chip_bimodality) = 0.26。
      参考库自己也是把它当 `chip_mode_mean_convergence`（其 5 日变化）用的，
      而「变化类」信息本文件已有 winner_rate_change_20d / chip_concentration_change_20d /
      chip_cost_convergence_20d 三个。（⚠️ 早先的草稿称「与 chip_bimodality 同一构造」，已更正。）

──────────────────────────────────────────────────────────────────────────
★ 已保留 24 个之间的实测共线性（2019 全年、逐日截面 Spearman 取中位）
──────────────────────────────────────────────────────────────────────────
    ρ ≥ 0.97： chip_resistance_distance ~ avg_cost_premium  −0.981
              winner_rate_ma5 ~ winner_rate                +0.979
              chip_cr3_factor ~ chip_peak_purity           +0.977
              chip_position ~ avg_cost_premium             +0.967
    ρ ≥ 0.90： chip_entropy_signal ~ chip_gini_factor      −0.912
    ρ ≥ 0.85： chip_range_normalized ~ chip_concentration  +0.865
   其余 12 对 < 0.8。**成因是共同的**：这些因子都在度量「现价 vs 成本分布」，横截面上
   由「股价相对成本水平」这一个共同因子驱动（涨得多的股票 winner_rate、price/mean、
   price/p25 同时高）。这是筹码家族的固有属性（参考库 39 个 Class-2 因子同样如此），
   不是实现缺陷 —— 下游做线性模型/等权合成前请先看这张表。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

# ── 起点：stock_cyq_chips 的真实起点，本家族的硬约束（见模块 docstring）
CHIP_START = "2018-01-02"

# ── 依赖：只读摘要表的因子依赖筹码层；用到现价的因子同时声明 stock_daily
#   （两者都进 deps，上游水位变化时才能正确触发回溯重算）
DEPS = ("stock_cyq_chips",)
DEPS_PX = ("stock_cyq_chips", "stock_daily")

# ── warmup_days：窗口 N 个交易日 → N×1.8+20（交易日→日历天 ×7/5 再留冗余）
W_FIELD = 20      # 无窗口（纯当日字段）：留 20 天冗余即可
W5 = 30           # 5 日窗口
W10 = 40          # 10 日窗口（二阶差分要回看 10 个交易日）
W20 = 60          # 20 日窗口

# ── 分母地板：价格类分母（元）。见模块 docstring「分母保护」
MIN_PRICE = 1e-6


# ══════════════════════════════════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════════════════════════════════
def _wr(ctx) -> np.ndarray:
    """获利盘比例 = below_close = 用**未复权** close 算的 F(close)，天然 ∈ [0,1]。"""
    x = np.asarray(ctx.chip("below_close"), dtype=np.float64)
    # 只挡浮点噪声（实测 below+above−1 ≤ 1.7e-15），保证 [0,1] 不变量传播到派生量
    return np.clip(x, 0.0, 1.0)


def _close(ctx) -> np.ndarray:
    """**未复权** close（与筹码档位同口径）。停牌日自动前向填充，但那天筹码行缺失，
    相乘/相除后仍是 NaN —— 不会拿停牌日的价格去配一个不存在的筹码分布。"""
    return np.asarray(ctx.px("close"), dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════
# 一、获利盘（winner_rate）家族 —— 5 个
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="winner_rate", group="chip", deps=DEPS,
    desc="获利盘比例 = 现价以下的筹码占比（未复权 close 口径）",
    formula='winner_rate = below_close; return cross_sectional_rank(-perf["winner_rate"])',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=False,
    note="★ 直接取摘要表的 below_close —— 它就是「用未复权 close 精确算出的 F(close)」"
         "（逐档比较 p <= close 后按归一化权重求和），与参考库 cyq_perf.winner_rate 同义。"
         "参考库 rank 取负（高获利盘=获利了结压力=反转信号）；本框架只出原始值，方向看此标注。"
         "已 clip 到 [0,1]（仅挡 1e-15 量级的浮点噪声）。",
))
def winner_rate(ctx):
    return _wr(ctx)






@register(FactorSpec(
    name="winner_rate_acceleration", group="chip", deps=DEPS,
    desc="获利盘比例的加速度（二阶差分：5 日变化的 5 日变化）",
    formula='wr_chg = wr.groupby(level="Code").transform(lambda s: s.diff(5)); '
            'accel = wr_chg.groupby(level="Code").transform(lambda s: s.diff(5)); '
            'return cross_sectional_rank(-accel)',
    start=CHIP_START, warmup_days=W10, higher_is_better=False,
    note="★ 名称冲突已定调：参考库有两个同名含义——Class1 的 `winner_rate_acceleration`"
         "是**一阶** 5 日变化（chip_deep.py），Class2 的 `chip_winner_rate_acceleration` 是"
         "**二阶**（「winner_rate_change_5d 的 5 日差分（二阶导数）」）。"
         "这里取**二阶**语义：一阶 5 日变化已在 winner_rate_change_20d 的同族里、且与"
         "Class1 的 winner_rate_change_5d 只是排序符号相反；二阶导才是独立信息（上涨加速=短期赶顶）。"
         "参考库 rank 取负（加速获利=赶顶信号）。要回看 10 个交易日，故 warmup 给 40。",
))
def winner_rate_acceleration(ctx):
    return ctx.diff(ctx.diff(_wr(ctx), 5), 5)


@register(FactorSpec(
    name="winner_rate_reversal_signal", group="chip", deps=DEPS,
    desc="获利盘极端反转信号 = −|获利盘 − 0.5|（50% 附近=多空平衡=排前）",
    formula='distance = -(cyq["winner_rate"] - 0.5).abs(); return cross_sectional_rank(distance)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=True,
    note="参考库 chip_deep.py 的原始形态，逐字保留。★ 这是本家族唯一**非单调**于"
         "获利盘的因子（V 形）：两端（>90% 极端获利 / <10% 极端深套）方向确定但反转压力最大，"
         "50% 附近是方向不确定的「不确定溢价」。值域 [−0.5, 0]。",
))
def winner_rate_reversal_signal(ctx):
    return -np.abs(_wr(ctx) - 0.5)


# ══════════════════════════════════════════════════════════════════════════
# 二、现价 vs 筹码成本（4 个；全部用未复权 close）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="avg_cost_premium", group="chip", deps=DEPS_PX,
    desc="平均成本溢价 = (现价 − 筹码加权均价) / 筹码加权均价",
    formula='premium = safe_divide(close_adj - weight_avg, weight_avg + 1e-10); '
            'premium = premium.clip(-1, 5); return cross_sectional_rank(premium)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=True,
    note="★ 口径偏离：参考库用 `_close_adj_basis(daily)`（复权），本项目**必须**用"
         "ctx.px('close')（未复权）—— 本摘要表的 mean 是未复权价（实测 close/p50≈1.01、"
         "close/mean≈1.07，见模块 docstring 的验证）。照抄参考库的复权折算会错两个数量级。"
         "★ 另一处偏离：**去掉**参考库的 `clip(-1, 5)` —— 契约规定 winsor 一律由引擎"
         "（1%/99% 截面 winsorize）统一做，因子内不做。单边下界 −1 本来也压不住什么，"
         "mean>0 时下溢只到 −1（close→0）。",
))
def avg_cost_premium(ctx):
    mean = ctx.chip("mean")
    close = _close(ctx)
    return ctx.safe_div(close - mean, mean, min_abs_den=MIN_PRICE)


@register(FactorSpec(
    name="chip_position", group="chip", deps=DEPS_PX,
    desc="现价在筹码分布中的位置 = (现价 − p10) / (p90 − p10)",
    formula='position = (close_adj - cost_5pct) / (cost_95pct - cost_5pct); '
            'return cross_sectional_rank(-position)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=False,
    note="★ 口径：close 用未复权（见 avg_cost_premium 的说明）；cost_5/95pct → p10/p90。"
         "★ 定义辨析（重要）：这是**价格空间**的相对位置，不是**筹码质量空间**的百分位。"
         "真正的「现价在筹码分布中的分位数」= F(close) = below_close = winner_rate 因子本身"
         "（已精确算好）；本因子是它的价格空间线性近似，二者秩相关高但不相等"
         "（价格空间对分布形态敏感，质量空间不敏感）。"
         "因此本因子取值可以越出 [0,1]（现价高于 p90 时 >1、低于 p10 时 <0），"
         "这是参考库的原式，不做截断——截断会把「突破筹码密集区」这一最有信息量的状态压平。"
         "参考库 rank 取负（位置高=接近上方套牢区=阻力大）。",
))
def chip_position(ctx):
    lo = ctx.chip("p10")
    hi = ctx.chip("p90")
    return ctx.safe_div(_close(ctx) - lo, hi - lo, min_abs_den=MIN_PRICE)




@register(FactorSpec(
    name="chip_resistance_distance", group="chip", deps=DEPS_PX,
    desc="上方压力距离 = p75 / 现价 − 1（现价距上方筹码密集带多远）",
    formula='result = cost_85pct / close_adj - 1; return cross_sectional_rank(-result)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=False,
    note="★ 参考库用 cost_85pct，本表取 p75（最近分位）。"
         "★ 口径：close 未复权。实测中位数 2018/2019/2020 = −0.021 / −0.041 / −0.033 —— "
         "**中位股票现价略高于 p75**（与 winner_rate 中位 0.79~0.87 一致），"
         "p25 分位 −0.24/−0.20/−0.15 才是「现价在成本带下方」的一侧；"
         "正值意味着现价已跌到成本带内（上方有解套抛压）。参考库 rank 取负。",
))
def chip_resistance_distance(ctx):
    return ctx.safe_div(ctx.chip("p75"), _close(ctx), min_abs_den=MIN_PRICE) - 1.0


@register(FactorSpec(
    name="chip_support_distance", group="chip", deps=DEPS_PX,
    desc="下方支撑距离 = 现价 / p25 − 1（现价距下方筹码密集带多远）",
    formula='result = close_adj / cost_15pct - 1; return cross_sectional_rank(result)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=True,
    note="★ 参考库用 cost_15pct，本表取 p25（最近分位，故支撑位比参考库**更低**、距离更大）。"
         "★ 口径：close 未复权。与 chip_resistance_distance 是一对镜像（同一构造、上下两侧）。",
))
def chip_support_distance(ctx):
    return ctx.safe_div(_close(ctx), ctx.chip("p25"), min_abs_den=MIN_PRICE) - 1.0


# ══════════════════════════════════════════════════════════════════════════
# 三、集中度 / 宽度（6 个）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="chip_concentration", group="chip", deps=DEPS,
    desc="筹码集中度（80% 筹码的相对宽度）= (p90 − p10) / p50",
    formula='spread = (perf["cost_95pct"] - perf["cost_5pct"]) / perf["cost_50pct"]; '
            'return cross_sectional_rank(-spread)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=False,
    note="★ 直接取摘要表的 width 字段（fea/chips.py 已算好，本表实测 width ≡ (p90−p10)/p50，"
         "最大误差 0.0，不是近似）。cost_5/95pct → p10/p90，即覆盖 80% 筹码区间"
         "（参考库是 90%），数值略小但截面排序同构。"
         "参考库 rank 取负（区间窄=筹码密集=排前）。"
         "★ 任务清单里的 cost_distribution_width 与本因子同式（只差 clip(0,5)），未重复实现。",
))
def chip_concentration(ctx):
    return ctx.chip("width")


@register(FactorSpec(
    name="chip_concentration_change_20d", group="chip", deps=DEPS,
    desc="筹码集中度的 20 日变化（正=区间收窄=筹码凝聚）",
    formula='concentration = -(cost_95pct - cost_5pct) / cost_50pct; '
            'chg = concentration.groupby(level="Code").transform(lambda s: s.diff(20)); '
            'return cross_sectional_rank(chg)',
    start=CHIP_START, warmup_days=W20, higher_is_better=True,
    note="参考库把 `concentration` 定义成**负**宽度再取 20 日变化，等价于"
         "「宽度收窄为正」；实现里直接对 width 取负差分（−ctx.diff(width,20)），语义相同。"
         "边际变化比绝对水平更有信息量（主力是否正在控筹）。2018 年开头约 20 个交易日 NaN。",
))
def chip_concentration_change_20d(ctx):
    return -ctx.diff(ctx.chip("width"), 20)


@register(FactorSpec(
    name="chip_range_normalized", group="chip", deps=DEPS,
    desc="归一化筹码区间（中间 50% 筹码的相对宽度）= (p75 − p25) / p50",
    formula='spread = (perf["cost_85pct"] - perf["cost_15pct"]) / perf["cost_50pct"]; '
            'return cross_sectional_rank(-spread)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=False,
    note="与 chip_concentration 的分位组合不同（这里 25/75，那里 10/90）："
         "一个度量核心 50% 筹码、一个度量 80% 筹码的宽度。cost_15/85pct → p25/p75。"
         "参考库 rank 取负（区间窄排前）。",
))
def chip_range_normalized(ctx):
    spread = ctx.chip("p75") - ctx.chip("p25")
    return ctx.safe_div(spread, ctx.chip("p50"), min_abs_den=MIN_PRICE)




@register(FactorSpec(
    name="chip_cost_convergence_20d", group="chip", deps=DEPS,
    desc="筹码成本收敛 = (p75 − p10)/p50 的 20 日变化的相反数（正=宽度收敛）",
    formula='width = safe_divide(cost_85pct - cost_5pct, cost_50pct); '
            'chg = width.groupby(level="Code").transform(lambda s: s.diff(20)); '
            'return cross_sectional_rank(-chg)',
    start=CHIP_START, warmup_days=W20, higher_is_better=True,
    note="★ 与 chip_concentration_change_20d 的区别在分位组合：本因子用 (cost_85−cost_5)/cost_50"
         "= (p75−p10)/p50（不对称：下侧取到 10%、上侧只到 75%），后者用 width=(p90−p10)/p50。"
         "两者都是「宽度 20 日变化」，秩相关偏高，但不对称口径对**上方**筹码堆积更敏感。"
         "参考库 rank 取负（收敛=正）。2018 年开头约 20 个交易日 NaN。",
))
def chip_cost_convergence_20d(ctx):
    width = ctx.safe_div(ctx.chip("p75") - ctx.chip("p10"), ctx.chip("p50"),
                         min_abs_den=MIN_PRICE)
    return -ctx.diff(width, 20)






# ══════════════════════════════════════════════════════════════════════════
# 四、分布形态（离散度 / 形态量，9 个）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="chip_cv_factor", group="chip", deps=DEPS,
    desc="筹码变异系数 = std / mean（低=相对离散度小=成本一致性强）",
    formula='s = _compute_chip_factor(..., "chip_cv"); return cross_sectional_rank(-s)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=False,
    note="★ 取负向（低 CV 排前）。用 std/mean 而非 std 原值：std 保留价格量纲、"
         "被股价水平主导（10 元股与 100 元股不可比），CV 消除价格水平后跨股可比。"
         "实测 2019 年 std/mean 中位 0.146、p99 0.52。与 chip_concentration 中位 ρ = 0.70"
         "（都度量分散度，但一个用 80% 分位区间、一个用整体二阶矩）。",
))
def chip_cv_factor(ctx):
    return ctx.safe_div(ctx.chip("std"), ctx.chip("mean"), min_abs_den=MIN_PRICE)


@register(FactorSpec(
    name="chip_gini_factor", group="chip", deps=DEPS,
    desc="筹码基尼系数（高=筹码集中在少数价位=价格锚定清晰）",
    formula='s = _compute_chip_factor(..., "chip_gini"); return cross_sectional_rank(s)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=True,
    note="摘要表的 gini 由 fea/chips.py 按加权基尼标准式算："
         "G = 2·Σ(i+1)·w_(i)/(n·Σw) − (n+1)/n（权重已归一化，Σw=1）。"
         "实测 2019：[0.22, 0.98]、中位数 0.866，恒非负 ✓（无负值、无 >1）。"
         "注：筹码按价格档**升序**排列后各档权重天然不等，故 gini 水平偏高，这是口径本身的"
         "性质（价格档密度在低价区更密），不是数据问题——截面排序仍可用。",
))
def chip_gini_factor(ctx):
    return ctx.chip("gini")










@register(FactorSpec(
    name="chip_p90_p10_factor", group="chip", deps=DEPS,
    desc="90% 筹码价格区间宽度 = p90 − p10（绝对价差）",
    formula='s = _compute_chip_factor(..., "chip_p90_p10"); return cross_sectional_rank(-s)',
    start=CHIP_START, warmup_days=W_FIELD, higher_is_better=False,
    note="★ 参考库的 chip_p90_p10 是**绝对价差**（未按中位价归一），本因子逐字照做。"
         "代价：它保留价格量纲，10 元股与 100 元股的数值不可直接比，截面排序里混有"
         "「股价水平」成分。之所以不换成归一化版本：归一化后的 (p90−p10)/p50 就是"
         "chip_concentration（width 字段），重复；参考库也确实是两个都收。"
         "若下游发现本因子与股价/市值高度共线，优先丢它、保留 chip_concentration。"
         "实测中位数 2018/2019/2020 = 2.11 / 1.62 / 1.56 元，p99 = 14.6 / 14.0 / 20.5 元，"
         "最大值 404 / 332 / 449 元（高价股的档宽天然大）—— 与 chip_concentration 中位 ρ 仅 0.47，"
         "因为后者已按 p50 归一。",
))
def chip_p90_p10_factor(ctx):
    return ctx.chip("p90") - ctx.chip("p10")
