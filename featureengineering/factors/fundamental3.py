"""财务明细·深化（16 个）—— 全部日频产出、只主板、跟随 `default_start`。

数据源：`stock_income` / `stock_balancesheet` / `stock_cashflow`（`ctx.ttm` / `ctx.point`）
+ `stock_financial_indicator`（`ctx.ind`）。

═══════════════════════════════════════════════════════════════════════════
一、★★★ 为什么这一族要长成这个样子（前两轮 prune 的教训）
═══════════════════════════════════════════════════════════════════════════

2026-09-15 与 2026-09-17 两轮「去糟粕」删掉的财务因子里，
**水平类**占了绝大多数：`asset_turnover` / `current_ratio` / `quick_ratio` /
`gpm_ttm` / `debt_asset_ratio` / `cash_profit_ratio` / `delta_roa` / `delta_roe` / `delta_gpm` /
`yoy_net_profit` / `yoy_roe` / `ncf_to_market` / `accruals_ratio` / `goodwill_to_assets` /
`receivable_turnover` / `fixed_asset_turnover` / `net_debt_ratio` / `equity_multiplier` …

共同特征：**ac1 ≈ 1.000**（一年只有约 4 个独立样本 —— 季报），换手≈0，
2026 单年的 IC 在统计上**不可信**。它们的死因不是「公式写错了」，
而是「**慢变量水平**在一年期的检验里没有统计功效」。

⇒ 本文件只做**三类**在结构上逃得开这个死因的东西：

  ① **per-unit 比率的同比**（参考库 Quality 的 `np_to_*_yoy` 族）：
     分子分母**同时动**，比值的同比是「效率的变化」，比任何水平都快。
     这与被删的 `delta_roe`（**慢水平的差分，还是慢**）有本质区别 ——
     被删的是 `X_t − X_{t−252}`，本文件做的是 `(NP/分母)_t / (NP/分母)_{t−4} − 1`，
     分母本身在变，所以它不是任何慢水平的线性函数。
  ② **盈利构成份额**（投资收益占比 / 公允价值占比 / 销售收现比）：
     分子分母同报告期、同日频，份额的**横截面离散度**远大于任何利润率水平。
  ③ **`ctx.ind()` 白名单字段**（26 个引擎认可的字段，**此前零个被消费**）：
     它们已经过 `fea/deriv.py` 的安全性筛选（时点比率 / 同期同比两类），
     本文件是它们的**首次启用**。

**两处沿用前一轮作者的明确排除决定**（不重建，理由记在此处免得下轮再问）：
· `minority_gain_ratio` —— `quality.py` 的分类学结论：少数股东损益占比属于
  「股权结构」而非「质量」维度，且 `n_income` 侧的口径在 A 股大量存在
  「归母为正、合计为负」的重组标的，比率会失真；
· `sell_exp_ratio` / `admin_exp_ratio` / `fin_exp_ratio` —— `quality.py` 已用
  `net_margin_ttm` 隐含了费用率的总效果，且 `fin_exp` 对 20%~34% 的公司**为负**
  （利息净收入），单独的费用率方向会翻转。
  ★ 本文件因此**不用 `fin_exp` 做分母**：`np_to_opex_yoy` 的费用口径只含
  `sell_exp + admin_exp + rd_exp`（三项恒为非负），这是与参考库 Quality #27
  「三费（含财务费用）」的**刻意偏离**，见该因子 note。

═══════════════════════════════════════════════════════════════════════════
二、两条必守的机制
═══════════════════════════════════════════════════════════════════════════

1. **每个因子必须声明 `fin_fields`**（`fea/engine.py`：不声明 = 触发**全字段**衍生层，
   单跑一个因子会白建 80 个字段的版本表，实测 6~25 s）。
2. **warmup_days = 700**（契约 §2）：TTM 4 季 + 同比再 4 季 + `ann_date` 最长滞后 15 个月。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

FIN_START = None
FIN_WARMUP = 700
# `stock_financial_indicator` 的口径：与已注册的 `interest_coverage` 一致
# （实测 `fin_exp_int_exp` 在 2012/2015 的非零率是 0.00%、2019 才 87% —— 早年是
#  「未披露」而不是「零」，不写起点会让头七年产出恒为 0/NaN 的假因子）
IND_START = "2019-05-01"

NP = "n_income_attr_p"       # 归母净利（Income 表）
REV = "revenue"              # 营业收入（★ 不用 total_revenue，见 CLAUDE.md 坑 4）
OP = "operate_profit"
TP = "total_profit"
NI = "n_income"
II = "invest_income"
FV = "fv_value_chg_gain"
OCF = "n_cashflow_act"
CSALE = "c_fr_sale_sg"
PAYROLL = "c_paid_to_for_empl"
INTEXP = "fin_exp_int_exp"
TAX = "income_tax"

TA = "total_assets"
EQ = "total_hldr_eqy_exc_min_int"
INV = "inventories"
FA = "fix_assets"
DTA = "defer_tax_assets"
ADVR = "adv_receipts"
PREPAY = "prepayment"

_DEP_IS = ("stock_income",)
_DEP_BS = ("stock_balancesheet",)
_DEP_CF = ("stock_cashflow",)
_DEP_IND = ("stock_financial_indicator",)


# ══════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════

def _yoy_ratio(ctx, num_ttm: str, den: str, den_mode: str = "point"):
    """per-unit 比率的同比：`r_t / |r_{t−4}| − 1`，其中 `r = 分子_TTM / 分母`。

    ★ 分母取**绝对值**：净利润可以跨零（亏损转盈利），
      直接用带符号的前期值会让「从 −1 亿变成 +0.5 亿」算成 −150% 而不是「改善」。
      这与 `factors/growth.py` 的既有做法一致。
    ★ 分母加上「比值本身」的地板（1e-6）：壳公司的分母科目可能精确为 0。
    返回**比值**、不返回排名（引擎统一做）。
    """
    if den_mode == "point":
        d0 = ctx.point(den)
        d4 = ctx.point(den, lag=4)
    else:
        d0 = ctx.ttm(den)
        d4 = ctx.lag_ttm(den, 4)
    r0 = ctx.safe_div(ctx.ttm(num_ttm), d0, 1e6)
    r4 = ctx.safe_div(ctx.lag_ttm(num_ttm, 4), d4, 1e6)
    return ctx.safe_div(r0, np.abs(r4), 1e-6) - 1.0


# ══════════════════════════════════════════════════════════════════════
# 1. 利润率与盈利构成（5 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="opm_npm_spread",
    group="quality",
    deps=_DEP_IS,
    desc="营业利润率 − 净利率（营业利润到归母之间的损耗）",
    formula="spread = operate_profit/revenue - n_income_attr_p/revenue",
    fin_fields=(OP, NP, REV),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=False,
    note=("**本文件新造**。★ 为什么用一个**价差**而不是两个水平："
          "`营业利润率` 与 `净利率` 两个水平各自都是慢变量，"
          "但它们的**差**在截面上的离散度大得多、时序也更快 ——"
          "差刻画的是「营业利润在非经常性损益、税、少数股东损益这一路上损耗了多少」，"
          "而这一路的构成**每年都在变**（减值、投资收益、税率优惠），"
          "所以价差不是慢变量的线性重标定。"
          "被删的 `gross_margin_change` / `net_margin_change` 是**时序差分**（还是慢变量），"
          "本因子是**同日截面内的两个水平之差**，机制不同。"
          "方向取负：损耗越大，盈利质量越差。"),
))
def opm_npm_spread(ctx):
    r = ctx.ttm(REV)
    return ctx.safe_div(ctx.ttm(OP), r, 1e6) - ctx.safe_div(ctx.ttm(NP), r, 1e6)


@register(FactorSpec(
    name="invest_income_share",
    group="quality",
    deps=_DEP_IS,
    desc="投资收益占利润总额的比重（非主业盈利依赖度）",
    formula="share = invest_income_TTM / total_profit_TTM",
    fin_fields=(II, TP),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=False,
    note=("**本文件新造**（参考库 Quality #10 `quality_composite` 用到 TotalProfit 族的比率，"
          "但没有单独发过投资收益占比）。"
          "**为什么是份额而不是水平**：份额的分母（利润总额）随公司规模变化，"
          "分子分母同报告期 ⇒ 截面离散度大、时序变化快，逃开「慢水平」的死因。"
          "**实测可用性**：`invest_income` 的非零公司占比从 2012 的 65% 升到 2022 的 94%，"
          "全历史可用，不是近年才有的字段。"
          "★ 分母 `total_profit` 可以为负（亏损），此时 `safe_div` 的地板（1e6）"
          "在金额上等价于「利润总额绝对值小于 1 百万元」⇒ NaN，"
          "语义正确（小额度的比重没有意义）。方向取负：越依赖投资收益，主业质量越差。"),
))
def invest_income_share(ctx):
    return ctx.safe_div(ctx.ttm(II), ctx.ttm(TP), 1e6)


@register(FactorSpec(
    name="fv_gain_share",
    group="quality",
    deps=_DEP_IS,
    desc="公允价值变动收益占利润总额的比重（纸面利润依赖度）",
    formula="share = fv_value_chg_gain_TTM / total_profit_TTM",
    fin_fields=(FV, TP),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=False,
    note=("**本文件新造**。公允价值变动是**未实现、会反转、不产生现金流**的收益 ——"
          "占比高说明当期利润里有一块「纸面富贵」，是典型的盈利质量扣分项，"
          "而全库此前没有任何因子刻画它。"
          "⚠ **结构性零膨胀（实测，务必先知）**：`fv_value_chg_gain` 在原表上就有"
          "**59.6% 的行恰好为 0**（另 15.0% 为负、25.4% 为正），"
          "TTM 汇总后本因子 2012 年的零值占比 **72.4%** —— "
          "**超过 `fea/eval.py` 的 `⚠SPARSE` 阈值（>70%），一定会被标 ⚠SPARSE**。"
          "沙箱自检已实测确认。**保留的理由**：它是一个语义完全成立的量"
          "（「利润里有多少是纸面重估」），零膨胀来自 A 股绝大多数公司不持有"
          "大额以公允价值计量的资产这一**真实事实**，而不是口径错误；"
          "同库已有 12 个因子带 ⚠SPARSE 仍在册。"
          "⚠ 若 eval 给出的 RankIC 也同时低于 0.005（→ 同时带 ⚠NOISE），"
          "则应删掉，不必保留。方向取负。"),
))
def fv_gain_share(ctx):
    return ctx.safe_div(ctx.ttm(FV), ctx.ttm(TP), 1e6)


@register(FactorSpec(
    name="share_issuance_yoy",
    group="growth",
    deps=_DEP_BS,
    desc="总股本同比（股本扩张 = 增发/送转/股权激励的摊薄幅度）",
    formula="growth = total_share(T) / total_share(T-4报告期) - 1",
    fin_fields=("total_share",),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=False,
    note=("**本文件新造**。用 `total_share`（资产负债表时点科目，"
          "在 `fea/deriv.py` 的 `POSITIVE_ONLY` 里 ⇒ 恒为正，比值天然安全）。"
          "**为什么必须做**：全库此前**没有任何因子**刻画股本的**扩张** ——"
          "`log_mv` / `free_share_ratio` 是水平、`bps_yoy` 是**扣除**股本扩张之后"
          "的结果，而本因子正是那个被扣掉的东西。"
          "两者构成一组完整的分解：`净资产同比 ≈ 每股净资产同比 + 股本同比`。"
          "经济含义：股本同比高 ⇒ 增发摊薄（老股东权益被稀释）或高送转；"
          "低/负 ⇒ 回购注销（近年才常见）。"
          "⚠ **送转是机械事件不是估值信号**：10 送 10 会让股本翻倍但价值不变，"
          "本因子会把它记成 +100% 的「扩张」。这是**已知的口径局限**，"
          "写在 note 里供下游取舍 —— 若要剔除送转，需要除权事件的复权因子，"
          "那属于另一族（`stock_adj_factor_changes`）的范畴。"
          "★ `total_share` 是**真实日频序列**（实测每股每年 5~15 个不同取值，"
          "变动日与 `adj_factor` 变动日同步），不是一年四次的季报值 ——"
          "所以本因子比同族 TTM 比率的活跃度高。方向取负（摊薄越大越差）。"),
))
def share_issuance_yoy(ctx):
    s0 = ctx.point("total_share")
    s4 = ctx.point("total_share", lag=4)
    return ctx.safe_div(s0, s4, 1e-6) - 1.0


@register(FactorSpec(
    name="cash_sales_ratio",
    group="quality",
    deps=(*_DEP_IS, *_DEP_CF),
    desc="销售收现比 = 销售商品收到的现金(TTM) / 营业收入(TTM)",
    formula="ratio = c_fr_sale_sg_TTM / revenue_TTM",
    fin_fields=(CSALE, REV),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=True,
    note=("**本文件新造**。★ 与已注册的 `ocf_to_revenue`（经营现金流/收入）**不是**同一个量："
          "`ocf` 已经扣掉了所有现金成本（采购、工资、税），"
          "而 `c_fr_sale_sg` 是**毛收款**（只扣增值税）——"
          "两个比值的中位数实测分别是 0.13 与 **1.0005**，差一个量级，"
          "刻画的是完全不同的环节：本因子测的是「**卖出去的钱收回来了没有**」"
          "（<1 = 应收堆积 / 渠道压货），`ocf_to_revenue` 测的是「最终剩了多少现金」。"
          "★ 为什么要做这个因子：中位数恒为 1.0005 ⇒ **水平本身无信息，"
          "信号全在离散度上**（左尾就是坏账风险）。这类「以 1 为基准的偏离度」因子"
          "在截面上是天然的双边分布，与所有利润率类因子都不同。"),
))
def cash_sales_ratio(ctx):
    return ctx.safe_div(ctx.ttm(CSALE), ctx.ttm(REV), 1e6)


# ══════════════════════════════════════════════════════════════════════
# 2. 现金流覆盖与营运资本（2 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="cfcr",
    group="quality",
    deps=(*_DEP_CF, *_DEP_IS),
    desc="现金流利息保障倍数 = 经营现金流(TTM) / 利息支出(TTM)",
    formula="CFCR = NetOperateCashFlow_TTM / InterestExpense_TTM",
    fin_fields=(OCF, INTEXP),
    start=IND_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=True,
    note=("参考库 `因子库.md` 5、Quality #38 `cfcr`，公式逐字。"
          "★ 与已注册的 `interest_coverage`（`icr` = **EBIT**_TTM / 利息支出）的区别是**分子**："
          "EBIT 是**权责发生制**的利润，经营现金流是**收付实现制**的现金。"
          "对有大量应计项的公司（应收激增、存货堆积），两者会**显著背离** ——"
          "EBIT 保障倍数只 1.2 倍但现金流保障倍数 4 倍的公司，偿债能力完全不同。"
          "★★ **必须 `start=\"2019-05-01\"`**（与 `interest_coverage` 完全一致的起点）："
          "实测 `fin_exp_int_exp` 在 2012/2015 的非零率是 **0.00%**、2019 才 87% ——"
          "早年是「未披露」而不是「零」，不写起点会让头七年产出恒 0（= 因子退化）。"
          "分母由 `safe_div` 的地板（1e6）保护；公司没有利息支出时给 NaN 而非 inf。"),
))
def cfcr(ctx):
    return ctx.safe_div(ctx.ttm(OCF), ctx.ttm(INTEXP), 1e6)


@register(FactorSpec(
    name="ar_ap_to_revenue",
    group="quality",
    deps=(*_DEP_BS, *_DEP_IS),
    desc="净预收款占收入比 = (预收款项 − 预付款项) / 营业收入(TTM)",
    formula="Ratio = (AdvanceReceipts - AdvancePayment) / OperatingRevenue",
    fin_fields=(ADVR, PREPAY, REV),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=True,
    note=("参考库 `因子库.md` 5、Quality #11，公式逐字。"
          "**经济含义**：预收款 = 客户先付钱（**下游议价力 / 订单储备**），"
          "预付款 = 先付给上游（**被上游占用**）。两者之差是**净占款能力**。"
          "★ 与已注册的 `cash_conversion_cycle` 的区别：那个测的是"
          "「存货+应收−应付」的**周转天数**（营运效率），"
          "本因子测的是预收/预付这一对**合同负债**（议价地位）——"
          "既不与周转天数共线，也不与任何利润率水平共线。"
          "★★ **零额外 TTM 机械**：两个分子科目都是**纯时点**（资产负债表），"
          "只有分母是 TTM；所以本因子的活跃度高于纯 TTM 比率。"
          "`prepayment` / `adv_receipts` 这两个字段**此前零个因子消费过**。"),
))
def ar_ap_to_revenue(ctx):
    return ctx.safe_div(ctx.point(ADVR) - ctx.point(PREPAY), ctx.ttm(REV), 1e6)


# ══════════════════════════════════════════════════════════════════════
# 3. 债务结构（2 个）—— ★ `ctx.ind()` 白名单的**首次启用**
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="ind_currentdebt_to_debt",
    group="risk",
    deps=_DEP_IND,
    desc="短期债务占总债务的比重（债务期限结构 / 展期风险）",
    formula="ctx.ind('currentdebt_to_debt')   # 供应商时点比率",
    fin_fields=("currentdebt_to_debt",),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=False,
    note=("★★ **本因子是 `ctx.ind()` 白名单字段的首次启用**（此前 26 个字段零个被消费）。"
          "字段本身是供应商算好的**时点比率**（`end_date` 上的瞬时量，"
          "不是累计 YTD，所以能当日频用 —— 见 `factors/DEVELOPING.md` §5 的白名单）。"
          "**为什么它是独立的维度**：已删的 `debt_asset_ratio`（负债/资产）是**杠杆水平**；"
          "本因子是**期限结构**（短期债务/总债务）——"
          "同样的杠杆率，全部是短期借款（一年内要还）与全部是长期债券，"
          "风险完全不同。两者在数学上**不是**单调变换关系。"
          "**实测覆盖**：中位 84%、q05 33%、q95 99%、100% 非空 ⇒ 填充良好且不零膨胀。"
          "方向取负（短期债务占比越高，展期风险越大）。"),
))
def ind_currentdebt_to_debt(ctx):
    return ctx.ind("currentdebt_to_debt")


@register(FactorSpec(
    name="ind_int_to_talcap",
    group="risk",
    deps=_DEP_IND,
    desc="有息负债占总资本的比重（融资性杠杆，剔除经营性负债）",
    formula="ctx.ind('int_to_talcap')   # 供应商时点比率",
    fin_fields=("int_to_talcap",),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=False,
    note=("★ `ctx.ind()` 白名单字段（见 `ind_currentdebt_to_debt` 的 note）。"
          "**为什么它是独立的维度**：已删的 `debt_asset_ratio` 把"
          "**应付账款、预收款、应交税费**这些**经营性负债**也算进了杠杆 ——"
          "而经营性负债恰恰是「占用上下游资金」的**能力**（越多越强势）。"
          "本因子只数**有息负债**（借款、债券），把「融资性杠杆」从经营性负债里分出来，"
          "这是 D/A 口径做不到的切分。"
          "⚠ **有一部分公司恰好为 0**（完全没有有息负债）—— 那是**真实值**（无杠杆），"
          "不是缺失，**不要**做 winsor 或抹成 NaN；引擎的 1%/99% winsor 会保护排名列。"
          "（实测 2012 年该因子零值占比 16.1%，2019 年后比例更低。）"
          "方向取负。"),
))
def ind_int_to_talcap(ctx):
    return ctx.ind("int_to_talcap")


# ══════════════════════════════════════════════════════════════════════
# 4. per-unit 比率的同比（5 个）—— 参考库 Quality #8 / #41 / #42 / #28 族
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="np_to_inventory_yoy",
    group="growth",
    deps=(*_DEP_IS, *_DEP_BS),
    desc="单位存货创利同比 = (净利_TTM / 存货) 的同比",
    formula="Ratio = NetProfit_Q / Inventories;  Growth = Ratio_t / Ratio_{t-4} - 1",
    fin_fields=(NP, INV),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=True,
    note=("参考库 `因子库.md` 5、Quality #8（per-unit 比率同比族）。"
          "★ **刻意偏离**：参考库分子用**单季**净利（`NetProfit_Q`），"
          "本项目的 `fea/deriv.py` **只暴露 TTM**（没有单季访问器，"
          "见 `factors/DEVELOPING.md` §3.2 的字段表），故用 `n_income_attr_p_TTM`。"
          "语义从「单季的存货创利效率」变成「**滚动一年的**存货创利效率」，"
          "少了季节性、也更平滑 —— 在单年 IC 检验里这**是好事**。"
          "★ 分母 `inventories` **不在** `POSITIVE_ONLY` 里（`fea/deriv.py`）——"
          "银行/券商没有存货，该字段是精确 0 或 NaN ⇒ 由 `safe_div` 的 1e6 地板"
          "给 NaN，**这正是期望行为**（金融股不该有这个因子）。"
          "★ 为什么这是 growth 而不是 quality：它测的是**效率的变化率**，"
          "而 `inventory_turnover`（在册）测的是**水平**。"),
))
def np_to_inventory_yoy(ctx):
    return _yoy_ratio(ctx, NP, INV, "point")


@register(FactorSpec(
    name="np_to_fixed_assets_yoy",
    group="growth",
    deps=(*_DEP_IS, *_DEP_BS),
    desc="单位固定资产创利同比 = (净利_TTM / 固定资产) 的同比",
    formula="Ratio = NetProfit_Q / FixedAssets;  Growth = Ratio_t / Ratio_{t-4} - 1",
    fin_fields=(NP, FA),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=True,
    note=("参考库 `因子库.md` 5、Quality #41。"
          "★ 与 `np_to_inventory_yoy` 同族、同偏离（单季→TTM），"
          "两处**共用同一个地板（1e6 元）**，保证两者的量级可比。"
          "**为什么固定资产这一支值得单独发**：重资产行业的产能利用率变化"
          "是盈利周期最直接的度量 —— 单位固定资产创利上升 = 产能被更充分利用"
          "（或刚做完减值、分母变干净）。"
          "与已删的 `fixed_asset_turnover`（收入/固定资产的**水平**）不同："
          "那个是效率水平，本因子是**效率的同比变化**（分子换成净利、取同比）。"),
))
def np_to_fixed_assets_yoy(ctx):
    return _yoy_ratio(ctx, NP, FA, "point")


@register(FactorSpec(
    name="np_to_salary_yoy",
    group="growth",
    deps=(*_DEP_IS, *_DEP_CF),
    desc="单位薪酬创利同比 = (净利_TTM / 支付给职工的现金_TTM) 的同比",
    formula="Ratio = NetProfit_TTM / StaffBehalfPaid_TTM;  Growth = Ratio_t / Ratio_{t-4} - 1",
    fin_fields=(NP, PAYROLL),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=True,
    note=("参考库 `因子库.md` 5、Quality #42。"
          "★ **本因子无需任何偏离**：参考库原文就是 `NetProfit_TTM / StaffBehalfPaid_TTM`"
          "（**TTM/TTM**），与 `np_to_inventory_yoy` / `np_to_fixed_assets_yoy`"
          "那两条「单季→TTM」的偏离不同。"
          "**经济含义**：每元薪酬产出多少利润 = **人力投入的运营杠杆**。"
          "这个比率上升可以由两头驱动：收入增长摊薄了固定人力成本"
          "（经营杠杆释放），或裁员降本。两者的后续走势完全不同，"
          "但作为「效率改善」的信号方向一致。"
          "★ `c_paid_to_for_empl` 由现金流量表提供，实测 2012 起 100% 非零。"),
))
def np_to_salary_yoy(ctx):
    return _yoy_ratio(ctx, NP, PAYROLL, "ttm")


@register(FactorSpec(
    name="np_to_deferred_tax_yoy",
    group="growth",
    deps=(*_DEP_IS, *_DEP_BS),
    desc="单位递延所得税资产创利同比 = (净利_TTM / 递延所得税资产) 的同比",
    formula="Ratio = NetProfit_Q / DeferredTaxAssets;  Growth = Ratio_t / Ratio_{t-4} - 1",
    fin_fields=(NP, DTA),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=True,
    note=("参考库 `因子库.md` 5、Quality #28。"
          "★ 同族偏离（单季→TTM），见 `np_to_inventory_yoy` 的 note。"
          "**为什么递延所得税资产这一支有独立信息**：递延所得税资产主要是"
          "**可抵扣暂时性差异与可结转亏损**的累积 —— "
          "它是「税务当局尚未认可的会计利润」最干净的单科目代理。"
          "企业只有在**预期未来能盈利**时才会确认这笔资产（否则要计提减值），"
          "所以它的相对规模变化携带了管理层对自身盈利前景的判断。"
          "此前**零个因子**用过这个字段。"),
))
def np_to_deferred_tax_yoy(ctx):
    return _yoy_ratio(ctx, NP, DTA, "point")


@register(FactorSpec(
    name="np_to_opex_yoy",
    group="growth",
    deps=_DEP_IS,
    desc="单位经营性费用创利同比 = (净利_TTM / (销售+管理+研发费用)_TTM) 的同比",
    formula="Opex = sell_exp + admin_exp + rd_exp (TTM)\n"
            "Ratio = NetProfit_TTM / Opex;  Growth = Ratio_t / Ratio_{t-4} - 1",
    fin_fields=(NP, "sell_exp", "admin_exp", "rd_exp"),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=True,
    note=("参考库 `因子库.md` 5、Quality #27 的 per-unit 族。"
          "★★ **刻意偏离：分母不含 `fin_exp`（财务费用）**（这是与前一轮作者的明确决定一致）。"
          "参考库的「三费」= 销售 + 管理 + **财务**费用，但本项目实测"
          "`fin_exp` 对 **20%~34% 的公司为负**（利息净收入大于利息支出），"
          "含它会让分母**跨零**⇒ 比值的符号静默翻转（"
          "`quality.py` 的排除清单里写明了这一条）。"
          "本实现改用 **销售 + 管理 + 研发**（三项恒为非负），"
          "语义从「三费」变成「**经营性费用**」（研发本就该算进经营费用，"
          "参考库把研发另计是它那一版的口径）。"
          "★ 与 `opm_ttm` / `opm_npm_spread` 的关系：那两个是**收入为分母**的利润率"
          "（测定价能力），本因子是**费用为分母**的产出率（测费用效率），"
          "分母的物理量不同、且本因子取同比。"),
))
def np_to_opex_yoy(ctx):
    opex = ctx.ttm("sell_exp") + ctx.ttm("admin_exp") + ctx.ttm("rd_exp")
    r0 = ctx.safe_div(ctx.ttm(NP), opex, 1e6)
    opex4 = ctx.lag_ttm("sell_exp", 4) + ctx.lag_ttm("admin_exp", 4) + ctx.lag_ttm("rd_exp", 4)
    r4 = ctx.safe_div(ctx.lag_ttm(NP, 4), opex4, 1e6)
    return ctx.safe_div(r0, np.abs(r4), 1e-6) - 1.0


# ══════════════════════════════════════════════════════════════════════
# 5. 供应商同比字段（2 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="ind_dt_netprofit_yoy",
    group="growth",
    deps=_DEP_IND,
    desc="扣非净利润同比（供应商同期同比字段，季节性自动抵消）",
    formula="ctx.ind('dt_netprofit_yoy')",
    fin_fields=("dt_netprofit_yoy",),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=True,
    note=("★ `ctx.ind()` 白名单的**同期同比**类字段（分子分母都是同期 YTD，"
          "季节性自动抵消，所以能当日频用 —— `factors/DEVELOPING.md` §5）。"
          "**这是参考库当年**拿不到**的字段**：参考库的 `earnings_cut_to_market` note 里"
          "明确写过它只能用「归母净利」代理扣非（「扣非」= 扣除非经常性损益），"
          "而供应商直接给了 `dt_netprofit_yoy`。"
          "★ 与被删的 `yoy_net_profit`（归母净利同比）的区别：**分子不同**。"
          "扣非剔除了政府补贴、资产处置、公允价值变动、投资收益等一次性项目 ——"
          "同一个「净利润同比 +30%」，全部来自卖楼 vs 全部来自主业，"
          "含义完全相反。而已删的 `earnings_cut_to_market` 是因为"
          "「用市值做分母」被删的，不是因为「扣非」这个想法。"),
))
def ind_dt_netprofit_yoy(ctx):
    return ctx.ind("dt_netprofit_yoy")


@register(FactorSpec(
    name="ind_bps_yoy",
    group="growth",
    deps=_DEP_IND,
    desc="每股净资产同比（股东权益的**每股**累积速度）",
    formula="ctx.ind('bps_yoy')",
    fin_fields=("bps_yoy",),
    start=FIN_START,
    warmup_days=FIN_WARMUP,
    higher_is_better=True,
    note=("★ `ctx.ind()` 的同期同比类字段。"
          "**为什么「每股」这个框定是本质的**：被删的 `yoy_equity`（净资产同比）"
          "**不扣除股本扩张** —— 一家靠增发把净资产做大 30% 的公司，"
          "`yoy_equity` 会给出 +30% 的「成长」，但老股东其实被摊薄了。"
          "`bps_yoy` 的分母是**每股**，增发被股本扩张抵消，"
          "测的是**老股东每股权益的累积**（= 留存收益驱动的内生增长）。"
          "两者在数学上**不是**单调变换关系（前者正比于股本增速的差）。"
          "⚠ **eval 必查项**：若实测与任何「净资产同比」序列的 rank 相关 > 0.95，"
          "说明「每股」这个调整没起作用（A 股多数年份股本变动很小），"
          "那就应当删掉 —— 这条判断写在 note 里是为了让下游有据可依。"),
))
def ind_bps_yoy(ctx):
    return ctx.ind("bps_yoy")
