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
# ★★ 2026-09-25：由 "2019-05-01" 改为 None（跟随全局下界 2018-01-01）。
#
#   上面那条注释说的是 2012/2015 的情况（非零率 0.00%），**但 2018 年不是这样**：
#   实测 `stock_income.fin_exp_int_exp` 在 2018 年非空 17630 行、精确 0 占 55.9%
#   ⇒ **44% 有真实值**。按 2019-05-01 截断会丢掉这些真实观测。
#   与 factors/quality.py 的 `RD_START` 同一处置：上游支持 ⇒ 放开真算、不填充。
IND_START = None

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



# ══════════════════════════════════════════════════════════════════════
# 1. 利润率与盈利构成（5 个）
# ══════════════════════════════════════════════════════════════════════









@register(FactorSpec(
    name="cash_sales_ratio", version=2,
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
    name="cfcr", version=2,
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




# ══════════════════════════════════════════════════════════════════════
# 3. 债务结构（2 个）—— ★ `ctx.ind()` 白名单的**首次启用**
# ══════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="ind_int_to_talcap", version=2,
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











# ══════════════════════════════════════════════════════════════════════
# 5. 供应商同比字段（2 个）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="ind_dt_netprofit_yoy", version=2,
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
    name="ind_bps_yoy", version=2,
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
