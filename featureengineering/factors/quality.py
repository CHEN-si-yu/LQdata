"""质量 / 盈利能力因子（20 个）—— 利润表 / 资产负债表 / 现金流量表 + 市值。

日频 / 只主板 / `default_start` 起这三条硬约束由引擎强制，本文件只写公式。

## 口径总纲

1. **一律 `ctx.ttm(field)` / `ctx.point(field)`**，不直接读财报 parquet。
   累计制科目（利润表 / 现金流量表）走 TTM；时点科目（资产负债表）走 point。
   TTM 的追溯修正、`ann_date` 前向填充、PIT 对齐全部由 `fea/deriv.py` 一次做完。
2. **净利润统一用归母口径**（`n_income_attr_p`）。理由：既有 `roe_ttm` /
   `yoy_net_profit` 就是归母口径，本家族跟着走，才能让杜邦恒等式
   `ROE = ROA × 权益乘数 = 净利率 × 总资产周转率 × 权益乘数` **逐格成立**
   （权益乘数也用归母权益）。参考库只写 `NetProfit`，没指明是否含少数股东。
   用含少数股东的 `n_income` 会把少数股东损益大的公司系统性抬高。
3. **营收一律 `revenue`（营业收入）**，不是 `total_revenue`（营业总收入）——
   见 `fea/deriv.py` 的实测（茅台差的是利息收入）。与既有 `yoy_revenue` 一致。
4. **除法一律 `ctx.safe_div(..., min_abs_den=...)`**，地板 **100 万元（1e6 元）**。
   上游财报单位是**元**（实测 2016 年报 `revenue` 中位数 1.03e9）。地板的作用：
   - 银行/券商的 `inventories` / `accounts_receiv` / `fix_assets` 是**精确 0.0**
     （结构性没有这些科目），0 分母必须返回 NaN，不许 `nan_to_num` 填 0；
   - 近零分母（壳公司的存货、盈亏平衡的净利、无有息负债的利息支出）
     会把比率炸成 ±1e5 量级的假值，把整个截面排名带偏。
   地板在 `note` 里逐个说明。
5. **市值自算**：`ctx.px("close") × ctx.px("total_share")`（**未复权**价 ×
   当期已披露股本，PIT 可控），不用 `stock_finance.total_mv`。
6. **`warmup_days=700`**：TTM 要 4 季 + `ann_date` 最长滞后 15 个月（契约 §2）。
7. **`start=None`**（跟随 `conf/config.yaml: default_start` = 2012-01-01）。
   **两个例外**：`rd_intensity` / `interest_coverage` 的上游字段在 2019 年前
   是「未披露」而不是「真值」（恒为精确 0.0），显式写 `start="2019-05-01"`，
   见各自 note 的实测数据 —— 契约 §1.3 允许受上游限制的因子显式写起点
   （与 `factors/event.py` 的 `start="2016-01-04"` 同一处置）。

## 参考库对照（`学习资料/因子库.md` 5、Quality 59 个 + 9、Value 11 个）

| 本文件 | 参考库 | 说明 |
|:--|:--|:--|
| `roa_ttm` | #53 `roa_ttm` | 逐字用其公式 |
| `net_margin_ttm` | #35 `npm_ttm` | 逐字用其公式 |
| `asset_turnover` | #12 `asset_turnover` | 逐字 |
| `inventory_turnover` | #52 `inventory_turnover` | 逐字（分子 Cost_TTM） |
| `receivable_turnover` | #37 `receivable_turnover` | 逐字 |
| `fixed_asset_turnover` | #34 `fixed_asset_turnover` | 逐字 |
| `equity_multiplier` | #4 `financial_leverage` | 逐字（= 权益乘数） |
| `interest_coverage` | #6 `icr` | 逐字（EBIT_TTM / InterestExpense_TTM） |
| `current_ratio` | #47 `current_ratio` | 公式逐字；偏离「直接用预计算字段」 |
| `quick_ratio` | #24 `quick_ratio` | 公式逐字；同上 |
| `fcf_to_market` | Value #5 `fcf_to_market` | 公式逐字；分子口径偏离（见 note） |
| 其余 9 个 | 参考库未收录 | 公式取标准定义，note 里写明 |

**本文件刻意不建的候选**（契约 §8：同号重复会被下游当成两个独立特征）：
- `cash_ratio`：`(money_cap+trad_asset)/total_cur_liab` 与 `quick_ratio` 同分母、
  分子是其子集，A 股两者相关系数 ~0.9；
- `earnings_quality`：与 `ocf_to_profit` 是同一个量（OCF/NP）；
- `intangible_ratio`：A 股「无形资产」以**土地使用权**为主（不是研发型无形资产），
  与 `goodwill_to_assets` 的信息高度重叠；
- `sell_exp_ratio` / `admin_exp_ratio` / `fin_exp_ratio` / `tax_rate_effective` /
  `minority_gain_ratio`：费用率与税率不属于「质量」维度（本族已用
  `net_margin_ttm` 隐含了费用率的总效果），且 `fin_exp` 对 20%~34% 的公司为负
  （利息净收入），比率方向会翻转；
- `capex_to_depreciation`：与 `capex_to_revenue` 同向（分母换成折旧），
  且折旧口径在 2019 年新租赁准则后断裂；
- `ocf_to_asset`：与 `roa_ttm` 只差分子（OCF vs 净利）——两者的差正是
  `accruals_ratio`，再建一个是三重共线。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

# ★ 起点留 None = 跟随 conf/config.yaml 的 default_start（当前 2012-01-01）
FIN_START = None
# TTM 4 季 + ann_date 最长滞后 15 个月 → 700 日历天（契约 §2）
FIN_WARMUP = 700
# ★ 上游字段受限的两个因子（见 rd_intensity / interest_coverage 的 note）
RD_START = "2019-05-01"

# ---------------------------------------------------------------- 上游字段
# 累计制（-> TTM）
NP = "n_income_attr_p"          # 归母净利润
REV = "revenue"                 # 营业收入
COST = "oper_cost"              # 营业成本
OCF = "n_cashflow_act"          # 经营活动现金流净额
EBIT = "ebit"                   # 息税前利润
INT_EXP = "fin_exp_int_exp"     # 财务费用：利息支出
RD_EXP = "rd_exp"               # 研发费用（费用化）
CAPEX = "c_pay_acq_const_fiolta"  # 购建固定资产、无形资产和其他长期资产支付的现金
FCF = "free_cashflow"           # 供应商自算的企业自由现金流
# 时点制（资产负债表）
TA = "total_assets"
TL = "total_liab"
EQ = "total_hldr_eqy_exc_min_int"
CA = "total_cur_assets"
CL = "total_cur_liab"
INV = "inventories"
AR = "accounts_receiv"
AP = "acct_payable"
FA = "fix_assets"
GW = "goodwill"
CASH = "money_cap"

# 货币口径的分母地板：**100 万元**（上游财报单位是元）
MIN_CUR = 1e6

# 上游依赖（用于输入水位失效判定）
DEP_I = ("stock_income",)
DEP_B = ("stock_balancesheet",)
DEP_C = ("stock_cashflow",)
DEP_IB = ("stock_income", "stock_balancesheet")
DEP_CB = ("stock_cashflow", "stock_balancesheet")
DEP_CIB = ("stock_cashflow", "stock_income", "stock_balancesheet")
DEP_PX = ("stock_daily", "stock_finance")


def _mktcap(ctx) -> np.ndarray:
    """总市值（元）= 未复权收盘价 × 当期已披露总股本。

    `close` 与 `total_share` 都是**状态量**，价格层已做前向填充（停牌期间沿用
    最后一个成交价与股本）—— 这正是市值该有的语义。不用供应商
    `stock_finance.total_mv`：日频快照表会被事后重算，自算才 PIT 可控。
    """
    px = np.asarray(ctx.px("close"), dtype=np.float64)
    sh = np.asarray(ctx.px("total_share"), dtype=np.float64)
    return px * sh


# ══════════════════════════════════════════════════════════════════════
# 盈利能力
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="roa_ttm", group="quality", deps=DEP_IB,
    desc="总资产收益率（TTM）= 归母净利润TTM / 期末总资产",
    formula="ROA = NetProfit / TotalAssets",
    start=FIN_START, warmup_days=FIN_WARMUP, higher_is_better=True,
    fin_fields=("n_income_attr_p", "total_assets"),
    note="★ 分子用**归母**净利润，与既有 roe_ttm / yoy_net_profit 同口径 —— "
         "这样 ROE = ROA × 权益乘数 的杜邦恒等式在本家族内逐格成立"
         "（权益乘数同样用归母权益）。参考库只写 NetProfit，未指明是否含少数股东。"
         "分母用**期末**总资产（参考库同）；平均资产口径要取前一期报表，PIT 更脆。"
         "银行的总资产收益率天然很低（~0.8%），是行业属性不是异常。",
))
def roa_ttm(ctx):
    return ctx.safe_div(ctx.ttm(NP), ctx.point(TA), min_abs_den=MIN_CUR)


@register(FactorSpec(
    name="net_margin_ttm", group="quality", deps=DEP_I,
    desc="销售净利率（TTM）= 归母净利润TTM / 营业收入TTM",
    formula="NPM = NetProfit / OperatingRevenue",
    start=FIN_START, warmup_days=FIN_WARMUP, higher_is_better=True,
    fin_fields=("n_income_attr_p", "revenue"),
    note="参考库 #35 npm_ttm 的公式未指明净利口径，这里统一归母（见文件头口径总纲）。"
         "营收用 revenue（营业收入），不是 total_revenue（营业总收入）。"
         "净利率可以为负（亏损），是正常的截面读数。"
         "地板 100 万元营收：TTM 营收低于此的主板公司等于空壳，比率无意义。",
))
def net_margin_ttm(ctx):
    return ctx.safe_div(ctx.ttm(NP), ctx.ttm(REV), min_abs_den=MIN_CUR)


@register(FactorSpec(
    name="ocf_to_revenue", group="quality", deps=DEP_CIB,
    desc="经营现金流占营收比（TTM）= 经营现金流TTM / 营业收入TTM",
    formula="OCFtoRevenue = NetOperateCashFlow_TTM / OperatingRevenue_TTM",
    start=FIN_START, warmup_days=FIN_WARMUP, higher_is_better=True,
    fin_fields=("n_cashflow_act", "revenue"),
    note="参考库未单列（其 ind 表里的 ocf_to_or 是**累计 YTD** 口径，被契约 §5 禁用，"
         "故这里用 ctx.ttm 从现金流量表重算）。"
         "**银行业例外**：经营现金流被存款/同业资金进出主导，该比值对银行没有"
         "「收入含金量」的含义 —— 银行会散布到截面两端，下游按行业中性化时会自动处理。",
))
def ocf_to_revenue(ctx):
    return ctx.safe_div(ctx.ttm(OCF), ctx.ttm(REV), min_abs_den=MIN_CUR)


@register(FactorSpec(
    name="ocf_to_profit", group="quality", deps=DEP_CIB,
    desc="盈利现金保障倍数（TTM）= 经营现金流TTM / 归母净利润TTM",
    formula="OCFtoProfit = NetOperateCashFlow_TTM / NetProfit_TTM",
    start=FIN_START, warmup_days=FIN_WARMUP, higher_is_better=True,
    fin_fields=("n_cashflow_act", "n_income_attr_p"),
    note="★ 分母用**带符号**的归母净利润并设 100 万元地板："
         "净利为负的公司得到**负值**（经营现金流覆盖不了亏损），符合语义。"
         "**刻意不用 |NetProfit|** —— 那会把「亏损但现金流为正」的公司排到截面最顶端"
         "（方向完全反了）；同文件的 cash_profit_ratio 用 |NP| 是因为它构造的是"
         "「超额现金流的相对量」，两者的分母保护动机不同。"
         "|NP| < 100 万元视为盈亏平衡、比率无意义 → NaN。",
))
def ocf_to_profit(ctx):
    return ctx.safe_div(ctx.ttm(OCF), ctx.ttm(NP), min_abs_den=MIN_CUR)




# ══════════════════════════════════════════════════════════════════════
# 资产使用效率（周转率 / 营业周期）
# ══════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="inventory_turnover", group="quality", deps=DEP_IB,
    desc="存货周转率（TTM，次/年）= 营业成本TTM / 期末存货",
    formula="InventoryTurnover = OperatingCost_TTM / Inventories",
    start=FIN_START, warmup_days=FIN_WARMUP, higher_is_better=True,
    fin_fields=("oper_cost", "inventories"),
    note="★ 分子用**营业成本** TTM（参考库 Cost_TTM），不是营收 —— 存货与成本配比。"
         "★ 分母地板 100 万元（元）：银行/券商的 inventories 是**精确 0.0**"
         "（结构性没有存货科目），0 分母必须返回 NaN 而不是 inf/巨值（契约要求，"
         "不许 nan_to_num 填）。此外 oper_cost 对金融股已在 deriv 层置 NaN，"
         "所以金融股天然不参与本因子 —— 这是两层保护。"
         "地产/建筑公司的存货周转率天然很低（~0.3），是行业属性。",
))
def inventory_turnover(ctx):
    return ctx.safe_div(ctx.ttm(COST), ctx.point(INV), min_abs_den=MIN_CUR)






@register(FactorSpec(
    name="cash_conversion_cycle", group="quality", deps=DEP_IB,
    desc="现金转换周期（天）= 存货周转天数 + 应收周转天数 − 应付周转天数（低优）",
    formula="CCC = DIO + DSO - DPO = 365×Inventories/OperatingCost_TTM "
            "+ 365×AccountsReceivable/OperatingRevenue_TTM "
            "- 365×AccountPayable/OperatingCost_TTM",
    start=FIN_START, warmup_days=FIN_WARMUP, higher_is_better=False,
    fin_fields=("oper_cost", "revenue", "inventories", "accounts_receiv", "acct_payable"),
    note="参考库未收录，用标准定义（天）。**方向低优**：周期越短越好，"
         "**可以为负**——负 CCC 意味着「先收钱后付钱」，占用上游资金经营"
         "（商超、白酒、预收款型公司），是强的质量信号，不要当异常值删掉。"
         "地板 100 万元设在**分母**（营业成本 / 营收 TTM）：存货为 0 的服务型公司 "
         "DIO 正确地等于 0（分子为 0 → 0），而金融股 oper_cost 已被 deriv 层置 NaN → "
         "整个 CCC 为 NaN（银行本就没有营业周期）。"
         "**与三个周转率因子的关系**：CCC = 365×存货周转率⁻¹ + 365×应收周转率⁻¹ − "
         "365×应付周转率⁻¹ —— 是它们的**倒数**组合，仍是独立信息（倒数放大低周转公司"
         "之间的差异），但高度相关，下游建模注意共线性。"
         "★ **实测尾部**：2012-2014 有 6.0% 的格子 > 1000 天、0.9% > 3650 天"
         "（最大 1.2e5 天）。逐条查过，不是数据错误，而是两类**真实**情形："
         "(a) 地产/建筑（土地储备与应收工程款 vs 当期确认成本，周转天数天然上千）；"
         "(b) 停业/壳公司 —— TTM 营收刚过 100 万元地板（如 600275.SH 2012 年营收 110 万元"
         "而应收 3.6 亿元）。地板再往上提到 1 亿元能清掉 (b)，但会连带丢掉 3.5% 的"
         "营收（10% 的成本）—— 代价大于收益，故保持 100 万元地板，"
         "由引擎的 1%/99% 缩尾处理尾部（p99≈3471 天）。",
))
def cash_conversion_cycle(ctx):
    cost = ctx.ttm(COST)
    rev = ctx.ttm(REV)
    dio = ctx.safe_div(365.0 * ctx.point(INV), cost, min_abs_den=MIN_CUR)
    dso = ctx.safe_div(365.0 * ctx.point(AR), rev, min_abs_den=MIN_CUR)
    dpo = ctx.safe_div(365.0 * ctx.point(AP), cost, min_abs_den=MIN_CUR)
    return dio + dso - dpo


# ══════════════════════════════════════════════════════════════════════
# 杠杆与偿债
# ══════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="interest_coverage", group="quality", deps=DEP_I,
    desc="利息保障倍数（TTM）= EBIT_TTM / 利息支出TTM（值越大偿债越安全）",
    formula="ICR = EBIT_TTM / InterestExpense_TTM",
    start=RD_START, warmup_days=FIN_WARMUP, higher_is_better=True,
    fin_fields=("ebit", "fin_exp_int_exp"),
    note="★ 参考库 #6 `icr` 逐字。**起点 2019-05-01（偏离 start=None）**："
         "上游 `fin_exp_int_exp`（财务费用中的利息支出）在 2019 年之前的报告期里"
         "**100% 是精确的 0.0**（实测 2012Q1~2017Q4 全部为 0；2018 年报只有 45% 非零；"
         "2019 年起 ~78% 非零），即「没披露」而不是「没有利息支出」。"
         "不设起点的话 2012-2018 的因子恒为 NaN，看起来像 bug。"
         "取 2019-05-01 = 全部主板公司 2018 年报（2019-04-30 前）都已公告，"
         "TTM 窗口不再混入旧披露口径。"
         "分母地板 100 万元：利息支出小于此视为无有息负债，比率无意义。"
         "EBIT 为负 → ICR 为负 → 「赚的钱不够付利息」，方向语义正确。",
))
def interest_coverage(ctx):
    return ctx.safe_div(ctx.ttm(EBIT), ctx.ttm(INT_EXP), min_abs_den=MIN_CUR)






# ══════════════════════════════════════════════════════════════════════
# 流动性
# ══════════════════════════════════════════════════════════════════════





# ══════════════════════════════════════════════════════════════════════
# 资产质量 / 投入强度
# ══════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="rd_intensity", group="quality", deps=DEP_I,
    desc="研发强度（TTM）= 研发费用TTM / 营业收入TTM",
    formula="RDIntensity = RDExpense_TTM / OperatingRevenue_TTM",
    start=RD_START, warmup_days=FIN_WARMUP, higher_is_better=True,
    fin_fields=("rd_exp", "revenue"),
    note="★ 参考库未收录（研发投入维度）。**起点 2019-05-01（偏离 start=None）**："
         "上游 `rd_exp` 在 2012~2017 的年报行里 **97%~100% 是精确的 0.0**"
         "（实测：2012-2014 正值占比 0.03%，2015 起 3%，2017 才 25%），"
         "因为 2018 年新准则（财会[2018]15号）之前研发费用普遍混在管理费用里、"
         "**不单独披露** —— 那是「没披露」而不是「没研发」。"
         "2018 年报起披露率跳到 87%~88%。不设起点的话 2012-2017 的因子值是"
         "「一大片 0 + 零星正值」，虽然不是常数，但截面完全没有区分度。"
         "取 2019-05-01 = 全部主板公司 2018 年报都已公告，TTM 窗口口径统一。"
         "口径说明：用**费用化**的 `rd_exp`；`r_and_d`（资产负债表时点字段）实测"
         "99% 是 0（只有资本化的开发支出），不值得用。",
))
def rd_intensity(ctx):
    return ctx.safe_div(ctx.ttm(RD_EXP), ctx.ttm(REV), min_abs_den=MIN_CUR)




