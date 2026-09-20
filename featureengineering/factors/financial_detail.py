"""2026-09-19 财务原始字段扩展：现金支付、融资结构、资产组成。

本轮是候选特征开发，不将多开发字段等同于预测有效；不恢复以前删去的同名因子。
公式为参考库财务比率方法的扩展，全部走财务版本表，累计现金流先转 TTM，
资产负债科目使用公告时点值。慢变量的短样本 IC 不足以证明无效或有效。
"""
import numpy as np
from fea.spec import FactorSpec, register

BS='stock_balancesheet';CF='stock_cashflow';IS='stock_income'
NOTE=('本轮扩展公式，采用参考库 Quality/现金流比率的归一化方法；'
      '财务版本表按 max(ann_date,f_ann_date,end_date) 生效，保留修订版本；'
      '累计现金流先计算 TTM，资产负债科目不做 TTM。缺失不补 0，分母绝对值至少 100 万元；'
      '负现金支付按原值保留（存在供应商冲销记录）。慢变量需在跨年度样本评价。')

def _spec(name,desc,formula,deps,fields,start=None,high=True):
    return FactorSpec(name=name,group='financial_detail',desc=desc,formula=formula,
                      deps=deps,fin_fields=fields,warmup_days=700,start=start,
                      note=NOTE,higher_is_better=high)

@register(_spec('cf_tax_cash_burden','现金税费占收入','TTM(c_paid_for_taxes)/TTM(revenue)',(CF,IS),('c_paid_for_taxes','revenue'),high=False))
def cf_tax_cash_burden(ctx):return ctx.safe_div(ctx.ttm('c_paid_for_taxes'),ctx.ttm('revenue'),1e6)

@register(_spec('cf_purchase_cash_intensity','采购现金占收入','TTM(c_paid_goods_s)/TTM(revenue)',(CF,IS),('c_paid_goods_s','revenue'),high=False))
def cf_purchase_cash_intensity(ctx):return ctx.safe_div(ctx.ttm('c_paid_goods_s'),ctx.ttm('revenue'),1e6)

@register(_spec('cf_distribution_cash_coverage','分红与利息现金支付占经营现金流','TTM(c_pay_dist_dpcp_int_exp)/abs(TTM(n_cashflow_act))',(CF,),('c_pay_dist_dpcp_int_exp','n_cashflow_act'),high=False))
def cf_distribution_cash_coverage(ctx):return ctx.safe_div(ctx.ttm('c_pay_dist_dpcp_int_exp'),np.abs(ctx.ttm('n_cashflow_act')),1e6)

@register(_spec('cf_tax_refund_share','税费返还占营业收入','TTM(recp_tax_rends)/TTM(revenue)',(CF,IS),('recp_tax_rends','revenue')))
def cf_tax_refund_share(ctx):return ctx.safe_div(ctx.ttm('recp_tax_rends'),ctx.ttm('revenue'),1e6)

@register(_spec('cf_net_borrowing_to_assets','净借款现金流占总资产','(TTM(c_recp_borrow)-TTM(c_prepay_amt_borr))/total_assets',(CF,BS),('c_recp_borrow','c_prepay_amt_borr','total_assets'),high=False))
def cf_net_borrowing_to_assets(ctx):return ctx.safe_div(ctx.ttm('c_recp_borrow')-ctx.ttm('c_prepay_amt_borr'),ctx.point('total_assets'),1e6)

@register(_spec('cf_borrowing_repayment_ratio','借款流入对偿债现金的覆盖','TTM(c_recp_borrow)/TTM(c_prepay_amt_borr)',(CF,),('c_recp_borrow','c_prepay_amt_borr'),high=False))
def cf_borrowing_repayment_ratio(ctx):return ctx.safe_div(ctx.ttm('c_recp_borrow'),ctx.ttm('c_prepay_amt_borr'),1e6)

@register(_spec('bs_near_term_debt_share','近端债务占主要有息债务','(st_borr+non_cur_liab_due_1y)/(st_borr+non_cur_liab_due_1y+lt_borr+bond_payable)',(BS,),('st_borr','non_cur_liab_due_1y','lt_borr','bond_payable'),high=False))
def bs_near_term_debt_share(ctx):
    near=ctx.point('st_borr')+ctx.point('non_cur_liab_due_1y')
    return ctx.safe_div(near,near+ctx.point('lt_borr')+ctx.point('bond_payable'),1e6)

@register(_spec('bs_net_notes_to_assets','净应收票据占资产','(notes_receiv-notes_payable)/total_assets',(BS,),('notes_receiv','notes_payable','total_assets'),high=False))
def bs_net_notes_to_assets(ctx):return ctx.safe_div(ctx.point('notes_receiv')-ctx.point('notes_payable'),ctx.point('total_assets'),1e6)

@register(_spec('bs_other_receiv_to_assets','其他应收款占总资产','oth_receiv/total_assets',(BS,),('oth_receiv','total_assets'),high=False))
def bs_other_receiv_to_assets(ctx):return ctx.safe_div(ctx.point('oth_receiv'),ctx.point('total_assets'),1e6)

@register(_spec('bs_construction_capital_share','在建工程占固定投入资本','cip_total/(fix_assets+cip_total)',(BS,),('cip_total','fix_assets'),high=False))
def bs_construction_capital_share(ctx):return ctx.safe_div(ctx.point('cip_total'),ctx.point('fix_assets')+ctx.point('cip_total'),1e6)

@register(_spec('bs_intangible_asset_share','无形资产占总资产','intan_assets/total_assets',(BS,),('intan_assets','total_assets'),high=False))
def bs_intangible_asset_share(ctx):return ctx.safe_div(ctx.point('intan_assets'),ctx.point('total_assets'),1e6)

@register(_spec('bs_net_contract_to_assets','净合同资产占总资产','(contract_assets-contract_liab)/total_assets',(BS,),('contract_assets','contract_liab','total_assets'),start='2020-05-01',high=False))
def bs_net_contract_to_assets(ctx):return ctx.safe_div(ctx.point('contract_assets')-ctx.point('contract_liab'),ctx.point('total_assets'),1e6)
