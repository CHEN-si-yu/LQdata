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

def _spec(name,desc,formula,deps,fields,start=None,high=True,align_fill=float("nan")):
    return FactorSpec(name=name, version=2,group='financial_detail',desc=desc,formula=formula,
                      deps=deps,fin_fields=fields,warmup_days=700,start=start,
                      align_fill=align_fill,   # 见 FactorSpec.align_fill（时间轴对齐）
                      note=NOTE,higher_is_better=high)


@register(_spec('cf_purchase_cash_intensity','采购现金占收入','TTM(c_paid_goods_s)/TTM(revenue)',(CF,IS),('c_paid_goods_s','revenue'),high=False))
def cf_purchase_cash_intensity(ctx):return ctx.safe_div(ctx.ttm('c_paid_goods_s'),ctx.ttm('revenue'),1e6)







@register(_spec('bs_other_receiv_to_assets','其他应收款占总资产','oth_receiv/total_assets',(BS,),('oth_receiv','total_assets'),high=False))
def bs_other_receiv_to_assets(ctx):return ctx.safe_div(ctx.point('oth_receiv'),ctx.point('total_assets'),1e6)



# ★★ 2026-09-25 时间轴对齐：`contract_assets`/`contract_liab` 在 2018 年**非空行里
#   99.3% / 98.0% 是精确 0** —— 上游对未披露的合同资产就是写 0（不是缺测）。
#   所以对齐区间（2018-01-01 ~ 2020-04-30）**填 0**，与上游口径一致；
#   填 NaN 反而会把"上游明确说 0"改写成"不知道"。
