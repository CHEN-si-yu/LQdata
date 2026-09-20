"""供应商筹码收益摘要：与原 stock_cyq_chips 推算版本分开命名。

依据 学习资料/factors.md 的 chip_concentration / chip_position /
winner_rate 系列，改用原始值返回，由引擎统一排名。新衍生公式明确标为扩展。
2026-09-19 实盘核查：winner_rate 为百分数，且少量超过 100；非法值留 NaN，
不能逐行猜单位或裁成 100。成本水平以供应商当日口径保留，不跨日直接计算价格收益。
当日收盘后生成，供下一交易日使用；供应商历史重述和发布时间仍需持续观测。
"""
import numpy as np
import pandas as pd
from fea.spec import FactorSpec, register

SOURCE = 'stock_cyq_perf'
FIELDS = ('his_low','his_high','cost_5pct','cost_15pct','cost_50pct',
          'cost_85pct','cost_95pct','weight_avg','winner_rate')
NOTE = ('来源：学习资料/factors.md 筹码收益系列；返回原始值，排名由引擎执行。'
        '与旧筹码档位推算因子分开命名，不能假定两者完全相等。精确按日落格，缺失不前填；'
        'winner_rate 按百分数除以 100，超出 [0,100] 置 NaN；非正成本和逆序分位点置 NaN。'
        '收盘后数据供下一交易日使用；滚动窗口要求全窗有效。')

def _spec(name,desc,formula,price=False,high=True):
    return FactorSpec(name=name,group='chip_perf',deps=(SOURCE,'stock_daily') if price else (SOURCE,),
                      start='2018-01-02',warmup_days=80,desc=desc,formula=formula,
                      higher_is_better=high,note=NOTE)

def _field(ctx,field):
    years=(int(ctx.panel.dates[0])//10000,int(ctx.panel.dates[-1])//10000)
    df=ctx.dataset(SOURCE,columns=['stock_code','trade_date',*FIELDS],years=years)
    if df.empty:return ctx.panel.empty()
    values=pd.to_numeric(df[field],errors='coerce').to_numpy(dtype=np.float64,copy=True)
    if field=='winner_rate':
        values[(values<0)|(values>100)]=np.nan
        values/=100.0
    elif field!='his_low':
        values[values<=0]=np.nan
    else:
        values[values<0]=np.nan
    if field.startswith('cost_'):
        q=df[['cost_5pct','cost_15pct','cost_50pct','cost_85pct','cost_95pct']].to_numpy(dtype=float)
        valid=np.all(np.isfinite(q)&(q>0),axis=1)&np.all(np.diff(q,axis=1)>=0,axis=1)
        values[~valid]=np.nan
    ci=ctx.code_index(df['stock_code'].to_numpy())
    ok=ci>=0
    return ctx.panel.place(ci[ok],ctx.date_col(df['trade_date'])[ok],values[ok])

def _width(ctx,lo='cost_5pct',hi='cost_95pct'):
    return ctx.safe_div(_field(ctx,hi)-_field(ctx,lo),_field(ctx,'cost_50pct'),1e-8)

@register(_spec('cyqp_winner_fraction','供应商获利盘占比','winner_rate / 100',high=False))
def cyqp_winner_fraction(ctx):return _field(ctx,'winner_rate')

@register(_spec('cyqp_winner_change_20','获利盘占比 20 日变化','winner_fraction(T) - winner_fraction(T-20)'))
def cyqp_winner_change_20(ctx):return ctx.diff(_field(ctx,'winner_rate'),20)

@register(_spec('cyqp_winner_acceleration_5','获利盘占比 5 日二阶差分','diff(diff(winner_fraction,5),5)',high=False))
def cyqp_winner_acceleration_5(ctx):return ctx.diff(ctx.diff(_field(ctx,'winner_rate'),5),5)

@register(_spec('cyqp_winner_volatility_20','获利盘占比 20 日波动','std(winner_fraction,20)',high=False))
def cyqp_winner_volatility_20(ctx):return ctx.roll_std(_field(ctx,'winner_rate'),20)

@register(_spec('cyqp_cost_width_90','90% 筹码成本相对宽度','(cost_95pct-cost_5pct)/cost_50pct',high=False))
def cyqp_cost_width_90(ctx):return _width(ctx)

@register(_spec('cyqp_cost_width_70','70% 筹码成本相对宽度','(cost_85pct-cost_15pct)/cost_50pct',high=False))
def cyqp_cost_width_70(ctx):return _width(ctx,'cost_15pct','cost_85pct')

@register(_spec('cyqp_cost_width_change_20','90% 成本宽度 20 日变化','diff((cost_95pct-cost_5pct)/cost_50pct,20)',high=False))
def cyqp_cost_width_change_20(ctx):return ctx.diff(_width(ctx),20)

@register(_spec('cyqp_cost_tail_asymmetry','上下成本尾部不对称（扩展）','(cost_95pct+cost_5pct-2*cost_50pct)/(cost_95pct-cost_5pct)',high=False))
def cyqp_cost_tail_asymmetry(ctx):
    lo,mid,hi=(_field(ctx,k) for k in ('cost_5pct','cost_50pct','cost_95pct'))
    return ctx.safe_div(hi+lo-2*mid,hi-lo,1e-8)

@register(_spec('cyqp_tail_width_share','两端尾部占 90% 区间的宽度比例（扩展）','((cost_95pct-cost_85pct)+(cost_15pct-cost_5pct))/(cost_95pct-cost_5pct)',high=False))
def cyqp_tail_width_share(ctx):
    a,b,c,d=(_field(ctx,k) for k in ('cost_5pct','cost_15pct','cost_85pct','cost_95pct'))
    return ctx.safe_div((d-c)+(b-a),d-a,1e-8)

@register(_spec('cyqp_mean_median_gap','筹码均值与中位数成本偏离（扩展）','weight_avg/cost_50pct - 1',high=False))
def cyqp_mean_median_gap(ctx):return ctx.safe_div(_field(ctx,'weight_avg'),_field(ctx,'cost_50pct'),1e-8)-1

@register(_spec('cyqp_price_cost_position','现价在 90% 筹码成本区间的位置','(close-cost_5pct)/(cost_95pct-cost_5pct)',price=True,high=False))
def cyqp_price_cost_position(ctx):
    lo,hi=_field(ctx,'cost_5pct'),_field(ctx,'cost_95pct')
    return ctx.safe_div(ctx.px('close')-lo,hi-lo,1e-8)

@register(_spec('cyqp_average_cost_premium','现价相对供应商加权成本的溢价','close/weight_avg - 1',price=True,high=False))
def cyqp_average_cost_premium(ctx):return ctx.safe_div(ctx.px('close'),_field(ctx,'weight_avg'),1e-8)-1

@register(_spec('cyqp_historical_range_position','中位成本在供应商历史价格区间的位置（扩展）','(cost_50pct-his_low)/(his_high-his_low)'))
def cyqp_historical_range_position(ctx):
    lo,hi=_field(ctx,'his_low'),_field(ctx,'his_high')
    return ctx.safe_div(_field(ctx,'cost_50pct')-lo,np.where(hi>lo,hi-lo,np.nan),1e-8)

@register(_spec('cyqp_cost_premium_change_20','现价相对筹码中位成本溢价的 20 日变化','diff(close/cost_50pct - 1,20)',price=True))
def cyqp_cost_premium_change_20(ctx):
    return ctx.diff(ctx.safe_div(ctx.px('close'),_field(ctx,'cost_50pct'),1e-8)-1,20)
