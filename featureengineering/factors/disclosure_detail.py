"""预告金额区间与封板字段的日频扩展；不强行使用缺公告日的质押表。"""
import numpy as np
import pandas as pd
from fea.spec import FactorSpec,register

FC='stock_forecast';LU='stock_limit_up'
FORECAST_NOTE=('扩展公式；预告金额与去年归母净利润同为供应商万元单位，比值约去单位；'
               '按 ann_date 生效，同一股票同日多条取中位数，不按报告期全表取最后版本；'
               '两端必须完整且有序，分母绝对值至少 100 万元。超过 365 日的披露置 NaN；'
               '财报分区按报告年读取额外 3 年，保留迟到公告。用于次日交易。')

def _forecast(ctx,mode):
    years=(int(ctx.panel.dates[0])//10000-3,int(ctx.panel.dates[-1])//10000+1)
    d=ctx.dataset(FC,columns=['stock_code','ann_date','net_profit_min','net_profit_max','last_parent_net'],years=years)
    if d.empty:return ctx.panel.empty()
    d=d.copy();ann=ctx.date_col(d['ann_date'])
    d=d[(ann>=19900101)&(ann<=int(ctx.panel.dates[-1]))].copy()
    if d.empty:return ctx.panel.empty()
    lo=d['net_profit_min'].to_numpy(dtype=float);hi=d['net_profit_max'].to_numpy(dtype=float)
    prev=d['last_parent_net'].to_numpy(dtype=float)
    num=(lo+hi)/2-prev if mode=='change' else hi-lo
    val=ctx.safe_div(num,np.abs(prev),100.)
    val=np.where((hi>=lo)&np.isfinite(lo)&np.isfinite(hi),val,np.nan)
    d['v']=val
    g=d.groupby(['stock_code','ann_date'],as_index=False)['v'].median()
    codes=g['stock_code'].to_numpy();days=ctx.date_col(g['ann_date'])
    out=ctx.asof_daily(codes,days,g['v'].to_numpy(dtype=float))
    ordinals=pd.to_datetime(g['ann_date']).to_numpy(dtype='datetime64[D]').astype(np.int64)
    last=ctx.asof_daily(codes,days,ordinals)
    now=pd.to_datetime(ctx.panel.dates.astype(str),format='%Y%m%d').to_numpy(dtype='datetime64[D]').astype(np.int64)
    return np.where(now[:,None]-last<=365,out,np.nan)

@register(FactorSpec(name='forecast_profit_midpoint_change',group='disclosure_detail',deps=(FC,),
    desc='预告利润区间中点相对去年同期的变化',formula='((net_profit_min+net_profit_max)/2-last_parent_net)/abs(last_parent_net)',
    warmup_days=700,note=FORECAST_NOTE))
def forecast_profit_midpoint_change(ctx):return _forecast(ctx,'change')

@register(FactorSpec(name='forecast_profit_range_uncertainty',group='disclosure_detail',deps=(FC,),
    desc='预告利润区间宽度相对去年利润',formula='(net_profit_max-net_profit_min)/abs(last_parent_net)',
    warmup_days=700,higher_is_better=False,note=FORECAST_NOTE))
def forecast_profit_range_uncertainty(ctx):return _forecast(ctx,'width')

SEAL_NOTE=('参考事件驱动因子方法的字段扩展。按日散点后取 20 交易日内有效涨停事件均值；'
           '没有事件为 NaN，不把空值当作没有封单。至少 1 个事件才有效，因此覆盖率天然稀疏；'
           '百分比按供应商原单位保留，负值无效。2018 年实测这些字段已可用；当晚生成供次日使用。')

def _seal(ctx,field,log=False):
    years=(int(ctx.panel.dates[0])//10000,int(ctx.panel.dates[-1])//10000)
    d=ctx.dataset(LU,columns=['stock_code','trade_date',field],years=years)
    if d.empty:return ctx.panel.empty()
    d=d.copy();d[field]=pd.to_numeric(d[field],errors='coerce')
    d=d[np.isfinite(d[field])&(d[field]>=0)]
    if d.empty:return ctx.panel.empty()
    g=d.groupby(['stock_code','trade_date'],as_index=False)[field].mean()
    codes=g['stock_code'].to_numpy();days=ctx.date_col(g['trade_date'])
    values=g[field].to_numpy(dtype=float)
    if log:values=np.log1p(values)
    total=ctx.roll_sum(ctx.event_grid(codes,days,values),20)
    count=ctx.roll_sum(ctx.event_grid(codes,days,np.ones(len(g))),20)
    return ctx.safe_div(total,count,.5)

@register(FactorSpec(name='seal_turnover_strength_20',group='disclosure_detail',deps=(LU,),
    desc='20 日涨停事件平均对数封单成交比',formula='event_mean(log1p(sealed_turnover_ratio),20)',
    warmup_days=60,start='2018-01-01',note=SEAL_NOTE))
def seal_turnover_strength_20(ctx):return _seal(ctx,'sealed_turnover_ratio',True)

@register(FactorSpec(name='seal_float_strength_20',group='disclosure_detail',deps=(LU,),
    desc='20 日涨停事件平均封单流通比',formula='event_mean(sealed_flow_ratio,20)',
    warmup_days=60,start='2018-01-01',note=SEAL_NOTE))
def seal_float_strength_20(ctx):return _seal(ctx,'sealed_flow_ratio')

@register(FactorSpec(name='seal_reopen_pressure_20',group='disclosure_detail',deps=(LU,),
    desc='20 日涨停事件平均对数开板次数',formula='event_mean(log1p(open_count),20)',
    warmup_days=60,start='2018-01-01',higher_is_better=False,note=SEAL_NOTE))
def seal_reopen_pressure_20(ctx):return _seal(ctx,'open_count',True)
