"""One frozen C candidate: within-company TTM margin-path dispersion.

Weak low-dispersion prior. This is an inference from quality persistence,
not a replication of either the local report or QMJ earnings volatility.
Only retained whole-row standard consolidated income versions are used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from fea.deriv import PERIOD_BASE, PERIOD_W, KEY_W, period_of

PARAMETERS={'signal':'gpm_ttm_path_dispersion','window_quarters':16,'min_valid_quarters':12}
FIELDS=('revenue','oper_cost')
MIN_REVENUE=1e6
MAX_AGE_DAYS=550

def _valid_dates(ctx, values):
    raw=np.asarray(ctx.date_col(values),dtype=np.int64)
    valid=pd.to_datetime(raw.astype(str),format='%Y%m%d',errors='coerce').notna()
    return np.where(valid & (raw>=19900101),raw,0)

def _report_table(ctx):
    cols=['stock_code','ann_date','f_ann_date','end_date','comp_type','report_type',*FIELDS]
    frame=ctx.dataset('stock_income',columns=cols,years=None)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=['code','period','pit',*FIELDS])
    end=_valid_dates(ctx,frame['end_date'])
    ann=_valid_dates(ctx,frame['ann_date'])
    fann=_valid_dates(ctx,frame['f_ann_date'])
    pit=np.maximum(np.maximum(end,ann),fann)
    codes=np.asarray(ctx.code_index(frame['stock_code'].to_numpy()),dtype=np.int64)
    comp=pd.to_numeric(frame['comp_type'],errors='coerce').to_numpy(dtype=float)
    report=pd.to_numeric(frame['report_type'],errors='coerce').to_numpy(dtype=float)
    keep=(codes>=0) & (comp==1) & (report==1)
    keep &= (end>0) & np.isin(end%10000,[331,630,930,1231])
    keep &= ((ann>0)|(fann>0)) & (pit<=int(ctx.panel.dates[-1]))
    data=pd.DataFrame({'code':codes[keep],'period':period_of(end[keep]),'pit':pit[keep]})
    for field in FIELDS:
        v=pd.to_numeric(frame[field],errors='coerce').to_numpy(dtype=float)[keep]
        data[field]=np.where(np.isfinite(v),v,np.nan)
    # Standard whole rows win, including a newly missing component.
    return data.drop_duplicates(['code','period','pit'],keep='last')

def _versions(data):
    data=data.sort_values(['code','period','pit'],kind='stable')
    code=data.code.to_numpy(dtype=np.int64)
    quarter=data.period.to_numpy(dtype=np.int64)
    return {'keys':(code*PERIOD_W+quarter)*KEY_W+data.pit.to_numpy(dtype=np.int64),
            'code':code,'period':quarter,
            **{f:data[f].to_numpy(dtype=float) for f in FIELDS}}

def _lookup(vt,code,quarter,pit,field):
    if not len(vt['keys']):
        return np.full(np.broadcast_shapes(np.shape(code),np.shape(quarter),np.shape(pit)),np.nan)
    query=(code*PERIOD_W+quarter)*KEY_W+pit
    pos=np.searchsorted(vt['keys'],query,side='right')-1
    safe=np.maximum(pos,0)
    ok=(pos>=0)&(vt['code'][safe]==code)&(vt['period'][safe]==quarter)
    return np.where(ok,vt[field][safe],np.nan)

def _ttm(vt,code,quarter,pit,field):
    current=_lookup(vt,code,quarter,pit,field)
    prior_year_q4=(quarter//4)*4-1
    annual=_lookup(vt,code,prior_year_q4,pit,field)
    same=_lookup(vt,code,quarter-4,pit,field)
    # Q4 is its own complete fiscal YTD; no unused older rows required.
    return np.where(quarter%4==3,current,current+annual-same)

def _events(data):
    events=data[['code','pit','period']].sort_values(['code','pit','period'],kind='stable')
    events=events.drop_duplicates(['code','pit'],keep='last').copy()
    events['q']=events.groupby('code',sort=False).period.cummax()
    return events[['code','pit','q']].reset_index(drop=True)

def _compute(events,vt):
    result=np.full(len(events),np.nan)
    for start in range(0,len(events),4096):
        e=events.iloc[start:start+4096]
        code=e.code.to_numpy(dtype=np.int64)[:,None]
        q=e.q.to_numpy(dtype=np.int64)[:,None]-np.arange(16)[None,:]
        pit=e.pit.to_numpy(dtype=np.int64)[:,None]
        revenue=_ttm(vt,code,q,pit,'revenue')
        cost=_ttm(vt,code,q,pit,'oper_cost')
        eligible=np.isfinite(revenue)&(revenue>=MIN_REVENUE)&np.isfinite(cost)&(cost>0)
        with np.errstate(all='ignore'):
            gpm=np.where(eligible,1-cost/revenue,np.nan)
        eligible &= np.isfinite(gpm)
        n=eligible.sum(axis=1)
        avg=np.sum(np.where(eligible,gpm,0),axis=1)/np.maximum(n,1)
        variance=np.sum(np.where(eligible,(gpm-avg[:,None])**2,0),axis=1)/np.maximum(n,1)
        value=np.sqrt(variance)
        ok=(n>=12)&eligible[:,0]&np.isfinite(value)
        result[start:start+len(e)]=np.where(ok,value,np.nan)
    return result

def _epoch_days(values):
    return pd.to_datetime(np.asarray(values).astype(str),format='%Y%m%d',errors='raise').to_numpy(dtype='datetime64[D]').astype(np.int64)

def _map(ctx,events,values):
    if not len(events):
        return np.full(ctx.panel.shape,np.nan)
    pit=events.pit.to_numpy(dtype=np.int64)
    effective=np.asarray(ctx.next_trading_day(pit),dtype=np.int64)
    keep=(effective>pit)&(effective<=int(ctx.panel.dates[-1]))
    if not keep.any():
        return np.full(ctx.panel.shape,np.nan)
    code=events.code.to_numpy(dtype=np.int64)[keep]
    q=events.q.to_numpy(dtype=np.int64)[keep]
    period_end=(PERIOD_BASE+q//4)*10000+np.array([331,630,930,1231])[q%4]
    state=ctx.panel.asof(code,effective[keep],values[keep],dtype=np.float64)
    end=ctx.panel.asof(code,effective[keep],_epoch_days(period_end).astype(float),dtype=np.float64)
    age=_epoch_days(ctx.panel.dates)[:,None]-end
    valid=(age>=0)&(age<=MAX_AGE_DAYS)&np.isfinite(state)
    valid &= np.asarray(ctx.universe,dtype=bool)&np.asarray(ctx.traded(),dtype=bool)
    return np.where(valid,state,np.nan)

def build(parameters):
    if parameters!=PARAMETERS:
        raise ValueError('Only one frozen 16-quarter GPM dispersion definition is permitted')
    def compute(ctx):
        data=_report_table(ctx)
        events=_events(data)
        return _map(ctx,events,_compute(events,_versions(data)))
    return compute
