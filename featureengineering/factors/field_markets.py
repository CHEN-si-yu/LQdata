"""板块/指数原始字段 -> 市场状态 -> 个股60日状态相关暴露。

每个历史日只汇总该日有记录的指数，不使用今天的板块成分映射。
市场状态本身全股相同，不能直接当截面因子；输出是个股收益对它的滚动相关。
分钟源逐批读取，只缓存小型日摘要，不缓存整张分钟表。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fea.spec import FactorSpec, register
from .field_events import ratio

SOURCE_FIELDS={}
# 全历史数据质量验收的保留/起点决定；不依据收益筛选。
REJECTED_CANDIDATES = {
    "mfx_dc_swing",   # 2026-09-22 去冗余：|ρ|≥0.95 簇的重复项，见 scripts/prune_factors.py 清单
    "mfx_minute_pressure",   # 2026-09-22 去冗余：|ρ|≥0.95 簇的重复项，见 scripts/prune_factors.py 清单
    "mfx_minute_realized_vol",   # 2026-09-22 去冗余：|ρ|≥0.95 簇的重复项，见 scripts/prune_factors.py 清单
    "mfx_tdx_pressure",   # 2026-09-22 去冗余：|ρ|≥0.95 簇的重复项，见 scripts/prune_factors.py 清单
    "mfx_tdx_range",   # 2026-09-22 去冗余：|ρ|≥0.95 簇的重复项，见 scripts/prune_factors.py 清单
    "mfx_ths_pct_change",   # 2026-09-22 去冗余：|ρ|≥0.95 簇的重复项，见 scripts/prune_factors.py 清单
    "mfx_ths_pressure",   # 2026-09-22 去冗余：|ρ|≥0.95 簇的重复项，见 scripts/prune_factors.py 清单
    "mfx_ths_range",   # 2026-09-22 去冗余：|ρ|≥0.95 簇的重复项，见 scripts/prune_factors.py 清单
    "mfx_ths_turnover_rate",   # 2026-09-22 去冗余：|ρ|≥0.95 簇的重复项，见 scripts/prune_factors.py 清单
}
LATE_STARTS = {}

DAILY={
    'index_daily':('ts_code',['open','high','low','close','pre_close','vol','amount']),
    'index_ths_daily':('ths_code',['open','high','low','close','pre_close','avg_price','pct_change','vol','turnover_rate']),
    'dc_daily':('block_code',['open','high','low','close','pct_change','vol','amount','swing','turnover_rate']),
    'tdx_daily':('board_code',['open','high','low','close','vol','amount']),
}
MINUTE_FIELDS=['open','high','low','close','vol','amount','pct_change','amplitude']
BROAD_A_INDICES=('000001.SH','000016.SH','000300.SH','000905.SH','000852.SH','399001.SZ')

def _bounds(ctx):
    return (int(ctx.panel.dates[0]), min(int(ctx.panel.dates[-1]),int((getattr(ctx.up,'audit_cutoff',None) or ctx.panel.dates[-1]))))

def _daily(ctx,ds):
    lo,hi=_bounds(ctx)
    ident,cols=DAILY[ds]
    d=ctx.up.read(ds,columns=[ident,'trade_date',*cols],years=(lo//10000,hi//10000),use_cache=False)
    if d.empty:return pd.DataFrame()
    d=d[(ctx.date_col(d.trade_date)>=lo)&(ctx.date_col(d.trade_date)<=hi)].copy()
    if ds=='index_daily':
        # 原表也含商品期货/海外代码；明确只取这组境内股票基准，不随末日库存选样。
        d=d[d[ident].isin(BROAD_A_INDICES)]
    d=d.drop_duplicates([ident,'trade_date'],keep='last')
    v=pd.DataFrame({'trade_date':d.trade_date})
    v['pressure']=ratio(d.close-d.open,d.high-d.low)
    v['range']=np.where(d.high>=d.low,ratio(d.high-d.low,d.close),np.nan)
    v['volume']=np.log1p(d.vol.where(d.vol>0))
    if 'amount' in d:v['amount']=np.log1p(d.amount.where(d.amount>0))
    if 'pre_close' in d:v['gap']=ratio(d.open,d.pre_close)-1
    if 'avg_price' in d:v['premium']=ratio(d.close,d.avg_price)-1
    for f in ['pct_change','swing','turnover_rate']:
        if f in d:v[f]=d[f]
    # 宽基至少3条，其余至少5条；等权中位数减少大板块支配。
    grouped=v.groupby('trade_date')
    out=grouped.median().where(grouped.count()>=(3 if ds=='index_daily' else 5))
    out.index=ctx.date_col(pd.Series(out.index))
    return out

def _minute(ctx):
    lo,hi=_bounds(ctx)
    parts=[]
    for year in range(lo//10000,hi//10000+1):
        path=Path(ctx.up.root)/'tdx_minute'/f'year={year}'/'data.parquet'
        if not path.exists():continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=131072,columns=['board_code','trade_time',*MINUTE_FIELDS]):
            d=batch.to_pandas()
            days=ctx.date_col(d.trade_time.str[:10])
            d=d.loc[(days>=lo)&(days<=hi)].copy()
            if d.empty:continue
            d['day']=days[(days>=lo)&(days<=hi)]
            body=ratio(d.close-d.open,d.high-d.low)
            valid=(np.isfinite(body)&(d.vol>=0)&(d.amount>=0)&np.isfinite(d['pct_change'])&np.isfinite(d.amplitude))
            d=d.loc[valid].copy();body=body[valid]
            x=pd.DataFrame({'board':d.board_code,'day':d.day,'n':1.,'body':body,
                'weighted':body*d.amount,'amount':d.amount,'vol':d.vol,'vol2':d.vol**2,
                'return2':d['pct_change']**2,'amplitude':d.amplitude})
            parts.append(x.groupby(['board','day']).sum())
    if not parts:return pd.DataFrame()
    x=pd.concat(parts).groupby(level=[0,1]).sum()
    # 一天48根5min；允许最多8根零振幅/缺失分钟，记录数过多的板块日不纳入。
    x=x.loc[x.n.between(40,48)]
    v=pd.DataFrame(index=x.index)
    v['pressure']=ratio(x.body,x.n)
    v['weighted_pressure']=ratio(x.weighted,x.amount)
    v['volume_concentration']=ratio(x.n*x.vol2,x.vol**2)
    v['realized_vol']=np.sqrt(x.return2)
    v['amplitude']=ratio(x.amplitude,x.n)
    g=v.groupby(level='day')
    return g.median().where(g.count()>=25)

def market(ctx,ds):
    key=(ds,*_bounds(ctx))
    cache=getattr(ctx.up,'_expansion_market_cache',{})
    if key not in cache:
        # 只保留同一个窗口，避免跨年/跨warmup累积。
        cache={k:v for k,v in cache.items() if k[1:]==key[1:]}
        cache[key]=_minute(ctx) if ds=='tdx_minute' else _daily(ctx,ds)
        ctx.up._expansion_market_cache=cache
    return cache[key]

def add(ds,kind,fields,start=None):
    name='mfx_'+{'index_daily':'index','index_ths_daily':'ths','dc_daily':'dc','tdx_daily':'tdx','tdx_minute':'minute'}[ds]+'_'+kind
    if name in REJECTED_CANDIDATES:return
    start=LATE_STARTS.get(name,start)
    SOURCE_FIELDS[name]={ds:list(fields),'stock_daily':['close','vol'],'stock_adj_factor':['adj_factor']}
    @register(FactorSpec(name=name,group='field_markets',deps=(ds,'stock_daily','stock_adj_factor'),
        desc=f'个股收益与{ds}的{kind}状态60日相关暴露',
        formula=f'corr60(ret_clean_hfq, cross_index_median({ds}.{kind})); volume/amount use log change',
        warmup_days=220,start=start,note='本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。'
        '不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。'
        '每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。'))
    def compute(ctx):
        frame=market(ctx,ds)
        if frame.empty or kind not in frame:return ctx.panel.empty()
        s=frame[kind].reindex(ctx.panel.dates).astype(float)
        if kind in ('volume','amount'):s=s.diff()
        ret=pd.DataFrame(ctx.ret_clean(),index=ctx.panel.dates)
        return ret.rolling(60,min_periods=45).corr(s).clip(-1,1).to_numpy(dtype=np.float64)
    globals()[name]=compute

for ds,(ident,fields) in DAILY.items():
    start='2018-01-01' if ds=='dc_daily' else None
    for kind,source in [('pressure',['open','high','low','close']),('range',['high','low','close']),('volume',['vol'])]:
        add(ds,kind,source,start)
    for kind,source in [('amount',['amount']),('gap',['open','pre_close']),('premium',['close','avg_price']),
                        ('pct_change',['pct_change']),('swing',['swing']),('turnover_rate',['turnover_rate'])]:
        if set(source)<=set(fields):add(ds,kind,source,start)
for kind,fields in [('pressure',['open','high','low','close']),
                    ('weighted_pressure',['open','high','low','close','amount']),
                    ('volume_concentration',['vol']),('realized_vol',['pct_change']),('amplitude',['amplitude'])]:
    # 共同有效分钟筛选也是真实依赖，审计中明确记录。
    add('tdx_minute',kind,MINUTE_FIELDS)
