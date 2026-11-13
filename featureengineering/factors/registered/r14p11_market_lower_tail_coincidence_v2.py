"""Version2 strict actual return endpoints; v1 immutable, GC no alpha claim."""
import numpy as np
from scipy.stats import rankdata
PARAMETERS={'window': 252, 'min_pairs': 200, 'quantile': 0.1, 'min_market_tail': 15, 'rank_method': 'average', 'market_index': '000300.SH', 'stock_return_endpoints': 'strict_actual_adjacent'}
def _market(ctx):
    days=np.asarray(ctx.panel.dates,dtype=np.int64)
    df=ctx.dataset("index_daily",columns=["ts_code","trade_date","pct_chg"],years=(int(days[0])//10000,int(days[-1])//10000))
    out=np.full(len(days),np.nan)
    if df is None or df.empty:return out
    df=df.loc[df.ts_code=="000300.SH"].copy()
    if df.empty:return out
    dates=ctx.date_col(df.trade_date)
    if len(np.unique(dates))!=len(dates):raise ValueError("Duplicate CSI300 date key")
    rows=np.searchsorted(days,dates);valid=rows<len(days);valid&=days[np.minimum(rows,len(days)-1)]==dates
    out[rows[valid]]=np.asarray(df.pct_chg,dtype=float)[valid]/100.
    return out

def estimate(r,tr,u,m,rows):
 out=np.full((len(rows),r.shape[1]),np.nan)
 for ti,t in enumerate(rows):
  rr=np.array(r[t-251:t+1],dtype=float);mm=np.broadcast_to(m[t-251:t+1,None],rr.shape).copy();good=np.isfinite(rr)&np.isfinite(mm)&tr[t-252:t]&tr[t-251:t+1];n=good.sum(axis=0)
  rr[~good]=np.nan;mm[~good]=np.nan
  lr=rankdata(rr,axis=0,method='average',nan_policy='omit')<=.1*(n+1)
  lm=rankdata(mm,axis=0,method='average',nan_policy='omit')<=.1*(n+1)
  tail=lm.sum(axis=0);use=(n>=200)&(tail>=15)&tr[t]&u[t]
  out[ti,use]=(lr&lm).sum(axis=0)[use]/tail[use]
 return out

def build(parameters):
    if parameters != PARAMETERS:raise ValueError("Only frozen strict-endpoint v2 parameters")
    def compute(ctx):
        n=len(ctx.panel.dates);out=np.full(ctx.panel.shape,np.nan)
        rows=np.arange(252,n)
        if len(rows):
            r=np.asarray(ctx.ret_clean(1),dtype=float);tr=np.asarray(ctx.traded(),dtype=bool)
            out[rows]=estimate(r,tr,ctx.universe,_market(ctx),rows)
        return out
    return compute
