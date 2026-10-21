"""Version2 strict actual return endpoints; v1 immutable, GC no alpha claim."""
import numpy as np
PARAMETERS={'window': 126, 'min_pairs': 100, 'min_each_excess_sign': 20, 'market_index': '000300.SH', 'amount_scale_cny': 1000000.0, 'stock_return_endpoints': 'strict_actual_adjacent'}
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

def estimate(r,tr,univ,amount,m,rows):
 out=np.full((len(rows),r.shape[1]),np.nan)
 for ti,t in enumerate(rows):
  rr=np.asarray(r[t-126:t],dtype=float);ee=rr-m[t-126:t,None];y=np.asarray(r[t-125:t+1],dtype=float)-m[t-125:t+1,None];q=np.sign(ee)*np.asarray(amount[t-126:t],dtype=float)/1e6
  good=np.isfinite(rr)&np.isfinite(ee)&np.isfinite(y)&np.isfinite(q)&(amount[t-126:t]>0)&tr[t-127:t-1]&tr[t-126:t]&tr[t-125:t+1]
  n=good.sum(axis=0);use=(n>=100)&((good&(ee>0)).sum(axis=0)>=20)&((good&(ee<0)).sum(axis=0)>=20)&tr[t]&univ[t]
  xx=np.where(good,rr,0);zz=np.where(good,q,0);yy=np.where(good,y,0);den=np.maximum(n,1)
  xx=np.where(good,xx-xx.sum(axis=0)/den,0);zz=np.where(good,zz-zz.sum(axis=0)/den,0);yy=np.where(good,yy-yy.sum(axis=0)/den,0)
  vx=(xx*xx).sum(axis=0);vz=(zz*zz).sum(axis=0);cxz=(xx*zz).sum(axis=0);cxy=(xx*yy).sum(axis=0);czy=(zz*yy).sum(axis=0)
  with np.errstate(divide='ignore',invalid='ignore'):
   rho=cxz/np.sqrt(vx*vz);residual_var=vz-cxz*cxz/vx;gamma=(czy-cxz*cxy/vx)/residual_var
  # Numerical full-rank gate on standardized [1,r,q] Gram eigenvalues.
  use&=(vx>0)&(vz>0)&np.isfinite(gamma)&((1-np.abs(rho))>3*np.finfo(float).eps*(1+np.abs(rho)))
  out[ti,use]=-gamma[use]
 return out

def build(parameters):
    if parameters != PARAMETERS:raise ValueError("Only frozen strict-endpoint v2 parameters")
    def compute(ctx):
        n=len(ctx.panel.dates);out=np.full(ctx.panel.shape,np.nan)
        rows=np.arange(127,n)
        if len(rows):
            r=np.asarray(ctx.ret_clean(1),dtype=float);tr=np.asarray(ctx.traded(),dtype=bool)
            out[rows]=estimate(r,tr,ctx.universe,ctx.px("amount"),_market(ctx),rows)
        return out
    return compute
