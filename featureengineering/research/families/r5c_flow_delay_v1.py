from __future__ import annotations
import numpy as np

def _shift(a, n=1):
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if len(a) > n:
        out[n:] = a[:-n]
    return out

def _fit(x, y, n, mc, ridge=0.01):
    """Inclusive trailing complete-case fit; coefficients in original units.
    Fixed standardized ridge: penalty=0.01*count. No future shifts.
    Rolling sums use direct window sums to keep start-independent rounding.
    """
    nt, nc, npred = x.shape
    coef = np.full((nt,nc,npred+1),np.nan)
    means = np.full((nt,nc,npred),np.nan)
    r2 = np.full((nt,nc),np.nan)
    for t in range(n-1,nt):
        xx=x[t-n+1:t+1]; yy=y[t-n+1:t+1]
        valid=np.isfinite(yy)&np.isfinite(xx).all(axis=2)
        count=valid.sum(axis=0).astype(float)
        z=np.where(valid[:,:,None],xx,0.)
        v=np.where(valid,yy,0.)
        den=np.maximum(count,1.)
        mux=z.sum(axis=0)/den[:,None]; muy=v.sum(axis=0)/den
        sxx=np.einsum('tcp,tcq->cpq',z,z)-count[:,None,None]*mux[:,:,None]*mux[:,None,:]
        sxy=np.einsum('tcp,tc->cp',z,v)-count[:,None]*mux*muy[:,None]
        syy=(v*v).sum(axis=0)-count*muy*muy
        diag=np.diagonal(sxx,axis1=1,axis2=2)
        sd=np.sqrt(np.maximum(diag/den[:,None],0.))
        good=(count>=mc)&(sd>1e-8).all(axis=1)&(syy>1e-12)
        ids=np.flatnonzero(good)
        if not len(ids): continue
        norm=sxx[ids]/sd[ids,:,None]/sd[ids,None,:]
        rhs=sxy[ids]/sd[ids]
        if ridge:
            norm=norm+ridge*count[ids,None,None]*np.eye(npred)[None,:,:]
        else:
            nonsingular=np.linalg.eigvalsh(norm)[:,0]>1e-8
            ids=ids[nonsingular]; norm=norm[nonsingular]; rhs=rhs[nonsingular]
        if not len(ids): continue
        b=np.linalg.solve(norm,rhs[:,:,None])[:,:,0]/sd[ids]
        coef[t,ids,1:]=b
        coef[t,ids,0]=muy[ids]-(b*mux[ids]).sum(axis=1)
        means[t,ids]=mux[ids]
        sse=syy[ids]-2*(b*sxy[ids]).sum(axis=1)+np.einsum('cp,cpq,cq->c',b,sxx[ids],b)
        r2[t,ids]=1-sse/syy[ids]
    return coef,means,r2

def _residual(r, universe):
    valid=universe&np.isfinite(r)
    count=valid.sum(axis=1)
    m=np.where(valid,r,0.).sum(axis=1)/np.maximum(count,1)
    m[count<100]=np.nan
    x=np.broadcast_to(m[:,None,None],r.shape+(1,))
    fit,_,_=_fit(x,r,60,45,0.)
    fit=_shift(fit)
    e=r-fit[:,:,0]-fit[:,:,1]*m[:,None]
    return np.where(valid,e,np.nan)

SIGNAL='flow_innovation_delayed_oo_forecast'

def _flow(ctx):
    cols=['stock_code','trade_date']+[f'{side}_{tier}_amount' for side in ('buy','sell') for tier in ('sm','md','lg','elg')]
    dates=ctx.panel.dates
    df=ctx.dataset('stock_main_fund_flow',columns=cols,years=(int(dates[0])//10000,int(dates[-1])//10000))
    if df is None or df.empty: return np.full(ctx.panel.shape,np.nan)
    if df.duplicated(['stock_code','trade_date']).any():
        raise ValueError('Duplicate stock_main_fund_flow stock/date keys')
    raw=df[cols[2:]].to_numpy(dtype=np.float64)
    den=raw.sum(axis=1)
    num=(df.buy_lg_amount+df.buy_elg_amount-df.sell_lg_amount-df.sell_elg_amount).to_numpy(dtype=np.float64)
    value=np.divide(num,den,out=np.full(len(den),np.nan),where=np.isfinite(raw).all(axis=1)&(den>0))
    codes=ctx.code_index(df.stock_code.to_numpy())
    days=ctx.date_col(df.trade_date)
    hit=codes>=0
    grid=ctx.panel.place(codes[hit],days[hit],value[hit],fill=np.nan,dtype=np.float64)
    return np.where(ctx.traded()&ctx.universe,grid,np.nan)

def _compute_arrays(r, opens, traded, universe, f):
    e=_residual(r,universe)
    x=np.stack((_shift(f),e),axis=2)
    coef,_,_=_fit(x,f,60,45,0.01)
    coef=_shift(coef)
    u=f-coef[:,:,0]-(coef[:,:,1:]*x).sum(axis=2)
    # A row at maturity day d targets open(d)/open(d-1)-1 and X_(d-2).
    # At T the last usable X is T-2, never T-1 or T.
    prev=_shift(opens)
    good=traded&(_shift(traded.astype(float))==1)&(opens>0)&(prev>0)
    mature_y=np.divide(opens,prev,out=np.full(opens.shape,np.nan),where=good)-1.
    train_x=np.stack((_shift(e,2),_shift(u,2),_shift(u,3)),axis=2)
    coef,means,_=_fit(train_x,mature_y,126,95,0.01)
    current=np.stack((e,u,_shift(u)),axis=2)
    value=(coef[:,:,2:]*(current[:,:,1:]-means[:,:,1:])).sum(axis=2)
    return np.where(universe&traded&np.isfinite(current).all(axis=2)&np.isfinite(value),value,np.nan)

def build(parameters):
    if parameters!={'signal':SIGNAL}: raise ValueError('Only frozen C signal permitted')
    def compute(ctx):
        r=np.asarray(ctx.ret_clean(1),dtype=np.float64)
        tr=np.asarray(ctx.traded(),dtype=bool)
        un=np.asarray(ctx.universe,dtype=bool)
        r=np.where(un&tr,r,np.nan)
        return _compute_arrays(r,np.asarray(ctx.hfq('open'),dtype=np.float64),tr,un,_flow(ctx))
    return compute
