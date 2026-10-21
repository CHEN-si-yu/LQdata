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

SIGNAL='residual_peer_delay_126'

def _graph(e,codes):
    valid=np.isfinite(e); mask=valid.astype(np.float64); z=np.where(valid,e,0.)
    n=mask.T@mask
    sx=z.T@mask; sx2=(z*z).T@mask
    cross=z.T@z
    den=np.maximum(n,1.)
    cov=cross-sx*sx.T/den
    vx=sx2-sx*sx/den
    vv=vx*vx.T
    corr=np.divide(cov,np.sqrt(np.maximum(vv,0.)),out=np.full(n.shape,np.nan),where=(n>=95)&(vv>1e-12))
    np.fill_diagonal(corr,np.nan)
    rows=[]
    code_order=np.argsort(np.asarray(codes).astype(str),kind='stable')
    for i in range(len(codes)):
        ids=code_order[np.isfinite(corr[i,code_order])&(corr[i,code_order]>0)&(corr[i,code_order]<=1+1e-10)]
        ids=ids[np.argsort(-corr[i,ids],kind='stable')][:10]
        rows.append(ids if len(ids)>=5 else np.empty(0,dtype=int))
    return rows

def _peer(e,dates,codes):
    p=np.full(e.shape,np.nan)
    graph=None
    for t in range(len(e)):
        if t==0 or int(dates[t])//100 != int(dates[t-1])//100:
            graph=_graph(e[t-126:t],codes) if t>=126 else None
        if graph is None: continue
        for i,ids in enumerate(graph):
            if not len(ids): continue
            vals=e[t,ids]; hit=np.isfinite(vals)
            if hit.sum()>=max(5,int(np.ceil(.7*len(ids)))):
                p[t,i]=vals[hit].mean()
    return p

def _compute_arrays(r,traded,universe,dates,codes):
    e=_residual(np.where(universe&traded,r,np.nan),universe)
    p=_peer(e,dates,codes)
    xx=np.stack((p,_shift(p),_shift(p,2),_shift(p,3)),axis=2)
    same=np.isfinite(e)&np.isfinite(xx).all(axis=2)
    yy=np.where(same,e,np.nan)
    xx=np.where(same[:,:,None],xx,np.nan)
    _,_,restricted=_fit(xx[:,:,:1],yy,126,95,0.)
    _,_,full=_fit(xx,yy,126,95,0.)
    # Fits use rows through T-1 only; current daily p does not enter delay.
    restricted=_shift(restricted); full=_shift(full)
    value=np.divide(restricted,full,out=np.full(e.shape,np.nan),where=full>1e-4)
    value=1.-value
    ok=(value>=-1e-10)&(value<=1+1e-10)
    return np.where(universe&traded&ok,np.clip(value,0.,1.),np.nan)

def build(parameters):
    if parameters!={'signal':SIGNAL}: raise ValueError('Only frozen D signal permitted')
    def compute(ctx):
        return _compute_arrays(np.asarray(ctx.ret_clean(1),dtype=np.float64),np.asarray(ctx.traded(),dtype=bool),np.asarray(ctx.universe,dtype=bool),ctx.panel.dates,ctx.panel.codes)
    return compute
