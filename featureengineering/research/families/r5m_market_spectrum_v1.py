"""R5-M one strict candidate: acceleration in full-pool correlation-Gram concentration.
S_t = c_t-c_(t-5), c=lambda_max(X X.T)/N over 60 days. Direction negative.
Missing centered entries have zero inner-product contribution by definition;
this is NOT a pairwise-complete correlation matrix or zero-filled traded return.
"""
from __future__ import annotations
import numpy as np
SIGNAL='spectral_crowding_accel5_60'

def concentration(window,min_names=500):
    w=np.asarray(window,dtype=np.float64)
    if w.ndim!=2 or w.shape[0]!=60:raise ValueError('R5M requires 60 complete date rows')
    good=np.isfinite(w)
    keep=(good.sum(axis=0)>=54)&good[-1]
    if int(keep.sum())<min_names:return np.nan
    w=w[:,keep];good=np.isfinite(w)
    means=np.sum(np.where(good,w,0.0),axis=0)/good.sum(axis=0)
    centered=np.where(good,w-means,0.0)
    norm=np.sqrt(np.sum(centered*centered,axis=0))
    use=np.isfinite(norm)&(norm>0)
    if int(use.sum())<min_names:return np.nan
    x=centered[:,use]/norm[use]
    gram=x@x.T
    top=float(np.linalg.eigvalsh(gram)[-1])
    return top/x.shape[1]

def spectrum_path(returns,min_names=500):
    r=np.asarray(returns,dtype=np.float64)
    c=np.full(r.shape[0],np.nan)
    for t in range(59,len(r)):c[t]=concentration(r[t-59:t+1],min_names)
    out=np.full(c.shape,np.nan)
    out[5:]=c[5:]-c[:-5]
    return out,c

def build(parameters):
    if parameters!={'signal':SIGNAL}:raise ValueError('R5M has one frozen strict candidate')
    def compute(ctx):
        returns=np.asarray(ctx.ret_clean(1),dtype=np.float64)
        returns=np.where(np.asarray(ctx.universe,dtype=bool),returns,np.nan)
        out,_=spectrum_path(returns,500)
        return out
    return compute
