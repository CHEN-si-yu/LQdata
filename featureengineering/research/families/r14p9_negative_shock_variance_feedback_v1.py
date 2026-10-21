"""Frozen P009 GC: historical asymmetric squared-return response, no alpha claim."""
import numpy as np
PARAMETERS={'window': 252, 'min_pairs': 200, 'min_each_sign': 60, 'max_condition': 100000000.0, 'scale': 0.0001}

def estimate(r,tr,universe,rows):
    out=np.full((len(rows),r.shape[1]),np.nan)
    for ti,t in enumerate(rows):
        rr=np.asarray(r[t-252:t],dtype=float);yy=np.asarray(r[t-251:t+1],dtype=float)**2/1e-4
        good=np.isfinite(rr)&np.isfinite(yy)&tr[t-252:t]&tr[t-251:t+1]
        x=np.where(good,rr*rr/1e-4,0);z=x*(rr<0);y=np.where(good,yy,0);n=good.sum(axis=0)
        eligible=(n>=200)&((good&(rr<0)).sum(axis=0)>=60)&((good&(rr>0)).sum(axis=0)>=60)&tr[t]&universe[t]
        sx=x.sum(axis=0);sz=z.sum(axis=0);sy=y.sum(axis=0)
        A=np.zeros((r.shape[1],3,3));B=np.stack((sy,(x*y).sum(axis=0),(z*y).sum(axis=0)),axis=1)
        A[:,0,0]=n;A[:,0,1]=A[:,1,0]=sx;A[:,0,2]=A[:,2,0]=sz
        A[:,1,1]=(x*x).sum(axis=0);A[:,2,2]=(z*z).sum(axis=0);A[:,1,2]=A[:,2,1]=(x*z).sum(axis=0)
        idx=np.flatnonzero(eligible)
        if not len(idx):continue
        aa=A[idx];nn=n[idx];varx=aa[:,1,1]-sx[idx]**2/nn;varz=aa[:,2,2]-sz[idx]**2/nn;cov=aa[:,1,2]-sx[idx]*sz[idx]/nn
        with np.errstate(divide='ignore',invalid='ignore'):rho=cov/np.sqrt(varx*varz);condition=np.sqrt((1+np.abs(rho))/(1-np.abs(rho)))
        goodcond=np.isfinite(condition)&(condition<1e8)&(varx>0)&(varz>0)
        idx=idx[goodcond]
        if len(idx):out[ti,idx]=np.linalg.solve(A[idx],B[idx,:,None])[:,2,0]
    return out

def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only frozen P009 parameters accepted")
    def compute(ctx):
        n = len(ctx.panel.dates)
        rows = np.arange(253,n)
        out = np.full(ctx.panel.shape,np.nan)
        if len(rows):
            out[rows] = estimate(ctx.ret_clean(1),ctx.traded(),ctx.universe,rows)
        return out
    return compute
