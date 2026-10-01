"""Causal equal-slot mean/variance utility; copied into independent strategy scripts."""
import numpy as np


def variance_return_history(px):
    adjusted=px.close*px.adj
    ret=np.full_like(adjusted,np.nan,dtype=np.float64)
    with np.errstate(divide='ignore',invalid='ignore'):
        ret[1:]=adjusted[1:]/adjusted[:-1]-1
    ret[~np.isfinite(ret)]=np.nan
    return np.clip(ret,-.20,.20)  # estimation only; executed prices stay unchanged


def variance_risk_information(pred,returns,days,lookback=63,min_observations=40,pool_size=100,shrinkage=.2):
    information={}
    for p,d in zip(pred,days):
        d=int(d);known=np.flatnonzero(np.isfinite(p))
        ranked=known[np.argsort(-p[known],kind='stable')][:pool_size]
        first=max(0,d-lookback+1);window=returns[first:d+1,ranked]
        finite=np.isfinite(window);counts=finite.sum(axis=0)
        means=np.nansum(window,axis=0)/np.maximum(counts,1)
        centered=np.where(finite,window-means,0.)
        # Gram normalization preserves PSD and each observed sample variance.
        usable=(counts>=min_observations)&((centered*centered).sum(axis=0)>1e-16)
        ids=ranked[usable];vectors=centered[:,usable]/np.sqrt(counts[usable]-1)
        covariance=vectors.T@vectors
        covariance=(1-shrinkage)*covariance+shrinkage*np.diag(np.diag(covariance))
        information[d]=dict(stock_to_index={int(c):i for i,c in enumerate(ids)},covariance=covariance,
            pool_size=len(ranked),usable_size=len(ids),first_history_day=first,last_history_day=d,
            min_observations=int(min_observations))
    return information


def variance_choose_fresh(fresh,carried,information,utility,risk_aversion=3.,n=5,forecast_members=1,slots=None):
    """Greedy marginal equal-slot expected utility; empty target slots remain cash."""
    if risk_aversion==0 or n==1:return fresh
    slots=max(0,n-len(carried)) if slots is None else slots
    if slots<=0:return []
    lookup=information['stock_to_index']
    if any(int(c) not in lookup for c in carried):return fresh  # no fake zero covariance
    candidates=[int(c) for c in fresh if int(c) in lookup and int(c) not in carried]
    if not candidates:return []
    covariance=information['covariance'];chosen=[];indices=[lookup[int(c)] for c in carried]
    for _ in range(min(slots,len(candidates))):
        ix=np.asarray([lookup[c] for c in candidates],dtype=int)
        incremental=np.diag(covariance)[ix].copy()
        if indices:incremental+=2*covariance[np.ix_(ix,indices)].sum(axis=1)
        score=np.asarray([utility[c] for c in candidates])-forecast_members*risk_aversion/n*incremental
        winner=int(np.argmax(score))
        if not np.isfinite(score[winner]) or score[winner]<0:break
        stock=candidates.pop(winner);chosen.append(stock);indices.append(lookup[stock])
    return chosen
