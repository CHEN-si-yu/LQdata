"""Causal portfolio selection helpers, to be copied into an independent strategy unit."""
import numpy as np


def portfolio_return_history(px):
    adjusted=px.close*px.adj
    result=np.full_like(adjusted,np.nan,dtype=np.float64)
    with np.errstate(divide='ignore',invalid='ignore'):
        result[1:]=adjusted[1:]/adjusted[:-1]-1
    result[~np.isfinite(result)]=np.nan
    # Covariance estimation only: actual execution prices and returns remain unchanged.
    return np.clip(result,-.20,.20)


def portfolio_risk_information(pred,returns,days,lookback=63,min_observations=40,pool_size=100):
    information={}
    for p,d in zip(pred,days):
        d=int(d);known=np.flatnonzero(np.isfinite(p))
        ranked=known[np.argsort(-p[known],kind='stable')][:pool_size]
        window=returns[max(0,d-lookback+1):d+1,ranked]
        finite=np.isfinite(window);counts=finite.sum(axis=0)
        mean=np.nansum(window,axis=0)/np.maximum(counts,1)
        centered=np.where(finite,window-mean,0.)
        norm=np.sqrt((centered*centered).sum(axis=0))
        usable=(counts>=min_observations)&(norm>1e-8)
        ids=ranked[usable];vectors=centered[:,usable];lengths=norm[usable]
        corr=(vectors.T@vectors)/np.outer(lengths,lengths) if len(ids) else np.empty((0,0))
        corr=np.clip(corr,-1.,1.)
        # Fixed shrinkage suppresses noisy off-diagonal estimates; negative correlation is not penalized.
        positive=np.maximum(.8*corr+.2*np.eye(len(ids)),0.).astype(np.float32)
        information[d]=dict(stock_to_index={int(c):i for i,c in enumerate(ids)},
            positive_correlation=positive,alpha_priority=(1-np.flatnonzero(usable)/max(len(ranked),1)).astype(np.float32),
            pool_size=len(ranked),usable_size=len(ids),first_history_day=max(0,d-lookback+1),last_history_day=d)
    return information


def portfolio_diverse_fresh(fresh,keepers,information,strength=.25,slots=5):
    """Choose fresh entries conditional on carried positions; do not replace the existing sell/fee rules."""
    if strength==0 or slots<=0:return fresh
    lookup=information['stock_to_index']
    # An unmeasurable carried position must not be treated as safely uncorrelated.
    if any(int(c) not in lookup for c in keepers):return fresh
    candidates=[int(c) for c in fresh if int(c) in lookup]
    if not candidates:return fresh
    corr=information['positive_correlation'];alpha=information['alpha_priority']
    carried=[lookup[int(c)] for c in keepers]
    penalty=corr[:,carried].sum(axis=1) if carried else np.zeros(len(alpha),np.float32)
    count=len(carried);available=np.zeros(len(alpha),bool)
    for c in candidates:available[lookup[c]]=True
    reverse={v:k for k,v in lookup.items()};chosen=[]
    for _ in range(min(slots,len(candidates))):
        score=alpha-strength*(penalty/count if count else 0.)
        index=int(np.argmax(np.where(available,score,-np.inf)))
        chosen.append(reverse[index]);available[index]=False
        penalty+=corr[:,index];count+=1
    chosen_set=set(chosen)
    return chosen+[int(c) for c in fresh if int(c) not in chosen_set]
