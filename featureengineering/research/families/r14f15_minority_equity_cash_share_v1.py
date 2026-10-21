"""F015 GC fixed definition: minority channel of external equity cash funding."""
import numpy as np
from research.families import r14f1_ccc_innovation_v2 as pit
PARAMETERS={'signal':'minority_external_equity_cash_share','raw_ytd_positive':True,'version':1}
FIELDS=('incl_cash_rec_saims','c_recp_cap_contrib')

def positive_ttm(v,code,q,cap,field):
    phase=q%4
    current=pit._lookup(v,code,q,cap,field)
    fy=pit._lookup(v,code,q-(phase+1),cap,field)
    prior=pit._lookup(v,code,q-4,cap,field)
    valid=np.isfinite(current)&(current>0)&np.where(phase==3,True,np.isfinite(fy)&(fy>0)&np.isfinite(prior)&(prior>0))
    value=np.where(phase==3,current,current+fy-prior)
    return np.where(valid&np.isfinite(value)&(value>0),value,np.nan)

def event_values(data):
    e=pit._frontier_events(data);v=pit._sorted_versions(data,FIELDS)
    c=e.code.to_numpy(dtype=np.int64);q=e.q.to_numpy(dtype=np.int64);cap=e.pit.to_numpy(dtype=np.int64)
    minor=positive_ttm(v,c,q,cap,FIELDS[0]);total=positive_ttm(v,c,q,cap,FIELDS[1])
    ok=np.isfinite(minor)&np.isfinite(total)&(total>1e6)&(minor>0)&(minor<=total)
    with np.errstate(divide='ignore',invalid='ignore'):
        values=np.where(ok,minor/total,np.nan)
    return e,values

def build(parameters=PARAMETERS):
    if parameters!=PARAMETERS:raise ValueError('Frozen F015 v1 only')
    def compute(ctx):
        data=pit._report_table(ctx,'stock_cashflow',FIELDS)
        e,v=event_values(data)
        return pit._map_fresh(ctx,e,v)
    return compute
