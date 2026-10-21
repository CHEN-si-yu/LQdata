"""F028 GC fixed definition: tax refund share of two-way tax cash flows (strict TTM)."""
import numpy as np
from research.families import r14f1_ccc_innovation_v2 as pit
PARAMETERS={'signal':'tax_cash_refund_share','raw_ytd_positive':True,'version':1}
FIELDS=('recp_tax_rends','c_paid_for_taxes')

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
    refund=positive_ttm(v,c,q,cap,FIELDS[0]);paid=positive_ttm(v,c,q,cap,FIELDS[1])
    total=refund+paid
    ok=np.isfinite(refund)&np.isfinite(paid)&(refund>0)&(paid>0)&(total>1e6)
    with np.errstate(divide='ignore',invalid='ignore'):
        values=np.where(ok,refund/total,np.nan)
    return e,values

def build(parameters=PARAMETERS):
    if parameters!=PARAMETERS:raise ValueError('Frozen F028 v1 only')
    def compute(ctx):
        data=pit._report_table(ctx,'stock_cashflow',FIELDS)
        e,v=event_values(data)
        return pit._map_fresh(ctx,e,v)
    return compute
