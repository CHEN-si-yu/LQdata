"""F027 GC fixed definition: supplier prepayment vs trade-credit direction share."""
import numpy as np
from research.families import r14f1_ccc_innovation_v2 as pit
PARAMETERS={'signal':'supplier_prepayment_direction_share','raw_balance_positive':True,'version':1}
FIELDS=('prepayment','acct_payable')

def event_values(data):
    e=pit._frontier_events(data);v=pit._sorted_versions(data,FIELDS)
    c=e.code.to_numpy(dtype=np.int64);q=e.q.to_numpy(dtype=np.int64);cap=e.pit.to_numpy(dtype=np.int64)
    prepay=pit._lookup(v,c,q,cap,FIELDS[0]);payable=pit._lookup(v,c,q,cap,FIELDS[1])
    total=prepay+payable
    ok=np.isfinite(prepay)&np.isfinite(payable)&(prepay>0)&(payable>0)&(total>1e6)
    with np.errstate(divide='ignore',invalid='ignore'):
        values=np.where(ok,prepay/total,np.nan)
    return e,values

def build(parameters=PARAMETERS):
    if parameters!=PARAMETERS:raise ValueError('Frozen F027 v1 only')
    def compute(ctx):
        data=pit._report_table(ctx,'stock_balancesheet',FIELDS)
        e,v=event_values(data)
        return pit._map_fresh(ctx,e,v)
    return compute
