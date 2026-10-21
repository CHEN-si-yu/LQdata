"""F029 GC fixed definition: intangible amortization share of three D&A adjustments (annual FY)."""
import numpy as np
from research.families import r14f1_ccc_innovation_v2 as pit
PARAMETERS={'signal':'intangible_amortization_share','annual_fy_positive':True,'version':1}
FIELDS=('depr_fa_coga_dpba','amort_intang_assets','lt_amort_deferred_exp')

def event_values(data):
    # Annual design: only FY (Q4 = annual YTD) rows create state; interim
    # publications must not advance the frontier nor clear the FY state.
    if not data.empty:
        data=data[data.period%4==3]
    e=pit._frontier_events(data);v=pit._sorted_versions(data,FIELDS)
    c=e.code.to_numpy(dtype=np.int64);q=e.q.to_numpy(dtype=np.int64);cap=e.pit.to_numpy(dtype=np.int64)
    dep=pit._lookup(v,c,q,cap,FIELDS[0]);amort=pit._lookup(v,c,q,cap,FIELDS[1]);lt=pit._lookup(v,c,q,cap,FIELDS[2])
    total=dep+amort+lt
    ok=np.isfinite(dep)&np.isfinite(amort)&np.isfinite(lt)&(dep>0)&(amort>0)&(lt>0)&(total>1e6)
    with np.errstate(divide='ignore',invalid='ignore'):
        values=np.where(ok,amort/total,np.nan)
    return e,values

def build(parameters=PARAMETERS):
    if parameters!=PARAMETERS:raise ValueError('Frozen F029 v1 only')
    def compute(ctx):
        data=pit._report_table(ctx,'stock_cashflow',FIELDS)
        e,v=event_values(data)
        return pit._map_fresh(ctx,e,v)
    return compute
