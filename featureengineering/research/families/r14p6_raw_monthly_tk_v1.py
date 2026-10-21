"""P006 raw-return CPT robustness branch; requires verified full calendar month ends."""
import numpy as np
PARAMETERS=dict(months=60,value_power=.88,loss_aversion=2.25,gain_weight=.61,loss_weight=.69,return_basis='raw_hfq_monthly')
def weight(p,d):
    return p**d/(p**d+(1-p)**d)**(1/d)
def tk(x):
    x=np.sort(x,axis=0); n=x.shape[0]; j=np.arange(1,n+1)[:,None]
    loss=weight(j/n,.69)-weight((j-1)/n,.69)
    gain=weight((n+1-j)/n,.61)-weight((n-j)/n,.61)
    return np.sum(np.where(x<0,-2.25*np.abs(x)**.88*loss,np.abs(x)**.88*gain),axis=0)
def calculate(days,close,traded,universe,calendar):
    ends=calendar[:-1][calendar[:-1]//100!=calendar[1:]//100]
    ends=ends[(ends>=days[0])&(ends<=days[-1])]
    rows=np.searchsorted(days,ends)
    if not np.array_equal(days[rows],ends): raise ValueError('Missing calendar month end')
    # Calendar endpoints, not merely observed final rows; consecutive monthly returns.
    monthly=close[rows[1:]]/close[rows[:-1]]-1
    good=traded[rows[1:]]&traded[rows[:-1]]&np.isfinite(monthly)&(close[rows[:-1]]>0)&(close[rows[1:]]>0)
    monthly=np.where(good,monthly,np.nan)
    out=np.full(close.shape,np.nan)
    for k in range(60,len(rows)):
        x=monthly[k-60:k]; value=tk(x)
        stop=rows[k+1] if k+1<len(rows) else len(days)
        out[rows[k]:stop]=value
    out[~(traded&universe)]=np.nan
    return out,ends


def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only frozen P006 parameters accepted")
    def compute(ctx):
        days=np.asarray(ctx.panel.dates,dtype=np.int64)
        calendar=np.asarray(ctx.cal.days,dtype=np.int64)
        if np.any(np.diff(calendar)<=0) or np.any(np.diff(days)<=0):
            raise ValueError("Calendar and panel dates must increase strictly")
        value,_=calculate(days,ctx.hfq("close"),ctx.traded(),ctx.universe,calendar)
        return value
    return compute
