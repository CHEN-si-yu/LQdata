"""R5-E frozen downside-shock recovery. Direction +, medium/weak prior.
Only ordered, nonoverlapping same-session events; previous trading-day BPV scale.
No labels or production writes. Source contract: delivery/e_intraday_recovery.
"""
from __future__ import annotations
import logging
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

log=logging.getLogger(__name__)
SIGNAL="downshock_recovery20"

def day_metrics(close, minute, previous_sigma):
    """Return current sigma, event sum/count, and source-valid flag for one stock/day."""
    c=np.asarray(close,dtype=np.float64)
    m=np.asarray(minute,dtype=np.int32)
    if c.size < 41 or c.size != m.size or np.any(np.diff(m)<=0):
        return np.nan,np.nan,np.nan,False
    allowed=((m>=570)&(m<=690))|((m>=780)&(m<=900))
    session=np.where(m<=690,0,1)
    adjacent=(np.diff(m)==5)&(session[1:]==session[:-1])
    price=np.isfinite(c)&(c>0)&allowed
    r=np.full(c.size,np.nan)
    use=adjacent&price[1:]&price[:-1]
    r[1:][use]=np.log(c[1:][use]/c[:-1][use])
    finite=np.isfinite(r)
    n=int(finite.sum())
    if n<40:
        return np.nan,np.nan,np.nan,False
    pair=finite[1:]&finite[:-1]
    bpv=np.pi/2*float(np.sum(np.abs(r[1:][pair]*r[:-1][pair])))
    sigma=float(np.sqrt(bpv/n)) if bpv>0 else np.nan
    if not np.isfinite(previous_sigma) or previous_sigma<=0:
        return sigma,np.nan,np.nan,True
    threshold=-2.0*previous_sigma
    total=0.0; count=0; j=1
    while j+3<c.size:
        if finite[j] and r[j]<=threshold and np.all(finite[j+1:j+4]) and np.all(np.diff(m[j:j+4])==5) and session[j]==session[j+3]:
            recovery=np.log(c[j+3]/c[j])/abs(r[j])
            total+=float(np.clip(recovery,-1.0,1.0));count+=1;j+=4
        else:
            j+=1
    return sigma,total,float(count),True

def _arrow(series):
    arr=getattr(series,"array",None)
    raw=getattr(arr,"_pa_array",None)
    return raw if raw is not None else pa.array(series)

def _groups(ctx,frame):
    """Explicit stock AND day boundaries; reject split/repeated stock/day keys."""
    codes=_arrow(frame['stock_code']); tt=_arrow(frame['trade_time'])
    d=pc.utf8_slice_codeunits(tt,0,10)
    size=len(frame)
    if not size:return None
    change=pc.or_(pc.not_equal(codes.slice(1,size-1),codes.slice(0,size-1)),pc.not_equal(d.slice(1,size-1),d.slice(0,size-1)))
    boundary=np.asarray(pc.fill_null(change,True).to_numpy(zero_copy_only=False),dtype=bool)
    starts=np.flatnonzero(np.r_[True,boundary]);ends=np.r_[starts[1:],size]
    cp=ctx.code_index(frame['stock_code'].iloc[starts].to_numpy())
    days=ctx.date_col(frame['trade_time'].iloc[starts].str.slice(0,10))
    keep=cp>=0
    keys=cp[keep].astype(np.int64)*100000000+days[keep]
    if np.unique(keys).size!=keys.size:raise ValueError('R5E: repeated or split stock/date groups')
    h=pc.cast(pc.utf8_slice_codeunits(tt,11,13),pa.int16())
    minute=pc.add(pc.multiply(h,pa.scalar(60,pa.int16())),pc.cast(pc.utf8_slice_codeunits(tt,14,16),pa.int16()))
    minute=np.asarray(minute.to_numpy(zero_copy_only=False),dtype=np.int16)
    return starts,ends,np.asarray(cp),np.asarray(days),minute

def build(parameters):
    if parameters!={'signal':SIGNAL}:raise ValueError('R5E has one frozen candidate')
    def compute(ctx):
        dates=np.asarray(ctx.panel.dates,dtype=np.int32)
        numerator=ctx.panel.empty(dtype=np.float64);count=ctx.panel.empty(dtype=np.float64)
        daypos={int(d):i for i,d in enumerate(dates)}
        last={}
        for year in range(int(dates[0])//10000,int(dates[-1])//10000+1):
            frame=ctx.dataset('stock_history_5min',columns=['stock_code','trade_time','close'],years=(year,year))
            if frame is None or frame.empty:continue
            starts,ends,cp,days,minute=_groups(ctx,frame)
            close=frame['close'].to_numpy(dtype=np.float64)
            accepted=invalid=events=0
            for a,b,c,d in zip(starts,ends,cp,days):
                ti=daypos.get(int(d))
                if c<0 or ti is None:continue
                c=int(c)
                previous=last.get(c)
                if previous is not None and ti<=previous[0]:raise ValueError('R5E: dates not increasing within stock')
                prior=previous[1] if previous is not None and previous[0]==ti-1 else np.nan
                sigma,total,n,valid=day_metrics(close[a:b],minute[a:b],prior)
                last[c]=(ti,sigma)
                if valid and np.isfinite(total):
                    numerator[ti,c]=total;count[ti,c]=n;accepted+=1;events+=int(n)
                else:invalid+=1
            log.info('r5e year=%s valid_base_days=%d invalid_or_no_prior=%d events=%d',year,accepted,invalid,events)
            del frame,close,minute,starts,ends,cp,days
        sums=ctx.roll_sum(numerator,20,min_count=15)
        counts=ctx.roll_sum(count,20,min_count=15)
        out=np.full(sums.shape,np.nan)
        np.divide(sums,counts,out=out,where=np.isfinite(sums)&np.isfinite(counts)&(counts>=5))
        return np.where(np.asarray(ctx.universe,dtype=bool)&np.isfinite(out),out,np.nan)
    return compute
