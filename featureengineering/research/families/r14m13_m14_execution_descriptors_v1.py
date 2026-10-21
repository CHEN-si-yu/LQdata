"""Frozen M013/M014 GC descriptors; source visibility exclusively ctx.dataset.
No disk/cache reads. Each annual frame is released and upstream cache cleared.
"""
from collections import deque
import gc
import numpy as np

SIGNALS=('five_cent_endpoint_cluster_20','volume_schedule_deviation_20')
COLUMNS=['stock_code','trade_time','close','vol']
GRID=tuple([f'{h:02d}:{m:02d}:00' for h in range(9,12) for m in range(0,60,5) if '09:35'<=f'{h:02d}:{m:02d}'<='11:30']+[f'{h:02d}:{m:02d}:00' for h in range(13,16) for m in range(0,60,5) if '13:05'<=f'{h:02d}:{m:02d}'<='15:00'])

def endpoint_counts(close):
    c=np.asarray(close,dtype=np.float64)
    if c.shape!=(48,) or not np.isfinite(c).all() or (c<=0).any():return None
    ticks=np.rint(c/.01)
    if np.max(np.abs(c/.01-ticks))>1e-6:return None
    ix=np.r_[np.arange(1,24),np.arange(25,48)]
    ix=ix[c[ix]!=c[ix-1]]
    return int(np.sum(ticks[ix].astype(np.int64)%5==0)),len(ix)

def share(v):
    v=np.asarray(v,dtype=np.float64)
    if v.shape!=(48,) or not np.isfinite(v).all() or (v<0).any() or v.sum()<=0:return None
    return v/v.sum()

def distance(p,h):
    # Identical operation order to the frozen source probe.
    return float(np.abs(np.cumsum(p)[:-1]-np.cumsum(np.asarray(h,dtype=np.float64).mean(axis=0))[:-1]).mean())

class Accumulator:
    def __init__(self,days,codes):
        self.days=np.asarray(days,dtype=np.int64);self.codes=np.asarray(codes).astype(str)
        if len(self.days)<21 or np.any(np.diff(self.days)<=0):raise ValueError('Need ordered calendar with20 prior rows')
        if len(set(self.codes))!=len(self.codes):raise ValueError('Duplicate stock axis')
        self.di={str(int(d))[:4]+'-'+str(int(d))[4:6]+'-'+str(int(d))[6:8]:k for k,d in enumerate(self.days)}
        self.ci={c:j for j,c in enumerate(self.codes)};self.slot={s:j for j,s in enumerate(GRID)}
        self.hist={c:deque() for c in self.codes};self.last={};self.key=None;self.rows=[]
        shape=(len(days)-20,len(codes));self.values=[np.full(shape,np.nan),np.full(shape,np.nan)]
    def add(self,c,tt,close,vol):
        if c not in self.ci or not isinstance(tt,str) or tt[:10] not in self.di:return
        key=(c,tt[:10])
        if key!=self.key:self.flush();self.key=key;self.rows=[]
        self.rows.append((self.slot.get(tt[11:],-1),close,vol))
    def flush(self):
        if self.key is None:return
        c,d=self.key;t=self.di[d];j=self.ci[c];rows=self.rows
        if c in self.last and t<=self.last[c]:raise ValueError('Repeated or nonmonotone stock-day')
        self.last[c]=t;valid=len(rows)==48 and sorted(z[0] for z in rows)==list(range(48));cc=p=None
        if valid:
            rows.sort();cc=endpoint_counts([z[1] for z in rows]);p=share([z[2] for z in rows])
        h=self.hist[c]
        while h and h[0][0]<t-20:h.popleft()
        prior=[z[2] for z in h if z[2] is not None]
        if t>=20:
            if p is not None and len(prior)>=15:self.values[1][t-20,j]=distance(p,prior)
            counts=[z[1] for z in h if z[0]>=t-19 and z[1] is not None]
            if cc is not None:counts.append(cc)
            total=sum(z[1] for z in counts)
            if cc is not None and len(counts)>=15 and total>=300:self.values[0][t-20,j]=sum(z[0] for z in counts)/total
        h.append((t,cc,p));self.key=None;self.rows=[]

def compute_both(ctx):
    dates=np.asarray(ctx.panel.dates,dtype=np.int64);calendar=np.asarray(ctx.cal.days,dtype=np.int64)
    if not len(dates) or np.any(np.diff(dates)<=0):raise ValueError('Invalid panel dates')
    lo=np.searchsorted(calendar,dates[0]);hi=np.searchsorted(calendar,dates[-1])
    if lo<20 or hi>=len(calendar) or not np.array_equal(calendar[lo:hi+1],dates):raise ValueError('Full official calendar and20 prior rows required')
    axis=calendar[lo-20:hi+1];acc=Accumulator(axis,ctx.panel.codes)
    for year in range(int(axis[0])//10000,int(axis[-1])//10000+1):
        frame=ctx.dataset('stock_history_5min',columns=COLUMNS,years=(year,year))
        try:
            if not frame.empty:
                if list(frame.columns)!=COLUMNS:frame=frame[COLUMNS]
                for row in frame.itertuples(index=False,name=None):acc.add(*row)
                acc.flush()
        finally:
            del frame
            ctx.up.clear_cache()
            gc.collect()
    u=np.asarray(ctx.universe,dtype=bool)
    if u.shape!=acc.values[0].shape:raise ValueError('Universe shape mismatch')
    return tuple(np.where(u,x,np.nan) for x in acc.values)

def build(parameters):
    if set(parameters)!={'signal'} or parameters['signal'] not in SIGNALS:raise ValueError('Frozen signals only')
    index=SIGNALS.index(parameters['signal'])
    def compute(ctx):
        return compute_both(ctx)[index]
    return compute
