"""五席现金账户：日线事件顺序明确，停止单不借用未来现金。"""
from dataclasses import dataclass,asdict,replace
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Spec:
    name:str='baseline'
    exit_rank:int=700
    min_hold:int=5
    stop_loss:float=.08
    take_profit:float=.20
    cooldown:int=0
    max_hold:int=0
    mode:str='close'
    atr_stop:float=0.
    trailing:float=0.
    trail_arm:float=.10
    gate_window:int=60
    hysteresis:float=0.
    immediate_gate:bool=True

def specs():
    base=Spec()
    cases=[replace(base,name='legacy_fixed',immediate_gate=False),replace(base,name='close_fixed'),
           replace(base,name='close_cool3',cooldown=3),replace(base,name='close_cool5',cooldown=5),
           replace(base,name='close_max30',cooldown=3,max_hold=30),
           replace(base,name='intraday_fixed',mode='intraday'),replace(base,name='intraday_cool3',mode='intraday',cooldown=3),
           replace(base,name='atr_close',atr_stop=2.5,cooldown=3),replace(base,name='atr_intraday',atr_stop=2.5,cooldown=3,mode='intraday'),
           replace(base,name='trail_close',trailing=.08,cooldown=3),
           replace(base,name='hysteresis_close',hysteresis=.01,cooldown=3),
           replace(base,name='hysteresis_intraday',hysteresis=.01,cooldown=3,mode='intraday')]
    for rank in (300,500,1000):cases.append(replace(base,name=f'rank{rank}_cool3',exit_rank=rank,cooldown=3))
    for stop,take in ((.06,.15),(.10,.25)):
        for mode in ('close','intraday'):
            cases.append(replace(base,name=f'{mode}_sl{int(stop*100)}_tp{int(take*100)}',stop_loss=stop,take_profit=take,cooldown=3,mode=mode))
    for stop,take in ((.06,.15),(.08,.20)):
        cases.append(replace(base,name=f'five_day_close_sl{int(stop*100)}_tp{int(take*100)}',
                             exit_rank=2115,stop_loss=stop,take_profit=take,cooldown=3,
                             max_hold=5,mode='close'))
    return cases

def fees(value,sell=False):return max(value*.00025,5)+value*.00001+(value*.0005 if sell else 0)

def market_features(px):
    close=px.close*px.adj
    ret=close[1:]/close[:-1]-1
    good=np.isfinite(ret)
    market=np.r_[0,np.where(good,ret,0).sum(axis=1)/np.maximum(1,good.sum(axis=1))]
    index=100*np.cumprod(1+market)
    prev=np.vstack([close[:1],close[:-1]])
    high=px.raw['high']*px.adj;low=px.raw['low']*px.adj
    tr=np.maximum(high-low,np.maximum(np.abs(high-prev),np.abs(low-prev)))
    atr=pd.DataFrame(tr).rolling(20,min_periods=10).mean().to_numpy()/close
    return dict(index=index,atr=atr)

def gate(features,spec):
    index=features['index'];ma=pd.Series(index).rolling(spec.gate_window,min_periods=spec.gate_window).mean().to_numpy()
    if not spec.hysteresis:return index>=ma
    on=np.zeros(len(index),bool);state=False
    for i in range(len(index)):
        if not np.isfinite(ma[i]):continue
        if state and index[i]<ma[i]*(1-spec.hysteresis):state=False
        elif not state and index[i]>=ma[i]*(1+spec.hysteresis):state=True
        on[i]=state
    return on

def simulate(pred,days,panel,px,spec,features,slippage=.0003,details=False):
    """Signal close -> next open sales -> next open buys -> old-position intraday stops -> close mark."""
    days=np.asarray(days,dtype=int)
    assert pred.shape==(len(days),len(panel.codes))
    assert len(days)>0 and np.all(np.diff(days)==1)
    on=gate(features,spec)
    cash=100000.;hold={};cool={};trades=[];curves=[]
    first=int(days[0])+1;last=min(int(days[-1])+2,len(panel.days)-1)
    by_day={int(d):p for d,p in zip(days,pred)}
    blocked_total=0;total_fee=0.;turnover=0.;corp=0
    def sell(c,d,raw,reason):
        nonlocal cash,total_fee,turnover
        pos=hold[c];price=raw*(1-slippage);value=pos['quantity']*price;fee=fees(value,True)
        cash+=value-fee;total_fee+=fee;turnover+=value
        trades.append(dict(date=str(panel.days[d]),code=str(panel.codes[c]),side='sell',quantity=pos['quantity'],price=float(price),fee=float(fee),reason=reason,entry_date=str(panel.days[pos['buy_day']])))
        if reason in ('stop_loss','take_profit','trailing','max_hold'):cool[c]=d+spec.cooldown
        del hold[c]
    for d in range(first,last+1):
        for c,pos in hold.items():
            ratio=px.adj[d,c]/px.adj[d-1,c]
            if not np.isfinite(ratio) or ratio<=0:raise ValueError('corporate action missing')
            pos['quantity']*=ratio
            if abs(ratio-1)>1e-8:corp+=1
        signal=d-1;p=by_day.get(signal);desired=list(hold);reasons={}
        if p is not None:
            candidates=np.flatnonzero(np.isfinite(p)) if on[signal] else np.array([],int)
            ranked=candidates[np.argsort(-p[candidates],kind='stable')]
            head=set(map(int,ranked[:spec.exit_rank]))
            for c,pos in hold.items():
                price=px.close[signal,c]*px.adj[signal,c]
                pos['peak']=max(pos['peak'],price)
                age=d-pos['buy_day'];reason=None
                if not on[signal] and spec.immediate_gate:reason='market_gate'
                elif c not in head and age>=spec.min_hold:reason='rank_exit'
                elif spec.max_hold and age>=spec.max_hold:reason='max_hold'
                elif spec.mode=='close':
                    if price<=pos['entry']*(1-pos['stop']):reason='stop_loss'
                    elif price>=pos['entry']*(1+pos['take']):reason='take_profit'
                    elif spec.trailing and pos['peak']>=pos['entry']*(1+spec.trail_arm) and price<=pos['peak']*(1-spec.trailing):reason='trailing'
                if reason:reasons[c]=reason
            desired=[c for c in hold if c not in reasons]
            for value in ranked:
                c=int(value)
                if len(desired)>=5:break
                if c in desired or c in reasons or d<cool.get(c,-1):continue
                if px.entry[d,c]:desired.append(c)
        if d==last:desired=[];reasons={c:'end_window' for c in hold}
        blocked=0
        for c in list(hold):
            if c in desired:continue
            if not px.exit[d,c] or hold[c]['buy_day']>=d:blocked+=1;continue
            sell(c,d,px.open[d,c],reasons.get(c,'rebalance'))
        if p is not None and d<last:
            equity=cash+sum(pos['quantity']*(px.open[d,c] if np.isfinite(px.open[d,c]) else px.close[d-1,c]) for c,pos in hold.items())
            for c in desired:
                if c in hold or len(hold)>=5 or d<cool.get(c,-1):continue
                price=px.open[d,c]*(1+slippage)
                amount=panel.amount[signal,c]
                budget=min(cash,equity/5,amount*.01 if np.isfinite(amount) else 0.)
                qty=int(max(0,budget-5)/(price*1.00026)/100)*100
                while qty>0 and qty*price+fees(qty*price)>budget:qty-=100
                if qty<=0:continue
                value=qty*price;fee=fees(value);cash-=value+fee;total_fee+=fee;turnover+=value
                stop=spec.stop_loss;take=spec.take_profit
                if spec.atr_stop:
                    atr=features['atr'][signal,c]
                    if np.isfinite(atr):stop=float(np.clip(atr*spec.atr_stop,.05,.12));take=float(np.clip(stop*2.5,.125,.30))
                entry=price*px.adj[d,c]
                hold[c]=dict(quantity=float(qty),buy_day=d,entry=entry,peak=entry,stop=stop,take=take)
                trades.append(dict(date=str(panel.days[d]),code=str(panel.codes[c]),side='buy',quantity=qty,price=float(price),fee=float(fee),reason='top5_entry',signal_date=str(panel.days[signal])))
        # Intraday proceeds become available only after today's opening purchases.
        if spec.mode=='intraday' and d<last:
            for c,pos in list(hold.items()):
                if pos['buy_day']>=d or not px.exit[d,c]:continue
                opening=px.open[d,c]*px.adj[d,c]
                low=px.raw['low'][d,c]*px.adj[d,c];high=px.raw['high'][d,c]*px.adj[d,c]
                stop=pos['entry']*(1-pos['stop']);take=pos['entry']*(1+pos['take'])
                if np.isfinite(low) and low<=stop:sell(c,d,min(opening,stop)/px.adj[d,c],'stop_loss')
                elif np.isfinite(high) and high>=take:sell(c,d,take/px.adj[d,c],'take_profit')
        equity=cash+sum(pos['quantity']*px.close[d,c] for c,pos in hold.items())
        assert cash>=-1e-6 and len(hold)<=5 and np.isfinite(equity)
        curves.append(dict(date=str(panel.days[d]),equity=float(equity),cash=float(cash),positions=len(hold),blocked_exits=blocked))
        blocked_total+=blocked
    values=np.r_[100000.,[c['equity'] for c in curves]];returns=values[1:]/values[:-1]-1
    dd=values/np.maximum.accumulate(values)-1
    metrics=dict(strategy=spec.name,net_return=float(values[-1]/100000.-1),max_drawdown=float(dd.min()),
                 sharpe=float(returns.mean()/returns.std()*np.sqrt(242)) if returns.std()>0 else 0.,
                 fees=total_fee,trades=len(trades),turnover=turnover,blocked_exits=blocked_total,
                 ending_positions=len(hold),corporate_actions=corp,start=str(panel.days[first]),end=str(panel.days[last]),
                 slippage=slippage,spec=asdict(spec))
    return metrics,curves if details else None,trades if details else None
