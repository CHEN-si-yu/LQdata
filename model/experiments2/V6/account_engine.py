"""Embedded audited account ledger for V6."""
import math
import numpy as np
import pandas as pd

COST={"account_money":100000.0,"lot_size":100,"max_participation":0.01,
      "commission_rate":0.00025,"min_commission":5.0,"transfer_rate":0.00001,
      "stamp_sell_rate":0.0005,"slippage_rate":0.0003}

def fee(notional,side):
    return max(COST["min_commission"],notional*COST["commission_rate"])+notional*COST["transfer_rate"]+(notional*COST["stamp_sell_rate"] if side=="sell" else 0.0)

def limittype(code):
    s=str(code)
    if s.startswith(("300","301","688","689")): return 0.195
    if s.startswith(("4","8","92")): return 0.295
    return 0.095

def account_sim(days,codes,a,schedule,target_names=None,force_names=None,rank_signal=None,k=0,features=None,direction=1,benchmark_mode="strategy",with_costs=True):
    # target_names at each scheduled execution maps exec-index to selected stock indices.
    op=a["open"]; high=a["high"]; low=a["low"]; prev=a["pre_close"]; close=a["close"]
    vol=a["vol"]; adj=pd.DataFrame(a["adj_factor"]).ffill().to_numpy(dtype=float); amount=a.get("amount",np.full_like(op,np.nan))
    nday,nc=op.shape; first=0; last=nday-1
    if first >= len(days): raise RuntimeError("empty evaluation window")
    shares=np.zeros(nc,dtype=float); cash=float(COST["account_money"]); prev_eq=float(COST["account_money"])
    lastmark=np.full(nc,np.nan); lastopen=np.full(nc,np.nan); lastadj=np.full(nc,np.nan)
    ntr=0; fees_total=0.; buy_notional=0.; sell_notional=0.; blocked_entries=0; blocked_exits=0
    curves=[]; trades=[]
    for i in range(first,last+1):
        # Corporate actions are reflected by adjusted share count before marking.
        if i>first:
            ratio=np.divide(adj[i],adj[i-1],out=np.ones(nc),where=np.isfinite(adj[i])&(adj[i]>0)&np.isfinite(adj[i-1])&(adj[i-1]>0))
            shares*=ratio
        mark=np.where(np.isfinite(op[i])&(op[i]>0),op[i],np.where(np.isfinite(close[i])&(close[i]>0),close[i],lastmark))
        mark=np.where(np.isfinite(mark)&(mark>0),mark,lastmark)
        if i>first:
            prior_stock_mv=np.nan_to_num(shares_prev*lastmark_prev,nan=0.0,posinf=0.0)
            prior_w=prior_stock_mv/max(prev_eq,1e-12)
            r=np.divide(op[i]*adj[i],op[i-1]*adj[i-1],out=np.ones(nc),where=np.isfinite(op[i])&(op[i]>0)&np.isfinite(adj[i])&(adj[i]>0)&np.isfinite(op[i-1])&(op[i-1]>0)&np.isfinite(adj[i-1])&(adj[i-1]>0))-1.0
            bench=float(np.sum(prior_w*np.nan_to_num(r,nan=0.0)))
        else: bench=0.0
        equity_pre=cash+float(np.nansum(shares*np.nan_to_num(mark,nan=0.0)))
        sig_ix=i-1
        desired=None
        if i in schedule:
            desired=schedule[i]
        is_final=(i==last)
        if is_final:
            desired=np.array([],dtype=int)
        if desired is not None:
            desired_set=set(int(x) for x in desired)
            eq_for_target=max(equity_pre,0.0)
            target_qty={}
            for c in desired:
                px=mark[c]
                if np.isfinite(px) and px>0:
                    target_qty[int(c)]=math.floor(eq_for_target/max(1,len(desired))/px/COST["lot_size"])*COST["lot_size"]
                else: target_qty[int(c)]=0
            # Sell exits/trims before funding entries.
            for c in np.flatnonzero(shares>1e-7):
                want=float(target_qty.get(int(c),0))
                delta=shares[c]-want
                if delta<=1e-7: continue
                if not (np.isfinite(op[i,c]) and op[i,c]>0 and np.isfinite(vol[i,c]) and vol[i,c]>0):
                    blocked_exits+=1; continue
                lim=limittype(codes[c])
                one=np.isfinite(high[i,c]) and np.isfinite(low[i,c]) and abs(high[i,c]-low[i,c])<=1e-8
                down=one and np.isfinite(prev[i,c]) and op[i,c]<=prev[i,c]*(1.0-lim+0.005)
                if down:
                    blocked_exits+=1; continue
                capamt=float(amount[sig_ix,c]) if sig_ix>=0 else 0.0
                cap=math.floor(max(capamt,0.0)*COST["max_participation"]/op[i,c]/COST["lot_size"])*COST["lot_size"]
                req=shares[c] if want<=1e-8 else math.floor(delta/COST["lot_size"])*COST["lot_size"]
                qty=min(req,cap)
                if want<=1e-8 and qty>=shares[c]-1e-7: qty=shares[c]
                else: qty=math.floor(qty/COST["lot_size"])*COST["lot_size"]
                if qty<=0: continue
                fill=op[i,c]*(1.0-COST["slippage_rate"]); notional=qty*fill
                f=fee(notional,"sell") if with_costs else 0.0
                cash+=notional-f; shares[c]-=qty; fees_total+=f; sell_notional+=notional; ntr+=1
                trades.append({"date":str(days[i]),"signal_date":str(days[sig_ix]) if sig_ix>=0 else "",
                               "code":str(codes[c]),"side":"sell","qty":float(qty),"notional":notional,"fee":f})
            # Buy targets in rank order to allocate remaining cash to the highest ranks first.
            order=list(desired)
            if rank_signal is not None and len(order):
                order=sorted(order,key=lambda c:rank_signal[sig_ix,c],reverse=True)
            for c0 in order:
                c=int(c0); want=float(target_qty.get(c,0)); delta=want-shares[c]
                if delta<COST["lot_size"]-1e-7: continue
                if not (np.isfinite(op[i,c]) and op[i,c]>0 and np.isfinite(vol[i,c]) and vol[i,c]>0):
                    blocked_entries+=1; continue
                lim=limittype(codes[c])
                one=np.isfinite(high[i,c]) and np.isfinite(low[i,c]) and abs(high[i,c]-low[i,c])<=1e-8
                up=one and np.isfinite(prev[i,c]) and op[i,c]>=prev[i,c]*(1.0+lim-0.005)
                if up:
                    blocked_entries+=1; continue
                capamt=float(amount[sig_ix,c]) if sig_ix>=0 else 0.0
                cap=math.floor(max(capamt,0.0)*COST["max_participation"]/op[i,c]/COST["lot_size"])*COST["lot_size"]
                qty=min(math.floor(delta/COST["lot_size"])*COST["lot_size"],cap)
                fill=op[i,c]*(1.0+COST["slippage_rate"])
                while qty>0:
                    notional=qty*fill; f=fee(notional,"buy") if with_costs else 0.0
                    if notional+f<=cash+1e-8: break
                    qty-=COST["lot_size"]
                if qty<=0: continue
                cash-=notional+f; shares[c]+=qty; fees_total+=f; buy_notional+=notional; ntr+=1
                trades.append({"date":str(days[i]),"signal_date":str(days[sig_ix]) if sig_ix>=0 else "",
                               "code":str(codes[c]),"side":"buy","qty":float(qty),"notional":notional,"fee":f})
        equity=cash+float(np.nansum(shares*np.nan_to_num(mark,nan=0.0)))
        daily=equity/prev_eq-1.0 if prev_eq>0 else 0.0
        invested=float(np.nansum(shares*np.nan_to_num(mark,nan=0.0)))
        curves.append({"date":str(days[i]),"equity":equity,"daily_return":daily,"matched_return":bench,
                       "cash":cash,"exposure":invested/max(equity,1e-12),"holdings":int(np.count_nonzero(shares>1e-7)),
                       "shares":shares.tolist()})
        shares_prev=shares.copy()
        lastmark_prev=mark.copy()
        prev_eq=equity; lastmark=mark.copy()
    return {"curve":curves,"trades":trades,"fees":fees_total,"ntr":ntr,"buy_notional":buy_notional,
            "sell_notional":sell_notional,"blocked_entries":blocked_entries,"blocked_exits":blocked_exits,
            "ending_shares":shares.tolist(),"ending_cash":cash}

def summarize(curve,initial):
    d=pd.DataFrame(curve)
    daily=d.daily_return.to_numpy(float); matched=d.matched_return.to_numpy(float)
    eq=np.r_[initial,d.equity.to_numpy(float)]
    mdd=float(np.min(eq/np.maximum.accumulate(eq)-1.0))
    total=float(d.equity.iloc[-1]/initial-1.0)
    years=len(d)/242.0
    ann=float(max(d.equity.iloc[-1]/initial,1e-12)**(1/years)-1) if years>0 else 0
    match_factor=float(np.prod(1+matched))
    gap=total-(match_factor-1.0)
    return {"days":int(len(d)),"cumulative_net_return":total,"annualized_return":ann,"max_drawdown":mdd,
            "matched_benchmark_return":match_factor-1.0,"matched_cumulative_excess":gap,
            "avg_exposure":float(d.exposure.mean()),"average_holdings":float(d.holdings.mean()),
            "ending_equity":float(d.equity.iloc[-1]),
            "yearly":{str(y):{"strategy_net_return":float(np.prod(1+d.loc[pd.to_datetime(d.date).dt.year==y,"daily_return"])-1),
                              "matched_return":float(np.prod(1+d.loc[pd.to_datetime(d.date).dt.year==y,"matched_return"])-1),
                              "daily_matched_excess":float(np.prod(1+d.loc[pd.to_datetime(d.date).dt.year==y,"daily_return"])-1)-float(np.prod(1+d.loc[pd.to_datetime(d.date).dt.year==y,"matched_return"])-1)}
                      for y in sorted(pd.to_datetime(d.date).dt.year.unique())}}