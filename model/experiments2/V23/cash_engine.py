#!/usr/bin/env python3
"""Execution-aware walk-forward ledger and benchmark report for V11."""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"): os.environ[k]="1"
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parents[1]/"trainingdata"
CODES=("601288.SH","601398.SH","601939.SH","601988.SH")
INITIAL=100000.0
LOT=100
COMMISSION=0.00025
MIN_COMMISSION=5.0
TRANSFER=0.00001
STAMP=0.0005
SLIPPAGE=0.0003
PARTICIPATION=0.01

def load_prices():
    meta=json.loads((DATA/"meta.json").read_text(encoding="utf-8"))
    years=sorted(int(y) for y in meta["built_years"])
    pieces=[]
    for y in years:
        p=pq.read_table(DATA/"prices"/f"year={y}"/"data.parquet",
                        columns=["trade_date","stock_code","open","high","low","close","pre_close","vol","adj_factor"],
                        filters=[("stock_code","in",list(CODES))]).to_pandas()
        a=pq.read_table(DATA/"amount"/f"year={y}"/"data.parquet",
                        filters=[("stock_code","in",list(CODES))]).to_pandas()
        ac=[c for c in a.columns if c not in ("trade_date","stock_code")]
        if len(ac)!=1: raise RuntimeError(f"amount schema unexpected: {ac}")
        a=a.rename(columns={ac[0]:"amount"})
        p=p.merge(a,on=["trade_date","stock_code"],how="left",validate="one_to_one")
        p["trade_date"]=p["trade_date"].astype(str).str[:10]; p["stock_code"]=p["stock_code"].astype(str)
        pieces.append(p)
    f=pd.concat(pieces,ignore_index=True)
    days=np.asarray(sorted(f.trade_date.unique()),dtype=str)
    ix=pd.MultiIndex.from_product([days,CODES],names=["trade_date","stock_code"])
    f=f.set_index(["trade_date","stock_code"]).reindex(ix).reset_index()
    n=len(days); c=len(CODES)
    out={"days":days}
    for col in ("open","high","low","close","pre_close","vol","adj_factor","amount"):
        out[col]=f[col].to_numpy(np.float64).reshape(n,c)
    adj=out["adj_factor"].copy()
    for j in range(c):
        valid=np.flatnonzero(np.isfinite(adj[:,j])&(adj[:,j]>0))
        if len(valid):
            adj[:valid[0],j]=adj[valid[0],j]
            for i in range(valid[0]+1,n):
                if not np.isfinite(adj[i,j]) or adj[i,j]<=0: adj[i,j]=adj[i-1,j]
    out["adj_factor"]=adj
    out["adj_close"]=out["close"]*adj
    out["index"]={d:i for i,d in enumerate(days)}
    return out

def load_predictions(quarters):
    frames=[]
    for q in quarters:
        p=ROOT/"quarters"/q/"predictions.csv"
        if not p.is_file(): raise FileNotFoundError(p)
        frames.append(pd.read_csv(p,dtype={"trade_date":str,"stock_code":str}))
    pred=pd.concat(frames,ignore_index=True)
    pred["trade_date"]=pred.trade_date.str[:10]
    pred["monthly_signal"]=pred.monthly_signal.astype(bool)
    pred["selected_top2"]=pred.selected_top2.astype(bool)
    if pred.duplicated(["trade_date","stock_code"]).any(): raise RuntimeError("duplicate score rows")
    return pred

def schedules(pred, prices):
    days=prices["days"]; idx=prices["index"]
    cand={}; mom={}; all_signal=[]
    dates=sorted(pred.loc[pred.monthly_signal,"trade_date"].unique())
    adj_close=prices["adj_close"]
    for d in dates:
        di=idx[d]; all_signal.append(d)
        sub=pred[(pred.trade_date==d)&pred.selected_top2]
        pick=sub.sort_values(["daily_rank","stock_code"]).stock_code.tolist()
        if len(pick)!=2: raise RuntimeError(f"{d}: V11 top2 count={len(pick)}")
        cand[d]={x:0.5 for x in pick}
        if di<60: raise RuntimeError(f"{d}: insufficient history for 60-session benchmark")
        trailing=adj_close[di]/adj_close[di-60]-1.0
        order=sorted(range(len(CODES)),key=lambda j:(-(trailing[j] if np.isfinite(trailing[j]) else -np.inf),CODES[j]))
        mom[d]={CODES[j]:0.5 for j in order[:2]}
    equal={dates[0]:{c:0.25 for c in CODES}}
    return {"V11 LambdaRank Top2":cand,"60d momentum Top2":mom,"equal-weight hold":equal},dates

def fee_for(qty,px,sell):
    notional=qty*px
    return max(MIN_COMMISSION,notional*COMMISSION)+notional*TRANSFER+(notional*STAMP if sell else 0.0)

def simulate(name, price, signals, start_day, end_day):
    days=price["days"]; ix=price["index"]; start=ix[start_day]; end=ix[end_day]
    orders={}
    for d,target in signals.items():
        si=ix[d]; ei=si+1
        if ei<=end: orders.setdefault(ei,[]).append(target)
    cash=INITIAL; gross_cash=INITIAL; shares=np.zeros(4,dtype=np.float64)
    last_close=np.full(4,np.nan); rows=[]; trades=[]
    fees_total=slip_total=turnover=0.0; nblocked=0; max_positions=0; min_cash=INITIAL
    open_=price["open"]; hi=price["high"]; lo=price["low"]; close=price["close"]
    prev=price["pre_close"]; vol=price["vol"]; amt=price["amount"]; adj=price["adj_factor"]
    for t in range(start,end+1):
        if t>0:
            ratio=np.divide(adj[t],adj[t-1],out=np.ones(4),where=np.isfinite(adj[t])&(adj[t]>0)&np.isfinite(adj[t-1])&(adj[t-1]>0))
            shares*=ratio
        op=open_[t]
        mark_open=np.where(np.isfinite(op)&(op>0),op,np.where(np.isfinite(last_close),last_close,0.0))
        if t in orders:
            target=orders[t][-1]
            weights=np.asarray([target.get(c,0.0) for c in CODES],dtype=np.float64)
            equity_open=cash+float(np.dot(shares,mark_open))
            # Sell excess holdings first; unavailable exits remain in the account.
            for j in range(4):
                if shares[j]<=0 or not np.isfinite(op[j]) or op[j]<=0: continue
                desired=equity_open*weights[j]; current=shares[j]*op[j]
                diff=current-desired
                full_exit=weights[j]<=0
                if not full_exit and diff<op[j]*LOT: continue
                one=np.isfinite(hi[t,j]) and np.isfinite(lo[t,j]) and abs(hi[t,j]-lo[t,j])<=1e-8
                down=one and np.isfinite(prev[t,j]) and op[j]<=prev[t,j]*0.905
                if down or not (np.isfinite(vol[t,j]) and vol[t,j]>0 and np.isfinite(amt[t,j]) and amt[t,j]>0):
                    nblocked+=1; continue
                cap_exact=max(0.0,float(amt[t,j]*PARTICIPATION/op[j]))
                # A zero-weight exit may sell the final odd lot only when the participation cap
                # can carry the entire residual. Otherwise sell whole lots, if any, and retry later.
                if full_exit and cap_exact+1e-8>=shares[j]:
                    qty=float(shares[j])
                else:
                    desired_qty=math.floor(max(0.0,diff/op[j])/LOT)*LOT
                    held_lots=math.floor(shares[j]/LOT)*LOT
                    cap_lots=math.floor(cap_exact/LOT)*LOT
                    qty=float(min(desired_qty,held_lots,cap_lots))
                full_odd_exit=full_exit and qty>0 and abs(qty-shares[j])<=1e-8
                if qty<LOT and not full_odd_exit:
                    nblocked+=1
                    continue
                if full_exit and qty+1e-8<shares[j]:
                    nblocked+=1
                fill=op[j]*(1-SLIPPAGE); fee=fee_for(qty,fill,True); notional=qty*fill
                cash+=notional-fee; gross_cash+=qty*op[j]; shares[j]-=qty
                fees_total+=fee; slip_total+=qty*op[j]*SLIPPAGE; turnover+=qty*op[j]
                trades.append({"trade_date":str(days[t]),"strategy":name,"stock_code":CODES[j],"side":"sell","shares":round(float(qty),4),"open":float(op[j]),"fill":float(fill),"notional":float(notional),"fee":float(fee),"slippage_cost":float(qty*op[j]*SLIPPAGE),"reason":"monthly_rebalance"})
            # Buy toward target weights using only available shared cash.
            for j in range(4):
                if weights[j]<=0 or not np.isfinite(op[j]) or op[j]<=0: continue
                current=shares[j]*op[j]; need=max(0.0,equity_open*weights[j]-current)
                if need<op[j]*LOT: continue
                one=np.isfinite(hi[t,j]) and np.isfinite(lo[t,j]) and abs(hi[t,j]-lo[t,j])<=1e-8
                up=one and np.isfinite(prev[t,j]) and op[j]>=prev[t,j]*1.095
                if up or not (np.isfinite(vol[t,j]) and vol[t,j]>0 and np.isfinite(amt[t,j]) and amt[t,j]>0):
                    nblocked+=1; continue
                fill=op[j]*(1+SLIPPAGE)
                qty=math.floor(need/fill/LOT)*LOT
                maxq=math.floor((amt[t,j]*PARTICIPATION/op[j])/LOT)*LOT
                qty=min(qty,maxq)
                while qty>=LOT and qty*fill+fee_for(qty,fill,False)>cash+1e-8: qty-=LOT
                if qty<LOT: continue
                fee=fee_for(qty,fill,False); notional=qty*fill
                cash-=notional+fee; gross_cash-=qty*op[j]; shares[j]+=qty
                fees_total+=fee; slip_total+=qty*op[j]*SLIPPAGE; turnover+=qty*op[j]
                trades.append({"trade_date":str(days[t]),"strategy":name,"stock_code":CODES[j],"side":"buy","shares":round(float(qty),4),"open":float(op[j]),"fill":float(fill),"notional":float(notional),"fee":float(fee),"slippage_cost":float(qty*op[j]*SLIPPAGE),"reason":"monthly_rebalance"})
                min_cash=min(min_cash,cash)
        valid_close=np.isfinite(close[t])&(close[t]>0)
        marks=np.where(valid_close,close[t],np.where(np.isfinite(last_close),last_close,0.0))
        last_close=np.where(valid_close,close[t],last_close)
        mv=shares*marks; equity=cash+float(mv.sum()); gross_equity=gross_cash+float(mv.sum())
        resid=equity-cash-float(mv.sum())
        active=int(np.count_nonzero(shares>1e-8)); max_positions=max(max_positions,active)
        rows.append({"trade_date":str(days[t]),"strategy":name,"equity_net":equity,"equity_gross":gross_equity,
                     "cash":cash,"positions":active,"fees_cumulative":fees_total,"slippage_cumulative":slip_total,
                     "turnover_cumulative":turnover,"accounting_residual":resid,
                     "shares_json":json.dumps({CODES[j]:round(float(shares[j]),4) for j in range(4) if shares[j]>1e-8})})
        min_cash=min(min_cash,cash)
    daily=pd.DataFrame(rows)
    daily["daily_return_net"]=daily.equity_net.pct_change().fillna(0.0)
    daily["drawdown"]=daily.equity_net/daily.equity_net.cummax()-1.0
    n=max(1,len(daily)-1); total=float(daily.equity_net.iloc[-1]/INITIAL-1.0)
    ann=(1+total)**(252/n)-1 if total>-1 else -1.0
    metrics={"strategy":name,"start":start_day,"end":end_day,"days":int(len(daily)),
      "net_return":total,"gross_return_same_fills":float(daily.equity_gross.iloc[-1]/INITIAL-1.0),
      "annualized_net_return":float(ann),"annualized_volatility":float(daily.daily_return_net.std(ddof=1)*math.sqrt(252)),
      "max_drawdown":float(daily.drawdown.min()),"fees":float(fees_total),"slippage_cost":float(slip_total),
      "total_trading_cost":float(fees_total+slip_total),"turnover":float(turnover),"trades":int(len(trades)),
      "blocked_trade_events":int(nblocked),"min_cash":float(min_cash),"max_positions":int(max_positions),
      "max_abs_accounting_residual":float(daily.accounting_residual.abs().max())}
    return daily, trades, metrics

def period_returns(daily, freq):
    data=daily.copy()
    data["period"]=pd.to_datetime(data.trade_date).dt.to_period(freq).astype(str)
    out=[]
    prev_net=INITIAL
    prev_gross=INITIAL
    for period,g in data.groupby("period",sort=True):
        end_net=float(g.equity_net.iloc[-1])
        end_gross=float(g.equity_gross.iloc[-1])
        out.append({"period":period,"strategy":g.strategy.iloc[0],
                    "net_return":end_net/prev_net-1.0,
                    "gross_return_same_fills":end_gross/prev_gross-1.0})
        prev_net=end_net
        prev_gross=end_gross
    return out
