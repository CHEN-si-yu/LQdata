#!/usr/bin/env python3
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"
import sys, json, math, time, hashlib, argparse, resource, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path("/root/autodl-fs/model/trainingdata")
OUT=Path("/root/autodl-fs/model/experiments2/V10/engine_artifacts")
OUT.mkdir(parents=True, exist_ok=True)
PROTOCOL=OUT/"protocol.json"
SCREEN=OUT/"train_screen.json"
CANDIDATES=[
 "id2_close_vs_pm_vwap_20","drawdown_duration_120","sortino_ratio_60",
 "bs_construction_capital_share","chip_peak_distance","gap_down_recover_freq_20d",
 "price_distance_from_52w_low","bw_overnight_lag_beta_20","beta_60","cp_quality_momentum",
 "momentum_20","momentum_60","rel_mom_ind_20d","short_term_reversal_5","reversal_2d"
]
PRICE=["open","high","low","pre_close","close","vol","adj_factor"]
COST={"account_money":100000.0,"lot_size":100,"max_participation":0.01,
      "commission_rate":0.00025,"min_commission":5.0,"transfer_rate":0.00001,
      "stamp_sell_rate":0.0005,"slippage_rate":0.0003}
if not PROTOCOL.exists():
    protocol={
      "experiment":"cross-sectional top-k low-turnover rank portfolio",
      "candidate_pool":CANDIDATES,
      "candidate_pool_source":"training-only stable 5d factor shortlist plus five prespecified controls; market-wide factors excluded because they are constant cross-sectionally",
      "snapshot":"trainingdata snapshot available on execution; train factors and rank IC use 2018-2022 only; OOS begins 2023-01-03 and ends at latest complete date",
      "training_signal":"last trading session of each ISO week; signal at close; execute next trading open",
      "training_target":"adjusted open-to-open return from next session open through five open intervals (open[t+6]/open[t+1]-1)",
      "training_ic":"weekly cross-sectional Spearman rank IC, equal-weight across valid names, averaged within each of 20 train quarters",
      "selection_rule":"direction is sign of pooled train weekly rank IC; confirmation requires positive signed mean quarterly rank IC and at least 16 of 20 positive quarterly ICs; rank eligible features by signed mean quarterly IC and select at most three",
      "exploration_if_none":"if no feature passes confirmation, run train-ranked top three (same direction and ranking, no OOS selection), label all such results exploratory",
      "oos_grid":{"holding_ranks":[2,3,4],"rebalance":["weekly","monthly"],"date_start":"2023-01-03","date_end":"latest available"},
      "signal_universe":"stocks with finite factor and positive close/open/volume at signal; rank descending after training-only direction",
      "execution":"single account; target equal weights among selected top K; T+1 raw open; board lot 100; 1% prior signal-day traded-amount capacity; one-price limit lock; 3bp slippage",
      "costs":COST,
      "return":"daily adjusted open-to-open mark-to-market; strategy net of transaction costs",
      "strict_matched_benchmark":"previous execution-day stock weights times each stock adjusted open-to-open return, no costs in matched leg; compare compounded net strategy and matched benchmark",
      "other_baselines":["same top-K basket bought at first OOS signal and held, account-size with same costs","four-bank project account buy-and-hold, same account and costs","initial OOS day equal-weight universe gross reference"],
      "no_oos_tuning":True
    }
    PROTOCOL.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")

def schema_col(path,exclude):
    names=pq.ParquetFile(path).schema_arrow.names
    return next((x for x in names if x not in exclude),None)

def load_panel(years,features,with_amount=True):
    frames=[]
    for y in years:
        fp=ROOT/"factors"/f"year={y}"/"data.parquet"
        pp=ROOT/"prices"/f"year={y}"/"data.parquet"
        ap=ROOT/"amount"/f"year={y}"/"data.parquet"
        if not fp.exists() or not pp.exists(): continue
        fcols=pq.ParquetFile(fp).schema_arrow.names
        use=[x for x in features if x in fcols]
        if not use: continue
        f=pq.read_table(fp,columns=["trade_date","stock_code",*use]).to_pandas()
        pcols=pq.ParquetFile(pp).schema_arrow.names
        p=pq.read_table(pp,columns=["trade_date","stock_code",*[x for x in PRICE if x in pcols]]).to_pandas()
        d=f.merge(p,on=["trade_date","stock_code"],how="left",validate="one_to_one")
        if with_amount and ap.exists():
            acol=schema_col(ap,{"trade_date","stock_code"})
            if acol:
                a=pq.read_table(ap,columns=["trade_date","stock_code",acol]).to_pandas().rename(columns={acol:"amount"})
                d=d.merge(a,on=["trade_date","stock_code"],how="left",validate="one_to_one")
        d["trade_date"]=d["trade_date"].astype(str)
        d["stock_code"]=d["stock_code"].astype(str)
        frames.append(d)
        print(f"loaded {y}: {len(d):,} rows; features={len(use)}",flush=True)
    if not frames: raise RuntimeError("no parquet panel loaded")
    df=pd.concat(frames,ignore_index=True,copy=False)
    if df.duplicated(["trade_date","stock_code"]).any(): raise RuntimeError("duplicate date/code")
    days=np.array(sorted(df.trade_date.unique()),dtype=str)
    codes=np.array(sorted(df.stock_code.unique()),dtype=str)
    di=pd.Index(days).get_indexer(df.trade_date)
    ci=pd.Index(codes).get_indexer(df.stock_code)
    shape=(len(days),len(codes))
    arrays={}
    fields=list(features)+[x for x in PRICE if x in df.columns]+(["amount"] if "amount" in df.columns else [])
    for name in dict.fromkeys(fields):
        if name not in df.columns: continue
        dtype=np.float32 if name in features else np.float64
        arr=np.full(shape,np.nan,dtype=dtype)
        arr[di,ci]=df[name].to_numpy(dtype=dtype,na_value=np.nan)
        arrays[name]=arr
    meta={}
    mpath=ROOT/"meta.json"
    if mpath.exists(): meta=json.loads(mpath.read_text(encoding="utf-8"))
    return days,codes,arrays,meta

def week_signal_indices(days):
    dt=pd.to_datetime(days)
    keys=np.array([f"{x.isocalendar().year}-{x.isocalendar().week:02d}" for x in dt])
    return np.array([i for i in range(len(days)) if i==len(days)-1 or keys[i]!=keys[i+1]],dtype=int)

def month_signal_indices(days):
    dt=pd.to_datetime(days)
    keys=dt.strftime("%Y-%m").to_numpy()
    return np.array([i for i in range(len(days)) if i==len(days)-1 or keys[i]!=keys[i+1]],dtype=int)

def spearman(x,y):
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<30:return np.nan
    a=pd.Series(x[ok]).rank(method="average").to_numpy(dtype=float)
    b=pd.Series(y[ok]).rank(method="average").to_numpy(dtype=float)
    if np.std(a)==0 or np.std(b)==0:return np.nan
    return float(np.corrcoef(a,b)[0,1])

def run_screen():
    years=list(range(2018,2023))
    days,codes,a,meta=load_panel(years,CANDIDATES,with_amount=False)
    op=a["open"]; adj=a["adj_factor"]
    # Use raw prices times forward-filled adjustment factors, with no target crossing 2022-12-30.
    adj=pd.DataFrame(adj).ffill().to_numpy(dtype=float)
    dts=pd.to_datetime(days)
    wk=week_signal_indices(days)
    signal=[i for i in wk if days[i]<="2022-12-30" and i+6<len(days) and days[i+6]<="2022-12-30"]
    print(f"train dates={len(days)}, codes={len(codes)}, weekly signals={len(signal)}",flush=True)
    rows=[]
    for i in signal:
        j=i+1; h=i+6
        y=np.full(len(codes),np.nan,dtype=float)
        ok=np.isfinite(op[j])&(op[j]>0)&np.isfinite(op[h])&(op[h]>0)&np.isfinite(adj[j])&(adj[j]>0)&np.isfinite(adj[h])&(adj[h]>0)
        y[ok]=(op[h,ok]*adj[h,ok])/(op[j,ok]*adj[j,ok])-1.0
        q=str(pd.Period(days[i],freq="Q"))
        for feat in CANDIDATES:
            if feat not in a: continue
            ic=spearman(a[feat][i],y)
            rows.append({"date":str(days[i]),"quarter":q,"feature":feat,"ic":ic})
    detail=pd.DataFrame(rows)
    summary=[]
    for feat in CANDIDATES:
        z=detail[detail.feature==feat]
        qic=z.groupby("quarter",sort=True).ic.mean().dropna()
        pooled=float(z.ic.mean()) if z.ic.notna().any() else float("nan")
        direction=1 if not np.isfinite(pooled) or pooled>=0 else -1
        sq=qic*direction
        positive=int((sq>0).sum())
        meanq=float(sq.mean()) if len(sq) else float("nan")
        summary.append({"feature":feat,"train_weekly_pooled_ic":pooled,"train_direction":direction,
                        "n_quarters":int(len(qic)),"signed_mean_quarter_ic":meanq,
                        "positive_quarters":positive,"positive_quarter_fraction":positive/len(qic) if len(qic) else 0.0,
                        "quarterly_ic":{str(k):float(v*direction) for k,v in qic.items()}})
    eligible=[x for x in summary if x["n_quarters"]==20 and x["signed_mean_quarter_ic"]>0 and x["positive_quarters"]>=16]
    eligible.sort(key=lambda x:x["signed_mean_quarter_ic"],reverse=True)
    confirmation=eligible[:3]
    exploratory=False
    if confirmation:
        selected=confirmation
    else:
        exploratory=True
        ranked=[x for x in summary if x["n_quarters"]>=16 and np.isfinite(x["signed_mean_quarter_ic"])]
        ranked.sort(key=lambda x:(x["positive_quarters"],x["signed_mean_quarter_ic"]),reverse=True)
        selected=ranked[:3]
    result={"built_at":time.strftime("%Y-%m-%d %H:%M:%S"),"snapshot_built_at":meta.get("built_at"),
            "snapshot_last_day":meta.get("last_upstream_day",days[-1]),"training_range":[days[0],days[-1]],
            "protocol_path":str(PROTOCOL),"signal_count":len(signal),"candidate_count":len(summary),
            "confirmation_count":len(confirmation),"exploratory_fallback":exploratory,
            "selected":[{"feature":x["feature"],"direction":x["train_direction"],
                         "signed_mean_quarter_ic":x["signed_mean_quarter_ic"],
                         "positive_quarters":x["positive_quarters"],"n_quarters":x["n_quarters"],
                         "status":"confirmation" if not exploratory else "exploratory"} for x in selected],
            "candidates":summary,"weekly_rank_ic":rows}
    SCREEN.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("TRAIN_SCREEN_SAVED",SCREEN,flush=True)
    for x in sorted(summary,key=lambda r:r["signed_mean_quarter_ic"],reverse=True):
        print(x["feature"],"dir",x["train_direction"],"meanQ",round(x["signed_mean_quarter_ic"],4),
              "positive",f'{x["positive_quarters"]}/{x["n_quarters"]}',"selected",any(y["feature"]==x["feature"] for y in selected),flush=True)
    return result

# Globals inherited by forked config workers
G={}

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
                    target_qty[int(c)]=math.floor(eq_for_target*0.50/px/COST["lot_size"])*COST["lot_size"]
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

def run_config(task):
    feature,k,freq,direction=task
    days=G["days"]; codes=G["codes"]; a=G["arrays"]
    signal_idx=week_signal_indices(days) if freq=="weekly" else month_signal_indices(days)
    schedule={}
    for si in signal_idx:
        if si+1>=len(days) or days[si+1]<"2023-01-03": continue
        vals=a[feature][si]*direction
        valid=np.isfinite(vals)&np.isfinite(a["close"][si])&(a["close"][si]>0)&np.isfinite(a["open"][si])&(a["open"][si]>0)&np.isfinite(a["vol"][si])&(a["vol"][si]>0)
        idx=np.flatnonzero(valid)
        if len(idx)==0: continue
        order=idx[np.argsort(vals[idx])[::-1]]
        schedule[si+1]=order[:k]
    # signal ranks used only to prioritize cash allocation, with no future data.
    r=np.zeros_like(a[feature],dtype=np.float32)
    for si in signal_idx:
        vals=a[feature][si]*direction
        valid=np.isfinite(vals)
        if valid.any(): r[si,valid]=vals[valid]
    result=account_sim(days,codes,a,schedule,rank_signal=r)
    s=summarize(result["curve"],COST["account_money"])
    s.update({"feature":feature,"direction":direction,"k":k,"rebalance":freq,"status":G["status"],
              "trades":result["ntr"],"fees":result["fees"],"buy_notional":result["buy_notional"],
              "sell_notional":result["sell_notional"],"round_trip_turnover_on_initial_equity":
                 (result["buy_notional"]+result["sell_notional"])/COST["account_money"],
              "average_annual_round_trip_turnover":
                 (result["buy_notional"]+result["sell_notional"])/COST["account_money"]/max(s["days"]/242.0,1e-9),
              "blocked_entries":result["blocked_entries"],"blocked_exits":result["blocked_exits"],
              "ending_positions":int(np.count_nonzero(np.array(result["ending_shares"])>1e-7))})
    return {"summary":s,"curve":result["curve"],"trades":result["trades"]}

def run_oos(workers=2):
    screen=json.loads(SCREEN.read_text(encoding="utf-8"))
    selected=screen["selected"]
    if not selected: raise RuntimeError("training screen selected no candidates")
    feats=[x["feature"] for x in selected]
    days,codes,a,meta=load_panel(list(range(2022,2027)),feats,with_amount=True)
    last=days[-1]
    selected=[x for x in selected if x["feature"] in a]
    tasks=[(x["feature"],k,freq,int(x["direction"])) for x in selected for k in (2,3,4) for freq in ("weekly","monthly")]
    G.update({"days":days,"codes":codes,"arrays":a,"status":"exploratory" if screen["exploratory_fallback"] else "confirmation"})
    # warm-up explicitly uses two single-thread workers, then expand only if host remains below 50%.
    warm=tasks[:min(2,len(tasks))]
    results={}
    ctx=mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(2,workers),mp_context=ctx) as ex:
        futs=[ex.submit(run_config,t) for t in warm]
        for f in as_completed(futs):
            z=f.result(); key=f'{z["summary"]["feature"]}|{z["summary"]["k"]}|{z["summary"]["rebalance"]}'
            results[key]=z
            print("pilot",key,round(z["summary"]["cumulative_net_return"]*100,2),flush=True)
    rest=[t for t in tasks if f'{t[0]}|{t[1]}|{t[2]}' not in results]
    if rest:
        try:
            load1=os.getloadavg()[0]; cores=os.cpu_count() or 48
            mem={line.split()[0].rstrip(":"):int(line.split()[1]) for line in open("/proc/meminfo") if len(line.split())>=2}
            used_gib=(mem.get("MemTotal",0)-mem.get("MemAvailable",0))/1048576
            can_expand=(load1/cores<0.50 and used_gib<180.0)
        except Exception:
            load1=0.0; used_gib=0.0; can_expand=True
        n=min(max(2,workers if can_expand else 2),len(rest))
        print(f"resource gate after 2-worker pilot: load={load1:.2f}/{os.cpu_count()}, used_ram={used_gib:.1f}GiB, expand={can_expand}, workers={n}",flush=True)
        with ProcessPoolExecutor(max_workers=n,mp_context=ctx) as ex:
            futs=[ex.submit(run_config,t) for t in rest]
            for f in as_completed(futs):
                z=f.result(); key=f'{z["summary"]["feature"]}|{z["summary"]["k"]}|{z["summary"]["rebalance"]}'
                results[key]=z
                print("done",key,round(z["summary"]["cumulative_net_return"]*100,2),
                      "excess",round(z["summary"]["matched_cumulative_excess"]*100,2),flush=True)
                slim={"snapshot":meta.get("built_at"),"latest":last,"screen":str(SCREEN),
                      "summaries":[v["summary"] for v in results.values()]}
                (OUT/"progress.json").write_text(json.dumps(slim,ensure_ascii=False,indent=2),encoding="utf-8")
    # Add factor-specific initial top-k hold and a four-bank project-account static hold.
    # For clean comparisons these use the same full simulator/costs and are no-rebalance positions.
    baseline={}
    for featinfo in selected:
        feat=featinfo["feature"]; direction=int(featinfo["direction"])
        for k in (2,3,4):
            sig_i=int(np.flatnonzero((days<"2023-01-03"))[-1])
            vals=a[feat][sig_i]*direction
            valid=np.isfinite(vals)&np.isfinite(a["close"][sig_i])&(a["close"][sig_i]>0)&np.isfinite(a["vol"][sig_i])&(a["vol"][sig_i]>0)
            idx=np.flatnonzero(valid); idx=idx[np.argsort(vals[idx])[::-1]][:k]
            schedule={int(np.searchsorted(days,"2023-01-03")):idx}
            z=account_sim(days,codes,a,schedule)
            s=summarize(z["curve"],COST["account_money"])
            s.update({"feature":feat,"k":k,"rebalance":"initial_topk_hold","status":G["status"],
                      "trades":z["ntr"],"fees":z["fees"],"buy_notional":z["buy_notional"],
                      "sell_notional":z["sell_notional"],"round_trip_turnover_on_initial_equity":
                         (z["buy_notional"]+z["sell_notional"])/COST["account_money"],
                      "blocked_entries":z["blocked_entries"],"blocked_exits":z["blocked_exits"],
                      "ending_positions":int(np.count_nonzero(np.array(z["ending_shares"])>1e-7))})
            baseline[f"{feat}|{k}|initial_topk_hold"]=s
    banks=("601288.SH","601398.SH","601939.SH","601988.SH")
    bankidx=np.array([np.where(codes==c)[0][0] for c in banks if np.any(codes==c)],dtype=int)
    first=int(np.searchsorted(days,"2023-01-03"))
    bz=account_sim(days,codes,a,{first:bankidx})
    bs=summarize(bz["curve"],COST["account_money"])
    bs.update({"feature":"project_four_bank_account","k":len(bankidx),"rebalance":"buy_and_hold","status":"benchmark",
               "trades":bz["ntr"],"fees":bz["fees"],"buy_notional":bz["buy_notional"],"sell_notional":bz["sell_notional"],
               "round_trip_turnover_on_initial_equity":(bz["buy_notional"]+bz["sell_notional"])/COST["account_money"],
               "blocked_entries":bz["blocked_entries"],"blocked_exits":bz["blocked_exits"],
               "ending_positions":int(np.count_nonzero(np.array(bz["ending_shares"])>1e-7))})
    # Gross static universe equal-weight baseline at the initial OOS date; fractional units by design.
    uv=a["open"][first]; ua=a["adj_factor"][first]
    ok=np.isfinite(uv)&(uv>0)&np.isfinite(ua)&(ua>0)
    weights=ok.astype(float)/max(1,int(ok.sum()))
    daily=[]
    for i in range(first+1,len(days)):
        ratio=np.divide(a["open"][i]*a["adj_factor"][i],a["open"][i-1]*a["adj_factor"][i-1],
                    out=np.ones(len(codes)),where=np.isfinite(a["open"][i])&(a["open"][i]>0)&np.isfinite(a["adj_factor"][i])&(a["adj_factor"][i]>0)&np.isfinite(a["open"][i-1])&(a["open"][i-1]>0)&np.isfinite(a["adj_factor"][i-1])&(a["adj_factor"][i-1]>0))
        r=ratio-1.0
        daily.append(float(np.sum(weights*np.nan_to_num(r,nan=0.0))))
    ug=float(np.prod(1+np.array(daily))-1)
    baseline["project_four_bank_account"]=bs
    baseline["static_equalweight_universe_gross"]={"feature":"all_initially_quoted_names","k":int(ok.sum()),
          "rebalance":"buy_and_hold_gross","cumulative_net_return":ug,"annualized_return":max(1+ug,1e-12)**(242/max(1,len(daily)))-1,
          "max_drawdown":None,"fees":0.0,"status":"benchmark"}
    output={"built_at":time.strftime("%Y-%m-%d %H:%M:%S"),"snapshot_built_at":meta.get("built_at"),
            "oos_range":["2023-01-03",last],"universe_codes":len(codes),
            "screen_selected":screen["selected"],"screen_status":"exploratory" if screen["exploratory_fallback"] else "confirmation",
            "protocol_path":str(PROTOCOL),"worker_count_requested":workers,
            "worker_count_after_resource_gate":n if rest else min(2,len(tasks)),
            "grid":[v["summary"] for v in results.values()],
            "baselines":baseline,
            "curves":{k:v["curve"] for k,v in results.items()},
            "trades":{k:v["trades"] for k,v in results.items()}}
    op=OUT/"oos_results.json"; op.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame(output["grid"]).to_csv(OUT/"oos_summary.csv",index=False)
    pd.DataFrame([{"feature":k,**v} for k,v in baseline.items()]).to_csv(OUT/"baselines.csv",index=False)
    print("OOS_SAVED",op,"grid",len(results),"latest",last,flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=["screen","oos"]); p.add_argument("--workers",type=int,default=2)
    x=p.parse_args()
    if x.mode=="screen":run_screen()
    else:run_oos(max(2,x.workers))
if __name__=="__main__":main()

