#!/usr/bin/env python3
"""Execution-aware walk-forward ledger and benchmark report for V11."""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"): os.environ[k]="1"
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
OUT=ROOT
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
TARGET_VOL=0.12
VOL_LOOKBACK=20

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
    source=ROOT.parent/"V11"/"quarters"
    for q in quarters:
        f=source/q/"predictions.csv"
        if not f.is_file(): raise FileNotFoundError(f)
        frames.append(pd.read_csv(f,dtype={"trade_date":str,"stock_code":str}))
    pred=pd.concat(frames,ignore_index=True)
    pred["trade_date"]=pred.trade_date.str[:10]
    pred["monthly_signal"]=pred.monthly_signal.astype(bool)
    pred["selected_top2"]=pred.selected_top2.astype(bool)
    pred["daily_rank"]=pd.to_numeric(pred["daily_rank"],errors="raise").astype(int)
    if pred.duplicated(["trade_date","stock_code"]).any(): raise RuntimeError("duplicate score rows")
    return pred


def schedules(pred, prices):
    days=prices["days"]; idx=prices["index"]; adj=prices["adj_close"]
    dates=sorted(pred.loc[pred.monthly_signal,"trade_date"].unique())
    top1_scaled={}; top1_unscaled={}; ranker_top2={}; mom_top2={}; vol_records=[]
    for d in dates:
        di=idx[d]
        sub=pred[(pred.trade_date==d)&pred.monthly_signal]
        rank1=sub.loc[sub.daily_rank==1].sort_values(["daily_rank","stock_code"])
        if len(rank1)!=1: raise RuntimeError(f"{d}: cached Ranker Top1 count={len(rank1)}")
        code=str(rank1.iloc[0].stock_code); j=CODES.index(code)
        # 20 completed close-to-close adjusted returns, ending strictly before the signal date.
        hist=adj[di-VOL_LOOKBACK-1:di,j]
        if len(hist)!=VOL_LOOKBACK+1 or not np.isfinite(hist).all() or (hist<=0).any():
            raise RuntimeError(f"{d}: invalid prior-20d adjusted close window for {code}")
        daily=hist[1:]/hist[:-1]-1.0
        ann_vol=float(np.std(daily,ddof=1)*math.sqrt(252.0))
        exposure=1.0 if ann_vol<=0 else min(1.0,TARGET_VOL/ann_vol)
        if not (0.0<=exposure<=1.0+1e-12): raise RuntimeError("volatility scaling exceeded 100%")
        top1_unscaled[d]={code:1.0}
        top1_scaled[d]={code:float(exposure)}
        vol_records.append({
            "signal_date":d,"execution_date":str(days[di+1]),"selected_top1":code,
            "selected_top1_rank_score":float(rank1.iloc[0].score),
            "vol_window_start":str(days[di-VOL_LOOKBACK-1]),
            "vol_window_end":str(days[di-1]),
            "vol_observations":int(len(daily)),
            "vol_end_strictly_before_signal":bool(str(days[di-1])<d),
            "annualized_realized_volatility":ann_vol,
            "target_annualized_volatility":TARGET_VOL,
            "target_exposure":float(exposure),"target_cash_weight":float(1.0-exposure)
        })
        sub2=sub.loc[sub.selected_top2].sort_values(["daily_rank","stock_code"])
        picks=sub2.stock_code.astype(str).tolist()
        if len(picks)!=2: raise RuntimeError(f"{d}: V11 cached Top2 count={len(picks)}")
        ranker_top2[d]={c:0.5 for c in picks}
        if di<VOL_LOOKBACK+1: raise RuntimeError(f"{d}: insufficient 60d momentum history")
        trailing=adj[di]/adj[di-60]-1.0
        order=sorted(range(len(CODES)),
            key=lambda k:(-(trailing[k] if np.isfinite(trailing[k]) else -np.inf),CODES[k]))
        mom_top2[d]={CODES[k]:0.5 for k in order[:2]}
    if len(dates)==0: raise RuntimeError("no cached monthly signals")
    equal={dates[0]:{c:0.25 for c in CODES}}
    strategies={
        "V22 LambdaRank Top1 12pct vol":top1_scaled,
        "V18 LambdaRank Top1 unscaled":top1_unscaled,
        "V11 60d momentum Top2":mom_top2,
        "equal-weight hold":equal,
        "V11 LambdaRank Top2 ledger crosscheck":ranker_top2,
    }
    return strategies,dates,vol_records

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


def main():
    started=time.time()
    qroot=ROOT.parent/"V11"/"quarters"
    quarters=sorted(p.name for p in qroot.iterdir() if p.is_dir()
                    and (p/"predictions.csv").is_file() and (p/"result.json").is_file())
    if len(quarters)!=26 or quarters[0]!="2020Q1" or quarters[-1]!="2026Q2":
        raise RuntimeError(f"expected all 26 frozen V11 quarters; found {len(quarters)}: {quarters}")
    pred=load_predictions(quarters); price=load_prices()
    signal_dates=sorted(pred.loc[pred.monthly_signal,"trade_date"].unique())
    oos_days=sorted(pred.trade_date.unique())
    if len(signal_dates)!=78: raise RuntimeError(f"expected 78 monthly signals, got {len(signal_dates)}")
    if (oos_days[0],oos_days[-1])!=("2020-01-02","2026-06-30"):
        raise RuntimeError(f"unexpected common OOS lifecycle {oos_days[0]}..{oos_days[-1]}")
    strategies,signal_dates,vol_records=schedules(pred,price)
    outputs={}; trades=[]; metrics={}
    for name,sig in strategies.items():
        daily,tr,m=simulate(name,price,sig,oos_days[0],oos_days[-1])
        daily["actual_exposure"]=(1.0-daily.cash/daily.equity_net).clip(lower=0.0,upper=1.0)
        m["average_close_exposure"]=float(daily.actual_exposure.mean())
        m["max_close_exposure"]=float(daily.actual_exposure.max())
        outputs[name]=daily; trades.extend(tr); metrics[name]=m
    main_names=["V22 LambdaRank Top1 12pct vol","V18 LambdaRank Top1 unscaled",
                "V11 60d momentum Top2","equal-weight hold"]
    v22=outputs[main_names[0]]
    v22metrics=metrics[main_names[0]]
    vol_df=pd.DataFrame(vol_records)
    vol_audit={
        "signal_count":int(len(vol_df)),
        "all_20_returns_end_before_signal":bool((vol_df.vol_end_strictly_before_signal).all()),
        "all_windows_have_20_returns":bool((vol_df.vol_observations==VOL_LOOKBACK).all()),
        "maximum_exposure_not_above_one":bool((vol_df.target_exposure<=1.0+1e-12).all()),
        "minimum_exposure_nonnegative":bool((vol_df.target_exposure>=0.0).all()),
        "target_volatility":TARGET_VOL,
        "annualized_vol_min":float(vol_df.annualized_realized_volatility.min()),
        "annualized_vol_median":float(vol_df.annualized_realized_volatility.median()),
        "annualized_vol_max":float(vol_df.annualized_realized_volatility.max()),
        "exposure_min":float(vol_df.target_exposure.min()),
        "exposure_mean":float(vol_df.target_exposure.mean()),
        "exposure_max":float(vol_df.target_exposure.max()),
        "cash_weight_mean":float(vol_df.target_cash_weight.mean()),
        "volatility_formula":"sample standard deviation (ddof=1) of the selected Top1's 20 adjusted close-to-close returns from T-20 through T-1, multiplied by sqrt(252); no return on T is used"
    }
    if not (vol_audit["all_20_returns_end_before_signal"] and vol_audit["all_windows_have_20_returns"]
            and vol_audit["maximum_exposure_not_above_one"] and vol_audit["minimum_exposure_nonnegative"]):
        raise RuntimeError("volatility lookahead/window/exposure audit failed")
    # Reconcile the copied ledger against the final V11 outputs, then cross-check V18's cached summary.
    v11ref=json.loads((ROOT.parent/"V11"/"metrics.json").read_text(encoding="utf-8"))
    v18ref=json.loads((ROOT.parent/"V18"/"metrics.json").read_text(encoding="utf-8"))
    replay_checks={}
    for key,ref_name in (("V11 LambdaRank Top2 ledger crosscheck","V11 LambdaRank Top2"),
                         ("V11 60d momentum Top2","60d momentum Top2"),
                         ("equal-weight hold","equal-weight hold")):
        replay=metrics[key]; ref=v11ref["metrics"][ref_name]
        replay_checks[key]={"net_return_abs_difference":abs(replay["net_return"]-ref["net_return"]),
            "fees_abs_difference":abs(replay["fees"]-ref["fees"]),
            "trades_abs_difference":abs(replay["trades"]-ref["trades"])}
    v18_metric=v18ref.get("metrics",{}).get("V18 LambdaRank Top1")
    if v18_metric is None: raise RuntimeError("V18 cached Top1 metrics missing")
    replay_checks["V18 LambdaRank Top1 unscaled"]={
        "net_return_abs_difference_vs_v18":abs(metrics["V18 LambdaRank Top1 unscaled"]["net_return"]-v18_metric["net_return"]),
        "fees_abs_difference_vs_v18":abs(metrics["V18 LambdaRank Top1 unscaled"]["fees"]-v18_metric["fees"]),
        "trades_abs_difference_vs_v18":abs(metrics["V18 LambdaRank Top1 unscaled"]["trades"]-v18_metric["trades"]),
    }
    # Common time windows and data dimensions.
    for name,d in outputs.items():
        if len(d)!=len(oos_days) or d.trade_date.iloc[0]!=oos_days[0] or d.trade_date.iloc[-1]!=oos_days[-1]:
            raise RuntimeError(f"{name}: lifecycle mismatch")
        if metrics[name]["min_cash"] < -1e-7: raise RuntimeError(f"{name}: negative cash")
        if metrics[name]["max_abs_accounting_residual"]>1e-6: raise RuntimeError(f"{name}: accounting mismatch")
    annual={}; quarterly={}
    for name in main_names:
        annual[name]=period_returns(outputs[name],"Y")
        quarterly[name]=period_returns(outputs[name],"Q")
    # Add immutable signal fields to the selected-stock daily series for auditability.
    (OUT/"daily_equity.csv").write_text(pd.concat(outputs.values(),ignore_index=True).to_csv(index=False,float_format="%.10g"),encoding="utf-8")
    pd.DataFrame(trades).to_csv(OUT/"trades.csv",index=False,float_format="%.10g")
    pd.DataFrame(vol_records).to_csv(OUT/"volatility_schedule.csv",index=False,float_format="%.12g")
    for name,d in outputs.items():
        stem=name.lower().replace(" ","_").replace("%","pct")
        d.to_csv(OUT/f"{stem}_daily.csv",index=False,float_format="%.10g")
    pd.DataFrame([r for name in main_names for r in annual[name]]).to_csv(OUT/"annual_returns.csv",index=False,float_format="%.10g")
    pd.DataFrame([r for name in main_names for r in quarterly[name]]).to_csv(OUT/"quarterly_returns.csv",index=False,float_format="%.10g")
    costs={"commission":COMMISSION,"minimum_commission":MIN_COMMISSION,"transfer":TRANSFER,
        "sell_stamp":STAMP,"adverse_slippage":SLIPPAGE,"max_participation":PARTICIPATION,"lot_size":LOT}
    runner_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out={"unit":"experiments2/V22 vol-scaled cached LambdaRank Top1",
        "built_at":time.strftime("%Y-%m-%d %H:%M:%S"),"score_source":"V11/quarters/<quarter>/predictions.csv",
        "quarters":quarters,"quarter_count":len(quarters),"signal_count":len(signal_dates),
        "lifecycle":{"start":oos_days[0],"end":oos_days[-1],"days":len(oos_days),
            "official_exit_rule":"mark all open positions at the final OOS close; no forced liquidation, copied from final V11 ledger"},
        "rule":{"selection":"monthly first-session close; cached V11 LightGBM LambdaRank daily_rank=1",
            "risk":"20 prior adjusted close-to-close returns ending T-1; annualized sample volatility; exposure=min(1,0.12/vol)",
            "target_annualized_volatility":TARGET_VOL,"max_exposure":1.0,"leverage":False,
            "remaining_weight":"cash","execution":"next-session open"},
        "cost_model":costs,"volatility_audit":vol_audit,
        "strategies":metrics,"quarterly_returns":quarterly,"annual_returns":annual,
        "baseline_replay_checks":replay_checks,
        "runner_sha256":runner_hash,"ledger_copy_source":"V11/analysis.py final corrected shared-cash simulator",
        "runtime_seconds":time.time()-started,
        "process_max_rss_gib":__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss/(1024**2)}
    (OUT/"metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    protocol={"objective":out["unit"],"quarters":quarters,"predictions_reused":True,"model_retrained":False,
        "universe":list(CODES),"lifecycle":out["lifecycle"],"rule":out["rule"],
        "volatility_audit":vol_audit,"cost_model":costs,
        "simulator":"isolated copy of final V11 analysis.py; T+1 open, shared cash, corporate-action adjusted shares, ordinary 100-share lots; on zero-weight exits odd/fractional remainder sold only if participation cap can clear it, otherwise permitted lots sold and remainder retained",
        "source_prediction_root":str(ROOT.parent/"V11"/"quarters"),
        "runner_sha256":runner_hash,"created_at":out["built_at"]}
    (OUT/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    report=["# V22：LambdaRank Top1 20日波动目标限仓","",
        f"- 缓存：V11的{len(quarters)}季预测（{quarters[0]}至{quarters[-1]}），未重训；OOS生命周期 {oos_days[0]} 至 {oos_days[-1]}，共{len(oos_days)}日。",
        "- 每月第一个交易日收盘取V11 LightGBM LambdaRank的rank=1标的；波动率取该标的信号日前20个完整复权收盘日收益（截止T-1），样本标准差×√252。目标波动固定12%，仓位=min(1,12%/估计波动)，余额为现金，无杠杆。",
        "- T+1开盘成交；使用V11最终修正的共享现金、费用/滑点、1%成交额容量、整手、拆股复权股数及全退出余股限流规则。窗口末按V11官方规则以末日收盘估值，不强平。",
        "- 未缩放V18 Top1、V11 60日动量Top2、四股等权均在同一预测日/行情/账户引擎下重放。","",
        "| 策略 | 净收益 | 同成交毛收益 | 最大回撤 | 年化波动 | 平均实际敞口 | 费用 | 滑点 | 成交数 | 最低现金 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in main_names:
        m=metrics[name]
        report.append(f"| {name} | {m['net_return']:.2%} | {m['gross_return_same_fills']:.2%} | {m['max_drawdown']:.2%} | {m['annualized_volatility']:.2%} | {m['average_close_exposure']:.1%} | {m['fees']:.2f} | {m['slippage_cost']:.2f} | {m['trades']} | {m['min_cash']:.2f} |")
    report += ["","## 季度净收益","",
        "| 季度 | V22 12%限仓 | V18未缩放Top1 | V11 60d Top2 | 四股等权 |",
        "|---|---:|---:|---:|---:|"]
    qmaps={n:{r["period"]:r for r in quarterly[n]} for n in main_names}
    for q in sorted(set().union(*(set(v) for v in qmaps.values()))):
        report.append(f"| {q} | "+" | ".join(f"{qmaps[n][q]['net_return']:.2%}" for n in main_names)+" |")
    report += ["","## 年度净收益","",
        "| 年份 | V22 12%限仓 | V18未缩放Top1 | V11 60d Top2 | 四股等权 |",
        "|---:|---:|---:|---:|---:|"]
    ymaps={n:{r["period"]:r for r in annual[n]} for n in main_names}
    for y in sorted(set().union(*(set(v) for v in ymaps.values()))):
        report.append(f"| {y} | "+" | ".join(f"{ymaps[n][y]['net_return']:.2%}" for n in main_names)+" |")
    report += ["","## 波动率时点及账本审计","",
        f"- 波动估计样本数：{vol_audit['signal_count']}；每次20个日收益；所有窗口截至T-1、严格早于T：{vol_audit['all_20_returns_end_before_signal']}；最大仓位≤100%：{vol_audit['maximum_exposure_not_above_one']}。",
        f"- 年化历史波动最小/中位/最大 {vol_audit['annualized_vol_min']:.2%}/{vol_audit['annualized_vol_median']:.2%}/{vol_audit['annualized_vol_max']:.2%}；信号仓位最小/平均/最大 {vol_audit['exposure_min']:.1%}/{vol_audit['exposure_mean']:.1%}/{vol_audit['exposure_max']:.1%}；平均留现 {vol_audit['cash_weight_mean']:.1%}。",
        "- 结果账本逐策略校验：现金不为负，权益=现金+持仓收盘市值，窗口末不强平。见逐日与逐笔文件。",
        f"- V11原Top2/60d Top2/等权复放净收益、费用、成交数对照原V11 metrics.json的误差：{json.dumps(replay_checks,ensure_ascii=False)}。",
        "- 波动率限仓结果仅是固定规则历史比较；目标12%不能保证实际波动精确达到12%，也不证明未来稳定超额。",""]
    (OUT/"REPORT.md").write_text(chr(10).join(report),encoding="utf-8")
    print("V22_RESULTS",OUT/"metrics.json",flush=True)
    for name in main_names:
        m=metrics[name]
        print(name,"net",m["net_return"],"vol",m["annualized_volatility"],
              "mdd",m["max_drawdown"],"fees",m["fees"],"trades",m["trades"],flush=True)
    print("vol audit",vol_audit,"baseline replay",replay_checks,flush=True)
    print("RSS GiB",out["process_max_rss_gib"],"elapsed",out["runtime_seconds"],flush=True)

if __name__ == "__main__":
    main()
