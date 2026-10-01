#!/usr/bin/env python3
"""Execution-aware monthly hybrid portfolio ledger for V21."""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"): os.environ[k]="1"
import argparse, json, math, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
V11_ROOT=ROOT.parent/"V11"
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
        p=V11_ROOT/"quarters"/q/"predictions.csv"
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
    ranker_top1={}; v11_top2={}; mom={}; hybrid={}; all_signal=[]
    dates=sorted(pred.loc[pred.monthly_signal,"trade_date"].unique())
    adj_close=prices["adj_close"]
    for d in dates:
        di=idx[d]; all_signal.append(d)
        sub=pred[pred.trade_date==d]
        rank=pd.to_numeric(sub["daily_rank"],errors="coerce")
        first=sub.loc[rank==1].sort_values("stock_code").stock_code.tolist()
        if len(first)!=1: raise RuntimeError(f"{d}: LambdaRank Top1 count={len(first)}")
        top2=sub.loc[sub.selected_top2].sort_values(["daily_rank","stock_code"]).stock_code.tolist()
        if len(top2)!=2: raise RuntimeError(f"{d}: V11 top2 count={len(top2)}")
        ranker_top1[d]={first[0]:1.0}
        v11_top2[d]={x:0.5 for x in top2}
        if di<60: raise RuntimeError(f"{d}: insufficient history for 60-session benchmark")
        trailing=adj_close[di]/adj_close[di-60]-1.0
        order=sorted(range(len(CODES)),key=lambda j:(-(trailing[j] if np.isfinite(trailing[j]) else -np.inf),CODES[j]))
        momentum_codes=[CODES[j] for j in order[:2]]
        mom[d]={c:0.5 for c in momentum_codes}
        weights={first[0]:0.5}
        for c in momentum_codes:
            weights[c]=weights.get(c,0.0)+0.25
        if abs(sum(weights.values())-1.0)>1e-12: raise RuntimeError(f"{d}: hybrid weights do not sum to 1")
        hybrid[d]=weights
    equal={dates[0]:{c:0.25 for c in CODES}}
    return {
      "V21 50% Ranker Top1 + 50% 60d Top2":hybrid,
      "V18 LambdaRank Top1":ranker_top1,
      "V11 LambdaRank Top2":v11_top2,
      "60d momentum Top2":mom,
      "equal-weight hold":equal
    },dates

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
    ap=argparse.ArgumentParser()
    ap.add_argument("--until",choices=None,default=None,help="optional checkpoint quarter, e.g. 2020Q4")
    args=ap.parse_args()
    qroot=V11_ROOT/"quarters"
    quarters=sorted(p.name for p in qroot.iterdir() if p.is_dir() and (p/"result.json").is_file())
    if len(quarters)!=26: raise SystemExit(f"expected all 26 completed V11 quarters, found {len(quarters)}")
    if args.until:
        if args.until not in quarters: raise SystemExit(f"unknown quarter {args.until}")
        quarters=quarters[:quarters.index(args.until)+1]
    missing=[q for q in quarters if not (qroot/q/"predictions.csv").is_file()]
    if missing: raise SystemExit(f"prediction caches missing: {missing}")
    pred=load_predictions(quarters); price=load_prices()
    oos_days=sorted(pred.trade_date.unique())
    if not oos_days: raise RuntimeError("no prediction dates")
    strategies,signal_dates=schedules(pred,price)
    outputs=[]; alltrades=[]; metrics_by={}
    for name,sig in strategies.items():
        d,tr,m=simulate(name,price,sig,oos_days[0],oos_days[-1])
        outputs.append(d); alltrades.extend(tr); metrics_by[name]=m
    daily=pd.concat(outputs,ignore_index=True)
    daily.to_csv(ROOT/"daily_equity.csv",index=False,float_format="%.10g")
    pd.DataFrame(alltrades).to_csv(ROOT/"trades.csv",index=False)
    target_rows=[]
    for name,sig in strategies.items():
        for d,target in sig.items():
            for code,weight in sorted(target.items()):
                target_rows.append({"signal_date":d,"strategy":name,"stock_code":code,"target_weight":float(weight)})
    target_df=pd.DataFrame(target_rows)
    target_df.to_csv(ROOT/"targets.csv",index=False,float_format="%.10g")
    for freq,filename in (("Y","annual_returns.csv"),("Q","quarterly_returns.csv")):
        periods=[r for d in outputs for r in period_returns(d,freq)]
        pd.DataFrame(periods).to_csv(ROOT/filename,index=False,float_format="%.10g")
    candidate_name="V21 50% Ranker Top1 + 50% 60d Top2"
    candidate=metrics_by[candidate_name]
    baselines=("V18 LambdaRank Top1","V11 LambdaRank Top2","60d momentum Top2","equal-weight hold")
    for baseline in baselines:
        candidate["vs_"+baseline.replace(" ","_").replace("-","_")] = {
          "net_return_difference_pp":100*(candidate["net_return"]-metrics_by[baseline]["net_return"]),
          "drawdown_difference_pp":100*(candidate["max_drawdown"]-metrics_by[baseline]["max_drawdown"])}
    # Verify additive monthly weights and ledger invariants from the generated artifacts.
    hybrid=strategies[candidate_name]; hybrid_sum_errors=[]
    for d,target in hybrid.items():
        hybrid_sum_errors.append(abs(sum(target.values())-1.0))
    overlap_count=0
    top1=strategies["V18 LambdaRank Top1"]; momentum=strategies["60d momentum Top2"]
    for d in signal_dates:
        code=next(iter(top1[d]))
        if code in momentum[d]: overlap_count+=1
    negative_cash=int((daily.cash < -1e-7).sum())
    negative_equity=int((daily.equity_net < -1e-7).sum())
    audit={
      "quarter_count":len(quarters),"signal_count":len(signal_dates),
      "hybrid_weight_sum_min":float(min(sum(x.values()) for x in hybrid.values())),
      "hybrid_weight_sum_max":float(max(sum(x.values()) for x in hybrid.values())),
      "hybrid_max_target_weight":float(max(max(x.values()) for x in hybrid.values())),
      "months_with_ranker_momentum_overlap":overlap_count,
      "target_weight_sum_max_abs_error":float(max(hybrid_sum_errors,default=0.0)),
      "daily_rows":int(len(daily)),"trade_rows":int(len(alltrades)),
      "negative_cash_daily_rows":negative_cash,"negative_equity_daily_rows":negative_equity,
      "max_abs_accounting_residual":float(daily.accounting_residual.abs().max()),
      "max_positions_by_strategy":{n:int(daily.loc[daily.strategy==n,"positions"].max()) for n in strategies},
      "min_cash_by_strategy":{n:float(daily.loc[daily.strategy==n,"cash"].min()) for n in strategies},
      "blocked_events_by_strategy":{n:int(metrics_by[n]["blocked_trade_events"]) for n in strategies}
    }
    (ROOT/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    # Record exact cache identities and basic dimensions; source files remain read-only.
    cache=[]
    for q in quarters:
        path=qroot/q/"predictions.csv"
        frame=pred.loc[pred.model_quarter.astype(str)==q] if "model_quarter" in pred.columns else pd.DataFrame()
        cache.append({"quarter":q,"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                      "rows":int(len(frame)),"bytes":int(path.stat().st_size)})
    cache_manifest={"source":"V11 cached quarter predictions; no training or score regeneration",
                    "quarter_count":len(cache),"quarters":cache}
    (ROOT/"cache_manifest.json").write_text(json.dumps(cache_manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    # Cross-check the recalculated pure-strategy references against their published ledgers.
    reference_checks={}
    for version,name in (("V11","V11 LambdaRank Top2"),("V11","60d momentum Top2"),
                         ("V11","equal-weight hold"),("V18","V18 LambdaRank Top1")):
        path=ROOT.parent/version/"metrics.json"
        reference=json.loads(path.read_text(encoding="utf-8"))
        expected=reference["metrics"][name]
        actual=metrics_by[name]
        reference_checks[f"{version}:{name}"]={
          "net_return_abs_difference":abs(float(actual["net_return"])-float(expected["net_return"])),
          "max_drawdown_abs_difference":abs(float(actual["max_drawdown"])-float(expected["max_drawdown"])),
          "reference_net_return":float(expected["net_return"]),
          "recomputed_net_return":float(actual["net_return"])}
    report={
      "strategy":candidate_name,"model":"V11 cached cross-sectional LightGBM LambdaRank scores, blended with fixed 60-session momentum",
      "quarters_included":quarters,"n_quarters":len(quarters),"test_dates":[oos_days[0],oos_days[-1]],
      "signal_count":len(signal_dates),
      "execution":"monthly close signal; next-session open execution; one account per strategy; shared cash; corrected V11 full-exit and split-adjusted odd-lot ledger",
      "allocation":{"ranker_top1":0.50,"momentum_top2_each":0.25,"overlap":"add component weights"},
      "costs":{"commission":COMMISSION,"minimum_commission":MIN_COMMISSION,"transfer":TRANSFER,"sell_stamp":STAMP,
               "adverse_slippage":SLIPPAGE,"max_participation":PARTICIPATION,"lot_size":LOT},
      "metrics":metrics_by,"audit":audit,"reference_checks":reference_checks,
      "annual_returns":{n:period_returns(d,"Y") for n,d in zip(strategies,outputs)},
      "quarter_returns":{n:period_returns(d,"Q") for n,d in zip(strategies,outputs)},
      "limitations":["Mark-to-market at adjusted share quantity and raw close; open positions are not liquidated at test end.",
                     "The tested universe contains four bank stocks; these results do not establish stable alpha."]
    }
    (ROOT/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    protocol={
      "status":"completed","protocol_file":"PROTOCOL.md","frozen_before_run":True,
      "strategy":candidate_name,"allocation":report["allocation"],
      "signal":"V11 monthly_signal dates; Ranker daily_rank 1 and 60-session adjusted-close momentum Top2",
      "execution":"V11 final corrected ledger; T+1 open; same cash, costs, participation, corporate-action and terminal mark rules for all strategies",
      "quarters":quarters,"signal_dates":signal_dates,"no_retraining":True,"no_parameter_sweep":True,
      "sources":["../V11/quarters/*/predictions.csv","../trainingdata/prices","../trainingdata/amount"],
      "outputs":["daily_equity.csv","trades.csv","targets.csv","annual_returns.csv","quarterly_returns.csv","metrics.json","audit.json","cache_manifest.json","SUMMARY.md"]
    }
    (ROOT/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    lines=["# V21 月频混合组合结果","",
      f"- 样本：{len(quarters)} 个季度（{quarters[0]} 至 {quarters[-1]}），OOS {oos_days[0]} 至 {oos_days[-1]}。",
      f"- 月度信号：{len(signal_dates)} 个；信号收盘生成，下一交易日开盘执行；期末按最后收盘盯市，不强制清仓。",
      "- V21 权重：Ranker Top1 50%；60 日动量 Top2 各 25%；同一股票重叠时权重相加。所有策略均为独立的 100,000 元单账户并使用同一 V11 修正成交账本。","",
      "| Strategy | Net return | Gross same fills | Max drawdown | Annual vol. | Fees | Slippage | Trades | Blocked | Min cash |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,m in metrics_by.items():
        lines.append(f"| {name} | {m['net_return']:.2%} | {m['gross_return_same_fills']:.2%} | {m['max_drawdown']:.2%} | {m['annualized_volatility']:.2%} | CNY {m['fees']:,.2f} | CNY {m['slippage_cost']:,.2f} | {m['trades']} | {m['blocked_trade_events']} | CNY {m['min_cash']:,.2f} |")
    lines+=["","## V21 相对基准",""]
    for baseline in baselines:
        item=candidate["vs_"+baseline.replace(" ","_").replace("-","_")]
        lines.append(f"- 相对 {baseline}：净收益差 {item['net_return_difference_pp']:+.2f} 个百分点；最大回撤差 {item['drawdown_difference_pp']:+.2f} 个百分点。")
    lines+=["","## 账本审计","",
      f"- 各月 V21 目标权重范围：{audit['hybrid_weight_sum_min']:.6f}–{audit['hybrid_weight_sum_max']:.6f}；最大单股权重 {audit['hybrid_max_target_weight']:.2%}；Ranker 与动量 Top2 重叠 {overlap_count}/{len(signal_dates)} 月。",
      f"- 最大记账残差：{audit['max_abs_accounting_residual']:.3g}；负现金行 {negative_cash}；V21 最大持仓数 {metrics_by[candidate_name]['max_positions']}；V21 阻塞成交事件 {metrics_by[candidate_name]['blocked_trade_events']}；最低现金 CNY {metrics_by[candidate_name]['min_cash']:.2f}。",
      "- 参考账本重算差异（净收益/回撤绝对差）："]
    for k,v in reference_checks.items():
        lines.append(f"  - {k}: {v['net_return_abs_difference']:.3g} / {v['max_drawdown_abs_difference']:.3g}")
    lines+=["","完整逐日权益、成交、信号目标、季度/年度统计及缓存散列见同目录 CSV/JSON。结果是四只银行股上的历史回测，不代表未来稳定收益。"]
    (ROOT/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    log=ROOT/"RUN_LOG.md"
    with log.open("a",encoding="utf-8") as f:
        f.write(f"\n- 运行完成：{len(quarters)} 个季度、{len(signal_dates)} 个月度信号；生成 {len(alltrades)} 条成交记录。\n")
        f.write(f"- V21 净收益 {candidate['net_return']:.4%}，最大回撤 {candidate['max_drawdown']:.4%}，最大持仓 {candidate['max_positions']}，阻塞成交 {candidate['blocked_trade_events']}，最低现金 CNY {candidate['min_cash']:.2f}。\n")
        f.write("- 数据/账本审计已写入 audit.json 与 metrics.json；没有训练模型，所有输出均写入 V21。\n")
    print("\n".join(lines))

if __name__=="__main__": main()
