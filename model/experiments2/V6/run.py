#!/usr/bin/env python3
"""Fixed no-retraining four-bank monthly 60-session momentum Top1 candidate."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_k]="1"
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
V1_DIR=Path("/root/autodl-fs/model/experiments2/V1"); sys.path.insert(0,str(V1_DIR))
import model as M
from account_engine import account_sim,summarize,COST
TEST=("2025-07-01","2026-06-30"); FULL_START="2023-01-03"; LOOKBACK=60; OUT=HERE
def month_first(days):
    m=np.array([str(d)[:7] for d in days]); return np.flatnonzero(np.r_[True,m[1:]!=m[:-1]])
def make_schedule(days,codes,a,start,end,k):
    adj=pd.DataFrame(a["adj_factor"]).ffill().to_numpy(float); px=a["close"]*adj
    sch={}; sig=[]
    for i in month_first(days):
        dt=str(days[i])
        if dt<start or dt>end or i<LOOKBACK or i+1>=len(days) or str(days[i+1])>end: continue
        old,now=px[i-LOOKBACK],px[i]; ok=np.isfinite(old)&(old>0)&np.isfinite(now)&(now>0)
        ranks=sorted(np.flatnonzero(ok).tolist(),key=lambda c:(-(now[c]/old[c]-1),str(codes[c])))
        pick=ranks[:k]
        if not pick: continue
        sch[i+1]=np.array(pick,int)
        sig.append({"signal_date":dt,"execution_date":str(days[i+1]),"lookback_date":str(days[i-LOOKBACK]),
          "ranked":[{"code":str(codes[c]),"return_60_session":float(now[c]/old[c]-1)} for c in ranks],
          "selected":[str(codes[c]) for c in pick]})
    assert all(x["signal_date"]<x["execution_date"] and x["lookback_date"]<=x["signal_date"] for x in sig)
    return sch,sig
def periods(curve,freq):
    d=pd.DataFrame(curve); keys=pd.PeriodIndex(pd.to_datetime(d.date),freq=freq).astype(str); out=[]
    for p in sorted(set(keys)):
        g=d.loc[keys==p]; net=float(np.prod(1+g.daily_return.to_numpy(float))-1)
        matched=float(np.prod(1+g.matched_return.to_numpy(float))-1)
        out.append({"period":str(p),"days":len(g),"net_return":net,"same_hold_no_fee_return":matched,
                    "friction_difference":net-matched})
    return out
def mark_residual(curve,a,n):
    last=np.full(n,np.nan); worst=0.
    for i,r in enumerate(curve):
        op,cl=a["open"][i],a["close"][i]
        mark=np.where(np.isfinite(op)&(op>0),op,np.where(np.isfinite(cl)&(cl>0),cl,last))
        mark=np.where(np.isfinite(mark)&(mark>0),mark,last)
        v=float(np.sum(np.asarray(r["shares"])*np.nan_to_num(mark,nan=0.)))
        worst=max(worst,abs(float(r["equity"])-float(r["cash"])-v)); last=mark.copy()
    return worst
def window(days,codes,aa,start,end):
    ix=np.flatnonzero((days>=start)&(days<=end))
    if not len(ix) or str(days[ix[0]])!=start or str(days[ix[-1]])!=end: raise RuntimeError(f"missing endpoints {start}..{end}")
    d=days[ix]; a={k:v[ix] for k,v in aa.items()}; configs={}; signals={}
    for k in (1,2):
        s,g=make_schedule(days,codes,aa,start,end,k)
        configs[f"monthly_top{k}"]={int(i-ix[0]):v for i,v in s.items() if ix[0]<=i<=ix[-1]}; signals[f"top{k}"]=g
    if not configs["monthly_top1"] or not configs["monthly_top2"]: raise RuntimeError("missing momentum schedule")
    configs["equal_four_hold"]={1:np.arange(len(codes),dtype=int)}
    out={}
    for name,s in configs.items():
        led=account_sim(d,codes,a,s); c=led["curve"]; stats=summarize(c,COST["account_money"])
        fees={}
        for t in led["trades"]: fees[t["date"]]=fees.get(t["date"],0.)+float(t["fee"])
        for row in c: row["fee_today"]=fees.get(row["date"],0.)
        cashmin=min(float(x["cash"]) for x in c); nend=int(np.count_nonzero(np.asarray(led["ending_shares"])>1e-7))
        audit={"days":len(d),"scheduled_rebalances":len(s),"cash_min":cashmin,"ending_cash":led["ending_cash"],
          "ending_positions":nend,"max_holdings":max(int(x["holdings"]) for x in c),
          "blocked_entries":led["blocked_entries"],"blocked_exits":led["blocked_exits"],"trade_count":led["ntr"],
          "fees":led["fees"],"cash_nonnegative":cashmin>=-1e-7,"final_liquidation_complete":nend==0,
          "max_abs_equity_cash_mark_residual":mark_residual(c,a,len(codes)),
          "signal_timing_pass":all(str(d[i-1])<str(d[i]) for i in s if i>0)}
        out[name]={"stats":stats,"audit":audit,"curve":c,"trades":led["trades"],
                   "quarterly":periods(c,"Q"),"yearly":periods(c,"Y")}
    return {"window":[start,end],"signals":signals,"strategies":out}
def summary(x):
    return {k:{a:b for a,b in v.items() if a not in ("curve","trades")} for k,v in x["strategies"].items()}
def write_report(path,meta,official,extended,spreads):
    names=[("monthly_top1","V6 monthly Top1"),("monthly_top2","Monthly Top2 baseline"),("equal_four_hold","Four-bank equal-weight hold")]
    lines=["# V6: Four-bank monthly 60-session relative momentum Top1","",
      f"- Snapshot: {meta['snapshot_built_at']}; data through {meta['last_day']}.",
      "- Fixed rule: on the first trading day close of each month, rank four banks by adjusted close return over the prior 60 trading intervals, hold Top1, and rebalance at next session open.",
      "- No model training and no parameter sweep. Top2 and equal-hold baselines use the same account ledger, date windows, costs, and execution rules.",
      "- Ledger: CNY 100,000 initial cash, 100-share lots, 1% signal-day amount capacity, limit-lock handling, 3bp slippage, commissions, transfer fee, and sell stamp duty.",
      "- Same-hold no-fee return isolates costs/execution friction. Net return spreads versus separate baselines are strategy comparisons, not alpha.","",
      "## Official window: 2025-07-01 to 2026-06-30 (242 sessions)","","| Strategy | Net return | MDD | Avg exposure | Fees | Trades | Same-hold no-fee | Friction gap |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key,title in names:
        x=official["strategies"][key]; s=x["stats"]; a=x["audit"]
        lines.append(f"| {title} | {s['cumulative_net_return']:.2%} | {s['max_drawdown']:.2%} | {s['avg_exposure']:.1%} | CNY {a['fees']:.2f} | {a['trade_count']} | {s['matched_benchmark_return']:.2%} | {s['matched_cumulative_excess']:.2%} |")
    lines += ["",f"Top1 minus Top2 net return: **{spreads['monthly_top2']['net_return_difference_pp']:+.2f} pp**; versus equal four-bank hold: **{spreads['equal_four_hold']['net_return_difference_pp']:+.2f} pp**. These are return spreads under a common ledger, not proof of factor alpha.","",
      "### Official quarterly results","","| Quarter | Strategy | Net | Same-hold no-fee | Friction gap |","|---|---|---:|---:|---:|"]
    for k,title in names:
        for r in official["strategies"][k]["quarterly"]:
            lines.append(f"| {r['period']} | {title} | {r['net_return']:.2%} | {r['same_hold_no_fee_return']:.2%} | {r['friction_difference']:.2%} |")
    lines += ["","## Extended window: 2023-01-03 to latest snapshot","","| Year | Strategy | Net | Same-hold no-fee | Friction gap |","|---|---|---:|---:|---:|"]
    for k,title in names:
        for r in extended["strategies"][k]["yearly"]:
            lines.append(f"| {r['period']} | {title} | {r['net_return']:.2%} | {r['same_hold_no_fee_return']:.2%} | {r['friction_difference']:.2%} |")
    lines += ["","## Ledger checks",""]
    for label,x in [("Official",official),("Extended",extended)]:
        for k,title in names:
            a=x["strategies"][k]["audit"]
            lines.append(f"- {label} {title}: min cash CNY {a['cash_min']:.2f}; max holdings {a['max_holdings']}; blocked buys/sells {a['blocked_entries']}/{a['blocked_exits']}; ending positions {a['ending_positions']}; max equity-cash-mark residual CNY {a['max_abs_equity_cash_mark_residual']:.3g}; signal timing {a['signal_timing_pass']}.")
    lines += ["","## Limits","","This is a fixed-rule historical backtest candidate. The extended period spans a limited number of market regimes, and Top1 is more concentrated than Top2. Historical return spreads do not establish stable alpha or future return stability.",""]
    path.write_text("\n".join(lines),encoding="utf-8")
def main():
    started=time.time(); panel=M.load_panel(); days=np.asarray(panel.days,dtype=str); codes=np.asarray(panel.codes,dtype=str)
    if tuple(codes)!=tuple(M.CODES): raise RuntimeError(f"unexpected bank universe: {codes}")
    aa={k:panel.prices.raw[k] for k in ("open","high","low","pre_close","close","vol","adj_factor")}; aa["amount"]=panel.amount
    if np.count_nonzero((days>=TEST[0])&(days<=TEST[1]))!=242: raise RuntimeError("official window is not 242 sessions")
    official=window(days,codes,aa,*TEST); latest=str(days[-1]); extended=window(days,codes,aa,FULL_START,latest)
    for label,x in (("official_242d",official),("extended",extended)):
        for k,v in x["strategies"].items():
            pd.DataFrame(v["curve"]).drop(columns=["shares"]).to_csv(OUT/f"{label}_{k}_daily.csv",index=False)
            tr=pd.DataFrame(v["trades"])
            if len(tr): tr.insert(0,"strategy",k)
            tr.to_csv(OUT/f"{label}_{k}_trades.csv",index=False)
        (OUT/f"{label}_signals.json").write_text(json.dumps(x["signals"],ensure_ascii=False,indent=2),encoding="utf-8")
    base=official["strategies"]; spreads={}
    for k in ("monthly_top2","equal_four_hold"):
        spreads[k]={"net_return_difference_pp":100*(base["monthly_top1"]["stats"]["cumulative_net_return"]-base[k]["stats"]["cumulative_net_return"]),
                    "interpretation":"separately traded baseline net-return spread; not alpha"}
    meta={"snapshot_built_at":panel.data_built_at,"last_day":latest}
    result={"unit":"experiments2/V6 fixed monthly 60-session four-bank Top1",
      "built_at":time.strftime("%Y-%m-%d %H:%M:%S"),**meta,"universe":[str(c) for c in codes],
      "rule":{"signal":"first trading session of month close","factor":"adjusted close at T / adjusted close T-60 sessions - 1",
        "selection":"rank four banks; hold Top1","execution":"next trading session open",
        "costs":COST,"no_training":True,"no_parameter_sweep":True,"lookback_sessions":LOOKBACK,
        "causal_timing":"signal uses values dated at/before close T; execution T+1 open"},
      "official_242d":{"window":list(TEST),"strategies":summary(official),"top1_spreads_vs_baselines":spreads},
      "extended":{"window":[FULL_START,latest],"strategies":summary(extended)},
      "runtime_seconds":time.time()-started}
    result["script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    protocol={"objective":"fixed no-retraining monthly four-bank relative momentum Top1",
      "prespecified_rule":"month first trading day close; 60-session adjusted price return; Top1; next session open",
      "baselines":["same-ledger monthly Top2","same-ledger four-bank equal-weight hold"],
      "official_window":list(TEST),"extended_window":[FULL_START,latest],"lookback_sessions":LOOKBACK,
      "no_training":True,"no_parameter_sweep":True,"data_snapshot_built_at":panel.data_built_at,
      "panel_loader":str(V1_DIR/"model.py"),"account_engine":"embedded account_engine.py",
      "source_sha256":result["script_sha256"],"created_at":result["built_at"]}
    (OUT/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    write_report(OUT/"REPORT.md",meta,official,extended,spreads)
if __name__=="__main__": main()
