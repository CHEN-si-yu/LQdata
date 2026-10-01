#!/usr/bin/env python3
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[k]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import sys,json,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
V1=ROOT.parent/"V1"
sys.path[:0]=[str(ROOT),str(V1)]
import model as M
import blend_common as C
START,END,EXEC_END="2025-07-01","2026-06-30","2026-07-01"
SCORE_BY_DATE={}
MODE="blend"

def load_scores():
    paths=[V1/"model_pred/ensemble/year=2025/data.parquet",V1/"model_pred/ensemble/year=2026/data.parquet"]
    df=pd.concat([pq.read_table(p,columns=["trade_date","stock_code","score"]).to_pandas() for p in paths],ignore_index=True)
    df=df[(df.trade_date>=START)&(df.trade_date<=END)]
    if df.duplicated(["trade_date","stock_code"]).any(): raise RuntimeError("duplicate scores")
    for date,g in df.groupby("trade_date",sort=False):
        SCORE_BY_DATE[str(date)]={str(r.stock_code):float(r.score) for r in g.itertuples()}
    return {"score_days":len(SCORE_BY_DATE),"score_rows":len(df),"min_day":min(SCORE_BY_DATE),"max_day":max(SCORE_BY_DATE)}

def ordinal(values,codes,valid):
    ordered=sorted(valid,key=lambda j:(float(values[j]),str(codes[j])))
    ranks=np.full(len(codes),-1,dtype=int)
    for rank,j in enumerate(ordered): ranks[j]=rank
    return ranks

def candidate_schedule(days,codes,arrays,start,end):
    close=arrays["close"]
    adj=pd.DataFrame(arrays["adj_factor"]).ffill().to_numpy(dtype=float)
    adjusted_close=close*adj
    schedule,signals={},[]
    for si in C.month_first_indices(days):
        d=str(days[si])
        if d<start or d>end or si<C.LOOKBACK or si+1>=len(days) or str(days[si+1])>end: continue
        day_scores=SCORE_BY_DATE.get(d)
        if day_scores is None: continue
        scores=np.array([day_scores.get(str(code),np.nan) for code in codes],float)
        old,now=adjusted_close[si-C.LOOKBACK],adjusted_close[si]
        valid=[j for j in range(len(codes)) if np.isfinite(scores[j]) and np.isfinite(old[j]) and old[j]>0 and np.isfinite(now[j]) and now[j]>0]
        if len(valid)<2: continue
        mom=np.full(len(codes),np.nan); mom[valid]=now[valid]/old[valid]-1
        sr=ordinal(scores,codes,valid); mr=ordinal(mom,codes,valid)
        if MODE=="momentum": key=lambda j:(mr[j],sr[j],str(codes[j]))
        elif MODE=="score": key=lambda j:(sr[j],mr[j],str(codes[j]))
        elif MODE=="blend": key=lambda j:(sr[j]+mr[j],sr[j],mr[j],str(codes[j]))
        else: raise RuntimeError(MODE)
        ranked=sorted(valid,key=key,reverse=True); chosen=ranked[:2]; ei=si+1
        schedule[ei]=np.asarray(chosen,dtype=int)
        signals.append({"signal_date":d,"execution_date":str(days[ei]),"mode":MODE,
          "selected":[str(codes[j]) for j in chosen],
          "ranked":[{"code":str(codes[j]),"score":float(scores[j]),"score_rank":int(sr[j]),
           "momentum_60d":float(mom[j]),"momentum_rank":int(mr[j]),"combined_rank":int(sr[j]+mr[j])} for j in ranked]})
    return schedule,signals

def period_stats(curve,freq):
    df=pd.DataFrame(curve); df=df[pd.to_datetime(df.date)<=pd.Timestamp(END)]; keys=pd.PeriodIndex(pd.to_datetime(df.date),freq=freq).astype(str); out=[]
    for p in sorted(set(keys)):
        g=df.loc[keys==p]; net=float(np.prod(1+g.daily_return.to_numpy(float))-1)
        matched=float(np.prod(1+g.matched_return.to_numpy(float))-1)
        out.append({"period":p,"days":len(g),"net_return":net,"matched_return":matched,"cost_execution_difference":net-matched})
    return out

def run_mode(days,codes,arrays,mode):
    global MODE
    MODE=mode; C.make_schedule=candidate_schedule
    return C.run_window(days,codes,arrays,START,EXEC_END)

def main():
    began=time.time(); score_audit=load_scores(); panel=M.load_panel()
    days,codes=panel.days,panel.codes
    arrays={k:panel.prices.raw[k] for k in ("open","high","low","pre_close","close","vol","adj_factor")}
    arrays["amount"]=panel.amount
    if tuple(str(c) for c in codes)!=tuple(M.CODES): raise RuntimeError("unexpected universe")
    if int(((days>=START)&(days<=END)).sum())!=242: raise RuntimeError("official window not 242 sessions")
    if score_audit["score_days"]!=242 or score_audit["score_rows"]!=242*4: raise RuntimeError(f"score coverage mismatch {score_audit}")
    runs={m:run_mode(days,codes,arrays,m) for m in ("momentum","score","blend")}
    for r in runs.values():
        r["selection_net_gain_pp"]={
          "versus_initial_top2_hold":100*(r["strategy"]["cumulative_net_return"]-r["initial_top2_hold"]["cumulative_net_return"]),
          "versus_four_bank_equal_weight_hold":100*(r["strategy"]["cumulative_net_return"]-r["four_bank_equal_weight_hold"]["cumulative_net_return"])}
    strategies={}
    for mode,r in runs.items():
        strategies[mode]={"stats":r["strategy"],"audit":r["audit"],"signals":r["signals"],
          "initial_top2_hold":r["initial_top2_hold"],"four_bank_equal_weight_hold":r["four_bank_equal_weight_hold"],
          "baseline_audit":r["baseline_audit"],"selection_net_gain_pp":r["selection_net_gain_pp"],
          "quarterly":period_stats(r["curve"],"Q"),"yearly":period_stats(r["curve"],"Y"),
          "execution_end_daily_return":float(r["curve"][-1]["daily_return"])}
    h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result={"unit":"experiments2/V7 score-momentum rank fusion","snapshot_built_at":panel.data_built_at,
      "snapshot_last_day":str(days[-1]),"signal_window":[START,END],"execution_end":EXEC_END,"universe":[str(c) for c in codes],
      "protocol":{"signal":"first trading session each month at close; execute next trading session open",
       "score":"cached V1 ensemble cross-sectional ordinal rank","momentum":"adjusted close(T)/adjusted close(T-60 sessions)-1 ordinal rank",
       "primary_rule":"equal weight of score and 60-session momentum ranks; Top2, equal weight",
       "ablations":["score-only Top2","momentum-only Top2"],"no_retraining":True,"no_parameter_sweep":True,
       "execution":"same audited cash ledger; 100-share lots, capacity, limit locks, fees and slippage"},
      "score_input_audit":score_audit,"strategies":strategies,"script_sha256":h,"runtime_seconds":time.time()-began,
      "interpretation":"Same-window comparison; matched gap measures costs/execution friction only, not stock-selection alpha."}
    (ROOT/"model_info/result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    protocol={**result["protocol"],"window":[START,END],"execution_end":EXEC_END,"snapshot_built_at":panel.data_built_at,
      "score_source":str(V1/"model_pred/ensemble/score_meta.json"),
      "score_source_sha256":hashlib.sha256((V1/"model_pred/ensemble/score_meta.json").read_bytes()).hexdigest(),
      "model_loader_sha256":hashlib.sha256((V1/"model.py").read_bytes()).hexdigest(),
      "engine_sha256":hashlib.sha256((ROOT/"rank_portfolio_lab.py").read_bytes()).hexdigest(),
      "script_sha256":h,"created_at":time.strftime("%Y-%m-%d %H:%M:%S")}
    (ROOT/"model_info/protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    for mode,r in runs.items():
        pd.DataFrame(r["curve"]).drop(columns=["shares"]).to_csv(ROOT/f"model_pred/{mode}_daily.csv",index=False)
        pd.DataFrame(r["trades"]).to_csv(ROOT/f"model_pred/{mode}_trades.csv",index=False)
        pd.DataFrame(r["signals"]).to_json(ROOT/f"model_pred/{mode}_signals.json",orient="records",force_ascii=False,indent=2)
    report=["# V7 四大行模型分数×60日动量融合（月频Top2）","",
      f"- 信号窗口：{START} 至 {END}（242个交易日）；统一计入 {EXEC_END} 的T+1清仓执行日。数据快照 {panel.data_built_at}，数据截至 {days[-1]}。",
      "- 固定规则：月初首个交易日收盘，对 V1 缓存集成分数和过去60个交易间隔复权收益分别横截面排序；等权平均序位，选Top2，下一交易日开盘成交。",
      "- 不重训、不调权重、不扫参数。score-only 和 momentum-only 是同窗组成策略消融，均使用同一账户成本。",
      "- 同持仓匹配差只度量执行与成本，不代表选股alpha。","",
      "| 策略 | 净收益 | 同持仓无费收益 | 成本/执行摩擦差 | 最大回撤 | 平均敞口 | 费用 | 交易 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode,r in runs.items():
        s=r["strategy"]; a=r["audit"]
        report.append(f"| {mode} | {s['cumulative_net_return']:.2%} | {s['matched_benchmark_return']:.2%} | {s['matched_cumulative_excess']:.2%} | {s['max_drawdown']:.2%} | {s['avg_exposure']:.1%} | ¥{a['fees']:.2f} | {a['trade_count']} |")
    report += ["","## 融合规则的固定持有基线","","| 基线 | 净收益 | 最大回撤 | 费用 | 交易 | 融合策略收益差 |","|---|---:|---:|---:|---:|---:|"]
    fusion=runs["blend"]
    for label,key,gain,audit_key in [
      ("首期融合Top2后买持","initial_top2_hold","versus_initial_top2_hold","initial_top2_hold"),
      ("四股等权买持","four_bank_equal_weight_hold","versus_four_bank_equal_weight_hold","four_bank_equal_weight_hold")]:
        st=fusion[key]; au=fusion["baseline_audit"][audit_key]
        report.append(f"| {label} | {st['cumulative_net_return']:.2%} | {st['max_drawdown']:.2%} | ¥{au['fees']:.2f} | {au['trade_count']} | {fusion['selection_net_gain_pp'][gain]:+.2f}pp |")
    report += ["","## 分季度收益","","| 策略 | 季度 | 净收益 | 成本/执行摩擦差 |","|---|---|---:|---:|"]
    for mode,r in runs.items():
        for q in period_stats(r["curve"],"Q"): report.append(f"| {mode} | {q['period']} | {q['net_return']:.2%} | {q['cost_execution_difference']:.2%} |")
    report += ["","## 7月1日退出日收益","",
      "| 策略 | 退出日净收益 | 生命周期累计收益 |",
      "|---|---:|---:|"]
    for mode,r in runs.items():
        report.append(f"| {mode} | {r['curve'][-1]['daily_return']:.2%} | {r['strategy']['cumulative_net_return']:.2%} |")
    report += ["","## 账务与时序检查",""]
    for mode,r in runs.items():
        a=r["audit"]
        report.append(f"- {mode}: {a['days']}日，{a['monthly_signals']}个信号；现金最低¥{a['cash_min']:.2f}，最多{a['maximum_simultaneous_holdings']}只，期末持仓{a['ending_positions']}只，阻塞买/卖{a['blocked_entries']}/{a['blocked_exits']}，账务残差¥{a['max_abs_equity_cash_plus_engine_mark_residual']:.3g}。")
    report += ["","- 信号仅用当日收盘前已得数据，成交为下一交易日开盘，未使用未来行情。",
      "- 本候选需与两个组成策略比较；未跑赢则淘汰，跑赢仍需独立年份/滚动确认。",""]
    (ROOT/"REPORT.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps({m:{"net":r["strategy"]["cumulative_net_return"],"mdd":r["strategy"]["max_drawdown"],
      "fees":r["audit"]["fees"],"trades":r["audit"]["trade_count"],"quarterly":period_stats(r["curve"],"Q")}
      for m,r in runs.items()},ensure_ascii=False))
if __name__=="__main__": main()
