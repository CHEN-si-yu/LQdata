#!/usr/bin/env python3
"""Lifecycle extension for frozen V6 242-session signal window; separate outputs only."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[k]="1"
import hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v6_base_run",HERE/"run.py")
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
from account_engine import account_sim, summarize, COST
SIGNAL_START="2025-07-01"; SIGNAL_END="2026-06-30"; LIFE_END="2026-07-01"
TAG="lifecycle_20250701_20260701"; N_SIGNAL=242; N_LIFE=243
NAMES={"monthly_top1":"V6月频Top1","monthly_top2":"月频Top2","equal_four_hold":"四股等权持有"}
CAP={"monthly_top1":1,"monthly_top2":2,"equal_four_hold":4}

def quarter_rows(curve):
    d=pd.DataFrame([r for r in curve if r["date"]<=SIGNAL_END])
    q=pd.PeriodIndex(pd.to_datetime(d.date),freq="Q").astype(str); out=[]
    for p in sorted(set(q)):
        g=d.loc[q==p]; net=float(np.prod(1+g.daily_return.to_numpy(float))-1)
        match=float(np.prod(1+g.matched_return.to_numpy(float))-1)
        out.append({"quarter":str(p),"days":len(g),"net_return":net,
          "same_hold_no_fee_return":match,"friction_difference":net-match,"fees":float(g.fee_today.sum())})
    return out

def residual(curve,a,n):
    last=np.full(n,np.nan); worst=0.
    for i,r in enumerate(curve):
        op,cl=a["open"][i],a["close"][i]
        mark=np.where(np.isfinite(op)&(op>0),op,np.where(np.isfinite(cl)&(cl>0),cl,last))
        mark=np.where(np.isfinite(mark)&(mark>0),mark,last)
        mv=float(np.sum(np.asarray(r["shares"],float)*np.nan_to_num(mark,nan=0.)))
        worst=max(worst,abs(float(r["equity"])-float(r["cash"])-mv)); last=mark.copy()
    return worst

def main():
    t0=time.time(); panel=base.M.load_panel()
    all_days=np.asarray(panel.days,dtype=str); codes=np.asarray(panel.codes,dtype=str)
    if tuple(codes)!=tuple(base.M.CODES): raise RuntimeError("unexpected universe")
    aa={k:panel.prices.raw[k] for k in ("open","high","low","pre_close","close","vol","adj_factor")}
    aa["amount"]=panel.amount
    ix=np.flatnonzero((all_days>=SIGNAL_START)&(all_days<=LIFE_END))
    days=all_days[ix]; a={k:v[ix] for k,v in aa.items()}
    if len(days)!=N_LIFE or days[0]!=SIGNAL_START or days[-1]!=LIFE_END: raise RuntimeError("expected lifecycle dates missing")
    schedules={}; signals={}
    for k,key in ((1,"monthly_top1"),(2,"monthly_top2")):
        gs,rows=base.make_schedule(all_days,codes,aa,SIGNAL_START,SIGNAL_END,k)
        if any(r["signal_date"]>SIGNAL_END or r["execution_date"]>SIGNAL_END for r in rows): raise RuntimeError("signal window leaked")
        schedules[key]={int(i-ix[0]):v for i,v in gs.items() if ix[0]<=i<=ix[-1]}
        signals[key]=rows
    schedules["equal_four_hold"]={1:np.arange(len(codes),dtype=int)}
    if any(i>=N_SIGNAL for key in ("monthly_top1","monthly_top2") for i in schedules[key]): raise RuntimeError("July 1 not exit-only")
    frozen=json.loads((HERE/"result.json").read_text())["official_242d"]["strategies"]
    outputs={}; trades_all=[]
    for key,sch in schedules.items():
        led=account_sim(days,codes,a,sch); curve=led["curve"]; fee_date={}
        for tr in led["trades"]:
            fee_date[tr["date"]]=fee_date.get(tr["date"],0.)+float(tr["fee"])
            tr2=dict(tr); tr2["strategy"]=key; trades_all.append(tr2)
        for row in curve: row["fee_today"]=fee_date.get(row["date"],0.)
        stat=summarize(curve,COST["account_money"]); signal_curve=[r for r in curve if r["date"]<=SIGNAL_END]
        signal_ret=float(np.prod(1+np.asarray([r["daily_return"] for r in signal_curve],float))-1)
        final=curve[-1]; exit_trades=[x for x in led["trades"] if x["date"]==LIFE_END]
        cash_min=min(float(x["cash"]) for x in curve); endpos=int(np.count_nonzero(np.asarray(led["ending_shares"])>1e-7))
        last_signal=max((x["execution_date"] for x in signals.get(key,[])),default=None)
        checks={
          "243_rows":len(curve)==N_LIFE,"signal_dates_through_2026_06_30":all(x["signal_date"]<x["execution_date"]<=SIGNAL_END for x in signals.get(key,[])),
          "2026_07_01_exit_only":all(x["side"]=="sell" for x in exit_trades),"cash_nonnegative":cash_min>=-1e-7,
          "holding_cap":max(int(x["holdings"]) for x in curve)<=CAP[key],"flat_after_exit":endpos==0,
          "blocked_entries_zero":int(led["blocked_entries"])==0,"blocked_exits_zero":int(led["blocked_exits"])==0,
          "ledger_residual_lt_1e_8":residual(curve,a,len(codes))<1e-8,"signal_window_242_rows":len(signal_curve)==N_SIGNAL,
          "final_date_2026_07_01":final["date"]==LIFE_END}
        audit={"checks":checks,"pass":all(checks.values()),"signal_count":len(signals.get(key,[])),
          "last_momentum_execution_date":last_signal,"exit_only_date":LIFE_END,"daily_rows":len(curve),
          "cash_min":cash_min,"ending_cash":float(led["ending_cash"]),"ending_positions":endpos,
          "max_holdings":max(int(x["holdings"]) for x in curve),"blocked_entries":int(led["blocked_entries"]),
          "blocked_exits":int(led["blocked_exits"]),"trade_count":int(led["ntr"]),"fees_total":float(led["fees"]),
          "exit_day_trades":len(exit_trades),"exit_day_fees":sum(float(x["fee"]) for x in exit_trades),
          "exit_day_sides":[x["side"] for x in exit_trades],"max_abs_equity_cash_mark_residual":residual(curve,a,len(codes))}
        outputs[key]={"stats_lifecycle":stat,"signal_window_return_through_2026_06_30_open":signal_ret,
          "signal_window_quarterly":quarter_rows(curve),"final_day":{"date":LIFE_END,"net_daily_return":float(final["daily_return"]),
            "same_hold_no_fee_return":float(final["matched_return"]),"fees":float(final["fee_today"]),
            "trade_count":len(exit_trades),"ending_equity":float(final["equity"]),"ending_cash":float(final["cash"])},
          "frozen_242d_reference":{"net_return":frozen[key]["stats"]["cumulative_net_return"],
            "max_drawdown":frozen[key]["stats"]["max_drawdown"]},
          "audit":audit,"curve":curve,"trades":led["trades"]}
    metrics={"unit":"V6 lifecycle extension; frozen official 242-row output remains unchanged",
      "signal_window":[SIGNAL_START,SIGNAL_END],"lifecycle_window":[SIGNAL_START,LIFE_END],
      "signal_window_sessions":N_SIGNAL,"lifecycle_sessions":N_LIFE,
      "rule":"No parameter changes. Momentum signals/rebalances end by 2026-06-30; 2026-07-01 is exit-only liquidation at next session open.",
      "data_snapshot_built_at":panel.data_built_at,"data_snapshot_last_day":str(all_days[-1]),
      "universe":[str(c) for c in codes],"no_training":True,"no_parameter_changes":True,"costs":COST,
      "strategies":{k:{a:b for a,b in v.items() if a not in ("curve","trades")} for k,v in outputs.items()},
      "overall_audit_pass":all(v["audit"]["pass"] for v in outputs.values()),"runtime_seconds":time.time()-t0}
    metrics["script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE/f"{TAG}_result.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (HERE/f"{TAG}_signals.json").write_text(json.dumps(signals,ensure_ascii=False,indent=2),encoding="utf-8")
    (HERE/f"{TAG}_audit.json").write_text(json.dumps({k:v["audit"] for k,v in outputs.items()},ensure_ascii=False,indent=2),encoding="utf-8")
    all_trades=[]
    for key,v in outputs.items():
        pd.DataFrame(v["curve"]).drop(columns=["shares"]).assign(strategy=key).to_csv(HERE/f"{TAG}_{key}_daily.csv",index=False)
        for x in v["trades"]:
            x2=dict(x); x2["strategy"]=key; all_trades.append(x2)
    pd.DataFrame(all_trades).to_csv(HERE/f"{TAG}_trades.csv",index=False)
    report=["# V6 生命周期对照：冻结官方信号窗并延迟至次日清仓","",
      "- 信号窗固定为2025-07-01至2026-06-30（242个交易日）；生命周期延至2026-07-01开盘，在该日统一强制清仓。",
      "- 2026-07-01不生成信号或新调仓；月频Top1、月频Top2及四股等权持有均使用原V6账本、资金和费用。",
      "- 原有V6 242行结果及官方输出文件未覆盖；生命周期结果单独保存。","",
      "## 生命周期总收益","",
      "| 策略 | 生命周期净收益（含7/1清仓） | 信号窗收益（截至6/30开盘） | 生命周期最大回撤 | 总费用 | 总成交 | 7/1净日收益 | 7/1费用 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
    name_rows=[("monthly_top1","V6月频Top1"),("monthly_top2","月频Top2"),("equal_four_hold","四股等权持有")]
    for key,title in name_rows:
        x=outputs[key]; s=x["stats_lifecycle"]; f=x["final_day"]; au=x["audit"]
        report.append(f"| {title} | {s['cumulative_net_return']:.4%} | {x['signal_window_return_through_2026_06_30_open']:.4%} | {s['max_drawdown']:.4%} | CNY {au['fees_total']:.2f} | {au['trade_count']} | {f['net_daily_return']:.8%} | CNY {f['fees']:.2f} |")
    report += ["","## 信号窗季度收益（不含7月1日清仓日）","","| 季度 | 策略 | 净收益 | 同持仓无费收益 | 摩擦差 | 费用 |","|---|---|---:|---:|---:|---:|"]
    for key,title in name_rows:
        for q in outputs[key]["signal_window_quarterly"]:
            report.append(f"| {q['quarter']} | {title} | {q['net_return']:.4%} | {q['same_hold_no_fee_return']:.4%} | {q['friction_difference']:.4%} | CNY {q['fees']:.2f} |")
    report += ["","## 2026-07-01精确清仓日","","| 策略 | 净日收益 | 同持仓无费收益 | 费用 | 清仓成交数 | 清仓后权益 |","|---|---:|---:|---:|---:|---:|"]
    for key,title in name_rows:
        f=outputs[key]["final_day"]
        report.append(f"| {title} | {f['net_daily_return']:.10%} | {f['same_hold_no_fee_return']:.10%} | CNY {f['fees']:.2f} | {f['trade_count']} | CNY {f['ending_equity']:.2f} |")
    report += ["","## 审计",""]
    for key,title in name_rows:
        x=outputs[key]["audit"]
        report.append(f"- {title}：{'通过' if x['pass'] else '失败'}；信号数{x['signal_count']}，最后动量执行日{x['last_momentum_execution_date']}；7/1成交方向{x['exit_day_sides']}；最低现金CNY {x['cash_min']:.2f}；期末持仓{x['ending_positions']}；阻塞买/卖{x['blocked_entries']}/{x['blocked_exits']}；账本残差CNY {x['max_abs_equity_cash_mark_residual']:.3g}。")
    report += ["","7月1日净日收益计入6月30日开盘至7月1日开盘的持仓价格变动及开盘清仓成本，不是7月1日收盘收益。各季度只累计2025-07-01至2026-06-30的242个信号窗交易日。",""]
    (HERE/f"{TAG}_REPORT.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps({"report":str(HERE/f"{TAG}_REPORT.md"),"audit_pass":metrics["overall_audit_pass"],
      "strategies":{k:{"lifecycle_net":v["stats_lifecycle"]["cumulative_net_return"],
        "signal_window_net":v["signal_window_return_through_2026_06_30_open"],
        "final_day_net":v["final_day"]["net_daily_return"],"mdd":v["stats_lifecycle"]["max_drawdown"],
        "fees":v["audit"]["fees_total"],"final_day_fees":v["final_day"]["fees"],"audit_pass":v["audit"]["pass"]}
        for k,v in outputs.items()}},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
