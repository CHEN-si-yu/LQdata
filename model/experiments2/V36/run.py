#!/usr/bin/env python3
"""V36: equal 50/50 allocation to two independently frozen Top2 signals."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
V30 = BASE / "V30"
V26 = BASE / "V26"
V34 = BASE / "V34"
sys.path.insert(0, str(V30))
import cash_engine as engine

START, END = "2024-01-02", "2026-06-30"
FACTOR26 = "mf_tier_flow_agreement_20"
FACTOR34 = "id2_close_vs_pm_vwap_20"
NAMES = {
    "mix": "V36 V26+V34 50/50 Top2 mix",
    "v26": "V26 mf_tier_flow_agreement_20 Top2",
    "v34": "V34 id2_close_vs_pm_vwap_20 Top2",
    "v11": "V11 LambdaRank Top2",
    "hold": "equal-weight hold",
}
QUARTERS = [f"{y}Q{q}" for y in range(2024, 2027) for q in range(1, 5) if not (y == 2026 and q > 2)]

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def json_write(path: Path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def main():
    protocol_path=ROOT/"protocol.json"
    protocol=json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("version")!="V36" or protocol.get("rule_frozen_before_backtest") is not True:
        raise RuntimeError("V36 protocol must be frozen before run")
    if protocol.get("holdout_dates")!=[START,END] or protocol.get("sleeve_weights")!=[0.5,0.5]:
        raise RuntimeError("V36 frozen protocol mismatch")
    protocol_sha=sha(protocol_path)
    v26_path=V26/"membership_audit.json"; v34_path=V34/"membership_audit.json"
    v26_sha,v34_sha=sha(v26_path),sha(v34_path)
    m26=json.loads(v26_path.read_text(encoding="utf-8")); m34=json.loads(v34_path.read_text(encoding="utf-8"))
    if len(m26)!=30 or len(m34)!=30: raise RuntimeError("expected exactly 30 frozen holdout memberships")
    for rows,factor in ((m26,FACTOR26),(m34,FACTOR34)):
        if any(r.get("factor_name")!=factor or r.get("factor_timestamp_is_signal_date") is not True or r.get("no_future_row_used") is not True for r in rows):
            raise RuntimeError(f"source membership failed no-lookahead audit: {factor}")
    pred=engine.load_predictions(QUARTERS)
    pred=pred[(pred.trade_date>=START)&(pred.trade_date<=END)].copy()
    pred_dates=sorted(pred.loc[pred.monthly_signal,"trade_date"].unique().tolist())
    prices=engine.load_prices()
    baseline,dates=engine.schedules(pred,prices)
    if dates!=pred_dates or len(dates)!=30: raise RuntimeError("monthly signal calendar mismatch")
    by26={r["signal_date"]:r for r in m26}; by34={r["signal_date"]:r for r in m34}
    if sorted(by26)!=dates or sorted(by34)!=dates: raise RuntimeError("factor membership dates mismatch")
    mix={}; monthly=[]
    codes=tuple(engine.CODES); idx=prices["index"]; days=prices["days"]
    for d in dates:
        r26,r34=by26[d],by34[d]
        if r26.get("execution_date")!=str(days[idx[d]+1]) or r34.get("execution_date")!=str(days[idx[d]+1]):
            raise RuntimeError(f"{d}: component execution must be next-session open")
        p26,p34=list(r26["selected_top2"]),list(r34["selected_top2"])
        if len(p26)!=2 or len(p34)!=2: raise RuntimeError(f"{d}: unexpected Top2 size")
        weights={c:0.0 for c in codes}
        for c in p26: weights[c]+=0.25
        for c in p34: weights[c]+=0.25
        if abs(sum(weights.values())-1.0)>1e-12: raise RuntimeError(f"{d}: allocation does not sum to 1")
        overlap=sorted(set(p26)&set(p34))
        mix[d]={c:w for c,w in weights.items() if w>0}
        row={"signal_date":d,"execution_date_t1_open":str(days[idx[d]+1]),"v26_top2":";".join(p26),"v34_top2":";".join(p34),"overlap_names":";".join(overlap),"overlap_count":len(overlap),"v26_allocation":0.5,"v34_allocation":0.5,"target_weight_sum":sum(weights.values())}
        row.update({f"target_weight_{c}":weights[c] for c in codes}); monthly.append(row)
    schedule26={r["signal_date"]:{c:0.5 for c in r["selected_top2"]} for r in m26}
    schedule34={r["signal_date"]:{c:0.5 for c in r["selected_top2"]} for r in m34}
    first=dates[0]
    schedules={NAMES["mix"]:mix,NAMES["v26"]:schedule26,NAMES["v34"]:schedule34,NAMES["v11"]:baseline[NAMES["v11"]],NAMES["hold"]:baseline[NAMES["hold"]]}
    all_daily=[]; all_trades=[]; metrics={}
    for name,sched in schedules.items():
        daily,trades,res=engine.simulate(name,prices,sched,START,END)
        if res["blocked_trade_events"]!=0 or res["min_cash"] < -1e-8 or res["max_abs_accounting_residual"]>1e-7:
            raise RuntimeError(f"ledger checks failed: {name}")
        daily["gross_return_daily"]=daily.equity_gross.pct_change().fillna(0.0)
        stem=name.replace(" ","_").replace("/","_")
        daily.to_csv(ROOT/(stem+"_daily.csv"),index=False,float_format="%.10g")
        pd.DataFrame(trades).to_csv(ROOT/(stem+"_trades.csv"),index=False,float_format="%.10g")
        all_daily.append(daily); all_trades.extend(trades); metrics[name]=res
    dailies=pd.concat(all_daily,ignore_index=True); trades=pd.DataFrame(all_trades)
    dailies.to_csv(ROOT/"daily_equity.csv",index=False,float_format="%.10g")
    trades.to_csv(ROOT/"trades.csv",index=False,float_format="%.10g")
    pd.DataFrame(monthly).to_csv(ROOT/"monthly_target_weights.csv",index=False,float_format="%.12g")
    qret=[]; yret=[]
    for _,g in dailies.groupby("strategy",sort=False):
        qret.extend(engine.period_returns(g.reset_index(drop=True),"Q")); yret.extend(engine.period_returns(g.reset_index(drop=True),"Y"))
    pd.DataFrame(qret).to_csv(ROOT/"quarterly_returns.csv",index=False,float_format="%.10g")
    pd.DataFrame(yret).to_csv(ROOT/"annual_returns.csv",index=False,float_format="%.10g")
    overlaps={str(k):int(v) for k,v in pd.Series([r["overlap_count"] for r in monthly]).value_counts().sort_index().items()}
    mixres=metrics[NAMES["mix"]]
    comparisons={n:{"net_return_difference_pp":100*(mixres["net_return"]-metrics[n]["net_return"]),"drawdown_difference_pp":100*(mixres["max_drawdown"]-metrics[n]["max_drawdown"])} for n in list(NAMES.values())[1:]}
    audit={"holdout":[START,END],"signal_count":len(dates),"same_signal_dates_as_v11":True,"v26_membership_sha256":v26_sha,"v34_membership_sha256":v34_sha,"v26_and_v34_signal_date_values_and_next_open_execution_verified":True,"no_lookahead_flags_passed_for_both_components":True,"monthly_weights_sum_to_one":bool(np.allclose([r["target_weight_sum"] for r in monthly],1.0,rtol=0,atol=1e-12)),"overlap_months_by_count":overlaps,"ledger_checks":{n:{"min_cash":r["min_cash"],"blocked_trade_events":r["blocked_trade_events"],"max_abs_accounting_residual":r["max_abs_accounting_residual"],"trades":r["trades"]} for n,r in metrics.items()},"protocol_unchanged_since_freeze":sha(protocol_path)==protocol_sha}
    json_write(ROOT/"audit.json",audit)
    json_write(ROOT/"metrics.json",{"holdout":[START,END],"metrics":metrics,"comparisons_vs_mix":comparisons,"overlap_months":overlaps})
    hashes={"run.py":sha(ROOT/"run.py"),"protocol.json_frozen_before_run":protocol_sha,"cash_engine.py":sha(V30/"cash_engine.py"),"V26/membership_audit.json":v26_sha,"V34/membership_audit.json":v34_sha}
    json_write(ROOT/"hashes.json",hashes)
    lines=["# V36 — V26 与 V34 因子 Top2 固定 50/50 组合","",f"严格留出期 {START} 至 {END}，共 {len(dates)} 个共同月初信号。组合及权重在回测前冻结；两条独立信号各分配总权益 50%，各自在 Top2 两只股票上分配 25%，重叠股票权重相加。","","| 策略 | 净收益 | 最大回撤 | 年化波动 | 费用 | 滑点 | 成交 | 最低现金 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n,r in metrics.items(): lines.append(f"| {n} | {r['net_return']:.2%} | {r['max_drawdown']:.2%} | {r['annualized_volatility']:.2%} | ¥{r['fees']:,.2f} | ¥{r['slippage_cost']:,.2f} | {r['trades']} | ¥{r['min_cash']:.2f} |")
    lines += ["","## 组合相对各基线","",*[f"- {n}: 净收益差 {x['net_return_difference_pp']:+.2f} 个百分点；回撤差 {x['drawdown_difference_pp']:+.2f} 个百分点。" for n,x in comparisons.items()],"","## 审计","",f"- 成分重叠月数（0/1/2只）：`{json.dumps(overlaps,ensure_ascii=False)}`。","- V26/V34 成分均来自信号日因子，下一交易日开盘执行；两者信号日期无未来行标记通过。","- 全组合权重每月和为 100%；所有账本现金非负、阻塞成交为 0，最大核算残差低于 1e-7。","- 收益为四家银行的历史留出结果，不代表未来稳定超额。",f"- SHA-256：run.py `{hashes['run.py']}`，冻结协议 `{protocol_sha}`。","","详细月度权重、逐日权益、成交、季度/年度结果及审计见本目录其它输出。"]
    (ROOT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__": main()
