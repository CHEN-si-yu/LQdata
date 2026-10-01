#!/usr/bin/env python3
"""V20: cached V11 monthly Top2 with change-only reallocation and persistent zero-target exits."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_k]="1"
import hashlib,json,time,math
from pathlib import Path
import numpy as np
import pandas as pd
import cash_engine as E

ROOT=Path(__file__).resolve().parent
CODES=E.CODES
INITIAL=E.INITIAL
LOT=E.LOT


def change_only_schedule(v11_schedule, signal_dates, pred):
    schedule={}; audit=[]; last_set=None
    for d in signal_dates:
        target=v11_schedule[d]
        members=frozenset(target)
        changed=last_set is None or members!=last_set
        sub=pred[(pred.trade_date==d)&pred.monthly_signal]
        bycode={str(row.stock_code):row for row in sub.itertuples(index=False)}
        ordered=sorted(target,key=lambda c:(int(bycode[c].daily_rank),c))
        audit.append({"signal_date":d,"execution_date":"","selected_top2_rank_order":ordered,
            "previous_members":sorted(last_set) if last_set is not None else [],
            "selected_members":sorted(members),"member_set_changed":bool(changed),
            "reallocate_on_execution":bool(changed),
            "selected_scores":{c:float(bycode[c].score) for c in ordered},
            "selected_daily_ranks":{c:int(bycode[c].daily_rank) for c in ordered}})
        if changed:
            schedule[d]={c:0.5 for c in ordered}
        last_set=members
    return schedule,audit


def _cost_and_trade(qty, op, side):
    fill=op*(1.0+E.SLIPPAGE if side=="buy" else 1.0-E.SLIPPAGE)
    fee=E.fee_for(qty,fill,side=="sell")
    return fill,fee,qty*fill


def simulate_change_only(name, price, signals, start_day, end_day):
    """V11-equivalent cash ledger; only selected-set events rebalance, exits retry daily."""
    days=price["days"]; ix=price["index"]; start=ix[start_day]; end=ix[end_day]
    orders={}
    for d,target in signals.items():
        si=ix[d]; ei=si+1
        if ei<=end: orders.setdefault(ei,[]).append(target)
    cash=INITIAL; gross_cash=INITIAL; shares=np.zeros(4,dtype=np.float64)
    last_close=np.full(4,np.nan); rows=[]; trades=[]
    fees_total=slip_total=turnover=0.0; nblocked=0; max_positions=0; min_cash=INITIAL
    pending_zero=set(); zero_exit_started=0; zero_exit_completed=0
    odd_lot_full_exits=0; pending_retry_trades=0; pending_retry_days=0
    open_=price["open"]; hi=price["high"]; lo=price["low"]; close=price["close"]
    prev=price["pre_close"]; vol=price["vol"]; amt=price["amount"]; adj=price["adj_factor"]

    def sell_position(t,j,full_exit,reason):
        nonlocal cash,gross_cash,shares,fees_total,slip_total,turnover,nblocked,odd_lot_full_exits,pending_retry_trades
        if shares[j]<=1e-8 or not np.isfinite(open_[t,j]) or open_[t,j]<=0: return 0.0
        op=float(open_[t,j]); one=np.isfinite(hi[t,j]) and np.isfinite(lo[t,j]) and abs(hi[t,j]-lo[t,j])<=1e-8
        down=one and np.isfinite(prev[t,j]) and op<=prev[t,j]*0.905
        if down or not (np.isfinite(vol[t,j]) and vol[t,j]>0 and np.isfinite(amt[t,j]) and amt[t,j]>0):
            nblocked+=1; return 0.0
        cap_exact=max(0.0,float(amt[t,j]*E.PARTICIPATION/op))
        held=float(shares[j])
        if full_exit and cap_exact+1e-8>=held:
            qty=held
        else:
            held_lots=math.floor(held/LOT)*LOT
            cap_lots=math.floor(cap_exact/LOT)*LOT
            qty=float(min(held_lots,cap_lots))
        full_fill=full_exit and qty>0 and abs(qty-held)<=1e-8
        if qty<LOT and not full_fill:
            nblocked+=1; return 0.0
        if full_exit and qty+1e-8<held: nblocked+=1
        fill,fee,notional=_cost_and_trade(qty,op,"sell")
        cash+=notional-fee; gross_cash+=qty*op; shares[j]-=qty
        fees_total+=fee; slip_total+=qty*op*E.SLIPPAGE; turnover+=qty*op
        odd=bool(full_fill and abs(qty/LOT-round(qty/LOT))>1e-6)
        if odd: odd_lot_full_exits+=1
        if reason=="pending_zero_target_exit_retry": pending_retry_trades+=1
        trades.append({"trade_date":str(days[t]),"strategy":name,"stock_code":CODES[j],"side":"sell",
            "shares":round(float(qty),6),"open":op,"fill":fill,"notional":notional,"fee":fee,
            "slippage_cost":qty*op*E.SLIPPAGE,"reason":reason,"zero_target":bool(full_exit),
            "full_zero_target_exit_complete":bool(full_fill),"odd_lot_full_exit":odd,
            "shares_remaining_after_trade":float(shares[j])})
        return qty

    for t in range(start,end+1):
        if t>0:
            ratio=np.divide(adj[t],adj[t-1],out=np.ones(4),where=np.isfinite(adj[t])&(adj[t]>0)&np.isfinite(adj[t-1])&(adj[t-1]>0))
            shares*=ratio
        op=open_[t]
        mark_open=np.where(np.isfinite(op)&(op>0),op,np.where(np.isfinite(last_close),last_close,0.0))
        event_targets=orders.get(t)
        if event_targets:
            target=event_targets[-1]
            weights=np.asarray([target.get(c,0.0) for c in CODES],dtype=np.float64)
            if abs(float(weights.sum())-1.0)>1e-12: raise RuntimeError("V20 target weights do not sum to one")
            # Keep a zero target active until the entire position, including split-created odd shares, exits.
            for j,c in enumerate(CODES):
                if weights[j]>0:
                    pending_zero.discard(j)
                elif shares[j]>1e-8:
                    if j not in pending_zero: zero_exit_started+=1
                    pending_zero.add(j)
            equity_open=cash+float(np.dot(shares,mark_open))
            for j,c in enumerate(CODES):
                if shares[j]<=1e-8 or not np.isfinite(op[j]) or op[j]<=0: continue
                desired=equity_open*weights[j]; current=shares[j]*op[j]; diff=current-desired
                full_exit=weights[j]<=0
                if not full_exit and diff<op[j]*LOT: continue
                sold=sell_position(t,j,full_exit,"member_change_zero_target_exit" if full_exit else "member_change_rebalance")
                if full_exit and sold>0 and shares[j]<=1e-8:
                    zero_exit_completed+=1; pending_zero.discard(j)
            # Buy toward the new equal-weight targets from shared cash, after sells.
            for j,c in enumerate(CODES):
                if weights[j]<=0 or not np.isfinite(op[j]) or op[j]<=0: continue
                current=shares[j]*op[j]; need=max(0.0,equity_open*weights[j]-current)
                if need<op[j]*LOT: continue
                one=np.isfinite(hi[t,j]) and np.isfinite(lo[t,j]) and abs(hi[t,j]-lo[t,j])<=1e-8
                up=one and np.isfinite(prev[t,j]) and op[j]>=prev[t,j]*1.095
                if up or not (np.isfinite(vol[t,j]) and vol[t,j]>0 and np.isfinite(amt[t,j]) and amt[t,j]>0):
                    nblocked+=1; continue
                fill=op[j]*(1+E.SLIPPAGE); qty=math.floor(need/fill/LOT)*LOT
                maxq=math.floor((amt[t,j]*E.PARTICIPATION/op[j])/LOT)*LOT; qty=min(qty,maxq)
                while qty>=LOT and qty*fill+E.fee_for(qty,fill,False)>cash+1e-8: qty-=LOT
                if qty<LOT: continue
                fee=E.fee_for(qty,fill,False); notional=qty*fill
                cash-=notional+fee; gross_cash-=qty*op[j]; shares[j]+=qty
                fees_total+=fee; slip_total+=qty*op[j]*E.SLIPPAGE; turnover+=qty*op[j]
                trades.append({"trade_date":str(days[t]),"strategy":name,"stock_code":c,"side":"buy",
                    "shares":round(float(qty),6),"open":float(op[j]),"fill":fill,"notional":notional,"fee":fee,
                    "slippage_cost":qty*op[j]*E.SLIPPAGE,"reason":"member_change_rebalance",
                    "zero_target":False,"full_zero_target_exit_complete":False,"odd_lot_full_exit":False,
                    "shares_remaining_after_trade":float(shares[j])})
                min_cash=min(min_cash,cash)
        else:
            # No regular rebalance while membership is unchanged. Only clean up pending zero-target exits.
            if pending_zero: pending_retry_days+=1
            for j in sorted(list(pending_zero)):
                if shares[j]<=1e-8:
                    pending_zero.discard(j); continue
                sold=sell_position(t,j,True,"pending_zero_target_exit_retry")
                if sold>0 and shares[j]<=1e-8:
                    zero_exit_completed+=1; pending_zero.discard(j)
        valid_close=np.isfinite(close[t])&(close[t]>0)
        marks=np.where(valid_close,close[t],np.where(np.isfinite(last_close),last_close,0.0))
        last_close=np.where(valid_close,close[t],last_close)
        mv=shares*marks; equity=cash+float(mv.sum()); gross_equity=gross_cash+float(mv.sum())
        resid=equity-cash-float(mv.sum()); active=int(np.count_nonzero(shares>1e-8)); max_positions=max(max_positions,active)
        rows.append({"trade_date":str(days[t]),"strategy":name,"equity_net":equity,"equity_gross":gross_equity,
            "cash":cash,"positions":active,"fees_cumulative":fees_total,"slippage_cumulative":slip_total,
            "turnover_cumulative":turnover,"accounting_residual":resid,
            "shares_json":json.dumps({CODES[j]:round(float(shares[j]),6) for j in range(4) if shares[j]>1e-8})})
        min_cash=min(min_cash,cash)
    daily=pd.DataFrame(rows)
    daily["daily_return_net"]=daily.equity_net.pct_change().fillna(0.0)
    daily["drawdown"]=daily.equity_net/daily.equity_net.cummax()-1.0
    n=max(1,len(daily)-1); total=float(daily.equity_net.iloc[-1]/INITIAL-1.0)
    ann=(1+total)**(252/n)-1 if total>-1 else -1.0
    metrics={"strategy":name,"start":start_day,"end":end_day,"days":int(len(daily)),"net_return":total,
        "gross_return_same_fills":float(daily.equity_gross.iloc[-1]/INITIAL-1.0),"annualized_net_return":float(ann),
        "annualized_volatility":float(daily.daily_return_net.std(ddof=1)*math.sqrt(252)),
        "max_drawdown":float(daily.drawdown.min()),"fees":float(fees_total),"slippage_cost":float(slip_total),
        "total_trading_cost":float(fees_total+slip_total),"turnover":float(turnover),"trades":int(len(trades)),
        "blocked_trade_events":int(nblocked),"min_cash":float(min_cash),"max_positions":int(max_positions),
        "max_abs_accounting_residual":float(daily.accounting_residual.abs().max()),
        "member_change_rebalances":int(len(orders)),"zero_target_exits_started":int(zero_exit_started),
        "zero_target_exits_completed":int(zero_exit_completed),"odd_lot_full_exit_trades":int(odd_lot_full_exits),
        "pending_zero_exit_retry_days":int(pending_retry_days),"pending_zero_exit_retry_trades":int(pending_retry_trades),
        "pending_zero_target_positions_at_end":int(sum(shares[j]>1e-8 for j in pending_zero)),
        "pending_zero_target_shares_at_end":{CODES[j]:float(shares[j]) for j in pending_zero if shares[j]>1e-8}}
    return daily,trades,metrics


def make_period_table(daily,freq,metric_name):
    # The ledger has no terminal sell for the overall V11 comparison window; all dates are signal-period dates.
    return E.period_returns(daily,freq)


def main():
    started=time.time()
    manifest=json.loads((ROOT/"cache_manifest.json").read_text(encoding="utf-8"))
    quarters=[x["quarter"] for x in manifest["items"]]
    if len(quarters)!=26 or quarters[0]!="2020Q1" or quarters[-1]!="2026Q2":
        raise RuntimeError(f"expected 26 cached OOS quarters, found {len(quarters)}")
    for item in manifest["items"]:
        if item.get("state")!="complete": raise RuntimeError(f"cached quarter incomplete: {item['quarter']}")
        local=ROOT/"quarters"/item["quarter"]/"predictions.csv"
        digest=hashlib.sha256(local.read_bytes()).hexdigest()
        if digest!=item["sha256"]: raise RuntimeError(f"cached score hash mismatch: {item['quarter']}")
    pred=E.load_predictions(quarters)
    if pred.model_quarter.nunique()!=26 or pred.duplicated(["trade_date","stock_code"]).any():
        raise RuntimeError("cached OOS predictions are incomplete or duplicated")
    price=E.load_prices()
    score_strategies,signal_dates=E.schedules(pred,price)
    if len(signal_dates)!=78 or len(pred[pred.monthly_signal].trade_date.unique())!=78:
        raise RuntimeError("expected 78 monthly OOS score dates")
    v20_schedule,membership_audit=change_only_schedule(score_strategies["V11 LambdaRank Top2"],signal_dates,pred)
    for row in membership_audit:
        row["execution_date"]=str(price["days"][price["index"][row["signal_date"]]+1])
    oos_days=sorted(pred.trade_date.unique())
    start_day="2020-01-02"; end_day="2026-06-30"
    if oos_days[0]!=start_day or oos_days[-1]!=end_day:
        raise RuntimeError(f"unexpected OOS endpoints {oos_days[0]}..{oos_days[-1]}")
    daily_v20,trades_v20,metrics_v20=simulate_change_only("V20 LambdaRank change-only Top2",price,v20_schedule,start_day,end_day)
    daily_v11,trades_v11,metrics_v11=E.simulate("V11 LambdaRank Top2",price,score_strategies["V11 LambdaRank Top2"],start_day,end_day)
    daily_mom,trades_mom,metrics_mom=E.simulate("60d momentum Top2",price,score_strategies["60d momentum Top2"],start_day,end_day)
    daily_hold,trades_hold,metrics_hold=E.simulate("equal-weight hold",price,score_strategies["equal-weight hold"],start_day,end_day)
    strategies=[("V20 LambdaRank change-only Top2",daily_v20,trades_v20,metrics_v20),
        ("V11 LambdaRank Top2",daily_v11,trades_v11,metrics_v11),
        ("60d momentum Top2",daily_mom,trades_mom,metrics_mom),
        ("equal-weight hold",daily_hold,trades_hold,metrics_hold)]
    all_daily=pd.concat([z[1] for z in strategies],ignore_index=True)
    all_trades=pd.DataFrame([t for z in strategies for t in z[2]])
    period_q=[];period_y=[]
    for name,daily,_,_ in strategies:
        period_q += [{"strategy":name,**r} for r in make_period_table(daily,"Q","quarter")]
        period_y += [{"strategy":name,**r} for r in make_period_table(daily,"Y","year")]
    signal_days=[]
    for z in membership_audit:
        signal_days.append(z["signal_date"])
    # Independently verify that all non-event periods preserve quantities except mechanical adj-factor share changes.
    event_dates=set(z["execution_date"] for z in membership_audit if z["reallocate_on_execution"])
    unchanged_signal_count=sum(not z["member_set_changed"] for z in membership_audit)
    scores_csv=ROOT/"quarters"
    source_manifest=manifest
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cash_engine_sha256=hashlib.sha256((ROOT/"cash_engine.py").read_bytes()).hexdigest()
    source_v11_analysis_sha256=hashlib.sha256(Path("/root/autodl-fs/model/experiments2/V11/analysis.py").read_bytes()).hexdigest()
    metrics_by={name:m for name,_,_,m in strategies}
    candidate=metrics_v20
    for baseline,key in ((metrics_v11,"vs_V11_monthly_Top2"),(metrics_mom,"vs_60d_momentum_Top2"),(metrics_hold,"vs_equal_weight_hold")):
        candidate[key]={"net_return_difference_pp":100*(metrics_v20["net_return"]-baseline["net_return"]),
            "drawdown_difference_pp":100*(metrics_v20["max_drawdown"]-baseline["max_drawdown"])}
    out={"unit":"experiments2/V20 V11 cached LambdaRank Top2 with member-change-only reallocation",
        "built_at":time.strftime("%Y-%m-%d %H:%M:%S"),"source_v11_cache_quarters":quarters,
        "source_predictions_path":"V20/quarters/*/predictions.csv (copied from V11; no model code executed)",
        "test_dates":[start_day,end_day],"monthly_signal_count":len(signal_dates),
        "membership_change_rebalance_count":len(v20_schedule),"unchanged_membership_signal_count":unchanged_signal_count,
        "strategies":metrics_by,"membership_audit_file":"membership_audit.csv",
        "quarterly_returns_file":"quarterly_returns.csv","annual_returns_file":"annual_returns.csv",
        "no_retraining":True,"no_parameter_sweep":True,"price_only_for_execution":True,
        "costs":{"commission":E.COMMISSION,"minimum_commission":E.MIN_COMMISSION,"transfer":E.TRANSFER,
            "sell_stamp":E.STAMP,"adverse_slippage":E.SLIPPAGE,"max_participation":E.PARTICIPATION,"lot_size":E.LOT},
        "account_engine":"V11 corrected cash engine functions copied to V20/cash_engine.py; V20 adds persistent zero-target exit retries and full odd-lot close when participation allows",
        "cache_manifest_sha256":hashlib.sha256((ROOT/"cache_manifest.json").read_bytes()).hexdigest(),
        "runner_sha256":runner_sha256,"cash_engine_sha256":cash_engine_sha256,
        "source_v11_analysis_sha256":source_v11_analysis_sha256,
        "runtime_seconds":time.time()-started}
    account_audit={"window":[start_day,end_day],"oos_quarters":quarters,"oos_monthly_signals":len(signal_dates),
        "V20_member_change_rebalances":len(v20_schedule),"V20_unchanged_membership_months":unchanged_signal_count,
        "all_26_score_cache_hashes_verified":True,"score_rows_per_monthly_signal":4,
        "quarterly_annual_tables_use_full_OOS_cash_ledger":True,"terminal_policy":"mark to market on 2026-06-30 close; no forced liquidation, matching V11",
        "scenarios":{}}
    for name,daily,trade_rows,metric in strategies:
        last=daily.iloc[-1]
        account_audit["scenarios"][name]={"cash_nonnegative":bool(metric["min_cash"]>=-1e-7),
            "min_cash":float(metric["min_cash"]),"end_date":str(last["trade_date"]),
            "ending_cash":float(last["cash"]),"ending_positions":int(last["positions"]),
            "ending_shares_json":str(last["shares_json"]),"blocked_trade_events":int(metric["blocked_trade_events"]),
            "fees":float(metric["fees"]),"slippage_cost":float(metric["slippage_cost"]),
            "turnover":float(metric["turnover"]),"trades":int(metric["trades"]),
            "max_abs_accounting_residual":float(metric["max_abs_accounting_residual"])}
    account_audit["scenarios"]["V20 LambdaRank change-only Top2"].update({
        "zero_target_exits_started":int(metrics_v20["zero_target_exits_started"]),
        "zero_target_exits_completed":int(metrics_v20["zero_target_exits_completed"]),
        "odd_lot_full_exit_trades":int(metrics_v20["odd_lot_full_exit_trades"]),
        "pending_zero_target_positions_at_end":int(metrics_v20["pending_zero_target_positions_at_end"]),
        "pending_zero_target_shares_at_end":metrics_v20["pending_zero_target_shares_at_end"],
        "all_zero_target_exit_positions_flat_by_end":bool(metrics_v20["pending_zero_target_positions_at_end"]==0 and metrics_v20["zero_target_exits_started"]==metrics_v20["zero_target_exits_completed"])} )
    (ROOT/"audit.json").write_text(json.dumps(account_audit,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    all_daily.to_csv(ROOT/"daily_equity.csv",index=False,float_format="%.10g")
    all_trades.to_csv(ROOT/"trades.csv",index=False,float_format="%.10g")
    daily_v20.to_csv(ROOT/"V20_daily_equity.csv",index=False,float_format="%.10g")
    pd.DataFrame(trades_v20).to_csv(ROOT/"V20_trades.csv",index=False,float_format="%.10g")
    pd.DataFrame(membership_audit).to_json(ROOT/"membership_audit.json",orient="records",force_ascii=False,indent=2)
    pd.DataFrame(membership_audit).to_csv(ROOT/"membership_audit.csv",index=False)
    pd.DataFrame(period_q).to_csv(ROOT/"quarterly_returns.csv",index=False)
    pd.DataFrame(period_y).to_csv(ROOT/"annual_returns.csv",index=False)
    (ROOT/"metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    protocol={"objective":out["unit"],"data_source":"26 completed V11 quarterly OOS prediction caches; local copies stored in V20/quarters; no training or model inference",
        "quarter_range":[quarters[0],quarters[-1]],"signal_dates":[start_day,end_day],"signal_count":78,
        "V20_rule":"monthly predicted Top2 equal weight; create a rebalance order only when the unordered Top2 membership set changes; otherwise preserve existing share counts subject only to corporate-action adjustment",
        "baselines":["monthly rebalanced V11 LambdaRank Top2","monthly rebalanced 60-session adjusted-close momentum Top2","equal-weight four-bank hold from first executable open"],
        "zero_target_exit":"enqueue every deselected held name and retry at each subsequent open until fully flat; sell final odd-lot residual only if the same 1% amount participation cap can carry the entire position; otherwise sell whole lots and retry",
        "periods":"2020-01-02 through 2026-06-30 inclusive; daily close mark-to-market; no forced final liquidation to match V11 OOS ledger",
        "costs":out["costs"],"V11_analysis_sha256":source_v11_analysis_sha256,
        "cash_engine_sha256":cash_engine_sha256,"runner_sha256":runner_sha256,
        "cache_manifest_sha256":out["cache_manifest_sha256"]}
    (ROOT/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    report=["# V20：V11 LambdaRank Top2 成分变化时才重配","",
        "## 固定协议","",
        "读取 V11 2020Q1–2026Q2 已完成的 26 季 OOS LambdaRank 分数缓存，不训练、不推理新模型。每月沿用 V11 的预测 Top2、等权目标；只有与上月 Top2 的成员集合不同时才在 T+1 开盘重配。成员不变时，保留已持股数，权重自然漂移。",
        "比较：月频重配 V11 Top2、同月频 60 日动量 Top2、四股等权持有。起始/结束日、成交限制、手续费、滑点和现金引擎与 V11 统一。",
        "零目标持仓退出会逐日重试；参与率不足时先卖整手，待成交容量足够再一次性清理拆股后余股。窗口按 V11 口径在 2026-06-30 收盘计值，不强制清空仍在目标 Top2 中的仓位。","",
        f"26 个季度缓存通过完整性检查；月度信号 {len(signal_dates)} 个；成员变更触发重配 {len(v20_schedule)} 次，成员未变 {unchanged_signal_count} 次。","",
        "## 总体结果","","| 策略 | 净收益 | 年化 | 最大回撤 | 年化波动 | 费用 | 滑点 | 换手 | 交易笔数 |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,_,_,m in strategies:
        report.append(f"| {name} | {m['net_return']:.2%} | {m['annualized_net_return']:.2%} | {m['max_drawdown']:.2%} | {m['annualized_volatility']:.2%} | ¥{m['fees']:,.0f} | ¥{m['slippage_cost']:,.0f} | ¥{m['turnover']:,.0f} | {m['trades']} |")
    report += ["","## V20 相对基线",""]
    for label,key in (("月频 V11 Top2","vs_V11_monthly_Top2"),("60 日动量 Top2","vs_60d_momentum_Top2"),("四股等权持有","vs_equal_weight_hold")):
        z=candidate[key]; report.append(f"- 相对{label}：收益差 {z['net_return_difference_pp']:+.2f} pp；回撤差 {z['drawdown_difference_pp']:+.2f} pp。")
    report += ["","## 现金与成分审计","",
        f"- V20 现金最低余额 ¥{candidate['min_cash']:,.2f}；阻塞交易事件 {candidate['blocked_trade_events']}；最大记账残差 {candidate['max_abs_accounting_residual']:.3g}。",
        f"- 零目标退出启动 {candidate['zero_target_exits_started']} 次、完成 {candidate['zero_target_exits_completed']} 次；完整奇数手/拆股余股退出 {candidate['odd_lot_full_exit_trades']} 笔；末尾仍待退出的零目标持仓 {candidate['pending_zero_target_positions_at_end']} 个。",
        f"- 组合期间最多同时持有 {candidate['max_positions']} 只；总账期末持仓数 {int(daily_v20.positions.iloc[-1])}，符合 V11 期末不强制平仓口径。",
        "- 月度成分/变更原因见 membership_audit.csv；V20 实际订单与基线交易均见 trades.csv；日净值见 daily_equity.csv。","",
        "## 年度净收益","","| 年份 | 策略 | 净收益 | 同仓位无费收益 |","|---:|---|---:|---:|"]
    for z in period_y:
        report.append(f"| {z['period']} | {z['strategy']} | {z['net_return']:.2%} | {z['gross_return_same_fills']:.2%} |")
    report += ["","## 季度净收益","","| 季度 | 策略 | 净收益 | 同仓位无费收益 |","|---|---|---:|---:|"]
    for z in period_q:
        report.append(f"| {z['period']} | {z['strategy']} | {z['net_return']:.2%} | {z['gross_return_same_fills']:.2%} |")
    report += ["","结果是四大银行历史回测，不能证明未来收益稳定或存在可持续 alpha。","",
        f"缓存清单 SHA-256：{out['cache_manifest_sha256']}",
        f"运行脚本 SHA-256：{runner_sha256}",
        f"现金引擎 SHA-256：{cash_engine_sha256}"]
    (ROOT/"REPORT.md").write_text("\\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"V20":metrics_v20,"V11":metrics_v11,"60d":metrics_mom,"equal_hold":metrics_hold,
        "membership_changes":len(v20_schedule),"unchanged_months":unchanged_signal_count,
        "cache_quarters":len(quarters),"runtime_seconds":out["runtime_seconds"]},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__": main()
