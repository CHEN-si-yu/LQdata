#!/usr/bin/env python3
"""V23 fixed 60-day risk-adjusted momentum, using V11 OOS scores and cash engine."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_k]="1"
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import cash_engine as E

ROOT=Path(__file__).resolve().parent
START="2020-01-02"
END="2026-06-30"
LOOKBACK=60


def risk_adjusted_momentum_schedule(days,codes,price,signal_dates):
    ac=price["adj_close"]; index=price["index"]
    schedule={}; signals=[]
    for date in signal_dates:
        i=index[date]
        if i<LOOKBACK: raise RuntimeError(f"not enough history for {LOOKBACK}d score at {date}")
        # Total return and each of the 60 realized daily returns terminate at the signal close.
        old=ac[i-LOOKBACK]; now=ac[i]
        window=ac[i-LOOKBACK:i+1]
        returns=window[1:]/window[:-1]-1.0
        if len(returns)!=LOOKBACK: raise RuntimeError("risk window must contain exactly 60 return intervals")
        momentum=now/old-1.0
        volatility=np.std(returns,axis=0,ddof=1)
        valid=np.isfinite(momentum)&np.isfinite(volatility)&(volatility>0)
        if np.count_nonzero(valid)<2: raise RuntimeError(f"insufficient valid risk scores at {date}")
        score=np.full(len(codes),np.nan,dtype=float); score[valid]=momentum[valid]/volatility[valid]
        ranked=sorted(np.flatnonzero(valid).tolist(),key=lambda j:(-score[j],str(codes[j])))
        selected=ranked[:2]
        ex_i=i+1
        if ex_i>=len(days): raise RuntimeError(f"no T+1 execution day after {date}")
        ex_date=str(days[ex_i])
        schedule[date]={str(codes[j]):0.5 for j in selected}
        signals.append({"signal_date":date,"execution_date":ex_date,"signal_index":int(i),
            "execution_index":int(ex_i),"lookback_intervals":LOOKBACK,
            "volatility_window_start_price_date":str(days[i-LOOKBACK]),
            "volatility_window_end_signal_date":str(days[i]),
            "volatility_return_count":int(len(returns)),"ranking":"descending 60d cumulative adjusted return / sample std of 60 trailing daily adjusted returns",
            "selected_top2":[str(codes[j]) for j in selected],
            "stocks":[{"code":str(codes[j]),"return_60d":float(momentum[j]),
                "volatility_60d_daily_sample_std":float(volatility[j]),"risk_adjusted_momentum":float(score[j]),
                "rank":int(rank+1)} for rank,j in enumerate(ranked)]})
    return schedule,signals


def metric_from_daily(name,daily,trades,metrics):
    last=daily.iloc[-1]
    cash_min=float(daily.cash.min())
    odd_sells=[]
    if len(trades):
        for tr in trades:
            if tr.get("side")=="sell":
                q=float(tr.get("shares",0.0))
                if abs(q/ E.LOT-round(q/E.LOT))>1e-6:
                    odd_sells.append(tr)
    return {"strategy":name,"net_return":float(metrics["net_return"]),
        "gross_return_same_fills":float(metrics["gross_return_same_fills"]),
        "annualized_net_return":float(metrics["annualized_net_return"]),
        "annualized_volatility":float(metrics["annualized_volatility"]),"max_drawdown":float(metrics["max_drawdown"]),
        "fees":float(metrics["fees"]),"slippage_cost":float(metrics["slippage_cost"]),
        "total_trading_cost":float(metrics["total_trading_cost"]),"turnover":float(metrics["turnover"]),
        "trades":int(metrics["trades"]),"blocked_trade_events":int(metrics["blocked_trade_events"]),
        "cash_min":cash_min,"cash_nonnegative":bool(cash_min>=-1e-7),"ending_cash":float(last.cash),
        "ending_positions":int(last.positions),"ending_shares_json":str(last.shares_json),
        "max_abs_accounting_residual":float(metrics["max_abs_accounting_residual"]),
        "odd_lot_zero_target_sell_trade_count":len(odd_sells),
        "odd_lot_exit_trades":[{k:tr.get(k) for k in ("trade_date","stock_code","shares","reason")} for tr in odd_sells]}


def build_period_rows(results):
    qrows=[]; yrows=[]
    for name,daily,_,_ in results:
        for freq,dest in (("Q",qrows),("Y",yrows)):
            for row in E.period_returns(daily,freq):
                dest.append({"strategy":name,**row})
    return qrows,yrows


def main():
    started=time.time()
    manifest=json.loads((ROOT/"cache_manifest.json").read_text(encoding="utf-8"))
    quarters=[x["quarter"] for x in manifest["items"]]
    if len(quarters)!=26 or quarters[0]!="2020Q1" or quarters[-1]!="2026Q2":
        raise RuntimeError(f"expected 26 cached V11 quarters, got {len(quarters)}")
    for item in manifest["items"]:
        if item.get("state")!="complete": raise RuntimeError(f"incomplete V11 score cache: {item['quarter']}")
        f=ROOT/"quarters"/item["quarter"]/"predictions.csv"
        if hashlib.sha256(f.read_bytes()).hexdigest()!=item["sha256"]:
            raise RuntimeError(f"score cache hash mismatch: {item['quarter']}")
    pred=E.load_predictions(quarters)
    if pred.model_quarter.nunique()!=26 or pred.duplicated(["trade_date","stock_code"]).any():
        raise RuntimeError("cached V11 predictions are incomplete or duplicated")
    price=E.load_prices()
    ref_schedules,signal_dates=E.schedules(pred,price)
    oos_dates=sorted(pred.trade_date.unique())
    if oos_dates[0]!=START or oos_dates[-1]!=END or len(signal_dates)!=78:
        raise RuntimeError(f"unexpected shared OOS lifecycle: {oos_dates[0]}..{oos_dates[-1]}, signals={len(signal_dates)}")
    risk_schedule,risk_signals=risk_adjusted_momentum_schedule(price["days"],E.CODES,price,signal_dates)
    if set(risk_schedule)!=set(ref_schedules["60d momentum Top2"]):
        raise RuntimeError("risk-adjusted and comparator signal calendars differ")
    for sig in risk_signals:
        if sig["execution_index"]!=sig["signal_index"]+1 or sig["volatility_window_end_signal_date"]!=sig["signal_date"] or sig["volatility_return_count"]!=60:
            raise RuntimeError(f"causal signal check failed at {sig['signal_date']}")
    schedules={"V23 60d risk-adjusted momentum Top2":risk_schedule,
        "60d raw momentum Top2":ref_schedules["60d momentum Top2"],
        "V11 LambdaRank Top2":ref_schedules["V11 LambdaRank Top2"],
        "four-bank equal-weight hold":ref_schedules["equal-weight hold"]}
    result_rows=[]; metric_map={}; all_trades=[]; daily_frames=[]
    for name,schedule in schedules.items():
        d,tr,m=E.simulate(name,price,schedule,START,END)
        daily_frames.append(d); all_trades.extend(tr)
        audit=metric_from_daily(name,d,tr,m)
        metric_map[name]={**m,"audit":audit}
        result_rows.append((name,d,tr,m))
    candidate=metric_map["V23 60d risk-adjusted momentum Top2"]
    for base_name,key in (("60d raw momentum Top2","vs_raw_60d_momentum"),
        ("V11 LambdaRank Top2","vs_V11_LambdaRank"),("four-bank equal-weight hold","vs_equal_weight_hold")):
        b=metric_map[base_name]
        candidate[key]={"net_return_difference_pp":100*(candidate["net_return"]-b["net_return"]),
            "drawdown_difference_pp":100*(candidate["max_drawdown"]-b["max_drawdown"])}
    qrows,yrows=build_period_rows(result_rows)
    risk_signal_dates=[]
    for s in risk_signals:
        risk_signal_dates.append({"signal_date":s["signal_date"],"execution_date":s["execution_date"],
            "selected_top2":"|".join(s["selected_top2"]),"volatility_window_start_price_date":s["volatility_window_start_price_date"],
            "volatility_window_end_signal_date":s["volatility_window_end_signal_date"],
            "volatility_return_count":s["volatility_return_count"],
            "top2_scores":"|".join(f"{x['code']}:{x['risk_adjusted_momentum']:.10g}" for x in s["stocks"][:2])})
    audit={"window":{"start":START,"end":END,"days":len(oos_dates),"monthly_signal_dates":len(signal_dates)},
        "source_cache_quarters":quarters,"all_26_quarter_cache_hashes_verified":True,
        "model_scores_read_from_cache_only":True,"model_training_or_inference_performed":False,
        "price_adjustment_source":"V11-compatible four-bank adjusted close from shared parquet cache",
        "risk_adjusted_score":"60-session adjusted close cumulative return divided by sample standard deviation of the 60 adjusted-close daily simple returns ending at signal close",
        "causal_checks":{"all_signals_use_60_return_intervals":all(s["volatility_return_count"]==60 for s in risk_signals),
            "all_volatility_end_dates_equal_signal_dates":all(s["volatility_window_end_signal_date"]==s["signal_date"] for s in risk_signals),
            "all_execution_dates_t_plus_1":all(s["execution_index"]==s["signal_index"]+1 for s in risk_signals),
            "signal_dates_match_all_comparators":set(risk_schedule)==set(ref_schedules["60d momentum Top2"])},
        "ledger":"cash_engine.py copied from V11 final analysis engine; same commission, stamp/transfer, slippage, 100-share lots, 1% participation, limit locks, corporate-action share adjustment, and zero-target odd-lot exit rule",
        "same_lifecycle_for_all_scenarios":True,"forced_final_liquidation":False,"scenarios":{}}
    for name,_,_,_ in result_rows:
        audit["scenarios"][name]=metric_map[name]["audit"]
    out={"unit":"experiments2/V23 60d risk-adjusted momentum Top2","built_at":time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_v11_cache_quarters":quarters,"test_dates":[START,END],"signal_count":len(signal_dates),
        "universe":list(E.CODES),"rule":{"signal":"same first trading session of each month as V11, at close",
          "factor":"60-session adjusted-close cumulative return / sample std of 60 trailing adjusted-close daily returns through signal close",
          "ranking":"descending cross-sectional factor score, fixed Top2, equal target weights",
          "execution":"T+1 open","no_training":True,"no_parameter_sweep":True,
          "comparators":["monthly 60d raw momentum Top2","monthly V11 LambdaRank Top2","equal-weight four-bank hold"]},
        "costs":{"commission":E.COMMISSION,"minimum_commission":E.MIN_COMMISSION,"transfer":E.TRANSFER,
          "sell_stamp":E.STAMP,"adverse_slippage":E.SLIPPAGE,"max_participation":E.PARTICIPATION,"lot_size":E.LOT},
        "strategies":metric_map,"quarterly_returns_file":"quarterly_returns.csv","annual_returns_file":"annual_returns.csv",
        "runtime_seconds":time.time()-started,
        "cache_manifest_sha256":hashlib.sha256((ROOT/"cache_manifest.json").read_bytes()).hexdigest(),
        "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cash_engine_sha256":hashlib.sha256((ROOT/"cash_engine.py").read_bytes()).hexdigest()}
    combined=pd.concat(daily_frames,ignore_index=True)
    combined.to_csv(ROOT/"daily_equity.csv",index=False,float_format="%.10g")
    for name,d,tr,m in result_rows:
        stem=name.replace(" ","_").replace("/","_")
        d.to_csv(ROOT/f"{stem}_daily.csv",index=False,float_format="%.10g")
        pd.DataFrame(tr).to_csv(ROOT/f"{stem}_trades.csv",index=False,float_format="%.10g")
    pd.DataFrame(all_trades).to_csv(ROOT/"trades.csv",index=False,float_format="%.10g")
    pd.DataFrame(risk_signals).to_json(ROOT/"risk_adjusted_signals.json",orient="records",force_ascii=False,indent=2)
    pd.DataFrame(risk_signal_dates).to_csv(ROOT/"signal_audit.csv",index=False)
    pd.DataFrame(qrows).to_csv(ROOT/"quarterly_returns.csv",index=False)
    pd.DataFrame(yrows).to_csv(ROOT/"annual_returns.csv",index=False)
    (ROOT/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (ROOT/"metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    protocol={"objective":out["unit"],"official_window":[START,END],"monthly_signals":78,
        "source_predictions":"copied 26 completed V11 quarterly OOS score files; used only for V11 comparator, no training or new inference",
        "signal_rule":out["rule"],"risk_window":"60 daily adjusted-close returns from close(T-60) through close(T), inclusive of T close; sample standard deviation ddof=1",
        "selection_rule":"risk-adjusted momentum = 60d cumulative adjusted return / 60d realized daily volatility; highest two scores; equal target weights",
        "execution_and_accounting":"T+1 open; final V11 account engine; 2020-01-02 through 2026-06-30; no forced terminal liquidation",
        "cost_model":out["costs"],"cache_manifest_sha256":out["cache_manifest_sha256"],
        "runner_sha256":out["runner_sha256"],"cash_engine_sha256":out["cash_engine_sha256"]}
    (ROOT/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    report=["# V23：60 日风险调整动量 Top2","",
        f"- 同一 OOS 生命周期：{START} 至 {END}；78 个月度信号；四银行价格快照复用共享缓存。",
        "- 规则：信号收盘计算复权 60 日累计收益，除以截至信号收盘的 60 个复权日收益样本标准差；按分数降序选 Top2，等权，下一交易日开盘执行。",
        "- V11 26 季分数缓存只用于 Ranker 基线校验；未重训，也未执行模型推理。四条策略使用 V11 同一现金引擎。","",
        "## 总体对比","","| 策略 | 净收益 | 年化 | 最大回撤 | 年化波动 | 费用 | 滑点 | 换手 | 交易数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,_,_,m in result_rows:
        report.append(f"| {name} | {m['net_return']:.2%} | {m['annualized_net_return']:.2%} | {m['max_drawdown']:.2%} | {m['annualized_volatility']:.2%} | ¥{m['fees']:,.0f} | ¥{m['slippage_cost']:,.0f} | ¥{m['turnover']:,.0f} | {m['trades']} |")
    report += ["","## 风险调整动量相对基线",""]
    for base_name,key in (("60 日原始动量 Top2","vs_raw_60d_momentum"),("V11 Ranker Top2","vs_V11_LambdaRank"),("四股等权持有","vs_equal_weight_hold")):
        z=candidate[key]; report.append(f"- 相对{base_name}：收益差 {z['net_return_difference_pp']:+.2f} pp；回撤差 {z['drawdown_difference_pp']:+.2f} pp。")
    report += ["","## 年度收益","","| 年份 | 策略 | 净收益 | 同成交毛收益 |","|---|---|---:|---:|"]
    for z in yrows: report.append(f"| {z['period']} | {z['strategy']} | {z['net_return']:.2%} | {z['gross_return_same_fills']:.2%} |")
    report += ["","## 季度收益","","| 季度 | 策略 | 净收益 | 同成交毛收益 |","|---|---|---:|---:|"]
    for z in qrows: report.append(f"| {z['period']} | {z['strategy']} | {z['net_return']:.2%} | {z['gross_return_same_fills']:.2%} |")
    report += ["","## 核心审计","",
        f"- 月度 60 日波动窗口均为 60 个已实现收益并截止当日信号收盘；T+1 日期检查通过；缓存完整性哈希检查通过。",
        f"- V23 现金最低余额 ¥{candidate['audit']['cash_min']:,.2f}；阻塞交易 {candidate['audit']['blocked_trade_events']}；账本最大残差 {candidate['audit']['max_abs_accounting_residual']:.3g}；零目标 odd-lot 退出 {candidate['audit']['odd_lot_zero_target_sell_trade_count']} 笔。",
        "- 所有策略同一 2020-01-02 至 2026-06-30 生命周期，期末按 V11 口径保留持仓并以收盘价计值，不强制清仓。","",
        "历史回测结果用于策略比较，不能证明未来收益稳定或存在可持续 alpha。","",
        f"运行脚本 SHA-256：{out['runner_sha256']}",f"现金引擎 SHA-256：{out['cash_engine_sha256']}",
        f"缓存清单 SHA-256：{out['cache_manifest_sha256']}"]
    (ROOT/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({name:{"net":m["net_return"],"mdd":m["max_drawdown"],"fees":m["fees"],"slippage":m["slippage_cost"],"turnover":m["turnover"]}
        for name,_,_,m in result_rows},ensure_ascii=False,indent=2),flush=True)
    print("V23_FILES",ROOT,"runtime",out["runtime_seconds"],flush=True)

if __name__=="__main__": main()
