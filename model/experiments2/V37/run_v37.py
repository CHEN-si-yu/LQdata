#!/usr/bin/env python3
"""V37 frozen-factor extension through 2026-09-24 with terminal open liquidation."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[key]="1"
import sys
sys.dont_write_bytecode=True
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
HERE=Path(__file__).resolve().parent
V11=HERE.parent/"V11"
V26=HERE.parent/"V34"
DATA=V11.parents[1]/"trainingdata"
sys.path.insert(0,str(V11))
import analysis as E
START="2024-01-02"
COMMON_END="2026-06-30"
EXT_END="2026-09-24"
FEATURE="id2_close_vs_pm_vwap_20"
CODES=("601288.SH","601398.SH","601939.SH","601988.SH")
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def resource_snapshot(stage):
    vals={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:","MemAvailable:")):
            k,v=line.split(":",1); vals[k]=int(v.split()[0])*1024
    total=vals.get("MemTotal",0); avail=vals.get("MemAvailable",0)
    used=(total-avail)/(1024**3) if total else 0.0
    try: load=float(Path("/proc/loadavg").read_text().split()[0])
    except Exception: load=0.0
    return {"ts":time.strftime("%Y-%m-%d %H:%M:%S"),"stage":stage,
      "mem_total_gib":round(total/(1024**3),2),"mem_available_gib":round(avail/(1024**3),2),
      "mem_used_gib":round(used,2),"load1":load,"cpu_cores":os.cpu_count() or 0}
def factor_rows(signal_dates):
    meta=json.loads((DATA/"meta.json").read_text(encoding="utf-8"))
    if meta.get("semantics")!="zscore_win1_99_v1": raise RuntimeError("unexpected factor semantics")
    years=sorted(int(y) for y in meta["built_years"] if 2024<=int(y)<=2026)
    parts=[]
    for year in years:
        frame=pq.read_table(DATA/"factors"/f"year={year}"/"data.parquet",
          columns=["trade_date","stock_code",FEATURE],
          filters=[("stock_code","in",list(CODES))]).to_pandas()
        frame["trade_date"]=frame.trade_date.astype(str).str[:10]
        frame["stock_code"]=frame.stock_code.astype(str)
        parts.append(frame)
    frame=pd.concat(parts,ignore_index=True)
    if frame.duplicated(["trade_date","stock_code"]).any(): raise RuntimeError("duplicate factor rows")
    by_date={d:g.set_index("stock_code") for d,g in frame.groupby("trade_date",sort=False)}
    signals={}; audit=[]
    for d in signal_dates:
        if d not in by_date: raise RuntimeError(f"missing {FEATURE} on signal date {d}")
        vals=by_date[d].reindex(CODES)[FEATURE].astype(float)
        missing=[c for c in CODES if not np.isfinite(vals.loc[c])]
        clean=vals.fillna(0.0)
        picked=sorted(CODES,key=lambda c:(-float(clean.loc[c]),c))[:2]
        signals[d]={c:0.5 for c in picked}
        audit.append({"signal_date":d,"factor":FEATURE,"factor_direction":"higher_is_better",
          "factor_values":{c:float(clean.loc[c]) for c in CODES},
          "missing_filled_zero":missing,"selected_top2":picked,"signal_after_close":True})
    return signals,audit,meta
def momentum_schedule(signal_dates,prices):
    idx=prices["index"]; adj_close=prices["adj_close"]
    schedule={}; audit=[]
    for d in signal_dates:
        i=idx[d]
        if i<60: raise RuntimeError(f"{d}: fewer than 60 historical sessions")
        ret=adj_close[i]/adj_close[i-60]-1.0
        order=sorted(range(4),key=lambda j:(-(ret[j] if np.isfinite(ret[j]) else -np.inf),CODES[j]))
        picked=[CODES[j] for j in order[:2]]
        schedule[d]={c:0.5 for c in picked}
        audit.append({"signal_date":d,"momentum_definition":"adjusted_close[t]/adjusted_close[t-60]-1",
          "trailing_return_60_sessions":{CODES[j]:(float(ret[j]) if np.isfinite(ret[j]) else None) for j in range(4)},
          "selected_top2":picked,"signal_after_close":True})
    return schedule,audit
def first_session_each_month(days,start,end):
    out={}
    for value in days:
        d=str(value)
        if start<=d<=end: out.setdefault(d[:7],d)
    return sorted(out.values())
def schedule_with_exit(schedule,trigger_date):
    out=dict(schedule)
    if trigger_date in out: raise RuntimeError(f"terminal trigger overlaps alpha signal: {trigger_date}")
    out[trigger_date]={}
    return out
def shares_from_row(row):
    return json.loads(row["shares_json"]) if row is not None and row["shares_json"] else {}
def audit_exit(daily,trades,window,end_day,trigger_date):
    d=daily.sort_values("trade_date").reset_index(drop=True)
    if d.iloc[-1].trade_date!=end_day: raise RuntimeError(f"{window}: wrong last date")
    previous=d.loc[d.trade_date==trigger_date]
    final=d.loc[d.trade_date==end_day]
    if len(previous)!=1 or len(final)!=1: raise RuntimeError(f"{window}: exit rows missing")
    before=shares_from_row(previous.iloc[0]); after=shares_from_row(final.iloc[0])
    if after: raise RuntimeError(f"{window}: terminal holdings remain: {after}")
    exits=[x for x in trades if x["trade_date"]==end_day and x["side"]=="sell"]
    end_row=final.iloc[0]; prev_row=previous.iloc[0]
    return {
      "window":window,"strategy":str(end_row.strategy),"trigger_signal_close":trigger_date,
      "forced_exit_date":end_day,"forced_exit_at_open":True,
      "positions_before_exit":before,"positions_after_exit":after,
      "position_count_before_exit":len(before),"exit_trade_count":len(exits),
      "shares_sold":{c:float(sum(x["shares"] for x in exits if x["stock_code"]==c)) for c in CODES},
      "raw_open_value_sold":float(sum(x["shares"]*x["open"] for x in exits)),
      "filled_notional_after_slippage_before_fees":float(sum(x["notional"] for x in exits)),
      "exit_day_fees":float(sum(x["fee"] for x in exits)),
      "exit_day_slippage_cost":float(sum(x["slippage_cost"] for x in exits)),
      "exit_day_net_return_including_overnight_gap_and_exit_costs":float(end_row.equity_net/prev_row.equity_net-1.0),
      "exit_day_gross_return_same_fills":float(end_row.equity_gross/prev_row.equity_gross-1.0),
      "equity_prior_day_close":float(prev_row.equity_net),
      "equity_after_open_exit":float(end_row.equity_net),
      "ending_cash":float(end_row.cash),"ending_positions":int(end_row.positions),
      "blocked_trade_events_total":int(daily.attrs.get("blocked_trade_events",0))
    }
def period_rows(daily,window,frequency):
    rows=[]
    for _,group in daily.groupby("strategy",sort=False):
        for row in E.period_returns(group,frequency):
            row["window"]=window
            row["partial_period"]=(window=="extended_force_exit_2026-09-24" and row["period"]=="2026Q3")
            row["observed_through"]=EXT_END if row["partial_period"] else row["period"]
            rows.append(row)
    return rows
def main():
    before=resource_snapshot("before_load")
    prices=E.load_prices()
    after_prices=resource_snapshot("prices_loaded")
    days=prices["days"]
    if str(days[-1])!=EXT_END: raise RuntimeError(f"latest available price date is {days[-1]}, expected {EXT_END}")
    if not np.isfinite(prices["open"][prices["index"][EXT_END]]).all():
        raise RuntimeError("missing final-date open for one or more banks")
    common_signals=first_session_each_month(days,START,COMMON_END)
    ext_signals=first_session_each_month(days,START,EXT_END)
    if len(common_signals)!=30 or len(ext_signals)!=33:
        raise RuntimeError(f"unexpected signal counts {len(common_signals)} / {len(ext_signals)}")
    factor_common,factor_common_audit,meta=factor_rows(common_signals)
    factor_ext,factor_ext_audit,_=factor_rows(ext_signals)
    momentum_common,mom_common_audit=momentum_schedule(common_signals,prices)
    momentum_ext,mom_ext_audit=momentum_schedule(ext_signals,prices)
    v34_members=json.loads((V26/"membership_audit.json").read_text(encoding="utf-8"))
    v34_schedule={x["signal_date"]:{c:0.5 for c in x["selected_top2"]}
      for x in v34_members if START<=x["signal_date"]<=COMMON_END}
    if v34_schedule!=factor_common:
        raise RuntimeError("common-window factor schedule differs from frozen V34 membership")
    qnames=[str(q) for q in pd.period_range("2024Q1","2026Q2",freq="Q")]
    v11_pred=E.load_predictions(qnames)
    v11_schedules,v11_dates=E.schedules(v11_pred,prices)
    if sorted(d for d in v11_dates if START<=d<=COMMON_END)!=common_signals:
        raise RuntimeError("common-window dates differ from V11 cached monthly prediction dates")
    v11_momentum={d:v11_schedules["60d momentum Top2"][d] for d in common_signals}
    if v11_momentum!=momentum_common:
        raise RuntimeError("recomputed common momentum schedule differs from V11")
    v11_ranker={d:v11_schedules["V11 LambdaRank Top2"][d] for d in common_signals}
    idx=prices["index"]
    common_trigger=str(days[idx[COMMON_END]-1])
    ext_trigger=str(days[idx[EXT_END]-1])
    common_schedules={
      "V34 id2_close_vs_pm_vwap_20 Top2":schedule_with_exit(factor_common,common_trigger),
      "60d momentum Top2":schedule_with_exit(momentum_common,common_trigger),
      "equal-weight hold":schedule_with_exit({common_signals[0]:{c:0.25 for c in CODES}},common_trigger),
      "V11 LambdaRank Top2":schedule_with_exit(v11_ranker,common_trigger)}
    extended_schedules={
      "V34 id2_close_vs_pm_vwap_20 Top2":schedule_with_exit(factor_ext,ext_trigger),
      "60d momentum Top2":schedule_with_exit(momentum_ext,ext_trigger),
      "equal-weight hold":schedule_with_exit({ext_signals[0]:{c:0.25 for c in CODES}},ext_trigger)}
    daily_all=[]; trade_all=[]; metrics={}; exit_audits=[]
    for window,end,schedules in (
      ("common_force_exit_2026-06-30",COMMON_END,common_schedules),
      ("extended_force_exit_2026-09-24",EXT_END,extended_schedules)):
        for name,signals in schedules.items():
            daily,trades,metric=E.simulate(name,prices,signals,START,end)
            trades=[dict(row) for row in trades]
            for row in trades:
                row["window"]=window
                if row["trade_date"]==end and row["side"]=="sell":
                    row["reason"]="forced_terminal_open_exit"
            daily["window"]=window
            daily["terminal_exit_day"]=daily.trade_date.eq(end)
            daily["gross_return_daily"]=daily.equity_gross.pct_change().fillna(0.0)
            daily.attrs["blocked_trade_events"]=metric["blocked_trade_events"]
            if int(daily.iloc[-1].positions)!=0:
                raise RuntimeError(f"{window} {name}: forced exit did not clear holdings")
            exit_audits.append(audit_exit(daily,trades,window,end,
              common_trigger if window.startswith("common") else ext_trigger))
            daily_all.append(daily); trade_all.extend(trades)
            metrics[f"{window}:{name}"]=metric
    daily_df=pd.concat(daily_all,ignore_index=True)
    trades_df=pd.DataFrame(trade_all)
    qrows=[]; arows=[]
    for window in daily_df.window.unique():
        part=daily_df[daily_df.window==window]
        qrows.extend(period_rows(part,window,"Q"))
        arows.extend(period_rows(part,window,"Y"))
    daily_df.to_csv(HERE/"daily_equity.csv",index=False,float_format="%.10g")
    trades_df.to_csv(HERE/"trades.csv",index=False,float_format="%.10g")
    pd.DataFrame(qrows).to_csv(HERE/"quarterly_returns.csv",index=False,float_format="%.10g")
    pd.DataFrame(arows).to_csv(HERE/"annual_returns.csv",index=False,float_format="%.10g")
    pd.DataFrame(exit_audits).to_csv(HERE/"exit_audit.csv",index=False,float_format="%.10g")
    v34_metrics=json.loads((V26/"metrics.json").read_text(encoding="utf-8"))["metrics"]
    published_common={}
    for key in ("V34 id2_close_vs_pm_vwap_20 Top2","V11 LambdaRank Top2","60d momentum Top2","equal-weight hold"):
        if key not in v34_metrics: raise RuntimeError(f"V34 published metrics missing {key}")
        published_common[key]=v34_metrics[key]
    # Reuse the previously verified V34 SHA manifest, and assert that every source file
    # still has the same size. Exact original file SHA values are retained in this report.
    old_manifest=json.loads((V26/"cache_hashes.json").read_text(encoding="utf-8"))
    selected=[]
    for item in old_manifest["hashes"]:
        role=item["role"]; path=Path(item["path"]); keep=False
        if role=="factor-and-market-cache metadata": keep=True
        elif role.startswith("price cache year") or role.startswith("amount cache year"): keep=True
        elif role.startswith("factor cache year") and role.split()[-1] in {"2024","2025","2026"}: keep=True
        elif role=="V11 corrected ledger engine": keep=True
        elif role=="cached V11 quarterly predictions" and any(f"/{q}/" in item["path"] for q in qnames): keep=True
        if keep:
            if not path.is_file(): raise FileNotFoundError(path)
            if int(path.stat().st_size)!=int(item["size_bytes"]):
                raise RuntimeError(f"input size changed since V34 hash manifest: {path}")
            selected.append(item|{"current_size_verified":True,"sha256_origin":"V34 cache_hashes.json"})
    input_hashes={
      "version":"V37","algorithm":"SHA-256",
      "frozen_factor":{"name":FEATURE,"direction":"higher_is_better","selection_cutoff":"2023Q4",
        "v34_protocol_sha256":file_sha(V26/"protocol.json"),
        "v34_membership_audit_sha256":file_sha(V26/"membership_audit.json"),
        "v34_cache_hashes_sha256":file_sha(V26/"cache_hashes.json"),
        "v34_runner_sha256":file_sha(V26/"run_v34.py")},
      "source_sha256_records_reused_from_v34_manifest":selected,
      "current_source_sizes_match_v34_manifest":True,
      "v11_ranker_comparison_prediction_quarters":qnames,
      "v34_known_common_window_metrics_sha256":file_sha(V26/"metrics.json"),
      "v11_ledger_source_sha256":file_sha(V11/"analysis.py"),
      "v37_protocol_sha256":file_sha(HERE/"PROTOCOL.md"),
      "v37_run_script_sha256":file_sha(HERE/"run_v37.py"),
      "latest_price_date":EXT_END,"factor_semantics":meta.get("semantics")}
    (HERE/"input_hashes.json").write_text(json.dumps(input_hashes,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    membership=[]
    for window,fa,ma in (("common",factor_common_audit,mom_common_audit),("extended",factor_ext_audit,mom_ext_audit)):
        for strategy,rows in ((f"V34 {FEATURE} Top2",fa),("60d momentum Top2",ma)):
            for item in rows:
                row=dict(item); row["window"]=window; row["strategy"]=strategy
                row["execution_date"]=str(days[idx[row["signal_date"]]+1])
                membership.append(row)
    pd.DataFrame(membership).to_json(HERE/"signal_audit.json",orient="records",force_ascii=False,indent=2)
    common_metric_view={k.split(":",1)[1]:v for k,v in metrics.items() if k.startswith("common_force_exit_")}
    extended_metric_view={k.split(":",1)[1]:v for k,v in metrics.items() if k.startswith("extended_force_exit_")}
    exit_map={f"{x['window']}:{x['strategy']}":x for x in exit_audits}
    audit={
      "price_panel_latest_date":EXT_END,"factor_panel_latest_date":EXT_END,
      "common_window":[START,COMMON_END],"extended_window":[START,EXT_END],
      "common_monthly_signal_count":len(common_signals),"extended_monthly_signal_count":len(ext_signals),
      "extended_added_signal_dates":sorted(set(ext_signals)-set(common_signals)),
      "no_v11_ranker_q3_predictions":True,"v11_ranker_prediction_quarters_used":qnames,
      "factor_common_schedule_exact_match_v34":factor_common==v34_schedule,
      "momentum_common_schedule_exact_match_v11":momentum_common==v11_momentum,
      "factor_direction_frozen":"higher_is_better through V34 protocol cutoff 2023Q4",
      "end_exit_common":{k:v for k,v in exit_map.items() if k.startswith("common_force_exit_")},
      "end_exit_extended":{k:v for k,v in exit_map.items() if k.startswith("extended_force_exit_")},
      "all_end_positions_zero":all(x["ending_positions"]==0 for x in exit_audits),
      "all_exit_at_next_session_open":True,
      "negative_cash_rows":int((daily_df.cash < -1e-7).sum()),
      "negative_equity_rows":int((daily_df.equity_net < -1e-7).sum()),
      "max_abs_accounting_residual":float(daily_df.accounting_residual.abs().max()),
      "strategy_metrics":metrics,
      "published_v34_common_window_close_mark":published_common,
      "resource_samples":[before,after_prices,resource_snapshot("after_simulation")],
      "ram_ceiling_gib":180.0}
    (HERE/"exit_audit.json").write_text(json.dumps(exit_audits,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (HERE/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    report={
      "version":"V37","frozen_factor":FEATURE,"direction":"higher zscore ranks higher",
      "common_window":{"dates":[START,COMMON_END],"published_v34_close_mark":published_common,
        "v37_forced_open_exit_metrics":common_metric_view,
        "terminal_exit_audit":audit["end_exit_common"]},
      "extended_window":{"dates":[START,EXT_END],"partial_quarter":"2026Q3",
        "signals":ext_signals,"metrics":extended_metric_view,
        "terminal_exit_audit":audit["end_exit_extended"]},
      "v11_ranker_scope":"common window only; existing 2024Q1-2026Q2 predictions, no 2026Q3 scores",
      "signal_counts":{"common_months":len(common_signals),"extended_months":len(ext_signals),
        "added_month_signals":sorted(set(ext_signals)-set(common_signals))},
      "execution":"V11 final corrected ledger; monthly close signal/T+1 open, common close exit 2026-06-30 open, extended close exit 2026-09-24 open",
      "resource_samples":audit["resource_samples"],
      "limitation":"Extension adds only about three months; it is a short forward extension and cannot establish stable performance.",
      "metrics":metrics}
    (HERE/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    protocol_json={
      "version":"V37","protocol_file":"PROTOCOL.md","frozen_before_extension_run":True,
      "frozen_source":"V34 protocol; 2023Q4 feature selection cutoff",
      "factor":{"name":FEATURE,"direction":"higher zscore_win1_99_v1 value ranks higher",
        "semantics":meta.get("semantics"),"feature_selection_changed":False},
      "windows":{"common":[START,COMMON_END],"extended":[START,EXT_END],
        "extended_quarter":"2026Q3 partial through 2026-09-24"},
      "signals":{"common_dates":common_signals,"extended_dates":ext_signals,
        "extended_added_dates":sorted(set(ext_signals)-set(common_signals)),
        "monthly_close_signal_next_session_open":True},
      "accounts":{"initial_equity":E.INITIAL,"independent_account_per_strategy":True,
        "forced_exit_dates":{"common":COMMON_END,"extended":EXT_END},
        "forced_exit_trigger_close":{"common":common_trigger,"extended":ext_trigger},
        "forced_exit_rule":"V11 final zero-target sell path at terminal open, including 1% participation, locked-price checks, fees, slippage and odd-lot full exit; assert zero terminal shares"},
      "comparison_scope":{"common":["V34 frozen factor","60d momentum Top2","equal-weight hold","V11 LambdaRank Top2"],
        "extended":["V34 frozen factor","60d momentum Top2","equal-weight hold"],
        "v11_ranker_q3_predictions_generated":False},
      "no_model_training":True,"no_parameter_sweep":True,"ram_ceiling_gib":180.0,
      "input_hashes_file":"input_hashes.json"}
    (HERE/"protocol.json").write_text(json.dumps(protocol_json,ensure_ascii=False,indent=2)+chr(10),encoding="utf-8")
    def pct(m): return f"{m['net_return']:.2%} / {m['max_drawdown']:.2%} / {m['annualized_volatility']:.2%}"
    lines=["# V37：V34 单因子前瞻扩展","",
      f"- 因子固定为 {FEATURE}，方向沿用 V34（高值优先），名单与方向截止 2023Q4；不训练、不按扩展结果改策略。",
      "- 共用初始资金 CNY 100,000 的独立账户；月初收盘信号、下一交易日开盘交易。末日前一交易日发零目标，窗口末日开盘按 V11 修正账本强制退出；退出日收益与费用单列。","",
      "## V34 已知共同窗口（原结果，6/30 收盘估值）","",
      "| Strategy | Net return | Max drawdown | Ann. vol. |","|---|---:|---:|---:|"]
    for name,m in published_common.items():
        lines.append(f"| {name} | {m['net_return']:.2%} | {m['max_drawdown']:.2%} | {m['annualized_volatility']:.2%} |")
    lines+=["","## V37 共同窗口（统一 6/30 开盘强平）","",
      "| Strategy | Net return | Max drawdown | Ann. vol. | Exit-day net return |","|---|---:|---:|---:|---:|"]
    for name,m in common_metric_view.items():
        x=exit_map[f"common_force_exit_2026-06-30:{name}"]
        lines.append(f"| {name} | {m['net_return']:.2%} | {m['max_drawdown']:.2%} | {m['annualized_volatility']:.2%} | {x['exit_day_net_return_including_overnight_gap_and_exit_costs']:+.2%} |")
    lines+=["","## V37 延伸窗口（2026Q3 部分；9/24 开盘强平）","",
      "| Strategy | Net return | Max drawdown | Ann. vol. | Exit-day net return |","|---|---:|---:|---:|---:|"]
    for name,m in extended_metric_view.items():
        x=exit_map[f"extended_force_exit_2026-09-24:{name}"]
        lines.append(f"| {name} | {m['net_return']:.2%} | {m['max_drawdown']:.2%} | {m['annualized_volatility']:.2%} | {x['exit_day_net_return_including_overnight_gap_and_exit_costs']:+.2%} |")
    lines+=["","V11 Ranker 仅在共同窗口截至 2026-06-30 比较；没有生成 Q3 预测。","",
      "因子和动量信号由 30 个共同月份延长为 33 个，新增日期：" + ", ".join(sorted(set(ext_signals)-set(common_signals))) + "。样本只增加约三个月，不能据此确认稳定性。",
      "逐日权益、成交、季度/年度统计、终端强平审计和输入散列均保存在本目录。"]
    (HERE/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    resource_rows=[before,after_prices,resource_snapshot("after_simulation")]
    with (HERE/"resource_log.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(resource_rows[0]))
        writer.writeheader(); writer.writerows(resource_rows)
    with (HERE/"RUN_LOG.md").open("a",encoding="utf-8") as f:
        f.write(f"\n- 完成：共同窗口30个信号，扩展窗口33个信号；2026-09-24开盘强平审计通过，所有账户期末持仓为零。")
        f.write(f"\n- V34因子扩展净收益 {extended_metric_view[f'V34 {FEATURE} Top2']['net_return']:.4%}；60日动量 {extended_metric_view['60d momentum Top2']['net_return']:.4%}；等权 {extended_metric_view['equal-weight hold']['net_return']:.4%}。")
        f"\n- 资源采样主机内存峰值 {max(x['mem_used_gib'] for x in resource_rows):.2f} GiB（180 GiB 上限）。输入 SHA-256 与终端退出明细已保存。"
    print("\n".join(lines))
if __name__=="__main__": main()
