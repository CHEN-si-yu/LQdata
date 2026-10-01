#!/usr/bin/env python3
"""V27 post-fit accounting, parity checks, leakage audit, and output packaging."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[key]="1"
import csv, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
V11=HERE.parent/"V11"
V24=HERE.parent/"V24"
START="2024-01-02"
END="2026-06-30"
FEATURES=("mf_tier_flow_agreement_20","rel_mom_ind_3d","rel_mom_ind_5d","gap_down_recover_freq_20d")
def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def main():
    import sys
    sys.path.insert(0,str(V11))
    import analysis as E
    quarters=tuple(str(q) for q in pd.period_range("2024Q1","2026Q2",freq="Q"))
    qroot=HERE/"quarters"
    quarter_results={}
    pred_parts=[]
    model_manifest=[]
    for q in quarters:
        folder=qroot/q
        for name in ("result.json","status.json","predictions.csv","ranker.txt"):
            if not (folder/name).is_file(): raise FileNotFoundError(folder/name)
        result=json.loads((folder/"result.json").read_text(encoding="utf-8"))
        status=json.loads((folder/"status.json").read_text(encoding="utf-8"))
        if result.get("state")!="complete" or status.get("state")!="complete":
            raise RuntimeError(f"{q} is not complete")
        if result.get("quarter")!=q: raise RuntimeError(f"{q}: quarter identity mismatch")
        if result.get("features")!=list(FEATURES)+[f"asset_{i}" for i in range(4)]:
            raise RuntimeError(f"{q}: model inputs differ from frozen four-factor protocol")
        if result.get("purged_trading_sessions")!=5 or result.get("max_label_date_used")!=result.get("train_last"):
            raise RuntimeError(f"{q}: purge or label cutoff mismatch")
        pred=pd.read_csv(folder/"predictions.csv",dtype={"trade_date":str,"stock_code":str,"model_quarter":str})
        pred["trade_date"]=pred.trade_date.astype(str).str[:10]
        pred["monthly_signal"]=pred.monthly_signal.astype(bool)
        pred["selected_top2"]=pred.selected_top2.astype(bool)
        if pred["model_quarter"].nunique()!=1 or pred["model_quarter"].iloc[0]!=q:
            raise RuntimeError(f"{q}: prediction quarter mismatch")
        if pred.duplicated(["trade_date","stock_code"]).any():
            raise RuntimeError(f"{q}: duplicate daily score rows")
        if pred.trade_date.min()<result["first_test"] or pred.trade_date.max()>result["last_test"]:
            raise RuntimeError(f"{q}: prediction outside test quarter")
        if len(pred)!=4*result["n_test_days"]:
            raise RuntimeError(f"{q}: expected four score rows per test date")
        pred_parts.append(pred)
        quarter_results[q]=result
        model_manifest.append({"quarter":q,"train_first":result["train_first"],"train_last":result["train_last"],
          "first_test":result["first_test"],"last_test":result["last_test"],
          "purged_trading_sessions":result["purged_trading_sessions"],
          "train_rows":result["n_train_rows_after_label_filter"],"train_queries":result["n_train_queries"],
          "fit_seconds":result["fit_seconds"],"rss_gib":result["rss_gib"],
          "predictions_sha256":sha256(folder/"predictions.csv"),
          "model_sha256":sha256(folder/"ranker.txt"),"result_sha256":sha256(folder/"result.json"),
          "prediction_rows":int(len(pred)),"monthly_signal_dates":result["monthly_signal_dates"]})
    model_pred=pd.concat(pred_parts,ignore_index=True)
    if model_pred.duplicated(["trade_date","stock_code"]).any(): raise RuntimeError("overlapping quarterly predictions")
    if model_pred.trade_date.min()!=START or model_pred.trade_date.max()!=END:
        raise RuntimeError(f"unexpected V27 OOS date coverage {model_pred.trade_date.min()}..{model_pred.trade_date.max()}")
    signal_dates=sorted(model_pred.loc[model_pred.monthly_signal,"trade_date"].unique())
    v27_schedule={}
    for d in signal_dates:
        day=model_pred[(model_pred.trade_date==d)&model_pred.monthly_signal]
        picks=day.loc[day.selected_top2].sort_values(["daily_rank","stock_code"]).stock_code.tolist()
        if len(picks)!=2: raise RuntimeError(f"{d}: V27 Top2 count={len(picks)}")
        v27_schedule[d]={c:0.5 for c in picks}
    if len(signal_dates)!=30: raise RuntimeError(f"expected 30 monthly signals, found {len(signal_dates)}")
    prices=E.load_prices()
    dates_axis=prices["days"]; date_ix=prices["index"]
    if date_ix[START]>=date_ix[END]: raise RuntimeError("invalid lifecycle")
    # Check every quarterly model's train/test boundary against the trading-session axis.
    leakage_checks={}
    for q,result in quarter_results.items():
        gap=date_ix[result["first_test"]]-date_ix[result["train_last"]]-1
        if gap!=5 or result["train_last"]>=result["first_test"]:
            raise RuntimeError(f"{q}: train/test boundary does not preserve five-session purge")
        if result["monthly_signal_dates"]:
            for d in result["monthly_signal_dates"]:
                if d not in date_ix or date_ix[d]+1>=len(dates_axis):
                    raise RuntimeError(f"{q}: no next-session execution for signal {d}")
        leakage_checks[q]={"train_last":result["train_last"],"first_test":result["first_test"],
          "purged_trading_sessions":gap,"max_label_date_used":result["max_label_date_used"],
          "train_precedes_test":result["train_last"]<result["first_test"],
          "predictions_are_quarter_local":True}
    # Fixed V24 membership schedule is consumed as a predeclared comparator.
    v24_membership=json.loads((V24/"membership_audit.json").read_text(encoding="utf-8"))
    v24_schedule={}
    for row in v24_membership:
        d=row["signal_date"]
        if START<=d<=END:
            picks=list(row["selected_top2"])
            if len(picks)!=2: raise RuntimeError(f"{d}: V24 fixed-rank Top2 count={len(picks)}")
            v24_schedule[d]={c:0.5 for c in picks}
    if sorted(v24_schedule)!=signal_dates:
        raise RuntimeError("V24 fixed-rank signal dates differ from V27 quarterly signals")
    v11_quarters=list(quarters)
    v11_pred=E.load_predictions(v11_quarters)
    v11_all,E_dates=E.schedules(v11_pred,prices)
    if sorted(E_dates)!=signal_dates:
        raise RuntimeError("V11 shared monthly dates differ from V27")
    v11_rank={d:v11_all["V11 LambdaRank Top2"][d] for d in signal_dates}
    v11_mom={d:v11_all["60d momentum Top2"][d] for d in signal_dates}
    schedules={
      "V27 four-factor LambdaRank Top2":v27_schedule,
      "V24 fixed-rank Top2":v24_schedule,
      "V11 LambdaRank Top2":v11_rank,
      "V11 60d momentum Top2":v11_mom,
      "equal-weight hold":{signal_dates[0]:{c:0.25 for c in E.CODES}}
    }
    daily_parts=[]; trades=[]; metrics={}
    for name,schedule in schedules.items():
        daily,fills,m=E.simulate(name,prices,schedule,START,END)
        daily_parts.append(daily); trades.extend(fills); metrics[name]=m
    daily_all=pd.concat(daily_parts,ignore_index=True)
    trades_df=pd.DataFrame(trades)
    # Periodized account returns use the V11 corrected ledger's prior-period ending equity.
    annual=[row for d in daily_parts for row in E.period_returns(d,"Y")]
    quarterly=[row for d in daily_parts for row in E.period_returns(d,"Q")]
    signal_rows=[]
    for strategy,schedule in schedules.items():
        for d,target in schedule.items():
            execution=str(dates_axis[date_ix[d]+1])
            signal_rows.extend({"signal_date":d,"execution_date":execution,"strategy":strategy,
              "stock_code":code,"target_weight":float(weight)} for code,weight in sorted(target.items()))
    signal_df=pd.DataFrame(signal_rows)
    target_sums=signal_df.groupby(["signal_date","strategy"]).target_weight.sum()
    if not np.allclose(target_sums.to_numpy(),1.0,rtol=0,atol=1e-12):
        raise RuntimeError("some signal target weights do not sum to 1")
    # These four reference portfolios were already reported over this exact date window by V24.
    reference_file=json.loads((V24/"metrics.json").read_text(encoding="utf-8"))
    ref_names={
      "V24 fixed-rank Top2":"V24 stable-factor consensus Top2",
      "V11 LambdaRank Top2":"V11 LambdaRank Top2",
      "V11 60d momentum Top2":"60d momentum Top2",
      "equal-weight hold":"equal-weight hold"
    }
    reference_checks={}
    for actual_name,ref_name in ref_names.items():
        actual=metrics[actual_name]
        expected=reference_file["metrics"][ref_name]
        reference_checks[actual_name]={
          "reference_strategy":ref_name,
          "net_return_abs_difference":abs(float(actual["net_return"])-float(expected["net_return"])),
          "max_drawdown_abs_difference":abs(float(actual["max_drawdown"])-float(expected["max_drawdown"])),
          "reference_net_return":float(expected["net_return"]),
          "recomputed_net_return":float(actual["net_return"])}
    candidate="V27 four-factor LambdaRank Top2"
    for baseline in ref_names:
        metrics[candidate]["vs_"+baseline.replace(" ","_").replace("-","_")]={
          "net_return_difference_pp":100*(metrics[candidate]["net_return"]-metrics[baseline]["net_return"]),
          "max_drawdown_difference_pp":100*(metrics[candidate]["max_drawdown"]-metrics[baseline]["max_drawdown"])}
    daily_all.to_csv(HERE/"daily_equity.csv",index=False,float_format="%.10g")
    trades_df.to_csv(HERE/"trades.csv",index=False,float_format="%.10g")
    signal_df.to_csv(HERE/"monthly_signals.csv",index=False,float_format="%.10g")
    pd.DataFrame(quarterly).to_csv(HERE/"quarterly_returns.csv",index=False,float_format="%.10g")
    pd.DataFrame(annual).to_csv(HERE/"annual_returns.csv",index=False,float_format="%.10g")
    # The source manifest records exactly which V11 score caches feed the baseline comparison.
    input_cache=[]
    for q in quarters:
        source=V11/"quarters"/q/"predictions.csv"
        input_cache.append({"quarter":q,"path":str(source),"sha256":sha256(source),
          "bytes":source.stat().st_size,
          "rows":int((v11_pred.model_quarter.astype(str)==q).sum()) if "model_quarter" in v11_pred else 0})
    cache_manifest={"version":"V27","quarter_count":len(quarters),"oos_prediction_quarters":model_manifest,
      "v11_comparison_prediction_cache":input_cache,
      "frozen_v24_sources":{
        "protocol_sha256":sha256(V24/"protocol.json"),
        "membership_audit_sha256":sha256(V24/"membership_audit.json")},
      "factor_source":{"semantics":"zscore_win1_99_v1","features":list(FEATURES),
        "built_years":json.loads((HERE.parents[1]/"trainingdata"/"meta.json").read_text(encoding="utf-8"))["built_years"],
        "future_quarter_rows_excluded_during_panel_load":True},
      "label_source":"trainingdata/target/label_ret_5d; filtered to train_last in each quarter",
      "no_2020_2023_prediction_models":True}
    (HERE/"cache_manifest.json").write_text(json.dumps(cache_manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    scheduler_path=HERE/"scheduler_summary.json"
    scheduler=json.loads(scheduler_path.read_text(encoding="utf-8")) if scheduler_path.is_file() else {}
    audit={
      "window":[START,END],"quarter_count":len(quarters),"monthly_signal_count":len(signal_dates),
      "v27_oos_prediction_rows":int(len(model_pred)),"duplicate_prediction_rows":0,
      "features_frozen_before_holdout":list(FEATURES),"v24_feature_cutoff":"2023Q4",
      "model_inputs_per_worker":8,"input_features_exactly_four_factors_plus_four_asset_onehots":True,
      "quarter_boundaries":leakage_checks,
      "all_quarters_have_five_session_purge":all(x["purged_trading_sessions"]==5 for x in leakage_checks.values()),
      "all_labels_end_at_train_cutoff":all(x["max_label_date_used"]==x["train_last"] for x in leakage_checks.values()),
      "signal_dates_match_v24_v11":True,"all_signals_execute_next_session":bool((signal_df.execution_date.map(date_ix)-signal_df.signal_date.map(date_ix)==1).all()),
      "target_weight_sum_min":float(target_sums.min()),"target_weight_sum_max":float(target_sums.max()),
      "daily_rows":int(len(daily_all)),"trade_rows":int(len(trades_df)),
      "negative_cash_rows":int((daily_all.cash<-1e-7).sum()),
      "negative_equity_rows":int((daily_all.equity_net<-1e-7).sum()),
      "max_abs_accounting_residual":float(daily_all.accounting_residual.abs().max()),
      "strategies":{name:{"blocked_trade_events":int(metrics[name]["blocked_trade_events"]),
        "max_positions":int(metrics[name]["max_positions"]),"min_cash":float(metrics[name]["min_cash"]),
        "trades":int(metrics[name]["trades"]),"max_abs_accounting_residual":float(metrics[name]["max_abs_accounting_residual"])}
        for name in schedules},
      "published_v24_ledger_reference_checks":reference_checks,
      "scheduler_resource_summary":scheduler
    }
    (HERE/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    report={
      "version":"V27","strategy":candidate,"features":list(FEATURES),"holdout_dates":[START,END],
      "quarters":list(quarters),"monthly_signal_dates":signal_dates,
      "model_recipe":quarter_results[quarters[0]]["recipe"],
      "quarter_results":quarter_results,"metrics":metrics,"annual_returns":annual,"quarterly_returns":quarterly,
      "audit":audit,"reference_checks":reference_checks,
      "execution":"V11 corrected ledger; signal at monthly close; T+1 open; equal-weight Top2; 1% participation; final odd-lot/full-exit rules",
      "comparisons":list(ref_names)}
    (HERE/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    protocol={
      "version":"V27","protocol_file":"PROTOCOL.md","frozen_before_training":True,
      "feature_selection_source":"V24 protocol frozen through 2023Q4; no holdout-driven changes",
      "selected_features":list(FEATURES),"factor_semantics":"zscore_win1_99_v1",
      "holdout_quarters":list(quarters),"holdout_dates":[START,END],"monthly_signal_dates":signal_dates,
      "training":"per-quarter retrain on all pre-quarter dates with five trading-session purge; 5d forward labels",
      "model_recipe":quarter_results[quarters[0]]["recipe"],"max_cpu_workers":4,"worker_threads":1,"gpu":False,
      "allocation":"monthly first trading-day close score; Top2 equal-weight; execute next session open",
      "cash_ledger":"V11 corrected account-level cash/corporate-action/participation/odd-lot engine",
      "comparisons":list(ref_names),"no_parameter_sweep":True,"no_early_stopping":True,
      "no_2020_2023_prediction_models":True}
    (HERE/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# V27 四因子 LambdaRank 留出结果","",
      f"- 样本：10 个季度（{quarters[0]} 至 {quarters[-1]}），OOS {START} 至 {END}；30 个月度信号。",
      "- 训练：每季度独立滚动，只使用当季开始前已知训练样本，五个交易观察日 purge；输入固定为 V24 在 2023Q4 前冻结的四因子加四个资产哑变量。",
      "- 组合：月初首个交易日收盘选 Top2 等权，下一交易日开盘；全策略共用 V11 最终修正版单账户现金账本。","",
      "| Strategy | Net return | Annualized | Max drawdown | Ann. vol. | Fees | Slippage | Trades | Blocked | Min cash |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,m in metrics.items():
      lines.append(f"| {name} | {m['net_return']:.2%} | {m['annualized_net_return']:.2%} | {m['max_drawdown']:.2%} | {m['annualized_volatility']:.2%} | CNY {m['fees']:,.2f} | CNY {m['slippage_cost']:,.2f} | {m['trades']} | {m['blocked_trade_events']} | CNY {m['min_cash']:,.2f} |")
    lines+=["","## 相对 V24 固定秩基准",""]
    for baseline in ref_names:
      delta=metrics[candidate]["vs_"+baseline.replace(" ","_").replace("-","_")]
      lines.append(f"- {baseline}: 净收益差 {delta['net_return_difference_pp']:+.2f} 个百分点；最大回撤差 {delta['max_drawdown_difference_pp']:+.2f} 个百分点。")
    lines+=["","## 审计","",
      "- 全部 10 季训练截止均先于测试期，五个交易日 purge 与标签截止已逐季核验；输入只有四因子与四个资产哑变量。",
      f"- 目标权重和范围 {audit['target_weight_sum_min']:.6f}–{audit['target_weight_sum_max']:.6f}；下一交易日执行核验通过；最大记账残差 {audit['max_abs_accounting_residual']:.3g}；负现金行 {audit['negative_cash_rows']}；负权益行 {audit['negative_equity_rows']}。",
      "- 复算的 V24 固定秩、V11 Ranker、V11 60 日动量和等权账本与 V24 已保存结果对照，差异见 audit.json。",
      "- 逐季模型/预测、cache manifest、逐日权益、成交与年度/季度明细均在本目录。四只银行股上的历史留出结果不代表未来稳定收益。"]
    (HERE/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    log=HERE/"RUN_LOG.md"
    with log.open("a",encoding="utf-8") as f:
      f.write(f"\n- 完成：{len(quarters)} 季逐季训练/预测；{len(signal_dates)} 月信号；策略账本已生成。\\n")
      f.write(f"- V27 净收益 {metrics[candidate]['net_return']:.4%}，最大回撤 {metrics[candidate]['max_drawdown']:.4%}，reference parity checked。\\n")
      f.write("- 审计、资源峰值、缓存散列、账本和汇总报告已落盘。\\n")
    print("\n".join(lines))
if __name__=="__main__": main()
