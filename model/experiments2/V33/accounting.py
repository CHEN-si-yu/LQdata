#!/usr/bin/env python3
"""V33 post-fit audit, common-ledger replay, comparison, and cache manifest."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[key]="1"
import hashlib, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
V11=HERE.parent/"V11"
V24=HERE.parent/"V24"
V26=HERE.parent/"V26"
V27=HERE.parent/"V27"
DATA=HERE.parents[1]/"trainingdata"
START="2024-01-02"
END="2026-06-30"
PROTOCOL=json.loads((HERE/"protocol.json").read_text(encoding="utf-8"))
FEATURES=tuple(PROTOCOL["model_recipe"]["factor_features"])
QUARTERS=tuple(PROTOCOL["holdout_quarters"])
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def load_quarter_predictions(root, quarters, expected_features=None):
    parts=[]
    results={}
    manifest=[]
    for q in quarters:
        folder=root/"quarters"/q
        for name in ("result.json","status.json","predictions.csv","ranker.txt"):
            if not (folder/name).is_file():
                raise FileNotFoundError(folder/name)
        result=json.loads((folder/"result.json").read_text(encoding="utf-8"))
        status=json.loads((folder/"status.json").read_text(encoding="utf-8"))
        if result.get("state")!="complete" or status.get("state")!="complete" or result.get("quarter")!=q:
            raise RuntimeError(f"{root.name}/{q}: quarter result is incomplete or mislabeled")
        if expected_features is not None:
            expected=list(expected_features)+[f"asset_{i}" for i in range(4)]
            if result.get("features")!=expected:
                raise RuntimeError(f"{root.name}/{q}: model inputs differ from frozen feature list")
        pred=pd.read_csv(folder/"predictions.csv",dtype={"trade_date":str,"stock_code":str,"model_quarter":str})
        pred["trade_date"]=pred.trade_date.astype(str).str[:10]
        for col in ("monthly_signal","selected_top2"):
            pred[col]=pred[col].astype(bool)
        if pred.duplicated(["trade_date","stock_code"]).any():
            raise RuntimeError(f"{root.name}/{q}: duplicate daily score rows")
        if pred["model_quarter"].nunique()!=1 or str(pred["model_quarter"].iloc[0])!=q:
            raise RuntimeError(f"{root.name}/{q}: predictions include scores from another quarter")
        if pred.trade_date.min()<result["first_test"] or pred.trade_date.max()>result["last_test"]:
            raise RuntimeError(f"{root.name}/{q}: scores fall outside the model's test quarter")
        if len(pred)!=4*int(result["n_test_days"]):
            raise RuntimeError(f"{root.name}/{q}: expected four scores per test day")
        parts.append(pred)
        results[q]=result
        manifest.append({
            "quarter":q,
            "train_first":result["train_first"],"train_last":result["train_last"],
            "first_test":result["first_test"],"last_test":result["last_test"],
            "purged_trading_sessions":result["purged_trading_sessions"],
            "max_label_date_used":result["max_label_date_used"],
            "train_rows":result["n_train_rows_after_label_filter"],
            "train_queries":result["n_train_queries"],
            "fit_seconds":result["fit_seconds"],
            "rss_gib":result["rss_gib"],
            "prediction_rows":int(len(pred)),
            "model_sha256":sha256(folder/"ranker.txt"),
            "predictions_sha256":sha256(folder/"predictions.csv"),
            "result_sha256":sha256(folder/"result.json"),
        })
    combined=pd.concat(parts,ignore_index=True)
    if combined.duplicated(["trade_date","stock_code"]).any():
        raise RuntimeError(f"{root.name}: overlapping quarterly predictions")
    return combined,results,manifest

def schedule_from_ranker(pred, signal_dates=None):
    found=sorted(pred.loc[pred.monthly_signal,"trade_date"].unique())
    if signal_dates is not None and found!=list(signal_dates):
        raise RuntimeError("quarter-local Ranker signal dates differ from the shared V11 calendar")
    schedule={}
    for d in found:
        day=pred[(pred.trade_date==d)&pred.monthly_signal]
        picks=day.loc[day.selected_top2].sort_values(["daily_rank","stock_code"]).stock_code.tolist()
        if len(picks)!=2:
            raise RuntimeError(f"{d}: expected exactly two selected securities")
        schedule[d]={code:0.5 for code in picks}
    return schedule,found

def schedule_from_membership(path, signal_dates, top2_field="selected_top2"):
    rows=json.loads(path.read_text(encoding="utf-8"))
    schedule={}
    for row in rows:
        d=row["signal_date"]
        if START<=d<=END:
            picks=list(row[top2_field])
            if len(picks)!=2:
                raise RuntimeError(f"{path}: {d} must select two names")
            schedule[d]={code:0.5 for code in picks}
    if sorted(schedule)!=list(signal_dates):
        raise RuntimeError(f"{path}: membership dates differ from common V11 calendar")
    return schedule

def check_against_saved(actual_name, schedule_metrics, daily_all, trades_df, reference_dir, reference_name, tolerance=1e-3):
    ref_metrics_all=json.loads((reference_dir/"metrics.json").read_text(encoding="utf-8"))["metrics"]
    if reference_name not in ref_metrics_all:
        raise RuntimeError(f"{reference_dir.name}: missing reference metric {reference_name}")
    actual=schedule_metrics[actual_name]
    expected=ref_metrics_all[reference_name]
    metric_fields=("net_return","fees","slippage_cost","turnover","max_drawdown","annualized_volatility","min_cash")
    metric_diff={field:abs(float(actual[field])-float(expected[field])) for field in metric_fields}
    trade_count_equal=actual["trades"]==expected["trades"]

    old_daily=pd.read_csv(reference_dir/"daily_equity.csv",dtype={"trade_date":str})
    old_daily=old_daily.loc[old_daily.strategy==reference_name].sort_values("trade_date").reset_index(drop=True)
    new_daily=daily_all.loc[daily_all.strategy==actual_name].sort_values("trade_date").reset_index(drop=True)
    if len(old_daily)!=len(new_daily) or old_daily.trade_date.tolist()!=new_daily.trade_date.tolist():
        raise RuntimeError(f"{actual_name}: reference daily date coverage differs")
    daily_fields=("equity_net","equity_gross","cash","fees_cumulative","slippage_cumulative",
                  "turnover_cumulative","accounting_residual","daily_return_net","drawdown")
    daily_diff={field:float(np.max(np.abs(old_daily[field].to_numpy(float)-new_daily[field].to_numpy(float))))
                for field in daily_fields}
    shares_exact=old_daily["shares_json"].astype(str).tolist()==new_daily["shares_json"].astype(str).tolist()

    old_trades=pd.read_csv(reference_dir/"trades.csv",dtype={"trade_date":str,"stock_code":str})
    old_trades=old_trades.loc[old_trades.strategy==reference_name]
    new_trades=trades_df.loc[trades_df.strategy==actual_name]
    keys=("trade_date","stock_code","side","reason")
    numeric=("shares","open","fill","notional","fee","slippage_cost")
    old_trades=old_trades.sort_values(list(keys)).reset_index(drop=True)
    new_trades=new_trades.sort_values(list(keys)).reset_index(drop=True)
    trade_rows_match=len(old_trades)==len(new_trades)
    if trade_rows_match:
        for field in keys:
            trade_rows_match=trade_rows_match and old_trades[field].astype(str).tolist()==new_trades[field].astype(str).tolist()
        trade_diff={field:(float(np.max(np.abs(old_trades[field].to_numpy(float)-new_trades[field].to_numpy(float))))
                           if len(old_trades) else 0.0) for field in numeric}
    else:
        trade_diff={field:float("inf") for field in numeric}
    passed=(all(x<=tolerance for x in metric_diff.values()) and trade_count_equal
            and all(x<=tolerance for x in daily_diff.values()) and shares_exact and trade_rows_match
            and all(x<=tolerance for x in trade_diff.values()))
    if not passed:
        raise RuntimeError(f"{actual_name}: ledger replay differs from {reference_dir.name} saved baseline")
    return {
        "reference_dir":reference_dir.name,"reference_strategy":reference_name,
        "metric_abs_differences":metric_diff,"trade_count_equal":trade_count_equal,
        "daily_rows":len(old_daily),"daily_max_abs_differences":daily_diff,
        "shares_json_exact_match":shares_exact,"trade_rows_match":trade_rows_match,
        "trade_max_abs_differences":trade_diff,"tolerance":tolerance
    }

def main():
    import sys
    sys.path.insert(0,str(V11))
    import analysis as E

    v33_pred,quarter_results,model_manifest=load_quarter_predictions(HERE,QUARTERS,FEATURES)
    if v33_pred.trade_date.min()!=START or v33_pred.trade_date.max()!=END:
        raise RuntimeError("V33 quarterly scores do not cover exact frozen holdout dates")
    signal_dates=sorted(v33_pred.loc[v33_pred.monthly_signal,"trade_date"].unique())
    if len(signal_dates)!=30:
        raise RuntimeError(f"expected 30 monthly V33 signal dates; got {len(signal_dates)}")
    v33_schedule,v33_dates=schedule_from_ranker(v33_pred,signal_dates)

    v27_pred,_,v27_models=load_quarter_predictions(V27,QUARTERS)
    v27_schedule,v27_dates=schedule_from_ranker(v27_pred,signal_dates)
    v24_schedule=schedule_from_membership(V24/"membership_audit.json",signal_dates)
    v26_schedule=schedule_from_membership(V26/"membership_audit.json",signal_dates)
    v11_pred=E.load_predictions(list(QUARTERS))
    prices=E.load_prices()
    v11_all,v11_dates=E.schedules(v11_pred,prices)
    if sorted(v11_dates)!=signal_dates or v27_dates!=signal_dates or v33_dates!=signal_dates:
        raise RuntimeError("shared monthly signal calendars differ across models")
    v11_rank={d:v11_all["V11 LambdaRank Top2"][d] for d in signal_dates}
    v11_mom={d:v11_all["60d momentum Top2"][d] for d in signal_dates}
    schedules={
        "V33 stable-20-factor LambdaRank Top2":v33_schedule,
        "V27 four-factor LambdaRank Top2":v27_schedule,
        "V24 fixed-rank Top2":v24_schedule,
        "V26 mf_tier_flow_agreement_20 Top2":v26_schedule,
        "V11 LambdaRank Top2":v11_rank,
        "60d momentum Top2":v11_mom,
        "equal-weight hold":{signal_dates[0]:{code:0.25 for code in E.CODES}},
    }

    daily_parts=[]
    all_trades=[]
    metrics={}
    for name,target_schedule in schedules.items():
        daily,trades,metric=E.simulate(name,prices,target_schedule,START,END)
        daily_parts.append(daily)
        all_trades.extend(trades)
        metrics[name]=metric
    daily_all=pd.concat(daily_parts,ignore_index=True)
    daily_all["gross_return_daily"]=daily_all.groupby("strategy",sort=False).equity_gross.pct_change().fillna(0.0)
    trades_df=pd.DataFrame(all_trades)
    quarterly=[row for daily in daily_parts for row in E.period_returns(daily,"Q")]
    annual=[row for daily in daily_parts for row in E.period_returns(daily,"Y")]

    date_axis=prices["days"]
    date_ix=prices["index"]
    leakage_checks={}
    for quarter,result in quarter_results.items():
        if result.get("purged_trading_sessions")!=5:
            raise RuntimeError(f"{quarter}: missing five-session purge")
        if result.get("max_label_date_used")!=result.get("train_last"):
            raise RuntimeError(f"{quarter}: label cutoff differs from training cutoff")
        if result.get("first_test") not in date_ix or result.get("train_last") not in date_ix:
            raise RuntimeError(f"{quarter}: boundary dates missing from V11 trading calendar")
        gap=date_ix[result["first_test"]]-date_ix[result["train_last"]]-1
        if gap!=5 or result["train_last"]>=result["first_test"]:
            raise RuntimeError(f"{quarter}: purge interval not exactly five trading sessions")
        q_start=pd.Period(quarter,freq="Q").start_time.strftime("%Y-%m-%d")
        q_end=pd.Period(quarter,freq="Q").end_time.strftime("%Y-%m-%d")
        if result["first_test"]<q_start or result["last_test"]>q_end:
            raise RuntimeError(f"{quarter}: model scored outside its own quarter")
        leakage_checks[quarter]={
            "train_first":result["train_first"],"train_last":result["train_last"],
            "first_test":result["first_test"],"last_test":result["last_test"],
            "purged_trading_sessions":gap,"max_label_date_used":result["max_label_date_used"],
            "train_precedes_test":result["train_last"]<result["first_test"],
            "scores_only_in_quarter":True,"n_train_rows":result["n_train_rows_after_label_filter"]
        }

    reference_map={
        "V27 four-factor LambdaRank Top2":(V27,"V27 four-factor LambdaRank Top2"),
        "V24 fixed-rank Top2":(V24,"V24 stable-factor consensus Top2"),
        "V26 mf_tier_flow_agreement_20 Top2":(V26,"V26 mf_tier_flow_agreement_20 Top2"),
        "V11 LambdaRank Top2":(V26,"V11 LambdaRank Top2"),
        "60d momentum Top2":(V26,"60d momentum Top2"),
        "equal-weight hold":(V26,"equal-weight hold"),
    }
    parity={}
    for actual_name,(ref_dir,ref_name) in reference_map.items():
        parity[actual_name]=check_against_saved(actual_name,metrics,daily_all,trades_df,ref_dir,ref_name)

    candidate="V33 stable-20-factor LambdaRank Top2"
    for baseline in reference_map:
        key="vs_"+baseline.replace(" ","_").replace("-","_")
        metrics[candidate][key]={
            "net_return_difference_pp":100*(metrics[candidate]["net_return"]-metrics[baseline]["net_return"]),
            "max_drawdown_difference_pp":100*(metrics[candidate]["max_drawdown"]-metrics[baseline]["max_drawdown"]),
        }

    signal_rows=[]
    for strategy,target_schedule in schedules.items():
        for signal_date,target in target_schedule.items():
            execution_date=str(date_axis[date_ix[signal_date]+1])
            signal_rows.extend({
                "signal_date":signal_date,"execution_date":execution_date,
                "strategy":strategy,"stock_code":code,"target_weight":float(weight)
            } for code,weight in sorted(target.items()))
    signal_df=pd.DataFrame(signal_rows)
    target_sums=signal_df.groupby(["signal_date","strategy"]).target_weight.sum()
    if not np.allclose(target_sums.to_numpy(),1.0,rtol=0,atol=1e-12):
        raise RuntimeError("a schedule's target weights do not sum to one")
    if not ((signal_df.execution_date.map(date_ix)-signal_df.signal_date.map(date_ix))==1).all():
        raise RuntimeError("one or more orders were not scheduled for next session")

    daily_all.to_csv(HERE/"daily_equity.csv",index=False,float_format="%.10g")
    trades_df.to_csv(HERE/"trades.csv",index=False,float_format="%.10g")
    signal_df.to_csv(HERE/"monthly_signals.csv",index=False,float_format="%.10g")
    pd.DataFrame(quarterly).to_csv(HERE/"quarterly_returns.csv",index=False,float_format="%.10g")
    pd.DataFrame(annual).to_csv(HERE/"annual_returns.csv",index=False,float_format="%.10g")
    pd.DataFrame(model_manifest).to_csv(HERE/"quarter_training_manifest.csv",index=False,float_format="%.10g")

    meta=json.loads((DATA/"meta.json").read_text(encoding="utf-8"))
    data_years=sorted(int(y) for y in meta["built_years"])
    paths=[]
    def add(path,role):
        path=Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append({"role":role,"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":sha256(path)})
    add(DATA/"meta.json","training-data metadata and factor semantics")
    add(V11/"analysis.py","V11 final corrected ledger engine")
    add(V11/"model.py","V11 reference model recipe source")
    add(HERE/"freeze_protocol.py","V33 pretraining feature-selection protocol generator")
    add(HERE/"protocol.json","V33 frozen protocol")
    add(HERE/"selected_feature_list.json","V33 pretraining top-20 factor list")
    add(HERE/"training_importance_ranking.csv","V33 full ranked pretraining feature importance table")
    add(HERE/"importance_hashes.json","V33 hashes of 16 pre-holdout V11 importance artifacts")
    add(HERE/"model.py","V33 V11-recipe ranker implementation")
    add(HERE/"scheduler.py","V33 four-worker CPU/resource scheduler")
    add(HERE/"run.py","V33 run entry point")
    for year in data_years:
        add(DATA/"factors"/f"year={year}"/"data.parquet",f"factor cache year {year}")
        add(DATA/"target"/f"year={year}"/"data.parquet",f"five-day label cache year {year}")
        add(DATA/"prices"/f"year={year}"/"data.parquet",f"market price cache year {year}")
        add(DATA/"amount"/f"year={year}"/"data.parquet",f"trading amount cache year {year}")
    for q in PROTOCOL["factor_selection"]["importance_quarters"]:
        add(V11/"quarters"/q/"result.json",f"pre-holdout training importance {q}")
    for q in QUARTERS:
        add(V11/"quarters"/q/"predictions.csv",f"V11 holdout predictions {q}")
        for filename,role in (("ranker.txt","V33 quarter model"),("predictions.csv","V33 quarter scores"),
                              ("result.json","V33 quarter training audit"),("status.json","V33 quarter status")):
            add(HERE/"quarters"/q/filename,f"{role} {q}")
        for filename,role in (("ranker.txt","V27 comparator quarter model"),("predictions.csv","V27 comparator quarter scores"),
                              ("result.json","V27 comparator training audit")):
            add(V27/"quarters"/q/filename,f"{role} {q}")
    for directory,files,label in (
        (V24,("protocol.json","membership_audit.json","metrics.json","daily_equity.csv","trades.csv"),"V24 comparator"),
        (V26,("protocol.json","membership_audit.json","metrics.json","daily_equity.csv","trades.csv"),"V26 comparator"),
        (V27,("PROTOCOL.md","metrics.json","daily_equity.csv","trades.csv"),"V27 comparator")):
        for filename in files:
            add(directory/filename,f"{label} {filename}")
    hashes={"algorithm":"SHA-256","scope":"Training feature/label/market cache parquet files, the exact V11 training-importance artifacts, all V11/V27 reference prediction/model artifacts, V24/V26/V27 frozen baseline ledgers, V33 protocol and all V33 per-quarter models/predictions/results.",
            "records":paths}
    (HERE/"cache_manifest.json").write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    scheduler_path=HERE/"scheduler_summary.json"
    scheduler=json.loads(scheduler_path.read_text(encoding="utf-8")) if scheduler_path.is_file() else {}
    resource_rows=[]
    resource_path=HERE/"resource_log.csv"
    if resource_path.is_file():
        resource_rows=pd.read_csv(resource_path).to_dict(orient="records")
    audit={
        "window":[START,END],"holdout_quarters":list(QUARTERS),
        "holdout_days":int(len(daily_parts[0])),"monthly_signal_count":len(signal_dates),
        "selected_factor_count":len(FEATURES),"selected_factors":list(FEATURES),
        "importance_cutoff":"2023Q4","importance_quarters_used":16,
        "training_importance_selection_hash_count":len(json.loads((HERE/"importance_hashes.json").read_text())["importance_files"]),
        "all_v33_quarters_completed":len(quarter_results)==10,
        "quarter_boundaries":leakage_checks,
        "all_quarters_have_five_session_purge":all(x["purged_trading_sessions"]==5 for x in leakage_checks.values()),
        "all_labels_end_at_train_cutoff":all(x["max_label_date_used"]==x["train_last"] for x in leakage_checks.values()),
        "v33_scores_are_quarter_local":all(x["scores_only_in_quarter"] for x in leakage_checks.values()),
        "signal_dates_match_v11_calendar":v33_dates==signal_dates,
        "all_orders_execute_next_session":True,
        "target_weight_sum_min":float(target_sums.min()),
        "target_weight_sum_max":float(target_sums.max()),
        "daily_rows":int(len(daily_all)),"trade_rows":int(len(trades_df)),
        "negative_cash_rows":int((daily_all.cash<-1e-7).sum()),
        "negative_equity_rows":int((daily_all.equity_net<-1e-7).sum()),
        "max_abs_accounting_residual":float(daily_all.accounting_residual.abs().max()),
        "strategies":{name:{
            "blocked_trade_events":int(metrics[name]["blocked_trade_events"]),
            "max_positions":int(metrics[name]["max_positions"]),
            "min_cash":float(metrics[name]["min_cash"]),
            "trades":int(metrics[name]["trades"]),
            "max_abs_accounting_residual":float(metrics[name]["max_abs_accounting_residual"])
        } for name in schedules},
        "common_baseline_replay_checks":parity,
        "scheduler_summary":scheduler,
        "resource_samples":resource_rows
    }
    (HERE/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

    table=["| Strategy | Net return | Annualized | Max drawdown | Ann. vol. | Fees | Slippage | Trades |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,metric in metrics.items():
        table.append(f"| {name} | {metric['net_return']:.2%} | {metric['annualized_net_return']:.2%} | {metric['max_drawdown']:.2%} | {metric['annualized_volatility']:.2%} | {metric['fees']:.2f} | {metric['slippage_cost']:.2f} | {metric['trades']} |")
    train_lines=["| Quarter | Train first–last | Test first–last | Purge sessions | Train rows | Fit seconds | RSS GiB |",
                 "|---|---|---|---:|---:|---:|---:|"]
    for quarter,result in quarter_results.items():
        train_lines.append(f"| {quarter} | {result['train_first']}–{result['train_last']} | {result['first_test']}–{result['last_test']} | {result['purged_trading_sessions']} | {result['n_train_rows_after_label_filter']} | {result['fit_seconds']:.2f} | {result['rss_gib']['at_completion']:.3f} |")
    report=[
        "# V33：V11训练期稳定20因子 LambdaRank 冻结留出",
        "",
        "## 预先冻结的因子与训练协议",
        "",
        "在任何V33季度训练开始前，只用V11 2020Q1–2023Q4的16份训练折重要性文件，按top-20出现次数降序、top-20内归一化gain均值降序、特征名升序冻结前20因子。名单、归一化gain、来源文件SHA-256见training_importance_ranking.csv、selected_feature_list.json和importance_hashes.json。",
        "",
        "训练期只包括2024Q1至2026Q2的10个walk-forward季度，每季单独拟合并只为该季打分。模型使用固定V11 LambdaRank配方，输入为冻结的20个因子加4个资产哑变量。每个季度严格purge五个交易日，标签末日不晚于训练截止；不重训早期预测、不早停、不扫参、不使用留出结果选择因子或模型。",
        "",
        "组合按月初首个交易日收盘选V33模型Top2等权，下一交易日开盘执行。比较项与V24/V26/V27/V11使用同一2024-01-02至2026-06-30生命周期和V11最终现金账本。期末按最后收盘估值，不强制平仓。",
        "",
        "## 回测结果",
        "",
        *table,
        "",
        "## 季度训练清单",
        "",
        *train_lines,
        "",
        "## 审计与资源",
        "",
        f"- 训练期因子选择来自2023Q4以前的16份V11重要性文件，冻结20个特征；五日purge、训练标签截止、当季内评分检查全部通过。",
        f"- 一共{len(signal_dates)}个月初信号，{len(QUARTERS)}个季度模型；逐日/逐笔共用V11修正账本，最大会计残差{audit['max_abs_accounting_residual']:.3g}，负现金行{audit['negative_cash_rows']}，负权益行{audit['negative_equity_rows']}。",
        f"- 调度器配置最多4个CPU worker、每worker一线程、GPU关闭、内存硬限180 GiB；采样峰值主机已用内存{scheduler.get('host_used_peak_sampled_gib','n/a')} GiB，worker RSS峰值合计{scheduler.get('worker_rss_sum_peak_sampled_gib','n/a')} GiB。",
        "- V27四因子Ranker、V24固定秩、V26单因子、V11完整Ranker、60日动量与等权账本均与已保存基准逐日、逐笔复核；误差见audit.json。",
        "- 原始训练/标签/行情缓存、重要性来源、逐季模型和预测、比较账本的SHA-256见cache_manifest.json。",
        "",
        "该样本只覆盖四家银行的历史留出期，不能单独证明未来稳定超额。"
    ]
    (HERE/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    (HERE/"metrics.json").write_text(json.dumps({
        "version":"V33","window":[START,END],"days":len(daily_parts[0]),
        "signal_count":len(signal_dates),"quarters":list(QUARTERS),
        "selected_features":list(FEATURES),"feature_selection":PROTOCOL["factor_selection"]["selected_statistics"],
        "metrics":metrics,"quarterly_returns":quarterly,"annual_returns":annual,
        "reference_parity":parity
    },ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("\n".join(table))
    print(f"Audited {len(signal_dates)} signals, {len(QUARTERS)} models; input/model cache hashes: {len(paths)} files.")

if __name__=="__main__":
    main()

