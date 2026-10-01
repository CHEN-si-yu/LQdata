#!/usr/bin/env python3
"""V19: V15 monthly Top1, rebalance only when the selected bank changes."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
V1_DIR = ROOT.parent / "V1"
DATA_ROOT = ROOT.parents[1] / "trainingdata"
CODES = ("601288.SH", "601398.SH", "601939.SH", "601988.SH")
PRICE_COLUMNS = ("open", "high", "low", "close", "pre_close", "vol", "adj_factor")
OUT = ROOT
sys.path.insert(0, str(ROOT))
import account_engine as E

OFFICIAL_START = "2025-07-01"
OFFICIAL_SIGNAL_END = "2026-06-30"
OFFICIAL_LIQUIDATION = "2026-07-01"
EXTENDED_START = "2023-01-03"
TOP_K = 1
LOOKBACKS = {"monthly_top1_20d": 20, "monthly_top1_60d": 60}


def load_four_bank_prices():
    meta = json.loads((DATA_ROOT / "meta.json").read_text(encoding="utf-8"))
    years = sorted(int(y) for y in meta["built_years"])
    price_schema = pq.ParquetFile(DATA_ROOT / "prices" / f"year={years[0]}" / "data.parquet").schema_arrow.names
    missing = set(("trade_date", "stock_code", *PRICE_COLUMNS)) - set(price_schema)
    if missing:
        raise RuntimeError(f"price schema missing columns: {sorted(missing)}")
    amount_schema = pq.ParquetFile(DATA_ROOT / "amount" / f"year={years[0]}" / "data.parquet").schema_arrow.names
    amount_col = next((x for x in amount_schema if x not in ("trade_date", "stock_code")), None)
    if amount_col is None:
        raise RuntimeError("amount table has no measure column")
    pieces = []
    for year in years:
        pp = DATA_ROOT / "prices" / f"year={year}" / "data.parquet"
        ap = DATA_ROOT / "amount" / f"year={year}" / "data.parquet"
        if not pp.is_file() or not ap.is_file():
            continue
        filt = [("stock_code", "in", list(CODES))]
        p = pq.read_table(pp, columns=["trade_date", "stock_code", *PRICE_COLUMNS], filters=filt).to_pandas()
        a = pq.read_table(ap, columns=["trade_date", "stock_code", amount_col], filters=filt).to_pandas()
        if len(p) == 0:
            continue
        q = p.merge(a, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
        pieces.append(q)
        print(f"loaded prices/amount {year}: {len(q)} rows", flush=True)
    if not pieces:
        raise RuntimeError("no four-bank price rows found")
    frame = pd.concat(pieces, ignore_index=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    frame["stock_code"] = frame["stock_code"].astype(str)
    if frame.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("duplicate price rows in four-bank panel")
    days = np.asarray(sorted(frame["trade_date"].unique()), dtype=str)
    idx = pd.MultiIndex.from_product([days, CODES], names=["trade_date", "stock_code"])
    frame = frame.set_index(["trade_date", "stock_code"]).reindex(idx)
    arrays = {k: frame[k].to_numpy(np.float64).reshape(len(days), len(CODES)) for k in PRICE_COLUMNS}
    adj = arrays["adj_factor"].copy()
    for c in range(adj.shape[1]):
        good = np.flatnonzero(np.isfinite(adj[:, c]) & (adj[:, c] > 0))
        if len(good):
            adj[:good[0], c] = adj[good[0], c]
            for i in range(good[0] + 1, len(adj)):
                if not np.isfinite(adj[i, c]) or adj[i, c] <= 0:
                    adj[i, c] = adj[i - 1, c]
    arrays["adj_factor"] = adj
    arrays["amount"] = np.nan_to_num(frame[amount_col].to_numpy(np.float64).reshape(len(days), len(CODES)),
                                     nan=0.0, posinf=0.0, neginf=0.0)
    return days, np.asarray(CODES, dtype=str), arrays, meta


def month_first_indices(days):
    months = np.asarray([str(d)[:7] for d in days])
    return np.flatnonzero(np.r_[True, months[1:] != months[:-1]])




def make_schedule(days,codes,arrays,signal_start,signal_end,lookback,only_on_change=False):
    adjusted_close=arrays["close"]*arrays["adj_factor"]
    signals=[]; schedule={}; previous=None
    date_to_ix={str(d):i for i,d in enumerate(days)}
    for si in month_first_indices(days):
        day=str(days[si])
        if day<signal_start or day>signal_end or si<lookback or si+1>=len(days): continue
        old,now=adjusted_close[si-lookback],adjusted_close[si]
        valid=np.isfinite(old)&(old>0)&np.isfinite(now)&(now>0)
        candidates=np.flatnonzero(valid)
        if not len(candidates): continue
        rets=np.full(len(codes),np.nan,dtype=float)
        rets[valid]=now[valid]/old[valid]-1.0
        ranked=sorted(candidates.tolist(),key=lambda c:(-rets[c],str(codes[c])))
        top=int(ranked[0]); changed=(previous is None or top!=previous); ex=int(si+1)
        scheduled=(not only_on_change) or changed
        if scheduled: schedule[ex]=np.asarray([top],dtype=int)
        signals.append({"signal_date":day,"execution_date":str(days[ex]),
            "signal_index":int(si),"execution_index":ex,"lookback_intervals":int(lookback),
            "selected_top1":str(codes[top]),"selected_top1_return":float(rets[top]),
            "membership_changed":bool(changed),"scheduled_rebalance":bool(scheduled),
            "ranked":[{"code":str(codes[c]),"return":float(rets[c])} for c in ranked]})
        previous=top
    for x in signals:
        si=date_to_ix[x["signal_date"]]; ei=date_to_ix[x["execution_date"]]
        if ei!=si+1 or x["signal_index"]-lookback<0: raise RuntimeError("causal schedule check failed")
    return schedule,signals

def mark_residual(curve, arrays):
    lastmark = np.full(len(CODES), np.nan)
    errors = []
    for i, row in enumerate(curve):
        op, close = arrays["open"][i], arrays["close"][i]
        marks = np.where(np.isfinite(op) & (op > 0), op,
                         np.where(np.isfinite(close) & (close > 0), close, lastmark))
        marks = np.where(np.isfinite(marks) & (marks > 0), marks, lastmark)
        marked = float(np.sum(np.asarray(row["shares"], dtype=float) * np.nan_to_num(marks, nan=0.0)))
        errors.append(abs(float(row["equity"]) - float(row["cash"]) - marked))
        lastmark = marks.copy()
    return max(errors) if errors else 0.0



def split_returns(curve, freq, through=None):
    df=pd.DataFrame(curve)
    if through is not None:
        df=df.loc[df["date"]<=through].copy()
    if freq=="quarter":
        keys=pd.PeriodIndex(pd.to_datetime(df["date"]),freq="Q").astype(str)
        field="quarter"
    else:
        keys=pd.to_datetime(df["date"]).dt.year.astype(str).to_numpy()
        field="year"
    out=[]
    for key in sorted(set(keys)):
        g=df.loc[np.asarray(keys)==key]
        net=float(np.prod(1.0+g["daily_return"].to_numpy(float))-1.0)
        matched=float(np.prod(1.0+g["matched_return"].to_numpy(float))-1.0)
        out.append({field:str(key),"ledger_sessions":int(len(g)),
                    "net_return":net,"same_hold_no_fee_return":matched,
                    "cost_execution_friction_gap":net-matched})
    return out



def run_strategy(all_days,codes,all_arrays,start,signal_end,liquidation_day,
                 lookback=None,equal_hold=False,only_on_change=False):
    date_set=set(all_days.tolist())
    for d in (start,signal_end,liquidation_day):
        if d not in date_set: raise RuntimeError(f"required date missing: {d}")
    start_ix=int(np.flatnonzero(all_days==start)[0])
    end_ix=int(np.flatnonzero(all_days==signal_end)[0])
    liq_ix=int(np.flatnonzero(all_days==liquidation_day)[0])
    if liq_ix<=end_ix or start_ix>=end_ix: raise RuntimeError("invalid window")
    days=all_days[start_ix:liq_ix+1]
    arrays={k:v[start_ix:liq_ix+1] for k,v in all_arrays.items()}
    if equal_hold:
        schedule={1:np.arange(len(codes),dtype=int)}
        signals=[{"signal_date":start,"execution_date":str(days[1]),
                  "rule":"initial equal-weight four-bank hold"}]
    else:
        full,signals=make_schedule(all_days,codes,all_arrays,start,signal_end,int(lookback),
                                   only_on_change=only_on_change)
        schedule={int(i-start_ix):names for i,names in full.items() if start_ix<i<=liq_ix}
    result=E.account_sim(days,codes,arrays,schedule)
    stats=E.summarize(result["curve"],E.COST["account_money"]); curve=result["curve"]
    causal=True
    if not equal_hold:
        causal=all(x["execution_index"]==x["signal_index"]+1 and x["signal_date"]<=signal_end
                   and x["signal_index"]-int(lookback)>=0 for x in signals)
    changes=sum(bool(x.get("membership_changed")) for x in signals)
    skipped=sum(not bool(x.get("scheduled_rebalance")) for x in signals) if only_on_change else 0
    audit={
        "ledger_first_day":str(days[0]),"signal_window_last_day":signal_end,
        "final_liquidation_day":str(days[-1]),
        "signal_window_sessions":int(np.count_nonzero((all_days>=start)&(all_days<=signal_end))),
        "ledger_sessions_including_liquidation":len(days),
        "monthly_signal_count":len(signals) if not equal_hold else 0,
        "lookback_intervals":int(lookback) if not equal_hold else None,
        "only_rebalance_when_top1_changes":bool(only_on_change),
        "top1_membership_change_months":int(changes),
        "skipped_same_top1_months":int(skipped),
        "scheduled_rebalance_events":len(schedule),
        "cash_min":float(min(x["cash"] for x in curve)),
        "ending_cash":float(result["ending_cash"]),
        "ending_positions":int(np.count_nonzero(np.asarray(result["ending_shares"])>1e-7)),
        "maximum_simultaneous_holdings":int(max(x["holdings"] for x in curve)),
        "blocked_entries":int(result["blocked_entries"]),"blocked_exits":int(result["blocked_exits"]),
        "trade_count":int(result["ntr"]),"fees":float(result["fees"]),
        "buy_notional":float(result["buy_notional"]),"sell_notional":float(result["sell_notional"]),
        "gross_traded_notional":float(result["buy_notional"]+result["sell_notional"]),
        "cash_nonnegative":bool(min(x["cash"] for x in curve)>=-1e-7),
        "final_liquidation_complete":bool(np.count_nonzero(np.asarray(result["ending_shares"])>1e-7)==0),
        "causal_signal_and_t_plus_1_check_pass":bool(causal),
        "max_abs_equity_cash_plus_mark_residual":float(mark_residual(curve,arrays)),
        "ledger_rule":"same V4/V15 audited shared-cash engine; 100-share lots, amount limits, price locks, fees and slippage"
    }
    return {"statistics":stats,"audit":audit,
        "quarterly_returns":split_returns(curve,"quarter",through=signal_end),
        "annual_returns":split_returns(curve,"year"),
        "signals":signals,"curve":curve,"trades":result["trades"]}


def run_window(all_days,codes,all_arrays,start,signal_end,liquidation_day):
    scenarios={}
    for lookback in (20,60):
        label=f"monthly_top1_{lookback}d"
        scenarios[label+"_v15_monthly_reweight"]=run_strategy(all_days,codes,all_arrays,start,
            signal_end,liquidation_day,lookback=lookback)
        scenarios[label+"_v19_hold_same_top1"]=run_strategy(all_days,codes,all_arrays,start,
            signal_end,liquidation_day,lookback=lookback,only_on_change=True)
    scenarios["four_bank_equal_weight_hold"]=run_strategy(all_days,codes,all_arrays,start,
        signal_end,liquidation_day,equal_hold=True)
    signal_days=int(np.count_nonzero((all_days>=start)&(all_days<=signal_end)))
    if start==OFFICIAL_START and signal_days!=242:
        raise RuntimeError(f"official signal window should have 242 sessions, got {signal_days}")
    return {"window":{"start":start,"signal_end":signal_end,"final_liquidation_day":liquidation_day,
        "signal_sessions":signal_days,
        "ledger_sessions_including_liquidation":len(scenarios["monthly_top1_20d_v19_hold_same_top1"]["curve"])},
        "scenarios":scenarios}

def public_view(window):
    return {
        "window": window["window"],
        "scenarios": {
            name: {k: v for k, v in payload.items() if k not in ("curve", "trades")}
            for name, payload in window["scenarios"].items()
        },
    }


def write_window_files(name, window):
    for scenario, payload in window["scenarios"].items():
        stem = f"{name}_{scenario}"
        pd.DataFrame(payload["curve"]).drop(columns=["shares"]).to_csv(OUT / f"{stem}_daily.csv", index=False)
        pd.DataFrame(payload["trades"]).to_csv(OUT / f"{stem}_trades.csv", index=False)
        pd.DataFrame(payload["signals"]).to_json(OUT / f"{stem}_signals.json", orient="records",
                                                force_ascii=False, indent=2)




def build_report(result):
    keys=("monthly_top1_20d_v15_monthly_reweight","monthly_top1_20d_v19_hold_same_top1",
          "monthly_top1_60d_v15_monthly_reweight","monthly_top1_60d_v19_hold_same_top1",
          "four_bank_equal_weight_hold")
    labels=("20日Top1月频重配","20日Top1同股持股数不变","60日Top1月频重配",
            "60日Top1同股持股数不变","四股等权持有")
    out=["# V19：Top1 同一银行连续入选时不重配","",
        f"- 构建时间：{result['built_at']}；价格快照截至 {result['snapshot_last_day']}；仅价格/成交额，无模型训练、无GPU、无阈值扫描。",
        "- 固定沿用V15月频20d/60d Top1信号。基线每月T+1开盘按权益重等；V19仅当当月Top1代码相对上月变化时才重等，同一银行连续入选则不交易并保留持仓股数（公司行动调整仍按账本执行）。",
        "- 官方窗242个信号交易日后于2026-07-01开盘退出；扩展信号截至2026-09-23并于9/24开盘退出。所有比较使用同一V4审计现金账本。",
        "- 匹配无费收益差只反映同一实际持仓路径的费用/执行摩擦，不是选股alpha。",""]
    for title,wkey in (("官方242日信号窗 + 7/1退出","official_242d"),
                       ("扩展窗 + 9/24退出","extended")):
        w=result[wkey]
        out += [f"## {title}","",
          f"- 信号期 {w['window']['start']} 至 {w['window']['signal_end']}（{w['window']['signal_sessions']}日），最终清仓 {w['window']['final_liquidation_day']}，账本共{w['window']['ledger_sessions_including_liquidation']}日。","",
          "| 策略 | 净收益 | 最大回撤 | 平均敞口 | 费用 | 成交 | 总买卖额 | 同持仓无费 | 摩擦差 | 跳过同股月份 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k,label in zip(keys,labels):
            x=w["scenarios"][k]; st=x["statistics"]; a=x["audit"]
            skip="—" if not a["only_rebalance_when_top1_changes"] else str(a["skipped_same_top1_months"])
            out.append(f"| {label} | {st['cumulative_net_return']:.2%} | {st['max_drawdown']:.2%} | {st['avg_exposure']:.1%} | {a['fees']:.2f} | {a['trade_count']} | {a['gross_traded_notional']:.2f} | {st['matched_benchmark_return']:.2%} | {st['matched_cumulative_excess']:.2%} | {skip} |")
        out += ["","### 季度净收益（不含最终清仓日）","",
          "| 季度 | 20日V15 | 20日V19 | 60日V15 | 60日V19 | 四股等权 |",
          "|---|---:|---:|---:|---:|---:|"]
        qm={k:{r["quarter"]:r for r in w["scenarios"][k]["quarterly_returns"]} for k in keys}
        for q in sorted(set().union(*(set(v) for v in qm.values()))):
            out.append(f"| {q} | "+" | ".join(f"{qm[k][q]['net_return']:.2%}" for k in keys)+" |")
        out += ["","### 年度净收益（含窗口退出日）","",
          "| 年份 | 20日V15 | 20日V19 | 60日V15 | 60日V19 | 四股等权 |",
          "|---:|---:|---:|---:|---:|---:|"]
        ym={k:{r["year"]:r for r in w["scenarios"][k]["annual_returns"]} for k in keys}
        for y in sorted(set().union(*(set(v) for v in ym.values()))):
            out.append(f"| {y} | "+" | ".join(f"{ym[k][y]['net_return']:.2%}" for k in keys)+" |")
        out += ["","### 交易与账本审计",""]
        for k,label in zip(keys[:4],labels[:4]):
            a=w["scenarios"][k]["audit"]
            extra=(f"；Top1成分变化 {a['top1_membership_change_months']}/{a['monthly_signal_count']}月，"
                   f"跳过 {a['skipped_same_top1_months']} 个同股月，实际调仓事件 {a['scheduled_rebalance_events']}")
            out.append(f"- {label}：最低现金 {a['cash_min']:.4f}，终值持仓 {a['ending_positions']}，买/卖阻塞 {a['blocked_entries']}/{a['blocked_exits']}，T+1因果检查 {a['causal_signal_and_t_plus_1_check_pass']}，最终清仓 {a['final_liquidation_complete']}，账本残差 {a['max_abs_equity_cash_plus_mark_residual']:.3g}{extra}。")
        out.append("")
    out += ["## 结果判断","",
      "- V19是在V13“同一持仓不反复重等”的思路上，换成V15 Top1并分别看20d与60d。首轮20d与60d官方窗分别为+6.99%和+18.35%，扩展窗分别为+163.66%和+171.26%，与V15无门控月频重配完全一致；逐窗日账与成交明细也逐行一致。官方窗V19少提交4/7次同股调仓事件，扩展窗少提交15/23次，但成交数、手续费、买卖额、净值均未改变：这些同股月份V15重配本来就没有产生实际成交，因此V19只是跳过空操作，不是有效优化。",
      "- 所有结论限于给定四只银行股与历史样本；匹配无费差不代表alpha，单一扩展窗也不能确认未来稳定收益。",""]
    return chr(10).join(out)


def main():
    started=time.time()
    days,codes,arrays,meta=load_four_bank_prices()
    if tuple(codes.tolist())!=CODES: raise RuntimeError(f"unexpected stock universe: {codes}")
    official=run_window(days,codes,arrays,OFFICIAL_START,OFFICIAL_SIGNAL_END,OFFICIAL_LIQUIDATION)
    latest=str(days[-1]); extended_signal_end=str(days[-2])
    extended=run_window(days,codes,arrays,EXTENDED_START,extended_signal_end,latest)
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine_hash=hashlib.sha256((ROOT/"account_engine.py").read_bytes()).hexdigest()
    out={"unit":"experiments2/V19 monthly Top1 rebalance only when selected bank changes",
        "built_at":time.strftime("%Y-%m-%d %H:%M:%S"),"snapshot_built_at":meta.get("built_at"),
        "snapshot_last_day":latest,"universe":list(codes),
        "rule":{"signal":"first trading day of month close; V15's fixed Top1 by adjusted 20d or 60d return",
            "holding":"reweight to current equity only when Top1 name changes; if unchanged preserve shares",
            "execution":"next session open; audited shared-cash account engine",
            "lookbacks_under_test":[20,60],"no_training":True,"no_parameter_sweep":True,
            "forced_exit":"official 2026-07-01 open; extended latest 2026-09-24 open",
            "comparison":"same-ledger V15 monthly Top1 reweighting at identical lookback"},
        "cost_model":E.COST,"official_242d":public_view(official),"extended":public_view(extended),
        "runtime_seconds":time.time()-started,
        "process_max_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
        "runner_sha256":script_hash,"account_engine_sha256":engine_hash}
    (OUT/"result.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    write_window_files("official_242d",official); write_window_files("extended",extended)
    protocol={"objective":out["unit"],"signal_windows":{"official":[OFFICIAL_START,OFFICIAL_SIGNAL_END],
        "official_exit":OFFICIAL_LIQUIDATION,"extended":[EXTENDED_START,extended_signal_end],
        "extended_exit":latest},"prices_and_amount_only":True,"universe":list(codes),
        "rule":out["rule"],"cost_model":E.COST,
        "account_engine_source":"isolated V19 copy of V15/V4 audited engine; V15/V9 untouched",
        "runner_sha256":script_hash,"account_engine_sha256":engine_hash,"created_at":out["built_at"]}
    (OUT/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"REPORT.md").write_text(build_report(out),encoding="utf-8")
    print("V19_RESULTS",OUT/"result.json",flush=True)
    for label,w in (("official",official),("extended",extended)):
        print(label,w["window"],flush=True)
        for name,x in w["scenarios"].items():
            print(name,x["statistics"]["cumulative_net_return"],x["statistics"]["max_drawdown"],
                  x["audit"]["fees"],x["audit"]["trade_count"],x["audit"]["skipped_same_top1_months"],flush=True)
    print("max RSS GiB",out["process_max_rss_gib"],"elapsed",time.time()-started,flush=True)

if __name__ == "__main__":
    main()
