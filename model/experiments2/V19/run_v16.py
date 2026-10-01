#!/usr/bin/env python3
"""V16: monthly Top1 held only when its own same-lookback momentum is positive."""
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



def make_schedule(days, codes, arrays, signal_start, signal_end, lookback, positive_gate=False):
    adjusted_close = arrays["close"] * arrays["adj_factor"]
    signals, schedule = [], {}
    date_to_ix = {str(d): i for i, d in enumerate(days)}
    for si in month_first_indices(days):
        d = str(days[si])
        if d < signal_start or d > signal_end or si < lookback or si + 1 >= len(days):
            continue
        old, now = adjusted_close[si-lookback], adjusted_close[si]
        valid = np.isfinite(old) & (old > 0) & np.isfinite(now) & (now > 0)
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            continue
        rets = np.full(len(codes), np.nan, dtype=float)
        rets[valid] = now[valid]/old[valid]-1.0
        ranked = sorted(candidates.tolist(), key=lambda c: (-rets[c], str(codes[c])))
        top = int(ranked[0])
        gate_pass = (float(rets[top]) > 0.0) if positive_gate else True
        ex = int(si+1)
        target = np.asarray([top] if gate_pass else [], dtype=int)
        schedule[ex] = target
        signals.append({
            "signal_date": d, "execution_date": str(days[ex]),
            "signal_index": int(si), "execution_index": ex,
            "lookback_intervals": int(lookback),
            "ranked": [{"code":str(codes[c]),"return":float(rets[c])} for c in ranked],
            "selected_top1": str(codes[top]), "selected_top1_return": float(rets[top]),
            "gate_enabled": bool(positive_gate), "gate_pass": bool(gate_pass),
            "target_codes": [str(codes[c]) for c in target],
        })
    for item in signals:
        si=date_to_ix[item["signal_date"]]; ei=date_to_ix[item["execution_date"]]
        if ei != si+1 or item["signal_index"]-lookback < 0:
            raise RuntimeError("causal schedule check failed")
    return schedule, signals

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


def run_strategy(all_days, codes, all_arrays, start, signal_end, liquidation_day,
                 lookback=None, equal_hold=False, positive_gate=False):
    date_set=set(all_days.tolist())
    for d in (start,signal_end,liquidation_day):
        if d not in date_set: raise RuntimeError(f"required date missing: {d}")
    start_ix=int(np.flatnonzero(all_days==start)[0])
    signal_end_ix=int(np.flatnonzero(all_days==signal_end)[0])
    liq_ix=int(np.flatnonzero(all_days==liquidation_day)[0])
    if liq_ix<=signal_end_ix or start_ix>=signal_end_ix: raise RuntimeError("invalid window")
    days=all_days[start_ix:liq_ix+1]
    arrays={k:v[start_ix:liq_ix+1] for k,v in all_arrays.items()}
    if equal_hold:
        schedule={1:np.arange(len(codes),dtype=int)}
        signals=[{"signal_date":start,"execution_date":str(days[1]),
                  "rule":"initial equal-weight four-bank hold"}]
    else:
        full,signals=make_schedule(all_days,codes,all_arrays,start,signal_end,int(lookback),
                                   positive_gate=positive_gate)
        schedule={int(i-start_ix):names for i,names in full.items()
                  if start_ix<i<=liq_ix}
    result=E.account_sim(days,codes,arrays,schedule)
    stats=E.summarize(result["curve"],E.COST["account_money"])
    curve=result["curve"]
    causal=True
    if not equal_hold:
        causal=all(x["execution_index"]==x["signal_index"]+1
                   and x["signal_date"]<=signal_end
                   and x["signal_index"]-int(lookback)>=0 for x in signals)
    empty_dates=[x["execution_date"] for x in signals
                 if x.get("gate_enabled") and not x.get("gate_pass")]
    curve_by_date={x["date"]:x for x in curve}
    sells={}
    for t in result["trades"]:
        if t["side"]=="sell": sells[t["date"]]=sells.get(t["date"],0)+1
    empty_checks=[]
    for d in empty_dates:
        row=curve_by_date[d]
        empty_checks.append({"execution_date":d,"sell_fills":int(sells.get(d,0)),
            "ending_holdings":int(row["holdings"]),
            "cash_minus_equity":float(row["cash"]-row["equity"]),
            "flat_cash_after_empty_target":bool(row["holdings"]==0 and abs(row["cash"]-row["equity"])<1e-6)})
    audit={
        "ledger_first_day":str(days[0]),"signal_window_last_day":signal_end,
        "final_liquidation_day":str(days[-1]),
        "signal_window_sessions":int(np.count_nonzero((all_days>=start)&(all_days<=signal_end))),
        "ledger_sessions_including_liquidation":len(days),
        "monthly_signal_count":len(signals) if not equal_hold else 0,
        "lookback_intervals":int(lookback) if not equal_hold else None,
        "positive_momentum_gate_enabled":bool(positive_gate),
        "gate_pass_months":int(sum(bool(x.get("gate_pass")) for x in signals)) if positive_gate else None,
        "gate_cash_months":int(sum(not bool(x.get("gate_pass")) for x in signals)) if positive_gate else 0,
        "empty_target_execution_dates":empty_dates,"empty_target_checks":empty_checks,
        "all_empty_targets_left_ledger_flat_in_cash":bool(all(x["flat_cash_after_empty_target"] for x in empty_checks)),
        "sell_fills_on_empty_target_dates":int(sum(x["sell_fills"] for x in empty_checks)),
        "cash_min":float(min(x["cash"] for x in curve)),
        "ending_cash":float(result["ending_cash"]),
        "ending_positions":int(np.count_nonzero(np.asarray(result["ending_shares"])>1e-7)),
        "maximum_simultaneous_holdings":int(max(x["holdings"] for x in curve)),
        "blocked_entries":int(result["blocked_entries"]),"blocked_exits":int(result["blocked_exits"]),
        "trade_count":int(result["ntr"]),"fees":float(result["fees"]),
        "buy_notional":float(result["buy_notional"]),"sell_notional":float(result["sell_notional"]),
        "cash_nonnegative":bool(min(x["cash"] for x in curve)>=-1e-7),
        "final_liquidation_complete":bool(np.count_nonzero(np.asarray(result["ending_shares"])>1e-7)==0),
        "causal_signal_and_t_plus_1_check_pass":bool(causal),
        "max_abs_equity_cash_plus_mark_residual":float(mark_residual(curve,arrays)),
        "ledger_rule":"copied V15/V4 audited cash ledger; lots, amount capacity, limit locks, fees and slippage enabled"
    }
    return {"statistics":stats,"audit":audit,
        "quarterly_returns":split_returns(curve,"quarter",through=signal_end),
        "annual_returns":split_returns(curve,"year"),
        "signals":signals,"curve":curve,"trades":result["trades"]}


def run_window(all_days,codes,all_arrays,start,signal_end,liquidation_day):
    scenarios={}
    for lookback in (20,60):
        label=f"monthly_top1_{lookback}d"
        scenarios[label+"_v15_ungated"]=run_strategy(all_days,codes,all_arrays,start,signal_end,
            liquidation_day,lookback=lookback)
        scenarios[label+"_v16_positive_gate"]=run_strategy(all_days,codes,all_arrays,start,signal_end,
            liquidation_day,lookback=lookback,positive_gate=True)
    scenarios["four_bank_equal_weight_hold"]=run_strategy(all_days,codes,all_arrays,start,
        signal_end,liquidation_day,equal_hold=True)
    signal_days=int(np.count_nonzero((all_days>=start)&(all_days<=signal_end)))
    if start==OFFICIAL_START and signal_days!=242:
        raise RuntimeError(f"official signal window should have 242 sessions, got {signal_days}")
    return {"window":{"start":start,"signal_end":signal_end,"final_liquidation_day":liquidation_day,
        "signal_sessions":signal_days,
        "ledger_sessions_including_liquidation":len(scenarios["monthly_top1_20d_v16_positive_gate"]["curve"])},
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
    keys=("monthly_top1_20d_v15_ungated","monthly_top1_20d_v16_positive_gate",
          "monthly_top1_60d_v15_ungated","monthly_top1_60d_v16_positive_gate",
          "four_bank_equal_weight_hold")
    labels=("20日Top1无门控","20日Top1正动量门控","60日Top1无门控",
            "60日Top1正动量门控","四股等权持有")
    out=["# V16：月频 Top1 同回看动量正值门控","",
        f"- 构建时间：{result['built_at']}；价格快照截至 {result['snapshot_last_day']}；仅价格/成交额，无训练、无GPU、无门槛扫描。",
        "- 每月首个交易日收盘先按20日或60日复权动量选Top1；该Top1的同周期动量严格大于0则T+1开盘持有一只，否则向同一账本提交空目标并转为现金。两个回看期固定作对照。",
        "- V15无门控Top1与V16门控Top1在同一V4审计账本重放；官方窗242个信号交易日后于2026-07-01开盘清仓，扩展窗于2026-09-24开盘清仓。",
        "- 匹配无费收益只衡量同一实际持仓路径的费用/执行摩擦差，不代表选股alpha。",""]
    for title,wkey in (("官方242日信号窗 + 7/1退出","official_242d"),
                       ("扩展窗 + 9/24退出","extended")):
        w=result[wkey]
        out += [f"## {title}","",
          f"- 信号区间 {w['window']['start']} 至 {w['window']['signal_end']}（{w['window']['signal_sessions']}个交易日）；最终开盘清仓 {w['window']['final_liquidation_day']}；账本覆盖 {w['window']['ledger_sessions_including_liquidation']}日。","",
          "| 策略 | 净收益 | 最大回撤 | 平均敞口 | 费用 | 成交 | 同持仓无费 | 摩擦差 | 门控现金月 | 空目标卖出成交 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k,label in zip(keys,labels):
            x=w["scenarios"][k]; st=x["statistics"]; a=x["audit"]
            months="—" if not a["positive_momentum_gate_enabled"] else f"{a['gate_cash_months']}/{a['monthly_signal_count']}"
            out.append(f"| {label} | {st['cumulative_net_return']:.2%} | {st['max_drawdown']:.2%} | {st['avg_exposure']:.1%} | {a['fees']:.2f} | {a['trade_count']} | {st['matched_benchmark_return']:.2%} | {st['matched_cumulative_excess']:.2%} | {months} | {a['sell_fills_on_empty_target_dates']} |")
        out += ["","### 季度净收益（不含最终清仓日）","",
          "| 季度 | 20日无门控 | 20日门控 | 60日无门控 | 60日门控 | 四股等权 |",
          "|---|---:|---:|---:|---:|---:|"]
        qmaps={k:{r["quarter"]:r for r in w["scenarios"][k]["quarterly_returns"]} for k in keys}
        for q in sorted(set().union(*(set(x) for x in qmaps.values()))):
            out.append(f"| {q} | "+" | ".join(f"{qmaps[k][q]['net_return']:.2%}" for k in keys)+" |")
        out += ["","### 年度净收益（含窗口退出日）","",
          "| 年份 | 20日无门控 | 20日门控 | 60日无门控 | 60日门控 | 四股等权 |",
          "|---:|---:|---:|---:|---:|---:|"]
        ymaps={k:{r["year"]:r for r in w["scenarios"][k]["annual_returns"]} for k in keys}
        for y in sorted(set().union(*(set(x) for x in ymaps.values()))):
            out.append(f"| {y} | "+" | ".join(f"{ymaps[k][y]['net_return']:.2%}" for k in keys)+" |")
        out += ["","### 门控与清仓审计",""]
        for k,label in zip(keys[:4],labels[:4]):
            a=w["scenarios"][k]["audit"]; extra=""
            if a["positive_momentum_gate_enabled"]:
                extra=(f"；门控通过 {a['gate_pass_months']}/{a['monthly_signal_count']} 月，空目标 {a['gate_cash_months']} 月，"
                       f"空目标日卖出成交 {a['sell_fills_on_empty_target_dates']} 笔，空目标后全部为空仓现金 {a['all_empty_targets_left_ledger_flat_in_cash']}")
            out.append(f"- {label}：最低现金 {a['cash_min']:.4f}，最终持仓 {a['ending_positions']}，阻塞买/卖 {a['blocked_entries']}/{a['blocked_exits']}，最终清仓 {a['final_liquidation_complete']}，最大账本残差 {a['max_abs_equity_cash_plus_mark_residual']:.3g}{extra}。")
        out.append("")
    out += ["## 与失败的 V8 四股均值门控的区别","",
      "- V8以四股60日收益的等权均值是否大于0决定持有Top2或清仓；其官方生命周期净收益为-9.55%，无门控V4 Top2为+11.58%，差-21.14个百分点。",
      "- V16逐月先排序取同回看期Top1，仅检查该Top1自身动量是否大于0。60日版只在四股60日收益全部非正时清仓，因为Top1就是四股最大值；V8则在均值非正时清仓，即使其中个股为正也可能空仓。V16持一股而非V8的Top2，20日版还把信号回看期缩短为20日。",
      "- 因而即使V16优于V8，也同时改变了选股集中度、持股数、门控统计量和现金暴露，不能把差异归结为单一门控因果效应。",
      "- 历史回测仅用于候选比较；净值和同持仓摩擦指标不构成未来稳定收益或可持续alpha的证明。",""]
    return chr(10).join(out)

def main():
    started = time.time()
    days, codes, arrays, meta = load_four_bank_prices()
    if tuple(codes.tolist()) != CODES:
        raise RuntimeError(f"unexpected stock universe: {codes}")
    official = run_window(days, codes, arrays, OFFICIAL_START, OFFICIAL_SIGNAL_END, OFFICIAL_LIQUIDATION)
    latest = str(days[-1])
    extended_signal_end = str(days[-2])
    extended = run_window(days, codes, arrays, EXTENDED_START, extended_signal_end, latest)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine_hash = hashlib.sha256((ROOT / "account_engine.py").read_bytes()).hexdigest()
    out = {
        "unit": "experiments2/V16 monthly Top1 same-lookback positive momentum gate",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_built_at": meta.get("built_at"),
        "snapshot_last_day": latest,
        "universe": list(codes),
        "rule": {
            "signal": "first observed trading session of each calendar month at close",
            "factor": "adjusted close(T) / adjusted close(T-lookback) - 1",
            "ranking": "descending cross-sectional total return; equal-weight fixed Top1",
            "execution": "next trading session open",
            "lookbacks_under_test": [20, 60],
            "positive_momentum_gate": "hold selected Top1 only when its own same-lookback return is strictly >0; else empty target/cash",
            "same_ledger_ungated_comparator": "V15 monthly Top1 for each lookback",
            "no_training": True, "no_parameter_sweep": True,
            "final_liquidation": "last day of each ledger window, forced at open by audited account engine",
        },
        "cost_model": E.COST,
        "official_242d": public_view(official),
        "extended": public_view(extended),
        "runtime_seconds": time.time() - started,
        "process_max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
        "runner_sha256": script_hash,
        "account_engine_sha256": engine_hash,
    }
    (OUT / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_window_files("official_242d", official)
    write_window_files("extended", extended)
    protocol = {
        "objective": out["unit"],
        "gate": out["rule"]["positive_momentum_gate"],
        "official_signal_window": [OFFICIAL_START, OFFICIAL_SIGNAL_END],
        "official_signal_sessions_expected": 242,
        "official_final_liquidation_day": OFFICIAL_LIQUIDATION,
        "extended_signal_window": [EXTENDED_START, extended_signal_end],
        "extended_final_liquidation_day": latest,
        "prices_and_amount_only": True,
        "data_root": str(DATA_ROOT),
        "universe": list(codes),
        "rule": out["rule"],
        "cost_model": E.COST,
        "account_engine_source": "isolated V16 copy of V15 audited account_engine.py; V15 and V9 untouched",
        "runner_sha256": script_hash,
        "account_engine_sha256": engine_hash,
        "created_at": out["built_at"],
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "REPORT.md").write_text(build_report(out), encoding="utf-8")
    print("V16_RESULTS", OUT / "result.json", flush=True)
    for wname, w in (("official", official), ("extended", extended)):
        print(wname, flush=True)
        for name, p in w["scenarios"].items():
            print(name, p["statistics"]["cumulative_net_return"], p["statistics"]["max_drawdown"],
                  p["audit"]["fees"], p["audit"]["trade_count"], flush=True)
        print("window", w["window"], flush=True)
    print("max RSS GiB", out["process_max_rss_gib"], "elapsed", time.time() - started, flush=True)


if __name__ == "__main__":
    main()

