#!/usr/bin/env python3
"""V9: fixed monthly 20-session momentum Top2, with same-ledger comparisons."""
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
TOP_K = 2
LOOKBACKS = {"monthly_top2_20d": 20, "monthly_top2_60d": 60}


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


def make_schedule(days, codes, arrays, signal_start, signal_end, lookback):
    adjusted_close = arrays["close"] * arrays["adj_factor"]
    signals, schedule = [], {}
    date_to_ix = {str(d): i for i, d in enumerate(days)}
    for si in month_first_indices(days):
        d = str(days[si])
        if d < signal_start or d > signal_end or si < lookback or si + 1 >= len(days):
            continue
        old = adjusted_close[si - lookback]
        now = adjusted_close[si]
        valid = np.isfinite(old) & (old > 0) & np.isfinite(now) & (now > 0)
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            continue
        rets = np.full(len(codes), np.nan, dtype=float)
        rets[valid] = now[valid] / old[valid] - 1.0
        ranked = sorted(candidates.tolist(), key=lambda c: (-rets[c], str(codes[c])))
        chosen = ranked[:TOP_K]
        ex = int(si + 1)
        schedule[ex] = np.asarray(chosen, dtype=int)
        signals.append({
            "signal_date": d,
            "execution_date": str(days[ex]),
            "signal_index": int(si),
            "execution_index": ex,
            "lookback_intervals": int(lookback),
            "ranked": [{"code": str(codes[c]), "return": float(rets[c])} for c in ranked],
            "selected": [str(codes[c]) for c in chosen],
        })
    for item in signals:
        si = date_to_ix[item["signal_date"]]
        ei = date_to_ix[item["execution_date"]]
        if ei != si + 1 or item["signal_index"] - lookback < 0:
            raise RuntimeError("causal schedule check failed")
    return schedule, signals



def make_rank_fusion_schedule(days, codes, arrays, signal_start, signal_end):
    """Monthly 20d/60d average ordinal rank; only signal-day close data is used."""
    adjusted_close = arrays["close"] * arrays["adj_factor"]
    signals, schedule = [], {}
    date_to_ix = {str(d): i for i, d in enumerate(days)}
    for si in month_first_indices(days):
        d = str(days[si])
        if d < signal_start or d > signal_end or si < 60 or si + 1 >= len(days):
            continue
        ex = int(si + 1)
        if str(days[ex]) > signal_end:
            continue
        ret20 = np.full(len(codes), np.nan, dtype=float)
        ret60 = np.full(len(codes), np.nan, dtype=float)
        for horizon, out in ((20, ret20), (60, ret60)):
            old = adjusted_close[si-horizon]
            now = adjusted_close[si]
            valid = np.isfinite(old) & (old > 0) & np.isfinite(now) & (now > 0)
            out[valid] = now[valid] / old[valid] - 1.0
        candidates = np.flatnonzero(np.isfinite(ret20) & np.isfinite(ret60))
        if len(candidates) < TOP_K:
            raise RuntimeError(f"fewer than Top2 banks with both momentum horizons at {d}")
        ranked20 = sorted(candidates.tolist(), key=lambda c: (-ret20[c], str(codes[c])))
        ranked60 = sorted(candidates.tolist(), key=lambda c: (-ret60[c], str(codes[c])))
        rank20 = {c: rank+1 for rank,c in enumerate(ranked20)}
        rank60 = {c: rank+1 for rank,c in enumerate(ranked60)}
        avg_rank = {c: (rank20[c] + rank60[c]) / 2.0 for c in candidates}
        ranked = sorted(candidates.tolist(), key=lambda c: (avg_rank[c], str(codes[c])))
        chosen = ranked[:TOP_K]
        schedule[ex] = np.asarray(chosen, dtype=int)
        signals.append({
            "signal_date": d, "execution_date": str(days[ex]),
            "signal_index": int(si), "execution_index": ex,
            "momentum_20d": {str(codes[c]): float(ret20[c]) for c in candidates},
            "momentum_60d": {str(codes[c]): float(ret60[c]) for c in candidates},
            "rank_20d": {str(codes[c]): int(rank20[c]) for c in candidates},
            "rank_60d": {str(codes[c]): int(rank60[c]) for c in candidates},
            "average_rank": {str(codes[c]): float(avg_rank[c]) for c in candidates},
            "ranked_by_average_rank": [{"code": str(codes[c]), "average_rank": float(avg_rank[c]),
                "rank_20d": int(rank20[c]), "rank_60d": int(rank60[c])} for c in ranked],
            "selected": [str(codes[c]) for c in chosen],
            "rank_fusion": "arithmetic mean of descending cross-sectional ordinal ranks; equal weight 20d and 60d"
        })
    for item in signals:
        si = date_to_ix[item["signal_date"]]
        ei = date_to_ix[item["execution_date"]]
        if ei != si + 1 or si - 60 < 0 or item["signal_date"] > signal_end:
            raise RuntimeError("rank-fusion causal schedule check failed")
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


def split_returns(curve, freq):
    df = pd.DataFrame(curve)
    if freq == "quarter":
        keys = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="Q").astype(str)
        field = "quarter"
    else:
        keys = pd.to_datetime(df["date"]).dt.year.astype(str).to_numpy()
        field = "year"
    out = []
    for key in sorted(set(keys)):
        g = df.loc[np.asarray(keys) == key]
        out.append({field: str(key), "ledger_sessions": int(len(g)),
                    "net_return": float(np.prod(1.0 + g["daily_return"].to_numpy(float)) - 1.0),
                    "same_hold_no_fee_return": float(np.prod(1.0 + g["matched_return"].to_numpy(float)) - 1.0)})
        out[-1]["cost_execution_friction_gap"] = out[-1]["net_return"] - out[-1]["same_hold_no_fee_return"]
    return out


def run_strategy(all_days, codes, all_arrays, start, signal_end, liquidation_day,
                 lookback=None, equal_hold=False, fused=False):
    date_set = set(all_days.tolist())
    for d in (start, signal_end, liquidation_day):
        if d not in date_set:
            raise RuntimeError(f"required date missing: {d}")
    start_ix = int(np.flatnonzero(all_days == start)[0])
    signal_end_ix = int(np.flatnonzero(all_days == signal_end)[0])
    liq_ix = int(np.flatnonzero(all_days == liquidation_day)[0])
    if liq_ix <= signal_end_ix or start_ix >= signal_end_ix:
        raise RuntimeError("invalid signal/liquidation window")
    days = all_days[start_ix:liq_ix + 1]
    arrays = {k: v[start_ix:liq_ix + 1] for k, v in all_arrays.items()}
    if equal_hold:
        schedule = {1: np.arange(len(codes), dtype=int)}
        signals = [{"signal_date": start, "execution_date": str(days[1]),
                    "rule": "initial equal-weight four-bank hold; no rebalance until final liquidation"}]
    elif fused:
        schedule_all, signals = make_rank_fusion_schedule(all_days, codes, all_arrays, start, signal_end)
        schedule = {int(gix - start_ix): names for gix, names in schedule_all.items()
                    if start_ix < gix <= liq_ix}
    else:
        schedule_all, signals = make_schedule(all_days, codes, all_arrays, start, signal_end, int(lookback))
        schedule = {int(gix - start_ix): names for gix, names in schedule_all.items()
                    if start_ix < gix <= liq_ix}
    result = E.account_sim(days, codes, arrays, schedule)
    stats = E.summarize(result["curve"], E.COST["account_money"])
    curve = result["curve"]
    causal = True
    if not equal_hold:
        required_lookback = 60 if fused else int(lookback)
        for sig in signals:
            causal &= (sig["execution_index"] == sig["signal_index"] + 1
                       and sig["signal_date"] <= signal_end
                       and sig["signal_index"] - required_lookback >= 0)
    signal_curve = curve[:-1]
    exitrow = curve[-1]
    exit_return = {"date": str(exitrow["date"]),
        "net_return_including_open_liquidation_cost": float(exitrow["daily_return"]),
        "same_hold_no_fee_return": float(exitrow["matched_return"]),
        "cost_execution_friction_gap": float(exitrow["daily_return"]-exitrow["matched_return"]),
        "interpretation": "final liquidation-day daily ledger return; excluded from signal-period quarter/year tables"}
    audit = {
        "ledger_first_day": str(days[0]), "signal_window_last_day": signal_end,
        "final_liquidation_day": str(days[-1]),
        "signal_window_sessions": int(np.count_nonzero((all_days >= start) & (all_days <= signal_end))),
        "ledger_sessions_including_liquidation": int(len(days)),
        "signal_period_quarterly_annual_exclude_final_liquidation_row": True,
        "exit_day_return_reported_separately": True,
        "monthly_signal_count": int(len(signals)) if not equal_hold else 0,
        "rebalance_execution_dates": [z["execution_date"] for z in signals if "execution_date" in z] if not equal_hold else [str(days[1])],
        "cash_min": float(min(z["cash"] for z in curve)), "ending_cash": float(result["ending_cash"]),
        "ending_positions": int(np.count_nonzero(np.asarray(result["ending_shares"]) > 1e-7)),
        "maximum_simultaneous_holdings": int(max(z["holdings"] for z in curve)),
        "blocked_entries": int(result["blocked_entries"]), "blocked_exits": int(result["blocked_exits"]),
        "trade_count": int(result["ntr"]), "fees": float(result["fees"]),
        "buy_notional": float(result["buy_notional"]), "sell_notional": float(result["sell_notional"]),
        "cash_nonnegative": bool(min(z["cash"] for z in curve) >= -1e-7),
        "final_liquidation_complete": bool(np.count_nonzero(np.asarray(result["ending_shares"]) > 1e-7) == 0),
        "causal_signal_and_t_plus_1_check_pass": bool(causal),
        "max_abs_equity_cash_plus_mark_residual": float(mark_residual(curve, arrays)),
        "ledger_rule": "same V4 account_engine.py; cash account, 100-share lots, signal-day amount capacity, one-price limit locks, 3bp slippage, commissions/transfer/sell stamp fees",
    }
    return {"statistics": stats, "audit": audit,
        "quarterly_returns": split_returns(signal_curve, "quarter"),
        "annual_returns": split_returns(signal_curve, "year"),
        "exit_day_return": exit_return, "signals": signals,
        "curve": curve, "trades": result["trades"]}


def run_window(all_days, codes, all_arrays, start, signal_end, liquidation_day):
    scenarios = {
        "rank_fusion_top2": run_strategy(all_days,codes,all_arrays,start,signal_end,liquidation_day,fused=True),
        "monthly_top2_60d": run_strategy(all_days,codes,all_arrays,start,signal_end,liquidation_day,lookback=60),
        "four_bank_equal_weight_hold": run_strategy(all_days,codes,all_arrays,start,signal_end,liquidation_day,equal_hold=True),
    }
    signal_days = int(np.count_nonzero((all_days >= start) & (all_days <= signal_end)))
    if start == OFFICIAL_START and signal_days != 242:
        raise RuntimeError(f"official signal window should have 242 sessions, got {signal_days}")
    return {"window":{"start":start,"signal_end":signal_end,"final_liquidation_day":liquidation_day,
        "signal_sessions":signal_days,"ledger_sessions_including_liquidation":len(scenarios["rank_fusion_top2"]["curve"])},
        "scenarios":scenarios}


def public_view(window):
    return {"window":window["window"],"scenarios":{
        name:{k:v for k,v in payload.items() if k not in ("curve","trades")}
        for name,payload in window["scenarios"].items()}}


def write_window_files(name, window):
    qrows=[]; arows=[]; exits=[]
    for scenario,payload in window["scenarios"].items():
        stem=f"{name}_{scenario}"
        pd.DataFrame(payload["curve"]).drop(columns=["shares"]).to_csv(OUT/f"{stem}_daily.csv",index=False)
        pd.DataFrame(payload["trades"]).to_csv(OUT/f"{stem}_trades.csv",index=False)
        pd.DataFrame(payload["signals"]).to_json(OUT/f"{stem}_signals.json",orient="records",force_ascii=False,indent=2)
        qrows += [{"window":name,"scenario":scenario,**z} for z in payload["quarterly_returns"]]
        arows += [{"window":name,"scenario":scenario,**z} for z in payload["annual_returns"]]
        exits.append({"window":name,"scenario":scenario,**payload["exit_day_return"]})
    pd.DataFrame(qrows).to_csv(OUT/f"{name}_quarterly.csv",index=False)
    pd.DataFrame(arows).to_csv(OUT/f"{name}_annual.csv",index=False)
    pd.DataFrame(exits).to_csv(OUT/f"{name}_exit_day_returns.csv",index=False)
    return qrows,arows,exits


def build_report(result):
    out=["# V17：20/60 日动量平均名次 Top2", "",
      f"- 价格快照截至 {result['snapshot_last_day']}；只读取四银行价格和成交额，没有训练模型或扫描参数。",
      "- 固定规则：月初收盘分别按 20 日、60 日复权动量从高到低横截面排名（最好为 1），两种名次等权平均，选平均名次最好的两只等权持有；下一交易日开盘执行。",
      "- 与月频 60 日动量 Top2、四股等权持有使用同一份 V4 账户引擎，统一费用、整手、成交额上限、涨跌停锁单和滑点。最终退出日为指定日开盘。季度/年度表排除最后的退出日，退出日账本收益单独列出。",""]
    for label,key in (("官方同窗","official_242d"),("扩展窗","extended")):
        w=result[key]
        out += [f"## {label}","",
          f"- 信号期：{w['window']['start']} 至 {w['window']['signal_end']}（{w['window']['signal_sessions']} 个交易日）；退出日：{w['window']['final_liquidation_day']}。","",
          "| 策略 | 全生命周期净收益 | 年化 | 最大回撤 | 平均敞口 | 费用 | 成交数 | 同持仓无费 | 摩擦差 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for sk,title in (("rank_fusion_top2","20/60 名次融合 Top2"),("monthly_top2_60d","月频 60 日 Top2"),("four_bank_equal_weight_hold","四股等权持有")):
            z=w["scenarios"][sk]; st,au=z["statistics"],z["audit"]
            out.append(f"| {title} | {st['cumulative_net_return']:.2%} | {st['annualized_return']:.2%} | {st['max_drawdown']:.2%} | {st['avg_exposure']:.1%} | ¥{au['fees']:.2f} | {au['trade_count']} | {st['matched_benchmark_return']:.2%} | {st['matched_cumulative_excess']:.2%} |")
        out += ["","### 信号期季度净收益（不含最终退出日）","",
          "| 季度 | 融合 Top2 | 60 日 Top2 | 四股等权 |","|---|---:|---:|---:|"]
        maps={name:{r['quarter']:r['net_return'] for r in w['scenarios'][name]['quarterly_returns']}
              for name in ('rank_fusion_top2','monthly_top2_60d','four_bank_equal_weight_hold')}
        for q in sorted(set().union(*(set(v) for v in maps.values()))):
            out.append(f"| {q} | {maps['rank_fusion_top2'][q]:.2%} | {maps['monthly_top2_60d'][q]:.2%} | {maps['four_bank_equal_weight_hold'][q]:.2%} |")
        out += ["","### 信号期年度净收益（不含最终退出日）","",
          "| 年份 | 融合 Top2 | 60 日 Top2 | 四股等权 |","|---:|---:|---:|---:|"]
        maps={name:{r['year']:r['net_return'] for r in w['scenarios'][name]['annual_returns']}
              for name in ('rank_fusion_top2','monthly_top2_60d','four_bank_equal_weight_hold')}
        for y in sorted(set().union(*(set(v) for v in maps.values()))):
            out.append(f"| {y} | {maps['rank_fusion_top2'][y]:.2%} | {maps['monthly_top2_60d'][y]:.2%} | {maps['four_bank_equal_weight_hold'][y]:.2%} |")
        out += ["","### 最终退出日收益（不纳入上述季度/年度表）","",
          "| 策略 | 退出日 | 净收益（含开盘清仓成本） | 同持仓无费 | 摩擦差 |","|---|---|---:|---:|---:|"]
        for sk,title in (("rank_fusion_top2","20/60 名次融合 Top2"),("monthly_top2_60d","月频 60 日 Top2"),("four_bank_equal_weight_hold","四股等权持有")):
            z=w['scenarios'][sk]['exit_day_return']
            out.append(f"| {title} | {z['date']} | {z['net_return_including_open_liquidation_cost']:.2%} | {z['same_hold_no_fee_return']:.2%} | {z['cost_execution_friction_gap']:.2%} |")
        out += ["","### 因果与账本审计",""]
        for sk,title in (("rank_fusion_top2","融合 Top2"),("monthly_top2_60d","60 日 Top2"),("four_bank_equal_weight_hold","四股等权")):
            a=w['scenarios'][sk]['audit']
            out.append(f"- {title}：现金非负 {a['cash_nonnegative']}；最终空仓 {a['final_liquidation_complete']}；信号到 T+1 因果检查 {a['causal_signal_and_t_plus_1_check_pass']}；季度/年度排除清仓行 {a['signal_period_quarterly_annual_exclude_final_liquidation_row']}；最大账本残差 {a['max_abs_equity_cash_plus_mark_residual']:.3g}；阻塞买入/卖出 {a['blocked_entries']}/{a['blocked_exits']}。")
        out.append("")
    out += ["- 解释边界：匹配无费回报只衡量实际持仓路径上的交易摩擦，不代表选股 alpha；候选历史表现不能证明未来收益稳定。",""]
    return "\n".join(out)


def main():
    started=time.time()
    days,codes,arrays,meta=load_four_bank_prices()
    if tuple(codes.tolist())!=CODES: raise RuntimeError(f"unexpected stock universe: {codes}")
    official=run_window(days,codes,arrays,OFFICIAL_START,OFFICIAL_SIGNAL_END,OFFICIAL_LIQUIDATION)
    latest=str(days[-1]); extended_signal_end=str(days[-2])
    extended=run_window(days,codes,arrays,EXTENDED_START,extended_signal_end,latest)
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine_hash=hashlib.sha256((ROOT/"account_engine.py").read_bytes()).hexdigest()
    out={"unit":"experiments2/V17 monthly 20d+60d average-rank Top2",
      "built_at":time.strftime("%Y-%m-%d %H:%M:%S"),"snapshot_built_at":meta.get("built_at"),
      "snapshot_last_day":latest,"universe":list(codes),
      "rule":{"signal":"first observed trading session of each calendar month at close",
        "factors":["adjusted close(T)/adjusted close(T-20 sessions)-1","adjusted close(T)/adjusted close(T-60 sessions)-1"],
        "ranking":"rank each momentum descending cross-sectionally (best=1), arithmetic mean of the two ordinal ranks, choose lowest-average-rank Top2; equal-weight holdings",
        "execution":"next trading session open","top_k":2,"horizons_sessions":[20,60],"horizon_rank_weights":[0.5,0.5],
        "no_training":True,"no_parameter_sweep":True,
        "final_liquidation":"official 2026-07-01 open; extended latest snapshot session open; forced by audited account engine"},
      "cost_model":E.COST,"official_242d":public_view(official),"extended":public_view(extended),
      "runtime_seconds":time.time()-started,
      "process_max_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
      "runner_sha256":script_hash,"account_engine_sha256":engine_hash}
    (OUT/"result.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    q1,a1,x1=write_window_files("official_242d",official)
    q2,a2,x2=write_window_files("extended",extended)
    pd.DataFrame(q1+q2).to_csv(OUT/"quarterly.csv",index=False)
    pd.DataFrame(a1+a2).to_csv(OUT/"annual.csv",index=False)
    pd.DataFrame(x1+x2).to_csv(OUT/"exit_day_returns.csv",index=False)
    protocol={"objective":out["unit"],"official_signal_window":[OFFICIAL_START,OFFICIAL_SIGNAL_END],
      "official_signal_sessions_expected":242,"official_final_liquidation_day":OFFICIAL_LIQUIDATION,
      "extended_signal_window":[EXTENDED_START,extended_signal_end],"extended_final_liquidation_day":latest,
      "prices_and_amount_only":True,"data_root":str(DATA_ROOT),"universe":list(codes),"rule":out["rule"],
      "cost_model":E.COST,"account_engine_source":"copied V4 audited account_engine.py; V4 and V9 left unchanged",
      "runner_sha256":script_hash,"account_engine_sha256":engine_hash,"created_at":out["built_at"]}
    (OUT/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"REPORT.md").write_text(build_report(out),encoding="utf-8")
    print("V17_RESULTS",OUT/"result.json",flush=True)
    for window_name,window in (("official",official),("extended",extended)):
        print(window_name,window["window"],flush=True)
        for strategy,payload in window["scenarios"].items():
            st=payload["statistics"]; au=payload["audit"]
            print(strategy,"net",st["cumulative_net_return"],"mdd",st["max_drawdown"],
                "fees",au["fees"],"trades",au["trade_count"],"exit",payload["exit_day_return"],flush=True)
            print("audit",strategy,au["cash_nonnegative"],au["final_liquidation_complete"],
                au["causal_signal_and_t_plus_1_check_pass"],au["max_abs_equity_cash_plus_mark_residual"],flush=True)
    print("max RSS GiB",out["process_max_rss_gib"],"elapsed",time.time()-started,flush=True)


if __name__ == "__main__":
    main()
