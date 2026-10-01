#!/usr/bin/env python3
"""V15: fixed monthly 20-session momentum Top1, with same-ledger comparisons."""
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


def run_strategy(all_days, codes, all_arrays, start, signal_end, liquidation_day, lookback=None, equal_hold=False):
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
    else:
        schedule_all, signals = make_schedule(all_days, codes, all_arrays, start, signal_end, int(lookback))
        schedule = {int(gix - start_ix): names for gix, names in schedule_all.items()
                    if start_ix < gix <= liq_ix}
    result = E.account_sim(days, codes, arrays, schedule)
    stats = E.summarize(result["curve"], E.COST["account_money"])
    curve = result["curve"]
    causal = True
    if not equal_hold:
        for s in signals:
            causal &= (s["execution_index"] == s["signal_index"] + 1
                       and s["signal_date"] <= signal_end
                       and s["signal_index"] - int(lookback) >= 0)
    audit = {
        "ledger_first_day": str(days[0]),
        "signal_window_last_day": signal_end,
        "final_liquidation_day": str(days[-1]),
        "signal_window_sessions": int(np.count_nonzero((all_days >= start) & (all_days <= signal_end))),
        "ledger_sessions_including_liquidation": int(len(days)),
        "monthly_signal_count": int(len(signals)) if not equal_hold else 0,
        "rebalance_execution_dates": [s["execution_date"] for s in signals if "execution_date" in s] if not equal_hold else [str(days[1])],
        "cash_min": float(min(r["cash"] for r in curve)),
        "ending_cash": float(result["ending_cash"]),
        "ending_positions": int(np.count_nonzero(np.asarray(result["ending_shares"]) > 1e-7)),
        "maximum_simultaneous_holdings": int(max(r["holdings"] for r in curve)),
        "blocked_entries": int(result["blocked_entries"]),
        "blocked_exits": int(result["blocked_exits"]),
        "trade_count": int(result["ntr"]),
        "fees": float(result["fees"]),
        "buy_notional": float(result["buy_notional"]),
        "sell_notional": float(result["sell_notional"]),
        "cash_nonnegative": bool(min(r["cash"] for r in curve) >= -1e-7),
        "final_liquidation_complete": bool(np.count_nonzero(np.asarray(result["ending_shares"]) > 1e-7) == 0),
        "causal_signal_and_t_plus_1_check_pass": bool(causal),
        "max_abs_equity_cash_plus_mark_residual": float(mark_residual(curve, arrays)),
        "ledger_rule": "same V4 account_engine.py; cash account, 100-share lots, signal-day amount capacity, one-price limit locks, 3bp slippage, commissions/transfer/sell stamp fees",
    }
    return {
        "statistics": stats,
        "audit": audit,
        "quarterly_returns": split_returns(curve[:-1], "quarter"),
        "annual_returns": split_returns(curve[:-1], "year"),
        "exit_day_net_return": float(curve[-1]["daily_return"]),
        "signals": signals,
        "curve": curve,
        "trades": result["trades"],
    }


def run_window(all_days, codes, all_arrays, start, signal_end, liquidation_day):
    scenarios = {}
    for name, lookback in LOOKBACKS.items():
        scenarios[name] = run_strategy(all_days, codes, all_arrays, start, signal_end, liquidation_day, lookback=lookback)
    scenarios["four_bank_equal_weight_hold"] = run_strategy(
        all_days, codes, all_arrays, start, signal_end, liquidation_day, equal_hold=True)
    active_names = ("monthly_top1_20d", "monthly_top1_60d", "four_bank_equal_weight_hold")
    signal_days = int(np.count_nonzero((all_days >= start) & (all_days <= signal_end)))
    if start == OFFICIAL_START and signal_days != 242:
        raise RuntimeError(f"official signal window should have 242 sessions, got {signal_days}")
    return {
        "window": {"start": start, "signal_end": signal_end, "final_liquidation_day": liquidation_day,
                   "signal_sessions": signal_days,
                   "ledger_sessions_including_liquidation": int(len(scenarios["monthly_top1_20d"]["curve"]))},
        "scenarios": {k: scenarios[k] for k in active_names},
    }


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
    out = ["# V15：四大行月频 20 日动量 Top1", "",
           f"- 构建时间：{result['built_at']}；四股价格快照截至 {result['snapshot_last_day']}。",
           "- V15 固定规则：每月首个交易日收盘计算复权收盘价相对 20 个交易间隔前的涨幅，等权持有排名第一；下一交易日开盘执行。没有训练模型或扫描参数。",
           "- 20 日规则与 60 日月频 Top1、四股等权持有均由同一份 V4 审计账户引擎独立重放，交易费、整手、成交额上限、涨跌停锁单和滑点一致。",
           "- 交易收益包含窗口最后信号日之后的最终交易日，并于最终交易日开盘强制清仓。严格匹配无费回报只用于估计同一实际仓位路径的成本/执行摩擦差，不代表选股 alpha。", ""]
    for label, key in (("官方同窗", "official_242d"), ("扩展窗", "extended")):
        w = result[key]
        out += [f"## {label}", "",
                f"- 信号/评估区间：{w['window']['start']} 至 {w['window']['signal_end']}；信号交易日 {w['window']['signal_sessions']} 天；最终清仓日 {w['window']['final_liquidation_day']}；账本共 {w['window']['ledger_sessions_including_liquidation']} 天。", "",
                "| 策略 | 净收益 | 年化 | 最大回撤 | 平均敞口 | 费用 | 成交数 | 同持仓无费 | 摩擦差 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for key_name, title in (("monthly_top1_20d", "月频 20 日 Top1"),
                                ("monthly_top1_60d", "月频 60 日 Top1"),
                                ("four_bank_equal_weight_hold", "四股等权持有")):
            p = w["scenarios"][key_name]
            st, au = p["statistics"], p["audit"]
            out.append(f"| {title} | {st['cumulative_net_return']:.2%} | {st['annualized_return']:.2%} | {st['max_drawdown']:.2%} | {st['avg_exposure']:.1%} | {au['fees']:.2f} | {au['trade_count']} | {st['matched_benchmark_return']:.2%} | {st['matched_cumulative_excess']:.2%} |")
        out += ["", "### 季度净收益", "",
                "| 季度 | 20日Top1 | 60日Top1 | 四股等权 |",
                "|---|---:|---:|---:|"]
        qmaps = {k: {r["quarter"]: r for r in w["scenarios"][k]["quarterly_returns"]}
                 for k in ("monthly_top1_20d", "monthly_top1_60d", "four_bank_equal_weight_hold")}
        for q in sorted(set().union(*(set(x) for x in qmaps.values()))):
            out.append(f"| {q} | {qmaps['monthly_top1_20d'][q]['net_return']:.2%} | {qmaps['monthly_top1_60d'][q]['net_return']:.2%} | {qmaps['four_bank_equal_weight_hold'][q]['net_return']:.2%} |")
        out += ["", "### 年度净收益", "",
                "| 年份 | 20日Top1 | 60日Top1 | 四股等权 |",
                "|---:|---:|---:|---:|"]
        ymaps = {k: {r["year"]: r for r in w["scenarios"][k]["annual_returns"]}
                 for k in ("monthly_top1_20d", "monthly_top1_60d", "four_bank_equal_weight_hold")}
        for y in sorted(set().union(*(set(x) for x in ymaps.values()))):
            out.append(f"| {y} | {ymaps['monthly_top1_20d'][y]['net_return']:.2%} | {ymaps['monthly_top1_60d'][y]['net_return']:.2%} | {ymaps['four_bank_equal_weight_hold'][y]['net_return']:.2%} |")
        out += ["", "### 最终退出日收益（单独列示，不计入信号窗季度/年度收益）", "",
                "| 策略 | 退出日 | 当日净收益 |", "|---|---|---:|"]
        for key_name, title in (("monthly_top1_20d", "月频 20 日 Top1"),
                                ("monthly_top1_60d", "月频 60 日 Top1"),
                                ("four_bank_equal_weight_hold", "四股等权持有")):
            p = w["scenarios"][key_name]
            out.append(f"| {title} | {w['window']['final_liquidation_day']} | {p['exit_day_net_return']:.2%} |")
        out += ["", "### 20日 Top1 月度信号与执行日期", "",
                "| 信号日（收盘） | 下一交易日执行 | Top1 |", "|---|---|---|"]
        for s in w["scenarios"]["monthly_top1_20d"]["signals"]:
            out.append(f"| {s['signal_date']} | {s['execution_date']} | {', '.join(s['selected'])} |")
        out += ["", "### 账本与因果审计", ""]
        for key_name, title in (("monthly_top1_20d", "20日 Top1"), ("monthly_top1_60d", "60日 Top1"),
                                ("four_bank_equal_weight_hold", "四股等权")):
            a = w["scenarios"][key_name]["audit"]
            out.append(f"- {title}：现金非负 {a['cash_nonnegative']}；最终空仓 {a['final_liquidation_complete']}；因果/T+1 检查 {a['causal_signal_and_t_plus_1_check_pass']}；最大账本残差 {a['max_abs_equity_cash_plus_mark_residual']:.3g}；阻塞买入/卖出 {a['blocked_entries']}/{a['blocked_exits']}；清仓日 {a['final_liquidation_day']}。")
        out.append("")
    out += ["- 结论边界：该固定规则的历史收益用于候选比较；历史窗口不足以证明未来收益稳定或存在可持续 alpha。", ""]
    return "\n".join(out)


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
        "unit": "experiments2/V15 monthly 20-session momentum Top1",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_built_at": meta.get("built_at"),
        "snapshot_last_day": latest,
        "universe": list(codes),
        "rule": {
            "signal": "first observed trading session of each calendar month at close",
            "factor": "adjusted close(T) / adjusted close(T-lookback) - 1",
            "ranking": "descending cross-sectional total return; equal-weight fixed Top1",
            "execution": "next trading session open",
            "lookback_intervals_under_test": 20,
            "same_ledger_comparator_lookback_intervals": 60,
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
        "account_engine_source": "copied from experiments2/V4/account_engine.py; V4 untouched",
        "runner_sha256": script_hash,
        "account_engine_sha256": engine_hash,
        "created_at": out["built_at"],
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "REPORT.md").write_text(build_report(out), encoding="utf-8")
    print("V15_RESULTS", OUT / "result.json", flush=True)
    for wname, w in (("official", official), ("extended", extended)):
        print(wname, flush=True)
        for name, p in w["scenarios"].items():
            print(name, p["statistics"]["cumulative_net_return"], p["statistics"]["max_drawdown"],
                  p["audit"]["fees"], p["audit"]["trade_count"], flush=True)
        print("window", w["window"], flush=True)
    print("max RSS GiB", out["process_max_rss_gib"], "elapsed", time.time() - started, flush=True)


if __name__ == "__main__":
    main()

