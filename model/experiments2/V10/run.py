#!/usr/bin/env python3
"""Fixed consensus candidate: monthly V1 score Top2 intersect 60-session momentum Top2."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "V1"
sys.path.insert(0, str(V1))
import model as M  # noqa: E402
import account_engine as E  # noqa: E402

START, END, EXIT_DAY = "2025-07-01", "2026-06-30", "2026-07-01"
LOOKBACK, TOP_K = 60, 2


def source_hashes():
    paths = [
        V1 / "model.py",
        V1 / "model_info/final_audit.json",
        V1 / "model_pred/ensemble/score_meta.json",
        V1 / "model_pred/ensemble/year=2025/data.parquet",
        V1 / "model_pred/ensemble/year=2026/data.parquet",
    ]
    return {str(p.relative_to(V1)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_scores():
    paths = [V1 / "model_pred/ensemble/year=2025/data.parquet",
             V1 / "model_pred/ensemble/year=2026/data.parquet"]
    df = pd.concat([pq.read_table(p, columns=["trade_date", "stock_code", "score"]).to_pandas()
                    for p in paths], ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df = df[(df.trade_date >= START) & (df.trade_date <= END)]
    if df.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("duplicate V1 cached scores")
    by_day = {str(d): {str(r.stock_code): float(r.score) for r in g.itertuples()}
              for d, g in df.groupby("trade_date", sort=False)}
    return by_day, {"score_days": len(by_day), "score_rows": len(df),
                    "score_min_date": min(by_day), "score_max_date": max(by_day),
                    "source_sha256": {p: source_hashes()[p] for p in (
                        "model_pred/ensemble/year=2025/data.parquet",
                        "model_pred/ensemble/year=2026/data.parquet")}}


def first_session_month_indices(days):
    month = pd.to_datetime(days).strftime("%Y-%m").to_numpy()
    return np.flatnonzero(np.r_[True, month[1:] != month[:-1]])


def rank_top2(values, codes):
    valid = [j for j in range(len(codes)) if np.isfinite(values[j])]
    ordered = sorted(valid, key=lambda j: (-float(values[j]), str(codes[j])))
    ranks = {int(j): rank + 1 for rank, j in enumerate(ordered)}
    return ordered[:TOP_K], ranks


def schedules(days, codes, arrays, score_by_day):
    adj = pd.DataFrame(arrays["adj_factor"]).ffill().to_numpy(dtype=float)
    adjusted_close = arrays["close"] * adj
    out = {name: ({}, []) for name in ("consensus", "score_only", "momentum_only")}
    for si in first_session_month_indices(days):
        d = str(days[si])
        if d < START or d > END or si < LOOKBACK or si + 1 >= len(days) or str(days[si + 1]) > END:
            continue
        date_scores = score_by_day.get(d)
        if date_scores is None:
            continue
        score = np.array([date_scores.get(str(code), np.nan) for code in codes], dtype=float)
        old, now = adjusted_close[si - LOOKBACK], adjusted_close[si]
        valid_mom = np.isfinite(old) & (old > 0) & np.isfinite(now) & (now > 0)
        momentum = np.full(len(codes), np.nan, dtype=float)
        momentum[valid_mom] = now[valid_mom] / old[valid_mom] - 1.0
        score_top, score_ranks = rank_top2(score, codes)
        momentum_top, momentum_ranks = rank_top2(momentum, codes)
        overlap = [j for j in score_top if j in set(momentum_top)]
        choices = {"consensus": overlap, "score_only": score_top, "momentum_only": momentum_top}
        signal = {
            "signal_date": d, "execution_date": str(days[si + 1]),
            "score_top2": [str(codes[j]) for j in score_top],
            "momentum_top2": [str(codes[j]) for j in momentum_top],
            "consensus": [str(codes[j]) for j in overlap],
            "rank_details": [{"code": str(codes[j]), "score": float(score[j]) if np.isfinite(score[j]) else None,
                              "score_rank": score_ranks.get(j), "momentum_60d": float(momentum[j]) if np.isfinite(momentum[j]) else None,
                              "momentum_rank": momentum_ranks.get(j)}
                             for j in range(len(codes))],
        }
        for name, selected in choices.items():
            schedule, signals = out[name]
            schedule[int(si + 1)] = np.asarray(selected, dtype=int)
            signals.append({**signal, "strategy": name,
                            "selected": [str(codes[j]) for j in selected]})
    return out


def daily_quarters(curve):
    df = pd.DataFrame(curve)
    df = df[pd.to_datetime(df.date) <= pd.Timestamp(END)]
    qkeys = pd.PeriodIndex(pd.to_datetime(df.date), freq="Q").astype(str)
    rows = []
    for q in sorted(set(qkeys)):
        g = df.loc[qkeys == q]
        net = float(np.prod(1.0 + g.daily_return.to_numpy(float)) - 1.0)
        matched = float(np.prod(1.0 + g.matched_return.to_numpy(float)) - 1.0)
        rows.append({"quarter": q, "days": int(len(g)), "net_return": net,
                     "matched_return": matched, "cost_execution_difference": net - matched})
    return rows


def mark_residual(curve, arrays, n_codes):
    lastmark = np.full(n_codes, np.nan, dtype=float)
    errors = []
    for i, row in enumerate(curve):
        op, close = arrays["open"][i], arrays["close"][i]
        marks = np.where(np.isfinite(op) & (op > 0), op,
                         np.where(np.isfinite(close) & (close > 0), close, lastmark))
        marks = np.where(np.isfinite(marks) & (marks > 0), marks, lastmark)
        mv = float(np.sum(np.asarray(row["shares"], dtype=float) * np.nan_to_num(marks, nan=0.0)))
        errors.append(abs(float(row["equity"]) - float(row["cash"]) - mv))
        lastmark = marks.copy()
    return max(errors) if errors else 0.0


def main():
    began = time.time()
    hashes_before = source_hashes()
    score_by_day, score_audit = load_scores()
    panel = M.load_panel()
    all_days, codes = panel.days, panel.codes
    if tuple(str(c) for c in codes) != tuple(M.CODES):
        raise RuntimeError(f"unexpected V1 universe: {codes}")
    official = int(((all_days >= START) & (all_days <= END)).sum())
    if official != 242 or len(score_by_day) != official or score_audit["score_rows"] != official * len(codes):
        raise RuntimeError(f"official day/score coverage mismatch days={official}, audit={score_audit}")
    ix = np.flatnonzero((all_days >= START) & (all_days <= EXIT_DAY))
    days = all_days[ix]
    if len(days) != 243 or str(days[0]) != START or str(days[-1]) != EXIT_DAY:
        raise RuntimeError(f"expected 243-day lifecycle through exit day: {days[0]}..{days[-1]} ({len(days)})")
    full_arrays = {k: panel.prices.raw[k] for k in ("open", "high", "low", "pre_close", "close", "vol", "adj_factor")}
    full_arrays["amount"] = panel.amount
    schedules_by_name = schedules(all_days, codes, full_arrays, score_by_day)
    arrays = {k: v[ix] for k, v in full_arrays.items()}

    strategies = {}
    for name, (schedule, signals) in schedules_by_name.items():
        local_schedule = {int(i - ix[0]): selected for i, selected in schedule.items()
                          if ix[0] <= i < ix[-1]}
        result = E.account_sim(days, codes, arrays, local_schedule)
        stats = E.summarize(result["curve"], E.COST["account_money"])
        curve = result["curve"]
        cash_min = min(float(r["cash"]) for r in curve)
        max_holdings = max(int(r["holdings"]) for r in curve)
        end_positions = int(np.count_nonzero(np.asarray(result["ending_shares"]) > 1e-7))
        residual = mark_residual(curve, arrays, len(codes))
        exit_row = curve[-1]
        quarter = daily_quarters(curve)
        audit = {
            "days_including_exit": len(curve), "signal_days": len(signals),
            "schedule_events_excluding_exit_liquidation": len(local_schedule),
            "cash_min": cash_min, "ending_cash": float(result["ending_cash"]),
            "ending_positions": end_positions, "maximum_simultaneous_holdings": max_holdings,
            "blocked_entries": int(result["blocked_entries"]), "blocked_exits": int(result["blocked_exits"]),
            "trade_count": int(result["ntr"]), "fees": float(result["fees"]),
            "minimum_cash_nonnegative": bool(cash_min >= -1e-7),
            "final_liquidation_complete": bool(end_positions == 0),
            "max_abs_equity_cash_plus_engine_mark_residual": float(residual),
            "exit_day": str(exit_row["date"]), "exit_day_net_return": float(exit_row["daily_return"]),
            "exit_day_cash": float(exit_row["cash"]), "exit_day_holdings": int(exit_row["holdings"]),
        }
        if audit["maximum_simultaneous_holdings"] > TOP_K or not audit["minimum_cash_nonnegative"] or not audit["final_liquidation_complete"] or residual > 1e-6:
            raise RuntimeError(f"account audit failed for {name}: {audit}")
        strategies[name] = {
            "stats": stats, "audit": audit, "quarterly_signal_window": quarter,
            "signals": signals, "curve": curve, "trades": result["trades"],
        }
        pd.DataFrame(curve).drop(columns=["shares"]).to_csv(ROOT / "model_pred" / f"{name}_daily.csv", index=False)
        pd.DataFrame(result["trades"]).to_csv(ROOT / "model_pred" / f"{name}_trades.csv", index=False)
        pd.DataFrame(signals).to_json(ROOT / "model_pred" / f"{name}_signals.json", orient="records", force_ascii=False, indent=2)

    hashes_after = source_hashes()
    if hashes_before != hashes_after:
        raise RuntimeError("V1 source hashes changed during run")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine_hash = hashlib.sha256((ROOT / "account_engine.py").read_bytes()).hexdigest()
    payload = {
        "unit": "experiments2/V10 score/momentum Top2 intersection",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_built_at": panel.data_built_at, "snapshot_last_day": str(all_days[-1]),
        "signal_window": [START, END], "execution_only_exit_day": EXIT_DAY,
        "universe": [str(c) for c in codes], "official_signal_days": official,
        "rule": {
            "signal": "first trading session of each calendar month at close",
            "score": "rank cached V1 ensemble score descending across finite-score names; select Top2",
            "momentum": "rank adjusted close(T)/adjusted close(T-60 sessions)-1 descending; select Top2",
            "selection": "hold only names in both independently ranked Top2 lists; 0/1/2 names allowed",
            "allocation": "target 50% of current pre-trade NAV per selected name; unused target allocation remains cash",
            "execution": "T+1 open; 100-share lots, 1% previous signal-day amount capacity, limit locks, 3bp slippage and fees",
            "no_retraining": True, "no_parameter_scan": True,
        },
        "score_input_audit": score_audit, "v1_source_hashes_before": hashes_before,
        "v1_source_hashes_after": hashes_after, "v1_source_hashes_unchanged": True,
        "engine": {"source": "copied from V7/rank_portfolio_lab.py", "sha256": engine_hash,
                   "fixed_window_change": "start account at first supplied lifecycle day",
                   "allocation_change": "0.50 NAV target for each selected symbol"},
        "strategies": {name: {k: v for k, v in obj.items() if k in (
            "stats", "audit", "quarterly_signal_window")}
                        for name, obj in strategies.items()},
        "script_sha256": script_hash, "runtime_seconds": time.time() - began,
        "interpretation": "Matched return is the same realized holdings path without transaction costs; the gap measures costs/execution only and is not selection alpha.",
    }
    (ROOT / "model_info/final_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    protocol = {"objective": "fixed monthly intersection of model-score Top2 and 60-session momentum Top2",
                "window": [START, END], "execution_only_exit_day": EXIT_DAY, "rule": payload["rule"],
                "V1_source_hashes": hashes_before, "engine_sha256": engine_hash,
                "script_sha256": script_hash, "snapshot_built_at": panel.data_built_at,
                "created_at": payload["built_at"]}
    (ROOT / "model_info/protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    report = ["# V10 四大行分数 Top2 ∩ 60 日动量 Top2（月频）", "",
              f"- 窗口：{START} 至 {END}，共 {official} 个信号交易日；另计 {EXIT_DAY} T+1 退出日。快照截至 {all_days[-1]}。",
              "- 固定规则：月首交易日收盘分别对 V1 缓存集成分数、60 交易间隔复权动量排序，各取 Top2，只持有交集。每只目标为交易前账户净值的 50%；只有一只时另一半留现金，无交集时转现金。次日开盘执行。",
              "- 不重训、不扫描参数。三个组合均使用复制自 V7 的账户账本；V10 引擎只调整生命周期起点与单名目标权重。",
              "- 同持仓无费匹配差只反映成本和执行摩擦，不代表选股 alpha。", "",
              "## 官方同窗和组成策略基线", "",
              "| 策略 | 净收益 | 同持仓无费收益 | 成本/执行摩擦差 | 最大回撤 | 平均敞口 | 费用 | 交易 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("consensus", "score_only", "momentum_only"):
        obj = strategies[name]
        s, a = obj["stats"], obj["audit"]
        label = {"consensus": "V10 交集策略", "score_only": "V1 分数 Top2", "momentum_only": "60 日动量 Top2"}[name]
        report.append(f"| {label} | {s['cumulative_net_return']:.2%} | {s['matched_benchmark_return']:.2%} | {s['matched_cumulative_excess']:.2%} | {s['max_drawdown']:.2%} | {s['avg_exposure']:.1%} | ¥{a['fees']:.2f} | {a['trade_count']} |")
    report += ["", "## V10 分季度信号期收益", "", "| 季度 | 交易日 | 净收益 | 同持仓无费收益 | 成本/执行摩擦差 |",
               "|---|---:|---:|---:|---:|"]
    for q in strategies["consensus"]["quarterly_signal_window"]:
        report.append(f"| {q['quarter']} | {q['days']} | {q['net_return']:.2%} | {q['matched_return']:.2%} | {q['cost_execution_difference']:.2%} |")
    report += ["", "## T+1 退出日", "", "| 策略 | 退出日收益 | 生命周期累计净收益 | 退出日持仓 |", "|---|---:|---:|---:|"]
    for name in ("consensus", "score_only", "momentum_only"):
        obj = strategies[name]
        report.append(f"| {name} | {obj['audit']['exit_day_net_return']:.2%} | {obj['stats']['cumulative_net_return']:.2%} | {obj['audit']['exit_day_holdings']} |")
    report += ["", "## 账本和时序审计", ""]
    for name in ("consensus", "score_only", "momentum_only"):
        a = strategies[name]["audit"]
        report.append(f"- {name}: {a['days_including_exit']} 日、{a['signal_days']} 个调仓信号；最低现金 ¥{a['cash_min']:.2f}，最多同时持有 {a['maximum_simultaneous_holdings']} 只，期末持仓 {a['ending_positions']} 只，阻塞买/卖 {a['blocked_entries']}/{a['blocked_exits']}，账务残差 ¥{a['max_abs_equity_cash_plus_engine_mark_residual']:.3g}。")
    report += ["- 买卖仅在信号次日开盘，最后一天只做强制清仓；规则不使用信号日之后的数据。",
               "- V1 模型加载器、分数元信息、2025/2026 缓存分数与最终审计文件均在运行前后核验 SHA-256，一致。",
               "- 结果仅代表固定规则在此数据快照上的回测，不能据此确认稳定超额。", ""]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (ROOT / "RUN_LOG.md").write_text(
        f"# V10 运行记录\n\n- 开始：{payload['built_at']}\n- 固定规则：月首分别选 V1 分数 Top2 与 60 日动量 Top2，仅取交集；每只目标 50% NAV。\n- 官方窗口：{START} 至 {END}；退出日：{EXIT_DAY}。\n- 不训练、不调参、不读写 V1；运行前后相关 V1 SHA-256 一致。\n- 运行耗时：{payload['runtime_seconds']:.1f} 秒。\n- V10 净收益：{strategies['consensus']['stats']['cumulative_net_return']:.6%}；最大回撤：{strategies['consensus']['stats']['max_drawdown']:.6%}；费用：{strategies['consensus']['audit']['fees']:.2f}。\n",
        encoding="utf-8")
    print(json.dumps({name: {"net": obj["stats"]["cumulative_net_return"],
                             "mdd": obj["stats"]["max_drawdown"],
                             "exposure": obj["stats"]["avg_exposure"],
                             "audit": obj["audit"], "quarterly": obj["quarterly_signal_window"]}
                      for name, obj in strategies.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
