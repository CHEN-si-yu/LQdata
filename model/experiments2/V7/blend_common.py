#!/usr/bin/env python3
"""Fixed four-bank 60-session momentum Top2, evaluated with the shared audited account engine."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import csv
import hashlib
import inspect
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

V1_DIR = Path("/root/autodl-fs/model/experiments2/V1")
V3_ENGINE = Path(__file__).resolve().parent / "rank_portfolio_lab.py"
OUT = Path(__file__).resolve().parent / "model_pred"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(V1_DIR))
import model as M

_spec = importlib.util.spec_from_file_location("rank_portfolio_v3_engine", V3_ENGINE)
_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine)
# Reuse the audited ledger verbatim, changing only its hard-coded OOS start so each
# requested window can start with its own fresh account.
_engine_src = inspect.getsource(_engine.account_sim)
_old_start = 'nday,nc=op.shape; first=int(np.searchsorted(days,"2023-01-03")); last=nday-1'
_new_start = 'nday,nc=op.shape; first=0; last=nday-1'
_old_guard = 'if days[first]>"2023-01-03": raise RuntimeError("OOS start date missing")'
_new_guard = 'if first >= len(days): raise RuntimeError("empty evaluation window")'
if _engine_src.count(_old_start) != 1 or _engine_src.count(_old_guard) != 1:
    raise RuntimeError("audited engine source changed; refusing an unverified patch")
_engine_src = _engine_src.replace(_old_start, _new_start).replace(_old_guard, _new_guard)
exec(compile(_engine_src, str(V3_ENGINE) + "::windowed", "exec"), _engine.__dict__)

START_FULL, END_FULL = "2023-01-03", "2026-09-24"
START_TEST, END_TEST = "2025-07-01", "2026-06-30"
LOOKBACK, TOP_K = 60, 2


def month_first_indices(days):
    months = np.array([str(d)[:7] for d in days])
    return np.flatnonzero(np.r_[True, months[1:] != months[:-1]])


def make_schedule(days, codes, arrays, start, end):
    close = arrays["close"]
    adj = pd.DataFrame(arrays["adj_factor"]).ffill().to_numpy(dtype=float)
    adjusted_close = close * adj
    signals, schedule = [], {}
    for si in month_first_indices(days):
        d = str(days[si])
        if d < start or d > end or si < LOOKBACK or si + 1 >= len(days) or str(days[si + 1]) > end:
            continue
        old = adjusted_close[si - LOOKBACK]
        now = adjusted_close[si]
        valid = np.isfinite(old) & (old > 0) & np.isfinite(now) & (now > 0)
        candidates = np.flatnonzero(valid)
        if len(candidates) == 0:
            continue
        returns = np.full(len(codes), np.nan, dtype=float)
        returns[valid] = now[valid] / old[valid] - 1.0
        ranked = sorted(candidates.tolist(), key=lambda c: (-returns[c], str(codes[c])))
        chosen = ranked[:TOP_K]
        schedule[int(si + 1)] = np.asarray(chosen, dtype=int)
        signals.append({
            "signal_date": d, "execution_date": str(days[si + 1]),
            "ranked": [{"code": str(codes[c]), "return_60d": float(returns[c])} for c in ranked],
            "selected": [str(codes[c]) for c in chosen],
        })
    return schedule, signals


def month_or_first_schedule(schedule):
    if not schedule:
        return {}
    first_exec = min(schedule)
    return {first_exec: schedule[first_exec].copy()}


def engine_mark_residual(curve, arrays, n_codes):
    lastmark = np.full(n_codes, np.nan, dtype=float)
    errors = []
    for i, row in enumerate(curve):
        op = arrays["open"][i]
        close = arrays["close"][i]
        marks = np.where(np.isfinite(op) & (op > 0), op,
                         np.where(np.isfinite(close) & (close > 0), close, lastmark))
        marks = np.where(np.isfinite(marks) & (marks > 0), marks, lastmark)
        shares = np.asarray(row["shares"], dtype=float)
        marked_value = float(np.sum(shares * np.nan_to_num(marks, nan=0.0)))
        errors.append(abs(float(row["equity"]) - float(row["cash"]) - marked_value))
        lastmark = marks.copy()
    return max(errors) if errors else 0.0


def run_window(all_days, all_codes, all_arrays, start, end):
    ix = np.flatnonzero((all_days >= start) & (all_days <= end))
    if len(ix) == 0 or str(all_days[ix[0]]) != start or str(all_days[ix[-1]]) != end:
        raise RuntimeError(f"requested window endpoints missing: {start}..{end}")
    days = all_days[ix]
    arrays = {k: v[ix] for k, v in all_arrays.items()}
    schedule, signals = make_schedule(all_days, all_codes, all_arrays, start, end)
    local_schedule = {int(i - ix[0]): names for i, names in schedule.items()
                      if ix[0] <= i <= ix[-1]}
    if not local_schedule:
        raise RuntimeError("no monthly rebalance signals in requested window")

    result = _engine.account_sim(days, all_codes, arrays, local_schedule)
    stats = _engine.summarize(result["curve"], _engine.COST["account_money"])

    first_signal = min(schedule)
    initial_hold = {first_signal: schedule[first_signal].copy()}
    held = _engine.account_sim(days, all_codes, arrays,
                               {int(i - ix[0]): names for i, names in initial_hold.items()
                                if ix[0] <= i <= ix[-1]})
    held_stats = _engine.summarize(held["curve"], _engine.COST["account_money"])
    first_exec = int(np.flatnonzero(days >= str(days[0]))[0]) + 1
    equal_schedule = {first_exec: np.arange(len(all_codes), dtype=int)} if first_exec < len(days) else {}
    equal = _engine.account_sim(days, all_codes, arrays, equal_schedule)
    equal_stats = _engine.summarize(equal["curve"], _engine.COST["account_money"])

    curve = result["curve"]
    cash_min = min(float(r["cash"]) for r in curve)
    max_holdings = max(int(r["holdings"]) for r in curve)
    max_cash_residual = engine_mark_residual(curve, arrays, len(all_codes))
    audit = {
        "days": len(days), "monthly_signals": len(signals),
        "schedule_events_excluding_final_liquidation": len(local_schedule),
        "cash_min": cash_min, "ending_cash": float(result["ending_cash"]),
        "ending_positions": int(np.count_nonzero(np.asarray(result["ending_shares"]) > 1e-7)),
        "maximum_simultaneous_holdings": max_holdings,
        "blocked_entries": int(result["blocked_entries"]),
        "blocked_exits": int(result["blocked_exits"]),
        "trade_count": int(result["ntr"]), "fees": float(result["fees"]),
        "minimum_cash_nonnegative": bool(cash_min >= -1e-7),
        "final_liquidation_complete": bool(np.count_nonzero(np.asarray(result["ending_shares"]) > 1e-7) == 0),
        "max_abs_equity_cash_plus_engine_mark_residual": float(max_cash_residual),
        "matching_method": "previous execution-day shares x previous-close-to-current-open adjusted return / previous equity; compounded without fees",
    }
    baseline_audit = {
        "initial_top2_hold": {"fees": float(held["fees"]), "trade_count": int(held["ntr"]),
                              "blocked_entries": int(held["blocked_entries"]),
                              "blocked_exits": int(held["blocked_exits"])},
        "four_bank_equal_weight_hold": {"fees": float(equal["fees"]), "trade_count": int(equal["ntr"]),
                                        "blocked_entries": int(equal["blocked_entries"]),
                                        "blocked_exits": int(equal["blocked_exits"])},
    }
    return {
        "window": [start, end], "strategy": stats, "initial_top2_hold": held_stats,
        "four_bank_equal_weight_hold": equal_stats, "audit": audit,
        "baseline_audit": baseline_audit,
        "signals": signals, "curve": curve, "trades": result["trades"],
    }


def annual_rows(curve):
    df = pd.DataFrame(curve)
    years = pd.to_datetime(df["date"]).dt.year
    out = []
    for y in sorted(years.unique()):
        g = df.loc[years == y]
        net = float(np.prod(1.0 + g["daily_return"].to_numpy(float)) - 1.0)
        matched = float(np.prod(1.0 + g["matched_return"].to_numpy(float)) - 1.0)
        out.append({"year": int(y), "days": int(len(g)), "net_return": net,
                    "matched_return": matched, "matched_excess": net - matched})
    return out


def main():
    started = time.time()
    print("loading four-bank panel via V1's existing loader", flush=True)
    panel = M.load_panel()
    days, codes = panel.days, panel.codes
    arrays = {k: panel.prices.raw[k] for k in
              ("open", "high", "low", "pre_close", "close", "vol", "adj_factor")}
    arrays["amount"] = panel.amount
    if tuple(str(x) for x in codes) != tuple(M.CODES):
        raise RuntimeError(f"unexpected bank universe: {codes}")
    test_days = np.flatnonzero((days >= START_TEST) & (days <= END_TEST))
    if len(test_days) != 242:
        raise RuntimeError(f"official evaluation window must contain 242 sessions; got {len(test_days)}")

    full = run_window(days, codes, arrays, START_FULL, END_FULL)
    test = run_window(days, codes, arrays, START_TEST, END_TEST)
    full["yearly"] = annual_rows(full["curve"])
    test["quarterly"] = []
    tdf = pd.DataFrame(test["curve"])
    qkeys = pd.PeriodIndex(pd.to_datetime(tdf["date"]), freq="Q").astype(str)
    for q in sorted(set(qkeys)):
        g = tdf.loc[qkeys == q]
        net = float(np.prod(1.0 + g["daily_return"].to_numpy(float)) - 1.0)
        matched = float(np.prod(1.0 + g["matched_return"].to_numpy(float)) - 1.0)
        test["quarterly"].append({"quarter": q, "days": int(len(g)),
                                  "net_return": net, "matched_return": matched,
                                  "matched_excess": net - matched})
    strategy_net = float(test["strategy"]["cumulative_net_return"])
    test["selection_net_gain_pp"] = {
        "versus_initial_top2_hold": 100.0 * (strategy_net - float(test["initial_top2_hold"]["cumulative_net_return"])),
        "versus_four_bank_equal_weight_hold": 100.0 * (strategy_net - float(test["four_bank_equal_weight_hold"]["cumulative_net_return"])),
    }
    test["matched_interpretation"] = (
        "成本/执行摩擦差：匹配基准按V4实际持仓路径逐日计无费收益；该差不是选股alpha。"
    )

    v1_path = V1_DIR / "model_info" / "final_audit.json"
    v1_metrics = None
    if v1_path.exists():
        v1 = json.loads(v1_path.read_text(encoding="utf-8"))
        v1_metrics = v1.get("metrics_total", {}).get("action")

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = {
        "unit": "experiments2/V4 bank_momentum_v4",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_built_at": panel.data_built_at,
        "snapshot_last_day": str(days[-1]),
        "universe": [str(c) for c in codes],
        "rule": {
            "signal": "first observed trading session of each calendar month, at T close",
            "factor": "adjusted close(T) / adjusted close(T-60 trading sessions) - 1",
            "ranking": "descending 60-session total return; fixed Top2, equal-weight targets",
            "execution": "T+1 open; shared audited cash ledger, 100-share lots, 1% signal-day amount capacity, one-price limit locks, 3bp slippage, commission/transfer/sell stamp fees",
            "lookback_trading_intervals": LOOKBACK, "top_k": TOP_K,
            "no_parameter_sweep": True, "no_training": True,
            "matched_benchmark_interpretation": "cost/execution friction only on the same realized holdings path; not stock-selection alpha",
        },
        "official_242d": {k: v for k, v in test.items() if k not in ("curve", "trades", "signals")},
        "official_242d_yearly": annual_rows(test["curve"]),
        "extended": {k: v for k, v in full.items() if k not in ("curve", "trades", "signals")},
        "extended_yearly": full["yearly"],
        "v1_reference_same_window": v1_metrics,
        "runtime_seconds": time.time() - started,
        "script_sha256": script_hash,
    }
    (OUT / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for key, payload in (("official_242d", test), ("extended", full)):
        pd.DataFrame(payload["curve"]).drop(columns=["shares"]).to_csv(OUT / f"{key}_daily.csv", index=False)
        pd.DataFrame(payload["trades"]).to_csv(OUT / f"{key}_trades.csv", index=False)
        pd.DataFrame(payload["signals"]).to_json(OUT / f"{key}_signals.json", orient="records", force_ascii=False, indent=2)
    protocol = {
        "objective": "independent fixed-rule four-bank monthly cross-sectional momentum candidate",
        "requested_by": "parent agent",
        "start_test": START_TEST, "end_test": END_TEST, "expected_test_sessions": 242,
        "extended_start": START_FULL, "extended_end": END_FULL,
        "rule": out["rule"], "account_engine_source": str(V3_ENGINE),
        "panel_loader_source": str(V1_DIR / "model.py"),
        "source_sha256": script_hash, "created_at": out["built_at"],
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# 四大行 60 日动量 Top2（月频）",
        "",
        f"- 数据快照：{panel.data_built_at}；数据截至 {days[-1]}。",
        f"- 固定规则：月初首个交易日收盘，按过去60个交易间隔复权收盘总收益率排序，持有Top2等权；下一交易日开盘调仓。",
        "- 不训练模型、不扫描参数。交易账本复用 V3 已审计引擎；费用、整手、成交额容量、涨跌停锁单和滑点均开启。",
        "- 严格匹配基准按前一执行日真实持仓市值权重，逐日计无费开盘到开盘收益；与策略的差额仅称成本/执行摩擦差，不是选股alpha。",
        "",
        "## 官方242日同窗（含费用）",
        "",
        "| 策略 | 净收益 | 相对月频Top2净收益差 | 同持仓无费收益 | 成本/执行摩擦差 | 最大回撤 | 平均敞口 | 费用 | 成交 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    compare_rows = [
        ("月初Top2动量", test["strategy"], "—", test["audit"]),
        ("首次Top2后持有", test["initial_top2_hold"],
         f"{test['selection_net_gain_pp']['versus_initial_top2_hold'] * -1:+.2f}pp",
         test["baseline_audit"]["initial_top2_hold"]),
        ("四股等权持有", test["four_bank_equal_weight_hold"],
         f"{test['selection_net_gain_pp']['versus_four_bank_equal_weight_hold'] * -1:+.2f}pp",
         test["baseline_audit"]["four_bank_equal_weight_hold"]),
    ]
    for name, stat, delta, audit in compare_rows:
        report.append(f"| {name} | {stat['cumulative_net_return']:.2%} | {delta} | {stat['matched_benchmark_return']:.2%} | {stat['matched_cumulative_excess']:.2%} | {stat['max_drawdown']:.2%} | {stat['avg_exposure']:.1%} | {audit['fees']:.2f} | {audit['trade_count']} |")
    report += ["",
               f"- 月频Top2净收益相对首次Top2持有提升 {test['selection_net_gain_pp']['versus_initial_top2_hold']:+.2f} 个百分点；相对四股等权持有提升 {test['selection_net_gain_pp']['versus_four_bank_equal_weight_hold']:+.2f} 个百分点。这是相对持有基线的轮动收益差。",
               "- 同持仓无费收益与月频Top2策略走的是相同实际持仓路径，差额只反映费用和执行摩擦；不能解读为选股alpha。",
               "", "### 季度拆分", "", "| 季度 | 交易日 | 净收益 | 同持仓无费收益 | 成本/执行摩擦差 |",
               "|---|---:|---:|---:|---:|"]
    for row in test["quarterly"]:
        report.append(f"| {row['quarter']} | {row['days']} | {row['net_return']:.2%} | {row['matched_return']:.2%} | {row['matched_excess']:.2%} |")
    report += ["", "### 2023-01 至最新扩展窗", "", "| 年份 | 交易日 | 净收益 | 同持仓无费收益 | 成本/执行摩擦差 |",
               "|---:|---:|---:|---:|---:|"]
    for row in full["yearly"]:
        report.append(f"| {row['year']} | {row['days']} | {row['net_return']:.2%} | {row['matched_return']:.2%} | {row['matched_excess']:.2%} |")
    report += ["", f"- 官方窗审计：现金最低 {test['audit']['cash_min']:.4f}，最大持仓数 {test['audit']['maximum_simultaneous_holdings']}，终值持仓 {test['audit']['ending_positions']}，阻塞买入/卖出 {test['audit']['blocked_entries']}/{test['audit']['blocked_exits']}。",
                f"- 扩展窗审计：现金最低 {full['audit']['cash_min']:.4f}，最大持仓数 {full['audit']['maximum_simultaneous_holdings']}，终值持仓 {full['audit']['ending_positions']}，阻塞买入/卖出 {full['audit']['blocked_entries']}/{full['audit']['blocked_exits']}。",
                "- 结果仅代表这条固定规则在既定数据快照上的回测，不是对稳定超额的确认。",
                ""]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("V4_RESULTS", OUT / "result.json", flush=True)
    print("official net/matched/mdd", test["strategy"]["cumulative_net_return"],
          test["strategy"]["matched_benchmark_return"], test["strategy"]["max_drawdown"], flush=True)
    print("extended yearly", full["yearly"], flush=True)
    print("elapsed", time.time() - started, "seconds", flush=True)


if __name__ == "__main__":
    main()

