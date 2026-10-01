# Packaged, path-rebased runner; frozen metrics were produced by run_frozen_reference.py.
#!/usr/bin/env python3
"""Fixed four-bank 60-session momentum Top2, evaluated with the shared audited account engine."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

V8_DIR = Path(__file__).resolve().parent
OUT = V8_DIR
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(V8_DIR))
import account_engine as _engine
import panel_loader as M

START_FULL, END_FULL = "2023-01-03", "2026-09-24"
START_TEST, END_SIGNAL_TEST, END_TEST_LIFECYCLE = "2025-07-01", "2026-06-30", "2026-07-01"
LOOKBACK, TOP_K = 60, 2


def month_first_indices(days):
    months = np.array([str(d)[:7] for d in days])
    return np.flatnonzero(np.r_[True, months[1:] != months[:-1]])



def make_schedule(days, codes, arrays, start, end, signal_end=None):
    close = arrays["close"]
    adj = pd.DataFrame(arrays["adj_factor"]).ffill().to_numpy(dtype=float)
    adjusted_close = close * adj
    signal_end = end if signal_end is None else signal_end
    signals, gated, ungated = [], {}, {}
    for si in month_first_indices(days):
        d = str(days[si])
        if d < start or d > signal_end or si < LOOKBACK or si + 1 >= len(days) or str(days[si + 1]) > end:
            continue
        old, now = adjusted_close[si - LOOKBACK], adjusted_close[si]
        valid = np.isfinite(old) & (old > 0) & np.isfinite(now) & (now > 0)
        if not bool(np.all(valid)):
            raise RuntimeError(f"all four bank 60-session returns must be valid on {d}")
        returns = now / old - 1.0
        ranked = sorted(range(len(codes)), key=lambda c: (-returns[c], str(codes[c])))
        chosen = ranked[:TOP_K]
        mean_return = float(np.mean(returns))
        gate_pass = mean_return > 0.0
        exec_i = int(si + 1)
        ungated[exec_i] = np.asarray(chosen, dtype=int)
        gated[exec_i] = np.asarray(chosen if gate_pass else [], dtype=int)
        signals.append({
            "signal_date": d,
            "execution_date": str(days[exec_i]),
            "mean_4bank_return_60d": mean_return,
            "gate": "strictly_greater_than_zero",
            "gate_pass": bool(gate_pass),
            "ranked": [{"code": str(codes[c]), "return_60d": float(returns[c])} for c in ranked],
            "ungated_top2": [str(codes[c]) for c in chosen],
            "selected": [str(codes[c]) for c in chosen] if gate_pass else [],
            "target_state": "equal_weight_top2" if gate_pass else "cash",
        })
    return gated, ungated, signals

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



def run_window(all_days, all_codes, all_arrays, start, end, signal_end=None):
    ix = np.flatnonzero((all_days >= start) & (all_days <= end))
    if len(ix) == 0 or str(all_days[ix[0]]) != start or str(all_days[ix[-1]]) != end:
        raise RuntimeError(f"requested window endpoints missing: {start}..{end}")
    days = all_days[ix]
    arrays = {k: v[ix] for k, v in all_arrays.items()}
    schedule, ungated_schedule, signals = make_schedule(all_days, all_codes, all_arrays, start, end, signal_end)
    local_schedule = {int(i - ix[0]): names for i, names in schedule.items()
                      if ix[0] <= i <= ix[-1]}
    local_ungated = {int(i - ix[0]): names for i, names in ungated_schedule.items()
                     if ix[0] <= i <= ix[-1]}
    if not local_schedule or not local_ungated:
        raise RuntimeError("no monthly rebalance signals in requested window")

    result = _engine.account_sim(days, all_codes, arrays, local_schedule)
    ungated = _engine.account_sim(days, all_codes, arrays, local_ungated)
    stats = _engine.summarize(result["curve"], _engine.COST["account_money"])
    ungated_stats = _engine.summarize(ungated["curve"], _engine.COST["account_money"])

    first_signal = min(ungated_schedule)
    initial_hold = {first_signal: ungated_schedule[first_signal].copy()}
    held = _engine.account_sim(days, all_codes, arrays,
                               {int(i - ix[0]): names for i, names in initial_hold.items()
                                if ix[0] <= i <= ix[-1]})
    held_stats = _engine.summarize(held["curve"], _engine.COST["account_money"])
    first_exec = 1
    equal_schedule = {first_exec: np.arange(len(all_codes), dtype=int)} if first_exec < len(days) else {}
    equal = _engine.account_sim(days, all_codes, arrays, equal_schedule)
    equal_stats = _engine.summarize(equal["curve"], _engine.COST["account_money"])

    curve = result["curve"]
    cash_min = min(float(r["cash"]) for r in curve)
    max_holdings = max(int(r["holdings"]) for r in curve)
    audit = {
        "days": len(days), "monthly_signals": len(signals),
        "gate_pass_signals": sum(bool(s["gate_pass"]) for s in signals),
        "gate_fail_signals": sum(not bool(s["gate_pass"]) for s in signals),
        "schedule_events_excluding_final_liquidation": len(local_schedule),
        "cash_min": cash_min, "ending_cash": float(result["ending_cash"]),
        "ending_positions": int(np.count_nonzero(np.asarray(result["ending_shares"]) > 1e-7)),
        "maximum_simultaneous_holdings": max_holdings,
        "blocked_entries": int(result["blocked_entries"]),
        "blocked_exits": int(result["blocked_exits"]),
        "trade_count": int(result["ntr"]), "fees": float(result["fees"]),
        "minimum_cash_nonnegative": bool(cash_min >= -1e-7),
        "final_liquidation_complete": bool(np.count_nonzero(np.asarray(result["ending_shares"]) > 1e-7) == 0),
        "max_abs_equity_cash_plus_engine_mark_residual": float(engine_mark_residual(curve, arrays, len(all_codes))),
        "matching_method": "previous execution-day shares x previous-close-to-current-open adjusted return / previous equity; compounded without fees",
    }
    ungated_audit = {
        "cash_min": min(float(r["cash"]) for r in ungated["curve"]),
        "ending_positions": int(np.count_nonzero(np.asarray(ungated["ending_shares"]) > 1e-7)),
        "maximum_simultaneous_holdings": max(int(r["holdings"]) for r in ungated["curve"]),
        "blocked_entries": int(ungated["blocked_entries"]),
        "blocked_exits": int(ungated["blocked_exits"]),
        "trade_count": int(ungated["ntr"]), "fees": float(ungated["fees"]),
        "max_abs_equity_cash_plus_engine_mark_residual": float(engine_mark_residual(ungated["curve"], arrays, len(all_codes))),
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
        "window": [start, end], "signal_date_cutoff": signal_end or end, "strategy": stats,
        "ungated_monthly_top2": ungated_stats, "ungated_audit": ungated_audit,
        "initial_top2_hold": held_stats, "four_bank_equal_weight_hold": equal_stats,
        "audit": audit, "baseline_audit": baseline_audit,
        "signals": signals, "curve": curve, "trades": result["trades"],
        "ungated_curve": ungated["curve"], "ungated_trades": ungated["trades"],
    }


def period_rows(curve, freq):
    df = pd.DataFrame(curve)
    dates = pd.to_datetime(df["date"])
    keys = dates.dt.to_period(freq).astype(str)
    out = []
    for period in sorted(keys.unique()):
        g = df.loc[keys == period]
        net = float(np.prod(1.0 + g["daily_return"].to_numpy(float)) - 1.0)
        matched = float(np.prod(1.0 + g["matched_return"].to_numpy(float)) - 1.0)
        out.append({"period": str(period), "days": int(len(g)), "net_return": net,
                    "matched_return": matched, "matched_excess": net - matched})
    return out


def annual_rows(curve):
    return [{**row, "year": int(row["period"])} for row in period_rows(curve, "Y")]

def main():
    started = time.time()
    print("loading four-bank panel via the V8 packaged panel loader", flush=True)
    panel = M.load_panel()
    days, codes = panel.days, panel.codes
    arrays = {k: panel.prices.raw[k] for k in
              ("open", "high", "low", "pre_close", "close", "vol", "adj_factor")}
    arrays["amount"] = panel.amount
    if tuple(str(x) for x in codes) != tuple(M.CODES):
        raise RuntimeError(f"unexpected bank universe: {codes}")
    test_signal_days = np.flatnonzero((days >= START_TEST) & (days <= END_SIGNAL_TEST))
    test_lifecycle_days = np.flatnonzero((days >= START_TEST) & (days <= END_TEST_LIFECYCLE))
    if len(test_signal_days) != 242 or len(test_lifecycle_days) != 243:
        raise RuntimeError(
            f"official lifecycle expected 242 signal-window + 1 exit session, got "
            f"{len(test_signal_days)}+{len(test_lifecycle_days)-len(test_signal_days)}"
        )

    full = run_window(days, codes, arrays, START_FULL, END_FULL)
    test = run_window(days, codes, arrays, START_TEST, END_TEST_LIFECYCLE,
                      signal_end=END_SIGNAL_TEST)

    full["yearly"] = annual_rows(full["curve"])
    full["ungated_yearly"] = annual_rows(full["ungated_curve"])
    test["signal_window_curve"] = [r for r in test["curve"] if r["date"] <= END_SIGNAL_TEST]
    test["ungated_signal_window_curve"] = [r for r in test["ungated_curve"] if r["date"] <= END_SIGNAL_TEST]
    if len(test["signal_window_curve"]) != 242 or test["curve"][-1]["date"] != END_TEST_LIFECYCLE:
        raise RuntimeError("official signal window or July 1 exit row mismatch")
    test["signal_window_metrics"] = {
        "window": [START_TEST, END_SIGNAL_TEST],
        "daily_sessions": len(test["signal_window_curve"]),
        "strategy": _engine.summarize(test["signal_window_curve"], _engine.COST["account_money"]),
        "ungated_monthly_top2": _engine.summarize(test["ungated_signal_window_curve"], _engine.COST["account_money"]),
        "quarterly": period_rows(test["signal_window_curve"], "Q"),
        "ungated_quarterly": period_rows(test["ungated_signal_window_curve"], "Q"),
        "signal_dates_cutoff": END_SIGNAL_TEST,
        "note": "signals/market-window reporting through June 30; July 1 final exit is excluded and reported separately",
    }
    test["quarterly"] = test["signal_window_metrics"]["quarterly"]
    test["ungated_quarterly"] = test["signal_window_metrics"]["ungated_quarterly"]
    exit_row = test["curve"][-1]
    ungated_exit_row = test["ungated_curve"][-1]
    exit_trades = [t for t in test["trades"] if t["date"] == END_TEST_LIFECYCLE]
    ungated_exit_trades = [t for t in test["ungated_trades"] if t["date"] == END_TEST_LIFECYCLE]
    test["exit_day"] = {
        "date": END_TEST_LIFECYCLE,
        "new_signal_generated": any(s["signal_date"] == END_TEST_LIFECYCLE for s in test["signals"]),
        "gated": {
            "net_return": float(exit_row["daily_return"]),
            "matched_no_fee_return": float(exit_row["matched_return"]),
            "matched_excess": float(exit_row["daily_return"] - exit_row["matched_return"]),
            "fees": float(sum(t["fee"] for t in exit_trades)),
            "trade_count": len(exit_trades),
            "cash_after_exit": float(exit_row["cash"]),
            "ending_holdings": int(exit_row["holdings"]),
        },
        "ungated_top2": {
            "net_return": float(ungated_exit_row["daily_return"]),
            "matched_no_fee_return": float(ungated_exit_row["matched_return"]),
            "matched_excess": float(ungated_exit_row["daily_return"] - ungated_exit_row["matched_return"]),
            "fees": float(sum(t["fee"] for t in ungated_exit_trades)),
            "trade_count": len(ungated_exit_trades),
            "cash_after_exit": float(ungated_exit_row["cash"]),
            "ending_holdings": int(ungated_exit_row["holdings"]),
        },
    }
    if test["exit_day"]["new_signal_generated"]:
        raise RuntimeError("July 1 must be exit-only and must not generate a new signal")
    test["net_gain_pp_vs_ungated_top2"] = 100.0 * (
        test["strategy"]["cumulative_net_return"] - test["ungated_monthly_top2"]["cumulative_net_return"])
    test["net_gain_pp_vs_initial_top2_hold"] = 100.0 * (
        test["strategy"]["cumulative_net_return"] - test["initial_top2_hold"]["cumulative_net_return"])
    test["net_gain_pp_vs_four_bank_equal_hold"] = 100.0 * (
        test["strategy"]["cumulative_net_return"] - test["four_bank_equal_weight_hold"]["cumulative_net_return"])

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rule = {
        "signal": "first observed trading session of each calendar month, at T close",
        "factor": "each bank adjusted close(T) / adjusted close(T-60 trading sessions) - 1",
        "gate": "hold only when the equal-weight mean of all four bank 60-session adjusted returns is strictly > 0; otherwise target cash",
        "selection": "when gate passes, descending individual 60-session return; fixed Top2, equal-weight targets",
        "execution": "T+1 open; shared audited cash ledger, 100-share lots, 1% signal-day amount capacity, one-price limit locks, 3bp slippage, commission/transfer/sell stamp fees",
        "lookback_trading_intervals": LOOKBACK, "top_k": TOP_K,
        "no_parameter_sweep": True, "no_training": True,
        "matched_benchmark_interpretation": "same realized holdings path, daily matched no-fee open-to-open benchmark; net gap measures trading cost/execution friction, not stock-selection alpha",
    }
    out = {
        "unit": "experiments2/V8 bank_momentum_60d_sign_gate",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_built_at": panel.data_built_at,
        "snapshot_last_day": str(days[-1]),
        "universe": [str(c) for c in codes],
        "rule": rule,
        "official_lifecycle_243d": {k: v for k, v in test.items() if k not in (
            "curve", "trades", "signals", "ungated_curve", "ungated_trades",
            "signal_window_curve", "ungated_signal_window_curve")},
        "official_242d_signal_window": test["signal_window_metrics"],
        "official_signal_window_quarterly": test["quarterly"],
        "official_exit_day": test["exit_day"],
        "extended": {k: v for k, v in full.items() if k not in ("curve", "trades", "signals", "ungated_curve", "ungated_trades")},
        "extended_yearly": full["yearly"],
        "extended_ungated_yearly": full["ungated_yearly"],
        "prior_sma_trend_gate_reference": {
            "source": "/autodl-fs/data/tmp/vlines/trend_gate_report.md",
            "window": "signal window 2023-01-03..2026-09-23, execute through 2026-09-24; prior report describes 904 sessions",
            "swept_rules_already_tested": "equal-weight total-return basket above SMA63/126/252, target exposure 50%/100%",
            "all_tested_strict_matched_excess": "negative, range -0.71pp to -5.63pp",
            "closest_risk_control_reference": {
                "rule": "SMA252 target 50%",
                "net_return": 0.6359, "matched_return": 0.6430,
                "matched_excess": -0.0071, "max_drawdown": -0.1080,
                "avg_exposure": 0.579,
            },
            "buy_and_hold": {
                "net_return": 1.4955, "matched_return": 1.4996,
                "matched_excess": -0.0040, "max_drawdown": -0.1549,
            },
            "static_half": {
                "net_return": 0.7425, "matched_return": 0.7444,
                "matched_excess": -0.0019, "max_drawdown": -0.1098,
            },
            "comparison_caveat": "prior SMA strategies gate a four-bank equal-weight basket and scale to 50%/100%; V8 instead uses a single 60-session cross-sectional mean-return sign to gate Top2 rotation. Similar trend-gating family, not a novelty claim. Prior SMA report's 904-session signal window differs slightly from V8 extended endpoint.",
        },
        "runtime_seconds": time.time() - started,
        "script_sha256": script_hash,
        "account_engine_sha256": hashlib.sha256((V8_DIR / "account_engine.py").read_bytes()).hexdigest(),
        "panel_loader_sha256": hashlib.sha256((V8_DIR / "panel_loader.py").read_bytes()).hexdigest(),
    }
    (OUT / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for key, payload in (("official_lifecycle_243d", test), ("extended", full)):
        pd.DataFrame(payload["curve"]).drop(columns=["shares"]).to_csv(OUT / f"{key}_daily.csv", index=False)
        pd.DataFrame(payload["trades"]).to_csv(OUT / f"{key}_trades.csv", index=False)
        pd.DataFrame(payload["signals"]).to_json(OUT / f"{key}_signals.json", orient="records", force_ascii=False, indent=2)
        pd.DataFrame(payload["ungated_curve"]).drop(columns=["shares"]).to_csv(OUT / f"{key}_ungated_top2_daily.csv", index=False)
        pd.DataFrame(payload["ungated_trades"]).to_csv(OUT / f"{key}_ungated_top2_trades.csv", index=False)
    pd.DataFrame(test["signal_window_curve"]).drop(columns=["shares"]).to_csv(OUT / "official_242d_signal_window_daily.csv", index=False)
    pd.DataFrame(test["ungated_signal_window_curve"]).drop(columns=["shares"]).to_csv(OUT / "official_242d_signal_window_ungated_top2_daily.csv", index=False)
    # Stable short aliases refer to the 242-session signal window, not the 243-session lifecycle.
    pd.DataFrame(test["signal_window_curve"]).drop(columns=["shares"]).to_csv(OUT / "official_242d_daily.csv", index=False)
    pd.DataFrame(test["ungated_signal_window_curve"]).drop(columns=["shares"]).to_csv(OUT / "official_242d_ungated_top2_daily.csv", index=False)
    pd.DataFrame([t for t in test["trades"] if t["date"] <= END_SIGNAL_TEST]).to_csv(OUT / "official_242d_trades.csv", index=False)
    pd.DataFrame([t for t in test["ungated_trades"] if t["date"] <= END_SIGNAL_TEST]).to_csv(OUT / "official_242d_ungated_top2_trades.csv", index=False)
    pd.DataFrame(test["signals"]).to_json(OUT / "official_242d_signals.json", orient="records", force_ascii=False, indent=2)

    protocol = {
        "objective": "fixed four-bank monthly 60-session momentum Top2 with one absolute mean-return sign gate",
        "requested_by": "parent agent",
        "start_test": START_TEST, "signal_end_test": END_SIGNAL_TEST, "lifecycle_end_test": END_TEST_LIFECYCLE,
        "expected_signal_window_sessions": 242, "expected_lifecycle_sessions": 243,
        "final_liquidation_timing": "2026-07-01 open; exit-only, no new signal",
        "extended_start": START_FULL, "extended_end": END_FULL,
        "extended_final_liquidation_timing": "snapshot-end 2026-09-24 open; no post-snapshot T+1 session",
        "rule": rule, "account_engine_source": "account_engine.py",
        "panel_loader_source": "panel_loader.py (frozen copy of experiments2/V1/model.py)",
        "no_parameter_sweep": True, "no_retraining": True,
        "source_sha256": script_hash,
        "sma_research_reference": "/autodl-fs/data/tmp/vlines/trend_gate_report.md",
        "created_at": out["built_at"],
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt_pct(x):
        return f"{float(x):.2%}"
    def row(name, stat, audit):
        return (f"| {name} | {fmt_pct(stat['cumulative_net_return'])} | "
                f"{fmt_pct(stat['matched_benchmark_return'])} | {fmt_pct(stat['matched_cumulative_excess'])} | "
                f"{fmt_pct(stat['max_drawdown'])} | {fmt_pct(stat['avg_exposure'])} | "
                f"{audit['fees']:.2f} | {audit['trade_count']} |")
    test_net = test["strategy"]["cumulative_net_return"]
    report = [
        "# V8 四大行月频 Top2：60日均值收益符号门控",
        "",
        f"- 数据快照：{panel.data_built_at}；交易数据截至 {days[-1]}。",
        "- 预先锁定规则：每月首个交易日 T 收盘，计算四只银行过去60个交易间隔的复权收盘收益；等权均值严格大于0时，下一交易日开盘买入个股动量Top2等权；均值小于等于0时，下一开盘清仓转现金。",
        "- 单一门槛为0，不扫门槛、不训练模型。买卖沿用随 V8 打包的共享现金账本：整手、1%信号日成交额容量、涨跌停锁单、3bp滑点、佣金/过户费/卖出印花税。",
        "- 同持仓匹配收益按前一执行日真实逐票持仓权重、对应调整后开盘至开盘收益逐日复利；净收益与该值之差只反映交易成本/执行摩擦，不是选股alpha。",
        "",
        "## 官方生命周期（2025-07-01至2026-07-01，共243日；信号日截止2026-06-30）",
        "",
        "| 策略 | 净收益 | 同持仓无费 | 成本/执行摩擦差 | 最大回撤 | 平均敞口 | 费用 | 成交 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("V8：60日均值符号门控 Top2", test["strategy"], test["audit"]),
        row("无门控月频 Top2（V4规则）", test["ungated_monthly_top2"], test["ungated_audit"]),
        row("首次Top2后持有", test["initial_top2_hold"], test["baseline_audit"]["initial_top2_hold"]),
        row("四股等权持有", test["four_bank_equal_weight_hold"], test["baseline_audit"]["four_bank_equal_weight_hold"]),
        "",
        f"- 官方窗门控相对无门控Top2净收益差：{test['net_gain_pp_vs_ungated_top2']:+.2f}pp；相对首次Top2持有：{test['net_gain_pp_vs_initial_top2_hold']:+.2f}pp；相对四股等权持有：{test['net_gain_pp_vs_four_bank_equal_hold']:+.2f}pp。",
        f"- 242日信号窗截至6/30开盘标记（未计7/1退出）的净收益：V8 {test['signal_window_metrics']['strategy']['cumulative_net_return']:.2%}，无门控Top2 {test['signal_window_metrics']['ungated_monthly_top2']['cumulative_net_return']:.2%}；完整生命周期以含退出费用的243日结果为准。",
        f"- 官方窗每月门控通过 {test['audit']['gate_pass_signals']}/{test['audit']['monthly_signals']} 次；回撤、收益及成本要结合门控带来的现金敞口变化解读。",
        "",
        "### 信号窗季度拆分（仅计至2026-06-30；共242日，不含7月1日退出）",
        "",
        "| 季度 | 交易日 | V8净收益 | V8同持仓无费 | V8成本/执行摩擦差 | 无门控Top2净收益 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    uq = {r["period"]: r for r in test["ungated_quarterly"]}
    for r in test["quarterly"]:
        report.append(f"| {r['period']} | {r['days']} | {r['net_return']:.2%} | {r['matched_return']:.2%} | {r['matched_excess']:.2%} | {uq[r['period']]['net_return']:.2%} |")
    ex = test["exit_day"]
    report += [
        "",
        "### 2026-07-01退出日（无新信号）",
        "",
        "| 策略 | 退出日净收益 | 同持仓无费收益 | 成本/执行摩擦差 | 退出费用 | 成交 | 日终持仓 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| V8门控 | {ex['gated']['net_return']:.2%} | {ex['gated']['matched_no_fee_return']:.2%} | {ex['gated']['matched_excess']:.2%} | {ex['gated']['fees']:.2f} | {ex['gated']['trade_count']} | {ex['gated']['ending_holdings']} |",
        f"| 无门控Top2 | {ex['ungated_top2']['net_return']:.2%} | {ex['ungated_top2']['matched_no_fee_return']:.2%} | {ex['ungated_top2']['matched_excess']:.2%} | {ex['ungated_top2']['fees']:.2f} | {ex['ungated_top2']['trade_count']} | {ex['ungated_top2']['ending_holdings']} |",
        "",
        "退出日收益是6月30日持仓至7月1日开盘退出的单日变化，单独列示，不并入上面的信号窗季度表；7月1日未生成新的月初信号。",
        "",
        "## 2023-01-03至最新扩展窗",

        "",
        "| 年份 | 交易日 | V8净收益 | V8同持仓无费 | V8成本/执行摩擦差 | 无门控Top2净收益 | 无门控Top2同持仓无费差 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    uyear = {r["year"]: r for r in full["ungated_yearly"]}
    for r in full["yearly"]:
        uy = uyear[r["year"]]
        report.append(f"| {r['year']} | {r['days']} | {r['net_return']:.2%} | {r['matched_return']:.2%} | {r['matched_excess']:.2%} | {uy['net_return']:.2%} | {uy['matched_excess']:.2%} |")
    report += [
        "",
        "### 与既有SMA趋势门控的关系",
        "",
        "既有研究已在四银行等权总回报篮子上测试 SMA63/126/252 × 50%/100%目标敞口，六种配置的严格匹配超额全部为负（-0.71至-5.63pp）。其中 SMA252/50%净收益+63.59%、匹配基准+64.30%、匹配差-0.71pp、最大回撤-10.80%；买持净收益+149.55%、回撤-15.49%，静态半仓净收益+74.25%、回撤-10.98%。该结论是风险缩放没有带来净匹配收益改善。",
        "V8改用单一60日等权平均收益正负号决定是否持有个股动量Top2，和SMA篮子/敞口缩放门控属于相近的趋势过滤思路；没有依据称为新颖方法。V8扩展窗含截至2026-09-24的交易日，既有SMA报告的信号窗记至09-23、执行至09-24，窗口口径略有差别。",
        "",
        "扩展窗最后一日为2026-09-24，账本在当日开盘强制清仓；未延伸到快照之后的T+1交易日。",
        "",
        "## 账本审计与结论",
        "",
        f"- 官方243日生命周期审计（含7/1退出）：现金最低 {test['audit']['cash_min']:.4f}；最大持仓数 {test['audit']['maximum_simultaneous_holdings']}；结束持仓 {test['audit']['ending_positions']}；买入/卖出阻塞 {test['audit']['blocked_entries']}/{test['audit']['blocked_exits']}；最大权益-现金-持仓市值误差 {test['audit']['max_abs_equity_cash_plus_engine_mark_residual']:.3g}。",
        f"- 扩展窗审计：现金最低 {full['audit']['cash_min']:.4f}；最大持仓数 {full['audit']['maximum_simultaneous_holdings']}；结束持仓 {full['audit']['ending_positions']}；买入/卖出阻塞 {full['audit']['blocked_entries']}/{full['audit']['blocked_exits']}；最大权益-现金-持仓市值误差 {full['audit']['max_abs_equity_cash_plus_engine_mark_residual']:.3g}。",
        "- V8只是固定单门槛候选；同持仓差是成本指标，比较无门控Top2/持有基准的收益差也不能单独证明可持续alpha。按实际收益、回撤和跨年/季度表现判断是否值得继续。",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("V8_RESULTS", OUT / "result.json", flush=True)
    print("official 243d lifecycle gated net/matched/mdd", test["strategy"]["cumulative_net_return"],
          test["strategy"]["matched_benchmark_return"], test["strategy"]["max_drawdown"], flush=True)
    print("extended yearly", full["yearly"], flush=True)
    print("elapsed", time.time() - started, "seconds", flush=True)


if __name__ == "__main__":
    main()

