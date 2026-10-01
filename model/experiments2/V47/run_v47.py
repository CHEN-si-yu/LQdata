#!/usr/bin/env python3
"""V47: frozen V34 Top2 membership with odd-month-only target updates."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"

import json
import re
import resource
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V11 = ROOT / "V11"
V34 = ROOT / "V34"
DATA = V11.parents[1] / "trainingdata"
START, END = "2024-01-02", "2026-06-30"
FEATURE = "id2_close_vs_pm_vwap_20"
CONTROL = "V34 id2_close_vs_pm_vwap_20 Top2"
CANDIDATE = "V47 id2_close_vs_pm_vwap_20 Top2 bimonthly"
PROTOCOL_SHA256 = "5920cfcd8b79c7620180eec73508130e4af0e8034756c0e48aec8fce3ff2da7d"
MEMORY_CAP_GIB = 180
DAILY_FIELDS = ["equity_net", "equity_gross", "cash", "fees_cumulative",
                "slippage_cumulative", "turnover_cumulative", "accounting_residual",
                "daily_return_net", "drawdown"]


def host_snapshot():
    output = subprocess.run(["top", "-bn1"], check=True, capture_output=True, text=True).stdout
    cpu_line = next((line for line in output.splitlines() if "%Cpu" in line), "")
    match = re.search(r"([0-9.]+)\s+id", cpu_line)
    if not match:
        raise RuntimeError(f"could not parse host CPU snapshot: {cpu_line}")
    idle = float(match.group(1))
    loads = Path("/proc/loadavg").read_text().split()[:3]
    cpus = os.cpu_count() or 1
    meminfo = Path("/proc/meminfo").read_text()
    available_kib = int(re.search(r"^MemAvailable:\s+(\d+)", meminfo, re.M).group(1))
    return {
        "host_cpu_utilization_pct": 100.0 - idle,
        "host_cpu_idle_pct": idle,
        "host_load_1m_5m_15m": [float(value) for value in loads],
        "host_logical_cpus": cpus,
        "host_load_1m_per_cpu": float(loads[0]) / cpus,
        "host_available_memory_gib": available_kib / 1024**2,
    }


# Apply a hard process address-space ceiling before importing NumPy or pandas.
requested_limit = MEMORY_CAP_GIB * 1024**3
old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
hard_limit = requested_limit if old_hard == resource.RLIM_INFINITY else min(requested_limit, old_hard)
resource.setrlimit(resource.RLIMIT_AS, (hard_limit, hard_limit))
resource_gate = host_snapshot()
if resource_gate["host_cpu_utilization_pct"] >= 50.0:
    raise RuntimeError(
        f"host CPU utilization is {resource_gate['host_cpu_utilization_pct']:.1f}%, "
        "at or above the frozen 50% launch gate; no waiting or run performed"
    )

import hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(V11))
import analysis as E
sys.path.insert(0, str(V34))
import run_v34 as V34Runner


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_comparison(candidate: dict, control: dict) -> dict:
    return {
        "net_return_difference_pp": 100.0 * (candidate["net_return"] - control["net_return"]),
        "annualized_return_difference_pp": 100.0 * (candidate["annualized_net_return"] - control["annualized_net_return"]),
        "max_drawdown_difference_pp": 100.0 * (candidate["max_drawdown"] - control["max_drawdown"]),
        "annualized_volatility_difference_pp": 100.0 * (candidate["annualized_volatility"] - control["annualized_volatility"]),
        "turnover_reduction_fraction": 1.0 - candidate["turnover"] / control["turnover"],
        "total_cost_savings_fraction": 1.0 - candidate["total_trading_cost"] / control["total_trading_cost"],
        "fees_savings_fraction": 1.0 - candidate["fees"] / control["fees"] if control["fees"] else None,
        "slippage_savings_fraction": 1.0 - candidate["slippage_cost"] / control["slippage_cost"] if control["slippage_cost"] else None,
        "return_over_volatility_difference": (
            candidate["annualized_net_return"] / candidate["annualized_volatility"]
            - control["annualized_net_return"] / control["annualized_volatility"]
        ),
    }


def peak_rss_gib() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def main() -> None:
    protocol_path = HERE / "protocol.json"
    protocol = read_json(protocol_path)
    if sha256(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("frozen V47 protocol hash changed; refusing to run")
    if protocol.get("version") != "V47" or protocol.get("no_frequency_change_after_results") is not True:
        raise RuntimeError("V47 protocol is missing required freeze fields")
    if protocol.get("experiment_status", {}).get("independent_unseen_holdout") is not False:
        raise RuntimeError("V47 must remain explicitly exploratory, not an unseen holdout")
    if protocol.get("holdout_dates_inclusive") != [START, END]:
        raise RuntimeError("runner window differs from frozen V47 protocol")

    source_artifacts = protocol["source_artifacts"]
    source_checks = {
        "v34_protocol_sha256": V34 / "protocol.json",
        "v34_runner_sha256": V34 / "run_v34.py",
        "v34_membership_audit_sha256": V34 / "membership_audit.json",
        "v34_metrics_sha256": V34 / "metrics.json",
        "v34_daily_equity_sha256": V34 / "daily_equity.csv",
        "v34_trades_sha256": V34 / "trades.csv",
        "v11_corrected_ledger_sha256": V11 / "analysis.py",
    }
    source_hashes = {key: sha256(path) for key, path in source_checks.items()}
    if source_hashes != source_artifacts:
        raise RuntimeError(f"frozen V34/V11 source artifacts changed: {source_hashes}")

    # Confirm the pre-holdout V34 factor-selection rule, without reselecting anything.
    v34_protocol = read_json(V34 / "protocol.json")
    features, importance_quarters, importance = V34Runner.choose_features()
    if (features != [FEATURE] or len(importance_quarters) != 16
            or importance[FEATURE]["top20_quarters"] != 10
            or v34_protocol["factor_selection"]["cutoff_quarter"] != "2023Q4"):
        raise RuntimeError("frozen V34 factor-selection record does not match its protocol")

    frozen_rows = read_json(V34 / "membership_audit.json")
    monthly_dates = [row["signal_date"] for row in frozen_rows]
    if monthly_dates != protocol["shared_v34_monthly_signal_dates"]:
        raise RuntimeError("V34 frozen membership dates differ from the V47 protocol")
    if len(monthly_dates) != 30 or len({row["signal_date"] for row in frozen_rows}) != 30:
        raise RuntimeError("V34 membership audit must contain exactly 30 unique monthly signals")

    update_dates = protocol["bimonthly_update_signal_dates"]
    expected_odd_dates = [
        date for date in monthly_dates if int(date[5:7]) % 2 == 1
    ]
    if update_dates != expected_odd_dates or len(update_dates) != 15:
        raise RuntimeError("odd-month update schedule differs from the frozen V47 rule")

    monthly_schedule = {}
    rows_by_date = {}
    for row in frozen_rows:
        date = row["signal_date"]
        picked = row["selected_top2"]
        if len(picked) != 2 or len(set(picked)) != 2 or not set(picked).issubset(E.CODES):
            raise RuntimeError(f"invalid frozen V34 Top2 membership on {date}: {picked}")
        if row.get("factor_name") != FEATURE or row.get("factor_timestamp_is_signal_date") is not True:
            raise RuntimeError(f"V34 membership audit is missing same-day factor evidence on {date}")
        monthly_schedule[date] = {code: 0.5 for code in picked}
        rows_by_date[date] = row

    prices = E.load_prices()
    price_index = prices["index"]
    expected_execution_dates = [
        str(prices["days"][price_index[date] + 1]) for date in update_dates
    ]
    if expected_execution_dates != protocol["bimonthly_update_execution_dates"]:
        raise RuntimeError("odd-month T+1 execution dates differ from the frozen V47 protocol")
    if any(price_index[date] + 1 > price_index[END] for date in update_dates):
        raise RuntimeError("an odd-month rebalance has no execution session inside the window")

    candidate_schedule = {date: monthly_schedule[date] for date in update_dates}
    control_daily, control_trades, control_metric = E.simulate(
        CONTROL, prices, monthly_schedule, START, END
    )
    candidate_daily, candidate_trades, candidate_metric = E.simulate(
        CANDIDATE, prices, candidate_schedule, START, END
    )
    daily_all = pd.concat([control_daily, candidate_daily], ignore_index=True)
    daily_all["gross_return_daily"] = daily_all.groupby(
        "strategy", sort=False
    ).equity_gross.pct_change().fillna(0.0)
    metrics = {CONTROL: control_metric, CANDIDATE: candidate_metric}
    trades_all = pd.DataFrame(control_trades + candidate_trades)

    # Replay the V34 monthly strategy and require its ledger to match the frozen output.
    v34_metrics = read_json(V34 / "metrics.json")["metrics"][CONTROL]
    v34_daily = pd.read_csv(V34 / "daily_equity.csv", dtype={"trade_date": str})
    v34_trades = pd.read_csv(V34 / "trades.csv", dtype={"trade_date": str})
    old_daily = v34_daily.loc[v34_daily.strategy == CONTROL].sort_values("trade_date").reset_index(drop=True)
    new_daily = control_daily.sort_values("trade_date").reset_index(drop=True)
    if old_daily.trade_date.tolist() != new_daily.trade_date.tolist() or len(old_daily) != 601:
        raise RuntimeError("monthly control has different coverage from frozen V34")
    daily_diffs = {
        field: float(np.max(np.abs(
            old_daily[field].to_numpy(dtype=float) - new_daily[field].to_numpy(dtype=float)
        )))
        for field in DAILY_FIELDS
    }
    shares_match = old_daily["shares_json"].astype(str).tolist() == new_daily["shares_json"].astype(str).tolist()
    metric_fields = (
        "net_return", "annualized_net_return", "fees", "slippage_cost", "total_trading_cost",
        "turnover", "max_drawdown", "annualized_volatility", "min_cash",
    )
    metric_diffs = {
        field: abs(float(metrics[CONTROL][field]) - float(v34_metrics[field]))
        for field in metric_fields
    }
    old_control_trades = v34_trades.loc[v34_trades.strategy == CONTROL].reset_index(drop=True)
    replay_trades = pd.DataFrame(control_trades).reset_index(drop=True)
    if len(old_control_trades) != len(replay_trades):
        raise RuntimeError("monthly control trade count differs from frozen V34")
    trade_string_fields = ("trade_date", "strategy", "stock_code", "side", "reason")
    trade_numeric_fields = ("shares", "open", "fill", "notional", "fee", "slippage_cost")
    trade_strings_match = all(
        old_control_trades[field].astype(str).tolist() == replay_trades[field].astype(str).tolist()
        for field in trade_string_fields
    )
    trade_numeric_diffs = {
        field: float(np.max(np.abs(
            old_control_trades[field].to_numpy(dtype=float) - replay_trades[field].to_numpy(dtype=float)
        )))
        for field in trade_numeric_fields
    }
    tolerance = 1e-3
    if (not trade_strings_match or max(trade_numeric_diffs.values()) > tolerance
            or not shares_match or max(daily_diffs.values()) > tolerance
            or max(metric_diffs.values()) > 1e-9):
        raise RuntimeError("V34 monthly control replay does not match the frozen V34 ledger")

    candidate_trade_dates = sorted({str(trade["trade_date"]) for trade in candidate_trades})
    only_update_execution_dates = set(candidate_trade_dates).issubset(set(expected_execution_dates))
    if not only_update_execution_dates:
        raise RuntimeError("V47 candidate traded outside odd-month T+1 execution dates")

    # Make every odd/even monthly signal's use (or carry-forward) explicit.
    membership_audit = []
    current_target = None
    for date in monthly_dates:
        row = dict(rows_by_date[date])
        update = date in update_dates
        if update:
            current_target = list(row["selected_top2"])
        if current_target is None:
            raise RuntimeError("V47 has no initial target before its first even-month signal")
        row["v47_action"] = "update_target" if update else "carry_forward_target"
        row["v47_target_top2"] = list(current_target)
        row["v47_target_weights"] = {code: 0.5 for code in current_target}
        row["v47_signal_changes_target"] = update
        row["v47_execution_date"] = str(prices["days"][price_index[date] + 1]) if update else ""
        row["v47_no_order_on_even_month_signal"] = not update
        row["v47_frozen_v34_membership_reused"] = True
        row["v47_initial_entry"] = date == update_dates[0]
        membership_audit.append(row)
    if [row["v47_execution_date"] for row in membership_audit if row["v47_execution_date"]] != expected_execution_dates:
        raise RuntimeError("membership audit execution rows differ from the frozen odd-month schedule")

    candidate_metric["return_over_volatility"] = (
        candidate_metric["annualized_net_return"] / candidate_metric["annualized_volatility"]
    )
    control_metric["return_over_volatility"] = (
        control_metric["annualized_net_return"] / control_metric["annualized_volatility"]
    )
    comparison = metric_comparison(candidate_metric, control_metric)
    metrics_output = {
        "window": [START, END],
        "days_per_strategy": int(len(control_daily)),
        "monthly_signal_count": len(monthly_dates),
        "bimonthly_update_count": len(update_dates),
        "selected_feature": FEATURE,
        "feature_selection": importance,
        "metrics": metrics,
        "bimonthly_vs_monthly": comparison,
        "v34_monthly_control_compatibility": {
            "reference": "V34 daily_equity.csv, trades.csv, metrics.json",
            "daily_rows": len(new_daily),
            "daily_max_abs_differences": daily_diffs,
            "metric_abs_differences": metric_diffs,
            "shares_json_exact_match": shares_match,
            "trade_string_fields_exact_match": trade_strings_match,
            "trade_numeric_max_abs_differences": trade_numeric_diffs,
            "trades_equal": len(old_control_trades) == len(replay_trades),
            "tolerance": tolerance,
        },
    }

    resource_usage = {
        **resource_gate,
        "process_address_space_cap_gib": hard_limit / 1024**3,
        "process_peak_rss_gib": peak_rss_gib(),
        "process_peak_rss_under_180_gib": peak_rss_gib() <= MEMORY_CAP_GIB,
        "host_cpu_gate_passed": resource_gate["host_cpu_utilization_pct"] < 50.0,
    }
    if not resource_usage["process_peak_rss_under_180_gib"]:
        raise RuntimeError("V47 process peak RSS exceeded the 180 GiB cap")

    quarterly_rows, annual_rows = [], []
    for _, frame in daily_all.groupby("strategy", sort=False):
        quarterly_rows.extend(E.period_returns(frame, "Q"))
        annual_rows.extend(E.period_returns(frame, "Y"))

    audit = {
        "window": [START, END],
        "exploratory": True,
        "independent_unseen_holdout": False,
        "motivation": "V41 quarterly-rebalance result",
        "monthly_signal_count": len(monthly_dates),
        "bimonthly_update_signal_dates": update_dates,
        "bimonthly_update_execution_dates": expected_execution_dates,
        "only_odd_month_signals_update_target": True,
        "even_month_signal_count_carried_forward": len(monthly_dates) - len(update_dates),
        "even_month_signals_place_no_orders": all(row["v47_no_order_on_even_month_signal"] for row in membership_audit if int(row["signal_date"][5:7]) % 2 == 0),
        "frozen_v34_membership_reused_for_all_updates": all(row["v47_frozen_v34_membership_reused"] for row in membership_audit),
        "initial_entry_is_january_2024_top2_equal_50_50": update_dates[0] == START and membership_audit[0]["v47_initial_entry"],
        "all_updates_execute_next_session_open": True,
        "candidate_trade_dates_are_odd_month_executions_only": only_update_execution_dates,
        "positions_marked_to_2026_06_30_close_without_forced_liquidation": True,
        "factor_pre_holdout_selection_record": {
            "factor": FEATURE,
            "selection_cutoff": "2023Q4",
            "top20_quarters": importance[FEATURE]["top20_quarters"],
            "total_quarters": len(importance_quarters),
        },
        "ledger_source": str((V11 / "analysis.py").resolve()),
        "ledger_cost_parameters": {
            "initial_equity": E.INITIAL,
            "commission_rate": E.COMMISSION,
            "minimum_commission": E.MIN_COMMISSION,
            "transfer_fee_rate": E.TRANSFER,
            "sell_stamp_duty_rate": E.STAMP,
            "adverse_slippage_rate": E.SLIPPAGE,
            "maximum_participation": E.PARTICIPATION,
            "lot_size": E.LOT,
        },
        "v34_monthly_replay": metrics_output["v34_monthly_control_compatibility"],
        "strategies": {
            name: {
                "cash_nonnegative": metric["min_cash"] >= -1e-7,
                "blocked_trade_events": metric["blocked_trade_events"],
                "max_abs_accounting_residual": metric["max_abs_accounting_residual"],
                "trades": metric["trades"],
                "turnover": metric["turnover"],
                "fees": metric["fees"],
                "slippage_cost": metric["slippage_cost"],
                "total_trading_cost": metric["total_trading_cost"],
                "min_cash": metric["min_cash"],
            }
            for name, metric in metrics.items()
        },
        "resource_usage": resource_usage,
    }

    (HERE / "membership_audit.json").write_text(
        json.dumps(membership_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    daily_all.to_csv(HERE / "daily_equity.csv", index=False, float_format="%.10g")
    trades_all.to_csv(HERE / "trades.csv", index=False, float_format="%.10g")
    pd.DataFrame(quarterly_rows).to_csv(HERE / "quarterly_returns.csv", index=False, float_format="%.10g")
    pd.DataFrame(annual_rows).to_csv(HERE / "annual_returns.csv", index=False, float_format="%.10g")
    (HERE / "metrics.json").write_text(
        json.dumps(metrics_output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (HERE / "resource_usage.json").write_text(
        json.dumps(resource_usage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    cache_hashes = []
    def add_hash(path: Path, role: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        cache_hashes.append({
            "role": role, "path": str(path.resolve()),
            "size_bytes": path.stat().st_size, "sha256": sha256(path),
        })
    add_hash(protocol_path, "frozen V47 protocol")
    add_hash(Path(__file__), "V47 runner")
    add_hash(V11 / "analysis.py", "V11 final corrected shared-cash ledger")
    add_hash(V34 / "protocol.json", "frozen V34 protocol")
    add_hash(V34 / "run_v34.py", "frozen V34 factor implementation")
    add_hash(V34 / "membership_audit.json", "frozen V34 monthly membership audit")
    add_hash(V34 / "metrics.json", "frozen V34 monthly metrics")
    add_hash(V34 / "daily_equity.csv", "frozen V34 monthly daily ledger")
    add_hash(V34 / "trades.csv", "frozen V34 monthly trades")
    add_hash(V34 / "cache_hashes.json", "V34 source cache hash manifest")
    add_hash(DATA / "meta.json", "training data cache metadata")
    (HERE / "cache_hashes.json").write_text(json.dumps({
        "algorithm": "SHA-256",
        "protocol_sha256": PROTOCOL_SHA256,
        "scope": "V47 frozen odd-month rule; reused V34 factor membership and monthly reference outputs; V11 corrected ledger source; V34 cache-hash manifest for source market, factor, and schedule inputs.",
        "hashes": cache_hashes,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pct(value):
        return f"{value:.2%}"
    table = [
        "| Strategy | Net return | Annualized | MDD | Ann. vol. | Fees | Slippage | Total cost | Turnover | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in metrics.items():
        table.append(
            f"| {name} | {pct(metric['net_return'])} | {pct(metric['annualized_net_return'])} "
            f"| {pct(metric['max_drawdown'])} | {pct(metric['annualized_volatility'])} "
            f"| {metric['fees']:.2f} | {metric['slippage_cost']:.2f} | "
            f"{metric['total_trading_cost']:.2f} | {metric['turnover']:.0f} | {metric['trades']} |"
        )
    report = [
        "# V47：V34 Top2 奇数月更新、偶数月沿用",
        "",
        "## 研究性质",
        "",
        "本实验受 V41 季度调仓结果启发，属于探索性频率比较。它复用 V34 的 2024-01-02 至 2026-06-30 同一历史窗口和已冻结成员，不是独立未见 holdout，不能作为独立样本验证。",
        "",
        "## 冻结规则",
        "",
        "从 2024 年 1 月开始，只在奇数月的首个共同月初信号收盘更新目标持仓，沿用 V34 当日冻结 Top2，等权各 50%，次一交易日开盘执行。偶数月信号不下单，沿用上一个奇数月目标。首笔为 2024-01-02 信号；规则在查看 V47 结果前固定，之后不改频率或参数。",
        "",
        "控制组按 V34 的全部 30 个共同月初信号逐月调仓。两组使用同一 V11 corrected ledger，窗口末按 2026-06-30 收盘估值，不强制平仓。",
        "",
        "## 结果",
        "",
        *table,
        "",
        "与 V34 月调相比，V47 净收益差 {:+.2f} 个百分点，年化收益差 {:+.2f} 个百分点，最大回撤差 {:+.2f} 个百分点，年化波动差 {:+.2f} 个百分点；总成本节省 {:.1%}，换手减少 {:.1%}。".format(
            comparison["net_return_difference_pp"],
            comparison["annualized_return_difference_pp"],
            comparison["max_drawdown_difference_pp"],
            comparison["annualized_volatility_difference_pp"],
            comparison["total_cost_savings_fraction"],
            comparison["turnover_reduction_fraction"],
        ),
        "",
        "## 审计与文件",
        "",
        f"- 使用 {len(update_dates)} 个奇数月调仓信号；其余 {len(monthly_dates) - len(update_dates)} 个偶数月信号明确沿用既有目标。",
        "- V34 月调控制组逐日权益、现金、持仓、交易和指标均与冻结结果对账；详情见 audit.json。",
        "- 逐月冻结成员与 V47 更新/沿用记录见 membership_audit.json；完整权益、成交和收益明细见 daily_equity.csv、trades.csv、quarterly_returns.csv、annual_returns.csv。",
        "- V11 corrected ledger、冻结输入及源代码 SHA-256 见 cache_hashes.json；CPU 门槛、180 GiB 地址空间硬限制与进程峰值 RSS 见 resource_usage.json。",
        "",
        "结果只描述这四家银行和同一历史窗口的探索性比较，不构成独立留出验证或未来收益承诺。",
    ]
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(HERE),
        "protocol_sha256": PROTOCOL_SHA256,
        "monthly_signal_count": len(monthly_dates),
        "bimonthly_update_signal_dates": update_dates,
        "metrics": metrics,
        "bimonthly_vs_monthly": comparison,
        "v34_monthly_control_compatibility": metrics_output["v34_monthly_control_compatibility"],
        "resource_usage": resource_usage,
        "audit": audit,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

