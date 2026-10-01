#!/usr/bin/env python3
"""V41: frozen quarterly rebalance replay of the V34 single-factor Top2 rule."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V11 = ROOT / "V11"
V34 = ROOT / "V34"
DATA = V11.parents[1] / "trainingdata"
START, END = "2024-01-02", "2026-06-30"
PROTOCOL_SHA256 = "ae10486dd10f870ce9f28ac89905f920e854a1e3e82183a44253af8efcc2d72f"
FEATURE = "id2_close_vs_pm_vwap_20"
CONTROL = "V34 id2_close_vs_pm_vwap_20 Top2"
CANDIDATE = "V41 id2_close_vs_pm_vwap_20 Top2 quarterly"
DAILY_FIELDS = ["equity_net", "equity_gross", "cash", "fees_cumulative",
                "slippage_cumulative", "turnover_cumulative", "accounting_residual",
                "daily_return_net", "drawdown"]

sys.path.insert(0, str(V11))
import analysis as E  # noqa: E402
sys.path.insert(0, str(V34))
import run_v34 as V34Runner  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def first_shared_signal_each_quarter(signal_dates: list[str]) -> list[str]:
    by_quarter: dict[str, list[str]] = {}
    for date in signal_dates:
        quarter = str(pd.Timestamp(date).to_period("Q"))
        by_quarter.setdefault(quarter, []).append(date)
    return [min(dates) for _, dates in sorted(by_quarter.items())]


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
        "return_over_volatility_difference": candidate["annualized_net_return"] / candidate["annualized_volatility"] - control["annualized_net_return"] / control["annualized_volatility"],
        "calmar_difference": candidate["annualized_net_return"] / abs(candidate["max_drawdown"]) - control["annualized_net_return"] / abs(control["max_drawdown"]),
    }


def main() -> None:
    protocol_path = HERE / "protocol.json"
    protocol = read_json(protocol_path)
    if sha256(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol hash changed; refusing to run")
    if protocol.get("version") != "V41" or protocol.get("no_frequency_change_after_results") is not True:
        raise RuntimeError("V41 protocol is missing required freeze fields")
    if [START, END] != protocol["holdout_dates_inclusive"]:
        raise RuntimeError("runner window differs from frozen protocol")

    # V34's factor and ranking policy are already frozen. Reuse that implementation,
    # and validate its pre-holdout selection rule before loading any holdout results.
    features, importance_quarters, importance = V34Runner.choose_features()
    if features != [FEATURE] or len(importance_quarters) != 16 or importance[FEATURE]["top20_quarters"] != 10:
        raise RuntimeError("V34 factor freeze does not match the preregistered source rule")

    prices = E.load_prices()
    quarter_dirs = sorted(path for path in (V11 / "quarters").iterdir()
                          if path.is_dir() and (path / "predictions.csv").is_file())
    quarters = [path.name for path in quarter_dirs]
    pred = E.load_predictions(quarters)
    all_schedules, _ = E.schedules(pred, prices)
    monthly_dates = sorted(date for date in pred.loc[pred.monthly_signal, "trade_date"].unique()
                           if START <= date <= END)
    if monthly_dates != protocol["shared_monthly_signal_dates"]:
        raise RuntimeError("shared V34 monthly signal dates differ from frozen protocol")
    quarterly_dates = first_shared_signal_each_quarter(monthly_dates)
    if quarterly_dates != protocol["quarterly_rebalance_signal_dates"]:
        raise RuntimeError("quarterly signal dates differ from frozen protocol")

    factors, factor_meta = V34Runner.load_factor_rows(features)
    if factor_meta.get("semantics") != protocol["source_strategy"]["factor_semantics"]:
        raise RuntimeError("factor semantics differ from frozen protocol")
    factor_by_date = {date: frame.set_index("stock_code")
                      for date, frame in factors.groupby("trade_date", sort=False)}
    factor_schedule, factor_audit = V34Runner.build_factor_schedule(factors, monthly_dates, features)
    monthly_schedule = {CONTROL: factor_schedule}
    quarterly_schedule = {CANDIDATE: {date: factor_schedule[date] for date in quarterly_dates}}

    # Build signal-close and next-session execution evidence for all common monthly dates.
    price_index = prices["index"]
    audit_by_date = {row["signal_date"]: row for row in factor_audit}
    for date in monthly_dates:
        if date not in factor_by_date:
            raise RuntimeError(f"missing factor rows for signal date {date}")
        day = factor_by_date[date].reindex(E.CODES)
        raw = day[FEATURE].astype(float)
        missing_codes = [code for code in E.CODES if not np.isfinite(raw.loc[code])]
        values = raw.fillna(0.0)
        row = audit_by_date[date]
        row["factor_name"] = FEATURE
        row["factor_values_at_signal_close"] = {code: float(values.loc[code]) for code in E.CODES}
        row["factor_missing_values_filled_with_zero"] = len(missing_codes)
        row["factor_timestamp_is_signal_date"] = True
        row["used_for_quarterly_rebalance"] = date in quarterly_dates
        row["execution_date"] = str(prices["days"][price_index[date] + 1]) if date in quarterly_dates else ""
        row["no_future_row_used"] = (date not in quarterly_dates or row["execution_date"] > date)
        row["monthly_signal_ignored_until_next_quarter"] = date not in quarterly_dates
        row["first_entry_50_50"] = date == quarterly_dates[0]
    membership_audit = [audit_by_date[date] for date in monthly_dates]

    control_daily, control_trades, control_metric = E.simulate(CONTROL, prices, factor_schedule, START, END)
    candidate_daily, candidate_trades, candidate_metric = E.simulate(
        CANDIDATE, prices, quarterly_schedule[CANDIDATE], START, END)
    daily_all = pd.concat([control_daily, candidate_daily], ignore_index=True)
    daily_all["gross_return_daily"] = daily_all.groupby("strategy", sort=False).equity_gross.pct_change().fillna(0.0)
    metrics = {CONTROL: control_metric, CANDIDATE: candidate_metric}
    trades_all = pd.DataFrame(control_trades + candidate_trades)

    # The monthly control must reproduce the frozen V34 result on the shared window.
    v34_metrics = read_json(V34 / "metrics.json")["metrics"][CONTROL]
    v34_daily_all = pd.read_csv(V34 / "daily_equity.csv", dtype={"trade_date": str})
    v34_trades_all = pd.read_csv(V34 / "trades.csv", dtype={"trade_date": str})
    old_daily = v34_daily_all.loc[v34_daily_all.strategy == CONTROL].sort_values("trade_date").reset_index(drop=True)
    new_daily = control_daily.sort_values("trade_date").reset_index(drop=True)
    if old_daily.trade_date.tolist() != new_daily.trade_date.tolist() or len(old_daily) != 601:
        raise RuntimeError("monthly control has different V34 daily coverage")
    daily_diffs = {
        field: float(np.max(np.abs(old_daily[field].to_numpy(dtype=float) - new_daily[field].to_numpy(dtype=float))))
        for field in DAILY_FIELDS
    }
    shares_match = old_daily["shares_json"].astype(str).tolist() == new_daily["shares_json"].astype(str).tolist()
    metric_fields = ("net_return", "fees", "slippage_cost", "turnover", "max_drawdown", "annualized_volatility")
    metric_diffs = {field: abs(float(metrics[CONTROL][field]) - float(v34_metrics[field])) for field in metric_fields}
    trade_control = v34_trades_all.loc[v34_trades_all.strategy == CONTROL].reset_index(drop=True)
    trade_replay = pd.DataFrame(control_trades).reset_index(drop=True)
    if len(trade_control) != len(trade_replay):
        raise RuntimeError("monthly control trade count differs from frozen V34")
    trade_string_fields = ("trade_date", "strategy", "stock_code", "side", "reason")
    trade_numeric_fields = ("shares", "open", "fill", "notional", "fee", "slippage_cost")
    trade_strings_match = all(trade_control[field].astype(str).tolist() == trade_replay[field].astype(str).tolist()
                              for field in trade_string_fields)
    trade_numeric_diffs = {
        field: float(np.max(np.abs(trade_control[field].to_numpy(dtype=float) - trade_replay[field].to_numpy(dtype=float))))
        for field in trade_numeric_fields
    }
    candidate_execution_dates = sorted({str(prices["days"][price_index[date] + 1]) for date in quarterly_dates
                                         if price_index[date] + 1 <= price_index[END]})
    candidate_trade_dates = sorted({trade["trade_date"] for trade in candidate_trades})
    no_trades_between_quarters = set(candidate_trade_dates).issubset(set(candidate_execution_dates))
    if not trade_strings_match or max(trade_numeric_diffs.values()) > 1e-3:
        raise RuntimeError("monthly control trade rows differ from frozen V34")
    if not no_trades_between_quarters:
        raise RuntimeError("quarterly candidate traded outside a quarterly T+1 execution date")
    if not shares_match or max(daily_diffs.values()) > 1e-3 or max(metric_diffs.values()) > 1e-9:
        raise RuntimeError("monthly control replay is incompatible with frozen V34 output")

    reference_compatibility = {
        "reference": "V34/daily_equity.csv, V34/trades.csv, V34/metrics.json",
        "daily_rows": len(new_daily),
        "daily_max_abs_differences": daily_diffs,
        "metric_abs_differences": metric_diffs,
        "shares_json_exact_match": shares_match,
        "trade_string_fields_exact_match": trade_strings_match,
        "trade_numeric_max_abs_differences": trade_numeric_diffs,
        "trades_equal": len(trade_control) == len(trade_replay),
        "tolerance": 1e-3,
    }
    metrics[CANDIDATE]["return_over_volatility"] = candidate_metric["annualized_net_return"] / candidate_metric["annualized_volatility"]
    metrics[CONTROL]["return_over_volatility"] = control_metric["annualized_net_return"] / control_metric["annualized_volatility"]
    metrics[CANDIDATE]["calmar_ratio"] = candidate_metric["annualized_net_return"] / abs(candidate_metric["max_drawdown"])
    metrics[CONTROL]["calmar_ratio"] = control_metric["annualized_net_return"] / abs(control_metric["max_drawdown"])
    comparison = metric_comparison(candidate_metric, control_metric)

    quarterly_rows, annual_rows = [], []
    for name, frame in daily_all.groupby("strategy", sort=False):
        quarterly_rows.extend(E.period_returns(frame, "Q"))
        annual_rows.extend(E.period_returns(frame, "Y"))

    metrics_output = {
        "window": [START, END],
        "days_per_strategy": int(len(control_daily)),
        "monthly_signal_count": len(monthly_dates),
        "quarterly_rebalance_count": len(quarterly_dates),
        "selected_feature": FEATURE,
        "feature_selection": importance,
        "metrics": metrics,
        "quarterly_vs_monthly": comparison,
        "v34_monthly_control_compatibility": reference_compatibility,
    }
    audit = {
        "window": [START, END],
        "monthly_signal_count": len(monthly_dates),
        "quarterly_rebalance_signal_dates": quarterly_dates,
        "only_first_shared_signal_each_quarter_used": True,
        "signals_ignored_between_rebalances": len(monthly_dates) - len(quarterly_dates),
        "factor_selected_before_holdout": max(importance_quarters) <= "2023Q4",
        "factor_training_top20_count": importance[FEATURE]["top20_quarters"],
        "all_factor_rows_on_monthly_signal_dates_available": True,
        "signal_uses_same_day_close_only": all(row["factor_timestamp_is_signal_date"] for row in membership_audit),
        "quarterly_execution_is_next_session_open": all(row["no_future_row_used"] for row in membership_audit),
        "quarterly_execution_dates": candidate_execution_dates,
        "candidate_trade_dates_are_quarterly_executions_only": no_trades_between_quarters,
        "initial_entry_is_first_signal_top2_equal_50_50": quarterly_dates[0] == START and membership_audit[0]["first_entry_50_50"],
        "positions_held_between_quarterly_signals": no_trades_between_quarters,
        "positions_marked_to_final_close_without_forced_liquidation": True,
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
        "v34_monthly_replay": reference_compatibility,
        "strategies": {
            name: {
                "cash_nonnegative": metric["min_cash"] >= -1e-7,
                "blocked_trade_events": metric["blocked_trade_events"],
                "max_abs_accounting_residual": metric["max_abs_accounting_residual"],
                "trades": metric["trades"],
                "turnover": metric["turnover"],
                "min_cash": metric["min_cash"],
            }
            for name, metric in metrics.items()
        },
    }

    (HERE / "membership_audit.json").write_text(json.dumps(membership_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    daily_all.to_csv(HERE / "daily_equity.csv", index=False, float_format="%.10g")
    trades_all.to_csv(HERE / "trades.csv", index=False, float_format="%.10g")
    pd.DataFrame(quarterly_rows).to_csv(HERE / "quarterly_returns.csv", index=False, float_format="%.10g")
    pd.DataFrame(annual_rows).to_csv(HERE / "annual_returns.csv", index=False, float_format="%.10g")
    (HERE / "metrics.json").write_text(json.dumps(metrics_output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (HERE / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cache_hashes = []
    def add_hash(path: Path, role: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        cache_hashes.append({"role": role, "path": str(path.resolve()),
                             "size_bytes": path.stat().st_size, "sha256": sha256(path)})

    add_hash(protocol_path, "frozen V41 protocol")
    add_hash(Path(__file__), "V41 runner")
    add_hash(V11 / "analysis.py", "V11 final corrected shared-cash ledger")
    add_hash(V34 / "protocol.json", "frozen V34 protocol")
    add_hash(V34 / "run_v34.py", "V34 monthly factor implementation")
    add_hash(V34 / "metrics.json", "frozen V34 comparison metrics")
    add_hash(V34 / "daily_equity.csv", "frozen V34 monthly daily ledger")
    add_hash(V34 / "trades.csv", "frozen V34 monthly trades")
    add_hash(DATA / "meta.json", "training data cache metadata")
    for path in sorted((V11 / "quarters").glob("*/predictions.csv")):
        add_hash(path, "cached V11 shared monthly signal schedule")
    for quarter in importance_quarters:
        add_hash(V11 / "quarters" / quarter / "result.json", "pre-holdout V11 factor importance")
    meta = read_json(DATA / "meta.json")
    for year in sorted(int(value) for value in meta["built_years"]):
        add_hash(DATA / "prices" / f"year={year}" / "data.parquet", f"price cache {year}")
        add_hash(DATA / "amount" / f"year={year}" / "data.parquet", f"amount cache {year}")
    for year in sorted(int(value) for value in meta["built_years"] if 2024 <= int(value) <= 2026):
        add_hash(DATA / "factors" / f"year={year}" / "data.parquet", f"factor cache {year}")
    (HERE / "cache_hashes.json").write_text(json.dumps({
        "algorithm": "SHA-256",
        "protocol_sha256": PROTOCOL_SHA256,
        "scope": "Frozen V41 and V34 protocols; V34 monthly reference outputs and runner; V11 ledger, prediction schedule and pre-holdout importance records; all price and amount cache years read by the ledger; 2024-2026 factor cache years.",
        "hashes": cache_hashes,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table = [
        "| Strategy | Net return | Annualized | Max drawdown | Ann. vol. | Fees | Slippage | Total cost | Turnover | Trades | Return/vol |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in metrics.items():
        table.append("| {} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2f} | {:.2f} | {:.2f} | {:.0f} | {} | {:.3f} |".format(
            name, metric["net_return"], metric["annualized_net_return"], metric["max_drawdown"],
            metric["annualized_volatility"], metric["fees"], metric["slippage_cost"],
            metric["total_trading_cost"], metric["turnover"], metric["trades"], metric["return_over_volatility"]))
    report = [
        "# V41：V34 冻结因子 Top2 季度调仓实验",
        "",
        "## 冻结规则",
        "",
        "先冻结 V34 的因子、方向和账本，只改变预先指定的调仓频率：每季度仅在共同月初信号中的首个交易日收盘读取 `id2_close_vs_pm_vwap_20`，四家银行按因子降序排名，Top2 各目标 50%，次日开盘调仓。季度内其他月初信号不下单，沿用原持仓；2024-01-02 首笔仍按 Top2 等权开始。窗口为 2024-01-02 至 2026-06-30。",
        "",
        "V34 因子在 2020Q1–2023Q4 的 16 个训练重要性季度中有 10 个季度进入 top-20，截止 2023Q4；V41 不重训、不扫参数，也不根据留出收益改频率。完整冻结内容见 `protocol.json`。",
        "",
        "## 结果",
        "",
        *table,
        "",
        "与共同 V34 月调仓对照相比，季度调仓换手减少 {:.1%}，账本总成本减少 {:.1%}（手续费节省 {:.1%}，滑点节省 {:.1%}）；净收益差 {:.2f} 个百分点，最大回撤差 {:.2f} 个百分点，年化波动差 {:.2f} 个百分点。年化收益/波动比差 {:.3f}，Calmar 比差 {:.3f}。".format(
            comparison["turnover_reduction_fraction"], comparison["total_cost_savings_fraction"],
            comparison["fees_savings_fraction"], comparison["slippage_savings_fraction"],
            comparison["net_return_difference_pp"], comparison["max_drawdown_difference_pp"],
            comparison["annualized_volatility_difference_pp"], comparison["return_over_volatility_difference"],
            comparison["calmar_difference"]),
        "",
        "费用沿用 V11 最终修正版共享现金账本，包括 T+1 开盘、1% 成交参与率、100 股整手、佣金/最低佣金/过户费/卖出印花税、反向滑点、涨跌停阻塞、复权股数及全退零股处理。期末按 2026-06-30 收盘估值，不强制平仓。季度和年度明细见 `quarterly_returns.csv`、`annual_returns.csv`，逐日权益和成交见 `daily_equity.csv`、`trades.csv`。",
        "",
        "## 审计",
        "",
        "- V34 月调仓对照从相同输入重放，并与 V34 原始逐日账本、成交数、持仓、指标按容差核对；核对细节见 `audit.json`。",
        "- `membership_audit.json` 记录全部 30 个共同月初信号、当日因子值、季度使用标记、Top2 以及次日执行日期；只有 10 个季度首信号进入调仓。",
        "- 每策略现金、受阻订单、账本残差与最低现金见 `audit.json`；输入和源代码 SHA-256 清单见 `cache_hashes.json`。",
        "",
        "结果是固定四家银行历史窗口上的回测比较，不代表未来稳定超额。",
    ]
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(HERE),
        "protocol_sha256": PROTOCOL_SHA256,
        "monthly_signal_count": len(monthly_dates),
        "quarterly_rebalance_dates": quarterly_dates,
        "metrics": metrics,
        "quarterly_vs_monthly": comparison,
        "v34_monthly_control_compatibility": reference_compatibility,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
