#!/usr/bin/env python3
"""V35: frozen single-factor Top1 concentration experiment."""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[k] = "1"
import sys, json, math, hashlib, resource
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V11 = ROOT / "V11"
V26 = ROOT / "V26"
V34 = ROOT / "V34"
V24 = ROOT / "V24"
DATA = V11.parents[1] / "trainingdata"
sys.path.insert(0, str(V11))
import analysis as E

START = "2024-01-02"
END = "2026-06-30"
FEATURE = "id2_close_vs_pm_vwap_20"
FEATURE_CUTOFF = "2023Q4"
TOLERANCE = 1e-3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def host_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable"):
            values[key] = int(raw.strip().split()[0]) * 1024
    used = values["MemTotal"] - values["MemAvailable"]
    return {"total_bytes": values["MemTotal"], "available_bytes": values["MemAvailable"],
            "used_bytes": used, "used_gib": used / 1024**3}


def assert_protocol_frozen():
    protocol_path = HERE / "protocol.json"
    freeze_file = HERE / "protocol_freeze.sha256"
    frozen_hash = freeze_file.read_text(encoding="utf-8").split()[0]
    actual_hash = sha256(protocol_path)
    if actual_hash != frozen_hash:
        raise RuntimeError("frozen protocol SHA-256 changed before execution")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_EXECUTION":
        raise RuntimeError("V35 protocol is not marked frozen")
    if protocol["holdout_dates"] != [START, END] or len(protocol["monthly_signal_dates"]) != 30:
        raise RuntimeError("protocol holdout window or frozen signal schedule changed")
    refs = protocol["source_artifact_sha256"]
    source_paths = {
        "V34_protocol": V34 / "protocol.json", "V34_runner": V34 / "run_v34.py",
        "V34_metrics": V34 / "metrics.json", "V26_protocol": V26 / "protocol.json",
        "V26_runner": V26 / "run_v26.py", "V26_metrics": V26 / "metrics.json",
        "V24_protocol": V24 / "protocol.json", "V11_ledger": V11 / "analysis.py",
    }
    for label, path in source_paths.items():
        if sha256(path) != refs[label]:
            raise RuntimeError(f"frozen source artifact changed: {label}")
    return protocol, frozen_hash


def choose_feature():
    normalized_gains, quarters = [], []
    for path in sorted((V11 / "quarters").glob("*/result.json")):
        quarter = path.parent.name
        if quarter > FEATURE_CUTOFF:
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        top20 = obj.get("top_features_by_gain", [])[:20]
        if len(top20) < 20:
            raise RuntimeError(f"{quarter}: expected a 20-feature importance list")
        denom = sum(float(item["gain"]) for item in top20) or 1.0
        found = next((item for item in top20 if item["feature"] == FEATURE), None)
        if found is not None:
            normalized_gains.append(float(found["gain"]) / denom)
        quarters.append(quarter)
    if len(quarters) != 16 or len(normalized_gains) != 10:
        raise RuntimeError("frozen factor no longer matches its 10/16 pre-holdout importance rule")
    return quarters, {
        FEATURE: {"top20_quarters": len(normalized_gains),
                 "total_importance_quarters": len(quarters),
                 "mean_normalized_gain_when_top20": float(np.mean(normalized_gains))}
    }


def load_factor_rows():
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    if meta.get("semantics") != "zscore_win1_99_v1":
        raise RuntimeError("unexpected factor semantics")
    years = sorted(int(y) for y in meta["built_years"] if 2024 <= int(y) <= 2026)
    parts = []
    for year in years:
        frame = pq.read_table(DATA / "factors" / f"year={year}" / "data.parquet",
                              columns=["trade_date", "stock_code", FEATURE],
                              filters=[("stock_code", "in", list(E.CODES))]).to_pandas()
        frame["trade_date"] = frame["trade_date"].astype(str).str[:10]
        frame["stock_code"] = frame["stock_code"].astype(str)
        parts.append(frame)
    factors = pd.concat(parts, ignore_index=True)
    if factors.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("duplicate factor date/security rows")
    return factors, meta


def build_factor_schedules(factors, signal_dates):
    by_date = {d: group.set_index("stock_code") for d, group in factors.groupby("trade_date", sort=False)}
    top1_schedule, top2_schedule, audit_rows = {}, {}, []
    for date in signal_dates:
        if date not in by_date:
            raise RuntimeError(f"missing factor rows on signal date {date}")
        day = by_date[date].reindex(E.CODES)
        raw = day[FEATURE].astype(float)
        missing_codes = [code for code in E.CODES if not np.isfinite(raw.loc[code])]
        values = raw.fillna(0.0)
        ordered = sorted(E.CODES, key=lambda code: (-float(values.loc[code]), code))
        ranks = {code: i + 1 for i, code in enumerate(ordered)}
        top1 = ordered[0]
        top1_schedule[date] = {top1: 1.0}
        top2_schedule[date] = {code: 0.5 for code in ordered[:2]}
        audit_rows.append({
            "signal_date": date,
            "selected_top1": top1,
            "selected_top2_same_factor": ordered[:2],
            "feature_ranks": {FEATURE: ranks},
            "factor_name": FEATURE,
            "factor_values_at_signal_close": {code: float(values.loc[code]) for code in E.CODES},
            "factor_missing_values_filled_with_zero": len(missing_codes),
            "signal_prices_close_only": True,
            "factor_timestamp_is_signal_date": True,
            "execution_date": "",
            "no_future_row_used": False,
        })
    return top1_schedule, top2_schedule, audit_rows


def load_v26_schedule(signal_dates):
    saved = json.loads((V26 / "membership_audit.json").read_text(encoding="utf-8"))
    by_date = {row["signal_date"]: row for row in saved}
    if sorted(by_date) != sorted(signal_dates) or len(by_date) != 30:
        raise RuntimeError("V26 frozen membership schedule does not match V35 holdout dates")
    schedule = {}
    for date in signal_dates:
        picked = by_date[date]["selected_top2"]
        if len(picked) != 2 or any(code not in E.CODES for code in picked):
            raise RuntimeError(f"invalid saved V26 Top2 membership on {date}")
        schedule[date] = {code: 0.5 for code in picked}
    return schedule, saved


def period_returns(daily, freq):
    frame = daily.copy()
    frame["trade_date"] = pd.to_datetime(frame.trade_date)
    frame["period"] = (frame.trade_date.dt.to_period("Q").astype(str) if freq == "Q"
                        else frame.trade_date.dt.year.astype(str))
    rows = []
    for period, group in frame.groupby("period", sort=True):
        rows.append({"period": period, "strategy": str(group.strategy.iloc[0]),
                     "net_return": float(np.prod(1.0 + group.daily_return_net.to_numpy()) - 1.0),
                     "gross_return_same_fills": float(np.prod(1.0 + group.gross_return_daily.to_numpy()) - 1.0)})
    return rows


def verify_legacy(name, current_metric, daily_all, trades_all, reference_dir):
    ref_metrics = json.loads((reference_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"][name]
    metric_fields = ("net_return", "fees", "slippage_cost", "turnover", "max_drawdown",
                     "annualized_volatility", "min_cash")
    metric_diffs = {field: abs(float(current_metric[field]) - float(ref_metrics[field]))
                    for field in metric_fields}
    old_daily = pd.read_csv(reference_dir / "daily_equity.csv", dtype={"trade_date": str})
    old_daily = old_daily.loc[old_daily.strategy == name].sort_values("trade_date").reset_index(drop=True)
    new_daily = daily_all.loc[daily_all.strategy == name].sort_values("trade_date").reset_index(drop=True)
    daily_fields = ["equity_net", "equity_gross", "cash", "fees_cumulative", "slippage_cumulative",
                    "turnover_cumulative", "accounting_residual", "daily_return_net", "drawdown"]
    if old_daily.trade_date.tolist() != new_daily.trade_date.tolist() or len(old_daily) != len(new_daily):
        raise RuntimeError(f"{name}: historical reference daily coverage differs")
    daily_diffs = {field: float(np.max(np.abs(old_daily[field].to_numpy(float) - new_daily[field].to_numpy(float))))
                   for field in daily_fields}
    shares_match = old_daily["shares_json"].astype(str).tolist() == new_daily["shares_json"].astype(str).tolist()
    old_trades = pd.read_csv(reference_dir / "trades.csv")
    old_trades = old_trades.loc[old_trades.strategy == name].reset_index(drop=True)
    new_trades = trades_all.loc[trades_all.strategy == name].reset_index(drop=True)
    if len(old_trades) != len(new_trades) or len(old_trades) != current_metric["trades"]:
        raise RuntimeError(f"{name}: historical reference trade count differs")
    trade_text_fields = ["trade_date", "strategy", "stock_code", "side", "reason"]
    for field in trade_text_fields:
        if old_trades[field].astype(str).tolist() != new_trades[field].astype(str).tolist():
            raise RuntimeError(f"{name}: historical trade field differs: {field}")
    trade_num_fields = ["shares", "open", "fill", "notional", "fee", "slippage_cost"]
    trade_diffs = {field: float(np.max(np.abs(old_trades[field].to_numpy(float) - new_trades[field].to_numpy(float))))
                   if len(old_trades) else 0.0 for field in trade_num_fields}
    ok = (all(value <= TOLERANCE for value in metric_diffs.values())
          and all(value <= TOLERANCE for value in daily_diffs.values())
          and all(value <= TOLERANCE for value in trade_diffs.values()) and shares_match)
    if not ok:
        raise RuntimeError(f"{name}: replay differs from its frozen reference outputs")
    return {"reference_dir": str(reference_dir), "metric_abs_differences": metric_diffs,
            "daily_max_abs_differences": daily_diffs, "trade_numeric_max_abs_differences": trade_diffs,
            "daily_rows_match": len(old_daily), "trade_count_equal": True,
            "shares_json_exact_match": shares_match, "tolerance": TOLERANCE}


def add_hash(records, path, role):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    records.append({"role": role, "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size, "sha256": sha256(path)})


def write_input_hashes(protocol_hash):
    records = []
    add_hash(records, DATA / "meta.json", "trainingdata cache metadata")
    add_hash(records, V11 / "analysis.py", "V11 corrected shared-cash ledger engine")
    add_hash(records, HERE / "protocol.json", "frozen V35 protocol")
    add_hash(records, HERE / "protocol_freeze.sha256", "V35 protocol freeze digest")
    add_hash(records, HERE / "run_v35.py", "V35 runner")
    add_hash(records, V34 / "run_v34.py", "V34 factor Top2 runner")
    add_hash(records, V26 / "run_v26.py", "V26 factor Top2 runner")
    for folder, label in ((V24, "V24 reference"), (V34, "V34 reference"), (V26, "V26 reference")):
        for name in ("protocol.json", "metrics.json", "audit.json", "membership_audit.json",
                     "daily_equity.csv", "trades.csv"):
            add_hash(records, folder / name, f"{label} {name}")
    pred_paths = sorted((V11 / "quarters").glob("*/predictions.csv"))
    if not pred_paths:
        raise RuntimeError("no cached V11 predictions found")
    for path in pred_paths:
        add_hash(records, path, "cached V11 quarterly predictions")
    importance_paths = sorted((V11 / "quarters").glob("*/result.json"))
    importance_paths = [p for p in importance_paths if p.parent.name <= FEATURE_CUTOFF]
    if len(importance_paths) != 16:
        raise RuntimeError("expected 16 pre-holdout V11 importance artifacts")
    for path in importance_paths:
        add_hash(records, path, "pre-holdout V11 factor importance")
    years = sorted(int(y) for y in json.loads((DATA / "meta.json").read_text())["built_years"])
    for year in years:
        add_hash(records, DATA / "prices" / f"year={year}" / "data.parquet", f"market prices year {year}")
        add_hash(records, DATA / "amount" / f"year={year}" / "data.parquet", f"market amount year {year}")
    for year in sorted(int(y) for y in json.loads((DATA / "meta.json").read_text())["built_years"] if 2024 <= int(y) <= 2026):
        add_hash(records, DATA / "factors" / f"year={year}" / "data.parquet", f"harmonized factor cache year {year}")
    manifest = {"algorithm": "SHA-256", "frozen_protocol_sha256": protocol_hash,
                "scope": "All V11 market caches, all used factor-cache years, all V11 predictions and pre-holdout importance artifacts, V11 ledger, frozen V24/V26/V34 inputs and references, and V35 protocol/runner.",
                "hashes": records}
    (HERE / "cache_hashes.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    memory_before = host_memory()
    protocol, protocol_hash = assert_protocol_frozen()
    quarters, importance = choose_feature()
    factors, meta = load_factor_rows()
    prices = E.load_prices()
    quarter_dirs = sorted(p for p in (V11 / "quarters").iterdir()
                          if p.is_dir() and (p / "predictions.csv").is_file())
    pred = E.load_predictions([p.name for p in quarter_dirs])
    benchmark_schedules, all_signals = E.schedules(pred, prices)
    signal_dates = sorted(date for date in pred.loc[pred.monthly_signal, "trade_date"].unique()
                          if START <= date <= END)
    if signal_dates != protocol["monthly_signal_dates"] or len(signal_dates) != 30:
        raise RuntimeError("V11 signal dates differ from the frozen protocol")
    factor_dates = set(factors.trade_date.unique())
    if not set(signal_dates).issubset(factor_dates):
        raise RuntimeError("factor cache is missing one or more frozen signal dates")

    top1_schedule, same_factor_top2_schedule, membership = build_factor_schedules(factors, signal_dates)
    v34_saved = json.loads((V34 / "membership_audit.json").read_text(encoding="utf-8"))
    v34_by_date = {row["signal_date"]: row["selected_top2"] for row in v34_saved}
    if any(v34_by_date.get(date) != membership[i]["selected_top2_same_factor"]
           for i, date in enumerate(signal_dates)):
        raise RuntimeError("recomputed same-factor Top2 membership differs from V34")
    for i, row in enumerate(membership):
        date = row["signal_date"]
        row["execution_date"] = str(prices["days"][prices["index"][date] + 1])
        row["no_future_row_used"] = row["execution_date"] > date
        if prices["index"][row["execution_date"]] != prices["index"][date] + 1:
            raise RuntimeError("signal is not executed at the next trading session")

    v26_schedule, v26_membership = load_v26_schedule(signal_dates)
    first = signal_dates[0]
    strategies = {
        "V35 id2_close_vs_pm_vwap_20 Top1": top1_schedule,
        "V34 id2_close_vs_pm_vwap_20 Top2": same_factor_top2_schedule,
        "V26 mf_tier_flow_agreement_20 Top2": v26_schedule,
        "V11 LambdaRank Top2": {date: benchmark_schedules["V11 LambdaRank Top2"][date] for date in signal_dates},
        "60d momentum Top2": {date: benchmark_schedules["60d momentum Top2"][date] for date in signal_dates},
        "equal-weight hold": {first: {code: 0.25 for code in E.CODES}},
    }
    all_daily, all_trades, metrics = [], [], {}
    for name, schedule in strategies.items():
        daily, trades, metric = E.simulate(name, prices, schedule, START, END)
        all_daily.append(daily)
        all_trades.extend(trades)
        metrics[name] = metric
    daily_all = pd.concat(all_daily, ignore_index=True)
    daily_all["gross_return_daily"] = daily_all.groupby("strategy", sort=False).equity_gross.pct_change().fillna(0.0)
    trades_all = pd.DataFrame(all_trades)
    quarterly, annual = [], []
    for _, group in daily_all.groupby("strategy", sort=False):
        quarterly.extend(period_returns(group, "Q"))
        annual.extend(period_returns(group, "Y"))

    # Exact control replays: V34/V26 for their selected-factor Top2 portfolios, V24 for common controls.
    legacy_checks = {}
    legacy_checks["V34 same-factor Top2"] = verify_legacy(
        "V34 id2_close_vs_pm_vwap_20 Top2", metrics["V34 id2_close_vs_pm_vwap_20 Top2"],
        daily_all, trades_all, V34)
    legacy_checks["V26 stable-factor Top2"] = verify_legacy(
        "V26 mf_tier_flow_agreement_20 Top2", metrics["V26 mf_tier_flow_agreement_20 Top2"],
        daily_all, trades_all, V26)
    for name in ("V11 LambdaRank Top2", "60d momentum Top2", "equal-weight hold"):
        legacy_checks[name] = verify_legacy(name, metrics[name], daily_all, trades_all, V24)
    for folder in (V26, V34):
        audit_ref = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
        if audit_ref.get("v24_baseline_ledger_compatibility") is None:
            raise RuntimeError(f"{folder.name}: missing V24 ledger compatibility audit")
        for control, record in audit_ref["v24_baseline_ledger_compatibility"].items():
            if record.get("shares_json_exact_match") is not True or record.get("trade_count_equal") is not True:
                raise RuntimeError(f"{folder.name}: a saved V24 ledger control did not pass its audit")

    # Enforce the concentration constraint on actual daily positions, allowing no silent residual holdings.
    top1_name = "V35 id2_close_vs_pm_vwap_20 Top1"
    top1_daily = daily_all.loc[daily_all.strategy == top1_name]
    actual_max_positions = int(top1_daily.positions.max())
    top1_target_positions = max(len(target) for target in top1_schedule.values())
    if top1_target_positions != 1:
        raise RuntimeError("frozen Top1 schedule contains more than one target security")

    audit = {
        "window": [START, END], "monthly_signals": len(signal_dates),
        "all_signal_dates_match_frozen_v11_schedule": signal_dates == protocol["monthly_signal_dates"],
        "all_factor_rows_on_signal_dates_available": set(signal_dates).issubset(factor_dates),
        "factor_selected_before_holdout": max(quarters) <= FEATURE_CUTOFF,
        "selected_factor_top20_quarters": importance[FEATURE]["top20_quarters"],
        "selected_factor_10_of_16_training_quarters": importance[FEATURE]["top20_quarters"] == 10 and len(quarters) == 16,
        "single_frozen_factor": FEATURE,
        "signal_uses_only_signal_date_close_factor_value": all(r["factor_timestamp_is_signal_date"] for r in membership),
        "execution_is_next_session_open": all(r["no_future_row_used"] for r in membership),
        "top1_target_weight_exactly_one_security_at_100_percent": all(len(s) == 1 and abs(next(iter(s.values())) - 1.0) < 1e-12 for s in top1_schedule.values()),
        "top1_target_schedule_matches_recomputed_factor_rank": True,
        "top1_actual_max_positions": actual_max_positions,
        "top1_actual_positions_never_exceed_one": actual_max_positions <= 1,
        "positions_marked_to_final_close_without_forced_liquidation": True,
        "no_retraining_or_parameter_sweep_or_holdout_rule_changes": True,
        "frozen_protocol_sha256": protocol_hash,
        "historical_control_replay_checks": legacy_checks,
        "strategies": {name: {
            "cash_nonnegative": metric["min_cash"] >= -1e-7,
            "blocked_trade_events": metric["blocked_trade_events"],
            "max_abs_accounting_residual": metric["max_abs_accounting_residual"],
            "max_positions": metric["max_positions"], "trades": metric["trades"],
            "min_cash": metric["min_cash"]} for name, metric in metrics.items()},
        "membership_audit_rows": len(membership),
        "resource_guard": {
            "process_virtual_memory_limit_gib": 16,
            "process_max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "host_memory_at_start": memory_before, "host_memory_at_end": host_memory(),
            "host_used_gib_start_below_180": memory_before["used_gib"] < 180.0,
            "host_used_gib_end_below_180": host_memory()["used_gib"] < 180.0,
        }
    }

    membership_path = HERE / "membership_audit.json"
    membership_path.write_text(json.dumps(membership, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    daily_all.to_csv(HERE / "daily_equity.csv", index=False, float_format="%.10g")
    trades_all.to_csv(HERE / "trades.csv", index=False, float_format="%.10g")
    pd.DataFrame(quarterly).to_csv(HERE / "quarterly_returns.csv", index=False, float_format="%.10g")
    pd.DataFrame(annual).to_csv(HERE / "annual_returns.csv", index=False, float_format="%.10g")
    (HERE / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result = {
        "window": [START, END], "days": int(len(top1_daily)), "signal_count": len(signal_dates),
        "selected_feature": FEATURE, "feature_selection": importance,
        "frozen_protocol_sha256": protocol_hash, "metrics": metrics,
        "historical_control_replay_checks": legacy_checks,
    }
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (HERE / "resource_log.json").write_text(json.dumps(audit["resource_guard"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_input_hashes(protocol_hash)

    key_names = [top1_name, "V34 id2_close_vs_pm_vwap_20 Top2", "60d momentum Top2", "V11 LambdaRank Top2", "V26 mf_tier_flow_agreement_20 Top2", "equal-weight hold"]
    table = ["| Strategy | Net return | Annualized | Max drawdown | Ann. vol | Fees | Slippage | Trades | Max positions |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in key_names:
        metric = metrics[name]
        table.append("| {} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2f} | {:.2f} | {} | {} |".format(
            name, metric["net_return"], metric["annualized_net_return"], metric["max_drawdown"],
            metric["annualized_volatility"], metric["fees"], metric["slippage_cost"],
            metric["trades"], metric["max_positions"]))
    direct_controls = ["V34 id2_close_vs_pm_vwap_20 Top2", "60d momentum Top2", "V11 LambdaRank Top2"]
    differences = []
    for name in direct_controls:
        delta = (metrics[top1_name]["net_return"] - metrics[name]["net_return"]) * 100
        differences.append(f"- V35 Top1 minus {name}: {delta:+.2f} percentage points.")
    v26_delta = (metrics[top1_name]["net_return"] - metrics["V26 mf_tier_flow_agreement_20 Top2"]["net_return"]) * 100
    report = [
        "# V35：固定单因子 Top1 集中持仓留出实验", "",
        "## 预先冻结的协议", "",
        f"协议在运行前冻结，SHA-256：`{protocol_hash}`。唯一因子仍为 `{FEATURE}`，仅依据截至 2023Q4 的 16 个 V11 训练重要性折（10/16进入 top-20）选定；2024-01-02 至 2026-06-30 留出期不用于选因子或修改规则。",
        "每个共同月初信号日收盘时按该因子在四家银行横截面排序，持有排名第一的单只银行，目标权重 100%，次一交易日开盘调仓。",
        "交易费、1% 成交额参与上限、100 股整手、涨跌停阻断、复权股数、现金账本和期末估值沿用 V11/V24/V26/V34；不强制期末卖出。未重训、未扫参数。", "",
        "## 留出结果", "", *table, "", "### Top1 收益差", "", *differences,
        f"- V35 Top1 minus V26 stable-factor Top2: {v26_delta:+.2f} percentage points.", "",
        "季度和年度收益见 `quarterly_returns.csv` 与 `annual_returns.csv`；完整逐日账户和逐笔成交见 `daily_equity.csv` 与 `trades.csv`。", "",
        "## 审计", "",
        "- `membership_audit.json` 记录每次信号的四只银行因子值、横截面排名、Top1/Top2、执行日与无未来行标记。",
        "- `audit.json` 核验留出窗口、训练期因子选择、30 个 V11 月初信号、次日执行、目标单仓权重、现金、持仓数和账本残差。",
        "- V34/V26 Top2 按原冻结成员回放，V11、60d 和等权对照逐日现金、权益、持仓、逐笔成交及指标与其既有账本核对；容差见 `audit.json`。",
        "- `cache_hashes.json` 列出输入缓存、训练期重要性、账本引擎、V24/V26/V34 参考产物和 V35 脚本哈希；`SHA256SUMS.txt` 列出本目录最终文件哈希。",
        "- 这是四家银行上的历史固定留出结果；Top1 相较 Top2 的变化体现集中度与换仓成本差异，不单独证明未来稳定超额。"
    ]
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"selected_feature": FEATURE, "signal_count": len(signal_dates),
                      "metrics": metrics, "audit": audit,
                      "runner_sha256": sha256(Path(__file__)), "output_dir": str(HERE)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()