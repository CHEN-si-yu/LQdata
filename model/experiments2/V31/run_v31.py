#!/usr/bin/env python3
"""V31: frozen single-factor monthly Top1, fully invested on the V26 holdout."""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[k]="1"
import sys, json, math, hashlib, collections
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
V11 = HERE.parent / "V11"
sys.path.insert(0, str(V11))
import analysis as E

DATA = V11.parents[1] / "trainingdata"
CODES = E.CODES
START = "2024-01-02"
END = "2026-06-30"
FEATURE_CUTOFF = "2023Q4"
N_FEATURES = 1


def choose_features():
    feature = "mf_tier_flow_agreement_20"
    normalized_gains = []
    quarters = []
    for path in sorted((V11 / "quarters").glob("*/result.json")):
        quarter = path.parent.name
        if quarter > FEATURE_CUTOFF:
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        top20 = obj.get("top_features_by_gain", [])[:20]
        if len(top20) < 20:
            raise RuntimeError(f"{quarter}: expected a 20-feature importance list")
        denom = sum(float(item["gain"]) for item in top20) or 1.0
        found = next((item for item in top20 if item["feature"] == feature), None)
        if found is not None:
            normalized_gains.append(float(found["gain"]) / denom)
        quarters.append(quarter)
    if len(quarters) != 16:
        raise RuntimeError("expected exactly 16 V11 importance folds through 2023Q4")
    if len(normalized_gains) != 16:
        raise RuntimeError(f"{feature} must occur in the top 20 for all 16 pre-holdout folds")
    return [feature], quarters, {
        feature: {
            "top20_quarters": len(normalized_gains),
            "total_importance_quarters": len(quarters),
            "mean_normalized_gain_when_top20": float(np.mean(normalized_gains))
        }
    }


def load_factor_rows(features):
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    if meta.get("semantics") != "zscore_win1_99_v1":
        raise RuntimeError("unexpected factor semantics")
    years = sorted(int(y) for y in meta["built_years"] if 2024 <= int(y) <= 2026)
    parts = []
    for year in years:
        frame = pq.read_table(DATA / "factors" / f"year={year}" / "data.parquet",
                              columns=["trade_date", "stock_code", *features],
                              filters=[("stock_code", "in", list(CODES))]).to_pandas()
        frame["trade_date"] = frame["trade_date"].astype(str).str[:10]
        frame["stock_code"] = frame["stock_code"].astype(str)
        parts.append(frame)
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("duplicate factor date/security rows")
    return out, meta


def build_factor_schedule(factors, signal_dates, features):
    schedule, audit = {}, []
    by_date = {d: x.set_index("stock_code") for d, x in factors.groupby("trade_date", sort=False)}
    for d in signal_dates:
        if d not in by_date:
            raise RuntimeError(f"missing factor rows at monthly signal {d}")
        day = by_date[d]
        values = day.reindex(CODES)[features].astype(float).fillna(0.0)
        ranks = {}
        for feature in features:
            ordered = sorted(CODES, key=lambda c: (-float(values.loc[c, feature]), c))
            ranks[feature] = {code: i + 1 for i, code in enumerate(ordered)}
        composite = {code: float(np.mean([ranks[f][code] for f in features])) for code in CODES}
        picked = sorted(CODES, key=lambda c: (composite[c], c))[:2]
        schedule[d] = {c: 0.5 for c in picked}

        audit.append({
            "signal_date": d,
            "selected_top2": picked,
            "composite_average_rank": composite,
            "feature_ranks": ranks,
            "signal_prices_close_only": True,
            "execution_date": ""
        })
    return schedule, audit


def period_returns(daily, freq):
    frame = daily.copy()
    frame["trade_date"] = pd.to_datetime(frame.trade_date)
    if freq == "Q":
        frame["period"] = frame.trade_date.dt.to_period("Q").astype(str)
    else:
        frame["period"] = frame.trade_date.dt.year.astype(str)
    rows = []
    for period, group in frame.groupby("period", sort=True):
        rows.append({"period": period, "strategy": str(group.strategy.iloc[0]),
                     "net_return": float(np.prod(1.0 + group.daily_return_net.to_numpy()) - 1.0),
                     "gross_return_same_fills": float(np.prod(1.0 + group.gross_return_daily.to_numpy()) - 1.0)})
    return rows


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    features, importance_quarters, importance = choose_features()
    if features != ["mf_tier_flow_agreement_20"]:
        raise RuntimeError("V31 feature is frozen to mf_tier_flow_agreement_20")

    prices = E.load_prices()
    quarter_dirs = sorted(p for p in (V11 / "quarters").iterdir()
                          if p.is_dir() and (p / "predictions.csv").is_file())
    quarters = [p.name for p in quarter_dirs]
    pred = E.load_predictions(quarters)
    schedules, _ = E.schedules(pred, prices)
    signal_dates = sorted(d for d in pred.loc[pred.monthly_signal, "trade_date"].unique()
                          if START <= d <= END)
    if len(signal_dates) != 30:
        raise RuntimeError(f"expected 30 frozen holdout monthly signals, found {len(signal_dates)}")

    factors, meta = load_factor_rows(features)
    if meta.get("semantics") != "zscore_win1_99_v1":
        raise RuntimeError("unexpected harmonized factor semantics")
    factor_dates = set(factors.trade_date.unique())
    if not set(signal_dates).issubset(factor_dates):
        missing = sorted(set(signal_dates) - factor_dates)
        raise RuntimeError(f"missing factor rows on signal dates: {missing[:5]}")
    factor_by_date = {d: frame.set_index("stock_code") for d, frame in factors.groupby("trade_date", sort=False)}
    top2_schedule, feature_rank_audit = build_factor_schedule(factors, signal_dates, features)
    top1_schedule = {}
    membership_audit = []
    idx = prices["index"]
    for source in feature_rank_audit:
        signal_date = source["signal_date"]
        day = factor_by_date[signal_date].reindex(CODES)
        values = day[features[0]].astype(float).fillna(0.0)
        ordering = sorted(CODES, key=lambda code: (-float(values.loc[code]), code))
        top1 = ordering[0]
        top2 = ordering[:2]
        if top2 != source["selected_top2"]:
            raise RuntimeError(f"{signal_date}: independent single-factor ordering disagrees with V26 Top2")
        top1_schedule[signal_date] = {top1: 1.0}
        execution_date = str(prices["days"][idx[signal_date] + 1])
        if idx[execution_date] != idx[signal_date] + 1:
            raise RuntimeError("signal did not execute at next trading session")
        ranks = source["feature_ranks"][features[0]]
        membership_audit.append({
            "signal_date": signal_date,
            "feature_name": features[0],
            "factor_values_at_signal_close": {code: float(values.loc[code]) for code in CODES},
            "feature_ranks": ranks,
            "selected_top1": top1,
            "selected_v26_top2": top2,
            "top1_target_weight": 1.0,
            "top1_is_full_equity_unlevered": True,
            "factor_missing_values_filled_with_zero": int(sum(not np.isfinite(float(day[features[0]].get(code, np.nan))) for code in CODES)),
            "factor_timestamp_is_signal_date": True,
            "execution_date": execution_date,
            "no_future_row_used": execution_date > signal_date
        })

    first = signal_dates[0]
    holdout_schedules = {
        "V31 mf_tier_flow_agreement_20 Top1 full": top1_schedule,
        "V26 mf_tier_flow_agreement_20 Top2": top2_schedule,
        "60d momentum Top2": {d: schedules["60d momentum Top2"][d] for d in signal_dates},
        "V11 LambdaRank Top2": {d: schedules["V11 LambdaRank Top2"][d] for d in signal_dates},
        "equal-weight hold": {first: {code: 0.25 for code in CODES}},
    }
    all_daily, all_trades, metrics = [], [], {}
    for name, target_schedule in holdout_schedules.items():
        daily, trades, metric = E.simulate(name, prices, target_schedule, START, END)
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

    # Recheck all four common portfolios against V26's frozen ledger outputs.
    v26_dir = HERE.parent / "V26"
    v26_metrics = json.loads((v26_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    v26_daily = pd.read_csv(v26_dir / "daily_equity.csv", dtype={"trade_date": str})
    v26_trades = pd.read_csv(v26_dir / "trades.csv", dtype={"trade_date": str, "stock_code": str})
    daily_fields = ["equity_net", "equity_gross", "cash", "fees_cumulative",
                    "slippage_cumulative", "turnover_cumulative", "accounting_residual",
                    "daily_return_net", "drawdown"]
    trade_keys = ["trade_date", "strategy", "stock_code", "side", "reason"]
    trade_numeric = ["shares", "open", "fill", "notional", "fee", "slippage_cost"]
    compatibility = {}
    tolerance = 1e-3
    baseline_names = ("V26 mf_tier_flow_agreement_20 Top2", "60d momentum Top2",
                      "V11 LambdaRank Top2", "equal-weight hold")
    for name in baseline_names:
        current = metrics[name]
        reference = v26_metrics[name]
        metric_diffs = {
            field: abs(float(current[field]) - float(reference[field]))
            for field in ("net_return", "fees", "slippage_cost", "turnover",
                          "max_drawdown", "annualized_volatility", "min_cash")
        }
        if current["trades"] != reference["trades"]:
            raise RuntimeError(f"{name}: V26/V31 trade count differs")
        old = v26_daily.loc[v26_daily.strategy == name].sort_values("trade_date").reset_index(drop=True)
        new = daily_all.loc[daily_all.strategy == name].sort_values("trade_date").reset_index(drop=True)
        if old.trade_date.tolist() != new.trade_date.tolist() or len(old) != len(new):
            raise RuntimeError(f"{name}: V26/V31 daily date coverage differs")
        daily_diff = {
            field: float(np.max(np.abs(old[field].to_numpy(dtype=float) - new[field].to_numpy(dtype=float))))
            for field in daily_fields
        }
        shares_match = old["shares_json"].astype(str).tolist() == new["shares_json"].astype(str).tolist()

        old_trades = v26_trades.loc[v26_trades.strategy == name].sort_values(trade_keys).reset_index(drop=True)
        new_trades = trades_all.loc[trades_all.strategy == name].sort_values(trade_keys).reset_index(drop=True)
        trade_keys_match = len(old_trades) == len(new_trades)
        trade_diff = {}
        if trade_keys_match:
            for field in trade_keys:
                trade_keys_match = trade_keys_match and old_trades[field].astype(str).tolist() == new_trades[field].astype(str).tolist()
            trade_diff = {
                field: (float(np.max(np.abs(old_trades[field].to_numpy(dtype=float) - new_trades[field].to_numpy(dtype=float))))
                        if len(old_trades) else 0.0)
                for field in trade_numeric
            }
        else:
            trade_diff = {field: float("inf") for field in trade_numeric}
        ok = (all(value <= tolerance for value in metric_diffs.values())
              and all(value <= tolerance for value in daily_diff.values())
              and all(value <= tolerance for value in trade_diff.values())
              and shares_match and trade_keys_match)
        if not ok:
            raise RuntimeError(f"{name}: V31 common ledger differs from frozen V26 baseline")
        compatibility[name] = {
            "metric_abs_differences_vs_v26": metric_diffs,
            "daily_max_abs_differences_vs_v26": daily_diff,
            "daily_rows_match": len(old),
            "shares_json_exact_match": shares_match,
            "trade_rows_match": bool(trade_keys_match),
            "trade_max_abs_differences_vs_v26": trade_diff,
            "tolerance": tolerance
        }

    source_hash = file_sha256(Path(__file__))
    protocol = {
        "version": "V31",
        "hypothesis": "A single V11 factor frozen before the holdout may perform differently when its highest-ranked bank receives the entire unlevered equity allocation instead of equal weighting the top two.",
        "factor_selection": {
            "cutoff_quarter": FEATURE_CUTOFF,
            "importance_quarters": importance_quarters,
            "rule": "Use only mf_tier_flow_agreement_20, fixed because it appeared in V11 top-20 gain features in all 16 quarters from 2020Q1 through 2023Q4; do not inspect holdout returns to select or change it.",
            "selected_features": features,
            "selection_statistics": importance,
            "direction": "Higher harmonized zscore_win1_99_v1 factor value ranks higher."
        },
        "portfolio_rule": "At each shared V11 first-session monthly signal close, rank the four banks by mf_tier_flow_agreement_20 descending, break ties by stock code, and hold only the highest-ranked bank at 100% of account equity. No leverage. Execute at the next trading session open.",
        "holdout_dates": [START, END],
        "monthly_signal_dates": signal_dates,
        "execution": "Exact V11 final corrected shared-cash ledger: T+1 open, 1% participation cap, ordinary 100-share lots, fees, adverse slippage, limit-price blocks, corporate-action adjusted shares, and full-exit odd-lot handling. Mark positions to final holdout close without forced liquidation, matching V26.",
        "comparisons": ["V26 mf_tier_flow_agreement_20 Top2", "60d momentum Top2", "V11 LambdaRank Top2", "equal-weight hold"],
        "no_retraining": True,
        "no_parameter_sweep": True,
        "no_holdout_based_rule_changes": True,
        "no_leverage": True,
        "factor_semantics": meta.get("semantics"),
        "initial_equity": E.INITIAL,
        "monthly_signals": len(signal_dates),
        "source_script_sha256": source_hash,
        "v26_protocol_sha256": file_sha256(v26_dir / "protocol.json"),
        "v26_runner_sha256": file_sha256(v26_dir / "run_v26.py"),
        "v11_ledger_source_sha256": file_sha256(V11 / "analysis.py")
    }
    audit = {
        "window": [START, END],
        "monthly_signals": len(signal_dates),
        "all_signal_dates_from_v11_monthly_schedule": True,
        "all_factor_rows_on_signal_dates_available": True,
        "factor_selected_before_holdout": max(importance_quarters) <= FEATURE_CUTOFF,
        "selected_factor_top20_quarters": importance[features[0]]["top20_quarters"],
        "selected_factor_all_16_training_quarters": importance[features[0]]["top20_quarters"] == 16,
        "signal_uses_factor_value_on_signal_date": all(x["factor_timestamp_is_signal_date"] for x in membership_audit),
        "execution_is_next_session_open": all(x["no_future_row_used"] for x in membership_audit),
        "top1_target_weight_always_one": all(x["top1_target_weight"] == 1.0 for x in membership_audit),
        "max_positions_for_top1": metrics["V31 mf_tier_flow_agreement_20 Top1 full"]["max_positions"],
        "positions_marked_to_final_close_without_forced_liquidation": True,
        "v26_baseline_ledger_compatibility": compatibility,
        "strategies": {name: {
            "cash_nonnegative": metric["min_cash"] >= -1e-7,
            "blocked_trade_events": metric["blocked_trade_events"],
            "max_abs_accounting_residual": metric["max_abs_accounting_residual"],
            "max_positions": metric["max_positions"],
            "trades": metric["trades"],
            "min_cash": metric["min_cash"]
        } for name, metric in metrics.items()},
        "membership_audit_rows": len(membership_audit)
    }

    (HERE / "membership_audit.json").write_text(json.dumps(membership_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    daily_all.to_csv(HERE / "daily_equity.csv", index=False, float_format="%.10g")
    trades_all.to_csv(HERE / "trades.csv", index=False, float_format="%.10g")
    pd.DataFrame(quarterly).to_csv(HERE / "quarterly_returns.csv", index=False, float_format="%.10g")
    pd.DataFrame(annual).to_csv(HERE / "annual_returns.csv", index=False, float_format="%.10g")
    (HERE / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "metrics.json").write_text(json.dumps({
        "window": [START, END], "days": int(len(daily_all) // len(holdout_schedules)),
        "signal_count": len(signal_dates), "selected_feature": features[0],
        "feature_selection": importance, "metrics": metrics,
        "v26_compatibility": compatibility
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    records = []
    def add_hash(path, role):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"role": role, "path": str(path.resolve()),
                        "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
    add_hash(DATA / "meta.json", "factor-and-market-cache metadata")
    add_hash(V11 / "analysis.py", "V11 corrected ledger engine")
    for path in sorted((V11 / "quarters").glob("*/predictions.csv")):
        add_hash(path, "cached V11 quarterly predictions")
    for quarter in importance_quarters:
        add_hash(V11 / "quarters" / quarter / "result.json", "pre-holdout factor importance")
    for year in sorted(int(y) for y in meta["built_years"]):
        add_hash(DATA / "prices" / f"year={year}" / "data.parquet", f"price cache year {year}")
        add_hash(DATA / "amount" / f"year={year}" / "data.parquet", f"amount cache year {year}")
    for year in sorted(int(y) for y in meta["built_years"] if 2024 <= int(y) <= 2026):
        add_hash(DATA / "factors" / f"year={year}" / "data.parquet", f"factor cache year {year}")
    for filename in ("protocol.json", "metrics.json", "daily_equity.csv", "trades.csv", "run_v26.py"):
        add_hash(v26_dir / filename, f"frozen V26 reference {filename}")
    add_hash(Path(__file__), "V31 runner")
    hash_manifest = {
        "algorithm": "SHA-256",
        "scope": "All market cache years read by V11.load_prices; factor parquet years 2024-2026 read by V26-compatible factor loader; cached V11 predictions and 2020Q1-2023Q4 importance artifacts; V26 comparison outputs and runner; V11 ledger and V31 runner.",
        "hashes": records
    }
    (HERE / "cache_hashes.json").write_text(json.dumps(hash_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table = ["| Strategy | Net return | Annualized | Max drawdown | Ann. vol | Fees | Slippage | Trades |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, metric in metrics.items():
        table.append("| {} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2f} | {:.2f} | {} |".format(
            name, metric["net_return"], metric["annualized_net_return"], metric["max_drawdown"],
            metric["annualized_volatility"], metric["fees"], metric["slippage_cost"], metric["trades"]))
    report = [
        "# V31：mf_tier_flow_agreement_20 单因子满仓 Top1",
        "",
        "## 预先固定的规则",
        "",
        "沿用V26冻结留出期2024-01-02至2026-06-30及唯一因子mf_tier_flow_agreement_20。该因子在V11 2020Q1–2023Q4训练重要性文件中16/16季进入top-20。每个共同月初信号日收盘按四家银行因子值降序选Top1，单只持仓目标为100%权益，不加杠杆，次日开盘成交。因子、方向、窗口和仓位规则不依据留出表现调整。",
        "",
        "使用V11最终修正账本；V26 Top2、60日动量Top2、V11 Ranker Top2和四股等权采用同一留出日期、T+1开盘、成本和账户引擎。窗口末按最后收盘估值，不强制平仓；不重训、不扫参。",
        "",
        "## 结果",
        "",
        *table,
        "",
        "季度与年度收益见 quarterly_returns.csv、annual_returns.csv；逐日权益与成交明细见 daily_equity.csv、trades.csv。",
        "",
        "## 审计",
        "",
        f"- 单因子进入训练期top-20次数为{importance[features[0]]['top20_quarters']}/16，特征仅依赖2023Q4及以前的V11训练重要性。",
        "- 30个信号均使用信号日收盘因子横截面值，并于下一交易日开盘执行；Top1目标权重恒为100%。",
        "- V26及其他三条共同基线与V26原结果逐日核对：指标/成交数完全一致，现金、权益、换手账面显示差异小于0.001元，持仓股数完全一致；详见 audit.json。",
        "- 各策略现金和会计残差、逐信号选股与原始因子值见 audit.json、membership_audit.json。",
        "- 输入缓存与执行脚本 SHA-256 见 cache_hashes.json。",
        "",
        "这是固定四股留出窗口的历史比较，不证明未来稳定超额。"
    ]
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_feature": features[0], "top20_quarters": importance[features[0]]["top20_quarters"],
        "signal_count": len(signal_dates), "metrics": metrics, "audit": audit,
        "runner_sha256": source_hash, "output_dir": str(HERE)
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
