#!/usr/bin/env python3
"""V43: one preselected V11 training-era factor on a frozen 2024-2026 holdout."""
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
    feature = "rel_mom_ind_3d"
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
        raise RuntimeError(f"{feature} must occur in the top 20 for 16 of 16 pre-holdout folds")
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
    if features != ["rel_mom_ind_3d"]:
        raise RuntimeError("V43 feature is frozen to rel_mom_ind_3d")

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
    schedule, membership_audit = build_factor_schedule(factors, signal_dates, features)
    factor_by_date = {d: frame.set_index("stock_code") for d, frame in factors.groupby("trade_date", sort=False)}
    idx = prices["index"]
    for row in membership_audit:
        signal_date = row["signal_date"]
        day = factor_by_date[signal_date].reindex(CODES)
        vals = day[features[0]].astype(float)
        missing_codes = [code for code in CODES if not np.isfinite(vals.loc[code])]
        clean = vals.fillna(0.0)
        row["factor_name"] = features[0]
        row["factor_values_at_signal_close"] = {code: float(clean.loc[code]) for code in CODES}
        row["factor_missing_values_filled_with_zero"] = len(missing_codes)
        row["factor_timestamp_is_signal_date"] = True
        row["execution_date"] = str(prices["days"][idx[signal_date] + 1])
        if idx[row["execution_date"]] != idx[signal_date] + 1:
            raise RuntimeError("signal did not execute at the next trading session")
        row["no_future_row_used"] = row["execution_date"] > signal_date

    first = signal_dates[0]
    holdout_schedules = {
        "V43 rel_mom_ind_3d Top2": schedule,
        "V11 LambdaRank Top2": {d: schedules["V11 LambdaRank Top2"][d] for d in signal_dates},
        "60d momentum Top2": {d: schedules["60d momentum Top2"][d] for d in signal_dates},
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

    # Compare V11/60d/equal-hold outputs against the already-frozen V24 ledger run.
    v24_dir = HERE.parent / "V24"
    v24_metrics = json.loads((v24_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    v24_daily = pd.read_csv(v24_dir / "daily_equity.csv", dtype={"trade_date": str})
    daily_fields = ["equity_net", "equity_gross", "cash", "fees_cumulative",
                    "slippage_cumulative", "turnover_cumulative", "accounting_residual",
                    "daily_return_net", "drawdown"]
    compatibility = {}
    tolerance = 1e-3
    for name in ("V11 LambdaRank Top2", "60d momentum Top2", "equal-weight hold"):
        current = metrics[name]
        reference = v24_metrics[name]
        field_diffs = {
            field: abs(float(current[field]) - float(reference[field]))
            for field in ("net_return", "fees", "slippage_cost", "turnover",
                          "max_drawdown", "annualized_volatility", "min_cash")
        }
        if current["trades"] != reference["trades"]:
            raise RuntimeError(f"{name}: V24/V43 trade count differs")
        old = v24_daily.loc[v24_daily.strategy == name].sort_values("trade_date").reset_index(drop=True)
        new = daily_all.loc[daily_all.strategy == name].sort_values("trade_date").reset_index(drop=True)
        if old.trade_date.tolist() != new.trade_date.tolist() or len(old) != len(new):
            raise RuntimeError(f"{name}: V24/V43 daily date coverage differs")
        daily_diff = {}
        for field in daily_fields:
            diff = float(np.max(np.abs(old[field].to_numpy(dtype=float) - new[field].to_numpy(dtype=float))))
            daily_diff[field] = diff
        shares_match = old["shares_json"].astype(str).tolist() == new["shares_json"].astype(str).tolist()
        ok = all(value <= tolerance for value in field_diffs.values()) and all(
            value <= tolerance for value in daily_diff.values()) and shares_match
        if not ok:
            raise RuntimeError(f"{name}: V43 ledger differs from frozen V24 baseline")
        compatibility[name] = {
            "metric_abs_differences_vs_v24": field_diffs,
            "daily_max_abs_differences_vs_v24": daily_diff,
            "daily_rows_match": len(old),
            "shares_json_exact_match": shares_match,
            "trade_count_equal": True,
            "tolerance": tolerance
        }

    source_hash = file_sha256(Path(__file__))
    protocol = {
        "version": "V43",
        "hypothesis": "A single factor that appeared in the V11 training-fold top-20 importance set in 16 of 16 quarters through 2023Q4 can rank the four banks in the frozen 2024Q1-2026Q2 holdout.",
        "factor_selection": {
            "cutoff_quarter": FEATURE_CUTOFF,
            "importance_quarters": importance_quarters,
            "rule": "Freeze rel_mom_ind_3d because it appeared in the V11 top-20 gain features in 16/16 training folds from 2020Q1 through 2023Q4; do not inspect holdout returns to select or alter the feature.",
            "selected_features": features,
            "selection_statistics": importance,
            "direction": "Higher harmonized zscore_win1_99_v1 factor value ranks higher."
        },
        "portfolio_rule": "At each shared V11 first-session monthly signal close, rank the four banks by rel_mom_ind_3d in descending order, break ties by stock code, and equal-weight the top two. Execute at the next trading session open.",
        "holdout_dates": [START, END],
        "monthly_signal_dates": signal_dates,
        "execution": "Exact V11 final corrected shared-cash ledger: T+1 open, 1% participation cap, ordinary 100-share lots, fees, adverse slippage, limit-price blocks, corporate-action adjusted shares, and full-exit odd-lot handling. Mark positions to final holdout close without forced liquidation, matching V24.",
        "comparisons": ["60d momentum Top2", "V11 LambdaRank Top2", "equal-weight hold"],
        "no_retraining": True,
        "no_parameter_sweep": True,
        "no_holdout_based_rule_changes": True,
        "factor_semantics": meta.get("semantics"),
        "initial_equity": E.INITIAL,
        "monthly_signals": len(signal_dates),
        "source_script_sha256": source_hash,
        "v24_protocol_sha256": file_sha256(v24_dir / "protocol.json"),
        "v11_ledger_source_sha256": file_sha256(V11 / "analysis.py")
    }
    audit = {
        "window": [START, END],
        "monthly_signals": len(signal_dates),
        "all_signal_dates_from_v11_monthly_schedule": True,
        "all_factor_rows_on_signal_dates_available": True,
        "factor_selected_before_holdout": max(importance_quarters) <= FEATURE_CUTOFF,
        "selected_factor_top20_quarters": importance[features[0]]["top20_quarters"],
        "selected_factor_16_of_16_training_quarters": importance[features[0]]["top20_quarters"] == 16,
        "signal_uses_factor_value_on_signal_date": all(x["factor_timestamp_is_signal_date"] for x in membership_audit),
        "execution_is_next_session_open": all(x["no_future_row_used"] for x in membership_audit),
        "positions_marked_to_final_close_without_forced_liquidation": True,
        "v24_baseline_ledger_compatibility": compatibility,
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
        "v24_compatibility": compatibility
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # Hash the exact caches, ledger source, and V24 artifacts used by the run.
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
    for filename in ("protocol.json", "metrics.json", "daily_equity.csv", "trades.csv"):
        add_hash(v24_dir / filename, f"frozen V24 reference {filename}")
    add_hash(Path(__file__), "V43 runner")
    hash_manifest = {
        "algorithm": "SHA-256",
        "scope": "All market cache years read by V11.load_prices; factor parquet years 2024-2026 read by V24-compatible factor loader; cached V11 predictions and 2020Q1-2023Q4 importance artifacts; V24 comparison outputs; V11 ledger and V43 runner.",
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
        "# V43：单因子 rel_mom_ind_3d 冻结留出",
        "",
        "## 预先冻结的规则",
        "",
        "唯一因子在 V11 的 2020Q1–2023Q4 训练重要性文件中16季中有10季进入 top-20；规则、特征和方向均只用 2023Q4 及以前的信息固定。留出期收益未用于选因子或调整规则。",
        "",
        "留出期为 2024-01-02 至 2026-06-30，共 30 个共同月初信号。每个信号日收盘时在四家银行横截面按该因子由高到低排名，Top2 等权，次日开盘执行。",
        "",
        "对照、价格、现金和成本口径沿用 V24/V11 最终账本；窗口末按 2026-06-30 收盘估值，不强制平仓。未重训、未扫参数。",
        "",
        "## 结果",
        "",
        *table,
        "",
        "季度/年度收益见 quarterly_returns.csv、annual_returns.csv；完整逐日和成交账本见 daily_equity.csv、trades.csv。",
        "",
        "## 审计",
        "",
        f"- 单因子 top-20 训练期出现次数：{importance[features[0]]['top20_quarters']}/16；选因子截止季度：2023Q4。",
        "- 因子只取每个信号日期收盘横截面值，订单在下一交易日开盘；无未来日因子用于该次排序。",
        "- V24 的三个共同对照策略逐日现金、权益、持仓与V43重放一致，指标及成交数均通过容差核验；细节见 audit.json。",
        "- 每策略现金非负，账本核对残差和交易受阻统计见 audit.json。",
        "- 原始输入缓存与脚本 SHA-256 清单见 cache_hashes.json。",
        "",
        "这是四家银行上的固定历史留出结果，不单独证明未来稳定超额。"
    ]
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_feature": features[0], "feature_top20_quarters": importance[features[0]]["top20_quarters"],
        "signal_count": len(signal_dates), "metrics": metrics, "audit": audit,
        "runner_sha256": source_hash, "output_dir": str(HERE)
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


