#!/usr/bin/env python3
"""V25: one-factor strict holdout ranking using V11's audited cash ledger."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import cash_engine as engine

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "trainingdata"
CODES = tuple(engine.CODES)
START = "2024-01-02"
END = "2026-06-30"
FACTOR = "rel_mom_ind_250d"
TRAIN_QUARTERS = [f"{y}Q{q}" for y in range(2020, 2024) for q in range(1, 5)]
OOS_QUARTERS = [f"{y}Q{q}" for y in range(2024, 2027) for q in range(1, 5) if not (y == 2026 and q > 2)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_factor_values(signal_dates: list[str]) -> pd.DataFrame:
    pieces = []
    for year in (2024, 2025, 2026):
        path = DATA / "factors" / f"year={year}" / "data.parquet"
        frame = pq.read_table(path, columns=["trade_date", "stock_code", FACTOR]).to_pandas()
        frame["trade_date"] = frame["trade_date"].astype(str).str[:10]
        frame["stock_code"] = frame["stock_code"].astype(str)
        frame = frame[frame["trade_date"].isin(signal_dates)]
        pieces.append(frame)
    values = pd.concat(pieces, ignore_index=True)
    values = values[values["stock_code"].isin(CODES)]
    if values.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("duplicate factor rows on signal dates")
    return values


def selected_schedule(factor: pd.DataFrame, signal_dates: list[str], prices: dict) -> tuple[dict, pd.DataFrame]:
    days = prices["days"]
    day_index = prices["index"]
    picks: dict[str, dict[str, float]] = {}
    audit_rows = []
    pivot = factor.pivot(index="trade_date", columns="stock_code", values=FACTOR).reindex(
        index=signal_dates, columns=CODES
    )
    for date in signal_dates:
        row = pivot.loc[date]
        if not np.isfinite(row.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{date}: factor values are missing/nonfinite for one or more banks")
        ranked = sorted(CODES, key=lambda code: (-float(row[code]), code))
        winners = set(ranked[:2])
        picks[date] = {code: 0.5 for code in ranked[:2]}
        di = day_index[date]
        if di + 1 >= len(days):
            raise RuntimeError(f"{date}: no next-session open for execution")
        execution_date = str(days[di + 1])
        for rank, code in enumerate(ranked, start=1):
            audit_rows.append({
                "signal_date": date,
                "feature": FACTOR,
                "stock_code": code,
                "feature_value_at_signal_close": float(row[code]),
                "rank_descending": rank,
                "selected_top2": code in winners,
                "target_weight": 0.5 if code in winners else 0.0,
                "execution_date_t1_open": execution_date,
                "selection_input_cutoff": "signal-date close",
                "uses_post_signal_data": False,
            })
    return picks, pd.DataFrame(audit_rows)


def read_importance_audit() -> pd.DataFrame:
    rows = []
    for quarter in TRAIN_QUARTERS:
        path = ROOT / "feature_importance_cache" / quarter / "result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        top = result.get("top_features_by_gain", [])
        hit = next(((i, x) for i, x in enumerate(top, 1) if x.get("feature") == FACTOR), None)
        rows.append({
            "training_oos_quarter": quarter,
            "v11_model_training_end": result.get("train_last"),
            "feature": FACTOR,
            "top20_rank_by_gain": hit[0] if hit else "",
            "gain": hit[1]["gain"] if hit else "",
            "entered_top20": hit is not None,
            "importance_source": "V11 saved result.json; top_features_by_gain",
        })
    return pd.DataFrame(rows)


def write_periods(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual, quarterly = [], []
    for strategy, group in daily.groupby("strategy", sort=False):
        annual.extend(engine.period_returns(group.reset_index(drop=True), "Y"))
        quarterly.extend(engine.period_returns(group.reset_index(drop=True), "Q"))
    annual_df = pd.DataFrame(annual)
    quarterly_df = pd.DataFrame(quarterly)
    annual_df.to_csv(ROOT / "annual_returns.csv", index=False, float_format="%.10g")
    quarterly_df.to_csv(ROOT / "quarterly_returns.csv", index=False, float_format="%.10g")
    return annual_df, quarterly_df


def main() -> None:
    quarters = OOS_QUARTERS
    pred = engine.load_predictions(quarters)
    pred = pred[(pred.trade_date >= START) & (pred.trade_date <= END)].copy()
    signal_dates = sorted(pred.loc[pred.monthly_signal, "trade_date"].unique().tolist())
    if not signal_dates or signal_dates[0] != START:
        raise RuntimeError(f"unexpected first holdout signal: {signal_dates[:1]}")

    prices = engine.load_prices()
    schedules, base_dates = engine.schedules(pred, prices)
    if signal_dates != base_dates:
        raise RuntimeError("factor and V11 baseline monthly signal dates do not match")
    factor = read_factor_values(signal_dates)
    v25_schedule, signal_audit = selected_schedule(factor, signal_dates, prices)
    schedules = {
        "V25 rel_mom_ind_250d Top2": v25_schedule,
        "60d momentum Top2": schedules["60d momentum Top2"],
        "V11 LambdaRank Top2": schedules["V11 LambdaRank Top2"],
        "equal-weight hold": schedules["equal-weight hold"],
    }

    strategy_daily, all_trades, metrics_by = [], [], {}
    for name, schedule in schedules.items():
        daily, trades, metrics = engine.simulate(name, prices, schedule, START, END)
        final = daily.iloc[-1]
        metrics.update({
            "ending_cash": float(final.cash),
            "ending_positions": int(final.positions),
            "ending_shares_json": final.shares_json,
            "all_daily_cash_nonnegative": bool((daily.cash >= -1e-8).all()),
        })
        if metrics["blocked_trade_events"] != 0:
            raise RuntimeError(f"{name}: blocked trade events={metrics['blocked_trade_events']}")
        if not metrics["all_daily_cash_nonnegative"]:
            raise RuntimeError(f"{name}: cash ledger went negative")
        if metrics["max_abs_accounting_residual"] > 1e-7:
            raise RuntimeError(f"{name}: accounting residual exceeds tolerance")
        strategy_daily.append(daily)
        all_trades.extend(trades)
        metrics_by[name] = metrics
        stem = name.replace(" ", "_").replace("/", "_")
        daily.to_csv(ROOT / f"{stem}_daily.csv", index=False, float_format="%.10g")
        pd.DataFrame(trades).to_csv(ROOT / f"{stem}_trades.csv", index=False, float_format="%.10g")

    daily_all = pd.concat(strategy_daily, ignore_index=True)
    lifecycle_counts = daily_all.groupby("strategy").trade_date.agg(["min", "max", "count"])
    if lifecycle_counts["min"].nunique() != 1 or lifecycle_counts["max"].nunique() != 1 or lifecycle_counts["count"].nunique() != 1:
        raise RuntimeError("strategies do not share the same daily lifecycle")
    daily_all.to_csv(ROOT / "daily_equity.csv", index=False, float_format="%.10g")
    pd.DataFrame(all_trades).to_csv(ROOT / "trades.csv", index=False, float_format="%.10g")
    signal_audit.to_csv(ROOT / "signal_audit.csv", index=False, float_format="%.12g")

    importance = read_importance_audit()
    importance.to_csv(ROOT / "feature_selection_audit.csv", index=False, float_format="%.12g")
    top20_count = int(importance.entered_top20.sum())
    if top20_count != 13 or len(importance) != 16:
        raise RuntimeError(f"predeclared factor-selection evidence changed: {top20_count}/{len(importance)}")

    annual, quarterly = write_periods(daily_all)
    candidate = metrics_by["V25 rel_mom_ind_250d Top2"]
    comparisons = {}
    for baseline in ("60d momentum Top2", "V11 LambdaRank Top2", "equal-weight hold"):
        ref = metrics_by[baseline]
        comparisons[baseline] = {
            "net_return_difference_pp": 100 * (candidate["net_return"] - ref["net_return"]),
            "max_drawdown_difference_pp": 100 * (candidate["max_drawdown"] - ref["max_drawdown"]),
        }
    signal_factor_csv = ROOT / "signal_audit.csv"
    audit = {
        "holdout_window": {"start": START, "end": END, "daily_rows_per_strategy": int(lifecycle_counts["count"].iloc[0])},
        "same_lifecycle": lifecycle_counts.reset_index().to_dict(orient="records"),
        "signal_count": len(signal_dates),
        "signal_dates": signal_dates,
        "factor": FACTOR,
        "factor_values_complete": bool(len(factor) == len(signal_dates) * len(CODES) and factor[FACTOR].notna().all()),
        "feature_selection_precommitted_from_v11_training_period_only": {
            "quarter_span": "2020Q1-2023Q4",
            "quarter_count": len(importance),
            "top20_count": top20_count,
            "rule": "The factor was selected solely because it appeared among V11's saved top 20 gain features in 13/16 training-window quarterly fits; no 2024-2026 holdout return was used to choose the factor or rule.",
        },
        "causality": {
            "factor_input": "daily rel_mom_ind_250d values keyed to the monthly signal-date row (signal-date close snapshot)",
            "ranking": "descending among the same four banks; ties broken by stock_code ascending",
            "execution": "next trading session open (T+1)",
            "uses_post_signal_data": False,
            "signal_rows_with_t1_execution": int(signal_audit.execution_date_t1_open.notna().sum()),
        },
        "ledger_checks": {
            strategy: {
                "same_cash_engine": "V11 analysis.py simulate() unchanged",
                "blocked_trade_events": metrics_by[strategy]["blocked_trade_events"],
                "min_cash": metrics_by[strategy]["min_cash"],
                "all_daily_cash_nonnegative": metrics_by[strategy]["all_daily_cash_nonnegative"],
                "max_abs_accounting_residual": metrics_by[strategy]["max_abs_accounting_residual"],
                "ending_cash": metrics_by[strategy]["ending_cash"],
                "ending_shares_json": metrics_by[strategy]["ending_shares_json"],
            }
            for strategy in metrics_by
        },
        "period_tables": {
            "quarter_rows": len(quarterly),
            "annual_rows": len(annual),
            "quarterly_excludes_no_lifecycle_days": True,
        },
        "factor_signal_file_sha256": sha256(signal_factor_csv),
    }
    atomic_json(ROOT / "audit.json", audit)

    manifest = {"source": "V11 cached out-of-sample predictions and saved model result summaries; trainingdata factor panel", "files": {}}
    for q in quarters:
        for fn in ("predictions.csv", "result.json"):
            p = ROOT / "quarters" / q / fn
            manifest["files"][f"quarters/{q}/{fn}"] = sha256(p)
    for q in TRAIN_QUARTERS:
        p = ROOT / "feature_importance_cache" / q / "result.json"
        manifest["files"][f"feature_importance_cache/{q}/result.json"] = sha256(p)
    atomic_json(ROOT / "cache_manifest.json", manifest)

    protocol = {
        "experiment": "V25 single-factor rel_mom_ind_250d strict holdout Top2",
        "decision_rule": "At the first available trading session of each month, rank four banks by rel_mom_ind_250d from the signal-date daily factor row; select the top two equally; submit target weights at the next trading session open.",
        "factor_choice_basis": "V11's saved feature_gain ranking during 2020Q1-2023Q4: rel_mom_ind_250d entered top20 in 13 of 16 quarter fits. The factor/rule was fixed before inspecting 2024-2026 holdout returns.",
        "holdout": {"start": START, "end": END, "signal_count": len(signal_dates)},
        "single_factor_only": FACTOR,
        "ties": "stock_code ascending",
        "execution_and_ledger": "V11 final cash/trade/split-factor/odd-lot engine; T+1 open; shared cash; sell before buy; no forced liquidation at lifecycle end.",
        "controls": {"training": False, "parameter_sweep": False, "gpu": False, "oos_rule_changes": False},
        "benchmarks": ["60d momentum Top2", "V11 LambdaRank Top2", "equal-weight hold"],
        "prices_and_costs": {"initial_cash": engine.INITIAL, "lot_size": engine.LOT, "commission": engine.COMMISSION, "minimum_commission": engine.MIN_COMMISSION, "transfer_fee": engine.TRANSFER, "sell_stamp_duty": engine.STAMP, "adverse_slippage": engine.SLIPPAGE, "max_participation": engine.PARTICIPATION},
        "frequency_tables": "Quarter/annual returns cover only the same signal-window lifecycle; there is no appended exit-only day in this window.",
        "artifacts": "daily_equity.csv, strategy-specific daily/trades, trades.csv, quarterly_returns.csv, annual_returns.csv, feature_selection_audit.csv, signal_audit.csv, audit.json, cache_manifest.json.",
    }
    atomic_json(ROOT / "protocol.json", protocol)

    metrics_doc = {
        "experiment": "V25",
        "start": START,
        "end": END,
        "signal_count": len(signal_dates),
        "strategies": metrics_by,
        "comparisons": comparisons,
        "factor_selection_evidence": {"period": "2020Q1-2023Q4", "top20_quarters": top20_count, "total_quarters": len(importance)},
    }
    atomic_json(ROOT / "metrics.json", metrics_doc)

    labels = list(metrics_by)
    lines = [
        "# V25 — single-factor strict holdout test",
        "",
        f"Holdout lifecycle: {START} to {END}; {len(signal_dates)} monthly signal dates. All strategies use the same V11 cash, fills, split adjustment, fees, and daily marks.",
        "",
        "The only candidate rule ranks the four banks by `rel_mom_ind_250d` at the first session's close of each month, selects the top two equally, and executes at the next session's open. Factor choice is fixed from V11 model training evidence: it appeared in the saved top-20 gain features in 13 of 16 quarters from 2020Q1 through 2023Q4. No holdout return was used to select the factor or alter the rule.",
        "",
        "| Strategy | Net return | Same-fill gross return | Max drawdown | Annualized vol. | Fees | Slippage | Trades | Min cash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in labels:
        m = metrics_by[name]
        lines.append(f"| {name} | {m['net_return']:.2%} | {m['gross_return_same_fills']:.2%} | {m['max_drawdown']:.2%} | {m['annualized_volatility']:.2%} | CNY {m['fees']:,.2f} | CNY {m['slippage_cost']:,.2f} | {m['trades']} | CNY {m['min_cash']:,.2f} |")
    lines += ["", "## Comparison to V25", ""]
    for baseline, diff in comparisons.items():
        lines.append(f"- Versus {baseline}: net return {diff['net_return_difference_pp']:+.2f} pp; maximum drawdown {diff['max_drawdown_difference_pp']:+.2f} pp (candidate minus baseline).")
    lines += [
        "",
        "## Audit",
        "",
        f"- Training-period feature selection: {top20_count}/16 quarterly V11 fits included this factor in their saved top 20 gain features.",
        "- Signal audit records four factor values and ranks per monthly date, the T+1 execution date, and a false post-signal-data flag.",
        "- All strategies share the same daily lifecycle; no end-of-period liquidation is imposed. Quarterly and annual return tables cover the lifecycle only.",
        f"- Cash remained nonnegative, blocked trade events were zero, and maximum accounting residual was {max(x['max_abs_accounting_residual'] for x in metrics_by.values()):.3g}.",
        "- The V11 Ranker baseline uses already-saved OOS scores; no fitting was run. The candidate uses only the single fixed factor and no parameter sweep.",
        "",
        "A positive or superior backtest outcome would not establish stable alpha. See `protocol.json`, `audit.json`, `signal_audit.csv`, `feature_selection_audit.csv`, daily/trade files, and period tables for the reproducible record.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
