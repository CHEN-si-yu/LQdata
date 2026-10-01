#!/usr/bin/env python3
"""V24: frozen consensus of stable V11 training-era factors, evaluated on 2024-2026 holdout."""
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
N_FEATURES = 4


def choose_features():
    counts = collections.defaultdict(list)
    files = sorted((V11 / "quarters").glob("*/result.json"))
    quarters = []
    for path in files:
        q = path.parent.name
        if q > FEATURE_CUTOFF:
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        feats = obj.get("top_features_by_gain", [])
        denom = sum(float(x["gain"]) for x in feats) or 1.0
        for item in feats:
            counts[item["feature"]].append(float(item["gain"]) / denom)
        quarters.append(q)
    if len(quarters) != 16:
        raise RuntimeError("expected 16 training-era quarter importance files")
    ranked = sorted(counts, key=lambda f: (-len(counts[f]), -float(np.mean(counts[f])), f))
    selected = ranked[:N_FEATURES]
    details = {f: {"top20_quarters": len(counts[f]), "mean_normalized_gain_when_top20": float(np.mean(counts[f]))} for f in selected}
    return selected, quarters, details


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


def main():
    features, importance_quarters, importance = choose_features()
    prices = E.load_prices()
    quarters = sorted(p.name for p in (V11 / "quarters").iterdir() if p.is_dir() and (p / "predictions.csv").is_file())
    pred = E.load_predictions(quarters)
    schedules, _ = E.schedules(pred, prices)
    v11_dates = sorted(d for d in pred.loc[pred.monthly_signal, "trade_date"].unique() if START <= d <= END)
    if not v11_dates:
        raise RuntimeError("no V11 monthly signal dates in the holdout")
    factors, meta = load_factor_rows(features)
    factor_dates = set(factors.trade_date.unique())
    if not set(v11_dates).issubset(factor_dates):
        missing = sorted(set(v11_dates) - factor_dates)
        raise RuntimeError(f"missing factor signal dates: {missing[:5]}")
    v24_schedule, membership_audit = build_factor_schedule(factors, v11_dates, features)
    idx = prices["index"]
    if idx[START] >= idx[END]:
        raise RuntimeError("invalid holdout window")
    for row in membership_audit:
        row["execution_date"] = str(prices["days"][idx[row["signal_date"]] + 1])
        if idx[row["execution_date"]] != idx[row["signal_date"]] + 1:
            raise RuntimeError("signal did not execute on next trading session")
    first = v11_dates[0]
    holdout_schedules = {
        "V24 stable-factor consensus Top2": v24_schedule,
        "V11 LambdaRank Top2": {d: schedules["V11 LambdaRank Top2"][d] for d in v11_dates},
        "60d momentum Top2": {d: schedules["60d momentum Top2"][d] for d in v11_dates},
        "equal-weight hold": {first: {c: 0.25 for c in CODES}},
    }
    all_daily, all_trades, metrics = [], [], {}
    for name, schedule in holdout_schedules.items():
        daily, trades, m = E.simulate(name, prices, schedule, START, END)
        all_daily.append(daily)
        all_trades.extend(trades)
        metrics[name] = m
    daily_all = pd.concat(all_daily, ignore_index=True)
    daily_all["gross_return_daily"] = daily_all.groupby("strategy", sort=False).equity_gross.pct_change().fillna(0.0)
    trades_all = pd.DataFrame(all_trades)
    quarterly, annual = [], []
    for _, group in daily_all.groupby("strategy", sort=False):
        quarterly.extend(period_returns(group, "Q"))
        annual.extend(period_returns(group, "Y"))
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {
        "version": "V24",
        "hypothesis": "A small consensus of factors repeatedly important in 2020Q1-2023Q4 V11 walk-forward training folds can rank banks in the unseen 2024Q1-2026Q2 holdout.",
        "factor_selection": {
            "cutoff_quarter": FEATURE_CUTOFF,
            "importance_quarters": importance_quarters,
            "rule": "Across the 16 V11 quarterly result files through 2023Q4, rank features by count in each fold's top-20 gain list; tie-break by mean gain normalized within that top-20 list, then feature name; freeze the first four.",
            "selected_features": features,
            "selection_statistics": importance,
            "direction": "Use the harmonized positive direction declared by zscore_win1_99_v1."
        },
        "portfolio_rule": "At each shared V11 monthly signal close, rank each selected feature cross-sectionally among the four banks, average ordinal ranks with equal weight, and equal-weight the two lowest average-rank banks. Execute at the next trading session open.",
        "holdout_dates": [START, END],
        "monthly_signal_dates": v11_dates,
        "execution": "V11 corrected shared-cash ledger; T+1 open, 1% participation cap, ordinary 100-share lots, fees, adverse slippage, limit-price blocks, corporate-action adjusted shares, full-exit odd-lot handling.",
        "comparisons": ["V11 LambdaRank Top2", "60d momentum Top2", "equal-weight hold"],
        "no_retraining": True,
        "no_parameter_sweep": True,
        "no_future_features": True,
        "factor_semantics": meta.get("semantics"),
        "source_script_sha256": source_hash
    }
    audit = {
        "window": [START, END],
        "monthly_signals": len(v11_dates),
        "factor_rows_available_for_all_signals": True,
        "signal_executes_next_session": True,
        "all_selected_features_from_training_period_only": max(importance_quarters) <= FEATURE_CUTOFF,
        "strategies": {name: {
            "cash_nonnegative": m["min_cash"] >= -1e-7,
            "blocked_trade_events": m["blocked_trade_events"],
            "max_abs_accounting_residual": m["max_abs_accounting_residual"],
            "max_positions": m["max_positions"],
            "trades": m["trades"],
            "min_cash": m["min_cash"]
        } for name, m in metrics.items()},
        "membership_audit_rows": len(membership_audit)
    }
    (HERE / "membership_audit.json").write_text(json.dumps(membership_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    daily_all.to_csv(HERE / "daily_equity.csv", index=False, float_format="%.10g")
    trades_all.to_csv(HERE / "trades.csv", index=False, float_format="%.10g")
    pd.DataFrame(quarterly).to_csv(HERE / "quarterly_returns.csv", index=False, float_format="%.10g")
    pd.DataFrame(annual).to_csv(HERE / "annual_returns.csv", index=False, float_format="%.10g")
    (HERE / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "metrics.json").write_text(json.dumps({"metrics": metrics, "feature_importance": importance, "selected_features": features}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = ["| Strategy | Net return | Annualized | Max drawdown | Ann. vol | Fees | Slippage | Trades |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in metrics.items():
        table.append("| {} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2f} | {:.2f} | {} |".format(
            name, m["net_return"], m["annualized_net_return"], m["max_drawdown"], m["annualized_volatility"],
            m["fees"], m["slippage_cost"], m["trades"]))
    report = [
        "# V24: stable-factor consensus on a frozen holdout",
        "",
        "## Protocol",
        "",
        "The four features were selected only from V11 model training-fold importance artifacts through 2023Q4. Feature selection stops before the 2024Q1 holdout. At each shared monthly signal close, the strategy averages the four feature ranks across the four banks and equal-weights the top two. Orders execute at the next session open.",
        "",
        "Selected features: " + ", ".join(features),
        "",
        "Holdout: {} through {} ({} monthly signals). No refit, parameter sweep, or use of the 2024Q1-2026Q2 strategy outcomes for feature choice.".format(START, END, len(v11_dates)),
        "",
        "## Results",
        "",
        *table,
        "",
        "Quarterly and annual results are in quarterly_returns.csv and annual_returns.csv. Full daily ledger, trades, frozen protocol, and membership audit are in this directory.",
        "",
        "This is a four-bank historical holdout. Positive return alone does not demonstrate stable future alpha."
    ]
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"selected_features": features, "metrics": metrics, "audit": audit, "source_sha256": source_hash}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


