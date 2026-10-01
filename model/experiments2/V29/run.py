#!/usr/bin/env python3
"""V29: quarterly causal updates to stable-factor selection, evaluated on 2024Q1-2026Q2."""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[k]="1"
import sys, json, hashlib, collections
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
V11 = HERE.parent / "V11"
V24 = HERE.parent / "V24"
sys.path.insert(0, str(V11))
import analysis as E
DATA = V11.parents[1] / "trainingdata"
CODES = E.CODES
START, END = "2024-01-02", "2026-06-30"
TOP_N, TOP_K = 20, 4


def choose_features(cutoff):
    counts = collections.defaultdict(list)
    quarters = []
    for path in sorted((V11 / "quarters").glob("*/result.json")):
        q = path.parent.name
        if q > cutoff:
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        items = obj.get("top_features_by_gain", [])
        denom = sum(float(x["gain"]) for x in items) or 1.0
        for item in items[:TOP_N]:
            counts[item["feature"]].append(float(item["gain"]) / denom)
        quarters.append(q)
    if not quarters:
        raise RuntimeError("no prior importance fold available")
    ranked = sorted(counts, key=lambda f: (-len(counts[f]), -float(np.mean(counts[f])), f))
    selected = ranked[:TOP_K]
    stats = {f: {"top20_quarters": len(counts[f]), "mean_normalized_gain_when_top20": float(np.mean(counts[f]))} for f in selected}
    return selected, quarters, stats


def load_factors(features):
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    if meta.get("semantics") != "zscore_win1_99_v1":
        raise RuntimeError("unexpected factor semantics")
    parts = []
    for year in sorted(int(y) for y in meta["built_years"] if 2024 <= int(y) <= 2026):
        frame = pq.read_table(DATA / "factors" / f"year={year}" / "data.parquet",
                              columns=["trade_date", "stock_code", *features],
                              filters=[("stock_code", "in", list(CODES))]).to_pandas()
        frame["trade_date"] = frame["trade_date"].astype(str).str[:10]
        frame["stock_code"] = frame["stock_code"].astype(str)
        parts.append(frame)
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("duplicate factor rows")
    return out, meta


def build_schedule(factors, signal_dates, feature_by_quarter, selection):
    by_date = {d: x.set_index("stock_code") for d, x in factors.groupby("trade_date", sort=False)}
    schedule, audit = {}, []
    for d in signal_dates:
        if d not in by_date:
            raise RuntimeError("missing factor rows at signal " + d)
        quarter = str(pd.Period(d, freq="Q"))
        features = feature_by_quarter[quarter]
        frame = by_date[d].reindex(CODES)[features].astype(float).fillna(0.0)
        ranks = {}
        for f in features:
            order = sorted(CODES, key=lambda c: (-float(frame.loc[c, f]), c))
            ranks[f] = {c: i + 1 for i, c in enumerate(order)}
        composite = {c: float(np.mean([ranks[f][c] for f in features])) for c in CODES}
        picked = sorted(CODES, key=lambda c: (composite[c], c))[:2]
        schedule[d] = {c: 0.5 for c in picked}
        audit.append({
            "signal_date": d, "signal_quarter": quarter,
            "importance_cutoff_quarter": selection[quarter]["cutoff"],
            "features_frozen_before_quarter": features,
            "selected_top2": picked, "average_ordinal_rank": composite,
            "feature_ranks": ranks, "signal_after_factor_observation": True,
            "execution_date": ""
        })
    return schedule, audit


def period_returns(daily, frequency):
    frame = daily.copy()
    frame["trade_date"] = pd.to_datetime(frame.trade_date)
    frame["period"] = (frame.trade_date.dt.to_period("Q").astype(str) if frequency == "Q"
                       else frame.trade_date.dt.year.astype(str))
    result = []
    for period, group in frame.groupby("period", sort=True):
        result.append({
            "period": period, "strategy": str(group.strategy.iloc[0]),
            "net_return": float(np.prod(1.0 + group.daily_return_net.to_numpy()) - 1.0),
            "gross_return_same_fills": float(np.prod(1.0 + group.gross_return_daily.to_numpy()) - 1.0)
        })
    return result


def main():
    prices = E.load_prices()
    quarter_dirs = sorted(p.name for p in (V11 / "quarters").iterdir()
                          if p.is_dir() and (p / "predictions.csv").is_file())
    pred = E.load_predictions(quarter_dirs)
    base_schedules, _ = E.schedules(pred, prices)
    signal_dates = sorted(d for d in pred.loc[pred.monthly_signal, "trade_date"].unique() if START <= d <= END)
    selection, feature_by_quarter = {}, {}
    for d in signal_dates:
        q = str(pd.Period(d, freq="Q"))
        if q in selection:
            continue
        cutoff = str(pd.Period(q, freq="Q") - 1)
        feats, quarters, stats = choose_features(cutoff)
        selection[q] = {"cutoff": cutoff, "importance_quarters": quarters, "features": feats, "statistics": stats}
        feature_by_quarter[q] = feats
    all_features = sorted(set(f for xs in feature_by_quarter.values() for f in xs))
    factors, meta = load_factors(all_features)
    factor_dates = set(factors.trade_date.unique())
    if not set(signal_dates).issubset(factor_dates):
        raise RuntimeError("factor rows missing for one or more signals")
    candidate, membership = build_schedule(factors, signal_dates, feature_by_quarter, selection)
    index = prices["index"]
    for row in membership:
        row["execution_date"] = str(prices["days"][index[row["signal_date"]] + 1])
        row["no_lookahead"] = index[row["execution_date"]] > index[row["signal_date"]]
        if not row["no_lookahead"]:
            raise RuntimeError("non-forward execution")
    v24_audit = json.loads((V24 / "membership_audit.json").read_text(encoding="utf-8"))
    v24_schedule = {r["signal_date"]: {c: 0.5 for c in r["selected_top2"]} for r in v24_audit}
    first = signal_dates[0]
    run_schedules = {
        "V29 quarterly-updated factor consensus Top2": candidate,
        "V24 frozen four-factor consensus Top2": v24_schedule,
        "V11 LambdaRank Top2": {d: base_schedules["V11 LambdaRank Top2"][d] for d in signal_dates},
        "60d momentum Top2": {d: base_schedules["60d momentum Top2"][d] for d in signal_dates},
        "equal-weight hold": {first: {c: 0.25 for c in CODES}}
    }
    daily_list, trade_list, metrics = [], [], {}
    for name, targets in run_schedules.items():
        daily, trades, m = E.simulate(name, prices, targets, START, END)
        daily_list.append(daily)
        trade_list.extend(trades)
        metrics[name] = m
    daily_all = pd.concat(daily_list, ignore_index=True)
    daily_all["gross_return_daily"] = daily_all.groupby("strategy", sort=False).equity_gross.pct_change().fillna(0.0)
    trades_all = pd.DataFrame(trade_list)
    quarterly, annual = [], []
    for _, group in daily_all.groupby("strategy", sort=False):
        quarterly.extend(period_returns(group, "Q"))
        annual.extend(period_returns(group, "Y"))
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {
        "version": "V29",
        "hypothesis": "Updating the stable-factor shortlist using only completed prior walk-forward training-fold importance may adapt better than freezing one 2023Q4 shortlist.",
        "window": [START, END], "monthly_signal_dates": signal_dates,
        "selection_rule": "For each signal quarter, use only V11 result.json importance folds through the immediately prior quarter; rank features by top-20 occurrence count, then mean normalized gain, then name; freeze the first four for that quarter.",
        "top_n_gain_list": TOP_N, "top_k_features": TOP_K,
        "selection_by_quarter": selection,
        "portfolio": "Equal-weight top two banks by average cross-sectional ordinal rank of the quarter's four frozen positive-direction features.",
        "execution": "V11 final corrected cash-account ledger; next-session open after close signal, 1% cap, lot and cost rules, corporate-action shares and odd-lot handling.",
        "comparisons": ["V24 frozen feature consensus", "V11 LambdaRank Top2", "60d momentum Top2", "equal-weight hold"],
        "no_retraining": True, "no_parameter_sweep": True, "feature_information_cutoff_precedes_signal_quarter": True,
        "factor_semantics": meta.get("semantics"), "source_script_sha256": source_hash
    }
    audit = {
        "monthly_signals": len(signal_dates), "quarter_count": len(selection),
        "factor_rows_present_for_all_signals": True,
        "each_importance_cutoff_strictly_before_signal_quarter": all(selection[q]["cutoff"] < q for q in selection),
        "all_executions_next_session": all(r["no_lookahead"] for r in membership),
        "selection_updates": {q: selection[q]["features"] for q in selection},
        "strategies": {name: {
            "min_cash": m["min_cash"], "cash_nonnegative": m["min_cash"] >= -1e-7,
            "blocked_trade_events": m["blocked_trade_events"], "trades": m["trades"],
            "max_abs_accounting_residual": m["max_abs_accounting_residual"],
            "max_positions": m["max_positions"]
        } for name, m in metrics.items()}
    }
    (HERE / "membership_audit.json").write_text(json.dumps(membership, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    daily_all.to_csv(HERE / "daily_equity.csv", index=False, float_format="%.10g")
    trades_all.to_csv(HERE / "trades.csv", index=False, float_format="%.10g")
    pd.DataFrame(quarterly).to_csv(HERE / "quarterly_returns.csv", index=False, float_format="%.10g")
    pd.DataFrame(annual).to_csv(HERE / "annual_returns.csv", index=False, float_format="%.10g")
    (HERE / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "metrics.json").write_text(json.dumps({"metrics": metrics}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = ["| Strategy | Net return | Annualized | Max drawdown | Ann. vol | Fees | Slippage | Trades |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in metrics.items():
        table.append("| {} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2f} | {:.2f} | {} |".format(
            name, m["net_return"], m["annualized_net_return"], m["max_drawdown"], m["annualized_volatility"],
            m["fees"], m["slippage_cost"], m["trades"]))
    report = [
        "# V29: quarterly causal feature updates on a frozen holdout", "",
        "At each holdout quarter boundary, the strategy uses only completed V11 training-fold importance artifacts through the previous quarter. It selects four factors by a fixed frequency/gain rule, then ranks the four banks at each monthly close and buys the two highest composites at the next open.",
        "",
        "Holdout: {} to {} ({} signals). The factor cutoffs and selected lists are shown in protocol.json; all cutoffs precede their signal quarter.".format(START, END, len(signal_dates)),
        "",
        "## Results", "", *table, "",
        "Quarterly/annual returns, daily equity, trades, membership audit, and cash-ledger audit are included in this directory.",
        "",
        "This is a four-bank historical holdout. It does not establish that the strategy will retain its performance."
    ]
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "selection_by_quarter": {q: selection[q]["features"] for q in selection},
                      "audit": audit, "source_hash": source_hash}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

