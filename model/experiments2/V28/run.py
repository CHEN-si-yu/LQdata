#!/usr/bin/env python3
"""V28: fixed two-window price-momentum rank consensus on a strict holdout."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd

import cash_engine as engine

ROOT = Path(__file__).resolve().parent
START = "2024-01-02"
END = "2026-06-30"
WINDOWS = (60, 250)
QUARTERS = [f"{y}Q{q}" for y in range(2024, 2027) for q in range(1, 5) if not (y == 2026 and q > 2)]
V24_NAME = "V24 stable-factor consensus Top2"
V28_NAME = "V28 60d+250d momentum rank Top2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_price_rank_schedule(prices: dict, signal_dates: list[str]) -> tuple[dict, pd.DataFrame]:
    adj_close = prices["adj_close"]
    dates = prices["days"]
    index = prices["index"]
    codes = tuple(engine.CODES)
    schedule: dict[str, dict[str, float]] = {}
    audit = []
    for date in signal_dates:
        i = index[date]
        if i < max(WINDOWS):
            raise RuntimeError(f"{date}: not enough trailing sessions")
        price_now = adj_close[i]
        price_60 = adj_close[i - 60]
        price_250 = adj_close[i - 250]
        ret_60 = price_now / price_60 - 1.0
        ret_250 = price_now / price_250 - 1.0
        if not (np.isfinite(price_now).all() and np.isfinite(price_60).all() and np.isfinite(price_250).all()
                and np.isfinite(ret_60).all() and np.isfinite(ret_250).all()):
            raise RuntimeError(f"{date}: one or more adjusted-price windows are not finite")
        rank_by_window = {}
        for window, returns in ((60, ret_60), (250, ret_250)):
            ordered = sorted(range(len(codes)), key=lambda j: (-float(returns[j]), codes[j]))
            rank_by_window[window] = {codes[j]: rank for rank, j in enumerate(ordered, start=1)}
        mean_rank = {
            code: (rank_by_window[60][code] + rank_by_window[250][code]) / 2.0
            for code in codes
        }
        ranked = sorted(codes, key=lambda code: (mean_rank[code], code))
        selected = set(ranked[:2])
        schedule[date] = {code: 0.5 for code in ranked[:2]}
        t1 = str(dates[i + 1])
        for j, code in enumerate(codes):
            audit.append({
                "signal_date": date,
                "stock_code": code,
                "adjusted_close_signal": float(price_now[j]),
                "adjusted_close_60_sessions_prior": float(price_60[j]),
                "date_60_sessions_prior": str(dates[i - 60]),
                "raw_60_session_adjusted_return": float(ret_60[j]),
                "rank_60d_descending": rank_by_window[60][code],
                "adjusted_close_250_sessions_prior": float(price_250[j]),
                "date_250_sessions_prior": str(dates[i - 250]),
                "raw_250_session_adjusted_return": float(ret_250[j]),
                "rank_250d_descending": rank_by_window[250][code],
                "mean_rank_50_50": float(mean_rank[code]),
                "ranked_mean_rank": ranked.index(code) + 1,
                "selected_top2": code in selected,
                "target_weight": 0.5 if code in selected else 0.0,
                "execution_date_t1_open": t1,
                "feature_cutoff": "signal-date close; both return windows end on signal_date",
                "uses_post_signal_data": False,
            })
    return schedule, pd.DataFrame(audit)


def verify_v24_source() -> tuple[dict, list[dict], dict[str, str]]:
    cache = ROOT / "V24_cache"
    original = ROOT.parent / "V24"
    digest = {}
    for filename in ("membership_audit.json", "protocol.json", "metrics.json"):
        cached = cache / filename
        source = original / filename
        if sha256(cached) != sha256(source):
            raise RuntimeError(f"V24 cached source mismatch: {filename}")
        digest[filename] = sha256(cached)
    protocol = json.loads((cache / "protocol.json").read_text(encoding="utf-8"))
    metrics = json.loads((cache / "metrics.json").read_text(encoding="utf-8"))
    membership = json.loads((cache / "membership_audit.json").read_text(encoding="utf-8"))
    return protocol, membership, metrics


def period_returns(daily: pd.DataFrame, freq: str) -> list[dict]:
    return engine.period_returns(daily.reset_index(drop=True), freq)


def main() -> None:
    prediction = engine.load_predictions(QUARTERS)
    prediction = prediction[(prediction.trade_date >= START) & (prediction.trade_date <= END)].copy()
    signal_dates = sorted(prediction.loc[prediction.monthly_signal, "trade_date"].unique().tolist())
    if len(signal_dates) != 30 or signal_dates[0] != START or signal_dates[-1] != "2026-06-01":
        raise RuntimeError(f"unexpected holdout monthly signal schedule: {len(signal_dates)} dates")

    prices = engine.load_prices()
    baseline, baseline_dates = engine.schedules(prediction, prices)
    if baseline_dates != signal_dates:
        raise RuntimeError("V11 baseline and V28 signal dates differ")
    v28_schedule, signal_audit = build_price_rank_schedule(prices, signal_dates)

    v24_protocol, v24_membership, v24_cached_metrics = verify_v24_source()
    v24_by_date = {row["signal_date"]: row for row in v24_membership}
    if sorted(v24_by_date) != signal_dates:
        raise RuntimeError("V24 membership evidence does not match the V28 holdout signals")
    v24_schedule = {}
    membership_check = []
    for date in signal_dates:
        picked = v24_by_date[date]["selected_top2"]
        if len(picked) != 2 or len(set(picked)) != 2:
            raise RuntimeError(f"{date}: malformed V24 Top2 membership")
        v24_schedule[date] = {code: 0.5 for code in picked}
        membership_check.append({"signal_date": date, "selected_top2": ";".join(picked),
                                 "execution_date": v24_by_date[date]["execution_date"],
                                 "copied_from_v24_membership_audit": True})

    first = signal_dates[0]
    schedules = {
        V28_NAME: v28_schedule,
        "60d momentum Top2": baseline["60d momentum Top2"],
        "V11 LambdaRank Top2": baseline["V11 LambdaRank Top2"],
        V24_NAME: v24_schedule,
        "equal-weight hold": baseline["equal-weight hold"],
    }
    if list(schedules["equal-weight hold"]) != [first]:
        raise RuntimeError("equal-weight hold does not use the first common executable signal")

    all_daily, all_trades, metrics = [], [], {}
    for name, schedule in schedules.items():
        daily, trades, result = engine.simulate(name, prices, schedule, START, END)
        daily["gross_return_daily"] = daily.equity_gross.pct_change().fillna(0.0)
        result.update({
            "ending_cash": float(daily.cash.iloc[-1]),
            "ending_positions": int(daily.positions.iloc[-1]),
            "ending_shares_json": str(daily.shares_json.iloc[-1]),
            "all_daily_cash_nonnegative": bool((daily.cash >= -1e-8).all()),
        })
        if result["blocked_trade_events"] != 0:
            raise RuntimeError(f"{name}: blocked trades={result['blocked_trade_events']}")
        if not result["all_daily_cash_nonnegative"]:
            raise RuntimeError(f"{name}: cash ledger went negative")
        if result["max_abs_accounting_residual"] > 1e-7:
            raise RuntimeError(f"{name}: accounting residual exceeds tolerance")
        all_daily.append(daily)
        all_trades.extend(trades)
        metrics[name] = result
        stem = name.replace(" ", "_").replace("/", "_")
        daily.to_csv(ROOT / f"{stem}_daily.csv", index=False, float_format="%.10g")
        pd.DataFrame(trades).to_csv(ROOT / f"{stem}_trades.csv", index=False, float_format="%.10g")

    daily_all = pd.concat(all_daily, ignore_index=True)
    lifecycle = daily_all.groupby("strategy").trade_date.agg(["min", "max", "count"])
    if lifecycle["min"].nunique() != 1 or lifecycle["max"].nunique() != 1 or lifecycle["count"].nunique() != 1:
        raise RuntimeError("strategy lifecycles differ")
    daily_all.to_csv(ROOT / "daily_equity.csv", index=False, float_format="%.10g")
    pd.DataFrame(all_trades).to_csv(ROOT / "trades.csv", index=False, float_format="%.10g")
    signal_audit.to_csv(ROOT / "signal_audit.csv", index=False, float_format="%.12g")
    pd.DataFrame(membership_check).to_csv(ROOT / "v24_membership_reproduction_audit.csv", index=False)

    quarters, years = [], []
    for _, group in daily_all.groupby("strategy", sort=False):
        quarters.extend(period_returns(group, "Q"))
        years.extend(period_returns(group, "Y"))
    quarterly = pd.DataFrame(quarters)
    annual = pd.DataFrame(years)
    quarterly.to_csv(ROOT / "quarterly_returns.csv", index=False, float_format="%.10g")
    annual.to_csv(ROOT / "annual_returns.csv", index=False, float_format="%.10g")

    # Re-running cached V24 membership through the same copied V11 ledger must reproduce V24.
    v24_reference = v24_cached_metrics["metrics"]
    v24_actual = metrics[V24_NAME]
    checked = ["net_return", "gross_return_same_fills", "fees", "slippage_cost", "turnover", "trades", "max_abs_accounting_residual"]
    v24_diffs = {k: float(v24_actual[k] - v24_reference[V24_NAME][k]) for k in checked}
    for key, delta in v24_diffs.items():
        if abs(delta) > 1e-8:
            raise RuntimeError(f"V24 ledger reproduction mismatch for {key}: {delta}")
    for name in ("60d momentum Top2", "V11 LambdaRank Top2", "equal-weight hold"):
        ref = v24_reference[name]
        got = metrics[name]
        for key in checked:
            if abs(float(got[key]) - float(ref[key])) > 1e-8:
                raise RuntimeError(f"V24 benchmark reproduction mismatch: {name} {key}")

    candidate = metrics[V28_NAME]
    comparisons = {}
    for name in ("60d momentum Top2", "V11 LambdaRank Top2", V24_NAME, "equal-weight hold"):
        comparisons[name] = {
            "net_return_difference_pp": 100 * (candidate["net_return"] - metrics[name]["net_return"]),
            "max_drawdown_difference_pp": 100 * (candidate["max_drawdown"] - metrics[name]["max_drawdown"]),
        }

    audit = {
        "holdout": {"start": START, "end": END, "signals": len(signal_dates), "trading_days_per_strategy": int(lifecycle["count"].iloc[0])},
        "signal_dates": signal_dates,
        "window_causality": {
            "price_series": "V11 price cache adj_close = close * adj_factor",
            "60d_return": "adj_close[signal_index] / adj_close[signal_index - 60] - 1; both endpoints at or before signal close",
            "250d_return": "adj_close[signal_index] / adj_close[signal_index - 250] - 1; both endpoints at or before signal close",
            "ranking": "rank each return descending across four banks; arithmetic mean of the two ordinal ranks with fixed 50/50 weights; select lowest two mean ranks; code ascending resolves ties",
            "execution": "T+1 next-session open",
            "uses_post_signal_data": False,
            "all_signal_windows_end_at_signal_close": bool((signal_audit["date_60_sessions_prior"] <= signal_audit["signal_date"]).all() and (signal_audit["date_250_sessions_prior"] <= signal_audit["signal_date"]).all()),
            "all_execution_dates_are_next_trading_day": bool((signal_audit.groupby("signal_date").execution_date_t1_open.nunique() == 1).all()),
        },
        "same_lifecycle": lifecycle.reset_index().to_dict(orient="records"),
        "v24_membership_reproduction": {"matches_cached_metrics": True, "checked_fields": checked, "metric_differences": v24_diffs},
        "ledger_checks": {name: {
            "cash_nonnegative": metrics[name]["all_daily_cash_nonnegative"],
            "min_cash": metrics[name]["min_cash"],
            "blocked_trade_events": metrics[name]["blocked_trade_events"],
            "max_abs_accounting_residual": metrics[name]["max_abs_accounting_residual"],
            "trades": metrics[name]["trades"],
            "ending_cash": metrics[name]["ending_cash"],
            "ending_positions": metrics[name]["ending_positions"],
        } for name in metrics},
        "period_tables": {"quarter_rows": len(quarterly), "annual_rows": len(annual), "end_date_is_signal_window_end_not_appended_exit_day": True},
    }
    write_json(ROOT / "audit.json", audit)

    cache_files = {}
    for quarter in QUARTERS:
        for filename in ("predictions.csv", "result.json"):
            p = ROOT / "quarters" / quarter / filename
            cache_files[f"quarters/{quarter}/{filename}"] = sha256(p)
    for filename in ("membership_audit.json", "protocol.json", "metrics.json"):
        cache_files[f"V24_cache/{filename}"] = sha256(ROOT / "V24_cache" / filename)
    write_json(ROOT / "cache_manifest.json", {"files": cache_files})

    protocol = {
        "version": "V28",
        "objective": "Test a fixed consensus rank of raw 60-session and 250-session adjusted-price returns on the frozen 2024-01-02 to 2026-06-30 holdout.",
        "holdout": [START, END],
        "monthly_signal_dates": signal_dates,
        "candidate_rule": "At the first available trading session of each month, calculate 60- and 250-session cumulative adjusted-close returns for each of the four banks, rank each window cross-sectionally in descending return order, average the two ordinal ranks with fixed equal 50/50 weights, and equal-weight the two lowest-mean-rank banks.",
        "causal_cutoff": "Both price windows end on the signal-date close; neither window uses a later price. Orders execute at the next trading session open.",
        "ties": "stock_code ascending within return ranks and mean-rank picks",
        "portfolio_execution": "V11 final shared-cash ledger; 50/50 Top2; sell before buy; same fills, costs, corporate-action share adjustment, constraints, and no forced liquidation at end.",
        "comparisons": ["60d momentum Top2", "V11 LambdaRank Top2", V24_NAME, "equal-weight hold"],
        "V24_source": "Frozen V24 membership_audit.json copied into V28 and verified byte-for-byte against the original; replayed under the same V11 ledger.",
        "constraints": {"retraining": False, "parameter_sweep": False, "holdout_driven_rule_changes": False, "gpu": False},
        "ledger_costs": {"initial_cash": engine.INITIAL, "commission": engine.COMMISSION, "minimum_commission": engine.MIN_COMMISSION, "transfer_fee": engine.TRANSFER, "sell_stamp_duty": engine.STAMP, "slippage": engine.SLIPPAGE, "participation_cap": engine.PARTICIPATION, "lot_size": engine.LOT},
        "frequency_tables": "Quarterly and annual returns include only this common lifecycle; no additional exit-only date is appended.",
        "source_hashes": {"run.py": sha256(ROOT / "run.py"), "cash_engine.py": sha256(ROOT / "cash_engine.py")},
    }
    write_json(ROOT / "protocol.json", protocol)
    hashes = {
        "run.py": sha256(ROOT / "run.py"),
        "cash_engine.py": sha256(ROOT / "cash_engine.py"),
        "protocol.json": sha256(ROOT / "protocol.json"),
        "V24_cache/membership_audit.json": sha256(ROOT / "V24_cache" / "membership_audit.json"),
    }

    lines = [
        "# V28 — 60d + 250d price momentum rank consensus",
        "",
        f"Strict holdout: {START} to {END}; {len(signal_dates)} monthly signals. All strategies use the same V11 final cash ledger and {int(lifecycle['count'].iloc[0])} trading-day lifecycle.",
        "",
        "At each month’s first available close, the candidate ranks four banks separately by their 60-session and 250-session adjusted-close returns. It averages those two ordinal ranks with fixed 50/50 weights, selects the two strongest mean ranks, and executes at the next session open. Both return windows end on the signal-date close.",
        "",
        "| Strategy | Net return | Same-fill gross | Max drawdown | Ann. volatility | Fees | Slippage | Trades | Min cash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['net_return']:.2%} | {m['gross_return_same_fills']:.2%} | {m['max_drawdown']:.2%} | {m['annualized_volatility']:.2%} | CNY {m['fees']:,.2f} | CNY {m['slippage_cost']:,.2f} | {m['trades']} | CNY {m['min_cash']:,.2f} |")
    lines += ["", "## Comparison with V28", ""]
    for name, d in comparisons.items():
        lines.append(f"- Versus {name}: net return {d['net_return_difference_pp']:+.2f} pp; max drawdown {d['max_drawdown_difference_pp']:+.2f} pp (candidate minus baseline).")
    lines += [
        "",
        "## Audit",
        "",
        "- Signal audit stores adjusted closes, exact 60/250-session endpoints, returns, individual ranks, mean ranks, selected members, and T+1 execution date for all 30 signals.",
        "- Both price windows end at or before the signal close. The candidate rule was fixed at 50/50; no holdout result was used to change it.",
        "- V24 membership was copied from its frozen audit and reproduced under this run’s V11 cash engine. All baseline metrics matched the V24 cached run within 1e-8.",
        f"- All ledgers kept nonnegative cash, had zero blocked trade events, and maximum accounting residual {max(x['max_abs_accounting_residual'] for x in metrics.values()):.3g}.",
        "- Quarterly and annual tables include the common holdout lifecycle only; no exit-only day was appended.",
        f"- SHA-256: run.py `{hashes['run.py']}`; protocol.json `{hashes['protocol.json']}`; cash_engine.py `{hashes['cash_engine.py']}`.",
        "",
        "The backtest is a four-bank historical holdout and does not establish stable future alpha. Protocol, detailed daily/trade records, and causal/ledger audits are provided alongside this report.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(ROOT / "hashes.json", hashes)
    metrics_doc = {"window": [START, END], "signal_count": len(signal_dates), "metrics": metrics, "comparisons": comparisons}
    write_json(ROOT / "metrics.json", metrics_doc)

    print("\n".join(lines))
    print("\nSHA256\n" + json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
