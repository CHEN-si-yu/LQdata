#!/usr/bin/env python3
"""V30: fixed 50/50 mix of frozen V26 factor Top2 and 60d momentum Top2."""
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
V26_NAME = "V26 mf_tier_flow_agreement_20 Top2"
MOM_NAME = "60d momentum Top2"
V11_NAME = "V11 LambdaRank Top2"
HOLD_NAME = "equal-weight hold"
MIX_NAME = "V30 V26+60d momentum 50/50 Top2 mix"
QUARTERS = [f"{y}Q{q}" for y in range(2024, 2027) for q in range(1, 5) if not (y == 2026 and q > 2)]
FACTOR = "mf_tier_flow_agreement_20"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_frozen_protocol() -> tuple[dict, str]:
    path = ROOT / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("version") != "V30" or protocol.get("rule_frozen_before_backtest") is not True:
        raise RuntimeError("V30 protocol is missing or not frozen")
    sleeves = protocol.get("sleeves", [])
    if len(sleeves) != 2 or sleeves[0].get("factor") != FACTOR:
        raise RuntimeError("frozen V30 sleeve definition does not match the specified factor")
    if [x.get("equity_allocation") for x in sleeves] != [0.5, 0.5] or [x.get("top2_each_weight") for x in sleeves] != [0.25, 0.25]:
        raise RuntimeError("V30 rule weights are not the predeclared 50/50, 25% per Top2 member")
    if protocol.get("holdout_dates") != [START, END]:
        raise RuntimeError("V30 protocol window mismatch")
    return protocol, sha256(path)


def verify_v26_inputs() -> tuple[dict, dict, list[dict], dict[str, str]]:
    cache = ROOT / "V26_cache"
    source = ROOT.parent / "V26"
    names = ("membership_audit.json", "protocol.json", "metrics.json", "daily_equity.csv", "trades.csv")
    digest = {}
    for name in names:
        local, original = cache / name, source / name
        if sha256(local) != sha256(original):
            raise RuntimeError(f"V26 cached input differs from source: {name}")
        digest[name] = sha256(local)
    v26_protocol = json.loads((cache / "protocol.json").read_text(encoding="utf-8"))
    selected = v26_protocol.get("factor_selection", {}).get("selected_features")
    if selected != [FACTOR] or v26_protocol.get("holdout_dates") != [START, END]:
        raise RuntimeError("V26 cache is not the expected frozen single-factor holdout")
    membership = json.loads((cache / "membership_audit.json").read_text(encoding="utf-8"))
    metrics = json.loads((cache / "metrics.json").read_text(encoding="utf-8"))
    return v26_protocol, metrics, membership, digest


def make_mix(v26_membership: list[dict], momentum: dict, signal_dates: list[str], price: dict) -> tuple[dict, pd.DataFrame]:
    ix = price["index"]
    days = price["days"]
    v26_by_date = {row["signal_date"]: row for row in v26_membership}
    if sorted(v26_by_date) != signal_dates:
        raise RuntimeError("V26 membership dates do not equal the common V11 monthly schedule")
    codes = tuple(engine.CODES)
    schedule, rows = {}, []
    for date in signal_dates:
        entry = v26_by_date[date]
        if entry.get("factor_name") != FACTOR or entry.get("factor_timestamp_is_signal_date") is not True or entry.get("no_future_row_used") is not True:
            raise RuntimeError(f"{date}: V26 membership lacks signal-date/no-lookahead audit")
        v26_top = list(entry["selected_top2"])
        mom_top = list(momentum[date])
        if len(v26_top) != 2 or len(mom_top) != 2:
            raise RuntimeError(f"{date}: expected two names from each Top2 sleeve")
        target = {code: 0.0 for code in codes}
        for code in v26_top:
            target[code] += 0.25
        for code in mom_top:
            target[code] += 0.25
        total = sum(target.values())
        if abs(total - 1.0) > 1e-12:
            raise RuntimeError(f"{date}: target weights sum to {total}")
        overlaps = sorted(set(v26_top) & set(mom_top))
        schedule[date] = {code: weight for code, weight in target.items() if weight > 0}
        di = ix[date]
        if entry.get("execution_date") != str(days[di + 1]):
            raise RuntimeError(f"{date}: V26 audit execution date is not T+1")
        row = {
            "signal_date": date,
            "execution_date_t1_open": str(days[di + 1]),
            "v26_factor_top2": ";".join(v26_top),
            "momentum_60d_top2": ";".join(mom_top),
            "overlap_names": ";".join(overlaps),
            "overlap_count": len(overlaps),
            "v26_sleeve_allocation": 0.5,
            "momentum_sleeve_allocation": 0.5,
            "target_weight_sum": total,
        }
        for code in codes:
            row[f"target_weight_{code}"] = target[code]
        rows.append(row)
    return schedule, pd.DataFrame(rows)


def assert_same_metrics(actual: dict, expected: dict, name: str) -> dict:
    keys = ("net_return", "gross_return_same_fills", "fees", "slippage_cost", "turnover", "trades",
            "max_drawdown", "annualized_volatility", "min_cash", "max_abs_accounting_residual")
    diff = {key: float(actual[key]) - float(expected[key]) for key in keys}
    for key, value in diff.items():
        if abs(value) > 1e-8:
            raise RuntimeError(f"{name} baseline reproduction mismatch {key}: {value}")
    return {"absolute_metric_differences": {k: abs(v) for k, v in diff.items()}, "checked_metrics": list(keys)}


def main() -> None:
    frozen_protocol, protocol_sha_before = check_frozen_protocol()
    v26_protocol, v26_metrics_cache, v26_membership, v26_cache_digests = verify_v26_inputs()
    prediction = engine.load_predictions(QUARTERS)
    prediction = prediction[(prediction.trade_date >= START) & (prediction.trade_date <= END)].copy()
    signal_dates = sorted(prediction.loc[prediction.monthly_signal, "trade_date"].unique().tolist())
    if len(signal_dates) != 30 or signal_dates[0] != START or signal_dates[-1] != "2026-06-01":
        raise RuntimeError("unexpected common 30-signal holdout schedule")

    prices = engine.load_prices()
    base_schedules, base_dates = engine.schedules(prediction, prices)
    if signal_dates != base_dates:
        raise RuntimeError("baseline signal dates differ from frozen V11 schedule")
    factor_schedule = {row["signal_date"]: {code: 0.5 for code in row["selected_top2"]} for row in v26_membership}
    mix_schedule, monthly_targets = make_mix(v26_membership, base_schedules[MOM_NAME], signal_dates, prices)
    if set(factor_schedule) != set(signal_dates):
        raise RuntimeError("V26 component schedule is incomplete")

    first = signal_dates[0]
    schedules = {
        MIX_NAME: mix_schedule,
        V26_NAME: factor_schedule,
        MOM_NAME: base_schedules[MOM_NAME],
        V11_NAME: base_schedules[V11_NAME],
        HOLD_NAME: base_schedules[HOLD_NAME],
    }
    if list(schedules[HOLD_NAME]) != [first]:
        raise RuntimeError("equal-weight hold does not start from the first signal")

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
        if result["blocked_trade_events"] != 0 or not result["all_daily_cash_nonnegative"]:
            raise RuntimeError(f"{name}: ledger cash/blocked-trade check failed")
        if result["max_abs_accounting_residual"] > 1e-7:
            raise RuntimeError(f"{name}: accounting residual exceeds tolerance")
        all_daily.append(daily)
        all_trades.extend(trades)
        metrics[name] = result
        stem = name.replace(" ", "_").replace("/", "_")
        daily.to_csv(ROOT / f"{stem}_daily.csv", index=False, float_format="%.10g")
        pd.DataFrame(trades).to_csv(ROOT / f"{stem}_trades.csv", index=False, float_format="%.10g")

    # Every baseline must reproduce the V26 frozen ledger/metric record before interpreting V30.
    reference_metrics = v26_metrics_cache["metrics"]
    reproduction = {
        name: assert_same_metrics(metrics[name], reference_metrics[name], name)
        for name in (V26_NAME, V11_NAME, MOM_NAME, HOLD_NAME)
    }
    reference_daily = pd.read_csv(ROOT / "V26_cache" / "daily_equity.csv", dtype={"trade_date": str})
    reproduced_daily = pd.concat(all_daily, ignore_index=True)
    daily_diff_audit = {}
    reference_trades = pd.read_csv(ROOT / "V26_cache" / "trades.csv", dtype={"trade_date": str})
    reproduced_trades = pd.DataFrame(all_trades)
    trade_diff_audit = {}
    for name in (V26_NAME, V11_NAME, MOM_NAME, HOLD_NAME):
        left = reference_daily[reference_daily.strategy == name].sort_values("trade_date").reset_index(drop=True)
        right = reproduced_daily[reproduced_daily.strategy == name].sort_values("trade_date").reset_index(drop=True)
        if len(left) != len(right) or left.trade_date.tolist() != right.trade_date.tolist():
            raise RuntimeError(f"{name}: V26 daily date rows differ")
        max_diff = {}
        for column in ("equity_net", "equity_gross", "cash", "fees_cumulative", "slippage_cumulative", "turnover_cumulative", "accounting_residual", "daily_return_net", "drawdown"):
            max_diff[column] = float(np.max(np.abs(left[column].to_numpy(float) - right[column].to_numpy(float))))
            if max_diff[column] > 0.001:
                raise RuntimeError(f"{name}: V26 daily path mismatch for {column}: {max_diff[column]}")
        if left.shares_json.tolist() != right.shares_json.tolist():
            raise RuntimeError(f"{name}: V26 daily share inventory mismatch")
        daily_diff_audit[name] = {"rows": len(left), "max_abs_daily_field_differences": max_diff, "shares_json_exact_match": True}
        left_trades = reference_trades[reference_trades.strategy == name].sort_values(
            ["trade_date", "stock_code", "side"]
        ).reset_index(drop=True)
        right_trades = reproduced_trades[reproduced_trades.strategy == name].sort_values(
            ["trade_date", "stock_code", "side"]
        ).reset_index(drop=True)
        if len(left_trades) != len(right_trades):
            raise RuntimeError(f"{name}: V26 trade count mismatch")
        text_columns = ["trade_date", "strategy", "stock_code", "side", "reason"]
        if any(not left_trades[c].equals(right_trades[c].astype(left_trades[c].dtype)) for c in text_columns):
            raise RuntimeError(f"{name}: V26 trade identity mismatch")
        numeric_columns = ["shares", "open", "fill", "notional", "fee", "slippage_cost"]
        trade_diffs = {
            c: float(np.max(np.abs(left_trades[c].to_numpy(float) - right_trades[c].to_numpy(float)))) if len(left_trades) else 0.0
            for c in numeric_columns
        }
        if any(value > 1e-5 for value in trade_diffs.values()):
            raise RuntimeError(f"{name}: V26 trade field mismatch: {trade_diffs}")
        trade_diff_audit[name] = {"rows": len(left_trades), "max_abs_trade_field_differences": trade_diffs,
                                  "trade_identity_exact_match": True, "numeric_tolerance": 1e-5}

    daily_all = reproduced_daily
    lifecycle = daily_all.groupby("strategy").trade_date.agg(["min", "max", "count"])
    if lifecycle["min"].nunique() != 1 or lifecycle["max"].nunique() != 1 or lifecycle["count"].nunique() != 1:
        raise RuntimeError("strategies do not share the same lifecycle")
    monthly_targets.to_csv(ROOT / "monthly_target_weights.csv", index=False, float_format="%.12g")
    daily_all.to_csv(ROOT / "daily_equity.csv", index=False, float_format="%.10g")
    pd.DataFrame(all_trades).to_csv(ROOT / "trades.csv", index=False, float_format="%.10g")

    quarter_returns, annual_returns = [], []
    for _, group in daily_all.groupby("strategy", sort=False):
        quarter_returns.extend(engine.period_returns(group.reset_index(drop=True), "Q"))
        annual_returns.extend(engine.period_returns(group.reset_index(drop=True), "Y"))
    quarterly_df, annual_df = pd.DataFrame(quarter_returns), pd.DataFrame(annual_returns)
    quarterly_df.to_csv(ROOT / "quarterly_returns.csv", index=False, float_format="%.10g")
    annual_df.to_csv(ROOT / "annual_returns.csv", index=False, float_format="%.10g")

    mix = metrics[MIX_NAME]
    comparisons = {}
    for name in (V26_NAME, MOM_NAME, V11_NAME, HOLD_NAME):
        comparisons[name] = {
            "net_return_difference_pp": 100 * (mix["net_return"] - metrics[name]["net_return"]),
            "max_drawdown_difference_pp": 100 * (mix["max_drawdown"] - metrics[name]["max_drawdown"]),
        }
    overlap_hist = {str(k): int(v) for k, v in monthly_targets.overlap_count.value_counts().sort_index().items()}
    audit = {
        "window": [START, END],
        "signal_count": len(signal_dates),
        "daily_rows_per_strategy": int(lifecycle["count"].iloc[0]),
        "signal_dates_match_v11_v26": True,
        "factor_name_and_weight_protocol_frozen_before_run": True,
        "v26_factor_membership_all_dates_has_signal_close_and_no_future_flags": all(
            row.get("factor_timestamp_is_signal_date") is True and row.get("no_future_row_used") is True for row in v26_membership
        ),
        "monthly_mix_checks": {
            "total_target_weights_equal_one_each_month": bool(np.allclose(monthly_targets.target_weight_sum.to_numpy(float), 1.0, atol=1e-12, rtol=0)),
            "min_target_weight_sum": float(monthly_targets.target_weight_sum.min()),
            "max_target_weight_sum": float(monthly_targets.target_weight_sum.max()),
            "overlap_month_counts_by_number_of_shared_names": overlap_hist,
            "max_per_name_target_weight": float(monthly_targets[[f"target_weight_{c}" for c in engine.CODES]].to_numpy().max()),
        },
        "execution_t1_open": True,
        "v26_baseline_reproduction": {
            "metric_differences": reproduction,
            "daily_path_differences": daily_diff_audit,
            "trade_path_differences": trade_diff_audit,
            "tolerance_for_daily_float_fields": 0.001,
        },
        "same_lifecycle": lifecycle.reset_index().to_dict(orient="records"),
        "ledger_checks": {name: {
            "cash_nonnegative": metrics[name]["all_daily_cash_nonnegative"],
            "min_cash": metrics[name]["min_cash"],
            "blocked_trade_events": metrics[name]["blocked_trade_events"],
            "fees": metrics[name]["fees"],
            "slippage_cost": metrics[name]["slippage_cost"],
            "trades": metrics[name]["trades"],
            "max_abs_accounting_residual": metrics[name]["max_abs_accounting_residual"],
            "ending_cash": metrics[name]["ending_cash"],
            "ending_positions": metrics[name]["ending_positions"],
        } for name in metrics},
        "period_return_rows": {"quarterly": len(quarterly_df), "annual": len(annual_df), "no_exit_only_day_appended": True},
        "protocol_hash_unchanged_since_pre_run_freeze": sha256(ROOT / "protocol.json") == protocol_sha_before,
    }
    write_json(ROOT / "audit.json", audit)

    cache_manifest = {"v11_quarter_cache": {}, "v26_baseline_inputs": v26_cache_digests}
    for quarter in QUARTERS:
        for filename in ("predictions.csv", "result.json"):
            path = ROOT / "quarters" / quarter / filename
            cache_manifest["v11_quarter_cache"][f"quarters/{quarter}/{filename}"] = sha256(path)
    write_json(ROOT / "cache_manifest.json", cache_manifest)

    hashes = {
        "run.py": sha256(ROOT / "run.py"),
        "protocol.json_frozen_before_results": protocol_sha_before,
        "cash_engine.py": sha256(ROOT / "cash_engine.py"),
        "V26_cache/membership_audit.json": sha256(ROOT / "V26_cache" / "membership_audit.json"),
    }
    write_json(ROOT / "hashes.json", hashes)
    write_json(ROOT / "metrics.json", {"window": [START, END], "signal_count": len(signal_dates), "metrics": metrics, "comparisons": comparisons, "overlap_month_counts": overlap_hist})

    lines = [
        "# V30 — V26 factor Top2 + 60d momentum Top2, fixed 50/50",
        "",
        f"Strict holdout: {START} to {END}; {len(signal_dates)} shared monthly signals and {int(lifecycle['count'].iloc[0])} trading days per strategy.",
        "",
        "The frozen protocol assigns 50% of total equity to V26's `mf_tier_flow_agreement_20` Top2 and 50% to 60d adjusted-price momentum Top2. Each sleeve assigns 25% to each selected bank. Overlapping names receive the sum of their sleeve weights. The rule was fixed before this backtest; weights were not adjusted from the result.",
        "",
        "| Strategy | Net return | Gross same fills | Max drawdown | Ann. volatility | Fees | Slippage | Trades | Min cash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in metrics.items():
        lines.append(f"| {name} | {result['net_return']:.2%} | {result['gross_return_same_fills']:.2%} | {result['max_drawdown']:.2%} | {result['annualized_volatility']:.2%} | CNY {result['fees']:,.2f} | CNY {result['slippage_cost']:,.2f} | {result['trades']} | CNY {result['min_cash']:,.2f} |")
    lines += ["", "## V30 minus each baseline", ""]
    for name, diff in comparisons.items():
        lines.append(f"- {name}: net return {diff['net_return_difference_pp']:+.2f} pp; max drawdown {diff['max_drawdown_difference_pp']:+.2f} pp.")
    lines += [
        "",
        "## Overlap and accounting audit",
        "",
        f"- Number of months by shared names: {json.dumps(overlap_hist, ensure_ascii=False)} (keys are 0, 1, or 2 overlapping stocks).",
        f"- Monthly target weights summed to 1.0 in every month; maximum single-name target weight was {audit['monthly_mix_checks']['max_per_name_target_weight']:.0%}.",
        "- Every strategy uses the V11 final ledger on the same dates. Cash stayed nonnegative; there were zero blocked trade events.",
        f"- Maximum accounting residual: {max(x['max_abs_accounting_residual'] for x in metrics.values()):.3g}.",
        "- V26, 60d, V11 and equal-weight metrics and daily holdings reproduce the V26 cache; trades match exactly by identity and within 1e-5 CNY/share units on saved numeric fields.",
        "- The end is marked at 2026-06-30 close without forced liquidation; quarter and year tables use this same lifecycle.",
        f"- SHA-256: run.py `{hashes['run.py']}`; frozen protocol.json `{hashes['protocol.json_frozen_before_results']}`; cash_engine.py `{hashes['cash_engine.py']}`.",
        "",
        "See monthly_target_weights.csv for each component pair, overlap and total weight, plus daily_equity.csv, trades.csv, quarterly_returns.csv, annual_returns.csv, and audit.json. This historical four-bank backtest does not establish stable future alpha.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nSHA256\n" + json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
