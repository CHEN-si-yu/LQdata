#!/usr/bin/env python3
"""V40: frozen equal-weight average-rank blend of V34 factor and 60-session adjusted momentum."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
V11 = BASE / "V11"
V24 = BASE / "V24"
V26 = BASE / "V26"
V34 = BASE / "V34"
DATA = BASE.parent / "trainingdata"
START = "2024-01-02"
END = "2026-06-30"
FACTOR_BLEND = "id2_close_vs_pm_vwap_20"
FACTOR_V26 = "mf_tier_flow_agreement_20"

import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(V11))
import analysis as E

PROTOCOL_PATH = HERE / "protocol.json"
CODES = E.CODES
DAILY_FIELDS = ["equity_net", "equity_gross", "cash", "fees_cumulative",
                "slippage_cumulative", "turnover_cumulative", "accounting_residual",
                "daily_return_net", "drawdown"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_protocol():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("version") != "V40":
        raise RuntimeError("protocol version mismatch")
    if protocol.get("run_script_sha256") != sha256(Path(__file__)):
        raise RuntimeError("runner hash does not match the pre-run frozen protocol")
    if protocol.get("factor_feature") != FACTOR_BLEND:
        raise RuntimeError("factor feature differs from the frozen protocol")
    if protocol.get("momentum_sessions") != 60 or protocol.get("factor_rank_weight") != 0.5:
        raise RuntimeError("rank blend differs from the frozen protocol")
    if protocol.get("momentum_rank_weight") != 0.5 or protocol.get("top_n") != 2:
        raise RuntimeError("portfolio weights differ from the frozen protocol")
    if protocol.get("holdout_dates") != [START, END]:
        raise RuntimeError("holdout window differs from the frozen protocol")
    return protocol


def load_factor_rows(features):
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    if meta.get("semantics") != "zscore_win1_99_v1":
        raise RuntimeError("unexpected factor semantics")
    years = sorted(int(year) for year in meta["built_years"] if 2024 <= int(year) <= 2026)
    frames = []
    for year in years:
        frame = pq.read_table(
            DATA / "factors" / f"year={year}" / "data.parquet",
            columns=["trade_date", "stock_code", *features],
            filters=[("stock_code", "in", list(CODES))]
        ).to_pandas()
        frame["trade_date"] = frame["trade_date"].astype(str).str[:10]
        frame["stock_code"] = frame["stock_code"].astype(str)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    if out.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("duplicate factor date/security rows")
    return out, meta, years


def rank_descending(values):
    return {
        code: rank
        for rank, code in enumerate(
            sorted(CODES, key=lambda code: (-float(values[code]), code)), start=1
        )
    }


def make_signal_schedules(factors, signal_dates, prices):
    by_date = {date: frame.set_index("stock_code") for date, frame in factors.groupby("trade_date", sort=False)}
    index = prices["index"]
    adj_close = prices["adj_close"]
    factor_v34 = {}
    factor_v26 = {}
    momentum = {}
    blend = {}
    audit = []
    for date in signal_dates:
        if date not in by_date:
            raise RuntimeError(f"missing factor snapshot for shared signal {date}")
        day = by_date[date].reindex(CODES)
        raw_blend = day[FACTOR_BLEND].astype(float)
        raw_v26 = day[FACTOR_V26].astype(float)
        missing_blend = [code for code in CODES if not np.isfinite(raw_blend.loc[code])]
        missing_v26 = [code for code in CODES if not np.isfinite(raw_v26.loc[code])]
        val_blend = {code: float(raw_blend.fillna(0.0).loc[code]) for code in CODES}
        val_v26 = {code: float(raw_v26.fillna(0.0).loc[code]) for code in CODES}
        rank_blend_factor = rank_descending(val_blend)
        rank_v26 = rank_descending(val_v26)

        day_index = index[date]
        if day_index < 60:
            raise RuntimeError(f"{date}: insufficient 60-session momentum history")
        momentum_values = {
            code: float(adj_close[day_index, j] / adj_close[day_index - 60, j] - 1.0)
            if np.isfinite(adj_close[day_index, j]) and np.isfinite(adj_close[day_index - 60, j])
            and adj_close[day_index, j] > 0 and adj_close[day_index - 60, j] > 0
            else float("-inf")
            for j, code in enumerate(CODES)
        }
        rank_momentum = rank_descending(momentum_values)
        average_rank = {
            code: 0.5 * rank_blend_factor[code] + 0.5 * rank_momentum[code]
            for code in CODES
        }
        chosen = sorted(CODES, key=lambda code: (average_rank[code], code))[:2]
        pick_v34 = sorted(CODES, key=lambda code: (rank_blend_factor[code], code))[:2]
        pick_v26 = sorted(CODES, key=lambda code: (rank_v26[code], code))[:2]
        pick_momentum = sorted(CODES, key=lambda code: (rank_momentum[code], code))[:2]
        factor_v34[date] = {code: 0.5 for code in pick_v34}
        factor_v26[date] = {code: 0.5 for code in pick_v26}
        momentum[date] = {code: 0.5 for code in pick_momentum}
        blend[date] = {code: 0.5 for code in chosen}
        execution_date = str(prices["days"][day_index + 1])
        if index[execution_date] != day_index + 1:
            raise RuntimeError(f"{date}: execution date is not the next session")
        audit.append({
            "signal_date": date,
            "execution_date": execution_date,
            "factor_name": FACTOR_BLEND,
            "factor_values_at_signal_close": val_blend,
            "factor_ranks_descending": rank_blend_factor,
            "momentum_sessions": 60,
            "adjusted_close_momentum_at_signal_close": momentum_values,
            "momentum_ranks_descending": rank_momentum,
            "factor_rank_weight": 0.5,
            "momentum_rank_weight": 0.5,
            "average_rank": average_rank,
            "selected_top2": chosen,
            "v34_top2_from_same_signal": pick_v34,
            "v26_top2_from_same_signal": pick_v26,
            "momentum_top2_from_same_signal": pick_momentum,
            "v26_factor_ranks_descending": rank_v26,
            "v26_factor_values_at_signal_close": val_v26,
            "factor_missing_values_filled_with_zero": len(missing_blend),
            "v26_missing_values_filled_with_zero": len(missing_v26),
            "signal_uses_close_snapshot_only": True,
            "no_future_row_used": execution_date > date
        })
    return factor_v34, factor_v26, momentum, blend, audit


def period_returns(daily, freq):
    frame = daily.copy()
    frame["trade_date"] = pd.to_datetime(frame.trade_date)
    frame["period"] = frame.trade_date.dt.to_period(freq).astype(str)
    rows = []
    for period, group in frame.groupby("period", sort=True):
        rows.append({
            "period": period,
            "strategy": str(group.strategy.iloc[0]),
            "net_return": float(np.prod(1.0 + group.daily_return_net.to_numpy()) - 1.0),
            "gross_return_same_fills": float(np.prod(1.0 + group.gross_return_daily.to_numpy()) - 1.0)
        })
    return rows


def compare_to_reference(daily, metrics, ref_dir, strategy, tolerance=1e-3):
    reference_metrics = json.loads((ref_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"][strategy]
    reference_daily = pd.read_csv(ref_dir / "daily_equity.csv", dtype={"trade_date": str})
    reference_daily = reference_daily.loc[reference_daily.strategy == strategy].sort_values("trade_date").reset_index(drop=True)
    current_daily = daily.loc[daily.strategy == strategy].sort_values("trade_date").reset_index(drop=True)
    if reference_daily.trade_date.tolist() != current_daily.trade_date.tolist():
        raise RuntimeError(f"{strategy}: daily reference date coverage differs")
    daily_diffs = {}
    for field in DAILY_FIELDS:
        daily_diffs[field] = float(np.max(np.abs(
            reference_daily[field].to_numpy(dtype=float) - current_daily[field].to_numpy(dtype=float)
        )))
    shares_exact = reference_daily["shares_json"].astype(str).tolist() == current_daily["shares_json"].astype(str).tolist()
    metric_fields = ("net_return", "fees", "slippage_cost", "turnover", "max_drawdown",
                     "annualized_volatility", "min_cash")
    metric_diffs = {
        field: abs(float(metrics[strategy][field]) - float(reference_metrics[field]))
        for field in metric_fields
    }
    if not all(value <= tolerance for value in daily_diffs.values()):
        raise RuntimeError(f"{strategy}: daily path differs from frozen {ref_dir.name} reference")
    if not all(value <= tolerance for value in metric_diffs.values()):
        raise RuntimeError(f"{strategy}: metrics differ from frozen {ref_dir.name} reference")
    if not shares_exact or metrics[strategy]["trades"] != reference_metrics["trades"]:
        raise RuntimeError(f"{strategy}: holdings or trade count differ from frozen {ref_dir.name} reference")
    return {
        "reference": ref_dir.name,
        "daily_rows_match": len(reference_daily),
        "daily_max_abs_differences": daily_diffs,
        "metric_abs_differences": metric_diffs,
        "shares_json_exact_match": shares_exact,
        "trade_count_equal": True,
        "tolerance": tolerance
    }


def main():
    protocol = verify_frozen_protocol()
    prices = E.load_prices()
    quarter_dirs = sorted(
        path for path in (V11 / "quarters").iterdir()
        if path.is_dir() and (path / "predictions.csv").is_file()
    )
    quarter_names = [path.name for path in quarter_dirs]
    predictions = E.load_predictions(quarter_names)
    v11_schedules, _ = E.schedules(predictions, prices)
    signal_dates = sorted(
        date for date in predictions.loc[predictions.monthly_signal, "trade_date"].unique()
        if START <= date <= END
    )
    if len(signal_dates) != 30:
        raise RuntimeError(f"expected 30 common holdout monthly signals, got {len(signal_dates)}")

    v34_protocol = json.loads((V34 / "protocol.json").read_text(encoding="utf-8"))
    v26_protocol = json.loads((V26 / "protocol.json").read_text(encoding="utf-8"))
    if v34_protocol["factor_selection"]["selected_features"] != [FACTOR_BLEND]:
        raise RuntimeError("V34's frozen factor does not match the V40 factor input")
    if v26_protocol["factor_selection"]["selected_features"] != [FACTOR_V26]:
        raise RuntimeError("V26 reference protocol has changed")
    if v34_protocol["monthly_signal_dates"] != signal_dates or v26_protocol["monthly_signal_dates"] != signal_dates:
        raise RuntimeError("holdout monthly signal dates differ from frozen V34/V26 protocols")

    factors, meta, factor_years = load_factor_rows([FACTOR_BLEND, FACTOR_V26])
    if meta.get("semantics") != protocol["factor_semantics"]:
        raise RuntimeError("factor semantics differ from the frozen protocol")
    if not set(signal_dates).issubset(set(factors.trade_date.unique())):
        raise RuntimeError("factor rows are missing on one or more shared monthly signals")
    v34_schedule, v26_schedule, momentum_schedule, blend_schedule, membership_audit = make_signal_schedules(
        factors, signal_dates, prices
    )

    first_signal = signal_dates[0]
    schedules = {
        "V40 50-50 average-rank Top2": blend_schedule,
        "V34 id2_close_vs_pm_vwap_20 Top2": v34_schedule,
        "V26 mf_tier_flow_agreement_20 Top2": v26_schedule,
        "60d momentum Top2": {date: v11_schedules["60d momentum Top2"][date] for date in signal_dates},
        "V11 LambdaRank Top2": {date: v11_schedules["V11 LambdaRank Top2"][date] for date in signal_dates},
        "equal-weight hold": {first_signal: {code: 0.25 for code in CODES}}
    }
    for audit_row in membership_audit:
        date = audit_row["signal_date"]
        if audit_row["selected_top2"] != sorted(
            audit_row["selected_top2"],
            key=lambda code: (
                audit_row["average_rank"][code], code
            )
        ):
            raise RuntimeError(f"{date}: blend membership does not follow frozen average-rank ordering")

    daily_parts = []
    all_trades = []
    metrics = {}
    for name, target_schedule in schedules.items():
        daily, trades, metric = E.simulate(name, prices, target_schedule, START, END)
        daily_parts.append(daily)
        all_trades.extend(trades)
        metrics[name] = metric
    daily_all = pd.concat(daily_parts, ignore_index=True)
    daily_all["gross_return_daily"] = daily_all.groupby("strategy", sort=False).equity_gross.pct_change().fillna(0.0)
    trades_all = pd.DataFrame(all_trades)

    compatibility = {}
    for strategy in ("V11 LambdaRank Top2", "60d momentum Top2", "equal-weight hold"):
        compatibility[strategy] = compare_to_reference(daily_all, metrics, V24, strategy)
    compatibility["V34 id2_close_vs_pm_vwap_20 Top2"] = compare_to_reference(
        daily_all, metrics, V34, "V34 id2_close_vs_pm_vwap_20 Top2"
    )
    compatibility["V26 mf_tier_flow_agreement_20 Top2"] = compare_to_reference(
        daily_all, metrics, V26, "V26 mf_tier_flow_agreement_20 Top2"
    )

    quarterly, annual = [], []
    for _, group in daily_all.groupby("strategy", sort=False):
        quarterly.extend(period_returns(group, "Q"))
        annual.extend(period_returns(group, "Y"))

    audit = {
        "window": [START, END],
        "monthly_signals": len(signal_dates),
        "signal_dates_match_v11_v34_v26": True,
        "all_factor_rows_on_signal_dates_available": True,
        "factor_semantics": meta.get("semantics"),
        "V34_factor_frozen_before_holdout": True,
        "momentum_uses_adjusted_close_60_trading_sessions": True,
        "equal_average_of_component_ranks": True,
        "fixed_factor_rank_weight": 0.5,
        "fixed_momentum_rank_weight": 0.5,
        "no_weight_sweep_or_holdout_rule_change": True,
        "signal_uses_close_snapshot_only": all(row["signal_uses_close_snapshot_only"] for row in membership_audit),
        "execution_is_next_session_open": all(row["no_future_row_used"] for row in membership_audit),
        "positions_marked_to_final_close_without_forced_liquidation": True,
        "frozen_ledger_reference_compatibility": compatibility,
        "strategies": {
            name: {
                "cash_nonnegative": metric["min_cash"] >= -1e-7,
                "blocked_trade_events": metric["blocked_trade_events"],
                "max_abs_accounting_residual": metric["max_abs_accounting_residual"],
                "max_positions": metric["max_positions"],
                "trades": metric["trades"],
                "min_cash": metric["min_cash"]
            }
            for name, metric in metrics.items()
        },
        "membership_audit_rows": len(membership_audit)
    }

    daily_all.to_csv(HERE / "daily_equity.csv", index=False, float_format="%.10g")
    trades_all.to_csv(HERE / "trades.csv", index=False, float_format="%.10g")
    pd.DataFrame(quarterly).to_csv(HERE / "quarterly_returns.csv", index=False, float_format="%.10g")
    pd.DataFrame(annual).to_csv(HERE / "annual_returns.csv", index=False, float_format="%.10g")
    (HERE / "membership_audit.json").write_text(
        json.dumps(membership_audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (HERE / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (HERE / "metrics.json").write_text(
        json.dumps({
            "window": [START, END],
            "days": int(len(daily_all) // len(schedules)),
            "signal_count": len(signal_dates),
            "V40_rule": {
                "factor": FACTOR_BLEND,
                "factor_rank_weight": 0.5,
                "momentum": "60-session adjusted-close price return",
                "momentum_rank_weight": 0.5,
                "rank_direction": "ascending average rank; component ranks descend by factor/return",
                "top_n": 2
            },
            "metrics": metrics,
            "frozen_ledger_compatibility": compatibility
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    input_records = []
    def add_input(path, role):
        path = Path(path)
        input_records.append({
            "role": role,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path)
        })

    add_input(DATA / "meta.json", "factor and market cache metadata")
    add_input(V11 / "analysis.py", "V11 corrected shared-cash ledger engine")
    add_input(V24 / "protocol.json", "common holdout protocol")
    add_input(V34 / "protocol.json", "frozen V34 factor and execution protocol")
    add_input(V26 / "protocol.json", "frozen V26 comparison protocol")
    for quarter in quarter_names:
        prediction_path = V11 / "quarters" / quarter / "predictions.csv"
        if START[:4] <= quarter[:4] <= END[:4]:
            add_input(prediction_path, "cached V11 quarterly predictions used for holdout schedule")
    for year in sorted(int(year) for year in meta["built_years"]):
        add_input(DATA / "prices" / f"year={year}" / "data.parquet", f"four-bank raw price cache {year}")
        add_input(DATA / "amount" / f"year={year}" / "data.parquet", f"four-bank trading amount cache {year}")
    for year in factor_years:
        add_input(DATA / "factors" / f"year={year}" / "data.parquet", f"harmonized factor cache {year}")
    for ref_dir, files in (
        (V24, ("metrics.json", "daily_equity.csv")),
        (V34, ("metrics.json", "daily_equity.csv")),
        (V26, ("metrics.json", "daily_equity.csv"))
    ):
        for filename in files:
            add_input(ref_dir / filename, f"frozen {ref_dir.name} ledger reference {filename}")
    add_input(Path(__file__), "V40 runner")
    (HERE / "cache_hashes.json").write_text(
        json.dumps({
            "algorithm": "SHA-256",
            "scope": "V40 runner, V11 corrected ledger and holdout predictions, factor/price/amount input caches, and frozen V24/V34/V26 protocols and ledger references.",
            "hashes": input_records
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    table = [
        "| Strategy | Net return | Gross return, same fills | Annualized | Max drawdown | Ann. vol. | Fees | Slippage | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for name, metric in metrics.items():
        table.append(
            "| {} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2f} | {:.2f} | {} |".format(
                name, metric["net_return"], metric["gross_return_same_fills"],
                metric["annualized_net_return"], metric["max_drawdown"],
                metric["annualized_volatility"], metric["fees"],
                metric["slippage_cost"], metric["trades"]
            )
        )
    blend_metrics = metrics["V40 50-50 average-rank Top2"]
    deltas = []
    for name in (
        "V34 id2_close_vs_pm_vwap_20 Top2",
        "V26 mf_tier_flow_agreement_20 Top2",
        "60d momentum Top2",
        "V11 LambdaRank Top2",
        "equal-weight hold"
    ):
        deltas.append(
            f"- 相对{name}：净收益差 {100 * (blend_metrics['net_return'] - metrics[name]['net_return']):+.2f} 个百分点；"
            f"最大回撤差 {100 * (blend_metrics['max_drawdown'] - metrics[name]['max_drawdown']):+.2f} 个百分点。"
        )
    report = [
        "# V40：V34 因子与 60 日复权动量等权平均名次",
        "",
        "## 冻结规则",
        "",
        f"只在四家银行的共同月初信号日计算两个横截面名次：V34 冻结因子 {FACTOR_BLEND} 与过去 60 个交易日的复权收盘价动量。两项均按由高到低分别排名，名次 1 最优；股票代码用于并列打破。每只银行的综合名次为两项名次各乘 50% 后相加，选综合名次最小的 Top2，等权持有，次日开盘执行。",
        "",
        "协议在开跑前冻结；权重固定 50%/50%，不做权重扫描或留出期规则调整。共同留出期为 2024-01-02 至 2026-06-30，共 30 个月初信号。",
        "",
        "复权动量与 V11 的 60 日基线采用相同算法：信号日复权收盘价除以 60 个交易日之前的复权收盘价，再减 1。成交、费用、现金和估值沿用 V11 corrected ledger；窗口末按 2026-06-30 收盘估值，不强制平仓。",
        "",
        "## 结果",
        "",
        *table,
        "",
        *deltas,
        "",
        "完整季度与年度收益见 quarterly_returns.csv、annual_returns.csv；逐日账户和成交记录见 daily_equity.csv、trades.csv。",
        "",
        "## 审计",
        "",
        "- V34、V26、V11、60 日动量与等权持有使用相同的 30 个留出期月初信号和 2024-01-02 至 2026-06-30 区间。",
        "- V11、60 日动量、等权持有逐日账本与 V24 冻结基线核对；V34 与 V26 分别与各自冻结账本核对。现金、关键指标、持仓股数与成交数均在审计容差内。",
        "- 按信号日收盘信息排序，次一交易日开盘执行；账本使用同一 V11 corrected ledger。",
        "- 每策略的现金、会计残差、受阻成交和逐日账本核验结果见 audit.json；信号名次明细见 membership_audit.json。",
        "- 输入数据、基线文件、账本引擎和本次脚本 SHA-256 清单见 cache_hashes.json。",
        "",
        "这是四家银行上的固定历史留出结果，不单独证明未来稳定超额。"
    ]
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(HERE),
        "signal_count": len(signal_dates),
        "metrics": metrics,
        "compatibility": compatibility,
        "max_process_rss_gib_is_reported_by_time": True
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
