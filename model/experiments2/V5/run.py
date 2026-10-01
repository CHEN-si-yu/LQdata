#!/usr/bin/env python3
"""experiments2/V5: monthly Top2 using cached V1 score ensemble, shared-cash 50/50."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "V1"
DATA = Path("/root/autodl-fs/model/trainingdata")
sys.path.insert(0, str(V1))
import model as M
import analysis as A

SPEC = dict(M.RECIPE["action"])
START, END = M.RECIPE["test_start"], M.RECIPE["test_end"]
CODES = tuple(M.CODES)
SEEDS = tuple(M.SEEDS)
QUARTERS = tuple(M.QUARTERS)
OUT_FILES = (
    "run.py", "README.md", "results.json", "quarterly.csv", "seed_results.csv",
    "daily.csv", "trades.csv", "audit.json", "provenance.json",
)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(path, index=False)

def layout_audit(require_all=False):
    expected = set(OUT_FILES)
    actual = {p.name for p in ROOT.iterdir() if p.is_file() and p.name not in ("__pycache__",)}
    actual.discard("layout.json")
    return {
        "expected_files": sorted(expected),
        "present_files": sorted(expected & actual),
        "missing_files": sorted(expected - actual),
        "extra_files": sorted(actual - expected),
        "passed": (not (actual - expected)) and (not require_all or expected <= actual),
    }

def load_predictions():
    per_seed = {seed: {} for seed in SEEDS}
    cache_files, cache_hashes = [], {}
    data_built_at = M.metadata().get("built_at")
    for q in QUARTERS:
        fold_predictions = []
        quarter_dates = None
        for fold in M.FOLDS:
            f = V1 / "model_train" / q / f"fold{fold}" / "best.pt"
            if not f.is_file():
                raise FileNotFoundError(f)
            with open(f, "rb") as h:
                payload = pickle.load(h)
            if payload.get("data_built_at") != data_built_at:
                raise RuntimeError(f"V1 prediction cache snapshot mismatch: {f}")
            dates = [str(x) for x in payload["score_dates"]]
            qdates = [str(x) for x in payload["test_dates"]]
            if quarter_dates is None:
                quarter_dates = qdates
            elif qdates != quarter_dates:
                raise RuntimeError(f"{q} folds disagree on test dates")
            pos = {d: i for i, d in enumerate(dates)}
            try:
                ix = [pos[d] for d in qdates]
            except KeyError as e:
                raise RuntimeError(f"{q} test date missing from cached prediction: {e}")
            arr = np.asarray(payload["seed_predictions"], dtype=np.float32)
            if arr.shape[0] != len(SEEDS) or arr.shape[2] != len(CODES):
                raise RuntimeError(f"unexpected seed prediction shape {arr.shape} in {f}")
            fold_predictions.append(arr[:, ix, :])
            cache_files.append(str(f))
            cache_hashes[str(f)] = sha256(f)
        avg = np.mean(np.stack(fold_predictions, axis=0), axis=0, dtype=np.float64).astype(np.float32)
        for si, seed in enumerate(SEEDS):
            for di, day in enumerate(quarter_dates):
                per_seed[seed][day] = avg[si, di].astype(np.float64)
    days = sorted(per_seed[SEEDS[0]])
    if any(set(per_seed[s]) != set(days) for s in SEEDS):
        raise RuntimeError("per-seed cache date axis mismatch")
    scores = {s: np.stack([per_seed[s][d] for d in days]) for s in SEEDS}
    ensemble = np.mean(np.stack([scores[s] for s in SEEDS]), axis=0)
    published = []
    for y in sorted(set(d[:4] for d in days)):
        p = V1 / "model_pred" / "ensemble" / f"year={y}" / "data.parquet"
        df = pq.read_table(p, columns=["trade_date", "stock_code", "score"]).to_pandas()
        df["trade_date"] = df.trade_date.astype(str)
        df["stock_code"] = df.stock_code.astype(str)
        df = df[df.trade_date.isin(days)]
        published.append(df)
    pub = pd.concat(published, ignore_index=True).pivot(index="trade_date", columns="stock_code", values="score")
    pub = pub.reindex(index=days, columns=CODES).to_numpy(dtype=np.float64)
    delta = np.abs(pub - ensemble)
    max_delta = float(np.nanmax(delta))
    if not np.isfinite(max_delta) or max_delta > 2e-6:
        raise RuntimeError(f"reconstructed V1 ensemble differs from published V1 scores: max_abs={max_delta}")
    return days, scores, ensemble, {
        "prediction_cache_files": cache_files,
        "prediction_cache_sha256": cache_hashes,
        "v1_published_score_max_abs_delta": max_delta,
    }

def read_small_panel(signal_days):
    pieces = []
    amount_col = None
    for year in (2025, 2026):
        p = DATA / "prices" / f"year={year}" / "data.parquet"
        a = DATA / "amount" / f"year={year}" / "data.parquet"
        ac = pq.ParquetFile(a).schema_arrow.names
        amount_col = next(x for x in ac if x not in ("trade_date", "stock_code"))
        prices = pq.read_table(p, columns=["trade_date", "stock_code", *M.PRICE_COLUMNS],
                               filters=[("stock_code", "in", list(CODES))]).to_pandas()
        amounts = pq.read_table(a, columns=["trade_date", "stock_code", amount_col],
                                filters=[("stock_code", "in", list(CODES))]).to_pandas()
        prices["trade_date"] = prices.trade_date.astype(str)
        amounts["trade_date"] = amounts.trade_date.astype(str)
        wanted_start = signal_days[0]
        prices = prices[(prices.trade_date >= wanted_start) & (prices.trade_date <= "2026-07-31")]
        amounts = amounts[(amounts.trade_date >= wanted_start) & (amounts.trade_date <= signal_days[-1])]
        prices = prices.merge(amounts, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
        pieces.append(prices)
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame[(frame.trade_date >= signal_days[0]) & (frame.trade_date <= "2026-07-31")]
    days = np.asarray(sorted(frame.trade_date.unique()), dtype=str)
    if not set(signal_days).issubset(set(days)):
        raise RuntimeError("price calendar does not include all official signal dates")
    if not days[-1] > signal_days[-1]:
        raise RuntimeError("missing T+1 execution date after official window")
    grid = pd.MultiIndex.from_product([days, CODES], names=["trade_date", "stock_code"])
    frame = frame.drop_duplicates(["trade_date", "stock_code"]).set_index(["trade_date", "stock_code"]).reindex(grid).reset_index()
    panel = type("Panel", (), {})()
    panel.days, panel.codes = days, np.asarray(CODES, dtype=str)
    panel.amount = np.nan_to_num(frame[amount_col].to_numpy(np.float64).reshape(len(days), len(CODES)),
                                 nan=0.0, posinf=0.0, neginf=0.0)
    panel.prices = M.Prices(frame, days, panel.codes)
    panel.meta = M.metadata()
    return panel, amount_col

def monthly_choices(days, scores):
    dates = np.asarray(days, dtype=str)
    picks, rebalance_days = {}, []
    for i, day in enumerate(dates):
        if i == 0 or day[:7] != dates[i - 1][:7]:
            ranking = sorted(range(len(CODES)), key=lambda c: (-float(scores[i, c]), str(CODES[c])))
            picks[day] = tuple(int(c) for c in ranking[:2])
            rebalance_days.append(day)
    return picks, rebalance_days

def run_account(scores, signal_days, panel, *, label):
    """Shared cash account; fixed 50/50 target on monthly rebalance, V1 execution costs."""
    signal_ix = np.asarray([int(np.searchsorted(panel.days, d)) for d in signal_days], dtype=np.int64)
    if not np.array_equal(panel.days[signal_ix], np.asarray(signal_days, dtype=str)):
        raise RuntimeError("signal dates do not align to reduced price panel")
    picks, rebalance_days = monthly_choices(signal_days, scores)
    pick_by_signal = {}
    active = ()
    for day in signal_days:
        if day in picks:
            active = picks[day]
        pick_by_signal[day] = active
    nc = len(CODES); lot = int(SPEC["lot_size"]); slip = float(SPEC["slippage_rate"])
    cash = float(M.RECIPE["account_money"]); shares = np.zeros(nc, dtype=np.float64)
    prev_exec = None; prev_equity = cash
    curve, trades = [], []
    total_fees = 0.0; blocked_entries = blocked_exits = corp_actions = 0
    min_cash = cash; max_exposure = 0.0; max_residual = 0.0
    matched_values = [cash]; strategy_values = [cash]
    last_marks = np.full(nc, np.nan)
    for pos, sd0 in enumerate(signal_ix):
        sd = int(sd0); ed = sd + 1
        if ed >= len(panel.days):
            raise RuntimeError(f"{panel.days[sd]} lacks T+1 price row")
        if prev_exec is not None:
            for c in range(nc):
                a0, a1 = float(panel.prices.adj[prev_exec, c]), float(panel.prices.adj[ed, c])
                if np.isfinite(a0) and np.isfinite(a1) and a0 > 0 and a1 > 0:
                    ratio = a1 / a0
                    if abs(ratio - 1.0) > 1e-8 and shares[c] > 0:
                        shares[c] *= ratio; corp_actions += 1
        marks = np.asarray([A._mark_price(panel.prices, ed, c) for c in range(nc)], dtype=float)
        for c in range(nc):
            if np.isfinite(marks[c]): last_marks[c] = marks[c]
            elif np.isfinite(last_marks[c]): marks[c] = last_marks[c]
        pretrade_equity = float(cash + np.sum(shares * np.nan_to_num(marks, nan=0.0)))
        matched_day = pretrade_equity / prev_equity - 1.0 if prev_equity > 0 else 0.0
        final_day = pos == len(signal_ix) - 1
        signal_day = str(panel.days[sd])
        do_rebalance = signal_day in picks
        selected = pick_by_signal[signal_day]
        target_qty = shares.copy() if not do_rebalance else np.zeros(nc, dtype=np.float64)
        if do_rebalance and not final_day:
            for c in selected:
                target_qty[c] = math.floor(max(pretrade_equity, 0.0) * 0.5 /
                                           marks[c] / lot) * lot if np.isfinite(marks[c]) and marks[c] > 0 else 0.0
        if final_day:
            target_qty[:] = 0.0
        for c, code in enumerate(CODES):
            delta = target_qty[c] - shares[c]
            if delta >= -1e-7:
                continue
            mark = marks[c]
            if not np.isfinite(mark) or mark <= 0 or not bool(panel.prices.exit[ed, c]):
                blocked_exits += 1
                continue
            cap = A._capacity_qty(panel, sd, c, mark, SPEC)
            req = float(shares[c]) if target_qty[c] <= 1e-7 else float(math.floor(abs(delta) / lot) * lot)
            qty = min(req, float((int(cap) // lot) * lot))
            if target_qty[c] <= 1e-7 and qty >= shares[c] - 1e-7:
                qty = float(shares[c])
            else:
                qty = float(math.floor(qty / lot) * lot)
            if qty <= 0:
                continue
            px = mark * (1.0 - slip); notional = qty * px
            fee = A._fees(notional, "sell", SPEC)
            cash += notional - fee; shares[c] -= qty
            if shares[c] < 1e-8: shares[c] = 0.0
            total_fees += fee
            trades.append({"signal_date": signal_day, "date": str(panel.days[ed]),
                           "code": str(code), "side": "sell", "quantity": qty,
                           "price": px, "notional": notional, "fees": fee})
        if do_rebalance and not final_day:
            for c in selected:
                delta = target_qty[c] - shares[c]
                if delta <= 1e-7:
                    continue
                mark = marks[c]
                if not np.isfinite(mark) or mark <= 0 or not bool(panel.prices.entry[ed, c]):
                    blocked_entries += 1
                    continue
                cap = A._capacity_qty(panel, sd, c, mark, SPEC)
                target_request = int(math.floor(delta / lot) * lot)
                qty = min(target_request, int(cap)); qty = (qty // lot) * lot
                while qty > 0:
                    px = mark * (1.0 + slip); notional = qty * px
                    fee = A._fees(notional, "buy", SPEC)
                    if notional + fee <= cash + 1e-8: break
                    qty -= lot
                if qty <= 0: continue
                cash -= notional + fee; shares[c] += qty; total_fees += fee
                if qty < target_request and cap >= target_request:
                    target_qty[c] = shares[c]
                trades.append({"signal_date": signal_day, "date": str(panel.days[ed]),
                               "code": str(CODES[c]), "side": "buy", "quantity": qty,
                               "price": px, "notional": notional, "fees": fee})
        equity_by_code = shares * np.nan_to_num(marks, nan=0.0)
        equity = float(cash + equity_by_code.sum())
        exposure = float(equity_by_code.sum() / equity) if equity > 0 else 0.0
        daily_return = equity / prev_equity - 1.0 if prev_equity > 0 else 0.0
        residual = abs(equity - cash - float(np.sum(shares * np.nan_to_num(marks, nan=0.0))))
        max_residual = max(max_residual, residual)
        min_cash = min(min_cash, cash); max_exposure = max(max_exposure, exposure)
        curve.append({"signal_date": signal_day, "date": str(panel.days[ed]),
                      "equity": equity, "cash": cash, "exposure": exposure,
                      "daily_return": daily_return, "matched_daily_return": matched_day,
                      "shares": shares.tolist(), "ticker_equity": equity_by_code.tolist(),
                      "selected_codes": [str(CODES[c]) for c in selected]})
        strategy_values.append(equity); matched_values.append(matched_values[-1] * (1.0 + matched_day))
        prev_equity, prev_exec = equity, ed
    values = np.asarray(strategy_values, dtype=float)
    dd = values / np.maximum.accumulate(values) - 1.0
    daily_net = np.asarray([r["daily_return"] for r in curve], dtype=float)
    daily_match = np.asarray([r["matched_daily_return"] for r in curve], dtype=float)
    stats = {
        "name": label, "start": str(signal_days[0]), "end": str(signal_days[-1]),
        "execution_end": str(curve[-1]["date"]), "signal_days": len(signal_days),
        "rebalance_months": len(rebalance_days), "rebalance_dates": rebalance_days,
        "return_value": float(values[-1] / values[0] - 1.0),
        "strict_previous_day_matched_return": float(np.prod(1.0 + daily_match) - 1.0),
        "strict_matched_alpha": float((values[-1] / values[0] - 1.0) - (np.prod(1.0 + daily_match) - 1.0)),
        "max_drawdown": float(dd.min()), "total_fees": float(total_fees),
        "trades": len(trades), "avg_exposure": float(np.mean([x["exposure"] for x in curve])),
        "ending_equity": float(values[-1]), "ending_cash": float(cash),
        "ending_positions": int(np.count_nonzero(shares > 1e-8)),
        "min_cash": float(min_cash), "max_exposure": float(max_exposure),
        "blocked_entries": int(blocked_entries), "blocked_exits": int(blocked_exits),
        "corporate_action_adjustments": int(corp_actions), "max_ledger_residual": float(max_residual),
        "cash_nonnegative": bool(min_cash >= -1e-6), "exposure_at_most_one": bool(max_exposure <= 1.0 + 1e-6),
        "final_flat": bool(np.count_nonzero(shares > 1e-8) == 0),
    }
    qrows = []
    for q in QUARTERS:
        mask = np.array([str(x["signal_date"])[:4] + "Q" + str((int(str(x["signal_date"])[5:7])-1)//3+1) == q for x in curve])
        if not mask.any(): continue
        qnet = float(np.prod(1.0 + daily_net[mask]) - 1.0)
        qmatch = float(np.prod(1.0 + daily_match[mask]) - 1.0)
        qrows.append({"quarter": q, "days": int(mask.sum()), "return_value": qnet,
                      "strict_previous_day_matched_return": qmatch, "strict_matched_alpha": qnet-qmatch,
                      "fees": float(sum(t["fees"] for t in trades if str(t["signal_date"])[:4] + "Q" + str((int(str(t["signal_date"])[5:7])-1)//3+1) == q)),
                      "trades": int(sum(1 for t in trades if str(t["signal_date"])[:4] + "Q" + str((int(str(t["signal_date"])[5:7])-1)//3+1) == q)),
                      "avg_exposure": float(np.mean([curve[i]["exposure"] for i in np.flatnonzero(mask)]))})
    return stats, qrows, curve, trades

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout-only", action="store_true")
    args = ap.parse_args()
    if args.layout_only:
        print(json.dumps(layout_audit(False), ensure_ascii=False, indent=2))
        return
    signal_days, seed_scores, ensemble, cache_info = load_predictions()
    official_days = [d for d in signal_days if START <= d <= END]
    ix = [signal_days.index(d) for d in official_days]
    seed_scores = {s: seed_scores[s][ix] for s in SEEDS}
    ensemble = ensemble[ix]
    if len(official_days) != 242 or official_days[0] != START or official_days[-1] != END:
        raise RuntimeError(f"official signal window mismatch: {len(official_days)} {official_days[0]}..{official_days[-1]}")
    panel, amount_col = read_small_panel(official_days)
    v1_guard = {}
    for rel in ("model.py", "analysis.py", "model_pred/ensemble/score_meta.json", "model_info/final_audit.json"):
        p = V1 / rel; v1_guard[rel] = sha256(p)
    runs = {}; quarter_out = []; seed_out = []; curves = {}; trades_all = []
    for name, scores in [("ensemble4x4", ensemble)] + [(f"seed_{s}", seed_scores[s]) for s in SEEDS]:
        stats, qrows, curve, trades = run_account(scores, official_days, panel, label=name)
        runs[name] = stats; curves[name] = curve
        if name == "ensemble4x4": quarter_out = qrows; trades_all = trades
        else: seed_out.append(stats)
    result = {"strategy": "V1 cached blend score, monthly first-trading-day Top2, 50/50 shared-cash targets, hold picks intramonth",
              "rule_fixed": True, "score_formula_source": "V1 cached 4 folds × 4 seed predictions; exact match to published V1 ensemble parquet",
              "window": {"signal_start": START, "signal_end": END, "signal_days": len(official_days),
                         "execution_end": runs["ensemble4x4"]["execution_end"]},
              "costs": SPEC, "amount_column": amount_col, "universe": list(CODES),
              "ensemble": runs["ensemble4x4"], "quarters": quarter_out, "seed_sensitivity": seed_out,
              "seed_summary": {"positive_matched_alpha": int(sum(x["strict_matched_alpha"] > 0 for x in seed_out)),
                  "mean_return": float(np.mean([x["return_value"] for x in seed_out])),
                  "mean_strict_matched_alpha": float(np.mean([x["strict_matched_alpha"] for x in seed_out])),
                  "return_min": float(min(x["return_value"] for x in seed_out)),
                  "return_max": float(max(x["return_value"] for x in seed_out))}}
    write_csv(ROOT / "quarterly.csv", quarter_out)
    write_csv(ROOT / "seed_results.csv", seed_out)
    write_csv(ROOT / "daily.csv", curves["ensemble4x4"])
    write_csv(ROOT / "trades.csv", trades_all)
    atomic_json(ROOT / "results.json", result)
    v1_guard_after = {}
    for rel in ("model.py", "analysis.py", "model_pred/ensemble/score_meta.json", "model_info/final_audit.json"):
        v1_guard_after[rel] = sha256(V1 / rel)
    v1_unchanged = (v1_guard_after == v1_guard)
    audit = {
      "window_exact_242_signals": True,
      "monthly_pick_at_first_signal_day_only": True,
      "intramonth_selection_unchanged": True,
      "execution_is_next_trading_day_open": True,
      "model_refit": False,
      "scores_reconstructed_from_cached_seed_predictions": True,
      "published_v1_ensemble_max_abs_delta": cache_info["v1_published_score_max_abs_delta"],
      "monthly_rebalance_dates": runs["ensemble4x4"]["rebalance_dates"],
      "strict_alpha_definition": "daily no-trade return from prior execution-day ticker shares/cash, compounded; current-day rebalance and its costs are excluded from matched benchmark",
      "ledger": {"cash_nonnegative": runs["ensemble4x4"]["cash_nonnegative"],
                 "max_exposure_at_most_one": runs["ensemble4x4"]["exposure_at_most_one"],
                 "final_flat": runs["ensemble4x4"]["final_flat"],
                 "max_cash_mark_ledger_residual": runs["ensemble4x4"]["max_ledger_residual"],
                 "fees_reconcile_to_trade_log": abs(runs["ensemble4x4"]["total_fees"]-sum(t["fees"] for t in trades_all)) < 1e-7,
                 "trade_dates_are_t_plus_1": all(panel.days[np.searchsorted(panel.days, t["signal_date"])+1] == t["date"] for t in trades_all)},
      "v1_guard_sha256_before": v1_guard, "v1_guard_sha256_after": v1_guard_after,
      "v1_unchanged": v1_unchanged, "prediction_cache": cache_info,
      "layout": layout_audit(False),
      "passed": bool(runs["ensemble4x4"]["cash_nonnegative"] and runs["ensemble4x4"]["exposure_at_most_one"]
                     and runs["ensemble4x4"]["final_flat"] and runs["ensemble4x4"]["max_ledger_residual"] < 1e-7
                     and abs(runs["ensemble4x4"]["total_fees"]-sum(t["fees"] for t in trades_all)) < 1e-7
                     and v1_unchanged),
    }
    atomic_json(ROOT / "audit.json", audit)
    provenance = {"created_by": "experiments2/V5/run.py", "v1_root": str(V1),
                  "v1_data_built_at": M.metadata().get("built_at"), "v1_source_sha256_before": v1_guard,
                  "v1_source_sha256_after": v1_guard_after, "v1_unchanged": v1_unchanged,
                  "prediction_cache_files_sha256": cache_info["prediction_cache_sha256"],
                  "data_range_loaded": [str(panel.days[0]), str(panel.days[-1])],
                  "price_rows_loaded": int(len(panel.days)*len(panel.codes)),
                  "trade_date_column": "trade_date", "stock_code_column": "stock_code"}
    atomic_json(ROOT / "provenance.json", provenance)
    (ROOT / "README.md").write_text(
      "# V5: 月度模型分数 Top2\n\n"
      "只读取 V1 四折×四种子缓存分数，不重训、不改 V1。每月首个测试交易日收盘，按原 V1 blend score 四大行固定选前二；下一交易日开盘调到组合总权益各 50% 的目标市值，月内不换票/不再平衡。最后一个官方信号日后于下一开盘平仓。\n\n"
      "费用、滑点、整手、成交额参与率和涨跌停判断沿用 V1 配置；V5 使用共享现金账本以兑现总资产 50/50 目标权重。严格匹配基准在每个开盘用前一执行日各股票持仓股数与现金计算当日不交易收益，并复利。此策略固定规则，不做 TopK/日期扫描。\n\n"
      "正式结果见 results.json、季度表和逐 seed 表；执行与账本校验见 audit.json。该结果只代表 2025-07-01 至 2026-06-30 官方测试信号窗。\n",
      encoding="utf-8")
    la = layout_audit(True)
    audit["layout"] = la; audit["passed"] = audit["passed"] and la["passed"]
    atomic_json(ROOT / "audit.json", audit)
    if not audit["passed"]:
        raise AssertionError(f"V5 audit failed: {audit}")
    print(json.dumps({"ensemble": runs["ensemble4x4"], "quarters": quarter_out,
                      "seed_summary": result["seed_summary"], "audit": audit}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

