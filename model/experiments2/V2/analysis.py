#!/usr/bin/env python3
"""Evaluation, action-account engine, and audit outputs for experiments2/V1."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import model as M
ROOT = M.ROOT
SPEC = M.RECIPE["action"]


def next_state(current, score, thresholds, levels):
    if not np.isfinite(score):
        return float(current)
    lo, hi = float(thresholds["lo"]), float(thresholds["hi"])
    band = max(0.0, float(thresholds.get("band", 0.0)))
    low, mid, high = map(float, levels)
    if current <= low + 1e-12:
        return mid if score > lo + band else low
    if current >= high - 1e-12:
        return mid if score < hi - band else high
    if score < lo - band:
        return low
    if score > hi + band:
        return high
    return mid


def _fees(notional, side, spec):
    commission = max(float(spec["min_commission"]), float(notional) * float(spec["commission_rate"]))
    transfer = float(notional) * float(spec["transfer_rate"])
    stamp = float(notional) * float(spec["stamp_sell_rate"]) if side == "sell" else 0.0
    return commission + transfer + stamp


def _mark_price(px, d, c):
    value = float(px.open[d, c])
    if np.isfinite(value) and value > 0:
        return value
    value = float(px.close[d, c])
    return value if np.isfinite(value) and value > 0 else float("nan")


def _capacity_qty(panel, signal_d, c, price, spec):
    lot = int(spec["lot_size"])
    if not hasattr(panel, "amount"):
        return 10**15
    amount = float(panel.amount[signal_d, c])
    if not np.isfinite(amount) or amount <= 0:
        return 0
    return max(0, int(math.floor(amount * spec["max_participation"] / price / lot)) * lot)


def _trade_record(panel, signal_d, exec_d, code, side, quantity, price, fee, old, new):
    q = float(quantity)
    return {
        "signal_date": str(panel.days[signal_d]), "date": str(panel.days[exec_d]),
        "code": str(code), "side": side,
        "quantity": int(round(q)) if abs(q - round(q)) < 1e-7 else q,
        "price": float(price), "notional": float(q * price), "fees": float(fee),
        "state_from": float(old), "state_to": float(new),
    }


def action_backtest(scores, panel, px, days, thresholds_by_day, spec,
                    hold_all=False, costs=True):
    """Signal at T close; execute the target at T+1 open in independent cash sleeves."""
    days = np.asarray(days, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(panel.codes):
        raise ValueError("scores 必须是 [日期, 股票] 矩阵")
    if len(thresholds_by_day) != len(days) or not len(days):
        raise ValueError("thresholds_by_day 必须与非空 signal days 等长")
    total_money = float(M.RECIPE["account_money"])
    nc = len(panel.codes)
    sleeve = total_money / nc
    cash = np.full(nc, sleeve, dtype=np.float64)
    shares = np.zeros(nc, dtype=np.float64)
    target_qty_state = np.zeros(nc, dtype=np.float64)
    states = np.full(nc, float(spec.get("initial_state", 0.0)), dtype=np.float64)
    previous_equity, previous_exec = total_money, None
    last_marks = np.full(nc, np.nan)
    curve, trades, daily_returns = [], [], []
    fees = 0.0
    blocked_entries = blocked_exits = corp_actions = 0
    min_cash, max_exposure = float("inf"), 0.0
    lot = int(spec["lot_size"])
    slip = float(spec["slippage_rate"]) if costs else 0.0

    for pos, signal_d0 in enumerate(days):
        signal_d = int(signal_d0)
        exec_d = signal_d + 1
        if exec_d >= len(panel.days):
            raise ValueError(f"最后信号日 {panel.days[signal_d]} 没有 T+1 开盘")
        if previous_exec is not None:
            for c in range(nc):
                a0, a1 = float(px.adj[previous_exec, c]), float(px.adj[exec_d, c])
                if np.isfinite(a0) and np.isfinite(a1) and a0 > 0 and a1 > 0:
                    ratio = a1 / a0
                    if abs(ratio - 1.0) > 1e-8 and shares[c] > 0:
                        shares[c] *= ratio
                        target_qty_state[c] *= ratio
                        corp_actions += 1
        marks = np.asarray([_mark_price(px, exec_d, c) for c in range(nc)], dtype=float)
        for c in range(nc):
            if np.isfinite(marks[c]):
                last_marks[c] = marks[c]
            elif np.isfinite(last_marks[c]):
                marks[c] = last_marks[c]
        desired = states.copy()
        if hold_all:
            desired[:] = float(spec["levels"][-1])
        else:
            thresholds = thresholds_by_day[pos]
            for c, code in enumerate(panel.codes):
                thr = thresholds.get(str(code), {"lo": 0.0, "hi": 0.0, "band": 0.0})
                desired[c] = next_state(states[c], scores[signal_d, c], thr, spec["levels"])
        effective = desired.copy()
        is_final = pos == len(days) - 1
        if is_final:
            effective[:] = float(spec["levels"][0])

        for c, code in enumerate(panel.codes):
            mark = marks[c]
            if not np.isfinite(mark) or mark <= 0:
                if effective[c] < states[c] or (is_final and shares[c] > 0):
                    blocked_exits += 1
                elif effective[c] > states[c]:
                    blocked_entries += 1
                continue
            equity = cash[c] + shares[c] * mark
            if is_final:
                target_qty = 0.0
            elif abs(effective[c] - states[c]) > 1e-8:
                target_qty_state[c] = math.floor(max(equity, 0.0) * effective[c] /
                                                 mark / lot) * lot
                target_qty = target_qty_state[c]
            else:
                # Once a state is entered, keep its share target stable; only retry an
                # unfilled difference after a halt/capacity block. Splits scale target shares.
                target_qty = target_qty_state[c]
            delta = target_qty - shares[c]
            side = "buy" if delta > 1e-7 else ("sell" if delta < -1e-7 else None)
            if side is None:
                continue
            allowed = getattr(px, "entry" if side == "buy" else "exit", None)
            if allowed is not None and not bool(allowed[exec_d, c]):
                if side == "buy":
                    blocked_entries += 1
                else:
                    blocked_exits += 1
                continue
            cap = _capacity_qty(panel, signal_d, c, mark, spec)
            if side == "buy":
                target_request = int(math.floor(delta / lot) * lot)
                qty = min(target_request, int(cap))
                qty = (qty // lot) * lot
                while qty > 0:
                    execution_price = mark * (1.0 + slip)
                    notional = qty * execution_price
                    fee = _fees(notional, side, spec) if costs else 0.0
                    if notional + fee <= cash[c] + 1e-8:
                        break
                    qty -= lot
                if qty <= 0:
                    continue
                cash[c] -= notional + fee
                shares[c] += qty
                if qty < target_request and cap >= target_request:
                    # If the account cannot fund another full lot after costs, accept the
                    # largest affordable position as the achieved target for this state.
                    target_qty_state[c] = shares[c]
            else:
                req = float(shares[c]) if effective[c] == 0.0 else float(math.floor(abs(delta) / lot) * lot)
                qty = min(req, float((int(cap) // lot) * lot))
                if effective[c] == 0.0 and qty >= shares[c] - 1e-7 and req >= shares[c] - 1e-7:
                    qty = float(shares[c])
                else:
                    qty = float(math.floor(qty / lot) * lot)
                if qty <= 0:
                    continue
                execution_price = mark * (1.0 - slip)
                notional = qty * execution_price
                fee = _fees(notional, side, spec) if costs else 0.0
                cash[c] += notional - fee
                shares[c] -= qty
                if shares[c] < 1e-8:
                    shares[c] = 0.0
            fees += fee
            trades.append(_trade_record(panel, signal_d, exec_d, code, side, qty,
                                        execution_price, fee, states[c], effective[c]))

        # Keep the signal state in the curve; the forced final liquidation only changes executed holdings.
        states = desired
        marked = np.nan_to_num(marks, nan=0.0)
        equity_by_code = cash + shares * marked
        equity_total = float(equity_by_code.sum())
        invested = float(np.sum(shares * marked))
        exposure = invested / equity_total if equity_total > 0 else 0.0
        ret = equity_total / previous_equity - 1.0 if previous_equity > 0 else 0.0
        daily_returns.append(float(ret))
        min_cash = min(min_cash, float(np.min(cash)))
        max_exposure = max(max_exposure, exposure)
        curve.append({
            "signal_date": str(panel.days[signal_d]), "date": str(panel.days[exec_d]),
            "cash": float(cash.sum()), "equity": equity_total, "exposure": float(exposure),
            "daily_return": float(ret), "states": states.tolist(), "shares": shares.tolist(),
            "ticket_cash": cash.tolist(), "ticket_equity": equity_by_code.tolist(),
            "ticket_exposure": (shares * marked / np.maximum(equity_by_code, 1e-12)).tolist(),
        })
        previous_equity, previous_exec = equity_total, exec_d

    end_value = float(curve[-1]["equity"])
    values = np.asarray([total_money] + [r["equity"] for r in curve], dtype=float)
    drawdown = values / np.maximum.accumulate(values) - 1.0
    total_return = end_value / total_money - 1.0
    years = len(curve) / 242.0
    annualized = max(end_value / total_money, 1e-12) ** (1.0 / years) - 1.0 if years else 0.0
    stats = {
        "money": total_money, "return_value": float(total_return),
        "gross_return": float(total_return) if not costs else None,
        "annualized": float(annualized), "max_drawdown": float(drawdown.min()),
        "total_fees": float(fees), "trades": len(trades),
        "avg_exposure": float(np.mean([r["exposure"] for r in curve])),
        "blocked_entries": int(blocked_entries), "blocked_exits": int(blocked_exits),
        "ending_positions": int(np.count_nonzero(shares > 1e-8)),
        "ending_shares": shares.tolist(), "ending_cash": cash.tolist(),
        "corporate_action_adjustments": int(corp_actions), "min_cash": float(min_cash),
        "max_exposure": float(max_exposure), "daily_returns": daily_returns,
        "final_equity": end_value, "curve_days": len(curve),
    }
    return stats, curve, trades


def _corr(x, y, ranked=False):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return None, int(ok.sum())
    if ranked:
        x = pd.Series(x[ok]).rank(method="average").to_numpy()
        y = pd.Series(y[ok]).rank(method="average").to_numpy()
    else:
        x, y = x[ok], y[ok]
    return float(np.corrcoef(x, y)[0, 1]), int(ok.sum())


def _metric_table(scores, panel, day_ix):
    out = {}
    for label in ("label_ret_1d", "label_ret_5d"):
        per_code, pvals, svals, ns = {}, [], [], []
        for c, code in enumerate(panel.codes):
            y = panel.Y[label][day_ix, c]
            p, n = _corr(scores[day_ix, c], y)
            s, _ = _corr(scores[day_ix, c], y, ranked=True)
            per_code[str(code)] = {"pearson": p, "spearman": s, "days": n,
                                   "standard_error": 1.0 / math.sqrt(max(n - 3, 1))}
            if p is not None:
                pvals.append(p)
            if s is not None:
                svals.append(s)
            ns.append(n)
        nbar = float(np.mean(ns)) if ns else 0.0
        out[label] = {"pearson": float(np.mean(pvals)) if pvals else None,
                      "spearman": float(np.mean(svals)) if svals else None,
                      "days": int(round(nbar)),
                      "standard_error": float(1.0 / math.sqrt(max(nbar - 3, 1))),
                      "per_code": per_code}
    return out


def _target_path(scores, panel, days, thresholds_by_day):
    states = np.full(len(panel.codes), float(SPEC.get("initial_state", 0.0)))
    target = np.zeros((len(days), len(panel.codes)), dtype=np.float32)
    for i, d in enumerate(days):
        for c, code in enumerate(panel.codes):
            thr = thresholds_by_day[i].get(str(code), {"lo": 0, "hi": 0, "band": 0})
            states[c] = next_state(states[c], scores[d, c], thr, SPEC["levels"])
        target[i] = states
    return target


def _hit_rates(target, panel, days):
    y = panel.Y["label_ret_1d"][days]
    prev = np.zeros(len(panel.codes), dtype=np.float32)
    add_flags = np.zeros_like(target, dtype=bool)
    exit_flags = np.zeros_like(target, dtype=bool)
    for i in range(len(days)):
        add_flags[i] = target[i] > prev + 1e-8
        exit_flags[i] = target[i] < prev - 1e-8
        prev = target[i]
    out, add, addbase, exits, exitbase = {}, [], [], [], []
    for c, code in enumerate(panel.codes):
        valid = np.isfinite(y[:, c])
        aa = add_flags[:, c] & valid
        ee = exit_flags[:, c] & valid
        ar = float(np.mean(y[aa, c] > 0)) if aa.any() else None
        ab = float(np.mean(y[valid, c] > 0)) if valid.any() else None
        er = float(np.mean(y[ee, c] < 0)) if ee.any() else None
        eb = float(np.mean(y[valid, c] < 0)) if valid.any() else None
        out[str(code)] = {"add_count": int(aa.sum()), "add_hit_rate": ar,
                          "positive_base_rate": ab, "exit_count": int(ee.sum()),
                          "exit_hit_rate": er, "negative_base_rate": eb}
        if ar is not None: add.append(ar)
        if ab is not None: addbase.append(ab)
        if er is not None: exits.append(er)
        if eb is not None: exitbase.append(eb)
    out["equal_weight"] = {
        "add_hit_rate": float(np.mean(add)) if add else None,
        "positive_base_rate": float(np.mean(addbase)) if addbase else None,
        "exit_hit_rate": float(np.mean(exits)) if exits else None,
        "negative_base_rate": float(np.mean(exitbase)) if exitbase else None}
    return out


def _label_price_check(panel):
    adj_open = panel.prices.open * panel.prices.adj
    out = {}
    for h in (1, 3, 5, 10, 20):
        label = panel.Y[f"label_ret_{h}d"]
        errors = []
        for d in range(max(0, len(panel.days) - h - 1)):
            p0, p1 = adj_open[d + 1], adj_open[d + 1 + h]
            ok = np.isfinite(label[d]) & np.isfinite(p0) & np.isfinite(p1) & (p0 > 0)
            if ok.any():
                errors.extend(np.abs(label[d, ok] - (p1[ok] / p0[ok] - 1.0)).tolist())
        maximum = float(max(errors)) if errors else None
        out[f"{h}d"] = {"n": len(errors), "max_abs_error": maximum,
                        "passed": bool(maximum is not None and maximum <= 1e-5)}
    return out



def _mean_threshold_maps(items):
    result = {}
    for code in M.CODES:
        result[code] = {key: float(np.mean([item[code][key] for item in items]))
                         for key in ("lo", "hi", "band")}
    return result


def _quarter_name(date):
    p = str(pd.Period(date, freq="Q"))
    return p[:4] + "Q" + p[-1]


def _source_hash_snapshot():
    source = M.SOURCE_ROOT
    paths = [source / "model.py", source / "analysis.py",
             source / "model_info" / "final_audit.json",
             source / "model_info" / "action_validation.json",
             source / "model_pred" / "ensemble" / "score_meta.json"]
    paths.extend((source / "model_train").rglob("*"))
    out = {}
    for path in sorted(x for x in paths if x.is_file()):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        out[path.relative_to(source).as_posix()] = digest.hexdigest()
    return out


def _banded_thresholds(base, fraction):
    return {str(code): {"lo": float(row["lo"]), "hi": float(row["hi"]),
                        "band": max(0.0, (float(row["hi"]) - float(row["lo"])) * fraction)}
            for code, row in base.items()}


def _collect_predictions(panel):
    seeds = list(M.SEEDS)
    test_ix = np.flatnonzero((panel.days >= M.RECIPE["test_start"]) &
                             (panel.days <= M.RECIPE["test_end"]))
    seed_scores = {seed: np.full((len(panel.days), len(panel.codes)), np.nan, np.float32)
                   for seed in seeds}
    ensemble_scores = np.full((len(panel.days), len(panel.codes)), np.nan, np.float32)
    variant_names = ("q20_80_b00", "q20_80_b10", "q20_80_b25",
                     "q30_70_b00", "q30_70_b10", "q30_70_b25", "q30_70_b200",
                     "q40_60_b00", "q40_60_b10", "q40_60_b25")
    variant_by_seed = {name: {seed: [None] * len(test_ix) for seed in seeds}
                       for name in variant_names}
    fold_payloads_by_q = {}

    for q in M.QUARTERS:
        period = pd.Period(q, freq="Q")
        qstart = period.start_time.strftime("%Y-%m-%d")
        qend = period.end_time.strftime("%Y-%m-%d")
        q_ix = np.flatnonzero((panel.days >= qstart) & (panel.days <= qend))
        if not len(q_ix):
            raise ValueError(f"{q} test 区间无日期")
        payloads, predictions = [], []
        for fold in M.FOLDS:
            base = M.SOURCE_ROOT / "model_train" / q / f"fold{fold}"
            done = base / "complete.json"
            if not done.is_file():
                raise FileNotFoundError(f"缺少训练折：{base}")
            info = json.loads(done.read_text(encoding="utf-8"))
            if info.get("data_built_at") != panel.data_built_at:
                raise RuntimeError(f"{q} fold{fold} 使用旧数据快照，请先重训")
            with open(base / "best.pt", "rb") as f:
                payload = pickle.load(f)
            arr = np.asarray(payload["seed_predictions"], dtype=np.float32)
            if arr.shape[0] != len(seeds) or arr.shape[2] != len(panel.codes):
                raise ValueError(f"{q} fold{fold} seed 预测形状不符")
            if arr.shape[1] < len(q_ix):
                raise ValueError(f"{q} fold{fold} score 天数短于 test")
            payloads.append(payload)
            predictions.append(arr[:, :len(q_ix), :])
        fold_payloads_by_q[q] = payloads
        by_seed = np.mean(np.stack(predictions, axis=0), axis=0)
        for si, seed in enumerate(seeds):
            seed_scores[seed][q_ix] = by_seed[si]
        ensemble_scores[q_ix] = np.mean(by_seed, axis=0)

        for name in variant_names:
            for si, seed in enumerate(seeds):
                thresholds = []
                for payload in payloads:
                    bundle = next(x for x in payload["bundles"] if int(x["seed"]) == int(seed))
                    variants = bundle.get("threshold_variants", {})
                    base_map = variants.get(name)
                    if name == "q30_70_b200":
                        base_map = variants.get("q30_70_b00")
                        if base_map is None:
                            raise ValueError(f"{q} fold{fold} seed{seed}: 缺少q30/q70因果验证阈值")
                        base_map = _banded_thresholds(base_map, 2.0)
                    if base_map is None:
                        base_map = bundle["thresholds"]
                    thresholds.append(base_map)
                avg = _mean_threshold_maps(thresholds)
                for day_pos in np.flatnonzero(np.isin(test_ix, q_ix)):
                    variant_by_seed[name][seed][day_pos] = avg

    # The most recent quarter model also scores dates after the standard test window.
    latest_q = M.QUARTERS[-1]
    latest_start = pd.Period(latest_q, freq="Q").start_time.strftime("%Y-%m-%d")
    latest_end_ix = np.flatnonzero(panel.days >= latest_start)
    qtest_count = int(np.sum((panel.days >= latest_start) & (panel.days <= M.RECIPE["test_end"])))
    post_ix = latest_end_ix[latest_end_ix >= (test_ix[-1] + 1)]
    latest_payloads = fold_payloads_by_q[latest_q]
    if len(post_ix):
        future_preds = []
        for payload in latest_payloads:
            future_preds.append(np.asarray(payload["seed_predictions"])[:, qtest_count:, :])
        future = np.mean(np.stack(future_preds, axis=0), axis=0)
        if future.shape[1] != len(post_ix):
            raise ValueError(f"最新模型增量预测长度错误：{future.shape[1]} != {len(post_ix)}")
        for si, seed in enumerate(seeds):
            seed_scores[seed][post_ix] = future[si]
        ensemble_scores[post_ix] = np.mean(future, axis=0)

    full_thresholds = {name: [None] * len(panel.days) for name in variant_names}
    for name in variant_names:
        for i, d in enumerate(test_ix):
            full_thresholds[name][d] = _mean_threshold_maps(
                [variant_by_seed[name][seed][i] for seed in seeds])
        if len(post_ix):
            latest_maps = []
            for seed in seeds:
                per_fold = []
                for payload in latest_payloads:
                    bundle = next(x for x in payload["bundles"] if int(x["seed"]) == int(seed))
                    variants = bundle.get("threshold_variants", {})
                    base_map = variants.get(name)
                    if name == "q30_70_b200":
                        base_map = variants.get("q30_70_b00")
                        if base_map is None:
                            raise ValueError("最新季度fold缺少q30/q70因果验证阈值")
                        base_map = _banded_thresholds(base_map, 2.0)
                    if base_map is None:
                        base_map = bundle["thresholds"]
                    per_fold.append(base_map)
                latest_maps.append(_mean_threshold_maps(per_fold))
            latest = _mean_threshold_maps(latest_maps)
            for d in post_ix:
                full_thresholds[name][d] = latest
    return test_ix, seed_scores, ensemble_scores, variant_by_seed, full_thresholds, post_ix, fold_payloads_by_q


def _summarize(stats, buy_hold_return):
    return {
        "return_value": float(stats["return_value"]),
        "annualized": float(stats["annualized"]),
        "max_drawdown": float(stats["max_drawdown"]),
        "total_fees": float(stats["total_fees"]),
        "trades": int(stats["trades"]),
        "avg_exposure": float(stats["avg_exposure"]),
        "blocked_entries": int(stats["blocked_entries"]),
        "blocked_exits": int(stats["blocked_exits"]),
        "ending_positions": int(stats["ending_positions"]),
        "alpha_vs_buy_hold": float(stats["return_value"] -
                                   stats["avg_exposure"] * buy_hold_return),
        "final_equity": float(stats["final_equity"]),
    }


def _quarter_stats(curve, trades, q):
    period_q = q[:4] + "Q" + q[-1]
    rows = [r for r in curve if _quarter_name(r["signal_date"]) == period_q]
    if not rows:
        raise ValueError(f"{q} 没有账户曲线")
    returns = np.asarray([r["daily_return"] for r in rows], dtype=float)
    qtrades = [t for t in trades if _quarter_name(t["signal_date"]) == period_q]
    return {
        "quarter": q, "days": len(rows),
        "return_value": float(np.prod(1.0 + returns) - 1.0),
        "avg_exposure": float(np.mean([r["exposure"] for r in rows])),
        "total_fees": float(sum(t["fees"] for t in qtrades)),
        "trades": len(qtrades),
    }


def _write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_cash_csv(path, curve, panel):
    rows = []
    for row in curve:
        for c, code in enumerate(panel.codes):
            rows.append({
                "signal_date": row["signal_date"], "trade_date": row["date"],
                "stock_code": str(code), "target_position": row["states"][c],
                "shares": row["shares"][c], "cash": row["ticket_cash"][c],
                "equity": row["ticket_equity"][c], "exposure": row["ticket_exposure"][c],
                "portfolio_equity": row["equity"], "daily_return": row["daily_return"],
            })
    fields = ["signal_date", "trade_date", "stock_code", "target_position", "shares",
              "cash", "equity", "exposure", "portfolio_equity", "daily_return"]
    _write_csv(path, rows, fields)


def _plot(path, action_curve, hold_curve):
    dates = [r["date"] for r in action_curve]
    action = np.asarray([r["equity"] for r in action_curve], dtype=float)
    hold = np.asarray([r["equity"] for r in hold_curve], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    ax.plot(dates, action / M.RECIPE["account_money"], label="action_3state")
    ax.plot(dates, hold / M.RECIPE["account_money"], label="buy_hold", alpha=0.8)
    ax.set_title("experiments2/V2 candidate · net cash-account equity")
    ax.set_ylabel("Equity / initial capital")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _audit_reloads(panel):
    max_diff, checks = 0.0, 0
    for q in M.QUARTERS:
        for fold in M.FOLDS:
            base = M.SOURCE_ROOT / "model_train" / q / f"fold{fold}"
            with open(base / "best.pt", "rb") as f:
                payload = pickle.load(f)
            saved = np.load(base / "score_predictions.npy")
            days = np.asarray([np.searchsorted(panel.days, d) for d in payload["score_dates"]])
            seed_preds = []
            for bundle in payload["bundles"]:
                p1 = M._predict_days(bundle["models"]["label_ret_1d"], panel.X, days)
                p5 = M._predict_days(bundle["models"]["label_ret_5d"], panel.X, days)
                pred = (M.RECIPE["score_blend"]["label_ret_1d"] *
                        (p1 - bundle["mean"]["label_ret_1d"]) / bundle["sd"]["label_ret_1d"] +
                        M.RECIPE["score_blend"]["label_ret_5d"] *
                        (p5 - bundle["mean"]["label_ret_5d"]) / bundle["sd"]["label_ret_5d"])
                seed_preds.append(pred)
            reproduced = np.mean(np.stack(seed_preds, axis=0), axis=0)
            if reproduced.shape != saved.shape:
                raise AssertionError(f"{q} fold{fold} 重载预测形状不一致")
            diff = float(np.nanmax(np.abs(reproduced - saved)))
            max_diff = max(max_diff, diff)
            checks += 1
    if max_diff > 1e-5:
        raise AssertionError(f"模型重载推理不一致，最大差 {max_diff}")
    return {"folds": checks, "max_abs_diff": max_diff, "passed": True}


def _write_actions_md(path, audit, latest_rows):
    total = audit["metrics_total"]
    action, hold = total["action"], total["buy_hold"]
    def pct(value):
        return "—" if value is None else f"{value:+.2%}"
    lines = [
        "# experiments2/V2候选：固定2.0倍验证分位距滞回带", "",
        f"- 数据快照：{audit['data']['built_at']}；上游末日 {audit['data']['last_upstream_day']}。",
        f"- 评价窗：{audit['window']['start']} 至 {audit['window']['end']}，"
        f"{audit['window']['days']} 个交易日。",
        "- T 日收盘出信号、T+1 开盘执行；四只银行分别记账，目标仓位 0/50/100%。", "",
        "## 含费账户结果", "",
        "| 策略 | 净收益 | 毛收益 | 成本拖累 | 年化 | 平均敞口 | 匹配 alpha | 回撤 | 费用 | 成交 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| action_3state | {pct(action['return_value'])} | {pct(action['gross_return'])} "
        f"| {pct(action['cost_drag'])} | {pct(action['annualized'])} "
        f"| {action['avg_exposure']:.1%} | {pct(action['alpha_vs_buy_hold'])} "
        f"| {pct(action['max_drawdown'])} | {action['total_fees']:.0f} | {action['trades']} |",
        f"| buy_hold | {pct(hold['return_value'])} | {pct(hold['gross_return'])} "
        f"| {pct(hold['cost_drag'])} | {pct(hold['annualized'])} "
        f"| {hold['avg_exposure']:.1%} | — | {pct(hold['max_drawdown'])} "
        f"| {hold['total_fees']:.0f} | {hold['trades']} |", "",
        "## 季度分项", "",
        "| 季度 | 策略净收益 | 敞口 | 匹配 alpha | 买入持有 | 费用 | 成交 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for q in audit["quarters"]:
        a, b = q["action"], q["buy_hold"]
        alpha = a["return_value"] - a["avg_exposure"] * b["return_value"]
        lines.append(f"| {q['quarter']} | {pct(a['return_value'])} | {a['avg_exposure']:.1%} "
                     f"| {pct(alpha)} | {pct(b['return_value'])} "
                     f"| {a['total_fees']:.0f} | {a['trades']} |")
    lines.extend(["", "## 时序信号", "",
                  "| 标签 | Pearson | Spearman | days | SE |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for label, row in audit["ts_ic"].items():
        lines.append(f"| {label} | {row['pearson'] if row['pearson'] is not None else '—'} "
                     f"| {row['spearman'] if row['spearman'] is not None else '—'} "
                     f"| {row['days']} | {row['standard_error']:.4f} |")
    hit = audit["action_hit_rate"]["equal_weight"]
    lines.extend(["", f"- 加仓命中率 / 基率：{pct(hit['add_hit_rate'])} / {pct(hit['positive_base_rate'])}。",
                  f"- 减仓命中率 / 基率：{pct(hit['exit_hit_rate'])} / {pct(hit['negative_base_rate'])}.",
                  "", "## 同权重阈值敏感性", "",
                  "| 分位阈值 | 滞回带 | 净收益 | 平均敞口 | 匹配 alpha | 成交 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in audit["sensitivity"]:
        label = row["variant"].split("_b")[0].replace("_", "/")
        lines.append(f"| {label} | {row['band_fraction']:.0%} | {pct(row['return_value'])} "
                     f"| {row['avg_exposure']:.1%} | {pct(row['alpha_vs_buy_hold'])} "
                     f"| {row['trades']} |")
    lines.extend(["", "## 最新模型动作", "",
                  "| 股票 | 信号日 | 评分 | 目标仓位 | 动作 | 阈值低/高 |",
                  "| --- | --- | ---: | ---: | --- | ---: |"])
    for row in latest_rows:
        lines.append(f"| {row['stock_code']} | {row['signal_date']} | {row['score']:+.3f} "
                     f"| {row['target_position']:.0%} | {row['action']} "
                     f"| {row['lo']:+.3f}/{row['hi']:+.3f} |")
    lines.extend(["", "## 干净前向测试块", "",
                  "| 季度 | 训练截止 | 测试开始 | 干净 |",
                  "| --- | --- | --- | --- |"])
    for row in audit["clean_blocks"]:
        lines.append(f"| {row['quarter']} | {row['train_end']} | {row['test_start']} "
                     f"| {'是' if row['clean'] else '否'} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(*, audit=False):
    lineage_before = _source_hash_snapshot()
    panel = M.load_panel()
    test_ix, seed_scores, ensemble_scores, threshold_variants, full_thresholds, post_ix, payloads = \
        _collect_predictions(panel)
    primary = "q30_70_b200"
    thresholds = [full_thresholds[primary][d] for d in test_ix]
    full_scores = np.full((len(panel.days), len(panel.codes)), np.nan, dtype=np.float32)
    full_scores[test_ix] = ensemble_scores[test_ix]
    if len(post_ix):
        full_scores[post_ix] = ensemble_scores[post_ix]

    net_action, action_curve, action_trades = action_backtest(
        full_scores, panel, panel.prices, test_ix, thresholds, SPEC)
    gross_action, _, _ = action_backtest(
        full_scores, panel, panel.prices, test_ix, thresholds, SPEC, costs=False)
    hold_thresholds = [{"": {"lo": 0.0, "hi": 0.0, "band": 0.0}} for _ in test_ix]
    empty_scores = np.zeros_like(full_scores)
    net_hold, hold_curve, hold_trades = action_backtest(
        empty_scores, panel, panel.prices, test_ix, hold_thresholds, SPEC, hold_all=True)
    gross_hold, _, _ = action_backtest(
        empty_scores, panel, panel.prices, test_ix, hold_thresholds, SPEC,
        hold_all=True, costs=False)
    hold_return = float(net_hold["return_value"])
    action_summary = _summarize(net_action, hold_return)
    action_summary["gross_return"] = float(gross_action["return_value"])
    action_summary["cost_drag"] = float(gross_action["return_value"] - net_action["return_value"])
    hold_summary = _summarize(net_hold, hold_return)
    hold_summary["gross_return"] = float(gross_hold["return_value"])
    hold_summary["cost_drag"] = float(gross_hold["return_value"] - net_hold["return_value"])
    hold_summary["alpha_vs_buy_hold"] = float(hold_summary["return_value"] -
                                               hold_summary["avg_exposure"] * hold_return)

    # Per-seed variability is kept separate from strategy sensitivity.
    seed_results = []
    for seed in M.SEEDS:
        sfull = np.full_like(full_scores, np.nan)
        sfull[test_ix] = seed_scores[seed][test_ix]
        sth = [threshold_variants[primary][seed][i] for i in range(len(test_ix))]
        stat, _, _ = action_backtest(sfull, panel, panel.prices, test_ix, sth, SPEC)
        alpha = stat["return_value"] - stat["avg_exposure"] * hold_return
        seed_results.append({"seed": int(seed), "return_value": float(stat["return_value"]),
                             "avg_exposure": float(stat["avg_exposure"]),
                             "alpha_vs_buy_hold": float(alpha), "total_fees": float(stat["total_fees"]),
                             "trades": int(stat["trades"])})
    rets = np.asarray([x["return_value"] for x in seed_results])
    alphas = np.asarray([x["alpha_vs_buy_hold"] for x in seed_results])
    seed_summary = {
        "n": len(seed_results), "return_mean": float(rets.mean()),
        "return_std": float(rets.std(ddof=1)), "return_min": float(rets.min()),
        "return_max": float(rets.max()), "alpha_mean": float(alphas.mean()),
        "alpha_std": float(alphas.std(ddof=1)), "positive_alpha_seeds": int((alphas > 0).sum()),
    }

    # Threshold and hysteresis sensitivity uses the same model scores and dates.
    sensitivity = []
    variants = ("q20_80_b00", "q20_80_b10", "q20_80_b25",
                "q30_70_b00", "q30_70_b10", "q30_70_b25", "q30_70_b200",
                "q40_60_b00", "q40_60_b10", "q40_60_b25")
    for variant in variants:
        day_thr = [full_thresholds[variant][d] for d in test_ix]
        stat, _, _ = action_backtest(full_scores, panel, panel.prices, test_ix, day_thr, SPEC)
        sensitivity.append({
            "variant": variant, "band_fraction": int(variant.rsplit("_b", 1)[1]) / 100.0,
            "return_value": float(stat["return_value"]), "avg_exposure": float(stat["avg_exposure"]),
            "alpha_vs_buy_hold": float(stat["return_value"] -
                                       stat["avg_exposure"] * hold_return),
            "total_fees": float(stat["total_fees"]), "trades": int(stat["trades"]),
        })

    ts = _metric_table(full_scores, panel, test_ix)
    target = _target_path(full_scores, panel, test_ix, thresholds)
    hit = _hit_rates(target, panel, test_ix)
    quarter_rows = []
    for q in M.QUARTERS:
        a = _quarter_stats(action_curve, action_trades, q)
        b = _quarter_stats(hold_curve, hold_trades, q)
        a["alpha_vs_buy_hold"] = float(a["return_value"] - a["avg_exposure"] * b["return_value"])
        quarter_rows.append({"quarter": q, "action": a, "buy_hold": b})

    clean = []
    for q in M.QUARTERS:
        info = [json.loads((M.SOURCE_ROOT / "model_train" / q / f"fold{f}" / "complete.json")
                           .read_text(encoding="utf-8")) for f in M.FOLDS]
        start = min(x["test_start"] for x in info)
        train_end = max(x["train_end"] for x in info)
        clean.append({"quarter": q, "train_end": train_end, "test_start": start,
                      "clean": bool(train_end < start),
                      "days": int(sum(x["test_days"] for x in info[:1]))})

    lineage_after = _source_hash_snapshot()
    lineage = {
        "unit": "experiments2/V2_candidate",
        "source_unit": "experiments2/V1",
        "source_root": str(M.SOURCE_ROOT),
        "source_artifact_count": len(lineage_before),
        "source_hashes_before": lineage_before,
        "source_hashes_after": lineage_after,
        "source_unchanged_during_run": lineage_before == lineage_after,
        "reuse_rule": (
            "Read-only reuse of V1 per-fold best.pt, score_predictions.npy, complete.json "
            "and fold metadata; no V1 files copied or written. Each seed/fold uses its saved "
            "causal q30/q70 validation lo/hi and sets band=2.0*(hi-lo). "
            "Four-fold and four-seed score ensemble is unchanged."
        ),
        "changed_policy": {"threshold_quantiles": [0.30, 0.70],
                           "hysteresis_fraction": 2.0,
                           "band_formula": "2.0 * (validation_q70 - validation_q30)"}
    }
    audit_doc = {
        "unit": "experiments2/V2_candidate",
        "data": {"built_at": panel.data_built_at,
                 "last_upstream_day": panel.meta["source"]["last_upstream_day"],
                 "factors": len(panel.factor_features),
                 "market_features": len(panel.market_features),
                 "model_features": int(panel.X.shape[-1])},
        "window": {"start": str(panel.days[test_ix[0]]), "end": str(panel.days[test_ix[-1]]),
                   "days": int(len(test_ix))},
        "metrics_total": {"action": action_summary, "buy_hold": hold_summary,
                          "buy_hold_gross_return": float(gross_hold["return_value"])},
        "quarters": quarter_rows, "ts_ic": ts, "action_hit_rate": hit,
        "seed_results": seed_results,
        "seed_summary": seed_summary, "sensitivity": sensitivity,
        "clean_blocks": clean, "lineage": lineage,
        "gross_account": {"action_return": float(gross_action["return_value"]),
                          "buy_hold_return": float(gross_hold["return_value"]),
                          "action_avg_exposure": float(gross_action["avg_exposure"]),
                          "action_fees_zero": gross_action["total_fees"] == 0.0},
    }
    label_check = _label_price_check(panel)
    engine = {
        "min_cash": float(net_action["min_cash"]), "max_exposure": float(net_action["max_exposure"]),
        "ending_positions": int(net_action["ending_positions"]),
        "blocked_entries": int(net_action["blocked_entries"]),
        "blocked_exits": int(net_action["blocked_exits"]),
        "corporate_action_adjustments": int(net_action["corporate_action_adjustments"]),
        "cash_nonnegative": bool(net_action["min_cash"] >= -1e-6),
        "exposure_at_most_one": bool(net_action["max_exposure"] <= 1.0 + 1e-6),
        "final_flat": bool(net_action["ending_positions"] == 0),
    }
    validation = {
        "unit": "experiments2/V2_candidate", "data_built_at": panel.data_built_at,
        "lineage": {"source_unit": lineage["source_unit"],
                    "source_root": lineage["source_root"],
                    "source_artifact_count": lineage["source_artifact_count"],
                    "source_unchanged_during_run": lineage["source_unchanged_during_run"],
                    "source_hashes": lineage["source_hashes_before"],
                    "reuse_rule": lineage["reuse_rule"],
                    "changed_policy": lineage["changed_policy"]},
        "label_price_check": label_check, "account_engine": engine,
        "model_reload_audit": _audit_reloads(panel) if audit else {"requested": False},
        "layout": M.validate_layout(allow_missing=True),
    }

    out_days = np.flatnonzero(np.isfinite(full_scores).any(axis=1))
    out_thresholds = [full_thresholds[primary][d] for d in out_days]
    out_states = _target_path(full_scores, panel, out_days, out_thresholds)
    pred_rows = []
    for i, d in enumerate(out_days):
        q = _quarter_name(panel.days[d])
        for c, code in enumerate(panel.codes):
            thr = full_thresholds[primary][d][str(code)]
            pred_rows.append({"trade_date": str(panel.days[d]), "stock_code": str(code),
                              "score": float(full_scores[d, c]),
                              "target_position": float(out_states[i, c]),
                              "threshold_lo": thr["lo"], "threshold_hi": thr["hi"],
                              "quarter_model": q})
    for year in (2025, 2026):
        rows = [r for r in pred_rows if r["trade_date"].startswith(str(year))]
        path = ROOT / "model_pred" / "ensemble" / f"year={year}" / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
    M.atomic_json(ROOT / "model_pred" / "ensemble" / "score_meta.json", {
        "unit": "experiments2/V2_candidate", "data_built_at": panel.data_built_at,
        "source_unit": "experiments2/V1",
        "threshold_policy": "causal q30/q70; fixed band=2.0*(q70-q30)",
        "score": "0.30 * centered_pred(1d)/valid_sd + 0.70 * centered_pred(5d)/valid_sd",
        "columns": ["trade_date", "stock_code", "score", "target_position",
                    "threshold_lo", "threshold_hi", "quarter_model"],
        "test_start": M.RECIPE["test_start"], "test_end": M.RECIPE["test_end"],
        "scored_days": len(out_days), "stock_codes": list(M.CODES),
        "features": {"stock": len(panel.factor_features), "market": len(panel.market_features)},
    })
    csvroot = ROOT / "model_pred" / "ensemble" / "CSV"
    _write_cash_csv(csvroot / "cash_action_3state.csv", action_curve, panel)
    _write_cash_csv(csvroot / "cash_buy_hold.csv", hold_curve, panel)
    fields = ["signal_date", "date", "code", "side", "quantity",
              "price", "notional", "fees", "state_from", "state_to"]
    _write_csv(csvroot / "trades_action_3state.csv", action_trades, fields)
    _write_csv(csvroot / "trades_buy_hold.csv", hold_trades, fields)
    _plot(ROOT / "model_pred" / "equity_curves.png", action_curve, hold_curve)

    latest = []
    if len(out_days):
        d = int(out_days[-1])
        prior_i = max(0, len(out_days) - 2)
        prior_state = out_states[prior_i]
        for c, code in enumerate(panel.codes):
            cur = float(out_states[-1, c])
            old = float(prior_state[c])
            action = "加仓" if cur > old else ("减仓" if cur < old else "持有")
            thr = full_thresholds[primary][d][str(code)]
            latest.append({"stock_code": str(code), "signal_date": str(panel.days[d]),
                           "score": float(full_scores[d, c]), "target_position": cur,
                           "action": action, "lo": thr["lo"], "hi": thr["hi"]})
    _write_actions_md(ROOT / "model_pred" / "actions.md", audit_doc, latest)

    M.atomic_json(ROOT / "model_info" / "final_audit.json", audit_doc)
    M.atomic_json(ROOT / "model_info" / "action_validation.json", validation)
    bad = {key: value for key, value in label_check.items() if not value["passed"]}
    if bad:
        raise AssertionError(f"label_price_check 未通过：{bad}")
    if not engine["cash_nonnegative"] or not engine["exposure_at_most_one"]:
        raise AssertionError(f"现金账户不变量失败：{engine}")
    layout = M.validate_layout(allow_missing=False)
    return {"audit": audit_doc, "validation": validation, "layout": layout}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="重载每折模型并逐值对拍预测")
    parser.add_argument("--from-scores", action="store_true")
    parser.add_argument("--actions-only", action="store_true")
    args = parser.parse_args(argv)
    log = ROOT / "model_logs" / "analysis_0.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8", buffering=1) as handle:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = handle
        try:
            result = run_analysis(audit=args.audit)
            stat = result["audit"]["metrics_total"]["action"]
            print(f"净收益 {stat['return_value']:+.3%}；敞口 {stat['avg_exposure']:.1%}；"
                  f"匹配 alpha {stat['alpha_vs_buy_hold']:+.3%}", flush=True)
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    print(json.dumps({"layout": M.validate_layout(allow_missing=False),
                      "analysis_log": str(log)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

