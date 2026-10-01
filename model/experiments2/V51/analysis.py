#!/usr/bin/env python3
"""V51 score evaluation and shared-cash portfolio audit."""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import model as M

ROOT = Path(M.ROOT)
RECIPE = M.RECIPE
MONEY = float(RECIPE.get("account_money", 100000.0))
TEST_START, TEST_END = str(RECIPE["test_start"]), str(RECIPE["test_end"])
SEARCH = RECIPE["strategy_search"]
CLEAN_CFG = RECIPE["clean_selection"]
EXEC_CFG = RECIPE["execution"]
REF_CFG = RECIPE["reference_baselines"]
SEEDS = tuple(int(x) for x in RECIPE["seeds"])
MODES = tuple(EXEC_CFG.get("gross_fee_scenarios",
                           ("gross_zero_cost", "explicit_fees_no_slippage", "net_with_slippage")))
MODE_KIND = {
    "gross_zero_cost": "gross", "explicit_fees_no_slippage": "fees_only",
    "net_with_slippage": "net", "gross": "gross", "fees_only": "fees_only", "net": "net",
}
SEARCH_KEYS = ("topk", "target_exposure", "threshold_modes", "buffer_slots", "rebalance_days")
CANDIDATES = tuple(
    {"k": int(k), "exposure": float(e), "threshold": str(t), "band": int(b), "period": int(p)}
    for k, e, t, b, p in itertools.product(*(SEARCH[key] for key in SEARCH_KEYS))
)
OUT = ROOT / "model_pred/ensemble/CSV"
CSV_CASH, CSV_TRADES = OUT / "cash_action_3state.csv", OUT / "trades_action_3state.csv"
CSV_HOLD, CSV_HOLD_TRADES = OUT / "cash_buy_hold.csv", OUT / "trades_buy_hold.csv"
MD, PNG = ROOT / "model_pred/actions.md", ROOT / "model_pred/equity_curves.png"
AUDIT, VALIDATION = ROOT / "model_info/final_audit.json", ROOT / "model_info/action_validation.json"
SCORE_META = ROOT / "model_pred/ensemble/score_meta.json"
SCORE_PARQUET = {
    2025: ROOT / "model_pred/ensemble/year=2025/data.parquet",
    2026: ROOT / "model_pred/ensemble/year=2026/data.parquet",
}


def _spec():
    return {
        "lot": int(EXEC_CFG["lot_size"]),
        "participation": float(EXEC_CFG["max_participation"]),
        "commission": float(EXEC_CFG["commission_rate"]),
        "min_commission": float(EXEC_CFG["min_commission"]),
        "transfer": float(EXEC_CFG["transfer_rate"]),
        "stamp": float(EXEC_CFG["stamp_sell_rate"]),
        "slippage": float(EXEC_CFG["slippage_rate"]),
        "limit_overrides": EXEC_CFG.get("price_limit_by_code", {}),
    }


def _fee_components(notional, side, spec):
    return {
        "commission": max(spec["min_commission"], notional * spec["commission"]),
        "transfer": notional * spec["transfer"],
        "stamp_duty": notional * spec["stamp"] if side == "sell" else 0.0,
    }


def _fee(notional, side, spec):
    return float(sum(_fee_components(notional, side, spec).values()))


def _load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as first:
        loader = getattr(getattr(M, "torch", None), "load", None)
        if loader is None:
            raise RuntimeError(f"cannot load checkpoint {path}: {first}") from first
        return loader(path, map_location="cpu", weights_only=False)


def _dates(value):
    if value is None:
        return None
    a = np.asarray(value)
    return np.asarray([str(x)[:10] for x in a], dtype=str) if a.ndim == 1 else None


def _pred(payload, keys):
    for key in keys:
        if key in payload:
            x = np.asarray(payload[key], dtype=np.float32)
            if x.ndim == 3:
                return x
    return None


def _normalize(pred, payload, panel):
    pred = np.asarray(pred, dtype=np.float32)
    expected_codes = tuple(str(x) for x in RECIPE["universe"])
    payload_codes = tuple(str(x) for x in payload.get("codes", ()))
    if tuple(str(x) for x in panel.codes) != expected_codes:
        raise ValueError("analysis panel code order differs from frozen RECIPE universe")
    if payload_codes != expected_codes:
        raise ValueError("checkpoint codes differ from frozen RECIPE order")
    ids = tuple(int(x) for x in payload.get("seed_ids", payload.get("seeds", ())))
    if ids != SEEDS:
        raise ValueError(f"checkpoint seed order {ids} differs from RECIPE order {SEEDS}")
    if pred.ndim != 3 or pred.shape[0] != len(SEEDS) or pred.shape[2] != len(expected_codes):
        raise ValueError(f"checkpoint prediction shape {pred.shape} does not match seeds and frozen codes")
    return pred


def _make_panel(source, codes=None):
    codes = tuple(str(x) for x in (codes or RECIPE["universe"]))
    source_codes = tuple(str(x) for x in source.codes)
    lookup = {x: i for i, x in enumerate(source_codes)}
    missing = [x for x in codes if x not in lookup]
    if missing:
        raise ValueError(f"RECIPE frozen pool absent from panel: {missing[:8]}")
    ix = np.asarray([lookup[x] for x in codes], dtype=np.int64)
    price_obj = getattr(source, "prices", None)
    raw = getattr(price_obj, "raw", price_obj)
    prices = {}
    for name in getattr(M, "PRICE_COLUMNS", ("open", "close", "pre_close", "vol", "adj_factor")):
        if isinstance(raw, dict) and name in raw:
            prices[name] = np.asarray(raw[name], dtype=np.float64)[:, ix]
        elif hasattr(price_obj, name):
            prices[name] = np.asarray(getattr(price_obj, name), dtype=np.float64)[:, ix]
    for name in ("open", "close", "pre_close", "vol", "adj_factor"):
        if name not in prices:
            raise ValueError(f"model panel missing price column {name}")
    ys = getattr(source, "Y", getattr(source, "labels", {}))
    labels = {}
    if isinstance(ys, dict):
        for name, value in ys.items():
            a = np.asarray(value, dtype=np.float64)
            if a.ndim == 2:
                labels[str(name)] = a[:, ix]
    if not labels:
        raise ValueError("panel must expose target matrices in Y or labels")
    p = type("PanelView", (), {})()
    p.days = np.asarray([str(x)[:10] for x in source.days], dtype=str)
    p.codes, p.amount, p.prices, p.Y = np.asarray(codes, dtype=str), np.asarray(source.amount)[:, ix], prices, labels
    p.source_codes, p.source_indices = source_codes, ix
    p.data_built_at = (getattr(source, "data_built_at", None)
                       or getattr(source, "meta", {}).get("built_at"))
    return p


def _quarters_folds():
    qs = tuple(getattr(M, "QUARTERS", ()))
    fs = tuple(int(x) for x in getattr(M, "FOLDS", (1, 2, 3, 4)))
    return qs or ("2025Q3", "2025Q4", "2026Q1", "2026Q2"), fs


def _quarter_for_date(date):
    text = str(date)[:10]
    year, month = int(text[:4]), int(text[5:7])
    return f"{year}Q{(month - 1) // 3 + 1}"


def _latest_model_quarter(date, quarters):
    text = str(date)[:10]
    eligible = []
    for quarter in quarters:
        year, q = str(quarter).split("Q")
        start = f"{int(year):04d}-{(int(q) - 1) * 3 + 1:02d}-01"
        if text >= start:
            eligible.append(str(quarter))
    return eligible[-1] if eligible else None


def _validated_checkpoint(panel, source, quarter, fold):
    base = ROOT / "model_train" / str(quarter) / f"fold{int(fold)}"
    payload = _load(base / "best.pt")
    complete = json.loads((base / "complete.json").read_text(encoding="utf-8"))
    training = json.loads((base / "training_info.json").read_text(encoding="utf-8"))
    stored_recipe = training.get("recipe")
    if not isinstance(stored_recipe, dict):
        raise ValueError(f"{quarter}/fold{fold} training_info lacks RECIPE")
    device = stored_recipe.get("device")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"{quarter}/fold{fold} has unknown recipe device {device!r}")
    expected_recipe_raw = copy.deepcopy(M.RECIPE)
    expected_recipe_raw.update(quarter=str(quarter), fold=int(fold),
                               data_built_at=source.meta.get("built_at"), device=device)
    expected_recipe_json = json.loads(json.dumps(expected_recipe_raw, ensure_ascii=False))
    if stored_recipe != expected_recipe_json:
        raise ValueError(f"{quarter}/fold{fold} training RECIPE differs from current frozen RECIPE")
    # Match the in-memory recipe representation used by training when verifying its signature.
    signature = M._fold_signature(expected_recipe_raw, source)
    if (payload.get("schema") != "v51.seed-bundle.v1"
            or payload.get("recipe_signature") != signature
            or complete.get("recipe_signature") != signature):
        raise ValueError(f"{quarter}/fold{fold} checkpoint signature mismatch")
    if (payload.get("recipe_name") != RECIPE["name"]
            or complete.get("recipe_name") != RECIPE["name"]
            or payload.get("quarter") != str(quarter)
            or complete.get("quarter") != str(quarter)
            or int(payload.get("fold", -1)) != int(fold)
            or int(complete.get("fold", -1)) != int(fold)):
        raise ValueError(f"{quarter}/fold{fold} checkpoint identity mismatch")
    if payload.get("data_built_at") != source.meta.get("built_at"):
        raise ValueError(f"{quarter}/fold{fold} data snapshot mismatch")
    expected_codes = tuple(str(x) for x in RECIPE["universe"])
    if tuple(str(x) for x in panel.codes) != expected_codes:
        raise ValueError("panel code order differs from RECIPE universe")
    if tuple(str(x) for x in payload.get("codes", ())) != expected_codes:
        raise ValueError(f"{quarter}/fold{fold} checkpoint code order mismatch")

    split = M.splits(panel.days, RECIPE["purge_horizon"], RECIPE["folds"], str(quarter))[int(fold) - 1]
    expected = {key: np.asarray(panel.days[split[key]], dtype=str)
                for key in ("valid", "test", "score")}
    actual = {
        "valid": _dates(payload.get("valid_dates")),
        "test": _dates(payload.get("test_dates")),
        "score": _dates(payload.get("score_dates")),
    }
    for key in expected:
        if actual[key] is None or not np.array_equal(actual[key], expected[key]):
            raise ValueError(f"{quarter}/fold{fold} {key} date axis differs from M.splits")
    seeds = _normalize(_pred(payload, ("seed_predictions",)), payload, panel)
    valid = _normalize(_pred(payload, ("valid_seed_predictions",)), payload, panel)
    if seeds.shape != (len(SEEDS), len(expected["score"]), len(panel.codes)):
        raise ValueError(f"{quarter}/fold{fold} score prediction shape mismatch: {seeds.shape}")
    if valid.shape != (len(SEEDS), len(expected["valid"]), len(panel.codes)):
        raise ValueError(f"{quarter}/fold{fold} valid prediction shape mismatch: {valid.shape}")
    if np.shape(payload.get("score_predictions_mean")) != (len(expected["score"]), len(panel.codes)):
        raise ValueError(f"{quarter}/fold{fold} mean score prediction shape mismatch")
    return {"payload": payload, "complete": complete, "split": split,
            "signature": signature, "score_dates": expected["score"],
            "valid_dates": expected["valid"], "test_dates": expected["test"],
            "seed_predictions": seeds, "valid_seed_predictions": valid}


def _load_clean(panel, source):
    cfg = CLEAN_CFG
    q, f = str(cfg["quarter"]), int(cfg["fold"])
    checked = _validated_checkpoint(panel, source, q, f)
    payload = checked["payload"]
    dates = checked["valid_dates"]
    pred = checked["valid_seed_predictions"]
    lookup = {str(d): i for i, d in enumerate(panel.days)}
    keep = [j for j, d in enumerate(dates) if str(d) in lookup and str(d) < TEST_START]
    if not keep:
        raise ValueError("clean validation block is missing or reaches into the test window")
    dates, pred = dates[keep], pred[:, keep]
    order = np.argsort(dates)
    dates, pred = dates[order], pred[:, order]
    ix = np.asarray([lookup[str(d)] for d in dates], dtype=np.int64)
    if len(ix) < int(cfg["contiguous_segments"]) or np.any(np.diff(ix) != 1):
        raise ValueError("clean validation dates must be continuous on the trading calendar")
    return dates, ix, pred, payload


def _train_end(payload, quarter, fold):
    for key in ("train_end", "training_end", "train_last_date"):
        if payload.get(key):
            return str(payload[key])[:10]
    path = ROOT / "model_train" / str(quarter) / f"fold{fold}" / "split.json"
    if not path.is_file():
        return None
    split = json.loads(path.read_text(encoding="utf-8"))
    train = split.get("train", [])
    if train and isinstance(train[0], dict):
        ends = [str(x["end"])[:10] for x in train if x.get("end")]
        return max(ends) if ends else None
    if train and isinstance(train[0], str):
        return max(str(x)[:10] for x in train)
    return None


def _rescore_signal(panel, source, prices, date, quarter):
    """Rescore a quarter-boundary signal from the quarter model that can trade it."""
    day = int(np.flatnonzero(panel.days == str(date))[0])
    _, folds = _quarters_folds()
    fold_sum = np.zeros((len(SEEDS), len(panel.codes)), dtype=np.float64)
    fold_count = np.zeros((len(SEEDS), len(panel.codes)), dtype=np.int16)
    for fold in folds:
        checked = _validated_checkpoint(panel, source, quarter, fold)
        payload = checked["payload"]
        completed = payload.get("completed")
        if not isinstance(completed, list):
            raise ValueError(f"{quarter}/fold{fold} checkpoint has no completed seed bundles")
        by_seed = {}
        for item in completed:
            bundle = item.get("bundle", {})
            seed = int(item.get("seed", bundle.get("seed", -1)))
            if seed in by_seed:
                raise ValueError(f"duplicate seed {seed} in {quarter}/fold{fold}")
            by_seed[seed] = bundle
        if tuple(by_seed) != SEEDS:
            raise ValueError(f"{quarter}/fold{fold} bundle seed order differs from RECIPE")
        for si, seed in enumerate(SEEDS):
            bundle = by_seed[seed]
            model = M.PredictModel(int(bundle["input_dim"]), int(bundle["market_dim"]),
                                   bundle.get("horizons")).cpu()
            model.load_state_dict(bundle["model"], strict=True)
            model.ridge = np.asarray(bundle["ridge"], dtype=np.float64)
            pred = M.predict(model, source, prices, np.asarray([day], dtype=np.int64), device="cpu")
            pred = np.asarray(pred, dtype=np.float32)
            if pred.shape != (1, len(panel.codes)):
                raise ValueError(f"unexpected rescored shape {pred.shape} for seed {seed}")
            good = np.isfinite(pred[0])
            fold_sum[si, good] += pred[0, good]
            fold_count[si, good] += 1
            del model, pred
    result = np.full((len(SEEDS), len(panel.codes)), np.nan, dtype=np.float32)
    np.divide(fold_sum, fold_count, out=result, where=fold_count > 0)
    return result, len(folds)


def _load_quarter_scores(panel, source):
    quarters, folds = _quarters_folds()
    outputs, signatures, records = {}, {}, []
    for quarter in quarters:
        q_dates = None
        q_sum = q_count = None
        signatures[str(quarter)] = {}
        for fold in folds:
            checked = _validated_checkpoint(panel, source, quarter, fold)
            dates = checked["score_dates"]
            preds = checked["seed_predictions"]
            if q_dates is None:
                q_dates = dates
                q_sum = np.zeros(preds.shape, dtype=np.float64)
                q_count = np.zeros(preds.shape, dtype=np.int16)
            elif not np.array_equal(q_dates, dates):
                raise ValueError(f"{quarter} fold score date axes disagree")
            good = np.isfinite(preds)
            q_sum[good] += preds[good]
            q_count[good] += 1
            signatures[str(quarter)][str(fold)] = checked["signature"]
        if q_dates is None:
            raise ValueError(f"{quarter} has no fold checkpoints")
        q_seed = np.full(q_sum.shape, np.nan, dtype=np.float32)
        np.divide(q_sum, q_count, out=q_seed, where=q_count > 0)
        outputs[str(quarter)] = {
            "dates": q_dates, "seed_predictions": q_seed,
            "fold_count": len(folds),
            "date_lookup": {str(d): i for i, d in enumerate(q_dates)},
        }
    execution_start = np.flatnonzero(panel.days >= TEST_START)
    if not len(execution_start) or int(execution_start[0]) == 0:
        raise ValueError("cannot determine pre-test signal boundary for score export")
    export_start = int(execution_start[0]) - 1
    prices = getattr(source, "prices", None)
    for signal in range(export_start, len(panel.days)):
        date = str(panel.days[signal])
        execution = signal + 1
        model_quarter = (_latest_model_quarter(panel.days[execution], quarters) if execution < len(panel.days)
                         else _latest_model_quarter(date, quarters))
        if model_quarter not in outputs:
            raise ValueError(f"no eligible quarter model for exported signal date {date}")
        output = outputs[model_quarter]
        score_index = output["date_lookup"].get(date)
        if score_index is not None:
            seed_scores = output["seed_predictions"][:, score_index, :]
        else:
            seed_scores, _ = _rescore_signal(panel, source, prices, date, model_quarter)
        valid = np.isfinite(seed_scores)
        count = valid.sum(axis=0)
        scores = np.full(len(panel.codes), np.nan, dtype=np.float32)
        np.divide(np.nansum(seed_scores, axis=0), count, out=scores, where=count > 0)
        for ci, code in enumerate(panel.codes):
            records.append({"trade_date": date, "stock_code": str(code),
                            "score": float(scores[ci]) if np.isfinite(scores[ci]) else None,
                            "quarter_model": str(model_quarter)})
    return outputs, signatures, records


def _load_test_scores(panel, needed_days, source, prices):
    quarter_scores, signatures, export_records = _load_quarter_scores(panel, source)
    all_seed = np.full((len(SEEDS), len(panel.days), len(panel.codes)), np.nan, dtype=np.float32)
    coverage = np.zeros(len(panel.days), dtype=np.int16)
    for day0 in needed_days:
        signal = int(day0)
        signal_date = str(panel.days[signal])
        execution = signal + 1
        if execution >= len(panel.days):
            raise ValueError(f"missing execution date for signal {signal_date}")
        model_quarter = _latest_model_quarter(panel.days[execution], tuple(quarter_scores))
        if model_quarter not in quarter_scores:
            raise ValueError(f"no quarter model for execution date {panel.days[execution]}")
        q = quarter_scores[model_quarter]
        score_index = q["date_lookup"].get(signal_date)
        if score_index is not None:
            all_seed[:, signal] = q["seed_predictions"][:, score_index, :]
            coverage[signal] = q["fold_count"]
        else:
            # A quarter model may be executable on its first session using the preceding
            # close, which is outside that quarter's saved score axis.
            preds, folds = _rescore_signal(panel, source, prices, signal_date, model_quarter)
            all_seed[:, signal] = preds
            coverage[signal] = folds
    return all_seed, coverage, quarter_scores, signatures, export_records


def _write_score_outputs(panel, quarter_scores, signatures, records):
    import pyarrow as pa
    import pyarrow.parquet as pq

    quarters = tuple(str(q) for q in quarter_scores)
    dates = sorted({str(r["trade_date"]) for r in records})
    unique_pairs = {(str(r["trade_date"]), str(r["stock_code"])) for r in records}
    per_date, quarter_by_date = {}, {}
    for row in records:
        date = str(row["trade_date"])
        per_date.setdefault(date, set()).add(str(row["stock_code"]))
        q = str(row["quarter_model"])
        if date in quarter_by_date and quarter_by_date[date] != q:
            raise ValueError(f"multiple quarter models assigned to signal date {date}")
        quarter_by_date[date] = q
    expected_codes = set(str(x) for x in panel.codes)
    if len(records) != len(dates) * len(expected_codes) or len(unique_pairs) != len(records):
        raise ValueError("score export must have unique (date,code) rows for the full frozen pool")
    execution_start = np.flatnonzero(panel.days >= TEST_START)
    if not len(execution_start) or int(execution_start[0]) == 0:
        raise ValueError("cannot validate score export signal date coverage")
    first_signal = str(panel.days[int(execution_start[0]) - 1])
    if (not dates or dates[0] != first_signal or dates[-1] != str(panel.days[-1])
            or len(dates) != len(panel.days) - int(execution_start[0]) + 1):
        raise ValueError("score export must cover pre-test signal boundary through data-axis end")
    if any(codes != expected_codes for codes in per_date.values()):
        raise ValueError("score export dates do not all contain the full frozen pool")
    schema = pa.schema([
        ("trade_date", pa.string()), ("stock_code", pa.string()),
        ("score", pa.float32()), ("quarter_model", pa.string()),
    ])
    for year, path in SCORE_PARQUET.items():
        selected = [r for r in records if str(r["trade_date"]).startswith(str(year))]
        table = pa.Table.from_pydict({
            "trade_date": [r["trade_date"] for r in selected],
            "stock_code": [r["stock_code"] for r in selected],
            "score": [r["score"] for r in selected],
            "quarter_model": [r["quarter_model"] for r in selected],
        }, schema=schema)
        safe_path = M.output_file(path)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, safe_path)
    meta = {
        "unit": "experiments2/V51",
        "recipe_name": RECIPE["name"],
        "data_built_at": getattr(panel, "data_built_at", None),
        "snapshot": {"data_built_at": getattr(panel, "data_built_at", None),
                     "pool_snapshot_date": RECIPE["pool"]["snapshot_date"]},
        "pool": {"taxonomy": RECIPE["pool"]["taxonomy"],
                 "level1_code": RECIPE["pool"]["level1_code"],
                 "level1_name": RECIPE["pool"]["level1_name"],
                 "level2_code": RECIPE["pool"]["level2_code"],
                 "level2_name": RECIPE["pool"]["level2_name"],
                 "snapshot_date": RECIPE["pool"]["snapshot_date"],
                 "member_count": len(panel.codes), "members": [str(x) for x in panel.codes],
                 "mean_pairwise_correlation": RECIPE["pool"]["mean_pairwise_correlation"],
                 "first_pc_variance_share": RECIPE["pool"]["first_pc_variance_share"]},
        "date_coverage": {"signal_start": dates[0] if dates else None,
                          "signal_end": dates[-1] if dates else None,
                          "date_count": len(dates), "row_count": len(records),
                          "stocks_per_date": len(expected_codes),
                          "quarter_model_by_date": {d: quarter_by_date[d] for d in dates}},
        "columns": ["trade_date", "stock_code", "score", "quarter_model"],
        "years": {
            str(year): {"path": path.relative_to(ROOT).as_posix(),
                        "rows": sum(str(r["trade_date"]).startswith(str(year)) for r in records)}
            for year, path in SCORE_PARQUET.items()
        },
        "quarter_models": {
            q: {"folds": list(signatures[q]), "fold_count": int(output["fold_count"]),
                "score_start": str(output["dates"][0]), "score_end": str(output["dates"][-1]),
                "exported_dates": int(sum(owner == q for owner in quarter_by_date.values()))}
            for q, output in quarter_scores.items()
        },
        "fold_signatures": signatures,
        "aggregation": "same-quarter four-fold average per seed, then four-seed mean",
        "assignment": "for each signal date, use the latest model quarter eligible for its next execution date; rescore quarter-boundary closes; final data-axis date uses the latest model quarter through data end",
        "test_window_used_for_selection": False,
    }
    _write_json(M.output_file(SCORE_META), meta)
    return meta


def _clean_info(dates, payload):
    cfg = CLEAN_CFG
    train_end = _train_end(payload, cfg["quarter"], cfg["fold"])
    if train_end is None:
        raise ValueError("clean block cannot be certified: split.json is missing the train span")
    if train_end >= str(dates[0]):
        raise ValueError(f"purge failure: train_end={train_end}, valid_start={dates[0]}")
    return {"quarter": cfg["quarter"], "fold": cfg["fold"], "start": str(dates[0]),
            "end": str(dates[-1]), "days": len(dates), "train_end": train_end,
            "purge_verified": bool(train_end and train_end < str(dates[0])),
            "alpha_formula": cfg["alpha_formula"]}


def _limit_pct(code, spec):
    override = spec["limit_overrides"]
    if isinstance(override, dict) and str(code) in override:
        return float(override[str(code)])
    prefix = str(code).split(".")[0]
    if prefix.startswith(("300", "301", "688", "689")):
        return 0.20
    if prefix.startswith(("4", "8", "92")):
        return 0.30
    return 0.10


def _can_trade(panel, day, c, side, spec):
    op = float(panel.prices["open"][day, c])
    pre = float(panel.prices["pre_close"][day, c])
    if not (np.isfinite(op) and op > 0 and np.isfinite(pre) and pre > 0):
        return False
    pct = _limit_pct(panel.codes[c], spec)
    up = math.floor(pre * (1 + pct) * 100 + 0.5) / 100
    dn = math.floor(pre * (1 - pct) * 100 + 0.5) / 100
    # At-open checks use only open and prior close, never full-session volume or H/L.
    return op < up - 1e-8 if side == "buy" else op > dn + 1e-8


def _capacity(panel, signal_day, c, price, spec):
    amount, lot = float(panel.amount[signal_day, c]), spec["lot"]
    if not np.isfinite(amount) or amount <= 0 or price <= 0:
        return 0
    return max(0, int(math.floor(amount * spec["participation"] / price / lot)) * lot)


def _rank(score, panel, params):
    good = np.isfinite(score)
    if params["threshold"] in ("score_gt_zero", "rank_and_score_gt_0"):
        good &= score > 0
    ids = np.flatnonzero(good).tolist()
    return sorted(ids, key=lambda c: (-float(score[c]), str(panel.codes[c])))


def _choose(order, held, params):
    k, band = int(params["k"]), int(params["band"])
    names = [int(c) for c in order[:k + band] if int(c) in held][:k]
    for c in order:
        if int(c) not in names:
            names.append(int(c))
        if len(names) >= k:
            break
    return names[:k]


def _sim(panel, signal_days, scores, params, mode="net", hold=False, rebalance=None):
    """Signal-time integer targets at T close; fill at T+1 open; mark at close."""
    signal_days = np.asarray(signal_days, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (len(signal_days), len(panel.codes)) or not len(signal_days):
        raise ValueError("scores need shape [signal days, stocks]")
    if np.any(signal_days + 1 >= len(panel.days)):
        raise ValueError("signal is missing its T+1 execution date")
    spec = _spec()
    mode_kind = MODE_KIND[mode]
    fees_on = mode_kind != "gross"
    slip = spec["slippage"] if mode_kind == "net" else 0.0
    n, lot = len(panel.codes), spec["lot"]
    if rebalance is None:
        rebalance = np.asarray([i % max(1, int(params["period"])) == 0 for i in range(len(signal_days))])
    else:
        rebalance = np.asarray(rebalance, dtype=bool)
    cash, shares = MONEY, np.zeros(n, dtype=float)
    targets, hold_budgets = np.zeros(n), np.zeros(n)
    prev_close, prev_adj, last_mark = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    prev_equity = MONEY
    fees_total = buy_fees = sell_fees = slip_total = 0.0
    buy_commission = sell_commission = transfer_fees = stamp_duty_fees = 0.0
    buy_notional = sell_notional = 0.0
    blocked_entries = blocked_exits = 0
    untradable_entries = untradable_exits = 0
    participation_limited_entries = participation_limited_exits = 0
    cash_limited_entries = lot_limited_entries = lot_limited_exits = 0
    curves, trades = [], []
    matched_growth = 1.0
    op, cl, adj = (panel.prices[x] for x in ("open", "close", "adj_factor"))

    for pos, signal0 in enumerate(signal_days):
        signal = int(signal0)
        execution = signal + 1
        prior_shares = shares.copy()

        # All desired lots are fixed with information observable at the signal close.
        signal_mark = np.where(np.isfinite(cl[signal]) & (cl[signal] > 0), cl[signal], last_mark)
        signal_mark = np.where(np.isfinite(signal_mark) & (signal_mark > 0), signal_mark, 0.0)
        signal_equity = cash + float(np.sum(shares * signal_mark))
        held = set(np.flatnonzero(shares > 1e-8))
        if hold:
            desired, target_exposure = list(range(n)), 1.0
            if pos == 0:
                hold_budgets[:] = signal_equity / max(n, 1)
            for c in range(n):
                if targets[c] <= 0 and hold_budgets[c] > 0 and signal_mark[c] > 0:
                    targets[c] = math.floor(hold_budgets[c] / signal_mark[c] / lot) * lot
        elif rebalance[pos]:
            order = _rank(scores[pos], panel, params)
            desired = _choose(order, held, params)
            target_exposure = float(params["exposure"])
            each = signal_equity * target_exposure / max(len(desired), 1)
            targets[:] = 0.0
            for c in desired:
                if signal_mark[c] > 0:
                    targets[c] = math.floor(each / signal_mark[c] / lot) * lot
        else:
            desired = np.flatnonzero(targets > 0).tolist()
            target_exposure = float(params["exposure"])
        desired_set = set(int(c) for c in desired)
        buy_order = list(desired)
        if not hold and rebalance[pos]:
            buy_order = _rank(scores[pos], panel, params)

        # Corporate actions from T to T+1 adjust both holdings and the T-close target lots.
        if pos:
            ratio = np.divide(adj[execution], adj[execution - 1], out=np.ones(n),
                              where=np.isfinite(adj[execution]) & (adj[execution] > 0)
                              & np.isfinite(adj[execution - 1]) & (adj[execution - 1] > 0))
            shares *= ratio
            targets *= ratio

        if pos:
            overnight = np.divide(op[execution] * adj[execution], prev_close * prev_adj,
                                  out=np.ones(n),
                                  where=np.isfinite(op[execution]) & (op[execution] > 0)
                                  & np.isfinite(adj[execution]) & (adj[execution] > 0)
                                  & np.isfinite(prev_close) & (prev_close > 0)
                                  & np.isfinite(prev_adj) & (prev_adj > 0)) - 1.0
            valid = np.isfinite(overnight)
            bh_overnight = float(np.mean(overnight[valid])) if valid.any() else 0.0
            prior_mv = np.nan_to_num(prior_shares * prev_close * prev_adj, nan=0.0)
            prior_exposure = float(prior_mv.sum() / max(prev_equity, 1e-12))
        else:
            bh_overnight, prior_exposure = 0.0, 0.0
        matched_overnight = prior_exposure * bh_overnight
        open_mark = np.where(np.isfinite(op[execution]) & (op[execution] > 0), op[execution], last_mark)
        open_mark = np.where(np.isfinite(open_mark) & (open_mark > 0), open_mark, 0.0)

        # Sell reductions and exits before buys. "blocked" counts only open tradability
        # or participation restrictions; lot rounding and cash limits have separate counts.
        for c in np.flatnonzero(shares > 1e-8):
            want = float(targets[c]) if int(c) in desired_set else 0.0
            delta = float(shares[c] - want)
            if delta <= 1e-7:
                continue
            full_exit = want <= 1e-8
            if not full_exit and delta < lot - 1e-7:
                lot_limited_exits += 1
                continue
            if not _can_trade(panel, execution, int(c), "sell", spec):
                blocked_exits += 1
                untradable_exits += 1
                continue
            px = float(op[execution, c])
            request = shares[c] if full_exit else math.floor(delta / lot) * lot
            if request <= 0:
                lot_limited_exits += 1
                continue
            cap = _capacity(panel, signal, int(c), px, spec)
            qty = (shares[c] if full_exit and cap >= shares[c] - 1e-7
                   else min(math.floor(request / lot) * lot, cap))
            if qty <= 0:
                blocked_exits += 1
                participation_limited_exits += 1
                continue
            capacity_short = qty + 1e-7 < request
            fill = px * (1.0 - slip)
            notional = qty * fill
            fee_parts = (_fee_components(notional, "sell", spec) if fees_on else
                         {"commission": 0.0, "transfer": 0.0, "stamp_duty": 0.0})
            fee = float(sum(fee_parts.values()))
            cash += notional - fee
            shares[c] -= qty
            if shares[c] < 1e-8:
                shares[c] = 0.0
            fees_total += fee
            sell_fees += fee
            sell_commission += fee_parts["commission"]
            transfer_fees += fee_parts["transfer"]
            stamp_duty_fees += fee_parts["stamp_duty"]
            sell_notional += notional
            slip_total += qty * (px - fill)
            trades.append({"signal_date": str(panel.days[signal]), "date": str(panel.days[execution]),
                           "code": str(panel.codes[c]), "side": "sell", "quantity": qty,
                           "open_price": px, "fill_price": fill, "notional": notional,
                           "fees": fee, "target_exposure": target_exposure, "cost_mode": mode})
            if capacity_short:
                blocked_exits += 1
                participation_limited_exits += 1

        for c0 in buy_order:
            c = int(c0)
            if c not in desired_set:
                continue
            delta = float(targets[c] - shares[c])
            if delta <= 1e-7:
                continue
            if delta < lot - 1e-7:
                lot_limited_entries += 1
                continue
            if not _can_trade(panel, execution, c, "buy", spec):
                blocked_entries += 1
                untradable_entries += 1
                continue
            px = float(op[execution, c])
            request = math.floor(delta / lot) * lot
            cap_qty = min(request, _capacity(panel, signal, c, px, spec))
            capacity_short = cap_qty < request
            if capacity_short:
                blocked_entries += 1
                participation_limited_entries += 1
            if cap_qty <= 0:
                continue
            qty = cap_qty
            while qty > 0:
                notional = qty * px * (1.0 + slip)
                fee_parts = (_fee_components(notional, "buy", spec) if fees_on else
                             {"commission": 0.0, "transfer": 0.0, "stamp_duty": 0.0})
                fee = float(sum(fee_parts.values()))
                if notional + fee <= cash + 1e-8:
                    break
                qty -= lot
            if qty <= 0:
                cash_limited_entries += 1
                continue
            if qty < cap_qty:
                cash_limited_entries += 1
            fill = px * (1.0 + slip)
            notional = qty * fill
            fee_parts = (_fee_components(notional, "buy", spec) if fees_on else
                         {"commission": 0.0, "transfer": 0.0, "stamp_duty": 0.0})
            fee = float(sum(fee_parts.values()))
            cash -= notional + fee
            shares[c] += qty
            fees_total += fee
            buy_fees += fee
            buy_commission += fee_parts["commission"]
            transfer_fees += fee_parts["transfer"]
            stamp_duty_fees += fee_parts["stamp_duty"]
            buy_notional += notional
            slip_total += qty * (fill - px)
            trades.append({"signal_date": str(panel.days[signal]), "date": str(panel.days[execution]),
                           "code": str(panel.codes[c]), "side": "buy", "quantity": qty,
                           "open_price": px, "fill_price": fill, "notional": notional,
                           "fees": fee, "target_exposure": target_exposure, "cost_mode": mode})

        close_mark = np.where(np.isfinite(cl[execution]) & (cl[execution] > 0), cl[execution],
                              np.where(open_mark > 0, open_mark, last_mark))
        close_mark = np.where(np.isfinite(close_mark) & (close_mark > 0), close_mark, 0.0)
        open_equity = cash + float(np.sum(shares * open_mark))
        open_exposure = float(np.sum(shares * open_mark) / max(open_equity, 1e-12))
        intraday = np.divide(cl[execution], op[execution], out=np.ones(n),
                             where=np.isfinite(cl[execution]) & (cl[execution] > 0)
                             & np.isfinite(op[execution]) & (op[execution] > 0)) - 1.0
        valid = np.isfinite(intraday)
        bh_intraday = float(np.mean(intraday[valid])) if valid.any() else 0.0
        matched_day = (1 + matched_overnight) * (1 + open_exposure * bh_intraday) - 1
        matched_growth *= 1 + matched_day
        equity = cash + float(np.sum(shares * close_mark))
        daily_return = equity / max(prev_equity, 1e-12) - 1
        exposure = float(np.sum(shares * close_mark) / max(equity, 1e-12))
        curves.append({"signal_date": str(panel.days[signal]), "date": str(panel.days[execution]),
                       "cash": cash, "equity": equity, "daily_return": daily_return,
                       "exposure": exposure, "shares": shares.tolist(), "target_shares": targets.tolist(),
                       "stock_equity": (shares * close_mark).tolist(),
                       "stock_exposure": (shares * close_mark / max(equity, 1e-12)).tolist(),
                       "matched_daily_return": matched_day})
        prev_equity, prev_close, prev_adj, last_mark = equity, close_mark.copy(), adj[execution].copy(), close_mark.copy()

    values = np.asarray([MONEY] + [x["equity"] for x in curves], dtype=float)
    mdd = float(np.min(values / np.maximum.accumulate(values) - 1))
    ret = float(curves[-1]["equity"] / MONEY - 1)
    buy_value = sum(x["notional"] for x in trades if x["side"] == "buy")
    sell_value = sum(x["notional"] for x in trades if x["side"] == "sell")
    ending_curve = curves[-1]
    ending_stock_value = float(sum(ending_curve["stock_equity"]))
    ending_balance_error = abs(float(ending_curve["equity"])
                               - float(ending_curve["cash"]) - ending_stock_value)
    balance_error = max(abs(float(row["equity"]) - float(row["cash"])
                            - float(sum(row["stock_equity"]))) for row in curves)
    fee_components = buy_commission + sell_commission + transfer_fees + stamp_duty_fees
    stats = {
        "return_value": ret, "total_fees": fees_total, "buy_fees": buy_fees, "sell_fees": sell_fees,
        "buy_commission": buy_commission, "sell_commission": sell_commission,
        "transfer_fees": transfer_fees, "stamp_duty_fees": stamp_duty_fees,
        "fee_component_balance_error": abs(fees_total - fee_components),
        "total_slippage": slip_total, "total_cost_including_slippage": fees_total + slip_total,
        "buy_notional": buy_value, "sell_notional": sell_value,
        "fee_rate_buy_bps": buy_fees / buy_value * 10000 if buy_value else 0.0,
        "fee_rate_sell_bps": sell_fees / sell_value * 10000 if sell_value else 0.0,
        "trades": len(trades), "avg_exposure": float(np.mean([x["exposure"] for x in curves])),
        "max_drawdown": mdd, "blocked_entries": blocked_entries, "blocked_exits": blocked_exits,
        "untradable_entries": untradable_entries, "untradable_exits": untradable_exits,
        "participation_limited_entries": participation_limited_entries,
        "participation_limited_exits": participation_limited_exits,
        "cash_limited_entries": cash_limited_entries, "lot_limited_entries": lot_limited_entries,
        "lot_limited_exits": lot_limited_exits,
        "ending_positions": int(np.count_nonzero(shares > 1e-7)), "ending_cash": float(cash),
        "ending_stock_value": ending_stock_value, "ending_equity": float(ending_curve["equity"]),
        "ending_balance_sheet_error": ending_balance_error,
        "balance_sheet_max_abs_error": balance_error,
        "min_cash": min(float(x["cash"]) for x in curves),
        "max_exposure": max(float(x["exposure"]) for x in curves),
        "risk_weight_match_diagnostic": float(matched_growth - 1),
        "days": len(curves), "cost_mode": mode,
    }
    return stats, curves, trades


def _alpha(stats, ew_return):
    return float(stats["return_value"] - stats["avg_exposure"] * ew_return)


def _clean_grid(panel, days, seed_preds):
    segment_count = int(CLEAN_CFG["contiguous_segments"])
    segments = np.array_split(days, segment_count)
    if any(not len(x) for x in segments):
        raise ValueError("empty clean validation segment")
    ens = np.nanmean(seed_preds, axis=0)
    ew_params = {"k": len(panel.codes), "exposure": 1.0, "threshold": "rank_only", "band": 0, "period": 1}
    ew = []
    for seg in segments:
        local = ens[np.searchsorted(days, seg)]
        ew.append(_sim(panel, seg, local, ew_params, mode="net", hold=True)[0]["return_value"])
    rows = []
    for prm in CANDIDATES:
        seg_rows, by_seed = [], [[] for _ in SEEDS]
        for j, seg in enumerate(segments):
            local_by_seed = seed_preds[:, np.searchsorted(days, seg), :]
            seed_stats = []
            for si in range(len(SEEDS)):
                st, _, _ = _sim(panel, seg, local_by_seed[si], prm, mode="net")
                a = _alpha(st, ew[j])
                by_seed[si].append(a)
                seed_stats.append({"seed": SEEDS[si], "net": st["return_value"],
                                   "avg_exposure": st["avg_exposure"], "alpha": a,
                                   "fees": st["total_fees"], "trades": st["trades"],
                                   "blocked_entries": st["blocked_entries"],
                                   "blocked_exits": st["blocked_exits"]})
            means = np.asarray([x["alpha"] for x in seed_stats])
            seg_rows.append({"segment": j + 1, "signal_start": str(panel.days[seg[0]]),
                             "signal_end": str(panel.days[seg[-1]]),
                             "execution_start": str(panel.days[seg[0] + 1]),
                             "execution_end": str(panel.days[seg[-1] + 1]),
                             "ew_net": ew[j], "seed_metrics": seed_stats,
                             "mean_alpha": float(means.mean()), "positive_seeds": int(np.sum(means > 0))})
        segment_alpha = np.asarray([x["mean_alpha"] for x in seg_rows])
        seed_alpha = np.asarray([np.mean(x) for x in by_seed])
        mean = float(np.mean(segment_alpha))
        sd = float(np.std(segment_alpha, ddof=1)) if len(segment_alpha) > 1 else 0.0
        positive_segments = int(np.sum(segment_alpha > 0))
        positive_seeds = int(np.sum(seed_alpha > 0))
        eligible = (positive_segments >= int(CLEAN_CFG["minimum_positive_segments"])
                    and positive_seeds >= int(CLEAN_CFG["minimum_positive_seeds"]))
        rows.append({**prm, "segments": seg_rows, "mean_alpha": mean, "segment_sd": sd,
                     "seed_mean_alpha": seed_alpha.tolist(), "positive_segments": positive_segments,
                     "positive_seeds": positive_seeds, "eligible": eligible,
                     "selection_score": mean - float(CLEAN_CFG.get("segment_dispersion_penalty", 0.5)) * sd,
                     "total_trades": int(sum(x["trades"] for s in seg_rows for x in s["seed_metrics"]))})
    eligible_rows = [x for x in rows if x["eligible"]]
    pool = eligible_rows if eligible_rows else rows
    pool.sort(key=lambda x: (-x["selection_score"], x["segment_sd"], x["total_trades"],
                             -x["k"], -x["exposure"], x["threshold"], x["band"], x["period"]))
    chosen = {key: pool[0][key] for key in ("k", "exposure", "threshold", "band", "period")}
    return chosen, rows, bool(eligible_rows)


def _metric(params, stat, ew_net=None):
    result = {**params, **stat}
    result["alpha"] = _alpha(stat, ew_net) if ew_net is not None else None
    return result


def _limit_rank_sign(score, panel, threshold):
    p = {"threshold": threshold}
    return _rank(score, panel, p)


def _signal_metrics(panel, scores, signals, executions, params):
    label = RECIPE.get("score_label", next(iter(panel.Y)))
    if label not in panel.Y:
        label = next(iter(panel.Y))
    y = panel.Y[label]
    ideal, can_open, ic, ric = [], [], [], []
    k = int(params["k"])
    for pos, (signal, execution) in enumerate(zip(signals, executions)):
        score = scores[pos]
        good = np.isfinite(score) & np.isfinite(y[int(signal)])
        if good.sum() >= 3 and np.std(score[good]) > 1e-12 and np.std(y[int(signal), good]) > 1e-12:
            ic.append(float(np.corrcoef(score[good], y[int(signal), good])[0, 1]))
            ric.append(float(np.corrcoef(_rank_values(score[good]), _rank_values(y[int(signal), good]))[0, 1]))
        order = _limit_rank_sign(score, panel, params["threshold"])
        names = order[:k]
        vals = y[int(signal), names] if names else np.asarray([])
        if len(vals) and np.isfinite(vals).any():
            ideal.append(float(np.nanmean(vals)))
        tradable = [c for c in names if _can_trade(panel, int(execution), int(c), "buy", _spec())]
        tv = y[int(signal), tradable] if tradable else np.asarray([])
        if len(tv) and np.isfinite(tv).any():
            can_open.append(float(np.nanmean(tv)))
    return {"label": label, "ideal_signal_mean_label_return": float(np.mean(ideal)) if ideal else None,
            "ideal_signal_days": len(ideal),
            "open_tradable_mean_label_return": float(np.mean(can_open)) if can_open else None,
            "open_tradable_days": len(can_open),
            "daily_pearson_ic": float(np.mean(ic)) if ic else None,
            "daily_rank_ic": float(np.mean(ric)) if ric else None, "ic_days": len(ic)}


def _rank_values(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and x[order[j]] == x[order[i]]:
            j += 1
        r[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return r


def _bank_panel(days, codes):
    import pandas as pd
    import pyarrow.parquet as pq

    root = Path(M.data_root())
    prices, amounts = [], []
    # The presentation starts in 2025; 2024 supplies the 60-session warm-up.
    for year in ("2024", "2025", "2026"):
        pp, ap = root / "prices" / f"year={year}" / "data.parquet", root / "amount" / f"year={year}" / "data.parquet"
        if not pp.is_file() or not ap.is_file():
            raise FileNotFoundError(f"bank reference price/amount partition missing: {year}")
        pcols = pq.ParquetFile(pp).schema_arrow.names
        pcols = ["trade_date", "stock_code"] + [x for x in M.PRICE_COLUMNS if x in pcols]
        prices.append(pq.read_table(pp, columns=pcols,
                      filters=[("stock_code", "in", list(codes))]).to_pandas())
        acols = pq.ParquetFile(ap).schema_arrow.names
        amount_name = next(x for x in acols if x not in ("trade_date", "stock_code"))
        amounts.append(pq.read_table(ap, columns=["trade_date", "stock_code", amount_name],
                       filters=[("stock_code", "in", list(codes))]).to_pandas()
                       .rename(columns={amount_name: "amount"}))
    df = pd.concat(prices, ignore_index=True).merge(pd.concat(amounts, ignore_index=True),
                                                    on=["trade_date", "stock_code"], how="left")
    day_ix, code_ix = {str(d): i for i, d in enumerate(days)}, {str(c): i for i, c in enumerate(codes)}
    p = type("BenchmarkPanel", (), {})()
    p.days, p.codes = np.asarray(days, dtype=str), np.asarray(codes, dtype=str)
    p.amount = np.zeros((len(days), len(codes)), dtype=float)
    p.prices = {name: np.full((len(days), len(codes)), np.nan)
                for name in ("open", "close", "pre_close", "vol", "adj_factor")}
    for row in df.itertuples(index=False):
        d, c = str(row.trade_date)[:10], str(row.stock_code)
        if d not in day_ix or c not in code_ix:
            continue
        i, j = day_ix[d], code_ix[c]
        for name in p.prices:
            value = getattr(row, name)
            p.prices[name][i, j] = float(value) if pd.notna(value) else np.nan
        p.amount[i, j] = float(row.amount) if pd.notna(row.amount) else 0.0
    for j in range(len(codes)):
        good = np.flatnonzero(np.isfinite(p.prices["adj_factor"][:, j]) & (p.prices["adj_factor"][:, j] > 0))
        if len(good):
            p.prices["adj_factor"][:good[0], j] = p.prices["adj_factor"][good[0], j]
            for i in range(good[0] + 1, len(days)):
                if not np.isfinite(p.prices["adj_factor"][i, j]):
                    p.prices["adj_factor"][i, j] = p.prices["adj_factor"][i - 1, j]
    return p


def _bank_scores(panel, signal_days):
    momentum = np.full((len(signal_days), len(panel.codes)), np.nan, dtype=np.float32)
    adj_close = panel.prices["close"] * panel.prices["adj_factor"]
    mask = np.zeros(len(signal_days), dtype=bool)
    for pos, day0 in enumerate(signal_days):
        signal = int(day0)
        execution = signal + 1
        if execution >= len(panel.days):
            continue
        month = str(panel.days[execution])[:7]
        first_execution_day = (execution == 0
                               or str(panel.days[execution - 1])[:7] != month)
        mask[pos] = first_execution_day and signal >= 60
        if not mask[pos]:
            continue
        old, now = adj_close[signal - 60], adj_close[signal]
        good = np.isfinite(old) & (old > 0) & np.isfinite(now) & (now > 0)
        momentum[pos, good] = (now[good] / old[good] - 1).astype(np.float32)
    return momentum, mask


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            clean = {}
            for key in fields:
                value = row.get(key)
                if isinstance(value, (list, tuple, dict, np.ndarray)):
                    value = json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))
                elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    value = ""
                clean[key] = value
            w.writerow(clean)


def _curve_rows(curve, codes, params, seed, mode, strategy):
    out = []
    for row in curve:
        for i, code in enumerate(codes):
            out.append({"strategy": strategy, "seed": seed, "cost_mode": mode, **params,
                        "signal_date": row["signal_date"], "trade_date": row["date"], "stock_code": code,
                        "shares": row["shares"][i], "target_shares": row["target_shares"][i],
                        "stock_equity": row["stock_equity"][i], "stock_exposure": row["stock_exposure"][i],
                        "cash": row["cash"], "portfolio_equity": row["equity"],
                        "portfolio_exposure": row["exposure"], "daily_return": row["daily_return"],
                        "matched_daily_return": row["matched_daily_return"]})
    return out


def _trade_rows(trades, params, seed, strategy):
    return [{**row, **params, "strategy": strategy, "seed": seed} for row in trades]


def _plot(curves, bh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    for mode, rows in curves.items():
        ax.plot([x["date"] for x in rows], np.asarray([x["equity"] for x in rows]) / MONEY, label=mode)
    ax.plot([x["date"] for x in bh], np.asarray([x["equity"] for x in bh]) / MONEY,
            label="same-pool EW buy & hold (net)")
    ax.set_title("V51 model-ranked shared-cash portfolio")
    ax.set_ylabel("Equity / initial cash")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG)
    plt.close(fig)


def _pct(x):
    return "—" if x is None else f"{float(x):+.2%}"


def _write_md(audit):
    selected, params = audit["selected_test"]["ensemble"], audit["selected_params"]
    lines = ["# experiments2/V51 模型排序策略", "",
             f"- 冻结股票池：{', '.join(audit['universe'])}",
             f"- 池体检：均值相关={audit['pool_health']['mean_pairwise_correlation']:.3f}，"
             f"PC1={audit['pool_health']['first_pc_variance_share']:.3f}；PC1<0.6，采用截面IC/RankIC/TopK。",
             (("- 结论：否。无候选达到干净块稳定性门槛；所列参数只是描述性回退，不称为成功策略。")
              if audit['selection']['fallback_params_descriptive_only'] else
              "- 结论：有候选通过预注册干净块稳定性门槛；测试窗结果仅作冻结参数展示。"),
             f"- 展示窗执行日：{TEST_START} 至 {TEST_END}，{audit['window']['execution_days']}日。",
             "- T日收盘打分、T+1开盘成交、每日收盘估值；期末保留持仓并按2026-06-30收盘价计值。",
             "- 选参只用2025Q3 fold4干净验证块四段。",
             f"- 冻结参数：TopK={params['k']}，目标敞口={params['exposure']:.0%}，"
             f"阈值={params['threshold']}，缓冲={params['band']}，调仓={params['period']}日。",
             "", "| 口径 | 毛收益 | 费用 | 净收益 | 平均敞口 | 主alpha | MDD | 成交 | 拒买/拒卖 | 买/卖费用bp |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode in MODES:
        key = MODE_KIND[mode]
        r = selected[key]
        lines.append(f"| {mode} | {_pct(selected['gross']['return_value'])} | {r['total_fees']:.0f} "
                     f"| {_pct(r['return_value'])} | {r['avg_exposure']:.1%} | {_pct(r['alpha'])} "
                     f"| {_pct(r['max_drawdown'])} | {r['trades']} | {r['blocked_entries']}/{r['blocked_exits']} "
                     f"| {r['fee_rate_buy_bps']:.2f}/{r['fee_rate_sell_bps']:.2f} |")
    lines.extend(["", "## 同窗同日基准", "",
                  "| 基准 | 口径 | 毛收益 | 费用 | 净收益 | 平均敞口 | MDD | 成交 | 拒买/拒卖 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name, modes in audit["benchmarks"].items():
        for mode in MODES:
            r = modes[MODE_KIND[mode]]
            lines.append(f"| {name} | {mode} | {_pct(modes['gross']['return_value'])} "
                         f"| {r['total_fees']:.0f} | {_pct(r['return_value'])} "
                         f"| {r['avg_exposure']:.1%} | {_pct(r['max_drawdown'])} "
                         f"| {r['trades']} | {r['blocked_entries']}/{r['blocked_exits']} |")
    lines.extend(["", "## 四种子分布（冻结参数净口径）", "",
                  "| seed | 净收益 | 费用 | 平均敞口 | 主alpha | MDD | 成交 |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for r in audit["selected_test"]["seed_distribution"]:
        lines.append(f"| {r['seed']} | {_pct(r['return_value'])} | {r['total_fees']:.0f} "
                     f"| {r['avg_exposure']:.1%} | {_pct(r['alpha'])} | {_pct(r['max_drawdown'])} | {r['trades']} |")
    lines.extend(["", "## 目标敞口费用敏感性", "",
                  "冻结 TopK、阈值、缓冲和调仓周期，仅将目标敞口设为 RECIPE 的各档。",
                  "gross 为零费用/零滑点；fees_only 计佣金、过户、印花；net 再计滑点。",
                  "| 目标敞口 | 口径 | 毛收益 | 该口径收益 | 买佣金 | 卖佣金 | 过户费 | 印花税 | 滑点 | 总费用 | 总成本 | 买/卖费率bp |",
                  "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for item in audit["target_exposure_fee_sensitivity"]:
        for mode in ("gross", "fees_only", "net"):
            r = item["modes"][mode]
            lines.append(f"| {item['target_exposure']:.0%} | {mode} "
                         f"| {_pct(r['gross_return'])} | {_pct(r['return_value'])} "
                         f"| {r['buy_commission']:.0f} | {r['sell_commission']:.0f} "
                         f"| {r['transfer_fees']:.0f} | {r['stamp_duty_fees']:.0f} "
                         f"| {r['total_slippage']:.0f} | {r['total_fees']:.0f} "
                         f"| {r['total_cost_including_slippage']:.0f} "
                         f"| {r['fee_rate_buy_bps']:.2f}/{r['fee_rate_sell_bps']:.2f} |")
    lines.extend(["", "理想信号、开盘可成交信号、现金账户结果分开保存在 final_audit.json。",
                  "blocked_entries/exits只计开盘可成交或参与率限制；现金不足、整手限制分列。",
                  "完整费用分项、干净块与242日参数敏感性网格也保存在 final_audit.json。", ""])
    MD.parent.mkdir(parents=True, exist_ok=True)
    MD.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(*, audit=False):
    source = M.Panel(load_x=True)
    prices = M.Prices(source)
    source.prices = prices
    panel = _make_panel(source)
    clean_dates, clean_days, clean_seed, clean_payload = _load_clean(panel, source)
    clean_info = _clean_info(clean_dates, clean_payload)
    chosen, clean_grid, eligible_exists = _clean_grid(panel, clean_days, clean_seed)
    chosen_row = next(x for x in clean_grid if all(x[k] == chosen[k] for k in chosen))

    test_exec = np.flatnonzero((panel.days >= TEST_START) & (panel.days <= TEST_END))
    expected = int(RECIPE.get("test_sessions", 242))
    if len(test_exec) != expected:
        raise ValueError(f"expected {expected} presentation sessions; got {len(test_exec)}")
    test_signal = test_exec - 1
    if test_signal[0] < 0:
        raise ValueError("missing T-1 signal session")
    all_seed, coverage, quarter_scores, signatures, export_records = _load_test_scores(panel, test_signal, source, prices)
    missing = [str(panel.days[d]) for d in test_signal if coverage[d] == 0]
    if missing:
        raise ValueError(f"missing OOS scores for T-1 signal dates: {missing[:8]}")
    test_seed = all_seed[:, test_signal]
    ensemble = np.nanmean(test_seed, axis=0)

    hold_params = {"k": len(panel.codes), "exposure": 1.0, "threshold": "rank_only", "band": 0, "period": 1}
    bank_codes = tuple(str(x) for x in REF_CFG["four_bank_codes"])
    bank = _bank_panel(panel.days, bank_codes)
    bank_score, month_mask = _bank_scores(bank, test_signal)
    bank_params1 = {"k": 1, "exposure": 1.0, "threshold": "rank_only", "band": 0, "period": 1}
    bank_params2 = {"k": 2, "exposure": 1.0, "threshold": "rank_only", "band": 0, "period": 1}
    benchmark_defs = {
        "same_pool_equal_weight": (panel, ensemble, hold_params, None, True),
        "four_bank_equal_weight": (bank, np.full((len(test_signal), len(bank_codes)), np.nan),
                                   hold_params, None, True),
        "four_bank_momentum_top1": (bank, bank_score, bank_params1, month_mask, False),
        "four_bank_momentum_top2": (bank, bank_score, bank_params2, month_mask, False),
    }
    bench_stats, bench_curves, bench_trades = {}, {}, {}
    for name, (bp, score, params, mask, is_hold) in benchmark_defs.items():
        modes, curves, trades = {}, {}, {}
        for mode in MODES:
            kind = MODE_KIND[mode]
            stat, curve, tx = _sim(bp, test_signal, score, params, mode=mode,
                                   hold=is_hold, rebalance=mask)
            modes[kind], curves[kind], trades[kind] = stat, curve, tx
        for kind in ("gross", "fees_only", "net"):
            modes[kind]["gross_return"] = modes["gross"]["return_value"]
            modes[kind]["cost_drag_from_gross"] = modes["gross"]["return_value"] - modes[kind]["return_value"]
        bench_stats[name], bench_curves[name], bench_trades[name] = modes, curves, trades

    ew_by_mode = {kind: bench_stats["same_pool_equal_weight"][kind]["return_value"]
                  for kind in ("gross", "fees_only", "net")}
    ew_net = ew_by_mode["net"]
    selected_stats, selected_curves, selected_trades = {}, {}, {}
    for mode in MODES:
        kind = MODE_KIND[mode]
        st, curve, tx = _sim(panel, test_signal, ensemble, chosen, mode=mode)
        st["gross_return"] = None
        st["alpha"] = _alpha(st, ew_by_mode[kind])
        st["risk_weight_match_diagnostic"] = st.pop("risk_weight_match_diagnostic")
        selected_stats[kind], selected_curves[kind], selected_trades[kind] = st, curve, tx
    gross_net = selected_stats["gross"]["return_value"]
    for kind in ("gross", "fees_only", "net"):
        selected_stats[kind]["gross_return"] = gross_net
        selected_stats[kind]["alpha"] = _alpha(selected_stats[kind], ew_by_mode[kind])
        selected_stats[kind]["cost_drag_from_gross"] = gross_net - selected_stats[kind]["return_value"]
        selected_stats[kind]["fee_drag"] = selected_stats["fees_only"]["return_value"] - selected_stats[kind]["return_value"]

    seeds = []
    cash_rows, trade_rows = [], []
    for si, seed in enumerate(SEEDS):
        st, curve, tx = _sim(panel, test_signal, test_seed[si], chosen, mode="net")
        st["alpha"] = _alpha(st, ew_net)
        seeds.append({"seed": seed, **st})
        cash_rows.extend(_curve_rows(curve, panel.codes, chosen, seed, "net", "selected_seed"))
        trade_rows.extend(_trade_rows(tx, chosen, seed, "selected_seed"))
    for kind in ("gross", "fees_only", "net"):
        cash_rows.extend(_curve_rows(selected_curves[kind], panel.codes, chosen, "ensemble", kind, "selected"))
        trade_rows.extend(_trade_rows(selected_trades[kind], chosen, "ensemble", "selected"))

    # Add fresh, same-date, same-account bank baselines and the same-pool EW comparator.
    for name, (bp, _, _, _, _) in benchmark_defs.items():
        for kind in ("gross", "fees_only", "net"):
            cash_rows.extend(_curve_rows(bench_curves[name][kind], bp.codes, {}, name, kind, name))
            trade_rows.extend(_trade_rows(bench_trades[name][kind], {}, name, name))
    hold_cash = [x for x in cash_rows if x["strategy"] in ("same_pool_equal_weight", "four_bank_equal_weight")]
    hold_trade = [x for x in trade_rows if x["strategy"] in ("same_pool_equal_weight", "four_bank_equal_weight")]

    test_grid = []
    for params in CANDIDATES:
        st, _, _ = _sim(panel, test_signal, ensemble, params, mode="net")
        st["alpha"] = _alpha(st, ew_net)
        test_grid.append({**params, **st})

    fee_sensitivity = []
    for exposure in sorted(set(float(x) for x in SEARCH["target_exposure"])):
        sensitivity_params = {**chosen, "exposure": exposure}
        mode_stats = {}
        for mode in MODES:
            kind = MODE_KIND[mode]
            st, _, _ = _sim(panel, test_signal, ensemble, sensitivity_params, mode=mode)
            st["alpha"] = _alpha(st, ew_by_mode[kind])
            mode_stats[kind] = st
        gross_return = mode_stats["gross"]["return_value"]
        for st in mode_stats.values():
            st["gross_return"] = gross_return
            st["cost_drag_from_gross"] = gross_return - st["return_value"]
        fee_sensitivity.append({"target_exposure": exposure, "params": sensitivity_params,
                                "modes": mode_stats})

    ideal = _signal_metrics(panel, ensemble, test_signal, test_exec, chosen)
    signal_diag = {
        "ideal_signal": ideal["ideal_signal_mean_label_return"],
        "open_tradable_signal": ideal["open_tradable_mean_label_return"],
        "cash_account": selected_stats["net"]["return_value"],
        "label": ideal["label"], "ideal_days": ideal["ideal_signal_days"],
        "open_tradable_days": ideal["open_tradable_days"],
        "pearson_ic": ideal["daily_pearson_ic"], "rank_ic": ideal["daily_rank_ic"],
        "ic_days": ideal["ic_days"],
    }

    audit_doc = {
        "unit": "experiments2/V51", "data_built_at": getattr(source, "data_built_at", None),
        "universe": list(panel.codes), "account_money": MONEY,
        "pool_health": {"mean_pairwise_correlation": float(RECIPE["pool"]["mean_pairwise_correlation"]),
                        "first_pc_variance_share": float(RECIPE["pool"]["first_pc_variance_share"]),
                        "signal_metrics": "first PC < 0.6; use cross-sectional IC, RankIC and TopK"},
        "window": {"start": TEST_START, "end": TEST_END, "execution_days": len(test_exec),
                   "execution_start": str(panel.days[test_exec[0]]),
                   "execution_end": str(panel.days[test_exec[-1]]),
                   "signal_start": str(panel.days[test_signal[0]]),
                   "signal_end": str(panel.days[test_signal[-1]]),
                   "end_valuation": "2026-06-30 close mark; retain ending holdings; no final-day liquidation"},
        "selection": {
            "clean_block": clean_info, "segments": chosen_row["segments"], "chosen_params": chosen,
            "chosen_mean_alpha": chosen_row["mean_alpha"], "chosen_segment_sd": chosen_row["segment_sd"],
            "chosen_positive_segments": chosen_row["positive_segments"],
            "chosen_positive_seeds": chosen_row["positive_seeds"],
            "meets_clean_stability_rule": chosen_row["eligible"],
            "any_candidate_meets_stability_rule": eligible_exists,
            "conclusion": ("有候选达到预注册干净块稳定性门槛" if eligible_exists else "否：无候选达到预注册干净块稳定性门槛"),
            "fallback_params_descriptive_only": not eligible_exists,
            "selection_rule": CLEAN_CFG["alpha_formula"] + "; maximize mean segment alpha - 0.5*segment SD",
            "candidate_count": len(clean_grid)},
        "selected_params": chosen,
        "selected_test": {"ensemble": selected_stats, "seed_distribution": seeds,
                          "seed_mean_net": float(np.mean([x["return_value"] for x in seeds])),
                          "seed_sd_net": float(np.std([x["return_value"] for x in seeds], ddof=1)),
                          "seed_mean_alpha": float(np.mean([x["alpha"] for x in seeds])),
                          "seed_sd_alpha": float(np.std([x["alpha"] for x in seeds], ddof=1))},
        "same_pool_equal_weight_buy_hold": bench_stats["same_pool_equal_weight"],
        "benchmarks": {k: bench_stats[k] for k in (
            "four_bank_momentum_top1", "four_bank_momentum_top2", "four_bank_equal_weight")},
        "reference_net_return_anchors_242d": REF_CFG["expected_net_return_242d"],
        "reference_comparison_method": (
            "Four-bank Top1/Top2 and equal-weight are freshly recomputed on the same 242 execution "
            "dates, with the shared 100k account, T+1 open fills, configured fees, and 2026-06-30 "
            "close valuation. The published net returns are anchors only; any gap can reflect the "
            "explicit end-of-window mark-to-close treatment and account execution assumptions."
        ),
        "reference_long_window_anchors": REF_CFG["long_window_net_return_2024_01_to_2026_06"],
        "three_evaluation_views": signal_diag,
        "clean_parameter_grid": clean_grid,
        "test_parameter_sensitivity_net": test_grid,
        "target_exposure_fee_sensitivity": fee_sensitivity,
        "oos_quarter_model_assignment": {
            "rule": "execution-date quarter owns T-1 signal; boundary signals are rescored with that quarter model",
            "signal_dates": int(len(test_signal)),
            "quarter_models": {q: {"fold_signatures": sig, "fold_count": len(sig)}
                               for q, sig in signatures.items()},
        },
        "oos_seed_prediction_fold_coverage": {
            "minimum": int(np.min(coverage[test_signal])), "maximum": int(np.max(coverage[test_signal]))},
        "execution": {"signal_time": EXEC_CFG["signal_time"], "execution_time": EXEC_CFG["execution_time"],
                      "mark_time": EXEC_CFG["mark_time"], "lot_size": EXEC_CFG["lot_size"],
                      "max_participation": EXEC_CFG["max_participation"],
                      "fees": {"commission": EXEC_CFG["commission_rate"],
                               "minimum_commission": EXEC_CFG["min_commission"],
                               "transfer": EXEC_CFG["transfer_rate"],
                               "sell_stamp": EXEC_CFG["stamp_sell_rate"],
                               "slippage": EXEC_CFG["slippage_rate"]},
                      "limit_rule": "conservative opening-price-only check; no execution-session high/low"},
        "test_window_used_for_parameter_selection": False,
    }
    validation = {
        "unit": "experiments2/V51", "clean_block": clean_info, "selected_params": chosen,
        "minimum_positive_segments": CLEAN_CFG["minimum_positive_segments"],
        "minimum_positive_seeds": CLEAN_CFG["minimum_positive_seeds"],
        "chosen_meets_stability_rule": chosen_row["eligible"],
        "clean_conclusion": ("有候选达到预注册干净块稳定性门槛" if eligible_exists else "否"),
        "fallback_params_descriptive_only": not eligible_exists,
        "score_output_paths": [SCORE_META.relative_to(ROOT).as_posix()] +
                              [p.relative_to(ROOT).as_posix() for p in SCORE_PARQUET.values()],
        "test_window_used_for_selection": False,
        "clean_parameter_grid": clean_grid,
        "oos_coverage_days": len(test_signal),
        "oos_coverage_min_folds": int(np.min(coverage[test_signal])),
        "oos_coverage_max_folds": int(np.max(coverage[test_signal])),
        "cash_nonnegative": all(x["min_cash"] >= -1e-6
                                for x in list(selected_stats.values()) +
                                [v for b in bench_stats.values() for v in b.values()]),
        "balance_sheet_consistent": all(
            v["balance_sheet_max_abs_error"] <= 1e-6
            for v in list(selected_stats.values()) + [m for b in bench_stats.values() for m in b.values()]),
        "selected_end_balance_sheet": {
            k: {"cash": v["ending_cash"], "stock_value": v["ending_stock_value"],
                "equity": v["ending_equity"], "identity_error": v["ending_balance_sheet_error"],
                "open_positions": v["ending_positions"]}
            for k, v in selected_stats.items()},
        "benchmark_end_balance_sheet": {
            b: {k: {"cash": v["ending_cash"], "stock_value": v["ending_stock_value"],
                    "equity": v["ending_equity"], "identity_error": v["ending_balance_sheet_error"],
                    "open_positions": v["ending_positions"]}
                for k, v in modes.items()} for b, modes in bench_stats.items()},
    }
    common = ["strategy", "seed", "cost_mode", "k", "exposure", "threshold", "band", "period"]
    cash_fields = common + ["signal_date", "trade_date", "stock_code", "shares", "target_shares",
                            "stock_equity", "stock_exposure", "cash", "portfolio_equity",
                            "portfolio_exposure", "daily_return", "matched_daily_return"]
    trade_fields = common + ["signal_date", "date", "code", "side", "quantity", "open_price",
                              "fill_price", "notional", "fees", "target_exposure"]
    _write_csv(CSV_CASH, cash_rows, cash_fields)
    _write_csv(CSV_TRADES, trade_rows, trade_fields)
    _write_csv(CSV_HOLD, hold_cash, cash_fields)
    _write_csv(CSV_HOLD_TRADES, hold_trade, trade_fields)
    _plot(selected_curves, bench_curves["same_pool_equal_weight"]["net"])
    score_output_meta = _write_score_outputs(panel, quarter_scores, signatures, export_records)
    audit_doc["score_outputs"] = score_output_meta
    _write_md(audit_doc)
    _write_json(AUDIT, audit_doc)
    _write_json(VALIDATION, validation)
    return {"audit": audit_doc, "validation": validation}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args(argv)
    log = ROOT / "model_logs/analysis_0.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8", buffering=1) as f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = f
        try:
            result = run_analysis(audit=args.audit)
            print(json.dumps(result["audit"]["selected_params"], ensure_ascii=False))
        finally:
            sys.stdout, sys.stderr = old_out, old_err


if __name__ == "__main__":
    main()
