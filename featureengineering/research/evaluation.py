"""Bounded yearly evaluation using the shared formal labels and quality metrics."""
from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd

from .catalog import Catalog, immutable_json, spec_for_candidate
from .protocol import load_protocol, utc_now, walk_forward_splits
from .runner import read_run


def matrix(frame, dates, codes):
    keys = ["trade_date", "stock_code"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate candidate/label key")
    expected = pd.MultiIndex.from_product([dates, codes], names=keys)
    actual = pd.MultiIndex.from_frame(frame[keys])
    if len(frame) != len(expected) or len(actual.difference(expected)) or len(expected.difference(actual)):
        raise ValueError("Candidate/label grid differs from the fixed requested dates and stock axis")
    val = frame.set_index(keys)["value"].reindex(expected).to_numpy(dtype=np.float64).reshape(len(dates), len(codes))
    if np.isinf(val).any():
        raise ValueError("Infinite factor/label value")
    return val


def evaluate_run(cfg, run_id):
    from fea.eval import (evaluate_panel, model_value_panel, summarize_daily,
                          EVALUATION_CONTRACT, MODEL_VALUE_CONTRACT)
    namespace, run = read_run(cfg, run_id)
    p = load_protocol(cfg)
    if p.identity != run["protocol_hash"]:
        raise ValueError("Current protocol differs from frozen calculation protocol; use a new trial")
    identity = run["candidate_id"]
    spec = spec_for_candidate(cfg, identity)
    result = []
    for year in sorted({int(d) // 10000 for d in run["trading_days"]}):
        dates = [f"{d // 10000:04d}-{d // 100 % 100:02d}-{d % 100:02d}" for d in run["trading_days"] if d // 10000 == year]
        def load(name):
            path = namespace / "data" / "factors" / name / f"year={year}" / "data.parquet"
            frame = pd.read_parquet(path)
            return matrix(frame, dates, run["codes"])
        if spec.is_market:
            frame = pd.read_parquet(namespace / "data" / "market_factors" / identity / f"year={year}" / "data.parquet")
            if frame.trade_date.duplicated().any() or sorted(frame.trade_date.astype(str)) != sorted(dates):
                raise ValueError("Market candidate must have exactly one row for every requested date")
            frame = frame.set_index("trade_date").reindex(dates)
            values = frame.value.to_numpy(dtype=np.float64)
            if np.isinf(values).any():
                raise ValueError("Infinite market candidate value")
            daily = pd.DataFrame({"factor": identity, "trade_date": dates, "horizon": 0,
                                  "market_value": values, "coverage": np.isfinite(values).astype(float),
                                  "evaluation_type": "market_context_quality_only"})
        else:
            raw = load(identity)
            values = model_value_panel(raw, higher_is_better=spec.higher_is_better, winsor=(.01, .99), clip=10.0)
            labels = {h: load(f"label_ret_{h}d") for h in p.horizons}
            daily = evaluate_panel(values, labels, dates, min_n=p.min_pairs, bins=p.bins, factor=identity)
        daily["candidate_id"] = identity
        daily["calculation_run_id"] = run_id
        daily["score_head"] = "market_context" if spec.is_market else "pure_factor"
        daily["value_transform"] = "native_market_scalar_v1" if spec.is_market else p.value_transform
        daily["n_pool"] = 1 if spec.is_market else len(run["codes"])
        result.append(daily)
    daily = pd.concat(result, ignore_index=True)
    if spec.is_market:
        values = daily.market_value.to_numpy(dtype=float)
        summary = [{"name": identity, "horizon": 0, "days": int(np.isfinite(values).sum()),
                    "dates": len(values), "coverage": float(np.isfinite(values).mean()),
                    "rankic": None, "ic": None,
                    "note": "Market scalar has no within-date stock ranking; evaluate interactions as separate stock candidates"}]
    else:
        summary = summarize_daily(daily, block_days=p.block_days)
    evaluation_id = "e_" + uuid.uuid4().hex
    folder = Catalog(cfg).root / "evaluations" / evaluation_id
    folder.mkdir(parents=True, exist_ok=False)
    daily.to_parquet(folder / "daily_metrics.parquet", index=False, compression=cfg.compression)
    record = {"evaluation_id": evaluation_id, "candidate_id": identity, "calculation_run_id": run_id,
              "created_at": utc_now(), "protocol_hash": p.identity, "definition_hash": run["definition_hash"],
              "source_identity_hash": run["source_identity_hash"], "summary": summary,
              "computation_identity_hash": run["computation_identity_hash"],
              "evaluation_contract": EVALUATION_CONTRACT, "model_value_contract": MODEL_VALUE_CONTRACT,
              "primary_horizons": list(p.primary_horizons), "auxiliary_horizons": list(p.auxiliary_horizons),
              "is_market": bool(spec.is_market),
              "interpretation": "market_context_quality_only" if spec.is_market else "retrospective_cross_sectional_screening", "validated_alpha": False,
              "model_contribution": None, "future_confirmation": None,
              "daily_file": "daily_metrics.parquet", "rows": len(daily),
              "walk_forward_plan": walk_forward_splits(run["trading_days"], p)}
    # Hash seals the daily evidence; promotion rechecks it.
    from .runner import file_hash
    record["evaluation_source_hash"] = file_hash(cfg.contract_root / "fea" / "eval.py")
    record["research_evaluation_source_hash"] = file_hash(cfg.contract_root / "research" / "evaluation.py")
    record["daily_file_hash"] = file_hash(folder / "daily_metrics.parquet")
    immutable_json(folder / "evaluation.json", record)
    Catalog(cfg).event(identity, "evaluation", {"evaluation_id": evaluation_id, "calculation_run_id": run_id})
    return record


def read_evaluation(cfg, evaluation_id):
    from .catalog import project_path, slug
    from .runner import file_hash
    folder = project_path(cfg, f"artifacts/research/evaluations/{slug(evaluation_id)}")
    record = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    if file_hash(folder / "daily_metrics.parquet") != record["daily_file_hash"]:
        raise ValueError("Evaluation daily evidence changed after publication")
    read_run(cfg, record["calculation_run_id"])
    return record
