"""Explicit review decisions linked to real immutable engineering evidence."""
from __future__ import annotations

import json
import math

from .catalog import Catalog, STATES, TRANSITIONS, clean_json, project_path, slug, spec_for_candidate
from .evaluation import read_evaluation
from .protocol import digest, load_protocol, utc_now
from .runner import (RAW_VISIBILITY_CONTRACTS, candidate_registry, computation_identity, dependency_specs,
                     read_run, compare_frames, factor_frame, validate_raw_visibility)


def read_verification(cfg, identity):
    path = project_path(cfg, f"artifacts/research/verifications/{slug(identity)}.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    contract = record.get("raw_visibility_contract")
    if contract not in RAW_VISIBILITY_CONTRACTS:
        raise ValueError("Verification did not use a supported raw availability/date contract")
    runs = {}
    for key in ("reference_run", "cutoff_run", "incremental_run"):
        namespace, run = read_run(cfg, record[key])
        validate_raw_visibility(run)
        if run["candidate_id"] != record["candidate_id"]:
            raise ValueError("Verification references a different candidate")
        if run.get("raw_visibility", {}).get("contract") != contract:
            raise ValueError("Verification record contract disagrees with a linked run; refusing an implicit v1/v2 upgrade")
        runs[key] = (namespace, run)
    reference = factor_frame(runs["reference_run"][0], record["candidate_id"])
    causal = compare_frames(reference[reference.trade_date.str.replace("-", "").astype(int) <= int(record["cutoff"])],
                            factor_frame(runs["cutoff_run"][0], record["candidate_id"]))
    parity = compare_frames(reference, factor_frame(runs["incremental_run"][0], record["candidate_id"]))
    identities = {run["source_identity_hash"] for _, run in runs.values()}
    computations = {run["computation_identity_hash"] for _, run in runs.values()}
    if not (causal["passed"] and parity["passed"] and len(identities) == 1 and len(computations) == 1):
        raise ValueError("Actual linked outputs no longer support the saved engineering verification")
    if causal != record["causal_truncation"] or parity != record["full_incremental_parity"]:
        raise ValueError("Verification summary does not match the actual linked outputs")
    return record


def _current_evidence_identity(cfg, identity, evaluation):
    """Only read contracts/source files; never build or watermark production."""
    from fea.engine import Engine
    _, reference = read_run(cfg, evaluation["calculation_run_id"])
    if reference["candidate_id"] != identity:
        raise ValueError("Current review references a different candidate")
    protocol = load_protocol(cfg)
    if reference.get("protocol_hash") != protocol.identity or evaluation.get("protocol_hash") != protocol.identity:
        raise ValueError("Current protocol changed; previous evidence cannot be reused")
    spec = spec_for_candidate(cfg, identity)
    with candidate_registry(spec, cfg) as registry:
        engine = Engine(cfg)
        current = computation_identity(cfg, engine, dependency_specs(spec, registry, protocol.horizons))
    current_hash = digest(current)
    if (reference.get("computation_identity") != current
            or reference.get("computation_identity_hash") != current_hash
            or evaluation.get("computation_identity_hash") != current_hash):
        raise ValueError("Current computation code, dependency versions or configuration changed; previous evidence cannot be reused")
    return current_hash


def _check_evidence(cfg, identity, evaluation_id, verification_id):
    if not evaluation_id or not verification_id:
        raise ValueError("Qualification requires actual evaluation and full/incremental/causal verification IDs")
    evaluation = read_evaluation(cfg, evaluation_id)
    verification = read_verification(cfg, verification_id)
    if evaluation["candidate_id"] != identity or verification["candidate_id"] != identity:
        raise ValueError("Evidence candidate does not match the lifecycle decision")
    if verification["reference_run"] != evaluation["calculation_run_id"]:
        raise ValueError("Evaluation and engineering verification must share the same reference calculation")
    if not verification["passed"]:
        raise ValueError("Engineering verification failed")
    _current_evidence_identity(cfg, identity, evaluation)
    # This is sample sufficiency, not an alpha significance threshold. Require
    # actual finite IC evidence for both preregistered primary horizons.
    if evaluation.get("is_market"):
        rows = evaluation["summary"]
        if len(rows) != 1 or int(rows[0].get("days", 0)) < 60:
            raise ValueError("Market context qualification needs at least 60 finite actual dates")
    for h in ([] if evaluation.get("is_market") else evaluation["primary_horizons"]):
        rows = [r for r in evaluation["summary"] if int(r["horizon"]) == int(h)]
        if len(rows) != 1:
            raise ValueError(f"Missing primary-horizon evidence: {h}")
        row = rows[0]
        n = row.get("n_days", row.get("days", row.get("ndays", row.get("valid_days", 0))))
        ric = row.get("rankic_mean", row.get("rank_ic_mean", row.get("rankic")))
        if int(n) < 60 or ric is None or not math.isfinite(float(ric)):
            raise ValueError(f"Primary {h}d needs at least 60 finite daily observations; no fixed alpha threshold is imposed")
    return {"evaluation_id": evaluation_id, "verification_id": verification_id,
            "reference_run": evaluation["calculation_run_id"], "protocol_hash": evaluation["protocol_hash"],
            "definition_hash": evaluation["definition_hash"], "source_identity_hash": evaluation["source_identity_hash"],
            "computation_identity_hash": evaluation.get("computation_identity_hash"),
            "engineering_passed": True, "historical_screening_available": True,
            "model_contribution": None, "future_confirmation": None, "validated_alpha": False}


def promote(cfg, identity, target, reason, evaluation_id=None, verification_id=None, external_evidence=None):
    if target not in STATES or target == "candidate":
        raise ValueError(f"Invalid lifecycle target: {target}")
    if not str(reason).strip():
        raise ValueError("An explicit review reason is required")
    catalog = Catalog(cfg)
    current = catalog.state(identity)
    if target not in TRANSITIONS[current]:
        raise ValueError(f"Lifecycle transition {current} -> {target} is not allowed")
    evidence = None
    if target in ("qualified", "shadow", "production"):
        evidence = _check_evidence(cfg, identity, evaluation_id, verification_id)
    if external_evidence is not None:
        if not isinstance(external_evidence, dict):
            raise ValueError("External evidence must be an object")
        # Optional downstream research can be attached later; it is never
        # mistaken for a computation performed inside this project.
        evidence = evidence or {}
        evidence["externally_supplied_evidence"] = clean_json(external_evidence)
        evidence["external_evidence_status"] = "provided_by_caller;not_executed_by_factor_engineering"
    release = None
    if target == "production":
        from fea.release import approve_candidate, load_library, set_status
        spec = spec_for_candidate(cfg, identity)
        library = load_library(cfg)
        existing = next((r for r in library["objects"] if r.get("candidate_id") == identity), None)
        if existing:
            release = set_status(cfg, spec.name, "production", str(reason))
        else:
            release = approve_candidate(cfg, identity, spec, {**evidence, "review_reason": str(reason), "reviewed_at": utc_now()})
    elif target in ("paused", "retired"):
        from fea.release import load_library, set_status
        library = load_library(cfg)
        existing = next((r for r in library["objects"] if r.get("candidate_id") == identity), None)
        if existing:
            release = set_status(cfg, existing["name"], target, str(reason))
    decision = {"from": current, "to": target, "review_reason": str(reason), "evidence": evidence,
                "production_release": release, "interpretation": "feature_engineering_lifecycle_decision;not_a_claim_of_future_profitability"}
    return catalog.event(identity, "lifecycle", decision)
