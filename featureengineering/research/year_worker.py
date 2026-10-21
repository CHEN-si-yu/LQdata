"""Proposed research/year_worker.py; draft only, not installed or executed.

The parent runner owns run IDs, Catalog events, candidate Manifest commits,
source/computation identity checks, and run.json. This worker owns exactly one
candidate-spec-year calculation and never finalizes shared state.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np


def _load_candidate_builder_sources(cfg, sources):
    """Load only catalog-bound project sources by path/hash in a spawned worker."""
    from research.catalog import Catalog

    if not isinstance(sources, list):
        raise TypeError("candidate_builder_sources must be a list")
    catalog = Catalog(cfg)
    root = Path(cfg.contract_root).resolve()
    loaded = {}
    seen_candidates = set()
    for item in sources:
        if not isinstance(item, dict):
            raise TypeError("candidate builder source record must be an object")
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in seen_candidates:
            raise ValueError("candidate builder source IDs must be unique strings")
        seen_candidates.add(candidate_id)
        record = catalog.read("candidates", candidate_id)
        reference = record["definition"].get("implementation")
        if item.get("candidate_definition_hash") != record.get("definition_hash"):
            raise ValueError("Worker candidate definition hash disagrees with its immutable catalog record")
        if not isinstance(reference, str) or not re.fullmatch(
                r"research\.[a-zA-Z0-9_.]+:[a-zA-Z_][a-zA-Z0-9_]*", reference):
            raise ValueError("Catalog contains an invalid candidate implementation reference")
        module_name, _function_name = reference.split(":", 1)
        if item.get("implementation") != reference or item.get("module_name") != module_name:
            raise ValueError("Worker candidate source disagrees with its immutable implementation reference")
        source_path = Path(item.get("source_path", ""))
        if source_path.is_absolute():
            raise ValueError("Candidate source path must be project-relative")
        path = (root / source_path).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError("Candidate source path is missing or outside the project")
        catalog_sha256 = record["definition"].get("source_hash")
        supplied_sha256 = item.get("source_sha256")
        if supplied_sha256 != catalog_sha256:
            raise ValueError("Worker candidate source hash disagrees with its immutable catalog record")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != supplied_sha256:
            raise ValueError("Candidate source changed after the parent sealed its source identity")
        prior = loaded.get(module_name)
        if prior is not None:
            if prior != (path, actual_sha256):
                raise ValueError("One candidate module maps to inconsistent project source identities")
            continue
        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                # Fall back only when the requested module itself is absent;
                # import errors inside an existing module must propagate.
                if exc.name != module_name:
                    raise
                module_spec = importlib.util.spec_from_file_location(module_name, path)
                if module_spec is None or module_spec.loader is None:
                    raise ImportError(f"Cannot load frozen candidate builder: {module_name}")
                module = importlib.util.module_from_spec(module_spec)
                sys.modules[module_name] = module
                try:
                    module_spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(module_name, None)
                    raise
        imported_path = Path(inspect.getfile(module)).resolve()
        if imported_path != path:
            raise ValueError(f"Imported module path disagrees with frozen source: {module_name}")
        imported_hash = hashlib.sha256(imported_path.read_bytes()).hexdigest()
        if imported_hash != supplied_sha256:
            raise ValueError(f"Imported candidate module hash mismatch: {module_name}")
        loaded[module_name] = (path, actual_sha256)
    return {name: value[1] for name, value in loaded.items()}


def run_year_worker(job: dict) -> dict:
    """Calculate one planned spec/year in the candidate's isolated phase cache.

    `job` contains only JSON-serializable values:
      config_raw, project_root, run_namespace, candidate_id, spec_name,
      phase, raw_cutoff, year, year_output_end, override_start,
      expected_days, derived_namespace, st_visibility_namespace, run_id.
    """
    from fea.config import Config, setup_logging
    from fea.engine import Engine
    from fea.universe import st_pool_fingerprint
    from research.catalog import spec_for_candidate
    from research.runner import candidate_registry

    started = time.time()
    root = Path(job["project_root"]).resolve()
    cfg = Config(raw=job["config_raw"], root=root)
    namespace = (root / job["run_namespace"]).resolve()
    if not namespace.is_relative_to(root):
        raise ValueError("worker run namespace escaped project root")
    sandbox = cfg.sandbox(namespace)
    phase = int(job["phase"])
    raw_cutoff = int(job["raw_cutoff"])
    year = int(job["year"])
    output_end = int(job["year_output_end"])
    cache_rel = str(job["derived_namespace"])
    st_rel = str(job["st_visibility_namespace"])
    expected_days = np.asarray(job["expected_days"], dtype=np.int32)
    if not expected_days.size or np.any(expected_days // 10000 != year):
        raise ValueError("worker requires a nonempty single-year output plan")
    if int(expected_days[-1]) > output_end:
        raise ValueError("planned output day exceeds year_output_end")
    expected_cache = f"data/derived/phase_{phase}_cutoff_{raw_cutoff}/year_{year}"
    expected_st = f"state/st_phase_{phase}_{raw_cutoff}"
    if cache_rel != expected_cache or st_rel != expected_st:
        raise ValueError("phase/year cache or ST visibility path mismatch")

    sandbox.raw["_audit_cutoff"] = raw_cutoff
    sandbox.raw["paths"]["derived"] = cache_rel
    cache_abs = (namespace / cache_rel).resolve()
    if not cache_abs.is_relative_to(namespace):
        raise ValueError("derived cache escaped candidate namespace")
    old_env = {key: os.environ.get(key) for key in
               ("FEA_CACHE_DIR", "FEA_AUDIT_TRACE", "FEA_LOG_NAME")}
    os.environ["FEA_CACHE_DIR"] = str(cache_abs)
    os.environ["FEA_AUDIT_TRACE"] = str(
        Path("artifacts/access_trace") / f"phase_{phase}" / f"year_{year}")
    os.environ["FEA_LOG_NAME"] = f"{job['run_id']}_p{phase}_y{year}"
    try:
        setup_logging(cfg=sandbox)
        worker_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if worker_source_sha256 != job.get("worker_source_sha256"):
            raise RuntimeError("worker source changed after the coordinator sealed its identity")
        builder_source_records = job.get("candidate_builder_sources", [])
        loaded_builder_sources = _load_candidate_builder_sources(cfg, builder_source_records)
        expected_modules = {item["module_name"] for item in builder_source_records}
        if set(loaded_builder_sources) != expected_modules:
            raise RuntimeError("Not every frozen candidate builder module was loaded")
        if (not expected_days.size or not np.array_equal(expected_days, np.unique(expected_days))
                or np.any(expected_days // 10000 != year)):
            raise ValueError("coordinator supplied an unsorted, duplicate, or cross-year plan")
        root_spec = spec_for_candidate(cfg, job["candidate_id"])
        with candidate_registry(root_spec, cfg) as registry:
            item = registry.get(job["spec_name"])
            if item is None:
                raise ValueError(f"spec is outside frozen candidate closure: {job['spec_name']}")
            engine = Engine(sandbox)
            engine.up.audit_cutoff = raw_cutoff
            st_record_path = (namespace / st_rel / "visible.json").resolve()
            if not st_record_path.is_relative_to(namespace) or not st_record_path.is_file():
                raise RuntimeError("parent phase ST visibility record is missing")
            st_record = json.loads(st_record_path.read_text(encoding="utf-8"))
            if st_record.get("phase") != phase or st_record.get("raw_cutoff") != raw_cutoff:
                raise RuntimeError("parent phase ST visibility record has wrong phase/cutoff")
            if st_pool_fingerprint(engine.up, engine.codes) != st_record.get("visible_st"):
                raise RuntimeError("raw visible ST changed after parent phase gate")
            # Existing state_writes=False prevents a worker from touching the
            # shared ST journal or candidate manifest. Parent did the phase ST
            # gate before dispatch and commits the manifest after the barrier.
            engine.state_writes = False
            engine.override_start = int(job["override_start"])
            todo = engine.prebuild([item], output_end, rebuild=False, jobs=1)
            matches = [(current, manifest, plan) for current, manifest, plan in todo
                       if current.name == item.name]
            if not matches:
                raise RuntimeError("coordinator scheduled a year with no worker plan")
            if len(matches) != 1:
                raise RuntimeError("worker prebuild returned duplicate spec plans")
            current, _manifest_read_only, plan = matches[0]
            if set(plan) != {year}:
                raise RuntimeError(f"year worker received cross-year plan keys: {sorted(plan)}")
            actual_days = np.asarray(plan[year], dtype=np.int32)
            if not np.array_equal(actual_days, expected_days):
                raise RuntimeError("worker plan differs from coordinator's immutable expected days")
            engine_result = engine.run_year(current, year, actual_days)
            if engine_result.get("error"):
                raise RuntimeError(f"Engine year task failed: {engine_result['error']}")
        factor_root = sandbox.market_factors_dir if item.is_market else sandbox.factors_dir
        partition = (factor_root / item.name / f"year={year}" / "data.parquet").resolve()
        if not partition.is_relative_to(namespace) or not partition.is_file():
            raise RuntimeError("standard annual output partition is missing or escaped the run namespace")
        partition_hasher = hashlib.sha256()
        with partition.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                partition_hasher.update(block)
        partition_sha256 = partition_hasher.hexdigest()
        return {
            "status": "year_complete",
            "candidate_id": job["candidate_id"],
            "candidate_definition_hash": job["candidate_definition_hash"],
            "protocol_hash": job["protocol_hash"],
            "run_id": job["run_id"],
            "spec": job["spec_name"],
            "phase": phase,
            "year": year,
            "year_output_start": int(actual_days[0]),
            "year_output_end": output_end,
            "raw_cutoff": raw_cutoff,
            "derived_namespace": cache_rel,
            "st_visibility_namespace": st_rel,
            "source_identity_hash": job["source_identity_hash"],
            "planned_days": actual_days.tolist(),
            "engine_result": engine_result,
            "partition_sha256": partition_sha256,
            "worker_source_sha256": worker_source_sha256,
            "candidate_builder_sources": job["candidate_builder_sources"],
            "worker_pid": os.getpid(),
            "seconds": round(time.time() - started, 3),
        }
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        setup_logging(cfg=cfg)
