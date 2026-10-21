"""Compute candidates, dependencies and formal labels in a closed namespace."""
from __future__ import annotations

import contextlib
import gc
import hashlib
import inspect
import json
import os
import traceback
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .catalog import Catalog, immutable_json, project_path, spec_for_candidate
from .protocol import digest, load_protocol, utc_now


RAW_VISIBILITY = "upstream_read_availability_date_cap_v1"
RAW_VISIBILITY_V2 = "upstream_read_availability_date_cap_per_year_v2"
RAW_VISIBILITY_CONTRACTS = frozenset((RAW_VISIBILITY, RAW_VISIBILITY_V2))


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def computation_identity(cfg, engine, specs):
    """Exact local computation sources and effective numerical contracts."""
    root = cfg.contract_root
    candidates = {r["candidate_id"]: r for r in Catalog(cfg).records("candidates")}
    objects = {}
    for spec in specs:
        source = Path(inspect.getsourcefile(spec.fn)).resolve() if spec.fn else None
        if source is not None and not source.is_relative_to(root):
            raise ValueError(f"Computed implementation is outside the project: {spec.name}")
        objects[spec.name] = {"version": spec.version, "recipe": engine._recipe(spec),
                              "is_label": spec.is_label, "is_market": spec.is_market,
                              "implementation_source": source.relative_to(root).as_posix() if source else None,
                              "implementation_source_sha256": file_hash(source) if source else None,
                              "candidate_definition_hash": candidates.get(spec.name, {}).get("definition_hash")}
    return {
        "objects": objects,
        "common_sources": {p.relative_to(root).as_posix(): file_hash(p)
                           for p in sorted((root / "fea").glob("*.py"))},
        "research_sources": {p.relative_to(root).as_posix(): file_hash(p)
                             for p in sorted((root / "research").glob("*.py"))},
    }


def resolve_candidate_closure(cfg, spec, registry):
    """Resolve explicit immutable research names, registered factors and raw contracts.

    A private candidate name is never guessed to be an upstream table. All
    candidate implementations are reloaded through the immutable catalog; an
    already approved research parent is subject to the same identity check.
    """
    from fea.release import definition
    from fea.upstream import Upstream
    records = Catalog(cfg).records("candidates")
    candidates = {r["candidate_id"]: r for r in records}
    if len(candidates) != len(records):
        raise ValueError("Duplicate immutable candidate names")
    source_contract = cfg.input_contract_path("frequency", "conf/frequency.yaml")
    if not source_contract.exists():
        raise ValueError("Research dependency resolution requires the project's source contracts")
    payload = yaml.safe_load(source_contract.read_text(encoding="utf-8")) or {}
    raw_names = set(payload.get("datasets", {}))
    upstream = Upstream(cfg.upstream, cfg.root)
    resolved, visiting, seen = [], set(), set()
    loaded = {spec.name: spec}

    def visit(current):
        if current.name in seen:
            return
        if current.name in visiting:
            raise ValueError(f"Research dependency cycle at {current.name}")
        if current.is_label or current.forward_days:
            raise ValueError(f"Research factors cannot consume future labels: {current.name}")
        if current.name in raw_names:
            raise ValueError(f"Ambiguous raw-source and factor name: {current.name}")
        previous = registry.get(current.name)
        if current.name in candidates and previous is not None and definition(previous, cfg) != definition(current, cfg):
            raise ValueError(f"Research candidate definition/name collision: {current.name}")
        visiting.add(current.name)
        for name in current.deps:
            if name in raw_names:
                if name in registry or name in candidates:
                    raise ValueError(f"Ambiguous raw-source and factor dependency: {name}")
                if not upstream.exists(name):
                    raise ValueError(f"Declared raw source has no local Parquet data: {name}")
            elif name in candidates:
                if name not in loaded:
                    loaded[name] = spec_for_candidate(cfg, name)
                visit(loaded[name])
            elif name in registry:
                visit(registry[name])
            else:
                raise ValueError(f"Unknown dependency {name!r}; declare an existing raw source contract or an immutable candidate/factor name")
        visiting.remove(current.name)
        seen.add(current.name)
        resolved.append(current)

    visit(spec)
    return resolved


@contextlib.contextmanager
def candidate_registry(spec, cfg):
    """Temporarily register the entire explicit candidate closure, then restore."""
    import factors  # noqa: F401
    from fea.spec import REGISTRY
    resolved = resolve_candidate_closure(cfg, spec, REGISTRY)
    private_names = {r["candidate_id"] for r in Catalog(cfg).records("candidates")}
    temporary = [s for s in resolved if s.name in private_names or s.name == spec.name]
    previous = {s.name: REGISTRY.get(s.name) for s in temporary}
    REGISTRY.update({s.name: s for s in temporary})
    try:
        yield REGISTRY
    finally:
        for name, original in previous.items():
            if original is None:
                REGISTRY.pop(name, None)
            else:
                REGISTRY[name] = original


def dependency_specs(spec, registry, horizons):
    result, seen, visiting = [], set(), set()

    def visit(current):
        if current.name in seen:
            return
        if current.name in visiting:
            raise ValueError("Candidate dependency cycle")
        visiting.add(current.name)
        for name in current.deps:
            if name in registry:
                parent = registry[name]
                if parent.is_label:
                    raise ValueError("An ordinary research factor cannot consume future labels")
                visit(parent)
        visiting.remove(current.name)
        seen.add(current.name)
        result.append(current)

    visit(spec)
    for h in horizons:
        name = f"label_ret_{h}d"
        if name not in registry or not registry[name].is_label:
            raise ValueError(f"Formal label is not registered: {name}")
        result.append(registry[name])
    return result


def _source_names(specs):
    names = {s.name for s in specs}
    return sorted({d for s in specs for d in s.deps if d not in names}
                  | {"stock_list", "basic_calendar", "stock_st_info"})


def _research_phase_engine(cfg, phase, cutoff):
    """Keep factor manifests shared, with a separate visible-ST gate per cap."""
    from fea.engine import Engine
    from fea.universe import st_pool_fingerprint
    state_path = f"state/st_phase_{phase}_{cutoff}"

    class PhaseEngine(Engine):
        def st_gate(self):
            # Re-read this implicit universe input on every gate. A cached
            # earlier read must not hide an actual change inside one phase.
            self.up.clear_cache()
            visible = st_pool_fingerprint(self.up, self.codes)
            record = {"phase": phase, "raw_cutoff": cutoff, "visible_st": visible}
            try:
                immutable_json(self.cfg.root / state_path / "visible.json", record)
            except ValueError as exc:
                raise RuntimeError("Visible ST changed within one fixed-cutoff research phase; refusing to continue") from exc
            # Reuse the unchanged production gate, but only its ST state path
            # is redirected. Incremental factor manifests remain in state/.
            original = self.cfg
            raw = {**original.raw, "paths": {**original.raw["paths"], "state": state_path}}
            self.cfg = replace(original, raw=raw)
            try:
                return super().st_gate()
            finally:
                self.cfg = original

    return PhaseEngine(cfg)


def _output_hashes(namespace):
    return {p.relative_to(namespace).as_posix(): file_hash(p)
            for p in sorted((namespace / "data").glob("**/*.parquet"))
            if "derived" not in p.relative_to(namespace).parts}


def candidate_builder_sources(cfg, specs):
    """Freeze the project-contained source files needed to re-import candidate builders.

    Spawned annual workers cannot inherit dynamically loaded parent sys.modules.
    Catalog source hashes and exact implementation module names let each worker
    reconstruct the same modules from disk without serializing Python objects.
    """
    catalog = Catalog(cfg)
    records = {r["candidate_id"]: r for r in catalog.records("candidates")}
    root = cfg.contract_root.resolve()
    by_module = {}
    entries = []
    for spec in specs:
        record = records.get(spec.name)
        if record is None:
            continue
        definition = record["definition"]
        reference = definition["implementation"]
        module_name, _function_name = reference.split(":", 1)
        if getattr(spec.fn, "__module__", None) != module_name:
            raise ValueError(f"Candidate callable module disagrees with immutable reference: {spec.name}")
        path = Path(inspect.getsourcefile(spec.fn)).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"Candidate builder source is outside the project: {spec.name}")
        source_sha256 = file_hash(path)
        if source_sha256 != definition.get("source_hash"):
            raise ValueError(f"Candidate builder source changed after registration: {spec.name}")
        entry = {"candidate_id": spec.name,
                 "candidate_definition_hash": record["definition_hash"],
                 "implementation": reference, "module_name": module_name,
                 "source_path": path.relative_to(root).as_posix(),
                 "source_sha256": source_sha256}
        old = by_module.get(module_name)
        if old and (old[0] != entry["source_path"]
                    or old[1] != entry["source_sha256"]):
            raise ValueError(f"Candidate module resolves to multiple source files: {module_name}")
        by_module[module_name] = (entry["source_path"], entry["source_sha256"])
        entries.append(entry)
    return sorted(entries, key=lambda item: item["candidate_id"])


def dependency_output_starts(engine, specs, requested_start):
    """Materialize private parent history needed by each annual child panel.

    Engine anchors a year at its real January/spec floor even for a late
    request. Its pre-anchor warmup therefore needs persisted parent values,
    not only raw prices. Reverse topological expansion includes grandparents;
    only dependency outputs expand, never the candidate or formal labels.
    """
    from fea.engine import minus_days
    starts = {s.name: int(requested_start) for s in specs}
    by_name = {s.name: s for s in specs}
    for current in reversed(specs):
        anchor = engine._anchor_for(current, np.asarray([starts[current.name]], dtype=np.int32))
        required = max(engine._start_floor, minus_days(anchor, current.warmup_days))
        for name in current.deps:
            if name in by_name:
                if by_name[name].is_label or by_name[name].forward_days:
                    raise ValueError("Research dependency history cannot include future labels")
                starts[name] = min(starts[name], required)
    return starts



def _resolve_year_worker_jobs(requested, cfg, specs):
    """0 selects 6 daily or 2 minute workers; explicit counts cap at 9."""
    import yaml
    from fea.resources import safe_jobs

    requested = int(requested or 0)
    if requested == 0:
        source_contract = cfg.input_contract_path("frequency", "conf/frequency.yaml")
        contract = yaml.safe_load(source_contract.read_text(encoding="utf-8")) or {}
        datasets = contract.get("datasets", {})
        raw_names = _source_names(specs)
        minute = any(datasets.get(name, {}).get("freq") == "minute" for name in raw_names)
        requested = 2 if minute else 6
    if not 1 <= requested <= 9:
        raise ValueError("Annual research worker count must be between 1 and 9")
    return safe_jobs(requested, policy=cfg)


def _dispatch_year_workers(jobs, max_workers):
    """Dispatch isolated processes and propagate the first worker exception."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from research.year_worker import run_year_worker

    if not jobs:
        return []
    executor = ProcessPoolExecutor(max_workers=min(max_workers, len(jobs)),
                                   mp_context=mp.get_context("spawn"),
                                   max_tasks_per_child=1)
    futures = {executor.submit(run_year_worker, job): job for job in jobs}
    results = []
    try:
        for future in as_completed(futures):
            # result() re-raises the original worker exception in the parent.
            results.append(future.result())
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return results


def _validate_year_worker_results(jobs, results, namespace, spec, phase,
                                  raw_cutoff, expected_worker_sha256):
    """Fail closed on missing/extra tasks, altered caps, days, or output bytes."""
    from pathlib import Path

    expected = {(job["spec_name"], int(job["phase"]), int(job["year"])): job
                for job in jobs}
    if len(expected) != len(jobs):
        raise RuntimeError("coordinator scheduled a duplicate (spec, phase, year) task")
    actual = {}
    worker_pids = set()
    for result in results:
        key = (result.get("spec"), result.get("phase"), result.get("year"))
        if key in actual:
            raise RuntimeError(f"duplicate worker result: {key}")
        if key not in expected:
            raise RuntimeError(f"unexpected worker result: {key}")
        job = expected[key]
        if result.get("status") != "year_complete":
            raise RuntimeError(f"worker did not complete: {key}")
        pid = result.get("worker_pid")
        if not isinstance(pid, int) or pid <= 0 or pid in worker_pids:
            raise RuntimeError("annual workers must run in distinct one-task processes")
        worker_pids.add(pid)
        for field in ("candidate_id", "run_id", "raw_cutoff", "derived_namespace",
                      "st_visibility_namespace", "source_identity_hash",
                      "worker_source_sha256", "candidate_builder_sources"):
            expected_value = {
                "candidate_id": job["candidate_id"], "run_id": job["run_id"],
                "raw_cutoff": raw_cutoff, "derived_namespace": job["derived_namespace"],
                "st_visibility_namespace": job["st_visibility_namespace"],
                "source_identity_hash": job["source_identity_hash"],
                "worker_source_sha256": expected_worker_sha256,
                "candidate_builder_sources": job["candidate_builder_sources"],
            }[field]
            if result.get(field) != expected_value:
                raise RuntimeError(f"worker {key} changed {field}")
        if result.get("year_output_end") != job["year_output_end"]:
            raise RuntimeError(f"worker {key} changed its annual output end")
        if result.get("planned_days") != job["expected_days"]:
            raise RuntimeError(f"worker {key} changed its planned date axis")
        engine_result = result.get("engine_result")
        if not isinstance(engine_result, dict) or engine_result.get("error"):
            raise RuntimeError(f"worker {key} returned no successful Engine result")
        if int(engine_result.get("year", -1)) != int(job["year"]):
            raise RuntimeError(f"worker {key} wrote the wrong annual result")
        partition = (Path(namespace) / "data" / ("market_factors" if spec.is_market else "factors")
                     / spec.name / f"year={job['year']}" / "data.parquet").resolve()
        if not partition.is_relative_to(Path(namespace).resolve()) or not partition.is_file():
            raise RuntimeError(f"worker {key} did not write its standard annual output partition")
        if file_hash(partition) != result.get("partition_sha256"):
            raise RuntimeError(f"worker {key} output partition hash mismatch")
        actual[key] = result
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise RuntimeError(f"missing year worker results: {missing}")
    return [actual[key] for key in sorted(actual, key=lambda k: k[2])]


def run_candidate(cfg, identity, start=None, end=None, jobs=0, phase_ends=None,
                  kind="calculation"):
    """Run a candidate with dependency barriers and isolated annual workers."""
    import copy
    from fea.config import setup_logging
    from fea.engine import Engine
    from fea.manifest import Manifest
    from fea.dates import int_to_str
    from research.year_worker import run_year_worker  # hashed top-level research source

    p = load_protocol(cfg)
    if cfg.default_start != p.output_start or cfg.frozen_universe is None:
        raise ValueError("Research requires the project's frozen universe and unchanged 2018-01-01 output floor")
    catalog = Catalog(cfg)
    record = catalog.read("candidates", identity)
    spec = spec_for_candidate(cfg, identity)
    # Resolve dependencies before choosing the auto worker default. This reads
    # definitions/contracts only and starts no calculation or official trial.
    with candidate_registry(spec, cfg) as registry:
        specs = dependency_specs(spec, registry, p.horizons)
    jobs = _resolve_year_worker_jobs(jobs, cfg, specs)
    start_int = int(str(start or p.output_start).replace("-", ""))
    if start_int < 20180101:
        raise ValueError("Research output cannot precede 2018-01-01")
    run_id = "r_" + uuid.uuid4().hex
    namespace = project_path(cfg, f"artifacts/research/runs/{run_id}")
    namespace.mkdir(parents=True, exist_ok=False)
    sandbox = cfg.sandbox(namespace)
    setup_logging(cfg=sandbox)
    begin = {"run_id": run_id, "candidate_id": identity, "kind": kind,
             "started_at": utc_now(), "definition_hash": record["definition_hash"],
             "protocol": p.to_dict(), "protocol_hash": p.identity,
             "start_requested": start_int, "end_requested": end, "jobs": jobs,
             "namespace": str(namespace.relative_to(cfg.root)), "status": "running",
             "validated_alpha": False}
    immutable_json(namespace / "begin.json", begin)
    catalog.event(identity, "trial_started", {"run_id": run_id, "kind": kind})
    old_env = {key: os.environ.get(key) for key in
               ("FEA_CACHE_DIR", "FEA_AUDIT_TRACE", "FEA_LOG_NAME")}
    os.environ["FEA_CACHE_DIR"] = str(namespace / "data" / "derived")
    os.environ["FEA_AUDIT_TRACE"] = str(namespace / "artifacts" / "access_trace")
    os.environ["FEA_LOG_NAME"] = run_id
    try:
        with candidate_registry(spec, cfg) as registry:
            specs = dependency_specs(spec, registry, p.horizons)
            builder_sources = candidate_builder_sources(cfg, specs)
            initial = Engine(sandbox)
            if len(initial.codes) != p.expected_stocks:
                raise ValueError(f"Expected {p.expected_stocks} fixed stocks, got {len(initial.codes)}")
            last = initial.baseline_last_day()
            end_int = min(int(str(end).replace("-", "")), last) if end else last
            if end_int < start_int:
                raise ValueError("Research end precedes requested start")
            sandbox.raw["_audit_cutoff"] = end_int
            initial.up.audit_cutoff = end_int
            initial.up.clear_cache()
            source_before = {name: initial.watermark(name) for name in _source_names(specs)}
            computation_before = computation_identity(cfg, initial, specs)
            days = initial.cal.between(start_int, end_int)
            if not days.size:
                raise ValueError("No actual trading days in the requested interval")
            universe_codes = initial.codes.astype(str).tolist()
            universe_hash = digest(universe_codes)
            environment = initial.universe_fp
            output_starts = dependency_output_starts(initial, specs, start_int)
            del initial
            gc.collect()
            ends = [int(str(e).replace("-", "")) for e in (phase_ends or [end_int])]
            if (ends[-1] != end_int or any(e < start_int or e > end_int for e in ends)
                    or ends != sorted(set(ends))):
                raise ValueError("Incremental phases must increase and finish at the run end")

            tasks, spec_barriers = [], []
            phase_year_paths, phase_st_paths = {}, []
            worker_path = Path(__file__).with_name("year_worker.py")
            worker_sha256 = file_hash(worker_path)
            source_identity_hash = digest(source_before)
            for phase, phase_end in enumerate(ends):
                sandbox.raw["_audit_cutoff"] = phase_end
                phase_cache_root = f"data/derived/phase_{phase}_cutoff_{phase_end}"
                st_path = f"state/st_phase_{phase}_{phase_end}"
                phase_st_paths.append(st_path)
                sandbox.raw["paths"]["derived"] = phase_cache_root
                os.environ["FEA_CACHE_DIR"] = str(namespace / phase_cache_root)

                # The coordinator owns exactly one visible-ST baseline per
                # phase; workers only verify its immutable fingerprint.
                phase_engine = _research_phase_engine(sandbox, phase, phase_end)
                phase_engine.up.audit_cutoff = phase_end
                phase_engine.up.clear_cache()
                phase_engine.st_gate()
                phase_engine.delay_gate()
                phase_engine._run_end = phase_end
                phase_engine._end_str = int_to_str(phase_end)

                for item in specs:  # dependency_specs is topological; labels last
                    phase_engine._wm_cache.clear()
                    phase_engine.override_start = output_starts[item.name]
                    manifest = Manifest.load(sandbox.state_dir, item.name)
                    old_recipe = manifest.recipe
                    # Existing serial runner resets on the first scheduled
                    # output-start year of phase 0. A whole-spec plan is the
                    # same reset once, before every annual task is dispatched.
                    rebuild = phase == 0 and output_starts[item.name] <= phase_end
                    plan = phase_engine.plan(item, manifest, phase_end, rebuild=rebuild)
                    if manifest.recipe != old_recipe:
                        manifest.save()  # coordinator-only reset persistence
                    plan = {int(y): np.sort(np.asarray(ds, dtype=np.int32))
                            for y, ds in plan.items() if np.size(ds)}
                    if not plan:
                        continue

                    jobs_for_spec = []
                    for year in sorted(plan):
                        expected_days = [int(d) for d in plan[year].tolist()]
                        if not expected_days or any(d // 10000 != year for d in expected_days):
                            raise RuntimeError(f"Coordinator planned an invalid year axis for {item.name}/{year}")
                        year_output_end = min(phase_end, year * 10000 + 1231)
                        derived_path = f"{phase_cache_root}/year_{year}"
                        phase_year_paths.setdefault(str(phase), {})[str(year)] = derived_path
                        jobs_for_spec.append({
                            "config_raw": copy.deepcopy(sandbox.raw),
                            "project_root": str(cfg.root.resolve()),
                            "run_namespace": str(namespace.relative_to(cfg.root)),
                            "candidate_id": identity, "candidate_definition_hash": record["definition_hash"],
                            "protocol_hash": p.identity, "spec_name": item.name,
                            "phase": phase, "raw_cutoff": phase_end, "year": year,
                            "year_output_end": year_output_end,
                            "override_start": max(output_starts[item.name], year * 10000 + 101),
                            "expected_days": expected_days,
                            "derived_namespace": derived_path,
                            "st_visibility_namespace": st_path, "run_id": run_id,
                            "source_identity_hash": source_identity_hash,
                            "worker_source_sha256": worker_sha256,
                            "candidate_builder_sources": builder_sources,
                        })

                    # One spec barrier: no dependent spec starts before every
                    # parent year has succeeded and parent coverage is saved.
                    worker_results = _dispatch_year_workers(jobs_for_spec, jobs)
                    worker_results = _validate_year_worker_results(
                        jobs_for_spec, worker_results, namespace, item, phase,
                        phase_end, worker_sha256)
                    engine_results = [r["engine_result"] for r in worker_results]
                    if {int(r["year"]) for r in engine_results} != set(plan):
                        raise RuntimeError(f"Missing or extra worker years for {item.name}")
                    stats = phase_engine._finalize(
                        item, manifest, plan, engine_results,
                        sum(float(r.get("seconds") or 0) for r in engine_results))
                    spec_barriers.append({"phase": phase, "factor": item.name,
                                          "years": sorted(plan), "finalize": stats})
                    for worker in worker_results:
                        engine_result = worker["engine_result"]
                        tasks.append({
                            "phase": phase, "factor": item.name,
                            "year": int(worker["year"]),
                            "raw_cutoff": phase_end,
                            "year_output_start": worker["year_output_start"],
                            "year_output_end": worker["year_output_end"],
                            "derived_namespace": worker["derived_namespace"],
                            "st_visibility_namespace": st_path,
                            "planned_days": worker["planned_days"],
                            "source_identity_hash": source_identity_hash,
                            "worker_source_sha256": worker_sha256,
                            "candidate_builder_sources": builder_sources,
                            "partition_sha256": worker["partition_sha256"],
                            "rows": engine_result.get("rows", 0),
                            "nonnull": engine_result.get("nonnull", 0),
                            "partition_rows": engine_result.get("partition_rows", 0),
                            "min_date": engine_result.get("min_date"),
                            "max_date": engine_result.get("max_date"),
                            "content_sha256": engine_result.get("content_sha256"),
                            "seconds": engine_result.get("seconds", 0),
                        })
                    gc.collect()

            final_engine = Engine(sandbox)
            final_engine.up.audit_cutoff = end_int
            source_after = {name: final_engine.watermark(name) for name in _source_names(specs)}
            computation_after = computation_identity(cfg, final_engine, specs)
            if source_before != source_after:
                raise RuntimeError("Upstream identity changed during research; trial cannot be published as reproducible")
            if computation_before != computation_after:
                raise RuntimeError("Computation code or recipe changed during research; trial cannot be published as reproducible")
            metadata = {**begin, "status": "complete", "finished_at": utc_now(),
                        "start": int(days[0]), "end": int(days[-1]),
                        "trading_days": days.tolist(), "codes": universe_codes,
                        "universe_hash": universe_hash, "environment": environment,
                        "sources": source_before, "source_identity_hash": source_identity_hash,
                        "phase_ends": ends,
                        "raw_visibility": {
                            "contract": RAW_VISIBILITY_V2,
                            "phase_cutoffs": ends,
                            "derived_phase_year_paths": phase_year_paths,
                            "st_phase_paths": phase_st_paths,
                        },
                        "st_visibility_records": [
                            json.loads((namespace / path / "visible.json").read_text(encoding="utf-8"))
                            for path in phase_st_paths],
                        "st_visibility_hashes": {
                            path + "/visible.json": file_hash(namespace / path / "visible.json")
                            for path in phase_st_paths},
                        "dependency_output_starts": output_starts,
                        "computation_identity": computation_before,
                        "computation_identity_hash": digest(computation_before),
                        "worker_source_sha256": worker_sha256,
                        "candidate_builder_sources": builder_sources,
                        "spec_barriers": spec_barriers,
                        "is_market": bool(spec.is_market),
                        "specs": [s.name for s in specs], "tasks": tasks,
                        "output_hashes": _output_hashes(namespace)}
            immutable_json(namespace / "run.json", metadata)
            catalog.event(identity, "trial_completed",
                          {"run_id": run_id, "source_identity_hash": source_identity_hash})
            return metadata
    except Exception as exc:
        failure = {"run_id": run_id, "candidate_id": identity,
                   "failed_at": utc_now(), "error": str(exc),
                   "traceback": traceback.format_exc(), "status": "failed"}
        immutable_json(namespace / "failure.json", failure)
        catalog.event(identity, "trial_failed", {"run_id": run_id, "error": str(exc)})
        raise
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        setup_logging(cfg=cfg)

def read_run(cfg, run_id, verify_hashes=True):
    from .catalog import slug
    path = project_path(cfg, f"artifacts/research/runs/{slug(run_id)}")
    metadata = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete":
        raise ValueError("Only completed immutable trials can be evaluated")
    if verify_hashes:
        for name, expected in {**metadata["output_hashes"], **metadata.get("st_visibility_hashes", {})}.items():
            actual = (path / name).resolve()
            if path not in actual.parents or file_hash(actual) != expected:
                raise ValueError(f"Research output changed after publication: {name}")
    return path, metadata


def validate_raw_visibility(run):
    """Validate legacy v1 or annual-worker v2 raw visibility evidence."""
    visibility = run.get("raw_visibility", {})
    contract = visibility.get("contract")
    ends = run.get("phase_ends", [])
    if visibility.get("phase_cutoffs") != ends:
        raise ValueError("Engineering verification requires actual per-phase raw availability/date caps")
    st_paths = visibility.get("st_phase_paths")
    expected_st = [f"state/st_phase_{i}_{cutoff}" for i, cutoff in enumerate(ends)]
    if st_paths != expected_st:
        raise ValueError("Research phases require separate recorded visible-ST baselines")
    records = run.get("st_visibility_records", [])
    if len(records) != len(ends) or any(
            record.get("phase") != i or record.get("raw_cutoff") != ends[i]
            for i, record in enumerate(records)):
        raise ValueError("Visible-ST records disagree with their actual phases")
    if "stock_st_info" not in run.get("sources", {}):
        raise ValueError("Engineering source identity must include the full physical ST input")
    if not ends or int(ends[-1]) < int(run["end"]):
        raise ValueError("Raw cap does not cover the declared output end")

    if contract == RAW_VISIBILITY:
        caches = visibility.get("derived_phase_paths")
        expected_caches = [f"data/derived/phase_{i}_cutoff_{cutoff}"
                           for i, cutoff in enumerate(ends)]
        if caches != expected_caches:
            raise ValueError("Legacy phases require separate derived visibility namespaces")
        for task in run.get("tasks", []):
            phase = task.get("phase")
            if (not isinstance(phase, int) or not 0 <= phase < len(ends)
                    or task.get("raw_cutoff") != ends[phase]
                    or task.get("derived_namespace") != caches[phase]
                    or task.get("st_visibility_namespace") != st_paths[phase]):
                raise ValueError("Legacy task disagrees with its visibility phase")
        return

    if contract != RAW_VISIBILITY_V2:
        raise ValueError("Unknown raw-visibility contract")
    paths_by_phase = visibility.get("derived_phase_year_paths")
    if not isinstance(paths_by_phase, dict):
        raise ValueError("Annual raw visibility needs a phase/year derived path map")
    seen = set()
    source_hash = run.get("source_identity_hash")
    worker_hash = run.get("worker_source_sha256")
    for task in run.get("tasks", []):
        phase, year = task.get("phase"), task.get("year")
        if not isinstance(phase, int) or not 0 <= phase < len(ends):
            raise ValueError("Annual task has an invalid phase")
        if not isinstance(year, int) or year < 1900 or year > 9999:
            raise ValueError("Annual task has an invalid year")
        key = (phase, task.get("factor"), year)
        if key in seen:
            raise ValueError("Duplicate annual factor/phase/year task")
        seen.add(key)
        expected_path = f"data/derived/phase_{phase}_cutoff_{ends[phase]}/year_{year}"
        mapped = paths_by_phase.get(str(phase), {}).get(str(year))
        if mapped != expected_path or task.get("derived_namespace") != expected_path:
            raise ValueError("Annual task derived path disagrees with its phase/year map")
        if task.get("raw_cutoff") != ends[phase] or task.get("st_visibility_namespace") != st_paths[phase]:
            raise ValueError("Annual task raw cap or visible-ST path disagrees with its phase")
        expected_output_end = min(ends[phase], year * 10000 + 1231)
        if task.get("year_output_end") != expected_output_end:
            raise ValueError("Annual output end must stay separate from the phase raw cutoff")
        dates = task.get("planned_days")
        if (not dates or dates != sorted(set(dates))
                or any(int(date) // 10000 != year for date in dates)
                or task.get("year_output_start") != dates[0]
                or dates[-1] > expected_output_end):
            raise ValueError("Annual task date axis is missing, duplicate, unsorted, or out of year")
        if task.get("source_identity_hash") != source_hash:
            raise ValueError("Annual task has a different source identity")
        if task.get("worker_source_sha256") != worker_hash:
            raise ValueError("Annual task has a different worker source identity")
    mapped_keys = {(int(p), int(y)) for p, years in paths_by_phase.items() for y in years}
    task_keys = {(phase, year) for phase, _, year in seen}
    if mapped_keys != task_keys:
        raise ValueError("Derived phase/year map contains missing or extra worker paths")

def factor_frame(namespace, name):
    from .catalog import slug
    files = sorted((namespace / "data" / "factors" / slug(name)).glob("year=*/data.parquet"))
    if not files:
        files = sorted((namespace / "data" / "market_factors" / slug(name)).glob("year=*/data.parquet"))
    if not files:
        raise ValueError(f"No candidate output: {name}")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def compare_frames(left, right):
    """Exact key/value/rank parity with NaNs equal; all-empty equality fails."""
    market = "stock_code" not in left.columns
    keys = ["trade_date"] if market else ["trade_date", "stock_code"]
    if ("stock_code" not in right.columns) != market:
        return {"passed": False, "reason": "incompatible_stock_market_schemas"}
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        return {"passed": False, "reason": "duplicate_keys"}
    a = left.sort_values(keys).reset_index(drop=True)
    b = right.sort_values(keys).reset_index(drop=True)
    keys_equal = a[keys].equals(b[keys])
    finite = int(np.isfinite(a["value"].to_numpy()).sum())
    columns = ("value",) if market else ("value", "rank")
    equal = keys_equal and all(np.array_equal(a[key].to_numpy(), b[key].to_numpy(), equal_nan=True) for key in columns)
    return {"passed": bool(equal and finite > 0), "same_grid": bool(keys_equal), "rows": len(a),
            "finite_values": finite, "exact_float32_value_rank": bool(equal)}


def verify_candidate(cfg, run_id, jobs=0, cutoff=None):
    """Run actual truncated calculation and actual two-phase incremental replay.

    This checks the candidate's ordinary time-array causality. It does not
    reconstruct unavailable historical vendor vintages, and records that limit.
    Future formal labels are excluded from the causal comparison by design.
    """
    namespace, full = read_run(cfg, run_id)
    validate_raw_visibility(full)
    if full["kind"] not in ("calculation", "full_validation"):
        raise ValueError("Verification reference must be a full calculation trial")
    if full["raw_visibility"]["contract"] != RAW_VISIBILITY_V2:
        raise ValueError("Annual-worker verification requires a v2 reference run; create a new full run under the current computation identity")
    days = full["trading_days"]
    if len(days) < 6:
        raise ValueError("Verification needs at least six actual trading days")
    cut = int(str(cutoff).replace("-", "")) if cutoff else int(days[-3])
    if cut not in days or cut == days[-1]:
        raise ValueError("Causal cutoff must be an earlier trading date inside the reference run")
    identity = full["candidate_id"]
    truncated = run_candidate(cfg, identity, full["start"], cut, jobs=jobs, kind="causal_truncation")
    increment = run_candidate(cfg, identity, full["start"], full["end"], jobs=jobs,
                              phase_ends=[cut, full["end"]], kind="incremental_validation")
    validate_raw_visibility(truncated)
    validate_raw_visibility(increment)
    linked_contracts = {full["raw_visibility"]["contract"],
                        truncated["raw_visibility"]["contract"],
                        increment["raw_visibility"]["contract"]}
    if len(linked_contracts) != 1:
        raise ValueError("A verification cannot link runs with different raw-visibility contracts")
    trunc_namespace, _ = read_run(cfg, truncated["run_id"])
    incr_namespace, _ = read_run(cfg, increment["run_id"])
    a = factor_frame(namespace, identity)
    causal = compare_frames(a[a.trade_date.str.replace("-", "").astype(int) <= cut], factor_frame(trunc_namespace, identity))
    parity = compare_frames(a, factor_frame(incr_namespace, identity))
    source_match = full["source_identity_hash"] == truncated["source_identity_hash"] == increment["source_identity_hash"]
    computation_match = full["computation_identity_hash"] == truncated["computation_identity_hash"] == increment["computation_identity_hash"]
    if not source_match or not computation_match:
        raise ValueError("Verification runs do not share exact source and computation identities")
    evidence = {"verification_id": "v_" + uuid.uuid4().hex, "candidate_id": identity,
                "created_at": utc_now(), "reference_run": run_id, "cutoff_run": truncated["run_id"],
                "incremental_run": increment["run_id"], "cutoff": cut, "causal_truncation": causal,
                "full_incremental_parity": parity, "same_source_identity": source_match,
                "same_computation_identity": computation_match,
                "passed": bool(causal["passed"] and parity["passed"] and source_match and computation_match),
                "raw_visibility_contract": next(iter(linked_contracts)),
                "scope": "candidate_and_declared_dependencies;actual_raw_availability_date_truncation_and_exact_incremental_parity",
                "limits": "Caps supported Upstream availability/date reads; direct external/raw-file IO is outside this boundary. Does not reconstruct historical vendor vintages or prove factor profitability"}
    immutable_json(Catalog(cfg).root / "verifications" / (evidence["verification_id"] + ".json"), evidence)
    Catalog(cfg).event(identity, "verification", evidence)
    return evidence
