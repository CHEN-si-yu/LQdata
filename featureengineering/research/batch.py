"""Bound existing immutable research trials under one process-group budget.

The coordinator never substitutes shared outputs for actual full/cut/increment
calculations. Each worker calls the existing runner/evaluator/current gate with
jobs=1. Only a completed, revalidated immutable checkpoint may be resumed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "research"

from .catalog import Catalog, clean_json, immutable_json, load_definition, slug
from .protocol import digest, load_protocol, utc_now
from .runner import file_hash


def _local(cfg, value):
    path = Path(value)
    path = (path if path.is_absolute() else cfg.root / path).resolve()
    if not path.is_relative_to(cfg.root.resolve()):
        raise ValueError(f"Batch path escapes project: {value}")
    return path


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _date(value):
    text = str(value)
    result = dt.date.fromisoformat(text)
    if result.isoformat() != text:
        raise ValueError("Use YYYY-MM-DD dates")
    return text


def production_inventory(cfg):
    """Metadata-only proof; reading atime and the legal shared lock are ignored."""
    inventory = {}
    lock = (cfg.state_dir / "daily.lock").resolve()
    for root in (cfg.root / "data", cfg.state_dir):
        for path in sorted(root.rglob("*")) if root.exists() else []:
            if not path.is_file() or path.resolve() == lock:
                continue
            actual = _local(cfg, path)
            stat = actual.stat()
            inventory[actual.relative_to(cfg.root.resolve()).as_posix()] = [stat.st_size, stat.st_mtime_ns]
    return inventory


def validate_freeze(cfg, path):
    frozen = _read(_local(cfg, path))
    if frozen.get("schema_version") != 1 or frozen.get("stock_count") != 2115:
        raise ValueError("Invalid fixed-pool source freeze")
    hashes = frozen.get("source_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Source freeze needs project-relative SHA256 entries")
    required = set()
    for directory in ("fea", "factors", "research"):
        required.update(p.relative_to(cfg.root).as_posix()
                        for p in (cfg.root / directory).rglob("*.py"))
    for path in (cfg.root / "main.py", cfg.root / "conf/config.yaml",
                 cfg.input_contract_path("frequency", "conf/frequency.yaml"),
                 cfg.root / "conf/research/protocol.yaml", cfg.frozen_universe):
        if path and path.exists():
            required.add(_local(cfg, path).relative_to(cfg.root.resolve()).as_posix())
    if not required <= set(hashes):
        raise ValueError("Source freeze omits running code/contracts: " + ",".join(sorted(required - set(hashes))))
    for relative, expected in hashes.items():
        if Path(relative).is_absolute() or _local(cfg, relative).relative_to(cfg.root.resolve()).as_posix() != relative:
            raise ValueError("Freeze keys must be canonical project-relative paths")
        if file_hash(_local(cfg, relative)) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    protocol = cfg.root / "conf/research/protocol.yaml"
    if not protocol.exists() or file_hash(protocol) != frozen.get("protocol_sha256"):
        raise ValueError("Frozen protocol bytes changed")
    if any((cfg.state_dir / name).exists() for name in ("daily_inflight.json", "publication_dirty.json")):
        raise ValueError("Production publication is unfinished or dirty; complete main.py before research")
    pointer = _read(cfg.state_dir / "current_release.json")
    if pointer.get("release_id") != frozen.get("release_id") or pointer.get("cutoff") != frozen.get("cutoff"):
        raise ValueError("Production publication changed since source freeze")
    _date(frozen["cutoff"])
    return frozen


def register_definitions(cfg, folder):
    """Register ideas, then families, then candidates; no production mutation."""
    folder = _local(cfg, folder)
    groups = {"ideas": [], "families": [], "candidates": []}
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".json", ".yaml", ".yml"))
    if not files:
        raise ValueError("No batch definitions found")
    manifests = [p for p in files if p.stem == "manifest"]
    manifest, declared_kinds = None, {}
    if manifests:
        if len(manifests) != 1:
            raise ValueError("Batch requires exactly one definition manifest")
        manifest = load_definition(manifests[0])
        declarations = manifest.get("definition_files") if isinstance(manifest, dict) else None
        if not isinstance(declarations, dict) or set(declarations) != set(groups):
            raise ValueError("Manifest definition_files must declare ideas, families and candidates")
        declared = []
        for kind, names in declarations.items():
            if not isinstance(names, list) or not names:
                raise ValueError(f"Manifest {kind} must be a nonempty path list")
            for name in names:
                if not isinstance(name, str) or Path(name).is_absolute():
                    raise ValueError("Manifest definitions must use relative paths")
                path = _local(cfg, folder / name)
                if not path.is_relative_to(folder) or path == folder or path.suffix.lower() not in (".json", ".yaml", ".yml"):
                    raise ValueError("Manifest definition path escapes its batch directory")
                if path in declared_kinds:
                    raise ValueError("Manifest lists a definition more than once")
                declared_kinds[path] = kind
                declared.append(path)
        if set(declared) != {p.resolve() for p in files if p not in manifests}:
            raise ValueError("Manifest file list differs from actual batch definitions")
        files = declared
    for path in files:
        payload = load_definition(_local(cfg, path))
        if isinstance(payload, dict) and set(payload) <= set(groups) and payload:
            for kind, records in payload.items():
                if not isinstance(records, list):
                    raise ValueError(f"Definition group {kind} must be a list")
                groups[kind].extend(records)
        elif isinstance(payload, dict) and "implementation" in payload:
            groups["candidates"].append(payload)
        elif isinstance(payload, dict) and "allowed_variants" in payload:
            groups["families"].append(payload)
        elif isinstance(payload, dict) and "hypothesis" in payload:
            groups["ideas"].append(payload)
        else:
            raise ValueError(f"Unknown batch definition: {path.name}")
        if manifest:
            kind = declared_kinds[path]
            expected = {"ideas": "hypothesis", "families": "allowed_variants", "candidates": "implementation"}[kind]
            if not isinstance(payload, dict) or expected not in payload:
                raise ValueError(f"Manifest definition has the wrong group: {path.name}")
    if manifest:
        for kind, key in (("ideas", "idea_count"), ("families", "family_count"), ("candidates", "candidate_count")):
            if (key in manifest or kind != "ideas") and manifest.get(key) != len(groups[kind]):
                raise ValueError(f"Manifest {key} differs from actual definitions")
        if len({r["name"] for r in groups["candidates"]}) != len(groups["candidates"]):
            raise ValueError("Batch candidate names must be unique")
        declared_protocol = manifest.get("protocol", {})
        protocol = load_protocol(cfg)
        expected_protocol = {"stocks": protocol.expected_stocks, "output_start": protocol.output_start,
                             "primary_horizons": list(protocol.primary_horizons),
                             "auxiliary_horizons": list(protocol.auxiliary_horizons)}
        if not isinstance(declared_protocol, dict) or declared_protocol != expected_protocol:
            raise ValueError("Manifest protocol differs from the current frozen research protocol")
    catalog = Catalog(cfg)
    for data in groups["ideas"]:
        catalog.register_idea(data)
    for data in groups["families"]:
        catalog.register_family(data)
    return [catalog.register_candidate(data)["candidate_id"] for data in groups["candidates"]]


def _candidate_ids(cfg, args, folder):
    if args.definitions_dir:
        identities = register_definitions(cfg, args.definitions_dir)
    else:
        payload = _read(_local(cfg, args.candidate_list))
        identities = payload if isinstance(payload, list) else payload.get("candidate_ids")
    if not isinstance(identities, list) or not identities or len(set(identities)) != len(identities):
        raise ValueError("Candidate list must contain unique immutable candidate IDs")
    for identity in identities:
        Catalog(cfg).read("candidates", slug(identity))
    immutable_json(folder / "candidate_list.json", {"candidate_ids": identities})
    return identities


def _cfg(plan):
    from fea.config import Config
    return Config(raw=plan["config"], root=Path(plan["project_root"]))


def _sources(cfg, folder, names):
    from fea.engine import Engine
    observer = Engine(cfg.sandbox(folder / "preflight"))
    return {name: observer.watermark(name) for name in names}


def _worker_count(cfg, requested):
    from fea.resources import available_cpus, resource_policy
    if not isinstance(requested, int) or isinstance(requested, bool) or not 1 <= requested <= 4:
        raise ValueError("Batch supports 1 to 4 independent jobs=1 workers")
    return min(requested, available_cpus(), int(resource_policy(cfg)["max_jobs"]))


def prepare_plan(cfg, folder, identities, args):
    from .catalog import spec_for_candidate
    from .runner import candidate_registry, dependency_specs, computation_identity, _source_names
    from fea.engine import Engine
    from fea.processes import upstream_writer_pids
    if cfg.raw.get("_sandbox") or cfg.contract_root != cfg.root.resolve():
        raise ValueError("Batch coordinates private trials from the production project config")
    frozen = validate_freeze(cfg, args.source_freeze)
    start, end, cutoff = map(_date, (args.start, args.end, args.cutoff))
    if not cfg.default_start <= start <= cutoff < end <= frozen["cutoff"]:
        raise ValueError("Batch dates must remain within the fixed floor/freeze with an earlier cutoff")
    _worker_count(cfg, args.workers)
    if upstream_writer_pids(cfg.upstream):
        raise RuntimeError("Upstream raw writer is active; freeze before research")
    protocol = load_protocol(cfg)
    observer = Engine(cfg.sandbox(folder / "preflight"))
    start_i, end_i, cut_i = (int(d.replace("-", "")) for d in (start, end, cutoff))
    days = observer.cal.between(start_i, end_i)
    if not len(days) or int(days[-1]) != end_i or cut_i not in set(days.tolist()):
        raise ValueError("Batch end/cutoff must be actual local trading dates")
    if observer.baseline_last_day() < end_i:
        raise ValueError("Raw daily source does not reach the requested end")
    names, computations, definitions = set(), {}, {}
    for identity in identities:
        spec = spec_for_candidate(cfg, identity)
        with candidate_registry(spec, cfg) as registry:
            specs = dependency_specs(spec, registry, protocol.horizons)
            names.update(_source_names(specs))
            computations[identity] = computation_identity(cfg, observer, specs)
        definitions[identity] = Catalog(cfg).read("candidates", identity)["definition_hash"]
    plan = {"schema_version": 1, "project_root": str(cfg.root.resolve()), "config": cfg.raw,
            "candidate_ids": identities, "definition_hashes": definitions,
            "start": start, "actual_start": int(days[0]), "end": end, "cutoff": cutoff,
            "worker_limit": 4, "worker_jobs": 1,
            "source_freeze_path": str(_local(cfg, args.source_freeze)), "source_freeze": frozen,
            "protocol_hash": protocol.identity, "computations": computations,
            "raw_sources": _sources(cfg, folder, sorted(names)),
            "scope": "independent_actual_full_cut_increment;no_production_mutation;no_promotion"}
    immutable_json(folder / "plan.json", plan)
    return plan


def _check_plan(cfg, folder, plan):
    if validate_freeze(cfg, plan["source_freeze_path"]) != plan["source_freeze"]:
        raise ValueError("Frozen batch contract changed")
    if load_protocol(cfg).identity != plan["protocol_hash"]:
        raise ValueError("Current protocol differs from batch plan")
    if _sources(cfg, folder, sorted(plan["raw_sources"])) != plan["raw_sources"]:
        raise ValueError("Physical raw sources changed since batch planning")


def _validate_full(cfg, identity, full, plan):
    from .runner import read_run, validate_raw_visibility
    _, actual = read_run(cfg, full["run_id"])
    validate_raw_visibility(actual)
    if actual != full:
        raise ValueError("Saved full checkpoint differs from immutable trial")
    if (actual["candidate_id"] != identity or actual["kind"] not in ("calculation", "full_validation") or
            actual["start"] != plan["actual_start"] or actual["end"] != int(plan["end"].replace("-", "")) or
            actual["definition_hash"] != plan["definition_hashes"][identity] or
            actual["protocol_hash"] != plan["protocol_hash"] or
            actual["computation_identity"] != plan["computations"][identity] or
            actual["computation_identity_hash"] != digest(plan["computations"][identity]) or
            actual["source_identity_hash"] != digest(actual["sources"])):
        raise ValueError("Completed full trial no longer matches immutable batch plan")
    if any(name not in plan["raw_sources"] or source != plan["raw_sources"][name]
           for name, source in actual["sources"].items()):
        raise ValueError("Candidate full trial used different physical raw sources")


def candidate_attempt(cfg, folder, plan, identity, *, resume=True):
    """One attempt, with immutable checkpoints; failed attempts are never reused."""
    from .runner import run_candidate, verify_candidate
    from .evaluation import evaluate_run, read_evaluation
    from .lifecycle import _check_evidence, read_verification
    attempts = folder / "candidates" / slug(identity) / "attempts"
    attempt_id = "a_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S%f") + "_" + uuid.uuid4().hex[:12]
    attempt = attempts / attempt_id
    immutable_json(attempt / "begin.json", {"candidate_id": identity, "created_at": utc_now(), "plan_hash": digest(plan)})
    stage = "contract"
    previous = None
    try:
        _check_plan(cfg, folder, plan)
        if resume:
            invalidated = {_read(p / "failure.json").get("resumed_from") for p in attempts.iterdir()
                           if (p / "failure.json").exists()}
            previous = next((p for p in sorted(attempts.iterdir(), reverse=True)
                             if p != attempt and p.name not in invalidated and not (p / "failure.json").exists()
                             and (p / "full.json").exists()), None)
        stage = "full"
        full = clean_json(_read(previous / "full.json") if previous else
                          run_candidate(cfg, identity, plan["start"], plan["end"], jobs=1))
        _validate_full(cfg, identity, full, plan)
        immutable_json(attempt / "full.json", full)
        stage = "evaluation"
        if previous and (previous / "evaluation.json").exists():
            evaluation = _read(previous / "evaluation.json")
            if clean_json(read_evaluation(cfg, evaluation["evaluation_id"])) != evaluation:
                raise ValueError("Saved evaluation checkpoint changed")
        else:
            evaluation = clean_json(evaluate_run(cfg, full["run_id"]))
        if evaluation["calculation_run_id"] != full["run_id"]:
            raise ValueError("Evaluation references a different full trial")
        immutable_json(attempt / "evaluation.json", evaluation)
        stage = "verification"
        if previous and (previous / "verification.json").exists():
            verification = _read(previous / "verification.json")
            if clean_json(read_verification(cfg, verification["verification_id"])) != verification:
                raise ValueError("Saved verification checkpoint changed")
        else:
            verification = clean_json(verify_candidate(cfg, full["run_id"], jobs=1, cutoff=plan["cutoff"]))
        immutable_json(attempt / "verification.json", verification)
        if not verification["passed"]:
            raise ValueError("Actual causal/full-incremental verification failed")
        stage = "current_gate"
        evidence = _check_evidence(cfg, identity, evaluation["evaluation_id"], verification["verification_id"])
        _check_plan(cfg, folder, plan)
        immutable_json(attempt / "current_gate.json", {"checked_at": utc_now(), "evidence": evidence,
                                                       "source_freeze_hash": digest(plan["source_freeze"])})
        result = {"candidate_id": identity, "status": "complete", "attempt": attempt_id,
                  "full_run_id": full["run_id"], "evaluation_id": evaluation["evaluation_id"],
                  "verification_id": verification["verification_id"], "current_gate_passed": True,
                  "resumed_from": previous.name if previous else None, "production_approved": False}
        immutable_json(attempt / "result.json", result)
        return result
    except Exception as exc:
        failure = {"candidate_id": identity, "status": "failed", "attempt": attempt_id,
                   "resumed_from": previous.name if previous else None,
                   "stage": stage, "failed_at": utc_now(), "error": str(exc), "traceback": traceback.format_exc()}
        immutable_json(attempt / "failure.json", failure)
        Catalog(cfg).event(identity, "batch_attempt_failed", failure)
        return failure


def _spawn_worker(command, cfg, stream):
    # Inherit the one coordinator process group monitored by run_bounded.
    return subprocess.Popen(command, cwd=cfg.contract_root, stdout=stream, stderr=subprocess.STDOUT,
                            start_new_session=False)


def _coordinate(plan_path, invocation, *, resume=True):
    plan = _read(plan_path)
    cfg, folder = _cfg(plan), Path(plan_path).parent
    _local(cfg, plan_path)
    _local(cfg, invocation)
    request = _read(invocation / "request.json")
    if request["plan_hash"] != digest(plan):
        raise ValueError("Coordinator plan differs from its immutable invocation")
    workers = _worker_count(cfg, request["workers"])
    pending, active, records = list(plan["candidate_ids"]), {}, []
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while pending or active:
            if stopping:
                raise InterruptedError("Batch coordinator interrupted")
            while pending and len(active) < workers:
                identity = pending.pop(0)
                log_path = invocation / (identity + ".log")
                stream = log_path.open("x", encoding="utf-8")
                command = [sys.executable, "-B", str(Path(__file__).resolve()), "--_worker-plan", str(plan_path),
                           "--_candidate-id", identity, "--_invocation", str(invocation)]
                if not resume:
                    command.append("--_no-resume")
                try:
                    process = _spawn_worker(command, cfg, stream)
                except BaseException:
                    stream.close()
                    raise
                active[identity] = process, stream
            for identity, (process, stream) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                result_path = invocation / (identity + ".json")
                record = _read(result_path) if result_path.exists() else {
                    "candidate_id": identity, "status": "failed", "error": "Worker exited without immutable result"}
                record = {**record, "worker_returncode": code}
                records.append(record)
                del active[identity]
            time.sleep(0.1)
        immutable_json(invocation / "results.json", {"candidates": records, "created_at": utc_now()})
        return 0 if all(row["status"] == "complete" and row["worker_returncode"] == 0 for row in records) else 1
    finally:
        for process, _ in active.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 5
        for process, stream in active.values():
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def run_batch(cfg, folder, plan, *, resume=True, workers=4):
    from fea.resources import run_bounded
    workers = _worker_count(cfg, workers)
    if _read(folder / "plan.json") != clean_json(plan):
        raise ValueError("Batch plan differs from its immutable saved plan")
    invocation = folder / "invocations" / ("b_" + uuid.uuid4().hex)
    invocation.mkdir(parents=True, exist_ok=False)
    before = production_inventory(cfg)
    immutable_json(invocation / "production_before.json", before)
    immutable_json(invocation / "request.json", {"resume": resume, "workers": workers, "worker_jobs": 1,
                                               "plan_hash": digest(plan)})
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--_coordinate-plan", str(folder / "plan.json"),
               "--_invocation", str(invocation)]
    if not resume:
        command.append("--_no-resume")
    thread_keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS")
    old_env = {key: os.environ.get(key) for key in thread_keys}
    for key in thread_keys:
        os.environ[key] = "1"
    try:
        with (invocation / "coordinator.log").open("x", encoding="utf-8") as stream:
            bounded = run_bounded(command, cwd=cfg.contract_root, stdout=stream, policy=cfg)
    except BaseException as exc:
        immutable_json(invocation / "failure.json", {"error": str(exc), "failed_at": utc_now(),
                                                     "traceback": traceback.format_exc()})
        raise
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        after = production_inventory(cfg)
        immutable_json(invocation / "production_after.json", after)
    records = _read(invocation / "results.json") if (invocation / "results.json").exists() else {"candidates": []}
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    report = {"schema_version": 1, "created_at": utc_now(), "invocation": invocation.name,
              "plan_hash": digest(plan), "resources": bounded, **records,
              "workers": workers, "worker_jobs": 1, "resume": resume,
              "production_unchanged": not changed, "production_changed_paths": changed,
              "complete": not changed and bounded["returncode"] == 0 and not bounded.get("memory_stopped") and
                          len(records["candidates"]) == len(plan["candidate_ids"]) and
                          all(row["status"] == "complete" and row.get("worker_returncode") == 0
                              for row in records["candidates"]),
              "production_approved": False}
    immutable_json(invocation / "batch.json", report)
    return report


def add_arguments(parser):
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--definitions-dir", type=Path)
    selection.add_argument("--candidate-list", type=Path)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--start", default="2018-01-02")
    parser.add_argument("--end", default="2026-09-30")
    parser.add_argument("--cutoff", default="2024-12-31")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source-freeze", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")


def cli(args, cfg):
    folder = _local(cfg, args.out)
    artifacts = cfg.project_path("artifacts", "artifacts")
    if folder == artifacts or not folder.is_relative_to(artifacts):
        raise ValueError("Keep the batch in its own project artifacts namespace")
    folder.mkdir(parents=True, exist_ok=True)
    identities = _candidate_ids(cfg, args, folder)
    if args.register_only:
        result = {"registered_candidates": identities, "candidate_list": str(folder / "candidate_list.json"),
                  "calculated": False, "production_approved": False}
    else:
        if not args.source_freeze:
            raise ValueError("Run requires --source-freeze after registration/code freeze")
        plan = prepare_plan(cfg, folder, identities, args)
        result = run_batch(cfg, folder, plan, resume=not args.no_resume, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if args.register_only or result["complete"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--_coordinate-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_candidate-id", help=argparse.SUPPRESS)
    parser.add_argument("--_invocation", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_no-resume", action="store_true", help=argparse.SUPPRESS)
    arguments = sys.argv[1:] if argv is None else argv
    if any(flag in arguments for flag in ("--_coordinate-plan", "--_worker-plan")):
        args = parser.parse_args(arguments)
        from fea.config import ROOT
        plan_path = (args._coordinate_plan or args._worker_plan).resolve()
        if not plan_path.is_relative_to(ROOT.resolve()):
            raise ValueError("Internal batch plan must remain inside the project")
        if not args._invocation:
            raise ValueError("Internal batch invocation is required")
        checked_plan = _read(plan_path)
        if Path(checked_plan["project_root"]).resolve() != ROOT.resolve():
            raise ValueError("Internal batch project root differs from its running source project")
        if args._coordinate_plan:
            return _coordinate(plan_path, args._invocation, resume=not args._no_resume)
        plan = _read(args._worker_plan)
        cfg = _cfg(plan)
        _local(cfg, args._worker_plan)
        _local(cfg, args._invocation)
        if _read(args._invocation / "request.json")["plan_hash"] != digest(plan):
            raise ValueError("Worker plan differs from its immutable invocation")
        if args._candidate_id not in plan["candidate_ids"]:
            raise ValueError("Worker candidate is outside immutable plan")
        from fea.resources import configure
        configure(cfg)
        import tempfile
        temporary = Path(args._worker_plan).parent / "worker_tmp" / slug(args._candidate_id)
        temporary.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(temporary)
        tempfile.tempdir = str(temporary)
        result = candidate_attempt(cfg, args._worker_plan.parent, plan, args._candidate_id,
                                   resume=not args._no_resume)
        immutable_json(args._invocation / (args._candidate_id + ".json"), result)
        return 0 if result["status"] == "complete" else 1
    add_arguments(parser)
    args = parser.parse_args(arguments)
    from fea.config import load
    return cli(args, load())


if __name__ == "__main__":
    raise SystemExit(main())
