"""Batch seals, independent evidence, resume, and the shared process budget."""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from fea.config import Config
from research import batch
from research.catalog import Catalog, immutable_json
from research.protocol import Protocol, digest
from research.runner import RAW_VISIBILITY, compare_frames, file_hash, read_run


class BatchWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = Path(__file__).resolve().parents[1] / "artifacts/tests/tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary)
        self.root = Path(self.temp.name).resolve()
        self.cfg = Config({"default_start": "2018-01-01", "paths": {"upstream": "raw"},
                           "resources": {"memory_budget_gib": 90}}, self.root)
        for relative, content in {
            "main.py": "# fixture entry\n", "fea/engine.py": "# fixture engine\n",
            "factors/labels.py": "# labels\n", "conf/config.yaml": "default_start: 2018-01-01\n",
            "conf/frequency.yaml": "datasets: {}\n", "conf/research/protocol.yaml": "{}\n",
            "research/builder.py": "def build(parameters):\n    return lambda ctx: parameters['scale']\n",
            "state/current_release.json": json.dumps({"release_id": "published", "cutoff": "2026-09-30"}),
            "data/factors/published/year=2026/data.parquet": "production fixture"}.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.module_name = "research.batch_builder_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(self.module_name, self.root / "research/builder.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module
        spec.loader.exec_module(module)
        self.folder = self.root / "artifacts/batch"
        self.folder.mkdir(parents=True)
        self.definitions = self.root / "conf/research/definitions"
        self.definitions.mkdir(parents=True)
        idea = {"idea_id": "batch_idea_v1", "hypothesis": "price response", "mechanism": "fixture",
                "expected_direction": "positive", "target_horizons": [1, 5],
                "falsifiable": "parity", "fields": ["stock_daily.close"]}
        family = {"family_id": "batch_family_v1", "idea_id": "batch_idea_v1", "trial_budget": 2,
                  "allowed_variants": [{"scale": 1}, {"scale": 2}], "baseline": "close"}
        candidates = [{"name": f"batch_candidate_{n}", "family_id": "batch_family_v1",
                       "implementation": self.module_name + ":build", "parameters": {"scale": n},
                       "spec": {"group": "batch", "deps": ["stock_daily"], "warmup_days": 180,
                                "formula": "fixture", "desc": "batch fixture"},
                       "units": "price", "availability": "at close", "missing_policy": "nan"} for n in (1, 2)]
        grouped = {"ideas": [idea], "families": [family], "candidates": candidates}
        declarations = {}
        for kind, records in grouped.items():
            declarations[kind] = []
            for number, record in enumerate(records):
                relative = f"{kind}/{number}.yaml"
                path = self.definitions / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record), encoding="utf-8")
                declarations[kind].append(relative)
        self.manifest = {"batch_id": "fixture", "candidate_count": 2, "family_count": 1,
                         "protocol": {"stocks": 2115, "output_start": "2018-01-01",
                                      "primary_horizons": [1, 5], "auxiliary_horizons": [3, 10, 20]},
                         "definition_files": declarations}
        self.write_manifest()
        self.freeze_path = self.folder / "source_freeze.json"
        self.freeze = {"schema_version": 1, "created_at": "fixture", "release_id": "published",
                       "cutoff": "2026-09-30", "stock_count": 2115,
                       "protocol_sha256": file_hash(self.root / "conf/research/protocol.yaml"),
                       "source_sha256": {p.relative_to(self.root).as_posix(): file_hash(p)
                                         for base in ("fea", "factors", "research", "conf")
                                         for p in sorted((self.root / base).rglob("*")) if p.is_file()}}
        self.freeze["source_sha256"]["main.py"] = file_hash(self.root / "main.py")
        self.write_freeze()
        self.ids = batch.register_definitions(self.cfg, self.definitions)
        self.identity = self.ids[0]
        self.sources = {"stock_daily": "raw-daily", "stock_st_info": "physical-st"}
        self.computation = {"objects": {self.identity: {"version": 1, "recipe": "fixture"}},
                            "common_sources": {"fea/engine.py": "original"}, "research_sources": {}}
        self.plan = {"schema_version": 1, "project_root": str(self.root), "config": self.cfg.raw,
                     "candidate_ids": self.ids, "definition_hashes": {
                         identity: Catalog(self.cfg).read("candidates", identity)["definition_hash"] for identity in self.ids},
                     "start": "2018-01-02", "actual_start": 20180102, "end": "2026-09-30",
                     "cutoff": "2024-12-31", "workers": 4, "worker_jobs": 1,
                     "source_freeze": self.freeze, "source_freeze_path": str(self.freeze_path),
                     "protocol_hash": Protocol().identity, "computations": {self.identity: self.computation},
                     "raw_sources": self.sources}
        immutable_json(self.folder / "plan.json", self.plan)
        self.full_count = self.eval_count = self.verify_count = 0
        self.verification_ok = True

    def tearDown(self):
        sys.modules.pop(self.module_name, None)
        self.temp.cleanup()

    def write_manifest(self):
        (self.definitions / "manifest.yaml").write_text(json.dumps(self.manifest), encoding="utf-8")

    def write_freeze(self):
        self.freeze_path.write_text(json.dumps(self.freeze), encoding="utf-8")

    def fixture_run(self, identity, kind="calculation", phase_ends=None):
        run_id = "r_" + uuid.uuid4().hex
        ends = phase_ends or [20260930]
        dates = [20180102, 20241231] + ([20260930] if ends[-1] >= 20260930 else [])
        frame = pd.DataFrame({"trade_date": [f"{d//10000:04d}-{d//100%100:02d}-{d%100:02d}" for d in dates],
                              "stock_code": ["000001.SZ"] * len(dates),
                              "value": np.arange(1, len(dates) + 1, dtype=np.float32),
                              "rank": np.ones(len(dates), dtype=np.float32)})
        namespace = self.root / "artifacts/research/runs" / run_id
        hashes = {}
        for year in sorted({d//10000 for d in dates}):
            path = namespace / f"data/factors/{identity}/year={year}/data.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame[frame.trade_date.str.startswith(str(year))].to_parquet(path, index=False)
            hashes[path.relative_to(namespace).as_posix()] = file_hash(path)
        caches = [f"data/derived/phase_{i}_cutoff_{cut}" for i, cut in enumerate(ends)]
        st_paths = [f"state/st_phase_{i}_{cut}" for i, cut in enumerate(ends)]
        st_records = [{"phase": i, "raw_cutoff": cut} for i, cut in enumerate(ends)]
        st_hashes = {}
        for path, record in zip(st_paths, st_records):
            target = namespace / path / "visible.json"
            immutable_json(target, record)
            st_hashes[path + "/visible.json"] = file_hash(target)
        record = {"run_id": run_id, "status": "complete", "candidate_id": identity, "kind": kind,
                  "start": dates[0], "end": dates[-1], "trading_days": dates, "codes": ["000001.SZ"],
                  "definition_hash": self.plan["definition_hashes"][identity],
                  "protocol_hash": self.plan["protocol_hash"], "sources": self.sources,
                  "source_identity_hash": digest(self.sources), "computation_identity": self.computation,
                  "computation_identity_hash": digest(self.computation), "output_hashes": hashes,
                  "phase_ends": ends, "raw_visibility": {"contract": RAW_VISIBILITY, "phase_cutoffs": ends,
                  "derived_phase_paths": caches, "st_phase_paths": st_paths}, "st_visibility_records": st_records,
                  "st_visibility_hashes": st_hashes, "tasks": [
                      {"phase": i, "raw_cutoff": end, "derived_namespace": caches[i], "st_visibility_namespace": st_paths[i]}
                      for i, end in enumerate(ends)]}
        immutable_json(namespace / "run.json", record)
        return record

    def calculate(self, cfg, identity, start, end, jobs):
        self.assertEqual((start, end, jobs), ("2018-01-02", "2026-09-30", 1))
        self.full_count += 1
        return self.fixture_run(identity)

    def evaluate(self, cfg, run_id):
        self.eval_count += 1
        _, full = read_run(cfg, run_id)
        identity = "e_" + uuid.uuid4().hex
        folder = self.root / "artifacts/research/evaluations" / identity
        folder.mkdir(parents=True)
        pd.DataFrame({"trade_date": ["2024-12-31"], "rankic": [0.01]}).to_parquet(folder / "daily_metrics.parquet")
        evaluation = {"evaluation_id": identity, "candidate_id": full["candidate_id"], "calculation_run_id": run_id,
                      "primary_horizons": [1, 5], "summary": [{"horizon": h, "days": 80, "rankic": 0.01} for h in (1, 5)],
                      "protocol_hash": full["protocol_hash"], "definition_hash": full["definition_hash"],
                      "source_identity_hash": full["source_identity_hash"],
                      "computation_identity_hash": full["computation_identity_hash"], "missing_metric": np.nan,
                      "daily_file_hash": file_hash(folder / "daily_metrics.parquet")}
        immutable_json(folder / "evaluation.json", evaluation)
        return evaluation

    def verify(self, cfg, run_id, jobs, cutoff):
        from research.runner import factor_frame
        self.verify_count += 1
        self.assertEqual((jobs, cutoff), (1, "2024-12-31"))
        namespace, full = read_run(cfg, run_id)
        truncated = self.fixture_run(self.identity, "causal_truncation", [20241231])
        increment = self.fixture_run(self.identity, "incremental_validation", [20241231, 20260930])
        reference = factor_frame(namespace, self.identity)
        causal = compare_frames(reference[reference.trade_date <= "2024-12-31"],
                                factor_frame(read_run(cfg, truncated["run_id"])[0], self.identity))
        parity = compare_frames(reference, factor_frame(read_run(cfg, increment["run_id"])[0], self.identity))
        identity = "v_" + uuid.uuid4().hex
        verification = {"verification_id": identity, "candidate_id": self.identity, "reference_run": run_id,
                        "cutoff_run": truncated["run_id"], "incremental_run": increment["run_id"], "cutoff": 20241231,
                        "causal_truncation": causal, "full_incremental_parity": parity,
                        "raw_visibility_contract": RAW_VISIBILITY, "passed": self.verification_ok}
        immutable_json(self.root / "artifacts/research/verifications" / (identity + ".json"), verification)
        return verification

    @contextlib.contextmanager
    def execution(self):
        # Numerical execution is deliberately a tiny file fixture. All resumed
        # file seals, phase metadata, comparisons and lifecycle gates are real.
        with patch("research.batch._sources", return_value=self.sources), \
             patch("research.runner.run_candidate", side_effect=self.calculate), \
             patch("research.evaluation.evaluate_run", side_effect=self.evaluate), \
             patch("research.runner.verify_candidate", side_effect=self.verify), \
             patch("research.lifecycle._current_evidence_identity", return_value=digest(self.computation)):
            yield

    def test_manifest_registers_idempotently_in_dependency_order(self):
        before = batch.production_inventory(self.cfg)
        self.assertEqual(batch.register_definitions(self.cfg, self.definitions), self.ids)
        self.assertEqual(len(Catalog(self.cfg).records("ideas")), 1)
        self.assertEqual(len(Catalog(self.cfg).records("families")), 1)
        self.assertEqual(len(Catalog(self.cfg).records("candidates")), 2)
        self.assertEqual(batch.production_inventory(self.cfg), before)

    def test_manifest_refuses_wrong_count_or_unlisted_file_or_group(self):
        self.manifest["candidate_count"] = 3
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "candidate_count"):
            batch.register_definitions(self.cfg, self.definitions)
        self.manifest["candidate_count"] = 2
        self.write_manifest()
        rogue = self.definitions / "unlisted.yaml"
        rogue.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file list"):
            batch.register_definitions(self.cfg, self.definitions)
        rogue.unlink()
        self.manifest["definition_files"]["ideas"], self.manifest["definition_files"]["families"] = (
            self.manifest["definition_files"]["families"], self.manifest["definition_files"]["ideas"])
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "wrong group"):
            batch.register_definitions(self.cfg, self.definitions)

    def test_manifest_refuses_escape_and_protocol_change(self):
        self.manifest["definition_files"]["ideas"] = ["../other.yaml"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "escapes"):
            batch.register_definitions(self.cfg, self.definitions)
        self.manifest["definition_files"]["ideas"] = ["ideas/0.yaml"]
        self.manifest["protocol"]["primary_horizons"] = [3]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "protocol"):
            batch.register_definitions(self.cfg, self.definitions)

    def test_freeze_checks_code_coverage_and_actual_bytes(self):
        self.assertEqual(batch.validate_freeze(self.cfg, self.freeze_path), self.freeze)
        del self.freeze["source_sha256"]["factors/labels.py"]
        self.write_freeze()
        with self.assertRaisesRegex(ValueError, "omits"):
            batch.validate_freeze(self.cfg, self.freeze_path)
        self.freeze["source_sha256"]["factors/labels.py"] = file_hash(self.root / "factors/labels.py")
        self.write_freeze()
        (self.root / "fea/engine.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "Frozen source changed"):
            batch.validate_freeze(self.cfg, self.freeze_path)

    def test_freeze_refuses_dirty_or_changed_publication(self):
        dirty = self.cfg.state_dir / "publication_dirty.json"
        dirty.write_text("{}")
        with self.assertRaisesRegex(ValueError, "unfinished or dirty"):
            batch.validate_freeze(self.cfg, self.freeze_path)
        dirty.unlink()
        (self.cfg.state_dir / "current_release.json").write_text('{"release_id":"other","cutoff":"2026-09-30"}')
        with self.assertRaisesRegex(ValueError, "Publication changed|publication changed"):
            batch.validate_freeze(self.cfg, self.freeze_path)

    def test_inventory_tracks_production_mutation_but_ignores_lock_and_atime(self):
        before = batch.production_inventory(self.cfg)
        lock = self.cfg.state_dir / "daily.lock"
        lock.write_text("legal lock")
        path = self.root / "data/factors/published/year=2026/data.parquet"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns + 1_000_000, stat.st_mtime_ns))
        self.assertEqual(batch.production_inventory(self.cfg), before)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        self.assertNotEqual(batch.production_inventory(self.cfg), before)

    def test_complete_and_resume_revalidate_independent_actual_file_evidence(self):
        before = batch.production_inventory(self.cfg)
        with self.execution():
            first = batch.candidate_attempt(self.cfg, self.folder, self.plan, self.identity)
            second = batch.candidate_attempt(self.cfg, self.folder, self.plan, self.identity)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(second["status"], "complete")
        self.assertEqual(second["resumed_from"], first["attempt"])
        self.assertEqual(second["full_run_id"], first["full_run_id"])
        self.assertEqual((self.full_count, self.eval_count, self.verify_count), (1, 1, 1))
        self.assertFalse(second["production_approved"])
        self.assertEqual(batch.production_inventory(self.cfg), before)

    def test_failed_actual_verification_keeps_failure_and_retry_uses_new_trial(self):
        self.verification_ok = False
        with self.execution():
            first = batch.candidate_attempt(self.cfg, self.folder, self.plan, self.identity)
            self.verification_ok = True
            second = batch.candidate_attempt(self.cfg, self.folder, self.plan, self.identity)
        self.assertEqual((first["status"], first["stage"]), ("failed", "verification"))
        self.assertEqual(second["status"], "complete")
        self.assertIsNone(second["resumed_from"])
        self.assertEqual((self.full_count, self.eval_count, self.verify_count), (2, 2, 2))
        attempts = self.folder / "candidates" / self.identity / "attempts"
        self.assertTrue((attempts / first["attempt"] / "failure.json").exists())
        events = Catalog(self.cfg).events(self.identity)
        self.assertTrue(any(event["kind"] == "batch_attempt_failed" for event in events))

    def test_resume_rejects_corrupt_parquet_and_current_computation_change(self):
        with self.execution():
            first = batch.candidate_attempt(self.cfg, self.folder, self.plan, self.identity)
            namespace, full = read_run(self.cfg, first["full_run_id"])
            output = namespace / next(iter(full["output_hashes"]))
            output.write_bytes(b"corrupt output")
            second = batch.candidate_attempt(self.cfg, self.folder, self.plan, self.identity)
            third = batch.candidate_attempt(self.cfg, self.folder, self.plan, self.identity)
        self.assertEqual((second["status"], second["stage"]), ("failed", "full"))
        self.assertEqual(second["resumed_from"], first["attempt"])
        self.assertEqual(third["status"], "complete")
        self.assertIsNone(third["resumed_from"])
        self.assertEqual(self.full_count, 2)
        self.assertIn("output changed", second["error"])

    def test_current_gate_is_rerun_on_resume_and_source_drift_blocks_before_compute(self):
        with self.execution():
            first = batch.candidate_attempt(self.cfg, self.folder, self.plan, self.identity)
            with patch("research.lifecycle._current_evidence_identity", side_effect=ValueError("Current computation changed")):
                second = batch.candidate_attempt(self.cfg, self.folder, self.plan, self.identity)
            (self.root / "fea/engine.py").write_text("# later source\n")
            third = batch.candidate_attempt(self.cfg, self.folder, self.plan, self.identity)
        self.assertEqual(first["status"], "complete")
        self.assertEqual((second["status"], second["stage"]), ("failed", "current_gate"))
        self.assertEqual((third["status"], third["stage"]), ("failed", "contract"))
        self.assertEqual(self.full_count, 1)

    def test_raw_sources_are_rechecked_before_compute(self):
        changed = {**self.sources, "stock_daily": "revised"}
        with self.execution(), patch("research.batch._sources", return_value=changed):
            result = batch.candidate_attempt(self.cfg, self.folder, self.plan, self.identity)
        self.assertEqual((result["status"], result["stage"]), ("failed", "contract"))
        self.assertEqual(self.full_count, 0)

    def test_plan_seals_dates_dependencies_but_workers_can_change_on_resume(self):
        folder = self.root / "artifacts/new_plan"
        args = argparse.Namespace(source_freeze=self.freeze_path, start="2018-01-02", end="2026-09-30",
                                  cutoff="2024-12-31", workers=1)
        observer = MagicMock()
        observer.cal.between.return_value = np.asarray([20180102, 20241231, 20260930])
        observer.baseline_last_day.return_value = 20260930
        with patch("fea.engine.Engine", return_value=observer), \
             patch("fea.processes.upstream_writer_pids", return_value=[]), \
             patch("research.runner.candidate_registry", side_effect=lambda *a: contextlib.nullcontext({})), \
             patch("research.runner.dependency_specs", return_value=[]), \
             patch("research.runner._source_names", return_value=set(self.sources)), \
             patch("research.runner.computation_identity", return_value=self.computation), \
             patch("research.batch._sources", return_value=self.sources):
            first = batch.prepare_plan(self.cfg, folder, self.ids, args)
            args.workers = 4
            second = batch.prepare_plan(self.cfg, folder, self.ids, args)
            args.cutoff = "2024-12-30"
            with self.assertRaisesRegex(ValueError, "actual local trading dates"):
                batch.prepare_plan(self.cfg, folder, self.ids, args)
        self.assertEqual(first, second)
        self.assertEqual(first["worker_jobs"], 1)
        self.assertEqual(first["raw_sources"], self.sources)
        self.assertEqual(set(first["computations"]), set(self.ids))
        self.assertEqual(first["actual_start"], 20180102)

    def test_worker_count_is_bounded_by_requested_cpu_and_project_caps(self):
        with patch("fea.resources.available_cpus", return_value=2):
            self.assertEqual(batch._worker_count(self.cfg, 4), 2)
        self.cfg.raw["resources"]["max_jobs"] = 1
        self.assertEqual(batch._worker_count(self.cfg, 4), 1)
        for invalid in (0, 5, True, 1.5):
            with self.assertRaisesRegex(ValueError, "1 to 4"):
                batch._worker_count(self.cfg, invalid)

    def test_no_resume_does_new_actual_trials_without_altering_the_plan(self):
        before = file_hash(self.folder / "plan.json")
        with self.execution():
            first = batch.candidate_attempt(self.cfg, self.folder, self.plan, self.identity)
            second = batch.candidate_attempt(self.cfg, self.folder, self.plan, self.identity, resume=False)
        self.assertEqual(second["status"], "complete")
        self.assertNotEqual(first["full_run_id"], second["full_run_id"])
        self.assertEqual((self.full_count, self.eval_count, self.verify_count), (2, 2, 2))
        self.assertEqual(file_hash(self.folder / "plan.json"), before)

    def test_one_outer_guard_covers_all_workers_and_inventory_refuses_write(self):
        calls = []
        def bounded(command, *, cwd, stdout, policy):
            calls.append((command, cwd, policy))
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            invocation = Path(command[command.index("--_invocation") + 1])
            immutable_json(invocation / "results.json", {"candidates": [
                {"candidate_id": identity, "status": "complete", "worker_returncode": 0} for identity in self.ids]})
            return {"returncode": 0, "memory_stopped": False, "project_memory_limit_bytes": 90 * 2**30,
                    "memory_guard_bytes": int(81 * 2**30)}
        with patch("fea.resources.run_bounded", side_effect=bounded):
            report = batch.run_batch(self.cfg, self.folder, self.plan)
        self.assertTrue(report["complete"])
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][2], self.cfg)
        self.assertEqual(report["resources"]["memory_guard_bytes"], 81 * 2**30)
        def mutate(*args, **kwargs):
            result = bounded(*args, **kwargs)
            (self.root / "state/unexpected.json").write_text("{}")
            return result
        with patch("fea.resources.run_bounded", side_effect=mutate):
            changed = batch.run_batch(self.cfg, self.folder, self.plan)
        self.assertFalse(changed["complete"])
        self.assertEqual(changed["production_changed_paths"], ["state/unexpected.json"])

    def test_memory_stop_or_nonzero_worker_cannot_claim_complete(self):
        def stopped(command, **kwargs):
            invocation = Path(command[command.index("--_invocation") + 1])
            immutable_json(invocation / "results.json", {"candidates": [
                {"candidate_id": identity, "status": "complete", "worker_returncode": 1} for identity in self.ids]})
            return {"returncode": 0, "memory_stopped": False}
        with patch("fea.resources.run_bounded", side_effect=stopped):
            self.assertFalse(batch.run_batch(self.cfg, self.folder, self.plan)["complete"])
        with patch("fea.resources.run_bounded", return_value={"returncode": 143, "memory_stopped": True}):
            report = batch.run_batch(self.cfg, self.folder, self.plan)
        self.assertFalse(report["complete"])
        self.assertTrue(report["production_unchanged"])

    def test_bounded_exception_leaves_inventory_and_failure_evidence(self):
        with patch("fea.resources.run_bounded", side_effect=RuntimeError("process group killed")):
            with self.assertRaisesRegex(RuntimeError, "group killed"):
                batch.run_batch(self.cfg, self.folder, self.plan)
        invocation = next((self.folder / "invocations").iterdir())
        self.assertTrue((invocation / "failure.json").exists())
        self.assertEqual(batch._read(invocation / "production_before.json"), batch._read(invocation / "production_after.json"))

    def test_coordinator_limits_active_workers_and_nonzero_exit_fails_batch(self):
        invocation = self.folder / "invocations/coordinate"
        invocation.mkdir(parents=True)
        immutable_json(invocation / "request.json", {"plan_hash": digest(self.plan), "workers": 1})
        count = {"active": 0, "peak": 0}
        class Process:
            def __init__(this, code):
                this.code, this.polls, this.done = code, 0, False
                count["active"] += 1
                count["peak"] = max(count["peak"], count["active"])
            def poll(this):
                this.polls += 1
                if this.polls == 1:
                    return None
                if not this.done:
                    count["active"] -= 1
                    this.done = True
                return this.code
        commands = []
        def spawn(command, cfg, stream):
            commands.append(command)
            identity = command[command.index("--_candidate-id") + 1]
            immutable_json(invocation / (identity + ".json"), {"candidate_id": identity, "status": "complete"})
            return Process(7 if identity == self.ids[-1] else 0)
        with patch("research.batch._spawn_worker", side_effect=spawn), patch("research.batch.time.sleep"):
            self.assertEqual(batch._coordinate(self.folder / "plan.json", invocation), 1)
        self.assertEqual(count, {"active": 0, "peak": 1})
        self.assertEqual(len(commands), len(self.ids))
        records = batch._read(invocation / "results.json")["candidates"]
        self.assertEqual([r["worker_returncode"] for r in records], [0, 7])

    def test_coordinator_interrupt_terminates_active_group_children(self):
        invocation = self.folder / "invocations/interrupt"
        invocation.mkdir(parents=True)
        immutable_json(invocation / "request.json", {"plan_hash": digest(self.plan), "workers": 1})
        process = MagicMock()
        process.poll.return_value = None
        process.wait.return_value = -15
        with patch("research.batch._spawn_worker", return_value=process), \
             patch("research.batch.time.sleep", side_effect=InterruptedError("stop")):
            with self.assertRaisesRegex(InterruptedError, "stop"):
                batch._coordinate(self.folder / "plan.json", invocation)
        process.terminate.assert_called_once()
        process.wait.assert_called_once()
        self.assertFalse((invocation / "results.json").exists())

    def test_registration_cli_does_not_calculate_and_refuses_production_out(self):
        parser = argparse.ArgumentParser()
        batch.add_arguments(parser)
        args = parser.parse_args(["--definitions-dir", str(self.definitions), "--register-only", "--out", str(self.folder)])
        with contextlib.redirect_stdout(io.StringIO()), patch("research.batch.run_batch") as runner:
            self.assertEqual(batch.cli(args, self.cfg), 0)
        runner.assert_not_called()
        args.out = self.root / "data/batch"
        with self.assertRaisesRegex(ValueError, "artifacts namespace"):
            batch.cli(args, self.cfg)
        self.assertFalse(args.out.exists())

    @unittest.skipIf(os.name == "nt", "Linux process groups are guarded on the production host")
    def test_actual_worker_inherits_the_guarded_process_group(self):
        output = self.folder / "group.log"
        with output.open("w") as stream:
            process = batch._spawn_worker([sys.executable, "-c", "import os; print(os.getpgrp())"], self.cfg, stream)
            self.assertEqual(process.wait(timeout=10), 0)
        self.assertEqual(int(output.read_text().strip()), os.getpgrp())


if __name__ == "__main__":
    unittest.main()
