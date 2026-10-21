"""Real immutable candidate definitions and temporary transitive registration."""
import importlib.util
import json
import logging
import sys
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from fea.config import Config
from fea.spec import FactorSpec, REGISTRY
from research.catalog import Catalog, spec_for_candidate
from research.protocol import Protocol
from research.runner import (_research_phase_engine, candidate_registry, compare_frames, dependency_output_starts,
                             dependency_specs, factor_frame, read_run,
                             resolve_candidate_closure, run_candidate,
                             validate_raw_visibility, verify_candidate)


class ResearchDependencyTests(unittest.TestCase):
    def setUp(self):
        import factors  # normal production import completes before approved candidates are loaded
        base = Path(__file__).resolve().parents[1] / "artifacts" / "tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base, prefix="")
        self.root = Path(self.temp.name)
        self.cfg = Config({"default_start": "2018-01-01", "paths": {"upstream": "raw", "frequency": "conf/frequency.yaml"}}, self.root)
        (self.root / "conf").mkdir()
        (self.root / "conf" / "frequency.yaml").write_text(yaml.safe_dump({"datasets": {"stock_daily": {}, "stock_adj_factor": {}}}), encoding="utf-8")
        for source in ("stock_daily", "stock_adj_factor"):
            path = self.root / "raw" / source
            path.mkdir(parents=True)
            pd.DataFrame({"trade_date": ["2026-01-02"], "stock_code": ["000001.SZ"], "value": [1.]}).to_parquet(path / "data.parquet")
        folder = self.root / "research"
        folder.mkdir()
        self.module_name = "research.test_builders_" + uuid.uuid4().hex
        code = '''import numpy as np
from fea.mathx import roll_mean
def parent(parameters):
    def compute(ctx):
        return np.asarray(ctx.hfq("close"),dtype=np.float64)
    return compute
def child(parameters):
    def compute(ctx):
        return np.asarray(ctx.load_factor(parameters["parent"]),dtype=np.float64)*parameters["scale"]
    return compute
def rolling_parent(parameters):
    def compute(ctx):
        return roll_mean(np.asarray(ctx.hfq("close"),dtype=np.float64),parameters["window"])
    return compute
def rolling_child(parameters):
    def compute(ctx):
        return roll_mean(np.asarray(ctx.load_factor(parameters["parent"]),dtype=np.float64),parameters["window"])
    return compute
def future_peek(parameters):
    def compute(ctx):
        ctx.accessed.add("stock_daily")
        # Deliberately invalid: broadcasts the last globally visible raw row
        # over the earlier panel. Projection omits the availability column.
        data = ctx.up.read("stock_daily",columns=["stock_code","close"])
        last = data.groupby("stock_code",sort=False).tail(1).set_index("stock_code")["close"]
        values = last.reindex(ctx.panel.codes).to_numpy(dtype=np.float64)
        return np.broadcast_to(values,ctx.panel.shape).copy()
    return compute
'''
        filename = folder / "builders.py"
        filename.write_text(code, encoding="utf-8")
        module_spec = importlib.util.spec_from_file_location(self.module_name, filename)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[self.module_name] = module
        module_spec.loader.exec_module(module)
        self.catalog = Catalog(self.cfg)
        self.catalog.register_idea({"idea_id": "closure_idea_v1", "hypothesis": "parent-child reuse", "mechanism": "scaling",
                                    "expected_direction": "positive", "target_horizons": [1, 5],
                                    "falsifiable": "exact identity", "fields": ["stock_daily.close"]})
        self.catalog.register_family({"family_id": "parent_family_v1", "idea_id": "closure_idea_v1", "trial_budget": 1,
                                      "allowed_variants": [{}], "baseline": "raw close"})
        parent = self.catalog.register_candidate(self.definition("parent_v1", "parent_family_v1", "parent", {}, ["stock_daily", "stock_adj_factor"]))
        self.parent_id = parent["candidate_id"]
        self.catalog.register_family({"family_id": "child_family_v1", "idea_id": "closure_idea_v1", "trial_budget": 1,
                                      "allowed_variants": [{"parent": self.parent_id, "scale": 2}], "baseline": "immutable parent"})
        child = self.catalog.register_candidate(self.definition("child_v1", "child_family_v1", "child", {"parent": self.parent_id, "scale": 2}, [self.parent_id]))
        self.child_id = child["candidate_id"]
        self.child = spec_for_candidate(self.cfg, self.child_id)

    def definition(self, name, family, builder, parameters, deps):
        return {"name": name, "family_id": family, "implementation": self.module_name + ":" + builder,
                "parameters": parameters, "spec": {"group": "research", "deps": deps, "formula": builder,
                "desc": "closure fixture", "warmup_days": 40}, "units": "price",
                "availability": "at close", "missing_policy": "nan"}

    def tearDown(self):
        REGISTRY.pop(self.parent_id, None)
        REGISTRY.pop(self.child_id, None)
        sys.modules.pop(self.module_name, None)
        for handler in list(logging.getLogger().handlers):
            filename = getattr(handler, "baseFilename", None)
            if filename and Path(filename).resolve().is_relative_to(self.root):
                logging.getLogger().removeHandler(handler)
                handler.close()
        self.temp.cleanup()

    def test_private_parent_loads_and_computes_before_child_without_production_adoption(self):
        original_names = set(REGISTRY)
        with candidate_registry(self.child, self.cfg) as registry:
            ordered = dependency_specs(self.child, registry, (1, 5))
            names = [s.name for s in ordered]
            self.assertLess(names.index(self.parent_id), names.index(self.child_id))
            values = np.arange(8., dtype=np.float64).reshape(4, 2)
            parent_ctx = type("ParentContext", (), {"hfq": lambda _, field: values})()
            parent_value = registry[self.parent_id].fn(parent_ctx)
            child_ctx = type("ChildContext", (), {"load_factor": lambda _, name: parent_value})()
            np.testing.assert_equal(registry[self.child_id].fn(child_ctx), values * 2)
        # Built-in imports are allowed; private candidates must disappear.
        self.assertNotIn(self.parent_id, REGISTRY)
        self.assertNotIn(self.child_id, REGISTRY)
        self.assertTrue(original_names <= set(REGISTRY))
        self.assertFalse((self.root / "conf" / "production.json").exists())

    def test_existing_approved_parent_is_reloaded_then_original_registration_restored(self):
        original = spec_for_candidate(self.cfg, self.parent_id)
        REGISTRY[self.parent_id] = original
        with candidate_registry(self.child, self.cfg) as registry:
            self.assertIn(self.parent_id, registry)
            self.assertEqual(registry[self.parent_id].recipe(self.cfg), original.recipe(self.cfg))
        self.assertIs(REGISTRY[self.parent_id], original)
        self.assertNotIn(self.child_id, REGISTRY)

    def test_same_name_with_a_different_immutable_definition_is_rejected(self):
        original = replace(spec_for_candidate(self.cfg, self.parent_id), version=2)
        REGISTRY[self.parent_id] = original
        with self.assertRaisesRegex(ValueError, "collision"):
            with candidate_registry(self.child, self.cfg):
                pass
        self.assertIs(REGISTRY[self.parent_id], original)
        self.assertNotIn(self.child_id, REGISTRY)

    def test_unknown_dependency_is_not_guessed_as_raw_source(self):
        changed = replace(self.child, deps=("unknown_parent_v1",))
        with self.assertRaisesRegex(ValueError, "Unknown dependency"):
            resolve_candidate_closure(self.cfg, changed, REGISTRY)

    def test_raw_contract_without_actual_data_is_rejected(self):
        (self.root / "raw" / "stock_daily" / "data.parquet").unlink()
        with self.assertRaisesRegex(ValueError, "no local Parquet"):
            resolve_candidate_closure(self.cfg, self.child, REGISTRY)

    def test_future_labels_cannot_be_parent_inputs(self):
        import factors
        changed = replace(self.child, deps=("label_ret_1d",))
        with self.assertRaisesRegex(ValueError, "future labels"):
            resolve_candidate_closure(self.cfg, changed, REGISTRY)

    def test_registered_factor_cycle_is_rejected_before_temporary_mutation(self):
        a = "research_cycle_a_" + uuid.uuid4().hex
        b = "research_cycle_b_" + uuid.uuid4().hex
        REGISTRY[a] = FactorSpec(name=a, group="test", deps=(b,), fn=lambda ctx: None)
        REGISTRY[b] = FactorSpec(name=b, group="test", deps=(a,), fn=lambda ctx: None)
        try:
            changed = replace(self.child, deps=(a,))
            with self.assertRaisesRegex(ValueError, "cycle"):
                resolve_candidate_closure(self.cfg, changed, REGISTRY)
            self.assertNotIn(self.child_id, REGISTRY)
        finally:
            REGISTRY.pop(a, None)
            REGISTRY.pop(b, None)

    def test_changed_parent_parameters_cannot_reuse_an_old_immutable_identity(self):
        path = self.catalog.root / "catalog" / "candidates" / (self.parent_id + ".json")
        record = json.loads(path.read_text(encoding="utf-8"))
        record["definition"]["parameters"]["changed"] = True
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            resolve_candidate_closure(self.cfg, self.child, REGISTRY)

    def test_recursive_parent_scope_uses_real_annual_anchor_and_stops_at_2018(self):
        from fea.engine import Engine
        # Use the actual anchor implementation without loading a raw universe.
        engine = Engine.__new__(Engine)
        engine.cfg = self.cfg
        engine._start_floor = 20180101
        grand = FactorSpec(name="grand", group="test", warmup_days=40)
        parent = FactorSpec(name="parent", group="test", deps=("grand",), warmup_days=40)
        child = FactorSpec(name="child", group="test", deps=("parent",), warmup_days=40)
        label = FactorSpec(name="label", group="test", is_label=True, forward_days=5)
        starts = dependency_output_starts(engine, [grand, parent, child, label], 20260601)
        self.assertEqual(starts, {"grand": 20241122, "parent": 20251122, "child": 20260601, "label": 20260601})
        self.assertTrue(all(s >= 20180101 for s in dependency_output_starts(engine, [grand, parent, child], 20180102).values()))

    def _real_price_fixture(self):
        from fea.prices import LEVELS, FLOWS, RATIOS
        original_root = Path(__file__).resolve().parents[1]
        codes = ["000001.SZ", "600000.SH"]
        frozen = self.root / "conf" / "universe.tsv"
        frozen.write_text("\n".join(c + "\t2000-01-01\t" for c in codes), encoding="utf-8")
        frequency = self.root / "conf" / "frequency.yaml"
        source_names = ["stock_daily", "stock_adj_factor", "stock_finance", "basic_calendar", "stock_list", "stock_st_info"]
        frequency.write_text(yaml.safe_dump({"datasets": {n: {} for n in source_names}}), encoding="utf-8")
        self.cfg.raw.update({"_project_root": str(original_root),
            "compute": {"revision_days": 10, "dep_backfill_days": {"default": 20}},
            "rank": {"winsor": [.01, .99], "min_cross_section": 2},
            "universe": {"main_board_prefixes": ["000", "600"], "exclude_st": False,
                         "min_listed_days": 0, "frozen_list": str(frozen)}})
        self.cfg.raw["paths"].update({"frequency": str(frequency),
            "production_registry": str(self.root / "conf" / "production.json")})
        dates = pd.bdate_range("2024-10-01", "2026-08-14").strftime("%Y-%m-%d")
        def snapshot(name, frame):
            folder = self.root / "raw" / name
            folder.mkdir(exist_ok=True)
            frame.to_parquet(folder / "data.parquet", index=False)
        snapshot("basic_calendar", pd.DataFrame({"date": dates, "is_open": True}))
        snapshot("stock_list", pd.DataFrame({"stock_code": codes, "list_date": "2000-01-01", "delist_date": None}))
        snapshot("stock_st_info", pd.DataFrame({"stock_code": pd.Series(dtype="string"), "trade_date": pd.Series(dtype="string")}))
        grid = pd.DataFrame({"trade_date": np.repeat(dates, 2), "stock_code": np.tile(codes, len(dates))})
        time = np.repeat(np.arange(len(dates), dtype=float), 2)
        price = 10 + time * .01 + np.sin(time / 7) * .1 + np.tile([0., 3.], len(dates))
        daily = grid.copy()
        for column in LEVELS:
            daily[column] = price
        for column in FLOWS:
            daily[column] = 100 + time
        for column in RATIOS:
            daily[column] = .1
        snapshot("stock_daily", daily)
        snapshot("stock_adj_factor", grid.assign(adj_factor=1.))
        snapshot("stock_finance", grid.assign(total_share=10000., float_share=9000., free_share=8000.))
        return daily

    def test_real_rolling_parent_short_window_and_two_phase_increment_match_full(self):
        """Actual Engine + bounded Parquet fixture; only test pool count is 2."""
        self._real_price_fixture()
        for family, variants in [("rolling_parent_family", [{"window": 5}])]:
            self.catalog.register_family({"family_id": family, "idea_id": "closure_idea_v1", "trial_budget": 1,
                                          "allowed_variants": variants, "baseline": "rolling close"})
        parent = self.catalog.register_candidate(self.definition("rp", "rolling_parent_family", "rolling_parent",
            {"window": 5}, ["stock_daily", "stock_adj_factor"]))
        parent_id = parent["candidate_id"]
        params = {"parent": parent_id, "window": 5}
        self.catalog.register_family({"family_id": "rolling_child_family", "idea_id": "closure_idea_v1", "trial_budget": 1,
                                      "allowed_variants": [params], "baseline": "rolling immutable parent"})
        child = self.catalog.register_candidate(self.definition("rc", "rolling_child_family", "rolling_child", params, [parent_id]))
        child_id = child["candidate_id"]
        # The public protocol remains fixed2115. A small explicit unit-test pool
        # exercises real raw reads, calculation, persisted parents and manifests.
        with patch("research.runner.load_protocol", return_value=Protocol(expected_stocks=2)):
            full = run_candidate(self.cfg, child_id, "2026-01-02", "2026-06-30", jobs=1)
            short = run_candidate(self.cfg, child_id, "2026-06-01", "2026-06-30", jobs=1)
            increment = run_candidate(self.cfg, child_id, "2026-06-01", "2026-06-30", jobs=1,
                                      phase_ends=[20260626, 20260630], kind="incremental_validation")
        full_ns, _ = read_run(self.cfg, full["run_id"])
        short_ns, _ = read_run(self.cfg, short["run_id"])
        incr_ns, _ = read_run(self.cfg, increment["run_id"])
        a = factor_frame(full_ns, child_id)
        b = factor_frame(short_ns, child_id)
        self.assertTrue(compare_frames(a[a.trade_date >= "2026-06-01"], b)["passed"])
        self.assertTrue(compare_frames(b, factor_frame(incr_ns, child_id))["passed"])
        self.assertEqual(short["dependency_output_starts"][parent_id], 20251122)
        self.assertEqual(short["dependency_output_starts"][child_id], 20260601)
        self.assertTrue(np.isfinite(b.value).all())
        # Candidate and every label retain the requested axis; only the parent
        # is persisted before it, entirely inside this trial's namespace.
        self.assertEqual(b.trade_date.min(), "2026-06-01")
        self.assertLess(factor_frame(short_ns, parent_id).trade_date.min(), "2026-01-01")
        for h in (1, 3, 5, 10, 20):
            label = factor_frame(short_ns, f"label_ret_{h}d")
            self.assertEqual(label.trade_date.min(), "2026-06-01")
            self.assertFalse(np.isfinite(label[label.trade_date == "2026-06-30"].value).any())
        validate_raw_visibility(increment)
        self.assertEqual(increment["raw_visibility"]["phase_cutoffs"], [20260626, 20260630])
        phase_year_paths = increment["raw_visibility"]["derived_phase_year_paths"]
        self.assertEqual(set(phase_year_paths), {"0", "1"})
        self.assertEqual(phase_year_paths["0"]["2026"],
                         "data/derived/phase_0_cutoff_20260626/year_2026")
        self.assertEqual(phase_year_paths["1"]["2026"],
                         "data/derived/phase_1_cutoff_20260630/year_2026")
        self.assertNotEqual(phase_year_paths["0"]["2026"], phase_year_paths["1"]["2026"])
        changed = dict(increment)
        changed["raw_visibility"] = {}
        with self.assertRaisesRegex(ValueError, "actual per-phase"):
            validate_raw_visibility(changed)
        self.assertNotIn(child_id, REGISTRY)
        self.assertNotIn(parent_id, REGISTRY)
        self.assertFalse((self.root / "conf" / "production.json").exists())

    def test_actual_raw_cutoff_detects_direct_future_peek_with_narrow_projection(self):
        daily = self._real_price_fixture()
        self.catalog.register_family({"family_id": "peek_family", "idea_id": "closure_idea_v1", "trial_budget": 1,
                                      "allowed_variants": [{}], "baseline": "deliberately invalid future read"})
        record = self.catalog.register_candidate(self.definition("fp", "peek_family", "future_peek", {}, ["stock_daily"]))
        identity = record["candidate_id"]
        with patch("research.runner.load_protocol", return_value=Protocol(expected_stocks=2)):
            full = run_candidate(self.cfg, identity, "2026-06-01", "2026-06-30", jobs=1)
            evidence = verify_candidate(self.cfg, full["run_id"], jobs=1, cutoff="2026-06-26")
        self.assertFalse(evidence["passed"])
        self.assertFalse(evidence["causal_truncation"]["passed"])
        self.assertTrue(evidence["same_source_identity"])
        self.assertTrue(evidence["same_computation_identity"])
        full_ns, _ = read_run(self.cfg, full["run_id"])
        trunc_ns, truncated = read_run(self.cfg, evidence["cutoff_run"])
        # Reference itself is capped at its end, not the raw file's August
        # tail. Earlier cutoff therefore changes the direct raw peek result.
        for namespace, cap in [(full_ns, "2026-06-30"), (trunc_ns, "2026-06-26")]:
            frame = factor_frame(namespace, identity)
            expected = daily[daily.trade_date <= cap].groupby("stock_code", sort=False).tail(1).set_index("stock_code").close
            for code, group in frame.groupby("stock_code"):
                np.testing.assert_array_equal(group.value.to_numpy(), np.full(len(group), expected[code], dtype=np.float32))
        self.assertEqual(truncated["raw_visibility"]["phase_cutoffs"], [20260626])

    def test_visible_st_new_event_is_isolated_per_phase_and_full_file_identity_is_shared(self):
        self._real_price_fixture()
        self.cfg.raw["universe"]["exclude_st"] = True
        st_file = self.root / "raw" / "stock_st_info" / "data.parquet"
        pd.DataFrame({"stock_code": ["300001.SZ", "000001.SZ"],
                      "trade_date": ["2026-06-01", "2026-06-30"]}).to_parquet(st_file, index=False)
        production_baseline = self.root / "state" / "universe_st.json"
        production_baseline.parent.mkdir(exist_ok=True)
        sentinel = b'{"sha1":"production-sentinel","production_only":true}'
        production_baseline.write_bytes(sentinel)
        with patch("research.runner.load_protocol", return_value=Protocol(expected_stocks=2)):
            full = run_candidate(self.cfg, self.parent_id, "2026-06-01", "2026-06-30", jobs=1)
            truncated = run_candidate(self.cfg, self.parent_id, "2026-06-01", "2026-06-26", jobs=1)
            increment = run_candidate(self.cfg, self.parent_id, "2026-06-01", "2026-06-30", jobs=1,
                                      phase_ends=[20260626, 20260630], kind="incremental_validation")
        full_ns, _ = read_run(self.cfg, full["run_id"])
        trunc_ns, _ = read_run(self.cfg, truncated["run_id"])
        incr_ns, _ = read_run(self.cfg, increment["run_id"])
        full_frame = factor_frame(full_ns, self.parent_id)
        self.assertTrue(compare_frames(full_frame[full_frame.trade_date <= "2026-06-26"],
                                       factor_frame(trunc_ns, self.parent_id))["passed"])
        self.assertTrue(compare_frames(full_frame, factor_frame(incr_ns, self.parent_id))["passed"])
        self.assertEqual({r["source_identity_hash"] for r in (full, truncated, increment)}, {full["source_identity_hash"]})
        for run in (full, truncated, increment):
            self.assertEqual(run["sources"]["stock_st_info"]["rows"], 2)
            validate_raw_visibility(run)
        self.assertEqual([r["visible_st"]["n_events"] for r in increment["st_visibility_records"]], [0, 1])
        self.assertEqual(increment["st_visibility_records"][0]["visible_st"]["sha1"], "pool-empty")
        self.assertEqual(increment["st_visibility_records"][1]["visible_st"]["max_date"], "2026-06-30")
        for path in increment["raw_visibility"]["st_phase_paths"]:
            self.assertTrue((incr_ns / path / "universe_st.json").exists())
        latest_st_stock = full_frame[(full_frame.trade_date == "2026-06-30") & (full_frame.stock_code == "000001.SZ")]
        self.assertTrue(latest_st_stock.value.isna().all())
        self.assertEqual(production_baseline.read_bytes(), sentinel)

    def test_same_phase_visible_st_change_is_rejected_even_after_cached_read(self):
        self._real_price_fixture()
        namespace = self.root / "artifacts" / "gate_test"
        namespace.mkdir()
        sandbox = self.cfg.sandbox(namespace)
        sandbox.raw["_audit_cutoff"] = 20260630
        engine = _research_phase_engine(sandbox, 0, 20260630)
        first = engine.st_gate()
        self.assertEqual(first["n_events"], 0)
        st_file = self.root / "raw" / "stock_st_info" / "data.parquet"
        pd.DataFrame({"stock_code": ["000001.SZ"], "trade_date": ["2026-06-30"]}).to_parquet(st_file, index=False)
        with self.assertRaisesRegex(RuntimeError, "Visible ST changed"):
            engine.st_gate()
        # The original immutable empty visible record remains as evidence;
        # no refresh/acceptance or production baseline operation occurred.
        saved = json.loads((namespace / "state" / "st_phase_0_20260630" / "visible.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["visible_st"]["n_events"], 0)
        self.assertFalse((self.root / "state" / "universe_st.json").exists())


if __name__ == "__main__":
    unittest.main()
