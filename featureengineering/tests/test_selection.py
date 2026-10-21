import argparse
from contextlib import contextmanager
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fea.config import Config
from fea.spec import FactorSpec
from research import selection


class DummyGuard:
    def __init__(self, policy=None):
        self.samples = 0
    def check(self):
        self.samples += 1
    def report(self):
        return {"samples": self.samples}


def serial_year(objects, context, jobs, guard, completed):
    selection._YEAR_CONTEXT = context
    for obj in objects:
        guard.check()
        completed(selection._annual_unit(obj))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def write_panel(self, name, dates, codes, values, market=False, label=False):
        path = self.root / "data" / ("market_factors" if market else "factors") / name / "year=2018" / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = selection.canonical_keys(dates, codes, market)
        table = keys.append_column("value", pa.array(np.asarray(values, dtype=np.float32).ravel(), type=pa.float32()))
        if not market:
            ranks = np.full(table.num_rows, np.nan, dtype=np.float32) if label else np.tile(np.linspace(0, 1, len(codes), dtype=np.float32), len(dates))
            table = table.append_column("rank", pa.array(ranks, type=pa.float32()))
        pq.write_table(table, path)
        return path

    def test_profile_separates_alignment_from_real_active_coverage(self):
        values = np.array([[np.nan]*40, [0.]*40, list(range(40)), [np.nan]*40])
        daily = selection.profile_panel(values, ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"],
                                        "event", valid_start="2018-01-03")
        summary = selection.summarize_profile(daily)
        self.assertEqual(summary["active_dates"], 3)
        self.assertEqual(summary["constant_cross_section_dates"], 1)
        self.assertEqual(summary["variable_active_dates"], 1)
        self.assertAlmostEqual(summary["active_coverage_mean"], 2/3)
        self.assertEqual(summary["first_active_value_date"], "2018-01-03")
        self.assertEqual(summary["prefix_finite_cells"], 0)

    def test_market_native_scalar_is_not_constant_cross_section_or_ic(self):
        daily = selection.profile_panel([2., 2., 2.], ["2018-01-02", "2018-01-03", "2018-01-04"], "market", "market")
        summary = selection.summarize_profile(daily)
        self.assertEqual(summary["constant_cross_section_dates"], 0)
        self.assertEqual(summary["annual"][0]["temporal_std"], 0)
        row = selection.recommendations([{"name":"market","kind":"market"}], [summary], [])[0]
        self.assertEqual(row["primary_evaluation"], [])
        self.assertEqual(row["research_evidence_tier"], "native_market_context_no_cross_section_ic")

    def test_negative_rankic_cannot_retire_or_flip_direction(self):
        profile = {"name":"alpha", "finite_active_dates":200, "variable_active_dates":200}
        evals = [{"name":"alpha", "horizon":h, "days":200, "rankic":-.9} for h in (1,5)]
        row = selection.recommendations([{"name":"alpha","kind":"stock"}], [profile], evals)[0]
        self.assertEqual(row["recommendation"], "retain_engineering_review")
        self.assertEqual(row["research_evidence_tier"], "negative_after_declared_direction_review")
        self.assertFalse(row["automatic_action"])
        self.assertFalse(row["validated_alpha"])

    def test_rank_recall_alone_does_not_recommend_redundancy(self):
        obj = {"name":"a","kind":"stock"}
        profile = {"name":"a", "finite_active_dates":200, "variable_active_dates":200}
        pair = {"a":"a","b":"b", "rank_corr":{"days":200,"mean":1.},
                "model_value_corr":{"days":200,"mean":.7}, "common_fraction_pool":.9, "coverage_jaccard":.9}
        row = selection.recommendations([obj], [profile], [], redundancy=[pair])[0]
        self.assertEqual(row["recommendation"], "retain_engineering_review")
        pair["model_value_corr"]["mean"] = .999
        row = selection.recommendations([obj], [profile], [], redundancy=[pair])[0]
        self.assertEqual(row["recommendation"], "redundancy_review")
        pair["coverage_jaccard"] = .1
        self.assertEqual(selection.recommendations([obj], [profile], [], redundancy=[pair])[0]["recommendation"], "retain_engineering_review")

    def test_formal_labels_are_retained_without_alpha_ranking(self):
        row = selection.recommendations([{"name":"label_ret_5d","kind":"label"}], [], [])[0]
        self.assertEqual(row["recommendation"], "retain_formal_benchmark")

    def test_grid_projection_rejects_duplicate_missing_and_order_changes(self):
        dates, codes = ["2018-01-02", "2018-01-03"], ["000001.SZ", "000002.SZ"]
        path = self.write_panel("a", dates, codes, [[1,2],[3,4]])
        expected = selection.canonical_keys(dates, codes)
        values, _ = selection.read_panel(path, expected, n_codes=2)
        np.testing.assert_array_equal(values, [[1,2],[3,4]])
        original = pq.read_table(path, columns=["trade_date","stock_code","value","rank"])
        for indices in ([0,0,2,3], [0,1,2], [3,2,1,0]):
            pq.write_table(original.take(pa.array(indices)), path)
            with self.assertRaisesRegex(ValueError, "grid/order/unique"):
                selection.read_panel(path, expected, n_codes=2)

    def test_formal_label_nonmissing_rank_is_rejected(self):
        path = self.write_panel("bad_label", ["2018-01-02"], ["a","b"], [[1,2]])
        with self.assertRaisesRegex(ValueError, "label rank"):
            selection.read_panel(path, selection.canonical_keys(["2018-01-02"], ["a","b"]), label=True)

    def test_source_freeze_detects_source_changes_and_external_paths(self):
        source = self.root / "source.py"
        source.write_text("version=1")
        cfg = SimpleNamespace(root=self.root)
        freeze = self.root / "freeze.json"
        payload = {"schema_version":1, "release_id":"release", "protocol_sha256":"protocol",
                   "source_sha256":{"source.py":selection.file_hash(source)}}
        freeze.write_text(json.dumps(payload))
        selection.verify_source_freeze(cfg, freeze, "release", "protocol")
        source.write_text("version=2")
        with self.assertRaisesRegex(ValueError, "source/config changed"):
            selection.verify_source_freeze(cfg, freeze, "release", "protocol")
        payload["source_sha256"] = {"../outside.py":"0"*64}
        freeze.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "project-relative"):
            selection.verify_source_freeze(cfg, freeze, "release", "protocol")

    def test_output_is_an_independent_artifacts_namespace_in_the_same_frozen_batch(self):
        cfg = Config(root=self.root, raw={"paths": {"artifacts": "artifacts"}})
        freeze = "artifacts/research/batches/batch_a/source_freeze.json"
        good = selection.selection_output_root(cfg, "artifacts/research/batches/batch_a/selection", freeze)
        self.assertEqual(good, self.root / "artifacts/research/batches/batch_a/selection")
        for forbidden in ("artifacts", "data/factors", "state", ".", "../outside"):
            with self.subTest(path=forbidden), self.assertRaises(ValueError):
                selection.selection_output_root(cfg, forbidden, freeze)
        with self.assertRaisesRegex(ValueError, "batch namespace"):
            selection.selection_output_root(cfg, "artifacts/research/batches/batch_b/selection", freeze)
        with self.assertRaisesRegex(ValueError, "batch namespace"):
            selection.selection_output_root(cfg, "artifacts/selection", freeze)
        self.assertEqual(selection.selection_output_root(cfg, "artifacts/selection", "freeze.json"),
                         self.root / "artifacts/selection")

    def test_cgroup_guard_accounts_for_cache_and_uses_project_budget(self):
        usage, stats = self.root/"usage", self.root/"stats"
        usage.write_text(str(82*1024**3))
        stats.write_text("total_rss 1024\ntotal_cache 88000000000\n")
        guard = object.__new__(selection.MemoryGuard)
        guard.usage, guard.stat, guard.anon_key = usage, stats, "total_rss"
        guard.limits = {"charged":81*1024**3,"anon":76*1024**3}
        guard.peak_charged = guard.peak_anon = guard.samples = 0
        with self.assertRaisesRegex(MemoryError, "completed annual units"):
            guard.check()
        self.assertEqual(guard.report()["peak_charged_bytes"], 82*1024**3)

    def test_empty_bin_units_share_a_canonical_arrow_schema(self):
        labels = {1:np.arange(40.)[None,:], 5:np.arange(40.)[None,:]}
        populated = selection.evaluate_panel(np.arange(40.)[None,:], labels, ["2018-01-02"], min_n=30)
        empty = selection.evaluate_panel(np.full((1,40),np.nan), labels, ["2018-01-02"], min_n=30)
        self.assertEqual(selection._evaluation_table(populated).schema, selection._evaluation_table(empty).schema)

    def test_complete_unit_resume_does_not_reread_production_and_detects_tampering(self):
        dates, codes = ["2018-01-02", "2018-01-03"], [f"{i:06d}.SZ" for i in range(40)]
        path = self.write_panel("a", dates, codes, np.tile(np.arange(40), (2,1)))
        before = path.read_bytes(), path.stat().st_mtime_ns
        selection._YEAR_CONTEXT = {"year":2018, "out":str(self.root/"result"), "request_sha256":"request",
            "dates":dates, "stock_keys":selection.canonical_keys(dates,codes), "market_keys":selection.canonical_keys(dates,codes,True),
            "n_codes":40, "evaluate":True, "labels":{1:np.tile(np.arange(40), (2,1)),5:np.tile(np.arange(40), (2,1))},
            "min_pairs":30, "bins":10, "value_transform":"cs_winsor_1_99_zscore_clip10_direction_v1", "fadvise_clean":False}
        obj = {"name":"a", "kind":"stock", "root":str(self.root/"data/factors"), "valid_start":"2018-01-01", "higher_is_better":False}
        first = selection._annual_unit(obj)
        with patch.object(selection, "read_panel", side_effect=AssertionError("must reuse checkpoint")):
            second = selection._annual_unit(obj)
        self.assertEqual(first, second)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        metrics = pd.read_parquet(self.root/"result/units/2018/a/evaluation.parquet")
        self.assertTrue((metrics.rankic < -.999).all())
        self.assertEqual(first["errors"], [])
        profile = self.root/"result/units/2018/a/profile.parquet"
        profile.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "unit changed"):
            selection._annual_unit(obj)

    def test_late_start_evaluation_masks_alignment_zeros_but_preserves_real_event_zeros(self):
        dates = ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"]
        codes = [f"{i:06d}.SZ" for i in range(40)]
        values = np.array([[0.] * 40, [0.] * 40, [0.] * 40, list(range(40))])
        path = self.write_panel("late_event", dates, codes, values)
        before = path.read_bytes(), path.stat().st_mtime_ns
        selection._YEAR_CONTEXT = {"year":2018, "out":str(self.root/"result"), "request_sha256":"request",
            "dates":dates, "stock_keys":selection.canonical_keys(dates,codes), "market_keys":selection.canonical_keys(dates,codes,True),
            "n_codes":40, "evaluate":True, "labels":{h:np.tile(np.arange(40), (4,1)) for h in (1,5)},
            "min_pairs":30, "bins":10, "value_transform":"cs_winsor_1_99_zscore_clip10_direction_v1", "fadvise_clean":False}
        obj = {"name":"late_event", "kind":"stock", "root":str(self.root/"data/factors"),
               "valid_start":"2018-01-04", "higher_is_better":True}
        record = selection._annual_unit(obj)
        self.assertEqual(record["errors"], [])
        metrics = pd.read_parquet(self.root/"result/units/2018/late_event/evaluation.parquet")
        for _, part in metrics.groupby("horizon"):
            self.assertEqual(part.trade_date.tolist(), dates)
            self.assertEqual(part.coverage.tolist(), [0.,0.,1.,1.])
            self.assertEqual(part.pairs.tolist(), [0,0,40,40])
            self.assertEqual(part.unique_values.tolist(), [0,0,1,40])
            self.assertTrue(part.zero_fraction.iloc[:2].isna().all())
            self.assertEqual(part.zero_fraction.iloc[2], 1.)
            self.assertEqual(part.reason.iloc[2], "constant_value_or_target")
            self.assertGreater(part.rankic.iloc[3], .999)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_corrupt_file_is_attributed_without_overwriting_production(self):
        path = self.root/"data/factors/broken/year=2018/data.parquet"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"broken")
        selection._YEAR_CONTEXT = {"year":2018,"out":str(self.root/"result"),"request_sha256":"req",
            "stock_keys":selection.canonical_keys(["2018-01-02"],["a"]),"fadvise_clean":False,"n_codes":1}
        obj = {"name":"broken","kind":"stock","root":str(self.root/"data/factors")}
        record = selection._annual_unit(obj)
        self.assertEqual(record["identity"]["name"], "broken")
        self.assertTrue(record["errors"])
        self.assertEqual(path.read_bytes(), b"broken")

    @unittest.skipUnless("fork" in mp.get_all_start_methods(), "Linux fork resource/equivalence check")
    def test_fork_matches_serial_units_and_keeps_shared_formal_labels_readonly(self):
        dates, codes = ["2018-01-02", "2018-01-03"], [f"{i:06d}.SZ" for i in range(40)]
        values = np.tile(np.arange(40), (2,1))
        paths = [self.write_panel("a",dates,codes,values),self.write_panel("empty",dates,codes,np.full((2,40),np.nan)),
                 self.write_panel("m",dates,codes,[2,3],market=True)]
        broken = self.root/"data/factors/broken/year=2018/data.parquet"
        broken.parent.mkdir(parents=True)
        broken.write_bytes(b"broken")
        paths.append(broken)
        before = {str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in paths}
        labels = {h:values.astype(float)/(h+1) for h in (1,3,5,10,20)}
        for label in labels.values():
            label.setflags(write=False)
        context = {"year":2018,"out":str(self.root/"serial"),"request_sha256":"same-request",
            "dates":dates,"stock_keys":selection.canonical_keys(dates,codes),"market_keys":selection.canonical_keys(dates,codes,True),
            "n_codes":40,"evaluate":True,"labels":labels,"min_pairs":30,"bins":10,
            "value_transform":"cs_winsor_1_99_zscore_clip10_direction_v1","fadvise_clean":False}
        objects = [{"name":name,"kind":"market" if name=="m" else "stock",
                    "root":str(self.root/("data/market_factors" if name=="m" else "data/factors")),
                    "valid_start":"2018-01-01","higher_is_better":True} for name in ("a","empty","m","broken")]
        serial, parallel = [], []
        serial_year(objects,context,1,DummyGuard(),serial.append)
        context = {**context,"out":str(self.root/"parallel")}
        selection._run_year(objects,context,2,DummyGuard(),parallel.append)
        key = lambda row: row["identity"]["name"]
        self.assertEqual(sorted(serial,key=key),sorted(parallel,key=key))
        self.assertEqual([r["identity"]["name"] for r in parallel if r["errors"]],["broken"])
        p1,e1 = selection._export_tables(self.root/"serial",objects)
        p2,e2 = selection._export_tables(self.root/"parallel",objects)
        self.assertEqual(selection._json_safe((p1,e1)),selection._json_safe((p2,e2)))
        self.assertEqual({str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in paths},before)
        for h,label in labels.items():
            self.assertFalse(label.flags.writeable)
            np.testing.assert_array_equal(label,values/(h+1))

    def test_end_to_end_reports_and_resumes_fixed_2115_axis_without_production_writes(self):
        from research.protocol import Protocol
        dates = ["2018-01-02", "2018-01-03"]
        codes = np.array([f"{i:06d}.SZ" for i in range(2115)])
        frozen = self.root/"conf/universe_frozen.tsv"
        frozen.parent.mkdir()
        frozen.write_text("\n".join(codes))
        protocol_file = self.root/"conf/research/protocol.yaml"
        protocol_file.parent.mkdir()
        protocol_file.write_text("{}")
        cfg = Config(root=self.root,raw={"paths":{"upstream":"raw","factors":"data/factors","market_factors":"data/market_factors","state":"state"},
            "universe":{"frozen_list":"conf/universe_frozen.tsv"},"default_start":"2018-01-01","resources":{"max_jobs":12}})
        calendar = self.root/"raw/basic_calendar/data.parquet"
        calendar.parent.mkdir(parents=True)
        pd.DataFrame({"date":dates,"is_open":[1,1]}).to_parquet(calendar,index=False)
        stock = self.write_panel("stock",dates,codes,np.tile(np.arange(2115),(2,1)))
        market = self.write_panel("market",dates,codes,[2,3],market=True)
        label_paths = [self.write_panel(f"label_ret_{h}d",dates,codes,np.tile(np.arange(2115)/(h+1),(2,1)),label=True) for h in Protocol().horizons]
        inputs = [stock,market,*label_paths,calendar]
        before = {str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in inputs}
        specs = [FactorSpec("stock","technical",higher_is_better=False),FactorSpec("market","market",is_market=True)]
        release = {"release_id":"r", "cutoff":"2018-01-03", "stock_count":2115, "universe_sha256":"u",
            "objects":[{"name":s.name,"recipe":"recipe","definition_sha256":"definition"} for s in specs]}
        cfg.state_dir.mkdir()
        (cfg.state_dir/"current_release.json").write_text(json.dumps({"release_id":"r"}))
        freeze = self.root/"freeze.json"
        freeze.write_text(json.dumps({"schema_version":1,"release_id":"r","cutoff":"2018-01-03","stock_count":2115,
            "protocol_sha256":selection.file_hash(protocol_file),"source_sha256":{"conf/research/protocol.yaml":selection.file_hash(protocol_file)}}))
        parser = argparse.ArgumentParser()
        selection.add_arguments(parser)
        args = parser.parse_args(["--source-freeze","freeze.json","--jobs","1","--profile-years","2018","2018",
                                  "--evaluation-years","2018","2018","--out","artifacts/selection"])
        @contextmanager
        def read_release(_):
            yield release
        with patch("fea.release.read_release",read_release),patch("fea.release.production_specs",return_value=specs),\
             patch.object(selection,"MemoryGuard",DummyGuard),patch.object(selection,"_run_year",serial_year):
            first = selection.run_selection(args,cfg)
            args.resume = first["run_id"]
            with patch.object(selection,"read_panel", wraps=selection.read_panel) as reader:
                resumed = selection.run_selection(args,cfg)
            self.assertEqual(reader.call_count,0)  # sealed complete run reused without any factor/label reread
        self.assertEqual(first["status"],"complete")
        folder = Path(first["output"])
        self.assertTrue((folder/"profile_daily.parquet").exists())
        self.assertTrue((folder/"evaluation_daily.parquet").exists())
        metrics = pd.read_parquet(folder/"evaluation_daily.parquet")
        self.assertEqual(set(metrics.horizon),{1,3,5,10,20})
        self.assertEqual(set(metrics.n_pool),{2115})
        self.assertEqual(set(metrics.value_transform),{Protocol().value_transform})
        self.assertEqual(first["recommendations"],resumed["recommendations"])
        self.assertEqual({str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in inputs},before)
        self.assertFalse((self.root/"conf/production.json").exists())


if __name__ == "__main__":
    unittest.main()
