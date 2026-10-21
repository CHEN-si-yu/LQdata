import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fea.config import Config
from fea.engine import Engine
from fea.manifest import Manifest
from fea.panel import Panel
from fea.release import publish
from fea.upstream import Upstream


class StGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.codes = np.array([f"{i:06d}.SZ" for i in range(2115)])
        frozen = self.root / "conf/universe_frozen.tsv"
        frozen.parent.mkdir()
        frozen.write_text("\n".join(self.codes), encoding="utf-8")
        self.cfg = Config(root=self.root, raw={
            "paths": {"upstream": "raw", "state": "state"},
            "default_start": "2018-01-01",
            "universe": {"frozen_list": "conf/universe_frozen.tsv", "exclude_st": True}})
        self.engine = object.__new__(Engine)
        self.engine.cfg = self.cfg
        self.engine.codes = self.codes
        self.engine.up = Upstream(self.cfg.upstream, self.root)
        self.baseline_path = self.cfg.state_dir / "universe_st.json"
        self.old_events = [(self.codes[0], "2026-09-28")]

    def write_events(self, events):
        path = self.cfg.upstream / "stock_st_info/data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(events, columns=["stock_code", "trade_date"]).to_parquet(path, index=False)
        self.engine.up.clear_cache()

    def baseline(self, events=None):
        self.write_events(self.old_events if events is None else events)
        self.engine.st_gate()
        return json.loads(self.baseline_path.read_text(encoding="utf-8"))

    def anchor(self, cutoff="2026-09-29"):
        return publish(self.cfg, [], cutoff, "completed-test-publication", {}, {})

    def coverage(self, name, end):
        manifest = Manifest.load(self.cfg.state_dir, name)
        manifest.add_coverage("2018-01-02", end)
        manifest.save()

    def rejects(self, message=None):
        before = self.baseline_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, message or "ST 内容闸门"):
            self.engine.st_gate()
        self.assertEqual(self.baseline_path.read_bytes(), before)

    def test_forward_suffix_is_accepted_with_unchanged_2115_axis_and_history(self):
        old = self.baseline(self.old_events * 2)
        release = self.anchor()
        self.coverage("already_written", "2026-09-29")
        data = self.cfg.factors_dir / "already_written/year=2026/data.parquet"
        data.parent.mkdir(parents=True)
        pd.DataFrame({"trade_date": ["2026-09-29"] * len(self.codes), "stock_code": self.codes,
                      "value": np.ones(len(self.codes), dtype=np.float32),
                      "rank": np.full(len(self.codes), 0.5, dtype=np.float32)}).to_parquet(data, index=False)
        before_data = data.read_bytes(), data.stat().st_mtime_ns
        self.write_events([(self.codes[1], "2026-10-01"), *self.old_events * 2,
                           (self.codes[0], "2026-09-30")])
        result = self.engine.st_gate()
        accepted = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        proof = accepted["accepted_forward_events"]
        self.assertNotEqual(result["sha1"], old["sha1"])
        self.assertEqual(proof["previous_sha1"], old["sha1"])
        self.assertEqual(proof["prefix_sha1"], old["sha1"])
        self.assertEqual(proof["prefix_n_events"], 2)
        self.assertEqual(proof["added_n_events"], 2)
        self.assertEqual(proof["protected_cutoff"], "2026-09-29")
        self.assertEqual(proof["release_id"], release["release_id"])
        self.assertEqual((data.read_bytes(), data.stat().st_mtime_ns), before_data)

        stock_list = self.cfg.upstream / "stock_list/data.parquet"
        stock_list.parent.mkdir()
        pd.DataFrame({"stock_code": self.codes, "list_date": ["2010-01-01"] * len(self.codes),
                      "delist_date": [""] * len(self.codes)}).to_parquet(stock_list, index=False)
        self.engine._st, self.engine._uni_cache = None, {}
        panel = Panel(np.array([20260929, 20260930]), self.codes)
        mask = self.engine.universe_for(panel)
        self.assertEqual(mask.shape, (2, 2115))
        self.assertTrue(mask[0].all())
        self.assertFalse(mask[1, 0])
        self.assertTrue(mask[1, 1:].all())
        np.testing.assert_array_equal(panel.codes, self.codes)

        # The ordinary equal-SHA path still needs no publication anchor and
        # keeps the acceptance evidence through later annual worker passes.
        (self.cfg.state_dir / "current_release.json").unlink()
        self.engine.st_gate()
        self.assertEqual(json.loads(self.baseline_path.read_text(encoding="utf-8"))["accepted_forward_events"], proof)

    def test_proven_pool_empty_baseline_accepts_only_future_pool_events(self):
        old = self.baseline([("999999.SH", "2020-01-02")])
        self.assertEqual((old["sha1"], old["n_events"]), ("pool-empty", 0))
        self.anchor()
        self.write_events([("999999.SH", "2020-01-02"), (self.codes[0], "2026-09-30")])
        self.engine.st_gate()
        proof = json.loads(self.baseline_path.read_text(encoding="utf-8"))["accepted_forward_events"]
        self.assertEqual((proof["prefix_sha1"], proof["prefix_n_events"]), ("pool-empty", 0))

    def test_readable_empty_table_is_zero_evidence_but_cannot_erase_old_events(self):
        old = self.baseline([])
        self.assertEqual((old["source"], old["n_events"], old["error"]), ("empty", 0, None))
        self.anchor()
        self.write_events([(self.codes[0], "2026-09-30")])
        self.engine.st_gate()
        proof = json.loads(self.baseline_path.read_text(encoding="utf-8"))["accepted_forward_events"]
        self.assertEqual((proof["prefix_sha1"], proof["prefix_n_events"]), ("pool-empty", 0))

        self.baseline_path.unlink()
        self.baseline()
        self.write_events([])
        self.rejects("保护截止及以前")

    def test_unavailable_current_or_previous_source_cannot_silently_accept_events(self):
        old = self.baseline()
        source = self.cfg.upstream / "stock_st_info/data.parquet"
        source.write_bytes(b"not a parquet file")
        self.engine.up.clear_cache()
        self.rejects("stock_st_info 读取失败")

        # Bootstrap also requires a truly readable source.
        self.baseline_path.unlink()
        with self.assertRaisesRegex(RuntimeError, "stock_st_info 读取失败"):
            self.engine.st_gate()
        self.assertFalse(self.baseline_path.exists())

        unknown = {**old, "source": "unavailable", "error": "previous read failed",
                   "sha1": None, "n_events": None, "max_date": None}
        self.baseline_path.write_text(json.dumps(unknown), encoding="utf-8")
        self.write_events([(self.codes[0], "2026-09-30")])
        self.rejects("旧基线读取失败且无指纹")

    def test_historical_add_delete_change_and_duplicate_cannot_hide_behind_new_max_date(self):
        for mutation in ("add", "delete", "change_code", "change_raw_date", "duplicate"):
            with self.subTest(mutation=mutation):
                old = [(self.codes[0], "2026-09-28"), (self.codes[1], "2026-09-29")]
                self.baseline_path.unlink(missing_ok=True)
                self.baseline(old)
                self.anchor()
                if mutation == "add":
                    changed = [*old, (self.codes[2], "2020-01-02")]
                elif mutation == "delete":
                    changed = old[1:]
                elif mutation == "change_code":
                    changed = [(self.codes[2], old[0][1]), old[1]]
                elif mutation == "change_raw_date":
                    changed = [(old[0][0], old[0][1] + " 00:00:00"), old[1]]
                else:
                    changed = [*old, old[0]]
                self.write_events([*changed, (self.codes[3], "2026-10-01")])
                self.rejects("保护截止及以前")

    def test_unpublished_manifest_coverage_is_also_protected(self):
        self.baseline()
        self.anchor()
        self.coverage("unpublished_object", "2026-09-30")
        self.write_events([*self.old_events, (self.codes[0], "2026-09-30"),
                           (self.codes[1], "2026-10-01")])
        self.rejects("保护截止及以前")

    def test_suffix_after_all_coverage_is_allowed(self):
        old = [*self.old_events, (self.codes[0], "2026-09-30")]
        self.baseline(old)
        self.anchor()
        self.coverage("unpublished_object", "2026-09-30")
        self.write_events([*old, (self.codes[1], "2026-10-01")])
        self.engine.st_gate()
        proof = json.loads(self.baseline_path.read_text(encoding="utf-8"))["accepted_forward_events"]
        self.assertEqual(proof["published_cutoff"], "2026-09-29")
        self.assertEqual(proof["protected_cutoff"], "2026-09-30")

    def test_missing_publication_anchor_rejects_forward_events(self):
        self.baseline()
        self.write_events([*self.old_events, (self.codes[0], "2026-09-30")])
        self.rejects("current_release")

    def test_old_baseline_beyond_protection_cannot_prove_its_prefix(self):
        old = [(self.codes[0], "2026-09-30")]
        self.baseline(old)
        self.anchor()
        self.write_events([*old, (self.codes[1], "2026-10-01")])
        self.rejects("旧整体基线含保护截止之后")

    def test_bad_publication_identity_rejects_forward_events(self):
        self.baseline()
        release = self.anchor()
        pointer = json.loads((self.cfg.state_dir / "current_release.json").read_text(encoding="utf-8"))
        release_path = self.root / pointer["path"]
        release["cutoff"] = "2026-09-30"
        release_path.write_text(json.dumps(release), encoding="utf-8")
        self.write_events([*self.old_events, (self.codes[0], "2026-10-01")])
        self.rejects("发布的身份/截止")

    def test_unreadable_or_invalid_manifest_cannot_authorize_forward_events(self):
        self.baseline()
        self.anchor()
        self.write_events([*self.old_events, (self.codes[0], "2026-09-30")])
        path = self.cfg.state_dir / "unknown_object.json"
        for content in ("{broken", json.dumps({"coverage": [["2018-01-02", "invalid"]]}),
                        json.dumps({"name": "unknown_object", "recipe": "v1", "partitions": {}})):
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                self.rejects()

    def test_unprovable_readability_count_and_pool_identity_reject_forward_events(self):
        original = self.baseline()
        self.anchor()
        self.write_events([*self.old_events, (self.codes[0], "2026-09-30")])
        for override in ({"source": "unavailable"}, {"n_events": 2}, {"universe_n": 2114},
                         {"n_events": None}, {"max_date": "invalid"}):
            with self.subTest(override=override):
                self.baseline_path.write_text(json.dumps({**original, **override}), encoding="utf-8")
                self.rejects()

    def test_missing_or_invalid_old_sha_cannot_accept_changed_historical_events(self):
        original = self.baseline()
        self.anchor()
        self.write_events([(self.codes[0], "2026-09-27"), (self.codes[1], "2026-09-30")])
        for invalid in (None, "", " ", "invalid", "pool-empty", "g" * 40, "A" * 40,
                        "0" * 39, "0" * 41, 0, False, [], {}):
            with self.subTest(sha1=invalid):
                self.baseline_path.write_text(json.dumps({**original, "sha1": invalid}), encoding="utf-8")
                self.rejects("旧基线缺失或非法 SHA")
        missing = {key: value for key, value in original.items() if key != "sha1"}
        self.baseline_path.write_text(json.dumps(missing), encoding="utf-8")
        self.rejects("旧基线缺失或非法 SHA")

    def test_existing_nonobject_baseline_is_not_first_bootstrap(self):
        self.baseline()
        self.write_events([(self.codes[0], "2026-09-27")])
        for invalid in (None, [], 0, False, "not a baseline"):
            with self.subTest(baseline=invalid):
                self.baseline_path.write_text(json.dumps(invalid), encoding="utf-8")
                self.rejects("不能当作首次 bootstrap")

    def test_sandbox_and_raw_cap_do_not_use_production_forward_acceptance(self):
        self.baseline()
        self.anchor()
        self.write_events([*self.old_events, (self.codes[0], "2026-09-30")])
        self.cfg.raw["_sandbox"] = True
        self.rejects("仅生产")
        self.cfg.raw.pop("_sandbox")
        self.engine.up.audit_cutoff = 20260930
        self.rejects("仅生产")

    def test_equal_sha_preserves_original_behavior_without_publication_or_coverage_reads(self):
        old = self.baseline()
        (self.cfg.state_dir / "broken_manifest.json").write_text("{broken", encoding="utf-8")
        self.assertEqual(self.engine.st_gate(), {
            key: old[key] for key in ("source", "error", "sha1", "n_events", "max_date")})


if __name__ == "__main__":
    unittest.main()
