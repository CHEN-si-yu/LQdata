"""Research isolation, identity, causal splits and bounded-builder contracts."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.builders import volume_price_exhaustion
from research.catalog import Catalog, build_spec, immutable_json, project_path, slug
from research.evaluation import matrix
from research.protocol import Protocol, walk_forward_splits
from research.runner import compare_frames, dependency_specs


class ResearchContractTests(unittest.TestCase):
    def test_causal_date_splits_purge_actual_label_endpoints(self):
        dates = np.asarray([int(s) for s in pd.bdate_range("2018-01-02", "2024-12-31").strftime("%Y%m%d")])
        p = Protocol()
        splits = walk_forward_splits(dates, p)
        self.assertGreater(len(splits), 5)
        positions = {int(d): i for i, d in enumerate(dates)}
        for fold in splits:
            tr, va, te = fold["train"], fold["inner_validation"], fold["outer_test"]
            self.assertLess(positions[tr[-1]] + p.purge_days, positions[va[0]])
            self.assertLess(positions[va[-1]] + p.purge_days, positions[te[0]])
            self.assertFalse(set(tr) & set(va))
            self.assertFalse(set(va) & set(te))

    def test_splits_reject_duplicate_dates(self):
        with self.assertRaises(ValueError):
            walk_forward_splits([20200102, 20200102])

    def test_immutable_record_rejects_overwrite_and_serializes_missing_metric(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.json"
            immutable_json(path, {"x": np.nan, "count": np.int64(3)})
            self.assertEqual(json.loads(path.read_text()), {"x": None, "count": 3})
            with self.assertRaises(ValueError):
                immutable_json(path, {"x": 0})

    def test_paths_cannot_escape_project(self):
        cfg = SimpleNamespace(root=Path("C:/project"))
        with self.assertRaises(ValueError):
            project_path(cfg, "../production")
        for unsafe in ("../x", "name/other", "bad-name", "../../secret"):
            with self.assertRaises(ValueError):
                slug(unsafe)

    def test_candidate_identity_and_family_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = SimpleNamespace(root=Path(folder))
            catalog = Catalog(cfg)
            catalog.register_idea({"idea_id": "idea_v1", "hypothesis": "x", "mechanism": "y",
                                   "expected_direction": "positive", "target_horizons": [1, 5],
                                   "falsifiable": "z", "fields": ["stock_daily.close"]})
            catalog.register_family({"family_id": "family_v1", "idea_id": "idea_v1", "trial_budget": 1,
                                     "allowed_variants": [{"window": 5}, {"window": 10}], "baseline": "momentum"})
            data = {"name": "sample_v1", "family_id": "family_v1", "implementation": "research.builders:test",
                    "parameters": {"window": 5}, "spec": {"group": "research", "deps": ["stock_daily"],
                    "formula": "close", "desc": "test", "warmup_days": 40}, "units": "native",
                    "availability": "close", "missing_policy": "nan"}
            with patch("research.catalog.implementation", return_value=(lambda params: lambda ctx: None, "hash1")):
                first = catalog.register_candidate(data)
                repeated = catalog.register_candidate(data)
                self.assertEqual(first["candidate_id"], repeated["candidate_id"])
                self.assertEqual(catalog.state(first["candidate_id"]), "candidate")
                changed = copy.deepcopy(data)
                changed["parameters"]["window"] = 10
                with self.assertRaises(ValueError):
                    catalog.register_candidate(changed)
            self.assertEqual(len(catalog.records("candidates")), 1)
            self.assertFalse(first["validated_alpha"])

    def test_factor_grid_rejects_missing_rows_and_duplicate_keys(self):
        dates, codes = ["2026-01-02", "2026-01-05"], ["000001.SZ", "600000.SH"]
        frame = pd.DataFrame({"trade_date": np.repeat(dates, 2), "stock_code": np.tile(codes, 2),
                              "value": [1., np.nan, 3., 4.], "rank": [0., np.nan, 0., 1.]})
        value = matrix(frame, dates, codes)
        self.assertEqual(value.shape, (2, 2))
        with self.assertRaises(ValueError):
            matrix(frame.iloc[:3], dates, codes)
        with self.assertRaises(ValueError):
            matrix(pd.concat([frame, frame.iloc[:1]]), dates, codes)

    def test_exact_parity_requires_real_finite_evidence(self):
        frame = pd.DataFrame({"trade_date": ["2026-01-02"], "stock_code": ["000001.SZ"],
                              "value": np.asarray([np.nan], dtype=np.float32), "rank": np.asarray([np.nan], dtype=np.float32)})
        self.assertFalse(compare_frames(frame, frame.copy())["passed"])
        frame.loc[0, "value"] = np.float32(.123)
        self.assertTrue(compare_frames(frame, frame.copy())["passed"])
        changed = frame.copy()
        changed.loc[0, "value"] = np.nextafter(np.float32(.123), np.float32(1.))
        self.assertFalse(compare_frames(frame, changed)["passed"])

    def test_market_parity_uses_date_and_value_without_stock_or_rank(self):
        frame = pd.DataFrame({"trade_date": ["2026-01-02", "2026-01-05"],
                              "value": np.asarray([1., np.nan], dtype=np.float32)})
        self.assertTrue(compare_frames(frame, frame.copy())["passed"])
        changed = frame.copy()
        changed.loc[0, "trade_date"] = "2026-01-06"
        self.assertFalse(compare_frames(frame, changed)["passed"])

    def test_candidates_cannot_consume_labels(self):
        ordinary = SimpleNamespace(name="candidate", deps=("label_ret_1d",), is_label=False)
        label = SimpleNamespace(name="label_ret_1d", deps=(), is_label=True)
        with self.assertRaises(ValueError):
            dependency_specs(ordinary, {label.name: label}, (1,))

    def test_builder_prefix_invariance_and_missing_policy(self):
        rng = np.random.default_rng(7)
        t, c = 110, 9
        close = 50 * np.exp(np.cumsum(rng.normal(.001, .008, (t, c)), axis=0))
        volume = np.exp(rng.normal(10, .3, (t, c)))
        traded = np.ones((t, c), dtype=bool)
        traded[-1, 0] = False
        class Context:
            def __init__(self, end):
                self.end = end
            def hfq(self, field):
                return close[:self.end]
            def px_raw(self, field):
                return volume[:self.end]
            def traded(self):
                return traded[:self.end]
        for expression in ("continuous", "event"):
            fn = volume_price_exhaustion({"window": 10, "expression": expression, "valid_fraction": .8})
            full, cut = fn(Context(t)), fn(Context(t-3))
            np.testing.assert_equal(full[:t-3], cut)
            self.assertTrue(np.isnan(full[-1, 0]))
            self.assertTrue(np.isfinite(full[25:]).any())
        with self.assertRaises(ValueError):
            volume_price_exhaustion({"window": 7, "expression": "continuous", "valid_fraction": .8})

    def test_candidate_cannot_set_future_label_fields(self):
        definition = {"parameters": {}, "spec": {"group": "research", "deps": ["stock_daily"],
                      "formula": "close", "desc": "test", "warmup_days": 40, "forward_days": 1}}
        with self.assertRaises(ValueError):
            build_spec(definition, lambda p: lambda ctx: None, "candidate_v1")

    def test_candidate_cannot_disable_required_2018_alignment(self):
        definition = {"parameters": {}, "spec": {"group": "research", "deps": ["stock_daily"],
                      "formula": "close", "desc": "test", "warmup_days": 40, "align_fill": None}}
        with self.assertRaisesRegex(ValueError, "required annual date grid"):
            build_spec(definition, lambda p: lambda ctx: None, "candidate_v1")


if __name__ == "__main__":
    unittest.main()
