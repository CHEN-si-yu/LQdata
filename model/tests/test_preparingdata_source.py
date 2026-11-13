import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import preparingdata as preparing


class ProductionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "data"
        self.registry = self.root / "conf" / "production.json"

    def product(self, kind, name):
        path = self.source / kind / name / "year=2026" / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {"trade_date": ["2026-10-23"], "value": [1.0]}
        if kind == "factors":
            values["stock_code"] = ["000001.SZ"]
        pq.write_table(pa.table(values), path)

    def production(self, names):
        self.registry.parent.mkdir(parents=True, exist_ok=True)
        self.registry.write_text(json.dumps({"objects": [
            {"name": name, "status": "production"} for name in names
        ]}))

    def test_retired_stock_and_market_files_are_excluded(self):
        for name in ("active", "retired", "label_ret_1d"):
            self.product("factors", name)
        for name in ("market_active", "market_retired"):
            self.product("market_factors", name)
        self.production(["active", "label_ret_1d", "market_active"])
        cfg = preparing.Cfg(source_data=self.source)
        self.assertEqual(preparing.list_factor_names(cfg), ["active", "label_ret_1d"])
        self.assertEqual(preparing.market_block_columns(cfg=cfg),
                         ["market_active", "market_active_z252"])
        self.assertTrue(preparing.upstream_specs(cfg)["label_ret_1d"]["is_label"])
        with self.assertRaisesRegex(ValueError, "生产清单"):
            preparing.upstream_specs(cfg, extra_features=["retired"])

    def test_registered_object_without_output_fails(self):
        self.product("factors", "active")
        self.production(["active", "missing"])
        with self.assertRaisesRegex(ValueError, "missing"):
            preparing.list_factor_names(preparing.Cfg(source_data=self.source))

    def test_duplicate_production_names_fail(self):
        self.product("factors", "active")
        self.production(["active", "active"])
        with self.assertRaisesRegex(ValueError, "重复"):
            preparing.list_factor_names(preparing.Cfg(source_data=self.source))

    def test_independent_source_without_registry_uses_parquet(self):
        self.product("factors", "standalone")
        cfg = preparing.Cfg(source_data=self.source)
        self.assertIsNone(cfg.production_registry)
        self.assertEqual(preparing.list_factor_names(cfg), ["standalone"])


if __name__ == "__main__":
    unittest.main()
