import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from fea.derived import DerivedCache


class DummyCache(DerivedCache):
    name = "dummy"
    version = 7

    def source_fingerprint(self, year):
        return {"exists": True, "raw_revision": 1, "year": year}

    def build_year(self, year, up):
        self.__dict__.setdefault("build_calls", []).append(year)
        return pd.DataFrame({"trade_date": pd.Series([f"{year}-01-02"], dtype="string"),
                             "stock_code": pd.Series(self.codes, dtype="string"),
                             "aggregate": np.array([self.cfg.raw.get("compute", {}).get("kernel_version", 1)], dtype=np.float32)})


class DerivedKernelVersionTests(unittest.TestCase):
    @staticmethod
    def _cfg(root, version=None):
        raw = {"paths": {"derived": "data/derived"}}
        if version is not None:
            raw["compute"] = {"kernel_version": version}
        return SimpleNamespace(root=root, raw=raw, compression="zstd")

    def test_v1_manifest_and_ready_cache_remain_exactly_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"FEA_CACHE_DIR": ""}):
            root = Path(temporary)
            original = DummyCache(None, self._cfg(root), np.array(["600001.SH"]))
            original.ensure(2025, 2026)
            before_manifest = original.man_path.read_bytes()
            paths = [original.root / f"year={year}" / "data.parquet" for year in (2025, 2026)]
            before_files = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
            unchanged = DummyCache(None, self._cfg(root, 1), np.array(["600001.SH"]))
            self.assertIs(type(unchanged.version), int)
            self.assertEqual(unchanged.version, 7)
            self.assertTrue(unchanged.ready(2025, 2026))
            unchanged.ensure(2025, 2026)
            self.assertEqual(unchanged.__dict__.get("build_calls", []), [])
            self.assertEqual(unchanged.man_path.read_bytes(), before_manifest)
            self.assertEqual([(path.read_bytes(), path.stat().st_mtime_ns) for path in paths], before_files)
            self.assertEqual(json.loads(before_manifest)["version"], 7)

    def test_v2_invalidates_each_old_year_and_ensure_rebuilds_real_parquet(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"FEA_CACHE_DIR": ""}):
            root = Path(temporary)
            old = DummyCache(None, self._cfg(root, 1), np.array(["600001.SH"]))
            old.ensure(2025, 2026)
            new = DummyCache(None, self._cfg(root, 2), np.array(["600001.SH"]))
            self.assertEqual(new.version, "7||common-kernels=v2")
            self.assertEqual(DummyCache.version, 7)
            self.assertFalse(new.ready(2025, 2026))
            new.ensure(2025, 2025)
            self.assertEqual(new.build_calls, [2025])
            self.assertTrue(new.ready(2025, 2025))
            self.assertFalse(new.ready(2026, 2026))
            self.assertEqual(pd.read_parquet(new.root / "year=2025/data.parquet")["aggregate"].iloc[0], 2.)
            self.assertEqual(pd.read_parquet(new.root / "year=2026/data.parquet")["aggregate"].iloc[0], 1.)
            new.ensure(2026, 2026)
            self.assertEqual(new.build_calls, [2025, 2026])
            self.assertTrue(new.ready(2025, 2026))
            self.assertEqual(pd.read_parquet(new.root / "year=2026/data.parquet")["aggregate"].iloc[0], 2.)
            manifest = json.loads(new.man_path.read_text())
            self.assertEqual(manifest["version"], "7||common-kernels=v2")
            self.assertEqual(set(manifest["years"]), {"2025", "2026"})
            reloaded = DummyCache(None, self._cfg(root, 2), np.array(["600001.SH"]))
            self.assertTrue(reloaded.ready(2025, 2026))
            reloaded.ensure(2025, 2026)
            self.assertEqual(reloaded.__dict__.get("build_calls", []), [])
            later = DummyCache(None, self._cfg(root, 3), np.array(["600001.SH"]))
            self.assertEqual(later.version, "7||common-kernels=v3")
            self.assertFalse(later.ready(2025, 2026))

    def test_all_production_derived_layers_inherit_kernel_identity(self):
        from fea.chips import ChipLayer
        from fea.chipwall import ChipWallLayer
        from fea.closing30 import Closing30Layer
        from fea.intraday import IntradayLayer
        from fea.market_morning import MorningLayer
        from fea.open5 import Open5Layer
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"FEA_CACHE_DIR": ""}):
            for cls in (ChipLayer, ChipWallLayer, Closing30Layer, IntradayLayer, MorningLayer, Open5Layer):
                with self.subTest(layer=cls.__name__):
                    self.assertIs(cls.__init__, DerivedCache.__init__)
                    old = cls(None, self._cfg(Path(temporary), 1), np.array(["600001.SH"]))
                    self.assertIs(type(old.version), int)
                    self.assertEqual(old.version, cls.version)
                    new = cls(None, self._cfg(Path(temporary), 2), np.array(["600001.SH"]))
                    self.assertEqual(new.version, f"{cls.version}||common-kernels=v2")

    def test_versioned_empty_rebuild_replaces_old_rows_and_in_memory_frame(self):
        class EmptyCache(DummyCache):
            def build_year(self, year, up):
                self.__dict__.setdefault("build_calls", []).append(year)
                if self.cfg.empty_mode == "none":
                    return None
                if self.cfg.empty_mode == "no_columns":
                    return pd.DataFrame()
                return pd.DataFrame({"trade_date": pd.Series(dtype="string"),
                                     "stock_code": pd.Series(dtype="string"),
                                     "aggregate": pd.Series(dtype="float32")})

        for mode in ("declared_columns", "no_columns", "none"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"FEA_CACHE_DIR": ""}):
                root = Path(temporary)
                old = DummyCache(None, self._cfg(root, 1), np.array(["600001.SH"]))
                old.ensure(2025, 2025)
                cfg = self._cfg(root, 2)
                cfg.empty_mode = mode
                new = EmptyCache(None, cfg, np.array(["600001.SH"]))
                self.assertFalse(new.ready(2025, 2025))
                # Even a previously materialized frame must stop exposing old values.
                self.assertEqual(new._read_all(2025, 2025)["aggregate"].iloc[0], 1.)
                new._cache[(2025,)] = np.array([1.])
                new.ensure(2025, 2025)
                empty = pd.read_parquet(new.root / "year=2025/data.parquet")
                self.assertTrue(empty.empty)
                self.assertEqual(list(empty.columns), ["trade_date", "stock_code", "aggregate"])
                self.assertEqual(str(empty["aggregate"].dtype), "float32")
                self.assertTrue(new.ready(2025, 2025))
                self.assertTrue(new._read_all(2025, 2025).empty)
                self.assertEqual(new._cache, {})
                new.ensure(2025, 2025)
                self.assertEqual(new.build_calls, [2025])
                self.assertEqual(json.loads(new.man_path.read_text())["version"], "7||common-kernels=v2")

    def test_common_kernel_version_requires_positive_integer(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"FEA_CACHE_DIR": ""}):
            for invalid in (0, -1, True, False, 1.5, "2", None):
                cfg = self._cfg(Path(temporary))
                cfg.raw["compute"] = {"kernel_version": invalid}
                with self.subTest(version=invalid), self.assertRaisesRegex(ValueError, "kernel_version"):
                    DummyCache(None, cfg, np.array(["600001.SH"]))
