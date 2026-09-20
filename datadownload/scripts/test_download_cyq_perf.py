"""Offline backfill tests; no production paths or network calls."""
import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from data_incremental import paths, registry
from data_incremental.core import client as C

spec = importlib.util.spec_from_file_location("cyq_full", Path(__file__).with_name("download_cyq_perf.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class DownloadTests(unittest.TestCase):
    def rows(self):
        return [{**dict.fromkeys(M.FIELDS, 85.11), "trade_date": d, "stock_code": "000001.SZ"}
                for d in ("2018-01-02", "2018-01-20")]

    def test_validate_duplicate_and_wrong_dates(self):
        row = self.rows()[0]
        with self.assertRaises(ValueError):
            M.validate(pd.DataFrame([row, row]), "2018-01-01", "2018-01-31")
        with self.assertRaises(ValueError):
            M.validate(pd.DataFrame([row]), "2018-01-03", "2018-01-31")

    def test_incomplete_pagination_refused(self):
        class Client:
            def fetch_all(*args):
                return C.FetchRows([], complete=False)
        with self.assertRaises(RuntimeError):
            M.download(Client(), registry.get("stock_cyq_perf"), "2018-01-01", "2018-01-31")

    def test_row_limit_splits_dates(self):
        rows = self.rows()
        class Client:
            def fetch_all(self, path, payload, size):
                a, b = payload["start_time"], payload["end_time"]
                if a < "2018-01-15" < b:
                    raise C.QueryLimitError("injected cap")
                return [r for r in rows if a <= r["trade_date"] <= b]
        self.assertEqual(len(M.download(Client(), registry.get("stock_cyq_perf"),
                                        "2018-01-01", "2018-01-31")), 2)

    def test_resume_repairs_missing_partition_without_refetch(self):
        rows = self.rows()
        counts = {"fetch": 0}
        class Client:
            def __init__(self, cfg): self.stats = {"requests": 0}
            def fetch_all(self, path, payload, size):
                counts["fetch"] += 1
                return [r for r in rows if payload["start_time"] <= r["trade_date"] <= payload["end_time"]]
            def call(self, *args): return {"total": len(rows)}
            def close(self): pass
        with tempfile.TemporaryDirectory() as t:
            patches = []
            try:
                for name, value in list(vars(paths).items()):
                    if isinstance(value, Path):
                        try: rel = value.relative_to("/autodl-fs/data")
                        except ValueError: continue
                        p = patch.object(paths, name, Path(t) / rel)
                        p.start(); patches.append(p)
                args = Namespace(start="2018-01-01", end="2018-01-20", refresh=False)
                with patch.object(C, "Client", Client):
                    M.run(args)
                    dest = paths.year_partition_path("stock_cyq_perf", 2018)
                    self.assertEqual(len(pd.read_parquet(dest)), 2)
                    dest.unlink()
                    M.run(args)
                    self.assertEqual(len(pd.read_parquet(dest)), 2)
                    self.assertEqual(counts["fetch"], 2)
                    self.assertEqual(pd.read_parquet(dest).winner_rate.iloc[0], 85.11)
                    report = json.loads((paths.STATE_ROOT / "cyq_perf_full/report.json").read_text())
                    self.assertTrue(report["complete"])
            finally:
                for p in reversed(patches): p.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
