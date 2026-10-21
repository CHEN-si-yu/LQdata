import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fea import store
from fea.dayhash import _day_md5, _hash_one, _verify


class DayHashScopeTests(unittest.TestCase):
    @staticmethod
    def _write(root, name, year, frame):
        path = store.year_path(root, name, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Small groups make requested and excluded days independent in Parquet.
        frame.to_parquet(path, engine="pyarrow", index=False, row_group_size=2)
        return path

    @staticmethod
    def _full_read_baseline(root, name, days):
        frame = store.read_factor(root, name, years=sorted({int(day[:4]) for day in days}))
        frame = frame[frame.trade_date.isin(set(days))]
        return [(name, str(day), *_day_md5(sub))
                for day, sub in frame.groupby("trade_date", sort=True)]

    def test_stock_requested_days_match_full_read_with_nan_and_legacy_dtypes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Unsorted stock codes and dates, category codes and float64 values
            # exercise the same read-time normalization as store.read_year.
            frame = pd.DataFrame({
                "trade_date": ["2026-01-02", "2026-01-02", "2026-01-05", "2026-01-05",
                               "2026-01-06", "2026-01-06"],
                "stock_code": pd.Categorical(["600002.SH", "600001.SH"] * 3),
                "value": [-0., np.nan, 3.5, np.inf, -np.inf, 0.125],
                "rank": [np.nan, 0.5, 0.75, 1., 0., np.nan],
            })
            self._write(root, "stock", 2026, frame.iloc[[4, 2, 0, 5, 3, 1]])
            days = ("2026-01-06", "2026-01-02", "2026-01-07", "2026-01-02")
            expected = self._full_read_baseline(root, "stock", days)
            self.assertEqual(_hash_one((str(root), "stock", days)), expected)
            self.assertEqual([row[1] for row in expected], ["2026-01-02", "2026-01-06"])
            self.assertEqual([row[3] for row in expected], [2, 2])

    def test_market_requested_days_match_full_read_with_nan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.DataFrame({"trade_date": pd.Series(["2026-01-02", "2026-01-05", "2026-01-06"], dtype="string"),
                                  "value": np.array([np.nan, 0.25, -0.], dtype=np.float32)})
            self._write(root, "market", 2026, frame)
            days = ("2026-01-06", "2026-01-02")
            expected = self._full_read_baseline(root, "market", days)
            self.assertEqual(_hash_one((str(root), "market", days)), expected)
            self.assertEqual([row[3] for row in expected], [1, 1])

    def test_cross_year_stock_and_market_skip_unrequested_corrupt_year_and_missing_year(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            days = ("2026-01-02", "2025-12-31", "2027-01-04")
            for market in (False, True):
                name = "market" if market else "stock"
                for year, dates in ((2025, ["2025-12-30", "2025-12-31"]),
                                    (2026, ["2026-01-02", "2026-01-05"])):
                    frame = pd.DataFrame({"trade_date": pd.Series(dates, dtype="string"),
                                          "value": np.array([np.nan, 1.125], dtype=np.float32)})
                    if not market:
                        frame.insert(1, "stock_code", pd.Series(["600001.SH"] * 2, dtype="string"))
                        frame["rank"] = np.array([0.5, np.nan], dtype=np.float32)
                    self._write(root, name, year, frame)
                unrequested = store.year_path(root, name, 2019)
                unrequested.parent.mkdir(parents=True, exist_ok=True)
                unrequested.write_bytes(b"corrupt year outside requested scope")
                with self.subTest(market=market):
                    expected = self._full_read_baseline(root, name, days)
                    self.assertEqual(_hash_one((str(root), name, days)), expected)
                    self.assertEqual([row[1] for row in expected], ["2025-12-31", "2026-01-02"])

    def test_missing_factor_empty_request_and_no_matching_day_return_no_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(_hash_one((str(root), "missing", ("2026-01-02",))), [])
            self.assertEqual(_hash_one((str(root), "missing", ())), [])
            frame = pd.DataFrame({"trade_date": ["2026-01-02"], "value": [1.]})
            self._write(root, "market", 2026, frame)
            self.assertEqual(_hash_one((str(root), "market", ("2026-01-05",))), [])

    def test_requested_corrupt_file_and_invalid_schema_still_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = store.year_path(root, "corrupt", 2026)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not a Parquet file")
            with self.assertRaises(Exception):
                _hash_one((str(root), "corrupt", ("2026-01-02",)))
            self._write(root, "bad_schema", 2026,
                        pd.DataFrame({"trade_date": ["2026-01-02"], "unexpected": [1.]}))
            with self.assertRaisesRegex(ValueError, "未知因子文件格式"):
                _hash_one((str(root), "bad_schema", ("2026-01-05",)))

    def test_market_duplicate_requested_day_still_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root, "market", 2026,
                        pd.DataFrame({"trade_date": ["2026-01-02"] * 2, "value": [1., 2.]}))
            with self.assertRaisesRegex(ValueError, "恰有一行"):
                _hash_one((str(root), "market", ("2026-01-02",)))

    def test_missing_baseline_does_not_pass_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertNotEqual(_verify(Path(temporary) / "dayhash.tsv", []), 0)
