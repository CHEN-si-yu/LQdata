"""dayhash 全历史模式：`_hash_all_one` 与 `_hash_one` 必须逐位一致。

全历史台账（--all）用于「重建前后逐格对拍」，一旦它的哈希口径与既有的
逐日过滤路径有任何差异，对拍就会产生假红/假绿 —— 本文件把这个等价性钉死。
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fea import store
from fea.dayhash import _hash_all_one, _hash_one


class DayHashAllTests(unittest.TestCase):
    @staticmethod
    def _write(root, name, year, frame):
        path = store.year_path(root, name, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, engine="pyarrow", index=False, row_group_size=2)
        return path

    def test_stock_all_mode_matches_requested_days(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.DataFrame({
                "trade_date": ["2026-01-02", "2026-01-02", "2026-01-05", "2026-01-05",
                               "2026-01-06", "2026-01-06"],
                "stock_code": pd.Categorical(["600002.SH", "600001.SH"] * 3),
                "value": [-0., np.nan, 3.5, np.inf, -np.inf, 0.125],
                "rank": [np.nan, 0.5, 0.75, 1., 0., np.nan],
            })
            self._write(root, "stock", 2026, frame.iloc[[4, 2, 0, 5, 3, 1]])
            days = ("2026-01-02", "2026-01-05", "2026-01-06")
            self.assertEqual(_hash_all_one((str(root), "stock", days)),
                             _hash_one((str(root), "stock", days)))

    def test_market_all_mode_matches_requested_days(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.DataFrame({
                "trade_date": pd.Series(["2026-01-02", "2026-01-05", "2026-01-06"], dtype="string"),
                "value": pd.Series([0.5, np.nan, -1.25], dtype="float32"),
            })
            self._write(root, "mkt", 2026, frame.iloc[[2, 0, 1]])
            days = ("2026-01-05", "2026-01-06")
            self.assertEqual(_hash_all_one((str(root), "mkt", days)),
                             _hash_one((str(root), "mkt", days)))

    def test_all_mode_skips_days_outside_request_and_missing_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.DataFrame({
                "trade_date": pd.Series(["2026-01-02", "2026-01-05"], dtype="string"),
                "stock_code": pd.Series(["600000.SH", "600000.SH"], dtype="string"),
                "value": pd.Series([1.0, 2.0], dtype="float32"),
                "rank": pd.Series([0.5, 1.0], dtype="float32"),
            })
            self._write(root, "stock", 2026, frame)
            rows = _hash_all_one((str(root), "stock", ("2026-01-05",)))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1], "2026-01-05")
            # 分区缺失的年份直接跳过，不报错（晚起点因子早年份没有分区是常态）。
            self.assertEqual(_hash_all_one((str(root), "stock", ("2019-01-02",))), [])


if __name__ == "__main__":
    unittest.main()
