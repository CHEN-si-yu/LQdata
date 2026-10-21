"""trim_recent_days 的两个核心状态函数：_sync_manifest 与 _trim_coverage。

删除工具的正确性全在这两处 —— manifest 字段口径必须与 engine._finalize 一致
（否则引擎的文件核对通道会判「输出损坏」而整年重算，对拍语义就错了），
coverage 截断必须精确（否则缺口检测的范围不对）。
端到端的「删除→增量→对拍」在生产上另有真实演练，本文件先把纯逻辑钉死。
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from trim_recent_days import _sync_manifest, _trim_coverage  # noqa: E402
from fea.manifest import Manifest  # noqa: E402


class SyncManifestTests(unittest.TestCase):
    def test_fields_match_frame_and_file(self):
        frame = pd.DataFrame({
            "trade_date": pd.Series(["2026-01-02", "2026-01-05"], dtype="string"),
            "stock_code": pd.Series(["600000.SH", "600001.SH"], dtype="string"),
            "value": np.array([1.0, np.nan], dtype="float32"),
            "rank": np.array([0.5, np.nan], dtype="float32"),
        })
        with tempfile.TemporaryDirectory() as temporary:
            d = Path(temporary)
            f = d / "data.parquet"
            frame.to_parquet(f, engine="pyarrow", index=False)
            man = Manifest(name="x", path=d / "x.json")
            _sync_manifest(man, 2026, frame, 3, f, "sha-test")
            part = man.partitions["2026"]
            self.assertEqual(part["rows"], 2)
            self.assertEqual(part["nonnull"], 1)          # NaN 不计 nonnull
            self.assertEqual(part["min_date"], "2026-01-02")
            self.assertEqual(part["max_date"], "2026-01-05")
            self.assertEqual(part["content_sha256"], "sha-test")
            st = f.stat()
            self.assertEqual(part["file_identity"], [st.st_size, st.st_mtime_ns])
            self.assertEqual(part["last_write"]["by"], "trim_recent_days")
            self.assertEqual(part["last_write"]["deleted_rows"], 3)


class TrimCoverageTests(unittest.TestCase):
    def _man(self, spans):
        man = Manifest(name="x", path=Path("/nonexistent/x.json"))
        man.coverage = [list(s) for s in spans]
        return man

    def test_tail_span_truncated_to_previous_day(self):
        man = self._man([["2018-01-02", "2018-12-28"],
                         ["2019-01-02", "2019-12-31"],
                         ["2020-01-02", "2020-12-31"]])
        _trim_coverage(man, "2020-12-25", "2020-12-24")
        self.assertEqual(man.coverage, [["2018-01-02", "2018-12-28"],
                                        ["2019-01-02", "2019-12-31"],
                                        ["2020-01-02", "2020-12-24"]])

    def test_span_entirely_inside_deleted_window_dropped(self):
        man = self._man([["2020-12-25", "2020-12-31"]])
        _trim_coverage(man, "2020-12-25", "2020-12-24")
        self.assertEqual(man.coverage, [])

    def test_untouched_spans_kept_verbatim(self):
        man = self._man([["2018-01-02", "2018-12-28"]])
        _trim_coverage(man, "2020-12-25", "2020-12-24")
        self.assertEqual(man.coverage, [["2018-01-02", "2018-12-28"]])


if __name__ == "__main__":
    unittest.main()
