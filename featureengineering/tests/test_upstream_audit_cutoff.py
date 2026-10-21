import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from fea.context import FactorContext
from fea.deriv import _load_one
from fea.derived import DerivedCache
from fea.panel import Panel
from fea.upstream import Upstream


class UpstreamAuditCutoffTests(unittest.TestCase):
    @staticmethod
    def _write(root, name, frame, year=None):
        folder = root / name if year is None else root / name / f"year={year}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "data.parquet"
        frame.to_parquet(path, index=False)
        return path

    def test_daily_narrow_reads_hide_future_and_cache_tracks_cutoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); raw = base / "raw"
            self._write(raw, "stock_daily", pd.DataFrame({"stock_code": ["A"], "trade_date": ["2025-12-31"], "close": [1.]}), 2025)
            self._write(raw, "stock_daily", pd.DataFrame({"stock_code": ["A"] * 3, "trade_date": ["2026-01-02", "2026-01-05", None], "close": [2., 5., 9.]}), 2026)
            up = Upstream(raw, base / "project")
            columns = ["close", "stock_code"]
            full = up.read("stock_daily", columns=columns)
            expected = pd.concat([pd.read_parquet(raw / "stock_daily/year=2025/data.parquet", columns=columns),
                                  pd.read_parquet(raw / "stock_daily/year=2026/data.parquet", columns=columns)], ignore_index=True)
            pd.testing.assert_frame_equal(full, expected)
            up.audit_cutoff = 20260102
            narrow = up.read("stock_daily", columns=columns)
            self.assertEqual(narrow.close.tolist(), [1., 2.])
            self.assertEqual(list(narrow.columns), columns)
            self.assertEqual(up.read("stock_daily", columns=columns, years=(2026, 2026)).close.tolist(), [2.])
            self.assertEqual(up.read("stock_daily").trade_date.tolist(), ["2025-12-31", "2026-01-02"])
            panel = Panel(np.array([20260102]), np.array(["A"]))
            ctx = FactorContext(panel, None, up, None, np.ones(panel.shape, bool))
            self.assertEqual(ctx.dataset("stock_daily", columns=["close"]).close.tolist(), [1., 2.])
            self.assertEqual(ctx.up.read("stock_daily", columns=["close"]).close.max(), 2.)
            up.audit_cutoff = 20260105
            self.assertEqual(up.read("stock_daily", columns=columns).close.tolist(), [1., 2., 5.])
            up.audit_cutoff = None
            self.assertIs(up.read("stock_daily", columns=columns), full)

    def test_financial_effective_announcement_matches_existing_vintage_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); raw = base / "raw"
            frame = pd.DataFrame({"stock_code": ["A"] * 7,
                                  "end_date": ["2025-12-31"] * 3 + ["2026-06-30"] + ["2025-12-31"] * 3,
                                  "ann_date": ["2026-03-01", None, "1970-01-01", "2026-03-01", "2026-03-01", "2026-07-01", None],
                                  "f_ann_date": ["2026-07-01", "2026-03-01", None, "2026-03-01", "2026-03-02", "2026-03-01", None],
                                  "total_assets": list(range(10, 17))})
            self._write(raw, "stock_balancesheet", frame, 2025)
            up = Upstream(raw, base / "project"); up.audit_cutoff = 20260501
            narrow = up.read("stock_balancesheet", columns=["total_assets", "stock_code"])
            self.assertEqual(narrow.total_assets.tolist(), [11, 14])
            self.assertEqual(list(narrow.columns), ["total_assets", "stock_code"])
            baseline = _load_one(SimpleNamespace(read=lambda *a, **k: frame.copy(), audit_cutoff=20260501),
                                 "stock_balancesheet", ["total_assets"], 2025, 2026)
            actual = _load_one(up, "stock_balancesheet", ["total_assets"], 2025, 2026)
            pd.testing.assert_frame_equal(actual.reset_index(drop=True), baseline.reset_index(drop=True))
            indicator = frame.drop(columns="f_ann_date").rename(columns={"total_assets": "bps"})
            self._write(raw, "stock_financial_indicator", indicator, 2025)
            self.assertEqual(up.read("stock_financial_indicator", columns=["bps"]).bps.tolist(), [10, 14])

    def test_forecast_uses_current_announcement_not_first_or_future_report_period(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); raw = base / "raw"
            self._write(raw, "stock_forecast", pd.DataFrame({"ann_date": ["2026-03-01", "2026-07-01"],
                        "first_ann_date": ["2026-02-01"] * 2, "end_date": ["2027-12-31"] * 2, "value": [1., 2.]}), 2027)
            self._write(raw, "stock_holder_number", pd.DataFrame({"ann_date": ["2026-03-01", "2026-07-01"],
                        "end_date": ["2025-12-31"] * 2, "holder_num": [10., 20.]}), 2025)
            up = Upstream(raw, base / "project"); up.audit_cutoff = 20260501
            self.assertEqual(up.read("stock_forecast", columns=["value"]).value.tolist(), [1.])
            self.assertEqual(up.read("stock_holder_number", columns=["holder_num"]).holder_num.tolist(), [10.])

    def test_minutes_include_entire_cutoff_day_and_restore_narrow_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); raw = base / "raw"
            for name, timestamps in (("stock_history_5min", ["2026-01-02 09:35:00", "2026-01-02 23:59:59", "2026-01-03 00:00:00", None]),
                                     ("tdx_minute", pd.to_datetime(["2026-01-02 09:35:00", "2026-01-02 15:00:00", "2026-01-03 09:35:00", None]))):
                self._write(raw, name, pd.DataFrame({"trade_time": timestamps, "value": [1., 2., 3., 4.]}), 2026)
            up = Upstream(raw, base / "project"); up.audit_cutoff = 20260102
            for name in ("stock_history_5min", "tdx_minute"):
                with self.subTest(name=name):
                    actual = up.read(name, columns=["value"])
                    self.assertEqual(actual.value.tolist(), [1., 2.])
                    self.assertEqual(list(actual.columns), ["value"])

    def test_event_date_modes_and_invalid_dates(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); raw = base / "raw"
            for name, column in (("stock_suspension", "suspend_date"), ("stock_adj_factor_changes", "date")):
                self._write(raw, name, pd.DataFrame({column: ["2026-01-02", "2026-01-03", "invalid", "2026-02-30"], "value": [1., 2., 3., 4.]}), 2026)
            up = Upstream(raw, base / "project"); up.audit_cutoff = 20260102
            for name in ("stock_suspension", "stock_adj_factor_changes"):
                self.assertEqual(up.read(name, columns=["value"]).value.tolist(), [1.])

    def test_calendar_and_stock_axis_contracts_remain_full_but_unknown_snapshots_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); raw = base / "raw"
            self._write(raw, "basic_calendar", pd.DataFrame({"date": ["2026-01-02", "2026-12-31"], "is_open": [1, 1]}))
            self._write(raw, "stock_list", pd.DataFrame({"stock_code": ["A", "B"], "list_date": ["2000-01-01", "2026-12-31"]}))
            self._write(raw, "dc_blocks", pd.DataFrame({"block_code": ["A"], "block_name": ["current"]}))
            self._write(raw, "stock_pledge_stat", pd.DataFrame({"end_date": ["2025-12-31"], "value": [1.]}), 2025)
            up = Upstream(raw, base / "project"); up.audit_cutoff = 20260102
            self.assertEqual(len(up.read("basic_calendar", columns=["is_open"])), 2)
            self.assertEqual(up.read("stock_list", columns=["stock_code"]).stock_code.tolist(), ["A", "B"])
            for name in ("dc_blocks", "stock_pledge_stat"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "历史可得日期"):
                    up.read(name)
            self.assertTrue(up.read("missing").empty)
            with self.assertRaises(Exception):
                up.read("stock_list", columns=["missing_column"])

    def test_watermark_stays_full_physical_identity_and_raw_files_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); raw = base / "raw"
            path = self._write(raw, "stock_daily", pd.DataFrame({"trade_date": ["2026-01-02", "2026-01-05"], "value": [1., 2.]}), 2026)
            before = (path.read_bytes(), path.stat().st_mtime_ns)
            up = Upstream(raw, base / "project")
            full = up.watermark("stock_daily")
            up.audit_cutoff = 20260102
            self.assertEqual(up.read("stock_daily").value.tolist(), [1.])
            self.assertEqual(up.watermark("stock_daily"), full)
            self.assertEqual(full["rows"], 2)
            self.assertEqual(full["max_pit"], 20260105)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_invalid_cutoffs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            up = Upstream(Path(temporary) / "raw", Path(temporary) / "project")
            for value in (True, "20260102", 20260230, 0):
                up.audit_cutoff = value
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "audit_cutoff"):
                    up.read("missing")


class DerivedAuditVisibilityTests(unittest.TestCase):
    def test_cutoff_fingerprint_invalidates_wider_cache_and_truncates_actual_parquet(self):
        class AuditCache(DerivedCache):
            name = "audit_fixture"

            def source_fingerprint(self, year):
                return {"exists": True, "raw_revision": 1}

            def build_year(self, year, up):
                self.__dict__.setdefault("build_calls", []).append(year)
                return pd.DataFrame({"trade_date": [f"{year}-01-02", f"{year}-01-03"], "value": [10., 20.]})

        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"FEA_CACHE_DIR": ""}):
            cfg = SimpleNamespace(root=Path(temporary), raw={}, compression="zstd")
            old = AuditCache(SimpleNamespace(audit_cutoff=None), cfg, np.array(["A"]))
            old.ensure(2026, 2026)
            self.assertNotIn("audit_cutoff", old._fp(2026))
            up = SimpleNamespace(audit_cutoff=20260102)
            narrow = AuditCache(up, cfg, np.array(["A"]))
            self.assertFalse(narrow.ready(2026, 2026))
            self.assertEqual(narrow._read_all(2026, 2026).value.tolist(), [10.])
            narrow.ensure(2026, 2026)
            self.assertTrue(narrow.ready(2026, 2026))
            self.assertEqual(pd.read_parquet(narrow.root / "year=2026/data.parquet").value.tolist(), [10.])
            up.audit_cutoff = 20260103
            self.assertFalse(narrow.ready(2026, 2026))
            narrow.ensure(2026, 2026)
            self.assertEqual(narrow._read_all(2026, 2026).value.tolist(), [10., 20.])
            self.assertEqual(narrow._fp(2026)["audit_cutoff"], 20260103)
            up.audit_cutoff = 20260101
            self.assertFalse(narrow.ready(2026, 2026))
            narrow.ensure(2026, 2026)
            self.assertTrue(narrow.ready(2026, 2026))
            self.assertTrue(narrow._read_all(2026, 2026).empty)
            self.assertTrue(pd.read_parquet(narrow.root / "year=2026/data.parquet").empty)
