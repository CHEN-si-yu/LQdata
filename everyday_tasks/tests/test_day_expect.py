"""保护验收口径：不把空基线、跨日 total 或不完整分页当成通过。"""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('day_expect_test', ROOT / 'scripts/day_expect.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class ExpectationTests(unittest.TestCase):
    def test_uses_effective_calendar_instead_of_only_observed_days(self):
        with patch.object(helper.state, 'effective_calendar', return_value=['2026-10-12', '2026-10-13']):
            self.assertEqual(helper.calendar(), ['2026-10-12', '2026-10-13'])

    def test_exact_rows_filters_day_and_deduplicates_business_keys(self):
        ds = helper.R.get('stock_st_info')
        rows = [{'trade_date': '2026-10-12', 'stock_code': '000001.SZ'},
                {'trade_date': '2026-10-13', 'stock_code': '000001.SZ'},
                {'trade_date': '2026-10-13', 'stock_code': '000001.SZ'}]
        client = SimpleNamespace(fetch_all=lambda *args, **kwargs: rows)
        self.assertEqual(helper.exact_day_rows(client, ds, '2026-10-13', []), 1)

    def test_incomplete_rows_cannot_pass(self):
        class Incomplete(list):
            complete = False
        client = SimpleNamespace(fetch_all=lambda *args, **kwargs: Incomplete())
        with self.assertRaisesRegex(RuntimeError, '分页不完整'):
            helper.exact_day_rows(client, helper.R.get('stock_daily'), '2026-10-13', [])

    def test_per_date_scalar_code_array_uses_requested_day(self):
        client = SimpleNamespace(fetch_all=lambda *args, **kwargs: ['000001.SZ', '000001.SZ', '000002.SZ'])
        self.assertEqual(helper.exact_day_rows(client, helper.R.get('stock_adj_factor_changes'),
                                             '2026-10-13', []), 2)

    def test_non_daily_holder_count_is_not_single_trade_day_comparable(self):
        self.assertFalse(helper.comparable_day(helper.R.get('stock_holder_number')))
        self.assertTrue(helper.comparable_day(helper.R.get('index_daily')))

    def test_snapshot_rejects_date_missing_from_calendar(self):
        with patch.object(helper, 'calendar', return_value=[]):
            with self.assertRaisesRegex(ValueError, '空基线'):
                helper.snapshot('2026-10-13')

    def test_verify_rejects_empty_pre_run_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'expect_2026-10-13.json').write_text(json.dumps({'tail_days': []}))
            with patch.object(helper, 'BACKTEST', root):
                self.assertEqual(helper.verify('2026-10-13'), 2)


if __name__ == '__main__':
    unittest.main()
