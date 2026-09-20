import unittest
from fea.engine import Engine

class BaselineTests(unittest.TestCase):
    def test_missing_price_baseline_fails_closed(self):
        e=object.__new__(Engine)
        e.watermark=lambda dep: {'max_pit':0}
        with self.assertRaisesRegex(RuntimeError,'stock_daily'):
            e.baseline_last_day()
        e.watermark=lambda dep: {'max_pit':20260918}
        self.assertEqual(e.baseline_last_day(),20260918)
