"""尾盘层与 3 个保留因子的最小口径回归。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fea.closing30 import Closing30Layer, EXPECTED_HM  # noqa: E402
from factors import closing30 as factors_closing30  # noqa: E402
from fea.spec import REGISTRY  # noqa: E402


def _bars() -> pd.DataFrame:
    hm = EXPECTED_HM
    df = pd.DataFrame({
        "stock_code": ["600000.SH"] * 48,
        "trade_time": [f"2025-09-24 {v // 100:02d}:{v % 100:02d}:00" for v in hm],
        "open": np.full(48, 10.0),
        "high": np.full(48, 10.0),
        "close": np.full(48, 10.0),
        "amount": np.r_[np.full(42, 100.0), np.full(6, 200.0)],
    })
    df.loc[45, "high"] = 11.0
    df.loc[47, "close"] = 10.4
    df.loc[47, "high"] = 10.4
    return df


class _Context:
    def __init__(self, row):
        self.row = row

    def closing30_field(self, field):
        return np.asarray([[self.row[field]]], dtype=np.float64)


class TestClosing30(unittest.TestCase):
    def setUp(self):
        self.layer = Closing30Layer.__new__(Closing30Layer)
        self.layer.codes = np.array(["600000.SH"])

    def _row(self, df):
        out = self.layer._aggregate_frame(df.sample(frac=1, random_state=7), 2025)
        self.assertEqual(len(out), 1)
        return out.iloc[0]

    def test_grid_and_tail_semantics(self):
        self.assertEqual(len(EXPECTED_HM), 48)
        np.testing.assert_array_equal(EXPECTED_HM[-6:],
                                      [1435, 1440, 1445, 1450, 1455, 1500])
        row = self._row(_bars())
        self.assertEqual(row.trade_date, "2025-09-24")
        self.assertEqual(row.n_bars, 48)
        self.assertEqual(row.grid_ok, 1)
        self.assertEqual(row.amt_day, 5400)
        self.assertEqual(row.amt_tail30, 1200)
        self.assertEqual(row.open_tail30, 10.0)
        self.assertEqual(row.high_tail30, 11.0)
        self.assertEqual(row.close_tail30, 10.4)
        ctx = _Context(row)
        # ★ 2026-10-03 清理复审：amt_share / crowded_fade（及其助手 _giveback）未通过并删除；
        #   本用例保留对留存对象 close30_ret 的数值断言，层聚合断言不变。
        self.assertAlmostEqual(factors_closing30.close30_ret(ctx)[0, 0], 0.04)

    def test_missing_or_duplicate_bar_is_not_a_valid_day(self):
        base = _bars()
        missing = self._row(base.drop(index=29))
        self.assertEqual(missing.n_bars, 47)
        self.assertEqual(missing.grid_ok, 0)
        self.assertTrue(np.isnan(missing.amt_day))

        duplicate = pd.concat([base.drop(index=29), base.iloc[[47]]], ignore_index=True)
        row = self._row(duplicate)
        self.assertEqual(row.n_bars, 48)
        self.assertEqual(row.grid_ok, 0)
        self.assertTrue(np.isnan(row.amt_tail30))

    def test_zero_day_vs_zero_tail_and_missing_price(self):
        df = _bars()
        df["amount"] = 0.0
        row = self._row(df)
        self.assertTrue(np.isnan(row.amt_day))
        self.assertTrue(np.isnan(row.open_tail30))

        df = _bars()
        df.loc[42:, "amount"] = 0.0
        row = self._row(df)
        self.assertEqual(row.amt_tail30, 0.0)
        # 尾段零成交 ⇒ 留存对象 close30_ret 不得给出伪造收益（开盘尾段缺失 → NaN）。
        self.assertTrue(np.isnan(row.open_tail30))
        self.assertTrue(np.isnan(factors_closing30.close30_ret(_Context(row))[0, 0]))

        df = _bars()
        df.loc[45, "high"] = np.nan
        row = self._row(df)
        self.assertTrue(np.isfinite(row.amt_tail30))
        self.assertTrue(np.isnan(row.high_tail30))

    def test_high_anomaly_is_expanded(self):
        df = _bars()
        df.loc[45, "high"] = 9.0
        df.loc[47, "high"] = 9.0
        row = self._row(df)
        self.assertEqual(row.high_tail30, 10.4)
        # _giveback 助手已随未通过的 crowded_fade 一并删除；留存对象 close30_ret 仍应给出正常收益。
        self.assertAlmostEqual(factors_closing30.close30_ret(_Context(row))[0, 0], 0.04)

    def test_registration_contract(self):
        # ★ 2026-10-03 清理复审后仅 close30_ret 留存；版本因「删未通过实现」身份变更已升至 3。
        for name in ("close30_ret",):
            spec = REGISTRY[name]
            self.assertEqual(spec.deps, ("stock_history_5min",))
            self.assertEqual(spec.warmup_days, 0)
            self.assertEqual(spec.version, 3)
            self.assertFalse(spec.is_market)


if __name__ == "__main__":
    unittest.main()
