"""轻量验证开盘容量因子的时间边界、金额口径和缺失处理。"""

import unittest
from types import SimpleNamespace

import numpy as np

from fea.context import FactorContext
from fea.mathx import roll_quantile
from factors import open_capacity


def context(amount, *, traded=None, n_bars=None):
    amount = np.asarray(amount, dtype=np.float64)
    traded = np.ones_like(amount, dtype=bool) if traded is None else np.asarray(traded, dtype=bool)
    n_bars = np.full_like(amount, 48.0) if n_bars is None else np.asarray(n_bars, dtype=np.float64)
    fields = {"amt_open5": amount, "n_bars": n_bars}
    return SimpleNamespace(
        open5_field=lambda name: fields[name],
        traded=lambda: traded,
        roll_quantile=roll_quantile,
        safe_log=FactorContext.safe_log,
        safe_div=FactorContext.safe_div,
    )


class OpenCapacityTests(unittest.TestCase):
    def test_amount_quantile_and_scale_free_stability(self):
        a = np.arange(1, 26, dtype=np.float64).reshape(-1, 1) * 1e6
        ctx = context(a)
        capacity = open_capacity.open5_capacity_p20_log(ctx)[:, 0]
        stability = open_capacity.open5_capacity_floor_ratio_20(ctx)[:, 0]
        self.assertTrue(np.isnan(capacity[:19]).all())
        self.assertAlmostEqual(float(capacity[19]), np.log(4.8e6), places=11)
        self.assertAlmostEqual(float(stability[19]), 4.8 / 10.5, places=11)
        self.assertTrue(np.isfinite(capacity[19:]).all())
        self.assertTrue(((stability[19:] > 0) & (stability[19:] <= 1)).all())

    def test_no_future_bar_and_suspension_not_zero(self):
        a = np.arange(1, 26, dtype=np.float64).reshape(-1, 1) * 1e6
        traded = np.ones_like(a, dtype=bool)
        traded[20] = False
        base = open_capacity.open5_capacity_p20_log(context(a, traded=traded))
        changed_future = a.copy()
        changed_future[21:] = 1e12
        after = open_capacity.open5_capacity_p20_log(context(changed_future, traded=traded))
        np.testing.assert_array_equal(base[:21], after[:21])
        self.assertTrue(np.isnan(base[20, 0]))
        self.assertTrue(np.isfinite(base[21, 0]))

    def test_window_requires_fifteen_real_bars_and_current_validity(self):
        a = np.arange(1, 23, dtype=np.float64).reshape(-1, 1) * 1e6
        a[:6] = np.nan
        capacity = open_capacity.open5_capacity_p20_log(context(a))
        self.assertTrue(np.isnan(capacity[19, 0]))  # 20 日窗只有 14 个成交日
        self.assertTrue(np.isfinite(capacity[20, 0]))  # 20 日窗已有 15 个成交日
        n_bars = np.full_like(a, 48.0)
        n_bars[21] = 1.0
        capacity = open_capacity.open5_capacity_p20_log(context(a, n_bars=n_bars))
        self.assertTrue(np.isnan(capacity[21, 0]))


if __name__ == "__main__":
    unittest.main()
