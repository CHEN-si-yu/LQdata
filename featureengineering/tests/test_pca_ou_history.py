"""PCA-OU donor training requires 252 sessions before the current month."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from research.families import r15n14_crossfit_pca_ou_displacement_v1 as ou


class PcaOuHistoryTests(unittest.TestCase):
    def compute(self, r, month_start):
        days = np.full(len(r), 20171229, dtype=np.int64)
        days[month_start:] = 20180102
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(divide="raise", invalid="raise"):
                return ou.panel_values(
                    r, np.ones_like(r, dtype=bool), days,
                    np.arange(r.shape[1]).astype(str), np.arange(r.shape[1]),
                    [201801])

    def test_insufficient_history_skips_std_and_keeps_missing(self):
        rng = np.random.default_rng(14)
        for size, month_start in [(300, 0), (300, 100), (100, 50), (252, 251)]:
            with self.subTest(size=size, month_start=month_start):
                r = rng.normal(0, .01, (size, 10))
                with patch.object(ou.np, "std", side_effect=AssertionError("short window")):
                    values, controls, qa = self.compute(r, month_start)
                self.assertTrue(np.isnan(values).all())
                for control in controls.values():
                    self.assertTrue(np.isnan(control).all())
                self.assertEqual([item['donors'] for item in qa], [0] * 5)

    def test_exactly_252_prior_sessions_keeps_donor_selection(self):
        r = np.random.default_rng(14).normal(0, .01, (253, 10))
        r[:, 0] = 0.0
        r[25, 1] = np.nan
        with patch.object(ou.np, "std", wraps=np.std) as std:
            values, _, qa = self.compute(r, 252)
        self.assertEqual(std.call_count, 5)
        self.assertTrue(all(call.args[0].shape == (252, 10)
                            for call in std.call_args_list))
        # Both unusable columns are excluded; each fold excludes its own codes.
        self.assertEqual([item['donors'] for item in qa], [7, 7, 6, 6, 6])
        self.assertTrue(np.isnan(values).all())  # Fewer than 500 donors.


if __name__ == "__main__":
    unittest.main()
