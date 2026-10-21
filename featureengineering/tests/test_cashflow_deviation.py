"""Seasonal deviation must reject invalid RMSE without emitting warnings."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.families import r15cf_cashflow_income_structure_v1 as cf


class CashflowDeviationTests(unittest.TestCase):
    def compute(self, levels, phases):
        events = pd.DataFrame({
            "code": np.arange(len(levels)),
            "pit": np.full(len(levels), 20260930),
            "period": 100 + np.asarray(phases),
        })
        with patch.object(cf, "_level_at", return_value=levels):
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with np.errstate(divide="raise", invalid="raise"):
                    return cf._deviation_batch(
                        events, {}, "CF06_op_inflow_sales_collection_share_deviation")

    def test_zero_history_is_missing_without_invalid_division(self):
        levels = np.zeros((8, cf.WINDOW + 1))
        levels[4:, -1] = 1.0
        actual = self.compute(levels, np.tile(np.arange(4), 2))
        self.assertTrue(np.isnan(actual).all())

    def test_mixed_histories_preserve_valid_values_and_rmse_floor(self):
        rng = np.random.default_rng(15)
        valid = rng.normal(size=(4, cf.WINDOW + 1))
        small = valid * 1e-8
        missing = valid.copy()
        missing[:, 3] = np.nan
        levels = np.vstack([valid, small, missing, np.zeros_like(valid)])
        phases = np.tile(np.arange(4), 4)
        actual = self.compute(levels, phases)
        expected = []
        for phase, row in enumerate(valid):
            periods = 100 + phase - np.arange(cf.WINDOW, 0, -1)
            X = np.column_stack([
                np.ones(cf.WINDOW), np.arange(cf.WINDOW),
                *(periods % 4 == season for season in (1, 2, 3)),
            ])
            beta = np.linalg.lstsq(X, row[:-1], rcond=None)[0]
            residual = row[:-1] - X @ beta
            rmse = np.sqrt(np.sum(residual * residual) / cf.RESIDUAL_DF)
            forecast = np.array([1., 12., phase == 1, phase == 2, phase == 3]) @ beta
            expected.append((row[-1] - forecast) / rmse)
        np.testing.assert_allclose(actual[:4], expected, rtol=1e-12, atol=1e-12)
        self.assertTrue(np.isnan(actual[4:]).all())


if __name__ == "__main__":
    unittest.main()
