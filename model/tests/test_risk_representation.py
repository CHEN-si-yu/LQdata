"""训练标签的秩语义与当日风格投影的实际不变量。"""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name + '_risk_test', ROOT / 'experiments' / name / 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RankTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load('V268')

    def test_training_target_ignores_monotone_amplitude_and_preserves_ties(self):
        y = np.asarray([-.2, -.1, 0., .1, .1, .8])
        a = self.model.gaussian_rank_label(y)
        b = self.model.gaussian_rank_label(np.exp(y))
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(a[3], a[4])
        np.testing.assert_array_equal(self.model.gaussian_rank_label(np.ones(8)), np.zeros(8))
        with self.assertRaises(ValueError):
            self.model.gaussian_rank_label(np.asarray([0., 1., np.nan]))


class RiskRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load('V269')

    def test_removes_only_observed_style_span_and_matches_cpu_torch(self):
        rng = np.random.default_rng(169)
        x = rng.normal(size=(100, 9)).astype(np.float32)
        x[:, 4] = x[:, 0] * 2  # 秩亏的风格基也必须稳定。
        indices = [0, 1, 4]
        with patch.dict(self.model.RECIPE, risk_style_indices=indices):
            residual = self.model.risk_neutral_features_numpy(x)
            basis = np.column_stack([np.ones(len(x)), x[:, indices]])
            np.testing.assert_allclose(basis.T @ residual, 0., atol=1e-10)
            np.testing.assert_allclose(self.model.risk_neutral_features_numpy(residual), residual, atol=1e-10)
            actual = self.model.risk_neutral_features_torch(self.model.torch.from_numpy(x)).numpy()
            np.testing.assert_allclose(actual, residual, atol=2e-7, rtol=2e-6)
            np.testing.assert_allclose(residual[:, indices], 0., atol=1e-12)
            self.assertGreater(np.linalg.norm(residual[:, 3]), 1.)

    def test_permutation_equivariance_and_no_future_day_dependency(self):
        rng = np.random.default_rng(269)
        panel = rng.normal(size=(3, 80, 7)).astype(np.float32)
        permutation = rng.permutation(80)
        with patch.dict(self.model.RECIPE, risk_style_indices=[0, 2]):
            original = self.model.risk_neutral_features_numpy(panel[0])
            panel[1:] = 1e8
            np.testing.assert_array_equal(original, self.model.risk_neutral_features_numpy(panel[0]))
            permuted = self.model.risk_neutral_features_numpy(panel[0, permutation])
            np.testing.assert_allclose(permuted, original[permutation], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
