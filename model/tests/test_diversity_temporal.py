"""IC与相关约束的有限梯度，以及时间卷积的严格因果性。"""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, folder):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'experiments' / folder / 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diversity = load('test_diversity_model', 'V263')
temporal = load('test_temporal_model', 'V264')


class DiversityTemporalTests(unittest.TestCase):
    def test_zero_initialized_head_has_finite_nonzero_ic_gradient(self):
        p = diversity.torch.zeros(64, requires_grad=True)
        target = diversity.torch.linspace(-1., 1., 64)
        teacher = target.square()
        loss, _, _ = diversity.ic_diversity_loss(p, target, teacher, [32, 32])
        loss.backward()
        self.assertTrue(diversity.torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum()), 0.)

    def test_equal_ic_prefers_lower_parent_dependence(self):
        angles = np.arange(64) * 2 * np.pi / 64
        vectors = np.column_stack([np.sin(angles), np.cos(angles), np.sin(2 * angles)])
        u, v, w = [diversity.torch.tensor(vectors[:, i], dtype=diversity.torch.float64) for i in range(3)]
        high = .5 * u + np.sqrt(.75) * v
        low = .5 * u + np.sqrt(.75) * w
        high_loss, high_ic, _ = diversity.ic_diversity_loss(high, u, v, [64])
        low_loss, low_ic, _ = diversity.ic_diversity_loss(low, u, v, [64])
        self.assertAlmostEqual(float(high_ic), float(low_ic), places=7)
        self.assertGreater(float(high_loss - low_loss), .02)

    def test_later_sessions_do_not_change_earlier_temporal_states(self):
        with patch.dict(temporal.RECIPE, temporal_width=8, technical_indices=[0], residual_hidden=[8, 4]):
            net = temporal.PredictModel(3, 2).eval()
        temporal.torch.manual_seed(72)
        x = temporal.torch.randn(3, 7, 3)
        before = net.temporal_states(x).detach()
        changed = x.clone()
        changed[:, 3:] *= -1
        after = net.temporal_states(changed).detach()
        np.testing.assert_array_equal(before[:, :3].numpy(), after[:, :3].numpy())
        self.assertGreater(float((before[:, -1] - after[:, -1]).abs().max()), 1e-5)
        changed = x.clone()
        changed[:, 0] *= -1
        first_changed = net.temporal_states(changed).detach()
        self.assertGreater(float((before[:, -1] - first_changed[:, -1]).abs().max()), 1e-5)


if __name__ == '__main__':
    unittest.main()
