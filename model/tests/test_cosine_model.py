"""固定非线性表示须支持交互、正确日权重及独立权重重载。"""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cosine_model_test', ROOT/'experiments/V271/model.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class CosineModelTests(unittest.TestCase):
    def test_convex_output_can_recover_interaction_missed_by_linear_map(self):
        x = model.torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        y = x[:, 0] * x[:, 1]
        linear = model.torch.cat([x, model.torch.ones((4, 1))], 1)
        self.assertLess(model.torch.linalg.norm(linear.T @ y).item(), 1e-12)
        z = model.cosine_features(x).numpy()
        phi = np.column_stack([z, np.ones(4)])
        coefficient = np.linalg.solve(phi.T @ phi + .1*np.eye(phi.shape[1]), phi.T @ y.numpy())
        np.testing.assert_allclose(phi @ coefficient, y.numpy(), atol=.02)

    def test_day_weight_does_not_enter_nonlinear_activation(self):
        x = model.torch.tensor([[1., -.4], [.2, .9], [-.5, -.8]])
        weights = model.torch.tensor([[.1], [.2], [.3]])
        actual = model.weighted_cosine_phi(model.torch.cat([x, weights], 1))
        expected = model.torch.cat([model.cosine_features(x), model.torch.ones_like(weights)], 1)*weights
        model.torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hidden_map_and_final_prediction_reload_without_global_rng_dependence(self):
        torch = model.torch
        with patch.dict(model.RECIPE, technical_indices=[0]):
            torch.manual_seed(79)
            state_before = torch.random.get_rng_state()
            net = model.PredictModel(2, 0)
            self.assertTrue(torch.equal(state_before, torch.random.get_rng_state()))
            net.linear_coef.copy_(torch.arange(net.linear_coef.numel(), dtype=torch.float32)[None]/1000)
            x = torch.tensor([[[.2, .6]], [[-.4, -.1]], [[1., -.8]]])
            expected = net(x)
            saved = {k:v.clone() for k,v in net.state_dict().items()}
            torch.manual_seed(989)
            restored = model.PredictModel(2, 0)
            restored.feature_projection.zero_()
            restored.load_state_dict(saved)
            torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
