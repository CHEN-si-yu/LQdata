"""残差教师的时间来源与AdamW断点可重复性。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('test_residual_model', ROOT / 'experiments/V262/model.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class ResidualLearningTests(unittest.TestCase):
    def test_selector_teacher_cannot_silently_use_delivery_weights(self):
        panel = SimpleNamespace(features=['a', 'b', 'c'], market_dim=2, days=np.asarray(['a', 'b', 'c']))
        recipe = {'snapshot_meta_sha256':'test'}
        signature = hashlib.sha256(json.dumps(recipe, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / 'experiments/V260'
            fold = parent / 'model_train/2025Q3/fold1'
            fold.mkdir(parents=True)
            (parent / 'model.py').write_text('RECIPE = ' + repr(recipe))
            done = dict(recipe_signature=signature, delivery_refit_complete=True,
                closed_form_fit=dict(selector_train_end='b', train_end='c'))
            (fold / 'complete.json').write_text(json.dumps(done))
            for filename, value, end in [('selector_best.pt', 1., 'b'), ('best.pt', 100., 'c')]:
                ck = dict(recipe_signature=signature, features=panel.features,
                    absolute_fit={'stages':[{'coef':np.zeros((2, 6)), 'fit_dates':2}]},
                    model=dict(linear_coef=model.torch.full((1, 4), value), absolute_coef=model.torch.zeros((2, 6))))
                model.torch.save(ck, fold / filename)
                model.torch.save(dict(train_end=end), fold / ('selector_market_risk.pt' if filename.startswith('selector') else 'market_risk.pt'))
            with patch.object(model, 'PROJECT_ROOT', root), patch.dict(model.RECIPE,
                    snapshot_meta_sha256='test', technical_indices=[0], residual_hidden=[8, 4]):
                net, _, source = model.frozen_teacher(panel, '2025Q3', 1, 'selector', np.arange(2))
                self.assertTrue(model.torch.equal(net.linear_coef, model.torch.ones((1, 4))))
                self.assertTrue(source['file'].endswith('selector_best.pt'))
                json.dumps(source, allow_nan=False)
                (fold / 'selector_best.pt').unlink()
                with self.assertRaises(FileNotFoundError):
                    model.frozen_teacher(panel, '2025Q3', 1, 'selector', np.arange(2))

    def test_interrupted_epoch_resume_matches_uninterrupted_dropout_training(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(8, 12, 3)).astype(np.float32)
        panel = SimpleNamespace(X=x, M=rng.normal(size=(8, 2)).astype(np.float32),
            Y={'label_ret_1d':rng.normal(size=(8, 12)).astype(np.float32)},
            days=np.asarray([str(i) for i in range(8)]), features=['a', 'b', 'c'], market_dim=2)
        panel.mask = lambda d:np.ones(12, bool)
        panel.inputs = lambda d, mask:np.ascontiguousarray(x[d, mask, None, :])
        def teacher(*args):
            net = model.PredictModel(3, 2)
            net.linear_coef.fill_(.1)
            net.ridge = None
            return net, {}, {'sha256':'frozen teacher', 'train_end':'7'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full, interrupted = root / 'model_train/2025Q3/fold1', root / 'model_train/2025Q3/fold2'
            full.mkdir(parents=True)
            interrupted.mkdir(parents=True)
            with patch.object(model, 'UNIT_ROOT', root), patch.object(model, 'frozen_teacher', side_effect=teacher), \
                    patch.dict(model.RECIPE, technical_indices=[0], residual_hidden=[8, 4], batch_days=4):
                expected, _, _ = model.fit_residual_stage(panel, None, '2025Q3', 1, np.arange(8), None,
                    model.torch.device('cpu'), full, 'delivery_refit', fixed_epochs=2)
                original_save = model.save_torch
                def stop_after_checkpoint(path, value):
                    original_save(path, value)
                    if Path(path).name == 'last.pt' and value['epoch'] == 0:
                        raise RuntimeError('simulated interruption after persisted epoch')
                with patch.object(model, 'save_torch', side_effect=stop_after_checkpoint):
                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                        model.fit_residual_stage(panel, None, '2025Q3', 1, np.arange(8), None,
                            model.torch.device('cpu'), interrupted, 'delivery_refit', fixed_epochs=2)
                actual, _, _ = model.fit_residual_stage(panel, None, '2025Q3', 1, np.arange(8), None,
                    model.torch.device('cpu'), interrupted, 'delivery_refit', fixed_epochs=2)
            for key, value in expected.state_dict().items():
                np.testing.assert_array_equal(value.numpy(), actual.state_dict()[key].numpy())
            histories = json.loads((interrupted / 'history.json').read_text())
            self.assertEqual(len(histories['delivery_refit']), 2)


if __name__ == '__main__':
    unittest.main()
