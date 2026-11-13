"""验证两阶段交付不会把最终重训权重混入旧验证分数。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('selection_refit_model', ROOT / 'experiments/V260/model.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class SelectionRefitTests(unittest.TestCase):
    def test_delivery_cutoff_keeps_longest_label_before_test(self):
        days = pd.bdate_range('2018-01-01', '2026-10-23').strftime('%Y-%m-%d').to_numpy(dtype=str)
        for quarter in model.LAYOUT_QUARTERS:
            deliveries = []
            for split in model.splits(days, quarter=quarter):
                train = model.delivery_train_days(days, split)
                self.assertLess(train[-1] + 21, split['test'][0])
                self.assertGreater(train[-1], split['train'][-1])
                deliveries.append(train)
            self.assertTrue(all(np.array_equal(deliveries[0], d) for d in deliveries))

    def test_refit_changes_delivery_but_preserves_selector_predictions(self):
        rng = np.random.default_rng(942)
        t, n, width = 120, 24, 3
        x = rng.normal(size=(t, n, width)).astype(np.float32)
        y = .01 * x[:, :, 0]
        y[48:] = .025 * x[48:, :, 1]
        panel = SimpleNamespace(days=pd.bdate_range('2025-01-01', periods=t).strftime('%Y-%m-%d').to_numpy(dtype=str),
            codes=np.asarray([str(i) for i in range(n)]), features=['a', 'b', 'c'], X=x,
            M=rng.normal(size=(t, 1)).astype(np.float32), market_dim=1, market_features=['m'],
            Y={'label_ret_1d':y, 'label_ret_5d':y * 2})
        panel.mask = panel.technical_mask = lambda d:np.ones(n, bool)
        panel.inputs = lambda d, mask:np.ascontiguousarray(x[d, mask, None, :])
        split = dict(quarter='2025Q3', fold=1, train=np.arange(48), valid=np.arange(70, 80),
            test=np.arange(110, 120), score=np.arange(110, 120))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.multiple(model, UNIT_ROOT=folder, RUN_ROOT=folder), \
                 patch.object(model, 'splits', return_value=[split]), \
                 patch.object(model, 'training_info', return_value={}), \
                 patch.dict(model.RECIPE, technical_indices=[0], absolute_sample_caps=[24]), \
                 patch.object(model, 'validation', return_value={'val_wei':0.0}) as validation:
                done = model.train_fold(1, device='cpu', panel=panel,
                    prices=SimpleNamespace(hot_frac=1.), quarter='2025Q3')
                self.assertEqual(validation.call_count, 1)
                fold = folder / 'model_train/2025Q3/fold1'
                selected = model.torch.load(fold / 'selector_best.pt', weights_only=False)
                delivered = model.torch.load(fold / 'best.pt', weights_only=False)
                net = model.PredictModel(width, 1)
                net.ridge = None
                net.load_state_dict(selected['model'])
                selector_scores = model.predict(net, panel, SimpleNamespace(hot_frac=1.), split['valid'], model.torch.device('cpu'))
                frozen = np.load(fold / 'valid_predictions.npy')[:, :, 0]
                np.testing.assert_array_equal(selector_scores, frozen)
                net.load_state_dict(delivered['model'])
                refit_scores = model.predict(net, panel, SimpleNamespace(hot_frac=1.), split['valid'], model.torch.device('cpu'))
                self.assertGreater(float(np.max(np.abs(refit_scores - frozen))), .1)
                selector_risk = model.torch.load(fold / 'selector_market_risk.pt', weights_only=False)
                delivered_risk = model.torch.load(fold / 'market_risk.pt', weights_only=False)
                self.assertEqual(selector_risk['train_end'], panel.days[47])
                self.assertEqual(delivered_risk['train_end'], panel.days[88])
                self.assertTrue(done['delivery_refit_complete'])
                # 完成后重入不会重训或把最终权重覆盖回选择权重。
                reloaded = model.train_fold(1, device='cpu', panel=panel,
                    prices=SimpleNamespace(hot_frac=1.), quarter='2025Q3')
                self.assertEqual(reloaded['closed_form_fit'], done['closed_form_fit'])
                np.testing.assert_array_equal(np.load(fold / 'valid_predictions.npy')[:, :, 0], frozen)

    def test_validation_refuses_missing_selector_risk_instead_of_fallback(self):
        spec = importlib.util.spec_from_file_location('selection_refit_analysis', ROOT / 'experiments/V260/analysis.py')
        analysis = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, model=model):
            spec.loader.exec_module(analysis)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'selector_complete.json').write_text(json.dumps({'recipe_signature':model.recipe_signature()}))
            model.torch.save({'unsafe':'final refit is present'}, folder / 'market_risk.pt')
            with self.assertRaises(FileNotFoundError):
                analysis.selector_market_risk(folder)


if __name__ == '__main__':
    unittest.main()
