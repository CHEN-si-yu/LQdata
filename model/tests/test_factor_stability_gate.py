"""Only mature training-block evidence may select factors; raw auxiliary paths stay intact."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
model=load('factor_stability_test',ROOT/'experiments/V282/model.py')
parent=load('factor_stability_parent',ROOT/'experiments/V263/model.py')


def panel_fixture():
    rng=np.random.default_rng(61)
    x=rng.normal(size=(24,32,3)).astype(np.float32)
    x[:,:,1]=-x[:,:,0]
    y=x[:,:,0]+.01*rng.normal(size=(24,32))
    panel=SimpleNamespace(X=x,Y={'label_ret_1d':y},features=['positive','negative','noise'],
        days=np.asarray([str(i) for i in range(24)]))
    panel.mask=lambda day:np.ones(32,bool)
    return panel


class FactorStabilityTests(unittest.TestCase):
    def test_daily_pearson_matches_direct_clipped_input_statistic(self):
        panel=panel_fixture();panel.X[0,0,0]=20
        actual=model.daily_factor_ic(panel,0)
        expected=[np.corrcoef(np.clip(panel.X[0,:,column],-5,5),panel.Y['label_ret_1d'][0])[0,1] for column in range(3)]
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_both_signs_survive_and_later_statistics_do_not_change_gate(self):
        panel=SimpleNamespace(days=np.asarray([str(i) for i in range(24)]),features=list('abcde'))
        ic=np.tile([.1,-.05,.1,0,.0001],(24,1));ic[12:16,2]=-.1
        panel.factor_ic_daily=ic.copy()
        with patch.dict(model.RECIPE,residual_training_window=16):
            gate,fit=model.fit_feature_gate(panel,np.arange(16))
            panel.factor_ic_daily[16:]=100
            other,repeat=model.fit_feature_gate(panel,np.arange(16))
        np.testing.assert_array_equal(gate,[1,1,0,0,1])
        np.testing.assert_array_equal(gate,other)
        self.assertEqual(fit['mask_sha256'],repeat['mask_sha256'])
        self.assertEqual(fit['train_end'],'15')

    def test_cache_matches_direct_and_future_labels_do_not_change_training_selection(self):
        panel=panel_fixture();before=panel.X.copy()
        expected=np.asarray([model.daily_factor_ic(panel,d) for d in range(16)])
        cache=model.build_factor_ic_cache(panel,np.arange(16),workers=2)
        np.testing.assert_array_equal(cache[:16],expected)
        self.assertFalse(cache.flags.writeable)
        with patch.dict(model.RECIPE,residual_training_window=16):
            gate,fit=model.fit_feature_gate(panel,np.arange(16))
            del panel.factor_ic_daily
            panel.Y['label_ret_1d'][16:]=100
            other,repeat=model.fit_feature_gate(panel,np.arange(16))
        np.testing.assert_array_equal(gate,other)
        self.assertEqual(fit['mask_sha256'],repeat['mask_sha256'])
        np.testing.assert_array_equal(panel.X,before)

    def test_no_stable_factors_fails_without_validation_fallback(self):
        panel=SimpleNamespace(days=np.asarray([str(i) for i in range(16)]),features=['constant'],factor_ic_daily=np.zeros((16,1)))
        with self.assertRaisesRegex(RuntimeError,'方向终止'):model.fit_feature_gate(panel,np.arange(16))

    def test_identity_gate_preserves_parent_rng_weights_and_auxiliary_then_masks_only_relative(self):
        configuration=dict(technical_indices=[0],residual_hidden=[8,4])
        with patch.dict(model.RECIPE,**configuration),patch.dict(parent.RECIPE,**configuration):
            torch.manual_seed(123);expected=parent.PredictModel(5,2).eval();rng=torch.get_rng_state()
            torch.manual_seed(123);actual=model.PredictModel(5,2).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for name,value in expected.state_dict().items():torch.testing.assert_close(actual.state_dict()[name],value,rtol=0,atol=0)
            with torch.no_grad():
                expected.residual[-1].weight.normal_(std=.05)
                expected.absolute_coef.normal_(std=.02);expected.linear_coef.normal_(std=.02)
            actual.load_state_dict(dict(expected.state_dict(),feature_gate=torch.ones(5)),strict=True)
            x=torch.randn(12,1,5);market=torch.randn(12,2)
            original=actual(x,market).detach();torch.testing.assert_close(original,expected(x,market),rtol=0,atol=0)
            teacher=actual.teacher_score(x).detach();absolute=actual.absolute_forward(x[:,-1],market).detach()
            actual.feature_gate[1]=0
            self.assertGreater(float((actual(x,market)[:,0]-original[:,0]).detach().abs().sum()),0.)
            torch.testing.assert_close(actual.absolute_forward(x[:,-1],market),absolute,rtol=0,atol=0)
            torch.testing.assert_close(actual.teacher_score(x),teacher,rtol=0,atol=0)
            restored=model.PredictModel(5,2).eval();restored.load_state_dict(actual.state_dict(),strict=True)
            torch.testing.assert_close(restored(x,market),actual(x,market),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
