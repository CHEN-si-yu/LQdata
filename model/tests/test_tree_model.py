"""树字节注册buffer的独立重载与分块原生断点。"""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('test_tree_model_module',ROOT/'experiments/V265/model.py')
model=importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class TreeModelTests(unittest.TestCase):
    def test_embedded_tree_bytes_reload_without_external_model_files(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(64,5)).astype(np.float32)
        booster=xgb.train(dict(objective='reg:squarederror',max_depth=2,nthread=1),
            xgb.DMatrix(x,label=x[:,0]-x[:,1]),num_boost_round=4,verbose_eval=False)
        with patch.dict(model.RECIPE,technical_indices=[0]):
            fitted=model.PredictModel(3,2)
            fitted.set_booster(booster.save_raw(raw_format='ubj'))
            inputs=model.torch.from_numpy(x[:,:3,None].transpose(0,2,1).copy())
            market=model.torch.from_numpy(x[:,3:].copy())
            expected=fitted(inputs,market).detach().numpy()
            reloaded=model.PredictModel(3,2)
            self.assertEqual(reloaded.booster_blob.numel(),0)
            reloaded.load_state_dict(fitted.state_dict(),strict=True)
            actual=reloaded(inputs,market).detach().numpy()
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(actual[:,0],booster.inplace_predict(x))

    def test_native_block_resume_preserves_selected_prefix(self):
        self._check_native_block_resume('reg:squarederror')

    def test_pseudohuber_native_block_resume_preserves_selected_prefix(self):
        self._check_native_block_resume('reg:pseudohubererror')

    def _check_native_block_resume(self,objective):
        rng=np.random.default_rng(52)
        x=rng.normal(size=(32,24,3)).astype(np.float32)
        y=.01*(x[:,:,0]+.3*x[:,:,1])
        panel=SimpleNamespace(X=x,M=rng.normal(size=(32,2)).astype(np.float32),
            Y={'label_ret_1d':y,'label_ret_5d':y*2},features=['a','b','c'],market_dim=2,
            days=np.asarray([str(i) for i in range(32)]),codes=np.arange(24),amount=np.full((32,24),1e8))
        panel.mask=lambda d:np.ones(24,bool)
        def auxiliary(*args):
            net=model.PredictModel(3,2)
            net.ridge=None
            return net,{},dict(sha256='fixed auxiliary',absolute_fit={})
        parameters=dict(model.RECIPE['tree_parameters'],objective=objective,max_depth=2,max_bin=16,min_child_weight=1)
        if objective=='reg:pseudohubererror':
            parameters['huber_slope']=1.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            full,broken=root/'model_train/2025Q3/fold1',root/'model_train/2025Q3/fold2'
            full.mkdir(parents=True);broken.mkdir(parents=True)
            with patch.object(model,'UNIT_ROOT',root),patch.object(model,'frozen_teacher',side_effect=auxiliary), \
                 patch.dict(model.RECIPE,technical_indices=[0],tree_parameters=parameters,max_epochs=8,
                    early_stop_patience=20,tree_block_rounds=4):
                expected,_,expected_fit=model.fit_tree_stage(panel,'2025Q3',1,np.arange(20),np.arange(20,29),
                    model.torch.device('cpu'),full,'selector')
                save=model.save_torch
                def interrupt(path,value):
                    save(path,value)
                    if Path(path).name=='last.pt':raise RuntimeError('checkpoint persisted')
                with patch.object(model,'save_torch',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'checkpoint persisted'):
                        model.fit_tree_stage(panel,'2025Q3',1,np.arange(20),np.arange(20,29),
                            model.torch.device('cpu'),broken,'selector')
                actual,_,actual_fit=model.fit_tree_stage(panel,'2025Q3',1,np.arange(20),np.arange(20,29),
                    model.torch.device('cpu'),broken,'selector')
            self.assertEqual(expected_fit['selected_trees'],actual_fit['selected_trees'])
            np.testing.assert_array_equal(expected.booster_blob.numpy(),actual.booster_blob.numpy())


if __name__=='__main__':unittest.main()
