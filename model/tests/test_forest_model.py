"""A fixed forest is one training round with64 trees, and validation never sizes it."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('forest_model_under_test',ROOT/'experiments/V280/model.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


class ForestModelTests(unittest.TestCase):
    def test_full_forest_reload_preserves_all64_trees_in_one_round(self):
        rng=np.random.default_rng(23)
        x=rng.normal(size=(1024,7)).astype(np.float32)
        parameters=dict(model.RECIPE['tree_parameters'],device='cpu',nthread=1,
            seed=model.RECIPE['seed'],min_child_weight=1)
        matrix=xgb.QuantileDMatrix(x,label=x[:,0]-.2*x[:,1],max_bin=128,nthread=1)
        fitted=model.native_global_round_train(parameters,matrix,1)
        self.assertEqual(fitted.num_boosted_rounds(),1)
        self.assertEqual(len(fitted.get_dump()),64)
        with patch.dict(model.RECIPE,technical_indices=[0]):
            net=model.PredictModel(5,2);net.set_booster(fitted.save_raw(raw_format='ubj'))
            restored=model.PredictModel(5,2);restored.load_state_dict(net.state_dict(),strict=True)
            stock=model.torch.from_numpy(x[:,:5,None].transpose(0,2,1).copy())
            market=model.torch.from_numpy(x[:,5:].copy())
            actual=restored(stock,market).detach().numpy()[:,0]
        np.testing.assert_array_equal(actual,fitted.inplace_predict(x))

    def test_validation_labels_do_not_change_fixed_forest_weights(self):
        rng=np.random.default_rng(32)
        x=rng.normal(size=(40,24,5)).astype(np.float32)
        y=.01*(x[:,:,0]-.3*x[:,:,1])
        panel=SimpleNamespace(X=x,M=np.ones((40,2),np.float32),
            Y={'label_ret_1d':y.copy(),'label_ret_5d':y*2},features=list('abcde'),market_dim=2,
            days=np.asarray([str(i) for i in range(40)]),codes=np.arange(24),amount=np.full((40,24),1e8))
        panel.mask=lambda day:np.ones(24,bool)
        def auxiliary(*args):
            net=model.PredictModel(5,2);net.ridge=None
            return net,{},dict(sha256='fixed auxiliary',absolute_fit={},relative_teacher_used=False)
        parameters=dict(model.RECIPE['tree_parameters'],device='cpu',nthread=1,min_child_weight=1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            first,second=[root/f'model_train/2025Q3/fold{fold}' for fold in [1,2]]
            first.mkdir(parents=True);second.mkdir(parents=True)
            with patch.object(model,'UNIT_ROOT',root),patch.object(model,'frozen_teacher',side_effect=auxiliary), \
                 patch.dict(model.RECIPE,technical_indices=[0],tree_parameters=parameters):
                expected,_,fit=model.fit_tree_stage(panel,'2025Q3',1,np.arange(24),np.arange(24,37),model.torch.device('cpu'),first,'selector')
                panel.Y['label_ret_1d'][24:]*=-5
                panel.Y['label_ret_5d'][24:]*=-5
                actual,_,other=model.fit_tree_stage(panel,'2025Q3',2,np.arange(24),np.arange(24,37),model.torch.device('cpu'),second,'selector')
            np.testing.assert_array_equal(expected.booster_blob.numpy(),actual.booster_blob.numpy())
            self.assertEqual(fit['physical_tree_count'],64)
            self.assertEqual(other['selected_trees'],1)
            self.assertFalse(other['epoch_selection'])


if __name__=='__main__':unittest.main()
