"""DART must retain historical normalized weights and global iteration RNG state."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dart_model_under_test',ROOT/'experiments/V279/model.py')
model=importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class DartModelTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(84)
        self.x=rng.normal(size=(1024,5)).astype(np.float32)
        self.y=(self.x[:,0]-.3*self.x[:,1]+.1*rng.normal(size=1024)).astype(np.float32)
        self.params=dict(model.RECIPE['tree_parameters'],device='cpu',nthread=1,
            seed=model.RECIPE['seed'],max_depth=2,max_bin=16,min_child_weight=1)

    def matrix(self):
        return xgb.QuantileDMatrix(self.x,label=self.y,max_bin=16,nthread=1)

    def test_global_rounds_match_native_and_serialized_resume(self):
        matrix=self.matrix()
        expected=xgb.train(self.params,matrix,num_boost_round=48,verbose_eval=False)
        full=model.native_global_round_train(self.params,matrix,48)
        partial=model.native_global_round_train(self.params,matrix,16)
        restored=xgb.Booster(self.params,cache=[matrix],model_file=partial.save_raw(raw_format='ubj'))
        actual=model.native_global_round_train(self.params,matrix,32,xgb_model=restored)
        self.assertEqual(bytes(full.save_raw(raw_format='ubj')),bytes(expected.save_raw(raw_format='ubj')))
        self.assertEqual(bytes(actual.save_raw(raw_format='ubj')),bytes(expected.save_raw(raw_format='ubj')))

    def test_cached_prefixes_are_demand_driven_and_keep_earlier_normalization(self):
        calls=[]
        def factory():
            calls.append(1)
            return self.matrix(),dict(fit_rows=len(self.y),fit_dates=1,input_width=5)
        with tempfile.TemporaryDirectory() as directory:
            first,info=model.cached_native_prefix(self.params,factory,12,48,{'input':'fixed'},directory)
            self.assertEqual(info['shared_compute_trees'],12)
            later,_=model.cached_native_prefix(self.params,factory,32,48,{'input':'fixed'},directory)
            old,info=model.cached_native_prefix(self.params,factory,12,48,{'input':'fixed'},directory)
            self.assertEqual(first,old)
            self.assertTrue(info['relative_fit_reused'])
            self.assertEqual(len(calls),2)
            expected=model.native_global_round_train(self.params,self.matrix(),12)
            self.assertEqual(first,bytes(expected.save_raw(raw_format='ubj')))
            final=xgb.Booster(model_file=bytearray(later))
            prefix=xgb.Booster(model_file=bytearray(first))
            self.assertGreater(np.max(np.abs(final.predict(self.matrix(),iteration_range=(0,12))-prefix.predict(self.matrix()))),1e-5)

    def test_interrupted_prefix_cache_replays_exactly_and_detects_corruption(self):
        def factory():return self.matrix(),dict(fit_rows=len(self.y),fit_dates=1,input_width=5)
        def interrupt(count):
            if count==20:raise RuntimeError('persisted checkpoint')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'persisted checkpoint'):
                model.cached_native_prefix(self.params,factory,32,48,{'input':'fixed'},directory,interrupt)
            actual,info=model.cached_native_prefix(self.params,factory,32,48,{'input':'fixed'},directory)
            expected=model.native_global_round_train(self.params,self.matrix(),32)
            self.assertEqual(actual,bytes(expected.save_raw(raw_format='ubj')))
            path=Path(directory)/info['cache_key']/'prefix_0032.ubj'
            path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'bytes changed'):
                model.cached_native_prefix(self.params,factory,32,48,{'input':'fixed'},directory)

    def test_selector_resume_keeps_full_best_snapshot(self):
        x=self.x[:960].reshape(40,24,5)
        y=.01*(x[:,:,0]-.3*x[:,:,1])
        panel=SimpleNamespace(X=x,M=np.ones((40,2),np.float32),
            Y={'label_ret_1d':y,'label_ret_5d':y*2},features=list('abcde'),market_dim=2,
            days=np.asarray([str(i) for i in range(40)]),codes=np.arange(24),amount=np.full((40,24),1e8))
        panel.mask=lambda day:np.ones(24,bool)
        def auxiliary(*args):
            net=model.PredictModel(5,2);net.ridge=None
            return net,{},dict(sha256='fixed auxiliary',absolute_fit={},relative_teacher_used=False)
        with tempfile.TemporaryDirectory() as directory:
            full,broken=Path(directory)/'model_train/2025Q3/fold1',Path(directory)/'model_train/2025Q3/fold2'
            full.mkdir(parents=True);broken.mkdir(parents=True)
            with patch.object(model,'UNIT_ROOT',Path(directory)),patch.object(model,'frozen_teacher',side_effect=auxiliary),patch.dict(model.RECIPE,
                    technical_indices=[0],tree_parameters=self.params,max_epochs=24,
                    early_stop_patience=30,tree_block_rounds=8):
                expected,_,expected_fit=model.fit_tree_stage(panel,'2025Q3',1,np.arange(24),np.arange(24,37),
                    model.torch.device('cpu'),full,'selector')
                save=model.save_torch
                def interrupt(path,value):
                    save(path,value)
                    if Path(path).name=='last.pt':raise RuntimeError('persisted selector')
                with patch.object(model,'save_torch',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'persisted selector'):
                        model.fit_tree_stage(panel,'2025Q3',1,np.arange(24),np.arange(24,37),
                            model.torch.device('cpu'),broken,'selector')
                actual,_,actual_fit=model.fit_tree_stage(panel,'2025Q3',1,np.arange(24),np.arange(24,37),
                    model.torch.device('cpu'),broken,'selector')
            self.assertEqual(expected_fit['selected_trees'],actual_fit['selected_trees'])
            np.testing.assert_array_equal(expected.booster_blob.numpy(),actual.booster_blob.numpy())

    def test_delivery_cache_matches_independent_fit_and_retains_fold_auxiliary(self):
        x=self.x[:960].reshape(40,24,5)
        y=.01*(x[:,:,0]-.3*x[:,:,1])
        panel=SimpleNamespace(X=x,M=np.ones((40,2),np.float32),
            Y={'label_ret_1d':y,'label_ret_5d':y*2},features=list('abcde'),market_dim=2,
            days=np.asarray([str(i) for i in range(40)]),codes=np.arange(24))
        panel.mask=lambda day:np.ones(24,bool)
        def auxiliary(panel,quarter,fold,*args):
            net=model.PredictModel(5,2);net.ridge=None
            net.absolute_coef.fill_(fold)
            return net,{},dict(sha256=f'fold{fold}',absolute_fit={},relative_teacher_used=False)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            folders=[root/f'model_train/2025Q3/fold{fold}' for fold in [1,2,3]]
            for folder in folders:folder.mkdir(parents=True)
            with patch.object(model,'UNIT_ROOT',root),patch.object(model,'frozen_teacher',side_effect=auxiliary), \
                 patch.dict(model.os.environ,MX_TREE_PREFIX_CACHE_DIR=str(root/'cache')), \
                 patch.dict(model.RECIPE,technical_indices=[0],tree_parameters=self.params,max_epochs=24,tree_block_rounds=8):
                expected,_,_=model._UncachedTreeFit(panel,'2025Q3',3,np.arange(24),[],model.torch.device('cpu'),folders[2],'delivery_refit',16)
                first,_,_=model.fit_tree_stage(panel,'2025Q3',1,np.arange(24),[],model.torch.device('cpu'),folders[0],'delivery_refit',8)
                second,_,_=model.fit_tree_stage(panel,'2025Q3',2,np.arange(24),[],model.torch.device('cpu'),folders[1],'delivery_refit',16)
                resumed,_,fit=model.fit_tree_stage(panel,'2025Q3',2,np.arange(24),[],model.torch.device('cpu'),folders[1],'delivery_refit',16)
            np.testing.assert_array_equal(second.booster_blob.numpy(),expected.booster_blob.numpy())
            np.testing.assert_array_equal(resumed.booster_blob.numpy(),second.booster_blob.numpy())
            self.assertTrue(fit['resumed_fold_checkpoint'])
            self.assertTrue((first.absolute_coef==1).all())
            self.assertTrue((second.absolute_coef==2).all())


if __name__=='__main__':unittest.main()
