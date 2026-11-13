"""Input-only cross-section contract; raw mean labels and absolute head stay unchanged."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name+'_copula_test',ROOT/'experiments'/name/'model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class CopulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=load('V275');cls.parent=load('V260')

    def tearDown(self):
        self.m._COPULA_INPUT_LOOKUP.clear()

    def test_midrank_monotonicity_constants_and_permutation(self):
        x=np.asarray([[-2,1],[-1,1],[0,1],[0,1],[3,1]],np.float32)
        direct=self.m.normal_score_features_direct(x)
        np.testing.assert_array_equal(direct,self.m.normal_score_features_direct(np.exp(x)))
        np.testing.assert_array_equal(direct[:,1],0.)
        self.assertEqual(direct[2,0],direct[3,0])
        perm=np.asarray([3,0,4,1,2])
        np.testing.assert_array_equal(self.m.normal_score_features_direct(x[perm]),direct[perm])
        self.assertTrue(np.isfinite(direct).all())

    def test_loss_subset_is_taken_after_feature_only_transform(self):
        x=np.asarray([[[-3,1],[-1,2],[0,3],[1,4],[3,5]],[[8,9]]*5],np.float32)
        panel=SimpleNamespace(X=x,Y={'label':np.zeros((2,5))},mask=lambda day:np.ones(5,bool))
        mask=np.asarray([True,False,True,False,True])
        expected=self.m.normal_score_features_direct(x[0])[mask]
        actual=self.m.copula_day(panel,0,mask)
        np.testing.assert_array_equal(expected,actual)
        panel.Y['label'][:]=np.nan;panel.X[1]=1e5
        np.testing.assert_array_equal(actual,self.m.copula_day(panel,0,mask))
        # Transforming only labeled rows would give a different empirical CDF.
        self.assertFalse(np.array_equal(actual,self.m.normal_score_features_direct(x[0,mask])))

    def test_readonly_cache_and_direct_paths_match(self):
        rng=np.random.default_rng(275);raw=np.clip(rng.normal(size=(25,6)).astype(np.float32),-5,5)
        matrix=np.zeros((31,6),np.float32);indices=np.arange(25)+3
        expected=self.m.normal_score_features_direct(raw);matrix[indices]=expected;matrix.setflags(write=False)
        self.m._COPULA_INPUT_LOOKUP[self.m.copula_input_key(raw)]=(matrix,indices)
        np.testing.assert_array_equal(self.m.normal_score_features(raw),expected)
        self.m._COPULA_INPUT_LOOKUP.clear()
        np.testing.assert_array_equal(self.m.normal_score_features(raw),expected)

    def test_raw_absolute_head_matches_parent_and_reloads(self):
        m=self.m;torch=m.torch
        torch.set_num_threads(1)
        parent=self.parent.PredictModel(len(m.RECIPE['feature_names']),3)
        net=m.PredictModel(len(m.RECIPE['feature_names']),3)
        rng=np.random.default_rng(275)
        with torch.no_grad():
            parent.absolute_coef.copy_(torch.from_numpy(rng.normal(size=parent.absolute_coef.shape).astype(np.float32)))
            parent.linear_coef.copy_(torch.from_numpy(rng.normal(size=parent.linear_coef.shape).astype(np.float32)))
        net.load_state_dict(parent.state_dict())
        x=torch.from_numpy(rng.normal(size=(40,1,len(m.RECIPE['feature_names']))).astype(np.float32))
        market=torch.from_numpy(rng.normal(size=(40,3)).astype(np.float32))
        a=parent(x,market).detach().numpy();b=net(x,market).detach().numpy()
        np.testing.assert_array_equal(a[:,1],b[:,1]);self.assertFalse(np.allclose(a[:,0],b[:,0]))
        clone=m.PredictModel(len(m.RECIPE['feature_names']),3);clone.load_state_dict(net.state_dict())
        np.testing.assert_array_equal(b,clone(x,market).detach().numpy())


if __name__=='__main__':
    unittest.main()
