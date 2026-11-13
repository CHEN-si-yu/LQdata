"""Probability calibration uses past walk-forward labels and no account-specific fitting."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/_archive/20261024_20261025/scripts'))
spec=importlib.util.spec_from_file_location('market_calibration_test',ROOT/'experiments/_archive/20261024_20261025/scripts/large_market_calibration_20261025.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


class MarketProbabilityCalibrationTests(unittest.TestCase):
    def test_closed_form_weight_is_interior_brier_optimum(self):
        prior=np.tile([1/3]*3,(10,1));raw=np.tile([.8,.1,.1],(10,1))
        classes=np.asarray([0]*5+[1]*3+[2]*2)
        fit=study.brier_shrinkage(raw,prior,classes)
        self.assertGreater(fit['alpha'],0);self.assertLess(fit['alpha'],1)
        residual=study.probability_mixture(raw,prior,fit['alpha'])-np.eye(3)[classes]
        self.assertAlmostEqual(float(np.sum(residual*(raw-prior))),0.,places=12)
        self.assertLess(fit['fitted_brier'],fit['prior_brier'])
        self.assertLess(fit['fitted_brier'],fit['raw_brier'])

    def test_boundaries_identical_forecasts_and_simplex_guards(self):
        labels=np.asarray([0,1,2,1]);prior=np.tile([1/3]*3,(4,1));perfect=np.eye(3)[labels]
        self.assertEqual(study.brier_shrinkage(perfect,prior,labels)['alpha'],1.)
        self.assertEqual(study.brier_shrinkage(prior,prior,labels)['alpha'],0.)
        wrong=np.tile([.8,.1,.1],(4,1))
        self.assertEqual(study.brier_shrinkage(wrong,prior,np.ones(4,dtype=int))['alpha'],0.)
        np.testing.assert_array_equal(study.probability_mixture(perfect,prior,0.),prior)
        np.testing.assert_array_equal(study.probability_mixture(perfect,prior,1.),perfect)
        with self.assertRaises(ValueError):study.probability_mixture(perfect,prior,1.1)
        with self.assertRaises(ValueError):study.probability_mixture(perfect,prior*2,.5)

    def test_real_forward_fits_enforce_label_guard_and_exclude_future_data(self):
        rng=np.random.default_rng(572);n=440
        market=rng.normal(size=(n,2));returns=.055*np.tanh(market[:,0])+rng.normal(0,.012,n)
        panel=SimpleNamespace(days=np.asarray([f'day{i:03}' for i in range(n)]),M=market.copy(),
            market_features=['x','y'],Y={'label_ret_5d':returns[:,None]+rng.normal(0,.001,(n,12))})
        fit=study.fit_forward_calibrator(panel,np.arange(400),folds=3,fold_size=20)
        for fold in fit['folds']:
            self.assertLess(fold['train_last_index']+21,fold['validation_first_index'])
        self.assertEqual(fit['calibration_dates'][0],'day340')
        self.assertEqual(fit['calibration_dates'][-1],'day399')
        panel.M[400:]=1e6;panel.Y['label_ret_5d'][400:]=1e6
        again=study.fit_forward_calibrator(panel,np.arange(400),folds=3,fold_size=20)
        self.assertEqual(fit,again)
        panel.Y['label_ret_5d'][380:400]=.2
        later=study.fit_forward_calibrator(panel,np.arange(400),folds=3,fold_size=20)
        np.testing.assert_array_equal(fit['raw_probability'][:20],later['raw_probability'][:20])
        self.assertEqual(fit['folds'][0],later['folds'][0])

    def test_raw_endpoint_preserves_original_utility_budget_path(self):
        probability=np.asarray([[.1,.8,.1],[.8,.1,.1],[.1,.1,.8],[.1,.8,.1]])
        prior=np.tile([.2,.6,.2],(4,1))
        utilities=np.tile([[0,-.012,-.03],[0,.002,.003],[0,.016,.038]],(4,1,1))
        expected=study.distribution.utility_budgets(probability,utilities)
        actual=study.distribution.utility_budgets(study.probability_mixture(probability,prior,1.),utilities)
        np.testing.assert_array_equal(actual[0],expected[0]);self.assertEqual(actual[1],expected[1])


if __name__=='__main__':
    unittest.main()
