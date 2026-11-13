"""Market allocation must ignore future labels and charge directional switching costs."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/_archive/20261024_20261025/scripts'))
spec=importlib.util.spec_from_file_location('market_distribution_test',ROOT/'experiments/_archive/20261024_20261025/scripts/large_market_distribution_20261025.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


class MarketDistributionTests(unittest.TestCase):
    def test_future_labels_and_market_rows_do_not_change_earlier_fit(self):
        rng=np.random.default_rng(82)
        returns=np.resize(np.asarray([-.06,.002,.07]),300)
        panel=SimpleNamespace(Y={'label_ret_5d':np.tile(returns[:,None],(1,4))},
            M=rng.normal(size=(300,4)),days=np.asarray([str(i) for i in range(300)]),market_features=list('abcd'))
        expected=study.fit_distribution(panel,np.arange(220))
        panel.Y['label_ret_5d'][220:]=100
        panel.M[220:]=-100
        actual=study.fit_distribution(panel,np.arange(220))
        self.assertEqual(expected,actual)
        self.assertLessEqual(actual['probability_reload_error'],1e-12)
        np.testing.assert_array_equal(study.market_classes([-.04,-.039,0,.039,.04]),[0,1,1,1,2])

    def test_transition_cost_can_prevent_small_gain_and_does_not_impose_recovery_wait(self):
        p=np.tile([0.,1.,0.],(3,1))
        rewards=np.asarray([[0,.00008,.0002],[0,-.004,-.01],[0,.004,.01]])
        u=np.repeat(rewards[:,None,:],3,axis=1)
        budget,diagnostic=study.utility_budgets(p,u)
        np.testing.assert_array_equal(budget,[0,0,1])
        self.assertAlmostEqual(diagnostic[0]['transition_cost'][2],study.BUY_COST)
        full,_=study.utility_budgets(p[:1],np.zeros((1,3,3)),initial=1.)
        np.testing.assert_array_equal(full,[1.])
        changed=u.copy();changed[2]=100
        other,_=study.utility_budgets(p,changed)
        np.testing.assert_array_equal(budget[:2],other[:2])

    def test_train_only_single_and_two_class_fallbacks_are_probabilities(self):
        rng=np.random.default_rng(23)
        panel=SimpleNamespace(Y={'label_ret_5d':np.full((160,4),.01)},M=rng.normal(size=(160,3)),
            days=np.asarray([str(i) for i in range(160)]),market_features=list('abc'))
        fit=study.fit_distribution(panel,np.arange(150))
        np.testing.assert_array_equal(study.predict_distribution(fit,panel.M[:2]),[[0,1,0],[0,1,0]])
        panel.Y['label_ret_5d'][::3]=-.08
        fit=study.fit_distribution(panel,np.arange(150))
        probability=study.predict_distribution(fit,panel.M)
        np.testing.assert_allclose(probability.sum(axis=1),1,atol=1e-12)
        np.testing.assert_array_equal(probability[:,2],0)
        diagnostic=study.forecast_diagnostics(probability,np.r_[np.full(150,.01),np.full(10,np.nan)])
        self.assertEqual(diagnostic['mature_days'],150)


if __name__=='__main__':unittest.main()
