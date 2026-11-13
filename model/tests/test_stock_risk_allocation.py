"""Cross-stock volatility changes only target dollars; all fills retain the L1 contract."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/_archive/20261024_20261025/scripts'))
spec=importlib.util.spec_from_file_location('stock_risk_test',ROOT/'experiments/_archive/20261024_20261025/scripts/large_stockrisk_20261025.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


def fixture(amount=2e9):
    rng=np.random.default_rng(42);t,n=130,6
    close=10*np.exp(np.cumsum(rng.normal(size=(t,n))*np.linspace(.004,.035,n),axis=0))
    panel=SimpleNamespace(days=pd.bdate_range('2025-01-01',periods=t).strftime('%Y-%m-%d').to_numpy(dtype=str),
        codes=np.asarray([f'{c:06}.SZ' for c in range(n)]),amount=np.full((t,n),amount))
    px=SimpleNamespace(open=close.copy(),close=close.copy(),adj=np.ones((t,n)),
        limit_up=close*1.1,limit_down=close*.9,entry=np.ones((t,n),bool),exit=np.ones((t,n),bool),
        raw={'close':close.copy(),'adj_factor':np.ones((t,n))})
    days=np.arange(80,86);pred=np.tile(np.arange(n,0,-1),(len(days),1)).astype(float)
    pred[2:]=pred[2:,::-1]
    return panel,px,days,pred


def fees(gross,sell=False):
    return max(gross*.00025,5)+gross*(.00001+(.0005 if sell else 0))


class StockRiskAllocationTests(unittest.TestCase):
    def test_inverse_risk_order_budget_and_cap_with_redistribution(self):
        risk=np.r_[.000001,np.linspace(.01,.08,499)]
        weights,info=study.stock_risk_weights(risk,500)
        self.assertAlmostEqual(float(weights.sum()),1.,places=12)
        self.assertAlmostEqual(float(weights.max()),.004,places=12)
        self.assertTrue((np.diff(weights)<=1e-15).all())
        self.assertLessEqual(info['maximum_target_weight'],.004+1e-12)
        self.assertGreater(info['target_effective_names'],250)
        uniform,_=study.stock_risk_weights(np.ones(500)*.02,500)
        np.testing.assert_allclose(uniform,np.full(500,.002),rtol=0,atol=1e-15)

    def test_missing_and_zero_risk_use_pool_median_or_uniform_and_partial_pool_keeps_cash(self):
        weights,info=study.stock_risk_weights([.01,.02,.03,np.nan,0],5)
        self.assertAlmostEqual(weights[1],weights[3]);self.assertAlmostEqual(weights[1],weights[4])
        self.assertAlmostEqual(info['missing_fraction'],.4)
        weights,_=study.stock_risk_weights([np.nan,0],5)
        np.testing.assert_allclose(weights,[.2,.2],rtol=0,atol=1e-15)
        self.assertEqual(study.stock_risk_weights([],5)[1]['requested_budget'],0.)
        with self.assertRaises(ValueError):study.stock_risk_weights([.1,.2],1)

    def test_volatility_adjustment_split_invariance_and_no_close_forwardfill(self):
        _,px,_,_=fixture()
        expected=study.causal_stock_volatility(px.raw['close'],px.raw['adj_factor'])
        close=px.raw['close'].copy();adj=px.raw['adj_factor'].copy()
        close[60:,0]*=.5;adj[60:,0]*=2
        actual=study.causal_stock_volatility(close,adj)
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-14,equal_nan=True)
        close[15:80,1]=np.nan
        actual=study.causal_stock_volatility(close,adj)
        self.assertTrue(np.isnan(actual[70,1]));self.assertTrue(np.isnan(actual[110,1]))
        self.assertTrue(np.isfinite(actual[125,1]))

    def test_neutral_target_mode_exactly_reproduces_original_ledger(self):
        panel,px,days,pred=fixture()
        expected=study.cash.simulate(pred,panel,px,days,fees,money=100000,n=3)
        actual=study.simulate_stock_risk(pred,panel,px,days,fees,money=100000,n=3)
        self.assertEqual(actual,expected)

    def test_future_shocks_do_not_change_past_risk_targets_or_orders(self):
        panel,px,days,pred=fixture();changed=copy.deepcopy(px)
        changed.raw['close'][84:]*=2;changed.open[84:]*=2;changed.close[84:]*=2
        changed.limit_up[84:]*=2;changed.limit_down[84:]*=2
        original=study.causal_stock_volatility(px.raw['close'],px.raw['adj_factor'])
        altered=study.causal_stock_volatility(changed.raw['close'],changed.raw['adj_factor'])
        np.testing.assert_array_equal(original[:84],altered[:84])
        _,_,left=study.simulate_stock_risk(pred,panel,px,days,fees,money=100000,n=3,
            allocation_mode='invvol63_cap2x',volatility=original)
        _,_,right=study.simulate_stock_risk(pred,panel,changed,days,fees,money=100000,n=3,
            allocation_mode='invvol63_cap2x',volatility=altered)
        self.assertEqual([t for t in left if t['date']<panel.days[84]],[t for t in right if t['date']<panel.days[84]])

    def test_risk_allocation_preserves_real_capacity_tplus1_and_residual_holdings(self):
        panel,px,days,pred=fixture(2e5)
        volatility=study.causal_stock_volatility(px.raw['close'],px.raw['adj_factor'])
        metrics,curve,trades=study.simulate_stock_risk(pred,panel,px,days,fees,money=100000,n=3,
            allocation_mode='invvol63_cap2x',volatility=volatility)
        self.assertLessEqual(metrics['max_positions_observed'],3)
        self.assertGreater(metrics['ending_positions'],0)
        self.assertFalse(metrics['fully_liquidated'])
        self.assertTrue(all(row['cash']>=-1e-6 for row in curve))
        for trade in trades:
            self.assertEqual(trade['time'],'09:30')
            self.assertLessEqual(trade['used_capacity_after'],trade['prior_amount_capacity']+1e-6)
            if trade['side']=='sell':self.assertLessEqual(trade['quantity'],trade['sellable_before']+1e-6)
        self.assertAlmostEqual(metrics['return_value'],metrics['gross_return_same_positions']-metrics['fee_drag']-metrics['slippage_drag'],places=12)


if __name__=='__main__':
    unittest.main()
