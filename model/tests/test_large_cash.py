"""大资金部分成交必须真实留仓，不能用目标持仓覆盖账本。"""
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
spec=importlib.util.spec_from_file_location('large_cash_test',ROOT/'experiments/_archive/20261024_20261025/scripts/large_cash_20261025.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


def fixture(amount=2e5):
    t,n=9,6
    panel=SimpleNamespace(days=pd.date_range('2025-07-01',periods=t).strftime('%Y-%m-%d').to_numpy(dtype=str),
        codes=np.asarray([f'{c:06}.SZ' for c in range(n)]),amount=np.full((t,n),amount))
    px=SimpleNamespace(open=np.full((t,n),10.),close=np.full((t,n),10.),adj=np.ones((t,n)),
        limit_up=np.full((t,n),11.),limit_down=np.full((t,n),9.),
        entry=np.ones((t,n),bool),exit=np.ones((t,n),bool))
    days=np.arange(5)
    pred=np.tile(np.arange(n,0,-1),(5,1)).astype(float)
    return panel,px,days,pred


def fees(gross,sell=False):
    return max(gross*.00025,5)+gross*(.00001+(.0005 if sell else 0))


class LargeCashTests(unittest.TestCase):
    def test_partial_liquidation_keeps_real_shares_and_reconciles_flat_market(self):
        panel,px,days,pred=fixture()
        stat,curve,trades=study.simulate(pred,panel,px,days,fees,money=100000,n=2)
        self.assertEqual(stat['ending_positions'],2)
        self.assertFalse(stat['fully_liquidated'])
        self.assertGreater(stat['ending_position_value'],0)
        self.assertGreater(stat['order_constraints']['capacity_sell'],0)
        self.assertAlmostEqual(stat['gross_return_same_positions'],0.,places=12)
        self.assertAlmostEqual(stat['return_value'],-stat['fee_drag']-stat['slippage_drag'])
        self.assertLessEqual(stat['max_positions_observed'],2)
        self.assertTrue(all(r['cash']>=0 for r in curve))
        for tr in trades:
            self.assertEqual(tr['time'],'09:30')
            self.assertLessEqual(tr['used_capacity_after'],tr['prior_amount_capacity']+1e-6)
            if tr['side']=='buy':
                self.assertAlmostEqual(tr['quantity']%100,0)
            else:
                self.assertLessEqual(tr['quantity'],tr['sellable_before'])
        self.assertFalse(any(tr['side']=='sell' and tr['date']==panel.days[1] for tr in trades))

    def test_blocked_old_position_occupies_slot_until_it_really_exits(self):
        panel,px,days,pred=fixture(1e10)
        pred[1:]=[5,4,6,3,2,1]
        px.exit[2,1]=False
        stat,curve,trades=study.simulate(pred,panel,px,days,fees,money=100000,n=2)
        self.assertGreater(stat['order_constraints']['blocked_sell'],0)
        self.assertGreater(stat['order_constraints']['slots_buy'],0)
        self.assertIn(str(panel.codes[1]),curve[1]['holdings'])
        self.assertFalse(any(t['date']==panel.days[2] and t['code']==panel.codes[2] and t['side']=='buy' for t in trades))

    def test_fixed_retention_reduces_rank_boundary_churn(self):
        panel,px,days,pred=fixture(1e10)
        pred[1]=[5,4,6,3,2,1]
        pred[3]=[5,4,6,3,2,1]
        a,_,_=study.simulate(pred,panel,px,days,fees,money=100000,n=2,liquidate=False)
        b,_,_=study.simulate(pred,panel,px,days,fees,money=100000,n=2,retention_rank=3,liquidate=False)
        self.assertLess(b['trades'],a['trades'])
        self.assertGreater(b['return_value'],a['return_value'])

    def test_future_prices_and_scores_do_not_change_earlier_orders(self):
        panel,px,days,pred=fixture()
        changed=copy.deepcopy(px);changed.open[4:]=10.5;changed.close[4:]=10.5
        p=pred.copy();p[3:]=p[3:,::-1]
        _,_,a=study.simulate(pred,panel,px,days,fees,money=100000,n=2)
        _,_,b=study.simulate(p,panel,changed,days,fees,money=100000,n=2)
        cutoff=panel.days[3]
        self.assertEqual([t for t in a if t['date']<=cutoff],[t for t in b if t['date']<=cutoff])

    def test_economic_share_adjustment_does_not_create_split_profit(self):
        panel,px,days,pred=fixture(1e10)
        px.adj[3:]=2.
        px.open[3:]=px.close[3:]=5.
        px.limit_up[3:]=5.5
        px.limit_down[3:]=4.5
        stat,_,_=study.simulate(pred,panel,px,days,fees,money=100000,n=2,liquidate=False)
        self.assertAlmostEqual(stat['gross_return_same_positions'],0.,places=12)


if __name__=='__main__':
    unittest.main()
