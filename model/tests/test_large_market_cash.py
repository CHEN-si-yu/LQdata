"""Risk requests are targets; actual unsold shares and later restoration must persist."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from test_large_cash import fixture,fees,study as baseline

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('market_cash_test',ROOT/'experiments/_archive/20261024_20261025/scripts/large_market_cash_20261025.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


class LargeMarketCashTests(unittest.TestCase):
    def test_unit_budget_reproduces_original_fills(self):
        p,x,d,s=fixture()
        a,ac,at=baseline.simulate(s,p,x,d,fees,money=100000,n=2)
        b,bc,bt=study.simulate_risk(s,p,x,d,fees,money=100000,n=2)
        self.assertEqual(at,bt)
        for key in a:self.assertEqual(a[key],b[key])
        for left,right in zip(ac,bc):
            for key in left:self.assertEqual(left[key],right[key])

    def test_zero_budget_clears_over_multiple_opens_then_can_restore(self):
        p,x,d,s=fixture();budgets=np.asarray([1.,1.,0.,0.,1.])
        stat,curve,trades=study.simulate_risk(s,p,x,d,fees,money=100000,n=2,budgets=budgets,liquidate=False)
        self.assertGreater(curve[2]['positions'],0)
        self.assertEqual(curve[3]['positions'],0)
        self.assertGreater(curve[4]['positions'],0)
        self.assertEqual(stat['cash_requested_days'],2)
        self.assertEqual(stat['cash_requested_days_with_positions'],1)
        self.assertTrue(all(t['side']=='sell' for t in trades if t['date'] in p.days[[3,4]]))
        json.dumps(dict(metrics=stat,curve=curve,trades=trades),allow_nan=False)
        future=copy.deepcopy(x);future.open[4:]=10.5;future.close[4:]=10.5
        changed=budgets.copy();changed[3:]=.4
        _,_,other=study.simulate_risk(s,p,future,d,fees,money=100000,n=2,budgets=changed,liquidate=False)
        cutoff=p.days[3]
        self.assertEqual([t for t in trades if t['date']<=cutoff],[t for t in other if t['date']<=cutoff])


if __name__=='__main__':unittest.main()
