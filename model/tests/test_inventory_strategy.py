"""Desired inventory cap is distinct from flow capacity and never fakes liquidation."""
import copy
import json
import importlib.util
from pathlib import Path
import unittest

from test_large_cash import fixture,fees,study as baseline

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('inventory_test',ROOT/'experiments/_archive/20261024_20261025/scripts/large_inventory_20261025.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


class InventoryTests(unittest.TestCase):
    def test_same_target_reproduces_original_actual_orders(self):
        p,x,d,s=fixture()
        a,ac,at=baseline.simulate(s,p,x,d,fees,money=100000,n=2)
        b,bc,bt=study.simulate_inventory(s,p,x,d,fees,money=100000,n=2,target_mode='daily_top500')
        self.assertEqual(at,bt)
        for key in a:self.assertEqual(a[key],b[key])
        for left,right in zip(ac,bc):
            for key in left:self.assertEqual(left[key],right[key])

    def test_inventory_target_prevents_multiday_accumulation_in_flat_liquid_case(self):
        p,x,d,s=fixture()
        a,_,_=baseline.simulate(s,p,x,d,fees,money=100000,n=2)
        b,curve,trades=study.simulate_inventory(s,p,x,d,fees,money=100000,n=2)
        json.dumps(dict(metrics=b,curve=curve,trades=trades),allow_nan=False)
        self.assertFalse(a['fully_liquidated']);self.assertTrue(b['fully_liquidated'])
        self.assertLess(b['avg_exposure'],a['avg_exposure'])
        self.assertLess(b['trades'],a['trades'])
        self.assertAlmostEqual(b['gross_return_same_positions'],0.)

    def test_collapsing_liquidity_and_blocked_exit_can_still_leave_inventory(self):
        p,x,d,s=fixture()
        p.amount[3:]=0.;x.exit[4:]=False
        stat,curve,trades=study.simulate_inventory(s,p,x,d,fees,money=100000,n=2)
        self.assertFalse(stat['fully_liquidated']);self.assertGreater(stat['ending_position_value'],0.)
        self.assertGreater(stat['mean_zero_proxy_capacity_names'],0.)
        changed=copy.deepcopy(p);changed.amount[4:]=1e9
        _,_,newtrades=study.simulate_inventory(s,changed,x,d,fees,money=100000,n=2)
        cutoff=p.days[3]
        self.assertEqual([t for t in trades if t['date']<=cutoff],[t for t in newtrades if t['date']<=cutoff])


if __name__=='__main__':
    unittest.main()
