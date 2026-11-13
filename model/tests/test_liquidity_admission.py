"""Buy admission uses known history and must not force an existing inventory exit."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from test_large_cash import fixture,fees,study as baseline

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('admission_study_test',ROOT/'experiments/_archive/20261024_20261025/scripts/large_admission_20261025.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


class LiquidityAdmissionTests(unittest.TestCase):
    def test_unrestricted_admission_reproduces_baseline_fills(self):
        p,x,d,s=fixture()
        a,ac,at=baseline.simulate(s,p,x,d,fees,money=100000,n=2)
        b,bc,bt=study.simulate_admission(s,p,x,d,fees,money=100000,n=2)
        self.assertEqual(at,bt)
        for key in a:
            if key=='order_constraints':
                for name,value in a[key].items():self.assertEqual(value,b[key][name])
            else:self.assertEqual(a[key],b[key])
        for left,right in zip(ac,bc):
            for key,value in left.items():self.assertEqual(value,right[key])

    def test_later_failed_admission_preserves_existing_shares(self):
        p,x,d,s=fixture();capacity=np.zeros((len(p.days),len(p.codes)))
        capacity[0]=np.inf
        stat,curve,trades=study.simulate_admission(s,p,x,d,fees,money=100000,n=2,admission_capacity=capacity,liquidate=False)
        self.assertTrue(any(t['side']=='buy' for t in trades))
        self.assertFalse(any(t['side']=='sell' for t in trades))
        self.assertTrue(all(row['positions']>0 for row in curve))
        self.assertGreater(stat['order_constraints']['admission_buy'],0)
        changed=capacity.copy();changed[3:]=np.inf
        _,_,other=study.simulate_admission(s,p,x,d,fees,money=100000,n=2,admission_capacity=changed,liquidate=False)
        self.assertEqual([t for t in trades if t['date']<=p.days[3]],[t for t in other if t['date']<=p.days[3]])

    def test_capacity_uses_only_known_five_day_minimum(self):
        amount=np.asarray([[100],[80],[90],[120],[110],[70],[200]],float)
        expected=study.admission_capacity(amount)
        np.testing.assert_array_equal(expected[:4],0)
        self.assertAlmostEqual(expected[4,0],.8)
        self.assertAlmostEqual(expected[5,0],.7)
        amount[5:]=1e12
        np.testing.assert_array_equal(study.admission_capacity(amount)[:5],expected[:5])


if __name__=='__main__':unittest.main()
