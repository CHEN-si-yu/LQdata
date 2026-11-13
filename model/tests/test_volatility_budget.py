"""风险预算只能依赖已经可得的收盘路径。"""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/_archive/20261024_20261025/scripts'))
spec=importlib.util.spec_from_file_location('volatility_policy_test',ROOT/'experiments/_archive/20261024_20261025/scripts/strategy_volatility_20261025.py')
study=importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class VolatilityBudgetTests(unittest.TestCase):
    def test_future_shocks_do_not_change_preceding_opening_risk_plan(self):
        path=np.tile(np.asarray([-.01,.01]),40)
        a,sd=study.volatility_budget(path)
        changed=path.copy()
        changed[51:]=np.tile([-.3,.3],15)[:len(changed)-51]
        b,_=study.volatility_budget(changed)
        np.testing.assert_array_equal(a[:51],b[:51])
        self.assertGreater(a[55],b[55])
        self.assertGreaterEqual(float(b.min()),0.)
        self.assertLessEqual(float(b.max()),1.)
        self.assertAlmostEqual(a[40],.15/np.sqrt(242)/sd[40])


if __name__=='__main__':
    unittest.main()
