"""统一Top500等权分配的资金、容量和未定义相关性合同。"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common=load('test_large_common',ROOT/'experiments/_archive/20261024_20261025/scripts/strategy_20261024.py')
with patch.dict(sys.modules,strategy_20261024=common):
    study=load('test_equal_allocation',ROOT/'experiments/_archive/20261024_20261025/scripts/large_equal_weight_20261025.py')


class LargeAllocationTests(unittest.TestCase):
    def test_small_capacities_leave_cash_without_filling_outside_top500(self):
        _,analysis=common.load_analysis()
        amount=np.full((1,600),4e6)
        amount[0,:4]=[1e6,0,np.inf,np.nan]
        panel=SimpleNamespace(amount=amount,Y={'label_ret_1d':np.full((1,600),.01,np.float32)})
        score=np.arange(600,0,-1,dtype=float)[None]
        returns,invested,diagnostics=study.equal_weight_daily(score,panel,np.asarray([0]),analysis)
        expected=(496*3e6+1e6)/1.5e9
        self.assertAlmostEqual(float(invested[0]),expected)
        self.assertAlmostEqual(float(returns[0]),.01*expected)
        self.assertLessEqual(diagnostics['maximum_single_weight'],1/500)
        self.assertEqual(diagnostics['median_funded_names'],497)
        self.assertLessEqual(diagnostics['median_effective_names'],497)

    def test_no_capacity_keeps_cash_and_constant_return_correlation_is_undefined(self):
        _,analysis=common.load_analysis()
        panel=SimpleNamespace(amount=np.zeros((1,600)),Y={'label_ret_1d':np.ones((1,600))})
        score=np.arange(600,0,-1,dtype=float)[None]
        returns,invested,diagnostics=study.equal_weight_daily(score,panel,np.asarray([0]),analysis)
        self.assertEqual(float(returns[0]),0.)
        self.assertEqual(float(invested[0]),0.)
        self.assertEqual(diagnostics['median_effective_names'],0.)
        matrix=study.correlation_matrix({'cash':[0.,0.,0.],'risk':[.1,-.1,.1]})['pearson']
        self.assertIsNone(matrix[0][1])

    def test_risk_budget_resizes_target_before_capacity_clipping(self):
        _,analysis=common.load_analysis()
        amount=np.full((1,600),4e6)
        amount[0,0]=1e6
        labels=np.zeros((1,600),np.float32)
        labels[0,0]=.03
        panel=SimpleNamespace(amount=amount,Y={'label_ret_1d':labels})
        score=np.arange(600,0,-1,dtype=float)[None]
        full,_,_=study.equal_weight_daily(score,panel,np.asarray([0]),analysis)
        reduced,_,_=study.equal_weight_daily(score,panel,np.asarray([0]),analysis,np.asarray([.5]))
        # 第一只仍由100万元容量限制，不能直接把原收益乘0.5。
        self.assertAlmostEqual(float(reduced[0]),float(full[0]))
        self.assertGreater(float(reduced[0]),float(full[0])*.5)


if __name__=='__main__':unittest.main()
