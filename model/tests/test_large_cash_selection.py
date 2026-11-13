"""模型内嵌现金函数和选轮接口须与已审阅引擎对齐。"""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from test_large_cash import fixture,fees,study

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cash_selection_test',ROOT/'experiments/V274/model.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


class LargeCashSelectionTests(unittest.TestCase):
    def test_embedded_account_equals_independent_strategy_implementation(self):
        panel,px,days,pred=fixture()
        a=study.simulate(pred,panel,px,days,fees,money=100000,n=2)
        b=model.large_cash_simulate(pred,panel,px,days,model.large_cash_fee,money=100000,n=2)
        self.assertEqual(a,b)

    def test_selector_uses_large_cash_and_only_validation_realization_tail(self):
        panel,px,days,pred=fixture()
        y=np.tile(np.arange(6,dtype=np.float32)*.01,(9,1))
        panel.Y={'label_ret_1d':y,'label_ret_5d':y*2}
        changed=copy.deepcopy(px)
        changed.open[7:]=1000.;changed.close[7:]=1000.
        with patch.object(model,'_macro_base_validation',side_effect=lambda *args,**kwargs:{}),patch.object(model,'predict',return_value=pred):
            a=model.validation(object(),panel,px,days,model.torch.device('cpu'))
            b=model.validation(object(),panel,changed,days,model.torch.device('cpu'))
        self.assertEqual(a,b)
        m=a['macro_validation']
        self.assertEqual(m['money'],1500000000)
        self.assertEqual(m['topn'],500)
        self.assertFalse(m['small_account_used_for_epoch_selection'])
        self.assertEqual(m['cash_end'],panel.days[6])
        self.assertGreater(m['cash_ending_positions'],0)


if __name__=='__main__':
    unittest.main()
