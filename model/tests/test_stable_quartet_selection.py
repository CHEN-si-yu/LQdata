"""Critical guards for monthly risk, label maturity and common-universe fusion."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stable_quartet_helpers',ROOT/'experiments/_archive/stable_quartet_20261030/scripts/common.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
class StableSelectionTests(unittest.TestCase):
    def test_high_total_profit_does_not_hide_bad_calendar_month(self):
        dates=np.asarray(pd.bdate_range('2026-01-05',periods=60).strftime('%Y-%m-%d'));r=np.r_[np.full(20,.012),np.full(20,-.009),np.full(20,.012)];s=helper.stats_from_returns(r,dates)
        self.assertGreater(s['return_value'],0)
        self.assertIn('month_below_minus5pct',helper.actual_gate(s,.05,.5))
    def test_unavailable_open_to_open_labels_cannot_change_selection_IC(self):
        x=np.tile([0.,1.,2.,3.],(5,1));y=np.array([[0.,2.,1.,3.],[3.,0.,2.,1.],[0.,1.,3.,2.],[0.,1.,2.,3.],[0.,1.,2.,3.]])
        p=SimpleNamespace(Y={'label_ret_1d':y});before=helper.ic_stats(x,p,np.arange(5),4);y[3:]=np.array([[1e9,-1e9,0.,0.],[-1e9,1e9,1e9,-1e9]]);self.assertEqual(before,helper.ic_stats(x,p,np.arange(5),4))
        y[2]=[3.,2.,1.,0.];self.assertNotEqual(before,helper.ic_stats(x,p,np.arange(5),4))
    def test_cash_months_and_quarters_compound_without_reset(self):
        s=helper.stats_from_returns([.1,-.05,.02],['2025-12-30','2025-12-31','2026-01-05']);self.assertAlmostEqual(s['return_value'],1.1*.95*1.02-1);self.assertAlmostEqual(np.prod([1+v for v in s['quarterly'].values()])-1,s['return_value'])
    def test_zero_variance_account_dependence_is_undefined(self):
        self.assertIsNone(helper.pearson(np.zeros(20),np.arange(20)))
    def test_fusion_ranks_only_the_common_finite_stock_pool(self):
        x=np.array([[0.,1.,2.,np.nan]]);y=np.array([[2.,2.,np.nan,0.]]);z=helper.fuse([x,y]);np.testing.assert_allclose(z[0,:2],[.375,.625]);self.assertTrue(np.isnan(z[0,2:]).all())
    def test_positive_cash_profit_with_weak_signal_does_not_qualify(self):
        s=helper.stats_from_returns(np.full(60,.001),pd.bdate_range('2026-01-05',periods=60).strftime('%Y-%m-%d'));self.assertIn('weak_IC_or_ICIR',helper.actual_gate(s,.005,.05))
if __name__=='__main__':unittest.main()
