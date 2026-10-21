import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fea import mathx as mx
from fea.eval import (evaluate_panel,evaluate_pair,model_value_panel,
                      mean_uncertainty,summarize_daily,write_evaluation,validate_frame_grid)
from fea.targets import HORIZONS,open_to_open_labels,last_mature_signal_date


class LabelContractTests(unittest.TestCase):
    def test_helper_matches_formal_label_arithmetic_all_horizons(self):
        rng=np.random.default_rng(3)
        prices=np.exp(rng.normal(size=(70,10)))
        traded=rng.random((70,10))>.2
        traded[-4:]=False
        prices[1,0]=1e-12
        prices[2,1]=np.inf
        prices[3,2]=np.nan
        for h in HORIZONS:
            raw=mx.safe_div(mx.shift(prices,-(h+1)),mx.shift(prices,-1),min_abs_den=1e-12)-1
            endpoints=(mx.shift(traded.astype(float),-1)==1)&(mx.shift(traded.astype(float),-(h+1))==1)
            last=np.flatnonzero(traded.any(axis=1))[-1]
            expected=np.where((np.arange(len(prices))+h+1<=last)[:,None]&endpoints,raw,np.nan)
            np.testing.assert_array_equal(open_to_open_labels(prices,traded,h),expected)

    def test_suspension_cannot_become_label_by_forward_filling_prices(self):
        prices=np.tile(np.arange(1.,9.)[:,None],(1,3))
        traded=np.ones_like(prices,bool)
        traded[1,0]=False
        traded[2,1]=False
        labels=open_to_open_labels(prices,traded,1)
        self.assertTrue(np.isnan(labels[0,0]))
        self.assertTrue(np.isnan(labels[0,1]))
        self.assertAlmostEqual(labels[0,2],.5)
        self.assertTrue(np.isnan(labels[-2:]).all())
        self.assertEqual(last_mature_signal_date(np.arange(8),traded,1),5)


class EvaluationContractTests(unittest.TestCase):
    def test_all_horizons_missing_const_and_sparse_groups(self):
        values=np.array([[0,0,0,1,1,1],[1,1,1,1,1,1],[1,2,np.inf,4,5,6]],float)
        labels={h:np.tile(np.arange(6.),(3,1)) for h in HORIZONS}
        labels[20][-1]=np.nan
        daily=evaluate_panel(values,labels,['2026-01-01','2026-01-02','2026-01-03'],min_n=3)
        self.assertEqual(len(daily),15)
        self.assertEqual(set(daily.horizon),set(HORIZONS))
        binary=daily[(daily.horizon==1)&(daily.trade_date=='2026-01-01')].iloc[0]
        self.assertEqual(binary.actual_bins,2)
        self.assertEqual(binary.bin_counts,[3,3])
        self.assertAlmostEqual(binary.monotonicity,1)
        const=daily[(daily.horizon==1)&(daily.trade_date=='2026-01-02')].iloc[0]
        self.assertTrue(np.isnan(const.rankic))
        self.assertEqual(const.reason,'constant_value_or_target')
        self.assertEqual(daily[(daily.horizon==20)&(daily.trade_date=='2026-01-03')].iloc[0].pairs,0)
        summary=summarize_daily(daily)
        self.assertEqual(len(summary),5)
        self.assertEqual(next(r for r in summary if r['horizon']==20)['days'],1)

    def test_true_rank_monotonicity_differs_from_linear_correlation(self):
        values=np.arange(30.)[None,:]
        y=(np.arange(30.)**3)[None,:]
        daily=evaluate_panel(values,{5:y},['2026-01-01'],min_n=3,bins=10)
        self.assertAlmostEqual(daily.iloc[0].monotonicity,1.)

    def test_pairing_uses_same_stock_mask(self):
        a=np.array([[1,2,3,4,5,6],[6,5,4,3,2,1]],float)
        b=a*2
        b[:,0]=np.nan
        a[:,1]=np.nan
        labels={5:np.tile(np.arange(6.),(2,1))}
        da,db,comparison=evaluate_pair(a,b,labels,['2026-01-01','2026-01-02'],min_n=3)
        self.assertEqual(da.pairs.tolist(),[4,4])
        self.assertEqual(db.pairs.tolist(),[4,4])
        self.assertAlmostEqual(comparison[0]['difference']['mean'],0)

    def test_hac_accounts_for_persistent_date_sequence(self):
        values=np.repeat([1.,-1.,1.,-1.],20)
        naive=mean_uncertainty(values,block_days=1)['hac_se']
        robust=mean_uncertainty(values,block_days=20)['hac_se']
        self.assertGreater(robust,naive)
        self.assertGreaterEqual(mean_uncertainty(values,block_days=1,horizon=20)['hac_lag'],19)

    def test_model_transform_preserves_missing_and_orientation(self):
        values=np.array([[1,2,3,np.nan],[1,1,1,np.nan]],float)
        transformed=model_value_panel(values)
        self.assertTrue(np.isnan(transformed[:,-1]).all())
        np.testing.assert_allclose(transformed[1,:3],0)
        np.testing.assert_allclose(model_value_panel(values,False),-transformed)

    def test_full_grid_validation_distinguishes_missing_values_and_missing_rows(self):
        import pandas as pd
        frame=pd.DataFrame({'trade_date':['2026-01-05']*3,'stock_code':['A','B','C'],'value':[1,np.nan,3]})
        validate_frame_grid(frame,[20260105],['A','B','C'])
        with self.assertRaises(ValueError):
            validate_frame_grid(frame.iloc[:-1],[20260105],['A','B','C'])
        frame.loc[2,'stock_code']='A'
        with self.assertRaises(ValueError):
            validate_frame_grid(frame,[20260105],['A','B','C'])

    def test_runs_are_immutable_json_valid_and_daily_preserved(self):
        daily=evaluate_panel(np.arange(10.)[None,:],{1:np.arange(10.)[None,:]},['2026-01-01'],min_n=3)
        with tempfile.TemporaryDirectory() as directory:
            a=write_evaluation(daily,directory)
            old=(a/'summary.json').read_bytes()
            b=write_evaluation(daily,directory)
            self.assertNotEqual(a,b)
            self.assertEqual((a/'summary.json').read_bytes(),old)
            self.assertNotIn(b'NaN',old)
            self.assertEqual(json.loads((Path(directory)/'latest.json').read_text())['run_id'],b.name)


if __name__=='__main__':unittest.main()
