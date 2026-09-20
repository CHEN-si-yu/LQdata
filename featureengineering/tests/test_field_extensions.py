import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from fea.context import FactorContext
from fea.panel import Panel
from factors import cyq_perf as cp, disclosure_detail as dd, financial_detail as fd

def context(days,df):
    panel=Panel(np.asarray(days,dtype=np.int32),np.array(['600000.SH']))
    up=SimpleNamespace(read=lambda name,columns=None,years=None:df[columns].copy())
    return FactorContext(panel,None,up,None,np.ones(panel.shape,dtype=bool))

class FieldExtensionTests(unittest.TestCase):
    def test_winner_units_invalid_and_missing_day(self):
        rows=[]
        for day,win in [('2018-01-02',50.),('2018-01-03',100.4),('2018-01-05',0.)]:
            rows.append(dict(stock_code='600000.SH',trade_date=day,his_low=1,his_high=20,
                             cost_5pct=5,cost_15pct=6,cost_50pct=10,cost_85pct=14,cost_95pct=15,weight_avg=11,winner_rate=win))
        ctx=context([20180102,20180103,20180104,20180105],pd.DataFrame(rows))
        np.testing.assert_allclose(cp.cyqp_winner_fraction(ctx)[:,0],[.5,np.nan,np.nan,0],equal_nan=True)
        np.testing.assert_allclose(cp.cyqp_cost_width_90(ctx)[:,0],[1,1,np.nan,1],equal_nan=True)
        np.testing.assert_allclose(cp.cyqp_mean_median_gap(ctx)[:,0],[.1,.1,np.nan,.1],atol=1e-7,equal_nan=True)

    def test_forecast_future_disclosure_does_not_rewrite_history(self):
        df=pd.DataFrame({'stock_code':['600000.SH']*2,'ann_date':['2020-01-02','2020-01-06'],
                         'net_profit_min':[1000.,2000.],'net_profit_max':[2000.,3000.],'last_parent_net':[1000.,1000.]})
        a=dd.forecast_profit_midpoint_change(context([20200102,20200103],df))
        b=dd.forecast_profit_midpoint_change(context([20200102,20200103],df.iloc[:1]))
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(a,.5)
        wide=dd.forecast_profit_range_uncertainty(context([20200102,20200103],df))
        np.testing.assert_allclose(wide,1.)

    def test_forecast_staleness_and_outside_universe(self):
        df=pd.DataFrame({'stock_code':['600000.SH'],'ann_date':['2018-01-02'],
                         'net_profit_min':[1000.],'net_profit_max':[2000.],'last_parent_net':[1000.]})
        self.assertTrue(np.isnan(dd.forecast_profit_midpoint_change(context([20200102],df))).all())
        df['stock_code']='300001.SZ'
        self.assertTrue(np.isnan(dd.forecast_profit_midpoint_change(context([20180102],df))).all())

    def test_seal_event_means_ignore_non_event_days(self):
        days=np.array([int(d.strftime('%Y%m%d')) for d in pd.bdate_range('2020-01-01',periods=25)])
        df=pd.DataFrame({'stock_code':['600000.SH']*2,'trade_date':['2020-01-03','2020-01-10'],
                         'sealed_flow_ratio':[2.,6.]})
        arr=dd.seal_float_strength_20(context(days,df))[:,0]
        self.assertAlmostEqual(float(arr[19]),4.)
        self.assertAlmostEqual(float(arr[24]),6.)
        self.assertTrue(np.isnan(arr[:19]).all())

    def test_financial_formula_uses_ttm_and_point(self):
        vals={'c_recp_borrow':np.array([[8e6]]),'c_prepay_amt_borr':np.array([[3e6]]),'total_assets':np.array([[1e8]])}
        ctx=SimpleNamespace(ttm=lambda k:vals[k],point=lambda k:vals[k],safe_div=FactorContext.safe_div)
        np.testing.assert_allclose(fd.cf_net_borrowing_to_assets(ctx),.05)

if __name__=='__main__':unittest.main()
