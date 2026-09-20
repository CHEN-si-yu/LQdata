import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from fea.deriv import Derivative, load_vintages, period_of, financial_dependency, _load_one
from fea.panel import Panel

def vintages(rows, **values):
    df=pd.DataFrame(rows,columns=['end_date_i','pit'])
    df['stock_code']='A';df['period']=period_of(df.end_date_i.to_numpy())
    for k,v in values.items():df[k]=v
    return df

def panel(*days):return Panel(np.array(days),np.array(['A']))

class FinancialVintages(unittest.TestCase):
    def test_future_revision_preserves_past(self):
        vt=vintages([(20251231,20260301),(20251231,20260701)],total_assets=[100.,150.])
        p=panel(20260301,20260630,20260701)
        np.testing.assert_array_equal(Derivative(vt,np.array(['A'])).to_panel(p,'total_assets','point')[:,0],[100,100,150])
        np.testing.assert_array_equal(Derivative(vt.iloc[:1],np.array(['A'])).to_panel(panel(20260301,20260630),'total_assets','point')[:,0],[100,100])

    def test_tables_independent_and_missing_revision_preserved(self):
        vt=vintages([(20251231,20260301),(20251231,20260401),(20260331,20260501),(20251231,20260601)],
            total_assets=[100,np.nan,np.nan,np.nan], revenue=[np.nan,200,50,np.nan],
            _available_stock_balancesheet=[True,np.nan,np.nan,True],
            _available_stock_income=[np.nan,True,True,np.nan])
        p=panel(20260301,20260401,20260501,20260601)
        batch=Derivative(vt,np.array(['A'])).to_panel(p,'total_assets','point')[:,0]
        single=Derivative(vt.iloc[[0,3]].drop(columns=['_available_stock_balancesheet','_available_stock_income','revenue']),np.array(['A'])).to_panel(p,'total_assets','point')[:,0]
        np.testing.assert_array_equal(batch,[100,100,100,np.nan]);np.testing.assert_array_equal(batch,single)

    def test_old_revision_refreshes_ttm_without_period_rollback(self):
        vt=vintages([(20240331,20240420),(20241231,20250301),(20250331,20250420),(20240331,20250501)],revenue=[20,100,30,25])
        d=Derivative(vt,np.array(['A']));p=panel(20250420,20250501)
        np.testing.assert_array_equal(d.to_panel(p,'revenue')[:,0],[110,105])
        np.testing.assert_array_equal(d.to_panel(p,'revenue','point')[:,0],[30,30])
        np.testing.assert_array_equal(d.to_panel(p,'revenue','point',4)[:,0],[20,25])

    def test_lagged_ttm_uses_revision_known_at_current_time(self):
        vt=vintages([(20230331,20230420),(20231231,20240301),(20240331,20240420),(20241231,20250301),(20250331,20250420),(20230331,20250501)],revenue=[10,80,20,100,30,15])
        d=Derivative(vt,np.array(['A']))
        np.testing.assert_array_equal(d.to_panel(panel(20250420,20250501),'revenue','ttm',4)[:,0],[90,85])

    def test_annual_without_prior_year(self):
        d=Derivative(vintages([(20251231,20260301)],revenue=[100]),np.array(['A']))
        self.assertEqual(d.to_panel(panel(20260301),'revenue')[0,0],100)

    def test_invalid_announcements_and_physical_cutoff(self):
        raw=pd.DataFrame({'stock_code':['A']*4,'end_date':['2025-12-31']*4,'ann_date':[None,'1970-01-01',None,'2026-07-01'],'f_ann_date':[None,None,'2026-03-01',None],'total_assets':[1,2,3,4]})
        up=SimpleNamespace(read=lambda *a,**kw:raw.copy(),audit_cutoff=20260501)
        got=_load_one(up,'stock_balancesheet',['total_assets'],2025,2026)
        self.assertEqual(got.total_assets.tolist(),[3]);self.assertEqual(got.pit.tolist(),[20260301])

    def test_dependency_recipe_scope(self):
        a=SimpleNamespace(name='a',deps=('stock_income',));b=SimpleNamespace(name='b',deps=('a',));c=SimpleNamespace(name='c',deps=('stock_daily',))
        reg={s.name:s for s in [a,b,c]}
        self.assertTrue(financial_dependency(a,reg));self.assertTrue(financial_dependency(b,reg));self.assertFalse(financial_dependency(c,reg))

if __name__=='__main__':unittest.main()
