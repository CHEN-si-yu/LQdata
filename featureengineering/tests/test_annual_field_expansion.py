import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from fea.context import FactorContext
from fea.deriv import Derivative, load_vintages, period_of, FIELD_SOURCE
from fea.field_expansion import CATALOG, annual_alias
from fea.panel import Panel
from fea.spec import REGISTRY
import factors

DS = 'stock_financial_indicator'
FIELD = 'bps'
ALIAS = annual_alias(DS, FIELD)

def context(rows, days):
    vt = pd.DataFrame(rows, columns=['end_date_i', 'pit', ALIAS])
    vt['stock_code'] = 'A'
    vt['period'] = period_of(vt.end_date_i.to_numpy())
    panel = Panel(np.array(days, dtype=np.int32), np.array(['A']))
    return FactorContext(panel, Derivative(vt, panel.codes), None, None, np.ones(panel.shape, bool))

class AnnualExpansionTests(unittest.TestCase):
    def test_annual_is_not_overwritten_by_new_quarter(self):
        c = context([(20231231,20240310,10), (20241231,20250310,20), (20250331,20250420,999)],
                    [20250309,20250310,20250420])
        np.testing.assert_array_equal(c.annual(DS, FIELD)[:,0], [10,20,20])
        np.testing.assert_array_equal(c.annual(DS, FIELD,1)[:,0], [np.nan,10,10])

    def test_old_revision_changes_lag_only_on_disclosure(self):
        rows=[(20231231,20240310,10), (20241231,20250310,20), (20250331,20250420,999),
              (20231231,20250510,15)]
        c=context(rows,[20250509,20250510])
        np.testing.assert_array_equal(c.annual(DS,FIELD)[:,0], [20,20])
        np.testing.assert_array_equal(c.annual(DS,FIELD,1)[:,0], [10,15])
        np.testing.assert_array_equal(context(rows,[20250509]).annual(DS,FIELD,1),
                                      context(rows[:-1],[20250509]).annual(DS,FIELD,1))

    def test_missing_year_not_previous_available_year(self):
        c=context([(20221231,20230310,10),(20241231,20250310,20)], [20250310])
        self.assertTrue(np.isnan(c.annual(DS,FIELD,1)).all())

    def test_explicit_missing_revision_remains_missing(self):
        c=context([(20241231,20250310,20),(20241231,20250510,np.nan)], [20250509,20250510])
        np.testing.assert_array_equal(c.annual(DS,FIELD)[:,0], [20,np.nan])

    def test_no_annual_report_and_invalid_field(self):
        c=context([(20250331,20250420,3)], [20250420])
        self.assertTrue(np.isnan(c.annual(DS,FIELD)).all())
        with self.assertRaises(ValueError):c.annual(DS,FIELD,-1)
        with self.assertRaises(KeyError):c.annual(DS,'invented_field')

    def test_namespaced_alias_loader_and_existing_source_unchanged(self):
        ds1,ds2='stock_income','stock_financial_indicator'
        a1,a2=annual_alias(ds1,'ebitda'),annual_alias(ds2,'ebitda')
        def read(ds,columns,**kwargs):
            d={'stock_code':['A'],'ann_date':['2025-03-10'],'f_ann_date':['2025-03-10'],
               'end_date':['2024-12-31'],'ebit':[10 if ds==ds1 else 30],'ebitda':[10 if ds==ds1 else 30]}
            return pd.DataFrame(d)[columns]
        vt=load_vintages(SimpleNamespace(read=read),2024,2025,frozenset([a1,a2,'ebit']))
        self.assertEqual(vt[a1].iloc[0],10)
        self.assertEqual(vt[a2].iloc[0],30)
        self.assertEqual(vt.ebit.iloc[0],10)
        self.assertEqual(FIELD_SOURCE['ebit'],ds1)

    def test_every_catalog_formula_has_real_dependencies_and_finite_guard(self):
        for entry in CATALOG:
            spec=REGISTRY[entry['name']]
            calls=[]
            def annual(ds,f,lag_years=0):
                calls.append((ds,f))
                return np.array([[3.,np.nan,-2.,np.inf]]) if lag_years==0 else np.array([[1.,1.,0.,2.]])
            ctx=SimpleNamespace(annual=annual,safe_div=FactorContext.safe_div)
            out=spec.fn(ctx)
            self.assertEqual(out.shape,(1,4))
            self.assertFalse(np.isinf(out).any())
            self.assertEqual(set(calls),{(ds,f) for ds,fs in entry['source_fields'].items() for f in fs})
            self.assertEqual(set(spec.deps),set(entry['source_fields']))

    def test_income_per_share_growth_keeps_units_and_denominator_guard(self):
        # ★ 2026-09-22 去冗余：同一路径的 `afx_is_diluted_eps`（与 `afx_is_basic_eps`
        #   |ρ|=0.982）已删 —— 该口径下现在只剩这一个因子，单独覆盖同一条代码路径。
        for name in ('afx_is_basic_eps',):
            def annual(ds,f,lag_years=0):
                return np.array([[.4,-.1,.5]]) if not lag_years else np.array([[.2,-.2,0.]])
            c=SimpleNamespace(annual=annual,safe_div=FactorContext.safe_div)
            np.testing.assert_allclose(REGISTRY[name].fn(c),np.arcsinh([[1.,.5,np.nan]]),equal_nan=True)

if __name__=='__main__':unittest.main()
