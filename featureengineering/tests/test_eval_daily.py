import unittest
import numpy as np
from fea.eval import _daily_corr, _deciles


class EvalDailyTests(unittest.TestCase):
    def test_ic_preserves_days_for_single_year_dispersion(self):
        x=np.tile(np.arange(10,dtype=float),(3,1))
        y=np.array([np.arange(10),np.arange(10)[::-1],[0,2,4,6,8,1,3,5,7,9]],dtype=float)
        ic,ric,n=_daily_corr(x,y,np.arange(3),5)
        expected=np.array([np.corrcoef(a,b)[0,1] for a,b in zip(x,y)])
        np.testing.assert_allclose(ic,expected)
        self.assertEqual(n,3)
        self.assertTrue(np.isfinite(ic.mean()/ic.std(ddof=1)))
        self.assertEqual(ric.shape,(3,))

    def test_missing_and_constant_days_are_excluded(self):
        x=np.array([[1,2,3,4],[1,1,1,1],[np.nan,2,np.nan,4]],dtype=float)
        y=np.tile(np.arange(4),(3,1))
        ic,ric,n=_daily_corr(x,y,np.arange(3),3)
        self.assertEqual(n,1)
        np.testing.assert_allclose(ic,[1])

    def test_deciles_combine_by_observed_days(self):
        x=np.tile(np.arange(10,dtype=float),(3,1))
        a,c=_deciles(x,np.ones_like(x),5)
        b,d=_deciles(x[:1],np.ones_like(x[:1])*5,5)
        np.testing.assert_allclose((a+b)/(c+d),np.full(10,2.0))


if __name__=='__main__':unittest.main()
