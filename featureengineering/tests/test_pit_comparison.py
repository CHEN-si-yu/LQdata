import unittest
import pandas as pd
import numpy as np
from fea.pit_audit import compare_frames

class PitComparisonTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame({'trade_date':['2020-01-02']*2,'stock_code':['a','b'],'value':[1.,np.nan],'rank':[.5,np.nan]})
    def test_missing_or_partial_results_do_not_pass(self):
        for other in [self.sample().iloc[:0],self.sample().iloc[:1]]:
            with self.assertRaises(ValueError):compare_frames(self.sample(),other,0)
    def test_nan_and_rank_differences_are_checked(self):
        for column in ['value','rank']:
            other=self.sample();other.loc[1,column]=0
            with self.assertRaises(ValueError):compare_frames(self.sample(),other,0)
    def test_duplicates_do_not_pass(self):
        other=self.sample();other.loc[1,'stock_code']='a'
        with self.assertRaises(ValueError):compare_frames(self.sample(),other,0)
    def test_equal_frames_pass_in_any_row_order(self):
        self.assertEqual(compare_frames(self.sample(),self.sample().iloc[::-1],0),{'rows':2,'finite_values':1})
