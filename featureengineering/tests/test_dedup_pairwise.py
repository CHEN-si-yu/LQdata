import unittest
import numpy as np
from fea.dedup import pairwise_redundancy


class PairwiseRedundancyTests(unittest.TestCase):
    def test_same_rank_is_not_same_model_value_information(self):
        x=np.tile(np.linspace(1,50,100),(3,1))
        daily=pairwise_redundancy(x,x**4,min_n=3)
        np.testing.assert_allclose(daily.rank_corr,1)
        np.testing.assert_allclose(daily.rank_equal_fraction,1)
        self.assertLess(daily.model_value_corr.mean(),.9)
        self.assertLess(daily.model_equal_fraction.mean(),.1)

    def test_missing_pairs_not_zero_filled_and_overlap_reported(self):
        a=np.array([[1,2,3,4,np.nan,np.nan]],float)
        b=np.array([[2,4,6,np.nan,10,np.nan]],float)
        daily=pairwise_redundancy(a,b,min_n=3).iloc[0]
        self.assertEqual(daily.common,3)
        self.assertEqual(daily.union,5)
        self.assertAlmostEqual(daily.jaccard,.6)
        self.assertAlmostEqual(daily.rank_corr,1)
        self.assertAlmostEqual(daily.model_value_corr,1)

    def test_direction_contract_and_insufficient_coverage(self):
        a=np.tile(np.arange(10.),(2,1))
        daily=pairwise_redundancy(a,-a,min_n=3,higher_b=False)
        np.testing.assert_allclose(daily.model_value_corr,1)
        empty=pairwise_redundancy(a,-a,min_n=100)
        self.assertTrue(empty.rank_corr.isna().all())


if __name__=='__main__':unittest.main()
