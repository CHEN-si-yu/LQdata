"""A shared input snapshot can feed models with different frozen market subsets."""
import importlib.util
import os
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(os.environ.get('GREEDY_RELEASE_RUNNER',str(ROOT/'releases/main.py')))
spec=importlib.util.spec_from_file_location('market_contract_daily',SCRIPT)
daily=importlib.util.module_from_spec(spec);spec.loader.exec_module(daily)
class MarketContractTests(unittest.TestCase):
    def setUp(self):
        self.meta=dict(semantics='zscore_win1_99_v1',columns=dict(direction=dict(f=1)))
        self.contracts=dict(snapshots=dict(frozen=dict(semantics='zscore_win1_99_v1',feature_direction=dict(f=-1),market_columns=['a','b','c'],market_columns_by_model=dict(short=['a','b'],full=['a','b','c']))))
    def signs(self,name,market):
        return daily.direction_multipliers(Path(name),['f'],market,self.meta,'frozen','current',self.contracts)
    def test_shared_snapshot_accepts_each_exact_frozen_model_subset(self):
        self.assertEqual(self.signs('short',['a','b']),[-1])
        self.assertEqual(self.signs('full',['a','b','c']),[-1])
    def test_cross_model_and_reordered_columns_are_rejected(self):
        for name,market in [('short',['a','b','c']),('full',['a','b']),('short',['b','a'])]:
            with self.subTest(name=name,market=market),self.assertRaisesRegex(ValueError,'市场列名/顺序'):
                self.signs(name,market)
    def test_unknown_model_does_not_fall_back_to_snapshot_wide_columns(self):
        with self.assertRaisesRegex(ValueError,'缺少该模型'):
            self.signs('unknown',['a','b','c'])
    def test_legacy_single_market_contract_remains_strict(self):
        self.contracts['snapshots']['frozen'].pop('market_columns_by_model')
        self.assertEqual(self.signs('full',['a','b','c']),[-1])
        with self.assertRaisesRegex(ValueError,'市场列名/顺序'):
            self.signs('short',['a','b'])
if __name__=='__main__':unittest.main()
