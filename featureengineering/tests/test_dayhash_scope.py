import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from fea.dayhash import _hash_one,_verify

class DayHashScopeTests(unittest.TestCase):
    def test_only_requested_years_are_read(self):
        df=pd.DataFrame({'trade_date':['2025-12-31','2026-01-02'],'stock_code':['a','a'],'value':[1.,2.],'rank':[.5,.5]})
        with patch('fea.dayhash.store.read_factor',return_value=df) as read:
            rows=_hash_one(('/unused','factor',('2025-12-31','2026-01-02')))
            self.assertEqual(read.call_args.kwargs['years'],[2025,2026])
            self.assertEqual(len(rows),2)
    def test_missing_baseline_does_not_pass_verification(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertNotEqual(_verify(Path(td)/'dayhash.tsv',[]),0)
