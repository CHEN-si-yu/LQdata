"""Single-target training still reports distinct one/five-day auxiliary metrics."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from test_dart_model import model

ROOT=Path(__file__).resolve().parents[1]
previous=sys.modules.get('model')
sys.modules['model']=model
try:
    spec=importlib.util.spec_from_file_location('tree_report_test',ROOT/'experiments/V279/analysis.py')
    analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
finally:
    if previous is None:del sys.modules['model']
    else:sys.modules['model']=previous


class TreeEvaluationReportTests(unittest.TestCase):
    def test_single_label_has_two_distinct_report_horizons_and_correct_proxy(self):
        quarters=['2025Q3','2025Q4','2026Q1','2026Q2','2026Q3']
        records=[dict(quarter=q,fold=f,best_validation=dict(IC_1d=.03,IC_5d=.05,
            RankIC_1d=.04,RankIC_5d=.06,val_wei=.1,val_return_proxy=.001)) for q in quarters for f in range(1,5)]
        valid={(q,f):63 for q in quarters for f in range(1,5)}
        with patch.object(analysis,'axis',return_value=(np.asarray(['2025-07-01']),None)), \
             patch.object(analysis,'quarters',return_value=quarters),patch.object(analysis,'fold_valid_days',return_value=valid):
            self.assertEqual(analysis.rank_ic_horizons(),(1,5))
            rows=analysis.training_validation_rows(records)
        self.assertEqual(len(rows),6)
        self.assertAlmostEqual(rows[-1]['IC_1d'],.03)
        self.assertAlmostEqual(rows[-1]['IC_5d'],.05)
        self.assertAlmostEqual(rows[-1]['total_return'],100*((1.001)**63-1))


if __name__=='__main__':unittest.main()
