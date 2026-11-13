"""做T决定只读取事前预测/开盘报价；收盘失败按真实账本保留。"""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from test_research_round import fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/_archive/20261024_20261025/scripts'))
spec = importlib.util.spec_from_file_location('t_forecast_test', ROOT/'experiments/_archive/20261024_20261025/scripts/strategy_t_20261025.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
model, analysis = study.common.load_analysis()


class ForecastTTests(unittest.TestCase):
    def test_ignores_t_callback_when_disabled_and_keeps_existing_cash_path(self):
        panel, px, days, pred, aux, prob, context, _ = fixture()
        base = analysis.strategy_backtest(pred, aux, prob, panel, px, days, context=context)
        off = analysis.strategy_backtest(pred, aux, prob, panel, px, days, context=context,
                                         t_decider=lambda *args: False)
        self.assertEqual(base[1:], off[1:])
        for key in ('return_value', 'total_fees', 'gross_return_same_positions', 't_attempts'):
            self.assertEqual(base[0][key], off[0][key])

    def test_opening_plan_does_not_use_actual_buyback_quote(self):
        panel, px, days, pred, aux, prob, context, _ = fixture()
        changed = copy.deepcopy(px)
        changed.raw['close'][2] = np.nan
        a = analysis.strategy_backtest(pred, aux, prob, panel, px, days, context=context,
                                      t_decider=lambda *args: True)
        b = analysis.strategy_backtest(pred, aux, prob, panel, changed, days, context=context,
                                      t_decider=lambda *args: True)
        sells = lambda x: [t for t in x[2] if t['date'] == panel.days[2] and t['reason'] == 't_sell_overnight']
        self.assertTrue(sells(a))
        self.assertEqual(sells(a), sells(b))
        self.assertGreater(b[0]['t_failed_buybacks'], 0)
        self.assertFalse(any(t['date'] == panel.days[2] and t['reason'] == 't_buyback' for t in b[2]))
        study.validate_cash(*b)

    def test_minimum_commission_can_reject_t_with_same_negative_forecast(self):
        small = study.expected_t_edge(-.005, 100, 10., .0003, analysis.fees)
        large = study.expected_t_edge(-.005, 1000, 10., .0003, analysis.fees)
        self.assertLess(small, 0)
        self.assertGreater(large, 0)
        self.assertLess(study.expected_t_edge(0., 1000, 10., .0003, analysis.fees), 0)
        self.assertIsNone(study.expected_t_edge(np.nan, 1000, 10., .0003, analysis.fees))

    def test_training_prefix_never_reads_a_later_execution_label(self):
        rng = np.random.default_rng(77)
        panel = SimpleNamespace(days=np.asarray([f'2025-01-{d+1:02}' for d in range(10)]),
                                codes=np.asarray(['A', 'B', 'C', 'D', 'E', 'F']),
                                features=['x', 'y'], market_dim=1,
                                X=rng.normal(size=(10, 6, 2)).astype(np.float32),
                                M=rng.normal(size=(10, 1)).astype(np.float32),
                                mask=lambda d: np.ones(6, bool))
        px = SimpleNamespace(raw=dict(open=np.full((10, 6), 10.),
                                     close=rng.uniform(9.8, 10.2, (10, 6)), vol=np.ones((10, 6))))
        changed = copy.deepcopy(px)
        changed.raw['close'][5:] = 10000.
        with patch.object(model, 'device_for', return_value=model.torch.device('cpu')):
            a = study.fit_prefixes(panel, px, [3], model)[3]
            b = study.fit_prefixes(panel, changed, [3], model)[3]
        self.assertEqual(a, b)
        self.assertEqual(a['last_training_execution'], panel.days[4])


if __name__ == '__main__':
    unittest.main()
