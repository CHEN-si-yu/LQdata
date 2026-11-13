"""用小型账本验证约束，避免仅凭净收益曲线判断执行正确。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
model = load('research_model', ROOT / 'experiments/V253/model.py')
previous = sys.modules.get('model')
sys.modules['model'] = model
try:
    analysis = load('research_analysis', ROOT / 'experiments/V253/analysis.py')
finally:
    if previous is None:
        del sys.modules['model']
    else:
        sys.modules['model'] = previous

def fixture():
    t, c = (10, 6)
    days = pd.date_range('2025-07-01', periods=t).strftime('%Y-%m-%d').to_numpy(dtype=str)
    panel = SimpleNamespace(days=days, codes=np.asarray([f'{i:06}.SZ' for i in range(c)]), amount=np.full((t, c), 100000000.0))
    px = SimpleNamespace(open=np.full((t, c), 10.0), close=np.full((t, c), 10.0), adj=np.ones((t, c)), limit_up=np.full((t, c), 11.0), limit_down=np.full((t, c), 9.0), entry=np.ones((t, c), bool), exit=np.ones((t, c), bool))
    px.close[:] = 10.4
    px.raw = {'close': np.full((t, c), 10.4), 'pre_close': np.full((t, c), 10.0), 'vol': np.full((t, c), 1000000.0)}
    px.entry_ok = lambda: px.entry
    signals = np.arange(5)
    pred = np.tile(np.arange(c, 0, -1), (len(signals), 1)).astype(float)
    aux = np.full((*pred.shape, 2), 0.02)
    prob = np.zeros(len(signals))
    context = {'mkt_above_ma20_ratio': np.ones(t), 'mkt_idx_csi300_ret1': np.zeros(t), 'mkt_limit_down_ratio': np.zeros(t)}
    morning = SimpleNamespace(signal=np.full((t, c), 10.2), sell=np.full((t, c), 10.2), buy=np.full((t, c), 10.0), sell_cap=np.full((t, c), 10000000.0), buy_cap=np.full((t, c), 10000000.0))
    return (panel, px, signals, pred, aux, prob, context, morning)

class CashContractTests(unittest.TestCase):

    def test_recent_account_marks_existing_positions_without_future_liquidation(self):
        panel,px,days,pred,aux,prob,context,_=fixture()
        panel.days=panel.days[:6]
        stat,curve,trades=analysis.strategy_backtest(pred,aux,prob,panel,px,days,context=context,liquidate=False)
        self.assertEqual(len(curve),len(days))
        self.assertEqual(stat['end'],panel.days[-1])
        self.assertEqual(stat['ending_positions'],5)
        self.assertTrue(all(t['time']=='09:30' for t in trades))

    def test_t_buyback_uses_same_day_closing_quote(self):
        _,px,_,_,_,_,_,_=fixture()
        _,_,trades=self.run_account('open_close_t')
        buys=[t for t in trades if t['reason']=='t_buyback']
        self.assertTrue(buys)
        for trade in buys:
            self.assertEqual(trade['time'],'15:00')
            self.assertAlmostEqual(trade['price'],10.4*(1+.0003))

    def run_account(self, rule='baseline', **changes):
        panel, px, days, pred, aux, prob, context, morning = fixture()
        for name, value in changes.items():
            if name == 'prob':
                prob = value
            elif name == 'morning':
                morning = value
            elif name == 'px':
                px = value
        return analysis.strategy_backtest(pred, aux, prob, panel, px, days, rule=rule, context=context)

    def test_uniform_baseline_matches_existing_engine(self):
        panel, px, days, pred, aux, prob, context, morning = fixture()
        expected, _, _ = analysis.cash_backtest(pred, panel, px, days, n=5, period=1, band=40, cost_gate_multiplier=0)
        actual, _, _ = analysis.strategy_backtest(pred, aux, prob, panel, px, days, context=context)
        for key in ('return_value', 'max_drawdown', 'trades', 'total_fees', 'gross_return_same_positions'):
            self.assertAlmostEqual(expected[key], actual[key], places=10)

    def test_open_close_t_only_sells_overnight_and_orders_are_chronological(self):
        stat, curve, trades = self.run_account('open_close_t')
        self.assertGreater(stat['t_attempts'], 0)
        self.assertTrue(all((t['date'] != '2025-07-02' for t in trades if t['reason'] == 't_sell_overnight')))
        self.assertEqual([(t['date'], t['time']) for t in trades], sorted(((t['date'], t['time']) for t in trades)))
        self.assertTrue(all((t['time'] in ('09:30', '15:00') for t in trades)))
        self.assertLessEqual(max((r['positions'] for r in curve)), 5)
        self.assertTrue(all((r['cash'] >= 0 for r in curve)))
        self.assertAlmostEqual(stat['gross_return_same_positions'] - stat['fee_drag'] - stat['slippage_drag'], stat['return_value'])

    def test_missing_buyback_does_not_create_a_fictitious_round_trip(self):
        _, px, _, _, _, _, _, _ = fixture()
        px.raw['close'][2, :] = np.nan
        stat, _, trades = self.run_account('open_close_t', px=px)
        self.assertGreater(stat['t_failed_buybacks'], 0)
        self.assertFalse(any((t['reason'] == 't_buyback' and t['date'] == '2025-07-03' for t in trades)))

    def test_risk_exit_respects_blocked_sells(self):
        _, px, _, _, _, _, _, _ = fixture()
        px.exit[2, :] = False
        px.open[2, :] = 9.0
        prob = np.asarray([0.0, 0.9, 0.9, 0.9, 0.9])
        stat, curve, trades = self.run_account('market_cash', px=px, prob=prob)
        self.assertGreater(stat['blocked_exits'], 0)
        self.assertEqual(curve[1]['positions'], 5)
        self.assertFalse(any((t['date'] == '2025-07-03' and t['side'] == 'sell' for t in trades)))
        self.assertEqual(curve[2]['positions'], 0)

    def test_cost_gate_can_stay_entirely_in_cash(self):
        panel, px, days, pred, aux, prob, context, morning = fixture()
        aux[:] = -0.01
        stat, _, trades = analysis.strategy_backtest(pred, aux, prob, panel, px, days, rule='cost_cash', context=context)
        self.assertEqual(stat['return_value'], 0.0)
        self.assertEqual(trades, [])

class CausalityTests(unittest.TestCase):

    def test_cash_state_requires_two_recovery_signals(self):
        *_, context, _ = fixture()
        state, recovery = analysis.market_budget(0.9, 0, context, 1.0, 0)
        self.assertEqual(state, 0.0)
        state, recovery = analysis.market_budget(0.25, 1, context, state, recovery)
        self.assertEqual(state, 0.0)
        state, recovery = analysis.market_budget(0.1, 2, context, state, recovery)
        self.assertEqual((state, recovery), (0.0, 1))
        state, recovery = analysis.market_budget(0.1, 3, context, state, recovery)
        self.assertEqual((state, recovery), (1.0, 0))

    def test_all_fold_label_endpoints_precede_next_block(self):
        days = pd.bdate_range('2018-01-01', '2026-10-23').strftime('%Y-%m-%d').to_numpy(dtype=str)
        for quarter in model.LAYOUT_QUARTERS:
            for split in model.splits(days, quarter=quarter):
                self.assertLess(split['train'][-1] + 21, split['valid'][0])
                self.assertLess(split['valid'][-1] + 21, split['test'][0])

    def test_market_risk_fit_ignores_future_labels(self):
        rng = np.random.default_rng(1)
        panel = SimpleNamespace(M=rng.normal(size=(300, 3)).astype(np.float32), market_features=['a', 'b', 'c'], days=np.asarray([str(i) for i in range(300)]), Y={k: np.full((300, 4), 0.001) for k in ('label_ret_1d', 'label_ret_5d')})
        panel.Y['label_ret_1d'][::5] = -0.03
        panel.Y['label_ret_5d'][::5] = -0.05
        first = model.fit_market_risk(panel, np.arange(200))
        for value in panel.Y.values():
            value[200:] = 100.0
        second = model.fit_market_risk(panel, np.arange(200))
        np.testing.assert_array_equal(first['coef'], second['coef'])
        self.assertEqual(first['intercept'], second['intercept'])
if __name__ == '__main__':
    unittest.main()
