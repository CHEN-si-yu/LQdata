"""市场条件映射与慢跌策略的时间、容量合同。"""
import importlib.util
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experts = load('test_experts_model', ROOT / 'experiments/V261/model.py')
study = load('test_paired_strategy', ROOT / 'experiments/_archive/20261024_20261025/scripts/strategy_20261024.py')
with patch.dict(sys.modules, strategy_20261024=study):
    mean_study = load('test_mean_strategy', ROOT / 'experiments/_archive/20261024_20261025/scripts/strategy_mean_20261024.py')
with patch.dict(sys.modules, strategy_20261024=study, strategy_mean_20261024=mean_study):
    alpha_study = load('test_alpha_strategy', ROOT / 'experiments/_archive/20261024_20261025/scripts/strategy_alpha_20261025.py')


class MarketConditioningTests(unittest.TestCase):
    def test_overlapping_alpha_calibration_dates_count_once_and_ignore_future(self):
        _, analysis = study.load_analysis()
        t, n = 130, 128
        z = np.linspace(-1., 1., n)
        z = (z - z.mean()) / z.std(ddof=1)
        panel = SimpleNamespace(days=np.asarray([str(i) for i in range(t)]), codes=np.arange(n),
            Y={'label_ret_1d':np.broadcast_to(.01 * z, (t, n)).copy()})
        splits = [dict(valid=np.arange(30 + 10 * f, 60 + 10 * f), test=np.arange(115, 125)) for f in range(4)]
        fake_model = SimpleNamespace(LAYOUT_QUARTERS=['2025Q3'], splits=lambda *args, **kwargs:splits)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for f in range(1, 5):
                folder = root / f'experiments/V999/model_train/2025Q3/fold{f}'
                folder.mkdir(parents=True)
                values = np.zeros((30, n, 3))
                values[:, :, 0] = f * z + f
                np.save(folder / 'valid_predictions.npy', values)
            with patch.object(alpha_study, 'ROOT', root):
                first = alpha_study.calibration('V999', panel, fake_model, analysis)
                panel.Y['label_ret_1d'][90:] += 100
                second = alpha_study.calibration('V999', panel, fake_model, analysis)
        self.assertEqual(first, second)
        self.assertEqual(first['2025Q3']['unique_days'], 60)
        self.assertAlmostEqual(first['2025Q3']['gamma'], .01, places=10)

    def test_market_mean_fit_is_unchanged_by_future_market_and_labels(self):
        rng = np.random.default_rng(91)
        panel = SimpleNamespace(M=rng.normal(size=(100, 3)).astype(np.float32),
            Y={'label_ret_1d':rng.normal(scale=.01, size=(100, 12)).astype(np.float32)},
            days=np.asarray([str(i) for i in range(100)]))
        first = mean_study.fit_market_mean(panel, np.arange(60))
        panel.M[60:] *= -100
        panel.Y['label_ret_1d'][60:] += 100
        second = mean_study.fit_market_mean(panel, np.arange(60))
        self.assertEqual(first, second)

    def test_one_day_target_keeps_distinct_one_and_five_day_report_columns(self):
        with patch.dict(sys.modules, model=experts):
            analysis = load('test_expert_analysis', ROOT / 'experiments/V261/analysis.py')
        metrics = dict(IC_1d=.01, IC_5d=.05, RankIC_1d=.02, RankIC_5d=.06,
            val_wei=.03, val_return_proxy=.001)
        records = [dict(quarter='2025Q3', fold=f, best_validation=metrics) for f in range(1, 5)]
        with patch.object(analysis, 'quarters', return_value=['2025Q3']), \
             patch.object(analysis, 'axis', return_value=(np.asarray(['2025-07-01']),)), \
             patch.object(analysis, 'fold_valid_days', return_value={('2025Q3', f):63 for f in range(1, 5)}):
            rows = analysis.training_validation_rows(records)
        self.assertEqual(rows[0]['IC_1d'], .01)
        self.assertEqual(rows[0]['IC_5d'], .05)
        self.assertAlmostEqual(rows[0]['total_return'], 100 * (1.001 ** 63 - 1))

    def test_future_changes_do_not_change_train_partition_or_experts(self):
        rng = np.random.default_rng(143)
        t, n, width = 80, 16, 3
        x = rng.normal(size=(t, n, width)).astype(np.float32)
        market = rng.normal(scale=.15, size=(t, 2)).astype(np.float32)
        market[:, 0] += np.where(np.arange(t) % 2 == 0, 3., -3.)
        y = x[:, :, 0] * np.sign(market[:, 0, None])
        panel = SimpleNamespace(X=x, M=market, Y={'label_ret_1d':y},
            features=['a', 'b', 'c'], market_dim=2, days=np.asarray([str(i) for i in range(t)]))
        panel.mask = lambda d:np.ones(n, bool)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = (root / 'model_train/2025Q3/fold1', root / 'model_train/2025Q3/fold2')
            a.mkdir(parents=True)
            b.mkdir(parents=True)
            with patch.object(experts, 'UNIT_ROOT', root), patch.dict(experts.RECIPE, technical_indices=[0]):
                first, _ = experts.fit_relative_experts(panel, np.arange(50), experts.torch.device('cpu'), a, 'selector')
                panel.X[50:] += 100
                panel.M[50:] -= 100
                panel.Y['label_ret_1d'][50:] *= -100
                second, _ = experts.fit_relative_experts(panel, np.arange(50), experts.torch.device('cpu'), b, 'selector')
            for key, value in first.state_dict().items():
                np.testing.assert_array_equal(value.numpy(), second.state_dict()[key].numpy())
            sample = experts.torch.from_numpy(x[0, :, None, :])
            positive = first(sample, experts.torch.tensor([3., 0.]))[:, 0].detach().numpy()
            negative = first(sample, experts.torch.tensor([-3., 0.]))[:, 0].detach().numpy()
            self.assertLess(float(np.corrcoef(positive, negative)[0, 1]), -.9)

    def test_slow_veto_retains_two_signal_recovery(self):
        _, analysis = study.load_analysis()
        context = dict(mkt_idx_csi300_ma20_gap=np.asarray([-.01, .01, .01]),
            mkt_above_ma20_ratio=np.asarray([.4, .6, .6]),
            mkt_idx_csi300_ret1=np.zeros(3), mkt_limit_down_ratio=np.zeros(3))
        probability = np.full(3, .1)
        actual = study.budget_path(np.arange(3), probability, context, 'slow_trend_cash', analysis.market_budget)
        np.testing.assert_array_equal(actual, [0., 0., 1.])
        context['mkt_idx_csi300_ma20_gap'][2] = -1
        context['mkt_above_ma20_ratio'][2] = 0
        changed = study.budget_path(np.arange(3), probability, context, 'slow_trend_cash', analysis.market_budget)
        np.testing.assert_array_equal(changed[:2], actual[:2])

    def test_partial_budget_reallocates_capacity_instead_of_scaling_old_holdings(self):
        _, analysis = study.load_analysis()
        panel = SimpleNamespace(Y={'label_ret_1d':np.asarray([[.1, -.1]])}, amount=np.asarray([[9e8, 9e8]]))
        score = np.asarray([[2., 1.]])
        full = study.large_daily(score, panel, np.asarray([0]), np.asarray([1.]), analysis)
        reduced = study.large_daily(score, panel, np.asarray([0]), np.asarray([.4]), analysis)
        self.assertAlmostEqual(float(full[0]), .02)
        self.assertAlmostEqual(float(reduced[0]), .04)
        self.assertNotAlmostEqual(float(reduced[0]), float(full[0]) * .4)

    def test_literal_parity_when_one_float32_return_fills_full_budget(self):
        _, analysis = study.load_analysis()
        y = np.asarray([[-.050854702, .02]], np.float32)
        panel = SimpleNamespace(Y={'label_ret_1d':y, 'label_ret_5d':y}, amount=np.asarray([[2e9, 9e8]]))
        score = np.asarray([[2., 1.]])
        reference, _, _ = analysis.top_portfolio_daily(score, panel, np.asarray([0]))
        actual = study.large_daily(score, panel, np.asarray([0]), np.asarray([1.]), analysis)
        np.testing.assert_allclose(actual * 100, reference, rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
