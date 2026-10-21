"""日报命名、输入投影、融合与重复执行的关键合同。"""
import importlib.util
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / 'releases' / 'main.py'
spec = importlib.util.spec_from_file_location('release_daily', SCRIPT)
daily = importlib.util.module_from_spec(spec)
spec.loader.exec_module(daily)


def entry(day):
    row = dict(rank=1, code='000001.SZ', name='示例', score=.8)
    return dict(day=day, units=['V1'], next_day=None, common_count=1,
                fusion=[dict(row, ranks={'V1': 1})], models={'V1': [row]},
                quarters={'V1': '2026Q4'}, direction_flips={'V1': 0}, snapshot_sha256='fixture',
                semantics='zscore_win1_99_v1')


class DailyContractTests(unittest.TestCase):
    def test_daily_names_follow_execution_date(self):
        directory = Path('/tmp/reports')
        self.assertEqual(daily.daily_path(directory, datetime(2026, 10, 7)),
                         directory / 'DAILY-20261007.md')
        self.assertNotEqual(daily.daily_path(directory, datetime(2026, 10, 7)),
                            daily.daily_path(directory, datetime(2026, 10, 8)))

    def test_market_order_comes_from_training_record(self):
        module = SimpleNamespace(RECIPE={'feature_names': ['b', 'a'], 'output_horizons': [5, 1]})
        ck = dict(model={}, features=['b', 'a'], ridge=None, input_dim=2, market_dim=2, horizons=[5, 1])
        info = dict(feature_columns=['b', 'a'], market_columns=['m2', 'm1'])
        meta = dict(columns={'features': ['extra', 'a', 'b']},
                    market_factors={'columns': ['m1', 'extra_market', 'm2']})
        self.assertEqual(daily.validate_columns(Path('V1'), ck, info, module, meta),
                         (['b', 'a'], ['m2', 'm1']))
        with self.assertRaisesRegex(ValueError, '缺少权重必需'):
            daily.validate_columns(Path('V1'), ck, info, module,
                                   dict(meta, columns={'features': ['a']}))
        with self.assertRaisesRegex(ValueError, '顺序不一致'):
            daily.validate_columns(Path('V1'), ck, dict(info, feature_columns=['a', 'b']), module, meta)

    def test_direction_conversion_handles_raw_signed_and_mixed_snapshots(self):
        contract = {'snapshots': {'training': dict(semantics='zscore_win1_99_v1',
                                                   feature_direction={'b': -1, 'a': 1}, market_columns=['m2', 'm1'])}}
        meta = dict(semantics='zscore_win1_99_v1', columns={'direction': {'a': 1, 'b': 1}})
        convert = lambda value: daily.direction_multipliers(Path('V1'), ['b', 'a'], ['m2', 'm1'],
                                                            value, 'training', 'current', contract)
        self.assertEqual(convert(meta), [-1, 1])
        signed = dict(meta, columns={'direction': {'a': 1, 'b': -1}})
        self.assertEqual(convert(signed), [1, 1])
        mixed = dict(meta, columns={'direction': {'a': -1, 'b': 1}})
        self.assertEqual(convert(mixed), [-1, -1])
        with self.assertRaisesRegex(ValueError, '缺少训练方向合同'):
            daily.direction_multipliers(Path('V1'), ['b', 'a'], [], meta, 'unknown', 'current', contract)
        with self.assertRaisesRegex(ValueError, '无法确认训练/当前因子方向'):
            convert(dict(meta, columns={'direction': {'a': 1}}))
        with self.assertRaisesRegex(ValueError, '市场列名/顺序不一致'):
            daily.direction_multipliers(Path('V1'), ['b', 'a'], ['m1', 'm2'], meta,
                                        'training', 'current', contract)

    def test_panel_converts_direction_once_and_preserves_missing_coverage(self):
        class Panel:
            pass
        module = SimpleNamespace(Panel=Panel, UNIT='V1', RECIPE=dict(flow_indices=[0], fund_indices=[1],
                                   technical_indices=[0], market_clip=10.))
        raw = np.array([[[2., 3.], [np.nan, 4.]]], np.float32)
        kwargs = dict(module=module, data=Path('/tmp'), meta={}, days=['2026-09-30'],
                      codes=np.array(['A', 'B']), features=['b', 'a'], market=[],
                      feature_positions={'b': 0, 'a': 1}, shared_market={})
        raw_panel = daily.make_panel(shared_x=raw, multipliers=[-1, 1], **kwargs)
        signed_panel = daily.make_panel(shared_x=raw * [-1, 1], multipliers=[1, 1], **kwargs)
        np.testing.assert_array_equal(raw_panel.X, [[[-2., 3.], [0., 4.]]])
        np.testing.assert_array_equal(raw_panel.X, signed_panel.X)
        np.testing.assert_array_equal(raw_panel.flow_coverage, [[1., 0.]])
        self.assertTrue(np.isnan(raw[0, 1, 0]))
        self.assertEqual(raw[0, 0, 0], 2.)  # 不污染其他模型共用的输入。

    def test_local_frozen_contract_has_every_training_direction(self):
        import json
        root = SCRIPT.parent
        contracts = json.loads((root / 'input_contracts.json').read_text())
        for unit in ['V220', 'V230', 'V240', 'V252']:
            info = json.loads((root / unit / 'model_train/2026Q4/fold1/training_info.json').read_text())
            features = info['feature_columns']
            meta = dict(semantics='zscore_win1_99_v1', columns={'direction': dict.fromkeys(features, 1)})
            signs = daily.direction_multipliers(Path(unit), features, info['market_columns'], meta,
                                                info['recipe']['snapshot_meta_sha256'], 'raw_snapshot', contracts)
            frozen = contracts['snapshots'][info['recipe']['snapshot_meta_sha256']]
            self.assertEqual(signs, [frozen['feature_direction'][name] for name in features])

    def test_read_block_reorders_keys_and_columns_and_rejects_missing_stock(self):
        with TemporaryDirectory() as root:
            path = Path(root) / 'factors/year=2026/data.parquet'
            path.parent.mkdir(parents=True)
            frame = pd.DataFrame(dict(trade_date=['2026-09-30'] * 2, stock_code=['B', 'A'],
                                      a=[20., 10.], b=[2., 1.], extra=[99., 99.]))
            frame.to_parquet(path, index=False)
            values = daily.read_block(Path(root), 'factors', np.array(['2026-09-30']),
                                      np.array(['A', 'B']), ['b', 'a'])
            np.testing.assert_array_equal(values, [[[1., 10.], [2., 20.]]])
            frame.iloc[:1].to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, '轴不一致'):
                daily.read_block(Path(root), 'factors', np.array(['2026-09-30']),
                                 np.array(['A', 'B']), ['b', 'a'])

    def test_four_folds_are_required_for_latest_quarter(self):
        with TemporaryDirectory() as root:
            unit = Path(root) / 'V1'
            for quarter in ['2026Q3', '2026Q4']:
                for k in range(1, 5):
                    fold = unit / 'model_train' / quarter / f'fold{k}'
                    fold.mkdir(parents=True)
                    for name in ['best.pt', 'complete.json', 'training_info.json']:
                        (fold / name).write_text('{}')
            self.assertEqual(daily.discover_checkpoints(unit)[0], '2026Q4')
            (unit / 'model_train/2026Q4/fold4/best.pt').unlink()
            with self.assertRaisesRegex(ValueError, '2026Q4 四折不齐'):
                daily.discover_checkpoints(unit)

    def test_fusion_uses_common_finite_pool_and_average_ties(self):
        scores = {'V1': np.array([9., 1., 1., np.nan]),
                  'V2': np.array([3., 3., 0., 100.])}
        actual, count = daily.fuse_scores(scores)
        self.assertEqual(count, 3)
        np.testing.assert_allclose(actual[:3], [(.833333333333 + .666666666667) / 2,
                                               (.333333333333 + .666666666667) / 2,
                                               (.333333333333 + .166666666667) / 2])
        self.assertTrue(np.isnan(actual[3]))
        np.testing.assert_array_equal(daily.rank_order(np.array([1., 1., np.nan]),
                                                       np.array(['B', 'A', 'C'])), [1, 0])

    def test_merge_is_idempotent_and_preserves_manual_notes(self):
        dates = np.array(['2026-09-29', '2026-09-30'])
        entries = {day: entry(day) for day in dates}
        old = '# 日报\n\n## 2026-09-30\n\n人工备注：观察持仓。\n\n## 2026-09-28\n\n旧窗口\n'
        rendered = daily.merge_markdown(old, entries, dates, dates[-1])
        self.assertIn('人工备注：观察持仓。', rendered)
        self.assertNotIn('## 2026-09-28', rendered)
        self.assertEqual(rendered.count(daily.BEGIN), 2)
        self.assertEqual(daily.merge_markdown(rendered, entries, dates, dates[-1]), rendered)
        self.assertLess(rendered.index('## 2026-09-30'), rendered.index('## 2026-09-29'))

    def test_duplicate_dates_fail_before_writing(self):
        with self.assertRaisesRegex(ValueError, '重复日期'):
            daily.merge_markdown('## 2026-09-30\nA\n## 2026-09-30\nB\n',
                                 {'2026-09-30': entry('2026-09-30')}, ['2026-09-30'], '2026-09-30')

    def test_atomic_write_skips_unchanged_content(self):
        with TemporaryDirectory() as root:
            path = Path(root) / 'DAILY-20261007.md'
            self.assertTrue(daily.atomic_text(path, 'report\n'))
            before = path.stat().st_mtime_ns
            self.assertFalse(daily.atomic_text(path, 'report\n'))
            self.assertEqual(path.stat().st_mtime_ns, before)


if __name__ == '__main__':
    unittest.main()
