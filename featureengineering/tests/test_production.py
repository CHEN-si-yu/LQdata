"""Production regressions: real Engine writes, pool execution, resume and label maturity."""
import contextlib
import io
import json
import os
import fcntl
import signal
import time
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from fea.config import Config
from fea.dates import series_to_int
from fea.manifest import Manifest
from fea.production import ProductionEngine, plan_factor, run, select_specs, _ensure_cache, _execute
from fea.spec import FactorSpec, REGISTRY
from fea import store

PROJECT = Path(__file__).resolve().parents[1]


def stock(ctx):
    values = np.arange(ctx.panel.T, dtype=float)[:, None] + np.arange(ctx.panel.C)[None, :]*3
    return ctx.roll_mean(values, 3, min_count=1)


def changed_stock(ctx):
    return stock(ctx) * 1.5


def child(ctx):
    return ctx.load_factor('test_prod_stock') * 2


def market(ctx):
    return np.arange(ctx.panel.T, dtype=float)


def label(ctx):
    values = np.arange(ctx.panel.T, dtype=float)[:, None] + np.arange(ctx.panel.C)[None, :]*3 + 100
    result = ctx.shift(values, -2) / values - 1
    actual = ctx.dataset('stock_daily', columns=['trade_date'])
    last = int(series_to_int(actual.trade_date).max())
    dates = ctx.panel.dates
    for i in range(len(dates)):
        if i+2 >= len(dates) or dates[i+2] > last:
            result[i] = np.nan
    return result


def failure(ctx):
    raise RuntimeError('intentional production failure')


def killed_worker(ctx):
    os.kill(os.getpid(), signal.SIGKILL)


def sparse_warning_stock(ctx):
    with warnings.catch_warnings(record=True) as observed:
        empty = np.full((2, 2), np.nan)
        mean = np.nanmean(empty, axis=1)
        std = np.nanstd(empty, axis=1, ddof=1)
        warnings.warn('unexpected numerical warning', RuntimeWarning)
        warnings.warn('Mean of empty slice', UserWarning)
    messages = [(str(w.message), w.category) for w in observed]
    if messages != [('unexpected numerical warning', RuntimeWarning),
                    ('Mean of empty slice', UserWarning)]:
        raise AssertionError(f'warning filter is too broad or missing: {messages}')
    if not (np.isnan(mean).all() and np.isnan(std).all()):
        raise AssertionError('empty-window NaN semantics changed')
    return stock(ctx)


class ProductionTests(unittest.TestCase):
    def setUp(self):
        temporary = PROJECT / 'artifacts/tests/tmp'
        temporary.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary)
        self.root = Path(self.temp.name)
        self.up = self.root / 'upstream'
        days = pd.bdate_range('2025-11-01', '2026-01-16').strftime('%Y-%m-%d').tolist()
        self.days = np.array([int(d.replace('-', '')) for d in days], dtype=np.int32)
        self.write('basic_calendar', pd.DataFrame({'date': days, 'is_open': 1}))
        self.write('stock_list', pd.DataFrame({'stock_code': ['000001.SZ', '600000.SH'],
                                             'list_date': ['2010-01-01']*2, 'delist_date': [None]*2}))
        self.cfg = Config({'paths': {'upstream': str(self.up)}, 'default_start': '2025-12-01',
                           'universe': {'main_board_prefixes': ['000', '600'], 'exclude_st': False},
                           'rank': {'winsor': [0.01, 0.99], 'min_cross_section': 1},
                           'compute': {'kernel_version': 1, 'revision_days': 10},
                           'resources': {'default_jobs': 2, 'max_jobs': 4}}, self.root)
        self.specs = [FactorSpec('test_prod_stock', 'test', fn=stock, warmup_days=12),
                      FactorSpec('test_prod_child', 'test', fn=child, deps=('test_prod_stock',), warmup_days=12),
                      FactorSpec('test_prod_market', 'test', fn=market, is_market=True, warmup_days=12),
                      FactorSpec('test_prod_label', 'test', fn=label, deps=('stock_daily',),
                                 is_label=True, forward_days=2, warmup_days=12)]
        self.original = {s.name: REGISTRY.get(s.name) for s in self.specs}
        REGISTRY.update({s.name: s for s in self.specs})
        self.source_until(20260109)

    def tearDown(self):
        for n, old in self.original.items():
            if old is None:
                REGISTRY.pop(n, None)
            else:
                REGISTRY[n] = old
        self.temp.cleanup()

    def write(self, name, frame, year=None):
        path = self.up / name
        if year:
            path /= f'year={year}'
        path.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path/'data.parquet', index=False)

    def source_until(self, end):
        dates = self.days[self.days <= end]
        for year in np.unique(dates//10000):
            ds = dates[dates//10000 == year]
            self.write('stock_daily', pd.DataFrame({'trade_date': [f'{d//10000:04}-{d//100%100:02}-{d%100:02}' for d in ds]}), int(year))

    def invoke(self, cfg=None, specs=None, jobs=2, rebuild=False, start=None, end=None, dry=False, refresh=False):
        args = SimpleNamespace(factors=[], group=None, jobs=jobs, rebuild=rebuild, refresh=refresh,
                               start=start, end=end, progress='off', dry_run=dry)
        with patch('fea.production.select_specs', return_value=specs or self.specs), contextlib.redirect_stdout(io.StringIO()):
            return run(args, cfg or self.cfg)

    def outputs(self, cfg):
        return {(s.name, y): pd.read_parquet(store.year_path(cfg.factor_root(s), s.name, y))
                for s in self.specs for y in store.factor_years(cfg.factor_root(s), s.name)}

    def test_explicit_jobs_reach_pool_above_project_and_cpu_limits(self):
        from concurrent.futures import ProcessPoolExecutor
        with patch('fea.resources.available_cpus', return_value=1), patch(
                'concurrent.futures.ProcessPoolExecutor', wraps=ProcessPoolExecutor) as pool, patch(
                'fea.production.MemoryAdmission.read', return_value=(100*2**30, 100*2**30)):
            self.assertEqual(self.invoke(jobs=5), 0)  # project max_jobs is 4
        self.assertEqual(pool.call_args.kwargs['max_workers'], 5)
        self.assertEqual(len(self.outputs(self.cfg)), 8)

    def test_automatic_jobs_keep_memory_protection(self):
        with patch('fea.production.MemoryAdmission.read', return_value=(100*2**30, 100*2**30)):
            with self.assertRaisesRegex(MemoryError, '可用内存不足'):
                self.invoke(jobs=0)
        self.assertFalse(self.outputs(self.cfg))

    def test_workers_ignore_only_expected_empty_window_warnings(self):
        spec = self.specs[0]
        self.assertEqual(self.invoke(specs=[spec], jobs=1), 0)
        expected = self.outputs(self.cfg)
        warning_cfg = self.cfg.sandbox(self.root / 'warning_output')
        with patch.object(spec, 'fn', sparse_warning_stock):
            self.assertEqual(self.invoke(cfg=warning_cfg, specs=[spec], jobs=2), 0)
        actual = self.outputs(warning_cfg)
        for key, frame in expected.items():
            pd.testing.assert_frame_equal(frame, actual[key], check_exact=True)

    def test_full_incremental_serial_parallel_and_same_input_skip(self):
        self.assertEqual(self.invoke(jobs=1), 0)
        before = self.outputs(self.cfg)
        old_paths = [store.year_path(self.cfg.factor_root(s), s.name, y) for s in self.specs
                     for y in store.factor_years(self.cfg.factor_root(s), s.name)]
        mtimes = {p: p.stat().st_mtime_ns for p in old_paths}
        with patch('fea.production._execute', side_effect=AssertionError('should skip computation')):
            self.assertEqual(self.invoke(), 0)
        self.assertEqual(mtimes, {p: p.stat().st_mtime_ns for p in old_paths})
        self.source_until(20260116)
        self.assertEqual(self.invoke(), 0)
        full = Config(json.loads(json.dumps(self.cfg.raw)), self.root / 'full')
        self.assertEqual(self.invoke(cfg=full, rebuild=True, jobs=2), 0)
        incremental, rebuilt = self.outputs(self.cfg), self.outputs(full)
        self.assertEqual(set(incremental), set(rebuilt))
        for key in incremental:
            pd.testing.assert_frame_equal(incremental[key], rebuilt[key], check_exact=True)
        for (name, year), previous in before.items():
            if name == 'test_prod_label':
                continue
            merged = incremental[(name, year)]
            existing = merged[merged.trade_date.isin(previous.trade_date)].reset_index(drop=True)
            pd.testing.assert_frame_equal(existing, previous, check_exact=True)
        # Cross-year label windows mature when new source dates arrive.
        label_frame = incremental[('test_prod_label', 2026)]
        self.assertTrue(label_frame.loc[label_frame.trade_date == '2026-01-09', 'value'].notna().all())
        self.assertTrue(label_frame.loc[label_frame.trade_date == '2026-01-16', 'value'].isna().all())

    def test_missing_partition_recovery_and_new_factor_skip(self):
        self.assertEqual(self.invoke(), 0)
        original = self.outputs(self.cfg)
        spec = self.specs[0]
        store.year_path(self.cfg.factor_root(spec), spec.name, 2025).unlink()
        self.assertEqual(self.invoke(), 0)
        restored = self.outputs(self.cfg)
        for key in restored:
            pd.testing.assert_frame_equal(restored[key], original[key], check_exact=True)
        summary = json.loads((self.cfg.state_dir/'production_last_run.json').read_text())
        self.assertEqual(summary['skipped'], 2)  # parent repair also recomputes its child

    def test_failure_blocks_child_and_resume_retains_pending_work(self):
        parent = self.specs[0]
        old_fn = parent.fn
        parent.fn = failure
        self.assertEqual(self.invoke(specs=self.specs[:2]), 1)
        state = Manifest.load(self.cfg.state_dir, parent.name)
        self.assertFalse(state.coverage)
        self.assertEqual(state.last_run['pending_years'], [2025, 2026])
        parent.fn = old_fn
        self.assertEqual(self.invoke(specs=self.specs[:2]), 0)
        self.assertFalse(Manifest.load(self.cfg.state_dir, parent.name).last_run['pending_years'])

    def test_refresh_and_dated_rebuild_preserve_later_data(self):
        self.assertEqual(self.invoke(), 0)
        old = self.outputs(self.cfg)
        self.assertEqual(self.invoke(start='2025-12-08', end='2025-12-12', refresh=True), 0)
        for key, df in self.outputs(self.cfg).items():
            pd.testing.assert_frame_equal(df, old[key], check_exact=True)
        self.assertEqual(self.invoke(start='2025-12-01', end='2025-12-12', rebuild=True), 0)
        for key, df in self.outputs(self.cfg).items():
            pd.testing.assert_frame_equal(df, old[key], check_exact=True)
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(json.loads((self.cfg.state_dir/'production_last_run.json').read_text())['tasks'], 0)

    def test_dry_run_does_not_create_manifests(self):
        self.assertEqual(self.invoke(dry=True), 0)
        self.assertFalse(list(self.cfg.state_dir.glob('*.json')))

    def test_rebuild_is_real_computation_and_no_fingerprints(self):
        with patch('fea.engine.numeric_env_fingerprint', side_effect=AssertionError('fingerprint')),\
             patch('fea.engine.partition_content_sha256', side_effect=AssertionError('output digest')),\
             patch('fea.engine.frozen_fingerprint', side_effect=AssertionError('pool digest')):
            self.assertEqual(self.invoke(), 0)
            self.assertEqual(self.invoke(rebuild=True), 0)
        report = json.loads((self.cfg.state_dir/'production_last_run.json').read_text())
        self.assertEqual(report['tasks'], 8)
        self.assertEqual(report['skipped'], 0)
        man = Manifest.load(self.cfg.state_dir, self.specs[0].name)
        self.assertFalse(man.recipe)
        self.assertFalse(man.input_watermark)

    def test_narrow_child_request_generates_parent_warmup(self):
        self.assertEqual(self.invoke(specs=self.specs[:2], start='2026-01-08'), 0)
        self.assertIn(2025, store.factor_years(self.cfg.factors_dir, self.specs[0].name))
        self.assertEqual(store.factor_years(self.cfg.factors_dir, self.specs[1].name), [2026])

    def test_aligned_late_start_fills_prefix_without_computing(self):
        spec = self.specs[0]
        spec.start = '2026-01-06'
        self.assertEqual(self.invoke(specs=[spec]), 0)
        df = pd.read_parquet(store.year_path(self.cfg.factors_dir, spec.name, 2025))
        self.assertTrue(df.value.isna().all())
        df = pd.read_parquet(store.year_path(self.cfg.factors_dir, spec.name, 2026))
        self.assertTrue(df.loc[df.trade_date < '2026-01-06', 'value'].isna().all())
        self.assertTrue(df.loc[df.trade_date >= '2026-01-06', 'value'].notna().all())

    def test_new_registration_only_computes_the_new_factor(self):
        self.assertEqual(self.invoke(), 0)
        old = self.outputs(self.cfg)
        paths = [store.year_path(self.cfg.factor_root(s), s.name, y)
                 for s in self.specs for y in store.factor_years(self.cfg.factor_root(s), s.name)]
        mtimes = {p: p.stat().st_mtime_ns for p in paths}
        spec = FactorSpec('test_prod_new', 'test', fn=child, deps=('test_prod_stock',), warmup_days=12)
        REGISTRY[spec.name] = spec
        try:
            from fea.production import select_specs
            with patch('fea.production.registered_specs', return_value=self.specs+[spec]):
                selected = select_specs(self.cfg)
            self.assertEqual(self.invoke(specs=selected), 0)
            report = json.loads((self.cfg.state_dir/'production_last_run.json').read_text())
            self.assertEqual(report['skipped'], 4)
            self.assertEqual(report['tasks'], 2)
            self.assertEqual(mtimes, {p: p.stat().st_mtime_ns for p in paths})
            for y in (2025, 2026):
                new = pd.read_parquet(store.year_path(self.cfg.factors_dir, spec.name, y))
                pd.testing.assert_frame_equal(new, old[('test_prod_child', y)], check_exact=True)
        finally:
            REGISTRY.pop(spec.name, None)

    def test_interruption_after_a_checkpoint_resumes_remaining_partitions(self):
        from fea.production import commit_task
        def interrupt(*args, **kwargs):
            commit_task(*args, **kwargs)
            raise KeyboardInterrupt
        with patch('fea.production.commit_task', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke(jobs=1)
        parent = Manifest.load(self.cfg.state_dir, self.specs[0].name)
        self.assertEqual(len(parent.partitions), 1)
        completed_year = int(next(iter(parent.partitions)))
        path = store.year_path(self.cfg.factors_dir, parent.name, completed_year)
        stamp = path.stat().st_mtime_ns
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        report = json.loads((self.cfg.state_dir/'production_last_run.json').read_text())
        self.assertEqual(report['tasks'], 7)
        full = Config(json.loads(json.dumps(self.cfg.raw)), self.root/'reference')
        self.assertEqual(self.invoke(cfg=full, rebuild=True), 0)
        for key, frame in self.outputs(full).items():
            pd.testing.assert_frame_equal(frame, self.outputs(self.cfg)[key], check_exact=True)

    def test_dead_worker_fails_promptly_instead_of_hanging(self):
        self.specs[0].fn = killed_worker
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, '异常退出'):
            self.invoke(specs=self.specs[:1], jobs=1)
        self.assertLess(time.monotonic()-started, 10)
        self.assertFalse(Manifest.load(self.cfg.state_dir, self.specs[0].name).coverage)
        self.specs[0].fn = stock
        self.assertEqual(self.invoke(specs=self.specs[:1]), 0)

    def test_exclusive_lock_rejects_a_second_writer(self):
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        with (self.cfg.state_dir/'daily.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.invoke(), 2)
        self.assertEqual(self.invoke(), 0)

    def test_damaged_partition_is_repaired_without_changing_other_outputs(self):
        self.assertEqual(self.invoke(), 0)
        original = self.outputs(self.cfg)
        path = store.year_path(self.cfg.factor_root(self.specs[2]), self.specs[2].name, 2025)
        path.write_bytes(b'interrupted parquet write')
        self.assertEqual(self.invoke(), 0)
        for key, frame in self.outputs(self.cfg).items():
            pd.testing.assert_frame_equal(frame, original[key], check_exact=True)
        report = json.loads((self.cfg.state_dir/'production_last_run.json').read_text())
        self.assertEqual(report['tasks'], 1)

    def test_output_without_manifest_recovers_from_the_date_column(self):
        self.assertEqual(self.invoke(), 0)
        spec = self.specs[2]
        (self.cfg.state_dir/(spec.name+'.json')).unlink()
        with patch('fea.production._execute', side_effect=AssertionError('should recover existing output')):
            self.assertEqual(self.invoke(specs=[spec]), 0)
        self.assertEqual(len(Manifest.load(self.cfg.state_dir, spec.name).partitions), 2)

    def test_tiny_minute_aggregate_reuses_cache_and_rebuilds_on_source_update(self):
        from fea.production import _ensure_cache, _cache_ready
        from fea.derived import DerivedCache
        from fea.upstream import Upstream
        class TinyMinute(DerivedCache):
            name = 'market_morning'
            def build_year(inner, year, up):
                return up.read('stock_history_5min', years=(year, year), use_cache=False)
        frame = pd.DataFrame({'trade_date': ['2026-01-02'], 'value': [1.]})
        self.write('stock_history_5min', frame, 2026)
        layer = TinyMinute(Upstream(self.up, self.cfg.root), self.cfg, np.array(['000001.SZ']))
        _ensure_cache(layer, 2026, 2026)
        path = layer.root/'year=2026/data.parquet'
        self.assertTrue(_cache_ready(layer, 2026, 2026))
        stamp = path.stat().st_mtime_ns
        _ensure_cache(layer, 2026, 2026)
        self.assertEqual(stamp, path.stat().st_mtime_ns)
        frame.value = 2.
        self.write('stock_history_5min', frame, 2026)
        source = self.up/'stock_history_5min/year=2026/data.parquet'
        os.utime(source, ns=(stamp+1000000, stamp+1000000))
        self.assertFalse(_cache_ready(layer, 2026, 2026))
        next_run = TinyMinute(Upstream(self.up, self.cfg.root), self.cfg, np.array(['000001.SZ']))
        _ensure_cache(next_run, 2026, 2026)
        self.assertEqual(pd.read_parquet(path).value.iloc[0], 2.)

    def test_output_matches_the_original_engine_including_parquet_bytes(self):
        from fea.engine import Engine
        self.assertEqual(self.invoke(), 0)
        cfg = Config(json.loads(json.dumps(self.cfg.raw)), self.root/'legacy')
        engine = Engine(cfg)
        engine._run_end, engine._end_str = 20260109, '2026-01-09'
        engine._deriv_fields_run = frozenset()
        for spec in self.specs:
            engine.factor_io()
            days = engine.cal.between(20251201, 20260109)
            for year in np.unique(days//10000):
                engine.run_year(spec, int(year), days[days//10000 == year])
            engine._factor_io = None
        for key, expected in self.outputs(cfg).items():
            pd.testing.assert_frame_equal(expected, self.outputs(self.cfg)[key], check_exact=True)
            name, year = key
            spec = REGISTRY[name]
            original = store.year_path(cfg.factor_root(spec), name, year).read_bytes()
            new = store.year_path(self.cfg.factor_root(spec), name, year).read_bytes()
            self.assertEqual(original, new)

    def test_interrupted_rebuild_does_not_accept_old_files_as_new_results(self):
        from fea.production import commit_task
        self.assertEqual(self.invoke(), 0)
        self.specs[0].fn = changed_stock
        def interrupt(*args, **kwargs):
            commit_task(*args, **kwargs)
            raise KeyboardInterrupt
        with patch('fea.production.commit_task', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke(jobs=1, rebuild=True)
        state = Manifest.load(self.cfg.state_dir, self.specs[0].name)
        self.assertEqual(state.last_run['pending_years'], [2026])
        self.assertEqual(self.invoke(), 0)
        cfg = Config(json.loads(json.dumps(self.cfg.raw)), self.root/'new_reference')
        self.assertEqual(self.invoke(cfg=cfg, rebuild=True), 0)
        for key, expected in self.outputs(cfg).items():
            pd.testing.assert_frame_equal(expected, self.outputs(self.cfg)[key], check_exact=True)

    def test_progress_rows_align_and_failed_state_is_retained(self):
        from fea.progress import ProductionProgress
        specs = self.specs[:2]
        targets = {s.name: np.arange(5) for s in specs}
        plans = {s.name: {2025: np.arange(2), 2026: np.arange(3)} for s in specs}
        stream = io.StringIO()
        progress = ProductionProgress(specs, targets, plans, 'plain', stream)
        progress.start(specs[0].name, 2025)
        progress.start(specs[0].name, 2026)
        progress.finish(specs[0].name, 2025, 2, 'failure')
        progress.finish(specs[0].name, 2026, 3, rows=6)
        self.assertEqual(progress.states[specs[0].name], 'failed')
        rows = [line for line in stream.getvalue().splitlines() if '(test)' in line]
        self.assertEqual(rows[0].index('(test)'), rows[1].index('(test)'))
        self.assertRegex(stream.getvalue(), r'\[\d\d:\d\d:\d\d\]')
        self.assertIn('写入', stream.getvalue())

    def test_daily_update_into_a_new_year_refreshes_previous_year_labels(self):
        (self.up/'stock_daily/year=2026/data.parquet').unlink()
        self.source_until(20251231)
        self.assertEqual(self.invoke(), 0)
        label_spec = self.specs[3]
        old = pd.read_parquet(store.year_path(self.cfg.factors_dir, label_spec.name, 2025))
        self.assertTrue(old.loc[old.trade_date == '2025-12-31', 'value'].isna().all())
        previous = {s.name: store.year_path(self.cfg.factor_root(s), s.name, 2025).stat().st_mtime_ns
                    for s in self.specs[:3]}
        self.source_until(20260105)
        self.assertEqual(self.invoke(), 0)
        current = pd.read_parquet(store.year_path(self.cfg.factors_dir, label_spec.name, 2025))
        self.assertTrue(current.loc[current.trade_date == '2025-12-31', 'value'].notna().all())
        for s in self.specs[:3]:
            self.assertEqual(previous[s.name], store.year_path(self.cfg.factor_root(s), s.name, 2025).stat().st_mtime_ns)
            self.assertTrue(store.year_path(self.cfg.factor_root(s), s.name, 2026).exists())
        self.assertEqual(json.loads((self.cfg.state_dir/'production_last_run.json').read_text())['tasks'], 5)
        cfg = Config(json.loads(json.dumps(self.cfg.raw)), self.root/'year_boundary_reference')
        self.assertEqual(self.invoke(cfg=cfg, rebuild=True), 0)
        for key, expected in self.outputs(cfg).items():
            pd.testing.assert_frame_equal(expected, self.outputs(self.cfg)[key], check_exact=True)


class ResourceSchedulingTests(unittest.TestCase):
    def test_manual_mode_dispatches_40_while_automatic_mode_keeps_reservations(self):
        from concurrent.futures import Future, wait
        spec = SimpleNamespace(name='test', deps=(), warmup_days=0)
        engine = SimpleNamespace(cfg=SimpleNamespace(raw={'resources': {'memory_budget_gib': 90}}))
        plans = {'test': {year: np.array([year*10000+101]) for year in range(1980, 2020)}}
        for enforce_limits, expected_first_wave in ((False, 40), (True, 25)):
            with self.subTest(enforce_limits=enforce_limits):
                pool, progress = MagicMock(), MagicMock()
                manifest = SimpleNamespace(partitions={})
                waves = []

                def submit(fn, task):
                    future = Future()
                    future.set_result({'year': task[1], 'rows': 0, 'partition_rows': 0})
                    return future

                def observe_wait(futures, **kwargs):
                    waves.append(len(futures))
                    return wait(futures, **kwargs)

                pool.submit.side_effect = submit
                with patch('concurrent.futures.ProcessPoolExecutor', return_value=pool), patch(
                        'concurrent.futures.wait', side_effect=observe_wait), patch(
                        'fea.production.MemoryAdmission.read', return_value=(0, 0)), patch(
                        'fea.production.commit_task'):
                    self.assertEqual(_execute(engine, [spec], {'test': manifest}, plans, 20200101,
                                              40, progress, enforce_limits=enforce_limits), ({}, set()))
                self.assertEqual(waves[0], expected_first_wave)
                self.assertEqual(pool.submit.call_count, 40)


if __name__ == '__main__':
    unittest.main()
