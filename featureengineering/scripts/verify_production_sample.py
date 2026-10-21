#!/usr/bin/env python3
"""Read-only real inputs, isolated outputs: compare direct production to the original Engine."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'


def main():
    from fea.config import load, setup_logging
    from fea.production import run, select_specs
    from fea.engine import Engine
    from fea.resources import configure
    from fea.dates import str_to_int
    from fea.manifest import _atomic_json
    from fea import store
    import numpy as np
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factors', nargs='+', default=['momentum_20', 'roe_ttm', 'mkt_up_ratio', 'label_ret_5d',
        'r13gc_turnover_commonality_beta_126_v1_8be87d88402aac70'])
    parser.add_argument('--start', default='2026-09-01')
    parser.add_argument('--end', default='2026-09-04')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--out', default='artifacts/tests/production_parity')
    args = parser.parse_args()
    base = load()
    configure(base.raw.get('resources'))
    root = (base.root / args.out).resolve()
    if not root.is_relative_to(base.root.resolve()) or root == base.root.resolve():
        raise ValueError('test outputs must be inside the project')
    direct = base.sandbox(root/'direct')
    reference = base.sandbox(root/'original')
    setup_logging(cfg=direct)
    specs = select_specs(base, args.factors)
    begin = time.monotonic()
    options = SimpleNamespace(factors=args.factors, group=None, jobs=args.jobs, start=args.start,
                              end=args.end, rebuild=True, refresh=False, dry_run=False, progress='plain')
    if run(options, direct):
        raise RuntimeError('direct production failed')
    direct_seconds = time.monotonic()-begin
    begin = time.monotonic()
    engine = Engine(reference)
    engine._run_end, engine._end_str = str_to_int(args.end), args.end
    for name in ('intraday', 'chips', 'open5', 'closing30', 'chipwall', 'market_morning'):
        engine._layer(name)
    engine.factor_io()
    tasks = []
    for spec in specs:
        days = engine.cal.between(str_to_int(args.start), str_to_int(args.end))
        tasks.extend((spec.name, int(y), days[days//10000 == y]) for y in np.unique(days//10000))
    # This sample has independent factors. Dependency samples are covered by unittest.
    if any(d in {s.name for s in specs} for s in specs for d in s.deps):
        raise ValueError('use independent factors for the real-input parallel reference')
    results = engine.run_parallel(tasks, args.jobs)
    failures = [r for r in results if r.get('error')]
    if failures:
        raise RuntimeError(str(failures))
    original_seconds = time.monotonic()-begin
    partitions = []
    for name, year, _ in tasks:
        spec = next(s for s in specs if s.name == name)
        original = store.year_path(reference.factor_root(spec), name, year)
        new = store.year_path(direct.factor_root(spec), name, year)
        expected, actual = pd.read_parquet(original), pd.read_parquet(new)
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        same_bytes = original.read_bytes() == new.read_bytes()
        if not same_bytes:
            raise AssertionError(f'Parquet bytes differ: {name}/{year}')
        partitions.append({'factor': name, 'year': year, 'rows': len(actual),
                           'finite': int(np.isfinite(actual.value).sum()),
                           'values_exact': True, 'parquet_bytes_exact': same_bytes})
    report = {'start': args.start, 'end': args.end, 'stocks': len(engine.codes),
              'direct_seconds': round(direct_seconds, 2), 'original_compute_seconds': round(original_seconds, 2),
              'partitions': partitions, 'scope': 'original Engine computation; excludes its legacy daily orchestration'}
    _atomic_json(report, root/'comparison.json')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
