#!/usr/bin/env python3
"""Standalone research commands; keep main.py dedicated to production."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from fea.config import load
    from fea.resources import configure
    from research.cli import add_parser, dispatch
    cfg = load()
    configure(cfg.raw.get('resources'))
    parser = argparse.ArgumentParser(description=__doc__)
    add_parser(parser.add_subparsers(dest='command', required=True))
    args = parser.parse_args(['research', *(sys.argv[1:] if argv is None else argv)])
    return dispatch(args, cfg)


if __name__ == '__main__':
    shared = Path(os.environ.get('FEA_PYTHON', '/autodl-fs/data/miniconda3/bin/python'))
    if shared.exists() and Path(sys.executable).resolve() != shared.resolve():
        os.execv(str(shared), [str(shared), str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.dont_write_bytecode = True
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault('FEA_MEMORY_BUDGET_GIB', '55')
    tmp = ROOT / 'artifacts/tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(tmp)
    raise SystemExit(main())
