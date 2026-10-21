#!/usr/bin/env python3
"""兼容旧命令；所有审计通过主入口及同一资源预算执行。"""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def _relax_address_space():
    """Legacy hook: apply the project policy; never raise it to container180GiB."""
    from fea.config import load
    from fea.resources import configure
    return configure(load())

if __name__=='__main__':
    python=Path(os.environ.get('FEA_PYTHON','/autodl-fs/data/miniconda3/bin/python'))
    if not python.exists(): python=Path(sys.executable)
    os.execv(str(python),[str(python),str(ROOT/'main.py'),'dedup',*sys.argv[1:]])
