#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑本工程的单元测试（`tests/test_*.py`）。

## 为什么要一个 runner，而不直接 `python -m unittest discover -s tests`

1. **环境里没有 pytest**（共享盘那份 miniconda 只装了项目需要的库），测试全是 `unittest`；
2. `tests/` **没有 `__init__.py`**，`unittest discover -s tests -t .` 会直接报
   `ImportError: Start directory is not importable`；
3. 测试内部有 `from main import ...`（要工程根目录在 `sys.path` 上），
   所以不能简单地 `cd tests` 再跑。

本 runner 逐文件加载再汇总，等价于 discover；退出码 = 有失败则 1。

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/run_tests.py          # 一次几秒；不带任何数据读写（除少数读 state 的用例）
"""

from __future__ import annotations

import glob
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(ROOT))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for f in sorted(glob.glob(str(ROOT / "tests" / "test_*.py"))):
        name = "t_" + Path(f).stem
        spec = importlib.util.spec_from_file_location(name, f)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        suite.addTests(loader.loadTestsFromModule(mod))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
