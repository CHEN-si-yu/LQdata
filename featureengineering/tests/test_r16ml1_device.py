"""设备切换时仍保护缓存，同时允许 CPU 的不同年度各自计算。"""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from research.families import r16_ml1_official_v2 as ml


class DeviceLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path(__file__).resolve().parents[1] / 'artifacts/tests/tmp'
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def test_cpu_can_run_while_gpu_lock_is_owned(self):
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as directory:
            ctx = self.context(Path(directory), 20261013)
            with ml._compute_device_lock(ctx, SimpleNamespace(type='cuda')):
                with ml._compute_device_lock(ctx, SimpleNamespace(type='cpu')):
                    pass
                with self.assertRaisesRegex(RuntimeError, 'GPU'):
                    with ml._compute_device_lock(ctx, SimpleNamespace(type='cuda')):
                        pass

    def test_cpu_excludes_same_cache_and_allows_other_year(self):
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as directory:
            root = Path(directory)
            ctx = self.context(root, 20261013)
            cpu = SimpleNamespace(type='cpu')
            with ml._compute_device_lock(ctx, cpu):
                with self.assertRaisesRegex(RuntimeError, 'CPU'):
                    with ml._compute_device_lock(ctx, cpu):
                        pass
                with ml._compute_device_lock(self.context(root, 20251013), cpu):
                    pass
                with ml._compute_device_lock(self.context(root, 20261013, root / 'sandbox'), cpu):
                    pass
            with ml._compute_device_lock(ctx, cpu):
                pass

    @staticmethod
    def context(root, last_date, derived=None):
        cfg = SimpleNamespace(contract_root=root,
                              project_path=lambda *args: derived or root / 'derived')
        return SimpleNamespace(cfg=cfg, panel=SimpleNamespace(dates=np.array([last_date])),
                               up=SimpleNamespace(audit_cutoff=None))


if __name__ == '__main__':
    unittest.main()
