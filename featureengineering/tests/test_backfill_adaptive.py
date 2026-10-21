"""自适应调档控制律（scripts/backfill_history._next_jobs）的表驱动测试。

只钉住纯函数行为：目标带内保持、低于下沿加两档（封顶）、高于上沿减半（保底）、
测量未知不动。不碰数据、不起子进程。
"""
import resource
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

_soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
from backfill_history import _next_jobs  # noqa: E402
# backfill_history 顶层会调 configure() 把本进程 RLIMIT_AS 降到 24GiB；
# 测试进程里把它恢复（soft 可升回 hard 以内的值），避免影响其它用例。
resource.setrlimit(resource.RLIMIT_AS, (_soft, _hard))

B = 60_000_000_000


class NextJobsTests(unittest.TestCase):
    def test_below_band_adds_two_capped_by_hi(self):
        self.assertEqual(_next_jobs(int(0.30 * B), B, 4, 1, 12), 6)
        self.assertEqual(_next_jobs(int(0.49 * B), B, 11, 1, 12), 12)

    def test_in_band_holds(self):
        for ratio in (0.50, 0.60, 0.75, 0.85):
            self.assertEqual(_next_jobs(int(ratio * B), B, 8, 1, 12), 8, ratio)

    def test_above_band_halves_with_floor(self):
        self.assertEqual(_next_jobs(int(0.86 * B), B, 8, 1, 12), 4)
        self.assertEqual(_next_jobs(int(0.99 * B), B, 1, 1, 12), 1)

    def test_unknown_measurement_holds(self):
        self.assertEqual(_next_jobs(0, B, 8, 1, 12), 8)
        self.assertEqual(_next_jobs(int(0.2 * B), 0, 8, 1, 12), 8)


if __name__ == "__main__":
    unittest.main()
