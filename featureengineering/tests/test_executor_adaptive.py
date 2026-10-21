"""多路年份流执行器的车道控制律（纯函数，表驱动）。

对应 `fea/executor.py::_next_lanes` 与 `_jobs_for_streams`（2026-10-03 并行执行器）。
纪律来源：清理期实测「+2 路需峰值 <38GB 且 CPU ≥0.65×档位」+ 旧 `_next_jobs` 的
0.50/0.85 目标带；两处一起钉住，防止将来「为了快」把内存前提改掉。
"""
import unittest

from fea.executor import _jobs_for_streams, _next_lanes

GiB = 2 ** 30
GUARD = 54 * GiB          # 60 GiB 容器下的 charged 护栏（0.90×60）


class NextLanesTests(unittest.TestCase):
    def test_holds_inside_band(self):
        # 峰值 40 GiB：≥38 GiB 的加档前提不满足、不安全线也没到 → 保持
        self.assertEqual(_next_lanes(40 * GiB, cpu_seconds=60, wall_seconds=10,
                                     lanes=8, lo=1, hi=16, guard_bytes=GUARD), 8)

    def test_ramps_up_only_with_enough_cpu_duty(self):
        # 峰值 20 GiB（<38）且 CPU 6.5 核 ≥ 0.65×8 → +2
        self.assertEqual(_next_lanes(20 * GiB, 65, 10, 8, 1, 16, GUARD), 10)
        # 同样内存但 CPU 只有 4 核（<0.65×8=5.2）→ 不加档（喂不饱就别加）
        self.assertEqual(_next_lanes(20 * GiB, 40, 10, 8, 1, 16, GUARD), 8)

    def test_ramp_up_respects_ceiling(self):
        # 15 路、峰值 20 GiB、CPU 16 核（≥0.65×15=9.75）→ 想 +2 但封顶在 hi=16
        self.assertEqual(_next_lanes(20 * GiB, 160, 10, 15, 1, 16, GUARD), 16)

    def test_holds_when_charged_is_merely_high(self):
        # ★ 2026-10-03 实测教训：真实全库重建年块计费稳定在 48–53.5 GiB（护栏 54），
        #   但其中大头是**可回收页缓存**（anon 33/46）→ 必须保持不动。
        #   旧阈值 0.85×护栏（45.9）会把车道一路减半到 2 → 后 4 个年块爬行 40 分钟。
        self.assertEqual(_next_lanes(48 * GiB, 300, 60, 6, 1, 10, GUARD,
                                     anon_bytes=33 * GiB, anon_cap_bytes=46 * GiB), 6)
        self.assertEqual(_next_lanes(53 * GiB, 300, 60, 6, 1, 10, GUARD,
                                     anon_bytes=35 * GiB, anon_cap_bytes=46 * GiB), 6)

    def test_anon_danger_steps_down_by_two(self):
        # 不可回收 42 GiB > 0.90×46 = 41.4 → 降一档（-2，不是减半）
        self.assertEqual(_next_lanes(40 * GiB, 300, 60, 6, 3, 10, GUARD,
                                     anon_bytes=42 * GiB, anon_cap_bytes=46 * GiB), 4)

    def test_high_charged_with_low_anon_does_not_step_down(self):
        # 计费 53 GiB（接近护栏 54）但 anon 只有 20 GiB → 页缓存可回收，**不降档**。
        # （判定只看 anon；charged 是 run_bounded 的硬线，不是这里的调档依据。）
        self.assertEqual(_next_lanes(53 * GiB, 80, 10, 12, 3, 16, GUARD,
                                     anon_bytes=20 * GiB, anon_cap_bytes=46 * GiB), 12)

    def test_step_down_respects_floor(self):
        self.assertEqual(_next_lanes(53 * GiB, 80, 10, 3, 3, 16, GUARD,
                                     anon_bytes=44 * GiB, anon_cap_bytes=46 * GiB), 3)

    def test_unknown_peak_holds(self):
        self.assertEqual(_next_lanes(0, 80, 10, 8, 1, 16, GUARD), 8)


class JobsForStreamsTests(unittest.TestCase):
    def test_splits_total_budget(self):
        self.assertEqual(_jobs_for_streams(12, 2, 9), (2, 6))
        self.assertEqual(_jobs_for_streams(8, 2, 9), (2, 4))
        self.assertEqual(_jobs_for_streams(10, 2, 9), (2, 5))

    def test_single_year_cannot_stream(self):
        self.assertEqual(_jobs_for_streams(8, 2, 1)[0], 1)

    def test_never_exceeds_total_budget_per_stream(self):
        streams, jps = _jobs_for_streams(6, 4, 9)
        self.assertLessEqual(streams * jps, 6)

    def test_floor_of_two_lanes_to_open_a_stream(self):
        # 总车道 3 → 只能 1 路（每路 ≥2 才值得分）
        self.assertEqual(_jobs_for_streams(3, 2, 9)[0], 1)


if __name__ == "__main__":
    unittest.main()
