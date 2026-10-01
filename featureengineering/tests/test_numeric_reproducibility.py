# -*- coding: utf-8 -*-
"""数值可复现性回归测试（2026-09-25 新增）。

背景（实测）：因子落盘值的**末位**取决于浮点实现，不只是代码和输入。

  · `arcsinh(float32)` 走 numpy 的 SIMD 核，与「双精度算完再舍入」在 **24.76%**
    的输入上不一致（最多 2 ULP）；同一个函数在 float64 下与 libm **逐位相同**。
  · 实测后果：同一份代码 + 同一份输入（逐文件 md5 核对）+ 同一个计算计划，
    换一套 numpy 就让 87 个对象的落盘值在末位漂移，`main.py` 只能报成
    「无法解释的历史变化」并返回非零。

因此这里钉住三件事：
  1. `ctx.safe_div` 必须返回 float64（它是唯一一处引擎主动把中间量压成 float32 的地方，
     而调用方紧接着就要做超越函数 —— 见 fea/context.py 的说明）；
  2. 数值环境指纹必须**便宜、确定、且对 float32 核敏感**（环境变了要能看见）；
  3. float32 超越函数探针本身要能抓到调用（它是审计「还有谁把 float32 送进 asinh/log」
     的工具，见 fea/resources.py）。
"""

from __future__ import annotations

import unittest

import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fea.context import FactorContext                     # noqa: E402
from fea.resources import (install_float32_guard, numeric_env_fingerprint,  # noqa: E402
                           set_guard_factor)


class TestSafeDivPrecision(unittest.TestCase):
    def test_returns_float64(self):
        num = np.array([1.0, 2.0, 3.0])
        den = np.array([4.0, 8.0, 0.0])
        out = FactorContext.safe_div(num, den, min_abs_den=1e-12)
        self.assertEqual(out.dtype, np.float64,
                         "safe_div 返回 float32 会让调用方的 asinh/log 走 numpy 的 "
                         "float32 SIMD 核 —— 末位随 numpy 版本变，落盘值不可复现")

    def test_zero_and_nan_denominator(self):
        out = FactorContext.safe_div(np.array([1.0, 1.0]), np.array([0.0, np.nan]))
        self.assertTrue(np.isnan(out).all())

    def test_matches_float64_division(self):
        num = np.array([1.0, -2.5, 3.0])
        den = np.array([3.0, 7.0, -11.0])
        out = FactorContext.safe_div(num, den)
        np.testing.assert_array_equal(out, num / den)


class TestNumericEnvFingerprint(unittest.TestCase):
    def test_stable_within_process(self):
        self.assertEqual(numeric_env_fingerprint(), numeric_env_fingerprint())

    def test_shape_and_version_binding(self):
        fp = numeric_env_fingerprint()
        self.assertIsInstance(fp, str)
        self.assertEqual(len(fp), 12)
        # 指纹必须把 numpy 版本绑进去：换了 numpy 就是换了数值环境
        import hashlib
        self.assertNotEqual(fp, hashlib.md5(b"numpy-does-not-matter").hexdigest()[:12])

    def test_float32_kernel_sensitivity(self):
        """指纹必须对 float32 超越函数核的差异敏感 —— 否则环境漂移无法被看见。

        做法：临时把 `np.arcsinh` 换成"双精度算完再舍入"的实现（= 实测中另一套
        环境的行为），指纹必须变。不这样构造的话，本测试在任何单机上都只能证明
        "同环境同值"，证明不了"换核会变"。
        """
        import numpy as np
        from fea import resources
        before = resources.numeric_env_fingerprint()
        saved_fp = resources._NUMENV_FP
        saved_arcsinh = np.arcsinh
        try:
            resources._NUMENV_FP = None
            np.arcsinh = lambda x, *a, **k: (  # noqa: E731
                np.float32(np.arcsinh(x.astype(np.float64))) if x.dtype == np.float32
                else saved_arcsinh(x, *a, **k))
            after = resources.numeric_env_fingerprint()
        finally:
            np.arcsinh = saved_arcsinh
            resources._NUMENV_FP = saved_fp
        self.assertNotEqual(before, after,
                            "数值指纹对 float32 核的变化不敏感 —— 换 numpy 会静默漂移")
        self.assertEqual(before, numeric_env_fingerprint())


class TestFloat32Guard(unittest.TestCase):
    def test_guard_records_float32_transcendental(self):
        import json
        import tempfile
        from pathlib import Path as _P
        with tempfile.TemporaryDirectory() as td:
            out = _P(td) / "guard.jsonl"
            install_float32_guard(str(out))
            try:
                set_guard_factor("unit_test_factor")
                np.arcsinh(np.array([0.1, 0.2], dtype=np.float32))   # 应该被抓到
                np.arcsinh(np.array([0.1, 0.2], dtype=np.float64))   # 不该被抓到
            finally:
                set_guard_factor(None)
                # atexit 还没跑，手动触发落盘：直接读内存集合
                from fea import resources
                rows = sorted(resources._guard_seen)
                self.assertTrue(any(r[0] == "unit_test_factor" and r[1] == "arcsinh" for r in rows),
                                f"探针没抓到 float32 的 arcsinh：{rows}")
                self.assertTrue(all(r[1] != "sqrt" for r in rows))


if __name__ == "__main__":
    unittest.main()
