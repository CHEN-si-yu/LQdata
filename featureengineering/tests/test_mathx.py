"""fea/mathx.py 的属性测试 —— 逐格与 pandas 对拍。

跑法（**本环境没有 pytest**，用标准库 unittest）：
    /autodl-fs/data/miniconda3/bin/python -m unittest discover -s tests -v

为什么值得写这么细：mathx 是 250 个因子的公共地基，它错一格，
所有价格/技术/日内因子都错，而且**多数是静默错**（看起来只是"因子有点噪"）。
"""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fea import mathx as mx  # noqa: E402


T, C = 400, 8
N = 20


def make_panel(seed: int = 7) -> np.ndarray:
    """构造带各种病态列的测试面板。

    列 0: 常数    列 1: 全 NaN      列 2: 只有一个有限值
    列 3: |x| ~ 1e8（抵消探针，模拟成交量/成交额的真实量级）
    列 4: 25σ 离群点（模拟涨跌停/复牌这种真实极端值）
    列 5: 阶跃（regime break）       列 6/7: 正常收益序列

    ⚠️ 刻意**不**放「0.02 量级的序列里塞一个 1e6 的离群点」这种构造 ——
    那不是真实因子输入，而且它会让任何 cumsum 差分的方差估计都失效
    （窗口内所有值都远离列均值时精度全丢）。见 `roll_var` 的 docstring。
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.02, size=(T, C))
    x[:, 0] = 0.5
    x[:, 1] = np.nan
    x[:, 2] = np.nan
    x[10, 2] = 1.0
    x[:, 3] = rng.normal(1e8, 1.0, size=T)
    x[200, 4] = 0.5
    x[150:, 5] += 0.5
    # 随机注入 8% NaN
    mask = rng.random((T, C)) < 0.08
    x[mask] = np.nan
    return x


def ref(x: np.ndarray, n: int, how: str, min_count=None, other=None):
    """pandas 参考实现，**显式写出毒化策略**以对齐 mathx 的默认语义。"""
    df = pd.DataFrame(x)
    mp = n if min_count is None else min_count
    r = getattr(df.rolling(n, min_periods=mp), how)()
    if other is not None:
        r = df.rolling(n, min_periods=mp).corr(pd.DataFrame(other))
    if min_count is None:
        cnt = df.rolling(n).count()
        r = r.where(cnt == n)
    return r.to_numpy()


class TestNoSideEffects(unittest.TestCase):
    """所有原语都不得就地修改入参，也不得产生 warning。"""

    def test_primitives(self):
        x = make_panel()
        fns = [
            ("roll_sum", lambda a: mx.roll_sum(a, N)),
            ("roll_mean", lambda a: mx.roll_mean(a, N)),
            ("roll_std", lambda a: mx.roll_std(a, N)),
            ("roll_var", lambda a: mx.roll_var(a, N)),
            ("roll_count", lambda a: mx.roll_count(a, N)),
            ("roll_max", lambda a: mx.roll_max(a, N)),
            ("roll_min", lambda a: mx.roll_min(a, N)),
            ("roll_rank", lambda a: mx.roll_rank(a, N)),
            ("roll_skew", lambda a: mx.roll_skew(a, N)),
            ("roll_kurt", lambda a: mx.roll_kurt(a, N)),
            ("roll_argmax", lambda a: mx.roll_argmax(a, N)),
            ("roll_prod", lambda a: mx.roll_prod(a, N)),
            ("ewm_mean", lambda a: mx.ewm_mean(a, span=N)),
            ("decay_linear", lambda a: mx.decay_linear(a, N)),
            ("nan_fill_ffill", mx.nan_fill_ffill),
            ("shift", lambda a: mx.shift(a, 5)),
            ("diff", lambda a: mx.diff(a, 5)),
            ("pct_change", lambda a: mx.pct_change(a, 5)),
            ("cs_demean", mx.cs_demean),
            ("cs_zscore", mx.cs_zscore),
            ("cs_winsor", mx.cs_winsor),
        ]
        for name, fn in fns:
            with self.subTest(fn=name):
                before = x.copy()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    out = fn(x)
                self.assertTrue(np.array_equal(x, before, equal_nan=True),
                                f"{name} 就地修改了入参")
                self.assertEqual(out.shape, x.shape, f"{name} 输出形状变了")
                self.assertEqual(out.dtype, np.float64, f"{name} 输出不是 float64")
                self.assertFalse(np.isinf(out).any(), f"{name} 产出 ±inf")
                noisy = [str(z.message) for z in w
                         if "RuntimeWarning" in str(z.category)]
                self.assertEqual(noisy, [], f"{name} 产生 warning: {noisy}")


class TestRolling(unittest.TestCase):

    def setUp(self):
        self.x = make_panel()

    def _cmp(self, got, want, tol, label):
        """混合绝对+相对容差。

        测试面板里有一列 `|x| ~ 1e8` 的**故意病态**数据，用来逼出抵消误差。
        判据写成分三项，每一项都有明确出处：
          · `tol·scale`            —— 算法本身的相对精度要求
          · `8·eps·max|x|·√N`      —— **float64 表示极限**：一个 ~1e8 的数最小间隔
            就是 `eps·1e8 ≈ 2.2e-8`，窗口内还累加了 √N 倍。任何算法都做不到更准，
            这不是 bug。实测 `roll_std` 在该列上误差 1.07e-7，正落在这个界内。
          · `1e-9·scale·N`         —— cumsum 差分的累积误差（求和不像均值/方差
            那样能靠中心化规避）
        """
        both = np.isfinite(got) & np.isfinite(want)
        self.assertTrue(np.array_equal(np.isfinite(got), np.isfinite(want)),
                        f"{label}: 有效值位置不一致 "
                        f"(got {np.isfinite(got).sum()} vs want {np.isfinite(want).sum()})")
        if both.any():
            g, w = got[both], want[both]
            scale = np.nanmax(np.abs(w)) if w.size else 1.0
            xmax = np.nanmax(np.abs(self.x))
            eps = np.finfo(np.float64).eps
            atol = (1e-9 * max(scale, 1.0) * N
                    + 8 * eps * xmax * np.sqrt(N))
            err = np.max(np.abs(g - w))
            self.assertLess(err, atol + tol * scale,
                            f"{label}: 最大绝对误差 {err:g} > "
                            f"{atol + tol * scale:g} (scale={scale:g}, xmax={xmax:g})")

    def test_sum_mean_count(self):
        self._cmp(mx.roll_sum(self.x, N), ref(self.x, N, "sum"), 1e-10, "roll_sum")
        self._cmp(mx.roll_mean(self.x, N), ref(self.x, N, "mean"), 1e-10, "roll_mean")

    def test_count_head_is_nan(self):
        """roll_count **不毒化**（它就是用来度量缺口的），但前 n-1 行窗口不足 -> NaN。

        pandas 的 `rolling(n, min_periods=0).count()` 会给前 n-1 行部分计数，
        这不是我们要的语义（框架里不存在「半个窗口」）。
        """
        got = mx.roll_count(self.x, N)
        want = ref(self.x, N, "count", min_count=N)
        self.assertTrue(np.isnan(got[:N - 1]).all())
        self.assertTrue(np.array_equal(got[N - 1:], want[N - 1:]))

    def test_std_var(self):
        # ★ pandas 的 `rolling.std()/var()` 默认 **ddof=1**（样本口径），
        #   mathx 用 ddof=0（总体口径，与 numpy 一致）。两者差 sqrt(n/(n-1))，
        #   对同一横截面是常数倍，不改变排序。测试必须显式对齐 ddof。
        df = pd.DataFrame(self.x)
        for how, tol in (("std", 1e-8), ("var", 1e-8)):
            for name, fn in ((f"roll_{how}", getattr(mx, f"roll_{how}"),),):
                want = getattr(df.rolling(N, min_periods=N), how)(ddof=0)
                want = want.where(df.rolling(N).count() == N).to_numpy()
                self._cmp(fn(self.x, N), want, tol, name)

    def test_max_min(self):
        self._cmp(mx.roll_max(self.x, N), ref(self.x, N, "max"), 1e-12, "roll_max")
        self._cmp(mx.roll_min(self.x, N), ref(self.x, N, "min"), 1e-12, "roll_min")

    def test_corr_against_direct_computation(self):
        """★ 参考实现**不能**用 `pandas.rolling.corr` —— 实测它在 `|x| ~ 1e8`
        的列上会算出 **−3.16**（相关系数越界），因为它用的是同一个未中心化的
        不稳定计算式。这里改用逐窗直接计算作为基准。
        """
        # 用独立的**无 NaN** 面板，否则凑不满整窗（make_panel 会给所有列注入 NaN）
        rng = np.random.default_rng(31)
        a = rng.normal(0, 0.02, size=(120, 5))
        a[:, 3] = rng.normal(1e8, 1.0, size=120)      # 大均值 + 小波动
        b = rng.normal(0, 0.02, size=(120, 5))
        b[:, 3] = rng.normal(1e8, 1.0, size=120)
        got = mx.roll_corr(a, b, N)
        self.assertTrue(np.nanmax(np.abs(got)) <= 1.0 + 1e-9,
                        "mathx 算出的相关系数越界了")
        checked = 0
        for c in range(5):
            for t in rng.choice(np.arange(N - 1, 120), size=6, replace=False):
                want = np.corrcoef(a[t - N + 1:t + 1, c], b[t - N + 1:t + 1, c])[0, 1]
                self.assertAlmostEqual(got[t, c], want, places=9,
                                       msg=f"cell ({t},{c}): {got[t, c]} vs {want}")
                checked += 1
        self.assertGreater(checked, 5, "没抽到足够多的完整窗口，测试没起作用")

    def test_corr_beats_pandas_on_large_magnitude(self):
        """记录一个实测事实：pandas 的 `rolling.corr` 在 1e8 量级列上会**越界**，
        而 mathx（先中心化）与逐窗直接计算一致。这条测试是防止未来有人
        「优化」掉 `_centered`。"""
        rng = np.random.default_rng(31)
        a = rng.normal(1e8, 1.0, size=(120, 1))
        b = rng.normal(1e8, 1.0, size=(120, 1))
        pd_corr = pd.DataFrame(a).rolling(N, min_periods=N).corr(pd.DataFrame(b)).to_numpy()
        mx_corr = mx.roll_corr(a, b, N)
        self.assertTrue(np.nanmax(np.abs(pd_corr)) > 1.5,
                        "pandas 这次居然没越界 —— 该测试的立论变了，需重新评估")
        self.assertTrue(np.nanmax(np.abs(mx_corr)) <= 1.0 + 1e-9)

    def test_skew_kurt(self):
        # 用「收益量级」的干净数据验证去偏系数是否与 pandas 一致
        rng = np.random.default_rng(3)
        z = rng.normal(0, 0.02, size=(300, 3))
        z[50:80, 1] *= 3.0
        self._cmp(mx.roll_skew(z, N), ref(z, N, "skew"), 1e-9, "roll_skew")
        self._cmp(mx.roll_kurt(z, N), ref(z, N, "kurt"), 1e-9, "roll_kurt")

    def test_ewm(self):
        """在**无 NaN** 的面板上逐格对拍（NaN 语义两边不同，见下）。

        ⚠️ 语义差异（有意为之，已在 docstring 写明）：
          mathx 遇到 NaN **不推进衰减**（把缺失当作「没有新信息」）；
          pandas 的 `ignore_na=False` 会让 NaN 位置也走一步衰减。
        对因子面板（NaN 密集）而言，mathx 的行为更不容易把缺失当信号，
        所以这里只在无 NaN 数据上验证数值一致性。
        """
        rng = np.random.default_rng(21)
        clean = rng.normal(0, 0.02, size=(300, 4))
        want = pd.DataFrame(clean).ewm(span=N, adjust=False).mean().to_numpy()
        got = mx.ewm_mean(clean, span=N)
        err = np.max(np.abs(got - want))
        self.assertLess(err, 1e-14, f"ewm_mean 误差 {err:g}")

    def test_rank_against_manual(self):
        """roll_rank 用一个小面板手工复算（窗口最后一个值的百分位）。"""
        rng = np.random.default_rng(5)
        z = rng.normal(size=(60, 3))
        got = mx.roll_rank(z, 10)
        for t in range(9, 60):
            for c in range(3):
                w = z[t - 9:t + 1, c]
                want = (w < w[-1]).sum() / (len(w) - 1)
                self.assertAlmostEqual(got[t, c], want, places=12)

    def test_argmax_is_bars_since_high(self):
        z = np.zeros((30, 1))
        z[20, 0] = 5.0                     # 窗口内最高点在 t=20
        got = mx.roll_argmax(z, 10)
        self.assertEqual(got[25, 0], 5.0)  # t=25 距 t=20 有 5 个 bar
        self.assertEqual(got[29, 0], 9.0)  # t=29 距今 9 个 bar

    def test_shift_direction(self):
        a = np.arange(10, dtype=float).reshape(10, 1)
        self.assertEqual(mx.shift(a, 1)[1, 0], 0.0)     # k>0 取过去
        self.assertTrue(np.isnan(mx.shift(a, 1)[0, 0]))
        self.assertEqual(mx.shift(a, -1)[0, 0], 1.0)    # k<0 取未来（仅标签用）
        self.assertTrue(np.isnan(mx.shift(a, -1)[-1, 0]))

    def test_partial_window_variance_uses_valid_count(self):
        """★ 回归测试：`min_count < n` 时方差/相关必须用**有效值个数**做分母。

        曾经的写法把缺失当 0 进 cumsum、分母仍取固定 n，于是系统性低估：
        实测 20 日窗口缺 3 个时算得 1.125 而正确值 1.219（偏小 8%）。
        相关同理（−0.2063 vs −0.2083）——**不报错，只是"因子有点噪"**。

        ⚠️ 注意契约：本框架**没有前导部分窗口**（前 n-1 行恒为 NaN），
        所以只能与 pandas 在 `t >= n-1` 上比对。
        """
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, size=(80, 2))
        x[10:20, 0] = np.nan
        y = rng.normal(0, 1, size=(80, 2))
        n = 20
        got = mx.roll_var(x, n, min_count=10)
        want = pd.DataFrame(x).rolling(n, min_periods=10).var(ddof=0).to_numpy()
        sl = slice(n - 1, None)
        m = np.isfinite(got[sl]) & np.isfinite(want[sl])
        self.assertTrue(np.array_equal(np.isfinite(got[sl]), np.isfinite(want[sl])),
                        "有效值位置不一致（t>=n-1 区间内）")
        self.assertLess(float(np.max(np.abs(got[sl][m] - want[sl][m]))), 1e-12)
        # 相关
        g3 = mx.roll_corr(x, y, n, min_count=10)
        err = 0.0
        for t in range(n - 1, 80):
            for c in range(2):
                a, b = x[t - n + 1:t + 1, c], y[t - n + 1:t + 1, c]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 10 or not np.isfinite(g3[t, c]):
                    continue
                err = max(err, abs(g3[t, c] - np.corrcoef(a[ok], b[ok])[0, 1]))
        self.assertLess(err, 1e-12)

    def test_no_partial_leading_window(self):
        """契约：前 n-1 行恒为 NaN（不存在「半个窗口」）。"""
        rng = np.random.default_rng(2)
        x = rng.normal(size=(40, 2))
        for fn, kw in ((mx.roll_mean, {}), (mx.roll_std, {}), (mx.roll_max, {}),
                       (mx.roll_rank, {}), (mx.roll_skew, {})):
            with self.subTest(fn=fn.__name__):
                out = fn(x, 20, **kw)
                self.assertTrue(np.isnan(out[:19]).all())

    def test_min_count_relaxes_poisoning(self):
        z = np.full((30, 1), np.nan)
        z[5:15, 0] = 1.0
        # 满窗+无 NaN：t=14 的窗口 5..14 全是 1 -> 有值；t=15 窗口 6..15 含 NaN -> NaN
        s = mx.roll_sum(z, 10)
        self.assertEqual(s[14, 0], 10.0)
        self.assertTrue(np.isnan(s[15, 0]))
        # min_count=5：t=15 的窗口里有 9 个有效值 -> 不毒化
        s2 = mx.roll_sum(z, 10, min_count=5)
        self.assertEqual(s2[15, 0], 9.0)


class TestCrossSection(unittest.TestCase):

    def test_zscore_matches_pandas(self):
        # 同样对齐 ddof：mathx 用总体口径 ddof=0，pandas row-wise std 默认 ddof=1
        x = make_panel()
        got = mx.cs_zscore(x)
        df = pd.DataFrame(x)
        want = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=0), axis=0).to_numpy()
        both = np.isfinite(got) & np.isfinite(want)
        self.assertTrue(np.allclose(got[both], want[both], rtol=1e-10, atol=1e-12),
                        f"最大偏差 {np.max(np.abs(got[both]-want[both])):g}")

    def test_winsor_all_nan_row_survives(self):
        x = np.full((5, 4), np.nan)
        x[2] = [1.0, 2.0, 3.0, 100.0]
        out = mx.cs_winsor(x)
        self.assertTrue(np.isnan(out[0]).all())
        self.assertLess(out[2].max(), 100.0)

    def test_roll_corr_clipped_to_unit_interval(self):
        """★ 回归测试：近常数窗口（长期停牌前向填充出的平序列）里
        `roll_corr` 的分子分母都是浮点噪声，实测会吐出 |ρ| 高达 14.4 的垃圾。
        必须兜底成 NaN（「无定义判缺失」），不能当成极值信号传给下游。"""
        rng = np.random.default_rng(9)
        x = rng.normal(size=(120, 3))
        y = rng.normal(size=(120, 3))
        x[40:80, 0] = 5.0          # 平序列
        y[40:80, 0] = 7.0
        got = mx.roll_corr(x, y, 20)
        finite = got[np.isfinite(got)]
        self.assertTrue(np.all(np.abs(finite) <= 1.0 + 1e-12),
                        f"相关系数越界：max|ρ| = {np.max(np.abs(finite))}")

    def test_safe_div(self):
        num = np.array([[1.0, 1.0, 1.0, np.nan]])
        den = np.array([[2.0, 0.0, np.nan, 1.0]])
        out = mx.safe_div(num, den)
        self.assertEqual(out[0, 0], 0.5)
        self.assertTrue(np.isnan(out[0, 1]))   # 分母恰为 0
        self.assertTrue(np.isnan(out[0, 2]))   # 分母 NaN
        self.assertTrue(np.isnan(out[0, 3]))   # 分子 NaN


if __name__ == "__main__":
    unittest.main(verbosity=2)
