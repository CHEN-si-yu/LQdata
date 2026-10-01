#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`scripts/run_screen.py` 的向量化统计与「朴素实现」逐位对拍。

## 为什么要单独测这个

质检脚本把截面统计改成了行向量化（`sort`+`diff` 数不同取值、`sum` 算均值方差……），
**向量化的错法往往是静默的**：数值看着像模像样，只是偏离口径。
一旦偏了，整轮「删哪些因子」的结论就建在错的数上。

所以这里对每个向量化 helper 配一个**故意写得很笨但显然正确**的实现
（逐日循环 + `np.unique` / `pd.qcut` / `pd.Series.rank`），随机数据上逐位比对。

★ 顺带钉住一个**口径事实**：`_rank_avg` 必须是 **average** 名次（pandas 默认），
  不是 argsort 的序数名次 —— 零膨胀因子（一天里大半是 0）靠这个才不会
  「凭空造出截面区分度」。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_screen():
    spec = importlib.util.spec_from_file_location(
        "_run_screen", ROOT / "scripts" / "run_screen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_screen = _load_screen()
_rank_avg = _screen._rank_avg
_pearson = _screen._pearson
_daily_ic = _screen._daily_ic
_deciles = _screen._deciles
_uniq_per_day = _screen._uniq_per_day


def _rand_panel(T=25, C=140, nan_frac=0.15, seed=0, zero_frac=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, C)).astype(np.float64)
    if zero_frac:
        x[rng.random((T, C)) < zero_frac] = 0.0
    m = rng.random((T, C)) < nan_frac
    x[m] = np.nan
    return x


class TestRankAvg(unittest.TestCase):
    def test_matches_pandas_average_rank(self):
        for seed in range(6):
            x = _rand_panel(T=3, C=300, nan_frac=0.0, seed=seed)[0]
            got = _rank_avg(x)
            want = pd.Series(x).rank().to_numpy()
            np.testing.assert_allclose(got, want, atol=1e-12)

    def test_ties_get_average_not_ordinal(self):
        """零膨胀的典型形状：大半取值相同 —— 必须是平均名次。"""
        x = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 7.0])
        got = _rank_avg(x)
        want = pd.Series(x).rank().to_numpy()
        np.testing.assert_allclose(got, want, atol=1e-12)
        # 序数名次会给出 1,2,3,4 —— 那是错的
        self.assertAlmostEqual(got[0], 2.5)


class TestUniqPerDay(unittest.TestCase):
    def test_matches_np_unique(self):
        V = _rand_panel(nan_frac=0.2, seed=3, zero_frac=0.5)
        got = _uniq_per_day(V)
        for t in range(V.shape[0]):
            fin = V[t][np.isfinite(V[t])]
            want = len(np.unique(fin)) if fin.size else 0
            self.assertEqual(int(got[t]), want, f"第 {t} 日不同取值数不符")

    def test_all_nan_day_is_zero(self):
        V = np.full((3, 10), np.nan)
        np.testing.assert_array_equal(_uniq_per_day(V), np.zeros(3, dtype=int))

    def test_constant_day_is_one(self):
        V = np.full((2, 10), 3.5)
        np.testing.assert_array_equal(_uniq_per_day(V), np.ones(2, dtype=int))


class TestDailyIC(unittest.TestCase):
    def test_ic_matches_manual_pearson(self):
        V = _rand_panel(nan_frac=0.1, seed=5)
        L = _rand_panel(nan_frac=0.1, seed=6)
        ic, ric = _daily_ic(V, L, min_n=30)
        # 朴素复算（与 fea/eval.py::_daily_corr 同一写法）
        want_ic, want_ric = [], []
        for t in range(V.shape[0]):
            m = np.isfinite(V[t]) & np.isfinite(L[t])
            n = int(m.sum())
            if n < 30:
                continue
            x, y = V[t][m], L[t][m]
            sx, sy = x.std(), y.std()
            if not (sx > 0 and sy > 0):
                continue
            want_ic.append(float(((x - x.mean()) * (y - y.mean())).sum() / (n * sx * sy)))
            rx = pd.Series(x).rank().to_numpy()
            ry = pd.Series(y).rank().to_numpy()
            want_ric.append(_pearson(rx, ry))
        np.testing.assert_allclose(ic, want_ic, atol=1e-12)
        np.testing.assert_allclose(ric, want_ric, atol=1e-12)

    def test_min_n_gate_and_degenerate_days(self):
        V = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0], [1.0, 2.0, 3.0]])
        L = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        ic, ric = _daily_ic(V, L, min_n=3)
        # 第 2 行因子是常数 ⇒ 跳过；第 1、3 行完全共线 ⇒ IC=+1
        self.assertEqual(len(ic), 2)
        np.testing.assert_allclose(ic, [1.0, 1.0], atol=1e-12)


class TestDeciles(unittest.TestCase):
    """分层口径：按**名次**切 10 层（每条腿样本数尽量相等）。

    ★ 与 eval 的 `pd.qcut(..., duplicates="drop")` 的差别只有两处，都是刻意的：
      ① n 不能被 10 整除时，边界点落在哪一边会差 1 个样本（两侧都只是"四舍五入"）；
      ② 并列值：名次法把并列段**摊到相邻层**，qcut 会把并列整体塞进一层
         （稀疏事件因子一天里大半是 0 ⇒ 层会被撑爆）。
      下游只用 D9−D0 与单调性，这两处差别不改变判定。
    """

    def test_exactly_matches_qcut_when_n_divisible_by_10(self):
        """无缺失（n = C = 400，能被 10 整除）且无并列 ⇒ 与 qcut **逐位相同**。"""
        V = _rand_panel(T=8, C=400, nan_frac=0.0, seed=11)
        L = _rand_panel(T=8, C=400, nan_frac=0.0, seed=12)
        acc, cnt = _deciles(V, L, min_n=100)
        w_acc, w_cnt = np.zeros(10), np.zeros(10)
        for t in range(V.shape[0]):
            x, y = V[t], L[t]
            q = pd.qcut(pd.Series(x), 10, labels=False, duplicates="drop").to_numpy()
            for k in range(10):
                s = q == k
                if s.any():
                    w_acc[k] += y[s].mean()
                    w_cnt[k] += 1
        np.testing.assert_allclose(acc, w_acc, atol=1e-12)
        np.testing.assert_array_equal(cnt, w_cnt)
        np.testing.assert_array_equal(np.bincount(
            pd.qcut(pd.Series(V[0]), 10, labels=False, duplicates="drop").to_numpy(),
            minlength=10), np.full(10, 40))

    def test_boundary_rounding_when_n_not_divisible(self):
        """n 不能被 10 整除时，两侧只差 ±1 个样本 —— 不是口径分歧。"""
        V = _rand_panel(T=4, C=362, nan_frac=0.0, seed=21)
        L = _rand_panel(T=4, C=362, nan_frac=0.0, seed=22)
        for t in range(4):
            r = _rank_avg(V[t])
            mine = np.bincount(np.minimum(((r - 1.0) / 362 * 10).astype(np.int64), 9),
                               minlength=10)
            other = np.bincount(
                pd.qcut(pd.Series(V[t]), 10, labels=False, duplicates="drop").to_numpy(),
                minlength=10)
            self.assertLessEqual(int(np.abs(mine - other).max()), 1)

    def test_constant_factor_yields_nan_spread(self):
        """全常数因子：所有样本落在同一层 ⇒ 其余层无样本 ⇒ D9−D0 必须是 NaN。

        ★ 关键：**不能静默返回一个数**。常数因子的分层价差没有意义，
          返回 NaN 才会在上游被当成"该因子无区分度"处理。
        """
        V = np.zeros((5, 200))
        L = _rand_panel(T=5, C=200, nan_frac=0.0, seed=13)
        acc, cnt = _deciles(V, L, min_n=50)
        self.assertEqual(int((cnt > 0).sum()), 1)
        D = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        self.assertTrue(np.isnan(D[9] - D[0]))


class TestPearson(unittest.TestCase):
    def test_constant_is_nan(self):
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0, 2.0, 3.0])
        self.assertTrue(np.isnan(_pearson(a, b)))

    def test_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(_pearson(a, a), 1.0, places=12)
        self.assertAlmostEqual(_pearson(a, -a), -1.0, places=12)


if __name__ == "__main__":
    unittest.main()
