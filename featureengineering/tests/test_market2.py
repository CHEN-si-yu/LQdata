#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二批市场因子（`factors/market2.py`）与个股可交易性（`factors/tradability.py`）的回归测试。

## 测什么、不测什么

这里只测**能在小样本上确定性判定**的东西：
  · 纯函数的语义（折叠口径、事件计数、滞后均值）；
  · 注册契约（名字唯一、`is_market`、`deps` 声明、滞后声明、起点）；
  · 行业候选的交叉核验（含"交集为空必须失败"这条负向路径）。

**不测**真实数据上的数值正确性 —— 那由独立复算（另一条代码路径直接聚合原始 parquet）负责。
单元测试守的是"以后改坏了能立刻发现"。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import factors                                        # noqa: E402,F401  导入即注册
import factors.market2 as m2                          # noqa: E402
import factors.tradability as td                      # noqa: E402
from fea.spec import REGISTRY, FactorSpec             # noqa: E402


class TestCrossSectionStats(unittest.TestCase):
    """`_cs` 的逐日横截面统计 —— 与逐行朴素实现逐位比。"""

    def _naive(self, M, kind, need):
        M = np.asarray(M, dtype=np.float64)
        out = []
        for row in M:
            fin = row[np.isfinite(row)]
            if fin.size < need:
                out.append(np.nan)
                continue
            if kind == "mean":
                out.append(float(fin.mean()))
            elif kind == "std":
                out.append(float(fin.std()))
            elif kind == "sum":
                out.append(float(fin.sum()))
            elif kind == "up_ratio":
                out.append(float((fin > 0).mean()))
        return np.array(out)

    def test_matches_naive_for_every_kind(self):
        rng = np.random.default_rng(7)
        M = rng.normal(size=(12, 50))
        M[rng.random(M.shape) < 0.25] = np.nan
        for kind in ("mean", "std", "sum", "up_ratio"):
            got = m2._cs(M, kind, need=10)
            want = self._naive(M, kind, 10)
            np.testing.assert_allclose(got, want, atol=1e-12, equal_nan=True)

    def test_below_need_is_nan_not_zero(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(m2._cs(M, "mean", need=3),
                                      np.array([np.nan, np.nan]))

    def test_all_nan_row_is_nan(self):
        M = np.full((3, 20), np.nan)
        out = m2._cs(M, "std", need=5)
        self.assertTrue(np.isnan(out).all())

    def test_exactly_zero_std_is_zero_not_nan(self):
        """常数截面：std 应为 0（真的是 0），不是 NaN —— NaN 表示"样本不够"。"""
        M = np.full((2, 20), 3.5)
        np.testing.assert_allclose(m2._cs(M, "std", need=5), [0.0, 0.0], atol=1e-12)


class TestEventCountByDay(unittest.TestCase):
    """`_count_by_day`：覆盖范围内"没记录 = 0"，覆盖范围外 = NaN。

    ★ 这条口径直接决定 `mkt_limit_*` / `mkt_suspension_ratio` 是"0 家涨停"还是"没有数据"，
      实测原实现把两者混成一个 NaN，让 `limit_net_ratio` 白丢 17% 的交易日。
    """

    DAYS = np.array([20200102, 20200103, 20200106, 20200107], dtype=np.int64)

    def test_no_record_inside_cover_is_zero(self):
        got = m2._count_by_day(self.DAYS, np.array([20200103]), cover=np.array(
            [20200102, 20200103, 20200106, 20200107]))
        np.testing.assert_allclose(got, [0.0, 1.0, 0.0, 0.0])

    def test_outside_cover_is_nan_not_zero(self):
        """覆盖范围之外必须是 NaN：否则 2018 年会满屏"涨停 0 家"这种假事实。"""
        got = m2._count_by_day(self.DAYS, np.array([20200103]),
                               cover=np.array([20200103, 20200106]))
        np.testing.assert_allclose(got, [np.nan, 1.0, 0.0, np.nan], equal_nan=True)

    def test_weights_sum_within_a_day(self):
        got = m2._count_by_day(self.DAYS, np.array([20200102, 20200102]),
                               weights=np.array([2.5, 3.5]), cover=self.DAYS)
        np.testing.assert_allclose(got, [6.0, 0.0, 0.0, 0.0])

    def test_without_cover_stays_nan(self):
        """没给 cover 就无法区分"没记录"与"没数据" ⇒ 保守地保持 NaN，不猜成 0。"""
        got = m2._count_by_day(self.DAYS, np.array([20200102]))
        np.testing.assert_allclose(got, [1.0, np.nan, np.nan, np.nan], equal_nan=True)


class TestRollSumNan(unittest.TestCase):
    def test_any_nan_in_window_poisons_the_window(self):
        M = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0]])
        got = m2._roll_sum_nan(M, 3)[:, 0]
        np.testing.assert_allclose(got, [np.nan, np.nan, np.nan, np.nan, np.nan],
                                   equal_nan=True)

    def test_matches_manual_sum(self):
        M = np.arange(20, dtype=float).reshape(10, 2)
        got = m2._roll_sum_nan(M, 4)
        for t in range(3, 10):
            np.testing.assert_allclose(got[t], M[t - 3:t + 1].sum(axis=0), atol=1e-12)

    def test_prefix_shorter_than_window_is_nan(self):
        got = m2._roll_sum_nan(np.ones((5, 1)), 8)
        self.assertTrue(np.isnan(got).all())


class TestLagSemantics(unittest.TestCase):
    def test_lag_moves_forward_in_time(self):
        np.testing.assert_allclose(m2._lag(np.array([1.0, 2.0, 3.0]), 1),
                                   [np.nan, 1.0, 2.0], equal_nan=True)

    def test_shift_mean_excludes_today(self):
        """★ 不含当日：当日值同时进分子分母会把比率拉向 1。"""
        x = np.array([10.0, 20.0, 30.0, 40.0])
        got = m2._shift_mean(x, 2)
        # t=2 时用的是 (t-2, t-1] = 第 0、1 天的值（不含第 2 天）
        self.assertAlmostEqual(got[2], 15.0, places=12)

    def test_lag2d_keeps_shape_and_fills_head(self):
        M = np.arange(6, dtype=float).reshape(3, 2)
        got = m2._lag2d(M, 1)
        self.assertEqual(got.shape, M.shape)
        self.assertTrue(np.isnan(got[0]).all())
        np.testing.assert_allclose(got[1], M[0])


class TestIndustryBoardVerification(unittest.TestCase):
    """行业候选必须**交叉核验名称**：TDX 0 类里混着总市值/涨跌家数这类统计序列。"""

    def _ctx(self, tdx_rows, dc_rows):
        from types import SimpleNamespace
        tdx = pd.DataFrame(tdx_rows, columns=["block_code", "block_name", "block_type"])
        dc = pd.DataFrame(dc_rows, columns=["block_code", "block_name", "block_type"])
        tables = {"tdx_blocks": tdx, "dc_blocks": dc}
        return SimpleNamespace(
            panel=SimpleNamespace(dates=np.array([20250102, 20250103], np.int32),
                                  codes=np.array(["A"]), size=1),
            dataset=lambda name, columns=None, years=None: tables[name][columns].copy(),
            up=object())

    def test_only_name_matched_industries_survive(self):
        ctx = self._ctx(
            [("880001.TDX", "银行", 0), ("880002.TDX", "白酒", 0),
             ("880003.TDX", "总市值", 0)],
            [("BK1.DC", "银行", "行业板块"), ("BK2.DC", "白酒", "行业板块")])
        m2._CACHE.clear()
        got = m2._industry_boards(ctx)
        np.testing.assert_array_equal(got, ["880001", "880002"])   # 后缀被剥掉；"总市值" 被排除

    def test_empty_intersection_raises_instead_of_falling_back(self):
        """★ 负向路径：交集为空必须**失败**，绝不回退到整个 0 类。"""
        ctx = self._ctx([("880001.TDX", "总市值", 0)],
                        [("BK1.DC", "银行", "行业板块")])
        m2._CACHE.clear()
        with self.assertRaises(RuntimeError):
            m2._industry_boards(ctx)


class TestRegistrationContract(unittest.TestCase):
    """注册契约：名字唯一、市场因子无 rank、deps 必须声明、滞后源必须声明。"""

    def setUp(self):
        self.new_market = sorted(n for n, s in REGISTRY.items()
                                 if s.group == "market"
                                 and getattr(s.fn, "__module__", "").endswith("market2"))
        self.new_stock = sorted(n for n, s in REGISTRY.items() if s.group == "tradability")

    def test_expected_counts(self):
        # ★ 2026-09-24：冗余审计（newfactor_corr.py）查出 6 个与**已有**因子 |ρ|≥0.90
        #   （mkt_full_up_ratio 与 mkt_up_ratio 0.984 等），已删 ⇒ 37 → 31。
        self.assertEqual(len(self.new_market), 31)
        self.assertEqual(len(self.new_stock), 5)

    def test_all_have_deps_and_note(self):
        for n in self.new_market + self.new_stock:
            s = REGISTRY[n]
            self.assertTrue(s.deps, f"{n} 没声明 deps")
            self.assertTrue(s.desc and s.formula, f"{n} 缺 desc/formula")
            self.assertTrue(s.note, f"{n} 缺 note（口径必须写在 note 里）")

    def test_market_flags(self):
        for n in self.new_market:
            s = REGISTRY[n]
            self.assertTrue(s.is_market, f"{n} 应为市场因子")
            self.assertFalse(s.is_label)
        for n in self.new_stock:
            s = REGISTRY[n]
            self.assertFalse(s.is_market, f"{n} 应为股票因子")
            self.assertFalse(s.is_label)

    def test_lagged_dependency_is_declared(self):
        """★ 两融是唯一的滞后源：依赖它却漏声明 `lagged_ok` 的因子必须被挑出来。"""
        from fea.delay import load as load_delays
        lagged = set(load_delays())
        for n in self.new_market + self.new_stock:
            s = REGISTRY[n]
            need = lagged & set(s.deps)
            self.assertTrue(need <= set(s.lagged_ok),
                            f"{n} 依赖滞后表 {need} 但没声明 lagged_ok")

    def test_limit_list_factors_start_at_2020(self):
        """`stock_limit_list` 实测最早 2020-01-02；声明更早起点会产出"涨停 0 次"的假事实。"""
        for n in self.new_stock:
            if "stock_limit_list" in REGISTRY[n].deps:
                self.assertTrue(str(REGISTRY[n].start).startswith("2020"),
                                f"{n} 的显式起点应不早于 2020，实际 {REGISTRY[n].start}")

    def test_verify_actual_shift_for_lagged_factors(self):
        """声明了滞后还不够 —— 函数体里必须真的位移过（否则 T 日值会静默变化）。"""
        import inspect
        for n in self.new_market:
            s = REGISTRY[n]
            if "stock_margin_detail" not in s.deps:
                continue
            src = inspect.getsource(s.fn)
            self.assertIn("_lag", inspect.getsource(m2._margin),
                          f"{n} 依赖两融但 _margin 里没有行位移")
            del src


if __name__ == "__main__":
    unittest.main()
