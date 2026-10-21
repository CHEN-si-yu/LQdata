"""Preregistered batch contracts and causal formulas on finite raw fixtures."""
from __future__ import annotations

import tempfile
import json
import pathlib
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from fea.config import Config
from fea.context import FactorContext
from fea.engine import Engine
from fea.panel import Panel
from fea.upstream import Upstream
from research.catalog import build_spec, load_definition
from research.families import selection_20261001_v1 as batch


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = ROOT / "conf/research/batch_20261001"


class FixturePrices:
    def __init__(self, fields, traded):
        self.fields, self.traded = fields, traded

    def raw_panel(self, panel, field):
        return self.fields[field][:panel.T].copy()

    def panel(self, panel, field):
        if field == "traded":
            return self.traded[:panel.T].copy()
        if field.startswith("hfq_"):
            return (self.fields[field[4:]] * self.fields["adj_factor"])[:panel.T].copy()
        return self.raw_panel(panel, field)

    def mask(self, panel, field):
        assert field == "traded"
        return self.traded[:panel.T].copy()

    def ret(self, panel, k):
        close = self.panel(panel, "hfq_close")
        result = np.full(close.shape, np.nan)
        result[k:] = close[k:] / close[:-k] - 1.
        return result


class FixtureSource:
    def __init__(self, frame):
        self.frame, self.reads = frame, []

    def read(self, name, columns=None, years=None):
        assert name == "stock_main_fund_flow"
        self.reads.append((name, tuple(columns or ()), years))
        frame = self.frame
        if years:
            y = pd.to_datetime(frame.trade_date).dt.year
            frame = frame.loc[y.between(*years)]
        return frame.loc[:, columns].copy() if columns else frame.copy()


class BatchFixture:
    def __init__(self, count=180, stocks=8):
        rng = np.random.default_rng(6101)
        days = pd.bdate_range("2019-01-02", periods=count)
        self.dates = days.strftime("%Y%m%d").to_numpy(dtype=np.int32)
        self.codes = np.asarray([f"{i:06d}.SZ" for i in range(1, stocks + 1)])
        daily_return = rng.normal(.0002, .015, size=(count, stocks))
        adjusted_close = 12. * np.exp(np.cumsum(daily_return, axis=0))
        adjustment = np.ones_like(adjusted_close)
        adjustment[85:] = 1.4
        close = adjusted_close / adjustment
        opening = close * np.exp(rng.normal(0., .006, size=close.shape))
        upper = rng.uniform(.001, .025, size=close.shape)
        lower = rng.uniform(.001, .025, size=close.shape)
        high = np.maximum(opening, close) * (1. + upper)
        low = np.minimum(opening, close) * (1. - lower)
        amount = np.exp(19. + 8. * np.abs(daily_return) + rng.normal(0., .45, size=close.shape))
        self.fields = {"open": opening, "high": high, "low": low,
                       "close": close, "amount": amount, "vol": amount / close,
                       "adj_factor": adjustment}
        self.traded = np.ones(close.shape, bool)
        # Both sides describe the same gross transactions, with different size
        # allocations. Total buy-minus-sell is zero; individual tiers are not.
        buy_share = rng.dirichlet([2., 2., 2., 2.], size=(count, stocks))
        sell_share = rng.dirichlet([2., 2., 2., 2.], size=(count, stocks))
        gross = amount / 1e4
        flow = {"stock_code": np.tile(self.codes, count),
                "trade_date": np.repeat(days.strftime("%Y-%m-%d"), stocks)}
        for side, share in (("buy", buy_share), ("sell", sell_share)):
            for i, tier in enumerate(batch.TIERS):
                bucket_amount = gross * share[..., i]
                flow[f"{side}_{tier}_amount"] = bucket_amount.ravel()
                # Yuan/share = (amount in WAN / volume in HAND) * 100.
                price_offset = 1.01 if side == "buy" else .99
                flow[f"{side}_{tier}_vol"] = (bucket_amount * 100. / (close * price_offset)).ravel()
        self.flow = pd.DataFrame(flow)

    def context(self, count=None, source=None, start=0):
        count = len(self.dates) - start if count is None else count
        stop = start + count
        panel = Panel(self.dates[start:stop], self.codes)
        return FactorContext(panel, None, source or FixtureSource(self.flow),
                             SimpleNamespace(min_cross_section=3),
                             np.ones(panel.shape, bool),
                             prices=FixturePrices({k: v[start:stop] for k, v in self.fields.items()}, self.traded[start:stop]))


class BatchContractTests(unittest.TestCase):
    def test_all_definitions_build_as_distinct_fixed_mechanisms(self):
        manifest = load_definition(DEFINITIONS / "manifest.yaml")
        self.assertEqual(manifest["candidate_count"], 16)
        self.assertEqual(manifest["family_count"], 4)
        self.assertEqual(manifest["protocol"]["stocks"], 2115)
        self.assertEqual(manifest["protocol"]["primary_horizons"], [1, 5])
        families = {f["family_id"]: f for f in
                    (load_definition(DEFINITIONS / p) for p in manifest["definition_files"]["families"])}
        signals, names = set(), set()
        # Baselines must exist in the actual production code, not invented IDs.
        import factors
        from fea.spec import REGISTRY
        # ★ 2026-10-03 清理复审：部分 baseline 对象已按审批删除独立实现并退出注册表；
        #   这里要求 baseline 只能引用「当前注册表或已记录在案的清理删除集」，仍不得凭空造 ID。
        removed=set()
        # ★ 2026-10-06：原路径的决策文件已随证据归档迁入 artifacts/backups/（证据保留规则），
        #   测试按「主路径 → 两个归档路径」顺序查找；找不到时**显式断言失败**，不得静默当空集。
        _root=pathlib.Path(__file__).resolve().parents[1]
        _dec=next((p for p in (
            _root/'artifacts/research/legacy_gates_20261002/delivery/final_mechanism_review.json',
            _root/'artifacts/backups/research_evidence_20261003/delivery/final_mechanism_review.json',
            _root/'artifacts/backups/market_factor_removal_20261004/research_legacy_gates_20261002/delivery/final_mechanism_review.json',
        ) if p.exists()), None)
        self.assertIsNotNone(_dec, "机制决策文件缺失：baseline 参考校验无法进行（不得静默退化为空集）")
        removed={r['name'] for r in json.loads(_dec.read_text())['records'] if not r['approved_for_retention']}
        for family in families.values():
            self.assertEqual(family["trial_budget"], 4)
            self.assertEqual(len(family["allowed_variants"]), 4)
            self.assertTrue(set(family["baseline"]) <= (set(REGISTRY) | removed))
        for path in manifest["definition_files"]["candidates"]:
            definition = load_definition(DEFINITIONS / path)
            spec = build_spec(definition, batch.build, definition["name"])
            signals.add(definition["parameters"]["signal"])
            names.add(spec.name)
            self.assertEqual(set(definition["parameters"]), {"signal"})
            self.assertIn(definition["parameters"], families[definition["family_id"]]["allowed_variants"])
            nested = definition["parameters"]["signal"] in (
                "downside_impact_surprise", "upside_impact_fragility", "sell_shock_next_day_repair")
            self.assertEqual(spec.warmup_days, 240 if nested else 180)
            self.assertFalse(spec.is_label)
            self.assertEqual(spec.start, None)
            self.assertFalse(any("net_mf" in x for x in definition["source_fields"]))
            self.assertTrue(set(definition["closest_existing_baselines"]) <= (set(REGISTRY) | removed))
            self.assertTrue(definition["semantic_difference"])
            self.assertEqual(definition["pre_registered_direction"],
                             "higher_is_better" if spec.higher_is_better else "lower_is_better")
        self.assertEqual(names.__len__(), 16)
        self.assertEqual(signals, set(batch.SIGNALS))

    def test_unknown_or_searched_parameters_are_rejected(self):
        for parameters in ({}, {"signal": "unknown"},
                           {"signal": batch.SIGNALS[0], "window": 20},
                           {"signal": batch.SIGNALS[0], "direction": -1}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                batch.build(parameters)


class BatchCalculationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = BatchFixture()

    def test_all_signals_have_values_and_preserve_full_vs_cutoff_prefix(self):
        for signal in batch.SIGNALS:
            with self.subTest(signal=signal):
                full_ctx, short_ctx = self.fixture.context(), self.fixture.context(135)
                compute = batch.build({"signal": signal})
                full, short = compute(full_ctx), compute(short_ctx)
                self.assertEqual(full.shape, (180, 8))
                self.assertEqual(full.dtype, np.float64)
                self.assertGreater(np.isfinite(full[125:]).sum(), 300)
                self.assertGreater(np.abs(full[125:][np.isfinite(full[125:])]).max(), 1e-8)
                np.testing.assert_equal(full[:135].astype(np.float32), short.astype(np.float32))
                self.assertTrue(full_ctx.accessed <= {"stock_daily", "stock_adj_factor", "stock_main_fund_flow"})
                for _, columns, _ in full_ctx.up.reads:
                    self.assertFalse(any("net_mf" in c for c in columns))

    def test_current_target_is_excluded_from_fitting_coefficients(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(120, 2))
        second = rng.normal(size=x.shape)
        y = .3 + 1.2 * x - .7 * second + rng.normal(0., .1, size=x.shape)
        expected, sigma = batch._prediction(y, x, second)
        changed = y.copy()
        changed[95:] += 100.
        prediction, changed_sigma = batch._prediction(changed, x, second)
        np.testing.assert_equal(prediction[:96], expected[:96])
        np.testing.assert_equal(changed_sigma[:96], sigma[:96])
        # Independent OLS calculation for one current row and stock.
        a, z, target = x[35:95, 0], second[35:95, 0], y[35:95, 0]
        centered = np.column_stack([a - a.mean(), z - z.mean()])
        cov = centered.T @ centered / 60.
        ridge = 1e-6 * np.trace(cov) + 1e-12
        beta = np.linalg.solve(cov + ridge * np.eye(2), centered.T @ (target - target.mean()) / 60.)
        want = target.mean() + beta @ [x[95, 0] - a.mean(), second[95, 0] - z.mean()]
        self.assertAlmostEqual(float(expected[95, 0]), float(want), places=12)

    def test_no_trading_or_required_daily_missing_produces_nan(self):
        self.fixture.traded[130, 0] = False
        self.fixture.fields["amount"][131, 1] = np.nan
        for signal in batch.SIGNALS:
            with self.subTest(signal=signal):
                values = batch.build({"signal": signal})(self.fixture.context())
                self.assertTrue(np.isnan(values[130, 0]))
                if signal in batch.PRICE_SIGNALS + batch.FLOW_SIGNALS[:4] + ("block_buy_large_sell_close",):
                    self.assertTrue(np.isnan(values[131, 1]))

    def test_invalid_nontrading_raw_price_and_flow_perturbations_do_not_enter_history(self):
        fixture = self.fixture
        fixture.traded[130, 0] = False
        before = {s: batch.build({"signal": s})(fixture.context()) for s in batch.SIGNALS}
        for field in ("open", "high", "low", "close", "amount"):
            fixture.fields[field][130, 0] *= 100.
        fixture.fields["amount"][130, 0] = 0.
        flow_row = 130 * len(fixture.codes)
        for tier in batch.TIERS:
            fixture.flow.loc[flow_row, f"buy_{tier}_amount"] *= 100.
        for signal in batch.SIGNALS:
            with self.subTest(signal=signal):
                after = batch.build({"signal": signal})(fixture.context())
                self.assertTrue(np.isnan(after[130, 0]))
                np.testing.assert_equal(after, before[signal])

    def test_illegal_ohlc_prices_and_amount_do_not_enter_price_histories(self):
        fixture = self.fixture
        # Fix an independent invalidity (low <= 0) so all perturbed positive
        # fields remain an invalid OHLC source record in both calculations.
        fixture.fields["low"][130, 0] = 0.
        price_signals = batch.PRICE_SIGNALS + batch.FLOW_SIGNALS[:4] + ("block_buy_large_sell_close",)
        before = {s: batch.build({"signal": s})(fixture.context()) for s in price_signals}
        for field in ("open", "high", "close", "amount"):
            fixture.fields[field][130, 0] *= 100.
        for signal in price_signals:
            with self.subTest(signal=signal):
                after = batch.build({"signal": signal})(fixture.context())
                self.assertTrue(np.isnan(after[130, 0]))
                np.testing.assert_equal(after, before[signal])

    def test_missing_flow_row_is_nan_without_forward_fill(self):
        row = (self.fixture.flow.stock_code == self.fixture.codes[2]) & (
            self.fixture.flow.trade_date == pd.Timestamp(str(self.fixture.dates[140])).strftime("%Y-%m-%d"))
        self.fixture.flow = self.fixture.flow.loc[~row]
        for signal in batch.FLOW_SIGNALS:
            with self.subTest(signal=signal):
                value = batch.build({"signal": signal})(self.fixture.context())
                self.assertTrue(np.isnan(value[140, 2]))
                self.assertTrue(np.isfinite(value[141, 2]))

    def test_four_bucket_net_total_zero_does_not_erase_structure(self):
        frame = self.fixture.flow
        buy = frame[[f"buy_{t}_amount" for t in batch.TIERS]].sum(axis=1)
        sell = frame[[f"sell_{t}_amount" for t in batch.TIERS]].sum(axis=1)
        np.testing.assert_allclose(buy, sell, rtol=1e-14)
        hhi = batch.build({"signal": "buy_sell_concentration_gap"})(self.fixture.context())
        self.assertGreater(float(np.nanstd(hhi)), .05)
        side_price = batch.build({"signal": "big_buy_sell_price_gap"})(self.fixture.context())
        np.testing.assert_allclose(side_price, 1.01 / .99 - 1., atol=1e-14)
        # Scaling same-table units on both sides leaves the dimensionless signal.
        for column in frame.columns:
            if column.endswith("_amount"):
                frame[column] *= 1e4
            if column.endswith("_vol"):
                frame[column] *= 100.
        scaled = batch.build({"signal": "big_buy_sell_price_gap"})(self.fixture.context())
        np.testing.assert_allclose(scaled, side_price, atol=1e-14)

    def test_readable_empty_zero_net_and_invalid_volume_are_distinct(self):
        empty = FixtureSource(self.fixture.flow.iloc[:0])
        for signal in batch.FLOW_SIGNALS:
            with self.subTest(signal=signal):
                values = batch.build({"signal": signal})(self.fixture.context(source=empty))
                self.assertTrue(np.isnan(values).all())
        # Both sides identical with nonzero gross amount is an observed zero.
        for tier in batch.TIERS:
            self.fixture.flow[f"sell_{tier}_amount"] = self.fixture.flow[f"buy_{tier}_amount"]
        observed_zero = batch.build({"signal": "buy_sell_concentration_gap"})(self.fixture.context())
        np.testing.assert_equal(observed_zero, np.zeros((180, 8)))
        self.fixture.flow.loc[0, ["buy_lg_vol", "buy_elg_vol"]] = 0.
        invalid = batch.build({"signal": "big_buy_sell_price_gap"})(self.fixture.context())
        self.assertTrue(np.isnan(invalid[0, 0]))
        self.assertTrue(np.isfinite(invalid[0, 1]))

    def test_duplicate_stock_day_is_rejected(self):
        self.fixture.flow = pd.concat([self.fixture.flow, self.fixture.flow.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate stock/day"):
            batch.build({"signal": "buy_sell_concentration_gap"})(self.fixture.context())

    def test_fixed_pool_uses_current_members_and_enforces_sample_minimum(self):
        ctx = self.fixture.context(2)
        values = np.asarray([[1., 2., 3., 4., 5., 6., 7., 1000.], [1.] * 8])
        ctx.universe[:, -1] = False
        got = batch._pool_relative(ctx, values)
        np.testing.assert_equal(got[0, :7], np.arange(1., 8.) - 4.)
        self.assertTrue(np.isnan(got[:, -1]).all())
        ctx.universe[1, 2:] = False
        self.assertTrue(np.isnan(batch._pool_relative(ctx, values)[1]).all())

    def test_nested_impact_needs_120_history_and_repair_an_extra_day(self):
        fixture = BatchFixture(count=400)
        target, tail = 300, 12
        for signal in ("downside_impact_surprise", "upside_impact_fragility", "sell_shock_next_day_repair"):
            with self.subTest(signal=signal):
                compute = batch.build({"signal": signal})
                required = 121 if signal == "sell_shock_next_day_repair" else 120
                expanded = compute(fixture.context(target + tail))[target:]
                sufficient = compute(fixture.context(required + tail, start=target - required))[required:]
                np.testing.assert_allclose(sufficient, expanded, rtol=1e-10, atol=1e-11, equal_nan=True)
                # min_count=12 can emit plausible finite shocks from a clipped
                # outer history. Nonempty alone does not establish full warmup.
                too_short = compute(fixture.context(100 + tail, start=target - 100))[100:]
                common = np.isfinite(expanded) & np.isfinite(too_short)
                self.assertTrue(common.any())
                self.assertGreater(float(np.max(np.abs(expanded[common] - too_short[common]))), 1e-5)

    def test_actual_parquet_flow_read_has_cutoff_and_exact_grid_missingness(self):
        base = ROOT / "artifacts/tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base, prefix="batch_") as folder:
            project = Path(folder)
            raw = project / "raw/stock_main_fund_flow/year=2019"
            raw.mkdir(parents=True)
            self.fixture.flow.to_parquet(raw / "data.parquet", index=False)
            up = Upstream(project / "raw", project)
            up.audit_cutoff = int(self.fixture.dates[134])
            cutoff = batch.build({"signal": "buy_sell_concentration_gap"})(self.fixture.context(source=up))
            self.assertTrue(np.isfinite(cutoff[:135]).all())
            self.assertTrue(np.isnan(cutoff[135:]).all())
            up.audit_cutoff = int(self.fixture.dates[-1])
            extended = batch.build({"signal": "buy_sell_concentration_gap"})(self.fixture.context(source=up))
            np.testing.assert_equal(cutoff[:135], extended[:135])
            self.assertTrue(np.isfinite(extended).all())

    def test_real_annual_engine_long_prewarmup_short_start_and_cross_year_increment(self):
        """All mechanisms use real prices/flow files, annual anchors and ST masks."""
        from fea.prices import LEVELS, FLOWS, RATIOS
        fixture = BatchFixture(count=600)
        base = ROOT / "artifacts/tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base, prefix="batch_year_") as folder:
            project = Path(folder)
            conf = project / "conf"
            conf.mkdir()
            frozen = conf / "universe.tsv"
            frozen.write_text("\n".join(c + "\t2000-01-01\t" for c in fixture.codes), encoding="utf-8")
            cfg = Config({"default_start": "2018-01-01", "paths": {"upstream": "raw"},
                          "compute": {"revision_days": 10},
                          "rank": {"winsor": [.01, .99], "min_cross_section": 3},
                          "universe": {"main_board_prefixes": ["000"], "exclude_st": True,
                                       "min_listed_days": 0, "frozen_list": "conf/universe.tsv"}}, project)
            dates = pd.to_datetime(fixture.dates.astype(str)).strftime("%Y-%m-%d")
            grid = pd.DataFrame({"trade_date": np.repeat(dates, len(fixture.codes)),
                                 "stock_code": np.tile(fixture.codes, len(dates))})
            def write(name, frame):
                path = project / "raw" / name
                path.mkdir(parents=True)
                frame.to_parquet(path / "data.parquet", index=False)
            write("basic_calendar", pd.DataFrame({"date": dates, "is_open": True}))
            write("stock_list", pd.DataFrame({"stock_code": fixture.codes,
                                               "list_date": "2000-01-01", "delist_date": None}))
            write("stock_st_info", pd.DataFrame({"stock_code": [fixture.codes[0], fixture.codes[0]],
                                                  "trade_date": ["2020-08-12", "2021-02-01"]}))
            daily = grid.copy()
            for column in LEVELS:
                daily[column] = fixture.fields.get(column, fixture.fields["close"]).ravel()
            for column in FLOWS:
                daily[column] = fixture.fields["vol" if column.endswith("vol") else "amount"].ravel()
            for column in RATIOS:
                daily[column] = .1
            write("stock_daily", daily)
            write("stock_adj_factor", grid.assign(adj_factor=fixture.fields["adj_factor"].ravel()))
            write("stock_finance", grid.assign(total_share=10000., float_share=9000., free_share=8000.))
            write("stock_main_fund_flow", fixture.flow)
            definitions = [load_definition(p) for p in sorted((DEFINITIONS / "candidates").glob("*.yaml"))]
            specs = [build_spec(d, batch.build, d["name"]) for d in definitions]
            full_cfg, late_cfg, incr_cfg = [cfg.sandbox(project / name) for name in ("full", "late", "increment")]
            def calculate(current_cfg, start, end):
                current_cfg.raw["_audit_cutoff"] = end
                engine = Engine(current_cfg)
                engine._deriv_fields_run = frozenset()
                days = engine.cal.between(start, end)
                for spec in specs:
                    for year in sorted(set(days // 10000)):
                        task_days = days[days // 10000 == year]
                        self.assertEqual(engine._anchor_for(spec, task_days), int(year) * 10000 + 101)
                        engine.run_year(spec, int(year), task_days)
                return engine
            full_engine = calculate(full_cfg, 20200102, 20210331)
            calculate(late_cfg, 20200601, 20210331)
            calculate(incr_cfg, 20200601, 20201230)
            calculate(incr_cfg, 20201231, 20210331)
            # The pool-relative control excludes an actual current ST event.
            panel = Panel(np.asarray([20200812]), full_engine.codes)
            ctx = FactorContext(panel, None, full_engine.up, full_cfg,
                                full_engine.universe_for(panel))
            values = np.arange(8., dtype=float)[None, :]
            relative = batch._pool_relative(ctx, values)
            self.assertTrue(np.isnan(relative[0, 0]))
            np.testing.assert_equal(relative[0, 1:], np.arange(1., 8.) - 4.)
            for spec in specs:
                with self.subTest(signal=spec.name):
                    def frames(current_cfg):
                        return pd.concat([pd.read_parquet(current_cfg.factors_dir / spec.name / f"year={year}/data.parquet")
                                          for year in (2020, 2021)], ignore_index=True)
                    full, late, incremental = frames(full_cfg), frames(late_cfg), frames(incr_cfg)
                    pd.testing.assert_frame_equal(full[full.trade_date >= "2020-06-01"].reset_index(drop=True), late)
                    pd.testing.assert_frame_equal(late, incremental)
                    self.assertTrue(np.isfinite(full[full.trade_date == "2020-01-02"].value).all())
                    for st_date in ("2020-08-12", "2021-02-01"):
                        event = full[(full.trade_date == st_date) & (full.stock_code == fixture.codes[0])]
                        self.assertEqual(len(event), 1)
                        self.assertTrue(event.value.isna().all())


if __name__ == "__main__":
    unittest.main()
