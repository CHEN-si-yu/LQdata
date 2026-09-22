"""覆盖报告期对齐、修订时点、源表占位及稳定性统计的真实边界。"""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fea.context import FactorContext
from fea.deriv import Derivative, IND_SAFE, load_vintages, period_of
from fea.panel import Panel
from fea.spec import REGISTRY
from factors import quarterly_quality as qf


def raw_reports():
    ends = [20240331, 20240630, 20240930, 20241231, 20250331]
    anns = [20240420, 20240820, 20241020, 20250320, 20250420]
    d = pd.DataFrame({"stock_code": ["A"] * 5, "end_date_i": ends, "pit": anns})
    d["period"] = period_of(d.end_date_i.to_numpy())
    for i, field in enumerate(qf.FIELDS):
        d[field] = np.array([1., 2., 3., 4., 8.]) + i
    return d


def context(vintages, days):
    p = Panel(np.asarray(days, dtype=np.int32), np.array(["A"]))
    return FactorContext(p, Derivative(vintages.copy(), p.codes), None, None,
                         np.ones(p.shape, dtype=bool))


class QuarterlyQualityTests(unittest.TestCase):
    def test_approved_fields_only_and_complete_dependencies(self):
        self.assertTrue(set(qf.FIELDS) <= set(IND_SAFE))
        self.assertNotIn("roe", IND_SAFE)
        self.assertNotIn("q_eps", IND_SAFE)
        specs = [s for s in REGISTRY.values() if s.group == "quarterly_quality"]
        self.assertEqual(len(specs), 12)
        for spec in specs:
            self.assertEqual(spec.deps, ("stock_financial_indicator",))
            self.assertEqual(set(spec.fin_fields), set(qf.FIELDS))
            self.assertFalse(spec.is_label)
            self.assertTrue(spec.note)

    def test_closed_form_values_use_report_period_lags_and_percent_units(self):
        c = context(raw_reports(), [20250420, 20250421])
        expected = {
            "qf_roe_yoy_change": .07,
            "qf_core_roe_yoy_change": .07,
            "qf_roa_yoy_change": .07,
            "qf_sales_growth_accel": .04,
            "qf_sales_growth_floor_4q": .06,
            "qf_sales_growth_vol_4q": np.std([6, 7, 8, 12]) / 100,
            "qf_core_roe_floor_4q": .03,
            "qf_core_roe_vol_4q": np.std([3, 4, 5, 9]) / 100,
            "qf_noncore_roe_gap": -.01,
            "qf_cash_margin_yoy_change": .07,
            "qf_cash_margin_floor_4q": .05,
            "qf_cash_margin_vol_4q": np.std([5, 6, 7, 11]) / 100,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                out = REGISTRY[name].fn(c)
                self.assertEqual(out.shape, (2, 1))
                np.testing.assert_allclose(out, value, atol=1e-8)

    def test_future_report_and_revision_do_not_rewrite_history(self):
        raw = raw_reports()
        revision = raw.iloc[[0]].copy()
        revision["pit"] = 20250601
        revision["q_roe"] = 5.
        full = pd.concat([raw, revision], ignore_index=True)
        for spec in REGISTRY.values():
            if spec.group != "quarterly_quality":
                continue
            past = [20250319, 20250320, 20250419]
            np.testing.assert_array_equal(spec.fn(context(full, past)),
                                          spec.fn(context(raw[raw.pit <= past[-1]], past)))
        out = qf.qf_roe_yoy_change(context(full, [20250530, 20250601]))
        np.testing.assert_allclose(out[:, 0], [.07, .03], atol=1e-8)

    def test_missing_quarter_cannot_be_replaced_with_previous_available(self):
        raw = raw_reports().drop(index=2)
        c = context(raw, [20250420])
        self.assertTrue(np.isnan(qf.qf_core_roe_floor_4q(c)).all())
        self.assertTrue(np.isnan(qf.qf_sales_growth_vol_4q(c)).all())
        np.testing.assert_allclose(qf.qf_roe_yoy_change(c), .07)

    def test_missing_revision_stays_missing(self):
        raw = raw_reports()
        revision = raw.iloc[[-1]].copy()
        revision["pit"] = 20250501
        revision["q_dt_roe"] = np.nan
        c = context(pd.concat([raw, revision]), [20250430, 20250501])
        out = qf.qf_core_roe_yoy_change(c)[:, 0]
        self.assertAlmostEqual(out[0], .07)
        self.assertTrue(np.isnan(out[1]))

    def test_joint_zero_placeholder_propagates_through_lag_without_fallback(self):
        raw = raw_reports()
        raw.loc[1, list(qf.FIELDS)] = 0.
        c = context(raw, [20240820, 20250420])
        self.assertTrue(np.isnan(qf.qf_noncore_roe_gap(c)[0, 0]))
        self.assertTrue(np.isnan(qf.qf_core_roe_floor_4q(c)[1, 0]))
        np.testing.assert_allclose(qf.qf_roe_yoy_change(c)[1, 0], .07)

    def test_individual_zero_negative_and_nonfinite(self):
        raw = raw_reports()
        raw.loc[4, "q_dt_roe"] = 0.
        raw.loc[4, "q_ocf_to_sales"] = -10.
        raw.loc[4, "q_npta"] = np.inf
        c = context(raw, [20250420])
        np.testing.assert_allclose(qf.qf_noncore_roe_gap(c), .08)
        np.testing.assert_allclose(qf.qf_cash_margin_floor_4q(c), -.1)
        self.assertTrue(np.isnan(qf.qf_roa_yoy_change(c)).all())

    def test_physical_source_cutoff_and_invalid_announcement(self):
        d = raw_reports()
        d["ann_date"] = pd.to_datetime(d.pit.astype(str)).dt.strftime("%Y-%m-%d")
        d["end_date"] = pd.to_datetime(d.end_date_i.astype(str)).dt.strftime("%Y-%m-%d")
        invalid = d.iloc[[0]].copy()
        invalid["ann_date"] = "1970-01-01"
        raw = pd.concat([d, invalid], ignore_index=True)
        up = SimpleNamespace(read=lambda *a, **kw: raw[kw["columns"]].copy(), audit_cutoff=20250419)
        vt = load_vintages(up, 2024, 2025, fields=frozenset(qf.FIELDS))
        self.assertEqual(len(vt), 4)
        self.assertLessEqual(vt.pit.max(), 20250419)
        self.assertTrue(np.isnan(qf.qf_roe_yoy_change(context(vt, [20250419]))).all())


if __name__ == "__main__":
    unittest.main()
