"""小样本检验供给墙的价格边界、归一化和缺失语义。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fea.chipwall import ChipWallLayer
from fea.context import FactorContext
from fea.panel import Panel
from fea.upstream import Upstream
from factors.chipwall import chip_near_overhang_5pct


class ChipWallTests(unittest.TestCase):
    def test_near_overhang_normalization_and_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            for name in ("stock_cyq_chips", "stock_daily"):
                (source / name / "year=2018").mkdir(parents=True)

            chips = pd.DataFrame([
                # 分母是 150，不是 100；close 本身不算，1.05*close 的边界算。
                ("2018-01-02", "600000.SH", 9.0, 40.0),
                ("2018-01-02", "600000.SH", 10.0, 20.0),
                ("2018-01-02", "600000.SH", 10.2, 30.0),
                ("2018-01-02", "600000.SH", 10.5, 50.0),
                ("2018-01-02", "600000.SH", 10.6, 10.0),
                ("2018-01-02", "600000.SH", -1.0, 99.0),
                ("2018-01-02", "600001.SH", 9.0, 2.0),
                ("2018-01-02", "600001.SH", 10.0, 3.0),
                ("2018-01-03", "600000.SH", 10.1, 0.0),
                ("2018-01-03", "600000.SH", 10.2, 0.0),
                ("2018-01-04", "600000.SH", 10.1, 1.0),
                ("2018-01-04", "600000.SH", 10.2, 1.0),
            ], columns=["trade_date", "stock_code", "price", "percent"])
            daily = pd.DataFrame([
                ("2018-01-02", "600000.SH", 10.0),
                ("2018-01-02", "600001.SH", 10.0),
                ("2018-01-03", "600000.SH", 10.0),
                ("2018-01-05", "600000.SH", 10.0),
            ], columns=["trade_date", "stock_code", "close"])
            chips.to_parquet(source / "stock_cyq_chips" / "year=2018" / "data.parquet", index=False)
            daily.to_parquet(source / "stock_daily" / "year=2018" / "data.parquet", index=False)

            cfg = SimpleNamespace(root=root / "project", upstream=source, compression="zstd")
            up = Upstream(source, cfg.root)
            codes = np.array(["600000.SH", "600001.SH"])
            panel = Panel(np.array([20180102, 20180103, 20180104, 20180105]), codes)
            layer = ChipWallLayer(up, cfg, codes)
            out = layer.panel(panel, "near_overhang_5pct")

            self.assertAlmostEqual(out[0, 0], 80.0 / 150.0)
            self.assertEqual(out[0, 1], 0.0)
            self.assertTrue(np.isnan(out[1, 0]))  # percent 总和为 0
            self.assertTrue(np.isnan(out[2, 0]))  # 无当日 close，不沿用旧收盘价
            self.assertTrue(np.isnan(out[3, 0]))  # 无当日筹码行
            self.assertTrue(np.isnan(out[1:, 1]).all())

            product = cfg.root / "data" / "derived" / "chipwall" / "year=2018" / "data.parquet"
            self.assertTrue(product.exists())
            fresh = ChipWallLayer(up, cfg, codes).panel(panel, "near_overhang_5pct")
            np.testing.assert_allclose(out, fresh, equal_nan=True)

            ctx = FactorContext(
                panel, None, up, cfg, np.ones(panel.shape, dtype=bool), chipwall=layer,
            )
            np.testing.assert_allclose(chip_near_overhang_5pct(ctx), out, equal_nan=True)
            self.assertEqual(ctx.accessed, {"stock_cyq_chips", "stock_daily"})


if __name__ == "__main__":
    unittest.main()
