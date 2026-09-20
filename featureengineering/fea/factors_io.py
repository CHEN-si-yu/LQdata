"""读已落盘的因子 —— 供「因子耦合」类因子使用（Class 5）。

耦合因子的定义就是「其它因子的组合」（如 `(bp + momentum_20) / 2`）。
它们必须能拿到其它因子的 **value** 面板，并且**切到当前自己正在算的 (T, C)**。

三个必须做对的地方：
  1. **缺失因子**：耦合因子的依赖可能还没算出来。返回全 NaN 而不是抛错 ——
     让下游在 `check`/`eval` 里看到"这个因子今天是空的"，而不是整个 run 崩掉。
  2. **面板对齐**：别的因子的落盘日期可能与自己不完全一致（起点不同、上游延迟不同），
     必须按 (日期, 股票) 精确落格，**不能**用 asof 前向填充 ——
     前向填充会把「昨天没有值」当成「沿用昨天的值」，把耦合因子的覆盖度虚高。
  3. **PIT**：读的是 T 日的因子值，而因子值本身已是「T 日收盘后可得」，
     所以耦合不引入额外的前视。前提是**被读的因子自己 PIT 正确**。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import store
from .dates import series_to_int

log = logging.getLogger("fea.factors_io")


class FactorIO:
    def __init__(self, cfg, cal):
        self.cfg = cfg
        self.cal = cal
        self._cache: dict[tuple, pd.DataFrame] = {}

    def _read(self, name: str, y0: int, y1: int) -> pd.DataFrame:
        key = (name, y0, y1)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        frames = []
        for y in range(max(y0, int(self.cfg.default_start[:4])), y1 + 1):
            df = store.read_year(self.cfg.factors_dir, name, y)
            if len(df):
                df = df[df["trade_date"] >= self.cfg.default_start]
                frames.append(df[["trade_date", "stock_code", "value"]])
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["trade_date", "stock_code", "value"])
        self._cache[key] = out
        return out

    def exists(self, name: str) -> bool:
        return bool(store.factor_years(self.cfg.factors_dir, name))

    def load(self, name: str, panel) -> np.ndarray:
        """把因子 `name` 的 value 落到 `panel` 的 (T,C) 网格上（精确匹配，缺则 NaN）。"""
        y0 = int(str(panel.dates[0])[:4])
        y1 = int(str(panel.dates[-1])[:4])
        df = self._read(name, y0, y1)
        if df.empty:
            log.warning("耦合因子读到空因子 %s（可能还没算）-> 全 NaN", name)
            return panel.empty()

        codes = pd.Series(np.arange(panel.C), index=panel.codes)
        ci = codes.reindex(df["stock_code"].to_numpy()).to_numpy()
        keep = np.isfinite(ci)
        vals = pd.to_numeric(df["value"], errors="coerce").to_numpy()[keep]
        # 只保留有值的行：panel.place 用 fill=NaN，所以 0 值的行也照落
        good = np.isfinite(vals)
        return panel.place(ci[keep][good].astype(np.int64),
                           series_to_int(df["trade_date"])[keep][good],
                           vals[good])
