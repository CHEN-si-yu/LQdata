"""收盘价附近的筹码供给墙：原始筹码档位到日频摘要。

`stock_cyq_chips.price` 与 `stock_daily.close` 都是当日未复权价格。只计
``close < price <= 1.05 * close`` 的有效档位，除以同一股票日全部有效档位的
``sum(percent)``。上游的 percent 总和经常不是 100，不能直接除以 100。

单独使用 `data/derived/chipwall/`，避免改动现有 ChipLayer 的版本并重建
已有筹码因子。Engine 在 fork 因子 worker 前确保本层的年度缓存。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dates import int_to_str_vec, series_to_int
from .derived import DerivedCache


FIELDS = ("psum", "n_levels", "near_overhang_5pct")
COLUMNS = ("trade_date", "stock_code", *FIELDS)
_CODE_WIDTH = 100_000


class ChipWallLayer(DerivedCache):
    name = "chipwall"
    version = 1

    def source_fingerprint(self, year: int) -> dict:
        chips = self.cfg.upstream / "stock_cyq_chips" / f"year={year}" / "data.parquet"
        if not chips.exists():
            return {"exists": False}
        import pyarrow.parquet as pq

        raw_stat = chips.stat()
        daily = self.cfg.upstream / "stock_daily" / f"year={year}" / "data.parquet"
        daily_stat = daily.stat() if daily.exists() else None
        return {
            "exists": True,
            "rows": pq.ParquetFile(chips).metadata.num_rows,
            "bytes": raw_stat.st_size,
            "mtime_ns": raw_stat.st_mtime_ns,
            "daily": [daily_stat.st_size, daily_stat.st_mtime_ns] if daily_stat else None,
        }

    def _close_lookup(self, year: int, up) -> tuple[np.ndarray, np.ndarray]:
        """返回按 (日期, 池内代码下标) 排序的当日原始收盘价。"""
        daily = up.read(
            "stock_daily", columns=["trade_date", "stock_code", "close"],
            years=(year, year), use_cache=False,
        )
        if daily.empty:
            return np.empty(0, np.int64), np.empty(0, np.float64)

        code_pos = pd.Series(np.arange(self.codes.size), index=self.codes)
        ci = code_pos.reindex(daily["stock_code"].to_numpy()).to_numpy()
        day = series_to_int(daily["trade_date"])
        close = pd.to_numeric(daily["close"], errors="coerce").to_numpy(np.float64)
        valid = np.isfinite(ci) & (day > 0) & np.isfinite(close) & (close > 0)
        if not valid.any():
            return np.empty(0, np.int64), np.empty(0, np.float64)

        key = day[valid].astype(np.int64) * _CODE_WIDTH + ci[valid].astype(np.int64)
        close = close[valid]
        order = np.argsort(key, kind="stable")
        key, close = key[order], close[order]
        first = np.concatenate(([True], key[1:] != key[:-1]))
        return key[first], close[first]

    def build_year(self, year: int, up) -> pd.DataFrame:
        src = self.cfg.upstream / "stock_cyq_chips" / f"year={year}" / "data.parquet"
        if not src.exists():
            return pd.DataFrame(columns=COLUMNS)
        close_key, close_value = self._close_lookup(year, up)
        frames = []
        # 约 6,000 万行/年；与现有 ChipLayer 一样按季度读，内存不叠全年。
        for month in (1, 4, 7, 10):
            lo = f"{year}-{month:02d}-01"
            hi = f"{year + 1}-01-01" if month == 10 else f"{year}-{month + 3:02d}-01"
            raw = pd.read_parquet(
                src,
                columns=["trade_date", "stock_code", "price", "percent"],
                filters=[
                    ("trade_date", ">=", lo), ("trade_date", "<", hi),
                    ("stock_code", "in", self.codes.tolist()),
                ],
            )
            frame = self._aggregate_frame(raw, close_key, close_value)
            if not frame.empty:
                frames.append(frame)
            del raw, frame
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)

    def _aggregate_frame(
        self, raw: pd.DataFrame, close_key: np.ndarray, close_value: np.ndarray,
    ) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame(columns=COLUMNS)

        code_pos = pd.Series(np.arange(self.codes.size), index=self.codes)
        ci = code_pos.reindex(raw["stock_code"].to_numpy()).to_numpy()
        day = series_to_int(raw["trade_date"])
        price = pd.to_numeric(raw["price"], errors="coerce").to_numpy(np.float64)
        pct = pd.to_numeric(raw["percent"], errors="coerce").to_numpy(np.float64)
        valid = (np.isfinite(ci) & (day > 0) & np.isfinite(price) & (price > 0)
                 & np.isfinite(pct) & (pct >= 0))
        if not valid.any():
            return pd.DataFrame(columns=COLUMNS)

        ci = ci[valid].astype(np.int64)
        day, price, pct = day[valid], price[valid], pct[valid]
        key = day.astype(np.int64) * _CODE_WIDTH + ci
        order = np.argsort(key, kind="stable")
        key, day, ci = key[order], day[order], ci[order]
        price, pct = price[order], pct[order]
        starts = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
        ends = np.concatenate((starts[1:], [key.size]))
        n_levels = ends - starts
        group_key = key[starts]

        close = np.full(group_key.size, np.nan, dtype=np.float64)
        if close_key.size:
            pos = np.searchsorted(close_key, group_key)
            clipped = np.clip(pos, 0, close_key.size - 1)
            hit = close_key[clipped] == group_key
            close[hit] = close_value[clipped[hit]]

        # reduceat 对连续的 (股票, 日期) 段求和。close 只按该段重复，保持未复权同口径。
        close_row = np.repeat(close, n_levels)
        near = ((price > close_row)
                & (price <= np.nextafter(1.05 * close_row, np.inf)))
        psum = np.add.reduceat(pct, starts)
        wall_sum = np.add.reduceat(np.where(near, pct, 0.0), starts)
        good = (np.isfinite(close) & (close > 0) & np.isfinite(psum)
                & (psum > 0) & (n_levels >= 2))
        wall = np.full(group_key.size, np.nan, dtype=np.float64)
        wall[good] = np.clip(wall_sum[good] / psum[good], 0.0, 1.0)

        return pd.DataFrame({
            "trade_date": int_to_str_vec(day[starts]),
            "stock_code": self.codes[ci[starts]],
            "psum": psum,
            "n_levels": n_levels,
            "near_overhang_5pct": wall,
        })[list(COLUMNS)]

    def panel(self, panel, field: str) -> np.ndarray:
        if field not in FIELDS:
            raise KeyError(f"筹码供给墙层不认识字段 {field!r}。可用：{FIELDS}")
        key = (int(panel.dates[0]), int(panel.dates[-1]), panel.C, field)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        y0, y1 = int(panel.dates[0]) // 10000, int(panel.dates[-1]) // 10000
        self.ensure(y0, y1)
        frame = self._read_all(y0, y1)
        if frame.empty:
            out = panel.empty(dtype=np.float64)
        else:
            code_pos = pd.Series(np.arange(panel.C), index=panel.codes)
            ci = code_pos.reindex(frame["stock_code"].to_numpy()).to_numpy()
            valid = np.isfinite(ci)
            vals = pd.to_numeric(frame[field], errors="coerce").to_numpy(np.float64)
            out = panel.place(
                ci[valid].astype(np.int64),
                series_to_int(frame["trade_date"])[valid],
                vals[valid],
            )
        self._cache[key] = out
        return out
