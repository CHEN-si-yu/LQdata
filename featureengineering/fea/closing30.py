"""尾盘 30 分钟 5 分钟棒 -> 日频派生层。

T 日收盘后可用。厂商时间戳是区间右端：14:35 的 open 是 14:30
价格，最后六根 14:35..15:00 覆盖 14:30..15:00。

独立于已有 intraday 层，避免改变它的缓存版本与历史因子。仅使用金额
（元），不使用在 2025-12 改过单位的 vol。停牌、全天零成交、缺棒、
重复时间戳和金额缺测均不产出信号。价格型字段还要求尾盘有成交。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .dates import int_to_str_vec, series_to_int
from .derived import DerivedCache

log = logging.getLogger("fea.closing30")


def _clock_grid() -> np.ndarray:
    """标准 48 根右端时间戳，午间不含交易棒。"""
    minutes = np.r_[np.arange(9 * 60 + 35, 11 * 60 + 31, 5),
                    np.arange(13 * 60 + 5, 15 * 60 + 1, 5)]
    return (minutes // 60 * 100 + minutes % 60).astype(np.int32)


EXPECTED_HM = _clock_grid()
assert EXPECTED_HM.size == 48

FIELDS = ["amt_day", "amt_tail30", "open_tail30", "high_tail30",
          "close_tail30", "n_bars", "grid_ok"]


class Closing30Layer(DerivedCache):
    name = "closing30"
    version = 1

    def source_fingerprint(self, year: int) -> dict:
        p = self.cfg.upstream / "stock_history_5min" / f"year={year}" / "data.parquet"
        if not p.exists():
            return {"exists": False}
        import pyarrow.parquet as pq
        st = p.stat()
        return {"exists": True, "rows": pq.ParquetFile(p).metadata.num_rows,
                "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}

    def build_year(self, year: int, up) -> pd.DataFrame:
        src = self.cfg.upstream / "stock_history_5min" / f"year={year}" / "data.parquet"
        if not src.exists():
            return pd.DataFrame()
        frames = []
        # 季度块控制峰值内存；先按冻结股票池过滤。
        for month in (1, 4, 7, 10):
            lo = f"{year}-{month:02d}-01"
            hi = f"{year + 1}-01-01" if month == 10 else f"{year}-{month + 3:02d}-01"
            df = pd.read_parquet(src, columns=["stock_code", "trade_time", "open",
                                               "high", "close", "amount"],
                                 filters=[("trade_time", ">=", lo),
                                          ("trade_time", "<", hi),
                                          ("stock_code", "in", self.codes.tolist())])
            frame = self._aggregate_frame(df, year)
            if not frame.empty:
                frames.append(frame)
            del df
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _aggregate_frame(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        cpos = pd.Series(np.arange(self.codes.size), index=self.codes)
        ci = cpos.reindex(df["stock_code"].to_numpy()).to_numpy()
        keep = np.isfinite(ci)
        if not keep.any():
            return pd.DataFrame()
        df = df.loc[keep]
        ci = ci[keep].astype(np.int64)

        tt = df["trade_time"].astype("string")
        di = series_to_int(tt.str.slice(0, 10))
        hm = (tt.str.slice(11, 13).astype("int32") * 100
              + tt.str.slice(14, 16).astype("int32")).to_numpy(np.int32)
        o = pd.to_numeric(df["open"], errors="coerce").to_numpy(np.float64)
        h = pd.to_numeric(df["high"], errors="coerce").to_numpy(np.float64)
        c = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)
        a = pd.to_numeric(df["amount"], errors="coerce").to_numpy(np.float64)

        # 按 (股票, 日期, 时钟) 排序，同一股票日形成连续段。
        day_code = pd.factorize(di, sort=False)[0]
        key = ci * 1_000_003 + day_code
        order = np.lexsort((hm, key))
        key, ci, di, hm = key[order], ci[order], di[order], hm[order]
        o, h, c, a = o[order], h[order], c[order], a[order]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        ends = np.r_[starts[1:], key.size]
        n_bars = ends - starts

        # 长度 48 之外还逐根校验时钟，以拒绝“少一根又重复一根”的伪完整日。
        pos_map = np.full(2401, -1, dtype=np.int16)
        pos_map[EXPECTED_HM] = np.arange(48, dtype=np.int16)
        pos = np.where((hm >= 0) & (hm <= 2400),
                       pos_map[np.clip(hm, 0, 2400)], -1)
        step_ok = np.ones(key.size, dtype=bool)
        step_ok[1:] = np.diff(pos) == 1
        step_ok[starts] = True
        grid_ok = ((n_bars == 48) & (pos[starts] == 0) &
                   (pos[ends - 1] == 47) &
                   (np.add.reduceat((~step_ok).astype(np.int8), starts) == 0))

        amount_ok = np.isfinite(a) & (a >= 0)
        bad_amount = np.add.reduceat((~amount_ok).astype(np.int8), starts) > 0
        a0 = np.where(amount_ok, a, 0.0)
        day_amount = np.add.reduceat(a0, starts)

        # 对无效短日也安全索引；真正的可用性由 grid_ok 控制。
        tail_idx = np.minimum(starts[:, None] + np.arange(42, 48),
                              (ends - 1)[:, None])
        tail_amount = a0[tail_idx].sum(axis=1)
        tail_o, tail_h, tail_c = o[tail_idx], h[tail_idx], c[tail_idx]
        tail_price_ok = (np.isfinite(tail_o) & (tail_o > 0) &
                         np.isfinite(tail_h) & (tail_h > 0) &
                         np.isfinite(tail_c) & (tail_c > 0)).all(axis=1)
        # 厂商有少量 high 低于本棒 open/close 的异常；扩张范围保证回落非负。
        tail_high = np.maximum(np.maximum(tail_o, tail_h), tail_c).max(axis=1)
        valid_amount = grid_ok & ~bad_amount & (day_amount > 0)
        valid_price = valid_amount & tail_price_ok & (tail_amount > 0)

        log.info("closing30 %d：%d 棒 -> %d 股票日；完整时钟 %d，有效成交额 %d",
                 year, key.size, starts.size, int(grid_ok.sum()), int(valid_amount.sum()))
        return pd.DataFrame({
            "trade_date": int_to_str_vec(di[starts]),
            "stock_code": self.codes[ci[starts]],
            "amt_day": np.where(valid_amount, day_amount, np.nan),
            "amt_tail30": np.where(valid_amount, tail_amount, np.nan),
            "open_tail30": np.where(valid_price, tail_o[:, 0], np.nan),
            "high_tail30": np.where(valid_price, tail_high, np.nan),
            "close_tail30": np.where(valid_price, tail_c[:, -1], np.nan),
            "n_bars": n_bars.astype(np.float64),
            "grid_ok": grid_ok.astype(np.float64),
        })

    def panel(self, panel, field: str) -> np.ndarray:
        if field not in FIELDS:
            raise KeyError(f"closing30 层不认识字段 {field!r}。可用：{FIELDS}")
        key = (int(panel.dates[0]), int(panel.dates[-1]), panel.C, field)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        y0 = int(str(panel.dates[0])[:4])
        y1 = int(str(panel.dates[-1])[:4])
        self.ensure(y0, y1)
        df = self._read_all(y0, y1)
        if df.empty:
            out = panel.empty()
        else:
            cpos = pd.Series(np.arange(panel.C), index=panel.codes)
            ci = cpos.reindex(df["stock_code"].to_numpy()).to_numpy()
            keep = np.isfinite(ci)
            from .prices import day_ints
            values = pd.to_numeric(df[field], errors="coerce").to_numpy(np.float64)
            out = panel.place(ci[keep].astype(np.int64),
                              day_ints(df["trade_date"])[keep], values[keep])
        self._cache[key] = out
        return out
