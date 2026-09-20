"""筹码峰 → 日频摘要（派生缓存）。

## 为什么要预聚合

`stock_cyq_chips`：**6.24 亿行 / 613 MB / 3484 只 / 2111 个交易日**，
每 (股票, 日) 约 85 个价格档。34 个筹码因子若各自扫一遍原始表 = 34 × 6.24 亿行，
**不可行**。这里一次性聚合成每 (股票, 日) 一行的摘要表（约 735 万行 / 200 MB），
之后每个因子只读它。

实测：单年分区（约 6300 万行）读 3.9s / 3.2 GB，加权聚合 10.7s。
9 个分区约 **2~4 分钟**一次，之后完全免费。

## 实测的两个数据坑

1. **`percent` 每个股票日不保证和为 100**：实测中位数 100.02、**p99 = 182.9**
   （上游疑似把两条价格阶梯叠加了）。不归一化的话 1% 的格子误差可达 83%。
   → **一律按当日 `Σpercent` 归一化**，归一后断言和为 1。
2. **只覆盖主板**（2026-09-14 提速时只下了 3484 只），起点 **2018-01-02**。
   → 筹码因子一律 `start="2018-01-02"`。

## 产出字段（(trade_date, stock_code) 一行）

    psum      当日 Σpercent（归一化前的原始和，用于诊断）
    n_levels  价格档数
    mean      加权均价（用归一化后的权重）
    p10/p25/p50/p75/p90   加权分位数
    std/skew/kurt         加权标准差 / 偏度 / 峰度
    cr3       前 3 档占的筹码比例（集中度）
    entropy   归一化 Shannon 熵（0=全集中一档，1=完全分散）
    gini      基尼系数
    iqr       p75 − p10（相对中位价的宽度）
    width     (p90 − p10) / p50
    mode      众数价（percent 最大的档）
    peak_purity  众数档的权重占比（峰纯度）
    n_peaks   局部极大值个数（筹码峰个数）
    peak_gap  最大两个峰的价差 / p50（双峰分离度）
    w_mean    加权均价 vs 现价的偏离相关的原料（见下）
    below_close  现价以下的筹码占比 = **获利盘比例**
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .derived import DerivedCache
from .prices import day_ints

log = logging.getLogger("fea.chips")

COLUMNS = ["trade_date", "stock_code", "psum", "n_levels", "mean",
           "p10", "p25", "p50", "p75", "p90", "std", "skew", "kurt",
           "cr3", "entropy", "gini", "iqr", "width", "mode", "peak_purity",
           "n_peaks", "peak_gap", "below_close", "above_close", "top10_share"]


def _weighted_quantile(p: np.ndarray, w: np.ndarray, q: np.ndarray) -> np.ndarray:
    """加权分位数（按 price 升序）。p=(N,) 价格, w=(N,) 已归一化权重, q=(K,) 分位。"""
    cw = np.cumsum(w)
    cw /= cw[-1] if cw[-1] > 0 else 1.0
    return np.interp(q, cw, p)


class ChipLayer(DerivedCache):
    name = "chips"
    version = 1

    # ---------------------------------------------------------------- 指纹
    def source_fingerprint(self, year: int) -> dict:
        p = self.cfg.upstream / "stock_cyq_chips" / f"year={year}" / "data.parquet"
        if not p.exists():
            return {"exists": False}
        import pyarrow.parquet as pq
        st = p.stat()
        price_file = self.cfg.upstream / "stock_daily" / f"year={year}" / "data.parquet"
        price_stat = price_file.stat() if price_file.exists() else None
        return {"price": [price_stat.st_size, price_stat.st_mtime_ns] if price_stat else None,
                "exists": True, "rows": pq.ParquetFile(p).metadata.num_rows,
                "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}

    # ---------------------------------------------------------------- 构建
    def build_year(self, year: int, up) -> pd.DataFrame:
        src = self.cfg.upstream / "stock_cyq_chips" / f"year={year}" / "data.parquet"
        if not src.exists():
            return pd.DataFrame()
        # Aggregate one calendar quarter at a time, preserving complete stock-days.
        # Filtering before conversion avoids loading an entire minute/chip year.
        frames = []
        for month in (1, 4, 7, 10):
            lo = f"{year}-{month:02d}-01"
            hi = f"{year + 1}-01-01" if month == 10 else f"{year}-{month + 3:02d}-01"
            df = pd.read_parquet(src, columns=["trade_date", "stock_code", "price", "percent"],
                                 filters=[("trade_date", ">=", lo), ("trade_date", "<", hi),
                                          ("stock_code", "in", self.codes.tolist())])
            frame = self._aggregate_frame(df, year)
            if not frame.empty:
                frames.append(frame)
            del df
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _aggregate_frame(self, df, year):
        if df.empty:
            return pd.DataFrame()
        # 只保留面板里的股票（主板）
        cpos = pd.Series(np.arange(self.codes.size), index=self.codes)
        ci = cpos.reindex(df["stock_code"].to_numpy()).to_numpy()
        keep = np.isfinite(ci)
        df = df.loc[keep]
        codes_i = ci[keep].astype(np.int64)
        di = day_ints(df["trade_date"])
        price = pd.to_numeric(df["price"], errors="coerce").to_numpy(np.float64)
        pct = pd.to_numeric(df["percent"], errors="coerce").to_numpy(np.float64)
        # 负零 / 缺失 / 非正价格一律丢掉
        ok = np.isfinite(price) & (price > 0) & np.isfinite(pct) & (pct >= 0)
        codes_i, di, price, pct = codes_i[ok], di[ok], price[ok], pct[ok]

        # 分段：按 (日期, 股票) 排序后找边界。
        # ★ 合成一个 int64 键做**单次** argsort，比 lexsort/factorize 快得多：
        #   di 是 YYYYMMDD，同一分区内后 4 位就是月日（1..1231，组内单调）；
        #   codes_i < 3484。键 = 股票 × 10000 + 月日，量级 ~3.5e7，远小于 int64 上限。
        dmd = di - (year * 10000)                 # 年内月日，组内单调
        key = codes_i * 10000 + dmd
        order = np.argsort(key, kind="stable")
        key, price, pct = key[order], price[order], pct[order]
        codes_i, di = codes_i[order], di[order]
        starts = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
        ends = np.concatenate((starts[1:], [key.size]))

        n = starts.size
        # 现价（未复权 close）—— 筹码档位本身就是未复权价，同口径
        close_map = self._close_lookup(year)

        out = {c: np.zeros(n, dtype=np.float64) for c in COLUMNS[2:]}
        o_day = np.zeros(n, dtype=np.int32)
        o_code = np.zeros(n, dtype=np.int64)
        qs = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
        for i in range(n):
            a, b = starts[i], ends[i]
            p = price[a:b]
            w = pct[a:b]
            t = w.sum()
            o_day[i] = di[a]
            o_code[i] = codes_i[a]
            out["psum"][i] = t
            out["n_levels"][i] = b - a
            if t <= 0 or p.size < 2:
                for c in COLUMNS[2:]:
                    out[c][i] = np.nan
                out["psum"][i] = t
                out["n_levels"][i] = b - a
                continue
            w = w / t                                   # ★ 归一化（见模块 docstring）
            out["mean"][i] = float((p * w).sum())
            qv = _weighted_quantile(p, w, qs)
            out["p10"][i], out["p25"][i], out["p50"][i], out["p75"][i], out["p90"][i] = qv
            mu = out["mean"][i]
            var = float(((p - mu) ** 2 * w).sum())
            sd = np.sqrt(max(var, 0.0))
            out["std"][i] = sd
            if sd > 0:
                out["skew"][i] = float(((p - mu) ** 3 * w).sum() / sd ** 3)
                out["kurt"][i] = float(((p - mu) ** 4 * w).sum() / sd ** 4 - 3.0)
            # 集中度
            srt = np.argsort(-w)
            out["cr3"][i] = float(w[srt[:3]].sum())
            out["top10_share"][i] = float(w[srt[:10]].sum())
            nz = w[w > 0]
            out["entropy"][i] = float(-(nz * np.log(nz)).sum() / np.log(nz.size)) \
                if nz.size > 1 else 0.0
            # 基尼：筹码在各价格档之间的**不均匀程度**（0=完全均匀，→1=全挤在一档）。
            # 用加权基尼的标准式：把 w 升序排，G = 2·Σ(i+1)·w_(i) / (n·Σw) − (n+1)/n
            # （先前的写法是「洛伦兹曲线」式但符号/分母都错了，会算出负数。）
            ws = np.sort(w)
            nn = ws.size
            out["gini"][i] = float(
                (2.0 * np.dot(np.arange(1, nn + 1, dtype=np.float64), ws)) / nn - (nn + 1.0) / nn
            ) if nn > 1 else 0.0
            out["iqr"][i] = float(qv[3] - qv[0])
            out["width"][i] = float((qv[4] - qv[0]) / qv[2]) if qv[2] > 0 else np.nan
            # 众数价与峰纯度
            k = int(srt[0])
            out["mode"][i] = float(p[k])
            out["peak_purity"][i] = float(w[k])
            # 局部极大值个数（筹码峰）与最大两峰的间距
            peaks = np.flatnonzero((w[1:-1] > w[:-2]) & (w[1:-1] > w[2:])) + 1
            out["n_peaks"][i] = float(peaks.size)
            if peaks.size >= 2:
                top2 = peaks[np.argsort(-w[peaks])[:2]]
                out["peak_gap"][i] = float(abs(p[top2[0]] - p[top2[1]]) / qv[2]) \
                    if qv[2] > 0 else np.nan
            else:
                out["peak_gap"][i] = np.nan
            # 获利盘
            cl = close_map.get((int(o_day[i]), int(o_code[i])), np.nan)
            if np.isfinite(cl):
                out["below_close"][i] = float(w[p <= cl].sum())
                out["above_close"][i] = float(w[p > cl].sum())
            else:
                out["below_close"][i] = np.nan
                out["above_close"][i] = np.nan

        from .dates import int_to_str_vec
        res = pd.DataFrame({
            "trade_date": int_to_str_vec(o_day),
            "stock_code": self.codes[o_code],
            **{c: out[c] for c in COLUMNS[2:]},
        })
        return res[COLUMNS]

    # ---------------------------------------------------------------- 现价表
    def _close_lookup(self, year: int) -> dict:
        """(YYYYMMDD, code_idx) -> 未复权 close，用于算获利盘。"""
        p = self.cfg.upstream / "stock_daily" / f"year={year}" / "data.parquet"
        if not p.exists():
            return {}
        d = pd.read_parquet(p, columns=["stock_code", "trade_date", "close"])
        cpos = pd.Series(np.arange(self.codes.size), index=self.codes)
        ci = cpos.reindex(d["stock_code"].to_numpy()).to_numpy()
        keep = np.isfinite(ci)
        di = day_ints(d["trade_date"])
        cl = pd.to_numeric(d["close"], errors="coerce").to_numpy(np.float64)
        return {(int(a), int(b)): float(c)
                for a, b, c in zip(di[keep], ci[keep].astype(np.int64), cl[keep])}

    # ---------------------------------------------------------------- 读取
    def panel(self, panel, field: str) -> np.ndarray:
        key = (int(panel.dates[0]), int(panel.dates[-1]), panel.C, field)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if field not in COLUMNS[1:]:
            raise KeyError(f"筹码层不认识字段 {field!r}。可用：{COLUMNS[1:]}")
        y0 = int(str(panel.dates[0])[:4])
        y1 = int(str(panel.dates[-1])[:4])
        self.ensure(y0, y1)               # ★ 缓存缺失就当场构建，不靠调用方记得
        df = self._read_all(y0, y1)
        if df.empty:
            out = panel.empty()
        else:
            cpos = pd.Series(np.arange(panel.C), index=panel.codes)
            ci = cpos.reindex(df["stock_code"].to_numpy()).to_numpy()
            keep = np.isfinite(ci)
            vals = pd.to_numeric(df[field], errors="coerce").to_numpy(np.float64)
            # ★ 精确落格（缺则 NaN）—— 筹码数据缺失时必须是 NaN，
            #   用 asof 前向填充会让"这只股今天没筹码数据"变成"沿用昨天的"，
            #   覆盖率虚高且静默。
            out = panel.place(ci[keep].astype(np.int64),
                              day_ints(df["trade_date"])[keep], vals[keep])
        self._cache[key] = out
        return out
