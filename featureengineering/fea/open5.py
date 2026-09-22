"""开盘首段（首 5 分钟）流动性的日频宽表（派生缓存）。

## 为什么单独开一层，而不是往 `IntradayLayer` 里加字段

`IntradayLayer` 的 `version` 一变就是**全量重建**，而它的产物是现有 **33 个日内因子**
的输入。为了加两个字段去动那一层，等于让 33 个因子的输入集体换一次版本 ——
收益（省一次读盘）远小于风险（一次无谓的全量重建 + 与正在跑的模型训练抢盘）。
所以这里**新开一个层**：`data/derived/open5/`，与 `intraday/` 平级，
`DerivedCache` 的失效判定逐层独立，两边互不影响。

## 为什么要它（对应评审建议 S-07）

`stock_history_5min`：**6.87 亿行 / 8.99 GB / 2010→2026**。
现有 33 个日内因子吃的是**全天**聚合（`amt`、`am_amt`…），
**没有任何一个字段是"开盘那 5 分钟"**。而执行层真正关心的是：

    全日成交额 ≠ 开盘可成交额。

一笔按 T+1 开盘价下的单，能不能成交、会被冲击多少，取决于**开盘那几分钟的流动性池**，
而不是全天总量。把"开盘 5 分钟成交额 / 全日成交额"做成日频因子，
下游就能给容量上限加一个**按当日开盘时段**收口的约束，而不是拿全天的 1% 去近似。

## 口径（写死在这里）

* **"首 5 分钟" = 每个 (股票, 日) 的**第一根** 5 分钟 bar**。
  厂商的 bar 时间戳是**区间右端**：实测最早一根是 `09:35:00`（覆盖 09:30–09:35），
  每天 48 根。所以"第一根"就是 09:35 那根，不需要按时钟硬编码 ——
  同时也把 `first_hm` 落盘，日后厂商改了 bar 对齐方式能立刻看出来。

* `amt_*` 单位**全程是元**（实测与 `stock_daily.amount` 一致），**不涉及 `vol` 的单位翻转**
  （`vol` 在 2025-11-28/12-01 之间从"手"翻成"股"，见 `fea/intraday.py`）。
  本层**故意只落金额字段**：比值与量级都不需要股数，也就绕开了那个陷阱。

* 价格**未复权**（本层不含价格，天然不涉及前复权红线）。

* 停牌 / 未上市 / 已退市当天**没有行** → `panel()` 用**精确落格**而非 asof 前向填充，
  空洞直接暴露成 NaN（与日内层同一处置）。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .derived import DerivedCache
from .dates import int_to_str_vec, series_to_int

log = logging.getLogger("fea.open5")

# 只落金额类原料：比值与量级都够用，且完全避开 vol 的单位翻转
FIELDS = ["amt_open5", "amt_day", "n_bars", "first_hm"]


class Open5Layer(DerivedCache):
    name = "open5"
    # v2（2026-09-21）：`trade_date` 改存**字符串** 'YYYY-MM-DD'。
    #   v1 存的是 int（20180102），而 `panel.place()` 的落格路径
    #   （`day_ints(df["trade_date"])` → `searchsorted(panel.dates, ...)`）吃的是字符串日期 ——
    #   于是所有格都对不上，**因子整年全 NaN 而且不报错**（实测 2018 年
    #   513,945 行 value/rank 全空，退出码仍是 0）。
    #   `IntradayLayer` 存的就是 `int_to_str_vec(...)` 的字符串，本层对齐它。
    version = 2

    # ---------------------------------------------------------------- 指纹
    def source_fingerprint(self, year: int) -> dict:
        p = self.cfg.upstream / "stock_history_5min" / f"year={year}" / "data.parquet"
        if not p.exists():
            return {"exists": False}
        import pyarrow.parquet as pq
        st = p.stat()
        return {"exists": True, "rows": pq.ParquetFile(p).metadata.num_rows,
                "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}

    # ---------------------------------------------------------------- 构建
    def build_year(self, year: int, up) -> pd.DataFrame:
        src = self.cfg.upstream / "stock_history_5min" / f"year={year}" / "data.parquet"
        if not src.exists():
            return pd.DataFrame()
        # 按季度分块读：与 IntradayLayer 同一手法，避免一次性把一年 6300 万行读进来。
        # `stock_code in codes` 的谓词下推把行数先砍到冻结池内。
        frames = []
        for month in (1, 4, 7, 10):
            lo = f"{year}-{month:02d}-01"
            hi = f"{year + 1}-01-01" if month == 10 else f"{year}-{month + 3:02d}-01"
            df = pd.read_parquet(src, columns=["stock_code", "trade_time", "amount"],
                                 filters=[("trade_time", ">=", lo), ("trade_time", "<", hi),
                                          ("stock_code", "in", self.codes.tolist())])
            frame = self._aggregate_frame(df, year)
            if not frame.empty:
                frames.append(frame)
            del df
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _aggregate_frame(self, df, year) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        cpos = pd.Series(np.arange(self.codes.size), index=self.codes)
        ci = cpos.reindex(df["stock_code"].to_numpy()).to_numpy()
        keep = np.isfinite(ci)
        df = df.loc[keep]
        if df.empty:
            return pd.DataFrame()
        ci = ci[keep].astype(np.int64)

        tt = df["trade_time"].astype("string")
        di = series_to_int(tt.str.slice(0, 10))
        hm = (tt.str.slice(11, 13).astype("int32") * 100
              + tt.str.slice(14, 16).astype("int32")).to_numpy(dtype=np.int32)
        a = pd.to_numeric(df["amount"], errors="coerce").to_numpy(np.float64)

        # 用 (code, day, hm) 排序 —— 保证每个 (code,day) 落在连续区间里，
        # 之后 reduceat / 取段首就都是对的。
        day_code = pd.factorize(di, sort=False)[0]
        key = (ci * 1_000_003 + day_code).astype(np.int64)
        order = np.lexsort((hm, key))
        key, ci, di, hm, a = key[order], ci[order], di[order], hm[order], a[order]

        starts = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
        ends = np.concatenate((starts[1:], [key.size]))
        log.info("open5 %d：%d 行 -> %d 个 (股票, 日)", year, key.size, starts.size)

        a0 = np.nan_to_num(a, nan=0.0)
        out = pd.DataFrame({
            # ★ 必须是字符串日期，与 `IntradayLayer` 一致（见类头的 v2 说明）
            "trade_date": int_to_str_vec(di[starts]),
            "stock_code": self.codes[ci[starts]],
            "amt_open5": a[starts],                                  # ★ 段首那根 = 首 5 分钟
            "amt_day": np.add.reduceat(a0, starts),
            "n_bars": (ends - starts).astype(np.float64),
            "first_hm": hm[starts].astype(np.float64),
        })
        return out

    # ---------------------------------------------------------------- 读取
    def panel(self, panel, field: str) -> np.ndarray:
        key = (int(panel.dates[0]), int(panel.dates[-1]), panel.C, field)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if field not in FIELDS:
            raise KeyError(f"open5 层不认识字段 {field!r}。可用：{sorted(FIELDS)}")
        y0 = int(str(panel.dates[0])[:4])
        y1 = int(str(panel.dates[-1])[:4])
        self.ensure(y0, y1)               # 缓存缺失就当场构建，不靠调用方记得
        df = self._read_all(y0, y1)
        if df.empty:
            out = panel.empty()
        else:
            cpos = pd.Series(np.arange(panel.C), index=panel.codes)
            ci = cpos.reindex(df["stock_code"].to_numpy()).to_numpy()
            keep = np.isfinite(ci)
            from .prices import day_ints
            vals = pd.to_numeric(df[field], errors="coerce").to_numpy(np.float64)
            out = panel.place(ci[keep].astype(np.int64),
                              day_ints(df["trade_date"])[keep], vals[keep])
        self._cache[key] = out
        return out
