"""5 分钟行情 → 日频宽表（派生缓存）。

## 为什么要预聚合

`stock_history_5min`：**6.86 亿行 / 8.4 GB / 48 根每日 / 2010→2026**。
日内因子只吃**日频**结果（用户硬约束：最终产物必须日频），所以一次性把 48 根
聚合成约 35 个日频字段，之后每个因子只读几十 MB。

实测（2025 分区 6306 万行）：读 7.2s / 5.8 GB，分组聚合 15.5s。
17 个分区单进程约 6~9 分钟，产物约 0.9 GB（对比原始 8.4 GB）。

## ★★ 实测到的真实数据陷阱：`vol` 的单位分段翻转

`stock_history_5min.vol` 的单位**不是恒定的**：

    2010-01-04 ~ 2025-11-28   手   (Σvol × 100 == stock_daily.vol)
    2025-12-01 起             股   (Σvol     == stock_daily.vol)

而 `stock_daily.vol` **全程是股**、两表的 `amount`（元）全程一致。
不归一化的话，跨 2025-11/12 边界的量比类因子会有 **100 倍的假跳变** ——
而且**全程不报错**。

**归一方法**：逐 (股票, 日) 比较 `Σamount / Σvol` 与该日收盘价。
若 ≈ `close` 则 vol 已是股；若 ≈ `100 × close` 则 vol 是手，乘 100。
（用日内 VWAP 与收盘价比会更稳，但收盘价就够 —— 两者相差 <1%。）

## 其它口径

* 价格**未复权**（与 `stock_daily` 一致）。做跨日比较时必须配 `adj_factor`，
  因子层用 `ctx.hfq(...)` 或 `ctx.ret(...)`，本层的字段都是**日内**量，不跨日。
* 隔夜跳空这类**跨日**量需要 `adj_factor`（除权日的前收要按旧因子还原），
  所以本层只存**原料**（`prev_close5`、`open5`），由因子层配合复权因子算。
  → 这样加新日内因子**不用重建缓存**。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .derived import DerivedCache
from .dates import series_to_int

log = logging.getLogger("fea.intraday")

# 存「原料」而不是只存最终比值：这样新增日内因子不需要重跑 6.86 亿行
FIELDS = [
    "n_bars", "open5", "close5", "hi5", "lo5", "vol", "amt", "vwap",
    "am_open5", "am_close5", "am_hi", "am_lo", "am_vol", "am_amt",
    "pm_open5", "pm_close5", "pm_hi", "pm_lo", "pm_vol", "pm_amt",
    "ret_sum", "ret2_sum", "absret_sum", "posret2_sum", "negret2_sum",
    "hi_pos", "lo_pos", "max_dd", "max_ru", "path_len",
    "vol2_sum", "amt2_sum", "n_up", "n_zero",
]


class IntradayLayer(DerivedCache):
    name = "intraday"
    version = 2   # v2: 修 vol2_sum 的平方量单位

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
        # Aggregate one calendar quarter at a time, preserving complete stock-days.
        # Filtering before conversion avoids loading an entire minute/chip year.
        frames = []
        for month in (1, 4, 7, 10):
            lo = f"{year}-{month:02d}-01"
            hi = f"{year + 1}-01-01" if month == 10 else f"{year}-{month + 3:02d}-01"
            df = pd.read_parquet(src, columns=["stock_code", "trade_time", "open", "high",
                                           "low", "close", "vol", "amount"],
                                 filters=[("trade_time", ">=", lo), ("trade_time", "<", hi),
                                          ("stock_code", "in", self.codes.tolist())])
            frame = self._aggregate_frame(df, year)
            if not frame.empty:
                frames.append(frame)
            del df
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _aggregate_frame(self, df, year):
        if df.empty:
            return pd.DataFrame()

        cpos = pd.Series(np.arange(self.codes.size), index=self.codes)
        ci = cpos.reindex(df["stock_code"].to_numpy()).to_numpy()
        keep = np.isfinite(ci)
        df = df.loc[keep]
        ci = ci[keep].astype(np.int64)

        # 日期用现成的向量化工具（numpy 的字符串数组不支持 `a[:, 0:4]` 那种字符切片）
        tt = df["trade_time"].astype("string")
        di = series_to_int(tt.str.slice(0, 10))
        hm = (tt.str.slice(11, 13).astype("int32") * 100
              + tt.str.slice(14, 16).astype("int32")).to_numpy(dtype=np.int32)
        o = pd.to_numeric(df["open"], errors="coerce").to_numpy(np.float64)
        h = pd.to_numeric(df["high"], errors="coerce").to_numpy(np.float64)
        lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(np.float64)
        c = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)
        v = pd.to_numeric(df["vol"], errors="coerce").to_numpy(np.float64)
        a = pd.to_numeric(df["amount"], errors="coerce").to_numpy(np.float64)

        # ★ 用 (code, day, hm) 排序 —— 保证每个 (code,day) 是一段连续区间
        day_code = pd.factorize(di, sort=False)[0]
        key = (ci * 1_000_003 + day_code).astype(np.int64)
        order = np.lexsort((hm, key))
        key, ci, di, hm = key[order], ci[order], di[order], hm[order]
        o, h, lo, c, v, a = o[order], h[order], lo[order], c[order], v[order], a[order]

        starts = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
        ends = np.concatenate((starts[1:], [key.size]))
        n = starts.size
        log.info("intraday %d：%d 行 -> %d 个 (股票, 日)", year, key.size, n)

        def rsum(x):
            return np.add.reduceat(np.nan_to_num(x, nan=0.0), starts)

        def rmax(x):
            return np.maximum.reduceat(x, starts)

        def rmin(x):
            return np.minimum.reduceat(x, starts)

        # 日内收益（对每个 (code,day) 段内做 diff，跨界处置 NaN）
        prev_c = np.empty_like(c)
        prev_c[0] = np.nan
        prev_c[1:] = c[:-1]
        prev_c[starts] = np.nan                  # 段首没有前一棒
        with np.errstate(all="ignore"):
            r = np.where(np.isfinite(prev_c) & (prev_c > 0), c / prev_c - 1.0, np.nan)
        r = np.where(np.isfinite(r), r, np.nan)
        r0 = np.nan_to_num(r, nan=0.0)

        last = ends - 1
        out = {
            "n_bars": (ends - starts).astype(np.float64),
            "open5": o[starts], "close5": c[last],
            "hi5": rmax(np.nan_to_num(h, nan=-np.inf)),
            "lo5": rmin(np.nan_to_num(lo, nan=np.inf)),
            "vol": rsum(v), "amt": rsum(a),
            "ret_sum": rsum(r0), "ret2_sum": rsum(r0 * r0),
            "absret_sum": rsum(np.abs(r0)),
            "posret2_sum": rsum(np.where(r0 > 0, r0 * r0, 0.0)),
            "negret2_sum": rsum(np.where(r0 < 0, r0 * r0, 0.0)),
            "vol2_sum": rsum(v * v), "amt2_sum": rsum(a * a),
            "n_up": rsum((r0 > 0).astype(np.float64)),
            "n_zero": rsum((r0 == 0).astype(np.float64)),
            "path_len": rsum(np.abs(r0)),
        }
        out["hi5"] = np.where(np.isfinite(out["hi5"]), out["hi5"], np.nan)
        out["lo5"] = np.where(np.isfinite(out["lo5"]), out["lo5"], np.nan)

        # 日内累计路径的极值位置 / 最大回撤 / 最大反弹 / 量能峰值位置
        vol_peak = _argmax_per_seg(np.nan_to_num(v, nan=-np.inf), starts, ends)
        out["vol_peak_pos"] = vol_peak / np.maximum(out["n_bars"] - 1, 1)

        # ---- 上/下午分组：用「段 × 时段」的二级键做 reduceat/累加 ----
        # ★ 不能用段内相对下标去调 reduceat（它要的是**绝对**下标，且总是归约到
        #   下一个下标为止）；下面这个写法把 (段, 时段) 编码成一个键，天然正确。
        am = (hm <= 1130)
        pm = (hm >= 1301)
        sess = np.where(am, 0, np.where(pm, 1, 2)).astype(np.int64)
        key2 = key * 4 + sess                       # 组内已按 hm 升序 -> key2 升序
        st2 = np.flatnonzero(np.concatenate(([True], key2[1:] != key2[:-1])))
        en2 = np.concatenate((st2[1:], [key2.size]))
        # ★ `reduceat(x, idx)` 归约的是 [idx[i], idx[i+1]) —— 所以**光给起点不行**，
        #   它会一路吃到下一个起点（那是下一个一级段的 am 块），把 pm 块也算进来。
        #   正确做法是把 (起点, 终点) 交替铺开，再隔一个取一个。
        seg_of = np.clip(np.searchsorted(starts, st2, side="right") - 1, 0, n - 1)

        def range_reduce(red, x, s, e, pad):
            """按任意 [s_i, e_i) 区间归约。

            ★ 两个坑：
              1. `reduceat(x, idx)` 归约的是 [idx[i], idx[i+1])，所以必须把
                 (起点, 终点) 交替铺开再隔一个取一个 —— 只给起点会一路吃到下一个起点。
              2. 最后一个区间的终点等于 N，而 reduceat 不接受 idx == N（越界）。
                 末尾补一个**单位元**（和=0 / max=−inf / min=+inf），N 就成了合法下标。
            """
            xp = np.concatenate([x, np.array([pad], dtype=x.dtype)])
            idx = np.empty(s.size * 2, dtype=np.int64)
            idx[0::2] = s
            idx[1::2] = e
            return red(xp, idx)[0::2]

        for tag, sid in (("am", 0), ("pm", 1)):
            m = (key2[st2] % 4) == sid
            s2, e2_, r = st2[m], en2[m], seg_of[m]
            z = np.full(n, np.nan)
            for f in ("open5", "close5", "hi", "lo", "vol", "amt"):
                out[f"{tag}_{f}"] = z.copy()
            if s2.size == 0:
                continue
            out[f"{tag}_open5"][r] = o[s2]
            out[f"{tag}_close5"][r] = c[e2_ - 1]
            out[f"{tag}_vol"][r] = range_reduce(
                np.add.reduceat, np.nan_to_num(v, nan=0.0), s2, e2_, 0.0)
            out[f"{tag}_amt"][r] = range_reduce(
                np.add.reduceat, np.nan_to_num(a, nan=0.0), s2, e2_, 0.0)
            mx = range_reduce(np.maximum.reduceat,
                              np.nan_to_num(h, nan=-np.inf), s2, e2_, -np.inf)
            mn = range_reduce(np.minimum.reduceat,
                              np.nan_to_num(lo, nan=np.inf), s2, e2_, np.inf)
            out[f"{tag}_hi"][r] = np.where(np.isfinite(mx), mx, np.nan)
            out[f"{tag}_lo"][r] = np.where(np.isfinite(mn), mn, np.nan)

        # 段内累计极值的位置
        hi_pos, lo_pos, mdd, mru = _path_stats(np.nan_to_num(c, nan=np.nan), starts, ends)
        out["hi_pos"] = hi_pos / np.maximum(out["n_bars"] - 1, 1)
        out["lo_pos"] = lo_pos / np.maximum(out["n_bars"] - 1, 1)
        out["max_dd"] = mdd
        out["max_ru"] = mru

        # ★ vol 单位归一化（见模块 docstring）：逐段判断该段是「手」还是「股」
        with np.errstate(all="ignore"):
            vwap_raw = out["amt"] / np.where(out["vol"] > 0, out["vol"], np.nan)
            ratio = vwap_raw / np.where(out["close5"] > 0, out["close5"], np.nan)
        # ratio ≈ 1 -> 已是股；ratio ≈ 100 -> 是手
        mult = np.where(np.isfinite(ratio) & (ratio > 10.0), 100.0, 1.0)
        n_flip = int((mult > 1).sum())
        if n_flip:
            log.info("intraday %d：%d 个 (股票,日) 的 vol 是「手」，已 ×100 归一", year, n_flip)
        for k in ("vol", "am_vol", "pm_vol"):
            out[k] = out[k] * mult
        # ★ `vol2_sum` 是**平方量**，乘子要平方 —— 写成 `* mult` 会让
        #   `vol2_sum / vol²` 在单位翻转前后量纲不一致（实测 2025 年 90.5% 的格子
        #   违反柯西下界 1/n_bars）。这个 bug 由日内家族的开发 Agent 发现，
        #   它当时在因子内用 `vol2_sum * vol_mult / vol²` 补偿；现在层里修好了，
        #   因子侧的那次补偿必须同时去掉，否则会双重放大。
        out["vol2_sum"] = out["vol2_sum"] * mult * mult
        out["vol_mult"] = mult
        with np.errstate(all="ignore"):
            out["vwap"] = out["amt"] / np.where(out["vol"] > 0, out["vol"], np.nan)

        from .dates import int_to_str_vec
        res = pd.DataFrame({
            "trade_date": int_to_str_vec(di[starts]),
            "stock_code": self.codes[ci[starts]],
            **{k: np.asarray(vv, dtype=np.float64) for k, vv in out.items()},
        })
        return res

    # ---------------------------------------------------------------- 读取
    def panel(self, panel, field: str) -> np.ndarray:
        key = (int(panel.dates[0]), int(panel.dates[-1]), panel.C, field)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if field not in self._cols_available():
            raise KeyError(f"日内层不认识字段 {field!r}。可用：{sorted(self._cols_available())}")
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
            from .prices import day_ints
            vals = pd.to_numeric(df[field], errors="coerce").to_numpy(np.float64)
            # ★ 精确落格：源冻结（stock_history_5min 停在 2026-09-11）时必须暴露成 NaN，
            #   用 asof 会永久前向填充陈旧值，空洞被彻底藏起来。
            out = panel.place(ci[keep].astype(np.int64),
                              day_ints(df["trade_date"])[keep], vals[keep])
        self._cache[key] = out
        return out

    def _cols_available(self) -> set:
        return set(FIELDS) | {"vol_peak_pos", "vol_mult"}


# ---------------------------------------------------------------- 段内工具
def _seg_sum(x: np.ndarray, s: np.ndarray, e: np.ndarray) -> np.ndarray:
    cs = np.concatenate(([0.0], np.cumsum(x)))
    return cs[e] - cs[s]


def _seg_max(x: np.ndarray, s: np.ndarray, e: np.ndarray) -> np.ndarray:
    m = np.maximum.reduceat(x, s)
    return np.where(e - s > 0, m, np.nan)


def _seg_min(x: np.ndarray, s: np.ndarray, e: np.ndarray) -> np.ndarray:
    m = np.minimum.reduceat(x, s)
    return np.where(e - s > 0, m, np.nan)


def _first_true_per_seg(mask: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """每个段内第一个 True 的位置（段内相对下标）；没有则 NaN。"""
    idx = np.where(mask, np.arange(mask.size), mask.size)
    return np.minimum.reduceat(idx, starts).astype(np.float64) - starts


def _argmax_per_seg(x: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """每个段内 argmax 的**段内相对**下标。"""
    m = np.maximum.reduceat(x, starts)
    idx = np.where(x >= np.repeat(m, ends - starts), np.arange(x.size), x.size)
    first = np.minimum.reduceat(idx, starts)
    return (first - starts).astype(np.float64)


def _path_stats(c: np.ndarray, starts: np.ndarray, ends: np.ndarray):
    """逐段的：最高点位置、最低点位置、最大回撤、最大反弹（全是段内相对下标）。"""
    n = starts.size
    hi_pos = np.full(n, np.nan)
    lo_pos = np.full(n, np.nan)
    mdd = np.full(n, np.nan)
    mru = np.full(n, np.nan)
    for i in range(n):
        s, e = starts[i], ends[i]
        seg = c[s:e]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        hi_pos[i] = float(np.argmax(seg))
        lo_pos[i] = float(np.argmin(seg))
        run_max = np.maximum.accumulate(seg)
        with np.errstate(all="ignore"):
            mdd[i] = float(np.min(seg / run_max - 1.0))
        run_min = np.minimum.accumulate(seg)
        with np.errstate(all="ignore"):
            mru[i] = float(np.max(seg / run_min - 1.0))
    return hi_pos, lo_pos, mdd, mru
