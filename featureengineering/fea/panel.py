"""面板（Panel）—— 因子计算的基本几何结构。

**核心约定：因子函数的输入/输出都是 `(T, C)` 的 float32 numpy 数组**，
T = 交易日数，C = 股票数。不出现 DataFrame、不出现字符串列。

为什么这么设计（都是实测/血泪）：
  1. 内存：T×C 一年约 243×3300 ≈ 80 万格 × 4 B = 3.2 MB。
     若用 DataFrame 装两个字符串列，按 pandas 3.0 的 Arrow 字符串 51 B/行
     就要 82 MB，再叠加 concat/去重/排序的 4~5 倍峰值就是 400 MB（见 §13.11）。
  2. 速度：滚动窗口用 `cumsum` 差分是 O(T·C)，而 `groupby().rolling()`
     在长表上要几十秒——这是同类项目最经典的一个坑。
  3. 正确性：网格按构造即有序（`np.repeat` + `np.tile`），
     永远不需要 `sort_values`，也就不会出现排序不稳定导致的错位。

网格索引 i = t * C + c（按天为主序、股票为次序）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dates import CODE_W


class Panel:
    """交易日 × 股票的规则网格。"""

    def __init__(self, dates: np.ndarray, codes: np.ndarray):
        self.dates = np.ascontiguousarray(dates, dtype=np.int32)   # (T,) 已排序
        self.codes = np.asarray(codes)                             # (C,)
        self.T = int(self.dates.size)
        self.C = int(self.codes.size)
        self.shape = (self.T, self.C)
        self.size = self.T * self.C

        # 构造即有序：grid_day 按天重复，grid_code 循环铺开
        self.grid_day = np.repeat(self.dates, self.C)              # (T*C,)
        self.grid_code = np.tile(np.arange(self.C, dtype=np.int32), self.T)
        self._kg = self.grid_code.astype(np.int64) * CODE_W + self.grid_day

    # ---------------------------------------------------------------- 索引
    def empty(self, fill=np.nan, dtype=np.float32) -> np.ndarray:
        return np.full(self.shape, fill, dtype=dtype)

    def row_of(self, day: int) -> int:
        return int(np.searchsorted(self.dates, day))

    def rows_of(self, days: np.ndarray) -> np.ndarray:
        """把一批交易日映射成行号（不存在则返回 T，调用方需自行过滤）。"""
        return np.searchsorted(self.dates, days)

    def day_mask(self, days: np.ndarray) -> np.ndarray:
        """(T,) bool —— 标记这些交易日。"""
        m = np.zeros(self.T, dtype=bool)
        pos = np.searchsorted(self.dates, days)
        ok = (pos < self.T) & (self.dates[np.clip(pos, 0, self.T - 1)] == days)
        m[pos[ok]] = True
        return m

    # ---------------------------------------------------------------- PIT 前向对齐
    def asof(self, src_code: np.ndarray, src_day: np.ndarray, src_val: np.ndarray,
             dtype=np.float32) -> np.ndarray:
        """as-of 前向填充：网格第 d 天取「src_day <= d 的最近一条」的值。

        要求 src_* 已按 (code, day) 升序排列。用 `searchsorted` 而不是
        `pd.merge_asof`：后者带 `by=` 时要整体重排+复制，在千万行的网格上是秒级开销。

        ★ `same-code` 校验不可省略：复合键排序后，某只股票若在其所处位置之前
        没有任何记录，`searchsorted` 会落到**上一只股票**的记录上，
        从而静默地把别人的财务数据算到自己头上。
        """
        ks = src_code.astype(np.int64) * CODE_W + src_day.astype(np.int64)
        pos = np.searchsorted(ks, self._kg, side="right") - 1
        ok = pos >= 0
        p = np.where(ok, pos, 0)
        ok &= src_code[p] == self.grid_code
        out = np.where(ok, src_val[p], np.nan)
        return out.astype(dtype).reshape(self.shape)

    def scatter(self, src_code: np.ndarray, src_day: np.ndarray,
                weights: np.ndarray | None = None, dtype=np.float32) -> np.ndarray:
        """把稀疏事件铺到网格上（同一格多条则累加）。用于涨停这类「事件」型数据。

        与 `asof` 的区别是语义性的：**状态量**（净资产、总股本）要前向填充，
        **事件**（今天涨停了没有）要散点+补零。用一个 asof 糊弄两者，
        `limit_up_count_20` 就会退化成「0 或者永远的陈旧值」。
        """
        t = np.searchsorted(self.dates, src_day)
        t = np.clip(t, 0, self.T - 1)
        ok = self.dates[t] == src_day
        idx = t[ok].astype(np.int64) * self.C + src_code[ok].astype(np.int64)
        w = weights[ok] if weights is not None else np.ones(idx.size, dtype=np.float64)
        flat = np.bincount(idx, weights=w, minlength=self.size)
        return flat.astype(dtype).reshape(self.shape)

    def place(self, src_code: np.ndarray, src_day: np.ndarray, src_val: np.ndarray,
              fill=np.nan, dtype=np.float64) -> np.ndarray:
        """**精确落格**：只有 (day, code) 恰好命中的格子有值，其余为 `fill`。

        这是第三种语义，与已有的两个都不同、且都不可替代：
          · `asof`    —— **前向填充**。状态量（股本、复权因子）用它；
                        但停牌日的成交量若用 asof，会拿到上一个交易日的旧值。
          · `scatter` —— 缺失补 **0**。事件量（今天涨停了没有）用它；
                        但价格/成交量补 0 是灾难（0 会被当成真实的 0 成交）。
          · `place`   —— 缺失补 **NaN**。日频派生根数据（价格、5min 聚合、筹码摘要）用它。
                        停牌日必须是 NaN，滚动窗口才会被毒化而不是被灌进假 0。

        `src_day` 不在网格上的行会被丢弃（例如上游落在非交易日的脏行）。
        要求 (day, code) 唯一 —— 日频派生化天然满足。
        """
        src_code = np.asarray(src_code, dtype=np.int64)
        src_day = np.asarray(src_day, dtype=np.int32)
        src_val = np.asarray(src_val, dtype=dtype)
        t = np.searchsorted(self.dates, src_day)
        t = np.clip(t, 0, self.T - 1)
        ok = (self.dates[t] == src_day) & (src_code >= 0) & (src_code < self.C)
        idx = t[ok].astype(np.int64) * self.C + src_code[ok].astype(np.int64)
        out = np.full(self.size, fill, dtype=dtype)
        out[idx] = src_val[ok]
        return out.reshape(self.shape)

    # ---------------------------------------------------------------- 工具
    def to_long(self, values: np.ndarray, days: np.ndarray,
                codes: np.ndarray | None = None) -> pd.DataFrame:
        """把子集 (len(days), len(codes)) 的数组转成输出用的长表。"""
        codes = self.codes if codes is None else codes
        sub = self._select_rows(values, days)
        T, C = sub.shape
        df = pd.DataFrame({
            "trade_date": np.repeat(np.asarray(days, dtype=np.int32), C),
            "stock_code": np.tile(np.asarray(codes), T),
            "value": sub.reshape(-1).astype(np.float32, copy=False),
        })
        df["trade_date"] = df["trade_date"].map(_int_to_str)
        return df

    def _select_rows(self, values: np.ndarray, days: np.ndarray) -> np.ndarray:
        pos = self.rows_of(np.asarray(days, dtype=np.int32))
        return values[pos]

    def roll_sum(self, mat: np.ndarray, n: int) -> np.ndarray:
        """沿时间轴滚动求和（含当前，窗口 n），**输出与输入同形 (T, C)**。

        用 nan-cumsum 差分而不是 `groupby().rolling()`：后者在长表上要几十秒，
        这里 243×3300 的窗口是毫秒级。前 n-1 行因窗口不足返回 NaN。
        """
        T, C = mat.shape
        z = np.zeros((1, C), dtype=np.float64)
        # ★ 先转 float64 再累加：若入参是 float32（常见），对元级金额（1e8 量级）
        #   的 cumsum 会累积约 1e-4 的相对误差，而且**误差随面板长度变化** ——
        #   表现为「同一个因子在不同 `--end` 下值不同」，是 PIT 前缀一致性的破坏。
        x = np.where(np.isfinite(mat), np.asarray(mat, dtype=np.float64), 0.0)
        cs = np.vstack([z, np.cumsum(x, axis=0)])                 # (T+1, C)
        out = np.full((T, C), np.nan, dtype=np.float64)
        out[n - 1:] = cs[n:] - cs[:-n]
        # 窗口内只要出现过 NaN，结果就置 NaN（避免把缺失当成 0 累加）
        bad = ~np.isfinite(mat)
        if bad.any():
            cnt = np.vstack([np.zeros((1, C), dtype=np.int64), np.cumsum(bad, axis=0)])
            hasnan = (cnt[n:] - cnt[:-n]) > 0
            tail = out[n - 1:]
            tail[hasnan] = np.nan
        return out


def _int_to_str(v: int) -> str:
    return f"{v // 10000:04d}-{(v // 100) % 100:02d}-{v % 100:02d}"


_int_to_str_vec = np.vectorize(_int_to_str, otypes=[object])


# -------------------------------------------------------------------- 截面处理
def cs_rank(mat: np.ndarray, winsor=(0.01, 0.99), mask: np.ndarray | None = None,
            min_count: int = 100) -> np.ndarray:
    """逐行（逐交易日）做缩尾 + 百分位排名，返回 [0,1]，原始 NaN 保持 NaN。

    用 pandas 的 `rank(axis=1, pct=True)` 而不是 numpy argsort：
    前者对**并列值取平均名次**（method='average'）。这在 A 股因子上很重要——
    `limit_up_count_20` 这类因子有大量并列的 0，argsort 会给它们强行排出
    先后次序，凭空制造出截面区分度。
    """
    x = np.asarray(mat, dtype=np.float32)
    if mask is not None:
        x = np.where(mask, x, np.nan)
    df = pd.DataFrame(x)
    if winsor is not None:
        lo, hi = winsor
        q = df.quantile([lo, hi], axis=1).T          # (T, 2)
        lo_v = q.iloc[:, 0].to_numpy(dtype=np.float32)[:, None]
        hi_v = q.iloc[:, 1].to_numpy(dtype=np.float32)[:, None]
        x = np.clip(x, lo_v, hi_v)
        df = pd.DataFrame(x)
    # ★ copy=True：DataFrame.to_numpy() 可能返回只读视图，后续就地赋值会抛
    #   "assignment destination is read-only"（实测踩过）
    out = np.array(df.rank(axis=1, pct=True, na_option="keep").to_numpy(dtype=np.float32),
                   copy=True)
    if min_count:
        n_valid = np.isfinite(x).sum(axis=1)
        out[n_valid < min_count, :] = np.nan
    return out
