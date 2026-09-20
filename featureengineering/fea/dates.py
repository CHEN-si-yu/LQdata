"""日期工具。

内部一律用 **int32 的 YYYYMMDD** 表示日期（如 20260911），理由：
  - `merge_asof` / `searchsorted` 拒绝字符串日期（实测 `MergeError`），必须数字化；
  - YYYYMMDD 是可读的、且「去年同期」就是简单的 `- 10000`；
  - int32 每行 4 字节，而 pandas 3.0 的 Arrow 字符串是 51 B/行（见 QUANT_PLATFORM §13.11）。
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

# YYYYMMDD < 1e8，用它做 (code, date) 复合键的位宽
CODE_W = 100_000_000


def str_to_int(s: str) -> int:
    """"2026-09-11" -> 20260911"""
    return int(s[:4]) * 10000 + int(s[5:7]) * 100 + int(s[8:10])


def int_to_str(v: int) -> str:
    """20260911 -> '2026-09-11'"""
    return f"{v // 10000:04d}-{(v // 100) % 100:02d}-{v % 100:02d}"


def series_to_int(s: pd.Series) -> np.ndarray:
    """把 "YYYY-MM-DD" 字符串列向量化转成 int32 YYYYMMDD，非法值一律给 0。

    ★★ 性能（2026-09-15 重写）：这是**整个引擎最热的函数**。原先用 pandas 的
      `.str.slice() + to_numeric`，50 万行的列要 **2.4 秒**；而价格层给每个字段落格时
      都会重转一次同一列日期（实测单任务 47 次调用，**占 84% 的运行时间** ——
      增量跑里 119s 的任务有 112s 花在这上面）。
      现在走 numpy 的 UCS-4 视图直接取字符做算术，实测快 **20 倍以上**（同列约 0.1s）。
      `series_to_int` 被 deriv/prices/universe/intraday/factors_io 全都用到，所以这一处
      的收益是全链路的。

    ★ 必须容忍空串/短串：`stock_list.delist_date` 对未退市股票就是空值，
    直接 `astype("int32")` 会抛 `invalid literal for int() with base 10: ''`（实测踩过）。
    快路径用「第 4、7 位是不是连字符」判合法，等价于原来的 `len>=10`。
    """
    a = s.to_numpy(dtype="U10", na_value="")
    try:
        u = a.view(np.uint32).reshape(len(a), 10)          # numpy 默认 UCS-4
        ok = (u[:, 4] == 0x2D) & (u[:, 7] == 0x2D)         # '-'
        if not ok.any():
            return np.zeros(len(a), dtype=np.int32)
        g = (u[:, 0].astype(np.int32) - 48) * 1000 + (u[:, 1].astype(np.int32) - 48) * 100 \
            + (u[:, 2].astype(np.int32) - 48) * 10 + (u[:, 3].astype(np.int32) - 48)
        mo = (u[:, 5].astype(np.int32) - 48) * 10 + (u[:, 6].astype(np.int32) - 48)
        dy = (u[:, 8].astype(np.int32) - 48) * 10 + (u[:, 9].astype(np.int32) - 48)
        out = np.where(ok, g * 10000 + mo * 100 + dy, 0)
        return out.astype(np.int32)
    except Exception:                                      # noqa: BLE001
        # 兜底：非 UCS-4 构建 / dtype 异常时回到 pandas 路径（正确性优先）
        x = s.astype("string").fillna("")
        okm = x.str.len() >= 10
        y = pd.to_numeric(x.str.slice(0, 4).where(okm), errors="coerce")
        m = pd.to_numeric(x.str.slice(5, 7).where(okm), errors="coerce")
        d = pd.to_numeric(x.str.slice(8, 10).where(okm), errors="coerce")
        return (y * 10000 + m * 100 + d).fillna(0).to_numpy(dtype="int32")


def add_days(date_str: str, n: int) -> str:
    d = _dt.date.fromisoformat(date_str) + _dt.timedelta(days=n)
    return d.isoformat()


def shift_year(v: int, delta: int) -> int:
    """YYYYMMDD 整体平移年份（用于「去年同期」）。"""
    return v + delta * 10000


def year_of(v: int) -> int:
    return v // 10000


def quarter_of(v: int) -> int:
    """1..4。财报的 end_date 只会落在 3/6/9/12 月（实测确认，见 fea/deriv.py）。"""
    return (v // 100) % 100 // 3


def today_int() -> int:
    t = _dt.date.today()
    return t.year * 10000 + t.month * 100 + t.day


def int_to_str_vec(v: np.ndarray) -> list[str]:
    """向量化的 int32 YYYYMMDD -> "YYYY-MM-DD"。比 .map() 快，且不给每行建 Series。

    ★ 空数组必须返回 `[]`：否则会一路走到 `np.char.add` 内部的
      「zero-size array to reduction operation maximum which has no identity」，
      那个报错**完全指错地方** —— 真正的病根几乎总在上游（某段窗口没有数据）。
      实测两个因子开发 Agent 都被这个误导读数浪费过时间。
    """
    v = np.asarray(v, dtype=np.int64)
    if v.size == 0:
        return []
    y = (v // 10000).astype(str)
    m = ((v // 100) % 100).astype(str)
    d = (v % 100).astype(str)
    return np.char.add(np.char.add(np.char.add(np.char.add(y, "-"),
                                              np.char.zfill(m, 2)), "-"),
                       np.char.zfill(d, 2)).tolist()


class Calendar:
    """交易日历（来自上游 basic_calendar）。"""

    def __init__(self, days: np.ndarray):
        # days: 已排序的 int32 YYYYMMDD（仅交易日）
        self.days = np.ascontiguousarray(days, dtype=np.int32)
        self.day_strs = [int_to_str(int(d)) for d in self.days]

    @classmethod
    def load(cls, upstream) -> "Calendar":
        p = upstream.calendar_path
        df = pd.read_parquet(p, columns=["date", "is_open"])
        df = df[df["is_open"] == 1]
        v = series_to_int(df["date"])
        return cls(np.unique(v))

    def between(self, start: int, end: int) -> np.ndarray:
        """闭区间 [start, end] 内的交易日。"""
        lo = int(np.searchsorted(self.days, start, side="left"))
        hi = int(np.searchsorted(self.days, end, side="right"))
        return self.days[lo:hi]

    def last(self) -> int:
        return int(self.days[-1])
