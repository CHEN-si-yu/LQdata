"""跨折 / 跨头集成 —— **逐日截面 z-score 后相加**。

为什么要 z 而不是直接平均原始打分：不同模型（岭回归 vs 树）的打分量纲完全不同
（回归输出 ±0.03，树输出 ±0.005），直接平均等于让量纲大的那个独占权重。
参考工程的做法是：**每一折/每一头先在自己内部做「逐日截面 z」**，再相加 ——
这样每个成员对当天的排序贡献等权，且天然免疫量纲与波动率差异。

★ 只做**逐日截面**统计量（z 只用当天的均值/标准差），不使用跨日统计量 —— 铁律之一。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import store


def cs_z(df: pd.DataFrame, col: str = "value") -> pd.Series:
    """逐日截面 z-score（当天有值的行内标准化）。"""
    g = df.groupby("trade_date", observed=True)[col]
    mu = g.transform("mean")
    sd = g.transform("std")
    z = (df[col] - mu) / sd.replace(0.0, np.nan)
    return z.astype("float32")


def _align(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把若干 (trade_date, stock_code, value) 对齐成一张宽表（外连接，缺=NaN）。"""
    out = None
    for k, df in frames.items():
        s = df[["trade_date", "stock_code", "value"]].rename(columns={"value": k})
        out = s if out is None else out.merge(s, on=["trade_date", "stock_code"], how="outer")
    if out is None:
        raise ValueError("combine: 空的成员集合")
    return out


def combine_weighted(per_head: dict[str, pd.DataFrame], weights: dict[str, float] | None = None,
                     name: str = "value") -> pd.DataFrame:
    """逐日截面 z 后按权重相加 → 4 列契约产物。"""
    if not per_head:
        raise ValueError("combine: 没有成员")
    weights = weights or {k: 1.0 for k in per_head}
    parts = {}
    for k, df in per_head.items():
        if df is None or not len(df):
            continue
        d = df[["trade_date", "stock_code", "value"]].copy()
        d["value"] = cs_z(d)
        parts[k] = d
    wide = _align(parts)
    num = np.zeros(len(wide), dtype=np.float64)
    wsum = np.zeros(len(wide), dtype=np.float64)
    for k, w in weights.items():
        if k not in wide.columns:
            continue
        v = wide[k].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        num[ok] += w * v[ok]
        wsum[ok] += abs(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.where(wsum > 0, num / wsum, np.nan)
    out = wide[["trade_date", "stock_code"]].copy()
    out[name] = val.astype("float32")
    out["rank"] = store.cs_rank_from_score(out["trade_date"], out[name])
    out["trade_date"] = out["trade_date"].astype("string")
    out["stock_code"] = out["stock_code"].astype("string")
    out[name] = out[name].astype("float32")
    out["rank"] = out["rank"].astype("float32")
    return out.sort_values(["trade_date", "stock_code"], kind="stable").reset_index(drop=True)


def combine_equal(per_head: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """等权版（最常用）。"""
    return combine_weighted(per_head)


def zscore_across_folds(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """跨折集成：每折先逐日截面 z，再对同一 (日, 股) 取均值 → 该头的单份打分。"""
    return combine_equal({f"f{k}": v for k, v in frames.items()})
