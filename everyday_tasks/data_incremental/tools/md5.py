"""指纹工具 —— 回测、回归、单日台账共用同一套算法。

★ 核心取舍：**不用 parquet 文件字节当等价判据。**
   parquet 的 row group 切分、字典回退、zstd 帧边界都会随数据量/写入顺序变化，
   同一份数据两次写出往往字节不同 —— 拿文件 MD5 判等价会**必然假红**。
   文件 MD5 只用作"变没变"的廉价旁证；**等价判据是逻辑指纹**（行数 + 键集合 + 逐列值）。

其它决定报告真红还是假红的关键点：
    · 列名**排序后**参与哈希 —— 列序变化不是数据变化
    · 列的**存在性**单独校验 —— 多一列/少一列是硬错（dump 的 trade_date 坑）
    · float 走 float64 的 **bit 模式**，不用 == 比较 —— 避免 1e-15 级噪声假红
    · 空值统一归一成哨兵 —— None / NaN / "" 在本场景无差别
    · 键集合用 **set 语义** —— 行序由 sort_values 决定，与数据正确性无关
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

NULL_SENTINEL = b"\x00NULL\x00"
COL_SEP = b"\x1f"


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    """文件字节 MD5（只作"变没变"的旁证，不作等价判据）。"""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
    except OSError:
        return ""
    return h.hexdigest()


def col_digest(s: pd.Series) -> str:
    """单列指纹。按 dtype 分支，保证同一份数据跨次计算稳定。"""
    h = hashlib.md5()
    try:
        if pd.api.types.is_float_dtype(s):
            a = np.ascontiguousarray(s.to_numpy(dtype="float64", na_value=np.nan))
            # NaN 的 bit 模式在不同构造路径下可能不同 → 先记 NaN 掩码，再把 NaN 置 0
            h.update(np.ascontiguousarray(np.isnan(a)).view("uint8").tobytes())
            a = np.where(np.isnan(a), 0.0, a)
            h.update(np.ascontiguousarray(a).view("uint8").tobytes())
        elif pd.api.types.is_integer_dtype(s):
            a = s.to_numpy(dtype="int64", na_value=-2**63)
            h.update(np.ascontiguousarray(a).view("uint8").tobytes())
        elif pd.api.types.is_bool_dtype(s):
            h.update(np.ascontiguousarray(s.to_numpy(dtype=bool)).view("uint8").tobytes())
        else:
            v = s.astype(str).to_numpy(dtype=object)
            for x in v:
                if x is None or x == "nan" or x == "None" or x == "" or x == "<NA>":
                    h.update(NULL_SENTINEL)
                else:
                    h.update(str(x).encode("utf-8"))
                h.update(COL_SEP)
    except (TypeError, ValueError):
        h.update(str(s.values).encode("utf-8", errors="replace"))
    return h.hexdigest()


def frame_fingerprint(df: pd.DataFrame, keys: tuple | list | None = None,
                      sort: bool = True) -> dict:
    """一张表的逻辑指纹。

    返回 {"rows", "cols", "all_md5", "keys_md5", "cols_md5": {列: 指纹}}
    """
    out: dict = {"rows": int(len(df)), "cols": sorted(map(str, df.columns))}
    if len(df) == 0:
        out.update({"all_md5": hashlib.md5(b"EMPTY").hexdigest(),
                    "keys_md5": None, "cols_md5": {}})
        return out

    d = df
    if sort and keys:
        k = [c for c in keys if c in d.columns]
        if k:
            d = d.sort_values(k, kind="stable")
    d = d.reset_index(drop=True)

    cols_md5 = {str(c): col_digest(d[c]) for c in sorted(d.columns, key=str)}
    h = hashlib.md5()
    for name, dig in cols_md5.items():
        h.update(name.encode("utf-8"))
        h.update(COL_SEP)
        h.update(dig.encode())
        h.update(COL_SEP)
    out["all_md5"] = h.hexdigest()
    out["cols_md5"] = cols_md5

    # 键集合指纹（set 语义：排序后再哈希，与物理行序无关）
    if keys:
        k = [c for c in keys if c in d.columns]
        if k:
            kh = hashlib.md5()
            for tup in sorted(map(tuple, d[k].astype(str).to_numpy())):
                kh.update(("|".join(tup)).encode("utf-8"))
                kh.update(COL_SEP)
            out["keys_md5"] = kh.hexdigest()
    return out


def diff_columns(fp_a: dict, fp_b: dict) -> list[str]:
    """两次指纹里**变化的列名**（用于"厂商改了哪些字段"的定位）。"""
    ca, cb = fp_a.get("cols_md5") or {}, fp_b.get("cols_md5") or {}
    changed = [k for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)]
    return changed


def rows_to_json(fp: dict) -> str:
    return json.dumps(fp, ensure_ascii=False, sort_keys=True)
