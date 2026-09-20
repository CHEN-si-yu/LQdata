"""Parquet 分区存储层。

约定：
  data/<dataset>/year=YYYY/data.parquet     按年分区（时间序列）
  data/<dataset>/data.parquet               单文件（快照类、小表）

所有写入都是**原子**的：先写同目录 .tmp 再 os.replace，
在任何时刻中断都不会留下半个文件。

合并语义：按 keys 去重，新数据覆盖旧数据（keep='last'），保证
追溯修正的数据能收敛。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

TMP_SUFFIX = ".tmp"


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + TMP_SUFFIX)
    df.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
    os.replace(tmp, path)


def _cast_like(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """把 new 的列类型对齐到 old，避免多次合并后类型漂移。

    只在能无损转换时才转；转换失败则保留原样（例如 old 是 int 而 new 出现 NaN）。
    """
    for col in new.columns:
        if col not in old.columns:
            continue
        want = old[col].dtype
        if new[col].dtype == want:
            continue
        try:
            new[col] = new[col].astype(want)
        except (ValueError, TypeError):
            # 例如 old=int64 而 new 含空值 —— 提升为可空类型而不是丢数据
            if pd.api.types.is_integer_dtype(want):
                try:
                    new[col] = new[col].astype("Int64")
                except (ValueError, TypeError):
                    pass
    return new


def _unify_columns(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """两侧列集合取并集，缺失列补 NA，保证 concat 后不丢列。"""
    cols = list(dict.fromkeys([*a.columns, *b.columns]))
    if list(a.columns) != cols:
        a = a.reindex(columns=cols)
    if list(b.columns) != cols:
        b = b.reindex(columns=cols)
    return a, b


def read_parquet(path: Path) -> pd.DataFrame:
    """读取单个 parquet 文件；不存在返回空 DataFrame。"""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def upsert(
    path: Path,
    new: pd.DataFrame,
    keys: list[str] | tuple[str, ...],
    sort_by: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """把 new 合并进 path 指向的 parquet 并原子写回。返回合并后的完整表。"""
    if new is None or len(new) == 0:
        return read_parquet(path)

    old = read_parquet(path)
    if len(old) == 0:
        merged = new.copy()
    else:
        old, new = _unify_columns(old, new)
        new = _cast_like(new, old)
        merged = pd.concat([old, new], ignore_index=True)

    k = [c for c in keys if c in merged.columns]
    if k:
        merged = merged.drop_duplicates(subset=k, keep="last")

    s = [c for c in (sort_by or keys) if c in merged.columns]
    if s:
        merged = merged.sort_values(s, kind="stable").reset_index(drop=True)

    _atomic_write(merged, path)
    return merged


def overwrite(path: Path, df: pd.DataFrame, sort_by=None) -> pd.DataFrame:
    """整表覆盖（用于快照类数据）。"""
    if df is None or len(df) == 0:
        return read_parquet(path)
    if sort_by:
        s = [c for c in sort_by if c in df.columns]
        if s:
            df = df.sort_values(s, kind="stable").reset_index(drop=True)
    _atomic_write(df, path)
    return df


def year_partition_path(root: Path, dataset: str, year: int) -> Path:
    return root / dataset / f"year={year}" / "data.parquet"


def flat_path(root: Path, dataset: str) -> Path:
    return root / dataset / "data.parquet"


def list_partitions(root: Path, dataset: str) -> list[int]:
    """列出已存在的年份分区。"""
    d = root / dataset
    if not d.exists():
        return []
    years = []
    for p in d.iterdir():
        if p.is_dir() and p.name.startswith("year="):
            try:
                years.append(int(p.name.split("=", 1)[1]))
            except ValueError:
                continue
    return sorted(years)


def read_dataset(root: Path, dataset: str, columns: list[str] | None = None) -> pd.DataFrame:
    """读取整个数据集（所有年份分区）。供因子模块使用。"""
    d = root / dataset
    if not d.exists():
        return pd.DataFrame()
    files = sorted(d.glob("year=*/data.parquet"))
    flat = d / "data.parquet"
    if flat.exists():
        files.append(flat)
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    return pd.concat(frames, ignore_index=True)


def dataset_size(root: Path, dataset: str) -> int:
    d = root / dataset
    if not d.exists():
        return 0
    return sum(f.stat().st_size for f in d.rglob("*.parquet"))
