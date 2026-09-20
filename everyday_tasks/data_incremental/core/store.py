"""Parquet 存储层 —— 本工程自建，**产出与现有数据格式逐项对齐**。

对齐清单（对照现有 parquet 实测）：
    · 引擎   pyarrow        · 压缩   zstd
    · index=False           · 路径   <ds>/year=YYYY/data.parquet 或 <ds>/data.parquet
    · 排序   按 sort_by（缺省=keys），kind="stable"（列与值都不改，只改行序）
    · 去重   drop_duplicates(keys, keep="last")，**新数据赢**

★ 只有一条写入路径（`upsert`）会读旧表——parquet 是整文件重写，没有 append。
  日增量的行数远小于自适应阈值，所以数据都在收尾那一次显式 flush 落盘，
  调用方必须在 `mark_done` **之前**调用 flush（顺序反了会留下"状态说完成、
  数据还在内存"的永久空洞）。
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .. import paths

TMP_SUFFIX = ".tmp"


# ---------------------------------------------------------------- 底层写
def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    """写 .tmp 再原子替换 —— 任何时刻断电都不会留下半个文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + TMP_SUFFIX)
    df.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
    paths.chmod_shared(tmp)             # ★ 同组可读，见 paths.chmod_shared 的说明
    os.replace(tmp, path)


def write_atomic(path: Path, df: pd.DataFrame) -> None:
    """原子写一个 parquet（`.tmp` → `os.replace`），并设成同组可读写。

    给**非分区**的小文件用（daily_dump 的按日缓存、台账快照等）——
    它们不走 `upsert`（不需要合并/去重），但同样要「要么完整、要么不存在」。
    """
    _atomic_write(df, path)


class StoreReadError(RuntimeError):
    """**已存在的** parquet 读不出来。绝不能把它当成"空表"。"""


def read_parquet(path: Path, *, on_error: str = "raise") -> pd.DataFrame:
    """读分区。`path` 不存在 → 空表（合法，首次落库）。

    ★★ 2026-09-15 修一个**会整年丢数据**的坑：
       旧实现把"读失败"也返回空表。而 `upsert` 的语义是
       `old = read_parquet(path)` → `len(old)==0` 就当"没有旧数据" → **整文件替换**。
       于是网络盘上一次瞬时读失败 / 读到截断文件（ArrowInvalid 是 ValueError 子类，
       会被 `except ValueError` 吞掉）→ 该年分区**只剩这一天的增量行，历史整年消失**；
       随后 `_save` 还把 `partitions[year].rows` 改小，coverage 却不变 →
       `_missing_ranges` 看不到缺口 → 报告还是 ✔、退出码还是 0。

       现在：**文件在、但读不出来 → 抛 `StoreReadError`**，宁可让这一轮失败，
       也绝不写一个会把历史抹掉的"新表"。只有"文件不存在"才算空表。

       `on_error="empty"` 供只读场景（台账、删除预检）显式选择宽松行为。
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001  任何读失败都不能当成空表
        if on_error == "empty":
            return pd.DataFrame()
        raise StoreReadError(
            f"分区存在但读不出来：{path}（{type(exc).__name__}: {exc}）—— "
            f"拒绝把它当成空表继续写（那会把整个分区覆盖掉）") from exc


# ---------------------------------------------------------------- 合并辅助
def _unify_columns(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """列取并集，缺失侧补 NA。列序 = 旧表列序 + 新表独有列。

    ⚠️ 这是 schema 漂移的唯一防线：如果新数据多出一列（例如 dump 带了个
       `trade_date`），这里会**给整张旧表补一列 NA** —— 对 4400 万行的分区
       就是灾难。所以调用方必须先把 df 的列对齐好（见 dump_bridge 的 reindex）。
    """
    cols = list(old.columns) + [c for c in new.columns if c not in old.columns]
    return old.reindex(columns=cols), new.reindex(columns=cols)


def _cast_like(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """把 new 的 dtype 对齐到 old（避免 concat 时隐式升型）。转换失败就退化为可空整型。"""
    for col in new.columns:
        if col not in old.columns:
            continue
        want = old[col].dtype
        have = new[col].dtype
        if want == have:
            continue
        try:
            new[col] = new[col].astype(want)
        except (ValueError, TypeError):
            if pd.api.types.is_integer_dtype(want):
                try:
                    new[col] = new[col].astype("Int64")
                except (ValueError, TypeError):
                    pass
    return new


# ---------------------------------------------------------------- 对外写
def upsert(path: Path, new: pd.DataFrame, keys: tuple, sort_by: tuple | None = None,
           date_field: str | None = None, replace_suffix: bool = False) -> pd.DataFrame:
    """把 new 合并进 path 指向的分区。返回合并后的表。

    语义（必须与旧工程完全一致，否则两份数据会分叉）：
        old + new → 按 keys 去重（keep="last" → 新数据赢）→ 按 sort_by 稳定排序 → 原子写

    ★★ `replace_suffix=True` 的尾部追加快路径（2026-09-15 加；2026-09-17 收语义）：
       动机：`stock_cyq_chips` 的 2026 分区 **5,553 万行**，通用路径的
       `concat + drop_duplicates + sort_values` 实测 **51 秒**，占一轮日更的一半。
       用它可以省掉这次全表排序。

       **语义（2026-09-17 起）：`date >= n_min` 的后缀里，只有与 `new`
       主键相同的旧行被 new 取代，其余旧行原样保留**（`n_min = new[date_field].min()`）。
       等价于通用路径的 `drop_duplicates(keys, keep="last")` —— 因为 new 的日期
       全在 `[n_min, +∞)`，重复只可能发生在后缀内。

       ⚠️ 旧语义是"后缀**整段**被 new 取代（含 new 没覆盖到的行）"，它有两个后果：
         · 一次漏发/撤数（服务端某天少给几只票）会**静默删掉本地那些行** ——
           与用户 2026-09-15 的口径冲突（「历史数据消失…不需要特意删除以前的数据，
           只打报告」）；
         · 分多次 flush 时后一批会把前一批抹掉（2026-09-15 实测：09-11 只剩
           1054/3484 只）。
       收紧后这两个风险都不存在了 —— 这个开关现在**只有性能含义，没有正确性含义**，
       调用方不必再证明"new 覆盖了整个后缀"。
       （仍然保留调用方的单次全量 flush 约定：分多次 flush 只会让请求白跑。）
    """
    if new is None or len(new) == 0:
        return read_parquet(path)

    k = [c for c in keys if c in new.columns]
    if keys and len(k) != len(keys):
        missing = [c for c in keys if c not in new.columns]
        raise ValueError(f"keys 里这些列在数据里不存在：{missing}（会导致静默不去重，必须修）")

    old = read_parquet(path)
    if len(old) == 0:
        merged = new.copy()
        if k:
            merged = merged.drop_duplicates(subset=k, keep="last")
        s = [c for c in (sort_by or keys) if c in merged.columns]
        if s:
            merged = merged.sort_values(s, kind="stable")
        merged = merged.reset_index(drop=True)
    else:
        old, new2 = _unify_columns(old, new)
        new2 = _cast_like(new2, old)

        s = [c for c in (sort_by or keys) if c in old.columns]
        merged = None
        if (replace_suffix and date_field and s and s[0] == date_field
                and date_field in new2.columns and date_field in old.columns and len(new2)):
            try:
                if k:
                    # 快路径不跨表去重，new 自己要先去重（它很小，代价可忽略）
                    new2 = new2.drop_duplicates(subset=k, keep="last")
                n_min = new2[date_field].min()
                pre = old[old[date_field] < n_min]
                suf = old[old[date_field] >= n_min]
                # ★★ 2026-09-17 语义收紧：**只丢掉与 new 主键逐列相同的旧行**
                #   （= 这些行已被 new 取代），**保留 new 没覆盖到的行**。
                #   旧语义是"date >= n_min 的旧行全部丢弃"，于是一次漏发/撤数
                #   （服务端某天少给了几只票）就会**静默删掉本地那些行** ——
                #   与用户 2026-09-15 的口径直接冲突：
                #   「遇上历史数据消失…**不需要特意删除以前的数据**，只打报告」。
                #   现在 §尾窗替换不再有任何删除能力，只有"新数据赢"。
                if k and len(suf):
                    ki = pd.MultiIndex.from_frame(suf[k].astype(str))
                    ni = pd.MultiIndex.from_frame(new2[k].astype(str))
                    suf = suf[~ki.isin(ni)]
                # prefix 全部 < n_min、suffix 全部 >= n_min → **只排 suffix** 就保持全局有序
                tail = pd.concat([suf, new2], ignore_index=True)
                if s:
                    tail = tail.sort_values(s, kind="stable")
                merged = pd.concat([pre, tail], ignore_index=True).reset_index(drop=True)
            except (TypeError, ValueError):
                merged = None

        if merged is None:
            merged = pd.concat([old, new2], ignore_index=True)
            if k:
                merged = merged.drop_duplicates(subset=k, keep="last")
            if s:
                merged = merged.sort_values(s, kind="stable").reset_index(drop=True)
            else:
                merged = merged.reset_index(drop=True)

    _atomic_write(merged, path)
    return merged


def overwrite(path: Path, df: pd.DataFrame, sort_by: tuple | None = None) -> pd.DataFrame:
    """整表替换（快照用）。空 df **不写盘**（防止一次空响应把已有数据清空）。"""
    if df is None or len(df) == 0:
        return read_parquet(path)
    out = df.sort_values(list(sort_by), kind="stable").reset_index(drop=True) if sort_by else df.reset_index(drop=True)
    _atomic_write(out, path)
    return out


# ---------------------------------------------------------------- 查询辅助
def partition_path(dataset: str, year: int) -> Path:
    return paths.year_partition_path(dataset, year)


def flat_path(dataset: str) -> Path:
    return paths.flat_path(dataset)


def list_years(dataset: str) -> list[int]:
    d = paths.DATA_ROOT / dataset
    if not d.exists():
        return []
    out = []
    for p in d.iterdir():
        if p.is_dir() and p.name.startswith("year="):
            try:
                out.append(int(p.name[5:]))
            except ValueError:
                pass
    return sorted(out)


def disk_max_date(dataset: str, date_field: str) -> str | None:
    """磁盘上该数据集的最大日期（读所有年分区，只取日期列）。零 API 请求。"""
    import pyarrow.parquet as pq

    mx: str | None = None
    years = list_years(dataset)
    targets = [partition_path(dataset, y) for y in years] or [flat_path(dataset)]
    for f in targets:
        if not f.exists():
            continue
        try:
            pf = pq.ParquetFile(f)
            if date_field not in pf.schema_arrow.names:
                continue
            # 逐批次扫（大分区不能整列载入）
            for batch in pf.iter_batches(batch_size=2_000_000, columns=[date_field]):
                s = batch.column(0).to_pandas().astype(str).str[:10]
                if len(s):
                    m = s.max()
                    if mx is None or m > mx:
                        mx = m
        except (OSError, ValueError):
            continue
    return mx
