"""因子落盘：按年分区 + 原子写 + 按天合并。

统一输出格式（所有因子完全一致）：
    data/factors/<name>/year=YYYY/data.parquet
        trade_date  string   "2026-09-11"
        stock_code  string   "600000.SH"
        value       float32  原始因子值（可解释、可再标准化）
        rank        float32  当日截面百分位 [0,1]（先 winsorize 再 rank）

写盘用「读旧分区 → 去掉要覆盖的交易日 → 拼新数据 → 原子写回」。
每个分区在一次运行里**最多写一次**（引擎按年组织计算），
所以模块① 踩过的 `upsert` O(n²) 整文件重写坑在这里不会发生。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

import pandas as pd

COLUMNS = ["trade_date", "stock_code", "value", "rank"]

# ★ 用户的硬性要求：「互相之间保持统一的格式，仅有日期允许长短不一样」。
#   列名/列序/dtype/目录结构/分区方式/语义**全部逐字相同**，唯一允许的差异是起止日期。
#   所以 dtype 在这里写死 —— 不能依赖各因子自己碰巧一致。
MARKET_COLUMNS = ["trade_date", "value"]
MARKET_DTYPES = {"trade_date": "string", "value": "float32"}

DTYPES = {"trade_date": "string", "stock_code": "string",
          "value": "float32", "rank": "float32"}


def _assert_schema(df: pd.DataFrame) -> pd.DataFrame:
    """写盘前的最后一道闸门：列序 + dtype 必须与契约完全一致。"""
    columns = MARKET_COLUMNS if list(df.columns) == MARKET_COLUMNS else COLUMNS
    dtypes = MARKET_DTYPES if columns == MARKET_COLUMNS else DTYPES
    if list(df.columns) != columns:
        raise ValueError(f"落盘列必须是 {COLUMNS}，实际是 {list(df.columns)}。"
                         f"（新因子不许自定义输出列 —— 见 fea/store.py 的统一格式契约）")
    for c, want in dtypes.items():
        got = str(df[c].dtype)
        if got != want:
            raise ValueError(f"列 {c} 的 dtype 必须是 {want}，实际是 {got}")
    return df


def year_path(root: Path, name: str, year: int) -> Path:
    return root / name / f"year={year}" / "data.parquet"


def factor_years(root: Path, name: str) -> list[int]:
    d = root / name
    if not d.is_dir():
        return []
    out = []
    for p in d.iterdir():
        if p.is_dir() and p.name.startswith("year="):
            try:
                out.append(int(p.name.split("=", 1)[1]))
            except ValueError:
                continue
    return sorted(out)


def empty_frame() -> pd.DataFrame:
    """**带正确 dtype** 的空表。

    ★ 不能返回 `pd.DataFrame(columns=COLUMNS)` —— 那是全 object dtype，
      下游 `np.isfinite(df["value"])` 会抛
      `TypeError: ufunc 'isfinite' not supported for the input types`，
      而病根（某因子某年整年无值）离报错点十万八千里。
    """
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in DTYPES.items()})


def read_year(root: Path, name: str, year: int) -> pd.DataFrame:
    """读某个年分区，**并校正 dtype**。

    ★ 校正这一步不能省：早期版本写盘时 `stock_code` 被转成 `category`、
      空分区读回来是全 object dtype。下游 `np.isfinite(df["value"])` 会抛
      `TypeError: ufunc 'isfinite' not supported` —— 报错点离病根极远，
      排查成本很高（实测两个 Agent 都被误导过）。
      在**读取处**统一 dtype，让病根在这里就消失。
    """
    p = year_path(root, name, year)
    if not p.exists():
        return empty_frame()
    df = pd.read_parquet(p)
    if list(df.columns) not in (COLUMNS, MARKET_COLUMNS):
        raise ValueError(f"未知因子文件格式：{p}: {list(df.columns)}")
    dtypes = MARKET_DTYPES if list(df.columns) == MARKET_COLUMNS else DTYPES
    for c, t in dtypes.items():
        if str(df[c].dtype) != t:
            df[c] = df[c].astype(t)
    return df


def read_factor(root: Path, name: str, years: list[int] | None = None) -> pd.DataFrame:
    ys = factor_years(root, name) if years is None else years
    frames = [read_year(root, name, y) for y in ys]
    frames = [f for f in frames if len(f)]
    if not frames:
        return empty_frame()
    return pd.concat(frames, ignore_index=True)


def _atomic_write(df: pd.DataFrame, path: Path, compression: str = "zstd") -> None:
    """原子写 parquet。临时文件名带 PID —— 与 `manifest._atomic_json` 同一条理由。

    同一进程内不同任务写的是不同路径，所以本函数今天没有实际竞态；但把 tmp 名
    做成**确定性的**是同一个隐患：只要将来出现「两个进程写同一个分区」（例如
    两个 `main.py run` 实例、或审计与生产并行），就会互相截断、`os.replace` 抛
    `FileNotFoundError`。加 PID 的成本是零，就地一起修掉。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    try:
        df.to_parquet(tmp, engine="pyarrow", compression=compression, index=False)
        os.replace(tmp, path)
    except BaseException:
        try: tmp.unlink()
        except OSError: pass
        raise


# ================================================================ 写盘内容比对
#: 上一次 `upsert_year` 的写盘判定（供 `Engine.run_year` 取用，随 manifest 落盘）。
#: ★ 用模块级变量而不是改返回值：`upsert_year` 的调用点有多处，改签名波及面太大；
#:   调用方是"写完立刻读"，每年一个独立进程，不存在并发歧义。
LAST_WRITE: dict = {}


def _content_identical(old: pd.DataFrame | None, new: pd.DataFrame,
                       keys: list[str]) -> bool:
    """两份分区是否**逐格相同**（按主键排序后比数值与字符串）。

    为什么比内容而不是比文件字节：parquet 每次写出的**字节**都不保证一致
    （压缩块边界、页脚、写盘时间戳），而"内容有没有变"才是真正关心的语义。

    ★ 这是「历史不可变」守卫的**观测层**（2026-09-29 加）：
      · 相同 ⇒ `upsert_year` 跳过写盘（省 IO；且 mtime 不变 ⇒ 下次 plan 的 L1a
        不会把"逐字节重写"误判成"分区被外部改动"）；
      · 不同 ⇒ 照写（本轮不拦截），但把差异摘要记进 `LAST_WRITE`，
        由 `run_year` 落到 manifest、`daily.py` 汇总进 `report.json.partition_rewrites`。
    """
    if old is None or len(old) != len(new):
        return False
    if list(old.columns) != list(new.columns):
        return False
    if len(old) == 0:
        return True
    a = old.sort_values(keys, kind="stable").reset_index(drop=True)
    b = new.sort_values(keys, kind="stable").reset_index(drop=True)
    # 用 pandas 自己的 `Series.equals`：dtype 感知、NaN==NaN、
    # 且对 `string` 扩展列安全（`np.array_equal(..., equal_nan=True)` 在
    # 字符串列上会 `TypeError: ufunc 'isnan' not supported`，实测踩过）。
    for c in a.columns:
        if not a[c].equals(b[c]):
            return False
    return True


def upsert_year(root: Path, name: str, year: int, new: pd.DataFrame,
                compression: str = "zstd", prune_after: int | None = None,
                prune_before: int | None = None,
                replace_days: "np.ndarray | None" = None) -> pd.DataFrame:
    """把 new 合并进该年份分区；new 覆盖同 (trade_date, stock_code) 的旧行。

    ★ `prune_after`（YYYYMMDD）：顺便删掉分区里**晚于这个日期**的旧行。
      为什么需要：合并语义是「用新数据里有值的日期替换旧行，其余旧行原样保留」——
      所以如果某次运行**少写了尾巴**（比如右端点从 09-15 收到 09-14），
      落在新数据之外的旧行会**静默留存**。实测 `accruals_ratio` 等 3 个因子
      在被 OOM 中断的那一轮留下了 2026-09-15 的行，后续跑到 09-14 的运行
      并没有把它们清掉。传 `self._run_end` 就能保证分区的末尾不超过本次运行的右端点。

    ★ `prune_before`（YYYYMMDD）：对称地删掉**早于因子声明起点**的旧行。
      坑长什么样（2026-09-17 实测）：`dividend_yield_3y_avg` 的起点从默认改成实测的
      `2013-01-11` 之后，重建只重算了 ≥ 起点的那几天，而合并语义会**原样保留**
      分区里更早的旧行 —— 2013 分区因此留下 5 天（17,420 行）起点之前的 NaN 行，
      被 `main.py check` 的「≥起点」判据抓出来（只有 1 个因子中招，但这是**结构性**的：
      任何"起点后移"的因子都会留这种残留）。传因子起点即可自愈。
    """
    path = year_path(root, name, year)
    if new is None or len(new) == 0:
        return read_year(root, name, year)

    columns = MARKET_COLUMNS if list(new.columns) == MARKET_COLUMNS else COLUMNS
    dtypes = MARKET_DTYPES if columns == MARKET_COLUMNS else DTYPES
    _assert_schema(new)
    keys = ["trade_date"] + (["stock_code"] if "stock_code" in columns else [])
    if new.duplicated(keys).any():
        raise ValueError(f"{name}: 重复主键")
    new = new[columns].copy()
    old = read_year(root, name, year)
    if len(old):
        old = old[columns].copy()
    old_prev = old          # ★ 写盘前的分区原样（下面 old 会被"按本次重算的天"裁掉一段）
    if len(old):
        # ★★ 覆盖语义必须按「**本次重算了哪些天**」定，而不是按「新数据里出现了哪些天」定。
        #   两者只在一种情况下不同，但那一种**会静默留脏数据**：
        #   某天重算后**整列都是 NaN**（引擎不留行）→ 新数据里根本没有这一天 →
        #   旧行原样保留 → 该天的旧值永远删不掉。
        #   实测（2026-09-15）：`label_ret_20d` 修好尾段掩码后，09-04~09-08 的旧值
        #   （用前向填充的陈旧价算出的假标签）仍留在分区里，重算也清不掉。
        #   `replace_days` 由 `Engine.run_year` 传入（= 本次计划重算的交易日）。
        if replace_days is not None:
            days = pd.Index([f"{int(d) // 10000:04d}-{int(d) // 100 % 100:02d}-{int(d) % 100:02d}"
                             for d in np.asarray(replace_days).ravel()])
        else:
            days = pd.Index(new["trade_date"].unique())
        old = old[~old["trade_date"].isin(days)]
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new

    if prune_after:
        cut = f"{prune_after // 10000:04d}-{(prune_after // 100) % 100:02d}-{prune_after % 100:02d}"
        n0 = len(merged)
        merged = merged[merged["trade_date"] <= cut]
        if len(merged) != n0:
            logging.getLogger("fea.store").warning(
                "%s/%d：裁掉 %d 行晚于 %s 的陈旧行", name, year, n0 - len(merged), cut)
    if prune_before:
        cut = f"{prune_before // 10000:04d}-{(prune_before // 100) % 100:02d}-{prune_before % 100:02d}"
        n0 = len(merged)
        merged = merged[merged["trade_date"] >= cut]
        if len(merged) != n0:
            logging.getLogger("fea.store").warning(
                "%s/%d：裁掉 %d 行早于声明起点 %s 的陈旧行", name, year, n0 - len(merged), cut)
    merged = merged.sort_values(keys, kind="stable").reset_index(drop=True)
    # ★ 统一 dtype 后再写：以前这里把 stock_code 转成 `category`（省内存），
    #   结果是 parquet 里成 dictionary 编码、读回来是 category 而不是 string ——
    #   跨因子不一致，违背「统一格式」。现在写盘前强制成契约里的 dtype。
    for c, t in dtypes.items():
        merged[c] = merged[c].astype(t)
    _assert_schema(merged)
    # ★★ 2026-09-29：写盘前的**内容比对**（「历史不可变」守卫的观测层，见 _content_identical）。
    #   相同 ⇒ 跳过写盘；不同 ⇒ 照写，但把差异摘要放进 LAST_WRITE 供上层落台账。
    if len(old_prev):
        for c, t in dtypes.items():
            if c in old_prev.columns and old_prev[c].dtype != t:
                old_prev[c] = old_prev[c].astype(t)
    identical = _content_identical(old_prev, merged, keys)
    LAST_WRITE.clear()
    LAST_WRITE.update({
        "factor": name, "year": int(year), "identical": bool(identical),
        "rows_old": int(len(old_prev)), "rows_new": int(len(merged)),
        "nonnull_old": int(np.isfinite(old_prev["value"].to_numpy()).sum()) if len(old_prev) else 0,
        "nonnull_new": int(np.isfinite(merged["value"].to_numpy()).sum()),
    })
    if identical:
        log = logging.getLogger("fea.store")
        log.debug("%s/%d：内容与旧分区逐格相同，跳过写盘", name, year)
        return merged
    _atomic_write(merged, path, compression)
    return merged


def factor_size(root: Path, name: str) -> int:
    d = root / name
    if not d.is_dir():
        return 0
    return sum(f.stat().st_size for f in d.rglob("*.parquet"))
