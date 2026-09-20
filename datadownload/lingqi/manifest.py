"""数据集状态记录 —— 增量更新与断点续传的依据。

每个数据集一个 JSON：state/<dataset>.json
  partitions : 各年份分区的行数/日期范围/更新时间
  coverage   : 已完整覆盖的日期区间（用于判断增量缺口）
  done       : 细粒度完成标记（如逐股票拉分钟数据时已完成的股票代码）
  last_run   : 上一次运行摘要

写入同样原子化，进程被 kill 也不会写坏。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


def _atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Manifest:
    name: str
    path: Path
    partitions: dict = field(default_factory=dict)
    coverage: list = field(default_factory=list)      # [[start, end], ...]
    done: dict = field(default_factory=dict)          # 细粒度完成集合
    last_run: dict = field(default_factory=dict)
    columns: list = field(default_factory=list)       # 列顺序（schema 稳定）
    data_start: str | None = None                     # 实测数据起点
    suspect: dict = field(default_factory=dict)       # 可疑区间（空但预期非空）
    spec_version: int = 1
    dirty: bool = False

    # ---- 载入 / 保存 ----
    @classmethod
    def load(cls, state_dir: Path, name: str) -> "Manifest":
        p = Path(state_dir) / f"{name}.json"
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                raw = {}
        else:
            raw = {}
        return cls(
            name=name,
            path=p,
            partitions=raw.get("partitions", {}),
            coverage=raw.get("coverage", []),
            done=raw.get("done", {}),
            last_run=raw.get("last_run", {}),
            columns=raw.get("columns", []),
            data_start=raw.get("data_start"),
            suspect=raw.get("suspect", {}),
            spec_version=raw.get("spec_version", 1),
        )

    def save(self) -> None:
        _atomic_json(
            {
                "name": self.name,
                "spec_version": self.spec_version,
                "partitions": self.partitions,
                "coverage": self.coverage,
                "done": self.done,
                "last_run": self.last_run,
                "columns": self.columns,
                "data_start": self.data_start,
                "suspect": self.suspect,
                "saved_at": _now(),
            },
            self.path,
        )
        self.dirty = False

    # ---- 分区 ----
    def mark_partition(self, year: int, rows: int, min_date: str | None, max_date: str | None) -> None:
        self.partitions[str(year)] = {
            "rows": int(rows),
            "min_date": min_date,
            "max_date": max_date,
            "updated_at": _now(),
        }
        self.dirty = True

    def partition_rows(self) -> int:
        return sum(int(v.get("rows", 0)) for v in self.partitions.values())

    def max_partition_date(self) -> str | None:
        ds = [v.get("max_date") for v in self.partitions.values() if v.get("max_date")]
        return max(ds) if ds else None

    def min_partition_date(self) -> str | None:
        ds = [v.get("min_date") for v in self.partitions.values() if v.get("min_date")]
        return min(ds) if ds else None

    # ---- 覆盖区间 ----
    def add_coverage(self, start: str, end: str) -> None:
        """并入一个已完整覆盖的区间，自动合并相邻/重叠区间。"""
        spans = [[s, e] for s, e in self.coverage]
        spans.append([start, end])
        spans.sort(key=lambda x: x[0])
        merged: list[list[str]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        self.coverage = merged
        self.dirty = True

    def missing_ranges(self, want_start: str, want_end: str) -> list[tuple[str, str]]:
        """在 [want_start, want_end] 里找出未被覆盖的缺口。"""
        out: list[tuple[str, str]] = []
        cursor = want_start
        for s, e in sorted(self.coverage):
            if e < cursor:
                continue
            if s > want_end:
                break
            if s > cursor:
                out.append((cursor, _prev_day(s)))
            cursor = max(cursor, _next_day(e))
            if cursor > want_end:
                break
        if cursor <= want_end:
            out.append((cursor, want_end))
        return [(s, e) for s, e in out if s <= e]

    # ---- 细粒度完成标记 ----
    def mark_done(self, bucket: str, value: str) -> None:
        self.done.setdefault(bucket, {})
        self.done[bucket][value] = 1
        self.dirty = True

    def is_done(self, bucket: str, value: str) -> bool:
        return bool(self.done.get(bucket, {}).get(value))

    def done_count(self, bucket: str) -> int:
        return len(self.done.get(bucket, {}))

    def finish_run(self, **kw) -> None:
        self.last_run = {"finished_at": _now(), **kw}
        self.dirty = True

    def summary(self) -> str:
        rows = self.partition_rows()
        parts = len(self.partitions)
        lo, hi = self.min_partition_date(), self.max_partition_date()
        rng = f"{lo}→{hi}" if lo else "空"
        return f"{rows:,} 行 / {parts} 分区 [{rng}]"


def _next_day(d: str) -> str:
    import datetime as _dt
    return (_dt.date.fromisoformat(d) + _dt.timedelta(days=1)).isoformat()


def _prev_day(d: str) -> str:
    import datetime as _dt
    return (_dt.date.fromisoformat(d) - _dt.timedelta(days=1)).isoformat()


class DumpQuota:
    """daily_dump 配额台账。

    服务端规则：**每个日期每天最多下载 10 次**，超过则该日期被封禁 3 天。
    这里把每次调用都记账，并且硬性拒绝同一 (date, level) 重复下载，
    宁可少下一次也不能触发封禁。
    """

    LIMIT_PER_DATE = 10
    BAN_DAYS = 3

    def __init__(self, state_dir: Path):
        self.path = Path(state_dir) / "dump_quota.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def _key(self, date: str, level: str) -> str:
        return f"{date}|{level}"

    def count(self, date: str, level: str) -> int:
        return int(self.data.get(self._key(date, level), {}).get("count", 0))

    def count_by_date(self, date: str) -> int:
        """按**日期**汇总所有级别的调用次数。

        ★ 2026-09-13 修：服务端规则是"一个日期一天最多只能下载 10 次"，**不分级别**
        （文档原文）。原来的 can_download 只按 (date, level) 判限，配置多个 level 时
        本地台账会低估总量 —— 例如两个级别各下 5 次，各自都"未超限"，实际已 10/10。
        """
        return sum(int(v.get("count", 0)) for k, v in self.data.items()
                   if k.split("|")[0] == date)

    def can_download(self, date: str, level: str) -> bool:
        return self.count_by_date(date) < self.LIMIT_PER_DATE

    def already_done(self, date: str, level: str) -> bool:
        return self.count(date, level) > 0

    def record(self, date: str, level: str, bytes_in: int = 0) -> None:
        k = self._key(date, level)
        rec = self.data.setdefault(k, {"count": 0, "bytes": 0})
        rec["count"] += 1
        rec["bytes"] = rec.get("bytes", 0) + bytes_in
        rec["last_at"] = _now()
        _atomic_json(self.data, self.path)

    def stats(self) -> dict:
        return {
            "dates_used": len({k.split("|")[0] for k in self.data}),
            "total_calls": sum(v.get("count", 0) for v in self.data.values()),
        }
