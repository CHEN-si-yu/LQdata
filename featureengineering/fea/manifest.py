"""每个因子一个状态文件 `state/<factor>.json` —— 增量与断点续传的依据。

**一个因子一个文件**（不是全局一个）：因子是独立的开发单元，
明天新加第 11 个因子时它自己的 coverage 是空的，会从头建；
而不会因为「全局说这段日期算过了」就留下一个空洞。

内容：
  recipe         : 因子逻辑指纹。变了就拒绝增量，必须 --rebuild
  coverage       : 已完整覆盖的日期区间（[[a,b], ...]，自动合并相邻）
  partitions     : 各年份分区的行数/非空数/日期范围
  input_watermark: 上运行时的上游水位（行数 + 最新公告日）
  last_run       : 上一次运行摘要
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _next_day(d: str) -> str:
    return (_dt.date.fromisoformat(d) + _dt.timedelta(days=1)).isoformat()


def _prev_day(d: str) -> str:
    return (_dt.date.fromisoformat(d) - _dt.timedelta(days=1)).isoformat()


@dataclass
class Manifest:
    name: str
    path: Path
    recipe: str = ""
    coverage: list = field(default_factory=list)
    partitions: dict = field(default_factory=dict)
    input_watermark: dict = field(default_factory=dict)
    last_run: dict = field(default_factory=dict)

    @classmethod
    def load(cls, state_dir: Path, name: str) -> "Manifest":
        p = Path(state_dir) / f"{name}.json"
        raw = {}
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                raw = {}
        return cls(
            name=name, path=p,
            recipe=raw.get("recipe", ""),
            coverage=raw.get("coverage", []),
            partitions=raw.get("partitions", {}),
            input_watermark=raw.get("input_watermark", {}),
            last_run=raw.get("last_run", {}),
        )

    def save(self) -> None:
        _atomic_json({
            "name": self.name,
            "recipe": self.recipe,
            "coverage": self.coverage,
            "partitions": self.partitions,
            "input_watermark": self.input_watermark,
            "last_run": self.last_run,
            "saved_at": _now(),
        }, self.path)
        self.dirty = False

    # ---- 覆盖区间 ----
    def add_coverage(self, start: str, end: str) -> None:
        spans = [[s, e] for s, e in self.coverage]
        spans.append([start, end])
        spans.sort(key=lambda x: x[0])
        merged: list[list[str]] = []
        for s, e in spans:
            if merged and s <= _next_day(merged[-1][1]):
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        self.coverage = merged

    def missing_ranges(self, want_start: str, want_end: str) -> list[tuple[str, str]]:
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

    def covered_until(self) -> str | None:
        return max((e for _, e in self.coverage), default=None)

    # ---- 分区 ----
    def mark_partition(self, year: int, rows: int, nonnull: int,
                       min_date: str | None, max_date: str | None) -> None:
        self.partitions[str(year)] = {
            "rows": int(rows), "nonnull": int(nonnull),
            "min_date": min_date, "max_date": max_date, "updated_at": _now(),
        }

    def partition_rows(self) -> int:
        return sum(int(v.get("rows", 0)) for v in self.partitions.values())

    def nonnull_ratio(self) -> float:
        tot = self.partition_rows()
        nn = sum(int(v.get("nonnull", 0)) for v in self.partitions.values())
        return (nn / tot) if tot else 0.0

    def summary(self) -> str:
        rows = self.partition_rows()
        if not rows:
            return "—"
        ds = [v.get("max_date") for v in self.partitions.values() if v.get("max_date")]
        de = [v.get("min_date") for v in self.partitions.values() if v.get("min_date")]
        return (f"{rows:,}行 {len(self.partitions)}年 "
                f"[{min(de) if de else '?'}→{max(ds) if ds else '?'}] "
                f"非空{self.nonnull_ratio():.0%}")

    def reset(self, recipe: str) -> None:
        self.recipe = recipe
        self.coverage = []
        self.partitions = {}
        self.input_watermark = {}
