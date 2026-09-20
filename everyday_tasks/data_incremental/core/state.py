"""Manifest 状态层 —— 自建，但**JSON 形状与现有 `datadownload/state/*.json` 逐字段一致**。

为什么形状必须一致：`coverage` / `done` 是"哪些数据已经在本地"的唯一真相，
下游（模块② 因子工程的 `upstream.py`、以及 datadownload 自己的 `audit.py` /
`gate.py` / `progress_bar.py`）都直接读它。形状一变，那些工具全部失明。

    {
      "name": "stock_daily",
      "spec_version": 1,
      "partitions": {"2026": {"rows": 929573, "min_date": "...", "max_date": "...", "updated_at": "..."}},
      "coverage":   [["2010-01-01", "2010-01-15"], ...],       # 已确认"无缺口"的区间
      "done":       {"dates": {...}, "stocks": {...}, "entities": {...}, "dumps": {...}},
      "last_run":   {"finished_at": "...", "status": "ok", "rows": 0, "seconds": 1.2},
      "columns":    ["trade_date", "stock_code", ...],          # 首次落库的列序，作为规范
      "data_start": "2010-01-01",
      "suspect":    {"区间或键": 计数},
      "saved_at":   "..."
    }

⚠️ 两个坑（旧工程都踩过）：
   1. `done[bucket]` 是 **dict**（`{值: 1}`），不是 list。清空用 `= {}`。
      `is_done` 的实现是 `self.done.get(bucket, {}).get(value)`，写成 list 会 AttributeError。
   2. `suspect` 在 14 个现存文件里是 `[]`（list），而代码期望 dict。本工程在 load 时
      强制归一成 dict，**这是零信息损失的**（它们本来就是空的）。
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import paths

_JSON_SUFFIX = ".json.tmp"


class ManifestUnreadable(RuntimeError):
    """manifest 存在但读不出来 / 内容坏了 —— **绝不能当成空 manifest**。"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=_JSON_SUFFIX)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        paths.chmod_shared(tmp)         # ★ mkstemp 建出来是 0600，必须放开同组读
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class Manifest:
    """一个数据集的状态文件。读-改-写都在内存里，`save()` 才落盘。"""

    def __init__(self, name: str, path: Path, raw: dict):
        self.name = name
        self.path = path
        self._extra: dict = {}                       # 保留未识别的字段，回写时不丢
        self.partitions: dict = raw.get("partitions") or {}
        self.coverage: list = raw.get("coverage") or []
        # ★ suspect 类型归一：现存 14 个文件是 []，代码期望 dict
        susp = raw.get("suspect")
        self.suspect: dict = susp if isinstance(susp, dict) else {}
        done = raw.get("done") or {}
        self.done: dict = {k: (v if isinstance(v, dict) else {}) for k, v in done.items()} if isinstance(done, dict) else {}
        self.last_run: dict = raw.get("last_run") or {}
        self.columns: list = raw.get("columns") or []
        self.data_start: str | None = raw.get("data_start")
        for k, v in raw.items():
            if k not in ("name", "partitions", "coverage", "suspect", "done",
                         "last_run", "columns", "data_start", "saved_at"):
                self._extra[k] = v

    # ------------------------------------------------------------ 读写
    @classmethod
    def load(cls, name: str) -> "Manifest":
        """读 manifest。

        ★★ 2026-09-15 修一个**会清空全部状态**的静默降级：
           旧实现把"读不出来"（含 PermissionError）和"JSON 坏了"都当成**空 manifest**，
           于是 `partitions={}` / `coverage=[]` / `done={}` / `columns=[]`。
           一轮跑完再 `man.save()` → **真实的 coverage/done/columns 被覆盖成空**，
           `_missing_ranges` 随即把"2010 年至今"整段当成缺口 → 请求风暴，
           而且此前的覆盖记录**永久丢失**。

           实测触发场景（不是假设）：本工程被 root 和 claude 两个用户先后执行，
           root 写出的 manifest 是 `0600 root:root`，claude 读不到 →
           下一个 claude 跑的夜晚就会踩中。所以现在**读不出来就抛错**，
           并且只在"文件确实不存在"时才返回空 manifest（那是合法的首次落库）。
        """
        p = paths.manifest_path(name)
        raw: dict = {}
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8")
            except OSError as exc:
                raise ManifestUnreadable(
                    f"manifest 存在但读不出来：{p}（{type(exc).__name__}: {exc}）\n"
                    f"  拒绝把它当成空 manifest —— 那会让 coverage/done/columns 被清空重写。\n"
                    f"  最常见原因：**不同用户交替执行本工程**，文件属主/权限不一致。\n"
                    f"  当前进程 uid={os.getuid()}，文件 {oct(p.stat().st_mode)[-3:]} "
                    f"uid={p.stat().st_uid}。\n"
                    f"  修法：`chgrp -R root {paths.STATE_ROOT}` 然后 "
                    f"`chmod -R g+rw {paths.STATE_ROOT}`（claude 在 root 组里）。"
                ) from exc
            try:
                obj = json.loads(text)
                raw = obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError as exc:
                raise ManifestUnreadable(
                    f"manifest 内容坏了：{p}（{exc}）\n"
                    f"  **不要直接删它** —— 那等于把 coverage/done 全丢。\n"
                    f"  先看同目录有没有 `{p.name}.*.bak` 可以还原；"
                    f"确实要重建，再改成手动确认过的空 manifest。"
                ) from exc
        return cls(name, p, raw)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "spec_version": self._extra.get("spec_version", 1),
            "partitions": self.partitions,
            "coverage": self.coverage,
            "done": self.done,
            "last_run": self.last_run,
            "columns": self.columns,
            "data_start": self.data_start,
            "suspect": self.suspect,
            "saved_at": _now(),
            **{k: v for k, v in self._extra.items() if k != "spec_version"},
        }

    def save(self) -> None:
        _atomic_json(self.path, self.to_dict())

    # ------------------------------------------------------------ 分区
    def mark_partition(self, year: int, rows: int, min_date: str | None, max_date: str | None) -> None:
        self.partitions[str(year)] = {
            "rows": int(rows),
            "min_date": str(min_date)[:10] if min_date else None,
            "max_date": str(max_date)[:10] if max_date else None,
            "updated_at": _now(),
        }

    def partition_rows(self) -> int:
        return sum(int(v.get("rows") or 0) for v in self.partitions.values())

    def max_partition_date(self) -> str | None:
        ds = [v.get("max_date") for v in self.partitions.values() if v.get("max_date")]
        return max(ds) if ds else None

    def min_partition_date(self) -> str | None:
        ds = [v.get("min_date") for v in self.partitions.values() if v.get("min_date")]
        return min(ds) if ds else None

    def summary(self) -> str:
        rows = self.partition_rows()
        rng = ""
        mn, mx = self.min_partition_date(), self.max_partition_date()
        if mn or mx:
            rng = f" [{mn}→{mx}]"
        return f"已有 {rows:,} 行 / {len(self.partitions)} 分区{rng}"

    # ------------------------------------------------------------ done 标记
    def is_done(self, bucket: str, value: str) -> bool:
        return bool(self.done.get(bucket, {}).get(value))

    def mark_done(self, bucket: str, value: str, flag: int = 1) -> None:
        self.done.setdefault(bucket, {})[value] = flag

    def clear_done(self, bucket: str) -> None:
        """★ 清空必须用 dict 不是 list（见模块 docstring 的坑 1）。"""
        self.done[bucket] = {}

    def done_count(self, bucket: str) -> int:
        return len(self.done.get(bucket, {}))

    # ------------------------------------------------------------ 覆盖区间
    def add_coverage(self, start: str, end: str) -> None:
        """并入一段"我确认这段没有缺口"的区间，自动合并重叠/相邻。"""
        s, e = str(start)[:10], str(end)[:10]
        cur = [(a, b) for a, b in (self.coverage or [])]
        cur.append((s, e))
        cur.sort()
        merged: list[list[str]] = []
        for a, b in cur:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        self.coverage = merged

    def trim_coverage_from(self, cutoff: str) -> int:
        """把 coverage 里 >= cutoff 的部分裁掉（让增量重新把它当缺口）。返回裁掉的行数。"""
        c = str(cutoff)[:10]
        kept, dropped = [], 0
        for a, b in (self.coverage or []):
            if a >= c:
                dropped += 1
                continue
            if b >= c:
                kept.append([a, _prev_day(c)])
                dropped += 1
            else:
                kept.append([a, b])
        self.coverage = kept
        return dropped

    # ------------------------------------------------------------ 运行记录
    def finish_run(self, **kw: Any) -> None:
        self.last_run = {"finished_at": _now(), **kw}

    # ------------------------------------------------------------ suspect
    def mark_suspect(self, key: str, n: int = 1) -> None:
        self.suspect[key] = int(self.suspect.get(key, 0)) + n

    def clear_suspect(self) -> int:
        n = len(self.suspect)
        self.suspect = {}
        return n


def _prev_day(s: str) -> str:
    from datetime import date as _d
    from datetime import timedelta
    return (_d.fromisoformat(s[:10]) - timedelta(days=1)).isoformat()


def load_calendar() -> list[str]:
    """交易日历（升序，只要 is_open==1 的）。读 `data/basic_calendar/`。"""
    import pyarrow.parquet as pq

    f = paths.flat_path("basic_calendar")
    if not f.exists():
        return []
    try:
        t = pq.read_table(f, columns=["date", "is_open"])
    except (OSError, ValueError, KeyError):
        return []
    days = t.column("date").to_pylist()
    opens = t.column("is_open").to_pylist()
    return sorted({str(dv)[:10] for dv, op in zip(days, opens) if dv is not None and int(op or 0) == 1})


def shift_back(cal: list[str], anchor: str, n: int) -> str | None:
    """从 anchor 在日历里往回数 n 个交易日。"""
    idx = None
    for i, x in enumerate(cal):
        if x <= anchor:
            idx = i
        else:
            break
    if idx is None:
        return None
    j = idx - n
    return cal[j] if j >= 0 else None


def calendar_ext_load() -> list[str]:
    """从服务端返回值自愈出来的交易日（补充 basic_calendar 的不足）。"""
    p = paths.CALENDAR_EXT
    if not p.exists():
        return []
    try:
        v = json.loads(p.read_text(encoding="utf-8"))
        return sorted(str(x)[:10] for x in v) if isinstance(v, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def calendar_ext_add(days: list[str]) -> int:
    cur = set(calendar_ext_load())
    before = len(cur)
    cur.update(str(d)[:10] for d in days if d)
    if len(cur) != before:
        _atomic_json(paths.CALENDAR_EXT, sorted(cur))
    return len(cur) - before


def effective_calendar() -> list[str]:
    """日历 ∪ 自愈日历。闸门的候选日枚举用它。"""
    return sorted(set(load_calendar()) | set(calendar_ext_load()))


# ================================================================ dump 配额
class DumpQuota:
    """daily_dump 的配额台账 —— 与旧工程**共用同一份文件**（配额是服务端级的）。

    规则：**每个日期每天最多 10 次**（不分级别），超过则该日期**被封禁 3 天**。
    ★ 必须在**发请求之前**记账：服务端按**请求次数**计费，超时/重试也扣额度。

    ★★ 2026-09-17 修两处（都是"台账自己把日期封死"）：
      ① **按日历日分桶**。旧实现 `count_by_date` 把该数据日期**有史以来**的次数求和，
         而计数**从不重置** —— 服务端是"每天重新给 10 次"，我们这边却是
         "一辈子累计 10 次就永久拒绝"。而耗配额的**只有坏日子**（完整日 0 请求），
         等于"坏得最厉害的那天也最快变得不可修复"。现在按 `by_day`
         （请求发生在哪一天）记账，与"封禁 3 天"的语义对齐；旧格式（无 by_day）
         仍按累计口径兜底，保持与旧工程可互相读取。
      ② **按 HTTP 尝试次数记账**。旧实现一次 `client.call` 只记 1 次，
         而 `call` 内部最多重试 5 次（固定档位 1/3/5/10/30s）→ 真实请求 6 次、
         台账只记 1 次，最多**差 6 倍**。厂商按请求次数计，超了直接封 3 天。
         现在调用方按 `stats["requests"]` 的实际增量补记。
    """

    LIMIT_PER_DATE = 10
    BAN_DAYS = 3

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else paths.DUMP_QUOTA_FILE
        self.data: dict = {}
        if self.path.exists():
            try:
                v = json.loads(self.path.read_text(encoding="utf-8"))
                self.data = v if isinstance(v, dict) else {}
            except (json.JSONDecodeError, OSError):
                self.data = {}

    def count_by_date(self, date: str) -> int:
        """按**数据日期**汇总所有级别的**历史累计**次数（展示用）。"""
        d = str(date)[:10]
        return sum(int(v.get("count") or 0) for k, v in self.data.items() if k.startswith(d + "|"))

    def count_on(self, date: str, day: str | None = None) -> int:
        """**某一天（默认今天）**对某个数据日期已经取了几次 —— 服务端的真实口径。"""
        d = str(date)[:10]
        day = day or _now()[:10]
        key = f"{d}|"
        n = 0
        for k, v in self.data.items():
            if not k.startswith(key):
                continue
            bd = v.get("by_day")
            if isinstance(bd, dict):
                n += int(bd.get(day) or 0)
            else:
                # 旧格式（没有 by_day）：退回累计口径（只会更保守）
                n += int(v.get("count") or 0)
        return n

    def can_download(self, date: str, level: str = "5min") -> bool:
        return self.count_on(date) < self.LIMIT_PER_DATE

    def record(self, date: str, level: str, n: int = 1, bytes_in: int = 0) -> None:
        n = int(n)
        if n <= 0:
            return
        day = _now()[:10]
        key = f"{str(date)[:10]}|{level}"
        cur = self.data.get(key) or {}
        cur["count"] = int(cur.get("count") or 0) + n
        cur["bytes"] = int(cur.get("bytes") or 0) + bytes_in
        cur["last_at"] = _now()
        bd = cur.get("by_day")
        if not isinstance(bd, dict):
            bd = {}
        bd[day] = int(bd.get(day) or 0) + n
        # 只保留 BAN_DAYS + 1 天 —— 更早的计数服务端已经不算数了（避免文件无界增长）
        keep = sorted(bd)[-(self.BAN_DAYS + 1):]
        cur["by_day"] = {k: bd[k] for k in keep}
        self.data[key] = cur
        _atomic_json(self.path, self.data)

    def reset(self, date: str | None = None) -> int:
        """清掉某个数据日期的配额记录（`None` = 清空全部）。返回清掉的键数。"""
        if date is None:
            n = len(self.data)
            self.data = {}
        else:
            d = str(date)[:10] + "|"
            n = sum(1 for k in list(self.data) if k.startswith(d))
            self.data = {k: v for k, v in self.data.items() if not k.startswith(d)}
        _atomic_json(self.path, self.data)
        return n
