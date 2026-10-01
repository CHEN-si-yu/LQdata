"""上游数据读取（模块① 的产出）。

设计要点：
  - **列裁剪**：`stock_financial_indicator` 有 190 列 / 333 MB，全读进来是纯浪费。
    所有读取都要显式传 `columns`。
  - **分区裁剪**：财报按 `end_date` 分年，但 `ann_date` 可以滞后一年多
    （实测最晚：2024-12-31 的报告 2026-03-20 才公告），所以瞄准输出窗口
    [a, b] 时要读 `end_date` 年份 **[year(a)-3, year(b)]**，不能只读当年。
  - **快照 vs 分区**：`stock_list` / `basic_calendar` 是单文件平铺，
    其余是 `year=YYYY/data.parquet`，两种都要支持。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .dates import year_of


class Upstream:
    def __init__(self, root: Path, project_root: Path | None = None):
        self.root = Path(root)
        # ★ 2026-09-25：输入身份缓存回归项目内 `state/input_hashes/`。
        #   原来写 `self.root.parent.parent / "CodeX" / ...` —— 那是「上游目录的祖父」，
        #   既与项目位置隐式耦合、又指向本项目单元之外。改为显式接收项目根，
        #   默认取本模块自己的 ROOT（测试用临时上游时不传即落到生产 state，
        #   所以调用方应传 cfg.root）。
        if project_root is None:
            from .config import ROOT as project_root          # 延迟导入，避免循环
        self.project_root = Path(project_root)
        self._cache: dict[tuple, pd.DataFrame] = {}

    # ---------------------------------------------------------------- 路径
    def _dir(self, name: str) -> Path:
        return self.root / name

    def clear_cache(self) -> None:
        """丢掉缓存的上游表。

        ★ 并行前必须调用：`fork` 出来的 worker 是写时复制的，但 Python 的引用计数
        会在**读取**对象时改头部字段，从而触发整页复制。上游财报缓存有好几百 MB
        （6 年 × 3 张表），4 个 worker 各复制一遍，光 COW 开销就比省下的计算还贵。
        衍生层建好之后这些原始表就不再需要了，清掉即可。
        """
        self._cache.clear()

    @property
    def calendar_path(self) -> Path:
        return self._dir("basic_calendar") / "data.parquet"

    def exists(self, name: str) -> bool:
        d = self._dir(name)
        return d.is_dir() and any(d.rglob("*.parquet"))

    def years(self, name: str) -> list[int]:
        d = self._dir(name)
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

    # ---------------------------------------------------------------- 读取
    def read(self, name: str, columns: list[str] | None = None,
             years: tuple[int, int] | None = None, use_cache: bool = True) -> pd.DataFrame:
        """读取一个数据集。`years=(y0, y1)` 闭区间裁剪年份分区。"""
        key = (name, tuple(columns) if columns else None, years)
        if use_cache and key in self._cache:
            return self._cache[key]

        d = self._dir(name)
        files: list[Path] = []
        if d.is_dir():
            flat = d / "data.parquet"
            if flat.exists():
                files.append(flat)
            for y in (self.years(name) if years is None else
                      [y for y in self.years(name) if years[0] <= y <= years[1]]):
                p = d / f"year={y}" / "data.parquet"
                if p.exists():
                    files.append(p)
        if not files:
            df = pd.DataFrame()
        else:
            parts = []
            for f in files:
                try:
                    parts.append(pd.read_parquet(f, columns=columns))
                except Exception:                       # 列不存在等情况：整读再裁
                    t = pd.read_parquet(f)
                    parts.append(t[columns] if columns else t)
            df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

        if use_cache:
            self._cache[key] = df
        return df

    def years_for_window(self, start: int, end: int, lag_years: int = 3) -> tuple[int, int]:
        """输出窗口 [start, end] 需要读的分区年份。

        `lag_years=3` 的依据：财报按 end_date 分区，ann_date 最晚滞后 15 个月
        （实测），再加 TTM 需要往前 4 个季度 → 最老的 end_date 年份是 start 年减 2~3。
        """
        return (year_of(start) - lag_years, year_of(end))

    # ---------------------------------------------------------------- 水位
    def watermark(self, name: str, pit_col: str | None = None) -> dict:
        """输入水位：行数、最新日期和年度文件身份（小型快照另含 MD5）。用于「上游变了就重算」的失效判定。

        为什么不能只看输出日期缺口：上游财报被追溯修正时，它的 `ann_date`
        可能远在过去（实测最长 15 个月），只按日期找缺口会让修正永久
        传播不到历史，而且悄无声息。

        ★ **元数据快路径**（2026-09-14 加，实测 10× 提速）：
          原实现逐分区 `read_parquet(columns=[col])`，实测 `stock_daily` 2.39 s、
          `stock_adj_factor` 2.34 s、`stock_finance` 2.14 s —— 20 个依赖就是每次
          run 白花 ~45 s 的固定成本（250 个因子时会直接拖垮日增量）。
          改成「`num_rows` 只读页脚求和 + 只读**最后一个非空分区**的日期列」，
          实测 0.23 s，**语义完全不变**：

            · 行数：parquet 页脚里就有 `num_rows`，不用解压数据页；
            · 最新日期：这些数据集是**按年追加**的分区，最大值只可能在最后一个
              非空分区里；旧年份数值修订由各文件 size/mtime 身份识别，不能只依赖总行数。
        """
        d = self._dir(name)
        if not d.is_dir():
            return {"exists": False, "rows": 0, "max_pit": 0}
        files = sorted(d.glob("year=*/data.parquet")) + sorted(d.glob("data.parquet"))
        if not files:
            return {"exists": False, "rows": 0, "max_pit": 0}

        import hashlib, json
        col = pit_col or "trade_date"
        rows, mx, parts = 0, 0, {}
        cache_root = self.project_root / "state" / "input_hashes"
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = cache_root / (name + ".json")
        try: cached = json.loads(cache_file.read_text())
        except (OSError, ValueError): cached = {}
        changed_cache = False
        newest_mtime = 0
        for f in files:
            stat = f.stat()
            newest_mtime = max(newest_mtime, stat.st_mtime_ns)
            key = f.parent.name if f.parent.name.startswith("year=") else "snapshot"
            identity = [stat.st_size, stat.st_mtime_ns]
            entry = cached.get(key, {})
            # 内容哈希只在文件身份变化时重算；相同字节的重写不使因子失效。
            if entry.get("identity") != identity:
                h = hashlib.md5()
                with f.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                        h.update(chunk)
                entry = {"identity": identity, "md5": h.hexdigest()}
                cached[key] = entry
                changed_cache = True
            parts[key] = identity + [entry["md5"]]
            pf = pq.ParquetFile(f)
            rows += pf.metadata.num_rows
            names = pf.schema_arrow.names
            if col not in names:
                continue  # 无日期快照水位为0，不能猜第一列是日期。
            ci = names.index(col)
            stats = [pf.metadata.row_group(i).column(ci).statistics for i in range(pf.num_row_groups)]
            values = [v.max for v in stats if v is not None and v.has_min_max and v.max is not None]
            if len(values) != len(stats) and pf.metadata.num_rows:
                series = pd.read_parquet(f, columns=[col])[col].dropna()
                values = [series.max()] if len(series) else []
            for value in values:
                text = str(value)[:10]
                digits = text.replace("-", "")[:8]
                if len(digits) == 8 and digits.isdigit():
                    mx = max(mx, int(digits))
        if changed_cache:
            from .manifest import _atomic_json
            _atomic_json(cached, cache_file)
        # 直接读取上游发布的逐日台账，不 import 上游工程代码。
        ledger = self.root.parent.parent / "everyday_tasks/state/dayhash" / (name + ".parquet")
        date_hashes, fresh = {}, False
        if ledger.exists():
            frame = pd.read_parquet(ledger, columns=["date", "md5"])
            date_hashes = dict(zip(frame["date"].astype(str), frame["md5"].astype(str)))
            fresh = ledger.stat().st_mtime_ns >= newest_mtime
        return {"exists": True, "rows": int(rows), "max_pit": int(mx), "files": parts,
                "date_hashes": date_hashes, "ledger_fresh": fresh}
