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
        self.root = Path(root).resolve()
        # ★ 2026-09-25：输入身份缓存回归项目内 `state/input_hashes/`。
        #   原来写 `self.root.parent.parent / "CodeX" / ...` —— 那是「上游目录的祖父」，
        #   既与项目位置隐式耦合、又指向本项目单元之外。改为显式接收项目根，
        #   默认取本模块自己的 ROOT（测试用临时上游时不传即落到生产 state，
        #   所以调用方应传 cfg.root）。
        if project_root is None:
            from .config import ROOT as project_root          # 延迟导入，避免循环
        self.project_root = Path(project_root).resolve()
        self._cache: dict[tuple, pd.DataFrame] = {}
        self.audit_cutoff: int | None = None

    # ---------------------------------------------------------------- 路径
    def _dir(self, name: str) -> Path:
        import re
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(name)):
            raise ValueError(f"非法上游数据集名称：{name!r}")
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"上游数据集越出原始数据目录：{path}")
        return path

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
    def _audit_limit(self) -> int | None:
        """Audit visibility is a calendar date; normal production has no cap."""
        cutoff = self.audit_cutoff
        if cutoff is None:
            return None
        import datetime
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or len(str(cutoff)) != 8:
            raise ValueError("audit_cutoff 必须是 YYYYMMDD 整数日期")
        try:
            datetime.datetime.strptime(str(cutoff), "%Y%m%d")
        except ValueError as error:
            raise ValueError("audit_cutoff 必须是有效的 YYYYMMDD 日期") from error
        return cutoff

    @staticmethod
    def _audit_dates(values: pd.Series):
        """Parse real calendar dates, including the date part of minute bars."""
        text = values.astype("string").str.slice(0, 10)
        parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
        compact = text.str.fullmatch(r"\d{8}", na=False)
        if compact.any():
            parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
        return (parsed.dt.year * 10000 + parsed.dt.month * 100 + parsed.dt.day).fillna(0).to_numpy(dtype="int32")

    @staticmethod
    def _audit_columns(name: str, names: list[str]) -> tuple[str, list[str]]:
        # These fixed contracts define the calendar and stock axis, not a
        # historically captured information snapshot.
        if name in {"basic_calendar", "stock_list"}:
            return "contract", []
        announcements = [column for column in ("ann_date", "f_ann_date") if column in names]
        if name in {"stock_income", "stock_balancesheet", "stock_cashflow", "stock_financial_indicator"}:
            if announcements and "end_date" in names:
                return "financial", [*announcements, "end_date"]
        elif announcements:
            # Forecast end_date may be in the future: a forecast is visible at
            # this announcement, not at first_ann_date or its predicted period.
            return "announcement", announcements
        else:
            for column in ("trade_time", "trade_date", "suspend_date", "date"):
                if column in names:
                    return "dated", [column]
        raise ValueError(f"{name} 缺少可用于审计截断的有效日期列；不能将报告期或当前快照当作历史可得日期")

    @classmethod
    def _audit_frame(cls, frame: pd.DataFrame, mode: str, dates: list[str], cutoff: int) -> pd.DataFrame:
        if mode == "contract":
            return frame
        values = [cls._audit_dates(frame[column]) for column in dates]
        import numpy as np
        effective = np.maximum.reduce(values)
        if mode in {"financial", "announcement"}:
            announcements = values[:-1] if mode == "financial" else values
            known = np.logical_or.reduce([value >= 19900101 for value in announcements])
        else:
            known = effective > 0
        return frame.loc[known & (effective <= cutoff)].copy()

    def read(self, name: str, columns: list[str] | None = None,
             years: tuple[int, int] | None = None, use_cache: bool = True) -> pd.DataFrame:
        """读取一个数据集。`years=(y0, y1)` 闭区间裁剪年份分区。"""
        key = (name, tuple(columns) if columns else None, years)
        cutoff = self._audit_limit()
        if cutoff is not None:
            key = (name, tuple(columns) if columns is not None else None, years, cutoff)
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
                # Missing columns are an explicit schema error; never silently
                # expand a narrow financial read into an entire wide table.
                if cutoff is None:
                    parts.append(pd.read_parquet(f, columns=columns))
                else:
                    mode, dates = self._audit_columns(name, pq.ParquetFile(f).schema_arrow.names)
                    needed = None if columns is None else list(dict.fromkeys([*columns, *dates]))
                    frame = pd.read_parquet(f, columns=needed)
                    frame = self._audit_frame(frame, mode, dates, cutoff)
                    parts.append(frame if columns is None else frame[columns])
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
        if not cache_root.resolve().is_relative_to(self.project_root):
            raise ValueError(f"输入身份缓存越出因子项目：{cache_root}")
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
        # All content identities above are derived directly from raw Parquet.
        # No external updater ledger is consulted. A file revision is reported
        # conservatively by the engine and bounded by its historical policy.
        date_hashes, fresh = {}, False
        return {"exists": True, "rows": int(rows), "max_pit": int(mx), "files": parts,
                "date_hashes": date_hashes, "ledger_fresh": fresh}
