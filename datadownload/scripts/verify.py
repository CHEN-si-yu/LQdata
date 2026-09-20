#!/usr/bin/env python3
"""灵启数据下载器 · 落地校验（独立脚本，不改动下载主流程）。

为什么需要单独一个校验入口：下载链路只能保证"请求成功 + 写入成功"，
下面三类偏差它自己发现不了，必须重新问一次服务端或回头核对本地：

  1. 行数不符   —— 分页丢页、服务端偶发空响应，本地比服务端少（或多）
  2. 覆盖缺口   —— manifest.coverage 声明"这个区间已覆盖"，本地却没有数据
  3. 可疑区间   —— 下载时"预期非空却返回空"被记进 manifest.suspect，需要复查

做法：从**本地已落地的数据**里随机抽 N 个日期区间 / 实体 / 变体，按 spec 里
声明的原始请求方式重新拉一遍，逐行比对。

比对口径：默认只比 **行数 + 主键集合**。逐字段比对成本高，而且服务端会追溯
修正历史值（财报、复权因子），逐值比对会产生大量无意义告警；行数对不上、
主键集合对不上，才是真正值得追的落地错误。

服务端硬限制：page * page_size ≤ 100000，超了直接拒。所以抽样一律用小窗口，
命中不到数据再放大、超上限再缩小（见 _probe_window）。

限速：官方上限 280 次/分钟，而下载进程可能同时在跑，本脚本目标 ≤120 次/分钟。
不另写节流逻辑，直接把配置里的速率改小后复用 LingqiClient 的令牌桶。

用法：
    python scripts/verify.py --dataset basic_calendar
    python scripts/verify.py --dataset stock_daily --samples 5
    python scripts/verify.py --all
    python scripts/verify.py --all --fix          # 把确认缺失的区间重拉并写回

退出码：0 = 全部通过，1 = 发现不一致（便于挂到 cron 上）。
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lingqi import spec as spec_mod                       # noqa: E402
from lingqi import store                                  # noqa: E402
from lingqi.client import (ApiError, LingqiClient,        # noqa: E402
                           RetryableError, extract_list, extract_total)
from lingqi.engine import CHUNK_DAYS, Engine, today_str   # noqa: E402
from lingqi.manifest import Manifest                      # noqa: E402

MAX_ROWS_PER_QUERY = 100_000      # 服务端硬限制：page * page_size 上限
VERIFY_RATE_PER_MIN = 120         # 自限速目标（服务端 280，另有下载进程在跑）
PAGE_SIZE_CAP = 10_000            # 服务端单页上限

BASE_WINDOW_DAYS = 3              # 抽样窗口起点：小窗口便宜，空结果再放大
MAX_ADJUST = 4                    # 窗口自适应次数上限
WINDOW_CAP_DAYS = 400             # 窗口放大上限，别让一次抽样变成全量重下
PROBE_FULL_MAX = 50_000           # 本地行数小于此值才做"整表重取"校验

FIX_RANGE_LIMIT = 20              # --fix 单次最多回写的缺口区间数
FIX_DATE_LIMIT = 50               # --fix 单次最多回写的缺失日期数
RUNNING_FRESH_SEC = 600           # status.json 多久内算"下载进程正在跑"

# 不属于"问题"的 finding 级别：note 是中性说明，suspect_empty 是复查后确认无数据
NON_PROBLEM = {"note", "suspect_empty"}


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _col_key(s: pd.Series) -> pd.Series:
    """日期/时间列归一成 YYYY-MM-DD 字符串，跨类型也能比较。"""
    return s.astype(str).str.slice(0, 10)


def _arrow_key(table, col: str, prefix: int | None = None):
    """取一列的（可选截断的）字符串形式：时间戳/日期型先 cast 再截断。

    直接用 arrow 做去重，避免把大表的整列读成 Python 对象（几百万行会很吃内存）。
    """
    arr = table.column(col)
    if not pa.types.is_string(arr.type):
        arr = arr.cast(pa.string())
    return pc.utf8_slice_codeunits(arr, 0, prefix) if prefix else arr


@dataclass
class LocalFilter:
    """本地取数条件（纯内存筛选，不碰 API）。

    eq      : 列名 → 允许取值的集合（实体代码在本地可能带后缀，故用集合）
    between : (列名, 起, 止)，按 YYYY-MM-DD 前缀做字符串比较
    years   : 只读这几年的分区，避免为了抽 3 天把十年数据全读进来
    """

    eq: dict = field(default_factory=dict)
    between: tuple | None = None
    years: tuple | None = None


@dataclass
class Sample:
    """一次抽样：服务端请求体 + 本地对应的筛选条件。"""

    label: str
    payload: dict
    flt: LocalFilter
    kind: str = "range"                 # range/date/stock/entity/table/variant
    a: str | None = None                # range 抽样的窗口
    b: str | None = None
    window_days: int = BASE_WINDOW_DAYS
    variants: list = field(default_factory=list)   # 需要逐变体请求并合并的变体


@dataclass
class SampleResult:
    label: str
    local_n: int = 0
    api_n: int = 0
    miss: int = 0          # 服务端有、本地没有 → 本地缺数据
    extra: int = 0         # 本地有、服务端没有 → 本地多/脏数据
    note: str = ""         # 中性说明（不构成差异）
    error: str = ""        # 没能校验成（请求失败/超上限）——这本身算差异

    @property
    def bad(self) -> bool:
        return bool(self.miss or self.extra or self.error)


@dataclass
class DatasetResult:
    name: str
    local_rows: int = 0
    manifest_rows: int = 0
    samples: list = field(default_factory=list)
    findings: list = field(default_factory=list)    # (级别, 文字, 计数, 明细)
    fixed: list = field(default_factory=list)
    skipped: str = ""
    error: str = ""
    seconds: float = 0.0
    requests: int = 0
    busy: bool = False         # 下载进程正在写这个数据集

    def _sum(self, level: str) -> int:
        return sum(int(f[2]) for f in self.findings
                   if f[0] == level and len(f) > 2)

    @property
    def miss_rows(self) -> int:
        return sum(s.miss for s in self.samples) + self._sum("rows_missing")

    @property
    def extra_rows(self) -> int:
        return sum(s.extra for s in self.samples)

    @property
    def problems(self) -> list:
        return [f for f in self.findings if f[0] not in NON_PROBLEM] + \
               [s for s in self.samples if s.bad]

    @property
    def verdict(self) -> str:
        if self.error:
            return f"请求失败: {self.error[:36]}"
        if self.skipped and not self.samples:
            return f"跳过（{self.skipped}）" if not self.problems else "本地无数据"
        parts = []
        if self.miss_rows:
            parts.append(f"缺 {self.miss_rows:,} 行")
        if self.extra_rows:
            parts.append(f"多 {self.extra_rows:,} 行")
        days = self._sum("gap_days")
        if days:
            parts.append(f"缺 {days:,} 个交易日")
        ents = self._sum("gap_entities")
        if ents:
            parts.append(f"缺 {ents:,} 个实体")
        failed = sum(1 for s in self.samples if s.error)
        if failed:
            parts.append(f"{failed} 个抽样未能校验")
        if any(f[0] == "suspect_empty" for f in self.findings):
            parts.append("可疑未复现")
        return " / ".join(parts) if parts else "OK"

    @property
    def ok(self) -> bool:
        return not self.problems and not self.error


class Verifier:
    def __init__(self, cfg: dict, root: Path, samples: int = 3, fix: bool = False,
                 seed: int | None = None, do_gaps: bool = True, verbose: bool = False):
        self.root = root
        self.data_root = root / cfg["paths"]["data"]
        self.state_root = root / cfg["paths"]["state"]
        self.log_root = root / cfg["paths"]["logs"]
        self.samples = max(1, samples)
        self.fix = fix
        self.verbose = verbose
        # 注意别叫 check_gaps：那是下面的方法名，实例属性会把它盖掉
        self.do_gaps = do_gaps
        self.rng = random.Random(seed)
        self.end = today_str()

        # ★ 限速：把配置速率压到 ≤120/分钟后再交给客户端，复用它已有的令牌桶。
        #   绝不另起第二个客户端 —— 两条限速通道等于把实际速率翻倍。
        vcfg = copy.deepcopy(cfg)
        vcfg["api"]["rate_limit_per_min"] = min(int(vcfg["api"]["rate_limit_per_min"]),
                                               VERIFY_RATE_PER_MIN)
        self.client = LingqiClient(vcfg, None)

        # 借用引擎的落库能力（_to_df 归一日期列、_save → store.upsert 写回），
        # 但把它的 client 换成上面那个，保证全局只有一条限速通道；
        # 同时关掉 status.json 写入，避免打扰正在跑的下载进程。
        self.engine = Engine(vcfg, root)
        self.engine.client = self.client
        self.engine.board.status.enabled = False

        self._cal: list | None = None
        self._uniq_cache: dict = {}

    # ================================================================ 本地读
    def _partition_paths(self, spec) -> list:
        """本地该数据集的所有 parquet 文件（单文件 or 按年分区）。"""
        d = self.data_root / spec.name
        if not d.exists():
            return []
        files = [store.year_partition_path(self.data_root, spec.name, y)
                 for y in store.list_partitions(self.data_root, spec.name)]
        flat = store.flat_path(self.data_root, spec.name)
        if flat.exists():
            files.append(flat)
        return [p for p in files if p.exists()]

    def _local_row_count(self, spec) -> int:
        """用 parquet 元数据数行数，不读数据 —— 大表也几乎零成本。"""
        n = 0
        for p in self._partition_paths(spec):
            try:
                n += pq.ParquetFile(p).metadata.num_rows
            except Exception:      # noqa: BLE001 - 元数据读不了就当 0，不影响结论
                pass
        return n

    def _uniq_values(self, spec, col: str, prefix: int | None = None) -> list:
        """单列去重值（只读这一列）。"""
        key = (spec.name, col, prefix)
        if key in self._uniq_cache:
            return self._uniq_cache[key]
        out: set = set()
        for p in self._partition_paths(spec):
            try:
                t = pq.read_table(p, columns=[col])
            except Exception:      # noqa: BLE001 - 缺列/坏文件直接跳过
                continue
            out |= set(pc.unique(_arrow_key(t, col, prefix)).to_pylist())
        vals = sorted(str(x) for x in out if x is not None and str(x))
        self._uniq_cache[key] = vals
        return vals

    def _distinct_dates(self, spec) -> list:
        """本地出现过的日期（升序）—— 抽样锚点与缺口比对都靠它。"""
        return self._uniq_values(spec, spec.date_field, prefix=10)

    def _local_slice(self, spec, flt: LocalFilter) -> pd.DataFrame:
        """按抽样条件取本地切片：只读需要的列、只读相关的年分区。"""
        cols = {spec.date_field, spec.code_param, *spec.keys}
        for v in (spec.variants or []):
            cols |= set(v)
        paths = self._partition_paths(spec)
        if flt.years and spec.partition != "none":
            want = {f"year={y}" for y in range(flt.years[0], flt.years[1] + 1)}
            paths = [p for p in paths if p.parent.name in want]
        frames = []
        for p in paths:
            try:
                have = set(pq.ParquetFile(p).schema_arrow.names)
            except Exception:      # noqa: BLE001
                continue
            use = [c for c in cols if c in have]
            if use:
                frames.append(pd.read_parquet(p, columns=use))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)

        mask = pd.Series(True, index=df.index)
        if flt.between:
            col, a, b = flt.between
            if col not in df.columns:
                return df.iloc[0:0]
            k = _col_key(df[col])
            mask &= (k >= a) & (k <= b)
        for col, vals in flt.eq.items():
            if col not in df.columns:
                # 接口不回该列时本地也没有，只能放弃这个条件（例：per_entity 的实体列）
                continue
            mask &= df[col].astype(str).isin([str(v) for v in vals])
        return df[mask]

    def trading_days(self, lo: str, hi: str) -> list:
        """交易日历（取自本地 basic_calendar）。没有它，周末会被算成缺口。"""
        if self._cal is None:
            p = self.data_root / "basic_calendar" / "data.parquet"
            if not p.exists():
                self._cal = []
            else:
                df = pd.read_parquet(p, columns=["date", "is_open"])
                self._cal = sorted(_col_key(df.loc[df["is_open"] == 1, "date"]).tolist())
        return [x for x in self._cal if lo <= x <= hi]

    # ================================================================ 请求
    def _fetch_pages(self, spec, payload: dict, page_size: int | None = None):
        """按页拉完一个请求体，返回 (rows, total, truncated)。

        truncated=True 表示服务端 total 超过单次查询上限（page*page_size≤100000）：
        此时**不继续翻页**，硬翻必被拒，必须由调用方缩小区间重来。
        """
        ps = min(int(page_size or spec.page_size or PAGE_SIZE_CAP), PAGE_SIZE_CAP)
        rows: list = []
        total = None
        page = 0
        while page * ps <= MAX_ROWS_PER_QUERY:      # 保证每个请求都满足服务端限制
            body = {**payload, "page": page, "page_size": ps}
            # expect_rows 跟随 spec：下载端认定"不该为空"的数据集这里也让它自动重试，
            # 免得服务端偶发空响应被误判成"本地缺数据"。
            data = self.client.call(spec.path, body, method=spec.method,
                                    expect_rows=spec.expect_rows and not rows)
            batch = extract_list(data)
            if total is None:
                total = extract_total(data)
                if total is not None and total > MAX_ROWS_PER_QUERY:
                    return [], total, True
            rows.extend(batch)
            if not batch or (total is not None and len(rows) >= total):
                break
            page += 1
        return rows, (total if total is not None else len(rows)), False

    def _fetch_variants(self, spec, smp: Sample):
        """按变体逐次请求并合并。变体参数要落到行上，才能和本地带该列的数据对齐
        （下载端 engine.run_* 就是这么做的）。"""
        if not smp.variants:
            return self._fetch_pages(spec, smp.payload)
        rows: list = []
        total = 0
        for v in smp.variants:
            vrows, vtotal, trunc = self._fetch_pages(spec, {**smp.payload, **v})
            if trunc:
                return [], 0, True
            for r in vrows:
                if isinstance(r, dict):
                    r.update(v)
            rows.extend(vrows)
            total += int(vtotal or 0)
        return rows, total, False

    def _probe_window(self, spec, a: str, b: str, w: int):
        """拉 [a,b]；空则放大窗口，超上限则缩小窗口。返回 (rows,total,a,b,truncated)。"""
        cap = min(_d(b) + dt.timedelta(days=WINDOW_CAP_DAYS), _d(self.end))
        if cap < _d(a):
            cap = _d(b)
        for _ in range(MAX_ADJUST + 1):
            payload = {**spec.params, spec.start_param: a, spec.end_param: b}
            rows, total, trunc = self._fetch_pages(spec, payload)
            if trunc:
                w = max(1, w // 4)
                b = min(_d(a) + dt.timedelta(days=w - 1), cap).isoformat()
                continue
            if rows:
                return rows, total, a, b, False
            # 空结果：可能是服务端偶发空，也可能窗口落在无数据区，放大再看一次
            w = min(w * 4, WINDOW_CAP_DAYS)
            nb = min(_d(a) + dt.timedelta(days=w - 1), cap).isoformat()
            if nb == b:
                return [], total, a, b, False
            b = nb
        return [], 0, a, b, True

    # ================================================================ 抽样构造
    def _pick(self, pool: list, n: int) -> list:
        return self.rng.sample(pool, min(n, len(pool))) if pool else []

    def _samples_snapshot(self, spec, man, res) -> list:
        """快照类：整表就是一次请求，直接整表重取比对最彻底。"""
        out = []
        variants = spec.variants or []
        if variants:
            # 有变体的快照（tdx_blocks 的 4 种 block_type）：按变体抽样比对，
            # 变体列本来就在本地数据里，正好逐变体核对
            for v in self._pick(variants, self.samples):
                payload = {**spec.params, **v}
                if spec.range_params:
                    payload[spec.start_param] = spec.start
                    payload[spec.end_param] = self.end
                out.append(Sample(label=str(v), payload=payload, kind="variant",
                                  variants=[v],
                                  flt=LocalFilter(eq={k: {v[k]} for k in v})))
            return out
        if res.local_rows > PROBE_FULL_MAX:
            res.findings.append(("note", f"本地 {res.local_rows:,} 行，跳过整表重取"))
            return out
        payload = dict(spec.params)
        flt = LocalFilter()
        if spec.range_params:
            payload[spec.start_param] = spec.start
            payload[spec.end_param] = self.end
            flt = LocalFilter(between=(spec.date_field, spec.start, self.end))
        out.append(Sample(label="整表", payload=payload, kind="table", flt=flt))
        return out

    def _samples_range(self, spec, man, res) -> list:
        """区间类：以本地出现过的日期为锚点，随机取 N 个小窗口重拉。

        窗口起点：日线类 3 天足够（单日几千行），事件类（end_date/ann_date，
        隔几个月才有一条）起点就得宽一些，否则抽到的窗口必然为空。
        """
        dates = self._distinct_dates(spec)
        if not dates:
            res.findings.append(("problem", "本地无任何日期数据（coverage 却声明已覆盖）"))
            return []
        dense = spec.date_field == "trade_date"
        w0 = BASE_WINDOW_DAYS if dense else min(
            max(spec.chunk_days or 0, CHUNK_DAYS.get(spec.tier, 30)), 90)
        out = []
        for anchor in self._pick(dates, self.samples):
            a = anchor
            b = min(_d(a) + dt.timedelta(days=w0 - 1), _d(self.end)).isoformat()
            out.append(Sample(label=f"{a}~{b}", payload=dict(spec.params), kind="range",
                              a=a, b=b, window_days=w0,
                              flt=LocalFilter(between=(spec.date_field, a, b),
                                              years=(_d(a).year, _d(b).year))))
        return out

    def _samples_per_date(self, spec, man, res) -> list:
        """逐日类：从已标记 done 的日期里抽样（没 done 的属于缺口，另有检查）。"""
        pool = sorted(man.done.get("dates", {})) or self._distinct_dates(spec)
        if not pool:
            res.findings.append(("problem", "本地无任何日期数据"))
            return []
        out = []
        for day in self._pick(pool, self.samples):
            out.append(Sample(label=day, payload={**spec.params, spec.date_param: day},
                              kind="date",
                              flt=LocalFilter(eq={spec.date_field: {day}},
                                              years=(_d(day).year, _d(day).year)),
                              variants=list(spec.variants or [])))
        return out

    def _codes_with_local_data(self, spec, pool: list) -> list:
        """本地确实已有数据的代码。

        只用来给"边下边校验"的场景收窄抽样池：下载进程正在跑、或 done 标记还没
        落盘时，抽到"还没轮到的实体"会被误报成缺口，收窄后就不会。
        """
        local = {str(x).split(".")[0] for x in self._uniq_values(spec, spec.code_param)}
        if not local:
            return []
        return [c for c in pool if str(c).split(".")[0] in local]

    def _samples_per_stock(self, spec, man, res) -> list:
        """逐股类：从已完成的股票里抽样，比对这只股票的全历史主键集合。"""
        done = sorted(man.done.get("stocks", {}))
        pool = done
        if not done or res.busy:
            pool = self._codes_with_local_data(spec, done or self.engine.stock_codes())
        if not pool:
            res.findings.append(("note", "done.stocks 为空（尚未跑过或刚起步），跳过抽样"))
            return []
        out = []
        for code in self._pick(pool, self.samples):
            out.append(Sample(label=code, payload={**spec.params, spec.code_param: code},
                              kind="stock", flt=LocalFilter(eq={spec.code_param: {code}})))
        return out

    def _samples_per_entity(self, spec, man, res) -> list:
        """逐实体类：从已完成的实体里抽样；区间型的实体只取其中一年，控制单次行数。"""
        done = sorted(man.done.get("entities", {}))
        pool = done
        if not done or res.busy:
            pool = self._codes_with_local_data(spec, done or self.engine._entity_codes(spec))
            if not pool and not done:
                res.findings.append(("note", "实体清单为空，跳过抽样"))
                return []
            if not pool:
                pool = done

        out = []
        for code in self._pick(pool, self.samples):
            vals = self._match_entity(spec, code)
            a = b = None
            if spec.entity_range:
                pick = self._pick_years_for(spec, code, vals)
                if pick is None:
                    res.findings.append(("note", f"{code} 本地无年份数据，跳过"))
                    continue
                a, b = f"{pick}-01-01", f"{pick}-12-31"
                a = max(a, spec.start)
                b = min(b, self.end)
            payload = {**spec.params, spec.code_param: code}
            if a:
                payload[spec.start_param] = a
                payload[spec.end_param] = b
            out.append(Sample(label=code + (f" {a[:7]}" if a else ""), payload=payload,
                              kind="entity",
                              flt=LocalFilter(eq={spec.code_param: vals},
                                              years=(int(a[:4]), int(a[:4])) if a else None)))
        return out

    def _pick_years_for(self, spec, code: str, vals: set) -> int | None:
        """挑一个"本地确实有该实体数据"的年份，保证抽样有信息量。

        随机顺序试，命中就返回；一个都没有时说明该实体本地为空（可能是接口本身
        就没数据），退化成比较最近一年。
        """
        years = sorted({int(y[:4]) for y in self._distinct_dates(spec) if y[:4].isdigit()},
                       reverse=True)
        for y in self._pick(years, len(years)):
            flt = LocalFilter(eq={spec.code_param: vals}, years=(y, y))
            if len(self._local_slice(spec, flt)):
                return y
        return years[0] if years else None

    def _match_entity(self, spec, code: str) -> set:
        """实体代码在本地可能带后缀（请求 880201 → 落地 880201.TDX），两种都算命中。"""
        if spec.code_param not in (spec.keys or ()) and \
                spec.code_param not in self._local_columns(spec):
            return {code}
        local = set(self._uniq_values(spec, spec.code_param))
        if code in local:
            return {code}
        base = code.split(".")[0]
        return {code} | {x for x in local if x.split(".")[0] == base}

    def _local_columns(self, spec) -> set:
        cols: set = set()
        for p in self._partition_paths(spec)[:1]:
            try:
                cols = set(pq.ParquetFile(p).schema_arrow.names)
            except Exception:      # noqa: BLE001
                pass
        return cols

    def build_samples(self, spec, man, res) -> list:
        mode = spec.mode
        if mode == "snapshot":
            return self._samples_snapshot(spec, man, res)
        if mode == "range":
            return self._samples_range(spec, man, res)
        if mode == "per_date":
            return self._samples_per_date(spec, man, res)
        if mode == "per_stock":
            return self._samples_per_stock(spec, man, res)
        if mode == "per_entity":
            return self._samples_per_entity(spec, man, res)
        res.skipped = f"不支持的 mode={mode}"
        return []

    # ================================================================ 抽样执行
    def run_sample(self, spec, smp: Sample, res: DatasetResult) -> SampleResult:
        out = SampleResult(label=smp.label)
        try:
            if smp.kind == "range":
                a, b = smp.a, smp.b
                rows, total, a, b, trunc = self._probe_window(spec, a, b, smp.window_days)
                if trunc:
                    out.error = "窗口行数超单次查询上限，未能校验"
                    return out
                out.label = f"{a}~{b}"
                smp.flt = LocalFilter(between=(spec.date_field, a, b),
                                      years=(_d(a).year, _d(b).year))
            else:
                rows, total, trunc = self._fetch_variants(spec, smp)
                if trunc:
                    out.error = f"服务端 total={total:,} 超单次查询上限，未能校验"
                    return out
                if smp.kind == "date":
                    # 逐日接口的日期列可能不由服务端返回，下载端会补上；这里也补，
                    # 否则主键列对不齐，就只能退化成比对行数了
                    day = smp.payload.get(spec.date_param)
                    for r in rows:
                        if isinstance(r, dict):
                            r.setdefault(spec.date_field, day)
        except (ApiError, RetryableError) as exc:
            out.error = f"请求失败: {str(exc)[:80]}"
            return out

        local = self._local_slice(spec, smp.flt)
        out.local_n = len(local)
        out.api_n = len(rows)
        # 主键交集：接口不回某列时（如 per_entity 的实体列）退化为比对剩余主键
        keys = [k for k in spec.keys if k in local.columns and rows and k in rows[0]]
        if not keys:
            if out.api_n != out.local_n:
                diff = out.api_n - out.local_n
                out.miss = max(0, diff)
                out.extra = max(0, -diff)
            out.note = "仅比对行数（无共同主键列）"
            return out
        lk = {tuple(str(r[k]) for k in keys) for r in local.to_dict("records")}
        ak = {tuple(str(r.get(k)) for k in keys) for r in rows}
        out.miss = len(ak - lk)
        out.extra = len(lk - ak)
        if not out.miss and not out.extra and out.api_n != out.local_n:
            # 主键集合一致但行数不同：服务端同一主键有多条（如股东人数同日多次披露），
            # 落库时按主键去重过，本地是对的，只是行数天然比服务端少。
            out.note = (f"行数差 {abs(out.api_n - out.local_n):,}"
                        f"（主键集合一致，服务端含重复主键行）")
        return out

    # ================================================================ 本地缺口
    def check_manifest_drift(self, spec, man, res) -> None:
        if res.manifest_rows and res.local_rows != res.manifest_rows:
            res.findings.append(
                ("note", f"manifest 记 {res.manifest_rows:,} 行，实际 {res.local_rows:,} 行"
                         f"（下载进程可能正在写）", 0))

    def check_gaps(self, spec, man, res) -> None:
        """覆盖缺口：manifest 说覆盖了，本地却没有对应数据。纯本地检查，不发请求。

        各模式"完整"的定义不同：
          snapshot   文件存在且非空
          per_date   done.dates 覆盖到 spec.start 以来的每个交易日
          per_stock  股票清单里的每只都在 done.stocks
          per_entity 实体清单里的每个都在 done.entities
          range      coverage 内的年份分区必须存在；日线类再逐日核对
        """
        mode = spec.mode
        if mode in ("dump", "snapshot"):
            if mode == "snapshot" and res.local_rows == 0:
                res.findings.append(("problem", "本地无数据", 0))
            return

        if mode == "per_date":
            done = sorted(man.done.get("dates", {}))
            if not done:
                return
            have = set(done)
            missing = [d for d in self.trading_days(spec.start, done[-1]) if d not in have]
            if missing:
                res.findings.append(
                    ("gap_days", f"done.dates 缺 {len(missing)} 个交易日"
                                 f"（{missing[0]} ~ {missing[-1]}）",
                     len(missing), missing))
            return

        if mode == "per_stock":
            done = set(man.done.get("stocks", {}))
            if not done:
                return
            try:
                codes = set(self.engine.stock_codes())
            except SystemExit:
                return
            missing = sorted(codes - done)
            if missing:
                res.findings.append(
                    ("gap_entities", f"缺 {len(missing)} 只股票的下载记录"
                                     f"（如 {missing[0]}）", len(missing), missing))
            return

        if mode == "per_entity":
            done = set(man.done.get("entities", {}))
            if not done:
                return
            codes = set(self.engine._entity_codes(spec))
            # 实体清单里的代码与落地的代码可能差一个后缀，两边都归一后比较
            norm = lambda c: str(c).split(".")[0]                      # noqa: E731
            missing = sorted(c for c in codes if norm(c) not in {norm(x) for x in done})
            if missing:
                res.findings.append(
                    ("gap_entities", f"缺 {len(missing)} 个实体的下载记录"
                                     f"（如 {missing[0]}）", len(missing), missing))
            return

        # ---- range ----
        spans = [(s, e) for s, e in man.coverage]
        if not spans:
            if res.local_rows == 0 and man.last_run:
                res.findings.append(("problem", "coverage 为空且本地无数据", 0))
            return
        lo = max(min(s for s, _ in spans), man.data_start or spec.start)
        hi = min(max(e for _, e in spans), self.end)
        if spec.partition != "none":
            have_years = {int(p.parent.name.split("=")[1]) for p in self._partition_paths(spec)
                          if p.parent.name.startswith("year=")}
            gone = [y for y in range(int(lo[:4]), int(hi[:4]) + 1) if y not in have_years]
            if gone:
                res.findings.append(
                    ("gap_days", f"coverage 内的年份分区缺失: {gone[:5]}", 0, []))

        # 逐日核对只对"每个交易日都该有数据"的数据集做，否则必然误报：
        #   * expect_rows=False 的是稀疏事件（龙虎榜/停牌），空是正常的
        #   * 财报类 date_field 是 end_date/ann_date，本来就不对齐交易日
        #   * 周期 K 线（weekly/monthly）的 trade_date 只落在部分交易日
        if not spec.expect_rows or spec.date_field != "trade_date":
            return
        if any("period" in v for v in (spec.variants or [])):
            return
        # 末日不核对：coverage 到今天，但当天数据服务端可能还没出，会误报
        gap_hi = min(hi, (dt.date.today() - dt.timedelta(days=1)).isoformat())
        if gap_hi < lo:
            return
        have = set(self._distinct_dates(spec))
        missing = [d for d in self.trading_days(lo, gap_hi)
                   if d >= (man.data_start or spec.start) and d not in have]
        if missing:
            res.findings.append(
                ("gap_days", f"coverage 声明已覆盖，但本地缺 {len(missing)} 个交易日"
                             f"（{missing[0]} ~ {missing[-1]}）", len(missing), missing))

    # ================================================================ 可疑区间
    def check_suspect(self, spec, man, res) -> list:
        """man.suspect 里的区间全部重拉确认。返回待 --fix 处理的动作列表。"""
        pending = []
        if not man.suspect:
            return pending
        if spec.mode != "range":
            res.findings.append(("note", f"suspect 有 {len(man.suspect)} 项，"
                                        f"{spec.mode} 模式暂不复查", 0))
            return pending
        for seq, cnt in sorted(man.suspect.items()):
            if "~" not in seq:
                continue
            a, b = seq.split("~", 1)
            try:
                rows, total, trunc = self._fetch_pages(
                    spec, {**spec.params, spec.start_param: a, spec.end_param: b})
            except (ApiError, RetryableError) as exc:
                res.findings.append(("note", f"suspect {seq} 复查请求失败: {str(exc)[:60]}", 0))
                continue
            if trunc:
                res.findings.append(("note", f"suspect {seq} 超单次查询上限，未复查", 0))
                continue
            local = self._local_slice(
                spec, LocalFilter(between=(spec.date_field, a, b),
                                  years=(_d(a).year, _d(b).year)))
            if rows:
                # 当初"预期非空却返回空"被标可疑，现在拉到了 → 本地确实缺这一块
                res.findings.append((
                    "rows_missing",
                    f"suspect {seq} 复查拿到 {len(rows):,} 行、本地 {len(local):,} 行"
                    f"（被标记 {cnt} 次）", max(0, len(rows) - len(local))))
                pending.append(("write", seq, a, b, rows))
            else:
                res.findings.append((
                    "suspect_empty",
                    f"suspect {seq} 复查仍为空、本地 {len(local):,} 行"
                    f"（确认该区间无数据）", 0))
                pending.append(("clear", seq, a, b, None))
        return pending

    # ================================================================ 修复
    def is_being_written(self, name: str) -> bool:
        """下载进程是否正在写这个数据集（并发 upsert 会互相覆盖，必须避开）。"""
        p = self.state_root / "status.json"
        if not p.exists():
            return False
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return False
        if st.get("dataset") != name:
            return False
        try:
            ts = time.mktime(time.strptime(st["updated_at"], "%Y-%m-%d %H:%M:%S"))
        except (KeyError, ValueError):
            return False
        return (time.time() - ts) < RUNNING_FRESH_SEC

    def apply_fix(self, spec, man, res, pending) -> None:
        """把确认缺失的区间重拉并写回。

        写回复用引擎的 _to_df/_save（内部即 store.upsert + Manifest.mark_partition），
        日期列归一化、列顺序、分区规则与下载端完全一致，不会写出格式不一致的数据。
        """
        if self.is_being_written(spec.name):
            res.findings.append(("note", "下载进程正在写该数据集，已跳过 --fix", 0))
            return

        jobs = []          # (kind, label, a, b, rows)
        seen: set = set()  # 同一区间只拉一次（suspect 复查已经拉过的就不再拉）
        for kind, seq, a, b, rows in pending:
            if kind == "write":
                jobs.append(("range", f"suspect {seq}", a, b, rows))
                seen.add((a, b))
            elif kind == "clear":
                man.suspect.pop(seq, None)
                res.fixed.append(f"清除可疑标记 {seq}（复查确认无数据）")

        # 缺口补拉：range 按缺失交易日聚成连续区间，per_date 逐日补
        for f in list(res.findings):
            if f[0] != "gap_days" or len(f) <= 3 or not isinstance(f[3], list):
                continue
            days = [x for x in f[3] if len(x) == 10]
            if spec.mode == "per_date" and days:
                for d in days[:FIX_DATE_LIMIT]:
                    jobs.append(("date", d, d, d, None))
                if len(days) > FIX_DATE_LIMIT:
                    res.findings.append(
                        ("note", f"缺口 {len(days)} 天，本次只补前 {FIX_DATE_LIMIT} 天", 0))
            elif spec.mode == "range" and days:
                for a, b in self._group_days(days)[:FIX_RANGE_LIMIT]:
                    if (a, b) in seen:
                        continue
                    seen.add((a, b))
                    jobs.append(("range", f"{a}~{b}", a, b, None))

        if not jobs:
            if res.fixed:
                man.save()
            return

        for kind, label, a, b, rows in jobs:
            try:
                if kind == "date":
                    data = self.client.call(spec.path,
                                            {**spec.params, spec.date_param: a},
                                            method=spec.method, expect_rows=False)
                    rows = extract_list(data)
                    for r in rows:
                        if isinstance(r, dict):
                            r.setdefault(spec.date_field, a)
                    if not rows:
                        continue
                elif rows is None:
                    rows, _, trunc = self._fetch_pages(
                        spec, {**spec.params, spec.start_param: a, spec.end_param: b})
                    if trunc or not rows:
                        continue
                self._write_back(spec, man, rows, a, b)
                res.fixed.append(f"回写 {label}（{len(rows):,} 行）")
            except (ApiError, RetryableError) as exc:
                res.findings.append(("note", f"回写 {label} 失败: {str(exc)[:60]}", 0))

        for kind, seq, a, b, rows in pending:      # 复查过的可疑标记都该撤掉
            man.suspect.pop(seq, None)
        man.save()

    def _group_days(self, days: list) -> list:
        """把缺失交易日聚成连续区间（跨周末也合并，少发几次请求）。"""
        out = []
        cur_a = cur_b = None
        for d in sorted(days):
            cd = _d(d)
            if cur_a is None:
                cur_a = cur_b = cd
            elif (cd - cur_b).days <= 4:
                cur_b = cd
            else:
                out.append((cur_a.isoformat(), cur_b.isoformat()))
                cur_a = cur_b = cd
        if cur_a is not None:
            out.append((cur_a.isoformat(), cur_b.isoformat()))
        return out

    def _write_back(self, spec, man, rows: list, a: str, b: str) -> None:
        """按年拆分写回。"""
        df = self.engine._to_df(rows, spec, man)
        if len(df) == 0:
            return
        col = spec.date_field
        if spec.partition == "none" or col not in df.columns:
            year = int(a[:4]) if a[:4].isdigit() else 0
            self.engine._save(spec, man, df, year, spec.keys)
            man.add_coverage(a, b)
            return
        ys = {int(str(v)[:4]) for v in df[col] if v is not None and str(v)[:4].isdigit()}
        for y in sorted(ys):
            sub = df[df[col].astype(str).str.slice(0, 4) == str(y)]
            self.engine._save(spec, man, sub, y, spec.keys)
        man.add_coverage(a, b)

    # ================================================================ 主流程
    def check_dataset(self, spec) -> DatasetResult:
        t0 = time.time()
        req0 = self.client.stats.snapshot()["requests"]
        man = Manifest.load(self.state_root, spec.name)
        res = DatasetResult(name=spec.name, manifest_rows=man.partition_rows())
        res.local_rows = self._local_row_count(spec)

        if spec.mode == "dump":
            res.skipped = "dump 通道有配额限制，不做抽样请求"
            res.seconds = round(time.time() - t0, 1)
            return res
        if not self._partition_paths(spec):
            # 没落地数据就没得比；但"上次下载失败"值得顺手带出来，不然容易被忽略
            res.skipped = "本地无数据"
            if man.last_run.get("status") == "failed":
                res.skipped += f"；上次下载失败: {str(man.last_run.get('error'))[:50]}"
            res.seconds = round(time.time() - t0, 1)
            return res

        busy = res.busy = self.is_being_written(spec.name)
        self.check_manifest_drift(spec, man, res)
        if self.do_gaps:
            try:
                self.check_gaps(spec, man, res)
            except Exception as exc:      # noqa: BLE001 - 缺口扫描失败不该中断整体校验
                res.findings.append(("note", f"缺口扫描异常: {type(exc).__name__}: {exc}", 0))

        pending = []
        try:
            pending = self.check_suspect(spec, man, res)
        except Exception as exc:          # noqa: BLE001
            res.findings.append(("note", f"suspect 复查异常: {type(exc).__name__}: {exc}", 0))

        try:
            for smp in self.build_samples(spec, man, res):
                sr = self.run_sample(spec, smp, res)
                res.samples.append(sr)
                self._detail(f"    抽样 {sr.label}: 本地 {sr.local_n:,} / "
                             f"服务端 {sr.api_n:,}"
                             f"{f'  缺 {sr.miss:,} 行' if sr.miss else ''}"
                             f"{f'  多 {sr.extra:,} 行' if sr.extra else ''}"
                             f"{f'  [!{sr.error}]' if sr.error else ''}"
                             f"{f'  [{sr.note}]' if sr.note else ''}",
                             bad=sr.bad)
        except (ApiError, RetryableError) as exc:
            res.error = str(exc)

        # 下载进程正在写这个数据集时，"还没跑到"会被误判成缺口 → 降级为提示
        if busy:
            for i, f in enumerate(res.findings):
                if f[0] in ("gap_days", "gap_entities"):
                    res.findings[i] = ("note", f[1] + "（下载进程正在更新，可能只是还没跑到）",
                                       f[2], f[3])
            res.findings.append(("note", "下载进程正在写该数据集，抽样结果仅供参考", 0))

        if self.fix and not res.error:
            try:
                self.apply_fix(spec, man, res, pending)
            except Exception as exc:      # noqa: BLE001 - 修复失败不该丢掉已得的校验结论
                res.findings.append(("note", f"--fix 失败: {type(exc).__name__}: {exc}", 0))
            if res.fixed:
                # 上面的结论是修复前的，这里明确提示一下，免得看着"缺 N 行"以为没修
                print(f"    ↻ 已修复 {len(res.fixed)} 处，重跑一次校验可确认")
        res.seconds = round(time.time() - t0, 1)
        res.requests = self.client.stats.snapshot()["requests"] - req0
        return res

    # ================================================================ 输出
    def _detail(self, msg: str, bad: bool = True) -> None:
        """抽样明细：有问题必打，没问题只在 --verbose 时打。"""
        if bad or self.verbose:
            print(msg, flush=True)

    def report(self, results: list, log) -> int:
        print()
        print(f"{'数据集':<34}{'本地行数':>13}{'抽样':>6}  {'结论'}")
        print("-" * 100)
        bad = 0
        for r in results:
            if not r.ok:
                bad += 1
            mark = " *" if r.busy else ""
            print(f"{r.name:<34}{r.local_rows:>13,}{len(r.samples):>6}  "
                  f"{r.verdict}{mark}")
            if log is not None:
                for f in r.findings:
                    log.write(f"  [{f[0]}] {f[1]}\n")
                for s in r.samples:
                    flag = "差异" if s.bad else "一致"
                    log.write(f"  [sample/{flag}] {s.label}: 本地 {s.local_n:,} / "
                              f"服务端 {s.api_n:,} 缺 {s.miss} 多 {s.extra}"
                              f"{' ' + s.error if s.error else ''}"
                              f"{' ' + s.note if s.note else ''}\n")
                for x in r.fixed:
                    log.write(f"  [fixed] {x}\n")
                log.write(f"  [meta] 用时 {r.seconds}s 请求 {r.requests} 次\n")
        print("-" * 100)
        if any(r.busy for r in results):
            print("* 该数据集正被下载进程写入：未跑到的部分会被算成差异，结论仅供参考")
        st = self.client.stats.snapshot()
        good = sum(1 for r in results if r.ok)
        print(f"汇总：{len(results)} 个数据集 · {good} 通过 · {bad} 有差异 "
              f"| 请求 {st['requests']} 次 · 重试 {st['retries']} · "
              f"空响应重试 {st['empty_retries']} · 限速 {VERIFY_RATE_PER_MIN}/分钟")
        return 1 if bad else 0

    def run(self, names: list) -> int:
        for name in names:
            if name not in spec_mod.REGISTRY:
                print(f"✘ 未知数据集: {name}（用 python main.py list 查看可用名称）")
                return 2
        print(f"灵启数据校验 · 截止 {self.end} · 抽样 {self.samples} · "
              f"限速 {VERIFY_RATE_PER_MIN}/分钟 · 写回={'开' if self.fix else '关'}")
        print("=" * 100)
        results = []
        for name in names:
            spec = spec_mod.get(name)
            print(f"▶ {name}  ({spec.desc}) [{spec.mode}]")
            try:
                res = self.check_dataset(spec)
            except KeyboardInterrupt:
                print("\n收到中断，退出。")
                return 130
            results.append(res)

        need_log = any(r.problems for r in results) or any(
            f[0] == "suspect_empty" for r in results for f in r.findings)
        if self.verbose or need_log:
            self.log_root.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = self.log_root / f"verify_{stamp}.log"
            with open(path, "w", encoding="utf-8") as log:
                log.write(f"# 灵启数据校验 {time.strftime('%Y-%m-%d %H:%M:%S')} "
                          f"抽样={self.samples} fix={self.fix}\n")
                code = self.report(results, log)
            print(f"详情日志：{path}")
        else:
            code = self.report(results, None)
        return code


def load_cfg() -> dict:
    with open(ROOT / "conf" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="verify.py", description="灵启数据下载器 · 落地校验",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--dataset", "-d", action="append", default=[],
                   help="只校验指定数据集，可重复；不指定则校验所有本地已有数据的")
    p.add_argument("--all", action="store_true", help="校验所有数据集（含本地暂无数据的）")
    p.add_argument("--samples", "-n", type=int, default=3, help="每个数据集的抽样数（默认 3）")
    p.add_argument("--fix", action="store_true",
                   help="把确认缺失的区间重新拉取并写回（store.upsert + Manifest）")
    p.add_argument("--seed", type=int, default=None, help="随机种子（复现同一批抽样）")
    p.add_argument("--no-gaps", action="store_true", help="跳过覆盖缺口扫描（省本地 IO）")
    p.add_argument("--verbose", "-v", action="store_true", help="打印抽样明细并始终写日志")
    args = p.parse_args()

    cfg = load_cfg()
    v = Verifier(cfg, ROOT, samples=args.samples, fix=args.fix, seed=args.seed,
                 do_gaps=not args.no_gaps, verbose=args.verbose)

    names = list(dict.fromkeys(args.dataset))
    if not names:
        # 默认只校验"本地已有数据"的：没有落地数据就没有可比对象
        have = [s.name for s in spec_mod.all_specs() if v._partition_paths(s)]
        if args.all:
            rest = [s.name for s in spec_mod.all_specs() if s.name not in have]
            names = have + rest
        else:
            names = have
            print(f"未指定 --dataset，校验本地已有数据的 {len(names)} 个数据集")
    return v.run(names)


if __name__ == "__main__":
    raise SystemExit(main())
