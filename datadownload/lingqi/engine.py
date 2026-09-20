"""通用执行引擎 —— 把 Spec 变成实际的数据落地。

六种模式：
  snapshot   整表快照（可多 variant），覆盖写
  range      按日期区间拉全市场，按年分区，增量水位驱动
  per_date   一个交易日一次请求
  per_stock  遍历股票列表拉全历史
  per_entity 遍历实体（指数/板块）拉区间
  dump       daily_dump 特殊通道（最近 90 天 + 配额台账）

稳健性设计（针对实测到的"服务端偶发空响应"）：
  * 分页收齐后校验 行数 == 服务端 total，不足则整体重取
  * 区间抓取若返回空但区间内含交易日，记入 manifest.suspect 供事后复查
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path

import pandas as pd

from . import store
from .client import ApiError, LingqiClient, RetryableError, extract_list, extract_total
from .manifest import DumpQuota, Manifest
from .progress import Board, StatusWriter, fmt_rows
from .spec import Spec

log = logging.getLogger("lingqi.engine")

# 各 tier 默认的分块天数 —— 控制单次内存占用
CHUNK_DAYS = {"large": 7, "medium": 30, "small": 90, "slow": 1}

# 服务端硬限制：page * page_size 不能超过 100000。
# 即单次查询最多只能取回 10 万行，超过必须换遍历维度（缩小日期区间 / 按实体拆分）。
MAX_ROWS_PER_QUERY = 100_000

# 必须归一成 YYYY-MM-DD 的纯日期列（分钟级时间列不在此列）
PURE_DATE_COLS = {
    "trade_date", "end_date", "ann_date", "date", "list_date", "delist_date",
    "suspend_date", "report_date", "start_date", "finish_date",
}


def today_str() -> str:
    return dt.date.today().isoformat()


def year_of(d: str) -> int:
    return int(d[:4])


def d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def split_by_year(start: str, end: str) -> list[tuple[str, str]]:
    """把区间按自然年切开。"""
    out = []
    cur = d(start)
    last = d(end)
    while cur <= last:
        y_end = dt.date(cur.year, 12, 31)
        seg_end = min(y_end, last)
        out.append((cur.isoformat(), seg_end.isoformat()))
        cur = seg_end + dt.timedelta(days=1)
    return out


def _months_between(start: str, end: str) -> list[str]:
    """生成 [start, end] 覆盖到的所有月份，格式 YYYY-MM（升序）。"""
    out = []
    y, m = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


# 已完成实体每轮重访的**尾部月数**。
# 取 6 而不是 2，是为了盖过厂商的发布延迟：月频数据实测有 1~2 个月延迟
# （见 QUANT_PLATFORM.md 第 843 行的 index_weight 记录），留足缓冲才不会
# 在"上游刚补发、我们下个月才去看"的边界上漏掉。
TAIL_MONTHS = 6


def _tail_months(spec: Spec, end: str) -> list[str]:
    """月频表尾部重访用的月份列表；**非月频表返回空**。

    为什么只对 `date_step="month"` 开这个口子：日频/逐年表（stock_history_5min、
    tdx_minute、tdx_daily 这些，实体数 618~5901）若也做尾部重访，每轮会白烧
    上万次请求，而它们的日常增量归 `everyday_tasks` 管，不归这里。
    月频表实体只有 200 个、每实体每期 1 次请求，重访成本可控。
    """
    if spec.date_step != "month":
        return []
    y, m = int(end[:4]), int(end[5:7])
    m -= TAIL_MONTHS - 1
    while m < 1:                      # 跨年回退
        y, m = y - 1, m + 12
    return _months_between(f"{y:04d}-{m:02d}", end)


def split_chunks(start: str, end: str, days: int) -> list[tuple[str, str]]:
    """把区间按固定天数切块。"""
    if days <= 0:
        return [(start, end)]
    out = []
    cur = d(start)
    last = d(end)
    while cur <= last:
        seg_end = min(cur + dt.timedelta(days=days - 1), last)
        out.append((cur.isoformat(), seg_end.isoformat()))
        cur = seg_end + dt.timedelta(days=1)
    return out


class Engine:
    def __init__(self, cfg: dict, root: Path):
        self.cfg = cfg
        self.root = Path(root)
        self.data_root = self.root / cfg["paths"]["data"]
        self.state_root = self.root / cfg["paths"]["state"]
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.client = LingqiClient(cfg, log)
        self.board = Board(StatusWriter(self.state_root))
        self.dump_quota = DumpQuota(self.state_root)
        self._cal: list[str] | None = None
        self._stocks: list[str] | None = None
        self.history_start = cfg["download"]["history_start"]
        self.revision_days = int(cfg["download"].get("revision_days", 10))
        # 写入缓冲：见 _save 的说明
        self._buf: dict[tuple, list] = {}
        self._buf_ctx: dict[tuple, tuple] = {}
        self._flush_rows = int(cfg["download"].get("write_buffer_rows", 300_000))
        # per_entity 的周期性 checkpoint 间隔（任务数）。0 = 关闭。
        # 为什么需要：done.entities 原来只在整轮跑完后写一次，13 小时的任务中途
        # 被 kill 就丢全部进度（见 run_per_entity 里的说明）。
        self._ckpt_every = int(cfg["download"].get("per_entity_checkpoint_every", 5000))

    # ------------------------------------------------------------ 基础数据
    def calendar(self) -> list[str]:
        """交易日列表（升序）。日历数据集是其它数据集增量判断的基准。"""
        if self._cal is None:
            p = self.data_root / "basic_calendar" / "data.parquet"
            if not p.exists():
                raise SystemExit(
                    "缺少交易日历。请先运行:  python main.py run basic_calendar"
                )
            df = pd.read_parquet(p)
            self._cal = sorted(df.loc[df["is_open"] == 1, "date"].astype(str).tolist())
        return self._cal

    def trading_days_between(self, start: str, end: str) -> list[str]:
        cal = self.calendar()
        return [x for x in cal if start <= x <= end]

    def stock_codes(self) -> list[str]:
        if self._stocks is None:
            p = self.data_root / "stock_list" / "data.parquet"
            if not p.exists():
                raise SystemExit("缺少股票列表。请先运行:  python main.py run stock_list")
            self._stocks = sorted(pd.read_parquet(p)["stock_code"].astype(str).tolist())
        return self._stocks

    def _entity_codes(self, spec: Spec) -> list[str]:
        """per_entity 模式的实体列表来源（声明式：spec.entity_from）。

        优先只读最近一个年分区——实体清单是"当前有效集合"，
        没必要把十年历史全读进来（index_daily 全史有 1000 万行）。
        """
        if spec.entity_codes:
            return list(spec.entity_codes)
        src = spec.entity_from
        if src is None:
            return []
        ds, col = src
        years = store.list_partitions(self.data_root, ds)
        if years:
            p = store.year_partition_path(self.data_root, ds, years[-1])
        else:
            p = store.flat_path(self.data_root, ds)
        if not p.exists():
            return []
        s = pd.read_parquet(p, columns=[col])[col].dropna().astype(str)
        codes = sorted(s.unique().tolist())
        # ★ 实体代码前缀白名单（spec.entity_prefix_filter）。放在这里而不是 run_per_entity
        #   内部，是为了让**下载 / doctor / verify 三处口径天然一致** —— 本函数是实体列表的
        #   唯一出口（engine.py:653 下载、main.py doctor、scripts/verify.py 抽样都用它）。
        #   被过滤掉的代码从不出现在 tasks 里 → 永不写进 done.entities → 将来去掉白名单
        #   即可增量补回，不需要清理已落地数据。
        flt = getattr(spec, "entity_prefix_filter", None)
        if flt:
            codes = [c for c in codes if c.split(".")[0][:3] in flt]
        return codes

    # ------------------------------------------------------------ 落库
    def _to_df(self, rows: list[dict], spec: Spec, man: Manifest) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.columns = [str(c) for c in df.columns]
        # 统一纯日期列格式：服务端有时返回 "2024-03-01 00:00:00"，必须归一成 YYYY-MM-DD，
        # 否则分区、合并、下游因子对齐都会出错。
        for c in PURE_DATE_COLS & set(df.columns):
            df[c] = df[c].astype(str).str[:10].where(df[c].notna(), None)
        # 稳定列顺序：首次落库的列序作为规范，之后新增列追加在后面
        if not man.columns:
            man.columns = list(df.columns)
            man.dirty = True
        else:
            ordered = [c for c in man.columns if c in df.columns]
            extra = [c for c in df.columns if c not in man.columns]
            if extra:
                man.columns = list(man.columns) + extra
                man.dirty = True
            df = df.reindex(columns=ordered + extra)
        return df

    def _save(self, spec: Spec, man: Manifest, df: pd.DataFrame, year: int,
              keys: tuple, parts_out: list | None = None) -> int:
        """把数据并入年分区。

        这里**不立刻落盘**，而是先攒进内存缓冲，原因：
        parquet 是整文件重写，一个年分区被反复写 10 次就是 O(n²) 的 I/O，
        而且 upsert 在主线程串行执行，会直接卡住整个并发流水线。
        实测 st_info 一个数据集 331 次请求里，主线程写盘成了瓶颈。
        缓冲到阈值（默认 30 万行）再一次性合并，重写次数能降一个数量级。
        """
        if df is None or len(df) == 0:
            return 0
        key = (spec.name, year)
        self._buf.setdefault(key, []).append(df)
        self._buf_ctx[key] = (spec, keys)
        self.client.stats.add(rows=len(df))
        if sum(len(x) for x in self._buf[key]) >= self._flush_threshold(man, year):
            self._flush(spec, man, year, keys)
        return len(df)

    def _flush_threshold(self, man: Manifest, year: int) -> int:
        """自适应写缓冲阈值 —— 分区越大，攒得越多才写。

        parquet 是**整文件重写**（读旧表 → concat → 去重 → 排序 → 写回），
        所以单个分区的写入总代价 ≈ Σ 每次 flush 时的分区大小。固定 30 万行
        阈值下，一个 1.4 亿行的分区要被重写 ~470 次，总代价是 O(n²)。

        实测（本机 / 共享盘）：分区 500 万行时写 30 万行要 9s，2000 万行要 31s。
        按固定阈值外推，cyq_chips 单年分区要写 100+ 小时 —— 比抓取本身还长
        一个数量级。让阈值随分区大小线性放大后，flush 次数从 ~470 降到 ~26，
        代价变成几何级数（约 3% 的额外 I/O）。

        阈值上限是分区的 1/4，因此缓冲内存 ≈ 分区大小/4（cyq_chips 单年
        1.4 亿行 × 51B ≈ 7.1GB → 缓冲峰值 ~1.8GB/年，9 年 ~16GB）。
        """
        on_disk = int(man.partitions.get(str(year), {}).get("rows", 0))
        return max(self._flush_rows, on_disk // 4)

    def _flush(self, spec: Spec, man: Manifest, year: int, keys: tuple) -> int:
        """把某个 (数据集, 年份) 的缓冲真正合并进 parquet 并更新 manifest。"""
        frames = self._buf.pop((spec.name, year), None)
        self._buf_ctx.pop((spec.name, year), None)
        if not frames:
            return 0
        df = pd.concat(frames, ignore_index=True)
        if spec.partition == "none":
            path = store.flat_path(self.data_root, spec.name)
        else:
            path = store.year_partition_path(self.data_root, spec.name, year)
        merged = store.upsert(path, df, keys, sort_by=spec.sort_by or keys)
        n = len(merged)
        mn = mx = None
        fld = spec.date_field
        if fld in merged.columns and n:
            try:
                mn = str(merged[fld].min())[:10]
                mx = str(merged[fld].max())[:10]
            except (TypeError, ValueError):
                pass
        man.mark_partition(year, n, mn, mx)
        return len(df)

    def _rows_display(self, man: Manifest) -> int:
        """进度显示用的行数 = 已落盘 + 还在缓冲里的。

        只有已落盘部分记在 manifest.partitions 里，直接拿它显示会出现
        "明明在下数据、进度条却一直 rows=0" 的误导。
        """
        n = man.partition_rows()
        for (name, _y), frames in self._buf.items():
            if name == man.name:
                n += sum(len(f) for f in frames)
        return n

    def _flush_all(self, man: Manifest) -> None:
        """把所有缓冲落盘。manifest 落盘前必须先调它，
        否则会出现「manifest 说已完成、但数据还在内存里」的不一致。"""
        for (name, year) in list(self._buf.keys()):
            spec, keys = self._buf_ctx.get((name, year), (None, None))
            if spec is not None:
                self._flush(spec, man, year, keys)

    def _maybe_save(self, man: Manifest, i: int, every: int = 200,
                    flush: bool = True) -> None:
        """周期性落盘 manifest。

        没有它的话，进程被 kill / 超时中断时，parquet 已经写进磁盘，
        但"哪些日期已经下过"的记录还在内存里 —— 重跑会把已完成的区间再拉一遍。

        flush=False 用于**循环内不写完成标记**的调用方（目前只有 per_entity）：
        那种情况下 manifest 不记录缓冲中的行，所以不落盘数据也不会出现
        "状态说完成了、数据还在内存里"的不一致；而强制全量 flush 会把
        自适应阈值废掉（每 100 个任务就把所有大分区重写一遍）。
        """
        if i and i % every == 0:
            if flush:
                self._flush_all(man)
            man.save()

    # ------------------------------------------------------------ 并发调度
    def _do_fetch(self, spec: Spec, payload: dict,
                  empty_retries: int | None = None) -> tuple[list[dict], str | None]:
        """在 worker 线程里执行一次完整抓取（含分页）。异常转成字符串返回，
        不让它炸掉整个线程池 —— 单个任务失败只该影响它自己。"""
        try:
            if spec.paginated:
                return self._fetch_paginated(spec, payload,
                                             empty_retries=empty_retries), None
            data = self.client.call(spec.path, payload, method=spec.method,
                                    expect_rows=spec.expect_rows,
                                    empty_retries=empty_retries)
            return extract_list(data), None
        except (ApiError, RetryableError, RuntimeError) as exc:
            return [], f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            return [], f"{type(exc).__name__}: {exc}"

    def _iter_concurrent(self, tasks: list, bar, workers: int | None = None,
                         slow: bool = False):
        """并发抓取并按完成顺序 yield (task, rows, error)。

        为什么要有它：引擎原本是纯串行的，每个请求 ~1 秒，实际只跑到约
        50 次/分钟，而限额是 280 —— 只用了 20% 的额度。加并发后能把
        数十小时的下载压到几小时。

        并发只发生在**网络抓取**阶段；合并、去重、落盘仍在主线程串行执行，
        避免多线程同时改同一个 parquet / manifest 造成损坏。

        in-flight 上限定为 workers*2，既喂饱线程池又不让结果在内存里堆积。
        """
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        # slow 档单独限流：这类接口单次要 10~25 秒（服务端重查询），
        # 并发太多会给服务端压力，但用 2~4 个并发就能把 12 小时压到 1~2 小时。
        if workers is None:
            workers = (self.cfg["api"].get("slow_concurrency", 2)
                       if slow else self.cfg["api"]["concurrency"])
        workers = max(1, workers)
        it = iter(tasks)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            inflight: dict = {}

            def fill() -> None:
                while len(inflight) < workers * 2:
                    try:
                        t = next(it)
                    except StopIteration:
                        return
                    key, fetch_fn = t
                    inflight[ex.submit(fetch_fn)] = key

            fill()
            while inflight:
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for fut in done:
                    key = inflight.pop(fut)
                    rows, err = fut.result()
                    yield key, rows, err
                fill()

    # ------------------------------------------------------------ 分页校验
    def _fetch_paginated(self, spec: Spec, payload: dict, bar=None,
                         empty_retries: int | None = None) -> list[dict]:
        """分页拉取并校验总行数；不足则整段重取（应对服务端偶发空页）。"""
        for attempt in range(1, 4):
            rows: list[dict] = []
            total = None
            page = 0
            while page < 100000:
                body = {**payload, "page": page, "page_size": spec.page_size}
                data = self.client.call(spec.path, body, method=spec.method,
                                        expect_rows=spec.expect_rows and page == 0,
                                        empty_retries=empty_retries)
                batch = extract_list(data)
                if total is None:
                    total = extract_total(data)
                    # ★ 服务端**没给** total 时，extract_total 会退化成"当前页长度"
                    #   （client.py:179-180）。此时若当前页正好满 page_size，下面的
                    #   `len(rows) >= total` 会立刻成立 → **只发一次请求就静默截断在
                    #   page_size 行，且不打任何警告**。满页时这个"假 total"不可信，
                    #   丢弃它、退回"翻到空页才停"的规则（多一次请求，但不会丢数据）。
                    #   仅在满页时丢弃，是为了不影响数组型响应（data 是 list）的既有行为。
                    if (isinstance(data, dict) and data.get("total") is None
                            and len(batch) >= spec.page_size):
                        total = None
                    # ★ 前置拦截：服务端硬限制 page*page_size ≤ 100000。
                    #   如果这一段总量本身就超了，继续翻页到 page=11 会收到一个
                    #   很难懂的 code=400（"请使用日期遍历方式获取数据"），
                    #   而且**整个 chunk 会失败**。不如在第一页就问出来，
                    #   直接告诉调用者该调小 chunk_days。
                    elif total is not None and total > MAX_ROWS_PER_QUERY:
                        raise RuntimeError(
                            f"{spec.name} 单个任务要取 {total:,} 行，超过服务端上限 "
                            f"{MAX_ROWS_PER_QUERY:,} 行（page*page_size ≤ 十万）。"
                            f"请调小该数据集的 chunk_days（当前 {spec.chunk_days or '未设，按 tier 默认'}）"
                            f"，或改为按实体遍历。"
                        )
                rows.extend(batch)
                if bar is not None and batch:
                    self.board.tick(bar, 0, rows=fmt_rows(len(rows)))
                if not batch:
                    break
                if total is not None and len(rows) >= total:
                    break
                page += 1
            if len(rows) >= MAX_ROWS_PER_QUERY and (total is None or len(rows) < total):
                # 撞到 page*page_size 上限：数据被截断，必须缩小切分粒度
                raise RuntimeError(
                    f"{spec.name} 单任务取回 {len(rows):,} 行已达服务端上限 "
                    f"{MAX_ROWS_PER_QUERY:,}（total={total}）。"
                    f"请调小该数据集的 chunk_days 或改为按实体遍历。"
                )
            if total is None or len(rows) >= total or not rows:
                return rows
            log.warning("%s 分页不足 (%d/%d)，整段重取 (第%d次)", spec.name, len(rows), total, attempt)
            self.client.stats.add(retries=1)
            time.sleep(1.5 * attempt)
        # ★ 3 次「分页不足」重试全部用尽：**绝不能把残缺数据当成功返回**。
        #   调用方 run_per_entity 见 err=None 会 ok_tasks_per_entity[c] += 1，
        #   进而把该实体标进 done.entities —— 残缺数据就成了**永久空洞**且无人察觉
        #   （与 QUANT_PLATFORM.md §16.3 的 index_daily 是同一类 bug，只是触发路径不同）。
        #   抛出去 → _do_fetch 转成 err → 任务不计成功 → 实体不标 done → 下轮重试。
        raise RuntimeError(
            f"{spec.name} 分页校验连续 3 次失败（最后一次拿到 {len(rows):,} / 应有 {total:,} 行）。"
            f"不返回残缺数据，留给下轮重试。"
        )

    # ------------------------------------------------------------ 模式实现
    def run_snapshot(self, spec: Spec, man: Manifest, ctx: dict) -> dict:
        bar = self.board.task(len(spec.variants or [None]), f"{spec.name} 快照")
        end = ctx.get("end") or today_str()
        frames = []
        for variant in (spec.variants or [None]):
            payload = {**spec.params, **(variant or {})}
            if spec.range_params:
                payload[spec.start_param] = spec.start
                payload[spec.end_param] = end
            if spec.paginated:
                rows = self._fetch_paginated(spec, payload, bar)
            else:
                data = self.client.call(spec.path, payload, method=spec.method,
                                        expect_rows=spec.expect_rows)
                rows = extract_list(data)
            if rows:
                df = pd.DataFrame(rows)
                for k, v in (variant or {}).items():
                    df[k] = v
                frames.append(df)
            self.board.tick(bar, 1)
        if not frames:
            return {"rows": 0}
        df = pd.concat(frames, ignore_index=True)
        man.columns = list(df.columns)
        out = store.overwrite(store.flat_path(self.data_root, spec.name), df,
                              sort_by=spec.sort_by or spec.keys)
        fld = spec.date_field
        mn = mx = None
        if fld in out.columns and len(out):
            try:
                mn = str(out[fld].min())[:10]
                mx = str(out[fld].max())[:10]
            except (TypeError, ValueError):
                pass
        man.mark_partition(0, len(out), mn, mx)
        self.client.stats.add(rows=len(out))
        man.add_coverage(spec.start, end)
        return {"rows": len(out)}

    def _plan_ranges(self, spec: Spec, man: Manifest, ctx: dict) -> list[tuple[str, str]]:
        end = ctx.get("end") or today_str()
        start = spec.start
        if ctx.get("full"):
            rng = [(start, end)]
        else:
            rng = man.missing_ranges(start, end)
            # 修正窗口：最近 N 天总是重刷，让追溯修正的数据收敛
            rev_start = (d(end) - dt.timedelta(days=self.revision_days)).isoformat()
            if rev_start <= end:
                rng.append((max(rev_start, start), end))
            # 去重
            seen, uniq = set(), []
            for a, b in rng:
                if (a, b) not in seen:
                    seen.add((a, b))
                    uniq.append((a, b))
            rng = uniq
        return rng

    def run_range(self, spec: Spec, man: Manifest, ctx: dict) -> dict:
        ranges = self._plan_ranges(spec, man, ctx)
        if not ranges:
            return {"rows": 0, "skipped": True}

        variants = spec.variants or [None]
        chunk_days = spec.chunk_days or CHUNK_DAYS.get(spec.tier, 30)
        tasks: list[tuple] = []
        for a, b in ranges:
            for ya, yb in split_by_year(a, b):
                for chunk in split_chunks(ya, yb, chunk_days):
                    for vi in range(len(variants)):
                        tasks.append((*chunk, vi))

        bar = self.board.task(len(tasks), f"{spec.name} 区间")
        added = 0
        engine_self = self

        def make_payload(a: str, b: str, vi: int) -> dict:
            return {**spec.params, **(variants[vi] or {}),
                    spec.start_param: a, spec.end_param: b}

        # 区间任务只重试 1 次空响应：早年区间"合法为空"很常见，
        # 多重退避会拖垮吞吐。真正的漏数据由 suspect + verify 兜底。
        work = [((a, b, vi),
                 (lambda a=a, b=b, vi=vi: engine_self._do_fetch(
                     spec, make_payload(a, b, vi), empty_retries=1)))
                for a, b, vi in tasks]

        # 按区间汇总"是否拿到过数据"：变体之间取或，避免可变体互相覆盖判定
        range_hits: dict[tuple, bool] = {}
        for task_i, (key, rows, err) in enumerate(
                self._iter_concurrent(work, bar, slow=(spec.tier == "slow"))):
            a, b, vi = key
            variant = variants[vi]
            if err:
                log.warning("%s %s~%s 失败: %s", spec.name, a, b, err)
                # ★ 2026-09-13 修：这里原本写 `= True`，注释说"出错不算确认为空"，
                #   但 True 在下游的含义是 **"拿到了数据 → 标记已覆盖"**，
                #   于是**抓取失败的区间反而被记成已覆盖，永远不会重试**。
                #   实测代价：index_daily 的 chunk_days=60 要取 31 万行，分页到
                #   page=11 时撞上服务端 `page*page_size ≤ 100000` 硬上限报 400，
                #   整个 2026 年的 159 个交易日就这样被静默标记成"已覆盖"
                #   （QUANT_PLATFORM.md §16.3 问题 1 的真正成因）。
                #   现在记为 "error" 哨兵：既不标覆盖，也不当作"确认为空"，
                #   而是记入 suspect 等下次重试。
                range_hits[(a, b)] = "error"
                self.board.tick(bar, 1)
                continue
            if rows:
                if variant:
                    for r in rows:
                        r.update(variant)
                df = self._to_df(rows, spec, man)
                col = spec.date_field if spec.date_field in df.columns else df.columns[0]
                for year in sorted({int(str(v)[:4]) for v in df[col]
                                    if v is not None and str(v)[:4].isdigit()}):
                    sub = df[df[col].astype(str).str[:4] == str(year)]
                    added += self._save(spec, man, sub, year, spec.keys)
                if man.data_start is None:
                    man.data_start = a
                range_hits[(a, b)] = True
            else:
                range_hits.setdefault((a, b), False)
            self.board.tick(bar, 1, rows=fmt_rows(self._rows_display(man)))
            self.board.status.update(force=False, dataset=spec.name,
                                         rows=self._rows_display(man))
            self._maybe_save(man, task_i)

        # 覆盖标记：只有"确实拿到过数据"或"确认该区间无交易日"才算已覆盖。
        # ⚠️ 三种状态必须分清，否则就是 index_daily 那个 bug：
        #   True   拿到数据        → 标覆盖
        #   False  确认为空        → 含交易日且 expect_rows 就记 suspect，否则标覆盖
        #   "error" 抓取失败       → **绝不标覆盖**，记 suspect 等下次重试
        for (a, b), hit in range_hits.items():
            seq = f"{a}~{b}"
            if hit == "error":
                man.suspect[seq] = man.suspect.get(seq, 0) + 1
                log.warning("%s %s 抓取失败，不标覆盖、记入 suspect 等下次重试", spec.name, seq)
                continue
            if hit:
                man.add_coverage(a, b)
                continue
            tdays = self.trading_days_between(a, b)
            if spec.expect_rows and tdays:
                man.suspect[seq] = man.suspect.get(seq, 0) + 1
                log.warning("%s %s 返回空但含 %d 个交易日，标记可疑待复查",
                            spec.name, seq, len(tdays))
            else:
                man.add_coverage(a, b)
        return {"rows": added, "tasks": len(tasks)}

    def run_per_date(self, spec: Spec, man: Manifest, ctx: dict) -> dict:
        end = ctx.get("end") or today_str()
        start = spec.start
        all_days = self.trading_days_between(start, end)
        if ctx.get("full"):
            days = all_days
        else:
            days = [x for x in all_days if not man.is_done("dates", x)]
            rev_start = (d(end) - dt.timedelta(days=self.revision_days)).isoformat()
            days = sorted(set(days) | {x for x in all_days if x >= rev_start})
        if not days:
            return {"rows": 0, "skipped": True}

        bar = self.board.task(len(days), f"{spec.name} 逐日")
        added = 0
        engine_self = self

        def fetch_day(day: str) -> tuple[list[dict], str | None]:
            """一天的请求可能包含多个变体（如热度榜的 6 个榜单），合并成一批。"""
            out: list[dict] = []
            for variant in (spec.variants or [None]):
                payload = {**spec.params, **(variant or {}), spec.date_param: day}
                data = None
                try:
                    if spec.paginated:
                        # ★ 2026-09-14 修复（全库核查发现）：原实现直接 client.call，
                        #   **既不分页也不显式传 page_size** → 服务端按**默认每页**返回。
                        #   /api/stock/dragon_tiger 的默认 page_size 是 **20**，于是每个
                        #   交易日只落地了前 20 行：本地 49,541 行 vs 上游约 226 万行
                        #   （**只有 2.19%**），而被丢掉的行永远不会被 coverage/done 检查发现。
                        #   更坑的是服务端此时把 `total` 也谎报成 20（实测），
                        #   所以"用 total 校验行数"这条路也走不通 —— 只有显式传 page_size 才拿得全。
                        #   改走 _fetch_paginated（与 range / snapshot 同一条通道），
                        #   它会显式传 page_size 并翻页收齐，收不齐就抛异常而不是静默截断。
                        batch = engine_self._fetch_paginated(spec, payload)
                    else:
                        data = engine_self.client.call(spec.path, payload,
                                                       method=spec.method, expect_rows=False)
                        batch = extract_list(data)
                except ApiError as exc:
                    log.warning("%s %s 业务错误: %s", spec.name, day, exc)
                    continue
                if not batch and isinstance(data, dict):
                    batch = data.get("list") or []
                # 标量数组响应（如 adj_factor/changes 直接给代码列表）统一包成记录
                if batch and not isinstance(batch[0], dict):
                    col = spec.array_of or "value"
                    batch = [{col: v} for v in batch]
                for r in batch:
                    r.setdefault(spec.date_field, day)
                    if variant:
                        r.update(variant)
                out.extend(batch)
            return out, None

        work = [(day, (lambda day=day: fetch_day(day))) for day in days]
        for day_i, (day, rows, err) in enumerate(
                self._iter_concurrent(work, bar, slow=(spec.tier == "slow"))):
            if err:
                log.warning("%s %s 失败: %s", spec.name, day, err)
                self.board.tick(bar, 1)
                continue
            if rows:
                df = self._to_df(rows, spec, man)
                added += self._save(spec, man, df, year_of(day), spec.keys)
            man.mark_done("dates", day)
            self.board.tick(bar, 1, rows=fmt_rows(self._rows_display(man)))
            self._maybe_save(man, day_i)
        return {"rows": added, "dates": len(days)}

    def run_per_stock(self, spec: Spec, man: Manifest, ctx: dict) -> dict:
        codes = self.stock_codes()
        todo = [c for c in codes if ctx.get("full") or not man.is_done("stocks", c)]
        if not todo:
            return {"rows": 0, "skipped": True}

        batch = max(1, spec.batching)
        groups = [todo[i:i + batch] for i in range(0, len(todo), batch)]
        bar = self.board.task(len(todo), f"{spec.name} 逐股")
        added = 0
        engine_self = self

        def payload_for(grp):
            return {**spec.params, spec.code_param: grp if batch > 1 else grp[0]}

        work = [(grp, (lambda grp=grp: engine_self._do_fetch(spec, payload_for(grp))))
                for grp in groups]

        for si, (grp, rows, err) in enumerate(self._iter_concurrent(work, bar)):
            if err:
                log.warning("%s %s 失败: %s", spec.name, grp, err)
                self.board.tick(bar, len(grp))
                continue                      # 不标记完成，下次重跑会补
            if rows:
                df = self._to_df(rows, spec, man)
                col = spec.date_field if spec.date_field in df.columns else None
                if col:
                    for year in sorted({year_of(str(v)[:10]) for v in df[col]
                                        if v is not None and str(v)[:4].isdigit()}):
                        sub = df[df[col].astype(str).str[:4] == str(year)]
                        added += self._save(spec, man, sub, year, spec.keys)
                else:
                    added += self._save(spec, man, df, 0, spec.keys)
            for c in grp:
                man.mark_done("stocks", c)
            self.board.tick(bar, len(grp), rows=fmt_rows(self._rows_display(man)))
            self._maybe_save(man, si, every=100)
        return {"rows": added, "stocks": len(todo)}

    def run_per_entity(self, spec: Spec, man: Manifest, ctx: dict) -> dict:
        codes = self._entity_codes(spec)
        if not codes:
            return {"rows": 0, "error": "实体列表为空，请先下载其依赖数据集"}
        end = ctx.get("end") or today_str()
        tasks: list[tuple] = []
        n_revisit = 0
        for c in codes:
            if not ctx.get("full") and man.is_done("entities", c):
                # ★ 2026-09-16 修「done 实体永久跳过」——
                #   原实现直接 continue，于是 `done.entities` 满员后这张表**每轮 0 任务**：
                #   `main.py run index_weight` 永远「新增 0 行 / 请求 0」，上游哪怕补发了
                #   新数据也永远抓不回来（index_weight 的 2026-08/09 就是这么丢的）。
                #   改为仍重访**尾部窗口**（月频表近 TAIL_MONTHS 个月），
                #   于是「落后 → 补窗追上 → 稳定」成为常态，而不是永久冻结。
                tail = _tail_months(spec, end)
                if tail:
                    tasks.extend((c, m, m) for m in tail)
                    n_revisit += 1
                continue
            if not spec.entity_range:
                tasks.append((c, None, None))
            elif spec.date_step == "month":
                # 按月遍历：服务端按月存储（如指数权重），传 YYYY-MM 粒度即可
                tasks.extend((c, m, m) for m in _months_between(spec.start, end))
            else:
                for ya, yb in split_by_year(spec.start, end):
                    for chunk in split_chunks(ya, yb, spec.chunk_days or 3660):
                        tasks.append((c, chunk[0], chunk[1]))
        if not tasks:
            return {"rows": 0, "skipped": True}
        if n_revisit:
            log.info("%s 尾部重访：%d/%d 个已完成实体 × 近 %d 个月（截至 %s）",
                     spec.name, n_revisit, len(codes), TAIL_MONTHS, end[:7])

        bar = self.board.task(len(tasks), f"{spec.name} 逐实体")
        added = 0
        # 只有"该实体的所有分片都成功"才算完成，失败的留到下次重跑
        n_tasks_per_entity: dict[str, int] = {}
        ok_tasks_per_entity: dict[str, int] = {}
        for c, _, _ in tasks:
            n_tasks_per_entity[c] = n_tasks_per_entity.get(c, 0) + 1

        engine_self = self

        def make_payload(c, a, b):
            p = {**spec.params, spec.code_param: c}
            if a and spec.date_step == "month":
                p[spec.date_param] = a            # YYYY-MM
            elif a:
                p[spec.start_param] = a
                p[spec.end_param] = b
            return p

        # 并发抓取（网络）→ 主线程落盘（避免多线程同时改同一个 parquet/manifest）
        work = [((c, a, b),
                 (lambda c=c, a=a, b=b: engine_self._do_fetch(spec, make_payload(c, a, b))))
                for c, a, b in tasks]

        for task_i, (key, rows, err) in enumerate(self._iter_concurrent(work, bar)):
            c, a, b = key
            if err:
                log.warning("%s %s 失败: %s", spec.name, c, err)
                self.board.tick(bar, 1)
                continue
            ok_tasks_per_entity[c] = ok_tasks_per_entity.get(c, 0) + 1
            if rows:
                df = self._to_df(rows, spec, man)
                if spec.code_param not in df.columns:
                    df[spec.code_param] = c
                if spec.partition == "none":
                    added += self._save(spec, man, df, 0, spec.keys)
                else:
                    col = spec.date_field if spec.date_field in df.columns else None
                    years = ([int(str(v)[:4]) for v in df[col]
                              if v is not None and str(v)[:4].isdigit()] if col
                             else [year_of(a) if a else 0])
                    for year in sorted(set(years)):
                        sub = (df[df[col].astype(str).str[:4] == str(year)] if col else df)
                        added += self._save(spec, man, sub, year, spec.keys)
            self.board.tick(bar, 1, rows=fmt_rows(self._rows_display(man)))
            # flush=False：本循环不写完成标记，manifest 不记录缓冲行，安全。
            # （per_stock / per_date 循环内 mark_done，必须保持 flush=True）
            self._maybe_save(man, task_i, every=100, flush=False)

            # ── 周期性 checkpoint（2026-09-13 加）──────────────────────────
            # 原实现的致命问题：done.entities 只在**整轮跑完后**写一次。
            # cyq_chips 满载 13 小时，若在第 12 小时被 kill / OOM / 超时，
            # 数据虽不脏（parquet 已在盘上），但 manifest 一条完成标记都没有
            # → 重跑会把 12 小时的工作**全部重下一遍**，白烧约 6 万次请求额度。
            #
            # ⚠️ 顺序不能反：**先 _flush_all 落盘数据，再 mark_done，最后 save**。
            #    反过来会出现"状态说完成、数据还在内存缓冲里"，此时崩溃会留下
            #    永久空洞 —— 下次增量跳过该实体，数据再也补不回来。
            #
            # 代价：强制 flush 会绕过自适应阈值（见 _maybe_save 的说明），
            # 多写若干次大分区。取 5000 而非 100，就是为了让额外开销可控：
            # 按 §11.2 实测量级，cyq_chips 自然 flush 约 234 次，
            # 间隔 5000 再加约 99 次 → 写盘 +约 40%，换崩溃最多丢约 1.2 小时。
            if self._ckpt_every and (task_i + 1) % self._ckpt_every == 0:
                self._flush_all(man)
                if not ctx.get("full"):
                    n_done = 0
                    for c, n in n_tasks_per_entity.items():
                        if (ok_tasks_per_entity.get(c, 0) >= n
                                and not man.is_done("entities", c)):
                            man.mark_done("entities", c)
                            n_done += 1
                man.save()
                log.info("%s checkpoint @ %d/%d 任务（已完成 %d 个实体）",
                         spec.name, task_i + 1, len(tasks),
                         len(man.done.get("entities", {})))

        self._flush_all(man)      # ★ 先落盘数据，再写完成标记（否则中断会丢数据）
        if not ctx.get("full"):
            for c, n in n_tasks_per_entity.items():
                if ok_tasks_per_entity.get(c, 0) >= n:
                    man.mark_done("entities", c)
        man.save()
        return {"rows": added, "entities": len(n_tasks_per_entity),
                "revisit": n_revisit,
                "incomplete": sum(1 for c, n in n_tasks_per_entity.items()
                                  if ok_tasks_per_entity.get(c, 0) < n)}

    def run_dump(self, spec: Spec, man: Manifest, ctx: dict) -> dict:
        """daily_dump 通道：最近 90 天 + 严格配额台账。"""
        levels = ctx.get("levels") or self.cfg["minute"]["levels"]
        end = ctx.get("end") or today_str()
        look = int(self.cfg["minute"].get("dump_lookback_days", 90))
        start = max((d(end) - dt.timedelta(days=look)).isoformat(), spec.start)
        days = self.trading_days_between(start, end)
        tasks = [(lv, day) for lv in levels for day in days]
        todo = [(lv, day) for lv, day in tasks if not man.is_done("dumps", f"{lv}|{day}")]
        if not todo:
            return {"rows": 0, "skipped": True}

        bar = self.board.task(len(todo), f"{spec.name} 打包")
        added = 0
        for lv, day in todo:
            if not self.dump_quota.can_download(day, lv):
                log.error("配额已满，跳过 %s %s（避免封禁）", day, lv)
                self.board.tick(bar, 1)
                continue
            # ★ 乐观记账（2026-09-13 修）：服务端按**请求次数**计配额，不是按成功次数；
            #   原来 record() 放在请求成功之后，超时/重试消耗的额度不计入台账 → 低报，
            #   可能在"台账显示未超限"时触到服务端封禁（超限封该日期 3 天）。
            #   DumpQuota 的意图本就是"宁可少下一次也不能触发封禁"，改到请求前才符合。
            self.dump_quota.record(day, lv)
            payload = {"date": day, "level": lv}
            try:
                # expect_rows=False：dump 的空值语义是"该日期没有数据"，且下面会 mark_done，
                # 不该走"空结果重试"。原为 True，配合 client 的 Map 误判会把
                # 一次请求放大成 5 次真实下载（见 client.call 的注释）。
                data = self.client.call(spec.path, payload, method="POST",
                                        expect_rows=False)
            except ApiError as exc:
                # 超出 90 天窗口等业务拒绝：记录下来不再重试
                log.warning("dump %s %s 业务拒绝: %s", day, lv, exc)
                man.mark_done("dumps", f"{lv}|{day}")
                self.board.tick(bar, 1)
                continue
            except RetryableError as exc:
                log.warning("dump %s %s 失败: %s", day, lv, exc)
                self.board.tick(bar, 1)
                continue
            out_name = f"stock_dump_{lv}"
            df = _dump_to_df(data, day, lv)
            if df is not None and len(df):
                sub = Manifest.load(self.state_root, out_name)
                sub.columns = sub.columns or list(df.columns)
                keys = (["trade_date", "stock_code"] if lv == "daily"
                        else ["trade_date", "stock_code", "trade_time"])
                merged = store.upsert(
                    store.year_partition_path(self.data_root, out_name, year_of(day)),
                    df, keys, sort_by=keys)
                sub.mark_partition(year_of(day), len(merged), day, day)
                sub.add_coverage(day, day)
                sub.save()
                added += len(df)
            man.mark_done("dumps", f"{lv}|{day}")
            man.save()
            self.board.tick(bar, 1, rows=fmt_rows(added))
        return {"rows": added, "dumps": len(todo)}

    # ------------------------------------------------------------ 分发
    MODES = {
        "snapshot": run_snapshot, "range": run_range, "per_date": run_per_date,
        "per_stock": run_per_stock, "per_entity": run_per_entity, "dump": run_dump,
    }

    def run(self, spec: Spec, full: bool = False, end: str | None = None,
            levels: list[str] | None = None) -> dict:
        man = Manifest.load(self.state_root, spec.name)
        ctx = {"full": full, "end": end, "levels": levels}
        t0 = time.time()
        self.board.dataset = spec.name
        self.board.note(f"▶ {spec.name}  ({spec.desc})  已有 {man.summary()}")
        try:
            fn = self.MODES[spec.mode]
            result = fn(self, spec, man, ctx)
        except Exception as exc:  # noqa: BLE001
            man.finish_run(status="failed", error=str(exc))
            man.save()
            raise
        self._flush_all(man)          # ★ 先落盘数据，再落盘状态，保证两者一致
        man.finish_run(status="ok", **{k: v for k, v in result.items() if k != "rows"},
                       rows=result.get("rows", 0), seconds=round(time.time() - t0, 1))
        man.save()
        el = time.time() - t0
        st = self.client.stats.snapshot()
        self.board.note(
            f"✔ {spec.name}  新增 {result.get('rows', 0):,} 行  用时 {el:.1f}s  "
            f"| 累计 {man.summary()} | 请求 {st['requests']} 重试 {st['retries']}"
        )
        return result


def _dump_to_df(data, day: str, level: str) -> pd.DataFrame | None:
    """daily_dump 的两种返回形态 → DataFrame。

    level=daily  : data 是记录数组
    level=Xmin   : data 是 {股票代码: [[时间,开,高,低,收,量,额], ...]}
    """
    if data is None:
        return None
    if isinstance(data, list):
        df = pd.DataFrame(data)
        if "trade_date" not in df.columns:
            df["trade_date"] = day
        if "stock_code" not in df.columns and "code" in df.columns:
            df = df.rename(columns={"code": "stock_code"})
        return df.drop_duplicates(subset=["trade_date", "stock_code"], keep="last")

    if isinstance(data, dict):
        rows = []
        for code, bars in data.items():
            for b in bars or []:
                if not isinstance(b, (list, tuple)) or len(b) < 7:
                    continue
                t = str(b[0])
                if len(t) == 5:            # "09:35"
                    t = f"{day} {t}:00"
                elif len(t) == 8:          # "09:35:00"
                    t = f"{day} {t}"
                rows.append({
                    "trade_date": day, "stock_code": code, "trade_time": t,
                    "open": b[1], "high": b[2], "low": b[3], "close": b[4],
                    "vol": b[5], "amount": b[6],
                })
        if not rows:
            return None
        df = pd.DataFrame(rows)
        # 实测 5min 打包数据存在完全重复行，按主键去重
        return df.drop_duplicates(subset=["trade_date", "stock_code", "trade_time"], keep="last")
    return None
