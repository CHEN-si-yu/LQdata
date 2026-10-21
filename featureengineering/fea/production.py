"""Direct production: coverage planning, a persistent pool and per-task checkpoints.

No recipe/input/output hashes, audits, evaluation or release pipeline.
Formula execution uses the existing Engine so date anchors and PIT semantics stay intact.
"""
from __future__ import annotations

import copy
import datetime as dt
import fcntl
import gc
import importlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import time
import traceback
import warnings
from types import MethodType

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import store
from .dates import int_to_str, int_to_str_vec, series_to_int, str_to_int
from .deriv import financial_dependency
from .engine import Engine, minus_days
from .manifest import Manifest, _atomic_json
from .progress import ProductionProgress, announce
from .resources import memory_limits, safe_jobs
from .spec import REGISTRY

log = logging.getLogger(__name__)


def registered_specs(cfg):
    """Load enabled registrations, including production builders, without reviews."""
    import factors  # noqa: F401
    path = cfg.input_contract_path("production_registry", "conf/production.json")
    if path.exists():
        library = json.loads(path.read_text(encoding="utf-8"))
        from research.catalog import build_spec
        for item in library.get("objects", []):
            if item.get("status") != "production" or item["name"] in REGISTRY:
                continue
            identity = item.get("candidate_id")
            if not identity or not re.fullmatch(r"[A-Za-z0-9_]+", identity):
                raise ValueError(f"缺少注册实现：{item['name']}")
            definition_path = cfg.contract_root / "artifacts/research/catalog/candidates" / (identity + ".json")
            definition = json.loads(definition_path.read_text(encoding="utf-8"))["definition"]
            reference = definition["implementation"]
            if not re.fullmatch(r"research\.[A-Za-z0-9_.]+:[A-Za-z_][A-Za-z0-9_]*", reference):
                raise ValueError(f"非法因子实现：{reference}")
            module_name, builder_name = reference.split(":")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                module = importlib.import_module(module_name)
            if not Path(module.__file__).resolve().is_relative_to(cfg.contract_root):
                raise ValueError(f"因子实现越出项目：{reference}")
            REGISTRY[item["name"]] = build_spec(definition, getattr(module, builder_name), item["name"])
    return [s for s in REGISTRY.values() if s.enabled]


def select_specs(cfg, names=None, groups=None):
    available = {s.name: s for s in registered_specs(cfg)}
    if names:
        unknown = set(names) - available.keys()
        if unknown:
            raise ValueError("未知因子：" + ", ".join(sorted(unknown)))
        selected = [available[n] for n in dict.fromkeys(names)]
    else:
        selected = [s for s in available.values() if not groups or s.group in groups]
    result, visiting = {}, set()
    def visit(spec):
        if spec.name in visiting:
            raise ValueError(f"因子依赖循环：{spec.name}")
        if spec.name in result:
            return
        visiting.add(spec.name)
        for dep in spec.deps:
            if dep in REGISTRY:
                visit(REGISTRY[dep])
        visiting.remove(spec.name)
        result[spec.name] = spec
    for spec in selected:
        visit(spec)
    return list(result.values())


def latest_source_day(up):
    """Read only the newest stock_daily footer; fall back to one date column."""
    root = up.root / "stock_daily"
    files = sorted(root.glob("year=*/data.parquet"), reverse=True)
    if (root / "data.parquet").exists():
        files.append(root / "data.parquet")
    for path in files:
        pf = pq.ParquetFile(path)
        if not pf.metadata.num_rows:
            continue
        index = pf.schema.names.index("trade_date")
        maxima = []
        for rg in range(pf.metadata.num_row_groups):
            stats = pf.metadata.row_group(rg).column(index).statistics
            if not stats or not stats.has_min_max:
                maxima = []
                break
            value = stats.max.decode() if isinstance(stats.max, bytes) else str(stats.max)
            maxima.append(str_to_int(value))
        if maxima:
            return max(maxima)
        dates = series_to_int(pf.read(columns=["trade_date"]).to_pandas()["trade_date"])
        if dates.size and dates.max() > 0:
            return int(dates.max())
    raise RuntimeError("上游 stock_daily 没有有效交易日期")


def remove_coverage(man, lo, hi):
    """Invalidate before dispatch; an interrupted rebuild resumes as missing work."""
    left = (dt.date.fromisoformat(lo) - dt.timedelta(days=1)).isoformat()
    right = (dt.date.fromisoformat(hi) + dt.timedelta(days=1)).isoformat()
    spans = []
    for a, b in man.coverage:
        if b < lo or a > hi:
            spans.append([a, b])
        else:
            if a < lo:
                spans.append([a, left])
            if b > hi:
                spans.append([right, b])
    man.coverage = spans


def plan_factor(engine, spec, man, end, start=None, rebuild=False, refresh=False):
    floor = spec.align_start_int(engine.cfg) if spec.align_fill is not None else spec.start_int(engine.cfg)
    lo = max(floor, start or floor)
    target = engine.cal.between(lo, end)
    if not target.size:
        return target, {}
    root = engine.output_root(spec)
    if not (root / spec.name).is_dir():
        return target, {int(y): target[target // 10000 == y] for y in np.unique(target // 10000)}
    required = np.ones(len(target), dtype=bool) if rebuild or refresh else np.zeros(len(target), dtype=bool)
    if not (rebuild or refresh):
        for a, b in man.missing_ranges(int_to_str(lo), int_to_str(end)):
            required |= (target >= str_to_int(a)) & (target <= str_to_int(b))
    for year in np.unique(target // 10000):
        path = store.year_path(root, spec.name, int(year))
        mask = target // 10000 == year
        if not path.exists():
            required |= mask
            continue
        try:
            pf = pq.ParquetFile(path)
            meta = man.partitions.get(str(year))
            # A recorded complete partition needs only its footer, never a full value scan.
            if not meta or pf.metadata.num_rows != int(meta.get("rows", -1)):
                dates, counts = np.unique(series_to_int(pf.read(columns=["trade_date"]).to_pandas()["trade_date"]),
                                          return_counts=True)
                valid = dates[counts == (1 if spec.is_market else len(engine.codes))]
                required |= mask & ~np.isin(target, valid)
                if (not meta and not (rebuild or refresh)
                        and int(year) not in man.last_run.get("pending_years", [])):
                    for a, b in engine._trading_runs(valid):
                        man.add_coverage(int_to_str(a), int_to_str(b))
                    man.mark_partition(int(year), pf.metadata.num_rows, 0,
                                       int_to_str(int(dates.min())) if len(dates) else None,
                                       int_to_str(int(dates.max())) if len(dates) else None)
                    required[mask] = ~np.isin(target[mask], valid)
                    man._coverage_recovered = True
        except Exception:
            # A damaged partition must be restored on its complete existing year axis.
            required |= mask
    # Only new source days make the immature label tail worth computing again.
    previous = man.last_run.get("source_end") or man.covered_until()
    if spec.forward_days and previous and end > str_to_int(str(previous)):
        old_end = str_to_int(str(previous))
        pos = np.searchsorted(engine.cal.days, old_end, side="right")
        tail = engine.cal.days[max(0, pos-spec.forward_days-1):pos]
        required |= np.isin(target, tail)
    days = target[required]
    return target, {int(y): days[days // 10000 == y] for y in np.unique(days // 10000)}


_CACHE_SOURCES = {"intraday": ("stock_history_5min",), "open5": ("stock_history_5min",),
                  "closing30": ("stock_history_5min",), "market_morning": ("stock_history_5min",),
                  "chips": ("stock_cyq_chips",), "chipwall": ("stock_cyq_chips", "stock_daily")}


def _cache_ready(layer, y0, y1):
    for year in range(y0, y1+1):
        sources = [layer.up.root / n / f"year={year}" / "data.parquet"
                   for n in _CACHE_SOURCES[layer.name]]
        sources = [p for p in sources if p.exists()]
        if not sources:
            continue
        path = layer.root / f"year={year}" / "data.parquet"
        if not path.exists() or path.stat().st_mtime_ns < max(p.stat().st_mtime_ns for p in sources):
            return False
    return True


def _ensure_cache(layer, y0, y1):
    checked = layer.__dict__.setdefault("_production_checked", set())
    if all(y in checked for y in range(y0, y1+1)):
        return
    layer.root.mkdir(parents=True, exist_ok=True)
    # Same layer serializes cache construction. Independent layers/factors keep computing.
    with (layer.root / ".build.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for year in range(y0, y1+1):
            if year in checked:
                continue
            if not _cache_ready(layer, year, year):
                frame = layer.build_year(year, layer.up)
                if frame is not None and len(frame.columns):
                    store._atomic_write(frame, layer.root / f"year={year}" / "data.parquet", layer.cfg.compression)
                layer.clear_raw()
                layer._cache.clear()
            checked.add(year)


class ProductionEngine(Engine):
    def __init__(self, cfg):
        cfg = copy.deepcopy(cfg)
        cfg.raw["_production_only"] = True
        super().__init__(cfg)
        self.state_writes = False
        self._task_year = None
        self._task_count = 0
        self._financial_fields = None
        self._output_roots = {False: self.cfg.factors_dir, True: self.cfg.market_factors_dir}

    def output_root(self, spec):
        return self._output_roots[bool(spec.is_market)]

    def prepare_task(self, spec, year):
        if self._task_year != year or self._task_count % 16 == 0:
            self.up.clear_cache()
            self._prices = self._deriv = None
            self._deriv_window = None
            self._uni_cache.clear()
            for key in ("_intraday", "_chips", "_open5", "_closing30", "_chipwall"):
                setattr(self, key, None)
            self.up._market_morning = None
            gc.collect()
        self._task_year = year
        self._task_count += 1
        self._factor_io = None  # Parent output may have advanced since this worker's last task.
        self.factor_io()
        self._deriv_fields_run = (self._financial_fields
                                 if financial_dependency(spec, REGISTRY) else frozenset())
        for name in _CACHE_SOURCES:
            self._layer(name)  # Constructors are lazy; unused layers read no data.

    def _layer(self, name):
        layer = super()._layer(name)
        if not getattr(layer, "_production_wrapped", False):
            layer.ensure = MethodType(_ensure_cache, layer)
            layer.ready = MethodType(_cache_ready, layer)
            layer._production_wrapped = True
        return layer

    def _upsert_output(self, spec, year, frame, compression, **kwargs):
        """Full partitions write directly; increments read/merge once, without hashes."""
        store._assert_schema(frame)
        root = self.output_root(spec)
        path = store.year_path(root, spec.name, year)
        days = np.asarray(kwargs.get("replace_days", []), dtype=np.int32)
        floor = spec.align_start_int(self.cfg) if spec.align_fill is not None else spec.start_int(self.cfg)
        complete = self.cal.between(max(year*10000+101, floor),
                                    min(year*10000+1231, self._run_end))
        full = np.array_equal(days, complete)
        if path.exists() and (not full or getattr(self, "preserve_later", False)):
            old = store.read_year(root, spec.name, year)
            old = old[~old.trade_date.isin(int_to_str_vec(days))]
            frame = pd.concat([old, frame], ignore_index=True)
        # A dated run only changes requested dates; preserve any later saved rows.
        keys = ["trade_date"] if spec.is_market else ["trade_date", "stock_code"]
        if frame.duplicated(keys).any():
            raise ValueError(f"{spec.name}/{year}: 重复主键")
        frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
        for c, dtype in (store.MARKET_DTYPES if spec.is_market else store.DTYPES).items():
            frame[c] = frame[c].astype(dtype)
        store._assert_schema(frame)
        store._atomic_write(frame, path, compression)
        self._written_stats = {"partition_rows": len(frame),
                               "nonnull": int(np.isfinite(frame.value.to_numpy()).sum()),
                               "min_date": str(frame.trade_date.min()),
                               "max_date": str(frame.trade_date.max())}
        store.LAST_WRITE.clear()
        return frame


_WORKER_ENGINE = None


def _worker_init(lock_fd):
    # Empty/insufficient windows legitimately produce NaN. NumPy emits these
    # through warnings.warn, so np.errstate alone cannot keep progress clean.
    for message in (r"^Mean of empty slice\.?$",
                    r"^Degrees of freedom <= 0 for slice\.?$"):
        warnings.filterwarnings("ignore", message=message, category=RuntimeWarning)
    if lock_fd is not None:
        os.close(lock_fd)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    # Linux also terminates workers if the coordinator is killed with SIGKILL.
    # This prevents orphan writers from outliving the exclusive production lock.
    try:
        import ctypes
        parent = os.getppid()
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != parent:
            raise SystemExit(1)
    except (AttributeError, OSError):
        pass


def _compute(task):
    name, year, days = task
    engine = _WORKER_ENGINE
    try:
        spec = REGISTRY[name]
        engine.prepare_task(spec, year)
        result = engine.run_year(spec, year, days)
        result.update(engine._written_stats)
        # The checkpoint describes the whole physical partition, including rows after --end.
        if result.get("rows", 0) == 0 or not result.get("min_date"):
            raise RuntimeError("任务未产出有效日期轴")
        return result
    except Exception:
        return {"factor": name, "year": year, "error": traceback.format_exc()}


class MemoryAdmission:
    """Limit in-flight tasks using cgroup headroom and a per-worker reservation."""
    def __init__(self, policy, *, enabled=True):
        self.enabled = enabled
        root = Path("/sys/fs/cgroup")
        if (root / "memory.current").exists():
            self.used_path, self.stat_path = root / "memory.current", root / "memory.stat"
            self.anon_key, limit_path = "anon", root / "memory.max"
        else:
            root /= "memory"
            self.used_path, self.stat_path = root / "memory.usage_in_bytes", root / "memory.stat"
            self.anon_key, limit_path = "total_rss", root / "memory.limit_in_bytes"
        try:
            limit = int(limit_path.read_text())
            if limit > 1 << 62:
                limit = None
        except (OSError, ValueError):
            limit = None
        self.limits = memory_limits(limit, policy) if enabled else None
        self.base = self.read()[1]
        self.reserve = int(float(policy.get("worker_memory_gib", 3)) * 2**30)

    def read(self):
        try:
            used = int(self.used_path.read_text())
            stats = dict(line.split() for line in self.stat_path.read_text().splitlines())
            return used, int(stats.get(self.anon_key, 0))
        except (OSError, ValueError):
            return 0, 0

    def allows(self, active):
        if not self.enabled:
            return True
        used, anon = self.read()
        projected = max(anon, self.base + active*self.reserve) + self.reserve
        return projected < self.limits["anon"] and (used < self.limits["charged"] or active == 0)


def commit_task(engine, spec, man, days, result, end):
    year = int(result["year"])
    man.mark_partition(year, result["partition_rows"], result["nonnull"],
                       result["min_date"], result["max_date"])
    for a, b in engine._trading_runs(days):
        man.add_coverage(int_to_str(a), int_to_str(b))
    pending = [y for y in man.last_run.get("pending_years", []) if int(y) != year]
    source_end = max(int_to_str(end), str(man.last_run.get("source_end") or ""))
    man.last_run = {"finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "source_end": source_end, "task_days": len(days),
                    "task_seconds": result.get("seconds", 0), "year": year,
                    "pending_years": pending}
    man.save()


def dependency_starts(engine, specs, start, end):
    """Parents must also cover the child's warmup, even for a narrow output request."""
    starts = {s.name: start or s.align_start_int(engine.cfg) for s in specs}
    for spec in reversed(specs):
        days = engine.cal.between(starts[spec.name], end)
        if not days.size:
            continue
        anchor = engine._anchor_for(spec, days)
        needed = max(engine._start_floor, minus_days(anchor, spec.warmup_days))
        for dep in spec.deps:
            if dep in starts:
                starts[dep] = min(starts[dep], needed)
    return starts


def _execute(engine, specs, manifests, plans, end, jobs, progress, lock_fd=None, *, enforce_limits=True):
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    from concurrent.futures.process import BrokenProcessPool
    global _WORKER_ENGINE
    _WORKER_ENGINE = engine
    by_name = {s.name: s for s in specs}
    remaining = {(n, y): ds for n, plan in plans.items() for y, ds in plan.items()}
    pending_counts = {n: len(p) for n, p in plans.items()}
    failures, active, blocked = {}, {}, set()
    memory = MemoryAdmission(engine.cfg.raw.get("resources", {}), enabled=enforce_limits)
    pool = ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("fork"),
                               initializer=_worker_init, initargs=(lock_fd,))
    futures = {}
    try:
        while remaining or active:
            for (name, year), days in sorted(list(remaining.items()),
                                             key=lambda item: (item[0][1], -by_name[item[0][0]].warmup_days)):
                deps = [d for d in by_name[name].deps if d in by_name]
                if any(d in failures or d in blocked for d in deps):
                    blocked.add(name)
                    if progress.states[name] != "blocked":
                        progress.block(name)
                    remaining.pop((name, year))
                    pending_counts[name] -= 1
                    continue
                if len(active) >= jobs or not memory.allows(len(active)):
                    continue
                if any(pending_counts[d] > 0 for d in deps):
                    continue
                remaining.pop((name, year))
                active[(name, year)] = days
                progress.start(name, year)
                futures[pool.submit(_compute, (name, year, days))] = (name, year)
            if not active:
                if remaining:
                    if not memory.allows(0):
                        raise MemoryError("可用内存不足以启动一个计算任务；已完成任务已保存")
                    raise RuntimeError("依赖任务无法调度")
                break
            done, _ = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
            if not done:
                progress.heartbeat(len(active), memory.read()[0]/2**30)
                continue
            for future in done:
                name, year = futures.pop(future)
                try:
                    result = future.result()
                except BrokenProcessPool as exc:
                    raise RuntimeError("计算进程异常退出；已完成分区已保存，可重新执行入口补齐") from exc
                except Exception:
                    result = {"factor": name, "year": year, "error": traceback.format_exc()}
                days = active.pop((name, year))
                pending_counts[name] -= 1
                if result.get("error"):
                    failures[name] = result["error"]
                    log.error("%s/%s\n%s", name, year, result["error"])
                else:
                    old_rows = int(manifests[name].partitions.get(str(year), {}).get("rows", 0))
                    result["added_rows"] = max(0, result["partition_rows"] - old_rows)
                    commit_task(engine, by_name[name], manifests[name], days, result, end)
                progress.finish(name, year, len(days), result.get("error", "").splitlines()[-1] if result.get("error") else None,
                                rows=result.get("rows", 0), added=result.get("added_rows", 0))
        pool.shutdown(wait=True)
    except BaseException:
        processes = list((pool._processes or {}).values())
        for process in processes:
            if process.is_alive():
                process.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
        raise
    finally:
        _WORKER_ENGINE = None
    return failures, blocked


def run(args, cfg):
    specs = select_specs(cfg, args.factors, args.group)
    if not specs:
        announce("没有选中因子")
        return 0
    jobs = safe_jobs(args.jobs, cfg.raw.get("resources"))
    state_dir = cfg.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "daily.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            announce("已有计算任务运行，不能同时写入同一输出目录")
            return 2
        started = time.monotonic()
        stock = sum(not s.is_market and not s.is_label for s in specs)
        market = sum(s.is_market for s in specs)
        labels = sum(s.is_label for s in specs)
        announce(f"① {'全量重算' if args.rebuild else '增量生产'}：股票 {stock} / 市场 {market} / 标签 {labels} · 共 {len(specs)} 项")
        engine = ProductionEngine(cfg)
        source_end = latest_source_day(engine.up)
        end = min(str_to_int(args.end), source_end) if args.end else source_end
        start = str_to_int(args.start) if args.start else None
        if start and start > end:
            raise ValueError("开始日期晚于上游截止/请求截止")
        engine._run_end, engine._end_str = end, int_to_str(end)
        engine.preserve_later = bool(args.end)
        financial = [s for s in specs if financial_dependency(s, REGISTRY)]
        engine._financial_fields = (None if any(not s.fin_fields for s in financial) else
                                    frozenset(f for s in financial for f in s.fin_fields))
        starts = dependency_starts(engine, specs, start, end)
        manifests, plans, targets = {}, {}, {}
        for spec in specs:
            man = manifests[spec.name] = Manifest.load(state_dir, spec.name)
            targets[spec.name], plans[spec.name] = plan_factor(engine, spec, man, end, starts[spec.name], args.rebuild, args.refresh)
            parent_days = [np.concatenate(list(plans[d].values())) for d in spec.deps if plans.get(d)]
            if parent_days:
                first = min(int(ds.min()) for ds in parent_days)
                changed = targets[spec.name][targets[spec.name] >= first]
                for year in np.unique(changed // 10000):
                    plans[spec.name][int(year)] = np.union1d(plans[spec.name].get(int(year), []), changed[changed // 10000 == year]).astype(np.int32)
        tasks = sum(map(len, plans.values()))
        skipped = sum(not p for p in plans.values())
        announce(f"② 任务规划：上游最新 {int_to_str(source_end)} · 本次截止 {int_to_str(end)} · "
                 f"待算 {tasks} 个年分区 / 跳过 {skipped} 项 · 用时 {time.monotonic()-started:.2f}s")
        if args.dry_run:
            for spec in specs:
                plan = plans[spec.name]
                announce(f"{spec.name}: " + (", ".join(f"{y}={len(ds)}天" for y, ds in plan.items()) if plan else "skip"))
            return 0
        # Checkpoints alone provide recovery. Invalidate only the scheduled dates before writes.
        for spec in specs:
            man = manifests[spec.name]
            if plans[spec.name]:
                for days in plans[spec.name].values():
                    for a, b in engine._trading_runs(days):
                        remove_coverage(man, int_to_str(a), int_to_str(b))
                man.last_run["pending_years"] = sorted(set(man.last_run.get("pending_years", [])) | set(plans[spec.name]))
                man.save()
            elif getattr(man, "_coverage_recovered", False):
                # Persist recovered coverage only on a real run.
                man.save()
        resource_mode = "手动指定，豁免项目资源限制" if args.jobs > 0 else "自动选择，启用资源保护"
        announce(f"③ 并行生成：{jobs} 个计算进程 · {resource_mode} · 输出 {cfg.factors_dir}")
        progress = ProductionProgress(specs, targets, plans, args.progress, manifests=manifests)
        old_term = signal.getsignal(signal.SIGTERM)
        def terminate(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        try:
            failures, blocked = _execute(engine, specs, manifests, plans, end, jobs, progress,
                                         lock.fileno(), enforce_limits=args.jobs <= 0) if tasks else ({}, set())
        finally:
            progress.close()
            signal.signal(signal.SIGTERM, old_term)
        failed = len(set(failures) | blocked)
        announce(f"④ 完成：成功 {len(specs)-skipped-failed} / 跳过 {skipped} / 失败及阻塞 {failed} · 用时 {time.monotonic()-started:.2f}s")
        _atomic_json({"cutoff": int_to_str(end), "mode": "rebuild" if args.rebuild else "incremental",
                      "factors": len(specs), "skipped": skipped, "tasks": tasks,
                      "seconds": round(time.monotonic()-started, 2), "failures": failures,
                      "blocked": sorted(blocked)}, state_dir / "production_last_run.json")
        return int(bool(failed))
