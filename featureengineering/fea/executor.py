"""多路年份流执行器 —— 把「按年串行」的日更/重建改成 K 路并行，台账仍只有一个。

## 结构（照搬 `research/batch.py` 已经验证过的模式，2026-10-03）

    driver（`fea/daily.py::run`：持 state/daily.lock、规划、台账、发布）
      └─ run_bounded(main.py run --coordinate ...)      ← **单一内存护栏包住整批**
           └─ coordinator（本模块 `coordinate()`）
                ├─ 逐块 spawn：main.py run <names> --start/--end --jobs J
                │              --slice-worker --plan-file ... --results-file ... --no-state-writes
                ├─ 块完成 → 读结果文件 → `Engine._finalize` 落台账（**单写者**）
                └─ progress.json / 每块日志 / 车道自适应 / 资源采样

为什么必须 coordinator 单写者：`state/<因子>.json` 是「整体 load→改→save」，
两个年块各自 load 再整文件覆写会互相抹掉另一年的 coverage/partitions（2026-10-03
写面审计确认这是并行的头号障碍；代码注释里也记录过这类丢失更新造成的事故）。
所以：切片 worker 一律 `--no-state-writes`（只算数、只写数据分区），
台账由 coordinator 在块完成时逐个提交 —— 写者序列
`driver（规划/重置）→ coordinator（逐块 finalize）→ driver（对账/发布）` 无重叠窗口。

内存纪律（doc/因子工程.md:60GB 绝对上限）：
  · 护栏唯一权威仍是 `run_bounded`（charged 0.90×有效上限 / anon 0.75 / PSS），
    它监控的是 coordinator 的整个进程组 —— 所有年块 worker 都在组内，一停全停；
  · coordinator 自己按 2s 采样 cgroup 只用于**车道自适应**（`_next_lanes`），
    加档条件沿用清理期实测纪律：峰值 <38 GB 且 CPU ≥0.65×当前车道。
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import config as cfg_mod
from .dates import int_to_str
from .manifest import Manifest, _atomic_json

log = logging.getLogger("fea.executor")

GiB = 2 ** 30


# ══════════════════════════════════════════════════════════════════════
# cgroup 采样（v2 优先、退 v1；都没有则返回 None，不阻断运行）
# ══════════════════════════════════════════════════════════════════════

def _cgroup_files():
    """当前 cgroup 的 (用量, 上限) 文件。v1/v2 两种布局都试，都没有返回 (None, None)。"""
    v2 = (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max"))
    if v2[0].exists() and v2[1].exists():
        return v2
    v1 = (Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
          Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    if v1[0].exists() and v1[1].exists():
        return v1
    return None, None


MEM_CURRENT, MEM_MAX = _cgroup_files()


def _mem_gb() -> float:
    if MEM_CURRENT is None:
        return 0.0
    try:
        return int(MEM_CURRENT.read_text()) / GiB
    except Exception:                                        # noqa: BLE001
        return 0.0


def _mem_limit_gb() -> float:
    if MEM_MAX is None:
        return 0.0
    try:
        return int(MEM_MAX.read_text()) / GiB
    except Exception:                                        # noqa: BLE001
        return 0.0


def _anon_gb() -> float:
    """不可回收内存（cgroup v2 `memory.stat: anon`；v1 退 `total_rss`）。

    ★ 这是「能不能再开几路」的唯一靠谱依据（页缓存可回收、charged 会把缓存算进来）。
    """
    if MEM_CURRENT is None:
        return 0.0
    stat_path = (MEM_CURRENT.parent / "memory.stat"
                 if MEM_CURRENT.name == "memory.current"
                 else MEM_CURRENT.parent / "memory.stat")
    try:
        stat = dict(line.split() for line in stat_path.read_text().splitlines())
        key = "anon" if "anon" in stat else "total_rss"
        return int(stat.get(key, 0)) / GiB
    except Exception:                                        # noqa: BLE001
        return 0.0


class _MemSampler(threading.Thread):
    """跑一块期间每 2 秒采一次 cgroup 内存（charged + anon），报告峰值。

    ★ 与 `run_bounded`（驱动侧、1s、权威护栏）不同：这里**只做观测与调档依据**，
      不做击杀。击杀永远由驱动侧的 `run_bounded` 执行（单一权威）。
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0.0
        self.peak_anon = 0.0
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.peak = max(self.peak, _mem_gb())
            self.peak_anon = max(self.peak_anon, _anon_gb())
            self._stop_event.wait(2.0)

    def stop(self) -> None:
        self._stop_event.set()


# ══════════════════════════════════════════════════════════════════════
# 车道自适应（纯函数，便于表驱动单测）
# ══════════════════════════════════════════════════════════════════════

RAMP_PEAK_BYTES = 38 * GiB          # 清理期实测纪律：「+2 路需峰值 <38GB」
ANON_HIGH_RATIO = 0.90              # 不可回收内存 > 0.90×上限 → 降档
STEP = 2                            # 降档步长（-2）；加档也是 +2
DUTY_MIN = 0.65                     # CPU 实际占用达到 0.65×车道才认为「喂得饱」


def _next_lanes(peak_charged: int, cpu_seconds: float, wall_seconds: float, lanes: int,
                lo: int, hi: int, guard_bytes: int, anon_bytes: int = 0,
                anon_cap_bytes: int = 0) -> int:
    """按实测调下一块的车道数。

    控制律（与 `scripts/backfill_history._next_jobs` 同源，输入换成「车道数」，
    并加了清理期用过的加档前提）：

      · **危险**（anon > 0.90×anon上限 或 计费 > 0.97×护栏）→ 降一档（-2，下限 lo）；
      · 峰值 < 38 GiB **且** 本块 CPU 占用 ≥ 0.65×车道 → +2（上限 hi）——
        「加档要有实测依据」，喂不饱就别加；
      · 其余（含峰值未知）→ 保持不动。

    ★★ 2026-10-03 实测教训（三条，全部踩过）：
      ① 阈值原来写 0.85×护栏 → 真实全库重建的年块计费稳定在 48–53.5 GiB（护栏 54），
         被这条规则一路减半到 2 车道，后 4 个块爬行 40 分钟；
      ② 单看 charged 会误判：计费里含**可回收页缓存**（共享 cgroup 里其它会话的读
         也计进来），53.5 GiB 里 anon 只有 33/46 —— 那是**稳定运行**的水平，不是危险；
      ③ 降档用「减半」在低档位太狠（6→3→1）；改成每步 -2 并保留下限（coordinator 端 lo≥3）。

    因此判定只看 **anon（不可回收）**：> 0.90×上限 才降一档。计费护栏（charged）仍是
    `run_bounded` 的硬线（触发即 killpg，driver 会降半重试）——那条线越界的代价是
    「重跑一块」，而这里误降档的代价是「后面所有块都慢」，两害相权取其轻。

    CPU 占用 = cpu_seconds / wall_seconds（单位：核）。
    """
    if anon_bytes and anon_cap_bytes and anon_bytes > ANON_HIGH_RATIO * anon_cap_bytes:
        return max(lanes - STEP, lo)
    if peak_charged and peak_charged < RAMP_PEAK_BYTES and wall_seconds > 0:
        cores = cpu_seconds / wall_seconds
        if cores >= DUTY_MIN * lanes:
            return min(lanes + STEP, hi)
    return lanes


def _jobs_for_streams(total_jobs: int, streams: int, years: int) -> tuple[int, int]:
    """把总车道预算分成 K 条流。返回 (streams, 每流车道)。"""
    streams = max(1, min(int(streams), int(years), max(1, int(total_jobs) // 2)))
    return streams, max(1, int(total_jobs) // streams)


def _bounded_job_lanes(total_jobs: int, requested_streams: int, years: int,
                       max_jobs: int, jobs_min: int = 1) -> tuple[int,int,int,int]:
    """Resolve (streams, initial lanes, lower lanes, upper lanes) within cap."""
    cap=max(1,min(32,int(max_jobs)))
    total=max(1,min(int(total_jobs),cap))
    streams,lanes=_jobs_for_streams(total,requested_streams,years)
    hi=max(1,cap//max(1,streams))
    lo=min(max(1,int(jobs_min or 1)),hi)
    return streams,min(lanes,hi),lo,hi


# ══════════════════════════════════════════════════════════════════════
# 计划辅助（自 scripts/backfill_history.py 迁入；那里保留同名再导出）
# ══════════════════════════════════════════════════════════════════════

def _real_gap(eng, man, start: str, end: str) -> bool:
    """`missing_ranges` 是**日历日**口径：年与年之间只要隔了非交易日（元旦、春节），
    就会被判成一段"缺口" —— 实测每个因子每一年都会报 1~2 段假缺口。

    后果很严重：`_year_done` 永远返回 False ⇒ 脚本每次都把 15 年**全部重跑一遍**
    （230 个因子 × 15 年，实测单块峰值 45~58 GB）。所以这里必须用交易日历再筛一遍。

    ⚠️ 别改回 `man.missing_ranges(...)` 直接判空 —— 假缺口不会让 `main.py run` 产出
       任何东西（`Engine.plan` 里 `cal.between` 出来是空数组就跳过），但会让驱动
       反复重跑已完成的年份。
    """
    for a, b in man.missing_ranges(start, end):
        lo, hi = int(a.replace("-", "")), int(b.replace("-", ""))
        if eng.cal.between(lo, hi).size:
            return True
    return False


def _scan_thin(cfg, year: int, min_ratio: float = 0.05,
               specs: list | None = None) -> tuple[list[str], list[str]]:
    """扫 manifest 报告本年非空率过低的因子，分两档（零额外读盘）：

    · **全 NaN（0.0%）** = 一定是垃圾分区（该年整年没有值）→ 必须按
      `growth.py::DEEP_LAG_START` / `fundamental2.py::FIN_START_DEEP20` 的范式
      补一个**实测的**显式 `start=`，再单独重跑该因子。
    · **稀疏（0 < 非空率 < 5%）** = 不一定错：事件类因子（龙虎榜 3~4%、
      涨停/公告类）本来就稀疏，`main.py check` 的「口径太稀」在它们身上是**预期**。
    """
    from .spec import all_specs
    zero, thin = [], []
    for s in (specs if specs is not None else all_specs()):
        if year < int(s.resolved_start(cfg)[:4]):
            continue
        man = Manifest.load(cfg.state_dir, s.name)
        v = man.partitions.get(str(year))
        if not v or not int(v.get("rows", 0)):
            continue
        ratio = int(v.get("nonnull", 0)) / int(v["rows"])
        if ratio == 0:
            zero.append(f"{s.name}(0%)")
        elif ratio < min_ratio:
            thin.append(f"{s.name}({ratio:.1%})")
    return zero, thin


def _year_done(eng, cfg, year: int, specs: list | None = None) -> bool:
    """本年是否已经**全部**算过（决定能否跳过）。

    ⚠️ 本年的右端要**截到上游基准表的最后一天**：日历会延伸到未来，不截的话
       每个因子在最后一年都恒报"缺一段未来的日子"。
    """
    from .spec import all_specs
    y1 = min(f"{year}-12-31", int_to_str(eng.baseline_last_day()))
    y0 = f"{year}-01-01"
    for s in (specs if specs is not None else all_specs()):
        # ★ 要求的**分区下界**是 `align_start`（= default_start），不是真实起点：
        #   对齐区间（`resolved_start` 之前的年份）也必须有分区，否则晚起点因子
        #   会被当成"已覆盖"而永远不补对齐分区。见 FactorSpec.align_fill。
        if not s.enabled or s.align_start(cfg) > y1:
            continue
        start = max(y0, s.align_start(cfg))
        man = Manifest.load(cfg.state_dir, s.name)
        if man.recipe != eng._recipe(s) or _real_gap(eng, man, start, y1):
            return False
        partition = cfg.factor_root(s) / s.name / f"year={year}" / "data.parquet"
        if not partition.exists():
            return False
        expected = man.partitions.get(str(year), {}).get("file_identity")
        if expected:
            st = partition.stat()
            if expected != [st.st_size, st.st_mtime_ns]:
                return False
    return True


def _record_rebuild_attempt(cfg, specs, first_year, last_year):
    """记录强制重建的区间与**执行前**的 coverage 快照（审计证据；只写证据，不改台账）。

    ★★ 2026-10-06（事故复盘）：本函数原为 `_invalidate_coverage`，除落证据外还会把
    区间内 coverage **清空**（防 `--resume` 把"未刷新完成"的年当作已完成）。该清空在
    被中断时留下"账实背离"：2026-10-05 全库 force 重建被中断后，180 个因子的台账停在
    2018–2021，之后每次日常运行都被迫重排 2022–2026（CPU 空烧、反复触发内存护栏降档）。
    "必须重算"的语义改由子进程 `--refresh` 承担（`plan()` 的 force_refresh 分支本就不读
    coverage）；台账保持完好，中断后可直接按台账增量续跑。
    证据快照仍是 `scripts/restore_coverage_from_attempt.py` 的输入。
    """
    import datetime
    lo, hi = f"{first_year}-01-01", f"{last_year}-12-31"
    saved = {}
    selected = specs if specs is not None else __import__(
        "fea.spec", fromlist=["all_specs"]).all_specs()
    for spec in selected:
        if not spec.enabled:
            continue
        man = Manifest.load(cfg.state_dir, spec.name)
        saved[spec.name] = man.coverage
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    _atomic_json({"interval": [lo, hi], "coverage_before": saved,
                  "note": "evidence only; manifests not modified (2026-10-06)"},
                 cfg.state_dir / "rebuild_attempts" / f"{stamp}.json")


def _next_jobs(peak_charged: int, budget: int, jobs: int, lo: int, hi: int) -> int:
    """自适应调档（旧控制律，保留给 `scripts/backfill_history` 的调用方与测试）。

      · 峰值 < 0.50×预算 → 加两档（+2），上限 hi；
      · 峰值 > 0.85×预算 → 减半，下限 lo；
      · 其间 → 保持；峰值未知（<=0）→ 不动（不知道就别乱动档位）。
    """
    if not budget or not peak_charged or peak_charged <= 0:
        return jobs
    ratio = peak_charged / budget
    if ratio < 0.50:
        return min(jobs + 2, hi)
    if ratio > 0.85:
        return max(jobs // 2, lo)
    return jobs


# ══════════════════════════════════════════════════════════════════════
# CPU 归因（按进程树，采样时归属到具体年块）
# ══════════════════════════════════════════════════════════════════════

_HZ = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _read_proc_cpu() -> dict[int, tuple[int, int, float]]:
    """一次 /proc 扫描：{pid: (ppid, ticks, cpu_seconds_cumulative)}。

    CPU = utime+stime+cutime+cstime（含**已回收子进程**的累计），
    这样 fork 出来的因子 worker 退出后其 CPU 仍会被父进程的 cutime/cstime 记着。
    """
    out: dict[int, tuple[int, int, float]] = {}
    proc = Path("/proc")
    if not proc.is_dir():
        return out
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            raw = (p / "stat").read_text()
            # comm 里可能有空格/括号 → 从最后一个 ')' 之后切
            rp = raw.rindex(")")
            head, tail = raw[:rp], raw[rp + 2:].split()
            pid = int(head.split()[0])
            ppid = int(tail[1])
            utime, stime, cutime, cstime = (int(tail[11]), int(tail[12]),
                                            int(tail[13]), int(tail[14]))
            out[pid] = (ppid, utime + stime, (utime + stime + cutime + cstime) / _HZ)
        except (OSError, ValueError, IndexError):
            continue
    return out


def _tree_cpu(proc_map: dict, root_pid: int) -> float:
    """root_pid 及其全部后代（含已回收子进程）的累计 CPU 秒。"""
    children: dict[int, list[int]] = {}
    for pid, (ppid, _t, _c) in proc_map.items():
        children.setdefault(ppid, []).append(pid)
    total = 0.0
    stack = [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        rec = proc_map.get(pid)
        if rec is not None:
            total += rec[2]
        stack.extend(children.get(pid, ()))
    return total


# ══════════════════════════════════════════════════════════════════════
# coordinator：逐块调度 + 单写者 finalize
# ══════════════════════════════════════════════════════════════════════

def _load_plan(run_dir: Path) -> dict[str, dict[str, list[int]]]:
    return json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))


_SPEC_CACHE: dict = {}


def spec_map(cfg) -> dict:
    """因子名 → FactorSpec（**正式清单口径**；沙箱退回全注册表）。

    ★★ 2026-10-03 实测踩过：正式对象里有 4 个是**研究来源**（`s26_*` / `r2own_*` /
      `r3b_*`），它们由 `production_specs` 在加载正式库时经
      `research.catalog.spec_for_candidate` 注册进 REGISTRY —— coordinator 只
      `import factors` 是看不到的。曾把 179 个任务里的 4 个判成「未注册」：
      它们的台账永远提交不上 ⇒ 收尾对账判缺口 ⇒ **整批发布被挡**。
    """
    cached = _SPEC_CACHE.get("map")
    if cached is not None:
        return cached
    from .spec import REGISTRY
    if not cfg.raw.get("_sandbox"):
        try:
            from .release import production_specs
            out = {s.name: s for s in production_specs(cfg)}
            _SPEC_CACHE["map"] = out
            return out
        except Exception:                                     # noqa: BLE001
            log.exception("正式清单解析失败，退回全注册表（可能少研究来源对象）")
    _SPEC_CACHE["map"] = dict(REGISTRY)
    return _SPEC_CACHE["map"]


def _by_year(plan: dict[str, dict[str, list[int]]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for name, years in plan.items():
        for y in years:
            out.setdefault(int(y), []).append(name)
    return out


def _year_cost(plan, names: list[str], year: int) -> int:
    """该年块的粗略成本 = 计划天数合计（用于长块优先的派发顺序）。"""
    return sum(len(plan[n].get(str(year), ())) for n in names)


def _finalize_chunk(cfg, eng, run_dir: Path, names: list[str], rc: int,
                    result_file: str | None = None) -> dict:
    """把某个年块（或耦合尾批）的结果提交进 `state/<因子>.json`（**coordinator 是唯一写者**）。

    结果文件由切片 worker 原子写入（`--results-file`）。即便整块退出码非零，
    只要某个因子的结果齐全，也照常提交 —— coverage 必须按**实际产出**记，
    这样中断重跑时已完成的因子不会被无谓重算（与 `_finalize` 的产出守卫同源）。

    年份由**结果自身的 `year` 字段**给出（尾批可能一个因子跨多年），再与 plan.json
    对账（`plan_days` 必须非空），防止「结果与计划脱节」被静默记成完成。

    返回 {finalized, failed, tasks, task_seconds, years}。
    """
    specs_by_name = spec_map(cfg)
    path = run_dir / (result_file or "results.json")
    data = None
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:                              # noqa: BLE001
            log.warning("结果文件读不出来 %s：%s", path, exc)
    factors = (data or {}).get("factors", {})
    finalized, failed = [], []
    finalized_years_by_factor = {}
    tasks = 0
    task_seconds = 0.0
    years: set[int] = set()
    for name in names:
        rec = factors.get(name)
        results = [r for r in (rec or {}).get("results", []) if isinstance(r, dict)]
        error = (rec or {}).get("error")
        if error or not results:
            failed.append(f"{name}({error or '无结果'})")
            continue
        spec = specs_by_name.get(name)
        if spec is None:
            failed.append(f"{name}(未注册)")
            continue
        by_year: dict[int, list[dict]] = {}
        for r in results:
            by_year.setdefault(int(r["year"]), []).append(r)
        plan_slice = {}
        bad = []
        for y in sorted(by_year):
            ds = np.asarray(plan_days(run_dir, name, y), dtype=np.int32)
            if ds.size == 0:
                bad.append(y)
            else:
                plan_slice[y] = ds
        if bad:
            failed.append(f"{name}(结果年份不在计划里:{bad})")
            continue
        man = Manifest.load(cfg.state_dir, name)
        # ★ 自愈（2026-10-03 净室测试）：
        #   · recipe **为空** = 这份 manifest 是全新/被清空的，而本块的数据刚刚由本进程
        #     用当前 recipe 算出来 ⇒ 直接补记身份，不重置（重置会把刚算的覆盖清掉）；
        #   · recipe **过期**（非空且不等于当前）⇒ 拒绝把两套口径并进同一条 coverage，
        #     按新 recipe 重置后提交。
        cur = eng._recipe(spec)
        if man.recipe != cur:
            if man.recipe:
                log.warning("%s 台账 recipe 过期（driver 未落盘？）→ 按新 recipe 重置后提交", name)
                man.reset(cur)
            else:
                man.recipe = cur
        try:
            eng._finalize(spec, man, plan_slice, results,
                          sum(float(r.get("seconds") or 0.0) for r in results))
        except Exception as exc:                              # noqa: BLE001
            log.exception("提交台账失败 %s", name)
            failed.append(f"{name}({exc})")
            continue
        finalized.append(name)
        finalized_years_by_factor[name] = sorted(int(y) for y in by_year)
        tasks += len(results)
        task_seconds += sum(float(r.get("seconds") or 0.0) for r in results)
        years |= set(by_year)
    return {"finalized": finalized, "finalized_years_by_factor": finalized_years_by_factor,
            "failed": failed, "tasks": tasks, "task_seconds": round(task_seconds, 2),
            "rc": int(rc), "years": sorted(years)}


_PLAN_CACHE: dict = {}


def plan_days(run_dir: Path, name: str, year: int) -> list[int]:
    """取该因子该年的计划交易日（带缓存；plan.json 在一次运行内不变）。"""
    plan = _PLAN_CACHE.get("plan")
    if plan is None or _PLAN_CACHE.get("run_dir") != str(run_dir):
        plan = _load_plan(run_dir)
        _PLAN_CACHE.update({"plan": plan, "run_dir": str(run_dir)})
    return plan.get(name, {}).get(str(year), [])


def coordinate(cfg, args) -> int:
    """coordinator 进程入口（`main.py run --coordinate --run-dir DIR ...`）。

    单线程轮询调度（不用线程池：coordinator 只做 Popen+轮询+落账，
    多进程并行由年块子进程完成；单线程让 state/ 的写者天然唯一）。

    ★ 耦合因子尾批：`deps` 里含**别的因子**的因子（当前只有 1 个：
      `cp_momentum_lowvol_20 ← idio_vol_60`）要读父因子的**上一年**分区做预热。
      多路年份流下父因子所在年可能正在被另一条流重写 ⇒ 全部普通年块跑完后，
      再单独跑一批耦合因子（父因子已全部就位），值才与串行一致。
    """
    run_dir = Path(args.run_dir).resolve()
    # ★★ 归属自检（2026-10-03 实测踩过）：coordinator 的配置必须与 driver 的一致。
    #   若 `--sandbox` 在命令行上丢了，coordinator 会加载生产配置，于是
    #   `_finalize` 把台账写进**生产** state/ —— 而 driver 以为在沙箱里。
    #   run_dir 恒在 `<state_dir>/runs/` 下，跨配置必然对不上，这里直接拒跑。
    runs_root = (Path(cfg.state_dir).resolve() / "runs")
    if not run_dir.is_relative_to(runs_root):
        raise SystemExit(f"coordinator 配置与 run-dir 不一致：{run_dir} 不在 {runs_root} 下"
                         "（--sandbox 是否漏传？）")
    eng = _ensure_engine(cfg)          # 只做 finalize 的轻量引擎（不建层、不算因子）
    plan = _load_plan(run_dir)
    if not plan:
        print("计划为空，无需计算。", flush=True)
        return 0
    specs_by_name = spec_map(cfg)
    coupling = sorted(n for n in plan if n in specs_by_name
                      and any(d in specs_by_name for d in specs_by_name[n].deps))
    plain = {n: ys for n, ys in plan.items() if n not in coupling}
    by_year = _by_year(plain)
    cutoff_str = args.cutoff or int_to_str(_baseline_cutoff(cfg))
    years = sorted(by_year)
    # 长块优先：任务天数多的年先派发（削尾，和波次内 LPT 同一目的）
    order = sorted(years, key=lambda y: -_year_cost(plan, by_year[y], y))
    sweep_years = sorted({int(y) for n in coupling for y in plan[n]})

    chunk_timeout = int(getattr(args, "chunk_timeout", 0) or 0)
    from .resources import available_cpus, resource_policy
    policy=resource_policy(cfg.raw.get("resources"))
    cap=min(int(policy["max_jobs"]),int(available_cpus()),32)
    requested_cap=int(getattr(args,"jobs_max",0) or 0)
    if requested_cap>0:
        cap=min(cap,requested_cap)
    streams,jps,jobs_lo,jobs_hi=_bounded_job_lanes(
        args.jobs,args.streams,len(order) or 1,cap,getattr(args,"jobs_min",1))
    from .resources import memory_limits
    _limits = memory_limits(policy=cfg.raw.get("resources"),
                            container_limit=_container_limit())
    budget = int(getattr(args, "budget_bytes", 0) or 0) or int(_limits["charged"])
    anon_cap = int(_limits["anon"])

    sampler = _MemSampler()
    sampler.start()

    sandbox = cfg.raw.get("_sandbox")
    py = sys.executable
    main_py = str(Path(__file__).resolve().parents[1] / "main.py")
    stopping = {"flag": False}

    def _stop(signum, frame):                                  # noqa: ARG001
        stopping["flag"] = True

    old_handlers = {s: signal.signal(s, _stop) for s in (signal.SIGTERM, signal.SIGINT)}

    pending = list(order)
    active: dict[str, dict] = {}          # key: 年块显示名（"2019" / "耦合尾批"）
    completed: list[dict] = []
    samples: dict[str, float] = {}
    last_cpu: dict[str, float] = {}
    total_tasks = 0
    total_task_seconds = 0.0
    rc_all = 0
    first_dispatch: float | None = None
    last_completion: float | None = None

    def _write_progress() -> None:
        _atomic_json({"completed": completed, "streams": streams,
                      "jobs_per_stream": jps, "pending": pending,
                      "active": sorted(active),
                      "first_dispatch": first_dispatch, "last_completion": last_completion},
                     run_dir / "progress.json")

    def _dispatch(key: str, names: list[str], days_lo: str, days_hi: str,
                  result_file: str, label: str) -> None:
        nonlocal first_dispatch
        log_path = run_dir / f"{label}.log"
        stream = log_path.open("a" if log_path.exists() else "w", encoding="utf-8")
        cmd = [py, main_py, "run", *names,
               "--start", days_lo, "--end", days_hi,
               "--jobs", str(jps), "--slice-worker",
               "--plan-file", str(run_dir / "plan.json"),
               "--results-file", str(run_dir / result_file),
               "--no-state-writes"]
        if sandbox:
            cmd += ["--sandbox", str(cfg.root)]
        print(f"[启动] {label} · {len(names)} 项 · jobs={jps}", flush=True)
        env = dict(os.environ)
        env["FEA_LOG_NAME"] = f"factors.{label}"      # 日志分文件，避免 FUSE append 交错
        proc = subprocess.Popen(cmd, cwd=str(cfg.contract_root), stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=False, env=env)
        first_dispatch = first_dispatch or time.time()
        active[key] = {"proc": proc, "stream": stream, "t0": time.time(), "names": names,
                       "label": label, "result_file": result_file,
                       "peak": _mem_gb(), "peak_anon": _anon_gb()}   # ★ 块级初值按派发时取

    def _finish_block(key: str, rec: dict, rc: int) -> None:
        nonlocal last_completion, total_tasks, total_task_seconds, jps, rc_all
        _reap(rec)
        elapsed = time.time() - rec["t0"]
        last_completion = time.time()
        cpu_s = samples.pop(key, 0.0)
        last_cpu.pop(key, None)
        record = _finalize_chunk(cfg, eng, run_dir, rec["names"], rc, rec["result_file"])
        total_tasks += record["tasks"]
        total_task_seconds += record["task_seconds"]
        peak_bytes = int(rec["peak"] * GiB)
        chunk = {"year": key, "returncode": int(rc), "seconds": round(elapsed, 2),
                 "cpu_seconds": round(cpu_s, 1),
                 "peak_cgroup_bytes": peak_bytes,
                 "peak_anon_bytes": int(rec["peak_anon"] * GiB),
                 "peak_shared_anon_bytes": int(rec["peak_anon"] * GiB),
                 "peak_rss_bytes": 0,
                 "memory_stopped": False, "stop_reason": "",
                 "tasks": record["tasks"], "task_seconds": record["task_seconds"],
                 "finalized": len(record["finalized"]),
                 "finalized_factor_names": list(record["finalized"]),
                 "finalized_years_by_factor": record["finalized_years_by_factor"],
                 "failed": record["failed"][:8]}
        completed.append(chunk)
        for y in (record["years"] or [int(rec["label"][:4]) if rec["label"][:4].isdigit() else 0]):
            if not y:
                continue
            specs = [specs_by_name[n] for n in rec["names"] if n in specs_by_name]
            zero, thin = _scan_thin(cfg, y, specs=specs)
            if zero:
                print(f"    ✘ {y} 年有 {len(zero)} 个因子整年全 NaN（垃圾分区，"
                      f"按 DEEP_LAG_START 范式补显式 start 后单独重跑）："
                      f"{' '.join(zero[:12])}", flush=True)
            if thin:
                print(f"    ⚠ {y} 年有 {len(thin)} 个因子非空率 <5%（事件类属预期）："
                      f"{' '.join(thin[:12])}", flush=True)
        print(f"[完成] {rec['label']} · rc={rc} · {elapsed/60:.1f} min · "
              f"CPU {cpu_s:.0f}s（{cpu_s/max(elapsed,1e-9):.1f} 核）· "
              f"峰值 {peak_bytes/GiB:.1f} GiB · 提交 {record['finalized']} 项"
              + (f" · 失败 {len(record['failed'])}: {record['failed'][:3]}" if record["failed"] else ""),
              flush=True)
        if rc != 0:
            rc_all = rc_all or rc
        nxt = _next_lanes(peak_bytes, cpu_s, elapsed, jps, jobs_lo, jobs_hi, budget,
                          anon_bytes=int(rec["peak_anon"] * GiB), anon_cap_bytes=anon_cap)
        if nxt != jps:
            print(f"    → 车道自适应：{jps} → {nxt}"
                  f"（峰值 {peak_bytes/GiB:.1f} GiB · anon {rec['peak_anon']:.1f} GiB · "
                  f"CPU {cpu_s/max(elapsed,1e-9):.1f}/{jps} 核）", flush=True)
            jps = nxt
        del active[key]
        _write_progress()

    print(f"年块调度：{len(order)} 普通块 + {1 if coupling else 0} 耦合尾批 · "
          f"{streams} 路 × {jps} 车道 · 内存预算 {budget/GiB:.1f} GiB · 块序 {order}", flush=True)
    try:
        while pending or active:
            if stopping["flag"]:
                print("收到中断信号：停止派发并终止在跑的块。", flush=True)
                for rec in active.values():
                    _kill_tree(rec["proc"])
                for rec in active.values():
                    _reap(rec)
                active.clear()
                rc_all = rc_all or 1
                break
            while pending and len(active) < streams:
                year = pending.pop(0)
                _dispatch(str(year), by_year[year], f"{year}-01-01",
                          min(f"{year}-12-31", cutoff_str),
                          f"results.{year}.json", str(year))
            # ---- 采样 + 完成轮询 ----
            proc_map = _read_proc_cpu()
            for key, rec in list(active.items()):
                cpu = _tree_cpu(proc_map, rec["proc"].pid)
                samples[key] = samples.get(key, 0.0) + max(0.0, cpu - last_cpu.get(key, cpu))
                last_cpu[key] = cpu
                rec["peak"] = max(rec["peak"], _mem_gb())
                rec["peak_anon"] = max(rec["peak_anon"], _anon_gb())
                rc = rec["proc"].poll()
                if rc is None:
                    # ★ 单块超时：卡死的块会拖住整条流；给它一个宽上限（默认不限）。
                    if chunk_timeout and time.time() - rec["t0"] > chunk_timeout:
                        print(f"[超时] {rec['label']} 超过 {chunk_timeout}s，终止该块。", flush=True)
                        _kill_tree(rec["proc"])
                        rc = 1
                    else:
                        continue
                _finish_block(key, rec, rc)
            time.sleep(0.25)
        # ---- 耦合尾批：所有普通年块的父因子都已落盘，再算耦合因子 ----
        if coupling and not stopping["flag"]:
            _dispatch("耦合尾批", coupling, f"{sweep_years[0]}-01-01", cutoff_str,
                      "results.sweep.json", "coupling_sweep")
            key = "耦合尾批"
            while key in active:
                if stopping["flag"]:
                    _kill_tree(active[key]["proc"])
                    _reap(active[key])
                    active.clear()
                    rc_all = rc_all or 1
                    break
                proc_map = _read_proc_cpu()
                rec = active[key]
                cpu = _tree_cpu(proc_map, rec["proc"].pid)
                samples[key] = samples.get(key, 0.0) + max(0.0, cpu - last_cpu.get(key, cpu))
                last_cpu[key] = cpu
                rec["peak"] = max(rec["peak"], _mem_gb())
                rec["peak_anon"] = max(rec["peak_anon"], _anon_gb())
                rc = rec["proc"].poll()
                if rc is not None:
                    _finish_block(key, rec, rc)
                    break
                time.sleep(0.25)
    finally:
        sampler.stop()
        sampler.join(timeout=5)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        _write_progress()

    # 汇总（供 driver 的资源台账）
    _atomic_json({"completed": completed, "streams": streams, "jobs_per_stream": jps,
                  "jobs_final": jps, "tasks": total_tasks,
                  "task_seconds": round(total_task_seconds, 1),
                  "first_dispatch": first_dispatch, "last_completion": last_completion,
                  "peak_anon_gib": round(sampler.peak_anon, 2)},
                 run_dir / "progress.json")
    print(f"全部块结束：{len(completed)} 块 · {total_tasks} 任务 · "
          f"{total_task_seconds/60:.0f} 任务分钟 · 峰值 charged {sampler.peak:.1f} GiB / "
          f"anon {sampler.peak_anon:.1f} GiB", flush=True)
    return int(bool(rc_all))


def _descendants(root_pid: int) -> list[int]:
    """root_pid 的全部后代（含多级），用于整树击杀。"""
    proc_map = _read_proc_cpu()
    children: dict[int, list[int]] = {}
    for pid, (ppid, _t, _c) in proc_map.items():
        children.setdefault(ppid, []).append(pid)
    out: list[int] = []
    stack = list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        if pid in out:
            continue
        out.append(pid)
        stack.extend(children.get(pid, []))
    return out


def _kill_tree(proc) -> None:
    """终止**整棵进程树**：先 SIGTERM，宽限后 SIGKILL。

    ★ 只 terminate 块进程本身会把它的 ProcessPoolExecutor 子进程（因子 worker）
      留成孤儿继续写 `data/` —— 孤儿与下一次运行并发写同一条路径（写面审计 P1-5）。
      块进程与它的 pool 子进程同属 coordinator 的进程组，不能用 killpg（会连自己一起杀），
      所以按 /proc 的 ppid 树逐个杀。
    """
    if proc.poll() is not None:
        return
    kids = _descendants(proc.pid)
    for pid in kids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    try:
        proc.terminate()
    except OSError:
        pass
    deadline = time.time() + 8
    while time.time() < deadline and proc.poll() is None:
        time.sleep(0.2)
    if proc.poll() is None:
        for pid in kids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            proc.kill()
        except OSError:
            pass


def _reap(rec) -> None:
    try:
        rec["proc"].wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            rec["proc"].kill()
            rec["proc"].wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        rec["stream"].close()
    except Exception:                                          # noqa: BLE001
        pass


def _container_limit():
    if MEM_MAX is None:
        return None
    lim = int(MEM_MAX.read_text()) if MEM_MAX.exists() else None
    if lim is None or lim > (1 << 62):
        return None
    return lim


def _baseline_cutoff(cfg) -> int:
    from .engine import Engine
    return Engine(cfg).baseline_last_day()


# coordinator 里的轻量 Engine：只做 `_finalize`（不建层、不跑因子）。
_ENGINE = None


def set_engine(eng) -> None:
    """给 coordinator 注入做 finalize 的引擎（测试用；生产走懒建）。"""
    global _ENGINE
    _ENGINE = eng


def _ensure_engine(cfg):
    global _ENGINE
    if _ENGINE is None:
        from .engine import Engine
        _ENGINE = Engine(cfg)
    return _ENGINE


# ══════════════════════════════════════════════════════════════════════
# driver 侧：一次 run_bounded 包住整批
# ══════════════════════════════════════════════════════════════════════

def run_batch(cfg, run_dir: Path, *, streams: int, jobs: int, cutoff: int,
              coordinator_log: Path | None = None, timeout: int | None = None) -> dict:
    """driver 调用：以**单一护栏**跑完所有年份块。

    返回 run_bounded 的结果字典（含 returncode / memory_stopped / 峰值 / cpu_seconds），
    外加 `chunks`（从 progress.json 读回）与 `actual_streams/jobs`。
    """
    from .resources import run_bounded, safe_jobs
    total_jobs = safe_jobs(jobs, cfg.raw.get("resources"))
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "main.py"),
           "run", "--coordinate",
           "--run-dir", str(run_dir),
           "--streams", str(int(streams)),
           "--jobs", str(int(total_jobs)),
           "--cutoff", int_to_str(int(cutoff))]
    if timeout:
        cmd += ["--chunk-timeout", str(int(timeout))]
    # ★★ 2026-10-03 实测踩过：漏传 --sandbox 时 coordinator 会加载**生产配置**，
    #   `_finalize` 就写到生产 state/-，而 driver 还以为在沙箱里（值相同但台账被改）。
    #   sandbox 标记必须原样传下去；coordinate() 里另有一道 run-dir 归属自检兜底。
    if cfg.raw.get("_sandbox"):
        cmd += ["--sandbox", str(cfg.root)]
    log_path = coordinator_log or (run_dir / "coordinator.log")
    with log_path.open("a", encoding="utf-8") as stream:
        result = run_bounded(cmd, cwd=cfg.contract_root, stdout=stream, policy=cfg)
    progress = {}
    p = run_dir / "progress.json"
    if p.exists():
        try:
            progress = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            progress = {}
    result = dict(result)
    result["chunks"] = progress.get("completed", [])
    result["streams"] = progress.get("streams", int(streams))
    result["jobs_per_stream"] = progress.get("jobs_per_stream", 0)
    result["first_dispatch"] = progress.get("first_dispatch")
    result["last_completion"] = progress.get("last_completion")
    result["peak_anon_gib"] = progress.get("peak_anon_gib", 0.0)
    return result


# ══════════════════════════════════════════════════════════════════════
# 重建驱动（main.py rebuild 与 scripts/backfill_history.py 共用同一条路径）
# ══════════════════════════════════════════════════════════════════════

def rebuild_plan(cfg, specs, y_from: int, y_to: int) -> list[tuple[int, bool]]:
    """逐年在 `_year_done` 上打计划：[(年, 是否已完成)]。"""
    from .engine import Engine
    eng = Engine(cfg)
    out = []
    for y in range(y_from, y_to + 1):
        out.append((y, _year_done(eng, cfg, y, specs)))
    return out


def run_rebuild(cfg, *, specs, y_from: int, y_to: int, force: bool = True,
                jobs: int = 0, streams: int = 0, budget_bytes: int = 0,
                jobs_min: int = 1, jobs_max: int = 0,
                dry_run: bool = False) -> int:
    """全量重建：一次 `main.py run --start/--end` 走多路年份流执行器。

    `force=True` 时：先落 `state/rebuild_attempts/*.json` 的 coverage_before 审计证据
    （`_record_rebuild_attempt`），子进程带 `--refresh`（区间内全量重算的强制语义）；
    **不再预先清空 manifest coverage** —— 2026-10-05 的教训：清空被中断后会把
    "已算好但未回补"的年份留成缺口，之后的任何日常运行被迫全量重排。
    `force=False`（--resume）只补缺口。

    与旧 backfill 的差别：**不再逐块串行、每块一遍完整日更台账** ——
    一次规划、一次台账、一块年一个进程（内存隔离不变），块间按 `--streams` 并行。
    """
    plan = rebuild_plan(cfg, specs, y_from, y_to)
    for y, done in plan:
        print(f"  {y}: {'已覆盖，跳过' if done and not force else '待补'}")
    todo = [y for y, done in plan if force or not done]
    if dry_run:
        print(f"共 {len(todo)} 块待跑：{todo}")
        return 0
    if not todo:
        print("没有需要补的年份。")
        return 0
    if force:
        _record_rebuild_attempt(cfg, specs, y_from, y_to)
    end = min(f"{y_to}-12-31", int_to_str(_baseline_cutoff(cfg)))
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "main.py"),
           "run", "--start", f"{y_from}-01-01", "--end", end,
           "--jobs", str(int(jobs)), "--streams", str(int(streams)),
           "--run-mode", "rebuild"]      # 只影响资源台账的 mode 标注，不改任何计算语义
    if force:
        # ★★ 2026-10-06：force 的"区间内必须重算"由子进程的 `--refresh` 承担，
        #   替代原来的"先清空 coverage"（清空 + 中断 = 账实背离，见函数 docstring）。
        cmd.append("--refresh")
    if budget_bytes:
        cmd += ["--budget-bytes", str(int(budget_bytes))]
    if jobs_min and int(jobs_min) != 1:
        cmd += ["--jobs-min", str(int(jobs_min))]
    if jobs_max:
        cmd += ["--jobs-max", str(int(jobs_max))]
    # 只在「不是全库」时显式列出因子名（全库时省略 = 与生产清单一致，也让发布条件成立）。
    from .spec import all_specs
    all_enabled = {s.name for s in all_specs() if s.enabled}
    if {s.name for s in specs} != all_enabled:
        cmd += [s.name for s in specs]
    if cfg.raw.get("_sandbox"):
        cmd += ["--sandbox", str(cfg.root)]
    print(f"重建 {y_from}..{y_to} · 待跑 {len(todo)} 年 · jobs={jobs} · streams={streams or 'auto'}"
          f" · {' '.join(cmd[1:6])} …", flush=True)
    return subprocess.call(cmd, cwd=str(cfg.contract_root))


# ══════════════════════════════════════════════════════════════════════
# 资源台账
# ══════════════════════════════════════════════════════════════════════

LEDGER = "resource_ledger.jsonl"


def build_resources_record(cfg, run_dir: Path, *, mode: str, cutoff: int,
                           wall: float, batch: dict, phases: dict | None = None,
                           extra: dict | None = None) -> dict:
    """一次运行的资源账（写 runs/<stamp>/resources.json 与滚动 JSONL）。

    CPU 利用率口径 = cpu_seconds / (wall × available_cpus)，如实记录、不做粉饰；
    「计算阶段」的墙钟只看年块执行（phases.tasks），整轮含计划/台账/发布。
    """
    from .resources import available_cpus, memory_limits
    cpus = available_cpus()
    cpu_seconds = float(batch.get("cpu_seconds") or 0.0)
    chunks = batch.get("chunks", [])
    task_seconds = sum(float(c.get("task_seconds") or 0.0) for c in chunks)
    tasks = sum(int(c.get("tasks") or 0) for c in chunks)
    # ★ 计算阶段墙钟 = **makespan**（首批派发 → 末批完成）。多流并行时各块墙钟重叠，
    #   用 max(块墙钟) 会把利用率系统性低估（2 流时约 2×），而它就是用户的验收指标。
    fd, lc = batch.get("first_dispatch"), batch.get("last_completion")
    compute_wall = (max(0.0, float(lc) - float(fd)) if fd and lc
                    else max((float(c.get("seconds") or 0.0) for c in chunks), default=0.0))
    anon_cap = int(memory_limits(policy=cfg.raw.get("resources"),
                                 container_limit=batch.get("cgroup_limit_bytes") or None)["anon"])
    target = float((cfg.raw.get("resources") or {}).get("target_cpu_util", 0.5) or 0.5)
    rec = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "mode": mode,
        "cutoff": int_to_str(int(cutoff)),
        "wall_seconds": round(wall, 1),
        "cpus": cpus,
        "cpu_seconds": round(cpu_seconds, 1),
        "cpu_utilization": round(cpu_seconds / (wall * cpus), 4) if wall > 0 else None,
        "target_cpu_util": target,
        "peak": {"charged_bytes": int(batch.get("peak_cgroup_bytes") or 0),
                 # 权威 anon 来自 run_bounded 的 1s 采样；coordinator 的 2s 采样兜底
                 "anon_bytes": int(batch.get("peak_shared_anon_bytes")
                                   or (batch.get("peak_anon_gib") or 0) * GiB),
                 "pss_bytes": int(batch.get("peak_pss_bytes") or 0),
                 "rss_bytes": int(batch.get("peak_rss_bytes") or 0),
                 "guard_bytes": int(batch.get("memory_guard_bytes") or 0),
                 "anon_cap_bytes": anon_cap,
                 "container_limit_bytes": int(batch.get("cgroup_limit_bytes") or 0)},
        "memory_stopped": bool(batch.get("memory_stopped")),
        "stop_reason": batch.get("stop_reason", ""),
        "streams": int(batch.get("streams") or 1),
        "jobs_per_stream": int(batch.get("jobs_per_stream") or 0),
        "tasks": tasks,
        "task_seconds": round(task_seconds, 1),
        "compute_wall_seconds": round(compute_wall, 1),
        "compute_cpu_utilization": (round(cpu_seconds / (compute_wall * cpus), 4)
                                    if compute_wall > 0 else None),
        "phases": phases or {},
        "chunks": chunks,
    }
    if extra:
        rec.update(extra)
    return rec


def write_ledger(cfg, run_dir: Path, record: dict) -> None:
    _atomic_json(record, run_dir / "resources.json")
    try:
        ledger = cfg.state_dir / LEDGER
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as out:
            out.write(json.dumps(_ledger_row(record), ensure_ascii=False) + "\n")
    except Exception:                                          # noqa: BLE001
        log.exception("资源台账写入失败（不阻断运行）")


def _ledger_row(rec: dict) -> dict:
    """JSONL 只留一行摘要（明细在 runs/<stamp>/resources.json）。"""
    return {k: rec.get(k) for k in (
        "schema_version", "run_id", "mode", "cutoff", "wall_seconds", "cpu_seconds",
        "cpu_utilization", "compute_wall_seconds", "compute_cpu_utilization",
        "streams", "jobs_per_stream", "tasks", "task_seconds", "memory_stopped")} | {
        "peak_charged_bytes": (rec.get("peak") or {}).get("charged_bytes", 0),
        "peak_guard_bytes": (rec.get("peak") or {}).get("guard_bytes", 0),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def print_summary(rec: dict) -> None:
    peak = rec.get("peak") or {}
    util = rec.get("cpu_utilization")
    cutil = rec.get("compute_cpu_utilization")
    target = rec.get("target_cpu_util")
    hit = (cutil is not None and target and cutil >= target)
    lines = [
        "─" * 72,
        f"资源账：{rec.get('mode')} · 截止 {rec.get('cutoff')} · outcome {rec.get('outcome','?')}"
        f" · 墙钟 {rec.get('wall_seconds')}s",
        f"  CPU {rec.get('cpu_seconds')}s / {rec.get('cpus')} 核 → 整轮利用率 "
        f"{(util * 100 if util is not None else float('nan')):.1f}%"
        + (f" · 计算阶段 {cutil * 100:.1f}%（目标 {target * 100:.0f}%"
           + ("，达标 ✓" if hit else "，未达标 ✗") + "）" if cutil is not None and target else ""),
        f"  内存峰值 charged {peak.get('charged_bytes', 0)/GiB:.1f} GiB"
        f"（护栏 {peak.get('guard_bytes', 0)/GiB:.1f} / anon 上限 "
        f"{peak.get('anon_cap_bytes', 0)/GiB:.1f}）"
        f" · anon {peak.get('anon_bytes', 0)/GiB:.1f} GiB",
        f"  任务 {rec.get('tasks')} 个 · {rec.get('task_seconds', 0)/60:.0f} 任务分钟 · "
        f"{rec.get('streams')} 流 × {rec.get('jobs_per_stream')} 车道",
    ]
    chunks = rec.get("chunks") or []
    slow = sorted(chunks, key=lambda c: -(c.get("seconds") or 0))[:5]
    if slow:
        lines.append("  最慢块：" + " · ".join(
            f"{c.get('year')}({c.get('seconds', 0)/60:.1f}min)" for c in slow))
    lines.append("─" * 72)
    print("\n".join(lines), flush=True)


def resources_cmd(cfg, args) -> int:
    """`main.py resources [--last N]`：打印最近的资源台账。"""
    ledger = cfg.state_dir / LEDGER
    if not ledger.exists():
        print(f"还没有资源台账（{ledger}）。跑一次 `python main.py` 或 rebuild 就有了。")
        return 0
    rows = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    rows = rows[-int(args.last):] if args.last else rows
    if not rows:
        print("台账为空。")
        return 0
    print(f"{'时间':<20}{'模式':<10}{'截止':<12}{'墙钟':>8}{'CPU':>9}{'利用率':>8}"
          f"{'计算利用率':>10}{'内存峰值':>10}{'任务':>7}")
    print("-" * 100)
    for r in rows:
        util = r.get("cpu_utilization")
        cutil = r.get("compute_cpu_utilization")
        print(f"{r.get('created_at', ''):<20}{r.get('mode', ''):<10}{r.get('cutoff', ''):<12}"
              f"{r.get('wall_seconds', 0):>8.0f}{r.get('cpu_seconds', 0):>9.0f}"
              f"{(util * 100 if util is not None else float('nan')):>7.1f}%"
              f"{(cutil * 100 if cutil is not None else float('nan')):>9.1f}%"
              f"{r.get('peak_charged_bytes', 0)/GiB:>9.1f}G{r.get('tasks', 0):>7}")
    print("-" * 100)
    print("利用率 = CPU 秒 /（墙钟 × 核数）；「计算利用率」只算年块执行阶段（不含计划/台账/发布）。")
    return 0
