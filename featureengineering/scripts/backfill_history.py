#!/usr/bin/env python3
"""全历史因子回填驱动 —— 按年分块、可断点续跑、逐块报内存峰值。

## 为什么按年分块（不是为了正确性，纯粹是为了内存）

2026-09-17 起面板下界锚在「本任务所在年的 1 月 1 日」，所以**每一年的值本身**
只依赖 [本年 1/1 − warmup, 本年] 这一段 —— 分块与否，值完全一样（时间轴是连续的，
落盘只是按年分区）。

但**进程级缓存是按窗口累积的**：`PriceLayer._ensure_loaded` / `DerivedCache._read_all`
只会扩窗、不会缩窗（扩窗=重读）。一个进程连跑 15 年 → 每个 worker 手里都会攒下
全历史价格层（实测单进程峰值 7.1~8.1 GB）→ 8 进程直接顶到 cgroup 上限。
**每年一个新进程**避免跨年缓存累积；实际峰值随原始数据、窗口和并发变化。
2026-09-19 全年串行实测最大子进程 RSS 约 9 GiB，不能承诺恒定 2~3 GB。
推荐统一使用 `main.py rebuild --jobs 1`，最大并发 2。

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/backfill_history.py --dry-run          # 只列计划，不动数据
    $PY scripts/backfill_history.py --jobs 2           # 真跑（断点续跑：已覆盖的年自动跳过）
    $PY scripts/backfill_history.py --from-year 2018   # 只补 2018 起
    $PY scripts/backfill_history.py --factors foo bar  # ★ 只补这几个因子的历史（最省内存）
    $PY scripts/backfill_history.py --force --from-year 2026   # 强制重跑某年

每块一个独立进程 + 一份日志 `logs/backfill_<年>.log`；内存峰值从 cgroup 直接采样。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fea.resources import configure, safe_jobs
configure()

from fea import config as cfg_mod            # noqa: E402
from fea.engine import Engine, int_to_str   # noqa: E402
from fea.manifest import Manifest            # noqa: E402
from fea.spec import all_specs               # noqa: E402

import factors                               # noqa: F401,E402

PY = "/autodl-fs/data/miniconda3/bin/python"
MEM_CURRENT = Path("/sys/fs/cgroup/memory.current")
MEM_MAX = Path("/sys/fs/cgroup/memory.max")


def _mem_gb() -> float:
    try:
        return int(MEM_CURRENT.read_text()) / 2 ** 30
    except Exception:
        return 0.0


class _MemSampler(threading.Thread):
    """跑一块期间每 5 秒采一次 cgroup 内存，报告峰值（用户明确要求"别超内存"）。"""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0.0
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.peak = max(self.peak, _mem_gb())
            self._stop_event.wait(5.0)

    def stop(self) -> None:
        self._stop_event.set()


def _real_gap(eng: Engine, man: "Manifest", start: str, end: str) -> bool:
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
    """扫 manifest 报告本年非空率过低的因子，分两档：

    · **全 NaN（0.0%）** = 一定是垃圾分区（该年整年没有值）→ 必须按
      `growth.py::DEEP_LAG_START` / `fundamental2.py::FIN_START_DEEP20` 的范式
      补一个**实测的**显式 `start=`，再单独重跑该因子。
    · **稀疏（0 < 非空率 < 5%）** = 不一定错：事件类因子（龙虎榜 3~4%、
      涨停/公告类）本来就稀疏，`main.py check` 的「口径太稀」在它们身上是**预期**。
      先看一眼是不是"数据本身就这么少"，别急着调 start。

    ★ 零额外读盘：`mark_partition` 已经把 (rows, nonnull) 记在 `state/<因子>.json` 里。
    """
    zero, thin = [], []
    for s in (specs if specs is not None else all_specs()):
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


def _year_done(eng: Engine, cfg, year: int, specs: list | None = None) -> bool:
    """本年是否已经**全部**算过（决定能否跳过）。

    ⚠️ 本年的右端要**截到上游基准表的最后一天**：日历会延伸到未来（今天 09-17 时
       日历里已经有 09-17..12-31 的交易日），不截的话每个因子在最后一年都恒报"缺一段
       未来的日子"，`--dry-run` 永远显示本年待补、脚本每次都白跑一块。
    """
    y1 = min(f"{year}-12-31", int_to_str(eng.baseline_last_day()))
    y0 = f"{year}-01-01"
    for s in (specs if specs is not None else all_specs()):
        if not s.enabled or s.resolved_start(cfg) > y1:
            continue
        # ★ 起点晚于年初的因子（如研发类 2019-05-01）只要求它自己起点之后的那段
        start = max(y0, s.resolved_start(cfg))
        man = Manifest.load(cfg.state_dir, s.name)
        if man.recipe != eng._recipe(s) or _real_gap(eng, man, start, y1):
            return False
        partition = cfg.factors_dir / s.name / f"year={year}" / "data.parquet"
        if not partition.exists():
            return False
        expected = man.partitions.get(str(year), {}).get("file_identity")
        if expected:
            st = partition.stat()
            if expected != [st.st_size, st.st_mtime_ns]:
                return False
    return True


def _invalidate_coverage(cfg, specs, first_year, last_year):
    """Mark the forced interval pending before any worker starts.

    Existing parquet stays in place. If interrupted, --resume cannot confuse
    old coverage with a successfully refreshed year. Preserve outside ranges.
    """
    from fea.manifest import _next_day, _prev_day, _atomic_json
    import datetime
    lo, hi = f"{first_year}-01-01", f"{last_year}-12-31"
    saved = {}
    selected = specs if specs is not None else all_specs()
    for spec in selected:
        if not spec.enabled:
            continue
        man = Manifest.load(cfg.state_dir, spec.name)
        saved[spec.name] = man.coverage
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    _atomic_json({"interval": [lo, hi], "coverage_before": saved},
                 cfg.state_dir / "rebuild_attempts" / f"{stamp}.json")
    for name, spans in saved.items():
        man = Manifest.load(cfg.state_dir, name)
        kept = []
        for a, b in spans:
            if b < lo or a > hi:
                kept.append([a, b])
            else:
                if a < lo:
                    kept.append([a, _prev_day(lo)])
                if b > hi:
                    kept.append([_next_day(hi), b])
        man.coverage = kept
        man.save()


def main() -> int:
    ap = argparse.ArgumentParser(description="全历史因子回填（按年分块）")
    ap.add_argument("--from-year", type=int, default=None, help="起始年（默认 = default_start 的年）")
    ap.add_argument("--to-year", type=int, default=None, help="结束年（默认 = 上游基准表最后一天的年）")
    ap.add_argument("--jobs", type=int, default=1, help="默认串行，最多 2 个进程")
    ap.add_argument("--factors", nargs="*", default=None,
                    help="只回填这些因子（默认=全部）。★ 补单个/少量因子的历史时用它："
                         "块内只有 1~2 个任务，内存最省（全量 230 因子时单块峰值 45~58 GB）")
    ap.add_argument("--force", action="store_true", help="忽略「已覆盖」检查，强制重跑")
    ap.add_argument("--dry-run", action="store_true", help="只列计划")
    args = ap.parse_args()
    args.jobs = safe_jobs(args.jobs)

    cfg = cfg_mod.load()
    eng = Engine(cfg)
    specs = None
    if args.factors:
        want = set(args.factors)
        known = {s.name for s in all_specs()}
        unknown = sorted(want - known)
        if unknown:
            print(f"✘ 不认识的因子：{' '.join(unknown)}")
            return 1
        specs = [s for s in all_specs() if s.name in want]
        print(f"★ 只回填 {len(specs)} 个因子：{' '.join(s.name for s in specs)}")
    y_from = args.from_year or int(cfg.default_start[:4])
    y_to = args.to_year or int(str(eng.baseline_last_day())[:4])
    if y_to < y_from:
        print(f"年份区间为空：{y_from}..{y_to}")
        return 1
    print(f"回填 {y_from}..{y_to} · default_start={cfg.default_start} · "
          f"并行 {args.jobs} · cgroup 上限 {int(MEM_MAX.read_text()) / 2 ** 30:.0f} GB · "
          f"当前 {_mem_gb():.2f} GB")
    plan = []
    for y in range(y_from, y_to + 1):
        done = _year_done(eng, cfg, y, specs)
        plan.append((y, done))
        print(f"  {y}: {'已覆盖，跳过' if done and not args.force else '待补'}")
    todo = [y for y, done in plan if not done or args.force]
    if args.dry_run:
        print(f"共 {len(todo)} 块待跑：{todo}")
        return 0
    if not todo:
        print("没有需要补的年份。")
        return 0

    if args.force:
        _invalidate_coverage(cfg, specs, y_from, y_to)

    (ROOT / "logs").mkdir(exist_ok=True)
    t_all = time.time()
    for i, y in enumerate(todo, 1):
        end = min(f"{y}-12-31", int_to_str(eng.baseline_last_day()))
        cmd = [PY, "main.py", "run", "--jobs", str(args.jobs)]
        if args.factors:
            cmd += list(args.factors)
        if end:
            cmd += ["--end", end]
        if args.force:
            cmd.append("--refresh")
        # ★★ 2026-09-18 修（**13 个因子永远重建不了**的真 bug）：
        #   原来只传 `--end`、**不传 `--start`** ⇒ `Engine.plan_years` 会从因子的
        #   `resolved_start` 起算全部年份。对 `deps` 是**因子名**的那 13 个耦合因子
        #   （`cp_*` / `fundflow_retail_inst_divergence` / `mf_flow_factor_momentum_20`），
        #   父因子重建会触发「水位失效传播 ⇒ 从起点重算」，于是 `plan_years` 对
        #   `--end 2014` 也返回 `[2012, 2013, 2014]` ⇒ `len >= 3` 恒成立 ⇒
        #   被 `main.py` 的「跨 ≥3 年就摘出」守卫**每块都摘掉**，且它们自己的报错
        #   建议恰恰是"请用 backfill_history.py --factors"——**死锁**。
        #   实测后果：股票池冻结重建后，这 13 个因子的 2014..2026 全是旧的 3484 池数据
        #   （2026-09-18 由 `scripts/verify_frozen_universe.py` 抓到）。
        #   `--start` 的语义正是"本次只产出这一天之后的因子值、不影响 warmup、
        #   值与全量重建逐格一致"（见 `main.py run --help`），所以这是**收窄本次产出范围**，
        #   不改变任何计算结果。加上它后 `plan_years` 只返回本年 ⇒ 守卫不再误伤。
        #   ⚠️ 必须**每一块都传**（含第一块）—— 第一块正是要修的那块，
        #      漏掉它 `plan_years(end=y_from)` 仍返回 `[start..y_from]`（≥3 年）而被摘出。
        cmd += ["--start", f"{y}-01-01"]
        log = ROOT / "logs" / f"backfill_{y}.log"
        print("-" * 96)
        print(f"[{i}/{len(todo)}] {y} · {' '.join(cmd[1:])} · 日志 {log.name}", flush=True)
        t0, base = time.time(), _mem_gb()
        samp = _MemSampler()
        samp.start()
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        samp.stop()
        samp.join(timeout=6)
        el = time.time() - t0
        tail = ""
        try:
            head = log.read_text(encoding="utf-8").strip().splitlines()
            tail = head[-1] if head else ""
        except Exception:
            pass
        print(f"    退出码 {p.returncode} · 用时 {el/60:.1f} min · "
              f"内存峰值 {samp.peak:.1f} GB（起步 {base:.1f} GB）· {tail}", flush=True)
        zero, thin = _scan_thin(cfg, y, specs=specs)
        if zero:
            print(f"    ✘ {y} 年有 {len(zero)} 个因子**整年全 NaN**（垃圾分区，"
                  f"按 DEEP_LAG_START 范式补实测显式 start 后单独重跑）：", flush=True)
            print(f"      {' '.join(zero[:12])}" + (" …" if len(zero) > 12 else ""), flush=True)
        if thin:
            print(f"    ⚠ {y} 年有 {len(thin)} 个因子非空率 <5%（稀疏事件类因子属预期，"
                  f"先看一眼数据本身）：", flush=True)
            print(f"      {' '.join(thin[:12])}" + (" …" if len(thin) > 12 else ""), flush=True)
        if p.returncode != 0:
            if p.returncode == 2 and y == y_to:
                # 最后一块不带 --end，`Engine.plan` 会把「本年之前的所有缺口」一起算进去。
                # 若这些因子连早年都没算过，`main.py` 的多年守卫会（正确地）拒绝启动。
                print(f"ℹ {y} 块被 main.py 的**多年守卫**拦下了：它一次要算 {y_from}..{y}。\n"
                      f"  从更早的年份开始跑即可（每块只算一年）：\n"
                      f"      python scripts/backfill_history.py --from-year {y_from} "
                      f"{'--factors ' + ' '.join(args.factors) if args.factors else ''}")
            print(f"✘ {y} 失败，已停止（日志 {log}）。修好后重跑本命令即可续上 —— "
                  f"已覆盖的年份会自动跳过。")
            return p.returncode
    print("-" * 96)
    print(f"✔ 全部 {len(todo)} 块完成 · 总用时 {(time.time()-t_all)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
