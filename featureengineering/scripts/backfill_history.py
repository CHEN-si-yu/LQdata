#!/usr/bin/env python3
"""全历史因子回填驱动 —— 2026-10-03 起**委托给多路年份流执行器**（`fea/executor.py`）。

## 这个脚本现在的职责

保留同名 CLI 与全部历史语义（`--force` 强制重算 / `--resume` 跳过已完成年 /
`--factors` 只补若干因子 / `--dry-run` 只看计划），实际执行走：

    main.py run --start <from>-01-01 --end <to>-12-31 --jobs N --streams K

即：**一次规划、一次台账、K 路年份流并行**（旧版是「逐块串行 + 每块跑一遍完整
日更台账」，9 块 → 9 遍固定开销）。按年分块的两个初衷都不变：
  · 每个年块仍是一个独立进程（内存隔离，缓存不跨年累积）；
  · 值逐格不变（面板锚点仍是「本年 1 月 1 日」，与全量重建一致）。

## 为什么还留着这个文件

1. `_next_jobs` / `_year_done` / `_scan_thin` / `_record_rebuild_attempt` 有回归测试
   直接钉着（`tests/test_backfill_adaptive.py`、`tests/test_rebuild_contract.py`），
   它们现在从 `fea.executor` 再导出；
2. 手册与历史日志里引用着它的用法。

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/backfill_history.py --dry-run                    # 只列计划，不动数据
    $PY scripts/backfill_history.py --jobs 8 --streams 2         # 真跑（断点续跑）
    $PY scripts/backfill_history.py --force --from-year 2026     # 强制重跑某段
    $PY scripts/backfill_history.py --factors foo bar            # 只补这几个因子的历史

推荐入口同旧版：`python main.py rebuild …`。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fea.resources import configure, safe_jobs           # noqa: E402
configure()

from fea import config as cfg_mod                          # noqa: E402
from fea.engine import Engine, int_to_str                 # noqa: E402
from fea.spec import all_specs                            # noqa: E402
# 兼容再导出：这些助手 2026-10-03 起实现位于 fea/executor.py（有测试直接 import 这里）。
from fea.executor import (                                # noqa: E402,F401
    _cgroup_files, _mem_gb, _mem_limit_gb, _MemSampler, _next_jobs,
    _real_gap, _scan_thin, _year_done, _record_rebuild_attempt,
    rebuild_plan, run_rebuild,
)

import factors                                            # noqa: F401,E402

PY = "/autodl-fs/data/miniconda3/bin/python"


def main() -> int:
    ap = argparse.ArgumentParser(description="全历史因子回填（多路年份流）")
    ap.add_argument("--from-year", type=int, default=None, help="起始年（默认 = default_start 的年）")
    ap.add_argument("--to-year", type=int, default=None, help="结束年（默认 = 上游基准表最后一天的年）")
    ap.add_argument("--jobs", type=int, default=0,
                    help="总车道预算（默认 0 = 用配置的 default_jobs；按 --streams 均分到各流）")
    ap.add_argument("--streams", type=int, default=0,
                    help="并行年份流数（0=auto：待算 ≥2 年且总车道 ≥6 时取 2）")
    ap.add_argument("--factors", nargs="*", default=None,
                    help="只回填这些因子（默认=全部）。★ 补单个/少量因子的历史时用它")
    ap.add_argument("--force", action="store_true", help="忽略「已覆盖」检查，强制重跑")
    ap.add_argument("--dry-run", action="store_true", help="只列计划")
    ap.add_argument("--adaptive", action="store_true",
                    help="（已内置：车道按实测内存/CPU 逐块自适应，此开关保留兼容）")
    ap.add_argument("--budget-bytes", type=int, default=0,
                    help="车道自适应的内存预算，字节（默认 0 = min(项目预算, 容器上限)×0.90）")
    ap.add_argument("--jobs-min", type=int, default=1, help="自适应下探下限（默认 1）")
    ap.add_argument("--jobs-max", type=int, default=0,
                    help="自适应上探上限（默认 = min(项目 max_jobs, CPU 配额)）")
    args = ap.parse_args()
    args.jobs = safe_jobs(args.jobs)

    cfg = cfg_mod.load()
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
    eng = Engine(cfg)
    y_from = args.from_year or int(cfg.default_start[:4])
    y_to = args.to_year or int(str(eng.baseline_last_day())[:4])
    if y_to < y_from:
        print(f"年份区间为空：{y_from}..{y_to}")
        return 1
    print(f"回填 {y_from}..{y_to} · default_start={cfg.default_start} · 并行 {args.jobs}"
          f" · streams={args.streams or 'auto'} · cgroup 上限 "
          f"{f'{_mem_limit_gb():.0f} GB' if _mem_limit_gb() else '未知'} · 当前 {_mem_gb():.2f} GB")
    return run_rebuild(cfg, specs=specs if specs is not None else [s for s in all_specs() if s.enabled],
                       y_from=y_from, y_to=y_to, force=args.force, jobs=args.jobs,
                       streams=args.streams, budget_bytes=args.budget_bytes,
                       jobs_min=args.jobs_min, jobs_max=args.jobs_max, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
