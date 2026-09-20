#!/usr/bin/env python3
"""增量生成测试：**删掉最新一天 → 跑增量 → 与全量结果逐格比对 + 逐因子计时**。

用户 2026-09-15 的交办（原话）：

> 「我们包括两个任务，一个是全量生成（只生成 2026 年以后）。你还需要测试一个任务是
>   增量生成：假设上游数据每天更新一天，你的因子数据需要增量的更新一天。
>   这两个之间的逻辑框架体系可能存在差异，我们要追求增量更新的效率就需要做相应的调整。
>   你需要测试『我们把最新的一天 0914 删了，然后测试增量生成的代码，
>   首先是否和全量生成的结果一致、其次是记录每个因子增量生成的时间是否高效』」

## 测试怎么做的（三段）

1. **`prepare`** —— 把当前 `data/factors` 当作**全量参照**备份到 `--ref` 目录；
   然后把每个因子的 `year=2026` 分区**截到最后交易日之前**（默认裁掉 2026-09-14 及以后），
   并同步把 manifest 的 `coverage` / `partitions` 一起裁掉 ——
   这样引擎**看不出来**这是人为裁的，它面对的正是"昨天跑完、今天上游多了一天"的真实状态。
2. **`run`** —— 跑**普通增量** `main.py run`（**不带 `--rebuild`**），
   并计时：整轮墙钟 + 每个因子的 `task_seconds`（引擎在 manifest 里逐因子记的）。
3. **`verify`** —— 逐因子、逐 `(trade_date, stock_code)` 格子比对**增量结果 vs 全量参照**：
   - **必须完全一致**（同一个 T 的值不能因为"分两次算"而变）—— 这是增量的正确性红线；
   - 同时报出**每个因子这次重算了多少天 / 花了多少秒** —— 这是增量的效率指标。

## 为什么"一致"是硬要求

增量跑的是同一个因子函数，只是把面板截到更小的窗口。`warmup_days` 的设计目的就是让
"增量窗口算出来的值"与"全量重建算出来的值"**逐格相同**。任何差异都说明：
要么 warmup 给少了，要么某个环节用了**面板之外**的信息（例如不可复现的浮点中心化、
或隐含的全历史窗口）—— 两者都是必须修的真问题。

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/test_incremental.py prepare          # 备份 + 裁掉最新一天
    $PY scripts/test_incremental.py run             # 跑增量并计时
    $PY scripts/test_incremental.py verify          # 逐格比对 + 耗时报告
    # 或者一把梭： $PY scripts/test_incremental.py all
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
    tmp.replace(path)


def _factors(cfg) -> list[str]:
    return sorted(p.name for p in cfg.factors_dir.iterdir()
                  if p.is_dir() and (p / "year=2026").is_dir())


def cmd_prepare(args, cfg) -> int:
    ref = Path(args.ref)
    ref.mkdir(parents=True, exist_ok=True)
    cutoff = args.cutoff                      # 例如 2026-09-11（最后交易日的前一天）
    names = _factors(cfg)
    print(f"参照备份 → {ref}   （{len(names)} 个因子，裁到 {cutoff}）")
    n_rows_cut = 0
    for nm in names:
        src = cfg.factors_dir / nm / "year=2026" / "data.parquet"
        dst = ref / f"{nm}.parquet"
        if not dst.exists():
            shutil.copy2(src, dst)            # 参照只备份一次（幂等）
        df = pd.read_parquet(dst)
        keep = df[df["trade_date"] <= cutoff]
        cut = len(df) - len(keep)
        n_rows_cut += cut
        _atomic_write(keep, src)
        # manifest 同步裁掉（否则引擎以为这段已经算过，不会补）
        mp = cfg.state_dir / f"{nm}.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            cov = []
            for a, b in m.get("coverage", []):
                if a > cutoff:
                    continue
                cov.append([a, min(b, cutoff)])
            m["coverage"] = cov
            for y, p in (m.get("partitions") or {}).items():
                if str(p.get("max_date", "")) > cutoff:
                    p["max_date"] = cutoff
                    sel = keep[keep["trade_date"] >= str(p.get("min_date", "0000-00-00"))]
                    p["rows"] = int(len(sel))
                    p["nonnull"] = int(sel["value"].notna().sum() if "value" in sel else 0)
            mp.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    print(f"已裁掉 {n_rows_cut:,} 行（每个因子的 2026 分区现在止于 {cutoff}）")
    print("→ 现在跑：scripts/test_incremental.py run")
    return 0


def cmd_run(args, cfg) -> int:
    log = ROOT / "logs" / f"inc_test_{time.strftime('%m%d_%H%M')}.log"
    t0 = time.time()
    print(f"跑增量（普通 run，不带 --rebuild）… 日志 {log}")
    with log.open("w") as fh:
        rc = subprocess.call([PY, "-u", "-W", "ignore", str(ROOT / "main.py"), "run",
                              "--jobs", str(args.jobs)], stdout=fh, stderr=subprocess.STDOUT,
                             cwd=str(ROOT))
    el = time.time() - t0
    print(f"退出码 {rc} · 墙钟 {el:.1f}s")
    (Path(args.ref) / "_last_run.json").write_text(json.dumps(
        {"seconds": round(el, 1), "rc": rc, "log": str(log),
         "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False))
    print("→ 现在比对：scripts/test_incremental.py verify")
    return 0 if rc == 0 else 1


def cmd_verify(args, cfg) -> int:
    ref = Path(args.ref)
    if not ref.exists():
        print(f"没有参照目录 {ref}，先跑 prepare")
        return 1
    names = _factors(cfg)
    print(f"逐格比对：增量结果 vs 全量参照（{len(names)} 个因子）")
    print("-" * 100)
    bad, rows_checked = [], 0
    timings = []
    for nm in names:
        rp = ref / f"{nm}.parquet"
        if not rp.exists():
            print(f"  ? {nm}：参照缺失，跳过")
            continue
        ref_df = pd.read_parquet(rp)
        cur = pd.read_parquet(cfg.factors_dir / nm / "year=2026" / "data.parquet")
        rows_checked += len(ref_df)
        m = ref_df.merge(cur, on=["trade_date", "stock_code"], how="outer",
                         suffixes=("_r", "_c"), indicator=True)
        only_r = int((m["_merge"] == "left_only").sum())
        only_c = int((m["_merge"] == "right_only").sum())
        both = m[m["_merge"] == "both"]
        vr = pd.to_numeric(both["value_r"], errors="coerce").to_numpy(np.float64)
        vc = pd.to_numeric(both["value_c"], errors="coerce").to_numpy(np.float64)
        same = (np.isnan(vr) & np.isnan(vc)) | (vr == vc)
        n_diff = int((~same).sum())
        rr = pd.to_numeric(both["rank_r"], errors="coerce").to_numpy(np.float64)
        rc_ = pd.to_numeric(both["rank_c"], errors="coerce").to_numpy(np.float64)
        same_r = (np.isnan(rr) & np.isnan(rc_)) | (rr == rc_)
        n_diff_r = int((~same_r).sum())
        # 这次重算了多少天 / 花了多少秒（引擎写在 manifest.last_run 里）
        mp = cfg.state_dir / f"{nm}.json"
        lr = json.loads(mp.read_text()).get("last_run", {}) if mp.exists() else {}
        timings.append({"factor": nm, "task_seconds": lr.get("task_seconds"),
                        "task_days": lr.get("task_days"), "rows": len(cur)})
        if only_r or only_c or n_diff or n_diff_r:
            bad.append((nm, only_r, only_c, n_diff, n_diff_r))
            print(f"  ✘ {nm}：仅参照 {only_r} / 仅增量 {only_c} / value 不同 {n_diff} / rank 不同 {n_diff_r}")
    print("-" * 100)
    print(f"比对 {rows_checked:,} 行 · 不一致因子 {len(bad)} 个")
    if not bad:
        print("✅ 增量结果与全量**逐格完全一致**")
    # ---- 耗时报告
    print("\n逐因子增量耗时（task_seconds = 该因子本次各任务实际耗时之和）")
    print(f"{'因子':<30}{'重算天数':>8}{'耗时(s)':>10}{'分区行数':>12}")
    print("-" * 100)
    ts = [t for t in timings if t["task_seconds"] is not None]
    for t in sorted(ts, key=lambda x: -(x["task_seconds"] or 0))[: args.top]:
        print(f"{t['factor']:<30}{str(t['task_days']):>8}{t['task_seconds']:>10.1f}{t['rows']:>12,}")
    tot = sum(t["task_seconds"] or 0 for t in ts)
    days = [(t["task_days"] or 0) for t in ts]
    lr = json.loads((ref / "_last_run.json").read_text()) if (ref / "_last_run.json").exists() else {}
    print("-" * 100)
    print(f"合计 {len(ts)} 个因子 · 计算耗时合计 {tot:.0f}s（并行 8 进程）"
          f" · 墙钟 {lr.get('seconds', '?')}s")
    if days:
        print(f"每个因子平均重算 {sum(days)/len(days):.1f} 个交易日"
              f"（{min(days)}~{max(days)}）—— 这是**回刷窗口**（revision_days + L2 水位）决定的")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="增量生成测试（删最新一天 → 增量 → 比对 + 计时）")
    ap.add_argument("stage", choices=["prepare", "run", "verify", "all"])
    ap.add_argument("--cutoff", default="2026-09-11",
                    help="裁到哪一天（含）；默认 2026-09-11 = 最后交易日 09-14 的前一天")
    ap.add_argument("--ref", default="/tmp/inc_ref", help="全量参照目录")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--top", type=int, default=25, help="耗时榜显示前 N 个")
    args = ap.parse_args()

    from fea.config import load as load_cfg
    cfg = load_cfg()
    for stage in (["prepare", "run", "verify"] if args.stage == "all" else [args.stage]):
        rc = {"prepare": cmd_prepare, "run": cmd_run, "verify": cmd_verify}[stage](args, cfg)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
