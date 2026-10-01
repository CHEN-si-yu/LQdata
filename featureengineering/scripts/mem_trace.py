#!/usr/bin/env python
"""并行度评估采样器：重建期间每 5 秒记一次内存与 CPU 占用。

用途：判断并行度能否从 2 提到 3。需要三个数就能判断：

  ① 容器物理内存（cgroup `total_rss`，v1）/ `anon`（v2）—— 90 GB 上限的红线；
  ② **批内进程合计 RSS**（`run_bounded` 的 64 GB 判据；共享页会被每进程各算一次，
     所以它天然大于 ①，两者都要看）；
  ③ 实际吃掉的 CPU 核数（判断有没有算力余量）。

每行 CSV：时间戳、块号、cgroup_rss、cgroup_cache、batch_rss、n_proc、cpu_cores

用法：python mem_trace.py <out.csv> [采样间隔秒]
"""
import csv
import os
import sys
import time
from pathlib import Path

OUT = Path(sys.argv[1])
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

# cgroup v1 / v2
_V2 = Path("/sys/fs/cgroup/memory.stat")
if _V2.exists():
    STAT_PATH, RSS_KEY = _V2, "anon"
else:
    STAT_PATH, RSS_KEY = Path("/sys/fs/cgroup/memory/memory.stat"), "total_rss"
_CACHE_KEY = "total_cache" if RSS_KEY == "total_rss" else "file"


def cgroup() -> tuple[float, float]:
    try:
        st = dict(l.split() for l in STAT_PATH.read_text().splitlines())
        return int(st.get(RSS_KEY, 0)) / 2**30, int(st.get(_CACHE_KEY, 0)) / 2**30
    except OSError:
        return 0.0, 0.0


def batch():
    """批内进程（backfill_history / main.py run|rebuild）+ 合计 RSS 与 CPU 时间。"""
    rss = cpu = 0
    n = 0
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            raw = (d / "cmdline").read_bytes().decode("utf-8", "replace")
            if not any(k in raw for k in ("backfill_history.py", "main.py")):
                continue
            stat = (d / "status").read_text()
            for line in stat.splitlines():
                if line.startswith("VmRSS:"):
                    rss += int(line.split()[1]) * 1024
                    n += 1
            f = (d / "stat").read_text().split()
            cpu += (int(f[13]) + int(f[14])) / os.sysconf("SC_CLK_TCK")
        except (OSError, ValueError, IndexError):
            pass
    return rss, cpu, n


def current_block() -> str:
    """当前正在跑的年份块（从 backfill 的日志文件名推断）。"""
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    logs = sorted(log_dir.glob("backfill_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1].stem.split("_")[-1] if logs else "?"


with OUT.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["t", "block", "cgroup_rss_gb", "cgroup_cache_gb", "batch_rss_gb",
                "n_proc", "cpu_cores", "cgroup_pct_of_90"])
    prev_cpu = None
    while True:
        t0 = time.time()
        rss, cache = cgroup()
        b_rss, cpu, n = batch()
        if prev_cpu is not None and t0 > prev_t0:
            cores = (cpu - prev_cpu) / (t0 - prev_t0)
        else:
            cores = 0.0
        prev_cpu, prev_t0 = cpu, t0
        w.writerow([time.strftime("%F %T"), current_block(), f"{rss:.2f}", f"{cache:.2f}",
                    f"{b_rss/2**30:.2f}", n, f"{cores:.2f}", f"{rss/90*100:.1f}"])
        fh.flush()
        time.sleep(max(1.0, INTERVAL - (time.time() - t0)))
