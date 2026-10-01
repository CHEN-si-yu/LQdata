#!/usr/bin/env python3
"""V51 quarter/fold scheduler with cgroup memory and GPU-aware concurrency."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import model as M

UNIT = Path(__file__).resolve().parent
TASKS = [(q, f) for q in M.QUARTERS for f in M.FOLDS]
PER_FOLD_GIB = 12.0  # conservative preflight reservation; actual RSS is recorded per fold
MEMORY_RESERVE_GIB = 16.0


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def memory_state():
    """Return cgroup memory in GiB, discounting only inactive file cache."""
    pairs = [
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current",
         "/sys/fs/cgroup/memory.stat", "inactive_file"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
         "/sys/fs/cgroup/memory/memory.usage_in_bytes",
         "/sys/fs/cgroup/memory/memory.stat", "total_inactive_file"),
    ]
    for limit_path, used_path, stat_path, cache_key in pairs:
        try:
            limit, used = int(_read(limit_path)), int(_read(used_path))
            if limit <= 0 or limit >= 2**60:
                continue
            cache = 0
            for line in _read(stat_path).splitlines():
                fields = line.split()
                if len(fields) == 2 and fields[0] == cache_key:
                    cache = int(fields[1])
                    break
            gib = float(2**30)
            used_effective = max(0.0, (used - cache) / gib)
            return {"limit_gib": limit / gib, "used_gib": used / gib,
                    "effective_used_gib": used_effective,
                    "available_gib": max(0.0, (limit - used + cache) / gib)}
        except (ValueError, OSError):
            continue
    return None


def visible_gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [x.strip() for x in env.split(",") if x.strip() and x.strip() != "-1"]
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def cpu_count():
    for quota_path, period_path in (
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        try:
            if period_path is None:
                quota, period = _read(quota_path).split()[:2]
            else:
                quota, period = _read(quota_path), _read(period_path)
            if quota != "max" and int(quota) > 0:
                return max(1, int(quota) // int(period))
        except (ValueError, OSError):
            pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def launch(q, fold, device, gpu=None, force=False):
    log = UNIT / "model_logs" / f"{q}_{fold}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    done = UNIT / "model_train" / q / f"fold{fold}" / "complete.json"
    if done.is_file() and not force:
        return None
    command = [sys.executable, "-B", "-u", str(UNIT / "model.py"),
               "--quarter", q, "--fold", str(fold), "--device", device]
    if force:
        command.append("--force")
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    handle = open(log, "a", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(command, cwd=UNIT, env=env, stdout=handle,
                            stderr=subprocess.STDOUT)
    return proc, handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=0,
                        help="maximum concurrent folds; 0 uses current memory and GPU capacity")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quarter", choices=M.QUARTERS)
    parser.add_argument("--fold", type=int, choices=M.FOLDS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if (args.quarter is None) != (args.fold is None):
        parser.error("--quarter and --fold must be supplied together")
    if args.jobs < 0:
        parser.error("--jobs must be >= 0")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be >= 1")
    M.validate_layout(allow_missing=True)

    tasks = [(args.quarter, args.fold)] if args.quarter else list(TASKS)
    gpus = [] if args.device == "cpu" else visible_gpus()
    if args.device == "cuda" and not gpus:
        raise SystemExit("--device cuda requested but no GPU is visible")
    device = "cuda" if gpus else "cpu"
    state = memory_state()
    if state:
        mem_slots = max(1, int(max(0.0, state["available_gib"] - MEMORY_RESERVE_GIB) // PER_FOLD_GIB))
        print("cgroup memory: " + ", ".join(f"{k}={v:.1f}GiB" for k, v in state.items()), flush=True)
    else:
        mem_slots = 1
        print("cgroup memory unavailable; limiting scheduler to one fold", flush=True)
    hardware_slots = len(gpus) if gpus else max(1, cpu_count() // 8)
    jobs = min(len(tasks), args.jobs or hardware_slots, hardware_slots, mem_slots)
    jobs = max(1, jobs)
    print(f"V51: {len(tasks)} folds, max concurrent={jobs}, device={device}, "
          f"GPU rotation={gpus or 'none'}, per-fold reserve={PER_FOLD_GIB:g}GiB", flush=True)

    pending = list(tasks)
    running = {}
    failed = []
    launched = 0
    start = time.time()
    while pending or running:
        while pending and len(running) < jobs:
            mem = memory_state()
            if mem and mem["available_gib"] < PER_FOLD_GIB + MEMORY_RESERVE_GIB:
                print(f"memory gate: {mem['available_gib']:.1f}GiB available; waiting for a slot", flush=True)
                break
            task = pending.pop(0)
            gpu = gpus[launched % len(gpus)] if gpus else None
            child = launch(*task, device=device, gpu=gpu, force=args.force)
            if child is None:
                print(f"{task[0]} fold{task[1]} already complete; skipped", flush=True)
                continue
            proc, handle = child
            running[task] = (proc, handle, time.time())
            launched += 1
            print(f"started {task[0]} fold{task[1]} ({len(running)}/{jobs} running" +
                  (f", GPU {gpu}" if gpu is not None else "") + ")", flush=True)
        for task, (proc, handle, started) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            del running[task]
            print(f"finished {task[0]} fold{task[1]} rc={code}, elapsed={time.time()-started:.0f}s", flush=True)
            if code:
                failed.append(task)
        if pending or running:
            time.sleep(args.poll_seconds)
    print(f"V51 scheduler finished: {len(tasks)-len(failed)}/{len(tasks)} folds ready, "
          f"elapsed={(time.time()-start)/60:.1f}min", flush=True)
    M.validate_layout(allow_missing=True)
    if failed:
        raise SystemExit(f"failed folds: {failed}; see model_logs/<quarter>_<fold>.log")


if __name__ == "__main__":
    main()
