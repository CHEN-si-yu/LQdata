#!/usr/bin/env python3
"""experiments2/V1 顺序训练调度器；每折限单线程，留出共享 CPU/内存。"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import model as M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1, help="保留兼容参数；本单元默认顺序执行")
    parser.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    parser.add_argument("--quarter", choices=M.QUARTERS)
    parser.add_argument("--fold", type=int, choices=M.FOLDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (args.quarter is None) != (args.fold is None):
        parser.error("--quarter 与 --fold 要么同时指定，要么都不指定")
    if args.jobs < 1:
        parser.error("--jobs 必须 ≥ 1")
    if args.jobs > 1:
        print(f"请求并发 {args.jobs}；为共享资源与低内存占用，本轮仍按单折顺序训练。", flush=True)
    panel = M.load_panel()
    tasks = [(args.quarter, args.fold)] if args.quarter else [
        (quarter, fold) for quarter in M.QUARTERS for fold in M.FOLDS]
    started = time.time()
    for n, (quarter, fold) in enumerate(tasks, 1):
        path = M.ROOT / "model_logs" / f"{quarter}_{fold}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{n}/{len(tasks)}] {quarter} fold{fold} 开始", flush=True)
        with open(path, "w", encoding="utf-8", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = M.train_fold(panel, quarter, fold, force=args.force)
                print(f"完成：{result.get('seconds', 0):.1f}s；"
                      f"峰值 RSS={result.get('peak_rss_gib', 0):.3f} GiB", flush=True)
        print(f"[{n}/{len(tasks)}] {quarter} fold{fold} 完成", flush=True)
    print(f"训练调度结束：{len(tasks)} 折，用时 {(time.time() - started) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()

