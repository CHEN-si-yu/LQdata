#!/usr/bin/env python3
"""V2 训练调度：把 16 个「季度×折」跑完，并发数由**内存**（或 --jobs）决定。

用户 2026-09-26 定：**本文件只干并发调度这一件事** —— `--jobs=4` 就是「时刻保持 4 个子进程在跑」，
谁先结束谁的位置立刻补下一个；不给 `--jobs` 就按「启动那一刻的可用内存 ÷ 每折 15 GiB」估。
可见的 GPU 在启动时数一次，并发折按**轮转**钉到各张卡上（`CUDA_VISIBLE_DEVICES`），多卡机器上均匀铺开。
路径与评价回测在 model.py / analysis.py。
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

UNIT = Path(__file__).resolve().parent          # 评价窗与 model.py:quarters() 同源
TASKS = [(q, f) for q in ('2025Q3', '2025Q4', '2026Q1', '2026Q2') for f in (1, 2, 3, 4)]
PER_FOLD_GIB = 15.0                             # 一折的常驻内存（用户 2026-09-26 定；实测峰值见 complete.json.peak_rss_gib）


def stamp():
    return time.strftime('%H:%M:%S')


def cpu_count():
    """本进程真正能用的核数：优先 cgroup 配额（容器里可见核数常远大于配额），再退回亲和掩码。"""
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)


def free_memory_gib():
    """**启动那一瞬间**的可用内存（GiB）—— 只读一次，跑起来之后不再看（用户 2026-09-26 定）。

    = cgroup 上限 − 已用 + 可回收的 file cache。页缓存会被内核回收，不能算成"已占用"，
    否则一台刚读过数据的机器会被误判成内存不够。cgroup 读不到就退回 `/proc/meminfo`
    的 `MemAvailable`；再读不到返回 None，由调用方兜成串行。
    """
    def read(path):
        try:
            return Path(path).read_text()
        except OSError:
            return ''

    def single(text):
        """单值文件（`memory.max` / `memory.current`）里的那个数；`max` 或读不到 → None。"""
        try:
            return float(text.split()[0])
        except (IndexError, ValueError):
            return None

    def keyed(text, key):
        """`键 值` 形式的文件（`memory.stat` / `/proc/meminfo`）里 key 对应的值，单位随文件。"""
        for line in text.splitlines():
            if line.startswith(key):
                try:
                    return float(line.split()[1].rstrip(':'))
                except (IndexError, ValueError):
                    return None
        return None

    stat = read('/sys/fs/cgroup/memory.stat')                    # v2 / v1 同名，键名不同
    cache = keyed(stat, 'inactive_file') or keyed(stat, 'total_inactive_file') or 0.0
    limit = single(read('/sys/fs/cgroup/memory.max')) or single(read('/sys/fs/cgroup/memory/memory.limit_in_bytes'))
    used = single(read('/sys/fs/cgroup/memory.current')) or single(read('/sys/fs/cgroup/memory/memory.usage_in_bytes'))
    if limit is not None and used is not None:
        return max(0.0, (limit - used + cache) / 2 ** 30)
    available_kb = keyed(read('/proc/meminfo'), 'MemAvailable:')
    return available_kb / 2 ** 20 if available_kb else None


def jobs_default():
    """默认并发 = 启动时可用内存 ÷ 每折 15 GiB —— 内存是这套模型的硬约束，不按显存/核数猜。"""
    free = free_memory_gib()
    if free is None:
        print('读不到内存上限/余量：按串行（1 并发）跑', flush=True)
        return 1
    print(f'可用内存 {free:.1f} GiB ÷ 每折 {PER_FOLD_GIB:g} GiB', flush=True)
    return max(1, int(free / PER_FOLD_GIB))


def visible_gpus():
    """可见 GPU 的**标识列表**（子进程按它设 `CUDA_VISIBLE_DEVICES`，启动时读一次）。

    ★ 用户显式设过 `CUDA_VISIBLE_DEVICES` 就原样用它：**不能**改写成 `0,1,2…` —— 那是按
      "容器里第几张" 编号，用户写 `2,3` 时改写成 `0` 会指到别的卡上去。
    ★ 没设就数一张张卡：优先 torch（与子进程实际看到的是同一套可见性规则），
      没装 torch 才退回 `/dev/nvidia[0-9]*` 的个数。
    """
    env = os.environ.get('CUDA_VISIBLE_DEVICES')
    if env is not None:
        return [v.strip() for v in env.split(',') if v.strip()]
    try:
        import torch
        count = torch.cuda.device_count()
    except ImportError:
        count = len(list(Path('/dev').glob('nvidia[0-9]*')))
    return [str(i) for i in range(count)]


def spawn(quarter, fold, device, gpu=None):
    """一折一个子进程；已完成则追一行日志并返回 None，否则返回 (进程, 日志句柄)。

    `gpu` 非空时把这一折**钉在那张卡上**：只给这个子进程设 `CUDA_VISIBLE_DEVICES=<gpu>`，
    于是它里面的 `cuda:0` 就是这一张卡 —— 多卡机器上按启动顺序轮转，均匀铺开。
    """
    log = UNIT / 'model_logs' / f'{quarter}_{fold}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    if (UNIT / 'model_train' / quarter / f'fold{fold}' / 'complete.json').is_file():
        with open(log, 'a', buffering=1) as handle:
            handle.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {quarter} fold{fold} 已完成，本次复用\n')
        return None
    command = [sys.executable, '-B', '-u', str(UNIT / 'model.py'),
               '--quarter', quarter, '--fold', str(fold), '--device', device]
    env = dict(os.environ)
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = gpu
    handle = open(log, 'a', buffering=1)
    return subprocess.Popen(command, cwd=UNIT, env=env, stdout=handle, stderr=subprocess.STDOUT), handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=0,
                        help='并发折数；0 = 按启动时可用内存自动估（每折 15 GiB）')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    jobs = max(1, min(args.jobs or jobs_default(), len(TASKS)))
    gpus = [] if args.device == 'cpu' else visible_gpus()
    if args.device == 'cuda' and not gpus:
        raise SystemExit('--device cuda：一张卡都看不到')
    os.environ.setdefault('MX_THREADS', str(max(1, cpu_count() // jobs)))    # 每折分到的线程数
    print(f'{len(TASKS)} 折，并发 {jobs}，每折 {os.environ["MX_THREADS"]} 线程'
          + (f'，{len(gpus)} 张卡轮转：{" / ".join(gpus)}' if gpus else '，不用 GPU'), flush=True)

    pending, running, failed, started = list(TASKS), {}, [], time.time()
    launched = 0
    while pending or running:
        while pending and len(running) < jobs:                       # 补满在跑的名额
            task = pending.pop(0)
            gpu = gpus[launched % len(gpus)] if gpus else None       # 轮转：均匀铺到各张卡上
            child = spawn(*task, 'cuda' if gpu else args.device, gpu)
            if child is None:
                print(f'[{stamp()}] {task[0]} fold{task[1]} 已完成，跳过', flush=True)
                continue
            launched += 1
            running[task] = (*child, time.time())
            print(f'[{stamp()}] 启动 {task[0]} fold{task[1]}（在跑 {len(running)}/{jobs}'
                  + (f'，GPU {gpu}' if gpu else '') + '）', flush=True)
        for task, (proc, handle, t0) in list(running.items()):       # 收走已结束的
            if proc.poll() is None:
                continue
            handle.close()
            del running[task]
            print(f'[{stamp()}] {task[0]} fold{task[1]} 退出码 {proc.returncode}，'
                  f'用时 {time.time() - t0:.0f}s', flush=True)
            if proc.returncode:
                failed.append(task)
        time.sleep(2)

    print(f'训练结束：{len(TASKS) - len(failed)} 折就绪、{len(failed)} 折失败，'
          f'用时 {(time.time() - started) / 60:.1f} 分钟', flush=True)
    if failed:
        raise SystemExit(f'失败的折 {failed}：见各自 model_logs/<季度>_<折>.log')


if __name__ == '__main__':
    main()
