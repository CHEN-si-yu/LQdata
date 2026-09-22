"""以容器内存监控整个任务进程组；训练/分析始终串行，提前留下断点退出。"""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
logs = root / 'logs'
logs.mkdir(exist_ok=True)
limit_raw = Path('/sys/fs/cgroup/memory.max').read_text().strip()
hard = int(limit_raw) if limit_raw != 'max' else 90_000_000_000
# 保守按十进制GB解释90GB；给工具采样间隔与瞬时内存波动至少留6GB。
total_stop = min(84_000_000_000, int(hard * .89))
anon_stop = min(72_000_000_000, int(hard * .78))

# ★ 2026-09-21 修正：`memory.current` **含可回收的 page cache**（初加工读盘会把它顶到 40 GiB+），
#   而 cache 在接近 cgroup 上限时由内核自动回收，不是 OOM 的风险来源。原判据把 cache 一起算，
#   实测在 **anon 只有 41.9 GiB** 时误停（current=85.1 GiB，file=43 GiB）。
#   现改为对「current − file」（≈ anon + 内核态）判 total_stop，并对 anon 保留更严的 72 GiB 闸门。
#   ★ 90 GiB 的 cgroup 硬上限没变，`current` 仍逐行落盘，事后可核对。
command = sys.argv[1:]
proc = subprocess.Popen(command, start_new_session=True)
start = time.time()
(logs/'run_control.json').write_text(json.dumps(dict(guard_pid=os.getpid(),process_group=proc.pid,command=command,started_at=start)))
def stop_requested(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')
signal.signal(signal.SIGTERM,stop_requested)
signal.signal(signal.SIGINT,stop_requested)
reason = None
peaks = {'current':0, 'anon':0}
try:
    with (logs/'memory_guard.csv').open('w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['elapsed','pid','current_bytes','anon_bytes','file_bytes','effective_bytes'])
        tick = 0
        while proc.poll() is None:
            current = int(Path('/sys/fs/cgroup/memory.current').read_text())
            stat = dict((k,int(v)) for k,v in (x.split() for x in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines()))
            peaks['current'] = max(peaks['current'],current)
            peaks['anon'] = max(peaks['anon'],stat['anon'])
            if tick % 8 == 0:
                w.writerow([round(time.time()-start,2),proc.pid,current,stat['anon'],stat['file'],
                            current-stat.get('file',0)])
                f.flush()
            effective = current - stat.get('file', 0)
            if effective > total_stop or stat['anon'] > anon_stop:
                reason = (f'memory_guard effective={effective} anon={stat["anon"]} '
                          f'current={current} file={stat.get("file", 0)}')
                print(reason,flush=True)
                os.killpg(proc.pid,signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                break
            tick += 1
            time.sleep(.25)
finally:
    # 调度器异常退出时，其工作进程仍可能活着；始终清理本任务独立进程组。
    try:
        os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        code = proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        code = proc.wait()
    # 主进程已退出并不代表孙进程均已回收。
    try:
        os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
    result = dict(command=command,pid=proc.pid,seconds=time.time()-start,exit_code=code,
                  reason=reason,peak_bytes=peaks,total_stop=total_stop,anon_stop=anon_stop)
    (logs/'memory_guard_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
sys.exit(75 if reason else code)
