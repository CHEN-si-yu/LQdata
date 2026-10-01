# -*- coding: utf-8 -*-
"""带计量的运行器：跑一条命令，记**墙钟耗时**与**峰值内存**，落一行 JSON 台账。

为什么不用 `free`：本机 `free` 看到的是宿主机 461 GiB，而容器 cgroup 上限只有 90 GiB
（`/sys/fs/cgroup/memory/memory.limit_in_bytes` = 96,636,764,160）。**看错了会批过头**
（项目实测爆过一次容器重启）。所以这里只读 cgroup：

  · `memory.usage_in_bytes` —— 计费内存，含可回收页缓存；
  · `memory.stat` 的 `anon` —— 不可回收的匿名内存，**这才是"真占了多少"**；
  · 两个都采，取运行全程的**最大值**（1 秒一采，采样协程在独立子进程里，不干扰被测进程）。

★ 为什么要独立子进程采样：被测进程自己也会爆内存；采样器必须是"即使它死了也还在"的那个。
  另外 cgroup 的用量是**整个容器**的（含同时段的别的进程），所以台账里同时记"基线用量"，
  净增量 = 峰值 − 基线，才是这条命令自己的开销。

用法：
  python runmeter.py --label 快照增量 --cwd /root/autodl-fs/model -- \
      /autodl-fs/data/miniconda3/bin/python preparingdata.py
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CG = Path('/sys/fs/cgroup/memory')
LIMIT = int((CG / 'memory.limit_in_bytes').read_text().strip())
#: 计量台账落点：默认放平台的 state/ 下，可用 --out 或 $MX_TIMINGS 覆盖。
#: ★ 不要落在脚本自己的目录里 —— scripts/ 是代码目录，不该混进运行状态。
OUT = Path(os.environ.get('MX_TIMINGS', '/root/autodl-fs/everyday_tasks/state/timings.jsonl'))


def sample():
    usage = int((CG / 'memory.usage_in_bytes').read_text().strip())
    anon = 0
    for line in (CG / 'memory.stat').read_text().splitlines():
        if line.startswith('anon '):
            anon = int(line.split()[1])
            break
        if line.startswith('rss '):          # ★ 本机 cgroup v1 的 stat 没有 anon，用 rss
            anon = int(line.split()[1])
    return usage, anon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', default=None, help='计量台账 jsonl 落点（默认 $MX_TIMINGS 或平台 state/）')
    ap.add_argument('--cwd', default=None)
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [c for c in a.cmd if c != '--']

    base_u, base_a = sample()
    # 采样峰值**每个 tick 都落盘**（覆盖写）：被测进程若把容器撑爆，采样器会被一起杀掉，
    # 只有已经写到磁盘的那个值能活下来。用 stdout 的话缓冲会丢，等于白采。
    peakfile = OUT.parent / f'.peak_{os.getpid()}'
    sampler = subprocess.Popen([sys.executable, '-c', '''
import time, sys
CG = "/sys/fs/cgroup/memory"
peak_u = peak_a = 0
path = sys.argv[1]
while True:
    u = int(open(CG + "/memory.usage_in_bytes").read())
    a = 0
    for line in open(CG + "/memory.stat"):
        if line.startswith("anon "):
            a = int(line.split()[1]); break
        if line.startswith("rss "):
            a = int(line.split()[1])
    if u > peak_u or a > peak_a:
        peak_u, peak_a = max(peak_u, u), max(peak_a, a)
        open(path, "w").write(f"{peak_u} {peak_a}")
    time.sleep(1.0)
''', str(peakfile)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=a.cwd)
    wall = time.time() - t0
    time.sleep(1.2)                      # 让采样器把最后一个 tick 写完
    sampler.terminate()
    try:
        peak_u, peak_a = map(int, peakfile.read_text().split())
    except Exception:
        peak_u = peak_a = 0
    peakfile.unlink(missing_ok=True)

    rec = dict(label=a.label, cmd=' '.join(cmd), cwd=a.cwd, rc=proc.returncode,
               wall_seconds=round(wall, 1), wall_human=f'{wall/60:.1f} 分钟',
               peak_usage_gib=round(peak_u / 1024**3, 2), peak_anon_gib=round(peak_a / 1024**3, 2),
               base_usage_gib=round(base_u / 1024**3, 2), base_anon_gib=round(base_a / 1024**3, 2),
               net_anon_gib=round((peak_a - base_a) / 1024**3, 2),
               cgroup_limit_gib=round(LIMIT / 1024**3, 1),
               finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(OUT, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"\n[METER] {a.label}: 退出码 {proc.returncode}｜耗时 {rec['wall_human']}｜"
          f"峰值 anon {rec['peak_anon_gib']} GiB（净增 {rec['net_anon_gib']}）｜"
          f"计费峰值 {rec['peak_usage_gib']} GiB / 上限 {rec['cgroup_limit_gib']} GiB", flush=True)
    return proc.returncode


if __name__ == '__main__':
    sys.exit(main())
