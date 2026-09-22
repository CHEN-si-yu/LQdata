"""best：实战组的一键编排 —— 9 组 × 4 折 = 36 个训练任务，训完直接集成出票。

`train.sh` 用 `memory_guard.py` 包住本脚本，所以内存闸门看的是**整个进程组**。
并发上限由 `--jobs` 控制（默认 6：实测单折 ≈8.4 GiB anon，6 并发 ≈50 GiB，稳在 84 GiB 闸门内）。

★ 训练与分析**串行**：分析（集成 + 门槛 + 出票）放在全部 36 个任务之后，不并行。

    $PY model.py --quarter s3301 --fold 1     # 单折（调试/续跑）
    $PY run.py --jobs 6                       # 全量（train.sh 走这条）
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import model as M  # noqa: E402  —— 复用 PROD_GROUPS，避免组名两处维护

FOLDS = (1, 2, 3, 4)


def tasks():
    """训练任务清单：**组在前、折在后** —— 同一组四折容易被调度在一起，便于早看结果。"""
    return [(g, f) for g in M.PROD_GROUPS for f in FOLDS]


def done_path(group, fold):
    return ROOT / 'model_train' / group / f'fold{fold}' / 'complete.json'


def main():
    ap = argparse.ArgumentParser(description='best：36 个训练任务 → 集成出票')
    ap.add_argument('--jobs', type=int, default=6, help='并发训练任务数（默认 6）')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--skip-analysis', action='store_true', help='只训练，不出票')
    ap.add_argument('--force', action='store_true', help='忽略已完成记录，重训全部')
    args = ap.parse_args()

    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    todo = [(g, f) for g, f in tasks() if args.force or not done_path(g, f).exists()]
    skipped = len(tasks()) - len(todo)
    print(f'任务总数 {len(tasks())}，已完成 {skipped}，本次待跑 {len(todo)}，并发 {args.jobs}', flush=True)
    if not todo:
        print('全部已完成 —— 直接进入集成出票', flush=True)

    t0 = time.time()
    running, results, failed = {}, [], []
    queue = list(todo)
    while queue or running:
        while queue and len(running) < args.jobs:
            group, fold = queue.pop(0)
            log = (logs / f'{group}_fold{fold}.log').open('w')
            cmd = [sys.executable, '-u', str(ROOT / 'model.py'),
                   '--quarter', group, '--fold', str(fold), '--device', args.device]
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT), start_new_session=True)
            running[p] = (group, fold, log, time.time())
            print(f'▶ {group} fold{fold} 启动（在跑 {len(running)}）', flush=True)
        time.sleep(2)
        for p in list(running):
            if p.poll() is None:
                continue
            group, fold, log, t_start = running.pop(p)
            log.close()
            secs = time.time() - t_start
            if p.returncode != 0:
                failed.append((group, fold, p.returncode))
                print(f'✗ {group} fold{fold} 失败 rc={p.returncode}（{secs:.0f}s）'
                      f'，日志 {log.name}', flush=True)
            else:
                results.append((group, fold, secs))
                print(f'✓ {group} fold{fold} 完成（{secs:.0f}s），剩 {len(queue) + len(running)}', flush=True)

    print(f'\n训练结束：成功 {len(results)}，失败 {len(failed)}，'
          f'墙钟 {(time.time() - t0) / 60:.1f} 分钟', flush=True)
    if failed:
        # 不静默继续：任务缺失会让集成少一组，出票结果就不是九组口径了。
        raise SystemExit(f'有 {len(failed)} 个任务失败：{failed}；修好后续跑即可（已完成的不重训）')

    if args.skip_analysis:
        return 0

    print('\n=== 全部训练完成，进入集成 + 门槛 + 出票 ===', flush=True)
    rc = subprocess.call([sys.executable, '-u', str(ROOT / 'analysis.py')], cwd=str(ROOT))
    if rc != 0:
        raise SystemExit(f'analysis.py 退出码 {rc}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
