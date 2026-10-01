"""External research orchestration; each experiment keeps its three-script contract."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time

PROJECT = Path('/root/autodl-fs/model')
AUDIT = PROJECT / 'history/risk_quantile_20260930'
PRIMARY = PROJECT / 'history/diversification_20260930'
ORDER = ['V54','V55','V56','V57']
ENV = dict(os.environ, MX_DATA=str(PRIMARY / 'trainingdata_frozen'), MX_THREADS='2',
           PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
           CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='0')
STOP = False
RUNNING = {}
DONE = []
FAILED = []


def request_stop(signum, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)


def write_status(phase, pending):
    value = dict(pid=os.getpid(), checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                 phase=phase, stopped=STOP, pending=len(pending),
                 running=[dict(task=k, pid=p.pid) for k, (p, _, _) in RUNNING.items()],
                 completed=DONE, failed=FAILED,
                 planned_units=ORDER, planned_folds=64, max_concurrent_folds=4)
    temp = AUDIT / 'batch_status.tmp'
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temp, AUDIT / 'batch_status.json')


def run_phase(phase, tasks, jobs=4):
    global STOP
    pending = list(tasks)
    while pending or RUNNING:
        STOP = STOP or (AUDIT / 'STOP').exists() or (PRIMARY / 'STOP').exists()
        if STOP:
            for p, handle, _ in RUNNING.values():
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                handle.close()
            RUNNING.clear()
            write_status(phase, pending)
            raise SystemExit('stopped by user; partial checkpoints retained')
        while pending and len(RUNNING) < jobs:
            unit, args, label = pending.pop(0)
            d = PROJECT / 'experiments' / unit
            command = [sys.executable, '-B', '-u', str(d / ('analysis.py' if phase == 'analysis' else 'run.py')), *args]
            handle = (AUDIT / f'{unit}_{label}.log').open('a', buffering=1)
            if phase == 'analysis':
                # The required analysis log is inside the closed layout.
                handle.close()
                handle = (d / 'model_logs/analysis_0.log').open('a', buffering=1)
            proc = subprocess.Popen(command, cwd=d, env=ENV, stdout=handle,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            RUNNING[f'{unit}:{label}'] = (proc, handle, time.time())
            print('START', phase, unit, label, proc.pid, flush=True)
        for key, (proc, handle, started) in list(RUNNING.items()):
            if proc.poll() is None:
                continue
            handle.close()
            record = dict(task=key, exit_code=proc.returncode, seconds=round(time.time()-started, 1))
            (FAILED if proc.returncode else DONE).append(record)
            del RUNNING[key]
            print('EXIT', phase, record, flush=True)
            if proc.returncode:
                # Do not launch more models after a new implementation/data failure.
                pending.clear()
        write_status(phase, pending)
        if FAILED and not RUNNING:
            raise SystemExit('batch failed; inspect logs before resuming')
        time.sleep(5)


if __name__ == '__main__':
    assert json.loads((AUDIT / 'preflight.json').read_text())['ok']
    # A live handle, rather than a stale status file, determines whether the GPU owner is active.
    while True:
        if STOP or (AUDIT/'STOP').exists() or (PRIMARY/'STOP').exists():
            write_status('stopped_before_dispatch',[])
            raise SystemExit('stopped by user')
        try:
            state=json.loads((PRIMARY/'batch_status.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            write_status('waiting_for_primary_status',ORDER)
            time.sleep(5)
            continue
        primary_pid=json.loads((PRIMARY/'batch_process.json').read_text())['pid']
        proc_path=Path('/proc')/str(primary_pid)
        live=False
        if proc_path.exists():
            try:
                cmd=(proc_path/'cmdline').read_bytes().split(b'\0')
                live=str(PRIMARY/'innovation_batch_driver.py').encode() in cmd and (proc_path/'stat').read_text().split()[2]!='Z'
            except FileNotFoundError:
                live=False
        if state.get('phase')=='analysis' and not state.get('failed'):
            # Fresh quantile fits do not depend on earlier model weights or audits.
            # CPU-only parent audits can overlap after every GPU training task ends.
            assert all(':analysis' in r['task'] for r in state.get('running', []))
            assert all(len(list((PROJECT/'experiments'/unit/'model_train').glob('*/fold*/complete.json')))==16
                       for unit in ['V46','V47','V48','V49','V50','V51','V52','V53'])
            break
        if state.get('phase')=='complete' and not live:
            for unit in ['V46','V47','V48','V49','V50','V51','V52','V53']:
                record=json.loads((PROJECT/'experiments'/unit/'model_info/final_audit.json').read_text())
                assert record['model_train_read'] and record['folds_checked']==16
            break
        phase='waiting_for_primary_live' if live and not state.get('failed') else 'waiting_for_primary_review'
        write_status(phase,ORDER)
        (AUDIT/'queue_wait.json').write_text(json.dumps(dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            primary_pid=primary_pid,primary_live=live,primary_phase=state.get('phase'),primary_failures=state.get('failed'))))
        time.sleep(15)
    # Clean fold is derived by the preflight predicate, rather than assumed from its number.
    clean = json.loads((AUDIT / 'preflight.json').read_text())['results']['V54']['clean_blocks']
    assert all(f == 4 for _, f in clean)
    # First paired quarter exposes implementation failures before larger jobs are dispatched.
    run_phase('first_clean_quarter', [(u, ['--jobs', '1', '--device', 'cuda', '--quarters', '2025Q3', '--folds', '4'], 'clean_Q3') for u in ORDER])
    run_phase('remaining_clean_quarters', [(u, ['--jobs', '1', '--device', 'cuda', '--quarters', '2025Q4', '2026Q1', '2026Q2', '--folds', '4'], 'clean_remaining') for u in ORDER])
    run_phase('remaining_fixed_folds', [(u, ['--jobs', '1', '--device', 'cuda', '--folds', '1', '2', '3'], 'other_folds') for u in ORDER])
    run_phase('analysis', [(u, ['--audit'], 'analysis') for u in ORDER], jobs=1)
    write_status('complete', [])
    print('BATCH_COMPLETE: new units are research evidence, no release was modified', flush=True)
