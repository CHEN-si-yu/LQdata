"""Finish only the four previously designed fresh fusion units in the closure manifest."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

ROOT=Path('/root/autodl-fs/model')
A=ROOT/'history/flow_technical_joint_20260930'
Q=ROOT/'history/risk_quantile_20260930'
PRIMARY=ROOT/'history/diversification_20260930'
UNITS=['V66','V67','V68','V69']
ENV=dict(os.environ,MX_DATA=str(PRIMARY/'trainingdata_frozen'),MX_THREADS='2',
         PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0')
running={};completed=[];failed=[];stop=False


def request_stop(*args):
    global stop
    stop=True


signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)


def guard():
    global stop
    stop=stop or (A/'STOP').exists() or (PRIMARY/'STOP').exists()
    if stop:
        for p,h,_ in running.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            h.close()
        raise SystemExit('stopped; partial own checkpoints preserved')


def write(phase,pending):
    value=dict(pid=os.getpid(),checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),phase=phase,
        pending=len(pending),running=[dict(task=k,pid=v[0].pid) for k,v in running.items()],
        completed=completed,failed=failed,planned_units=UNITS,planned_folds=64,new_research_allowed=False,max_concurrent_gpu_folds=4)
    tmp=A/'batch_status.tmp';tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,A/'batch_status.json')


def verify_sources():
    plan=json.loads((A/'protocol.json').read_text())
    for unit in UNITS:
        for name,sha in plan['source_sha256'][unit].items():
            assert hashlib.sha256((ROOT/'experiments'/unit/name).read_bytes()).hexdigest()==sha,(unit,name)


def phase(name,tasks,jobs):
    pending=list(tasks)
    while pending or running:
        guard()
        while pending and len(running)<jobs and not failed:
            unit,script,args,label=pending.pop(0);verify_sources()
            d=ROOT/'experiments'/unit
            log=A/f'{unit}_{label}.log'
            if name=='analysis':log=d/'model_logs/analysis_0.log'
            log.parent.mkdir(exist_ok=True)
            h=log.open('a',buffering=1)
            p=subprocess.Popen([sys.executable,'-B','-u',str(d/script),*args],cwd=d,env=ENV,
                stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            running[f'{unit}:{label}']=(p,h,time.time());print('START',name,unit,label,p.pid,flush=True)
        for key,(p,h,t) in list(running.items()):
            if p.poll() is None:continue
            h.close();r=dict(task=key,exit_code=p.returncode,seconds=round(time.time()-t,1))
            (failed if p.returncode else completed).append(r);del running[key]
            print('EXIT',name,json.dumps(r),flush=True)
            if p.returncode:pending.clear()
        write(name,pending)
        if failed and not running:raise SystemExit('implementation/data failure; no further dispatch')
        time.sleep(5)


if __name__=='__main__':
    assert json.loads((ROOT/'history/research_closure_20260930/scope_manifest.json').read_text())['remaining']['fusion_units']==UNITS
    assert json.loads((A/'preflight.json').read_text())['ok']
    verify_sources()
    # Fresh closed-form branches fit on CPU while the existing quantile GPU jobs finish.
    phase('prepare_absolute_components',[(u,'model.py',['--device','cpu','--prepare-absolute-only'],'absolute_prepare') for u in UNITS],1)
    while True:
        guard();state=json.loads((Q/'batch_status.json').read_text())
        counts=[len(list((ROOT/'experiments'/u/'model_train').glob('*/fold*/complete.json'))) for u in ['V54','V55','V56','V57']]
        if state.get('phase') in ['analysis','complete'] and not state.get('failed') and counts==[16]*4:break
        if state.get('failed'):raise SystemExit('prior quantile batch failed; finite queue retained for review')
        write('waiting_for_existing_quantile_gpu_training',[]);time.sleep(15)
    phase('first_clean_quarter',[(u,'run.py',['--jobs','1','--device','cuda','--quarters','2025Q3','--folds','4'],'clean_Q3') for u in UNITS],4)
    phase('remaining_clean_quarters',[(u,'run.py',['--jobs','1','--device','cuda','--quarters','2025Q4','2026Q1','2026Q2','--folds','4'],'clean_remaining') for u in UNITS],4)
    phase('remaining_fixed_folds',[(u,'run.py',['--jobs','1','--device','cuda','--folds','1','2','3'],'other_folds') for u in UNITS],4)
    phase('analysis',[(u,'analysis.py',['--audit'],'analysis') for u in UNITS],1)
    for u in UNITS:
        r=json.loads((ROOT/'experiments'/u/'model_info/final_audit.json').read_text())
        assert r['model_train_read'] and r['folds_checked']==16
    write('complete',[]);print('FINITE_FUSION_COHORT_COMPLETE; no further research dispatch',flush=True)
