"""One finite documentation task; does not create or train any research candidate."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path('/root/autodl-fs/model');H=ROOT/'history';A=H/'research_closure_20260930'
JOINT=H/'flow_technical_joint_20260930';BASE=H/'fixed_test_review_20260930'
UNITS=[f'V{u}' for u in range(46,70)]
ENV=dict(os.environ,MX_DATA=str(H/'diversification_20260930/trainingdata_frozen'),MX_THREADS='2',PYTHONDONTWRITEBYTECODE='1')


def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def status(phase,remaining=None,**extra):
    v=dict(pid=os.getpid(),checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),phase=phase,
        remaining=remaining,new_research_allowed=False,scope='finish already planned audit and documentation only',**extra)
    p=A/'finalizer_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));os.replace(t,p)


def cohort_live(cohort):
    live=[]
    for name in ['batch_status.json','status.json','batch_process.json','validation0_process.json','validation1_process.json','fixed_test_process.json','quantile_validation_lab_process.json','quantile_strategy_lab_process.json']:
        v=read(H/cohort/name)
        if not v or not v.get('pid'):continue
        p=Path('/proc')/str(v['pid'])
        if p.exists():
            try:
                if (p/'stat').read_text().split()[2]!='Z' and cohort in (p/'cmdline').read_bytes().decode(errors='replace'):live.append(v['pid'])
            except FileNotFoundError:pass
    return live


while True:
    if (A/'STOP').exists():status('stopped_before_archive');raise SystemExit('stopped')
    pending=[]
    for u in UNITS:
        audit=read(ROOT/'experiments'/u/'model_info/final_audit.json')
        result=(JOINT/'fixed_test_review/units' if int(u[1:])>=66 else BASE/'units')/(u+'.json')
        if not audit or not audit.get('model_train_read') or audit.get('folds_checked')!=16 or not result.exists():pending.append(u)
    for cohort in ['risk_quantile_20260930','flow_technical_joint_20260930']:
        s=read(H/cohort/'batch_status.json')
        if s and s.get('failed'):
            status('requires_implementation_review',pending,cohort=cohort,failures=s['failed']);raise SystemExit('finite batch failure; no further dispatch')
    grids=len(list((JOINT/'clean_cash').glob('V*.json'))) if (JOINT/'clean_cash').exists() else 0
    live=sum([cohort_live(c) for c in ['risk_quantile_20260930','flow_technical_joint_20260930','fixed_test_review_20260930']],[])
    if not pending and grids==16 and not live:break
    status('waiting_for_existing_tasks',pending,fusion_grid_blocks=grids,live_cohort_pids=live)
    time.sleep(20)
status('writing_final_documentation',[])
log=(A/'final_archive.log').open('a',buffering=1)
proc=subprocess.run([sys.executable,'-B',str(A/'collect_research_closure.py'),'--final'],env=ENV,cwd=A,stdout=log,stderr=subprocess.STDOUT)
log.close()
if proc.returncode:
    status('requires_archive_review',[],archive_exit_code=proc.returncode);raise SystemExit(proc.returncode)
receipt=read(A/'completion_receipt.json');assert receipt and receipt['status']=='completed_and_stopped'
status('complete',[],completion_receipt=str(A/'completion_receipt.json'))
print('PLANNED_TASKS_AND_DOCUMENTATION_COMPLETE; no new research will be dispatched',flush=True)
