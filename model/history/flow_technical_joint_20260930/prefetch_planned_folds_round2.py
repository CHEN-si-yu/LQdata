"""Prefetch two existing planned folds only; hand off before the core queue reaches 2026Q2."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
R=Path('/root/autodl-fs/model');A=R/'history/flow_technical_joint_20260930'
plan=json.loads((A/'protocol.json').read_text());jobs={}
env=dict(os.environ,MX_DATA=str(R/'history/diversification_20260930/trainingdata_frozen'),MX_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0')
for u in ['V67','V69']:
 d=R/'experiments'/u
 for name,digest in plan['source_sha256'][u].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==digest
 assert not (d/'model_train/2026Q2/fold2/complete.json').exists()
 # Hand off immediately if the core queue enters Q2; no same-fold writers.
 assert not (d/'model_train/2026Q2/fold1/complete.json').exists()
 h=(d/'model_logs/2026Q2_2.log').open('a',buffering=1)
 p=subprocess.Popen([sys.executable,'-B','-u',str(d/'model.py'),'--quarter','2026Q2','--fold','2','--device','cuda'],cwd=d,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
 jobs[u]=(p,h)
receipt=dict(pid=os.getpid(),started_at=time.strftime('%Y-%m-%d %H:%M:%S'),new_research_allowed=False,scope='two already planned 2026Q2 fold2 fits, unchanged recipes and 64-fit scope',maximum_training_processes=6,cpu_threads_per_fit=2,targets={u:p.pid for u,(p,h) in jobs.items()},completed={})
(A/'prefetch2_process.json').write_text(json.dumps(receipt,indent=2))
while jobs:
 s=json.loads((A/'batch_status.json').read_text())
 stop=bool(s.get('failed')) or (A/'STOP').exists() or (R/'history/diversification_20260930/STOP').exists()
 cmds=[]
 for x in Path('/proc').iterdir():
  if not x.name.isdigit():continue
  try:cmds.append((int(x.name),(x/'cmdline').read_bytes().split(b'\0')))
  except (FileNotFoundError,ProcessLookupError):pass
 for u,(p,h) in list(jobs.items()):
  approaching=any(pid!=p.pid and str(R/'experiments'/u/'model.py').encode() in args and b'2026Q2' in args for pid,args in cmds)
  if p.poll() is None and (stop or approaching):
   os.killpg(p.pid,signal.SIGTERM);p.wait();outcome='own atomic checkpoint handed back to existing queue before Q2 begins'
  elif p.poll() is None:continue
  else:outcome='completed' if p.returncode==0 else 'failed; existing frozen queue may resume own checkpoint'
  h.close();receipt['completed'][u]=dict(exit_code=p.returncode,outcome=outcome,complete=(R/'experiments'/u/'model_train/2026Q2/fold2/complete.json').exists());del jobs[u]
  receipt['checked_at']=time.strftime('%Y-%m-%d %H:%M:%S')
  (A/'prefetch2_process.json').write_text(json.dumps(receipt,indent=2))
 time.sleep(2)
receipt['phase']='complete';receipt['ended_at']=time.strftime('%Y-%m-%d %H:%M:%S')
(A/'prefetch2_process.json').write_text(json.dumps(receipt,indent=2))
print('FINITE_PREFETCH_COMPLETE',json.dumps(receipt),flush=True)
