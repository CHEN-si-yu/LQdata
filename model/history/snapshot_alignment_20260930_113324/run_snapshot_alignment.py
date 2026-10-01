import os,sys,json,time,subprocess,traceback,hashlib
from pathlib import Path
from datetime import datetime
root=Path('/root/autodl-fs/model')
audit=Path(__file__).resolve().parent
stage=audit/'new_trainingdata'
status=audit/'status.json'
def save(x):
 tmp=status.with_suffix('.tmp')
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2))
 os.replace(tmp,status)
def identities():
 out={}
 for name in ['factors','market_factors']:
  base=root.parent/'featureengineering/data'/name
  for p in base.glob('*/year=*/data.parquet'):
   st=p.stat();out[str(p)]=[st.st_size,st.st_mtime_ns]
 return out
started=time.time()
save({'phase':'starting','started_at':datetime.now().isoformat(),'stage':str(stage)})
try:
 sys.path.insert(0,str(root))
 import preparingdata as prep
 cfg=prep.Cfg()
 cutoff=prep.last_upstream_day(cfg)
 sc=prep.scan(cfg,jobs=4)
 assert sc['span'][1] <= cutoff,(sc['span'],cutoff)
 before=identities()
 (audit/'source_before.json').write_text(json.dumps(before,indent=2))
 (audit/'original_meta.json').write_text((root/'trainingdata/meta.json').read_text())
 save({'phase':'building','cutoff':cutoff,'stage':str(stage),'source_partitions':len(before),'seconds':round(time.time()-started,1)})
 args=[sys.executable,'-B',str(root/'preparingdata.py'),'--full','--out',str(stage),'--jobs','4']
 result=subprocess.run(args,cwd=str(root))
 if result.returncode:
  raise RuntimeError('Snapshot build failed: '+str(result.returncode))
 meta=json.loads((stage/'meta.json').read_text())
 assert meta['axis']['end']==cutoff,(meta['axis'],cutoff)
 assert not meta.get('partial')
 save({'phase':'checking','cutoff':cutoff,'stage':str(stage),'seconds':round(time.time()-started,1)})
 check=prep.check(prep.Cfg(trainingdata=stage))
 report={k:check[k] for k in ['ok','problems','years']}
 (audit/'check_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 after=identities()
 (audit/'source_after.json').write_text(json.dumps(after,indent=2))
 assert before==after,'Source partitions changed during snapshot build'
 save({'phase':'built_and_checked','cutoff':cutoff,'stage':str(stage),'check_ok':report['ok'],'check_problems':report['problems'],'source_stable':True,'feature_count':len(meta['columns']['features']),'labels':meta['columns']['labels'],'market_factor_count':meta['market_factors']['n_factors'],'axis':{k:v for k,v in meta['axis'].items() if k!='codes'},'seconds':round(time.time()-started,1)})
except BaseException as exc:
 traceback.print_exc()
 save({'phase':'failed','error':str(exc),'seconds':round(time.time()-started,1)})
 sys.exit(1)
