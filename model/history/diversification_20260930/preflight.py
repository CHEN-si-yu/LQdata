from pathlib import Path
import os,sys,json,importlib.util,time
import numpy as np
import torch
from scipy.stats import rankdata
r=Path('/root/autodl-fs/model')
results={}
for v in ['V46','V50']:
 d=r/'experiments'/v
 for name in ['model','analysis']:sys.modules.pop(name,None)
 sys.path.insert(0,str(d))
 import model as m
 import analysis as a
 print('IMPORTED',v,flush=True)
 n=len(m.feature_columns(m.metadata()));assert len(m.fixed_files())==159
 m.validate_layout(allow_missing=True)
 days,codes=m.axis();assert days[-1]=='2026-09-29'
 blocks=[]
 for q in m.LAYOUT_QUARTERS:
  for sp in m.splits(days,quarter=q):
   assert np.intersect1d(sp['train'],sp['valid']).size==0
   assert max(sp['train'].max(),sp['valid'].max())+m.RECIPE['purge_horizon']+1 < sp['test'][0]
   if sp['train'].max()<sp['valid'].min():blocks.append((q,sp['fold']))
 assert len(blocks)==4
 torch.manual_seed(11)
 net=m.PredictModel(n,0).eval()
 x=torch.randn(20,n) if v=='V46' else torch.randn(20,8,n)
 cpu=net(x).detach().numpy();single=net(x[:3]).detach().numpy()
 assert np.allclose(cpu[:3],single,atol=2e-5,rtol=2e-5)
 gpu=net.to('cuda')(x.to('cuda')).detach().cpu().numpy()
 assert np.allclose(cpu,gpu,atol=2e-5,rtol=2e-5)
 if v=='V50':
  toy=object.__new__(m.Panel);toy.X=np.random.default_rng(7).normal(size=(20,5,n)).astype(np.float32)
  mask=np.array([True,False,True,True,False]);before=toy.inputs(10,mask).copy()
  assert before.shape==(3,8,n) and np.array_equal(before[:,0],toy.X[3,mask])
  toy.X[11:]=100000
  assert np.array_equal(before,toy.inputs(10,mask))
 print('NUMERICAL',v,n,'CLEAN',blocks,flush=True)
 results[v]=dict(features=n,fixed_files=159,clean_blocks=blocks,gpu_cpu_max_error=float(np.max(np.abs(cpu-gpu))),stock_subset_inference=True,causal_future_invariance=v=='V50')
 sys.path.pop(0)
out=r/'history/diversification_20260930/preflight.json'
out.write_text(json.dumps(dict(ok=True,results=results,checked_at=time.strftime('%Y-%m-%d %H:%M:%S')),indent=2))
print('PREFLIGHT_OK',flush=True)
