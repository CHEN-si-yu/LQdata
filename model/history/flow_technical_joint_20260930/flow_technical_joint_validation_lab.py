"""Evaluate the previously frozen validation grid for fresh composite models only."""
from pathlib import Path
import argparse
import gc
import inspect
import json
import os
import sys
import time
import numpy as np
import pyarrow as pa
import torch
from scipy.stats import rankdata,spearmanr

ROOT=Path('/root/autodl-fs/model');A=ROOT/'history/flow_technical_joint_20260930'
parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,choices=[0,1],required=True)
worker=parser.parse_args().worker
units=['V66','V67','V68','V69'][worker::2];quarters=['2025Q3','2025Q4','2026Q1','2026Q2']
os.nice(12);pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
out=A/'clean_cash';out.mkdir(exist_ok=True)
forecasts=A/'clean_forecasts';forecasts.mkdir(exist_ok=True)
completed=[]


def atomic(p,value):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(tmp,p)


def guard():
    if (A/'STOP').exists():raise SystemExit('stopped; partial research records retained')


while len(completed)<8:
    guard()
    for unit in units:
        done_files={q:out/f'{unit}_{q}.json' for q in quarters}
        completed=sorted({*completed,*[f'{unit}_{q}' for q in quarters if done_files[q].exists()]})
        ready=[q for q in quarters if not done_files[q].exists() and (ROOT/'experiments'/unit/'model_train'/q/'fold4/complete.json').exists()]
        if not ready:continue
        for name in ['model','analysis']:sys.modules.pop(name,None)
        d=ROOT/'experiments'/unit;sys.path.insert(0,str(d));import model as m;import analysis as a
        torch.set_num_threads(1)
        panel=m.Panel(m.RECIPE['features']);px=m.Prices(panel)
        source=inspect.getsource(a.cash_backtest)
        assert source.count('gate_scores=None):')==1
        source=source.replace('gate_scores=None):','gate_scores=None,phase=0):',1)
        source=source.replace('% period == 0','% period == phase')
        space=dict(a.__dict__);exec(compile(source,'predeclared_phase_sensitivity','exec'),space);cash=space['cash_backtest']
        for q in ready:
            guard();sp=m.splits(panel.days,quarter=q)[3];days=sp['valid']
            assert sp['train'].max()+m.RECIPE['purge_horizon']+1<days.min()
            ck=torch.load(d/'model_train'/q/'fold4/best.pt',map_location='cpu',weights_only=False)
            assert ck['recipe_signature']==m.recipe_signature()
            net=m.PredictModel(len(panel.features),panel.market_dim);net.load_state_dict(ck['model']);net.ridge=ck.get('ridge');net.eval()
            heads=m.predict_heads(net,panel,days,torch.device('cpu'))
            gate=m.predict_absolute(net,panel,days,torch.device('cpu'))
            assert np.allclose(gate,heads[:,:,3],atol=2e-7,rtol=2e-5,equal_nan=True)
            f=np.full(heads.shape[:2],np.nan,np.float64);t=np.full_like(f,np.nan)
            correlations=[]
            for i,row in enumerate(heads):
                known=np.isfinite(row).all(axis=1);n=known.sum()
                if n<2:continue
                f[i,known]=(rankdata(row[known,1])-.5)/n;t[i,known]=(rankdata(row[known,3])-.5)/n
                correlations.append(float(spearmanr(row[known,1],row[known,3]).statistic))
            native_score=m.score_of(heads)
            assert np.array_equal(native_score,(.5*f+.5*t).astype(np.float32),equal_nan=True)
            native=a.cash_backtest(native_score,panel,px,days,n=5,period=5,band=40,forecast_members=1,gate_scores=gate)
            primary=cash(native_score,panel,px,days,n=5,period=5,band=40,forecast_members=1,gate_scores=gate,phase=0)
            assert native==primary
            rows=[];started=time.time()
            for weight in [0.,.25,.5,.75,1.]:
                scores=(weight*f+(1-weight)*t).astype(np.float32)
                for n in [5,10,20]:
                    for period in [1,5]:
                        for phase in range(period):
                            for gate_multiplier in [0.,1.]:
                                guard();st,_,_=cash(scores,panel,px,days,n=n,period=period,phase=phase,band=40,forecast_members=1,cost_gate_multiplier=gate_multiplier,gate_scores=gate)
                                assert abs(st['gross_return_same_positions']-st['fee_drag']-st['slippage_drag']-st['return_value'])<1e-10
                                rows.append(dict(flow_weight=weight,topn=n,period=period,phase=phase,band=40,cost_gate_multiplier=gate_multiplier,
                                    net_return=st['return_value'],max_drawdown=st['max_drawdown'],sharpe=st['sharpe'],trades=st['trades'],
                                    avg_exposure=st['avg_exposure'],fee_drag=st['fee_drag'],slippage_drag=st['slippage_drag']))
            assert len(rows)==180
            np.savez_compressed(forecasts/f'{unit}_{q}_fold4.npz',scores=native_score,absolute_1d=gate,heads=heads,days=days,dates=panel.days[days])
            value=dict(unit=unit,seed=m.RECIPE['seed'],quarter=q,fold=4,days=len(days),start=str(panel.days[days[0]]),end=str(panel.days[days[-1]]),
                rows=rows,native_primary=native[0],mean_internal_component_rank_correlation=float(np.mean(correlations)),
                clean_training_before_validation=True,phase_zero_native_exact=True,absolute_raw_head_rederived=True,
                recipe_signature=m.recipe_signature(),seconds=round(time.time()-started,2),
                scope='selected-epoch clean validation, four overlapping windows; all frozen candidates retained, no test selection or new grid')
            atomic(done_files[q],value);completed=sorted({*completed,f'{unit}_{q}'})
            atomic(A/f'validation_worker{worker}_status.json',dict(pid=os.getpid(),worker=worker,completed=completed,planned=8,checked_at=time.strftime('%Y-%m-%d %H:%M:%S')))
            print('FRESH_FUSION_CLEAN_DONE',unit,q,len(rows),flush=True)
        sys.path.pop(0);del panel,px,net,m,a,heads,f,t,ck;gc.collect()
    state=json.loads((A/'batch_status.json').read_text()) if (A/'batch_status.json').exists() else {}
    if state.get('failed'):raise SystemExit('model batch failed; retain existing measurements')
    if len(completed)<8:time.sleep(20)
print('ALL_FROZEN_FUSION_VALIDATION_CASES_COMPLETE',worker,flush=True)
