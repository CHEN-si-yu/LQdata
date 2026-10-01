"""Read frozen completed folds; disclose only predeclared fixed test strategies."""
from pathlib import Path
import gc
import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import torch

ROOT=Path('/root/autodl-fs/model');A=ROOT/'history/fixed_test_review_20260930'
FROZEN=ROOT/'history/diversification_20260930/trainingdata_frozen'
OUT=A/'units';SCORES=A/'scores';OUT.mkdir(exist_ok=True);SCORES.mkdir(exist_ok=True)
PLAN=json.loads((A/'protocol.json').read_text());UNITS=PLAN['units']
os.nice(12);pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)


def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def stopped():
    return (A/'STOP').exists() or (ROOT/'history/diversification_20260930/STOP').exists()


def atomic(p,value):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(temp,p)


def summarize():
    rows=[read(p) for p in OUT.glob('V*.json')];rows=[r for r in rows if r]
    families={}
    for name,units in PLAN['families'].items():
        part=[r for r in rows if r['unit'] in units]
        primary=[next(a for a in r['accounts'] if a['strategy']!='top1_1d') for r in part]
        values=[a['net_return']*100 for a in primary]
        families[name]=dict(completed_units=[r['unit'] for r in part],planned_units=units,
            all_seeds_complete=len(part)==len(units),primary_net_mean_pct=float(np.mean(values)) if values else None,
            primary_net_seed_sample_std_pp=float(np.std(values,ddof=1)) if len(values)>1 else None,
            primary_max_drawdown_mean_pct=float(np.mean([a['max_drawdown']*100 for a in primary])) if primary else None,
            quarter_seed_means={q:float(np.mean([next(t['return_value'] for t in a['quarters'] if t['quarter']==q)*100 for a in primary]))
                                for q in PLAN['quarters']} if primary else {},
            interpretation='only frozen native primary settings; seed means describe separate accounts, not an executable seed ensemble; overlapping training/validation histories and one shared test market')
    atomic(A/'fixed_test_summary.json',dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed_units=len(rows),planned_units=len(UNITS),
        rows=rows,families=families,scope='fixed242 signal days, saved frozen fit forecasts; official checkpoint reload/layout audits still separately required; no test-based tuning, selection or release'))


done={p.stem for p in OUT.glob('V*.json')}
while len(done)<len(UNITS):
    if stopped():raise SystemExit('stopped by user')
    for unit in UNITS:
        if unit in done:continue
        d=ROOT/'experiments'/unit
        folders=[d/'model_train'/q/f'fold{f}' for q in PLAN['quarters'] for f in range(1,5)]
        if not all((p/'complete.json').exists() for p in folders):continue
        for name in ['model','analysis']:sys.modules.pop(name,None)
        sys.path.insert(0,str(d));import model as m;import analysis as a
        assert m.data_root().resolve()==FROZEN.resolve()
        assert len(a.STRATEGIES)==2 and [s['name'] for s in a.STRATEGIES].count('top1_1d')==1
        print('FIXED_TEST_LOAD',unit,flush=True)
        panel=m.Panel(m.RECIPE['features'],load_x=False);px=m.Prices(panel)
        components=[];dates=None;hashes={};split_evidence=[]
        for f in range(1,5):
            chunks=[];fd=[]
            for q in PLAN['quarters']:
                p=d/'model_train'/q/f'fold{f}';c=read(p/'complete.json');assert c and c['quarter']==q and c['fold']==f
                test=np.load(p/'test_predictions.npy');score=np.load(p/'score_predictions.npy')
                assert test.shape==(len(c['test_dates']),len(panel.codes))
                assert np.array_equal(test,score[:len(test)],equal_nan=True)
                sp=m.splits(panel.days,quarter=q)[f-1]
                assert np.array_equal(panel.days[sp['test']],c['test_dates'])
                # Every training/validation label is purged before that quarter's test.
                assert sp['train'].max()<sp['test'].min() and sp['valid'].max()<sp['test'].min()
                chunks.append(test);fd.extend(c['test_dates'])
                hashes[f'{q}_fold{f}']=hashlib.sha256((p/'test_predictions.npy').read_bytes()).hexdigest()
                split_evidence.append(dict(quarter=q,fold=f,train_max=str(panel.days[sp['train'].max()]),
                    valid_max=str(panel.days[sp['valid'].max()]),test_start=c['test_dates'][0],test_end=c['test_dates'][-1]))
            assert fd==sorted(set(fd))
            if dates is None:dates=fd
            else:assert dates==fd
            components.append(np.concatenate(chunks))
        pred=sum(components).astype(np.float32);days=np.searchsorted(panel.days,dates)
        pred,days=a.evaluation_slice(pred,panel,days);assert len(days)==242
        label_proof=px.verify_labels(panel,days)
        path=SCORES/f'{unit}_242.npz';temp=path.with_suffix('.tmp')
        with temp.open('wb') as handle:np.savez_compressed(handle,scores=pred,days=days,dates=panel.days[days])
        os.replace(temp,path)
        accounts=[]
        for spec in a.STRATEGIES:
            args=dict(n=spec['topn'],period=spec['period'],band=spec['band'])
            if 'forecast_members' in inspect.signature(a.cash_backtest).parameters:args['forecast_members']=4
            st,curve,trades=a.cash_backtest(pred,panel,px,days,**args)
            assert abs(st['gross_return_same_positions']-st['fee_drag']-st['slippage_drag']-st['return_value'])<1e-10
            eq=np.asarray([r['equity'] for r in curve]);daily=eq/np.r_[100000.,eq[:-1]]-1
            quarterly=a.quarter_metrics('ensemble',spec['name'],curve,trades,pred,panel,days)
            assert abs(np.prod([1+t['return_value'] for t in quarterly])-1-st['return_value'])<1e-10
            accounts.append(dict(strategy=spec['name'],frozen_settings=args,net_return=st['return_value'],max_drawdown=st['max_drawdown'],
                sharpe=st['sharpe'],trades=st['trades'],total_fees=st['total_fees'],fee_drag=st['fee_drag'],slippage_drag=st['slippage_drag'],
                gross_return_same_positions=st['gross_return_same_positions'],avg_exposure=st['avg_exposure'],
                net_daily_mean_annual_pct=float(daily.mean()*242*100),quarters=quarterly,
                execution_start=curve[0]['date'],execution_end=curve[-1]['date']))
            np.savez_compressed(SCORES/f'{unit}_{spec["name"]}_cash.npz',equity=eq,dates=np.asarray([r['date'] for r in curve]))
        origin='full parent checkpoint reuse, strategy-only' if unit in PLAN['strategy_only_units'] else 'fresh fitted weights'
        value=dict(unit=unit,recipe_family=m.RECIPE.get('family'),seed=m.RECIPE['seed'],weight_origin=origin,
            test_window=[str(panel.days[days[0]]),str(panel.days[days[-1]])],signal_days=len(days),
            ensemble='raw four-fold sum, no normalization; calibrated absolute cost gate uses forecast_members4',
            source_hashes={name:hashlib.sha256((d/name).read_bytes()).hexdigest() for name in ['model.py','analysis.py','run.py']},
            checkpoint_forecast_hashes=hashes,split_evidence=split_evidence,label_price_check=label_proof,IC=a.ic_table(pred,panel,days),
            accounts=accounts,official_audit=read(m.info_root()/'final_audit.json'),
            scope='reconstruct only fixed test from saved frozen training forecasts; no new inference or official checkpoint/layout audit claim; no test-based adjustment or release')
        atomic(OUT/f'{unit}.json',value);done.add(unit);summarize()
        print('FIXED_TEST_RECORDED',unit,[(a['strategy'],a['net_return'],a['max_drawdown']) for a in accounts],flush=True)
        sys.path.pop(0);del panel,px,pred,components,score,test,m,a;gc.collect()
    # Attach the subsequent official audits without recomputing test accounts.
    for unit in list(done):
        path=OUT/f'{unit}.json';v=read(path)
        p=ROOT/'experiments'/unit/'model_info/final_audit.json'
        if not v.get('official_audit') and p.exists():
            v['official_audit']=read(p);atomic(path,v)
    summarize()
    atomic(A/'status.json',dict(pid=os.getpid(),checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed=sorted(done),planned=UNITS,
        phase='complete' if len(done)==len(UNITS) else 'waiting_for_completed_folds'))
    if len(done)<len(UNITS):time.sleep(20)
print('ALL_FIXED_TEST_STRATEGIES_RECORDED',flush=True)
