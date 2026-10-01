"""Describe information-family redundancy on matching frozen validation dates."""
from pathlib import Path
import itertools
import json
import os
import time

import numpy as np
from scipy.stats import rankdata,spearmanr

PROJECT=Path('/root/autodl-fs/model');PRIMARY=PROJECT/'history/diversification_20260930'
AUDIT=PROJECT/'history/absolute_ridge_20260930';OUT=AUDIT/'family_correlations';OUT.mkdir(exist_ok=True)
os.nice(15)
FAMILIES=dict(legacy=['V31','V36','V38'],financial=['V46','V47','V48','V49'],flow=['V50','V51','V52','V53'],technical_absolute=['V62','V63','V64','V65'])


def percentile(scores):
    out=np.full_like(scores,np.nan)
    for i,row in enumerate(scores):
        known=np.isfinite(row);n=known.sum()
        if n>1:out[i,known]=(rankdata(row[known])-.5)/n
    return out


def compare(p,q):
    assert p.shape==q.shape
    corr=[];overlap=[]
    for a,b in zip(p,q):
        ok=np.isfinite(a)&np.isfinite(b)
        if ok.sum()<50:continue
        corr.append(float(spearmanr(a[ok],b[ok]).statistic))
        def top(x):
            ii=np.flatnonzero(np.isfinite(x));return set(ii[np.argsort(-x[ii],kind='stable')[:5]])
        overlap.append(len(top(a)&top(b))/5)
    return dict(daily_rank_correlation=float(np.mean(corr)),top5_overlap=float(np.mean(overlap)),days=len(corr))


completed={p.stem for p in OUT.glob('202*Q*.json')}
while len(completed)<4:
    if (AUDIT/'STOP').exists() or (PRIMARY/'STOP').exists():raise SystemExit('stopped by user')
    for q in ['2025Q3','2025Q4','2026Q1','2026Q2']:
        if q in completed:continue
        try:ready=json.loads((AUDIT/'clean_absolute_results.json').read_text())['rows']
        except (FileNotFoundError,json.JSONDecodeError):continue
        if {r['unit'] for r in ready if r['quarter']==q}!=set(FAMILIES['technical_absolute']):continue
        paths={}
        for family,units in FAMILIES.items():
            folder=(AUDIT/'clean_absolute') if family=='technical_absolute' else PRIMARY/('clean_references' if family=='legacy' else 'clean_innovations')
            paths[family]=[folder/f'{u}_{q}_fold4.npz' for u in units]
        if not all(p.exists() for files in paths.values() for p in files):continue
        scores={};days=None;dates=None
        for family,files in paths.items():
            scores[family]=[]
            for p in files:
                data=np.load(p)
                if days is None:days=data['days'];dates=data['dates']
                assert np.array_equal(days,data['days'])
                scores[family].append(data['scores'])
        # Rank families preserve the earlier frozen percentile-aggregation convention.
        # Absolute forecasts remain in return units; average all four sampling seeds.
        aggregate={family:sum(values)/len(values) if family=='technical_absolute' else sum(percentile(v) for v in values)/len(values)
                   for family,values in scores.items()}
        across={f'{x}__{y}':compare(aggregate[x],aggregate[y]) for x,y in itertools.combinations(FAMILIES,2)}
        within={family:{f'{FAMILIES[family][i]}__{FAMILIES[family][j]}':compare(values[i],values[j])
                for i,j in itertools.combinations(range(len(values)),2)} for family,values in scores.items()}
        cash={};cashdates=None
        for family,units in FAMILIES.items():
            for unit in units:
                if family=='technical_absolute':p=AUDIT/'clean_absolute'/f'{unit}_{q}_cash_gate1.0.npz'
                else:
                    period=5 if family=='financial' else 1
                    folder=PRIMARY/('clean_references' if family=='legacy' else 'clean_innovations')
                    p=folder/f'{unit}_{q}_cash_{period}d.npz'
                if not p.exists():break
                data=np.load(p)
                if cashdates is None:cashdates=data['dates']
                assert np.array_equal(cashdates,data['dates'])
                eq=data['equity'];cash[unit]=eq/np.r_[100000.,eq[:-1]]-1
        if len(cash)!=sum(map(len,FAMILIES.values())):continue
        cashcorr={f'{x}__{y}':float(np.corrcoef(cash[x],cash[y])[0,1]) for x,y in itertools.combinations(cash,2)}
        value=dict(quarter=q,start=str(dates[0]),end=str(dates[-1]),days=len(days),families=FAMILIES,
            aggregate_rank_pairs=across,within_family_rank_pairs=within,individual_cash_return_correlations=cashcorr,
            score_aggregation='legacy/financial/flow each equal daily percentile mean; technical absolute equal raw forecast mean; every seed retained',
            cash_scope='correlation of separately executed native primary accounts, not portfolio profit or a combined account; financial weekly, others daily; technical cost gate1',
            interpretation='overlapping selected-stage validation diagnostics; low rank correlation alone is not a release criterion or stable profit proof')
        target=OUT/f'{q}.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));os.replace(temp,target)
        completed.add(q);print('FAMILY_CORRELATIONS_DONE',q,across,flush=True)
    status=AUDIT/'family_correlation_status.json';temp=status.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed=sorted(completed),planned=4),indent=2));os.replace(temp,status)
    try:state=json.loads((AUDIT/'batch_status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):state=None
    if state and state.get('failed'):raise SystemExit('training failed; retain correlations')
    if len(completed)<4:time.sleep(20)
print('ALL_FAMILY_CORRELATION_WINDOWS_COMPLETE',flush=True)
