"""Distinguish raw equal-score legacy controls from rank-normalized family controls."""
from pathlib import Path
import inspect
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
from scipy.stats import rankdata
import torch

ROOT=Path('/root/autodl-fs/model')
AUDIT=ROOT/'history/diversification_20260930'
OUT=AUDIT/'matched_legacy_controls'
OUT.mkdir(exist_ok=True)
pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
sys.path.insert(0,str(ROOT/'experiments/V46'))
import model as m
import analysis as a
source=inspect.getsource(a.cash_backtest)
assert source.count('money=None):')==1 and source.count('% period == 0')==2
source=source.replace('money=None):','money=None, phase=0):',1).replace('% period == 0','% period == phase')
space=dict(a.__dict__);exec(compile(source,'matched_reference_phase','exec'),space)
backtest=space['cash_backtest']
panel=m.Panel(load_x=False);prices=m.Prices(panel)
protocol=json.loads((AUDIT/'mixture_protocol.json').read_text())


def percentile(scores):
    result=np.full_like(scores,np.nan)
    for i,row in enumerate(scores):
        ok=np.isfinite(row);n=ok.sum()
        if n>1:result[i,ok]=(rankdata(row[ok])-.5)/n
    return result


for quarter in ['2025Q3','2025Q4','2026Q1','2026Q2']:
    target=OUT/f'{quarter}.json'
    if target.exists():continue
    saved=[np.load(AUDIT/'clean_references'/f'{v}_{quarter}_fold4.npz') for v in ['V31','V36','V38']]
    days=saved[0]['days']
    assert all(np.array_equal(x['days'],days) for x in saved)
    raw=sum(x['scores'] for x in saved)/3
    ranked=sum(percentile(x['scores']) for x in saved)/3
    old_rows=json.loads((AUDIT/'mixture_validation'/f'{quarter}.json').read_text())['rows']
    old_index={(r['topn'],r['period'],r['band'],r['phase']):r for r in old_rows if r['mixture']=='old_family'}
    rows=[];start=time.time()
    for control,scores in [('legacy_raw_mean3_cleanfold',raw),('legacy_percentile_mean3_cleanfold',ranked)]:
        for n in protocol['strategy_grid']['topn']:
            for period in protocol['strategy_grid']['period']:
                for band in protocol['strategy_grid']['band']:
                    for phase in range(period):
                        if (AUDIT/'STOP').exists():raise SystemExit('stopped by user')
                        st,_,_=backtest(scores,panel,prices,days,n=n,period=period,band=band,phase=phase)
                        assert abs(st['gross_return_same_positions']-st['fee_drag']-st['slippage_drag']-st['return_value'])<1e-10
                        if control=='legacy_percentile_mean3_cleanfold':
                            prior=old_index[n,period,band,phase]
                            assert abs(st['return_value']-prior['net_return'])<1e-12
                            assert abs(st['max_drawdown']-prior['max_drawdown'])<1e-12
                        rows.append(dict(control=control,topn=n,period=period,band=band,phase=phase,
                            net_return=st['return_value'],max_drawdown=st['max_drawdown'],sharpe=st['sharpe'],
                            gross_return_same_positions=st['gross_return_same_positions'],fee_drag=st['fee_drag'],
                            slippage_drag=st['slippage_drag'],fees=st['total_fees'],trades=st['trades']))
    assert len(rows)==72
    result=dict(quarter=quarter,start=str(panel.days[days[0]]),end=str(panel.days[days[-1]]),days=len(days),
        rows=rows,seconds=round(time.time()-start,2),rank_control_reproduces_prior=True,
        weights='three existing clean-fold checkpoints; no release or checkpoint changed',
        scope='matched train-before-validation window; raw3-model control is not the live12-fold release ensemble',
        selection='all predeclared candidates retained; no test or winning-phase selection')
    temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2));os.replace(temp,target)
    print('MATCHED_LEGACY_DONE',quarter,result['seconds'],flush=True)
print('ALL_MATCHED_LEGACY_CONTROLS_COMPLETE',flush=True)
