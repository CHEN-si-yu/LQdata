"""Frozen family mixtures in a single executable cash account; validation only."""
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
OUT=AUDIT/'mixture_validation'
OUT.mkdir(exist_ok=True)
protocol=json.loads((AUDIT/'mixture_protocol.json').read_text())
pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
sys.path.insert(0,str(ROOT/'experiments/V46'))
import model as m
import analysis as a
source=inspect.getsource(a.cash_backtest).replace('money=None):','money=None, phase=0):',1)
assert source.count('% period == 0')==2
source=source.replace('% period == 0','% period == phase')
space=dict(a.__dict__);exec(compile(source,'mixture_phase_backtest','exec'),space)
backtest=space['cash_backtest']
panel=m.Panel(load_x=False);prices=m.Prices(panel)


def percentile(scores):
    out=np.full_like(scores,np.nan)
    for i,row in enumerate(scores):
        known=np.isfinite(row);n=int(known.sum())
        if n>1:out[i,known]=(rankdata(row[known])-.5)/n
    return out


done={p.stem for p in OUT.glob('202*Q*.json')}
while len(done)<4:
    if (AUDIT/'STOP').exists():raise SystemExit('stopped by user')
    for quarter in ['2025Q3','2025Q4','2026Q1','2026Q2']:
        if quarter in done:continue
        inputs={};days=None
        for family in ['old','quality','flow']:
            files=[AUDIT/('clean_references' if family=='old' else 'clean_innovations')/f'{u}_{quarter}_fold4.npz' for u in protocol[family]]
            if not all(p.exists() for p in files):break
            component=[]
            for p in files:
                o=np.load(p)
                if days is None:days=o['days']
                assert np.array_equal(days,o['days'])
                component.append(percentile(o['scores']))
            inputs[family]=sum(component)/len(component)
        if len(inputs)!=3:continue
        start=time.time();rows=[]
        for name,weights in protocol['mixtures'].items():
            # A zero-weight family must not remove eligible stocks through 0 * NaN.
            scores=sum(w*inputs[fam] for fam,w in zip(['old','quality','flow'],weights) if w>0)
            for n in protocol['strategy_grid']['topn']:
                for period in protocol['strategy_grid']['period']:
                    for band in protocol['strategy_grid']['band']:
                        for phase in range(period):
                            if (AUDIT/'STOP').exists():raise SystemExit('stopped by user')
                            st,curve,trades=backtest(scores,panel,prices,days,n=n,period=period,band=band,phase=phase)
                            assert abs(st['gross_return_same_positions']-st['fee_drag']-st['slippage_drag']-st['return_value'])<1e-10
                            rows.append(dict(mixture=name,weights=weights,topn=n,period=period,band=band,phase=phase,
                                net_return=st['return_value'],gross_return_same_positions=st['gross_return_same_positions'],
                                max_drawdown=st['max_drawdown'],sharpe=st['sharpe'],fees=st['total_fees'],
                                fee_drag=st['fee_drag'],slippage_drag=st['slippage_drag'],trades=st['trades'],avg_exposure=st['avg_exposure']))
        assert len(rows)==324
        o=dict(quarter=quarter,start=str(panel.days[days[0]]),end=str(panel.days[days[-1]]),days=len(days),
               seconds=round(time.time()-start,2),grid_rows=len(rows),rows=rows,
               note='all seeds used, all candidates and phases retained; one shared100000 account; validation diagnostics, no test selection')
        (OUT/f'{quarter}.json').write_text(json.dumps(o,ensure_ascii=False,indent=2))
        done.add(quarter);print('MIXTURE_DONE',quarter,o['seconds'],flush=True)
    (AUDIT/'mixture_status.json').write_text(json.dumps(dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed=sorted(done)),indent=2))
    if (AUDIT/'batch_status.json').exists() and json.loads((AUDIT/'batch_status.json').read_text()).get('failed'):
        raise SystemExit('training failure: stop mixture evaluation pending repair')
    time.sleep(20)
print('ALL_MIXTURES_MEASURED',flush=True)
