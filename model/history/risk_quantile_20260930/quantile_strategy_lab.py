"""Frozen lower-tail penalties crossed with all holding-size/turnover/phase candidates."""
from pathlib import Path
import inspect
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import torch

PROJECT=Path('/root/autodl-fs/model')
PRIMARY=PROJECT/'history/diversification_20260930'
AUDIT=PROJECT/'history/risk_quantile_20260930'
OUT=AUDIT/'quantile_strategy_grid'
OUT.mkdir(exist_ok=True)
pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
sys.path.insert(0,str(PROJECT/'experiments/V54'))
import model as m
import analysis as a

source=inspect.getsource(a.cash_backtest)
assert source.count('money=None):')==1 and source.count('% period == 0')==2
source=source.replace('money=None):','money=None, phase=0):',1)
source=source.replace('    money = float(', "    if period < 1 or not 0 <= phase < period:\n        raise ValueError('invalid rebalance phase')\n    money = float(",1)
source=source.replace('% period == 0','% period == phase')
space=dict(a.__dict__);exec(compile(source,'quantile_phase_cash_backtest','exec'),space)
backtest=space['cash_backtest']
panel=m.Panel(load_x=False);prices=m.Prices(panel)
completed=set()
for p in OUT.glob('V*_202*Q*.json'):
    o=json.loads(p.read_text());assert len(o['rows'])==576
    completed.add((o['unit'],o['quarter']))

while len(completed)<16:
    if (AUDIT/'STOP').exists() or (PRIMARY/'STOP').exists():raise SystemExit('stopped by user')
    for unit in ['V54','V55','V56','V57']:
        for quarter in ['2025Q3','2025Q4','2026Q1','2026Q2']:
            if (unit,quarter) in completed:continue
            path=AUDIT/'clean_quantiles'/f'{unit}_{quarter}_fold4.npz'
            if not path.exists():continue
            data=np.load(path);quantiles,days=data['quantiles'],data['days']
            scores=quantiles[:,:,1]-.5*(quantiles[:,:,1]-quantiles[:,:,0])
            assert np.array_equal(scores,data['scores'],equal_nan=True)
            expected,_,_=a.cash_backtest(scores,panel,prices,days,n=5,period=5,band=40)
            actual,_,_=backtest(scores,panel,prices,days,n=5,period=5,band=40,phase=0)
            assert actual==expected
            rows=[];started=time.time()
            for lam in [0.0,.25,.5,.75]:
                candidate=quantiles[:,:,1]-lam*(quantiles[:,:,1]-quantiles[:,:,0])
                for n in [5,10,20]:
                    for period in [1,5,10]:
                        for band in [0,20,40]:
                            for phase in range(period):
                                if (AUDIT/'STOP').exists() or (PRIMARY/'STOP').exists():raise SystemExit('stopped by user')
                                st,_,_=backtest(candidate,panel,prices,days,n=n,period=period,band=band,phase=phase)
                                assert abs(st['gross_return_same_positions']-st['fee_drag']-st['slippage_drag']-st['return_value'])<1e-10
                                rows.append(dict(risk_lambda=lam,topn=n,period=period,band=band,phase=phase,
                                    net_return=st['return_value'],gross_return_same_positions=st['gross_return_same_positions'],
                                    max_drawdown=st['max_drawdown'],sharpe=st['sharpe'],trades=st['trades'],
                                    fees=st['total_fees'],fee_drag=st['fee_drag'],slippage_drag=st['slippage_drag'],
                                    avg_exposure=st['avg_exposure']))
            assert len(rows)==576
            result=dict(unit=unit,quarter=quarter,start=str(panel.days[days[0]]),end=str(panel.days[days[-1]]),
                days=len(days),grid_rows=len(rows),rows=rows,seconds=round(time.time()-started,2),
                zero_phase_exactly_reproduced=True,
                paired='every risk penalty uses identical frozen quantile weights, selected epochs, dates and execution simulator',
                selection='none; all seeds, frozen candidates and phases retained; overlapping selected-epoch validation blocks')
            target=OUT/f'{unit}_{quarter}.json';temp=target.with_suffix('.tmp')
            temp.write_text(json.dumps(result,ensure_ascii=False,indent=2));os.replace(temp,target)
            completed.add((unit,quarter));print('QUANTILE_GRID_DONE',unit,quarter,result['seconds'],flush=True)
    status=AUDIT/'quantile_grid_status.json';temp=status.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed_blocks=len(completed),
        planned_blocks=16,grid_rows_per_block=576,completed=sorted(completed)),indent=2));os.replace(temp,status)
    try:state=json.loads((AUDIT/'batch_status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):state=None
    if state and state.get('failed'):raise SystemExit('training failed; retain measured grids')
    if len(completed)<16:time.sleep(20)
print('ALL_QUANTILE_STRATEGY_GRIDS_COMPLETE',flush=True)
