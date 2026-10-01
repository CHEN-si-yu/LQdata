"""Frozen cost gates crossed with every execution size, period, band and phase."""
from pathlib import Path
import inspect
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import torch

PROJECT=Path('/root/autodl-fs/model');PRIMARY=PROJECT/'history/diversification_20260930'
AUDIT=PROJECT/'history/absolute_ridge_20260930';OUT=AUDIT/'absolute_strategy_grid';OUT.mkdir(exist_ok=True)
os.nice(14);pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
sys.path.insert(0,str(PROJECT/'experiments/V62'));import model as m;import analysis as a
source=inspect.getsource(a.cash_backtest)
assert source.count('cost_gate_multiplier=None):')==1 and source.count('% period == 0')==2
source=source.replace('cost_gate_multiplier=None):','cost_gate_multiplier=None, phase=0):',1)
source=source.replace('    money = float(',"    if period < 1 or not 0 <= phase < period:\n        raise ValueError('invalid rebalance phase')\n    money = float(",1)
source=source.replace('% period == 0','% period == phase')
space=dict(a.__dict__);exec(compile(source,'absolute_phase_cash_backtest','exec'),space);backtest=space['cash_backtest']
panel=m.Panel(load_x=False);prices=m.Prices(panel)
completed=set()
for p in OUT.glob('V*_202*Q*.json'):
    v=json.loads(p.read_text());assert len(v['rows'])==576;completed.add((v['unit'],v['quarter']))

while len(completed)<16:
    if (AUDIT/'STOP').exists() or (PRIMARY/'STOP').exists():raise SystemExit('stopped by user')
    for unit in ['V62','V63','V64','V65']:
        for q in ['2025Q3','2025Q4','2026Q1','2026Q2']:
            if (unit,q) in completed:continue
            path=AUDIT/'clean_absolute'/f'{unit}_{q}_fold4.npz'
            if not path.exists():continue
            data=np.load(path);scores,days=data['scores'],data['days'];rows=[];started=time.time()
            expected,_,_=a.cash_backtest(scores,panel,prices,days,n=5,period=1,band=0,forecast_members=1,cost_gate_multiplier=1.)
            actual,_,_=backtest(scores,panel,prices,days,n=5,period=1,band=0,forecast_members=1,cost_gate_multiplier=1.,phase=0)
            assert expected==actual
            for gate in [0.,.5,1.,2.]:
                for n in [5,10,20]:
                    for period in [1,5,10]:
                        for band in [0,20,40]:
                            for phase in range(period):
                                if (AUDIT/'STOP').exists() or (PRIMARY/'STOP').exists():raise SystemExit('stopped by user')
                                st,_,_=backtest(scores,panel,prices,days,n=n,period=period,band=band,forecast_members=1,cost_gate_multiplier=gate,phase=phase)
                                assert abs(st['gross_return_same_positions']-st['fee_drag']-st['slippage_drag']-st['return_value'])<1e-10
                                rows.append(dict(cost_gate_multiplier=gate,topn=n,period=period,band=band,phase=phase,
                                    net_return=st['return_value'],gross_return_same_positions=st['gross_return_same_positions'],
                                    max_drawdown=st['max_drawdown'],sharpe=st['sharpe'],trades=st['trades'],fees=st['total_fees'],
                                    fee_drag=st['fee_drag'],slippage_drag=st['slippage_drag'],avg_exposure=st['avg_exposure']))
            assert len(rows)==576
            result=dict(unit=unit,quarter=q,start=str(panel.days[days[0]]),end=str(panel.days[days[-1]]),days=len(days),rows=rows,
                seconds=round(time.time()-started,2),zero_phase_exactly_reproduced=True,
                paired='identical fresh fitted weights, dates and execution; every gate/phase retained',
                selection='none; frozen before fitting; overlapping selected-stage validation diagnostics')
            target=OUT/f'{unit}_{q}.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));os.replace(temp,target)
            completed.add((unit,q));print('ABSOLUTE_GRID_DONE',unit,q,result['seconds'],flush=True)
    status=AUDIT/'absolute_grid_status.json';temp=status.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed_blocks=len(completed),planned_blocks=16,
        grid_rows_per_block=576,completed=sorted(completed)),indent=2));os.replace(temp,status)
    try:state=json.loads((AUDIT/'batch_status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):state=None
    if state and state.get('failed'):raise SystemExit('training failed; retain measured grids')
    if len(completed)<16:time.sleep(20)
print('ALL_ABSOLUTE_STRATEGY_GRIDS_COMPLETE',flush=True)
