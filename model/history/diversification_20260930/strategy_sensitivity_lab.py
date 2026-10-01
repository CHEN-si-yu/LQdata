"""Predeclared clean-validation grid, including all rebalance phases; no test selection."""
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
AUDIT=PROJECT/'history/diversification_20260930'
OUT=AUDIT/'strategy_sensitivity'
OUT.mkdir(exist_ok=True)
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
torch.set_num_threads(1)
sys.path.insert(0,str(PROJECT/'experiments/V46'))
import model as m
import analysis as a

# Copy the unit's actual execution simulator into scratch; only expose the predeclared phase.
source=inspect.getsource(a.cash_backtest)
assert source.count('money=None):')==1
source=source.replace('money=None):','money=None, phase=0):',1)
source=source.replace('    money = float(', "    if period < 1 or not 0 <= phase < period:\n        raise ValueError('invalid rebalance phase')\n    money = float(",1)
assert source.count('% period == 0')==2
source=source.replace('% period == 0','% period == phase')
space=dict(a.__dict__)
exec(compile(source,'phase_cash_backtest','exec'),space)
backtest=space['cash_backtest']
panel=m.Panel(load_x=False)
prices=m.Prices(panel)
completed=set()
for p in OUT.glob('V*_202*Q*.json'):
    o=json.loads(p.read_text());completed.add((o['unit'],o['quarter']))

while len(completed)<32:
    if (AUDIT/'STOP').exists():
        raise SystemExit('stopped by user')
    for unit in ['V46','V50','V47','V51','V48','V52','V49','V53']:
        for q in ['2025Q3','2025Q4','2026Q1','2026Q2']:
            if (unit,q) in completed:
                continue
            p=AUDIT/'clean_innovations'/f'{unit}_{q}_fold4.npz'
            if not p.exists():
                continue
            data=np.load(p);scores,days=data['scores'],data['days']
            # With default phase, the lab must reproduce the closed-unit simulator exactly.
            expected,_,_=a.cash_backtest(scores,panel,prices,days,n=5,period=5,band=40)
            actual,_,_=backtest(scores,panel,prices,days,n=5,period=5,band=40,phase=0)
            assert expected==actual
            rows=[];started=time.time()
            for n in [5,10,20]:
                for period in [1,5,10]:
                    for band in [0,20,40]:
                        for phase in range(period):
                            if (AUDIT/'STOP').exists():
                                raise SystemExit('stopped by user')
                            st,curve,trades=backtest(scores,panel,prices,days,n=n,period=period,band=band,phase=phase)
                            assert abs(st['gross_return_same_positions']-st['fee_drag']-st['slippage_drag']-st['return_value'])<1e-10
                            rows.append(dict(topn=n,period=period,band=band,phase=phase,
                                net_return=st['return_value'],gross_return_same_positions=st['gross_return_same_positions'],
                                max_drawdown=st['max_drawdown'],sharpe=st['sharpe'],trades=st['trades'],
                                fees=st['total_fees'],fee_drag=st['fee_drag'],slippage_drag=st['slippage_drag'],
                                avg_exposure=st['avg_exposure']))
            o=dict(unit=unit,quarter=q,start=str(panel.days[days[0]]),end=str(panel.days[days[-1]]),
                days=len(days),snapshot=str(panel.root),grid_rows=len(rows),seconds=round(time.time()-started,2),
                zero_phase_exactly_reproduced=True,rows=rows,
                selection='none; all frozen candidates, all phases; clean validation only, no test consulted')
            assert len(rows)==144
            (OUT/f'{unit}_{q}.json').write_text(json.dumps(o,ensure_ascii=False,indent=2))
            completed.add((unit,q))
            print('SENSITIVITY_DONE',unit,q,len(rows),o['seconds'],flush=True)
    (AUDIT/'sensitivity_status.json').write_text(json.dumps(dict(
        checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed_blocks=len(completed),
        grid_rows_per_block=144,planned_blocks=32,completed=sorted(completed)),indent=2))
    try:
        state=json.loads((AUDIT/'batch_status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        state=None  # shared-disk replacement may briefly be invisible; retain completed work
    if state is not None and state.get('failed'):
        raise SystemExit('training failure: stop grid pending repair')
    time.sleep(20)
print('ALL_SENSITIVITY_COMPLETE',flush=True)
