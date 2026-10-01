"""Every frozen mean/variance utility candidate on identical fitted validation scores."""
from pathlib import Path
import inspect
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import torch
from portfolio_variance_core import variance_return_history,variance_risk_information,variance_choose_fresh

ROOT=Path('/root/autodl-fs/model');A=ROOT/'history/portfolio_variance_20260930';OUT=A/'clean_cash_grid';OUT.mkdir(exist_ok=True)
PRIMARY=ROOT/'history/diversification_20260930';ABSOLUTE=ROOT/'history/absolute_ridge_20260930'
PROTOCOL=json.loads((A/'protocol.json').read_text())
os.nice(12);pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
sys.path.insert(0,str(ROOT/'experiments/V62'));import model as m;import analysis as a
source=inspect.getsource(a.cash_backtest)
assert source.count('cost_gate_multiplier=None):')==1 and source.count('% period == 0')==2
source=source.replace('cost_gate_multiplier=None):','cost_gate_multiplier=None, risk_aversion=0., risk_information=None, phase=0):',1)
source=source.replace('    money = float(',"    if period < 1 or not 0 <= phase < period:\n        raise ValueError('invalid rebalance phase')\n    money = float(",1)
source=source.replace('% period == 0','% period == phase')
needle='                desired = list(keep) + fresh[:max(0, n - len(keep))]'
assert source.count(needle)==1
replacement='''                if n>1 and risk_aversion>0:
                    carried=set(keep)|{c for c in holdings if c not in keep and (not px.exit[d,c] or buys[c]>=d)}
                    fresh=variance_choose_fresh(fresh,carried,risk_information[int(signal)],utility,
                        risk_aversion=risk_aversion,n=n,forecast_members=forecast_members,slots=max(0,n-len(carried)))
                desired = list(keep) + fresh[:max(0, n - len(keep))]'''
source=source.replace(needle,replacement)
space=dict(a.__dict__,variance_choose_fresh=variance_choose_fresh);exec(compile(source,'variance_cash_backtest','exec'),space)
backtest=space['cash_backtest'];panel=m.Panel(load_x=False);prices=m.Prices(panel);history=variance_return_history(prices)
completed={p.stem for p in OUT.glob('V*_202*Q*.json')}
while len(completed)<16:
    if (A/'STOP').exists() or (PRIMARY/'STOP').exists():raise SystemExit('stopped by user')
    for unit in PROTOCOL['parents']:
        for q in ['2025Q3','2025Q4','2026Q1','2026Q2']:
            if f'{unit}_{q}' in completed:continue
            p=ABSOLUTE/'clean_absolute'/f'{unit}_{q}_fold4.npz'
            if not p.exists():continue
            data=np.load(p);scores,days=data['scores'],data['days'];started=time.time()
            information=variance_risk_information(scores,history,days)
            expected,_,_=a.cash_backtest(scores,panel,prices,days,n=5,period=1,band=0,forecast_members=1,cost_gate_multiplier=1.)
            actual,_,_=backtest(scores,panel,prices,days,n=5,period=1,band=0,forecast_members=1,cost_gate_multiplier=1.,risk_aversion=0.,risk_information=information)
            assert actual==expected
            rows=[]
            for kappa in PROTOCOL['risk_aversion_candidates']:
                for n in PROTOCOL['target_size_candidates']:
                    for period in PROTOCOL['period_candidates']:
                        for phase in range(period):
                            if (A/'STOP').exists() or (PRIMARY/'STOP').exists():raise SystemExit('stopped by user')
                            st,_,_=backtest(scores,panel,prices,days,n=n,period=period,band=0,phase=phase,
                                forecast_members=1,cost_gate_multiplier=1.,risk_aversion=kappa,risk_information=information)
                            assert abs(st['gross_return_same_positions']-st['fee_drag']-st['slippage_drag']-st['return_value'])<1e-10
                            rows.append(dict(risk_aversion=kappa,topn=n,period=period,phase=phase,band=0,
                                net_return=st['return_value'],max_drawdown=st['max_drawdown'],sharpe=st['sharpe'],
                                gross_return_same_positions=st['gross_return_same_positions'],fee_drag=st['fee_drag'],slippage_drag=st['slippage_drag'],
                                fees=st['total_fees'],trades=st['trades'],avg_exposure=st['avg_exposure']))
            assert len(rows)==54
            value=dict(unit=unit,quarter=q,start=str(panel.days[days[0]]),end=str(panel.days[days[-1]]),days=len(days),rows=rows,
                seconds=round(time.time()-started,2),zero_risk_exactly_reproduced=True,
                covariance_available_mean=float(np.mean([v['usable_size'] for v in information.values()])),
                inference='saved clean fitted absolute forecasts; no new model training or test-based selection',
                risk_proxy='past63 clipped adjusted-close returns, sample-variance PSD Gram,20% diagonal shrink; equal-slot marginal utility; blocked carried holdings included, unknown carry covariance falls back to native ordering')
            target=OUT/f'{unit}_{q}.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));os.replace(temp,target)
            completed.add(f'{unit}_{q}');print('VARIANCE_GRID_DONE',unit,q,value['seconds'],flush=True)
            status=A/'status.json';temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(pid=os.getpid(),checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed=sorted(completed),completed_blocks=len(completed),planned_blocks=16,accounts_per_block=54),indent=2));os.replace(temp,status)
    if len(completed)<16:time.sleep(20)
print('ALL_VARIANCE_GRIDS_COMPLETE',flush=True)
