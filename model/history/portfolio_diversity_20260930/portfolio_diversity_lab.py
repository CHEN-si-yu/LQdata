"""All-seed covariance-aware fresh entries in the unchanged real-execution cash simulator."""
from pathlib import Path
import inspect
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import torch

ROOT=Path('/root/autodl-fs/model')
PRIMARY=ROOT/'history/diversification_20260930'
AUDIT=ROOT/'history/portfolio_diversity_20260930'
AUDIT.mkdir(exist_ok=True)
OUT=AUDIT/'clean_cash_grid';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(AUDIT))
from portfolio_diversity_core import portfolio_return_history,portfolio_risk_information,portfolio_diverse_fresh
pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
sys.path.insert(0,str(ROOT/'experiments/V50'))
import model as m
import analysis as a
panel=m.Panel(load_x=False);prices=m.Prices(panel)
history=portfolio_return_history(prices)
source=inspect.getsource(a.cash_backtest)
assert source.count('money=None):')==1 and source.count('% period == 0')==2
source=source.replace('money=None):','money=None, phase=0, diversity_strength=0.0, risk_info=None):',1)
source=source.replace('% period == 0','% period == phase')
line="fresh = [int(c) for c in ranked if entry_ok[d, c] and c not in keep]"
assert source.count(line)==1
source=source.replace(line,line+"\n                if n>1 and diversity_strength>0:\n                    fresh=portfolio_diverse_fresh(fresh,keep,risk_info[signal],strength=diversity_strength,slots=max(0,n-len(keep)))")
space=dict(a.__dict__,portfolio_diverse_fresh=portfolio_diverse_fresh)
exec(compile(source,'portfolio_diversity_cash','exec'),space);backtest=space['cash_backtest']

for unit in ['V50','V51','V52','V53']:
    for quarter in ['2025Q3','2025Q4','2026Q1','2026Q2']:
        target=OUT/f'{unit}_{quarter}.json'
        if target.exists():continue
        if (PRIMARY/'STOP').exists() or (AUDIT/'STOP').exists():raise SystemExit('stopped by user')
        data=np.load(PRIMARY/'clean_innovations'/f'{unit}_{quarter}_fold4.npz')
        scores,days=data['scores'],data['days']
        start=time.time();risk=portfolio_risk_information(scores,history,days)
        actual,_,_=backtest(scores,panel,prices,days,n=5,period=1,band=40,diversity_strength=0.,risk_info=risk)
        expected,_,_=a.cash_backtest(scores,panel,prices,days,n=5,period=1,band=40)
        assert actual==expected
        rows=[]
        for strength in [0.,.10,.25]:
            for n in [5,10,20]:
                for period in [1,5]:
                    for phase in range(period):
                        if (PRIMARY/'STOP').exists() or (AUDIT/'STOP').exists():raise SystemExit('stopped by user')
                        st,curve,trades=backtest(scores,panel,prices,days,n=n,period=period,band=40,
                            phase=phase,diversity_strength=strength,risk_info=risk)
                        assert abs(st['gross_return_same_positions']-st['fee_drag']-st['slippage_drag']-st['return_value'])<1e-10
                        rows.append(dict(diversity_strength=strength,topn=n,period=period,band=40,phase=phase,
                            net_return=st['return_value'],gross_return_same_positions=st['gross_return_same_positions'],
                            max_drawdown=st['max_drawdown'],sharpe=st['sharpe'],trades=st['trades'],
                            fees=st['total_fees'],fee_drag=st['fee_drag'],slippage_drag=st['slippage_drag'],
                            avg_exposure=st['avg_exposure']))
        assert len(rows)==54
        result=dict(unit=unit,quarter=quarter,days=len(days),start=str(panel.days[days[0]]),
            end=str(panel.days[days[-1]]),rows=rows,grid_rows=len(rows),seconds=round(time.time()-start,2),
            zero_strength_exactly_reproduced=True,
            primary_candidate=.25,weights='actual existing V50-V53 cleanfold weights; no new training or weight alteration',
            scope='selected-epoch validation, all four seeds, all frozen strengths and phases; no formal test consulted')
        temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2));os.replace(temp,target)
        print('PORTFOLIO_GRID_DONE',unit,quarter,result['seconds'],flush=True)
        (AUDIT/'status.json').write_text(json.dumps(dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            completed_blocks=len(list(OUT.glob('V*_202*Q*.json'))),planned_blocks=16),indent=2))
print('ALL_PORTFOLIO_DIVERSITY_DIAGNOSTICS_COMPLETE',flush=True)
