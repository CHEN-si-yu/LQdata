"""Summarize the entire frozen cost/size/period/band/phase grid without selecting."""
from pathlib import Path
import json
import time
import numpy as np

A=Path('/root/autodl-fs/model/history/absolute_ridge_20260930')
units=['V62','V63','V64','V65'];quarters=['2025Q3','2025Q4','2026Q1','2026Q2']
grid={}
for unit in units:
    for q in quarters:
        v=json.loads((A/'absolute_strategy_grid'/f'{unit}_{q}.json').read_text())
        assert len(v['rows'])==576
        grid[unit,q]={(r['cost_gate_multiplier'],r['topn'],r['period'],r['band'],r['phase']):r for r in v['rows']}
        assert len(grid[unit,q])==576
rows=[]
for q in quarters:
    for gate in [0.,.5,1.,2.]:
        for n in [5,10,20]:
            for period in [1,5,10]:
                for band in [0,20,40]:
                    part=[];seedmean=[];deltas=[];old=[]
                    for unit in units:
                        phase=[grid[unit,q][gate,n,period,band,p] for p in range(period)]
                        ref=[grid[unit,q][0.,n,period,band,p] for p in range(period)]
                        part.extend(phase);old.extend(ref)
                        seedmean.append(float(np.mean([r['net_return'] for r in phase])))
                        deltas.extend([(r['net_return']-b['net_return'])*100 for r,b in zip(phase,ref)])
                    rows.append(dict(quarter=q,cost_gate_multiplier=gate,topn=n,period=period,band=band,
                        seeds=4,all_phases=period,accounts=len(part),seed_phase_mean_net_return_pct=float(np.mean(seedmean))*100,
                        seed_sample_std_of_phase_means_pp=float(np.std(seedmean,ddof=1))*100,
                        mean_max_drawdown_pct=float(np.mean([r['max_drawdown'] for r in part]))*100,
                        mean_exposure=float(np.mean([r['avg_exposure'] for r in part])),
                        mean_fee_and_slippage_drag_pp=float(np.mean([r['fee_drag']+r['slippage_drag'] for r in part]))*100,
                        paired_net_delta_vs_same_no_gate_mean_pp=float(np.mean(deltas)),
                        paired_net_improved_accounts=sum(d>0 for d in deltas),
                        paired_net_total_accounts=len(deltas)))
assert len(rows)==432
ablation=[r for r in rows if r['topn']==5 and r['period']==1]
value=dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed_blocks=16,total_executed_cash_accounts=9216,
    rows=rows,daily_top5_cost_gate_vs_rank_buffer=ablation,selection='none; report every frozen combination and seed, no formal test tuning',
    aggregation='within seed all phases averaged descriptively, then four seeds mean/sample std; phase or seed averages are not single executable accounts; overlapping selected-stage validation')
(A/'all_absolute_strategy_summary.json').write_text(json.dumps(value,indent=2))
print('ALL_GRID_SUMMARY',value['completed_blocks'],value['total_executed_cash_accounts'])
for q in quarters:
    print('DAILY_TOP5_ABLATION',q,[dict(gate=r['cost_gate_multiplier'],band=r['band'],net=round(r['seed_phase_mean_net_return_pct'],2),dd=round(r['mean_max_drawdown_pct'],2),delta=round(r['paired_net_delta_vs_same_no_gate_mean_pp'],2)) for r in ablation if r['quarter']==q])
