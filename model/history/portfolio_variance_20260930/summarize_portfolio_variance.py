from pathlib import Path
import json
import time
import numpy as np

A=Path('/root/autodl-fs/model/history/portfolio_variance_20260930')
units=['V62','V63','V64','V65'];quarters=['2025Q3','2025Q4','2026Q1','2026Q2'];grid={};rows=[]
for u in units:
    for q in quarters:
        d=json.loads((A/'clean_cash_grid'/f'{u}_{q}.json').read_text());assert len(d['rows'])==54
        grid[u,q]={(r['risk_aversion'],r['topn'],r['period'],r['phase']):r for r in d['rows']}
for q in quarters:
    for kappa in [0.,1.,3.]:
        for n in [5,10,20]:
            for period in [1,5]:
                part=[];seedmean=[];pairs=[]
                for u in units:
                    phase=[grid[u,q][kappa,n,period,p] for p in range(period)];base=[grid[u,q][0.,n,period,p] for p in range(period)]
                    part.extend(phase);seedmean.append(float(np.mean([r['net_return'] for r in phase])))
                    pairs.extend(zip(phase,base))
                rows.append(dict(quarter=q,risk_aversion=kappa,topn=n,period=period,band=0,seeds=4,phases=period,accounts=len(part),
                    net_seed_phase_mean_pct=float(np.mean(seedmean))*100,net_seed_std_of_phase_means_pp=float(np.std(seedmean,ddof=1))*100,
                    max_drawdown_mean_pct=float(np.mean([r['max_drawdown'] for r in part]))*100,
                    avg_exposure=float(np.mean([r['avg_exposure'] for r in part])),
                    mean_fee_and_slippage_drag_pp=float(np.mean([r['fee_drag']+r['slippage_drag'] for r in part]))*100,
                    paired_net_delta_mean_pp=float(np.mean([(r['net_return']-b['net_return'])*100 for r,b in pairs])),
                    paired_net_improved_accounts=sum(r['net_return']>b['net_return'] for r,b in pairs),
                    paired_drawdown_improved_accounts=sum(r['max_drawdown']>b['max_drawdown'] for r,b in pairs)))
assert len(rows)==72
primary=[r for r in rows if r['topn']==5 and r['period']==1]
value=dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed_blocks=16,executed_accounts=864,summary_rows=72,
    rows=rows,primary=primary,selection='none; kappa3 fixed before test readout; every frozen seed/parameter/phase retained',
    interpretation='overlapping selected-stage validation, all phase means descriptive not executable accounts; no new trained weights')
(A/'all_variance_summary.json').write_text(json.dumps(value,indent=2))
for q in quarters:print('PRIMARY',q,[r for r in primary if r['quarter']==q])
