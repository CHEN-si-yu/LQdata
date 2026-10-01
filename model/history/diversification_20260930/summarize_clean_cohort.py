"""Describe every seed and frozen strategy; do not collapse overlapping blocks into tests."""
from pathlib import Path
import json
import time

import numpy as np

ROOT=Path('/root/autodl-fs/model')
AUDIT=ROOT/'history/diversification_20260930'
data=json.loads((AUDIT/'clean_innovation_results.json').read_text())
assert len(data['rows'])==32
summaries={}
for quarter in ['2025Q3','2025Q4','2026Q1','2026Q2']:
    raw=json.loads((AUDIT/'matched_legacy_controls'/f'{quarter}.json').read_text())
    mix=json.loads((AUDIT/'mixture_validation'/f'{quarter}.json').read_text())
    selected=[r for r in data['rows'] if r['quarter']==quarter]
    assert len(selected)==8
    report=dict(start=raw['start'],end=raw['end'],signal_days=raw['days'],
                seed_results={},frozen_strategy_comparisons=[])
    for family in ['quality_additive','flow_temporal']:
        part=[r for r in selected if r['family']==family]
        assert len(part)==4
        key='top5_5d' if family=='quality_additive' else 'top5_1d'
        vectors=[r['accounts'][key]['net_return']*100 for r in part]
        report['seed_results'][family]=dict(seeds=[r['seed'] for r in part],unit_names=[r['unit'] for r in part],
            primary=key,net_return_pct=vectors,net_return_mean_pct=float(np.mean(vectors)),
            net_return_sample_std_pp=float(np.std(vectors,ddof=1)),
            max_drawdown_pct=[r['accounts'][key]['max_drawdown']*100 for r in part],
            rank_correlation_vs_legacy_mean=float(np.mean([
                x['daily_rank_correlation'] for r in part for x in r['correlation_vs_reference'].values()])),
            top5_overlap_vs_legacy_mean=float(np.mean([
                x['top5_overlap'] for r in part for x in r['correlation_vs_reference'].values()])))
    for n,period,band in [(5,1,40),(5,5,40),(20,5,40)]:
        controls=[r for r in raw['rows'] if (r['topn'],r['period'],r['band'])==(n,period,band)]
        strategies=[dict(r,control=r['mixture']) for r in mix['rows'] if (r['topn'],r['period'],r['band'])==(n,period,band)
                    and r['mixture'] in ['quality_family','flow_family','three_equal','old_half','quality_half']]
        combined=controls+strategies
        old=[r for r in controls if r['control']=='legacy_raw_mean3_cleanfold']
        assert len(old)==period
        old_index={r['phase']:r for r in old}
        group=dict(topn=n,period=period,band=band,strategies={})
        for name in sorted(set(r['control'] for r in combined)):
            part=[r for r in combined if r['control']==name]
            assert len(part)==period
            x=np.array([r['net_return']*100 for r in part])
            dd=np.array([r['max_drawdown']*100 for r in part])
            net_delta=np.array([(r['net_return']-old_index[r['phase']]['net_return'])*100 for r in part])
            dd_delta=np.array([(r['max_drawdown']-old_index[r['phase']]['max_drawdown'])*100 for r in part])
            group['strategies'][name]=dict(phases=sorted(r['phase'] for r in part),
                net_return_phase_mean_pct=float(x.mean()),net_return_phase_min_pct=float(x.min()),
                net_return_phase_max_pct=float(x.max()),net_return_phase_sample_std_pp=float(x.std(ddof=1)) if len(x)>1 else None,
                max_drawdown_phase_mean_pct=float(dd.mean()),
                cost_drag_phase_mean_pp=float(np.mean([(r['fee_drag']+r['slippage_drag'])*100 for r in part])),
                gross_return_same_positions_phase_mean_pct=float(np.mean([r['gross_return_same_positions']*100 for r in part])),
                paired_net_delta_vs_legacy_raw_phase_mean_pp=float(net_delta.mean()),
                paired_net_improved_phases=int((net_delta>0).sum()),
                paired_drawdown_improved_phases=int((dd_delta>0).sum()),
                paired_drawdown_delta_phase_mean_pp=float(dd_delta.mean()))
        report['frozen_strategy_comparisons'].append(group)
    # All members, periods and phases retained when assessing concentration sensitivity.
    for family in ['quality_additive','flow_temporal']:
        part=[r for r in selected if r['family']==family]
        sensitivity={}
        for n in [5,10,20]:
            per_seed=[]
            for seed_row in part:
                grid=json.loads((AUDIT/'strategy_sensitivity'/f'{seed_row["unit"]}_{quarter}.json').read_text())['rows']
                values=[r for r in grid if (r['topn'],r['period'],r['band'])==(n,5,40)]
                assert len(values)==5
                per_seed.append(dict(seed=seed_row['seed'],unit=seed_row['unit'],
                    net_phase_mean_pct=float(np.mean([r['net_return']*100 for r in values])),
                    net_phase_min_pct=float(np.min([r['net_return']*100 for r in values])),
                    max_drawdown_phase_mean_pct=float(np.mean([r['max_drawdown']*100 for r in values]))))
            x=[r['net_phase_mean_pct'] for r in per_seed]
            sensitivity[str(n)]=dict(per_seed=per_seed,net_phase_mean_seed_mean_pct=float(np.mean(x)),
                net_phase_mean_seed_sample_std_pp=float(np.std(x,ddof=1)),
                seed_and_phase_worst_net_pct=float(min(r['net_phase_min_pct'] for r in per_seed)))
        report['seed_results'][family]['weekly_band40_size_sensitivity']=sensitivity
    summaries[quarter]=report

result=dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed_clean_blocks=32,summary=summaries,
    standard_test=dict(start='2025-07-01',end='2026-06-30',signals=242,used_for_selection=False,
                       status='remaining fixed folds training; no complete new-model test conclusion'),
    interpretation=[
        'Every validation window also selects epochs and overlaps other windows; these diagnostics are not independent out-of-sample returns.',
        'All four seeds are retained. Family mixtures combine all seed predictions into one executable account, not average separate cash curves.',
        'Phase mean is descriptive, not a deployable multi-phase portfolio; each row represents its own executable account.',
        'Raw legacy control uses three clean fold checkpoints on matched windows; it is not the live release12-fold ensemble.',
        'Gross is the same actual positions with costs added back as idle cash; it is not a separately reinvested cost-free strategy.',
        'No winning seed, phase, penalty or mixture selected, and no release promoted.'])
(AUDIT/'all_clean_cohort_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
for quarter,d in summaries.items():
    print('BLOCK',quarter,d['start'],d['end'],d['signal_days'])
    for fam,s in d['seed_results'].items():
        print('SEEDS',fam,round(s['net_return_mean_pct'],2),round(s['net_return_sample_std_pp'],2),
              'rank_corr',round(s['rank_correlation_vs_legacy_mean'],3),'overlap',round(s['top5_overlap_vs_legacy_mean'],3))
    config=d['frozen_strategy_comparisons'][2]
    for name in ['legacy_raw_mean3_cleanfold','legacy_percentile_mean3_cleanfold','quality_family','old_half']:
        r=config['strategies'][name]
        print('N20_WEEKLY',name,'net',round(r['net_return_phase_mean_pct'],2),
              'dd',round(r['max_drawdown_phase_mean_pct'],2),'cost',round(r['cost_drag_phase_mean_pp'],2),
              'paired_delta',round(r['paired_net_delta_vs_legacy_raw_phase_mean_pp'],2),
              'improved',r['paired_net_improved_phases'],r['paired_drawdown_improved_phases'])
