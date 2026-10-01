"""Measure all completed clean blocks; does not select seeds or touch releases."""
from pathlib import Path
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT = Path('/root/autodl-fs/model')
AUDIT = PROJECT / 'history/diversification_20260930'
OUT = AUDIT / 'clean_innovations'
OUT.mkdir(exist_ok=True)
RESULT = AUDIT / 'clean_innovation_results.json'
torch.set_num_threads(2)


def compare_scores(scores, other):
    assert scores.shape == other.shape
    correlations, overlaps = [], []
    for a, b in zip(scores, other):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 50:
            continue
        correlations.append(float(spearmanr(a[ok], b[ok]).statistic))
        aa = set(np.flatnonzero(np.isfinite(a))[np.argsort(-a[np.isfinite(a)], kind='stable')[:5]])
        bb = set(np.flatnonzero(np.isfinite(b))[np.argsort(-b[np.isfinite(b)], kind='stable')[:5]])
        overlaps.append(len(aa & bb) / 5)
    return dict(daily_rank_correlation=float(np.mean(correlations)),
                top5_overlap=float(np.mean(overlaps)), days=len(correlations))


def publish(rows):
    summaries = {}
    for family in ['quality_additive', 'flow_temporal']:
        part = [r for r in rows if r['family'] == family]
        quarters = sorted(set(r['quarter'] for r in part))
        summaries[family] = dict(completed_blocks=len(part), quarters={})
        for q in quarters:
            qr = [r for r in part if r['quarter'] == q]
            key = 'top5_5d' if family == 'quality_additive' else 'top5_1d'
            x = [r['accounts'][key]['net_return'] * 100 for r in qr]
            summaries[family]['quarters'][q] = dict(seeds=[r['seed'] for r in qr],
                completed_seeds=len(qr), net_return_mean_pct=float(np.mean(x)),
                net_return_sample_std_pp=float(np.std(x, ddof=1)) if len(x)>1 else None,
                positive_seeds=sum(v > 0 for v in x),
                max_drawdown_mean_pct=float(np.mean([r['accounts'][key]['max_drawdown']*100 for r in qr])))
    temp = RESULT.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        rows=rows, summary=summaries,
        note='every completed train-before-validation block; selected-epoch validation diagnostics, not independent tests; overlapping quarters not independent; no seed or strategy selection'), ensure_ascii=False, indent=2))
    os.replace(temp, RESULT)


if __name__ == '__main__':
    rows = json.loads(RESULT.read_text())['rows'] if RESULT.exists() else []
    measured = {(r['unit'],r['quarter']) for r in rows}
    while len(measured) < 32:
        if (AUDIT / 'STOP').exists():
            raise SystemExit('stopped by user')
        changed = False
        for unit in ['V46','V50','V47','V51','V48','V52','V49','V53']:
            d = PROJECT / 'experiments' / unit
            ready = [q for q in ['2025Q3','2025Q4','2026Q1','2026Q2']
                     if (unit,q) not in measured and (d/'model_train'/q/'fold4/complete.json').exists()]
            if not ready:
                continue
            for name in ['model','analysis']:
                sys.modules.pop(name,None)
            sys.path.insert(0,str(d))
            import model as m
            import analysis as a
            print('MEASURE_LOAD',unit,ready,flush=True)
            panel = m.Panel(m.RECIPE['features'])
            prices = m.Prices(panel)
            for q in ready:
                if (AUDIT / 'STOP').exists():
                    raise SystemExit('stopped by user')
                sp = m.splits(panel.days,quarter=q)[3]
                assert sp['train'].max() < sp['valid'].min()
                va = sp['valid']
                folder=d/'model_train'/q/'fold4'
                ck = torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
                assert ck['features']==panel.features
                net = m.PredictModel(len(panel.features),panel.market_dim)
                net.load_state_dict(ck['model']);net.ridge=ck.get('ridge')
                td = sp['test'][:3]
                reload = m.predict(net,panel,prices,td,torch.device('cpu'))
                saved = np.load(folder/'test_predictions.npy')[:3]
                assert np.allclose(reload,saved,atol=2e-4,rtol=2e-4,equal_nan=True)
                error = float(np.nanmax(np.abs(reload-saved)))
                scores = m.predict(net,panel,prices,va,torch.device('cpu'))
                np.savez_compressed(OUT/f'{unit}_{q}_fold4.npz',scores=scores,days=va,dates=panel.days[va])
                row=dict(unit=unit,family=m.RECIPE['family'],seed=m.RECIPE['seed'],quarter=q,fold=4,
                    start=str(panel.days[va[0]]),end=str(panel.days[va[-1]]),days=len(va),
                    cpu_reload_max_error=error,accounts={},correlation_vs_reference={})
                for period in [1,5]:
                    st,curve,trades=a.cash_backtest(scores,panel,prices,va,n=5,period=period,band=40)
                    assert np.isclose(st['gross_return_same_positions']-st['fee_drag']-st['slippage_drag'],st['return_value'],atol=1e-10)
                    eq=np.asarray([c['equity'] for c in curve]);daily=eq/np.r_[100000.,eq[:-1]]-1
                    key=f'top5_{period}d'
                    row['accounts'][key]=dict(net_return=st['return_value'],max_drawdown=st['max_drawdown'],
                        sharpe=st['sharpe'],trades=st['trades'],fees=st['total_fees'],
                        gross_return_same_positions=st['gross_return_same_positions'],fee_drag=st['fee_drag'],
                        slippage_drag=st['slippage_drag'],avg_exposure=st['avg_exposure'],
                        net_arithmetic_annual_pct=float(daily.mean()*242*100))
                    np.savez_compressed(OUT/f'{unit}_{q}_cash_{period}d.npz',
                        dates=np.asarray([c['date'] for c in curve]),equity=eq)
                for ref in ['V31','V36','V38']:
                    p=AUDIT/'clean_references'/f'{ref}_{q}_fold4.npz'
                    if p.exists():
                        o=np.load(p);assert np.array_equal(o['days'],va)
                        row['correlation_vs_reference'][ref]=compare_scores(scores,o['scores'])
                rows.append(row);measured.add((unit,q));publish(rows);changed=True
                print('MEASURED',unit,q,row['accounts'],row['correlation_vs_reference'],flush=True)
            sys.path.pop(0)
            del panel,prices,net,scores,ck,m,a
            gc.collect()
        # References can finish after a new model: fill missing correlations without rerunning inference.
        for row in rows:
            for ref in ['V31','V36','V38']:
                p=AUDIT/'clean_references'/f'{ref}_{row["quarter"]}_fold4.npz'
                if ref in row['correlation_vs_reference'] or not p.exists():
                    continue
                other=np.load(p);new=np.load(OUT/f'{row["unit"]}_{row["quarter"]}_fold4.npz')
                assert np.array_equal(new['days'],other['days'])
                row['correlation_vs_reference'][ref]=compare_scores(new['scores'],other['scores']);changed=True
        if changed:
            publish(rows)
        state = json.loads((AUDIT/'batch_status.json').read_text()) if (AUDIT/'batch_status.json').exists() else {}
        if state.get('failed'):
            raise SystemExit('training failed: stop measuring until repair')
        time.sleep(15)
    publish(rows)
    print('ALL_CLEAN_BLOCKS_MEASURED',flush=True)
