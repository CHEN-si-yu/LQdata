"""All frozen absolute-model seeds and train-before-validation blocks; no selection."""
from pathlib import Path
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import torch
from scipy.stats import spearmanr

PROJECT=Path('/root/autodl-fs/model')
PRIMARY=PROJECT/'history/diversification_20260930'
AUDIT=PROJECT/'history/absolute_ridge_20260930'
OUT=AUDIT/'clean_absolute';OUT.mkdir(exist_ok=True)
RESULT=AUDIT/'clean_absolute_results.json'
UNITS=['V62','V63','V64','V65']
os.nice(12);pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)


def stopped():
    return (AUDIT/'STOP').exists() or (PRIMARY/'STOP').exists()


def compare_scores(scores,other):
    assert scores.shape==other.shape
    corr=[];overlap=[]
    for p,q in zip(scores,other):
        ok=np.isfinite(p)&np.isfinite(q)
        if ok.sum()<50:continue
        corr.append(float(spearmanr(p[ok],q[ok]).statistic))
        def top(x):
            ii=np.flatnonzero(np.isfinite(x));return set(ii[np.argsort(-x[ii],kind='stable')[:5]])
        overlap.append(len(top(p)&top(q))/5)
    return dict(daily_rank_correlation=float(np.mean(corr)),top5_overlap=float(np.mean(overlap)),days=len(corr))


def daily_returns(equity):
    return equity/np.r_[100000.,equity[:-1]]-1


def calibration(heads,panel,days,m):
    out={}
    for j,h in enumerate(m.RECIPE['label_horizons']):
        p=heads[:,:,j].astype(np.float64)
        raw=panel.Y[f'label_ret_{h}d'][days].astype(np.float64)
        known=np.isfinite(p)&np.isfinite(raw)
        pp=p[known];yy=raw[known];target=np.clip(yy,-m.RECIPE['target_clip'][h],m.RECIPE['target_clip'][h])
        cov=np.mean((pp-pp.mean())*(target-target.mean()))
        bins=[]
        for lo,hi in [(-np.inf,-.005),(-.005,0.),(0.,.00162),(.00162,.005),(.005,np.inf)]:
            mask=(pp>=lo)&(pp<hi)
            bins.append(dict(forecast_interval=[None if not np.isfinite(lo) else lo,None if not np.isfinite(hi) else hi],
                observations=int(mask.sum()),forecast_mean=float(pp[mask].mean()) if mask.any() else None,
                actual_raw_mean=float(yy[mask].mean()) if mask.any() else None,
                actual_clipped_mean=float(target[mask].mean()) if mask.any() else None))
        out[str(h)]=dict(observations=len(pp),forecast_mean=float(pp.mean()),actual_raw_mean=float(yy.mean()),
            actual_clipped_mean=float(target.mean()),clipped_rmse=float(np.sqrt(np.mean((pp-target)**2))),
            raw_rmse=float(np.sqrt(np.mean((pp-yy)**2))),clipped_mae=float(np.mean(np.abs(pp-target))),
            descriptive_calibration_slope=float(cov/np.var(pp)) if np.var(pp)>0 else None,
            bins=bins,aggregation='all eligible stock-date forecasts; bins descriptive, not independent samples or cash positions')
    return out


def publish(rows):
    quarters={}
    for q in sorted({r['quarter'] for r in rows}):
        qr=[r for r in rows if r['quarter']==q];gates={}
        for gate in ['0.0','0.5','1.0','2.0']:
            rr=[r['accounts'][gate] for r in qr];values=[r['net_return']*100 for r in rr]
            gates[gate]=dict(seeds=[r['seed'] for r in qr],completed_seeds=len(qr),
                net_return_mean_pct=float(np.mean(values)),net_return_sample_std_pp=float(np.std(values,ddof=1)) if len(values)>1 else None,
                max_drawdown_mean_pct=float(np.mean([r['max_drawdown']*100 for r in rr])),
                avg_exposure_mean=float(np.mean([r['avg_exposure'] for r in rr])),
                fee_drag_mean_pp=float(np.mean([r['fee_drag']*100 for r in rr])),
                net_delta_vs_no_gate_mean_pp=float(np.mean([(r['accounts'][gate]['net_return']-r['accounts']['0.0']['net_return'])*100 for r in qr])))
        quarters[q]=dict(windows=[[r['start'],r['end']] for r in qr],gates=gates)
    value=dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed_blocks=len(rows),planned_blocks=16,
        rows=rows,summary=quarters,
        interpretation='selected-stage validation diagnostics, overlapping windows; every seed retained; no formal test selection; prediction calibration is not trade execution profit')
    temp=RESULT.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(temp,RESULT)


if __name__=='__main__':
    rows=json.loads(RESULT.read_text())['rows'] if RESULT.exists() else []
    measured={(r['unit'],r['quarter']) for r in rows}
    freeze=json.loads((AUDIT/'source_freeze.json').read_text())
    while len(measured)<16:
        if stopped():raise SystemExit('stopped by user')
        for unit in UNITS:
            d=PROJECT/'experiments'/unit
            ready=[q for q in ['2025Q3','2025Q4','2026Q1','2026Q2'] if (unit,q) not in measured and (d/'model_train'/q/'fold4/complete.json').exists()]
            if not ready:continue
            for name,digest in freeze[unit].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==digest
            for name in ['model','analysis']:sys.modules.pop(name,None)
            sys.path.insert(0,str(d));import model as m;import analysis as a
            panel=m.Panel(m.RECIPE['features']);prices=m.Prices(panel)
            print('ABSOLUTE_MEASURE_LOAD',unit,ready,flush=True)
            for q in ready:
                if stopped():raise SystemExit('stopped by user')
                sp=m.splits(panel.days,quarter=q)[3];assert sp['train'].max()<sp['valid'].min();days=sp['valid']
                folder=d/'model_train'/q/'fold4';ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
                assert ck['features']==panel.features
                net=m.PredictModel(len(panel.features),panel.market_dim);net.load_state_dict(ck['model']);net.ridge=ck.get('ridge')
                reload=m.predict(net,panel,prices,sp['test'][:3],torch.device('cpu'))
                saved=np.load(folder/'test_predictions.npy')[:3]
                assert np.allclose(reload,saved,atol=1e-6,rtol=1e-5,equal_nan=True)
                heads=m.predict_heads(net,panel,days,torch.device('cpu'))
                scores=m.predict(net,panel,prices,days,torch.device('cpu'))
                assert np.allclose(scores,m.score_of(heads),equal_nan=True)
                target=OUT/f'{unit}_{q}_fold4.npz';temp=target.with_suffix('.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,scores=scores,heads=heads,days=days,dates=panel.days[days])
                os.replace(temp,target)
                row=dict(unit=unit,seed=m.RECIPE['seed'],quarter=q,fold=4,start=str(panel.days[days[0]]),end=str(panel.days[days[-1]]),
                    days=len(days),checkpoint_sha256=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest(),
                    cpu_reload_max_error=float(np.nanmax(np.abs(reload-saved))),calibration=calibration(heads,panel,days,m),
                    accounts={},correlation_vs_reference={},cash_return_correlation_vs_reference={})
                for gate in [0.0,.5,1.,2.]:
                    st,curve,trades=a.cash_backtest(scores,panel,prices,days,n=5,period=1,band=0,forecast_members=1,cost_gate_multiplier=gate)
                    assert abs(st['gross_return_same_positions']-st['fee_drag']-st['slippage_drag']-st['return_value'])<1e-10
                    eq=np.asarray([c['equity'] for c in curve]);dates=np.asarray([c['date'] for c in curve])
                    row['accounts'][str(gate)]=dict(net_return=st['return_value'],max_drawdown=st['max_drawdown'],sharpe=st['sharpe'],
                        trades=st['trades'],fees=st['total_fees'],fee_drag=st['fee_drag'],slippage_drag=st['slippage_drag'],
                        gross_return_same_positions=st['gross_return_same_positions'],avg_exposure=st['avg_exposure'],
                        net_arithmetic_annual_pct=float(daily_returns(eq).mean()*242*100))
                    np.savez_compressed(OUT/f'{unit}_{q}_cash_gate{gate}.npz',dates=dates,equity=eq)
                    if gate==1.:
                        for ref in ['V31','V36','V38']:
                            o=np.load(PRIMARY/'clean_references'/f'{ref}_{q}_cash_1d.npz');assert np.array_equal(o['dates'],dates)
                            row['cash_return_correlation_vs_reference'][ref]=float(np.corrcoef(daily_returns(eq),daily_returns(o['equity']))[0,1])
                for ref in ['V31','V36','V38']:
                    o=np.load(PRIMARY/'clean_references'/f'{ref}_{q}_fold4.npz');assert np.array_equal(o['days'],days)
                    row['correlation_vs_reference'][ref]=compare_scores(scores,o['scores'])
                rows.append(row);measured.add((unit,q));publish(rows)
                print('ABSOLUTE_MEASURED',unit,q,row['accounts'],row['correlation_vs_reference'],flush=True)
            sys.path.pop(0);del panel,prices,net,ck,heads,scores,m,a;gc.collect()
        try:state=json.loads((AUDIT/'batch_status.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):state=None
        if state and state.get('failed'):raise SystemExit('training failed; retain measured blocks')
        if len(measured)<16:time.sleep(20)
    publish(rows);print('ALL_ABSOLUTE_CLEAN_BLOCKS_COMPLETE',flush=True)
