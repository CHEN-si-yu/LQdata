"""All-seed calibration and risk-ranking diagnostics on train-before-validation blocks."""
from pathlib import Path
import gc
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
AUDIT=PROJECT/'history/risk_quantile_20260930'
OUT=AUDIT/'clean_quantiles'
OUT.mkdir(exist_ok=True)
RESULT=AUDIT/'quantile_validation_results.json'
UNITS=['V54','V55','V56','V57']
QUARTERS=['2025Q3','2025Q4','2026Q1','2026Q2']
LAMBDAS=[0.0,0.25,0.5,0.75]
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
torch.set_num_threads(1)


def stopped():
    return (AUDIT/'STOP').exists() or (PRIMARY/'STOP').exists()


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return None


def write_json(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    os.replace(temp,path)


def compare(x,y):
    assert x.shape==y.shape
    correlations,overlaps=[],[]
    for a,b in zip(x,y):
        ok=np.isfinite(a)&np.isfinite(b)
        if ok.sum()<50:continue
        correlations.append(float(spearmanr(a[ok],b[ok]).statistic))
        aa=set(np.flatnonzero(np.isfinite(a))[np.argsort(-a[np.isfinite(a)],kind='stable')[:5]])
        bb=set(np.flatnonzero(np.isfinite(b))[np.argsort(-b[np.isfinite(b)],kind='stable')[:5]])
        overlaps.append(len(aa&bb)/5)
    return dict(daily_rank_correlation=float(np.mean(correlations)),top5_overlap=float(np.mean(overlaps)),
                comparable_days=len(correlations))


def calibration(qhat,panel,days):
    levels=np.array([.25,.5,.75])
    losses,coverage,width,mature=[],[],[],[]
    for i,d in enumerate(days):
        y=panel.Y['label_ret_5d'][d]
        ok=np.isfinite(qhat[i]).all(axis=1)&np.isfinite(y)
        if ok.sum()<3:continue
        target=y[ok].astype(np.float64)
        target-=np.median(target)
        err=target[:,None]-qhat[i,ok]
        losses.append(np.maximum(levels*err,(levels-1)*err).mean())
        coverage.append((target[:,None]<=qhat[i,ok]).mean(axis=0))
        width.append(np.mean(qhat[i,ok,2]-qhat[i,ok,0]))
        mature.append(int(ok.sum()))
    return dict(day_equal_pinball_loss=float(np.mean(losses)),quantile_levels=levels.tolist(),
                day_equal_coverage=np.mean(coverage,axis=0).tolist(),eligible_mature_days=len(losses),
                eligible_mature_stock_rows=sum(mature),mean_interquartile_width=float(np.mean(width)),
                target='raw snapshot 5d return minus per-date median of finite scored labels',
                scope='selected-epoch validation calibration; labels never supplied as inference inputs')


def account(a,scores,panel,prices,days,period):
    st,curve,trades=a.cash_backtest(scores,panel,prices,days,n=5,period=period,band=40)
    assert abs(st['gross_return_same_positions']-st['fee_drag']-st['slippage_drag']-st['return_value'])<1e-10
    return dict(net_return=st['return_value'],max_drawdown=st['max_drawdown'],sharpe=st['sharpe'],
                gross_return_same_positions=st['gross_return_same_positions'],fee_drag=st['fee_drag'],
                slippage_drag=st['slippage_drag'],fees=st['total_fees'],trades=st['trades'],
                avg_exposure=st['avg_exposure'])


def publish(rows):
    summary={}
    for quarter in QUARTERS:
        part=[r for r in rows if r['quarter']==quarter]
        if not part:continue
        summary[quarter]=dict(completed_seeds=len(part),lambdas={},calibration={
            'mean_day_equal_pinball_loss':float(np.mean([r['calibration']['day_equal_pinball_loss'] for r in part])),
            'mean_day_equal_coverage':np.mean([r['calibration']['day_equal_coverage'] for r in part],axis=0).tolist()})
        for lam in LAMBDAS:
            key=str(lam);values=[r['lambda_accounts'][key]['top5_1d']['net_return']*100 for r in part]
            summary[quarter]['lambdas'][key]=dict(net_return_mean_pct=float(np.mean(values)),
                net_return_sample_std_pp=float(np.std(values,ddof=1)) if len(values)>1 else None,
                positive_seeds=sum(x>0 for x in values),
                max_drawdown_mean_pct=float(np.mean([r['lambda_accounts'][key]['top5_1d']['max_drawdown']*100 for r in part])))
    write_json(RESULT,dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),rows=rows,summary=summary,
        note='all four seeds and all frozen risk penalties retained; same weights and epochs within lambda comparison; no independent test consulted or candidate selected; validation blocks overlap'))


if __name__=='__main__':
    existing=read_json(RESULT)
    rows=existing['rows'] if existing else []
    measured={(r['unit'],r['quarter']) for r in rows}
    while len(measured)<16:
        if stopped():raise SystemExit('stopped by user')
        for index,unit in enumerate(UNITS):
            folder=PROJECT/'experiments'/unit
            ready=[q for q in QUARTERS if (unit,q) not in measured and (folder/'model_train'/q/'fold4/complete.json').exists()]
            if not ready:continue
            for name in ['model','analysis']:sys.modules.pop(name,None)
            sys.path.insert(0,str(folder))
            import model as m
            import analysis as a
            panel=m.Panel(m.RECIPE['features'])
            prices=m.Prices(panel)
            assert m.RECIPE['risk_lambda_candidates']==LAMBDAS and m.RECIPE['risk_lambda']==.5
            for quarter in ready:
                if stopped():raise SystemExit('stopped by user')
                sp=m.splits(panel.days,quarter=quarter)[3]
                assert sp['train'].max()<sp['valid'].min()
                days=sp['valid'];fold=folder/'model_train'/quarter/'fold4'
                checkpoint=torch.load(fold/'best.pt',map_location='cpu',weights_only=False)
                assert checkpoint['features']==panel.features
                net=m.PredictModel(len(panel.features),panel.market_dim)
                net.load_state_dict(checkpoint['model']);net.ridge=checkpoint.get('ridge')
                reload=m.predict(net,panel,prices,sp['test'][:3],torch.device('cpu'))
                saved=np.load(fold/'test_predictions.npy')[:3]
                assert np.allclose(reload,saved,rtol=2e-4,atol=2e-4,equal_nan=True)
                qhat=m.predict_quantiles(net,panel,days,torch.device('cpu'))
                scores=qhat[:,:,1]-.5*(qhat[:,:,1]-qhat[:,:,0])
                actual=m.predict(net,panel,prices,days[:3],torch.device('cpu'))
                assert np.allclose(actual,scores[:3],rtol=2e-4,atol=2e-4,equal_nan=True)
                finite=np.isfinite(qhat).all(axis=-1)
                assert np.all(qhat[:,:,0][finite]<=qhat[:,:,1][finite])
                assert np.all(qhat[:,:,1][finite]<=qhat[:,:,2][finite])
                row=dict(unit=unit,seed=m.RECIPE['seed'],quarter=quarter,fold=4,days=len(days),
                    start=str(panel.days[days[0]]),end=str(panel.days[days[-1]]),
                    cpu_reload_max_error=float(np.nanmax(np.abs(reload-saved))),
                    calibration=calibration(qhat,panel,days),lambda_accounts={},correlation_vs_reference={})
                validation=m.validation(net,panel,prices,days,torch.device('cpu'))
                assert np.isclose(row['calibration']['day_equal_pinball_loss'],validation['quantile_pinball_loss'],rtol=1e-6)
                for lam in LAMBDAS:
                    candidate=qhat[:,:,1]-lam*(qhat[:,:,1]-qhat[:,:,0])
                    row['lambda_accounts'][str(lam)]={}
                    for period in [1,5]:
                        row['lambda_accounts'][str(lam)][f'top5_{period}d']=account(a,candidate,panel,prices,days,period)
                for reference,root in [(f'V{50+index}',PRIMARY/'clean_innovations'),
                                       *[(v,PRIMARY/'clean_references') for v in ['V31','V36','V38']]]:
                    path=root/f'{reference}_{quarter}_fold4.npz'
                    if path.exists():
                        other=np.load(path);assert np.array_equal(other['days'],days)
                        row['correlation_vs_reference'][reference]=compare(scores,other['scores'])
                np.savez_compressed(OUT/f'{unit}_{quarter}_fold4.npz',quantiles=qhat,scores=scores,days=days,dates=panel.days[days])
                rows.append(row);measured.add((unit,quarter));publish(rows)
                print('QUANTILE_VALIDATION_DONE',unit,quarter,row['calibration'],flush=True)
            sys.path.pop(0)
            del panel,prices,net,m,a,checkpoint,qhat,scores
            gc.collect()
        state=read_json(AUDIT/'batch_status.json')
        if state and state.get('failed'):raise SystemExit('training failure; retain existing diagnostics')
        write_json(AUDIT/'quantile_validation_status.json',dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            completed_blocks=len(measured),planned_blocks=16,completed=sorted(measured)))
        if len(measured)<16:time.sleep(20)
    publish(rows)
    print('ALL_QUANTILE_VALIDATION_MEASURED',flush=True)
