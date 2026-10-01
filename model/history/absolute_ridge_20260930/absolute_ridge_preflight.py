"""Synthetic algebra/causality and real snapshot integrity checks; not strategy performance."""
from pathlib import Path
import ast
import gc
import json
import sys
import time
import types

import numpy as np
import pyarrow as pa
import torch

ROOT=Path('/root/autodl-fs/model')
AUDIT=ROOT/'history/absolute_ridge_20260930'
PARENT=ROOT/'history/diversification_20260930'
pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
results={}
real_days=None


class ToyPrices:
    hot_frac=1.
    def next_entry_ok(self):return np.ones((42,80),bool)


for unit in ['V62','V63','V64','V65']:
    d=ROOT/'experiments'/unit
    for name in ['model','analysis']:sys.modules.pop(name,None)
    sys.path.insert(0,str(d))
    import model as m
    import analysis as a
    torch.set_num_threads(1)
    m.validate_layout(allow_missing=True)
    assert len(m.fixed_files())==159
    assert len(list(d.glob('*.py')))==3
    assert a.STRATEGIES[0]==dict(name='top5_1d',topn=5,period=1,band=0),a.STRATEGIES
    features=m.feature_columns(m.metadata());market=m.market_columns(m.metadata())
    assert len(features)==44 and len(market)==61
    if real_days is None:real_days=m.axis()[0]
    clean=[(q,s['fold']) for q in m.LAYOUT_QUARTERS for s in m.splits(real_days,quarter=q)
           if s['train'].max()<s['valid'].min()]
    assert len(clean)==4
    # Exercise the sampler with small nested capacities only inside this in-memory fixture.
    frozen_caps=m.RECIPE['sample_caps'];m.RECIPE['sample_caps']=[8,16,24,32,40]
    rng=np.random.default_rng(21);toy=object.__new__(m.Panel)
    toy.X=rng.normal(size=(42,80,44)).astype(np.float32)
    toy.M=rng.normal(size=(42,61)).astype(np.float32);toy.market_dim=61
    toy.codes=np.arange(80);toy.coverage=np.ones((42,80),np.float32)
    toy.Y={f'label_ret_{h}d':rng.normal(0,.03,(42,80)).astype(np.float32) for h in [1,3,5,10,20]}
    toy.Y['label_ret_1d']=.004*toy.X[:,:,0]+.002*toy.M[:,0,None]
    toy.Y['label_ret_5d']=.015*toy.X[:,:,1]+.005*toy.M[:,1,None]
    toy.Y['label_ret_5d'][6,0]=np.nan
    train=np.arange(5,13);valid=np.arange(14,18)
    net=m.PredictModel(44,61)
    stages=m.closed_form_stages(net,toy,train,m.RECIPE['seed'])
    repeated=m.closed_form_stages(net,toy,train,m.RECIPE['seed'])
    assert all(np.array_equal(x['coef'],y['coef']) for x,y in zip(stages,repeated))
    assert all(x['normal_equation_relative_residual']<1e-8 for x in stages)
    assert all(x['fit_dates']==len(train) for x in stages)
    assert all(stages[i]['sample_rows']<stages[i+1]['sample_rows'] for i in range(4))
    with torch.no_grad():net.coef.copy_(torch.from_numpy(stages[-1]['coef']))
    x=torch.from_numpy(toy.X[14]);context=torch.from_numpy(toy.M[14])
    expected=net.basis_numpy(toy.X[14],toy.M[14])@stages[-1]['coef'].T
    with torch.no_grad():
        cpu=net(x,context).numpy()
        assert np.allclose(cpu,expected,atol=2e-7,rtol=2e-5)
        assert np.allclose(net(x[:7],context).numpy(),cpu[:7],atol=2e-7,rtol=2e-5)
        gpu=net.to('cuda')(x.to('cuda'),context.to('cuda')).cpu().numpy()
        assert np.allclose(cpu,gpu,atol=2e-7,rtol=2e-5)
    net=net.cpu()
    metrics=m.validation(net,toy,ToyPrices(),valid,torch.device('cpu'))
    assert abs(metrics['val_wei']+metrics['absolute_normalized_mse'])<1e-12
    # Direct rowwise loss independently checks the sufficient-statistics quadratic.
    losses=[]
    for day in train:
        mask=toy.mask(day).copy()
        for h in net.horizons:mask&=np.isfinite(toy.Y[f'label_ret_{h}d'][day])
        candidates=np.flatnonzero(mask)
        sampler=np.random.default_rng(np.random.SeedSequence([m.RECIPE['seed'],int(day)]))
        selected=candidates[sampler.permutation(len(candidates))[:40]]
        actual=net.basis_numpy(toy.X[day,selected],toy.M[day])@stages[-1]['coef'].T
        errors=[]
        for j,h in enumerate(net.horizons):
            target=np.clip(toy.Y[f'label_ret_{h}d'][day,selected],-m.RECIPE['target_clip'][h],m.RECIPE['target_clip'][h])
            errors.append(m.RECIPE['label_weights'][h]*np.mean((actual[:,j]-target)**2)/m.RECIPE['loss_scale'][h]**2)
        losses.append(sum(errors)/sum(m.RECIPE['label_weights'].values()))
    assert abs(np.mean(losses)-stages[-1]['train_loss'])<1e-6
    toy.X[20:]=1e6;toy.M[20:]=-1e6
    for y in toy.Y.values():y[20:]=1e6
    future=m.closed_form_stages(net,toy,train,m.RECIPE['seed'])
    assert all(np.array_equal(x['coef'],y['coef']) for x,y in zip(stages,future))
    # No real model artifacts are made by these synthetic checks.
    assert not list(d.rglob('*.pt'))
    records=[dict(quarter=q,fold=f,best_validation=metrics,argmax_validation=metrics,
                  bag=dict(topk=5,epochs=[1,2,3,4,5])) for q in m.LAYOUT_QUARTERS for f in m.LAYOUT_FOLDS]
    lengths={(q,s['fold']):len(s['valid']) for q in m.LAYOUT_QUARTERS for s in m.splits(real_days,quarter=q)}
    old=a.fold_valid_days;a.fold_valid_days=lambda:lengths
    assert len(a.training_validation_rows(records))==5
    a.fold_valid_days=old;assert a.bag_vs_argmax(records)['folds']==16
    m.RECIPE['sample_caps']=frozen_caps
    results[unit]=dict(features=44,market_features=61,basis_width=net.basis_dim,fixed_files=159,
        deterministic_solver=True,causal_future_invariance=True,daily_weighted_loss_exact=True,
        negative_proper_mse_criterion=True,normal_equation_max_relative_residual=max(s['normal_equation_relative_residual'] for s in stages),
        cpu_gpu_max_error=float(np.max(np.abs(cpu-gpu))),subset_inference_exact=True,report_compatibility=True,
        clean_blocks=clean,no_trained_artifacts_created=True)
    print('RIDGE_NUMERICAL_PREFLIGHT_PASS',unit,results[unit],flush=True)
    sys.path.pop(0);del net,toy,m,a,stages,repeated,future;gc.collect()

# Actual frozen prices and labels are checked; arbitrary forecasts only exercise execution scaling.
sys.path.insert(0,str(ROOT/'experiments/V62'))
sys.modules.pop('model',None);sys.modules.pop('analysis',None)
import model as m
import analysis as a
panel=m.Panel(load_x=False);prices=m.Prices(panel)
test=np.concatenate([m.splits(panel.days,quarter=q)[0]['test'] for q in m.LAYOUT_QUARTERS])
price_check=prices.verify_labels(panel,test)
days=m.splits(panel.days,quarter='2025Q3')[3]['valid'][:70]
scores=np.tile(np.linspace(.0022,.0028,len(panel.codes),dtype=np.float32),(len(days),1))
scores[:15]*=.3
single,_,_=a.cash_backtest(scores,panel,prices,days,n=5,period=1,band=0,forecast_members=1)
summed,_,_=a.cash_backtest(scores*4,panel,prices,days,n=5,period=1,band=0,forecast_members=4)
assert single==summed
parent_source=(ROOT/'experiments/V46/analysis.py').read_text()
cash_node=next(n for n in ast.parse(parent_source).body if getattr(n,'name',None)=='cash_backtest')
space=dict(a.__dict__);exec(compile(ast.unparse(cash_node),'parent_execution_simulator','exec'),space)
parent_cash=space['cash_backtest']
zero,_,_=a.cash_backtest(scores,panel,prices,days,n=5,period=1,band=0,cost_gate_multiplier=0.)
original,_,_=parent_cash(scores,panel,prices,days,n=5,period=1,band=0)
assert zero==original
one,_,_=a.cash_backtest(scores,panel,prices,days,n=1,period=1,band=0)
one_original,_,_=parent_cash(scores,panel,prices,days,n=1,period=1,band=0)
assert one==one_original
assert abs(single['gross_return_same_positions']-single['fee_drag']-single['slippage_drag']-single['return_value'])<1e-10
receipt=dict(ok=True,checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),results=results,
    real_snapshot_label_price_check=price_check,
    ensemble_sum_scaling_cost_gate_exact=True,zero_cost_gate_reproduces_parent=True,top1_baseline_exact=True,
    gross_fee_slippage_identity=True,scope='synthetic algebra and execution fixtures plus real label/price consistency; no strategy-performance conclusion')
(AUDIT/'preflight.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print('ALL_ABSOLUTE_RIDGE_PREFLIGHT_PASS',flush=True)
