"""Implementation fixtures and frozen-snapshot integrity; no performance selection."""
from pathlib import Path
import ast
import copy
import gc
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import torch
from scipy.stats import rankdata

ROOT=Path('/root/autodl-fs/model')
A=ROOT/'history/flow_technical_joint_20260930'
pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
results={};real_days=None


class ToyPrices:
    hot_frac=1.
    def next_entry_ok(self):return np.ones((100,80),bool)


for unit in ['V66','V67','V68','V69']:
    d=ROOT/'experiments'/unit
    for name in ['model','analysis']:sys.modules.pop(name,None)
    sys.path.insert(0,str(d))
    import model as m
    import analysis as a
    torch.set_num_threads(1)
    m.validate_layout(allow_missing=True)
    assert len(m.fixed_files())==159 and len(list(d.glob('*.py')))==3
    assert a.STRATEGIES==[dict(name='top5_5d',topn=5,period=5,band=40),dict(name='top1_1d',topn=1,period=1,band=0)]
    features=m.feature_columns(m.metadata());market=m.market_columns(m.metadata())
    assert len(features)==149 and len(market)==61
    assert not set(m.RECIPE['flow_feature_names']).intersection(m.RECIPE['technical_feature_names'])
    if real_days is None:real_days=m.axis()[0]
    clean=[(q,s['fold']) for q in m.LAYOUT_QUARTERS for s in m.splits(real_days,quarter=q)
           if s['train'].max()<s['valid'].min()]
    assert len(clean)==4
    caps=m.RECIPE['absolute_sample_caps'];m.RECIPE['absolute_sample_caps']=[8,16,24,32,40]
    rng=np.random.default_rng(21);toy=object.__new__(m.Panel)
    toy.X=rng.normal(size=(100,80,149)).astype(np.float32)
    toy.M=rng.normal(size=(100,61)).astype(np.float32);toy.market_dim=61
    toy.codes=np.array([f'{i:06d}.SZ' for i in range(80)]);toy.features=features
    toy.days=real_days[np.flatnonzero(np.char.startswith(real_days,'2025'))[:100]]
    toy.coverage=np.ones((100,80),np.float32)
    toy.flow_coverage=toy.coverage.copy();toy.technical_coverage=toy.coverage.copy()
    toy.Y={f'label_ret_{h}d':rng.normal(0,.03,(100,80)).astype(np.float32) for h in [1,3,5,10,20]}
    toy.amount=np.full((100,80),1e8)
    ti=m.RECIPE['technical_indices']
    toy.Y['label_ret_1d']=.004*toy.X[:,:,ti[0]]+.002*toy.M[:,0,None]
    toy.Y['label_ret_5d']=.015*toy.X[:,:,ti[1]]+.005*toy.M[:,1,None]
    toy.Y['label_ret_5d'][10,0]=np.nan
    train=np.arange(8,20);valid=np.arange(44,48)
    net=m.PredictModel(149,61)
    fitted=m.fit_absolute_component(net,toy,train,m.RECIPE['seed'])
    repeated=m.fit_absolute_component(net,toy,train,m.RECIPE['seed'])
    stages=fitted['stages']
    assert all(np.array_equal(x['coef'],y['coef']) for x,y in zip(stages,repeated['stages']))
    assert all(x['normal_equation_relative_residual']<1e-8 and x['fit_dates']==len(train) for x in stages)
    assert all(stages[i]['sample_rows']<stages[i+1]['sample_rows'] for i in range(4))
    assert np.array_equal(net.absolute_coef.detach().numpy(),np.mean(np.stack([s['coef'] for s in stages]),axis=0,dtype=np.float32))
    raw=m.predict_heads(net,toy,valid,torch.device('cpu'))
    score=m.score_of(raw)
    independent=np.stack([.5*((rankdata(row[:,1])-.5)/80)+.5*((rankdata(row[:,3])-.5)/80) for row in raw]).astype(np.float32)
    assert np.allclose(score,independent,atol=1e-7,rtol=0.)
    aux=m.predict_absolute(net,toy,valid,torch.device('cpu'))
    assert np.allclose(aux,raw[:,:,3],atol=2e-7,rtol=2e-5)
    x=torch.from_numpy(toy.inputs(44,toy.mask(44)));context=torch.from_numpy(toy.M[44])
    net.eval()
    with torch.no_grad():
        cpu=net(x,context).numpy()
        subset=net(x[:7],context).numpy()
        assert np.allclose(subset,cpu[:7],atol=2e-7,rtol=2e-5)
        gpu=net.to('cuda')(x.to('cuda'),context.to('cuda')).cpu().numpy()
        print('CPU_GPU_RAW_DIAGNOSTIC',unit,float(np.max(np.abs(cpu-gpu))),flush=True)
        assert np.allclose(cpu,gpu,atol=2e-4,rtol=2e-4)
    net=net.cpu()
    # Independently check sufficient-statistic losses by direct sampled rows.
    losses=[]
    for day in train:
        known=toy.technical_mask(day).copy()
        for h in [1,5]:known &= np.isfinite(toy.Y[f'label_ret_{h}d'][day])
        candidates=np.flatnonzero(known)
        sampler=np.random.default_rng(np.random.SeedSequence([m.RECIPE['seed'],int(day)]))
        selected=candidates[sampler.permutation(len(candidates))[:40]]
        fitted_y=net.absolute_basis_numpy(toy.X[day,selected],toy.M[day])@stages[-1]['coef'].T
        errors=[m.RECIPE['absolute_label_weights'][h]*np.mean((fitted_y[:,j]-np.clip(toy.Y[f'label_ret_{h}d'][day,selected],-m.RECIPE['absolute_target_clip'][h],m.RECIPE['absolute_target_clip'][h]))**2)/m.RECIPE['absolute_loss_scale'][h]**2 for j,h in enumerate([1,5])]
        losses.append(sum(errors)/sum(m.RECIPE['absolute_label_weights'].values()))
    assert abs(np.mean(losses)-stages[-1]['train_loss'])<1e-6
    toy.flow_coverage[40,0]=0.;toy.technical_coverage[44,1]=0.
    assert not toy.mask(44)[0] and not toy.mask(44)[1] and toy.technical_mask(44)[0]
    toy.flow_coverage[40,0]=1.;toy.technical_coverage[44,1]=1.
    before=m.predict_heads(net,toy,valid,torch.device('cpu'))
    toy.X[60:]=1e6;toy.M[60:]=-1e6
    for y in toy.Y.values():y[60:]=1e6
    future=m.fit_absolute_component(net,toy,train,m.RECIPE['seed'])
    assert all(np.array_equal(x['coef'],y['coef']) for x,y in zip(stages,future['stages']))
    assert np.array_equal(before,m.predict_heads(net,toy,valid,torch.device('cpu')))
    assert not list(d.rglob('*.pt'))
    extra={}
    if unit=='V66':
        # Exercise the complete training/prefit/resume path in explicitly synthetic history folders.
        original_roots=(m.UNIT_ROOT,m.RUN_ROOT,m.INPUT_ROOT)
        original_splits=m.splits;original_epochs=m.RECIPE['max_epochs']
        m.RECIPE['max_epochs']=2
        test=np.arange(80,84);score_days=np.arange(80,92)
        synthetic_splits=[dict(quarter='2025Q3',fold=f,train=train,valid=valid,test=test,score=score_days) for f in range(1,5)]
        m.splits=lambda *args,**kwargs:synthetic_splits
        fixture=A/'implementation_fixtures';fixture.mkdir(exist_ok=True)
        outputs=[]
        for name,prefit in [('uninterrupted',False),('prepared_then_resumed',True)]:
            root=fixture/name;root.mkdir(exist_ok=True)
            m.UNIT_ROOT=root.resolve();m.RUN_ROOT=m.UNIT_ROOT;m.INPUT_ROOT=m.UNIT_ROOT
            if prefit:
                prepared=m.train_fold(4,'cpu',toy,ToyPrices(),'2025Q3',absolute_only=True)
                folder=root/'model_train/2025Q3/fold4'
                assert prepared['neural_epochs']==0 and not (folder/'best.pt').exists() and not (folder/'complete.json').exists()
                wrong=m.RECIPE['flow_fusion_weight'];m.RECIPE['flow_fusion_weight']=.25
                try:m.train_fold(4,'cpu',toy,ToyPrices(),'2025Q3');raise AssertionError('stale resume unexpectedly accepted')
                except ValueError as exc:assert 'fingerprint' in str(exc)
                finally:m.RECIPE['flow_fusion_weight']=wrong
            result=m.train_fold(4,'cpu',toy,ToyPrices(),'2025Q3')
            folder=root/'model_train/2025Q3/fold4'
            assert len(list(folder.iterdir()))==8
            checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            assert checkpoint['absolute_fit']['origin'].startswith('fresh fits')
            outputs.append((np.load(folder/'score_predictions.npy'),checkpoint['model']))
        assert np.array_equal(outputs[0][0],outputs[1][0],equal_nan=True)
        assert all(torch.equal(outputs[0][1][k],outputs[1][1][k]) for k in outputs[0][1])
        extra=dict(two_epoch_fixture_completed=True,absolute_prefit_not_a_completed_fold=True,prefit_resume_exact=True,stale_recipe_resume_rejected=True,all_eight_artifacts=True)
        m.UNIT_ROOT,m.RUN_ROOT,m.INPUT_ROOT=original_roots;m.splits=original_splits;m.RECIPE['max_epochs']=original_epochs
    m.RECIPE['absolute_sample_caps']=caps
    results[unit]=dict(features=149,flow=105,technical=44,market=61,basis_width=194,fixed_files=159,
        clean_blocks=clean,fresh_five_fit_mean_exact=True,date_equal_solver_loss_exact=True,
        future_input_label_invariance=True,separate_component_coverage=True,independent_fusion_score_exact=True,
        aux_equals_raw_last_head=True,cpu_gpu_max_raw_head_error=float(np.max(np.abs(cpu-gpu))),
        subset_inference_invariant=True,no_real_trained_artifacts=True,**extra)
    print('FUSION_PREFLIGHT_UNIT',unit,json.dumps(results[unit]),flush=True)
    sys.path.pop(0);del net,toy,m,a;gc.collect()

# Current labels/prices, cash execution, gate scaling and score persistence.
sys.path.insert(0,str(ROOT/'experiments/V66'));sys.modules.pop('model',None);sys.modules.pop('analysis',None)
import model as m
import analysis as a
panel=m.Panel(load_x=False);prices=m.Prices(panel)
test=np.concatenate([m.splits(panel.days,quarter=q)[0]['test'] for q in m.LAYOUT_QUARTERS])
price_check=prices.verify_labels(panel,test)
days=test[:70]
score=np.tile(np.linspace(0.,1.,len(panel.codes),dtype=np.float32),(len(days),1))
gate=np.tile(np.linspace(.001,.003,len(panel.codes),dtype=np.float32),(len(days),1))
one=a.cash_backtest(score,panel,prices,days,n=5,period=5,band=40,forecast_members=1,gate_scores=gate)
four=a.cash_backtest(score*4,panel,prices,days,n=5,period=5,band=40,forecast_members=4,gate_scores=gate*4)
assert one==four
source=(ROOT/'experiments/V50/analysis.py').read_text()
cash_node=next(n for n in ast.parse(source).body if getattr(n,'name',None)=='cash_backtest')
space=dict(a.__dict__);exec(compile(ast.unparse(cash_node),'parent_cash_reference','exec'),space)
native=space['cash_backtest']
zero=a.cash_backtest(score,panel,prices,days,n=5,period=5,band=40,cost_gate_multiplier=0.)
original=native(score,panel,prices,days,n=5,period=5,band=40)
assert zero==original
baseline=a.cash_backtest(score,panel,prices,days,n=1,period=1,band=0)
assert baseline==native(score,panel,prices,days,n=1,period=1,band=0)
idle=a.cash_backtest(score,panel,prices,days,n=5,period=5,band=40,gate_scores=np.zeros_like(gate))
assert idle[0]['trades']==0 and idle[0]['return_value']==0.
assert abs(one[0]['gross_return_same_positions']-one[0]['fee_drag']-one[0]['slippage_drag']-one[0]['return_value'])<1e-10
fixture=A/'implementation_fixtures/saved_score_roundtrip';fixture.mkdir(exist_ok=True)
original_roots=(m.UNIT_ROOT,a.RUN_ROOT,a.INPUT_ROOT)
m.UNIT_ROOT=fixture.resolve();a.RUN_ROOT=m.UNIT_ROOT;a.INPUT_ROOT=m.UNIT_ROOT
panel.absolute_forecast=np.full((len(panel.days),len(panel.codes)),np.nan,np.float32);panel.absolute_forecast[days]=gate*4
folder=fixture/'model_pred/ensemble'
a.write_scores(score*4,panel,days,folder)
a.save_score_metadata(folder,dict(clean_blocks=dict(blocks=[])),panel.days[days].tolist())
original_load=torch.load
def forbidden_load(*args,**kwargs):raise AssertionError('saved-score path read a model checkpoint')
torch.load=forbidden_load
try:
    saved,saved_days,meta=a.read_saved_scores(folder,panel)
    assert np.array_equal(saved,score*4) and np.array_equal(saved_days,days)
    assert np.array_equal(panel.absolute_forecast[days],gate*4)
    assert a.cash_backtest(saved,panel,prices,saved_days,n=5,period=5,band=40)==four
finally:torch.load=original_load
m.UNIT_ROOT,a.RUN_ROOT,a.INPUT_ROOT=original_roots
receipt=dict(ok=True,checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),results=results,
    real_snapshot_label_price_check=price_check,auxiliary_fourfold_gate_scale_exact=True,
    zero_gate_native_cash_exact=True,top1_baseline_exact=True,zero_forecast_idle_exact=True,
    saved_score_auxiliary_roundtrip_exact=True,saved_score_cash_no_checkpoint_reads=True,
    gross_fee_slippage_identity=True,scope='synthetic implementation fixtures and real current price/label checks only; no fitted real-model performance conclusion')
(A/'preflight.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print('ALL_FRESH_FUSION_PREFLIGHT_PASS',flush=True)
