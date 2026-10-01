"""Numerical/causal/contract checks only; synthetic data is never reported as returns."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pyarrow as pa
import torch

ROOT=Path('/root/autodl-fs/model')
AUDIT=ROOT/'history/risk_quantile_20260930'
pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
results={}
for unit in ['V54','V55','V56','V57']:
    for key in ['model','analysis']:sys.modules.pop(key,None)
    d=ROOT/'experiments'/unit;sys.path.insert(0,str(d))
    import model as m
    import analysis as a
    assert len(m.fixed_files())==159
    m.validate_layout(allow_missing=True)
    n=len(m.feature_columns(m.metadata()));market=len(m.market_columns(m.metadata()))
    assert n==105 and market==61
    torch.manual_seed(7)
    net=m.PredictModel(n,market).eval()
    x=torch.randn(20,8,n);state=torch.randn(market)
    with torch.no_grad():
        q=net.quantiles(x,state);score=net(x,state)
        assert torch.all(q[:,0]<=q[:,1]) and torch.all(q[:,1]<=q[:,2])
        assert torch.allclose(score[:,0],q[:,1]-.5*(q[:,1]-q[:,0]))
        assert torch.allclose(net(x[:3],state),score[:3],atol=2e-5,rtol=2e-5)
        gpu=net.to('cuda')(x.to('cuda'),state.to('cuda')).cpu()
        assert torch.allclose(score,gpu,atol=2e-5,rtol=2e-5)
    net=net.cpu()
    target=torch.linspace(-.08,.12,20)
    net.train();loss=m.quantile_loss(net.quantiles(x,state),target)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    # Toy panel has exactly the same causal gather mechanism as the full panel.
    toy=object.__new__(m.Panel)
    rng=np.random.default_rng(11)
    toy.X=rng.normal(size=(24,20,n)).astype(np.float32)
    toy.coverage=np.ones((24,20),np.float32);toy.codes=np.arange(20)
    toy.M=rng.normal(size=(24,market)).astype(np.float32);toy.market_dim=market
    toy.Y={f'label_ret_{h}d':rng.normal(0,.04,size=(24,20)).astype(np.float32) for h in [1,3,5,10,20]}
    days=np.arange(9,14)
    class ToyPrices:
        hot_frac=1.0
        def next_entry_ok(self):return np.ones((24,20),bool)
    net.eval();before=m.predict_quantiles(net,toy,np.array([12]),torch.device('cpu'))
    toy.X[13:]=1e5;toy.M[13:]=-1e5
    after=m.predict_quantiles(net,toy,np.array([12]),torch.device('cpu'))
    assert np.array_equal(before,after,equal_nan=True)
    metrics=m.validation(net,toy,ToyPrices(),days,torch.device('cpu'))
    assert all(k in metrics for k in ['IC_1d','IC_5d','RankIC_1d','RankIC_5d','quantile_pinball_loss','quantile_empirical_coverage'])
    assert abs(metrics['val_wei']+metrics['quantile_pinball_loss'])<1e-12
    qhat=m.predict_quantiles(net,toy,days,torch.device('cpu'));check=[]
    for i,day in enumerate(days):
        raw=toy.Y['label_ret_5d'][day].astype(np.float64);center=raw-np.median(raw)
        error=center[:,None]-qhat[i];levels=np.array([.25,.5,.75])
        check.append(np.maximum(levels*error,(levels-1)*error).mean())
    assert abs(np.mean(check)-metrics['quantile_pinball_loss'])<1e-10
    # Report compatibility checks the changed head set, without inventing trained results.
    records=[dict(quarter=q,fold=f,best_validation=metrics,argmax_validation=metrics,bag=dict(topk=5,epochs=[1]))
             for q in m.LAYOUT_QUARTERS for f in m.LAYOUT_FOLDS]
    actual=m.axis()[0]
    real_days={(q,s['fold']):len(s['valid']) for q in m.LAYOUT_QUARTERS for s in m.splits(actual,quarter=q)}
    saved_reader=a.fold_valid_days
    a.fold_valid_days=lambda: real_days
    assert len(a.training_validation_rows(records))==5
    a.fold_valid_days=saved_reader
    assert a.bag_vs_argmax(records)['folds']==16
    clean=[(q,s['fold']) for q in m.LAYOUT_QUARTERS for s in m.splits(actual,quarter=q)
           if s['train'].max()<s['valid'].min()]
    assert len(clean)==4
    results[unit]=dict(stock_features=n,market_features=market,source_scripts=3,fixed_files=159,
        ordered_quantiles=True,finite_training_gradient=True,causal_future_invariance=True,
        cpu_gpu_max_error=float(torch.max(torch.abs(score-gpu))),stock_subset_inference=True,
        proper_loss_selection=True,legacy_report_fields=True,clean_blocks=clean)
    print('QUANTILE_PREFLIGHT_OK',unit,flush=True)
    sys.path.pop(0)
(AUDIT/'preflight.json').write_text(json.dumps(dict(ok=True,checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),
    test_data='synthetic numerical/causal checks; no financial performance measured',results=results),indent=2))
print('ALL_QUANTILE_PREFLIGHT_OK',flush=True)
