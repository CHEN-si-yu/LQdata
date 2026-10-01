"""Reinfer all old fixed fold models on the same frozen test snapshot, read-only."""
from pathlib import Path
import gc
import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import torch

ROOT=Path('/root/autodl-fs/model');A=ROOT/'history/fixed_test_review_20260930';OUT=A/'legacy_references';OUT.mkdir(exist_ok=True)
FROZEN=ROOT/'history/diversification_20260930/trainingdata_frozen'
os.nice(14);pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
protocol=json.loads((A/'legacy_protocol.json').read_text());records=[];components=[];shared_days=None;policies=[]


def account(a,scores,panel,px,days,spec):
    st,curve,trades=a.cash_backtest(scores,panel,px,days,n=spec['topn'],period=spec['period'],band=spec['band'])
    fees=sum(t['fee'] for t in trades)
    slip=sum(t['quantity']*t['price']*(.0003/(1.0003 if t['side']=='buy' else .9997)) for t in trades)
    eq=np.asarray([r['equity'] for r in curve]);daily=eq/np.r_[100000.,eq[:-1]]-1
    q=a.quarter_metrics('ensemble',spec['name'],curve,trades,scores,panel,days)
    assert abs(np.prod([1+t['return_value'] for t in q])-1-st['return_value'])<1e-10
    return dict(strategy=spec['name'],frozen_settings=spec,net_return=st['return_value'],max_drawdown=st['max_drawdown'],
        sharpe=st['sharpe'],trades=st['trades'],total_fees=fees,slippage_cost=slip,
        gross_return_same_positions=st['return_value']+(fees+slip)/100000.,fee_drag=fees/100000.,slippage_drag=slip/100000.,
        quarters=q,net_daily_mean_annual_pct=float(daily.mean()*242*100),
        cost_definition='same actual positions; add measured fees and3bps executed-side slippage back as idle cash; no independently reinvested gross account'),eq,np.asarray([r['date'] for r in curve])


for unit in protocol['units']:
    if (A/'STOP').exists() or (ROOT/'history/diversification_20260930/STOP').exists():raise SystemExit('stopped by user')
    d=ROOT/'experiments'/unit
    for name in ['model','analysis']:sys.modules.pop(name,None)
    sys.path.insert(0,str(d));import model as m;import analysis as a
    assert m.data_root().resolve()==FROZEN.resolve()
    panel=m.Panel(m.RECIPE['features']);px=m.Prices(panel)
    print('REFERENCE_FIXED_LOAD',unit,len(panel.features),flush=True)
    folds=[];dates=None;weights={};evidence=[]
    for f in range(1,5):
        chunks=[];fd=[]
        for q in protocol['quarters']:
            folder=d/'model_train'/q/f'fold{f}';done=json.loads((folder/'complete.json').read_text())
            sp=m.splits(panel.days,quarter=q)[f-1];days=sp['test']
            assert np.array_equal(panel.days[days],done['test_dates'])
            stored=json.loads((folder/'split.json').read_text())
            train_end=max(r['end'] for r in stored['train']);valid_end=max(r['end'] for r in stored['valid'])
            assert train_end==str(panel.days[sp['train'].max()]) and valid_end==str(panel.days[sp['valid'].max()])
            assert max(sp['train'].max(),sp['valid'].max())+m.RECIPE['purge_horizon']+1<days.min()
            cp=folder/'best.pt';before=hashlib.sha256(cp.read_bytes()).hexdigest()
            ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['features']==panel.features
            net=m.PredictModel(len(panel.features),panel.market_dim);net.load_state_dict(ck['model']);net.ridge=ck.get('ridge');net.eval()
            pred=m.predict(net,panel,px,days,torch.device('cpu')) if len(inspect.signature(m.predict).parameters)==5 else m.predict(net,panel,days,torch.device('cpu'))
            assert pred.shape==(len(days),len(panel.codes))
            assert hashlib.sha256(cp.read_bytes()).hexdigest()==before
            weights[f'{q}_fold{f}']=before;chunks.append(pred);fd.extend(done['test_dates'])
            evidence.append(dict(quarter=q,fold=f,train_max=train_end,valid_max=valid_end,test_start=done['test_dates'][0],
                max_label_end_before_test=True,test_first_signal=str(panel.days[days[0]])))
        if dates is None:dates=fd
        else:assert dates==fd
        folds.append(np.concatenate(chunks))
    pred=sum(folds).astype(np.float32);days=np.searchsorted(panel.days,dates);pred,days=a.evaluation_slice(pred,panel,days);assert len(days)==242
    if shared_days is None:shared_days=days
    else:assert np.array_equal(shared_days,days)
    components.append(pred.copy())
    policy=dict(entry_rule=m.RECIPE['entry_rule'],money=m.RECIPE['backtest_money'],strategies=a.STRATEGIES)
    policies.append(policy)
    np.savez_compressed(OUT/f'{unit}_242.npz',scores=pred,days=days,dates=panel.days[days])
    accounts=[]
    for spec in a.STRATEGIES:
        row,eq,cashdates=account(a,pred,panel,px,days,spec);accounts.append(row)
        np.savez_compressed(OUT/f'{unit}_{spec["name"]}_cash.npz',equity=eq,dates=cashdates)
    record=dict(unit=unit,signal_days=len(days),weight_sha256=weights,split_evidence=evidence,IC=a.ic_table(pred,panel,days),accounts=accounts,
        inference='current frozen snapshot, all16 existing fold weights reloaded read-only; original stored split bounds verified; four raw-fold sum; not stale saved old forecasts')
    records.append(record)
    (OUT/f'{unit}.json').write_text(json.dumps(record,indent=2))
    print('REFERENCE_FIXED_RECORDED',unit,[(r['strategy'],r['net_return']) for r in accounts],flush=True)
    if unit!=protocol['units'][-1]:
        sys.path.pop(0);del panel,px,pred,net,ck,m,a;gc.collect()

# All three controls have the same execution rules. Their raw scores are combined in one account.
assert all(p==policies[0] for p in policies)
combined=sum(components)/len(components);accounts=[]
for spec in a.STRATEGIES:
    row,eq,cashdates=account(a,combined,panel,px,shared_days,spec);accounts.append(row)
    np.savez_compressed(OUT/f'legacy_raw_mean3_{spec["name"]}_cash.npz',equity=eq,dates=cashdates)
np.savez_compressed(OUT/'legacy_raw_mean3_242.npz',scores=combined,days=shared_days,dates=panel.days[shared_days])
value=dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed_units=3,records=records,
    old_family_single_account=dict(models=protocol['units'],score_rule='equal mean of each model raw fourfold sum, no cross-sectional normalization',
        accounts=accounts,scope='single100000 executable account, not average cash curves; three closely related old models are not three independent seeds'),
    no_mutations='release and old experiment code/weights/forecasts unchanged; new inference only in research history',
    selection='frozen benchmark only; no new model or parameter selected using test; survivorship and source factor PIT limitations remain')
(A/'legacy_fixed_test_summary.json').write_text(json.dumps(value,indent=2));print('ALL_OLD_FIXED_TEST_CONTROLS_COMPLETE',flush=True)
