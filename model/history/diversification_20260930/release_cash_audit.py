"""Recompute live-release accounts in scratch, with no writes to releases."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path('/root/autodl-fs/model')
AUDIT = ROOT / 'history/diversification_20260930'
sys.path.insert(0, str(ROOT / 'experiments/V38'))
import model as m
import analysis as a
torch.set_num_threads(2)

panel = m.Panel(load_x=False)
prices = m.Prices(panel)
series, positions, accounts = {}, {}, {}
expected = json.loads((ROOT/'releases/ensemble.json').read_text())
for unit in ['V31','V36','V38']:
    if (AUDIT/'STOP').exists():
        raise SystemExit('stopped by user')
    scores=[]; dates=None
    for f in range(1,5):
        d=ROOT/'releases'/unit/'model_train/2026Q3'/f'fold{f}'
        meta=json.loads((d/'complete.json').read_text())
        if dates is None:dates=meta['score_dates']
        assert dates==meta['score_dates']
        scores.append(np.load(d/'score_predictions.npy'))
    prediction=sum(scores)
    ix=np.searchsorted(panel.days,dates)
    assert panel.days[ix].tolist()==dates
    # The fixed account liquidates at T+2; missing future prices are excluded, never fabricated.
    ok=ix+2<len(panel.days)
    assert ok.sum()==expected['eval_window']['backtest_days']
    stat,curve,trades=a.cash_backtest(prediction[ok],panel,prices,ix[ok],n=5,period=1,band=40)
    eq=np.asarray([c['equity'] for c in curve]);date=[c['date'] for c in curve]
    series[unit]=pd.Series(eq/np.r_[100000.,eq[:-1]]-1,index=date)
    current={};history={}
    trades_by_date={}
    for t in trades:trades_by_date.setdefault(t['date'],[]).append(t)
    for day in date:
        for t in trades_by_date.get(day,[]):
            if t['side']=='buy':current[t['code']]=True
            else:current.pop(t['code'],None)
        history[day]=set(current)
    positions[unit]=history
    ref=expected['per_unit'][unit]['account']['compounded_pct']/100
    assert abs(stat['return_value']-ref)<2e-5, (unit,stat['return_value'],ref)
    accounts[unit]=dict(net_return=stat['return_value'],max_drawdown=stat['max_drawdown'],fees=stat['total_fees'],trades=stat['trades'])
aligned=pd.concat(series,axis=1).dropna()
overlap=[]
for i,u in enumerate(positions):
    for v in list(positions)[i+1:]:
        days=sorted(set(positions[u])&set(positions[v]));values=[]
        for day in days:
            x,y=positions[u][day],positions[v][day]
            if x|y:values.append(len(x&y)/len(x|y))
        overlap.append(dict(pair=[u,v],holdings_jaccard_mean=float(np.mean(values))))
result=dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),signal_window=expected['eval_window'],
            execution_window=[str(aligned.index[0]),str(aligned.index[-1])],execution_days=len(aligned),
            net_daily_return_correlation=aligned.corr().to_dict(),actual_holdings_overlap=overlap,
            accounts=accounts,existing_release_account_reproduced=True)
(AUDIT/'release_cash_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('RELEASE_CASH_AUDIT',json.dumps(result,ensure_ascii=False),flush=True)
