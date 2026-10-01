"""Repair missing reporting fields by actual inference, preserving trained weights/selection."""
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

ROOT=Path('/root/autodl-fs/model')
AUDIT=ROOT/'history/diversification_20260930'
BACKUP=AUDIT/'report_fields_originals'
BACKUP.mkdir(exist_ok=True)
pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
required=[f'IC_{h}d' for h in [1,3,5,10,20]]
repairs=[]


def missing(record):
    return [key for key in required if key not in record]


def measurements(m, panel, scores, days, keys):
    result={}
    for key in keys:
        horizon=int(key.split('_')[1][:-1]);values=[]
        for p,d in zip(scores,days):
            known=np.isfinite(p);y=panel.Y[f'label_ret_{horizon}d'][d]
            values.append(m.corr(p[known],np.where(np.isfinite(y[known]),y[known],0.)))
        result[key]=float(np.mean(values))
    return result


while True:
    if (AUDIT/'STOP').exists():raise SystemExit('stopped by user')
    for unit in [f'V{i}' for i in range(46,54)]:
        d=ROOT/'experiments'/unit;pending=[]
        for p in (d/'model_train').glob('*/fold*/complete.json'):
            o=json.loads(p.read_text())
            if missing(o['best_validation']) or missing(o.get('argmax_validation',{})):
                pending.append(p)
        if not pending:continue
        for name in ['model','analysis']:sys.modules.pop(name,None)
        sys.path.insert(0,str(d));import model as m
        panel=m.Panel();prices=m.Prices(panel)
        for path in pending:
            if (AUDIT/'STOP').exists():raise SystemExit('stopped by user')
            text=path.read_text();o=json.loads(text)
            q,f=o['quarter'],o['fold'];va=m.splits(panel.days,quarter=q)[f-1]['valid']
            ck=torch.load(path.parent/'best.pt',map_location='cpu',weights_only=False)
            assert ck['features']==panel.features
            net=m.PredictModel(len(panel.features),panel.market_dim);net.ridge=ck.get('ridge')
            archive=BACKUP/unit/f'{q}_fold{f}_complete.json';archive.parent.mkdir(exist_ok=True)
            if not archive.exists():archive.write_text(text)
            changed={}
            keys=missing(o['best_validation'])
            if keys:
                cached=AUDIT/'clean_innovations'/f'{unit}_{q}_fold{f}.npz'
                if cached.exists():
                    data=np.load(cached);assert np.array_equal(data['days'],va)
                    scores=data['scores']
                else:
                    net.load_state_dict(ck['model'])
                    scores=m.predict(net,panel,prices,va,torch.device('cpu'))
                values=measurements(m,panel,scores,va,keys)
                o['best_validation'].update(values);changed['bagged']=values
            keys=missing(o.get('argmax_validation',{}))
            if keys:
                assert o.get('argmax_validation'), 'missing argmax record cannot be invented'
                if 'model_argmax' not in ck:
                    assert o['bag']['used']<=1
                net.load_state_dict(ck.get('model_argmax',ck['model']))
                scores=m.predict(net,panel,prices,va,torch.device('cpu'))
                values=measurements(m,panel,scores,va,keys)
                o['argmax_validation'].update(values);changed['argmax']=values
            original=json.loads(text)
            for section in ['best_validation','argmax_validation']:
                for key,val in original[section].items():assert o[section][key]==val
            o['report_fields_backfill']=dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                snapshot=str(panel.root),old_complete_sha256=hashlib.sha256(text.encode()).hexdigest(),
                changes=changed,weights_modified=False,original_metrics_preserved=True,
                source='actual saved bagged/argmax weights; raw target labels on exact original validation dates')
            m.atomic_json(path,o)
            repairs.append(dict(unit=unit,quarter=q,fold=f,fields=changed))
            (AUDIT/'report_backfill_status.json').write_text(json.dumps(dict(
                checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),repairs=repairs),indent=2))
            print('REPORT_REPAIRED',unit,q,f,changed,flush=True)
        sys.path.pop(0);del panel,prices,net,ck,scores,m;gc.collect()
    state=json.loads((AUDIT/'batch_status.json').read_text())
    if state.get('failed'):raise SystemExit('batch failure; inspect before further repairs')
    if state.get('phase')=='complete':break
    time.sleep(20)
print('REPORT_BACKFILL_COMPLETE',flush=True)
