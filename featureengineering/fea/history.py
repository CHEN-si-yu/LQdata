"""Prune factor outputs before the configured start, retaining checksum evidence."""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from . import store
from .manifest import Manifest, _atomic_json
from .spec import all_specs

def run(cfg, apply=False, names=None):
    selected=[s for s in all_specs() if not names or s.name in names]
    records=[]
    root=cfg.factors_dir.resolve()
    for s in selected:
        cutoff=s.resolved_start(cfg)
        for p in sorted((cfg.factors_dir/s.name).glob('year=*/data.parquet')):
            if p.is_symlink() or not p.resolve().is_relative_to(root):
                raise ValueError(f'Unsafe factor partition: {p}')
            y=int(p.parent.name[5:])
            if y>int(cutoff[:4]):continue
            action='delete'
            if y==int(cutoff[:4]):
                import pyarrow.parquet as pq
                days=pq.read_table(p,columns=['trade_date']).column('trade_date').to_pylist()
                if not any(d<cutoff for d in days):continue
                action='trim'
            st=p.stat()
            rec={'factor':s.name,'path':str(p),'year':y,'cutoff':cutoff,'action':action,
                 'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
            if apply:
                h=hashlib.md5(usedforsecurity=False)
                with p.open('rb') as f:
                    while b:=f.read(8*1024**2):h.update(b)
                rec['md5']=h.hexdigest()
            records.append(rec)
    print(f'起点裁剪：{len(records)} 个分区，原文件合计 {sum(r["bytes"] for r in records)/1024**3:.2f} GiB；apply={apply}',flush=True)
    if not apply or not records:return records
    audit=cfg.state_dir/'history_prune'/datetime.now().strftime('%Y%m%d_%H%M%S')
    audit.mkdir(parents=True,exist_ok=False)
    _atomic_json({'files':records,'note':'Hashes cannot restore deleted contents.'},audit/'checksums.json')
    for s in selected:
        mine=[r for r in records if r['factor']==s.name]
        if not mine:continue
        man=Manifest.load(cfg.state_dir,s.name)
        if man.path.exists():
            _atomic_json(json.loads(man.path.read_text()),audit/(s.name+'.json'))
        for r in mine:
            p=Path(r['path']);st=p.stat()
            if st.st_size!=r['bytes'] or st.st_mtime_ns!=r['mtime_ns']:
                raise RuntimeError(f'Partition changed before pruning: {p}')
            if r['action']=='delete':
                p.unlink()
                if not any(p.parent.iterdir()):p.parent.rmdir()
                man.partitions.pop(str(r['year']),None)
            else:
                df=store.read_year(cfg.factors_dir,s.name,r['year'])
                df=df[df.trade_date>=r['cutoff']].reset_index(drop=True)
                store._atomic_write(df,p,cfg.compression)
                man.mark_partition(r['year'],len(df),int(np.isfinite(df.value.to_numpy()).sum()),str(df.trade_date.min()),str(df.trade_date.max()))
        cutoff=s.resolved_start(cfg)
        man.coverage=[[max(a,cutoff),b] for a,b in man.coverage if b>=cutoff]
        man.save()
    _atomic_json({'complete':True,'partitions':len(records),'logical_bytes':sum(r['bytes'] for r in records)},audit/'result.json')
    print('裁剪完成，校验记录：',audit,flush=True)
    return records
