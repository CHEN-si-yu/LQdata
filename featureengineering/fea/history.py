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

def cutoff_of(spec, cfg) -> str:
    """该因子允许保留的最早日期（早于它的分区会被裁掉）。

    ★★ 2026-09-25：**必须跟着对齐契约走**。声明了对齐（`align_fill is not None`，
    现在这是默认值）的因子，其分区下界是 `align_start`（= `default_start`），
    `resolved_start` 之前的年份是**合法的对齐分区**；按 `resolved_start` 裁会把它们
    整年删掉，并把台账 coverage 同步收紧（账实一起消失、检测不到）。
    实测：按旧口径会删/裁 35 个分区、18 个因子。
    只有显式关掉对齐（`align_fill=None`）的因子才按真实起点裁。
    """
    return spec.resolved_start(cfg) if spec.align_fill is None else spec.align_start(cfg)


def run(cfg, apply=False, names=None):
    """把配置起点**之前**的旧产物裁掉（先预览，`apply=True` 才真删）。

    ★★ 2026-09-25 修：裁剪下界必须跟着**对齐契约**走，否则会删掉合法产物。

    原来下界用 `s.resolved_start(cfg)`（**真实值起点**）。但新契约要求起点之前的
    年份也必须有分区（对齐区间，见 `FactorSpec.align_fill`）—— 按真实起点裁，
    会把这些**刚产出的对齐分区整年删掉**，并把台账 `coverage` 同步收紧，
    于是账实一起消失、任何"账实不符"检测都看不见。
    实测枚举（2026-09-25）：会删/裁 **35 个分区、18 个因子**，全部是 2018/2019
    的对齐分区；而 `main.py rebuild` 成功后**无条件调用本函数** ⇒ 下一轮重建即引爆。

    同时修掉一处口径不一致：原来用 `cfg.factors_dir` 取数，而市场因子在
    `data/market_factors/` 下 ⇒ **市场因子永远不会被裁剪**，同一批晚起点因子里
    股票因子的旧年被删、市场因子的留着。改用 `cfg.factor_root(s)`。

    声明了对齐的因子，下界取 `align_start`（= `default_start`）；`align_fill=None`
    的因子（显式关掉对齐）维持原行为。
    """
    selected=[s for s in all_specs() if not names or s.name in names]
    records=[]
    for s in selected:
        root=cfg.factor_root(s).resolve()
        cutoff=cutoff_of(s,cfg)
        for p in sorted((cfg.factor_root(s)/s.name).glob('year=*/data.parquet')):
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
                df=store.read_year(cfg.factor_root(s),s.name,r['year'])
                df=df[df.trade_date>=r['cutoff']].reset_index(drop=True)
                store._atomic_write(df,p,cfg.compression)
                man.mark_partition(r['year'],len(df),int(np.isfinite(df.value.to_numpy()).sum()),str(df.trade_date.min()),str(df.trade_date.max()))
                # ★ trim 分支也要补回文件身份：mark_partition 会整条覆盖该年条目，
                #   不补就把「分区被外部改动」的检测关掉了（见 rebuild_manifest 的同名坑）。
                _st=p.stat()
                man.partitions[str(r['year'])]['file_identity']=[_st.st_size,_st.st_mtime_ns]
        cutoff=cutoff_of(s,cfg)
        man.coverage=[[max(a,cutoff),b] for a,b in man.coverage if b>=cutoff]
        man.save()
    _atomic_json({'complete':True,'partitions':len(records),'logical_bytes':sum(r['bytes'] for r in records)},audit/'result.json')
    print('裁剪完成，校验记录：',audit,flush=True)
    return records
