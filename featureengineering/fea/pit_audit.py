"""Dynamic PIT checks with complete comparisons and dependency-aware rebuilds."""
from pathlib import Path
import copy
import gc
import json
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

from .dates import int_to_str,str_to_int
from .engine import Engine,minus_days
from .manifest import Manifest
from .resources import safe_jobs
from .spec import REGISTRY
from . import store


def compare_frames(reference, candidate, tol):
    if reference.empty or candidate.empty:
        raise ValueError('Missing production or recomputed rows')
    keys=['trade_date','stock_code']
    if reference.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise ValueError('Duplicate factor keys')
    a=reference.sort_values(keys,ignore_index=True)
    b=candidate.sort_values(keys,ignore_index=True)
    if len(a)!=len(b) or not a[keys].equals(b[keys]):
        raise ValueError('Date/stock grid differs')
    mismatches={}
    for column in ('value','rank'):
        x=a[column].to_numpy(dtype=np.float64)
        y=b[column].to_numpy(dtype=np.float64)
        bad=~np.isclose(x,y,rtol=tol,atol=0,equal_nan=True)
        if bad.any():mismatches[column]=int(bad.sum())
    if mismatches:raise ValueError(f'Value/rank/NaN mismatch: {mismatches}')
    return {'rows':len(a),'finite_values':int(np.isfinite(a.value.to_numpy()).sum())}


def parents_of(specs):
    names=set()
    pending=list(specs)
    while pending:
        spec=pending.pop()
        for dep in spec.deps:
            if dep in REGISTRY and dep not in names:
                names.add(dep);pending.append(REGISTRY[dep])
    return [REGISTRY[name] for name in sorted(names)]


def _compute(cfg,specs,start,end,jobs):
    if not specs:return
    eng=Engine(cfg);eng.override_start=start
    todo=eng.prebuild(specs,end,rebuild=True)
    remaining={s.name:(s,man,plan) for s,man,plan in todo}
    while remaining:
        wave=[item for item in remaining.values() if not any(dep in remaining for dep in item[0].deps)]
        if not wave:raise ValueError('Cyclic factor dependencies')
        tasks=[(s.name,year,days) for s,_,plan in wave for year,days in sorted(plan.items())]
        results=eng.run_parallel(tasks,min(jobs,len(tasks))) if jobs>1 else [eng.run_year(REGISTRY[name],year,days) for name,year,days in tasks]
        failed=[r for r in results if r.get('error')]
        if failed:raise RuntimeError(f'Recomputation failed: {failed}')
        for spec,man,plan in wave:
            own=[r for r in results if r['factor']==spec.name]
            if len(own)!=len(plan):raise RuntimeError(f'Incomplete recomputation: {spec.name}')
            eng._finalize(spec,man,plan,own,sum(r.get('seconds',0) for r in own))
            remaining.pop(spec.name)
    from . import engine as engine_module
    engine_module._WORKER_ENGINE=None
    del eng
    gc.collect()


def run_dynamic(args,cfg,specs):
    specs=[s for s in specs if not s.is_label]
    if not specs:
        print('标签允许未来收益，不参与因子的动态因果审计。')
        return 1
    baseline=Engine(cfg)
    end=min(str_to_int(args.end),baseline.baseline_last_day()) if args.end else baseline.baseline_last_day()
    dates=[str_to_int(v['max_date']) for s in specs for v in Manifest.load(cfg.state_dir,s.name).partitions.values() if v.get('max_date')]
    if not dates:
        print('没有已落盘的因子，不能完成动态比对。');return 1
    end=min(end,max(dates))
    days=baseline.cal.between(str_to_int(cfg.default_start),end)
    if not len(days):print('请求区间没有交易日。');return 1
    n=max(1,int(args.sample or 1))
    picks=sorted(set([int(days[len(days)//2])]+([int(d) for d in days[-(n-1):]] if n>1 else [])))
    jobs=safe_jobs(args.jobs)
    report={'sample_dates':[int_to_str(d) for d in picks],'comparisons':[],'errors':[],'skipped':[]}
    for cutoff in picks:
        sandbox=Path(tempfile.mkdtemp(prefix='factor_pit_'))
        local=copy.deepcopy(cfg)
        local.raw['_audit_cutoff'] = cutoff
        local.raw['paths'].update(factors=str(sandbox/'factors'),state=str(sandbox/'state'))
        active=[s for s in specs if s.start_int(cfg)<=cutoff]
        report['skipped'] += [[s.name,int_to_str(cutoff),'before effective start'] for s in specs if s not in active]
        try:
            parents=parents_of(active)
            if parents:
                floor=max(str_to_int(cfg.default_start),min(minus_days(cutoff//10000*10000+101,s.warmup_days) for s in active if any(d in REGISTRY for d in s.deps)))
                for year in range(floor//10000,cutoff//10000+1):
                    _compute(local,parents,max(floor,year*10000+101),min(cutoff,year*10000+1231),jobs)
            _compute(local,active,cutoff//10000*10000+101,cutoff,jobs)
            for spec in active:
                reference=store.read_year(cfg.factors_dir,spec.name,cutoff//10000)
                candidate=store.read_year(local.factors_dir,spec.name,cutoff//10000)
                if reference.empty or candidate.empty:
                    raise ValueError(f'{spec.name}: missing partition')
                day=int_to_str(cutoff)
                try:
                    rec=compare_frames(reference[reference.trade_date==day],candidate[candidate.trade_date==day],float(args.tol))
                    report['comparisons'].append({'name':spec.name,'date':day,**rec})
                except ValueError as exc:
                    report['errors'].append({'name':spec.name,'date':day,'error':str(exc)})
                    print(f'  ✘ {spec.name} @ {day}: {exc}',flush=True)
            print(f'  {int_to_str(cutoff)}：累计完成 {len(report["comparisons"])} 项比对，错误 {len(report["errors"])}',flush=True)
        except Exception as exc:
            report['errors'].append({'date':int_to_str(cutoff),'error':str(exc)})
            print(f'  ✘ 本采样日未能完整重算：{exc}',flush=True)
        finally:
            shutil.rmtree(sandbox)
    if not report['comparisons']:
        report['errors'].append({'error':'No completed comparisons'})
    dest=cfg.state_dir/'pit_audit'/time.strftime('%Y%m%d_%H%M%S')
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'动态审计报告：{dest / "report.json"}')
    return int(bool(report['errors']))
