import importlib.util, json, os, sys
from pathlib import Path
work=Path('/autodl-fs/data/model/r17_factor_validation_20261019')
man=json.load(open(work/'scripts/unit_manifest.json'))
for u in man:
    unit=Path('/autodl-fs/data/model')/u['unit']
    snap=Path('/autodl-fs/data/model')/u['snapshot']
    os.environ['MX_DATA']=str(snap)
    os.chdir(unit)
    spec=importlib.util.spec_from_file_location('m_'+unit.name, unit/'model.py')
    m=importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:
        print(unit.name,'IMPORT ERROR',repr(e)); continue
    r=m.RECIPE
    n=len(r['feature_names']); f=len(r['flow_feature_names']); uf=len(r['fund_feature_names']); t=len(r['technical_feature_names'])
    fi=r['flow_indices']; ui=r['fund_indices']; ti=r['technical_indices']
    problems=[]
    if len(set(r['feature_names']))!=n: problems.append('duplicate feature_names')
    for name,idx,key in (('flow',fi,'flow'),('fund',ui,'fund'),('tech',ti,'technical')):
        if len(set(idx))!=len(idx): problems.append(f'{name} dup idx')
        if any(i<0 or i>=n for i in idx): problems.append(f'{name} idx out of range')
        nameset={r['feature_names'][i] for i in idx}
        declared=set(r[f'{key}_feature_names'])
        if nameset!=declared: problems.append(f'{name} idx/name mismatch')
    if unit.name.startswith(('A_','B_')) and 'V220' in unit.name:
        if set(fi)&set(ui) or set(fi)&set(ti) or set(ui)&set(ti): problems.append('V220 branch overlap')
        if sorted(fi+ui+ti)!=list(range(n)): problems.append(f'V220 partition not complete {len(fi)+len(ui)+len(ti)} vs {n}')
    print(unit.name, 'n',n,'flow',f,'fund',uf,'tech',t,'idx',len(fi),len(ui),len(ti),'problems',problems,
          'end',r['snapshot_cutoff'],'pin',r['snapshot_meta_sha256'][:8],'name',r['name'])
