"""逐字段台账。旧覆盖来自2026-09-20读取审计+qf第一批，新覆盖来自显式计算依赖。

不把读取了但未进入公式的列当新增覆盖，不把本工具声称为全库重新instrumentation。
用法：python scripts/audit_field_coverage.py --output artifacts/audits/field_coverage_20260921
"""
from pathlib import Path
import argparse
import csv
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pyarrow.parquet as pq
import factors
from fea.config import load
from fea.spec import REGISTRY
from fea.field_expansion import CATALOG
from factors.field_events import SOURCE_FIELDS as EVENTS
from factors.field_markets import SOURCE_FIELDS as MARKETS

def audit(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((ROOT/'conf/field_coverage_baseline.json').read_text(encoding='utf8'))
    exclusions=json.loads((ROOT/'conf/field_exclusions.json').read_text(encoding='utf8'))
    refs={}
    mapping={e['name']:e['source_fields'] for e in CATALOG}|EVENTS|MARKETS
    for name,sources in mapping.items():
        if name not in REGISTRY or not REGISTRY[name].enabled:continue
        for ds,fields in sources.items():
            for field in fields:refs.setdefault((ds,field),[]).append(name)
    rows=[];summary=[]
    for ds in sorted(baseline):
        files=sorted((load().upstream/ds).glob('year=*/data.parquet')) or list((load().upstream/ds).glob('data.parquet'))
        if not files:raise FileNotFoundError(ds)
        fields=pq.read_schema(files[-1])
        b=baseline[ds];old=set(b['used']);analytic=set(b['analytic'])
        for f in fields:
            names=refs.get((ds,f.name),[])
            if f.name in old:
                status='existing';reason='2026-09-20项目读取审计；另含2026-09-21 qf首批5字段'
            elif names and f.name in analytic:
                status='new';reason='进入本批已启用因子的计算公式/有效样本筛选'
            elif f.name not in analytic:
                status='key_or_metadata';reason='主键/报告身份/日期等结构列，沿用旧审计681分母，不计为分析覆盖'
            else:
                status='deferred';reason=exclusions.get(ds+'.'+f.name)
                if not reason:raise ValueError('缺少未开发原因: '+ds+'.'+f.name)
            rows.append(dict(dataset=ds,field=f.name,dtype=str(f.type),legacy_analytic=int(f.name in analytic),
                status=status,new_factors=';'.join(sorted(names)),reason=reason))
        own=[r for r in rows if r['dataset']==ds]
        summary.append(dict(dataset=ds,total=len(own),analytic=len(analytic),existing=len(old),
                            newly_used=sum(r['status']=='new' for r in own),deferred=sum(r['status']=='deferred' for r in own)))
    with (output/'FIELD_COVERAGE.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    totals={k:sum(s[k] for s in summary) for k in ['total','analytic','existing','newly_used','deferred']}
    result={'method':'旧读取审计基线+新公式显式源依赖；不是重新运行全部旧因子的读取审计','totals':totals,'datasets':summary}
    (output/'field_coverage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    md=['# 上游字段覆盖台账','',result['method']+'。所有未接入字段均在CSV中列出具体原因。',
        '',f"总字段 {totals['total']}；沿用旧口径的分析字段 {totals['analytic']}；此前已使用 {totals['existing']}；本批新增使用 {totals['newly_used']}。",
        '', '旧681分母包含ths_code、board_name等少量标识；本次保留分母以便对比，不把标识当新数值信号。',
        '', '| 数据集 | 分析字段 | 此前使用 | 本批新增 | 暂不接入 |','|:--|--:|--:|--:|--:|']
    md += [f"| {s['dataset']} | {s['analytic']} | {s['existing']} | {s['newly_used']} | {s['deferred']} |" for s in summary]
    md += ['', '精确字段、因子名称及原因见同目录 FIELD_COVERAGE.csv。覆盖率不代表因子有效性、收益或彼此独立。']
    (output/'FIELD_COVERAGE.md').write_text('\n'.join(md)+'\n',encoding='utf8')
    print(json.dumps(totals,ensure_ascii=False));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    audit(parser.parse_args().output)
