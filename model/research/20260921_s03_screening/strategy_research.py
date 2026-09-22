"""预登记的小排名缓冲对照：只读各版本预测，不导入其它版本代码。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import evaluation_core as M
import evaluation_core as A

def source_predictions(path, panel):
    variants={}
    provenance=[]
    dates=None
    for fold in range(1,5):
        chunks=[]
        these=[]
        for quarter in M.quarters(panel.days):
            folder=path/'model_train'/quarter/f'fold{fold}'
            done=json.loads((folder/'complete.json').read_text())
            lock=json.loads((folder/'recipe.lock.json').read_text())
            assert done['run_id']==lock['run_id']
            assert lock['panel_digest']==panel.meta['panel_digest']
            files={str(y):panel.meta['years'][str(y)]['files'] for y in panel.meta['built_years']}
            assert lock['files']==files,'数据内容与原始训练来源不一致'
            predfile=folder/'test_predictions.npy'
            pred=np.load(predfile)
            assert pred.shape==(len(done['test_dates']),len(panel.codes))
            chunks.append(pred)
            these.extend(done['test_dates'])
            provenance.append(dict(quarter=quarter,fold=fold,run_id=done['run_id'],device=done['device'],
                                   prediction_sha256=hashlib.sha256(predfile.read_bytes()).hexdigest()))
        assert dates is None or dates==these
        dates=these
        variants[f'fold{fold}']=np.concatenate(chunks)
    variants['ensemble']=sum(variants.values()).astype(np.float32)
    assert dates[0]=='2025-07-01' and dates[-1]=='2026-06-30' and len(dates)==242
    return variants,np.searchsorted(panel.days,dates),provenance

def quarter_cash(curve,days,panel):
    prev=100000.
    cursor=0
    out=[]
    qs=M.quarters(panel.days)
    for i,q in enumerate(qs):
        qix=[d for d in days if str(pd.Period(panel.days[d],freq='Q'))==q]
        end=panel.days[qix[-1]+(2 if i==len(qs)-1 else 1)]
        start=cursor
        while cursor<len(curve) and curve[cursor]['date']<=end:
            cursor+=1
        eq=np.r_[prev,[r['equity'] for r in curve[start:cursor]]]
        out.append(dict(quarter=q,net=float(eq[-1]/prev-1),maxdd=float((eq/np.maximum.accumulate(eq)-1).min())))
        prev=eq[-1]
    assert cursor==len(curve)
    assert np.isclose(np.prod([1+x['net'] for x in out]),curve[-1]['equity']/100000)
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sources',nargs='+',default=['V22','V28','V29','V30'])
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    out=root/'strategy_research'
    out.mkdir(exist_ok=True)
    specs=[dict(name='top1_1d',topn=1,period=1),dict(name='top5_1d',topn=5,period=1)]
    specs += [s for s in A.STRATEGIES if s['name'].startswith('buffer_')]
    M.atomic_json(out/'plan.json',dict(sources=args.sources,strategies=specs,
                   status='predeclared_fixed_grid',selection_bias='Fixed research window is already repeatedly inspected; no independent holdout claim.'))
    panel=M.Panel(load_x=False)
    px=M.Prices(panel)
    rows=[]
    quarters=[]
    origins={}
    for source in args.sources:
        variants,days,origin=source_predictions(root.parent/source,panel)
        origins[source]=origin
        px.verify_labels(panel,days)
        for name,pred in variants.items():
            baseline={}
            for spec in specs:
                stat,curve,trades=A.cash_backtest(pred,panel,px,days,n=spec['topn'],
                                                period=spec.get('period',1),exit_rank=spec.get('exit_rank'),min_hold=1)
                eq=np.r_[100000.,[x['equity'] for x in curve]]
                returns=eq[1:]/eq[:-1]-1
                if spec['name'] in ['top1_1d','top5_1d']:
                    baseline[spec['topn']]=stat
                ref=baseline[spec['topn']]
                qrows=quarter_cash(curve,days,panel)
                row=dict(unit=source,source=name,strategy=spec['name'],
                         delta_net=stat['return_value']-ref['return_value'],delta_fees=stat['total_fees']-ref['total_fees'],
                         delta_maxdd=stat['max_drawdown']-ref['max_drawdown'],positive_quarters=sum(x['net']>0 for x in qrows),
                         remove_best1=float(np.prod(1+np.sort(returns)[:-1])-1),
                         remove_best2=float(np.prod(1+np.sort(returns)[:-2])-1),
                         remove_best5=float(np.prod(1+np.sort(returns)[:-5])-1),**stat)
                rows.append(row)
                quarters.extend(dict(unit=source,source=name,strategy=spec['name'],**q) for q in qrows)
                if name=='ensemble':
                    pd.DataFrame(curve).to_csv(out/f'{source}_{spec["name"]}_curve.csv',index=False)
                    pd.DataFrame(trades).to_csv(out/f'{source}_{spec["name"]}_trades.csv',index=False)
            print(source,name,'completed',flush=True)
        pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
        pd.DataFrame(quarters).to_csv(out/'quarters.csv',index=False)
        M.atomic_json(out/'prediction_provenance.json',origins)
    df=pd.DataFrame(rows)
    summary=[]
    for strategy,g in df.groupby('strategy'):
        fold=g[g.source!='ensemble']
        ens=g[g.source=='ensemble']
        summary.append(dict(strategy=strategy,ensemble_pairs=len(ens),ensemble_better=int((ens.delta_net>0).sum()),
                            ensemble_mean_delta=float(ens.delta_net.mean()),ensemble_min_delta=float(ens.delta_net.min()),
                            fold_pairs=len(fold),fold_better=int((fold.delta_net>0).sum()),fold_median_delta=float(fold.delta_net.median())))
    pd.DataFrame(summary).to_csv(out/'robustness.csv',index=False)
    text='# 小排名缓冲策略：预登记对照\n\n全部为固定研究窗内结果；各折共享时间窗，不能当独立年份。\n\n'
    cols=['策略','集成胜出/总数','集成平均收益差','集成最差收益差','单折胜出/总数','单折收益差中位数']
    text+=A.markdown_table(cols,[[r['strategy'],f"{r['ensemble_better']}/{r['ensemble_pairs']}",f"{r['ensemble_mean_delta']:+.2%}",
                                f"{r['ensemble_min_delta']:+.2%}",f"{r['fold_better']}/{r['fold_pairs']}",f"{r['fold_median_delta']:+.2%}"] for r in summary])
    text+='\n比较对象为相同持仓数的每日换手；净收益含项目既定费用、滑点、整手和拒单规则。原始曲线、交易和来源指纹都保存在本目录。\n'
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    print(text,flush=True)

if __name__=='__main__':
    main()
