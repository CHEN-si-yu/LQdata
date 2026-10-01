"""Fresh technical absolute-return models, with copied independent source and no checkpoints."""
from pathlib import Path
import ast
import json
import hashlib
from absolute_ridge_components import TECHNICAL_FEATURES,MODEL,SOLVER,TRAIN

ROOT=Path('/root/autodl-fs/model')
PARENT=ROOT/'experiments/V46'
AUDIT=ROOT/'history/absolute_ridge_20260930'


def node(source,name):
    for n in ast.parse(source).body:
        if getattr(n,'name',None)==name:return n
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return n
    raise KeyError(name)


def replace(source,name,body):
    n=node(source,name);lines=source.splitlines(keepends=True)
    start=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1
    return ''.join(lines[:start])+body.rstrip()+'\n'+''.join(lines[n.end_lineno:])


def create(unit,seed,write=True):
    dest=ROOT/'experiments'/unit
    if write and dest.exists():raise FileExistsError(dest)
    model=(PARENT/'model.py').read_text().replace('top5_5d','top5_1d')
    analysis=(PARENT/'analysis.py').read_text().replace('top5_5d','top5_1d')
    run=(PARENT/'run.py').read_text()
    model=model.replace(model.splitlines()[0],repr(f'{unit}: technical absolute-return ridge; seed{seed}; fresh independent closed-form fits.'),1)
    recipe={kw.arg:ast.unparse(kw.value) for kw in node(model,'RECIPE').value.keywords}
    recipe.update(name=repr(unit),family=repr('technical_absolute_ridge'),seed=str(seed),
        architecture=repr('44 daily technical/reversal/risk inputs; fixed nonlinear basis plus61 causal market inputs; closed-form absolute1/5d forecasts'),
        label_horizons='[1,5]',label_weights='{1:.75,5:.25}',valid_ic_weights='{1:.75,5:.25}',
        # Keep the framework's raw1/5d report and5d label proxy; actual delivered forecast is the1d head.
        score_blend=repr('head:label_ret_1d'),score_label=repr('label_ret_5d'),prediction_horizon='1',
        market_gate='True',hidden='()',dropout='()',sequence_days='1',input_clip='5.',
        optimizer=repr('closed_form_ridge'),lr='0.',weight_decay='0.',max_epochs='5',
        sample_caps='[256,384,512,640,768]',stock_ridge='.05',market_ridge='.25',intercept_ridge='1e-8',
        target_clip='{1:.12,5:.30}',loss_scale='{1:.03,5:.08}',
        valid_criterion=repr('absolute_mse'),
        valid_metric=repr('negative day-equal normalized MSE of raw clipped absolute1/5d returns; epochs are nested per-date sample-coverage fits'),
        training_target=repr('raw next-open to future-open returns, symmetrically clipped; no daily rank/mean/std target normalization'),
        training_backend=repr('CPU normal equations; requested device used for inference and validation'),
        bag_topk='5',bag_recalibrate='False',early_stop_patience='6',
        hold_band='0',picks_holding='1',picks_topn='5',cost_gate_multiplier='1.0',
        cost_gate_candidates='[0.,.5,1.,2.]',ensemble_sum_members='4',
        weights_origin=repr('fresh independent normal-equation fits; V46 scripts only copied, no checkpoint reuse'),
        research_protocol=repr('four stock-sampling seeds3253/4253/5253/6253; same4quarters x4folds and21-day purge; no random weight initialization; all seeds retained; no test selection'),
        strategy_candidates=repr(dict(topn=[5,10,20],period=[1,5,10],band=[0,20,40],
            cost_gate_multiplier=[0.,.5,1.,2.],phase='all phases')))
    model=replace(model,'RECIPE','RECIPE=dict(\n'+''.join(f'    {k}={v},\n' for k,v in recipe.items())+')')
    features=f'''def feature_columns(meta):
    expected={TECHNICAL_FEATURES!r}
    missing=set(expected)-set(meta['columns']['features'])
    if missing:raise ValueError(f'missing frozen daily features: {{sorted(missing)}}')
    return [f for f in meta['columns']['features'] if f in set(expected)]
'''
    model=replace(model,'feature_columns',features)
    model=replace(model,'PredictModel',MODEL+'\n\n'+SOLVER)
    model=replace(model,'train_fold',TRAIN)
    model=model.replace("caps=recipe['sample_caps'],target_clip=recipe['target_clip'],loss_scale=recipe['loss_scale'],",
        "caps=recipe['sample_caps'],target_clip=recipe['target_clip'],loss_scale=recipe['loss_scale'],\n        input_clip=recipe['input_clip'],intercept_ridge=recipe['intercept_ridge'],\n        label_weights=recipe['label_weights'],valid_criterion=recipe['valid_criterion'],\n        market_columns=market_columns(metadata()),")
    valid=ast.unparse(node(model,'validation'))
    valid=valid.replace("criterion not in ('rankic', 'tradable_topn')","criterion not in ('rankic','tradable_topn','absolute_mse')")
    marker="primary = ret if criterion == 'tradable_topn' else RECIPE['valid_ic_scale'] * ic_term"
    assert valid.count(marker)==1
    criterion='''absolute_mse={}
for j,h in enumerate(model.horizons):
    errors=[]
    for i,d in enumerate(days):
        y=panel.Y[f'label_ret_{h}d'][d]
        ok=np.isfinite(heads[i,:,j])&np.isfinite(y)
        if ok.sum()<3:continue
        target=np.clip(y[ok],-RECIPE['target_clip'][h],RECIPE['target_clip'][h])
        errors.append(float(np.mean((heads[i,ok,j]-target)**2)))
    absolute_mse[h]=float(np.mean(errors)) if errors else float('inf')
normalized_mse=sum(RECIPE['label_weights'][h]*absolute_mse[h]/RECIPE['loss_scale'][h]**2 for h in model.horizons)/sum(RECIPE['label_weights'].values())
primary=-normalized_mse if criterion=='absolute_mse' else ret if criterion=='tradable_topn' else RECIPE['valid_ic_scale']*ic_term'''
    valid=valid.replace(marker,'\n    '.join(criterion.splitlines()))
    valid=valid.replace('return dict(val_wei=primary,','return dict(absolute_target_mse_by_horizon=absolute_mse,absolute_normalized_mse=normalized_mse,val_wei=primary,')
    model=replace(model,'validation',valid)
    # The fixed ensemble still sums four folds. Only the cost gate interprets its known scale.
    cash=ast.unparse(node(analysis,'cash_backtest'))
    cash=cash.replace('money=None):','money=None, forecast_members=1, cost_gate_multiplier=None):',1)
    money="money = float(RECIPE['backtest_money'] if money is None else money)"
    cash=cash.replace(money,money+"\n    if forecast_members<1:raise ValueError('invalid forecast member count')\n    gate=RECIPE['cost_gate_multiplier'] if cost_gate_multiplier is None else cost_gate_multiplier\n    cost_reference=2*max(.00025,5/(money/n))+.00002+.0005+.0006\n    score_threshold=forecast_members*gate*cost_reference")
    candidates="candidates = np.flatnonzero(np.isfinite(p))"
    assert cash.count(candidates)==1
    cash=cash.replace(candidates,candidates+"\n            utility=p.copy()\n            if n>1 and gate>0:\n                utility-=score_threshold\n                for held in holdings:utility[held]=p[held]\n                candidates=candidates[utility[candidates]>=0.]")
    rank="ranked = candidates[np.argsort(-p[candidates], kind='stable')]"
    assert cash.count(rank)==1
    cash=cash.replace(rank,"ranked = candidates[np.argsort(-utility[candidates], kind='stable')]")
    analysis=replace(analysis,'cash_backtest',cash)
    evaluator=ast.unparse(node(analysis,'evaluate_fixed_strategies'))
    call="cash_backtest(ev, panel, px, eval_days, n=spec['topn'], period=spec['period'], band=spec['band'])"
    assert evaluator.count(call)==1
    evaluator=evaluator.replace(call,call[:-1]+", forecast_members=RECIPE['ensemble_sum_members'])")
    evaluator=evaluator.replace("report['label_price_check'] = px.verify_labels(panel, days)",
        "report['absolute_forecast_policy']={'fold_score_units':'raw clipped-return forecast','ensemble':'sum4 folds, no normalization','cost_gate_scale':RECIPE['ensemble_sum_members'],'cost_proxy':'initial nominal equal slot, commission minimum+transfer+stamp+slippage; actual execution costs still separately charged'}\n        report['label_price_check'] = px.verify_labels(panel, days)")
    analysis=replace(analysis,'evaluate_fixed_strategies',evaluator)
    # Convert the copied weekly primary to the intended one-day strategy, preserving159 files.
    analysis=analysis.replace("dict(name='top5_1d', topn=5, period=5, band=RECIPE['hold_band'])",
        "dict(name='top5_1d', topn=5, period=1, band=RECIPE['hold_band'])")
    picks=ast.unparse(node(analysis,'write_daily_picks'))
    score="score = pred[index]"
    assert picks.count(score)==1
    picks=picks.replace(score,score+"\n        estimated_return=score/RECIPE['ensemble_sum_members']\n        fee_reference=2*max(.00025,5/(RECIPE['backtest_money']/topn))+.00002+.0005+.0006")
    picks=picks.replace("candidates = np.flatnonzero(np.isfinite(score))",
        "candidates = np.flatnonzero(np.isfinite(score)&(estimated_return>=RECIPE['cost_gate_multiplier']*fee_reference))")
    pf=ast.parse(picks)
    fn=pf.body[0]
    loop=next(n for n in fn.body if isinstance(n,ast.For))
    for i,n in enumerate(loop.body):
        if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='text' and isinstance(n.value,ast.IfExp):
            loop.body[i]=ast.parse("text += f'模型估计下一开盘至再下一开盘的截尾收益；达到费用参考才安排新买，换股也考虑费用。下一可执行开盘：{buy if buy else \"下一交易日开盘\"}。已有持仓可继续保留，出售仍受可成交性限制。\\n\\n'").body[0]
    picks=ast.unparse(ast.fix_missing_locations(fn))
    picks=picks.replace("f'{float(score[c]):.4f}'","f'{float(estimated_return[c])*100:.3f}%'")
    picks=picks.replace("['排名', '代码', '名称', '打分']","['排名', '代码', '名称', '截尾收益估计']")
    picks=picks.replace("'当日没有有效打分。\\n\\n'","'当日没有达到费用参考的有效新买候选。\\n\\n'")
    analysis=replace(analysis,'write_daily_picks',picks)
    scripts=dict(zip(['model.py','analysis.py','run.py'],[model,analysis,run]))
    for name,source in scripts.items():compile(source,str(dest/name),'exec')
    if not write:return scripts
    dest.mkdir()
    for name,source in scripts.items():(dest/name).write_text(source)
    return {name:hashlib.sha256(source.encode()).hexdigest() for name,source in scripts.items()}


if __name__=='__main__':
    AUDIT.mkdir(exist_ok=True)
    assert not any((ROOT/'experiments'/f'V{v}').exists() for v in range(62,66)),'never overwrite existing experiments'
    freeze={f'V{62+i}':create(f'V{62+i}',seed) for i,seed in enumerate([3253,4253,5253,6253])}
    (AUDIT/'source_freeze.json').write_text(json.dumps(freeze,indent=2))
    (AUDIT/'protocol.json').write_text(json.dumps(dict(units=list(freeze),seeds=[3253,4253,5253,6253],
        features=TECHNICAL_FEATURES,weights='fresh closed-form fits, not reused checkpoints',
        data=str(ROOT/'history/diversification_20260930/trainingdata_frozen'),
        criterion='normalized raw clipped absolute1/5d MSE; sample stages and bagging retained',
        sample_caps=[256,384,512,640,768],stock_ridge=.05,market_ridge=.25,
        fixed_strategy=dict(topn=5,period=1,band=0,cost_gate_multiplier=1.,selection='new-entry utility forecast minus fee proxy; carried-position utility raw forecast; nonnegative utility only'),
        gate_candidates=[0.,.5,1.,2.],no_test_selection=True,formal_test=['2025-07-01','2026-06-30'],
        four_seeds_mean_std_required=True,purge_sessions=21,
        stage_semantics='all stages use the same complete training dates; nested random stock samples, no random initialization'),indent=2))
    print('CREATED_ABSOLUTE_RIDGE_UNITS',list(freeze),'no weights trained yet')
