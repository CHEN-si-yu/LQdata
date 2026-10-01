"""Four self-contained fresh fusion units; three scripts only, no parent artifacts."""
from pathlib import Path
import ast
import hashlib
import json
import time

from flow_technical_fusion_components import MODEL, SCORE, MASK, WRITE_SCORES, READ_SCORES

ROOT = Path('/root/autodl-fs/model')
A = ROOT/'history/flow_technical_joint_20260930'
PARENT = ROOT/'experiments/V50'
ABSOLUTE_PARENT = ROOT/'experiments/V62'


def node(source, name):
    for item in ast.parse(source).body:
        if getattr(item, 'name', None) == name:
            return item
        if isinstance(item, ast.Assign) and any(isinstance(x, ast.Name) and x.id == name for x in item.targets):
            return item
    raise KeyError(name)


def fragment(source, name):
    item = node(source, name)
    start = min([item.lineno]+[x.lineno for x in getattr(item,'decorator_list',[])])-1
    return ''.join(source.splitlines(keepends=True)[start:item.end_lineno])


def replace(source, name, text):
    item = node(source, name)
    lines = source.splitlines(keepends=True)
    start = min([item.lineno]+[x.lineno for x in getattr(item,'decorator_list',[])])-1
    return ''.join(lines[:start])+text.strip('\n')+'\n'+''.join(lines[item.end_lineno:])


def prepare(unit, seed, meta):
    model = (PARENT/'model.py').read_text().replace('top5_1d','top5_5d')
    analysis = (PARENT/'analysis.py').read_text().replace('top5_1d','top5_5d')
    absolute_model = (ABSOLUTE_PARENT/'model.py').read_text()
    absolute_analysis = (ABSOLUTE_PARENT/'analysis.py').read_text()
    space = {}
    exec(fragment(model,'feature_columns'),space)
    flow = space['feature_columns'](meta)
    exec(fragment(absolute_model,'feature_columns'),space)
    technical = space['feature_columns'](meta)
    assert len(flow)==105 and len(technical)==44 and not set(flow).intersection(technical)
    features = [f for f in meta['columns']['features'] if f in set(flow+technical)]
    assert len(features)==149
    model = model.replace('import re\n','import re\nimport hashlib\n',1)
    model = model.replace(model.splitlines()[0],repr(f'{unit}: fresh flow/technical two-component rank fusion; seed{seed}.'),1)
    fields = {x.arg:ast.unparse(x.value) for x in node(model,'RECIPE').value.keywords}
    fields.update(name=repr(unit),seed=str(seed),family=repr('flow_technical_fresh_fusion'),
        architecture=repr('disjoint105-flow eight-session GRU and44-technical fixed-basis absolute1/5d ridge with61 causal market inputs; fresh internal percentile rank fusion plus separate cost gate'),
        parent_sources=repr(['V50','V62']),weights_origin=repr('scripts only copied; both components newly fitted in this unit; no parent checkpoint read'),
        market_gate='True',bag_recalibrate='False',feature_names=repr(features),
        flow_feature_names=repr(flow),technical_feature_names=repr(technical),
        flow_indices=repr([features.index(x) for x in flow]),technical_indices=repr([features.index(x) for x in technical]),
        output_horizons='[3,5,10,1]',score_blend=repr('flow_technical_percentile'),flow_fusion_weight='.5',
        picks_topn='5',picks_holding='5',hold_band='40',
        absolute_sample_caps='[256,384,512,640,768]',absolute_stock_ridge='.05',absolute_market_ridge='.25',
        absolute_intercept_ridge='1e-8',absolute_target_clip='{1:.12,5:.30}',absolute_loss_scale='{1:.03,5:.08}',
        absolute_label_weights='{1:.75,5:.25}',absolute_stage_bag=repr('all five, no stage selection'),
        cost_gate_multiplier='1.',ensemble_sum_members='4',
        valid_metric=repr('weighted raw-label RankIC of delivered internal50/50 fusion; absolute branch fitted training-only and held constant through neural epoch selection'),
        training_target=repr('flow3/5/10d daily standardized returns via WPCC; technical raw1/5d symmetrically clipped returns via date-equal normal equations'),
        final_prediction_backend=repr('CPU canonical rank outputs; CUDA train/validation raw heads numerically checked'),
        strategy_candidates=repr(dict(topn=[5,10,20],period=[1,5],band=[40],flow_weight=[0.,.25,.5,.75,1.],cost_gate_multiplier=[0.,1.],phase='all phases; primary Top5 weekly phase0')),
        research_protocol=repr('new149-feature two-component unit; four seeds3253/4253/5253/6253,4quarters x4folds,purge21. Fifty/fifty andTop5 weeklyband40 gate1 frozen from clean validation/framework. Both components fitted anew; no test-based weight/seed/phase selection; no automatic releases promotion'))
    model = replace(model,'RECIPE','RECIPE=dict(\n'+''.join(f'    {k}={v},\n' for k,v in fields.items())+')')
    model = replace(model,'feature_columns',"def feature_columns(meta):\n    names=RECIPE['feature_names']\n    if not set(names).issubset(meta['columns']['features']):raise ValueError('frozen149-feature view changed')\n    return list(names)")
    model = replace(model,'head_of',"def head_of(label):\n    return RECIPE['output_horizons'].index(int(label.split('_ret_')[1].rstrip('d')))")
    model = replace(model,'PredictModel',MODEL)
    model = replace(model,'score_of',SCORE)
    panel = fragment(model,'Panel')
    marker = '        self.coverage = np.empty((T, C), np.float32) if load_x else None'
    assert panel.count(marker)==1
    panel = panel.replace(marker,marker+'\n        self.flow_coverage=np.empty((T,C),np.float32) if load_x else None\n        self.technical_coverage=np.empty((T,C),np.float32) if load_x else None')
    marker = '                self.coverage[rows] = np.isfinite(x).mean(axis=2)'
    assert panel.count(marker)==1
    panel = panel.replace(marker,marker+"\n                self.flow_coverage[rows]=np.isfinite(x[:,:,RECIPE['flow_indices']]).mean(axis=2)\n                self.technical_coverage[rows]=np.isfinite(x[:,:,RECIPE['technical_indices']]).mean(axis=2)")
    panel = panel[:panel.index('    def mask(')]+MASK.strip('\n')+'\n'
    model = replace(model,'Panel',panel)
    average = fragment(model,'average_state_dicts')
    marker = '        if not first.is_floating_point():'
    assert average.count(marker)==1
    average = average.replace(marker,"        if key=='absolute_coef':\n            if not all(torch.equal(first,s[key]) for s in states):raise ValueError('static absolute fit changed across epochs')\n            averaged[key]=first.clone()\n            continue\n"+marker)
    model = replace(model,'average_state_dicts',average)
    train = fragment(model,'train_fold')
    train = train.replace("quarter=None):","quarter=None, absolute_only=False):",1)
    train = train.replace("    info = training_info(recipe)","    info = training_info(recipe)\n    signature=recipe_signature()",1)
    train = train.replace("        old = json.loads(done.read_text())","        old = json.loads(done.read_text())\n        if old.get('recipe_signature')!=signature:raise ValueError('completed recipe fingerprint changed; never reuse stale folds')",1)
    marker = '    model = PredictModel(len(panel.features), panel.market_dim).to(dev)'
    assert train.count(marker)==1
    train = train.replace(marker,marker+"\n    if (out/'last.pt').exists():\n        prepared=torch.load(out/'last.pt',map_location='cpu',weights_only=False)\n        if prepared.get('recipe_signature')!=signature:raise ValueError('resume recipe fingerprint changed')\n        absolute_fit_info=prepared['absolute_fit']\n    else:\n        absolute_fit_info=fit_absolute_component(model,panel,split['train'],seed)\n    absolute_fit_summary={k:v for k,v in absolute_fit_info.items() if k!='stages'}\n    absolute_fit_summary['stages']=[{k:v for k,v in s.items() if k!='coef'} for s in absolute_fit_info['stages']]\n    print('ABSOLUTE_COMPONENT_READY',len(absolute_fit_info['stages']),'fresh training fits',flush=True)")
    train = train.replace('dict(model=model.state_dict(),', 'dict(recipe_signature=signature,absolute_fit=absolute_fit_info,model=model.state_dict(),')
    marker = '    # --- 逐轮训练 ---'
    assert train.count(marker)==1
    prefit = '''    if absolute_only:
        if not last.exists():
            save_torch(last,dict(recipe_signature=signature,absolute_fit=absolute_fit_info,
                model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                epoch=-1,best=-float('inf'),stale=0,history=[],bank=[],
                numpy_rng=np.random.get_state(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if dev.type=='cuda' else []))
            atomic_json(out/'history.json',[])
        print('ABSOLUTE_PREPARED_ONLY',quarter,fold,'neural updates',start,flush=True)
        return dict(quarter=quarter,fold=fold,absolute_prepared=True,neural_epochs=start,
            recipe_signature=signature,absolute_fit=absolute_fit_summary)

'''
    train = train.replace(marker,prefit+marker,1)
    train = train.replace("    pred = predict(model, panel, prices, split['score'], dev)","    model=model.to('cpu')\n    pred = predict(model,panel,prices,split['score'],torch.device('cpu'))",1)
    train = train.replace('result = dict(quarter=quarter, fold=fold,','result = dict(recipe_signature=signature,absolute_fit=absolute_fit_summary,quarter=quarter, fold=fold,',1)
    model = replace(model,'train_fold',train)
    main = fragment(model,'main')
    main = main.replace("    args = parser.parse_args()","    parser.add_argument('--prepare-absolute-only',action='store_true',help='仅准备本单元新拟合技术分支，不形成完成折')\n    args = parser.parse_args()",1)
    main = main.replace('train_fold(fold, args.device, panel, prices, q)','train_fold(fold,args.device,panel,prices,q,absolute_only=args.prepare_absolute_only)',1)
    model = replace(model,'main',main)

    # The new strategy ranks in percentiles and admits using a separate raw-return column.
    analysis = analysis.replace('blend_linear, hot_filter, predict_heads,','blend_linear, hot_filter, predict_heads, predict_absolute,',1)
    cash = fragment(absolute_analysis,'cash_backtest')
    cash = cash.replace('forecast_members=1, cost_gate_multiplier=None):',
        'forecast_members=4, cost_gate_multiplier=None, gate_scores=None):',1)
    cash = cash.replace('    cash = money',"    if n>1 and gate>0:\n        if gate_scores is None:\n            if not hasattr(panel,'absolute_forecast'):raise ValueError('composite gate forecast not supplied')\n            gate_scores=panel.absolute_forecast[days]\n        if gate_scores.shape!=pred.shape:raise ValueError('gate forecast axes differ')\n    gate_by_day={} if gate_scores is None else {int(d):p for d,p in zip(days,gate_scores)}\n    cash = money",1)
    old = '''            if n>1 and gate>0:
                utility-=score_threshold
                for held in holdings:utility[held]=p[held]
                candidates=candidates[utility[candidates]>=0.]'''
    new = '''            if n>1 and gate>0:
                admission=gate_by_day[signal].copy()-score_threshold
                for held in holdings:admission[held]=gate_by_day[signal][held]
                candidates=candidates[np.isfinite(admission[candidates]) & (admission[candidates]>=0.)]'''
    assert cash.count(old)==1
    cash = cash.replace(old,new,1)
    analysis = replace(analysis,'cash_backtest',cash)
    strategies = "STRATEGIES=[dict(name='top5_5d',topn=5,period=5,band=RECIPE['hold_band']),dict(name='top1_1d',topn=1,period=1,band=0)]"
    analysis = replace(analysis,'STRATEGIES',strategies)
    analysis = replace(analysis,'write_scores',WRITE_SCORES)
    analysis = replace(analysis,'read_saved_scores',READ_SCORES)
    meta_source = fragment(analysis,'save_score_metadata')
    meta_source = meta_source.replace("['trade_date','stock_code','value','rank']","['trade_date','stock_code','value','rank','absolute_forecast_1d']")
    meta_source = meta_source.replace("    atomic_json(folder/'score_meta.json',metadata)","    metadata['absolute_forecast_1d']='raw sum of four fresh technical components, return units; rank score is separate'\n    atomic_json(folder/'score_meta.json',metadata)")
    analysis = replace(analysis,'save_score_metadata',meta_source)
    main = fragment(analysis,'main')
    main = main.replace("load_x=args.audit","load_x=not args.from_scores",1)
    main = main.replace('    variants = {}','    variants = {}\n    absolute_variants = {}',1)
    main = main.replace('        chunks = []','        chunks = []\n        absolute_chunks=[]',1)
    main = main.replace('            completions.append(done)',"            completions.append(done)\n            checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)\n            net=PredictModel(len(panel.features),panel.market_dim)\n            net.load_state_dict(checkpoint['model']);net.ridge=checkpoint.get('ridge');net.eval()\n            absolute_days=np.searchsorted(panel.days,done['score_dates'] if quarter==qs[-1] else done['test_dates'])\n            absolute_chunks.append(predict_absolute(net,panel,absolute_days,torch.device('cpu')))",1)
    main = main.replace("        variants[f'fold{fold}'] = np.concatenate(chunks)","        variants[f'fold{fold}'] = np.concatenate(chunks)\n        absolute_variants[f'fold{fold}']=np.concatenate(absolute_chunks)",1)
    main = main.replace("    pred=variants['ensemble']","    panel.absolute_forecast=np.full((len(panel.days),len(panel.codes)),np.nan,np.float32)\n    panel.absolute_forecast[days]=sum(absolute_variants.values()).astype(np.float32)\n    pred=variants['ensemble']",1)
    main = main.replace("                  ensemble='四折打分直接相加（不归一化）',","                  ensemble='四折打分直接相加（不归一化）；辅助1日预测也原值相加，门槛乘4',\n                  fusion=dict(flow_weight=.5,rank_scale='within submodel explicit family percentiles; raw fourfold sum afterward',absolute_gate='separate training-fitted1d return, never percentile as return'),",1)
    analysis = replace(analysis,'main',main)
    clean = fragment(analysis,'clean_block_summary')
    clean = clean.replace("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')", "device = torch.device('cpu')  # Same canonical ranking backend as saved forecasts.",1)
    clean = clean.replace('period=spec[\'period\'], band=spec[\'band\'])',
        "period=spec['period'],band=spec['band'],forecast_members=1,gate_scores=predict_absolute(net,panel_x,va,device))",1)
    assert 'forecast_members=1,gate_scores=' in clean
    analysis = replace(analysis,'clean_block_summary',clean)
    picks = fragment(analysis,'write_daily_picks')
    marker = '        candidates = np.flatnonzero(np.isfinite(score))'
    assert picks.count(marker)==1
    picks = picks.replace(marker,marker+"\n        cost_ref=2*max(.00025,5/(RECIPE['backtest_money']/5))+.00002+.0005+.0006\n        forecast=panel.absolute_forecast[int(days[index])]\n        candidates=candidates[np.isfinite(forecast[candidates]) & (forecast[candidates]>=4*RECIPE['cost_gate_multiplier']*cost_ref)]",1)
    picks = picks.replace('模型输出 {horizon} 日收益排序分数；现金策略', '模型融合资金流与技术面排名，并以技术面1日收益预测检查新买入费用门槛；现金策略')
    analysis = replace(analysis,'write_daily_picks',picks)
    evaluate = fragment(analysis,'evaluate_fixed_strategies')
    evaluate = evaluate.replace("        clean_block = clean_block_summary(panel, px, STRATEGIES[0])","        clean_block = (report.get('clean_blocks') if from_saved else clean_block_summary(panel,px,STRATEGIES[0]))",1)
    evaluate = evaluate.replace("    atomic_json(info_root()/'final_audit.json', dict(","    report['clean_blocks']=clean_block\n    atomic_json(info_root()/'final_audit.json', dict(",1)
    evaluate = evaluate.replace("        bag_vs_argmax=bag_summary))","        bag_vs_argmax=bag_summary,fusion=report.get('fusion'),auxiliary_forecast='separate raw1d technical return; existing yearly parquet column; same fixed159 paths'))",1)
    analysis = replace(analysis,'evaluate_fixed_strategies',evaluate)
    for source,filename in [(model,'model.py'),(analysis,'analysis.py')]:
        compile(source,str(ROOT/'experiments'/unit/filename),'exec')
    return {'model.py':model,'analysis.py':analysis,'run.py':(PARENT/'run.py').read_text()}


def create():
    A.mkdir(exist_ok=True)
    frozen=ROOT/'history/diversification_20260930/trainingdata_frozen'
    meta=json.loads((frozen/'meta.json').read_text())
    assert not (A/'protocol.json').exists(),'never overwrite frozen cohort'
    prepared={u:prepare(u,s,meta) for u,s in [('V66',3253),('V67',4253),('V68',5253),('V69',6253)]}
    assert all(not (ROOT/'experiments'/u).exists() for u in prepared)
    recipe_source=prepared['V66']['model.py']
    protocol=dict(frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),units=list(prepared),seeds=[3253,4253,5253,6253],
        parents=['V50','V62'],weight_origin='scripts only; both branches freshly fitted; no old model artifacts copied',
        snapshot_root=str(frozen),feature_counts=dict(flow=105,technical=44,total=149,market=61),
        primary=dict(topn=5,period=5,band=40,phase=0,flow_weight=.5,cost_gate_multiplier=1.),
        selection='weighted delivered fusion RankIC on valid only; absolute all five training fits bagged equally. No fixed test seed/weight/phase selection.',
        forecast_contract='rank scores remain2D float32; within-submodel cross-family percentile transform explicit; four raw fold scores summed. Separate absolute1d forecasts rederived from own checkpoints and stored as one added column in existing yearly score files; fourfold raw sum gate scaled4; Top1 baseline unchanged.',
        artifact_contract='three scripts only; unchanged159 files/28dirs,4quarters x4folds/eight files. No history or other unit runtime imports.',
        source_sha256={u:{n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()} for u,sources in prepared.items()},
        planned_folds=64,former_reserved_identifiers='V66–V69 previously only reserved for rejected variance prototype, never created; these now identify fresh fusion models')
    (A/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    for unit,sources in prepared.items():
        d=ROOT/'experiments'/unit;d.mkdir()
        for name,source in sources.items():(d/name).write_text(source)
        (d/'run.py').chmod(0o755)
    docs=ROOT/'docs/iterations.md';s=docs.read_text()
    for unit,seed in zip(prepared,protocol['seeds']):
        assert f'### {unit}\n' not in s
        s+=f'\n### {unit}\n- 改动：仅复制V50/V62源代码构成独立双组件，seed={seed}；105资金流8日GRU与44技术/61当日市场非线性岭回归均在本单元重新拟合；技术分支1/5日原收益五嵌套样本日期等权全部袋化，神经分支3/5/10日WPCC及融合RankIC选轮。单折50/50家族百分位融合，四折原分数相加，独立技术1日预测控制Top5周频band40新入场费用门槛（原预测相加、门槛乘4），Top1基线保留。\n- 结果：仅三个独立脚本创建，尚未拟合/训练；159文件/28目录合同、辅助预测在既有年度parquet加列、own-checkpoint复算和from-scores独立读取等待预检。原Top5验证网格相比同门槛资金流端点四窗平均净收益改善+1.69/+13.82/+8.40/+2.34个百分点；这些是已有预测原型，不能当新模型收益。\n- 结论：在独立全新拟合中检验跨信息家族融合；不跨单元/历史运行依赖，不用固定测试挑种子/权重/相位，不复用父权重；此前被拒绝方差原型未创建这些编号，本版权重来源另立。\n'
    docs.write_text(s)
    print(json.dumps(protocol,ensure_ascii=False))


if __name__=='__main__':create()
