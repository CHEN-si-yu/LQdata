"""Archive measured results and stop-scope evidence; never dispatch research jobs."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import time
import numpy as np
from scipy.stats import spearmanr

ROOT=Path('/root/autodl-fs/model');H=ROOT/'history';A=H/'research_closure_20260930'
BASE=H/'fixed_test_review_20260930';JOINT=H/'flow_technical_joint_20260930'
parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
QUARTERS=['2025Q3','2025Q4','2026Q1','2026Q2']
FAMILIES=dict(financial=[f'V{x}' for x in range(46,50)],flow=[f'V{x}' for x in range(50,54)],
    quantile_risk=[f'V{x}' for x in range(54,58)],portfolio_strategy=[f'V{x}' for x in range(58,62)],
    technical_absolute=[f'V{x}' for x in range(62,66)],fresh_fusion=[f'V{x}' for x in range(66,70)])
LABELS=dict(financial='财务质量加性模型',flow='资金流时序模型',quantile_risk='下尾风险分位数模型',
    portfolio_strategy='持仓相关性惩罚策略',technical_absolute='技术面绝对收益模型',fresh_fusion='资金流与技术面新拟合融合')


def read(p):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return None


def atomic(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));os.replace(t,p)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def primary(v):return next(a for a in v['accounts'] if a['strategy']!='top1_1d')


records={};sources={};pending=[];layouts={};fold_counts={};hash_mismatches=[]
for family,units in FAMILIES.items():
    for unit in units:
        d=ROOT/'experiments'/unit
        fold_counts[unit]=len(list(d.glob('model_train/*/fold*/complete.json')))
        record_path=(JOINT/'fixed_test_review/units' if family=='fresh_fusion' else BASE/'units')/(unit+'.json')
        v=read(record_path);official=read(d/'model_info/final_audit.json')
        if not v or not official or not official.get('model_train_read') or official.get('folds_checked')!=16:
            pending.append(dict(unit=unit,folds=fold_counts[unit],fixed_test_record=bool(v),official_audit=bool(official)));continue
        assert v['signal_days']==242
        for name,value in v['source_hashes'].items():
            if sha(d/name)!=value:hash_mismatches.append(unit+'/'+name)
        # Re-run the fixed layout predicate, without loading any feature panel or model.
        sys.modules.pop('model',None);sys.path.insert(0,str(d));import model as m
        layouts[unit]=m.validate_layout();assert layouts[unit]['files']==159 and layouts[unit]['directories']==28
        sys.path.pop(0);sys.modules.pop('model',None)
        v['official_audit']=official;v['family']=family
        records[unit]=v;sources[unit]=str(record_path)
assert not hash_mismatches,hash_mismatches
legacy=read(BASE/'legacy_fixed_test_summary.json');assert legacy['completed_units']==3
for v in legacy['records']:
    unit=v['unit']
    for key,digest in v['weight_sha256'].items():
        q,f=key.split('_');assert sha(ROOT/'experiments'/unit/'model_train'/q/f/'best.pt')==digest
    v['family']='legacy';v['weight_origin']='旧发布基准权重，只读重推当前快照'
    records[unit]=v;sources[unit]=str(BASE/'legacy_references'/(unit+'.json'))

scope=read(A/'scope_manifest.json');source_identities=read(H/'diversification_20260930/protocol.json')['snapshot_identities']
identity_mismatches=[]
for rel,expected in source_identities.items():
    p=ROOT/'trainingdata'/rel;st=p.stat()
    if dict(size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)!=expected:identity_mismatches.append(rel)
assert not identity_mismatches,identity_mismatches
frozen=H/'diversification_20260930/trainingdata_frozen'
for rel in source_identities:
    if rel.endswith('.parquet'):
        a=(ROOT/'trainingdata'/rel).stat();b=(frozen/rel).stat();assert (a.st_size,a.st_mtime_ns,a.st_ino)==(b.st_size,b.st_mtime_ns,b.st_ino)
lineage={}
for unit,parent in zip(FAMILIES['portfolio_strategy'],FAMILIES['flow']):
    if unit not in records:continue
    files=[f'model_train/{q}/fold{f}/{name}' for q in QUARTERS for f in range(1,5) for name in ['best.pt','last.pt']]
    assert all(sha(ROOT/'experiments'/unit/p)==sha(ROOT/'experiments'/parent/p) for p in files)
    lineage[unit]=dict(parent=parent,identical_checkpoint_files=32,fresh_training=False)

family_summary={}
for family,units in FAMILIES.items():
    part=[records[u] for u in units if u in records];accounts=[primary(v) for v in part]
    values=[x['net_return']*100 for x in accounts]
    family_summary[family]=dict(label=LABELS[family],completed=len(part),planned=4,
        net_mean_pct=float(np.mean(values)) if values else None,seed_sample_std_pp=float(np.std(values,ddof=1)) if len(values)>1 else None,
        max_drawdown_mean_pct=float(np.mean([x['max_drawdown']*100 for x in accounts])) if accounts else None,
        positive_seeds=sum(x>0 for x in values),quarter_seed_means={q:float(np.mean([next(t['return_value'] for t in x['quarters'] if t['quarter']==q)*100 for x in accounts])) for q in QUARTERS} if accounts else {},
        interpretation='四个独立账户的种子统计，不是已经执行的种子集成；同家族种子不算不同信息信号')

rank=[];strategy=[]
for unit,v in records.items():
    ac=primary(v);ic=next(x for x in v['IC'] if x['label']=='label_ret_5d')
    row=dict(unit=unit,family=v['family'],net_pct=ac['net_return']*100,max_drawdown_pct=ac['max_drawdown']*100,
        sharpe=ac['sharpe'],ic5=ic['Pearson_IC'],rankic5=ic['RankIC'],primary_strategy=ac['strategy'],fee_drag_pp=ac['fee_drag']*100,
        slippage_drag_pp=ac['slippage_drag']*100,quarters={t['quarter']:t['return_value']*100 for t in ac['quarters']},
        weight_origin=v.get('weight_origin','fresh fitted weights'),source=sources[unit],
        framework_metrics=(v['official_audit']['metrics_total'] if v['family']!='legacy' else None))
    (strategy if v['family']=='portfolio_strategy' else rank).append(row)
rank.sort(key=lambda x:-x['net_pct']);strategy.sort(key=lambda x:-x['net_pct'])

live=[]
for cohort in ['risk_quantile_20260930','flow_technical_joint_20260930','fixed_test_review_20260930']:
    for name in ['batch_status.json','status.json']:
        state=read(H/cohort/name)
        if not state or not state.get('pid'):continue
        proc=Path('/proc')/str(state['pid'])
        if proc.exists() and (proc/'stat').read_text().split()[2]!='Z':live.append(dict(cohort=cohort,record=name,pid=state['pid']))
for cohort,names in [('risk_quantile_20260930',['batch_process.json','quantile_validation_lab_process.json','quantile_strategy_lab_process.json']),
                     ('flow_technical_joint_20260930',['batch_process.json','validation0_process.json','validation1_process.json','fixed_test_process.json','prefetch_process.json','prefetch2_process.json']),
                     ('fixed_test_review_20260930',['process.json'])]:
    for name in names:
        receipt=read(H/cohort/name)
        if not receipt or not receipt.get('pid'):continue
        proc=Path('/proc')/str(receipt['pid'])
        if proc.exists() and (proc/'stat').read_text().split()[2]!='Z':live.append(dict(cohort=cohort,record=name,pid=receipt['pid']))
joint_grids=list((JOINT/'clean_cash').glob('V*.json')) if (JOINT/'clean_cash').exists() else []
quantile_valid=read(H/'risk_quantile_20260930/quantile_validation_status.json')
quantile_grid=read(H/'risk_quantile_20260930/quantile_grid_status.json')
complete=(not pending and len(joint_grids)==16 and quantile_valid['completed_blocks']==16 and quantile_grid['completed_blocks']==16 and not live)
if args.final:assert complete,dict(pending=pending,joint_grid_blocks=len(joint_grids),live=live)

correlations=read(BASE/'fixed_test_correlations.json')
if args.final:
    score_data={};cash_data={};dates=None;cashdates=None
    for row in rank:
        u=row['unit'];folder=BASE/'legacy_references' if row['family']=='legacy' else (JOINT/'fixed_test_review/scores' if row['family']=='fresh_fusion' else BASE/'scores')
        z=np.load(folder/f'{u}_242.npz')
        if dates is None:dates=z['dates']
        assert np.array_equal(dates,z['dates']);score_data[u]=z['scores']
        c=np.load(folder/f'{u}_{row["primary_strategy"]}_cash.npz')
        if cashdates is None:cashdates=c['dates']
        assert np.array_equal(cashdates,c['dates']);cash_data[u]=c['equity']/np.r_[100000.,c['equity'][:-1]]-1
    pairs={}
    for x,y in itertools.combinations(score_data,2):
        daily=[]
        for p,q in zip(score_data[x],score_data[y]):
            ok=np.isfinite(p)&np.isfinite(q)
            if ok.sum()>50:daily.append(float(spearmanr(p[ok],q[ok]).statistic))
        rankcorr=float(np.mean(daily));cashcorr=float(np.corrcoef(cash_data[x],cash_data[y])[0,1])
        pairs[x+'__'+y]=dict(daily_rank_correlation_mean=rankcorr,net_daily_cash_return_correlation=cashcorr,signal_days=len(daily),below_point7_rank=rankcorr<.7,below_point7_cash=cashcorr<.7)
    correlations=dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),window=['2025-07-01','2026-06-30'],human_correlation_threshold=.7,pairs=pairs,
        scope='23 frozen model/native-primary accounts; descriptive fixed-test correlation, not a tuning/selection or future stability claim')
    atomic(A/'fixed_test_correlations.json',correlations)

fusion_validation=[]
for q in QUARTERS:
    blocks=[read(p) for p in joint_grids if read(p)['quarter']==q]
    if not blocks:continue
    summary={}
    for w in [0.,.5,1.]:
        phase_means=[];drawdowns=[]
        for b in blocks:
            cases=[x for x in b['rows'] if x['flow_weight']==w and x['topn']==5 and x['period']==5 and x['cost_gate_multiplier']==1.]
            assert len(cases)==5;phase_means.append(np.mean([x['net_return']*100 for x in cases]));drawdowns.extend(x['max_drawdown']*100 for x in cases)
        summary[str(w)]=dict(net_seed_phase_mean_pct=float(np.mean(phase_means)),drawdown_mean_pct=float(np.mean(drawdowns)))
    fusion_validation.append(dict(quarter=q,seeds=len(blocks),values=summary,
        mix_minus_flow_pp=summary['0.5']['net_seed_phase_mean_pct']-summary['1.0']['net_seed_phase_mean_pct'],
        mix_minus_technical_pp=summary['0.5']['net_seed_phase_mean_pct']-summary['0.0']['net_seed_phase_mean_pct']))

quantile=read(H/'risk_quantile_20260930/quantile_validation_results.json')
variance=read(H/'portfolio_variance_20260930/branch_decision.json')
prototype=read(H/'flow_absolute_fusion_20260930/top5_framework_candidate_summary.json')
status='completed_and_stopped' if args.final else 'finishing_frozen_tasks'
now=time.strftime('%Y-%m-%d %H:%M:%S')
value=dict(as_of=now,status=status,new_research_allowed=False,scope_manifest=scope,remaining=pending,
    fresh_training_folds=sum(c for u,c in fold_counts.items() if u not in FAMILIES['portfolio_strategy']),planned_fresh_training_folds=320,
    fold_counts=fold_counts,layouts=layouts,lineage=lineage,current_snapshot_identities_unchanged=True,frozen_parquet_hardlinks_unchanged=True,
    old48_reference_weights_unchanged=True,common_axes_proof=read(A/'common_axes_proof.json'),model_top10=rank[:10],all_comparable_models=rank,strategy_variants=strategy,
    family_summary=family_summary,correlations=correlations,fresh_fusion_validation=fusion_validation,
    quantile_validation=quantile['summary'],variance_decision=variance,prototype_top5=prototype,
    live_processes=live,fusion_validation_grid_blocks=len(joint_grids),fusion_validation_grid_accounts=sum(len(read(p)['rows']) for p in joint_grids),
    closure_acceleration=[read(JOINT/n) for n in ['prefetch_process.json','prefetch2_process.json']],source_records=sources,test_window=['2025-07-01','2026-06-30'],signal_days=242,
    interpretation='既定原生Top5策略真实现金净收益，统一快照日期本金成本；日频与周频策略各自冻结。旧历史快照排行不混入。种子统计不等于可执行集成。固定测试曾在历史迭代中被查看，属于同窗回看，未独立证明未来收益。',
    releases_promoted=False)
atomic(A/'result_records.json',value)

lines=['# 模型与策略研究收尾报告','',f'记录时间：{now}。状态：'+('既定任务完成，已停止新增研究。' if args.final else '新增研究范围已冻结，正在完成既定任务。'),'',
    '本次研究围绕旧发布模型高度相关的问题，重新检查因子和训练快照，分别研究财务质量、资金流时序、绝对收益、下尾风险及组合执行。已有证据显示资金流与技术面能提供相关性较低的信号，但目前不能把一次固定窗口的较高收益解释为未来稳定提升。',
    '', '## 研究范围与完成状态','',
    f'共规划20个新拟合配置、320折训练，以及4个复用资金流权重的策略版本。实际已完成新拟合{value["fresh_training_folds"]}/320折；完成正式审计和固定测试的配置见下表。四个同家族种子用于检查离散度，不能算作四个独立信息来源。', '',
    '最新用户指令限定完成已设计的分位数V54至V57和融合V66至V69，以及它们的验证、正式审计和归档。停止新增方向、模型族和参数范围；releases没有晋级。', '',
    '## 数据检查与快照对齐','',
    '因子侧注册550个对象，4950个年度分区均存在，但未达到全量更新状态。日线最新为2026年9月29日，5分钟线止于9月28日；9月29日有41个非标签因子全空。2026分区还含17个未来交易日，295个非标签因子在未来日期有有效数值。原始上游数据未被本次修复。', '',
    'trainingdata已通过统一生成流程对齐当前因子，截止2026年9月29日，包含2122个交易日、2115只股票、484个股票特征、5个标签及61个市场特征。54个年度Parquet分区已核对；未来行按截止日剔除，真实缺失保留。研究全程使用冻结快照，并核对源文件身份未变化。原训练快照保存在history/snapshot_alignment_20260930_113324/previous_trainingdata。', '',
    '## 固定测试前十排行','',
    f'范围为当前已完成的{len(rank)}个可比配置，最终完整范围为20个新拟合配置与3个旧发布基准。旧基准用原16折权重在当前快照只读重推。测试信号为2025年7月1日至2026年6月30日，共242日；10万元本金，T日收盘信号、T加1开盘成交，计入整手、涨跌停、停牌、容量、佣金、过户费、卖出税和每边3基点滑点。按各配置冻结的原生Top5主策略累计净收益排序，日频与周频同时展示；排序不用于挑种子或晋级。', '',
    '| 排名 | 配置 | 主策略 | 累计净收益 | 最大回撤 | 5日RankIC |','| ---: | --- | --- | ---: | ---: | ---: |']
for i,x in enumerate(rank[:10],1):lines.append(f'| {i} | {x["unit"]} | {x["primary_strategy"]} | {x["net_pct"]:+.2f}% | {x["max_drawdown_pct"]:.2f}% | {x["rankic5"]:.4f} |')
lines+=['','## 家族稳定性与季度表现','','下表每行是四个独立账户的均值。收益标准差描述种子变化，季度数值来自连续实际账户分段；各季度复合可还原单账户全年收益。','',
    '| 家族 | 已完成种子 | 净收益均值 | 种子标准差 | 最大回撤均值 | 2025Q3 | 2025Q4 | 2026Q1 | 2026Q2 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for f,s in family_summary.items():
    if not s['completed']:lines.append(f'| {s["label"]} | 0/4 | 待完成 | — | — | — | — | — | — |');continue
    std=f'{s["seed_sample_std_pp"]:.2f}个百分点' if s['seed_sample_std_pp'] is not None else '待多种子齐备'
    values=[f'{s["net_mean_pct"]:+.2f}%',std,f'{s["max_drawdown_mean_pct"]:.2f}%',*[f'{s["quarter_seed_means"][q]:+.2f}%' for q in QUARTERS]]
    lines.append('| '+s['label']+' | '+str(s['completed'])+'/4 | '+' | '.join(values)+' |')

if family_summary['fresh_fusion']['completed']==4:
    ff=family_summary['fresh_fusion'];flow=family_summary['flow'];tech=family_summary['technical_absolute']
    delta=ff['net_mean_pct']-flow['net_mean_pct']
    top1=float(np.mean([next(x['net_return'] for x in records[u]['accounts'] if x['strategy']=='top1_1d')*100 for u in FAMILIES['fresh_fusion']]))
    verdict=('当前冻结融合的平均净收益低于资金流家族，保留为收益与风险取舍的证据。' if delta<0 else '当前冻结融合的平均净收益高于资金流家族；种子和季度差异仍需按完整结果理解。')
    lines+=['',f'新拟合融合固定测试：四种子主策略净收益均值{ff["net_mean_pct"]:+.2f}%，标准差{ff["seed_sample_std_pp"]:.2f}个百分点，平均最大回撤{ff["max_drawdown_mean_pct"]:.2f}%。与资金流原生主策略的收益均值差{delta:+.2f}个百分点，与技术面的差{ff["net_mean_pct"]-tech["net_mean_pct"]:+.2f}个百分点；最后季度均值{ff["quarter_seed_means"]["2026Q2"]:+.2f}%。原生策略含日频和周频，这些差值为冻结配置的描述对照。Top1日频基准的四种子平均净收益{top1:+.2f}%。'+verdict+'不据该窗口挑种子、调参或晋级releases。']

lines+=['','持仓去相关版本V58至V61复用对应V50至V53全部32个权重文件，仅修改执行策略。这一组平均净收益29.70%，相比资金流31.98%少2.28个百分点；平均最大回撤从−23.66%改善至−20.04%，但种子收益标准差从9.36增加至13.88个百分点。它体现了收益与回撤的交换，不能称为一致收益提升或新信号。', '',
    '## 相关性与融合证据','',
    '相关性分别报告匹配日期的每日横截面排名相关均值，以及实际现金账户逐日收益相关。以V51与V62为例，二者为0.445和0.574，均低于用户提出的0.7参考线；同窗旧V31与V36分别约0.979和0.840。低相关性提供了融合依据，但仍需要两个端点的收益与IC，以及融合账户结果共同成立。', '',
    '旧三个发布模型直接等权融合的同窗Top5账户净收益为7.27%，低于三个单模型16.99%至24.99%。相关性高和简单平均不能自动带来收益提升。', '',
    '已有预测的Top5周频融合原型完成16个验证块、2880个真实现金账户。50比50融合、同一技术面入场门槛、全部周频相位平均，相比同门槛资金流端点四窗提高1.69、13.82、8.40和2.34个百分点；最后一窗仍落后技术面端点6.29个百分点。这里的四窗长427至473日、彼此重叠，且用于选轮，数值不是季度收益、不是独立固定测试。', '',
    '正式融合V66至V69把105个资金流特征与44个技术特征分支分别在本单元新拟合，使用61个当前市场特征。内部50比50百分位排名与技术面1日原收益入场门槛分开计算；四折原分数和辅助预测直接相加，费用门槛乘4。其主策略冻结为Top5周频、缓冲40、相位0；不依据固定测试改权重、种子或相位。融合含有两端原有信号，不能再当作第三个独立信息来源。', '']
if fusion_validation:
    lines+=['新拟合融合的验证结果如下。各格仍为四种子、五个相位的描述平均，正式单元只执行相位0。','','| 验证窗口 | 已完成种子 | 融合净收益均值 | 相比资金流端点 | 相比技术端点 |','| --- | ---: | ---: | ---: | ---: |']
    for s in fusion_validation:lines.append(f'| {s["quarter"]} | {s["seeds"]}/4 | {s["values"]["0.5"]["net_seed_phase_mean_pct"]:+.2f}% | {s["mix_minus_flow_pp"]:+.2f}个百分点 | {s["mix_minus_technical_pp"]:+.2f}个百分点 |')
else:lines+=['正式融合训练及测试尚未完成，不能引用原型收益代替新模型结果。']
lines+=['','## 分位数校准与失败尝试','',
    '分位数路线完成16个验证块及9216个冻结策略组合。四窗25%、50%、75%预测的实际覆盖率约为23.5%至24.5%、48.9%至49.4%、74.0%至75.8%，校准较接近目标。固定下尾惩罚0.5在最早窗口使四种子净收益标准差从40.99降至8.09个百分点，同时均值从23.47%降至12.19%；其余窗口收益和风险变化并不一致。覆盖率准确不能单独证明策略更赚钱。固定测试四种子的净收益均值为6.70%、标准差11.74个百分点，最后季度均值−14.63%，尚未形成稳定收益优势。', '',
    '财务质量路线相对旧模型排名差异较大，但固定测试四种子净收益均值−4.77%，最后季度均值−18.71%。差异化信号仍需足够收益质量。', '',
    '方差惩罚原型完成864个账户后被拒绝：固定惩罚3的Top5日频在16个种子窗口中0个收益改善，14个回撤改善；四窗平均净收益下降43.65、60.06、52.35、108.11个百分点。惩罚1也在四窗均值全部下降。没有据此创建正式方差模型，V66至V69实际对应后来已设计的独立融合。', '',
    '## 迭代经验与结论','',
    '1. 先分信息家族，再看模型相关性。财务、资金流与技术面有不同信息基础；更换种子或复制权重不能形成新的独立信号。',
    '2. 同时看净收益、回撤、种子离散和季度结果。单种子排行领先、IC较高或校准较准，都不足以解释为稳定收益提升。',
    '3. 融合排序时可以显式对齐排名尺度；交易入场门槛必须保留原收益单位。四折求和与门槛的尺度必须一致。',
    '4. 降风险可能明显牺牲收益。持仓去相关与方差惩罚的实际失败或交换结果应保留，不只记录最好参数。',
    '5. 只有训练全在验证之前的块用于探索策略；窗口重叠、选轮与固定测试回看均明确记录。测试窗不再用于调参数。',
    '6. 并行按服务器实际分配资源设置。容器为12核、90GiB、一张12GiB显卡；GPU折任务从2路提高到4路，收尾阶段分两次将四个既定后续折提前计算，并行峰值6路、每折2线程；没有改变模型配方或64折范围。观察到91%至99%显卡利用率。宿主机资源百分比不代表本容器剩余资源。', '',
    '本次能够确认数据对齐、若干低相关信号以及实际的收益和风险交换。能否稳定提高未来收益仍未被充分证实。本批结果保留为可复核研究证据；完成既定收尾后停止新的研究与探索。', '',
    '## 核验与文件位置','',
    '每个完成单元保留3个脚本、159个文件、28个目录；4季度乘4折，每折8个训练产物。正式分析实际重载16个权重，核对当日独立推理、标签价格关系和真实现金费用。融合预检还验证了先准备再恢复与直接训练一致、未来输入改动不影响历史输出、辅助预测复读不加载其他单元权重。GPU与CPU原始头最大差约0.0000057，最终排名按CPU保存与审计。', '',
    '项目文档：docs/research_closure_20260930.md。逐版三行记录：docs/iterations.md。结构化汇总与停止范围：history/research_closure_20260930/result_records.json和scope_manifest.json。各路线的源代码、启动记录、逐窗口网格、权重来源与日志保存在各自history目录，模型产物在experiments对应单元。', '',
    '边界：存续且从未ST的股票池有幸存者偏差，上游因子的完整时点可用性尚未证明，5分钟数据缺口未修复。报告数据均为已执行回测；未执行的内容明确标为待完成。']
report='\n'.join(lines)+'\n'
(ROOT/'docs/research_closure_20260930.md').write_text(report)
(A/'report.md').write_text(report)
notice='## 本次研究收尾状态\n\n'+('2026年9月30日本次全市场研究的既定任务已完成，已停止新增研究。' if args.final else '2026年9月30日已冻结新增研究范围，正在完成此前设计好的任务。')+'结果与边界见[模型与策略研究收尾报告](research_closure_20260930.md)。以下历史发现保留原记录，不作为本次重新验证的结论。\n\n'
p=ROOT/'docs/research_summary.md';old=p.read_text();old=re.sub(r'\A## 本次研究收尾状态\n.*?(?=# 历史研究要点)', '',old,flags=re.S);p.write_text(notice+old)
if args.final:
    iterations=ROOT/'docs/iterations.md';text=iterations.read_text()
    for unit,v in records.items():
        if v['family']=='legacy':continue
        ac=primary(v);quarters='、'.join(q+' '+f'{next(t["return_value"] for t in ac["quarters"] if t["quarter"]==q)*100:+.2f}%' for q in QUARTERS)
        pattern=r'(?ms)(^### '+unit+r'\n)(.*?)(?=^### |\Z)';match=re.search(pattern,text);assert match,unit
        block=match.group(2)
        mt=v['official_audit']['metrics_total']
        four=f'框架四指标IC={mt["IC"]:.5f}、ICIR={mt["ICIR"]:.5f}、top_return={mt["top_return"]:.5f}%/日、top_return_stability={mt["top_return_stability"]:.5f}（Top500无摩擦口径，区别于真实现金收益）'
        result=f'- 结果：4季度×4折齐备、159文件/28目录和16折CPU重载审计通过；{four}；统一快照242信号日，冻结主策略{ac["strategy"]}实际现金净收益{ac["net_return"]*100:+.2f}%，最大回撤{ac["max_drawdown"]*100:.2f}%，5日RankIC {next(x["RankIC"] for x in v["IC"] if x["label"]=="label_ret_5d"):.5f}；季度实际收益{quarters}。'
        conclusion='- 结论：本次既定研究已完成并停止新增探索；同家族四种子统计及失败分支见research_closure_20260930.md，不据测试挑种子/调参/晋级releases。'+('本版仅复用父权重修改策略，不属于新模型信号。' if v['family']=='portfolio_strategy' else '本版权重在本单元实际新拟合，收益为回看证据，未证明未来稳定提升。')
        block=re.sub(r'^- 结果：.*$',lambda _:result,block,flags=re.M);block=re.sub(r'^- 结论：.*$',lambda _:conclusion,block,flags=re.M)
        text=text[:match.start(2)]+block+text[match.end(2):]
    iterations.write_text(text)
    atomic(A/'completion_receipt.json',dict(completed_at=now,status=status,new_research_allowed=False,fresh_folds=320,
        model_units=20,strategy_only_units=4,old_readonly_references=3,formal_audits=24,live_processes=[],no_release_promotion=True,
        report_sha256=sha(ROOT/'docs/research_closure_20260930.md'),structured_results_sha256=sha(A/'result_records.json')))
print('CLOSURE_ARCHIVE',json.dumps(dict(as_of=now,status=status,remaining=pending,live=live,report=str(ROOT/'docs/research_closure_20260930.md'),models=len(rank),fresh_folds=value['fresh_training_folds']),ensure_ascii=False))
