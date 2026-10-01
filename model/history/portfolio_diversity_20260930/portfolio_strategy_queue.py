"""Create four independent strategy-only units from audited parents; retain exact weight lineage."""
from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path('/root/autodl-fs/model')
PRIMARY=ROOT/'history/diversification_20260930'
AUDIT=ROOT/'history/portfolio_diversity_20260930'
ORDER=[('V50','V58'),('V51','V59'),('V52','V60'),('V53','V61')]
ENV=dict(os.environ,MX_DATA=str(PRIMARY/'trainingdata_frozen'),MX_THREADS='2',
    PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
STOP=False
RUNNING=None


def stopped():
    return STOP or (PRIMARY/'STOP').exists() or (AUDIT/'STOP').exists()


def request_stop(sig,frame):
    global STOP
    STOP=True


signal.signal(signal.SIGTERM,request_stop)
signal.signal(signal.SIGINT,request_stop)


def read_json(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));os.replace(tmp,path)


def live_primary():
    state=read_json(PRIMARY/'batch_process.json')
    if not state:return None
    proc=Path('/proc')/str(state['pid'])
    try:
        return str(PRIMARY/'innovation_batch_driver.py').encode() in (proc/'cmdline').read_bytes().split(b'\0') and (proc/'stat').read_text().split()[2]!='Z'
    except (FileNotFoundError,IndexError):return False


def source_node(source,name):
    for n in ast.parse(source).body:
        if getattr(n,'name',None)==name:return n
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return n
    raise KeyError(name)


def replace_node(source,name,body):
    n=source_node(source,name);lines=source.splitlines(keepends=True)
    start=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1
    return ''.join(lines[:start])+body.rstrip()+'\n'+''.join(lines[n.end_lineno:])


def prepare_sources(parent,unit):
    src=ROOT/'experiments'/parent;dest=ROOT/'experiments'/unit
    model=(src/'model.py').read_text();recipe=source_node(model,'RECIPE')
    fields={kw.arg:ast.unparse(kw.value) for kw in recipe.value.keywords}
    fields.update(name=repr(unit),weights_origin=repr(f'strategy-only full copy of {parent}; all32 checkpoint files byte-identical; no new model training'),
        strategy_parent=repr(parent),portfolio_diversity_strength='0.25',
        portfolio_history='63',portfolio_min_observations='40',portfolio_pool='100',
        portfolio_policy=repr('carry-aware positive-correlation penalty on fresh entries; Top1 baseline and existing40-name hold buffer unchanged'))
    model=replace_node(model,'RECIPE','RECIPE=dict(\n'+''.join(f'    {k}={v},\n' for k,v in fields.items())+')')
    analysis=(src/'analysis.py').read_text()
    core=(AUDIT/'portfolio_diversity_core.py').read_text().split('import numpy as np\n',1)[1].strip()
    fn=source_node(analysis,'cash_backtest');cash=ast.unparse(fn)
    start="money = float(RECIPE['backtest_money'] if money is None else money)"
    assert cash.count(start)==1
    cash=cash.replace(start,
        "risk_info=(portfolio_risk_information(pred,portfolio_return_history(px),days,lookback=RECIPE['portfolio_history'],min_observations=RECIPE['portfolio_min_observations'],pool_size=RECIPE['portfolio_pool']) if n>1 else None)\n    "+start)
    line="fresh = [int(c) for c in ranked if entry_ok[d, c] and c not in keep]"
    assert cash.count(line)==1
    cash=cash.replace(line,line+"\n                if n>1:\n                    fresh=portfolio_diverse_fresh(fresh,keep,risk_info[signal],strength=RECIPE['portfolio_diversity_strength'],slots=max(0,n-len(keep)))")
    analysis=replace_node(analysis,'cash_backtest',core+'\n\n\n'+cash)
    fn=source_node(analysis,'evaluate_fixed_strategies')
    policy=ast.parse("report['portfolio_policy']={'parent_weights':RECIPE['strategy_parent'],'strength':RECIPE['portfolio_diversity_strength'],'lookback':63,'min_observations':40,'candidate_pool':100,'scope':'fresh buys conditional on actual carried holdings; model IC and raw ensemble forecasts remain parent forecasts; strategy-only weight reuse'}").body[0]
    offset=1 if isinstance(fn.body[0],ast.Expr) and isinstance(fn.body[0].value,ast.Constant) and isinstance(fn.body[0].value.value,str) else 0
    fn.body.insert(offset,policy)
    analysis=replace_node(analysis,'evaluate_fixed_strategies',ast.unparse(ast.fix_missing_locations(fn)))
    # Daily picks are potential fresh-entry candidates; actual carried positions still follow the buffer.
    picks=ast.unparse(source_node(analysis,'write_daily_picks'))
    loop="for index in range(len(days) - 1, first - 1, -1):"
    assert picks.count(loop)==1
    picks=picks.replace(loop,"candidate_px=Prices(panel)\n    candidate_risk=portfolio_risk_information(pred[first:],portfolio_return_history(candidate_px),days[first:],lookback=63,min_observations=40,pool_size=100)\n    "+loop)
    rank="picks = candidates[np.argsort(-score[candidates], kind='stable')][:topn]"
    assert picks.count(rank)==1
    picks=picks.replace(rank,"ranked=candidates[np.argsort(-score[candidates],kind='stable')]\n        picks=portfolio_diverse_fresh(ranked.tolist(),set(),candidate_risk[int(days[index])],strength=RECIPE['portfolio_diversity_strength'],slots=topn)[:topn]")
    picks=picks.replace("text += f'## {signal_day} 收盘\\n\\n'","text += f'## {signal_day} 收盘\\n\\n'\n        text += '组合候选考虑股票同涨同跌的程度；已有持仓按缓冲区规则继续保留。\\n\\n'")
    analysis=replace_node(analysis,'write_daily_picks',picks)
    return model,analysis


def build(parent,unit):
    src=ROOT/'experiments'/parent;dest=ROOT/'experiments'/unit
    assert not dest.exists(),f'never overwrite an existing unit: {dest}'
    record=read_json(src/'model_info/final_audit.json')
    assert record and record['model_train_read'] and record['folds_checked']==16
    for name in ['model','analysis']:sys.modules.pop(name,None)
    sys.path.insert(0,str(src));import model as m
    m.validate_layout(allow_missing=False)
    expected=m.fixed_files()
    assert len(expected)==159
    sys.path.pop(0);sys.modules.pop('model',None)
    weights={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in src.glob('model_train/*/fold*/*.pt')}
    assert len(weights)==32
    model,analysis=prepare_sources(parent,unit)
    preflight=read_json(AUDIT/'strategy_unit_preflight.json')
    assert preflight and preflight['ok']
    frozen=preflight['prepared_sources'][unit]
    assert hashlib.sha256(model.encode()).hexdigest()==frozen['model_sha256'],'parent model source changed after strategy preflight'
    assert hashlib.sha256(analysis.encode()).hexdigest()==frozen['analysis_sha256'],'parent analysis source changed after strategy preflight'
    for name,source in [('model.py',model),('analysis.py',analysis),('run.py',(src/'run.py').read_text())]:
        compile(source,str(dest/name),'exec')
    shutil.copytree(src,dest,ignore=shutil.ignore_patterns('__pycache__'))
    (dest/'model.py').write_text(model);(dest/'analysis.py').write_text(analysis)
    assert all(hashlib.sha256((dest/path).read_bytes()).hexdigest()==digest for path,digest in weights.items())
    # Copied plots and report files are provisional until this strategy is independently recomputed.
    write_json(dest/'model_info/final_audit.json',dict(run_name=unit,model_train_read=False,
        weights_reloaded=False,folds_checked=0,pending_strategy_analysis=True,parent_weights=parent))
    receipt=dict(parent=parent,unit=unit,created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        weight_origin='reused, not newly trained',checkpoint_sha256=weights,
        source_sha256={name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in ['model.py','analysis.py','run.py']},
        primary_strength_frozen=.25)
    write_json(AUDIT/f'{unit}_lineage.json',receipt)
    docs=ROOT/'docs/iterations.md';text=docs.read_text()
    assert f'### {unit}\n' not in text
    text+=f'\n\n### {unit}\n- 改动：完整复制已审计{parent}单元，32个checkpoint逐字节复用；仅在新买入排序中加入过去63日、收缩正相关的持仓分散惩罚0.25，考虑保留持仓；无新模型训练，三个独立脚本和159文件框架保留。\n- 结果：父权重复制已核验；本策略正在独立重算242日测试及逐日CPU重载审计，复制的旧报表不作为本策略结果。\n- 结论：验证模型同涨同跌与持仓集中是不同的问题；本单元是策略实验，不把复用权重视为新模型，也不据测试选择风险惩罚或晋级。\n'
    docs.write_text(text)
    return dest


if __name__=='__main__':
    completed=[]
    while True:
        if stopped():raise SystemExit('stopped before strategy publication')
        state=read_json(PRIMARY/'batch_status.json')
        grid=list((AUDIT/'clean_cash_grid').glob('V*_202*Q*.json'))
        ready=state and state.get('phase')=='complete' and live_primary() is False and len(grid)==16
        write_json(AUDIT/'strategy_queue_status.json',dict(pid=os.getpid(),checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            phase='waiting_for_parent_audit' if not ready else 'ready',primary_live=live_primary(),
            clean_grid_blocks=len(grid),planned_units=[u for _,u in ORDER],completed=completed))
        if ready:break
        time.sleep(20)
    for parent,unit in ORDER:
        if stopped():raise SystemExit('stopped before next strategy unit')
        dest=build(parent,unit)
        handle=(dest/'model_logs/analysis_0.log').open('a')
        RUNNING=subprocess.Popen([sys.executable,'-B','-u',str(dest/'analysis.py'),'--audit'],cwd=dest,
            env=ENV,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(AUDIT/'strategy_queue_status.json',dict(pid=os.getpid(),phase='analysis',
            running=dict(unit=unit,pid=RUNNING.pid),completed=completed))
        while RUNNING.poll() is None:
            if stopped():
                os.killpg(RUNNING.pid,signal.SIGTERM);handle.close()
                raise SystemExit('stopped; research artifacts retained')
            time.sleep(5)
        handle.close()
        assert RUNNING.returncode==0,f'{unit} analysis failed; do not continue publication'
        record=read_json(dest/'model_info/final_audit.json')
        assert record['model_train_read'] and record['folds_checked']==16
        lineage=read_json(AUDIT/f'{unit}_lineage.json')
        assert all(hashlib.sha256((dest/path).read_bytes()).hexdigest()==digest for path,digest in lineage['checkpoint_sha256'].items())
        completed.append(unit)
        write_json(AUDIT/'strategy_queue_status.json',dict(pid=os.getpid(),phase='audited',completed=completed))
        print('STRATEGY_UNIT_AUDITED',unit,'exact parent weights reused',flush=True)
    write_json(AUDIT/'strategy_queue_status.json',dict(pid=os.getpid(),phase='complete',completed=completed))
