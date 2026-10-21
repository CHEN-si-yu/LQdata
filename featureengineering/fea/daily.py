"""日常单入口：互斥、按年新进程、依赖顺序、近期台账、输入稳定性。"""
import copy
import datetime
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from .dates import int_to_str, str_to_int
from .engine import Engine
from .manifest import Manifest, _atomic_json
from .spec import REGISTRY
from .dayhash import _hash_one
from .processes import upstream_writer_pids

def expand_dependencies(specs):
    out, visiting = {}, set()
    def visit(spec):
        if spec.name in visiting: raise ValueError(f"因子循环依赖：{spec.name}")
        if spec.name in out: return
        visiting.add(spec.name)
        for dep in spec.deps:
            if dep in REGISTRY: visit(REGISTRY[dep])
        visiting.remove(spec.name)
        out[spec.name] = spec
    for spec in specs: visit(spec)
    return list(out.values())

def file_snapshot(cfg, specs):
    deps={d for s in specs for d in s.deps if d not in REGISTRY}
    deps.update(("basic_calendar","stock_list","stock_st_info"))
    result={}
    for dep in sorted(deps):
        root=cfg.upstream/dep
        result[dep]={str(p.relative_to(root)):[p.stat().st_size,p.stat().st_mtime_ns]
            for p in sorted(root.rglob("*.parquet"))}
    project=cfg.contract_root
    tracked=[project/"conf/config.yaml",cfg.frozen_universe,
        cfg.input_contract_path("frequency","conf/frequency.yaml"),
        cfg.input_contract_path("production_registry","conf/production.json")]
    tracked+=list(project.glob("*.py"))+list((project/"fea").glob("*.py"))+list((project/"factors").glob("*.py"))
    tracked+=list((project/"research").rglob("*.py"))
    tracked+=list((project/"conf/research").rglob("*.yaml"))
    result["__contracts__"]={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in tracked if p and p.exists()}
    return result

def content_identity(w):
    """忽略相同内容重写的文件时间；兼容尚未包含 MD5 的旧状态。"""
    return {"exists":w.get("exists"),"rows":w.get("rows"),"max_pit":w.get("max_pit"),
        "files":{k:(v[2] if len(v)>=3 else v[:2]) for k,v in w.get("files",{}).items()}}


def snapshot_hashes(cfg,specs,days,*,planned_repairs=None,errors=None):
    out={}
    repair_keys={(r["factor"],int(r["year"])) for r in (planned_repairs or [])}
    for spec in specs:
        for year in sorted({int(day[:4]) for day in days}):
            selected=tuple(day for day in days if int(day[:4])==year)
            try:
                rows=_hash_one((str(cfg.factor_root(spec)),spec.name,selected))
            except Exception as exc:
                if (spec.name,year) not in repair_keys:
                    raise
                if errors is not None:
                    errors.append({"factor":spec.name,"year":year,"days":list(selected),
                                   "error":repr(exc),"reason":"planned_output_repair_unreadable_baseline"})
                continue
            for name,day,digest,n in rows:
                out[name+"|"+day]={"md5":digest,"rows":n}
    return out

def save_hashes(path,records):
    path.write_text("factor\ttrade_date\tmd5\trows\n"+"".join(
        f"{key.replace('|',chr(9))}\t{v['md5']}\t{v['rows']}\n"
        for key,v in sorted(records.items())),encoding="utf-8")


def output_repair_reasons(records,factor):
    return [f"output_repair:{r['year']}:{r['reason']}" for r in records if r["factor"]==factor]


def _resolve_streams(cfg,args,years):
    """并行年份流数：显式 > 配置 > auto（待算 ≥2 年且总车道 ≥6 时取 2，否则 1）。"""
    n=int(getattr(args,"streams",0) or 0)
    if n<=0:
        raw=(cfg.raw.get("resources") or {}).get("streams","auto")
        if str(raw).isdigit():n=int(raw)
    if n<=0:
        n=2 if len(years)>=2 and int(getattr(args,"jobs",0) or 0)>=6 else 1
    return max(1,min(n,max(1,len(years))))


def _committed_factor_years(progress):
    """Return only coordinator-committed (factor, year) pairs from current progress.

    Older progress records contain counts/coverage only and are intentionally not
    sufficient proof. A worker's partial partition is not a coordinator commit.
    """
    if not isinstance(progress, dict):
        return set()
    committed=set()
    for chunk in progress.get("completed",[]):
        if not isinstance(chunk,dict) or chunk.get("returncode") != 0:
            continue
        names=chunk.get("finalized_factor_names")
        by_factor=chunk.get("finalized_years_by_factor")
        if not isinstance(names,list) or not isinstance(by_factor,dict):
            continue
        try:
            if int(chunk.get("finalized",-1)) != len(names):
                continue
        except (TypeError,ValueError):
            continue
        if any(not isinstance(name,str) for name in names) or len(set(names)) != len(names):
            continue
        for name in names:
            years=by_factor.get(name)
            if not isinstance(years,(list,tuple)):
                continue
            for year in years:
                try:
                    committed.add((name,int(year)))
                except (TypeError,ValueError):
                    continue
    return committed


def _prune_finished(cfg,root,plans):
    """Drop a planned pair only after this run explicitly committed it.

    Manifest coverage is a secondary completeness check; historical coverage
    cannot establish that the current retry attempt completed the pair.
    """
    path=root/"plan.json"
    data=json.loads(path.read_text())
    try:
        progress=json.loads((root/"progress.json").read_text())
    except (OSError,ValueError,TypeError):
        progress={}
    committed=_committed_factor_years(progress)
    kept={};dropped=0
    manifests={}
    for name,years in data.items():
        stay={}
        for y,days in years.items():
            try:
                year=int(y)
            except (TypeError,ValueError):
                stay[y]=days
                continue
            if (name,year) not in committed:
                stay[y]=days
                continue
            if name not in manifests:
                manifests[name]=Manifest.load(cfg.state_dir,name)
            spans=[(int(a.replace("-","")),int(b.replace("-","")))
                   for a,b in manifests[name].coverage]
            ds=[int(d) for d in days]
            missing=[d for d in ds if not any(a<=d<=b for a,b in spans)]
            if missing:stay[y]=missing
            else:dropped+=1
        if stay:kept[name]=stay
    _atomic_json(kept,path)
    return dropped


def _save_resources(cfg,root,args,batch,t_run0,phase_marks,outcome,end):
    """写资源台账（`runs/<stamp>/resources.json` + `state/resource_ledger.jsonl`）并打印摘要。

    在任何退出路径上都尽力落盘 —— 「记录资源花费以便优化调整」的前提是失败批也有账。
    写入失败不阻断运行（台账是观测，不是闸门）。
    """
    try:
        from .executor import build_resources_record, write_ledger, print_summary
        mode=(getattr(args,"run_mode",None)
              or ("rebuild" if getattr(args,"rebuild",False)
                  else "refresh" if getattr(args,"refresh",False)
                  else "scoped" if (getattr(args,"factors",None) or getattr(args,"start",None))
                  else "daily"))
        rec=build_resources_record(cfg,root,mode=mode,cutoff=int(end),
                                   wall=time.time()-t_run0,batch=batch,
                                   phases={k:v for k,v in phase_marks.items() if v is not None},
                                   extra={"outcome":outcome})
        write_ledger(cfg,root,rec)
        print_summary(rec)
    except Exception:                                          # noqa: BLE001
        logging.getLogger("fea.daily").exception("资源台账写入失败（不阻断运行）")

def run(args,cfg,pick):
    # 同一个入口等待上游收尾，期间不写生产因子；最长6小时后明确失败。
    deadline=time.monotonic()+int(cfg.raw.get("daily",{}).get("wait_seconds",21600))
    last_notice=0
    while True:
        busy=upstream_writer_pids(cfg.upstream)
        if not busy:break
        now=time.monotonic()
        if now>=deadline:
            print("等待上游超过6小时；本轮尚未写入因子。",flush=True)
            return 3
        if now-last_notice>=60:
            print(f"等待上游更新完成（进程 {busy}）；完成后自动继续。",flush=True)
            last_notice=now
        time.sleep(5)
    specs=expand_dependencies(pick(args.factors,args.group))
    if not specs: return 1
    cfg.state_dir.mkdir(parents=True,exist_ok=True)
    lock=(cfg.state_dir/"daily.lock").open("a+")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print("已有因子写入任务占用共享锁，本轮未修改产物。",flush=True)
        return 2
    try:
        t_run0=time.time()          # ★ 资源台账：整轮墙钟起点（含计划/建层/发布）
        phase_marks={"plan":None,"layers":None,"tasks":None}
        lock.seek(0);lock.truncate()
        lock.write(json.dumps({"pid":os.getpid(),"host":__import__("socket").gethostname(),"started":time.time()}));lock.flush()
        journal=cfg.state_dir/"daily_inflight.json"
        recovery=None
        if journal.exists():
            recovery=json.loads(journal.read_text())
            previous_specs=[REGISTRY[n] for n in recovery["factors"] if n in REGISTRY and REGISTRY[n].enabled]
            specs=expand_dependencies(specs+previous_specs)
            print("发现未完成批次，恢复原计划及原始MD5基线。",flush=True)
        stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # 运行账本属于运行产物：放项目内 state/runs/（原来在 ../CodeX/featureengineering_runs）。
        root=cfg.state_dir/"runs"/stamp
        root.mkdir(parents=True,exist_ok=True)
        eng=Engine(cfg)
        # 必需数据集整体缺失是运行故障；不能把缺源产生的空值当作合法历史修订。
        required_sources=sorted({d for spec in specs for d in spec.deps if d not in REGISTRY})
        missing_sources=[d for d in required_sources if not eng.watermark(d).get("exists")]
        if missing_sources:
            raise RuntimeError("必需上游数据集缺失，未启动因子写入：" + ", ".join(missing_sources))
        end=str_to_int(args.end) if args.end else eng.baseline_last_day()
        if end>eng.baseline_last_day():
            raise ValueError("请求截止日期晚于上游日线水位")
        eng.override_start=str_to_int(args.start) if args.start else None
        eng.force_refresh=bool(args.refresh)
        input_before=file_snapshot(cfg,specs)
        same_input=bool(recovery and recovery.get("inputs")==file_snapshot(cfg,previous_specs))
        if recovery and not same_input:
            # 未完成批次可能已用较新水位覆盖状态，但旧年份尚未更新；不能仅回刷最近几天。
            for spec in previous_specs:
                m=Manifest.load(cfg.state_dir,spec.name);m.reset(m.recipe);m.save()
            print("未完成批次的输入或代码已变化，相关因子重新核验全历史。",flush=True)
        days=[int_to_str(int(d)) for d in eng.cal.days[eng.cal.days<=end][-7:]]
        changed_recipes=[]
        reasons={}
        unverified={}
        todo=[]
        for spec in specs:
            man=Manifest.load(cfg.state_dir,spec.name)
            why=[]
            unverified[spec.name]=[day for day in days if not any(a<=day<=b for a,b in man.coverage)]
            if same_input:
                unverified[spec.name]=sorted(set(unverified[spec.name])|set(recovery.get("unverified",{}).get(spec.name,[])))
            if man.recipe and man.recipe!=eng._recipe(spec):
                changed_recipes.append(spec.name)
                why.append("recipe_changed")
            for dep in spec.deps:
                if dep in REGISTRY:
                    why.extend("parent:"+dep+":"+x for x in reasons.get(dep,[]))
                else:
                    current=eng.watermark(dep)
                    previous=man.input_watermark.get(dep)
                    if previous is None:why.append("input_baseline_missing:"+dep)
                    elif content_identity(previous)!=content_identity(current):why.append("input_content_changed:"+dep)
            if same_input:why.extend(recovery.get("reasons",{}).get(spec.name,[]))
            # ★★ 2026-10-03（执行器重构 P0，净室测试抓到的冷启动 bug）：**recipe 落盘**。
            #   `plan()` 会在**内存里** `man.reset(recipe)`（L0 分支 / 冷启动 recipe 为空），
            #   所以"规划之后再比较是否变化"永远为假 —— 磁盘上的 recipe 于是永远停在旧值
            #   （冷启动时是空串）。后果：① 每轮都判 L0 全量重建；② **监控的 manifest_recipe
            #   检查全库报错、批次永不通过质量闸门**（净室实测 180/180 全中）。
            #   正确做法是在**调用 plan() 之前**取旧值，变了就把"已清空"的状态落盘。
            recipe_before=man.recipe
            plan=eng.plan(spec,man,end,args.rebuild)
            if man.recipe!=recipe_before:
                if recipe_before and not args.rebuild:
                    print(f"  ↻ {spec.name}: 逻辑/口径指纹已变，台账清零（本轮全量重建）",flush=True)
                man.save()
            elif any(h["factor"]==spec.name for h in getattr(eng,"coverage_healed",())):
                # ★★ 2026-10-06：coverage 已从**已验证分区**回填 ⇒ 台账落盘，
                #   下次运行/对账/status 直接采信，不再依赖本进程内存。
                man.save()
            why.extend(output_repair_reasons(eng.output_repairs,spec.name))
            reasons[spec.name]=sorted(set(why))
            if plan:todo.append((spec,man,plan))
        if same_input:
            existing={item[0].name:item for item in todo}
            for name,old_plan in recovery.get("plan",{}).items():
                if name not in REGISTRY:continue
                spec=REGISTRY[name]
                if not spec.enabled:continue
                if name not in existing:
                    item=(spec,Manifest.load(cfg.state_dir,name),{})
                    todo.append(item);existing[name]=item
                plan=existing[name][2]
                for year,values in old_plan.items():
                    ds=np.asarray(values,dtype=np.int32)
                    if len(ds) and (int(ds.max())>end or (args.start and int(ds.min())<str_to_int(args.start))):
                        raise ValueError("本次日期范围无法覆盖未完成批次；请使用无日期限制的main.py恢复。")
                    year=int(year)
                    plan[year]=np.union1d(plan.get(year,np.array([],dtype=np.int32)),ds)
        eng._propagate_parent_plans(todo,end)
        # ★★ 2026-10-03（执行器重构 P0）：**闸门必须在 driver 跑**。
        #   切片 worker 一律 `--no-state-writes`、不再跑 `st_gate`（它要写 ST 基线），
        #   所以整条并行路径上这里成了唯一调用点。少跑它不会"变慢"，而是**静默用错口径**：
        #   `stock_st_info` 不在任何因子 deps 里，池内 ST 前缀变了没人知道。
        #   放在写 journal 之前：闸门失败不该留下"有未完成批次"的假象。
        eng.st_gate()
        eng.delay_gate()
        # ★★ 2026-09-29：「历史不可变」—— 被**冻结掉**的"上游历史变化"（只记录、不回填）。
        #   必须在下面 `del eng` 之前取出来。见 `Engine._frozen_span`。
        #   这是"厂商是否原地改写了历史值"的唯一观测通道。
        frozen=list(eng.frozen_input_changes)
        # ★★ 2026-09-30：「历史不可变」的另两项观测（用户硬约定）
        #   · identity_refreshed：分区身份与台账不符但**内容核对一致** ⇒ 只刷新身份、未重建
        #   · history_recompute：日常运行里被排进计划的冻结下界之前的日期（正常应为空）
        refreshed=list(eng.identity_refreshed)
        hist_recomp=list(eng.history_recompute)
        output_repairs=list(eng.output_repairs)
        healed=list(getattr(eng,"coverage_healed",[]))   # ★ 2026-10-06：台账回填观测
        plans={s.name:plan for s,_,plan in todo}
        baseline_errors=[]
        before=snapshot_hashes(cfg,specs,days,planned_repairs=output_repairs,errors=baseline_errors)
        for failed_baseline in baseline_errors:
            name=failed_baseline["factor"]
            unverified[name]=sorted(set(unverified.get(name,[]))|set(failed_baseline["days"]))
        if same_input:
            # 重试不能把上次无法解释的变化当成新的正确基线。
            before.update(recovery.get("before",{}))
        save_hashes(root/"before.tsv",before)
        _atomic_json(input_before,root/"inputs_before.json")
        _atomic_json({s.name:{str(y):[int(d) for d in ds] for y,ds in plan.items()} for s,_,plan in todo},root/"plan.json")
        _atomic_json({"factors":[s.name for s in specs],"inputs":input_before,"before":before,"reasons":reasons,"unverified":unverified,
            "plan":{n:{str(y):[int(d) for d in ds] for y,ds in plan.items()} for n,plan in plans.items()},
            "run_dir":str(root)},journal)
        from .release import mark_dirty
        mark_dirty(cfg,"daily_or_maintenance_batch_started")
        phase_marks["plan"]=round(time.time()-t_run0,1)
        print(f"日常更新：{len(specs)} 项（含依赖）· 上游截止 {int_to_str(end)} · 台账 {root}",flush=True)
        cal=eng.cal          # ★ 事后对账要用；eng 下面就被释放了
        # ★★ 2026-10-03 多路年份流：派生层必须在**派发年块之前**由单一进程建好 ——
        #   每层的 `_manifest.json` 是整份 load→save，多个年块并发建同一层的不同年份
        #   会互相丢 `years` 条目（昂贵层年分区白重建；写面审计 2026-10-03）。
        #   层间仍按层并行（build_derived_layers 内部处理）。
        t_layers0=time.time()
        try:
            eng.build_derived_layers(todo, jobs=args.jobs)
        finally:
            phase_marks["layers"]=round(time.time()-t_layers0,1)
        del eng
        gc.collect()
        years=sorted({y for plan in plans.values() for y in plan})
        streams=_resolve_streams(cfg,args,years)
        from .executor import run_batch
        batch=None
        for attempt in range(3):
            batch=run_batch(cfg,root,streams=streams,jobs=args.jobs,cutoff=end)
            if not batch.get("memory_stopped") or attempt>=2:
                break
            # ★ 内存护栏触发 → 降档重跑（沿用 backfill 的重试语义）。**必须先把
            #   已完成年份从 plan.json 里剔掉**，否则护栏误触一次就把已经算完的
            #   年份原样重烧一遍（旧版逐块串行天然没这个问题，并行下要有）。
            pruned=_prune_finished(cfg,root,plans)
            streams=max(1,streams//2);args.jobs=max(1,args.jobs//2)
            print(f"⚠ 内存护栏触发（{batch.get('stop_reason')}）：降档为 "
                  f"{streams} 流 × {args.jobs} 车道重跑（已剔出完成年份 {pruned} 个）…",flush=True)
        completed=batch["chunks"]
        phase_marks["tasks"]=round(time.time()-t_run0- (phase_marks["layers"] or 0),1)
        _atomic_json({"completed":completed},root/"progress.json")
        _cg=batch.get("peak_cgroup_bytes",0);_lim=batch.get("cgroup_limit_bytes")
        # ⚠️ 用 GiB（2**30）而不是十进制 GB：cgroup 上限按 1e9 打印会显示成「97 GB」，
        #    看起来跟 90 对不上，极易误导。
        print(f"  年块执行：退出码 {batch['returncode']} · {len(completed)} 块 · "
              f"峰值计费内存 {_cg/2**30:.2f} GiB"
              + (f" / {_lim/2**30:.0f} GiB" if _lim else "")
              + f" · 进程组 CPU {batch.get('cpu_seconds',0):.0f}s"
              + (f" ⚠ 内存护栏触发：{batch['stop_reason']}" if batch.get("memory_stopped") else ""),
              flush=True)
        if batch['returncode']:
            clog=root/"coordinator.log"
            print(clog.read_text()[-6000:] if clog.exists() else "",flush=True)
            _save_resources(cfg,root,args,batch,t_run0,phase_marks,"failed",end)
            return int(batch['returncode'])
        # ★★ 2026-09-25 新增：**绕开计划的独立事后对账**（勘察结论 D3）。
        #
        #   为什么要它：本流程里其它检查全部以 `plan()` 的输出为输入，而 `plan()`
        #   正是可能出错的那一环。09-25 实测：`plan()` 的显式计划分支把 26 个因子的
        #   对齐年份整段滤掉 ⇒ 「没有任务」与「无事可做」在代码里长得一模一样 ⇒
        #   整轮打印「完成」、退出码 0，而磁盘上什么都没写。
        #
        #   这里不看计划，直接拿**台账 + 交易日历**对：本次参与的每个因子，其
        #   coverage 必须从各自的**分区下界**（声明对齐的取 align_start，否则取真实起点）
        #   起、覆盖到本次右端点 `end`。纯读 manifest，成本可忽略。
        holes=[];want_lo_s=want_hi_s=None
        # ★★ 2026-09-29：「历史不可变」守卫的**观测汇总**（见 fea/store.py 的 _content_identical）。
        #   manifest 在这个循环里已经 load 过了 ⇒ **零额外成本**。
        #   只统计**本次计划到**的年份，避免把上一次运行留下的 last_write 当成本次的。
        rewrites=[];skipped=0
        for s in specs:
            man=Manifest.load(cfg.state_dir,s.name)
            _planned={str(y) for y in plans.get(s.name,{})}
            for _y,_info in (man.partitions or {}).items():
                if _y not in _planned or not isinstance(_info,dict):
                    continue
                _w=_info.get("last_write")
                if not _w:
                    continue
                if _w.get("identical"):
                    skipped+=1
                else:
                    rewrites.append({**_w,"factor":s.name})
            want_lo=s.start_int(cfg) if s.align_fill is None else s.align_start_int(cfg)
            span=cal.between(want_lo,end)
            if span.size==0:continue
            want_lo_s,want_hi_s=int_to_str(int(span[0])),int_to_str(int(span[-1]))
            cov=[(int(a.replace("-","")),int(b.replace("-",""))) for a,b in man.coverage]
            if (not cov or min(a for a,_ in cov)>int(span[0])
                    or max(b for _,b in cov)<int(span[-1])):
                holes.append((s.name,
                              int_to_str(min((a for a,_ in cov),default=0)) if cov else "空",
                              int_to_str(max((b for _,b in cov),default=0)) if cov else "空"))
        if holes:
            print(f"✘ 事后对账不通过：{len(holes)}/{len(specs)} 个因子的台账覆盖不足"
                  f"（要求覆盖到 {want_lo_s} ~ {want_hi_s}）。"
                  f"这通常意味着某些 (因子,年) 被计划漏掉了。前 8 个：",flush=True)
            for n,a,b in holes[:8]:
                print(f"    {n}: 台账实际覆盖 {a} ~ {b}",flush=True)
            if len(holes)>8:print(f"    … 另 {len(holes)-8} 个",flush=True)
            _save_resources(cfg,root,args,batch,t_run0,phase_marks,"holes_failed",end)
            return 1
        print(f"✔ 事后对账通过：{len(specs)} 个因子的台账覆盖均达 {want_lo_s} ~ {want_hi_s}",flush=True)
        after=snapshot_hashes(cfg,specs,days)
        save_hashes(root/"after.tsv",after)
        input_after=file_snapshot(cfg,specs)
        _atomic_json(input_after,root/"inputs_after.json")
        drift=input_before!=input_after
        changed=[k for k in sorted(before.keys()&after.keys()) if before[k]!=after[k]]
        gone=sorted(before.keys()-after.keys())
        new=sorted(after.keys()-before.keys())
        previous_path=cfg.state_dir/"daily_inputs.json"
        source_revised=any(any("input_" in x for x in why) for why in reasons.values())
        change_reasons={}
        for key in changed:
            name,day=key.split("|")
            why=list(reasons.get(name,[]))
            if day in unverified.get(name,[]):why.append("previous_date_unverified:"+day)
            change_reasons[key]=why
        unexplained=[k for k in changed if not change_reasons[k]]
        report={"cutoff":int_to_str(end),"factors":len(specs),"dates":days,"completed":completed,
            "changed":changed,"gone":gone,"new":new,"recipe_changes":changed_recipes,
            "upstream_changed_since_previous":source_revised,"upstream_changed_during_run":drift,
            "unexplained_changes":unexplained,"change_reasons":change_reasons,
            # ★★ 2026-09-29：「历史不可变」两项观测（见 Engine._frozen_span / store._content_identical）
            #   · frozen_input_changes：上游的**历史**变了，按「只记录不回填」处理（差多少交易日）
            #   · partition_rewrites：本次重算后**内容真的变了**的 (因子,年) 分区；
            #     _skipped 是"重算后逐格相同、已跳过写盘"的数量。历史项应为 0。
            "frozen_input_changes":frozen,
            "partition_rewrites":rewrites,
            "partition_rewrites_skipped":skipped,
            # ★★ 2026-09-30：「历史不可变」补充观测（用户硬约定：日常增量不得重算历史）
            #   · identity_refreshed：身份不符但内容核对一致 ⇒ 只刷新台账、未重建
            #     （2026-09-30 就是它被误判成损坏，导致 550 因子 × 整年 2026 空转）
            #   · history_recompute：日常运行里排入冻结下界之前的日期；**正常必须为空**
            "identity_refreshed":refreshed,
            # ★ 2026-10-06：coverage 从已验证分区回填的 (因子,年) 明细（台账自愈观测）。
            "coverage_healed":healed,
            "history_recompute":hist_recomp,"output_repairs":output_repairs,
            "unreadable_before_baselines":baseline_errors}
        _atomic_json(report,root/"report.json")
        if not drift:_atomic_json(input_after,previous_path)
        print(f"台账比对：变化 {len(changed)}，恢复/新增 {len(new)}，消失 {len(gone)}；报告 {root/'report.json'}",flush=True)
        # ★★ 2026-09-29：「历史不可变」观测 —— 每天一眼能看出"上游是否改了历史"
        if frozen:
            deps=sorted({x['dep'] for x in frozen})
            print(f"⚠️ 上游有 {len(frozen)} 处**历史**变化（{','.join(deps[:5])}{' …' if len(deps)>5 else ''}）"
                  f"，按「只记录不回填」处理（明细见 report.json 的 frozen_input_changes）",flush=True)
        else:
            print("✔ 上游无历史变化（无被冻结的回填）",flush=True)
        if rewrites:
            print(f"⚠️ 本次有 {len(rewrites)} 个 (因子,年) 分区**内容真的变了**"
                  f"（跳过写盘 {skipped} 个）；明细见 partition_rewrites",flush=True)
        # ★★ 2026-09-30：「历史不可变」的两条每日播报（用户硬约定）
        if refreshed:
            print(f"ℹ️ {len(refreshed)} 个 (因子,年) 分区的台账身份已刷新"
                  f"（内容与台账一致、**未重建**）",flush=True)
        if healed:
            print(f"ℹ️ {len(healed)} 个 (因子,年) 分区的 coverage 已从**已验证分区**回填"
                  f"（未重算；明细见 report.json 的 coverage_healed）",flush=True)
        if hist_recomp:
            n=sum(x['days'] for x in hist_recomp); facs=len({x['factor'] for x in hist_recomp})
            print(f"ℹ️ 「历史不可变」剔出计划：{facs} 个因子 · 共 {n} 天（台账已覆盖且早于冻结下界）"
                  f" ⇒ 引擎**不会**自动重算；要看明细／要重算请走显式操作：report.json 的 history_recompute",flush=True)
        else:
            print("✔ 「历史不可变」：本次无任何已覆盖的历史日期被排入计划",flush=True)
        if drift:
            for spec in specs:
                m=Manifest.load(cfg.state_dir,spec.name);m.reset(m.recipe);m.save()
            print("本轮输入变化，已撤销本轮覆盖声明；下次同一入口会完整恢复，避免漏修历史。",flush=True)
            _save_resources(cfg,root,args,batch,t_run0,phase_marks,"input_drift",end)
            return 3
        if gone or unexplained:
            print("发现缺失或无法解释的历史变化，请查看报告。",flush=True)
            _save_resources(cfg,root,args,batch,t_run0,phase_marks,"unexplained",end)
            return 1
        # The complete production set alone can publish; selected/subset runs
        # remain useful maintenance tasks but never masquerade as a full batch.
        from .release import production_specs, publish
        complete=(not cfg.raw.get("_sandbox")
                  and not getattr(args,"slice_worker",False)
                  and (not args.start or str_to_int(args.start)<=str_to_int(cfg.default_start))
                  and end==int(cal.days[cal.days<=end][-1])
                  and not args.end
                  and {s.name for s in specs}=={s.name for s in production_specs(cfg)})
        monitor_report=None
        if complete and cfg.raw.get("daily",{}).get("monitor",True):
            from .monitor import run_monitor
            monitor_report=run_monitor(cfg,run_id=stamp,end=int_to_str(end))
            _atomic_json(monitor_report,root/"quality.json")
            if monitor_report.get("status") in {"failed","error"} or monitor_report.get("errors"):
                print("质量检查未通过，本批未发布；完整原因见 quality.json。",flush=True)
                _save_resources(cfg,root,args,batch,t_run0,phase_marks,"monitor_failed",end)
                return 1
        if complete and cfg.raw.get("daily",{}).get("catalog",True):
            from .catalog import generate
            generate(cfg)
        if complete and cfg.raw.get("daily",{}).get("publish",True):
            release=publish(cfg,specs,int_to_str(end),stamp,report,monitor_report)
            report["release_id"]=release["release_id"]
            _atomic_json(report,root/"report.json")
            print(f"发布完成：{release['release_id'][:16]} · {len(release['objects'])} 项。",flush=True)
        journal.unlink(missing_ok=True)
        _save_resources(cfg,root,args,batch,t_run0,phase_marks,
                        "published" if complete else "complete_scoped",end)
        print(f"完成：全部选定因子已对齐 {int_to_str(end)}。",flush=True)
        return 0
    except BaseException as exc:
        failure={"error":repr(exc),"at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "recovery":"Run the same main.py entry; unfinished journal is retained"}
        _atomic_json(failure,(root/"failed.json") if "root" in locals() else cfg.state_dir/"last_failure.json")
        # 失败批也要有资源账（尽力而为；台账是观测不是闸门）。
        if "root" in locals() and "batch" in locals() and batch is not None:
            try:
                _save_resources(cfg,root,args,batch,t_run0,phase_marks,"exception",end)
            except Exception:                                  # noqa: BLE001
                pass
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()
