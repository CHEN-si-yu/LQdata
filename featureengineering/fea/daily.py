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
    tracked=[cfg.root/"conf/config.yaml",cfg.root/"conf/universe_frozen.tsv",
        cfg.root.parent/"datadownload/conf/frequency.yaml",
        cfg.root.parent/"everyday_tasks/state/delay_history.json"]
    tracked+=list(cfg.root.glob("*.py"))+list((cfg.root/"fea").glob("*.py"))+list((cfg.root/"factors").glob("*.py"))
    result["__contracts__"]={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in tracked if p.exists()}
    return result

def content_identity(w):
    """忽略相同内容重写的文件时间；兼容尚未包含 MD5 的旧状态。"""
    return {"exists":w.get("exists"),"rows":w.get("rows"),"max_pit":w.get("max_pit"),
        "files":{k:(v[2] if len(v)>=3 else v[:2]) for k,v in w.get("files",{}).items()}}


def snapshot_hashes(cfg,specs,days):
    out={}
    for spec in specs:
        for name,day,digest,n in _hash_one((str(cfg.factor_root(spec)),spec.name,tuple(days))):
            out[name+"|"+day]={"md5":digest,"rows":n}
    return out

def save_hashes(path,records):
    path.write_text("factor\ttrade_date\tmd5\trows\n"+"".join(
        f"{key.replace('|',chr(9))}\t{v['md5']}\t{v['rows']}\n"
        for key,v in sorted(records.items())),encoding="utf-8")

def run(args,cfg,pick):
    # 同一个入口等待上游收尾，期间不写生产因子；最长6小时后明确失败。
    deadline=time.monotonic()+6*3600
    last_notice=0
    while True:
        busy=upstream_writer_pids()
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
        root=Path(cfg.root)/"state"/"runs"/stamp
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
            reasons[spec.name]=sorted(set(why))
            plan=eng.plan(spec,man,end,args.rebuild)
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
        # ★★ 2026-09-29：「历史不可变」—— 被**冻结掉**的"上游历史变化"（只记录、不回填）。
        #   必须在下面 `del eng` 之前取出来。见 `Engine._frozen_span`。
        #   这是"厂商是否原地改写了历史值"的唯一观测通道。
        frozen=list(eng.frozen_input_changes)
        # ★★ 2026-09-30：「历史不可变」的另两项观测（用户硬约定）
        #   · identity_refreshed：分区身份与台账不符但**内容核对一致** ⇒ 只刷新身份、未重建
        #   · history_recompute：日常运行里被排进计划的冻结下界之前的日期（正常应为空）
        refreshed=list(eng.identity_refreshed)
        hist_recomp=list(eng.history_recompute)
        plans={s.name:plan for s,_,plan in todo}
        before=snapshot_hashes(cfg,specs,days)
        if same_input:
            # 重试不能把上次无法解释的变化当成新的正确基线。
            before.update(recovery.get("before",{}))
        save_hashes(root/"before.tsv",before)
        _atomic_json(input_before,root/"inputs_before.json")
        _atomic_json({s.name:{str(y):[int(d) for d in ds] for y,ds in plan.items()} for s,_,plan in todo},root/"plan.json")
        _atomic_json({"factors":[s.name for s in specs],"inputs":input_before,"before":before,"reasons":reasons,"unverified":unverified,
            "plan":{n:{str(y):[int(d) for d in ds] for y,ds in plan.items()} for n,plan in plans.items()},
            "run_dir":str(root)},journal)
        print(f"日常更新：{len(specs)} 项（含依赖）· 上游截止 {int_to_str(end)} · 台账 {root}",flush=True)
        if args.rebuild:
            # 重建只清状态中的覆盖；数据逐分区原子替换，失败可从缺口续跑。
            for spec,man,_ in todo:
                man.reset(eng._recipe(spec));man.save()
        cal=eng.cal          # ★ 事后对账要用；eng 下面就被释放了
        del eng
        gc.collect()
        years=sorted({y for plan in plans.values() for y in plan})
        completed=[]
        for year in years:
            names=[s.name for s in specs if year in plans.get(s.name,{})]
            lo=max(f"{year}-01-01",args.start or cfg.default_start)
            hi=min(f"{year}-12-31",int_to_str(end))
            cmd=[sys.executable,str(Path(cfg.root)/"main.py"),"run",*names,
                 "--start",lo,"--end",hi,"--jobs",str(args.jobs or 1),"--slice-worker","--plan-file",str(root/"plan.json")]
            if getattr(args,"sandbox",None):cmd+=["--sandbox",str(args.sandbox)]
            print(f"  {year}: {len(names)} 项，独立进程运行；日志 {root/year.__str__()}",flush=True)
            log=root/f"{year}.log"
            t=time.time()
            with log.open("w") as stream:
                from .resources import run_bounded
                child=run_bounded(cmd,cwd=cfg.root,stdout=stream)
            completed.append({"year":year,**child,"seconds":round(time.time()-t,2)})
            # ★ 2026-09-25：同时打印两个量。`peak_cgroup_bytes` 是**真实计费内存**
            #   （护栏主判据用的就是它）；`peak_rss_bytes` 把共享页按进程重复计数、
            #   实测虚高 1.3~1.5×，只能当诊断 —— 以前只打这一个，导致误触发无法复盘。
            _cg=child.get("peak_cgroup_bytes",0);_lim=child.get("cgroup_limit_bytes")
            # ⚠️ 用 GiB（2**30）而不是十进制 GB：cgroup 上限是 90 GiB，
            #    按 1e9 打印会显示成「97 GB」，看起来跟 90 对不上，极易误导。
            print(f"  {year}: 退出码 {child['returncode']}，{time.time()-t:.1f} 秒，"
                  f"真实内存 {_cg/2**30:.2f} GiB"
                  + (f" / {_lim/2**30:.0f} GiB" if _lim else "")
                  + f"（批内RSS诊断 {child['peak_rss_bytes']/2**30:.2f} GiB）"
                  + (f" ⚠ 内存护栏触发：{child['stop_reason']}" if child.get("memory_stopped") else ""),
                  flush=True)
            _atomic_json({"completed":completed},root/"progress.json")
            if child['returncode']:
                print(log.read_text()[-6000:],flush=True)
                return child['returncode']
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
            "history_recompute":hist_recomp}
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
            return 3
        if gone or unexplained:
            print("发现缺失或无法解释的历史变化，请查看报告。",flush=True)
            return 1
        journal.unlink(missing_ok=True)
        print(f"完成：全部选定因子已对齐 {int_to_str(end)}。",flush=True)
        return 0
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()
