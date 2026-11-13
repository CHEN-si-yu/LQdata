#!/usr/bin/env python3
"""只读共享面板；根据当前容器实测负载逐步增加并行折数。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import contextlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('MX_THREADS','2')
os.environ.setdefault('NVIDIA_NVML_BASED_CUDA_CHECK','1')
UNIT=Path(__file__).resolve().parent
QUARTERS=('2025Q3','2025Q4','2026Q1','2026Q2','2026Q3')
_PANEL=_PRICES=None


def number(path,default=0):
    try:return int(Path(path).read_text().strip())
    except (OSError,ValueError):return default


def keyed(path):
    try:return {k:int(v) for k,v in (line.split() for line in Path(path).read_text().splitlines())}
    except (OSError,ValueError):return {}


def limits():
    affinity=len(os.sched_getaffinity(0))
    try:
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        cores=min(affinity,float(quota)/float(period)) if quota!='max' else affinity
    except (OSError,ValueError):
        quota=number('/sys/fs/cgroup/cpu/cpu.cfs_quota_us',-1);period=number('/sys/fs/cgroup/cpu/cpu.cfs_period_us',100000)
        cores=min(affinity,quota/period) if quota>0 else affinity
    memory=number('/sys/fs/cgroup/memory.max')
    if not memory:
        memory=number('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    return dict(cpu_cores=cores,memory_bytes=min(memory,48_000_000_000))


def gpu_load():
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True)
        allowed=os.environ.get('CUDA_VISIBLE_DEVICES')
        allowed=None if allowed is None else {x.strip() for x in allowed.split(',') if x.strip()}
        rows=[]
        for line in result.stdout.splitlines():
            index,uuid,util,used,total=[x.strip() for x in line.split(',')]
            if allowed is None or index in allowed or uuid in allowed:
                rows.append(dict(id=index if allowed is None or index in allowed else uuid,
                    utilization=float(util),used_mib=float(used),total_mib=float(total)))
        return rows
    except (OSError,ValueError,subprocess.SubprocessError):return []


def headroom(info):
    # Total cgroup includes page cache; reserve remains below user's 62GB ceiling.
    return info['memory_bytes'] - number('/sys/fs/cgroup/memory.current') - 2*2**30


def initialize_worker(counter,gpus,queues=None):
    slot=0
    if gpus:
        with counter.get_lock():slot=counter.value;counter.value+=1
        os.environ['CUDA_VISIBLE_DEVICES']=gpus[slot%len(gpus)]
    if queues is not None:
        import torch.multiprocessing
        from multiprocessing.util import Finalize
        _PANEL.gpu_X,_PANEL.gpu_coverage=queues[slot].get(timeout=60)
        Finalize(None,release_gpu_panel,exitpriority=10)

def release_gpu_panel():
    if _PANEL is not None and hasattr(_PANEL,'gpu_X'):
        import torch
        torch.cuda.synchronize()
        delattr(_PANEL,'gpu_X')
        if hasattr(_PANEL,'gpu_coverage'):delattr(_PANEL,'gpu_coverage')
        torch.cuda.ipc_collect()

def gpu_cache_server(path,queues,ready,stop,gpu):
    """独立spawn进程持有一份CUDA只读派生面板，直到所有训练接收者退出。"""
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    try:
        import numpy as np
        import torch
        import torch.multiprocessing
        import warnings
        start=time.monotonic()
        source=np.load(path,mmap_mode='r',allow_pickle=False)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='The given NumPy array is not writable')
            tensor=torch.from_numpy(source).to('cuda')
        coverage=np.stack([np.load(Path(path).parent/(name+'.npy'),mmap_mode='r') for name in ['flow_coverage','fund_coverage','technical_coverage']],axis=2).astype(np.float32)
        coverage_tensor=torch.from_numpy(coverage).to('cuda')
        # Same readonly stock/observed coverage bytes; no labels enter GPUcache.
        points=np.linspace(0,len(source)-1,9,dtype=int)
        if not np.array_equal(tensor[points].cpu().numpy(),source[points]):
            raise RuntimeError('GPU缓存输入对拍失败')
        for queue in queues:queue.put((tensor,coverage_tensor))
        ready.put(dict(source=str(path),bytes=int(source.nbytes+coverage.nbytes),shape=list(source.shape),coverage_shape=list(coverage.shape),
            seconds=round(time.monotonic()-start,3),receivers=len(queues),verified_slices=len(points)))
        stop.wait()
        for queue in queues:
            queue.close();queue.join_thread()
        del tensor,coverage_tensor
        torch.cuda.synchronize();torch.cuda.ipc_collect()
    except BaseException as error:
        ready.put(dict(error=repr(error)))

def stop_gpu_cache(cache):
    if cache is None:return
    cache['stop'].set();cache['process'].join(timeout=30)
    if cache['process'].is_alive():
        cache['process'].terminate();cache['process'].join(timeout=10)
    for queue in cache['queues']:queue.close()
    cache['ready'].close()

def start_gpu_cache(path,workers,gpu):
    ctx=mp.get_context('spawn')
    queues=[ctx.Queue(maxsize=1) for _ in range(workers)]
    ready=ctx.Queue();stop=ctx.Event()
    process=ctx.Process(target=gpu_cache_server,args=(path,queues,ready,stop,gpu))
    cache=dict(process=process,queues=queues,ready=ready,stop=stop)
    process.start()
    try:
        meta=ready.get(timeout=45)
        if 'error' in meta:raise RuntimeError(meta['error'])
        cache['meta']=meta
        print('GPU_PANEL_CACHE_READY',json.dumps(meta,ensure_ascii=False),flush=True)
        return cache
    except Exception as error:
        print('GPU_PANEL_CACHE_SKIP',repr(error),flush=True)
        stop_gpu_cache(cache)
        return None


def work(quarter,fold,device):
    from model import train_fold,torch
    log=UNIT/'model_logs'/f'{quarter}_{fold}.log';log.parent.mkdir(exist_ok=True)
    with log.open('a',buffering=1) as handle,contextlib.redirect_stdout(handle),contextlib.redirect_stderr(handle):
        result=train_fold(fold,device=device,panel=_PANEL,prices=_PRICES,quarter=quarter)
        if device=='cuda':torch.cuda.empty_cache()
        return result



def schedule_tasks(tasks):
    import hashlib
    costs={};sources=[]
    for q,f in tasks:
        path=UNIT.parent/'V300'/'model_train'/q/f'fold{f}'/'complete.json'
        if path.exists():
            value=json.loads(path.read_text());seconds=value.get('seconds')
            if isinstance(seconds,(int,float)) and seconds>0:
                costs[(q,f)]=float(seconds)
                sources.append(dict(path=str(path.relative_to(UNIT.parents[1])),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=seconds))
    order={task:i for i,task in enumerate(tasks)}
    ordered=sorted(tasks,key=lambda t:(-costs.get(t,0.),order[t])) if costs else list(tasks)
    return ordered,dict(kind='longest prior duration first; runtime only, same task set/splits/seeds',reference='V300',sources=sources,original_order=tasks,actual_order=ordered)


def main():
    global _PANEL,_PRICES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=0,help='并行上限；0按当前资源估计，从1折逐步增加')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--quarters',nargs='+',choices=(*QUARTERS,'2026Q4'))
    parser.add_argument('--folds',nargs='+',type=int,choices=[1,2,3,4])
    parser.add_argument('--resources-only',action='store_true')
    args=parser.parse_args();info=limits();gpus=[] if args.device=='cpu' else gpu_load()
    if args.resources_only:
        print(json.dumps(dict(**info,gpus=gpus,headroom_bytes=headroom(info)),ensure_ascii=False,indent=2));return
    if args.device=='auto':args.device='cuda' if gpus else 'cpu'
    if args.device=='cuda' and not gpus:raise RuntimeError('指定 CUDA，但没有可探测的可见 GPU')
    from model import Panel,Prices,RECIPE,validate_layout,torch
    torch.set_num_threads(1)
    validate_layout(allow_missing=True,allow_temporary=True)
    tasks=[(q,f) for q in (args.quarters or QUARTERS) for f in (args.folds or [1,2,3,4])]
    tasks,scheduling=schedule_tasks(tasks)
    t0=time.monotonic();_PANEL=Panel(RECIPE['features']);_PRICES=Prices(_PANEL)
    cpu_ridge=RECIPE.get('optimizer')=='closed_form_ridge'
    compact_residual=bool(RECIPE.get('runtime_compact_mlp'))  # Compact parent network plus fixed linear score;1536MiB reserved.
    private_bytes=(4 if RECIPE.get('runtime_recurrent') else 2 if cpu_ridge or compact_residual else 4)*2**30
    poll_seconds=1 if cpu_ridge else 2
    adjust_seconds=2 if cpu_ridge else 6
    available=headroom(info)
    if available is not None and available<private_bytes:raise RuntimeError('共享面板加载后私有工作空间不足')
    threads=max(1,int(os.environ['MX_THREADS']))
    # 交替CPU/GPU阶段可适度超额预留工作进程；按实测负载扩容，不强行同时跑满。
    cap=max(1,int(math.ceil(2*info['cpu_cores']/threads)))
    # Idle fork workers share the read-only panel. Admit active tasks from live headroom below, not startup headroom.
    gpu_workspace=5120 if RECIPE.get('runtime_recurrent') else 512 if cpu_ridge else 1536 if compact_residual else 2048
    if gpus:cap=min(cap,max(1,int(sum(max(0,g['total_mib']-g['used_mib']-1024)//gpu_workspace for g in gpus))))
    cap=min(cap,len(tasks),args.jobs if args.jobs>0 else cap)
    cache=None
    if RECIPE.get('runtime_recurrent') and len(gpus)==1 and hasattr(_PANEL.X,'filename'):
        free=gpus[0]['total_mib']-gpus[0]['used_mib'];cache_mib=(_PANEL.X.nbytes+len(_PANEL.days)*len(_PANEL.codes)*3*4)/2**20
        if free>cache_mib+gpu_workspace+1536:
            cap=min(cap,max(1,int((free-cache_mib-1536)//gpu_workspace)))
            cache=start_gpu_cache(str(_PANEL.X.filename),cap,gpus[0]['id'])
    print(f'{len(tasks)}折，共享面板{_PANEL.X.nbytes/2**30:.2f}GiB，容器{info["cpu_cores"]:g}核，自动并行1→{cap}',flush=True)
    ctx=mp.get_context('fork');counter=ctx.Value('i',0)
    target=1;active={};pending=list(tasks);finished=0;samples=[];last_cpu=keyed('/sys/fs/cgroup/cpu.stat').get('usage_usec',0);last_sample=time.monotonic();last_adjust=last_sample
    try:
        with ProcessPoolExecutor(max_workers=cap,mp_context=ctx,initializer=initialize_worker,
                initargs=(counter,[g['id'] for g in gpus],cache['queues'] if cache else None)) as pool:
            while pending or active:
                while pending and len(active)<target:
                    q,f=pending.pop(0);active[pool.submit(work,q,f,args.device)]=(q,f)
                completed,_=wait(active,timeout=poll_seconds,return_when=FIRST_COMPLETED)
                for future in completed:
                    task=active.pop(future);result=future.result();finished+=1
                    print(f'[{time.strftime("%H:%M:%S")}] {task} 就绪{finished}/{len(tasks)}，累计{(time.monotonic()-t0)/60:.1f}分钟，选轮{result["best_epoch"]}',flush=True)
                now=time.monotonic()
                if now-last_sample<poll_seconds:continue
                current_cpu=keyed('/sys/fs/cgroup/cpu.stat').get('usage_usec',last_cpu)
                cpu=100*(current_cpu-last_cpu)/1e6/(now-last_sample)/info['cpu_cores'];last_cpu=current_cpu;last_sample=now
                gpu=gpu_load() if gpus else []
                utilization=sum(g['utilization'] for g in gpu)/len(gpu) if gpu else None
                room=headroom(info)
                row=dict(seconds=round(now-t0,1),active=len(active),target=target,cpu_percent=round(cpu,2),
                    gpu_percent=utilization,headroom_gib=None if room is None else round(room/2**30,2),queue=len(pending))
                samples.append(row)
                if now-last_adjust>=adjust_seconds and len(samples)>=2:
                    window=samples[-2:];cpu_mean=sum(x['cpu_percent'] for x in window)/len(window)
                    gpu_mean=sum(x['gpu_percent'] or 0 for x in window)/len(window)
                    pressured=(room is not None and room<3*2**30) or any(g['total_mib']-g['used_mib']<768 for g in gpu)
                    if pressured:target=max(1,target-1)
                    elif pending and target<cap and max(cpu_mean,gpu_mean)<=80:
                        desired=math.ceil(target*82/max(cpu_mean,gpu_mean,5))
                        extra=cap if room is None else max(0,int((room-3*2**30)//private_bytes))
                        target=min(cap,target+extra,max(target+1,min(2*target,desired)))
                    print('RESOURCE',json.dumps(dict(**row,next_target=target,mean_cpu_percent=cpu_mean,mean_gpu_percent=gpu_mean),ensure_ascii=False),flush=True)
                    last_adjust=now
    finally:
        stop_gpu_cache(cache)
    useful=samples
    summary=dict(container=info,gpus=gpus,parallel_limit=cap,target_percent=80,samples=samples,
        scheduling=scheduling,gpu_panel_cache=cache['meta'] if cache else None,task_scope=tasks,sampling_scope='whole container during this invocation, including concurrent jobs',
        mean_cpu_percent=sum(s['cpu_percent'] for s in useful)/len(useful) if useful else None,
        mean_gpu_percent=sum(s['gpu_percent'] or 0 for s in useful)/len(useful) if useful and gpus else None,
        definition='all sampled observations include ramp and tail; whole container includes concurrent jobs; input loading before first sample excluded; memory reserve is a constraint, not utilization target')
    log=UNIT/'model_logs/analysis_0.log';log.parent.mkdir(exist_ok=True)
    with log.open('a') as handle:handle.write('RESOURCE_SUMMARY_JSON '+json.dumps(summary,ensure_ascii=False)+'\n')
    print('训练完成',flush=True)


if __name__=='__main__':main()
