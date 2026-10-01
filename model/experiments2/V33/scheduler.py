#!/usr/bin/env python3
"""Four-worker CPU-only scheduler with RAM/load admission gates."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[key]="1"
import concurrent.futures as cf
import csv
import multiprocessing as mp
import os
import time
from pathlib import Path
from model import QUARTERS, train_quarter

ROOT=Path(__file__).resolve().parent
MEMORY_LIMIT_GIB=180.0
SOFT_USED_GIB=145.0
LOAD_SOFT=40.0
def snapshot():
    vals={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:","MemAvailable:")):
            key,value=line.split(":",1); vals[key]=int(value.split()[0])*1024
    total=vals.get("MemTotal",0); avail=vals.get("MemAvailable",0)
    try: load=float(Path("/proc/loadavg").read_text().split()[0])
    except Exception: load=0.0
    used=(total-avail)/(1024**3) if total else 0.0
    return {"ts":time.strftime("%Y-%m-%d %H:%M:%S"),"mem_total_gib":round(total/(1024**3),2),
      "mem_available_gib":round(avail/(1024**3),2),"mem_used_gib":round(used,2),
      "load1":load,"cpu_cores":os.cpu_count() or 0}
def worker_rss(pool):
    vals=[]
    try:
        processes=list(pool._processes.values())
    except Exception:
        processes=[]
    for proc in processes:
        try:
            for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    vals.append(int(line.split()[1])/1024**2)
                    break
        except (FileNotFoundError,ProcessLookupError,AttributeError):
            pass
    return round(sum(vals),3),round(max(vals,default=0),3)
def record(snap,active,peak):
    path=ROOT/"resource_log.csv"; fresh=not path.exists()
    row=snap|{"active_jobs":active,"worker_rss_sum_gib":peak[0],"worker_rss_max_gib":peak[1]}
    with path.open("a",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(row))
        if fresh: writer.writeheader()
        writer.writerow(row)
def under_pressure(s):
    return s["mem_used_gib"]>=SOFT_USED_GIB or s["load1"]>=LOAD_SOFT
def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--jobs",type=int,default=4)
    parser.add_argument("--quarters",nargs="*",choices=QUARTERS)
    args=parser.parse_args()
    if not 1<=args.jobs<=4: parser.error("--jobs must be from 1 through 4")
    quarters=list(args.quarters or QUARTERS)
    pending=[q for q in quarters if not (ROOT/"quarters"/q/"result.json").is_file()]
    if not pending:
        print("No pending V33 quarters.",flush=True); return
    initial=snapshot()
    print(f"V33: queued={len(pending)} pool={args.jobs}/4, RAM used={initial['mem_used_gib']:.2f}/{MEMORY_LIMIT_GIB:.0f} GiB, available={initial['mem_available_gib']:.2f} GiB, load1={initial['load1']:.2f}/{LOAD_SOFT:.0f}.",flush=True)
    if initial["mem_used_gib"]>=SOFT_USED_GIB:
        raise SystemExit("RAM admission gate: host already at soft limit; no worker started.")
    ctx=mp.get_context("spawn")
    iterator=iter(pending); active={}; exhausted=False; complete=0; failed=[]
    peak_host=initial["mem_used_gib"]; peak_worker_sum=0.0; peak_worker=0.0
    last_log=0.0
    with cf.ProcessPoolExecutor(max_workers=args.jobs,mp_context=ctx) as pool:
        while active or not exhausted:
            snap=snapshot()
            peak_host=max(peak_host,snap["mem_used_gib"])
            ws=worker_rss(pool)
            peak_worker_sum=max(peak_worker_sum,ws[0]); peak_worker=max(peak_worker,ws[1])
            if time.time()-last_log>=2:
                record(snap,len(active),ws); last_log=time.time()
                print(f"RESOURCE RAM={snap['mem_used_gib']:.1f}GiB avail={snap['mem_available_gib']:.1f}GiB load1={snap['load1']:.1f} workers={len(active)} RSSsum={ws[0]:.2f}GiB",flush=True)
            while not exhausted and len(active)<args.jobs and not under_pressure(snap):
                try: q=next(iterator)
                except StopIteration: exhausted=True; break
                future=pool.submit(train_quarter,q); active[future]=q
                print(f"START {q} active={len(active)}/{args.jobs}",flush=True)
                snap=snapshot()
                peak_host=max(peak_host,snap["mem_used_gib"])
            if not active:
                if exhausted: break
                print("Admission gate active; checking again in 3s.",flush=True)
                time.sleep(3)
                continue
            done,_=cf.wait(active,timeout=2,return_when=cf.FIRST_COMPLETED)
            for future in done:
                quarter=active.pop(future)
                try:
                    result=future.result(); complete+=1
                    print(f"DONE {quarter} fit={result['fit_seconds']:.2f}s total={result['seconds_total']:.2f}s rows={result['n_train_rows_after_label_filter']} purge={result['purged_trading_sessions']} RSS={result['rss_gib']['at_completion']:.3f}GiB",flush=True)
                except Exception as exc:
                    failed.append(quarter)
                    print(f"FAILED {quarter}: {exc!r}",flush=True)
            snap=snapshot()
            if snap["mem_used_gib"]>=MEMORY_LIMIT_GIB:
                # This branch is a hard-limit alarm. The admission gate and tiny four-bank
                # worker footprint leave substantial reserve before it can be reached.
                raise RuntimeError(f"server memory hard limit reached: {snap}")
    final=snapshot()
    summary={"quarters_requested":quarters,"quarter_count_complete":complete,"failed_quarters":failed,
      "worker_limit":args.jobs,"max_worker_limit":4,"per_worker_numerical_threads":1,
      "memory_limit_gib":MEMORY_LIMIT_GIB,"soft_used_limit_gib":SOFT_USED_GIB,
      "host_used_gib_start":initial["mem_used_gib"],"host_used_peak_sampled_gib":peak_host,
      "host_used_gib_end":final["mem_used_gib"],"worker_rss_sum_peak_sampled_gib":peak_worker_sum,
      "worker_rss_max_peak_sampled_gib":peak_worker,"load1_start":initial["load1"],"load1_end":final["load1"]}
    import json
    (ROOT/"scheduler_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(f"BATCH complete={complete} failed={failed} host_peak_sampled={peak_host:.2f}GiB worker_rss_sum_peak={peak_worker_sum:.3f}GiB",flush=True)
    if failed: raise SystemExit(1)
if __name__=="__main__": main()
