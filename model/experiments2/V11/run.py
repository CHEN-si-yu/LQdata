#!/usr/bin/env python3
"""Bounded CPU-only quarterly scheduler; resumes from completed quarter artifacts."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[key]="1"
import argparse, concurrent.futures as cf, csv, multiprocessing as mp, time
from pathlib import Path
from model import QUARTERS, train_quarter

ROOT=Path(__file__).resolve().parent
def snapshot():
    vals={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:","MemAvailable:")):
            k,v=line.split(":",1); vals[k]=int(v.split()[0])*1024
    total=vals.get("MemTotal",0); avail=vals.get("MemAvailable",0)
    try: load=float(Path("/proc/loadavg").read_text().split()[0])
    except Exception: load=0.0
    used=(total-avail)/(1024**3) if total else 0.0
    return {"ts":time.strftime("%Y-%m-%d %H:%M:%S"),"mem_total_gib":round(total/(1024**3),2),
            "mem_available_gib":round(avail/(1024**3),2),"mem_used_by_availability_gib":round(used,2),
            "load1":load,"cpu_cores":os.cpu_count() or 0}
def pressure(s):
    return s["mem_used_by_availability_gib"]>=150 or s["load1"]>=40
def record(s):
    p=ROOT/"resource_log.csv"; fresh=not p.exists()
    with p.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(s))
        if fresh: w.writeheader()
        w.writerow(s)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--jobs",type=int,default=8)
    ap.add_argument("--limit",type=int,default=None,help="run only first N pending quarters in this batch")
    ap.add_argument("--quarters",nargs="*",choices=QUARTERS)
    a=ap.parse_args()
    if not 1<=a.jobs<=8: ap.error("--jobs must be 1..8")
    pending=[]
    for q in (a.quarters or list(QUARTERS)):
        if not (ROOT/"quarters"/q/"result.json").is_file(): pending.append(q)
    if a.limit is not None: pending=pending[:a.limit]
    if not pending:
        print("No pending quarters.",flush=True); return
    print(f"Submitting {len(pending)} quarter jobs; pool cap={a.jobs}; CPU only; gates: RAM<150 GiB and load1<40.",flush=True)
    ctx=mp.get_context("spawn")
    iterator=iter(pending); active={}; completed=0; last_record=0.0
    with cf.ProcessPoolExecutor(max_workers=a.jobs,mp_context=ctx) as pool:
        exhausted=False
        while active or not exhausted:
            snap=snapshot()
            if time.time()-last_record>=10:
                record(snap); last_record=time.time()
                print(f"RESOURCE used~{snap['mem_used_by_availability_gib']:.1f}GiB avail={snap['mem_available_gib']:.1f}GiB load1={snap['load1']:.1f}; active={len(active)}",flush=True)
            while not exhausted and len(active)<a.jobs and not pressure(snap):
                try: q=next(iterator)
                except StopIteration: exhausted=True; break
                fut=pool.submit(train_quarter,q); active[fut]=q
                print(f"START {q} pid-pool-slot; submitted={len(active)}/{a.jobs}",flush=True)
                snap=snapshot()
            if not active:
                if exhausted: break
                time.sleep(10); continue
            done,_=cf.wait(active,timeout=10,return_when=cf.FIRST_COMPLETED)
            for fut in done:
                q=active.pop(fut)
                try:
                    r=fut.result(); completed+=1
                    print(f"DONE {q}: fit={r['fit_seconds']}s total={r['seconds_total']}s train_rows={r['n_train_rows_after_label_filter']} RSS={r['rss_gib']['peak_process']}GiB",flush=True)
                except Exception as exc:
                    print(f"FAILED {q}: {exc!r}",flush=True)
            snap=snapshot()
            if pressure(snap): print(f"PAUSE submissions: resource gate tripped {snap}",flush=True)
    print(f"BATCH EXIT completed={completed} submitted={len(pending)}",flush=True)
if __name__=="__main__": main()
