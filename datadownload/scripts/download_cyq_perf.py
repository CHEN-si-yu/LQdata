#!/usr/bin/env python
"""Full, resumable stock/cyq_perf backfill; incremental scheduling stays in everyday_tasks.

Run as the shared-data owner (claude). Uses everyday_tasks' shared client, lock,
atomic store and manifest format, without importing the frozen legacy engine.
15-calendar-day checkpoints keep each request below 100k rows. Completed years
are merged once, preserving local rows subsequently withdrawn by the vendor.
Use --refresh to re-fetch checkpoints when auditing revised historical values.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "everyday_tasks"))
FIELDS = ("trade_date", "stock_code", "his_low", "his_high", "cost_5pct",
          "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct", "weight_avg", "winner_rate")


def windows(start, end, days=15):
    while start <= end:
        hi = min(start + timedelta(days=days - 1), end)
        yield start.isoformat(), hi.isoformat()
        start = hi + timedelta(days=1)


def validate(df, lo, hi):
    if not set(FIELDS).issubset(df.columns):
        raise ValueError(f"Missing required columns: {set(FIELDS) - set(df.columns)}")
    if df.empty:
        return
    if df[["trade_date", "stock_code"]].isna().any().any():
        raise ValueError("Null business keys")
    if not df.trade_date.between(lo, hi).all():
        raise ValueError(f"Rows outside requested interval {lo}..{hi}")
    if df.duplicated(["trade_date", "stock_code"]).any():
        raise ValueError("Duplicate business keys; refusing to certify pagination")


def download(client, ds, lo, hi):
    import pandas as pd
    from data_incremental.core.client import QueryLimitError
    try:
        rows = client.fetch_all(ds.path, {"start_time": lo, "end_time": hi}, 10000)
    except QueryLimitError:
        a, b = date.fromisoformat(lo), date.fromisoformat(hi)
        if a == b:
            raise
        mid = a + (b - a) // 2
        return pd.concat([download(client, ds, lo, mid.isoformat()),
                          download(client, ds, (mid + timedelta(days=1)).isoformat(), hi)],
                         ignore_index=True)
    if not getattr(rows, "complete", True):
        raise RuntimeError(f"Incomplete pagination: {lo}..{hi}")
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=FIELDS)
    validate(df, lo, hi)
    return df


def run(args):
    import pandas as pd
    from data_incremental import paths, registry, config
    from data_incremental.core import lock, state, store
    from data_incremental.core.client import Client, extract_total

    ds = registry.get("stock_cyq_perf")
    lo, hi = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if lo < date(2018, 1, 1) or hi < lo or hi > date.today():
        raise ValueError("Require 2018-01-01 <= start <= end <= today")
    cfg = config.load()
    cfg["api"].update(concurrency=1, max_inflight=1)
    cfg["api"]["api_key_file"] = str(paths.API_KEY_FILE)
    cfg["api"]["global_rate_limit_file"] = str(paths.RATELIMIT_FILE)
    checkpoint = paths.STATE_ROOT / "cyq_perf_full"
    checkpoint.mkdir(parents=True, exist_ok=True)
    report = {"start": str(lo), "end": str(hi), "years": [], "complete": False,
              "winner_rate": "vendor raw values; no rescaling"}
    started = time.monotonic()
    with lock.acquire():
        client = Client(cfg)
        try:
            man = state.Manifest.load(ds.name)
            for year in range(lo.year, hi.year + 1):
                a, b = max(lo, date(year, 1, 1)), min(hi, date(year, 12, 31))
                pieces = []
                for start, end in windows(a, b):
                    f = checkpoint / f"{start}_{end}.parquet"
                    meta = f.with_suffix(".json")
                    use_cache = False
                    if not args.refresh and f.exists() and meta.exists():
                        info = json.loads(meta.read_text())
                        use_cache = info.get("sha256") == hashlib.sha256(f.read_bytes()).hexdigest()
                    if use_cache:
                        df = pd.read_parquet(f)
                        validate(df, start, end)
                    else:
                        df = download(client, ds, start, end)
                        store.write_atomic(f, df)
                        state._atomic_json(meta, {"rows": len(df), "start": start, "end": end,
                                                 "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
                    pieces.append(df)
                    print(f"CHUNK {start}..{end} rows={len(df)} cached={use_cache}", flush=True)
                fresh = pd.concat(pieces, ignore_index=True)
                del pieces
                validate(fresh, str(a), str(b))
                total = extract_total(client.call(ds.path, {"start_time": str(a), "end_time": str(b),
                                                           "page": 0, "page_size": 1}))
                if total is None or len(fresh) != total:
                    raise RuntimeError(f"{year}: downloaded={len(fresh)}, vendor total={total}; "
                                       "no coverage certified; rerun with --refresh if upstream changed")
                if total == 0:
                    raise RuntimeError(f"{year}: unexpected empty year; refusing to certify it")
                dest = paths.year_partition_path(ds.name, year)
                merged = store.upsert(dest, fresh, ds.keys, ds.sort_by)
                man.columns = list(merged.columns)
                man.data_start = ds.start
                man.mark_partition(year, len(merged), merged.trade_date.min(), merged.trade_date.max())
                man.add_coverage(str(a), str(b))
                man.finish_run(status="ok", rows=len(fresh), seconds=round(time.monotonic() - started, 1))
                man.save()  # Only after atomic parquet commit and total validation.
                item = {"year": year, "vendor_rows": total, "downloaded_rows": len(fresh),
                        "stored_rows": len(merged), "first": str(merged.trade_date.min()),
                        "last": str(merged.trade_date.max()), "unique_days": int(merged.trade_date.nunique()),
                        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
                report["years"].append(item)
                state._atomic_json(checkpoint / "report.json", report)
                print("YEAR " + json.dumps(item), flush=True)
                del fresh, merged, df
                gc.collect()
            report.update(complete=True, seconds=round(time.monotonic() - started, 1),
                          requests=client.stats["requests"],
                          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            state._atomic_json(checkpoint / "report.json", report)
            print("COMPLETE " + json.dumps(report), flush=True)
        finally:
            client.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--refresh", action="store_true", help="re-fetch all requested checkpoints")
    p.add_argument("--memory-gib", type=int, default=16, choices=range(4, 25))
    args = p.parse_args()
    os.environ.update(OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", ARROW_NUM_THREADS="2")
    cap = args.memory_gib * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
