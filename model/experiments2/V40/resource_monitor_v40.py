#!/usr/bin/env python3
"""Launch the frozen V40 replay with an address-space ceiling and capture peak RSS."""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_v40.py"
LIMIT_BYTES = 180 * 1024 ** 3
env = os.environ.copy()
env.update({
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1"
})

def set_memory_limit():
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT_BYTES, LIMIT_BYTES))

start = datetime.datetime.now().astimezone().isoformat()
with (HERE / "run.log").open("w", encoding="utf-8") as stdout, \
     (HERE / "runner_stderr.log").open("w", encoding="utf-8") as stderr:
    process = subprocess.Popen(
        ["/usr/local/bin/python3", "-B", str(RUNNER)],
        cwd=str(HERE), env=env, stdout=stdout, stderr=stderr,
        preexec_fn=set_memory_limit
    )
    while process.poll() is None:
        time.sleep(0.5)
    return_code = process.wait()

usage = resource.getrusage(resource.RUSAGE_CHILDREN)
runner_hash = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
monitor_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
result = {
    "started_at": start,
    "finished_at": datetime.datetime.now().astimezone().isoformat(),
    "exit_code": return_code,
    "address_space_limit_gib": 180,
    "peak_child_rss_kib": int(usage.ru_maxrss),
    "peak_child_rss_gib": round(usage.ru_maxrss / 1024 / 1024, 4),
    "run_v40_sha256": runner_hash,
    "resource_monitor_sha256": monitor_hash,
    "stdout_file": str(HERE / "run.log"),
    "stderr_file": str(HERE / "runner_stderr.log")
}
(HERE / "resource.log").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
raise SystemExit(return_code)
