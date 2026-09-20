"""Conservative defaults for the 60 GiB shared host (no GPU required)."""
import os

def safe_jobs(value=0):
    n = int(value or 1)
    if not 1 <= n <= 2:
        raise ValueError("因子计算并行度必须为 1 或 2；默认 1，避免超出共享内存")
    return n

def configure():
    import resource
    target = 24 * 1024**3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    for bound in (soft, hard):
        if bound != resource.RLIM_INFINITY:
            target = min(target, bound)
    resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
        os.environ[key] = "1"
    return target
