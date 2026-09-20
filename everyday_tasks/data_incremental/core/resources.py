"""Apply the manual CLI's memory/thread policy before NumPy/Pandas import."""
import os


def configure(cfg):
    limits = cfg.get('resources', {})
    gib = int(limits.get('max_address_space_gib', 24))
    threads = int(limits.get('numeric_threads', 2))
    if not 4 <= gib <= 48 or not 1 <= threads <= 8:
        raise ValueError('resources: 地址空间须为 4–48 GiB，数值线程须为 1–8')
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    target = gib * 1024**3
    if soft != resource.RLIM_INFINITY:
        target = min(target, soft)
    if hard != resource.RLIM_INFINITY:
        target = min(target, hard)
    resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_NUM_THREADS'):
        os.environ[name] = str(threads)
    return {'max_address_space_bytes': target, 'numeric_threads': threads}
