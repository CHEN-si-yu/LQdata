"""Conservative process and aggregate limits for the shared host."""
import os


# ══════════════════════════════════════════════════════════════════════
# 数值环境指纹（2026-09-25 新增）
# ══════════════════════════════════════════════════════════════════════

def numeric_env_fingerprint() -> str:
    """当前数值环境的指纹：numpy 版本 + 一段固定算例的**位模式**哈希。

    ## 为什么因子指纹里必须带上它

    因子落盘值的末位取决于**浮点实现**，不只是代码和输入。实测（2026-09-25）：

      · `arcsinh(float32)` 走 numpy 的 SIMD 核，与「双精度算完再舍入」在
        **24.76%** 的输入上不一致（最多 2 ULP）；float64 下与 libm **0.000%**。
      · 同一份代码 + 同一份输入（逐文件 md5 核对）+ 同一个计算计划，
        换一个解释器/另一套 numpy 跑，87 个对象的历史值就在末位漂移；
        而 `main.py` 只能把它报成「无法解释的历史变化」并返回非零。

    所以：**环境也是输入**。把指纹拼进 `recipe` 后，环境一变 → 逻辑指纹变 →
    引擎按既有规则强制全量重建（`fea/engine.py` 的「逻辑/口径已变，强制全量重建」），
    并在日志里留下原因，而**不是**让新旧位混在同一份产物里静默漂移。

    指纹本身必须便宜（每进程一次）、确定性（同环境同值）：
      · numpy 版本字符串；
      · 一组固定输入上的 float64 与 float32 超越函数结果的位模式 md5。
        带 float32 是有意的 —— 正是它让「换了一套 float32 SIMD 核」这件事可见。
    """
    global _NUMENV_FP
    if _NUMENV_FP is not None:
        return _NUMENV_FP
    import hashlib
    import numpy as np

    # 固定输入：不用随机数，避免任何种子差异
    x = (np.arange(1, 65, dtype=np.float64) / 64.0) - 0.5          # ±0.5
    y = np.abs(np.arange(1, 65, dtype=np.float64) / 4096.0) + 1e-9  # 与"极小值格子"同量级
    h = hashlib.md5()
    h.update(np.__version__.encode("utf-8"))
    with np.errstate(all="ignore"):
        for arr, fns in ((x, ("arcsinh", "log1p", "expm1", "tanh", "power")),
                         (y, ("log1p", "expm1"))):
            for cv in ("float64", "float32"):
                a = arr.astype(cv)
                for fn in fns:
                    f = getattr(np, fn)
                    # power 取绝对值：负底数的非整数次幂是 NaN，探针就失去分辨力了
                    v = f(np.abs(a), 1.5) if fn == "power" else f(a)
                    h.update(np.asarray(v).tobytes())
    _NUMENV_FP = h.hexdigest()[:12]
    return _NUMENV_FP


_NUMENV_FP: str | None = None


# ══════════════════════════════════════════════════════════════════════
# float32 超越函数探针（2026-09-25 新增，只在 FEA_FP_GUARD 打开时生效）
# ══════════════════════════════════════════════════════════════════════

_GUARD_FNS = ("arcsinh", "log", "log1p", "log2", "log10", "exp", "expm1",
              "power", "tanh", "sinh", "cosh", "arctanh", "sin", "cos",
              "arcsin", "arccos", "arctan")
_guard_factor: str | None = None
_guard_seen: set = set()
_guard_originals: dict = {}


def uninstall_float32_guard():
    """Restore NumPy entrypoints after a scoped audit or test."""
    import numpy as np
    for name, original in _guard_originals.items():
        setattr(np, name, original)
    _guard_originals.clear()


def set_guard_factor(name: str | None) -> None:
    """告诉探针"现在算的是哪个因子"。探针没打开时是空操作（一次全局判断）。"""
    global _guard_factor
    _guard_factor = name


def install_float32_guard(report: str | None = None, project_root=None):
    """把 numpy 的超越函数包一层：**入参是 float32 就记一笔**（不改数值、不抛错）。

    用途：找出"哪些因子还在把 float32 送进超越函数"—— 这条路径的末位不可复现
    （见 `numeric_env_fingerprint` 与 `fea/context.py::safe_div`）。
    比逐文件读源码可靠得多：它按**实际执行**取证，能覆盖 lambda、辅助函数和派生层。
    记录写到 `report`（JSONL，按 (因子, 函数, 调用点) 去重），不改任何计算结果。

        FEA_FP_GUARD=/tmp/guard.jsonl python main.py run <因子> --sandbox /tmp/x ...
    """
    import json
    import sys
    import numpy as np
    uninstall_float32_guard()

    if report is None:
        report = os.environ.get("FEA_FP_GUARD") or ""
    if not report:
        return None
    from pathlib import Path
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    report = (root / report).resolve()
    if not report.is_relative_to(root):
        raise ValueError("FEA_FP_GUARD 必须在本次项目运行目录内")
    report.parent.mkdir(parents=True, exist_ok=True)

    for name in _GUARD_FNS:
        orig = getattr(np, name, None)
        if orig is None:
            continue
        _guard_originals[name] = orig

        def wrapper(*a, _orig=orig, _name=name, **kw):
            for v in a:
                # ★ 必须看 `dtype` 而不是 `isinstance(v, np.ndarray)`：
                #   `np.log1p(df["volume_ratio"])` 这种调用传进来的是 **pandas Series**，
                #   ufunc 内部照样按 float32 走 SIMD 核，但 Series 不是 ndarray
                #   —— 实测这个盲点让第一轮探针整轮零命中（假阴性）。
                dt = getattr(v, "dtype", None)
                if dt is not None and dt in (np.float32, np.float16,
                                             np.dtype("float32"), np.dtype("float16")):
                    try:
                        fr = sys._getframe(1)
                        where = f"{fr.f_code.co_filename.rsplit('/', 1)[-1]}:{fr.f_lineno}"
                    except Exception:                              # noqa: BLE001
                        where = "?"
                    row = (_guard_factor or "?", _name, where)
                    if row not in _guard_seen:
                        _guard_seen.add(row)
                        # Persist immediately without retaining an open handle.
                        # A killed worker still leaves its observed evidence.
                        with report.open("a", encoding="utf-8") as out:
                            out.write(json.dumps({"factor": row[0], "fn": row[1], "at": row[2]},
                                                 ensure_ascii=False) + "\n")
                    break
            return _orig(*a, **kw)

        wrapper.__name__ = name
        setattr(np, name, wrapper)
    return report


_POLICY = {"default_jobs": 8, "max_jobs": 12, "memory_budget_gib": 90,
           "process_address_space_gib": 24}


def resource_policy(policy=None):
    """Resolve project limits; the task budget may be tightened, never >90GiB."""
    if hasattr(policy, "raw"):
        policy = policy.raw.get("resources", {})
    out = {**_POLICY, **(policy or {})}
    for env, key in (("FEA_MAX_JOBS", "max_jobs"),
                     ("FEA_DEFAULT_JOBS", "default_jobs"),
                     ("FEA_MEMORY_BUDGET_GIB", "memory_budget_gib")):
        if os.environ.get(env):
            out[key] = float(os.environ[env]) if key == "memory_budget_gib" else int(os.environ[env])
    if not 1 <= float(out["memory_budget_gib"]) <= 90:
        raise ValueError("单次因子任务 memory_budget_gib 必须在 1~90 GiB")
    if int(out["max_jobs"]) < 1 or int(out["default_jobs"]) < 1:
        raise ValueError("default_jobs/max_jobs 必须为正整数")
    if float(out["process_address_space_gib"]) <= 0:
        raise ValueError("process_address_space_gib 必须为正数")
    return out


def available_cpus():
    """Respect Linux affinity and CPU quotas, including the current v1 host."""
    from pathlib import Path
    count = os.cpu_count() or 1
    try:
        count = min(count, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    try:
        v2 = Path("/sys/fs/cgroup/cpu.max")
        if v2.exists():
            quota, period = v2.read_text().split()
            if quota != "max":
                count = min(count, max(1, int(quota) // int(period)))
        else:
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                count = min(count, max(1, quota // period))
    except (OSError, ValueError):
        pass
    return max(1, count)


def memory_limits(container_limit=None, policy=None):
    """Charged-memory guard leaves 10% headroom under both independent caps."""
    project_limit = int(float(resource_policy(policy)["memory_budget_gib"]) * 1024**3)
    effective = min(project_limit, container_limit) if container_limit else project_limit
    return {"project": project_limit, "effective": effective,
            "charged": int(effective * 0.90),
            "anon": max(int(effective * 0.75), effective - 14*1024**3)}


def safe_jobs(value=0, policy=None):
    """Honor explicit positive workers; zero selects a project/CPU-capped default."""
    n = int(value or 0)
    if n < 0:
        raise ValueError("因子计算并行度必须为正整数；0 使用自动配置")
    if n > 0:
        return n
    resolved = resource_policy(policy)
    cap = min(int(resolved["max_jobs"]), available_cpus())
    return min(int(resolved["default_jobs"]), cap)


def configure(policy=None, *, enforce_limits=True):
    """Configure worker threads; explicit production jobs skip project memory limits."""
    global _POLICY
    _POLICY = resource_policy(policy)
    target = None
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
        os.environ[key] = "1"
    if enforce_limits:
        target = int(float(_POLICY["process_address_space_gib"]) * 1024**3)
        try:
            import resource
        except ImportError:
            return target
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        for bound in (soft, hard):
            if bound != resource.RLIM_INFINITY:
                target = min(target, bound)
        resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    return target


def run_bounded(cmd, *, cwd, stdout, policy=None):
    'Run this child group under the <=90 GiB project budget. Charged cgroup memory includes cached pages. PSS counts shared pages proportionally; summed RSS is diagnostic only.'
    import signal
    import subprocess
    import time
    from pathlib import Path
    _HZ = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    peak_rss=peak_anon=peak_used=peak_pss=0
    peak_cpu_ticks=0
    stopped=False;reason=""
    # ---- cgroup v1 / v2：真实计费用量、真实上限、不可回收量键 ----
    if Path("/sys/fs/cgroup/memory.current").exists() and Path("/sys/fs/cgroup/memory.max").exists():
        _use_path=Path("/sys/fs/cgroup/memory.current")
        _lim_path=Path("/sys/fs/cgroup/memory.max")
        _stat_path=Path("/sys/fs/cgroup/memory.stat");_stat_key="anon"
    else:
        _use_path=Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        _lim_path=Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        _stat_path=Path("/sys/fs/cgroup/memory/memory.stat")
        _stat_key="total_rss"          # v1 没有 anon 键
    def _read_int(p):
        try:return int(p.read_text().strip())
        except (OSError,ValueError):return None
    _limit=_read_int(_lim_path)
    if _limit is None or _limit>1<<62:_limit=None      # "max"/无限大哨兵值
    # 主判据留 10% 余量：吸收写入期脏页、同 cgroup 内其他任务与内核自身。
    limits = memory_limits(_limit, policy)
    _project_limit, _used_cap = limits["project"], limits["charged"]
    # 不可回收量离上限 14 GB 就该停（本机 swap=0，anon 顶满即 OOM）。
    _anon_cap = limits["anon"]
    # Validate policy before starting a subprocess; an invalid budget must not
    # leave an unmonitored child running.
    child=subprocess.Popen(cmd,cwd=cwd,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True)
    # ★★ 2026-10-03 实测踩过：driver 被 SIGTERM 时 Python 默认直接退出、不跑清理，
    #   留在独立会话里的整棵子进程树（coordinator + 年块 + 因子 worker）会变孤儿
    #   继续写 data/。这里装一对转发处理器：收到信号先 killpg 整组，再按默认语义退出。
    _old_handlers={}
    def _forward(signum, frame):                                   # noqa: ARG001
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try: os.killpg(child.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError): pass
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    try:
        for _sig in (signal.SIGTERM, signal.SIGINT):
            _old_handlers[_sig]=signal.signal(_sig,_forward)
    except ValueError:
        _old_handlers={}          # 非主线程调用时不能装处理器；退回默认语义
    try:
        while child.poll() is None:
            rss=pss=0;cpu_ticks=0
            for directory in Path("/proc").iterdir():
                if not directory.name.isdigit():continue
                try:
                    if os.getpgid(int(directory.name))!=child.pid:continue
                    for line in (directory/"status").read_text().splitlines():
                        if line.startswith("VmRSS:"):rss+=int(line.split()[1])*1024
                    # PSS counts shared fork pages proportionally rather than
                    # duplicating every worker's RSS. RSS is diagnostic only.
                    try:
                        for line in (directory/"smaps_rollup").read_text().splitlines():
                            if line.startswith("Pss:"):pss+=int(line.split()[1])*1024
                    except OSError:
                        pass
                    # ★ 2026-10-03 资源记账：顺带累计进程组 CPU（含 cutime/cstime，
                    #   即已回收子进程的 CPU）→ 返回 cpu_seconds 供利用率台账。
                    #   累计量单调不减，取采样最大值即全程消耗。
                    try:
                        raw=(directory/"stat").read_text()
                        rp=raw.rindex(")");parts=raw[rp+2:].split()
                        cpu_ticks+=int(parts[11])+int(parts[12])+int(parts[13])+int(parts[14])
                    except (OSError,ValueError,IndexError):
                        pass
                except (OSError,ProcessLookupError):pass
            try:
                stat=dict(line.split() for line in _stat_path.read_text().splitlines())
                anon=int(stat.get(_stat_key,0))
            except OSError:anon=0
            used=_read_int(_use_path) or 0
            peak_rss=max(peak_rss,rss);peak_anon=max(peak_anon,anon)
            peak_used=max(peak_used,used)
            peak_pss=max(peak_pss,pss);peak_cpu_ticks=max(peak_cpu_ticks,cpu_ticks)
            if used>_used_cap:
                stopped=True;reason=f"计费内存 {used/2**30:.1f} GiB > 项目护栏 {_used_cap/2**30:.1f} GiB"
            elif anon>_anon_cap:
                stopped=True;reason=f"不可回收内存 {anon/2**30:.1f} GB > {_anon_cap/2**30:.1f} GB"
            elif pss>_used_cap:
                stopped=True;reason=f"本批 PSS {pss/2**30:.1f} GiB > 项目护栏 {_used_cap/2**30:.1f} GiB"
            if stopped:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                break
            time.sleep(1)
        rc=child.wait()
    except BaseException:
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    # `peak_cgroup_bytes` 是**真实计费内存**峰值（主判据用的就是它）；
    # `peak_rss_bytes` 保留但只作诊断 —— 它把共享页重复计数，会明显虚高，别拿它当结论。
    # `cpu_seconds` 是整个进程组（含全部 worker）消耗的 CPU 秒，供资源台账算利用率。
    for _sig,_handler in _old_handlers.items():
        try: signal.signal(_sig,_handler)
        except ValueError: pass
    return {"returncode":rc,"peak_rss_bytes":peak_rss,"peak_shared_anon_bytes":peak_anon,
            "peak_cgroup_bytes":peak_used,"peak_pss_bytes":peak_pss,
            "cgroup_limit_bytes":_limit,"project_memory_limit_bytes":_project_limit,
            "memory_guard_bytes":_used_cap,"cpu_seconds":round(peak_cpu_ticks/_HZ,1),
            "memory_stopped":stopped,"stop_reason":reason}
