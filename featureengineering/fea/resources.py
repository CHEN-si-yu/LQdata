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


def set_guard_factor(name: str | None) -> None:
    """告诉探针"现在算的是哪个因子"。探针没打开时是空操作（一次全局判断）。"""
    global _guard_factor
    _guard_factor = name


def install_float32_guard(report: str | None = None):
    """把 numpy 的超越函数包一层：**入参是 float32 就记一笔**（不改数值、不抛错）。

    用途：找出"哪些因子还在把 float32 送进超越函数"—— 这条路径的末位不可复现
    （见 `numeric_env_fingerprint` 与 `fea/context.py::safe_div`）。
    比逐文件读源码可靠得多：它按**实际执行**取证，能覆盖 lambda、辅助函数和派生层。
    记录写到 `report`（JSONL，按 (因子, 函数, 调用点) 去重），不改任何计算结果。

        FEA_FP_GUARD=/tmp/guard.jsonl python main.py run <因子> --sandbox /tmp/x ...
    """
    import atexit
    import json
    import sys
    import numpy as np

    if report is None:
        report = os.environ.get("FEA_FP_GUARD") or ""
    if not report:
        return None
    out = open(report, "a", encoding="utf-8")

    def cleanup():
        try:
            for row in sorted(_guard_seen):
                out.write(json.dumps({"factor": row[0], "fn": row[1], "at": row[2]},
                                     ensure_ascii=False) + "\n")
        finally:
            out.close()

    atexit.register(cleanup)

    for name in _GUARD_FNS:
        orig = getattr(np, name, None)
        if orig is None:
            continue

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
                    _guard_seen.add((_guard_factor or "?", _name, where))
                    break
            return _orig(*a, **kw)

        wrapper.__name__ = name
        setattr(np, name, wrapper)
    return report


def safe_jobs(value=0):
    """因子计算的并行进程数校验（默认 1 = 串行）。

    ★ 2026-09-25：上限由硬编码的 2 改成可用 **`FEA_MAX_JOBS`** 调整（默认仍是 2）。

    为什么：单进程只吃 ~1 核（本机 25 核），实测全量重建 18.3 min/年、CPU 大量空闲，
    时间几乎全花在等单个因子的 pandas 运算上。用户要求试 3 路并行来判断能否再提速。
    但**内存是硬约束**：每年块要装下价格层 + 派生层 + 面板（jobs=1 实测 cgroup 峰值
    29~31 GB；本机 90 GB 上限、批内 RSS 保护 64 GB、共享内存保护 80 GB）。
    所以上限**默认不动**，只在明确评估过内存余量时用环境变量显式放开：

        FEA_MAX_JOBS=3 python main.py rebuild --jobs 3

    内存由 `run_bounded` 每秒采样看护：批内 RSS > 64 GB 或容器内存 > 80 GB 会**停止本批**
    （而不是让容器 OOM）。放开上限前先实测峰值 —— 采样器见
    `scripts/mem_trace.py`（重建期间每 5 秒记一次 cgroup 与批内 RSS）。
    """
    n = int(value or 1)
    # ★ 2026-09-25：默认上限 2 → **12**。原值是在「护栏判据虚高、真实余量未知」时
    #   定的保守值，后果是 README 宣传的层并行 3.19× **在默认配置下根本拿不到**
    #   （`prebuild` 取 min(jobs, 层数)，要 4 才能展开）。现在有实测了：
    #
    #     单年块（2018，513 因子）阶梯实测：
    #       jobs=4  墙钟 350s · 真实计费内存峰值 37.5 GB / 90 GB
    #       jobs=8  墙钟 240s · 45.4 GB
    #       jobs=12 墙钟 205s · 52.2 GB   ← 最快
    #       jobs=16 墙钟 215s · 58.8 GB   ← **反而更慢**，曲线在 12 处拐弯
    #
    #   拐弯的原因是共享盘 I/O 争用（12 个进程同读同一批年分区），不是 CPU 也不是内存
    #   —— 25 核在 jobs=12 时只用到 11.8 核，内存离 90 GB 还有 38 GB。所以上限设 12
    #   是「实测最优」，不是拍的；再往上加只会更慢且更接近护栏。
    cap = int(os.environ.get("FEA_MAX_JOBS", "12"))
    if not 1 <= n <= cap:
        raise ValueError(f"因子计算并行度必须在 1~{cap}；默认串行。"
                         f"要提高上限设 FEA_MAX_JOBS（见 fea/resources.py 的说明），"
                         f"并先确认内存余量")
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
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    return target


def run_bounded(cmd, *, cwd, stdout):
    """只管理本次启动的进程组；按**容器真实计费内存**决定是否停批。

    ## 判据的演进（2026-09-25：两个方向相反的错，实测都抓到了）

    1. **触发太晚（危险）** —— 原来把 `total_rss` 当"匿名内存"、阈值写死 80 GB。
       但容器被计费的是 `total_rss + total_cache`：实测本机
       `total_rss 32.9 GB` + `total_cache 11.0 GB` = `usage_in_bytes 44.6 GB`，
       判据**漏看 11.7 GB**。于是"anon>80 GB 才停"意味着真实用量要涨到 ~91.7 GB
       才触发 —— **会在护栏动作之前先被内核 OOM 打死**。本工作负载专读
       6.86 亿行 / 6.24 亿行的 parquet，页缓存正是它最大的一块波动量。
       （与 `scripts/backfill_history.py` 里那个写死 v2 的 `memory.max` 是同一类问题。）
    2. **触发太早（浪费）** —— 批内判据把进程组内每个进程的 `VmRSS` 相加，
       而 fork 出的 worker 与父进程共享价格层/派生层（COW），还 mmap 了 parquet。
       共享页被**每个进程各算一遍**，实测虚高 1.3~1.5×：jobs=4 时判据峰值 39.5 GB、
       同期真实计费 37.5 GB，而真实 `total_rss` 只有 26.6 GB。按 64 GB 停批≈
       真实用量 44 GB 就掐断，**容器一半内存用不上**，直接压死并行度。

    现在：**主判据 = 容器真实计费内存 vs 真实上限**（两者都从 cgroup 动态读，
    不再写死 80/90）；第二道 = 不可回收量（本机 swap=0、shmem=0，`total_rss`/`anon`
    完全不可回收，顶满即 OOM）；批内 VmRSS 求和**降级为诊断输出**，只留一个
    远高于真实需求的兜底阈值，防止"进程组异常膨胀但 cgroup 没反映"这种情况。
    """
    import signal
    import subprocess
    import time
    from pathlib import Path
    child=subprocess.Popen(cmd,cwd=cwd,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True)
    peak_rss=peak_anon=peak_used=0
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
    _used_cap=int(_limit*0.90) if _limit else 80*1024**3
    # 不可回收量离上限 14 GB 就该停（本机 swap=0，anon 顶满即 OOM）。
    _anon_cap=(_limit-14*1024**3) if _limit else _used_cap
    # 批内 VmRSS 兜底：只防"进程组异常膨胀"，远高于真实需求，不再当主判据。
    _batch_backstop=160*1024**3
    try:
        while child.poll() is None:
            rss=0
            for directory in Path("/proc").iterdir():
                if not directory.name.isdigit():continue
                try:
                    if os.getpgid(int(directory.name))!=child.pid:continue
                    for line in (directory/"status").read_text().splitlines():
                        if line.startswith("VmRSS:"):rss+=int(line.split()[1])*1024
                except (OSError,ProcessLookupError):pass
            try:
                stat=dict(line.split() for line in _stat_path.read_text().splitlines())
                anon=int(stat.get(_stat_key,0))
            except OSError:anon=0
            used=_read_int(_use_path) or 0
            peak_rss=max(peak_rss,rss);peak_anon=max(peak_anon,anon)
            peak_used=max(peak_used,used)
            if used>_used_cap:
                stopped=True;reason=f"容器计费内存 {used/2**30:.1f} GB > {_used_cap/2**30:.1f} GB"
            elif anon>_anon_cap:
                stopped=True;reason=f"不可回收内存 {anon/2**30:.1f} GB > {_anon_cap/2**30:.1f} GB"
            elif rss>_batch_backstop:
                stopped=True;reason=f"批内 RSS 异常 {rss/2**30:.1f} GB > {_batch_backstop/2**30:.1f} GB"
            if stopped:
                os.killpg(child.pid,signal.SIGTERM)
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
    return {"returncode":rc,"peak_rss_bytes":peak_rss,"peak_shared_anon_bytes":peak_anon,
            "peak_cgroup_bytes":peak_used,"cgroup_limit_bytes":_limit,
            "memory_stopped":stopped,"stop_reason":reason}
