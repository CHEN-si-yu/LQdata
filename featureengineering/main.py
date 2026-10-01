#!/usr/bin/env python3
"""因子工程 —— 主入口（模块②）。

日常增量更新（你每天手动跑这一条即可）：
    python main.py run

首次全量 / 重算全部：
    python main.py rebuild --jobs 1

只跑某几个因子：
    python main.py run roe_ttm yoy_revenue

其它：
    python main.py list           列出所有因子与本地进度
    python main.py status         汇总已落地因子统计
    python main.py check          基础体检：universe / 值域 / 格式；PIT 用 audit-pit
    python main.py docs           刷新 artifacts/catalog 的完整因子字典及 README 摘要

★ 上游数据（模块①）产出在 ../datadownload/data，本模块只读不写。
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    import importlib.util

    # ★★ 2026-09-25：**无条件钉死共享解释器**（以前只在当前解释器缺 numpy 时才切）。
    #
    #   为什么必须钉：因子落盘值的末位取决于**浮点实现**。实测同一份代码 + 同一份输入
    #   （逐文件 md5 核对）+ 同一个计算计划，只因为跑在另一套 numpy 上，87 个对象的
    #   历史值就在末位变了，而 `main.py` 只能把它报成「无法解释的历史变化」并返回非零
    #   （`arcsinh(float32)` 在 numpy 的 SIMD 核里与正确舍入差 24.76% 的输入）。
    #   容器镜像里存在**第二套 python**（`/root/miniconda3/bin/python`，带自己的 numpy），
    #   而本模块跑在与上游共享的盘上、多个端口并行 —— 不钉死就会出现「同一入口、
    #   不同数值」且完全不留痕。缺 numpy 才切的老逻辑正好漏掉这种情况：那套 python
    #   自己是带 numpy 的。
    #
    #   钉死之后：数值环境由 `fea/resources.py::numeric_env_fingerprint()` 写进每个因子的
    #   `recipe`，环境一变 → 指纹变 → 引擎强制全量重建并在日志里写明原因，
    #   不会再把两套浮点实现的结果混进同一份产物。
    shared = Path(__file__).resolve().parent.parent / "miniconda3/bin/python"
    if shared.exists() and Path(sys.executable).resolve() != shared.resolve():
        missing = [n for n in ("numpy", "pandas", "pyarrow", "yaml")
                   if importlib.util.find_spec(n) is None]
        if missing:
            print(f"[main] 当前解释器缺 {missing}，切换到共享环境 {shared}", flush=True)
        os.execv(str(shared), [str(shared), str(Path(__file__).resolve()), *sys.argv[1:]])
    if not shared.exists():
        print(f"⚠️ 找不到共享解释器 {shared}，将用当前解释器运行；"
              f"数值环境可能与其他运行不一致（见 fea/resources.py 的数值指纹）", flush=True)
    sys.dont_write_bytecode = True

# ★★ BLAS 线程数 —— **必须在 `import numpy` 之前设置**（numpy/BLAS 在导入时读这些变量）。
#
#   本机环境默认 `OMP_NUM_THREADS=32` / `MKL_NUM_THREADS=32`。而本模块的并行模型是
#   **多进程**（`--jobs 8`）：每个 worker 再各开 32 个线程 → 250+ 线程抢 32 个核，
#   实测 load average 冲到 **60**，单任务耗时翻好几倍（同一批任务：增量跑里
#   `adx_14` 99.8s，而单进程串行时不到 2s）。
#
#   因子计算是**逐元素、内存带宽受限**的 pandas/numpy 运算，多线程几乎不加速，
#   在多进程下只会互相踩踏。所以默认压到 **1**。
#   CLI 随后的 resources.configure() 会将主要数值库线程固定为 1，服从内存保护设置。
_THREADS = os.environ.get("FEA_BLAS_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _THREADS

from fea.resources import configure, safe_jobs
if __name__ == "__main__":
    configure()
    # ★ 数值环境指纹：登录式地印一行，方便把「这次的值是哪个环境算的」对上号。
    #   它同时拼在每个因子的 recipe 里（见 fea/engine.py::_recipe）。
    from fea.resources import numeric_env_fingerprint as _numenv
    import numpy as _np
    print(f"[env] python {sys.version.split()[0]} · numpy {_np.__version__} · "
          f"数值指纹 {_numenv()}", flush=True)
    # float32 超越函数探针（默认关闭）：FEA_FP_GUARD=<jsonl> 时按实际执行取证。
    import os as _os
    if _os.environ.get("FEA_FP_GUARD"):
        from fea.resources import install_float32_guard
        install_float32_guard(_os.environ["FEA_FP_GUARD"])
        print(f"[guard] float32 超越函数探针已启用 → {_os.environ['FEA_FP_GUARD']}", flush=True)

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import factors  # noqa: F401,E402  —— import 即完成因子注册
from fea import config as cfg_mod                        # noqa: E402
from fea import store                                    # noqa: E402
from fea.dates import int_to_str, str_to_int, today_int   # noqa: E402
from fea.engine import Engine                            # noqa: E402
from fea.manifest import Manifest                        # noqa: E402
from fea.spec import all_specs, get, REGISTRY            # noqa: E402

BANNER = """
日频因子工程：固定2115只股票；市场因子每日一个标量；默认输出2018年至上游最新日。
股票/标签：trade_date, stock_code, value, rank。市场：trade_date, value。
价格使用原始价与当时累计复权因子；滞后数据按可得日；未来收益仅作标签。
"""


def _mutating_cli(argv):
    if "--dry-run" in argv or "--sandbox" in argv:
        return False
    for i, arg in enumerate(argv):
        base = Path(arg).name
        if base == "backfill_history.py":
            return True
        if base == "main.py" and (i == len(argv)-1 or argv[i+1] in {"run", "rebuild"}):
            return True
    return False


def _running_pids() -> list[int]:
    """找出其它正在跑的**本模块** `main.py run`（共享盘无文件锁，两个实例同时写会打架）。

    ★ 必须限定到本模块目录：模块① 也叫 `main.py`，且常年有 `main.py run` 在跑，
    只按 `main.py run` 匹配会把上游下载进程误判成冲突，导致本模块永远起不来。
    判定依据：argv 里 main.py 的目录，或该进程的 cwd，落在本模块根目录下。
    """
    me = os.getpid()
    found = []
    proc = Path("/proc")
    # A yearly worker is allowed inside its own rebuild driver, but not beside
    # a different rebuild. Ignore only real ancestors, not all rebuild commands.
    ancestors = set()
    ancestor = os.getppid()
    while ancestor > 1 and ancestor not in ancestors:
        ancestors.add(ancestor)
        try:
            status = (proc / str(ancestor) / "status").read_text()
            ancestor = int(next(x.split()[1] for x in status.splitlines() if x.startswith("PPid:")))
        except (OSError, StopIteration, ValueError):
            break
    if not proc.is_dir():
        return found
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid == me or pid in ancestors:
            continue
        try:
            raw = (p / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [a for a in raw.decode("utf-8", "replace").split("\0") if a]
        # ★ 沙箱运行不占用生产：它的产物在独立目录，不算冲突（见 cmd_run 的说明）
        if "--sandbox" in argv:
            continue
        # ★★ 必须跳过包在外面的「壳」进程：`timeout 900 python main.py run …` 的 argv 里
        #   同样含 "main.py run"，会被误判成"已有实例在跑"而拒绝启动（实测踩过两次，
        #   且报出的 PID 每次都变 —— 因为那是壳进程）。真正的因子进程一定是 python。
        try:
            exe = Path(os.readlink(p / "exe")).name.lower()
        except OSError:
            exe = Path(argv[0]).name.lower() if argv else ""
        if "python" not in exe:
            continue
        if not _mutating_cli(argv):
            continue
        try:
            cwd = Path(os.readlink(p / "cwd")).resolve()
        except OSError:
            cwd = None
        if cwd == ROOT:
            found.append(pid)
            continue
        for a in argv:
            # ★ 只认绝对路径：相对路径的 "main.py" 会相对**本进程**的 cwd 解析，
            #   从而把上游那个 main.py 误判成自己人（实测踩过）
            if a.endswith("main.py") and a.startswith("/"):
                try:
                    if Path(a).resolve().parent == ROOT:
                        found.append(pid)
                except OSError:
                    pass
                break
    return found


def _pick(names: list[str], group: list[str] | None):
    if names:
        out = []
        for n in names:
            if n not in REGISTRY:
                raise SystemExit(f"未知因子：{n}\n用 `python main.py list` 查看可用名称")
            out.append(get(n))
        return out
    specs = all_specs()
    if group:
        specs = [s for s in specs if s.group in group]
    return [s for s in specs if s.enabled]


# ------------------------------------------------------------------ 命令
def cmd_run(args, cfg) -> int:
    """只运行已经限制在一年的分片；外层负责内存隔离与台账。"""
    if not getattr(args, "slice_worker", False):
        from fea.daily import run
        return run(args, cfg, _pick)
    import json
    from fea.daily import expand_dependencies
    specs = _pick(args.factors, args.group)
    if getattr(args, "plan_file", None):
        cfg.raw["_explicit_plan"] = json.loads(Path(args.plan_file).read_text())
    engine = Engine(cfg)
    engine.force_refresh = bool(args.refresh)
    end = str_to_int(args.end) if args.end else engine.baseline_last_day()
    engine.override_start = str_to_int(args.start) if args.start else None
    # jobs 同时给派生层并行用：那段在因子 worker fork 之前，两者不重叠。
    todo = engine.prebuild(specs, end, rebuild=args.rebuild,
                           jobs=safe_jobs(args.jobs))
    pending = {s.name: (s, m, plan) for s, m, plan in todo}
    failed, total_rows = {}, 0
    while pending:
        wave = [item for item in pending.values()
                if not any(d in pending for d in item[0].deps)]
        if not wave: raise ValueError("因子依赖存在循环")
        ready = []
        for item in wave:
            spec = item[0]
            if any(d in failed for d in spec.deps):
                failed[spec.name] = "父因子失败，未计算"
                pending.pop(spec.name)
            else: ready.append(item)
        tasks = [(s.name, y, ds) for s, _, plan in ready for y, ds in sorted(plan.items())]
        tasks.sort(key=lambda t:(t[1], REGISTRY[t[0]].warmup_days, t[0]))
        def report(r):
            print(f"{r['factor']} {r['year']}: " + (r.get("error") or f"{r.get('rows',0)} 行 / {r.get('seconds',0)}s"), flush=True)
        jobs = safe_jobs(args.jobs)
        if tasks and jobs > 1:
            results = engine.run_parallel(tasks, jobs, on_done=report)
        else:
            results = []
            for name, year, days in tasks:
                try: r = engine.run_year(REGISTRY[name], year, days)
                except Exception as exc:
                    logging.exception("因子计算失败 %s %s", name, year)
                    r = {"factor": name, "year": year, "error": repr(exc)}
                results.append(r); report(r)
        # 在子因子读取之前完成父因子的状态提交，并丢弃父因子读取缓存。
        for spec, man, plan in ready:
            own = [r for r in results if r["factor"] == spec.name]
            errors = [r["error"] for r in own if r.get("error")]
            if errors or len(own) != len(plan):
                failed[spec.name] = errors[0] if errors else "任务未全部返回"
            else:
                stats = engine._finalize(spec, man, plan, own, sum(r["seconds"] for r in own))
                total_rows += stats["rows"]
            pending.pop(spec.name)
        engine._factor_io = None
        engine.factor_io()
        engine._wm_cache.clear()
    print(f"完成 {len(todo)-len(failed)}/{len(todo)} 项，共写 {total_rows:,} 行；失败 {failed}", flush=True)
    return int(bool(failed))


def default_jobs() -> int:
    """默认并行度：留几个核给系统，且不超过核数。"""
    n = os.cpu_count() or 4
    return 1  # 60 GiB 共享服务器：默认串行；显式并行最多 2 个


def cmd_list(args, cfg) -> int:
    print(f"\n{'因子':<24}{'类别':<12}{'起点':<12}{'warmup':>7}  {'方向':<6}{'本地状态'}")
    print("-" * 108)
    for s in all_specs():
        man = Manifest.load(cfg.state_dir, s.name)
        flag = "" if s.enabled else " (禁用)"
        # manifest 里存的是 `因子指纹||股票池口径`，这里只比对前半段
        cur = man.recipe.split("||")[0] if man.recipe else ""
        stale = " ⚠逻辑已变需--rebuild" if cur and cur != s.recipe(cfg) else ""
        explicit = "" if s.start else "*"
        print(f"{s.name+flag:<24}{s.group:<12}"
              f"{s.resolved_start(cfg)+explicit:<12}{s.warmup_days:>7}  "
              f"{'高优' if s.higher_is_better else '低优':<6}{man.summary()}{stale}")
    print("-" * 108)
    print(f"共 {len(all_specs())} 个因子    起点带 * = 跟随 conf 的 default_start"
          f"（当前 {cfg.default_start}）；其余为受上游数据起点限制的显式值")
    print("约束：日频 / 固定2115只股票或市场标量 / 禁止未来信息")
    return 0


def cmd_status(args, cfg) -> int:
    print(f"\n{'因子':<24}{'行数':>13}{'年数':>5}{'非空':>7}{'占用':>10}  {'最早':<11}{'最晚':<11}")
    print("-" * 92)
    grand = 0
    for s in all_specs():
        man = Manifest.load(cfg.state_dir, s.name)
        rows = man.partition_rows()
        if not rows:
            continue
        grand += rows
        size = store.factor_size(cfg.factor_root(s), s.name)
        ds = [v.get("min_date") for v in man.partitions.values() if v.get("min_date")]
        de = [v.get("max_date") for v in man.partitions.values() if v.get("max_date")]
        print(f"{s.name:<24}{rows:>13,}{len(man.partitions):>5}"
              f"{man.nonnull_ratio():>7.0%}{size/1e6:>9.1f}M  "
              f"{min(ds) if ds else '?':<11}{max(de) if de else '?':<11}")
    print("-" * 92)
    print(f"合计 {grand:,} 行")
    return 0


def _report_contracts(engine) -> int:
    """报告 ST 内容闸门与可用时点契约的状态（S-01，2026-09-21）；返回新增的问题数。

    为什么放在逐因子体检**之前**：这两处都是「不报错、但让因子值停在过时口径上」的
    静默不一致。契约不对，后面逐因子的行数/值域再干净也不代表口径是对的。
    """
    import json as _json
    from pathlib import Path

    from fea.delay import audit_contract
    from fea.universe import st_pool_fingerprint

    bad = 0

    # ---- ① ST 内容闸门 ----
    st = st_pool_fingerprint(engine.up, engine.codes)
    p = Path(engine.cfg.state_dir) / "universe_st.json"
    base = None
    if p.exists():
        try:
            base = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            base = None
    if base is None:
        print(f"⚠️ ST 内容闸门：尚未建立基线（{p}）—— 下次 `main.py run` 会自动建立并留痕")
    elif st["sha1"] is None:
        print(f"⚠️ ST 内容闸门：本轮读不到 stock_st_info（source={st['source']}），无法与基线核对；"
              "冻结池口径下池内无 ST 事件、不改变结果")
    elif base.get("sha1") != st["sha1"]:
        bad += 1
        print(f"✘ ST 内容闸门：池内 ST 事件与基线不一致 "
              f"（基线 sha1={base.get('sha1')} n={base.get('n_events')} → "
              f"现在 sha1={st['sha1']} n={st['n_events']}）\n"
              "    → stock_st_info 不在任何因子的 deps 里，不重算就不会传播；"
              "需要一次全量重建，或显式刷新基线文件")
    else:
        print(f"✓ ST 内容闸门：池内 ST 事件与基线一致（n={st['n_events']} · source={st['source']} · "
              f"池 {len(engine.codes)} 只）")

    # ---- ② 可用时点契约 ----
    r = audit_contract()
    if r["authoritative_state"] != "ok":
        print(f"⚠️ 可用时点契约：读不到日更侧 registry._DELAY（state={r['authoritative_state']}），"
              "本轮**无法核对**（不能当作通过）")
    elif r["unexplained"]:
        bad += 1
        print(f"✘ 可用时点契约：{len(r['unexplained'])} 条未解释差异（因子侧声明比日更侧**更松**）")
        for x in r["unexplained"][:5]:
            print(f"    · {x['table']}：{x['why']}")
        print(f"    → 改 frequency.yaml 并同步 registry._DELAY，或写进 {r['ack_file']} 显式接受")
    else:
        print(f"✓ 可用时点契约：声明 / 日更侧 / 观测 三方无「声明更松」差异"
              f"（已解释 {len(r['acked'])} 条 · 更保守 {len(r['conservative'])} 条 · "
              f"声明 {r['declared_n']} 张 · 有观测 {r['observed_n']} 张）")
    return bad


def cmd_check(args, cfg) -> int:
    from fea.validation import check
    return check(cfg)


def _check_format(cfg) -> int:
    """格式统一性 —— 遍历全部因子的**全部年分区**逐条断言。

    用户的硬性要求：**互相之间保持统一的格式，仅有日期允许长短不一样**。
    所以这里断言的是「所有因子的列签名完全相同」，只允许行数与日期范围不同。

    六条：① 列名与列序 ② dtype ③ trade_date 格式 ④ stock_code 全是主板前缀
          ⑤ rank ∈ [0,1] ∪ {NaN} ⑥ 主键 (trade_date, stock_code) 无重复
    """
    from fea.store import COLUMNS, DTYPES
    print("格式统一性检查（列名/列序/dtype/日期格式/主板/rank 值域/主键唯一）")
    bad = 0
    names = [s.name for s in all_specs()]
    sigs_seen: dict[tuple, list[str]] = {}       # 实际观测到的 (列, dtype) 签名 -> 谁
    for name in names:
        years = store.factor_years(cfg.factors_dir, name)
        if not years:
            continue
        issues = []
        for y in years:
            p = store.year_path(cfg.factors_dir, name, y)
            if not p.exists():
                continue
            try:
                df = pd.read_parquet(p)
            except Exception as exc:                      # noqa: BLE001
                issues.append(f"{y}: 读取失败 {exc}")
                continue
            if list(df.columns) != COLUMNS:
                issues.append(f"{y}: 列序 {list(df.columns)} != {COLUMNS}")
            for c, want in DTYPES.items():
                if c in df.columns and str(df[c].dtype) != want:
                    issues.append(f"{y}.{c}: dtype {df[c].dtype} != {want}")
            if "trade_date" in df.columns:
                bad_d = int((~df["trade_date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")).sum())
                if bad_d:
                    issues.append(f"{y}: {bad_d} 行 trade_date 格式非法")
            if "rank" in df.columns:
                r = pd.to_numeric(df["rank"], errors="coerce").to_numpy()
                bad_r = int((np.isfinite(r) & ((r < -1e-6) | (r > 1 + 1e-6))).sum())
                if bad_r:
                    issues.append(f"{y}: {bad_r} 行 rank 越界")
            if "stock_code" in df.columns:
                bad_c = int((~df["stock_code"].astype(str)
                             .str.startswith(tuple(cfg.board_prefixes))).sum())
                if bad_c:
                    issues.append(f"{y}: {bad_c} 行 stock_code 非主板")
            dup = int(df.duplicated(["trade_date", "stock_code"]).sum()) \
                if {"trade_date", "stock_code"}.issubset(df.columns) else 0
            if dup:
                issues.append(f"{y}: {dup} 行主键重复")
            # ★ 记录**实际观测到的**列签名（不是契约常量）—— 否则「跨因子一致」
            #   这一条永远为真，等于没查。
            if list(df.columns) == COLUMNS:
                sigs_seen.setdefault(
                    tuple(str(df[c].dtype) for c in COLUMNS), []).append(f"{name}/{y}")
        if issues:
            bad += 1
            print(f"  ✘ {name}: " + "; ".join(issues[:4])
                  + (f"  …另 {len(issues) - 4} 条" if len(issues) > 4 else ""))
    want = tuple(DTYPES[c] for c in COLUMNS)
    if len(sigs_seen) > 1:
        bad += 1
        print(f"  ✘ 实际落盘的列签名有 {len(sigs_seen)} 种，必须只有 1 种：")
        for sig, who in sigs_seen.items():
            print(f"      {dict(zip(COLUMNS, sig))}  ← {len(who)} 个分区，如 {who[:3]}")
    elif sigs_seen:
        n = sum(len(v) for v in sigs_seen.values())
        print(f"  ✔ {n} 个分区 / {len({w.split('/')[0] for v in sigs_seen.values() for w in v})} "
              f"个因子的实际列签名完全一致：{dict(zip(COLUMNS, want))}")
    else:
        print("  （没有可检查的因子产物）")
    return bad


def cmd_eval(args, cfg) -> int:
    """因子有效性评价：覆盖率 / 分布 / IC / RankIC / ICIR / 自相关 / 分层单调性。

    纯本地、零 API。内存有界（外层按年、内层按因子流式处理）。
    没有 scipy，所以 RankIC 用「先取秩再算 Pearson」实现（`df.corr(method='spearman')`
    在本环境会直接抛 ModuleNotFoundError）。
    """
    from fea.eval import evaluate_all
    return evaluate_all(args, cfg)


def cmd_dayhash(args, cfg) -> int:
    """逐截面 MD5 台账（见 fea/dayhash.py 的口径说明）。"""
    from fea.dayhash import cmd_dayhash as _impl
    return _impl(args, cfg)


def cmd_dedup(args, cfg) -> int:
    """因子冗余检测（取精华去糟粕的证据来源）。"""
    from fea.dedup import cmd_dedup as _impl
    return _impl(args, cfg)


def cmd_audit_pit(args, cfg) -> int:
    """★ 前视/滞后审计：静态扫描 + 抽样截断复算。

    ## 为什么要"截断复算"这一招

    静态扫描只能抓**写出来**的未来引用（`shift(-1)`、`[t+1]`）。真正的泄漏常常是
    隐式的：全样本标准化、用整段面板算的极值、按未来窗口对齐 …… 这些在源码里
    看不出来。唯一可靠的判据是那条定义本身：

        **在 T 日能算出的值，必须只用到 ≤ T 的数据。**

    做法：把输出范围截断到 T（面板也跟着截断），重算该因子，与生产落盘的值比对。
    只用到了 ≤T 的数据 → 逐格相同；用了未来 → 必然不同。
    （因子函数拿到的是 `(T, C)` 面板，面板右端就是 T，所以"截断"天然成立。）

    ⚠️ 复算在**沙箱目录**里进行，绝不碰生产 `data/` 与 `state/`。
    """
    import copy
    import re
    import shutil
    import tempfile

    from fea.engine import Engine
    from fea.manifest import Manifest
    from fea.spec import LAGGED_DATASETS

    specs = _pick(args.factors, getattr(args, "group", None))
    specs = [s for s in specs if s.enabled]
    if not specs:
        print("没有匹配的因子")
        return 1

    print(BANNER)
    print("一、静态扫描（源码里写出来的未来引用 / 滞后表未处理）")
    print("-" * 96)
    problems = 0
    flagged_lag, flagged_future = [], []
    def _src_of(fn) -> str:
        """函数源码 + 它**闭包里**被调用的族辅助函数的源码。

        ★ 为什么要追闭包：`factors/market.py` / `market2.py` / `tradability.py` 这类
          「一个族一个 dispatcher」的模块，`spec.fn` 只是 `compute(ctx)[key]` 一行，
          真正的取数与位移都在 `compute` 指向的族函数里。只看 `spec.fn` 会把
          已经正确位移过的因子全部误报成「依赖滞后表却没做位移」——
          实测 2026-09-24 新增的 6 个两融因子就是这么被误报的。
          追一层闭包即可覆盖当前所有家庭形态，且是**精确**的（拿的是真函数对象，
          不是按名字猜）。
        """
        out = ""
        try:
            out = inspect.getsource(fn)
        except Exception:                                    # noqa: BLE001
            return out
        for cell in (getattr(fn, "__closure__", None) or ()):
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if callable(val) and getattr(val, "__code__", None) is not None:
                try:
                    out += "\n" + inspect.getsource(val)
                except Exception:                            # noqa: BLE001
                    pass
        return out

    for s in specs:
        src = _src_of(s.fn)
        if not src:
            continue
        # ① 负位移 / 未来索引：只有标签（is_label）允许
        future = bool(re.search(r"\.shift\([^)]*,\s*-\d", src)
                      or re.search(r"shift\([^,)]+,\s*-\d", src)
                      or re.search(r"\[\s*[a-z_]+\s*\+\s*1\s*\]", src))
        if future and not s.is_label:
            flagged_future.append(s.name)
            print(f"  ✘ {s.name}：源码里出现负位移/未来索引，但不是标签")
            problems += 1
        # ② 依赖了滞后表却没做位移
        lat = [d for d in s.deps if d in LAGGED_DATASETS]
        if lat and "lag_grid" not in src and "asof_daily_lagged" not in src:
            flagged_lag.append((s.name, lat))
            print(f"  ✘ {s.name}：依赖滞后表 {lat}，但函数体里没有 lag_grid/asof_daily_lagged")
            problems += 1
        # ③ ★ 2026-09-15 晚补：**位移天数也要够**。
        #    旧判据只查"源码里有没有 `lag_grid` 字样"，于是给 delay=2 的表写
        #    `lag_grid(x, 1)` 会静默通过（少移一天 = 少丢一天？不 —— 是**泄漏一天**：
        #    T 日的因子值用到了 T+1 才发布的数据，且会在数据到达后静默改变）。
        #    这里只**警告**不报错：静态解析表达式本就只能尽力（`lag_grid` 的实参可能是
        #    中间变量、可能被封装），但只要写的是字面量就能查出来。
        elif lat:
            lags = [int(n) for n in re.findall(r"lag_grid\s*\([^,)]+,\s*(\d+)\s*\)", src)]
            need = max(LAGGED_DATASETS[d] for d in lat)
            if lags and max(lags) < need:
                print(f"  ⚠️ {s.name}：依赖滞后表 {lat}（最长需 {need} 天），"
                      f"但源码里的 lag_grid 天数只有 {sorted(set(lags))} —— 可能位移不足")
                problems += 1
    if not flagged_future and not flagged_lag:
        print(f"  ✔ {len(specs)} 个因子的源码里没有发现未来引用，"
              f"滞后表依赖也都做了位移")
    print(f"  提示：`lagged_ok` 已声明的表 → {sorted(LAGGED_DATASETS)}")

    if getattr(args, "static_only", False):
        print("-" * 96)
        print(f"{'全部通过 ✔' if not problems else f'✘ {problems} 个问题'}")
        return 1 if problems else 0

    from fea.pit_audit import run_dynamic
    problems += run_dynamic(args,cfg,specs)
    print('全部通过 ✔' if not problems else f'✘ 审计未通过，共 {problems} 类问题')
    return int(bool(problems))


def cmd_docs(args, cfg) -> int:
    from fea.catalog import generate
    return generate(cfg)


def main() -> int:
    p = argparse.ArgumentParser(prog="main.py", description="因子工程（模块②）",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("-v", "--verbose", action="store_true", help="同时输出到屏幕")
    sub = p.add_subparsers(dest="cmd", required=True)
    rb = sub.add_parser("rebuild", help="按年完整重建，低并发、可断点续跑")
    rb.add_argument("--jobs", type=int, default=1)
    rb.add_argument("--from-year", type=int, default=None)
    rb.add_argument("--to-year", type=int, default=None)
    rb.add_argument("--factors", nargs="*", default=None)
    rb.add_argument("--resume", action="store_true", help="跳过已验收完成的年；默认强制重算")
    rb.add_argument("--dry-run", action="store_true")

    def _add_common(sp):
        sp.add_argument("--sandbox", default=None, metavar="DIR",
                        help="★ 沙箱模式：因子产物与状态写到 DIR 下，不碰生产 data/ 与 state/")
        sp.add_argument("--force", action="store_true", help="忽略单实例保护")

    r = sub.add_parser("run", help="计算/增量更新因子")
    r.add_argument("factors", nargs="*", help="只跑指定因子")
    r.add_argument("--slice-worker", action="store_true", help=argparse.SUPPRESS)
    r.add_argument("--plan-file", default=None, help=argparse.SUPPRESS)
    r.add_argument("--refresh", action="store_true", help="重算请求区间，保留其它区间 coverage")
    r.add_argument("--rebuild", action="store_true", help="全量重建（忽略已覆盖区间）")
    r.add_argument("--end", default=None,
                   help="截止日期 YYYY-MM-DD，默认 = 上游 stock_daily 的最后一天")
    r.add_argument("--start", default=None,
                   help="★ 本次只产出这一天之后的因子值（YYYY-MM-DD）。"
                        "用于「先只跑一段、验证无误再往前推」。不影响 warmup，"
                        "值与全量重建逐格一致")
    r.add_argument("--group", nargs="*", help="按类别过滤 quality/growth/risk/event/...")
    r.add_argument("--jobs", type=int, default=0,
                   help=f"并行进程数（默认 {default_jobs()}；1 = 串行）")
    r.add_argument("--allow-multiyear", action="store_true",
                   help=argparse.SUPPRESS)
    _add_common(r)

    lp = sub.add_parser("list", help="列出因子与本地进度")
    _add_common(lp)
    sp2 = sub.add_parser("status", help="已落地因子统计")
    _add_common(sp2)
    cp = sub.add_parser("check", help="基础体检：universe / 值域 / 格式；PIT 用 audit-pit")
    _add_common(cp)
    dp = sub.add_parser("docs", help="刷新 artifacts/catalog 的完整因子字典及 README 摘要")
    _add_common(dp)
    ep = sub.add_parser("eval", help="因子有效性评价：IC / RankIC / ICIR / 分层 / 覆盖")
    ep.add_argument("factors", nargs="*", help="只评价指定因子")
    ep.add_argument("--group", nargs="*", help="按类别过滤")
    ep.add_argument("--years", nargs=2, type=int, default=None, metavar=("Y0", "Y1"),
                    help="只评价这些年（默认全部）")
    ep.add_argument("--horizons", default="1,3,5,10,20",
                    help="标签周期，逗号分隔（默认 1,3,5,10,20）")
    ep.add_argument("--jobs", type=int, default=0, help="并行进程数")
    ep.add_argument("--out", default=None, help="报告输出目录（默认 state/eval）")
    _add_common(ep)

    dp2 = sub.add_parser("dedup", help="因子冗余检测：|ρ|≥阈值的重复簇 + 完全重复对")
    dp2.add_argument("factors", nargs="*", help="只检查指定因子")
    dp2.add_argument("--years", nargs=2, type=int, default=None, metavar=("Y0", "Y1"),
                     help="只检查这些年（默认全部）")
    dp2.add_argument("--threshold", type=float, default=0.95,
                     help="判定重复的 |ρ| 阈值（默认 0.95）")
    dp2.add_argument("--matrix", action="store_true",
                     help="老的一次性长矩阵路径（全历史 658 因子约 23 GB）；只用于对拍分块路径")
    _add_common(dp2)

    dh = sub.add_parser("dayhash", help="逐截面 MD5 台账（用户交办：验证历史不变 + 增量==全量）")
    dh.add_argument("factors", nargs="*", help="只做指定因子")
    dh.add_argument("--date", default=None, help="截止日 YYYY-MM-DD（默认 = 上游最后一天）")
    dh.add_argument("--days", type=int, default=7, help="记录最近几个交易日（默认 7）")
    dh.add_argument("--out", default=None, help="输出目录（默认 artifacts/dayhash/YYYY-MM-DD）")
    dh.add_argument("--jobs", type=int, default=0, help="并行进程数（默认 1，最多 2）")
    dh.add_argument("--verify", action="store_true",
                    help="与 .prev.tsv 里的旧台账比对，报出发生变化的 (因子, 日期)")
    _add_common(dh)

    ap = sub.add_parser("audit-pit", help="★ 前视/滞后审计：静态扫描 + 抽样截断复算")
    ap.add_argument("factors", nargs="*", help="只审计指定因子")
    ap.add_argument("--group", nargs="*", help="按类别过滤")
    ap.add_argument("--end", default=None, help="审计到哪一天（默认 = 上游基准表最后一天）")
    ap.add_argument("--jobs", type=int, default=0,
                    help=f"复算的并行进程数（默认 {default_jobs()}；★ 内存敏感，别乱加）")
    ap.add_argument("--sample", type=int, default=2, help="抽几个交易日做截断复算（默认取「中点+尾部」，共 N 个）")
    ap.add_argument("--tol", type=float, default=1e-4, help="复算相对容差（默认 1e-4）")
    ap.add_argument("--static-only", action="store_true", help="只做静态扫描（秒出）")
    _add_common(ap)

    ph = sub.add_parser("prune-history", help="清理配置起点之前的因子产物（先预览）")
    ph.add_argument("--apply", action="store_true")
    ph.add_argument("--factors", nargs="*", default=None)
    args = p.parse_args(sys.argv[1:] or ["run"])
    if args.cmd == "prune-history":
        if _running_pids():
            raise RuntimeError("已有因子任务运行，不能同时裁剪")
        from fea.history import run as prune
        prune(cfg_mod.load(), apply=args.apply, names=args.factors)
        return 0
    if args.cmd == "rebuild":
        if not args.dry_run and _running_pids():
            print("已有因子更新或重建在运行，拒绝同时改写数据。")
            return 2
        import subprocess
        cmd = [sys.executable, str(ROOT / "scripts/backfill_history.py"), "--jobs", str(safe_jobs(args.jobs))]
        for key in ("from_year", "to_year"):
            if getattr(args, key) is not None:
                cmd.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
        if args.factors: cmd.extend(["--factors", *args.factors])
        if not args.resume: cmd.append("--force")
        if args.dry_run: cmd.append("--dry-run")
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc == 0 and not args.dry_run:
            from fea.history import run as prune
            prune(cfg_mod.load(), apply=True, names=args.factors)
        return rc
    cfg_mod.setup_logging(args.verbose)
    cfg = cfg_mod.load()

    # ★ --sandbox <dir>：把因子产物与状态重定向到临时目录。
    #   多 Agent 并行开发必需 —— 否则每个 Agent 都会往生产的 data/ 与 state/ 里写。
    #   上游数据始终只读，不受影响。
    sb = getattr(args, "sandbox", None)
    if sb:
        sb = Path(sb).resolve()
        cfg.raw["paths"]["factors"] = str(sb / "factors")
        cfg.raw["paths"]["state"] = str(sb / "state")
        cfg.raw["paths"]["market_factors"] = str(sb / "market_factors")
        (sb / "factors").mkdir(parents=True, exist_ok=True)
        (sb / "state").mkdir(parents=True, exist_ok=True)
        print(f"[sandbox] 因子产物 -> {sb / 'factors'}")
        print(f"[sandbox] 状态文件 -> {sb / 'state'}")

    return {"run": cmd_run, "list": cmd_list, "status": cmd_status,
            "check": cmd_check, "docs": cmd_docs,
            "eval": cmd_eval, "dedup": cmd_dedup,
            "audit-pit": cmd_audit_pit,
            "dayhash": cmd_dayhash}[args.cmd](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
