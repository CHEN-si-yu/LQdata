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
    python main.py docs           更新 README.md 中的因子字典

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
╔══════════════════════════════════════════════════════════════════════════╗
║  ★ 因子开发的四条硬约束（下游是 A 股日横断面回归排序任务）                ║
║    ① 全部因子必须**日频**——对齐 (trade_date, stock_code) 面板后落盘      ║
║    ② 只考虑**主板 + 非 ST**，约 3000+ 只                                  ║
║       600/601/603/605（沪） + 000/001/002/003（深，含原中小板）           ║
║       排除 创业板 300/301/302、科创板 688/689、北交所 832/833/920         ║
║    ③ 输出起点由 conf/config.yaml 的 default_start 控制（见该配置，勿在此硬编码）║
║    ④ 历史因子值不得因未来的分红事件而变化（禁用前复权 qfq）               ║
║  ★ 统一格式：全部因子 4 列 trade_date/stock_code/value/rank，同 dtype；   ║
║    唯一允许的差异是**起止日期**（见 main.py check 的格式统一性检查）      ║
╚══════════════════════════════════════════════════════════════════════════╝
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
    # ★ 沙箱运行**不参与**单实例保护：它的产物与状态都重定向到独立目录，
    #   和生产 run 互不干扰。多 Agent 并行开发时每个 Agent 一个沙箱，
    #   原来会被这条保护挡住（"已有因子进程在跑"），白白浪费时间。
    if not args.force and not getattr(args, "sandbox", None):
        others = _running_pids()
        if others:
            pids = " ".join(str(p) for p in others)
            print(f"\n✘ 已有因子进程在跑（PID {pids}），拒绝启动。\n"
                  f"  两个实例同时写共享盘会互相覆盖。\n"
                  f"  确认要停：kill {pids}   （★ 不要用 pkill -f，会误杀自己）\n"
                  f"  确实要并行：--force")
            return 2

    specs = _pick(args.factors, args.group)
    if not specs:
        print("没有匹配的因子")
        return 1

    mode = "全量重建" if args.rebuild else "增量"
    engine = Engine(cfg)
    engine.force_refresh = bool(getattr(args, "refresh", False))
    # ★ 默认右端点 = **上游基准表实际覆盖到的最后一天**，不是「今天」。
    #   见 Engine.baseline_last_day 的说明：日历会延伸到未来，而财务类因子
    #   只依赖已公告的季报，用「今天」会在一个上游数据还不存在的交易日上产出值。
    if args.end:
        end_i = str_to_int(args.end)
        end_src = "（显式指定）"
    else:
        end_i = engine.baseline_last_day()
        end_src = "（= 上游 stock_daily 的最后一天，闸门自动判定）"
    print(BANNER)
    print(f"因子工程 · {mode}更新 · 共 {len(specs)} 个因子 · "
          f"截止 {int_to_str(end_i)} {end_src}")
    print("-" * 78)
    if args.start:
        engine.override_start = str_to_int(args.start)
        print(f"★ 只产出 {args.start} 之后的因子值（--start）")
    print(f"股票池：主板 {len(engine.codes)} 只 · 交易日历 {len(engine.cal.days)} 天")

    # ★★ 多年守卫（2026-09-17 加，因为**真的爆过**）：真正决定内存的不是"请求跨度"，
    #   而是**计划里实际有几年的活**。价格层/派生层缓存只扩不缩，一旦计划跨多年，
    #   每个 worker 都会把全历史读进内存（实测 7~8 GB/worker）——
    #   2026-09-17 那次是「2 个因子 × 14 年 × jobs 4~8」，两次都在第 27 个任务上
    #   被 OOM-Kill（报 `BrokenProcessPool`，不是 MemoryError，很容易误判成"偶发"）。
    #   日增量/单年重建只跨 1~2 年，永远不会触发这里；多年历史请走按年分块的脚本。
    years = engine.plan_years(specs, end_i, rebuild=args.rebuild)
    span_all = sorted({y for ys in years.values() for y in ys})
    heavy = sorted(n for n, ys in years.items() if len(ys) >= 3)
    if heavy and not args.allow_multiyear:
        # ★ 只把**跨 ≥3 年的那几个因子**摘出去，其余因子照常跑 ——
        #   「某个因子缺多年历史」不该连累每日增量（那条命令必须稳、快、幂等）。
        heavy_set = set(heavy)
        specs = [s for s in specs if s.name not in heavy_set]
        lo = min(min(years[n]) for n in heavy)
        hi = max(max(years[n]) for n in heavy)
        print(f"\n⚠ 以下 {len(heavy)} 个因子缺 {lo}..{hi} 的多段历史，已从**本次运行**中摘出：\n"
              f"    {' '.join(heavy[:6])}{' …' if len(heavy) > 6 else ''}\n"
              f"  理由：价格层 / 派生层缓存**只扩不缩** —— 一个因子跨 ≥3 年时，跑它的 worker\n"
              f"  会把全历史读进内存（实测 7~8 GB/worker），并行数一上去就顶爆 cgroup 上限\n"
              f"  （表现是 BrokenProcessPool，不是 MemoryError —— 2026-09-17 就是这么爆的）。\n"
              f"  补这些因子的历史请按年分块跑（每块一个独立进程）：\n"
              f"    /autodl-fs/data/miniconda3/bin/python main.py rebuild --jobs 1 "
              f"--from-year {lo} --to-year {hi}\n"
              f"  （确实要单进程硬跑全部年份：加 --allow-multiyear，内存自负）", flush=True)
        if not specs:
            print("✘ 本次要跑的因子全被摘出，没有可执行的活。")
            return 2
        print(f"  → 本次继续跑其余 {len(specs)} 个因子（各自缺的那几天）\n", flush=True)
    elif len(span_all) >= 3:
        # 单个因子都不跨多年，但合起来跨了 —— 计划整体仍可能让 worker 攒下多年窗口。
        print(f"⚠ 本次计划的年份跨度 {min(span_all)}..{max(span_all)}（{len(span_all)} 年），"
              f"建议停止本次运行，改用 main.py rebuild 按年运行以控制内存。")
    t0 = time.time()

    # ---- 规划 + 预建共享状态（衍生层/ST 事件），供 fork 出的 worker 写时复制继承
    todo = engine.prebuild(specs, end_i, rebuild=args.rebuild)
    if not todo:
        print("所有因子都已是最新，无需重算。")
        return 0

    # ★ 任务排序用「年优先」而不是「因子优先」：相邻任务共享同一个输出年
    #   → warmup 相同 → 面板窗口相同 → 各 worker 的派生缓存/价格缓存不必反复清空重建。
    #   因子优先的话每个 worker 会沿着年份来回跳，缓存几乎每任务都失效。
    #
    # ★★ 耦合因子（`deps` 里写的是**别的因子名**，即 `ctx.load_factor` 的父因子）
    #    必须**等父因子落盘之后**再算 —— `FactorIO.load` 读的是产物文件，
    #    跑在父因子前面会静默拿到全 NaN（它只 warning，不报错）。
    #    所以分两趟：第一趟 = 全部非耦合因子；第二趟 = 耦合因子。
    def _is_coupling(sp) -> bool:
        return any(d in REGISTRY for d in sp.deps)

    todo_nc = [t for t in todo if not _is_coupling(t[0])]
    todo_cp = [t for t in todo if _is_coupling(t[0])]
    if todo_cp:
        print(f"耦合因子 {len(todo_cp)} 个：等第一趟（{len(todo_nc)} 个）全部落盘后再算")

    results: list[dict] = []
    done = {"n": 0, "total": 1}

    def on_done(r):
        done["n"] += 1
        tag = "✓" if not r.get("error") else "✘"
        print(f"  [{done['n']:>3}/{done['total']}] {tag} "
              f"{r.get('factor','?'):<22} {r.get('year','?')}  "
              f"{r.get('rows',0):>9,} 行"
              f"{r.get('seconds',0):>7.1f}s"      # ★ 逐任务耗时（增量效率靠它看）
              + (f"   失败: {r['error'][:70]}" if r.get("error") else ""), flush=True)

    for wave in (todo_nc, todo_cp):
        if not wave:
            continue
        tasks = [(spec.name, int(y), plan[y])
                 for y in sorted({y for _, _, plan in wave for y in plan})
                 for spec, _, plan in wave if y in plan]
        # Reuse equal panel windows together; ordering changes no formula.
        tasks.sort(key=lambda t: (t[1], REGISTRY[t[0]].warmup_days, t[0]))
        if not tasks:
            continue
        done["total"] = done["n"] + len(tasks)
        jobs = max(1, min(safe_jobs(args.jobs), len(tasks)))
        if jobs > 1 and len(tasks) > 1:
            print(f"并行：{jobs} 个进程 · {len(tasks)} 个任务（因子 × 年）"
                  f" · 共享状态已预建 {time.time() - t0:.1f}s")
        else:
            print(f"串行：{len(tasks)} 个任务")
        print("-" * 78)
        if jobs > 1 and len(tasks) > 1:
            results += engine.run_parallel(tasks, jobs, on_done=on_done)
        else:
            for name, year, days in tasks:
                r = engine.run_year(REGISTRY[name], year, days)
                results.append(r)
                on_done(r)

    # ---- 父进程统一收尾：写 manifest（worker 不碰状态文件）
    by_factor: dict[str, list[dict]] = {}
    for r in results:
        by_factor.setdefault(r["factor"], []).append(r)

    failed = []
    total_rows = 0
    # Save parent manifests before child input watermarks are captured.
    for spec, man, plan in todo_nc + todo_cp:
        rs = by_factor.get(spec.name, [])
        errs = [r for r in rs if r.get("error")]
        if errs:
            failed.append((spec.name, errs[0]["error"]))
            continue
        st = engine._finalize(spec, man, plan, rs, time.time() - t0)
        total_rows += st["rows"]

    el = time.time() - t0
    print("-" * 78)
    print(f"完成 {len(todo) - len(failed)}/{len(todo)} 个因子 · 共写 {total_rows:,} 行 · "
          f"用时 {el:.1f}s")
    if failed:
        print("\n以下因子失败（可单独重跑，会自动续传）：")
        for n, e in failed:
            print(f"  - {n}: {e}")
    return 1 if failed else (2 if heavy and not args.allow_multiyear else 0)


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
    print("★ 约束：日频 / 只主板~3000只 / 历史值不得随未来分红变化")
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
        size = store.factor_size(cfg.factors_dir, s.name)
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
    """基础体检：universe 合规、值域与格式；因果性另用 audit-pit 截断复算。"""
    from fea.engine import Engine
    engine = Engine(cfg)
    prefix = tuple(cfg.board_prefixes)
    problems = 0
    baseline = int_to_str(engine.baseline_last_day())

    from fea.spec import QFQ_DATASETS
    print(BANNER)
    print("基础体检不证明无前视；PIT 未执行，请另用 main.py audit-pit 做历史截断复算。")
    problems += _report_contracts(engine)
    print("-" * 96)
    print(f"{'因子':<24}{'行数':>12}{'主板合规':>9}{'≥起点':>7}"
          f"{'值域异常':>9}{'日均截面':>9}  复权口径")
    print("-" * 96)
    for s in all_specs():
        man = Manifest.load(cfg.state_dir, s.name)
        if not man.partition_rows():
            if s.resolved_start(cfg) <= baseline:
                problems += 1
                print(f"✘ {s.name}: 尚未生成，应补齐产物")
            else:
                print(f"{s.name}: 尚未到有效起点，本次不要求产物")
            continue
        missing_years = [y for y in man.partitions if not store.year_path(cfg.factors_dir, s.name, int(y)).exists()]
        if missing_years:
            problems += 1
            print(f"✘ {s.name}: 状态记录中的分区文件缺失 {missing_years}")
        df = store.read_factor(cfg.factors_dir, s.name)
        if df.empty:
            problems += 1
            print(f"✘ {s.name}: 状态记录有行数，但没有可读取的产物")
            continue
        codes = df["stock_code"].astype(str)
        bad_board = int((~codes.str.startswith(prefix)).sum())
        early = int((df["trade_date"] < s.resolved_start(cfg)).sum())
        v = df["value"].to_numpy()
        # ★ 2026-09-15 晚（用户拍板「NaN 还是落盘的好」）后口径变更：
        #   NaN 现在是**合法值**（表示"这天算不出/无效"），产物里整块面板都落盘
        #   → 这里只把 **±inf** 与量级离谱（>1e8）算异常；NaN 单独报成覆盖率。
        bad_val = int(np.isinf(v).sum() + (np.abs(np.nan_to_num(v)) > 1e8).sum())
        nan_ratio = float(np.isnan(v).mean()) if len(v) else 0.0
        # 「日均截面」保持原语义 = 每天**非空**的格子数（不能再用行数 —— 行数现在是面板大小）
        per_day = df[df["value"].notna()].groupby("trade_date").size()
        if nan_ratio > 0.95:       # 与剔除脚本同一条红线：>95% 取不到值 = 口径太稀
            problems += 1
            print(f"  ⚠ {s.name}: NaN 占比 {nan_ratio*100:.1f}%（>95%，口径太稀）", flush=True)
        qfq = [d for d in s.deps if d in QFQ_DATASETS]
        if qfq and not s.allow_qfq:
            problems += 1
        if bad_board or early or bad_val:
            problems += 1
        print(f"{s.name:<24}{len(df):>12,}{('✔' if not bad_board else f'✘{bad_board}'):>9}"
              f"{('✔' if not early else f'✘{early}'):>7}"
              f"{('✔' if not bad_val else f'✘{bad_val}'):>9}"
              f"{per_day.median():>9.0f}  "
              f"{'✘前复权:'+','.join(qfq) if (qfq and not s.allow_qfq) else '✔无前复权输入' if not qfq else '✔已放行'}")
    print("-" * 96)
    print("-" * 96)
    problems += _check_format(cfg)          # ★ 格式统一性（用户点名）
    print("-" * 96)
    print(f"{'基础体检通过 ✔（PIT 请另运行 audit-pit）' if not problems else f'⚠ 发现 {problems} 项问题'}")
    return 0 if not problems else 1


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
    for s in specs:
        try:
            src = inspect.getsource(s.fn)
        except Exception:                                    # noqa: BLE001
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
    from fea.documentation import update_section
    lines = [
        "# 因子字典（featureengineering · 模块②）",
        "",
        "> 本节由 `python main.py docs` 生成，请勿手工编辑本节；其他章节保持不变。",
        "",
        "## ★ 因子开发的三条硬约束（下游是 A 股日横断面回归排序任务）",
        "",
        "1. **全部因子必须日频** —— 对齐到 `(trade_date, stock_code)` 面板后落盘。",
        "2. **固定主板股票池 2115 只** —— 使用 `conf/universe_frozen.tsv`；",
        "   排除创业板 `300/301/302`、科创板 `688/689`、北交所 `832/833/920`。",
        f"3. **输出起点由 `conf/config.yaml` 的 `default_start` 控制** —— 当前 "
        f"**`{cfg.default_start}`**；实际取全局下界与因子可得起点的较晚者，历史输入保留作预热。",
        "",
        "## ★★ PIT 红线：历史因子值不得因未来的分红事件而变化",
        "",
        "前复权（qfq）把整条价格序列按**最新**的复权因子缩放，一旦发生新的分红/送转，",
        "全部历史价格会一起被重算 —— 同一段历史今天算和昨天算结果不同，回测里就是前视偏差。",
        "",
        "- **禁止**把 `stock_kline_adj` / `stock_daily_adj`（前复权）直接喂给因子函数；",
        "  `register()` 会直接拒绝，确需使用必须显式 `allow_qfq=True` 并写明理由。",
        "- 价格**水平**类因子（市值、book-to-market）→ 用**未复权**价 × 当期已披露股本。",
        "- 收益**比率**类因子（动量、波动）→ 用未复权价 + `stock_adj_factor` 截至**当日**的",
        "  累计复权因子还原（等价于「截至当日的后复权」，后复权锚定序列起点，历史值稳定）。",
        "",
        "## 统一输出格式",
        "",
        "```",
        "data/factors/<因子名>/year=YYYY/data.parquet",
        "  trade_date  string   \"2026-09-11\"",
        "  stock_code  string   \"600000.SH\"",
        "  value       float32  原始因子值（可解释、可再标准化）",
        "  rank        float32  当日截面百分位 [0,1]（先 1%/99% winsorize 再 rank）",
        "```",
        "",
        "因子语义：按既定数据可得日，T 日收盘后生成，供 T+1 使用；公告到达时间仍受供应商记录限制。",
        "未来收益标签仅供训练/评价，须等待未来行情成熟，不能作为当日可用特征。",
        "",
        "## 因子清单",
        "",
        f"共 {sum(not s.is_label for s in all_specs())} 个因子、{sum(s.is_label for s in all_specs())} 个标签。",
        "",
        "> ★ 所有因子的**列名 / 列序 / dtype / 目录结构 / 分区方式 / 语义完全一致**，",
        "> **唯一允许的差异是起止日期**（下表最后两列）。",
        "",
        "| 因子 | 类别 | 方向 | 定义 | 公式 | 起点 | 实际起止 | warmup(天) | 依赖 |",
        "|:--|:--|:--|:--|:--|:--|:--|--:|:--|",
    ]
    def esc(t: str) -> str:
        return t.replace("|", "\\|")           # 公式里的 |x| 会破坏 markdown 表格

    for s in all_specs():
        man = Manifest.load(cfg.state_dir, s.name)
        parts = man.partitions or {}
        if parts:
            lo = min(str(v.get("min_date", "9999-12-31")) for v in parts.values())
            hi = max(str(v.get("max_date", "")) for v in parts.values())
            rng = f"{lo} → {hi}"
        else:
            rng = "—（未生成）"
        lines.append(
            f"| `{s.name}` | {s.group} | {'高优' if s.higher_is_better else '低优'} | "
            f"{esc(s.desc)} | `{esc(s.formula)}` | {s.resolved_start(cfg)} | {rng} | "
            f"{s.warmup_days} | {', '.join(s.deps) or '—'} |")
    lines += ["", "## 口径备注（实测坑）", ""]
    for s in all_specs():
        if s.note:
            lines.append(f"- **`{s.name}`**：{s.note}")
    out = update_section(ROOT, "factor-catalog", "\n".join(lines) + "\n")
    print(f"已写入 {out}")
    return 0


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
                   help="★ 明知跨度 ≥3 年仍要在**一个进程**里算完（会吃满内存，不推荐）。"
                        "默认拒绝并指向按年分块的 main.py rebuild")
    _add_common(r)

    lp = sub.add_parser("list", help="列出因子与本地进度")
    _add_common(lp)
    sp2 = sub.add_parser("status", help="已落地因子统计")
    _add_common(sp2)
    cp = sub.add_parser("check", help="基础体检：universe / 值域 / 格式；PIT 用 audit-pit")
    _add_common(cp)
    dp = sub.add_parser("docs", help="更新 README.md 中的因子字典")
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
