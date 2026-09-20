#!/usr/bin/env python
"""模型单元 V16 · 唯一入口（薄入口：只做 argparse + 调度，逻辑在 mx/ 包里）

============================================================================
一、这一版的文件怎么摆（★ 2026-09-20 收敛：顶层只有「代码 + 三个产物桶」）
============================================================================

    V16/
    ├── main.py          入口    —— 本文件，只做 argparse + 调度
    ├── train.sh         启动    —— 折分批并发训练（断点续跑）
    ├── run.py           启动    —— 单折训练（train.sh 就是并发地调它）
    ├── analysis.py      回测    —— 评价 + 策略矩阵 + 回测
    ├── mx/              引擎包（代码）：数据/切分/预处理/评价/回测/产物契约
    ├── model.py         本版配方（代码）：架构与超参的**唯一权威定义**
    ├── model_train/     训练过程产物（逐折权重 / 指标 / 逐折打分 / 折汇总）
    ├── model_pred/      回测相关结果（集成打分 / 榜单 / 回测 JSON / 净值图）
    └── logs/            记录文件（逐折日志 / 冻结配方 recipe.yaml）

    ★ 顶层**只允许**上面这四份入口脚本、`mx/`、`model.py` 与三个产物桶。
      再冒出别的文件或目录就是走样了。
    ★ 冒烟（`--smoke`）与小样本（`--sample`）**嵌在桶内**，不平铺在顶层：
      `model_train/smoke/`、`model_pred/sample/`、`logs/smoke/`。
      分树本身不能省 —— 500 只票训出来的模型绝不能覆盖正式产物，更不能覆盖
      `model_pred/leaderboard.json`（`promote` 读的正是那份榜单）。

============================================================================
二、怎么跑
============================================================================

    PY=/autodl-fs/data/miniconda3/bin/python     # ★ 必须全路径，非登录 shell 的 python 没有 pandas

    $PY main.py doctor          # 环境自检（依赖 / 内存上限 / 并发 / 磁盘 / GPU）
    $PY main.py split V16       # 零成本看 train/valid/test 窗口（不训练）
    $PY main.py freeze V16      # 首次训练前：把引擎哈希与切分口径冻进 logs/recipe.yaml

    bash train.sh               # 训练（续跑：日志里有 EXIT:0 的折自动跳过）
    $PY analysis.py             # 评价 + 策略矩阵 + 回测 → model_pred/leaderboard.json

    ★ **必须在单元目录里跑**（`cd V16` 之后）。从模块根发 `python main.py run V16` 会被引擎的
      防分叉闸门拒绝 —— 那是**有意的**：模块根没有引擎，每个单元用自己那份。
      `main.py` / `run.py` / `analysis.py` 三者都按**本目录**定位引擎，所以仓内跑与打包跑行为一致。

其余子命令：`units`（列单元）· `status`（数据 + 各单元到哪了）· `data`（看面板）·
`eval`（打印上次榜单）· `backtest`（单独重跑回测）· `audit-pit`（前视审计）· `promote`（晋级 best）。

============================================================================
三、数据放哪
============================================================================

单元按三候选解析数据根，**判据是 `meta.json` 存在**（不是「目录存在」）：

    1. $MX_DATA                      ← 显式指定，权威（设了就用它，不回落）
    2. <单元>/trainingdata/          ← 打包发送时把数据放这里
    3. <单元>/../trainingdata/       ← 在模块根下原位运行（现在就是这个）

`$PY main.py status` 会打印它实际找到的是哪一个。
数据由模块根的 `preparingdata.py` 生成，八块产物各自的含义见模块根 `README.md`。
**单元侧只读，绝不写 `trainingdata/`。**

============================================================================
四、迭代：怎么做一个新版本
============================================================================

    cd /autodl-fs/data/model
    cp -r V16 V17
    cd V17
    # 改 model.py 里的 NAME / TITLE / DATA / SPLIT / heads() / STRATEGIES
    $PY main.py freeze V17
    bash train.sh && $PY analysis.py

    ★ **只增不改** —— 改 V16 本身会让它已有的结论失去对照。要改口径就开新版本。
    ★ 头名（`heads()` 里的 `build(cfg, "<名字>")`）就是产物目录名，**两个头不能同名**，
      否则会互相覆盖（V6 首跑踩过）。
    ★ 改 `model.py` 里的 `STRATEGIES` 不必重训 —— 策略是评价层的事，重跑 `analysis.py` 即可。

============================================================================
五、引擎：`V16/mx/` 就是权威副本
============================================================================

**顶层 `mx/` 已经删除，`V16/mx/` 取代它成为引擎的权威副本**（V12/V13/V14 已删）。
改引擎的流程：

    1. 在 `V16/mx/` 里改；
    2. 手工同步到其它单元：把 `V16/mx/` 整个覆盖过去，但**保留**目标单元的 `mx/conf.yaml`
       （每个单元的 conf.yaml 里 units/best 两行可能不同）；
    3. 各单元重新 `$PY main.py freeze <单元>`。

    ★ `V16/mx/` 里没有 `prepared.py` / `upstream.py` —— 那两个是**构建期**模块，
      只在 `preparingdata.py` 里（它现在把它们内联在一个文件里了）。单元不该知道上游在哪。

### 自包含体检（改完引擎按这五条自查）

    1. `<单元>/mx/` 存在且 `<单元>/main.py` 存在；
    2. `<单元>/mx/prepared.py` 与 `<单元>/mx/upstream.py` **不存在**；
    3. `<单元>/logs/recipe.yaml` 存在（freeze 过）★ 此条原为「requirements.txt 存在」，
       2026-09-20 依赖清单并入本 docstring 后，改用冻结配方作判据；
    4. `<单元>/mx/**/*.py` 里**没有真实的上游依赖**（AST 级判据）：`import fea…`、
       属性访问 `.factors_root/.factors_dir/.factors_state`、字符串常量里出现
       `featureengineering` / `datadownload`。
       ★ **注释与 docstring 里提到不算违规** —— 本工程明文规定「注释讲为什么」，
       一个「注释提一句就报红」的判据最后只会被人绕开；
    5. `<单元>/train.sh` 里没有 `ROOT=".."`（应为 `ROOT="."`）。

============================================================================
六、依赖（原 `requirements.txt` 的内容，已并入此处）
============================================================================

    装法（**必须 sudo + 全路径**；不要装到 `~/.local` —— 不在共享盘上，换机器就丢）：

        sudo /autodl-fs/data/miniconda3/bin/pip install \
             scikit-learn==1.9.1 scipy==1.18.1 lightgbm==4.7.0 xgboost==3.4.1 \
             matplotlib==3.11.2 torch==2.14.0

    ★ 这些版本都实测有 cp314 / py3-none 轮子。
    ★ torch 用 PyPI 默认 wheel（自带 cu12 运行时）：**CPU 机器直接能跑，换到有卡机器
      同一份环境吃 GPU**。在 GPU 机器上先跑 `main.py doctor` 复核 `torch.version.cuda`
      与驱动是否匹配；不匹配时按显卡驱动的最高 CUDA 版本换 wheel，例如：

        sudo /autodl-fs/data/miniconda3/bin/pip install torch==2.14.0 \
             --index-url https://download.pytorch.org/whl/cu126

    ★ 已随环境自带、无需再装：numpy 2.5.3 · pandas 3.0.5 · pyarrow 25.0.1 · PyYAML 6.0.3
    ★ 未纳入的两项：joblib · tqdm —— 全库无 import，装了也用不上。

============================================================================
七、环境与纪律
============================================================================

    ★ 一律用 `/autodl-fs/data/miniconda3/bin/python`。
    ★ `MX_THREADS` 压 BLAS/OMP 线程数（默认 8）—— 容器预置了 `OMP_NUM_THREADS=32`，
      不压制会让多折并发互相踩踏（模块② 实测 load 冲到 60）。
      用**硬赋值**而不是 `setdefault`，否则会被容器预置值顶掉。
    ★ 三条铁律（详见 `mx/conf.yaml` 抬头）：
        1) 一切预处理只用**当日截面**统计量 —— 禁止全样本 mean/std/quantile（那是未来信息）
        2) 训练/验证/测试**按时间切分**，且标签重叠 h 天 ⇒ 边界两侧各丢 h 天（purge + embargo）
        3) 数据只从 `trainingdata/` 读；本单元**绝不写**上游与 `trainingdata`
"""
from __future__ import annotations

# ★★ 必须在 import numpy 之前压线程数：BLAS/OpenMP 默认吃满宿主机核数，
#   与 LightGBM 的 n_jobs、以及"多折并发"互相踩踏（模块② 实测 load 冲到 60）。
#   固定下来还有个好处：浮点求和顺序稳定 → 同种子两次训练逐位一致（可复现判据）。
#   ★ 注意用**硬赋值**：容器预置了 OMP_NUM_THREADS=32，`setdefault` 会被它顶掉（实测踩到）。
#   要改就设环境变量 `MX_THREADS`。
import os as _os
_MX_THREADS = _os.environ.get("MX_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_v] = _MX_THREADS

import argparse                    # noqa: E402
import subprocess                  # noqa: E402
import sys                         # noqa: E402
from pathlib import Path           # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mx import config as C         # noqa: E402
from mx import evaluate as ev      # noqa: E402
# ★ `panel_io` 而不是 `prepared`：本文件要在**自包含单元**里也能 import 成功，
#   而 `prepared` 带着 `fea` 依赖（单元包里没有它）。用到的 `load_meta`/`root_of`
#   本来就住在 `panel_io`。构建期的那几个子命令在单元里会给出明确的不可用提示。
from mx import panel_io as P       # noqa: E402
from mx import state               # noqa: E402
from mx import unit as U           # noqa: E402


# ================================================================ doctor
def cmd_doctor(args, cfg) -> int:
    """环境自检 —— 回答"这台机器能跑到什么程度"（CPU 机器 / GPU 机器都跑它）。"""
    print("=" * 88)
    print("  模块③ 模型工程 · 环境体检")
    print("=" * 88)
    print(f"  Python      : {sys.version.split()[0]}  ({sys.executable})")
    pkgs = ["numpy", "pandas", "pyarrow", "yaml", "scipy", "sklearn", "joblib",
            "lightgbm", "xgboost", "matplotlib", "torch"]
    print("  关键依赖    :")
    for p in pkgs:
        try:
            m = __import__(p)
            v = getattr(m, "__version__", "?")
            extra = ""
            if p == "torch":
                extra = (f"  · CUDA 构建 {getattr(m.version, 'cuda', None)}"
                         f"  · 可用 GPU {'✔ ' + str(m.cuda.device_count()) + ' 张' if m.cuda.is_available() else '✘（CPU 模式）'}")
            print(f"      {p:12} ✔ {v}{extra}")
        except Exception as exc:          # noqa: BLE001
            print(f"      {p:12} ✘ {type(exc).__name__}")
    try:
        import torch
        if not torch.cuda.is_available() and torch.version.cuda:
            print("  ⚠️ torch 是 CUDA 构建但当前检测不到 GPU —— 在 GPU 机器上应自动可用")
    except Exception:                     # noqa: BLE001
        pass

    # ★ 可用核数必须取 min(亲和集, cgroup CPU 配额)：本机亲和集放行宿主 224 核，
    #   但 cgroup 只给 25 核配额 —— 按亲和集算会把线程数开到 89（见 mx/unit.py:avail_cores）
    avail = U.avail_cores()
    aff = len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") else (_os.cpu_count() or 0)
    conc = U.default_conc()
    th = U.default_threads(conc)
    per = U.panel_fold_gb()
    print(f"  CPU 核数    : 容器可用 {avail}（亲和集 {aff} · 宿主 {_os.cpu_count()}）· "
          f"BLAS/OMP 线程已设为 {_os.environ.get('OMP_NUM_THREADS')}")
    print(f"  内存上限    : {U.cgroup_mem_gb():.1f} GiB（cgroup；★ 换机器会变，跑前现读）")
    print(f"  单折预算    : ≈ {per:.1f} GiB（按 trainingdata 的 X 体积估：面板 + 每头一份预处理副本）")
    print(f"  并发规则    : 折并发 = 可用内存×0.8/单折预算 → **{conc} 折** · "
          f"单折线程 = 核数×0.8/并发 → **{th}**")
    print("  ★ 全历史面板（X≈11 GB）单折约 35~45 GB ⇒ 会**自动降到 1 折**；"
          "想提速请先按年分块，别手改并发")
    print("  ★ 定 n_jobs / 并行度请按**容器可用核数**与上面这条规则，不要按宿主核数拍")
    print(f"  训练设备    : {cfg.device}（resolve_device：有 CUDA 且 device=auto 即用 GPU）")
    # ★ `factors_dir` 在**自包含单元**里是 None（交付版的 conf.yaml 没有这条路径）——
    #   打印"不适用"而不是让 `None.exists()` 把 `doctor` 打崩：`doctor` 恰恰是
    #   换机器/交付后第一个要跑的命令，它自己崩掉最说不过去。
    for p, label in ((cfg.factors_dir, "上游因子产物"), (cfg.trainingdata, "初加工产物"),
                     (cfg.units_root, "单元目录")):
        if p is None:
            print(f"  {label:12}: — 不适用（自包含运行，本包不含上游路径配置）")
            continue
        print(f"  {label:12}: {'✔' if p.exists() else '✘'} {p}")
    st = _os.statvfs(str(ROOT))
    print(f"  磁盘可用    : {st.f_bavail * st.f_frsize / 2**30:.0f} GiB")
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        print(f"  nvidia-smi  : {r.stdout.strip() or r.stderr.strip()[:80] or '无输出'}")
    except Exception:                     # noqa: BLE001
        print("  nvidia-smi  : 不存在（当前确实没有 GPU）")
    print("=" * 88)
    return 0


# ================================================================ units / status
def cmd_units(args, cfg) -> int:
    names = U.list_units(cfg)
    if not names:
        print(f"  还没有任何单元（{cfg.units_root}/<名称>/model.py）")
        return 1
    print("=" * 104)
    print(f"  模型单元（{cfg.units_root}）")
    print("=" * 104)
    P_ = ev.pad
    print("  " + P_("单元", 8) + P_("标题", 30) + P_("折", 5, True) + P_("配方", 7, True)
          + P_("已训折", 9, True) + P_("打分", 7, True) + "  切分 / 说明")
    for n in names:
        try:
            u = U.load_unit(cfg, n, log=lambda *a: None)
        except SystemExit as exc:
            print(f"  {ev.pad(n, 8)}✘ 载入失败：{exc}")
            continue
        folds = state.unit_train_dir(cfg, n)
        done = len(list(folds.glob("fold*/**/*.pkl"))) if folds.exists() else 0
        n_fold_done = len({p.parts[-3] for p in folds.glob("fold*/**/*.pkl")}) if done else 0
        pd_ = state.unit_pred_dir(cfg, n)
        preds = [d for d in pd_.glob("score__*") if d.is_dir()] if pd_.exists() else []
        frozen = "✔" if u.recipe else "✘"
        sp = u.split_cfg(cfg)
        if sp.get("mode") == "date":
            cut = ("无 test·全量" if not sp.get("test_start")
                   else f"test {sp['test_start']}~{sp['test_end']}")
        else:
            cut = f"{sp.get('mode')} 口径"
        note = ""
        if u.archived:
            note = "🗄 已冻结（历史口径，不与现行口径比）"
        elif u.is_production:
            note = "🚀 实战单元（无 test，全量数据训练）"
        elif u.drift and u.drift.get("engine"):
            note = "⚠️ 引擎已漂移"
        print("  " + P_(n, 8) + P_(u.title[:28], 30) + P_(len(u.seeds), 5, True)
              + P_(frozen, 7, True) + P_(n_fold_done, 9, True) + P_(len(preds), 7, True)
              + f"  {cut}  {note}")
    print("-" * 104)
    print("  ▸ 单元 = 独立可训练单元（model.py/run.py/train.sh/analysis.py）；"
          "`best` 是晋级后的实战版（无 test）")
    return 0


def cmd_status(args, cfg) -> int:
    # ★ 上游那一段在**自包含单元**里必然不可用（没有 fea、conf 里也没有 factors_*）。
    #   整个包进 try：取不到就一行说明跳过，而不是让 `status` 直接崩 ——
    #   单元里 `status` 仍然要能回答"我的数据在哪、训到哪了"这些**本地**问题。
    try:
        from mx import upstream
        names = upstream.list_factor_names(cfg)
        print(f"  上游因子产物：{len(names)} 个（{cfg.factors_dir}）")
        try:
            print(f"  上游最新因子日：{upstream.last_upstream_day(cfg)}")
        except Exception as exc:            # noqa: BLE001
            print(f"  上游最新因子日：取不到（{type(exc).__name__}）")
    except Exception as exc:                # noqa: BLE001
        print(f"  上游：不可用（{type(exc).__name__}）—— 自包含运行，或没装 fea。"
              f"这不影响训练/评价/回测。")
    for tag, root in (("全量", cfg.trainingdata), ("小样本", cfg.trainingdata_sample)):
        m = P.load_meta(root)
        if not m:
            print(f"  初加工产物（{tag}）：尚未生成 —— `python preparingdata.py`")
            continue
        ax = m.get("axis") or {}
        print(f"  初加工产物（{tag}）：{len(m.get('built_years') or [])} 年 · "
              f"{ax.get('start')} ~ {ax.get('end')} · {ax.get('n_days')} 天 × "
              f"{ax.get('n_codes')} 只 · 特征 {len(m['columns']['features'])} · "
              f"指纹 {m.get('panel_digest')}（建于 {m.get('built_at')}）")
    for n in U.list_units(cfg):
        lb = state.load_json(state.unit_leaderboard_path(cfg, n), {})
        ls = state.load_json(state.unit_pred_dir(cfg, n) / "analysis_summary.json", {})
        lb = lb or ls
        n_folds = len(lb.get("folds") or [])
        note = f"榜单 {lb.get('n_rows','?')} 行 · {lb.get('finished_at','—')}" if lb else "尚未分析"
        print(f"    {n:10} 折 {n_folds} · {note}")
    return 0


# ================================================================ 初加工 / 切分推演 / 面板
def cmd_prepare(argv: list[str]) -> int:
    """把参数**原样**转给 `preparingdata.py`（那里是初加工的唯一参数定义处）。

    ★ 不走 argparse 的子命令：`nargs=REMAINDER` 对**前置选项**（`prepare --check`）不生效，
      会被主解析器当成自己的选项而报错。直接透传最稳、也少一处参数重复定义。

    ★ 初加工是**构建期**职责，只存在于模块根。自包含单元里没有 `preparingdata.py`
      （它要读 230 个上游因子），所以这里给一句明确的话，而不是 `FileNotFoundError`。
    """
    script = ROOT / "preparingdata.py"
    if not script.exists():
        # ★ 本函数收不到 cfg（走的是 `cmd_prepare(argv[1:])` 这条透传路径），自己load 一次
        td = C.load().trainingdata
        print(f"✘ 本目录没有 `preparingdata.py` —— 初加工是**构建期**职责，只在模块根提供。\n"
              f"    （它是把上游因子加工成 `trainingdata/` 的写入口；单元只**消费**产物。）\n"
              f"    本目录是自包含运行形态，数据请放在：\n"
              f"      {td}\n"
              f"    或在别处建好后用 `MX_DATA=<路径>` 指过来。")
        return 1
    cmd = [sys.executable, str(script), *argv]
    print(f"▸ {' '.join(cmd)}")
    return subprocess.call(cmd)


def cmd_split(args, cfg) -> int:
    """★ 零成本推演某单元的切分窗口（**不训练**）。

    只读**一列**特征 + 标签 + 股票池就能把 train/valid/test 的日期边界算出来，
    用来回答"这一版的隔断对不对、purge 有没有按标签的 h 让开"。
    """
    from mx import data as D
    from mx import dataset as DS
    u = U.load_unit(cfg, args.unit)
    sp = u.split_cfg(cfg)
    labels = [args.label] if args.label else u.my_labels(cfg)
    print("=" * 88)
    print(f"  切分推演 · 单元 {u.name}（{u.title}）")
    print("=" * 88)
    print(f"  生效口径：mode={sp.get('mode')} · train_end={sp.get('train_end') or '数据末日'} · "
          f"test={sp.get('test_start') or '无'} ~ {sp.get('test_end') or '无'} · "
          f"valid_ratio={sp.get('valid_ratio')} · embargo="
          f"{sp.get('embargo_days') if sp.get('embargo_days') is not None else '按标签 h'}"
          + ("　🚀 实战口径：无 test、train+valid 用全部数据" if u.is_production else ""))
    days = D.available_dates(cfg)
    feats = P.load_meta(P.root_of(cfg, args.sample)).get("columns", {}).get("features") or []
    panel = D.load_for_unit(cfg, u, features=feats[:1], sample=args.sample, log=lambda *a: None)
    print(f"  面板：{len(panel.dates)} 天（{panel.dates[0]} ~ {panel.dates[-1]}）× "
          f"{len(panel.codes)} 只 · 样本口径 = 池内 ∩ 标签非空（推演用单列特征近似）")
    rc = 0
    for lb in labels:
        if lb not in panel.y:
            print(f"  ⊘ {lb}：产物里没有这个标签")
            continue
        try:
            s = DS.make_splits(panel, lb, cfg, split=sp, log=print)[0]
        except SystemExit as exc:
            print(f"  ⊘ {lb}：{exc}")
            rc = 1
            continue
        d = s.as_dict()
        print(f"     可用样本：train {d['n_train']:,} · valid {d['n_valid']:,} · "
              f"test {d['n_test']:,}（purge/embargo {d['embargo']} 天）")
    return rc


def cmd_data(args, cfg) -> int:
    """看面板（默认**全量**加载，全历史约 11 GB 内存；只想快看就用 --sample）。"""
    from mx import data as D
    from mx import preprocess
    u = U.load_unit(cfg, args.unit) if args.unit else None
    panel = (D.load_for_unit(cfg, u, sample=args.sample, features=args.features or None)
             if u else D.load(cfg, sample=args.sample, features=args.features or None))
    print(f"  产物目录：{panel.meta['source']}（建于 {panel.meta['built_at']}）")
    print(f"  面板指纹：{panel.meta['panel_digest']} · 本次选择指纹：{panel.meta['digest']}")
    print(f"  形状：{panel.n_rows:,} 行 × {len(panel.feats)} 特征 · "
          f"{len(panel.dates)} 天 × {len(panel.codes)} 只 · 池内 {int(panel.universe.sum()):,} 格")
    print(f"  特征：{', '.join(panel.feats[:6])} …（共 {len(panel.feats)}）")
    if not args.no_screen:
        # 模块② 的单因子 IC 先验（对照用：模型该跑赢"最强单因子"才叫有用）
        preprocess.screen_report(cfg, panel.feats, log=print)
    return 0


# ================================================================ freeze / run / analysis
def cmd_freeze(args, cfg) -> int:
    U.freeze(cfg, args.unit, scan_panel=not args.no_panel, log=print)
    return 0


def cmd_run(args, cfg) -> int:
    """训一折（--fold）或全部折（--all）。train.sh 是并发的等价入口。"""
    from mx import train as T
    u = U.load_unit(cfg, args.unit)
    folds = U.folds(u) if args.all or args.fold is None else [int(args.fold)]
    rc = 0
    for f in folds:
        try:
            s = T.train_fold(cfg, u, f, smoke=args.smoke, sample=args.sample, log=print)
            print(f"  ✔ 折 {f}：{s['n_runs']} 个 (头 × 标签) · {s['seconds']}s")
        except SystemExit as exc:
            print(f"  ⊘ 折 {f} 失败：{exc}")
            rc = 1
    if len(folds) > 1:
        print(f"\n  下一步： python main.py analysis {u.name}")
    return rc


def cmd_analysis(args, cfg) -> int:
    from mx import analyze, backtest
    u = U.load_unit(cfg, args.unit)
    s = analyze.run(cfg, u, smoke=args.smoke, sample=args.sample, top_n=args.top,
                    do_backtest=not args.no_backtest, log=print,
                    label=getattr(args, "label", None))
    print(analyze.render(s))
    if s.get("strategy_matrix"):
        print()
        print(analyze.strategy_table(s["strategy_matrix"]))
    for k, bt in (s.get("backtest") or {}).items():
        print(f"\n  【回测 · {k} · 头={bt.get('head')}】")
        print(backtest.render(bt))
    return 0


def cmd_promote(args, cfg) -> int:
    U.promote(cfg, args.unit, label=args.label, log=print)
    return 0


# ================================================================ audit-pit / eval / backtest
def cmd_audit_pit(args, cfg) -> int:
    from mx import predict as P_
    u = U.load_unit(cfg, args.unit)
    head = args.head or U.head_names_safe(cfg, u)[0]
    label = args.label or u.primary_label or u.my_labels(cfg)[0]
    folds = [int(args.fold)] if args.fold else U.folds(u)
    rc = 0
    for f in folds:
        print(f"  ── 折 {f} · 头 {head} · 标签 {label}")
        rc |= P_.audit_pit_unit(cfg, u, f, head, label, cut=args.cut, smoke=args.smoke,
                                sample=args.sample, log=print)
    return rc


def cmd_eval(args, cfg) -> int:
    """打印某单元最近一次分析的榜单（不重算）。"""
    from mx import analyze
    u = U.load_unit(cfg, args.unit)
    s = state.load_json(state.unit_leaderboard_path(cfg, u, args.smoke, args.sample), {})
    if not s:
        print(f"  还没有分析结果 —— 先跑 `main.py analysis {u.name}`")
        return 1
    print(analyze.render(s))
    if s.get("strategy_matrix"):
        print()
        print(analyze.strategy_table(s["strategy_matrix"]))
    return 0


def cmd_backtest(args, cfg) -> int:
    from mx import analyze, backtest, store
    u = U.load_unit(cfg, args.unit)
    label = args.label or u.primary_label
    p = state.unit_pred_dir(cfg, u) / f"score__{u.name}__{label}"
    if not p.exists():
        print(f"  没有 {p} —— 先跑 `main.py analysis {u.name}`")
        return 1
    # 窗口与榜单一致：研究单元 = test 窗，实战单元 = 最后一个验证窗
    from mx import data as D
    from mx import analyze as A
    panel = D.load_for_unit(cfg, u, log=lambda *_: None)
    lo, hi, _win = A._eval_window_dates(cfg, u, panel, label, False)   # noqa: SLF001
    res = backtest.run(cfg, u.name, label, log=print, top_n=args.top, hold=args.hold,
                       dates_lo=lo, dates_hi=hi, frame=store.read_frame(p), panel=panel)
    print(backtest.render(res))
    state.save_json(state.unit_pred_dir(cfg, u) / f"backtest__{u.name}__{label}.json", res)
    return 0


# ================================================================ main
def _add_sample_flag(p):
    p.add_argument("--sample", action="store_true", help="用 trainingdata/sample/ 小样本（快）")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py", description="模块③ 模型工程（版本化单元）",
        epilog="初加工走 `main.py prepare <参数>`（参数原样转给 preparingdata.py，"
               "例：`main.py prepare --full`）；它不在下面的子命令列表里。")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("doctor", help="环境自检（CPU/内存/磁盘/GPU/torch）").set_defaults(fn=cmd_doctor)
    sub.add_parser("units", help="列出所有模型单元").set_defaults(fn=cmd_units)
    sub.add_parser("list", help="同 units").set_defaults(fn=cmd_units)
    sub.add_parser("status", help="上游 + 初加工产物 + 各单元状态").set_defaults(fn=cmd_status)

    sp = sub.add_parser("split", help="★ 零成本推演某单元的训练/验证/测试窗口（不训练）")
    sp.add_argument("unit")
    sp.add_argument("--label", default=None)
    _add_sample_flag(sp)
    sp.set_defaults(fn=cmd_split)

    dt = sub.add_parser("data", help="看面板清单/覆盖率/可用样本数")
    dt.add_argument("unit", nargs="?", default=None)
    dt.add_argument("--features", nargs="*", default=None, help="只加载这些特征（默认全部）")
    dt.add_argument("--no-screen", action="store_true", help="跳过模块② 单因子 IC 先验对照")
    _add_sample_flag(dt)
    dt.set_defaults(fn=cmd_data)

    f = sub.add_parser("freeze", help="冻结单元配方（引擎哈希/上游指纹/生效切分）")
    f.add_argument("unit")
    f.add_argument("--no-panel", action="store_true", help="跳过面板扫描")
    f.set_defaults(fn=cmd_freeze)

    r = sub.add_parser("run", help="训练某单元的折（默认全部折）")
    r.add_argument("unit")
    r.add_argument("--fold", type=int, default=None, help="只训这一折")
    r.add_argument("--all", action="store_true", help="训全部折")
    r.add_argument("--smoke", action="store_true", help="冒烟：小模型 + 放宽最小块天数")
    _add_sample_flag(r)
    r.set_defaults(fn=cmd_run)

    t = sub.add_parser("train", help="同 run")
    t.add_argument("unit")
    t.add_argument("--fold", type=int, default=None)
    t.add_argument("--all", action="store_true")
    t.add_argument("--smoke", action="store_true")
    _add_sample_flag(t)
    t.set_defaults(fn=cmd_run)

    an = sub.add_parser("analysis", help="装配 + 评价 + 榜单 + 回测 + 落盘单元打分")
    an.add_argument("unit")
    an.add_argument("--smoke", action="store_true")
    an.add_argument("--no-backtest", action="store_true")
    an.add_argument("--top", type=int, default=None)
    # ★ 主腿标签：策略矩阵与对照回测都只跑这一条腿。默认取单元声明的 PRIMARY_LABEL。
    #   为什么要能改：本轮实测**只有 20d 期限的可执行 Σtop1 为正**（README §8.6），
    #   而多数单元声明的 PRIMARY_LABEL 是 5d —— 想验证"是不是换条腿就能赚钱"就得能指定。
    an.add_argument("--label", default=None, help="主腿标签（默认单元声明的 PRIMARY_LABEL）")
    _add_sample_flag(an)
    an.set_defaults(fn=cmd_analysis)

    pm = sub.add_parser("promote", help="把某单元晋级为 best（实战版：自动去掉 test 窗）")
    pm.add_argument("unit")
    pm.add_argument("--label", default=None, help="主腿标签（默认单元声明的 PRIMARY_LABEL）")
    pm.set_defaults(fn=cmd_promote)

    a = sub.add_parser("audit-pit", help="前视审计：截断复算（打分不许用未来数据）")
    a.add_argument("unit")
    a.add_argument("--fold", type=int, default=None)
    a.add_argument("--head", default=None)
    a.add_argument("--label", default=None)
    a.add_argument("--cut", default=None, help="截断日（默认倒数第二个交易日）")
    a.add_argument("--smoke", action="store_true")
    _add_sample_flag(a)
    a.set_defaults(fn=cmd_audit_pit)

    e = sub.add_parser("eval", help="打印单元最近一次分析的榜单")
    e.add_argument("unit")
    e.add_argument("--smoke", action="store_true")
    _add_sample_flag(e)
    e.set_defaults(fn=cmd_eval)

    b = sub.add_parser("backtest", help="对单元打分跑组合回测")
    b.add_argument("unit")
    b.add_argument("--label", default=None)
    b.add_argument("--top", type=int, default=None)
    b.add_argument("--hold", type=int, default=None)
    b.set_defaults(fn=cmd_backtest)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["units"] + argv          # 裸命令 = 看看有哪些单元（比默认开跑更安全）
    if argv[0] == "prepare":             # ★ 原样透传给 preparingdata.py（见 cmd_prepare）
        return cmd_prepare(argv[1:])
    args = build_parser().parse_args(argv)
    cfg = C.load()
    cfg.ensure_dirs()
    fn = getattr(args, "fn", cmd_units)
    return fn(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
