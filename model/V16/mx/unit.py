"""模型单元（`V{N}`）—— 与参考工程 `Model/V{N}` 同构的「独立模型单元」约定。

## 单元是什么

一个目录 `Model/V{N}/`，自带**从零可训练**的全部能力，四个文件的使命固定、不混用：

| 文件 | 使命 |
|:--|:--|
| `model.py` | 本版**架构 / 损失 / 配方**的唯一权威定义 —— 唯一随版本变的文件 |
| `run.py` | 训练入口，**每折一进程**（折 = 一次「种子 × 划分」的独立训练） |
| `train.sh` | 折分批并发调度（并发 = cgroup 内存 GB / 20；断点续跑） |
| `analysis.py` | 对本单元**自训** checkpoint 推演 → 打分 → 榜单 → 回测 |

外加 `logs/recipe.yaml`（冻结配方：引擎哈希 / 面板 digest / 上游台账哈希 / 种子计划），
以及三个产物桶：`model_train/`（训练产物）· `model_pred/`（回测结果）· `logs/`（记录）——
桶内一律不入库，只有 `logs/recipe.yaml` 与 `logs/recipe.lock.json` 例外。

## 三条纪律（来自参考工程的血泪教训）

1. **研究可跨版本，落地必须成单元**：跨版本集成/网格搜索都可以做，但胜出方案必须
   重构成新的独立 `V{N}`；只写一个「读旧版产物再拼」的脚本 ≠ 新版本。
2. **引擎共享 + 哈希锁定**：`mx/` 不随版本复制（避免 12 模块 × N 版本爆炸），
   但单元的 `logs/recipe.yaml` 里锁死 `engine_hash`；引擎漂移会被检出（train 警告、promote 拒绝）。
3. **单种子纪录 = 种子中奖**：任何「新纪录」必须换种子复刻过才认（参考工程 V29 → V29b
   的反面教材：同配方换种子 +73.7% → +2.7%）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from . import state
from .config import BEST, ROOT, Cfg

PER_FOLD_MEM_GB = 20.0      # 每折峰值内存预算的**兜底值**（估算不出面板时用）
REQUIRED = ("NAME", "TITLE", "SEEDS", "heads")


#: 折峰值 ≈ MULT × X（面板本身 + 一份预处理副本 + 训练段切片 + 库内部的临时量）。
#: ★ 这个数是**实测**出来的，不是推的：2026-09-18 全量跑 V3 折 1
#:   （2018 窗 · X = 7,368,660 × 223 × 4 B = 6.57 GiB · 头 = ridge_w + gbdt_w），
#:   cgroup 峰值 32.49 GiB ⇒ 倍率 4.94。峰值出现在 **LightGBM** 那段
#:   （它要把训练段转成自己的分箱 Dataset，约 1 B/特征/行，外加直方图与多线程缓冲）；
#:   纯线性头的那段远低于此。
#:   ★ 换成"没有树头"的单元时会**高估**（少开并发，只是慢）；
#:     换成更大的树 / 更多树头时会**低估** —— 所以任何一次改配方后都要重新实测一次，
#:     方法见 README 的内存纪律（`/tmp/memprobe.sh` 那套读 cgroup 高水位的做法）。
_FOLD_MULT = 4.9


def panel_fold_gb(unit=None) -> float:
    """单折峰值内存预算（GB）—— 按**本单元实际会加载的那段面板**推，不用拍死的常数。

    ★ 为什么必须动态算：折峰值 ≈ `_FOLD_MULT` × X，而 X 随"数据窗"变化巨大 ——
        · 2026 单年（V1 口径）：X 约 0.5 GB → 单折约 1.3 GB
        · 2018 起（V2b/V3 口径）：X 约 6.6 GB → 单折约 17 GB
        · 全历史 2012 起        ：X 约 11.2 GB → 单折约 **29 GB**
      拿一个常数套所有口径，要么浪费并发、要么让 3 折直接把容器打爆
      （90 GiB 上限、无 swap、OOM-Kill 无警告）。

    `unit` 给定时按其 `DATA` 声明的窗口**只累加落在窗口内的年份分区**；
      不给则按全历史 —— 这是"报忧不报喜"的方向，宁可低估并发。
    """
    try:
        # ★ 走 `panel_io` 而不是 `prepared`：这里只需要读 meta，而 `prepared`
        #   带着 `fea` 依赖 —— 自包含单元里没有它（`train.sh` 每次都要调本函数
        #   算并发，所以这是"单元跑不起来"的第一个现场）。
        from . import panel_io
        from .config import load as _load
        cfg = _load()
        m = panel_io.load_meta(cfg.trainingdata)
        d = (unit.data_cfg(cfg) if unit is not None else {}) or {}
        start, end = d.get("start"), d.get("end")
        rows = 0
        for v in (m.get("years") or {}).values():
            if start and str(v.get("end")) < str(start)[:10]:
                continue
            if end and str(v.get("start")) > str(end)[:10]:
                continue
            rows += int(v.get("rows", 0))
        nf = len(d.get("features") or (m.get("columns") or {}).get("features") or [])
        if rows and nf:
            return max(1.0, _FOLD_MULT * rows * nf * 4 / 2 ** 30)
    except Exception:                       # noqa: BLE001
        pass
    return PER_FOLD_MEM_GB


# ------------------------------------------------------------------ 哈希 / 环境
def _hash_files(paths: list[Path], base: Path | None = None) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        rel = p.relative_to(base).as_posix() if base else p.name
        h.update(rel.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def engine_hash(root: Path = ROOT) -> str:
    """引擎指纹 = `mx/**/*.py` + `main.py`（版本配方里锁的就是它）。"""
    files = [p for p in (root / "mx").rglob("*.py") if "__pycache__" not in p.parts]
    files.append(root / "main.py")
    return _hash_files(files, root)[:16]


def unit_hash(unit_dir: Path) -> str:
    """单元指纹 = 该目录下的 *.py + `logs/recipe.yaml`（不含产物）。"""
    files = [p for p in unit_dir.glob("*.py")]
    rp = unit_dir / "logs" / "recipe.yaml"
    if rp.exists():
        files.append(rp)
    return _hash_files(files, unit_dir)[:16]


def cgroup_mem_gb() -> float:
    """读 cgroup 内存上限（容器里 `os.cpu_count()` 会骗人，内存同理要读 cgroup）。"""
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v.isdigit():
                gb = int(v) / 1024 ** 3
                if gb < 1024:                     # 排除 "max" 与离谱值
                    return round(gb, 1)
        except OSError:
            continue
    return 0.0


def default_conc(per_fold_gb: float | None = None, cap: int = 8, unit=None) -> int:
    """并发折数 = 可用内存 / 单折预算（留 20% 余量；读不到 cgroup 时保守取 1）。

    单折预算按**本单元数据窗**下的面板体积算（见 `panel_fold_gb`）。
    ★ 这是估算，不是实测 —— 真跑之前先单折实测一次峰值内存（见 README 的内存纪律），
      估算值只用来定一个不会撞墙的起点。
    """
    gb = cgroup_mem_gb()
    if not gb:
        return 1
    per = float(per_fold_gb) if per_fold_gb else panel_fold_gb(unit)
    return max(1, min(cap, int(gb * 0.8 / per)))


def avail_cores() -> int:
    """**容器真正能用的核数** = min(CPU 亲和集, cgroup CPU 配额)。

    ★ 为什么不能只用 `len(os.sched_getaffinity(0))`（2026-09-18 换机器后实测踩到）：
      这台容器的亲和集是宿主机的 **224** 核全放行，但 cgroup 的 `cpu.max` 只给
      `2500000 100000` = **25 核的配额**。按亲和集算 ⇒ `default_threads(2)` 得出 **89**，
      让 LightGBM/BLAS 在 25 核的额度上开 89 个线程 —— 只会互相踩踏，越跑越慢。
      （`nproc` 反而能报对 25，因为它读的就是 cgroup 配额。）
    """
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 4
    try:                                       # cgroup v2: "2500000 100000" = 25 核
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cores = min(cores, max(1, int(float(quota) / float(period))))
    except (OSError, ValueError):
        try:                                   # cgroup v1
            q = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            p = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0:
                cores = min(cores, max(1, int(q / p)))
        except (OSError, ValueError):
            pass
    return max(1, cores)


def default_threads(conc: int | None = None) -> int:
    """单折线程数：可用核数 / 并发折数，留 20% 余量（核数取 `avail_cores`，不是亲和集）。"""
    cores = avail_cores()
    conc = conc or default_conc()
    return max(1, int(cores * 0.8 / conc))


# ------------------------------------------------------------------ 单元装载
@dataclass
class Unit:
    name: str
    dir: Path
    mod: object
    recipe: dict
    drift: dict = None            # {"engine": bool, "unit": bool, ...}

    @property
    def title(self) -> str:
        return str(getattr(self.mod, "TITLE", ""))

    @property
    def seeds(self) -> list[int]:
        return [int(s) for s in self.mod.SEEDS]

    @property
    def labels(self) -> list[str] | None:
        v = getattr(self.mod, "LABELS", None)
        return [str(x) for x in v] if v else None

    def my_labels(self, cfg: Cfg) -> list[str]:
        return self.labels or cfg.labels

    @property
    def primary_label(self) -> str:
        return str(getattr(self.mod, "PRIMARY_LABEL", ""))

    @property
    def archived(self) -> bool:
        """历史单元（口径已过期、不再参与新口径对比）：`ARCHIVED = True`。"""
        return bool(getattr(self.mod, "ARCHIVED", False))

    @property
    def is_production(self) -> bool:
        """实战单元：`best` 本身，或单元里声明 `PRODUCTION = True`。"""
        return self.name == BEST or bool(getattr(self.mod, "PRODUCTION", False))

    def split_cfg(self, cfg: Cfg) -> dict:
        """本单元**生效**的切分配置 = 全局 split ← 单元 `SPLIT` ← 生产覆盖。

        ★ 生产覆盖（无 test、train+valid 用全部数据）放在**引擎**里做，是为了让
          `main.py promote V{N} → best` 保持"纯文件复制"：不必改写单元代码里的 SPLIT。
          否则 best 会把 V{N} 的 test 窗一起继承过去 —— 那就把研究口径搬进实战了。
        """
        sp = dict(cfg.split_cfg)
        sp.update(getattr(self.mod, "SPLIT", None) or {})
        if self.is_production:
            prod = dict(sp.get("production") or {})
            if prod.get("drop_test", True):
                sp["test_start"] = sp["test_end"] = None
            sp["train_end"] = prod.get("train_end")          # 通常 None = 放到数据末日
        return sp

    def data_cfg(self, cfg: Cfg) -> dict:
        """本单元**生效**的数据窗（传给 `mx.data.load`）——模块级 `DATA = {...}` 声明。

        为什么单元需要这个能力：`trainingdata/` 是 2012~2026 的**全历史面板**，
        但不同版本对"用多长的历史"有不同主张：

          · 2012~2017 的因子覆盖率只有 43%~55%（40 个因子那时整列为空、两融因子只覆盖两融标的），
            而 2018 起 225 个因子基本齐全、覆盖率 68%~83%；
          · 全历史 X ≈ 6.6 GB/折（2018 起）~11 GB/折（2012 起），直接决定能几折并行。

        ★ 只允许**收窄**、不允许放宽到产物之外：`start/end` 由 `data.load` 落到年份分区上，
          越界只是读不到那些年，不会凭空造数据。`features` 给名字列表时才生效
          （`"core_2018"` 这类预制特征集由 `meta.json:feature_sets` 解析，见 `mx/data.py`）。
        """
        d = dict(getattr(self.mod, "DATA", None) or {})
        out = {k: d.get(k) for k in ("start", "end", "features") if d.get(k) is not None}
        if out.get("features") and isinstance(out["features"], str):
            out["features"] = _feature_set(cfg, out["features"])
        # ★ `trainingdata`：**换一份初加工快照**（相对模块根）。用途只有一个 ——
        #   上游来了新因子时做「同一套配方、只换特征集」的干净 A/B：
        #   现行 `trainingdata/` 是冻结快照，原地重建会让所有已出结论失去对照组。
        #   写进快照的进程见 `preparingdata.py --out`。
        if d.get("trainingdata"):
            out["data_root"] = str(d["trainingdata"])
        return out

    def heads(self, cfg: Cfg, seed: int) -> list:
        return list(self.mod.heads(cfg, seed))

    def combine(self, per_head: dict):
        fn = getattr(self.mod, "combine", None)
        return fn(per_head) if fn else _default_combine(per_head)

    def as_dict(self) -> dict:
        return {"name": self.name, "title": self.title, "dir": str(self.dir),
                "n_folds": len(self.seeds), "seeds": self.seeds,
                "labels": self.labels, "primary_label": self.primary_label,
                "archived": self.archived, "production": self.is_production,
                "recipe": self.recipe, "drift": self.drift}


def _feature_set(cfg: Cfg, name: str) -> list[str]:
    """把 `DATA["features"] = "core_2018"` 这类**预制特征集名**解析成列名列表。"""
    from . import data as D
    return D.feature_set(cfg, name)


def _default_combine(per_head: dict):
    """默认汇总：把各头**逐日截面 z 后等权相加**（参考工程的跨折/跨族集成口径）。"""
    from . import combine as C
    return C.combine_equal(per_head)


def list_units(cfg: Cfg) -> list[str]:
    """"所有单元名（迭代区 + 实战区）。"""
    return cfg.unit_names()


def unit_dir(cfg: Cfg, name: str) -> Path:
    return cfg.unit_dir(name)


def load_unit(cfg: Cfg, name: str, log=print) -> Unit:
    """把 `Model/<name>/model.py` 当模块载入并校验契约。"""
    d = unit_dir(cfg, name)
    mp = d / "model.py"
    if not mp.exists():
        raise SystemExit(f"✘ 单元 {name} 不存在（缺 {mp}）。已有单元：{list_units(cfg)}")

    # ★★ 硬闸门：**自包含单元必须用它自己的 mx/ 跑**。
    #   单元包里带一份 mx/，模块根也有一份 —— 两份都会 `import mx` 成功，
    #   都会读写**同一个** `<单元>/state/leaderboard.json`，却可能算出不同数字
    #   （引擎一旦分叉）。这种分叉**不报错、只在事后对不上账**，必须当场挡住。
    #   `cfg.root` 由 `mx/` 的位置决定，所以从单元内部运行时两者相等、闸门自动放行。
    if cfg.is_selfcontained(d) and Path(d).resolve() != Path(cfg.root).resolve():
        raise SystemExit(
            f"✘ 单元 `{name}` 是**自包含**的（它自带引擎 `{d}/mx/`），"
            f"不能用模块根的 mx/ 跑。\n"
            f"    两套引擎会写同一个 `{d}/state/leaderboard.json`，却可能给出不同数字 ——\n"
            f"    不报错，只在事后对不上账。\n"
            f"    正确用法：\n"
            f"      cd {d}\n"
            f"      python main.py analysis {name}      # 或用 run.py / train.sh / analysis.py\n"
            f"    （从单元内部跑时，`mx/` 就在它旁边，`cfg.root` 会自动指向单元目录。）")
    mod_name = f"mx_unit_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, mp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    # ★ 单元 `model.py` 是**动态加载**的，`SourceFileLoader` 会顺手在**单元顶层**
    #   留下一份 `__pycache__/model.cpython-*.pyc`。而单元顶层按约定只允许有
    #   入口脚本 + `mx/` + `model.py` + 三个产物桶（用户 2026-09-20 要求），
    #   所以就在这一下临时关掉字节码写入，加载完立刻还原。
    #   （`mx/` 自己的 `__pycache__` 不受影响 —— 它在 `mx/` 里面，是允许的位置。）
    _prev_bc = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)                  # type: ignore[union-attr]
    finally:
        sys.dont_write_bytecode = _prev_bc
    missing = [k for k in REQUIRED if not hasattr(mod, k)]
    if missing:
        raise SystemExit(f"✘ 单元 {name} 的 model.py 缺契约项：{missing}（需 {list(REQUIRED)}）")
    if not list(getattr(mod, "SEEDS")):
        raise SystemExit(f"✘ 单元 {name} 的 SEEDS 为空")
    recipe = _read_recipe(d)
    unit = Unit(name=name, dir=d, mod=mod, recipe=recipe or {})
    unit.drift = check_drift(unit, cfg, log=log)
    return unit


def _read_recipe(d: Path) -> dict:
    """`logs/recipe.yaml`（人类可读）+ `logs/recipe.lock.json`（机器读）。

    ★ 放 `logs/` 而不是单元顶层：单元顶层只留入口脚本与三个产物桶（用户 2026-09-20
      要求「逻辑划分清晰」），而冻结配方本质就是**一条记录**，归 `logs/`。
    """
    lock = d / "logs" / "recipe.lock.json"
    if lock.exists():
        return state.load_json(lock, {}) or {}
    rp = d / "logs" / "recipe.yaml"
    if not rp.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(rp.read_text(encoding="utf-8")) or {}
    except Exception:            # noqa: BLE001
        return {}


def _upstream_fingerprint(cfg: Cfg) -> dict:
    """上游指纹 —— **从 `trainingdata/meta.json` 读**，不再现算。

    ★ 原先走 `upstream.fingerprint(cfg)`，那要 `fea` + 模块② 的台账文件，
      是"单元跑不起来"的最后一道障碍之一。但那些信息**快照里早就有**：
      `meta["source"]` 是初加工时写进去的，含因子指纹、台账 sha256、覆盖区间。

    ★ 而且读快照**比现算更准**：现算拿到的是"现在的上游"，而这里的语义是
      "这一版模型建立在**哪份数据**上"—— 那是建快照那一刻的上游，不是今天的。
      台账 `ledger`（`log*/dayhash.tsv` 的 sha256）是唯一回答"建在哪一天的数据上"
      的东西，务必保留。
    """
    from . import panel_io                                   # 延迟导入（避免环）
    m = panel_io.load_meta(cfg.trainingdata)
    src = m.get("source") or {}
    return {
        "fingerprint": src.get("fingerprint"),
        "universe_fp": src.get("universe_fp"),
        "n_products": src.get("n_products"),
        "ledger": src.get("ledger"),
        "panel_digest": m.get("panel_digest"),
        "prices_digest": (m.get("prices") or {}).get("digest"),
        "built_at": m.get("built_at"),
        # ★ 标明来源：这是**快照记录**，不是"现在去问上游"得到的
        "source": "trainingdata/meta.json",
    }


# ------------------------------------------------------------------ 配方冻结 / 漂移检查
def freeze(cfg: Cfg, name: str, scan_panel: bool = True, log=print) -> dict:
    """把当前配方冻结进 `Model/<name>/recipe.yaml` + `recipe.lock.json`。

    冻结内容 = 引擎哈希 + 单元哈希 + 配置快照 + 上游指纹（因子台账/面板 digest）。
    """
    d = unit_dir(cfg, name)
    unit = load_unit(cfg, name, log=lambda *a: None)
    snap = {
        "name": name,
        "title": unit.title,
        "frozen_at": state.now(),
        "engine_hash": engine_hash(cfg.root),
        "python": sys.version.split()[0],
        "config": {k: cfg.raw.get(k) for k in ("data", "preprocess", "models", "train",
                                               "strategy", "backtest")},
        # ★ 记**本单元生效**的切分（全局 ← 单元 SPLIT ← 生产覆盖），不是全局那份
        "split": unit.split_cfg(cfg),
        "seeds": unit.seeds,
        "labels": unit.my_labels(cfg),
        "primary_label": unit.primary_label,
        "archived": unit.archived,
        "production": unit.is_production,
        "upstream": _upstream_fingerprint(cfg),
    }
    if scan_panel:
        from .data import available_dates
        try:
            days = available_dates(cfg)
            snap["panel"] = {"n_days": len(days), "start": days[0], "end": days[-1]}
        except SystemExit as exc:
            snap["panel"] = {"error": str(exc)}
            log(f"  ⚠️ 扫不到面板（先跑 preparingdata.py 建 trainingdata）：{exc}")
    snap["unit_hash"] = unit_hash(d)                # 注意：在写入 recipe 之后才算得准 → 见下
    _write_recipe(d, snap)
    snap["unit_hash"] = unit_hash(d)
    _write_recipe(d, snap)
    log(f"  ✔ 配方已冻结：{d / 'logs' / 'recipe.yaml'}（引擎 {snap['engine_hash']}）")
    return snap


def _write_recipe(d: Path, snap: dict) -> None:
    import yaml
    out = d / "logs"                      # ★ 与 _read_recipe 同处（见那里的说明）
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {snap['name']} · {snap.get('title','')}",
        "# 本文件由 `python main.py freeze %s` 生成 —— 冻结配方，不要手改。" % snap["name"],
        "# 改任何东西都请开新版本 V{N+1}（参考工程纪律：版本只增不改）。",
        "",
    ]
    body = yaml.safe_dump(snap, allow_unicode=True, sort_keys=False, default_flow_style=False)
    (out / "recipe.yaml").write_text("\n".join(lines) + body, encoding="utf-8")
    state.save_json(out / "recipe.lock.json", snap)


def check_drift(unit: Unit, cfg: Cfg, log=print) -> dict:
    """引擎/单元哈希是否与冻结时一致。"""
    if not unit.recipe:
        return {"frozen": False, "engine": False, "unit": False}
    eng_now, unit_now = engine_hash(cfg.root), unit_hash(unit.dir)
    drift = {"frozen": True,
             "engine": unit.recipe.get("engine_hash") != eng_now,
             "unit": unit.recipe.get("unit_hash") != unit_now,
             "engine_hash": eng_now, "frozen_engine": unit.recipe.get("engine_hash")}
    if drift["engine"] and getattr(unit, "archived", False):
        return drift                     # 历史单元：口径已过期，漂移是预期，不再刷屏告警
    if drift["engine"]:
        log(f"  ⚠️ {unit.name}: 引擎哈希与冻结时不同 —— 本版结论可能不再可复现"
            f"（冻结 {drift['frozen_engine']} → 现在 {eng_now}）。")
        log("     · 只是修 bug/加功能：跑完训练用 `main.py freeze` 刷新配方（并记进 README）。")
        log("     · 要严格复现旧结论：把 `mx/` 回退到冻结时的提交再跑。")
    return drift


# ------------------------------------------------------------------ 折 / 种子
def fold_seed(unit: Unit, fold: int) -> int:
    """折号（1 起）→ 种子。超出种子表就按 base+fold*1000 外推（参考工程惯例）。"""
    seeds = unit.seeds
    if 1 <= fold <= len(seeds):
        return seeds[fold - 1]
    return int(seeds[0] + fold * 1000)


def folds(unit: Unit) -> list[int]:
    return list(range(1, len(unit.seeds) + 1))


def head_names_safe(cfg: Cfg, unit: Unit) -> list[str]:
    """本版的头名（拿不到就返回空 —— doctor/status 这类命令不该因构造失败而崩）。"""
    try:
        return [m.name for m in unit.heads(cfg, unit.seeds[0])]
    except Exception:                       # noqa: BLE001
        return []


# ------------------------------------------------------------------ 晋级 best
def promote(cfg: Cfg, name: str, label: str | None = None, log=print) -> dict:
    """把单元 `Model/<name>` 晋级为 `Model/best`（实战版）。

    参考工程的 best 是**与 Vi 同构的独立单元**（可自训、可推演），而不是一份产物拷贝；
    所以这里复制的是「代码 + 冻结配方 + 本版打分」，best 自己随时能 `bash train.sh` 重训。

    晋级前会拦两件事：① 引擎漂移；② 没有分析产物（没跑过 analysis 的版本不给晋级）。
    """
    import shutil
    u = load_unit(cfg, name)
    if u.archived:
        raise SystemExit(f"✘ {name} 是**历史单元**（ARCHIVED）—— 口径已过期，不能晋级为 best。"
                         f"要晋级请在现行口径下新建/选出单元。")
    if u.drift and u.drift.get("engine"):
        raise SystemExit(
            f"✘ {name} 的引擎哈希已漂移（冻结 {u.drift.get('frozen_engine')} → 现在 {u.drift.get('engine_hash')}）。\n"
            f"   先 `python main.py freeze {name}` 刷新配方并在这里记一笔，再晋级。")
    label = label or u.primary_label or u.my_labels(cfg)[0]
    score = state.unit_pred_dir(cfg, name) / f"score__{name}__{label}"
    if not score.exists():
        raise SystemExit(f"✘ 没有单元打分产物 {score} —— 先跑 `python main.py analysis {name}`")
    lb = state.load_json(state.unit_leaderboard_path(cfg, name), {}) or {}
    if not lb:
        raise SystemExit(f"✘ 没有 {name} 的分析结果 —— 先跑 `python main.py analysis {name}`")

    best = cfg.unit_dir("best")
    best.mkdir(parents=True, exist_ok=True)
    for f in ("model.py", "run.py", "train.sh", "analysis.py"):
        src = cfg.unit_dir(name) / f
        if src.exists():
            shutil.copy2(src, best / f)
    # ★ 配方在 `logs/` 里（单元顶层只留入口脚本 + 三个产物桶）—— 在 best 里保持同样的位置，
    #   否则 `_read_recipe` 会在 best 目录下找不到 lock、把「已冻结」判成「未冻结」。
    (best / "logs").mkdir(parents=True, exist_ok=True)
    for f in ("recipe.yaml", "recipe.lock.json"):
        src = cfg.unit_dir(name) / "logs" / f
        if src.exists():
            shutil.copy2(src, best / "logs" / f)
    (best / "version.txt").write_text(name + "\n", encoding="utf-8")
    out_pred = state.unit_pred_dir(cfg, "best")          # Model/best/model_pred/
    out_pred.mkdir(parents=True, exist_ok=True)
    shutil.copy2(score, out_pred / score.name)

    row = next((r for r in (lb.get("leaderboard") or [])
                if r.get("head") == name and r.get("label") == label), None)
    bt = (lb.get("backtest") or {}).get("unit") or {}
    t = (row or {}).get("eval") or (row or {}).get("test") or {}
    # 策略矩阵的主源单元格才是用户裁定的业绩口径（现金级真实净值，≤5 只）
    cells = ((lb.get("strategy_matrix") or {}).get("cells") or [])
    hl = (lb.get("strategy_matrix") or {}).get("headline") or ""
    strat = next((c for c in cells if c.get("source") == hl and c.get("strategy") == "top5_hold"
                  and (c.get("params") or {}).get("hold_days") == 5),
                 next((c for c in cells if c.get("source") == hl and c.get("scope") == "ens"), None))
    sps = u.split_cfg(cfg)
    card = [
        f"# best —— 当前实战版（来源单元 {name}）",
        "",
        f"- 晋级时间：{state.now()}",
        f"- 来源：`{name}/`（{u.title}）",
        f"- 主腿标签：`{label}`；头：`{name}`（本单元汇总口径）",
        f"- 数据截止：{((lb.get('panel') or {}).get('n_days'))} 个交易日 · 面板 digest {str((lb.get('panel') or {}).get('digest'))[:12]}",
        f"- 配方：`logs/recipe.yaml`（引擎 {u.recipe.get('engine_hash')} · 冻结于 {u.recipe.get('frozen_at')}）",
        "",
        "## 切分（★ 实战口径 = 来源单元的口径去掉 test 窗）",
        "",
        f"- 本目录生效：mode `{sps.get('mode')}` · train_end `{sps.get('train_end') or '数据末日'}` · "
        f"test {'无' if not sps.get('test_start') else str(sps.get('test_start')) + '~' + str(sps.get('test_end'))}",
        f"- 来源单元 {name} 的研究口径：test `{u.recipe.get('split', {}).get('test_start')}` "
        f"~ `{u.recipe.get('split', {}).get('test_end')}`（train+valid 到 "
        f"`{u.recipe.get('split', {}).get('train_end')}`）",
        f"- ★ 下面这张指标表来自**研究口径的 test 窗**（样本外）；best 自己重训会用**全部数据**"
        f"（`bash train.sh`），那时没有样本外指标可比 —— 两者不可混为一谈。",
        "",
        "## 指标（test 窗口，来自晋升时的 analysis）",
        "",
        f"| 项 | 值 |",
        f"|:--|:--|",
        f"| 评价窗 RankIC | {t.get('rankic_mean')} |",
        f"| 评价窗 ICIR / t | {t.get('rankic_icir')} / {t.get('rankic_t')} |",
        f"| 评价窗分层 D9−D0 | {t.get('spread')} |",
        f"| 组合回测（等权 top{N}，含费用）净年化 | {bt.get('ann_return_net')} |",
        f"| 组合回测夏普 / 最大回撤 | {bt.get('sharpe')} / {bt.get('max_drawdown')} |",
        (f"| 策略（{strat.get('strategy')} {strat.get('params')}）累计 | {strat.get('total_ret')} |"
         f"  ★ 小资金口径（≤5 只、T+1 开盘、含费用）"
         if strat else "| 策略矩阵 | （本次分析没有产出） |"),
        "",
        "## 用法",
        "",
        "```bash",
        "cd " + str(best),
        "bash train.sh              # 需要重训时（同构单元，可自训）",
        "python analysis.py         # 推演 + 榜单 + 回测",
        "```",
        "",
        "★ 纪律：best 只能通过 `python main.py promote <单元>` 晋级，不要手工改本目录里的代码；",
        "  要改就开新版本 `Model/V{N+1}`，验证通过后再晋级。",
    ]
    (best / "CARD.md").write_text("\n".join(card), encoding="utf-8")
    log(f"  ✔ 已晋级：Model/{name} → Model/best（主腿 {label}，打分 {score.name}）")
    log(f"    发版卡：{best / 'CARD.md'}")
    return {"from": name, "to": "best", "label": label, "card": str(best / "CARD.md")}
