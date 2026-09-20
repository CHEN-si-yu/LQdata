#!/usr/bin/env python
"""初加工 —— 上游因子 → `trainingdata/`（模型可直接读的训练输入，**按年拆分**）。

**这个脚本只干一件事**：把上游因子产物拼成一份快照，并维护它的增量。
全量 / 增量两条路径都在 `prepare()` 里，块怎么算在 `build_year()` 与 `BLOCKS` 里。

    trainingdata/
      meta.json                              清单 / 边界 / 上游指纹 / 逐年摘要 / 覆盖率
      factors/year=YYYY/data.parquet         trade_date, stock_code, <337 个特征>
      target/year=YYYY/data.parquet          trade_date, stock_code, <5 个 target>
      amount/year=YYYY/data.parquet          trade_date, stock_code, amount  ← 每日成交额（元）
      fac_sample/year=YYYY/data.parquet      trade_date, stock_code, <20 个抽样特征>

## 四块产物各自的定位（2026-09-20 由原来的七块收敛）

| 块 | 装什么 | 谁用 | 动它会改 `panel_digest` 吗 |
|:--|:--|:--|:--|
| `factors` | 全部特征的**当日截面 rank**，方向已统一成"越大越好" | 训练/评价 | **会** |
| `target` | 1d/3d/5d/10d/20d 五个 target（`label_ret_*d`）的收益原值 | 训练/评价 | **会** |
| `amount` | 每日**成交额（元）**，用户口径叫"市场所有的流动性" | 回测/策略 | 不会 |
| `fac_sample` | `factors` 里**随机抽 20 列**（seed=42），值逐字取自 `factors` | 冒烟/最小输入 | 不会 |

★ **`panel_digest` 只哈希 `columns.features` + `columns.labels` + 逐年 `files.factors.sha`。**
  哈希串里**不含块名**（块名只用来查 key），所以块改名本身**不换锚点** —— 但前提是
  块名常量与 meta 的 key **同步**改；只改一头会静默换掉锚点（实测漏改会得到
  `d90b20cf37710f94`，而正确值是 `798ccb32214853a8`）。
  所以 `amount` / `fac_sample` 是**只增不改**的：补建它们不会作废任何已有结论。
  反过来，`factors`/`target` 一变，所有锚在这个指纹上的结论都要重跑 —— 这条界线要守住。

## 增量：窗口取多少天，是标签定的

窗口 = `[上次已建末日 − lb, 上游最新日]`，`lb = max(标签 h) + 1 + REDUNDANT_DAYS(7)` = 28。

- **`h + 1` 是硬下限**：标签 `T` 用到 `open(T+1+h)`，所以 `T` 要等到 `T+1+h` 那天才定型。
  窗口比这短，那些天一旦出窗就再也不会被重算 ⇒ **20d 标签永久停在 NaN**（上游后来补上也不捡）。
- **`+7` 是冗余覆盖**（用户 2026-09-20 要的）：即使标签已定型，也把那 7 个交易日重算一遍，
  用来吃掉上游的小幅回改。
- 为什么**不**写成"每个标签各自刷最新 7 个有效行"：那样跨度大时会漏 ——
  若上次跑在 30 个交易日之前，20d 标签是在 `T_max−51 ~ T_max−21` 里陆续定型的，
  只刷最新 7 行就漏掉 `T_max−51 ~ T_max−29`，它们**永久是 NaN**。
  `上次末日 − lb` 这种写法对任何间隔都成立。

## 这个脚本为什么是**单文件**

它原先拆成 `mx/prepared.py` + `panel_io.py` + `upstream.py` + `config.py` + `state.py` 五个模块，
好让"只想读个 parquet"的模型侧不必拖上上游依赖。2026-09-20 重构后顶层 `mx/` 被删、
模型侧每个单元各自带一份引擎，于是这套拆分对**构建期**就只剩坏处：
副作用是本脚本不再 import 项目内任何东西（**只依赖 numpy / pandas / pyarrow + 上游 `fea`**），
换机器/交接时一个文件就是全部。

★ 两处**看着能简化、其实不能动**的地方，改之前先读它们的 docstring：
`_place` / `_cols`（热路径：满格快路 + 线程池，单年 285 s → 76 s）与
`build_year` 里的不变式校验（每条都对应一次真实事故）。

## 用法

    python preparingdata.py                    # 自动：没有产物→全量；有→增量
    python preparingdata.py --full             # 全量重建
    python preparingdata.py --years 2026       # 只加工指定年份
    python preparingdata.py --check            # 只校验不写：meta ↔ 文件 ↔ 上游
    python preparingdata.py --amount-only      # 只补 amount（每日成交额）
    python preparingdata.py --fac-sample-only  # 只补 fac_sample（因子抽样）
    python preparingdata.py --meta-only [--freeze]  # 只补 meta 派生字段（不碰数据、不读因子侧）
    python preparingdata.py --out trainingdata_new --full   # 写到另一份目录（现有产物一字不动）

    --jobs 4      读并发（默认 6；共享盘上有别的任务在跑时别开大）
    --lookback N  增量回溯交易日数（默认 = 最长标签 h + 1 + 7，见上）

★ **本脚本是唯一的写入口**：模型侧（各单元的 `mx/`）只读 `trainingdata/`，绝不写。
★ 必须用 `/autodl-fs/data/miniconda3/bin/python`（非登录 shell 的 `python` 没有 pandas）。
★ 产物位置：`--out` > `$MX_DATA` > `<脚本目录>/trainingdata`。

## 上游在哪

三个数据源都在 `featureengineering/` 的**上一级**（= `/autodl-fs/data/`）：

| 常量 | 默认值 | 装什么 |
|:--|:--|:--|
| `FACTORS_DIR` | `../featureengineering/data/factors` | 每个因子一个目录，`year=YYYY/data.parquet` |
| `FACTORS_STATE` | `../featureengineering/state` | 因子的 manifest（零 I/O 读覆盖区间） |
| `FACTORS_ROOT` | `../featureengineering` | 只为 `import fea.*`（读层 / 股票池 / 价格层） |
| `RAW_DATA` | `../datadownload/data` | 模块① 的原始表（`stock_daily` 等），只给 `amount` 用 |

换机器时用 `MX_UPSTREAM` / `MX_FACTORS_DIR` / `MX_FACTORS_STATE` / `MX_RAW_DATA` 覆盖。
"""
from __future__ import annotations

# ★ 必须在 import numpy 之前压线程数：BLAS/OpenMP 默认吃满宿主机核数，
#   与"读并发"以及机器上别的任务互相踩踏。要改就设环境变量 MX_THREADS。
import os

_MX_THREADS = os.environ.get("MX_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _MX_THREADS

import argparse                                                  # noqa: E402
import gc                                                        # noqa: E402
import hashlib                                                   # noqa: E402
import json                                                      # noqa: E402
import re                                                        # noqa: E402
import shutil                                                    # noqa: E402
import sys                                                       # noqa: E402
import tempfile                                                  # noqa: E402
import time                                                      # noqa: E402
from concurrent.futures import ThreadPoolExecutor                # noqa: E402
from datetime import datetime, timedelta                         # noqa: E402
from functools import lru_cache                                  # noqa: E402
from pathlib import Path                                         # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
import pyarrow as pa                                             # noqa: E402
import pyarrow.parquet as pq                                     # noqa: E402


# ============================================================================
# 0. 路径与配置
# ============================================================================
#: 本脚本所在目录 = 模型工程根（`preparingdata.py` 与 `trainingdata/` 同级）
ROOT = Path(__file__).resolve().parent

#: 上游因子工程根（模块②）。只为 `import fea.*` 用 —— 读层、股票池、价格层都在它里面。
FACTORS_ROOT = Path(os.environ.get("MX_UPSTREAM", ROOT / "../featureengineering")).resolve()
#: 因子产物目录：每个因子一个子目录，里面 `year=YYYY/data.parquet`
FACTORS_DIR = Path(os.environ.get("MX_FACTORS_DIR", FACTORS_ROOT / "data" / "factors")).resolve()
#: 因子的状态目录（manifest 在这，回答"某因子覆盖哪些年"时零 I/O）
FACTORS_STATE = Path(os.environ.get("MX_FACTORS_STATE", FACTORS_ROOT / "state")).resolve()
#: 模块① 的原始数据表。**只有 `amount` 块用** —— 四块里唯一不来自因子侧的一块。
RAW_DATA = Path(os.environ.get("MX_RAW_DATA", FACTORS_ROOT.parent / "datadownload" / "data")).resolve()

#: 五个 target 的名字。上游把它们和普通因子存在同一个目录（靠 `is_label` 区分）。
LABEL_NAMES = ("label_ret_1d", "label_ret_3d", "label_ret_5d",
               "label_ret_10d", "label_ret_20d")


class Cfg:
    """构建期的极简配置对象。

    ★ **属性名与原先的 `mx/config.py:Cfg` 保持一致**（`factors_dir` / `factors_root` /
      `root` / `trainingdata` / `labels` / `label_horizon`）—— 这样第 1~4 节那些搬运过来的
      函数体**一个字都不用改**。这才是这个类存在的唯一理由。
    ★ 它**不是**模型侧的配置：`split` / `models` / `strategy` 那些跟构建期无关的东西一概没有。
    """

    def __init__(self, root: Path = ROOT, trainingdata: Path | None = None):
        self.root = Path(root)
        self.factors_root = FACTORS_ROOT
        self.factors_dir = FACTORS_DIR
        self.factors_state = FACTORS_STATE
        self.raw_data = RAW_DATA
        self.labels = list(LABEL_NAMES)
        #: 产物根：`--out` > `$MX_DATA` > `<root>/trainingdata`
        env = os.environ.get("MX_DATA")
        self.trainingdata = (Path(trainingdata) if trainingdata is not None
                             else Path(env).expanduser() if env
                             else self.root / "trainingdata")

    def label_horizon(self, label: str) -> int:
        """`label_ret_5d` → 5（用于增量回溯窗口与 purge）。"""
        s = str(label).rsplit("_", 1)[-1]
        return int(s[:-1]) if s.endswith("d") and s[:-1].isdigit() else 1

# ============================================================================
# 1. 状态落盘（原 `mx/state.py`）
# ============================================================================


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, default=_default)
        try:
            os.chmod(tmp, 0o664)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _default(o):
    """把 numpy 标量/数组转成可序列化的东西（训练指标里到处是 np.float32）。"""
    try:
        import numpy as np
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:            # noqa: BLE001
        pass
    return str(o)


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# 2. 产物 I/O 原语（原 `mx/panel_io.py`）
# ============================================================================

# ---- 块名。2026-09-20 由七块收敛成四块：factors / target / amount / fac_sample
#: **必建**的两块：特征矩阵与标签。`panel_digest` 只认这两块。
FK, TK = "factors", "target"
#: 布局的必建两块。另两块 `amount` / `fac_sample` 在第 5 节定义，同样必建（见 `ALL_KINDS`）。
KINDS = (FK, TK)
KEY = ("trade_date", "stock_code")


# ================================================================ 路径
def root_of(cfg: Cfg) -> Path:
    """产物根目录（`--out` > `$MX_DATA` > `<脚本目录>/trainingdata`，解析在 `Cfg` 里）。"""
    return cfg.trainingdata


def meta_path(root: Path) -> Path:
    return Path(root) / "meta.json"


def year_file(root: Path, kind: str, year: int) -> Path:
    return Path(root) / kind / f"year={int(year)}" / "data.parquet"


def years_of(root: Path, kind: str = FK) -> list[int]:
    """产物里已有的年份（只扫目录名，不读文件）。"""
    d = Path(root) / kind
    if not d.exists():
        return []
    out = []
    for p in d.iterdir():
        m = re.match(r"year=(\d{4})$", p.name)
        if m and p.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)


def load_meta(root: Path) -> dict:
    return load_json(meta_path(root), {}) or {}


# ================================================================ 日期换算
def to_int(d: str) -> int:
    """'2026-01-05' → 20260105"""
    return int(str(d).replace("-", "")[:8])


def to_str(d: int) -> str:
    """20260105 → '2026-01-05'"""
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


# ================================================================ 落盘
def atomic_parquet(tab: pa.Table, path: Path) -> None:
    """原子写 Parquet（zstd-3）—— 支撑"meta 最后写"的崩溃安全口径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(tab, tmp, compression="zstd", compression_level=3)
    try:
        os.chmod(tmp, 0o664)
    except OSError:
        pass
    os.replace(tmp, path)


# ================================================================ 读产物
def read_frame(root: Path, kind: str, years: list[int] | None = None,
               columns: list[str] | None = None, codes: list[str] | None = None,
               start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """读若干年的产物 → `trade_date/stock_code + 列` 的长表（按日期、代码升序）。

    `columns=None` 读全部列；`codes/start/end` 在下推过滤（先按 parquet 过滤再 pandas 过滤，
    避免把整年读进内存再裁）。
    """
    root = Path(root)
    ys = years if years is not None else years_of(root, kind)
    parts = []
    for y in ys:
        p = year_file(root, kind, int(y))
        if not p.exists():
            continue
        cols = columns
        if cols is not None:
            cols = list(dict.fromkeys(list(KEY) + [c for c in cols if c not in KEY]))
        t = pq.read_table(p, columns=cols)
        if codes is not None:
            t = t.filter(pa.compute.is_in(t["stock_code"], value_set=pa.array(codes)))
        df = t.to_pandas()
        del t
        if start is not None:
            df = df[df["trade_date"] >= str(start)[:10]]
        if end is not None:
            df = df[df["trade_date"] <= str(end)[:10]]
        if len(df):
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=list(KEY) + list(columns or []))
    df = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str).str[:10]
    df["stock_code"] = df["stock_code"].astype(str)
    return df.sort_values(list(KEY), kind="stable").reset_index(drop=True)


# ============================================================================
# 3. 上游只读适配（原 `mx/upstream.py`）
# ============================================================================

# ---------------------------------------------------------------- fea 导入
_FEA_READY = False


def _ensure_fea(cfg: Cfg) -> None:
    """把模块② 的工程根加进 sys.path 并 import 它（幂等、懒加载）。"""
    global _FEA_READY
    if _FEA_READY:
        return
    root = str(cfg.factors_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    # import 即生效；`fea.*` 是模块② 的公开包路径
    import fea.config as _fcfg          # noqa: F401
    import fea.dates as _fdates         # noqa: F401
    import fea.store as _fstore         # noqa: F401
    _FEA_READY = True


@lru_cache(maxsize=1)
def _factors_registered(cfg_root: str) -> bool:
    """确保 `factors/` 被 import —— 因子注册靠 import 触发（见模块② DEVELOPING.md）。"""
    if cfg_root not in sys.path:
        sys.path.insert(0, cfg_root)
    import factors  # noqa: F401
    return True


# ---------------------------------------------------------------- 读因子产物
def list_factor_names(cfg: Cfg) -> list[str]:
    """产物目录下的全部因子名（含 5 个标签）。"""
    d = cfg.factors_dir
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


# ---------------------------------------------------------------- 股票池 / 价格层
@lru_cache(maxsize=4)
def _engine(cfg_root: str, cfg_yaml: str):
    """构建模块② 的 Engine（**注意**：`lru_cache` 按参数缓存，Engine 本身很轻）。

    用模块② 自己的 conf/config.yaml 构建 —— 这样股票池口径（主板/ST/上市窗口）
    与因子层**逐格一致**，不会出现"模型认为有效的样本，因子侧认为不在池内"。

    ★ 自己确保 `sys.path` 里有上游根：调用方（回测/策略）可能**先** `import fea.panel`
      再调这里。2026-09-18 之前不是问题（那时数据层会读上游、顺手把路径加好），
      改成读 trainingdata 之后就踩到了 —— 别把这个副作用再依赖在调用顺序上。
    """
    if cfg_root and cfg_root not in sys.path:
        sys.path.insert(0, cfg_root)
    import fea.config as fcfg
    from fea.engine import Engine

    return Engine(fcfg.load())


#   从布局里摘掉了（实测它恒为 True，详见模块头部）。它们曾是本文件里**唯一**构造
#   上游 `fea.Panel` 的地方，删掉顺带让每年的构建少一次 `universe_for`（纯白算：掩码
#   以前也没进 `_coverage`，只被原样写盘）。要恢复请从 git 取回。
#   注意：上游**股票池口径指纹** `universe_fp` 仍在 `meta.source` 里，那是另一回事，
#   由 `_universe_fp()` 算，与本块无关。


def last_upstream_day(cfg: Cfg) -> str:
    """上游日频基准表的最新交易日（**不是"今天"** —— 见模块② `baseline_last_day` 的坑）。"""
    _ensure_fea(cfg)
    eng = _engine(str(cfg.factors_root), str(cfg.root))
    return to_str(eng.baseline_last_day())


# ---------------------------------------------------------------- 因子侧先验
def upstream_eval_summary(cfg: Cfg) -> list[dict]:
    """模块② 的因子评价表（`state/eval/summary.json`）——用于特征初筛与基线对照。

    ⚠️ 已知：该文件的 `icir`/`t` 在同一份数据上**全是 NaN**（模块② 按"年份分区"算，
    当前只有 2026 一个分区）→ 模型侧要用自己从日频 IC 序列算的 ICIR/t。
    """
    p = cfg.factors_state / "eval" / "summary.json"
    if not p.exists():
        return []
    import json
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []


# ---------------------------------------------------------------- 上游指纹（版本配方锁定用）
def upstream_fingerprint(cfg: Cfg) -> dict:
    """上游数据的指纹 —— 写进单元 recipe，回答「这一版结论建立在哪份数据上」。

    三样东西：
      ① `factors_dir` 与最新因子日（产物里实际出现的最后一天）；
      ② 模块② 的**日台账**（`log*/dayhash.tsv`，按天 MD5）最新一份的哈希与最后一天；
      ③ 股票池口径的来源（`factors_state/eval/summary.json` 的行数/时间戳）。
    """
    import hashlib
    fp: dict = {"factors_dir": str(cfg.factors_dir)}
    try:
        fp["last_factor_day"] = last_upstream_day(cfg)
    except Exception as exc:                       # noqa: BLE001
        fp["last_factor_day"] = f"<err: {type(exc).__name__}>"

    ledgers = sorted(cfg.factors_root.glob("log*/dayhash.tsv"))
    if ledgers:
        p = ledgers[-1]
        try:
            text = p.read_text(encoding="utf-8")
            lines = [l for l in text.splitlines() if l.strip()]
            fp["ledger"] = {"path": str(p.relative_to(cfg.factors_root)),
                            "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                            "n_days": max(0, len(lines) - 1),
                            "last": lines[-1].split("\t")[0] if len(lines) > 1 else "",
                            "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
        except OSError:
            pass
    s = cfg.factors_state / "eval" / "summary.json"
    if s.exists():
        fp["eval_summary"] = {"n": len(upstream_eval_summary(cfg)),
                              "mtime": datetime.fromtimestamp(s.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
    return fp


# ============================================================================
# 4. 初加工（原 `mx/prepared.py`）
# ============================================================================

#: 2 = 版本 2 的产物布局（**2026-09-20 起 = factors / target / amount / fac_sample 四块**；
#:     此前同一版本号下是 X / Y / universe / P / amount / fac_sample 六块）。
#: 1 = 三块。★ 只写不读，纯标记 —— 没有任何消费者校验它，所以布局改了也没动它。
META_VERSION = 2
# 逐日覆盖率的告警规则：**相对**当年均值（低于它的一半）或低于绝对下限。
# ★ 不能用"理想值"当门槛：实测正常年份的均值就是 0.43（2012，40 个因子起点晚）
#   ~ 0.83（2020 年代）—— 拿 0.6 去比会把早年整片报成"有问题"，那种体检等于没体检。
#   规则要抓的是"上游写盘写到一半"这类事故：那天会**明显偏离当年自身的水位**。
LOW_COV_REL = 0.5               # 低于当年均值的一半 → 告警
LOW_COV_ABS = 0.15              # 或绝对值低于它 → 告警
DEFAULT_JOBS = 6                # 共享盘 + 常有别的任务在跑：默认别开大
REDUNDANT_DAYS = 7              # 增量回溯的**冗余覆盖**天数（用户 2026-09-20 要的，见 `lookback_for`）
ROOT_SCRIPT = Path(__file__).resolve()      # 本脚本自己的字节也进上游指纹


def lookback_for(cfg: Cfg, labels: list[str] | None = None) -> int:
    """增量回溯窗口（交易日）= **最长标签的 h + 余量**。

    ★ 为什么不能拍一个常数：标签 `T` 用到 `open(T+1+h)`，所以一个交易日要等到
      **h+1 个交易日之后**它的标签才定型。若窗口比这短，那些天一旦离开窗口就再也不会被重算，
      20d 标签会**永久停在 NaN**（上游后来补上了也不会被捡回来）。默认取 max(h)+2 最稳。
    """
    ls = labels if labels is not None else cfg.labels
    return max((cfg.label_horizon(l) for l in ls), default=1) + 1 + REDUNDANT_DAYS


# ================================================================ 路径
# ★ `root_of` / `meta_path` / `year_file` / `years_of` / `load_meta` 定义在第 2 节。
#   它们原先被单独拆成一个模块（`mx/panel_io.py`），好让模型侧不拖上游依赖；
#   本脚本是构建期的、本来就要上游，所以合并回来。


# ================================================================ 上游扫描（零 I/O / 极低 I/O）
def upstream_specs(cfg: Cfg) -> dict[str, dict]:
    """`{因子名: {is_label, higher_is_better, group}}` —— 只取**有产物目录**的那些。"""
    _ensure_fea(cfg)                                  # noqa: SLF001
    _factors_registered(str(cfg.factors_root))        # noqa: SLF001
    from fea.spec import all_specs

    have = set(list_factor_names(cfg))
    out: dict[str, dict] = {}
    for s in all_specs():
        if s.name in have:
            out[s.name] = {"is_label": bool(getattr(s, "is_label", False)),
                           "higher_is_better": bool(getattr(s, "higher_is_better", True)),
                           "group": str(getattr(s, "group", ""))}
    return out


def _factor_years(cfg: Cfg, name: str) -> list[int]:
    from fea.store import factor_years
    return [int(y) for y in factor_years(cfg.factors_dir, name)]


def _read_days(cfg: Cfg, name: str, year: int) -> list[str]:
    p = year_file(cfg.factors_dir, name, year)
    if not p.exists():
        return []
    t = pq.read_table(p, columns=["trade_date"]).column("trade_date").to_pylist()
    return sorted({str(d)[:10] for d in t})


def _span_of(cfg: Cfg, name: str) -> tuple[str | None, str | None]:
    """一个因子的产物覆盖区间：先读 manifest（零 I/O），没有再退回读首末年的日期列。"""
    from fea.manifest import Manifest

    m = Manifest.load(cfg.factors_state, name)
    lo = min((v["min_date"] for v in m.partitions.values() if v.get("min_date")), default=None)
    hi = max((v["max_date"] for v in m.partitions.values() if v.get("max_date")), default=None)
    if lo and hi:
        return str(lo), str(hi)
    ys = _factor_years(cfg, name)
    if not ys:
        return None, None
    first, last = _read_days(cfg, name, ys[0]), _read_days(cfg, name, ys[-1])
    return (min(first) if first else None, max(last) if last else None)


def scan(cfg: Cfg, names: list[str] | None = None, jobs: int = DEFAULT_JOBS, log=print) -> dict:
    """扫上游：每因子的年份与覆盖区间（manifest 优先，零 I/O）。"""
    names = names or sorted(upstream_specs(cfg))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        spans = list(pool.map(lambda n: _span_of(cfg, n), names))
    per = {n: {"start": lo, "end": hi, "years": _factor_years(cfg, n)}
           for n, (lo, hi) in zip(names, spans)}
    los = [v["start"] for v in per.values() if v["start"]]
    his = [v["end"] for v in per.values() if v["end"]]
    res = {"n_factors": len(names),
           "span": (min(los) if los else None, max(his) if his else None),
           "years": sorted({y for v in per.values() for y in v["years"]}),
           "per_factor": per, "seconds": round(time.time() - t0, 1)}
    log(f"  上游扫描：{len(names)} 个因子 · 分区年 {len(res['years'])} 个 · "
        f"覆盖 {res['span'][0]} ~ {res['span'][1]}（{res['seconds']}s，零 I/O）")
    return res


# ================================================================ 日期轴 / 股票轴
def build_axis(cfg: Cfg, lo: str, hi: str, log=print) -> tuple[list[str], np.ndarray]:
    """(交易日列表, 股票代码数组) —— 日历 ∩ [lo, hi] × Engine.codes（含退市股，无幸存者偏差）。"""
    eng = _engine(str(cfg.factors_root), str(cfg.root))     # noqa: SLF001
    days = [str(d)[:10] for d in eng.cal.day_strs if lo <= str(d)[:10] <= hi]
    if not days:
        raise SystemExit(f"✘ 日历 ∩ [{lo}, {hi}] 是空的 —— 检查上游产物端点")
    codes = np.asarray(eng.codes, dtype=object)
    log(f"  日期轴：{len(days)} 个交易日（{days[0]} ~ {days[-1]}）· 股票轴：{len(codes)} 只")
    return days, codes


# ================================================================ 落格：一个因子 → 扁平网格
def _place(df: pd.DataFrame, *, source: str, flip: bool, codes_index: pd.Index,
           days_index: pd.Index, n_days: int, C: int,
           codes: np.ndarray | None = None,
           days_int: np.ndarray | None = None) -> tuple[np.ndarray, int, int]:
    """把一列因子落到 (T×C,) 扁平网格（日为主序）。返回 (数组, 命中数, 丢弃数)。

    `source`：特征取 `rank` 列（当日截面百分位），标签取 `value` 列（收益原值）。

    ## 两条路径

    上游产物绝大多数是"整年满格"（每个交易日 × 全部代码都有一行）。这时**不需要**逐行做
    字符串→索引的映射：行序天然就是 (日, 码)，直接 `reshape` 即可。这条快路很重要 ——
    通用路径的 `Index.get_indexer` 要哈希 84 万条字符串，**握着 GIL**，8 个线程也跑不满 1 个核
    （实测 1.1 核、单年 4 分钟；快路后降到几十秒）。

    快路的正确性由三件事保证（都通过才走）：行数 = 天数×代码数、日期向量**逐元素**等于
    `repeat(days, C)`、首末两天的代码向量等于权威 codes。任一不满足就退回通用路径。
    """
    if codes is not None and days_int is not None and len(df) == n_days * C and n_days:
        from fea.dates import series_to_int
        di = series_to_int(df["trade_date"])
        if np.array_equal(di, np.repeat(days_int, C)):
            if (np.array_equal(df["stock_code"].head(C).to_numpy(dtype=object), codes)
                    and np.array_equal(df["stock_code"].tail(C).to_numpy(dtype=object), codes)):
                v = df[source].to_numpy(dtype=np.float32, copy=True)
                if flip:
                    v = np.float32(1.0) - v      # 方向统一：1−rank 仍是 [0,1]，NaN 保持 NaN
                return v, int(v.size), 0

    di = days_index.get_indexer(df["trade_date"].to_numpy(dtype=object))
    ci = codes_index.get_indexer(df["stock_code"].to_numpy(dtype=object))
    ok = (di >= 0) & (ci >= 0)
    v = df[source].to_numpy(dtype=np.float32, copy=False)
    out = np.full(n_days * C, np.nan, dtype=np.float32)
    out[di[ok] * C + ci[ok]] = v[ok]
    if flip:
        out = np.float32(1.0) - out          # 方向统一：1−rank 仍是 [0,1]，NaN 保持 NaN
    return out, int(ok.sum()), int((~ok).sum())


def _cols(cfg: Cfg, year: int, names: list[str], flip_map: dict[str, bool], source: str,
          codes_index: pd.Index, days_index: pd.Index, n_days: int, C: int,
          *, jobs: int, what: str, codes: np.ndarray, days_int: np.ndarray,
          log=print) -> tuple[dict[str, np.ndarray], dict]:
    """并发读若干因子的**一年**产物并落格。返回 (列字典, 逐因子统计)。"""
    from fea.store import read_year

    out: dict[str, np.ndarray] = {}
    stat: dict[str, dict] = {}

    def _one(nm: str):
        df = read_year(cfg.factors_dir, nm, int(year))
        if df is None or len(df) == 0:
            return nm, None
        return nm, _place(df, source=source, flip=flip_map[nm], codes_index=codes_index,
                          days_index=days_index, n_days=n_days, C=C,
                          codes=codes, days_int=days_int)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for k, (nm, r) in enumerate(pool.map(_one, names), start=1):
            if r is None:
                out[nm] = np.full(n_days * C, np.nan, dtype=np.float32)
                stat[nm] = {"hit": 0, "dropped": 0, "missing_year": True}
            else:
                arr, hit, dropped = r
                out[nm] = arr
                stat[nm] = {"hit": hit, "dropped": dropped, "missing_year": False}
            if log and (k % 50 == 0 or k == len(names)):
                log(f"    {what} {k}/{len(names)} …")
    return out, stat


def _coverage(cols: dict[str, np.ndarray], n_days: int, C: int) -> dict:
    """覆盖率报告：逐因子非空占比 + **逐日**非空占比（不堆大矩阵，避免 ×F 的额外内存）。

    ★ 逐日覆盖率**整条存进 meta**（一年几百个数，JSON 几十 KB），而不是只存"低于阈值的日子"
      —— 阈值是代码里的常量、会随经验调整，存整数清单会让老数据永远用旧阈值解释
      （踩过一次：门槛从 0.9 改到 0.6，老 meta 仍显示"242/242 天覆盖不足"）。
      逐日序列另有一个用途：查"最后一个覆盖率达标的数据就绪日"。
    """
    if not cols:
        return {"per_factor": {}, "by_day": [], "mean": 0.0, "min": 0.0}
    cnt = np.zeros(n_days, dtype=np.int64)
    per_factor = {}
    for k, v in cols.items():
        fin = np.isfinite(v)
        per_factor[k] = round(float(fin.mean()), 4)
        cnt += fin.reshape(n_days, C).sum(axis=1)
    by_day = cnt / float(C * len(cols))
    return {"per_factor": per_factor,
            "by_day": [round(float(x), 4) for x in by_day],
            "mean": round(float(by_day.mean()), 4), "min": round(float(by_day.min()), 4),
            "first_day_cov": round(float(by_day[0]), 4),
            "last_day_cov": round(float(by_day[-1]), 4)}


def _sha_cols(cols: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(cols):
        h.update(k.encode())
        h.update(np.ascontiguousarray(cols[k]).tobytes())
    return h.hexdigest()[:16]


# ================================================================ 构建一年
def build_year(cfg: Cfg, year: int, days_new: list[str], codes: np.ndarray,
               feats: list[str], labels: list[str], directions: dict[str, int],
               *, jobs: int = DEFAULT_JOBS, log=print, keep: dict | None = None,
               root: Path | None = None) -> dict:
    """构建（或增量重算）一年的 factors/target 并原子落盘。

    `days_new`：**本次要重算**的日子（增量时=窗口内的那几天，不是整年！）；`keep`：要保留在
    **前面**的既有行（`{"days": [...], "factors": {...}, "target": {...}}` —— slot 名直接用
    块名，免得又多一层"内部叫 X、落盘叫 factors"的对照）。
    ★ 增量调用必须传"窗口内"的日子：传整年会把窗口外的旧行算第二遍 → 该年出现重复 (date,code)。
    """
    t0 = time.time()
    from fea.dates import int_to_str_vec

    C = len(codes)
    if not days_new:
        raise SystemExit(f"✘ {year} 年没有任何要计算的日子")
    keep_days = list(keep["days"]) if keep else []
    days = keep_days + list(days_new)
    n_days, n_new = len(days), len(days_new)
    # ★ 不变式：合并后的日子必须**互不重复**（重复 = 增量把窗口外的旧行算了两遍，
    #   后果是行数虚高 + (date,code) 重复 + 下游静默双计 —— 而且 check() 抓不到，
    #   因为它也按同一个虚高的天数算期望行数）。这一条把事故变成当场报错。
    if n_days != len(set(days)):
        dup = n_days - len(set(days))
        raise SystemExit(f"✘ {year} 年合并后有 {dup} 个重复交易日 —— 增量窗口传错了"
                         f"（days_new 必须是窗口内的那些天，不是整年）")
    di_new = np.asarray([int(d[:4]) * 10000 + int(d[5:7]) * 100 + int(d[8:10])
                         for d in days_new], dtype=np.int32)

    codes_index = pd.Index(codes)
    days_index = pd.Index(np.asarray(days_new, dtype=object))
    # ★ get_indexer 的哈希表是**懒建**的：先串行预热一次，再进线程池并发（否则有竞态）
    codes_index.get_indexer(codes[:1])
    days_index.get_indexer(np.asarray(days_new[:1], dtype=object))

    flip_map = {f: (directions.get(f, 1) < 0) for f in feats}
    X, stat_x = _cols(cfg, year, feats, flip_map, "rank", codes_index, days_index, n_new, C,
                      jobs=jobs, what="特征", codes=codes, days_int=di_new, log=log)
    Y, _ = _cols(cfg, year, labels, {l: False for l in labels}, "value", codes_index, days_index,
                 n_new, C, jobs=jobs, what="标签", codes=codes, days_int=di_new, log=log)
    # ★ 2026-09-20：原来这里还算一块 `universe`（股票池掩码）并落盘。删掉的原因见模块
    #   头部 —— 轴就是冻结池本身，掩码实测恒为 True，是恒等块，没有信息。

    # ---- 增量：把窗口之外的既有行拼在前面（两边各自有序 ⇒ 整体有序）
    n_keep = len(keep_days)
    if n_keep:
        for cols, name in ((X, FK), (Y, TK)):
            old = (keep or {}).get(name) or {}
            if set(old) != set(cols):
                raise SystemExit(f"✘ {year} 年 {name} 的既有列与新算列不一致 —— 请全量重建")
            if len(next(iter(old.values()))) != n_keep * C:
                raise SystemExit(f"✘ {year} 年 {name} 的既有行数不等于 {n_keep}×{C} —— 产物坏了，请全量重建")
            for k in cols:
                cols[k] = np.concatenate([old[k], cols[k]])

    n = n_days * C
    di = np.asarray([int(d[:4]) * 10000 + int(d[5:7]) * 100 + int(d[8:10]) for d in days],
                    dtype=np.int32)
    td = pa.array(np.repeat(np.asarray(int_to_str_vec(di), dtype=object), C))
    sc = pa.array(np.tile(np.asarray(codes, dtype=object), n_days))
    sizes = {}
    for kind, cols in ((FK, X), (TK, Y)):
        for k, v in cols.items():
            if len(v) != n:              # ★ 别静默截断：长度不齐会造成 X/Y 逐行错位
                raise SystemExit(f"✘ {year}/{kind} 列 {k} 长度 {len(v)} ≠ {n}（{n_days}×{C}）")
        fields = {"trade_date": td, "stock_code": sc}
        for k, v in cols.items():
            fields[k] = pa.array(v)
        tab = pa.table(fields)
        # ★★ `root` 必须**一路传下来**，不能在这里回落 `root_of(cfg)` ——
        #   2026-09-18 的事故就是这里：`prepare(root=...)` 只重定向了 meta 的写入，
        #   而年分区仍旧按全局配置写 ⇒ 新数据写进了 `trainingdata/`、meta 却写进了新目录，
        #   结果是"数据与清单分家"，而且**一声不响**。
        p = year_file(root if root is not None else root_of(cfg), kind, year)
        atomic_parquet(tab, p)
        sizes[kind] = {"rows": int(tab.num_rows), "cols": int(tab.num_columns),
                       "mb": round(p.stat().st_size / 1e6, 1), "bytes": int(p.stat().st_size),
                       "sha": _sha_cols(cols)}
        del tab, fields
    cov = _coverage(X, n_days, C)
    miss = sorted(k for k, v in stat_x.items() if v["missing_year"])
    dropped = sum(v["dropped"] for v in stat_x.values())
    # ★ 增量时"窗口外"的行**本来就该被丢弃** —— 它们由 `keep` 原样提供，不是脏数据。
    #   不扣掉这一块的话，日跑会天天报"上亿行被丢弃"，真有脏数据时反而被淹没。
    #   （实测：2026 增量跑，dropped = 103,349,475 = 145 天窗口外 × 2115 只 × 337 因子，分毫不差）
    n_have = sum(1 for v in stat_x.values() if not v["missing_year"])
    dirty = dropped - n_keep * C * n_have
    if dirty > 0:
        log(f"    ⚠️ {dirty:,} 行落在日期轴/股票轴之外被丢弃（上游有非交易日或非主板代码的脏行？）")
    log(f"  {year}: {n_days} 天 × {C} 只 = {n:,} 行"
        + (f"（保留 {n_keep} 天 + 新算 {n_new} 天）" if n_keep else "")
        + f" · factors {sizes[FK]['mb']}MB · 覆盖 均{cov['mean']:.3f}/末日{cov['last_day_cov']:.3f}"
        + (f" · ⚠️ {len(miss)} 个因子本年无数据" if miss else "")
        + f" · {round(time.time() - t0, 1)}s")
    return {"year": int(year), "days": n_days, "rows": int(n),
            "start": days[0], "end": days[-1], "files": sizes,
            "coverage": cov, "missing_year": miss, "dropped_rows": dropped}


def _days_of(root: Path, kind: str, year: int) -> list[str]:
    """某年产物里实际出现的交易日（只读日期列，零解压）。"""
    p = year_file(root, kind, year)
    if not p.exists():
        return []
    col = pq.read_table(p, columns=["trade_date"]).column("trade_date").to_pylist()
    return sorted({str(d)[:10] for d in col})

# ================================================================ 读产物


# ================================================================ 指纹
def _universe_fp(cfg: Cfg) -> str:
    """模块② 的**股票池指纹**（含冻结名单的内容哈希）。

    ★ 为什么必须进快照指纹：`rank` 是**当日截面百分位**，股票池变了（3485 → 2115），
      同一天的 rank 全体变化。若只跑增量，就只有窗口内那几十天是新口径、
      其余 15 年还是旧口径 —— **静默不一致**（网格/行数/覆盖率全都正常）。
      模块② 的 `Engine.universe_fp` 正好把这口径（前缀/ST/上市天数/冻结名单哈希）编码成一个串。
    """
    try:
        eng = _engine(str(cfg.factors_root), str(cfg.root))     # noqa: SLF001
        return str(getattr(eng, "universe_fp", "") or "")
    except Exception:                        # noqa: BLE001
        return ""


def _fingerprint(cfg: Cfg, sc: dict, names: list[str], directions: dict) -> dict:
    """上游指纹：因子集合 + 方向表 + 覆盖端点 + 各因子的分区年 + 股票池 + 日台账哈希。

    `per_factor_years` 是用来发现"上游回填了历史"的：单个年份新增分区 → 那一年必须整年重建
    （增量窗口只有最近几十天，盖不住历史回填）。
    `universe_fp` 是用来发现"股票池口径变了"的：变了就必须**全量重建**（见 `prepare` 的判定）。
    """
    h = hashlib.sha256()
    h.update("|".join(sorted(names)).encode())
    h.update("|".join(f"{k}:{directions[k]}" for k in sorted(directions)).encode())
    led = (upstream_fingerprint(cfg).get("ledger") or {})
    h.update(str(led.get("sha256", "")).encode())
    h.update(f"{sc['span'][0]}~{sc['span'][1]}".encode())
    py = {n: sorted(int(y) for y in (v.get("years") or [])) for n, v in sc["per_factor"].items()}
    h.update("|".join(f"{n}:{','.join(map(str, ys))}" for n, ys in sorted(py.items())).encode())
    ufp = _universe_fp(cfg)
    h.update(ufp.encode())
    h.update((ROOT_SCRIPT.read_bytes() if ROOT_SCRIPT.exists() else b""))   # 脚本自身也进指纹
    return {"factors_dir": str(cfg.factors_dir), "fingerprint": h.hexdigest()[:16],
            "universe_fp": ufp,
            "n_products": sc["n_factors"], "first_upstream_day": sc["span"][0],
            "last_upstream_day": sc["span"][1], "ledger": led,
            "per_factor_years": py}


def _years_with_new_partitions(meta_old: dict, sc: dict) -> set[int]:
    """上游哪些年份**新增了分区**（= 回填了历史）—— 增量窗口盖不住，必须整年重建。"""
    old = ((meta_old.get("source") or {}).get("scan") or {}).get("per_factor_years") or {}
    if not old:
        return set()
    grew: set[int] = set()
    for name, v in (sc.get("per_factor") or {}).items():
        o = set(old.get(name) or [])
        grew |= set(v.get("years") or []) - o
    return grew


def derive_feature_sets(meta: dict, min_cov: float = 0.005) -> dict[str, list[str]]:
    """从 meta 已有的**逐年覆盖率**推出"从某年起可用的特征集" —— ★ 零因子侧访问。

    动机：上游 230 个因子的时间覆盖长短不一（40 个晚起，最晚的 `rd_intensity` 到 2019-05），
    所以"从 YYYY 年起可用的因子有哪些"是个每次都要问的问题。与其让每个单元去扫
    `per_factor_years`，不如在初加工时算好写进 meta，单元只写 `DATA={"features": "core_2018"}`。

    ★ 为什么能在**冻结快照**上算：它只用 meta 里已经存着的 `years[*].coverage.per_factor`
      （每个因子在每年的非空占比），不需要读任何上游文件 —— 所以"补这个字段"不等于"重建数据"。

    `core_YYYY` = 在 **YYYY 到最后一个已建年份里每年都有非空值**的特征（覆盖 ≥ `min_cov`）。
    """
    feats = list((meta.get("columns") or {}).get("features") or [])
    years = sorted(int(y) for y in (meta.get("years") or {}))
    if not feats or not years:
        return {}
    out = {"all": feats}
    for y0 in years:
        use = [str(y) for y in years if y >= y0]
        keep = []
        for f in feats:
            ok = True
            for ys in use:
                cov = ((meta["years"].get(ys) or {}).get("coverage") or {}).get("per_factor") or {}
                if float(cov.get(f, 0.0)) < min_cov:
                    ok = False
                    break
            if ok:
                keep.append(f)
        out[f"core_{y0}"] = keep
    # 只保留"相对上一年有变化"的档（把 15 个几乎相同的档压成几档），外加最早的与 all
    slim = {"all": feats}
    prev: list[str] | None = None
    for y0 in years:
        cur = out[f"core_{y0}"]
        if prev is None or len(cur) != len(prev):
            slim[f"core_{y0}"] = cur
        prev = cur
    # ★ `sample_20`（因子抽样块对应的列清单）在这里一并产出，而不是等 `--fac-sample-only`
    #   写进 meta —— 因为 `attach_meta_fields`（`--meta-only`）会**整个覆盖** `feature_sets`，
    #   不在这里生成的话，跑一次 `--meta-only` 就把它抹掉了。
    #   有了它，单元只写 `DATA={"features": "sample_20"}` 就能对 X 做 20 列的快速冒烟，
    #   **引擎完全不需要知道 `fac_sample` 这个物理块的存在**。
    slim["sample_20"] = fac_sample_features(meta)
    return slim


def attach_meta_fields(cfg: Cfg, *, freeze: bool = False, unfreeze: bool = False,
                       reason: str = "", log=print) -> dict:
    """**只补 meta 的派生字段**（特征集、冻结标记），不碰任何数据文件、不读因子侧。

    用途：需要给下游（单元 `DATA={...}`）补料、或改快照的冻结状态时走这条，
    而不是重跑初加工 —— 重跑会换掉 `panel_digest`，让所有已出结论作废。

    ★ `--unfreeze` 也走这里：**解冻不该逼人重建**。日常跑 `preparingdata.py` 做增量
      要求快照不是冻结态，而"解冻"本身只是一次标记翻转。
    """
    root = root_of(cfg)
    meta = load_meta(root)
    if not meta:
        raise SystemExit(f"✘ 还没有产物（{root}）—— 没有 meta 就无从补字段")
    meta["feature_sets"] = derive_feature_sets(meta)
    if unfreeze and (meta.get("frozen") or {}).get("on"):
        old = meta.pop("frozen")
        log(f"    🔓 已解冻（原锚点 panel_digest={((old.get('anchors') or {}).get('panel_digest'))}）"
            f"—— 之后 `preparingdata.py` 可正常跑增量")
    if freeze:
        cur = {k: meta[k] for k in ("panel_digest", "built_at") if k in meta}
        meta["frozen"] = {
            "on": True, "since": now(), "reason": reason or "因子侧要在别处继续开发，模型侧只用当前快照",
            "anchors": {"panel_digest": cur.get("panel_digest"),
                        "upstream_fingerprint": (meta.get("source") or {}).get("fingerprint"),
                        "n_features": len(meta["columns"]["features"]),
                        "n_labels": len(meta["columns"]["labels"]),
                        "axis": [meta["axis"]["start"], meta["axis"]["end"]]}}
    save_json(meta_path(root), meta)
    sizes = {k: len(v) for k, v in meta["feature_sets"].items()}
    log(f"  ✔ meta 派生字段已更新（**未触碰任何数据文件、未读因子侧**）")
    log(f"    特征集：{sizes}")
    if freeze:
        f = meta["frozen"]
        log(f"    🔒 快照已冻结：panel_digest={f['anchors']['panel_digest']} · "
            f"上游指纹={f['anchors']['upstream_fingerprint']}")
        log(f"       要重建请显式 `preparingdata.py --full --unfreeze`")
    return meta


def panel_digest(meta: dict) -> str:
    """面板指纹：列清单 + 逐年文件指纹（写进训练 summary，回答"这一版建在哪份数据上"）。"""
    h = hashlib.sha256()
    h.update("|".join(meta["columns"]["features"]).encode())
    h.update("|".join(meta["columns"]["labels"]).encode())
    for y in sorted(meta.get("years") or {}):
        v = meta["years"][y]
        h.update(f"{y}:{v.get('rows')}:{(v.get('files') or {}).get(FK, {}).get('sha')}".encode())
    return h.hexdigest()[:16]


# ================================================================ 主流程
def prepare(cfg: Cfg, *, mode: str = "auto", years: list[int] | None = None,
            jobs: int = DEFAULT_JOBS, lookback: int | None = None,
            unfreeze: bool = False, root: str | Path | None = None,
            features: list[str] | None = None, direction: dict[str, int] | None = None,
            log=print) -> dict:
    """全量 / 增量 / 指定年份 构建 trainingdata。`mode` ∈ {auto, full, incremental}。

    ★ **快照冻结闸门**：`meta.json:frozen.on` 为真时，本函数直接拒绝——因为重建会把
      "上游当前的样子"煮进快照，而那个样子可能正在被别处改（见 README「数据快照冻结」）。
      要显式重建得加 `--unfreeze`。只补 meta 派生字段走 `attach_meta_fields`（不碰数据）。
    """
    t_all = time.time()
    # ★ `root` 允许**另建一份快照**（不进原地）。为什么需要它：现行 `trainingdata/` 是
    #   **冻结快照**，所有已出结论都锚在它的 `panel_digest` 上；上游来了新因子时
    #   若原地重建，旧结论全部作废、且**没有对照组**。另建一份才能做
    #   "同一套配方、只换特征集"的干净 A/B（见 README §3「数据快照冻结」）。
    root = Path(root).resolve() if root else root_of(cfg)
    if root != root_of(cfg):
        log(f"  ★ 本次写到**另一份快照**：{root}（现有快照 {root_of(cfg)} 不动）")
    meta_old = load_meta(root)
    fr = meta_old.get("frozen") or {}
    if fr.get("on") and not unfreeze:
        a = fr.get("anchors") or {}
        raise SystemExit(
            "✘ trainingdata 快照已冻结，拒绝重跑初加工。\n"
            f"    锚点：panel_digest={a.get('panel_digest')} · 上游指纹={a.get('upstream_fingerprint')}\n"
            f"    冻结于 {fr.get('since')}：{fr.get('reason')}\n"
            "    · 只想补 meta 派生字段（特征集等）→ `python preparingdata.py --meta-only`\n"
            "    · 确实要按上游**当前**的样子重建 → `python preparingdata.py --full --unfreeze`\n"
            "      （重建后面板指纹会变，所有基于旧快照的结论都需要重跑）")
    if unfreeze and fr.get("on"):
        log("  ⚠️ --unfreeze：本次将按上游当前状态重建，旧的 panel_digest 锚点随即失效")
    force_full = (mode == "full") or not meta_old
    log(f"══ 初加工 trainingdata{'（全量）' if mode == 'full' else ''}")
    log(f"  产物：{root}")

    sc = scan(cfg, log=log)
    specs = upstream_specs(cfg)
    feats = sorted(n for n, v in specs.items() if not v["is_label"])
    labels = sorted(n for n, v in specs.items() if v["is_label"])
    directions = {n: (1 if v["higher_is_better"] else -1) for n, v in specs.items()}
    # ★ `features` 非空 = **锁定特征清单**（`--features-from`）：只收清单里的列，且顺序照它。
    #   两个用途：① 与旧产物逐位对拍（上游因子集变了也还能比）；② 冻结特征集不随上游漂移。
    if features:
        miss = [f for f in features if f not in specs]
        if miss:
            raise SystemExit(f"✘ --features-from 里有 {len(miss)} 个因子在当前上游不存在：{miss[:5]}")
        if direction:
            # 方向表变了 = 那个特征的符号翻了 —— 对拍会全线不一致，且原因极难查。当场报。
            diff = {f: (direction.get(f), directions[f]) for f in features
                    if f in direction and direction[f] != directions[f]}
            if diff:
                raise SystemExit(f"✘ 方向表与清单不一致（{len(diff)} 个）：{list(diff.items())[:3]}\n"
                                 f"    —— 上游改了 higher_is_better，锁清单也复现不出旧产物")
        feats = list(features)
        labels = [l for l in labels if l in set(specs)] or labels
    lo, hi = sc["span"]
    if not lo:
        raise SystemExit("✘ 上游没有任何因子产物")
    log(f"  特征 {len(feats)} · 标签 {len(labels)} · 需翻转 "
        f"{sum(1 for f in feats if directions[f] < 0)}（这些存 1−rank，语义统一成「越大越好」）")

    days, codes = build_axis(cfg, lo, hi, log=log)
    by_year: dict[int, list[str]] = {}
    for d in days:
        by_year.setdefault(int(d[:4]), []).append(d)
    lb = int(lookback) if lookback is not None else lookback_for(cfg, labels)
    log(f"  增量回溯窗口：{lb} 个交易日"
        f"（= 最长标签 h + 1 + 冗余 {REDUNDANT_DAYS}；h+1 保证标签定型，冗余吃掉上游小幅回改）")

    if years:
        want = sorted({int(y) for y in years} & set(by_year))
        if not want:
            raise SystemExit(f"✘ --years {years} 与日期轴 {min(by_year)}~{max(by_year)} 无交集")
        by_year = {y: by_year[y] for y in want}
    built = sorted(by_year)
    fp = _fingerprint(cfg, sc, feats + labels, directions)
    have_years = {int(y) for y in (meta_old.get("built_years") or [])}

    # ---- 决定做什么
    plan: dict[int, tuple[str, list[str] | None]] = {}
    if force_full or years:
        plan = {y: ("full", None) for y in built}
        why = "全量重建" if force_full else f"指定年份 {built}"
    else:
        same_fp = fp["fingerprint"] == ((meta_old.get("source") or {}).get("fingerprint"))
        last_old = (meta_old.get("axis") or {}).get("last_built_day")
        if same_fp and last_old and hi <= str(last_old):
            log(f"  ⊘ 无新增：上游最新日 {hi} ≤ 已建 {last_old}，且上游指纹未变 → 跳过（零 I/O）")
            # ★ 布局里的两个**可选块**（amount / fac_sample）走**独立计划轴**，不能跟着这里
            #   一起早退（见 `stale_block_years`）：它们与因子侧的新增无关，可能只是还没建过。
            ensure_block(cfg, AK, root, log=log)
            ensure_block(cfg, SK, root, log=log)
            return {"action": "skip", "reason": "无新增", "rebuilt": {},
                    "meta": load_meta(root) or meta_old}
        # ★★ 股票池口径变了（如 2026-09-18 的 3485 → 2115 冻结名单）必须**全量重建**：
        #    rank 是当日截面百分位 ⇒ 池子一换，每天的值全体变化。若只跑增量，
        #    窗口内那几十天是新口径、其余 15 年还是旧口径 —— 网格/行数/覆盖率**全都正常**，
        #    没有任何体检能发现，只能靠这里挡住。
        uni_old = (meta_old.get("source") or {}).get("universe_fp")
        uni_now = fp.get("universe_fp")
        missing = [y for y in built if y not in have_years]
        grew = _years_with_new_partitions(meta_old, sc)     # 上游回填了哪些年
        # ★ 判定用 `uni_old != uni_now`（**缺锚点也算变了**）：老 meta 里没有这个字段就
        #   无法证明口径没变，宁可全量重建也不要留下"半新半旧"的快照。
        if uni_now and uni_old != uni_now:
            plan = {y: ("full", None) for y in built}
            why = (f"★ 股票池口径变了（或旧 meta 缺锚点）→ 全量重建\n"
                   f"    universe_fp: {uni_old[-30:] if uni_old else '（旧 meta 无此字段）'}\n"
                   f"              → {uni_now[-30:]}\n"
                   f"    （增量只会让窗口内那几天变成新口径，其余年份静默留在旧口径）")
        else:
            idx = {d: i for i, d in enumerate(days)}
            j_last = idx.get(str(last_old))
            if j_last is None:                   # 上次末日不在新轴上（上游回撤过数据）
                j_last = max((i for d, i in idx.items() if d <= str(last_old)), default=0)
            win_lo = days[max(0, j_last - lb)]
            for y in built:
                tail = [d for d in by_year[y] if d >= win_lo]
                if tail:
                    plan[y] = ("incr", tail)
            # ★ 上游**回填历史**（新增年份分区）时窗口盖不住 → 那些年份整年重建
            for y in (set(missing) | set(grew)):
                plan[y] = ("full", None)
            why = (f"增量：窗口 {win_lo} ~ {hi}（只重写受影响的年份）"
                   + (f"；{len(missing)} 个年份从未建过 → 整年重建" if missing else "")
                   + (f"；上游新增分区 {sorted(grew)} → 整年重建" if grew else ""))

    log(f"  ▸ {why}")
    if not plan:
        log("  ⊘ 没有需要处理的年份")
        return {"action": "none", "rebuilt": {}, "meta": meta_old}

    # ---- 执行
    years_meta = dict(meta_old.get("years") or {})
    for y in sorted(plan):
        act, tail = plan[y]
        keep = None
        if act == "incr":
            keep = _load_keep(cfg, y, tail[0], feats, labels, log=log, root=root)
            if keep is None:
                act, tail = "full", None
        # ★★ 增量时必须只传**窗口内的那些天**（`tail`），绝不能传整年（`by_year[y]`）：
        #    keep 里已经有窗口外的旧行，再算一遍整年 → 那些天在文件里出现两次
        #    （行数虚高、(date,code) 重复、下游静默双计）。这个坑踩过一次，别再踩。
        info = build_year(cfg, y, tail if act == "incr" else by_year[y], codes, feats, labels,
                          directions, jobs=jobs, log=log, keep=keep, root=root)
        info["action"] = act
        # ★★ `build_year` **只产出 `factors` / `target`** —— 这一年的 `amount` / `fac_sample`
        #   文件还在原地没动（它们走独立计划轴），所以它们在该年 `files` 里的记录（sha/bytes）
        #   必须**从旧 meta 搬过来**。整年记录被 `info` 顶替掉的话，这两块的指纹就没了，
        #   而 `ensure_block` 只补"文件陈旧/缺失"的块 —— 文件好好的，它不会补，于是**静默丢失**。
        #   （2026-09-20 实测踩到：跑一次日常增量后 `years.2026.files.amount` 就成了空。）
        _old_files = (years_meta.get(str(y)) or {}).get("files") or {}
        info["files"] = {**{k: v for k, v in _old_files.items() if k not in info["files"]},
                         **info["files"]}
        years_meta[str(y)] = info

    # ---- meta（★ 最后写）
    kept = [str(y) for y in sorted(int(k) for k in years_meta)]
    meta = {
        "version": META_VERSION,
        "built_at": now(),
        "source": {**fp, "scan": {"n": sc["n_factors"], "partition_years": sc["years"],
                                  "per_factor_years": fp["per_factor_years"]}},
        "columns": {"features": feats, "labels": labels,
                    "direction": {n: directions[n] for n in feats},
                    "n_flipped": sum(1 for f in feats if directions[f] < 0)},
        "axis": {"start": min(v["start"] for v in years_meta.values()),
                 "end": max(v["end"] for v in years_meta.values()),
                 "n_days": len(days), "n_codes": len(codes),
                 "codes": [str(c) for c in codes],
                 "first_built_day": min(v["start"] for v in years_meta.values()),
                 "last_built_day": max(v["end"] for v in years_meta.values())},
        "years": years_meta, "built_years": kept, "partial": bool(years),
        "seconds": round(time.time() - t_all, 1),
    }
    # ★★ 上面这个 meta 是**从零重建**的，而 `amount` / `fac_sample` 的记录是事后由
    #   `_record_block` 单独写进去的（它们走独立计划轴，见下）—— 不搬过来的话，
    #   **任何一次 `prepare()`（含日常增量）都会把这两块的记录抹掉**；随后 `ensure_block`
    #   只补"文件陈旧"的块，文件没陈旧就一声不吭，于是"这块数据是哪一版"的指纹**静默消失**。
    #   （2026-09-20 实测踩到：跑一次日常增量后 meta 顶层就没这两个 key 了。）
    #   先原样搬过来，下面 `ensure_block` 真重建时会用新的 `_record_block` 覆盖掉。
    for _k in ALL_KINDS:
        if _k in meta_old:
            meta[_k] = meta_old[_k]

    meta["feature_sets"] = derive_feature_sets(meta)      # 单元 DATA={"features":"core_2018"} 用
    meta["panel_digest"] = panel_digest(meta)
    save_json(meta_path(root), meta)
    log(f"  ✔ meta 已写（本次 {meta['seconds']}s）")

    # ★★ 两个**可选块**（amount / fac_sample）走**独立计划轴**，且必须在主 meta 写完之后
    #   —— 它们会再读一次 meta、补上 `years[y].files.<块>` 与顶层同名 key 后写回，
    #   放在前面会被这里覆盖。为什么不能塞进上面的 plan：`prepare()` 的"无新增"快路径
    #   会直接 return，而这两块与因子侧没有共同的失效条件（见 `stale_block_years`）。
    ensure_block(cfg, AK, Path(root), log=log)
    ensure_block(cfg, SK, Path(root), log=log)

    _summary(meta, log=log)
    # ★ `rebuilt` 交回给调用方喂台账：`{"年": "full"/"incr"}` —— 只有**本次真的重建了**
    #   的年份在里面。别看 `meta["years"][y]["action"]`，那个对没重建的年份是上一次的旧值。
    return {"action": "built", "years": kept, "meta": meta,
            "rebuilt": {(k, int(y)): plan[y][0] for y in plan for k in (FK, TK)}}


def _load_keep(cfg: Cfg, year: int, win_lo: str, feats: list[str], labels: list[str],
               log=print, root: Path | None = None) -> dict | None:
    """增量：把增量窗口之外的既有行读回来（拼在新数据前面）。没有就返回 None（→ 整年重建）。

    ★ `root` 与 `build_year` 同源：读旧行与写新行**必须指向同一个目录**，
      否则会出现"从 A 读旧行 + 往 B 写新行"这种半新半旧的结果。
    """
    root = root if root is not None else root_of(cfg)
    if not year_file(root, FK, year).exists():
        return None
    keep: dict = {FK: {}, TK: {}, "days": []}
    n_rows = None
    for kind, cols in ((FK, feats), (TK, labels)):
        df = read_frame(root, kind, years=[year], columns=cols, end=_prev_day(win_lo))
        if df.empty:
            return None
        keep[kind] = {c: df[c].to_numpy() for c in df.columns if c not in KEY}
        # ★ 两块产物必须**行数一致**，否则拼回去就是逐行错位（比缺数据危险得多）
        if n_rows is None:
            n_rows = len(df)
            keep["days"] = sorted(set(df["trade_date"].astype(str)))
        elif len(df) != n_rows:
            raise SystemExit(f"✘ {year} 年 {kind} 的既有行数 {len(df):,} ≠ {FK} 的 {n_rows:,}"
                             f" —— 产物坏了，请 `preparingdata.py --full` 重建")
        del df
    log(f"    保留 {year} 年窗口前的 {len(keep['days'])} 天（< {win_lo}）")
    return keep


def _prev_day(day: str) -> str:
    """字符串日期减一天（只用于过滤，不必是交易日）。"""
    import datetime as _dt
    d = _dt.date(int(day[:4]), int(day[5:7]), int(day[8:10])) - _dt.timedelta(days=1)
    return d.isoformat()


# ================================================================ 校验
def check(cfg: Cfg, *, root: str | Path | None = None, log=print) -> dict:
    """只校验不写：meta ↔ 文件 ↔ 上游（单条记录级体检，不做跨因子深度分析）。

    `root` 指定要校验的快照（配 `preparingdata.py --check --out DIR`）——
    另建的快照要能**先体检再换上去**，否则只能盲切。
    """
    root = Path(root).resolve() if root else root_of(cfg)
    meta = load_meta(root)
    if not meta:
        log("  ⊘ 还没有产物 —— 先跑 `python preparingdata.py`（首次会自动全量）")
        return {"ok": False, "problems": ["没有 meta.json"], "years": []}
    built = meta.get("built_years") or []
    problems: list[str] = []
    log(f"══ 校验 {root}（{len(built)} 年：{built[0]} ~ {built[-1]}）")
    sc = scan(cfg, log=log)
    ax = meta.get("axis") or {}
    lo_now, hi_now = sc["span"]
    if lo_now and str(lo_now) < str(ax.get("start")):
        problems.append(f"上游已回溯到 {lo_now} < 产物起点 {ax.get('start')} —— 需要 --full 扩展")
    if hi_now and str(hi_now) > str(ax.get("last_built_day")):
        problems.append(f"上游最新日 {hi_now} > 已建 {ax.get('last_built_day')} —— 需要跑增量")
    rows_total = 0
    C = int(ax.get("n_codes") or 0)
    feats = list((meta.get("columns") or {}).get("features") or [])
    for y in built:
        info = meta["years"][y]
        exp = int(info["days"]) * C
        for kind in ALL_KINDS:
            p = year_file(root, kind, int(y))
            if not p.exists():
                # 四块都必建。缺 `amount`/`fac_sample` 时用 `--amount-only` / `--fac-sample-only` 补。
                problems.append(f"{y}/{kind}: 文件缺失")
                continue
            pf = pq.ParquetFile(p)
            if pf.metadata.num_rows != exp:
                problems.append(f"{y}/{kind}: 行数 {pf.metadata.num_rows:,} ≠ 期望 {exp:,}")
            # ★ 字节数核对：meta 是**最后**写的，所以"X 写完了、Y 没写完"这种半成品
            #   会让 meta 与磁盘对不上（只比行数是抓不到的）
            want_b = ((info.get("files") or {}).get(kind) or {}).get("bytes")
            if want_b and p.stat().st_size != want_b:
                problems.append(f"{y}/{kind}: 文件字节 {p.stat().st_size} ≠ meta 记录 {want_b}"
                                f"（上次构建中途失败过？重跑 preparingdata.py）")
            # ★★ 逐日行数核对（只读一列）：必须**每个交易日恰好 C 行**。
            #   这是"增量把窗口外旧行算了两遍"的**唯一现场证据** —— 行数总和与
            #   meta 里虚高的天数自洽，比行数/比字节都抓不到它（真踩过）。
            #   只查"与 factors 逐日同形"的那几块 —— 这是四块的共同契约。
            if kind in ALL_KINDS and C:
                td = pq.read_table(p, columns=["trade_date"]).column("trade_date").to_pylist()
                cnt = pd.Series(td).value_counts()
                off = cnt[cnt != C]
                if len(off):
                    problems.append(f"{y}/{kind}: {len(off)} 个交易日的行数 ≠ {C}"
                                    f"（例 {off.index[0]} 有 {int(off.iloc[0])} 行）"
                                    f" —— 增量重复写入或产物损坏，请 --full 重建")
        cov = info["coverage"]
        ycols = list(cov["per_factor"])
        # 因子"全年无值"只有在**它本该有值**时才算问题：比对该因子的起点年
        # （晚起点因子在更早的年份里整列为空是正常的，不该刷屏）
        first_y = {n: (min(ys) if ys else 9999)
                   for n, ys in ((meta.get("source") or {}).get("scan") or {})
                   .get("per_factor_years", {}).items()}
        bad = sorted(k for k, v in cov["per_factor"].items()
                     if v == 0.0 and int(y) >= first_y.get(k, 0))
        early = sorted(k for k, v in cov["per_factor"].items()
                       if v == 0.0 and int(y) < first_y.get(k, 0))
        if bad:
            problems.append(f"{y}: {len(bad)} 个因子全年无值、但它们的起点更早"
                            f"（例：{bad[:3]}）—— 上游这一年的分区是不是少的？")
        by_day = cov.get("by_day") or []
        floor = max(LOW_COV_ABS, LOW_COV_REL * float(cov["mean"]))
        low = [i for i, r in enumerate(by_day) if r < floor]
        if low:
            problems.append(f"{y}: {len(low)}/{info['days']} 天的特征覆盖率 <{floor:.2f}"
                            f"（当年均值 {cov['mean']:.3f}；首日索引 {low[0]}）"
                            f"—— 上游这几天是不是写盘写了一半？")
        log(f"  {y}: {info['days']} 天 × {C} 只 = {exp:,} 行 · 覆盖 均{cov['mean']:.3f} "
            f"末日{cov['last_day_cov']:.3f} · 因子 {len(ycols)} 个"
            + (f"（另有 {len(early)} 个晚起点因子本年无值，正常）" if early else "")
            + (f" · ⚠️ {len(low)}/{info['days']} 天低于 {floor:.2f}" if low else ""))
        rows_total += exp
    # ---- 取值域抽查（★ 这条是"最贵的错"的廉价探测器）
    # 只抽若干列：X 必须是 [0,1] 的截面百分位（翻转过的是 1−rank，仍在 [0,1]）。
    # 历史教训：V1 时代 `clip:[0.001,0.999]` 作用在 `−rank`（值域 [−1,0]）上，把 83 个被翻转的
    # 特征整列压成常数 0.001 —— 37% 的特征等于没有，而当时所有体检都没发现。
    probe = feats[:: max(1, len(feats) // 8)][:8] if feats else []
    if probe:
        bad = []
        for y in built:
            t = pq.read_table(year_file(root, FK, int(y)), columns=probe)
            for c in probe:
                a = t[c].to_numpy(zero_copy_only=False)
                fin = np.isfinite(a)
                if not fin.any():
                    continue
                lo_v, hi_v = float(a[fin].min()), float(a[fin].max())
                if lo_v < -1e-6 or hi_v > 1 + 1e-6:
                    bad.append(f"{y}/{c}: [{lo_v:.4f}, {hi_v:.4f}]")
        if bad:
            problems.append(f"特征取值超出 [0,1]：{bad[:4]}（方向翻转/裁剪口径有问题？）")
        log(f"  ▸ 取值域抽查：{len(built)} 年 × {len(probe)} 列 · "
            + ("✔ 全部落在 [0,1]" if not bad else f"✘ {len(bad)} 列越界"))

    log(f"  ▸ factors 共 {rows_total:,} 行 · 面板指纹 {meta.get('panel_digest')}")
    for p in problems:
        log(f"  ✘ {p}")
    log("  ✔ 校验通过" if not problems else f"  ⚠️ {len(problems)} 处需要处理")
    return {"ok": not problems, "problems": problems, "years": built, "meta": meta}


def _summary(meta: dict, log=print) -> None:
    ym = meta["years"]
    tot = sum(v["rows"] for v in ym.values())
    xmb = sum((v.get("files") or {}).get(FK, {}).get("mb", 0) for v in ym.values())
    log("─" * 62)
    log(f"  trainingdata 就绪：{len(ym)} 年 · {tot:,} 行 · factors {xmb:.0f} MB")
    log(f"  轴 {meta['axis']['start']} ~ {meta['axis']['end']} · "
        f"{meta['axis']['n_days']} 天 × {meta['axis']['n_codes']} 只")
    log(f"  特征 {len(meta['columns']['features'])} · 标签 {len(meta['columns']['labels'])} · "
        f"翻转 {meta['columns']['n_flipped']} · 指纹 {meta['panel_digest']}")
    log("  下一步：python main.py split <单元>    # 零成本看训练/验证/测试窗口")


# ================================================================ 台账
#: 台账目录（在快照根下）。★ 放在 `trainingdata/` 里而不是模块根 ——
#: 台账记的是**这份快照**的状态，快照搬走/另建时它应该跟着走。
LEDGER_DIR = "log"


def write_ledger(root: str | Path, *, action: str,
                 rebuilt: dict[tuple[str, int], str] | None = None, log=print) -> Path:
    """把快照里**每个数据文件的当前状态**写成一份**带日期的台账**（TSV）。

    用户 2026-09-20 要的：每天增量更新时生成一个新文件，好回答"某天那份快照长什么样"。
    - **一天一个文件**：`log/ledger_YYYY-MM-DD.tsv`。同日多次运行就覆盖 —— 要的是"最新台账"，
      不是运行流水。
    - **一行一个数据文件**（四块 × 九年 = 36 行），列取自 `meta.json:years[*].files[*]`；
      `mtime` 取自文件本身，所以一眼能看出**今天动了哪几个**。
    - 为什么 TSV 不是 JSON：好 `diff` 好 `grep`，与上游 `dayhash.tsv` 同路子。
    - ★ 由 `main()` 调，**不放进 `prepare()`** —— 台账是**运行记录**，运行从 CLI 来；
      放 `prepare()` 里会让探针/程序化调用也写（字节对拍时一天要跑几十次 `prepare`）。

    `action` / `rebuilt` 必须由调用方传**本次运行**的信息，不能读
    `meta["years"][y]["action"]` —— 那个只对本次重建的年份是新的，其余年份留着**上一次**的旧值
    （实测：日常增量跑完，`years.2018.action` 仍写着 `'full'`，那是几天前全量构建留下的）。

    ★ `rebuilt` 的键是 **`(块名, 年)`**，不是只按年 —— `--amount-only` 只重建 amount 块，
      若只按年记，台账会把 `amount` 盖到那一年的 factors/target 行上，看着像它们也被重写了。
    """
    root = Path(root)
    meta = load_meta(root)
    if not meta:
        raise SystemExit(f"✘ {root} 没有 meta.json，写不了台账")
    rb = {(str(k), int(y)): v for (k, y), v in (rebuilt or {}).items()}
    d = root / LEDGER_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"ledger_{datetime.now().strftime('%Y-%m-%d')}.tsv"

    head = [
        f"# trainingdata 台账 · 生成 {now()}",
        f"# panel_digest={meta.get('panel_digest')}"
        f"  上游指纹={(meta.get('source') or {}).get('fingerprint')}",
        f"# 本次动作={action or '-'}"
        + (f"  重建={sorted({y for _k, y in rb})}" if rb else ""),
        "\t".join(("block", "year", "rows", "cols", "bytes", "sha", "mtime", "action")),
    ]
    body: list[str] = []
    for kind in ALL_KINDS:
        for y in sorted(int(v) for v in (meta.get("years") or {})):
            rec = ((meta["years"].get(str(y)) or {}).get("files") or {}).get(kind) or {}
            f = year_file(root, kind, y)
            if not rec and not f.exists():
                continue
            mt = (datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                  if f.exists() else "-")
            body.append("\t".join((kind, str(y), str(rec.get("rows", "-")), str(rec.get("cols", "-")),
                                   str(rec.get("bytes", "-")), str(rec.get("sha", "-")), mt,
                                   rb.get((kind, y), "-"))))
    p.write_text("\n".join(head + body) + "\n", encoding="utf-8")
    log(f"  ✔ 台账已写：{p}（{len(body)} 个文件"
        + (f" · 本次重建 {sorted({y for _k, y in rb})}" if rb else "") + "）")
    return p


# ============================================================================
# 5. 可选块：`amount`（每日成交额）与 `fac_sample`（因子抽样）
# ============================================================================
#: 每日成交额（元）。原注释写"策略用来限制单票仓位"，但那条路径至今没实现 ——
#: 用户 2026-09-20 明确要留它（"市场所有的流动性"），所以留着，别当死代码删。
AK = "amount"
#: 因子抽样块 = 从 `factors` 的特征里随机抽 N 列
SK = "fac_sample"
AMOUNT_COLUMNS = ("amount",)          # 除 KEY 外的列
FAC_SAMPLE_N = 20                   # 抽多少列
FAC_SAMPLE_SEED = 42                # 固定种子 —— 抽样结果必须可复现，否则交付给别人对不上

#: 布局里的**全部四块**，缺一不可（2026-09-20 起 `amount` / `fac_sample` 也是正式成员）。
ALL_KINDS = tuple(KINDS) + (AK, SK)


def fac_sample_features(meta: dict, n: int = FAC_SAMPLE_N,
                          seed: int = FAC_SAMPLE_SEED) -> list[str]:
    """从 `factors` 的特征清单里**用固定种子**随机抽 n 列 —— 纯函数，谁算都一样。

    ★ 抽样在**排序后的**清单上做：`upstream_specs` 出来本来就是排序的，但这里再排一次，
      免得以后有人改了上游的遍历顺序，抽样结果就跟着变（那就不可复现了）。
    """
    feats = sorted(meta["columns"]["features"])
    rng = np.random.default_rng(seed)
    k = min(int(n), len(feats))
    return sorted(rng.choice(np.asarray(feats, dtype=object), size=k, replace=False).tolist())


def _block_digest(years_meta: dict, years: list[int], kind: str) -> str:
    """某一块的指纹 —— 各年该块文件 sha 按年拼起来哈希。

    ★ 为什么每块要自己的一份：块的更新是**独立**的（`--amount-only` 只动 amount）。
      没有这一项，块更新后没人能证明"这两个回测数字是不是同一份额度数据"。
      它**不参与 `panel_digest`** —— 后者只哈希 columns + 逐年 `files[FK].sha`，见模块头部。
    """
    h = hashlib.sha256()
    for y in sorted(years):
        f = ((years_meta.get(str(y)) or {}).get("files") or {}).get(kind) or {}
        h.update(f"{y}:{f.get('sha')}".encode())
    return h.hexdigest()[:16]


def _record_block(meta: dict, kind: str, years: list[int], extra: dict, root: Path,
                  log=print) -> dict:
    """把一块的清单写进 meta 并落盘（**不动 `panel_digest`**）。"""
    meta[kind] = {"years": sorted(int(y) for y in years),
                  "built_at": now(),
                  "digest": _block_digest(meta.get("years") or {}, years, kind), **extra}
    save_json(meta_path(root), meta)
    log(f"  ✔ {kind} 块完成，meta 已更新（panel_digest 保持 {meta.get('panel_digest')}，"
        f"{kind} 指纹 {meta[kind]['digest']}）")


def _write_block_year(root: Path, kind: str, year: int, days: list[str],
                      codes: np.ndarray, cols: dict[str, np.ndarray]) -> dict:
    """把一块的**一年**写成 `year=YYYY/data.parquet`（与 X 逐日同形）。

    ★ 按 `(trade_date, stock_code)` 显式对齐后再落格，**不按行号**。
      踩过的形状：上游价格层按列号落格，两轴不一致时甲的价会**静默**写进乙的列
      （2026-09-18 换池时真发生过，3484 只 vs 2115 只）。这里一律先映射索引。
    ★ 日期直接用 `days`（它已经是规范的 `YYYY-MM-DD`），**不走 `fea.dates`** ——
      于是 `--amount-only` / `--fac-sample-only` 对上游因子侧是**零依赖**的，
      因子工程不在场也能补这两块。
    """
    n_days, C = len(days), len(codes)
    n = n_days * C
    for k, v in cols.items():
        if len(v) != n:
            raise SystemExit(f"✘ {year}/{kind} 列 {k} 长度 {len(v)} ≠ {n}（{n_days}×{C}）")
    fields: dict = {"trade_date": pa.array(np.repeat(np.asarray(days, dtype=object), C)),
                    "stock_code": pa.array(np.tile(np.asarray(codes, dtype=object), n_days))}
    for k, v in cols.items():
        fields[k] = pa.array(v)
    tab = pa.table(fields)
    p = year_file(root, kind, year)
    atomic_parquet(tab, p)
    info = {"year": int(year), "days": n_days, "rows": int(n),
            "start": days[0], "end": days[-1],
            "files": {kind: {"rows": int(tab.num_rows), "cols": int(tab.num_columns),
                             "mb": round(p.stat().st_size / 1e6, 1),
                             "bytes": int(p.stat().st_size), "sha": _sha_cols(cols)}}}
    del tab, fields
    return info


# ---------------------------------------------------------------- amount
def build_year_amount(cfg: Cfg, year: int, days: list[str], codes: np.ndarray,
                      cols: tuple[str, ...], *, log=print, root: Path | None = None) -> dict:
    """导出**一年**的 `amount/year=YYYY/data.parquet` —— 每日成交额（元），用户口径叫"市场所有的流动性"。

    ## 为什么直读模块① 而不走价格层

    用户口径是"**去数据端拿**"。价格层（`fea.PriceLayer`）也能给出 `amount`，但它要先
    把 2011 年起的三张原始表全部加载进内存，而且 `_ensure_loaded` 是**单调扩窗**的 ——
    逐年推进等于把越来越大的 frame 反复重扫（实测 15 年累计约 9 分钟，O(年数²)）。
    成交额这一列就是 `stock_daily.amount`，直读一年一个 parquet，秒级。

    ★ `amount` 是 **FLOWS** 类字段（上游 `fea/prices.py` 的语义）：**不做前向填充**。
      停牌日没有成交就是 NaN —— 对"这天最多能买多少"这个问题，NaN 才是正确答案，
      填成 0 或填成昨天的值都会让策略以为"还有额度"。
    ★ 存 **float64**：金额量级到 1e10，float32 只有 7 位有效数字，会丢到元以下不稳。
    """
    t0 = time.time()
    C = len(codes)
    if not days:
        raise SystemExit(f"✘ {year} 年没有交易日，无法导出 amount 块")
    p = Path(cfg.raw_data) / "stock_daily" / f"year={int(year)}" / "data.parquet"
    if not p.exists():
        raise SystemExit(f"✘ 模块① 没有 {year} 年的 stock_daily：{p}\n"
                         f"    （`MX_RAW_DATA` 指错了吧？当前 = {cfg.raw_data}）")
    df = pd.read_parquet(p, columns=["stock_code", "trade_date", "amount"])
    df["trade_date"] = df["trade_date"].astype(str).str[:10]
    df["stock_code"] = df["stock_code"].astype(str)

    days_index = pd.Index(np.asarray(days, dtype=object))
    codes_index = pd.Index(codes)
    di = days_index.get_indexer(df["trade_date"].to_numpy(dtype=object))
    ci = codes_index.get_indexer(df["stock_code"].to_numpy(dtype=object))
    ok = (di >= 0) & (ci >= 0)
    keep = di[ok] * C + ci[ok]
    if len(np.unique(keep)) != len(keep):
        raise SystemExit(f"✘ {year}/amount：模块① 里有重复的 (交易日, 股票) 行 —— "
                         f"落格会静默覆盖，先查上游")
    out = np.full(len(days) * C, np.nan, dtype=np.float64)
    out[keep] = df["amount"].to_numpy(dtype=np.float64)[ok]
    dropped = int((~ok).sum())
    info = _write_block_year(root if root is not None else root_of(cfg), AK, year, days, codes,
                             {cols[0]: out})
    n_fin = int(np.isfinite(out).sum())
    log(f"  {year}: amount 块 {info['days']} 天 × {C} 只 = {info['rows']:,} 行 · "
        f"非空 {n_fin:,}（{n_fin / max(1, info['rows']):.1%}） · "
        f"{info['files'][cols[0]]['mb']}MB"
        + (f" · 丢弃 {dropped:,} 行（日期/代码轴之外）" if dropped else "")
        + f" · {round(time.time() - t0, 1)}s")
    del df
    return info


class Block:
    """一个**派生块**的登记项 —— 只装"块特有"的三件事。

    通用流程（陈旧判定、逐年循环、写 meta）在 `stale_block_years` / `build_block` 里。
    两个块的 `stale_*` / `build_*` / `_ensure_*` 原先各写一遍、几乎逐字相同，合并到这里。
    """

    __slots__ = ("kind", "title", "cols", "build_year", "meta_extra")

    def __init__(self, kind: str, title: str, cols, build_year, meta_extra) -> None:
        self.kind = kind              #: 块名（同时是 meta 顶层 key）
        self.title = title            #: 日志里的中文名
        self.cols = cols              #: (meta) -> 列清单；fac_sample 依赖 meta 的特征清单
        self.build_year = build_year  #: (cfg, year, days, codes, cols, *, log, root) -> info
        self.meta_extra = meta_extra  #: (meta, cols) -> 写进 meta 的额外字段


def stale_block_years(root: str | Path, kind: str, *, log=None) -> list[int]:
    """某块的**陈旧年份** —— 两个块共用一套判据：

    ① 文件不存在；
    ② 交易日集合与 `factors` 不一致（四块的共同契约就是"与 factors 逐日同形"）；
    ③ 列清单与期望不一致 —— 这条是为 schema 演进留的：精度从 float32 提到 float64
       这类改动只比交易日集合抓不到（产物看着"行数对得上"，消费端读列时才炸）。
    """
    root = Path(root)
    meta = load_meta(root)
    blk = BLOCKS[kind]
    want = tuple(KEY) + tuple(blk.cols(meta)) if meta.get("columns") else ()
    out = []
    for y in years_of(root, FK):
        p = year_file(root, kind, y)
        if not p.exists() or set(_days_of(root, FK, y)) != set(_days_of(root, kind, y)):
            out.append(y)
            continue
        if want and tuple(pq.ParquetFile(p).schema_arrow.names) != want:
            out.append(y)
    if log and out:
        log(f"  {blk.title}：{len(out)} 个年份需要（重）建 —— {out}")
    return sorted(out)


def build_block(cfg: Cfg, kind: str, *, years: list[int] | None = None, log=print,
                root: str | Path | None = None) -> dict:
    """导出某一块（全部年份或指定年份）。

    ★ **不动 `factors`/`target`，因此不改 `panel_digest`** —— 已有结论与配方全部保持有效。
      年份与日子一律**从既有 `factors` 分区读**（不重算日期轴），保证四块严格同形。
    """
    root = Path(root).resolve() if root else root_of(cfg)
    meta = load_meta(root)
    if not meta:
        raise SystemExit(f"✘ {root} 还没有 meta.json —— 先跑一次完整的 `preparingdata.py`")
    blk = BLOCKS[kind]
    cols = tuple(blk.cols(meta))
    keys = sorted(int(y) for y in (meta.get("built_years") or []))
    want = sorted(set(int(y) for y in years) & set(keys)) if years else keys
    if years is not None and not want:
        raise SystemExit(f"✘ 指定年份 {years} 与已建的 {keys[0]}~{keys[-1]} 无交集")
    if not want:
        log(f"  ⊘ {blk.title}：没有需要（重）建的年份")
        return {"action": "skip", "years": [], "meta": meta}

    codes = np.asarray(meta["axis"]["codes"], dtype=object)
    log(f"══ 导出{blk.title}（{len(want)} 年：{want[0]} ~ {want[-1]}）· {root}"
        + (f" · {len(cols)}/{len(meta['columns']['features'])} 列" if kind == SK else ""))
    years_meta = dict(meta.get("years") or {})
    for y in want:
        days = _days_of(root, FK, y)      # ★ 从既有 factors 分区取日子，保证四块同形
        info = blk.build_year(cfg, y, days, codes, cols, log=log, root=root)
        rec = dict(years_meta.get(str(y)) or {})
        rec.setdefault("files", {})
        rec["files"].update(info["files"])
        years_meta[str(y)] = rec
    meta["years"] = years_meta
    _record_block(meta, kind, want, blk.meta_extra(meta, cols), root, log=log)
    return {"action": kind, "years": want}


def ensure_block(cfg: Cfg, kind: str, root: Path, *, log=print) -> list[int]:
    """把陈旧的某块补上（`prepare()` 的出口调它）。★ 必须在主 meta 写完之后 ——
    `build_block` 会再读一次 meta 再写回，放前面会被主流程的 save_json 覆盖掉。"""
    stale = stale_block_years(root, kind, log=log)
    if not stale:
        return []
    build_block(cfg, kind, years=stale, log=log, root=root)
    return stale


# ---------------------------------------------------------------- fac_sample
def build_year_fac_sample(cfg: Cfg, year: int, days: list[str], codes: np.ndarray,
                          cols: tuple[str, ...], *, log=print, root: Path | None = None) -> dict:
    """导出**一年**的 `fac_sample/year=YYYY/data.parquet`。

    ★ 值**逐字取自 `factors`**（不重算）：读那几列、原样写出去。这样抽样块与 `factors`
      必然一致 —— "冒烟时看到的数就是训练时看到的数"，不会出现两套口径。
    """
    t0 = time.time()
    root = root if root is not None else root_of(cfg)
    df = read_frame(root, FK, years=[year], columns=list(cols))
    if df.empty:
        raise SystemExit(f"✘ {year} 年的 factors 分区是空的，无法导出 fac_sample")
    if sorted(df["trade_date"].unique().tolist()) != sorted(days):
        raise SystemExit(f"✘ {year}/fac_sample：factors 的交易日与计划轴不一致 —— 先跑 --check")
    sub = df.sort_values(list(KEY), kind="stable").reset_index(drop=True)
    out = {c: sub[c].to_numpy(dtype=np.float32) for c in cols}
    info = _write_block_year(root, SK, year, days, codes, out)
    log(f"  {year}: fac_sample 块 {info['days']} 天 × {len(codes)} 只 × {len(cols)} 列 = "
        f"{info['rows']:,} 行 · {info['files'][SK]['mb']}MB · {round(time.time() - t0, 1)}s")
    del df, sub
    return info



#: 两个派生块的登记表。★ 放在这里（而不是常量区）是因为它引用 `build_year_*` ——
#: 那两个函数必须先定义；模块级 dict 只要在**调用前**建好就行。
BLOCKS: dict[str, Block] = {
    AK: Block(kind=AK, title="交易额度块 amount",
              cols=lambda meta: AMOUNT_COLUMNS,
              build_year=build_year_amount,
              meta_extra=lambda meta, cols: {"columns": list(cols)}),
    SK: Block(kind=SK, title="因子抽样块 fac_sample",
              cols=fac_sample_features,
              build_year=build_year_fac_sample,
              meta_extra=lambda meta, cols: {"columns": list(cols),
                                             "seed": FAC_SAMPLE_SEED, "n": len(cols)}),
}

# ============================================================================
# 6. 命令行
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="preparingdata.py",
        description="初加工：上游因子 → trainingdata/"
                    "（factors / target / amount / fac_sample 四块）")
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--full", "-f", action="store_true", help="全量重建（删旧、逐年重建）")
    m.add_argument("--incremental", "-i", action="store_true", help="强制增量（默认自动判断）")
    m.add_argument("--check", "-c", action="store_true", help="只校验不写")
    m.add_argument("--meta-only", action="store_true",
                   help="只补 meta 的派生字段（特征集等）—— ★ 不碰数据文件、不读因子侧")
    m.add_argument("--amount-only", action="store_true",
                   help="★ 只建/刷新 amount（每日成交额）—— 同上，不改 panel_digest")
    m.add_argument("--fac-sample-only", action="store_true",
                   help="★ 只建/刷新 fac_sample（因子抽样）—— 同上，不改 panel_digest")
    ap.add_argument("--freeze", action="store_true",
                    help="（配 --meta-only）把当前快照标记为冻结：之后 prepare 会拒绝重跑")
    ap.add_argument("--unfreeze", action="store_true",
                    help="★ 解除冻结。配 --meta-only = 只翻转标记（不重建、不换指纹）；"
                         "不配 = 允许本次按上游当前状态重建（会换 panel_digest）")
    ap.add_argument("--reason", default="", help="冻结原因（写进 meta，便于日后追溯）")
    ap.add_argument("--years", "-y", type=int, nargs="+", default=None,
                    help="只加工这些年份（例：--years 2025 2026）")
    ap.add_argument("--jobs", "-j", type=int, default=DEFAULT_JOBS,
                    help=f"读并发（默认 {DEFAULT_JOBS}）")
    ap.add_argument("--lookback", "-l", type=int, default=None,
                    help="增量回溯窗口（交易日）。默认 = 最长标签 h + 1 + 7 —— "
                         "h+1 是硬下限（太短会让 20d 标签出了窗口就再也不刷新、永久停在 NaN），"
                         "7 是冗余覆盖")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="★ 写到**另一份目录**（相对脚本所在目录），现有 trainingdata/ 完全不动")
    ap.add_argument("--features-from", default=None, metavar="META_JSON",
                    help="★ 按指定的 meta.json 的 `columns.features` 建（而不是用上游当前的因子集）。"
                         "用途：① 与旧产物逐位对拍（保真验证）；② 冻结特征集，不随上游漂移")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    out = None if not a.out else (ROOT / a.out).resolve()
    cfg = Cfg(trainingdata=out)
    if out is not None:
        print(f"  ★ 本次写到**另一份目录**：{out}（现有 {ROOT / 'trainingdata'} 不动）")

    if a.check:
        r = check(cfg)
        return 0 if r["ok"] else 1
    if a.meta_only:
        attach_meta_fields(cfg, freeze=a.freeze, unfreeze=a.unfreeze, reason=a.reason)
        return 0
    if a.amount_only:
        r = build_block(cfg, AK, years=a.years, log=print)
        write_ledger(root_of(cfg), action=r.get("action", ""),
                     rebuilt={(AK, int(y)): AK for y in (r.get("years") or [])}, log=print)
        return 0
    if a.fac_sample_only:
        r = build_block(cfg, SK, years=a.years, log=print)
        write_ledger(root_of(cfg), action=r.get("action", ""),
                     rebuilt={(SK, int(y)): SK for y in (r.get("years") or [])}, log=print)
        return 0
    if a.freeze:
        print("  ⚠️ --freeze 只与 --meta-only 搭配使用（冻结是给**当前快照**打标记，不是重建）")
        return 1
    if a.unfreeze and a.check:
        print("  ⚠️ --unfreeze 要配 --meta-only（只翻标记）或单独用（允许重建）")
        return 1

    feats = direction = None
    if a.features_from:
        # ★ 既接受目录（`.../trainingdata`），也接受 meta.json 本身 —— 两种写法都会有人用
        fp = Path(a.features_from).expanduser().resolve()
        pm = load_json(fp, None) if fp.is_file() else load_meta(fp)
        if not pm or "columns" not in pm:
            print(f"  ✘ 读不到特征清单：{a.features_from}"
                  f"（给目录或 meta.json 都行，但里面要有 columns.features）")
            return 1
        feats = list(pm["columns"]["features"])
        direction = dict(pm["columns"].get("direction") or {})
        print(f"  ★ 特征清单锁定为 {a.features_from} 的 {len(feats)} 列（不随上游漂移）")

    mode = "full" if a.full else ("incremental" if a.incremental else "auto")
    if out is not None and not a.full:
        print("  ⚠️ --out 建议配 --full 用（另建目录的语义就是「从头建一份」）")
    r = prepare(cfg, mode=mode, years=a.years, jobs=max(1, a.jobs),
                lookback=a.lookback, unfreeze=a.unfreeze,
                features=feats, direction=direction)
    write_ledger(root_of(cfg), action=r.get("action", ""),
                 rebuilt=r.get("rebuilt") or {}, log=print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
