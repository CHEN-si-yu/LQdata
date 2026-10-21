#!/usr/bin/env python
"""直接整理落盘因子 → trainingdata/，无需上游 Python、配置、注册表或 state。

输入默认为 ../featureengineering/data/（即 /root/autodl-fs/featureengineering/data）：
  factors/<name>/year=YYYY/data.parquet       trade_date, stock_code, value[, rank]
  market_factors/<name>/year=YYYY/data.parquet trade_date, value
只有确实存在年分区文件的目录参与打包。label_ret_<h>d 自动识别为收益标签，
其余股票对象全部纳入。日期与股票轴取入选股票产物主键的并集，缺格保留 NaN。

输出仍使用原布局：factors、target、fac_sample、market_factors、amount、prices。
股票特征沿用 zscore_win1_99_v1：逐日 1%/99% 缩尾、z-score、截断 ±10；
标签保存 value 原值；市场因子保存原值与截至当日的滚动 252 日 z-score。
Parquet 没有优劣方向：默认沿用本地 meta.json 已有方向，新因子保持原始方向。
另建快照时可继承本项目 trainingdata/meta.json 的方向；--direction-policy raw 全部不翻转。
本脚本不会从因子名称、rank 或未来收益推断优劣方向。

--jobs 同时控制主键扫描线程、因子处理进程和市场读取线程。因子进程完整执行
读取 → 对齐 → 标准化，最多排队 2×jobs 个任务；年份逐个处理以控制内存。
默认最多 12 个进程，同时遵守 CPU affinity 与容器配额；每进程默认 1 个计算线程。
tqdm 显示扫描、年度打包、每年股票特征/标签、市场因子和附加块进度。
本地 .source_index.json 缓存扫描结果，按源路径/大小/mtime_ns 失效，校验模式不写缓存。

成交额及回测价格继续从 ../datadownload/data/ 的原始 Parquet 复制；不复权、不填充。
--factors-only 仅打包因子、标签、抽样与市场块，无需原始价格目录。
增量刷新最长标签 h + 1 + 冗余 7 个交易日；列集、方向或股票轴变化自动全量重建。

用法：
  python preparingdata.py                         # 首次全量，以后增量
  python preparingdata.py --jobs 16               # 指定并行度
  python preparingdata.py --full --progress bars  # 全量并强制显示进度
  python preparingdata.py --source-data /path/data --factors-only --out packed
  python preparingdata.py --check                 # 只读校验
  python preparingdata.py --years 2026 --out trial # 隔离打包指定年份
  python preparingdata.py --features-from /path/meta.json  # 固定股票特征清单
  python preparingdata.py --amount-only           # 只重建指定附加块
  python preparingdata.py --prices-only
  python preparingdata.py --market-only
  python preparingdata.py --fac-sample-only

路径：--source-data > MX_SOURCE_DATA > MX_UPSTREAM/data；MX_FACTORS_DIR 和
MX_MARKET_FACTORS_DIR 可分别覆盖默认因子目录；MX_RAW_DATA 覆盖原始行情目录。
输出：--out > MX_DATA > 脚本旁 trainingdata。依赖 numpy、pandas、pyarrow、tqdm。
"""
from __future__ import annotations

# ★ 必须在 import numpy 之前压线程数：BLAS/OpenMP 默认吃满宿主机核数，
#   与"读并发"以及机器上别的任务互相踩踏。要改就设环境变量 MX_THREADS。
import os

_MX_THREADS = os.environ.get("MX_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _MX_THREADS

import argparse                                                  # noqa: E402
import gc                                                        # noqa: E402
import hashlib                                                   # noqa: E402
import json                                                      # noqa: E402
import multiprocessing as mp                                     # noqa: E402
import re                                                        # noqa: E402
import shutil                                                    # noqa: E402
import sys                                                       # noqa: E402
import tempfile                                                  # noqa: E402
import time                                                      # noqa: E402
import warnings                                                  # noqa: E402
from concurrent.futures import (ThreadPoolExecutor, ProcessPoolExecutor,
                                wait, FIRST_COMPLETED)           # noqa: E402
from datetime import datetime, timedelta                         # noqa: E402
from pathlib import Path                                         # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
import pyarrow as pa                                             # noqa: E402
import pyarrow.compute as pc                                     # noqa: E402
import pyarrow.parquet as pq                                     # noqa: E402
from tqdm.auto import tqdm                                       # noqa: E402

_PROGRESS_MODE = "auto"


def _progress(items, **kwargs):
    quiet = kwargs.pop("disable", False)
    quiet = quiet or kwargs.get("total") == 0 or _PROGRESS_MODE == "off"
    quiet = quiet or (_PROGRESS_MODE == "auto" and not sys.stderr.isatty())
    return tqdm(items, disable=quiet, dynamic_ncols=True, **kwargs)


def progress_log(message: str) -> None:
    """日志与正在显示的进度条共存。"""
    tqdm.write(str(message))


def _bounded_results(pool, function, items, jobs):
    """最多保留 2×jobs 个在途任务，及时回收完成结果，限制排队内存。"""
    iterator = iter(items)
    pending = set()
    for _ in range(max(1, jobs) * 2):
        item = next(iterator, None)
        if item is None:
            break
        pending.add(pool.submit(function, item))
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()
            item = next(iterator, None)
            if item is not None:
                pending.add(pool.submit(function, item))


# ============================================================================
# 0. 路径与配置
# ============================================================================
#: 本脚本所在目录 = 模型工程根（`preparingdata.py` 与 `trainingdata/` 同级）
ROOT = Path(__file__).resolve().parent

#: 仅用于解析数据路径；不导入上游代码、不读配置/注册表/状态文件。
FACTORS_ROOT = Path(os.environ.get("MX_UPSTREAM", ROOT / "../featureengineering")).resolve()
SOURCE_DATA = Path(os.environ.get("MX_SOURCE_DATA", FACTORS_ROOT / "data")).resolve()
#: 因子产物目录：每个因子一个子目录，里面 `year=YYYY/data.parquet`
FACTORS_DIR = Path(os.environ.get("MX_FACTORS_DIR", SOURCE_DATA / "factors")).resolve()
#: 模块① 的原始数据表。**`amount` 与 `prices` 两块用** —— 五块里仅有的两块不来自因子侧。
RAW_DATA = Path(os.environ.get("MX_RAW_DATA", FACTORS_ROOT.parent / "datadownload" / "data")).resolve()
#: 市场因子产物目录（模块② 的**另一个**输出根，与 `data/factors` 并列；每日一行、无股票列）。
MARKET_FACTORS_DIR = Path(os.environ.get(
    "MX_MARKET_FACTORS_DIR", SOURCE_DATA / "market_factors")).resolve()

#: 默认标签名；实际扫描用 label_ret_<h>d 命名识别，支持新增持有期。
LABEL_NAMES = ("label_ret_1d", "label_ret_3d", "label_ret_5d",
               "label_ret_10d", "label_ret_20d")

# ---------------------------------------------------------------- 因子列的**值口径**
#: 写进 `meta.semantics`。面板此前**没有任何值口径标记**（`panel_digest` 只哈希列名与文件 sha，
#: 抓不到值语义），于是"换了口径"与"没换"在产物上完全看不出来 —— 下游只能靠人记。
#: 有了这个字段，一份面板是哪套编码可以自证。**改口径必须同时改这个字符串。**
VALUES_SEMANTICS = "zscore_win1_99_v1"

#: 逐日横截面标准化的三个参数。取值刻意与**因子侧算 rank 时**用的是同一套
#: （`featureengineering/conf/config.yaml` 的 `winsor: [0.01, 0.99]` / `min_cross_section: 100`），
#: 这样面板的值与上游 `rank` 列**排序完全一致**（见 `_standardize` 的说明）。
Z_WINSOR = (0.01, 0.99)
Z_MIN_COUNT = 100
#: 标准化结果的兜底截断。实测"缩尾+z"的 |z| 最大约 7，正常交易日**不会生效**；
#: 它只防"某天截面标准差异常小"时冒出天文数字（纯 z-score 实测有个股 |z| 达 27~44）。
Z_CLAMP = 10.0
#: 市场因子滚动 z 的窗口（交易日）。只用 t 及**之前**的值 → PIT 安全。
MK_Z_WINDOW = 252


class Cfg:
    """仅包含输入 Parquet 路径与本地快照设置，不加载上游工程。"""

    def __init__(self, root: Path = ROOT, trainingdata: Path | None = None,
                 source_data: Path | None = None):
        self.root = Path(root)
        self.factors_root = FACTORS_ROOT
        self.factors_dir = FACTORS_DIR
        self.market_factors_dir = MARKET_FACTORS_DIR
        if source_data is not None:
            self.factors_dir = Path(source_data).resolve() / "factors"
            self.market_factors_dir = Path(source_data).resolve() / "market_factors"
        self.raw_data = RAW_DATA
        self.labels = list(LABEL_NAMES)
        self.jobs = DEFAULT_JOBS
        self.factors_only = False
        self.direction_policy = "preserve"
        self._data_index = None
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
# 3. 落盘数据发现
# ============================================================================

# ---------------------------------------------------------------- 因子目录
def _product_names(directory: Path) -> list[str]:
    """只发现至少有一个年分区文件的因子，忽略空目录/临时目录。"""
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.iterdir()
                  if p.is_dir() and any(year_file(directory, p.name, y).is_file()
                                        for y in years_of(directory, p.name)))


def list_factor_names(cfg: Cfg) -> list[str]:
    return _product_names(cfg.factors_dir)


def list_market_factor_names(cfg: Cfg) -> list[str]:
    return _product_names(cfg.market_factors_dir)


def upstream_specs(cfg: Cfg, extra_features: list[str] | None = None) -> dict[str, dict]:
    """数据目录就是清单。label_ret_<h>d 为标签，其余股票对象全部纳入。"""
    names = list_factor_names(cfg)
    missing = sorted(set(extra_features or []) - set(names))
    if missing:
        raise ValueError(f"额外因子没有落盘的年分区：{missing[:5]}")
    return {n: {"is_label": bool(re.fullmatch(r"label_ret_\d+d", n))}
            for n in names}


def market_specs(cfg: Cfg) -> list[str]:
    return list_market_factor_names(cfg)


def last_upstream_day(cfg: Cfg) -> str:
    return scan(cfg, log=None)["span"][1]


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


def _available_cpus() -> int:
    """同时尊重 CPU affinity 与容器 CPU quota。"""
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return count


DEFAULT_JOBS = min(12, _available_cpus())
REDUNDANT_DAYS = 7              # 增量回溯的**冗余覆盖**天数（用户 2026-09-20 要的，见 `lookback_for`）


def lookback_for(cfg: Cfg, labels: list[str] | None = None) -> int:
    """增量回溯窗口（交易日）= **最长标签的 h + 余量**。

    ★ 为什么不能拍一个常数：标签 `T` 用到 `open(T+1+h)`，所以一个交易日要等到
      **h+1 个交易日之后**它的标签才定型。若窗口比这短，那些天一旦离开窗口就再也不会被重算，
      20d 标签会**永久停在 NaN**（上游后来补上了也不会被捡回来）。默认加 1 日及 7 日冗余。
    """
    ls = labels if labels is not None else cfg.labels
    return max((cfg.label_horizon(l) for l in ls), default=1) + 1 + REDUNDANT_DAYS


# ================================================================ 路径
# ★ `root_of` / `meta_path` / `year_file` / `years_of` / `load_meta` 定义在第 2 节。
#   它们原先被单独拆成一个模块（`mx/panel_io.py`），好让模型侧不拖上游依赖；
#   本脚本是构建期的、本来就要上游，所以合并回来。


# ================================================================ 直接扫描落盘数据
SOURCE_INDEX_FILE = ".source_index.json"


def _inspect_partition(item):
    """只解压主键列，取日期/股票并集；不读取 value/rank。"""
    key, path, signature = item
    pf = pq.ParquetFile(path)
    stock = key.startswith("stock/")
    columns = ["trade_date", "stock_code"] if stock else ["trade_date"]
    required = set(columns + ["value"])
    if not required <= set(pf.schema_arrow.names):
        raise ValueError(f"因子文件缺少必要列 {sorted(required - set(pf.schema_arrow.names))}：{path}")
    if not stock and "stock_code" in pf.schema_arrow.names:
        raise ValueError(f"市场因子文件不应有 stock_code：{path}")
    dates, codes = set(), set()
    for batch in pf.iter_batches(batch_size=131072, columns=columns, use_threads=False):
        for name, output in (("trade_date", dates), ("stock_code", codes)):
            if name not in columns:
                continue
            values = pc.unique(batch.column(columns.index(name))).to_pylist()
            if any(v is None for v in values):
                raise ValueError(f"因子主键 {name} 含空值：{path}")
            output.update(str(v)[:10] if name == "trade_date" else str(v) for v in values)
    if any(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) for d in dates):
        raise ValueError(f"trade_date 必须是 YYYY-MM-DD：{path}")
    axis = {"days": sorted(dates), "codes": sorted(codes)}
    digest = hashlib.sha256(json.dumps(axis, sort_keys=True).encode()).hexdigest()
    if [path.stat().st_size, path.stat().st_mtime_ns] != signature:
        raise RuntimeError(f"扫描时源文件发生变化，请稍后重跑：{path}")
    return key, {"signature": signature, "axis": digest, "rows": pf.metadata.num_rows}, axis


def _load_data_index(cfg: Cfg, *, jobs: int, log=progress_log) -> dict:
    """本地缓存按路径、文件大小、mtime_ns 失效；不读取任何上游状态文件。"""
    if cfg._data_index is not None:
        return cfg._data_index
    old = load_json(getattr(cfg, "_index_root", root_of(cfg)) / SOURCE_INDEX_FILE, {}) or {}
    roots = {"stock": str(cfg.factors_dir.resolve()), "market": str(cfg.market_factors_dir.resolve())}
    if old.get("roots") != roots or old.get("version") != 1:
        old = {}
    axes, partitions, todo = {}, {}, []
    for kind, directory, names in (("stock", cfg.factors_dir, list_factor_names(cfg)),
                                   ("market", cfg.market_factors_dir, list_market_factor_names(cfg))):
        for name in names:
            for year in years_of(directory, name):
                path = year_file(directory, name, year)
                if not path.is_file():
                    continue
                st = path.stat()
                signature = [st.st_size, st.st_mtime_ns]
                key = f"{kind}/{name}/{year}"
                previous = (old.get("partitions") or {}).get(key)
                if (previous and previous.get("signature") == signature
                        and previous.get("axis") in (old.get("axes") or {})):
                    partitions[key] = previous
                    axes[previous["axis"]] = old["axes"][previous["axis"]]
                else:
                    todo.append((key, path, signature))
    cached = len(partitions)
    if log:
        log(f"  落盘分区：{cached + len(todo)} 个 · 缓存命中 {cached} 个 · 待扫描 {len(todo)} 个")
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = _bounded_results(pool, _inspect_partition, todo, jobs)
        for key, record, axis in _progress(results, total=len(todo), desc="扫描主键", unit="分区",
                                      disable=log is None, leave=False):
            partitions[key] = record
            axes[record["axis"]] = axis
    cfg._data_index = {"version": 1, "roots": roots, "partitions": partitions, "axes": axes}
    return cfg._data_index


def _factor_years(cfg: Cfg, name: str) -> list[int]:
    return [y for y in years_of(cfg.factors_dir, name)
            if year_file(cfg.factors_dir, name, y).is_file()]


def scan(cfg: Cfg, names: list[str] | None = None, jobs: int = DEFAULT_JOBS, log=progress_log) -> dict:
    """从真实 Parquet 主键汇总覆盖范围与坐标轴（首次扫描；以后复用本地缓存）。"""
    t0 = time.time()
    names = sorted(upstream_specs(cfg)) if names is None else names
    index = _load_data_index(cfg, jobs=jobs, log=log)
    days, codes, per = set(), set(), {}
    for name in names:
        part_days, years = set(), []
        for year in _factor_years(cfg, name):
            record = index["partitions"][f"stock/{name}/{year}"]
            axis = index["axes"][record["axis"]]
            part_days.update(axis["days"])
            codes.update(axis["codes"])
            years.append(year)
        days.update(part_days)
        per[name] = {"start": min(part_days) if part_days else None,
                     "end": max(part_days) if part_days else None, "years": years}
    days = sorted(days)
    result = {"n_factors": len(names), "span": (days[0], days[-1]) if days else (None, None),
              "years": sorted({y for v in per.values() for y in v["years"]}),
              "per_factor": per, "seconds": round(time.time() - t0, 1),
              "days": days, "codes": sorted(codes)}
    if log:
        log(f"  上游扫描：{len(names)} 个股票对象 · 分区年 {len(result['years'])} 个 · "
            f"覆盖 {result['span'][0]} ~ {result['span'][1]}（{result['seconds']}s，直接读落盘数据）")
    return result


# ================================================================ 日期轴 / 股票轴
def build_axis(cfg: Cfg, lo: str, hi: str, log=progress_log,
               scanned: dict | None = None) -> tuple[list[str], np.ndarray]:
    """全部入选股票产物的主键并集：不读取上游日历、股票池或配置。"""
    sc = scanned if scanned is not None else scan(cfg, jobs=cfg.jobs, log=None)
    days = [d for d in sc["days"] if lo <= d <= hi]
    codes = np.asarray(sc["codes"], dtype=object)
    if not days or not len(codes):
        raise ValueError("落盘股票因子中没有有效的日期/股票主键")
    if log:
        log(f"  日期轴：{len(days)} 个交易日（{days[0]} ~ {days[-1]}）· 股票轴：{len(codes)} 只")
    return days, codes


# ================================================================ 落格：一个因子 → 扁平网格
def _place(df: pd.DataFrame, *, source: str, codes_index: pd.Index,
           days_index: pd.Index, n_days: int, C: int,
           codes: np.ndarray | None = None,
           days_int: np.ndarray | None = None) -> tuple[np.ndarray, int, int]:
    """把一列因子落到 (T×C,) 扁平网格（日为主序）。返回 (数组, 命中数, 丢弃数)。

    `source`：特征与标签都取 `value` 列。**特征取原值**（随后由 `_standardize` 做
    逐日缩尾+z-score），标签取原值直接落盘（未来收益，不做任何变换）。

    ★ 本函数**不做方向翻转**（2026-09-25 改口径时从 `_place` 挪进 `_standardize`）。
      翻转必须与标准化在同一处，否则会出现"先 `1−rank` 再 z-score"这类半新半旧的组合。

    ## 两条路径

    上游产物绝大多数是"整年满格"（每个交易日 × 全部代码都有一行）。这时**不需要**逐行做
    字符串→索引的映射：行序天然就是 (日, 码)，直接 `reshape` 即可。这条快路很重要 ——
    通用路径的 `Index.get_indexer` 要索引 84 万条字符串，**握着 GIL**，8 个线程也跑不满 1 个核
    （实测 1.1 核、单年 4 分钟；快路后降到几十秒）。

    快路的正确性由三件事保证（都通过才走）：行数 = 天数×代码数、日期向量**逐元素**等于
    `repeat(days, C)`、每一天的代码向量均等于输出 codes。任一不满足就退回通用路径。
    """
    if codes is not None and days_int is not None and len(df) == n_days * C and n_days:
        di = df["trade_date"].to_numpy(dtype=object).reshape(n_days, C)
        expected_days = days_index.to_numpy(dtype=object)[:, None]
        if np.all(di == expected_days):
            # 每一天都核对股票顺序，避免中间某天乱序却误走快路。
            if np.all(df["stock_code"].to_numpy(dtype=object).reshape(n_days, C) == codes[None, :]):
                v = df[source].to_numpy(dtype=np.float32, copy=True)
                return v, int(v.size), 0

    di = days_index.get_indexer(df["trade_date"].to_numpy(dtype=object))
    ci = codes_index.get_indexer(df["stock_code"].to_numpy(dtype=object))
    ok = (di >= 0) & (ci >= 0)
    v = df[source].to_numpy(dtype=np.float32, copy=False)
    out = np.full(n_days * C, np.nan, dtype=np.float32)
    positions = di[ok] * C + ci[ok]
    if len(np.unique(positions)) != len(positions):
        raise ValueError("因子文件有重复的 (trade_date, stock_code)，不能静默覆盖")
    out[positions] = v[ok]
    return out, int(ok.sum()), int((~ok).sum())


def _standardize(flat: np.ndarray, n_days: int, C: int, flip: bool) -> np.ndarray:
    """把落格后的**原值**做成逐日横截面的「缩尾 + z-score」，并按需翻转方向。

    ## 口径（`VALUES_SEMANTICS = "zscore_win1_99_v1"`）

    每行（= 每个交易日）独立地：
      ① 按 1%/99% 分位缩尾 ② 减均值除标准差 ③ 兜底截断 ±`Z_CLAMP` ④ 需要则取负。

    ★ **逐行计算 ⇒ PIT 安全**：只用当日截面，不碰任何未来数据。
    ★ **不改当日排序**：缩尾分位与因子侧算 `rank` 时用的是同一套，所以本列的名次与上游
      `rank` 列一致（只可能在并列处不同）。这是"换口径不动 IC/名次结论"的依据。

    ## 两处刻意的处理

    **缩尾分位点重合时跳过缩尾**：稀疏事件因子（如只有一个涨停日）的 1%/99% 分位点会相等，
      此时缩尾会把唯一的事件压成常数。因子侧 `cs_rank` 对这个情形有同样的处理
      （`featureengineering/fea/panel.py:193-195`），这里保持一致的语义。

    **缩尾后退化则退回原值**：离散/事件型因子（实测 `limit_down_event_5` 全截面只有 5 个
      不同取值）缩尾后可能整行同值、标准差为 0。若直接标准化会把整列变成 NaN（`cs_zscore`
      的保护行为），等于**静默丢掉一个因子**。这类行改用原始值再标准化一次，保住信号。
    """
    x = flat.reshape(n_days, C).astype(np.float64)
    # ★ `np.errstate` 只管浮点异常，管不住 nan-functions 走的 `warnings.warn`。
    #   全 NaN 行（晚起点因子在更早的年份）必然会触发 "All-NaN slice / Mean of empty slice /
    #   Degrees of freedom <= 0" —— 这是**预期**情形且已被下面的逻辑显式处理，
    #   不静音的话每个因子-年都会刷一遍，日志会被淹掉。
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        quantiles = np.nanquantile(x, Z_WINSOR, axis=1)
        lo, hi = quantiles[0, :, None], quantiles[1, :, None]
        usable = np.isfinite(lo) & np.isfinite(hi) & (lo < hi)   # 分位点重合 → 不缩尾
        xw = np.where(usable, np.clip(x, lo, hi), x)

        sd = np.nanstd(xw, axis=1)
        degen = (~(sd > 0)) & (np.isfinite(xw).sum(axis=1) >= Z_MIN_COUNT)
        if degen.any():
            xw[degen] = x[degen]

        count = np.isfinite(xw).sum(axis=1)
        mu = np.nanmean(xw, axis=1)
        sd = np.nanstd(xw, axis=1)
        mu = np.where(count >= Z_MIN_COUNT, mu, np.nan)
        sd = np.where(count >= Z_MIN_COUNT, sd, np.nan)
        z = (xw - mu[:, None]) / sd[:, None]
        z[~np.isfinite(z)] = np.nan
        z = np.clip(z, -Z_CLAMP, Z_CLAMP)          # NaN 经 clip 仍是 NaN
        if flip:
            z = -z                                  # 仅按本地已有方向翻转
    return z.astype(np.float32).ravel()


_COLUMN_CONTEXT = None


def _init_column_worker(factors_dir, year, codes, days, flip_map, source, standardize):
    """每个工作进程只初始化一次坐标索引，不把大面板传给子进程。"""
    global _COLUMN_CONTEXT
    codes_index, days_index = pd.Index(codes), pd.Index(days)
    codes_index.get_indexer(codes[:1])
    days_index.get_indexer(days[:1])
    _COLUMN_CONTEXT = (Path(factors_dir), year, codes, days, codes_index, days_index,
                       flip_map, source, standardize)
    pa.set_cpu_count(max(1, int(_MX_THREADS)))
    pa.set_io_thread_count(max(1, int(_MX_THREADS)))


def _read_column(nm, context):
    directory, year, codes, days, codes_index, days_index, flip_map, source, standardize = context
    path = year_file(directory, nm, year)
    if not path.is_file():
        return nm, np.full(len(days) * len(codes), np.nan, dtype=np.float32), {
            "hit": 0, "dropped": 0, "missing_year": True}
    # 增量只解压窗口内的主键和值；rank 从不读取。
    tab = pq.read_table(path, columns=[*KEY, source], use_threads=False,
                        filters=[("trade_date", ">=", days[0]), ("trade_date", "<=", days[-1])])
    df = tab.to_pandas()
    del tab
    df["trade_date"] = df["trade_date"].astype(str).str[:10]
    df["stock_code"] = df["stock_code"].astype(str)
    arr, hit, dropped = _place(df, source=source, codes_index=codes_index,
                               days_index=days_index, n_days=len(days), C=len(codes),
                               codes=codes, days_int=np.asarray([to_int(d) for d in days]))
    del df
    if standardize:
        arr = _standardize(arr, len(days), len(codes), flip_map.get(nm, False))
    return nm, arr, {"hit": hit, "dropped": dropped, "missing_year": False}


def _column_worker(nm):
    return _read_column(nm, _COLUMN_CONTEXT)


def _cols(cfg: Cfg, year: int, names: list[str], flip_map: dict[str, bool], source: str,
          codes_index: pd.Index, days_index: pd.Index, n_days: int, C: int,
          *, jobs: int, what: str, codes: np.ndarray, days_int: np.ndarray,
          log=progress_log, standardize: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    """因子任务并行完成读取、落格及标准化；标签只读原值。"""
    if not names:
        return {}, {}
    jobs = min(max(1, jobs), len(names))
    days = days_index.to_numpy(dtype=object)
    args = (cfg.factors_dir, int(year), codes, days, flip_map, source, standardize)
    context = (cfg.factors_dir, int(year), codes, days, codes_index, days_index,
               flip_map, source, standardize)
    out, stat = {}, {}
    if jobs == 1:
        results = (_read_column(n, context) for n in names)
        for nm, arr, record in _progress(results, total=len(names), desc=f"{year} {what}",
                                    unit="因子", disable=log is None, leave=False):
            out[nm], stat[nm] = arr, record
    else:
        # spawn 避免继承主进程 Arrow/BLAS 的线程状态；标签无需 CPU 标准化，用轻量线程池。
        if standardize:
            pool = ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn"),
                                       initializer=_init_column_worker, initargs=args)
            worker = _column_worker
        else:
            pool = ThreadPoolExecutor(max_workers=jobs)
            worker = lambda nm: _read_column(nm, context)
        with pool:
            results = _bounded_results(pool, worker, names, jobs)
            for nm, arr, record in _progress(results, total=len(names), desc=f"{year} {what} ({jobs}并行)",
                                        unit="因子", disable=log is None, leave=False):
                out[nm], stat[nm] = arr, record
    # 完成顺序不影响输出列顺序，单进程/多进程快照一致。
    return {n: out[n] for n in names}, {n: stat[n] for n in names}


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


# ================================================================ 构建一年
def build_year(cfg: Cfg, year: int, days_new: list[str], codes: np.ndarray,
               feats: list[str], labels: list[str], directions: dict[str, int],
               *, jobs: int = DEFAULT_JOBS, log=progress_log, keep: dict | None = None,
               root: Path | None = None) -> dict:
    """构建（或增量重算）一年的 factors/target 并原子落盘。

    `days_new`：**本次要重算**的日子（增量时=窗口内的那几天，不是整年！）；`keep`：要保留在
    **前面**的既有行（`{"days": [...], "factors": {...}, "target": {...}}` —— slot 名直接用
    块名，免得又多一层"内部叫 X、落盘叫 factors"的对照）。
    ★ 增量调用必须传"窗口内"的日子：传整年会把窗口外的旧行算第二遍 → 该年出现重复 (date,code)。
    """
    t0 = time.time()

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
    # 先串行预热索引，再进入线程池，避免索引首次初始化发生竞态。
    codes_index.get_indexer(codes[:1])
    days_index.get_indexer(np.asarray(days_new[:1], dtype=object))

    flip_map = {f: (directions.get(f, 1) < 0) for f in feats}
    # ★ 2026-09-25 口径变更：特征从"读上游算好的 `rank` 列"改成"读 `value` 原值 +
    #   `_standardize` 自己做逐日缩尾+z-score"。同一次改动里加 `semantics` 标记，
    #   否则换了口径在产物上完全看不出来。
    X, stat_x = _cols(cfg, year, feats, flip_map, "value", codes_index, days_index, n_new, C,
                      jobs=jobs, what="特征", codes=codes, days_int=di_new, log=log,
                      standardize=True)
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
            if old and len(next(iter(old.values()))) != n_keep * C:
                raise SystemExit(f"✘ {year} 年 {name} 的既有行数不等于 {n_keep}×{C} —— 产物坏了，请全量重建")
            for k in cols:
                cols[k] = np.concatenate([old[k], cols[k]])

    n = n_days * C
    di = np.asarray([int(d[:4]) * 10000 + int(d[5:7]) * 100 + int(d[8:10]) for d in days],
                    dtype=np.int32)
    td = pa.array(np.repeat(np.asarray(days, dtype=object), C))
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
                       "mb": round(p.stat().st_size / 1e6, 1), "bytes": int(p.stat().st_size)}
        del tab, fields
    cov = _coverage(X, n_days, C)
    miss = sorted(k for k, v in stat_x.items() if v["missing_year"])
    dropped = sum(v["dropped"] for v in stat_x.values())
    # 窗口外旧行已在 Parquet 读取时过滤，不计作脏行。
    dirty = dropped  # 窗口外的行已在 Parquet 读取时过滤，不计入丢弃
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


def source_description(cfg: Cfg, sc: dict) -> dict:
    """记录上游位置、日期范围和分区覆盖。"""
    per_years = {n: sorted(int(y) for y in (v.get("years") or []))
                 for n, v in sc["per_factor"].items()}
    return {"factors_dir": str(cfg.factors_dir), "n_products": sc["n_factors"],
            "market_factors_dir": str(cfg.market_factors_dir),
            "first_upstream_day": sc["span"][0], "last_upstream_day": sc["span"][1],
            "per_factor_years": per_years}


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

    ★ 为什么能独立于上游计算：它只用 meta 里已经存着的 `years[*].coverage.per_factor`
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


def attach_meta_fields(cfg: Cfg, *, log=progress_log) -> dict:
    """只补 meta 中的特征集，不改数据文件、不读取上游因子。"""
    root = root_of(cfg)
    meta = load_meta(root)
    if not meta:
        raise SystemExit(f"✘ 还没有产物（{root}）")
    meta["feature_sets"] = derive_feature_sets(meta)
    save_json(meta_path(root), meta)
    log(f"  ✔ meta 特征集已更新：{ {k: len(v) for k, v in meta['feature_sets'].items()} }")
    return meta


# ================================================================ 主流程
def prepare(cfg: Cfg, *, mode: str = "auto", years: list[int] | None = None,
            jobs: int = DEFAULT_JOBS, lookback: int | None = None,
            root: str | Path | None = None,
            features: list[str] | None = None, direction: dict[str, int] | None = None,
            extra_features: list[str] | None = None,
            log=progress_log) -> dict:
    """全量、增量或指定年份构建训练数据。"""
    t_all = time.time()
    root = Path(root).resolve() if root else root_of(cfg)
    cfg.jobs = max(1, jobs)
    cfg._data_index = None
    cfg._index_root = root
    if root != root_of(cfg):
        log(f"  ★ 本次写到**另一份快照**：{root}（现有快照 {root_of(cfg)} 不动）")
    meta_old = load_meta(root)
    force_full = (mode == "full") or not meta_old
    log(f"══ 初加工 trainingdata{'（全量）' if mode == 'full' else ''}")
    log(f"  产物：{root}")

    specs = upstream_specs(cfg, extra_features)
    feats = sorted(n for n, v in specs.items() if not v["is_label"])
    labels = sorted(n for n, v in specs.items() if v["is_label"])
    # Parquet 没有因子优劣方向。只沿用本地快照/显式传入的方向，未知因子保持原值方向。
    local_meta = meta_old or load_meta(ROOT / "trainingdata")
    known_direction = dict((local_meta.get("columns") or {}).get("direction") or {})
    raw_direction = set((local_meta.get("columns") or {}).get("raw_direction_features") or [])
    if direction is not None:
        known_direction.update(direction)
        raw_direction -= set(direction)
    if cfg.direction_policy == "raw":
        known_direction = {}
        raw_direction = set(specs)
    invalid = {n: v for n, v in known_direction.items() if v not in (-1, 1)}
    if invalid:
        raise ValueError(f"本地方向只能取 -1 或 1：{list(invalid.items())[:5]}")
    directions = {n: int(known_direction.get(n, 1)) for n in specs}
    raw_direction |= set(specs) - set(known_direction)
    # ★ `features` 非空 = **锁定特征清单**（`--features-from`）：只收清单里的列，且顺序照它。
    #   两个用途：① 与旧产物逐位对拍（上游因子集变了也还能比）；② 冻结特征集不随上游漂移。
    if features:
        miss = [f for f in features if f not in specs]
        if miss:
            raise SystemExit(f"✘ --features-from 里有 {len(miss)} 个因子在当前上游不存在：{miss[:5]}")
        feats = list(dict.fromkeys(list(features) + list(extra_features or [])))
        if any(specs[n]["is_label"] for n in feats):
            raise ValueError("--features-from/--extra-features 不能把收益标签作为特征")
    # 源身份只跟随本次实际写入的特征与标签。新增但未纳入锁定清单的因子
    # 不应触发 `_years_with_new_partitions` 把 2018 起所有年度重建一遍。
    sc = scan(cfg, names=sorted(set(feats) | set(labels)), jobs=cfg.jobs, log=log)
    lo, hi = sc["span"]
    if not lo:
        raise SystemExit("✘ 上游没有任何因子产物")
    log(f"  特征 {len(feats)} · 标签 {len(labels)} · 需翻转 "
        f"{sum(1 for f in feats if directions[f] < 0)}"
        f"（沿用本地方向；{sum(f in raw_direction for f in feats)} 个因子保持原始方向）"
        f"· 值口径 {VALUES_SEMANTICS}")

    days, codes = build_axis(cfg, lo, hi, log=log, scanned=sc)
    old_columns = meta_old.get("columns") or {}
    changed = bool(meta_old) and (
        old_columns.get("features") != feats or old_columns.get("labels") != labels
        or old_columns.get("direction") != {n: directions[n] for n in feats}
        or (meta_old.get("axis") or {}).get("codes") != codes.tolist()
        or meta_old.get("semantics") != VALUES_SEMANTICS)
    if meta_old and not years:
        old_end = str((meta_old.get("axis") or {}).get("last_built_day") or "")
        old_years = set((meta_old.get("years") or {}).keys())
        old_days = sum(int(v["days"]) for v in (meta_old.get("years") or {}).values())
        changed = changed or len([d for d in days if d[:4] in old_years and d <= old_end]) != old_days
    if changed:
        if years:
            raise ValueError("列集/方向/股票轴发生变化，必须重建全部年份；请去掉 --years，或 --out 到新目录")
        force_full = True
        log("  ★ 列集/方向/股票轴变化 → 自动全量重建，避免新旧分区混用")
    log(f"  并行：{cfg.jobs} 个因子工作进程 · 每进程 { _MX_THREADS } 个计算线程")
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
    source = source_description(cfg, sc)
    have_years = {int(y) for y in (meta_old.get("built_years") or [])}

    # ---- 决定做什么
    plan: dict[int, tuple[str, list[str] | None]] = {}
    if force_full or years:
        plan = {y: ("full", None) for y in built}
        why = "全量重建" if force_full else f"指定年份 {built}"
    else:
        last_old = (meta_old.get("axis") or {}).get("last_built_day")
        missing = [y for y in built if y not in have_years]
        grew = _years_with_new_partitions(meta_old, sc)
        # 每次执行都刷新回溯窗口，以纳入上游对近期数据的修订。
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
    years_meta = dict(meta_old.get("years") or {}) if years or not force_full else {}
    for y in _progress(sorted(plan), desc="年度打包", unit="年", disable=log is None):
        act, tail = plan[y]
        keep = None
        if act == "incr":
            keep = _load_keep(cfg, y, tail[0], feats, labels, log=log, root=root)
            if keep is None:
                act, tail = "full", None
        plan[y] = (act, tail)
        # ★★ 增量时必须只传**窗口内的那些天**（`tail`），绝不能传整年（`by_year[y]`）：
        #    keep 里已经有窗口外的旧行，再算一遍整年 → 那些天在文件里出现两次
        #    （行数虚高、(date,code) 重复、下游静默双计）。这个坑踩过一次，别再踩。
        info = build_year(cfg, y, tail if act == "incr" else by_year[y], codes, feats, labels,
                          directions, jobs=jobs, log=log, keep=keep, root=root)
        info["action"] = act
        # ★★ `build_year` **只产出 `factors` / `target`** —— 这一年的 `amount` / `fac_sample`
        #   文件还在原地没动（它们走独立计划轴），所以它们在该年 `files` 里的记录（行列数与文件大小）
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
        "source": {**source, "scan": {"n": sc["n_factors"], "partition_years": sc["years"],
                                  "per_factor_years": source["per_factor_years"]},
                   "discovery": "parquet_direct", "extra_features": list(extra_features or []),
                   "raw_blocks": not cfg.factors_only},
        "columns": {"features": feats, "labels": labels,
                    "direction": {n: directions[n] for n in feats},
                    "n_flipped": sum(1 for f in feats if directions[f] < 0),
                    "direction_policy": cfg.direction_policy,
                    "raw_direction_features": [f for f in feats if f in raw_direction]},
        # ★ 因子列的**值口径**（2026-09-25 新增）。此前面板没有任何值口径标记，
        #   `panel_digest` 只哈希列名与文件 sha ⇒ "换了口径"在产物上完全看不出来。
        #   改口径必须同时改 `VALUES_SEMANTICS`，否则这个字段会说谎。
        "semantics": VALUES_SEMANTICS,
        "axis": {"start": min(v["start"] for v in years_meta.values()),
                 "end": max(v["end"] for v in years_meta.values()),
                 "n_days": sum(v["days"] for v in years_meta.values()), "n_codes": len(codes),
                 "codes": [str(c) for c in codes],
                 "first_built_day": min(v["start"] for v in years_meta.values()),
                 "last_built_day": max(v["end"] for v in years_meta.values())},
        "years": years_meta, "built_years": kept,
        "partial": set(years_meta) != {d[:4] for d in days},
        "seconds": round(time.time() - t_all, 1),
    }
    # ★★ 上面这个 meta 是**从零重建**的，而 `amount` / `fac_sample` 的记录是事后由
    #   `_record_block` 单独写进去的（它们走独立计划轴，见下）—— 不搬过来的话，
    #   **任何一次 `prepare()`（含日常增量）都会把这两块的记录抹掉**；随后 `ensure_block`
    #   （2026-09-20 实测踩到：跑一次日常增量后 meta 顶层就没这两个 key 了。）
    #   先原样搬过来，下面 `ensure_block` 真重建时会用新的 `_record_block` 覆盖掉。
    for _k in (*ALL_KINDS, MK):
        if cfg.factors_only and _k in (AK, PK):
            continue
        if _k == MK and not market_specs(cfg):
            continue
        if _k in meta_old:
            meta[_k] = meta_old[_k]

    meta["feature_sets"] = derive_feature_sets(meta)      # 单元 DATA={"features":"core_2018"} 用
    save_json(root / SOURCE_INDEX_FILE, cfg._data_index)
    save_json(meta_path(root), meta)
    log(f"  ✔ meta 已写（本次 {meta['seconds']}s）")

    # ★★ 两个**可选块**（amount / fac_sample）走**独立计划轴**，且必须在主 meta 写完之后
    #   —— 它们会再读一次 meta、补上 `years[y].files.<块>` 与顶层同名 key 后写回，
    #   放在前面会被这里覆盖。为什么不能塞进上面的 plan：`prepare()` 的"无新增"快路径
    #   会直接 return，而这两块与因子侧没有共同的失效条件（见 `stale_block_years`）。
    #   `rebuilt_years` 必须传：这三块的陈旧判据（文件在不在 / 日期轴 / 列名）**抓不到"值变了"**，
    #   而 `fac_sample` 是 `factors` 的逐字副本 ⇒ 不传的话换口径后它会留着旧值（详见 `ensure_block`）。
    ensure_block(cfg, SK, Path(root), rebuilt_years=plan.keys(), log=log)
    if not cfg.factors_only:
        ensure_block(cfg, AK, Path(root), rebuilt_years=plan.keys(), log=log)
        ensure_block(cfg, PK, Path(root), rebuilt_years=plan.keys(), log=log)
    ensure_market(cfg, Path(root), rebuilt_years=plan.keys(), log=log)

    meta = load_meta(root)
    _summary(meta, log=log)
    # ★ `rebuilt` 交回给调用方喂台账：`{"年": "full"/"incr"}` —— 只有**本次真的重建了**
    #   的年份在里面。别看 `meta["years"][y]["action"]`，那个对没重建的年份是上一次的旧值。
    return {"action": "built", "years": kept, "meta": meta,
            "rebuilt": {(k, int(y)): plan[y][0] for y in plan for k in (FK, TK)}}


def _load_keep(cfg: Cfg, year: int, win_lo: str, feats: list[str], labels: list[str],
               log=progress_log, root: Path | None = None) -> dict | None:
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
def check(cfg: Cfg, *, root: str | Path | None = None, log=progress_log) -> dict:
    """只校验不写：meta ↔ 文件 ↔ 上游（单条记录级体检，不做跨因子深度分析）。

    `root` 指定要校验的快照（配 `preparingdata.py --check --out DIR`）——
    另建的快照要能**先体检再换上去**，否则只能盲切。
    """
    root = Path(root).resolve() if root else root_of(cfg)
    meta = load_meta(root)
    if not meta:
        log("  ⊘ 还没有产物 —— 先跑 `python preparingdata.py`（首次会自动全量）")
        return {"ok": False, "problems": ["没有 meta.json"], "years": []}
    cfg._data_index = None
    cfg._index_root = root
    built = meta.get("built_years") or []
    problems: list[str] = []
    log(f"══ 校验 {root}（{len(built)} 年：{built[0]} ~ {built[-1]}）")
    sc = scan(cfg, jobs=cfg.jobs, log=log)
    ax = meta.get("axis") or {}
    lo_now, hi_now = sc["span"]
    if lo_now and str(lo_now) < str(ax.get("start")) and not meta.get("partial"):
        problems.append(f"上游已回溯到 {lo_now} < 产物起点 {ax.get('start')} —— 需要 --full 扩展")
    if hi_now and str(hi_now) > str(ax.get("last_built_day")):
        problems.append(f"上游最新日 {hi_now} > 已建 {ax.get('last_built_day')} —— 需要跑增量")
    rows_total = 0
    C = int(ax.get("n_codes") or 0)
    feats = list((meta.get("columns") or {}).get("features") or [])
    for y in built:
        info = meta["years"][y]
        exp = int(info["days"]) * C
        # ★ 只查**双键块**（`ALL_KINDS`）。市场块是单键、每日一行，期望行数是 `days` 而不是
        #   `days × C`，混进来会被报成"行数不符" —— 它在本函数后半段有自己的一段校验。
        kinds = (FK, TK, SK) if not (meta.get("source") or {}).get("raw_blocks", True) else ALL_KINDS
        for kind in kinds:
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
        # 因子"全年无值"只有在**它本该有值**时才算问题。
        # ★ 判据不能是"分区/起点年"，两者都**结构性地答不了这个问题**：
        #   ① 上游按对齐契约给每个因子都铺了 2018 起的年分区（`scan()` 的
        #      `per_factor_years` 与 manifest 的 `min_date` 对晚起点因子同样是 2018-01-02）；
        #   ② `FactorSpec.start` 只表示"从哪天起有真实值"，其前年份是**对齐区间**，
        #      值由 `align_fill` 决定 —— 整列为空正是 `align_fill=nan` 的**设计结果**。
        #   2026-09-25 实测：按分区年判会把 36 个因子-年误报成"上游分区少了"，真问题被淹没。
        # ⇒ 唯一站得住的判据是**去上游实测那一年那一列有没有方差**：
        #   整列无有限值（上游本年没有观测）或恒为常量（上游以 0 表达"未披露"）⇒
        #   逐日 z-score 必然为空，**是设计内行为，不是丢数据**（常量经 `cs_zscore`
        #   的常量保护得 NaN —— 见 `_standardize` 的说明）。
        #   只对**当年零覆盖**的那几个因子读上游（2018/2019 各十几列，其余年份 0 列），
        #   所以这条判据的代价与问题数量成正比，不会随面板变宽而变贵。
        def _upstream_variance(nm: str, yy: str) -> bool | None:
            """上游那一年那一列有没有方差。None = 分区文件缺失（问不到）。"""
            p = year_file(cfg.factors_dir, nm, int(yy))
            if not p.exists():
                return None
            a = pq.read_table(p, columns=["value"]).column("value").to_numpy(zero_copy_only=False)
            a = a[np.isfinite(a)]
            return bool(a.size and a.std() > 0)

        bad, no_obs = [], []
        for k, v in cov["per_factor"].items():
            if v != 0.0:
                continue
            if _upstream_variance(k, y) is not True:
                no_obs.append(k)     # 上游本年无方差 ⇒ 标准化后为空，正常
            else:                    # 有方差却整列空（或分区都没了）⇒ 真丢了
                bad.append(k)
        bad, no_obs = sorted(bad), sorted(no_obs)
        if bad:
            problems.append(f"{y}: {len(bad)} 个因子整年无值、但上游本年**有方差**"
                            f"（例：{bad[:3]}）—— 对齐或落格丢了数据")
        by_day = cov.get("by_day") or []
        floor = max(LOW_COV_ABS, LOW_COV_REL * float(cov["mean"]))
        low = [i for i, r in enumerate(by_day) if r < floor]
        if low:
            problems.append(f"{y}: {len(low)}/{info['days']} 天的特征覆盖率 <{floor:.2f}"
                            f"（当年均值 {cov['mean']:.3f}；首日索引 {low[0]}）"
                            f"—— 上游这几天是不是写盘写了一半？")
        log(f"  {y}: {info['days']} 天 × {C} 只 = {exp:,} 行 · 覆盖 均{cov['mean']:.3f} "
            f"末日{cov['last_day_cov']:.3f} · 因子 {len(ycols)} 个"
            + (f"（另有 {len(no_obs)} 个因子本年上游无方差、整列为空，正常）" if no_obs else "")
            + (f" · ⚠️ {len(low)}/{info['days']} 天低于 {floor:.2f}" if low else ""))
        rows_total += exp
    # ---- 市场因子块（**单键** `trade_date`、每日一行）★ 不能混进上面那个循环：
    #   那里的期望行数是 `days × C`，市场块是 `days`，混着查会把正常的块报成坏的。
    if meta.get(MK):
        mcols = list((meta.get(MK) or {}).get("columns") or [])
        raw_cols = [c for c in mcols if not c.endswith(MK_Z_SUFFIX)]
        mrows_total = 0
        for y in built:
            p = year_file(root, MK, int(y))
            if not p.exists():
                problems.append(f"{y}/{MK}: 文件缺失（用 --market-only 补）")
                continue
            pf = pq.ParquetFile(p)
            days_f = set(_days_of(root, FK, int(y)))
            rm = pq.read_table(p, columns=["trade_date"]).column("trade_date").to_pylist()
            if len(rm) != len(set(rm)):
                problems.append(f"{y}/{MK}: trade_date 有重复 —— 市场块的主键是单键，必须唯一")
            if set(rm) != days_f:
                problems.append(f"{y}/{MK}: 日期轴与 factors 不一致"
                                f"（缺 {len(days_f - set(rm))} 天 / 多 {len(set(rm) - days_f)} 天）")
            if pf.metadata.num_rows != len(days_f):
                problems.append(f"{y}/{MK}: 行数 {pf.metadata.num_rows} ≠ 交易日数 {len(days_f)}")
            if mcols and tuple(pf.schema_arrow.names) != ("trade_date",) + tuple(mcols):
                problems.append(f"{y}/{MK}: 列签名与 meta 不一致"
                                f"（因子侧增删了市场因子？跑 --market-only 重建）")
            # 原值列：**只允许前缀缺值**（晚起点因子在该因子声明起点之前没有产物），
            # 首个有效值之后**不允许再有空洞** —— 那才是"数据丢了"。
            # ★ 不能写成"整列不得有 NaN"：实测 7 个 `mkt_limit_*` 声明起点是 2020-01-01、
            #   `mkt_idx_growth_value_spread20` 是 2019-07-02，本块的日期轴却与 `factors` 对齐
            #   （2018 起），reindex 出来的前缀自然全是 NaN —— 那是**设计如此**，不是上游坏了。
            if raw_cols:
                t2 = pq.read_table(p, columns=raw_cols)
                for c in t2.column_names:
                    a = t2[c].to_numpy(zero_copy_only=False)
                    fin = np.isfinite(a)
                    if fin.all():
                        continue
                    if not fin.any():
                        continue          # 整列为空 = 该因子本年还没开始，正常
                    first = int(np.flatnonzero(fin)[0])
                    hole = int((~fin[first:]).sum())
                    if hole:
                        problems.append(f"{y}/{MK}/{c}: 首个有效值之后还有 {hole} 个空洞"
                                        f" —— 上游市场因子出洞了")
            mrows_total += pf.metadata.num_rows
        log(f"  ▸ {MK}：{len(built)} 年 · {len(raw_cols)} 个因子 × 2 列 · {mrows_total:,} 行"
            + ("（晚起点因子的前缀、z 列前 252 个交易日为预热期，均属正常）" if mcols else ""))
    # ---- 取值域抽查（★ 这条是"最贵的错"的廉价探测器）
    # 只抽若干列：X 必须是**逐日横截面**的「缩尾 + z-score」，取值有界（±Z_CLAMP）、
    # 逐日均值≈0、逐日标准差≈1。
    # 历史教训：V1 时代 `clip:[0.001,0.999]` 作用在 `−rank`（值域 [−1,0]）上，把 83 个被翻转的
    # 特征整列压成常数 0.001 —— 37% 的特征等于没有，而当时所有体检都没发现。
    # 2026-09-25 起口径从 [0,1] 的 rank 换成 z-score，这套探针改成"越界 + 逐日矩"两条：
    # 前者抓"翻转到一半/裁剪口径错"，后者抓"根本没做标准化（比如误读了上游 rank 列）"。
    probe = feats[:: max(1, len(feats) // 8)][:8] if feats else []
    if probe:
        bad = []
        for y in built:
            t = pq.read_table(year_file(root, FK, int(y)), columns=["trade_date"] + probe)
            dates = np.asarray(t["trade_date"].to_pylist(), dtype=object)
            for c in probe:
                a = t[c].to_numpy(zero_copy_only=False)
                fin = np.isfinite(a)
                if not fin.any():
                    continue
                lo_v, hi_v = float(a[fin].min()), float(a[fin].max())
                if lo_v < -(Z_CLAMP + 1e-4) or hi_v > Z_CLAMP + 1e-4:
                    bad.append(f"{y}/{c}: [{lo_v:.4f}, {hi_v:.4f}] 越界")
                    continue
                # 逐日矩：只查**该列当天有效样本足够**的日子，避免稀疏列误报
                dfp = pd.DataFrame({"d": dates, "v": a})
                g = dfp.groupby("d")["v"].agg(["count", "mean", "std"])
                g = g[g["count"] >= Z_MIN_COUNT]
                if len(g) == 0:
                    continue
                off = ((g["mean"].abs() > 1e-3) | ((g["std"] - 1).abs() > 5e-3)).sum()
                if off > max(1, int(0.05 * len(g))):
                    bad.append(f"{y}/{c}: {off}/{len(g)} 天的均值/标准差不符（均{lo_v:.3f}）")
        if bad:
            problems.append(f"特征取值口径异常：{bad[:4]}（方向翻转/标准化口径有问题？）")
        log(f"  ▸ 取值口径抽查：{len(built)} 年 × {len(probe)} 列 · "
            + ("✔ 全部有界且逐日 mean≈0/std≈1" if not bad else f"✘ {len(bad)} 列异常"))

    log(f"  ▸ factors 共 {rows_total:,} 行")
    for p in problems:
        log(f"  ✘ {p}")
    log("  ✔ 校验通过" if not problems else f"  ⚠️ {len(problems)} 处需要处理")
    return {"ok": not problems, "problems": problems, "years": built, "meta": meta}


def _summary(meta: dict, log=progress_log) -> None:
    ym = meta["years"]
    tot = sum(v["rows"] for v in ym.values())
    xmb = sum((v.get("files") or {}).get(FK, {}).get("mb", 0) for v in ym.values())
    log("─" * 62)
    log(f"  trainingdata 就绪：{len(ym)} 年 · {tot:,} 行 · factors {xmb:.0f} MB")
    log(f"  轴 {meta['axis']['start']} ~ {meta['axis']['end']} · "
        f"{meta['axis']['n_days']} 天 × {meta['axis']['n_codes']} 只")
    log(f"  特征 {len(meta['columns']['features'])} · 标签 {len(meta['columns']['labels'])} · "
        f"翻转 {meta['columns']['n_flipped']} · 值口径 {meta.get('semantics', '?')}")


# ================================================================ 台账
#: 台账目录（在快照根下）。★ 放在 `trainingdata/` 里而不是模块根 ——
#: 台账记的是**这份快照**的状态，快照搬走/另建时它应该跟着走。
LEDGER_DIR = "log"


def write_ledger(root: str | Path, *, action: str,
                 rebuilt: dict[tuple[str, int], str] | None = None, log=progress_log) -> Path:
    """把快照里**每个数据文件的当前状态**写成一份**带日期的台账**（TSV）。

    用户 2026-09-20 要的：每天增量更新时生成一个新文件，好回答"某天那份快照长什么样"。
    - **一天一个文件**：`log/ledger_YYYY-MM-DD.tsv`。同日多次运行就覆盖 —— 要的是"最新台账"，
      不是运行流水。
    - **一行一个数据文件**（四块 × 九年 = 36 行，外加已启用的可选块），列取自
      `meta.json:years[*].files[*]`；`mtime` 取自文件本身，所以一眼能看出**今天动了哪几个**。
    - ★ 台账要覆盖**每一个被维护的块**，不能只写 `ALL_KINDS`：`market_factors`（与
      `amount`/`fac_sample` 同类，由日常增量维护）也必须出现在台账里，否则"某天那份快照
      长什么样"对它是**答不出来**的 —— 而 `meta["years"][y]["files"]` 里本来就有它
      （2026-09-25 修：此前台账只遍历 `ALL_KINDS`，市场块 9 行全部缺失）。
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
        f"# 本次动作={action or '-'}"
        + (f"  重建={sorted({y for _k, y in rb})}" if rb else ""),
        "\t".join(("block", "year", "rows", "cols", "bytes", "action")),
    ]
    body: list[str] = []
    kinds = list(ALL_KINDS) + [MK]
    for kind in kinds:
        for y in sorted(int(v) for v in (meta.get("years") or {})):
            rec = ((meta["years"].get(str(y)) or {}).get("files") or {}).get(kind) or {}
            f = year_file(root, kind, y)
            if not rec and not f.exists():
                continue
            body.append("\t".join((kind, str(y), str(rec.get("rows", "-")), str(rec.get("cols", "-")),
                                   str(rec.get("bytes", "-")),
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
#: 回测价格块 = 模块① `stock_daily`(7 列) + `stock_adj_factor`(adj_factor)。见 `build_year_prices`。
#: ★ 常量必须在这里（`ALL_KINDS` 在下面就要用它），构建函数在文件靠后的 prices 一节。
PK = "prices"
AMOUNT_COLUMNS = ("amount",)          # 除 KEY 外的列
#: prices 块除 KEY 外的列；与 `experiments/V2/model.py:PRICE_COLUMNS` 逐字一致（消费方按它取列）。
PRICE_COLUMNS = ("open", "high", "low", "close", "pre_close", "pct_chg", "vol", "adj_factor")
FAC_SAMPLE_N = 20                   # 抽多少列
FAC_SAMPLE_SEED = 42                # 固定种子 —— 抽样结果必须可复现，否则交付给别人对不上

#: 布局里的**全部五块**，缺一不可（2026-09-20 起 `amount` / `fac_sample` 也是正式成员；
#: 2026-09-26 起 `prices` 同样是）。★ 成员资格 = `check()` 按「双键、每日恰好 C 行」校验它、
#: `prepare()` 的 meta 保留与 `write_ledger()` 的台账都带上它。
ALL_KINDS = tuple(KINDS) + (AK, SK, PK)


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


def _record_block(meta: dict, kind: str, years: list[int], extra: dict, root: Path,
                  log=progress_log) -> dict:
    """把数据块的分区和列信息写入 meta。"""
    meta[kind] = {"years": sorted(int(y) for y in meta.get("built_years", years)
                                 if kind in meta.get("years", {}).get(str(y), {}).get("files", {})),
                  "built_at": now(), **extra}
    save_json(meta_path(root), meta)
    log(f"  ✔ {kind} 块完成，meta 已更新")


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
                             "bytes": int(p.stat().st_size)}}}
    del tab, fields
    return info


# ---------------------------------------------------------------- amount
def build_year_amount(cfg: Cfg, year: int, days: list[str], codes: np.ndarray,
                      cols: tuple[str, ...], *, log=progress_log, root: Path | None = None) -> dict:
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

    __slots__ = ("kind", "title", "cols", "build_year", "meta_extra", "keys")

    def __init__(self, kind: str, title: str, cols, build_year, meta_extra,
                 keys: tuple[str, ...] = KEY) -> None:
        self.kind = kind              #: 块名（同时是 meta 顶层 key）
        self.title = title            #: 日志里的中文名
        self.cols = cols              #: (meta) -> 列清单；fac_sample 依赖 meta 的特征清单
        self.build_year = build_year  #: (cfg, year, days, codes, cols, *, log, root) -> info
        self.meta_extra = meta_extra  #: (meta, cols) -> 写进 meta 的额外字段
        #: 该块的主键列。默认股票两键；**市场因子块只有 `trade_date` 一键**
        #: （每日一行、无股票维），所以陈旧判据里的期望列签名必须跟着变。
        self.keys = tuple(keys)


def stale_block_years(root: str | Path, kind: str, *, log=None, cfg: Cfg | None = None) -> list[int]:
    """某块的**陈旧年份** —— 各块共用一套判据：

    ① 文件不存在；
    ② 交易日集合与 `factors` 不一致（各块的共同契约就是"与 factors 同一条日期轴"）；
    ③ 列清单与期望不一致 —— 这条是为 schema 演进留的：精度从 float32 提到 float64
       这类改动只比交易日集合抓不到（产物看着"行数对得上"，消费端读列时才炸）。
    """
    root = Path(root)
    meta = load_meta(root)
    blk = BLOCKS[kind]
    cols = market_block_columns(directory=cfg.market_factors_dir) if kind == MK and cfg else blk.cols(meta)
    want = tuple(blk.keys) + tuple(cols) if meta.get("columns") else ()
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


def build_block(cfg: Cfg, kind: str, *, years: list[int] | None = None, log=progress_log,
                root: str | Path | None = None) -> dict:
    """导出某一块（全部年份或指定年份）。

    只更新指定的数据块。
      年份与日子一律**从既有 `factors` 分区读**（不重算日期轴），保证四块严格同形。
    """
    root = Path(root).resolve() if root else root_of(cfg)
    meta = load_meta(root)
    if not meta:
        raise SystemExit(f"✘ {root} 还没有 meta.json —— 先跑一次完整的 `preparingdata.py`")
    blk = BLOCKS[kind]
    cols = tuple(market_block_columns(directory=cfg.market_factors_dir) if kind == MK else blk.cols(meta))
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
    for y in _progress(want, desc=blk.title, unit="年", disable=log is None, leave=False):
        days = _days_of(root, FK, y)      # ★ 从既有 factors 分区取日子，保证四块同形
        info = blk.build_year(cfg, y, days, codes, cols, log=log, root=root)
        rec = dict(years_meta.get(str(y)) or {})
        rec.setdefault("files", {})
        rec["files"].update(info["files"])
        years_meta[str(y)] = rec
    meta["years"] = years_meta
    _record_block(meta, kind, want, blk.meta_extra(meta, cols), root, log=log)
    return {"action": kind, "years": want}


def ensure_block(cfg: Cfg, kind: str, root: Path, *, rebuilt_years=(), log=progress_log) -> list[int]:
    """把陈旧的某块补上（`prepare()` 的出口调它）。★ 必须在主 meta 写完之后 ——
    `build_block` 会再读一次 meta 再写回，放前面会被主流程的 save_json 覆盖掉。

    `rebuilt_years`：本次**主流程真的重建过**的年份（`plan.keys()`）。这些年份必须连带重建，
    因为 `stale_block_years` 的三条判据（文件在不在 / 日期轴 / 列名）**抓不到"值变了"**。

    ★ 这一条不是可选的：`fac_sample` 是 `factors` 的**逐字副本**，而且它是**默认训练块**
      （`experiments/V1/run.py` 的 `MX_DATA` 默认指向它）。2026-09-25 换值口径时，
      日期轴与列名一个字都没变 ⇒ 不给 `rebuilt_years` 的话 `fac_sample` **不会重建**，
      会静静留着旧口径的值，而下游读的是它。
    """
    stale = sorted(set(stale_block_years(root, kind, log=log, cfg=cfg)) | {int(y) for y in rebuilt_years})
    if not stale:
        return []
    build_block(cfg, kind, years=stale, log=log, root=root)
    return stale


# ---------------------------------------------------------------- fac_sample
def build_year_fac_sample(cfg: Cfg, year: int, days: list[str], codes: np.ndarray,
                          cols: tuple[str, ...], *, log=progress_log, root: Path | None = None) -> dict:
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


# ---------------------------------------------------------------- prices（回测价格）
#: 回测价格块：模块① 的原始日线（7 列）+ 复权因子，按 `(trade_date, stock_code)` 落进快照。
#:
#: ★ 用户 2026-09-26 定：**从模块① 复制一块进来**，让消费方（`experiments/V2`）只读本地
#:   快照、不再自己连 `datadownload/`。这是对 2026-09-25"价格不属于本层"那条边界的
#:   **有意反转**，理由与仍然成立的部分都写在模块头部；核心是**与因子同轴刷新**：
#:   价格与 factors 落在同一条日期轴上，不会出现两轴错位。
#: ★ 只复制、不加工：不复权、不前向填充、不裁停牌 —— 加工口径在消费方（V2 的 `Prices`）。
#: 常量 `PK` / `PRICE_COLUMNS` 定义在第 5 节的常量区（`ALL_KINDS` 要先用到它们）。


def build_year_prices(cfg: Cfg, year: int, days: list[str], codes: np.ndarray,
                      cols: tuple[str, ...], *, log=progress_log, root: Path | None = None) -> dict:
    """导出**一年**的 `prices/year=YYYY/data.parquet` —— 回测用的原始价 + 复权因子。

    ★ 与 `amount` 同一套读法：直读模块① 的两张原始表、按 `(trade_date, stock_code)`
      显式映射落格（**不按行号** —— 上游按列号落格踩过静默错位的坑），
      日期直接用 `days`（已规范成 `YYYY-MM-DD`）。
    ★ 存 **float64**，与模块① 逐位一致：
      ① 价格要参与涨跌停的**分位取整**判定（`pre_close × 1.1` 四舍五入到分），float32 的
         尾差足以让边界样本翻面；② `experiments/V2` 的 `Prices.verify_labels()` 会拿它重算
         5 个 horizon 的收益与 `target` 逐格对拍（阈值 1e-5）。
    ★ 缺格（停牌 / 池内票当年没有日线）保持 **NaN** —— 消费方正是靠这个 NaN 判"买不进"，
      填 0 或前向填充都会把"不可交易"变成"可以交易"。
    """
    t0 = time.time()
    C = len(codes)
    if not days:
        raise SystemExit(f"✘ {year} 年没有交易日，无法导出 prices 块")
    days_index = pd.Index(np.asarray(days, dtype=object))
    codes_index = pd.Index(codes)
    out: dict[str, np.ndarray] = {}
    dropped = 0
    daily = tuple(c for c in cols if c != "adj_factor")
    for table, names in (("stock_daily", daily), ("stock_adj_factor", ("adj_factor",))):
        p = Path(cfg.raw_data) / table / f"year={int(year)}" / "data.parquet"
        if not p.exists():
            raise SystemExit(f"✘ 模块① 没有 {year} 年的 {table}：{p}\n"
                             f"    （`MX_RAW_DATA` 指错了吧？当前 = {cfg.raw_data}）")
        df = pd.read_parquet(p, columns=["stock_code", "trade_date", *names])
        df["trade_date"] = df["trade_date"].astype(str).str[:10]
        df["stock_code"] = df["stock_code"].astype(str)
        di = days_index.get_indexer(df["trade_date"].to_numpy(dtype=object))
        ci = codes_index.get_indexer(df["stock_code"].to_numpy(dtype=object))
        ok = (di >= 0) & (ci >= 0)
        keep = di[ok] * C + ci[ok]
        if len(np.unique(keep)) != len(keep):
            raise SystemExit(f"✘ {year}/prices：模块① {table} 里有重复的 (交易日, 股票) 行 —— "
                             f"落格会静默覆盖，先查上游")
        for name in names:
            arr = np.full(len(days) * C, np.nan, dtype=np.float64)
            arr[keep] = df[name].to_numpy(dtype=np.float64)[ok]
            out[name] = arr
        dropped += int((~ok).sum())
        del df
    info = _write_block_year(root if root is not None else root_of(cfg), PK, year, days, codes, out)
    filled = int(np.isfinite(out["close"]).sum())
    log(f"  {year}: prices 块 {info['days']} 天 × {C} 只 × {len(cols)} 列 = {info['rows']:,} 行 · "
        f"收盘价非空 {filled / max(1, info['rows']):.1%}（缺口=停牌/退市，消费方判不可交易） · "
        f"{info['files'][PK]['mb']}MB"
        + (f" · 丢弃 {dropped:,} 行（日期/代码轴之外）" if dropped else "")
        + f" · {round(time.time() - t0, 1)}s")
    return info


# ---------------------------------------------------------------- market_factors


# ---------------------------------------------------------------- market_factors
#: 市场因子块：上游 `data/market_factors/<名>/year=YYYY/` 的**每日一行标量**。
#:
#: ★ 它与其余各块有两处根本不同，所以走**自己的写入器**、不用 `_write_block_year`：
#:   ① 主键只有 `trade_date`（无股票维）—— `_write_block_year` 硬校验 `n_days×C` 行
#:      并强制生成 `stock_code` 列（`len(v) != n` 直接 SystemExit）。
#:   ② 每个因子落**两列**：原值 + 滚动 z-score。
MK = "market_factors"
MK_Z_SUFFIX = "_z252"


def market_block_columns(meta=None, *, directory: Path | None = None) -> list[str]:
    """市场块的全部列 = 每个因子「原值一列 + 滚动 z 一列」。

    ★ 直接读**上游目录**而不是记在 meta 里：这样因子侧新增一个市场因子时，
      列签名与产物立刻不一致 ⇒ `stale_block_years` 判陈旧 ⇒ 下次自动重建。
      记在 meta 里就会漏掉新增（`fac_sample` 靠 `meta['columns']['features']` 是对的，
      因为它的列清单本来就由主流程维护）。
    """
    d = directory if directory is not None else MARKET_FACTORS_DIR
    out: list[str] = []
    names = _product_names(d)
    if any(n.endswith(MK_Z_SUFFIX) for n in names):
        raise ValueError(f"市场因子原名不能以 {MK_Z_SUFFIX} 结尾，以免与派生列重名")
    for name in names:
        out += [name, name + MK_Z_SUFFIX]
    return out


#: 本轮进程内的市场因子全历史缓存（`{(目录, 名): Series}`）。整块只有 ~1MB，读一次就够；
#: 不加缓存的话「9 个年份各建一次 ⇒ 每个因子被重读 9 遍」。
_MARKET_CACHE: dict[tuple, pd.Series] = {}


def _market_series(cfg: Cfg, name: str) -> pd.Series:
    """某市场因子的**全历史**序列（按日期升序）。

    ★ 必须取全历史、不能只读当年：滚动 z 要往前看 252 个交易日，只读当年的话
      **每年开头 252 天会算错**（而"值取决于我们碰巧从哪一年开始建"本身就是错的锚点）。
      这与因子侧"预热锚定输出年份"是同一个原则。
    """
    files = [year_file(cfg.market_factors_dir, name, y) for y in years_of(cfg.market_factors_dir, name)]
    files = [p for p in files if p.is_file()]
    key = (str(cfg.market_factors_dir), name, tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in files))
    hit = _MARKET_CACHE.get(key)
    if hit is not None:
        return hit

    parts = []
    for p in files:
        parts.append(pq.ParquetFile(p).read(columns=["trade_date", "value"], use_threads=False).to_pandas())
    if not parts:
        s = pd.Series(dtype=np.float64)
    else:
        df = pd.concat(parts, ignore_index=True)
        df["trade_date"] = df["trade_date"].astype(str).str[:10]
        if df["trade_date"].duplicated().any():
            raise ValueError(f"✘ 市场因子 {name}：跨年分区有重复交易日")
        s = df.sort_values("trade_date", kind="stable").set_index("trade_date")["value"]
        s = s.astype(np.float64)
    _MARKET_CACHE[key] = s
    return s


def _rolling_z(s: pd.Series, window: int = MK_Z_WINDOW) -> pd.Series:
    """过去 `window` 个交易日（**含当日**）的 z-score。

    ★ 只用 t 及之前 ⇒ PIT 安全。窗口不足 `window` 天时为 NaN
      （所以 2018 全年该列为 NaN —— 原值列始终有值，信息不丢）。
    ★ 不做缩尾：实测 61 个市场因子的 |偏度| 全部 < 5，没有厚尾，缩尾只会白添一个参数。
    """
    if s.empty:
        return s
    m = s.rolling(window=window, min_periods=window).mean()
    sd = s.rolling(window=window, min_periods=window).std(ddof=0)
    with np.errstate(all="ignore"):
        z = (s - m) / sd
    return z.where(np.isfinite(z))


def build_year_market(cfg: Cfg, year: int, days: list[str], codes: np.ndarray,
                      cols: tuple[str, ...], *, log=progress_log, root: Path | None = None) -> dict:
    """导出**一年**的 `market_factors/year=YYYY/data.parquet`（每日一行、`trade_date` 单键）。"""
    t0 = time.time()
    root = root if root is not None else root_of(cfg)
    names = [c for c in cols if not c.endswith(MK_Z_SUFFIX)]
    idx = pd.Index(np.asarray(days, dtype=object))
    fields: dict[str, pa.Array] = {"trade_date": pa.array(np.asarray(days, dtype=object))}
    def one(nm):
        s = _market_series(cfg, nm)
        if s.empty:
            raw = np.full(len(days), np.nan)
            z = np.full(len(days), np.nan)
        else:
            raw = s.reindex(idx).to_numpy(dtype=np.float64)
            z = _rolling_z(s).reindex(idx).to_numpy(dtype=np.float64)
        if np.isinf(raw).any() or np.isinf(z).any():
            raise ValueError(f"✘ 市场因子 {nm}：出现无穷值")
        return nm, raw.astype(np.float32), z.astype(np.float32)

    jobs = min(cfg.jobs, max(1, len(names)))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = _bounded_results(pool, one, names, jobs)
        values = {}
        for nm, raw, z in _progress(results, total=len(names), desc=f"{year} 市场因子", unit="因子",
                               disable=log is None, leave=False):
            values[nm] = (raw, z)
    for nm in names:
        fields[nm] = pa.array(values[nm][0])
        fields[nm + MK_Z_SUFFIX] = pa.array(values[nm][1])
    tab = pa.table(fields)
    p = year_file(root, MK, year)
    atomic_parquet(tab, p)
    info = {"year": int(year), "days": len(days), "rows": int(tab.num_rows),
            "start": days[0], "end": days[-1],
            "files": {MK: {"rows": int(tab.num_rows), "cols": int(tab.num_columns),
                           "mb": round(p.stat().st_size / 1e6, 1),
                           "bytes": int(p.stat().st_size)}}}
    log(f"  {year}: market_factors {info['rows']} 行 × {len(names)} 因子 × 2 列 "
        f"· {info['files'][MK]['mb']}MB · {round(time.time() - t0, 1)}s")
    del tab, fields
    return info


def ensure_market(cfg: Cfg, root: Path, *, rebuilt_years=(), log=progress_log) -> list[int]:
    """市场因子块：显式建立后由日常增量维护（与 amount / fac_sample 同一套）。"""
    if not market_block_columns(directory=cfg.market_factors_dir):
        log("  ⊘ 上游没有市场因子目录，跳过 market_factors 块")
        return []
    return ensure_block(cfg, MK, Path(root), rebuilt_years=rebuilt_years, log=log)


#: 派生块的登记表。★ 放在这里（而不是常量区）是因为它引用 `build_year_*` ——
#: 那些函数必须先定义；模块级 dict 只要在**调用前**建好就行。
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
    PK: Block(kind=PK, title="回测价格块 prices",
              cols=lambda meta: PRICE_COLUMNS,          # noqa: ARG005  (Block.cols 的签名)
              build_year=build_year_prices,
              meta_extra=lambda meta, cols: {"columns": list(cols),
                                             "source": "模块① stock_daily + stock_adj_factor"}),
    # ★ 单键块：`keys` 必须显式给，否则陈旧判据会按 (trade_date, stock_code) 两键去对列签名。
    MK: Block(kind=MK, title="市场因子块 market_factors",
              cols=market_block_columns,
              build_year=build_year_market,
              meta_extra=lambda meta, cols: {"columns": list(cols),
                                             "z_window": MK_Z_WINDOW,
                                             "n_factors": len(cols) // 2},
              keys=("trade_date",)),
}

# ============================================================================
# 6. 命令行
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="preparingdata.py",
        description="初加工：上游因子 → trainingdata/"
                    "（factors 股票因子 / target 标签 / amount 流量 / "
                    "market_factors 市场因子 / prices 回测价格，附 fac_sample 抽样副本）")
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--full", "-f", action="store_true", help="全量重建所有入选年份")
    m.add_argument("--incremental", "-i", action="store_true", help="强制增量（默认自动判断）")
    m.add_argument("--check", "-c", action="store_true", help="只校验不写")
    m.add_argument("--meta-only", action="store_true",
                   help="只补 meta 的派生字段（特征集等）—— ★ 不碰数据文件、不读因子侧")
    m.add_argument("--amount-only", action="store_true",
                   help="★ 只建/刷新 amount（每日成交额）")
    m.add_argument("--fac-sample-only", action="store_true",
                   help="★ 只建/刷新 fac_sample（因子抽样）")
    m.add_argument("--market-only", action="store_true",
                   help="★ 只建/刷新 market_factors（市场因子：每日一行、原值+滚动z 两列）")
    m.add_argument("--prices-only", action="store_true",
                   help="★ 只建/刷新 prices（回测价格：模块① 的 7 列原始价 + 复权因子）")
    ap.add_argument("--years", "-y", type=int, nargs="+", default=None,
                    help="只加工这些年份（例：--years 2025 2026）")
    ap.add_argument("--jobs", "-j", type=int, default=DEFAULT_JOBS,
                    help=f"读取/对齐/标准化并行进程数（默认 {DEFAULT_JOBS}，按容器 CPU 配额选择）")
    ap.add_argument("--source-data", type=Path, default=None, metavar="DIR",
                    help="直接读取的因子 data 根目录（含 factors/、market_factors/）")
    ap.add_argument("--factors-only", action="store_true",
                    help="只打包因子、标签、市场因子及抽样块，不读取原始成交额/价格表")
    ap.add_argument("--direction-policy", choices=("preserve", "raw"), default="preserve",
                    help="preserve 沿用本地已有方向、未知因子保持原方向；raw 全部保持原方向")
    ap.add_argument("--progress", choices=("auto", "bars", "off"), default="auto",
                    help="tqdm 进度：auto 终端显示、bars 强制显示、off 关闭")
    ap.add_argument("--lookback", "-l", type=int, default=None,
                    help="增量回溯窗口（交易日）。默认 = 最长标签 h + 1 + 7 —— "
                         "h+1 是硬下限（太短会让 20d 标签出了窗口就再也不刷新、永久停在 NaN），"
                         "7 是冗余覆盖")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="★ 写到**另一份目录**（相对脚本所在目录），现有 trainingdata/ 完全不动")
    ap.add_argument("--features-from", default=None, metavar="META_JSON",
                    help="★ 把特征清单**钉在**指定 meta.json 的 `columns.features` 上"
                         "（默认 = 上游当前的因子集，上游增删就跟着变）。给目录或 meta.json 都行。"
                         "用途：① 与某份旧产物逐位对拍；② 冻结列集，不随上游漂移")
    ap.add_argument("--extra-features", default=None, metavar="NAMES",
                    help="兼容旧参数：校验落盘名字；配 --features-from 时追加列。默认已纳入所有落盘因子。"
                         "支持逗号分隔或 JSON 文件（feature_names/features/列表）")
    return ap


def main(argv: list[str] | None = None) -> int:
    global _PROGRESS_MODE
    a = build_parser().parse_args(argv)
    _PROGRESS_MODE = a.progress
    out = None if not a.out else (ROOT / a.out).resolve()
    cfg = Cfg(trainingdata=out, source_data=a.source_data)
    cfg.jobs = max(1, a.jobs)
    cfg.factors_only = a.factors_only
    cfg.direction_policy = a.direction_policy
    if out is not None:
        print(f"  ★ 本次写到**另一份目录**：{out}（现有 {ROOT / 'trainingdata'} 不动）")

    if a.check:
        r = check(cfg)
        return 0 if r["ok"] else 1
    if a.meta_only:
        attach_meta_fields(cfg)
        return 0
    if a.market_only:
        r = build_block(cfg, MK, years=a.years, log=progress_log)
        return 0
    if a.amount_only:
        r = build_block(cfg, AK, years=a.years, log=progress_log)
        write_ledger(root_of(cfg), action=r.get("action", ""),
                     rebuilt={(AK, int(y)): AK for y in (r.get("years") or [])}, log=progress_log)
        return 0
    if a.fac_sample_only:
        r = build_block(cfg, SK, years=a.years, log=progress_log)
        write_ledger(root_of(cfg), action=r.get("action", ""),
                     rebuilt={(SK, int(y)): SK for y in (r.get("years") or [])}, log=progress_log)
        return 0
    if a.prices_only:
        r = build_block(cfg, PK, years=a.years, log=progress_log)
        write_ledger(root_of(cfg), action=r.get("action", ""),
                     rebuilt={(PK, int(y)): PK for y in (r.get("years") or [])}, log=progress_log)
        return 0
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
        print(f"  ★ 特征清单钉在 {a.features_from} 的 {len(feats)} 列（不随上游漂移）")
    else:
        print("  ★ 直接打包 data/factors 下所有落盘股票对象（不依赖上游代码、配置、注册表或 state）")

    mode = "full" if a.full else ("incremental" if a.incremental else "auto")
    if out is not None and not a.full:
        print("  ⚠️ --out 建议配 --full 用（另建目录的语义就是「从头建一份」）")

    extra = None
    if a.extra_features:
        raw = a.extra_features.strip()
        fp = Path(raw).expanduser()
        if fp.is_file():                       # JSON 文件：feature_names / features / 裸列表
            data = load_json(fp, None)
            if isinstance(data, dict):
                data = data.get("feature_names") or data.get("features") or []
            if not isinstance(data, list) or not data:
                print(f"  ✘ --extra-features 的 JSON 里没有可用的名字列表：{a.extra_features}")
                return 1
            extra = [str(x) for x in data]
        else:                                  # 逗号分隔的名字
            extra = [x.strip() for x in raw.split(",") if x.strip()]
        if not extra:
            print(f"  ✘ --extra-features 没解析出任何名字：{a.extra_features}")
            return 1
        print(f"  ★ 校验/追加 {len(extra)} 个落盘因子：{extra[:5]}"
              f"{' …' if len(extra) > 5 else ''}")

    r = prepare(cfg, mode=mode, years=a.years, jobs=max(1, a.jobs),
                lookback=a.lookback,
                features=feats, direction=direction, extra_features=extra)
    write_ledger(root_of(cfg), action=r.get("action", ""),
                 rebuilt=r.get("rebuilt") or {}, log=progress_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
