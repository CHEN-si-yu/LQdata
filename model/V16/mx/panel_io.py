"""产物 I/O —— `trainingdata/` 的**全部**读写原语，零上游依赖。

## 为什么单独一层

`prepared.py`（初加工）是**构建期**代码：它要读上游 230 个因子、建股票池、算覆盖率，
天然依赖 `fea` + `featureengineering/`。而模型侧（训练/评价/回测）**完全不碰上游**，
只需要"把 Parquet 读成表"这一件事。

原先这两件事混在一个文件里，导致**每个模型单元都被迫拖上整个上游依赖** ——
而这正是"单元无法独立拷贝出去"的根因（见 `README §1「独立单元」`）。
本模块把纯 I/O 剥出来：

    prepared.py   →  panel_io.py  ←  data.py / unit.py / prices.py / backtest.py
        （构建期，带 fea）            （运行期，零 fea）

★ 本模块**只允许** import：标准库 + numpy/pandas/pyarrow + `mx.config` + `mx.state`。
  任何 `fea` / `featureengineering` / `datadownload` 的引用都是架构违规。

## 四块产物

    trainingdata/
      meta.json                          清单 / 边界 / 逐年摘要 / 覆盖率 / 面板指纹
      X/year=YYYY/data.parquet           trade_date, stock_code, <306 特征>   ← 输入
      Y/year=YYYY/data.parquet           trade_date, stock_code, <5 标签>     ← 输出
      universe/year=YYYY/data.parquet    trade_date, stock_code, in_universe
      P/year=YYYY/data.parquet           trade_date, stock_code, <价格/成交量> ← 回测用
      sample/                            同构四件套（范围小，冒烟/调试用）

★★ 关于 `P`（价格块）：**它存的是"原料"，不是算好的价格**。
   `hfq_*`（后复权价）在上游是 `ffill(原始价) × ffill(复权因子)` 的**逐点**乘积
   （`fea/prices.py:177-181`），不是从窗口起点累乘 —— 所以唯一可能的窗口依赖来自
   `ffill` 在**窗口首行无历史可续**。若在导出时就把 `hfq_*` 算死，等于把"导出窗口"
   焊进数据里：换一个读取窗口，首行就对不上了。
   ⇒ 导出原料、由 `mx/prices.py` 对**任意窗口**套用同一套运算，才是逐位保真的做法。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import state
from .config import Cfg

# ---- 四块产物的名字
XK, YK, UK, PK = "X", "Y", "universe", "P"
KINDS = (XK, YK, UK, PK)
KEY = ("trade_date", "stock_code")

#: `P` 块承载的列（除 KEY 外）。**原料口径**，见模块注释。
#: 由它们可以推出回测/策略要的全部字段：
#:   hfq_{open,high,low,close} = ffill(原始) × ffill(adj_factor)
#:   traded = isfinite(vol) · ret1 = pct_chg/100
#:
#: ★ 为什么**没有** `float_share`：它唯一的用途是推 `turnover`，而
#:   `turnover` 在 `backtest.py`/`strategy.py`/`analyze.py` 里**零引用**
#:   （`analyze.py` 里的 `turnover` 是换手率**评价指标**，与本字段同名不同物）。
#:   带上它等于为一条没人走的路径多存一列、多一处要保真的派生。
#: ★ `amount`（成交额）保留：用户明确要"价格与交易量"，且它零派生、零风险。
PRICE_COLUMNS = ("open", "high", "low", "close", "vol", "amount",
                 "pct_chg", "adj_factor")


# ================================================================ 路径
def root_of(cfg: Cfg, sample: bool = False) -> Path:
    """产物根目录。`sample=True` 取小样本树（`<root>/sample/`）。"""
    return cfg.trainingdata_sample if sample else cfg.trainingdata


def meta_path(root: Path) -> Path:
    return Path(root) / "meta.json"


def year_file(root: Path, kind: str, year: int) -> Path:
    return Path(root) / kind / f"year={int(year)}" / "data.parquet"


def years_of(root: Path, kind: str = XK) -> list[int]:
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


def has_kind(root: Path, kind: str) -> bool:
    """该块是否已经建过（**只**看有没有年份分区，不算 meta）。"""
    return bool(years_of(root, kind))


def load_meta(root: Path) -> dict:
    return state.load_json(meta_path(root), {}) or {}


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


def axis_days(root: Path, kind: str = XK) -> list[str]:
    """产物里实际出现的交易日（升序）—— 只读日期列。"""
    out: set[str] = set()
    for y in years_of(root, kind):
        p = year_file(root, kind, y)
        col = pq.read_table(p, columns=["trade_date"]).column("trade_date").to_pylist()
        out.update(str(d)[:10] for d in col)
    return sorted(out)


def read_price_frame(root: Path, *, years: list[int] | None = None,
                     start: str | None = None, end: str | None = None,
                     codes: list[str] | None = None,
                     columns: tuple[str, ...] | list[str] | None = None) -> pd.DataFrame:
    """读 `P` 块的原料列。语义同 `read_frame(root, PK, ...)`，只是把列默认值钉死。

    ★ `start` 往前多读一年由**调用方**（`mx/prices.py`）负责 —— 那是 ffill 语义的要求，
      不是 I/O 层该猜的事情。这里保持"读你让我读的区间"。

    ★ `years` 是 2026-09-20 补的：`mx/prices.py:_load` 一直是按"与窗口相交的年份分区"
      去要数据的（`IO.read_price_frame(root, years=years)`），但这个函数**从来没有这个参数**
      —— 于是**每一次回测都在这里抛 TypeError**，被 `analyze` 的 try/except 接住、
      降级成"不套可买掩码"的口径（打了 ⚠️ 但流程照跑）。
      后果是**回测路径自诞生起就没真正跑通过**，而由于降级得很安静，谁也没发现。
      ⇒ 教训：`except` 兜底必须同时兜住"功能没了"，否则它会替你藏住 bug。
    """
    cols = list(columns) if columns is not None else list(PRICE_COLUMNS)
    return read_frame(root, PK, years=years, columns=cols, codes=codes, start=start, end=end)
