#!/usr/bin/env python3
"""数据字典生成器 —— 扫描本地 parquet，产出 `DATA_CATALOG.md`。

为什么需要它：因子模块只认 `data/**/*.parquet`，而 parquet 本身不自带
"这列是什么语义、能不能为空、取值范围多大、主键到底唯不唯一"。
靠人翻 spec.py + 手工 head 几张表，既慢又会随数据增长而过时。
把"扫本地 → 出文档"固化成一条命令，字典就永远和服务端数据同步。

为什么全程只读本地、绝不发请求：下载进程正在全速跑，任何额外请求都会
抢限速配额（服务端 280 次/分钟是全进程共享的），而字典要的信息本地全有。
所以本文件**刻意不 import lingqi.client / lingqi.engine**，从源头上杜绝误发请求。

为什么不能直接 `pd.read_parquet` 全读：最大的表全史约 1800 万行，整表进内存
会 OOM。所以按"代价从低到高"分四级取数，能便宜拿到就绝不读数据（见 `_scan_dataset`）：

    1. 行数 / 分区 / 占用        ← 文件元数据（只读 footer，零解压）
    2. 非空率 / 取值范围         ← parquet 列统计（零解压，元数据里就有）
    3. 样例值                    ← 只读"代表性分区的首个行组"，且有字节预算
    4. 主键唯一性 / 精确基数     ← 只读 keys 列；整表读仅限小表

下载进程同时在写盘：`store` 的写是"先写 .tmp 再 os.replace"的原子写，
所以读到的永远是完整文件；但仍对每个文件单独 try，坏一个只跳过不崩。

用法：
    python scripts/catalog.py                     # 全量扫描 + 生成 DATA_CATALOG.md
    python scripts/catalog.py --quick             # 跳过字段级统计（大表友好）
    python scripts/catalog.py --dataset stock_daily
    python scripts/catalog.py --list              # 只列数据集，不扫描
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lingqi import spec as spec_mod                       # noqa: E402
from lingqi import store                                  # noqa: E402
from lingqi.manifest import Manifest                      # noqa: E402
from lingqi.progress import fmt_rows                      # noqa: E402

# ---------------------------------------------------------------- 阈值
# 小表阈值：整表读一遍拿到"精确"的非空率/唯一值数/样例，比逐列查元数据更省事，
# 但必须同时卡行数和解压后体积 —— 200 行 × 200 列同样是几百万个格子。
SMALL_FULL_ROWS = 1_000_000
SMALL_FULL_BYTES = 1 << 30            # 1 GB（parquet 里记录的解压后大小）
# 采样：只读一个行组的前若干行，给文档放样例值用，不参与任何统计口径
SAMPLE_ROWS = 300
SAMPLE_BUDGET_BYTES = 256 << 20       # 单个行组解压超此值就不采样，只留元数据统计
# 主键唯一性：整表读（跨分区精确）的行数上限；超了改为逐分区统计
GLOBAL_UNIQ_ROWS = 5_000_000
# 单个分区文件超过这个行数就不读 keys 了（分钟级表一年能到千万行）
PART_SCAN_MAX_ROWS = 12_000_000

MD_PATH = ROOT / "DATA_CATALOG.md"
META_RE = "catalog-meta:"             # MD 内嵌元信息标记，供 --dataset 增量合并

_GROUP_ORDER = ["basic", "stock", "index", "tdx", "dc", "ths", "dump"]
_GROUP_CN = {
    "basic": "基础", "stock": "股票", "index": "指数",
    "tdx": "TDX 板块", "dc": "DC 板块", "ths": "THS", "dump": "特殊通道",
}

# 典型用途：**是给下游的引导，不是结论**。能写死就写死（下游最关心这几张），
# 写不动的按数据形态给个方向，避免字典里出现一片"暂无"。
_USAGE = {
    "basic_calendar": "所有因子的日期轴：对齐交易日、算持有期、避开非交易日",
    "stock_list": "股票池构建（剔退市/ST/次新）、行业与地域中性化分组",
    "stock_daily": "量价因子的底座：动量/反转/波动率/换手/流动性",
    "stock_daily_adj": "同上，但价格连续，做跨期收益必须用它（前复权）",
    "stock_adj_factor": "自建复权：与原始价相乘即得后复权价，也是除权事件源",
    "stock_adj_factor_changes": "除权除息日事件；触发复权类数据的回刷",
    "stock_finance": "日频估值因子：PE/PB/PS/换手，可与量价做正交",
    "stock_financial_indicator": "基本面因子：ROE/毛利率/负债率/成长性",
    "stock_income": "基本面因子：营收与利润增速（PIT 注意 ann_date）",
    "stock_balancesheet": "基本面因子：资产结构/杠杆/营运资本",
    "stock_cashflow": "基本面因子：经营现金流质量、应计项",
    "stock_forecast": "业绩预告事件：预告类型与幅度，财报季的短周期信号",
    "stock_holder_number": "股东户数变化率：筹码集中度 / 散户拥挤度",
    "stock_pledge_stat": "股权质押比例：质押风险因子、风险事件预警",
    "stock_margin_detail": "两融：融资余额变化、杠杆情绪、多空分歧",
    "stock_limit_up": "涨停生态：连板高度、封单强度、炸板率（情绪周期）",
    "stock_limit_list": "涨跌停统计：涨跌停家数比、市场情绪温度",
    "stock_main_fund_flow": "资金流：大中小单净流入、主力净额占比",
    # stock_dc_block_fund_flow / stock_ths_block_fund_flow / ths_hot
    # 均已于 2026-09-19 彻底删除（时间覆盖不足，见 registry.py）
    "stock_st_info": "ST 状态：股票池剔除、风险警示事件",
    "stock_suspension": "停牌：可交易性过滤、复牌事件",
    "stock_dragon_tiger": "龙虎榜机构/游资席位：资金属性因子",
    "stock_top_list": "龙虎榜每日明细：上榜原因、净买卖额",
    # stock_kline 已于 2026-09-16 删除：周/月线语义错误（实为「抓取范围∩周期」的聚合）
    # 且与 stock_daily 列完全相同（零额外信息）。需要周/月频请从 stock_daily 现算。
    "stock_cyq_chips": "筹码分布：获利盘比例、成本集中度",
    "stock_cyq_perf": "筹码收益：成本分位数、加权成本与获利比例（原始单位）",
    "stock_market_distribution_history": "全市场涨跌分布：市场宽度 / 情绪指标",
    "index_daily": "基准与相对强弱：指数收益、Beta 估计、择时信号",
    "index_history": "指数分钟级：日内动量、隔夜跳空、高频波动",
    "index_ths_daily": "同花顺板块指数日线：板块轮动与相对强弱",
    # index_ths_sector_categories / index_ths_constituent_stocks / tdx_block_stocks
    # 已于 2026-09-19 彻底删除（快照表，无历史版本，回测必然前视）
    "tdx_blocks": "TDX 板块字典：板块归属映射",
    "tdx_daily": "板块日线：板块动量/轮动",
    "tdx_minute": "板块分钟线：日内板块强度",
    "dc_blocks": "DC 板块字典：板块归属映射",
    "dc_daily": "板块日线：板块动量/轮动",
    "stock_daily_dump": "分钟级量价：日内因子（仅最近 90 天）",
    "stock_history_5min": "全市场 5 分钟量价（每只每日 48 根）：日内动量/波动/流动性因子",
}

_USAGE_HINT = (
    (("adj", "factor"), "复权与因子计算的基础数据"),
    (("fund_flow", "flow"), "资金流因子：大小单净额与占比"),
    (("holder", "pledge", "margin"), "筹码/杠杆类因子"),
    (("income", "balance", "cashflow", "financial", "finance"), "基本面因子"),
    (("kline", "daily", "minute", "history"), "量价因子：动量/波动/流动性"),
    (("blocks", "constituent", "category", "sector"), "板块归属与选股域"),
    (("limit", "hot", "forecast"), "情绪与事件类因子"),
)


def _guess_usage(name: str) -> str:
    if name in _USAGE:
        return _USAGE[name]
    low = name.lower()
    for keys, text in _USAGE_HINT:
        if any(k in low for k in keys):
            return text + "（按字段名推断，用前请核对）"
    return "待确认 —— spec 未标注用途，请按字段自行判断"


# ---------------------------------------------------------------- 取数工具
def _fmt_size(n: int) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n}B"


def _fmt_num(v) -> str:
    """统计量显示：浮点保留 6 位有效数字，避免 0.30000000000000004 这种噪声。"""
    if isinstance(v, float):
        if v != v:                       # NaN
            return "NaN"
        if v == 0:
            return "0"                   # parquet 统计里会出现 -0.0，显示成 0 更少干扰
        # 金额/成交量这类大数用千分位，科学计数法（7.5961e+06）在字典里没法读
        if abs(v) >= 1e5 and abs(v) < 1e15:
            return f"{v:,.0f}"
        return f"{v:.6g}"
    s = str(v)
    # 空串是"有值但不是有效值"，直接显示出来会让表格看起来像坏了，标明白更好
    return "（空串）" if s == "" or (s.strip() == "" and s != "") else s


# ---- 字符串列的"格式体检"：混格式是下游最容易踩的坑（同一列既是 09:25:00 又是 95947）
_FMT_PATTERNS = (
    ("日期时间", re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}")),
    ("日期 YYYY-MM-DD", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("日期 YYYYMMDD", re.compile(r"^\d{8}$")),
    ("时间 HH:MM:SS", re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")),
    ("纯数字串", re.compile(r"^[+-]?[\d.]+([eE][+-]?\d+)?$")),
)


def _classify(v: str) -> str:
    for label, rx in _FMT_PATTERNS:
        if rx.match(v):
            return label
    return "文本"


def _detect_format(values: list[str], bounds: tuple = ()) -> tuple[str, bool]:
    """看样例值（外加列统计的 min/max）猜列格式，并判断是否"一列混了几种格式"。

    为什么要带上 min/max：混格式的值往往不在前几百行里（实测 first_limit_time
    的样例全是 `09:25:00`，而全表 max 是 `95947`），只靠样例会漏。
    只对 string 列做，且目标是给下游提示而非数据校验，所以判据偏保守：
    只有「日期/时间」与别的格式同列出现才算异常。
    """
    kinds: set[str] = {_classify(v) for v in values}
    bl = [_classify(b) for b in bounds if b]
    if not values:
        # 没有样例值时只有 min/max 两个点，两者一致才敢说这列是什么格式
        if len(bl) != 2 or bl[0] != bl[1]:
            return "", False
    kinds |= set(bl)
    if not kinds:
        return "", False
    if len(kinds) == 1:
        only = next(iter(kinds))
        return ("" if only in ("文本",) else only), False
    # 判据：只要"日期/时间"与别的格式同列出现就算混格式。反过来，
    # 「纯数字串 + 文本」很常见（代码列里夹着说明），不值得报。
    timeish = {k for k in kinds if "日期" in k or "时间" in k}
    if timeish and len(kinds) > 1:
        return "＋".join(sorted(kinds)), True
    if len(kinds) == 1 and timeish:
        return next(iter(timeish)), False
    return "", False


def _dtype_name(t) -> str:
    s = str(t)
    # 落盘用的是 large_string，对下游没区别，统一显示 string
    return s.replace("large_string", "string")


def _type_family(t: str) -> str:
    """判"类型是否漂移"时用的归类：string / large_string 是同一族。

    不归类的后果是满屏"large_string×14 / string×4"这种无害告警，
    真正要命的分区差异（int64 vs double、整列 null）反而被淹掉。
    """
    return "string" if t in ("string", "large_string") else t


def _files_of(data_root: Path, name: str) -> list[Path]:
    """与 store.read_dataset 保持同一套取文件规则，避免两处口径不一致。"""
    d = data_root / name
    if not d.exists():
        return []
    files = sorted(d.glob("year=*/data.parquet"))
    flat = d / "data.parquet"
    if flat.exists():
        files.append(flat)
    return files


def _year_label(f: Path) -> str:
    parent = f.parent.name
    return parent.split("=", 1)[1] if parent.startswith("year=") else "快照"


@dataclass
class ColInfo:
    name: str
    dtype: str
    null_rate: float | None = None       # 非空率的补数：空值占比
    n_unique: int | None = None          # None = 未计算（大表且未采样）
    lo: str | None = None
    hi: str | None = None
    samples: list[str] = field(default_factory=list)
    counts: list[tuple[str, int]] = field(default_factory=list)   # 低基数取值分布
    src: str = "stats"                   # full | stats | sample
    fmt: str = ""                        # 字符串列检测到的格式（如"日期 YYYYMMDD"）
    mixed: bool = False                  # 同一列混了几种格式

    @property
    def kind(self) -> str:
        if self.counts:
            return "cat"
        return "range"


@dataclass
class Report:
    name: str
    desc: str = ""
    group: str = ""
    mode: str = ""
    tier: str = ""
    keys: tuple = ()
    date_field: str = ""
    note: str = ""
    registered: bool = True
    rows: int = 0
    n_files: int = 0
    n_parts: int = 0
    part_kind: str = "none"
    size: int = 0
    years: list[tuple[str, int]] = field(default_factory=list)
    dmin: str | None = None
    dmax: str | None = None
    date_src: str = ""
    date_used: str = ""                  # 实际取时间范围用的列（可能不是 spec 声明的那个）
    dup_rows: int | None = None
    dup_scope: str = ""                  # 全局 / 分区内 / 未校验
    cols: list[ColInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    coverage: list = field(default_factory=list)     # manifest 声明的覆盖区间
    suspect_n: int = 0
    quick: bool = False
    scanned: bool = False
    usage: str = ""


# ---------------------------------------------------------------- 元数据层
@dataclass
class _Meta:
    """纯 parquet footer 扫描的结果 —— 零解压，大表也秒回。"""
    rows: int = 0
    uncompressed: int = 0
    years: list[tuple[str, int]] = field(default_factory=list)
    nulls: dict = field(default_factory=dict)        # 列 → 空值数
    values: dict = field(default_factory=dict)       # 列 → 非空值数
    lo: dict = field(default_factory=dict)
    hi: dict = field(default_factory=dict)
    types: dict = field(default_factory=dict)        # 列 → {arrow 类型: [出现在哪些分区]}
    schema: dict = field(default_factory=dict)       # 列 → 代表类型（逐文件汇总后定）
    first_name: str = ""                             # 第一个成功读取的分区名
    order: list = field(default_factory=list)
    no_stats: set = field(default_factory=set)
    bad_files: list = field(default_factory=list)
    drift: list = field(default_factory=list)
    rowgroups: list = field(default_factory=list)    # (文件, 行组号, 行数, 解压字节)
    tmp_files: list = field(default_factory=list)


def _scan_meta(files: list[Path]) -> _Meta:
    """只读 parquet 元数据：行数、列统计（min/max/null_count）、schema。

    列统计是 parquet 格式自带的，读取它**不需要解压数据页**，
    因此这一步在 1800 万行的表上和在 1 万行的表上一样快。
    """
    m = _Meta()
    for f in files:
        try:
            pf = pq.ParquetFile(f)
            md = pf.metadata
        except Exception as exc:                     # noqa: BLE001
            # 文件正在被替换 / 截断 / 损坏：记下来跳过，绝不让字典生成崩掉
            m.bad_files.append((str(f), f"{type(exc).__name__}: {exc}"))
            continue

        m.rows += md.num_rows
        m.years.append((_year_label(f), md.num_rows))
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            m.rowgroups.append((f, i, rg.num_rows, int(rg.total_byte_size or 0)))
            m.uncompressed += int(rg.total_byte_size or 0)
            for j in range(rg.num_columns):
                c = rg.column(j)
                col = c.path_in_schema
                if "." in col:                       # 嵌套列的子字段，本项目不该有
                    continue
                if col not in m.order:
                    m.order.append(col)
                    m.nulls[col] = 0
                    m.values[col] = 0
                m.values[col] += int(c.num_values or 0)
                st = c.statistics
                if st is None or not st.has_null_count:
                    m.no_stats.add(col)
                    continue
                m.nulls[col] += int(st.null_count or 0)
                if not st.has_min_max:
                    m.no_stats.add(col)
                    continue
                try:
                    lo, hi = _stat_val(st.min), _stat_val(st.max)
                except Exception:                     # noqa: BLE001
                    m.no_stats.add(col)
                    continue
                if col not in m.lo or (lo is not None and lo < m.lo[col]):
                    m.lo[col] = lo
                if col not in m.hi or (hi is not None and hi > m.hi[col]):
                    m.hi[col] = hi

        # schema 漂移检查：同一数据集不同分区的列集合**和列类型**不一致，
        # 是因子模块最容易踩的坑（concat 时静默出错，或某年的列全是空值）
        got = {n: str(t) for n, t in zip(pf.schema_arrow.names, pf.schema_arrow.types)}
        if not m.types:
            m.first_name = f.parent.name
        else:
            want = set(m.types)
            if set(got) != want:
                m.drift.append(f"{f.parent.name}: 列集合不同，"
                               f"多出 {sorted(set(got) - want)} / 缺少 {sorted(want - set(got))}")
        for n, t in got.items():
            m.types.setdefault(n, {}).setdefault(t, []).append(f.parent.name)

    # 定代表类型：某个分区整列为空时 pyarrow 会把该列类型推断成 null，
    # 若拿它当展示类型，字典就会说"这列是 null 类型"——那是错的，要挑有信息量的。
    for col, variants in m.types.items():
        m.schema[col] = max(variants.items(),
                            key=lambda kv: (kv[0] != "null", len(kv[1])))[0]
        real = {_type_family(t) for t in variants if t != "null"}
        detail = " / ".join(f"{t}×{len(v)}" for t, v in sorted(variants.items()))
        if len(real) > 1:
            m.drift.append(f"{col}: 各分区类型不一致（{detail}），下游 concat 可能出错")
        elif "null" in variants and len(variants) > 1:
            # 某分区整列为空 → pyarrow 推断成 null 类型；合并时该分区会全变 NaN
            where = "、".join(variants["null"][:3])
            m.drift.append(f"{col}: 在 {where} 等分区整列为空（类型被推断为 null）")

    # 残留的 .tmp：说明有一次写入被中断（不影响读取，但要能看见）
    for f in files:
        for t in f.parent.glob("*.tmp"):
            m.tmp_files.append(str(t))
    return m


def _stat_val(v):
    """parquet 统计值 → 可比较的字符串（跨行组比较必须同类型）。"""
    if v is None:
        return None
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", "replace")
        except Exception:                             # noqa: BLE001
            return v.hex()
    return str(v)


# ---------------------------------------------------------------- 采样层
def _sample_windows(m: _Meta, files: list[Path]) -> list[tuple[Path, int]]:
    """挑两个采样窗口，覆盖"老字段"和"新字段"两种形态。

    为什么是两个：
      * 最大分区的首个行组 —— 数据最多，分布最有代表性
      * 最新分区的末个行组 —— 接口后加的列只在近期数据里有值，
        只看老数据会得到一片空白（实测 stock_limit_up 的封单类字段就是这样）
    """
    per_file: dict[str, int] = {}
    for f, _i, n, _b in m.rowgroups:
        per_file[str(f)] = per_file.get(str(f), 0) + n
    if not per_file:
        return []
    biggest = Path(max(per_file, key=per_file.get))
    latest = Path(sorted(per_file, key=lambda s: (_year_label(Path(s)), s))[-1])
    out = [(biggest, 0)]
    try:
        n_rg = pq.ParquetFile(latest).metadata.num_row_groups
    except Exception:                                 # noqa: BLE001
        n_rg = 0
    if n_rg:
        win = (latest, n_rg - 1)
        if win != out[0]:
            out.append(win)
    return out


def _read_window(f: Path, rg: int, columns: list[str]) -> dict[str, list[str]]:
    """读单个行组的开头若干行；预算不够或读失败就返回空（字典少个样例值而已）。"""
    picks: dict[str, list[str]] = {}
    try:
        pf = pq.ParquetFile(f)
        md = pf.metadata
        if rg >= md.num_row_groups:
            return picks
        want = [c for c in columns if c in set(pf.schema_arrow.names)]
        if not want:
            return picks
        # 只对**请求的这几列**算解压体积，宽表里没被请求的列不计入预算
        r = md.row_group(rg)
        budget = sum(int(r.column(j).total_uncompressed_size or 0)
                     for j in range(r.num_columns)
                     if r.column(j).path_in_schema in want)
        if budget > SAMPLE_BUDGET_BYTES:
            return picks
        n = min(SAMPLE_ROWS, r.num_rows)
        batch = next(iter(pf.iter_batches(batch_size=n, columns=want)), None)
        df = batch.to_pandas() if batch is not None else None
    except Exception:                                 # noqa: BLE001
        return picks
    if df is None:
        return picks

    for c in want:
        if c not in df.columns:
            continue
        vals, seen = [], set()
        for v in df[c].tolist():
            if v is None or (isinstance(v, float) and v != v):
                continue
            t = _fmt_num(v)
            if t not in seen:
                seen.add(t)
                vals.append(t)
            if len(vals) >= 30:
                break
        picks[c] = vals
    return picks


def _read_sample(m: _Meta, files: list[Path], columns: list[str]) -> dict[str, list[str]]:
    """取样例值：两个窗口互补，先到先得（第一个窗口拿到的就不再覆盖）。"""
    picks: dict[str, list[str]] = {}
    if not files or not columns:
        return picks
    for f, rg in _sample_windows(m, files):
        got = _read_window(f, rg, columns)
        for c, vals in got.items():
            if vals and not picks.get(c):
                picks[c] = vals
    return picks


# ---------------------------------------------------------------- 主扫描
def _scan_small(root: Path, name: str) -> tuple[dict, dict, dict, dict]:
    """小表：整表读一遍，拿精确的非空率 / 唯一值数 / 样例值与低基数分布。

    复用 store.read_dataset（因子模块读数据的同一条路径），保证"字典里写的"
    和"下游读到的"是同一份东西。
    """
    df = store.read_dataset(root, name)
    null_rate, nuniq, samples, counts = {}, {}, {}, {}
    for c in df.columns:
        s = df[c]
        n = len(s)
        null_rate[c] = float(s.isna().sum()) / n if n else 0.0
        nu = int(s.nunique(dropna=True))
        nuniq[c] = nu
        vals, seen = [], set()
        for v in s.dropna().tolist():
            t = _fmt_num(v)
            if t not in seen:
                seen.add(t)
                vals.append(t)
            if len(vals) >= 30:
                break
        samples[c] = vals
        if nu <= 15 and n:                            # 低基数列直接给出取值分布
            vc = s.value_counts(dropna=True).head(15)
            counts[c] = [(_fmt_num(k), int(v)) for k, v in vc.items()]
    return null_rate, nuniq, samples, counts


def _check_keys(m: _Meta, root: Path, name: str, keys: tuple, date_field: str,
                files: list[Path], quick: bool) -> tuple[int | None, str, list[str]]:
    """主键唯一性 —— 直接对齐 store.upsert 的去重口径（keys 上 keep='last'）。

    为什么分两条路：
      * 行数不大 → 一次读全表的 keys 列，全局去重，结果精确
      * 行数很大 → 逐分区统计。若 date_field 本身在 keys 里，那么重复键必然
        落在同一个年份分区内（键相同 ⇒ 日期相同 ⇒ 年份相同），逐分区求和仍然精确；
        否则只能得到"分区内唯一"，跨分区重复查不出来，必须如实标注。
    """
    k = [c for c in keys if c in m.schema]
    if not k:
        return None, "无主键", []
    if quick:
        return None, "未校验（--quick）", []

    notes: list[str] = []
    exact = m.rows <= GLOBAL_UNIQ_ROWS
    if exact:
        try:
            df = store.read_dataset(root, name, columns=k)
            dup = int(len(df) - len(df.drop_duplicates(subset=k)))
            return dup, "全局", notes
        except Exception as exc:                      # noqa: BLE001
            notes.append(f"整表主键校验失败，退化为逐分区：{type(exc).__name__}: {exc}")
            exact = False

    per_file: dict[str, int] = {}
    for f, _i, n, _b in m.rowgroups:
        per_file[str(f)] = per_file.get(str(f), 0) + n

    dup_total = 0
    skipped = []
    for f in files:
        n = per_file.get(str(f), 0)
        if n > PART_SCAN_MAX_ROWS:
            skipped.append(f"{f.parent.name}({n:,}行)")
            continue
        try:
            # 只读主键列：即使一年上千万行，两三个字符串列也在可控范围内
            df = pd.read_parquet(f, columns=k)
            dup_total += int(len(df) - len(df.drop_duplicates(subset=k)))
        except Exception as exc:                      # noqa: BLE001
            skipped.append(f"{f.parent.name}({type(exc).__name__})")

    if skipped:
        notes.append("以下分区未参与主键校验：" + "、".join(skipped))
        scope = "分区内（部分跳过）"
    elif date_field in k:
        scope = "分区内（键含日期，等价全局）"
    else:
        scope = "分区内（跨分区重复未查）"
    return dup_total, scope, notes


# 快照类数据集没有 spec.date_field（默认值 trade_date 并不存在于表里）时，
# 按这个顺序找一个能代表"时间"的列 —— 找不到就说明它确实没有时间维度。
_DATE_CANDIDATES = ("date", "trade_date", "end_date", "ann_date", "report_date",
                    "list_date", "suspend_date", "trade_time")


def _date_range(m: _Meta, root: Path, name: str, field_: str, quick: bool):
    """时间覆盖：优先用 parquet 列统计（免费），拿不到才去读那一列。

    返回 (最早, 最晚, 取值方式, 实际用的字段)。最后一项单列出来，
    是为了让文档能说清楚"spec 声明的是 A，这张表其实用 B"。
    """
    src = "列统计"
    if not field_ or field_ not in m.schema:
        # spec 声明的日期字段不在表里（快照类常见），退而求其次挑一个能代表时间的列
        hit = next((c for c in _DATE_CANDIDATES if c in m.schema), None)
        if hit is None:
            return None, None, "无日期字段", ""
        field_ = hit
    if field_ in m.lo and field_ in m.hi:
        return m.lo[field_], m.hi[field_], src, field_
    if quick:
        return None, None, "未取（--quick）", field_
    try:
        df = store.read_dataset(root, name, columns=[field_])
        s = df[field_].dropna()
        if len(s):
            return _fmt_num(s.min()), _fmt_num(s.max()), "读列", field_
    except Exception:                                 # noqa: BLE001
        pass
    return None, None, "未取到", field_


def _scan_dataset(data_root: Path, state_root: Path, sp, name: str,
                  quick: bool) -> Report:
    rep = Report(name=name, quick=quick)
    if sp is not None:
        rep.desc, rep.group, rep.mode = sp.desc, sp.group, sp.mode
        rep.tier, rep.keys, rep.date_field, rep.note = sp.tier, tuple(sp.keys), sp.date_field, sp.note
    else:
        rep.desc, rep.registered = "（spec.py 未登记，仅按落盘文件描述）", False

    files = _files_of(data_root, name)
    rep.n_files = len(files)
    rep.size = store.dataset_size(data_root, name)
    years = store.list_partitions(data_root, name)
    rep.n_parts = len(years) if years else (1 if files else 0)
    rep.part_kind = "year" if years else ("none" if files else "空")
    rep.usage = _guess_usage(name)

    man = Manifest.load(state_root, name)
    rep.coverage = list(man.coverage)
    rep.suspect_n = len(man.suspect or {})

    if not files:
        # 未落地不是"质量问题"（spec 里 45 个数据集本来就只能陆续下），
        # 但目录存在却没有 parquet 说明是刚建目录/下载中断，值得提一句
        if (data_root / name).exists():
            rep.warnings.append("目录存在但没有 parquet 文件（下载未开始或只建了目录）")
        return rep

    m = _scan_meta(files)
    rep.rows = m.rows
    rep.years = list(m.years)
    if m.rows == 0:
        rep.warnings.append("分区文件存在但一行都没有（下载刚开始或写入异常）")
    for f, msg in m.bad_files:
        rep.warnings.append(f"文件读取失败，已跳过：{f} —— {msg}")
    # 漂移可能命中很多分区，只留前几条，避免把字典刷成一屏告警
    rep.warnings.extend(f"schema 漂移 → {d}" for d in m.drift[:5])
    if len(m.drift) > 5:
        rep.warnings.append(f"schema 漂移共 {len(m.drift)} 处，其余从略")
    if m.tmp_files:
        rep.warnings.append(f"发现 {len(m.tmp_files)} 个残留 .tmp（写入被中断的痕迹，不影响读取）")

    rep.dmin, rep.dmax, rep.date_src, rep.date_used = _date_range(
        m, data_root, name, rep.date_field, quick)

    # 主键唯一性：这是字典里最该信的一条，所以即使 --quick 也先把范围说明白
    rep.dup_rows, rep.dup_scope, knotes = _check_keys(
        m, data_root, name, rep.keys, rep.date_field, files, quick)
    rep.warnings.extend(knotes)
    if rep.dup_rows:
        rep.warnings.append(f"主键 {rep.keys} 存在 {rep.dup_rows:,} 行重复（正常应为 0）")

    if quick:
        rep.scanned = False
        return rep
    rep.scanned = True

    # ---- 字段清单：小表整表读，大表元数据 + 采样
    small = (m.rows <= SMALL_FULL_ROWS and m.uncompressed <= SMALL_FULL_BYTES)
    null_rate: dict = {}
    nuniq: dict = {}
    samples: dict = {}
    counts: dict = {}
    src = "stats"
    if small:
        try:
            null_rate, nuniq, samples, counts = _scan_small(data_root, name)
            src = "full"
            if set(null_rate) != set(m.schema):
                rep.warnings.append("整表读到的列与元数据列不一致，可能读到了写盘中的分区")
        except Exception as exc:                      # noqa: BLE001
            rep.warnings.append(f"整表读取失败，退化为元数据+采样：{type(exc).__name__}: {exc}")
            small = False
    if not small:
        src = "stats"
        picks = _read_sample(m, files, list(m.order))
        if not picks:
            rep.warnings.append("未采样（行组解压体积超预算或读取失败），样例值为空")
        samples = {k: v[:8] for k, v in picks.items()}

    for col in m.order:
        t = m.schema.get(col)
        ci = ColInfo(name=col, dtype=_dtype_name(t) if t is not None else "?")
        if col in null_rate:
            ci.null_rate = null_rate[col]
        elif col in m.nulls and col not in m.no_stats:
            tot = m.values.get(col, 0) + m.nulls.get(col, 0)
            ci.null_rate = (m.nulls[col] / tot) if tot else None
        ci.n_unique = nuniq.get(col)
        ci.lo, ci.hi = m.lo.get(col), m.hi.get(col)
        ci.samples = samples.get(col, [])
        ci.counts = counts.get(col, [])
        ci.src = src
        # 只有大表的低基数列，才允许用采样值反推"这是分类列"（小表已精确统计）
        if src != "full" and not ci.counts and ci.samples and len(ci.samples) <= 3:
            ci.counts = [(v, -1) for v in ci.samples]
        if col in m.no_stats and src != "full":
            ci.src = "stats*"
        if ci.dtype in ("string", "binary"):
            ci.fmt, ci.mixed = _detect_format(ci.samples, (ci.lo, ci.hi))
        rep.cols.append(ci)

    _audit_columns(rep)
    return rep


def _audit_columns(rep: Report) -> None:
    """字段级体检：把"下游一定会踩、但从 schema 上看不出来"的坑挑出来。

    只报三类可确定的：
      * 整列为空 —— 接口有字段但服务端从不填，写因子时会得到全 NaN
      * 常量列   —— 唯一值只有 1 个，没有任何区分度
      * 混格式   —— 同一列既是时间又是数字，直接 astype 会得到垃圾值
    """
    for c in rep.cols:
        if c.null_rate is not None and c.null_rate >= 0.9999:
            rep.warnings.append(f"字段 `{c.name}` 整列为空（接口有该列但服务端未返回过值）")
            continue
        if c.n_unique == 1:
            rep.warnings.append(f"字段 `{c.name}` 是常量列（唯一值 1 个，无区分度）")
        if c.mixed:
            rep.warnings.append(
                f"字段 `{c.name}` 同一列混用多种格式（{c.fmt}），"
                "下游按单一格式解析会得到脏值，建议先归一化")


# ---------------------------------------------------------------- 终端输出
def _range_label(r: Report) -> str:
    """时间范围显示：快照类没有日期字段时不要写成 "? → ?"，那是误导。"""
    if r.dmin or r.dmax:
        return f"{r.dmin or '?'} → {r.dmax or '?'}"
    return "—（快照，无日期字段）" if r.date_src == "无日期字段" else "—（未取到）"


def _range_label_v(row: dict) -> str:
    """同上，输入是 MD 里读回来的速览表行（重建总表时用）。"""
    if row.get("dmin") or row.get("dmax"):
        return f"{row.get('dmin') or '?'} → {row.get('dmax') or '?'}"
    return "—（快照）"


def _print_overview(reports: list[Report], quick: bool) -> None:
    print(f"\n{'数据集':<36}{'分组':<7}{'行数':>12}{'分区':>6}{'占用':>9}  "
          f"{'时间范围':<25}{'主键':<10}{'重复':>7} {'字段':>4}")
    print("-" * 124)
    for r in reports:
        rng = _range_label(r)
        dup = "—" if r.dup_rows is None else f"{r.dup_rows:,}"
        key = "+".join(r.keys) if r.keys else "无"
        ncol = len(r.cols) if r.scanned else "—"
        print(f"{r.name:<36}{r.group or '?':<7}{fmt_rows(r.rows):>12}{r.n_parts:>6}"
              f"{_fmt_size(r.size):>9}  {rng:<25}{key:<10}{dup:>7} {str(ncol):>4}")
    print("-" * 124)
    print(f"共 {len(reports)} 个数据集 · 合计 {fmt_rows(sum(r.rows for r in reports))} 行 · "
          f"{_fmt_size(sum(r.size for r in reports))}"
          + ("  [--quick：未做字段级扫描]" if quick else ""))


def _print_detail(r: Report) -> None:
    print(f"\n{'=' * 96}\n{r.name} · {r.desc}")
    print(f"{'=' * 96}")
    print(f"  分组 {r.group or '?'} · 模式 {r.mode or '?'} · tier {r.tier or '?'} · "
          f"分区 {r.part_kind}({r.n_parts}) · 文件 {r.n_files}")
    print(f"  行数 {r.rows:,} · 占用 {_fmt_size(r.size)}")
    print(f"  时间范围 {r.dmin or '?'} → {r.dmax or '?'}"
          f"（字段 {r.date_used or '无'}，来源 {r.date_src}）")
    print(f"  主键 {'+'.join(r.keys) if r.keys else '无'} · 重复行 "
          f"{'—' if r.dup_rows is None else format(r.dup_rows, ',')} · 口径 {r.dup_scope}")
    if r.years:
        dist = " · ".join(f"{y}:{n:,}" for y, n in r.years[:12])
        more = f" · …(+{len(r.years) - 12})" if len(r.years) > 12 else ""
        print(f"  按年分布 {dist}{more}")
    if r.cols:
        print(f"\n  {'字段':<26}{'类型':<12}{'非空率':>9}{'唯一值':>10}  取值范围 / 样例")
        print("  " + "-" * 106)
        for c in r.cols:
            nn = "—" if c.null_rate is None else f"{(1 - c.null_rate) * 100:.2f}%"
            nu = "—" if c.n_unique is None else f"{c.n_unique:,}"
            print(f"  {c.name:<26}{_type_cell(c):<14}{nn:>9}{nu:>10}  {_value_text(c, 60)}")
    for w in r.warnings:
        print(f"  ! {w}")


def _value_text(c: ColInfo, width: int) -> str:
    if c.kind == "cat":
        parts = [v if n < 0 else f"{v}({n:,})" for v, n in c.counts[:10]]
        return ("取值：" + " / ".join(parts))[:width]
    bits = []
    if c.lo is not None or c.hi is not None:
        bits.append(f"{c.lo} ~ {c.hi}")
    if c.samples:
        bits.append("样例 " + ", ".join(c.samples[:4]))
    return " ｜ ".join(bits)[:width] if bits else "—"


# ---------------------------------------------------------------- Markdown
def _md_escape(s: str) -> str:
    """表格里的 | 会截断单元格，反引号保持原样（代码风格更好读）。"""
    return str(s).replace("|", "\\|")


def _md_field_table(r: Report) -> str:
    out = [
        "| 字段 | 类型 | 非空率 | 唯一值 | 取值范围 / 样例 |",
        "|:--|:--|--:|--:|:--|",
    ]
    for c in r.cols:
        nn = "—" if c.null_rate is None else f"{(1 - c.null_rate) * 100:.2f}%"
        nu = "—" if c.n_unique is None else f"{c.n_unique:,}"
        if c.kind == "cat":
            parts = [f"`{_md_escape(v)}`" if n < 0 else f"`{_md_escape(v)}`({n:,})"
                     for v, n in c.counts[:12]]
            val = " / ".join(parts)
        else:
            bits = []
            if c.hi is not None:
                bits.append(f"`{_md_escape(c.lo)}` ~ `{_md_escape(c.hi)}`")
            if c.samples:
                bits.append("样例 " + "、".join(f"`{_md_escape(s)}`" for s in c.samples[:3]))
            val = " ｜ ".join(bits) if bits else "—"
        out.append(f"| `{c.name}` | {_type_cell(c)} | {nn} | {nu} | {val} |")
    return "\n".join(out)


def _type_cell(c: ColInfo) -> str:
    """类型列顺带把字符串列的实际格式标出来 —— 这是下游最需要知道的信息之一。"""
    if c.mixed:
        return f"{c.dtype} ⚠ {c.fmt}"
    return f"{c.dtype}（{c.fmt}）" if c.fmt else c.dtype


def _md_years(r: Report) -> str:
    """按年分布渲染成两列对，36 个分区也只有 18 行，不占版面。"""
    if not r.years:
        return ""
    pairs = []
    ys = list(r.years)
    for i in range(0, len(ys), 2):
        left = f"`{ys[i][0]}` {ys[i][1]:,}"
        right = f"`{ys[i + 1][0]}` {ys[i + 1][1]:,}" if i + 1 < len(ys) else ""
        pairs.append((left, right))
    out = ["| 分区 | 行数 | 分区 | 行数 |", "|:--|--:|:--|--:|"]
    for left, right in pairs:
        lk, _, lv = left.partition(" ")
        rk, _, rv = right.partition(" ") if right else ("", "", "")
        out.append(f"| {lk} | {lv or '—'} | {rk or '—'} | {rv or '—'} |")
    return "\n".join(out)


def _section(r: Report) -> str:
    L = [f"## `{r.name}` · {r.desc}", ""]
    if not r.registered:
        L.append("> ⚠ spec.py 里没有这个数据集，以下描述仅来自落盘文件本身。")
        L.append("")
    L.append(f"- **分组**：{_GROUP_CN.get(r.group, r.group or '—')}（`{r.group or '—'}`）"
             f" ｜ **模式**：`{r.mode or '—'}` ｜ **tier**：`{r.tier or '—'}`")
    L.append(f"- **行数**：{r.rows:,} ｜ **分区**：{r.n_parts} 个（{r.part_kind}）"
             f" ｜ **文件**：{r.n_files} ｜ **磁盘占用**：{_fmt_size(r.size)}")
    if r.date_used:
        extra = (f"；spec 声明的是 `{r.date_field}`，但表里没有这列"
                 if r.date_field and r.date_used != r.date_field else "")
        L.append(f"- **时间覆盖**：{r.dmin or '?'} → {r.dmax or '?'}"
                 f"（字段 `{r.date_used}`，取自{r.date_src}{extra}）")
    else:
        L.append("- **时间覆盖**：无日期字段（快照类，本表没有时间维度）")
    if r.keys:
        dup = "未校验" if r.dup_rows is None else (
            "无重复 ✓" if r.dup_rows == 0 else f"**{r.dup_rows:,} 行重复**")
        L.append(f"- **主键**：{' + '.join(f'`{k}`' for k in r.keys)} —— {dup}"
                 f"（口径：{r.dup_scope or '—'}）")
    else:
        L.append("- **主键**：未声明 —— 整表为追加型，下游自行按需去重")
    L.append(f"- **典型用途**：{r.usage}")
    if r.note:
        L.append(f"- **下载侧备注**：{_md_escape(r.note)}")
    # 复用 manifest：它记着"下载器认为已覆盖到哪"，与实测范围对照着看最有用
    if r.coverage:
        spans = sorted(r.coverage)
        L.append(f"- **下载进度**（manifest 声明）：已覆盖 {len(spans)} 段，"
                 f"{spans[0][0]} → {max(e for _s, e in spans)}"
                 + (f" ｜ 可疑区间 {r.suspect_n} 段" if r.suspect_n else ""))
    elif r.registered and r.mode != "snapshot":
        L.append("- **下载进度**：无 manifest 记录（数据集尚未跑过或正在首次下载）")
    if r.quick:
        L.append("- ⏩ 本次为 `--quick`，未做字段级扫描（字段表缺失）")

    if r.cols:
        L += ["", f"### 字段（{len(r.cols)} 列）", "", _md_field_table(r)]
        srcs = {c.src for c in r.cols}
        if "full" in srcs:
            L += ["", "> 统计口径：整表读取，非空率/唯一值数为精确值。"]
        else:
            L += ["", "> 统计口径：非空率与取值范围来自 parquet 列统计（全量精确）；"
                      "唯一值数未计算；样例值取自行数最多的分区的前 "
                      f"{SAMPLE_ROWS} 行，**仅为示例，不代表取值范围**。"]
    if r.years and len(r.years) > 1:
        L += ["", "### 按年行数分布", "", _md_years(r)]

    if r.warnings:
        L += ["", "### 数据质量提示", ""]
        L += [f"- ⚠ {_md_escape(w)}" for w in r.warnings]
    return "\n".join(L)


_HEADER = """# 数据字典 · A 股量化数据平台

> **本文件由 `scripts/catalog.py` 自动生成，请勿手工编辑**（改了下一次生成就没了）。
> 重新生成：`python scripts/catalog.py`（单个数据集用 `--dataset <名>`，大表用 `--quick`）。
>
> 服务对象：下游因子模块（`featureengineering`）。数据落在 `data/<数据集>/`，
> 按年分区为 `year=YYYY/data.parquet`，快照类为单文件 `data.parquet`。
>
> 读数据请统一走 `lingqi.store.read_dataset(root, "<数据集>")`，列类型与本文档一致。

"""


def _load_prev(md_path: Path) -> tuple[dict, dict]:
    """读回上一版 MD 的速览表数据 + 各章节正文。

    为什么要解析自己生成的文件：`--dataset X` 只重扫一个数据集，
    但输出必须仍是**完整**的字典（其它章节不能凭空消失）。与其再落一个
    sidecar 文件（多了个会不同步的东西），不如让 MD 自己携带状态：
    速览表数据藏在顶部注释里，章节正文用 `<!-- section:名 -->` 包起来。
    """
    rows: dict[str, dict] = {}
    sections: dict[str, str] = {}
    if not md_path.exists():
        return rows, sections
    try:
        txt = md_path.read_text(encoding="utf-8")
    except OSError:
        return rows, sections
    i = txt.find(META_RE)
    if i >= 0:
        j, k = txt.find("{", i), txt.find("-->", i)
        if 0 <= j < k:
            try:
                data = json.loads(txt[j:k])
                rows = data.get("datasets") or data      # 容错：早期版本没有外层包裹
            except ValueError:
                rows = {}
    for m in re.finditer(r"<!-- section:(\w+) -->\n(.*?)\n<!-- /section:\1 -->", txt, re.S):
        sections[m.group(1)] = m.group(2)
    return rows, sections


def _dump_meta(rows: dict) -> str:
    """速览数据的内嵌元信息：一个数据集一行，人扫一眼能核对，程序也容易解析。

    放在文件**末尾**：Markdown 渲染时它整个被吃掉（HTML 注释），
    但直接读原文的人会先看到表格和章节，而不是几百行 JSON。
    """
    lines = [f"  {json.dumps(n, ensure_ascii=False)}: "
             f"{json.dumps(v, ensure_ascii=False, separators=(',', ':'))}"
             for n, v in rows.items()]
    # 带 version：将来改结构时，老文件能被识别出来而不是解析出一堆垃圾
    raw = '{\n "version": 1,\n "datasets": {\n' + ",\n".join(lines) + "\n }\n}"
    return ("<!-- catalog-meta: 本块供 `--dataset` 增量更新时重建总表用，勿手工改动\n"
            + raw.replace("-->", "--\\u003e") + "\n-->")


def _row_of(r: Report) -> dict:
    return {
        "name": r.name, "desc": r.desc, "group": r.group, "mode": r.mode, "tier": r.tier,
        "rows": r.rows, "dmin": r.dmin, "dmax": r.dmax, "keys": list(r.keys),
        "usage": r.usage, "landed": bool(r.n_files),
        "at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _sort_key(name: str, row: dict):
    g = row.get("group") or ""
    return (_GROUP_ORDER.index(g) if g in _GROUP_ORDER else 99, name)


def write_markdown(md_path: Path, reports: list[Report], partial: bool) -> None:
    # 章节正文总是继承：全量扫描会把命中的章节整段覆盖，`--quick` 则要留住旧字段表。
    # 速览表数据只在 `--dataset` 增量时才继承（全量要丢掉已经不存在的数据集）。
    prev_rows, sections = _load_prev(md_path)
    rows = dict(prev_rows) if partial else {}
    for r in reports:
        prev_at = (prev_rows.get(r.name) or {}).get("at")
        rows[r.name] = _row_of(r)
        if not r.n_files:
            sections.pop(r.name, None)      # 未落地的不占正文章节，只进"待下载"表
        elif r.quick and sections.get(r.name):
            # --quick 的定位是"快速刷新行数/覆盖"，不是"重出一份字典"。
            # 字段表是这里最贵也最稳定的部分，直接留着比抹掉有用得多。
            sections[r.name] = (
                f"> ⏩ 本次以 `--quick` 运行，未重扫字段：下表来自 {prev_at or '上一次'} "
                f"的完整扫描，行数与覆盖以速览表为准。\n\n" + sections[r.name])
        else:
            sections[r.name] = _section(r)

    ordered = sorted(rows.items(), key=lambda kv: _sort_key(*kv))
    live = [(n, v) for n, v in ordered if v.get("landed")]
    todo = [(n, v) for n, v in ordered if not v.get("landed")]
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = [_HEADER]
    body.append(f"生成时间：{ts} ｜ 已落地 {len(live)} 个数据集 ｜ "
                f"合计 {sum(int(v.get('rows', 0)) for _n, v in live):,} 行 ｜ "
                f"待下载 {len(todo)} 个\n")

    ov = ["| 数据集 | 中文名 | 分组 | 行数 | 时间范围 | 主键 | 典型用途推测 |",
          "|:--|:--|:--|--:|:--|:--|:--|"]
    for n, v in live:
        key = " + ".join(f"`{k}`" for k in v.get("keys") or []) or "—"
        ov.append(f"| [`{n}`](#{n.replace('_', '-')}) | {_md_escape(v.get('desc', ''))} | "
                  f"{_GROUP_CN.get(v.get('group', ''), v.get('group') or '—')} | "
                  f"{int(v.get('rows', 0)):,} | {_range_label_v(v)} | "
                  f"{key} | {_md_escape(v.get('usage', ''))} |")
    body += ["## 数据集速览", "", "\n".join(ov), ""]

    if todo:
        td = ["| 数据集 | 中文名 | 分组 | 模式 | tier |", "|:--|:--|:--|:--|:--|"]
        for n, v in todo:
            td.append(f"| `{n}` | {_md_escape(v.get('desc', ''))} | "
                      f"{_GROUP_CN.get(v.get('group', ''), v.get('group') or '—')} | "
                      f"`{v.get('mode', '')}` | `{v.get('tier', '')}` |")
        body += ["## 尚未落地（spec 已声明，本地暂无数据）", "",
                 f"共 {len(todo)} 个。下载完成后重跑 `python scripts/catalog.py` 即可补进上文。",
                 "", "\n".join(td), ""]

    toc = [f"{i}. [`{n}`](#{n.replace('_', '-')}) · {v.get('desc', '')}"
           for i, (n, v) in enumerate(live, 1)]
    body += ["## 目录", "", "\n".join(toc), "", "---", ""]
    for n, _v in live:
        body += [f"<!-- section:{n} -->", sections.get(n, ""), f"<!-- /section:{n} -->", ""]

    body += [_dump_meta(dict(ordered)), ""]

    # 与下载器的原子写保持一致：先写 .tmp 再 replace，避免中途被读到半个文件
    tmp = md_path.with_suffix(md_path.suffix + ".tmp")
    tmp.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")
    tmp.replace(md_path)


# ---------------------------------------------------------------- CLI
def _list_datasets(data_root: Path) -> None:
    """只列数据集，不做任何扫描 —— 秒级返回，用于确认名字写对了。"""
    print(f"\n{'数据集':<36}{'分组':<7}{'模式':<11}{'tier':<8}{'落地':<6}{'描述'}")
    print("-" * 116)
    landed = 0
    seen = set()
    for sp in spec_mod.all_specs():
        seen.add(sp.name)
        files = _files_of(data_root, sp.name)
        mark = "✓" if files else "—"
        landed += 1 if files else 0
        print(f"{sp.name:<36}{sp.group:<7}{sp.mode:<11}{sp.tier:<8}{mark:<6}{sp.desc}")
    # 已落盘但 spec 里没登记的目录也要能看见（拼错名 / 新数据集）
    extra = sorted(p.name for p in data_root.iterdir()
                   if p.is_dir() and p.name not in seen) if data_root.exists() else []
    for n in extra:
        files = _files_of(data_root, n)
        print(f"{n:<36}{'?':<7}{'?':<11}{'?':<8}{'✓' if files else '—':<6}(spec 未登记)")
    print("-" * 116)
    print(f"spec 共 {len(seen)} 个数据集 · 已落地 {landed} 个"
          + (f" · 另有 {len(extra)} 个未登记目录" if extra else ""))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="catalog.py", description="生成本地数据字典（DATA_CATALOG.md）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--dataset", "-d", action="append", default=None,
                   help="只处理指定数据集，可重复；结果合并进已有 DATA_CATALOG.md")
    p.add_argument("--quick", action="store_true",
                   help="跳过字段级扫描，只出行数/时间范围/分区分布（大表友好）；"
                        "已有的字段表会原样保留，不会被抹掉")
    p.add_argument("--list", action="store_true", help="只列数据集，不扫描")
    p.add_argument("--out", default=str(MD_PATH), help=f"Markdown 输出路径（默认 {MD_PATH.name}）")
    p.add_argument("--no-md", action="store_true", help="只打终端，不写 Markdown")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = ROOT / "data"
    state_root = ROOT / "state"

    if args.list:
        _list_datasets(data_root)
        return 0

    known = {s.name: s for s in spec_mod.all_specs()}
    if args.dataset:
        names = []
        for n in args.dataset:
            if n not in known:
                files = _files_of(data_root, n)
                if not files:
                    near = difflib.get_close_matches(n, sorted(known), n=3, cutoff=0.5)
                    tip = f"，是不是想写：{', '.join(near)}" if near else ""
                    print(f"未知数据集：{n}{tip}\n用 `python scripts/catalog.py --list` 看全部名称")
                    return 1
                print(f"[提示] {n} 未在 spec.py 登记，仅按落盘文件描述")
            names.append(n)
    else:
        # 全量 = spec 里所有数据集 ∪ data/ 下所有目录，保证"字典覆盖全部落地数据"
        disk = sorted(p.name for p in data_root.iterdir() if p.is_dir()) if data_root.exists() else []
        extra = [n for n in disk if n not in known]
        if extra:
            print(f"[提示] data/ 下有 {len(extra)} 个 spec 未登记的目录：{', '.join(extra)}")
        names = [s.name for s in spec_mod.all_specs()] + extra

    t0 = time.time()
    reports: list[Report] = []
    print(f"\n扫描 {len(names)} 个数据集{'（--quick，跳过字段级统计）' if args.quick else ''} …")
    for i, n in enumerate(names, 1):
        r = _scan_dataset(data_root, state_root, known.get(n), n, args.quick)
        reports.append(r)
        print(f"  [{i}/{len(names)}] {n:<36} {fmt_rows(r.rows):>10} 行"
              f"  {len(r.cols) if r.scanned else '—':>4} 字段"
              f"  {len(r.warnings)} 告警", flush=True)

    # 排序：先按分组优先级，再按名字，保证每次生成的顺序稳定（便于 diff）
    reports.sort(key=lambda r: (_GROUP_ORDER.index(r.group) if r.group in _GROUP_ORDER else 99,
                                r.name))
    live = [r for r in reports if r.n_files]
    _print_overview(live, args.quick)
    todo = len(reports) - len(live)
    if todo:
        print(f"另有 {todo} 个数据集尚未落地（用 `--list` 查看清单）")

    # 只扫一个数据集时，终端上直接把字段表打全，省得再去翻 MD
    if args.dataset and len(reports) <= 3:
        for r in reports:
            _print_detail(r)

    bad = [r for r in live if r.warnings]
    if bad:
        print(f"\n数据质量提示（{len(bad)} 个数据集）：")
        for r in bad:
            for w in r.warnings:
                print(f"  ! {r.name}: {w[:110]}")

    if not args.no_md:
        out = Path(args.out)
        try:
            write_markdown(out, reports, partial=bool(args.dataset))
            print(f"\n已写入 {out}（用时 {time.time() - t0:.1f}s）")
        except OSError as exc:
            print(f"\n写入 {out} 失败：{exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
