"""数据集注册表 —— 本工程**自建**，不 import `datadownload/lingqi/spec.py`。

按用户要求（2026-09-14）：新工程不依赖旧爬虫框架，但**接口配置可以照抄**。
本文件里的 path / 参数名 / 主键 / 分区方式全部**逐字对齐**线上那份 spec，
改动它时必须同步核对 `datadownload/lingqi/spec.py`（`main.py doctor` 会做双向覆盖校验）。

字段说明
    mode        range / per_date / per_stock / per_entity / snapshot / dump
    keys        业务主键，去重用（新数据赢）
    date_field  增量水位依据的字段
    partition   year（年分区）/ none（单文件）
    freq        更新频率类别（对齐 datadownload/conf/frequency.yaml）
    delay_days  该表比最后交易日晚几个交易日才可得（闸门判据用）
    redundancy_days  尾部窗口回捞几个交易日（默认取全局配置）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 全局起点
GLOBAL_START = "2010-01-01"

# 沪深主板代码前缀白名单（与旧工程、模块② 三处保持一致）
MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")

# 纯日期列 —— 落盘前统一截成 YYYY-MM-DD（旧工程 engine._to_df 的同名契约）
PURE_DATE_COLS = (
    "trade_date", "end_date", "ann_date", "date", "list_date", "delist_date",
    "suspend_date", "report_date", "start_date", "finish_date", "trade_time",
)

# 每页最多 10 万行（服务端硬限制 page * page_size <= 100000）
MAX_ROWS_PER_QUERY = 100_000


@dataclass
class DS:
    """一个数据集的声明式规格。"""
    name: str
    path: str
    mode: str
    keys: tuple = ()
    date_field: str = "trade_date"
    partition: str = "year"
    method: str = "POST"
    start_param: str = "start_time"
    end_param: str = "end_time"
    date_param: str = "trade_date"
    code_param: str = "stock_code"
    page_size: int = 10000
    paginated: bool = True
    start: str = GLOBAL_START
    enabled: bool = True
    expect_rows: bool = True
    tier: str = "medium"
    chunk_days: int = 0
    batching: int = 1
    params: dict = field(default_factory=dict)
    variants: list | None = None
    array_of: str = ""
    range_params: bool = False
    entity_from: tuple | None = None
    entity_range: bool = True
    entity_codes: list | None = None
    date_step: str = "year"
    entity_prefix_filter: tuple | None = None
    sort_by: tuple = ()
    # ---- 本工程新增的调度元数据 ----
    freq: str = "daily_full"
    delay_days: int = 0
    # ★ 已知的**厂商侧数据洞**（上游撤回了这一天、且迟迟不补）。
    #   闸门对"期望日期正好落在洞里"的表**不再阻塞**（否则它会拖满 4 小时预算），
    #   但报告里会显著标注，且尾部窗口每天照常重试（上游一发布就自动补回）。
    known_holes: frozenset = field(default_factory=frozenset)
    redundancy_days: int | None = None      # None = 用全局配置
    revision_days: int | None = None        # None = 用全局配置（range/per_date 的历史回刷窗口）
    entity_window_days: int | None = None   # per_entity 尾部窗口（None = 用全局冗余）
    entity_batch: int = 1                   # per_entity 批量编组大小（>1 需服务端支持数组）
    snapshot_every_days: int = 1            # snapshot 刷新周期（天）
    # 注：曾有个 `force_refresh` 字段（"快照类每轮无条件重抓"），全工程**无人读取** ——
    #     快照的节奏实际由 `snapshot_every_days` 决定（2026-09-17 删除该字段）。
    # ★ 拉取轴：date=按时间段拿全市场（天然增量） / entity=按实体拿全历史（天然全量刷新）
    #   / both=都支持（按窗口成本选） / snapshot=无时间轴
    pull_axis: str = "date"

    # ---------------------------------------------------------------- 台账键
    # ★★ 2026-09-16 新增。**只影响单日 MD5 台账，绝不改抓取语义**（抓取仍用 `date_field`）。
    #
    # 为什么需要它：`stock_holder_number` 的 `date_field='end_date'`（统计截止日），
    # 而**同一个截止日会由不同公司在其后的几天到几个月里陆续披露**
    # （实测 `end_date=2026-09-10` 有 8 个不同 `ann_date`、`end_date=2026-06-30` 有 56 个、
    # 横跨两个月）。于是 `end_date` 桶**只增不减、永远在长**，而台账判据是
    # 「重算的 md5 ≠ 台账里的 md5 ⇒ 上游改了历史」——
    # **这张表每跑必报红，报红成了常态，真出问题时反而被淹没**。
    # ⇒ 换成 `ann_date`（公告日）：它是**封闭事件**（那天发了多少条），那天过了就定稿。
    #
    # ★★ 2026-09-17 扩展到 4 张季频财报（income / balancesheet / cashflow / financial_indicator）：
    #   它们与 `stock_holder_number` **同构** —— `end_date` 是统计截止日，各公司在其后
    #   1~4 个月里陆续披露（实测 `stock_income` 的 `end_date=2026-03-31` 桶，公告一直排到 08-06；
    #   `end_date=2026-06-30` 桶排到 09-10）。`hash-audit --deep` 里仅剩的 4 张「复算不符」
    #   就是它们，根因是**桶在长、而台账只认 end_date**；且它们的台账**永远轮不到重算**
    #   （尾部窗口是交易日，`end_date` 是季末）⇒ 报红是常态，台账对它们失效。
    ledger_field: str = ""              # "" = 用 date_field（所有其它表的现状）
    # 公告日含周末（实测 09-12 有 4 条、09-13 有 1 条），而尾部窗口默认只取**交易日**，
    # 会让周末那几格永远不被指纹化 → 台账漏天。这类表要按**日历日**取窗口。
    ledger_calendar_days: bool = False
    # 公告日当天发出的记录，厂商可能**几小时后**才挂到接口上（与 `stock_margin_detail`
    # 的 delay=1 同类："数据到达晚"）。滞后 N 天再比对，避免每天报一次假红。
    ledger_lag_days: int = 0

    def window(self, global_redundancy: int) -> int:
        return self.redundancy_days if self.redundancy_days is not None else global_redundancy

    def ledger_date_field(self) -> str:
        """台账用哪个日期列做键（默认与抓取口径一致）。"""
        return self.ledger_field or self.date_field


REGISTRY: dict[str, DS] = {}


def _reg(ds: DS) -> DS:
    REGISTRY[ds.name] = ds
    return ds


def get(name: str) -> DS:
    return REGISTRY[name]


def all_ds() -> list[DS]:
    return list(REGISTRY.values())


def enabled() -> list[DS]:
    return [d for d in REGISTRY.values() if d.enabled]


def by_mode(mode: str) -> list[DS]:
    return [d for d in REGISTRY.values() if d.enabled and d.mode == mode]


def daily_tables() -> list[DS]:
    """日频表（闸门管的那些）。"""
    return [d for d in REGISTRY.values() if d.enabled and d.freq in ("daily_full", "daily_sparse")]


# ---------------------------------------------------------------- 闸门档位
# ★★ 2026-09-15 深夜新增（用户拍板「闸门要等所有数据都达到预期状态才启动」）。
#
# 背景：改造前闸门只遍历 `daily_tables()`（20 张），其余 22 张 enabled 表完全不进闸门
#   —— 于是「跑批开始时非日频表还没发布」这件事**没人等**，只能靠事后补跑一轮
#   （2026-09-15 当晚就是 21:14 一轮 + 23:41 补一轮）。
#
# ★★★ 为什么分三档，而不是「所有表都必须更新」：
#   **第二档不能做成"阻塞"判据，否则会死锁。** 闸门在跑批**之前**运行，而服务端随时在
#   发新数据，所以跑批前「本地 < 服务端」是**常态**；唯一能追平的办法就是跑批，
#   而跑批正被闸门挡着。⇒ 对这类表，「服务端有新的」不等于「该等」。
#   实证：`index_weight`（月频）服务端自己停在 2026-07-01 已 54 天，若要求它"必须更新"，
#   闸门会每晚白等满预算再 partial 跳过。
#
#   wait  = 有干净日级日历轴、且**每天都会发当天数据** → 阻塞等「服务端在 T-delay 有数据」
#   watch = 其余（月频/季频/不定期/周月线）→ 探测并展示，**不阻塞**
#   skip  = 无日历轴的纯快照 → 不参与闸门
#
# 档位推导只依赖注册表字段（纯函数，可测）；逐表覆写放 `_GATE_TIER`。
_GATE_TIER: dict[str, str] = {
    # ★ 2026-09-19：原 `"stock_cyq_chips": "wait"` 的补丁已删 —— 它的 `freq` 已更正为
    #   `daily_full`（实测每个交易日全覆盖），`gate_tier` 自然判 wait，不再需要手工覆写。
    #   当年加它的原因（"freq=irregular 不足以表达它其实每天发 T"）已由分类本身解决。
}

# ★ 2026-09-15 深夜：`tdx_block_stocks` / `index_ths_constituent_stocks` 的
#   `date_field` 从 `'trade_date'` 改成 **`''`（无日历轴）**。
#   它们的**数据里根本没有这一列**（实测列：`block_code/block_name/block_type/stock_code`
#   与 `index_code/code/stock_code/stock_name/plate_name`），原值是从旧工程 spec 抄来的残留。
#   后果有两个：① 它们是唯二"既躲过 mode=='snapshot' 的跳过规则、又永远算不出指纹"的表；
#   ② 哪天接口真的开始返回 `trade_date`，单日 MD5 台账会**静默地只从那天开始记**。
#   现在把"无日期轴"变成**声明**：`gate_tier` 判 skip、`dayhash.trackable` 判不记账。
#   （`partition='none'` 的写入分支有 `if ds.date_field in merged.columns` 守卫，置空安全。）


def gate_tier(ds: DS) -> str:
    """闸门档位：wait（阻塞等待）/ watch（探测不阻塞）/ skip（不参与）。"""
    if ds.name in _GATE_TIER:
        return _GATE_TIER[ds.name]
    # 无日历轴：快照类。★ 必须同时看 freq / mode / entity_range 三个字段 ——
    #   只看 mode=='snapshot' 会漏掉 `tdx_block_stocks` / `index_ths_constituent_stocks`
    #   （它们 mode='per_entity'、freq='snapshot'、entity_range=False，数据里根本没有日期列），
    #   这个坑已经让改造前 `gate.py` 里那条 snapshot 分支形同虚设。
    if ds.freq == "snapshot" or ds.mode == "snapshot" or not ds.date_field:
        return "skip"
    if ds.entity_range is False:
        return "skip"
    if ds.freq in ("daily_full", "daily_sparse", "minute"):
        return "wait"
    return "watch"


def replaces_whole(ds: DS) -> bool:
    """这张表每轮是**整表替换**（而不是按 keys 合并）。

    ★★ 2026-09-15 深夜新增。为什么需要它：`store.upsert` **从不删行** ——
      对无日期轴的纯快照表（成分股、板块列表），被调出/退市的成员**永远删不掉**，
      下游会拿到"幽灵成分股"。所以这类表要走 `store.overwrite`（整表替换）。

    ⚠️ 但它有个**必须配套**的前提：替换的语义是"新数据即全部"，
       **中途分批 flush 会让后一批把前一批覆盖掉**
       （`run_per_entity` 每 12 批 flush 一次，`tdx_block_stocks` 有 618 个实体
       → 51 次 flush → 最后只剩最后 12 个实体，**灾难性的静默丢数据**）。
       所以 `strategies._Buf` 对这类表会把 flush 阈值抬到无穷，**只在收尾时写一次**。

    `basic_calendar` 是刻意的例外：它虽然也是 `freq=snapshot`，但数据**有** `date` 列、
       靠 upsert 累积 2010→T+30 的历史，换成替换式会把它打回"只有本次抓到的那段"。
    """
    return ds.freq == "snapshot" and ds.name != "basic_calendar"


def collapses_periods(ds: DS) -> bool:
    """周/月线表：同一 (代码, 周期, 周期键) 只保留 `trade_date` 最大的那一行。

    ⚠️⚠️ **当前无使用者（恒返回 False）—— 请勿当作"坏了"或"漏删"顺手修好。**
    ---------------------------------------------------------------------------
    它唯一命中的表 `stock_kline` 已于 **2026-09-16 按用户指示彻底删除**
    （既不更新也不保存，防下游误用）。`ds.freq == "weekly_monthly"` 现在对全库
    43 张表**全部为假**，所以 `strategies.py` 里那个 `if` 永远走 else 分支。
    **函数与折叠写路径都刻意保留**：折叠分支嵌在**所有表共用**的写路径上，
    为删死代码去动它的收益为零、风险非零（2026-09-16 与用户确认走此方案，
    要真删须单开一轮并自带验证）。
    将来若重新引入周/月频表，这段机制可直接复用 —— 但**先读完下面这段再决定**。

    ---------------------------------------------------------------------------
    ★ 2026-09-16 追加实测（比原先记的"未收盘 bar"严重得多，5,550 只全市场对账）：
      原结论只说"当期未收盘所以在变"。真实语义是 ——
        **接口返回的周/月线 = 「本次抓取请求范围 ∩ 该周期」内日线的聚合，
          不是日历周期的聚合。**
      证据（当天抓取范围 09-09~09-16）：
        · 周线 `09-11` == 日线[09-09+09-10+09-11] → 5,550 只 **100%** 吻合、中位相对差 **0**
        · 周线 `09-11` vs 日线[09-07~09-11 完整一周] → 仅 **0.1%** 吻合、中位相对差 **41.5%**
        · 月线 `09-16` == 日线[09-09..09-16] → **100%** 吻合（已由另一会话独立复算）
      ⇒ **抓取窗口下界一滑动，已结束周期的值也会被改写**（不只是当期在变）。
        这是 `stock_kline` 每次跑批都报 MD5 不一致的真正原因。
      ⇒ 另：它的 14 个数据列与 `stock_daily` **完全相同**（只多一个 `period` 标签），
        覆盖区间还比日线短两天 —— **零额外信息**。要周/月频特征请从 `stock_daily` 现算。

    ---------------------------------------------------------------------------
    （以下为 2026-09-15 原记录，保留作历史）
    实测问题：接口返回"本周期至今"的聚合行，`trade_date` 被盖成我们请求的 T ——
        · 周线 2026-09-14 的 `vol = 77,187,104` 恰好等于 `stock_daily` 09-14 当天的 vol
        · 周线 2026-09-15 的 `vol = 166,543,300` 恰好等于 09-14 + 09-15 两天之和
      ⇒ 每个跑批日都会新增一整套（5550 周 + 5550 月），键含 `trade_date` 所以全部留存。
    口径（用户 2026-09-15 拍板）：每期只留最近一次抓到的、最完整的那一行。
    """
    return ds.freq == "weekly_monthly"


def gate_tables(tier: str | None = None) -> list[DS]:
    """参与闸门的表（tier=None 表示全部）。"""
    out = [d for d in REGISTRY.values() if d.enabled and gate_tier(d) != "skip"]
    if tier is not None:
        out = [d for d in out if gate_tier(d) == tier]
    return out


# ================================================================ 注册表
# ★ 下面每条都是**照抄** datadownload/lingqi/spec.py 的线上配置（2026-09-14 快照）。
#   只写与默认值不同的字段，保持可读。

# ------------------------------------------------------------------ 基础/快照
_reg(DS(name='basic_calendar', path='/basic/calendar', mode='snapshot', method='GET', keys=('date',), date_field='date', partition='none', paginated=False, expect_rows=False, tier='small', range_params=True, sort_by=('date',), freq='snapshot'))
_reg(DS(name='stock_list', path='/stock/list', mode='snapshot', method='GET', keys=('stock_code',), date_field='trade_date', partition='none', paginated=False, tier='small', sort_by=('stock_code',), freq='snapshot'))
# ★★ 4 张季频财报：台账键 `end_date` → **`ann_date`**（2026-09-17，与 stock_holder_number 同一套修法）。
#   抓取口径**一字未动**（仍是 `date_field='end_date'`），只改台账统计维度 —— 见 DS.ledger_field 的说明。
_reg(DS(name='stock_financial_indicator', path='/stock/financial_indicator', mode='per_stock', keys=('stock_code', 'end_date', 'ann_date'), date_field='end_date', ledger_field='ann_date', ledger_calendar_days=True, ledger_lag_days=1, batching=100, freq='quarterly'))
_reg(DS(name='stock_income', path='/stock/income', mode='per_stock', keys=('stock_code', 'end_date', 'ann_date'), date_field='end_date', ledger_field='ann_date', ledger_calendar_days=True, ledger_lag_days=1, batching=100, freq='quarterly'))
_reg(DS(name='stock_balancesheet', path='/stock/balancesheet', mode='per_stock', keys=('stock_code', 'end_date', 'ann_date'), date_field='end_date', ledger_field='ann_date', ledger_calendar_days=True, ledger_lag_days=1, batching=100, freq='quarterly'))
_reg(DS(name='stock_cashflow', path='/stock/cashflow', mode='per_stock', keys=('stock_code', 'end_date', 'ann_date'), date_field='end_date', ledger_field='ann_date', ledger_calendar_days=True, ledger_lag_days=1, batching=100, freq='quarterly'))
_reg(DS(name='stock_finance', path='/stock/finance', mode='range', keys=('stock_code', 'trade_date'), date_field='trade_date', tier='large', chunk_days=20, freq='daily_full'))
_reg(DS(name='stock_forecast', path='/stock/forecast', mode='per_stock', keys=('stock_code', 'ann_date', 'end_date'), date_field='ann_date', start_param='start_date', end_param='finish_date', tier='small', freq='irregular'))
_reg(DS(name='stock_daily', path='/stock/daily', mode='range', keys=('trade_date', 'stock_code'), date_field='trade_date', tier='large', chunk_days=15, sort_by=('trade_date', 'stock_code'), freq='daily_full'))
# 新增隐藏接口，编号 32；全市场日频，2018 年起，原始数值不缩放。
_reg(DS(name='stock_cyq_perf', path='/stock/cyq_perf', mode='range', keys=('trade_date', 'stock_code'), date_field='trade_date', start='2018-01-01', tier='large', chunk_days=15, sort_by=('trade_date', 'stock_code'), freq='daily_full', delay_days=0))
_reg(DS(name='stock_daily_adj', path='/stock/daily_adj', mode='range', keys=('trade_date', 'stock_code'), date_field='trade_date', enabled=False, tier='large', chunk_days=15, params={'algo': 'recursive'}, sort_by=('trade_date', 'stock_code'), freq=None))
_reg(DS(name='stock_adj_factor', path='/stock/adj_factor', mode='range', keys=('trade_date', 'stock_code'), date_field='trade_date', tier='large', chunk_days=15, sort_by=('trade_date', 'stock_code'), freq='daily_full'))
_reg(DS(name='stock_adj_factor_changes', path='/stock/adj_factor/changes', mode='per_date', keys=('date', 'stock_code'), date_field='date', date_param='date', expect_rows=False, tier='small', array_of='stock_code', freq='daily_sparse'))
# ★★ stock_kline 已于 2026-09-16 按用户指示**彻底删除**（既不更新也不保存 ——
#   防下游看到它并使用）。删因是本文件 `collapses_periods()` 上方的实测结论：
#   它的周/月线实为「抓取范围 ∩ 周期」内日线的聚合，**窗口下界一滑动，
#   已结束周期的值也会被改写**；且 14 个数据列与 stock_daily 完全相同、零额外信息。
#   这也使它成为全库**唯一**曾被 `collapses_periods()` 命中的表 —— 见该函数的注释。
#   备份：everyday_tasks/state/backtest/stock_kline_full_20260916_225817/
_reg(DS(name='stock_market_distribution_history', path='/stock/market_distribution_history', mode='per_date', keys=('trade_time',), date_field='trade_time', date_param='date', paginated=False, expect_rows=False, tier='slow', freq='minute'))
_reg(DS(name='stock_history_5min', path='/stock/history', mode='per_entity', keys=('stock_code', 'trade_time'), date_field='trade_time', expect_rows=False, tier='large', chunk_days=3660, params={'level': '5min'}, entity_from=('stock_list', 'stock_code'), freq='minute'))
_reg(DS(name='stock_suspension', path='/stock/suspension', mode='per_date', method='GET', keys=('stock_code', 'suspend_date'), date_field='suspend_date', paginated=False, expect_rows=False, tier='small', freq='daily_sparse'))
_reg(DS(name='stock_st_info', path='/stock/st_info', mode='range', keys=('stock_code', 'trade_date'), date_field='trade_date', start='2016-08-01', tier='small', chunk_days=60, freq='daily_sparse', delay_days=1))
_reg(DS(name='stock_limit_up', path='/stock/limit_up', mode='range', keys=('trade_date', 'stock_code'), date_field='trade_date', start='2015-01-01', freq='daily_sparse'))
_reg(DS(name='stock_limit_list', path='/stock/limit_list', mode='range', keys=('trade_date', 'stock_code'), date_field='trade_date', start='2020-01-01', freq='daily_sparse'))
_reg(DS(name='stock_main_fund_flow', path='/stock/main_fund_flow', mode='range', keys=('stock_code', 'trade_date'), date_field='trade_date', tier='large', chunk_days=15, freq='daily_full'))
# ★★ stock_dc_block_fund_flow 已于 2026-09-19 按用户指示**彻底删除**（既不更新也不保存
#   —— 防下游看到它并使用）。删因是本文件原 `start='2026-01-01'` 声明得太晚：
#   2026-09-19 的巡检实测，厂商 `/stock/dc_block_fund_flow` **从 2025-02-26 就有数据**
#   （逐月问过 2025-03/06/09/12，厂商各约 20 天，本地 0 天），本地却只有 2026-01-05 起
#   的 174 个交易日 —— 比 2 年窗口短了 10 个月。
#   用户判定「时间覆盖太短」→ 不再保留，而不是把起点改早再回填 21 万行。
#   备份：everyday_tasks/state/backtest/stock_dc_block_fund_flow_full_20260919_111427/
# stock_ths_block_fund_flow 已于 2026-09-19 彻底删除（用户判定"时间覆盖太短"）。★ 实测记录：
#   本地只有 2025-01-02 起（1.7 年），厂商侧探针 2024-06/2023-06/2021-06/2019-06 **均为空**、
#   但 **2024-12-02 返回 387 行** ⇒ 厂商其实还有约半年（起点约 2024 年年中），
#   即"覆盖短"有一部分是配置起点（2025-01-01）卡出来的、不是厂商没有。
#   用户权衡后仍判定"多半年也还是不足 2.2 年、不值得维护" ⇒ 连数据带脚本一并清理。
#   备份：everyday_tasks/state/backtest/two_tables_removed_20260919_142653/
# ★★ 2026-09-19：`freq` 由 `irregular` 更正为 **`daily_full`**。实测证据（用户质询"频率存疑"）：
#   取老票 600000.SH，**2018 年覆盖全部 243 个交易日、2026 年覆盖全部 174 个，零缺失** ⇒
#   它就是一个"每个交易日每只股票都有"的日频表，`irregular` 是误标。
#   当年是靠 `_GATE_TIER["stock_cyq_chips"] = "wait"` 手工打补丁让它进等待档的
#   （补丁注释写着"freq=irregular 不足以表达它其实每天发 T"）—— 现在分类正确了，补丁随之删除。
#   ⚠️ 影响：它现在会进 `daily_tables()` ⇒ 跑批结论由 "16/16 张日频表" 变 **"17/17"**，
#      并且 `report.py` 的水位检查开始覆盖它（per_entity 表的水位取分区 max，不会误报）。
_reg(DS(name='stock_cyq_chips', path='/stock/cyq_chips', mode='per_entity', keys=('trade_date', 'stock_code', 'price'), date_field='trade_date', start='2018-01-01', expect_rows=False, tier='huge', chunk_days=3660, entity_from=('stock_list', 'stock_code'), entity_prefix_filter=('600', '601', '603', '605', '000', '001', '002', '003'), freq='daily_full'))
# ★★ 台账键与抓取键**故意不同**（2026-09-16 用户拍板，见 DS.ledger_field 的说明）：
#   抓取仍按 `end_date`（窗口 [T-400, T] 才能捞到迟到的公告）；
#   台账改用 `ann_date`，并把窗口切成**日历日**（公告日含周末）、比对**滞后 1 天**
#   （公告当天厂商可能几小时后才挂出，与 stock_margin_detail 的 delay=1 同类）。
#   改前实测：`end_date=2026-09-10` 每跑必长（449→628），每次都报"上游改了历史"。
_reg(DS(name='stock_holder_number', path='/stock/holder_number', mode='range', keys=('stock_code', 'end_date', 'ann_date'), date_field='end_date', ledger_field='ann_date', ledger_calendar_days=True, ledger_lag_days=1, start='2016-01-01', freq='irregular'))
_reg(DS(name='stock_pledge_stat', path='/stock/pledge_stat', mode='range', keys=('stock_code', 'end_date'), date_field='end_date', start='2014-12-01', expect_rows=False, tier='small', freq='quarterly'))
_reg(DS(name='stock_margin_detail', path='/stock/margin_detail', mode='range', keys=('stock_code', 'trade_date'), date_field='trade_date', start='2011-01-01', tier='large', chunk_days=20, freq='daily_full', delay_days=1))
# ★★ 2026-09-15 晚（实战第 1 天）主键修正：4 列 → **7 列**（用户拍板"完全按你说的做"）。
#   旧主键（trade_date, stock_code, org_name, direction）把**同名的多个机构席位合并成一条**
#   —— `机构专用` 是匿名化的多个机构席位，同名但金额不同，是**不同的记录**。
#   实测（同日逐条比对服务端）：
#     09-14：服务端 780 行 → 旧主键只存 557 行（丢 223，其中机构专用 122→43）
#     09-15：服务端 643 行 → 旧主键只存 413 行（丢 230，其中机构专用 121→40）
#   加 `reason + buy_amount + sell_amount` 后与服务端**逐行等价**（780/780、643/643），
#   且不会把"同一席位因多个上榜原因重复出现"的冗余带回来（那部分金额完全相同，仍会去重）。
#   ⚠️ 下游（模块② 因子）若算"机构净买入"，必须自己按 (日期,股票,方向) 聚合 ——
#      旧的合并结果会让它只取到 N 条机构记录里的**任意一条**，系统性偏小。
#   ⚠️ `datadownload/lingqi/spec.py` 的同名定义已同步改成 7 列（两工程口径必须一致）。
_reg(DS(name='stock_dragon_tiger', path='/stock/dragon_tiger', mode='per_date', keys=('trade_date', 'stock_code', 'org_name', 'direction', 'reason', 'buy_amount', 'sell_amount'), date_field='trade_date', date_param='date', start='2013-01-01', expect_rows=False, freq='daily_sparse'))
_reg(DS(name='stock_top_list', path='/stock/top_list', mode='per_date', keys=('trade_date', 'stock_code', 'reason'), date_field='trade_date', expect_rows=False, freq='daily_sparse'))
# ths_hot 已于 2026-09-19 彻底删除（用户判定"时间覆盖太短"）。★ 实测记录：
#   本地只有 2024-02-01 起（2.6 年）；厂商侧探针 2024-01（起点前半月）/2023-06/2022-06/
#   2020-06/2018-06 **全部为空**（阳性对照通过，空是真空）⇒ **厂商确实没有更早的数据**。
#   备份：everyday_tasks/state/backtest/two_tables_removed_20260919_142653/
_reg(DS(name='index_daily', path='/index/daily', mode='range', keys=('ts_code', 'trade_date'), date_field='trade_date', start_param='start_date', end_param='end_date', page_size=8000, chunk_days=7, freq='daily_full'))
_reg(DS(name='index_history', path='/index/history', mode='range', keys=('index_code', 'trade_time'), date_field='trade_time', code_param='index_code', enabled=False, tier='large', chunk_days=1, params={'level': '5min'}, freq=None))
_reg(DS(name='tdx_blocks', path='/tdx/blocks', mode='snapshot', method='GET', keys=('block_code',), date_field='trade_date', partition='none', tier='small', variants=[{'block_type': 0}, {'block_type': 1}, {'block_type': 2}, {'block_type': 3}], freq='snapshot'))
# tdx_block_stocks 已于 2026-09-19 彻底删除（快照表，无历史版本，回测必然前视）
_reg(DS(name='tdx_daily', path='/tdx/daily', mode='per_entity', method='GET', keys=('board_code', 'trade_date'), date_field='trade_date', start_param='start_date', end_param='end_date', code_param='board_code', expect_rows=False, chunk_days=3660, entity_from=('tdx_blocks', 'block_code'), freq='daily_full'))
_reg(DS(name='tdx_minute', path='/tdx/minute', mode='per_entity', keys=('board_code', 'trade_time'), date_field='trade_time', code_param='board_code', start='2011-12-01', expect_rows=False, tier='large', params={'level': '5min'}, entity_from=('tdx_blocks', 'block_code'), freq='minute'))
_reg(DS(name='dc_blocks', path='/dc/blocks', mode='snapshot', keys=('block_code',), date_field='trade_date', partition='none', page_size=8000, tier='small', freq='snapshot'))
_reg(DS(name='dc_daily', path='/dc/daily', mode='per_date', keys=('block_code', 'trade_date'), date_field='trade_date', page_size=8000, start='2018-01-01', expect_rows=False, freq='daily_full'))
# ★★ 2026-09-19：`index_ths_sector_categories` / `index_ths_constituent_stocks` 按用户
#   指示**彻底删除**。删因：这两张是**快照**（厂商接口的请求参数里根本没有日期字段，
#   只有"当前状态"），而模型训练是回溯的 —— 拿"今天的成分名单"解释过去必然前视，
#   用户判定无研究意义。接口随时可重抓，删掉不损失不可再生的东西。
#   备份：everyday_tasks/state/backtest/snapshot_tables_removed_20260919_113054/
#
#   ⚠️ `index_ths_sector_categories` 原是 `index_ths_daily` 的**名单来源**，
#      直接删会让后者失去抓取对象，故先把名单**固化**成静态文件（见下方 _THS_ROSTER）。
#      这不是权宜之计，顺带解决了一个真问题：名单原会漂移（实测 2026-03-02 多 763 只、
#      2026-09-14 少 291 只），每次都让 index_ths_daily 出现覆盖台阶，
#      且新增实体的更早历史**永远补不齐**。


def _load_ths_roster() -> list[str]:
    """THS 指数名单：现读 `conf/ths_index_codes.txt`（每行一个代码，`#` 开头是注释）。

    ★ 为什么固化而不是从源表取：源表是快照，会漂移（见上方说明）。
      固化后覆盖范围**固定、可复现**，名单变化变成一件**要人做**的事。

    ⚠️ 文件缺失/为空时**大声失败**，不返回空清单 —— 空清单会让 index_ths_daily
      一个指数都不抓，而报告照旧显示 ✔（静默停更，正是本工程反复防的那类坑）。
    """
    p = Path(__file__).resolve().parent.parent / "conf" / "ths_index_codes.txt"
    if not p.exists():
        raise FileNotFoundError(
            f"THS 指数名单缺失：{p}\n"
            f"它已从快照表 index_ths_sector_categories 固化为静态文件，"
            f"请从 state/backtest/snapshot_tables_removed_*/ 恢复或重新生成。")
    codes = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    if not codes:
        raise ValueError(f"THS 指数名单为空：{p}（会造成 index_ths_daily 静默停更）")
    return codes


_THS_ROSTER = _load_ths_roster()
_reg(DS(name='index_ths_daily', path='/index/ths_daily', mode='per_entity', keys=('ths_code', 'trade_date'), date_field='trade_date', code_param='ths_code', expect_rows=False, chunk_days=3660, entity_codes=_THS_ROSTER, freq='daily_full', delay_days=1))
_reg(DS(name='stock_daily_dump', path='/stock/daily_dump', mode='dump', keys=(), date_field='trade_date', start_param='date', tier='large', freq='minute'))


# ================================================================ 调度元数据覆盖
# ★ 这些是**本工程自己的**增量调度参数，不属于接口配置（旧工程里没有）。
#   单独列出来是为了让"哪些是照抄、哪些是本工程新增"一眼可辨。

# ---- 1) 尾部窗口冗余：默认全局 5 个交易日；个别表单独指定
_REDUNDANCY = {
    "stock_history_5min": 5,        # 5min：靠 daily_dump 回捞 5 天
    "tdx_minute": 5,                # 分钟：批量回捞 5 天
    # ★ 2026-09-15 从 1 放宽到 3。
    #   原注释写"窗口收窄到 1 天；靠 revision 兜底" —— 但 **revision 对 per_entity
    #   根本不生效**（`run_per_entity` 只用 `ds.window()`，不看 `ctx.revision`），
    #   所以实际退化为"只能自愈最近 2 个交易日"：删 T / T-1 能补回，删 T-2 及更早
    #   **永远补不回**（实体已 done → 不回填；窗口又不含那天）。
    #   成本说明：per_entity 的请求数 = 实体批次数（3484/100=35），**与窗口宽度无关**；
    #   加宽窗口只让每个请求多回几天数据（100 只 × 4 天 × ~99 行 ≈ 4 万行，仍远低于
    #   服务端单请求 10 万行上限）。用同样的 35 个请求换 2 倍的自愈纵深，划算。
    "stock_cyq_chips": 3,
}

# ---- 2) range/per_date 的历史回刷窗口（日历天）
#    ★ 全局 10 天对低频表语义就是错的 —— 这正是旧工程 stock_pledge_stat
#      "coverage 声明到 09-13、数据止于 04-30、永远不补"那个活口的根因。
_REVISION = {
    "stock_limit_up": 20, "stock_limit_list": 20, "stock_st_info": 20,
    "stock_pledge_stat": 400, "stock_holder_number": 400,
    # ★ 2026-09-15 新增。`stock_forecast` 走的是自己的快路径
    #   `lo = max(ds.start, ctx.rev_start())`，而全局 revision_days=0 时
    #   `rev_start()` = T - 0 天 = **T 本身** → 查询区间退化成 `[T, T]`，
    #   只抓"今天公告的预告"。实测后果：本地 ann_date 停在 2026-08-12，
    #   而 T=09-14 —— **整整一个月的预告全部丢失**，且窗口每天前移、
    #   离窗口的日期永远追不回来（该表没有 coverage 兜底，`_missing_ranges`
    #   只服务于 range 模式）。90 天回看足以覆盖漏抓与重述。
    "stock_forecast": 90,
}

# ---- 3) per_entity 批量编组（>1 时把 N 个实体合成 1 个请求）
#    ⚠️ 标 ★需实测 的，首次跑 `main.py doctor` 会真实探测并回写结论
_ENTITY_BATCH = {
    "tdx_minute": 100,              # ✅ 文档确认 board_code 支持 string[]（618 → 7）
    "index_ths_daily": 100,         # ✅ 文档 + 实测确认 ths_code 支持 string[]（1676 → 17）
    "stock_cyq_chips": 100,         # ✅ 2026-09-14 实测确认可用：单只 99 行 / 2 只 203 / 3 只 313
                                    #    （旧文档记的"传数组返回 0 行"已不成立 —— 3484 → 35 个请求）
    "tdx_daily": 1,                 # 文档标单值；改走"按 trade_date 查当日全部板块"（★需实测）
    "stock_history_5min": 1,        # 走 daily_dump 通道，不走 per_entity 逐股
}

# ---- 4) 快照刷新周期（天）
_SNAPSHOT_EVERY = {
    "basic_calendar": 1,            # 每天（且要更新到 T+30）
    "stock_list": 1,                # 每天（5901 行，1 请求）
    "tdx_blocks": 7, "dc_blocks": 7,
    # index_ths_sector_categories / tdx_block_stocks / index_ths_constituent_stocks
    # 已于 2026-09-19 彻底删除（快照表，无历史版本，回测必然前视）
}

# ---- 5) delay_days 覆盖
#    ★ 2026-09-14 21:06 实测：20 张日频表里 17 张已有 T=09-14；
#      缺的 3 张 = stock_margin_detail（用户已确认）、stock_dragon_tiger、stock_top_list（本次新发现）
_DELAY = {
    # ★★ 2026-09-15 晚（实战第 1 天）实测与决策：
    #   实测（20:12 闸门探测 + 20:20 逐日探测，两条独立证据）：T=09-15 当晚 20:20 时
    #   `/stock/margin_detail 2026-09-14`（=T-1）仍 **0 行**，本地水位停在 09-11（=T-2）。
    #   ⚠️ 我一度按"只上不下"把它改成 2，**用户 2026-09-15 20:32 拍板改回 1**：
    #      「margin_detail **一定是 1**」——这张表就是"当晚迟一点发布"的类型，
    #      正确做法是**让闸门等它**（等到了立刻抓，拿到的是最新一天），
    #      而不是把窗口上界收到 T-2 永远晚一天拿。
    #      代价：T 当晚如果它一直不发，闸门会按用户设定的规则最多等 4 小时
    #      （`conf/daily.yaml` gate.max_wait_hours），超时后走 on_timeout=partial
    #      跳过它并显著标注，下一轮尾部窗口自动补 —— 不丢数据。
    "stock_margin_detail": 1,
    # ★★ 2026-09-15 晚（实战第 1 天，**用户拍板**）：1 → **0**。
    #   用户原话：「那这种情况说明我们的 delay 记录错误了！！你应该把他改成 0，重新爬取」。
    #   取证（T=2026-09-15 的 21:0x 实测，三条独立探测一致）：
    #     · `/stock/top_list      trade_date=2026-09-15` → **有**（64 行）
    #     · `/stock/adj_factor/changes date=2026-09-15` → **有**（21 行）
    #     · `/stock/dragon_tiger  date=2026-09-15` → **有**（服务端返回非空）
    #   即 T 当天（收盘后）这三张表**当天就有数据**，声明 1 会让闸门去看 T-1、
    #   取数窗口上界也停在 T-1 → **白等一天才拿到**，是配置错误而不是保守。
    #   ⚠️ 代价（用户已知并接受）：delay=0 后闸门对它们看的是 T 本身 ——
    #      若厂商当晚迟迟不发（或像 dc_daily 那样发了又撤），闸门会一直等到
    #      4 小时预算耗尽才转 partial（与 margin_detail 同一套"等它"策略）。
    "stock_dragon_tiger": 0,
    #   同上（用户拍板 1 → 0）：T 当天服务端就有（21 行）。
    "stock_adj_factor_changes": 0,
    # ★★ 2026-09-15 晚（实战第 1 天）：1 → **0**（用户拍板"top_list 上调到 delay=0"）。
    #   实测：`/stock/top_list 2026-09-15`（=T 当天）→ 64 行，T 当天就有数据；
    #   而声明 1 会让闸门去看 T-1、窗口上界也停在 T-1 → 白等一天才拿到。
    #   改 0 后：闸门看 T、窗口上界 = T，当天就能拿到（实测 20:20 探测 T-1/T-2/T-3 也都有）。
    "stock_top_list": 0,
    # ths_hot 已于 2026-09-19 删除（原 delay=0）。它当年 delay 在 0/1 之间来回的完整取证
    #   （"发布晚 ≠ 不发布"那一课）已并入 `README.md` §7.1，此处不再留。

    # ★★ 2026-09-15 深夜（**用户拍板**）：1 → **0**，与 ths_hot 同一批、同一类问题。
    #   两张表同晚 7 次 monitor 采样的滞后序列（新→旧）：
    #     · stock_st_info   0,1,1,1,0,0,0
    #     · index_ths_daily 0,1,1,1,0,0,0
    #   18:11 / 21:01 / 21:22 滞后 1（当晚还没发布），**23:32 变 0**（服务端已有 T=09-15）。
    #   即与 ths_hot 一样是"当天有、深夜才发布"型；声明 1 让本地白晚一天
    #   （当晚实测两张表都是 服务端=09-15 / 本地=09-14 / local_lag=1）。
    #   ⚠️ 这里用 `_DELAY` 覆写而不是改 `_reg` 的内联值（`:140` / `:178` 的 delay_days=1）：
    #      与 ths_hot / dc_daily / stock_top_list 一样，把"实测拍板"的结论集中放这里，
    #      内联值按 `registry.py:338-339` 的约定保留为"首次观测到缺口时的初始值"。
    #   ⚠️ 代价与复核口径同 ths_hot（闸门会等到 4 小时预算耗尽；delay_history 无独立证据）。
    "stock_st_info": 0,
    "index_ths_daily": 0,
    # ★ 2026-09-15 补。实测证据：T=09-14 时 dc_daily 的服务端最新只到 09-11
    #   （manifest 里 8 条 `2026-09-14|empty` 的 suspect、delay_history 里
    #   `dc_daily: server_max="无"`）。声明 delay=0 会让闸门每天为它白等，
    #   而 `delay.absent_streak` 现在也会独立观测到同一件事（双保险）。
    #
    #   ★ 2026-09-15 晚复核（用户点名这张表）：「以稳定存在的为主，不要去冒险
    #     提取最新的数据」。当日再测一次，证据一致：
    #       /dc/daily trade_date=2026-09-14 → 0 行（total=0）
    #       /dc/daily trade_date=2026-09-11 → 1,031 行
    #       /dc/daily trade_date=2026-09-10 → 1,031 行
    #     也就是说 T 当天它**有时有、有时没有**（波动），T-1 才稳定。
    #     配套改动：取数窗口上界现在也按 delay 收口（`Ctx.data_hi`），
    #     所以不会再去请求"按契约还不存在"的 T —— 晚一天拿到，但拿到的一定是稳定值。
    #
    #   ★★ 2026-09-15 晚（实战第 1 天，**用户明确"先记录、不改配置"**）——
    #     这张表真正的毛病不是 delay，是**厂商会撤回已发布的那一天**：
    #       · 2026-09-14 21:06 服务端有 09-14（1,031 行）→ 21:54 变成 **0 行**（厂商撤回）
    #       · 2026-09-15 20:20 逐日探测：09-15 **有** 1,031 行、09-14 **仍无**、09-11 有
    #     ⇒ 结论：**09-14 是厂商侧的数据洞**（撤回后没补回来），不是我们没抓到，
    #       也不是 delay 配小了（把 delay 调到 2 只会让"期望日期"正好落在洞上，
    #       明天 T=09-16 时 expected=T-2=09-14 反而又白等一次）。
    #     ⇒ 处理：洞由尾部窗口每天重试（发布即自动补）；
    #       期间报告里那条"dc_daily 部分失败（6 个交易日 / 新增 0 行）"是**真信号**。
    #       ★ 后续：09-14 已在 09-15 20:45 被厂商自己补回来，当天那轮就抓回了本地
    #         （实测 1,031 行，与既有 09-11 起的序列无缝）。
    #
    #   ★★ 2026-09-15 晚（实战第 1 天，**用户拍板**）：1 → **0**。
    #   用户原话：「那这种情况说明我们的 delay 记录错误了！！你应该把他改成 0，重新爬取」。
    #   取证（T=2026-09-15 收盘后实测）：`/dc/daily trade_date=2026-09-15` → **有 1,031 行**
    #   （同一时刻 09-14 已恢复、09-11/09-10 也都在）→ **T 当天就有当天数据**，
    #   声明 1 会让取数窗口上界停在 T-1，白等一天。
    #   ⚠️ 与它的"波动前科"（09-14 晚 21:06 有 → 21:54 被撤回）并存：delay=0 之后，
    #      若厂商在当晚把 T 撤掉，闸门会等到 4 小时预算耗尽再转 partial（"等它"策略）；
    #      已在 `_KNOWN_HOLES` 保留 09-14 作安全网（数据在时该条目完全惰性）。
    "dc_daily": 0,
}
# 注意：新建标的 delay 只在**首次观测到缺口**时才是 1，天天如此才算恒定。
#      `pipeline/delay.py` 会连续观测并自动标定（只上不下），这里是初始值。

# ---- 5b) expect_rows 覆盖（★ 本工程修正：对稀疏表把"空"当异常 = 白烧请求 + 假告警）
#    `expect_rows=True` 的语义是"这个接口正常不该返回空"：空响应会被 `client.call`
#    当成"服务端偶发空响应"**自动重试 4 次**（`empty_retries`），仍为空才交给策略层
#    记 suspect + ⚠️。对**稀疏/不定期**表这是错的 —— `/stock/report_rc`（券商研报）
#    一天 0~431 篇，连着几天没有研报完全正常。
#
#    2026-09-15 实测取证（T=09-14，90 天回刷窗口按 chunk_days=30 切成 4 块）：
#      · 服务端该表最新发布日期 = 09-10（09-11、09-14 都还没发布）
#      · 最后一块退化成 `[09-14, 09-14]` 单日 → 服务端 0 行 → 触发 4 次空响应自旋
#      · 报告因此出两条**假告警**：
#          ⚠️ stock_report_rc 部分失败（未完成，下轮应自动重试）：1 个区间 / 新增 0 行
#          ⚠️ stock_report_rc 重试率 50.0% 超 2% 基线（服务端抖动？）   ← 4/8 全是自旋
#      数据一行没丢（本地 2026 分区止于 09-10，与上游一致）。改成 False 后：
#      空块走"合法为空"分支（记 coverage），不再自旋、不再记 suspect。
#   ⚠️ 2026-09-17：唯一的使用者 `stock_report_rc` 已随该数据集一并删除（用户拍板），
#      本表**当前为空**。机制保留 —— 将来再纳入"合法为空"的稀疏表时往这里加即可。
_EXPECT_ROWS: dict[str, bool] = {}

# ---- 5c) ★ 已知的**厂商侧数据洞**（用户 2026-09-15 20:35 拍板："dc_daily 识别到这个问题先记录"）
#   语义：这一天**厂商自己撤回了数据且迟迟不补**，不是我们没抓到、也不是 delay 配小了。
#   闸门的行为（`pipeline/gate.py`）：期望日期正好落在洞里的表**不再阻塞闸门**
#   ——否则它会一直拖到 4 小时预算耗尽才开跑（实测：T=09-15 当晚 dc_daily 期望 09-14，
#   而 09-14 是洞，导致整轮要等到 00:33）。
#   ⚠️ 不等于"放弃这一天"：
#     · 尾部窗口每天照常重试它（上游一旦补发，立刻抓回）；
#     · 闸门输出里标 `⏭`，报告里带 note —— **看得见**，不会被静默吞掉。
_KNOWN_HOLES = {
    # 2026-09-14 21:06 服务端曾返回 1,031 行 → 21:54 被厂商撤回为 0 行；
    # 2026-09-15 20:20 / 20:41 两次复核仍是 0 行（而 09-15 当天反而有 1,031 行）。
    "dc_daily": {"2026-09-14"},
}

for _d in REGISTRY.values():
    if _d.name in _REDUNDANCY:
        _d.redundancy_days = _REDUNDANCY[_d.name]
    if _d.name in _KNOWN_HOLES:
        _d.known_holes = frozenset(_KNOWN_HOLES[_d.name])
    if _d.name in _REVISION:
        _d.revision_days = _REVISION[_d.name]
    if _d.name in _ENTITY_BATCH:
        _d.entity_batch = _ENTITY_BATCH[_d.name]
    if _d.name in _SNAPSHOT_EVERY:
        _d.snapshot_every_days = _SNAPSHOT_EVERY[_d.name]
    if _d.name in _DELAY:
        _d.delay_days = _DELAY[_d.name]
    if _d.name in _EXPECT_ROWS:
        _d.expect_rows = _EXPECT_ROWS[_d.name]

# ★ stock_daily_dump 是 **daily_dump 通道**，本身不是一个独立数据集：
#   它的数据要经 dump_bridge 落进 stock_history_5min（同一张表、同一套主键）。
#   所以这里禁用它的独立落盘路径，避免它写一个只有 8 列、没有 trade_date 的怪表。
if "stock_daily_dump" in REGISTRY:
    REGISTRY["stock_daily_dump"].enabled = False
# ★ stock_history_5min 改走 dump_bridge（1 个请求拿全市场，而不是逐股 5901 个）
_DUMP_BRIDGED = {"stock_history_5min"}

# ---- 7) 快照类的刷新节奏
#   由 `_SNAPSHOT_EVERY`（上面第 4 项）决定：当日/7 天/30 天。
#   ⚠️ 更早的等水位/周期都有过 bug，改这里前先读 `strategies._period_skip`
#   与 `runner._run_group` 里"⊘ 跳过不刷新 finished_at"的注释。



# ================================================================ ★ 拉取轴（pull_axis）
# 用户 2026-09-14 强调的设计原则：
#   **增量更新的逻辑与全量更新的逻辑完全不同，要按接口的天然能力选策略。**
#
#   接口只有两种"天然形状"，选错了策略要么白烧请求、要么拿不到追溯修正：
#
#   ① pull_axis = "date"   —— 接口能按**时间段**拿到**全市场**数据（不需要实体代码）
#         → 天然适合**增量**：只请求新增的那几天。成本与"天数"成正比，与"股票数"无关。
#         → 例：/stock/daily 不传 stock_code 就是全市场按日返回。
#
#   ② pull_axis = "entity" —— 接口**必须**传实体代码，且返回该实体的**全部历史**
#         → 天然适合**全量刷新**（且必须批量编组才划算）：一次拿一只股票的全部历史，
#           做增量反而要按日期切片、请求数不变却更复杂，而且**拿不到追溯修正**
#           （财报重述、复权因子调整都发生在历史期）。
#         → 例：/stock/income 传 stock_code 返回该股全部报告期。
#           100 只一批 → 5901 只只要 60 个请求，全量重拉比维护增量窗口更便宜也更正确。
#
#   ③ pull_axis = "both"   —— 两个轴向都支持
#         → 按**哪个轴向的窗口更省**来选：
#           · cyq_chips：单股单年 2.4 万行，全量要 3484 股 × 9 年 = 31356 次请求
#             → 选 date 轴（每天 1 天窗口 × 3484 股），成本与天数成正比
#           · tdx_minute / index_ths_daily：实体数少（618 / 1676）且支持数组批量
#             → 选 date 轴 + 批量编组，1676 → 17 个请求
#
#   ⚠️ 判据不是"哪个看起来快"，而是：**请求数随什么增长**。
#      随天数增长 → date 轴；随实体数增长且不批量 → 必须批量化或收紧窗口。
_PULL_AXIS = {
    # ---- ① date 轴：按时间段拿全市场 → 纯增量
    "stock_daily": "date", "stock_adj_factor": "date", "stock_finance": "date",
    "stock_main_fund_flow": "date",
    "stock_margin_detail": "date", "index_daily": "date",
    "stock_limit_up": "date", "stock_limit_list": "date",
    "stock_st_info": "date", "stock_holder_number": "date", "stock_pledge_stat": "date",
    "dc_daily": "date", "stock_dragon_tiger": "date",
    "stock_top_list": "date", "stock_suspension": "date",
    "stock_adj_factor_changes": "date", "stock_market_distribution_history": "date",
    "stock_daily_dump": "date",
    # ---- ② entity 轴：必须传代码、返回全历史 → 全量刷新（批量）
    "stock_financial_indicator": "entity", "stock_income": "entity",
    "stock_balancesheet": "entity", "stock_cashflow": "entity",
    "stock_forecast": "entity",          # ⚠️ 但支持 ann_date 区间 → 见下方特例
    # tdx_block_stocks / index_ths_constituent_stocks 已于 2026-09-19 删除
    # ---- ③ both：两个轴向都支持 → 按窗口成本选
    "stock_cyq_chips": "both", "stock_history_5min": "both",
    "tdx_minute": "both", "tdx_daily": "both", "index_ths_daily": "both",
    # ---- ④ 快照：无时间轴
    "basic_calendar": "snapshot", "stock_list": "snapshot",
    "tdx_blocks": "snapshot", "dc_blocks": "snapshot",
    # index_ths_sector_categories 已于 2026-09-19 删除（名单已固化进 conf/）
}

# ★ 特例：/stock/forecast 虽然是 entity 形状（要 stock_code），
#   但文档明说「start_date + finish_date（= ann_date 区间）可不传 stock_code」
#   → 走 date 轴只要 1~2 个请求/天，而 entity 轴要 5901 个。这是"按接口能力选策略"的典型。
_DATE_AXIS_EXCEPTIONS = {"stock_forecast"}

for _d in REGISTRY.values():
    _d.pull_axis = _PULL_AXIS.get(_d.name, "date")
for _n in _DATE_AXIS_EXCEPTIONS:
    if _n in REGISTRY:
        REGISTRY[_n].pull_axis = "date"


def by_axis(axis: str) -> list[DS]:
    return [d for d in REGISTRY.values() if d.enabled and d.pull_axis == axis]
