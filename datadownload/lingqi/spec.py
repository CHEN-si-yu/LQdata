"""数据集声明式规格 —— 一个接口一条，描述"怎么下、怎么存、怎么增量"。

mode 说明：
  snapshot      整表快照，一次拉完覆盖写
  range         按日期区间拉全市场（可分页），按年分区，支持增量
  per_date      一个交易日一次请求
  per_stock     遍历股票列表，每只拉全历史（可分批）
  per_entity    遍历实体（指数/板块代码）拉区间数据
  dump          daily_dump 特殊通道（仅最近 90 天、有配额）

keys      业务主键，合并去重用（后者覆盖前者）
date_field 增量水位依据的字段
partition  year=按年分区 / none=单文件
tier       small（秒级）/ medium / large（大表）/ slow（单次极慢）
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ★ 全局数据起点（用户 2026-09-13 拍板，见 QUANT_PLATFORM.md §10.8）：
#   所有数据集只爬 2010-01-01 以后，更早的历史不需要。
#   在这里统一兜底而不是逐个改 25 处 start= —— 政策要调整只动这一行。
#   ⚠️ 只影响「往后还爬什么」：已经落到 data/ 的更早年份分区不会自动消失，
#   要删用 scripts/drop_before.py（带 --dry-run）。
GLOBAL_START = "2010-01-01"

# ★ 沪深主板代码前缀白名单（用户 2026-09-14 拍板，见 QUANT_PLATFORM.md）。
#   沪主板 600/601/603/605（1844 只）+ 深主板 000/001/002/003（1640 只，002 为原中小板，
#   2021-04 已并入主板）= 3484 只，占 stock_list 全部 5901 只的 59%。
#   ⚠️ **口径必须与模块② featureengineering/conf/config.yaml 的
#   `universe.main_board_prefixes` 保持一致**，否则「模块① 下了什么」与
#   「模块② 认为能用什么」会错位（模块② 的硬约束是「只做主板 ~3000+ 只」）。
MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")


@dataclass
class Spec:
    name: str
    group: str
    path: str
    desc: str = ""
    method: str = "POST"
    mode: str = "range"

    # 请求参数
    params: dict = field(default_factory=dict)
    start_param: str = "start_time"
    end_param: str = "end_time"
    date_param: str = "trade_date"
    code_param: str = "stock_code"
    page_size: int = 10000
    paginated: bool = True

    # 存储
    keys: tuple = ()
    date_field: str = "trade_date"
    partition: str = "year"          # year | none
    sort_by: tuple = ()

    # 调度
    start: str = "1990-12-19"
    batching: int = 1                # per_stock 模式下一次请求带多少只股票
    tier: str = "medium"
    enabled: bool = True
    expect_rows: bool = True
    date_format: str = "%Y-%m-%d"
    note: str = ""
    # 参数变体：同一个接口用不同参数各拉一份（如 tdx_blocks 的 4 种 block_type）。
    # 变体的键会作为列写进数据，便于下游区分。
    variants: list | None = None
    # 单个任务的日期跨度上限（天）。0 = 按 tier 默认。大表必须切块控内存。
    chunk_days: int = 0
    # snapshot 模式是否需要把 [start, 今天] 作为日期参数传过去（如交易日历）
    range_params: bool = False
    # per_entity 模式的实体来源 (数据集名, 取码列)。None = 用内置规则。
    entity_from: tuple | None = None
    # per_entity 是否需要按日期区间切分；False = 每个实体只发一次请求（快照型）
    entity_range: bool = True
    # 接口返回标量数组（非对象数组）时，用这个列名把元素包装成记录
    array_of: str = ""
    # 显式指定实体清单，优先于 entity_from（用于只需少量重要指数的场景）
    entity_codes: list | None = None
    # per_entity 的时间粒度：year=按年切分；month=按月切分（配合 date_param 用 YYYY-MM）
    date_step: str = "year"
    # per_entity 的实体代码前缀白名单（如 MAIN_BOARD_PREFIXES）。None = 不过滤。
    # 在 engine._entity_codes() 里生效，因此**下载 / doctor / verify 三处口径天然一致**。
    # ⚠️ 被过滤掉的实体从不出现在 tasks 里 → 永远不会写进 done.entities →
    #    将来去掉本字段即可增量补回，无需清理已落地数据。
    entity_prefix_filter: tuple | None = None

    def __post_init__(self) -> None:
        """把 start 统一兜底到 GLOBAL_START。

        只往上抬、不往下压：start 本来就晚于 2010 的数据集（如 cyq_chips 2018、
        ths_hot 2024）保持原样，不会被拉回到 2010 白跑空区间。
        """
        if self.start and self.start < GLOBAL_START:
            self.start = GLOBAL_START


REGISTRY: dict[str, Spec] = {}


def register(spec: Spec) -> Spec:
    REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> Spec:
    return REGISTRY[name]


def all_specs() -> list[Spec]:
    return list(REGISTRY.values())


# ---------------------------------------------------------------- 基础数据
register(Spec(
    name="basic_calendar", group="basic", path="/basic/calendar", method="GET",
    desc="交易日历", mode="snapshot", partition="none", keys=("date",),
    date_field="date", start="1990-01-01", tier="small", paginated=False,
    expect_rows=False, sort_by=("date",), range_params=True,
))

# ---------------------------------------------------------------- 股票·列表
register(Spec(
    name="stock_list", group="stock", path="/stock/list", method="GET",
    desc="股票列表（含退市）", mode="snapshot", partition="none",
    keys=("stock_code",), paginated=False, tier="small", sort_by=("stock_code",),
))

# ---------------------------------------------------------------- 股票·财务
_FIN_KEYS = ("stock_code", "end_date", "ann_date")
for _n, _p, _d in (
    ("financial_indicator", "/stock/financial_indicator", "财务指标报表"),
    ("income", "/stock/income", "利润表"),
    ("balancesheet", "/stock/balancesheet", "资产负债表"),
    ("cashflow", "/stock/cashflow", "现金流量表"),
):
    register(Spec(
        name=f"stock_{_n}", group="stock", path=_p, desc=_d,
        mode="per_stock", partition="year", keys=_FIN_KEYS, date_field="end_date",
        # ★ batching=100：文档写明这四个接口的 stock_code 支持 string[]（实测有效）。
        #   5901 只股票 → 60 组 × 2 页 ≈ 120 次请求，而逐股要 5901 次（差 ~50 倍）。
        #   实测内存无压力：100 只 × 全历史 = 10000 行 × 166 列，原始 +88MB。
        #   注意 /stock/forecast 的 stock_code 是单值（文档标 string），不能批量。
        batching=100, tier="medium",
        note="按股票批量遍历（每次 100 只），返回全历史；报告期可能被追溯修正",
    ))

register(Spec(
    name="stock_finance", group="stock", path="/stock/finance",
    desc="每日财务指标（PE/PB/换手率等）", mode="range", partition="year",
    keys=("stock_code", "trade_date"), date_field="trade_date", tier="large",
))

register(Spec(
    name="stock_forecast", group="stock", path="/stock/forecast",
    desc="财报预告", mode="per_stock", partition="year",
    keys=("stock_code", "ann_date", "end_date"), date_field="ann_date",
    start_param="start_date", end_param="finish_date", batching=1, tier="small",
    start="1998-01-01",
    note="★ 本地实测最早公告 1999-01-08（原用默认 1990-12-19，白白多查 8 年空区间）",
))

# ---------------------------------------------------------------- 股票·行情
register(Spec(
    name="stock_daily", group="stock", path="/stock/daily",
    desc="日K线（不复权）", mode="range", partition="year",
    keys=("trade_date", "stock_code"), date_field="trade_date",
    sort_by=("trade_date", "stock_code"), tier="large",
    note="全市场单日约 5500 行；不传 stock_code 时按 10000/页分页",
))

register(Spec(
    name="stock_daily_adj", group="stock", path="/stock/daily_adj",
    desc="日K线（前复权 qfq）", mode="range", partition="year",
    keys=("trade_date", "stock_code"), date_field="trade_date",
    sort_by=("trade_date", "stock_code"), tier="large",
    params={"algo": "recursive"},
    note="除权除息会改变历史值，依赖 adj_factor 变更触发回刷",
    # ★ 用户 2026-09-13 17:40 拍板：不下。
    #   理由：这是**前复权(qfq)** —— 按最新复权因子缩放整条序列，新的分红会让
    #   全部历史价格重算，同一段历史今天算和昨天算不同 → 回测前视偏差。
    #   模块② 硬约束第 4 条已在 fea/spec.py 里禁止依赖它（会直接抛错）。
    #   要后复权(hfq)用 stock_adj_factor 自己算（那个保留）。
    enabled=False,
))

register(Spec(
    name="stock_adj_factor", group="stock", path="/stock/adj_factor",
    desc="复权因子（涨跌幅算法）", mode="range", partition="year",
    keys=("trade_date", "stock_code"), date_field="trade_date",
    sort_by=("trade_date", "stock_code"), tier="large",
))

register(Spec(
    name="stock_adj_factor_changes", group="stock", path="/stock/adj_factor/changes",
    desc="复权因子变更（除权除息日）", mode="per_date", partition="year",
    keys=("date", "stock_code"), date_field="date", date_param="date",
    start="2010-01-01", tier="small", expect_rows=False, array_of="stock_code",
    note="返回纯股票代码数组（非对象数组），array_of 负责包装；用于决定何时回刷复权数据",
))

# ★ stock_kline 已于 2026-09-16 按用户指示**彻底删除**（不更新、不保存）。
#   原因（实测，非推测）：接口返回的周/月线 = 「本次抓取请求范围 ∩ 该周期」内日线的聚合，
#   **不是日历周期的聚合**。证据：抓取范围 09-09~09-16 时，周线 `09-11` 恰好等于
#   日线[09-09+09-10+09-11]（5,550 只 100% 吻合、中位相对差 0），而对照"完整一周
#   09-07~09-11"只有 0.1% 吻合、中位相对差 41.5% —— 少了两天。
#   ⇒ 抓取窗口下界一滑动，**已结束周期的值也会被改写**（不只是当期在变）。
#   且它 14 个数据列与 stock_daily **完全相同**（只多一个 period 标签），
#   覆盖区间还比日线短 —— **零额外信息**。需要周/月频特征请从 stock_daily 现算。
#   备份：everyday_tasks/state/backtest/stock_kline_full_20260916_225817/


register(Spec(
    name="stock_market_distribution_history", group="stock",
    path="/stock/market_distribution_history",
    desc="历史市场涨跌分布（分钟级）", mode="per_date", partition="year",
    keys=("trade_time",), date_field="trade_time", date_param="date",
    start="2010-01-01", tier="slow", expect_rows=False, paginated=False,
    note="★ 单次 9~25 秒，走低并发通道",
    # ★ 用户 2026-09-13 17:40 曾拍板不下；同日稍后用户推翻该决定，重新启用。
    #   原决定理由（保留备查）：分钟级全市场涨跌分布，每个交易日一次请求、共约 4,055 次，
    #   是剩余主干里最慢的一个。后来又砍过一轮，用户最终要求恢复。
    #   ⚠️ 恢复时**不要**还原 state 备份（见下），数据已删，从零重建即可。
    #      state/stock_market_distribution_history.json.predrop.bak 里 802 个 done.dates
    #      若被还原，增量会把这些"已覆盖但没有数据"的日期全部跳过 → 静默丢数据。
    enabled=True,
))

register(Spec(
    name="stock_history_5min", group="stock", path="/stock/history",
    desc="个股历史分时（5min，未复权）", mode="per_entity", partition="year",
    keys=("stock_code", "trade_time"), date_field="trade_time",
    code_param="stock_code", entity_from=("stock_list", "stock_code"),
    params={"level": "5min"}, start="2010-01-01", tier="large",
    chunk_days=3660, expect_rows=False,
    note="★ 逐股拉：stock_code **必填且仅支持单只**（文档明确，实测传数组返回 0 行）。"
         "→ **不能用 per_stock 模式**（它不传日期区间，而本接口 start_time/end_time 必填，必然 422）；"
         "per_entity 是唯一会按任务把 start/end 传出去的模式。"
         "★ 名字带级别后缀是**必须的**：5min 与 1min 的 bar 会在 09:35/09:40… 这些时点撞主键，"
         "两个级别若共用 data/<name>/ 会被 drop_duplicates 静默丢掉一种。加 1min 时须另建 spec。"
         "★ chunk_days=3660 显式写死（= split_by_year 已保证的一块一自然年）："
         "单股单年 5min ≈ 1.2 万行（2 页）、1min ≈ 5.8 万行（6 页），都在 10 万上限内。"
         "★ 日期格式已验证：main.py doctor 传纯日期 YYYY-MM-DD 可用（返回 1392 行 = 29 交易日 × 48 根）。"
         "★ expect_rows=False：退市股/停牌期/上市前合法为空，设 True 会白烧 4 倍请求。"
         "🔴 **绝不要用 --end 限制一个较早的日期来跑它**（同 stock_cyq_chips）：done.entities 是"
         "**实体级、不分年份**，用较早的 --end 跑一轮会把股票全标成 done，"
         "之后即便不带 --end 也永远只覆盖到那个年份、后续年份成为永久空洞。"
         "调试请用沙箱（不同 root），别在生产 state 上试。"
         "⚠️ 成本（QUANT_PLATFORM.md §10.7 实测口径）：5min 每股每年约 2 次请求；"
         "5901 股 × 每年 ≈ 1.2 万任务 / 约 50 分钟 / 0.7 亿行。**起点待小样本探测后与用户确认**。",
))

register(Spec(
    name="stock_suspension", group="stock", path="/stock/suspension", method="GET",
    desc="停牌信息", mode="per_date", partition="year",
    keys=("stock_code", "suspend_date"), date_field="suspend_date",
    date_param="trade_date", start="1999-05-01", tier="small",
    paginated=False, expect_rows=False,
    note="只接受单日 trade_date；全表 56.9 万行超 10 万上限，只能按交易日遍历。无参请求返回 405",
))

register(Spec(
    name="stock_st_info", group="stock", path="/stock/st_info",
    desc="ST 信息", mode="range", partition="year",
    keys=("stock_code", "trade_date"), date_field="trade_date",
    # ★ 2026-09-13 从 2016-01-01 收到 2016-08-01（audit.py 查出来的）：
    #   本地实测最早 2016-08-09（QUANT_PLATFORM.md §8），原来那 7 个月是真空区间，
    #   每轮都被当"含交易日却返回空"记 suspect + 重试 7 次。
    start="2016-08-01", tier="small",
))

# ---------------------------------------------------------------- 独立数据
register(Spec(
    name="stock_limit_up", group="stock", path="/stock/limit_up",
    desc="涨停数据（封单/连板/原因）", mode="range", partition="year",
    keys=("trade_date", "stock_code"), date_field="trade_date",
    start="2015-01-01", tier="medium",
))

register(Spec(
    name="stock_limit_list", group="stock", path="/stock/limit_list",
    desc="涨跌停数据", mode="range", partition="year",
    keys=("trade_date", "stock_code"), date_field="trade_date",
    start="2020-01-01", tier="medium",
))

register(Spec(
    name="stock_main_fund_flow", group="stock", path="/stock/main_fund_flow",
    desc="大小单资金流向", mode="range", partition="year",
    keys=("stock_code", "trade_date"), date_field="trade_date",
    start="2010-01-01", tier="large",
))


# stock_dc_block_fund_flow 已于 2026-09-19 彻底删除
# （厂商可供数据到 2025-02-26，本地只覆盖到 2026-01-05 起，用户判定时间覆盖太短）

# stock_ths_block_fund_flow 已于 2026-09-19 彻底删除
# （本地仅 2025-01-02 起 1.7 年；厂商探针显示 2024-12-02 有数据、2024-06 为空，
#  即厂商还多约半年 —— 用户权衡后仍判定"不足 2.2 年、不值得维护"）
# 备份：everyday_tasks/state/backtest/two_tables_removed_20260919_142653/

register(Spec(
    name="stock_cyq_chips", group="stock", path="/stock/cyq_chips",
    desc="筹码峰分布", mode="per_entity", partition="year",
    keys=("trade_date", "stock_code", "price"), date_field="trade_date",
    code_param="stock_code", entity_from=("stock_list", "stock_code"),
    start="2018-01-01", tier="huge", chunk_days=3660, expect_rows=False,
    entity_prefix_filter=MAIN_BOARD_PREFIXES,
    note="★ 单股单年 2.4 万行；单日全市场 2026 年实测 67 万行（2018 年的 30 万翻了一倍多），"
         "远超 10 万上限，必须逐股拉。2018 起 = 9 年 × 5900 股 ≈ 12.7 亿行、单年分区 1.4 亿行。"
         "写盘靠 engine._flush_threshold() 的**自适应 flush**（max(30万, 分区已有行数//4)）——"
         "实测把 flush 次数从 ~467 降到 ~26、写盘从 120 小时降到 ~3 小时（QUANT_PLATFORM.md §11.2）。"
         "★ chunk_days=3660 显式写死（= split_by_year 已保证的一块一自然年，单股单年 2.4 万行 < 10 万），"
         "防将来有人改了 run_per_entity 的兜底值把单股单年切成 365 份。"
         "★ expect_rows=False：新股上市前 / 退市股后段**合法为空**，设 True 每命中一次白烧 4 次请求。"
         "⚠️ per_entity **没有回刷窗口** → 回填完成后 max_date 不再前进、每天 0 任务。"
         "🔴 **绝不要用 --end 限制一个较早的日期来跑它**：done.entities 是**实体级**（不分年份），"
         "标记后按实体跳过；用 --end 2019-12-31 跑一轮会把 5901 只全标成 done，"
         "之后即便不带 --end 也**永远只覆盖到 2019、后续年份成为永久空洞**。"
         "（本数据集已在 2026-09-13 用小样本探测过，那次用的是沙箱、未污染生产 state。）"
         "★ **2026-09-14 提速改动**：加 entity_prefix_filter=MAIN_BOARD_PREFIXES —— 只下主板，"
         "任务数 53109 → 31356（3484 只 × 9 年），抓取时间砍掉约 41%。"
         "决策依据：实测请求速率已占满 260/min 上限的 96.5%、每任务请求数已是 ceil(行数/10000) 的最优，"
         "抓取端没有余量，只能从数据范围上挤。回滚 = 删掉本行参数，非主板会被当新实体增量补回。",
    # ★ 用户 2026-09-13 17:40 曾拍板不下（约 12.7 亿行 / 13 小时，是当时最重的一个）；
    #   同日稍后用户推翻该决定，重新启用。
    #   ⚠️ **恢复前必须先把占位 state 挪走**，否则会被当"已完成"整轮跳过：
    #     mv state/stock_cyq_chips.json state/stock_cyq_chips.json.deferred.bak
    #   原因：那份 state 里的 done.entities（5901 条）是**占位标记不是真数据**，
    #   data/stock_cyq_chips/ 在此之前从未存在过。
    #   ⓘ 历史注记：原注释引用的 engine.run_per_entity_parts **并不存在**（分片通道从未实现），
    #     已删除该误导性引用。当前保护就是上面说的自适应 flush。
    enabled=True,
))

register(Spec(
    name="stock_holder_number", group="stock", path="/stock/holder_number",
    desc="股东人数", mode="range", partition="year",
    # ★ 2026-09-14 全库核查修复：原 keys=("stock_code","end_date") **漏了 ann_date**。
    #   服务端对同一报告期可能存在两次公告（如 600283.SH 2025-10-31 的股东人数
    #   在 2025-11-21 与 2026-02-09 各披露过一次），旧 keys 会让
    #   drop_duplicates 把后一条**静默删掉** —— 抽样窗口 30 天就丢了 7 行，
    #   按比例全库约损失 0.1~0.2%，且 coverage/done 都查不出来。
    #   与服务端实测口径一致后改为三列主键。
    keys=("stock_code", "end_date", "ann_date"), date_field="end_date",
    start="2016-01-01", tier="medium",
))

register(Spec(
    name="stock_pledge_stat", group="stock", path="/stock/pledge_stat",
    desc="股票质押", mode="range", partition="year",
    keys=("stock_code", "end_date"), date_field="end_date",
    start="2014-12-01", tier="small",
    note="★ 本地实测最早 2014-12-31；原写 2010 会让 2010~2014 的 55 个季度分片"
         "全部空转（每轮都重试一遍）",
    # ★ 2026-09-13 补：expect_rows=False（audit.py 查出来的）。
    #   它的 end_date 是**不规则**的（2014-12-31, 2015-04-30, 2015-07-31, 2015-12-31,
    #   2016-09-30 …），大量季度根本没发布 —— 是"合法为空"，不是服务端抖动。
    #   原来 expect_rows=True（默认）导致每轮为这些空季度白重试 7 次：
    #   实测 state 里积了 57 条 suspect、每条"重试 7 次" ≈ 每轮浪费约 400 次请求。
    #   这是 QUANT_PLATFORM.md §13.12 记录的同一个坑。
    expect_rows=False,
))

register(Spec(
    name="stock_margin_detail", group="stock", path="/stock/margin_detail",
    desc="融资融券明细", mode="range", partition="year",
    keys=("stock_code", "trade_date"), date_field="trade_date",
    start="2011-01-01", tier="large",
))

register(Spec(
    name="stock_dragon_tiger", group="stock", path="/stock/dragon_tiger",
    desc="龙虎榜机构明细", mode="per_date", partition="year",
    # ★★ 2026-09-15 主键 4 列 → 7 列（与 everyday_tasks/data_incremental/registry.py 同步）。
    #   旧 4 列把**同名的多个机构席位合并成一条**（`机构专用` 是匿名化的多个机构席位，
    #   同名不同金额 = 不同记录）：实测 09-14 服务端 780 行被去重成 557 行、机构专用 122→43；
    #   09-15 服务端 643 → 413、机构专用 121→40。加 reason+金额后与服务端逐行等价，
    #   且不会把"同一席位因多个上榜原因重复出现"的冗余带回来（那部分金额完全相同）。
    keys=("trade_date", "stock_code", "org_name", "direction",
          "reason", "buy_amount", "sell_amount"), date_field="trade_date",
    date_param="date", start="2013-01-01", tier="medium", expect_rows=False,
))

register(Spec(
    name="stock_top_list", group="stock", path="/stock/top_list",
    desc="龙虎榜每日明细", mode="per_date", partition="year",
    keys=("trade_date", "stock_code", "reason"), date_field="trade_date",
    date_param="trade_date", start="2007-01-01", tier="medium", expect_rows=False,
))


# ths_hot 已于 2026-09-19 彻底删除
# （本地仅 2024-02-01 起 2.6 年；厂商探针 2024-01/2023-06/2022-06/2020-06/2018-06 全为空
#  ⇒ 厂商确实没有更早数据。备份同上。）

# ---------------------------------------------------------------- 指数
register(Spec(
    name="index_daily", group="index", path="/index/daily",
    desc="指数历史日K", mode="range", partition="year",
    keys=("ts_code", "trade_date"), date_field="trade_date",
    start_param="start_date", end_param="end_date",
    page_size=8000, start="1990-12-19", tier="medium", chunk_days=30,
    note="★ 实测不传代码时按区间返回全指数（单日约 1175 个），比逐指数快",
))

register(Spec(
    name="index_history", group="index", path="/index/history",
    desc="指数历史分时（5min）", mode="range", partition="year",
    keys=("index_code", "trade_time"), date_field="trade_time",
    code_param="index_code", start="2006-12-01", tier="large",
    params={"level": "5min"}, chunk_days=1,
    note="不传 index_code 返回全部交易所指数（单日约 2.5 万行）；早期 2006~2008 volume 恒为 0",
    # ★ 用户 2026-09-13 拍板：不下。这是 5 分钟级「指数分时」，不是日频 ——
    #   和 index_daily（日频指数行情）不是一回事，删掉等于丢一整类数据。
    #   但模块② 的硬约束是「全部因子必须日频」+ 只做主板股票，5 分钟指数数据用不上。
    #   它也是剩余任务里最慢的一个（chunk_days=1，7238 个交易日逐个请求，
    #   且速率随数据量递增从 48/min 掉到 37/min，ETA 呈拉长趋势）。
    #   想恢复：把 enabled 改回 True，重新跑 `main.py run index_history` 约 3 小时。
    enabled=False,
))


register(Spec(
    name="tdx_blocks", group="tdx", path="/tdx/blocks", method="GET",
    desc="TDX 板块列表", mode="snapshot", partition="none",
    keys=("block_code",), paginated=True,
    variants=[{"block_type": t} for t in (0, 1, 2, 3)],
    start="1990-01-01", tier="small",
    note="block_type 必传：0行业/1风格/2概念/3指数，四种都拉，block_type 落列",
))

# tdx_block_stocks 已于 2026-09-19 彻底删除（快照表，无历史版本，回测必然前视）

register(Spec(
    name="tdx_daily", group="tdx", path="/tdx/daily", method="GET",
    desc="TDX 板块日K", mode="per_entity", partition="year",
    keys=("board_code", "trade_date"), date_field="trade_date",
    code_param="board_code", start_param="start_date", end_param="end_date",
    page_size=10000, start="1996-09-01", tier="medium",
    entity_from=("tdx_blocks", "block_code"), chunk_days=3660,
    # ★ 必须 expect_rows=False：新板块早年不存在，空区间是常态而非"服务端抖动"。
    #   默认 True 会把每个空任务重试 4 次 —— 实测本轮跑了 3 小时，
    #   其中约 2/3 是这种无谓重试（全站重试计数 48736/129393）。
    expect_rows=False,
    note="★ 传了 board_code 时服务端默认 page_size 会变成 100，必须显式传 10000",
))

register(Spec(
    name="tdx_minute", group="tdx", path="/tdx/minute",
    desc="TDX 板块分钟K线", mode="per_entity", partition="year",
    keys=("board_code", "trade_time"), date_field="trade_time",
    code_param="board_code", start="2011-12-01", tier="large",
    entity_from=("tdx_blocks", "block_code"),
    params={"level": "5min"}, expect_rows=False,   # ★ 同 tdx_daily：新板块早年为空是常态
    note="★ amount ≡ vol×100 是推导值不是真实成交额，两侧日期都必传",
))

register(Spec(
    name="dc_blocks", group="dc", path="/dc/blocks",
    desc="DC 板块列表", mode="snapshot", partition="none",
    keys=("block_code",), paginated=True, page_size=8000,
    start="1990-01-01", tier="small",
))


register(Spec(
    name="dc_daily", group="dc", path="/dc/daily",
    desc="DC 板块日K", mode="per_date", partition="year",
    keys=("block_code", "trade_date"), date_field="trade_date",
    start="2018-01-01", tier="medium", page_size=8000, expect_rows=False,
    note="按日期一次拉全板块，比逐板块快。★ 实测数据起点 2018-01-02，"
         "原先写 2010 会白跑 ~1944 个空交易日（约 8 分钟限速额度）",
))

# ★★ 2026-09-19：`index_ths_sector_categories` / `index_ths_constituent_stocks`
#   按用户指示**彻底删除**（快照表，无历史版本，回测必然前视）。
#   `index_ths_sector_categories` 原是本表 `index_ths_daily` 的实体来源，故把名单
#   **固化**到 everyday_tasks/conf/ths_index_codes.txt（1,676 只），两边共用同一份。
#   ⚠️ 跨工程读同一个文件是有意的：**名单只能有一个真源**，复制一份迟早对不上。
#   读取失败时大声报错，不返回空清单 —— 空清单会让本表一个指数都不抓且不报错。
def _ths_roster() -> list[str]:
    import pathlib
    p = (pathlib.Path(__file__).resolve().parents[2]
         / "everyday_tasks" / "conf" / "ths_index_codes.txt")
    if not p.exists():
        raise FileNotFoundError(f"THS 指数名单缺失：{p}（会让 index_ths_daily 静默停更）")
    codes = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    if not codes:
        raise ValueError(f"THS 指数名单为空：{p}")
    return codes


register(Spec(
    name="index_ths_daily", group="ths", path="/index/ths_daily",
    desc="THS 指数日线", mode="per_entity", partition="year",
    keys=("ths_code", "trade_date"), date_field="trade_date",
    code_param="ths_code", start="2007-08-01", tier="medium",
    entity_codes=_ths_roster(), chunk_days=3660,
    # ★ 同上，而且这里更极端：33520 个任务里约 90% 是空（多数 THS 指数是近年才发布的）。
    #   实测本轮 53 任务/分钟 → ETA 10 小时；关掉重试后约 2.2 小时。
    expect_rows=False,
    note="两个日期都必传；字段用 pct_change 而非 pct_chg；最早几天 pre_close=0",
))

# ---------------------------------------------------------------- 特殊通道
register(Spec(
    name="stock_daily_dump", group="dump", path="/stock/daily_dump",
    desc="全市场分时/日线打包（最近90天）", mode="dump", partition="year",
    keys=(), date_field="trade_date", start_param="date", tier="large",
    note="★ 每日期每天限 10 次，超限封禁 3 天；仅最近 90 天",
    # ★ 用户 2026-09-13 17:50 曾拍板不下；同日稍后请求恢复，重新启用。
    #   原决定理由（保留备查）：5 分钟级全市场数据，实测**单个任务要 3 分钟**
    #   （返回 26.4 万行 = 5500 股 × 48 根），64 个交易日要跑 3 小时。
    #   ⚠️ 它是**唯一**能拿到近期分钟数据的通道（daily_dump 只能取最近 90 天、不能回补历史）。
    #   ⚠️ 配额台账 state/dump_quota.json 由 DumpQuota 维护（限 10 次/日、超限封禁 3 天）。
    #      engine.run_dump 在发请求**之前**硬检查 can_download，--force 也绕不过（已核实）。
    enabled=True,
))

# ---------------------------------------------------------------- 大表分块
# ★ 服务端硬限制 page*page_size ≤ 100000：**单个任务必须取回 10 万行以内**，
#   否则请求会被拒（code=400）。下面的天数按"单日行数 × 天数 < 9 万"倒推，
#   留一成余量给新股上市带来的增长。
#     单日行数参考：daily 5500 / finance 4000 / main_fund_flow 5550 /
#     mff_overview 4340 / margin_detail 3840 / index_daily 1180
_CHUNK_OVERRIDES = {
    "stock_daily": 15,                    # 5500 × 15 ≈ 8.3 万
    "stock_daily_adj": 15,
    "stock_adj_factor": 15,               # 5475 × 15 ≈ 8.2 万（含退市股行数更多）
    "stock_main_fund_flow": 15,
    "stock_finance": 20,                  # 4000 × 20 ≈ 8.0 万
    "stock_margin_detail": 20,
    "stock_st_info": 60,                  # 单日仅少量 ST 股
    # ★ index_daily 2026-09-13 从 60 改成 7：
    #   原来注释写 "1180 × 60 ≈ 7.1 万"，但 **1180 是本地去重后的行数**，
    #   而服务端 `page*page_size ≤ 100000` 的上限卡的是**上游原始行数**。
    #   实测上游约 **9,000 行/天**（同一指数以多个代码重复返回，本地按
    #   (ts_code, trade_date) 去重后剩 ~1,175）。60 天 = 54 万行，要 54 页，
    #   第 11 页就撞上限报 400 → 整个 chunk 失败 →（旧代码）被误标为"已覆盖"。
    #   按 9,000/天算必须 ≤ 11 天，取 7 留足余量（7 × 9000 = 6.3 万）。
    #   ⚠️ 教训：**chunk_days 要按上游原始行数估，不是本地落库行数。**
    "index_daily": 7,
    "index_history": 1,                   # 全指数分钟级，按天
}
for _k, _v in _CHUNK_OVERRIDES.items():
    if _k in REGISTRY:
        REGISTRY[_k].chunk_days = _v


# 2026-09-19 新增：完整回填用 scripts/download_cyq_perf.py；日更在 everyday_tasks。
register(Spec(name="stock_cyq_perf", group="stock", path="/stock/cyq_perf", desc="筹码收益分位数和获利比例（原始单位）", mode="range", keys=("trade_date", "stock_code"), date_field="trade_date", start="2018-01-01", tier="large", chunk_days=15, sort_by=("trade_date", "stock_code")))
