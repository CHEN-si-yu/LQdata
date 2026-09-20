"""六种模式的增量策略 —— 本工程的核心。

★ 设计原则（用户 2026-09-14 强调）：
   **增量更新的逻辑和全量更新的逻辑完全不一样，要按每个接口的天然能力选策略。**
   接口只有两种"天然形状"（见 registry.pull_axis）：

   ① 能按**时间段**拿全市场（date 轴）
        → 天然适合**增量**：只请求新增的那几天。成本随"天数"增长，与"股票数"无关。

   ② 必须传**实体代码**、返回该实体**全历史**（entity 轴）
        → 天然适合**全量刷新**（且必须批量编组才划算）。做增量反而要按日期切片、
          请求数不变却更复杂，而且**拿不到追溯修正**（财报重述、复权调整都发生在历史期）。

   ③ 两者都支持的（both）→ 选**请求数增长更慢**的那个轴（见各 strategy 的注释）。

★ 另一条贯穿所有模式的铁律：**尾部窗口无条件重抓**。
   不看 coverage/done 决定尾部窗口抓不抓，永远抓。一条同时解决三个历史根因：
     · per_entity/per_stock 没有回刷窗口 → 拿不到追溯修正
     · expect_rows=False 吞掉服务端偶发空响应 → 永久空洞
     · 数据被误删后无法自愈
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date as _date

import pandas as pd

from .. import paths, registry as R
from ..core import state, store
from ..core.client import Client, ApiError, QueryLimitError, FetchRows, extract_list
from ..core.progress import fmt_rows
from . import calendar as cal_mod


# ================================================================ 上下文
@dataclass
class Ctx:
    client: Client
    cfg: dict
    cal: list[str]
    T: str
    runner: object = None
    dry_run: bool = False
    only: set[str] | None = None
    ready: set[str] | None = None      # 闸门 partial 时只跑这些
    # ★ 2026-09-19：一次性存量回填模式（`main.py run --report-backfill`）——
    #   把 4 张财报表的**全部**历史报告期都抓一遍（而不是只抓最近 3 期 + ann_date 窗口）。
    report_backfill: bool = False

    @property
    def redundancy(self) -> int:
        return int(self.cfg["download"]["redundancy_days"])

    @property
    def revision(self) -> int:
        return int(self.cfg["download"]["revision_days"])

    def tail(self, n: int | None = None) -> tuple[str, str]:
        return cal_mod.tail_window(self.cal, self.T, n if n is not None else self.redundancy)

    # ------------------------------------------------------------ delay 取数上界
    def data_hi(self, ds: R.DS) -> str:
        """该表**按声明应当已经有数据**的最新交易日 = `T - delay_days`。

        ★★ 为什么取数窗口也要按 delay 收口（2026-09-15 加，实测驱动）：
            `delay_days` 的语义是"该表比最后交易日晚 N 个交易日才可得"。那么在 T 这天
            去请求 T，就是在问一个**按契约还不存在**的东西。实测（本轮对话取证）：
              · `dc_daily`（delay=1）在 T=09-14 请求 09-14 → 服务端 0 行
                （09-11 及更早都是 1,031 行，稳定）；
              · 于是 `run_per_date` 走了"该表每天都该有数据、这天却空"的分支 →
                记 suspect、状态 ⚠️、报告出"部分失败（未完成，下轮应自动重试）"告警；
              · 而数据一行都没丢 —— **纯假告警 + 白烧请求**，还会顺着
                `report._judge` 的"本地水位 < 应到日期"再报一次。
            用户 2026-09-15 的原则：「不要去冒险提取最新的数据，以稳健为主」。
            所以取数窗口统一收敛到 `[T-delay-n, T-delay]`：数据晚一天到，
            但不再问不存在的日期、不再产生假告警、不再烧配额。
        """
        d = int(getattr(ds, "delay_days", 0) or 0)
        if d <= 0:
            return self.T
        return state.shift_back(self.cal, self.T, d) or self.T

    def data_window(self, ds: R.DS, n: int | None = None) -> tuple[str, str]:
        """取数窗口：以 `data_hi(ds)` 为**上界**、往前数 n 个交易日。

        下界也一起锚到上界（而不是锚在 T）——否则 `delay > 窗口天数` 时会出现
        `lo > hi` 的空窗口（例如窗口 1 天 + delay 2 天）。锚到上界后窗口宽度不变，
        只是整体前移，`lo <= hi` 恒成立。
        """
        hi = self.data_hi(ds)
        lo = state.shift_back(self.cal, hi, n if n is not None else self.redundancy) or hi
        return lo, hi

    def rev_start(self, ds: R.DS | None = None) -> str:
        """revision 窗口的起始（**日历天**，不是交易日）—— 与尾部窗口是两回事。

        ★★ 2026-09-15 修：必须按**该表自己的** `revision_days` 算。
           旧实现固定用 `self.revision`（= 全局 `download.revision_days`，默认 0），
           而所有调用点都是先算出 `rev = ds.revision_days or ctx.revision`、
           判断 `if rev > 0:` 之后，却调用这个**不看 ds** 的函数 ——
           于是 `_REVISION` 里那 9 个逐表覆写（`stock_pledge_stat`:400、
           `stock_holder_number`:400、`stock_report_rc`:90、`stock_forecast`:90 …）
           **全部失效**：`rev_start()` 恒等于 T，`min(tail_lo, T)` 还是 tail_lo，
           回刷窗口等于没开。这正是 registry 注释里点名要治的那类"活口"
           （`stock_pledge_stat` 的 coverage 声明到很新、数据却停在旧的日期）。
        """
        rev = int(self.revision)
        if ds is not None and ds.revision_days is not None:
            rev = int(ds.revision_days)
        from datetime import datetime, timedelta
        return (datetime.fromisoformat(self.T) - timedelta(days=rev)).date().isoformat()

    def selected(self, ds: R.DS, *, ignore_ready: bool = False) -> bool:
        if self.only and ds.name not in self.only:
            return False
        # ★★ 闸门只管**日频表**，所以 `ready` 里只可能有日频表的名字
        #    （`gate._plan_targets` 遍历的就是 `R.daily_tables()`）。
        #    旧实现在这里对**所有**数据集套用同一个 `ready` 判据，于是
        #    `runner.py` 里 daily / other 两组都被过滤 —— 闸门一旦 partial 超时，
        #    22 张非日频表（5min / cyq_chips / 4 张季频财报表 / 2 张月周线 /
        #    分钟表 / 全部快照）**一张都不跑**，而日志只会打印一句
        #    "④ 其他频率 0 张"。用户以为 partial 只是"跳过没到齐的日频表"。
        #    非日频表交给 `ignore_ready=True`（只看 --only）。
        if (self.ready is not None and R.gate_tier(ds) == "wait"
                and ds.name not in self.ready):
            return False
        return True


@dataclass
class Result:
    name: str
    mode: str
    rows: int = 0
    requests: int = 0
    retries: int = 0
    # ★ `retries` 里还混着一类**自旋重试**：`expect_rows=True` 的接口返回空时，
    #   `client.call` 会自己重试（`empty_retries`，默认 4 次）。
    #   它和"网络抖动/5xx"完全不是一回事 —— 报告的重试率基线（2%）只该管后者。
    #   2026-09-15 实测：`stock_report_rc` 报"重试率 50%"，8 个请求里 4 个是
    #   空响应自旋（那天本来就没有研报），一条真实抖动都没有。
    empty_retries: int = 0
    seconds: float = 0.0
    status: str = "✔"
    note: str = ""
    ranges: list = field(default_factory=list)
    # ★★ 2026-09-15 晚（用户交办）：**服务端本次对每个交易日实际给了多少行**。
    #   用途 = 检测"历史数据消失"：某天本地有数据、服务端这次却给 0 行
    #   → 报告里打一条 ℹ️（数据是厂商撤走的，**本地那份照常保留、绝不删**）。
    #   见 `report._judge` 的 vanished 分支与 `README.md` §9 第 7 条「上游撤数」。
    server_counts: dict = field(default_factory=dict)
    # ★ 2026-09-16 新增：尾部窗口下界被锚定到本地末次数据时的实际下界（None = 没锚）。
    #   用途是让「窗口会不会滑过未抓到的期」这件事在日志和报告里可观测。
    anchored_lo: str | None = None

    def as_dict(self) -> dict:
        return {"name": self.name, "mode": self.mode, "rows": self.rows,
                "requests": self.requests, "retries": self.retries,
                "empty_retries": self.empty_retries,
                "seconds": round(self.seconds, 1), "status": self.status,
                "note": self.note}


# ================================================================ 落盘辅助
def _norm_dates(df: pd.DataFrame, ds: R.DS) -> pd.DataFrame:
    """把纯日期列统一截成 YYYY-MM-DD（旧工程 engine._to_df 的同名契约）。"""
    for c in R.PURE_DATE_COLS:
        if c in df.columns and c != "trade_time":     # trade_time 要保留到秒
            df[c] = df[c].astype(str).str[:10]
    return df


class _Buf:
    """写缓冲 —— **攒够再合并写一次**。

    为什么必须要有：parquet 是**整文件重写**。逐批/逐实体落盘会让同一个年分区
    被反复重写（O(n²)）—— 实测一跑就卡死：`stock_financial_indicator` 60 个批次
    × 17 个年分区 = 1020 次整文件重写，14 分钟都没跑完第一个数据集。
    旧工程 §11.2 记过同一件事：写盘从 3 小时涨到 120 小时。

    ★ 顺序铁律：`_save`（落数据）→ 调用方 `mark_done` → `man.save()`（落状态）。
      反过来会留下"状态说完成、数据还在内存"的永久空洞。
    """

    def __init__(self, ds: R.DS, man: state.Manifest, flush_rows: int = 800_000,
                 flush_batches: int = 12, replace_suffix: bool = False):
        self.ds, self.man = ds, man
        # ★★ 2026-09-15 深夜：**整表替换的表绝不能中途 flush**。
        #   替换的语义是"新数据即全部"，而 `run_per_entity` 每 12 批就 flush 一次
        #   （`tdx_block_stocks` 有 618 个实体 → 51 次 flush）—— 那样最后只会剩下
        #   最后 12 个实体，**静默丢掉 98% 的数据**。所以把阈值抬到无穷，
        #   让它在收尾时一次性写（这些表都在百万行以内，内存扛得住）。
        if R.replaces_whole(ds):
            flush_rows = flush_batches = 10**9
        self.flush_rows, self.flush_batches = flush_rows, flush_batches
        # ★ 只有调用方能证明「本次 flush 的 new 覆盖了整个尾随后缀」时才可置 True。
        #   默认 False —— 见 store.upsert 的 docstring（误用会静默丢数据）。
        self.replace_suffix = replace_suffix
        self._frames: list[pd.DataFrame] = []
        self._rows = 0
        self.added = 0

    def add(self, df: pd.DataFrame | None) -> None:
        if df is None or len(df) == 0:
            return
        self._frames.append(df)
        self._rows += len(df)
        if self._rows >= self.flush_rows or len(self._frames) >= self.flush_batches:
            self.flush()

    def flush(self) -> int:
        if not self._frames:
            return 0
        df = pd.concat(self._frames, ignore_index=True)
        self._frames, self._rows = [], 0
        n = _save(self.ds, self.man, df, self.replace_suffix)
        self.added += n
        self.man.save()                 # 先落数据、再落状态
        return n

    @property
    def rows(self) -> int:
        return self.added


def _period_key_of(ds: R.DS, dates: pd.Series, periods: pd.Series) -> pd.Series:
    """把 (日期, 周期) 压成"这是哪一期"—— 周线用 ISO 周、月线用 YYYY-MM、其它按日。"""
    d = dates.astype(str).str[:10]
    p = periods.astype(str).str.lower()
    dt = pd.to_datetime(d, errors="coerce")
    iso = dt.dt.isocalendar()
    wk = (iso["year"].astype("Int64").astype(str) + "-W"
          + iso["week"].astype("Int64").astype(str).str.zfill(2))
    out = d.mask(p.str.startswith("m"), d.str[:7])   # monthly* → YYYY-MM
    out = out.mask(p.str.startswith("w"), wk)        # weekly*  → ISO 周
    return out


def _collapse_periods(ds: R.DS, df: pd.DataFrame) -> pd.DataFrame:
    """周/月线去重：同一 `(代码, period, 周期)` 只保留 `trade_date` 最大的那一行。

    见 `registry.collapses_periods` 的说明（接口返回"本周期至今"、每天新增一整套）。
    ⚠️ 保留下来的是**未收盘**的那一行 —— 这是刻意的，口径由用户拍板。
    """
    if not len(df) or "period" not in df.columns or ds.date_field not in df.columns:
        return df
    code_col = next((k for k in ds.keys if k not in (ds.date_field, "period")), None)
    if code_col is None or code_col not in df.columns:
        return df
    key = _period_key_of(ds, df[ds.date_field], df["period"])
    tmp = df.assign(_d=df[ds.date_field].astype(str).str[:10], _k=key)
    tmp = tmp.sort_values("_d", kind="stable").drop_duplicates(
        subset=[code_col, "period", "_k"], keep="last")
    return tmp.drop(columns=["_d", "_k"]).reset_index(drop=True)


# 快照替换的行数守卫：新数据不足旧数据的这个比例 → 判定"抓残了"，**拒绝覆盖**。
SNAPSHOT_SHRINK_FLOOR = 0.9


class SnapshotShrinkError(RuntimeError):
    """纯快照表的新数据比旧数据少太多 —— 拒绝覆盖，避免把好数据换成残的。"""


def _replace_snapshot(ds: R.DS, man: state.Manifest, df: pd.DataFrame, path) -> int:
    """无日期轴的纯快照表：**整表替换**（而不是 upsert 合并）。

    为什么要替换：`upsert` 从不删行 → 调出的成分股 / 消失的板块永远留在本地（幽灵成员），
    下游按成员表算板块因子会把这些早就不在的股票算进去。

    为什么要有守卫：替换的语义是"新数据即全部"，一次**残缺响应会把整张表打小**
      —— 比照 `dump_bridge.cache_is_final` 的思路，新行数低于旧行数的
      `SNAPSHOT_SHRINK_FLOOR` 就**抛错拒绝写入**（异常会被 `run_one` 记成 `✘` 并进报告），
      宁可不更新，也不能把好数据换成残的；下轮尾部窗口会自动重试。
    """
    # ★★ 2026-09-17：整表替换**按主键去重**（新旧两侧都去）。
    #   旧实现只 `overwrite`，而快照路径是"新数据即全部" → 服务端返回的重复行会
    #   **原样落库**：实测 `stock_list` 落了 900 只退市股各 2 行（6801 行 / 5901 个唯一码），
    #   下游按代码 join 会翻倍。（旧工程的 upsert 会按 keys 去重，所以这是回归。）
    #   ⚠️ 守卫必须拿**同一口径**比：新数据去重后 5,903 行 vs 旧数据**没去重**的
    #   6,801 行 → 会误判成"抓残了"并拒绝更新（2026-09-17 实测踩到）。
    k = [c for c in (ds.keys or ()) if c in df.columns]
    if k and len(df) > len(df.drop_duplicates(subset=k, keep="last")):
        df = df.drop_duplicates(subset=k, keep="last")
    old_df = store.read_parquet(path) if path.exists() else None
    if old_df is not None and len(old_df) and k:
        ok = [c for c in k if c in old_df.columns]
        if ok:
            old_df = old_df.drop_duplicates(subset=ok, keep="last")
    before = len(old_df) if old_df is not None else 0
    after = len(df)
    if before and after < before * SNAPSHOT_SHRINK_FLOOR:
        man.mark_suspect(f"snapshot_shrink|{after}/{before}")
        raise SnapshotShrinkError(
            f"{ds.name}: 快照替换被拒绝 —— 新数据 {after:,} 行 < 旧数据 {before:,} 行的 "
            f"{SNAPSHOT_SHRINK_FLOOR:.0%}（疑似抓残；已保持本地原样，等下轮重试）")
    store.overwrite(path, df, sort_by=ds.sort_by or ds.keys)
    man.mark_partition(0, after, None, None)
    # ★ 行数**减少是预期行为**（成分股调出），所以净增可能为负 —— 统一返回 0，
    #   别让报告把它显示成"负的新增行数"。
    return max(0, after - before)


def _save(ds: R.DS, man: state.Manifest, df: pd.DataFrame,
          replace_suffix: bool = False) -> int:
    """按年分组落盘 + 更新 manifest 分区元数据。返回新增行数（估算）。

    ⚠️ 列序以 manifest.columns 为准（首次落库的列序是规范）——
       多出一列会让旧表被 reindex 补一列 NA，对大分区是灾难（dump 的 trade_date 坑）。
    """
    if df is None or len(df) == 0:
        return 0
    df = _norm_dates(df.copy(), ds)
    if man.columns:
        keep = [c for c in man.columns if c in df.columns]
        extra = [c for c in df.columns if c not in man.columns]
        df = df.reindex(columns=keep + extra)
    else:
        man.columns = list(df.columns)

    if ds.partition == "none":
        path = store.flat_path(ds.name)
        # ★★ 2026-09-15 深夜：**无日期轴的纯快照表改走"替换式写入"**（原先一律 upsert）。
        #   问题：`upsert` 只做"按 keys 合并、新数据赢"，**从不删行** ——
        #   于是成分股被调出、板块消失、股票退市之后，本地旧行**永久保留**（幽灵成员），
        #   下游按成员表算板块因子会带上"已经不在该板块"的股票。
        #   受影响 5 张：`stock_list` / `tdx_blocks` / `dc_blocks` / `tdx_block_stocks` /
        #   `index_ths_constituent_stocks`（判据见 `registry.replaces_whole`；
        #   注意 `dc_blocks` 声明了 `trade_date` 但数据里并没有，所以不能只看声明）。
        #   ⚠️ 配套：`_Buf` 对这类表**只在收尾 flush 一次** —— 否则分批替换会丢数据。
        if R.replaces_whole(ds):
            return _replace_snapshot(ds, man, df, path)
        before = len(store.read_parquet(path))
        merged = store.upsert(path, df, keys=ds.keys, sort_by=ds.sort_by or ds.keys,
                     date_field=ds.date_field, replace_suffix=replace_suffix)
        mn = mx = None
        if ds.date_field in merged.columns and len(merged):
            s = merged[ds.date_field].astype(str).str[:10]
            mn, mx = s.min(), s.max()
        man.mark_partition(0, len(merged), mn, mx)
        return max(0, len(merged) - before)

    added = 0
    col = ds.date_field
    if col not in df.columns:
        raise ValueError(f"{ds.name}: 数据里没有 date_field={col}，无法按年分区")
    df["_y"] = df[col].astype(str).str[:4]
    for y in sorted(df["_y"].unique()):
        if not str(y).isdigit():
            continue
        sub = df[df["_y"] == y].drop(columns=["_y"])
        path = store.partition_path(ds.name, int(y))
        before = len(store.read_parquet(path))
        if R.collapses_periods(ds):
            # ★★ 周/月线：**"合并 + 按周期折叠 + 整分区替换"**，不走 upsert。
            #   因为要去重的是"同一期的多行"，而 upsert 只按 keys 去重
            #   （keys 含 trade_date，同一期的 5 行是 5 个不同的 key）→ 它删不掉重叠行。
            #   见 `registry.collapses_periods`（接口返回"本周期至今"、每天新增一整套）。
            old = store.read_parquet(path)
            base = pd.concat([old, sub], ignore_index=True) if len(old) else sub
            merged = _collapse_periods(ds, base)
            store.overwrite(path, merged, sort_by=ds.sort_by or ds.keys)
        else:
            merged = store.upsert(path, sub, keys=ds.keys, sort_by=ds.sort_by or ds.keys,
                         date_field=ds.date_field, replace_suffix=replace_suffix)
        added += max(0, len(merged) - before)
        s = merged[col].astype(str).str[:10] if col in merged.columns else pd.Series(dtype=str)
        man.mark_partition(int(y), len(merged), s.min() if len(s) else None,
                           s.max() if len(s) else None)
    return added


# ================================================================ ① range
def run_range(ds: R.DS, man: state.Manifest, ctx: Ctx) -> Result:
    """date 轴：只请求新增的日期区间 + 无条件重抓尾部窗口。

    成本：每表 1~3 个请求（按 chunk_days 切）。与股票数无关。
    """
    res = Result(ds.name, ds.mode)
    t0 = time.monotonic()
    rev = ds.revision_days if ds.revision_days is not None else ctx.revision

    # 1) coverage 里的历史缺口（一次性回填用；日常通常为空）
    #    ★ 上界同样按 delay 收口：历史缺口的回填也不必去够"按契约还不存在"的那天
    lo, hi = ctx.data_window(ds, ds.window(ctx.redundancy))   # 上界 = T - delay
    gaps = _missing_ranges(man, ds.start, hi)

    # 2) 尾部窗口：**无条件**重抓（用户要求冗余 5 个交易日）
    # 3) 可选的额外回刷窗口（默认 0；只有低频表在 registry 里覆写成大值）
    if rev > 0:
        lo = min(lo, max(ds.start, ctx.rev_start(ds)))
    gaps.append((lo, hi))
    # Previously covered tail ranges can fail on a later refresh. Coverage alone
    # must not let these recorded failures disappear when the tail moves on.
    pending_ranges = {}
    for key in man.suspect:
        span = key.split("|")[0]
        if len(span) == 21 and span[10] == "~":
            a, b = span.split("~")
            if ds.start <= a <= b <= hi:
                pending_ranges[key] = (a, b)
                gaps.append((a, b))
    successful_ranges = []
    gaps = _merge_ranges([g for g in gaps if g[1] >= ds.start])

    res.ranges = gaps
    if ctx.dry_run:
        res.note = f"将请求 {len(gaps)} 个区间: {gaps[:4]}{' …' if len(gaps) > 4 else ''}"
        res.seconds = time.monotonic() - t0
        return res

    total_added = 0
    buf = _Buf(ds, man, flush_batches=6)
    bad_all: list[str] = []
    for (a, b) in gaps:
        # ★★ 2026-09-17：coverage **按 chunk 标**（而不是按整段 gap 标）。
        #    判据 = 「这个 chunk 的**所有 variant**都拿到了一次干净的成功响应」。
        #    三种情况都**不**标覆盖，等下轮重取：
        #      · 请求失败（502/422/超时）
        #      · `expect_rows=True` 的表却返回空（稠密表返回空 = 可疑，不是"没有"）
        #      · 分页"自然结束"但服务端报的 total 与实际行数不符（`total_mismatch`）
        #    为什么必须逐 chunk：coverage 是"这段没缺口"的**唯一真相**，
        #    `_missing_ranges` 只认它。旧实现在 chunk 循环外无条件
        #    `add_coverage(a, b)`，一个 chunk 失败整段照样算已覆盖 →
        #    **永久空洞**（旧工程 index_daily 缺 159 天的同一根因）。
        for (ca, cb) in _split_chunks(a, b, ds, ctx):
            chunk_ok = True
            # ★ 必须遍历 variants：`stock_kline` 的 `period`（weekly/monthly）是**必填**参数，
            #   漏了会直接被服务端 422 拒绝（`period: Field required`）。
            for variant in (ds.variants or [None]):
                payload = {**ds.params, **(variant or {}),
                           ds.start_param: ca, ds.end_param: cb}
                mm0 = ctx.client.stats.get("total_mismatch", 0)
                try:
                    rows = ctx.client.fetch_all(ds.path, payload, ds.page_size,
                                                method=ds.method, expect_rows=ds.expect_rows,
                                                max_rows=ctx.client.max_rows_per_query)
                except ApiError as exc:
                    # ★ 失败**绝不标 coverage**（旧工程就是在这里把失败记成"已覆盖"，
                    #   导致 index_daily 缺 159 个交易日却永远不补）
                    chunk_ok = False
                    man.mark_suspect(f"{ca}~{cb}|error")
                    bad_all.append(f"{ca}~{cb} 失败({str(exc)[:50]})")
                    res.status = "⚠️"
                    continue
                except Exception as exc:  # noqa: BLE001
                    chunk_ok = False
                    man.mark_suspect(f"{ca}~{cb}|error")
                    bad_all.append(f"{ca}~{cb} 异常({str(exc)[:50]})")
                    res.status = "⚠️"
                    continue

                if not getattr(rows, "complete", True) or ctx.client.stats.get("total_mismatch", 0) > mm0:
                    chunk_ok = False
                    man.mark_suspect(f"{ca}~{cb}|total_mismatch")
                    bad_all.append(f"{ca}~{cb} total 不符（数据已收，未标覆盖）")
                    res.status = "⚠️"

                if rows:
                    df = pd.DataFrame(rows)
                    for k, v in (variant or {}).items():
                        df[k] = v
                    buf.add(df)
                    # 记下"服务端这段里每天各给了多少行" → 报告用它检测"历史数据消失"
                    if ds.date_field in df.columns:
                        vc = df[ds.date_field].astype(str).str[:10].value_counts()
                        for _day, _n in vc.items():
                            res.server_counts[str(_day)] = res.server_counts.get(str(_day), 0) + int(_n)
                elif ds.expect_rows:
                    # 合法为空（早于数据起点/非交易日）才标覆盖；expect_rows=True 的空要记 suspect
                    chunk_ok = False
                    man.mark_suspect(f"{ca}~{cb}")
                    bad_all.append(f"{ca}~{cb} 预期非空却返回空")
                    res.status = "⚠️"
            buf.flush()
            # Only commit coverage after the parquet write has succeeded. run_one
            # catches disk failures and the runner still persists this manifest.
            if chunk_ok:
                man.add_coverage(ca, cb)
                successful_ranges.append((ca, cb))
                for key, (pa, pb) in pending_ranges.items():
                    if any(sa <= pa and pb <= sb for sa, sb in _merge_ranges(successful_ranges)):
                        man.suspect.pop(key, None)
                man.save()
        if ctx.runner:
            n_bad = sum(1 for x in bad_all if x.startswith(a))
            ctx.runner.note(f"   {ds.name:34} {a}~{b} → {fmt_rows(buf.rows)} 行"
                            + ("" if not n_bad else f"  ⚠️ {n_bad} 个分片未标覆盖"))

    buf.flush()
    total_added = buf.rows
    res.rows = total_added
    res.seconds = time.monotonic() - t0
    if bad_all:
        res.status = "⚠️"
        res.note = (f"{len(bad_all)} 个分片未完成/未标覆盖，下轮自动重取："
                    + "; ".join(bad_all[:3]) + (" …" if len(bad_all) > 3 else ""))
    elif not res.note:
        res.note = f"{len(gaps)} 个区间 / 新增 {total_added:,} 行"
    return res


# ================================================================ ② per_date
def run_per_date(ds: R.DS, man: state.Manifest, ctx: Ctx) -> Result:
    """date 轴：一个交易日一个请求。只抓尾部窗口 + revision 窗口内的交易日。

    ★ 稀疏表的空响应处理（修旧工程的"根因 B"）：
      某天返回 0 行时，若该表"有数据日占比 ≥0.8"（说明它每天都该有数据），
      就记 suspect 且 **不写 done 标记** → 下一轮的尾部窗口自动重抓。
      成本 0 请求，且不会像 expect_rows=True 那样把请求放大 4 倍。
    """
    from . import probe as probe_mod

    res = Result(ds.name, ds.mode)
    t0 = time.monotonic()
    rev = ds.revision_days if ds.revision_days is not None else ctx.revision
    # ★ 上界 = T - delay（`dc_daily` 这类表的 T 当天数据按契约还不存在，
    #   问了只会拿到 0 行 → 记 suspect + 假告警）
    lo, hi = ctx.data_window(ds, ds.window(ctx.redundancy))
    # Resume from the last successful checked date (including legitimately empty
    # sparse dates), not merely a moving tail that loses long outages.
    checked = [d for d in man.done.get("dates", {}) if ds.start <= d <= hi]
    last = max(checked, default=man.max_partition_date() or "")
    if last:
        lo = min(lo, max(ds.start, str(last)[:10]))
    if rev > 0:
        lo = min(lo, max(ds.start, ctx.rev_start(ds)))
    # ★ 尾部窗口**无条件重抓**：不看 done.dates。这是"删了能补回"和
    #   "偶发空响应自愈"的保证（旧工程信 done 标记，于是永久空洞）。
    need = cal_mod.trading_days_between(ctx.cal, lo, hi)
    # Failed dates remain retryable after they leave the ordinary tail window.
    pending = [k.split("|")[0] for k in man.suspect
               if "|" in k and k.split("|")[1] in ("error", "empty", "total_mismatch")]
    need = sorted(set(need) | {d for d in pending if ds.start <= d <= hi and d in ctx.cal})

    if ctx.dry_run:
        res.note = (f"将抓 {len(need)} 个交易日（{need[0] if need else '-'} ~ "
                    f"{need[-1] if need else '-'}） × {len(ds.variants) if ds.variants else 1} 变体")
        res.seconds = time.monotonic() - t0
        return res

    ratio = probe_mod.trailing_nonzero_ratio(ds.name, ds.date_field)
    buf = _Buf(ds, man, flush_batches=10)
    holes_hit: list[str] = []
    failed_days: list[str] = []
    for d in need:
        frames = []
        # ★ 2026-09-17：把"抓取成功"与"服务端有没有数据"分开。
        #   旧实现在**抓取抛异常**后仍可能走到下面的 `elif ratio < 0.8: mark_done`，
        #   于是稀疏表里"这一天抓失败了"被记成"这天合法为空、已完成"。
        day_ok = True
        for variant in (ds.variants or [None]):
            payload = {**ds.params, **(variant or {}), ds.date_param: d}
            try:
                if ds.paginated:
                    rows = ctx.client.fetch_all(ds.path, payload, ds.page_size,
                                                method=ds.method, expect_rows=False,
                                                max_rows=ctx.client.max_rows_per_query)
                else:
                    data = ctx.client.call(ds.path, payload, method=ds.method, expect_rows=False)
                    rows = extract_list(data)
                if not getattr(rows, "complete", True):
                    day_ok = False
                    man.mark_suspect(f"{d}|total_mismatch")
                    res.status = "⚠️"
                    failed_days.append(d)
                # ★ 标量数组响应（如 `/stock/adj_factor/changes` 直接返回股票代码数组）
                #   必须包成记录 —— 这一段**两条分支都要走**，只写在非分页分支会漏
                #   （实测报错：keys 里的 stock_code 在数据里不存在）。
                if rows and not isinstance(rows[0], dict):
                    rows = [{ds.array_of or "value": v} for v in rows]
            except Exception as exc:  # noqa: BLE001
                day_ok = False
                man.mark_suspect(f"{d}|error")
                res.status = "⚠️"
                failed_days.append(d)
                res.note = f"{d} 抓取失败: {str(exc)[:70]}"
                continue
            if rows:
                df = pd.DataFrame(rows)
                if ds.date_field not in df.columns:
                    df[ds.date_field] = d
                for k, v in (variant or {}).items():
                    df[k] = v
                frames.append(df)
                res.server_counts[d] = res.server_counts.get(d, 0) + len(df)
            elif d in (getattr(ds, "known_holes", ()) or ()):
                # ★ 已知厂商洞（`registry._KNOWN_HOLES`，用户 2026-09-15 拍板）：
                #   这天上游确实**没有**（不是偶发空响应），所以既不当异常、也不标 done
                #   —— 下一轮尾部窗口照常再试，上游一补发就自动抓回。
                #   `continue` 跳过下面的 `mark_done`，正是"每天重试"想要的效果。
                holes_hit.append(d)
                res.server_counts.setdefault(d, 0)     # 报告要能看到"这天上游是空的"
                continue
            elif ratio >= 0.8:
                # 该表每天都该有数据，这天却是空的 → 可疑，不标 done，下轮重抓
                man.mark_suspect(f"{d}|empty")
                res.status = "⚠️"
                frames = frames or []
            res.server_counts.setdefault(d, 0)   # 走到这里 = 这天服务端给的是空
        if frames:
            buf.add(pd.concat(frames, ignore_index=True))
            # ★ 先落盘再标完成
            buf.flush()
            if day_ok:
                man.mark_done("dates", d)
                for suffix in ("error", "empty", "total_mismatch"):
                    man.suspect.pop(f"{d}|{suffix}", None)
        elif day_ok and ratio < 0.8:
            man.mark_done("dates", d)     # 稀疏表：**抓取成功且**合法为空
            for suffix in ("error", "empty", "total_mismatch"):
                man.suspect.pop(f"{d}|{suffix}", None)
        if ctx.runner and frames:
            ctx.runner.note(f"   {ds.name:34} {d} → {fmt_rows(buf.rows)} 行")

    buf.flush()
    res.rows = buf.rows
    res.seconds = time.monotonic() - t0
    if failed_days:
        res.status = "⚠️"
        res.note = (f"{len(failed_days)} 天抓取失败（**未**标完成，下轮尾部窗口自动重试）："
                    f"{', '.join(failed_days[:5])}")
    elif not res.note:
        res.note = f"{len(need)} 个交易日 / 新增 {res.rows:,} 行"
        if holes_hit:
            res.note += (f"（其中 {len(holes_hit)} 天是**已知厂商洞**、已跳过："
                         f"{', '.join(holes_hit)} —— 见 registry._KNOWN_HOLES）")
    return res


# ================================================================ ③ per_entity
def _parallel_map(fn, items: list, workers: int):
    """并发跑 `fn(item) -> (rows, err)`，按完成顺序 yield `(item, rows, err)`。

    ★ 为什么需要它（2026-09-15 实测）：`stock_cyq_chips` 一轮 **76 个请求 / 651 秒
      = 8.6 秒/请求**，而请求数本身已经是批量化后的理论最优
      （`总行数 ÷ page_size(10000)`，**改批大小改变不了请求数**）。
      唯一的杠杆就是**并发**：76 请求 × 8.6s ÷ 12 并发 ≈ **55 秒**，而不是 11 分钟。

    限速由 `Client` 里的**跨进程全局限速器**兜底（260/分钟），
    并发只会把额度用满、不会超限 —— 实测并发下速率仍是 260/min。

    写在**主线程**（`_Buf` / `_save` / `manifest`）必须串行：共享盘没有文件锁，
    并发写同一个分区会互相覆盖。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if workers <= 1 or len(items) <= 1:
        for it in items:
            try:
                rows, err = fn(it)
            except Exception as exc:  # noqa: BLE001
                rows, err = [], exc
            yield it, rows, err
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fu in as_completed(futs):
            it = futs[fu]
            try:
                rows, err = fu.result()
            except Exception as exc:  # noqa: BLE001
                rows, err = [], exc
            yield it, rows, err


def _filter_rows(rows: list[dict], codes: list[str], code_param: str) -> list[dict]:
    """把"按日查"返回的超集过滤回本表的实体清单。

    ⚠️ **代码格式不一致是这个过滤的第一个坑**（2026-09-14 实测踩到）：
       `tdx_blocks.block_code` 存的是**裸码** `880201`，
       而 `/tdx/daily` 返回的 `board_code` 带后缀 `880201.TDX` —— 直接精确匹配交集为 0，
       结果是"服务端明明有 2238 行，本地却一行都写不进去，而且不报错"。
       所以这里按**清单本身的格式**决定比对方式：
         · 清单带点（如 stock_list 的 `000001.SZ`）→ 精确匹配
         · 清单不带点（如 tdx_blocks 的 `880201`）→ 去掉返回值的后缀再比
    """
    if not rows or not codes:
        return rows
    if any("." in str(c) for c in codes):
        keep = {str(c) for c in codes}
        return [r for r in rows if str(r.get(code_param)) in keep]
    keep_norm = {str(c).split(".")[0] for c in codes}
    return [r for r in rows if str(r.get(code_param)).split(".")[0] in keep_norm]


def _months_between(start: str, end: str) -> list[str]:
    """闭区间的月份列表：`2026-07-31`→`2026-09-14` 得 `['2026-07','2026-08','2026-09']`。"""
    out: list[str] = []
    try:
        y, m = int(str(start)[:4]), int(str(start)[5:7])
        ey, em = int(str(end)[:4]), int(str(end)[5:7])
    except (TypeError, ValueError):
        return out
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


# 落后时缩短节流用的下限（天）：数据没跟上就少等几天再试，但也不能每轮都刷。
CATCHUP_EVERY_DAYS = 7
# 尾部窗口下界最多往前放宽多少天 —— 防止个别长期没跑的表一次撑出上万请求。
ANCHOR_MAX_LOOKBACK_DAYS = 400


def _minus_days(date_str: str, n: int) -> str:
    from datetime import date, timedelta
    return (date.fromisoformat(str(date_str)[:10]) - timedelta(days=n)).isoformat()


def _expected_latest_month() -> str:
    """月频表「本该有的最新一期」= 上个月（YYYY-MM）。

    为什么取上个月而不是本月：月频快照打在每月 1 日，本月那期要到下月初才出现，
    月中来问必然还没有。这里只用来做「是否落后」的粗判 —— 宁可判成落后（多刷几次），
    也不要判成跟上（那就退回永久冻结了）。
    """
    from datetime import date
    d = date.today()
    y, m = d.year, d.month - 1
    if m < 1:
        y, m = y - 1, 12
    return f"{y:04d}-{m:02d}"


def _is_behind(ds: R.DS, man: state.Manifest) -> bool:
    """本地末月是否早于「日历上的上个月」（目前只判月频表）。

    ⚠️⚠️ **这个判据看不到服务端**，所以对「厂商自己就滞后」的表会**永久为真**。
    2026-09-16 实测 `index_weight`：本地 2026-07、服务端 `server_max` **也是** 2026-07
    （厂商月频延迟 ≥1 个月），可 `_is_behind` 永远返回 True —— 它只比"日历上的上个月"。

    这是**有意接受的取舍**：宁可多刷几次，也不要退回"30 天冻结 → 上游补发也抓不到"
    （那正是本轮修掉的 bug）。代价是 index_weight 的节流从 30 天缩到 7 天，
    约每周多 400 个请求 —— 轻微效率损失，不是正确性问题。

    ⇒ 因此**调用处的措辞不能说"本地落后"**（会误导成我方故障），见 `_period_skip`。
    要根治得让它感知服务端水位（闸门已经在探 `server_max`，只是没传到这里）。
    """
    if not ds.entity_range or ds.date_step != "month":
        return False
    last = man.max_partition_date()
    if not last:
        return False                    # 本地一行都没有 → 走正常回填路径，不算"落后"
    return str(last)[:7] < _expected_latest_month()


def _anchor_lo(ds: R.DS, man: state.Manifest, lo: str, hi: str) -> str:
    """把 per_entity 的**尾部窗口下界**锚到本地最后一期，防止窗口滑过未抓到的期。

    背景（2026-09-16 修）：下界原本只按 `T − 固定天数` 算。一旦本地数据落后
    （如 `index_weight` 冻在 2026-07），窗口会随 T 一起前移，把还没抓到的月份滑出视野
    —— 上游补发也抓不回来，而且窗口越滑越远，**永远不会自愈**。
    锚定后语义变成「本地到哪儿就从哪儿往后补」，落后多少补多少，跟上后自然收敛。

    ⚠️ 只对有日期轴的表生效（`entity_range=False` 的纯快照表没有日期轴）。
    ⚠️ 放宽有上限，避免长期没跑的表一次撑出上万请求。
    """
    if not ds.entity_range:
        return lo
    last = man.max_partition_date()
    if not last:
        return lo
    last = str(last)[:10]
    if last >= str(lo)[:10]:
        return lo                       # 本地不落后 → 窗口原样不动
    return max(last, _minus_days(hi, ANCHOR_MAX_LOOKBACK_DAYS))


def _entity_periods(ds: R.DS, start: str, end: str) -> list[tuple]:
    """把 `[start, end]` 按该表的 `date_step` 切成"请求期"。

    ★★ 2026-09-15 补回**移植时丢掉的两条约定**（旧工程 `engine.make_payload` 有）：
      · `entity_range=False` → 该表**没有日期轴**（纯快照，如 `tdx_block_stocks`、
        `index_ths_constituent_stocks`）→ 请求里**不能带任何日期参数**；
      · `date_step="month"` → **月频**表（如 `index_weight`）→ 传
        `date_param=YYYY-MM`，**不是** start_time/end_time 区间。

    本工程的 `DS` 声明了这两个字段（registry.py），却**没有任何代码读它们** ——
    于是 `index_weight` 被当成"按天查"的表，发的是
    `{index_code, start_time: T, end_time: T}`，而它的数据打在**每月 1 日**，
    这个请求**必然返回 0 行**。实测后果：数据冻在 2026-07-01，落后 T 达 53 个交易日，
    且 `done.entities` 已满使按年回填分支不再执行 → 永不更新。
    """
    if not ds.entity_range:
        return [(None, None)]
    if ds.date_step == "month":
        return [(m, m) for m in _months_between(start, end)]
    out: list[tuple] = []
    try:
        y0, y1 = int(str(start)[:4]), int(str(end)[:4])
    except (TypeError, ValueError):
        return [(start, end)]
    for y in range(y0, y1 + 1):
        a, b = max(f"{y}-01-01", start), min(f"{y}-12-31", end)
        if a <= b:
            out.append((a, b))
    return out or [(start, end)]


def _entity_payload(ds: R.DS, g: list[str], a, b) -> dict:
    """per_entity 的请求体 —— 按 `date_step` / `entity_range` 决定日期参数怎么放。"""
    p = {**ds.params, ds.code_param: g if len(g) > 1 else g[0]}
    if not a:
        return p                        # 无日期轴的纯快照表
    if ds.date_step == "month":
        p[ds.date_param] = a            # YYYY-MM
    else:
        p[ds.start_param] = a
        p[ds.end_param] = b
    return p


def _period_skip(ds: R.DS, man: state.Manifest) -> str | None:
    """快照型/低频表按 `snapshot_every_days` 周期刷新；没到期就跳过。

    ★ 实测依据（2026-09-14）：`tdx_block_stocks`（618 实体）与
      `index_ths_constituent_stocks`（1676 实体）每天全刷要 **20+ 分钟**、
      2294 个请求，而成分股本来一周才变一次 —— 这是纯浪费。
    """
    every = int(ds.snapshot_every_days or 1)
    if every <= 1:
        return None
    fin = (man.last_run or {}).get("finished_at")
    if not fin:
        return None
    try:
        from datetime import datetime
        last = datetime.fromisoformat(str(fin)[:19])
    except ValueError:
        return None
    from datetime import datetime as _dt
    age_days = (_dt.now() - last).total_seconds() / 86400.0
    # ★ 2026-09-16 修：本地数据落后于该表自身节奏时**缩短节流**。
    #   原实现只看"距上次跑多久"，与"数据到底跟上没有"完全无关 —— index_weight
    #   冻在 2026-07、上游 8/9 月也没发，可只要 30 天没到就一路跳过，
    #   于是「上游补发」与「我们去抓」之间没有任何机制把两者连起来。
    #   现在改成：落后 → 最多等 CATCHUP_EVERY_DAYS 天就再试一次（但仍不每轮都刷，
    #   避免 200 个实体 × 每期 1 个请求的表把日更额度吃光）。
    behind = _is_behind(ds, man)
    effective = min(every, CATCHUP_EVERY_DAYS) if behind else every
    if age_days < effective:
        tag = ""
        if behind:
            # ⚠️ 措辞刻意**不写"本地落后"**：_is_behind 看不到服务端，对厂商自身就滞后的
            #    表（如 index_weight）它永久为真 —— 说成"本地落后"会误导成我方故障。
            #    详见 _is_behind 的 docstring。
            lm = str(man.max_partition_date())[:7]
            tag = (f"；★ 末次数据 {lm} 早于日历上的上个月，节流主动缩为每 {effective} 天重试"
                   f"（上游可能本就滞后，非我方故障）")
        return (f"快照类，每 {effective} 天刷一次；上次 {str(fin)[:16]}"
                f"（{age_days:.1f} 天前），本次跳过{tag}")
    return None


def _entity_codes(ds: R.DS) -> list[str]:
    """实体清单：entity_codes 显式给了就用它；否则从 entity_from 源表的**最新年分区**取。"""
    if ds.entity_codes:
        return list(ds.entity_codes)
    if not ds.entity_from:
        return []
    src, col = ds.entity_from
    years = store.list_years(src)
    f = store.partition_path(src, years[-1]) if years else store.flat_path(src)
    if not f.exists():
        return []
    import pyarrow.parquet as pq
    try:
        t = pq.read_table(f, columns=[col])
    except (OSError, ValueError, KeyError):
        return []
    codes = sorted({str(x) for x in t.column(0).to_pylist() if x})
    if ds.entity_prefix_filter:
        pre = tuple(ds.entity_prefix_filter)
        codes = [c for c in codes if c.startswith(pre)]
    return codes


def _entity_list_dates(ds: R.DS) -> dict[str, str]:
    """实体 → 上市日（`list_date`）。源表没有这一列就返回空 dict。

    ★★ 2026-09-18 新增：**新股特殊处理**。

    未完成实体的回填本来一律从 `ds.start` 开始（`stock_cyq_chips` = 2018-01-01）。
    于是一只 09-17 才上市的新股要发 **9 个年请求，其中 8 个必然返回 0 行** ——
    既白烧配额，又让日志里"回填 N 实体（M 个任务）"的 M 完全看不出实际工作量。
    实测标的：`601091.SH`（C沈鼓，`list_date=2026-09-17`，2026-09-18 首次进池）。

    回填下界抬到上市日后，同一只票只需 1 个请求。

    ⚠️ **只在源表是 `stock_list` 时生效**（显式白名单，不是"有这列就用"）——
       2026-09-18 实测：`index_ths_sector_categories`（`index_ths_daily` /
       `index_ths_constituent_stocks` 的实体源）**也有** `list_date` 列，但它的语义是
       **板块/指数自己的发布日期**（`700001.TI → 20061229`），跟"个股上市日"不是一回事。
       若不收窄，`index_ths_daily` 的回填下界会被这些值改写 —— 那是一张完全无关的表，
       不该被"新股特殊处理"顺手改掉。（`tdx_blocks` 没有该列，天然返回 `{}`。）
    ⚠️ 取不到某只票的上市日（列缺失/值为空）时**不猜**，该组回落到 `ds.start`（宁可多发请求）。
    """
    if not ds.entity_from or ds.entity_from[0] != "stock_list":
        return {}
    src, col = ds.entity_from
    years = store.list_years(src)
    f = store.partition_path(src, years[-1]) if years else store.flat_path(src)
    if not f.exists():
        return {}
    import pyarrow.parquet as pq
    try:
        # 列不存在时 pyarrow 抛 `ArrowInvalid`（它是 ValueError 的子类），被下面接住
        t = pq.read_table(f, columns=[col, "list_date"])
    except (OSError, ValueError, KeyError):
        return {}
    out: dict[str, str] = {}
    for c, d in zip(t.column(0).to_pylist(), t.column(1).to_pylist()):
        if c and d:
            out[str(c)] = str(d)[:10]
    return out


def run_per_entity(ds: R.DS, man: state.Manifest, ctx: Ctx) -> Result:
    """both 轴：**按请求数增长更慢的轴选策略**。

    · `stock_history_5min` / `tdx_minute` / `index_ths_daily` / `tdx_daily`（实体数 618~5901）
        → 尾部窗口 + **批量编组**（服务端支持数组时 1676→17、618→7 个请求）
    · `stock_cyq_chips`（3484 实体、单股单年 2.4 万行）
        → 尾部窗口 + 批量 100（2026-09-14 实测数组可用 → 3484→35 个请求）
    · `tdx_block_stocks` / `index_ths_constituent_stocks`（**快照型**，无日期轴）
        → 按 `snapshot_every_days` 周期刷新（默认每周）。实测每天全刷要
          618 + 1676 个请求、**20+ 分钟**，而成分股本来一周才变一次。
    · `index_weight`（月频）→ 按 30 天周期刷

    ★ 死锁解除：旧实现在 `done.entities` 满之后**每轮 0 个任务**，`max_date` 永久冻结
      （`tdx_daily`/`index_ths_daily` 明天就会因此让旧闸门永久 exit 3）。
      这里对**已完成实体也补尾部窗口**，所以永远在前进。
    """
    res = Result(ds.name, ds.mode)
    t0 = time.monotonic()

    # ★ 周期闸门：快照型/低频表不必每天刷（它们的实体数大、每天刷是纯粹的浪费）
    skip = _period_skip(ds, man)
    if skip:
        res.status = "⊘"
        res.note = skip
        res.seconds = time.monotonic() - t0
        return res

    codes = _entity_codes(ds)
    if not codes:
        res.status = "⚠️"
        res.note = "取不到实体清单"
        return res

    win = ds.entity_window_days if ds.entity_window_days is not None else ds.window(ctx.redundancy)
    lo, hi = ctx.data_window(ds, win)      # 上界 = T - delay（如 index_ths_daily delay=1）
    # ★ 2026-09-16 修「尾部窗口会滑走」：把下界锚到本地最后一期，
    #   否则本地一落后，窗口就随 T 前移、把没抓到的期永久滑出视野。
    lo_raw = lo
    lo = _anchor_lo(ds, man, lo, hi)
    if lo != lo_raw:
        # 可观测判据：锚定生效时必须在日志里看得见，否则没法证明新逻辑真的跑了
        anchor_note = (f"   ★ {ds.name} 尾部窗口下界锚定：{lo}（原 {lo_raw}）"
                       f"—— 本地落后，从末次数据起补，防止窗口滑过未抓到的期")
        if ctx.runner:
            ctx.runner.note(anchor_note)
        res.anchored_lo = lo
    else:
        res.anchored_lo = None

    # ★★ 按日查优化（2026-09-14 实测）：有些 per_entity 接口**同时支持按日期查全部实体**
    #    （`/tdx/daily` 传 trade_date 一次返回 1119 个板块）。
    #    这时 618 个实体只要 **1 个请求/天**，而不是逐个实体 618 个。
    if ds.name in BY_DATE_OVERRIDE:
        return _run_by_date(ds, man, ctx, codes, lo, hi, res)

    if ctx.dry_run:
        n_batch = max(1, ds.entity_batch)
        win_p = _entity_periods(ds, lo, hi)
        step = ("无日期轴" if not ds.entity_range
                else "按月" if ds.date_step == "month" else "按年/区间")
        res.note = (f"{len(codes)} 个实体 × 窗口 {lo}~{hi}（{step}，{len(win_p)} 期）"
                    f"（批量 {n_batch} → 约 {_ceil(len(codes), n_batch) * len(win_p)} 个请求）")
        res.seconds = time.monotonic() - t0
        return res

    # 未完成的实体要全量回填；已完成的实体只补尾部窗口
    todo_new = [c for c in codes if not man.is_done("entities", c)]
    rest = [c for c in codes if c not in set(todo_new)]
    batch = max(1, ds.entity_batch)
    buf = _Buf(ds, man)

    workers = int(ctx.cfg.get("api", {}).get("concurrency", 12))
    errs: list[str] = []

    def _fetch(task: tuple) -> tuple[list[dict], Exception | None]:
        """一个任务 = (实体组, 期起, 期止)。**只抓数据，不碰 buf/manifest**（并发安全）。

        ★★ 2026-09-18 修（**新股回填完全失效**）：这里原来写死 `g, a, b = task`，
          而**回填**任务自"失败时整组不标完成"那次改动起是 **4 元组**
          `(gi, g, a, b)`（见下面 `back_tasks`）—— 于是每一个回填任务都抛
          `ValueError: too many values to unpack (expected 3, got 4)`，
          被下面的 `except` 吞成"请求失败"，整组不标完成。
          后果：**不在 `done.entities` 里的实体永远补不上历史**，每晚只留一条 ⚠️。
          实测 601091.SH（C沈鼓，list_date=2026-09-17）在 `stock_cyq_chips` 里
          **行数为 0**，9 个年任务全是同一个错。
        取**末 3 位**而不是去改回填任务的形状：尾部窗口的 `tail_tasks` 本来就是
        3 元组，两种形状都必须能跑。
        """
        g, a, b = task[-3:]
        payload = _entity_payload(ds, g, a, b)
        try:
            rows = ctx.client.fetch_all(ds.path, payload, ds.page_size,
                                        method=ds.method, expect_rows=False,
                                        max_rows=ctx.client.max_rows_per_query)
            if not getattr(rows, "complete", True):
                return rows, ApiError("分页 total 不符，保留已取数据但不标完成")
            return rows, None
        except QueryLimitError as exc:
            # A batch that fits a daily tail may exceed 100k rows for a year's
            # new-entity backfill. Split the entity group; never mark a failed
            # group done or retry the same oversized request forever.
            if len(g) <= 1:
                return [], exc
            mid = len(g) // 2
            left, le = _fetch((g[:mid], a, b))
            right, re = _fetch((g[mid:], a, b))
            return FetchRows([*left, *right], complete=not (le or re)), le or re
        except Exception as exc:  # noqa: BLE001
            return [], exc

    # ① 未完成实体：按 `date_step`（年/月/无日期轴）回填，粒度与历史数据一致
    #    任务带上组号 gi，失败时整组不标完成（下轮重试）
    #
    #    ★★ 2026-09-18 **新股特殊处理**：回填下界不再一律用 `ds.start`，
    #      而是抬到该组内**最早的上市日**（`list_date`；源表没有这一列时回落 `ds.start`）。
    #      一只 09-17 上市的新股本来要发 9 个年请求、其中 8 个必然空手；
    #      抬到上市日后只需 1 个请求（实测 601091.SH / C沈鼓）。
    #      · 取组内 **min**（最早的上市日）而不是逐实体算：批量编组下这样最保守 ——
    #        组里只要有一只老票，窗口就按老票铺满，**绝不会因为组里有新股而漏掉老票的历史**；
    #      · 再夹在 `[ds.start, hi]` 内，保证 `_entity_periods` 拿到合法区间
    #        （上市日晚于 `hi` 时不会构造出 start > end 的畸形请求）。
    starts = _entity_list_dates(ds)
    back_tasks: list[tuple[int, list[str], str, str]] = []
    group_periods: dict[int, list[tuple]] = {}
    for gi, i in enumerate(range(0, len(todo_new), batch)):
        g = todo_new[i:i + batch]
        g_start = min((starts.get(c) or ds.start for c in g), default=ds.start)
        g_start = min(max(str(g_start)[:10], str(ds.start)[:10]), str(hi)[:10])
        per = _entity_periods(ds, g_start, hi)
        group_periods[gi] = per          # 成功回填后照它清 suspect 残迹
        for (a, b) in per:
            back_tasks.append((gi, g, a, b))

    if back_tasks:
        bad_groups: set[int] = set()
        for idx, ((gi, g, a, b), rows, err) in enumerate(
                _parallel_map(_fetch, back_tasks, workers), 1):
            if err is not None:
                bad_groups.add(gi)
                errs.append(f"回填 {a}~{b}: {str(err)[:60]}")
                man.mark_suspect(f"{g[0] if len(g) == 1 else f'{len(g)}只'}|{a}~{b}|error")
            buf.add(pd.DataFrame(rows) if rows else None)
            if idx % 40 == 0:
                buf.flush()
        # ★ 先落盘再标完成（顺序反了会留"状态说完成、数据还在内存"的永久空洞）
        buf.flush()
        for gi, i in enumerate(range(0, len(todo_new), batch)):
            if gi in bad_groups:
                continue
            g = todo_new[i:i + batch]
            # ★ 2026-09-18：回填成功后**清掉前几轮留下的 suspect 残迹**。
            #   失败时写的是 `{实体}|{年区间}|error`（见上面的 mark_suspect），
            #   而成功后没人删它 —— 实测 601091.SH 修好后仍有 9 条
            #   `601091.SH|YYYY-01-01~YYYY-12-31|error` 挂在 manifest 上；
            #   又因为实体已 done、回填分支不再执行，这 9 条会**永久留存**。
            #   （`suspect` 目前不进报告，所以不产生假告警，但它把"这个实体有历史缺口"
            #    这条审计线索污染成了长期噪声。）
            label = g[0] if len(g) == 1 else f"{len(g)}只"
            for (a, b) in group_periods.get(gi, []):
                man.suspect.pop(f"{label}|{a}~{b}|error", None)
            for c in g:
                man.mark_done("entities", c)
        man.save()
        if ctx.runner:
            ctx.runner.note(f"   {ds.name:34} 回填 {len(todo_new)} 实体"
                            f"（{len(back_tasks)} 个任务）→ {fmt_rows(buf.rows)} 行")

    # ② 已完成实体：**尾部窗口无条件重抓**（这是解除死锁的关键）
    #    ★ 并发发出：cyq_chips 35 个任务，串行要十几分钟、6 worker 约 25 秒。
    #
    #    ⚠️⚠️ **必须一次性收齐再落盘，不能边抓边 flush** ——
    #    落盘的整段替换快路径（`replace_suffix=True`）是**"替换整个尾随后缀"**语义，
    #    分多批 flush 会让后一批把前一批刚写的实体**整段抹掉**。
    #    2026-09-15 实测踩到：`_Buf` 每 12 帧 flush 一次，cyq_chips 的 09-11
    #    最后只剩 **1054/3484 只**（只剩最后一批），是 MD5 台账精确报出来的。
    #    尾部窗口同样按 `date_step` 切期：月频表（index_weight）要的是 `YYYY-MM`
    #    而不是一个日期区间 —— 见 `_entity_periods` 的注释。
    tail_periods = _entity_periods(ds, lo, hi)
    tail_tasks = [(rest[i:i + batch], a, b)
                  for i in range(0, len(rest), batch)
                  for (a, b) in tail_periods]
    pending = man._extra.setdefault("pending_entity_windows", [])
    for task in pending:
        g = [c for c in task["codes"] if c in codes]
        if g and (not task["end"] or task["end"] <= hi):
            tail_tasks.append((g, task["start"], task["end"]))
    unique = {}
    for g, a, b in tail_tasks:
        unique[(tuple(g), a, b)] = (g, a, b)
    tail_tasks = list(unique.values())
    frames: list[pd.DataFrame] = []
    completed_tasks: list[dict] = []
    for idx, (_t, rows, err) in enumerate(_parallel_map(_fetch, tail_tasks, workers), 1):
        if err is not None:
            errs.append(f"尾部窗口 {lo}~{hi}: {str(err)[:60]}")
            g, a, b = _t
            task = {"codes": g, "start": a, "end": b}
            if task not in pending:
                pending.append(task)
        else:
            g, a, b = _t
            # Remove a pending request only after its rows have been written.
            completed_tasks.append({"codes": g, "start": a, "end": b})
        if rows:
            frames.append(pd.DataFrame(rows))
        if ctx.runner and idx % 40 == 0:
            ctx.runner.note(f"   {ds.name:34} 尾部窗口 {lo}~{hi} {idx}/{len(tail_tasks)} 批"
                            f" → {sum(len(f) for f in frames):,} 行")

    # ★★ 2026-09-17：`replace_suffix` 的语义已收紧成
    #   「后缀里**只有与 new 主键相同的旧行**被取代，其余旧行原样保留」
    #   —— 与通用路径 `drop_duplicates(keys, keep='last')` 等价（见 store.upsert），
    #   所以**不再需要**"new 覆盖了整个后缀"这个前提，这里无条件打开：
    #     · 语义正确（漏发/撤数时本地不再被删 —— 用户 2026-09-15 的口径）
    #     · 快 ~18 倍（省掉 5,500 万行分区的 concat+去重+全表排序，实测 51s → 秒级）
    #   仍然**只 flush 一次**（`_Buf` 的阈值抬到无穷）：分多次 flush 会让请求白跑。
    tbuf = _Buf(ds, man, flush_rows=10**9, flush_batches=10**9,
                replace_suffix=True)
    for f in frames:
        tbuf.add(f)
    tbuf.flush()                        # ★ 一次性落盘
    # Data write succeeded. Requests that failed this time remain in the queue.
    man._extra["pending_entity_windows"] = [t for t in pending if t not in completed_tasks]

    if errs:
        res.status = "⚠️"
        res.note = f"{len(errs)} 个请求失败，例：{errs[0]}"

    res.rows = buf.rows + tbuf.rows
    res.seconds = time.monotonic() - t0
    if not res.note:
        anchor_tag = "，★ 下界锚定" if res.anchored_lo else ""
        res.note = (f"{len(codes)} 实体（回填 {len(todo_new)} / 补窗 {len(rest)}），"
                    f"窗口 {lo}~{hi}{anchor_tag}，新增 {res.rows:,} 行")
    return res


# 支持"按日期一次查全部实体"的 per_entity 数据集 —— 实测确认后才会放进这里。
# 判据：`{"<date_param>": D}` 而不传实体代码，返回结果覆盖绝大多数实体。
BY_DATE_OVERRIDE = {
    "tdx_daily",      # 实测 2026-09-11 传 trade_date 返回 1119 个板块（本地实体清单 618 个）
}

# 四个财报表：实测**只传报告期 `end_date` 就能拿全市场**（一次 6,300+ 行），
# 所以按报告期查即可，不必逐股。
REPORT_PERIOD_TABLES = {
    "stock_income", "stock_balancesheet", "stock_cashflow", "stock_financial_indicator",
}
# 每次回看几个报告期（覆盖追溯重述：老报告期被重新公告）
PERIOD_LOOKBACK = 3


def recent_report_periods(T: str, n: int = 3) -> list[str]:
    """T 及以前的最近 n 个报告期（03-31 / 06-30 / 09-30 / 12-31），新的在前。"""
    y = int(str(T)[:4])
    allp = [f"{yy}-{md}" for yy in (y - 1, y, y + 1)
            for md in ("03-31", "06-30", "09-30", "12-31")]
    past = [p for p in allp if p <= str(T)[:10]]
    return past[-n:][::-1]


# ★★ 2026-09-19 新增（用户交办「**确认 main 能自己抓到，而不是每次靠 agent 审计才发现**」）。
#
# 为什么需要 `ann_date` 扫描：`recent_report_periods` 只枚举**最近 3 个报告期**，
#   而报告期是**有限枚举** —— 某一期一旦滑出窗口，厂商之后对它的任何新增
#   （追溯重述、补公告）就**永远拿不到**。实测存量代价（2010+ 全量扫描）：
#   4 张财报合计缺 **25,833 行**、**60/66 期有缺口**，最重的
#   `stock_financial_indicator` 缺 6.7%（2023-12-31 缺 1,285 / 2024-12-31 缺 1,190，
#   **落在模型训练窗口内**）。详见 README §9.11。
#
# 而 `ann_date` 是**真正的日期轴**：实测 `/stock/financial_indicator?ann_date=2026-09-10`
#   一次返回**跨 3 个不同 end_date**（2024-12-31 / 2025-12-31 / 2026-06-30）的行 ——
#   即「当天公告的全部内容，不分报告期」。于是**新报告期、老期重述、非标准报告期**
#   （README §9.12 那个机制缺口）**一并自动覆盖**，不再依赖枚举。
#
# ⚠️ 窗口必须用**日历日**：实测 353 个公告日里 **70 个不在交易日历**
#   （62 个周六 + 节假日）—— 用交易日历做窗口会**系统性漏掉周末公告**。
ANN_LOOKBACK_DAYS = 14


def all_report_periods(start: str, T: str) -> list[str]:
    """`[start, T]` 内的**全部**标准季末（升序）。用于一次性存量回填。"""
    lo, hi = str(start)[:10], str(T)[:10]
    out = [f"{y}-{md}" for y in range(int(lo[:4]), int(hi[:4]) + 1)
           for md in ("03-31", "06-30", "09-30", "12-31")]
    return [p for p in out if lo <= p <= hi]


# ★★ 26 号 `stock_forecast` 的**周期性全量兜底**（2026-09-19）。
#
# 为什么它比 25/27 脆：26 走的是特例分支，取数窗口 = `ann_date ∈ [T - revision, T]`，
#   而它的 `revision_days` 只有 **90 天**（25/27 是 400 天）。窗口越短，滑出去越快。
#   实测（2026-09-19）：90 天窗口外**每年都在漂**，2012~2026 合计缺 **460 行**
#   （5 / 4 / 9 / 14 / 23 / 15 / 26 / 18 / 16 / 24 / 69 / 68 / 55 / 113 / 1）。
#   而 25/27 因为窗口 400 天，窗口外扫描 **0 缺口**。
#
# 兜底 = 每 `FORECAST_SWEEP_EVERY_DAYS` 天做一次**全历史 ann_date 区间刷新**。
#   ⚠️ 不能一次查全历史：API 硬约束 `page × page_size ≤ 100000`，
#      全表 11 万行会被 400 拒掉 ⇒ **必须按年切**（17 个请求 / 实测 27.6 秒）。
FORECAST_SWEEP_STATE = paths.STATE / "forecast_sweep.json"
FORECAST_SWEEP_EVERY_DAYS = 30


def forecast_sweep_due(today: str | None = None) -> tuple[bool, str | None]:
    """距上次全历史刷新是否已达 `FORECAST_SWEEP_EVERY_DAYS` 天。"""
    import json as _json
    from datetime import date as _d
    if not FORECAST_SWEEP_STATE.exists():
        return True, None
    try:
        last = _json.loads(FORECAST_SWEEP_STATE.read_text(encoding="utf-8"))["last_sweep"]
    except (OSError, KeyError, ValueError):
        return True, None
    try:
        age = (_d.fromisoformat(today or _d.today().isoformat())
               - _d.fromisoformat(str(last)[:10])).days
    except ValueError:
        return True, None
    return age >= FORECAST_SWEEP_EVERY_DAYS, str(last)[:10]


def mark_forecast_swept(day: str) -> None:
    import json as _json
    FORECAST_SWEEP_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FORECAST_SWEEP_STATE.with_suffix(".tmp")
    tmp.write_text(_json.dumps({"last_sweep": day}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(FORECAST_SWEEP_STATE)


def ann_scan_days(T: str, n: int = ANN_LOOKBACK_DAYS) -> list[str]:
    """ann_date 扫描窗口：截至**今天**的最近 n 个**日历日**（升序）。

    上界取 `max(T, 今天)` —— `T` 是服务端已发布的最新交易日，而**公告可以在周末/假期发**
    （实测 62/353 个公告日在周六），取今天才不会漏掉 T 之后那个周末的公告。
    """
    from datetime import date as _d, timedelta as _td
    hi = max(_d.fromisoformat(str(T)[:10]), _d.today())
    return [(hi - _td(days=k)).isoformat() for k in range(n - 1, -1, -1)]


def _run_by_date(ds: R.DS, man: state.Manifest, ctx: Ctx, codes: list[str],
                 lo: str, hi: str, res: Result) -> Result:
    """按日期一次拿全部实体 —— 请求数从"实体数"降到"天数"。"""
    t0 = time.monotonic()
    days = cal_mod.trading_days_between(ctx.cal, lo, hi)
    pending_days = [k.split("|")[0] for k in man.suspect
                    if "|" in k and k.split("|")[1] in ("error", "empty", "total_mismatch")]
    days = sorted(set(days) | {d for d in pending_days if ds.start <= d <= hi and d in ctx.cal})
    if ctx.dry_run:
        res.note = f"按日查：{len(days)} 天 × 1 请求（而非 {len(codes)} 个实体 × {len(days)} 天）"
        res.seconds = time.monotonic() - t0
        return res

    buf = _Buf(ds, man, flush_batches=8)
    for d in days:
        payload = {**ds.params, ds.date_param: d}
        try:
            rows = ctx.client.fetch_all(ds.path, payload, ds.page_size, method=ds.method,
                                        expect_rows=False,
                                        max_rows=ctx.client.max_rows_per_query)
        except Exception as exc:  # noqa: BLE001
            man.mark_suspect(f"{d}|error")
            res.status = "⚠️"
            res.note = f"{d} 失败: {str(exc)[:70]}"
            continue
        # 只保留本表实体清单里的（按日查会返回超集，避免悄悄扩大数据集口径）
        complete = getattr(rows, "complete", True)
        if not complete:
            man.mark_suspect(f"{d}|total_mismatch")
            res.status = "⚠️"
            res.note = f"{d} 分页不完整，下轮重试"
        n_all = len(rows)
        rows = _filter_rows(rows, codes, ds.code_param)
        if n_all and not rows:
            # 一行都没匹配上 —— 多半是代码格式不一致（见 _filter_rows 的注释），必须显式报警
            res.status = "⚠️"
            res.note = (f"{d} 返回 {n_all} 行但全部被实体过滤掉 —— 检查代码格式"
                        f"（样例 {str(rows[0].get(ds.code_param)) if rows else '?'}）")
        if rows:
            buf.add(pd.DataFrame(rows))
            buf.flush()
            if complete:
                man.mark_done("dates", d)
                for suffix in ("error", "empty", "total_mismatch"):
                    man.suspect.pop(f"{d}|{suffix}", None)
        elif not n_all and (ds.expect_rows or ds.freq == "daily_full"):
            man.mark_suspect(f"{d}|empty")
            res.status = "⚠️"
            res.note = f"{d} 预期有实体数据却为空，下轮重试"
        if ctx.runner:
            ctx.runner.note(f"   {ds.name:34} {d} → {len(rows)} 行")
    buf.flush()
    res.rows = buf.rows
    res.seconds = time.monotonic() - t0
    if not res.note:
        res.note = f"按日查 {len(days)} 天 → 新增 {res.rows:,} 行（{len(codes)} 个实体）"
    return res


# ================================================================ ④ per_stock
def run_per_stock(ds: R.DS, man: state.Manifest, ctx: Ctx) -> Result:
    """entity 轴，但**用报告期（end_date）当轴** —— 一次请求拿全市场。

    ★★ 2026-09-14 实测推翻的旧设计：
      原来的想法是"批量 100 只逐股全量重拉"（60 批 × 2 页 = 120 请求、**18 分钟**）。
      但实测发现这四个接口**只传 `end_date`（报告期）就能返回全市场所有公司**：
        /stock/income?end_date=2026-06-30         → 6,309 行
        /stock/financial_indicator?end_date=…     → 6,534 行
        /stock/cashflow?end_date=…                → 6,307 行
      于是每天只要拉**最近 3 个报告期**（覆盖追溯重述）：
        ~6,500 行/期 × 3 期 ÷ 10000/页 ≈ **6 个请求 / 几秒**。

      这正是用户说的"按接口的天然能力选策略"：这个接口天然是
      **按报告期（date 轴）** 的形状，不是按股票（entity 轴）的形状。
    """
    res = Result(ds.name, ds.mode)
    t0 = time.monotonic()

    # 四个财报表：按 end_date（报告期）取最近 N 期 + **ann_date 日历日窗口扫描**
    if ds.name in REPORT_PERIOD_TABLES:
        backfill = bool(getattr(ctx, "report_backfill", False))
        periods = (all_report_periods(ds.start, ctx.T) if backfill
                   else recent_report_periods(ctx.T, n=PERIOD_LOOKBACK))
        # ann_date 扫描：只在常规模式下做 —— 回填模式已按 end_date 覆盖全部历史，
        # 再来一遍 ann_date 是纯冗余（回填那次单独跑，不叠日常开销）。
        ann_days = [] if backfill else ann_scan_days(ctx.T)
        # A failed request must not disappear when its date/period leaves the
        # rolling window. Retain and retry it until a complete response is saved.
        periods = sorted(set(periods) | {k.split("|")[1] for k in man.suspect
                         if k.startswith("period|") and k.endswith("|error")}, reverse=True)
        if not backfill:
            ann_days = sorted(set(ann_days) | {k.split("|")[1] for k in man.suspect
                              if k.startswith("ann|") and k.endswith("|error")})
        if ctx.dry_run:
            res.note = (f"{'★回填：' if backfill else '按报告期查 '}"
                        f"{len(periods)} 个报告期 {periods[:2]}…{periods[-1:]}"
                        + (f" ＋ ann_date 扫描 {len(ann_days)} 天"
                           f"（{ann_days[0]}~{ann_days[-1]}）" if ann_days else "")
                        + "（而非逐股 5901 个）")
            res.seconds = time.monotonic() - t0
            return res

        # flush_batches：要 > 常规模式的总帧数（3 期 + 14 天 = 17），否则 `add` 会中途反复
        # flush，每次都整文件重写同一年分区（O(n²)，见 _Buf 的说明）。
        # 但**要封顶 20** —— 回填模式一次 66 期，若全堆内存就是 ~40 万行 × 上百列
        # （balancesheet/income 各 ~150 列）≈ 数百 MB，没必要。
        buf = _Buf(ds, man, flush_batches=min(20, max(4, len(periods) + len(ann_days) + 2)))
        got: list[str] = []
        ann_hit = 0
        completed_checks: list[str] = []

        def _mk_fetch(param: str):
            """只抓数据，不碰 buf/manifest（并发安全）。"""
            def _f(v: str):
                try:
                    rows = ctx.client.fetch_all(
                        ds.path, {**ds.params, param: v}, ds.page_size,
                        method=ds.method, expect_rows=False,
                        max_rows=ctx.client.max_rows_per_query)
                    if not getattr(rows, "complete", True):
                        return rows, ApiError("分页 total 不符，下轮重试")
                    return rows, None
                except Exception as exc:  # noqa: BLE001
                    return [], exc
            return _f

        # ★ 并发发这些请求：它们是**互相独立**的，串行纯粹是白等。
        #   实测每请求 6~21 秒（`stock_financial_indicator` 3 个请求共 56 秒、
        #   `stock_balancesheet` 42 秒），占一轮运行里"其他频率"段的相当一块。
        #   限速由 Client 的**跨进程全局限速器**兜底（260/分钟），并发只会用满不超限。
        _workers = int(ctx.cfg.get("api", {}).get("concurrency", 6))
        for p, rows, err in _parallel_map(_mk_fetch("end_date"), periods, _workers):
            if err is not None:
                man.mark_suspect(f"period|{p}|error")
                res.status = "⚠️"
                res.note = f"报告期 {p} 失败: {str(err)[:70]}"
                continue
            if rows:
                buf.add(pd.DataFrame(rows))
                got.append(p)
            completed_checks.append(f"period|{p}|error")
            if ctx.runner:
                ctx.runner.note(f"   {ds.name:34} 报告期 {p} → {len(rows):,} 行")

        # ---- ann_date 扫描（★ 兜住"滑出 3 期窗口后再也没有回头路"的更新）----
        ann_bad: list[str] = []
        for d, rows, err in _parallel_map(_mk_fetch("ann_date"), ann_days, _workers):
            if err is not None:
                man.mark_suspect(f"ann|{d}|error")
                ann_bad.append(d)
                continue
            if rows:
                buf.add(pd.DataFrame(rows))
                ann_hit += len(rows)
            completed_checks.append(f"ann|{d}|error")
        if ann_bad:
            res.status = "⚠️"

        buf.flush()
        for key in completed_checks:
            man.suspect.pop(key, None)
        res.rows = buf.rows
        res.seconds = time.monotonic() - t0
        if not res.note:
            base = (f"★回填 {len(got)} 个报告期 → 新增 {res.rows:,} 行" if backfill
                    else f"报告期 {got} → 新增 {res.rows:,} 行")
            res.note = (base + (f" ＋ ann_date 扫描命中 {ann_hit:,} 行"
                                f"（{len(ann_days)} 天）" if ann_days else "")
                        + (f"；{len(ann_bad)} 天扫描失败（下轮重试）" if ann_bad else ""))
        return res

    # stock_forecast 特例：支持 ann_date 区间查询，不需要逐股 → 1~2 个请求
    if ds.name == "stock_forecast":
        lo, hi = max(ds.start, ctx.rev_start(ds)), ctx.data_hi(ds)
        due, last = forecast_sweep_due()
        if ctx.dry_run:
            res.note = (f"走 ann_date 区间 {lo}~{hi}（1 个请求，而非 5901 个）"
                        + (f" ＋ ★周期性全历史兜底（上次 {last or '从未'}，按年切 17 请求）"
                           if due else f"（全历史兜底未到期，上次 {last}）"))
            res.seconds = time.monotonic() - t0
            return res
        try:
            rows = ctx.client.fetch_all(ds.path, {**ds.params,
                                                  "start_date": lo, "finish_date": hi},
                                        ds.page_size, method=ds.method,
                                        expect_rows=False,
                                        max_rows=ctx.client.max_rows_per_query)
        except Exception as exc:  # noqa: BLE001
            res.status = "⚠️"
            res.note = f"ann_date 区间查询失败: {str(exc)[:80]}"
            res.seconds = time.monotonic() - t0
            return res
        added = _save(ds, man, pd.DataFrame(rows)) if rows else 0
        if not getattr(rows, "complete", True):
            res.status = "⚠️"
            res.note = "ann_date 区间分页不完整，下轮重试"

        # ---- ★ 周期性全历史兜底（90 天窗口外的漂移，只有它能修）----
        sweep_note, sweep_added = "", 0
        if due:
            y0 = int(str(ds.start)[:4])
            y1 = int(str(ctx.T)[:4])
            bad = 0
            for y in range(y0, y1 + 1):
                try:
                    r2 = ctx.client.fetch_all(
                        ds.path, {**ds.params, "start_date": f"{y}-01-01",
                                  "finish_date": f"{y}-12-31"},
                        ds.page_size, method=ds.method, expect_rows=False,
                        max_rows=ctx.client.max_rows_per_query)
                except Exception:  # noqa: BLE001
                    bad += 1            # 某年失败：**不标已 swept**，下轮整体重来
                    continue
                if not getattr(r2, "complete", True):
                    bad += 1
                if r2:
                    sweep_added += _save(ds, man, pd.DataFrame(r2))
            if bad == 0:
                mark_forecast_swept(ctx.T)
                sweep_note = f" ＋ ★全历史兜底 {y0}~{y1} 补齐 {sweep_added:,} 行"
            else:
                sweep_note = f" ＋ 全历史兜底 {bad} 个年份失败（下轮重试）"
                res.status = "⚠️"
            if ctx.runner:
                ctx.runner.note(f"   {ds.name:34} ★全历史兜底 → 补齐 {sweep_added:,} 行")

        res.rows = added + sweep_added
        res.note = f"ann_date 区间 {lo}~{hi} → {added:,} 行（快路径）" + sweep_note
        res.seconds = time.monotonic() - t0
        return res

    codes = _stock_codes()
    if not codes:
        res.status = "⊘"
        res.note = "取不到股票清单（stock_list 为空？）"
        return res
    batch = max(1, ds.batching)
    groups = [codes[i:i + batch] for i in range(0, len(codes), batch)]

    if ctx.dry_run:
        res.note = f"{len(codes)} 只 / 批量 {batch} → 约 {len(groups)} 个请求（全历史重拉）"
        res.seconds = time.monotonic() - t0
        return res

    buf = _Buf(ds, man)
    for gi, grp in enumerate(groups):
        payload = {**ds.params, ds.code_param: grp if len(grp) > 1 else grp[0]}
        try:
            rows = ctx.client.fetch_all(ds.path, payload, ds.page_size,
                                        method=ds.method, expect_rows=False,
                                        max_rows=ctx.client.max_rows_per_query)
        except Exception as exc:  # noqa: BLE001
            res.status = "⚠️"
            res.note = f"批次 {gi} 失败: {str(exc)[:70]}"
            continue
        buf.add(pd.DataFrame(rows) if rows else None)
        if gi % 10 == 0:
            buf.flush()
            if ctx.runner:
                ctx.runner.note(f"   {ds.name:34} 批次 {gi}/{len(groups)} → {fmt_rows(buf.rows)} 行")
    buf.flush()

    res.rows = buf.rows
    res.seconds = time.monotonic() - t0
    if not res.note:
        res.note = f"{len(codes)} 只 / {len(groups)} 批全量重拉 → 新增 {res.rows:,} 行"
    return res


# ================================================================ ⑤ snapshot
def run_snapshot(ds: R.DS, man: state.Manifest, ctx: Ctx) -> Result:
    """无时间轴：按周期全量刷新。

    ⚠️ `basic_calendar` 特殊处理：要更新到 **T+30**，且用 upsert 而不是 overwrite
       （overwrite 一次短响应就把 2010 年以来的历史日历整表删掉）。
    """
    res = Result(ds.name, ds.mode)
    t0 = time.monotonic()
    if ds.name == "basic_calendar":
        res.status = "✔"
        res.note = "由 calendar 模块专门处理（更新到 T+30，upsert 保全历史）"
        res.seconds = time.monotonic() - t0
        return res

    if ctx.dry_run:
        res.note = f"快照刷新（周期 {ds.snapshot_every_days} 天），变体 {len(ds.variants or [None])} 个"
        res.seconds = time.monotonic() - t0
        return res

    frames = []
    for variant in (ds.variants or [None]):
        payload = {**ds.params, **(variant or {})}
        if ds.range_params:
            payload[ds.start_param] = ds.start
            payload[ds.end_param] = ctx.T
        try:
            if ds.paginated:
                rows = ctx.client.fetch_all(ds.path, payload, ds.page_size,
                                            method=ds.method, expect_rows=ds.expect_rows,
                                            max_rows=ctx.client.max_rows_per_query)
                if not getattr(rows, "complete", True):
                    res.status = "⚠️"
                    res.note = "快照分页不完整，未替换本地数据，下轮重试"
            else:
                data = ctx.client.call(ds.path, payload, method=ds.method,
                                       expect_rows=ds.expect_rows)
                rows = extract_list(data)
        except Exception as exc:  # noqa: BLE001
            res.status = "⚠️"
            res.note = f"变体 {variant} 失败: {str(exc)[:70]}"
            continue
        if rows:
            df = pd.DataFrame(rows)
            for k, v in (variant or {}).items():
                df[k] = v
            frames.append(df)
    # Whole-snapshot replacement is all-or-nothing across variants. A failed
    # small variant can evade the row-count shrink guard and erase valid rows.
    if frames and res.status == "✔":
        res.rows = _save(ds, man, pd.concat(frames, ignore_index=True))
    elif not frames and res.status == "✔":
        res.status = "⚠️"
        res.note = "快照返回空，未替换本地数据，下轮重试"
    res.seconds = time.monotonic() - t0
    if not res.note:
        res.note = f"快照 {res.rows:,} 行"
    return res


# ================================================================ ⑥ dump
def run_dump(ds: R.DS, man: state.Manifest, ctx: Ctx) -> Result:
    """daily_dump → stock_history_5min（P4 阶段实现，见 dump_bridge）。"""
    from . import dump_bridge
    return dump_bridge.run(ds, man, ctx)


# ================================================================ 分派
STRATEGIES = {
    "range": run_range,
    "per_date": run_per_date,
    "per_entity": run_per_entity,
    "per_stock": run_per_stock,
    "snapshot": run_snapshot,
    "dump": run_dump,
}


# ★ 走 daily_dump 通道的数据集：1 个请求拿全市场当日 5min，
#   替代逐股 5901 个请求（实测 685,917,526 行的表逐股补一天要 5901 次）。
DUMP_BRIDGED = {"stock_history_5min"}


def run_one(ds: R.DS, man: state.Manifest, ctx: Ctx) -> Result:
    if ds.name in DUMP_BRIDGED:
        from . import dump_bridge
        return dump_bridge.run(ds, man, ctx)
    fn = STRATEGIES.get(ds.mode)
    if fn is None:
        return Result(ds.name, ds.mode, status="⊘", note=f"未知模式 {ds.mode}")
    t0 = time.monotonic()
    s0 = dict(ctx.client.stats)
    try:
        res = fn(ds, man, ctx)
    except Exception as exc:  # noqa: BLE001
        import traceback
        res = Result(ds.name, ds.mode, status="✘", note=f"{type(exc).__name__}: {str(exc)[:120]}")
        traceback.print_exc()
    res.requests = ctx.client.stats["requests"] - s0.get("requests", 0)
    res.retries = ctx.client.stats["retries"] - s0.get("retries", 0)
    res.empty_retries = ctx.client.stats.get("empty_retries", 0) - s0.get("empty_retries", 0)
    if (ctx.client.stats.get("total_mismatch", 0) > s0.get("total_mismatch", 0)
            and res.status == "✔"):
        res.status = "⚠️"
        res.note += "；本轮存在分页 total 不符，不能确认完整"
    if res.seconds <= 0:
        res.seconds = time.monotonic() - t0
    return res


# ================================================================ 工具
def _merge_ranges(ranges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not ranges:
        return []
    rs = sorted((a, b) for a, b in ranges if a <= b)
    out = [list(rs[0])]
    for a, b in rs[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _missing_ranges(man: state.Manifest, start: str, end: str) -> list[tuple[str, str]]:
    """coverage 之外、[start, end] 之内的缺口。"""
    cov = sorted((a, b) for a, b in (man.coverage or []) if b >= start and a <= end)
    out = []
    cur = start
    for a, b in cov:
        if a > cur:
            out.append((cur, _prev_day(a)))
        cur = max(cur, _next_day(b))
        if cur > end:
            break
    if cur <= end:
        out.append((cur, end))
    return [(a, b) for a, b in out if a <= b]


def _prev_day(s: str) -> str:
    from datetime import timedelta
    return (_date.fromisoformat(s[:10]) - timedelta(days=1)).isoformat()


def _next_day(s: str) -> str:
    from datetime import timedelta
    return (_date.fromisoformat(s[:10]) + timedelta(days=1)).isoformat()


def _split_chunks(a: str, b: str, ds: R.DS, ctx: Ctx) -> list[tuple[str, str]]:
    """按 chunk_days 切块（只对 range 的大表有意义；日常尾部窗口很小，通常一块）。"""
    from datetime import datetime, timedelta
    cd = ds.chunk_days or {"large": 7, "medium": 30, "small": 90, "huge": 7, "slow": 1}.get(ds.tier, 30)
    if cd <= 0:
        return [(a, b)]
    out = []
    cur = datetime.fromisoformat(a)
    end = datetime.fromisoformat(b)
    while cur <= end:
        nxt = min(cur + timedelta(days=cd - 1), end)
        out.append((cur.date().isoformat(), nxt.date().isoformat()))
        cur = nxt + timedelta(days=1)
    return out


def _ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def _stock_codes() -> list[str]:
    from .. import paths
    import pyarrow.parquet as pq
    f = store.flat_path("stock_list")
    if not f.exists():
        return []
    try:
        t = pq.read_table(f, columns=["stock_code"])
        return sorted({str(x) for x in t.column(0).to_pylist() if x})
    except (OSError, ValueError, KeyError):
        return []
