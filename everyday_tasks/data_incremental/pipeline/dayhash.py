"""单日 MD5 台账 —— 检测厂商偷偷修改历史数据。

用户 2026-09-14 的要求（原话）：
    「为了确保历史下载的数据不会随着时间改变（防止数据厂商偷偷修改历史数据），
     还需要记录好单日界面数据的 MD5，这个需要针对每一个数据文件进行单独保存，
     确保不同时间下载的数据保持统一，如果出现变化能够快速的定位是哪一个数据那一天。」

做法：
    · 存储：`state/dayhash/<数据集>.parquet`，列 [date, md5, rows, cols_md5, updated_at]
            （一天一行；一个数据集最多几千行，读写都是毫秒级）
    · 时机：**每次增量之后**，对本轮的尾部窗口（T-5…T）重算指纹并与台账比对
    · 命中：MD5 不同 = 上游改了历史 → 报告里定位到 **数据集 / 日期 / 变化的列**，
            旧值→新值、旧行数→新行数一并给出
    · 首建：`main.py init-hashes` 一次性给全部历史补台账（读全库约 5~10 分钟）

⚠️ 指纹算法用的是逻辑指纹（见 tools/md5.py），不是 parquet 文件字节 ——
   文件字节会因 row group 切分/zstd 帧边界变化，用它判"数据变了"必然假红。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import pyarrow.parquet as pq

from .. import paths, registry as R
from ..core import store
from ..tools import md5 as M


@dataclass
class DayChange:
    dataset: str
    date: str
    old_md5: str
    new_md5: str
    old_rows: int
    new_rows: int
    changed_cols: list[str]

    def as_dict(self) -> dict:
        return {"dataset": self.dataset, "date": self.date,
                "old_md5": self.old_md5, "new_md5": self.new_md5,
                "old_rows": self.old_rows, "new_rows": self.new_rows,
                "changed_cols": self.changed_cols}


def _ledger_path(dataset: str):
    return paths.DAYHASH_DIR / f"{dataset}.parquet"


def load_ledger(dataset: str) -> pd.DataFrame:
    p = _ledger_path(dataset)
    if not p.exists():
        return pd.DataFrame(columns=["date", "md5", "rows", "cols_md5", "updated_at"])
    # 台账是只读参考，读坏了当空即可（不要让台账问题阻断增量主流程）
    return store.read_parquet(p, on_error="empty")


def _candidate_files(dataset: str, ds: R.DS, years: set[int] | None = None):
    """该表**可能**含目标日期的分区文件清单。

    ★★ 2026-09-16 修既有 bug（被 `ledger_field` 改动暴露出来的）。

    **问题**：分区是按**抓取键**（`ds.date_field`）的年份切的，而台账按 `ledger_field` 取键。
      台账代码原先假设「日期的年份 == 分区年份」（`partition_path(ds.name, int(date[:4]))`）——
      对所有 `ledger_field == date_field` 的表**都成立**，所以从没暴露。
      但 `stock_holder_number` 两者不同（`ann_date` vs `end_date`），于是：
      实测 `ann_date=2018-01-01` 那条的 `end_date=2017-12-29` → 它住在 **`year=2017`** 分区，
      而旧代码去翻 `year=2018` → **找不到**，台账凭空少 44 天（全是跨年边界）。

    **做法**：两者的年份口径一致时走原来的**快路径**（只读那一个分区，保持既有性能）；
      不一致时走**慢路径**（扫全部分区）—— 这类表目前只有 `stock_holder_number`
      （11 个分区 / 42.7 万行，全扫代价可忽略），不会拖慢 `stock_cyq_chips` 那种大表。
    """
    col = ds.ledger_date_field()
    if years:
        ys = sorted(years)
        # 快路径：台账键就是抓取键 → 行必在同一年的分区里
        if col == ds.date_field and len(ys) == 1:
            f = store.partition_path(dataset, ys[0])
            if f.exists():
                return [f]
            f = store.flat_path(dataset)
            return [f] if f.exists() else []
    all_years = store.list_years(dataset)
    if not all_years:
        f = store.flat_path(dataset)
        return [f] if f.exists() else []
    if col == ds.date_field and years:
        all_years = [y for y in all_years if int(y) in set(ys)]
    files = [store.partition_path(dataset, y) for y in all_years]
    return [f for f in files if f.exists()]


def _read_day(dataset: str, ds: R.DS, date: str) -> pd.DataFrame:
    """读出某一天的**全部行**。

    ★ 用的是 `ds.ledger_date_field()`（台账键），**不是** `ds.date_field`（抓取键）——
      两者对多数表相同，但 `stock_holder_number` 故意不同（见 `DS.ledger_field`）。
      分区年份与台账日期的年份**可能不是一回事**，见 `_candidate_files`。
    注意日期列可能是 `trade_time`（分钟表）—— 按前 10 字符切。
    """
    col = ds.ledger_date_field()
    try:
        year = int(str(date)[:4])
    except ValueError:
        return pd.DataFrame()
    want = str(date)[:10]
    parts = []
    for f in _candidate_files(dataset, ds, {year}):
        try:
            pf = pq.ParquetFile(f)
            if col not in pf.schema_arrow.names:
                continue
            for batch in pf.iter_batches(batch_size=2_000_000):
                d = batch.to_pandas()
                s = d[col].astype(str).str[:10]
                sub = d[s == want]
                if len(sub):
                    parts.append(sub)
        except (OSError, ValueError):
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _read_days(dataset: str, ds: R.DS, dates: list[str]) -> dict[str, pd.DataFrame]:
    """**一次扫描**分区，把所有目标日的行切出来。

    ★★ 2026-09-15 性能修复。旧实现是"每个日期调一次 `_read_day`"，
       而 `_read_day` 每次都要把**整个年分区**扫一遍再按日期过滤 ——
       尾部窗口 6 个交易日 = **6 次全表扫描**。对 `stock_cyq_chips`
       （2026 分区 5,500 万行）和 `stock_history_5min`（6 亿行 / 600MB）纯属浪费：
       实测「MD5 台账」阶段占一轮运行 **173 秒（23%），仅次于抓取本身**。
       扫一次、切多天，成本直接降到 1/6。
    """
    col = ds.ledger_date_field()          # ★ 台账键（默认=date_field），见 DS.ledger_field
    want = {str(d)[:10] for d in dates}
    by_year: dict[int, set[str]] = {}
    for d in want:
        try:
            by_year.setdefault(int(d[:4]), set()).add(d)
        except ValueError:
            continue
    out: dict[str, list[pd.DataFrame]] = {d: [] for d in want}
    # ★ 两条路（见 _candidate_files 的说明）：
    #   快路径 —— 台账键 == 抓取键：按年单读、每年只切那年的日期（**与改前逐字等价**）。
    #   慢路径 —— 两者不同：分区年份与台账日期年份可能不是一回事，一次扫全部分区、切全部目标日。
    if col == ds.date_field:
        groups = [(by_year[y], _candidate_files(dataset, ds, {y})) for y in by_year]
    else:
        groups = [(want, _candidate_files(dataset, ds, set(by_year)))]
    for days, files in groups:
        for f in files:
            try:
                pf = pq.ParquetFile(f)
                if col not in pf.schema_arrow.names:
                    continue
                for batch in pf.iter_batches(batch_size=2_000_000):
                    d = batch.to_pandas()
                    s = d[col].astype(str).str[:10]
                    hit = s.isin(days)
                    if not bool(hit.any()):
                        continue
                    sub = d[hit]
                    for day, grp in sub.groupby(s[hit], sort=False):
                        out.setdefault(str(day)[:10], []).append(grp)
            except (OSError, ValueError):
                continue
    return {d: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
            for d, v in out.items()}


def trackable(ds: R.DS) -> bool:
    """这张表该不该有单日 MD5 台账。

    ★★ 2026-09-15 深夜抽出：改造前这条判据**三处各写一遍**
      （`runner.py` 的每轮阶段、`dayhash.init_hashes`、以及报告侧），
      而它们已经漂移过 —— `tdx_block_stocks` / `index_ths_constituent_stocks` 因为
      `mode='per_entity'` 躲过了 `mode in ("snapshot","dump")` 那条，成了"既不被跳过、
      又永远算不出指纹"的表。现在统一到这一个函数，三处共用。
    """
    if not ds.enabled:
        return False
    if ds.mode in ("snapshot", "dump"):
        return False
    if ds.freq == "snapshot":
        return False
    if not ds.ledger_date_field():   # 无日历轴（含 entity_range=False 的纯快照）
        return False
    return True


def _fingerprint(df: pd.DataFrame, ds: R.DS, date: str) -> dict | None:
    if df is None or len(df) == 0:
        return None
    fp = M.frame_fingerprint(df, keys=ds.keys)
    return {
        "date": str(date)[:10],
        "md5": fp["all_md5"],
        "rows": fp["rows"],
        "cols_md5": M.rows_to_json(fp.get("cols_md5") or {}),
        # ★ 2026-09-15 深夜补：`frame_fingerprint` **早就算出了 keys_md5**，这里却把它丢了，
        #   于是"主键是否漂移"只能靠"所有天的 md5 都变了"反推，无法直接比对。
        #   存下来之后 `hash-audit --deep` 才能一句话回答"这次重建是不是因为改了主键"。
        "keys_md5": fp.get("keys_md5") or "",
        "keys": ",".join(ds.keys),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def compute_day(dataset: str, ds: R.DS, date: str) -> dict | None:
    """算某一天的逻辑指纹。当天没有数据返回 None。"""
    return _fingerprint(_read_day(dataset, ds, date), ds, date)


def update_and_compare(dataset: str, ds: R.DS, dates: list[str],
                       write: bool = True) -> tuple[list[DayChange], int]:
    """对给定日期重算指纹并与台账比对。

    返回 (变化清单, 新增/更新条数)。变化 = 上游修改了历史数据。
    """
    ledger = load_ledger(dataset)
    old = {}
    if len(ledger):
        ledger["date"] = ledger["date"].astype(str).str[:10]
        for _, r in ledger.iterrows():
            old[r["date"]] = r

    frames = _read_days(dataset, ds, dates)      # ★ 一次扫描，切多天
    changes: list[DayChange] = []
    new_recs: list[dict] = []
    for d in dates:
        rec = _fingerprint(frames.get(str(d)[:10]), ds, d)
        if rec is None:
            # ★★ 2026-09-17：**本地整天数据消失**必须报出来。
            #   旧实现直接 `continue` —— 也就是说"本地某天的数据整个没了"
            #   在跑批报告里**完全不可见**（`report._vanished_notes` 管的是相反方向：
            #   本地有、服务端没有）。实测数据被误删/被半途覆盖时，
            #   这条链路是静默的，只有人工跑 `hash-audit --deep` 的 orphans 才看得到。
            prev = old.get(str(d)[:10])
            if prev is not None and int(prev["rows"] or 0) > 0:
                changes.append(DayChange(
                    dataset=dataset, date=str(d)[:10],
                    old_md5=str(prev["md5"]), new_md5="",
                    old_rows=int(prev["rows"] or 0), new_rows=0,
                    changed_cols=["(本地整天数据消失)"]))
            continue
        prev = old.get(rec["date"])
        if prev is not None:
            if str(prev["md5"]) != rec["md5"]:
                changed = _changed_cols(prev, rec)
                changes.append(DayChange(
                    dataset=dataset, date=rec["date"],
                    old_md5=str(prev["md5"]), new_md5=rec["md5"],
                    old_rows=int(prev["rows"] or 0), new_rows=rec["rows"],
                    changed_cols=changed))
        new_recs.append(rec)

    if write and new_recs:
        p = _ledger_path(dataset)
        merged = store.upsert(p, pd.DataFrame(new_recs), keys=("date",), sort_by=("date",))
        del merged
    return changes, len(new_recs)


# "台账早于跑批"只在这张表的最新台账日**距今足够近**时才判 —— 见 audit() 里的说明。
STALE_RECENT_DAYS = 12


def data_days(ds: R.DS, years: list[int] | None = None) -> set[str]:
    """扫出该表数据里出现过的日期（只读 `date_field` 一列，不载入整表）。

    ★★ 2026-09-15 深夜加 `years` 裁剪：**不传就扫全历史**，而 `stock_cyq_chips`
      （6.2 亿行）/ `stock_history_5min`（6.9 亿行）全历史扫描实测要十几分钟，
      让 `hash-audit --deep` 慢到不可用。调用方按台账日期范围算出该扫哪几个年份即可
      （大多数表的台账只有最近 7 天 → 只需扫当年那一个分区）。
    """
    out: set[str] = set()
    col = ds.ledger_date_field()          # ★ 台账键（默认=date_field）
    if not col:
        return out
    # ★ 候选分区走 _candidate_files：台账键 == 抓取键时按年裁剪（快路径，与改前一致）；
    #   两者不同则扫全部分区 —— 因为分区年份按**抓取键**切，与台账日期的年份可能不是一回事。
    all_years = store.list_years(ds.name)
    if col == ds.date_field:
        if not all_years:
            files = [store.flat_path(ds.name)]
        else:
            use = [y for y in all_years if years is None or int(y) in set(years)]
            files = [store.partition_path(ds.name, y) for y in use]
    else:
        files = _candidate_files(ds.name, ds, set(years or []))
    for f in files:
        if not f.exists():
            continue
        try:
            pf = pq.ParquetFile(f)
            if col not in pf.schema_arrow.names:
                continue
            for b in pf.iter_batches(batch_size=2_000_000, columns=[col]):
                out.update(b.column(0).to_pandas().astype(str).str[:10].unique().tolist())
        except (OSError, ValueError):
            continue
    return {d for d in out if d and d[0].isdigit()}


def audit(only: list[str] | None = None, deep: bool = False,
          sample: int = 10, log=print) -> dict:
    """单日 MD5 台账的**完整性自检**（只读）。

    ★★ 2026-09-15 深夜新增。回答"台账是否覆盖全、是否新鲜"—— 改造前没有任何工具能做：
      `day_expect verify` 只比"基线快照里已有的天"，`compare` 只比"旧快照里有的天"，
      所以**"这张表压根没有台账"是完全静默的**（`report._vanished_notes` 遇到空台账直接 return）。

    默认（秒级，只读台账 + manifest）：
      A 覆盖：该有台账的表里谁没有；有台账的天数与日期范围
      B 新鲜度：`台账 updated_at.max()` vs `manifest.last_run.finished_at`
        ★ 必须用 `finished_at` 而不是 `skipped_at` —— runner 明确"跳过不等于跑过"。
      C 主键：台账里存的 `keys` 与当前 `ds.keys` 是否一致（老台账没这列 → 标"未记录"）

    `deep=True`（要读数据，慢）：
      D 陈旧残留：台账里有、但当前数据里已经没有（或变成 0 行）的日期
        —— 这正是 `update_and_compare` 里 `continue` 永不删旧记录的盲区
      E 漏天：数据里有、台账里没有
      F 抽样复算：`update_and_compare(..., write=False)` 逐条比 md5
    """
    from ..core import state as state_mod

    results: list[dict] = []
    for ds in R.enabled():
        if only and ds.name not in only:
            continue
        row: dict = {"name": ds.name, "trackable": trackable(ds), "reason": "",
                     "ledger_days": 0, "ledger_min": None, "ledger_max": None,
                     "updated_at": None, "finished_at": None, "stale": False,
                     "keys_drift": None, "data_days": None,
                     "orphans": [], "missing": [], "mismatch": []}
        if not row["trackable"]:
            if ds.mode == "dump":
                row["reason"] = "dump 通道"
            elif not ds.ledger_date_field():
                row["reason"] = "无日历轴（无台账日期列）"
            elif ds.freq == "snapshot" or ds.mode == "snapshot":
                row["reason"] = "快照类"
            results.append(row)
            continue

        try:
            man = state_mod.Manifest.load(ds.name)
            row["finished_at"] = (man.last_run or {}).get("finished_at")
        except Exception:  # noqa: BLE001
            pass

        led = load_ledger(ds.name)
        if not len(led):
            row["reason"] = "⚠️ 无台账文件"
            results.append(row)
            continue
        dates = led["date"].astype(str).str[:10]
        row["ledger_days"] = len(dates)
        row["ledger_min"], row["ledger_max"] = dates.min(), dates.max()
        if "updated_at" in led.columns:
            row["updated_at"] = str(led["updated_at"].astype(str).max())[:19]
        # ★ 2026-09-16 修误报：**"台账早于跑批"只对"日期会落进尾部窗口"的表成立**。
        #   4 张季频财报（date_field=end_date）的日期是季末 03-31/06-30/…，
        #   而台账每轮只重算**尾部窗口（最近 6 个交易日）** —— 它们天然永远不会被刷新，
        #   于是刚补建完全历史台账就被误报成 "台账早于跑批"。
        #   判据：只有当台账最新那天距今足够近（说明这套机制**适用**）时才比时间戳。
        if row["updated_at"] and row["finished_at"] and row["ledger_max"]:
            recent_cut = (datetime.now() - timedelta(days=STALE_RECENT_DAYS)).strftime("%Y-%m-%d")
            if str(row["ledger_max"]) >= recent_cut:
                row["stale"] = row["updated_at"] < str(row["finished_at"])[:19]
        if "keys" in led.columns:
            got = {str(k) for k in led["keys"].dropna().astype(str) if str(k)}
            want = ",".join(ds.keys)
            if got and want not in got:
                row["keys_drift"] = f"台账 {sorted(got)[:2]} ≠ 当前 {want}"
        else:
            row["keys_drift"] = "未记录（旧台账）"

        if deep:
            # ★ 只扫台账涉及的年份分区（台账通常只有最近 7 天 → 只扫当年那一个分区）。
            #   不裁剪的话 stock_cyq_chips / stock_history_5min 的全历史扫描要十几分钟。
            led_years = sorted({int(d[:4]) for d in dates if str(d)[:4].isdigit()})
            dd = data_days(ds, years=led_years)
            row["data_days"] = len(dd)
            led_set = set(dates)
            # ★ 2026-09-16 修误报：**"漏天"只算台账自身范围之内的缺口**。
            #   范围之外（例如台账只从 09-07 开始，而数据从 2020 年就有）属于
            #   「台账不覆盖这段历史」—— 那是已知状态（只有 4 张表建了全历史台账），
            #   不是缺陷。原先直接把 `数据日 - 台账日` 当漏天，导致 25 张表被误报"漏天 20"。
            lo, hi = row["ledger_min"], row["ledger_max"]
            row["missing"] = sorted(d for d in dd if lo <= d <= hi and d not in led_set)[:20]
            row["orphans"] = sorted(led_set - dd)[:20]
            sample_days = (sorted(dd)[-3:]
                           + [d for d in sorted(dd)[::max(1, len(dd) // max(1, sample))]][:sample])
            sample_days = sorted(set(sample_days)) or sorted(led_set)[-sample:]
            try:
                changes, _ = update_and_compare(ds.name, ds, sample_days, write=False)
                row["mismatch"] = [c.as_dict() if hasattr(c, "as_dict") else str(c) for c in changes][:10]
            except Exception as exc:  # noqa: BLE001
                row["mismatch"] = [f"复算异常: {str(exc)[:70]}"]
        results.append(row)
    return {"tables": results, "deep": deep}


def _changed_cols(prev_row, rec: dict) -> list[str]:
    import json
    try:
        a = json.loads(str(prev_row["cols_md5"]) or "{}")
        b = json.loads(rec.get("cols_md5") or "{}")
    except (ValueError, TypeError):
        return []
    return [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def sample_old_days(ds: R.DS, n: int, exclude: set[str]) -> list[str]:
    """从**台账里已有的日期**中挑 n 天做「历史不变」抽查（尾窗之外、确定性轮换）。

    用户的核心诉求是「确保**不同时间下载的数据保持统一**、防止厂商偷偷改历史」，
    而每轮只重算尾部窗口（约 6 个交易日）—— 历史那部分**本来完全没有保护**，
    而 `conf/daily.yaml` 里的 `dayhash.sample_old_days` 是个**死配置**（全工程无人读）。

    ★ 只从台账已有的日期里挑：没有台账的日子算不出"变了没有"（只能记新基线，
      那是 `init-hashes` 的活）。所以台账越铺越开，这个抽查才越有意义 ——
      这也是"台账补历史"的另一个收益。

    抽样是**确定性**的（按"今天是本年第几天"错开）：同一天内可复现，
    隔天换一批，长期覆盖全部历史日期。
    """
    if n <= 0:
        return []
    led = load_ledger(ds.name)
    if not len(led):
        return []
    days = sorted({str(d)[:10] for d in led["date"]} - set(exclude))
    if not days:
        return []
    import datetime as _dt
    off = _dt.date.today().timetuple().tm_yday
    out: list[str] = []
    for i in range(min(n, len(days))):
        out.append(days[(off * 7 + i * 31) % len(days)])
    # 去重但保持顺序（表小的时候上面的步长会撞到一起）
    seen: set[str] = set()
    return [d for d in out if not (d in seen or seen.add(d))]


def forget_day(dataset: str, date: str) -> int:
    """从台账里删掉某一天（`delete-day` 删本地数据时同步调用）。

    为什么要删：`delete-day` 删了数据却留着台账那一行 → `hash-audit --deep` 每轮
    报"陈旧残留"，而提示的修法 `init-hashes` **不会删任何台账行**（它只 upsert
    现存数据的日期）→ 这个告警**修不掉**。删掉之后，重抓那轮会把新数据记成新基线。
    （回测不依赖台账：`backtest verify` 比的是 `baseline_*.json` 里的指纹。）
    """
    led = load_ledger(dataset)
    if not len(led) or "date" not in led.columns:
        return 0
    keep = led[led["date"].astype(str).str[:10] != str(date)[:10]]
    n = len(led) - len(keep)
    if n:
        store._atomic_write(keep.reset_index(drop=True), _ledger_path(dataset))
    return n


def init_hashes(only: list[str] | None = None, log=print) -> dict:
    """一次性给全部历史补台账。

    ★★ 2026-09-17 改**分批**。`_read_days` 会把"目标日期的所有行"一次性载入内存，
       而这里传的是该表**全部历史** → 每个分区所有行都会命中 → 整表常驻内存。
       实测标定：947,366 行 → 峰值 +0.63 GB（≈700 B/行）；按 parquet 元数据外推
       `stock_cyq_chips`（6.25 亿行）≈400 GB、`stock_history_5min`（6.87 亿行）≈440 GB，
       远超容器上限 120 GiB → **必被 OOM killer 杀掉**，而 `hash-audit` 恰好把
       `init-hashes --only <表>` 推荐成"陈旧残留/漏天"的标准修法（照做就 OOM）。
       现在按 40 天一批，峰值内存与"一天的数据"同量级。
    """
    stats: dict[str, dict] = {}
    for ds in R.enabled():
        if only and ds.name not in only:
            continue
        if not trackable(ds):          # ★ 判据统一（原先这里与 runner 各写一遍，已漂移过）
            continue
        years = store.list_years(ds.name)
        files = [store.partition_path(ds.name, y) for y in years] if years else [store.flat_path(ds.name)]
        files = [f for f in files if f.exists()]
        if not files:
            stats[ds.name] = {"days": 0, "note": "无数据"}
            continue
        # 逐分区取日期集合，再逐日算指纹（分区块读，避免整表载入）
        days: set[str] = set()
        _col = ds.ledger_date_field()          # ★ 台账键（默认=date_field）
        for f in files:
            try:
                pf = pq.ParquetFile(f)
                if _col not in pf.schema_arrow.names:
                    continue
                for b in pf.iter_batches(batch_size=2_000_000, columns=[_col]):
                    days.update(b.column(0).to_pandas().astype(str).str[:10].unique().tolist())
            except (OSError, ValueError):
                continue
        days = sorted(d for d in days if d and d[0].isdigit())
        if not days:
            stats[ds.name] = {"days": 0, "note": "分区里没有日期列"}
            continue
        # ★ 分批（见 docstring）：40 天一批，避免整表常驻内存 → OOM
        BATCH = 40
        changes: list[DayChange] = []
        n = 0
        for i in range(0, len(days), BATCH):
            ch, m = update_and_compare(ds.name, ds, days[i:i + BATCH], write=True)
            changes += ch
            n += m
            if (i // BATCH) % 10 == 9:
                log(f"      … {min(i + BATCH, len(days))}/{len(days)} 天")
        stats[ds.name] = {"days": n, "changes": len(changes)}
        log(f"   {ds.name:34} {n:>5} 天台账已建"
            + (f"  🔴 {len(changes)} 处与既有台账不符" if changes else ""))
    return stats
