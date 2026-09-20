"""删除某个日期（或各数据集"自己的最新一天"）的数据 —— 回测与演练用。

用途有二：
  1. **日常演练**：`python main.py delete-day --date 2026-09-14` 后再跑增量，
     看"补回这一天要多久、要多少请求"。
  2. **破坏性回测**（`backtest_increment.py` 的核心步骤）：删掉各数据集**最新一天**，
     跑增量，再用 MD5 逐日比对 —— 验证"删了能原样补回"。

★ 关键设计（决定回测能不能发现 bug）：
   · **故意不动 `coverage` / `done` / `suspect`** —— 只重算 `partitions` 的行数与日期范围。
     这正是被测对象：如果增量逻辑只看 coverage 而不看实际数据，那它就不会去补，
     测试就会红。旧工程 `index_daily` 缺 159 个交易日却以为已覆盖，
     就是这个机制潜伏至今的（QUATN_PLATFORM.md §16.3 问题 1）。
   · **只写日期所属年份的那个分区** —— 旧脚本第一版把备份合并进了所有年分区，
     导致某天的 1071 行被复制到 17 个年文件（自伤，已修）。
   · 原子替换（写 .tmp 再 os.replace），中断不会留半个文件。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pyarrow.parquet as pq

from .. import paths, registry as R
from ..core import state, store
from ..tools import md5 as M


@dataclass
class Deleted:
    dataset: str
    date: str
    rows: int
    backup: str
    day_md5: str


def _partition_files(ds: R.DS) -> list:
    years = store.list_years(ds.name)
    if years:
        return [store.partition_path(ds.name, y) for y in years]
    return [store.flat_path(ds.name)]


def find_latest_day(ds: R.DS) -> tuple[str | None, int]:
    """该数据集**自己**的最新一天及当天行数（不是全局最新交易日）。"""
    if not ds.date_field:
        return None, 0
    mx, cnt = None, 0
    for f in _partition_files(ds):
        if not f.exists():
            continue
        try:
            pf = pq.ParquetFile(f)
            if ds.date_field not in pf.schema_arrow.names:
                continue
            col = []
            for b in pf.iter_batches(batch_size=2_000_000, columns=[ds.date_field]):
                col.append(b.column(0).to_pandas().astype(str).str[:10])
            if not col:
                continue
            s = pd.concat(col, ignore_index=True)
            fmax = s.max()
            if mx is None or fmax > mx:
                mx = fmax
        except (OSError, ValueError):
            continue
    if mx is None:
        return None, 0
    # 数一下这一天的行数
    for f in _partition_files(ds):
        if not f.exists():
            continue
        try:
            pf = pq.ParquetFile(f)
            if ds.date_field not in pf.schema_arrow.names:
                continue
            for b in pf.iter_batches(batch_size=2_000_000, columns=[ds.date_field]):
                s = b.column(0).to_pandas().astype(str).str[:10]
                cnt += int((s == mx).sum())
        except (OSError, ValueError):
            continue
    return mx, cnt


def latest_day_fast(ds: R.DS, man: state.Manifest | None = None) -> str | None:
    """取该数据集**实际存在的**最新一天 —— 用 manifest 当提示 + **实测校验**。

    ⚠️ 为什么不直接用 `find_latest_day`：它为了"绝对准确"会把**每一个年分区**的
       日期列都读一遍。对 `stock_history_5min` 就是 17 个分区 / 9 GB，实测
       **9 分钟连一个数据集都没跑完**（按 registry 顺序，第 14 个才轮到小表）。
       而 `delete-day`（不给 --date）走的正是这条路径 —— 也就是说
       "删各表最新一天"这个官方演练流程，光"找最新一天"就要先把全库扫一遍。

    ★ 这里是**提示 + 校验**，不是"信标记"：
       manifest 的 max_date 是每次落盘时由实际数据 `s.max()` 重算的，可作提示；
       再调 `compute_day` **实测**确认那天真有数据。校验不过（提示说谎、
       或那天已被删空）才退回慢扫描。
    """
    from ..pipeline import dayhash as DH      # 延迟导入，避免 tools↔pipeline 环

    man = man or state.Manifest.load(ds.name)
    hint = man.max_partition_date()
    if hint and DH.compute_day(ds.name, ds, hint) is not None:
        return hint
    d, _n = find_latest_day(ds)
    return d


def delete_day(ds: R.DS, date: str, *, backup: bool = True,
               dry_run: bool = False, trim_coverage: bool = True) -> Deleted | None:
    """删掉 `date` 这一天的所有行。返回被删信息（没删到返回 None）。

    `dry_run=True` 时**零副作用**：只算出"会删多少行"并返回，不写备份、不改任何文件。
    （留给 `delete_all` 做"先预演、判占比、再真删"——见那里的注释。）

    `trim_coverage=True`（默认）时，把 manifest 的 coverage 从 `date` 起裁掉，
    让 `run_range._missing_ranges` 能重新把这天当缺口 → **下一轮会自动补回**。
    这是"人工修复某一天"的正确行为。回测若想检验"不看标记也能自愈"，
    显式传 `trim_coverage=False` 保留假标记。
    """
    if not ds.date_field:
        return None
    target = str(date)[:10]
    year = int(target[:4])
    f = store.partition_path(ds.name, year)
    if not f.exists():
        f = store.flat_path(ds.name)
        if not f.exists():
            return None
    df = store.read_parquet(f, on_error="empty")
    if df is None or len(df) == 0 or ds.date_field not in df.columns:
        return None
    s = df[ds.date_field].astype(str).str[:10]
    mask = s == target
    n = int(mask.sum())
    if n == 0:
        return None

    gone = df[mask]
    day_md5 = M.frame_fingerprint(gone, keys=ds.keys)["all_md5"]

    # ★★ dry-run 必须在写备份**之前**返回。
    #    旧实现把备份写在前面，于是 `--dry-run` 会真的写盘；而且备份文件名是固定的
    #    `deleted_<表>__<日期>.parquet`，第二轮 dry-run 会把上一轮的备份**覆盖掉** ——
    #    一个被声明为"只打印不写"的开关，同时破坏数据安全网。
    if dry_run:
        return Deleted(ds.name, target, n, "", day_md5)

    bak_path = ""
    if backup:
        paths.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        bf = paths.BACKTEST_DIR / f"deleted_{ds.name}__{target}.parquet"
        store._atomic_write(gone.reset_index(drop=True), bf)
        bak_path = str(bf)

    keep = df[~mask].reset_index(drop=True)
    store._atomic_write(keep, f)

    # ★ 先落数据（上面已写），再同步 manifest 的 partitions
    man = state.Manifest.load(ds.name)
    try:
        ser = keep[ds.date_field].astype(str).str[:10]
        man.mark_partition(0 if ds.partition == "none" else year, len(keep),
                           ser.min() if len(keep) else None, ser.max() if len(keep) else None)
    except (TypeError, ValueError):
        man.mark_partition(0 if ds.partition == "none" else year, len(keep), None, None)
    if trim_coverage:
        # ★ coverage 是"这段没有缺口"的唯一真相。删了数据却留着 coverage，
        #   `_missing_ranges` 就会一直认为这段是好的 → 这天永远补不回
        #   （只有恰好落在尾部窗口 [T-5,T] 内才有救）。这里把它裁掉，让缺口可见。
        man.trim_coverage_from(target)
    man.save()
    # ★ 2026-09-17：同步销掉台账里这一天的记录。
    #   不销的话，`hash-audit --deep` 每轮报"陈旧残留"，而它推荐的修法
    #   `init-hashes` **不会删任何台账行**（只 upsert 现存数据的日期）→ 修不掉。
    #   下一轮重抓后会把新数据记成新基线（回测不受影响：`backtest verify`
    #   比的是 `baseline_*.json` 里的指纹，不是台账）。
    try:
        from ..pipeline import dayhash as DH
        DH.forget_day(ds.name, target)
    except Exception:  # noqa: BLE001 —— 台账销账失败不该让删除本身失败
        pass
    return Deleted(ds.name, target, n, bak_path, day_md5)


def delete_all(date: str | None = None, *, only: set[str] | None = None,
               skip: set[str] | None = None, dry_run: bool = False,
               max_ratio: float = 0.30, trim_coverage: bool = True,
               fast: bool = True, on_each=None, log=print) -> list[Deleted]:
    """对全部数据集删除指定日期；`date=None` 时删各自的最新一天。

    `max_ratio`：如果某天行数占该数据集的比例超过它，跳过（防误删整张小表）。
    ★ 这个判据必须在**真正删除之前**算出来（见下面的预演注释）。

    `trim_coverage=True`（默认）：同时把 manifest 的 coverage 从该日期裁掉，
    让下一轮增量把这一天重新当缺口补回来（人工修复某一天的正常路径）。
    `fast=True`：用 manifest 的 `partitions.max_date` 当"最新一天"的提示，
    省掉全分区扫描（见 `backtest_increment.latest_day_fast` 的说明）。
    """
    skip = set(skip or set()) | {"stock_daily_dump", "stock_dump_5min"}
    out: list[Deleted] = []
    for ds in R.enabled():
        if ds.name in skip or ds.mode == "dump":
            continue
        if only and ds.name not in only:
            continue
        if ds.mode == "snapshot" or not ds.date_field:
            continue
        d = date
        if d is None:
            d = latest_day_fast(ds) if fast else find_latest_day(ds)[0]
            if d is None:
                continue
        man = state.Manifest.load(ds.name)
        total = man.partition_rows() or 1

        # ★★ 先预演（零副作用）拿到行数 → 判占比 → 再真删。
        #    旧实现是"先删完再判"：超限时只打一行「跳过（防误删）」，**数据其实已经删了**，
        #    而且该表不进 `gone` → `remember_deleted` 不记它 → 回测比对时走
        #    "未参与删除" 分支，把真正的「删了没补回」伪装成「本来就没删」。
        #    一个防误删的开关，最后成了掩盖 bug 的开关。
        peek = delete_day(ds, d, backup=False, dry_run=True)
        if peek is None:
            continue
        if peek.rows / total > max_ratio:
            log(f"   ⊘ {ds.name:34} {d} 占该表 {peek.rows/total:.0%} > "
                f"{max_ratio:.0%}，跳过（防误删，**未删除任何数据**）")
            continue

        got = delete_day(ds, d, dry_run=dry_run, trim_coverage=trim_coverage)
        if got is None:
            continue
        out.append(got)
        # ★ 2026-09-17：每删完一张就回调一次 —— 删除台账要**当场落盘**。
        #   旧实现是全部删完（实测约 2 小时）之后才写一次，中途任何中断都会让
        #   `last_deleted.json` 缺失 → 回测把每一张都判成"未参与删除" → bad=0
        #   → **一次真的删遍了全平台的演练报告"通过"**（向不安全方向失败）。
        if on_each is not None and not dry_run:
            try:
                on_each(got)
            except Exception as exc:  # noqa: BLE001
                log(f"   ⚠️ 删除台账写入失败（数据已删）：{str(exc)[:80]}")
        log(f"   ✔ {ds.name:34} 删 {d} 的 {got.rows:>8,} 行"
            + ("  [dry-run]" if dry_run else ""))
    return out
