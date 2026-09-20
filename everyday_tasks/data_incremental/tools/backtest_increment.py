"""破坏性回测 —— 证明（或证伪）增量更新"删了能补回"。

`main.py backtest` 之前是个空壳（只打印一句提示）。这个模块把它补上：
**基线 → 删除 → 重跑增量 → 逐日指纹比对**，四步都可单独执行、可反复执行。

★ 关键设计：**基线按"每个数据集自己的最新一天"取，而不是全表统一某一天。**
   理由（用户 2026-09-15 指出）：本工程有 `delay_days` 机制，一批表（margin_detail /
   dragon_tiger / top_list / st_info / ths_daily …）的最新数据本来就晚 1 个交易日。
   拿固定日期当靶子会**整个跳过这些表** —— 而它们恰恰是最容易出问题的一批。
   "删各表自己的最新一天"才等价于"模拟更新最新的数据"。

其它设计要点（每一条都是为了"测试真的能发现 bug"）：
  1. **指纹判据用 tools/md5 的逻辑指纹，不是文件字节** —— 与单日台账同一套算法，
     否则 row group 切分变化会造成假红。
  2. **基线要在删除之前算**，并且**落盘**（`state/backtest/baseline_<label>.json`）——
     否则删除后就没有"正确答案"可比了。
  3. **故意不动 coverage/done**（见 delete_day.py）—— 被测对象正是
     "增量逻辑到底看标记还是看实际数据"。只看标记的表会在这里露馅。
  4. 判定给出**三档次**，不要只给"成功/失败"：
        ✔ 完全一致   —— 行数与每列指纹都对得上
        ⚠️ 补回但不同 —— 行数对得上、内容不同（上游改了数据，或落盘口径漂移）
        ✘ 没补回     —— 仍然为空 / 行数明显不足
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from .. import paths, registry as R
from ..core import state
from ..pipeline import dayhash as DH
from . import delete_day as DD


# ================================================================ 基线
def baseline_path(label: str):
    return paths.BACKTEST_DIR / f"baseline_{label}.json"


def _count_on_axis(ds: R.DS, date: str, col: str | None = None) -> int:
    """按**指定的日期列**数某天的行数（只读这一列，零转换）。"""
    import pyarrow.parquet as pq
    from datetime import date as _d
    from datetime import timedelta
    from ..core import store as _store

    col = col or ds.date_field
    d0 = str(date)[:10]
    d1 = (_d.fromisoformat(d0) + timedelta(days=1)).isoformat()
    years = _store.list_years(ds.name)
    files = ([_store.partition_path(ds.name, y) for y in years] if years
             else [_store.flat_path(ds.name)])
    n = 0
    for f in files:
        if not f.exists():
            continue
        try:
            t = pq.read_table(f, columns=[col],
                              filters=[(col, ">=", d0), (col, "<", d1)])
            n += t.num_rows
        except (OSError, ValueError, KeyError):
            continue
    return n


def _entry(ds: R.DS, date: str) -> dict:
    """一个数据集在某天的指纹条目（无数据时 rows=0 / md5=None）。

    ★★ 2026-09-17：**台账轴 ≠ 抓取轴**的表（4 张季频财报：抓 `end_date`、台账 `ann_date`）
       不能按台账轴取指纹 —— 演练删的是"`end_date` = 那个季末"的行，而指纹算在
       `ann_date` 上，两边都算不到东西（基线 0 行 / 现在 0 行），于是回测把
       **一次成功的补回**误报成「✘ 没补回（仍为空）」（实测 4 张全中）。
       这类表改按**抓取轴行数**验收，并在条目里标明 `axis="fetch"`。
    """
    if ds.ledger_date_field() != ds.date_field:
        return {"date": str(date)[:10], "axis": "fetch",
                "rows": _count_on_axis(ds, date), "md5": None, "cols_md5": None}
    rec = DH.compute_day(ds.name, ds, date)
    if rec is None:
        return {"date": date, "rows": 0, "md5": None, "cols_md5": None}
    return {"date": rec["date"], "rows": rec["rows"], "md5": rec["md5"],
            "cols_md5": rec["cols_md5"]}


def take_baseline_fixed(date: str, only: set[str] | None = None, log=print) -> dict:
    """记录**固定日期** `date` 在所有数据集上的指纹。"""
    date = str(date)[:10]
    return _take("fixed_" + date, "fixed", [(ds, date) for ds in _targets(only)], log)


def take_baseline_latest(only: set[str] | None = None, log=print,
                         fast: bool = True) -> dict:
    """★ 记录**每个数据集自己最新一天**的指纹（推荐口径）。

    这是"模拟更新最新数据"的正确靶子：delay 表的最新一天会自己落到 09-11，
    而不是被统一按 09-14 处理（那样会直接跳过它们）。

    `fast=True` 走"manifest 提示 + 实测校验"（见 `DD.latest_day_fast`），
    几十秒完成；`fast=False` 走全分区扫描（准确但可能要几十分钟）。
    """
    pairs: list[tuple[R.DS, str]] = []
    for ds in _targets(only):
        d = DD.latest_day_fast(ds) if fast else DD.find_latest_day(ds)[0]
        if d:
            pairs.append((ds, d))
        else:
            log(f"   ⊘ {ds.name:34} 本地无数据，跳过")
    return _take("latest", "latest", pairs, log)


def _targets(only: set[str] | None) -> list[R.DS]:
    return [d for d in R.enabled()
            if (not only or d.name in only) and d.mode != "snapshot" and d.date_field]


def _take(label: str, mode: str, pairs: list[tuple[R.DS, str]], log) -> dict:
    out: dict = {"label": label, "mode": mode,
                 "taken_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "datasets": {}}
    for ds, date in pairs:
        e = _entry(ds, date)
        out["datasets"][ds.name] = e
        log(f"   基线 {ds.name:34} {e['date']}  {e['rows']:>10,} 行  "
            f"md5={(e['md5'] or '-')[:12]}")
    paths.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path(label).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    log(f"\n   基线已存：{baseline_path(label)}（{len(pairs)} 个数据集）")
    return out


def load_baseline(label: str) -> dict | None:
    p = baseline_path(label)
    if not p.exists():
        return None
    try:
        v = json.loads(p.read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


# ================================================================ 比对
@dataclass
class Row:
    dataset: str
    date: str
    base_rows: int
    now_rows: int
    md5_same: bool
    changed_cols: list[str] = field(default_factory=list)
    verdict: str = ""

    def as_dict(self) -> dict:
        return {"dataset": self.dataset, "date": self.date,
                "base_rows": self.base_rows, "now_rows": self.now_rows,
                "md5_same": self.md5_same, "changed_cols": self.changed_cols,
                "verdict": self.verdict}


def _changed_cols(base_cols, now_cols) -> list[str]:
    def _load(v):
        if isinstance(v, dict):
            return v
        try:
            return json.loads(v or "{}")
        except (ValueError, TypeError):
            return {}
    a, b = _load(base_cols), _load(now_cols)
    return [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def compare(baseline_label: str, only: set[str] | None = None,
            deleted: dict | None = None, log=print) -> list[Row]:
    """比对每个数据集**在它自己基线日期上**的当前数据 vs 基线。

    `deleted` = {数据集: 被删信息}，用来区分"本来就没参与删除"与"删了没补回"。
    """
    base = load_baseline(baseline_label)
    if base is None:
        raise SystemExit(f"没有基线 {baseline_label}，先跑 take_baseline_latest")
    deleted = deleted or {}
    rows: list[Row] = []

    for ds in R.enabled():
        if only and ds.name not in only:
            continue
        if ds.mode == "snapshot" or not ds.date_field:
            continue
        b = (base.get("datasets") or {}).get(ds.name)
        if b is None:
            continue
        date = b["date"]
        was_deleted = ds.name in deleted and deleted[ds.name].get("rows")
        ddate = str((deleted.get(ds.name) or {}).get("date") or "")[:10]
        br = int(b.get("rows") or 0)

        if b.get("axis") == "fetch":
            # ★ 台账轴 ≠ 抓取轴的表：按抓取轴行数验收（见 _entry 的注释）
            nr = _count_on_axis(ds, date) if was_deleted else 0
            if not was_deleted:
                verdict = "－ 未参与删除"
            elif ddate and ddate != str(date)[:10]:
                verdict = f"✘ 未验证（基线 {date} ≠ 被删 {ddate}）"
            elif nr == 0:
                verdict = "✘ 没补回（仍为空）"
            elif br and nr < br * 0.98:
                verdict = f"✘ 没补全（{nr / br:.0%}）"
            elif nr == br:
                verdict = f"✔ 行数一致（抓取轴口径，{nr:,} 行）"
            else:
                verdict = f"⚠️ 行数不同（{br:,} → {nr:,}）"
            rows.append(Row(ds.name, date, br, nr, nr == br, [], verdict))
            log(f"   {verdict:22} {ds.name:34} {date}  基线 {br:>10,} → 现在 {nr:>10,}")
            continue

        rec = DH.compute_day(ds.name, ds, date)
        nr = int(rec["rows"]) if rec else 0
        same = bool(rec and b.get("md5") and rec["md5"] == b["md5"])
        cc = [] if same else _changed_cols(b.get("cols_md5"),
                                           rec.get("cols_md5") if rec else None)

        if not was_deleted:
            verdict = "－ 未参与删除"
        elif ddate and ddate != str(date)[:10]:
            # ★★ 2026-09-17：**基线日期 ≠ 被删日期**时必须判「未验证」。
            #   旧实现只按**数据集名**判断"参与过删除"，然后拿**基线自己的日期**重算 ——
            #   若删除的是另一天（`backtest baseline` 与 `delete-day` 之间跑过一轮日常更新，
            #   或者用了 `--date`），它比的是**根本没被删过的那天**：当然与基线一致
            #   → 打印"✔ 完全一致"= "自愈成功"，而**实际什么都没验证**。
            #   这里判 ✘（进 bad、`main.py backtest verify` 退出码 1）。
            #   验收工具宁可吵闹，也不能把"没测到"说成"通过"。
            verdict = f"✘ 未验证（基线 {date} ≠ 被删 {ddate}）"
        elif same:
            verdict = "✔ 完全一致"
        elif nr == 0:
            verdict = "✘ 没补回（仍为空）"
        elif br and nr < br * 0.98:
            verdict = f"✘ 没补全（{nr / br:.0%}）"
        else:
            verdict = "⚠️ 补回但内容不同"

        rows.append(Row(ds.name, date, br, nr, same, cc, verdict))
        log(f"   {verdict:16} {ds.name:34} {date}  基线 {br:>10,} → 现在 {nr:>10,}"
            + (f"  变化列 {cc[:4]}" if cc else ""))

    return rows


def summary(rows: list[Row]) -> dict:
    g = lambda p: [r for r in rows if r.verdict.startswith(p)]  # noqa: E731
    ok, warn, bad, skip = g("✔"), g("⚠️"), g("✘"), g("－")
    return {"ok": len(ok), "warn": len(warn), "bad": len(bad), "skip": len(skip),
            "ok_names": [r.dataset for r in ok], "bad_names": [r.dataset for r in bad],
            "warn_names": [r.dataset for r in warn]}


def render(rows: list[Row]) -> str:
    s = summary(rows)
    out = ["=" * 100, "  增量回测结果", "=" * 100,
           f"  ✔ 完全补回 {s['ok']:>2}    ⚠️ 补回但内容不同 {s['warn']:>2}    "
           f"✘ 没补回 {s['bad']:>2}    － 未参与 {s['skip']:>2}"]
    for tag, names in (("✘ 没补回", s["bad_names"]), ("⚠️ 补回但不同", s["warn_names"])):
        if names:
            out.append(f"\n  {tag}：")
            for r in rows:
                if r.dataset in names:
                    out.append(f"     {r.dataset:34} {r.date}  基线 {r.base_rows:>10,} "
                               f"→ {r.now_rows:>10,}"
                               + (f"  变化列 {r.changed_cols}" if r.changed_cols else ""))
    return "\n".join(out)


# ================================================================ 删除台账
def remember_deleted(gone: list[DD.Deleted]) -> dict:
    """把 delete-day 的结果落盘（**原子写**），供 compare 区分"没删"和"删了没补回"。"""
    d = {g.dataset: {"date": g.date, "rows": g.rows, "backup": g.backup,
                     "day_md5": g.day_md5} for g in gone}
    _write_deleted(d)
    return d


def _write_deleted(d: dict) -> None:
    from ..core import state as _state
    paths.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    # ★★ 2026-09-17：改**原子写**（tmp + os.replace + chmod）。
    #   旧实现是裸 `write_text`：删完 40 张表（实测约 2 小时）之后才写一次，
    #   中途 Ctrl-C / OOM / 权限问题都会留下"数据已删、台账没记"的最坏组合；
    #   而以 root 跑出来的 0644 root:root 文件，claude 用户再写还会 PermissionError。
    _state._atomic_json(paths.BACKTEST_DIR / "last_deleted.json", d)


def reset_deleted() -> None:
    """清空删除台账（每轮演练开始时调用，保证台账只反映**本轮**删了什么）。"""
    _write_deleted({})


def append_deleted(g: "DD.Deleted") -> dict:
    """**每删完一张表就落一次盘** —— 中断也不会出现"删了没记账"。"""
    d = load_deleted()
    d[g.dataset] = {"date": g.date, "rows": g.rows, "backup": g.backup,
                    "day_md5": g.day_md5}
    _write_deleted(d)
    return d


def load_deleted() -> dict:
    p = paths.BACKTEST_DIR / "last_deleted.json"
    try:
        v = json.loads(p.read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}
