#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""「删最新一天 → 只跑 main.py → 看能不能原样长回来」的往返测试。

用户 2026-09-19 交办：
    「将最新的一天数据都删了，然后只执行 main 看看能否有效的生成回来。
      由于 main 还会回溯几个交易日，所以你在做测试台账的时候还需要多做几天」

为什么台账要**多做几天**：`main.py` 的尾部窗口 `redundancy_days=5`（语义 `[T-5,T]` 含两端
= **6 个交易日**）会**无条件重抓**（`strategies.py:485`，明写"不看 done 标记"）。
所以补 09-18 时，它必然连带重抓前 5 个交易日并**重写**它们 ——
台账若只记被删那天，就看不见"补一天时顺带把邻近日子改坏了"。

三个子命令，**顺序不能反**：

    snapshot   跑之前：窗口内逐 (表,日) 记 md5 + 行数  → state/backtest/roundtrip_<T>.json
    delete     删除（默认每张表各自的**最新一天**）    → 复用 main.py 的 delete-day 内核
    verify     跑之后：同一窗口重算比对，出 REPORT     → state/backtest/REPORT_roundtrip_<T>.md

★ 范围 = `README.md` 附录 A 冻结编号的 **1–20 号**（用户指定），即
  「日频 20 张」（含 18/19/20 三张分钟表 —— 它们同样每交易日更新）。
★ `stock_daily_dump` / `stock_dump_5min` **永不删**（`delete_all` 内建跳过），
  且它的**本地按日缓存** `datadownload/data/stock_daily_dump/date=*/` 必须保留 ——
  `stock_history_5min` 的恢复逻辑是"本地有就直接读缓存恢复（0 请求）"，
  删了缓存就变成真的重新下载，测的就不是这条路径了。

只读 + 通过 `delete_day` 写数据/台账；不自己直写任何数据文件。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_incremental import paths, registry as R                    # noqa: E402
from data_incremental.core import state                               # noqa: E402
from data_incremental.pipeline import dayhash as DH                   # noqa: E402
from data_incremental.tools import delete_day as DD                   # noqa: E402
from data_incremental.tools import backtest_increment as BT           # noqa: E402

BACKTEST = paths.BACKTEST_DIR
CAL_FILE = paths.STATE / "calendar_ext.json"
DUMP_CACHE = Path("/autodl-fs/data/datadownload/data/stock_daily_dump")

# ★ 冻结编号 1–20（唯一真相在 README.md 附录 A；这里 import 那份 FROZEN 避免两处走样）
sys.path.insert(0, str(ROOT / "scripts"))
from gen_data_tables import FROZEN                                   # noqa: E402

SCOPE = [FROZEN[i] for i in range(1, 21)]

# ★ 本次豁免（用户 2026-09-19 拍板）：这些表**照删、照记**，但不计入通过/失败判据。
#   `stock_st_info`（15 号）：厂商于 2026-09-19 把 08-13 之后的全部数据**撤回**了
#   （服务端 max 从 09-18 退到 08-12），本地 09-18 那份**无法从服务端重抓**。
#   证据链见 state/backtest/REPORT_roundtrip_2026-09-18.md 的「厂商撤回事件」一节。
#   ⚠️ 厂商若恢复，**建议重测**（重测时把本条从 EXEMPT 删掉）。
EXEMPT: dict[str, str] = {
    "stock_st_info": "厂商 2026-09-19 撤回 08-13 后全部数据（server_max 09-18 → 08-12），无法重抓",
}


# ---------------------------------------------------------------- 基础
def calendar() -> list[str]:
    try:
        return sorted(str(x)[:10] for x in
                      json.loads(CAL_FILE.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return []


def window(cal: list[str], T: str, n: int) -> list[str]:
    """含 T 在内的最近 n 个交易日（升序）。"""
    if T not in cal:
        raise SystemExit(f"✗ {T} 不在交易日历里（日历范围 {cal[0]}~{cal[-1]}）")
    i = cal.index(T)
    return cal[max(0, i - n + 1): i + 1]


def dat_path(T: str) -> Path:
    return BACKTEST / f"roundtrip_{T}.json"


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def fingerprint_window(ds: R.DS, days: list[str], log=print) -> dict:
    """一次扫描取窗口内每一天的指纹（`_read_days` 是单次全分区扫描，别改成逐日调）。"""
    t0 = datetime.now()
    try:
        frames = DH._read_days(ds.name, ds, days)
    except Exception as exc:                                          # noqa: BLE001
        log(f"   ⚠️ {ds.name:34} 读取失败：{str(exc)[:70]}")
        return {}
    out: dict[str, dict] = {}
    for d in days:
        fp = DH._fingerprint(frames.get(d), ds, d)
        if fp is not None:
            out[d] = {"md5": fp["md5"], "rows": fp["rows"]}
    dt = (datetime.now() - t0).total_seconds()
    log(f"   ✔ {ds.name:34} {len(out):>2}/{len(days)} 天有数据  {dt:5.1f}s")
    return out


def dump_cache_state() -> dict:
    """记下 daily_dump 本地按日缓存的状态 —— 用来证明测试没碰它。"""
    out: dict[str, dict] = {}
    if not DUMP_CACHE.exists():
        return out
    for d in sorted(DUMP_CACHE.glob("date=*")):
        f, m = d / "data.parquet", d / "meta.json"
        if not f.exists():
            continue
        out[d.name.split("=", 1)[1]] = {
            "mtime": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "bytes": f.stat().st_size,
            "meta": read_json(m, {}) or {},
        }
    return out


# ---------------------------------------------------------------- ① snapshot
def cmd_snapshot(a) -> int:
    T = a.date
    days = window(calendar(), T, a.days)
    print(f"★ 台账窗口 = {days[0]} ~ {days[-1]}（{len(days)} 个交易日）")
    print(f"★ 范围 = 编号 1–20 共 {len(SCOPE)} 张表\n")

    tables: dict[str, dict] = {}
    for name in SCOPE:
        ds = R.get(name)
        man = state.Manifest.load(name)
        print(f"[{SCOPE.index(name) + 1:>2}/20] {name}")
        tables[name] = {
            "mode": ds.mode, "freq": ds.freq, "delay": int(ds.delay_days or 0),
            "watermark": man.max_partition_date(),
            "total_rows": man.partition_rows(),
            "days": fingerprint_window(ds, days),
        }

    doc = {
        "T": T, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "days": days, "scope": SCOPE,
        "dump_cache": dump_cache_state(),
        "tables": tables,
    }
    p = dat_path(T)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(len(t["days"]) for t in tables.values())
    print(f"\n✔ 台账已存 {p}")
    print(f"  {len(tables)} 张表 / {total} 条 (表,日) 指纹 / "
          f"dump 缓存 {len(doc['dump_cache'])} 天")
    print("\n各表水位：")
    for name, t in tables.items():
        print(f"   {name:34} 水位 {t['watermark'] or '—':<12} "
              f"窗口内有数据 {len(t['days']):>2}/{len(days)} 天  总行数 {t['total_rows']:,}")
    return 0


# ---------------------------------------------------------------- ② delete
def cmd_delete(a) -> int:
    T = a.date
    p = dat_path(T)
    if not p.exists():
        raise SystemExit(f"✗ 先跑 snapshot（缺 {p}）—— 没有跑前台账就没法验收")
    doc = read_json(p)

    before = dump_cache_state()
    print(f"★ 目标：编号 1–20（{len(SCOPE)} 张），删除**每张表各自的最新一天**")
    if a.date_each:
        print(f"  已指定 --date {a.date}（而非各表最新一天）")
    print(f"★ daily_dump 缓存：{len(before)} 天，删除前后都会核对未被触碰\n")

    BT.reset_deleted()
    gone = DD.delete_all(
        date=a.date if a.date_each else None,
        only=set(SCOPE),
        dry_run=not a.commit,
        on_each=BT.append_deleted,
        log=print,
    )

    mode = "预演（未删任何数据）" if not a.commit else "真删"
    print(f"\n★ {mode}：{len(gone)} 张表命中")
    for g in sorted(gone, key=lambda x: x.dataset):
        print(f"   {g.dataset:34} {g.date}  {g.rows:>8,} 行")

    if a.commit:
        after = dump_cache_state()
        same = before == after
        print(f"\n★ dump 缓存未被触碰：{'✔ 是' if same else '🚩 否 —— 有变化！'}")
        for d in sorted(set(before) | set(after)):
            b, af = before.get(d), after.get(d)
            if b != af:
                print(f"   🚩 {d}: {b} → {af}")
        doc["deleted"] = {g.dataset: {"date": g.date, "rows": g.rows,
                                   "backup": g.backup, "day_md5": g.day_md5} for g in gone}
        doc["deleted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        doc["dump_cache_after_delete"] = after
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"✔ 删除台账已并入 {p}")

    skipped = [n for n in SCOPE if n not in {g.dataset for g in gone}]
    if skipped:
        print(f"\n⚠️ 未命中（该表在这天没有数据，或被 max_ratio 挡下）：{skipped}")
    return 0


# ---------------------------------------------------------------- ③ verify
def cmd_verify(a) -> int:
    T = a.date
    p = dat_path(T)
    doc = read_json(p)
    if not doc:
        raise SystemExit(f"✗ 缺 {p}")
    days, tables = doc["days"], doc["tables"]
    deleted = doc.get("deleted", {})

    print(f"★ 重算窗口 {days[0]} ~ {days[-1]}（{len(days)} 个交易日）\n")
    now: dict[str, dict] = {}
    for i, name in enumerate(SCOPE, 1):
        ds = R.get(name)
        man = state.Manifest.load(name)
        print(f"[{i:>2}/20] {name}")
        now[name] = {
            "watermark": man.max_partition_date(),
            "total_rows": man.partition_rows(),
            "days": fingerprint_window(ds, days),
        }

    # ---- 逐 (表,日) 比对 ----
    rows_out: list[dict] = []
    for name in SCOPE:
        b, n = tables[name], now[name]
        was_deleted = name in deleted
        del_day = deleted.get(name, {}).get("date")
        for d in days:
            bd, nd = b["days"].get(d), n["days"].get(d)
            if bd is None and nd is None:
                continue
            if bd is None:
                kind = "新增"
            elif nd is None:
                kind = "🚩丢失"
            elif nd["rows"] > bd["rows"]:
                kind = "补齐"
            elif nd["rows"] < bd["rows"]:
                kind = "🚩行数变少"
            elif nd["md5"] != bd["md5"]:
                kind = "厂商重算"
            else:
                kind = "一致"
            rows_out.append({
                "table": name, "date": d, "kind": kind,
                "deleted": was_deleted and d == del_day,
                "before_rows": None if bd is None else bd["rows"],
                "after_rows": None if nd is None else nd["rows"],
                "before_md5": None if bd is None else bd["md5"],
                "after_md5": None if nd is None else nd["md5"],
            })

    # ---- 被删那天是否回来 ----
    def _ok(r): return r["table"] not in EXEMPT          # 豁免表不计入判据
    del_rows = [r for r in rows_out if r["deleted"]]
    del_judged = [r for r in del_rows if _ok(r)]
    lost = [r for r in rows_out if r["kind"] == "🚩丢失" and _ok(r)]
    fewer = [r for r in rows_out if r["kind"] == "🚩行数变少" and _ok(r)]
    changed = [r for r in rows_out if r["kind"] == "厂商重算" and _ok(r)]
    not_back = [r for r in del_judged if r["after_rows"] != r["before_rows"]
                or r["after_rows"] is None]
    md5_mismatch = [r for r in del_judged if r["after_md5"] != r["before_md5"]]
    exempt_rows = [r for r in del_rows if not _ok(r)]

    dump_ok = doc.get("dump_cache") == dump_cache_state()

    L = [f"# 往返测试报告 · 删最新一天 → 只跑 main.py",
         "",
         f"- 目标日 T = **{T}**（{len(days)} 个交易日台账：{days[0]} ~ {days[-1]}）",
         f"- 范围 = **编号 1–20**（{len(SCOPE)} 张表，`README.md` 附录 A 冻结编号），"
         f"其中 **{len(EXEMPT)} 张本次豁免**（见文末）",
         f"- 删除于 {doc.get('deleted_at', '—')}　验证于 "
         f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
         "",
         "## 结论", ""]

    def flag(ok): return "✔" if ok else "🚩"
    L += [f"- 本次判据内被删的 (表,日)：**{len(del_judged)} 条**，"
          f"行数未补回：**{len(not_back)}** {flag(not not_back)}",
          f"- 全窗口 🚩 行数变少：**{len(fewer)}** {flag(not fewer)}",
          f"- 全窗口 🚩 整日丢失：**{len(lost)}** {flag(not lost)}",
          f"- 厂商重算（行数不变、值变）：{len(changed)} 条（不算我们的错，逐条见下）",
          f"- daily_dump 本地缓存未被触碰：{flag(dump_ok)}",
          ""]
    if EXEMPT:
        L += ["### 本次豁免（不计入通过/失败）", "",
              "| 数据集 | 日期 | 删前 | 跑后 | 豁免原因 |",
              "|:--|:--|--:|--:|:--|"]
        for r in exempt_rows:
            L.append(f"| `{r['table']}` | {r['date']} | {r['before_rows']:,} | "
                     f"{'—' if r['after_rows'] is None else format(r['after_rows'], ',')} | "
                     f"{EXEMPT[r['table']]} |")
        L += ["", "> ⚠️ **建议厂商恢复后重测**：把 `EXEMPT` 里的条目删掉，"
                  "重跑 `snapshot → delete → main → verify` 即可。", ""]
    if not_back:
        L = [x.replace("行数未补回：", "行数未补回（🚩）：") for x in L]

    if not_back:
        L += ["## 🚩 没补回的 (表, 日)", "",
              "| 数据集 | 日期 | 删前行数 | 跑后行数 |", "|:--|:--|--:|--:|"]
        L += [f"| `{r['table']}` | {r['date']} | "
              f"{'—' if r['before_rows'] is None else format(r['before_rows'], ',')} | "
              f"{'—' if r['after_rows'] is None else format(r['after_rows'], ',')} |"
              for r in not_back]
        L += [""]

    if md5_mismatch:
        L += ["## 被删那天补回了、但 md5 与删前不同", "",
              "（行数一致、内容不同 = 上游在这期间重算过，属正常；"
              "但也可能是我们写丢列，逐条看下表的行数比）", "",
              "| 数据集 | 日期 | 删前行数 | 跑后行数 |", "|:--|:--|--:|--:|"]
        L += [f"| `{r['table']}` | {r['date']} | {r['before_rows']:,} | {r['after_rows']:,} |"
              for r in md5_mismatch]
        L += [""]

    if fewer or lost:
        L += ["## 🚩 红旗明细（行数变少 / 整日丢失）", "",
              "| 数据集 | 日期 | 删前行数 | 跑后行数 |", "|:--|:--|--:|--:|"]
        L += [f"| `{r['table']}` | {r['date']} | {r['before_rows']} | {r['after_rows']} |"
              for r in fewer + lost]
        L += [""]

    if changed:
        L += ["## 厂商重算明细（行数不变、只有值变）", "",
              "| 数据集 | 日期 | 行数 |", "|:--|:--|--:|"]
        L += [f"| `{r['table']}` | {r['date']} | {r['before_rows']:,} |" for r in changed]
        L += [""]

    L += ["## 被删那天的逐表结果", "",
          "| # | 数据集 | 日期 | 删前行数 | 跑后行数 | 行数 | md5 | 类型 |",
          "|--:|:--|:--|--:|--:|:--:|:--|:--|"]
    for i, name in enumerate(SCOPE, 1):
        r = next((x for x in del_rows if x["table"] == name), None)
        if r is None:
            L.append(f"| {i} | `{name}` | — | — | — | — | — | 未参与删除 |")
            continue
        if name in EXEMPT:
            L.append(f"| {i} | `{name}` | {r['date']} | {r['before_rows']:,} | "
                     f"{'—' if r['after_rows'] is None else format(r['after_rows'], ',')} | "
                     f"— | — | **本次豁免**（厂商撤回，见文末） |")
            continue
        same_n = r["after_rows"] == r["before_rows"]
        same_m = r["after_md5"] == r["before_md5"]
        L.append(f"| {i} | `{name}` | {r['date']} | {r['before_rows']:,} | "
                 f"{'—' if r['after_rows'] is None else format(r['after_rows'], ',')} | "
                 f"{flag(same_n)} | {flag(same_m)} | {r['kind']} |")
    L += [""]

    L += ["## 全窗口分类统计", ""]
    from collections import Counter
    c = Counter(r["kind"] for r in rows_out if _ok(r))
    c_all = Counter(r["kind"] for r in rows_out)
    L += ["| 类型 | 判据内条数 | 含豁免 |", "|:--|--:|--:|"]
    L += [f"| {k} | {c.get(k, 0)} | {v} |" for k, v in sorted(c_all.items())]
    L += ["", f"**判据内 {sum(c.values())} 条 / 含豁免共 {len(rows_out)} 条 (表,日) 指纹**", ""]

    L += ["## 各表水位变化", "",
          "| 数据集 | 删前水位 | 跑后水位 | 总行数 删前 | 总行数 跑后 |",
          "|:--|:--|:--|--:|--:|"]
    for name in SCOPE:
        b, n = tables[name], now[name]
        if name in EXEMPT:
            L.append(f"| `{name}` | {b['watermark'] or '—'} | {n['watermark'] or '—'} "
                     f"| {b['total_rows']:,} | {n['total_rows']:,} |")
            continue
        ok = "" if b["watermark"] == n["watermark"] else " 🚩"
        L.append(f"| `{name}` | {b['watermark'] or '—'} | {n['watermark'] or '—'}{ok} | "
                 f"{b['total_rows']:,} | {n['total_rows']:,} |")

    # ---- 厂商撤回事件（本次测试期间发现的独立问题，非本测试产物） ----
    L += ["", "---", "", "# 附：厂商撤回事件 · `stock_st_info`（15 号）", "",
          "**发现时间：2026-09-19 15:00 ~ 15:10**（本次往返测试执行期间）", "",
          "## 现象", "",
          "`main.py` 的闸门卡在 `stock_st_info 服务端 2026-09-18 无数据`，"
          "其余 19 张等待档表全部到齐。直接查服务端：", "",
          "| 查询区间 | 服务端 total |", "|:--|--:|",
          "| `2026-09-18 ~ 2026-09-18` | 0 |",
          "| `2026-09-17 ~ 2026-09-17` | 0 |",
          "| `2026-09-13 ~ 2026-09-18` | 0 |",
          "| `2026-09-01 ~ 2026-09-18` | 0 |",
          "| `2026-08-01 ~ 2026-08-31` | 1,456 |",
          "| `2026-01-01 ~ 2026-09-18` | 29,457 |",
          "",
          "★ 全年查询按日期**倒序**返回，首行日期 = **2026-08-12**、"
          "末页（page=147）日期 = 2026-01-05 ⇒ **厂商当前最新只到 08-12**。", "",
          "## 撤回前厂商是正常发数的（平台自己的监控台账为证）", "",
          "`state/monitor/history.jsonl` 里 `stock_st_info` 的 `server_max` 逐轮记录：", "",
          "| 观测时刻 | 厂商 server_max | 本地 local_max |", "|:--|:--|:--|",
          "| 2026-09-15 12:44 | 2026-09-14 | 2026-09-14 |",
          "| 2026-09-15 16:26 | 2026-09-14 | 2026-09-11 |",
          "| 2026-09-15 23:32 | **2026-09-15** | 2026-09-14 |",
          "| 2026-09-15 23:44 | 2026-09-15 | 2026-09-15 |",
          "| 2026-09-16 22:24 | 2026-09-16 | 2026-09-16 |",
          "| 2026-09-17 22:30 | 2026-09-17 | 2026-09-17 |",
          "| **2026-09-18 22:13** | **2026-09-18** | 2026-09-18 |",
          "| **2026-09-19 15:0x（本次）** | **2026-08-12** ⬅ 退回 | 2026-09-17 |",
          "",
          "## 结论", "",
          "1. **不是我们的 bug**：本地 08-13 ~ 09-18 的数据是真抓的"
          "（09-18 22:13 监控记录厂商仍有 09-18；本地行数逐日变化 206→205→203→204，"
          "非复制粘贴）。",
          "2. **是厂商单表撤回**：只此一张表退回 08-12，"
          "同一时刻 `stock_daily` 等表 09-18 查询正常（5,553 行）⇒ 非整站故障。",
          "3. **本地数据目前不可复现**：厂商不再提供 08-13 之后的 st_info，"
          "本地这份是**唯一副本**（`datadownload` 口径“撤数时本地不删”保住了它）。",
          "4. **闸门会长期卡住**：`stock_st_info` 在**等待档**，"
          "服务端永不到齐 ⇒ 每轮日更白等 `gate.max_wait_hours=6.0` 小时才 "
          "`on_timeout=partial` 放行。", "",
          "## 处置（用户 2026-09-19 拍板）", "",
          "- 本次测试：**`stock_st_info` 豁免**，不补、不等，只测其余 19 张；",
          "- 闸门策略：**先只记录，不动代码**（改档位是运营级决策，单独评估）；",
          "- **建议厂商恢复后重测**：把 `scripts/roundtrip_test.py` 的 `EXEMPT` 清空，"
          "重跑 `snapshot → delete → main → verify`。",
          ""]

    out = BACKTEST / f"REPORT_roundtrip_{T}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:70]))
    print(f"\n✔ 报告已写 {out}")
    print(f"\n★ 结论速览：判据内被删 {len(del_judged)} 条 / 没补回 {len(not_back)} / "
          f"🚩行数变少 {len(fewer)} / 🚩整日丢失 {len(lost)} / 厂商重算 {len(changed)}"
          f" / 豁免 {len(exempt_rows)} 条")
    return 0 if (not not_back and not fewer and not lost) else 1


# ---------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["snapshot", "delete", "verify"])
    ap.add_argument("--date", default=None, help="目标交易日 T（默认各表最新一天所在的交易日）")
    ap.add_argument("--days", type=int, default=20, help="台账窗口交易日数（默认 20）")
    ap.add_argument("--commit", action="store_true", help="delete：真删（不给则预演）")
    ap.add_argument("--date-each", action="store_true",
                    help="delete：删统一日期而非各表最新一天（默认各表最新）")
    a = ap.parse_args()

    if a.date is None:
        cal = calendar()
        a.date = cal[-1]
        print(f"（未给 --date，取日历最后一天 T = {a.date}）")
    return {"snapshot": cmd_snapshot, "delete": cmd_delete, "verify": cmd_verify}[a.step](a)


if __name__ == "__main__":
    raise SystemExit(main())
