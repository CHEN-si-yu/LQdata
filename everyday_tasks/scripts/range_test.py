#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""25/26/27 号的**更新逻辑验证**（range / 特例两条路径）。

用户 2026-09-19 交办：「核对 25-27。先从逻辑分析判断相关的数据是否能正确更新，
然后测试服务器的数据，最后进行破坏性实验实现闭合」。

## 逻辑分析（先读这段）

| # | 表 | 取数方式 | 取数区间（实测 T=09-18） | 自愈深度 | coverage |
|--:|:--|:--|:--|--:|:--|
| 25 | `stock_pledge_stat` | `run_range`，按 **end_date** 过滤 | `[2025-08-14, 2026-09-18]`（revision 400 天） | **400 天** | ✅ 有 |
| 26 | `stock_forecast` | **特例分支** `ann_date ∈ [lo,hi]` | `2026-06-20 ~ 2026-09-18`（revision 90 天） | **90 天** | ❌ **无** |
| 27 | `stock_holder_number` | `run_range`，按 **end_date** 过滤 | `[2025-08-14, 2026-09-18]`（revision 400 天） | **400 天** | ✅ 有 |

★ `start_time/end_time` 实测**按 `end_date` 过滤**（查 `2025-09-01~09-30` 只返回 `end_date=2025-09-30` 的行，
  而它们的 `ann_date` 在 10 月）—— 所以"窗口"是 **end_date 的窗口**。

## 四组删除（一次 main 跑完一起验收）

    A  窗口内、最新期        25/27 删 2026-09-11 / 2026-09-17 那期
       → 期望：**补回**（range 区间覆盖它，与 coverage 无关）

    B  窗口外、**保留 coverage**   25/27 删 end_date < 2025-08-14 的老期，且不裁 coverage
       → 期望：**不补**。这一组测的是**真实暴露面** —— 厂商给老期补数据时，
         没有缺口标记 ⇒ `_missing_ranges` 看不见 ⇒ 永远抓不到。
         （对照：默认 `delete-day` 会裁 coverage，那是"人工修复"路径，能补 —— 见 C 组）

    C  窗口外、**裁掉 coverage**   同 B 的日期，但裁 coverage
       → 期望：**补回**（验证"人工修复某天"的路径可用）

    D  26 专属：90 天窗口外   删 `ann_date < 2026-06-20` 的行
       → 期望：**不补**（26 没有 coverage，也没有别的兜底）

用法（顺序不能反）：

    python scripts/range_test.py snapshot [--date 2026-09-18]
    python scripts/range_test.py delete --commit
    python scripts/range_test.py verify
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd                                                   # noqa: E402

from data_incremental import paths, registry as R                     # noqa: E402
from data_incremental.core import state, store                        # noqa: E402
from data_incremental.tools import md5 as M                           # noqa: E402
from gen_data_tables import FROZEN                                     # noqa: E402

BACKTEST = paths.BACKTEST_DIR
T25, T26, T27 = FROZEN[25], FROZEN[26], FROZEN[27]
WIN_START = "2025-08-14"          # T - 400 天（25/27 的 range 下界）
ANN_START = "2026-06-20"          # T - 90 天（26 的 ann_date 下界）
# ★ 26 的 D 组只删**一个具体的老公告日**，不删整片 90 天外数据（那是 10.8 万行 ≈ 整表，
#   为了证明"窗口外不补"没必要冒这个险）。2012-01-31 有 234 行，够说明问题。
D_DATE = "2012-01-31"

# ★ 目标日期必须是**真实存在的 end_date**（否则删 0 行，测试空转）
#   25：避开 `2024-05-10` —— 那期本来就有 690 行存量缺口（单独记录），会污染判据
# ★★ B 与 C **必须是不同的日期**：它们删的是两批不同的行，
#   若用同一天就分不清"B 没补"和"C 补了"——判据直接失效（第一版就踩了这个坑）。
TARGETS = {
    T25: {"A": "2026-09-11", "B": "2024-05-31", "C": "2024-09-30"},
    T27: {"A": "2026-06-30", "B": "2024-01-10", "C": "2024-06-30"},
}


def dat_path(T: str) -> Path:
    return BACKTEST / f"range_{T}.json"


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_all(name: str) -> pd.DataFrame:
    fs = sorted((paths.DATA_ROOT / name).glob("year=*/*.parquet"))
    if not fs:
        return pd.DataFrame()
    return pd.concat([store.read_parquet(f, on_error="empty") for f in fs],
                     ignore_index=True)


def write_all(name: str, df: pd.DataFrame, key: str = "end_date") -> None:
    for f in sorted((paths.DATA_ROOT / name).glob("year=*/*.parquet")):
        y = f.parent.name.split("=", 1)[1]
        sub = df[df[key].astype(str).str[:4] == str(y)]
        store._atomic_write(sub.reset_index(drop=True), f)


def seg_fp(df: pd.DataFrame, ds: R.DS, mask: pd.Series) -> dict:
    sub = df[mask]
    if not len(sub):
        return {"rows": 0, "md5": None}
    fp = M.frame_fingerprint(sub, keys=ds.keys)
    return {"rows": fp["rows"], "md5": fp["all_md5"]}


# ---------------------------------------------------------------- ① snapshot
def cmd_snapshot(a) -> int:
    doc = {"T": a.date, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "win_start": WIN_START, "ann_start": ANN_START, "tables": {}}
    for name in (T25, T26, T27):
        ds = R.get(name)
        df = load_all(name)
        key = "ann_date" if name == T26 else "end_date"
        v = df[key].astype(str).str[:10]
        tgt = TARGETS.get(name, {})
        segs = {}
        if name == T26:
            segs = {"A": seg_fp(df, ds, v >= ANN_START),
                    "D": seg_fp(df, ds, v == D_DATE)}
        else:
            segs = {"A": seg_fp(df, ds, v == tgt["A"]),
                    "B": seg_fp(df, ds, v == tgt["B"]),
                    "C": seg_fp(df, ds, v == tgt["C"])}
        man = state.Manifest.load(name)
        doc["tables"][name] = {
            "key": key, "total_rows": int(len(df)), "freq": ds.freq, "mode": ds.mode,
            "revision": int(ds.revision_days or 0), "coverage_segs": len(man.coverage or []),
            "watermark": man.max_partition_date(), "segs": segs,
        }
        print(f"[{name}] 总行数 {len(df):,} | 水位 {man.max_partition_date()} "
              f"| coverage 段 {len(man.coverage or [])}")
        for k, fp in segs.items():
            print(f"      {k:>10}: {fp['rows']:>8,} 行  md5={(fp['md5'] or '—')[:10]}")
    p = dat_path(a.date)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✔ 台账已存 {p}")
    return 0


# ---------------------------------------------------------------- ② delete
def cmd_delete(a) -> int:
    doc = read_json(dat_path(a.date))
    if not doc:
        raise SystemExit("✗ 先跑 snapshot")
    print(f"★ {'真删' if a.commit else '预演'}\n")
    results = []
    for name in (T25, T26, T27):
        ds = R.get(name)
        df = load_all(name)
        key = doc["tables"][name]["key"]
        v = df[key].astype(str).str[:10]
        tgt = TARGETS.get(name, {})

        if name == T26:
            # A：90 天内（应补回）；D：90 天外（不补）
            plan = [("A", v >= ANN_START), ("D", v == D_DATE)]
        else:
            plan = [("A", v == tgt["A"]), ("B", v == tgt["B"]), ("C", v == tgt["C"])]

        print(f"[{name}]  过滤键 = {key}")
        for grp, mask in plan:
            hit = df[mask]
            rec = {"table": name, "grp": grp, "rows": int(len(hit)),
                   "desc": (f"{key} == {tgt.get(grp)}" if name != T26
                            else (f"{key} >= {ANN_START}" if grp == "A"
                                  else f"{key} == {D_DATE}")),
                   "trim": grp in ("A", "C"), "backup": ""}
            if a.commit and len(hit):
                BACKTEST.mkdir(parents=True, exist_ok=True)
                bf = BACKTEST / f"range_deleted_{name}__{grp}.parquet"
                store._atomic_write(hit.reset_index(drop=True), bf)
                rec["backup"] = str(bf)
            results.append(rec)
            print(f"      {grp} 删 {len(hit):>8,} 行  （{rec['desc']}）"
                  f"{'  +裁 coverage' if rec['trim'] else '  保留 coverage'}")

        if a.commit:
            # 三/两组都要**真的删掉**（区别只在 coverage 标不标）
            keep = df
            for _grp, _mask in plan:
                keep = keep[~_mask]
            write_all(name, keep.reset_index(drop=True), key)
            # ★ coverage 处理：A/C 裁掉、B/D 保留（B/D 测的正是"没有缺口标记"的真实暴露）
            #   ⚠️ 不能直接取 `TARGETS[name][grp]` —— 26 号不在 TARGETS 里（它按键是 ann_date）
            man = state.Manifest.load(name)
            for grp, _m in plan:
                if grp not in ("A", "C"):
                    continue
                t = (TARGETS.get(name, {}).get(grp)
                     or (ANN_START if grp == "A" else D_DATE))
                try:
                    man.trim_coverage_from(t)
                except Exception as exc:      # noqa: BLE001
                    print(f"      ⚠️ trim_coverage({t}) 失败：{str(exc)[:60]}")
            man.save()
        print()

    if a.commit:
        doc["deleted"] = results
        doc["deleted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dat_path(a.date).write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    return 0


# ---------------------------------------------------------------- ③ verify
def cmd_verify(a) -> int:
    doc = read_json(dat_path(a.date))
    if not doc:
        raise SystemExit("✗ 缺台账")
    L = ["# 25/26/27 更新逻辑验证报告", "",
         f"- 验证于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
         f"（删除于 {doc.get('deleted_at', '—')}）",
         f"- 25/27 `run_range`（按 end_date，revision 400 天，**有 coverage**）；"
         f"26 特例 ann_date 区间（90 天，**无 coverage**）", "",
         "| 组 | 删什么 | 期望 |", "|:--|:--|:--|",
         f"| A | 窗口内最新期 | 补回 |",
         f"| B | 窗口外 `end_date == 2024-06-28`，**保留 coverage** | **不补**（真实暴露面） |",
         f"| C | 同上但**裁掉 coverage** | 补回（人工修复路径） |",
         f"| D | 26 的 `ann_date == {D_DATE}`（90 天窗口外） | 不补（无 coverage） |", ""]

    rows = []
    for name in (T25, T26, T27):
        ds = R.get(name)
        df = load_all(name)
        key = doc["tables"][name]["key"]
        v = df[key].astype(str).str[:10]
        tgt = TARGETS.get(name, {})
        for grp in [x["grp"] for x in doc.get("deleted", []) if x["table"] == name] or ["A"]:
            if name == T26:
                mask = (v >= ANN_START) if grp == "A" else (v == D_DATE)
            else:
                mask = (v == tgt.get(grp))
            now = seg_fp(df, ds, mask)
            before = doc["tables"][name]["segs"].get(grp, {})
            rows.append({"table": name, "grp": grp, "key": key,
                         "before": before.get("rows", 0), "after": now["rows"],
                         "b_md5": before.get("md5"), "a_md5": now["md5"]})

    for grp, label in (("A", "A 组 · 窗口内"), ("B", "B 组 · 窗口外 + 保留 coverage"),
                       ("C", "C 组 · 窗口外 + 裁 coverage"), ("D", "D 组 · 26 窗口外")):
        sel = [x for x in rows if x["grp"] == grp]
        if not sel:
            continue
        L += [f"## {label}", "",
              "| 数据集 | 过滤 | 删前 | 跑后 | 判定 |", "|:--|:--|--:|--:|:--|"]
        for r in sel:
            if r["before"] == 0 and r["after"] == 0:
                j = "—（无此类行）"
            elif grp in ("B", "D"):
                j = "✔ 如预期未补" if r["after"] < r["before"] else "🚩 意外补回了"
            else:
                j = "✔ 补回" if r["after"] == r["before"] else f"🚩 未补回（{r['before']}→{r['after']}）"
            L.append(f"| `{r['table']}` | {r['key']} | {r['before']:,} | {r['after']:,} | {j} |")
        L += [""]

    out = BACKTEST / f"REPORT_range_{a.date}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n✔ 报告已写 {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["snapshot", "delete", "verify"])
    ap.add_argument("--date", default="2026-09-18")
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()
    return {"snapshot": cmd_snapshot, "delete": cmd_delete, "verify": cmd_verify}[a.step](a)


if __name__ == "__main__":
    raise SystemExit(main())
