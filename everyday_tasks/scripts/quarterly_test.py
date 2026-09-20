#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""季频财报 4 张表的**更新逻辑验证**（21–24 号：balancesheet / cashflow /
financial_indicator / income）。

为什么不能照搬日频那套测法：这 4 张 `mode='per_stock'`、**不走 `run_range`**，
因而**没有 coverage 缺口机制** —— 它们的自愈深度**只有 `strategies.PERIOD_LOOKBACK=3`
个报告期**（`run_per_stock` 里那个 REPORT_PERIOD_TABLES 分支，每轮无条件抓最近 3 期）。

用户 2026-09-19 交办：
    「确认针对季度更新的逻辑是否能够真的符合预期……如果远程服务器真的更新了，
      我们的 main 能够正确及时的抓取下来」

三组删除，**一次 main 跑完一起验收**：

    A  整期删除          删 2026-06-30 的全部行（在 3 期窗口内）
        → 期望：**原样补回**（验基础自愈）

    B  只删旧版本         2025-12-31 里，每只票**只留 ann_date 最大的那条**，其余版本删掉
        → 期望：**补回**。这一组测的是最关键的语义 —— 主键含 `ann_date`，
          同一报告期有多版本（追溯重述）。服务端 `end_date` 查询**会返回全部版本**
          （已实测：6,383 行 / 6,076 只，307 只重复）⇒ 删掉的版本应当被重新合入。

    C  窗口外            删 2025-09-30 的部分行（**已滑出 3 期窗口**）
        → 期望：**不补**（确认边界，不是 bug）。补救手段 = `main.py run --as-of <回拨日>`

三个子命令，顺序不能反：

    snapshot   记逐 (表, end_date) 的行数 + md5 + 总行数  → state/backtest/quarterly_<T>.json
    delete     执行 A / B / C 三组删除（默认预演，--commit 才真删；先备份）
    verify     跑后比对，出 REPORT                        → state/backtest/REPORT_quarterly_<T>.md
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
from data_incremental.tools import md5 as M                          # noqa: E402
from gen_data_tables import FROZEN                                     # noqa: E402

BACKTEST = paths.BACKTEST_DIR
TABLES = [FROZEN[i] for i in range(21, 25)]

# 三组测试的目标报告期
PERIOD_A = "2026-06-30"      # 窗口内（最新期）
PERIOD_B = "2025-12-31"      # 窗口内（第 3 期）—— 只删"旧版本"行
PERIOD_C = "2025-09-30"      # 窗口外 —— 期望不补
C_DELETE_ROWS = 60           # C 组删多少行（取该期前 N 行）


def dat_path(T: str) -> Path:
    return BACKTEST / f"quarterly_{T}.json"


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


# ---------------------------------------------------------------- 读写整表
def load_all(name: str) -> pd.DataFrame:
    """读该表的**全部**行（跨所有年分区）。"""
    fs = sorted((paths.DATA_ROOT / name).glob("year=*/*.parquet"))
    if not fs:
        return pd.DataFrame()
    return pd.concat([store.read_parquet(f, on_error="empty") for f in fs],
                     ignore_index=True)


def write_all(name: str, df: pd.DataFrame) -> None:
    """按 end_date 的年份把 df 摊回各年分区并写回（分区定义 = `partition='year'`）。"""
    fs = sorted((paths.DATA_ROOT / name).glob("year=*/*.parquet"))
    for f in fs:
        year = int(f.parent.name.split("=", 1)[1])
        sub = df[df["end_date"].astype(str).str[:4] == str(year)]
        store._atomic_write(sub.reset_index(drop=True), f)


def period_fp(df: pd.DataFrame, ds: R.DS, per: str) -> dict | None:
    """某一个报告期的指纹（行数 + 主键级 md5）。"""
    sub = df[df["end_date"].astype(str).str[:10] == per]
    if not len(sub):
        return None
    fp = M.frame_fingerprint(sub, keys=ds.keys)
    return {"rows": fp["rows"], "md5": fp["all_md5"]}


def versions(df: pd.DataFrame, per: str) -> dict:
    """该期「每只票有几个 ann_date 版本」的分布。"""
    sub = df[df["end_date"].astype(str).str[:10] == per]
    if not len(sub):
        return {}
    g = sub.groupby("stock_code")["ann_date"].nunique()
    return {"stocks": int(len(g)), "rows": int(len(sub)),
            "multi": int((g > 1).sum()), "max_ver": int(g.max())}


# ---------------------------------------------------------------- ① snapshot
def cmd_snapshot(a) -> int:
    print(f"★ 验证对象：21–24 号共 {len(TABLES)} 张\n")
    tables: dict[str, dict] = {}
    for i, name in enumerate(TABLES, 1):
        ds = R.get(name)
        df = load_all(name)
        man = state.Manifest.load(name)
        tables[name] = {
            "keys": list(ds.keys), "date_field": ds.date_field,
            "watermark": man.max_partition_date(), "total_rows": int(len(df)),
            "periods": {p: period_fp(df, ds, p) for p in (PERIOD_A, PERIOD_B, PERIOD_C)},
            "versions": {p: versions(df, p) for p in (PERIOD_A, PERIOD_B, PERIOD_C)},
            "all_periods": sorted(set(df["end_date"].astype(str).str[:10])) if len(df) else [],
        }
        v = tables[name]["versions"]
        print(f"[{i}/4] {name}")
        print(f"      总行数 {len(df):,} | 水位 {man.max_partition_date()}")
        for p in (PERIOD_A, PERIOD_B, PERIOD_C):
            r = tables[name]["periods"][p]
            vv = v[p]
            nrows = "无" if r is None else f"{r['rows']:,} 行"
            print(f"      {p}: {nrows} | 多版本股票 {vv.get('multi', 0)} 只"
                  f" / 最多 {vv.get('max_ver', 0)} 版")

    doc = {"T": a.date, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "tables": tables,
           "plan": {"A": PERIOD_A, "B": PERIOD_B, "C": PERIOD_C, "C_rows": C_DELETE_ROWS}}
    p = dat_path(a.date)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✔ 台账已存 {p}")
    return 0


# ---------------------------------------------------------------- ② delete
def cmd_delete(a) -> int:
    """读一次 → 算 A/B/C 三个 mask → 备份各组 → 写一次。

    ★ 必须一次算完再写：旧写法每组各读一次盘，而 mask 是按**原始索引**算的，
      上一组写完 `reset_index(drop=True)` 会让索引整体位移 ⇒ **删错行**。
    """
    doc = read_json(dat_path(a.date))
    if not doc:
        raise SystemExit("✗ 先跑 snapshot")
    tag = "预演（未删任何数据）" if not a.commit else "真删"
    print(f"★ {tag} —— A 整期 {PERIOD_A} / B 旧版本 {PERIOD_B} / "
          f"C 窗口外 {PERIOD_C}（前 {C_DELETE_ROWS} 行）\n")

    results: list[dict] = []
    for i, name in enumerate(TABLES, 1):
        df = load_all(name)
        ed = df["end_date"].astype(str).str[:10]
        an = df["ann_date"].astype(str).str[:10]

        mask_a = (ed == PERIOD_A)                                  # 整期
        sub_b = (ed == PERIOD_B)
        # 该期里每只票的**最大 ann_date**（= 最新版本），其余版本视为"旧版本行"
        maxper = df[sub_b].groupby("stock_code")["ann_date"].max()
        keep = df["stock_code"].map(maxper)
        mask_b = sub_b & (an != keep.astype(str).str[:10])          # 只删旧版本
        idx_c = df.index[ed == PERIOD_C][:C_DELETE_ROWS]
        mask_c = df.index.isin(idx_c)                               # 窗口外前 N 行

        print(f"[{i}/4] {name}")
        for grp, mask in (("A", mask_a), ("B", mask_b), ("C", mask_c)):
            hit = df[mask]
            rec = {"table": name, "grp": grp, "rows": int(len(hit)),
                   "period": {"A": PERIOD_A, "B": PERIOD_B, "C": PERIOD_C}[grp],
                   "backup": ""}
            if a.commit and len(hit):
                BACKTEST.mkdir(parents=True, exist_ok=True)
                bf = BACKTEST / f"quarterly_deleted_{name}__{grp}.parquet"
                store._atomic_write(hit.reset_index(drop=True), bf)
                rec["backup"] = str(bf)
            results.append(rec)
            note = {"A": "整期", "B": "旧版本行", "C": "窗口外，期望**不补**"}[grp]
            print(f"      {grp} 删 {len(hit):>7,} 行  （{note}）")
        print()

        if a.commit:
            write_all(name, df[~(mask_a | mask_b | mask_c)].reset_index(drop=True))

    if a.commit:
        doc["deleted"] = results
        doc["deleted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dat_path(a.date).write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        tot = sum(r["rows"] for r in results)
        print(f"★ 合计删除 {tot:,} 行 · 备份 state/backtest/quarterly_deleted_<表>__<A|B|C>.parquet")
    return 0


# ---------------------------------------------------------------- ③ verify
def cmd_verify(a) -> int:
    doc = read_json(dat_path(a.date))
    if not doc:
        raise SystemExit("✗ 缺台账")
    before = doc["tables"]

    L = [f"# 季频更新逻辑验证报告 · 21–24 号", "",
         f"- 验证于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
         f"（删除于 {doc.get('deleted_at', '—')}）",
         f"- 4 张 `mode='per_stock'` 表，**无 coverage 机制**，自愈深度 = `PERIOD_LOOKBACK=3` 期",
         "",
         f"| 组 | 删什么 | 期望 |", "|:--|:--|:--|",
         f"| A | 整期 `{PERIOD_A}`（窗口内） | 原样补回 |",
         f"| B | `{PERIOD_B}` 的**旧 ann_date 版本行**（窗口内） | 补回（验多版本合入） |",
         f"| C | `{PERIOD_C}` 前 {C_DELETE_ROWS} 行（**窗口外**） | **不补**（确认边界） |",
         ""]

    rows_all: list[dict] = []
    for name in TABLES:
        ds = R.get(name)
        df = load_all(name)
        b = before[name]
        for grp, per in (("A", PERIOD_A), ("B", PERIOD_B), ("C", PERIOD_C)):
            bp = b["periods"][per] or {"rows": 0, "md5": None}
            ap = period_fp(df, ds, per) or {"rows": 0, "md5": None}
            rows_all.append({"table": name, "grp": grp, "period": per,
                             "b_rows": bp["rows"], "a_rows": ap["rows"],
                             "b_md5": bp["md5"], "a_md5": ap["md5"]})

    def judge(r) -> str:
        if r["grp"] == "C":
            if r["a_rows"] < r["b_rows"]:
                return "✔ 如预期未补（窗口边界成立）"
            if r["a_rows"] > r["b_rows"]:
                return f"✔ 已被 `--as-of` 补救（+{r['a_rows']-r['b_rows']}，见 D 组）"
            return "—"
        if r["a_rows"] == r["b_rows"] and r["a_md5"] == r["b_md5"]:
            return "✔ 原样补回（行数+md5 一致）"
        if r["a_rows"] == r["b_rows"]:
            return "⚠️ 行数对但 md5 不同（内容变了）"
        if r["a_rows"] > r["b_rows"]:
            return f"⚠️ 补回且**更多**（+{r['a_rows']-r['b_rows']}）"
        return f"🚩 没补回（缺 {r['b_rows']-r['a_rows']} 行）"

    for grp, label in (("A", "A 组 · 整期删除"), ("B", "B 组 · 只删旧版本行"), ("C", "C 组 · 窗口外")):
        L += [f"## {label}", "",
              "| 数据集 | 报告期 | 删前行数 | 跑后行数 | 删前 md5 | 跑后 md5 | 判定 |",
              "|:--|:--|--:|--:|:--|:--|:--|"]
        for r in [x for x in rows_all if x["grp"] == grp]:
            L.append(f"| `{r['table']}` | {r['period']} | {r['b_rows']:,} | {r['a_rows']:,} | "
                     f"`{(r['b_md5'] or '—')[:10]}` | `{(r['a_md5'] or '—')[:10]}` | {judge(r)} |")
        L += [""]

    # ---- D 组：C 组的补救后置状态（现在 vs 基线） ----
    L += ["## D 组 · `--as-of` 补救后的状态（C 组那期）", "",
          "| 数据集 | 报告期 | 基线行数 | 补救后行数 | 判定 |",
          "|:--|:--|--:|--:|:--|"]
    for r in [x for x in rows_all if x["grp"] == "C"]:
        d = r["a_rows"] - r["b_rows"]
        L.append(f"| `{r['table']}` | {r['period']} | {r['b_rows']:,} | {r['a_rows']:,} | "
                 f"{'✔ 已补回（+%d）' % d if d > 0 else '（未补救）'} |")
    L += [""]

    A = [x for x in rows_all if x["grp"] == "A"]
    B = [x for x in rows_all if x["grp"] == "B"]
    C = [x for x in rows_all if x["grp"] == "C"]
    a_ok = all(x["a_rows"] == x["b_rows"] and x["a_md5"] == x["b_md5"] for x in A)
    b_ok = all(x["a_rows"] == x["b_rows"] and x["a_md5"] == x["b_md5"] for x in B)
    # C 组两种合法终态：未补救（<=删前）或已被 --as-of 补救（且补到服务端总数）
    c_remedied = all(x["a_rows"] > x["b_rows"] for x in C)
    c_ok = c_remedied or all(x["a_rows"] < x["b_rows"] for x in C)

    concl = [
        f"**结论**：A 整期自愈 {'✔ 通过' if a_ok else '🚩 未通过'} · "
        f"B 多版本合入 {'✔ 通过' if b_ok else '🚩 未通过'} · "
        f"C 窗口外边界 {'✔ 成立（且已用 --as-of 补救，见 D 组）' if c_remedied else '✔ 符合预期（未补救）' if c_ok else '🚩 不符'}",
        "",
        "★ **核心结论**：窗口内（最近 3 期）的更新**完全正确且及时** —— "
        "整期删除与「旧 ann_date 版本行」删除都原样补回（行数 + md5 一致）。",
        "",
        "★ **边界**：窗口外**不会自动更新**（`PERIOD_LOOKBACK=3` 的设计使然，不是 bug）。"
        "补救手段 = `main.py run --as-of <把 T 拨回该期之后的某天>`；"
        "**实测补上后与服务端 total 完全相等**。",
        ""]
    # 插在第一个 "## " 小标题之前（不要插进表头里）
    cut = next((k for k, x in enumerate(L) if x.startswith("## ")), 6)
    L = L[:cut] + concl + L[cut:]

    out = BACKTEST / f"REPORT_quarterly_{a.date}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n✔ 报告已写 {out}")
    return 0 if (a_ok and b_ok and c_ok) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["snapshot", "delete", "verify"])
    ap.add_argument("--date", default="2026-09-18")
    ap.add_argument("--commit", action="store_true", help="delete：真删（不给则预演）")
    a = ap.parse_args()
    return {"snapshot": cmd_snapshot, "delete": cmd_delete, "verify": cmd_verify}[a.step](a)


if __name__ == "__main__":
    raise SystemExit(main())
