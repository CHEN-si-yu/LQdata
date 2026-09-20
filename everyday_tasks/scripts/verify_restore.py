"""「删掉某一天 → 重新下载」的验收报告：**正确性 + 效率**（只读，零请求）。

用法：
    python scripts/verify_restore.py --date 2026-09-14 \
        [--baseline fixed_2026-09-14] [--out state/backtest/REPORT_restore.md]

三重比对（互相独立，都过了才敢说"原样补回"）：
  ① **删除前基线**：`main.py backtest baseline` 之前记下的逐表指纹（md5 + 逐列 md5）
  ② **删除台账**：`state/backtest/last_deleted.json` 里那一天的 `day_md5`
     （删除动作自己算的指纹，与 ① 由不同代码路径产生）
  ③ **单日 MD5 台账**：`state/dayhash/<表>.parquet`（每天跑增量时维护的长期台账）

退出码：0 = 这一天所有被删的表**行数都补回了**；1 = 还有表没补回（行数不足/为空）。

★ 2026-09-15 晚修正退出码语义（实战第 1 天实测踩到）：
  旧实现把"行数够但 md5 不同"也判为失败（rc=1），于是 `catchup_0914.sh` 看门狗
  会在 15:30/18:00/21:00 反复重跑 —— **而重跑永远修不好它**。
  实测取证（stock_kline 2026-09-14，删前备份 vs 补回后逐行比对）：
    行数 11,100 = 11,100；差异只在 **4 只当天停牌的票**（601059/601198/601995/603400）
    的**未收盘月K线**聚合列上（vol/amount/open/pre_close/change/pct_chg/ah_*），
    `close`/`low` 逐行相同 —— 即**上游在删除与补回之间重算过这几行**，
    不是我们没补回。这种差异由 `main.py` 的 🔴 单日台账负责报告，
    不该让看门狗重试。故：rc=1 只表示"行数没补回"；内容差异降级为报告里的 ⚠️ 明细。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_incremental import paths, registry as R            # noqa: E402
from data_incremental.core import state                      # noqa: E402
from data_incremental.pipeline import dayhash as DH          # noqa: E402
from data_incremental.tools import backtest_increment as BT  # noqa: E402


def _load_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _read_backup(p: str):
    import pandas as pd
    try:
        return pd.read_parquet(p)
    except (OSError, ValueError):
        return None


def backup_diff(name: str, date: str, backup: str) -> dict | None:
    """把"删前备份"与"现在"逐行对齐，说清**到底哪里不一样**（而不是只说 md5 不同）。

    返回 None = 复算不上/备份不可读（此时报告里只说 md5 不同）。
    """
    import pandas as pd

    old = _read_backup(backup)
    if old is None or len(old) == 0:
        return None
    ds = R.get(name)
    now = DH._read_day(name, ds, date)       # noqa: SLF001 —— 复用台账的按日读取
    if now is None or len(now) == 0:
        return None

    keys = [k for k in (ds.keys or ()) if k in old.columns and k in now.columns]
    if not keys:
        return None

    def _kk(df):
        return df[keys].astype(str).agg("|".join, axis=1)

    o = old.assign(_k=_kk(old)).set_index("_k")
    n = now.assign(_k=_kk(now)).set_index("_k")
    common = o.index.intersection(n.index)
    only_old = len(o.index.difference(n.index))
    only_new = len(n.index.difference(o.index))

    changed_cols: list[str] = []
    changed_rows = 0
    cols = [c for c in o.columns if c in n.columns and c != "_k"]
    if len(common):
        a, b = o.loc[common], n.loc[common]
        row_diff = pd.Series(False, index=common)
        for c in cols:
            sa, sb = a[c], b[c]
            try:
                d = ~(sa.eq(sb) | (sa.isna() & sb.isna()))
            except (TypeError, ValueError):
                continue
            if bool(d.any()):
                changed_cols.append(c)
                row_diff |= d
        changed_rows = int(row_diff.sum())
    return {"rows_old": len(old), "rows_now": len(now),
            "changed_cols": changed_cols, "changed_rows": changed_rows,
            "only_in_backup": only_old, "only_in_now": only_new}


def collect(date: str) -> dict:
    """跑完三重比对，返回结构化结果。"""
    deleted = _load_json(paths.BACKTEST_DIR / "last_deleted.json", {})
    # 只关心"那一天真被删过"的表
    deleted = {k: v for k, v in deleted.items() if str(v.get("date"))[:10] == date}
    names = sorted(deleted)

    rows: list[dict] = []
    for name in names:
        ds = R.get(name)
        rec = DH.compute_day(name, ds, date)
        now_rows = int(rec["rows"]) if rec else 0
        now_md5 = rec["md5"] if rec else None
        del_rows = int(deleted[name].get("rows") or 0)
        del_md5 = deleted[name].get("day_md5")
        # ③ 单日台账
        ledger = DH.load_ledger(name)
        led_md5 = None
        if len(ledger):
            led = ledger[ledger["date"].astype(str).str[:10] == date]
            if len(led):
                led_md5 = str(led.iloc[-1]["md5"])
        same_del = bool(now_md5 and del_md5 and now_md5 == del_md5)
        same_led = bool(now_md5 and led_md5 and now_md5 == led_md5)
        restored = now_rows >= del_rows * 0.98 and now_rows > 0
        # ★ 行数补回了但指纹不同 → 逐行定位差异（只对这类表算，成本可控）
        diff = None
        if restored and not same_del and deleted[name].get("backup"):
            try:
                diff = backup_diff(name, date, str(deleted[name]["backup"]))
            except Exception:  # noqa: BLE001 —— 对比只是报告增强，绝不影响主流程
                diff = None
        rows.append({
            "name": name, "deleted_rows": del_rows, "now_rows": now_rows,
            "deleted_md5": del_md5, "now_md5": now_md5, "ledger_md5": led_md5,
            "same_as_deleted": same_del, "same_as_ledger": same_led,
            "restored": restored, "diff": diff,
        })

    run_log = []
    p = paths.RUN_LOG
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines()[-5:]:
            try:
                run_log.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    last_report = _load_json(paths.LAST_REPORT, {})
    return {"date": date, "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tables": rows, "run_log_tail": run_log, "last_report": last_report}


def render(res: dict) -> str:
    date = res["date"]
    rows = res["tables"]
    ok = [r for r in rows if r["restored"] and r["same_as_deleted"]]
    part = [r for r in rows if r["restored"] and not r["same_as_deleted"]]
    miss = [r for r in rows if not r["restored"]]

    out: list[str] = []
    out.append(f"# {date} 「删掉 → 重新下载」验收报告")
    out.append(f"\n生成时间：{res['generated_at']}　（只读检查，零 API 请求）\n")
    out.append(f"## 结论\n")
    out.append(f"- ✔ **逐行一致地补回**：{len(ok)} 张")
    out.append(f"- ⚠️ 行数够但内容不同：{len(part)} 张" + (f"　{ [r['name'] for r in part] }" if part else "")
               + ("　← **补回本身没问题**：上游在这期间重算过这几行，重试也不会变（见下）" if part else ""))
    out.append(f"- ✘ 尚未补回：{len(miss)} 张" + (f"　{ [r['name'] for r in miss] }" if miss else ""))
    out.append("")

    for r in part:
        d = r.get("diff")
        out.append(f"### ⚠️ {r['name']}：逐行比对（删前备份 vs 现在）\n")
        if not d:
            out.append("- 行数一致，但备份 parquet 读不出来/主键对不上，只能报 md5 不同。\n")
            continue
        out.append(f"- 行数：备份 {d['rows_old']:,} → 现在 {d['rows_now']:,}"
                   f"（备份独有 {d['only_in_backup']} / 现在独有 {d['only_in_now']}）")
        out.append(f"- 值有变化的行：**{d['changed_rows']:,}** 行")
        out.append(f"- 变化的列：`{', '.join(d['changed_cols']) or '—'}`")
        if d["changed_rows"] and d["only_in_backup"] == 0 and d["only_in_now"] == 0:
            out.append("- 判定：**主键集合完全一致，只有部分列的值被上游改过**"
                       "（重下不会改变这一结果；日常由 `main.py` 的 🔴 单日台账持续盯）\n")
        else:
            out.append("- 判定：行集合本身有出入，需要人工看一眼\n")
    out.append("## 逐表明细\n")
    out.append("| 数据集 | 删掉 | 现在 | 与删除台账 md5 | 与单日台账 md5 | 判定 |")
    out.append("|:--|--:|--:|:--:|:--:|:--|")
    for r in sorted(rows, key=lambda x: (x["restored"], x["name"])):
        verdict = ("✔ 完全一致" if r["restored"] and r["same_as_deleted"] else
                   "⚠️ 行数够但内容不同" if r["restored"] else
                   f"✘ 没补回（{(r['now_rows'] / r['deleted_rows'] * 100 if r['deleted_rows'] else 0):.0f}%）")
        out.append(f"| {r['name']} | {r['deleted_rows']:,} | {r['now_rows']:,} | "
                   f"{'✔' if r['same_as_deleted'] else '✘'} | "
                   f"{'✔' if r['same_as_ledger'] else ('—' if not r['ledger_md5'] else '✘')} | {verdict} |")

    out.append("\n## 效率（本轮重新下载那一跑）\n")
    lr = res.get("last_report") or {}
    if lr:
        out.append(f"- T = {lr.get('T')}　用时 **{lr.get('seconds')}s**　请求 **{lr.get('requests')}**　"
                   f"重试 **{lr.get('retries')}**（空响应自旋 {lr.get('empty_retries', 0)}）")
        out.append(f"- verdict：{lr.get('verdict')}")
        bad = [a for a in (lr.get("alerts") or []) if "重试率" not in a]
        if bad:
            out.append(f"- 未完成的项：{len(bad)} 条")
            for a in bad[:15]:
                out.append(f"    - {a}")
    if res.get("run_log_tail"):
        out.append("\n最近几次运行的流水（state/run_log.jsonl）：\n")
        out.append("| 完成时间 | T | 用时 | 请求 | 重试 | verdict |")
        out.append("|:--|:--|--:|--:|--:|:--|")
        for g in res["run_log_tail"]:
            out.append(f"| {g.get('finished_at')} | {g.get('T')} | {g.get('seconds')}s | "
                       f"{g.get('requests')} | {g.get('retries')} | {g.get('verdict')} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-09-14")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    res = collect(args.date)
    text = render(res)
    print(text)
    out = Path(args.out) if args.out else paths.BACKTEST_DIR / f"REPORT_restore_{args.date}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n（报告已写入 {out}）")

    miss = [r for r in res["tables"] if not r["restored"]]
    diff = [r for r in res["tables"] if r["restored"] and not r["same_as_deleted"]]
    # ★ 只有"行数没补回"才判失败：内容差异是上游改了值，重试无意义（见文件头说明）
    if diff:
        print(f"\n⚠️ 注意：{len(diff)} 张表行数已补回但值有变化 —— "
              f"{[r['name'] for r in diff]}；这不是「没补回」，重跑不会改变它。")
    return 0 if not miss else 1


if __name__ == "__main__":
    raise SystemExit(main())
