#!/usr/bin/env python
"""删掉某个年份之前的数据（配合 lingqi/spec.py 的 GLOBAL_START）。

两类存储要分开处理：
  partition="year"  数据在 data/<数据集>/year=YYYY/data.parquet → 直接删目录
  partition="none"  整表单文件。里面又分两种：
                      · 真时间序列（如 basic_calendar，1990~2026 逐日）→ 按行过滤重写
                      · 快照（stock_list / tdx_blocks / dc_blocks …）→ **整表跳过**，
                        它没有时间维度，按日期过滤等于把快照删光

state/<数据集>.json 的 coverage **故意不动** —— 留着它，后续增量才不会把删掉的历史又拉回来。

    # 先看要删什么（默认 dry-run，什么都不改）
    /autodl-fs/data/miniconda3/bin/python scripts/drop_before.py

    # 确认后真删
    /autodl-fs/data/miniconda3/bin/python scripts/drop_before.py --apply

    # 只处理某个数据集
    /autodl-fs/data/miniconda3/bin/python scripts/drop_before.py -d stock_daily --apply

并发安全：manifest 是**按数据集分文件**的（state/<名>.json），而且 cmd_run 是
单趟顺序执行（main.py 的 `for i, s in enumerate(specs, 1)`），跑过的数据集不会再回头。
所以唯一会被覆盖的情况，是去改**当前正在跑的那个**数据集的 state 文件 ——
本脚本只拒绝这一种情况，不影响其它数据集。

basic_calendar 故意不在删除范围：它是 32KB 的交易日历参考表，不是行情数据。
保留完整的 1990~2026 反而有用 —— 万一将来有 bug 误查 2010 前的区间，
引擎能靠日历判定"本该有数据却为空"并记进 suspect；把日历也截断就会伪装成
"合法为空"，把 bug 藏起来。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
STATE = ROOT / "state"


def running(pattern: str) -> list[str]:
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if pattern in cmd and "drop_before" not in cmd:
            out.append(p.name)
    return out


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def atomic_write(df: pd.DataFrame, dest: Path) -> None:
    tmp = dest.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)


def trim_ranges(spans, cutoff: str) -> list:
    """把 [[start,end], ...] 裁到 cutoff 之后：整段在前的丢掉，跨越 cutoff 的裁左端。

    为什么必须裁：verify.py 会核对「coverage 声明已覆盖的年份分区是否存在」，
    留着 1990~2009 的 coverage 会让它报一堆假的「缺 N 个交易日」。
    为什么裁了也不会重新下载：engine._plan_ranges 的缺口计算是
    missing_ranges(spec.start, end)，而 spec.start 现在恒为 2010-01-01，
    早于它的区间永远不会进入比对。
    """
    out = []
    for item in spans or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        s, e = str(item[0]), str(item[1])
        if e < cutoff:
            continue
        out.append([cutoff, e] if s < cutoff else [s, e])
    return out


def single_file_plan(name: str, date_field: str, cutoff: str):
    """partition='none' 的数据集怎么处理。返回 (动作, 要删行数, 占用字节, 说明)。"""
    fs = sorted((DATA / name).glob("*.parquet"))
    if not fs:
        return ("skip", 0, 0, "无文件")
    df = pd.read_parquet(fs[0])
    if date_field not in df.columns:
        return ("skip", 0, 0, f"没有 {date_field} 列")
    s = df[date_field].astype(str)
    years = s.str[:4]
    years = years[years.str.isdigit()]
    uniq = sorted(set(years))
    if len(uniq) <= 1:
        return ("snapshot", 0, 0, f"快照（只有 {uniq[0] if uniq else '?'} 一个日期）→ 整表保留")
    n_old = int((s < cutoff).sum())
    if not n_old:
        return ("skip", 0, 0, "没有更早的行")
    nbytes = sum(f.stat().st_size for f in fs)
    return ("filter", n_old, nbytes, f"时间序列 {uniq[0]}~{uniq[-1]} → 按行过滤")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2010-01-01", help="早于这个日期的数据会被删（默认 2010-01-01）")
    ap.add_argument("--apply", action="store_true", help="真的删（默认只预览）")
    ap.add_argument("-d", "--dataset", action="append", help="只处理指定数据集，可重复")
    ap.add_argument("--force", action="store_true", help="跳过「有下载进程在跑」的检查")
    args = ap.parse_args()

    from lingqi import spec as S           # 延后导入：避免参数错误时也去建 spec

    cutoff_year = int(args.cutoff[:4])

    # 当前正在跑的数据集：只有它的 state 会被内存里的 manifest 覆盖
    inflight = None
    try:
        inflight = json.loads((STATE / "status.json").read_text(encoding="utf-8")).get("dataset")
    except (json.JSONDecodeError, OSError):
        pass
    procs = running("main.py run") + running("catchup_after_main")
    if procs and inflight and args.apply and not args.force:
        print(f"ℹ 检测到下载进程在跑（PID {', '.join(procs)}），当前数据集 = {inflight}")
        print(f"  只会跳过 {inflight} 的 state 文件，其余数据集不受影响。")
        print()

    plan: list[dict] = []
    for st in sorted(STATE.glob("*.json")):
        name = st.stem
        if name in ("ratelimit", "status", "basic_calendar"):
            continue
        if name == inflight:
            print(f"  (跳过 {name}: 正在跑，state 稍后会被进程覆盖)")
            continue
        if args.dataset and name not in args.dataset:
            continue
        if name not in S.REGISTRY:
            continue
        spec = S.get(name)
        try:
            man = json.loads(st.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        if spec.partition == "year":
            parts = man.get("partitions", {}) or {}
            old = sorted(k for k in parts if str(k).isdigit() and int(k) < cutoff_year)
            if not old:
                continue
            dirs = [DATA / name / f"year={y}" for y in old]
            dirs = [d for d in dirs if d.is_dir()]
            rows = sum(int(parts[str(y)].get("rows", 0)) for y in old)
            nb = sum(f.stat().st_size for d in dirs for f in d.rglob("*") if f.is_file())
            plan.append(dict(kind="dirs", name=name, dirs=dirs, state=st,
                             label=f"year={old[0]}~{old[-1]}", rows=rows, nbytes=nb))
        else:
            action, rows, nb, why = single_file_plan(name, spec.date_field, args.cutoff)
            if action != "filter":
                if args.dataset:
                    print(f"  (跳过 {name}: {why})")
                continue
            plan.append(dict(kind="filter", name=name, dirs=[DATA / name], state=st,
                             label=why, rows=rows, nbytes=nb))

    if not plan:
        print(f"没有早于 {args.cutoff} 的数据需要处理。")
        return 0

    tr = sum(p["rows"] for p in plan)
    tb = sum(p["nbytes"] for p in plan)
    print(f"{'数据集':34s} {'范围':>22s} {'行数':>13s} {'占用':>10s}")
    print("-" * 84)
    for p in plan:
        print(f"{p['name']:34s} {p['label']:>22s} {p['rows']:>13,} {human(p['nbytes']):>10s}")
    print("-" * 84)
    print(f"{'合计':34s} {'':>22s} {tr:>13,} {human(tb):>10s}")
    print()

    if not args.apply:
        print("以上是预览（dry-run），没有改动任何文件。确认后加 --apply 执行。")
        return 0

    for p in plan:
        st, name = p["state"], p["name"]
        bak = st.with_suffix(".json.predrop.bak")
        if not bak.exists():
            shutil.copy2(st, bak)

        if p["kind"] == "dirs":
            for d in p["dirs"]:
                shutil.rmtree(d)
            man = json.loads(st.read_text(encoding="utf-8"))
            for d in p["dirs"]:
                man.get("partitions", {}).pop(d.name.split("=", 1)[1], None)
            # coverage / suspect 一起裁到 cutoff 之后，否则 verify.py 会照着
            # 残留的 1990~2009 声明去核对已删掉的年份分区，报一堆假缺口
            n_cov = len(man.get("coverage") or [])
            man["coverage"] = trim_ranges(man.get("coverage"), args.cutoff)
            if "suspect" in man:
                man["suspect"] = trim_ranges(man.get("suspect"), args.cutoff)
            st.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"✔ {name}: 删除 {len(p['dirs'])} 个年分区 / {p['rows']:,} 行 / {human(p['nbytes'])}"
                  f"（coverage {n_cov}→{len(man['coverage'])} 段）")
        else:
            spec = S.get(name)
            f = sorted((DATA / name).glob("*.parquet"))[0]
            df = pd.read_parquet(f)
            before = len(df)
            df = df[df[spec.date_field].astype(str) >= args.cutoff].reset_index(drop=True)
            atomic_write(df, f)
            man = json.loads(st.read_text(encoding="utf-8"))
            pk = man.get("partitions", {}).get("0")
            if isinstance(pk, dict):
                pk["rows"] = len(df)
            st.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"✔ {name}: 行过滤 {before:,} → {len(df):,}（删 {before - len(df):,} 行）")

    print(f"\n完成：共删 {tr:,} 行 / {human(tb)}。state 已备份为 *.json.predrop.bak，后悔可还原。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
