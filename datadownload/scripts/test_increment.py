#!/usr/bin/env python
"""增量可恢复性测试 —— 删掉每个数据集的"最新一天"，看增量能不能正确补回来。

用户 2026-09-13 的要求（原话）：

> 测试将所有的数据最新的日期删掉（对，就是将目前的数据库中的最新日期的部分数据删除），
> 测试能否正确的返还。这里的最新指的是**本数据的最新**，例如普通日频数据的最新是 0911，
> 那就是把所有的 0911 的数据都删掉，而 margin_detail 检查出最新的日期是 0910，
> 则删除所有 0910 的数据，如果是季度数据，假设最新的季度数据是 20260630，
> 那么就是把这个文件中所有的 0630 的数据删掉。
> **注意删除数据的范围不要把所有的数据都删掉了。**

**为什么这是唯一能系统性发现那类 bug 的手段**：`engine._plan_ranges` 按 coverage 找缺口 ——
coverage 说"覆盖了"就不再请求。所以如果 coverage 与实际不符，删掉数据后**增量根本不会去补**，
而这个测试会**立刻变红**。`index_daily` 缺 159 个交易日却潜伏至今，就是这个机制
（见 ../QUANT_PLATFORM.md §16.3 问题 1）。本测试就是当年没做的那一步。

**三种子命令**：

    # ① 看看要删什么（默认就是 dry-run，什么都不改）
    python scripts/test_increment.py

    # ② 真删（会先把删掉的行备份到 /tmp/increment_test/）
    python scripts/test_increment.py --apply

    # ③ 跑完增量后，逐行校验是否原样补回
    python scripts/test_increment.py --verify

    # 后悔药：从备份还原
    python scripts/test_increment.py --restore

⚠️ 必须没有其它下载实例在跑（共享盘无文件锁，两个进程同时写会坏数据）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATE = ROOT / "state"
DATA = ROOT / "data"
BACKUP = Path("/tmp/increment_test")

# 删掉的行数占该数据集比例超过这个值就拒绝（用户明确要求"别把所有数据都删掉"）
MAX_DELETE_FRACTION = 0.5

# 这些没有时间维度，不参与（快照表删了就等于整表没了）
SKIP_FREQS = {"snapshot"}


def running(pattern: str) -> list[str]:
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if pattern in cmd and "test_increment" not in cmd:
            out.append(p.name)
    return out


def load_freq() -> dict:
    with open(ROOT / "conf" / "frequency.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plan_one(name: str, spec, date_field: str, data_root: Path,
             state_root: Path) -> dict | None:
    """算出这个数据集"最新一天"有多少行、占多少比例。"""
    dpath = data_root / name
    if not dpath.is_dir():
        return None
    files = sorted(dpath.glob("year=*/data.parquet")) + [dpath / "data.parquet"]
    files = [f for f in files if f.exists()]
    if not files:
        return None

    total = 0
    latest: str | None = None
    per_file: list[tuple[Path, int, int]] = []   # (文件, 总行, 该日行数)
    for f in files:
        try:
            df = pd.read_parquet(f, columns=[date_field])
        except Exception:
            continue
        if date_field not in df.columns or len(df) == 0:
            continue
        s = df[date_field].astype(str).str[:10]
        total += len(s)
        per_file.append((f, len(s), 0))
        mx = s.max()
        if latest is None or mx > latest:
            latest = mx
    if latest is None or total == 0:
        return None

    n_latest = 0
    for f, n, _ in per_file:
        df = pd.read_parquet(f, columns=[date_field])
        s = df[date_field].astype(str).str[:10]
        n_latest += int((s == latest).sum())

    skip = None
    if n_latest == 0:
        skip = "该数据集没有任何行落在最新日期上"
    elif n_latest > total * MAX_DELETE_FRACTION:
        skip = f"要删 {n_latest:,}/{total:,} 行（{n_latest/total:.0%}），超过 {MAX_DELETE_FRACTION:.0%} 安全线"

    return {"name": name, "date_field": date_field, "latest": latest,
            "rows_latest": n_latest, "rows_total": total, "files": files,
            "skip": skip, "keys": tuple(spec.keys)}


def do_delete(p: dict, spec, data_root: Path, state_root: Path) -> None:
    """删掉最新日期的行并备份。**故意不动 coverage** —— 这正是测试的关键。"""
    name, latest, col = p["name"], p["latest"], p["date_field"]
    BACKUP.mkdir(parents=True, exist_ok=True)
    removed_frames = []
    for f in p["files"]:
        df = pd.read_parquet(f)
        if col not in df.columns or len(df) == 0:
            continue
        s = df[col].astype(str).str[:10]
        mask = s == latest
        if not mask.any():
            continue
        removed_frames.append(df[mask])
        kept = df[~mask].reset_index(drop=True)
        tmp = f.with_suffix(".parquet.tmp")
        kept.to_parquet(tmp, index=False)
        tmp.replace(f)          # 原子替换
    if not removed_frames:
        return
    removed = pd.concat(removed_frames, ignore_index=True)
    removed.to_parquet(BACKUP / f"{name}__{latest}.parquet", index=False)

    # 同步 manifest 的 partitions 元数据（行数/日期范围），但**不碰 coverage**
    st = state_root / f"{name}.json"
    if st.exists():
        man = json.loads(st.read_text(encoding="utf-8"))
        for y in list((man.get("partitions") or {})):
            f = data_root / name / f"year={y}" / "data.parquet"
            if not f.exists():
                continue
            try:
                d = pd.read_parquet(f, columns=[col])
            except Exception:
                continue
            if len(d) == 0:
                man["partitions"].pop(y, None)
                continue
            ss = d[col].astype(str).str[:10]
            man["partitions"][y] = {"rows": len(d), "min_date": ss.min(),
                                    "max_date": ss.max(), "updated_at": man["partitions"][y].get("updated_at")}
        st.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")


def verify_one(bak: Path, spec, data_root: Path) -> dict:
    """把备份的"那一天"与现在的本地数据逐行比对。

    ⚠️ 必须按**备份文件**遍历，不能按"当前最新日"去找 —— 第一版就是那么写的，
    结果没补回来的数据集当前最新日已经变了，备份名对不上，全被误报成"没删过"。
    """
    name, latest = bak.stem.split("__", 1)
    col = spec.date_field
    want = pd.read_parquet(bak)
    dpath = data_root / name
    files = sorted(dpath.glob("year=*/data.parquet")) + [dpath / "data.parquet"]
    frames = []
    for f in files:
        if not f.exists():
            continue
        try:
            d = pd.read_parquet(f)
        except Exception:
            continue
        if col in d.columns and len(d):
            d = d[d[col].astype(str).str[:10] == latest]
            if len(d):
                frames.append(d)
    if not frames:
        return {"name": name, "date": latest, "status": "missing",
                "want_rows": len(want), "got_rows": 0}
    got = pd.concat(frames, ignore_index=True)

    keys = [c for c in (spec.keys or ()) if c in want.columns and c in got.columns]
    if keys:
        w = want.sort_values(keys).reset_index(drop=True)
        g = got.sort_values(keys).reset_index(drop=True)
        same_keys = list(zip(*(w[c].astype(str) for c in keys))) == \
                    list(zip(*(g[c].astype(str) for c in keys)))
    else:
        same_keys = len(want) == len(got)

    cols = [c for c in want.columns if c in got.columns]
    same_vals = True
    if same_keys and len(want) == len(got) and cols:
        w2 = want[cols].sort_values(keys or cols).reset_index(drop=True)
        g2 = got[cols].sort_values(keys or cols).reset_index(drop=True)
        same_vals = w2.equals(g2)

    ok = same_keys and same_vals and len(want) == len(got)
    return {"name": name, "date": latest,
            "status": "restored_identical" if ok else "differs",
            "want_rows": len(want), "got_rows": len(got),
            "same_keys": same_keys, "same_values": same_vals}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真的删（默认 dry-run）")
    ap.add_argument("--verify", action="store_true", help="校验是否补回（跑完增量后）")
    ap.add_argument("--restore", action="store_true", help="从备份还原")
    ap.add_argument("-d", "--dataset", action="append", help="只测指定数据集")
    ap.add_argument("--force", action="store_true", help="跳过「有下载进程在跑」检查")
    args = ap.parse_args()

    from lingqi import spec as S
    from main import ordered_specs

    fqc = load_freq()
    ds_cfg = fqc.get("datasets") or {}
    specs = [s for s in ordered_specs(None) if s.enabled]
    if args.dataset:
        specs = [s for s in specs if s.name in set(args.dataset)]

    if (args.apply or args.restore) and not args.force:
        procs = running("main.py run") + running("catchup_after_main")
        if procs:
            print(f"✖ 有下载进程在跑（PID {', '.join(procs)}），拒绝执行。")
            return 1

    # ------------------------------------------------------------ 还原
    if args.restore:
        n = 0
        for bak in sorted(BACKUP.glob("*__*.parquet")):
            name, date = bak.stem.split("__", 1)
            add = pd.read_parquet(bak)
            col = ds_cfg.get(name, {}).get("date_field", "trade_date")
            if col not in add.columns:
                print(f"  ⚠️ {name}: 备份里没有 {col} 列，跳过")
                continue
            dpath = DATA / name
            # ★ 只还原到**该日期所属年份**那个分区 —— 第一版把备份合并进了
            #   所有年分区，于是 stock_top_list 的 09-11 被复制到 17 个年文件里
            #   （17 × 63 = 1071 行）。年分区文件里混进别的年份的行，整库就脏了。
            year = str(add[col].astype(str).str[:10].iloc[0])[:4]
            f = dpath / f"year={year}" / "data.parquet"
            if not f.exists():
                f = dpath / "data.parquet"      # partition=none 的表
            if not f.exists():
                print(f"  ⚠️ {name}: 找不到目标分区文件，跳过")
                continue
            d = pd.read_parquet(f)
            add = add[[c for c in d.columns if c in add.columns]]
            merged = pd.concat([d, add], ignore_index=True)
            k = [c for c in S.get(name).keys if c in merged.columns]
            if k:
                merged = merged.drop_duplicates(subset=k, keep="last")
            tmp = f.with_suffix(".parquet.tmp")
            merged.to_parquet(tmp, index=False)
            tmp.replace(f)
            n += 1
            print(f"  还原 {name} {date} → year={year}")
        print(f"✔ 还原了 {n} 个分区文件")
        return 0

    # ------------------------------------------------------------ 计划 / 校验
    plans = []
    for s in specs:
        cfg = ds_cfg.get(s.name) or {}
        if cfg.get("freq") in SKIP_FREQS:
            continue
        if not s.date_field:
            continue
        p = plan_one(s.name, s, cfg.get("date_field") or s.date_field, DATA, STATE)
        if p:
            plans.append(p)

    if args.verify:
        baks = sorted(BACKUP.glob("*__*.parquet"))
        if not baks:
            print(f"{BACKUP} 下没有备份 —— 先跑 --apply 删除，再跑增量，再来校验。")
            return 1
        print(f"{'数据集':34s} {'删掉的那天':11s} {'应有':>10s} {'现有':>10s}  结论")
        print("-" * 90)
        ok_n = bad_n = 0
        for bak in baks:
            name = bak.stem.split("__", 1)[0]
            if args.dataset and name not in set(args.dataset):
                continue
            if name not in S.REGISTRY:
                continue
            r = verify_one(bak, S.get(name), DATA)
            if r["status"] == "restored_identical":
                verdict, ok_n = "✔ 逐行一致", ok_n + 1
            else:
                verdict = (f"✘ {r['status']}"
                           + (f" (键一致={r.get('same_keys')} 值一致={r.get('same_values')})"
                              if r["status"] == "differs" else ""))
                bad_n += 1
            print(f"{r['name']:34s} {r['date']:11s} "
                  f"{r.get('want_rows', 0):>10,} {r.get('got_rows', 0):>10,}  {verdict}")
        print("-" * 90)
        print(f"结论：{ok_n}/{ok_n + bad_n} 个数据集**原样补回**"
              + (f"，{bad_n} 个没补回来 ⚠️" if bad_n else " ✓"))
        return 1 if bad_n else 0

    print(f"{'数据集':34s} {'最新日':11s} {'要删行数':>12s} {'占该集比例':>10s}  说明")
    print("-" * 92)
    todo, skipped = [], []
    for p in plans:
        pct = p["rows_latest"] / max(p["rows_total"], 1)
        if p["skip"]:
            skipped.append(p)
            print(f"{p['name']:34s} {p['latest']:11s} {p['rows_latest']:>12,} {pct:>9.1%}  ⏭ {p['skip']}")
        else:
            todo.append(p)
            print(f"{p['name']:34s} {p['latest']:11s} {p['rows_latest']:>12,} {pct:>9.1%}")
    print("-" * 92)
    print(f"将删除 {len(todo)} 个数据集的「各自最新一天」，共 "
          f"{sum(p['rows_latest'] for p in todo):,} 行；跳过 {len(skipped)} 个")

    if not args.apply:
        print("\n以上是预览（dry-run），没有改动任何文件。")
        print("确认后：`--apply` 删除 → 跑 `main.py run` → `--verify` 校验。")
        return 0

    for p in todo:
        do_delete(p, S.get(p["name"]), DATA, STATE)
        print(f"  ✔ 已删并备份 {p['name']} {p['latest']}（{p['rows_latest']:,} 行）")
    print(f"\n备份在 {BACKUP}/。接下来跑 `python main.py run`，再 `--verify`。")
    print("想直接还原：`python scripts/test_increment.py --restore`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
