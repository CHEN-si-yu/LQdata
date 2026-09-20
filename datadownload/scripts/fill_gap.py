#!/usr/bin/env python
"""外科补缺 —— 只补指定日期的数据，不重跑整个数据集。

为什么需要它：`audit.py` 查出来的缺口，用常规手段补代价可能极大。
最极端的例子是 `index_ths_daily`（`per_entity` 模式）：
它的续传粒度是**实体**（`done.entities`，1676 条），**没有按日期的完成标记**。
所以想补 2 天缺失，常规做法只有"清空全部实体标记重跑" = 33,520 次请求 / 约 2 小时。
而真相是那 2 天上游明明有数据（实测 09-08 有 1879 行、09-09 有 1871 行），
只是当时服务端偶发返回空（QUANT_PLATFORM.md §7 约束 5）被当成了正常。
本脚本直接拉那几天、走**引擎自己的 `store.upsert`** 合并进年分区 —— 2 次请求搞定。

    # 先看要补什么（默认 dry-run，不发写操作）
    python scripts/fill_gap.py -d index_ths_daily --dates 2026-09-08,2026-09-09

    # 确认后真补
    python scripts/fill_gap.py -d index_ths_daily --dates 2026-09-08,2026-09-09 --apply

    # 从 audit.py 的结果自动取缺口（按数据集）
    python scripts/fill_gap.py -d index_daily --from-audit /tmp/audit3.json --apply

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


def running(pattern: str) -> list[str]:
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if pattern in cmd and "fill_gap" not in cmd:
            out.append(p.name)
    return out


def _prev_day(s: str) -> str:
    from datetime import date, timedelta
    return (date.fromisoformat(str(s)[:10]) - timedelta(days=1)).isoformat()


def dates_from_audit(path: str, dataset: str) -> list[str]:
    """从 audit.py 的 --json 结果里取某个数据集的缺失日期。"""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[str] = []
    for f in obj.get("findings", []):
        if f["dataset"] != dataset:
            continue
        det = f.get("detail")
        if isinstance(det, dict) and det.get("missing_days"):
            out.extend(det["missing_days"])
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("--dates", help="逗号分隔的 YYYY-MM-DD")
    ap.add_argument("--from-audit", help="从 audit.py 的 --json 结果里取缺失日")
    ap.add_argument("--apply", action="store_true", help="真的写（默认 dry-run）")
    ap.add_argument("--force", action="store_true", help="跳过「有下载进程在跑」检查")
    ap.add_argument("--trim-coverage-from", metavar="YYYY-MM-DD",
                    help="把 coverage 从这个日期起裁掉，然后让增量逻辑把该区间当缺口重下。"
                         "★ 这是 QUANT_PLATFORM.md §16.3 问题 1 规定的修法（**不要用 --full**，"
                         "--full 会先 drop 分区，中途中断就真丢数据）。裁完自动跑 main.py run。")
    args = ap.parse_args()

    from lingqi import spec as S
    from lingqi import store
    from lingqi.engine import Engine
    from lingqi.manifest import Manifest

    procs = running("main.py run") + running("catchup_after_main")
    if procs and args.apply and not args.force:
        print(f"✖ 有下载进程在跑（PID {', '.join(procs)}），拒绝 --apply。等它结束再加 --force 硬来。")
        return 1

    spec = S.get(args.dataset)
    if not spec.enabled:
        print(f"⚠️ {args.dataset} 在 spec.py 里是 enabled=False —— 确认你真的要给它补数据？")

    # ---------------------------------------------------------------- 模式二
    # 裁 coverage → 让增量逻辑把该区间当缺口重下（§16.3 问题 1 的修法）
    if args.trim_coverage_from:
        cutoff = args.trim_coverage_from
        st = STATE / f"{args.dataset}.json"
        if not st.exists():
            print(f"✖ {st} 不存在")
            return 1
        man = json.loads(st.read_text(encoding="utf-8"))
        cov = man.get("coverage") or []
        kept, dropped = [], []
        for a, b in cov:
            if b < cutoff:
                kept.append([a, b])
            elif a >= cutoff:
                dropped.append([a, b])
            else:                       # 跨越 cutoff 的段：裁到 cutoff 前一天
                kept.append([a, _prev_day(cutoff)])
                dropped.append([cutoff, b])
        print(f"coverage: {len(cov)} 段 → 保留 {len(kept)} 段，裁掉 {len(dropped)} 段（{cutoff} 起）")
        for x in dropped[:6]:
            print(f"    裁掉 {x[0]} ~ {x[1]}")
        if len(dropped) > 6:
            print(f"    … 共 {len(dropped)} 段")
        if not args.apply:
            print("\n以上是预览（dry-run）。加 --apply 执行裁剪并跑增量。")
            return 0
        bak = st.with_suffix(".json.trimcov.bak")
        if not bak.exists():
            shutil.copy2(st, bak)
            print(f"state 已备份 → {bak.name}")
        man["coverage"] = kept
        man["suspect"] = {}             # 该区间要重下了，清掉旧的可疑标记
        st.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        print("✔ coverage 已裁。现在跑增量让它重下该区间…\n")
        import subprocess
        rc = subprocess.call([sys.executable, str(ROOT / "main.py"), "run", args.dataset])
        print(f"\n增量退出码 {rc}。建议再跑 `python scripts/audit.py -d {args.dataset}` 复核。")
        return rc

    dates: list[str] = []
    if args.dates:
        dates += [x.strip() for x in args.dates.split(",") if x.strip()]
    if args.from_audit:
        got = dates_from_audit(args.from_audit, args.dataset)
        print(f"从 {args.from_audit} 取到 {len(got)} 个缺失日")
        dates += got
    dates = sorted(set(dates))
    if not dates:
        print("没有指定要补的日期（--dates 或 --from-audit）")
        return 1

    print(f"数据集: {args.dataset}  模式: {spec.mode}  主键: {spec.keys}")
    print(f"待补 {len(dates)} 天: {dates[0]} … {dates[-1]}")
    print()

    cfg = yaml.safe_load(open(ROOT / "conf" / "config.yaml", encoding="utf-8"))
    # ★ 用引擎自己的 _do_fetch，不要直接 client.call —— 后者返回的是**信封字典**
    #   {total, list}，而且不做分页。第一版就是直接 call 然后 extend(dict)，
    #   结果迭代了字典的 key，2 天的数据只补进来 2 行（'total' 和 'list'）。
    #   _do_fetch 内部做了 extract_list + 分页，和正式下载走的是同一条路径。
    engine = Engine(cfg, ROOT)

    # ---- 抓取 ----
    payloads = []
    if spec.mode in ("range", "per_date"):
        if spec.mode == "per_date":
            payloads = [{**spec.params, spec.date_param: x} for x in dates]
        else:
            # range 模式：按连续区间抓，能合并的合并（省请求）
            payloads = [{**spec.params, spec.start_param: dates[0], spec.end_param: dates[-1]}]
    else:
        payloads = [{**spec.params, spec.start_param: dates[0], spec.end_param: dates[-1]}]

    rows: list[dict] = []
    for p in payloads:
        got, err = engine._do_fetch(spec, p)
        if err:
            print(f"  ✖ 抓取失败: {err}")
            continue
        print(f"  请求 {p} → {len(got):,} 行")
        rows.extend(got)
    print(f"上游合计返回 {len(rows):,} 行")

    if not rows:
        print("✖ 上游返回 0 行 —— 说明这几天上游本来就没有数据，不是本地缺失，无需补。")
        return 0

    df = pd.DataFrame(rows)
    if spec.date_field in df.columns:
        have = sorted({str(x)[:10] for x in df[spec.date_field]})
        print(f"覆盖交易日: {have}")
        want = set(dates)
        missed = [x for x in dates if x not in set(have)]
        if missed:
            print(f"⚠️ 上游没有这几天的数据（会跳过）: {missed}")

    if not args.apply:
        years = sorted({int(str(v)[:4]) for v in df[spec.date_field]}) if spec.date_field in df.columns else []
        print(f"\n以上是预览（dry-run），未写入。涉及年分区: {years}")
        print("确认后加 --apply 执行。")
        return 0

    # ---- 写入（走引擎自己的 upsert）----
    st = STATE / f"{args.dataset}.json"
    if st.exists():
        bak = st.with_suffix(".json.fillgap.bak")
        if not bak.exists():
            shutil.copy2(st, bak)
            print(f"state 已备份 → {bak.name}")
    man = Manifest.load(STATE, args.dataset) if st.exists() else None

    written = 0
    for year in sorted({int(str(v)[:4]) for v in df[spec.date_field]}):
        sub = df[df[spec.date_field].astype(str).str[:4] == str(year)]
        path = store.year_partition_path(DATA, args.dataset, year)
        before = len(store.read_parquet(path))
        merged = store.upsert(path, sub, spec.keys, spec.sort_by or spec.keys)
        added = len(merged) - before
        written += added
        print(f"  year={year}: {before:,} → {len(merged):,} 行（新增 {added:+,}）")
        if man is not None:
            col = spec.date_field
            ds = sorted({str(x)[:10] for x in merged[col]})
            man.mark_partition(year, len(merged), ds[0] if ds else None, ds[-1] if ds else None)
    if man is not None:
        man.save()
        print("manifest 已更新（partitions 行数/日期范围）")

    print(f"\n✔ 补入 {written:,} 行。建议再跑一次 `python scripts/audit.py -d {args.dataset}` 复核。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
