#!/usr/bin/env python
"""把 state/*.json 的 `partitions` 元数据按**磁盘实际内容**重新同步。

只改 `partitions`（行数 / min_date / max_date / updated_at），
**不动** coverage / done / suspect / columns / data_start —— 那些是引擎的语义状态，
不能让这个脚本替它做决定。

什么时候需要它：任何绕过引擎直接改过 parquet 的操作之后。例如
`test_increment.py` 的删除/还原、`fill_gap.py` 的直补、手工修数据。
不同步的话 `audit.py` 会报一堆"state 说 N 行、磁盘实际 M 行"。

    python scripts/sync_manifest.py            # 预览
    python scripts/sync_manifest.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
STATE = ROOT / "state"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("-d", "--dataset", action="append")
    args = ap.parse_args()

    from lingqi import spec as S
    from main import ordered_specs

    changed = 0
    for spec in [s for s in ordered_specs(None) if s.enabled]:
        if args.dataset and spec.name not in set(args.dataset):
            continue
        st = STATE / f"{spec.name}.json"
        dpath = DATA / spec.name
        if not st.exists() or not dpath.is_dir():
            continue
        man = json.loads(st.read_text(encoding="utf-8"))
        old = dict(man.get("partitions") or {})
        new: dict = {}
        col = spec.date_field
        for part in sorted(dpath.glob("year=*")):
            f = part / "data.parquet"
            if not f.exists():
                continue
            year = part.name.split("=", 1)[1]
            try:
                t = pq.read_table(f)
            except Exception:
                continue
            n = t.num_rows
            mn = mx = None
            if n and col in t.schema.names:
                vals = [str(x)[:10] for x in t.column(col).to_pylist() if x is not None]
                if vals:
                    mn, mx = min(vals), max(vals)
            new[year] = {"rows": n, "min_date": mn, "max_date": mx,
                         "updated_at": (old.get(year) or {}).get("updated_at")
                                       or datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        flat = dpath / "data.parquet"
        if flat.exists():
            t = pq.read_table(flat)
            n = t.num_rows
            mn = mx = None
            if n and col in t.schema.names:
                vals = [str(x)[:10] for x in t.column(col).to_pylist() if x is not None]
                if vals:
                    mn, mx = min(vals), max(vals)
            new["0"] = {"rows": n, "min_date": mn, "max_date": mx,
                        "updated_at": (old.get("0") or {}).get("updated_at")
                                      or datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        if new == old:
            continue
        changed += 1
        if args.apply:
            man["partitions"] = new
            st.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  ✔ {spec.name}: {len(old)} → {len(new)} 个分区已同步")
        else:
            print(f"  · {spec.name}: {len(old)} → {len(new)} 个分区（预览）")
            for y in sorted(set(old) | set(new)):
                o, n2 = old.get(y), new.get(y)
                if o != n2:
                    fo = o.get("rows") if o else "—"
                    fn = n2.get("rows") if n2 else "—"
                    do = o.get("max_date") if o else "—"
                    dn = n2.get("max_date") if n2 else "—"
                    print(f"      year={y}: 行 {fo} → {fn} | 最新 {do} → {dn}")

    print(f"\n{'已同步' if args.apply else '需要同步'} {changed} 个数据集"
          + ("" if args.apply else "（加 --apply 执行）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
