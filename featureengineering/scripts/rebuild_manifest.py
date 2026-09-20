#!/usr/bin/env python
"""从磁盘重建 manifest 的 `coverage` / `partitions`。

## 为什么需要它

`Engine.plan` 在 `rebuild=True` 时会 **`man.reset(recipe)`** —— 把 `coverage` 与
`partitions` 一起清空，再用 `--start` 限制本次产出范围：

    if rebuild or man.recipe != recipe:
        man.reset(recipe)
        spans = [(start_i, end_i)]

所以 `main.py run --rebuild --start 2026-01-01` 跑完之后，manifest 里**只剩 2026 一年**。
数据没丢（磁盘上 `year=2012..2025` 都在），但 `Engine.plan` 读的是 `coverage`，
它会认为 2012–2025 全缺 → `main.py` 的**多年守卫**把该因子整个摘出本次运行。

本脚本扫描 `data/factors/<因子>/year=*/data.parquet`，按分区的实际内容重建这两项；
其它字段（`recipe` / `input_watermark` / `last_run`）**原样保留**。

## 口径

- `rows`    = 分区行数
- `nonnull` = **`value` 列**的非空数（与 `prepare` 裁分区时用的口径一致）
- `min_date` / `max_date` = 该分区 `trade_date` 的最小 / 最大
- `coverage` = 把各分区的 `[min_date, max_date]` 逐段 `add_coverage()`（相邻自动合并）
  ⚠️ 年与年之间隔着非交易日（元旦/春节）时不会合并 —— 这与引擎原有的形态一致：
  `missing_ranges` 是**日历日**口径，那种"假缺口"在 `Engine.plan` 里会被
  `cal.between()` 筛成空，不会产出任何东西。

    python scripts/rebuild_manifest.py --factors adx_14    # 先试一个
    python scripts/rebuild_manifest.py --dry-run           # 只看会改成什么
    python scripts/rebuild_manifest.py                     # 全部 230 个
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fea import config as cfg_mod          # noqa: E402
from fea.manifest import Manifest          # noqa: E402


def scan_partitions(fdir: Path) -> tuple[dict, list[tuple[str, str]]]:
    """扫一个因子的所有 year 分区 → (partitions, spans)。"""
    parts: dict[str, dict] = {}
    spans: list[tuple[str, str]] = []
    for yd in sorted(fdir.glob("year=*"), key=lambda p: int(p.name.split("=")[1])):
        f = yd / "data.parquet"
        if not f.exists():
            continue
        # 只读这两列：value 是统计 nonnull 用的，trade_date 是区间用的
        df = pd.read_parquet(f, columns=["trade_date", "value"])
        rows = int(len(df))
        if rows == 0:
            continue
        nn = int(df["value"].notna().sum())
        td = df["trade_date"].astype(str).str.slice(0, 10)
        mn, mx = str(td.min()), str(td.max())
        parts[yd.name.split("=")[1]] = {
            "rows": rows, "nonnull": nn, "min_date": mn, "max_date": mx,
        }
        spans.append((mn, mx))
    return parts, spans


def main() -> int:
    ap = argparse.ArgumentParser(description="从磁盘重建 manifest 的 coverage/partitions")
    ap.add_argument("factors", nargs="*", help="只处理指定因子（默认全部）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写盘")
    ap.add_argument("--backup", action="store_true", default=True,
                    help="先把 state/*.json 备份到 artifacts/archive/backtest/（默认开）")
    args = ap.parse_args()

    cfg = cfg_mod.load()
    factors_dir, state_dir = Path(cfg.factors_dir), Path(cfg.state_dir)
    names = sorted(p.name for p in factors_dir.iterdir() if p.is_dir())
    if args.factors:
        want = set(args.factors)
        names = [n for n in names if n in want]
    if not names:
        print("没有匹配的因子")
        return 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    bk = Path(cfg.root) / "artifacts" / "archive" / "backtest" / f"state_meta_before_rebuild_{stamp}"
    if not args.dry_run and args.backup:
        bk.mkdir(parents=True, exist_ok=True)
        for n in names:
            src = state_dir / f"{n}.json"
            if src.exists():
                shutil.copy2(src, bk / src.name)
        print(f"manifest 已备份 → {bk}\n")

    print(f"{'因子':<34}{'年数':>4}{'分区行数':>14}{'非空率':>8}   区间")
    print("-" * 96)
    t0 = time.time()
    changed = 0
    for i, nm in enumerate(names, 1):
        man = Manifest.load(state_dir, nm)          # 保留 recipe / input_watermark / last_run
        parts, spans = scan_partitions(factors_dir / nm)
        old_years = len(man.partitions)
        rows = sum(p["rows"] for p in parts.values())
        nn = sum(p["nonnull"] for p in parts.values())
        if not args.dry_run:
            man.partitions = parts
            man.coverage = []
            for mn, mx in sorted(spans):
                man.add_coverage(mn, mx)
            man.save()
        changed += 1
        if i <= 3 or i % 50 == 0 or i == len(names):
            lo = min((s for s, _ in spans), default="—")
            hi = max((e for _, e in spans), default="—")
            ratio = f"{nn / rows * 100:.0f}%" if rows else "—"
            print(f"{nm:<34}{len(parts):>4}{rows:>14,}{ratio:>8}   {lo} → {hi}"
                  + (f"   [原 {old_years} 年]" if old_years != len(parts) else ""))
    print("-" * 96)
    print(f"{'（dry-run，未写盘）' if args.dry_run else '已重建'} {changed} 个因子 · "
          f"用时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
