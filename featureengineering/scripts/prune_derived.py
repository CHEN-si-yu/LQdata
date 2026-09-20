#!/usr/bin/env python3
"""把 `data/derived/` 对齐到「当前只产出某一年」的策略：删掉不需要的年份分区。

## 背景（用户 2026-09-15 要求）

`conf/config.yaml` 的 `default_start` 已经收到 **2026-01-01**，因子只产出 2026 年。
但派生层（`data/derived/intraday` 2010–2026、`data/derived/chips` 2018–2026）
还留着全部历史 —— 那是上一轮跑全历史时建出来的，占 ~2.1 GB，与本轮的产出范围不一致。

## 必须知道的副作用（别以为删了就永远没了）

派生层的 `ensure()` 是**自愈**的：`panel()` 发现某年的缓存缺失/指纹变了就**当场重建**。
所以删掉某一年之后：

- 因子 warmup 触及该年时，那一年会被**自动重建**（这是设计，不是 bug）。
  2026 年的因子 warmup 最多 60 个日历天 → 面板起点约在上一年 11 月
  → **上一年（2025）必然会被重建回来**（intraday 约 30~60 秒 / 95 MB）。
- 更早的年份（2010–2024）不会重建，除非你把 `default_start` 改回去做全历史。

也就是说：**删掉 2010–2024 是永久生效的；2025 会在下一次 run 时回来**，
它相当于"2026 输出的 warmup 带"，属于必要数据。想连它也删，就得接受
2026 年头 1~2 个月的日内/筹码因子变成 NaN（窗口不足）——目前没做这个取舍。

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/prune_derived.py            # 预览（默认 dry-run）
    $PY scripts/prune_derived.py --apply    # 真删
    $PY scripts/prune_derived.py --keep 2026 --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description="把派生层裁剪到指定年份")
    ap.add_argument("--keep", type=int, nargs="+", default=None,
                    help="保留哪些年份（默认 = conf/config.yaml 的 default_start 年份；"
                         "★ 通常还要带上**上一年**，它是 warmup 带，见文件头说明）")
    ap.add_argument("--apply", action="store_true", help="真的删（默认只预览）")
    args = ap.parse_args()

    from fea.config import load as load_cfg
    cfg = load_cfg()
    keep = {int(y) for y in (args.keep or [int(str(cfg.default_start)[:4])])}
    base = Path(cfg.root) / "data" / "derived"

    print(f"派生层裁剪：保留 year={sorted(keep)}，目录 {base}")
    print("-" * 78)
    freed = 0
    for layer in sorted(p for p in base.iterdir() if p.is_dir()):
        man_path = layer / "_manifest.json"
        man = json.loads(man_path.read_text()) if man_path.exists() else {"years": {}}
        drop = [d for d in sorted(layer.glob("year=*"))
                if d.is_dir() and int(d.name.split("=", 1)[1]) not in keep]
        if not drop:
            print(f"{layer.name:<12} 已是 {sorted(keep)}，无需裁剪")
            continue
        size = sum(_dir_size(d) for d in drop)
        freed += size
        print(f"{layer.name:<12} 删除 {len(drop)} 个分区 "
              f"({', '.join(d.name.split('=')[1] for d in drop)}) · {size/1e6:.0f} MB")
        if args.apply:
            for d in drop:
                shutil.rmtree(d)
            man["years"] = {k: v for k, v in man.get("years", {}).items()
                            if int(k) in keep}
            man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2))
    print("-" * 78)
    print(f"可释放 {freed/1e6:.0f} MB" + ("（已执行）" if args.apply else "（dry-run，未改动）"))
    if not args.apply:
        print("\n提示：`--apply` 才会真删。删完记得重跑 `main.py run` 让因子与派生层对齐。")
    else:
        print("\n★ 注意：下一次 `main.py run` 会自动重建 warmup 需要的上一年（见脚本头部说明）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
