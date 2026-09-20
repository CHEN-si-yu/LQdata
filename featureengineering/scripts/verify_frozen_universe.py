#!/usr/bin/env python
"""重建后验收：**逐分区证明产物里只有冻结池的 2115 只**（用户 2026-09-18 交办"删冗余数据"）。

为什么需要它
----------
池子从 3485 缩到 2115 之后，`store.upsert_year` 的语义是
「用新数据替换**同键**旧行、其余原样保留」——
**被移出池子的那 1370 只股票的老行不会自动消失**。
按年分块重建时每年都被整分区重算替换，理论上不留残留；
但"理论上"不算证据，必须**逐分区实测**。

三条判据
-------
  A. **无池外代码**（硬错误）：每个年分区里 `stock_code` 的去重集合 ⊆ 冻结名单。
     池外代码 = 残留。这是"删冗余数据"是否做干净的唯一硬判据。
  B. **逐日行数不超「面板宽度」**（硬错误）：某日行数 ≤ 冻结池只数（2115）。
     ⚠️ 上界是**面板宽度**，不是"当日已上市的只数" —— 产物按设计**每天都写满整块面板**
     （未上市/无数据的格子落 NaN 行）。2026-09-18 第一版把上界写成"当日已上市的只数"，
     一次误报 1902 条（2012 年某天写 2115 行是对的，而当日只上市了 1473 只）。
  C. **池内覆盖**（信息项，非错误）：该因子全部年份出现过的池内代码数 / 2115。
     偏低不一定是错（起点晚、结构性稀疏都正常），列出来供人工判断。

★ 判据 A 才是"有没有池外残留"的**唯一硬判据**（残留行的代码必然在池外）；
  判据 B 是防"分区被写坏"的合理性上界。

用法
----
    python scripts/verify_frozen_universe.py                # 全量核验（读所有分区，慢）
    python scripts/verify_frozen_universe.py --sample 20    # 随机抽查 20 个因子
    python scripts/verify_frozen_universe.py --factors adx_14 chip_concentration

★ 只读，零写入。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_frozen() -> tuple[list[str], np.ndarray, np.ndarray]:
    """返回 `(代码列表, 上市日数组, 排序后的上市日数组)`（用于 searchsorted 算上限）。"""
    from fea import config as cfgmod
    from fea.universe import frozen_codes
    cfg = cfgmod.load()
    codes = frozen_codes(cfg)
    if codes is None:
        raise SystemExit("✘ conf 里没有启用 frozen_list，本脚本无意义")
    ld: dict[str, str] = {}
    for ln in cfg.frozen_universe.read_text(encoding="utf-8").splitlines():
        if not ln or ln.startswith("#"):
            continue
        p = ln.split("\t")
        if len(p) >= 2:
            ld[p[0]] = p[1]
    codes_l = codes.tolist()
    lds = np.array([ld.get(c, "1900-01-01") for c in codes_l])
    return codes_l, lds, np.sort(lds)


def check_factor(root: Path, name: str, want: set[str], sorted_ld: np.ndarray) -> tuple[list[str], int]:
    """核验单个因子。返回 `(问题列表, 出现过的池内代码数)`。

    ★ 性能（2026-09-18 实测踩到）：第一版对**每个日期**都做一次
      `(dates == dt).sum()` 全数组比较 —— 243 天 × 50 万行 = **每分区 1.2 亿次比较**，
      实测 26 秒/因子 ⇒ 311 个因子要 2 小时+。
      改用 pyarrow 原生 `pc.unique` / `pc.value_counts`（一次 O(n log n) 算完），
      并把 `to_pylist()` 限制在小结果集上（去重后的 2115 个代码 / 243 个日期），
      避免对 50 万行做 Python 字符串转换。
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    d = root / name
    if not d.is_dir():
        return [f"{name}: 目录不存在"], 0

    bad: list[str] = []
    seen: set[str] = set()
    for part in sorted(d.glob("year=*/data.parquet")):
        year = part.parent.name.split("=")[1]
        t = pq.read_table(part, columns=["trade_date", "stock_code"])

        uniq = set(pc.unique(t.column("stock_code")).to_pylist())   # 判据 A
        seen |= uniq
        extra = uniq - want
        if extra:
            bad.append(f"{name} {year}: **池外代码 {len(extra)} 个** "
                       f"(例 {sorted(extra)[:5]})")

        vc = pc.value_counts(t.column("trade_date"))                # 判据 B
        dts = vc.field("values").to_pylist()
        cnts = vc.field("counts").to_pylist()
        for dt_raw, n in zip(dts, cnts):
            dt = str(dt_raw)[:10]
            # ★ 2026-09-18 修正：判据 B 原来写的是「当日**已上市**的池内只数」，
            #   那是**错的** —— 产物按设计**每天都写满整块面板**（未上市/无数据的格子
            #   落 NaN 行，见 CLAUDE.md「NaN 还是落盘的好」）。实测 2012 年某天写
            #   2115 行是**正确**的，而"当日已上市"只有 1473 只 ⇒ 一次误报 1902 条。
            #   正确的上界就是**面板宽度 = 冻结池只数**。
            if int(n) > len(want):
                bad.append(f"{name} {year} {dt}: 行数 {n} > 面板宽度 {len(want)}")
                break
    return bad, len(seen)


def main() -> int:
    ap = argparse.ArgumentParser(description="核验产物是否只剩冻结池的股票")
    ap.add_argument("--factors", nargs="*", default=None, help="只查这些因子")
    ap.add_argument("--sample", type=int, default=0, help="随机抽查 N 个因子（0=全查）")
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--root", default=None,
                    help="因子产物根目录（默认 = conf 的 factors_dir；用于核验沙箱产物）")
    a = ap.parse_args()

    codes_l, _lds, sorted_ld = load_frozen()
    want = set(codes_l)
    if a.root:
        root = Path(a.root)
    else:
        from fea import config as cfgmod
        root = Path(cfgmod.load().factors_dir)
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    if a.factors:
        names = [n for n in names if n in set(a.factors)]
    elif a.sample:
        random.Random(a.seed).shuffle(names)
        names = sorted(names[:a.sample])
    print(f"冻结池：{len(want)} 只 · 待核验因子 {len(names)} 个 "
          f"（{'抽查' if a.sample else '全查'}）\n")

    problems: list[str] = []
    cov: list[tuple[int, str]] = []
    for i, n in enumerate(names, 1):
        if i % 25 == 0 or i == len(names):
            print(f"  [{i}/{len(names)}] {n}", flush=True)
        bad, seen = check_factor(root, n, want, sorted_ld)
        problems.extend(bad)
        cov.append((seen, n))

    print()
    lo = sorted(cov)[:8]
    print(f"判据 C（信息项）池内覆盖最低的 8 个：")
    for seen, n in lo:
        print(f"    {n:34} {seen:5}/{len(want)}  ({seen/len(want):.0%})")

    print()
    if problems:
        print(f"✘ 发现 {len(problems)} 个问题：")
        for p in problems[:40]:
            print("   " + p)
        if len(problems) > 40:
            print(f"   …另有 {len(problems)-40} 条")
        return 1
    print(f"✅ {len(names)} 个因子全部通过：**无池外代码**、逐日行数不超上限")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
