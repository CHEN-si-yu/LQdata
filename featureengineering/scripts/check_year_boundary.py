#!/usr/bin/env python3
"""跨年连续性筛查 —— 年份边界处的「截面换手」是不是离群值（只看产物，零重算）。

## 为什么要查这个

落盘按年分区，但**因子的时间轴必须连续**：本年第一格往前看 `warmup_days` 天，
年与年之间靠这段 warmup 重叠。若某个因子的实际回看窗口 > 声明的 warmup，
它就会在**每年年初**露出断点（值突变 / 成片 NaN）。

判据（与 `scripts/check_anchor_warmup.py` 的 A/B 互补：那个是"抽样的严格证明"，
这个是"全量的宽扫"）：

    churn(t) = 相邻两个交易日的**截面 rank 平均绝对变化**（只取两天都在池内的股票）
    ratio   = churn(每年第一个交易日) / median(churn(全年))

时间轴连续时，年初就是普通一天 ⇒ ratio ≈ 1；断了 ⇒ ratio 显著 >1。

⚠️ 这条是**筛查**不是判决：分红/调仓/季节性本身会让某些因子在年初跳一下
（`ratio` 到几倍是正常的）。**离群（几十倍以上）才要逐条看**。

用法：
    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/check_year_boundary.py                    # 全部因子
    $PY scripts/check_year_boundary.py --factors bp etp5  # 指定因子
    $PY scripts/check_year_boundary.py --years 2016 2024  # 限定年份
    $PY scripts/check_year_boundary.py --top 30           # 只列最离群的 N 个
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fea import config as cfg_mod            # noqa: E402
from fea import store                        # noqa: E402
from fea.spec import all_specs               # noqa: E402

# ★ 必须 import factors：注册表靠 `factors/__init__.py` 里的 import 装填，
#   不导入的话 `all_specs()` 是空的 —— 实测报「筛查 0 个因子 · 没有可筛查的产物」（2026-09-17）。
import factors                               # noqa: F401,E402


def _one(name: str, factors_dir: Path, years: list[int]) -> list[tuple]:
    """一个因子的逐年 churn 统计。返回 [(因子, 年, 年初日, 年初churn, 全年中位churn, ratio)]。"""
    parts = []
    for y in years:
        p = store.year_path(factors_dir, name, y)
        if not p.exists():
            continue
        try:
            parts.append(pd.read_parquet(p, columns=["trade_date", "stock_code", "rank"]))
        except Exception:
            continue
    if not parts:
        return []
    df = pd.concat(parts, ignore_index=True)
    df = df[df["rank"].notna()]
    if df.empty:
        return []
    # (日期, 股票) -> rank 的宽表；用因子值算 churn 会被量纲带偏，rank 是天然可比的
    wide = df.pivot_table(index="trade_date", columns="stock_code",
                          values="rank", aggfunc="last")
    days = sorted(wide.index)
    out = []
    for i in range(1, len(days)):
        a = wide.iloc[i - 1].to_numpy(np.float64)
        b = wide.iloc[i].to_numpy(np.float64)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 100:
            continue
        d = np.abs(b[m] - a[m]).mean()
        out.append((str(days[i])[:4], b[m].size, d))
    if not out:
        return []
    churn = pd.DataFrame(out, columns=["year", "n", "churn"])
    med_all = float(churn["churn"].median())
    res = []
    for y, g in churn.groupby("year"):
        first = g.iloc[0]
        res.append((name, int(y), float(first["churn"]),
                    med_all, float(first["churn"]) / med_all if med_all > 0 else np.nan,
                    int(first["n"])))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="跨年连续性筛查（只看产物，零重算）")
    ap.add_argument("factors", nargs="*")
    ap.add_argument("--years", nargs="*", type=int, default=None, help="限定年份（默认全部）")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--top", type=int, default=0, help="只列最离群的前 N 个（0 = 全部）")
    args = ap.parse_args()

    cfg = cfg_mod.load()
    names = args.factors or [s.name for s in all_specs() if s.enabled]
    years = args.years or list(range(2012, 2027))
    print(f"筛查 {len(names)} 个因子 · 年份 {years[0]}..{years[-1]} · 并行 {args.jobs}")

    t0 = time.time()
    rows: list[tuple] = []
    if args.jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("fork")) as ex:
            futs = [ex.submit(_one, n, cfg.factors_dir, years) for n in names]
            for f in futs:
                rows.extend(f.result())
    else:
        for n in names:
            rows.extend(_one(n, cfg.factors_dir, years))
    if not rows:
        print("没有可筛查的产物")
        return 1

    r = pd.DataFrame(rows, columns=["factor", "year", "churn_first", "churn_median",
                                    "ratio", "n_codes"])
    r = r.sort_values("ratio", ascending=False)
    worst = r.head(max(args.top or 30, 10))
    print("-" * 88)
    print(f"{'因子':<34}{'年':<7}{'年初churn':>11}{'全年中位':>11}{'倍数':>8}{'股票数':>8}")
    print("-" * 88)
    for _, x in worst.iterrows():
        flag = "  ← 看这里" if x["ratio"] >= 10 else ("  ← 偏高" if x["ratio"] >= 4 else "")
        print(f"{x['factor']:<34}{int(x['year']):<7}{x['churn_first']:>11.4f}"
              f"{x['churn_median']:>11.4f}{x['ratio']:>8.2f}{int(x['n_codes']):>8}{flag}")
    print("-" * 88)
    print(f"共 {len(r)} 个 (因子,年) 组合 · 倍数：中位 {r['ratio'].median():.2f} · "
          f"p95 {r['ratio'].quantile(0.95):.2f} · 最大 {r['ratio'].max():.2f}")
    n10 = int((r["ratio"] >= 10).sum())
    print(f"{'✔ 没有 ≥10 倍的离群（时间轴在年份边界处看不出断点）' if n10 == 0 else f'⚠ {n10} 个 (因子,年) 倍数 ≥10，需逐条定性'}"
          f"  用时 {time.time()-t0:.0f}s")
    return 0 if n10 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
