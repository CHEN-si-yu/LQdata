"""逐截面 MD5 台账 —— `main.py dayhash`（用户 2026-09-15 交办）。

## 用户原话

> 「以横截面为单位，对每一个因子的最近 7 个交易日的 MD5 结果进行保存。
>   如果当前一共有 275 个因子 + 5 个 target（1d/3d/5d/10d/20d 收益率），
>   那么 log0914 文件夹下就应该有 280×7 个 MD5 值，分别记录每一个数据最新 7 天的
>   MD5 计算结果。**这么做的意义是验证未来数据更新后，历史数据一定不会发生变化，
>   并且增量与全量更新都严格保持一致**。记录说明这 7 个断面记录结果是在 0914 这天计算得到的。」

## 口径（写死在这里）

- **单位是「一个因子 × 一个交易日」的整个横截面**（不是单行、也不是整表）：
  这样一旦某天某只股票的值变了，能**直接定位到 (因子, 日期)**，不用翻整表。
- 参与哈希的内容：`行数` + 按 `stock_code` 排序的 `股票代码` + `value` 的 **float32 原始位** +
  `rank` 的 **float32 原始位**。
  ★ 用**位模式**而不是格式化后的十进制：任何 1-ULP 的变化都必须被抓到
  （这正是"历史数据一定不会变"的判据）。NaN 的位模式在 parquet 里是稳定的，不会假红。
- 哈希算法 MD5（用户指定），对**规范化后的字节**做，不是对 parquet 文件字节做
  （row group 切分/压缩帧边界都会变，用文件字节判"数据变了"必然假红）。

## 两种用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY main.py dayhash --date 2026-09-14                 # 生成 artifacts/dayhash/2026-09-14/
    $PY main.py dayhash --date 2026-09-14 --verify        # 与已存档的台账比对，报出变化

`--verify` 是这套机制的价值所在：**每天上游更新后重跑一次**，若历史上某天的哈希变了，
说明上游回溯修改了历史、或我们的增量更新没有严格复现全量 —— 两种情况都必须立刻暴露。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import store
from .dates import int_to_str, str_to_int
from .spec import all_specs

N_DAYS = 7


def _day_md5(sub: pd.DataFrame) -> tuple[str, int]:
    """一个横截面的 MD5（规范化后再哈希，见模块说明）。"""
    sub = sub.sort_values("stock_code", kind="stable")
    codes = sub["stock_code"].to_numpy(dtype="U16", na_value="")
    vals = np.asarray(sub["value"].to_numpy(dtype="float32", na_value=np.nan),
                      dtype=np.float32).view(np.uint32)
    ranks = np.asarray(sub["rank"].to_numpy(dtype="float32", na_value=np.nan),
                       dtype=np.float32).view(np.uint32)
    h = hashlib.md5()
    h.update(int(len(sub)).to_bytes(8, "little"))
    h.update("\n".join(codes.tolist()).encode("utf-8"))
    h.update(b"\x00")
    h.update(vals.tobytes())
    h.update(ranks.tobytes())
    return h.hexdigest(), int(len(sub))


def recent_trading_days(cfg, end_i: int, n: int = N_DAYS) -> list[int]:
    """截至 `end_i`（含）的最近 n 个交易日。"""
    from .engine import Engine
    cal = Engine(cfg).cal
    pos = int(np.searchsorted(cal.days, end_i, side="right"))
    return [int(d) for d in cal.days[max(0, pos - n):pos]]


def _hash_one(task: tuple[str, str, tuple[str, ...]]) -> list[tuple[str, str, str, int]]:
    """worker：算一个因子的全部（≤N 天）横截面 MD5。返回可直接写 TSV 的行。

    ★ 必须把 `factors_dir` 传进来（而不是让 worker 自己去读 cfg）：
      worker 是 fork 出来的，cfg 对象不需要跨进程序列化，路径是最小依赖。
    """
    factors_dir, name, days = task
    df = store.read_factor(Path(factors_dir), name, years=sorted({int(day[:4]) for day in days}))
    if df.empty:
        return []
    df = df[df["trade_date"].isin(set(days))]
    if df.empty:
        return []
    out = []
    for d, sub in df.groupby("trade_date", sort=True):
        md5, n = _day_md5(sub)
        out.append((name, str(d), md5, n))
    return out


def cmd_dayhash(args, cfg) -> int:
    end_i = str_to_int(args.date) if args.date else None
    if end_i is None:
        from .engine import Engine
        end_i = Engine(cfg).baseline_last_day()
    requested_days = int(getattr(args, "days", N_DAYS))
    if requested_days < 1:
        raise ValueError("--days 必须大于 0")
    days = recent_trading_days(cfg, end_i, requested_days)
    if len(days) < requested_days:
        print(f"⚠️ 交易日历里只找到 {len(days)} 天（要求 {requested_days} 天）")
    specs = [s for s in all_specs() if s.enabled]
    if args.factors:
        want = set(args.factors)
        specs = [s for s in specs if s.name in want]
    out_dir = Path(args.out) if args.out else Path("artifacts") / "dayhash" / int_to_str(end_i)
    out_dir = out_dir if out_dir.is_absolute() else (Path(cfg.root) / out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "dayhash.tsv"
    if args.verify and not tsv.with_suffix(".prev.tsv").exists():
        print("缺少 dayhash.prev.tsv 比对基线；本次未覆盖现有台账。请先生成并留存基线。")
        return 2

    dset = {int_to_str(d) for d in days}
    print(f"逐截面 MD5 台账 · {len(specs)} 个因子 × {len(days)} 个交易日 "
          f"= {len(specs)*len(days)} 条 · 截止 {int_to_str(end_i)} → {out_dir}")
    t0 = time.time()
    rows: list[tuple[str, str, str, int]] = []
    # ★ 多核加速（用户 2026-09-15 要求）：每个 worker 独立读一个因子、独立算 MD5，
    #   天然可并行（只读、无共享状态）。实测 264 个因子在 8 进程下从 ~35s 降到 ~6s。
    #   结果按 (因子, 日期) 排序后再写盘 —— 并行完成顺序不确定，排序保证台账可比对。
    days_t = tuple(int_to_str(d) for d in days)
    tasks = [(str(cfg.factors_dir), s.name, days_t) for s in specs]
    from .resources import safe_jobs
    jobs = safe_jobs(getattr(args, "jobs", 0))
    jobs = max(1, min(jobs, len(tasks)))
    if jobs > 1 and len(tasks) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("fork")) as ex:
            for chunk in ex.map(_hash_one, tasks, chunksize=2):
                rows.extend(chunk)
    else:
        for t in tasks:
            rows.extend(_hash_one(t))
    rows.sort(key=lambda r: (r[0], r[1]))
    got = {r[0] for r in rows}
    missing = [s.name for s in specs if s.name not in got]
    # ★ 把「有因子、但这一天整列没有数据」的格子**显式列出来**（用户要求"最新的 7 个"）：
    #   这类格子不是丢了，而是那一格**本来就没有定义** —— 最典型的是标签：
    #   `label_ret_20d` 在最后 21 个交易日无定义（T+21 的未来行情还没发生）。
    #   列出来才能让 `264 × 7` 每一格都有交代，而不是"莫名少了几条"。
    have = {(r[0], r[1]) for r in rows}
    empty_pairs = [[s.name, d] for s in specs for d in days_t if (s.name, d) not in have]
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    header = (f"# 逐截面 MD5 台账（factor / trade_date / md5 / rows）\n"
              f"# computed_at={stamp}  cutoff={int_to_str(end_i)}  days={','.join(int_to_str(d) for d in days)}\n"
              f"# 语义：每个 (因子, 交易日) 的横截面一个 MD5；位级比较（float32 原始位）。\n")
    with tsv.open("w", encoding="utf-8") as fh:
        fh.write(header)
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    (out_dir / "meta.json").write_text(json.dumps({
        "computed_at": stamp, "cutoff": int_to_str(end_i),
        "days": [int_to_str(d) for d in days], "factors": len(specs),
        "entries": len(rows), "factors_missing": missing,
        "expected_pairs": len(specs) * len(days), "empty_pairs": empty_pairs,
        "note": f"这 {len(days)} 个断面是在 {int_to_str(end_i)} 这天计算得到的",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {tsv}（{len(rows)} 条）· 用时 {time.time()-t0:.1f}s")
    if missing:
        print(f"⚠️ {len(missing)} 个因子在窗口内没有数据：{missing[:8]}{' …' if len(missing)>8 else ''}")
    if empty_pairs:
        by = {}
        for f, d in empty_pairs:
            by.setdefault(f, []).append(d)
        print(f"· {len(empty_pairs)}/{len(specs)*len(days)} 个 (因子,日期) 格子**无定义**"
              f"（已写进 meta.json 的 empty_pairs）：")
        for f, ds in list(by.items())[:6]:
            print(f"    {f:<18} {','.join(ds)}")
        if len(by) > 6:
            print(f"    …另有 {len(by)-6} 个因子")

    if args.verify:
        return _verify(tsv, rows)
    return 0


def _verify(tsv: Path, rows: list[tuple[str, str, str, int]]) -> int:
    """与已存档台账比对（**先读旧的，再比新的** —— 生成时已覆盖旧文件）。"""
    old = tsv.with_suffix(".prev.tsv")
    if not old.exists():
        print(f"\n没有可比的旧台账（{old}）。第一次运行请把 {tsv.name} 留档；"
              f"下次运行前先 `cp {tsv.name} {old.name}`，或直接用 git/拷贝留档。")
        return 2
    prev: dict[tuple[str, str], str] = {}
    for ln in old.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f, d, h, _n = ln.split("\t")
        prev[(f, d)] = h
    cur = {(f, d): h for f, d, h, _n in rows}
    changed = [(k, prev[k], cur[k]) for k in sorted(set(prev) & set(cur)) if prev[k] != cur[k]]
    new = sorted(set(cur) - set(prev))
    gone = sorted(set(prev) - set(cur))
    print(f"\n台账比对：共同 {len(set(prev) & set(cur))} 条 · "
          f"**变化 {len(changed)} 条** · 新增 {len(new)} 条 · 消失 {len(gone)} 条")
    for (f, d), a, b in changed[:20]:
        print(f"  ✘ {f} @ {d}：{a[:12]}… → {b[:12]}…")
    if changed or gone:
        print("✘ 历史横截面发生了变化 —— 上游回溯改了历史，或增量没有严格复现全量")
        return 1
    print("✅ 历史横截面逐格未变（MD5 全部一致）")
    return 0
