#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全库因子质检采集：覆盖 / 波动 / 时序 / IC 五周期 / 冗余相关（一次扫描全出）。

## 为什么不让 `main.py eval` 干这件事

`fea/eval.py` 的口径是对的，但实现是**逐日 Python 循环 + `pd.qcut` + `np.unique`**：
实测 2 个因子 × 1 年 = 12.1 s，外推到 599 因子 × 9 年 ⇒ **约 5 小时**。
本脚本把同样的口径改成**行向量化**（截面统计全部用 `sum(axis=1)` 那一类），
并把「相关矩阵」这件 eval 不做的活并在同一趟里（复用 `fea.dedup` 的分块 Gram 口径）。
一趟扫描同时产出：质检指标 + 599² 相关矩阵 + 五周期 IC。

## 口径必须与既有实现逐位一致（否则质检结论不可比）

| 指标 | 对齐对象 | 说明 |
|:--|:--|:--|
| 覆盖率 / ac1 / IC / RankIC / 分层 | `fea/eval.py::evaluate_all` | 同样的有效样本掩码、同样的 `min_cross_section // 3` 门槛 |
| 相关矩阵 | `fea/dedup.py::_gram_years` | 落盘 `rank` 列 → 逐日截面 z-score → 长向量 Pearson，按年分块累加 `G = XᵀX` |
| 并列值名次 | pandas `rank()` 默认 `method="average"` | 本环境没有 scipy，eval 也是「先取秩再算 Pearson」 |

★ 有一处**故意不跟 eval 走**：eval 的 `cov` 分母是「当日 universe 股票数」，
本脚本同样用 `codes.size`（= 冻结池 2115）。两者一致，写在这里是为了留痕。

## 内存

- 年份外层、因子内层，逐年释放（与 `dedup._gram_years` 同一套做法）。
- 逐年块 `X (T·C × K) float32`：243 × 2115 × 599 × 4 B ≈ **1.24 GB**；
  相关矩阵 `G (K×K) float64` ≈ 2.9 MB；每因子的覆盖/取值数累加器合计 < 20 MB。
- 峰值在 GB 量级，远低于容器上限；脚本读 cgroup 上限只是为了**放宽 `RLIMIT_AS`**，
  不写死任何主机数字（不同机器的上限不一样）。

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/run_screen.py                          # 全库、全历史
    $PY scripts/run_screen.py --years 2025 2026        # 只扫这两年
    $PY scripts/run_screen.py bias_20 adx_14 --out /tmp/probe   # 小样本对拍

产物（脚本只落数据，报告文字由 Agent 写）：`screen_metrics.csv` / `screen_metrics.json` /
`corr.npy` / `corr_index.json`，默认写 `state/screen/`，可用 `--out` 重定向。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ★ BLAS 线程数必须在 import numpy **之前**设定（numpy/BLAS 在导入时读这些变量）。
#   因子侧的并行模型是多进程，单进程内的多线程只会互相踩踏。
_THREADS = os.environ.get("FEA_BLAS_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _THREADS


def _load_run_dedup():
    """复用 `scripts/run_dedup.py` 的两个资源工具函数。

    ★ 不复制粘贴：cgroup 上限的读法（v2/v1 回退）与「只放宽不收紧」的语义
      是 2026-09-21 那次地址空间耗尽事故的修复成果，抄一份出来迟早会走样。
       该模块顶层只设线程环境变量 + 插 sys.path，没有副作用。
    """
    p = ROOT / "scripts" / "run_dedup.py"
    spec = importlib.util.spec_from_file_location("_run_dedup", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RD = _load_run_dedup()
_relax_address_space = _RD._relax_address_space


def _anon_gib() -> float:
    """容器当前**不可回收**内存（GiB）。

    ★ 看 anon 而不是 `memory.current`：后者把可回收的页缓存也算进去，
      会把"缓存多"误读成"快爆了"（平台文档 §3.2 的同一条纪律）。
    """
    try:
        for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
            if line.startswith("anon "):
                return int(line.split()[1]) / 1024 ** 3
    except OSError:
        pass
    return float("nan")


def _rank_avg(x: "np.ndarray") -> "np.ndarray":
    """平均名次（1-based），等价 pandas `rank(method="average")`。

    ★ 必须处理并列：`limit_up_count_20` 这类零膨胀因子一天里大半是 0，
      用 argsort 强行排出先后会**凭空造出截面区分度**（`fea/panel.py::cs_rank`
      的注释里记着同一个坑）。本环境没有 scipy，eval 也是先取秩再算 Pearson。
    """
    import numpy as np
    order = np.argsort(x, kind="quicksort")
    xs = x[order]
    n = xs.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    newgrp = np.ones(n, dtype=bool)
    newgrp[1:] = xs[1:] != xs[:-1]
    starts = np.flatnonzero(newgrp)
    ends = np.append(starts[1:], n)
    # 1-based 名次：并列段 [s, e) 的平均名次 = (s+1 + e) / 2 = s + (e-s+1)/2
    avg = starts + (ends - starts + 1) / 2.0
    out = np.empty(n, dtype=np.float64)
    out[order] = np.repeat(avg, ends - starts)
    return out


def _pearson(a, b) -> float:
    import numpy as np
    a = a - a.mean()
    b = b - b.mean()
    d = float((a * a).sum()) * float((b * b).sum())
    return float((a * b).sum() / (d ** 0.5)) if d > 0 else float("nan")


def _daily_ic(V, L, min_n: int, want_rank: bool = True):
    """逐日截面 IC（Pearson on value）与 RankIC（Spearman，平均名次）。

    `_pearson` 的口径与 `fea/eval.py::_daily_corr` 逐字相同（含 `n*sx*sy` 分母）。
    这里保留逐日循环 —— 截面统计已经向量化了，剩下这个循环是**取秩**的固有成本。

    ★ `want_rank=False` 时**只算 Pearson**：取秩要 `argsort` 两次，是这趟扫描的绝对瓶颈
      （实测 599 因子 × 9 年：五周期全算 RankIC ≈ 75 min，只算 5d ≈ 20 min）。
      模型交付 head 是 1d，因此本轮同时算 1d、5d 的 RankIC；
      3d/10d/20d 保留 Pearson IC，不付额外取秩成本。
    """
    import numpy as np
    ics, rics = [], []
    for t in range(V.shape[0]):
        m = np.isfinite(V[t]) & np.isfinite(L[t])
        n = int(m.sum())
        if n < min_n:
            continue
        x = V[t][m]
        y = L[t][m]
        sx, sy = x.std(), y.std()
        if not (sx > 0 and sy > 0):
            continue
        dx, dy = x - x.mean(), y - y.mean()
        ics.append(float((dx * dy).sum() / (n * sx * sy)))
        if want_rank:
            rics.append(_pearson(_rank_avg(x), _rank_avg(y)))
    return ics, rics


def _deciles(V, L, min_n: int):
    """十分位分层平均收益（用名次切层，避免并列值把某一层撑爆）。

    与 `fea/eval.py::_deciles` 的 `qcut(..., duplicates="drop")` 等价到分层边界；
    差别只在并列值的归属，下游只用来算 D9−D0 与单调性，不影响判定。
    """
    import numpy as np
    acc = np.zeros(10)
    cnt = np.zeros(10)
    for t in range(V.shape[0]):
        m = np.isfinite(V[t]) & np.isfinite(L[t])
        n = int(m.sum())
        if n < min_n:
            continue
        x = V[t][m]
        y = L[t][m]
        r = _rank_avg(x)
        q = np.minimum(((r - 1.0) / n * 10).astype(np.int64), 9)
        for k in range(10):
            s = q == k
            if s.any():
                acc[k] += y[s].mean()
                cnt[k] += 1
    return acc, cnt


def _score_stats(seq):
    """逐日相关序列的描述统计；年、分期、全历史共用同一口径。"""
    import numpy as np

    a = np.asarray(seq, dtype=np.float64)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "icir": float("nan"), "t": float("nan"), "n": 0}
    m = float(a.mean())
    s = float(a.std(ddof=1)) if a.size > 1 else float("nan")
    return {"mean": m, "std": s,
            "icir": (m / s if s and s > 0 else float("nan")),
            "t": (m / (s / a.size ** 0.5) if s and s > 0 else float("nan")),
            "n": int(a.size)}


def _decile_stats(parts):
    """汇总若干年的十分位收益和有效天数。"""
    import numpy as np

    acc = np.sum([p[0] for p in parts], axis=0) if parts else np.zeros(10)
    cnt = np.sum([p[1] for p in parts], axis=0) if parts else np.zeros(10)
    d = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return {
        "spread": (float(d[9] - d[0]) if np.isfinite(d[0]) and np.isfinite(d[9])
                   else float("nan")),
        "mono": (_pearson(np.arange(10.0), d) if np.isfinite(d).all()
                 else float("nan")),
    }


def _stability_summary(rec: dict, all_years: list[int], horizons: list[int]):
    """在一趟扫描的逐年统计上生成年度和不重叠三年段报告。"""
    annual = {}
    for y in all_years:
        sy = str(y)
        if sy not in rec["years"]:
            continue
        annual[sy] = {
            "cov": rec["cov_year"][sy],
            "ic": {str(h): _score_stats(rec["ic_year"][h].get(sy, [])) for h in horizons},
            "rankic": {str(h): _score_stats(rec["ric_year"][h].get(sy, [])) for h in horizons},
            **_decile_stats([rec["dec_year"][sy]]),
        }

    periods = {}
    for i in range(0, len(all_years), 3):
        ys = [str(y) for y in all_years[i:i + 3] if str(y) in annual]
        if not ys:
            continue
        periods[f"{ys[0]}-{ys[-1]}"] = {
            "years": ys,
            "ic": {str(h): _score_stats([v for sy in ys
                                         for v in rec["ic_year"][h].get(sy, [])])
                   for h in horizons},
            "rankic": {str(h): _score_stats([v for sy in ys
                                             for v in rec["ric_year"][h].get(sy, [])])
                       for h in horizons},
            **_decile_stats([rec["dec_year"][sy] for sy in ys]),
        }
    return annual, periods


def _col_sig(col) -> tuple:
    """一列的「内容指纹」：长度 + 原始 buffer 的哈希（不解码成 Python 字符串）。

    ★ 为什么需要它：每个因子年的 `(trade_date, stock_code)` 网格**完全相同**
      （引擎用同一个 Panel 写盘，`main.py check` 也逐因子验过日期/股票轴），
      所以这两列每个因子都重新「字符串 → numpy」是纯浪费 ——
      实测 599 因子 × 9 年里，`series_to_int` 85 ms + `codes_idx.reindex` 158 ms
      占了单因子年耗时的一半以上。这里只哈希**原始字节**（约 6 MB，毫秒级），
      既拿到「网格没变」的**精确**证据，又不付解码的代价。
    """
    a = col.combine_chunks()
    h = hashlib.blake2b(digest_size=16)
    for b in a.buffers()[1:]:                 # [0] 是 validity，全有效时为 None
        if b is not None:
            h.update(b)
    return (len(a), h.hexdigest())


def _uniq_per_day(V) -> "np.ndarray":
    """每日截面的**不同取值个数**（向量化：排序后数相邻变化）。

    用来抓「卡片化」因子 —— 每天只有寥寥几个取值，分辨率不足以做排序。
    """
    import numpy as np
    S = np.sort(V, axis=1)
    fin = np.isfinite(S)
    # 相邻不等的位置计数 + 每行首个有限值算 1 个
    neq = np.zeros(S.shape, dtype=bool)
    neq[:, 1:] = fin[:, 1:] & (S[:, 1:] != S[:, :-1]) & fin[:, :-1]
    neq[:, 0] = fin[:, 0]
    return neq.sum(axis=1)


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser(description="全库因子质检采集（覆盖/波动/时序/IC/相关）")
    ap.add_argument("factors", nargs="*", help="只检查指定因子（默认全库股票因子）")
    ap.add_argument("--years", nargs=2, type=int, default=None, metavar=("Y0", "Y1"))
    ap.add_argument("--horizons", default="1,3,5,10,20", help="IC 的标签周期")
    ap.add_argument("--out", default="", help="输出目录（默认 state/screen）")
    args = ap.parse_args()

    import factors                                       # noqa: F401  导入即完成注册
    from fea import config as cfg_mod
    from fea import store
    from fea.dates import series_to_int
    from fea.engine import Engine
    from fea.panel import Panel
    from fea.spec import all_specs

    cfg_mod.setup_logging(False)
    cfg = cfg_mod.load()
    _relax_address_space()

    horizons = [int(x) for x in str(args.horizons).split(",")]
    h_main = 5 if 5 in horizons else horizons[0]
    rank_horizons = {h_main, 1} & set(horizons)

    specs = [s for s in all_specs() if s.enabled and not s.is_label and not s.is_market]
    if args.factors:
        want = set(args.factors)
        specs = [s for s in specs if s.name in want]
    if not specs:
        print("没有匹配的因子")
        return 1

    engine = Engine(cfg)
    cal = engine.cal
    codes = engine.codes
    import pandas as pd
    codes_idx = pd.Series(np.arange(codes.size), index=codes)
    min_n = cfg.min_cross_section // 3
    last_day = engine.baseline_last_day()
    floor = int(str(cfg.default_start).replace("-", ""))

    have = {s.name: set(store.factor_years(cfg.factors_dir, s.name)) for s in specs}
    all_years = sorted({y for s in specs for y in have[s.name]})
    if args.years:
        all_years = [y for y in all_years if args.years[0] <= y <= args.years[1]]
    if not all_years:
        print("没有可检查的因子产物")
        return 1

    # 列集合先定死（与 dedup 同语义）：只在被请求年份里有产物的因子占一列
    names = [s.name for s in specs if have[s.name] & set(all_years)]
    K = len(names)
    pos = {nm: i for i, nm in enumerate(names)}
    meta = {s.name: s for s in specs}

    print(f"因子质检采集 · {K} 个股票因子 · {all_years[0]}~{all_years[-1]} · "
          f"IC 周期 {horizons} · 主周期 {h_main}d")
    print(f"BLAS 线程 {_THREADS} · 起始 anon {_anon_gib():.2f} GiB")

    # 五周期标签：**读已落盘的产物**（09-24 审核已逐日验过），不现算 ——
    # 省掉 PriceLayer 的加载与窗口拼接，且与 eval 的交叉校验点保持一致。
    label_dir = cfg.factors_dir

    G = np.zeros((K, K), dtype=np.float64)
    cov_blocks, uniq_blocks, csstd_blocks = [], [], []
    stat = {nm: {
        "rows": 0, "nonnull": 0, "n_days": 0,
        "v_n": 0, "v_sum": 0.0, "v_sq": 0.0,
        "zero_n": 0, "const_days": 0, "card_days": 0,
        "ac_sum": 0.0, "ac_n": 0,
        "dec_acc": np.zeros(10), "dec_cnt": np.zeros(10),
        "ic": {h: [] for h in horizons}, "ric": {h: [] for h in horizons},
        # 保留逐年序列的引用；全历史序列仍照旧输出，额外内存仅是引用数组。
        "ic_year": {h: {} for h in horizons},
        "ric_year": {h: {} for h in horizons},
        "dec_year": {}, "cov_year": {},
        "years": {},
    } for nm in names}

    t0 = time.time()
    for yi, y in enumerate(all_years):
        gdays = cal.between(int(f"{y}0101"), int(f"{y}1231"))
        if gdays.size == 0:
            continue
        panel = Panel(gdays, codes)
        # 与 eval 同口径的统计行：裁到 [default_start, baseline_last_day]
        sdays = gdays[(gdays >= floor) & (gdays <= last_day)]
        rows = panel.day_mask(sdays)
        if not rows.any():
            continue

        labs = {}
        missing_lab = []
        for h in horizons:
            df = store.read_year(label_dir, f"label_ret_{h}d", y)
            if df.empty:
                missing_lab.append(h)
                labs[h] = np.full(panel.shape, np.nan)
                continue
            ci = codes_idx.reindex(df["stock_code"].to_numpy()).to_numpy()
            keep = np.isfinite(ci)
            labs[h] = panel.place(
                ci[keep].astype(np.int64), series_to_int(df["trade_date"])[keep],
                pd.to_numeric(df["value"], errors="coerce").to_numpy()[keep])
        if missing_lab:
            print(f"  ⚠️ {y} 年缺标签分区：{missing_lab}（这些周期的 IC 该年记 0 天）")

        here = [(meta[nm], pos[nm]) for nm in names if y in have[nm]]
        Xb = np.zeros((gdays.size * codes.size, K), dtype=np.float32)
        # ★ 数组按「统计行」分配（= sdays 的长度）；起点在年内的因子只有一部分行参与，
        #   所以要按 `act` 掩码写回对应位置 —— 直接把长度不足的数组整列赋值会广播失败
        #   （实测：2019 年某个 6 月才起算的因子触发了 186 vs 244 的 ValueError）。
        #   `uniq_y` 用 −1 当「该行没参与」的哨兵（取值数不可能是负的），
        #   聚合时先按 >= 0 过滤，避免未参与的行被当成「常数日」拉低中位数。
        cov_y = np.full((int(rows.sum()), K), np.nan, dtype=np.float32)
        uniq_y = np.full((int(rows.sum()), K), -1, dtype=np.int16)
        csstd_y = np.full((int(rows.sum()), K), np.nan, dtype=np.float32)

        ref_sig = None
        ref_cols = ref_days = None
        for sp, j in here:
            path = store.year_path(cfg.factors_dir, sp.name, y)
            if not path.exists():
                continue
            tbl = pq.read_table(path, columns=["trade_date", "stock_code", "value", "rank"])
            rec = stat[sp.name]
            rec["rows"] += int(tbl.num_rows)
            if tbl.num_rows == 0:
                continue
            sig = (_col_sig(tbl.column("trade_date")), _col_sig(tbl.column("stock_code")))
            if ref_sig is None:
                ref_sig = sig                       # 本年第一个因子定基准
                # 只用这一次做「字符串 → 下标」的解码，全年的因子都复用它
                _df = store.read_year(cfg.factors_dir, sp.name, y)
                _ci = codes_idx.reindex(_df["stock_code"].to_numpy()).to_numpy()
                _keep = np.isfinite(_ci) & (_ci >= 0)
                ref_cols = _ci[_keep].astype(np.int64)
                ref_days = series_to_int(_df["trade_date"])[_keep]
                del _df, _ci, _keep
            if sig == ref_sig:
                # ★ 网格与基准逐字节相同 ⇒ 直接复用预计算的行列下标，跳过全部字符串解码
                cc = ref_cols
                d_i = ref_days
                v_raw = tbl.column("value").to_numpy(zero_copy_only=False).astype(np.float64)
                r_raw = tbl.column("rank").to_numpy(zero_copy_only=False).astype(np.float64)
            else:
                # 网格变了（理论上不该发生）—— 退回慢路径，并**明确告警**而不是静默错位
                print(f"  ⚠️ {sp.name}/{y} 的日期或股票轴与本年基准不一致，退回逐因子解码")
                df = store.read_year(cfg.factors_dir, sp.name, y)
                ci = codes_idx.reindex(df["stock_code"].to_numpy()).to_numpy()
                keep = np.isfinite(ci) & (ci >= 0)
                cc = ci[keep].astype(np.int64)
                d_i = series_to_int(df["trade_date"])[keep]
                v_raw = pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype=np.float64)[keep]
                r_raw = pd.to_numeric(df["rank"], errors="coerce").to_numpy(dtype=np.float64)[keep]

            V = panel.place(cc, d_i, v_raw)
            R = panel.place(cc, d_i, r_raw)

            # ---- 相关矩阵：与 dedup 逐字一致（rank 列 → 逐日 z-score → NaN 记 0）
            valid_r = np.isfinite(R)
            cnt = valid_r.sum(axis=1, keepdims=True)
            tot = np.where(valid_r, R, 0.0).sum(axis=1, keepdims=True)
            ss = np.where(valid_r, R * R, 0.0).sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = tot / np.maximum(cnt, 1)
                var = np.maximum(ss / np.maximum(cnt, 1) - mu * mu, 0.0)
                sd = np.sqrt(var)
                z = (R - mu) / np.where(sd > 0, sd, np.nan)
            Xb[:, j] = np.where(np.isfinite(z), z, 0.0).reshape(-1)

            # ---- 统计只取与 eval 同口径的那些行
            active = sdays >= sp.start_int(cfg)
            Vs = V[rows][active]
            Rs = R[rows][active]
            if Vs.shape[0] == 0:
                continue

            fin = np.isfinite(Vs)
            n_day = fin.sum(axis=1)
            cov = n_day / codes.size
            cov_y[active, j] = cov
            rec["n_days"] += int(Vs.shape[0])
            rec["cov_year"][str(y)] = {
                "mean": float(cov.mean()), "p10": float(np.quantile(cov, 0.10)),
                "active_days": int(Vs.shape[0]),
            }

            un = _uniq_per_day(Vs)
            uniq_y[active, j] = np.clip(un, 0, 32767).astype(np.int16)
            rec["const_days"] += int(((un <= 1) & (n_day >= min_n)).sum())
            rec["card_days"] += int((un <= 20).sum())

            vv = Vs[fin]
            rec["nonnull"] += int(vv.size)
            if vv.size:
                rec["v_n"] += int(vv.size)
                rec["v_sum"] += float(vv.sum())
                rec["v_sq"] += float((vv * vv).sum())
                rec["zero_n"] += int((vv == 0).sum())

            # 当日截面 std（value 本身，未 winsor）—— 「波动性低」的横截面口径
            m1 = np.where(fin, Vs, 0.0).sum(axis=1)
            m2 = np.where(fin, Vs * Vs, 0.0).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu2 = m1 / np.maximum(n_day, 1)
                v2 = np.maximum(m2 / np.maximum(n_day, 1) - mu2 * mu2, 0.0)
            cs = np.sqrt(v2)
            cs[n_day < min_n] = np.nan
            csstd_y[active, j] = cs

            # ac1：秩网格 1 日自相关（与 eval 同口径：**年内**相邻两行配对）
            a = Rs[1:]
            b = Rs[:-1]
            mm = np.isfinite(a) & np.isfinite(b)
            nac = int(mm.sum())
            if nac > 10:
                aa, bb = a[mm], b[mm]
                den = aa.std() * bb.std()
                if den > 0:
                    rec["ac_sum"] += float(((aa - aa.mean()) * (bb - bb.mean())).mean() / den) * nac
                    rec["ac_n"] += nac

            for h in horizons:
                ic, ric = _daily_ic(Vs, labs[h][rows][active], min_n,
                                    want_rank=(h in rank_horizons))
                rec["ic"][h].extend(ic)
                rec["ric"][h].extend(ric)
                rec["ic_year"][h][str(y)] = ic
                rec["ric_year"][h][str(y)] = ric
            acc, cnt2 = _deciles(Vs, labs[h_main][rows][active], min_n)
            rec["dec_acc"] += acc
            rec["dec_cnt"] += cnt2
            rec["dec_year"][str(y)] = (acc, cnt2)
            # fast path 没有因子 DataFrame；`df` 在此可能仍指最后一个标签分区。
            rec["years"][str(y)] = int(tbl.num_rows)
            del V, R, Vs, Rs

        G += Xb.T @ Xb
        del Xb
        cov_blocks.append(cov_y)
        uniq_blocks.append(uniq_y)
        csstd_blocks.append(csstd_y)
        el = time.time() - t0
        done = yi + 1
        eta = el / done * (len(all_years) - done)
        print(f"  · {y} 年完成（{len(here)} 个因子）· 已用 {el / 60:.1f} min · "
              f"预计剩余 {eta / 60:.1f} min · anon {_anon_gib():.2f} GiB", flush=True)

    # ------------------------------------------------------------------ 汇总
    COV = np.concatenate(cov_blocks, axis=0) if cov_blocks else np.zeros((0, K), np.float32)
    UNI = np.concatenate(uniq_blocks, axis=0) if uniq_blocks else np.zeros((0, K), np.int16)
    CST = np.concatenate(csstd_blocks, axis=0) if csstd_blocks else np.zeros((0, K), np.float32)

    rows_out = []
    for j, nm in enumerate(names):
        rec = stat[nm]
        sp = meta[nm]
        cov = COV[:, j]
        cov = cov[np.isfinite(cov)]
        un = UNI[:, j]
        un = un[un >= 0]                      # −1 = 该因子在该行没参与（年内起算）
        cs = CST[:, j]
        cs = cs[np.isfinite(cs)]

        n_days = rec["n_days"]
        vn = rec["v_n"]
        vmean = rec["v_sum"] / vn if vn else float("nan")
        vvar = rec["v_sq"] / vn - vmean * vmean if vn else float("nan")
        ac1 = rec["ac_sum"] / rec["ac_n"] if rec["ac_n"] else float("nan")
        D = np.where(rec["dec_cnt"] > 0, rec["dec_acc"] / np.maximum(rec["dec_cnt"], 1), np.nan)

        # 同一次年度扫描给出逐年和不重叠三年段；没有重读任何因子或标签文件。
        annual, periods = _stability_summary(rec, all_years, horizons)

        rows_out.append({
            "name": nm, "group": sp.group, "start": sp.resolved_start(cfg),
            "n_days": n_days, "rows": rec["rows"], "nonnull": rec["nonnull"],
            "cov_mean": float(cov.mean()) if cov.size else float("nan"),
            "cov_p10": float(np.quantile(cov, 0.10)) if cov.size else float("nan"),
            "value_std": float(vvar ** 0.5) if np.isfinite(vvar) and vvar > 0 else 0.0,
            "cs_std_med": float(np.median(cs)) if cs.size else float("nan"),
            "uniq_med": float(np.median(un)) if un.size else float("nan"),
            "const_frac": (rec["const_days"] / n_days) if n_days else float("nan"),
            "card_frac": (rec["card_days"] / n_days) if n_days else float("nan"),
            "zero_frac": (rec["zero_n"] / vn) if vn else float("nan"),
            "ac1": ac1,
            "spread": (float(D[9] - D[0]) if np.isfinite(D[0]) and np.isfinite(D[9]) else float("nan")),
            "mono": (_pearson(np.arange(10.0), D) if np.isfinite(D).all() else float("nan")),
            "ic": {str(h): _score_stats(rec["ic"][h]) for h in horizons},
            "rankic": {str(h): _score_stats(rec["ric"][h]) for h in horizons},
            "h_main": h_main,
            "years": rec["years"],
            "annual": annual,
            "periods": periods,
        })

    outdir = Path(args.out) if args.out else (Path(cfg.state_dir) / "screen")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "screen_metrics.json").write_text(
        json.dumps(rows_out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    d = np.sqrt(np.diag(G))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = G / np.outer(np.where(d > 0, d, np.nan), np.where(d > 0, d, np.nan))
    np.fill_diagonal(corr, 1.0)
    np.save(outdir / "corr.npy", corr.astype(np.float64))
    (outdir / "corr_index.json").write_text(
        json.dumps({"names": names, "years": all_years}, ensure_ascii=False), encoding="utf-8")

    # CSV：一行一因子；保留主周期和模型交付 head 的 1d RankIC。
    import csv
    hm = str(h_main)
    cols = ["name", "group", "start", "n_days", "cov_mean", "cov_p10", "value_std",
            "cs_std_med", "uniq_med", "const_frac", "card_frac", "zero_frac", "ac1",
            "ic_mean", "ic_t", "rankic_mean", "rankic_icir",
            "rankic_1d_mean", "rankic_1d_icir", "spread", "mono"]
    with (outdir / "screen_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(rows_out, key=lambda x: x["name"]):
            w.writerow([r["name"], r["group"], r["start"], r["n_days"],
                        r["cov_mean"], r["cov_p10"], r["value_std"], r["cs_std_med"],
                        r["uniq_med"], r["const_frac"], r["card_frac"], r["zero_frac"], r["ac1"],
                        r["ic"][hm]["mean"], r["ic"][hm]["t"],
                        r["rankic"][hm]["mean"], r["rankic"][hm]["icir"],
                        (r["rankic"].get("1") or {}).get("mean"),
                        (r["rankic"].get("1") or {}).get("icir"),
                        r["spread"], r["mono"]])

    el = time.time() - t0
    print("-" * 88)
    print(f"共 {len(rows_out)} 个因子 · 耗时 {el / 60:.1f} min · 峰值 anon {_anon_gib():.2f} GiB")
    print(f"报告已写入 {outdir}（screen_metrics.csv/json · corr.npy · corr_index.json）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
