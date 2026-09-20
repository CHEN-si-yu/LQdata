"""因子冗余检测 —— `python main.py dedup`。

## 为什么需要它

「取精华去糟粕」的最大一块不是"因子无效"，而是**同一个东西做了好几遍**。
实测 272 个因子里，22 个簇的 |ρ|≥0.95（共 48 个因子），其中"价格在区间中的位置"
一个维度就挤了 9 个（`bollinger_position_20`/`keltner_position_20`/`donchian_position_20`/
`chandelier_position`/`williams_r_14`/`rsi_14`/`cci_20`/`stoch_slow_k`/`close_location_20d`）。

在下游的**日横断面回归**里，这些共线因子只会互相抢显著性、放大过拟合，
不会带来新信息。所以要把它们聚成簇，每簇只留一个代表。

## 口径（写死在这里，避免各说各话）

- 用落盘的 **`rank` 列**（已是当日截面百分位）作为因子值 —— 与下游用法一致；
- 逐日做**截面 z-score**（去均值、除以标准差），再把每天拼成一个长向量；
- 因子 i 与 j 的相关 = `Σ x_i x_j / sqrt(Σ x_i² · Σ x_j²)`（NaN 记 0，即"缺失不参与"）。
  这与 Pearson 在同一批有效样本上的结果一致；低覆盖因子会被**保守地**低估相关
  （所以它只会漏报，不会误报"重复"）。
- **秩相同 = 严格零信息**：另算一个 `identical_frac`（两因子 rank 逐格相同的比例），
  ≥0.999 的直接判为"完全重复"（数学上保证：下游只用 rank 列）。
  典型来源是"每日减同一个常数"（如 `rs_60 = momentum_60 − 指数收益`：
  同一天所有股票减的是同一个数，**秩完全不变**）与单调变换（`log`/`sqrt`/`zscore`）。

## 输出

`state/dedup/report.json` + 屏幕上的簇清单。每簇给出：
成员、两两 |ρ| 的范围、**建议保留的代表**（覆盖率最高、其次 |RankIC| 最大）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import store
from .dates import series_to_int
from .spec import all_specs


def _load_eval(path: Path) -> dict[str, dict]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return {r["name"]: r for r in rows}
    except Exception:                                         # noqa: BLE001
        return {}


def _matrix(specs, cfg, engine, years: list[int]) -> tuple[np.ndarray, dict]:
    """把每个因子的 rank 列铺成 (n_obs, K) 的截面 z-score 矩阵。

    行 = (交易日 × 股票)，列 = 因子。返回 (X, meta)。
    """
    cal, codes = engine.cal, engine.codes
    codes_idx = pd.Series(np.arange(codes.size), index=codes)
    days = np.concatenate([cal.between(int(f"{y}0101"), int(f"{y}1231")) for y in years]) \
        if years else np.array([], dtype=np.int64)
    from .panel import Panel
    panel = Panel(days, codes)
    n_obs = days.size * codes.size
    names, cols, meta = [], [], {}
    for sp in specs:
        df = store.read_factor(cfg.factors_dir, sp.name, years)
        if df.empty:
            continue
        ci = codes_idx.reindex(df["stock_code"].to_numpy()).to_numpy()
        keep = np.isfinite(ci)
        # panel.dates 就是 days → place 出来的 (T, C) 天然对齐，不需要再选行
        v = panel.place(ci[keep].astype(np.int64), series_to_int(df["trade_date"])[keep],
                        pd.to_numeric(df["rank"], errors="coerce").to_numpy()[keep])
        # 逐日截面 z-score（NaN 记 0 = 缺失不参与相关）。
        # 手工算而不用 nanmean/nanstd：整日为空时会刷一屏 RuntimeWarning。
        valid = np.isfinite(v)
        cnt = valid.sum(axis=1, keepdims=True)
        tot = np.where(valid, v, 0.0).sum(axis=1, keepdims=True)
        ss = np.where(valid, v * v, 0.0).sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = tot / np.maximum(cnt, 1)
            var = np.maximum(ss / np.maximum(cnt, 1) - mu * mu, 0.0)
            sd = np.sqrt(var)
            z = (v - mu) / np.where(sd > 0, sd, np.nan)
        z = np.where(np.isfinite(z), z, 0.0)
        cols.append(z.reshape(-1).astype(np.float32))
        names.append(sp.name)
        meta[sp.name] = {"rows": int(np.isfinite(v).sum())}
    X = np.column_stack(cols) if cols else np.zeros((n_obs, 0), np.float32)
    return X, {"names": names, "meta": meta}


def _identical_frac(cfg, years, a_name: str, b_name: str) -> float:
    """两个因子 rank 逐格相同的比例（只在两者都有值的格子上算）。"""
    a = _rank_frame(cfg, a_name, years)
    b = _rank_frame(cfg, b_name, years)
    if a is None or b is None:
        return 0.0
    m = a.merge(b, on=["trade_date", "stock_code"], suffixes=("_a", "_b"))
    if m.empty:
        return 0.0
    x, y = m["rank_a"].to_numpy(), m["rank_b"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return 0.0
    return float(np.mean(np.isclose(x[ok], y[ok], atol=1e-6)))


_FRAME_CACHE: dict[str, pd.DataFrame | None] = {}


def _rank_frame(cfg, name: str, years):
    if name not in _FRAME_CACHE:
        df = store.read_factor(cfg.factors_dir, name, years)
        _FRAME_CACHE[name] = df[["trade_date", "stock_code", "rank"]] if not df.empty else None
    return _FRAME_CACHE[name]


def _cluster(corr: np.ndarray, names: list[str], thr: float) -> list[list[int]]:
    """把 |ρ| ≥ thr 的因子并成簇（并查集）。"""
    n = len(names)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) >= thr:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def cmd_dedup(args, cfg) -> int:
    from .engine import Engine
    engine = Engine(cfg)
    specs = [s for s in all_specs() if s.enabled and not s.is_label]
    if getattr(args, "factors", None):
        want = set(args.factors)
        specs = [s for s in specs if s.name in want]
    years = sorted({y for s in specs for y in store.factor_years(cfg.factors_dir, s.name)})
    if getattr(args, "years", None):
        years = [y for y in years if args.years[0] <= y <= args.years[1]]
    if not years:
        print("没有可检查的因子产物")
        return 1

    thr = float(getattr(args, "threshold", 0.95))
    print(f"\n因子冗余检测 · {len(specs)} 个因子 · {years[0]}~{years[-1]} · |ρ| ≥ {thr} 视为重复")
    print("（口径：落盘 rank 列 → 逐日截面 z-score → 长向量 Pearson；缺失记 0，保守估计）")
    X, meta = _matrix(specs, cfg, engine, years)
    names = meta["names"]
    if X.shape[1] < 2:
        print("可比较的因子不足 2 个")
        return 1
    G = X.T @ X                                            # (K,K) 分子
    d = np.sqrt(np.diag(G))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = G / np.outer(np.where(d > 0, d, np.nan), np.where(d > 0, d, np.nan))
    np.fill_diagonal(corr, 1.0)

    ev = _load_eval(Path(cfg.state_dir) / "eval" / "summary.json")
    groups = _cluster(corr, names, thr)

    rows = []
    print("-" * 100)
    for g in sorted(groups, key=len, reverse=True):
        members = sorted((names[i] for i in g),
                         key=lambda nm: (-(ev.get(nm, {}).get("cov") or 0),
                                         -abs(ev.get(nm, {}).get("rankic") or 0)))
        reps = []
        for i in g:
            for j in g:
                if i < j:
                    reps.append(abs(corr[i, j]))
        # 完全重复（rank 逐格相同 —— 数学上零信息，因为下游只用 rank）
        ident = []
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                f = _identical_frac(cfg, years, names[g[a]], names[g[b]])
                if f >= 0.999:
                    ident.append((names[g[a]], names[g[b]], round(f, 4)))
        keep = members[0]
        drop = members[1:]
        rows.append({"size": len(g), "keep": keep, "drop": drop,
                     "rho_min": round(float(min(reps)), 4),
                     "rho_max": round(float(max(reps)), 4),
                     "identical_pairs": ident})
        print(f"簇 {len(g):>2} 个：保留 `{keep}`；重复：{', '.join(drop)}")
        print(f"        |ρ| ∈ [{min(reps):.3f}, {max(reps):.3f}]"
              + (f"   ★ 完全重复对：{ident}" if ident else ""))

    print("-" * 100)
    total_drop = sum(r["size"] - 1 for r in rows)
    print(f"共 {len(rows)} 簇 · 建议删除 {total_drop} 个（每簇留 1 个代表）")

    outdir = Path(cfg.state_dir) / "dedup"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.json").write_text(json.dumps(
        {"years": years, "threshold": thr, "clusters": rows,
         "corr": {names[i]: {names[j]: (None if not np.isfinite(corr[i, j])
                                        else round(float(corr[i, j]), 4))
                             for j in range(len(names))} for i in range(len(names))}},
        ensure_ascii=False), encoding="utf-8")
    print(f"报告已写入 {outdir / 'report.json'}")
    return 0
