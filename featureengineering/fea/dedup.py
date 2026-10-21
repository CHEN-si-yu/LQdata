"""因子冗余召回与成对精判 —— `python main.py dedup`。

## 为什么需要它

「取精华去糟粕」的最大一块不是"因子无效"，而是**同一个东西做了好几遍**。
实测 272 个因子里，22 个簇的 |ρ|≥0.95（共 48 个因子），其中"价格在区间中的位置"
一个维度就挤了 9 个（`bollinger_position_20`/`keltner_position_20`/`donchian_position_20`/
`chandelier_position`/`williams_r_14`/`rsi_14`/`cci_20`/`stoch_slow_k`/`close_location_20d`）。

相关性用于提出复查候选。相同秩也可能具有不同非线性值间距，
不能据此断言没有增量信息，更不能自动删除。

## 口径（写死在这里，避免各说各话）

- 用落盘 rank 列作为第一层快速召回；第二层同时检查原值的日截面秩及模型值；
- 逐日做**截面 z-score**（去均值、除以标准差），再把每天拼成一个长向量；
- 因子 i 与 j 的相关 = `Σ x_i x_j / sqrt(Σ x_i² · Σ x_j²)`（NaN 记 0，即"缺失不参与"）。
  这是零填充余弦召回分数，与成对有效 Pearson 不同。
- 相同 rank 只是诊断，不代表现代模型消费的标准化 value 具有相同信息。
  典型来源是"每日减同一个常数"（如 `rs_60 = momentum_60 − 指数收益`：
  同一天所有股票减的是同一个数，**秩完全不变**）与单调变换（`log`/`sqrt`/`zscore`）。

## 输出

`state/dedup/report.json` + 屏幕上的簇清单。每簇给出：
成员、召回分数、共同覆盖、逐日成对秩相关和标准化值相关。无自动删除。
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
        if (path.parent / "latest.json").exists():
            pointer = json.loads((path.parent / "latest.json").read_text(encoding="utf-8"))
            path = Path(pointer["summary"])
        content = json.loads(path.read_text(encoding="utf-8"))
        rows = content.get("records", []) if isinstance(content, dict) else content
        return {r["name"]: r for r in rows if int(r.get("horizon", 5)) == 5}
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


def _gram_years(specs, cfg, engine, years: list[int]):
    """按年分块累加 `G = XᵀX`，返回 `(G, names, rows)`。

    ## 为什么必须分块（2026-09-21，S-02）

    `_matrix()` 把 (观测数 × 因子数) 的长矩阵**一次性**放进内存：
    658 个因子 × 2018~2026 全历史 ≈ 447 万行 × 658 列 × float32 ≈ **11.7 GB**，
    而 `np.column_stack` 还要再开一份同样大的输出 ⇒ **峰值约 23 GB**。
    在「90 GiB 容器 + 模型侧正在训练」的前提下，那就是平台明令禁止的「两个重活并行」。

    ## 为什么分块是**等价**而不是近似

    - z-score 本来就是**逐日截面**算的（`_matrix` 里也是按 axis=1 逐日归一），
      把某一年单独拿出来算，同一天的 z 值逐位相同；
    - `(A;B)ᵀ(A;B) = AᵀA + BᵀB`。
    所以两者只差**浮点求和顺序**，不差口径。峰值内存从 ~23 GB 降到「单年一块」≈ 1.4 GB。

    `cmd_dedup --matrix` 保留老路径，用来随时对拍（实测 |Δ| 在 1e-6 量级）。
    """
    cal, codes = engine.cal, engine.codes
    codes_idx = pd.Series(np.arange(codes.size), index=codes)
    from .panel import Panel

    # ★ 列集合**先定死**：保持与 `_matrix()` 相同的语义 —— 只要这个因子在被请求的
    #   年份里有产物，它就在矩阵里占一列；缺的 (年) 用 0 填（z-score 后缺失本就记 0）。
    #   用 factor_years() 判有无（只读分区清单，不读数据）—— 若改成"按年读一次再看空不空"，
    #   就会把 11 GB 数据读两遍。
    have = {sp.name: (set(store.factor_years(cfg.factors_dir, sp.name)) & set(years))
            for sp in specs}
    names = [sp.name for sp in specs if have[sp.name]]
    K = len(names)
    pos = {nm: i for i, nm in enumerate(names)}
    G = np.zeros((K, K), dtype=np.float64)
    rows: dict[str, int] = dict.fromkeys(names, 0)

    for y in sorted(years):
        days = cal.between(int(f"{y}0101"), int(f"{y}1231"))
        if days.size == 0:
            continue
        here = [(sp, pos[sp.name]) for sp in specs if y in have[sp.name]]
        if not here:
            continue
        panel = Panel(days, codes)
        Xb = np.zeros((days.size * codes.size, K), dtype=np.float32)
        for sp, j in here:
            df = store.read_factor(cfg.factors_dir, sp.name, [y])
            if df.empty:
                continue
            ci = codes_idx.reindex(df["stock_code"].to_numpy()).to_numpy()
            keep = np.isfinite(ci)
            if not keep.any():
                continue
            v = panel.place(ci[keep].astype(np.int64), series_to_int(df["trade_date"])[keep],
                            pd.to_numeric(df["rank"], errors="coerce").to_numpy()[keep])
            # 逐日截面 z-score（与 _matrix 逐字一致：NaN 记 0 = 缺失不参与相关）
            valid = np.isfinite(v)
            cnt = valid.sum(axis=1, keepdims=True)
            tot = np.where(valid, v, 0.0).sum(axis=1, keepdims=True)
            ss = np.where(valid, v * v, 0.0).sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = tot / np.maximum(cnt, 1)
                var = np.maximum(ss / np.maximum(cnt, 1) - mu * mu, 0.0)
                sd = np.sqrt(var)
                z = (v - mu) / np.where(sd > 0, sd, np.nan)
            Xb[:, j] = np.where(np.isfinite(z), z, 0.0).reshape(-1)
            rows[sp.name] += int(valid.sum())
            del v, z, valid
        G += Xb.T @ Xb
        del Xb
    return G, names, rows


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


def pairwise_redundancy(values_a, values_b, dates=None, min_n=100,
                        higher_a=True, higher_b=True,winsor=(.01,.99)) -> pd.DataFrame:
    """逐日成对有效相关；零填充不参与精判，方向来自声明合同。"""
    from .eval import _pearson, _spearman, model_value_panel
    a,b = np.asarray(values_a,dtype=float),np.asarray(values_b,dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("paired panels must have equal (T,C) shape")
    dates = list(range(len(a))) if dates is None else dates
    if len(dates) != len(a):
        raise ValueError("date axis mismatch")
    za,zb = model_value_panel(a,higher_a,winsor),model_value_panel(b,higher_b,winsor)
    oriented_a,oriented_b = a if higher_a else -a,b if higher_b else -b
    records=[]
    for t,date in enumerate(dates):
        fa,fb = np.isfinite(a[t]),np.isfinite(b[t])
        common,union = fa&fb,fa|fb
        n=int(common.sum())
        rank_corr=value_corr=np.nan
        rank_equal=model_equal=np.nan
        if n>=min_n:
            x,y=oriented_a[t,common],oriented_b[t,common]
            rank_corr,value_corr=_spearman(x,y),_pearson(za[t,common],zb[t,common])
            rx,ry=pd.Series(x).rank(method="average"),pd.Series(y).rank(method="average")
            rank_equal=float(np.mean(np.isclose(rx,ry,atol=1e-6,rtol=0)))
            model_equal=float(np.mean(np.isclose(za[t,common],zb[t,common],atol=1e-6,rtol=0)))
        records.append({"trade_date":str(date),"n_pool":a.shape[1],"common":n,
            "union":int(union.sum()),"common_fraction_pool":n/max(a.shape[1],1),
            "jaccard":n/max(int(union.sum()),1),"rank_corr":rank_corr,"model_value_corr":value_corr,
            "rank_equal_fraction":rank_equal,"model_equal_fraction":model_equal})
    return pd.DataFrame(records)


def _refine_pair(cfg,engine,years,a_spec,b_spec):
    from .eval import _place_frame, _json_safe
    from .panel import Panel
    frames=[]
    for year in years:
        days=engine.cal.between(max(year*10000+101,int(cfg.default_start.replace("-",""))),
                                min(year*10000+1231,engine.baseline_last_day()))
        if not days.size:
            continue
        a,b=store.read_year(cfg.factors_dir,a_spec.name,year),store.read_year(cfg.factors_dir,b_spec.name,year)
        if a.empty or b.empty:
            continue
        panel=Panel(days,engine.codes)
        av,bv=_place_frame(a,panel,engine.codes),_place_frame(b,panel,engine.codes)
        frames.append(pairwise_redundancy(av,bv,days,cfg.min_cross_section,
                                          a_spec.higher_is_better,b_spec.higher_is_better,cfg.winsor))
        del panel,av,bv,a,b
    daily=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    result={"a":a_spec.name,"b":b_spec.name,"days":len(daily),"years":{},"action":"manual_review_only"}
    if not daily.empty:
        result.update(common_fraction_pool=float(daily.common_fraction_pool.mean()),
                      coverage_jaccard=float(daily.jaccard.mean()),common_cells=int(daily.common.sum()))
        for metric in ("rank_corr","model_value_corr","rank_equal_fraction","model_equal_fraction"):
            valid=daily[metric].to_numpy(dtype=float)
            valid=valid[np.isfinite(valid)]
            result[metric]={"days":len(valid),"mean":float(valid.mean()) if valid.size else None,
                "median":float(np.median(valid)) if valid.size else None,
                "abs_p90":float(np.quantile(np.abs(valid),.9)) if valid.size else None,
                "positive_fraction":float(np.mean(valid>0)) if valid.size else None}
        for year,group in daily.groupby(daily.trade_date.str[:4]):
            result["years"][year]={"days":len(group),"common_fraction_pool":float(group.common_fraction_pool.mean()),
                "rank_corr_mean":float(group.rank_corr.mean()),"model_value_corr_mean":float(group.model_value_corr.mean())}
    return _json_safe(result)


def cmd_dedup(args, cfg) -> int:
    from .engine import Engine
    engine = Engine(cfg)
    from .release import production_specs
    specs = [s for s in production_specs(cfg) if not s.is_label and not s.is_market]
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
    print(f"\n冗余复查 · {len(specs)} 个因子 · {years[0]}~{years[-1]} · |召回分数| ≥ {thr}")
    print("秩零填充余弦仅用于召回；精判使用共同有效股票、真实模型值及覆盖重叠。")
    if getattr(args, "matrix", False):
        if len(years) > 1 or len(specs) > 128:
            raise ValueError("--matrix 对拍仅允许一个年份、至多128个因子；常规使用年块路径")
        # 老路径：一次性长矩阵。全历史 658 因子峰值约 23 GB —— 只在与小样本对拍时用。
        print("（--matrix：一次性长矩阵路径，仅用于与分块路径对拍；全量跑会吃 ~23 GB）")
        X, meta = _matrix(specs, cfg, engine, years)
        names = meta["names"]
        if X.shape[1] < 2:
            print("可比较的因子不足 2 个")
            return 1
        G = X.T @ X                                        # (K,K) 分子
        del X
    else:
        G, names, _rows = _gram_years(specs, cfg, engine, years)
        if len(names) < 2:
            print("可比较的因子不足 2 个")
            return 1
    d = np.sqrt(np.diag(G))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = G / np.outer(np.where(d > 0, d, np.nan), np.where(d > 0, d, np.nan))
    np.fill_diagonal(corr, 1.0)

    ev = _load_eval(Path(cfg.state_dir) / "eval" / "summary.json")
    groups = _cluster(corr, names, thr)

    rows = []
    spec_lookup = {s.name:s for s in specs}
    refined = []
    max_pairs = max(1, int(getattr(args,"max_pairs",2000)))
    recall_pairs = [(i,j) for i in range(len(names)) for j in range(i+1,len(names))
                    if np.isfinite(corr[i,j]) and abs(corr[i,j]) >= thr]
    recall_pairs.sort(key=lambda pair:-abs(corr[pair[0],pair[1]]))
    for i,j in recall_pairs[:max_pairs]:
        record = _refine_pair(cfg,engine,years,spec_lookup[names[i]],spec_lookup[names[j]])
        record["recall_score"] = float(corr[i,j])
        refined.append(record)
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
        representative = members[0]
        rows.append({"size": len(g), "members":members,"review_first":representative,
                     "rho_min": round(float(min(reps)), 4),
                     "rho_max": round(float(max(reps)), 4),"action":"manual_review_only"})
        print(f"召回簇 {len(g):>2} 个：{', '.join(members)}")
        print(f"        |召回分数| ∈ [{min(reps):.3f}, {max(reps):.3f}]；传递连接不代表所有成员成对重复")

    print("-" * 100)
    print(f"共 {len(rows)} 个复查簇 · 精判 {len(refined)}/{len(recall_pairs)} 对 · 不自动删除")

    # `--out` 允许把报告写到别处（试跑/对拍时不覆盖生产报告）；默认仍是 state/dedup/。
    out = Path(getattr(args, "out", "") or (Path(cfg.state_dir) / "dedup" / "report.json"))
    if not out.is_absolute():
        out = cfg.root/out
    if not out.resolve().is_relative_to(cfg.root.resolve()):
        raise ValueError("dedup output must remain within project")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"years": years, "threshold": thr, "clusters": rows,"refined_pairs":refined,
         "recall_pairs":len(recall_pairs),"truncated":len(recall_pairs)>max_pairs,
         "deletion_policy":"manual_only; no automatic removal",
         "corr": {names[i]: {names[j]: (None if not np.isfinite(corr[i, j])
                                        else round(float(corr[i, j]), 4))
                             for j in range(len(names))} for i in range(len(names))}},
        ensure_ascii=False), encoding="utf-8")
    print(f"报告已写入 {out}")
    return 0
