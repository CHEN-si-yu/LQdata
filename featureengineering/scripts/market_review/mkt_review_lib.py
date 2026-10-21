"""市场因子复审共享库（scripts/market_review；随 freeze 冻结）。

职责：只读生产产物与上游原始数据，构建复审目标，提供 M2/M4/M5/M6 统计量。
边界（冻结于 doc/市场因子审核体系.md）：
- 绝不写 data/、state/、conf/；产物只写 artifacts/research/market_review_20261003/。
- 本文件位于 scripts/，不在任何身份哈希域内；freeze 自愿收录其 sha256。
- M5 统计量（冻结）：ρ = mean(zrank(x)·zrank(y))（共同有效样本）；
  HAC-Bartlett(19) 95% 区间由 fea/eval.py::mean_uncertainty 对逐日乘积序列提供
  —— 与股票 G4 同一 HAC 内核；分年 ρ_y = 年内乘积均值（年内共同有效日 ≥60 才计入年）。
- M6（冻结）：Newey-West/Bartlett(19) OLS t（本文件实现，无 scipy）；
  目标与全部回归列先做共同样本 zrank；控制集见 M6_CONTROLS。
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ———— 冻结参数（与 doc/市场因子审核体系.md 一致；改任何一项须出新 freeze 版本）————
REVIEW_ID = "market_review_20261003"
BLOCK_DAYS = 20          # HAC Bartlett 带宽参数（与股票口径一致：lag=19）
MIN_SIGNAL_DAYS = 60     # 单年计入的最少共同有效日
MIN_SIGNAL_YEARS = 6     # 有效年数下限
MIN_M4_OVERLAP = 250     # M4 成对最少共同有效日
M4_FAIL = 0.95           # |ρ|>0.95 判不过
M4_EXPLAIN = 0.90        # 0.90<|ρ|≤0.95 需书面论证
M6_MIN_N = 250
M6_T = 2.0               # |declared t| ≥ 2.0
M6_LAG = 19
M6_CONTROLS = (
    "mkt_idx_csi300_momentum20",
    "mkt_idx_csi300_volatility20",
    "mkt_dispersion",
    "mkt_up_ratio",
    "mkt_margin_balance_chg20",
    "mkt_am_amount_ratio20",
)
PRIMARY_HORIZONS = (1, 5)

ROOT = _ROOT


def cfg_load():
    from fea.config import load
    return load()


def project_path(*parts):
    return ROOT.joinpath(*parts)


# ———— 读取 ————

def axis_days(cfg):
    """正式交易日轴（2018-01-02..cutoff），int64。"""
    from fea.engine import Engine
    eng = Engine(cfg)
    return eng.cal.between(20180102, eng.baseline_last_day())


def market_specs(cfg):
    """正式批准的全部市场因子 spec（name -> spec）。"""
    from fea.release import production_specs
    out = {}
    for spec in production_specs(cfg):
        if spec.is_market:
            out[spec.name] = spec
    return out


def read_market_series(cfg, name):
    """读取市场因子全历史：返回 (days int64 升序, values float64)。"""
    base = cfg.root / "data" / "market_factors" / name
    files = sorted(base.glob("year=*/data.parquet"))
    if not files:
        raise FileNotFoundError(f"{name}: 无市场因子分区")
    frames = [pd.read_parquet(p, columns=["trade_date", "value"]) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df["day"] = df["trade_date"].str.replace("-", "", regex=False).astype(np.int64)
    df = df.sort_values("day", kind="stable")
    if df["day"].duplicated().any():
        raise ValueError(f"{name}: trade_date 重复")
    return df["day"].to_numpy(), df["value"].to_numpy(dtype=float)


def read_market_on_axis(cfg, name, axis):
    days, values = read_market_series(cfg, name)
    s = pd.Series(values, index=days)
    return s.reindex(axis).to_numpy(dtype=float)


def read_label_agg(cfg, horizon, how="mean"):
    """由正式标签派生的市场级序列（按 trade_date 聚合）：
    how ∈ {mean(前视等权市场收益), std(前视离散度), breadth(前视宽度 share>0)}。"""
    base = cfg.root / "data" / "factors" / f"label_ret_{horizon}d"
    files = sorted(base.glob("year=*/data.parquet"))
    if not files:
        raise FileNotFoundError(f"label_ret_{horizon}d: 无分区")
    out = []
    for p in files:
        df = pd.read_parquet(p, columns=["trade_date", "value"])
        g = df.groupby("trade_date", sort=False)["value"]
        if how == "mean":
            s = g.mean()
        elif how == "std":
            s = g.std(ddof=0)
        elif how == "breadth":
            s = (df["value"] > 0).groupby(df["trade_date"]).mean()
        else:
            raise ValueError(f"未知聚合：{how}")
        out.append(s)
    s = pd.concat(out)
    s.index = s.index.str.replace("-", "", regex=False).astype(np.int64)
    s = s.sort_index()
    if s.index.duplicated().any():
        raise ValueError(f"label_ret_{horizon}d: 聚合后重复日期")
    return s


def read_index_open_fwd(cfg, code, horizon, axis):
    """指数 open-to-open 前视收益（对齐标签端点约定）：open(t+1+h)/open(t+1)-1。"""
    from fea.upstream import Upstream
    up = Upstream(cfg.upstream, cfg.root)
    df = up.read("index_daily", columns=["ts_code", "trade_date", "open"],
                 years=(int(axis[0]) // 10000, int(axis[-1]) // 10000))
    df = df[df["ts_code"] == code]
    if df.empty:
        raise ValueError(f"index_daily 无 {code}")
    day = df["trade_date"].str.replace("-", "", regex=False).astype(np.int64)
    if day.duplicated().any():
        raise ValueError(f"index_daily {code}: 重复主键")
    o = pd.Series(df["open"].to_numpy(dtype=float), index=day).reindex(axis).to_numpy(dtype=float)
    t = len(o)
    fwd = np.full(t, np.nan)
    n = t - (horizon + 1)
    if n > 0:
        fwd[:n] = o[horizon + 1:horizon + 1 + n] / o[1:1 + n] - 1.0
    return fwd


# ———— 统计量 ————

def zrank(x):
    """共同样本秩标准化（平均秩法 → 去均值、除以总体 rms）；NaN 保持 NaN。"""
    x = np.asarray(x, dtype=float)
    r = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    r = np.where(np.isfinite(x), r, np.nan)
    good = np.isfinite(r)
    if good.sum() < 2:
        return np.full_like(r, np.nan)
    mu = r[good].mean()
    sd = r[good].std(ddof=0)
    if sd <= 0:
        return np.full_like(r, np.nan)
    return (r - mu) / sd


def signal_test(values, target, days, horizon, declared=0, block_days=BLOCK_DAYS):
    """M5 预测检验。返回 dict（含 raw/declared 双列）。"""
    v = np.asarray(values, dtype=float)
    y = np.asarray(target, dtype=float)
    d = np.asarray(days, dtype=np.int64)
    finite = np.isfinite(v) & np.isfinite(y)
    dd, vv, yy = d[finite], v[finite], y[finite]
    out = {"h": int(horizon), "declared_direction": int(declared), "days": int(finite.sum())}
    if dd.size < MIN_SIGNAL_DAYS:
        out.update({"rho": None, "rho_raw": None, "hac_se": None, "t": None,
                    "ci95_low": None, "ci95_high": None,
                    "declared_rho": None, "declared_t": None,
                    "declared_ci95_low": None, "declared_ci95_high": None,
                    "year_rho": {}, "valid_years": 0, "years_same_sign": None,
                    "insufficient": True})
        return out
    zx, zy = zrank(vv), zrank(yy)
    s = zx * zy  # 逐日乘积序列；mean(s) ≡ Spearman ρ
    from fea.eval import mean_uncertainty
    mu = mean_uncertainty(s, block_days=block_days, horizon=horizon)
    rho, se = mu["mean"], mu["hac_se"]
    years = dd // 10000
    year_rho = {}
    for year in sorted(set(years.tolist())):
        mask = years == year
        if int(mask.sum()) >= MIN_SIGNAL_DAYS:
            year_rho[int(year)] = float(s[mask].mean())
    same = None
    if declared:
        same = sum(1 for r in year_rho.values() if np.sign(r) == np.sign(declared))
    out.update({
        "rho": float(rho), "rho_raw": float(rho),
        "hac_se": float(se) if se is not None else None,
        "t": float(rho / se) if se else None,
        "hac_lag": mu["hac_lag"],
        "ci95_low": mu["ci95_low"], "ci95_high": mu["ci95_high"],
        "declared_rho": float(rho * declared) if declared else None,
        "declared_t": float(rho / se * declared) if (declared and se) else None,
        # 方向翻转须对区间两端同时取负并重排：declared CI = [min(low·d, high·d), max(low·d, high·d)]
        "declared_ci95_low": (float(min(mu["ci95_low"] * declared, mu["ci95_high"] * declared))
                              if declared else None),
        "declared_ci95_high": (float(max(mu["ci95_low"] * declared, mu["ci95_high"] * declared))
                               if declared else None),
        "year_rho": year_rho, "valid_years": len(year_rho),
        "years_same_sign": same, "insufficient": False,
    })
    return out


def m5_pass(row):
    """冻结的 M5 判据：主期限 (1 或 5) 上 declared CI 排除 0（同向），
    且同号年数 ≥ ceil(7×有效年数/9)，且有效年数 ≥ MIN_SIGNAL_YEARS。"""
    if not row.get("declared_direction"):
        return None
    for h in PRIMARY_HORIZONS:
        r = row.get(f"h{h}")
        if not r or r.get("insufficient"):
            continue
        same, vy = r.get("years_same_sign"), r.get("valid_years") or 0
        if (r["declared_ci95_low"] is not None and r["declared_ci95_low"] > 0
                and vy >= MIN_SIGNAL_YEARS and same is not None
                and same >= math.ceil(7 * vy / 9)):
            return h
    return None


def hac_ols(y, X, lag=M6_LAG):
    """OLS + Newey–West/Bartlett(lag) 协方差。y:(n,), X:(n,k)（不含截距）。
    返回 (beta 含截距, se, t)。"""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    Xd = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    n = len(y)
    Xe = Xd * resid[:, None]
    S = Xe.T @ Xe
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        G = Xe[k:].T @ Xe[:-k]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    return beta, se, t


def incremental_test(candidate, controls, target, horizon, declared=0):
    """M6 增量检验：zrank(目标) ~ zrank(候选) + zrank(控制集)。
    controls: {name: 全轴数组}（已排除候选自身）。判据见 m6_pass。"""
    cols = [("候选", np.asarray(candidate, dtype=float))] + \
           [(k, np.asarray(v, dtype=float)) for k, v in sorted(controls.items())]
    M = np.column_stack([v for _, v in cols])
    y0 = np.asarray(target, dtype=float)
    good = np.isfinite(y0) & np.isfinite(M).all(axis=1)
    n = int(good.sum())
    out = {"h": int(horizon), "declared_direction": int(declared), "n": n,
           "controls": [k for k, _ in cols[1:]], "self_excluded": None}
    if n < M6_MIN_N:
        out.update({"insufficient": True})
        return out
    X = np.column_stack([zrank(M[good, i]) for i in range(M.shape[1])])
    y = zrank(y0[good])
    beta, se, t = hac_ols(y, X, lag=M6_LAG)
    resid = y - np.column_stack([np.ones(n), X]) @ beta
    cond = float(np.linalg.cond(np.column_stack([np.ones(n), X])))
    b, tt = float(beta[1]), float(t[1])
    out.update({
        "beta": b, "t": tt, "se": float(se[1]),
        "declared_beta": b * declared if declared else None,
        "declared_t": tt * declared if declared else None,
        "residual_variance_fraction": float(1.0 - resid.var() / y.var()) if y.var() > 0 else None,
        "condition_number": cond, "insufficient": False,
    })
    return out


def m6_pass(row):
    """冻结的 M6 判据：|declared t| ≥ 2.0 且 declared t 为正（= 符号与登记方向一致）。"""
    if not row or row.get("insufficient") or row.get("declared_direction") in (0, None):
        return None
    for h in PRIMARY_HORIZONS:
        r = row.get(f"h{h}")
        if not r or r.get("insufficient"):
            continue
        dt = r.get("declared_t")
        if dt is not None and np.isfinite(dt) and dt >= M6_T:
            return h
    return None


def ts_rank_corr(x, y, min_overlap=MIN_M4_OVERLAP):
    """两条时序的共同样本 Spearman（= 乘积均值）。不足 min_overlap 返回 (nan, n)。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    n = int(good.sum())
    if n < min_overlap:
        return float("nan"), n
    zx, zy = zrank(x[good]), zrank(y[good])
    return float(np.mean(zx * zy)), n


def rank_corr_matrix(series, min_overlap=MIN_M4_OVERLAP):
    """M4：{name: 全轴数组} → (names, rho 矩阵, 成对列表按 |ρ| 降序)。"""
    names = sorted(series)
    k = len(names)
    rho = np.full((k, k), np.nan)
    np.fill_diagonal(rho, 1.0)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            r, n = ts_rank_corr(series[names[i]], series[names[j]], min_overlap)
            rho[i, j] = rho[j, i] = r
            if np.isfinite(r):
                pairs.append({"a": names[i], "b": names[j], "rho": r, "n": n})
    pairs.sort(key=lambda p: -abs(p["rho"]))
    return names, rho, pairs


# ———— M2 结构统计 ————

def m2_stats(cfg, spec, axis):
    """M2 覆盖与变化（冻结判据见 doc）：返回 dict + passed 标志。"""
    name = spec.name
    days, values = read_market_series(cfg, name)
    rs = int(str(spec.resolved_start(cfg)).replace("-", ""))
    fin = np.isfinite(values)
    pre = days < rs
    pre_fin = int((fin & pre).sum())
    align_fill = spec.align_fill
    has_fill = align_fill is not None and not (isinstance(align_fill, float) and math.isnan(align_fill))
    pre_leak = 0
    pre_fill_ok = True
    if has_fill:
        fill = float(np.float32(align_fill))
        pre_vals = values[pre & fin]
        pre_fill_ok = bool(np.all(pre_vals == fill)) if pre_vals.size else True
        if fill == 0.0:
            pre_leak = 0
    else:
        pre_leak = pre_fin
    post = days >= rs
    post_total = int(post.sum())
    post_fin = int((fin & post).sum())
    frac = post_fin / post_total if post_total else 0.0
    uniq_total = int(np.unique(values[fin]).size)
    years = days // 10000
    per_year = {}
    # 按年非常数判定只统计 resolved_start 之后（声明的对齐前缀是契约常量、不参与退化检测）
    post_mask = days >= rs
    for year in sorted(set(years[fin & post_mask].tolist())):
        mask = fin & post_mask & (years == year)
        cnt = int(mask.sum())
        if cnt >= MIN_SIGNAL_DAYS:
            per_year[int(year)] = int(np.unique(values[mask]).size)
    thin_years = {y: c for y, c in per_year.items() if c < 2}
    passed = (frac >= 0.98 and uniq_total >= 10 and not thin_years
              and pre_leak == 0 and pre_fill_ok)
    return {
        "name": name, "resolved_start": str(spec.resolved_start(cfg)),
        "align_fill": (None if not has_fill else float(np.float32(align_fill))),
        "post_start_days": post_total, "post_start_finite": post_fin,
        "post_start_finite_fraction": frac,
        "distinct_finite_total": uniq_total,
        "per_year_distinct": per_year, "thin_years": thin_years,
        "prefix_finite_cells": pre_fin, "prefix_leak_cells": pre_leak,
        "prefix_fill_ok": pre_fill_ok,
        "passed": bool(passed),
    }


def build_targets(cfg, axis):
    """复审目标（主 + 辅助）。主目标 = 前视等权市场收益（标签派生），h∈{1,5}。"""
    targets = {}
    for h in PRIMARY_HORIZONS:
        ew = read_label_agg(cfg, h, "mean")
        targets[f"primary_ew_{h}"] = ew.reindex(axis).to_numpy(dtype=float)
        std = read_label_agg(cfg, h, "std")
        targets[f"aux_dispersion_{h}"] = std.reindex(axis).to_numpy(dtype=float)
        breadth = read_label_agg(cfg, h, "breadth")
        targets[f"aux_breadth_{h}"] = breadth.reindex(axis).to_numpy(dtype=float)
        targets[f"aux_csi300_open_{h}"] = read_index_open_fwd(cfg, "000300.SH", h, axis)
        targets[f"aux_csi500_open_{h}"] = read_index_open_fwd(cfg, "000905.SH", h, axis)
    return targets


# ———— 杂项 ————

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path, root=ROOT):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


# ———— 自测（仅合成数据；freeze 前的开发校验）————

def _selftest():
    rng = np.random.default_rng(20261003)
    n = 2500
    days = np.sort(rng.choice(np.arange(20180102, 20261231), size=n, replace=False))
    x = rng.normal(size=n)
    y = x + rng.normal(scale=1.0, size=n)
    from fea.eval import _spearman, mean_uncertainty
    r = signal_test(x, y, days, 1, declared=1)
    assert abs(r["rho"] - _spearman(x, y)) < 1e-12, "乘积均值必须等于 Spearman"
    assert r["declared_ci95_low"] > 0 and r["declared_t"] > 2, "强相关应显著"
    mu = mean_uncertainty(zrank(x) * zrank(y), block_days=20, horizon=1)
    assert abs(mu["mean"] - r["rho"]) < 1e-12 and mu["hac_lag"] == 19
    # 独立变量应不显著
    z = rng.normal(size=n)
    r0 = signal_test(x, z, days, 5, declared=1)
    assert r0["declared_ci95_low"] < 0 < r0["declared_ci95_high"], "独立变量不应显著"
    # 负方向（declared=-1）：区间两端取负重排；强负相关应过、弱负相关不应过
    rneg = signal_test(-x, y, days, 1, declared=-1)
    assert rneg["declared_ci95_low"] > 0 and rneg["declared_t"] > 2, "强负相关应过"
    assert abs(rneg["declared_ci95_low"] - (-rneg["ci95_high"])) < 1e-12
    assert abs(rneg["declared_ci95_high"] - (-rneg["ci95_low"])) < 1e-12, \
        "declared 区间必须先按方向翻端再取低/高（历史 bug：直接 low×d）"
    xw = rng.normal(size=n) * 3.0 - 0.05 * x         # 弱负相关（t≈-1.3，区间含 0）
    rw = signal_test(xw, y, days, 1, declared=-1)
    assert rw["declared_ci95_low"] < 0, "弱负相关不应过（区间含 0 的声明端）"
    assert m5_pass({"declared_direction": -1, "h1": rneg}) == 1
    assert m5_pass({"declared_direction": -1, "h1": rw}) is None
    # HAC-OLS：y = 2x + 噪声
    y2 = 2.0 * x + rng.normal(scale=1.0, size=n)
    beta, se, t = hac_ols(y2, x[:, None])
    assert abs(beta[1] - 2.0) < 0.1 and t[1] > 20, ("hac_ols 系数或 t 异常", beta, t)
    # 增量（D.1 投影模式）：目标只由控制变量驱动、候选是控制的代理 → 候选不应显著
    ctrl = rng.normal(size=n)
    proxy = ctrl + rng.normal(scale=0.05, size=n)
    y_ctrl = 2.0 * ctrl + rng.normal(scale=1.0, size=n)
    inc = incremental_test(proxy, {"ctrl": ctrl}, y_ctrl, 1, declared=1)
    assert inc["declared_t"] < 2.0, ("代理候选不应显著", inc)
    # 目标由候选驱动、控制为噪声 → 候选应显著
    inc2 = incremental_test(x, {"noise": z}, y2, 1, declared=1)
    assert inc2["declared_t"] > 2.0, ("真驱动候选应显著", inc2)
    # M4
    names, rho, pairs = rank_corr_matrix({"a": ctrl, "b": proxy, "d": z})
    assert names == ["a", "b", "d"] and abs(rho[0, 1]) > 0.9 and abs(rho[0, 2]) < 0.2
    # m5/m6 判据
    assert m5_pass({"declared_direction": 1, "h1": r}) == 1
    assert m5_pass({"declared_direction": 1, "h1": r0}) is None
    assert m6_pass({"declared_direction": 1, "h1": inc2}) == 1
    assert m6_pass({"declared_direction": 1, "h1": inc}) is None
    assert m6_pass({"declared_direction": 0, "h1": inc2}) is None
    # zrank 与 pd 平均秩一致
    zx = zrank(x)
    ref = pd.Series(x).rank(method="average")
    ref = (ref - ref.mean()) / ref.std(ddof=0)
    assert np.allclose(zx, ref)
    print("selftest OK：M5/M6/M4/zrank/hac_ols 合成数据校验通过")


if __name__ == "__main__":
    _selftest()
