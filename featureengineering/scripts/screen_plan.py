#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 `run_screen.py` 的质检数据转成**可复核的删除候选清单**（不删任何东西）。

## 规则（用户 2026-09-24 拍板的口径：IC 低**单独不构成删除理由**）

| 档 | 判据 | 说明 |
|:--|:--|:--|
| **R1 直接相关** | 与某个实际保留者 `|ρ| ≥ θc` | 主删除源。比较最近三年 1d/5d RankIC，逐个给出直接相关证据 |
| **R2 缺失率高** | `cov_mean < 覆盖率阈值` | 仅与 R4 同时成立才入候选；结构性稀疏家族豁免 |
| **R3 取值退化** | 常数日、低基数、截面零方差或零膨胀 | 仅与 R4 同时成立才入候选；二值事件本身不是缺陷 |
| **R4 稳定弱 IC** | 1d 和 5d 全历史及至少两个非重叠三年段都弱 | 只作 R2/R3 的并列证据，不能单独删除 |

**硬守卫**（命中即从清单剔除并在报告里列出）：
1. **父依赖**：任何存活因子的 `deps` 里出现的因子名（`coupling` 族靠它读父因子产物，
   父因子被删 ⇒ 子因子**静默**变全 NaN）。★ 名单**现算**，不硬编码。
2. **簇代表**：每簇留下的那一个永不入列。
3. **范围**：市场因子、标签不入列。
4. **保护名单**：`PROTECTED`（沿用 `prune_factors.py` 的语义）。

## 为什么先出「阶梯表」再出清单

删除规模对 θc 极其敏感，而"删多少才合适"是**人的判断**不是脚本的判断。
所以脚本把 θc / 覆盖率阈值各自的边际影响一次性算出来（`ladder.json`），
让拍板的人看着"删 150 个 vs 删 230 个各是哪些"再决定，而不是让脚本偷偷替他选。

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/screen_plan.py --screen <run_screen 的输出目录> --out <交付目录>
    $PY scripts/screen_plan.py --theta 0.92 --cov-thr 0.5      # 直接定一档出清单
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 与 fea/eval.py 同口径的阈值常量（写在这里是为了让本脚本能独立读数据，
# 数值必须与 eval.py 保持一致 —— 改动时两边一起改）
NOISE_RANKIC = 0.005        # 暂定：待全库实测后由用户拍板
NOISE_MIN_DAYS = 500        # 两个全历史目标各需至少这么多有效 IC 日
NOISE_MIN_PERIODS = 2       # 至少两个非重叠三年段
NOISE_PERIOD_MIN_DAYS = 120 # 每个活跃三年段、每个目标的有效 IC 日
RECENT_MIN_DAYS = 60        # 比较代表者最近三年目标表现时的最低样本数
RECENT_1D_WEIGHT = 0.6      # 交付 head 是 1d，5d 仍占 0.4
SPARSE_ZERO_FRAC = 0.70

THETA_LADDER = (0.99, 0.98, 0.95, 0.92, 0.90, 0.85)
COV_LADDER = (0.30, 0.40, 0.50, 0.60, 0.70)
# ★ 事件型家族：覆盖率低是**事件属性**（龙虎榜只登记上榜股、涨停封单只在上板日有值），
#   不是"数据没到位"。用户 2026-09-24 明确「保留事件型稀疏因子」，
#   所以 R2（缺失率高）对这几个家族**不适用**；R3 必须与 R4 并列才入候选。
#   实测这些家族的 cov_mean 在 6%~24%，任何正常阈值都会把它们全删掉；
#   例外必须显式写出来，不能靠调阈值去"恰好避开"。
#   · `margin`：两融标的本身就是全市场的一个**子集**（实测池内 1516/2115 = 71.7%），
#     17 个因子的 cov_mean 紧贴在 0.560~0.592 —— 那是"两融标的占比"，不是数据缺口。
#   · 依赖这些因子的 `coupling` 因子继承同一性质（`cp_*_margin_*` 等），一并豁免。
STRUCTURAL_COV_GROUPS = ("event", "field_events", "disclosure_detail", "margin")


def _structural_cov(rows: list[dict]) -> set[str]:
    """覆盖率由**数据源的结构**决定（而非缺失）的因子集合 —— 现算，不硬编码。

    直接依赖豁免家族的耦合因子（`deps` 写的是父因子名）继承同一性质。
    """
    import factors                                     # noqa: F401
    from fea.spec import REGISTRY
    out = {r["name"] for r in rows if r.get("group") in STRUCTURAL_COV_GROUPS}
    changed = True
    while changed:                                     # 传递闭包：耦合链可能不止一层
        changed = False
        for nm, sp in REGISTRY.items():
            if nm in out:
                continue
            if any(d in out for d in sp.deps):
                out.add(nm)
                changed = True
    return out

# R3 的子判据是提示，只有 R4 同时成立才进入候选。
CONST_FRAC_MAX = 0.10
UNIQ_MED_MIN = 2
CS_STD_EPS = 1e-12


def _rankic_mean(stats: dict, h: str, min_days: int) -> float:
    """Signed RankIC mean; NaN means unavailable or too few valid days."""
    item = (stats.get("rankic") or {}).get(h) or {}
    mean = item.get("mean")
    if mean is None or int(item.get("n") or 0) < min_days:
        return float("nan")
    try:
        value = float(mean)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _recent_rankic(r: dict, period_label: str | None = None) -> tuple[str, float, float]:
    """A common recent period for comparisons, or factor's latest for display."""
    periods = r.get("periods") or {}
    labels = ([period_label] if period_label is not None else
              sorted(periods, key=lambda x: int(x.rsplit("-", 1)[-1]), reverse=True))
    for label in labels:
        if label not in periods:
            continue
        p = periods[label]
        one = _rankic_mean(p, "1", RECENT_MIN_DAYS)
        five = _rankic_mean(p, "5", RECENT_MIN_DAYS)
        if np.isfinite(one) and np.isfinite(five):
            return label, one, five
    return "", float("nan"), float("nan")


def _strength(one: float, five: float) -> float:
    if not (np.isfinite(one) and np.isfinite(five)):
        return -1.0
    return RECENT_1D_WEIGHT * abs(one) + (1.0 - RECENT_1D_WEIGHT) * abs(five)


def _quality_key(r: dict, latest_period: str) -> tuple:
    """Recent 1d/5d first, full-history 1d/5d second, coverage third."""
    _, one, five = _recent_rankic(r, latest_period)
    all_one = _rankic_mean(r, "1", RECENT_MIN_DAYS)
    all_five = _rankic_mean(r, "5", RECENT_MIN_DAYS)
    cov = r.get("cov_mean")
    coverage = float(cov) if cov is not None and np.isfinite(cov) else 0.0
    return (-_strength(one, five), -_strength(all_one, all_five),
            -coverage, r["name"])


def _greedy_prune(corr: np.ndarray, names: list[str], byname: dict,
                  thr: float, locked: set[str] | None = None) -> tuple[list[dict], dict[str, str]]:
    """**贪心去冗**：按共同近期双目标质量从高到低遍历，只有与**已保留**因子
    `|ρ| ≥ thr`」时才删掉，并记录它是被谁顶掉的。

    ★ 为什么不用单链接并查集聚类：`|ρ|≥thr` 的**连通分量**会链式串接 ——
      实测 θc=0.80 时最大簇有 11 个成员，但簇内**最小**成对 |ρ| 只有 **0.513**，
      也就是说"删掉的那 10 个"里有的跟留下的代表几乎不相关。那不是在删重复，
      是在借重复之名删弱相关因子。贪心法下**每一个被删的因子都有一条
      `|ρ|≥thr` 的直接证据指向某个保留者**，没有传递性。
    """
    iu = corr.shape[0]
    locked = locked or set()
    period_labels = {p for r in byname.values() for p in (r.get("periods") or {})}
    if not period_labels:
        raise ValueError("screen_metrics 缺少三年段统计；请使用新版 run_screen.py")
    latest_period = max(period_labels, key=lambda x: int(x.rsplit("-", 1)[-1]))
    order = sorted(range(iu), key=lambda i: _quality_key(byname[names[i]], latest_period))
    kept: list[int] = []
    drop: dict[str, str] = {}
    for i in order:
        # 父依赖/保护项必须真实留在集合里，仍可成为其它因子的直接比较对象。
        if names[i] in locked:
            kept.append(i)
            continue
        hit = None
        for j in kept:
            r = corr[i, j]
            if np.isfinite(r) and abs(r) >= thr:
                hit = j
                break
        if hit is None:
            kept.append(i)
        else:
            drop[names[i]] = names[hit]
    groups: dict[str, list[str]] = {}
    for nm, keep in drop.items():
        groups.setdefault(keep, []).append(nm)
    cl = [{"size": len(v) + 1, "keep": k, "drop": sorted(v),
           "rho_min": round(min(abs(float(corr[names.index(x), names.index(k)]))
                                for x in v), 4),
           "rho_max": round(max(abs(float(corr[names.index(x), names.index(k)]))
                                for x in v), 4)}
          for k, v in sorted(groups.items(), key=lambda x: -len(x[1]))]
    return cl, drop


def _clusters(corr: np.ndarray, thr: float) -> list[list[int]]:
    """`|ρ| ≥ thr` 的并查集连通分量（只返回 size>1 的簇）。"""
    n = corr.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    iu = np.triu_indices(n, k=1)
    rho = np.abs(corr[iu])
    for a, b in zip(iu[0][rho >= thr], iu[1][rho >= thr]):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def _parents() -> dict[str, list[str]]:
    """现算「哪些因子是被别的存活因子当父依赖引用的」。

    ★ 不硬编码名单：`coupling` 族的 `deps` 写的是父**因子名**，
      谁改了耦合关系，这里跟着变 —— 手抄一份名单迟早会和代码脱节。
    """
    import factors                                     # noqa: F401
    from fea.spec import REGISTRY
    out: dict[str, list[str]] = {}
    for nm, s in REGISTRY.items():
        if s.is_label:
            continue
        for d in s.deps:
            if d in REGISTRY:
                out.setdefault(d, []).append(nm)
    return out


def _protected() -> set[str]:
    """与 `scripts/prune_factors.py` 的 `PROTECTED` 同一语义（标签永不可删）。"""
    import factors                                     # noqa: F401
    from fea.spec import REGISTRY
    return {nm for nm, s in REGISTRY.items() if s.is_label}


def _rules_hit(r: dict, cov_thr: float, structural: set[str]) -> list[str]:
    """R2/R3 are candidates only when both formal horizons are stably weak."""
    if not _is_stable_noise(r):
        return []
    hit = []
    if (np.isfinite(r["cov_mean"]) and r["cov_mean"] < cov_thr
            and r["name"] not in structural):
        hit.append(f"R2覆盖率<{cov_thr}")
    # 二值或偶发常数列不等于无效；R4 已先验证两个目标跨时期都弱。
    degen = []
    if np.isfinite(r["const_frac"]) and r["const_frac"] > CONST_FRAC_MAX:
        degen.append(f"常数列{float(r['const_frac']):.0%}")
    if np.isfinite(r["cs_std_med"]) and r["cs_std_med"] <= CS_STD_EPS:
        degen.append("截面零方差")
    if np.isfinite(r["uniq_med"]) and r["uniq_med"] <= UNIQ_MED_MIN:
        degen.append(f"取值数{float(r['uniq_med']):.0f}")
    if degen:
        hit.append("R3a低区分度(" + ",".join(degen) + ")")

    if np.isfinite(r["zero_frac"]) and r["zero_frac"] > SPARSE_ZERO_FRAC:
        hit.append(f"R3b零膨胀{r['zero_frac']:.0%}")
    return hit


def _is_stable_noise(r: dict) -> bool:
    """Both formal heads weak globally and in every measurable 3-year period.

    Incomplete periods are not evidence of weakness. The latest active period
    must also have enough 1d and 5d observations; this protects newer and
    sparse event factors from being pruned on early-period noise alone.
    """
    if any(not np.isfinite(m) or abs(m) >= NOISE_RANKIC
           for m in (_rankic_mean(r, "1", NOISE_MIN_DAYS),
                     _rankic_mean(r, "5", NOISE_MIN_DAYS))):
        return False
    active = [p for p in (r.get("periods") or {}).values()
              if any(int(((p.get("rankic") or {}).get(h) or {}).get("n") or 0) > 0
                     for h in ("1", "5"))]
    if len(active) < NOISE_MIN_PERIODS:
        return False
    for p in active:
        if any(not np.isfinite(m) or abs(m) >= NOISE_RANKIC
               for m in (_rankic_mean(p, "1", NOISE_PERIOD_MIN_DAYS),
                         _rankic_mean(p, "5", NOISE_PERIOD_MIN_DAYS))):
            return False
    return True


def _build(rows: list[dict], corr: np.ndarray, names: list[str],
           theta: float, cov_thr: float, parents: dict, protected: set,
           structural: set[str]):
    """候选 = 直接相关冗余，或 R2/R3 缺陷与双周期跨期弱 IC 并列。"""
    byname = {r["name"]: r for r in rows}
    clusters_out, drop_map = _greedy_prune(
        corr, names, byname, theta, locked=set(parents) | protected)

    cand: dict[str, dict] = {}
    for ci, cl_ in enumerate(clusters_out):
        for nm in cl_["drop"]:
            cand.setdefault(nm, {"rules": [], "cluster": ci, "keep": cl_["keep"]})
            cand[nm]["rules"].append(
                f"R1(与保留者 {cl_['keep']} |ρ|={abs(float(corr[names.index(nm), names.index(cl_['keep'])])):.3f})")

    for r in rows:
        nm = r["name"]
        hit = _rules_hit(r, cov_thr, structural)
        if hit:
            cand.setdefault(nm, {"rules": [], "cluster": None, "keep": None})
            cand[nm]["rules"].extend(hit)

    # R4 只作并列证据；没有 R1/R2/R3 的因子绝不因低 IC 单独入列。
    for nm, c in cand.items():
        r = byname.get(nm)
        if r is not None and _is_stable_noise(r):
            c["rules"].append("R4双目标跨期弱IC(并列)")

    # ---- 守卫
    dropped = {}
    for nm, c in cand.items():
        if nm in protected:
            dropped[nm] = "保护名单（标签）"
        elif nm in parents:
            dropped[nm] = f"存活因子的父依赖 ← {','.join(sorted(parents[nm])[:3])}"
        elif nm not in byname:
            dropped[nm] = "无质检数据"
    for nm in dropped:
        del cand[nm]
    return cand, clusters_out, dropped


def _counts(rows, corr, names, theta, cov_thr, parents, protected, structural):
    cand, cl, _ = _build(rows, corr, names, theta, cov_thr, parents, protected, structural)
    n_r1 = sum(1 for c in cand.values() if any(x.startswith("R1") for x in c["rules"]))
    return {"theta": theta, "cov_thr": cov_thr,
            "clusters": len(cl), "drop_cluster": n_r1,
            "independent_only": len(cand) - n_r1,
            "total_drop": len(cand), "remaining": len(rows) - len(cand),
            "candidates": cand, "cluster_list": cl}


def main() -> int:
    ap = argparse.ArgumentParser(description="把质检数据转成删除候选清单（不删东西）")
    ap.add_argument("--screen", required=True, help="run_screen.py 的输出目录")
    ap.add_argument("--out", required=True, help="交付目录")
    ap.add_argument("--theta", type=float, default=None, help="直接指定相关阈值")
    ap.add_argument("--cov-thr", type=float, default=None, help="直接指定覆盖率阈值")
    args = ap.parse_args()

    sd = Path(args.screen)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = json.loads((sd / "screen_metrics.json").read_text(encoding="utf-8"))
    corr = np.load(sd / "corr.npy")
    names = json.loads((sd / "corr_index.json").read_text(encoding="utf-8"))["names"]
    assert len(names) == corr.shape[0] == corr.shape[1], "相关矩阵与名字表不一致"
    have = {r["name"] for r in rows}
    assert set(names) == have, f"质检表与相关矩阵的因子集不一致：{set(names) ^ have}"

    parents = _parents()
    protected = _protected()
    structural = _structural_cov(rows)
    print(f"因子 {len(rows)} 个 · 父依赖 {len(parents)} 个 · 保护 {len(protected)} 个 · "
          f"覆盖率结构豁免 {len(structural)} 个")

    # ---- 阶梯表：相关与覆盖率阈值暂按旧值展开，正式阈值待全库结果审阅。
    #   θc 固定用 0.95 扫覆盖率；覆盖率固定 0.5 扫 θc；再给一张交叉表。
    ladder = {"theta": [], "cov": [], "cross": []}
    for th in THETA_LADDER:
        c = _counts(rows, corr, names, th, 0.50, parents, protected, structural)
        c.pop("candidates"); c.pop("cluster_list")
        ladder["theta"].append(c)
    for cv in COV_LADDER:
        c = _counts(rows, corr, names, 0.95, cv, parents, protected, structural)
        c.pop("candidates"); c.pop("cluster_list")
        ladder["cov"].append(c)
    for th in THETA_LADDER:
        for cv in COV_LADDER:
            c = _counts(rows, corr, names, th, cv, parents, protected, structural)
            ladder["cross"].append({
                "theta": th, "cov_thr": cv,
                "clusters": c["clusters"], "drop_cluster": c["drop_cluster"],
                "independent_only": c["independent_only"],
                "total_drop": c["total_drop"], "remaining": c["remaining"]})

    (out / "ladder.json").write_text(
        json.dumps(ladder, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n★ θc 阶梯（覆盖率阈值固定 0.50）")
    print(f"  {'θc':>5}{'簇数':>7}{'簇内删':>8}{'独立命中':>9}{'合计删':>8}{'删后剩':>8}")
    for c in ladder["theta"]:
        print(f"  {c['theta']:>5.2f}{c['clusters']:>7}{c['drop_cluster']:>8}"
              f"{c['independent_only']:>9}{c['total_drop']:>8}{c['remaining']:>8}")
    print("\n★ 覆盖率阈值阶梯（θc 固定 0.95）")
    print(f"  {'cov<':>5}{'簇数':>7}{'簇内删':>8}{'独立命中':>9}{'合计删':>8}{'删后剩':>8}")
    for c in ladder["cov"]:
        print(f"  {c['cov_thr']:>5.2f}{c['clusters']:>7}{c['drop_cluster']:>8}"
              f"{c['independent_only']:>9}{c['total_drop']:>8}{c['remaining']:>8}")

    if args.theta is None or args.cov_thr is None:
        print("\n（未指定 --theta/--cov-thr ⇒ 只出阶梯表与簇清单，不出最终候选 CSV）")
        ans = _counts(rows, corr, names, 0.95, 0.50, parents, protected, structural)
        (out / "clusters_theta0.95.json").write_text(
            json.dumps(ans["cluster_list"], ensure_ascii=False, indent=2), encoding="utf-8")
        _dump_groups(out, rows, ans["candidates"])
        return 0

    ans = _counts(rows, corr, names, args.theta, args.cov_thr, parents, protected, structural)
    byname = {r["name"]: r for r in rows}
    with (out / "delete_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "group", "rules", "cluster", "cluster_keep",
                    "cov_mean", "cov_p10", "cs_std_med", "uniq_med", "const_frac",
                    "zero_frac", "ac1", "recent_period", "recent_rankic1",
                    "recent_rankic5", "rankic1", "rankic5", "ic1", "ic5",
                    "stable_weak_ic", "n_days"])
        for nm, c in sorted(ans["candidates"].items()):
            r = byname[nm]
            period, recent_one, recent_five = _recent_rankic(r)
            rk1 = (r.get("rankic") or {}).get("1", {})
            rk5 = (r.get("rankic") or {}).get("5", {})
            ic1 = (r.get("ic") or {}).get("1", {})
            ic5 = (r.get("ic") or {}).get("5", {})
            w.writerow([nm, r["group"], " | ".join(c["rules"]), c["cluster"], c["keep"],
                        r["cov_mean"], r["cov_p10"], r["cs_std_med"], r["uniq_med"],
                        r["const_frac"], r["zero_frac"], r["ac1"],
                        period, recent_one, recent_five, rk1.get("mean"), rk5.get("mean"),
                        ic1.get("mean"), ic5.get("mean"), _is_stable_noise(r), r["n_days"]])
    (out / f"plan_theta{args.theta}_cov{args.cov_thr}.json").write_text(
        json.dumps({"theta": args.theta, "cov_thr": args.cov_thr,
                    "noise_rankic": NOISE_RANKIC,
                    "noise_min_days": NOISE_MIN_DAYS,
                    "noise_min_periods": NOISE_MIN_PERIODS,
                    "noise_period_min_days": NOISE_PERIOD_MIN_DAYS,
                    "total_drop": ans["total_drop"], "remaining": ans["remaining"],
                    "clusters": ans["cluster_list"],
                    "candidates": {k: v["rules"] for k, v in ans["candidates"].items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    _dump_groups(out, rows, ans["candidates"])
    print(f"\n候选清单已写入 {out / 'delete_candidates.csv'}"
          f"（{ans['total_drop']} 个，删后剩 {ans['remaining']}）")
    return 0


def _dump_groups(out: Path, rows: list[dict], cand: dict) -> None:
    """按家族统计删除数 —— 一眼看出「这一刀主要砍在哪一族」。"""
    import collections
    g_all = collections.Counter(r["group"] for r in rows)
    byname = {r["name"]: r for r in rows}
    g_del = collections.Counter(byname[nm]["group"] for nm in cand if nm in byname)
    tbl = [{"group": g, "total": g_all[g], "drop": g_del.get(g, 0),
            "remain": g_all[g] - g_del.get(g, 0)}
           for g in sorted(g_all, key=lambda x: -(g_del.get(x, 0)))]
    (out / "by_group.json").write_text(
        json.dumps(tbl, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'家族':<20}{'总数':>7}{'删除':>7}{'保留':>7}")
    for t in tbl:
        print(f"{t['group']:<20}{t['total']:>7}{t['drop']:>7}{t['remain']:>7}")


if __name__ == "__main__":
    raise SystemExit(main())
