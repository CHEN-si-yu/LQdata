"""市场复审 M4/M5/M6 证据生成（R1；只读生产数据，只写复审 gates/）。

- M4：61×61 时序 Spearman（共同有效日 ≥250），阈值 0.90/0.95（冻结）。
- M5：预登记方向的时序预测检验（主目标=标签派生前视等权市场收益 h∈{1,5}；
  辅助目标=前视离散度/宽度 + 沪深300/中证500 open-to-open 前视收益，仅报告）。
- M6：HAC-OLS 增量检验（冻结控制集、自控制排除、Bartlett lag 19）。
全量 61 个因子统一计算（不论通道），裁决只对严格通道使用门判据。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "market_review"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mkt_review_lib import (  # noqa: E402
    M4_EXPLAIN, M4_FAIL, M6_CONTROLS, PRIMARY_HORIZONS, REVIEW_ID,
    axis_days, build_targets, cfg_load, incremental_test, m5_pass, m6_pass,
    rank_corr_matrix, read_market_on_axis, signal_test,
)
import mkt_review_freeze as fz  # noqa: E402

REVIEW_DIR = ROOT / "artifacts" / "research" / REVIEW_ID
GATES = REVIEW_DIR / "gates"
TABLE = REVIEW_DIR / "registration" / "channel_direction_table.json"

AUX_SUFFIX = ["aux_dispersion", "aux_breadth", "aux_csi300_open", "aux_csi500_open"]


def load_table():
    t = json.loads(TABLE.read_text(encoding="utf-8"))
    return {f["name"]: f for f in t["factors"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-freeze-check", action="store_true")
    args = ap.parse_args()
    cfg = cfg_load()
    if not args.skip_freeze_check:
        fz.verify(cfg, fz.DEFAULT_OUT)
    GATES.mkdir(parents=True, exist_ok=True)
    table = load_table()
    axis = axis_days(cfg)

    series = {name: read_market_on_axis(cfg, name, axis) for name in sorted(table)}
    targets = build_targets(cfg, axis)

    # ── M4 ──
    names, rho, pairs = rank_corr_matrix(series)
    pd.DataFrame(rho, index=names, columns=names).to_csv(GATES / "m4_corr_matrix.csv")
    pair_rows = []
    for p in pairs:
        flag = ("fail" if abs(p["rho"]) > M4_FAIL else
                "explain" if abs(p["rho"]) > M4_EXPLAIN else "pass")
        pair_rows.append({**p, "abs_rho": abs(p["rho"]), "flag": flag})
    pd.DataFrame(pair_rows).to_csv(GATES / "m4_pairs.csv", index=False)
    (GATES / "m4_pairs.json").write_text(json.dumps({
        "min_overlap": 250, "thresholds": {"explain": M4_EXPLAIN, "fail": M4_FAIL},
        "pairs": pair_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hi = [p for p in pair_rows if p["flag"] != "pass"]
    print(f"M4：{len(pair_rows)} 对；0.90–0.95 需论证 {sum(1 for p in hi if p['flag']=='explain')} 对，"
          f">0.95 判不过 {sum(1 for p in hi if p['flag']=='fail')} 对")
    for p in hi:
        print(f"   {p['flag']:7s} {p['a']} ↔ {p['b']}: ρ={p['rho']:.3f} (n={p['n']})")

    # ── M5 ──
    m5 = {}
    for name in sorted(table):
        declared = table[name]["direction"] or 0
        row = {"declared_direction": declared}
        for h in PRIMARY_HORIZONS:
            row[f"h{h}"] = signal_test(series[name], targets[f"primary_ew_{h}"], axis, h,
                                       declared=declared)
            aux = {}
            for suffix in AUX_SUFFIX:
                r = signal_test(series[name], targets[f"{suffix}_{h}"], axis, h, declared=declared)
                aux[suffix] = {"rho": r["rho"], "t": r["t"], "days": r["days"]}
            row[f"h{h}"]["aux"] = aux
        row["gate_pass_h"] = m5_pass(row) if declared else None
        m5[name] = row
    (GATES / "m5_signal.json").write_text(
        json.dumps(m5, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    rows = []
    for name, row in m5.items():
        for h in PRIMARY_HORIZONS:
            r = row[f"h{h}"]
            rows.append({
                "name": name, "direction": row["declared_direction"], "h": h,
                "days": r["days"], "rho": r["rho"], "t": r["t"],
                "ci95_low": r["ci95_low"], "ci95_high": r["ci95_high"],
                "declared_rho": r["declared_rho"], "declared_t": r["declared_t"],
                "declared_ci_low": r["declared_ci95_low"], "declared_ci_high": r["declared_ci95_high"],
                "valid_years": r["valid_years"], "years_same_sign": r["years_same_sign"],
                "gate": (row["gate_pass_h"] == h),
            })
    pd.DataFrame(rows).to_csv(GATES / "m5_signal.csv", index=False)
    n_pass = sum(1 for r in m5.values() if r["gate_pass_h"])
    print(f"M5：严格候选 {sum(1 for r in m5.values() if r['declared_direction'])} 个，"
          f"过 M5 判据 {n_pass} 个：{[n for n, r in m5.items() if r['gate_pass_h']]}")

    # ── M6 ──
    m6 = {}
    for name in sorted(table):
        declared = table[name]["direction"] or 0
        controls = {c: series[c] for c in M6_CONTROLS if c != name}
        row = {"declared_direction": declared, "self_excluded": name in M6_CONTROLS}
        for h in PRIMARY_HORIZONS:
            r = incremental_test(series[name], controls, targets[f"primary_ew_{h}"], h,
                                 declared=declared)
            r["declared_direction"] = declared
            row[f"h{h}"] = r
        row["gate_pass_h"] = m6_pass(row) if declared else None
        m6[name] = row
    (GATES / "m6_incremental.json").write_text(
        json.dumps(m6, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    rows = []
    for name, row in m6.items():
        for h in PRIMARY_HORIZONS:
            r = row[f"h{h}"]
            rows.append({
                "name": name, "direction": row["declared_direction"], "h": h,
                "n": r.get("n"), "insufficient": r.get("insufficient"),
                "beta": r.get("beta"), "t": r.get("t"),
                "declared_beta": r.get("declared_beta"), "declared_t": r.get("declared_t"),
                "residual_variance_fraction": r.get("residual_variance_fraction"),
                "condition_number": r.get("condition_number"),
                "gate": (row["gate_pass_h"] == h),
            })
    pd.DataFrame(rows).to_csv(GATES / "m6_incremental.csv", index=False)
    n_pass = sum(1 for r in m6.values() if r["gate_pass_h"])
    print(f"M6：过判据 {n_pass} 个：{[n for n, r in m6.items() if r['gate_pass_h']]}")
    print("M4–M6 证据完成 ->", GATES)


if __name__ == "__main__":
    main()
