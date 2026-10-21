"""市场复审裁决（R3）：按冻结规则树合并 M0–M7 → 逐因子结论 + 总表。

输入：registration/channel_direction_table.json、gates/m0_check_report.json、
gates/m2_profiles.json、gates/m3_summary.json、gates/m4_pairs.json、
gates/m5_signal.json、gates/m6_incremental.json、gates/m1_verification.json（R2 后）。
输出：verdicts/verdicts.json、verdicts/市场因子复审总表.csv、summary 打印。

处置口径（用户批准）：「只出结论与建议清单」——不删除、不降级、不动生产状态。
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

import pandas as pd  # noqa: E402
from mkt_review_lib import M4_EXPLAIN, M4_FAIL, PRIMARY_HORIZONS, REVIEW_ID  # noqa: E402

REVIEW_DIR = ROOT / "artifacts" / "research" / REVIEW_ID
GATES = REVIEW_DIR / "gates"
VERDICTS = REVIEW_DIR / "verdicts"


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", type=Path, default=GATES / "m1_verification.json")
    args = ap.parse_args()

    table = {f["name"]: f for f in load_json(
        REVIEW_DIR / "registration" / "channel_direction_table.json")["factors"]}
    m0 = load_json(GATES / "m0_check_report.json")
    m2 = load_json(GATES / "m2_profiles.json")
    m3 = load_json(GATES / "m3_summary.json")
    m4 = load_json(GATES / "m4_pairs.json")
    m5 = load_json(GATES / "m5_signal.json")
    m6 = load_json(GATES / "m6_incremental.json")
    m1 = load_json(args.m1)

    m0_by = {r["factor"]: (not r.get("issues")) for r in m0["records"]}
    m2_by = {n: r["passed"] for n, r in m2["records"].items()}
    m3_pass = bool(m3["pass"])

    # M4 每因子最差对
    worst = {}
    for p in m4["pairs"]:
        for side in (p["a"], p["b"]):
            other = p["b"] if side == p["a"] else p["a"]
            if side not in worst or p["abs_rho"] > worst[side]["abs_rho"]:
                worst[side] = {"other": other, "rho": p["rho"], "abs_rho": p["abs_rho"],
                               "flag": p["flag"], "n": p["n"]}

    VERDICTS.mkdir(parents=True, exist_ok=True)
    rows, machine = [], {}
    for name in sorted(table):
        f = table[name]
        ch = f["channel"]
        d = f["direction"] or 0
        m0p, m2p = m0_by.get(name), m2_by.get(name)
        w = worst.get(name, {})
        m4_flag = w.get("flag", "pass")
        r5, r6 = m5[name], m6[name]
        g5, g6 = r5["gate_pass_h"], r6["gate_pass_h"]
        m1_rec = (m1 or {}).get("records", {}).get(name)
        m1p = None if m1 is None else bool(m1_rec and m1_rec.get("passed"))

        notes = []
        if ch == "green_card":
            gate_ok = bool(m0p) and bool(m2p) and m3_pass
            verdict = "绿卡准入" if gate_ok else "待修订"
            action = "保留（描述用途）" if gate_ok else "修订后复审"
            if not gate_ok:
                notes.append("结构类门槛未过：" + (
                    ("M0 " if not m0p else "") + ("M2 " if not m2p else "")
                    + ("M3 " if not m3_pass else "")).strip())
            if m4_flag in ("fail", "explain"):
                notes.append(f"M4 记录：与 {w['other']} |ρ|={w['abs_rho']:.3f}（{'重复标注' if m4_flag=='fail' else '需说明'}）")
            if g5 or g6:
                notes.append("辅助：该描述量在回顾性直接检验/增量检验中亦有表现（不改变绿卡定性）")
        else:
            prereq = bool(m0p) and bool(m2p) and m3_pass and bool(m1p) and m1 is not None
            if m1 is None:
                notes.append("M1 未完成（缺 gates/m1_verification.json）")
            if not prereq:
                verdict, action = "待修订", "修订后复审"
                miss = [x for x, ok in (("M0", m0p), ("M1", m1p), ("M2", m2p), ("M3", m3_pass)) if not ok]
                notes.append("先决门未过：" + "/".join(miss))
            elif m4_flag == "fail":
                verdict, action = "建议剔除", "建议剔除（仅建议，未执行）"
                notes.append(f"重复：与 {w['other']} |ρ|={w['abs_rho']:.3f}>0.95，需合并取舍")
            elif g5 and g6:
                verdict, action = "严格通过", "保留（独立信号候选）"
                if m4_flag == "explain":
                    notes.append(f"M4 需书面论证：与 {w['other']} |ρ|={w['abs_rho']:.3f}（0.90–0.95）")
            elif g5:
                # M5 已获支持、由增量检验裁决（门序：第 3 条）
                neg = None
                for h in PRIMARY_HORIZONS:
                    r = r6.get(f"h{h}") or {}
                    if not r.get("insufficient") and r.get("declared_t") is not None \
                            and r["declared_t"] < 0:
                        neg = h
                if neg is not None:
                    verdict, action = "建议剔除", "建议剔除（仅建议，未执行）"
                    notes.append(f"增量检验符号与登记方向相反（h={neg}，风格投影特征，AGENTS.md D.1）")
                else:
                    verdict, action = "待修订", "修订后复审"
                    notes.append("M6 增量不足：同号但 declared t<2.0 或样本不足")
                    notes.append("建议：转为描述用途（若绿卡边界可裁定）或重新论证机制后复审")
            else:
                # M5 未获支持（门序：第 2 条先行）→ 待修订；如实记录增量检验观察
                verdict, action = "待修订", "修订后复审"
                notes.append("M5 未获支持：主期限 closed 区间含 0 或分年同号不足")
                negs = [h for h in PRIMARY_HORIZONS
                        if not (r6.get(f"h{h}") or {}).get("insufficient")
                        and (r6.get(f"h{h}") or {}).get("declared_t") is not None
                        and r6[f"h{h}"]["declared_t"] < 0]
                if negs:
                    notes.append(f"观察：增量检验在 h={negs} 符号反转（投影特征；M5 未获支持先落待修订，"
                                 "该观察列为优先复核项）")
                elif not g6:
                    notes.append("M6 增量不足：同号但 declared t<2.0 或样本不足")
                notes.append("建议：转为描述用途（若绿卡边界可裁定）或重新论证机制后复审")
        machine[name] = {
            "channel": ch, "direction": d, "prior_strength": f.get("prior_strength"),
            "m0": m0p, "m1": m1p, "m2": m2p, "m3": m3_pass,
            "m4_worst": w or None, "m5_pass_h": g5, "m6_pass_h": g6,
            "verdict": verdict, "action": action, "notes": notes,
        }
        h1 = r5.get("h1") or {}
        h5r = r5.get("h5") or {}
        i1 = (r6.get("h1") or {})
        i5 = (r6.get("h5") or {})
        rows.append({
            "因子名称": name, "因子族": f["family"], "复审通道": ch,
            "类别": f["category_cn"], "规则依据": f["rule_basis"], "机制/假设": f["mechanism"],
            "登记方向": ("" if not d else ("+1" if d > 0 else "-1")),
            "先验强度": f.get("prior_strength") or "",
            "M0结构": "" if m0p is None else ("过" if m0p else "不过"),
            "M1工程因果": "" if m1p is None else ("过" if m1p else "不过"),
            "M2覆盖变化": "" if m2p is None else ("过" if m2p else "不过"),
            "M3口径": "过" if m3_pass else "不过",
            "M4最大相关对": w.get("other", ""), "M4最大绝对相关": (round(w["abs_rho"], 4) if w else ""),
            "M4判定": {"pass": "过", "explain": "需论证", "fail": "不过"}[m4_flag],
            "M5_h1_rho": h1.get("rho"), "M5_h1_t": h1.get("t"),
            "M5_h5_rho": h5r.get("rho"), "M5_h5_t": h5r.get("t"),
            "M5分年同号": (f"{h1.get('years_same_sign')}/{h1.get('valid_years')}" if d else ""),
            "M5过门期限": g5 or "",
            "M6_h1_t": i1.get("t"), "M6_h1_declared_t": i1.get("declared_t"),
            "M6_h5_t": i5.get("t"), "M6_h5_declared_t": i5.get("declared_t"),
            "M6剩余方差比": (round(i1["residual_variance_fraction"], 4)
                            if i1.get("residual_variance_fraction") is not None else ""),
            "M6过门期限": g6 or "",
            "复审结论": verdict, "建议动作": action,
            "用途": ("数据用途" if ch == "green_card" else
                    "独立信号候选" if verdict == "严格通过" else "待定"),
            "备注": "；".join(notes),
        })

    pd.DataFrame(rows).to_csv(VERDICTS / "市场因子复审总表.csv", index=False)
    from collections import Counter
    counts = Counter(r["复审结论"] for r in rows)
    summary = {"review_id": REVIEW_ID, "counts": dict(counts),
               "channels": dict(Counter(r["复审通道"] for r in rows)),
               "m1_present": m1 is not None}
    (VERDICTS / "verdicts.json").write_text(
        json.dumps({"summary": summary, "records": machine}, ensure_ascii=False, indent=2,
                   default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("总表 ->", VERDICTS / "市场因子复审总表.csv")


if __name__ == "__main__":
    main()
