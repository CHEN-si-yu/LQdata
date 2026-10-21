"""市场复审 M0/M2/M3 证据生成（R1）。

- M0：运行 `main.py check`（全库体检，写入 artifacts/checks/<ts>/report.json），
  抽取 61 个市场记录存证；再运行期 freezeverify 保证源与输入未漂移。
- M2：逐因子覆盖与变化统计（mkt_review_lib.m2_stats，冻结判据）。
- M3：逐因子口径元数据表 + `main.py audit-pit --group market` 静态扫描与抽样动态截断复算
  （stdout 存证；动态复算在 audit-pit 自己的沙箱内进行）。

只写 artifacts/research/market_review_20261003/gates/。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "market_review"))

import pandas as pd  # noqa: E402
from mkt_review_lib import REVIEW_ID, axis_days, cfg_load, market_specs, m2_stats  # noqa: E402
import mkt_review_freeze as fz  # noqa: E402

REVIEW_DIR = ROOT / "artifacts" / "research" / REVIEW_ID
GATES = REVIEW_DIR / "gates"


def run_cmd(args, capture=True):
    return subprocess.run([sys.executable, str(ROOT / "main.py"), *args],
                          cwd=ROOT, capture_output=capture, text=True)


def gate_m0(cfg):
    proc = run_cmd(["check"])
    checks = sorted((ROOT / "artifacts" / "checks").glob("*/report.json"))
    if not checks:
        raise RuntimeError("main.py check 未产出 report.json")
    report_path = checks[-1]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    market = [r for r in report["records"] if r.get("kind") == "market"]
    passed = [r for r in market if not r.get("issues")]
    out = {
        "source_report": str(report_path),
        "source_sha256": fz.sha256_file(report_path),
        "check_rc": proc.returncode,
        "cutoff": report.get("cutoff"),
        "market_count": len(market),
        "market_pass": len(passed),
        "records": market,
        "pass": len(market) == 61 and len(passed) == 61 and proc.returncode == 0,
    }
    (GATES / "m0_check_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M0：check rc={proc.returncode}，市场记录 {len(market)}，全过 {len(passed)}，"
          f"pass={out['pass']}")
    return out


def gate_m2(cfg, axis):
    specs = market_specs(cfg)
    records = {}
    for name in sorted(specs):
        records[name] = m2_stats(cfg, specs[name], axis)
    all_pass = all(r["passed"] for r in records.values())
    (GATES / "m2_profiles.json").write_text(
        json.dumps({"records": records, "all_pass": all_pass}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    rows = []
    for name, r in records.items():
        rows.append({
            "name": name, "resolved_start": r["resolved_start"], "align_fill": r["align_fill"],
            "post_start_days": r["post_start_days"], "post_start_finite": r["post_start_finite"],
            "finite_fraction": round(r["post_start_finite_fraction"], 6),
            "distinct_finite_total": r["distinct_finite_total"],
            "thin_years": json.dumps(r["thin_years"], ensure_ascii=False),
            "prefix_finite_cells": r["prefix_finite_cells"], "prefix_leak_cells": r["prefix_leak_cells"],
            "passed": r["passed"],
        })
    pd.DataFrame(rows).to_csv(GATES / "m2_profiles.csv", index=False)
    print(f"M2：{len(records)} 个因子，all_pass={all_pass}；"
          f"不过门：{[n for n, r in records.items() if not r['passed']]}")
    return records


def gate_m3(cfg):
    specs = market_specs(cfg)
    rows = []
    for name in sorted(specs):
        s = specs[name]
        rows.append({
            "name": name, "group": s.group, "desc": s.desc, "formula": s.formula,
            "deps": ",".join(s.deps), "start": s.start,
            "resolved_start": str(s.resolved_start(cfg)), "warmup_days": s.warmup_days,
            "version": s.version, "higher_is_better": s.higher_is_better,
            "is_market": s.is_market,
            "lagged_ok": ",".join(s.lagged_ok or ()),
            "align_fill": ("" if s.align_fill is None else repr(s.align_fill)),
            "note": s.note,
        })
    meta_csv = GATES / "m3_metadata.csv"
    pd.DataFrame(rows).to_csv(meta_csv, index=False)

    static = run_cmd(["audit-pit", "--group", "market", "--static-only"])
    (GATES / "m3_audit_pit_static.txt").write_text(static.stdout + static.stderr, encoding="utf-8")
    dyn = run_cmd(["audit-pit", "--group", "market", "--sample", "1", "--jobs", "8"])
    (GATES / "m3_audit_pit_dynamic.txt").write_text(dyn.stdout + dyn.stderr, encoding="utf-8")
    static_flags = static.stdout.count("✘")
    dyn_flags = dyn.stdout.count("✘")
    out = {
        "metadata_csv": str(meta_csv), "metadata_sha256": fz.sha256_file(meta_csv),
        "audit_pit_static_rc": static.returncode, "audit_pit_static_flags": static_flags,
        # M3 判据（冻结文本）：逐因子口径成文 + audit-pit **静态**检查通过。
        # 抽样动态复算仅作辅助证据记录（其工具限制见 dynamic_aux.note），不参与门判定。
        "pass": static.returncode == 0 and static_flags == 0,
        "dynamic_aux": {
            "rc": dyn.returncode, "flags": dyn_flags,
            "note": "动态抽检仅辅助证据：sector 族 6 因子的截断重算因 tdx_blocks（快照表，无日期列）"
                    "被审计工具拒绝（工具限制而非数值不一致）；该族的因果证明由 M1（research verify）承担，"
                    "若 M1 同样受快照表限制，将在裁决中如实记「无法执行」而非「通过」。",
        },
    }
    (GATES / "m3_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M3：static rc={static.returncode} flags={static_flags}；"
          f"dynamic rc={dyn.returncode} flags={dyn_flags}；pass={out['pass']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-freeze-check", action="store_true")
    ap.add_argument("--only", choices=["all", "m0", "m2", "m3"], default="all",
                    help="只重跑其中一门（其余保留既有证据）")
    args = ap.parse_args()
    cfg = cfg_load()
    if not args.skip_freeze_check:
        fz.verify(cfg, fz.DEFAULT_OUT)
    GATES.mkdir(parents=True, exist_ok=True)
    if args.only in ("all", "m0"):
        gate_m0(cfg)
    if args.only in ("all", "m2"):
        axis = axis_days(cfg)
        print(f"交易日轴：{len(axis)} 天（{axis[0]}..{axis[-1]}）")
        gate_m2(cfg, axis)
    if args.only in ("all", "m3"):
        gate_m3(cfg)
    print("M0–M3 证据完成 ->", GATES)


if __name__ == "__main__":
    main()
