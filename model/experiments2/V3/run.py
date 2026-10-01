#!/usr/bin/env python3
"""V3候选入口：只读复用V1训练工件，只执行布局检查或独立V2动作审计。"""
import argparse
import json
from pathlib import Path
import model as M
import analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.layout_only:
        print(json.dumps(M.validate_layout(), ensure_ascii=False, indent=2))
        return
    analysis.main(["--audit"] if args.audit else [])
    audit = json.loads((M.ROOT / "model_info" / "final_audit.json").read_text(encoding="utf-8"))
    action = audit["metrics_total"]["action"]
    print(json.dumps({"unit": audit["unit"],
                      "return_value": action["return_value"],
                      "max_drawdown": action["max_drawdown"],
                      "total_fees": action["total_fees"],
                      "trades": action["trades"],
                      "alpha_vs_buy_hold": action["alpha_vs_buy_hold"],
                      "lineage_unchanged": audit["lineage"]["source_unchanged_during_run"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
