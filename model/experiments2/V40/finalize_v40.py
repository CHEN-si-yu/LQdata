#!/usr/bin/env python3
"""Complete V40 input and resource audit metadata after the frozen replay."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
manifest_path = HERE / "cache_hashes.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
known = {item["path"] for item in manifest["hashes"]}
prediction_files = sorted((HERE.parent / "V11" / "quarters").glob("*/predictions.csv"))
added = []
for path in prediction_files:
    if path.parent.name[:4] >= "2024" or str(path.resolve()) in known:
        continue
    added.append({
        "role": "V11 cached quarterly predictions loaded to derive the full monthly schedule; pre-holdout schedules were discarded",
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
    })
manifest["hashes"].extend(added)
manifest["scope"] = (
    "V40 runner, V11 corrected ledger and all quarterly prediction files read to derive "
    "the monthly schedule, factor/price/amount input caches, and frozen V24/V34/V26 "
    "protocols and ledger references."
)
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

resource_path = HERE / "resource.log"
resource = json.loads(resource_path.read_text(encoding="utf-8"))
total_kib = 461667644
available_kib = 369995824
total_gib = total_kib / 1024 / 1024
available_gib = available_kib / 1024 / 1024
used_gib = total_gib - available_gib
resource["shared_memory_at_launch"] = {
    "captured_at": resource["started_at"],
    "total_gib": round(total_gib, 2),
    "available_gib": round(available_gib, 2),
    "used_as_total_minus_available_gib": round(used_gib, 2),
    "used_fraction": round(used_gib / total_gib, 4),
    "under_180_gib_and_50_percent": used_gib < 180 and used_gib / total_gib < 0.5
}
resource["within_address_space_limit"] = resource["peak_child_rss_gib"] < resource["address_space_limit_gib"]
resource_path.write_text(json.dumps(resource, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

audit_path = HERE / "audit.json"
audit = json.loads(audit_path.read_text(encoding="utf-8"))
audit["resource_controls"] = {
    "shared_memory_at_launch": resource["shared_memory_at_launch"],
    "hard_address_space_limit_gib": resource["address_space_limit_gib"],
    "peak_child_rss_gib": resource["peak_child_rss_gib"],
    "within_hard_limit": resource["within_address_space_limit"],
    "numerical_threads": 1
}
audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

start_path = HERE / "resource_start.txt"
start_path.write_text(
    "Actual V40 process launch snapshot, from the same remote call just before launch:\n"
    f"captured_at={resource['started_at']}\n"
    f"total_gib={total_gib:.2f}\n"
    f"available_gib={available_gib:.2f}\n"
    f"used_as_total_minus_available_gib={used_gib:.2f}\n"
    f"used_fraction={used_gib / total_gib:.4f}\n"
    "under_180_gib_and_50_percent=True\n",
    encoding="utf-8"
)
print(json.dumps({
    "added_pre_holdout_prediction_hashes": len(added),
    "total_hashes": len(manifest["hashes"]),
    "resource_controls": audit["resource_controls"]
}, ensure_ascii=False, indent=2))
