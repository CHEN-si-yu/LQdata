#!/usr/bin/env python3
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
V11 = HERE.parent / "V11"
CUTOFF = "2023Q4"
IMPORTANCE_QUARTERS = [
    f"{year}Q{quarter}"
    for year in range(2020, 2024)
    for quarter in range(1, 5)
]
EXPECTED = [
    "mf_tier_flow_agreement_20",
    "rel_mom_ind_3d",
    "rel_mom_ind_5d",
    "gap_down_recover_freq_20d",
    "rel_mom_ind_250d",
    "rel_vol_ind_20d",
    "rel_mom_ind_10d",
    "tail_risk_pct_60",
    "volume_dry_up",
    "rel_mom_ind_60d",
    "price_to_52w_high",
    "id2_close_vs_pm_vwap_20",
    "inside_bar_count_20",
    "ret_kurt_20",
    "cost_skew_ratio",
    "overnight_sign_consistency_20d",
    "margin_balance_volatility_20d",
    "mfx_index_range",
    "mf_open_close_divergence_10d",
    "bollinger_width_20",
]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

normalized = defaultdict(list)
source_files = []
for quarter in IMPORTANCE_QUARTERS:
    result_path = V11 / "quarters" / quarter / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("quarter") != quarter or result.get("state") != "complete":
        raise RuntimeError(f"{quarter}: result is incomplete or mislabeled")
    top20 = result.get("top_features_by_gain", [])[:20]
    if len(top20) != 20:
        raise RuntimeError(f"{quarter}: expected 20 training importance entries")
    names = [str(row["feature"]) for row in top20]
    if len(names) != len(set(names)):
        raise RuntimeError(f"{quarter}: duplicate feature in importance top-20")
    denominator = sum(float(row["gain"]) for row in top20)
    if denominator <= 0:
        raise RuntimeError(f"{quarter}: invalid total top-20 gain")
    for row in top20:
        normalized[str(row["feature"])].append(float(row["gain"]) / denominator)
    source_files.append({
        "quarter": quarter,
        "path": str(result_path),
        "bytes": result_path.stat().st_size,
        "sha256": sha256(result_path),
        "top20_features": names,
    })

if len(source_files) != 16:
    raise RuntimeError("expected exactly 16 pre-holdout training importance folds")

ranked = sorted(
    normalized,
    key=lambda feature: (
        -len(normalized[feature]),
        -sum(normalized[feature]) / len(normalized[feature]),
        feature,
    ),
)
selected = ranked[:20]
if selected != EXPECTED:
    raise RuntimeError(
        "Programmatic 16-fold ranking does not match the pre-inspected expected list: "
        + json.dumps({"computed": selected, "expected": EXPECTED}, ensure_ascii=False)
    )

ranking_rows = []
for rank, feature in enumerate(ranked, 1):
    values = normalized[feature]
    ranking_rows.append({
        "rank": rank,
        "feature": feature,
        "top20_quarter_count": len(values),
        "mean_normalized_gain_when_top20": sum(values) / len(values),
        "selected_v33": feature in selected,
    })
with (HERE / "training_importance_ranking.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(ranking_rows[0]))
    writer.writeheader()
    writer.writerows(ranking_rows)

selection = {
    "cutoff_quarter": CUTOFF,
    "importance_quarters": IMPORTANCE_QUARTERS,
    "rule": "Sort V11 training-fold features by descending top-20 quarter count, descending mean gain normalized within each fold's top-20 list when present, then ascending feature name; freeze first 20.",
    "selected_features": selected,
    "selected_statistics": [row for row in ranking_rows if row["selected_v33"]],
    "selection_source_sha256": [{"quarter": row["quarter"], "sha256": row["sha256"]} for row in source_files],
}
(HERE / "selected_feature_list.json").write_text(
    json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(HERE / "importance_hashes.json").write_text(
    json.dumps({
        "algorithm": "SHA-256",
        "cutoff_quarter": CUTOFF,
        "source": "V11/quarters/<quarter>/result.json top_features_by_gain from 2020Q1 through 2023Q4 only",
        "importance_files": source_files,
        "selection_program_sha256": sha256(HERE / "freeze_protocol.py"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

quarters = [f"{year}Q{quarter}" for year in range(2024, 2027) for quarter in range(1, 5)]
quarters = quarters[:10]
recipe = {
    "implementation": "V11 LGBMRanker recipe, as used by V27",
    "objective": "lambdarank",
    "metric": "ndcg@2",
    "label": "ascending within-date rank of forward label_ret_5d; ties share rank; integer relevance 0..3",
    "factor_features": selected,
    "asset_one_hot_features": [f"asset_{i}" for i in range(4)],
    "n_estimators": 180,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 80,
    "feature_fraction": 0.80,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "reg_alpha": 2.0,
    "reg_lambda": 20.0,
    "label_gain": [0, 1, 2, 3],
    "seed": 17,
    "threads_per_worker": 1,
    "early_stopping": False,
    "sample_weight": None,
    "deterministic": True,
    "force_col_wise": True,
}
import_script_hash = sha256(V11 / "model.py")
protocol = {
    "version": "V33",
    "status": "frozen_before_any_V33_training",
    "factor_selection": selection,
    "protocol_cutoff": "2023Q4",
    "holdout_quarters": quarters,
    "holdout_dates": ["2024-01-02", "2026-06-30"],
    "walk_forward": {
        "one_new_model_per_quarter": True,
        "each_model_scores_its_test_quarter_only": True,
        "training_label": "label_ret_5d",
        "purge_rule": "train_day_index + 5 < first_test_day_index",
        "purged_trading_sessions": 5,
        "max_label_date_must_equal_train_last": True,
        "future_quarter_feature_rows_excluded": True,
        "no_2020_2023_prediction_models": True
    },
    "model_recipe": recipe,
    "portfolio_rule": "At each shared V11 first-session monthly signal close, rank banks using the quarter-local V33 LambdaRank score; equal-weight the top two; execute next-session open.",
    "ledger": {
        "source": "V11 final corrected account-level ledger",
        "initial_equity": 100000.0,
        "execution": "T+1 open",
        "participation_limit": 0.01,
        "lot_size": 100,
        "final_holdout_exit": "Mark to 2026-06-30 close; do not force liquidate, matching V24/V26/V27."
    },
    "comparisons": [
        "V27 four-factor LambdaRank Top2",
        "V24 fixed-rank Top2",
        "V26 mf_tier_flow_agreement_20 Top2",
        "V11 LambdaRank Top2",
        "60d momentum Top2",
        "equal-weight hold",
    ],
    "resources": {
        "max_cpu_workers": 4,
        "threads_per_worker": 1,
        "gpu": False,
        "memory_hard_limit_gib": 180,
        "scheduler_soft_used_limit_gib": 145
    },
    "no_parameter_sweep": True,
    "no_holdout_based_selection_or_tuning": True,
    "no_early_stopping": True,
    "v11_model_source_sha256": import_script_hash,
    "protocol_frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "freeze_script_sha256": sha256(HERE / "freeze_protocol.py"),
}
(HERE / "protocol.json").write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({
    "feature_count": len(selected),
    "selected_features": selected,
    "importance_sha256_count": len(source_files),
    "quarters": quarters,
    "protocol_sha256": sha256(HERE / "protocol.json"),
}, ensure_ascii=False, indent=2))

