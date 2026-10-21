# 已删除的历史快照副本（清单）

删除时间：2026-10-02 11:27:57

删除原因：这三份都是**训练快照的完整副本**（各约 7 GB、54 个 parquet），属于"上一版/冻结版数据"，不是脚本或结论；它们的用途（九年逐位对比、对齐核验）已经完成，结论保存在 `docs/` 与 `dsh_scratch/snapshot_update_report.md`。

- `history/diversification_20260930/trainingdata_frozen` — 删除时 7.0G
  - 说明：V46–V69 使用的冻结快照（484 列，last_upstream_day=2026-09-29，built_at 2026-09-30 11:46:42）
- `history/snapshot_alignment_20260930_113324/previous_trainingdata` — 删除时 7.0G
  - 说明：快照对齐前的上一版（484 列，built_at 2026-09-29 21:51:56）
- `history/snapshot_update_20261001/previous_trainingdata` — 删除时 7.0G
  - 说明：本轮快照重建前的上一版（484 列，built_at 2026-09-30 22:54:50）

**保留**：各目录下的脚本、`meta.json`、`build.log`、对比 JSON 与报告等小体积文件。

**影响**：需要这些历史快照才能原样复现的单元（如 V66/V69 使用 `trainingdata_frozen`）不能再按原快照重训；如需重训，请改用当前快照（486 列）并相应调整 `RECIPE["snapshot_cutoff"]`。
