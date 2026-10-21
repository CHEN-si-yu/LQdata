# model/history/model_archive_20261002

从 `model/experiments` 归档出来的模型单元。**归档策略：只保留脚本。**

保留内容：
- `run.py` / `model.py` / `analysis.py`（训练与评价入口，逐字节同源）
- `model_info/*.json`（若存在：审计结论、两策略结果等小体积指标）

**已删除**：`model_train/`（含每折 `best.pt` 与 `score_predictions.npy`）、`model_pred/`、
`model_logs/`。因此归档单元**不能直接复现历史打分**，需要重新训练（脚本与配方都在，
`training_info.json` 若保留则含完整 RECIPE）。

保留在 `model/experiments` 的 30 个单元带完整数据，其中 4 个（V132/V133/V136/V137）是
两个冠军配方（融合权重 0.7 / 费用判据）的**未跑完的复现种子**，保留供继续复现。

详见 `ARCHIVE_MANIFEST.md`。

---

> 注（2026-10-05 补记）：本 README 所述"保留在 `model/experiments` 的 30 个单元"已于 2026-10-05
> 归档至 `history/model_archive_20261005/`（只留脚本 + `model_info/*.json`）。
> 旧快照不可复现（副本 10-02 删除、快照 10-05 被覆盖）；`experiments/` 现保留 10 个新快照单元。
