# R17 因子批次验证（2026-10-19）

隔离工作目录：本目录之外不改动 `releases/`、`trainingdata/` 或 `featureengineering/`。

## 最终结果

见 **[RESULTS.md](RESULTS.md)**。结论：R17 的 50 个因子作为一个批次加入，
在 25Q3–26Q2、四个 release 模型、每季四折的配对实验中**没有提升样本外截面预测能力**：
四个模型的 IC_1d / IC_5d / RankIC_1d / RankIC_5d / ICIR 全部下降；
64 个折级配对 test 的 ΔRankIC_5d 均值为 −0.00338（22/64 为正）。
收益差异大且集中在少数季度，不能作为有效性证据。建议不将 50 个因子整批并入 releases 特征集。

## 内容

- `RESULTS.md`：最终 A/B 对比、分季度/折级统计、单因子诊断、结论与限制
- `results/PROTOCOL.md`：预登记协议（批次定义、快照、切分、口径）
- `results/quarter_model.csv`、`fold_valid.csv`、`fold_test.csv`、`comparison.json`、
  `univariate_new_factors.csv`：机器可读结果
- `snapshots/A_260`、`snapshots/B_310`：同源同日期、仅差 R17 50 列的隔离快照
- `units/{A,B}_{V220,V230,V240,V252}/`：8 个实验单元（release 脚本逐字复制 + 仅特征清单补丁）
- `scripts/`：生成、训练队列、汇总脚本
- `logs/`：快照构建、队列、逐单元训练/分析日志

## 状态

- 8 单元 × 16 折 = 128 折全部完成；8/8 `analysis.py --audit` 通过，
  `weights_reloaded=True`，`max_cpu_reload_error=0.0`。
- 训练队列 `scripts/run_queue.sh` 已结束；单折峰值内存 7.0–8.2 GiB，4 折并发。
- 单卡 RTX 4080 SUPER；cgroup 内存上限 90 GiB。

## 复现

```bash
python scripts/generate_units.py
bash scripts/run_queue.sh
python scripts/evaluate_batches.py
```
