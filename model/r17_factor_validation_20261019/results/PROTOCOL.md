# R17 因子批次验证协议（预登记）

- 生成时间：2026-10-19
- 工作目录：`/autodl-fs/data/model/r17_factor_validation_20261019/`（所有产物均在此目录内）
- 目的：验证 R17 新增 50 个正式股票因子对当前四个 release 模型在 25Q3–26Q2 标准评价窗上的增量效果。

## 批次定义

| 批次 | 股票特征数 | 定义 |
|---|---:|---|
| A（基线） | V230/V240/V252: 260；V220: 146 | 当前 releases 四个模型 2026-10-08 实战重训所用的特征集（LIVE 迁移清单） |
| B（新批次） | V230/V240/V252: 310；V220: 196 | A + R17 正式交付的 50 个股票因子 |

新增 50 个因子的分支归属：
- fund（30）：资产构成 20 + 负债构成 9 + reopen 1
- flow（20）：筹码几何 15 + 筹码事件 1 + 截面网络 4
- technical（0）

V220 的 flow/fund/technical 三分支保持互斥且覆盖全部特征；V230/V240/V252 的 MLP 使用全部股票特征，flow/fund 清单用于覆盖率掩码，technical 清单用于独立技术面分支。

## 数据快照

- **A / A_260**：260 股票特征、47 市场因子、5 标签；轴 2018-01-02 ~ 2026-10-16（2130 天 × 2115 只）；meta sha256 `bac19ce3a337a48eb7dfb2df956901097929aefd97549e624651068e82e898f0`
- **B / B_310**：310 股票特征、47 市场因子、5 标签；轴 2018-01-02 ~ 2026-10-16（2130 天 × 2115 只）；meta sha256 `daf9cfb276561f6eb4ef572ae561a51392821b6bee67b49ee3c69a62e5b308f5`

两个快照由 `preparingdata.py --features-from` 从同一份 `featureengineering/data` 源、同一日期范围全量构建；A/B 共有的 260 列在 2018 与 2026 年逐格相同（构建后核对，0 列差异）。

## 模型与训练

| 单元 | 基模型 | 批次 | 股票特征 | 分支 (flow/fund/tech) | 快照 |
|---|---|---|---:|---|---|
| A_V220 | V220 | A | 146 | 27/87/32 | A_260 |
| A_V230 | V230 | A | 260 | 213/240/32 | A_260 |
| A_V240 | V240 | A | 260 | 213/240/32 | A_260 |
| A_V252 | V252 | A | 260 | 213/240/32 | A_260 |
| B_V220 | V220 | B | 196 | 47/117/32 | B_310 |
| B_V230 | V230 | B | 310 | 233/270/32 | B_310 |
| B_V240 | V240 | B | 310 | 233/270/32 | B_310 |
| B_V252 | V252 | B | 310 | 233/270/32 | B_310 |

- 基模型代码：`releases/V220|V230|V240|V252` 的 `run.py`/`model.py`/`analysis.py` 逐字复制；仅在 `model.py` 末尾追加特征清单与快照钉住补丁。
- 使用实验模式几何：`2025Q3`, `2025Q4`, `2026Q1`, `2026Q2`，每季 4 折；训练/验证/测试切分与 `docs/baseline.md` 一致（purge 21 交易日）。
- 随机种子：35253（各折 seed+fold−1），与 releases 当前权重一致。
- 每个单元 16 折，共 8 单元 × 16 = 128 折；已完成折由 `complete.json` 断点跳过。
- 训练并发 4 折/单元；`MX_THREADS=4`；GPU：RTX 4080 SUPER 32GB；cgroup 内存上限 90GiB；实测单折峰值 7.0–8.2GiB。

## 评价

- 季度级：四折打分直接相加（项目既有口径），由 `analysis.py --audit` 生成 `model_info/final_audit.json` 与 `model_pred/picks.md` 的 242 日 test 表。
- 折级：从各折 `test_predictions.npy` 与 `target` 标签重算逐日截面 Pearson IC / Spearman RankIC，按日等权平均；用于稳定性诊断，不替代模型级四折集成口径。
- 口径与限制：单种子复本；test 窗仅 242 日；四季度不是独立市场样本；不以单季度或单模型最优值宣告结论。

## 产物

- `snapshots/A_260`, `snapshots/B_310`：隔离快照
- `units/<batch>_<model>/`：训练权重、逐折日志、`final_audit.json`、`picks.md`
- `results/`：汇总表、配对比较与本报告
- `logs/`：构建、训练队列与分析日志

## 复现入口

```bash
python scripts/generate_units.py          # 从 release 脚本生成 8 个单元
bash scripts/run_queue.sh                # 顺序训练+分析（断点续跑）
python scripts/evaluate_batches.py       # 汇总 A/B 对比
```
