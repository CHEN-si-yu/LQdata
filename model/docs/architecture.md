# 项目架构与独立单元规则

`preparingdata.py`维护共享数据快照；实验单元在 `experiments/Vx/`（全市场截面排序），各自完成训练、推理和评价；`docs/`保存项目规则与实验记录；`history/`保存研究材料，不参与单元运行；`releases/`用于存放经确认的实战单元。

实验单元**互不调用**、只共享 `trainingdata/`。`releases/` 不是实验层 —— 那是经用户明确确认后的实战单元。

## 项目结构与职责

| 路径 | 职责 |
| --- | --- |
| `preparingdata.py` | 对接上游数据并构建快照 |
| `trainingdata/` | 所有单元共享的只读输入 |
| `experiments/Vx/` | 相互独立的训练和评价单元（**全市场截面排序**：按打分买 TopK） |
| `docs/` | 架构、评价规范、逐版实验记录 |
| `releases/` | 经明确确认的实战单元 |
| `history/` | 历史算法与研究材料，不作为运行依赖 |

策略、父单元和权重来源由单元配置与实验记录明确说明。单元编号只用于唯一标识；不得用编号奇偶或编号先后推断策略类型、权重关系或交付状态。

## 单元继承与训练

每个单元只研究一个明确配置的交易策略。单元使用 `top5_1d`（每日调仓）、`top5_5d`（每五个交易日调仓）以及 `top1_1d` 基线。实际单元以自身配置和结果元数据为准。

创建新单元时必须明确父单元，并按改动范围选择起点：

| 改动范围 | 单元起点 | 权重处理 |
| --- | --- | --- |
| 训练配方、特征、标签、切分或选轮规则 | 复制父单元的 `run.py`、`model.py`、`analysis.py` | 不复用旧折产物；按新配方训练 |
| 仅策略、打分重算或报表 | 复制父单元的完整单元目录 | 明确复用的权重来源；按需重算评价产物 |

`RECIPE` 是训练配方的配置真源。折的完成判据读取完成状态及其必需产物，不自动证明配置指纹一致；改变训练配方后，必须确保对应旧训练产物不会被复用。仅改变打分变换且无需重新训练时，可使用单元实现提供的重打分流程。

逐版记录保持精简，写明改动、可核对的结果、结论及权重来源。具体指标和证据的写法见 [baseline.md](baseline.md)。

## 运行资源与输入边界

训练调度器在启动时探测可用内存和可见 GPU，并据此选择并发与设备；`--jobs`、`--device` 可用于显式覆盖。资源可用量随运行环境变化，不在项目规则中规定固定机器型号、显存、单折内存或并发数。

### 模型优先的训练与回测顺序（2026-10-06 用户要求）

**尽量以一个模型的16折优先，依次完成模型，而不是先将所有模型的同一折铺开。** 标准单复本为四个季度×每季度四折；在资源和内存预算内，优先并行该模型的未完成折。该模型16折齐全后，立即执行权重重载审计、统一指标回测和记录，再根据完整结果调整后续研究方向。

优先模型的未完成折均已启动时，可用后续模型填补空闲资源；并发不改变模型优先的队列顺序。回测可与下一个模型训练并行，重分析任务需控制内存和并发。当前会话的资源要求为平均利用率超过75%、实际内存严格低于150GB（服务器上限160GB）；分别记录CPU与GPU平均利用率，按实测报告是否达标，不能用瞬时利用率替代平均值。内存保护优先于追加并发。

调度顺序调整保留已完成折与已有epoch检查点，不改变运行中的训练配方、数据快照或评价窗；必要时受控重启调度器，从检查点续训。不得将部分季度或未齐全的折当作完整模型成绩。补种子时每个复本也须完成自己的16折，模型级成绩按全部复本打分等权均值计算，仍遵守[baseline.md](baseline.md)的触发门槛。

### 持续研究与方向调整注意事项（2026-10-06 用户要求）

模型侧在`experiments/`持续迭代，以**提高稳定的收益**为主要研究目标，直到用户明确要求停止。每次训练与回测记录配方、权重来源、统一指标、资源使用、结论与局限；发现错误时保留原结论及更正依据，依据实际结果调整下一步方向。不能只保留表现好的实验或重复声称未经验证的收益改善。

重点检验用户提出的新假设：**在大资金体系下训练出能捕捉正确信号的模型，可能在个人小资金交易中取得更好的收益。** 这是待检验的迁移假设，不是已经确认的普遍结论；“资金规模”与“股票市值风格”分开说明。采用同快照、同架构、配对训练目标的对照，在统一信号指标和相同小资金含费执行规则下比较，并结合季度、回撤、费用与种子稳定性评估，不能把无摩擦资金分配结果直接当作个人账户收益。

允许通过不同模型/策略形成不同风格，再检验互补能否提高稳定收益；同时检查单腿、固定组合、弱势季度、风险暴露和执行成本。融合权重与候选选择遵循[baseline.md](baseline.md)的验证/测试边界与复本规则，不按测试收益最大值给出最终模型成绩。研究方向探索和调整的具体证据登记到当日研究文档及`iterations.md`。

实验单元只读取 `trainingdata/` 和获准的展示辅助数据。股票名称表和交易日历仅用于报告展示，不进入模型指标计算；价格从 `trainingdata/prices/` 读取。单元不得依赖其他实验目录、临时脚本或未列明的外部数据路径 —— **包括不得 import `history/` 的代码**：共享能力靠**复制**而不是依赖（与"单元之间只共享 `trainingdata/`"同一条规矩）。

## 训练快照 `trainingdata/`

`preparingdata.py` 是快照的唯一写入入口，实验单元只读。快照负责对齐和增量保存上游数据，不在此层复权、填充、计算收益或修改标签。下游策略和模型需要的派生口径由单元实现。

```text
trainingdata/
├── meta.json                              数据清单、边界、覆盖率和值口径
├── factors/year=YYYY/data.parquet         trade_date, stock_code, 股票因子列
├── target/year=YYYY/data.parquet          trade_date, stock_code, 未来收益标签
├── amount/year=YYYY/data.parquet          trade_date, stock_code, amount
├── fac_sample/year=YYYY/data.parquet      抽样因子列
├── market_factors/year=YYYY/data.parquet  按交易日记录的市场因子
├── prices/year=YYYY/data.parquet          原始行情与复权因子，供回测读取
└── log/ledger_YYYY-MM-DD.tsv              快照增量台账
```

股票特征、市场特征和标签的列数、主键及有效范围以 `meta.json` 与当前数据快照为准。股票因子的解释口径以 `meta.json.semantics` 为准；本快照语义标识为 `zscore_win1_99_v1`：按日横截面缩尾、标准化、截断并按方向元数据翻转，缺失值按中性值 `0` 处理。消费方不得套用旧的 `[0,1]` 排名值或 `0.5` 缺失填充值。

市场因子按 `trade_date` 与股票因子对齐；带 `_z252` 的列使用截至当日的滚动标准化值。价格块保存原始行情，不在快照层复权或修补停牌；复权、涨跌停、停牌和成交规则由消费该价格的回测实现负责。

**特征集来源（2026-10-07 用户要求）**：`preparingdata.py` 直接扫描
`featureengineering/data/factors/<name>/year=YYYY/data.parquet` 与并列的
`market_factors/`，纳入全部已经落盘的因子，不读取上游 Python、配置、注册表、发布清单或状态文件。
股票目录中 `label_ret_<h>d` 为标签，其余为特征；日期与股票轴来自股票产物主键并集。
空目录不计入清单，缺格保留 NaN。列集、方向或股票轴变化自动重建所有年份，避免分区混用。

Parquet 不包含优劣方向。默认保留本地快照 `columns.direction`，新因子保持原始方向；
`--direction-policy raw` 可让全部因子保持原始方向。`columns.raw_direction_features` 明确记录保持原始方向的列，
不能据此声称这些列“越大越好”。`--features-from` 仍可固定特征清单及其已有方向。
旧 `--extra-features` 兼容为落盘名字校验，配固定清单时追加列；默认打包已无需这个参数。

```bash
python preparingdata.py --jobs 12 --progress bars
python preparingdata.py --source-data /root/autodl-fs/featureengineering/data --factors-only --out trial
```

`--jobs` 控制扫描线程与因子工作进程。读取、对齐、标准化在工作进程内完成，默认并发最多 12，
并尊重容器 CPU 配额；每个进程默认 1 个计算线程，可用 `MX_THREADS` 调整。
tqdm 展示扫描、年度、股票特征、标签、市场因子和附加块进度；`--progress auto` 仅在终端显示，
`bars` 强制显示，`off` 关闭。本地 `.source_index.json` 按源文件大小及修改时间缓存主键轴，
源文件变化会重新扫描；`--check` 只读。`--factors-only` 可仅依赖因子 data 目录，跳过原始成交额和价格表。

## 独立单元开发强制规则

1. 每个单元固定包含 `run.py`、`model.py`、`analysis.py` 三个实现脚本。训练调度放在 `run.py`，模型、训练、推理和路径合同放在 `model.py`，指标、回测及报告放在 `analysis.py`。
2. 不新增、删除、移动、重命名、拆分或合并单元内的文件与目录。配置与实现直接维护在固定脚本内；临时分析工具放在单元目录之外。
3. 单元只在固定产物路径原位写入或覆盖。不得新增独立报告、备份、日期目录、运行编号目录或未列入清单的结果文件。
4. 季度、折数、年度分区与策略文件名属于固定文件合同。调整结构必须先得到用户明确要求，再同步更新 `model.py:fixed_files()` 和本文件中的结构说明。
5. `validate_layout()` 核对固定清单。固定清单外的路径必须拒绝；自动生成的 `__pycache__/` 和 `.ipynb_checkpoints/` 不计入单元文件。

## 固定文件树

各单元使用相同的文件类型、目录展开规则和清单定义；策略名称由单元配置决定，数据季度、折数和年度分区按项目合同展开。产物保存逐日 TopK 结果。清单的唯一真源始终是该单元自己的 `model.py:fixed_files()`。

```text
Vx/
├── run.py
├── model.py
├── analysis.py
├── model_train/
│   └── <季度>/fold<折>/
│       ├── best.pt
│       ├── last.pt
│       ├── complete.json
│       ├── history.json
│       ├── score_predictions.npy
│       ├── test_predictions.npy
│       ├── split.json
│       └── training_info.json
├── model_pred/
│   ├── ensemble_curves.png
│   ├── picks.md
│   └── ensemble/
│       ├── score_meta.json
│       ├── year=<年份>/data.parquet
│       └── CSV/
│           ├── cash_top5_<1d或5d>.csv
│           ├── trades_top5_<1d或5d>.csv
│           ├── cash_top1_1d.csv
│           └── trades_top1_1d.csv
├── model_logs/
│   ├── <季度>_<折>.log
│   └── analysis_0.log
└── model_info/
    ├── final_audit.json
    └── two_strategy_validation.json
```

固定清单为 159 个文件、28 个子目录；实际文件名和检查逻辑以该单元 `model.py:fixed_files()` 为准。标准评价分区为 `2025Q3`、`2025Q4`、`2026Q1`、`2026Q2`，每季 4 折，预测分区按 2025、2026 年份展开。

## 固定产物含义

- `model_pred/picks.md`：开头依次放 test 评价窗和 valid 训练验证集两张表，表间不插入说明；之后放最近 10 个交易日的逐日 Top5 排名。
- `model_pred/ensemble/year=<年份>/data.parquet`：保存集成打分，供其他研究按自身规则复算策略。
- `model_pred/ensemble_curves.png`：比较单元配置的交易策略与 `top1_1d` 基线的含费曲线。
- `model_info/final_audit.json`：汇总指标、评价窗、打分来源和审计记录；逐版数字从该单元结果读取。
- `model_info/two_strategy_validation.json`：记录两类策略的账户实现值与口径自检。
- `model_logs/`：逐折训练日志和评价日志。

单元不生成 `docs/` 下的 Markdown 报告。每版研究说明由维护者写入 [iterations.md](iterations.md)；数据指标定义和判定方式见 [baseline.md](baseline.md)。

## 运行命令

在单元目录中运行：

```bash
./run.py
./run.py --jobs N --device cuda
python -B analysis.py --audit
python -B analysis.py --from-scores
python -B analysis.py --from-scores --picks-only
```

训练和评价是两个独立步骤。`--audit` 会执行实现支持的权重重载与推理对拍；`--from-scores` 只重算已有集成打分的评价产物。参数含义及当前实现以单元脚本的帮助和源码为准。

## `releases/`：实战单元（2026-09-27 建）

`releases/` **不是第三层实验层**，而是经用户明确确认的实战单元（见 §项目结构与职责）。
当前三个：`V31`、`V36`、`V38`。每个单元的三个实现脚本（`run.py` / `model.py` / `analysis.py`）
从对应的 `experiments/<V>/` **逐字复制**过来 —— 「原封不动」这条有可复做的判据，见
[releases/README.md](../releases/README.md)。

### 实战模型的定义

**一个实战模型 = 一个季度的四折集成**，与 experiments 的交付口径同构
（那边每个季度的交付打分就是四折打分相加）。**不是单个子模型**，也不是滚动多季度训练。

### 伪季度与切分

实战模式用**一个伪季度** `LIVE_QUARTER = '2026Q3'`，`splits()` 的几何与实验模式完全相同：

| 段 | 说明 |
| --- | --- |
| `train` | 第 k 折把第 k 块留作验证，训练集是其余部分（与实验模式同式） |
| `valid` | 第 k 块（488 天左右）；只用于早停 / bagging / BN 重标定 |
| `test` = `score` | 2026-07-01 → 数据末日；**四个折的训练集都不含这段 ⇒ 干净样本外** |

四折共用的 cutoff = 季首 − `purge_horizon` − 1 ≈ 2026-06-01。

★ 与实验模式**唯一**的代码差异：伪季度不在固定评价窗内，`splits()` 里的 `test`
取本季度自己的区间（与 `tmp/vlines/wf_unit` 的既有做法一致）。

### 实战模式开关与文件合同

- 开关 = 环境变量 `MX_LIVE=1`。**关掉时（默认）行为与 `experiments/<V>/` 逐字一致**，
  可用 `--split` 逐行对拍验证。
- 固定清单：`model.py:experiment_files()` = 实验模式清单（原 `fixed_files()` 的逐字副本）；
  `model.py:fixed_files()` 按当前模式返回；`model.py:known_files()` = **两者并集**。
  `validate_layout()` 判「多余」用并集 —— 因为没有它时，一个已训过实战模型的单元
  **无法再跑实验模式**（连 `--split` 都会被自己的历史产物挡住），「原封不动」这条判据就废了。
  并集仍是两个**枚举**清单，清单外的路径一样被拒。
- 实战清单（`LIVE_FILES`）：`model_train/2026Q3/fold{1..4}/` 的 8 个固定文件 +
  `model_logs/2026Q3_{1..4}.log` + `model_info/live_validation.json` + `model_pred/live_report.md`。

### 运行命令

```bash
cd releases/V38                # 或 V31 / V36
MX_LIVE=1 ./run.py --jobs 4 --device cuda     # 训练四折（断点续跑）
MX_LIVE=1 python -B analysis.py --live        # 四折集成在干净评估窗上的验证报告
MX_LIVE=1 python -B model.py --split          # 只看切分
```

★ **并发上限**：每折峰值 ~12 GiB，3 个单元 × 4 折 = 144 GiB，会越过 180 GiB 的 90% 上限。
**并发 6 是安全的**（≈92 GiB）。

### 验证口径

`analysis.py --live` 分两条报，**不能合并**：① 模型质量（评估窗 `RankIC_5d`，对窗口稳健，
是"与 experiments 同档"的主判据）；② 账户表现（照实报，但**不能**直接和多年年化的锚比 ——
评估窗只有 62 天，必须同时看同窗的 experiments 参照）。
细节、技术项与踩过的坑见 [releases/README.md](../releases/README.md)。
