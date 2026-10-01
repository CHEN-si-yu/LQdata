# 模型项目

本项目用于研究股票打分模型及其交易策略。数据快照、独立实验单元、评价口径和迭代记录各有固定职责；单元编号只作标识，研究分支和策略以配置为准。

## 项目结构

```text
model/
├── preparingdata.py       生成并维护数据快照
├── trainingdata/          单元共享的只读数据与元信息
├── experiments/Vx/        实验层一：全市场截面排序单元
├── experiments2/Vx/       实验层二：极少数股票的仓位动作单元
├── docs/                  项目规则、评价规范与逐版记录
├── releases/              经明确确认的实战单元
└── history/               历史研究材料，不参与单元运行
```

两层是**并列的实验层**（不是新旧关系），任务不同、互不调用，只共享 `trainingdata/`：

- `experiments/`：预测全市场约 2100 只票的**排序**，按打分买 TopK；评价用截面 IC / RankIC 与 TopK 组合，基准 `top1_1d`。
- `experiments2/`：只盯**极少数股票**（股票池由单元配置指定，如四大银行），判断每只票该**加仓还是清仓**；评价用该股自己的**时序 IC** 与**按票独立的仓位账户**，基准为同窗买入持有。

两层的单元都固定由 `run.py`、`model.py`、`analysis.py` 三个实现脚本组成，并遵守[固定目录和文件规则](docs/architecture.md#独立单元开发强制规则)；文件清单以各单元自己的 `model.py:fixed_files()` 为准，两层清单**同名不同物**（`picks.md` 对 `actions.md`）。单元之间只共享 `trainingdata/`，不调用其他单元的代码，**也不跨层调用**。

策略由单元配置明确指定：`experiments/` 用每日或每五日调仓的 `top5_1d` / `top5_5d` 及基线 `top1_1d`；`experiments2/` 用三档目标仓位的 `action_3state` 与基准 `buy_hold`。不能根据单元编号推断策略。

## 新单元与权重

建立单元时，先明确其研究问题和继承来源。改动训练配方时，以选定父单元的三个脚本为起点并重新训练；只改策略或报表时，复制完整单元以复用权重。每版记录改动、结果、结论和权重来源：`experiments/` 写进 [docs/iterations.md](docs/iterations.md)，`experiments2/` 写进 [docs/iterations2.md](docs/iterations2.md)（两层的单元编号各自独立，引用时写明全路径）。

## 运行入口

在单元目录中执行：

```bash
./run.py
python -B analysis.py --audit
```

`run.py`负责训练调度，`analysis.py`负责评价与产物。调度器根据运行环境探测可用资源；需要时可通过 `--jobs` 和 `--device` 覆盖自动选择。资源配置随环境而定，项目文档不固定某台机器的并发数或内存预算（`experiments2/` 只读池内那几只票，样本量小两个数量级，默认顺序执行即可）。

评价指标、报告口径与判定原则见 [docs/baseline.md](docs/baseline.md)（同一份文件按层分节）；目录职责、数据合同和文件清单见 [docs/architecture.md](docs/architecture.md)。标准评价窗按项目配置固定为 2025Q3 至 2026Q2，**两层相同**，因此两层的曲线可以放在同一根时间轴上对照。
