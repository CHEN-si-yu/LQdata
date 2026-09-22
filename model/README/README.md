# 模块③ · 模型迭代规程（怎么写一个新版本）

> 这一份只讲**规则与流程**：版本怎么编号、从哪个模板长出来、训练与打分的时间范围、交付什么、
> 命令怎么跑、产物落在哪。**不写配方数字与口径细节**（那些会过期、且以代码为准）——
> 配方看 `V16/model.py` 的 `RECIPE`，规格看 [SPEC.md](SPEC.md)，结论看 [LESSONS.md](LESSONS.md)，
> 待办看 [TODO.md](TODO.md)，排行榜看 [`../LEADERBOARD.md`](../LEADERBOARD.md)。

> **本文件包含**：§1 版本迭代规则 · §2 时间口径 · §3 交付格式 · §4 命令流水线 · §5 产物位置 · §6 文档分工

---

## 1. ★ 版本迭代规则（用户 2026-09-20 定）

**这个模块在做什么**：把 `trainingdata/`（初加工快照，**模型侧唯一数据源**）训成打分 ——
按季度滚动、四折集成，再在固定的评价窗上比版本。当前数据锚点 `panel_digest=798ccb32214853a8`
（337 特征 × **2115 只固定池**，★ 固定池有**已知幸存者偏差**，报告不能当"无偏的全市场实盘收益"读）。

| 规则 | 内容 |
|:--|:--|
| **一版一单元、自包含** | `V{N}/` 自带 `model.py`（配方 / 数据 / 切分 / 模型 / 损失 / 训练）+ `analysis.py`（推演拼接 / 评价 / 回测 / 报告 / 推荐表）+ `train.sh`（一键流水线）。**不 import 上游、不 import 别的版本**，只读 `trainingdata/` ⇒ 任何一版都能单独跑、单独交付 |
| **模板 = `V16`** | 新版本从 **`V16` 复制**（`cp -r V16 V17`）再改。★ **不要**从 `history_iterations/` 复制 —— 那些是旧口径，跑不起来 |
| **编号只增不减** | 下一版是 `V17`、再下一版 `V18`…**不在旧版本上就地改** —— 旧版本是"当时的结论"，改了就没法复查 |
| **归档不运行** | `history_iterations/` 里的 **`V1`~`V75`（`V16` 除外，共 74 个）** 只作历史：**别再跑、也别从那儿复制**（旧口径，跑不起来）。★ 2026-09-22 起归档**只删权重**（`model_train/`）—— `model_pred/`、`REPORT.md`、脚本、`logs/` 与 research 子目录**都还在**，可读、可引、可复查；`V16`（模板）与 `best/`（实战）**不在**归档集合内 |
| **`best/` 是晋级位**（当前空） | 把验证通过的那一版脚本复制进去、改成生产切分口径。★ **不根据测试结果自动晋级**，晋级是人的决定 |
| **改了配方 = 新版本** | 配方变 ⇒ `run_id` 变 ⇒ 旧产物一律不复用、不可比；要留住旧结论就**归档**，不能手改锁文件骗过恢复检查 |

★ 一个版本"改了什么"写进**它自己的 `REPORT.md`**（脚本生成）+ 版本目录里的配方锁文件，
不要只留在对话里。

## 2. ★ 时间口径（所有版本统一，不许自行调整）

| 项 | 规则 |
|:--|:--|
| **评价与回测窗** | **2025-07-01 ~ 2026-06-30**（= 2025Q3 / 2025Q4 / 2026Q1 / 2026Q2 四个季度，242 个交易日）。**不许**扩到 2026Q3，也**不许**缩成部分季度 |
| **训练数据** | 只用该季度**开始之前**、且标签已完成隔离（退 `h+1` 个交易日）的数据 |
| **推演（打分产物）** | 「**输入的 X 有多少就输出多少**」—— 一直推到**数据末日**，以支撑增量推演与最新打分；每季度重训四折，历史窗口逐季向前滚动 |
| **某一天由谁给分** | 「**最新一个没见过它的季度模型组**」；季度之后的尾段由最后一个季度那组负责 |
| **排行榜取值** | 只取**评价窗内**的总计值；推演窗比评价窗多出来的那一段**不参与评价** |

★ 为什么这么定、purge/embargo 与四折怎么轮换、`test` 与推演窗（`score`）的关系，见 [SPEC.md](SPEC.md) §4。

## 3. ★ 交付格式（每版必须齐，用户 2026-09-20 定）

1. **`REPORT.md`**（放在版本目录下）—— 以 **Top1 隔日换手为 baseline**：策略收益/回撤曲线、
   四季度统一指标表、四折稳定性表、配方与验证记录、资源采样与效率表；原始 CSV/JSON 可追溯。
   ★ **由 `analysis.py` 生成**，不靠 Agent 手工写。
2. **主评估指标表**：IC / ICIR / top 收益 / top 收益稳定性，**四个季度 + 一行总计 = 5 行**，**全部 1d 口径**
   → `model_pred/tables/main_metrics_1d.csv`（口径见 [SPEC.md](SPEC.md) §7）。
3. **最新推荐表**：`model_pred/latest/picks.md` + `recent_backtest.csv`（给实战 / 增量推演）。
4. **更新 [`../LEADERBOARD.md`](../LEADERBOARD.md)**：一行 = 一版，取评价窗总计值，
   排序 **IC → top 收益(1d) → ICIR / 稳定性**。

## 4. 命令流水线

```bash
PY=/autodl-fs/data/miniconda3/bin/python      # ★ 必须全路径（非登录 shell 里的 python 没有 pandas）
cd /autodl-fs/data/model
$PY preparingdata.py --check          # 初加工产物 ↔ 上游 ↔ meta 对账（★ 写 trainingdata/ 的唯一入口）
cd V16
$PY model.py --doctor                 # 环境 / 数据自检
$PY model.py --split                  # 打印四个季度 × 四折的 train / valid / test / 推演 区间
$PY model.py --rescore                # 只重推演、不重训：口径改了但配方没变时刷新打分
$PY model.py --quarter 2025Q3 --fold 1  # 单季度单折（调试用）
$PY analysis.py --pipeline --jobs 8   # 全部"季度×折"调度 + 资源采样 + 评价
$PY analysis.py --audit               # 完整评价 + 回测 + 重载推理对拍 + 报告 + 推荐表
$PY analysis.py --picks-only          # 只出最新一日推荐表（增量推演用，不跑评价）
$PY analysis.py --no-backtest         # 明确 IC-only，不出任何收益结论
bash train.sh                         # 一条命令跑完：默认上限 8 任务 + 资源记录 + 回测 + REPORT.md
```

数据路径按 `MX_DATA` > `V16/trainingdata` > `../trainingdata` 依次找，以 `meta.json` 是否存在判断。

**并发与内存纪律**（数字全部现读，换机器/换实例自动跟随）：

- 内存上限一律**现读 `/sys/fs/cgroup/memory.max`**，脚本里**不写死主机数字**；读不到或为 `max` 时
  按"本机无上限"处理 ⇒ 跳过高水位闸门，只按 `--jobs` 限流并明确提示。
- 最多 **8 个"季度×折"任务**（不是 8 折）；单任务 CPU 亲和按**本机可用核数的 80% 在并发任务间均分**（上限 3 核）。
- **两个重活绝不并行**（训练 × 分析会顶穿容器，实测爆过一次容器重启）；训练与最终分析串行。
- 跨折并发只有 `analysis.py --pipeline` 一套调度器；`model.py` 只做单折（`--quarter` / `--fold`）。

## 5. 产物位置

- `REPORT.md`：统一中文报告（`analysis.py` 生成）。
- **`model_pred/`**（csv/json 已收进子目录）：

  | 子目录 | 装什么 |
  |:--|:--|
  | **`tables/`** | 汇总表与报告 json：`report.json`、`main_metrics_1d.csv`、`quarterly_*.csv`、`strategy_summary.csv` |
  | **`ensemble/`** | 集成打分明细：`year=YYYY/data.parquet`（`trade_date/stock_code/value/rank`，**覆盖全推演窗**）＋ 各策略的每日收益 / 选股 / 资金曲线 / 逐笔成交 CSV |
  | **`latest/`** | 最新一日推荐表 `picks.md` / `picks.csv`（排名/代码/名称/打分）＋ `recent_backtest.csv`（近 10 个信号日的 top1 明细，含**累计**列，未定型的行留空） |
  | **`charts/`** | 策略收益/回撤曲线、可执行收益累加曲线、资源曲线 |

  ★ **只有集成落盘** —— 各折的指标仍会算（供四折稳定性表），但产物不写盘。
  用户 2026-09-20：「**4Fold 集成才是这个模型**」。

- **`logs/`**：`<季度>_fold<k>.log`（一个任务一个文件，**不要合并**）· `analysis.log` ·
  `resource_usage*.csv` / `resource_phases.json` / `resource_summary.json`（REPORT 资源表的数据源）·
  以及旧日志归档。
- **`model_train/<季度>/fold<k>/`**：`best.pt` / `last.pt` / `history.json` / `split.json` /
  `recipe.lock.json` / `complete.json` / `score_predictions.npy`（全推演窗）/ `test_predictions.npy`（季度切片）。
  ★ **产物的指纹与恢复规则见 [SPEC.md](SPEC.md) §4**（改配方/重建数据后必须归档，不能骗过恢复检查）。

## 6. 文档分工

| 文件 | 装什么 | 什么时候读 |
|:--|:--|:--|
| [`../LEADERBOARD.md`](../LEADERBOARD.md) | **排行榜**：一行一版、4 个指标（★ 历史 **1d** 口径，只加标注不重算） | 想知道旧口径下"哪版最好" |
| [`../LEADERBOARD_RD11.md`](../LEADERBOARD_RD11.md) | **RD11 榜**：五日换手 ＋ 止盈止损（**5d** 口径，按三粒种子最差净收益排） | 想知道 RD11 方向"哪套最强" |
| [`../LEADERBOARD_RD12.md`](../LEADERBOARD_RD12.md) | **RD12 榜**：每日打榜（涨停板买入）。**口径已预登记、执行路径未实现** | 要开工打板方向 |
| **本文件** | **规程**：编号 / 时间口径 / 交付 / 命令 / 产物位置 | 要开一个新版本 |
| [SPEC.md](SPEC.md) | **规格**：目录 · 硬约束 · 防泄漏铁律 · 数据层 · 切分与训练 · 评价口径 | 要改口径、查"什么不许动" |
| [LESSONS.md](LESSONS.md) | **结论**：核心规律 `R*`/`U*` · 已确认事实 · 死胡同 · 坑 | 想避免重踩 |
| [TODO.md](TODO.md) | **待办**：操作纪律 · 队列 · 研究方向 `RD*` | 要挑下一件事 |

★ **章节号 `§N` 全库统一**：`§8.5`、`§11-24` 这类引用在任何一份文件里都按同一张表解析；
`§5 / §6 / §6.5` 已随旧 `mx/` 引擎副本作废，**编号不回收**。
