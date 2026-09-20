# 模块③ · 模型工程 —— 文档入口

> 量化平台三模块之一（① 数据爬取 `datadownload/` → ② 因子工程 `featureengineering/` → ③ **模型**）。
> **输入** = 模块② 的因子产物；**输出** = A 股主板股票的**日频排序打分**（下游据此选股）。
> 本目录就是模块③ 的**全部文档**（原先在模块根的两份 `README.md` / `LOG.md` 已拆分归位，见下面的导航）。
> 平台级主记录在 [`../../QUANT_PLATFORM.md`](../../QUANT_PLATFORM.md)；
> 上游口径以 [`../../featureengineering/README.md`](../../featureengineering/README.md) 为准
> （旧文档里写的 `CLAUDE.md` / `FACTORS.md` **已不存在**，2026-09-19 并进了 `README.md`）。

> **本文件包含**：导航 · 当前状态（榜单作废说明）· §5 命令速查 · §13 附：参考工程与协作约定 ｜ **文档入口**：[`README.md`](README.md)
> ★ 章节号（`§N`）是**全库统一**的：指向别的文件时会写成「文件名.md §N」并带链接；对照表见 [`README.md`](README.md)。

## 导航

| 文件 | 里面是什么 |
|:--|:--|
| **[`SPEC.md`](SPEC.md)** | 结构与规范：目录约定 · 独立单元的定义与边界 · §1 硬约束 · §2 防泄漏铁律 · §4 切分口径（含 `best/` 约定）· §6 训练与内存纪律 · §6.5 损失函数引擎 |
| **[`DATA.md`](DATA.md)** | §3 数据层：`trainingdata/` 四块产物 · 三个口径 · 增量更新 · 换快照的正确姿势 · 上游变更史 · ★ 下游待对齐清单 |
| **[`EVAL.md`](EVAL.md)** | §7 评估三层与策略口径：术语约定（IC = RankIC）· 主判据 Σtop1 · 策略约束 |
| **[`VERSIONS.md`](VERSIONS.md)** | §8 版本履历：V2b~V15 每版的假设 → 配方 → 结论（**历史档案**，命令与特征数已过时） |
| **[`LESSONS.md`](LESSONS.md)** | §9 已确认的事实 · §10 死胡同清单 · §11 坑清单（都带证据） |
| **[`TODO.md`](TODO.md)** | §12 待办与操作纪律 · 引擎副本同步纪律 |
| **[`LOG.md`](LOG.md)** | ★ 迭代日志（倒序）：每一轮改了什么、为什么、结论、锚点 |

---

## 🏆 榜单状态（2026-09-20 起：旧榜作废，新榜待建）

> **当前数据锚点**：`panel_digest=798ccb32214853a8` · 337 特征 × 2018-01-02 ~ 2026-09-18
> （2116 天 × 2115 只 = 4,475,340 行）。★ 引用任何旧数字前，先核对它建在哪个 `panel_digest` 上。
>
> ⚠️ **2026-09-20 起旧榜已作废，暂不维护。**
> 原因是**数据被换掉了**：上游 2026-09-19 之后把因子整体重算过（2012~2017 分区被删、
> 因子从 306 增到 337、财报类因子全体微动），`trainingdata/` 已按上游当前的样子重建。
> 旧榜单（V12~V15）建在「306 特征 × 2012~2026」上，与现在的「337 特征 × 2018~2026」
> **特征集与年份范围都不同，不可比**。★ 那 39 行旧数字**已随 2026-09-20 的文档瘦身删除**
> （它们既不是结论也不是规律）。与数据无关的方法论留在了 [`VERSIONS.md`](VERSIONS.md) §8，
> 来龙去脉见 [`LOG.md`](LOG.md)。
>
> 新榜单要等新版本（从 `V16` 复制）跑出来再建。判定口径不变：**用 Σtop1**，先看折间区间再看均值。
>
> 新数据上的**第一条基线**已跑出（`V16`，1 折，只作链路证据、**不是结论**）：
> `xgb`/20d 评价 RankIC **+0.1233**、Σtop1 **+0.857**；集成把 IC 提到 +0.1379 但 Σtop1 掉到 +0.357
> —— 「一个弱头等权混进来会拖死集成」这个老问题**在新数据上原样重现**。
> 明细见 [`LOG.md`](LOG.md) 的「七、新建的模板单元 V16」。

---

## 5. 命令速查

```bash
PY=/autodl-fs/data/miniconda3/bin/python        # ★ 必须全路径
cd /autodl-fs/data/model                        # 数据准备在这里做

# —— 数据（唯一在模块根做的事）
$PY preparingdata.py                   # 日常增量（没有产物→自动全量）
$PY preparingdata.py --check           # 体检：meta↔文件↔上游 + 取值域抽查
$PY preparingdata.py --amount-only     # 只补 amount（每日成交额）
$PY preparingdata.py --fac-sample-only # 只补 fac_sample（因子抽样）
```

> ★ **日常就这一条**：`$PY preparingdata.py`。有上游新数据就走增量（重算最近 28 个交易日），
> 没有就零 I/O 跳过。快照当前**不是冻结状态**（要上锁用 `--meta-only --freeze`）。
>
> ⚠️ **代码层尚未对齐新块名**：`V16/mx/` 现在读的还是 `X`/`Y`/`universe`/`P`，
> 所以下面这些单元命令**暂时跑不通**。待对齐清单见 [`DATA.md`](DATA.md) §3 末。

**跑一个单元：先 `cd` 进那个单元**（★ 顶层没有 `main.py` 了，也没有模块根引擎）：

```bash
cd /autodl-fs/data/model/V16

# —— 看
$PY main.py doctor           # 环境自检（核数/内存/磁盘/GPU/torch + 该开几折几线程）
$PY main.py status           # 数据在哪 + 训到哪了
$PY main.py split V16        # ★ 零成本推演切分窗口（不训练，只读一列特征）
$PY main.py data V16         # 看面板清单/覆盖率/可用样本数

# —— 跑
$PY main.py freeze V16       # 冻结配方（引擎哈希/上游指纹/生效切分）
bash train.sh                # 全部折（并发；EXIT:0 的折自动跳过 = 断点续跑）
$PY run.py 1 --smoke         # 单折冒烟
$PY run.py 1                 # 单折正式
$PY analysis.py              # 装配 + 榜单 + 策略矩阵 + 回测 + 落盘打分
$PY analysis.py --no-backtest        # 只出榜单（不碰价格层，快）
$PY main.py audit-pit V16    # 前视审计：截断复算
$PY main.py eval V16         # 打印最近一次榜单（不重算）
```

**做一个新版本**（`V16` → `V17`；配方细节以 `../V16/model.py` 的文件头注释为准
—— ★ `V16/` 下**没有 `README.md`**，文档全在本目录）：

```bash
cd /autodl-fs/data/model && cp -r V16 V17 && cd V17
# 改 model.py：NAME / TITLE / DATA / SPLIT / heads() / STRATEGIES
$PY main.py freeze V17 && bash train.sh && $PY analysis.py
```

★ **工具脚本已全部删除**（`tools/` 那一套）。以前 `board.py` / `report.py` 生成榜单与
Top-8 模板，现在文档由 **AI Agent 维护** —— 要出榜单就说一声，不要去找脚本。

---

## 13. 附：参考工程与协作约定

- 参考工程 `CHEN-si-yu/LINGQIDATA`（只读镜像）是"版本化单元 + 策略矩阵 + `prepared_data.py`
  初加工"这套设计的来源。★ **本机当前没有它的克隆**（2026-09-18 换机器后 `/home/claude/ref/`
  与 `~/.ssh/` 都不存在了）；要用得按 `docs` 里的只读 deploy key 重新 clone 到 `/tmp`
  （**别落共享盘**）。它的「死胡同清单」与「已确认事实」两张表**本库已抄录本地化**（[`LESSONS.md`](LESSONS.md) §10 / §9），
  引用前先确认拿到的是参考工程的哪一版。
- 本模块代码**不进公开仓库**（`LQdata_sync.sh` 的 `MODULES` 只含 datadownload/featureengineering/
  everyday_tasks）—— 要不要把模型代码也推上去，需要用户拍板。
- 工作方式：中文交流与注释、注释讲"为什么"、操作都落在共享盘、增量/断点续跑必须稳、
  一个接口一个接口地做、允许并行 agent 干活。
