# LQdata · A 股量化平台（数据 → 因子 → 模型）

四段式量化流水线的代码：**数据爬取 → 因子工程 → 日更增量 → 模型训练**。

> 本仓库是**代码快照**——不含任何数据、状态文件与日志（`data/` `state/` `logs/` 均在
> `.gitignore` 中），也不含数据接口密钥（`datadownload/APIKey.txt` 需在本地单独放置）、
> 厂商接口文档与第三方研究资料。
>
> 平台级主记录（跨工程的工作约定、资源与数据纪律）**不对外**，
> 各工程自己的文档是自洽的，直接读下面「文档」一节即可。

## 目录

| 目录 | 模块 | 说明 |
|:--|:--|:--|
| `datadownload/` | ① 数据爬取 | 数据商接口的全量回填器：40+ 个数据集，按年分区落 parquet，带限速与断点续传（回填已完成后冻结） |
| `featureengineering/` | ② 因子工程 | 日频因子框架：声明式注册（`@register`）、PIT 前视守卫、增量/全量逐格一致、逐截面 MD5 台账 |
| `everyday_tasks/` | ①′ 日更增量 | 每日增量流水线：推进日历 → 闸门等上游到齐 → 增量落库 → 单日指纹与报告 |
| `model/` | ③ 模型 | 初加工快照（`preparingdata.py`）+ **版本化单元**（`V16/` 模板）：训练 / 评价 / 回测 |

四个工程**只用共享盘上的 parquet 解耦**：每个工程只读上游产物、只写自己的产物，
工程之间不互相 import —— 所以任何一环都能单独换掉、单独重跑。

## 目标与硬约束

下游任务是 **A 股日横断面回归排序**，因此：

1. **全部因子日频**，对齐 `(trade_date, stock_code)` 面板后落盘；
2. **只做主板**（沪 `600/601/603/605` + 深 `000/001/002/003`），排除创业板 / 科创板 / 北交所；
3. **严格 PIT（point-in-time）**：历史因子值不得因未来的分红事件或数据晚到而变化 ——
   禁用前复权价（前复权会按最新因子重算全部历史），晚到的数据集必须显式声明并整体位移；
4. **产物格式统一**：`trade_date / stock_code / value / rank` 四列，年分区 parquet，
   整块面板落盘（`NaN` 也写，覆盖率可审计）。

## 环境与运行

- Python 3.11+；依赖：`pandas` `numpy` `pyarrow` `requests` `tqdm` `PyYAML`（模型侧另需 `xgboost` 等）
- 每个工程的入口都是一层薄薄的 `main.py`，逻辑在各自的包内：

```bash
python datadownload/main.py --help             # 数据回填（run / list / status / doctor）
python featureengineering/main.py run          # 因子增量（--rebuild 全量 · check 体检）
python everyday_tasks/main.py                  # 日更流水线（裸命令 = 闸门 + 全量增量）
python model/preparingdata.py                  # 模型侧初加工快照（增量，幂等）
cd model/V16 && python main.py --help          # 单元：训练 / 评价 / 回测
```

## 文档

| 工程 | 文档 |
|:--|:--|
| `datadownload/` | [`README.md`](datadownload/README.md) —— 目录结构 + **接口实测约束**（限速、分页上限、`daily_dump` 配额…） |
| `featureengineering/` | [`README.md`](featureengineering/README.md) —— 总手册（现行口径 + 因子字典 + 质量体检）· [`HISTORY.md`](featureengineering/HISTORY.md) —— 维护记录与旧文档归档 |
| `everyday_tasks/` | [`README.md`](everyday_tasks/README.md) —— 该工程的唯一文档（含每日怎么跑、闸门口径、告警定性） |
| `model/` | [`README.md`](model/README.md) → [`README/`](model/README/README.md) —— 规范 / 数据层 / 评价口径 / 版本史 / 经验 / 待办 / 日志 共 8 份 |

厂商提供的接口文档、第三方研究资料与平台级主记录**不在本仓库内**。
