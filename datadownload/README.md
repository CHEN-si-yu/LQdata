# datadownload · 模块① 全量数据爬取（已冻结）

> **一句话**：本工程是量化平台的**历史全量回填器** —— 把灵启数据 API 的数据集从
> **2010-01-01 起一次性拉全**，落地为 parquet。
> **原有批量回填工作已经完成，旧批量入口继续冻结。2026-09-19 按用户要求新增筹码收益专用全量脚本，见下方。**
> 每日增量更新由**独立工程** [`../everyday_tasks`](../everyday_tasks/README.md) 负责。

---

## 新增 32 号接口：筹码收益全量下载（2026-09-19）

`POST /stock/cyq_perf` → `data/stock_cyq_perf/year=YYYY/data.parquet`。
供应商当前可供起点为 2018-01-02，按全市场下载，不限定主板。
主键为 `trade_date + stock_code`，11 列原样保留，特别是 `winner_rate` 不自动缩放。

```bash
runuser -u claude -- /autodl-fs/data/miniconda3/bin/python \
  /autodl-fs/data/datadownload/scripts/download_cyq_perf.py --end 2026-09-18
```

脚本使用 15 个日历日的小区间分页，检查点在 `state/cyq_perf_full/`，重跑可续传；`--refresh` 强制重取所选历史区间。只有年度唯一主键行数等于供应商年度 total 且数据写成功，才推进共享 manifest。使用同一份账户限速台账和跨项目锁，默认 16 GiB 地址空间上限，不使用 GPU。脚本复用独立日更工程的客户端、存储和 manifest，未重新启用旧批量调度器。

增量维护已接入 `../everyday_tasks/main.py`，随原有每日任务一起运行；接口注册表追加为 **32 号**，原 1–31 号不变。接口声明已同步到本工程 `lingqi/spec.py` 和 `conf/frequency.yaml`。

## 1. 现状（接手先读这一节）

| 项 | 值 |
|:--|:--|
| **代码状态** | **已冻结** —— 不再更新、不再运行。旧 `main.py run` 与历史抓取脚本不要再跑；新增 `scripts/download_cyq_perf.py` 是授权的专用入口 |
| **数据** | `data/` 约 **15 GB**，31 个已落地数据集、合计约 **14.5 亿行**，主体 **2010-01-01 起**（个别表更早：`stock_financial_indicator` 1990 起、`stock_list` 1990 起） |
| **⚠️ `data/` 与 `state/` 仍在被写入** | 增量工程 everyday_tasks 每天**直接写这两个目录**（见其 `conf/daily.yaml` 的 `shared_data` / `shared_state`）。这里的 `state/<数据集>.json` 是**它的**断点续传依据 —— **别删、别改、别手工编辑** |
| **数据水位 / 最新交易日** | 每天都在变，**不要在文档里写死**。现查：`$PY main.py list`（各表进度）或 `$PY main.py status`（已落地统计） |
| **权威口径** | 频率 / delay / 闸门以 **`../everyday_tasks/data_incremental/registry.py`** 为准。本目录 `conf/frequency.yaml` 是**冻结快照**（曾是因子侧读取的口径，两侧手工同步过），冻结后不再保证与日更侧一致 |

```bash
PY=/autodl-fs/data/miniconda3/bin/python      # ★ 必须全路径：非登录 shell 的 python 没有 pandas
```

---

## 2. 数据长什么样 / 下游怎么读

```
data/<数据集>/year=YYYY/data.parquet     # 按年分区（绝大多数表）
data/<数据集>/data.parquet               # 单文件（快照类表）
```

```python
import pandas as pd, glob
df = pd.concat([pd.read_parquet(f) for f in
                sorted(glob.glob('/autodl-fs/data/datadownload/data/stock_daily/year=*/*.parquet'))])

# 或统一走框架里的读函数（列类型与本工程一致）
from pathlib import Path
from lingqi.store import read_dataset
df = read_dataset(Path('/autodl-fs/data/datadownload/data'), 'stock_daily')
```

- **复权口径**：`stock_daily` 是**不复权**原始价。要**后复权(hfq)**请用 `stock_adj_factor`
  自己算（与原始价相乘即得）；**前复权 qfq 被硬约束明令禁止**，对应数据集 `stock_daily_adj` 从未下载。
- **周/月频**：没有现成表，**从 `stock_daily` 用交易日历切周期现算**（不要找 `stock_kline`，已删，见 §5）。
- **逐字段数据字典**（每个数据集的字段清单、类型、非空率、取值范围、主键唯一性、按年行数）
  曾以 `DATA_CATALOG.md` 的形式存在，现已归档（§6），可随时再生：

```bash
$PY scripts/catalog.py                 # 全量扫描 → 重新生成 DATA_CATALOG.md（大表较慢，可加 --quick）
$PY scripts/catalog.py --dataset stock_daily
```

---

## 3. 目录导航

| 路径 | 是什么 |
|:--|:--|
| `data/` | **产物**：落地数据（约 15 GB，按年分区）。★ 仍被日更工程写入 |
| `state/` | 各数据集 manifest / 进度 / 限速状态。★ **归日更工程维护**，`*.bak` 是历史改动的备份 |
| `main.py` | 冻结的下载器 CLI：`run`（抓取，**别再跑**）/ `list` / `status` / `doctor` |
| `lingqi/` | 爬虫框架：`client.py`（限速客户端，全局 280 请求/分钟）/ `spec.py`（数据集定义）/ `engine.py` / `store.py` / `gate.py` / `manifest.py` |
| `conf/frequency.yaml` | 频率 / delay / gap_checkable 口径（**冻结快照**，见 §1） |
| `conf/config.yaml` | 接口地址、限速参数等 |
| `scripts/` | 工具脚本，详见 §4 |
| `probe/` | 早期接口形态探测的残留（`coverage_by_year.json`） |
| `logs/` | 历史运行日志 |
| `APIKey.txt` | 接口密钥（日更工程也读这一份：`daily.yaml: api_key_file`） |
| `灵启数据API有权限接口文档 (1).md` | **厂商给的接口文档原文**（59 个接口，厂商生成于 2026-09-13）。本目录**唯一的另一份 md**，见 §6 |

**绝不用 `pkill -f`**（会杀掉执行它的 shell 自己，实测 exit 144）；用
`ps -eo pid,args | grep '[m]ain\.py'` 找 PID 再 `kill`。

---

## 4. 还能安全跑的命令

| 命令 | 说明 |
|:--|:--|
| `$PY main.py list` | 各数据集本地进度（**只读，随时可跑**） |
| `$PY main.py status` | 已落地数据的统计（同上） |
| `$PY scripts/catalog.py` | 生成数据字典（只读本地 parquet；较慢） |
| `$PY scripts/audit.py` | 全库完整性审计：分区对账 / 假覆盖 / 悬空 suspect（**纯本地、零 API**） |
| `$PY scripts/overview.py` | 快速总览（比 catalog 快得多） |
| `$PY scripts/verify.py --all` | 抽样校验 · ⚠️ **会发 API 请求**，别在日更跑批时跑（会抢全局限速额度） |

> ⚠️ **不要再跑 `main.py run`**：抓取职责已交给 everyday_tasks，两边同时跑会
> 争抢**全局**的 280 请求/分钟限速额度，并同时写同一份 `data/` 与 `state/`。

---

## 5. 已删除 / 从未下载的数据集（别去找）

| 数据集 | 状态 | 原因 |
|:--|:--|:--|
| `stock_kline` | **2026-09-16 彻底删除** | 接口返回的周/月线实为「抓取窗口 ∩ 该周期」的聚合，**非日历周期** —— 窗口一滑动，已结束周期的值就被改写；且 14 个列与 `stock_daily` 完全相同，零额外信息 |
| `index_weight` | **2026-09-17 彻底删除** | 厂商月频延迟 ≥1 个月，服务端自己停在旧月份 |
| `ths_hot` · `stock_dc_block_fund_flow` · `stock_ths_block_fund_flow` | **2026-09-19 彻底删除** | 时间覆盖不足 |
| `index_ths_sector_categories` · `index_ths_constituent_stocks` · `tdx_block_stocks` | **2026-09-19 彻底删除** | 快照表、无历史版本，回测必然前视 |
| `index_history`（指数 5min） · `stock_daily_adj`（前复权 qfq） | **从未下载** | 因子要求日频 / qfq 被禁用 |

> 旧代码或旧文档里若还出现这些名字，那是**过期引用**。

---

## 6. 历史文档去哪了

2026-09-19 整理：本工程顶层**只保留两份 md —— 本 README + 厂商接口文档**（后者见下表 ★）。
其余 5 份过程性 md 已打包归档到项目外：

```
/autodl-fs/data/_archive/datadownload_docs_20260919.tar.gz      # 100 KB
/autodl-fs/data/_archive/datadownload_docs_20260919.sha256      # 六份文件的校验和
```

| 归档文件 | 内容 |
|:--|:--|
| `TASKS.md` | 全量下载期间的详细任务清单、6 种执行模式、接口清单、性能与踩坑记录 |
| `DATA_CATALOG.md` | **逐字段数据字典**（2026-09-19 快照）—— 需要时可 `scripts/catalog.py` 再生 |
| `DATA_GUIDE.md` | 给下游（因子/模型）的数据使用指引（部分结论已被 §1 / §5 取代） |
| `CLAUDE.md` | 交接记录：delay 结论、发布时点台账、闸门三档语义、三个真缺陷的修复 |
| `probe/REPORT_dataset_shapes.md` | 接口形态实测报告：响应信封、服务端硬限制、逐接口的坑 |
| ~~`灵启数据API有权限接口文档 (1).md`~~ | ★ **已从归档取回、保留在本目录**（2026-09-19 按用户指示）。归档包里同样留有一份快照 |

恢复：`tar -xzf /autodl-fs/data/_archive/datadownload_docs_20260919.tar.gz -C <目标目录>`

> 接口文档的两个副本：本目录这份（厂商生成于 **2026-09-13**，**59 个接口**，较新）与
> `/home/claude/ref/LINGQIDATA/docs/灵启数据API有权限接口文档.md`（生成于 2026-08-14，58 个接口，旧）。
> 同目录还有 `document.md` / `new_detail.md` 两份镜像，接口文档不是单点。

> ⚠️ 归档是**历史快照**：里面的行数、水位、delay 表都可能已过期，
> 涉及**当前**状态的（哪些表在更新、delay 是多少、闸门怎么判）一律以
> `../everyday_tasks/README.md` 与 `../everyday_tasks/data_incremental/registry.py` 为准。

---

## 7. 接手入口

| 你要做的事 | 去哪 |
|:--|:--|
| **每日增量 / 跑批 / 补跑** | `../everyday_tasks/README.md`（该工程唯一文档） |
| 整个平台的主记录（现状、决策沿革、踩坑） | `../QUANT_PLATFORM.md` |
| 读数据 / 字段含义 | 本文件 §2 + 按需 `scripts/catalog.py` 生成本地字典 |
| 历史决策与实测证据 | §6 的归档包 |
