# 模块③ · 模型工程

> 量化平台三模块之一（① 数据爬取 `datadownload/` → ② 因子工程 `featureengineering/` → ③ **模型**）。
> **输入** = 模块② 的因子产物；**输出** = A 股主板股票的**日频排序打分**（下游据此选股）。
> 平台级主记录在 [`../QUANT_PLATFORM.md`](../QUANT_PLATFORM.md)；上游口径以
> [`../featureengineering/README.md`](../featureengineering/README.md) 为准
> （旧文档里的 `CLAUDE.md` / `FACTORS.md` 已不存在）。
>
> ★ **2026-09-20：文档已全部搬进 [`README/`](README/README.md)**（原先挤在本文件与 `LOG.md` 里的
> 内容拆成 8 份，章节号 `§N` 全库统一；2026-09-20 又删掉了作废榜单存档那份）。本文件只剩一个指路牌。

## 文档导航（都在 [`README/`](README/README.md)）

| 想做什么 | 看哪份 |
|:--|:--|
| 先搞清结构、规矩、怎么跑 | [`README/SPEC.md`](README/SPEC.md)（+ [`README/README.md`](README/README.md) §5 命令速查） |
| 动数据 / 换快照 / 上游变了 | [`README/DATA.md`](README/DATA.md) |
| 读榜单、判定一个模型好不好 | [`README/EVAL.md`](README/EVAL.md) |
| 查某版为什么这么设计 | [`README/VERSIONS.md`](README/VERSIONS.md)（V1~V15 历史档案） |
| 避坑 / 查已确认的事实 | [`README/LESSONS.md`](README/LESSONS.md) |
| 下一步干什么 | [`README/TODO.md`](README/TODO.md) |
| 每一轮改了什么 | [`README/LOG.md`](README/LOG.md) |

## 当前状态（2026-09-20）

- **数据锚点**：`panel_digest=798ccb32214853a8` · 337 特征 × 2018-01-02 ~ 2026-09-18
  （2116 天 × 2115 只 = 4,475,340 行）。★ 引用任何旧数字前先核对它建在哪个锚点上。
- **数据布局**：`trainingdata/` **四块** —— `factors` / `target` / `amount` / `fac_sample`
  （2026-09-20 由七块收敛 + 改名；`universe`/`P`/`sample/` 已删，后两者的构建代码也删了）。
  ★ **日常一条命令**：`$PY preparingdata.py`（有新增走增量、没有就零 I/O 跳过）。
  快照当前**未冻结**。
- **⚠️ 代码层尚未对齐新块名**：`V16/mx/` 读的还是 `X`/`Y`/`universe`/`P`，
  所以**单元命令暂时跑不通**。待对齐清单见 [`README/DATA.md`](README/DATA.md) §3 末。
- **模板单元**：`V16/`（自包含，新版本从它复制）；**`best/` 是空的**（上一轮那份随数据重建作废）。
- **归档**：`history_iterations/` = V1 ~ V15（**不要跑**，见 [`README/SPEC.md`](README/SPEC.md) §0）。
- **旧榜已作废、新榜待建**：判定口径见 [`README/EVAL.md`](README/EVAL.md) §7。

## 顶层只放这些

`preparingdata.py`（构建期唯一入口，唯一写 `trainingdata/` 的地方）· `trainingdata/`（模型只读的初加工产物）·
`V16/`（模板单元）· `best/`（实战单元）· `history_iterations/`（归档）· `README/`（文档）· 本文件。

```bash
PY=/autodl-fs/data/miniconda3/bin/python        # ★ 必须全路径
cd /autodl-fs/data/model && $PY preparingdata.py --check   # 数据体检（日常第一条）
```
