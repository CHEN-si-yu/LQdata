# GitHub 公开量化因子与因子挖掘资料

本目录整理 GitHub 与论文/项目主页中可核验的日频量价特征、因子编号线索和因子发现方法。仅做来源调研和短摘要，不开发或运行任何因子实现。

## 文件与数量

- [sources.json](sources.json)：22 条来源记录，包含访问时间、核验状态、许可证信息、精确文件/论文定位和局限。
- [factors.json](factors.json)：527 条记录，其中 518 条为 Qlib 特征定义，9 条为方法记录。
- [Qlib 参考快照](reference_snapshots/microsoft_qlib_be725493/)：仅保留 Qlib MIT 许可源码的两个窄范围文本快照（后缀 `.py.txt`）和原始许可证文本；SHA `be725493eb1a6bbb42bf11b37aa7669f59610ff1`。

Qlib 的 518 个 feature rows 不等于 518 个独立 alpha。按源码机制聚合为 48 组：Alpha158 有 42 组（9 个 K 线表达式、4 个当日价格比值、29 个滚动算子），Alpha360 有 6 组字段归一化机制。Alpha158 的 145 条窗口记录和 Alpha360 的 360 条滞后记录均标为 `parameter_variant`；各记录的 `mechanism_group` 和 `variant_parameter` 可用于去重。方法记录标为 `method`，不计入因子机制数。

## Qlib Alpha158 与 Alpha360

Qlib 源码按 commit 固定，见 `sources.json` 的 `qlib_loader`、`qlib_handler` 和 `qlib_benchmark`。Alpha158 handler 默认配置中，K 线组包含 9 个蜡烛特征；价格组取当日 OPEN/HIGH/LOW/VWAP 对 CLOSE 比值；滚动组使用 5、10、20、30、60 个交易日窗口，共 29 个算子、145 个窗口特征。源码还支持自定义窗口/算子，但本清单只登记 handler 默认的 158 条，不把配置变体误称为新机制。

Alpha360 由 CLOSE、OPEN、HIGH、LOW、VWAP、VOLUME 六组字段构成，每组取 0 至 59 个交易日滞后值。价格字段按当日 CLOSE 归一化，成交量按当日 VOLUME 加数值稳定项归一化。这些是供模型使用的时序特征维度，不是 360 个经证实的独立交易信号。

源码里的 Alpha158/Alpha360 默认监督标签使用负向 `Ref` 取未来价格。标签仅用于训练目标，不能混入当期因子输入。源码定义能确认特征计算方式，不能证明因子预测能力或实盘收益。

## Alpha101、GTJA 191 与代码出处

`wq101_paper` 保存 Kakushadze 的《101 Formulaic Alphas》论文入口，`wq101_code` 保存一份社区 Python 实现的文件/commit 定位；代码不是原作者实现，仓库未见明确许可证，因此没有将代码复制到本目录。`gtja191_dolphindb` 指向 DolphinDB 对 GTJA 191 模块的说明，记录其字段及文件定位；不复制该模块源码，也不把 191 个编号当成已逐条核验的独立有效因子。当前 factors.json 不扩展这两组的编号清单，避免重复登记和误读定义。

## 因子挖掘方法

方法条目覆盖 AlphaGen（组合导向强化学习）、AlphaForge（生成与动态组合）、AutoAlpha（分层遗传/PCA-QD 搜索）、AlphaEvolve（关系特征与 AutoML）、Alpha²（逻辑约束程序搜索）、Alpha-GFN、RD-Agent(Q)、OpenFE，以及 BigQuant AlphaMiner 的评价流程。`AlphaMiner` 在这里指文档所述因子清洗、分组、IC 与绩效分析框架，不是固定公式库或因子生成算法。AlphaGen、AlphaForge、AutoAlpha 等源的实验和适用市场各异；方法摘要不代表其生成的候选已经在 A 股验证。

OpenFE 是通用表格特征工程来源，不是金融时序因子证据。BigQuant 的 OpenFE 案例用于记录本地化研究线索，平台文章中的绩效或流程结论不作为独立复现实证。

## 事实与适配判断

`definition_summary` 和来源摘要记录源码、论文摘要或产品文档直接表达的内容；`a_share_relevance` 明确分成“来源事实”和“适配推断”。Qlib 的 benchmark 页面确认项目包含中国市场数据设置；适配 A 股仍须由研究者使用点时成分、统一复权口径和可交易日数据重新核验。

A 股日频使用时，Qlib 量价特征应按收盘后可得字段生成，并在下一可交易时点使用。需单独处理停牌、涨跌停、复权、上市/退市、指数成分历史和成交成本；财务字段还需按公告发布时间对齐。高低价和收盘价都只在当日结束后完整可知，不能用它们构造信号又假设当日成交。

## 核验与许可说明

`verified: true` 表示来源链接/条目已通过网页或仓库页面核验，不表示因子有效。`verification_status` 用 `full_text`、`abstract`、`metadata_only` 区分实际读到的材料范围。没有明确许可证的代码只保留链接、commit 和自写摘要，不在本目录镜像。Qlib 快照仅是 MIT 许可源码的小范围参考材料，原许可证随附。
