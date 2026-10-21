# 平台与比赛资料核验

核验日期：2026-10-07（UTC）  
范围：量化平台因子文档、官方数据 API 字段、金融类比赛官方页面。这里只做资料整理，没有实现或运行交易策略。

本目录包含 29 条来源记录和 403 条因子/字段/赛事任务元数据记录。记录不是 403 条独立有效性证据：其中 292 条是 Alpha101/Alpha191 的 `formula_record` 编号公式摘要，不应按编号数视为独立经济机制；47 条是米筐风险模型暴露因子的公开概念说明，57 条是平台数据字段或字段集合，1 条是研究工具机制，6 条是赛事任务。没有使用排行榜收益证明因子有效，也没有复录大段公式或页面原文。

## 核验结果

- **JoinQuant / 聚宽：** [Alpha101](https://www.test.demo.joinquant.com/data/dict/alpha101) 和 [Alpha191](https://www.test.demo.joinquant.com/data/dict/alpha191) 的官方托管文档列出编号公式及算子说明。Alpha101 源自 [WorldQuant 论文](https://arxiv.org/abs/1601.00991)；Alpha191 文档说明其日频个股价量输入、对部分缺失或不合理公式的调整，并明确是否适用A股需自行研究。目录为每个编号保存自写的变量/运算类型摘要，公式本身仍以链接页面为准。聚宽当前生产域在本环境被地区访问限制挡住，因此 [当前因子看板目录](https://www.joinquant.com/view/factorlib/list) 只记录为 `access_limited` 元数据线索，未猜写其当前完整清单或公式。历史更新公告只能证明曾有因子库、分类和列表 API，不能代替现行目录。
- **Ricequant / 米筐：** [公开风险模型文档](https://www.ricequant.com/doc/rqdata/python/risk-factors-mod.html) 列出 v1、v2、v2trd 等模型、预测期限、股票池，以及 STOM、BTOP、RSTR、ROA、DTOP、INDMOM 等 47 个唯一风格因子的分类和概念描述。公开页面没有完整计算公式；定制模型和白皮书要求联系米筐。记录只表示因子身份/概念已核验，不推断内部变换、标准化或中性化过程。
- **BigQuant：** [Alpha191公式页](https://bigquant.com/wiki/doc/Pyf0TYya6H) 和 [Alpha191数据表页](https://bigquant.com/data/datasources/cn_stock_factors_alpha_191) 在检索索引中可见，但本次正文抓取超时。表页仅记作 Alpha191 编号列及日期/标的字段的元数据线索；`definition_verified=false`，不将 BigQuant 的公式或字段口径写成已核验定义。[BigAlpha 因子平台](https://bigquant.com/wiki/doc/EOVmVtJMS5) 也只记录自定义因子提交/展示的工具入口。
- **Tushare 与 AKShare：** Tushare 官方接口文档支持核对日线、每日估值与换手率字段、技术指标字段、财务指标及三张报表字段；部分接口明确需要积分或单独权限。[AKShare 股票文档](https://akshare.akfamily.xyz/data/stock/stock.html) 描述的是封装上游行情接口的字段目录，实时快照与历史回测的PIT数据不可混为一谈。字段记录遵循 `factor_id_only`；除文档明确给出计算口径的字段外，不补全公式。

## 比赛范围与迁移边界

- **天池证券数据分析学习赛**使用沪市/深市指数和两融数据，作业涉及日、周、月图表、相关性及 MACD/KDJ 展示。页面中的年化收益或最大回撤是教学任务要求，不是公开、独立的因子有效性检验。
- **FDDC2018**是A股主板上市公司季度营收预测/公告抽取赛题，频率、目标和问题设定都不同于日频个股收益预测。数据目录仍在，但本次未核验下载授权或全量字段。
- **全国高校AI量化投资挑战赛（GIFP）**公告列出 IC 均值、夏普、最大回撤和换手率等评分维度；公告标注报名期从 2026-10-10 开始。本次核验日为 10 月 7 日，完整赛题、数据集和执行规则尚不能从该公告确认。
- **JPX Tokyo Stock Exchange Prediction（Kaggle）**是日本市场赛事，约两千标的按日排名，已于 2022 年结束。它可以作为日本市场时间序列任务案例，不能当成A股研究样本。
- **Two Sigma: Using News to Predict Stock Movements（Kaggle）**覆盖美国上市标的，以市场与新闻数据预测未来十日市场调整收益。赛事已关闭，官方页面说明数据已移除；新闻数据另受 Thomson Reuters 权利限制。
- **Optiver - Trading at the Close（Kaggle）**定位为美国股票收盘走势和收盘竞价任务，包含日内/竞价场景，不是A股日频因子赛题。

上述赛事可用于理解任务目标、数据结构、时间隔离和评分设计。排行榜只反映特定赛事数据、期限和评分规则下的成绩，不构成样本外可复现性、交易成本后收益或跨市场迁移能力的证明。

## 数据时点与许可注意事项

- Alpha101/Alpha191 公式通常使用当日 OHLCV、收益、滚动窗口或截面排序。若策略在收盘前决策，不得把当日收盘后才完整可得的高低收、成交量或当日截面排名当作已知值；截面股票池也要使用当时成分。
- Tushare 文档说明 `daily` 和 `daily_basic` 的入库时段；财务字段同时提供报告期和公告日期。财务回测须按公告实际可见时间取值，并明确处理更正/重述、复权和停牌。
- 米筐风险模型的公开概念说明不足以重建内部暴露；接入时要使用实际模型版本、行业映射和数据时间戳。
- AKShare 是数据接口封装层，不能替代上游数据授权。Tushare、聚宽、米筐和 BigQuant 的页面没有为本目录授予数据或公式的再分发权；调用权限、积分、会员、比赛条款及上游许可需单独遵守。此处仅保留短摘要和原始链接。

## 来源索引

完整摘要、核验状态、访问限制和许可/权限备注见 [sources.json](./sources.json)；逐因子/字段元数据见 [factors.json](./factors.json)。

### 平台与数据文档

- [JoinQuant Alpha101](https://www.test.demo.joinquant.com/data/dict/alpha101)、[原始论文 arXiv 页面](https://arxiv.org/abs/1601.00991)、[JoinQuant Alpha191](https://www.test.demo.joinquant.com/data/dict/alpha191)、[聚宽因子看板目录](https://www.joinquant.com/view/factorlib/list)、[平台更新公告](https://test.demo.joinquant.com/community/post/detailMobile?postId=553)
- [Ricequant 风险模型因子文档](https://www.ricequant.com/doc/rqdata/python/risk-factors-mod.html)、[Ricequant 文档目录](https://www.ricequant.com/doc/)
- [BigQuant Alpha191公式页](https://bigquant.com/wiki/doc/Pyf0TYya6H)、[Alpha191数据表](https://bigquant.com/data/datasources/cn_stock_factors_alpha_191)、[BigAlpha平台](https://bigquant.com/wiki/doc/EOVmVtJMS5)
- Tushare：[A股日线](https://tushare.pro/document/2?doc_id=27)、[每日指标](https://tushare.pro/document/2?doc_id=32)、[技术因子](https://tushare.pro/document/2?doc_id=296)、[财务指标](https://tushare.pro/document/2?doc_id=79)、[利润表](https://tushare.pro/document/2?doc_id=33)、[资产负债表](https://tushare.pro/document/2?doc_id=36)、[现金流量表](https://tushare.pro/document/2?doc_id=44)、[指数每日指标](https://tushare.pro/document/2?doc_id=128)、[美股财务指标](https://tushare.pro/document/2?doc_id=393)
- [AKShare 股票数据文档](https://akshare.akfamily.xyz/data/stock/stock.html)

### 比赛官方页面

- 天池：[证券数据分析学习赛](https://tianchi.aliyun.com/competition/entrance/531992/information)、[FDDC2018赛题索引](https://tianchi.aliyun.com/markets/tianchi/tianchideveloper)、[季度营收预测数据集目录](https://tianchi.aliyun.com/dataset/133410)
- 国内赛事：[GIFP全国高校AI量化投资挑战赛公告](https://www.gifp.cn/news/xhdt/115.html)
- 日本市场：[Kaggle JPX Tokyo Stock Exchange Prediction](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/overview)、[JPX主办方公告](https://www.jpx.co.jp/corporate/news/news-releases/6020/20221206-01.html)
- 美国市场：[Kaggle Two Sigma新闻赛题](https://www.kaggle.com/competitions/two-sigma-financial-news/overview)、[官方数据说明](https://www.kaggle.com/competitions/two-sigma-financial-news/data)、[Optiver Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview)

## JSON 字段约定

`factors.json` 保留要求的因子字段，并加上 `record_kind`、`definition_verified`、`usage_role`、必要时的 `model_versions`。Alpha101/Alpha191 的逐编号摘要统一为 `record_kind=formula_record`；这只表示原公式记录存在，不代表独立经济机制或独立验证。`usage_role` 区分 `alpha_candidate`、`risk_exposure`、`raw_data_field`、`research_method` 和 `competition_task`；因此米筐风险暴露和行情API字段不会被误读为已验证Alpha。`definition_verified=true` 表示原页面确实列出该公式、概念定义或赛事任务描述；它不代表预测有效性被验证。只有名称、字段 ID 或未能打开的 BigQuant 页面统一标为 `factor_id_only` 或 `method`，并将 `definition_verified=false`。`sources.json` 同时记录原要求的 `verified` 与补充的 `verification_status`（`full_text`、`abstract`、`metadata_only`），另加 `access_limited` 标记受地区、登录或数据权限影响的页面。
