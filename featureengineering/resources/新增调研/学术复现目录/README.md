# 学术因子复现目录

Open Source Asset Pricing 保存已发表研究的信号标识、定义和复现说明。它适合扩展研究线索，并帮助区分看起来相似的因子口径；美国市场证据需要结合A股数据和制度重新判断。

## 固定版本与记录

本地目录保存 331 条原始元数据对应记录，来源分类为 {'Predictor': 212, 'Placebo': 114, 'Drop': 5}。所有记录均标为 `factor_id_only`，不把目录行数当作独立有效机制数量。具体定义可在 [SignalDoc CSV](参考快照/SignalDoc.csv) 按 `Acronym` 查找 `Detailed Definition` 列；`source_row` 给出CSV物理行号。

版本固定为 `8db892442c2c3a3779b0f1eac4370d3655be15a1`，校验值见 [PROVENANCE](参考快照/PROVENANCE.json)。[GPL许可证](参考快照/LICENSE.txt) 与原文件一并保存。未引入或执行复现项目的计算代码。

## A股迁移时的阅读顺序

优先阅读价格成交和已披露财报类记录。分析师、机构持仓、期权及卖空类别需要额外历史快照或制度映射，不宜把美国13F、卖空余额、期权隐含波动字段直接等同A股数据。原研究的信号方向、收益结果和统计显著性不代表A股日频有效。

`Predictor` 与 `Placebo` 保留来源原分类，后者尤其不能当作有效预测因子。财报型记录可以按已知信息映射到每日，但这并不改变原研究的信号更新频率。

## 来源

- [作者项目数据说明](https://www.openassetpricing.com/data/)
- [固定版本信号文档](https://github.com/OpenSourceAP/CrossSection/blob/8db892442c2c3a3779b0f1eac4370d3655be15a1/SignalDoc.csv)
- [Global q 测试组合](https://global-q.org/testingportfolios.html)

Global q 另提供按主题组织的异常复现资料；其日频组合收益应与信号形成频率分开阅读。
