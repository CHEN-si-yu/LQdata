# 论文与学术证据：A股日频因子调研

调研日期：2026-10-07（UTC）。本目录仅保存来源索引、自写中文归纳与机制记录；未镜像论文全文，也未执行任何外部代码。

## 核验范围

本阶段核验 **36 条去重来源**，整理 **93 条因子/机制/方法记录**：其中机制 73 条、参数变体 12 条、因子索引 3 条、方法记录 5 条。

## 精选原文入口

| 主题 | 可直接打开的论文/机构来源 | 本次可核验的证据边界 |
|---|---|---|
| 中国规模与价值 | [Liu, Stambaugh & Yuan (2019), *Size and Value in China*](https://doi.org/10.1016/j.jfineco.2019.03.008) | 中国规模排除最小30%与E/P价值因子；月度因子模型证据。 |
| A股因子横向检验 | [Jansen, Swinkels & Zhou (2021), *Anomalies in the China A-share Market*](https://doi.org/10.1016/j.pacfin.2021.101607) | 32项A股异常分类测试；作者提醒样本期短，超额收益未扣交易成本。 |
| A股异常复现 | [Li et al. (2024), *Replicating and Digesting Anomalies in the Chinese A-Share Market*](https://doi.org/10.1287/mnsc.2023.4904) | 469变量复现；主板断点与市值加权下，多数候选未产生显著价差。 |
| 盈利与投资 | [Hou & Zhang (2019), PolyU机构库](https://ira.lib.polyu.edu.hk/handle/10397/90137)；[Fama & French (2015)](https://doi.org/10.1016/j.jfineco.2014.10.010)；[Hou, Xue & Zhang (2015)](https://doi.org/10.1093/rfs/hhu068) | 中国factor model结果与美股机制不同；FF5/q-factor只作为定义和迁移机制。 |
| 流动性/换手 | [Amihud (2002)](https://doi.org/10.1016/S1386-4181(01)00024-6)；[Datar, Naik & Radcliffe (1998)](https://doi.org/10.1016/S1386-4181(97)00004-9) | 日度Amihud输入和换手代理定义来自原始论文；国外机制需本地检验。 |
| MAX与特质波动 | [Wan (2018)](https://doi.org/10.1016/j.iref.2017.10.015)；[Hai et al. (2020)](https://doi.org/10.1016/j.najef.2020.101266) | 中国研究发现MAX/IVOL关系，但MAX是否解释IVOL在文献中并不一致。 |
| PEAD | [Truong (2011), *Post-earnings announcement abnormal return in the Chinese equity market*](https://doi.org/10.1016/j.intfin.2011.04.002)；[公告隔夜/日内研究 (2023)](https://doi.org/10.1016/j.jcorpfin.2023.102471) | 中国公告漂移与回报分段有直接证据；公告时间戳和事件窗口是关键限制。 |
| 投资者注意力 | [Dong et al. (2022)](https://doi.org/10.1016/j.intfin.2021.101499)；[Cai, Jiang & Liu (2022)](https://doi.org/10.1016/j.irfa.2022.102265) | 中国注意力因子及涨跌停计数代理；原文主要月度因子/横截面检验。 |
| 隔夜/日内与高频 | [Lin, Chang & Chou (2023)](https://doi.org/10.1016/j.pacfin.2023.102007)；[Zhang et al. (2019)](https://doi.org/10.1016/j.econmod.2018.08.009)；[Chu, Gu & Zhou (2019)](https://doi.org/10.1016/j.frl.2019.04.002) | 前者拆分异常组合回报；后两者需要半小时行情，不能用日线替代验证。 |
| 季节性 | [Liang, Liu & Zebedee (2022)](https://doi.org/10.1111/ajfs.12404)；[Meng, Du & Shu (2024)](https://doi.org/10.1016/j.pacfin.2024.102391) | 春节农历月份与个股同月收益季节性均有中国直接研究；不是每日新信息。 |

## 核验与频率口径

来源优先采用期刊出版页、作者/大学机构库、NBER、arXiv或SSRN原始记录。`verification_status` 描述实际查看材料：`full_text` = 可读全文；`abstract` = 已查看来源页摘要；`metadata_only` = 只核对题录或元数据。`verified` 表示来源URL、题名/题录和归纳范围经过核对，不代表本调研独立复现结论。作者、年份、刊物和DOI（如有）保存在 `sources.json`。SSRN/arXiv项目按预印本标注，证据等级低于同行评审论文。

`factors.json` 同时记录 `frequency`（本地建议落盘/更新节奏）和 `evidence_frequency`（原文实际形成、检验或报告收益的频率）。价格/成交/风险值可以日频生成，但论文若按月排序和报告月度收益，不代表日频策略已验证。财报指标可以每日落盘，但只在财报公开后更新；日频存储不等于日频产生信息。公告事件按正式公开时间对齐，不能用报告期末代替公开日。高频记录需要半小时、逐笔或账户数据，普通日线不足以验证。

`definition_verified=true` 仅用于来源中可核实的定义；摘要只支持“存在某机制”而不能确定精确窗口/公式时设为false，并在限制中注明。A股复现研究显示主板断点、市值加权、微盘容量及多重检验会明显改变结果；日频研究还需考虑T+1、涨跌停、停牌与交易成本。

文件：`sources.json` 是去重来源索引；`factors.json` 为机制表。每条机制记录含类别、定义核实状态、建议频率、论文频率、所需数据、A股关联、来源、PIT注意事项及限制。
