# V32：V26 单因子前瞻扩展

- 因子固定为 mf_tier_flow_agreement_20，方向沿用 V26（高值优先），名单与方向截止 2023Q4；不训练、不按扩展结果改策略。
- 共用初始资金 CNY 100,000 的独立账户；月初收盘信号、下一交易日开盘交易。末日前一交易日发零目标，窗口末日开盘按 V11 修正账本强制退出；退出日收益与费用单列。

## V26 已知共同窗口（原结果，6/30 收盘估值）

| Strategy | Net return | Max drawdown | Ann. vol. |
|---|---:|---:|---:|
| V26 mf_tier_flow_agreement_20 Top2 | 84.71% | -15.50% | 18.77% |
| V11 LambdaRank Top2 | 76.29% | -18.03% | 19.25% |
| 60d momentum Top2 | 81.69% | -17.41% | 19.03% |
| equal-weight hold | 70.44% | -15.42% | 18.36% |

原 V26 窗口结果按 6/30 收盘估值；下表 V32 共同窗口改为 6/30 开盘强平，两者退出价格口径不同。

## V32 共同窗口（统一 6/30 开盘强平）

| Strategy | Net return | Max drawdown | Ann. vol. | Exit-day net return |
|---|---:|---:|---:|---:|
| V26 mf_tier_flow_agreement_20 Top2 | 87.28% | -15.50% | 18.73% | -0.69% |
| 60d momentum Top2 | 83.45% | -17.41% | 19.00% | -0.75% |
| equal-weight hold | 72.80% | -15.42% | 18.31% | -0.73% |
| V11 LambdaRank Top2 | 78.58% | -18.03% | 19.21% | -0.77% |

## V32 延伸窗口（2026Q3 部分；9/24 开盘强平）

| Strategy | Net return | Max drawdown | Ann. vol. | Exit-day net return |
|---|---:|---:|---:|---:|
| V26 mf_tier_flow_agreement_20 Top2 | 121.20% | -15.50% | 19.16% | -0.03% |
| 60d momentum Top2 | 103.39% | -17.41% | 19.40% | -0.03% |
| equal-weight hold | 95.35% | -15.42% | 18.78% | -0.04% |

扩展段相对统一 6/30 开盘退出共同段的净收益变化：
- V26 mf_tier_flow_agreement_20 Top2: +33.91%。
- 60d momentum Top2: +19.94%。
- equal-weight hold: +22.54%。

V11 Ranker 仅在共同窗口截至 2026-06-30 比较；没有生成 Q3 预测。

因子和动量信号由 30 个共同月份延长为 33 个，新增日期：2026-07-01, 2026-08-03, 2026-09-01。样本只增加约三个月，不能据此确认稳定性。
逐日权益、成交、季度/年度统计、终端强平审计和输入散列均保存在本目录。
