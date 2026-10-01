# V23：60 日风险调整动量 Top2

- 同一 OOS 生命周期：2020-01-02 至 2026-06-30；78 个月度信号；四银行价格快照复用共享缓存。
- 规则：信号收盘计算复权 60 日累计收益，除以截至信号收盘的 60 个复权日收益样本标准差；按分数降序选 Top2，等权，下一交易日开盘执行。
- V11 26 季分数缓存只用于 Ranker 基线校验；未重训，也未执行模型推理。四条策略使用 V11 同一现金引擎。

## 总体对比

| 策略 | 净收益 | 年化 | 最大回撤 | 年化波动 | 费用 | 滑点 | 换手 | 交易数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V23 60d risk-adjusted momentum Top2 | 108.08% | 12.48% | -17.39% | 16.66% | ¥2,945 | ¥1,623 | ¥5,409,108 | 139 |
| 60d raw momentum Top2 | 117.56% | 13.29% | -17.41% | 16.97% | ¥2,773 | ¥1,489 | ¥4,964,796 | 140 |
| V11 LambdaRank Top2 | 114.89% | 13.06% | -18.06% | 16.62% | ¥5,045 | ¥2,877 | ¥9,590,335 | 190 |
| four-bank equal-weight hold | 102.17% | 11.96% | -15.48% | 15.99% | ¥26 | ¥30 | ¥99,006 | 4 |

## 风险调整动量相对基线

- 相对60 日原始动量 Top2：收益差 -9.48 pp；回撤差 +0.02 pp。
- 相对V11 Ranker Top2：收益差 -6.80 pp；回撤差 +0.68 pp。
- 相对四股等权持有：收益差 +5.91 pp；回撤差 -1.91 pp。

## 年度收益

| 年份 | 策略 | 净收益 | 同成交毛收益 |
|---|---|---:|---:|
| 2020 | V23 60d risk-adjusted momentum Top2 | -12.48% | -11.84% |
| 2021 | V23 60d risk-adjusted momentum Top2 | 2.11% | 2.73% |
| 2022 | V23 60d risk-adjusted momentum Top2 | 3.00% | 3.48% |
| 2023 | V23 60d risk-adjusted momentum Top2 | 23.77% | 24.31% |
| 2024 | V23 60d risk-adjusted momentum Top2 | 41.49% | 41.45% |
| 2025 | V23 60d risk-adjusted momentum Top2 | 31.64% | 31.43% |
| 2026 | V23 60d risk-adjusted momentum Top2 | -1.95% | -1.82% |
| 2020 | 60d raw momentum Top2 | -10.72% | -10.13% |
| 2021 | 60d raw momentum Top2 | -0.72% | 0.07% |
| 2022 | 60d raw momentum Top2 | 2.95% | 3.46% |
| 2023 | 60d raw momentum Top2 | 30.46% | 30.40% |
| 2024 | 60d raw momentum Top2 | 41.53% | 41.55% |
| 2025 | 60d raw momentum Top2 | 31.68% | 31.55% |
| 2026 | 60d raw momentum Top2 | -1.94% | -1.82% |
| 2020 | V11 LambdaRank Top2 | -8.69% | -7.84% |
| 2021 | V11 LambdaRank Top2 | -1.25% | -0.25% |
| 2022 | V11 LambdaRank Top2 | 6.61% | 7.66% |
| 2023 | V11 LambdaRank Top2 | 26.01% | 26.34% |
| 2024 | V11 LambdaRank Top2 | 50.00% | 49.31% |
| 2025 | V11 LambdaRank Top2 | 31.92% | 32.12% |
| 2026 | V11 LambdaRank Top2 | -10.35% | -9.67% |
| 2020 | four-bank equal-weight hold | -10.92% | -10.86% |
| 2021 | four-bank equal-weight hold | -0.53% | -0.53% |
| 2022 | four-bank equal-weight hold | 4.97% | 4.97% |
| 2023 | four-bank equal-weight hold | 27.07% | 27.05% |
| 2024 | four-bank equal-weight hold | 48.19% | 48.17% |
| 2025 | four-bank equal-weight hold | 24.88% | 24.87% |
| 2026 | four-bank equal-weight hold | -7.56% | -7.56% |

## 季度收益

| 季度 | 策略 | 净收益 | 同成交毛收益 |
|---|---|---:|---:|
| 2020Q1 | V23 60d risk-adjusted momentum Top2 | -10.87% | -10.66% |
| 2020Q2 | V23 60d risk-adjusted momentum Top2 | -1.67% | -1.43% |
| 2020Q3 | V23 60d risk-adjusted momentum Top2 | -1.92% | -1.67% |
| 2020Q4 | V23 60d risk-adjusted momentum Top2 | 1.81% | 1.81% |
| 2021Q1 | V23 60d risk-adjusted momentum Top2 | 13.97% | 13.89% |
| 2021Q2 | V23 60d risk-adjusted momentum Top2 | -6.60% | -6.40% |
| 2021Q3 | V23 60d risk-adjusted momentum Top2 | -3.51% | -3.23% |
| 2021Q4 | V23 60d risk-adjusted momentum Top2 | -0.58% | -0.41% |
| 2022Q1 | V23 60d risk-adjusted momentum Top2 | 5.02% | 5.11% |
| 2022Q2 | V23 60d risk-adjusted momentum Top2 | -3.55% | -3.33% |
| 2022Q3 | V23 60d risk-adjusted momentum Top2 | -0.31% | -0.13% |
| 2022Q4 | V23 60d risk-adjusted momentum Top2 | 2.00% | 1.98% |
| 2023Q1 | V23 60d risk-adjusted momentum Top2 | 6.27% | 6.32% |
| 2023Q2 | V23 60d risk-adjusted momentum Top2 | 9.80% | 9.79% |
| 2023Q3 | V23 60d risk-adjusted momentum Top2 | 5.78% | 5.85% |
| 2023Q4 | V23 60d risk-adjusted momentum Top2 | 0.29% | 0.60% |
| 2024Q1 | V23 60d risk-adjusted momentum Top2 | 8.70% | 8.77% |
| 2024Q2 | V23 60d risk-adjusted momentum Top2 | 8.32% | 8.13% |
| 2024Q3 | V23 60d risk-adjusted momentum Top2 | 11.42% | 11.35% |
| 2024Q4 | V23 60d risk-adjusted momentum Top2 | 7.85% | 8.01% |
| 2025Q1 | V23 60d risk-adjusted momentum Top2 | 2.12% | 2.16% |
| 2025Q2 | V23 60d risk-adjusted momentum Top2 | 9.53% | 9.48% |
| 2025Q3 | V23 60d risk-adjusted momentum Top2 | 5.36% | 5.33% |
| 2025Q4 | V23 60d risk-adjusted momentum Top2 | 11.71% | 11.56% |
| 2026Q1 | V23 60d risk-adjusted momentum Top2 | -0.59% | -0.51% |
| 2026Q2 | V23 60d risk-adjusted momentum Top2 | -1.36% | -1.32% |
| 2020Q1 | 60d raw momentum Top2 | -9.08% | -8.95% |
| 2020Q2 | 60d raw momentum Top2 | -0.70% | -0.62% |
| 2020Q3 | 60d raw momentum Top2 | -1.92% | -1.67% |
| 2020Q4 | 60d raw momentum Top2 | 0.83% | 1.00% |
| 2021Q1 | 60d raw momentum Top2 | 13.96% | 13.89% |
| 2021Q2 | 60d raw momentum Top2 | -7.55% | -7.35% |
| 2021Q3 | 60d raw momentum Top2 | -3.51% | -3.24% |
| 2021Q4 | 60d raw momentum Top2 | -2.34% | -1.99% |
| 2022Q1 | 60d raw momentum Top2 | 5.02% | 5.12% |
| 2022Q2 | 60d raw momentum Top2 | -3.58% | -3.35% |
| 2022Q3 | 60d raw momentum Top2 | -0.32% | -0.13% |
| 2022Q4 | 60d raw momentum Top2 | 1.99% | 1.98% |
| 2023Q1 | 60d raw momentum Top2 | 6.28% | 6.32% |
| 2023Q2 | 60d raw momentum Top2 | 14.58% | 14.30% |
| 2023Q3 | 60d raw momentum Top2 | 5.37% | 5.31% |
| 2023Q4 | 60d raw momentum Top2 | 1.68% | 1.90% |
| 2024Q1 | 60d raw momentum Top2 | 8.81% | 8.75% |
| 2024Q2 | 60d raw momentum Top2 | 8.30% | 8.23% |
| 2024Q3 | 60d raw momentum Top2 | 11.34% | 11.31% |
| 2024Q4 | 60d raw momentum Top2 | 7.87% | 8.05% |
| 2025Q1 | 60d raw momentum Top2 | 2.10% | 2.14% |
| 2025Q2 | 60d raw momentum Top2 | 9.56% | 9.54% |
| 2025Q3 | 60d raw momentum Top2 | 5.42% | 5.41% |
| 2025Q4 | 60d raw momentum Top2 | 11.66% | 11.55% |
| 2026Q1 | 60d raw momentum Top2 | -0.59% | -0.50% |
| 2026Q2 | 60d raw momentum Top2 | -1.36% | -1.33% |
| 2020Q1 | V11 LambdaRank Top2 | -12.77% | -12.64% |
| 2020Q2 | V11 LambdaRank Top2 | -0.52% | -0.36% |
| 2020Q3 | V11 LambdaRank Top2 | -1.41% | -1.08% |
| 2020Q4 | V11 LambdaRank Top2 | 6.73% | 7.03% |
| 2021Q1 | V11 LambdaRank Top2 | 7.75% | 7.85% |
| 2021Q2 | V11 LambdaRank Top2 | -4.65% | -4.44% |
| 2021Q3 | V11 LambdaRank Top2 | -3.36% | -3.00% |
| 2021Q4 | V11 LambdaRank Top2 | -0.55% | -0.22% |
| 2022Q1 | V11 LambdaRank Top2 | 6.34% | 6.46% |
| 2022Q2 | V11 LambdaRank Top2 | -0.33% | 0.00% |
| 2022Q3 | V11 LambdaRank Top2 | -2.00% | -1.73% |
| 2022Q4 | V11 LambdaRank Top2 | 2.65% | 2.89% |
| 2023Q1 | V11 LambdaRank Top2 | 5.93% | 6.08% |
| 2023Q2 | V11 LambdaRank Top2 | 11.92% | 11.82% |
| 2023Q3 | V11 LambdaRank Top2 | 4.59% | 4.62% |
| 2023Q4 | V11 LambdaRank Top2 | 1.61% | 1.80% |
| 2024Q1 | V11 LambdaRank Top2 | 9.28% | 9.25% |
| 2024Q2 | V11 LambdaRank Top2 | 7.99% | 7.91% |
| 2024Q3 | V11 LambdaRank Top2 | 12.81% | 12.52% |
| 2024Q4 | V11 LambdaRank Top2 | 12.67% | 12.56% |
| 2025Q1 | V11 LambdaRank Top2 | 2.26% | 2.44% |
| 2025Q2 | V11 LambdaRank Top2 | 6.48% | 6.54% |
| 2025Q3 | V11 LambdaRank Top2 | 6.19% | 6.35% |
| 2025Q4 | V11 LambdaRank Top2 | 14.10% | 13.84% |
| 2026Q1 | V11 LambdaRank Top2 | -6.56% | -6.23% |
| 2026Q2 | V11 LambdaRank Top2 | -4.05% | -3.67% |
| 2020Q1 | four-bank equal-weight hold | -10.83% | -10.78% |
| 2020Q2 | four-bank equal-weight hold | 0.37% | 0.37% |
| 2020Q3 | four-bank equal-weight hold | -0.93% | -0.93% |
| 2020Q4 | four-bank equal-weight hold | 0.47% | 0.47% |
| 2021Q1 | four-bank equal-weight hold | 10.32% | 10.31% |
| 2021Q2 | four-bank equal-weight hold | -5.98% | -5.98% |
| 2021Q3 | four-bank equal-weight hold | -3.51% | -3.51% |
| 2021Q4 | four-bank equal-weight hold | -0.62% | -0.62% |
| 2022Q1 | four-bank equal-weight hold | 5.57% | 5.57% |
| 2022Q2 | four-bank equal-weight hold | -1.48% | -1.48% |
| 2022Q3 | four-bank equal-weight hold | -0.55% | -0.55% |
| 2022Q4 | four-bank equal-weight hold | 1.48% | 1.47% |
| 2023Q1 | four-bank equal-weight hold | 5.57% | 5.57% |
| 2023Q2 | four-bank equal-weight hold | 10.83% | 10.82% |
| 2023Q3 | four-bank equal-weight hold | 5.27% | 5.27% |
| 2023Q4 | four-bank equal-weight hold | 3.17% | 3.17% |
| 2024Q1 | four-bank equal-weight hold | 10.67% | 10.66% |
| 2024Q2 | four-bank equal-weight hold | 7.21% | 7.20% |
| 2024Q3 | four-bank equal-weight hold | 12.59% | 12.58% |
| 2024Q4 | four-bank equal-weight hold | 10.95% | 10.94% |
| 2025Q1 | four-bank equal-weight hold | 1.92% | 1.92% |
| 2025Q2 | four-bank equal-weight hold | 8.59% | 8.59% |
| 2025Q3 | four-bank equal-weight hold | -0.17% | -0.17% |
| 2025Q4 | four-bank equal-weight hold | 13.02% | 13.02% |
| 2026Q1 | four-bank equal-weight hold | -3.58% | -3.58% |
| 2026Q2 | four-bank equal-weight hold | -4.13% | -4.13% |

## 核心审计

- 月度 60 日波动窗口均为 60 个已实现收益并截止当日信号收盘；T+1 日期检查通过；缓存完整性哈希检查通过。
- V23 现金最低余额 ¥19.82；阻塞交易 0；账本最大残差 2.91e-11；零目标 odd-lot 退出 18 笔。
- 所有策略同一 2020-01-02 至 2026-06-30 生命周期，期末按 V11 口径保留持仓并以收盘价计值，不强制清仓。

历史回测结果用于策略比较，不能证明未来收益稳定或存在可持续 alpha。

运行脚本 SHA-256：2a2c5851b665c2bc5119543d95051e99bbf1f7c9e00d11410ae30213bc69b682
现金引擎 SHA-256：b8015d3b2336a0c0262ce8738fabd45262847f5421c37262f911e8fe4a4b181c
缓存清单 SHA-256：e414d640f30954003144a597c221d4469adc55043fd8e81f7d2c666f536b2999
