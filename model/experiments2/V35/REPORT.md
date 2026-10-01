# V35：固定单因子 Top1 集中持仓留出实验

## 预先冻结的协议

协议在运行前冻结，SHA-256：`40359a710afb2f4c2701578c79d3f27b4d0a6cd7999ae56a8ac263c771febb70`。唯一因子仍为 `id2_close_vs_pm_vwap_20`，仅依据截至 2023Q4 的 16 个 V11 训练重要性折（10/16进入 top-20）选定；2024-01-02 至 2026-06-30 留出期不用于选因子或修改规则。
每个共同月初信号日收盘时按该因子在四家银行横截面排序，持有排名第一的单只银行，目标权重 100%，次一交易日开盘调仓。
交易费、1% 成交额参与上限、100 股整手、涨跌停阻断、复权股数、现金账本和期末估值沿用 V11/V24/V26/V34；不强制期末卖出。未重训、未扫参数。

## 留出结果

| Strategy | Net return | Annualized | Max drawdown | Ann. vol | Fees | Slippage | Trades | Max positions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V35 id2_close_vs_pm_vwap_20 Top1 | 88.21% | 30.42% | -12.10% | 19.45% | 3657.41 | 2165.57 | 47 | 1 |
| V34 id2_close_vs_pm_vwap_20 Top2 | 92.21% | 31.58% | -12.89% | 18.79% | 2362.69 | 1365.28 | 72 | 2 |
| 60d momentum Top2 | 81.69% | 28.50% | -17.41% | 19.03% | 1101.27 | 614.50 | 48 | 2 |
| V11 LambdaRank Top2 | 76.29% | 26.89% | -18.03% | 19.25% | 2123.41 | 1212.20 | 73 | 2 |
| V26 mf_tier_flow_agreement_20 Top2 | 84.71% | 29.40% | -15.50% | 18.77% | 1910.99 | 1103.43 | 63 | 2 |
| equal-weight hold | 70.44% | 25.10% | -15.42% | 18.36% | 25.91 | 29.88 | 4 | 4 |

### Top1 收益差

- V35 Top1 minus V34 id2_close_vs_pm_vwap_20 Top2: -4.00 percentage points.
- V35 Top1 minus 60d momentum Top2: +6.52 percentage points.
- V35 Top1 minus V11 LambdaRank Top2: +11.92 percentage points.
- V35 Top1 minus V26 stable-factor Top2: +3.50 percentage points.

季度和年度收益见 `quarterly_returns.csv` 与 `annual_returns.csv`；完整逐日账户和逐笔成交见 `daily_equity.csv` 与 `trades.csv`。

## 审计

- `membership_audit.json` 记录每次信号的四只银行因子值、横截面排名、Top1/Top2、执行日与无未来行标记。
- `audit.json` 核验留出窗口、训练期因子选择、30 个 V11 月初信号、次日执行、目标单仓权重、现金、持仓数和账本残差。
- V34/V26 Top2 按原冻结成员回放，V11、60d 和等权对照逐日现金、权益、持仓、逐笔成交及指标与其既有账本核对；容差见 `audit.json`。
- `cache_hashes.json` 列出输入缓存、训练期重要性、账本引擎、V24/V26/V34 参考产物和 V35 脚本哈希；`SHA256SUMS.txt` 列出本目录最终文件哈希。
- 这是四家银行上的历史固定留出结果；Top1 相较 Top2 的变化体现集中度与换仓成本差异，不单独证明未来稳定超额。
