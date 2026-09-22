# 2026-09-21 新因子交付：单季度质量与稳定性

**已完成：12 个新因子已按现有开发框架接入并落地，正式入口支持后续增量更新。**

- 工作范围：只读 `datadownload/data`，开发与交付在 `featureengineering`。
- 当前注册：349 个因子 + 5 个标签；旧有 342 个注册对象的计算配方未变。
- 数据范围：2018-01-02 至 2026-09-18，固定股票池 2115 只，108 个年度分区。
- 本批总行数：53,704,080；压缩 parquet 合计约 120.3 MiB。
- 这些是经过工程验收的研究候选；本轮没有运行 IC、相关性筛选、训练或交易回测，不声称提高收益。

## 与模型会话的分工

详见 [AGENTS.md](AGENTS.md)。模型会话 `codex://threads/01a0c1f7-4e1d-7e52-b5e9-deda2583e211` 负责模型与交易策略；本会话负责新因子。未修改 `model/`、`trainingdata/`、上游数据或共享 Python 环境。

未发现可直接发消息给独立对话的工具，也尚未收到对方确认。模型侧需要新因子时可在 `FACTOR_REQUESTS.md` 留下需求。纳入本批因子和更新模型快照由模型会话决定，保持训练输入版本可追溯。

## 因子清单

`P` 为当时已披露的最新报告期；`P-k` 按报告期回看，绝不是日频向前移动 k 天。

| 因子 | 含义 | 公式 | 解释方向 |
|---|---|---|---|
| `qf_cash_margin_floor_4q` | 最近四季经营现金收入比最低值 | `min(q_ocf_to_sales(P-k),k=0..3)/100` | 较高 |
| `qf_cash_margin_vol_4q` | 最近四季经营现金收入比波动 | `std_population(q_ocf_to_sales(P-k),k=0..3)/100` | 较低 |
| `qf_cash_margin_yoy_change` | 单季度经营现金收入比同比改善 | `(q_ocf_to_sales(P)-q_ocf_to_sales(P-4))/100` | 较高 |
| `qf_core_roe_floor_4q` | 最近四季扣非ROE最低值 | `min(q_dt_roe(P-k),k=0..3)/100` | 较高 |
| `qf_core_roe_vol_4q` | 最近四季扣非ROE波动 | `std_population(q_dt_roe(P-k),k=0..3)/100` | 较低 |
| `qf_core_roe_yoy_change` | 单季度扣非ROE同比改善 | `(q_dt_roe(P)-q_dt_roe(P-4))/100` | 较高 |
| `qf_noncore_roe_gap` | 单季度ROE中的非经常损益贡献差 | `(q_roe(P)-q_dt_roe(P))/100` | 较低 |
| `qf_roa_yoy_change` | 单季度资产净利率同比改善 | `(q_npta(P)-q_npta(P-4))/100` | 较高 |
| `qf_roe_yoy_change` | 单季度ROE同比改善 | `(q_roe(P)-q_roe(P-4))/100` | 较高 |
| `qf_sales_growth_accel` | 单季度收入同比增速的环比变化 | `(q_sales_yoy(P)-q_sales_yoy(P-1))/100` | 较高 |
| `qf_sales_growth_floor_4q` | 最近四季收入同比增长的最低值 | `min(q_sales_yoy(P-k),k=0..3)/100` | 较高 |
| `qf_sales_growth_vol_4q` | 最近四季收入同比增长波动 | `std_population(q_sales_yoy(P-k),k=0..3)/100` | 较低 |

方向只是经济解释，统一 `rank` 始终按原始值递增排名，函数不自行翻转符号。

## 框架与数据口径

- 家族文件：`factors/quarterly_quality.py`；注册：`factors/__init__.py`。
- 仅给 `fea/deriv.py` 的安全白名单新增 5 个已核对的单季度字段：`q_roe, q_dt_roe, q_npta, q_ocf_to_sales, q_sales_yoy`；`fea/context.py` 同步说明。未更改财务版本表算法，累计 YTD 指标仍禁止直接使用。
- 全部从 `stock_financial_indicator` 读取，显式声明 `deps` 与 `fin_fields`，复用 `FactorSpec → FactorContext → Derivative → Engine → Store`。
- 百分数字段除以 100 转为比例。同比改善为两个比例之差；四季波动用总体标准差，要求四个连续报告期均有效。无穷值转 NaN，真实负值和单字段零值保留。
- 2026Q2 有 52 只池内股票的五字段同时为零，疑似整组占位；本家族保守将该报告期置缺失，不回退到旧报告，也不把所有单独零值删除。此规则仅作用于新因子。
- `warmup_days=1100`，与 2200 日预热对照通过；当晚生成供次日使用。
- 沿用四列 `trade_date, stock_code, value, rank` 和 `string,string,float32,float32`，统一缩尾、排名和按年原子写入。

## 质量检查（全部新产物，非抽样）

| 因子 | 各年非空率范围 | 最低每日有效股票数中位数 | 最少每日 rank 档数 | 最大绝对值 |
|---|---:|---:|---:|---:|
| `qf_cash_margin_floor_4q` | 98.23%–100.00% | 2103 | 1966 | 11873.7 |
| `qf_cash_margin_vol_4q` | 98.23%–100.00% | 2103 | 1970 | 1.11117e+06 |
| `qf_cash_margin_yoy_change` | 99.21%–100.00% | 2115 | 2014 | 2.56718e+06 |
| `qf_core_roe_floor_4q` | 98.23%–100.00% | 2103 | 1834 | 23.1791 |
| `qf_core_roe_vol_4q` | 98.23%–100.00% | 2103 | 1970 | 10.0368 |
| `qf_core_roe_yoy_change` | 99.21%–100.00% | 2115 | 1998 | 4.70545 |
| `qf_noncore_roe_gap` | 99.77%–100.00% | 2115 | 1915 | 5.54777 |
| `qf_roa_yoy_change` | 99.21%–100.00% | 2115 | 1991 | 1.24304 |
| `qf_roe_yoy_change` | 99.21%–100.00% | 2115 | 2004 | 6.49189 |
| `qf_sales_growth_accel` | 99.77%–100.00% | 2114 | 2023 | 506.986 |
| `qf_sales_growth_floor_4q` | 98.23%–100.00% | 2103 | 1866 | 97.8895 |
| `qf_sales_growth_vol_4q` | 98.23%–100.00% | 2103 | 1970 | 219.422 |

108 个分区均通过列序/dtype、完整交易日×固定池网格、主键唯一、无无穷值、值域、rank 范围、value/rank 缺失一致性检查；没有整日常数 rank。

## 可复现验证

- 86 项单元与回归测试全部通过，含本轮新增 8 项；测试文件 `tests/test_quarterly_quality.py`。
- 源公告截断：2018-06-29、2022-09-30、2026-09-18 三个截点，分别比较 2018 年上半年、2022 年 9 月、2026 年 9 月对应区间，12 因子共 36 项比较；value、rank、NaN 全部逐位一致。
- 公告截断比较合计 3,908,520 行。
- 预热：2026-01-01 至 2026-01-16，1100 与 2200 日预热的 12 项比较逐位一致。
- 增量：先算 2026 年上半年，再沿框架 plan/finalize 路径补至 9 月 18 日，与一次性生成的 12 个 2026 分区逐位一致。
- 正式入口再次执行本组增量，按既有10日历日回刷机制重算尾部9个交易日（共228,420行），108 个 parquet SHA256 全部不变；幂等指内容一致，不是零I/O。
- 既有 3703 个产物/状态文件的大小及修改时间保持不变；旧有 342 个计算配方完全一致。
- 按年、单进程生成，数值库线程数 1；本次历史生成约 2.6 分钟，观测 cgroup 峰值约 1.9 GiB（含页缓存）。

完整证据与开发前备份位于 `artifacts/codex_factor_20260921_56292/`：

- `framework_before.tar.gz`：改动前代码/配置/文档备份。
- `upstream_quarterly_probe.json`：原始字段抽查。
- `validation_report.json`、`validation.log`：逐分区检查和截断/预热/增量对照。
- `publication_report.json`、`production_incremental.log`：正式交付与幂等检查。
- `validate_quarterly.py`：本轮可复现验证脚本；内部路径固定到本轮独立验证目录。

## 日常入口与模型接入

```bash
cd /autodl-fs/data/featureengineering
runuser -u claude -- /autodl-fs/data/miniconda3/bin/python main.py run --group quarterly_quality --jobs 1
```

不带组别的既有日常入口也会正常包含这 12 个已注册因子。本轮没有执行全库更新。

模型侧按既有文件接口读取 `data/factors/qf_*/year=YYYY/data.parquet`，并记录因子/训练快照版本；本轮没有替模型侧选择变量、调整收益方向或更新训练快照。

## 已知边界

- 固定股票池既有幸存者偏差和事后 ST 筛选口径保持不变。
- 截断测试只能证明相对于当前源表记录的因果一致性；源表若已经覆盖旧版本或追溯改写公告日，这些测试不能恢复从未保存的历史快照或证明数据商的真实到达时间。
- 单季度现金流和盈利存在季节性、行业差异及极端小分母；同比改善用于减少季节性影响，四季最差值/波动仍须由模型侧检验。现金流比率极端原值保持可见，排名由既有引擎处理。
- 五字段全零规则是保守的数据质量假设；若上游提供明确缺失标志，应以证据更新版本。
- 新因子非空率高不等于收益有效，未进行收益或稳定性承诺。
