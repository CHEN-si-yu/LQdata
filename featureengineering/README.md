# A 股日横断面因子工程总手册

`/autodl-fs/data/featureengineering`（规范路径；`/root/autodl-fs` 是同一目录的符号链接）的现行手册。
最近整理：**2026-09-20 拆分** —— 本文件此前 24255 行 / 1.59 MiB，现按用途拆成 3 份。

**核心手册由以下 3 份 Markdown 组成，各有单一职责；并行分工见 `AGENTS.md`，最新交付见下方链接：**

| 文件 | 职责 | 什么时候读 |
|:--|:--|:--|
| **`README.md`（本文件）** | 现行操作约定、开发契约、运维手册、自动因子字典、自动检查报告 | 冷启动 / 日常操作 / 写因子前 |
| [`HISTORY.md`](HISTORY.md) | 一次性维护记录、专项审计、旧版文档原文、原文档迁移索引 | 只在追溯历史结论时读 |
| [`REFERENCE_lingqi.md`](REFERENCE_lingqi.md) | 灵启因子库（数据商）原文，**冻结参考资料** | 对照参考公式时读 |

发生冲突时，一律以本文件前部的**现行约定**与代码/配置为准；`HISTORY.md` 与
`REFERENCE_lingqi.md` 里的旧日期、旧起点、旧并行度、旧因子数都只代表当时。

★ 本次拆分保留了一个**不能动的约束**：自动因子字典与两个验收块由
`fea/documentation.py` 按 `<!-- FEA:DOC:... -->` 标记**原地改写 `README.md`**，
标记缺失或重复会直接抛 `ValueError`。所以这三块**必须留在本文件**，
不能像历史记录那样搬去 `HISTORY.md` —— 除非同时修改 `fea/documentation.py`。


> **2026-09-21 新增交付**：已新增 12 个 `quarterly_quality` 候选因子；首批交付时注册为 349 个因子 + 5 个标签。旧因子的定义及历史产物保持不变。
> 并行开发请先读 [AGENTS.md](AGENTS.md)；公式、质量检查与使用边界见 [本轮开发报告](FACTOR_DEVELOPMENT_20260921.md)。下方 2026-09-20 体检与字段覆盖表为当时快照。

> **2026-09-21 第二批字段扩展已交付**：新增 **304 个候选因子**；分析字段覆盖增至 **531/681（78.0%）**，当前 **653 因子 + 5 标签**。详见 [交付与验收报告](FIELD_EXPANSION_20260921.md) 及 [逐字段台账](artifacts/experiments/field_expansion_20260921_56292/coverage/FIELD_COVERAGE.csv)。下方旧体检数字为当时快照。

### 导航

- [因子开发交付记录索引](FACTOR_DEVELOPMENT_LOG.md)（批次、验收证据、字段台账与维护事项）
- [★ 因子质量体检报告（2026-09-20）](#audit-20260920) ← **先看这个**
- [上游字段开发覆盖（2026-09-20 盘点）](#upstream-fields) ← 还差哪些字段没用
- [当前操作与开发约定](#current-guide)
- [目录职责与文件管理](#file-management)
- [本次文档拆分记录（2026-09-20）](#doc-split)
- [当前完整因子字典](#factor-catalog)（`main.py docs` 自动生成）
- [因子开发契约（写因子前必读）](#dev-contract)
- [运维手册：硬约束 / 速查 / 架构 / 已知的坑](#dev-handbook)
- [后续自动检查报告](#automatic-reports)
- → [`HISTORY.md`](HISTORY.md) 历史记录与参考资料 · → [`REFERENCE_lingqi.md`](REFERENCE_lingqi.md) 灵启参考原文

<a id="audit-20260920"></a>
## ★ 因子质量体检报告（2026-09-20）

> **只读体检，未改动任何因子、配置或数据。** 全部指标由 `/tmp/fea_audit/` 下的只读脚本算出，
> 唯一被修改的文件就是本文档及其两个分身（`HISTORY.md`、`REFERENCE_lingqi.md`）。
> 计算区间 **2018-01-02 ~ 2026-09-18**（全 9 年、337 个因子 + 5 个标签、固定池 2115 只），
> 未做任何抽样。

### 0. 结论摘要

| 维度 | 结论 |
|:--|:--|
| **覆盖率** | 中位 **99.5%**；302 / 337 个因子 ≥90%；最低 `dragon_tiger_org_net_20` 6.2%（龙虎榜本身是稀疏事件源）。**15 个因子覆盖率逐年后移下滑 >5pp**，根因是上游 `stock_cashflow` 缺年报行（见 P2-1）。 |
| **波动性** | 日截面离散度（IQR 中位）跨度 14 个数量级；13 个因子**日截面 IQR = 0**（大量并列值），但 `rank` 列仍可用。`value↔rank` 一致性 0.9972~1.0000（中位 1.0000）—— **rank 口径干净，没有漂移**。 |
| **与 target 相关性** | 5 日 RankIC 中位 **-0.0057**；\|RankIC\| > 0.02 的有 **139 / 337** 个。最强的一档是**反转/波动类**（全期负号）：`limit_up_count_20` −0.067、`intraday_ma_60d` −0.064、`atr_14_ratio` −0.063；正向最强 `cp_value_momentum_div` +0.062。 |
| **因子间冗余** | **等效独立因子数仅 47.6 / 337**（参与比）。\|ρ\|≥0.90 有 **14 簇 / 37 个因子**；\|ρ\|≥0.70 有 **42 簇 / 158 个因子**，即约 **47%** 的因子在 0.70 水平上已被别人覆盖。 |
| **发现的问题** | **3 个高优先（P1）**、**4 个中优先（P2）**、**8 个低优先（P3）**，见 §6。最重的两条：① 上游资产负债表 5 个字段**只在半年报/年报填充、Q1/Q3 写成 0**，`bs_construction_capital_share` 在 Q1 窗口里有 **30% 的股票恒等于 1.0**；② winsor `[0.01, 0.99]` 会把「≥99% 同值」的因子**整个截面压成常数 rank**，`one_word_limit_down_freq_20` 有 4 个年份 rank 全年只有 1 个取值。 |

### 1. 覆盖率

逐因子、逐年统计 `value` 非 NaN 的格子占比（面板是 T×C 全网格，故分母即当日池内股票数）。

| 分位 | 覆盖率 |
|:--|--:|
| 最低 | 6.2% |
| p5 | 59.3% |
| p25 | 98.5% |
| **中位** | 99.5% |
| p75 | 99.7% |
| p95 | 100.0% |
| 最高 | 100.0% |

- ≥90%：**302** 个 · <90%：35 个 · <50%：6 个 · <10%：1 个
- 覆盖率低的因子**绝大多数是设计使然**（事件稀疏 / 两融标的只占主板六成），不是算错；但要清楚它们在下游是「部分截面可用」。

| 覆盖率最低的 16 个因子 | 覆盖率 | 性质 |
|:--|--:|:--|
| `dragon_tiger_org_net_20` | 6.2% | 龙虎榜（稀疏事件源） |
| `top_list_net_rate_20` | 10.5% | 龙虎榜（稀疏事件源） |
| `seal_float_strength_20` | 17.2% | 打板/封板状态 |
| `seal_turnover_strength_20` | 17.2% | 打板/封板状态 |
| `seal_reopen_pressure_20` | 17.2% | 打板/封板状态 |
| `pegh5` | 45.8% | — |
| `short_interest_volatility_20d` | 56.4% | 两融（标的仅覆盖约六成主板） |
| `margin_repay_deceleration` | 58.8% | 两融（标的仅覆盖约六成主板） |
| `cp_bigflow_margin_20` | 59.0% | 两融（标的仅覆盖约六成主板） |
| `margin_repay_shock` | 59.1% | 两融（标的仅覆盖约六成主板） |
| `short_sell_volume_ratio` | 59.1% | 两融（标的仅覆盖约六成主板） |
| `cp_margin_trend_div` | 59.2% | 两融（标的仅覆盖约六成主板） |
| `margin_velocity` | 59.3% | 两融（标的仅覆盖约六成主板） |
| `margin_leverage_change_20d` | 59.3% | 两融（标的仅覆盖约六成主板） |
| `short_balance_ratio_change_20d` | 59.3% | 两融（标的仅覆盖约六成主板） |
| `margin_chip_cost_gap` | 59.3% | 两融（标的仅覆盖约六成主板） |

**★ 覆盖率随时间下滑的因子（P2）**：以下因子在 2019–2021 覆盖 0.97–0.99，2022 年起掉到 0.83–0.88，
2025 年进一步掉到 0.70–0.86。根因在上游而不是因子代码：

| 因子 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 前2年均值→后2年均值 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `cfp_ttm` | 0.97 | 0.99 | 0.99 | 0.99 | 0.89 | 0.94 | 0.94 | 0.83 | 0.86 | +0.133 |
| `ocf_to_revenue` | 0.96 | 0.98 | 0.98 | 0.98 | 0.88 | 0.92 | 0.93 | 0.83 | 0.86 | +0.124 |
| `ebitda_to_market` | 0.96 | 0.98 | 0.98 | 0.98 | 0.88 | 0.92 | 0.93 | 0.83 | 0.86 | +0.126 |
| `cash_sales_ratio` | 0.96 | 0.98 | 0.98 | 0.98 | 0.88 | 0.92 | 0.93 | 0.83 | 0.86 | +0.124 |
| `cfcr` | — | 0.81 | 0.81 | 0.84 | 0.77 | 0.82 | 0.82 | 0.73 | 0.76 | +0.067 |
| `np_to_salary_yoy` | 0.89 | 0.96 | 0.97 | 0.98 | 0.88 | 0.82 | 0.87 | 0.79 | 0.75 | +0.151 |
| `np_to_fixed_assets_yoy` | 0.87 | 0.82 | 0.79 | 0.77 | 0.78 | 0.76 | 0.73 | 0.72 | 0.68 | +0.146 |
| `cf_net_borrowing_to_assets` | 0.97 | 0.99 | 0.99 | 0.99 | 0.89 | 0.94 | 0.94 | 0.83 | 0.86 | +0.132 |
| `cf_distribution_cash_coverage` | 0.97 | 0.98 | 0.99 | 0.99 | 0.89 | 0.93 | 0.94 | 0.83 | 0.86 | +0.130 |
| `ocf_to_profit` | 0.96 | 0.97 | 0.98 | 0.98 | 0.88 | 0.92 | 0.93 | 0.83 | 0.86 | +0.124 |
| `cf_tax_refund_share` | 0.96 | 0.98 | 0.98 | 0.98 | 0.88 | 0.92 | 0.93 | 0.83 | 0.86 | +0.124 |
| `cf_tax_cash_burden` | 0.96 | 0.98 | 0.98 | 0.98 | 0.88 | 0.92 | 0.93 | 0.83 | 0.86 | +0.124 |
| `cf_purchase_cash_intensity` | 0.96 | 0.98 | 0.98 | 0.98 | 0.88 | 0.92 | 0.93 | 0.83 | 0.86 | +0.124 |
| `cf_borrowing_repayment_ratio` | 0.83 | 0.85 | 0.86 | 0.88 | 0.78 | 0.83 | 0.84 | 0.74 | 0.77 | +0.086 |
| `forecast_type_score` | 0.86 | 0.84 | 0.82 | 0.80 | 0.83 | 0.84 | 0.82 | 0.79 | 0.77 | +0.077 |

（全库共 **15 个**因子「前两年均值 − 后两年均值」下滑超过 5pp；另有 **50 个**同期上升，最典型的是两融类：标的范围逐年扩容，覆盖率从 0.36 升到 0.73。这两类变化都**不是因子代码改动**引起的。）

### 2. 波动性 / 截面分辨率

- `value↔rank` 一致性：全库 **0.9972 ~ 1.0000**（中位 1.0000）。rank 完全由当日 value + 缩尾决定，**没有口径漂移**。
- rank 分布：均值 0.50028（理论 0.5）、标准差中位 0.28867（均匀分布理论 0.28868）—— 截面百分位实现正确。
- 日截面 IQR 中位数 < 1e-6 的因子：**17** 个；= 0 的 13 个。这些是事件/计数类（涨停次数、连续跌停…），截面被并列的 0 支配。
- 日截面 std 的时序 CV 最高 5.42（`bw_capture_asym_20`）—— 离散度本身不稳，做 z-score 类处理时要留意。

**★ rank 截面分辨率（P1）** —— 这是本轮最意外的发现。`fea/panel.py::cs_rank` 先按当日
`winsor=[0.01, 0.99]` 缩尾、再取百分位排名。**当某日 ≥99% 的股票取值相同时，99% 分位就等于那个众数值，
`np.clip` 会把整个截面压成同一个数 → 全部并列 → rank 常数**。`value` 列仍有真实变化，
但**下游只用 rank 列**，等于该日零信息。

全库 3028 个「因子·年」中，有 **6 个因子**出现过整日零信息，其中 **27 个因子·年**的零信息日占比超过一半：

| 因子 | 零信息日占比（受影响年份） | 说明 |
|:--|:--|:--|
| `one_word_limit_down_freq_20` | 2018:80% 2019:100% 2020:92% 2021:100% 2022:100% 2023:100% 2024:100% 2025:92% 2026:88% | ★ 多个年份 rank **全年只有 1 个取值** |
| `consecutive_limit_down` | 2018:93% 2019:95% 2020:95% 2021:93% 2022:91% 2023:99% 2024:92% 2025:96% 2026:89% | 大部分交易日 rank 无区分度 |
| `limit_down_event_5` | 2018:60% 2019:73% 2020:55% 2021:47% 2022:39% 2023:81% 2024:56% 2025:58% 2026:34% | 大部分交易日 rank 无区分度 |
| `consecutive_limit_up` | 2018:69% 2019:61% 2020:38% 2021:12% 2022:19% 2023:70% 2024:42% 2025:9% 2026:11% | 大部分交易日 rank 无区分度 |
| `new_low_60_event` | 2018:2% 2019:20% 2020:9% 2022:8% 2023:2% 2024:31% 2025:5% 2026:3% | 少数极端日 |
| `limit_up_event_5` | 2020:1% | 少数极端日 |

另有 **26 个因子**在 ≥50% 的交易日里 rank 的不同取值数 ≤ 20，对 2115 只股票的截面排序来说**分辨率严重不足**（典型是 `count / window` 型定义：`zero_return_fraction_20` 只有 21 档、`marubozu_ratio_10d` 只有 11 档、`tail_risk_pct_60` 只有 61 档，缩尾后再折半）。

### 3. 与 target（未来收益）的相关性

口径：逐日截面 Pearson（IC）与 Spearman（RankIC），主标签 **`label_ret_5d`**；`main.py eval` 的标签是**内部现算**的（不读落盘的 `label_ret_*`），两者互为交叉校验。

- 5 日 RankIC 分布：min -0.0673 / p5 -0.0547 / **中位 -0.0057** / p95 +0.0298 / max +0.0619
- \|RankIC\| > 0.02：**139 / 337**；> 0.04：56；> 0.06：13

| # | 因子 | 家族 | RankIC | IC | ICIR | ac1 |
|--:|:--|:--|--:|--:|--:|--:|
| 1 | `limit_up_count_20` | event | -0.0673 | -0.0404 | -0.317 | 0.958 |
| 2 | `intraday_ma_60d` | pattern | -0.0643 | -0.0372 | -0.262 | 0.980 |
| 3 | `atr_14_ratio` | technical | -0.0630 | -0.0188 | -0.111 | 0.990 |
| 4 | `turnover_std_20` | liquidity | -0.0621 | -0.0331 | -0.237 | 0.990 |
| 5 | `cp_value_momentum_div` | coupling | +0.0619 | +0.0319 | +0.234 | 0.966 |
| 6 | `limit_up_event_5` | event | -0.0618 | -0.0432 | -0.384 | 0.825 |
| 7 | `amihud_parkinson_ratio` | coupling | +0.0616 | +0.0276 | +0.229 | 0.998 |
| 8 | `vol_of_vol_20` | risk | -0.0614 | -0.0233 | -0.152 | 0.974 |
| 9 | `extreme_move_event` | event | -0.0607 | -0.0393 | -0.315 | 0.833 |
| 10 | `turnover_f_20` | value | -0.0605 | -0.0321 | -0.212 | 0.998 |
| 11 | `id2_parkinson_vol` | intraday | -0.0604 | -0.0286 | -0.210 | 0.539 |
| 12 | `intraday_ma_20d` | pattern | -0.0603 | -0.0394 | -0.287 | 0.937 |
| 13 | `margin_velocity` | margin | -0.0600 | -0.0319 | -0.232 | 0.848 |
| 14 | `idio_vol_60` | risk | -0.0596 | -0.0201 | -0.117 | 0.997 |
| 15 | `idt_rv_daily` | intraday | -0.0584 | -0.0325 | -0.245 | 0.652 |
| 16 | `di_plus_minus_ratio_14` | technical | -0.0581 | -0.0314 | -0.246 | 0.939 |
| 17 | `rel_vol_ind_20d` | sector | -0.0573 | -0.0266 | -0.234 | 0.978 |
| 18 | `momentum_20` | momentum | -0.0554 | -0.0406 | -0.274 | 0.928 |
| 19 | `momentum_60` | momentum | -0.0553 | -0.0353 | -0.222 | 0.972 |
| 20 | `limit_board_streak_mean_60` | event | -0.0546 | -0.0215 | -0.181 | 0.985 |

**★ 这张表要配合 §2 一起读。** 排名靠前的 `limit_up_count_20` / `limit_up_event_5` / `extreme_move_event` / `limit_board_streak_mean_60` 正是 §2 里「rank 分辨率 ≤20 档、零信息日占比高」的同一批事件因子。它们的 RankIC 有很大一部分来自**「今天有没有发生事件」这个二值信息**，而不是连续的强度排序 —— 在下游被当作连续特征使用时，其真实可用信息比 RankIC 数字看起来的要少。

**方向性观察**：Top 20 里 **18 个是负号**，且集中在动量 / 波动 / 换手 / 技术类 —— 全期看 A 股的日横断面是**反转**占优（高动量、高波动、高换手在未来 5 日跑输）。这不是因子缺陷，是市场结构；但意味着**这批因子的符号会随市场状态翻转**（2025–2026 单看也是反转，与全期一致）。跨期稳定性比 IC 均值更值得看。

### 4. 因子间相关性冗余

口径与项目自带 `main.py dedup` 一致（落盘 `rank` 列 → 逐日截面 z-score → 相关），但多算了一套**逐日相关再对交易日取平均**（标准做法，不被年份间波动带偏）。缺失记 0 ⇒ 低覆盖因子的相关被**系统性低估**（只会漏报，不会误报重复）。

- 非对角 \|ρ\|：中位 **0.033**、p90 0.200、p99 0.552、最大 **0.9918**
- **等效独立因子数（参与比）= 47.6 / 337**；解释 90% 方差需 160 个主成分、99% 需 276 个；第一主成分独占 9.6% 方差。

| 阈值 \|ρ\| | 冗余簇数 | 涉及因子 | 每簇留 1 个可删 |
|--:|--:|--:|--:|
| 0.99 | 1 | 2 | 1 |
| 0.95 | 6 | 13 | 7 |
| 0.90 | 14 | 37 | 23 |
| 0.80 | 29 | 94 | 65 |
| 0.70 | 42 | 158 | 116 |

**\|ρ\|≥0.90 的 14 个簇**（★ = 建议保留：覆盖率最高者）：

- **簇 1**（8 个，簇内 \|ρ\| ∈ [0.837, 0.992]）：**`avg_cost_premium`**、`chip_median_distance`、`chip_position`、`chip_resistance_distance`、`chip_support_distance`、`cyqp_average_cost_premium`、`cyqp_price_cost_position`、`winner_rate`
- **簇 2**（4 个，簇内 \|ρ\| ∈ [0.727, 0.940]）：**`chip_concentration`**、`chip_range_normalized`、`cyqp_cost_width_70`、`cyqp_cost_width_90`
- **簇 3**（3 个，簇内 \|ρ\| ∈ [0.898, 0.934]）：**`amihud_daily_5`**、`id2_amihud_intraday_20`、`amihud_parkinson_ratio`
- **簇 4**（2 个，簇内 \|ρ\| ∈ [0.914, 0.914]）：**`bias_20`**、`roc_12`
- **簇 5**（2 个，簇内 \|ρ\| ∈ [0.979, 0.979]）：**`bollinger_width_20`**、`market_cap_concentration_20d`
- **簇 6**（2 个，簇内 \|ρ\| ∈ [0.916, 0.916]）：**`chip_concentration_change_20d`**、`cost_convergence_signal`
- **簇 7**（2 个，簇内 \|ρ\| ∈ [0.938, 0.938]）：**`chip_gini_factor`**、`chip_tail_risk`
- **簇 8**（2 个，簇内 \|ρ\| ∈ [0.919, 0.919]）：**`close_location_20d`**、`momentum_20`
- **簇 9**（2 个，簇内 \|ρ\| ∈ [0.951, 0.951]）：**`downside_upside_vol_60`**、`ret_skew_60`
- **簇 10**（2 个，簇内 \|ρ\| ∈ [0.914, 0.914]）：**`intraday_ret_momentum`**、`idt_overnight_minus_intraday`
- **簇 11**（2 个，簇内 \|ρ\| ∈ [0.944, 0.944]）：**`intraday_ma_20d`**、`intraday_ret_share_20`
- **簇 12**（2 个，簇内 \|ρ\| ∈ [0.933, 0.933]）：**`mf_big_order_ratio`**、`mf_big_small_divergence`
- **簇 13**（2 个，簇内 \|ρ\| ∈ [0.952, 0.952]）：**`momentum_60`**、`sortino_ratio_60`
- **簇 14**（2 个，簇内 \|ρ\| ∈ [0.931, 0.931]）：**`roe_ttm`**、`roe_ttm_lag63d`

**最冗余的 10 对**（逐日口径）：

| \|ρ\| | 因子 A | 因子 B | 年际 σ |
|--:|:--|:--|--:|
| 0.9918 | `avg_cost_premium` | `cyqp_average_cost_premium` | 0.004 |
| 0.9791 | `bollinger_width_20` | `market_cap_concentration_20d` | 0.025 |
| 0.9569 | `chip_position` | `cyqp_price_cost_position` | 0.019 |
| 0.9556 | `chip_median_distance` | `chip_support_distance` | 0.022 |
| 0.9523 | `momentum_60` | `sortino_ratio_60` | 0.015 |
| 0.9514 | `downside_upside_vol_60` | `ret_skew_60` | 0.005 |
| 0.9500 | `avg_cost_premium` | `chip_resistance_distance` | 0.029 |
| 0.9470 | `chip_median_distance` | `chip_resistance_distance` | 0.022 |
| 0.9436 | `intraday_ma_20d` | `intraday_ret_share_20` | 0.007 |
| 0.9419 | `chip_resistance_distance` | `cyqp_average_cost_premium` | 0.032 |

> 说明：`avg_cost_premium` × `cyqp_average_cost_premium`（ρ=0.992）是**同一含义在两个上游表里各做了一遍**；
> `roe_ttm` × `roe_ttm_lag63d`（0.931）是**有意滞后 63 天的副本**；
> 簇 1 的 8 个成员全部来自「筹码成本相对现价的位置」这一个维度。

### 5. `main.py check` 基础体检（通过）

直接跑项目自带入口（只读，不写任何文件）：

```bash
python main.py check        # 342 个注册对象 / 3073 个分区
```

```text
格式统一性检查（列名/列序/dtype/日期格式/主板/rank 值域/主键唯一）
  ✔ 3073 个分区 / 342 个因子的实际列签名完全一致：
    {'trade_date': 'string', 'stock_code': 'string', 'value': 'float32', 'rank': 'float32'}
基础体检通过 ✔（PIT 请另运行 audit-pit）
```

- **0 个问题**：没有缺失分区、没有非主板代码、没有早于起点的行、没有 ±inf、没有 `|value| > 1e8`、没有 NaN 占比 >95%、没有未放行的 qfq 依赖。
- 3073 个分区的列签名**只有 1 种**，与 `fea/store.py::_assert_schema` 的写死口径一致。
- ⚠️ 该命令打印的横幅里仍写「只考虑主板 + 非 ST，**约 3000+ 只**」，与已冻结的 **2115 只**不符（横幅在 `main.py` 的 `BANNER` 常量里，不在本文档内）。
- 注意 `check` 自己声明「基础体检不证明无前视」，**PIT 因果性不在本次体检范围内**（见 P3-6）。

### 6. 问题清单（按优先级）

#### P1 —— 影响下游使用，建议优先处理

**P1-1 · 上游资产负债表有 5 个字段只在半年报/年报填充，Q1/Q3 被写成 0，导致相关因子在两个报告窗口之间分布跳变。**
`stock_balancesheet` 里 Q1/Q3 报告的零值率 vs 半年报：

| 字段 | Q1 (03-31) | H1 (06-30) | Q3 (09-30) | 年报 (12-31) |
|:--|--:|--:|--:|--:|
| `oth_receiv` 其他应收款 | **96.1%** | 3.3% | **95.6%** | 3.1% |
| `oth_payable` 其他应付款 | **94.2%** | 3.2% | **93.6%** | — |
| `lt_payable` 长期应付款 | **100.0%** | 77.5% | **99.9%** | — |
| `cip` 在建工程（非 `cip_total`） | **50.0%** | 15.2% | **50.8%** | — |
| `fix_assets` 固定资产 | **34.0%** | 0.2% | **35.6%** | 0.2% |

资产负债表科目是**时点量**，走 `ctx.point()` 直接取值、没有累计差分保护（现金流量表的累计项
有 `fea/deriv.py::_ttm_at` 的差分兜底，所以 `ebitda_to_market` 用的 `depr_fa_coga_dpba`
虽然 Q1/Q3 也是 0，TTM 求和的差分恰好抵消，**不受影响** —— 这一条已实测确认）。

受影响且**已实测量化**的因子：

| 因子 | 临床表现 |
|:--|:--|
| `bs_construction_capital_share` | `cip_total/(fix_assets+cip_total)`，`fix_assets=0` 时恒等于 **1.0**。2024-06-28（最新报告=Q1）**30.0% 的股票恰好 =1.0**、90 分位 =1.0；2024-09-20（最新报告=H1）**0%**。全年 24.1% 的格子 =1.0。月度中位数在 0.083 ~ 0.184 之间摆动。 |
| `bs_other_receiv_to_assets` | `oth_receiv/total_assets`，Q1/Q3 窗口里 ~95% 的股票被写成 0；一年只有约 2 个月（H1 报告落地后）有真实截面。已逐股对拍确认因子值与上游完全一致 —— **问题在上游，不在因子代码**。 |
| `np_to_fixed_assets_yoy` | `_yoy_ratio(NP, FA, "point")`，`fix_assets=0` → 除零 → NaN。覆盖率随报告期在 **0.66~0.80 与 0.93~0.97 之间摆动**。 |

`oth_payable` / `lt_payable` / `cip` 目前没有任何因子使用，暂时无害。
*建议*：① 请模块① 核对这 5 个字段在 Q1/Q3 报告的抓取；② 在修好之前，
这三个因子的 `point()` 取值应改为「回退到最近一个非零值」或把 Q1/Q3 的 0 视为缺失，
否则它们在一年中有一半时间的截面是假的。
★ 注意：手册原有的「2026 年『其他应收』高零值比例已与源数据核对，保持稀疏标记」这句
**只核对了「因子 == 源数据」，没有发现「源数据本身在 Q1/Q3 是 0」**，所以按现状保留是错的。

**P1-2 · winsor 缩尾会把「≥99% 同值」的因子整个截面压成常数 rank。**
`fea/panel.py:188-193` 先 `np.clip(x, q01, q99)` 再排名。事件类因子（涨停/跌停/一字板/极端波动）
在多数交易日有 ≥99% 的股票取同一个值（通常是 0），此时 `q99 == 众数`，整列被 clip 成常数。
实测：`one_word_limit_down_freq_20` 在 **2019 / 2021 / 2022 / 2023 四个年份 rank 全年只有 1 个取值**
（0.500237），`consecutive_limit_down` 2023 年 99.2% 的交易日 rank 无区分度。
`value` 列是对的，坏的是下游唯一使用的 `rank` 列。
*建议*：对「零膨胀」因子改用不缩尾的排名，至少在 `cs_rank` 里加一条
「缩尾后截面不同取值数 ≤ N 就回退到未缩尾排名」的兜底；
`eval.py` 已有的 `⚠LOWCARD/⚠SPARSE` 标记可以直接复用来定位受影响因子。

**P1-3 · 约 1/3 的因子在 |ρ|≥0.70 上已被他人覆盖，等效独立因子数只有 47.6 / 337。**
最大的一簇是 8 个「筹码成本相对现价位置」因子（簇内 |ρ| 最高 0.992）。
它们在下游的日横断面回归里只会互相抢显著性、放大过拟合。
*建议*：按 `state/dedup` 的思路做一次簇内保留，或在下游改用「每簇取第一主成分」；
不需要删因子源文件，删的是喂给模型的列。

> ★ **2026-09-22 已出全量审计**（656 因子 × 2018~2026、|ρ|≥0.95）：**37 簇 / 58 个候选删除** ——
> 交付物 `artifacts/audits/dedup_20260922/`（`REPORT.md` + `representatives.json`），
> 机器报告 `state/dedup/report.json`（09-19 的 337×2026 旧版归档在
> `artifacts/audits/dedup_20260919_legacy/`）。★ 本报告上表的 47.6/337 是 **2026-09-20 的
> 状态**（337 因子、|ρ|≥0.70 口径），与这次 0.95 口径的 37 簇不是同一件事，别混引。
>
> ★★ **同日（2026-09-22 下午）已按该清单执行真删**：58 个候选里**删 57 个**
> （因子 656 → **599**），保留 `momentum_60`（两个存活耦合因子的父依赖，删了子因子会静默变全 NaN）。
> 删除前全量备份 `artifacts/backups/prune_20260922/`；删→代表对照表
> `artifacts/audits/dedup_20260922/replacement_map.tsv`；记录见本手册末的「因子分诊报告」区
> 与 `HISTORY.md` §2026-09-22 的「同日追加」小节。**模型侧需同步列清单**（通知在 `FACTOR_REQUESTS.md`）。

#### P2 —— 影响因子质量，但不阻塞使用

**P2-1 · 上游 `stock_cashflow` 缺年报行，导致现金流类 TTM 因子覆盖率阶梯式下滑。**
`stock_cashflow` 年报行（`end_date` 以 `12-31` 结尾）在冻结池内的覆盖：
2019 92.7% → **2021 76.2%** → 2022 99.2% → **2024 84.8%** → 2025 97.1%。
TTM 需要 4 个**连续**季度，缺一个年报就整段断档，于是 `cfp_ttm` / `ocf_to_revenue` /
`ebitda_to_market` / `cash_sales_ratio` / `cfcr` 等覆盖率从 0.98 掉到 0.83–0.86，
2025 年进一步掉到 0.70–0.86。（`stock_balancesheet` 年报覆盖 98.8–99.4% 稳定；
`stock_income` 2020 年 85.7% 偏低但其余年份 98–99%，明显好于现金流表。）
已抽查 `000333.SZ` / `000008.SZ`：**2021-12-31 年报行整行缺失**，而 2020/2022 年正常 ——
是抓取缺口而非源站没有。这是**模块① 的数据完整性缺口**，不是因子代码问题。
*建议*：核对 `datadownload` 侧 2021 年报与 2024 年报的抓取是否漏页。

**P2-2 · 26 个「count / window」型因子的 rank 分辨率 ≤20 档。**
`zero_return_fraction_20`（21 档）、`marubozu_ratio_10d`（11 档）、`tail_risk_pct_60`（61 档）等，
在 2115 只股票的截面上排序分辨率严重不足，缩尾后进一步折半。
*建议*：这类因子改用连续化的分母（如用日收益的分布而非固定阈值计数），或在 `note` 里显式声明分辨率上限。

**P2-3 · 16 个因子有 >50% 的格子 `value` 恰好为 0，其中 13 个 >70%，但被当连续因子喂给模型。**
零占比最高：`consecutive_limit_down` 99.5%、`one_word_limit_down_freq_20` 98.9%、
`consecutive_limit_up` 98.6%、`limit_down_event_5` 97.8%。
`eval.py` 已用 `⚠SPARSE` 标出，但标记不等于处理。
*建议*：下游对零膨胀因子改用「是否非零」的二值特征 + 非零部分的强度，两段建模。

**P2-4 · 上游延迟口径与代码注释不一致，滞后守卫的实际覆盖面比记录的窄。**
`fea/spec.py:131-139` 的注释仍称 `stock_dragon_tiger`/`stock_top_list`/`stock_report_rc`/`dc_daily`
晚 1 个交易日，而 `datadownload/conf/frequency.yaml`（2026-09-19）已全部置 0；
注册守卫实际只剩 `stock_margin_detail: 1` 一张表（影响 `margin` 家族 17 个因子）。
同一问题也出现在拆分前的文档里（本次已随拆分修掉）。
*建议*：要么更新注释与 `frequency.yaml` 对齐，要么确认这 4 张表确实不需要滞后 —— 二者必须有一个是准的。

#### P3 —— 一致性 / 可维护性

**P3-1 · 2 个文件共 3 处绕过价格层直读上游。** `factors/sector.py:194-196` 与
`factors/chips2.py:225` 用 `ctx.dataset("stock_daily")` 直读**未复权** `close`，
绕开了 `fea/prices.py` 的「非交易日脏行丢弃」与「|r|>0.60 置 NaN」两道清洗
（`sector.py` 自行 `close×adj_factor` 后算行业相关性矩阵，脏 `adj_factor` 可造出假暴跌）。
全库仅此 2 处，其余 199 个依赖 `stock_daily` 的因子都走价格层。

**P3-2 · `ctx.ret_clean()` 全库 0 处调用。** `fea/context.py:83` 提供了它，
且 `context.py:69-80` 明确说裸 `ctx.ret` 对波动率类是错的（停牌日贡献人造 0），
但各家族各自重写 `np.where(ctx.traded(), ctx.ret(1), np.nan)`。API 存在却没人用，口径靠人抄。

**P3-3 · 标签与因子的 universe 掩码语义不同且无注释。**
`fea/engine.py:545` 标签路径 `val = sub`（不套掩码），`:548` 因子路径 `np.where(submask, sub, np.nan)`。
冻结池下实际差异极小，但两条路径语义不同、无说明。

**P3-4 · `fea/prices.py:93` 死过滤**：`[c for c in cols if c in (...) or True]` 条件恒真，等于没过滤。
**`fea/spec.py:157-160` 死代码**：`if not isinstance(fn, ...): pass` 整块无效果。

**P3-5 · `fea/delay.py` 只遍历「已声明」的表**（`delay.py:147` 的 `for name, d in declared.items()`），
所以只出现在 `delay_history.json` 里的 5 张表永远不会被纳入滞后判定。碰巧这 5 张观测都是 0，暂无实害。

**P3-6 · 两个「声明了却从没接进去」的数据集常量。**
`factors/event2.py:79` 的 `PS = "stock_pledge_stat"`（股权质押）与 `:80` 的
`SP = "stock_suspension"`（停牌事件）**在全文从未被引用**，两张表也从未被任何因子读取。
停牌信息目前只靠「`stock_daily` 停牌日没有行」间接表达。
看代码的人容易误以为这两块已经被覆盖。**建议：要么用起来，要么删掉常量。**
（同批还发现 `index_ths_daily` / `dc_daily` / `tdx_minute` / `dc_blocks` 四张表
从没被读过 —— 完整清单见[上游字段开发覆盖](#upstream-fields)。）

**P3-7 · PIT 动态审计的证据面很薄**：默认 `--sample 2` 只抽「中点 + 最后一天」两个交易日；
最近落盘的 `state/pit_audit/20260919_185851/report.json` 只有 1 个采样日 × 2 个因子。
`check` 自己也声明「基础体检不证明无前视」。工具齐备，**证据稀疏**。
另：`fea/pit_audit.py:76-79` 在只选中标签因子时 `return 1`，会被 `main.py` 计入 problems，
让 `scripts/post_steps.sh` 误报退出码非 0。

**P3-8 · 自动因子字典里有 71 处表格断行。**
`main.py docs` 生成的字典把**多行 Python 源码**直接塞进表格单元，换行后行首没有 `|`，
表格被截断；更麻烦的是其中 **39 行恰好以 `# ` 开头**（Python 注释），
会被任何 `grep '^# '` 当成标题 —— 用它找标题会多出 39 个假 H1。
集中在自动字典区（本文的 `FEA:DOC:factor-catalog` 块）。
*建议*：改 `fea/documentation.py` 的字典渲染，把公式写进 ```python 围栏或转义换行；
在那之前**不要用 `grep '^# '` 在这个仓库的 Markdown 里找标题**。

### 7. 方法与可复现性


| 项 | 做法 |
|:--|:--|
| 数据源 | 只读 `data/factors/<name>/year=YYYY/data.parquet`（3073 个分区、7.4 GB）。**未运行任何因子计算。** |
| 区间 | 2018-01-02 ~ 2026-09-18（全量，未抽样，未按用户允许的「可缩到 2025–2026」缩窗） |
| 覆盖率/波动性 | 自写 `profile_values.py`，16 进程按因子并行；只读 parquet，`value` 是**未缩尾**原始值（`fea/engine.py:548`），故波动性同时报了 IQR 与 std |
| rank 分辨率 | 自写 `rank_cardinality.py`，统计每日「不同 rank 取值个数」 |
| IC / RankIC | **直接用项目自带 `main.py eval`**（`--years` 分 3 片并行、`--out` 指向 `/tmp`，不写 `state/`），标签由 `eval.py` 内部用 `PriceLayer` 现算 |
| 冗余 | 自写 `redundancy.py`，口径与 `fea/dedup.py` 一致（rank → 逐日 z-score → 相关，缺失记 0），额外给「逐日相关对交易日取均值」与特征谱 |
| 资源 | 峰值匿名内存 < 14 GiB（上限 60 GiB），未与其它重活并发；`main.py eval` 3 片并行各占 1 核 |
| 未做 | 未改因子公式、未重算因子、未动 `state/`、未删数据、未运行 `audit-pit` 动态复算（其报告会写 `state/`）。**本报告不替代 PIT 结论。** |
| 原始结果 | `/tmp/fea_audit/*.json`（`profile_values` / `rank_cardinality` / `redundancy_final` / `eval_A|B|C/summary.json`）。⚠️ `/tmp` 不持久，如需长期留存请复制到 `artifacts/audits/`。 |

复现命令（**全部只读**，脚本在 `/tmp/fea_audit/`）：

```bash
PY=/autodl-fs/data/miniconda3/bin/python
cd /tmp/fea_audit

# ① 覆盖率 / 波动性 / value↔rank 一致性（16 进程，约 3 分钟）
$PY profile_values.py                       # → profile_values.json

# ② rank 截面分辨率（16 进程，约 3 分钟）
$PY rank_cardinality.py                     # → rank_cardinality.json

# ③ IC / RankIC：用项目自带入口，按年份分 3 片并行、结果写到 /tmp（不碰 state/，约 25 分钟）
cd /autodl-fs/data/featureengineering
$PY main.py eval --years 2018 2021 --out /tmp/fea_audit/eval_A &
$PY main.py eval --years 2022 2024 --out /tmp/fea_audit/eval_B &
$PY main.py eval --years 2025 2026 --out /tmp/fea_audit/eval_C &

# ④ 因子间相关性冗余（9 进程按年并行，约 10 分钟）
cd /tmp/fea_audit
$PY redundancy.py && $PY redundancy_report.py   # → redundancy.json → redundancy_final.json

# ⑤ 合成这份报告
$PY make_report.py                          # → AUDIT_SECTION.md

# ⑥ 项目自带基础体检（只读，约 40 分钟）
cd /autodl-fs/data/featureengineering && $PY main.py check
```

**上面这些体检命令没有写任何数据或状态文件**：`eval` 的 `--out`、以及其余脚本的全部输出都
指向 `/tmp/fea_audit/`；`state/eval`、`state/dedup`、`state/pit_audit` 保持体检前的原样
（`state/dedup/report.json` 仍是 2026-09-19 23:21 那次**只跑 2026 年**的旧报告，
`state/eval/summary.json` 是 2026-09-18 的旧结果，**都不要拿它们当本次结论**）。

**唯一的例外**：`main.py` 的日志配置会把**每次运行**（包括只读的 `eval` / `check`）追加到
`logs/factors.log`。本次体检在那里新增了 **31 行 INFO/WARNING**（都是价格层加载信息，
例如「价格层：stock_daily 1536311 行」「stock_adj_factor 有 2 行落在非交易日，已丢弃」）。
这是运行项目自带入口的必然副作用，**没有改任何代码、配置、`data/`、`state/` 或 `conf/`**
（已用 `find -newermt` 逐目录核过）。

<a id="upstream-fields"></a>
## 上游字段开发覆盖（2026-09-20 盘点）

回答两个问题：**上游一共给了多少字段**、**因子工程用掉了多少**。

> **这份表会过期。** 它是 2026-09-20 的快照；每加一批因子都应该重跑一次盘点。
> 数字是**实测**出来的，不是读声明：给全部 342 个因子跑了一次 instrumented 沙箱（`pd.read_parquet` 钩子记录每一次读取），覆盖一个完整月窗口，
> 再叠加价格层（`fea/prices.py:45-47`）与派生层（`fea/intraday.py:75-88`、
> `fea/chips.py:94,218`）源码里写死的列。所以「未开发」= **真的没有任何因子读过**。

### 总览

| 项 | 数量 |
|:--|--:|
| 上游数据集 | 32 个（不含 `stock_daily_dump` 原始落盘） |
| 总列数 | 772 |
| 主键 / 标识列（`stock_code`、`ann_date`…） | 不计入 |
| **分析字段** | **681** |
| **已开发** | **205（30.1%）** |
| **未开发** | **476（69.9%）** |

另有派生层二次加工的 59 个特征（日内 36 + 筹码 23）来自 `stock_history_5min` 与 `stock_cyq_chips`，它们是**加工产物**，不计入上游字段。

### 各数据集明细

按未开发数降序。`开发率` = 已开发 / 分析字段。

| 数据集 | 分析字段 | 已开发 | 未开发 | 开发率 |
|:--|--:|--:|--:|--:|
| `stock_financial_indicator` | 163 | 26 | 137 | 16% |
| `stock_balancesheet` | 144 | 38 | 106 | 26% |
| `stock_cashflow` | 89 | 21 | 68 | 24% |
| `stock_income` | 77 | 20 | 57 | 26% |
| `stock_limit_list` | 15 | 1 | 14 | 7% |
| `index_ths_daily` ⚠ | 11 | 0 | 11 | 0% |
| `dc_daily` ⚠ | 10 | 0 | 10 | 0% |
| `stock_limit_up` | 14 | 4 | 10 | 29% |
| `stock_top_list` | 11 | 2 | 9 | 18% |
| `tdx_minute` ⚠ | 9 | 0 | 9 | 0% |
| `index_daily` | 9 | 1 | 8 | 11% |
| `stock_finance` | 17 | 10 | 7 | 59% |
| `stock_list` ⚠ | 7 | 0 | 7 | 0% |
| `stock_pledge_stat` ⚠ | 5 | 0 | 5 | 0% |
| `tdx_daily` | 6 | 1 | 5 | 17% |
| `stock_dragon_tiger` | 7 | 3 | 4 | 43% |
| `stock_forecast` | 8 | 5 | 3 | 62% |
| `stock_suspension` ⚠ | 3 | 0 | 3 | 0% |
| `dc_blocks` ⚠ | 2 | 0 | 2 | 0% |
| `stock_st_info` ⚠ | 1 | 0 | 1 | 0% |
| `basic_calendar` | 1 | 1 | 0 | 100% |
| `stock_adj_factor` | 1 | 1 | 0 | 100% |
| `stock_adj_factor_changes` | 0 | 0 | 0 | 100% |
| `stock_cyq_chips` | 2 | 2 | 0 | 100% |
| `stock_cyq_perf` | 9 | 9 | 0 | 100% |
| `stock_daily` | 11 | 11 | 0 | 100% |
| `stock_history_5min` | 6 | 6 | 0 | 100% |
| `stock_holder_number` | 1 | 1 | 0 | 100% |
| `stock_main_fund_flow` | 18 | 18 | 0 | 100% |
| `stock_margin_detail` | 7 | 7 | 0 | 100% |
| `stock_market_distribution_history` | 16 | 16 | 0 | 100% |
| `tdx_blocks` | 1 | 1 | 0 | 100% |

⚠ = 分析字段一个都没用上。

### 完全未开发的数据集

| 数据集 | 未用分析字段 | 实情 |
|:--|--:|:--|
| `dc_blocks` | 2 | 东财板块成分。**代码里 0 处引用。** |
| `dc_daily` | 10 | 东方财富板块日线。同上，只在 `delay.py` / `spec.py` 注释里出现。 |
| `index_ths_daily` | 11 | 同花顺行业指数日线。只在注释与 `fea/delay.py` 的滞后名单里出现，**没有任何因子读过它**。 |
| `stock_list` | 7 | 只用了 `stock_code` / `list_date` / `delist_date` 做股票池筛选，7 个分析字段（名称、行业、地域等）全未用。 |
| `stock_pledge_stat` | 5 | ★ `factors/event2.py:79` 定义了常量 `PS = "stock_pledge_stat"`，**但该常量在全文从未被引用** —— 声明了没接进去。 |
| `stock_st_info` | 1 | 只用了 `stock_code` / `trade_date` 做「是否 ST」筛选，分析字段未用。 |
| `stock_suspension` | 3 | ★ `factors/event2.py:80` 定义了常量 `SP = "stock_suspension"`，**同样从未被引用**。停牌信息目前只靠 `stock_daily` 停牌日无行来间接表达。 |
| `tdx_minute` | 9 | 通达信分钟线。`fea/engine.py:698` 把它列进「派生层年份范围规划」，但 `fea/intraday.py:75` 实际只读 `stock_history_5min` ⇒ **它从没被读过**，那行只是把年份窗口算大了一点。 |

合计 **48** 个分析字段挂在从没被读过的表上。其中 `stock_pledge_stat` 与 `stock_suspension` 是**声明了常量却没接进去**（死常量），看代码的人容易误以为股权质押 / 停牌已经被覆盖 —— 建议要么用起来，要么删掉常量。

### 最大的开发空间，以及一块「白捡」的增量

未开发字段的 **77% 集中在财务四张表**：

| 数据集 | 未开发 | 主要是什么 |
|:--|--:|:--|
| `stock_financial_indicator` | 137 | 比率 / 每股 / 结构项，见下 |
| `stock_balancesheet` | 106 | 资产负债表其余科目（应收/应付明细、其他流动与非流动项等） |
| `stock_cashflow` | 68 | 现金流量表其余科目（投资/筹资活动明细、付现明细） |
| `stock_income` | 57 | 利润表其余科目（费用明细、其他收益项） |

★ **`stock_financial_indicator` 的 137 个未用字段里，有 33 个 `q_*` 前缀字段**（`q_roe`、`q_netprofit_margin`、`q_gsprofit_margin`、`q_op_to_gr`、`q_dt_eps`…）。查上游 API 文档，这些**明确标注为「单季度」口径**，因此**不带累计 YTD 的锯齿轮，本来就可以当日频值直接用**。

但 `fea/deriv.py` 的 `IND_SAFE` 白名单只放了 15 个时点比率 + 11 个同比，**没有放这些单季度字段**；而 `ctx.ind()` 对白名单外的字段会**直接抛 `KeyError`**（`fea/context.py:160-170`）。所以这 33 个是**框架限制、不是数据质量问题** —— 扩一行白名单（`fea/deriv.py` 的 `IND_DIRECT`）就能解锁，是当前性价比最高的一块增量。

（另外 100 个「其余」确实多数是**累计 YTD** 的比率/每股/结构项，`IND_SAFE` 挡住它们是对的：当日频用会得到跨季锯齿、振幅约 4 倍，而且**不会报错**。要它们就得从原始三表自己算 TTM。）

### 未开发字段全清单

按数据集分组，只列**分析字段**（主键/标识列不计）。清单较长，折叠在此。

<details><summary>展开：476 个未开发字段的完整清单</summary>

**`stock_financial_indicator`** —— 未开发 137 / 163（已开发：`current_ratio`, `quick_ratio`, `cash_ratio`, `debt_to_assets`, `assets_to_eqt`, `ca_to_assets`, `nca_to_assets`, `tbassets_to_totalassets`, `int_to_talcap`, `currentdebt_to_debt`, `longdeb_to_debt`, `debt_to_eqt`, `tangibleasset_to_debt`, `ebitda_to_debt`, `turn_days`, `op_yoy`, `ebt_yoy`, `netprofit_yoy`, `dt_netprofit_yoy`, `ocf_yoy`, `roe_yoy`, `bps_yoy`, `assets_yoy`, `eqt_yoy`, `tr_yoy`, `or_yoy`）

- 未开发：`eps`, `dt_eps`, `total_revenue_ps`, `revenue_ps`, `capital_rese_ps`, `surplus_rese_ps`, `undist_profit_ps`, `extra_item`, `profit_dedt`, `gross_margin`, `invturn_days`, `arturn_days`, `inv_turn`, `ar_turn`, `ca_turn`, `fa_turn`, `assets_turn`, `op_income`, `valuechange_income`, `interst_income`, `daa`, `ebit`, `ebitda`, `fcff`, `fcfe`, `current_exint`, `noncurrent_exint`, `interestdebt`, `netdebt`, `tangible_asset`, `working_capital`, `networking_capital`, `invest_capital`, `retained_earnings`, `diluted2_eps`, `bps`, `ocfps`, `retainedps`, `cfps`, `ebit_ps`, `fcff_ps`, `fcfe_ps`, `netprofit_margin`, `grossprofit_margin`, `cogs_of_sales`, `expense_of_sales`, `profit_to_gr`, `saleexp_to_gr`, `adminexp_of_gr`, `finaexp_of_gr`, `impai_ttm`, `gc_of_gr`, `op_of_gr`, `ebit_of_gr`, `roe`, `roe_waa`, `roe_dt`, `roa`, `npta`, `roic`, `roe_yearly`, `roa2_yearly`, `roe_avg`, `opincome_of_ebt`, `investincome_of_ebt`, `n_op_profit_of_ebt`, `tax_to_ebt`, `dtprofit_to_profit`, `salescash_to_or`, `ocf_to_or`, `ocf_to_opincome`, `capitalized_to_da`, `dp_assets_to_eqt`, `eqt_to_talcapital`, `ocf_to_shortdebt`, `eqt_to_debt`, `eqt_to_interestdebt`, `tangasset_to_intdebt`, `tangibleasset_to_netdebt`, `ocf_to_debt`, `ocf_to_interestdebt`, `ocf_to_netdebt`, `ebit_to_interest`, `longdebt_to_workingcapital`, `roa_yearly`, `roa_dp`, `fixed_assets`, `profit_prefin_exp`, `non_op_profit`, `op_to_ebt`, `nop_to_ebt`, `ocf_to_profit`, `cash_to_liqdebt`, `cash_to_liqdebt_withinterest`, `op_to_liqdebt`, `op_to_debt`, `roic_yearly`, `total_fa_trun`, `profit_to_op`, `q_opincome`, `q_investincome`, `q_dtprofit`, `q_eps`, `q_netprofit_margin`, `q_gsprofit_margin`, `q_exp_to_sales`, `q_profit_to_gr`, `q_saleexp_to_gr`, `q_adminexp_to_gr`, `q_finaexp_to_gr`, `q_impair_to_gr_ttm`, `q_gc_to_gr`, `q_op_to_gr`, `q_roe`, `q_dt_roe`, `q_npta`, `q_opincome_to_ebt`, `q_investincome_to_ebt`, `q_dtprofit_to_profit`, `q_salescash_to_or`, `q_ocf_to_sales`, `q_ocf_to_or`, `basic_eps_yoy`, `dt_eps_yoy`, `cfps_yoy`, `q_gr_yoy`, `q_gr_qoq`, `q_sales_yoy`, `q_sales_qoq`, `q_op_yoy`, `q_op_qoq`, `q_profit_yoy`, `q_profit_qoq`, `q_netprofit_yoy`, `q_netprofit_qoq`, `equity_yoy`, `rd_exp`

**`stock_balancesheet`** —— 未开发 106 / 144（已开发：`total_share`, `money_cap`, `trad_asset`, `notes_receiv`, `accounts_receiv`, `oth_receiv`, `prepayment`, `inventories`, `total_cur_assets`, `fix_assets`, `cip`, `intan_assets`, `r_and_d`, `goodwill`, `defer_tax_assets`, `total_nca`, `total_assets`, `lt_borr`, `st_borr`, `notes_payable`, `acct_payable`, `adv_receipts`, `non_cur_liab_due_1y`, `total_cur_liab`, `bond_payable`, `defer_tax_liab`, `defer_inc_non_cur_liab`, `total_ncl`, `total_liab`, `treasury_share`, `minority_int`, `total_hldr_eqy_exc_min_int`, `total_hldr_eqy_inc_min_int`, `contract_assets`, `contract_liab`, `accounts_receiv_bill`, `accounts_pay`, `cip_total`）

- 未开发：`cap_rese`, `undistr_porfit`, `surplus_rese`, `special_rese`, `div_receiv`, `int_receiv`, `amor_exp`, `nca_within_1y`, `sett_rsrv`, `loanto_oth_bank_fi`, `premium_receiv`, `reinsur_receiv`, `reinsur_res_receiv`, `pur_resale_fa`, `oth_cur_assets`, `fa_avail_for_sale`, `htm_invest`, `lt_eqt_invest`, `invest_real_estate`, `time_deposits`, `oth_assets`, `lt_rec`, `const_materials`, `fixed_assets_disp`, `produc_bio_assets`, `oil_and_gas_assets`, `lt_amor_exp`, `decr_in_disbur`, `oth_nca`, `cash_reser_cb`, `depos_in_oth_bfi`, `prec_metals`, `deriv_assets`, `rr_reins_une_prem`, `rr_reins_outstd_cla`, `rr_reins_lins_liab`, `rr_reins_lthins_liab`, `refund_depos`, `ph_pledge_loans`, `refund_cap_depos`, `indep_acct_assets`, `client_depos`, `client_prov`, `transac_seat_fee`, `invest_as_receiv`, `cb_borr`, `depos_ib_deposits`, `loan_oth_bank`, `trading_fl`, `sold_for_repur_fa`, `comm_payable`, `payroll_payable`, `taxes_payable`, `int_payable`, `div_payable`, `oth_payable`, `acc_exp`, `deferred_inc`, `st_bonds_payable`, `payable_to_reinsurer`, `rsrv_insur_cont`, `acting_trading_sec`, `acting_uw_sec`, `oth_cur_liab`, `lt_payable`, `specific_payables`, `estimated_liab`, `oth_ncl`, `depos_oth_bfi`, `deriv_liab`, `depos`, `agency_bus_liab`, `oth_liab`, `prem_receiv_adva`, `depos_received`, `ph_invest`, `reser_une_prem`, `reser_outstd_claims`, `reser_lins_liab`, `reser_lthins_liab`, `indept_acc_liab`, `pledge_borr`, `indem_payable`, `policy_div_payable`, `ordin_risk_reser`, `forex_differ`, `invest_loss_unconf`, `total_liab_hldr_eqy`, `lt_payroll_payable`, `oth_comp_income`, `oth_eqt_tools`, `oth_eqt_tools_p_shr`, `lending_funds`, `acc_receivable`, `st_fin_payable`, `payables`, `hfs_assets`, `hfs_sales`, `cost_fin_assets`, `fair_value_fin_assets`, `oth_rcv_total`, `fix_assets_total`, `oth_pay_total`, `long_pay_total`, `debt_invest`, `oth_debt_invest`

**`stock_cashflow`** —— 未开发 68 / 89（已开发：`c_fr_sale_sg`, `recp_tax_rends`, `c_paid_goods_s`, `c_paid_to_for_empl`, `c_paid_for_taxes`, `n_cashflow_act`, `c_pay_acq_const_fiolta`, `n_cashflow_inv_act`, `c_recp_borrow`, `free_cashflow`, `c_prepay_amt_borr`, `c_pay_dist_dpcp_int_exp`, `n_cash_flows_fnc_act`, `n_incr_cash_cash_equ`, `prov_depr_assets`, `depr_fa_coga_dpba`, `amort_intang_assets`, `lt_amort_deferred_exp`, `invest_loss`, `decr_inventories`, `credit_impa_loss`）

- 未开发：`net_profit`, `finan_exp`, `n_depos_incr_fi`, `n_incr_loans_cb`, `n_inc_borr_oth_fi`, `prem_fr_orig_contr`, `n_incr_insured_dep`, `n_reinsur_prem`, `n_incr_disp_tfa`, `ifc_cash_incr`, `n_incr_disp_faas`, `n_incr_loans_oth_bank`, `n_cap_incr_repur`, `c_fr_oth_operate_a`, `c_inf_fr_operate_a`, `n_incr_clt_loan_adv`, `n_incr_dep_cbob`, `c_pay_claims_orig_inco`, `pay_handling_chrg`, `pay_comm_insur_plcy`, `oth_cash_pay_oper_act`, `st_cash_out_act`, `oth_recp_ral_inv_act`, `c_disp_withdrwl_invest`, `c_recp_return_invest`, `n_recp_disp_fiolta`, `n_recp_disp_sobu`, `stot_inflows_inv_act`, `c_paid_invest`, `n_disp_subs_oth_biz`, `oth_pay_ral_inv_act`, `n_incr_pledge_loan`, `stot_out_inv_act`, `proc_issue_bonds`, `oth_cash_recp_ral_fnc_act`, `stot_cash_in_fnc_act`, `incl_dvd_profit_paid_sc_ms`, `oth_cashpay_ral_fnc_act`, `stot_cashout_fnc_act`, `eff_fx_flu_cash`, `c_cash_equ_beg_period`, `c_cash_equ_end_period`, `c_recp_cap_contrib`, `incl_cash_rec_saims`, `uncon_invest_loss`, `decr_deferred_exp`, `incr_acc_exp`, `loss_disp_fiolta`, `loss_scr_fa`, `loss_fv_chg`, `decr_def_inc_tax_assets`, `incr_def_inc_tax_liab`, `decr_oper_payable`, `incr_oper_payable`, `others`, `im_net_cashflow_oper_act`, `conv_debt_into_cap`, `conv_copbonds_due_within_1y`, `fa_fnc_leases`, `im_n_incr_cash_equ`, `net_dism_capital_add`, `net_cash_rece_sec`, `use_right_asset_dep`, `oth_loss_asset`, `end_bal_cash`, `beg_bal_cash`, `end_bal_cash_equ`, `beg_bal_cash_equ`

**`stock_income`** —— 未开发 57 / 77（已开发：`revenue`, `fv_value_chg_gain`, `invest_income`, `total_cogs`, `oper_cost`, `sell_exp`, `admin_exp`, `fin_exp`, `assets_impair_loss`, `operate_profit`, `non_oper_income`, `non_oper_exp`, `total_profit`, `income_tax`, `n_income`, `n_income_attr_p`, `minority_gain`, `ebit`, `rd_exp`, `fin_exp_int_exp`）

- 未开发：`basic_eps`, `diluted_eps`, `total_revenue`, `int_income`, `prem_earned`, `comm_income`, `n_commis_income`, `n_oth_income`, `n_oth_b_income`, `prem_income`, `out_prem`, `une_prem_reser`, `reins_income`, `n_sec_tb_income`, `n_sec_uw_income`, `n_asset_mg_income`, `oth_b_income`, `ass_invest_income`, `forex_gain`, `int_exp`, `comm_exp`, `biz_tax_surchg`, `prem_refund`, `compens_payout`, `reser_insur_liab`, `div_payt`, `reins_exp`, `oper_exp`, `compens_payout_refu`, `insur_reser_refu`, `reins_cost_refund`, `other_bus_cost`, `nca_disploss`, `oth_compr_income`, `t_compr_income`, `compr_inc_attr_p`, `compr_inc_attr_m_s`, `ebitda`, `insurance_exp`, `undist_profit`, `distable_profit`, `fin_exp_int_inc`, `transfer_surplus_rese`, `transfer_housing_imprest`, `transfer_oth`, `adj_lossgain`, `withdra_legal_surplus`, `withdra_legal_pubfund`, `withdra_biz_devfund`, `withdra_rese_fund`, `withdra_oth_ersu`, `workers_welfare`, `distr_profit_shrhder`, `prfshare_payable_dvd`, `comshare_payable_dvd`, `capit_comstock_div`, `continued_net_profit`

**`stock_limit_list`** —— 未开发 14 / 15（已开发：`limit`）

- 未开发：`industry`, `close`, `pct_chg`, `amount`, `limit_amount`, `float_mv`, `total_mv`, `turnover_ratio`, `fd_amount`, `first_time`, `last_time`, `open_times`, `up_stat`, `limit_times`

**`index_ths_daily`** —— 未开发 11 / 11（已开发：无）

- 未开发：`ths_code`, `open`, `high`, `low`, `close`, `pre_close`, `avg_price`, `change`, `pct_change`, `vol`, `turnover_rate`

**`dc_daily`** —— 未开发 10 / 10（已开发：无）

- 未开发：`open`, `high`, `low`, `close`, `change`, `pct_change`, `vol`, `amount`, `swing`, `turnover_rate`

**`stock_limit_up`** —— 未开发 10 / 14（已开发：`consecutive_days`, `sealed_turnover_ratio`, `sealed_flow_ratio`, `open_count`）

- 未开发：`price`, `change_percent`, `first_limit_time`, `final_limit_time`, `sealed_volume`, `sealed_amount`, `boards`, `limit_type`, `is_limit_up`, `reason_text`

**`stock_top_list`** —— 未开发 9 / 11（已开发：`amount`, `net_amount`）

- 未开发：`close`, `pct_change`, `turnover_rate`, `l_sell`, `l_buy`, `l_amount`, `net_rate`, `amount_rate`, `float_values`

**`tdx_minute`** —— 未开发 9 / 9（已开发：无）

- 未开发：`board_name`, `open`, `high`, `low`, `close`, `vol`, `amount`, `pct_change`, `amplitude`

**`index_daily`** —— 未开发 8 / 9（已开发：`pct_chg`）

- 未开发：`open`, `high`, `low`, `close`, `pre_close`, `change`, `vol`, `amount`

**`stock_finance`** —— 未开发 7 / 17（已开发：`turnover_rate`, `pe_ttm`, `pb`, `ps_ttm`, `dv_ttm`, `total_share`, `float_share`, `free_share`, `total_mv`, `circ_mv`）

- 未开发：`close`, `turnover_rate_f`, `volume_ratio`, `pe`, `pe_ttm_percentile`, `ps`, `dv_ratio`

**`stock_list`** —— 未开发 7 / 7（已开发：无）

- 未开发：`area`, `industry`, `symbol`, `act_name`, `act_ent_type`, `list_status`, `is_hs`

**`stock_pledge_stat`** —— 未开发 5 / 5（已开发：无）

- 未开发：`pledge_count`, `unrest_pledge`, `rest_pledge`, `total_share`, `pledge_ratio`

**`tdx_daily`** —— 未开发 5 / 6（已开发：`close`）

- 未开发：`open`, `high`, `low`, `vol`, `amount`

**`stock_dragon_tiger`** —— 未开发 4 / 7（已开发：`org_name`, `buy_amount`, `sell_amount`）

- 未开发：`buy_ratio`, `sell_ratio`, `net_buy_amount`, `direction`

**`stock_forecast`** —— 未开发 3 / 8（已开发：`p_change_min`, `p_change_max`, `net_profit_min`, `net_profit_max`, `last_parent_net`）

- 未开发：`first_ann_date`, `summary`, `change_reason`

**`stock_suspension`** —— 未开发 3 / 3（已开发：无）

- 未开发：`suspend_date`, `suspend_start_time`, `suspend_end_time`

**`dc_blocks`** —— 未开发 2 / 2（已开发：无）

- 未开发：`block_type`, `level`

**`stock_st_info`** —— 未开发 1 / 1（已开发：无）

- 未开发：`type_name`

</details>

### 复现方法

三步，全程只读：

1. **拉全上游 schema**：遍历 `../datadownload/data/<数据集>/**/*.parquet`，
   用 `pyarrow.ParquetFile(f).schema_arrow.names` 取每张表的列名。
2. **记录真实读取**：给 `pandas.read_parquet` 打一个钩子 —— 保留原函数，
   调用后把 `(path, columns, 返回的列)` 追加到 jsonl，然后用 `PYTHONPATH` 挂一个 `sitecustomize.py`，跑
   `main.py run --sandbox /tmp/... --start YYYY-MM-01 --end YYYY-MM-30 --jobs 1`。
   跑**一个月**就够：读了哪些列与窗口长短无关。342 个因子约 14 分钟。
3. **求差**：把第 2 步的读取集合，叠加价格层（`fea/prices.py:45-47`）与派生层（`fea/intraday.py:75-88`、`fea/chips.py:94,218`）**源码里写死**的列（派生缓存已存在时不会触发重建，钩子抓不到它们），再与第 1 步的 schema 求差。

盘点脚本与中间结果本轮放在 `/tmp/fea_audit/`（`upstream_schema.json`、`col_usage.json`、`field_coverage.py`）。⚠️ `/tmp` 不持久；要长期留存请把这三个文件复制到 `artifacts/audits/`。

**本次没有改任何因子、配置或数据**：instrumented run 走的是 `main.py run --sandbox`，产物重定向到 `/tmp`，用完已删。唯一副作用是 `logs/factors.log` 被追加了几行运行日志。

<a id="current-guide"></a>
## 当前操作与开发约定

### 当前基线

截至 2026-09-19 验收，注册 **337 个因子 + 5 个未来收益标签**，固定主板池 **2115 只**。输出下界统一为 `2018-01-01`，第一笔普通日频输出为 `2018-01-02`，本轮行情基准末日为 `2026-09-18`。受数据可得性限制的因子可以晚于统一下界；日期对齐不代表每个因子每日都有非缺失值。后续运行后的实时范围以程序状态与下方因子字典为准。

本轮正式产物为 3073 个年度分区、1,527,326,100 行，约 7.34 GiB。新增 31 个候选因子：筹码 14、财务 12、公告披露 5。以上是本轮快照，不是代码中永久固定的因子数量。

### 日常手动运行

先更新 `/root/autodl-fs/everyday_tasks/main.py` 的上游数据并确认结果，再运行因子入口。因子工程读取本地上游，不代替下载任务；上游尚未发布或缺失的数据不能凭空补出。以 SSH 的 root 用户登录时，使用已有维护用户 `claude` 生成文件，避免权限混杂：

```bash
cd /autodl-fs/data/featureengineering
runuser -u claude -- /autodl-fs/data/miniconda3/bin/python main.py
```

`/root/autodl-fs` 对应规范路径 `/autodl-fs/data`。无参数即全部已注册因子与标签的日常增量更新，默认 1 个计算进程，最多 2 个。所有家族共用此入口，不必逐个启动家族文件。查看状态或刷新本文因子字典：

```bash
runuser -u claude -- /autodl-fs/data/miniconda3/bin/python main.py status
runuser -u claude -- /autodl-fs/data/miniconda3/bin/python main.py docs
```

`docs` 仅刷新本文的自动因子字典区，不覆盖维护记录。`check` 是基础检查，真实历史因果性应使用 `audit-pit` 的源数据截断复算；详细参数见 `main.py --help` 及对应子命令帮助。

### 历史重建、缺失修复与中断续跑

```bash
runuser -u claude -- /autodl-fs/data/miniconda3/bin/python main.py rebuild --jobs 1 --dry-run
runuser -u claude -- /autodl-fs/data/miniconda3/bin/python main.py rebuild --jobs 1
runuser -u claude -- /autodl-fs/data/miniconda3/bin/python main.py rebuild --resume --jobs 1
runuser -u claude -- /autodl-fs/data/miniconda3/bin/python main.py rebuild --from-year 2026 --to-year 2026 --jobs 1
```

以上为分别使用的操作示例，无需逐条全部执行。`rebuild` 按年启动独立进程，默认强制重算所选区间；`--resume` 适用于已开始的重建中断后继续补齐。上游历史数值修订即使日期覆盖完整也需要强制重算，不应只用 `--resume`。普通增量对单因子跨三年及以上重算会返回非零并提示重建，不能把保护性退出误读为“全部完成”。不要通过增加并行或绕过多年保护解决此问题。

### 数据与计算原则

1. 因子函数为 `def name(ctx)`，返回 `(T, C)` 数组，`C=2115`；不返回 DataFrame、不自行转置。统一输出 `trade_date, stock_code, value, rank`，类型为 string、string、float32、float32，按年存储。缺失保持 NaN，由引擎统一缩尾和截面排名；方向字段是解释元数据，不应擅自在函数内翻转符号。
2. 输出起点取全局下界与因子自身起点的较晚者。**不得因输出从 2018 年开始就删除更早的上游输入或必要缓存。** 滚动窗口、TTM、同比和状态衔接需要预热；`warmup_days` 为日历天。旧经验中的 700 天等只作初值，必须用跨年边界及更长预热对照验证。
3. 跨期收益通常使用 `ctx.hfq` 的截至当日后复权口径；价格水平、市值使用未复权价格与当时可得股本，筹码成本与行情必须同口径。不得直接用会随未来分红重写历史的前复权序列作为输入。核对元/万元、股/手、百分数/比例及分母有效性，不能只看公式名称。
4. 财务按各报表来源维护披露版本，以当时可得公告日及报告期约束选值。累计利润、现金流先转 TTM；资产负债表为时点量；同比滞后按报告期，不用固定 252 个交易日替代。声明 `fin_fields` 与全部数据依赖，包含复权因子和父因子；改公式、口径或版本后按受影响历史重建。未来修订不得覆盖过去已披露版本；明确 NaN 修订保持缺失。
5. 当前固定上游延迟仅两融明细 1 个交易日。组合因子按父因子顺序计算，不重复加已落实的延迟。短样本数据稳定不等于证明厂商到达时间；公告时点与历史实际到达仍受上游记录限制。
6. 状态数据与事件数据分开处理。日频网格上的“没有事件”和“数据缺失”不是同一件事；只有明确定义且源覆盖充分的事件统计才使用无事件为零，不能把稀疏公告、财务缺失统一补零。低覆盖应结合来源与公式解释，不能以通用 30% 阈值删除所有事件因子。
7. 固定股票池保留已接受的幸存者偏差和事后 ST 筛选口径。新增字段不得擅自改变股票池，也不能用当前行业成分还原没有历史日期的行业关系。未来收益标签只供训练和评价，末端尚未成熟处保持 NaN，不能当成当日可用特征。

开发普通因子优先在所属 `factors/<家族>.py` 内完成，注册入口由 `factors/__init__.py` 管理；确需扩展引擎或共享配置时，应明确影响范围并验证其他家族。历史文件里的“只允许一个 Agent 改一个文件”属于当时协作分工，不限制经用户授权的项目维护。开发不依赖 Agent，手动操作同样适用这些约定。

### 资源、验证和已知边界

- 因子生成主要使用 CPU 与系统内存，不能把显存额度当作可用内存。当前按系统内存 60 GiB 约束，默认串行、最多 2 个计算进程、数值库线程 1；单进程地址空间上限约 24 GiB。按年处理，及时释放面板和缓存，不同时启动多个生成任务，出现内存不足先检查窗口与缓存，勿盲目加并行重试。
- 本轮既有 63 项测试通过。53 个财务相关对象在两个历史截点各比较 112,095 行，源公告截断后的值、排名和 NaN 完全一致；默认增量后全部 3073 个正式分区 MD5 不变。详细报告和限制见维护记录。文档整理不等于再次完成全量因子计算。
- 完整性检查要同时核对文件存在、日期网格、固定池、数值与 NaN，不只读状态文件中的完成标志。历史强制重建需要真实刷新，不是简单跳过“已完成”。动态 PIT 检查必须截断原始财务公告，不能只缩短行情面板。
- 筹码胜率原始单位为 0–100，转为 0–1，越界值置缺失；不能把 100.64 自动截成 100。2026 年“其他应收”相关科目的高零值比例已与源数据核对，保持稀疏标记，不擅自换字段或填值。
- 本轮新增候选的覆盖、IC 和相关性用于研究筛选，不证明样本外有效。ICIR 使用逐日 IC 汇总；普通 t 值未修正重叠未来收益的序列相关。去重相关性可能受缺失填零等近似影响，不能据单年指标自动删除因子；未在本轮重评的旧因子明确保持相应状态。
- MD5/SHA256 是完整性证据，**不能恢复被删除的数据**。重建副本验证后已清理，保留小型代码/元数据及校验台账。日常整理不得批量清空 `state/`、删除预热历史或把学习资料中的代码示例当成生产注册对象。

<a id="file-management"></a>
## 目录职责与文件管理

```text
featureengineering/
├── main.py                 # 手动操作统一入口
├── README.md               # 唯一 Markdown 总手册
├── conf/                   # 配置、固定股票池
├── fea/                    # 引擎、数据接入与公共计算
├── factors/                # 因子家族与注册
├── scripts/                # 辅助维护、验证脚本
├── tests/                  # 回归测试
├── data/
│   ├── factors/            # 正式年度因子数据
│   └── derived/            # 可复用派生缓存
├── state/                  # 当前覆盖、水位、锁、运行检查与状态
├── logs/                   # 程序运行日志
├── artifacts/
│   ├── dayhash/            # 逐日截面校验台账
│   ├── audits/             # 已完成维护的报告、清单、校验与代码快照
│   ├── experiments/        # 隔离实验、增量/全量对照参照
│   └── archive/            # 历史回测副本与状态备份
└── resources/
    └── research/           # PDF、TXT 等非 Markdown 研究资料
```

`__pycache__/` 为 Python 自动生成的缓存目录，可能随运行出现。`scripts/` 保留既有文件名与相对位置，`main.py rebuild` 等调用继续有效。`fea/` 和 `factors/` 保持既有包结构，避免仅为目录美观改变计算逻辑和导入关系。

`data/`、`state/`、`logs/` 的当前运行路径及 `conf/config.yaml` 保持有效；上游仍为 `../datadownload/data/`。`state/eval`、`state/dedup`、`state/pit_audit`、`state/history_prune`、`state/rebuild_attempts` 是现有命令持续读写或用于续跑/排错的位置，继续留在状态区；已结束且不参与运行判定的历史维护材料归入 `artifacts/audits/`。

### 2026-09-19 文件布局调整与路径迁移

本次将历史台账、审计、学习资料和大体积实验副本按用途归类，采用同一文件系统内的目录重命名，不复制大数据、不重算因子。只删除确认为空的 `state_bak/`。历史回测与参照数据仍保留，本次整理不等于清理这些数据所占的磁盘空间。

| 旧路径 | 新路径 | 说明 |
| --- | --- | --- |
| 根目录 `log0915`、`log0916`、`log0917`、`log0918`、`log0918_post`、`log0918_eod`、`log0918_frozen` | `artifacts/dayhash/` 下的同名目录 | 保留原台账、基线和元信息，不重新挑选历史基线 |
| `docs/` 的剩余非 Markdown 文件 | `artifacts/audits/legacy_docs/` | 历史覆盖表与预热实测日志 |
| `学习资料/` 的剩余非 Markdown 文件 | `resources/research/` | 原 `量化策略/` 子目录原样保留 |
| `state/backtest/` | `artifacts/archive/backtest/` | 历史数据、状态副本；后续 manifest 备份也写入这里 |
| `state/backup/` | `artifacts/archive/state_backup/` | 旧状态压缩包 |
| `state/inc_ref/` | `artifacts/experiments/inc_ref/` | 历史增量/全量参照数据 |
| `state/rebuild_20260919/` | `artifacts/audits/rebuild_20260919/` | 本轮因子重建和财务修复的完整证据 |
| `state/documentation_20260919/` | `artifacts/audits/documentation_20260919/` | 文档合并清单及验收证据 |

迁移规则按前缀适用于目录中所有子文件。**后部维护记录、历史原文、旧 JSON 与日志里的路径保留验收时写法，请按此表查找新位置。** 原 Markdown 文件名仍按本文“原文档迁移索引”定位；例如 `学习资料/factors.md` 是本文内的原文标题，而其原有 PDF 参考现在位于 `resources/research/量化策略/`。没有创建会长期混淆新旧布局的兼容软链接。

新增台账默认写入 `artifacts/dayhash/YYYY-MM-DD/`，日期来自计算截止日，完整年份可避免跨年同月同日混放。历史 `log0918` 的实际截止日是 2026-09-16，`log0918_post` 是 2026-09-17，目录名不能代替 `meta.json` 的 `cutoff`。因此保留历史目录名，不把任一旧目录自动当成新日期目录的比较基线。检查旧台账时显式传入对应 `--out` 路径；新目录没有 `dayhash.prev.tsv` 时，`--verify` 会明确失败，不能据此认定历史已通过校验。

本次仅调整 `main.py` 的路径帮助、`fea/dayhash.py` 的台账默认位置、`scripts/rebuild_manifest.py` 的备份位置，并让 `scripts/post_steps.sh` 从脚本自身定位项目，避免写死 `/root` 路径。未改变公式、股票池、资源限制或正式数据目录。手动每日运行仍是：

```bash
cd /autodl-fs/data/featureengineering
runuser -u claude -- /autodl-fs/data/miniconda3/bin/python main.py
```

后续新增材料按上述职责入目录，不在根目录建立日期杂项目录。实验若产生全量副本，应先估算磁盘容量，结束后另行验收清理；不要把 MD5 当成可恢复备份。布局审计和完整文件迁移清单位于 `artifacts/audits/layout_20260919/`。

<!-- FEA:DOC:layout-validation:BEGIN -->
本次布局调整已完成验收：

- 14 个目录、2418 个文件完成迁移，合计 4,471,481,941 字节（约 4.16 GiB）；同文件系统内重命名，没有额外复制大数据。迁移前后文件大小、纳秒修改时间、inode、设备号一致，小于 2 MB 的非 Parquet 文件还逐项比对 SHA256。
- 以 `claude` 用户从项目目录外验证 `main.py` 帮助、状态、台账帮助、2026 年重建计划预览和文档刷新，均成功；重建仅预览，没有启动因子生成。手册自动字典之外的内容保持不变。
- **78 项回归测试通过**，其中新增 4 项布局测试，覆盖跨年台账路径、显式输出目录、旧基线隔离、状态备份归档和辅助脚本从任意工作目录定位项目。Shell 语法检查通过。
- 342 个因子/标签的逻辑指纹、配置和因子源码、运行状态 JSON 全部保持一致；3098 个项目数据/缓存 Parquet 与 445 个上游 Parquet 的文件身份保持不变。未重新计算因子，也未改变上游数据。
- 项目内仍只有 `README.md` 一份 Markdown。全部旧目录按表迁移，空 `state_bak/` 已移除；`claude` 可写入新的报告及归档目录。

详细映射和验证记录见 `artifacts/audits/layout_20260919/plan.json`、`migration.json`、`files_before.json`、`verification.json` 与 `tests.log`。这些记录用于追溯布局变更；历史副本的保留与后续清理需要结合研究用途单独判断。
<!-- FEA:DOC:layout-validation:END -->

### 总手册的自动更新约定

默认 `main.py docs`、复权检查和分诊报告都写入本文，后两者追加带时间的报告（落在文末的自动报告区）。显式指定 `--out` 或 `--report` 仍可按操作者要求独立导出。指定 README 作为整文件导出路径会被拒绝，省略参数即可安全追加。

★ **2026-09-20 起本目录有 3 份 Markdown**（见文件头的表）。自动写入**只认 `README.md`**，其余两份是纯静态归档，不会被任何命令改写。**不要在 `HISTORY.md` / `REFERENCE_lingqi.md` 里放 `FEA:DOC` 标记**，也不要手工重写 `README.md` 里的标记区。

不要手改 `FEA:DOC` 标记包围的自动因子字典，也不要删除自动报告区边界。自动写入使用锁和临时文件替换，标记缺失/重复会报错而不会覆盖手册；同时手工编辑时应先保存并等生成结束，再继续修改。


> ⚠️ 下面这个 `merge-validation` 块是 **2026-09-19 那次「合并成单文件」的验收记录**，
> 其中「项目内递归扫描仅剩 `README.md`」一句**已被 2026-09-20 的拆分取代**。
> 它由 `fea/documentation.py` 按标记管理、不能手工重写，故原样保留作为历史凭据。

<a id="doc-split"></a>
## 本次文档拆分记录（2026-09-20）

拆分前的 `README.md` 是 2026-09-19 由 19 份 Markdown 合并成的单文件（24255 行 / 1.59 MiB），
其中 80.6% 是自动因子字典与冻结参考库原文，另有 913 行是两份现行文档的**旧版拷贝**
（逐字重复 92.9% / 80.4%，且与现行口径互相矛盾）。本次只做**按用途物理分离**，
不删除任何历史内容、不改任何代码：

| 去向 | 内容 | 拆分后行数 |
|:--|:--|--:|
| `README.md` | 现行约定 + 契约 + 运维手册 + 自动因子字典 + 自动报告区 + **本次体检报告** | **2793**（原 24255） |
| `HISTORY.md` | 一次性维护记录、专项审计、旧版文档原文、来源索引 | **3355** |
| `REFERENCE_lingqi.md` | 灵启因子库原文（`学习资料/factors.md` + `因子库.md`） | **18802** |
| 合计 | — | **24950** |

**搬运规则**：区间原样搬移，逐行核对「新文件非空行集合 ⊇ 原文非空行集合」；
`<!-- SOURCE:... -->` 标记虽然没有代码依赖，也一并保留以便日后回溯。
被丢弃的非空行只有 6 条 `<details>` / `> 来源：` 外壳（现行正文从折叠块里取出、直接展开），
其余丢弃的 15 条都是空行。

**顺带做的一处规范化**：原文有 1262 行是 CRLF 行尾、其余是 LF（两批文档合并留下的）。
本次搬运经 Python 文本读写，行尾统一成了 LF。这是纯格式变化，**不改变任何一行的可见内容**，
但会让拆分后文件的字节数与「来源正文 SHA256」不再逐字节对齐 —— 那些 SHA256 校验的是
**合并前的原始文件**，本来就不是合并后的正文，所以不受影响。

**没有额外留存拆分前的整份副本**：内容已按上表全量搬入三份文件（逐行核对过），
唯一的净损失是 6 条 `<details>` 外壳行。

**同时修掉的 6 处旧口径**（拆分前它们与现行约定并存于同一文件，容易误读）：

1. `4 张表晚一个交易日` → 现行只有 `stock_margin_detail` 一张（旧版原文仍在
   `HISTORY.md`，已加废弃标注）。
2. 输出起点 `2012-01-01` → 现行 `2018-01-01`。
3. 股票池 `约 3000+ 只` → 现行固定 **2115 只**。
4. 并行度 `--jobs 8` → 现行 `--jobs 1`、最多 2。
5. `stock_report_rc` 仍在滞后表清单 → 该表已连同数据删除。
6. `/root/autodl-fs/...` 旧路径 → 规范路径 `/autodl-fs/data/...`。

**没有做的事**：没有改因子公式、没有重算任何因子、没有动 `state/`、没有删数据。
因子字典区仍是 `main.py docs` 的自动区，下一次运行会照常原地刷新。

<!-- FEA:DOC:merge-validation:BEGIN -->
已完成验收：项目内递归扫描仅剩 `README.md`，19 个原 Markdown 文件的 18 份不同内容已归并，完全重复报告只保留一次。

- `main.py docs` 真实连续运行 2 次成功；第二次文件内容完全一致，字典之外的手工说明、维护记录和历史材料逐字不变。字典包含 342 个注册对象。
- 新增 11 项文档安全与路由测试通过；连同原有测试共 **74 项通过**。辅助脚本使用模拟检查结果验证报告路径；未执行真实分诊删除，也未启动因子重建。
- 对照整理前后，342 个注册对象的逻辑指纹、57 个计算/配置文件内容相同；3098 个项目内因子/缓存 Parquet、445 个上游 Parquet 的路径、大小、纳秒修改时间与 inode 全部相同。此处是文件身份核对，不声称又做了一次全量数据内容哈希。
- 当前内部跳转 922 处全部有对应锚点，无重复显式锚点，无指向已移除 Markdown 的相对链接。17 份非自动字典正文在刷新字典后继续通过合并正文 SHA256 验证。
- 旧学习资料目录存在 root 权限限制：代码和手册仍由 `claude` 写入，仅对逐项校验通过的旧 Markdown 使用 root 完成清理，未修改该目录其余文件或权限。

来源原始 MD5/SHA256、正文迁移 SHA256、删除清单、代码变更前小型文本及测试日志位于 `artifacts/audits/documentation_20260919/`。最终验收记录为 `verification.json`。未备份或复制原始数据；本次整理的价值在于统一说明与避免重复生成文档，不作为大幅释放数据存储的措施。
<!-- FEA:DOC:merge-validation:END -->


<a id="factor-catalog"></a>
## 当前完整因子字典

<a id="source-18d9051eb91d"></a>
<!-- FEA:DOC:factor-catalog:BEGIN -->
# 因子字典（featureengineering · 模块②）

> 本节由 `python main.py docs` 生成，请勿手工编辑本节；其他章节保持不变。

## ★ 因子开发的三条硬约束（下游是 A 股日横断面回归排序任务）

1. **全部因子必须日频** —— 对齐到 `(trade_date, stock_code)` 面板后落盘。
2. **固定主板股票池 2115 只** —— 使用 `conf/universe_frozen.tsv`；
   排除创业板 `300/301/302`、科创板 `688/689`、北交所 `832/833/920`。
3. **输出起点由 `conf/config.yaml` 的 `default_start` 控制** —— 当前 **`2018-01-01`**；实际取全局下界与因子可得起点的较晚者，历史输入保留作预热。

## ★★ PIT 红线：历史因子值不得因未来的分红事件而变化

前复权（qfq）把整条价格序列按**最新**的复权因子缩放，一旦发生新的分红/送转，
全部历史价格会一起被重算 —— 同一段历史今天算和昨天算结果不同，回测里就是前视偏差。

- **禁止**把 `stock_kline_adj` / `stock_daily_adj`（前复权）直接喂给因子函数；
  `register()` 会直接拒绝，确需使用必须显式 `allow_qfq=True` 并写明理由。
- 价格**水平**类因子（市值、book-to-market）→ 用**未复权**价 × 当期已披露股本。
- 收益**比率**类因子（动量、波动）→ 用未复权价 + `stock_adj_factor` 截至**当日**的
  累计复权因子还原（等价于「截至当日的后复权」，后复权锚定序列起点，历史值稳定）。

## 统一输出格式

```
data/factors/<因子名>/year=YYYY/data.parquet
  trade_date  string   "2026-09-11"
  stock_code  string   "600000.SH"
  value       float32  原始因子值（可解释、可再标准化）
  rank        float32  当日截面百分位 [0,1]（先 1%/99% winsorize 再 rank）
```

因子语义：按既定数据可得日，T 日收盘后生成，供 T+1 使用；公告到达时间仍受供应商记录限制。
未来收益标签仅供训练/评价，须等待未来行情成熟，不能作为当日可用特征。

## 因子清单

共 598 个因子、5 个标签。

> ★ 所有因子的**列名 / 列序 / dtype / 目录结构 / 分区方式 / 语义完全一致**，
> **唯一允许的差异是起止日期**（下表最后两列）。

| 因子 | 类别 | 方向 | 定义 | 公式 | 起点 | 实际起止 | warmup(天) | 依赖 |
|:--|:--|:--|:--|:--|:--|:--|--:|:--|
| `adx_14` | technical | 高优 | 14 日趋势强度 ADX（只衡量强弱、不含方向），单位 % | `scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
di_plus, di_minus = _directional_movement(daily["high"], daily["low"], daily["pre_close"], scale, 14)
dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-8)
adx = dx.groupby(level="Code").transform(
    lambda s: s.rolling(14, min_periods=7).mean()
)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 300 | stock_daily, stock_adj_factor |
| `afx_bs_cap_rese` | field_balance | 高优 | 资本公积金年度结构占比 | `asinh(annual(stock_balancesheet.cap_rese)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_const_materials` | field_balance | 高优 | 工程物资年度结构占比 | `asinh(annual(stock_balancesheet.const_materials)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_debt_invest` | field_balance | 高优 | 债权投资年度结构占比 | `asinh(annual(stock_balancesheet.debt_invest)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2019-04-01 | 2019-04-01 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_decr_in_disbur` | field_balance | 高优 | 发放贷款及垫款年度结构占比 | `asinh(annual(stock_balancesheet.decr_in_disbur)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_deriv_assets` | field_balance | 高优 | 衍生金融资产年度结构占比 | `asinh(annual(stock_balancesheet.deriv_assets)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_deriv_liab` | field_balance | 高优 | 衍生金融负债年度结构占比 | `asinh(annual(stock_balancesheet.deriv_liab)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_div_payable` | field_balance | 高优 | 应付股利年度结构占比 | `asinh(annual(stock_balancesheet.div_payable)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_div_receiv` | field_balance | 高优 | 应收股利年度结构占比 | `asinh(annual(stock_balancesheet.div_receiv)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_estimated_liab` | field_balance | 高优 | 预计负债年度结构占比 | `asinh(annual(stock_balancesheet.estimated_liab)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_fix_assets_total` | field_balance | 高优 | 固定资产(合计)年度结构占比 | `asinh(annual(stock_balancesheet.fix_assets_total)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_fixed_assets_disp` | field_balance | 高优 | 固定资产清理年度结构占比 | `asinh(annual(stock_balancesheet.fixed_assets_disp)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_hfs_assets` | field_balance | 高优 | 持有待售的资产年度结构占比 | `asinh(annual(stock_balancesheet.hfs_assets)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_int_payable` | field_balance | 高优 | 应付利息年度结构占比 | `asinh(annual(stock_balancesheet.int_payable)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_int_receiv` | field_balance | 高优 | 应收利息年度结构占比 | `asinh(annual(stock_balancesheet.int_receiv)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_invest_real_estate` | field_balance | 高优 | 投资性房地产年度结构占比 | `asinh(annual(stock_balancesheet.invest_real_estate)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_loan_oth_bank` | field_balance | 高优 | 拆入资金年度结构占比 | `asinh(annual(stock_balancesheet.loan_oth_bank)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_long_pay_total` | field_balance | 高优 | 长期应付款(合计)年度结构占比 | `asinh(annual(stock_balancesheet.long_pay_total)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_lt_amor_exp` | field_balance | 高优 | 长期待摊费用年度结构占比 | `asinh(annual(stock_balancesheet.lt_amor_exp)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_lt_eqt_invest` | field_balance | 高优 | 长期股权投资年度结构占比 | `asinh(annual(stock_balancesheet.lt_eqt_invest)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_lt_payable` | field_balance | 高优 | 长期应付款年度结构占比 | `asinh(annual(stock_balancesheet.lt_payable)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_lt_payroll_payable` | field_balance | 高优 | 长期应付职工薪酬年度结构占比 | `asinh(annual(stock_balancesheet.lt_payroll_payable)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_lt_rec` | field_balance | 高优 | 长期应收款年度结构占比 | `asinh(annual(stock_balancesheet.lt_rec)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_nca_within_1y` | field_balance | 高优 | 一年内到期的非流动资产年度结构占比 | `asinh(annual(stock_balancesheet.nca_within_1y)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_ordin_risk_reser` | field_balance | 高优 | 一般风险准备年度结构占比 | `asinh(annual(stock_balancesheet.ordin_risk_reser)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_oth_assets` | field_balance | 高优 | 其他资产年度结构占比 | `asinh(annual(stock_balancesheet.oth_assets)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_oth_comp_income` | field_balance | 高优 | 其他综合收益年度结构占比 | `asinh(annual(stock_balancesheet.oth_comp_income)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_oth_cur_assets` | field_balance | 高优 | 其他流动资产年度结构占比 | `asinh(annual(stock_balancesheet.oth_cur_assets)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_oth_cur_liab` | field_balance | 高优 | 其他流动负债年度结构占比 | `asinh(annual(stock_balancesheet.oth_cur_liab)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_oth_debt_invest` | field_balance | 高优 | 其他债权投资年度结构占比 | `asinh(annual(stock_balancesheet.oth_debt_invest)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2019-04-01 | 2019-04-01 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_oth_eqt_tools` | field_balance | 高优 | 其他权益工具年度结构占比 | `asinh(annual(stock_balancesheet.oth_eqt_tools)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_oth_nca` | field_balance | 高优 | 其他非流动资产年度结构占比 | `asinh(annual(stock_balancesheet.oth_nca)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_oth_ncl` | field_balance | 高优 | 其他非流动负债年度结构占比 | `asinh(annual(stock_balancesheet.oth_ncl)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_oth_pay_total` | field_balance | 高优 | 其他应付款(合计)年度结构占比 | `asinh(annual(stock_balancesheet.oth_pay_total)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_oth_payable` | field_balance | 高优 | 其他应付款年度结构占比 | `asinh(annual(stock_balancesheet.oth_payable)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_oth_rcv_total` | field_balance | 高优 | 其他应收款(合计)年度结构占比 | `asinh(annual(stock_balancesheet.oth_rcv_total)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_payroll_payable` | field_balance | 高优 | 应付职工薪酬年度结构占比 | `asinh(annual(stock_balancesheet.payroll_payable)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_produc_bio_assets` | field_balance | 高优 | 生产性生物资产年度结构占比 | `asinh(annual(stock_balancesheet.produc_bio_assets)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_pur_resale_fa` | field_balance | 高优 | 买入返售金融资产年度结构占比 | `asinh(annual(stock_balancesheet.pur_resale_fa)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_sold_for_repur_fa` | field_balance | 高优 | 卖出回购金融资产款年度结构占比 | `asinh(annual(stock_balancesheet.sold_for_repur_fa)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_special_rese` | field_balance | 高优 | 专项储备年度结构占比 | `asinh(annual(stock_balancesheet.special_rese)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_specific_payables` | field_balance | 高优 | 专项应付款年度结构占比 | `asinh(annual(stock_balancesheet.specific_payables)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_surplus_rese` | field_balance | 高优 | 盈余公积金年度结构占比 | `asinh(annual(stock_balancesheet.surplus_rese)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_taxes_payable` | field_balance | 高优 | 应交税费年度结构占比 | `asinh(annual(stock_balancesheet.taxes_payable)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_trading_fl` | field_balance | 高优 | 交易性金融负债年度结构占比 | `asinh(annual(stock_balancesheet.trading_fl)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_bs_undistr_porfit` | field_balance | 高优 | 未分配利润年度结构占比 | `asinh(annual(stock_balancesheet.undistr_porfit)/annual(stock_balancesheet.total_liab_hldr_eqy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_balancesheet |
| `afx_cf_beg_bal_cash` | field_cashflow | 高优 | 减:现金的期初余额年度结构占比 | `asinh(annual(stock_cashflow.beg_bal_cash)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_beg_bal_cash_equ` | field_cashflow | 高优 | 减:现金等价物的期初余额年度结构占比 | `asinh(annual(stock_cashflow.beg_bal_cash_equ)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_c_cash_equ_end_period` | field_cashflow | 高优 | 期末现金及现金等价物余额年度结构占比 | `asinh(annual(stock_cashflow.c_cash_equ_end_period)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_c_disp_withdrwl_invest` | field_cashflow | 高优 | 收回投资收到的现金年度结构占比 | `asinh(annual(stock_cashflow.c_disp_withdrwl_invest)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_c_fr_oth_operate_a` | field_cashflow | 高优 | 收到其他与经营活动有关的现金年度结构占比 | `asinh(annual(stock_cashflow.c_fr_oth_operate_a)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_c_paid_invest` | field_cashflow | 高优 | 投资支付的现金年度结构占比 | `asinh(annual(stock_cashflow.c_paid_invest)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_c_recp_cap_contrib` | field_cashflow | 高优 | 吸收投资收到的现金年度结构占比 | `asinh(annual(stock_cashflow.c_recp_cap_contrib)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_c_recp_return_invest` | field_cashflow | 高优 | 取得投资收益收到的现金年度结构占比 | `asinh(annual(stock_cashflow.c_recp_return_invest)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_decr_def_inc_tax_assets` | field_cashflow | 高优 | 递延所得税资产减少年度结构占比 | `asinh(annual(stock_cashflow.decr_def_inc_tax_assets)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_decr_oper_payable` | field_cashflow | 高优 | 经营性应收项目的减少年度结构占比 | `asinh(annual(stock_cashflow.decr_oper_payable)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_eff_fx_flu_cash` | field_cashflow | 高优 | 汇率变动对现金的影响年度结构占比 | `asinh(annual(stock_cashflow.eff_fx_flu_cash)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_end_bal_cash_equ` | field_cashflow | 高优 | 加:现金等价物的期末余额年度结构占比 | `asinh(annual(stock_cashflow.end_bal_cash_equ)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_finan_exp` | field_cashflow | 高优 | 财务费用年度结构占比 | `asinh(annual(stock_cashflow.finan_exp)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_ifc_cash_incr` | field_cashflow | 高优 | 收取利息和手续费净增加额年度结构占比 | `asinh(annual(stock_cashflow.ifc_cash_incr)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_im_n_incr_cash_equ` | field_cashflow | 高优 | 现金及现金等价物净增加额(间接法)年度结构占比 | `asinh(annual(stock_cashflow.im_n_incr_cash_equ)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_im_net_cashflow_oper_act` | field_cashflow | 高优 | 经营活动产生的现金流量净额(间接法)年度结构占比 | `asinh(annual(stock_cashflow.im_net_cashflow_oper_act)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_incl_cash_rec_saims` | field_cashflow | 高优 | 其中:子公司吸收少数股东投资收到的现金年度结构占比 | `asinh(annual(stock_cashflow.incl_cash_rec_saims)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_incl_dvd_profit_paid_sc_ms` | field_cashflow | 高优 | 其中:子公司支付给少数股东的股利、利润年度结构占比 | `asinh(annual(stock_cashflow.incl_dvd_profit_paid_sc_ms)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_incr_def_inc_tax_liab` | field_cashflow | 高优 | 递延所得税负债增加年度结构占比 | `asinh(annual(stock_cashflow.incr_def_inc_tax_liab)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_incr_oper_payable` | field_cashflow | 高优 | 经营性应付项目的增加年度结构占比 | `asinh(annual(stock_cashflow.incr_oper_payable)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_loss_disp_fiolta` | field_cashflow | 高优 | 处置固定、无形资产和其他长期资产的损失年度结构占比 | `asinh(annual(stock_cashflow.loss_disp_fiolta)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_loss_fv_chg` | field_cashflow | 高优 | 公允价值变动损失年度结构占比 | `asinh(annual(stock_cashflow.loss_fv_chg)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_loss_scr_fa` | field_cashflow | 高优 | 固定资产报废损失年度结构占比 | `asinh(annual(stock_cashflow.loss_scr_fa)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_n_depos_incr_fi` | field_cashflow | 高优 | 客户存款和同业存放款项净增加额年度结构占比 | `asinh(annual(stock_cashflow.n_depos_incr_fi)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_n_disp_subs_oth_biz` | field_cashflow | 高优 | 取得子公司及其他营业单位支付的现金净额年度结构占比 | `asinh(annual(stock_cashflow.n_disp_subs_oth_biz)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_n_incr_clt_loan_adv` | field_cashflow | 高优 | 客户贷款及垫款净增加额年度结构占比 | `asinh(annual(stock_cashflow.n_incr_clt_loan_adv)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_n_incr_loans_oth_bank` | field_cashflow | 高优 | 拆入资金净增加额年度结构占比 | `asinh(annual(stock_cashflow.n_incr_loans_oth_bank)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_n_recp_disp_fiolta` | field_cashflow | 高优 | 处置固定资产、无形资产和其他长期资产收回的现金净额年度结构占比 | `asinh(annual(stock_cashflow.n_recp_disp_fiolta)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_n_recp_disp_sobu` | field_cashflow | 高优 | 处置子公司及其他营业单位收到的现金净额年度结构占比 | `asinh(annual(stock_cashflow.n_recp_disp_sobu)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_net_profit` | field_cashflow | 高优 | 净利润年度结构占比 | `asinh(annual(stock_cashflow.net_profit)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_oth_cash_pay_oper_act` | field_cashflow | 高优 | 支付其他与经营活动有关的现金年度结构占比 | `asinh(annual(stock_cashflow.oth_cash_pay_oper_act)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_oth_cash_recp_ral_fnc_act` | field_cashflow | 高优 | 收到其他与筹资活动有关的现金年度结构占比 | `asinh(annual(stock_cashflow.oth_cash_recp_ral_fnc_act)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_oth_cashpay_ral_fnc_act` | field_cashflow | 高优 | 支付其他与筹资活动有关的现金年度结构占比 | `asinh(annual(stock_cashflow.oth_cashpay_ral_fnc_act)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_oth_pay_ral_inv_act` | field_cashflow | 高优 | 支付其他与投资活动有关的现金年度结构占比 | `asinh(annual(stock_cashflow.oth_pay_ral_inv_act)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_oth_recp_ral_inv_act` | field_cashflow | 高优 | 收到其他与投资活动有关的现金年度结构占比 | `asinh(annual(stock_cashflow.oth_recp_ral_inv_act)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_pay_handling_chrg` | field_cashflow | 高优 | 支付手续费的现金年度结构占比 | `asinh(annual(stock_cashflow.pay_handling_chrg)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_proc_issue_bonds` | field_cashflow | 高优 | 发行债券收到的现金年度结构占比 | `asinh(annual(stock_cashflow.proc_issue_bonds)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_stot_cash_in_fnc_act` | field_cashflow | 高优 | 筹资活动现金流入小计年度结构占比 | `asinh(annual(stock_cashflow.stot_cash_in_fnc_act)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_stot_cashout_fnc_act` | field_cashflow | 高优 | 筹资活动现金流出小计年度结构占比 | `asinh(annual(stock_cashflow.stot_cashout_fnc_act)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_stot_inflows_inv_act` | field_cashflow | 高优 | 投资活动现金流入小计年度结构占比 | `asinh(annual(stock_cashflow.stot_inflows_inv_act)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_stot_out_inv_act` | field_cashflow | 高优 | 投资活动现金流出小计年度结构占比 | `asinh(annual(stock_cashflow.stot_out_inv_act)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_cf_use_right_asset_dep` | field_cashflow | 高优 | 使用权资产折旧年度结构占比 | `asinh(annual(stock_cashflow.use_right_asset_dep)/annual(stock_cashflow.c_inf_fr_operate_a))` | 2020-03-30 | 2020-03-30 → 2026-09-21 | 1100 | stock_cashflow |
| `afx_fi_adminexp_of_gr` | field_indicator | 高优 | 管理费用/营业总收入年度同期差 | `asinh(annual(stock_financial_indicator.adminexp_of_gr)-annual_lag1y(stock_financial_indicator.adminexp_of_gr))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ar_turn` | field_indicator | 高优 | 应收账款周转率年度同期差 | `asinh(annual(stock_financial_indicator.ar_turn)-annual_lag1y(stock_financial_indicator.ar_turn))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_arturn_days` | field_indicator | 高优 | 应收账款周转天数年度同期差 | `asinh(annual(stock_financial_indicator.arturn_days)-annual_lag1y(stock_financial_indicator.arturn_days))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_assets_turn` | field_indicator | 高优 | 总资产周转率年度同期差 | `asinh(annual(stock_financial_indicator.assets_turn)-annual_lag1y(stock_financial_indicator.assets_turn))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_basic_eps_yoy` | field_indicator | 高优 | 基本每股收益同比增长率(%)年度同期差 | `asinh(annual(stock_financial_indicator.basic_eps_yoy)-annual_lag1y(stock_financial_indicator.basic_eps_yoy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_bps` | field_indicator | 高优 | 每股净资产年度变化率 | `asinh((annual(stock_financial_indicator.bps)-annual_lag1y(stock_financial_indicator.bps))/abs(annual_lag1y(stock_financial_indicator.bps)))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ca_turn` | field_indicator | 高优 | 流动资产周转率年度同期差 | `asinh(annual(stock_financial_indicator.ca_turn)-annual_lag1y(stock_financial_indicator.ca_turn))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_capital_rese_ps` | field_indicator | 高优 | 每股资本公积年度每股净资产归一化 | `asinh(annual(stock_financial_indicator.capital_rese_ps)/annual(stock_financial_indicator.bps))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_capitalized_to_da` | field_indicator | 高优 | 资本支出/折旧和摊销年度同期差 | `asinh(annual(stock_financial_indicator.capitalized_to_da)-annual_lag1y(stock_financial_indicator.capitalized_to_da))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_cash_to_liqdebt` | field_indicator | 高优 | 货币资金／流动负债年度同期差 | `asinh(annual(stock_financial_indicator.cash_to_liqdebt)-annual_lag1y(stock_financial_indicator.cash_to_liqdebt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_cash_to_liqdebt_withinterest` | field_indicator | 高优 | 货币资金／带息流动负债年度同期差 | `asinh(annual(stock_financial_indicator.cash_to_liqdebt_withinterest)-annual_lag1y(stock_financial_indicator.cash_to_liqdebt_withinterest))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_cfps` | field_indicator | 高优 | 每股现金流量净额年度每股净资产归一化 | `asinh(annual(stock_financial_indicator.cfps)/annual(stock_financial_indicator.bps))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_cfps_yoy` | field_indicator | 高优 | 每股经营活动现金流净额同比增长率(%)年度同期差 | `asinh(annual(stock_financial_indicator.cfps_yoy)-annual_lag1y(stock_financial_indicator.cfps_yoy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_cogs_of_sales` | field_indicator | 高优 | 销售成本率年度同期差 | `asinh(annual(stock_financial_indicator.cogs_of_sales)-annual_lag1y(stock_financial_indicator.cogs_of_sales))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_current_exint` | field_indicator | 高优 | 无息流动负债年度投入资本占比 | `asinh(annual(stock_financial_indicator.current_exint)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_daa` | field_indicator | 高优 | 折旧与摊销年度投入资本占比 | `asinh(annual(stock_financial_indicator.daa)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_diluted2_eps` | field_indicator | 高优 | 期末摊薄每股收益年度每股净资产归一化 | `asinh(annual(stock_financial_indicator.diluted2_eps)/annual(stock_financial_indicator.bps))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_dp_assets_to_eqt` | field_indicator | 高优 | 权益乘数(杜邦分析)年度同期差 | `asinh(annual(stock_financial_indicator.dp_assets_to_eqt)-annual_lag1y(stock_financial_indicator.dp_assets_to_eqt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_dtprofit_to_profit` | field_indicator | 高优 | 扣除非经常损益后的净利润/净利润年度同期差 | `asinh(annual(stock_financial_indicator.dtprofit_to_profit)-annual_lag1y(stock_financial_indicator.dtprofit_to_profit))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ebit` | field_indicator | 高优 | 息税前利润年度投入资本占比 | `asinh(annual(stock_financial_indicator.ebit)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ebit_of_gr` | field_indicator | 高优 | 息税前利润/营业总收入年度同期差 | `asinh(annual(stock_financial_indicator.ebit_of_gr)-annual_lag1y(stock_financial_indicator.ebit_of_gr))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ebit_ps` | field_indicator | 高优 | 每股息税前利润年度每股净资产归一化 | `asinh(annual(stock_financial_indicator.ebit_ps)/annual(stock_financial_indicator.bps))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ebit_to_interest` | field_indicator | 高优 | 已获利息倍数(EBIT/利息费用)年度同期差 | `asinh(annual(stock_financial_indicator.ebit_to_interest)-annual_lag1y(stock_financial_indicator.ebit_to_interest))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ebitda` | field_indicator | 高优 | 息税折旧摊销前利润年度投入资本占比 | `asinh(annual(stock_financial_indicator.ebitda)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_eqt_to_debt` | field_indicator | 高优 | 归属于母公司的股东权益/负债合计年度同期差 | `asinh(annual(stock_financial_indicator.eqt_to_debt)-annual_lag1y(stock_financial_indicator.eqt_to_debt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_eqt_to_interestdebt` | field_indicator | 高优 | 归属于母公司的股东权益/带息债务年度同期差 | `asinh(annual(stock_financial_indicator.eqt_to_interestdebt)-annual_lag1y(stock_financial_indicator.eqt_to_interestdebt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_eqt_to_talcapital` | field_indicator | 高优 | 归属于母公司的股东权益/全部投入资本年度同期差 | `asinh(annual(stock_financial_indicator.eqt_to_talcapital)-annual_lag1y(stock_financial_indicator.eqt_to_talcapital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_equity_yoy` | field_indicator | 高优 | 净资产同比增长率年度同期差 | `asinh(annual(stock_financial_indicator.equity_yoy)-annual_lag1y(stock_financial_indicator.equity_yoy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_expense_of_sales` | field_indicator | 高优 | 销售期间费用率年度同期差 | `asinh(annual(stock_financial_indicator.expense_of_sales)-annual_lag1y(stock_financial_indicator.expense_of_sales))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_extra_item` | field_indicator | 高优 | 非经常性损益年度投入资本占比 | `asinh(annual(stock_financial_indicator.extra_item)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_fa_turn` | field_indicator | 高优 | 固定资产周转率年度同期差 | `asinh(annual(stock_financial_indicator.fa_turn)-annual_lag1y(stock_financial_indicator.fa_turn))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_fcfe` | field_indicator | 高优 | 股权自由现金流量年度投入资本占比 | `asinh(annual(stock_financial_indicator.fcfe)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_fcff` | field_indicator | 高优 | 企业自由现金流量年度投入资本占比 | `asinh(annual(stock_financial_indicator.fcff)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_finaexp_of_gr` | field_indicator | 高优 | 财务费用/营业总收入年度同期差 | `asinh(annual(stock_financial_indicator.finaexp_of_gr)-annual_lag1y(stock_financial_indicator.finaexp_of_gr))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_fixed_assets` | field_indicator | 高优 | 固定资产合计年度投入资本占比 | `asinh(annual(stock_financial_indicator.fixed_assets)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_gc_of_gr` | field_indicator | 高优 | 营业总成本/营业总收入年度同期差 | `asinh(annual(stock_financial_indicator.gc_of_gr)-annual_lag1y(stock_financial_indicator.gc_of_gr))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_gross_margin` | field_indicator | 高优 | 毛利年度投入资本占比 | `asinh(annual(stock_financial_indicator.gross_margin)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_impai_ttm` | field_indicator | 高优 | 资产减值损失/营业总收入年度同期差 | `asinh(annual(stock_financial_indicator.impai_ttm)-annual_lag1y(stock_financial_indicator.impai_ttm))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_interestdebt` | field_indicator | 高优 | 带息债务年度投入资本占比 | `asinh(annual(stock_financial_indicator.interestdebt)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_interst_income` | field_indicator | 高优 | 利息费用年度投入资本占比 | `asinh(annual(stock_financial_indicator.interst_income)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_inv_turn` | field_indicator | 高优 | 存货周转率年度同期差 | `asinh(annual(stock_financial_indicator.inv_turn)-annual_lag1y(stock_financial_indicator.inv_turn))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_invest_capital` | field_indicator | 高优 | 投入资本年度变化率 | `asinh((annual(stock_financial_indicator.invest_capital)-annual_lag1y(stock_financial_indicator.invest_capital))/abs(annual_lag1y(stock_financial_indicator.invest_capital)))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_investincome_of_ebt` | field_indicator | 高优 | 价值变动净收益/利润总额年度同期差 | `asinh(annual(stock_financial_indicator.investincome_of_ebt)-annual_lag1y(stock_financial_indicator.investincome_of_ebt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_invturn_days` | field_indicator | 高优 | 存货周转天数年度同期差 | `asinh(annual(stock_financial_indicator.invturn_days)-annual_lag1y(stock_financial_indicator.invturn_days))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_longdebt_to_workingcapital` | field_indicator | 高优 | 长期债务与营运资金比率年度同期差 | `asinh(annual(stock_financial_indicator.longdebt_to_workingcapital)-annual_lag1y(stock_financial_indicator.longdebt_to_workingcapital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_n_op_profit_of_ebt` | field_indicator | 高优 | 营业外收支净额/利润总额年度同期差 | `asinh(annual(stock_financial_indicator.n_op_profit_of_ebt)-annual_lag1y(stock_financial_indicator.n_op_profit_of_ebt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_netdebt` | field_indicator | 高优 | 净债务年度投入资本占比 | `asinh(annual(stock_financial_indicator.netdebt)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_networking_capital` | field_indicator | 高优 | 营运流动资本年度投入资本占比 | `asinh(annual(stock_financial_indicator.networking_capital)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_non_op_profit` | field_indicator | 高优 | 非营业利润年度投入资本占比 | `asinh(annual(stock_financial_indicator.non_op_profit)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_noncurrent_exint` | field_indicator | 高优 | 无息非流动负债年度投入资本占比 | `asinh(annual(stock_financial_indicator.noncurrent_exint)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_npta` | field_indicator | 高优 | 总资产净利润年度同期差 | `asinh(annual(stock_financial_indicator.npta)-annual_lag1y(stock_financial_indicator.npta))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ocf_to_debt` | field_indicator | 高优 | 经营活动产生的现金流量净额/负债合计年度同期差 | `asinh(annual(stock_financial_indicator.ocf_to_debt)-annual_lag1y(stock_financial_indicator.ocf_to_debt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ocf_to_interestdebt` | field_indicator | 高优 | 经营活动产生的现金流量净额/带息债务年度同期差 | `asinh(annual(stock_financial_indicator.ocf_to_interestdebt)-annual_lag1y(stock_financial_indicator.ocf_to_interestdebt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ocf_to_netdebt` | field_indicator | 高优 | 经营活动产生的现金流量净额/净债务年度同期差 | `asinh(annual(stock_financial_indicator.ocf_to_netdebt)-annual_lag1y(stock_financial_indicator.ocf_to_netdebt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ocf_to_opincome` | field_indicator | 高优 | 经营活动产生的现金流量净额/经营活动净收益年度同期差 | `asinh(annual(stock_financial_indicator.ocf_to_opincome)-annual_lag1y(stock_financial_indicator.ocf_to_opincome))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ocf_to_or` | field_indicator | 高优 | 经营活动产生的现金流量净额/营业收入年度同期差 | `asinh(annual(stock_financial_indicator.ocf_to_or)-annual_lag1y(stock_financial_indicator.ocf_to_or))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ocf_to_profit` | field_indicator | 高优 | 经营活动产生的现金流量净额／营业利润年度同期差 | `asinh(annual(stock_financial_indicator.ocf_to_profit)-annual_lag1y(stock_financial_indicator.ocf_to_profit))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_ocfps` | field_indicator | 高优 | 每股经营活动产生的现金流量净额年度每股净资产归一化 | `asinh(annual(stock_financial_indicator.ocfps)/annual(stock_financial_indicator.bps))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_op_income` | field_indicator | 高优 | 经营活动净收益年度投入资本占比 | `asinh(annual(stock_financial_indicator.op_income)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_op_to_debt` | field_indicator | 高优 | 营业利润／负债合计年度同期差 | `asinh(annual(stock_financial_indicator.op_to_debt)-annual_lag1y(stock_financial_indicator.op_to_debt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_op_to_ebt` | field_indicator | 高优 | 营业利润／利润总额年度同期差 | `asinh(annual(stock_financial_indicator.op_to_ebt)-annual_lag1y(stock_financial_indicator.op_to_ebt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_op_to_liqdebt` | field_indicator | 高优 | 营业利润／流动负债年度同期差 | `asinh(annual(stock_financial_indicator.op_to_liqdebt)-annual_lag1y(stock_financial_indicator.op_to_liqdebt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_opincome_of_ebt` | field_indicator | 高优 | 经营活动净收益/利润总额年度同期差 | `asinh(annual(stock_financial_indicator.opincome_of_ebt)-annual_lag1y(stock_financial_indicator.opincome_of_ebt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_profit_dedt` | field_indicator | 高优 | 扣除非经常性损益后的净利润（扣非净利润）年度投入资本占比 | `asinh(annual(stock_financial_indicator.profit_dedt)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_adminexp_to_gr` | field_indicator | 高优 | 管理费用／营业总收入(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_adminexp_to_gr)-annual_lag1y(stock_financial_indicator.q_adminexp_to_gr))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_dtprofit` | field_indicator | 高优 | 扣除非经常损益后的单季度净利润年度投入资本占比 | `asinh(annual(stock_financial_indicator.q_dtprofit)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_dtprofit_to_profit` | field_indicator | 高优 | 扣非净利润／净利润(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_dtprofit_to_profit)-annual_lag1y(stock_financial_indicator.q_dtprofit_to_profit))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_exp_to_sales` | field_indicator | 高优 | 销售期间费用率(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_exp_to_sales)-annual_lag1y(stock_financial_indicator.q_exp_to_sales))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_finaexp_to_gr` | field_indicator | 高优 | 财务费用／营业总收入(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_finaexp_to_gr)-annual_lag1y(stock_financial_indicator.q_finaexp_to_gr))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_gc_to_gr` | field_indicator | 高优 | 营业总成本／营业总收入(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_gc_to_gr)-annual_lag1y(stock_financial_indicator.q_gc_to_gr))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_gr_qoq` | field_indicator | 高优 | 营业总收入环比增长率(%)(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_gr_qoq)-annual_lag1y(stock_financial_indicator.q_gr_qoq))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_gr_yoy` | field_indicator | 高优 | 营业总收入同比增长率(%)(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_gr_yoy)-annual_lag1y(stock_financial_indicator.q_gr_yoy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_gsprofit_margin` | field_indicator | 高优 | 销售毛利率(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_gsprofit_margin)-annual_lag1y(stock_financial_indicator.q_gsprofit_margin))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_impair_to_gr_ttm` | field_indicator | 高优 | 资产减值损失／营业总收入(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_impair_to_gr_ttm)-annual_lag1y(stock_financial_indicator.q_impair_to_gr_ttm))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_investincome` | field_indicator | 高优 | 价值变动单季度净收益年度投入资本占比 | `asinh(annual(stock_financial_indicator.q_investincome)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_investincome_to_ebt` | field_indicator | 高优 | 价值变动净收益／利润总额(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_investincome_to_ebt)-annual_lag1y(stock_financial_indicator.q_investincome_to_ebt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_netprofit_margin` | field_indicator | 高优 | 销售净利率(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_netprofit_margin)-annual_lag1y(stock_financial_indicator.q_netprofit_margin))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_netprofit_qoq` | field_indicator | 高优 | 归母净利润环比增长率(%)(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_netprofit_qoq)-annual_lag1y(stock_financial_indicator.q_netprofit_qoq))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_netprofit_yoy` | field_indicator | 高优 | 归母净利润同比增长率(%)(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_netprofit_yoy)-annual_lag1y(stock_financial_indicator.q_netprofit_yoy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_ocf_to_or` | field_indicator | 高优 | 经营活动现金流净额／经营活动净收益(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_ocf_to_or)-annual_lag1y(stock_financial_indicator.q_ocf_to_or))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_op_qoq` | field_indicator | 高优 | 营业利润环比增长率(%)(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_op_qoq)-annual_lag1y(stock_financial_indicator.q_op_qoq))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_op_to_gr` | field_indicator | 高优 | 营业利润／营业总收入(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_op_to_gr)-annual_lag1y(stock_financial_indicator.q_op_to_gr))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_op_yoy` | field_indicator | 高优 | 营业利润同比增长率(%)(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_op_yoy)-annual_lag1y(stock_financial_indicator.q_op_yoy))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_opincome` | field_indicator | 高优 | 经营活动单季度净收益年度投入资本占比 | `asinh(annual(stock_financial_indicator.q_opincome)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_opincome_to_ebt` | field_indicator | 高优 | 经营活动净收益／利润总额(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_opincome_to_ebt)-annual_lag1y(stock_financial_indicator.q_opincome_to_ebt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_profit_qoq` | field_indicator | 高优 | 净利润环比增长率(%)(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_profit_qoq)-annual_lag1y(stock_financial_indicator.q_profit_qoq))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_saleexp_to_gr` | field_indicator | 高优 | 销售费用／营业总收入(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_saleexp_to_gr)-annual_lag1y(stock_financial_indicator.q_saleexp_to_gr))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_q_salescash_to_or` | field_indicator | 高优 | 销售商品提供劳务收到的现金／营业收入(单季度)年度同期差 | `asinh(annual(stock_financial_indicator.q_salescash_to_or)-annual_lag1y(stock_financial_indicator.q_salescash_to_or))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_rd_exp` | field_indicator | 高优 | 研发费用年度投入资本占比 | `asinh(annual(stock_financial_indicator.rd_exp)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_retained_earnings` | field_indicator | 高优 | 留存收益年度投入资本占比 | `asinh(annual(stock_financial_indicator.retained_earnings)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_retainedps` | field_indicator | 高优 | 每股留存收益年度每股净资产归一化 | `asinh(annual(stock_financial_indicator.retainedps)/annual(stock_financial_indicator.bps))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_revenue_ps` | field_indicator | 高优 | 每股营业收入年度每股净资产归一化 | `asinh(annual(stock_financial_indicator.revenue_ps)/annual(stock_financial_indicator.bps))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_roe_avg` | field_indicator | 高优 | 平均净资产收益率(增发条件)年度同期差 | `asinh(annual(stock_financial_indicator.roe_avg)-annual_lag1y(stock_financial_indicator.roe_avg))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_saleexp_to_gr` | field_indicator | 高优 | 销售费用/营业总收入年度同期差 | `asinh(annual(stock_financial_indicator.saleexp_to_gr)-annual_lag1y(stock_financial_indicator.saleexp_to_gr))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_salescash_to_or` | field_indicator | 高优 | 销售商品提供劳务收到的现金/营业收入年度同期差 | `asinh(annual(stock_financial_indicator.salescash_to_or)-annual_lag1y(stock_financial_indicator.salescash_to_or))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_surplus_rese_ps` | field_indicator | 高优 | 每股盈余公积年度每股净资产归一化 | `asinh(annual(stock_financial_indicator.surplus_rese_ps)/annual(stock_financial_indicator.bps))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_tangible_asset` | field_indicator | 高优 | 有形资产年度投入资本占比 | `asinh(annual(stock_financial_indicator.tangible_asset)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_tangibleasset_to_netdebt` | field_indicator | 高优 | 有形资产/净债务年度同期差 | `asinh(annual(stock_financial_indicator.tangibleasset_to_netdebt)-annual_lag1y(stock_financial_indicator.tangibleasset_to_netdebt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_tax_to_ebt` | field_indicator | 高优 | 所得税/利润总额年度同期差 | `asinh(annual(stock_financial_indicator.tax_to_ebt)-annual_lag1y(stock_financial_indicator.tax_to_ebt))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_total_fa_trun` | field_indicator | 高优 | 固定资产合计周转率年度同期差 | `asinh(annual(stock_financial_indicator.total_fa_trun)-annual_lag1y(stock_financial_indicator.total_fa_trun))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_valuechange_income` | field_indicator | 高优 | 价值变动净收益年度投入资本占比 | `asinh(annual(stock_financial_indicator.valuechange_income)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_fi_working_capital` | field_indicator | 高优 | 营运资金年度投入资本占比 | `asinh(annual(stock_financial_indicator.working_capital)/annual(stock_financial_indicator.invest_capital))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `afx_is_ass_invest_income` | field_income | 高优 | 其中:对联营企业和合营企业的投资收益年度结构占比 | `asinh(annual(stock_income.ass_invest_income)/annual(stock_income.total_revenue))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `afx_is_basic_eps` | field_income | 高优 | 基本每股收益年度每股收益变化率 | `asinh((annual(stock_income.basic_eps)-annual_lag1y(stock_income.basic_eps))/abs(annual_lag1y(stock_income.basic_eps)))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `afx_is_biz_tax_surchg` | field_income | 高优 | 减:营业税金及附加年度结构占比 | `asinh(annual(stock_income.biz_tax_surchg)/annual(stock_income.total_revenue))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `afx_is_comm_exp` | field_income | 高优 | 减:手续费及佣金支出年度结构占比 | `asinh(annual(stock_income.comm_exp)/annual(stock_income.total_revenue))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `afx_is_comm_income` | field_income | 高优 | 手续费及佣金收入年度结构占比 | `asinh(annual(stock_income.comm_income)/annual(stock_income.total_revenue))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `afx_is_compr_inc_attr_m_s` | field_income | 高优 | 归属于少数股东的综合收益总额年度结构占比 | `asinh(annual(stock_income.compr_inc_attr_m_s)/annual(stock_income.total_revenue))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `afx_is_compr_inc_attr_p` | field_income | 高优 | 归属于母公司(或股东)的综合收益总额年度结构占比 | `asinh(annual(stock_income.compr_inc_attr_p)/annual(stock_income.total_revenue))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `afx_is_continued_net_profit` | field_income | 高优 | 持续经营净利润年度结构占比 | `asinh(annual(stock_income.continued_net_profit)/annual(stock_income.total_revenue))` | 2026-04-09 | 2026-04-09 → 2026-09-21 | 1100 | stock_income |
| `afx_is_ebitda` | field_income | 高优 | 息税折旧摊销前利润年度结构占比 | `asinh(annual(stock_income.ebitda)/annual(stock_income.total_revenue))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `afx_is_fin_exp_int_inc` | field_income | 高优 | 财务费用:利息收入年度结构占比 | `asinh(annual(stock_income.fin_exp_int_inc)/annual(stock_income.total_revenue))` | 2019-02-22 | 2019-02-22 → 2026-09-21 | 1100 | stock_income |
| `afx_is_forex_gain` | field_income | 高优 | 加:汇兑净收益年度结构占比 | `asinh(annual(stock_income.forex_gain)/annual(stock_income.total_revenue))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `afx_is_int_exp` | field_income | 高优 | 减:利息支出年度结构占比 | `asinh(annual(stock_income.int_exp)/annual(stock_income.total_revenue))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `afx_is_int_income` | field_income | 高优 | 利息收入年度结构占比 | `asinh(annual(stock_income.int_income)/annual(stock_income.total_revenue))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `afx_is_n_oth_b_income` | field_income | 高优 | 加:其他业务净收益年度结构占比 | `asinh(annual(stock_income.n_oth_b_income)/annual(stock_income.total_revenue))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `afx_is_oth_compr_income` | field_income | 高优 | 其他综合收益年度结构占比 | `asinh(annual(stock_income.oth_compr_income)/annual(stock_income.total_revenue))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `afx_is_other_bus_cost` | field_income | 高优 | 其他业务成本年度结构占比 | `asinh(annual(stock_income.other_bus_cost)/annual(stock_income.total_revenue))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_income |
| `amihud_asymmetry_20` | liquidity | 低优 | 涨跌两侧的非流动性差：下跌日 Amihud / 上涨日 Amihud | `up_illiq = illiq.where(ret > 0).rolling(20, min_periods=5).mean(); down_illiq = illiq.where(ret < 0).rolling(20, min_periods=5).mean(); asym = safe_divide(down_illiq, up_illiq + 1e-12)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `amihud_daily_5` | liquidity | 高优 | Amihud 日频非流动性：5 日平均 \|收益\|/成交额（短周期版） | `amihud = \|ret\| / amount; amihud5 = roll(amihud, 5, "mean", min_periods=2)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | stock_daily, stock_adj_factor |
| `amihud_parkinson_ratio` | coupling | 低优 | 非流动性 / 波动率 = 20 日 Amihud ÷ 20 日 Parkinson 波动 | `Amihud = roll_mean(\|ret\| / amount, 20);
Parkinson = sqrt( roll_mean( ln(high/low)^2, 20) / (4 ln 2) );
Ratio = Amihud / Parkinson` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 200 | stock_daily, stock_adj_factor |
| `amount_mom_accel` | liquidity | 高优 | 成交额动量加速：3 日变化率 − 10 日变化率 | `p3 = amount.pct_change(3); p10 = amount.pct_change(10); vals = p3 - p10` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_daily |
| `amount_ratio_20` | liquidity | 低优 | 相对成交额：当日成交额 / 20 日均成交额 − 1 | `avg_amount = rolling(20, min_periods=10).mean(); a_ratio = amount / avg_amount.replace(0, np.nan) - 1.0` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily |
| `ar_ap_to_revenue` | quality | 高优 | 净预收款占收入比 = (预收款项 − 预付款项) / 营业收入(TTM) | `Ratio = (AdvanceReceipts - AdvancePayment) / OperatingRevenue` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_balancesheet, stock_income |
| `aroon_down_25` | technical | 低优 | Aroon 下行：25 日窗口内「距最近新低的交易日数」的位置，越接近新低越低 | `wide = _adjusted_close(daily).unstack("Code")
days_since = _rolling_extreme_age(wide, window=25, find_max=False)
aroon = safe_divide(25.0 - days_since, 25.0) * 100.0` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 70 | stock_daily, stock_adj_factor |
| `aroon_up_25` | technical | 高优 | Aroon 上行：25 日窗口内「距最近新高的交易日数」的位置，越接近新高越高 | `wide = _adjusted_close(daily).unstack("Code")
days_since = _rolling_extreme_age(wide, window=25, find_max=True)
aroon = safe_divide(25.0 - days_since, 25.0) * 100.0` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 70 | stock_daily, stock_adj_factor |
| `asset_growth_qoq` | growth | 低优 | 总资产环比增速（相对上一个报告期） | `Growth = TotalAssets_t / TotalAssets_{t-1Q} - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_balancesheet |
| `atr_14_ratio` | technical | 低优 | 相对波幅 = ATR14 / 后复权收盘价（已归一化，无量纲） | `close = daily["close"]
high = daily["high"]
low = daily["low"]
# ATR 折算到复权空间(×scale)后与复权基座 adj 同口径,避免除权日 close
# 跳变造成相对波幅虚高
scale = _adjusted_close(daily) / close.replace(0, np.nan)
adj = _adjusted_close(daily)

tr1 = high - low
tr2 = (high - daily["pre_close"]).abs()
tr3 = (low - daily["pre_close"]).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
atr = (tr * scale).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)

atr_pct = safe_divide(atr, adj + 1e-10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 48 | stock_daily, stock_adj_factor |
| `avg_cost_premium` | chip | 高优 | 平均成本溢价 = (现价 − 筹码加权均价) / 筹码加权均价 | `premium = safe_divide(close_adj - weight_avg, weight_avg + 1e-10); premium = premium.clip(-1, 5); return cross_sectional_rank(premium)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips, stock_daily |
| `beta_60` | risk | 低优 | 60 日市场贝塔（对沪深300，反向） | `beta = _rolling_beta(wide, mkt, 60, 30)  # Beta = Cov(StockReturn, IndexReturn) / Var(IndexReturn) over a rolling window` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_daily, stock_adj_factor, index_daily |
| `bias_20` | technical | 高优 | 20 日乖离率 = close/MA20 − 1 | `# 跨日 MA 窗口走复权基座,未复权 close 在除权日跳变会伪造负乖离
adj = _adjusted_close(daily_panel)
ma_20 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
bias = adj / ma_20.replace(0, np.nan) - 1.0` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `big_vs_small_divergence_5d` | fundflow | 高优 | 大小单背离的 5 日变化 = Δ5(大单净买率 − 小单净买率) | `big_net = (ff["buy_lg_amount"] + ff["buy_elg_amount"] - ff["sell_lg_amount"] - ff["sell_elg_amount"]) / _total_amount(ff); small_net = (ff["buy_sm_amount"] - ff["sell_sm_amount"]) / _total_amount(ff); divergence = big_net - small_net; div_5d = divergence.diff(5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 38 | stock_main_fund_flow, stock_daily |
| `bollinger_squeeze` | technical | 低优 | 布林带宽的 250 日**历史分位**（∈[0,1]）：低 = 波动压缩，变盘前夜 | `# 跨日 MA/std 窗口走复权基座,未复权 close 在除权日跳变会污染带宽
adj = _adjusted_close(daily_panel)
ma_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean())
std_20 = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).std())
bandwidth = 4 * std_20 / ma_20.replace(0, np.nan)
# Rank negative: narrow band = squeeze = ranked high
# 本项目口径：squeeze = ctx.roll_rank(bandwidth, 250)（带宽自身的历史分位），见 note` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 500 | stock_daily, stock_adj_factor |
| `bollinger_width_20` | technical | 低优 | 20 日布林带宽 = 4×std/ma（已按价格归一化，无量纲） | `# 跨日 MA/std 窗口走复权基座,未复权 close 在除权日跳变会污染带宽
adj = _adjusted_close(daily)
ma = rolling_group_mean(adj, 20)
std = rolling_group_std(adj, 20)
width = safe_divide(4 * std, ma)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `bp` | value | 高优 | 账面市值比 = 归母股东权益 / 总市值（高 = 价值股） | `bp = 1.0 / finance["pb"].replace(0, np.nan)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_balancesheet, stock_daily, stock_finance |
| `bs_construction_capital_share` | financial_detail | 低优 | 在建工程占固定投入资本 | `cip_total/(fix_assets+cip_total)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_balancesheet |
| `bs_intangible_asset_share` | financial_detail | 低优 | 无形资产占总资产 | `intan_assets/total_assets` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_balancesheet |
| `bs_near_term_debt_share` | financial_detail | 低优 | 近端债务占主要有息债务 | `(st_borr+non_cur_liab_due_1y)/(st_borr+non_cur_liab_due_1y+lt_borr+bond_payable)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_balancesheet |
| `bs_net_contract_to_assets` | financial_detail | 低优 | 净合同资产占总资产 | `(contract_assets-contract_liab)/total_assets` | 2020-05-01 | 2020-05-06 → 2026-09-21 | 700 | stock_balancesheet |
| `bs_net_notes_to_assets` | financial_detail | 低优 | 净应收票据占资产 | `(notes_receiv-notes_payable)/total_assets` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_balancesheet |
| `bs_other_receiv_to_assets` | financial_detail | 低优 | 其他应收款占总资产 | `oth_receiv/total_assets` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_balancesheet |
| `bw_beta_60` | breadth | 低优 | 个股收益对市场宽度变化的 60 日 β（市场参与度暴露） | `beta = roll_cov(ret, delta_up_share, 60, 30) / roll_var(delta_up_share, 60, 30)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_market_distribution_history, stock_daily, stock_adj_factor |
| `bw_beta_asym_20` | breadth | 高优 | 上行宽度日 β − 下行宽度日 β（参与度的不对称） | `b_up - b_down, 两腿各自用互斥的 NaN 掩码在 20 日窗内回归` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_market_distribution_history, stock_daily, stock_adj_factor |
| `bw_beta_change_20` | breadth | 高优 | 宽度 β 的短期变化 = β20 − β60（参与度暴露的抬升） | `beta20 = rolling_beta(20,10); beta60 = rolling_beta(60,30); change = beta20 - beta60` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_market_distribution_history, stock_daily, stock_adj_factor |
| `bw_breadthvol_response_20` | breadth | 低优 | \|个股收益\| 与 当日市场内部翻腾度 的 20 日相关（脆弱性） | `corr(abs(ret), bw_vol, 20, 10)，bw_vol = 当日分钟内 up 份额的 std` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_market_distribution_history, stock_daily, stock_adj_factor |
| `bw_capture_asym_20` | breadth | 低优 | 市场收敛日的捕获 / 市场扩散日的捕获（防御性） | `stock_on_down = mean(ret \| d_sh<0, 20d); stock_on_up = mean(ret \| d_sh>0, 20d); sens = \|down\| / \|up\|` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_market_distribution_history, stock_daily, stock_adj_factor |
| `bw_intraday_beta_20` | breadth | 高优 | 日内腿的宽度 β = 个股日内收益 对 市场日内宽度变化 的 20 日 β | `cov(close/open-1, sh_close-sh_open, 20, 10) / var(sh_close-sh_open, 20, 10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_market_distribution_history, stock_daily, stock_adj_factor, stock_history_5min |
| `bw_overnight_lag_beta_20` | breadth | 高优 | 隔夜跳空 与 昨日宽度变化 的 20 日相关（滞后反应） | `corr(gap_T, delta_up_share_{T-1}, 20, 10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_market_distribution_history, stock_daily, stock_adj_factor |
| `bw_resid_beta_60` | breadth | 低优 | 剔掉沪深300 β 之后的宽度 β（正交的参与度暴露） | `e = ret - beta_idx*ret_idx; resid_beta = roll_cov(e, d_sh, 60, 30)/roll_var(d_sh, 60, 30)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_market_distribution_history, stock_daily, stock_adj_factor, index_daily |
| `bw_session_follow_20` | breadth | 高优 | 上/下午「与市场宽度同向」频率之差（日内跟随的市场一致性） | `am: sign(am_ret)==sign(d_sh_am); pm: sign(pm_ret)==sign(d_sh_pm); delta = mean(am,20,10) - mean(pm,20,10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_market_distribution_history, stock_daily, stock_adj_factor, stock_history_5min |
| `bw_strength_sensitivity_20` | breadth | 高优 | 个股收益 与 全市场「涨超 5% 占比」的 20 日相关（对强势情绪的敏感度） | `up = (up_5_to_7+up_7_to_10+up_over_10)/total; corr(ret, up, 20, 10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_market_distribution_history, stock_daily, stock_adj_factor |
| `bw_tail_comove_60` | breadth | 低优 | 与宽度的尾部共振频率（双方 \|z\|>1.5 且同向的交易日占比，60 日） | `z = (x - rolling_mean)/rolling_std; freq(\|z_r\|>1.5 & \|z_m\|>1.5 & sign一致) 的 60 日均值` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_market_distribution_history, stock_daily, stock_adj_factor |
| `bw_weak_breadth_ret_20` | breadth | 高优 | 市场偏弱日（上涨家数 < 半数）的条件收益 − 自身 20 日均值 | `weak = sh_close < 0.5; cond = mean(ret \| weak, 20) - mean(ret, 20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_market_distribution_history, stock_daily, stock_adj_factor |
| `cash_conversion_cycle` | quality | 低优 | 现金转换周期（天）= 存货周转天数 + 应收周转天数 − 应付周转天数（低优） | `CCC = DIO + DSO - DPO = 365×Inventories/OperatingCost_TTM + 365×AccountsReceivable/OperatingRevenue_TTM - 365×AccountPayable/OperatingCost_TTM` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income, stock_balancesheet |
| `cash_sales_ratio` | quality | 高优 | 销售收现比 = 销售商品收到的现金(TTM) / 营业收入(TTM) | `ratio = c_fr_sale_sg_TTM / revenue_TTM` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income, stock_cashflow |
| `cf_borrowing_repayment_ratio` | financial_detail | 低优 | 借款流入对偿债现金的覆盖 | `TTM(c_recp_borrow)/TTM(c_prepay_amt_borr)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_cashflow |
| `cf_distribution_cash_coverage` | financial_detail | 低优 | 分红与利息现金支付占经营现金流 | `TTM(c_pay_dist_dpcp_int_exp)/abs(TTM(n_cashflow_act))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_cashflow |
| `cf_net_borrowing_to_assets` | financial_detail | 低优 | 净借款现金流占总资产 | `(TTM(c_recp_borrow)-TTM(c_prepay_amt_borr))/total_assets` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_cashflow, stock_balancesheet |
| `cf_purchase_cash_intensity` | financial_detail | 低优 | 采购现金占收入 | `TTM(c_paid_goods_s)/TTM(revenue)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_cashflow, stock_income |
| `cf_tax_cash_burden` | financial_detail | 低优 | 现金税费占收入 | `TTM(c_paid_for_taxes)/TTM(revenue)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_cashflow, stock_income |
| `cf_tax_refund_share` | financial_detail | 高优 | 税费返还占营业收入 | `TTM(recp_tax_rends)/TTM(revenue)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_cashflow, stock_income |
| `cfcr` | quality | 高优 | 现金流利息保障倍数 = 经营现金流(TTM) / 利息支出(TTM) | `CFCR = NetOperateCashFlow_TTM / InterestExpense_TTM` | 2019-05-01 | 2019-05-06 → 2026-09-21 | 700 | stock_cashflow, stock_income |
| `cfp_ttm` | value | 高优 | 经营现金流市值比 = 经营活动现金流净额TTM / 总市值 | `ocf_to_market = NetOperateCashFlow_TTM / MarketCap` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_cashflow, stock_daily, stock_finance |
| `chip_above_below_ratio` | chip | 低优 | 筹码压力比 = (现价 − p75) / (p25 − 现价)（正 ⟺ 现价落在 p25~p75 带内，绝对值越大越靠近 p25） | `close_adj = _close_adj_basis(daily); cost_85 = cyq["cost_85pct"]; cost_15 = cyq["cost_15pct"]; above = close_adj.loc[common] - cost_85.loc[common]; below = cost_15.loc[common] - close_adj.loc[common]; ratio = safe_divide(above, below); return cross_sectional_rank(-ratio)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips, stock_daily |
| `chip_concentration` | chip | 低优 | 筹码集中度（80% 筹码的相对宽度）= (p90 − p10) / p50 | `spread = (perf["cost_95pct"] - perf["cost_5pct"]) / perf["cost_50pct"]; return cross_sectional_rank(-spread)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips |
| `chip_concentration_change_20d` | chip | 高优 | 筹码集中度的 20 日变化（正=区间收窄=筹码凝聚） | `concentration = -(cost_95pct - cost_5pct) / cost_50pct; chg = concentration.groupby(level="Code").transform(lambda s: s.diff(20)); return cross_sectional_rank(chg)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 60 | stock_cyq_chips |
| `chip_cost_convergence_20d` | chip | 高优 | 筹码成本收敛 = (p75 − p10)/p50 的 20 日变化的相反数（正=宽度收敛） | `width = safe_divide(cost_85pct - cost_5pct, cost_50pct); chg = width.groupby(level="Code").transform(lambda s: s.diff(20)); return cross_sectional_rank(-chg)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 60 | stock_cyq_chips |
| `chip_cv_factor` | chip | 低优 | 筹码变异系数 = std / mean（低=相对离散度小=成本一致性强） | `s = _compute_chip_factor(..., "chip_cv"); return cross_sectional_rank(-s)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips |
| `chip_deep_trap_ratio` | chip | 低优 | 深套筹码占比 = 成本 > 1.1×现价 的筹码比例（高=上方深度套牢盘重） | `s = _chip_series(context, "chip_upper_110", need_close=True); return cross_sectional_rank(-s)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips, stock_daily |
| `chip_gini_factor` | chip | 高优 | 筹码基尼系数（高=筹码集中在少数价位=价格锚定清晰） | `s = _compute_chip_factor(..., "chip_gini"); return cross_sectional_rank(s)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips |
| `chip_high_float_ratio` | chip | 低优 | 高浮盈筹码占比 = 成本 < 0.9×现价 的筹码比例（高=获利丰厚、兑现压力大） | `s = _chip_series(context, "chip_below_90", need_close=True); return cross_sectional_rank(-s)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips, stock_daily |
| `chip_median_distance` | chip | 高优 | 中位数成本距离 = (现价 − p50) / 现价（正=过半持仓者盈利） | `median_series = _compute_chip_factor(..., "chip_median_price"); close = daily_panel["close"]; common = close.index.intersection(median_series.index); distance = (close.loc[common] - median_series.loc[common]) / close.loc[common].replace(0, np.nan); return cross_sectional_rank(distance)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips, stock_daily |
| `chip_p90_p10_factor` | chip | 低优 | 90% 筹码价格区间宽度 = p90 − p10（绝对价差） | `s = _compute_chip_factor(..., "chip_p90_p10"); return cross_sectional_rank(-s)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips |
| `chip_peak_distance` | chip | 高优 | 主峰距离 = (现价 − 众数价) / 现价（正=现价在最大筹码峰上方=有支撑） | `peak_series = _compute_chip_factor(..., "chip_peak_price"); close_adj = _close_adj_basis(daily_panel); common = close_adj.index.intersection(peak_series.index); distance = (close_adj.loc[common] - peak_series.loc[common]) / close_adj.loc[common].replace(0, np.nan); return cross_sectional_rank(distance)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips, stock_daily |
| `chip_peak_growing` | chip | 高优 | 主峰增强 = 主峰纯度（peak_purity）的 5 个交易日变化（升=筹码向核心价位凝聚） | `s = _compute_chip_factor(..., "chip_peak_dominance"); chg = s.groupby(level="Code").transform(lambda x: x.diff(5)); return cross_sectional_rank(chg)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 30 | stock_cyq_chips |
| `chip_position` | chip | 低优 | 现价在筹码分布中的位置 = (现价 − p10) / (p90 − p10) | `position = (close_adj - cost_5pct) / (cost_95pct - cost_5pct); return cross_sectional_rank(-position)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips, stock_daily |
| `chip_range_normalized` | chip | 低优 | 归一化筹码区间（中间 50% 筹码的相对宽度）= (p75 − p25) / p50 | `spread = (perf["cost_85pct"] - perf["cost_15pct"]) / perf["cost_50pct"]; return cross_sectional_rank(-spread)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips |
| `chip_tail_risk` | chip | 低优 | 筹码尾部风险 = 成本分布的超额峰度（高=极端价位筹码堆积=肥尾） | `s = _compute_chip_factor(..., "chip_kurtosis"); return cross_sectional_rank(-s)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips |
| `chip_win_peak_frac` | chip | 高优 | 获利筹码峰集中度 = 现价下方最大单档占比 / 下方筹码总量（高=获利盘锁筹集中） | `s = _chip_series(context, "chip_win_peak_frac", need_close=True); return cross_sectional_rank(s)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips, stock_daily |
| `close_location_20d` | technical | 高优 | 收盘位置 = 20 日收益 / 20 日振幅，度量上涨的「上攻效率」 | `h20 = roll(df, "high", 20, "max")
l20 = roll(df, "low", 20, "min")
chg20 = df.groupby("Code")["close"].shift(20)
vals = (df["close"] - chg20) / (h20 - l20 + 1e-8)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `consecutive_limit_down` | event | 低优 | 当前连续跌停天数（连续段逻辑：断段即归零，非跌停日为 0） | `is_ld = daily["pct_chg"].le(-9.8).astype(int)
code = is_ld.index.get_level_values("Code")
seg = (~is_ld.astype(bool)).groupby(level="Code").cumsum()
count = is_ld.groupby([code, seg]).cumsum()      # 参考库原文
# 本实现：同一「连续段」语义（断段归零），向量化为
#   count = 到 t 为止连续跌停的天数（np.maximum.accumulate 求最近一次 0 的位置）` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_limit_list, stock_daily, stock_adj_factor |
| `consecutive_limit_up` | event | 高优 | 当前连板数（连续涨停天数），断板即归零；非涨停日为 0 | `is_lu = daily["pct_chg"].ge(9.8).astype(int)
code = is_lu.index.get_level_values("Code")
seg = (~is_lu.astype(bool)).groupby(level="Code").cumsum()
count = is_lu.groupby([code, seg]).cumsum()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 29 | stock_limit_up |
| `cost_convergence_signal` | chip | 高优 | 成本收敛信号 = (p90 − p10) 的 20 个交易日**变化率**（收敛=筹码向成本中枢凝聚） | `cyq = context.load("cyq_perf.parquet"); width = cyq["cost_95pct"] - cyq["cost_5pct"]; chg = width.groupby(level="Code").transform(lambda s: s.pct_change(20, fill_method=None)); chg = chg.clip(-1, 1); return cross_sectional_rank(-chg)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 60 | stock_cyq_chips |
| `cost_distribution_skew` | chip | 高优 | 成本分布偏度（分位点口径）= (p50 − p25) / (p75 − p50)（<1 = 上方尾部更长/右偏） | `cyq = context.load("cyq_perf.parquet"); left_tail = cyq["cost_50pct"] - cyq["cost_15pct"]; right_tail = cyq["cost_85pct"] - cyq["cost_50pct"]; skew = left_tail / right_tail.replace(0, np.nan); return cross_sectional_rank(skew)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips |
| `cost_skew_ratio` | chip | 高优 | 成本偏度比率 = (p50 − p10) / (p90 − p50)（>1 = 下方尾部更长/左偏） | `cyq = context.load("cyq_perf.parquet"); lower_range = cyq["cost_50pct"] - cyq["cost_5pct"]; upper_range = cyq["cost_95pct"] - cyq["cost_50pct"]; skew = safe_divide(lower_range, upper_range + 1e-10); skew = skew.clip(0.1, 10); return cross_sectional_rank(skew)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips |
| `cp_bigflow_margin_20` | coupling | 高优 | 大单成本 × 两融加速 = z(大单成交均价偏离) × z(融资余额速度) | `CP = z(mf_large_order_avg_price) × z(margin_velocity)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 320 | mf_large_order_avg_price, margin_velocity |
| `cp_chip_support_reversal_5` | coupling | 高优 | 筹码位置 × 短期反转 = z(收盘相对筹码分布位置) × z(5日反转) | `CP = z(chip_position) × z(short_term_reversal_5)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 200 | chip_position, short_term_reversal_5 |
| `cp_chip_turnover` | coupling | 低优 | 筹码集中 × 换手波动 = z(筹码集中度) × z(换手率波动) | `CP = z(chip_concentration) × z(turnover_std_20)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 200 | chip_concentration, turnover_std_20 |
| `cp_margin_trend_div` | coupling | 高优 | 两融加速 − 趋势强度 = z(融资余额速度) − z(60日趋势强度) | `CP = z(margin_velocity) − z(trend_strength_60)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 320 | margin_velocity, trend_strength_60 |
| `cp_momentum_highvol_60` | coupling | 高优 | 进攻型动量 = z(60日动量) × z(120日波动) | `CP = z(momentum_60) × z(vol_120)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 320 | momentum_60, vol_120 |
| `cp_momentum_lowvol_20` | coupling | 高优 | 防守型动量 = z(20日动量) × z(−特质波动) | `CP = z(momentum_20) × z(−idio_vol_60)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 320 | momentum_20, idio_vol_60 |
| `cp_moneyflow_momentum_res` | coupling | 低优 | 资金流结构 × 动量 = z(订单规模熵) × z(20日动量) | `CP = z(mf_order_size_entropy) × z(momentum_20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 320 | mf_order_size_entropy, momentum_20 |
| `cp_quality_momentum` | coupling | 高优 | 质量动量共振 = z(ROE TTM) × z(60日动量) | `CP = z(roe_ttm) × z(momentum_60)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | roe_ttm, momentum_60 |
| `cp_rsi_moneyflow_res` | coupling | 低优 | RSI × 主力净流入 = z(RSI14) × z(净流入占比) | `CP = z(rsi_14) × z(mf_net_inflow_ratio)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 200 | rsi_14, mf_net_inflow_ratio |
| `cp_value_momentum_div` | coupling | 高优 | 价值−动量分歧 = z(BP) − z(20日动量) | `CP = z(bp) − z(momentum_20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 320 | bp, momentum_20 |
| `cp_value_quality` | coupling | 高优 | 价值质量共振 = z(账面市值比 BP) × z(ROE TTM) | `CP = z(bp) × z(roe_ttm)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | bp, roe_ttm |
| `cvar_95_120` | risk | 高优 | 120 日 CVaR(95%) = 窗口内 ≤5% 分位那部分收益的均值（尾部期望损失） | `quantiles = [ret_w.rolling(120, min_periods=60).quantile(q) for q in (0.01, 0.02, 0.03, 0.04, 0.05)]; cvar_w = sum(quantiles) / len(quantiles)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 240 | stock_daily, stock_adj_factor |
| `cyqp_cost_premium_change_20` | chip_perf | 高优 | 现价相对筹码中位成本溢价的 20 日变化 | `diff(close/cost_50pct - 1,20)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 80 | stock_cyq_perf, stock_daily |
| `cyqp_cost_tail_asymmetry` | chip_perf | 低优 | 上下成本尾部不对称（扩展） | `(cost_95pct+cost_5pct-2*cost_50pct)/(cost_95pct-cost_5pct)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 80 | stock_cyq_perf |
| `cyqp_cost_width_70` | chip_perf | 低优 | 70% 筹码成本相对宽度 | `(cost_85pct-cost_15pct)/cost_50pct` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 80 | stock_cyq_perf |
| `cyqp_cost_width_90` | chip_perf | 低优 | 90% 筹码成本相对宽度 | `(cost_95pct-cost_5pct)/cost_50pct` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 80 | stock_cyq_perf |
| `cyqp_cost_width_change_20` | chip_perf | 低优 | 90% 成本宽度 20 日变化 | `diff((cost_95pct-cost_5pct)/cost_50pct,20)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 80 | stock_cyq_perf |
| `cyqp_historical_range_position` | chip_perf | 高优 | 中位成本在供应商历史价格区间的位置（扩展） | `(cost_50pct-his_low)/(his_high-his_low)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 80 | stock_cyq_perf |
| `cyqp_mean_median_gap` | chip_perf | 低优 | 筹码均值与中位数成本偏离（扩展） | `weight_avg/cost_50pct - 1` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 80 | stock_cyq_perf |
| `cyqp_tail_width_share` | chip_perf | 低优 | 两端尾部占 90% 区间的宽度比例（扩展） | `((cost_95pct-cost_85pct)+(cost_15pct-cost_5pct))/(cost_95pct-cost_5pct)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 80 | stock_cyq_perf |
| `cyqp_winner_acceleration_5` | chip_perf | 低优 | 获利盘占比 5 日二阶差分 | `diff(diff(winner_fraction,5),5)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 80 | stock_cyq_perf |
| `cyqp_winner_change_20` | chip_perf | 高优 | 获利盘占比 20 日变化 | `winner_fraction(T) - winner_fraction(T-20)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 80 | stock_cyq_perf |
| `cyqp_winner_fraction` | chip_perf | 低优 | 供应商获利盘占比 | `winner_rate / 100` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 80 | stock_cyq_perf |
| `cyqp_winner_volatility_20` | chip_perf | 低优 | 获利盘占比 20 日波动 | `std(winner_fraction,20)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 80 | stock_cyq_perf |
| `di_plus_minus_ratio_14` | technical | 高优 | 14 日 DI+/DI- 比率：多头相对空头的趋势优势，>1 = 上升趋势占优 | `scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
di_plus, di_minus = _directional_movement(daily["high"], daily["low"], daily["pre_close"], scale, 14)
ratio = safe_divide(di_plus, di_minus + 1e-8)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 300 | stock_daily, stock_adj_factor |
| `dividend_yield_3y_avg` | value | 高优 | 3 年平均股息率 = 近 735 个交易日的（每股分红 ÷ 当期价）均值 / 当前价（%） | `dividend_yield_3y_avg = (SUM(ActualCashDiviRMB, 735) / 3) / ClosePrice
   = AVG(ActualCashDiviRMB, 3年) / ClosePrice` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1400 | stock_daily, stock_finance |
| `donchian_position_20` | technical | 高优 | Donchian 通道位置 ∈[0,1]：(close − 20日最低)/(20日最高 − 20日最低) | `close = daily["close"]
# 用每日复权系数 (后复权基座/close) 折算 high/low 后再取 20 日极值,
# 避免除权日污染通道上下轨
scale = _adjusted_close(daily) / close.replace(0, np.nan)
adj_high = daily["high"] * scale
adj_low = daily["low"] * scale
adj = _adjusted_close(daily)

previous_high = adj_high.groupby(level="Code").shift(1)
highest = previous_high.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).max()
)
lowest = adj_low.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).min()
)

position = safe_divide(adj - lowest, highest - lowest + 1e-10)
position = position.clip(0, 1)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `downside_upside_vol_60` | risk | 低优 | 下行/上行波动比 = 60 日负收益标准差 / 正收益标准差（反向） | `down = ret.where(ret < 0); up = ret.where(ret > 0); down_std = down.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=5).std()); up_std = up.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=5).std()); ratio = safe_divide(down_std, up_std + 1e-10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_daily, stock_adj_factor |
| `downside_vol_ratio_20` | risk | 低优 | 下行波动占比 = 20 日下行半波动 / 总波动（反向） | `neg = wide.clip(upper=0.0); var_all = (wide ** 2).rolling(20, min_periods=15).mean(); var_neg = (neg ** 2).rolling(20, min_periods=15).mean(); ratio = var_neg.pow(0.5) / var_all.pow(0.5).replace(0, np.nan)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_daily, stock_adj_factor |
| `dp_ttm` | value | 高优 | 滚动股息率（%）= 供应商 dv_ttm | `dp = finance["dv_ttm"]` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_finance |
| `dpo_20` | technical | 高优 | 20 日 DPO 去趋势：(11 日前收盘 − 当前 20 日均价)/20 日均价 | `adj = _adjusted_close(daily)
ma20 = adj.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
lagged = adj.groupby(level="Code").shift(11)
dpo = safe_divide(lagged - ma20, ma20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 82 | stock_daily, stock_adj_factor |
| `dragon_tiger_org_net_20` | event | 高优 | 机构席位净买 = 20 日「机构专用」席位净买额 / 20 日成交额（无机构席位则 NaN） | `OrgNet_20 = sum(OrgNetAmount, 20d) / sum(Amount, 20d)
OrgNetAmount = buy_amount - sell_amount, org_name == '机构专用'
Amount = 该股当日总成交额（stock_daily.amount，元）` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_dragon_tiger, stock_daily |
| `drawdown_duration_120` | risk | 低优 | 回撤持续期 = 价格低于 120 日滚动前高的连续天数（上限 120） | `rolling_high = adj.groupby(level="Code").transform(lambda s: s.rolling(120, min_periods=1).max()); in_dd = adj.lt(rolling_high * 0.999); duration = _consecutive_count(in_dd).clip(upper=120)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 240 | stock_daily, stock_adj_factor |
| `dv_stability_4q` | value | 低优 | 股息稳定性 = 60 交易日 dv_ttm 的变异系数（低 = 稳定，取负向） | `dv = fin['dv_ratio'].clip(0, 20)
roll_std = dv.groupby(level='Code').transform(
    lambda s: s.rolling(60, min_periods=20).std())
roll_mean = dv.groupby(level='Code').transform(
    lambda s: s.rolling(60, min_periods=20).mean())
cv = safe_divide(roll_std, roll_mean + 1e-10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 480 | stock_finance |
| `ebitda_to_market` | value | 高优 | EBITDA 市值比 = (EBIT + 折旧 + 无形资产摊销 + 长期待摊摊销)TTM / 总市值 | `ebitda_to_market = EBITDA / (ClosePrice × TotalShares)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_cashflow, stock_income, stock_balancesheet, stock_daily, stock_finance |
| `efx_annual_earnings_yield` | field_events | 高优 | 静态正盈利收益率 | `mean_20(1/pe where pe>0, observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_finance |
| `efx_dividend_gap` | field_events | 高优 | 静态与滚动股息率差 | `mean_20(dv_ratio-dv_ttm, observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_finance |
| `efx_dragon_net_intensity` | field_events | 高优 | 营业部净买入强度 | `mean_60(median(net_buy_amount/(buy_amount+sell_amount)), observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_dragon_tiger |
| `efx_dragon_ratio_balance` | field_events | 高优 | 营业部买卖占比差 | `mean_60(median(buy_ratio-sell_ratio across disclosed seats), observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_dragon_tiger |
| `efx_forecast_revision_delay` | field_events | 高优 | 业绩预告距首次披露的间隔 | `mean_120(calendar_days(ann_date-first_ann_date), available at ann_date, observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_forecast |
| `efx_free_turnover` | field_events | 高优 | 自由流通口径换手溢价 | `mean_20(turnover_rate_f-turnover_rate, observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_finance |
| `efx_limit_first` | field_events | 高优 | 首次触板时刻 | `mean_60(trading_minutes(first_time)/240, observed rows only)` | 2020-01-01 | 2020-01-02 → 2026-09-21 | 220 | stock_limit_list |
| `efx_limit_float_fraction` | field_events | 高优 | 触板股票流通市值比例 | `mean_60(float_mv/total_mv, observed rows only)` | 2020-01-01 | 2020-01-02 → 2026-09-21 | 220 | stock_limit_list |
| `efx_limit_openings` | field_events | 高优 | 触板后开板次数 | `mean_60(log1p(open_times), observed rows only)` | 2020-01-01 | 2020-01-02 → 2026-09-21 | 220 | stock_limit_list |
| `efx_limit_reseal_span` | field_events | 高优 | 首次至最后触板交易时间差 | `mean_60((last_time-first_time)/240 trading minutes, observed rows only)` | 2020-01-01 | 2020-01-02 → 2026-09-21 | 220 | stock_limit_list |
| `efx_limit_signed_move` | field_events | 高优 | 触板日涨跌幅 | `mean_60(pct_chg for known U/D/Z event types, observed rows only)` | 2020-01-01 | 2020-01-02 → 2026-09-21 | 220 | stock_limit_list |
| `efx_limit_streak` | field_events | 高优 | 连板天数暴露 | `mean_60(limit_times, observed rows only)` | 2020-01-01 | 2020-01-02 → 2026-09-21 | 220 | stock_limit_list |
| `efx_limit_trade_share` | field_events | 高优 | 涨跌停价成交金额占比 | `mean_60(limit_amount/amount, observed rows only)` | 2020-01-01 | 2020-01-02 → 2026-09-21 | 220 | stock_limit_list |
| `efx_limit_turnover` | field_events | 高优 | 触板日换手率 | `mean_60(turnover_ratio, observed rows only)` | 2020-01-01 | 2020-01-02 → 2026-09-21 | 220 | stock_limit_list |
| `efx_limit_win_fraction` | field_events | 高优 | 近期涨停次数密度 | `mean_60(up_stat numerator / denominator, observed rows only)` | 2020-01-01 | 2020-01-02 → 2026-09-21 | 220 | stock_limit_list |
| `efx_lu_board_density` | field_events | 高优 | 多日连板密度 | `mean_60(boards_count / days parsed from boards; first-board text missing, observed rows only)` | 2019-08-14 | 2019-08-14 → 2026-09-21 | 220 | stock_limit_up |
| `efx_lu_final` | field_events | 高优 | 涨停最终封板时刻 | `mean_60(trading_minutes(final_limit_time)/240, observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_limit_up |
| `efx_lu_first` | field_events | 高优 | 涨停首次封板时刻 | `mean_60(trading_minutes(first_limit_time)/240, observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_limit_up |
| `efx_lu_move` | field_events | 高优 | 涨停日涨幅暴露 | `mean_60(change_percent, observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_limit_up |
| `efx_lu_one_price` | field_events | 高优 | 一字涨停类型比例 | `mean_60(contains_one_price(limit_type) among known text events, observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_limit_up |
| `efx_lu_seal_amount` | field_events | 高优 | 封单金额规模 | `mean_60(log1p(sealed_amount in source units), observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_limit_up |
| `efx_lu_seal_volume` | field_events | 高优 | 封单数量规模 | `mean_60(log1p(sealed_volume in source units), observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_limit_up |
| `efx_seal_float` | field_events | 高优 | 封单金额流通市值比 | `mean_60(fd_amount/float_mv, observed rows only)` | 2020-01-01 | 2020-01-02 → 2026-09-21 | 220 | stock_limit_list |
| `efx_top_amount_rate` | field_events | 高优 | 龙虎榜成交占比原始口径 | `mean_60(amount_rate, observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_top_list |
| `efx_top_float` | field_events | 高优 | 上榜股票流通市值暴露 | `mean_60(log1p(float_values), observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_top_list |
| `efx_top_imbalance` | field_events | 高优 | 龙虎榜买卖不平衡 | `mean_60((l_buy-l_sell)/(l_buy+l_sell), observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_top_list |
| `efx_top_move` | field_events | 高优 | 上榜日价格变化 | `mean_60(pct_change, observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_top_list |
| `efx_top_turnover` | field_events | 高优 | 上榜股票换手暴露 | `mean_60(turnover_rate, observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_top_list |
| `efx_volume_ratio` | field_events | 高优 | 量比平滑 | `mean_20(log1p(volume_ratio), observed rows only)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | stock_finance |
| `elg_net_60d_to_mv` | fundflow | 高优 | 超大单 60 日累计净买入额 / 总市值（长线吸筹强度） | `net = (mf["buy_elg_amount"] - mf["sell_elg_amount"]) * 1e4
net60 = net.rolling(60, min_periods=20).sum()
circ_mv = fin["circ_mv"].reindex(net60.index)
raw = safe_divide(net60, circ_mv)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_main_fund_flow, stock_daily, stock_finance |
| `eom_14` | pattern | 高优 | 14 日 Ease of Movement：价格中点位移 ÷ (成交量/振幅) 的 14 日均值 | `mid=(adj_high+adj_low)/2; box_ratio=vol/(adj_high-adj_low); distance=mid-mid.shift(1); eom=distance/box_ratio; eom_avg=rolling(14, min_periods=7).mean()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 46 | stock_daily, stock_adj_factor |
| `ep_ttm` | value | 高优 | 盈利收益率 = 归母净利润TTM / 总市值（= 1/PE_TTM），亏损为负 | `earnings_to_price = NPParentCompanyOwners_TTM / (ClosePrice × TotalShares)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income, stock_daily, stock_finance |
| `etp5` | value | 高优 | 五年平均净利润 / 五年平均市值（≈ 5 年平均盈利收益率） | `ETP5 = RollingMean(NetProfit_Y, 1260) / RollingMean(MarketCap, 1260)
   Factor = CrossSectionalRank(ETP5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 2300 | stock_income, stock_daily, stock_finance |
| `extreme_move_event` | event | 低优 | 近 5 日极端波动事件的衰减加权（\|日收益\| ≥ 9.5%，半衰期 3 日） | `is_ext = daily["pct_chg"].abs().gt(7.0)
event = is_ext.astype(float).where(is_ext, np.nan)
decayed = event_decay(event, half_life=10)     # 参考库原文（阈值 7%、H=10）
# 本实现（任务书口径）：
#   ev = 1 if \|pct_chg\| >= 9.5 else 0      （停牌日 pct_chg 为 NaN -> 非事件）
#   decayed = Σ_{k=0}^{4} ev(t-k) × 0.5^(k/3)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 29 | stock_daily, stock_adj_factor |
| `forecast_profit_midpoint_change` | disclosure_detail | 高优 | 预告利润区间中点相对去年同期的变化 | `((net_profit_min+net_profit_max)/2-last_parent_net)/abs(last_parent_net)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_forecast |
| `forecast_profit_range_uncertainty` | disclosure_detail | 低优 | 预告利润区间宽度相对去年利润 | `(net_profit_max-net_profit_min)/abs(last_parent_net)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_forecast |
| `forecast_type_score` | event | 高优 | 业绩预告类型打分（预增/扭亏 +2 … 预减/首亏 −2），按 ann_date 前向填充 | `Score = map(type): 预增/扭亏=+2, 略增/续盈/减亏=+1, 不确定/其他=0,
                  略减=−1, 预减/首亏/续亏/增亏=−2
# PIT 对齐: ann_date <= T 的最新一条；同日多条取中位数` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_forecast |
| `free_share_ratio` | value | 低优 | 自由流通股占比 = free_share / total_share（低 = 筹码锁定度高） | `ratio = finance["free_share"] / finance["total_share"].replace(0, np.nan)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_finance |
| `fundflow_retail_inst_divergence` | fundflow | 高优 | 主力-散户背离 = rank(大单净买率) × rank(−散户成交占比)（机构买而散户卖的联合信号） | `big = ctx.load_factor("mf_big_order_ratio")
small = ctx.load_factor("mf_small_order_ratio")
big_r = _rank(big); small_r = _rank(-small)
divergence = big_r * small_r
return cross_sectional_rank(divergence)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | mf_big_order_ratio, mf_retail_dominance |
| `fv_gain_share` | quality | 低优 | 公允价值变动收益占利润总额的比重（纸面利润依赖度） | `share = fv_value_chg_gain_TTM / total_profit_TTM` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income |
| `gain_loss_asymmetry_60` | risk | 高优 | 涨跌幅度不对称 = 60 日平均涨幅 / \|平均跌幅\|（涨多跌少排前） | `mean_up = ret_w.where(ret_w > 0).rolling(60, min_periods=10).mean(); mean_down = ret_w.where(ret_w < 0).rolling(60, min_periods=10).mean(); asym = safe_divide(mean_up, mean_down.abs() + 1e-10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_daily, stock_adj_factor |
| `gap_down_recover_freq_20d` | pattern | 高优 | 低开高走频率：20 日内（隔夜<0 且 日内>0）的交易日占比（0~1） | `gap_down_rec = ((overnight<0) & (intraday>0)).astype(float); freq = rolling(20).mean()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `gap_event_decay_5` | event | 高优 | 近 5 日跳空事件的衰减加权（\|跳空幅度\| ≥ 3% 时按跳空幅度加权，半衰期 3 日） | `gap = safe_divide(daily["open"], daily["pre_close"]) - 1.0
is_gap = gap.abs().gt(0.05)
event = is_gap.astype(float).where(is_gap, np.nan)
decayed = event_decay(event, half_life=5)      # 参考库原文（0/1 事件）
# 本实现（任务书：跳空幅度 × 半衰期权重）：
#   g   = hfq(open)/hfq(pre_close) - 1        （当日有成交才有效）
#   decayed = Σ_{k=0}^{4} \|g(t-k)\| × 1[\|g\| >= 3%] × 0.5^(k/3)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 29 | stock_daily, stock_adj_factor |
| `gap_fill_tendency_10d` | pattern | 低优 | 缺口回补倾向：10 日内跳空（>1%）后被**当日收盘**回补的比例 | `gap=(open-pre_close)/pre_close; is_gap=\|gap\|>0.01; filled=((gap>0.01)&(close<pre_close))\|((gap<-0.01)&(close>pre_close)); rate=_roll_sum(filled,10,5)/(_roll_sum(is_gap,10,5)+0.01)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_daily, stock_adj_factor |
| `gap_up_fade_freq_20d` | pattern | 低优 | 高开低走频率：20 日内（隔夜>0 且 日内<0）的交易日占比（0~1） | `gap_up_fade = ((overnight>0) & (intraday<0)).astype(float); freq = rolling(20).mean()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `growth_stability` | growth | 高优 | 营收增速的稳定性 = 近 8 个报告期同比增速的均值 / 标准差 | `Stability = mean(YoY_{t-i}, i=0..7) / std(YoY_{t-i}, i=0..7) , YoY_{t-i} = Revenue_TTM_{t-i} / Revenue_TTM_{t-i-4} - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income |
| `hammer_ratio_20d` | pattern | 高优 | 锤子线频率：20 日内（下影 > 2×实体 且 上影 < 0.3×振幅 且 实体>0）的占比 | `body=\|close-open\|; lower=min(open,close)-low; upper=high-max(open,close); rng=high-low; is_hammer=(lower>2*body)&(upper<0.3*rng)&(body>0); ratio=rolling(20, min_periods=10).mean(is_hammer)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `high_open_low_close_frac_20` | event | 低优 | 高开低走占比 = 20 日内「高开≥2% 且收阴」的天数占比（出货特征） | `gapup = safe_divide(daily["open"], daily["pre_close"]) - 1.0
fade = (gapup >= 0.02) & (daily["close"] < daily["open"])
freq = fade.astype(float).groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=5).mean())` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_daily, stock_adj_factor |
| `holder_number_chg` | sentiment | 低优 | 股东户数变化率 = 本期 / 上期 − 1（户数减少 = 筹码集中，低者优） | `Chg = HolderNum_t / HolderNum_{t-1} - 1（两条腿都取「该公告日为止最新一版」）` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_holder_number |
| `id2_am_close_position` | intraday | 高优 | 上午收盘价在上午区间中的位置 ∈ [0,1] | `return cross_sectional_rank(_metric(context, 'am_hl_position'))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_am_pm_range_ratio` | intraday | 低优 | 上午振幅 / 下午振幅（取 ln，会话波动的时间分配） | `return cross_sectional_rank(-_metric(context, 'am_pm_hl_range_ratio'))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_am_pm_ret_gap` | intraday | 高优 | 上午段收益 − 下午段收益（会话动量的时间差） | `am_ret - pm_ret` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_am_pm_vwap_gap` | intraday | 高优 | 上午 VWAP / 下午 VWAP − 1（会话成交均价的时间位移） | `return cross_sectional_rank(_metric(context, 'vwap_am_pm_gap'))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_am_ret` | intraday | 高优 | 上午段收益 = 11:30 价 / 09:35 价 − 1 | `am = _compute_intraday_factor(context, 'am_momentum')` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_am_vol_share` | intraday | 高优 | 上午段成交量占全天的比重 | `avs = _compute_intraday_factor(context, 'am_vol_share'); rank(avs)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_amihud_intraday_20` | intraday | 低优 | 日内 Amihud 非流动性（20 日）：\|日内收益\| / 成交额 | `amihud_intraday = abs(ret_sum) / amt; 20 日均值` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_history_5min |
| `id2_close_vs_pm_vwap` | intraday | 低优 | 收盘价相对下午 VWAP 的偏离（收盘集合竞价的定价压力） | `return cross_sectional_rank(-_metric(context, 'vwap_dev'))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_close_vs_pm_vwap_20` | intraday | 低优 | 收盘相对下午 VWAP 偏离的 20 日水平（持续性尾盘溢价/折价） | `roll_mean(close5/pm_vwap - 1, 20, min_count=10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_history_5min |
| `id2_dd_ru_asym` | intraday | 低优 | 日内行程的不对称 = \|最大回撤\| / (\|最大回撤\| + 最大反弹) ∈ [0,1] | `abs(max_dd) / (abs(max_dd) + max_ru)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_hi_lo_pos_gap` | intraday | 高优 | 当日最高点与最低点的**时点**间隔（日内在时间轴上的铺开程度） | `hi_pos - lo_pos   （两者都是段内 argmax/argmin 归一化到 [0,1]）` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_intraday_max_runup` | intraday | 高优 | 日内最大反弹（从段内低点到其后高点的最大涨幅） | `ru = _compute_intraday_factor(context, 'intraday_max_runup'); rank(ru)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_lunch_gap` | intraday | 高优 | 午间跳空 = 13:05 价 / 11:30 价 − 1 | `lb = _compute_intraday_factor(context, 'lunch_break_ret')` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_parkinson_vol` | intraday | 低优 | Parkinson 极差波动（当日）= sqrt( ln(hi/lo)^2 / (4 ln2) ) | `parkinson_vol = sqrt(log(high/low)**2 / (4*log(2)))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_pm_ret` | intraday | 高优 | 下午段收益 = 收盘价 / 13:05 价 − 1 | `pm = _compute_intraday_factor(context, 'pm_momentum')` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_ret_concentration` | intraday | 低优 | 日内收益路径的集中度 = n·Σr² / (Σ\|r\|)²（反参与比） | `concentration = n_bars * ret2_sum / absret_sum**2` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_rv_parkinson_gap` | intraday | 低优 | 连续 vs 跳跃诊断 = ln(已实现波动 / Parkinson 极差波动) | `efficiency = safe_divide(rv_5min, parkinson_vol + 1e-10); rank(-efficiency)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_rv_parkinson_gap_20` | intraday | 低优 | 连续 vs 跳跃诊断的 20 日水平（单日太噪，20 日均值才稳） | `roll_mean(ln(rv_5min/parkinson_vol), 20, min_count=10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_history_5min |
| `id2_session_range_overlap` | intraday | 低优 | 跨会话区间结构 = (上午振幅 + 下午振幅) / 全天振幅 ∈ [1,2] | `(am_range + pm_range) / day_range` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_session_sign_agreement_20` | intraday | 高优 | 上午与下午同向的频率（20 日）∈ [0,1] | `roll_mean(sign(am_ret) == sign(pm_ret), 20, min_count=10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_history_5min |
| `id2_vol_amt_hhi_gap` | intraday | 高优 | 成交**额**集中度 − 成交**量**集中度（大单是否集中在高价区） | `n*amt2_sum/amt**2 - n*vol2_sum/vol**2` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_vol_peak_pos` | intraday | 低优 | 当日成交量的峰值时点（层已归一化到 [0,1]） | `vpt = _compute_intraday_factor(context, 'volume_peak_time'); rank(-vpt)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `id2_vol_peak_std_20` | intraday | 低优 | 放量时点的 20 日标准差（日内流动性节奏稳不稳） | `roll_std(vol_peak_pos, 20, min_count=10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_history_5min |
| `id2_zero_bar_share` | intraday | 低优 | 零收益 5min 棒占比 = (n_zero − 1) / (n_bars − 1) | `(n_zero - 1) / (n_bars - 1)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_history_5min |
| `idio_vol_60` | risk | 低优 | 60 日特质波动率 = 剔除市场暴露后残差的标准差（反向） | `beta = _rolling_beta(wide, mkt, 60, 30); resid = wide - beta.multiply(mkt, axis=0); idio = resid.rolling(60, min_periods=30).std()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_daily, stock_adj_factor, index_daily |
| `idt_intraday_max_drawdown` | intraday | 高优 | 日内最大回撤：从盘内高点到后续低点的最大跌幅（≤0） | `min(close5_t / cummax(close5) - 1)  （= 日内层 max_dd 字段）` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_history_5min |
| `idt_lo_pos` | intraday | 低优 | 日内最低价时点（归一化到 [0,1]）：越晚见低＝尾盘走弱（方向为负） | `argmin(close5) / (n_bars - 1)  （日内层 lo_pos，已在 [0,1]）` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_history_5min |
| `idt_overnight_gap` | intraday | 低优 | 隔夜跳空：后复权开盘 / 上一交易日后复权收盘 − 1 | `gap = open / pre_close - 1  （参考库 rank(-gap)）` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_daily, stock_adj_factor |
| `idt_overnight_minus_intraday` | intraday | 高优 | 隔夜 − 日内收益差：隔夜强于日内＝信息在开盘被消化 | `overnight - intraday  (= open/pre_close-1 与 close/open-1 之差)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_daily, stock_adj_factor |
| `idt_overnight_return_share_20` | intraday | 高优 | 隔夜收益占比：20 日 \|跳空\| / Σ(\|跳空\|+\|日内\|) | `roll_sum(\|gap\|, 20, 10) / roll_sum(\|gap\|+\|intra\|, 20, 10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_daily, stock_adj_factor |
| `idt_rv_daily` | intraday | 低优 | 日已实现波动率：当日 5min 收益平方和开根（**日频口径，未年化**） | `rv_daily = sqrt(sum(r_5min^2)) = sqrt(ret2_sum)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_history_5min |
| `idt_rv_term_structure_slope` | intraday | 低优 | 波动率期限结构斜率：rv_5日/rv_60日 − 1（陡峭＝短期波动高） | `rv_5min / rv_60min - 1,  rv_N = sqrt(roll_mean(ret2_sum, N))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_history_5min |
| `idt_up_minutes_ratio` | intraday | 高优 | 上涨棒占比：n_up / n_bars，买盘持续主导 | `n_up / n_bars   （5min 口径 = 参考库 intra_trend）` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_history_5min |
| `idt_vol_stability` | intraday | 低优 | 成交量稳定性：日内 5min 量的变异系数 std/mean（越大越不稳定） | `std(vol_5min) / mean(vol_5min) = sqrt(n_bars * vol2_sum*vol_mult / vol^2 - 1)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_history_5min |
| `ind_beta_60` | sector | 低优 | 个股对**所属行业**的 60 日 β（行业耦合强度） | `beta = roll_cov(ret, ind_ret, 60, 30) / roll_var(ind_ret, 60, 30)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `ind_bps_yoy` | growth | 高优 | 每股净资产同比（股东权益的**每股**累积速度） | `ctx.ind('bps_yoy')` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_financial_indicator |
| `ind_currentdebt_to_debt` | risk | 低优 | 短期债务占总债务的比重（债务期限结构 / 展期风险） | `ctx.ind('currentdebt_to_debt')   # 供应商时点比率` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_financial_indicator |
| `ind_disp_ma_20d` | sector | 低优 | 所属行业内部的收益分化度（行业成员截面 std 的 20 日均值） | `ind_disp = cross_std(ret within my industry); factor = ts_mean(ind_disp, 20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `ind_disp_ma_5d` | sector | 低优 | 所属行业内部的收益分化度（行业成员截面 std 的 5 日均值） | `ind_disp = cross_std(ret within my industry); factor = ts_mean(ind_disp, 5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `ind_dt_netprofit_yoy` | growth | 高优 | 扣非净利润同比（供应商同期同比字段，季节性自动抵消） | `ctx.ind('dt_netprofit_yoy')` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_financial_indicator |
| `ind_int_to_talcap` | risk | 低优 | 有息负债占总资本的比重（融资性杠杆，剔除经营性负债） | `ctx.ind('int_to_talcap')   # 供应商时点比率` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_financial_indicator |
| `ind_mom_accel` | sector | 高优 | 行业动量加速度 = 行业 5 日均收益 − 行业 20 日均收益 | `accel = ts_mean(ind_ret, 5) - ts_mean(ind_ret, 20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `ind_ret_ma_20d` | sector | 高优 | 所属行业的 20 日平均日收益（行业动能） | `ind_ret = industry_index_daily_return; factor = ts_mean(ind_ret, 20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `ind_ret_ma_3d` | sector | 高优 | 所属行业的 3 日平均日收益（行业动能） | `ind_ret = industry_index_daily_return; factor = ts_mean(ind_ret, 3)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `ind_ret_ma_5d` | sector | 高优 | 所属行业的 5 日平均日收益（行业动能） | `ind_ret = industry_index_daily_return; factor = ts_mean(ind_ret, 5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `ind_ret_ma_60d` | sector | 高优 | 所属行业的 60 日平均日收益（行业动能） | `ind_ret = industry_index_daily_return; factor = ts_mean(ind_ret, 60)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `inside_bar_count_20` | pattern | 高优 | 孕线（内包线）频率：20 日内「今高 ≤ 昨高 且 今低 ≥ 昨低」的交易日占比 | `prev_high=shift(high,1); prev_low=shift(low,1); inside=(high<=prev_high)&(low>=prev_low); count=_roll_sum(inside,20,5); 参考库: cross_sectional_rank(count)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `interest_coverage` | quality | 高优 | 利息保障倍数（TTM）= EBIT_TTM / 利息支出TTM（值越大偿债越安全） | `ICR = EBIT_TTM / InterestExpense_TTM` | 2019-05-01 | 2019-05-06 → 2026-09-21 | 700 | stock_income |
| `intraday_ma_20d` | pattern | 高优 | 日内收益均值（20 个交易日） | `intraday = close/open - 1; ma20 = rolling(20).mean()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `intraday_ma_5d` | pattern | 高优 | 日内收益均值（5 个交易日） | `intraday = close/open - 1; ma5 = rolling(5).mean()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | stock_daily, stock_adj_factor |
| `intraday_ma_60d` | pattern | 高优 | 日内收益均值（60 个交易日） | `intraday = close/open - 1; ma60 = rolling(60).mean()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_daily, stock_adj_factor |
| `intraday_ret_momentum` | pattern | 高优 | 日内收益因子：(close − open) / open，单日口径 | `intraday = (close - open) / open   # 参考库 cross_sectional_rank(intraday)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | stock_daily, stock_adj_factor |
| `intraday_ret_share_20` | coupling | 低优 | 日内收益占比 = 20 日 Σ(日内收益) / 20 日 Σ(\|隔夜\| + \|日内\|) | `intraday(t) = hfq_close(t)/hfq_open(t) − 1;  overnight(t) = hfq_open(t)/hfq_close(t−1) − 1;
Share = roll_sum(intraday, 20) / roll_sum(\|overnight\| + \|intraday\|, 20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 200 | stock_daily, stock_adj_factor |
| `intraday_vol_ratio_5d` | pattern | 高优 | 日内/隔夜波动比：5 日 \|日内收益\| 均值 ÷ 5 日 \|隔夜收益\| 均值 | `_ia = \|intraday\|; _oa = \|overnight\|; m_ia = rolling(5).mean(_ia); m_oa = rolling(5).mean(_oa); x = m_ia / (m_oa + 1e-8)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | stock_daily, stock_adj_factor |
| `inventory_turnover` | quality | 高优 | 存货周转率（TTM，次/年）= 营业成本TTM / 期末存货 | `InventoryTurnover = OperatingCost_TTM / Inventories` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income, stock_balancesheet |
| `invest_income_share` | quality | 低优 | 投资收益占利润总额的比重（非主业盈利依赖度） | `share = invest_income_TTM / total_profit_TTM` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income |
| `kdj_k_minus_d` | technical | 高优 | 日频 KDJ 的 K − D（KD 金叉/死叉的横截面形态），>0 = 多头排列 | `# 参考库只有分钟级 kdj_k_d_distance = (K-D)/\|D\|，日频版沿用 kdj_daily_j 的链：
wide = _adjusted_close(daily).unstack("Code")
ll9 = wide.rolling(9, min_periods=5).min()
hh9 = wide.rolling(9, min_periods=5).max()
rsv = safe_divide(wide - ll9, hh9 - ll9 + 1e-10) * 100.0
k = rsv.ewm(alpha=1.0 / 3.0, adjust=False).mean()
d = k.ewm(alpha=1.0 / 3.0, adjust=False).mean()
return k - d` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 300 | stock_daily, stock_adj_factor |
| `label_ret_10d` | label | 高优 | T+1 开盘买入、T+11 开盘卖出，持有 10 个交易日的收益 | `label = open_hfq(T+1+10) / open_hfq(T+1) - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 5 | stock_daily, stock_adj_factor |
| `label_ret_1d` | label | 高优 | T+1 开盘买入、T+2 开盘卖出，持有 1 个交易日的收益 | `label = open_hfq(T+1+1) / open_hfq(T+1) - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 5 | stock_daily, stock_adj_factor |
| `label_ret_20d` | label | 高优 | T+1 开盘买入、T+21 开盘卖出，持有 20 个交易日的收益 | `label = open_hfq(T+1+20) / open_hfq(T+1) - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 5 | stock_daily, stock_adj_factor |
| `label_ret_3d` | label | 高优 | T+1 开盘买入、T+4 开盘卖出，持有 3 个交易日的收益 | `label = open_hfq(T+1+3) / open_hfq(T+1) - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 5 | stock_daily, stock_adj_factor |
| `label_ret_5d` | label | 高优 | T+1 开盘买入、T+6 开盘卖出，持有 5 个交易日的收益 | `label = open_hfq(T+1+5) / open_hfq(T+1) - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 5 | stock_daily, stock_adj_factor |
| `large_order_timing_signal` | fundflow | 高优 | 大单净流入占比 × (1 − 20 日价格位置)：大钱在低位买 | `big_net = (buy_lg+buy_elg-sell_lg-sell_elg)/_total_amount(mf)
high_20 = adj.rolling(20,10).max(); low_20 = ...
position = (adj-low_20)/(high_20-low_20)
signal = big_net*(1-position)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_main_fund_flow, stock_daily, stock_adj_factor |
| `limit_board_streak_mean_60` | event | 高优 | 平均连板高度 = 60 日封板天数 / 连板启动次数（历史拉板惯性） | `sealed = (daily["pct_chg"] >= _LIMIT_UP).astype(float)
prev_sealed = sealed.astype(bool).groupby(level="Code").shift(1)
    .fillna(False).astype(bool)
start = (sealed.astype(bool) & ~prev_sealed).astype(float)
days60 = sealed.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=1).sum())
starts60 = start.astype(float).groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=1).sum())
avg = safe_divide(days60, starts60).fillna(0.0)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_limit_up |
| `limit_down_event_5` | event | 低优 | 近 5 个交易日跌停事件的指数衰减加权值（半衰期 3 日） | `down = daily["pct_chg"].le(-9.8)
event = down.astype(float).where(down, np.nan)
decayed = event_decay(event, half_life=3)      # 参考库原文
# 本实现（任务书范式：加权和，窗口 N=5）：
#   is_ld   = (limit == "D") if stock_limit_list 有记录
#             else (pct_chg <= -9.5 and close == low)
#   decayed = Σ_{k=0}^{4} is_ld(t-k) × 0.5^(k/3)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 29 | stock_limit_list, stock_daily, stock_adj_factor |
| `limit_up_count_20` | event | 高优 | 过去 20 个交易日的涨停次数 | `Count = sum(IsLimitUp, window=20 trading days)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_limit_up |
| `limit_up_event_5` | event | 高优 | 近 5 个交易日涨停事件的指数衰减加权值（半衰期 3 日）——跌停侧的对照组 | `event = daily["pct_chg"].ge(9.8).astype(float).where(
    daily["pct_chg"].ge(9.8), np.nan)
decayed = event_decay(event, half_life=3)      # 参考库原文
# 本实现（与 limit_down_event_5 同一套口径，只把方向反过来）：
#   is_lu   = (limit == "U") if stock_limit_list 有记录
#             else (pct_chg >= 9.5 and close == high)
#   decayed = Σ_{k=0}^{4} is_lu(t-k) × 0.5^(k/3)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 29 | stock_limit_list, stock_daily, stock_adj_factor |
| `liquidity_shock_20` | liquidity | 低优 | 流动性冲击：20 日 Amihud 均值相对再前 20 日的变化（放大 = 流动性恶化） | `amihud = safe_divide(\|pct_chg\| / 100.0, amount); log_amihud = np.log(amihud + 1e-12); ma_now = rolling(20, min_periods=10).mean(); ma_prev = ma_now.shift(20); shock = ma_now - ma_prev` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 95 | stock_daily, stock_adj_factor |
| `log_mv` | value | 低优 | 对数总市值 ln(close × total_share)（元），规模因子的基准 | `log_mv = ln(TotalMV), TotalMV = ClosePrice × TotalShares` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_finance |
| `macd_daily_hist_5d` | technical | 高优 | 日频 MACD 柱（归一化 DIF − DEA）的 5 日变化：动能的二阶导 | `wide = _adjusted_close(daily).unstack("Code")
ema12 = wide.ewm(span=12, adjust=False).mean()
ema26 = wide.ewm(span=26, adjust=False).mean()
# Normalise by the slow EMA before cross-sectional comparison.  Raw MACD
# is denominated in price units and would otherwise mostly reflect the
# arbitrary base level of each self-built adjusted-price index.
dif = safe_divide(ema12 - ema26, ema26)
dea = dif.ewm(span=9, adjust=False).mean()
hist = dif - dea
chg = hist.diff(5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 300 | stock_daily, stock_adj_factor |
| `margin_balance_20d` | margin | 高优 | 融资余额 20 日变化率（中期杠杆资金趋势） | `chg = m["rzye"].groupby(level="Code").transform(lambda s: s.pct_change(20, fill_method=None)); chg = chg.clip(-0.5, 1.0); return cross_sectional_rank(chg)【本实现：clip/rank 交给引擎；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_margin_detail |
| `margin_balance_5d` | margin | 高优 | 融资余额 5 日变化率（短期杠杆资金进出速度） | `chg = m["rzye"].groupby(level="Code").transform(lambda s: s.pct_change(5, fill_method=None)); chg = chg.clip(-0.5, 1.0); return cross_sectional_rank(chg)【本实现：clip/rank 交给引擎；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | stock_margin_detail |
| `margin_balance_ma_divergence` | margin | 高优 | 融资余额偏离 20 日均线幅度 = (rzye − MA20)/MA20（极端偏离预示均值回归） | `rzye = m["rzye"]
ma20 = rzye.groupby(level="Code").transform(
    lambda s: s.rolling(20, min_periods=10).mean())
div = safe_divide(rzye - ma20, ma20); div = div.clip(-0.1, 0.1)
return cross_sectional_rank(div)
★ 本实现的状态量口径（见 note）：asof 前向填充 + **20 个交易日新鲜度掩码**
stale = 最近 20 个交易日内无该股两融记录 → div = NaN` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_margin_detail |
| `margin_balance_volatility_20d` | margin | 低优 | 融资余额 20 日变异系数 CV=std/mean（杠杆资金稳定性，低者优） | `roll_std = rzye.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).std()); roll_mean = rzye.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean()); cv = safe_divide(roll_std, roll_mean); cv = cv.clip(0, 0.5); return cross_sectional_rank(-cv)【本实现：不取反（方向交给 higher_is_better=False）；clip 交给引擎缩尾；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_margin_detail |
| `margin_buyer_avg_cost_premium` | margin | 高优 | 融资盘成本溢价 = 现价 / 近 20 日融资买入加权平均成本 − 1 | `cost = _margin_weighted_cost(margin, daily); close = daily["close"].reindex(cost.index); raw = safe_divide(close, cost) - 1.0; return cross_sectional_rank(raw)【本实现：加权成本 = Σ(rzmre × hfq_close) / Σ(rzmre) over 20 日（原函数未公开，见 note）；配对价格用后复权 close；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_margin_detail, stock_daily, stock_adj_factor |
| `margin_chg_rel_5d` | margin | 高优 | 融资余额变化率之加速度 = 5 日变化率 − 20 日变化率 | `chg5 = m["rzye"].groupby(level="Code").transform(lambda s: s.pct_change(5, fill_method=None)); chg20 = m["rzye"].groupby(level="Code").transform(lambda s: s.pct_change(20, fill_method=None)); return cross_sectional_rank(chg5 - chg20)   ★ 参考库原条目为 margin_chg_rel_ind_5d（融资余额5日变化率**行业相对**）【本实现：行业表在本框架不可得（见 note），改为「相对自身中期趋势」；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_margin_detail |
| `margin_chip_cost_gap` | margin | 低优 | 融资盘成本 / 全市场筹码均价 − 1（融资盘相对市场平均的建仓位置，高位接盘者低优） | `cost = _margin_weighted_cost(margin, daily)   # Σ(rzmre × close)/Σ(rzmre) 近 20 日
weight_avg = cyq["weight_avg"].reindex(cost.index)
gap = safe_divide(cost, weight_avg) - 1.0
return cross_sectional_rank(-gap)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 60 | stock_margin_detail, stock_cyq_chips, stock_daily |
| `margin_flow_asymmetry_10d` | margin | 高优 | 10 日累计融资净买入 / 10 日累计融资交易额（方向持续性） | `net = m["rzmre"] - m["rzche"]; total = m["rzmre"] + m["rzche"]; net_10d = net.groupby(level="Code").transform(lambda s: s.rolling(10, min_periods=5).sum()); total_10d = total.groupby(level="Code").transform(lambda s: s.rolling(10, min_periods=5).sum()); asymmetry = safe_divide(net_10d, total_10d); asymmetry = asymmetry.clip(-1, 1); return cross_sectional_rank(asymmetry)【本实现：clip/rank 交给引擎；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_margin_detail |
| `margin_leverage_change_20d` | margin | 高优 | 融资杠杆变化 =（融资余额 / 流通市值）的 20 个交易日变化 | `total_mv_m = fin["total_mv"].reindex(m.index); leverage = safe_divide(m["rzye"], total_mv_m); change = leverage.groupby(level="Code").diff(20); return cross_sectional_rank(change)【本实现：total_mv → 流通市值 = 未复权 close × float_share（用户口径）；diff(20) 即 20 个交易日之差；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_margin_detail, stock_daily, stock_finance |
| `margin_repay_deceleration` | margin | 低优 | 融资偿还额 5 日变化率（偿还减速=看空力量减弱，低者优） | `chg = m["rzche"].groupby(level="Code").transform(lambda s: s.pct_change(5, fill_method=None)); chg = chg.clip(-0.5, 0.5); return cross_sectional_rank(-chg)【本实现：不取反（方向交给 higher_is_better=False）；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | stock_margin_detail |
| `margin_repay_shock` | margin | 低优 | 融资偿还冲击 = 当日偿还额 / 20 日均偿还额（突然放大=恐慌平仓，低者优） | `repay_ma20 = rzche.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean()); shock = safe_divide(rzche, repay_ma20); shock = shock.clip(0, 5); return cross_sectional_rank(-shock)【本实现：不取反（方向交给 higher_is_better=False）；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_margin_detail |
| `margin_velocity` | margin | 高优 | 融资周转速度 =（融资买入 + 融资偿还）/ 融资余额 | `total_flow = m["rzmre"] + m["rzche"]; velocity = safe_divide(total_flow, m["rzye"]); velocity = velocity.clip(0, 2); return cross_sectional_rank(velocity)【本实现：clip/rank 交给引擎；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | stock_margin_detail |
| `marubozu_ratio_10d` | pattern | 高优 | 光头光脚阳线频率：10 日内（上下影合计 < 10%×振幅 且 收>开）的交易日占比 | `body=\|close-open\|; upper=high-max(open,close); lower=min(open,close)-low; rng=high-low; is_marubozu=((upper+lower)/rng < 0.1); is_green=(close>open); signal=is_marubozu*is_green; ratio=rolling(10, min_periods=5).mean(signal)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_daily, stock_adj_factor |
| `max_drawdown_120` | risk | 高优 | 120 日回撤深度 = 后复权价 / 最近 120 日最高价 − 1（≤0） | `peak = adj.groupby(level="Code").transform(lambda s: s.rolling(120, min_periods=60).max()); drawdown = adj / peak.replace(0, np.nan) - 1.0  # ≤ 0` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 240 | stock_daily, stock_adj_factor |
| `max_drawdown_60` | risk | 高优 | 60 日回撤深度 = 后复权价 / 最近 60 日最高价 − 1（≤0） | `peak = adj.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).max()); drawdown = adj / peak.replace(0, np.nan) - 1.0  # ≤ 0` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_daily, stock_adj_factor |
| `mf_amount_weighted_direction` | fundflow | 高优 | 金额加权方向 = 四档方向信号 sign(该档净额) 按该档毛额占比加权求和（多空结构综合） | `sm_dir = np.sign(ff["buy_sm_amount"] - ff["sell_sm_amount"])   # md/lg/elg 同理
sm_w = (ff["buy_sm_amount"] + ff["sell_sm_amount"]) / total_amt   # md/lg/elg 同理
composite = sm_dir*sm_w + md_dir*md_w + lg_dir*lg_w + elg_dir*elg_w
return cross_sectional_rank(composite)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_main_fund_flow, stock_daily |
| `mf_avg_trade_price_momentum` | fundflow | 高优 | 成交均价动量 = (大单VWAP / 小单VWAP) 的 5 日变化率（比率上升=机构买入更急迫） | `big_vol = (buy_lg_vol + sell_lg_vol + buy_elg_vol + sell_elg_vol)
big_amt = (buy_lg_amount + sell_lg_amount + buy_elg_amount + sell_elg_amount)
big_vwap = safe_divide(big_amt, big_vol); sm_vwap = safe_divide(sm_amt, sm_vol)
ratio = safe_divide(big_vwap, sm_vwap).clip(0.5, 3.0)
mom = ratio.groupby(level="Code").transform(
    lambda s: s.pct_change(5, fill_method=None)).clip(-0.3, 0.5)
return cross_sectional_rank(mom)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 38 | stock_main_fund_flow, stock_daily |
| `mf_big_mid_net_corr_20` | fundflow | 高优 | 大单净额 与 中单净额 的 20 日滚动相关（机构与中户是否同向） | `b = (buy_lg+buy_elg-sell_lg-sell_elg); m = (buy_md-sell_md)
factor = roll_corr(b, m, 20, 10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_main_fund_flow |
| `mf_big_order_net_ac1_20` | fundflow | 高优 | 大单净额占比的一阶自相关（20 日）（资金流入的持续性） | `x = big_net_ratio; factor = roll_corr(x, shift(x,1), 20, 10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_main_fund_flow |
| `mf_big_order_net_kurt_20` | fundflow | 高优 | 大单净额占比的 20 日峰度（脉冲式建仓 vs 匀速滴灌） | `x = (mf["buy_lg_amount"]+mf["buy_elg_amount"]-mf["sell_lg_amount"]-mf["sell_elg_amount"])
kurt = x.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).kurt())` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_main_fund_flow |
| `mf_big_order_ratio` | fundflow | 高优 | 大单+超大单净买入率 = (lg+elg 净额) / 当日八列毛额 | `big_net = (ff["buy_lg_amount"] - ff["sell_lg_amount"] + ff["buy_elg_amount"] - ff["sell_elg_amount"]); ratio = big_net / _total_amount(ff)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_main_fund_flow, stock_daily |
| `mf_big_small_divergence` | fundflow | 高优 | 大小单背离 = (大单净买额 − 小单净买额) / 当日八列毛额（机构与散户分歧，分歧顶点伴转折） | `big_net = (ff["buy_lg_amount"] - ff["sell_lg_amount"]
           + ff["buy_elg_amount"] - ff["sell_elg_amount"])
small_net = ff["buy_sm_amount"] - ff["sell_sm_amount"]
divergence = (big_net - small_net) / _total_amount(ff)
return cross_sectional_rank(divergence)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_main_fund_flow, stock_daily |
| `mf_elg_lg_split_20` | fundflow | 高优 | 超大单净额占比 − 大单净额占比（20 日均值）：哪一级机构在买 | `e = elg_net/total; l = lg_net/total; factor = mean(e,20,10) - mean(l,20,10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_main_fund_flow |
| `mf_extra_large_sell_pressure` | fundflow | 低优 | 超大单**卖出**毛额占八列毛额的比重（顶级资金的派发压力） | `total = Σ(buy_*_amount) + Σ(sell_*_amount); elg_sell_ratio = sell_elg_amount / total; return cross_sectional_rank(-elg_sell_ratio)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_main_fund_flow |
| `mf_flow_factor_momentum_20` | fundflow | 高优 | 主力资金流因子的 20 日动量 = Δ20(主力净流入率)（机构态度的边际改善） | `mf = ctx.load_factor("mf_net_inflow_ratio")
delta = _delta(mf, 20)
return cross_sectional_rank(delta)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | mf_net_inflow_ratio |
| `mf_flow_price_absorption_20` | fundflow | 高优 | 20 日大单净流入占比累计 / 20 日累计绝对收益（单位价格冲击吸收的流量） | `flow = roll_sum(big_net_ratio, 20, 10); move = roll_sum(abs(ret), 20, 10); factor = flow / move` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_main_fund_flow, stock_daily, stock_adj_factor |
| `mf_large_order_avg_price` | fundflow | 高优 | 大单均价相对全日成交均价的偏离（正 = 机构在高价位成交） | `large_avg = safe_divide(large_amt, large_vol)   # lg+elg 的买+卖毛额/毛量
total_avg = safe_divide(total_amt, total_vol)
ratio = safe_divide(large_avg, total_avg).clip(0.5, 2.0)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_main_fund_flow, stock_daily |
| `mf_large_order_net_5d` | fundflow | 高优 | 大单净买入率 5 日均值（只算 lg 档，不含超大单） | `large_net = ff["buy_lg_amount"] - ff["sell_lg_amount"]; ratio = safe_divide(large_net, _total_amount(ff)); ratio_ma5 = ratio.rolling(5, min_periods=3).mean(); ratio_ma5.clip(-0.5, 0.5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 29 | stock_main_fund_flow, stock_daily |
| `mf_net_amount_intensity` | fundflow | 高优 | 主力资金净额强度 = 净流入额 / 总市值（消除规模效应） | `intensity = net_mf_amount / circ_mv   （参考库用流通市值）
本实现：net_amount * 1e4 / (close * total_share)   # 万元->元；总市值=元` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_main_fund_flow, stock_daily, stock_finance |
| `mf_net_amount_mom5_to_mv` | fundflow | 高优 | 厂商净流入额的 5 日变化 / 总市值（无尺度的流量动量） | `mom = net_mf_amount.diff(5); return cross_sectional_rank(mom)   # 参考库原式
本实现：mom * 1e4 / (close * total_share)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 29 | stock_main_fund_flow, stock_daily |
| `mf_net_inflow_5d` | fundflow | 高优 | 5 日累计主力净流入率 | `daily_ratio = ff["net_mf_amount"] / _total_amount(ff); cum_ratio = daily_ratio.rolling(5, min_periods=3).sum()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 29 | stock_main_fund_flow, stock_daily |
| `mf_net_inflow_ratio` | fundflow | 高优 | 主力资金净流入率 = 供应商净额 / 当日四档买卖总额（T 日单日） | `ratio = ff["net_mf_amount"] / _total_amount(ff)   # _total_amount = 买4+卖4 八列金额之和` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_main_fund_flow, stock_daily |
| `mf_net_vol_surprise_20` | fundflow | 高优 | 厂商净流入量的 20 日意外度 = (净量 − 20 日均) / 毛量 20 日均 | `ma_20 = net_vol.rolling(20,10).mean()
divergence = safe_divide(net_vol, ma_20.abs()+1e-10) - 1.0   # 参考库原式
本实现：(net_vol - ma_20) / mean(total_vol, 20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_main_fund_flow |
| `mf_open_close_divergence_10d` | fundflow | 低优 | 资金态度摇摆度 = 净流入率相对其 5 日均线偏离的 10 日波动率 | `net_rate = safe_divide(ff["net_mf_amount"], _total_amount(ff)); trend_5 = rolling_group_mean(net_rate, 5); divergence = safe_divide(net_rate - trend_5, trend_5.abs() + 1e-8); div_std = rolling_group_std(divergence, 10); return cross_sectional_rank(div_std)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 47 | stock_main_fund_flow, stock_daily |
| `mf_order_size_entropy` | fundflow | 高优 | 四档成交量分布的信息熵（归一化到 [0,1]，大 = 参与结构多样） | `sm_share = (buy_sm_vol + sell_sm_vol) / _total_vol(ff);  # md/lg/elg 同理
entropy = -(Σ p·ln(p + 1e-10)) / ln(4)      # 本框架加了 /ln4 归一化` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_main_fund_flow, stock_daily |
| `mf_order_size_entropy_chg_5d` | fundflow | 高优 | 四档成交量分布信息熵的 5 日变化（参与结构形状的迁移） | `sm_share = (buy_sm_vol+sell_sm_vol)/total_vol;  # md/lg/elg 同理
entropy = -(Σ p·ln(p+1e-10))/ln(4);  factor = entropy.diff(5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 29 | stock_main_fund_flow, stock_daily |
| `mf_retail_dominance` | fundflow | 低优 | 散户成交占比 = 小单买+卖金额 / 全部成交额（高者劣） | `retail = ff["buy_sm_amount"] + ff["sell_sm_amount"]; ratio = safe_divide(retail, _total_amount(ff)); return cross_sectional_rank(-ratio)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_main_fund_flow, stock_daily |
| `mf_small_order_avg_price_dev` | fundflow | 低优 | 小单成交均价相对当日 VWAP 的绝对偏离（散户的成交价劣势） | `sm_avg_price = safe_divide(sm_total_amt, sm_total_vol)
ratio = safe_divide(sm_avg_price, vwap)
return cross_sectional_rank(-deviation)   # deviation = (ratio-1).abs()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_main_fund_flow, stock_daily |
| `mf_tier_flow_agreement_20` | fundflow | 高优 | 「大单与中单同向」频率 − 「中单与小单同向」频率（20 日） | `agree(a,b) = sign(a)==sign(b); factor = mean(agree(big,mid),20,10) - mean(agree(mid,small),20,10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_main_fund_flow |
| `mf_vol_amount_divergence` | fundflow | 高优 | 资金流的量-额背离 = 净流入量占比 − 净流入额占比 | `net_vol_ratio = safe_divide(ff["net_mf_vol"], _total_vol(ff)); net_amt_ratio = safe_divide(ff["net_mf_amount"], _total_amount(ff)); divergence = (net_vol_ratio - net_amt_ratio).clip(-0.5, 0.5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_main_fund_flow, stock_daily |
| `mfi_14` | technical | 高优 | 14 日资金流量指标 MFI（量加权的 RSI），值域 [0,100] | `# TP 方向判定为跨日比较,走复权口径避免除权日误判方向
# (scale=adj/close 折算 high/low,close 直接用复权基座)
adj = _adjusted_close(daily)
scale = adj / daily["close"].replace(0, np.nan)
tp = (daily["high"] * scale + daily["low"] * scale + adj) / 3.0
raw_flow = tp * daily["vol"]
tp_chg = tp.groupby(level="Code").diff()
pos_flow = raw_flow.where(tp_chg > 0, 0.0)
neg_flow = raw_flow.where(tp_chg < 0, 0.0)
pos_w = pos_flow.unstack("Code").rolling(14, min_periods=7).sum()
neg_w = neg_flow.unstack("Code").rolling(14, min_periods=7).sum()
ratio = safe_divide(pos_w, neg_w + 1e-10)
mfi = 100.0 - 100.0 / (1.0 + ratio)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 48 | stock_daily, stock_adj_factor |
| `mfx_dc_amount` | field_markets | 高优 | 个股收益与dc_daily的amount状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(dc_daily.amount)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | dc_daily, stock_daily, stock_adj_factor |
| `mfx_dc_pct_change` | field_markets | 高优 | 个股收益与dc_daily的pct_change状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(dc_daily.pct_change)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | dc_daily, stock_daily, stock_adj_factor |
| `mfx_dc_pressure` | field_markets | 高优 | 个股收益与dc_daily的pressure状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(dc_daily.pressure)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | dc_daily, stock_daily, stock_adj_factor |
| `mfx_dc_range` | field_markets | 高优 | 个股收益与dc_daily的range状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(dc_daily.range)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | dc_daily, stock_daily, stock_adj_factor |
| `mfx_dc_turnover_rate` | field_markets | 高优 | 个股收益与dc_daily的turnover_rate状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(dc_daily.turnover_rate)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | dc_daily, stock_daily, stock_adj_factor |
| `mfx_dc_volume` | field_markets | 高优 | 个股收益与dc_daily的volume状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(dc_daily.volume)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | dc_daily, stock_daily, stock_adj_factor |
| `mfx_index_amount` | field_markets | 高优 | 个股收益与index_daily的amount状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(index_daily.amount)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | index_daily, stock_daily, stock_adj_factor |
| `mfx_index_gap` | field_markets | 高优 | 个股收益与index_daily的gap状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(index_daily.gap)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | index_daily, stock_daily, stock_adj_factor |
| `mfx_index_pressure` | field_markets | 高优 | 个股收益与index_daily的pressure状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(index_daily.pressure)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | index_daily, stock_daily, stock_adj_factor |
| `mfx_index_range` | field_markets | 高优 | 个股收益与index_daily的range状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(index_daily.range)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | index_daily, stock_daily, stock_adj_factor |
| `mfx_index_volume` | field_markets | 高优 | 个股收益与index_daily的volume状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(index_daily.volume)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | index_daily, stock_daily, stock_adj_factor |
| `mfx_minute_amplitude` | field_markets | 高优 | 个股收益与tdx_minute的amplitude状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(tdx_minute.amplitude)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | tdx_minute, stock_daily, stock_adj_factor |
| `mfx_minute_volume_concentration` | field_markets | 高优 | 个股收益与tdx_minute的volume_concentration状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(tdx_minute.volume_concentration)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | tdx_minute, stock_daily, stock_adj_factor |
| `mfx_minute_weighted_pressure` | field_markets | 高优 | 个股收益与tdx_minute的weighted_pressure状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(tdx_minute.weighted_pressure)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | tdx_minute, stock_daily, stock_adj_factor |
| `mfx_tdx_amount` | field_markets | 高优 | 个股收益与tdx_daily的amount状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(tdx_daily.amount)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | tdx_daily, stock_daily, stock_adj_factor |
| `mfx_tdx_volume` | field_markets | 高优 | 个股收益与tdx_daily的volume状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(tdx_daily.volume)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | tdx_daily, stock_daily, stock_adj_factor |
| `mfx_ths_gap` | field_markets | 高优 | 个股收益与index_ths_daily的gap状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(index_ths_daily.gap)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | index_ths_daily, stock_daily, stock_adj_factor |
| `mfx_ths_premium` | field_markets | 高优 | 个股收益与index_ths_daily的premium状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(index_ths_daily.premium)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | index_ths_daily, stock_daily, stock_adj_factor |
| `mfx_ths_volume` | field_markets | 高优 | 个股收益与index_ths_daily的volume状态60日相关暴露 | `corr60(ret_clean_hfq, cross_index_median(index_ths_daily.volume)); volume/amount use log change` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 220 | index_ths_daily, stock_daily, stock_adj_factor |
| `momentum_10` | momentum | 高优 | 10 个交易日的后复权动量（2 周趋势动能） | `Mom10 = hfq_close(T) / hfq_close(T-10) - 1  # 参考库: adj.groupby(Code).pct_change(10, fill_method=None)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 38 | stock_daily, stock_adj_factor |
| `momentum_120` | momentum | 高优 | 120 个交易日的后复权动量（半年趋势质量） | `Mom120 = hfq_close(T) / hfq_close(T-120) - 1  # 参考库: adj.groupby(Code).pct_change(120, fill_method=None)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 236 | stock_daily, stock_adj_factor |
| `momentum_20` | momentum | 高优 | 20 个交易日的后复权动量（Jegadeesh-Titman 月度动量区间） | `Mom20 = hfq_close(T) / hfq_close(T-20) - 1  # 参考库: adj.groupby(Code).pct_change(20, fill_method=None)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_daily, stock_adj_factor |
| `momentum_250` | momentum | 高优 | 250 个交易日的后复权动量（年度动量，最慢最稳） | `Mom250 = hfq_close(T) / hfq_close(T-250) - 1  # 参考库: adj.groupby(Code).pct_change(250, fill_method=None)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 480 | stock_daily, stock_adj_factor |
| `net_margin_ttm` | quality | 高优 | 销售净利率（TTM）= 归母净利润TTM / 营业收入TTM | `NPM = NetProfit / OperatingRevenue` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income |
| `new_high_60_event` | event | 高优 | 近 60 个交易日新高事件的指数衰减加权（当日最高价 = 60 日最高，半衰期 5 日） | `adj = _adjusted_close(daily)
is_high = adj.eq(adj.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).max()))
event = is_high.astype(float).where(is_high, np.nan)
decayed = event_decay(event, half_life=5)      # 参考库原文
# 本实现：
#   is_high = hfq(high) == max(hfq(high), 60)  且 当日有成交
#   decayed = Σ_{k=0}^{9} is_high(t-k) × 0.5^(k/5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 164 | stock_daily, stock_adj_factor |
| `new_high_frequency_60` | event | 高优 | 60 日新高频率 = 60 日内「收盘价创 60 日新高」的天数占比 | `adj = _adjusted_close(daily)
is_high = adj.eq(adj.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).max()))
freq = is_high.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).mean())` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_daily, stock_adj_factor |
| `new_low_60_event` | event | 低优 | 近 60 个交易日新低事件的指数衰减加权（当日最低价 = 60 日最低，半衰期 5 日） | `adj = _adjusted_close(daily)
is_low = adj.eq(adj.groupby(level="Code").transform(
    lambda s: s.rolling(60, min_periods=30).min()))
event = is_low.astype(float).where(is_low, np.nan)
decayed = event_decay(event, half_life=5)      # 参考库原文
# 本实现：
#   is_low  = hfq(low) == min(hfq(low), 60)  且 当日有成交
#   decayed = Σ_{k=0}^{9} is_low(t-k) × 0.5^(k/5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 164 | stock_daily, stock_adj_factor |
| `np_to_deferred_tax_yoy` | growth | 高优 | 单位递延所得税资产创利同比 = (净利_TTM / 递延所得税资产) 的同比 | `Ratio = NetProfit_Q / DeferredTaxAssets;  Growth = Ratio_t / Ratio_{t-4} - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income, stock_balancesheet |
| `np_to_fixed_assets_yoy` | growth | 高优 | 单位固定资产创利同比 = (净利_TTM / 固定资产) 的同比 | `Ratio = NetProfit_Q / FixedAssets;  Growth = Ratio_t / Ratio_{t-4} - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income, stock_balancesheet |
| `np_to_inventory_yoy` | growth | 高优 | 单位存货创利同比 = (净利_TTM / 存货) 的同比 | `Ratio = NetProfit_Q / Inventories;  Growth = Ratio_t / Ratio_{t-4} - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income, stock_balancesheet |
| `np_to_opex_yoy` | growth | 高优 | 单位经营性费用创利同比 = (净利_TTM / (销售+管理+研发费用)_TTM) 的同比 | `Opex = sell_exp + admin_exp + rd_exp (TTM)
Ratio = NetProfit_TTM / Opex;  Growth = Ratio_t / Ratio_{t-4} - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income |
| `np_to_salary_yoy` | growth | 高优 | 单位薪酬创利同比 = (净利_TTM / 支付给职工的现金_TTM) 的同比 | `Ratio = NetProfit_TTM / StaffBehalfPaid_TTM;  Growth = Ratio_t / Ratio_{t-4} - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income, stock_cashflow |
| `ocf_to_profit` | quality | 高优 | 盈利现金保障倍数（TTM）= 经营现金流TTM / 归母净利润TTM | `OCFtoProfit = NetOperateCashFlow_TTM / NetProfit_TTM` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_cashflow, stock_income, stock_balancesheet |
| `ocf_to_revenue` | quality | 高优 | 经营现金流占营收比（TTM）= 经营现金流TTM / 营业收入TTM | `OCFtoRevenue = NetOperateCashFlow_TTM / OperatingRevenue_TTM` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_cashflow, stock_income, stock_balancesheet |
| `one_word_limit_down_freq_20` | event | 低优 | 一字跌停占比 = 20 日内「全天无波动且跌停」的天数占比 | `no_range = (daily["high"] - daily["low"]).abs().lt(1e-6)
one_word = (no_range & daily["pct_chg"].le(-9.8)).astype(float)
freq = _roll_mean(one_word, 20, 5)
return cross_sectional_rank(-freq)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_limit_list, stock_daily, stock_adj_factor |
| `open5_amt_log` | intraday | 高优 | 开盘首 5 分钟成交额的对数（绝对开盘容量，供执行层估单笔上限） | `log(amt_open5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_history_5min |
| `open5_amt_share` | intraday | 高优 | 开盘首 5 分钟成交额 / 全日成交额（执行容量口径，非 alpha） | `amt_open5 / amt_day` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_history_5min |
| `open5_amt_share_pct20` | intraday | 高优 | 开盘首 5 分钟成交额占比的 20 日时序分位（去掉长期漂移与个股规模） | `Ts_Rank(amt_open5/amt_day, 20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_history_5min |
| `opm_npm_spread` | quality | 低优 | 营业利润率 − 净利率（营业利润到归母之间的损耗） | `spread = operate_profit/revenue - n_income_attr_p/revenue` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income |
| `order_size_concentration` | fundflow | 高优 | 订单规模集中度 = 四档成交额 HHI × sign(大单净额)（机构主导的高集中 vs 散户主导的高集中） | `sm_total = ff["buy_sm_amount"] + ff["sell_sm_amount"]   # md/lg/elg 同理
total_amount = sm_total + md_total + lg_total + elg_total
hhi = ((sm_total/total_amount)**2 + (md_total/total_amount)**2
       + (lg_total/total_amount)**2 + (elg_total/total_amount)**2)
smart_direction = (ff["buy_elg_amount"] - ff["sell_elg_amount"]
                   + ff["buy_lg_amount"] - ff["sell_lg_amount"])
signed_hhi = hhi * np.sign(smart_direction)
return cross_sectional_rank(signed_hhi)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_main_fund_flow, stock_daily |
| `order_size_ratio_change` | fundflow | 高优 | 订单规模比变化 = 大单+超大单毛额占比的 5 日变化（大单占比提升=机构参与加深） | `big_total = (ff["buy_lg_amount"] + ff["sell_lg_amount"]
             + ff["buy_elg_amount"] + ff["sell_elg_amount"])
total = _total_amount(ff); big_ratio = big_total / total
chg = big_ratio.groupby(level="Code").transform(lambda s: s.diff(5))
return cross_sectional_rank(chg)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 38 | stock_main_fund_flow, stock_daily |
| `outside_bar_count_20` | pattern | 高优 | 吞没/突破频率：20 日内「收盘 > 昨高 或 收盘 < 昨低」的交易日占比 | `prev_high=shift(high,1); prev_low=shift(low,1); outside=(close>prev_high)\|(close<prev_low); count=_roll_sum(outside,20,5); 参考库: cross_sectional_rank(count)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `overnight_gap_vol_20` | pattern | 低优 | 20 日隔夜跳空波动率（样本标准差 ddof=1，高波动=信息不确定性高） | `overnight_ret = (open - pre_close)/(pre_close + 1e-8); gap_vol = rolling(20, min_periods=10).std()   # 参考库 rank(-gap_vol)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `overnight_intraday_ratio_20d` | pattern | 高优 | 隔夜/日内收益强度比：20 日隔夜均值 ÷ \|20 日日内均值\| | `oma = rolling(20).mean(overnight); ima = rolling(20).mean(intraday); x = oma / (ima.abs() + 1e-6)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `overnight_ma_20d` | pattern | 高优 | 隔夜收益均值（20 个交易日） | `overnight = open/pre_close - 1; ma20 = rolling(20).mean()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `overnight_ma_5d` | pattern | 高优 | 隔夜收益均值（5 个交易日） | `overnight = open/pre_close - 1; ma5 = rolling(5).mean()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | stock_daily, stock_adj_factor |
| `overnight_ma_60d` | pattern | 高优 | 隔夜收益均值（60 个交易日） | `overnight = open/pre_close - 1; ma60 = rolling(60).mean()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_daily, stock_adj_factor |
| `overnight_sign_consistency_20d` | pattern | 高优 | 隔夜方向一致性：20 日内 overnight > 0 的交易日占比（0~1） | `overnight_pos = (overnight > 0).astype(float); x = rolling(20).mean()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `overnight_skewness_20d` | pattern | 高优 | 20 日隔夜收益偏度的**绝对值取反**（−\|skew\|，越接近 0 = 信息冲击越不极端） | `skew = rolling(20, min_periods=10).skew(); skew = skew.clip(-5, 5); factor = -skew.abs()   # 参考库: cross_sectional_rank(-skew.abs())` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `overnight_std_5d` | pattern | 低优 | 隔夜收益标准差（5 个交易日，样本口径 ddof=1） | `overnight = open/pre_close - 1; std5 = rolling(5).std()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | stock_daily, stock_adj_factor |
| `panic_selling_ratio_60` | risk | 高优 | 放量下跌占比 = 60 日内「放量且下跌」天数 / 放量天数（恐慌抛售排前） | `ma20 = vol.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean()); big = vol.gt(1.5 * ma20); down = ret.lt(0); panic = (big & down).astype(float); n_big = big.astype(float)...rolling(60, min_periods=1).sum(); n_panic = panic...rolling(60, min_periods=1).sum(); ratio = safe_divide(n_panic, n_big)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 164 | stock_daily, stock_adj_factor |
| `pegh5` | value | 低优 | PEG 的 5 年增长版 = 未复权收盘价 / (5 年 EPS 复合增速 × EPS_TTM) | `1. EPS_Growth_5Y = (BasicEPS_Y_t / BasicEPS_Y_{t-1260})^(1/5) - 1
2. PEGH5 = ClosePrice / (EPS_Growth_5Y * BasicEPS_TTM)
   Factor = -CrossSectionalRank(PEGH5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 2600 | stock_income, stock_balancesheet, stock_daily, stock_finance |
| `price_distance_from_52w_low` | momentum | 高优 | 距 52 周低点的距离 = (现价 − 250 日最低收盘) / 250 日最低收盘 | `Dist = (hfq_close - roll_min(hfq_close, 252)) / roll_min(hfq_close, 252)  # 参考库: (adj - low_252) / low_252` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 480 | stock_daily, stock_adj_factor |
| `price_to_52w_high` | momentum | 高优 | 52 周高点接近度 = 现价 / 250 日最高收盘价 − 1（越接近 0 越靠近年内高点） | `Proximity = hfq_close / roll_max(hfq_close, 252) - 1  # 参考库: adj / adj.rolling(252, min_periods=120).max() - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 480 | stock_daily, stock_adj_factor |
| `qf_cash_margin_floor_4q` | quarterly_quality | 高优 | 最近四季经营现金收入比最低值 | `min(q_ocf_to_sales(P-k),k=0..3)/100` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `qf_cash_margin_vol_4q` | quarterly_quality | 低优 | 最近四季经营现金收入比波动 | `std_population(q_ocf_to_sales(P-k),k=0..3)/100` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `qf_cash_margin_yoy_change` | quarterly_quality | 高优 | 单季度经营现金收入比同比改善 | `(q_ocf_to_sales(P)-q_ocf_to_sales(P-4))/100` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `qf_core_roe_floor_4q` | quarterly_quality | 高优 | 最近四季扣非ROE最低值 | `min(q_dt_roe(P-k),k=0..3)/100` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `qf_core_roe_vol_4q` | quarterly_quality | 低优 | 最近四季扣非ROE波动 | `std_population(q_dt_roe(P-k),k=0..3)/100` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `qf_core_roe_yoy_change` | quarterly_quality | 高优 | 单季度扣非ROE同比改善 | `(q_dt_roe(P)-q_dt_roe(P-4))/100` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `qf_noncore_roe_gap` | quarterly_quality | 低优 | 单季度ROE中的非经常损益贡献差 | `(q_roe(P)-q_dt_roe(P))/100` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `qf_roa_yoy_change` | quarterly_quality | 高优 | 单季度资产净利率同比改善 | `(q_npta(P)-q_npta(P-4))/100` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `qf_roe_yoy_change` | quarterly_quality | 高优 | 单季度ROE同比改善 | `(q_roe(P)-q_roe(P-4))/100` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `qf_sales_growth_accel` | quarterly_quality | 高优 | 单季度收入同比增速的环比变化 | `(q_sales_yoy(P)-q_sales_yoy(P-1))/100` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `qf_sales_growth_floor_4q` | quarterly_quality | 高优 | 最近四季收入同比增长的最低值 | `min(q_sales_yoy(P-k),k=0..3)/100` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `qf_sales_growth_vol_4q` | quarterly_quality | 低优 | 最近四季收入同比增长波动 | `std_population(q_sales_yoy(P-k),k=0..3)/100` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 1100 | stock_financial_indicator |
| `quality_composite` | quality | 高优 | 质量复合 = ROE(TTM)、毛利率(TTM)、经营现金流/总资产 三项截面 z 分数等权均值 | `Ratio_i = ① NetProfit/Equity ② (Revenue−Cost)/Revenue ③ OCF/TotalAssets
Factor = mean_i( cs_zscore(Ratio_i) )   # 至少 2 项有效才合成` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_cashflow, stock_income, stock_balancesheet |
| `rd_intensity` | quality | 高优 | 研发强度（TTM）= 研发费用TTM / 营业收入TTM | `RDIntensity = RDExpense_TTM / OperatingRevenue_TTM` | 2019-05-01 | 2019-05-06 → 2026-09-21 | 700 | stock_income |
| `rel_mom_ind_10d` | sector | 高优 | 行业相对动量（个股 10 日收益 − 所属行业 10 日收益） | `rel_mom_ind = ret_10d(stock) - ret_10d(industry_index)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `rel_mom_ind_20d` | sector | 高优 | 行业相对动量（个股 20 日收益 − 所属行业 20 日收益） | `rel_mom_ind = ret_20d(stock) - ret_20d(industry_index)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `rel_mom_ind_250d` | sector | 高优 | 行业相对动量（个股 250 日收益 − 所属行业 250 日收益） | `rel_mom_ind = ret_250d(stock) - ret_250d(industry_index)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 480 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `rel_mom_ind_3d` | sector | 高优 | 行业相对动量（个股 3 日收益 − 所属行业 3 日收益） | `rel_mom_ind = ret_3d(stock) - ret_3d(industry_index)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `rel_mom_ind_5d` | sector | 高优 | 行业相对动量（个股 5 日收益 − 所属行业 5 日收益） | `rel_mom_ind = ret_5d(stock) - ret_5d(industry_index)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `rel_mom_ind_60d` | sector | 高优 | 行业相对动量（个股 60 日收益 − 所属行业 60 日收益） | `rel_mom_ind = ret_60d(stock) - ret_60d(industry_index)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `rel_turnover_ind_20d` | sector | 低优 | 行业内相对换手 = 个股 20 日平均换手率 − 所属行业成员均值 | `rel_turnover_ind = turnover_rate - ind_mean(turnover_rate)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor, stock_finance |
| `rel_vol_ind_20d` | sector | 低优 | 行业内相对波动 = 个股 20 日波动 − 所属行业成员 20 日波动均值 | `vol20 = roll_std(ret, 20, min_periods=5); rel_vol_ind_20d = vol20 - ind_mean(vol20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `ret_autocorr_1d_20` | momentum | 高优 | 日收益一阶自相关（20 日窗口）：正 = 涨后跟涨（动量），负 = 均值回归（反转） | `AC = Corr(ret, shift(ret, 1), 20)  # 参考库: ret_w.rolling(20).corr(ret_w.shift(1))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_daily, stock_adj_factor |
| `ret_efficiency_20` | momentum | 高优 | 20 日路径效率 = \|20 日净收益\| / 20 日收益的绝对值和（Kaufman ER） | `ER = \|Σ_{20} r\| / Σ_{20} \|r\|, r = 日收益  # 参考库同名侧写: kama_efficiency_20（\|adj.diff(20)\| / rolling_sum(\|adj.diff()\|)）` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_daily, stock_adj_factor |
| `ret_ind_rel_1d` | sector | 高优 | 行业相对 1 日收益 = 个股当日收益 − 所属行业当日收益 | `ind_ret = df.groupby(['Date','industry'])['ret'].transform('mean'); ret_ind_rel_1d = df['ret'] - ind_ret` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | tdx_daily, tdx_blocks, stock_daily, stock_adj_factor |
| `ret_kurt_20` | risk | 低优 | 20 日收益峰度（日收益滚动四阶矩，反向） | `kurt = _ret_wide(daily).rolling(20, min_periods=15).kurt()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_daily, stock_adj_factor |
| `ret_skew_20` | risk | 低优 | 20 日收益偏度（日收益滚动三阶矩，反向） | `skew = _ret_wide(daily).rolling(20, min_periods=15).skew()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_daily, stock_adj_factor |
| `revenue_cagr_3y` | growth | 高优 | 3 年营业收入复合增速 = (营收TTM_t / 营收TTM_{t−12Q})^(1/3) − 1 | `EPS_Growth_5Y = (BasicEPS_Y_t / BasicEPS_Y_{t-1260})^(1/5) - 1  （因子库.md #2 Growth 章节 PEG 系 pegh5 的复合增速式，逐字；本实现把 5 年/1260 天换成 3 年/12 个报告期，标的换成营业收入 TTM）` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income |
| `reversal_2d` | momentum | 低优 | 2 日收益（超短周期反转的原始信号，方向在因子外部用） | `Reversal2 = hfq_close(T) / hfq_close(T-2) - 1  # 参考库: close.groupby(Code).pct_change(2)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 23 | stock_daily, stock_adj_factor |
| `roa_ttm` | quality | 高优 | 总资产收益率（TTM）= 归母净利润TTM / 期末总资产 | `ROA = NetProfit / TotalAssets` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income, stock_balancesheet |
| `roc_12` | technical | 高优 | 12 日 ROC 变动率（后复权），无量纲 | `adj = _adjusted_close(daily)
roc = adj.groupby(level="Code").transform(
    lambda s: s.pct_change(12, fill_method=None)
)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 44 | stock_daily, stock_adj_factor |
| `roe_ttm` | quality | 高优 | 净资产收益率（TTM）= 归母净利润TTM / 归母股东权益 | `ROE = NetProfit_Parent_TTM / SE_without_MI` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income, stock_balancesheet |
| `roe_ttm_lag63d` | quality | 高优 | 63 个交易日（约一季度）前的 ROE(TTM) —— 刻画「季报之间的漂移」 | `ROE = NetProfit_Parent / TotalEquity
   Factor = ROE_TTM(T − 63 个交易日)   # 参考库 lag 参数默认 0` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income, stock_balancesheet |
| `rsi_14` | technical | 低优 | 14 日 Wilder RSI（相对强弱指标），值域 [0,100] | `1. Gain = Max(Close - PrevClose, 0)
2. Loss = Max(PrevClose - Close, 0)
3. AvgGain = EMA(Gain, 14) [使用 Wilder's Smoothing]
4. AvgLoss = EMA(Loss, 14) [使用 Wilder's Smoothing]
5. RS = AvgGain / AvgLoss
6. RSI = 100 - 100 / (1 + RS)
其中 Wilder's Smoothing 等价于 EMA with com = period - 1
参数: period: RSI 周期，默认为 14` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 300 | stock_daily, stock_adj_factor |
| `rsi_spread_6_14` | technical | 高优 | 日频 RSI(6) − RSI(14)：短期动能相对中期动能的强弱差 | `delta = pct_chg / 100
def _rsi(delta, n):
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing: alpha=1/n.  min_periods avoids presenting a
    # partially initialised oscillator as a fully formed RSI value.
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = safe_divide(avg_gain, avg_loss + 1e-10)
    return 100.0 - 100.0 / (1.0 + rs)

spread = _rsi(delta, 6) - _rsi(delta, 14)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 300 | stock_daily, stock_adj_factor |
| `rsrs_beta_18` | technical | 高优 | RSRS 斜率 beta：18 日 high~low 滚动 OLS 斜率（阻力相对支撑的上升速度） | `# 折算到复权空间再回归,避免除权日 high/low 阶跃污染斜率(见 _compute_rsrs_beta)
scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
high = daily["high"] * scale
low = daily["low"] * scale
beta, _ = _compute_rsrs_beta(high, low, window=18)
# 本项目口径：beta = roll_cov(low, high, 18) / roll_var(high, 18)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_daily, stock_adj_factor |
| `rsrs_zscore_18` | technical | 高优 | RSRS 标准分：18 日 OLS 斜率相对自身 200 日历史的标准分 | `scale = _adjusted_close(daily) / daily["close"].replace(0, np.nan)
high = daily["high"] * scale
low = daily["low"] * scale
beta, _ = _compute_rsrs_beta(high, low, window=18)

# Z-score relative to 400-day rolling window
roll_mean = rolling_group_mean(beta, 400, min_periods=100)
roll_std = rolling_group_std(beta, 400, min_periods=100)
zscore = safe_divide(beta - roll_mean, roll_std + 1e-8)
# 本项目口径：窗口取 200 个交易日（M=200，见 因子库.md 的 rsrs 条目）` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 400 | stock_daily, stock_adj_factor |
| `rv_term_structure` | coupling | 低优 | 波动率期限结构 = 20 日 RV / 60 日 RV（短端相对长端的高低） | `RV_n = std(ret(1), n);  TermStructure = RV_20 / RV_60` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 320 | stock_daily, stock_adj_factor |
| `seal_float_strength_20` | disclosure_detail | 高优 | 20 日涨停事件平均封单流通比 | `event_mean(sealed_flow_ratio,20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_limit_up |
| `seal_reopen_pressure_20` | disclosure_detail | 低优 | 20 日涨停事件平均对数开板次数 | `event_mean(log1p(open_count),20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_limit_up |
| `seal_turnover_strength_20` | disclosure_detail | 高优 | 20 日涨停事件平均对数封单成交比 | `event_mean(log1p(sealed_turnover_ratio),20)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_limit_up |
| `share_issuance_yoy` | growth | 低优 | 总股本同比（股本扩张 = 增发/送转/股权激励的摊薄幅度） | `growth = total_share(T) / total_share(T-4报告期) - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_balancesheet |
| `short_balance_ratio_change_20d` | margin | 低优 | 融券余额占比变化 =（融券余额 / 流通市值）的 20 个交易日变化（空头加仓，低者优） | `total_mv_m = fin["total_mv"].reindex(m.index); short_ratio = safe_divide(m["rqye"], total_mv_m); change = short_ratio.groupby(level="Code").diff(20); return cross_sectional_rank(-change)【本实现：total_mv → 流通市值 = 未复权 close × float_share（用户口径）；不取反（方向交给 higher_is_better=False）；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_margin_detail, stock_daily, stock_finance |
| `short_interest_volatility_20d` | margin | 低优 | 融券余量 20 日变异系数 CV=std/mean（空头仓位稳定性，低者优） | `roll_std = rqyl.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).std()); roll_mean = rqyl.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean()); cv = safe_divide(roll_std, roll_mean); cv = cv.clip(0, 2); return cross_sectional_rank(-cv)【本实现：不取反（方向交给 higher_is_better=False）；clip 交给引擎缩尾；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_margin_detail |
| `short_sell_volume_ratio` | margin | 高优 | 融券卖出占比 = 融券卖出量 / 当日成交量（活跃做空） | `vol_t = d["vol"].reindex(m.index); ratio = safe_divide(m["rqmcl"], vol_t); ratio = ratio.clip(0, 1); return cross_sectional_rank(ratio)【本实现：分子分母取**同一个交易日**（见 note），再整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | stock_margin_detail, stock_daily |
| `short_squeeze_risk` | margin | 高优 | 逼空风险 = 融券余量 / 融资余额（做空拥挤，高者优=潜在逼空反转） | `squeeze = safe_divide(m["rqyl"], m["rzye"]); squeeze = squeeze.clip(0, 10); return cross_sectional_rank(squeeze)【本实现：clip/rank 交给引擎；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | stock_margin_detail |
| `short_term_reversal_5` | momentum | 高优 | 5 日短周期反转 = −(5 日动量)（近 5 日超买者排前） | `ShortReversal5 = -Mom5 = -(hfq_close(T) / hfq_close(T-5) - 1)  # 参考库: cross_sectional_rank(-mom5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 29 | stock_daily, stock_adj_factor |
| `small_order_crowding` | fundflow | 低优 | 小单拥挤度 = 小单毛量占比相对自身 20 日中枢的偏离（散户异常拥挤=见顶信号，低者优） | `small_vol = mf["buy_sm_vol"] + mf["sell_sm_vol"]
total_vol = (buy_sm_vol + sell_sm_vol + buy_md_vol + sell_md_vol
             + buy_lg_vol + sell_lg_vol + buy_elg_vol + sell_elg_vol)
small_pct = small_vol / total_vol.replace(0, np.nan)
return cross_sectional_rank(-small_pct)
★ 本实现（口径级偏离，见 note）：
crowd = small_pct / small_pct.rolling(20, min_periods=10).mean() - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_main_fund_flow, stock_daily |
| `sortino_ratio_60` | risk | 高优 | 60 日 Sortino 比率 = 均收益 / 下行波动（正向） | `mean60 = ret.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).mean()); down = ret.where(ret < 0); down_std = down.groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=5).std()); sortino = safe_divide(mean60, down_std + 1e-10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_daily, stock_adj_factor |
| `sp_ttm` | value | 高优 | 营收市值比（市销率倒数）= 营业收入TTM / 总市值 | `sp = 1.0 / finance["ps_ttm"].replace(0, np.nan)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income, stock_daily, stock_finance |
| `super_large_order_intensity` | fundflow | 高优 | 超大单强度 = 5 日累计超大单净买入额 / 5 日累计八列毛额（主力大额持续收集筹码） | `elg_net = (mf["buy_elg_amount"] - mf["sell_elg_amount"]) / _total_amount(mf)
return cross_sectional_rank(elg_net)
★ 本实现（口径级偏离，见 note）：分子分母各自 5 日累计后再相除
intensity = elg_net.rolling(5, min_periods=3).sum() / total.rolling(5, min_periods=3).sum()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 29 | stock_main_fund_flow, stock_daily |
| `tail_risk_pct_60` | risk | 低优 | 尾风险频率 = 60 日内 \|z\|>2 的极端收益占比（反向） | `z = safe_divide(ret - mean60, std60 + 1e-10); freq = z.abs().gt(2).groupby(level="Code").transform(lambda s: s.rolling(60, min_periods=30).mean())` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 240 | stock_daily, stock_adj_factor |
| `td_setup_count` | pattern | 高优 | TD setup：20 日内 close ≤ 4 日前 close 的交易日**占比**（0~1，超卖衰竭程度） | `cond = (adj_close <= adj_close.shift(4)).astype(int); seg = (~cond).groupby(Code).cumsum(); count = cond.groupby([Code, seg]).cumsum(); 参考库: cross_sectional_rank(count)   # 连续段计数` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 70 | stock_daily, stock_adj_factor |
| `three_black_crows` | pattern | 高优 | 三只黑鸦：20 日内「收盘<开盘 且 收盘<前收」的交易日**占比**（0~1，连续阴跌强度） | `cond = (close < open) & (close < pre_close); seg = (~cond).groupby(Code).cumsum(); count = cond.groupby([Code, seg]).cumsum(); 参考库: cross_sectional_rank(count)   # 连续段计数` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `time_since_52w_high` | momentum | 低优 | 距最近一次 250 日新高的**交易日**数（0 = 今天创的年内新高） | `at_high = (hfq_close >= roll_max(hfq_close, 252));  DaysSince = bars since the last at_high（截断到 249）` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 960 | stock_daily, stock_adj_factor |
| `top_list_net_rate_20` | event | 高优 | 龙虎榜净买率 = 20 日上榜净买额 / 20 日上榜成交额（无上榜则 NaN） | `NetRate_20 = sum(NetAmount, 20d) / sum(Amount, 20d)
# NetAmount = 龙虎榜买入额 - 卖出额；Amount = 龙虎榜成交额
# 参考库 net_rate 的日频定义 = NetAmount / Amount * 100，本因子是它的 20 日聚合` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_top_list |
| `total_leverage_ratio` | margin | 低优 | 总杠杆率 = 融资融券余额 / 流通市值（高杠杆=平仓风险大，低者优） | `mv_t = f["total_mv"].reindex(m.index); ratio = safe_divide(m["rzrqye"], mv_t); ratio = ratio.clip(0, 0.5); return cross_sectional_rank(-ratio)【本实现：total_mv → 流通市值 = 未复权 close × float_share（用户口径）；方向用 higher_is_better=False 标注而不是写进值；结果整体下移 1 个交易日】` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | stock_margin_detail, stock_daily, stock_finance |
| `trend_strength_60` | momentum | 高优 | 60 日趋势强度 = 带符号的线性回归 R²（价格对时间回归） | `R2 = Corr(t, hfq_close, 60)^2 （= OLS 的 R²）;  Factor = sign(Corr) * R2  # 参考库无同名因子；同类见 rsrs_r2_18 / macd_trend_strength` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_daily, stock_adj_factor |
| `trix_12_20` | technical | 高优 | TRIX：(后复权收盘价的 12 日三重 EMA) 的 20 日变化率 | `wide = _adjusted_close(daily).unstack("Code")
trix = _trix_wide(wide)
# _trix_wide = 三重指数平滑(12) 的 20 日变化率` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 400 | stock_daily, stock_adj_factor |
| `trix_signal_gap` | technical | 高优 | TRIX 与其 20 日信号线的乖离（TRIX − MA20(TRIX)）/ \|MA20(TRIX)\| | `wide = _adjusted_close(daily).unstack("Code")
trix = _trix_wide(wide)
signal = trix.rolling(20, min_periods=10).mean()
gap = trix.sub(signal).div(signal.abs() + 1e-10)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 400 | stock_daily, stock_adj_factor |
| `turnover_anomaly_20` | liquidity | 低优 | 换手率异常：20 日均换手 / 60 日均换手 − 1 | `to_20 = rolling_group_mean(turnover, 20); to_60 = rolling_group_mean(turnover, 60); anomaly = to_20 / to_60 - 1.0` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 128 | stock_daily, stock_finance |
| `turnover_f_20` | value | 低优 | 20 日平均自由流通换手率（% = vol/free_share×100），低换手排前 | `avg_turnover = turnover_rate_f.groupby(level='Code').transform(
    lambda s: s.rolling(20, min_periods=10).mean())` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_finance |
| `turnover_f_delta_5` | value | 低优 | 自由流通换手率的 5 日变化率（下降 = 浮筹被吸收） | `delta = turnover_rate_f.groupby(level='Code').transform(
    lambda s: s.pct_change(5, fill_method=None))
delta = delta.clip(-1, 3)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_finance |
| `turnover_std_20` | liquidity | 低优 | 换手率波动：20 日换手率标准差（流动性不稳定的代理） | `to_vol = rolling(20, min_periods=10).std(); Factor = -to_vol` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_finance |
| `ulcer_index_20` | risk | 低优 | 溃疡指数 = 20 日窗口内回撤平方均值的平方根（回撤面积，反向） | `peak = adj.groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).max()); dd = adj / peak.replace(0, np.nan) - 1.0; dd_sq = (dd ** 2).groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).mean()); ulcer = dd_sq.pow(0.5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 92 | stock_daily, stock_adj_factor |
| `var_95_20` | risk | 高优 | 20 日 VaR(95%) = 日收益的 5% 历史分位（≤0，越负尾部越厚） | `var = _ret(daily).groupby(level="Code").transform(lambda s: s.rolling(20, min_periods=10).quantile(0.05))` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_daily, stock_adj_factor |
| `vol_120` | risk | 低优 | 120 个交易日已实现波动率（日收益滚动标准差） | `ReturnStd = StdDev(DailyReturn) over 6*21 trading days` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 240 | stock_daily, stock_adj_factor |
| `vol_clustering_20` | risk | 低优 | 波动聚集 = \|日收益\| 的 20 日 lag-1 自相关（反向） | `abs_ret = _ret(daily).abs().unstack("Code"); lag = abs_ret.shift(1); ac = abs_ret.rolling(20, min_periods=10).corr(lag)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `vol_of_rv` | coupling | 低优 | 波动的波动 = 20 日 RV 在 60 日窗口内的标准差（不稳定度） | `VoV = std( roll_std(ret(1), 20), 60 )` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 320 | stock_daily, stock_adj_factor |
| `vol_of_vol_20` | risk | 低优 | 波动之波动 = \|日收益\| 的 20 日标准差（反向） | `roll(df, "absret", 20, "std", min_periods=5)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 56 | stock_daily, stock_adj_factor |
| `vol_ratio_ma5_ma20` | liquidity | 高优 | 量能短长比：5 日均量 / 20 日均量 | `ma_5 = rolling_group_mean(vol, 5); ma_20 = rolling_group_mean(vol, 20); ratio = safe_divide(ma_5, ma_20 + 1e-8)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily |
| `volume_dry_up` | liquidity | 低优 | 缩量（地量）：20 日最低成交量 / 20 日均量 | `vol_min_20 = rolling(20, min_periods=10).min(); vol_ma_20 = rolling(20, min_periods=10).mean(); dryness = vol_min_20 / vol_ma_20.replace(0, np.nan)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily |
| `volume_price_divergence_score` | pattern | 低优 | 量价背离得分：20 日价动量截面 z − 20 日量动量截面 z（值高 = 价升量缩） | `price_mom=adj_close.pct_change(20); vol_mom=vol.pct_change(20); pr=cs_rank(price_mom); vr=cs_rank(vol_mom); 参考库: cross_sectional_rank(pr - vr)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `volume_ratio` | value | 低优 | 量比 = 当日成交量 / 过去 5 日平均成交量（低者优） | `vol_ratio = finance["volume_ratio"]
return cross_sectional_rank(-vol_ratio)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 40 | stock_daily, stock_finance |
| `volume_tilt_20` | liquidity | 高优 | 量能倾斜：20 日量加权收益 − 等权收益（正 = 收益主要来自放量日） | `wsum = rolling_sum(pct_chg * vol, 20, min_periods=10); vsum = rolling_sum(vol, 20, min_periods=10); vw = safe_divide(wsum, vsum); ew = rolling_mean(pct_chg, 20, min_periods=10); tilt = vw - ew` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily, stock_adj_factor |
| `vwap_daily_deviation` | liquidity | 高优 | 收盘价对当日 VWAP 的偏离：close / (amount/vol) − 1 | `vwap = safe_divide(amount, vol); deviation = safe_divide(close - vwap, vwap)` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 30 | stock_daily |
| `winner_rate` | chip | 低优 | 获利盘比例 = 现价以下的筹码占比（未复权 close 口径） | `winner_rate = below_close; return cross_sectional_rank(-perf["winner_rate"])` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips |
| `winner_rate_acceleration` | chip | 低优 | 获利盘比例的加速度（二阶差分：5 日变化的 5 日变化） | `wr_chg = wr.groupby(level="Code").transform(lambda s: s.diff(5)); accel = wr_chg.groupby(level="Code").transform(lambda s: s.diff(5)); return cross_sectional_rank(-accel)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 40 | stock_cyq_chips |
| `winner_rate_reversal_signal` | chip | 高优 | 获利盘极端反转信号 = −\|获利盘 − 0.5\|（50% 附近=多空平衡=排前） | `distance = -(cyq["winner_rate"] - 0.5).abs(); return cross_sectional_rank(distance)` | 2018-01-02 | 2018-01-02 → 2026-09-21 | 20 | stock_cyq_chips |
| `yoy_revenue` | growth | 高优 | 营业收入（TTM）同比增速 | `YoY = Revenue_TTM_t / Revenue_TTM_{t-4Q} - 1` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 700 | stock_income |
| `zero_return_fraction_20` | liquidity | 低优 | 零收益日占比：20 日里 \|涨跌幅\| < 0.1% 的交易日的比例 | `zero = daily["pct_chg"].abs().lt(0.1).astype(float); freq = rolling(20, min_periods=10).mean()` | 2018-01-01 | 2018-01-02 → 2026-09-21 | 60 | stock_daily |

## 口径备注（实测坑）

- **`adx_14`**：参考库出处：factors.md `类别 price` / trend_pattern.py 的 adx_14。★ 参考库的 `_directional_movement` 是私有函数、文档未展开；本文件按 Wilder 的标准定义实现（up=high−high[-1]、dn=low[-1]−low，plus_dm 取 up>dn 且 up>0，minus_dm 对称，各自与 TR 一起做 alpha=1/14 的 Wilder 平滑，DI=100×DM/ATR）。★ 偏离：参考库用未复权 `pre_close` 且乘 `scale` 折算到复权空间；本文件直接用`shift(hfq_close,1)` 当昨收（复权空间里两者在有交易的日子恒等，停牌日更准），不再需要 scale。★ `min_count=7` 对齐参考库的 `min_periods=7`：DX 在「多空双方 DM 同时为 0」（长期停牌/一字板）时是 NaN，满窗要求会让 ADX 在真实数据上留下 14 天的空洞。★ ADX ∈ [0,100]；DX = 100×|DI+−DI-|/(DI++DI-) ∈ [0,100]，其 14 日均值同界。
- **`afx_bs_cap_rese`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_const_materials`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_debt_invest`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_decr_in_disbur`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_deriv_assets`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_deriv_liab`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_div_payable`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_div_receiv`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_estimated_liab`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_fix_assets_total`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_fixed_assets_disp`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_hfs_assets`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_int_payable`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_int_receiv`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_invest_real_estate`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_loan_oth_bank`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_long_pay_total`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_lt_amor_exp`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_lt_eqt_invest`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_lt_payable`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_lt_payroll_payable`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_lt_rec`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_nca_within_1y`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_ordin_risk_reser`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_oth_assets`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_oth_comp_income`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_oth_cur_assets`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_oth_cur_liab`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_oth_debt_invest`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_oth_eqt_tools`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_oth_nca`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_oth_ncl`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_oth_pay_total`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_oth_payable`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_oth_rcv_total`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_payroll_payable`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_produc_bio_assets`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_pur_resale_fa`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_sold_for_repur_fa`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_special_rese`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_specific_payables`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_surplus_rese`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_taxes_payable`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_trading_fl`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_bs_undistr_porfit`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_beg_bal_cash`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_beg_bal_cash_equ`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_c_cash_equ_end_period`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_c_disp_withdrwl_invest`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_c_fr_oth_operate_a`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_c_paid_invest`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_c_recp_cap_contrib`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_c_recp_return_invest`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_decr_def_inc_tax_assets`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_decr_oper_payable`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_eff_fx_flu_cash`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_end_bal_cash_equ`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_finan_exp`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_ifc_cash_incr`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_im_n_incr_cash_equ`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_im_net_cashflow_oper_act`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_incl_cash_rec_saims`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_incl_dvd_profit_paid_sc_ms`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_incr_def_inc_tax_liab`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_incr_oper_payable`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_loss_disp_fiolta`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_loss_fv_chg`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_loss_scr_fa`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_n_depos_incr_fi`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_n_disp_subs_oth_biz`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_n_incr_clt_loan_adv`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_n_incr_loans_oth_bank`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_n_recp_disp_fiolta`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_n_recp_disp_sobu`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_net_profit`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_oth_cash_pay_oper_act`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_oth_cash_recp_ral_fnc_act`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_oth_cashpay_ral_fnc_act`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_oth_pay_ral_inv_act`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_oth_recp_ral_inv_act`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_pay_handling_chrg`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_proc_issue_bonds`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_stot_cash_in_fnc_act`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_stot_cashout_fnc_act`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_stot_inflows_inv_act`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_stot_out_inv_act`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_cf_use_right_asset_dep`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_adminexp_of_gr`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ar_turn`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_arturn_days`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_assets_turn`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_basic_eps_yoy`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_bps`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ca_turn`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_capital_rese_ps`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_capitalized_to_da`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_cash_to_liqdebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_cash_to_liqdebt_withinterest`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_cfps`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_cfps_yoy`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_cogs_of_sales`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_current_exint`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_daa`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_diluted2_eps`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_dp_assets_to_eqt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_dtprofit_to_profit`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ebit`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ebit_of_gr`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ebit_ps`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ebit_to_interest`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ebitda`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_eqt_to_debt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_eqt_to_interestdebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_eqt_to_talcapital`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_equity_yoy`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_expense_of_sales`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_extra_item`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_fa_turn`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_fcfe`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_fcff`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_finaexp_of_gr`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_fixed_assets`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_gc_of_gr`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_gross_margin`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_impai_ttm`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_interestdebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_interst_income`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_inv_turn`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_invest_capital`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_investincome_of_ebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_invturn_days`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_longdebt_to_workingcapital`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_n_op_profit_of_ebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_netdebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_networking_capital`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_non_op_profit`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_noncurrent_exint`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_npta`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ocf_to_debt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ocf_to_interestdebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ocf_to_netdebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ocf_to_opincome`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ocf_to_or`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ocf_to_profit`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_ocfps`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_op_income`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_op_to_debt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_op_to_ebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_op_to_liqdebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_opincome_of_ebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_profit_dedt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_adminexp_to_gr`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_dtprofit`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_dtprofit_to_profit`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_exp_to_sales`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_finaexp_to_gr`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_gc_to_gr`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_gr_qoq`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_gr_yoy`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_gsprofit_margin`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_impair_to_gr_ttm`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_investincome`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_investincome_to_ebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_netprofit_margin`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_netprofit_qoq`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_netprofit_yoy`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_ocf_to_or`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_op_qoq`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_op_to_gr`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_op_yoy`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_opincome`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_opincome_to_ebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_profit_qoq`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_saleexp_to_gr`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_q_salescash_to_or`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_rd_exp`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_retained_earnings`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_retainedps`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_revenue_ps`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_roe_avg`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_saleexp_to_gr`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_salescash_to_or`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_surplus_rese_ps`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_tangible_asset`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_tangibleasset_to_netdebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_tax_to_ebt`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_total_fa_trun`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_valuechange_income`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_fi_working_capital`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_ass_invest_income`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_basic_eps`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_biz_tax_surchg`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_comm_exp`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_comm_income`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_compr_inc_attr_m_s`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_compr_inc_attr_p`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_continued_net_profit`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_ebitda`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_fin_exp_int_inc`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_forex_gain`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_int_exp`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_int_income`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_n_oth_b_income`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_oth_compr_income`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`afx_is_other_bus_cost`**：本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。只取已公告的12月31日年度报告；单季度字段在年报中为Q4口径。
- **`amihud_asymmetry_20`**：两个滚动均值各自在**同类日**上取（涨跌日分别成池），min_count=5 = 参考库 min_periods=5 —— 注意它数的是「窗口内有几天上涨」而不是「有几个有效值」，所以极端单边行情（20 日全涨）会让分母变 NaN，这是参考库的原意。收益用 ctx.ret（后复权），停牌日 NaN 自动落在两个池子之外。★ 分母保护：用 safe_div(min_abs_den=1e-12) 代替原文的 `down/(up+1e-12)`。两者**只在 up≫1e-12 时等价**——原文那个加法项在参考库的 ×1e8 量纲下是纯零保护（illiq~1e-2），而本族不带 ×1e8（见模块 docstring 第 4 条），illiq 只有 ~1e-10，照抄会把比值系统性扰动。用 pandas 逐格复算实测：与原文写法差 max 1.5e-1 / 10.6%（正是这个常数项造成），改用 safe_div 后语义等价（up 为 NaN 时同样得 NaN）且无扰动。注：正/负两侧都少于 5 个有效交易日时分母为 NaN → 输出 NaN，这是参考库的原意。
- **`amihud_daily_5`**：★ 偏离参考库：去掉原文的 ×1e8 常数（原文 `df['absret']/df['amount']*1e8`），与同族 amihud_daily_20 / liquidity_shock_20 保持同一量纲（1/元）；截面 rank 完全不受常数影响。min_count=2 = 参考库 min_periods=2。5 日窗对停牌更敏感、对资金流出更灵敏，与 20 日版互补。
- **`amihud_parkinson_ratio`**：参考库同名因子的思路：**单位波动带来的价格冲击**。分母用 Parkinson 波动（只用日内高低价、对成交稀疏不敏感），比用收盘价标准差更干净。⚠️ 单位：`stock_daily.amount` 是元，Amihud 保留元级量纲（不乘 1e8）——这里只取**比率**，量纲自己约掉，乘不乘常数不影响截面排序。⚠️ high/low 用**未复权**价：Parkinson 是日内比值，除权日会有一个交易日的失真，但 20 日均值把它摊薄到可忽略（与参考库一致）。
- **`amount_mom_accel`**：量能一阶动量反映资金流入方向，二阶（加速）反映资金行为切换。ctx.pct_change 的 min_abs_den=1e-12 对成交额（元）足够松，不会误伤。停牌日成交额 NaN → 变化率 NaN → 该格为 NaN（不补 0）。
- **`amount_ratio_20`**：成交额比成交量更能反映资金参与度。窗口**含当日**（参考库就是含当日的 rolling），所以放量日 a_ratio > 0。min_count=10 对齐参考库。成交额是元、量级 1e7~1e9，分母用 safe_div(min_abs_den=1.0) 挡 0。
- **`ar_ap_to_revenue`**：参考库 `因子库.md` 5、Quality #11，公式逐字。**经济含义**：预收款 = 客户先付钱（**下游议价力 / 订单储备**），预付款 = 先付给上游（**被上游占用**）。两者之差是**净占款能力**。★ 与已注册的 `cash_conversion_cycle` 的区别：那个测的是「存货+应收−应付」的**周转天数**（营运效率），本因子测的是预收/预付这一对**合同负债**（议价地位）——既不与周转天数共线，也不与任何利润率水平共线。★★ **零额外 TTM 机械**：两个分子科目都是**纯时点**（资产负债表），只有分母是 TTM；所以本因子的活跃度高于纯 TTM 比率。`prepayment` / `adv_receipts` 这两个字段**此前零个因子消费过**。
- **`aroon_down_25`**：参考库出处：factors.md `类别 price` / technical_daily.py 的 aroon_down_25。参考库取 `rank(-aroon)`（近期破位排后），本因子返回原始 aroon 值并标 higher_is_better=False —— 高 = 距上次新低很久 = 空头趋势弱，是好事。★ 值域 [4, 100]，同 aroon_up_25。基于 hfq 基座，除权日不产生假新低。
- **`aroon_up_25`**：参考库出处：factors.md `类别 price` / technical_daily.py 的 aroon_up_25。★ 实现：`ctx.roll_argmax(close, 25)` 的返回值就是「窗口极值距今天多少个 bar」（0 = 今天创的窗口新高），正好是文档里的 `days_since`，不需要额外换算。★ 值域：[4, 100]（days_since 取整 0..24，(25−24)/25×100 = 4 是下界）。  参考库的 `_rolling_extreme_age` 若以 1 为起点则会得到 (25−25)/25=0，  本文件按框架语义（0 起）实现，下界是 4 —— 差一个 bar 的常数，  对同一截面的排序没有影响。★ 窗口 25 天用 `roll_max/argmax`（O(T·C·n)）没有问题；不做 n>250 的版本。
- **`asset_growth_qoq`**：★ 文档的 t-1 必须理解为「上一个**报告期**」而不是「63 个交易日前」。按日频 lag 实现的话，一年里大部分时间比较的是同一期数据，因子会塌成 0
- **`atr_14_ratio`**：参考库出处：factors.md `类别 risk` / technical_pattern.py 的 atr_ratio_20（窗口 20）。本因子按任务命名为 `atr_14_ratio`，窗口改成 14 ——与 RSI14 / MFI14 / ADX14 的周期保持一致，便于和同族振荡器对齐观察。★ **已按价格归一化**（÷ hfq 收盘价）：这是本任务明确要求的 - ATR 是价格单位，后复权基座大的股票 ATR 天然大，不归一化就是在选复权因子。★ 参考库取 `rank(-atr_pct)`（低波平稳排前，低波异象），本因子返回原始比率并标 `higher_is_better=False`。★ TR 用复权昨收（`shift(hfq_close,1)`），不是未复权 pre_close，参考库的 `× scale` 折算由价格层统一完成。★ 末尾 `np.maximum(...,0)`：TR 恒 >= 0，比值也恒 >= 0，  该保护只消掉 cumsum 差分的 ±1e-16 负残差（实测 min = −2e-16）。
- **`avg_cost_premium`**：★ 口径偏离：参考库用 `_close_adj_basis(daily)`（复权），本项目**必须**用ctx.px('close')（未复权）—— 本摘要表的 mean 是未复权价（实测 close/p50≈1.01、close/mean≈1.07，见模块 docstring 的验证）。照抄参考库的复权折算会错两个数量级。★ 另一处偏离：**去掉**参考库的 `clip(-1, 5)` —— 契约规定 winsor 一律由引擎（1%/99% 截面 winsorize）统一做，因子内不做。单边下界 −1 本来也压不住什么，mean>0 时下溢只到 −1（close→0）。
- **`beta_60`**：★ 基准偏离：参考库的 `_market_proxy(wide)` 是**全池等权**市场收益，本实现按要求改用**沪深300（000300.SH）**——它是可投资的、有真实指数数据的基准，且与下游基准一致。min_count=30 = 参考 `_rolling_beta(..., 60, 30)`。实现：市场收益取 (T,1) 广播进 roll_cov/roll_var（**两者都按各自窗口内的有效对数/有效天数做分母**，与 pandas 的 `rolling().cov()/var()` 同口径）。停牌日 ret 为 NaN → 该窗口若有效天数不足 30 则为 NaN。★★ v2（2026-09-17）：`mathx.roll_cov` 修了分母用固定 n 的 bug（见该函数注释）——本因子是它的下游，历史值整体重算过；修前在「窗口内有停牌/缺失」的股票上是失真的值。
- **`bias_20`**：参考库出处：factors.md `类别 price` / price.py 的 bias_20。实现为 `safe_div(close − ma20, ma20)`，与 `close/ma20 − 1` 在数值上略有差异（后者在 ma20 很大时更稳），但同为无量纲比值，截面排序一致。★ 复权：走 hfq_close，除权日不伪造负乖离（参考库注释的原话）。★ 方向：参考库正向排名（价格偏离中期成本的程度，极端正 = 超买但强势延续）。
- **`big_vs_small_divergence_5d`**：分层净额口径，两项都用**同一个**分母（八列毛额之和），所以它衡量的是「大单相对小单的净流入加速」而不是绝对水平。注意与 `mf_smart_dumb_divergence` 的区别：后者按**各自档位毛额**归一（当日水平），本条按**同一个总毛额**归一（5 日变化）——实测两者相关约 0.5，不冗余。
- **`bollinger_squeeze`**：参考库出处：factors.md `类别 price` / trend_pattern.py 的 bollinger_squeeze。★ **偏离（按任务要求）**：参考库给的是 `rank(-bandwidth)` —— 带宽的**截面**排名；任务明确要求 `bollinger_squeeze` = 带宽的**历史分位**，所以本文件实现为 `ctx.roll_rank(bandwidth, 250)`：今天的带宽在**自己过去 250 个交易日**里的百分位（∈[0,1]）。这不是量纲问题而是**信息维度**不同：截面排名度量「今天谁比谁窄」，历史分位度量「这只股票今天相对自己一年来的水平窄不窄」——后者才是「低波动后往往伴随方向性突破」这句话的正确横截面形态（每只股票与自己比）。★ 250 日窗口：任务提到 `roll_max/min` 做 n>250 需要先报备。本因子用的是 `roll_rank`（分块跨步视图，O(T·C·n) 时间 / O(step·C·n) 内存，实测不构成瓶颈），不是 `roll_max/min`，所以按契约要求在此明示。若主 Agent 认为 250 过大，改成 120 只需改这一个常量。★ `higher_is_better=False`：分位低 = 挤压 = 突破前兆。
- **`bollinger_width_20`**：参考库出处：factors.md `类别 price` / trend_pattern.py 的 bollinger_width_20。参考库取 `rank(-width)`（窄幅 = 蓄力），本因子返回原始带宽并标 higher_is_better=False。★ **已按价格归一化**（÷ ma）：后复权价的绝对水平取决于累计复权因子，不归一化的话这个因子实际在选「复权因子大」的股票而不是选波动。  这与参考库口径一致（它的公式里本来就除了 ma）。★ `roll_std` 口径 ddof=0，见模块 docstring 第 5 条。
- **`bp`**：★ 偏离参考库实现：factors.md 的 bp 直接用供应商 pb 字段（1/pb），我们改用 **PIT 对齐的归母权益 / 自算市值** —— 供应商 pb 的净资产是哪个版本无从审计，而 ctx.point() 按 ann_date 严格对齐，可复现。两者定义等价（PB = 市值/净资产）。口径选择：**不加**递延所得税资产（因子库 book_to_market 那版加了，差的是「递延所得税资产/市值」，量级 <1%，先按最常见口径）。★ 框架的 POSITIVE_ONLY 已把「非正权益」置 NaN（fea/deriv.py），所以负净资产公司天然落在 bp 之外 —— 与 Fama-French 剔除负账面价值的惯例一致。分母（市值）恒正，但仍走 safe_div + min_abs_den=1e6（元）做保护。
- **`bs_construction_capital_share`**：本轮扩展公式，采用参考库 Quality/现金流比率的归一化方法；财务版本表按 max(ann_date,f_ann_date,end_date) 生效，保留修订版本；累计现金流先计算 TTM，资产负债科目不做 TTM。缺失不补 0，分母绝对值至少 100 万元；负现金支付按原值保留（存在供应商冲销记录）。慢变量需在跨年度样本评价。
- **`bs_intangible_asset_share`**：本轮扩展公式，采用参考库 Quality/现金流比率的归一化方法；财务版本表按 max(ann_date,f_ann_date,end_date) 生效，保留修订版本；累计现金流先计算 TTM，资产负债科目不做 TTM。缺失不补 0，分母绝对值至少 100 万元；负现金支付按原值保留（存在供应商冲销记录）。慢变量需在跨年度样本评价。
- **`bs_near_term_debt_share`**：本轮扩展公式，采用参考库 Quality/现金流比率的归一化方法；财务版本表按 max(ann_date,f_ann_date,end_date) 生效，保留修订版本；累计现金流先计算 TTM，资产负债科目不做 TTM。缺失不补 0，分母绝对值至少 100 万元；负现金支付按原值保留（存在供应商冲销记录）。慢变量需在跨年度样本评价。
- **`bs_net_contract_to_assets`**：本轮扩展公式，采用参考库 Quality/现金流比率的归一化方法；财务版本表按 max(ann_date,f_ann_date,end_date) 生效，保留修订版本；累计现金流先计算 TTM，资产负债科目不做 TTM。缺失不补 0，分母绝对值至少 100 万元；负现金支付按原值保留（存在供应商冲销记录）。慢变量需在跨年度样本评价。
- **`bs_net_notes_to_assets`**：本轮扩展公式，采用参考库 Quality/现金流比率的归一化方法；财务版本表按 max(ann_date,f_ann_date,end_date) 生效，保留修订版本；累计现金流先计算 TTM，资产负债科目不做 TTM。缺失不补 0，分母绝对值至少 100 万元；负现金支付按原值保留（存在供应商冲销记录）。慢变量需在跨年度样本评价。
- **`bs_other_receiv_to_assets`**：本轮扩展公式，采用参考库 Quality/现金流比率的归一化方法；财务版本表按 max(ann_date,f_ann_date,end_date) 生效，保留修订版本；累计现金流先计算 TTM，资产负债科目不做 TTM。缺失不补 0，分母绝对值至少 100 万元；负现金支付按原值保留（存在供应商冲销记录）。慢变量需在跨年度样本评价。
- **`bw_beta_60`**：★ 本因子是**个股对市场宽度的敏感度**，不是市场水平量 —— 市场级数据在横断面上的离散度恒为 0，排序时与截距共线（见模块 docstring §一）。 这是全家族的**基准量**：个股对「今天有多少股票在涨」这个变化的敏感度。与已注册的 `beta_60`（对沪深300）**不是**同一个回归量：沪深300 是被大市值主导的加权指数，宽度变化是**等权**的、由中小市值主导 —— A 股的小盘股对宽度的 β 显著高于对指数的 β。⚠ 两者相关性预计较高（0.7~0.9），`dedup` 阶段定量裁决；若 |ρ| ≥ 0.95 应优先保留 `bw_resid_beta_60`（剔掉指数 β 后的正交部分）。方向取负（高暴露 = 高系统性风险）。
- **`bw_beta_asym_20`**：★ 本因子是**个股对市场宽度的敏感度**，不是市场水平量 —— 市场级数据在横断面上的离散度恒为 0，排序时与截距共线（见模块 docstring §一）。 **本文件新造**（参考库只做了对称的单 β）。构造：`d_sh > 0` 的交易日只留 `r` 与 `d_sh`，其余置 NaN；`d_sh < 0` 的日子同理 —— 两腿的掩码**互斥**，两次回归在样本上完全不相交。每腿在窗口内至少要有 6 个有效日（20 日窗的 30%）——★ 不按 N//2=10 取：一腿只占窗口中约一半的日子，`min_count=10` 等于要求「两腿都几乎满格」，会把绝大多数格子打成 NaN；6 ≈ 该腿期望命中数（10）的 60%，与契约里「N//2」的宽松度对齐。经济含义：> 0 ⇒ 市场扩散日跟涨、收敛日不跟跌（**顺周期但抗跌**）；< 0 ⇒ 市场扩散日不涨、收敛日大跌（**脆弱**）。与已注册的 `downside_vol_ratio_20` / `gain_loss_asymmetry_60` 不同：那几个的不对称是**个股自身**收益分布的不对称，本因子的不对称是**个股对市场状态的反应**不对称（条件对象完全不同）。
- **`bw_beta_change_20`**：★ 本因子是**个股对市场宽度的敏感度**，不是市场水平量 —— 市场级数据在横断面上的离散度恒为 0，排序时与截距共线（见模块 docstring §一）。 源自参考库 Class1 `market_beta_change_20`（把回归量从市场收益换成宽度变化）。★ **方向与 `bw_beta_60` 相反**：水平高是风险（排后），但**水平在抬升**是时点信号（市场关注度/资金参与度正在向这只股票集中，排前）。这正是「水平 vs 变化」在 A 股里的经典分野 —— 慢变量的水平几乎没有 alpha（已删的 `delta_*` 一族就是死在「慢水平的差分也慢」），但**快窗口相对慢窗口的变化**是对「当下发生了什么事」的直接刻画。两个窗口的 min_count 分别取 10 / 30（= N//2，见契约）。
- **`bw_breadthvol_response_20`**：★ 本因子是**个股对市场宽度的敏感度**，不是市场水平量 —— 市场级数据在横断面上的离散度恒为 0，排序时与截距共线（见模块 docstring §一）。 **本文件新造**。`bw_vol` 是「当天市场内部来回翻脸的程度」（分钟内 up 份额的标准差）—— 它与**收益的大小**无关：市场可以收平但全天剧烈震荡（bw_vol 高），也可以单边大涨（bw_vol 低）。本因子问：这只股票**在市场内部乱的时候会不会跟着乱**。> 0 ⇒ 高脆弱性（市场一有内部分歧就波动放大）。与 `vol_of_vol_20`（个股自身波动的波动）的区别：那个是**自身**波动的时间序列变化，本因子是**与市场内部状态的联动**。★ `bw_vol` 是用分钟内份额算的，不是收益 —— 所以本因子的量纲是「绝对收益 vs 份额离散度」的相关，同日截面内一致可比。
- **`bw_capture_asym_20`**：★ 本因子是**个股对市场宽度的敏感度**，不是市场水平量 —— 市场级数据在横断面上的离散度恒为 0，排序时与截距共线（见模块 docstring §一）。 抄参考库 Class1 `market_regime_sensitivity_60`（窗口 60→20、市场状态判据由「等权市场收益的符号」换成「宽度变化的符号」）。参考库原式的方向是负（防御型排前）。与已注册的 `downside_upside_vol_60` 的区别：那个是**波动**的上下行之比（离散度），本因子是**条件均值**之比（收益的**方向性**捕获）—— 一只股票可以上下行波动都很小但下行均值显著为负。分母 `|up|` 趋 0 时由 `safe_div` 给 NaN。
- **`bw_intraday_beta_20`**：★ 本因子是**个股对市场宽度的敏感度**，不是市场水平量 —— 市场级数据在横断面上的离散度恒为 0，排序时与截距共线（见模块 docstring §一）。 与 `bw_beta_60` 是**同一构念的不同腿与不同期限**：本因子的个股腿是**日内**（`hfq_close/hfq_open − 1`，剔除隔夜跳空），市场腿也用**日内**（`sh_close − sh_open`，剔除隔夜），且窗口是 20 日而不是 60 日。**收盘对收盘**的版本（`bw_beta_60`）含隔夜信息，而隔夜跳空在 A 股里由完全不同的机制驱动（集合竞价、外盘、公告），所以剥掉隔夜后的 β 更纯粹地反映「**盘中**跟随市场扩散的能力」。⚠ 与 `bw_beta_60` 相关性可能较高，`dedup` 阶段定量裁决。个股腿用 `ctx.hfq`（本项目唯一允许的价格口径），并以 `ctx.traded()` 掩码挡掉停牌日的 ffill 假收益。
- **`bw_overnight_lag_beta_20`**：★ 本因子是**个股对市场宽度的敏感度**，不是市场水平量 —— 市场级数据在横断面上的离散度恒为 0，排序时与截距共线（见模块 docstring §一）。 ★★ **全家族唯一的预测性因子** —— 其余 11 个都是同日（contemporaneous）的。合法性：`trade_date = T−1` 的宽度在 T−1 的 15:00 就已定稿，而 T 日的隔夜跳空发生在 T 日 09:25 集合竞价 —— **用 T−1 收盘后已知的量解释 T 日开盘的跳空，没有用到 T 日的任何信息**。（与 `fea/delay.py` 的「D 日当晚发布、下游 D+1 交易」同一前提。）经济含义：> 0 ⇒ 市场昨日的扩散/收敛会在**次日开盘**继续影响这只股票（滞后反应），< 0 ⇒ 隔夜被过度反应后回吐。★ 与 `idt_overnight_gap` 的区别：那个是**当日跳空本身**（同期量），本因子是**跳空与昨日市场状态的关系**（跨期、市场条件）。隔夜跳空用 `hfq_open(T)/hfq_close(T−1)−1` 并以 `ctx.traded()` 掩码（停牌日 ffill 会造出假跳空，见 `factors/intraday.py` 的模块 docstring §2）。
- **`bw_resid_beta_60`**：★ 本因子是**个股对市场宽度的敏感度**，不是市场水平量 —— 市场级数据在横断面上的离散度恒为 0，排序时与截距共线（见模块 docstring §一）。 ★ 构造：先对沪深300 回归取残差 `e = r − β_idx·r_idx`（β 同样用 60 日滚动、只用窗口内数据），再让残差对宽度变化做 β。**这是宽度暴露里真正与市场 β 正交的那一块** —— 如果 `bw_beta_60` 与 已注册的 `beta_60` 高度相关，本因子就是那个还剩下信息量的版本。经济含义：剔掉「随大盘涨跌」之后，还剩多少「随市场**扩散/收敛**」的暴露。依赖 `index_daily`（沪深300）—— 与 `factors/volatility.py` 同一口径。
- **`bw_session_follow_20`**：★ 本因子是**个股对市场宽度的敏感度**，不是市场水平量 —— 市场级数据在横断面上的离散度恒为 0，排序时与截距共线（见模块 docstring §一）。 **本文件新造**，它是模块 docstring §三 那条「真·48 点日内相关做不到」的**可得替代**：日内层不存个股的逐棒路径，个股侧**唯一**能拿到的日内分段就是上/下午两段（`am_close5/am_open5` 与 `close5/pm_open5`）。为了让两条腿**同步**，市场侧也切成同样的两段：上午 = `sh_am(11:30) − sh_open(09:30)`，下午 = `sh_close(15:00) − sh_am(11:30)`；个股侧两个比值是**同日**的（复权因子是常数，无需调整）。> 0 ⇒ 上午跟得比下午紧（隔夜信息主导的跟随）；< 0 ⇒ 下午才跟上市场。★★ 同向判定用 `np.sign(a) == np.sign(b)` 并**先做有限值掩码** —— `np.sign(NaN)` 是 NaN，`NaN != NaN` 为 True，写成 `!=` 会把缺失日静默计成「不同向」（这类 bug 已在 `mf_flow_stability_20d` 的 note 里记过一次）。
- **`bw_strength_sensitivity_20`**：★ 本因子是**个股对市场宽度的敏感度**，不是市场水平量 —— 市场级数据在横断面上的离散度恒为 0，排序时与截距共线（见模块 docstring §一）。 **本文件新造**（参考库 `up_down_count_ratio_20` 是最接近的，但那个是**市场水平量本身**，截面上零离散；本因子把它变成**个股对该量的相关**）。与 `bw_beta_60` / `bw_intraday_beta_20` 的回归量**不同**：那两个用**上涨家数份额**（连续、覆盖全市场、代表「扩散度」），本因子用**涨超 5% 的占比**（稀疏、右偏、代表「赚钱效应/强势股集中度」）——A 股里「大涨家数」是情绪周期的刻度，与「涨家数多不多」是两个不同的状态变量（可以普涨但没有强势股，也可以强势股很多但市场收平）。★ 与已注册的 `limit_up_count_20`（个股**自身**的 20 日涨停次数）完全不同：那个是自身行为，本因子是对**市场情绪**的反应。★★ 不用 `limit_up_count`：它在 2010–2019 恒等于 0（模块 docstring §二.5），`roll_corr` 会因为回归量零方差而在头十年给**全 NaN**。
- **`bw_tail_comove_60`**：★ 本因子是**个股对市场宽度的敏感度**，不是市场水平量 —— 市场级数据在横断面上的离散度恒为 0，排序时与截距共线（见模块 docstring §一）。 抄参考库 Class1 `tail_corr_60`（把市场收益换成宽度变化）。★ 用 `roll_mean(布尔, 60, 30)` 而不是自己数个数：框架的 `roll_count`**不毒化**，所以直接数会给出「0 次」这种看起来有效、其实是「窗口里全是缺失」的值。★★ 掩码写法：`|z_r|>1.5 & |z_m|>1.5 & sign(z_r)==sign(z_m)` **必须**先要求两侧 z 都有限，再算真假 —— 否则 `NaN > 1.5` 为 False 会被当成「不共振」计入分母，把缺失静默摊薄成低频。判定为 False 时写 0.0（确实不共振），判定为缺失时写 NaN。方向取负（参考库口径：尾部共振 = 脆弱）。
- **`bw_weak_breadth_ret_20`**：★ 本因子是**个股对市场宽度的敏感度**，不是市场水平量 —— 市场级数据在横断面上的离散度恒为 0，排序时与截距共线（见模块 docstring §一）。 **本文件新造**（与参考库 `tail_corr_60` 是不同构念：那个测**相关性**，本因子测**条件收益**）。★ 为什么减去自身均值：不同股票的 β 不同，直接比条件收益会被 β 主导；减去自身 20 日均值后剩下的是「**相对于自己平时**，在市场偏弱的日子里表现如何」——这是一个去均值的条件统计量，**不是** β 的重标定。★★ **条件变量为什么用「上涨家数份额 < 0.5」这个绝对阈值，而不是滚动分位** —— 这是沙箱 + `main.py check` 抓出来的一个**真实缺陷**：首版写的是「今日宽度变化 ≤ 其自身 20 日窗的 5% 分位」，看起来是标准的尾部条件化写法，但**在单边上涨行情里「今天的宽度变化处于自己近 20 日的最差 5%」可以连续几十天一次都不发生**⇒ 整条截面全 NaN ⇒ `main.py check` 报 **✘CONST**（实测 2019/2024 各有 69/103 个全 NaN 交易日，2026 也有 29 天）。这是**构念本身的固有缺陷**，不是实现 bug —— 换成任何自参照的滚动分位阈值都会有同样的病（趋势市里「今天相对过去很弱」永远不成立）。改用**绝对水平阈值**后：实测 2012–2026 全样本 `sh_close < 0.5` 占 53.6% 的交易日，而「连续 ≥0.5」的**最长**区间只有 **8 天** ⇒ 任意 20 日窗内至少 12 天命中，命中数恒 ≥ 1，**因子在所有交易日都有定义**。★ 与 `bw_capture_asym_20` 的区别：那个按宽度**变化**的符号分组（一阶差分），本因子按宽度**水平**是否过半分组 —— 「今天比昨天差」与「今天多数股票在跌」是两个不同的状态。
- **`cash_conversion_cycle`**：参考库未收录，用标准定义（天）。**方向低优**：周期越短越好，**可以为负**——负 CCC 意味着「先收钱后付钱」，占用上游资金经营（商超、白酒、预收款型公司），是强的质量信号，不要当异常值删掉。地板 100 万元设在**分母**（营业成本 / 营收 TTM）：存货为 0 的服务型公司 DIO 正确地等于 0（分子为 0 → 0），而金融股 oper_cost 已被 deriv 层置 NaN → 整个 CCC 为 NaN（银行本就没有营业周期）。**与三个周转率因子的关系**：CCC = 365×存货周转率⁻¹ + 365×应收周转率⁻¹ − 365×应付周转率⁻¹ —— 是它们的**倒数**组合，仍是独立信息（倒数放大低周转公司之间的差异），但高度相关，下游建模注意共线性。★ **实测尾部**：2012-2014 有 6.0% 的格子 > 1000 天、0.9% > 3650 天（最大 1.2e5 天）。逐条查过，不是数据错误，而是两类**真实**情形：(a) 地产/建筑（土地储备与应收工程款 vs 当期确认成本，周转天数天然上千）；(b) 停业/壳公司 —— TTM 营收刚过 100 万元地板（如 600275.SH 2012 年营收 110 万元而应收 3.6 亿元）。地板再往上提到 1 亿元能清掉 (b)，但会连带丢掉 3.5% 的营收（10% 的成本）—— 代价大于收益，故保持 100 万元地板，由引擎的 1%/99% 缩尾处理尾部（p99≈3471 天）。
- **`cash_sales_ratio`**：**本文件新造**。★ 与已注册的 `ocf_to_revenue`（经营现金流/收入）**不是**同一个量：`ocf` 已经扣掉了所有现金成本（采购、工资、税），而 `c_fr_sale_sg` 是**毛收款**（只扣增值税）——两个比值的中位数实测分别是 0.13 与 **1.0005**，差一个量级，刻画的是完全不同的环节：本因子测的是「**卖出去的钱收回来了没有**」（<1 = 应收堆积 / 渠道压货），`ocf_to_revenue` 测的是「最终剩了多少现金」。★ 为什么要做这个因子：中位数恒为 1.0005 ⇒ **水平本身无信息，信号全在离散度上**（左尾就是坏账风险）。这类「以 1 为基准的偏离度」因子在截面上是天然的双边分布，与所有利润率类因子都不同。
- **`cf_borrowing_repayment_ratio`**：本轮扩展公式，采用参考库 Quality/现金流比率的归一化方法；财务版本表按 max(ann_date,f_ann_date,end_date) 生效，保留修订版本；累计现金流先计算 TTM，资产负债科目不做 TTM。缺失不补 0，分母绝对值至少 100 万元；负现金支付按原值保留（存在供应商冲销记录）。慢变量需在跨年度样本评价。
- **`cf_distribution_cash_coverage`**：本轮扩展公式，采用参考库 Quality/现金流比率的归一化方法；财务版本表按 max(ann_date,f_ann_date,end_date) 生效，保留修订版本；累计现金流先计算 TTM，资产负债科目不做 TTM。缺失不补 0，分母绝对值至少 100 万元；负现金支付按原值保留（存在供应商冲销记录）。慢变量需在跨年度样本评价。
- **`cf_net_borrowing_to_assets`**：本轮扩展公式，采用参考库 Quality/现金流比率的归一化方法；财务版本表按 max(ann_date,f_ann_date,end_date) 生效，保留修订版本；累计现金流先计算 TTM，资产负债科目不做 TTM。缺失不补 0，分母绝对值至少 100 万元；负现金支付按原值保留（存在供应商冲销记录）。慢变量需在跨年度样本评价。
- **`cf_purchase_cash_intensity`**：本轮扩展公式，采用参考库 Quality/现金流比率的归一化方法；财务版本表按 max(ann_date,f_ann_date,end_date) 生效，保留修订版本；累计现金流先计算 TTM，资产负债科目不做 TTM。缺失不补 0，分母绝对值至少 100 万元；负现金支付按原值保留（存在供应商冲销记录）。慢变量需在跨年度样本评价。
- **`cf_tax_cash_burden`**：本轮扩展公式，采用参考库 Quality/现金流比率的归一化方法；财务版本表按 max(ann_date,f_ann_date,end_date) 生效，保留修订版本；累计现金流先计算 TTM，资产负债科目不做 TTM。缺失不补 0，分母绝对值至少 100 万元；负现金支付按原值保留（存在供应商冲销记录）。慢变量需在跨年度样本评价。
- **`cf_tax_refund_share`**：本轮扩展公式，采用参考库 Quality/现金流比率的归一化方法；财务版本表按 max(ann_date,f_ann_date,end_date) 生效，保留修订版本；累计现金流先计算 TTM，资产负债科目不做 TTM。缺失不补 0，分母绝对值至少 100 万元；负现金支付按原值保留（存在供应商冲销记录）。慢变量需在跨年度样本评价。
- **`cfcr`**：参考库 `因子库.md` 5、Quality #38 `cfcr`，公式逐字。★ 与已注册的 `interest_coverage`（`icr` = **EBIT**_TTM / 利息支出）的区别是**分子**：EBIT 是**权责发生制**的利润，经营现金流是**收付实现制**的现金。对有大量应计项的公司（应收激增、存货堆积），两者会**显著背离** ——EBIT 保障倍数只 1.2 倍但现金流保障倍数 4 倍的公司，偿债能力完全不同。★★ **必须 `start="2019-05-01"`**（与 `interest_coverage` 完全一致的起点）：实测 `fin_exp_int_exp` 在 2012/2015 的非零率是 **0.00%**、2019 才 87% ——早年是「未披露」而不是「零」，不写起点会让头七年产出恒 0（= 因子退化）。分母由 `safe_div` 的地板（1e6）保护；公司没有利息支出时给 NaN 而非 inf。
- **`cfp_ttm`**：现金流不受折旧/摊销/减值/非经常损益影响，比 ep_ttm 难操纵，是盈利质量的交叉验证项。经营现金流可以为负（扩张期垫资），保留负值不置 NaN —— 分母恒正是市值，不存在爆炸风险。
- **`chip_above_below_ratio`**：★ 出处：参考库 `factors/chip_extended.py`（Class 1，基于 cyq_perf）。定义辨析（重要）：这是**距离比**（现价距上方成本带 vs 距下方成本带），**不是**筹码质量比。实测 ρ(·, winner_rate) = +0.1150（几乎正交）——所谓「上方套牢>下方获利」的说法在这个公式里并不成立，它**不是**获利盘因子的重述。chips.py 当年因预算耗尽（清单第 29 位）没实现它，本批补上。cost_15/85pct → p25/p75。★ **符号的真实几何**（2026 全年 54.2 万格逐格验证，一致率 100.0000%）：值 > 0 ⟺ 现价落在 [p25, p75] 带内；带内时值随现价单调**递减**（在 p75 处 → 0，越靠近 p25 越大，p25 上发散）；值 < 0 ⟺ 现价已**突破 p75**（上方无近端筹码）或已**跌破 p25**（下方筹码全部套牢）。2026 实测：48.2% 的格为正、51.8% 为负。★ 正因为有符号翻转 + p25 处的极点，原始值**无界**（2026 实测 [−1.1e5, +2.2e5]），参考库取 `rank(−ratio)` 也压不住 —— 本因子只靠引擎的截面 winsor；下游若直接用原始值做线性合成，先看它被 winsor 后的分布。★ 参考库对 close 用复权价，本因子用未复权（口径 1）；方向标注按参考库 `rank(−ratio)`（值大=压力大排后）→ higher_is_better=False。
- **`chip_concentration`**：★ 直接取摘要表的 width 字段（fea/chips.py 已算好，本表实测 width ≡ (p90−p10)/p50，最大误差 0.0，不是近似）。cost_5/95pct → p10/p90，即覆盖 80% 筹码区间（参考库是 90%），数值略小但截面排序同构。参考库 rank 取负（区间窄=筹码密集=排前）。★ 任务清单里的 cost_distribution_width 与本因子同式（只差 clip(0,5)），未重复实现。
- **`chip_concentration_change_20d`**：参考库把 `concentration` 定义成**负**宽度再取 20 日变化，等价于「宽度收窄为正」；实现里直接对 width 取负差分（−ctx.diff(width,20)），语义相同。边际变化比绝对水平更有信息量（主力是否正在控筹）。2018 年开头约 20 个交易日 NaN。
- **`chip_cost_convergence_20d`**：★ 与 chip_concentration_change_20d 的区别在分位组合：本因子用 (cost_85−cost_5)/cost_50= (p75−p10)/p50（不对称：下侧取到 10%、上侧只到 75%），后者用 width=(p90−p10)/p50。两者都是「宽度 20 日变化」，秩相关偏高，但不对称口径对**上方**筹码堆积更敏感。参考库 rank 取负（收敛=正）。2018 年开头约 20 个交易日 NaN。
- **`chip_cv_factor`**：★ 取负向（低 CV 排前）。用 std/mean 而非 std 原值：std 保留价格量纲、被股价水平主导（10 元股与 100 元股不可比），CV 消除价格水平后跨股可比。实测 2019 年 std/mean 中位 0.146、p99 0.52。与 chip_concentration 中位 ρ = 0.70（都度量分散度，但一个用 80% 分位区间、一个用整体二阶矩）。
- **`chip_deep_trap_ratio`**：★ 出处：参考库 `factors/chip_deep_extra.py`。摘要表算不出（需要任意价格点的 CDF），本因子扫原始档位精确计算：Σpercent(price > 1.1×close)，分母为当日 Σpercent（与 below_close 同口径归一）。★ 为什么不能插值：摘要表只有 p10..p90 五个点，而 **1.1×close 在 47.4% 的格子上已经高过 p90**（0.9×close 更糟：70.8% 的格子跌破 p10；两个探针同时落在区间内的只有 6.4%，见模块 docstring 的探针几何）→ 插值／外推等于凭空捏造。chips.py 的 docstring 第 6 条也是用「摘要表算不出任意价格点的 CDF」为由排除了同类因子。★ 与 `above_close`（**全部**上方筹码占比，摘要表字段）的区别：本因子聚焦**深度**套牢区（>10%），实测 ρ(·, above_close 字段) = +0.7533、ρ(·, winner_rate) = −0.7533 —— 高度相关但不等价，深套区才是「解套抛压最重」的那部分。⚠️ 但它**不是**新维度：实测对首批 16 个的最大 |ρ| 达 0.9025（~chip_resistance_distance +0.9025、~avg_cost_premium −0.8978、~chip_position −0.8429、~chip_support_distance −0.7022）——「现价上方 10% 以外的筹码占比」本质上就是「现价相对成本有多低」的另一种说法，下游压因子数时应删掉本因子。★ 现价用 stock_daily 的未复权 close（与 fea/chips.py 的 close_map 同源），不用 ctx.px('close')（后者停牌前向填充，会造出摘要表里是 NaN 的格子）。
- **`chip_gini_factor`**：摘要表的 gini 由 fea/chips.py 按加权基尼标准式算：G = 2·Σ(i+1)·w_(i)/(n·Σw) − (n+1)/n（权重已归一化，Σw=1）。实测 2019：[0.22, 0.98]、中位数 0.866，恒非负 ✓（无负值、无 >1）。注：筹码按价格档**升序**排列后各档权重天然不等，故 gini 水平偏高，这是口径本身的性质（价格档密度在低价区更密），不是数据问题——截面排序仍可用。
- **`chip_high_float_ratio`**：★ 出处：参考库 `factors/chip_deep_extra.py`。与 chip_deep_trap_ratio 同一次扫描产出（见模块 docstring）。Σpercent(price < 0.9×close)/Σpercent。★ 它**不是** winner_rate 的重述：winner_rate 是 F(close)（阈值恰好是现价），本因子是 F(0.9×close)（阈值下移 10%），只统计「深度获利」的那部分。实测 ρ(·, winner_rate) = +0.5612、ρ(·, avg_cost_premium) = +0.6315、ρ(·, chip_deep_trap_ratio) = −0.4155 —— 与获利盘**总量**只算中度相关；对首批 16 个因子的最大 |ρ| = 0.6531（~chip_support_distance）→ 确实是新维度。⚠️ chips.py 当年用 5 点插值**近似**算过同类量，实测 ρ(·, winner_rate) = 0.907，那是它排除该因子的理由；本因子是**精确版**（原始档位逐档比较），共线度掉到 0.56 —— 说明当年那个 0.907 里有一大半是插值误差造成的假共线。★ 2026 实测值域 [0, 0.9925]，中位 0.0138（绝大多数格子的 0.9×close 都在 p10 之下，见模块 docstring 的探针几何）→ 它是个**右偏**的稀疏量，靠引擎 winsor。★ 两条阈值（0.9 / 1.1）不对称是参考库的原始口径，保留。
- **`chip_median_distance`**：★ 出处：参考库 `factors/chip_deep.py`。口径偏离：参考库用复权 close 与复权口径的 chip_median_price，本因子两边都用**未复权**（摘要表的 p50 就是未复权价，见模块 docstring 口径 1）。★ 与 avg_cost_premium 的关系：都是「现价 vs 成本中枢」的归一，但中枢不同（p50 vs 加权 mean）、分母不同（close vs mean）→ **没有任何代数关系**。实测逐日截面 ρ(·, avg_cost_premium) = +0.9150、ρ(·, chip_support_distance) = +0.9300（chips.py 2019 年实测 0.974 / 0.979，同一量级）→ **高度共线**。chips.py 曾以 0.974 为由排除它（预算耗尽），本批按用户清单实现；下游若压因子数，这个与 avg_cost_premium 二选一。
- **`chip_p90_p10_factor`**：★ 参考库的 chip_p90_p10 是**绝对价差**（未按中位价归一），本因子逐字照做。代价：它保留价格量纲，10 元股与 100 元股的数值不可直接比，截面排序里混有「股价水平」成分。之所以不换成归一化版本：归一化后的 (p90−p10)/p50 就是chip_concentration（width 字段），重复；参考库也确实是两个都收。若下游发现本因子与股价/市值高度共线，优先丢它、保留 chip_concentration。实测中位数 2018/2019/2020 = 2.11 / 1.62 / 1.56 元，p99 = 14.6 / 14.0 / 20.5 元，最大值 404 / 332 / 449 元（高价股的档宽天然大）—— 与 chip_concentration 中位 ρ 仅 0.47，因为后者已按 p50 归一。
- **`chip_peak_distance`**：★ 出处：参考库 `factors/chip_deep.py`。`chip_peak_price` = **percent 最大的那一档的价格**，对应摘要表的 `mode`（同一个定义，见 fea/chips.py 的 build_year）。★ 本因子是三个「距离」里**唯一**与 avg_cost_premium 共线度低的一个（实测 ρ = +0.6439；另两个是 +0.9150 / +1.0000）：众数价是分布的**局部**特征（对档宽与单档权重敏感），均值/中位数是**整体**特征。实测 ρ(·, mode 字段) = −0.0684（该字段本身没被任何因子用过，这里只作为定义核对）。★ 口径：未复权 close（同 chip_median_distance）。
- **`chip_peak_growing`**：★ 出处：参考库 `factors/chip_deep.py`。`chip_peak_dominance` = 最大单档权重占比，对应摘要表的 `peak_purity` 字段（参考库另有 `chip_peak_purity` = 该量的**水平**；本地首批 16 个因子没有注册它，`peak_purity` / `skew` / `kurt` 这几个字段都是本批第一次用）。★ 主峰纯度的水平受**股价水平**影响（档宽固定 0.1/0.01 元时，股价越高单档占比越小），做 5 日**差分**正好把这个个体固定效应消掉：实测 ρ(·, peak_purity 字段) = +0.2082（2026 逐日截面中位）→ 与水平近乎正交，是比水平更干净的一版。★ 契约的 NaN 策略：窗口内出现 NaN 即 NaN（停牌日筹码行缺失 → 该日 NaN）。
- **`chip_position`**：★ 口径：close 用未复权（见 avg_cost_premium 的说明）；cost_5/95pct → p10/p90。★ 定义辨析（重要）：这是**价格空间**的相对位置，不是**筹码质量空间**的百分位。真正的「现价在筹码分布中的分位数」= F(close) = below_close = winner_rate 因子本身（已精确算好）；本因子是它的价格空间线性近似，二者秩相关高但不相等（价格空间对分布形态敏感，质量空间不敏感）。因此本因子取值可以越出 [0,1]（现价高于 p90 时 >1、低于 p10 时 <0），这是参考库的原式，不做截断——截断会把「突破筹码密集区」这一最有信息量的状态压平。参考库 rank 取负（位置高=接近上方套牢区=阻力大）。
- **`chip_range_normalized`**：与 chip_concentration 的分位组合不同（这里 25/75，那里 10/90）：一个度量核心 50% 筹码、一个度量 80% 筹码的宽度。cost_15/85pct → p25/p75。参考库 rank 取负（区间窄排前）。
- **`chip_tail_risk`**：★ 出处：参考库 `factors/chip_deep.py`（Class 2，读 cyq_chips 原始档位）。摘要表的 `kurt` 字段 = 加权四阶中心矩 − 3（超额峰度），与参考库的 `chip_kurtosis` 逐字同口径（都是 Σw·(p−μ)^4/σ^4 − 3）——**本批之前没有任何因子用过这个字段**。2026 实测：中位 13.44、p99 169.05、min −1.9529、max 418.34，恒 > −3（chips.py 2019 年实测中位数 25、p99 164，量级一致 ✓）。★ 注意与 chips.py 的 `chip_cost_kurtosis_20d` 区分：那个是分位点代理(p75−p25)/(p90−p10)，是**有界**的比值；本因子是四阶矩，量纲为 1 但值域极宽（2026 实测 [−1.95, 418]）→ 引擎的 1%/99% 截面 winsor 后仍有少数极端值，这是**原口径**（参考库也没截断）。实测两者 ρ = −0.0970（几乎正交，不是同一个量）。★ 参考库的 `chip_kurtosis` 在 std≈0 时取 −3，本表同情形给 NaN（fea/chips.py 在 sd>0 时才写 kurt）—— 2026 实测摘要表 `n_levels` 最小 6、中位 101，单档筹码的格子 **0 个**，无影响。
- **`chip_win_peak_frac`**：★ 出处：参考库 `factors/chip_deep_extra.py`。参考库定义：max(percents[prices ≤ close]) / Σpercents[prices ≤ close]。★ 与 `peak_purity`（**全分布**最大单档权重，摘要表字段）的区别：本因子把分母限制在**现价下方**（只统计获利盘），因此「下方集中」与「全分布集中」是两件事 —— 实测 ρ(·, peak_purity 字段) = +0.6942（中度相关，不是重述）、ρ(·, below_close 字段) = −0.5300（获利盘越多、下方峰反而越分散：获利筹码被摊到更宽的价位上）、ρ(·, chip_deep_trap_ratio) = +0.4252；对首批 16 个因子的最大 |ρ| = 0.6847（~chip_cr3_factor，同为「筹码向少数价位集中」的度量，但本因子的分母只含现价下方）→ 新维度。2026 实测值域 [0.0117, 1]、中位 0.3138。★ 语义：获利筹码集中成峰=主力成本密集单一（吸筹完成/锁筹），分散=浮筹多、涨时兑现压力大。是「吸筹 vs 出货」的形态维度。
- **`close_location_20d`**：参考库出处：factors.md `类别 price` / fac_new_daily.py 的 close_location_20d。★ 复权：参考库这条**没有**走 `_adjusted_close`（用的是原始 high/low/close），  在除权日会被污染；本文件按项目硬约束改成 hfq 口径 ——   `(hfq_close − shift(hfq_close,20)) / (max(hfq_high,20) − min(hfq_low,20))`。★ 值域：分子 ≤ 分母（20 日振幅包含这 20 日的全部价格），所以理论上 ∈ [−1, 1]；  实测在 [−1,1] 内（分子分母同口径时严格成立）。★ 分母保护：20 日完全无振幅时 safe_div 给 NaN。
- **`consecutive_limit_down`**：★ 与 `consecutive_limit_up`（event2.py）**镜像不重复**：那个用上游 `stock_limit_up.consecutive_days`（供应商算好的连板高度），跌停侧**没有对应的表**（`stock_limit_list` 只有 `limit_times`，实测在 D 行上 25% 分位=1、中位=1、max=29，与自算连跌天数同量级，但只有 2020 起），故本因子在面板上自攒连续段：`_run_len()` 与参考库的 `groupby + cumsum` 分段逐格等价（已用 200 组随机 0/1 序列验证），但全向量化、不写 Python 循环。★ 事件口径与 `limit_down_event_5` **完全共用**（LL 优先 + 价格近似回退）。★ 停牌日 = 没成交 = 不可能跌停 → 记 0，**连续段被打断**（与参考库 `pct_chg.le(-9.8)` 在 NaN 上取 False 的行为一致）。warmup 给 W60（128 日历天 ≈ 60 个交易日）：连续段的起点依赖窗口前段的历史，实测历史上最长连续跌停 29 天（LL `limit_times` 的 max），60 天足够覆盖。★ 零膨胀（固有）：**实测零值占比 99.41%**（非零仅 0.59%），2026 年取到的最大值 5（连续 5 个跌停）。这是「状态量」的必然结果 —— 与 `consecutive_limit_up`（连板数）同款：非连板日为 0 是**定义**，不是缺失。
- **`consecutive_limit_up`**：★ 与 `limit_up_count_20`（event.py，20 日累计次数）**互补不重复**：本因子是「当下处于第几板」的状态，断板当天归零。口径用上游 `stock_limit_up.consecutive_days`（供应商已算好的连板高度，实测取值 1~30、无 0）而不是自己用 pct_chg 攒连续段——参考库的 cumsum 分段在面板上要逐列循环，且除权/停牌边界更脆。缺失补 0 = 当天没涨停 = 连板数 0。分布：约 70% 的涨停样本是首板，长尾到 30 板（大量 0/1 + 长尾）。
- **`cost_convergence_signal`**：★ 出处：参考库 `factors/chip_cost_extended.py`。值是**收敛量**（= 参考库的 −chg，正 = 宽度收窄），与 chips.py 的 `chip_concentration_change_20d`（= −diff(width)）同一约定 —— 参考库 rank(−chg) 「分布收窄=筹码集中排前」的方向由 higher_is_better=True 表达，数值不取反两次。★ 与 `chip_concentration_change_20d` 的差别是**差分口径**：本因子是**绝对价差** (p90−p10) 的 20 日**变化率**（pct_change，无量纲、自动按股价水平归一），那个是**归一化宽度** width=(p90−p10)/p50 的 20 日**差分**（带 p50 量纲）。两者都度量「宽度在收窄」，实测逐日截面 ρ = +0.9241 —— **高度共线**，下游若要压因子数，这两个里留一个即可（本因子对低价股的绝对价差变动更敏感、对股价水平不敏感，那个相反）。★ 参考库的 `clip(-1, 1)` 未保留（winsor 由引擎统一做）：去掉后值域是 「宽度放大 >100% 时为负」——2026 实测 value ∈ [−11.65, +0.937]、中位 0.0037、5.24% 的格子 < −1（那 5% 恰好是参考库 clip 掉的部分，截面排序不受影响，因为 rank 只关心顺序）。★ 2018 年开头约 20 个交易日为 NaN（20 日窗口要读到 2017 年，上游无数据）。
- **`cost_distribution_skew`**：★ 参考库 `factors/chip_deep.py` 的 cost_distribution_skew = (cost_50−cost_15)/(cost_85−cost_50)，1:1 照搬（分位映射 15/50/85 → p25/p50/p75）。⚠️ 本地**没有** `chip_cost_skew` / `chip_cost_asymmetry` 这两个因子（首批 16 个都没注册）→ **不是重复列**，对本地是新信息。★ 与摘要表的 `skew` 字段**不是一回事**：`skew` 是加权三阶标准矩（2026 实测：中位 1.2765、p1 −4.63、p99 11.98、min −8.85、max 16.40），本因子只用两个分位差（对离群价稳健）—— 实测两者 ρ = −0.1528，确实不是同一量。★ 值域无界：p75−p50 → 0 时发散（2026 实测 [0, 76.79]）→ 截面 winsor 由引擎做。
- **`cost_skew_ratio`**：★ 参考库 `factors/chip_cost_extended.py` 公式 = (cost_50−cost_5)/(cost_95−cost_50)，1:1 照搬（分位映射 5/50/95 → p10/p50/p90）。⚠️ 本地**没有** `chip_cost_asymmetry` / `chip_cost_skew`（首批 16 个未注册）→ 不是重复列。★ 与 cost_distribution_skew 的区别在分位点：本因子用 5/50/95（覆盖 90% 区间），那个用 15/50/85（覆盖 70% 区间），实测两者 ρ = +0.8125（高度相关但不等价）。★ 参考库的 `clip(0.1, 10)` 未保留（winsor 由引擎统一做）；2026 实测值域 [0, 64.67]（分母 p90−p50 → 0 时发散）。
- **`cp_bigflow_margin_20`**：两个「聪明钱」维度：大单成交价（成本端）与杠杆资金加速（资金端）。预期非空率 ≈ 50%（受 margin 覆盖限制）。
- **`cp_chip_support_reversal_5`**：筹码位置高 = 收盘价高于大部分持仓成本（获利盘多）；叠加超跌 → 「获利盘被套」的反转机会。★ 起点受上游 `stock_cyq_chips` 限制（2018-01-02 起）。预期非空率 ≈ 90%（筹码层覆盖主板，与股票池一致）。
- **`cp_chip_turnover`**：筹码集中 + 换手剧烈 = 典型的「变盘前夜」。预期非空率 ≈ 90%。
- **`cp_margin_trend_div`**：分歧型：杠杆资金在加速、但价格趋势还没走出来（或反之）。★ 父因子 `margin_velocity` 是**滞后表**（stock_margin_detail 晚 1 个交易日）做出来的，它自己已经 `lag_grid(1)` 过了，所以本因子**不需要**再位移 —— 父因子的滞后处置会被继承，重复位移等于白丢一天信息。预期非空率 ≈ 50%（只有两融标的）。
- **`cp_momentum_highvol_60`**：与 `cp_momentum_lowvol_20` 是**对照组**（同样两个维度、方向相反）。两个都留着是有意的：A 股在不同区间对「高波动动量」的定价方向会翻，下游回归可以各自取值。预期非空率 ≈ 100%。
- **`cp_momentum_lowvol_20`**：★ 低波动本身没有方向（低波动异象的方向是「低波动跑赢」），乘以动量后表达的是「强势但波动小」的股票。父因子 `momentum_20` 与 `idio_vol_60` 实测在 2026 单年|RankIC| 分别 0.0424（负向）与 0.0752（负向），两者都强且相关性不高 → 乘积有增量。预期非空率 ≈ 两父因子交集（两者都是价格类，接近 100%）。
- **`cp_moneyflow_momentum_res`**：`mf_order_size_entropy` 实测是 fundflow 族里最单调的因子（分层单调性 −0.99）：熵高 = 订单规模分散 = 散户主导。乘以动量后表达「散户主导的上涨」。预期非空率 ≈ 100%（两张表都是全市场日频）。
- **`cp_quality_momentum`**：质量动量（QMJ 的 A 股版本）。慢变量（roe_ttm 季度才变） × 快变量（动量）→ 整体换手由动量部分决定。预期非空率 ≈ 100%。
- **`cp_rsi_moneyflow_res`**：技术面超买 × 资金面流入同时高 = 情绪与资金共振，A 股里通常是反向信号。⚠️ `mf_net_inflow_ratio` 的 ac1 ≈ 0.04（日频全换手），本因子换手也很高，必须按**成本后**收益复核。预期非空率 ≈ 100%。
- **`cp_value_momentum_div`**：★ 用**差**而不是积：表达的是「便宜但近期没涨」（错配），与「又便宜又强」（共振）是两种不同的信号。参考库里这种 `rank(A) − rank(B)` 型有几十个，本文件只保留 3 个分歧型（价值-动量、两融-趋势、筹码-换手各一）。
- **`cp_value_quality`**：经典的「便宜且好」（格雷厄姆式）。★ 两个父因子的 ac1 都 ≈0.999（季度才变一次），所以本因子**也是慢变量**：2026 单年只有约 4 个独立样本，IC 不可信；它的价值在于作为**风格暴露**进模型，而不是独立 alpha。
- **`cvar_95_120`**：公式行逐字抄自 Class1 risk / cvar_95_120。★ 偏离参考库**实现**：参考用 5 个分位数取平均做黎曼近似（它自己的注释说明这是为了向量化），本实现按任务书口径取「窗口内 ≤5% 分位那部分的**均值**」= 精确历史 CVaR（参考库「意义」里写的也是「取最坏 5% 日收益的均值」）。120 日窗口下尾部期望 6 个样本，两种算法差异很小。min_count=60 = 参考 min_periods。实现：roll_quantile 取 q05（NaN 忽略）→ 掩码 r<=q05 → 窗口内尾部求和 / 尾部天数（都是 cumsum 类，无逐日循环）。
- **`cyqp_cost_premium_change_20`**：来源：学习资料/factors.md 筹码收益系列；返回原始值，排名由引擎执行。与旧筹码档位推算因子分开命名，不能假定两者完全相等。精确按日落格，缺失不前填；winner_rate 按百分数除以 100，超出 [0,100] 置 NaN；非正成本和逆序分位点置 NaN。收盘后数据供下一交易日使用；滚动窗口要求全窗有效。
- **`cyqp_cost_tail_asymmetry`**：来源：学习资料/factors.md 筹码收益系列；返回原始值，排名由引擎执行。与旧筹码档位推算因子分开命名，不能假定两者完全相等。精确按日落格，缺失不前填；winner_rate 按百分数除以 100，超出 [0,100] 置 NaN；非正成本和逆序分位点置 NaN。收盘后数据供下一交易日使用；滚动窗口要求全窗有效。
- **`cyqp_cost_width_70`**：来源：学习资料/factors.md 筹码收益系列；返回原始值，排名由引擎执行。与旧筹码档位推算因子分开命名，不能假定两者完全相等。精确按日落格，缺失不前填；winner_rate 按百分数除以 100，超出 [0,100] 置 NaN；非正成本和逆序分位点置 NaN。收盘后数据供下一交易日使用；滚动窗口要求全窗有效。
- **`cyqp_cost_width_90`**：来源：学习资料/factors.md 筹码收益系列；返回原始值，排名由引擎执行。与旧筹码档位推算因子分开命名，不能假定两者完全相等。精确按日落格，缺失不前填；winner_rate 按百分数除以 100，超出 [0,100] 置 NaN；非正成本和逆序分位点置 NaN。收盘后数据供下一交易日使用；滚动窗口要求全窗有效。
- **`cyqp_cost_width_change_20`**：来源：学习资料/factors.md 筹码收益系列；返回原始值，排名由引擎执行。与旧筹码档位推算因子分开命名，不能假定两者完全相等。精确按日落格，缺失不前填；winner_rate 按百分数除以 100，超出 [0,100] 置 NaN；非正成本和逆序分位点置 NaN。收盘后数据供下一交易日使用；滚动窗口要求全窗有效。
- **`cyqp_historical_range_position`**：来源：学习资料/factors.md 筹码收益系列；返回原始值，排名由引擎执行。与旧筹码档位推算因子分开命名，不能假定两者完全相等。精确按日落格，缺失不前填；winner_rate 按百分数除以 100，超出 [0,100] 置 NaN；非正成本和逆序分位点置 NaN。收盘后数据供下一交易日使用；滚动窗口要求全窗有效。
- **`cyqp_mean_median_gap`**：来源：学习资料/factors.md 筹码收益系列；返回原始值，排名由引擎执行。与旧筹码档位推算因子分开命名，不能假定两者完全相等。精确按日落格，缺失不前填；winner_rate 按百分数除以 100，超出 [0,100] 置 NaN；非正成本和逆序分位点置 NaN。收盘后数据供下一交易日使用；滚动窗口要求全窗有效。
- **`cyqp_tail_width_share`**：来源：学习资料/factors.md 筹码收益系列；返回原始值，排名由引擎执行。与旧筹码档位推算因子分开命名，不能假定两者完全相等。精确按日落格，缺失不前填；winner_rate 按百分数除以 100，超出 [0,100] 置 NaN；非正成本和逆序分位点置 NaN。收盘后数据供下一交易日使用；滚动窗口要求全窗有效。
- **`cyqp_winner_acceleration_5`**：来源：学习资料/factors.md 筹码收益系列；返回原始值，排名由引擎执行。与旧筹码档位推算因子分开命名，不能假定两者完全相等。精确按日落格，缺失不前填；winner_rate 按百分数除以 100，超出 [0,100] 置 NaN；非正成本和逆序分位点置 NaN。收盘后数据供下一交易日使用；滚动窗口要求全窗有效。
- **`cyqp_winner_change_20`**：来源：学习资料/factors.md 筹码收益系列；返回原始值，排名由引擎执行。与旧筹码档位推算因子分开命名，不能假定两者完全相等。精确按日落格，缺失不前填；winner_rate 按百分数除以 100，超出 [0,100] 置 NaN；非正成本和逆序分位点置 NaN。收盘后数据供下一交易日使用；滚动窗口要求全窗有效。
- **`cyqp_winner_fraction`**：来源：学习资料/factors.md 筹码收益系列；返回原始值，排名由引擎执行。与旧筹码档位推算因子分开命名，不能假定两者完全相等。精确按日落格，缺失不前填；winner_rate 按百分数除以 100，超出 [0,100] 置 NaN；非正成本和逆序分位点置 NaN。收盘后数据供下一交易日使用；滚动窗口要求全窗有效。
- **`cyqp_winner_volatility_20`**：来源：学习资料/factors.md 筹码收益系列；返回原始值，排名由引擎执行。与旧筹码档位推算因子分开命名，不能假定两者完全相等。精确按日落格，缺失不前填；winner_rate 按百分数除以 100，超出 [0,100] 置 NaN；非正成本和逆序分位点置 NaN。收盘后数据供下一交易日使用；滚动窗口要求全窗有效。
- **`di_plus_minus_ratio_14`**：参考库出处：factors.md `类别 price` / trend_pattern.py 的 di_plus_minus_ratio_14。★ 偏离（分母保护）：参考库写 `di_minus + 1e-8`，但 DI 的量纲是百分数，分母踩到 1e-8 时比值会飙到 1e10 —— 直接触发引擎「值域异常」（|value|>1e8）。本文件用 `safe_div(..., min_abs_den=1e-4)`：DI- < 0.0001%（几个月没有一个有效下移）时给 NaN，其余情况比值上界 1e6，安全。★ 复权：同 adx_14，昨收走 `shift(hfq_close,1)`。
- **`dividend_yield_3y_avg`**：★★ **数据源偏离**：上游**没有分红明细表**（`ActualCashDiviRMB` 无从取得），唯一的股息来源是供应商日频快照 `stock_finance.dv_ttm`（滚动 12 个月股息率，百分数；实测 2024 年非空率 100%、其中 31.2% 是精确的 0 = 不分红，真实值）。**重建口径**（逐字对齐参考库，而不是简单地对 dv_ttm 求均值）：
    每股分红_TTM(t) = dv_ttm(t)/100 × ClosePrice(t)      ← 由定义反推
    dividend_yield_3y_avg(T) = mean_{t∈735}(每股分红_TTM(t)) / ClosePrice(T)
即「3 年的平均每股分红 ÷ 今天的股价」，与参考库公式一致。★ 为什么不直接 `mean(dv_ttm, 735)`（更省事、但**不等价**）：那个量是「3 年里各时点股息率的均值」，分母是**当时的股价**；参考库的分母是**今天的股价**。在上涨行情里二者会系统性分叉（朴素版高估、且随行情漂移）。本实现用同一个分母，是真正的「用今天价格衡量的历史分红水平」。分子里 dv_ttm(t)×P(t) 的 P(t) 用**未复权价**（分红/送转导致的除权跳空正是每股分红的来源，用后复权价会把它抹掉）。★ 窗口 735 = 参考库的 245×3（交易日），严格窗口：**上市不满 3 年**的公司为 NaN，这是「3 年平均」的固有要求。★ 停牌日 close 是前向填充的状态量，所以停牌不产生缺口、也不产生假值。★ **窗口均值用本文件的 `_roll_mean_causal` 而不是 `ctx.roll_mean`**：`audit-pit` 的截断复算在本因子上抓到 1.644e-04 的相对差（> 1e-4 容差），根因是 `ctx.roll_mean` 内部的数值中心化用了**全样本列均值**（`fea/mathx.py` 的 `_col_mean`），会让 T 的值随面板右端之后的行数变化。换成 float64 前缀和差分后，T 的值只由 `<= T` 的数据决定（见该函数 docstring）。★ `fin_fields` 说明：本因子**不读任何财报表字段**，声明 `total_share` 只是为了让引擎走「按字段子集建表」的快路径 —— 引擎约定空元组 = **全字段全建**（见 `fea/engine.py` 的 `frozenset(s.fin_fields) or None`），那会让同一次 run 退化成 80 字段版本表、且每个任务的字段子集都触发重建。
- **`donchian_position_20`**：参考库出处：factors.md `类别 price` / technical_pattern.py 的 donchian_position_20。★ 口径**逐字照搬**了参考库的**不对称窗口**：上轨排除了当日（`shift(1)` 后再取 20 日 max，窗口 = [T−20, T−1] 的 high），下轨含当日（窗口 = [T−19, T] 的 low）。这不是笔误 —— 突破的定义就是「今天的价格超过了此前的高点」，所以上轨必须排除今天。★ `clip(0,1)` 同样是参考库的口径（通道位置的定义域边界），不是 winsor；  代价是「突破」的信息被截断在 1.0，要突破幅度请用 donchian_breakout_20。★ 值域 [0,1]，无量纲（分子分母同为价格单位）。★ 复权：high/low/close 全部走 hfq，参考库的 `scale` 折算在本框架里  由价格层统一完成，不需要再乘一次。
- **`downside_upside_vol_60`**：逐字抄自 Class1 risk / downside_upside_vol_60。min_count=5 是**必须**的放松：正/负收益各自稀疏（60 日里约各 28 天），参考库自己也写「正/负样本各自稀疏:60日窗口内 min_periods=5 即可估计」。★ 框架语义：min_count 下分母仍是满窗 60，而分子分母同缩放，比值本身仍然成立。
- **`downside_vol_ratio_20`**：逐字抄自 Class1 risk / downside_vol_ratio_20。口径注意：参考的 clip(upper=0) 把**上涨日**置 0（不是 NaN），所以分母是「窗口内所有交易日」而不是「下跌日个数」——即标准半方差比，不是「下跌日条件波动」。min_count=15 = min_periods；停牌日 ret 是 NaN，会随 r² 一起毒化窗口。
- **`dp_ttm`**：★ 单位是**百分数**（如 2.4 表示 2.4%），沿用供应商口径，不做换算。★ dv_ttm 有 32%~40% 的精确 0 值（实测 2012 年 40.0% / 2013 年 33.8% /2014 年 31.5%）—— 那是「不分红」的**真实值，不是缺失**，一律保留，绝不置 NaN（置 NaN 会让「分红稳定性」一类因子把不分红公司整体排除，而它们恰恰是最该被识别的一组）。上游没有分红明细表，dv_ttm 是唯一来源，故走 ctx.dataset + asof 前向填充。
- **`dpo_20`**：参考库出处：factors.md `类别 price` / technical_daily.py 的 dpo_20。★ shift(11) 是「20 日周期的一半 + 1」（去趋势的经典参数），不是 20；  `lagged` 是 11 个**交易日**前的复权收盘价。★ 已归一化（÷ ma20），无量纲。warmup 要覆盖 20 日窗 + 11 日位移。★ 参考库取正向排名（DPO > 0 = 11 日前的价格高于当前均线 = 近期在回落），本因子同样标 `higher_is_better=True`，方向语义以参考库为准。
- **`dragon_tiger_org_net_20`**：★ **不再做滞后位移**（2026-09-15 晚，v2→v3）：`stock_dragon_tiger` 实测**当天可得** ——日更工程 T=2026-09-15 21:14 的逐日观测 `delay_obs = 0`。去掉了 v2 的 `lag_grid(grid, 1)`，T 日直接用 ≤ T 日的席位明细（整条序列相对 v2 前移一个交易日）。与 `top_list_net_rate_20` 同一处理，前提与更正史见它的 note。⚠️ 本因子还叠加了一次**输入侧变更**：2026-09-15 晚该表主键由 4 列改 7 列并全量回填（旧主键把同名机构席位合并，09-14 一天 557→780 行），所以 v3 与 v2 的差异同时来自「去掉位移」和「机构席位不再被合并」。★ `stock_dragon_tiger.net_buy_amount` 的**符号不可信**：与 `buy_amount − sell_amount` ★ `stock_dragon_tiger.net_buy_amount` 的**符号不可信**：与 `buy_amount − sell_amount` 在 32.2% 的行上符号不一致（178 万行全表实测，例如某行 buy=778万/sell=0 却给 net=−778万）。本因子一律自己算净额，不碰该列。★ 分母用**个股 20 日成交额**（价格层，元）而不是机构席位自身的买卖总额：后者会让「只有一笔小单的机构席位」拿到 ±1 的极端值，丢掉了资金量级信息。跨表前先确认量纲：机构席位买卖额与 `stock_daily.amount` 同为元（实测 机构 gross/当日成交额 中位数 4.4%），无需换算。★ 覆盖率的两个来源：① 20 日内没有任何「机构专用」席位 → NaN（无定义）；② 停牌造成 20 日成交额窗口不完整 → `min_count=10` 显式放松（对齐契约 §3.4，理由就是停牌），放松后仍有值的股票占绝大多数。非空率 ≈ 10%（机构席位本就只出现在少数上榜股票里）。值域约 [−1, 1]。★ 精度：分子分母都升到 float64 再滚动（元级大额 + 引擎的 float32 cumsum，见 `top_list_net_rate_20` 的说明）。
- **`drawdown_duration_120`**：逐字抄自 Class1 risk / drawdown_duration_120。★ 偏离任务书的一句话描述：任务书写「px < 历史最高」，本实现按参考库用 **120 日滚动前高**（roll_max(px,120)）——全历史最高需要全历史面板，增量重算无法给出「与全量一致」的 warmup。0.999 的容忍带、上限 120 都照抄参考库；min_count=1 = 参考 min_periods=1（上市首日即有前高）。连续段用 maximum.accumulate 向量化，没有逐日 Python 循环。
- **`dv_stability_4q`**：★ 命名来自参考库（dv_stability_4q），参考库自己的 2026-08-05 修正注明：它的**代码**实现是 60 个交易日（约 1 个季度）的滚动 CV，而它的**描述/名字**说的是 4 个季度 —— 参考库自己承认这两者矛盾。★ 偏离（关键）：本实现取 **250 个交易日（≈4 个季度，与名字一致）**。实测依据（2012–2014 沙箱）：60 日窗口下日均截面 1,488 只（低于契约 §7 的1500 红线）—— 该窗口只覆盖「近 3 个月内除权过」的股票；换成 250 日后日均截面中位 1,619（2012 年 1,474 / 2013 年 1,614 / 2014 年 1,646），接近截面满覆盖的 dp_ttm（2,111）的 77%，且符合名字与描述的原意（分红稳定性本来就是年度尺度的属性）。代价：与参考库的 60 日数值不同（不是同一个数，但方向一致：低 CV = 稳定）。★ 用 dv_ttm（滚动股息率）而不是 dv_ratio：dv_ttm 是 TTM 口径，随股价每日变动，CV 才有意义；dv_ratio 是年度口径、季度才跳一次。★ 不分红的公司 dv_ttm 恒为 0 → 标准差与均值同时为 0 → CV **无定义**（NaN），实测这类占 23% 的格子（含上市不足 60 个交易日的新股）——即 250 日内从未分过红的公司被判为无定义。这是有意为之：「从来没分过红」不是「分红很稳定」，两者不能混为一谈。★ min_count=60：窗口的 1/4。dv_ttm 是阶梯状态量（只在除权日跳），有效值个数实际等价于「上市满 60 个交易日」，与参考库 min_periods=20对 60 日窗口（1/3）的放松幅度相当。★ 偏离：不加参考库的 clip(0, 20)（引擎统一 winsorize）；不用参考库的 dv_ratio（那是年度口径）。
- **`ebitda_to_market`**：参考库 §9 Value #6 的公式逐字，但 **EBITDA 用白名单字段近似**（上游 `stock_financial_indicator` 有 `ebitda` 水平值，但它是**累计 YTD**、且不在 `ctx.ind()` 的安全字段表里 —— 见汇报的「需要引擎扩字段」）：
    EBITDA = ebit + depr_fa_coga_dpba + amort_intang_assets + lt_amort_deferred_exp
★ **实测精度**（2024FY，4831 只可比）：与供应商 `ebitda` 的**中位相对差 −4.0%**，|差| < 5% 占 32.8%、< 20% 占 69.0% —— 自算值系统性**偏低**，因为白名单的折旧字段（固定资产折旧、油气资产折耗、生产性生物资产折旧）不含**使用权资产折旧 /投资性房地产折旧**等科目（2019 年新租赁准则后对零售、航空、餐饮影响较大）。所以本因子是「EBITDA 的下界近似」，量级与排序可用，绝对值不要与供应商口径混用。★ 四个分项在原始表里实测**非空率 100%**（2012/2016/2020/2024 年报均如此），「非零率」62%~99%（长期待摊摊销最常为 0，那是真实值）—— 故直接相加，**不做 nan→0**（缺失只可能来自报告期缺失，此时整格应为 NaN）。★ 与 `valuation.cfp_ttm` 的分工：EBITDA 加回了折旧摊销（非现金），比经营现金流更贴近「经营性盈利能力」，且不受营运资本变动的影响。
- **`efx_annual_earnings_yield`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_dividend_gap`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_dragon_net_intensity`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_dragon_ratio_balance`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_forecast_revision_delay`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_free_turnover`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_limit_first`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_limit_float_fraction`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_limit_openings`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_limit_reseal_span`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_limit_signed_move`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_limit_streak`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_limit_trade_share`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_limit_turnover`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_limit_win_fraction`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_lu_board_density`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_lu_final`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_lu_first`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_lu_move`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_lu_one_price`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_lu_seal_amount`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_lu_seal_volume`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_seal_float`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_top_amount_rate`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_top_float`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_top_imbalance`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_top_move`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_top_turnover`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`efx_volume_ratio`**：本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。
- **`elg_net_60d_to_mv`**：★ 单位：分子 `* 1e4`（万元→元），参考库写得很清楚，本实现照抄。★ 市值口径偏离：参考库用 `fin['circ_mv']`（流通市值），本实现用 **总市值** = `ctx.px('close') × ctx.px('total_share')`（项目约定，与本族 `mf_net_amount_intensity` 一致）。close 是**未复权**价（PIT 安全）。`min_count=20` = 参考库 `min_periods=20`。无北向持股数据时，超大单是外资/产业资本的最佳代理。
- **`eom_14`**：参考库 Class1 `eom_14`（technical_daily.py）逐字如上，`min_periods=7` = 14//2，与本实现的 min_count=7（N//2）**一致**。参考库返回 `cross_sectional_rank(eom_avg)`，本因子返回原始值。★ 参考库注释明确写了「**除** box_ratio 而不是乘」（「Multiplication would reward high-volume/narrow-range days and is the inverse of the named indicator」），本实现照抄除法。★ 中点位移走**后复权**口径（参考库用 `scale = adj/close` 折算 high/low，数学上就是复权价），避免除权日伪位移。★ 振幅 0（一字板）→ `box_ratio` 分母 0 → `eom` NaN（参考库 `.replace(0, np.nan)` 同义）。★ 掩码 `traded(T) & traded(T-1)`：`distance` 是跨日量；停牌日 `vol` 本来就是 NaN，所以这层掩码只额外挡住「复牌日」（mid 从停牌前的陈旧值跳到复牌价）。★ 量纲是 **元²/股**（数量级 1e-9~1e-5），不要与参考库的绝对水平对齐；截面 rank 不受常数因子影响。值域无界但极小。
- **`ep_ttm`**：★ 负 PE 的口径选择：**保留**。这里用 PIT 对齐的归母净利润TTM 做分子，亏损股的 ep 是真正的负数（不是 0 也不是 NaN），排名上自然落到最「贵」的一端，语义连续、不丢样本。对照组 pe_ttm_absolute 走的是「置 NaN」口径，两者互补。分母是市值（恒正），但仍走 safe_div + min_abs_den=1e6（元）。
- **`etp5`**：参考库 §9 Value #3。「5 年滚动均值」在本框架的实现（两边都逐字）：① **分子** = 5 个年度观测的均值 `mean(lag_ttm(NP, 0/4/8/12/16))`，即最近 5 个报告期的 TTM（每个都是滚动年度口径，含季节性已差分）；**5 期全有效**才算（NaN 传播，不做 min_count 放松）。② **分母** = **1260 个交易日**（5 年 × 252）的总市值滚动均值，严格窗口；用本文件的 `_roll_mean_causal`（而非 `ctx.roll_mean`）—— 后者的内部数值中心化用了全样本列均值，会让 T 的值随面板右端之后的行数变化（见该函数 docstring）。③ 两个约束合起来 ⇒ 本因子只覆盖「上市满 5 年 + 有 5 年连续财报」的公司，次新股系统性缺失（这是「5 年平均」的固有要求，不是 bug）。④ **量纲**：读数是「每 1 元市值对应多少元年均利润」，典型 0.02~0.10，与 `valuation.ep_ttm`（当期口径）互补：这个是 5 年平滑版，几乎不含单年噪声，但**对近两年的盈利变化反应极慢**（1/5 权重）。⑤ 分母用**市值的历史均值**而不是当期市值（参考库口径）：分子分母都是 5 年平均，比值是「长期盈利 / 长期估值」，不是当期估值。⑥ 起点限制见文件头「已知限制」。
- **`extreme_move_event`**：★ 阈值口径（**主 Agent 2026-09-15 拍板：7.0%**，取参考库原值）：先按任务书的 9.5% 实现过，实测它与 `limit_up_event_5` 的逐日截面 Spearman **ρ = 0.883** —— 被引擎 `dedup` 直接判成重复簇。根因是 9.5% 与主板 ±10% 的涨跌幅上限只差 0.5pp，而涨停事件比跌停多约 3 倍 → 秩几乎被涨停侧定住。降到参考库原值 7.0% 后 ρ = 0.708（8.0% 是 0.797），保住了「大涨/大跌但没到板」（7~9.5% 那一段）的独立信息。**v2 起生效。**★ 时间结构与参考库的偏离：半衰期 10 → **3**、窗口取 5 个交易日（任务书的 `_5` 系列范式，Σ_{k<5} 0.5^(k/3)）。★ 用 `ctx.px("pct_chg")`（供应商日收益）而不是 `ctx.ret(1)`：与同文件涨跌停阈值同源同尺度（都是交易所口径的当日涨跌幅，除权日已调整），且 `ctx.ret(1)` 会对 |r|>60% 做清洗（对涨跌停判定无关但会引入口径差）。停牌日 pct_chg 为 NaN → 直接记 0（停牌不是异动事件）。向上取 `higher_is_better=False` 只影响文档方向：值大 = 近期异动剧烈。★ 实测零值占比 89.31%、|value|max = 3.3205、非空率 100.00%。
- **`forecast_profit_midpoint_change`**：扩展公式；预告金额与去年归母净利润同为供应商万元单位，比值约去单位；按 ann_date 生效，同一股票同日多条取中位数，不按报告期全表取最后版本；两端必须完整且有序，分母绝对值至少 100 万元。超过 365 日的披露置 NaN；财报分区按报告年读取额外 3 年，保留迟到公告。用于次日交易。
- **`forecast_profit_range_uncertainty`**：扩展公式；预告金额与去年归母净利润同为供应商万元单位，比值约去单位；按 ann_date 生效，同一股票同日多条取中位数，不按报告期全表取最后版本；两端必须完整且有序，分母绝对值至少 100 万元。超过 365 日的披露置 NaN；财报分区按报告年读取额外 3 年，保留迟到公告。用于次日交易。
- **`forecast_type_score`**：打分依据《事件驱动策略之一》的结论排序（扭亏最好、预减最差、预增居中偏上；续盈/略增/略减样本少且特征不明显 → 弱档）。打分是**序数**不是基数，下游做截面回归时会自动再标准化。未知类型（上游新增枚举）→ NaN，不默认给 0，避免把「没见过的类型」混进中性档。同 (股票, 公告日) 多条取中位数。覆盖：与 forecast_p_change_median 同源，但**不依赖 p_change**，所以少数只有类型没有幅度的预告也能给值。
- **`free_share_ratio`**：直接取股本比（参考库 float_mv_ratio 的定义），等价于自由流通市值/总市值。方向：参考库 rank(−ratio) —— 自由流通占比**低**代表筹码锁定度高、实际可交易盘小、波动弹性大，故 higher_is_better=False。实测 free_share 有约 0.04% 的 0 值（上游未跟踪），此时 ratio=0 ——这是「已知自由流通股本为 0」还是「未知」无法区分，保留为 0 并在下游按极端值处理。
- **`fundflow_retail_inst_divergence`**：★ **耦合因子**（参考库 `coupling.py` 的口径）：`deps` 写的是两个**父因子名**，引擎据此把它排在第二趟（等父因子落盘后再算，见 `main.py` 的两趟调度）。★ **父因子的替换（与参考库的字面差异，必须看）**：参考库读的是 `mf_small_order_ratio`（散户**净额**占比），该因子已被主 Agent 从 `factors/fundflow.py` 删除（其 `_DROPPED` 记录：与既有因子精确重复），本实现改用**现存**的散户侧代表因子 `mf_retail_dominance`（= 小单毛额占比，`higher_is_better=False`，即值大=散户主导）。两者是同一枚硬币的两面：`mf_retail_dominance` 度量「散户**参与度**」（谁在交易，实测与已被删的 `mf_small_order_ratio` 相关仅 −0.27），取负排名后语义与参考库的 `rank(-small)` 一致 —— 「散户不占主导」。机构侧父因子 `mf_big_order_ratio` 未变。★ 这里**必须**用到截面排名（`_cs_rank_pct`）—— 排名是公式的组成部分（乘积 = 「同时高」的联合信号），不是对本因子输出的后处理；本因子的输出仍会被引擎再排名一次。参考库用的 `_rank` 同样是截面百分位排名，口径一致。★ 与直接相减的区别（参考库的 `mf_smart_dumb_divergence` 是「大单率 − 小单率」，该因子同样已被删除）：乘积要求**两端同时**极端（机构排前 20% 且散户排后 20% 才进 top 4%），相减只要求一端领先 —— 前者是「共振」，后者是「差值」；本族保留的 `mf_big_small_divergence` 是相减的另一种写法（见其 note）。★ 与 `big_vs_small_divergence_5d` 的区别：那条是**变化率**（Δ5），本因子是**当日水平**的联合排名。★ 沙箱 2026 实测（vs 全部 259 个既有生产因子的逐日截面 rank 相关）：max|corr| = **+0.681**（父因子 `mf_big_order_ratio` 本身）、`mf_retail_dominance` −0.594、`mf_order_size_entropy` +0.554 —— 对一个「两个父因子排名的乘积」来说这是**结构性下限**（乘积必然与两个因子都相关），0.68 已经远低于任一父因子自身的水平，剩下的部分就是「共振」这个新信息。★ 缺失处置：任一父因子为 NaN（停牌/无记录）→ 乘积为 NaN，不参与截面。★ 若父因子尚未落盘，`ctx.load_factor` 会**静默**返回全 NaN（它只 warning），表现为本因子非空率 0% —— 首次全量跑时确认两个父因子已在同一批次里。
- **`fv_gain_share`**：**本文件新造**。公允价值变动是**未实现、会反转、不产生现金流**的收益 ——占比高说明当期利润里有一块「纸面富贵」，是典型的盈利质量扣分项，而全库此前没有任何因子刻画它。⚠ **结构性零膨胀（实测，务必先知）**：`fv_value_chg_gain` 在原表上就有**59.6% 的行恰好为 0**（另 15.0% 为负、25.4% 为正），TTM 汇总后本因子 2012 年的零值占比 **72.4%** —— **超过 `fea/eval.py` 的 `⚠SPARSE` 阈值（>70%），一定会被标 ⚠SPARSE**。沙箱自检已实测确认。**保留的理由**：它是一个语义完全成立的量（「利润里有多少是纸面重估」），零膨胀来自 A 股绝大多数公司不持有大额以公允价值计量的资产这一**真实事实**，而不是口径错误；同库已有 12 个因子带 ⚠SPARSE 仍在册。⚠ 若 eval 给出的 RankIC 也同时低于 0.005（→ 同时带 ⚠NOISE），则应删掉，不必保留。方向取负。
- **`gain_loss_asymmetry_60`**：逐字抄自 Class1 risk / gain_loss_asymmetry_60。min_count=10 = 参考库 min_periods（涨/跌样本各自稀疏）。与 downside_upside_vol_60 互补：这个是**一阶矩**比，那个是二阶矩比。
- **`gap_down_recover_freq_20d`**：参考库 Class1 `gap_down_recover_freq_20d`（fac_new_daily.py）逐字如上，`roll` 默认 min_periods=1 → 本实现 min_count=10（N//2）。★ 与 `gap_up_fade_freq_20d` 是**镜像**关系但**不是** 1 − x：「隔夜>0 且 日内<0」与「隔夜<0 且 日内>0」互不覆盖（隔夜/日内同号的日子两边都不计），所以两个都必须独立保留。★ 方向：值高 = 恐慌被消化、买盘韧性好（参考库 `意义` 一栏），`higher_is_better=True`。★ 停牌一律 NaN，不补 0。值域 [0,1]。
- **`gap_event_decay_5`**：★ 与参考库的四处偏离（逐条给理由）：① **事件值用跳空幅度而不是 0/1**（任务书「跳空幅度 × 半衰期权重」）：同样是「五天前跳空」，跳 9% 与跳 3% 应该不同；0/1 口径把这个信息丢了。② 阈值 5% → **3%**：实测 5% 在 2026 年主板只有 1.38% 的 stock-day 命中，5 日窗口内至少一次的比例 4.2% → **零值占比 95.8%，越过「>95% 说明口径太稀」的红线**；3% 命中率 3.45%、零值占比 89.9%。而且 3% 与同族 `big_gap_reversal_5`（event2.py，高开 >3%）的门槛一致，家族内部口径统一。③ **显式挡停牌日**：`hfq(open)` 与 `hfq(pre_close)` 在停牌日各自被前向填充，比值 = **停牌前那天的跳空**（不是 0、也不是 NaN）——不挡的话停牌期间会每天重复触发同一个跳空事件（模块 docstring 三.1）。④ 半衰期 5 → 3、窗口 5 个交易日（任务书 `_5` 系列范式）。★ 跳空的除权日安全性：`hfq(open)/hfq(pre_close)` 的分子分母同用一个 `adj_factor(t)`，复权在该比值上是**恒等变换**（模块 docstring 三.4）——真正保证除权日不产生假跳空的是 `pre_close` 本身是**除权后基准价**（实测 2026 年 93.5 万行 `close/pre_close − 1` 与 `pct_chg` 偏差 < 0.005pp）。仍统一走 `ctx.hfq`（契约硬约束 4）。★ 取绝对值（不保留方向）：任务书写的是「跳空幅度」，方向信息由同族的 `big_gap_reversal_5`（高开后走势）承担。★ **实测**：零值占比 86.97%、|value|max = 4.5838（≈ 连续 5 天 9% 跳空的加权和）、非空率 100.00%。截面秩相关：与 `extreme_move_event` 0.464、与 `limit_up_event_5` 0.388、与 `new_low_60_event` −0.094 —— **不在任何重复簇里**（引擎 dedup 未报）。
- **`gap_fill_tendency_10d`**：参考库 Class1 `gap_fill_tendency_10d`（price_deep.py）逐字如上，`min_periods=5` 照抄。★ **回补 = 当日收盘回到前收之上/之下**（完全回补），不是「盘中触及缺口边缘」，也不是「之后 N 日内回补」—— 后者按字面实现要读 `T+1..T+N`，是**未来数据泄露**（面板右端就是 T），本文件**不**做。窗口只向过去看（模块 docstring 的专门一节）。★ 分母 `+0.01` 照抄参考库（避免除 0）；没有跳空日的股票 → 分母 0.01 → 值 0.0（不是 NaN），这一点与 `gap_open_follow_ratio_20` 不同，是参考库的原样行为。★ 跳空阈值 **1%** 照抄参考库（比 `gap_open_follow_ratio_20` 的 0.5% 严）。★ 掩码 `traded(T) & traded(T-1)`。值域 [0,1]（分子 ≤ 分母）。
- **`gap_up_fade_freq_20d`**：参考库 Class1 `gap_up_fade_freq_20d`（fac_new_daily.py）逐字如上，`roll` 默认 min_periods=1 → 本实现 min_count=10（N//2）。★ 方向：值高 = 冲高抛压重、上方套牢盘多（参考库 `意义` 一栏），故 `higher_is_better=False`。★ 判据**不用**阈值（隔夜 > 0 就算高开，哪怕只高开 0.001%）—— 参考库原样。★ 停牌 / 前一日停牌一律 NaN，且**不补 0**：补 0 会把停牌多的股票推成「从不低走」（第 4 条）。值域 [0,1]。
- **`growth_stability`**：★★ 实现要点：8 期同比**全部由 lag_ttm 在报告期粒度上构造**（lag 0..11：第 i 期同比 = lag_ttm(REV,i)/lag_ttm(REV,i+4)−1），再逐格做 mean/std。**没有**在日频上做 rolling —— 日频 rolling 会把同一个报告期的值重复计入200 多次，而且窗口边界横跨报告期，得到的既不是「8 期」也不是任何可解释的量。★ 起点显式写 2014-01-01（不是 default_start）：本因子要 lag_ttm(REV, 11)，而上游财报最早只到 2010Q1、TTM 最早算到第 83 期（2010FY）→ lag 11 最早在当期报告期 >= 94（2013Q3）时可达，实测首个有效日 2013-10-11、2014-03 起覆盖才稳定。2012 整年会是全 NaN，而引擎目前在整年全 NaN 时会崩（见文件头 DEEP_LAG_START 的说明）。★ 性质：这不是「纯稳定性」，而是成长的**信息比**（均值/标准差）——它同时奖励高增速与低波动；稳定下滑的公司（均值 −0.2、标准差 0.05）得 −4，排在最末。若要纯波动率口径请取 `-std`（本因子不含）。★ 标的选营业收入而非净利润：净利润同比在亏损时符号会翻转，均值与标准差都会被「正负跳变」主导，得到一个与经营无关的巨值。★ 分母 std 有 min_abs_den=1e-3 的地板：8 期增速高度一致时（真实存在的，如公用事业/高速公路）std 近零会让比值爆掉。★ 传播规则：8 期里任何一期为 NaN（早期数据不足），结果即 NaN（与 `roll_sum` 的 NaN 策略一致，不做 min_count 放松）。
- **`hammer_ratio_20d`**：参考库 Class1 `hammer_ratio_20d`（trend_pattern.py）逐字如上，`min_periods=10` 与本实现的 min_count=10（N//2）**完全一致**（这一族里唯一一个原生就对齐的）。参考库返回 `cross_sectional_rank(ratio)`，本因子返回原始占比（rank 由引擎做）。★ 一字板（`high == low`）时 `upper < 0.3*rng` 是 0 < 0 型无定义，参考库靠 `.replace(0, nan)` 判成 False；本实现用 `rng > 0` 显式判 `ok` → 给 **NaN**（判据不可得 ≠ 形态不成立，第 4 条）。实测一字板占 0.20%~1.81%（按年，模块 docstring 第 3 条）。★ 掩码只用 `traded(T)`：判据全是**当日** OHLC，不跨日。值域 [0,1]。
- **`high_open_low_close_frac_20`**：高开与收阴都是**当日**比较（open 与 pre_close 复权后同尺度、close 与 open 同日），所以口径上不会踩除权日的坑；仍统一走 `ctx.hfq`（本项目唯一允许的价格口径）。`min_count=5` 对齐参考库 min_periods=5（停牌日 pct_chg/gap 为 NaN，不参与均值）。与 `big_gap_reversal_5` 的区别：后者看「高开 3% 之后还涨不涨」（事件+收益），本因子看「高开 2% 却收阴」的**频率**（当日日内反转的出货指纹）。值域 [0,1]，绝大多数股票接近 0（高开低走是少数形态）。
- **`holder_number_chg`**：上游披露期不规律（1~12 月都有），所以对「与上一期间的间隔」做保护，间隔超过 6 个月的样本置 NaN，避免拿两个不同年份的观察值相比。★ 2026-09-17 修 PIT 违规：旧实现对**全表** `drop_duplicates(keep='last')`，而同一个 `end_date` 会在一周内被不同公司陆续披露 —— 新公告一到就挤掉旧公告并把自己的 `ann_date` 盖上去，于是**历史生效日被改写**（实测一次重建改写 9 天）。现在改成 **as-of 状态机**：逐条公告推进「最新两期」的读数，每条公告只看得到它自己那天（含）之前的信息 → 历史值不再随新公告变化。
- **`id2_am_close_position`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 抄参考库 Class3 `am_hl_position`（上午收盘在上午高低区间的位置）。★ 与已被删的 `idt_close_position` **构念相同、对象不同**：那个是**全天**收盘在全天区间的位置，本因子是**上午 11:30** 在**上午区间**的位置。保留它的理由是本文件 docstring §一.2 ——`am_close5`（11:30）是日内层**唯一新增的两个价格点之一**，日线层与已注册因子都拿不到它。若 `dedup` 实测与全天版 |ρ| ≥ 0.95 则砍掉。
- **`id2_am_pm_range_ratio`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 源自参考库 Class3 `am_pm_hl_range_ratio`（方向负）。★ 与已被删的 `idt_am_pm_rv_ratio` 是**同一个构念、不同的估计量**：那个用两段各自的 `ret2_sum`（真实已实现波动，需要逐 session 的二阶量，本项目层里**没有**分段的 ret2_sum），本因子退一步用 Parkinson 式「极差」做段内波动代理。参考库只此一个版本，而「上下午波动比」是 A 股日内择时文献里的标准量，故保留。★ 必须区间扩张，否则极差可能 ≤0 使 ln 出 NaN。
- **`id2_am_pm_ret_gap`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 ★ **故意偏离参考库**：参考库 Class3 是 `am_pm_return_ratio`（比值），但它因分母 `pm_ret → 0` 时会爆出 ±1e3 量级的假值，已于 2026-09-17 以 `idt_am_pm_return_ratio` 之名被删（去糟粕）。本因子用**差**：有界、不会被小分母放大，而且在截面上「am−pm」与「am/pm」**不是**单调同序关系（比值对小分母敏感、差值不敏感），所以它不是一个改头换面的同义因子。经济含义：>0 ⇒ 上午强、下午回吐（隔夜信息驱动）；<0 ⇒ 下午拉升（日内资金持续流入）。
- **`id2_am_pm_vwap_gap`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 抄参考库 Class3 `vwap_am_pm_gap`。★ 会话 VWAP 是层里**新出现**的对象（`am_amt/am_vol`、`pm_amt/pm_vol`），已注册的 `vwap_daily_deviation` 只覆盖**全天** VWAP 与收盘的偏离，不区分会话。同日比值、同单位约掉，未复权无妨。与 `id2_am_pm_ret_gap` 的区别：那个比较两个会话的**端点价差**（首尾两点），本因子比较两个会话的**成交均价**（整段重心）。均价位移 > 端点位移 ⇒ 该段成交集中在高位（放量拉抬）。
- **`id2_am_ret`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 抄参考库 Class3 `am_momentum`。★ 与已注册的三个同类**都不重复**：`overnight_intraday_ratio_20d` 是隔夜腿 vs 全天日内腿的占比；`intraday_ret_momentum` 是**全天** close/open−1；`intraday_ma_{5,20,60}d` 是全天日内收益的多日滚动均值。本因子是**上午这半天单独**的收益，与下午段正交（A 股的上午段承载了大部分信息发布后的即时反应）。同日比值，价格未复权也无妨（同一天内复权因子是常数）。★ 09:35 是 5min 首根棒的结束时刻，不是开盘价 —— 这不是偏离，是 5min 口径的固有粒度（参考库 1min 版从 09:31 起）。
- **`id2_am_vol_share`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 抄参考库 Class3 `am_vol_share`。★ 单位自动约掉：日内层已逐 (股票,日) 判过 `vol` 是「手」还是「股」并 ×100 归一（`fea/intraday.py` 的 `vol_mult`），`am_vol` 与 `vol` 用的是**同一个** `mult`，所以比值在任何年份都成立 —— 跨 2025-11/12 的单位翻转边界**没有** 100 倍假跳变。取值域 [0,1]。★ 不做 `pm_vol/vol`：那是本因子的**精确补集**（1 − x），截面排名会完全反向，纯属重复。高 ⇒ 交易集中在上午（信息驱动）；低 ⇒ 尾盘/下午放量（资金驱动）。
- **`id2_amihud_intraday_20`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 源自参考库 Class3 `amihud_5min`。★ **与已注册的两个 Amihud 因子的唯一区别是「哪条腿」**：`amihud_daily_5` 用 close-to-close（含隔夜跳空），`amihud_asymmetry_20` 是涨跌日的不对称，本因子用**日内腿**（`ret_sum` = 48 根棒收益之和，剔除了隔夜跳空）。日内腿的 Amihud 衡量的是「**盘中**每元成交额推动的价格变化」。⚠ **本文件置信度最低的一行**：`amihud_daily_20` 曾在 2026-09-17 的去糟粕清单里（虽未注册），与已注册的两个 Amihud 家族的相关性未实测。**`dedup` 阶段与 `amihud_daily_5` / `amihud_asymmetry_20` 逐一比 |ρ|，超过 0.95 就地删除。**分子是单位无关的收益、分母是元 —— 同日截面内一致，可跨日比。
- **`id2_close_vs_pm_vwap`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 源自参考库 Class3 `vwap_dev`（方向负）。★ 与已注册的 `vwap_daily_deviation` **不是**同一个量：那个是**全天 VWAP** 与收盘的偏离，本因子把 VWAP 收窄到**下午段**（13:05–15:00），从而把「上午的成交重心」从基准里剔掉 —— 剩下的是纯粹的**尾盘定价**：A 股 14:57–15:00 是收盘集合竞价，收盘 > 下午均价 ⇒ 尾盘有主动买盘承接。A 股实施收盘集合竞价制度后（2018-08 起深市、2018-08-20 起沪市），这个量的信息含量前后**不同质** —— 下游若按年切分训练需注意。方向取负（参考库口径：偏离越大排后）。
- **`id2_close_vs_pm_vwap_20`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 `id2_close_vs_pm_vwap` 的 20 日滚动均值。**为什么值得单独留**：单日的尾盘偏离有相当一部分是买卖盘随机冲击，但「这只股票**长期**收盘都高于下午均价」是一个持续性的资金行为特征（机构按收盘价建仓/指数化资金），在截面上与单日值的相关性不高（单日噪声占主导）。★ 2018-08 收盘集合竞价制度变更前，这个量的均值结构不同 —— 跨年比较时留意（与单日版同一条 note）。
- **`id2_dd_ru_asym`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 **本文件新造**。层里 `max_dd ≤ 0`（= min(seg/run_max − 1)，回撤是负的）而 `max_ru ≥ 0`（反弹是正的），所以取绝对值后相加是「这天所有极端行程的总量」，比值就是**向下的占比**。= 0.5 ⇒ 上下对称；> 0.5 ⇒ 向下行程占主导（日内抛压）。与已注册的 `gain_loss_asymmetry_60` 的区别：那个是 60 日**逐日收益**的正负半方差比（跨日、低频），本因子是**单日之内**的路径极值比（同日、路径形状）—— 两者衡量的时间尺度完全不同。分母趋 0（全天几乎不动）时由 `safe_div` 给 NaN，语义正确。
- **`id2_hi_lo_pos_gap`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 **本文件新造**。`hi_pos`/`lo_pos` 层里已经归一化到 [0,1]（= 段内 argmin/argmax ÷ (n_bars−1)），**不要再除 n_bars**。`|hi_pos − lo_pos|` 大 ⇒ 极值分居上下午，日内走了单边；小 ⇒ 高低点挨在一起，V 形/倒 V 反转。与 `idt_lo_pos` 不重复：那个只看**最低点在哪**（位置水平），本因子看**两个极值的时间距离**（位置差的绝对值，与整体早晚无关）。取绝对值后再给方向：铺得开 = 日内趋势清晰。
- **`id2_intraday_max_runup`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 抄参考库 Class3 `intraday_max_runup`。★ **层里早就算好了 `max_ru`，但此前 0 个因子用它** —— 已注册的 `idt_intraday_max_drawdown` 只做了回撤那一侧（层里 `_path_stats` 同时产出 `max_dd` 与 `max_ru`，参考库也是两个都发，本项目只建了回撤侧）。注意口径：这是**日内路径**上的最大反弹，不是「从昨日低点反弹」，与 `price_distance_from_52w_low` / `rebound_from_low_20` 无关。≥ 0。
- **`id2_lunch_gap`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 抄参考库 Class3 `lunch_break_ret`。★ 这是**日线层完全看不见**的量：日线只有 昨收/开/高/低/收，而 11:30→13:05 这条断口把「上午收盘」和「下午开盘」连接起来，期间是午间公告、午间舆情、以及 12:00 前后发布的宏观数据的反应窗口。同日比值、未复权，量级通常极小（中位数 < 0.2%），但截面上的**符号与大小**承载了午间信息的即时定价。
- **`id2_parkinson_vol`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 逐字抄参考库 Class3 `parkinson_vol`（high/low 用本日值，不需复权因子）。★ 必做区间扩张：上游 high/low 不总包住 close/open（实测 0.075% 越界），不扩张会让 ln(hi/lo) 变负、开方出 NaN 或假值。⚠ 与已注册的 `amihud_parkinson_ratio` 有关联但**不是**同一个量：那个是 20 日 Parkinson 均值与 Amihud 之比，本因子是单日水平。方向取负（波动越低越好），与全库波动类口径一致。
- **`id2_pm_ret`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 抄参考库 Class3 `pm_momentum`。下午段 = 13:05 开盘到 15:00 收盘，含尾盘集合竞价（A 股 14:57-15:00）。与上午段**不是**互补关系（两者之间还夹着午间跳空，见 `id2_lunch_gap`），所以 `am + pm + 午间` 才等于全天。同日比值，无需复权。
- **`id2_ret_concentration`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 **本文件新造**（参考库没有；它靠逐根棒的数据直接算更高阶矩，本项目只有聚合量）。数学性质：由柯西-施瓦茨，`1 ≤ n·Σr²/(Σ|r|)² ≤ n`。取 1 ⇒ **单根棒扛下了整条路径**（其余 47 根不动）= 跳跃/瞬时重定价；取 n ⇒ 每根棒幅度相同 = 均匀推进。**这是在没有逐棒数据的前提下，唯一还能拿到的「路径形状」高阶量**（名字取自逆参与比 / inverse participation ratio）。方向取负：集中度高 = 单点驱动，与「平滑趋势」相比更可能是噪声。
- **`id2_rv_parkinson_gap`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 源自参考库 Class5 增强 `microstructure_efficiency`（`rv_5min / parkinson_vol`，方向取负）。原式是比值，本因子取 ln —— 比值在截面上的分布极度右偏（分母可以极小），ln 后可比性更好，且与排名的单调关系不变。**经济含义**：Parkinson 只用当日极差，看不到路径；RV 用逐棒收益平方和。两者相等 ⇒ 价格走的是**单调趋势**（每根棒都在同一方向积累）；RV 远大于 Parkinson ⇒ 路径**来回震荡**；RV 远小于 Parkinson ⇒ **一根大棒 + 其余不动**（跳空/集合竞价式重定价）。参考库的方向是负（「效率高」排后），此处沿用。
- **`id2_rv_parkinson_gap_20`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 `id2_rv_parkinson_gap` 的 20 日滚动均值。★ 为什么必须平均：单日的 `rv/parkinson` 分母可以极小（一字板日极差趋 0），单日值的截面分布有长尾；20 日均值把它压成「这只股票**一贯**是趋势型还是震荡型」的稳定描述。min_count=10 = N//2，见模块 docstring §一.5；`_roll()` 另外要求当日输入有限。与单日版**不是**同一维度的重标定（一个是当日状态、一个是风格），两者并存由 `dedup` 定量裁决。
- **`id2_session_range_overlap`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 **本文件新造**（参考库没有跨会话区间结构的任何因子）。数学性质：上午与下午是两个不相交的时间段，`max(am_hi,pm_hi) − min(am_lo,pm_lo) ≥ max(段内振幅)`，而全天振幅由区间扩张后的 hi/lo 给出，故比值 ∈ [1,2]。= 1 ⇒ **下午从未走出上午的区间**（盘整、方向未变）；= 2 ⇒ 两个会话探索了**互不相交**的价格区间（重定价、方向翻转）。与 `id2_am_pm_range_ratio` 的关键区别：那个是两段**振幅之比**（只看大小），本因子是**区间是否错开**（看位置），两个会话可以振幅相同但完全错开（比值=1，本因子=2）。分母全天振幅趋 0（一字板）时由 `safe_div` 给 NaN —— 正确语义：无信息。
- **`id2_session_sign_agreement_20`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 **本文件新造**。上午与下午的**符号**是否一致 —— 高 ⇒ 日内方向持续（趋势型股票），低 ⇒ 日内反转频繁（震荡型）。★ 用 `==` 比较**有限值掩码后**的符号，不用 `np.sign(a) != np.sign(b)`：后者在任一侧为 NaN 时 `np.sign(NaN) = NaN`，`NaN != NaN` 为 **True**，会把缺失日静默计成「方向不一致」（这类 bug 已在 `mf_flow_stability_20d` 的 note 里记录过一次）。★★ **截面取值卡片化警告**：20 日均值只有 21 个可能取值（0, 1/20, …, 1），且大多数股票会挤在两端。`fea/eval.py` 的退化判据是「单值占比 > 75%」，本因子可能触发 ——沙箱自检阶段必看每日截面唯一值数，不足就地砍。
- **`id2_vol_amt_hhi_gap`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 **本文件新造**。`n·Σx²/(Σx)²` 是逆参与比形式的集中度（与 `id2_ret_concentration` 同一族，但用在成交量/额上）∈ [1, n]。两项之差 > 0 ⇒ 成交**额**比成交**量**更集中 ⇒ 放量的那几根棒价格**高于**当日均价（大单成交在高位）；< 0 ⇒ 放量集中在低价区（承接/吸筹）。★ 用到的 `amt2_sum` **此前 0 个因子消费过**。★ 单位：`vol2_sum` 层里已按 `mult²` 修正（`fea/intraday.py` 的注释），所以本比值跨 2025-11/12 单位翻转边界**量纲一致**，无需在因子内补偿。⚠ 与 `idt_vol_stability` 的关系：那个是**单个** HHI 的水平，本因子是**两个不同物理量**的 HHI 之差，不是它的重标定。
- **`id2_vol_peak_pos`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 抄参考库 Class3 `volume_peak_time`（方向负：放量越晚排越后）。★ 层里 `vol_peak_pos = argmax(vol) / (n_bars − 1)` **已经是 [0,1] 的归一化时点**，不要再除 `n_bars`。与已注册的 `idt_vol_stability` 不重复：那个是 `Σv²/(Σv)²`（成交量的**离散度**），本因子是**峰值在哪**（位置的**一阶**统计量）——同样的离散度可以对应早盘放量或尾盘放量。值小 ⇒ 开盘/早盘放量（隔夜信息消化）；值大 ⇒ 尾盘放量（资金驱动、可能的收盘操纵）。
- **`id2_vol_peak_std_20`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 **本文件新造**。`id2_vol_peak_pos` 看「今天几点放量」，本因子看「**每天几点放量这件事稳不稳**」。稳定的日内流动性节奏（标准差小）= 有固定的做市/被动资金，峰位乱跳 = 交易由事件驱动、不可预期。与 `vol_of_vol_20` / `vol_clustering_20` 的区别：那两个算的是**收益率波动**的波动，本因子算的是**成交量时点**的波动 —— 一个是价格维、一个是时间维。
- **`id2_zero_bar_share`**：★ 5min 口径（参考库 Class3/4 是 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：日内层已把 48 根聚合成日频字段。 **本文件新造**（已注册的 `zero_return_fraction_20` 是**日频**的 |pct_chg| < 0.1% 占比，20 日窗口；本因子是**单日之内**在 5min 粒度上的零收益棒占比，灵敏度高一个数量级）。★★ `−1` 是**强制**的：层里每个 (股票,日) 段的**第一根棒** `prev_c = NaN → r = NaN → r0 = 0`，于是 `n_zero` 恒含一个假零。不减去它，全市场每只股票每天都凭空多一根零收益棒（48 根里多 1 根 = 2.1% 的系统性高估），而且**不报错**。分母同时减 1（第一根本来就没有收益可比）。经济含义：一字板 / 极度不活跃 / 长时间无成交。
- **`idio_vol_60`**：逐字抄自 Class1 risk / idio_vol_60；基准同 beta_60（沪深300）。min_count=30 = 参考 min_periods。残差 = ret − beta×mkt，beta 用同一 60 日窗口的滚动估计（与参考的 _rolling_beta 一致，不做全样本回归，故是严格 PIT 的）。★★ v2（2026-09-17）：随 `mathx.roll_cov` 的分母 bug 修复整体重算（残差里含 beta，beta 错则残差也错）。
- **`idt_intraday_max_drawdown`**：★ 5min 口径（参考库 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：参考库 Class3 `intraday_max_drawdown` 的**定义**是「日内从最高点到后续最低点的最大跌幅」，公式却写 `rank(-dd.abs())`——`max_dd` 本身 ≤0，所以 `-|dd| ≡ dd`，**参考库的排序等价于「回撤浅的排前」**，本因子方向取 True（= 参考库的实际方向）。★ 取值 ∈[−1,0]（价格恒正），实测中位数 ≈ −0.02。★ 日内层用**收盘价序列**的 running max（不是 bar 的 high）——与参考库「从最高点到**后续**最低点」的口径一致（顺序敏感，不含「先跌后涨」式的回撤）。★ 5min 棒数从 240 降到 48，会**轻微低估**回撤深度（棒内极值被抹平），方向为对所有股票的同向压缩。
- **`idt_lo_pos`**：★ 5min 口径（参考库 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：★★ **本因子在参考库里没有同名条目**：Class3 只有 `intraday_high_time`（最高价时点），没有 `intraday_low_time`。这里按镜像构造（同一字段族、同一归一化口径），方向取负 = 「最低点越晚出现越差」（尾盘才见低 = 尾盘抛压）。**方向是本次推断的，未经 V8 独立筛查，下游首次使用前应用样本外 IC 复核。**★ 与 `idt_hi_pos` 同为 [0,1] 的归一化时点，直接透传字段、不再除 n_bars。★ 与 hi_pos 的分析口径一致：分辨率 48 档，大量并列。
- **`idt_overnight_gap`**：参考库 Class1 `overnight_gap`（alternative.py）逐字为 `-(open-pre_close)/pre_close` 截面排名 —— 高开=反转信号排后，故本因子（有符号的跳空）方向取负。★ 口径：`pre_close` 换成 `hfq_open(T)/hfq_close(T-1)-1`，两者数学等价（交易所前收 = 昨收 × 除权调整），但走的是 PIT 安全的「未复权价 × 累计复权因子」路径，不依赖供应商的 pre_close 列。★ **不用日内层的 open5/close5**：那是未复权价，跨日相除在除权日会造出 −30% 量级的假跳空且不报错（模块 docstring 第 1 条）。★ 停牌日（`ctx.traded()` 为 False）置 NaN：价格层对 open/close 做了前向填充，不掩码的话停牌日会产出「上一个成交日自己的日内涨跌」。★ 本因子**不依赖** `stock_history_5min`，所以 2026-09-11 之后仍有值（数据冻结只影响吃日内层的因子）。
- **`idt_overnight_minus_intraday`**：参考库 Class1 `overnight_minus_intraday`（fac_cand_daily.py）逐字为 `df['overnight'] - df['intraday']`，其中 overnight=open/pre_close-1、intraday=close/open-1；文档记录 V8 筛查 **meanIC 0.030 / ICIR 0.20**，是 27 个日频候选里最强的，方向为正。★ 两条腿都走 `ctx.hfq`（后复权）且都用 `ctx.traded()` 掩码，保证「隔夜强于日内」不会被停牌日的假收益污染。★ 本因子**不依赖** `stock_history_5min`，2026-09-11 之后仍有值。
- **`idt_overnight_return_share_20`**：参考库 Class1 `overnight_return_share_20`（momentum_structure.py）逐字为 `num=_roll_sum(abs_gap,20,10); den=_roll_sum(abs_gap+abs_intra,20,10); share=safe_divide(num,den)` —— 参考库自己就用 **min_count=10 = 20//2**，本文件与它口径一致（模块 docstring 第 7 条）。★ 分子分母同量纲（都是 |收益| 之和），占比 ∈[0,1]。★ 停牌日两腿都 NaN，被 min_count 跳过（不是当 0 计入，否则停牌多的股票占比会被稀释）。★ 本因子**不依赖** `stock_history_5min`，2026-09-11 之后仍有值（分母的分母是价格层）。
- **`idt_rv_daily`**：★ 5min 口径（参考库 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：参考库 `rv_daily` 用 240 个 1min 收益，本因子用 48 个 5min 收益，**RV 系统性偏小一档**（5min 棒吃掉了棒内的价格路径，也少掉了买卖价差跳动带来的正偏差）；这是对**所有股票**方向的同向压缩，截面排序不受影响，但绝对水平不要与参考库对齐。★ **这是「日」波动率不是年化**：中位数 ≈ 0.02（2%）。要年化自己 ×√244。参考库方向为负（低波排前）。★ 停牌日 NaN（日内层无行），不是 0 —— 填 0 会造出「零波动日」。
- **`idt_rv_term_structure_slope`**：★ 5min 口径（参考库 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：参考库 `rv_term_structure_slope` = 「rv_5min/rv_60min-1」，指的是**5 分钟窗 / 60 分钟窗**的 RV 之比（注意它的名字里 5min/60min 是**窗口长度**、不是采样频率）—— 本项目按「5 日 / 60 日」两档实现，语义即「短期波动 / 长期波动」，与参考库的「短窗/长窗」一致，只是把窗口从分钟级拉到日级（本项目产物必须日频）。★ 分母是 60 日窗：`min_count=30 = 60//2`（模块 docstring 第 7 条）；分子 `min_count=2 = 5//2`。两边都开根后再相比，所以是「波动率之比」而不是「方差之比」（与参考库一致）。★ **额外要求当日 `ret2_sum` 有限**：否则窗口缺最近几天时，min_count 会放行一个**陈旧**的斜率，把长期停牌与 2026-09-11 的数据冻结藏起来。★ 停牌日 NaN 而非 0（roll_mean 只在有效日上取均值）。
- **`idt_up_minutes_ratio`**：★ 5min 口径（参考库 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：★ 参考库 `up_minutes_ratio` 是「1 分钟正收益分钟数/总分钟数」；本项目是 5 分钟棒，所以本因子**精确等于**参考库同族的 `intra_trend`（其定义逐字为「正收益 5 分钟区间占比」）。二者同义不同分辨率，方向均为正（买盘主导排前）。★ 取值 ∈[0,1]：分子 `n_up` 只数 `r>0` 的棒。★ 日内层的每根棒的首棒相对上一棒，**当日第一根棒没有前收**（段首 diff=NaN→0）所以在所有口径下都不计入 `n_up`，与参考库 1min 口径一致（240 根里最新一根也没有下一根）。副作用：全天一字涨停（所有棒收益为 0）会得到 `n_up=0` → 本因子 0.0 —— 这是**真的**（没有一根棒在涨），不是缺失；如果下游想区分「一字板」与「全天阴跌」，请配 `idt_rv_daily`（一字板 RV=0）或 `idt_close_position`（NaN）。★ 停牌日 NaN（日内层无行）。
- **`idt_vol_stability`**：★ 5min 口径（参考库 history_1min 240 根/日 → 本项目 stock_history_5min 48 根/日）：★ 参考库选型说明：父任务清单里的 `vol_concentration` 在参考库中的定义是「(开盘 30 分 + 收盘 30 分) 成交量 / 全日成交量」，**预聚合表没有首/尾 30 分钟切片，无法复现**。改用参考库 Class3 `vol_stability`：「5 分钟成交量 std/均值截面排名（取负向=不稳定排后）」——它**本身就是 5 分钟口径**，本项目可以**零退化**复现；而且它与「量能时间集中度」互为单调变换（CV = √(n·HHI − 1)，HHI = vol2_sum/vol²），是同一维度的等价刻画。★ 用**总体**标准差（ddof=0）与参考库 pandas 的 `std` 默认口径一致；ddof=1 只差常数因子 √(n/(n−1))，截面 rank 完全不变。★ 量单位为「手 vs 股」的翻转在本公式里自动约掉（比值），所以不受 2025-11/12 边界影响。★ `vol=0`（当天只有停牌级别的异常）→ NaN。
- **`ind_beta_60`**：**本文件新造**（参考库没有这一条；它的 `beta_*` 族全部对宽基指数）。★ 与已注册的 `beta_60`（对沪深300）**不是**同一个回归量：沪深300 是大市值加权，行业指数是行业内市值加权；一只小盘股对沪深300 的 β 很低，但对自己行业的 β 可以很高。★ 本因子用**动态行业归属**（逐年变化）：β 的变动里既含「耦合真的变了」也含「归属换了行业」，**这是有意为之** ——归属本身是按历史相关性定的，换行业就意味着耦合结构真的重构了。经济含义：β 高 ⇒ 个股基本由板块驱动（选股空间小、更像 β 交易标的）；β 低 ⇒ 有独立行情（alpha 标的）。方向取负。60 日窗、min_count=30。
- **`ind_bps_yoy`**：★ `ctx.ind()` 的同期同比类字段。**为什么「每股」这个框定是本质的**：被删的 `yoy_equity`（净资产同比）**不扣除股本扩张** —— 一家靠增发把净资产做大 30% 的公司，`yoy_equity` 会给出 +30% 的「成长」，但老股东其实被摊薄了。`bps_yoy` 的分母是**每股**，增发被股本扩张抵消，测的是**老股东每股权益的累积**（= 留存收益驱动的内生增长）。两者在数学上**不是**单调变换关系（前者正比于股本增速的差）。⚠ **eval 必查项**：若实测与任何「净资产同比」序列的 rank 相关 > 0.95，说明「每股」这个调整没起作用（A 股多数年份股本变动很小），那就应当删掉 —— 这条判断写在 note 里是为了让下游有据可依。
- **`ind_currentdebt_to_debt`**：★★ **本因子是 `ctx.ind()` 白名单字段的首次启用**（此前 26 个字段零个被消费）。字段本身是供应商算好的**时点比率**（`end_date` 上的瞬时量，不是累计 YTD，所以能当日频用 —— 见 `factors/DEVELOPING.md` §5 的白名单）。**为什么它是独立的维度**：已删的 `debt_asset_ratio`（负债/资产）是**杠杆水平**；本因子是**期限结构**（短期债务/总债务）——同样的杠杆率，全部是短期借款（一年内要还）与全部是长期债券，风险完全不同。两者在数学上**不是**单调变换关系。**实测覆盖**：中位 84%、q05 33%、q95 99%、100% 非空 ⇒ 填充良好且不零膨胀。方向取负（短期债务占比越高，展期风险越大）。
- **`ind_disp_ma_20d`**：★★ **口径已改，与参考库不同，这是刻意的**：参考库 cat-sector-c1 的 `ind_disp_ma_*` 是「**行业日收益横截面** std 的均值」—— 那是一个**市场级常数**（每天全市场一个值），在横断面上的离散度恒为 0，**对排序任务零信息**（与 `factors/breadth.py` 模块 docstring §一 同一条理由）。本文件改成「**所属行业**成员日收益截面 std 的均值」：每只股票取**自己行业**的分化度，不同行业拿不同的值，截面重新有区分度。语义也随之从「今天市场分化不分化」变成「**我这行**今天分化不分化」。经济含义：行业内分化高 = 选股空间大但也更难；低 = 板块共振、适合 β 交易。方向取负（分化 = 不确定性）。
- **`ind_disp_ma_5d`**：★★ **口径已改，与参考库不同，这是刻意的**：参考库 cat-sector-c1 的 `ind_disp_ma_*` 是「**行业日收益横截面** std 的均值」—— 那是一个**市场级常数**（每天全市场一个值），在横断面上的离散度恒为 0，**对排序任务零信息**（与 `factors/breadth.py` 模块 docstring §一 同一条理由）。本文件改成「**所属行业**成员日收益截面 std 的均值」：每只股票取**自己行业**的分化度，不同行业拿不同的值，截面重新有区分度。语义也随之从「今天市场分化不分化」变成「**我这行**今天分化不分化」。经济含义：行业内分化高 = 选股空间大但也更难；低 = 板块共振、适合 β 交易。方向取负（分化 = 不确定性）。
- **`ind_dt_netprofit_yoy`**：★ `ctx.ind()` 白名单的**同期同比**类字段（分子分母都是同期 YTD，季节性自动抵消，所以能当日频用 —— `factors/DEVELOPING.md` §5）。**这是参考库当年**拿不到**的字段**：参考库的 `earnings_cut_to_market` note 里明确写过它只能用「归母净利」代理扣非（「扣非」= 扣除非经常性损益），而供应商直接给了 `dt_netprofit_yoy`。★ 与被删的 `yoy_net_profit`（归母净利同比）的区别：**分子不同**。扣非剔除了政府补贴、资产处置、公允价值变动、投资收益等一次性项目 ——同一个「净利润同比 +30%」，全部来自卖楼 vs 全部来自主业，含义完全相反。而已删的 `earnings_cut_to_market` 是因为「用市值做分母」被删的，不是因为「扣非」这个想法。
- **`ind_int_to_talcap`**：★ `ctx.ind()` 白名单字段（见 `ind_currentdebt_to_debt` 的 note）。**为什么它是独立的维度**：已删的 `debt_asset_ratio` 把**应付账款、预收款、应交税费**这些**经营性负债**也算进了杠杆 ——而经营性负债恰恰是「占用上下游资金」的**能力**（越多越强势）。本因子只数**有息负债**（借款、债券），把「融资性杠杆」从经营性负债里分出来，这是 D/A 口径做不到的切分。⚠ **有一部分公司恰好为 0**（完全没有有息负债）—— 那是**真实值**（无杠杆），不是缺失，**不要**做 winsor 或抹成 NaN；引擎的 1%/99% winsor 会保护排名列。（实测 2012 年该因子零值占比 16.1%，2019 年后比例更低。）方向取负。
- **`ind_mom_accel`**：参考库没有这一条，但它是 `ind_ret_ma_5d` 与 `ind_ret_ma_20d` 的自然组合（同族的 `momentum_accel_60_120` 在个股层已被删为 NOISE，但那是在**个股**层：个股短长动量差被噪声主导；行业层是 184 个成员的平均，噪声被压掉一个量级，同样的构念才站得住）。★ 截面自由度同样是行业数（184），见 `ind_ret_ma_*` 的 note。> 0 ⇒ 板块动能正在**抬升**（短均线高于长均线）。
- **`ind_ret_ma_20d`**：抄参考库 cat-sector-c1 `ind_ret_ma_*`（`行业等权日收益的 k 日均值`）。★ **本因子的截面自由度是行业数（184）而不是股票数（3484）** ——同行业的所有股票拿到**同一个值**，是行业层的阶梯函数。对横断面排序仍有信息（模型能学到「哪个板块在动」），但下游做行业中性化 / 特征重要性时**必须知道这一点**，别把它当成个股信号。★ 行业收益用指数点位口径（模块 docstring §三.3），不是等权均值。
- **`ind_ret_ma_3d`**：抄参考库 cat-sector-c1 `ind_ret_ma_*`（`行业等权日收益的 k 日均值`）。★ **本因子的截面自由度是行业数（184）而不是股票数（3484）** ——同行业的所有股票拿到**同一个值**，是行业层的阶梯函数。对横断面排序仍有信息（模型能学到「哪个板块在动」），但下游做行业中性化 / 特征重要性时**必须知道这一点**，别把它当成个股信号。★ 行业收益用指数点位口径（模块 docstring §三.3），不是等权均值。
- **`ind_ret_ma_5d`**：抄参考库 cat-sector-c1 `ind_ret_ma_*`（`行业等权日收益的 k 日均值`）。★ **本因子的截面自由度是行业数（184）而不是股票数（3484）** ——同行业的所有股票拿到**同一个值**，是行业层的阶梯函数。对横断面排序仍有信息（模型能学到「哪个板块在动」），但下游做行业中性化 / 特征重要性时**必须知道这一点**，别把它当成个股信号。★ 行业收益用指数点位口径（模块 docstring §三.3），不是等权均值。
- **`ind_ret_ma_60d`**：抄参考库 cat-sector-c1 `ind_ret_ma_*`（`行业等权日收益的 k 日均值`）。★ **本因子的截面自由度是行业数（184）而不是股票数（3484）** ——同行业的所有股票拿到**同一个值**，是行业层的阶梯函数。对横断面排序仍有信息（模型能学到「哪个板块在动」），但下游做行业中性化 / 特征重要性时**必须知道这一点**，别把它当成个股信号。★ 行业收益用指数点位口径（模块 docstring §三.3），不是等权均值。
- **`inside_bar_count_20`**：参考库 Class1 `inside_bar_count_20`（structure_patterns.py）用 `_roll_sum(inside, 20, 5)` = **20 日窗口内满足形态的交易日**（计数，不是比率）。本实现按任务书做成**占比**（`roll_mean(inside, 20, 10)`）：满窗 20 天时`count/20` 与 `count` 只差常数 20 → **截面 rank 完全相同**；但占比对停牌日天然免疫（计数会因为窗口里少了 3 天而系统性偏小，而且偏小的幅度与停牌频率相关）。min_periods 由 5 提到 10（N//2，模块 docstring 第 5 条）。★ 参考库注明「跨日比较用复权口径高低点」——本实现用 `ctx.hfq('high'/'low')`。★ 掩码 `traded(T) & traded(T-1)`（要跟昨日高低比）。值域 [0,1]。
- **`interest_coverage`**：★ 参考库 #6 `icr` 逐字。**起点 2019-05-01（偏离 start=None）**：上游 `fin_exp_int_exp`（财务费用中的利息支出）在 2019 年之前的报告期里**100% 是精确的 0.0**（实测 2012Q1~2017Q4 全部为 0；2018 年报只有 45% 非零；2019 年起 ~78% 非零），即「没披露」而不是「没有利息支出」。不设起点的话 2012-2018 的因子恒为 NaN，看起来像 bug。取 2019-05-01 = 全部主板公司 2018 年报（2019-04-30 前）都已公告，TTM 窗口不再混入旧披露口径。分母地板 100 万元：利息支出小于此视为无有息负债，比率无意义。EBIT 为负 → ICR 为负 → 「赚的钱不够付利息」，方向语义正确。
- **`intraday_ma_20d`**：参考库 Class1 `intraday_ma_20d`（fac_new_daily.py），min_periods=1 → 本实现 min_count=10。掩码只用 `traded(T)`（当日量）。
- **`intraday_ma_5d`**：参考库 Class1 `intraday_ma_5d`（fac_new_daily.py），`roll` 默认 min_periods=1 → 本实现 min_count=2（N//2）。掩码只用 `traded(T)`（当日量）。
- **`intraday_ma_60d`**：参考库 Class1 `intraday_ma_60d`（fac_new_daily.py），min_periods=1 → 本实现 min_count=30。掩码只用 `traded(T)`（当日量）。
- **`intraday_ret_momentum`**：参考库 Class1 `intraday_ret_momentum`（price_deep.py）逐字为 `(close - open)/open` 再取截面排名 —— 名字里有 momentum，但**定义就是单日日内收益**，没有任何滚动窗口。本因子返回原始值（rank 由引擎做），方向不变。★ 与同族 `intraday_ma_5d` 的区别：那个是 5 日均值，这个是**当日**值（所以 warmup 只给 W5 冗余，面板右端 T 就是 T）。★ 掩码只用 `traded(T)`。参考库用 `open.replace(0, np.nan)` 挡 0 价，本实现用 `safe_div(min_abs_den=1e-8)`（后复权价最小量级 0.01，不误伤）。
- **`intraday_ret_share_20`**：与 P0-2 家族的 `overnight_*` 是**同一现象的两个视角**（那边是绝对量、这边是占比）——保留两者是因为占比口径对停牌/低流动性股票更稳（分母自带缩放）。★ 用 `ctx.hfq`（后复权）算两条腿，停牌日用 `ctx.traded()` 掩码挡掉。
- **`intraday_vol_ratio_5d`**：参考库 Class1 `intraday_vol_ratio_5d`（fac_short_term.py）逐字为 `m_ia / (m_oa + 1e-8)`，`roll` 默认 min_periods=1 → 本实现两条腿 min_count=2（N//2）。★ 注意分母是 **|隔夜| 的均值**（与 `overnight_intraday_ratio_20d` 不同 ——那个的分母是 |隔夜均值的绝对值|）。分子分母都 ≥0，比值 ≥0，**不是有界量**（隔夜全为 0 时溢出），保留 `+1e-8` 地板即参考库口径。★ 两条腿同用 `traded(T) & traded(T-1)`（模块 docstring 第 1 条）。
- **`inventory_turnover`**：★ 分子用**营业成本** TTM（参考库 Cost_TTM），不是营收 —— 存货与成本配比。★ 分母地板 100 万元（元）：银行/券商的 inventories 是**精确 0.0**（结构性没有存货科目），0 分母必须返回 NaN 而不是 inf/巨值（契约要求，不许 nan_to_num 填）。此外 oper_cost 对金融股已在 deriv 层置 NaN，所以金融股天然不参与本因子 —— 这是两层保护。地产/建筑公司的存货周转率天然很低（~0.3），是行业属性。
- **`invest_income_share`**：**本文件新造**（参考库 Quality #10 `quality_composite` 用到 TotalProfit 族的比率，但没有单独发过投资收益占比）。**为什么是份额而不是水平**：份额的分母（利润总额）随公司规模变化，分子分母同报告期 ⇒ 截面离散度大、时序变化快，逃开「慢水平」的死因。**实测可用性**：`invest_income` 的非零公司占比从 2012 的 65% 升到 2022 的 94%，全历史可用，不是近年才有的字段。★ 分母 `total_profit` 可以为负（亏损），此时 `safe_div` 的地板（1e6）在金额上等价于「利润总额绝对值小于 1 百万元」⇒ NaN，语义正确（小额度的比重没有意义）。方向取负：越依赖投资收益，主业质量越差。
- **`kdj_k_minus_d`**：★ 参考库**没有**日频的 kdj_k_minus_d 条目：factors.md `类别 intraday` Class 4 只有分钟级的 `kdj_k_d_distance`（(K−D)/|D|，归一化以消除 K 的水平）。本因子按任务要求做**日频**版，直接取 K−D（不除以 |D|）——因为 D 可能穿过 0 使 |D| 变成极小的分母，除法会把一条平滑的动能差变成带尖刺的比值；K−D 本身有界（∈[−100,100]），截面排名不关心量纲，更稳。★ 该因子与 kdj_daily_j 取自同一条链，两者相关性会很高（J=3K−2D），下游建模时注意共线性。
- **`label_ret_10d`**：★ 这是**标签不是因子**：引擎跳过了 winsor 与截面 rank（rank 列恒为 NaN），value 即未来收益。复权开盘价 = 未复权开盘价 × 当日累计 adj_factor（后复权锚定，PIT 安全）。★★ v2 修了一个**静默算错**的 bug（2026-09-15）：面板为标签向后延伸了 `forward_days` 天，但延伸段**没有真实行情**，而价格层对开盘价做了前向填充 →`o[T+21]` 取到「最后一个真实交易日」的陈旧价，一条 4~6 天的短收益被冒充成 20 天标签（实测 `label_ret_20d` 在 09-04/07/08 仍有值，而它们的 T+21 在 10 月中旬）。现在按「卖出行必须落在还有成交的范围内」截断，超出的一律 NaN。
- **`label_ret_1d`**：★ 这是**标签不是因子**：引擎跳过了 winsor 与截面 rank（rank 列恒为 NaN），value 即未来收益。复权开盘价 = 未复权开盘价 × 当日累计 adj_factor（后复权锚定，PIT 安全）。★★ v2 修了一个**静默算错**的 bug（2026-09-15）：面板为标签向后延伸了 `forward_days` 天，但延伸段**没有真实行情**，而价格层对开盘价做了前向填充 →`o[T+21]` 取到「最后一个真实交易日」的陈旧价，一条 4~6 天的短收益被冒充成 20 天标签（实测 `label_ret_20d` 在 09-04/07/08 仍有值，而它们的 T+21 在 10 月中旬）。现在按「卖出行必须落在还有成交的范围内」截断，超出的一律 NaN。
- **`label_ret_20d`**：★ 这是**标签不是因子**：引擎跳过了 winsor 与截面 rank（rank 列恒为 NaN），value 即未来收益。复权开盘价 = 未复权开盘价 × 当日累计 adj_factor（后复权锚定，PIT 安全）。★★ v2 修了一个**静默算错**的 bug（2026-09-15）：面板为标签向后延伸了 `forward_days` 天，但延伸段**没有真实行情**，而价格层对开盘价做了前向填充 →`o[T+21]` 取到「最后一个真实交易日」的陈旧价，一条 4~6 天的短收益被冒充成 20 天标签（实测 `label_ret_20d` 在 09-04/07/08 仍有值，而它们的 T+21 在 10 月中旬）。现在按「卖出行必须落在还有成交的范围内」截断，超出的一律 NaN。
- **`label_ret_3d`**：★ 这是**标签不是因子**：引擎跳过了 winsor 与截面 rank（rank 列恒为 NaN），value 即未来收益。复权开盘价 = 未复权开盘价 × 当日累计 adj_factor（后复权锚定，PIT 安全）。★★ v2 修了一个**静默算错**的 bug（2026-09-15）：面板为标签向后延伸了 `forward_days` 天，但延伸段**没有真实行情**，而价格层对开盘价做了前向填充 →`o[T+21]` 取到「最后一个真实交易日」的陈旧价，一条 4~6 天的短收益被冒充成 20 天标签（实测 `label_ret_20d` 在 09-04/07/08 仍有值，而它们的 T+21 在 10 月中旬）。现在按「卖出行必须落在还有成交的范围内」截断，超出的一律 NaN。
- **`label_ret_5d`**：★ 这是**标签不是因子**：引擎跳过了 winsor 与截面 rank（rank 列恒为 NaN），value 即未来收益。复权开盘价 = 未复权开盘价 × 当日累计 adj_factor（后复权锚定，PIT 安全）。★★ v2 修了一个**静默算错**的 bug（2026-09-15）：面板为标签向后延伸了 `forward_days` 天，但延伸段**没有真实行情**，而价格层对开盘价做了前向填充 →`o[T+21]` 取到「最后一个真实交易日」的陈旧价，一条 4~6 天的短收益被冒充成 20 天标签（实测 `label_ret_20d` 在 09-04/07/08 仍有值，而它们的 T+21 在 10 月中旬）。现在按「卖出行必须落在还有成交的范围内」截断，超出的一律 NaN。
- **`large_order_timing_signal`**：逐字抄参考库 fund_flow 的 `large_order_timing_signal`。★ 参考库用 `_adjusted_close`（**前复权**）算价格位置，本项目**禁止** qfq（`fea/spec.py` 的红线）⇒ 改用 `ctx.hfq`（截至当日的后复权），数学上等价于「截至当日的复权序列」，且历史值稳定。★ 为什么这是一个**交互**而不是两个因子的乘积：「大单净买入」单独看无法区分「低位吸筹」与「高位接盘」，乘上 `1 − 位置` 之后才表达「在大钱**便宜**的时候买」。`safe_div` 挡掉 20 日区间为 0（长期一字板）的退化格。
- **`limit_board_streak_mean_60`**：`sealed` 用 `stock_limit_up` 的**行本身**（= 收盘封板），不是 pct_chg≥9.8：上游表只收录封板成功的票，比价格阈值更干净（炸板票不在表里）。「启动日」在网格上用 `ctx.shift(sealed,1)` 求（面板即交易日历，移一行 = 移一个交易日），首行 NaN 视为未封板（对齐参考库 fillna(False)）。★ 偏离参考库一处：`.fillna(0.0)` 保留（60 日内没封过板的股票 avg = 0），这是参考库明确的选择（「保证全市场覆盖」），0 = 「没有连板基因」，方向上也对。分布：约 1/3 的股票为 0，非零样本集中在 1.0~3.0（首板/二板），长尾到几十。
- **`limit_down_event_5`**：★ 与参考库的三处偏离（逐条给理由）：① **事件判定优先用 `stock_limit_list.limit == 'D'`**（2020-01-02 起），2020 年前才退到价格近似 `pct_chg ≤ −9.5 且 close == low`（任务书口径）。实测两口径在 2020~2026 主板高度一致：D 标记 23,562 行里 23,555 行 close == low；价格近似多出的 198 行全部是供应商标记为 `Z`（触板未封）而收盘恰在跌停价上的样本，本文件以 LL 为准（≈0.7% 的事件数差异）。② **衰减用「加权和」而不是参考库 `event_decay` 的「最近一次事件 × 指数」**：参考库只保留最近一次事件（3 天内跌停两次 = 只跌停一次），本批按任务书范式取 Σ_{k=0}^{4} 0.5^(k/3)，多次跌停会累积 —— 这正是「恐慌加速」的语义，也让因子对「连续跌停」与「单日跌停」有区分度。③ 参考库阈值 −9.8% 只用在**价格近似**分支，且放宽到 −9.5%：实测 D 标记里有 153 行的 pct_chg 落在 (−9.8, −9.5]（四舍五入报 −9.5x%），用 −9.8 会漏掉它们。★ **零膨胀（固有，非口径太稀）**：跌停本身是罕见事件 —— 2026 年主板（非 ST）跌停事件约 0.4% 的 stock-day，5 日窗口内至少一次的比例约 3%，故本因子 **实测零值占比 97.54%**（2026 全年 518,758 行 / 170 天 / 日均 3050 只，非零格子上界 3.3205 = 5 个连续跌停的完整和 ∑0.5^(k/3)）。>95% 的零值占比不通过「调窗口」来救：因子的名字与定义就是 5 日窗口，且截面 rank 在大量并列 0 上仍然可用（rank 由非零样本分层，非零样本内部按 0.5^(k/3) 的权重自然分层）。若要更稠密，应换连续量（如 `limit_down_rebound_10` 那类事件后收益），不是放宽窗口。
- **`limit_up_count_20`**：纯交易日滚动计数，与财务口径无关，是这批因子里唯一的行为面因子
- **`limit_up_event_5`**：★ 涨停侧本文件**只做这一个**（任务书：与 event2.py 的 18 个涨停因子不重复）。event2 覆盖的是计数/连板/封板时间/炸板率/龙虎榜/事件后收益，没有「涨停事件的时间结构」——本因子补的就是这个：同样是「近 5 日涨停过」，昨天涨停得 1.0、五天前涨停得 0.397。★ 事件判定与 `limit_down_event_5` **完全对称共用**（LL 优先 + 价格近似回退），偏离理由同该因子。实测 2020~2026 主板 U 标记 87,071 行里 86,981 行 pct_chg ≥ 9.5（99.9%）；U 标记比价格近似多的 169 行是 pct_chg 落在 [9.5, 9.8) 的样本。★ 与 `limit_up_fade_10`（event2）的区别：那个是「10 日内涨停过 × 随后 10 日累计收益」（事件**后**的表现），本因子是「涨停事件本身的时间加权强度」（不含任何未来信息，也不含收益）。★ 零膨胀：**实测零值占比 91.58%**（比跌停侧的 97.54% 低，因为涨停事件本身比跌停频繁约 3 倍 —— 2026 年主板 U 标记 8.7 万行 vs D 标记 2.8 万行），|value|max = 3.3205（5 连板）。★ **与 `extreme_move_event` 高度冗余（实测 |ρ|=0.887，被引擎 `dedup` 判为重复簇）**，原因见 `extreme_move_event` 的 note —— 那是「|pct_chg| ≥ 9.5%」阈值与主板 ±10% 涨跌幅撞车，与本因子无关。
- **`liquidity_shock_20`**：先取 log 再作差（消除量纲/尺度差异），所以**不能**给 amihud 加 ×1e8 —— 参考库这里也是不带 ×1e8 的，本族统一（见模块 docstring §4）。`+1e-12` 是参考库的下限，对零收益日（|ret|=0 → amihud=0）给出 −27.6 的底值，保留原样不改。有效窗口是 40 个交易日（20 日窗 + 往前 20 日），所以 warmup 给 40×1.8+20。停牌日 amihud 是 NaN，两个窗口各自跳过。
- **`log_mv`**：★ 市值自己算：未复权 close（停牌前向填充）× 当期已披露 total_share。不用供应商 total_mv —— 日频快照表会被事后重算，自算是 PIT 可控的（DEVELOPING §3.1）。实测对拍（见 mv_vendor_gap）：供应商 total_mv ≡ close × total_share，**单位是元，不是 /1e4**；逐行相对偏差 ≤ 5.4e-10（双精度往返）。方向：小市值长期有溢价 → higher_is_better=False。
- **`macd_daily_hist_5d`**：参考库出处：factors.md `类别 price` / technical_daily.py 的 macd_daily_hist_5d。本文件逐字照搬其归一化口径（除以 EMA26 后再算 DEA，而不是算完原始柱再除）——这样 DIF 与 macd_daily_dif 是同一个数，两个因子可以放在一起看。★ `hist.diff(5)` 用交易日位移（面板本身就是交易日历），不是 5 个日历天。★ 数值：柱是 12/26 两个 EMA 之差再减去其 9 日 EMA，量级 ~1e-3，float32 下有效位约 4 位；这是参考库的口径，保持不变。
- **`margin_balance_20d`**：★ 滞后表：`lagged_ok` 声明 + `ctx.lag_grid(grid, 1)` 是**最后一步**。分子分母都是同一列、间隔 20 个交易日，分母用 |rzye(T-20)| 并设 1 元地板（余额被清零的股票直接给 NaN，而不是 ±1e6 的假变化率）。上游实测 2026-09-10 截面 p1/p50/p99 = -0.45 / -0.026 / +1.20 —— 「刚纳入两融名单」的股票会出现 +100% 以上的变化率（真实事件，引擎 1%/99% 缩尾会处理）；参考库的 clip(-0.5, 1.0) 同理，未实现。停牌日：余额是状态量，asof 前向填充；实测上游停牌日仍有行且余额在变。
- **`margin_balance_5d`**：与 `margin_balance_20d` 同口径，窗口 5 日（短期边际变化）。上游实测 2026-09-10 p1/p50/p99 = -0.30 / -0.0079 / +0.59。两个窗口**不重复**：沙箱实测（2012~2015）日内 rank 相关中位数 0.516（5 日抓拐点、20 日抓趋势），保留两者。
- **`margin_balance_ma_divergence`**：★ 两融 PIT 铁律：`lagged_ok=("stock_margin_detail",)` + **最后一步** `ctx.lag_grid(grid, 1)` —— T 日只用 ≤ T−1 的两融记录。参考库写的是 `margin_detail.parquet`（其面板已 shift(1)），本实现等价。★ 偏离：不做参考库的 `.clip(-0.1, 0.1)`（契约禁止因子内 winsor）。实测 2026-09-10 上游原始表 p1/p50/p99 = −0.106 / −0.0006 / +0.107 —— 参考库的 clip(±0.1) 恰好压在 p1/p99 上（说明它本来就是手工缩尾），引擎的截面 1%/99% 缩尾覆盖同一批极端值，且口径统一。★★ **冗余警示（实测，必须看）**：本因子与既有 margin 因子的相关很高 ——沙箱 2026 逐日截面 rank 相关：`margin_leverage_trend_10d` **+0.883**、`margin_flow_asymmetry_10d` +0.818、`margin_balance_20d` +0.811（单日最高 0.930）。根因是数学的：余额平滑增长时「(rzye − MA20)/MA20 ≈ g × 9.5」而「20 日变化率 ≈ g × 20」（g = 日均增速），两者都是同一个 g 的（近似）单调变换 —— 截面上必然高相关。★ 试过的解耦方案**实测反而更差**，故保留参考库原式：把偏离按自身 60 日标准差归一化（「极端度」）后，与 `margin_flow_asymmetry_10d` 的相关升到 **+0.794**（未归一化时 +0.659，同一口径对比），因为归一化引入的波动本身与资金流波动同向。**处置建议**：本因子与上述三条属于同一个「杠杆趋势」簇，下游做因子筛选时**至多保留其一**；保留本因子的唯一理由是它多了一层「相对自身中枢的位置」语义（见下方方向说明）。`min_count=10` = 参考库 min_periods。★ 方向标注为「大者优」沿用参考库的 `rank(div)`；但请注意其**语义是均值回归**（两端偏离都可能回归），即该因子的有效信息是**非单调**的；而上面那三条同簇因子是**趋势**语义（大者优）—— 两条高相关、语义相反的因子同时进线性模型会互相抵消，这一点务必交由筛选环节处理。停牌日：余额是状态量，`asof` 前向填充（实测上游停牌日仍有行且余额在变）。★★ **「已退出两融名单」的股票怎么处理（实测驱动的一次修复）**：余额是状态量，asof 会把最后一条记录无限前向填充 —— 于是**退市/退出两融的股票**在 20 日窗内余额恒定 → 偏离度恒等于 0，即「余额没变化」的假信号。更严重的是**一致性**：掩码之前，同一格的值取决于面板起点（全量跑的 warmup 伸到上一年、读得到上一年的最后一条记录 → 0；增量跑读不到 → NaN），实测 7 只 2025 年 1~4 月退出两融的股票在 2026-09-04~09-14 每天 7~20 格两趟不一致。现在 `_mg_state` 统一加**新鲜度掩码**（最近 20 个交易日内必须有该股的两融记录，否则 NaN），并按交易日距离（而非「读到没读到」）判定 —— 实测：全量跑 vs `--start 2026-09-01` 的尾部跑，19,540 格**逐格相同、非空模式一致**；`main.py audit-pit`（截断到 2026-05-18 / 2026-09-14 重算）两个样本日 **0 个不一致**。实测值域：2026 年 |value|max = 2.93（未 clip，重尾），中位数 −0.0125。
- **`margin_balance_volatility_20d`**：参考库用 pandas `.std()`（ddof=1），本框架 `ctx.roll_std` 是总体口径 ddof=0；在 20 日窗上两者差一个 n/(n-1) 的常数因子，**截面排名完全不受影响**（同一个 n 对所有股票相同），故不另做修正。实测 2026-09-10 CV p1/p50/p99 = 0.0076 / 0.049 / 0.455 —— 参考库的 clip(0, 0.5) 恰好落在 p99 上，说明它本来就是手工缩尾，引擎的 1%/99% 缩尾覆盖同一批极端值。分母（余额均值）设 1 元地板：余额接近 0 的股票直接 NaN，而不是 ±1e6 的假 CV。
- **`margin_buyer_avg_cost_premium`**：★ 参考库的 `_margin_weighted_cost` 源码未公开（factors.md 只给了调用），本实现按「近 20 日融资买入额加权的成交价」落地：cost = Σ(rzmre × close) / Σ(rzmre)，min_count=10（20 日窗的半数，与同族一致）。★ 价格用**后复权** `ctx.hfq("close")`（分子分母同一口径）：如果用未复权 close，除权日会凭空造出 −10% 的「融资盘被套」，而后复权把分红还原成收益，得到的才是真正的「浮盈/被套」；且后复权是 PIT 安全的（历史值不被未来分红改写）。★ 参考库 note 写「margin 数据已 shift(1)，配对价格用 close 逐股 shift(1)」—— 本实现等价：整个网格一起下移一格，T 日的值用 T-1 的融资买入、T-1 的收盘价。实测 2026-09-10 该因子 p1/p50/p99 ≈ **−0.15 / −0.006 / +0.15**（p25/p75 = −0.04 / +0.02，标准差 0.057；融资盘加权成本紧贴现价，符合「融资盘平均持仓期很短」的事实 —— 若这个因子某天开始出现 ±0.5 以上的截面离散度，先怀疑成本端用错了复权口径）。该统计用上游原始表直接重算（20 日窗、min_count=10、仅两融标的），与沙箱落盘值的差异来自 universe 掩码与停牌处置，量级一致。
- **`margin_chg_rel_5d`**：★★ 偏离参考库（口径级，必须看）：factors.md 对应条目是 `margin_chg_rel_ind_5d` —— 「融资余额5日变化率**减去行业等权均值**」。本框架**没有行业表**（`ctx` 无行业字段，`fea/**` 也不提供），硬做只能去读 `stock_list.industry`，而那是**当前时点快照**、用在 2012 年的因子上属于前视（本族的全部意义就是 PIT 正确，不为此破例）。同时，「减去**全市场**当日等权均值」是**每日常数平移**，不改变截面排名 —— 那样写出来会和 `margin_balance_5d` 的 rank 完全重复。故改为**相对自身中期趋势**：chg5 − chg20，即「短期杠杆资金流入相对中期趋势的加速度」。它既保留了「剥离整体杠杆环境」的原意（短期 vs 中期用的是同一只股票），又不是另两个变化率因子的复制品 —— 沙箱实测（2012~2015）日内 rank 相关中位数：与 `margin_balance_5d` **−0.149**、与 `margin_balance_20d` **−0.886**。★ 与 20 日变化率的强负相关是「加速度」定义的固有性质（chg5 − chg20 里 chg20 是主导项，符号相反），不是错误；但它意味着下游若已经在用 `margin_balance_20d`，本因子主要提供的是「反向 + 5 日增量」，建议与 20 日变化率二选一或做正交化。若后续框架补上**时点化**的行业表，本因子应改写为行业相对口径（version +1）。
- **`margin_chip_cost_gap`**：★★ 两融 PIT 铁律（本因子的处置是：**两边各自算完 → 相减 → 最后统一下移一格**）：`rzmre` 的 T 日值要到 T+1 才可得，而筹码层是当日可得 —— 若只位移融资那一侧，会在两融名单变动的日子里造出「跨日拼接」的假信号；统一 `ctx.lag_grid(grid, 1)` 后，T 日的因子值只用到 ≤ T−1 的两融记录与 ≤ T−1 的筹码快照（保守但严格无泄漏）。★ 价格口径（本因子最容易踩的坑）：融资买入的加权成本用 **`ctx.px("close")`（未复权）**，因为筹码层的 `mean` 是**未复权**口径（`fea/chips.py` docstring「口径一」）。若按兄弟因子 `margin_buyer_avg_cost_premium` 那样用 `ctx.hfq` 取成本，比值会被整体放大 `adj_factor` 倍（实测 2019 年 108~142，差两个数量级）——**同口径 > 复权**，这里必须用未复权。★ 起点 `2018-01-02`：`stock_cyq_chips` 的真实起点（且只覆盖主板），在此之前无筹码数据，不做补偿。★ **覆盖率天生很低**：分母（筹码层）只覆盖主板且从 2018 起，分子（两融）只覆盖两融标的（2026 年主板约 1900 只）——两者求交后截面只剩约 1800~1900 只、非空率约 55%~60%（沙箱实测见交付报告），这是**数据源覆盖**的结果，不是实现缺陷。下游若要求全截面覆盖，本因子只能作为「两融标的子样本」因子使用。★ 不取负（方向交给 `higher_is_better=False`）：`value` = 融资盘成本相对筹码均价的**溢价率**，0 = 与市场平均成本持平，正 = 杠杆盘高位接盘。★ 与 `margin_buyer_avg_cost_premium` 的分工：那条比的是「**现价** / 融资成本」（浮盈视角），本因子比的是「**融资成本** / 筹码成本」（成本结构视角），两者共用同一套加权成本口径（本文件的 `_mg_cost`）。★ 沙箱 2026 实测（vs 全部 259 个既有生产因子的逐日截面 rank 相关）：max|corr| = **+0.523**（`avg_cost_premium`，筹码族的「现价/筹码均价」，单日最高 0.762）—— 两者共享「筹码成本」这一半，但本因子的分子是**融资成本**（外部增量信息），这正是它与筹码族不重复的部分。
- **`margin_flow_asymmetry_10d`**：两个滚动和都用 `min_count=5`（对齐参考库 `min_periods=5`），分子分母的**有效观测完全同步**（两列来自同一行、缺一起缺），所以比值天然落在 [-1, 1]。★ 这里必须用 `_roll_sum`（= `roll_mean × roll_count`）：`ctx.roll_sum` 没有 min_count 参数，窗口里缺一天就整体变 NaN，10 日窗会大面积作废。
- **`margin_leverage_change_20d`**：★ 分母是**流通市值**（用户口径），参考库用 `finance.total_mv`（总市值）—— 两者对同一只股票的截面排名有系统性差异（总市值含未流通部分，银行/次新股的差异最大），本文件统一用流通市值，与 `total_leverage_ratio` / `short_balance_ratio_change_20d` 保持一致。★ `diff` 而不是 `pct_change`：占比本身已经是标准化量，参考库用的就是 `.diff(20)`，照抄（占比的绝对值变化）。★ 滞后的配合：市值用**原始网格当日**的 close（也就是下移后的 T-1 日收盘），与参考库「Date=T 使用可获得的 T-1 融资数据与 T 日市值」不同 —— 参考库那句话自相矛盾（它自己的 note 又说 close 逐股 shift(1)）。本实现把整个网格一起下移，分子分母**严格同日**（都是 T-1），这才是无未来函数且口径自洽的做法。★ 与 `total_leverage_ratio`（同一个比值的**水平**）实测日内 rank 相关中位数只有 **0.268**（沙箱 2012~2015）—— 「变化」与「水平」基本正交，两个都保留是有增量的，不要因为名字像就当成重复因子。
- **`margin_repay_deceleration`**：分子的 `rzche` 是**流量**（覆盖外 NaN），分母是 5 个交易日前的同一列，地板 1 元 —— 实测 96.8% 的格子有值（只有 3.2% 因为「5 天前偿还额恰好为 0」被地板挡成 NaN，这是正确的：从 0 到任意值的「变化率」没有定义）。实测 2026-09-10 p1/p50/p99 = -0.87 / -0.042 / +5.96，是重尾的正数分布。★ 重尾的实测厚度（沙箱 2012~2015）：|值| > 10 只占 **1.02%**、> 100 占 0.11%、> 1000 占 0.013%，最大 1.19e6（来自「5 天前偿还额≈1 元 → 今天正常偿还」的个股，是真实数据不是脏值）。契约上限 1e8 内，且引擎的 `cs_rank` 会做 (0.01, 0.99) winsor，故不影响排序；但**下游若要用原始值（不是 rank）请自行截尾**。
- **`margin_repay_shock`**：参考库 `min_periods=10` → 本实现 `min_count=10`（窗口内至少 10 个有效观测）。★ 分母均值的有效个数口径由 `ctx.roll_mean` 保证（它在有效值上取均值），不是「把缺失当 0 除以 20」。实测 2026-09-10 p1/p50/p99 = 0.055 / 0.855 / 3.29（参考库 clip(0,5) 在 p99 之外，未实现）。停牌日 `rzche` 仍是上游真值（实测停牌期间融资盘会继续还款），不会变 NaN。
- **`margin_velocity`**：周转速度 = 当日融资交易额 / 存量余额，量纲 1/日。实测 2026-09-10 p1/p50/p99 = 0.0093 / 0.081 / 1.21（参考库 clip(0,2) 在 p99 之外，未实现）。与 `margin_buy_pressure` / `margin_net_flow_ratio` 共用分母但含义不同：本因子是**双边**成交强度（周转），那两个分别是**单边买入**与**净买入**。⚠️ 但**与 `margin_buy_pressure` 实测高度相关**：沙箱（2012~2015）日内 rank 相关中位数 **0.944**（上游原始表同日截面 0.966）—— 因为 A 股融资盘买入/偿还额量级接近，两项之和近似是买入项的 2 倍常数缩放。**下游用法建议：与 `margin_buy_pressure` 二选一**；若都要留，本因子的增量信息主要在「偿还端活跃度」，可考虑改用 (rzche−rzmre)/rzye 取正交残差。（此处不改定义是为了忠于参考库的公式原文。）
- **`marubozu_ratio_10d`**：参考库 Class1 `marubozu_ratio_10d`（trend_pattern.py）逐字如上，`min_periods=5` = 10//2，与本实现的 min_count=5 **完全一致**。窗口是 **10 日**（不是 20 日）——任务书里唯一一个 10 日窗的形态因子，本实现照抄。★ `is_marubozu` 用 `is_green` **相乘**而不是与（参考库原样）：`(upper+lower)/rng` 无定义（rng=0）时参考库靠 `.replace(0, nan)` 让`nan < 0.1` 为 False → 0；本实现用 `ok = rng > 0` 显式判 NaN。★ 掩码只用 `traded(T)`。值域 [0,1]。
- **`max_drawdown_120`**：★ 窗口口径（想清楚后写在这里）：dd = hfq_close / roll_max(hfq_close,120) − 1，即「当前价相对**含当日在内的最近 120 个交易日**最高价」的回撤，恒 ≤0（roll_max 的窗口含当日，故比值 ≤1；停牌期间 hfq 前向填充，不会造假回撤）。**不用**教科书式的「窗口内峰谷最大回撤」：那个要窗口内的运行最大值（峰值可早于窗口起点），一次 roll_max 表达不了，得物化 (T,C,120) 张量；而且参考库明说 drawdown_120 是「8.10 删除的 max_drawdown_120 的合规重建」，口径就是本文这个。min_count=60 = min_periods。
- **`max_drawdown_60`**：逐字抄自 Class1 risk / drawdown_60（即被删除的 max_drawdown_60 的合规重建），窗口口径同 max_drawdown_120。min_count=30 = min_periods。
- **`mf_amount_weighted_direction`**：逐字复刻参考库（`fund_flow.py`）。★ 它**不是**另一个「净流入率」：本因子刻意把每档的**幅度**丢掉、只留**方向**，再按「这一档在当天的成交里占多大比重」加权 —— 于是「小单巨量卖出、大单微量买入」与「小单微量卖出、大单巨量买入」得到相反结论，而按净额加总时两者可能都接近 0。★ 沙箱 2026 实测（vs 全部 259 个既有生产因子的逐日截面 rank 相关）：max|corr| = **−0.441**（`mf_big_order_ratio`），其次 `big_vs_small_divergence_5d` −0.299、`consecutive_limit_up` −0.121 —— 与全库**最高的相关也才 0.44**，是独立性最好的一批（方向信号与幅度信号不同向，正常）。取值域：Σw ≡ 1（四档毛额之和 = 八列总额，实测 max|差| < 2e-9），且四档 net 之和 ≡ 0 使方向不可能全同号，故 composite ∈ [−1, 1]。实测 2026 年 p1/p50/p99 = −0.50 / +0.06 / +0.47（近似以 0 为心）；沙箱 |value|max = 0.997。分子分母同为万元（毛额口径），无量纲。
- **`mf_avg_trade_price_momentum`**：逐字复刻参考库（`fund_flow_vol.py`），去掉两处 `clip`（契约禁止因子内 winsor，引擎统一做截面 1%/99% 缩尾；且实测 2026 年该比率 p1/p99 ≈ 0.94/1.06，参考库的 clip(0.5, 3) 与 clip(-0.3, 0.5) 在本平台都是空操作）。★ **纯同表因子，不需要任何单位换算**：`big_vwap` 与 `sm_vwap` 都是「万元/手」（= 100 元/股），相除时量纲自约，`ratio` 是无量纲的价格比（≈1）。★ 与 `mf_large_order_avg_price` 的区别（这是它最近的邻居）：后者是「大单均价 / **全日**均价」的**单日水平**，本因子是「大单 / **小单**」的**5 日变化**——换了参照系（小单 vs 全日）与阶数（变化 vs 水平），实测相关 **+0.638**（不冗余）。取值实测 2026 年中位数 0.0000、p1/p99 = −0.016/+0.017、最大 0.84（重尾来自小单成交极少的格子，靠截面缩尾处理）。`pct_change(5)` 的分母 `ratio` ≈ 1，`min_abs_den=1e-3` 只挡掉病态格子。
- **`mf_big_mid_net_corr_20`**：**本文件新造**。（双记口径：Σ4buy ≡ Σ4sell ⇒ 四档净额之和恒为 0，见模块 docstring §一.①） ★ 关键：**滚动相关跳出了恒等式的线性张成** —— 四档净额虽只活在 3 维空间里，但「两个序列的**相关**」是二阶量，不是任何单一档位净额的线性函数，所以它不是既有因子的重标定。经济含义：> 0 ⇒ 中户跟着机构同向（**跟风一致**，信号可信度高）；< 0 ⇒ 机构买、中户卖（**对手盘**，筹码从散户向机构转移）。与已注册的 `mf_big_small_divergence`（大单净 − 小单净，**方向差**）不同：一个是两序列的**协动性**，一个是两序列的**水平差**，可以「大单大幅净买、中单也小幅净买」相关为 1 而差值为大。用**净额占比**（÷八列毛额）而不是净额原值，避免大市值股票主导相关。
- **`mf_big_order_net_ac1_20`**：**本文件新造**。（双记口径：Σ4buy ≡ Σ4sell ⇒ 四档净额之和恒为 0，见模块 docstring §一.①） 与「净流入天数 / 连续性」类因子的分工：**已删**的 `mf_flow_streak_5d` / `mf_flow_stability_20d` 用的是**计数与波动**（几天为正、标准差多大），是**零阶/一阶矩**的时序统计；本因子是**自相关**（二阶），衡量「今天的流入能不能预测明天的流入」——一只每天小幅正流入的股票（低波动、高持续性）与一只「流入流出交替但长期累计为正」的股票（高波动、负持续性）在计数与波动口径下可能一模一样，本因子能把它们分开。★ 与已注册的 `mf_flow_factor_momentum_20`（同一序列的 20 日差分）不同：差分是**水平的变化**，自相关是**时序结构**，两者正交性高。用净额占比（÷八列毛额）归一化，避免大市值主导。
- **`mf_big_order_net_kurt_20`**：抄参考库 fund_flow 的 `big_order_net_kurt_20`（窗口 20）。（双记口径：Σ4buy ≡ Σ4sell ⇒ 四档净额之和恒为 0，见模块 docstring §一.①） 用**净额占比**（÷八列毛额）而不是净额原值：峰度对尺度不变，但原值口径下大市值股票的流量序列会被市值量级主导，归一化后比较的是**形状**而不是规模。★★ **必须带退化守卫**（抄 `ret_kurt_20` 的既有做法）：窗口内若二阶矩趋 0（连续多日净额几乎不变），峰度的分母趋 0 会产出1e8 量级的垃圾值，触发 `main.py check` 的「值域异常」。守卫：`|kurt| > 10000` → NaN。经济含义：峰度高 ⇒ 建仓是**几次大额脉冲**（事件驱动 / 大宗 / 指数调仓），峰度低 ⇒ **匀速滴灌**（算法拆单的被动配置）。这两类资金对后续收益的含义完全不同：脉冲式更可能是短期冲击。
- **`mf_big_order_ratio`**：分层净额口径。参考库的 `mf_big_order_ratio` 与 `ext_mf_big_order_net_amount_ratio` 是同一个式子（后者只是命名前缀不同），故这一条同时覆盖两者。★ 这也是 `smart_money_concentration` 的**精确重复**：由 双记口径：本表每笔成交同时进买方桶与卖方桶，Σ4buy ≡ Σ4sell，故参考库「买方四桶和 − 卖方四桶和」恒为 0（常数因子）。 可推出四档净额之和 ≡ 0，于是（大单净额 − 噪音净额）= 2×大单净额，参考库那个因子与本因子截面排名相关 **= 1.000**，已砍掉（见文件末尾 `_DROPPED`）。
- **`mf_big_small_divergence`**：逐字复刻参考库（`fund_flow.py`）。★★ **这是本族唯一保留的高相关因子，下游请按需剔除（诚实的量化交代）**：由 双记口径：本表每笔成交同时进买方桶与卖方桶，Σ4buy ≡ Σ4sell（见 fundflow.py docstring ①）。 可知四档净额之和恒为 0，于是「大单净额 − 小单净额 = 2×大单净额 + 中单净额」—— 本因子在代数上就是「大单净额」加一个固定权重的「中单净额」，**任何窗口平滑都消不掉这个恒等关系**。实测（2026 年逐日截面 rank）：与 `mf_big_order_ratio` **+0.938**、与 `mf_smart_dumb_divergence` **+0.963**、与 `mf_small_order_ratio` −0.906（后两者判定当时在册，随后已被主 Agent 从 `fundflow.py` 收口时裁掉；数字来自上游表复算，不依赖它们在册）。试过的三条改口径路线都被否掉：(a) 换成量口径 → 相关 0.938（几乎不变，量/额在双记下等价）；(b) 按各自档位毛额归一 → 就是参考库的 `mf_smart_dumb_divergence` 本身（相关 1.000）；(c) 改成买/卖对数不对称之差 → 与它相关 **1.000**（单调变换保序）。保留的理由只有两条：参考库 `fund_flow.py` 里它就是这个名字与这个公式（下游可能按名字对接），且「机构买、散户卖」的**联合方向**读起来比单看大单净额直观。**若下游做因子筛选，本因子与上述两条高度共线，建议只保留其中一条。**分母 = 八列毛额（万元，含双记的 ×2 常数，对截面排名无影响）；分子分母同表同量纲，无需换算。
- **`mf_elg_lg_split_20`**：**本文件新造**。（双记口径：Σ4buy ≡ Σ4sell ⇒ 四档净额之和恒为 0，见模块 docstring §一.①） ★ **与「大单」合计口径的区别**：既有的所有因子都用 `big = lg + elg` 把这两档**合并**了 ——合并会丢掉「**哪一级**在买」这一维。在双记恒等式下，同日的四档净额只活在 3 维空间里，但 `elg − lg` 是其中**一个独立方向**（不与 `lg + elg` 共线），而且本因子取的是**20 日均值之差**（不是同日之差），在时间维上进一步与同日线性组合分离。经济含义：超大单 = 公募/保险/量化通道，大单 = 游资/私募 ——> 0 ⇒ 配置型资金主导，< 0 ⇒ 交易型资金主导。⚠ 与已注册的 `super_large_order_intensity`（5 日累计超大单）可能有重叠，`dedup` 阶段定量裁决。
- **`mf_extra_large_sell_pressure`**：抄参考库 fund_flow 的 `ext_mf_extra_large_sell_pressure`（方向负）。（双记口径：Σ4buy ≡ Σ4sell ⇒ 四档净额之和恒为 0，见模块 docstring §一.①） ★ **只有「卖」这一侧是新的**：已注册的 `order_size_concentration` 用买卖**合计**的四档毛额结构，`mf_retail_dominance` 是小单（散户）侧，`mf_big_order_ratio` 是大单的**净**额 —— 「超大单的**毛卖出**占全市场成交的比重」此前**没有**因子覆盖。它与「超大单净额」（买卖相抵后的差）不是同一个量：一只股票可以超大单净买入为正、但毛卖出占比同时创高（大资金一边大举卖出、一边更大举买入 = 换手激烈的大资金博弈）。取值域 [0,1]，分母是当日该股的全部成交额。
- **`mf_flow_factor_momentum_20`**：★ **耦合因子**：`deps` 写父因子名 `mf_net_inflow_ratio`（已在 `factors/fundflow.py` 实现），引擎把它排在第二趟。`_delta(x, 20)` = `x(T) − x(T−20)`，本实现用 `ctx.diff(mat, 20)`（同一口径）。★ 与同族别的「净流入」因子的关系：`mf_net_inflow_ratio` 是**水平**（今天净流入多少）、`mf_cumulative_flow_20d` 是**累计**（近 20 天累计多少）、本因子是**水平的变化**（今天相比 20 天前改善了没有）—— 水平高但边际转弱的股票，本因子给出与水平口径**相反**的信号，这正是参考库 thesis 讲的「边际改善」。★ warmup=56：`diff(20)` 需要 20 个交易日的记忆（20×1.8+20 = 56）。★ 父因子未落盘时 `ctx.load_factor` **静默**返回全 NaN（只 warning）→ 首次全量跑时确认 `mf_net_inflow_ratio` 在同一批次里。
- **`mf_flow_price_absorption_20`**：**本文件新造**。（双记口径：Σ4buy ≡ Σ4sell ⇒ 四档净额之和恒为 0，见模块 docstring §一.①） 分子用**大单净额占八列毛额之比**（量纲自约的无尺度流量），分母用 20 日**累计绝对日收益**（价格走了多远，不看方向）。比值高 ⇒ 机构净买入很多但价格几乎没动 = **潜伏式建仓**（吸筹被市场流动性无声吸收）；比值低 ⇒ 流量不大却价格大幅波动 = **薄盘**或**派发**。★ 与同日尺度的 `large_order_timing_signal` 互补：那个是「今天在什么位置买」，本因子是「20 天里买进去了多少、价格让了多少」——一个横截面位置量、一个时序弹性量，窗口与信息都不同。`roll_sum` 用 (T,C) 原语；`ctx.roll_sum` 支持 `min_count`（2026-09-17 已修那处重复定义的坑），此处用显式 `min_count=N//2`。
- **`mf_large_order_avg_price`**：**纯同表因子，不需要任何单位换算**：分子 `large_amt/large_vol` 与分母 `total_amt/total_vol` 都等于「万元/手」= 100 元/股，量纲在相除时自动约掉。★ 返回 `ratio − 1`（真正的「偏离」，与因子释义一致）；参考库返回 ratio 本身，两者截面排名完全相同（全体平移一个常数）。★ 不做参考库的 `.clip(0.5, 2.0)`（契约禁止因子内 winsor）。注意：参考库描述里写「相对**收盘价**」，实现是「相对**全日 VWAP**」——本实现跟实现（VWAP），不跟描述。
- **`mf_large_order_net_5d`**：与 `mf_big_order_ratio`（lg+elg、单日）的区别是**两维都不同**：只取大单档 + 5 日均值。大单（20~100 万/笔）比超大单更连续、更少脉冲，5 日均值进一步滤掉单日噪声。★ 不做参考库的 `.clip(-0.5, 0.5)`（契约禁止因子内 winsor）。`min_count=3` = 参考库 `min_periods=3`。
- **`mf_net_amount_intensity`**：★ 两处偏离：(1) **单位**：net_mf_amount 是**万元**、市值是**元**，跨表相除前 ×1e4 统一到元（参考库两列都是万元所以没写换算，本框架必须显式写，否则整体差 1e4 倍）；(2) 参考库用 `circ_mv`（流通市值），本实现用**总市值** = `ctx.px('close') × ctx.px('total_share')` —— 与同家族的 `elg_net_60d_to_mv` 保持同一市值口径，且两者都不影响截面排名的相对次序（只是分母口径差，rank 相关约 0.99）。市值取**未复权**收盘价 × 当期已披露股本（PIT 安全：股本随披露前向填充，历史值不因未来的送转变化）。数值量级约 1e-4（净额 500 万元 / 市值 50 亿元），引擎会做截面 winsor+rank。
- **`mf_net_amount_mom5_to_mv`**：源自参考库 `net_mf_amount_momentum_5d`。★★ **必须归一化**：参考库原式的单位是**万元且未除以任何规模量** ⇒ 它在横截面上的排序**实质上就是市值排序**（大票的万元流量天然大几个量级），与 `log_mv` rank 相关极高，作为「流量动量」因子完全失效。本实现除以**自算总市值**（`close × total_share`，未复权价 × 当期已披露股本），与同家族的 `mf_net_amount_intensity` / `elg_net_60d_to_mv` 同一市值口径（**不用** `stock_finance.total_mv` —— 日频快照表会被厂商事后重算）。★ 厂商 `net_mf_amount` 是独立序列（§一.③），本因子不与四档净额重复。
- **`mf_net_inflow_5d`**：日净流入率的 5 日**求和**（不是均值）。窗口内停牌 = NaN；`min_count=3` 即参考库的 `min_periods=3`（显式放松，否则 2015 年大面积停牌会把非空率压到 50% 以下）。
- **`mf_net_inflow_ratio`**：★ 用**供应商现成的** net_mf_amount，不是四桶差（四桶差恒为 0，见模块 docstring ①）。实测 net_mf_amount 与「(lg+elg) 净额」的相关只有 0.60~0.67、与「四桶和之差」的相关 −0.001 —— 它是供应商自己的口径，是独立的净额序列。分母 `tot_amt`（八列金额之和，万元）与分子同表同量纲，**比值无需任何换算**；它等于当日成交额的 2 倍（双记），对本因子只是全体同乘一个常数，不影响截面排名。缺失/停牌 -> NaN（`_Flow.g` 用「出现指示」+ `ctx.traded()` 双重保证），分母由 safe_div 保护（成交额为 0 的壳股 -> NaN）。
- **`mf_net_vol_surprise_20`**：★★ **故意偏离参考库**（`mf_net_vol_ma_divergence`）。参考库原式是`net_vol / |MA20(net_vol)| − 1`，有两个静默缺陷：① 分母取**绝对值** ⇒ 净流量长期为**负**的股票（持续净流出）「流出扩大」会被读成「意外**上升**」，符号完全翻转；② 净流量的 20 日均值可以**跨零**，比值在均值附近会爆成 ±1e3 量级。本实现改成标准的 z-型意外度 `(本期 − 20 日均) / 毛量 20 日均`：**有符号、有界、无尺度**（分母用总成交量而不是净流量，永远为正）。★ 为什么用 `vendor_vol`（厂商净量）而不是四档净额之和：模块 docstring §一.③ —— 厂商净额与四档净额是**两个独立序列**（rank 相关仅 0.60~0.67），四档之和恒为 0，厂商列才是真正可用的「净」量。★ 量版与额版**只能选一个**（100 元/手常数 ⇒ 两者 rank 相关 0.99999，见 §一.②）—— 本因子用量版；同族额版已在 `mf_net_amount_intensity` 覆盖。
- **`mf_open_close_divergence_10d`**：★ 名称与内容是错位的：factors.md 自己注明「2026-08-05 描述与实现统一（原描述声称开盘/收盘背离，实现为日频资金流代理）」，本实现照抄那个**已统一**的日频代理，不要按名字去找开盘/收盘数据。★ 两处偏离：(1) 分母的 `+1e-8` 换成 `ctx.safe_div(..., min_abs_den=1e-3)`：trend_5 是「净额/成交额」的量级（约 1e-2），1e-8 的地板挡不住近零值，会让偏离度炸到 1e6；取 1e-3（≈成交额的 0.1%）是真正的「几乎无净流入」边界。(2) 方向标注 `higher_is_better=False`：参考库代码 `rank(div_std)` 把摇摆剧烈者排前，但其 `意义` 明说「偏离频繁放大 = 主力态度反复、方向不确定」——按语义应为低优。数值本身没动，方向标注只影响文档。
- **`mf_order_size_entropy`**：★ 归一化：参考库返回裸熵（取值 [0, ln4]），本项目要求 `[0,1]`（契约第 4 条：四档占比的 Shannon 熵，归一化到 [0,1]），故除以 ln(4)。截面排名与裸熵等价。用**成交量**口径（参考库原文用 `*_vol` / `_total_vol`）；分母 = 四档毛量之和，比值严格是四档占比（Σp ≡ 1），量纲自约。`p + 1e-10` 保留参考库写法：p=0 时该项贡献恰好为 0。
- **`mf_order_size_entropy_chg_5d`**：**本文件新造**：在已注册的 `mf_order_size_entropy`（**水平**）上取 5 日差分。★ 为什么它**不是** `mf_order_size_entropy` 的重复、也不是`order_size_ratio_change` 的重复：熵是四档占比的**非线性**函数（`−Σp ln p`），它的变化量不在「某一档占比的变化」所张成的空间里 ——同样的「大单占比 +2%」，从中档迁移过来与从小单迁移过来，熵的变化完全不同。★ 口径与 `mf_order_size_entropy` **逐字一致**（**成交量**口径、除以 ln4 归一化到 [0,1]、`p+1e-10`），保证差分是同一序列的差分。★ 分母 = 四档毛量之和（Σp ≡ 1，量纲自约，无需单位换算）。
- **`mf_retail_dominance`**：毛额口径（分子分母同为万元），不受双记影响。返回**未取负**的散户占比原值，参考库 `rank(-ratio)` 由 `higher_is_better=False` 表达。★ 与 `mf_small_order_ratio` 的区别是**净额 vs 毛额**：本因子看「散户参与度」（谁在交易），后者看「散户方向」（散户在买还是卖）；实测两者截面排名相关仅 −0.27 左右，是两条不同的信息。对齐参考库的 `ext_mf_small_order_amount_ratio` / `mf_retail_dominance`（同式异名）。
- **`mf_small_order_avg_price_dev`**：抄参考库 `ext_mf_small_order_average_price_deviation`。★ 参考库的**方向标注是负**（偏离越小排越前），所以本因子返回**绝对偏离** `|比值 − 1|`，与参考库排序方向一致；写成 `比值 − 1` 会让「散户大幅折价买入」和「大幅溢价买入」分居两端，语义与参考库不符。★ 为什么**只做小单**版：额版的兄弟 `ext_mf_large_order_avg_price` 已经以 `mf_large_order_avg_price` 之名注册（超大单成交均价水平），本因子是散户侧、且是**相对 VWAP 的偏离**（不是绝对价格水平）。★ 量纲：`sm_amt/sm_vol` 与 `amt/vol` 都是「元/股」，比值自约；停牌 / 无成交由 `safe_div` 给 NaN。
- **`mf_tier_flow_agreement_20`**：**本文件新造**。（双记口径：Σ4buy ≡ Σ4sell ⇒ 四档净额之和恒为 0，见模块 docstring §一.①） 与 `mf_big_mid_net_corr_20` 的分工：那个用**相关系数**（对幅度敏感，被大流量日主导），本因子用**符号一致频率**（只看方向、对幅度免疫）——同样的相关性可以由「每天小幅同向」或「几天大幅同向」产生，两者对「资金结构是否稳定」的含义完全不同。★★ 符号判定**必须**先做有限值掩码再比较：`np.sign(NaN)` 是 NaN 而 `NaN != NaN` 为 True，写成 `sign(a) != sign(b)` 会把缺失日**静默计成「不同向」**（这类 bug 已在 `mf_flow_stability_20d` 的 note 里记录过一次）。⚠ **取值卡片化警告**：20 日均值只有 21 个可能取值，`fea/eval.py` 的 LOWCARD 判据是「每日唯一值中位 ≤ 20」——沙箱自检阶段必看每日截面唯一值数，不足就地砍。
- **`mf_vol_amount_divergence`**：正 = 净流入的「手数占比」大于「金额占比」→ 净买入发生在**低价位**；负 = 金额占比更大 → 净买入发生在**高价位**（拉升式买入）。两个占比都是**同表内的无量纲分数**：分子分母同单位（手/手、万元/万元），所以**不需要任何换算**（跨表才需要）。★ 不做参考库的 `.clip(-0.5, 0.5)`（契约禁止因子内 winsor，引擎统一缩尾）。
- **`mfi_14`**：参考库出处：factors.md `类别 price` / technical_daily.py 的 mfi_14。★ 复权：TP = (hfq_high + hfq_low + hfq_close)/3，方向判定 `diff(tp)` 因此  在除权日不会误判（参考库的 `scale` 折算由价格层统一完成）。★ 成交量语义：`vol` 是**流量**，停牌日是 NaN（价格层明确不补 0）。  但停牌日 tp 也被前向填充 -> tp_chg == 0 -> 正负两个条件都不成立 ->   该日资金流计 0。这是**正确**的：停牌当天确实没有资金流，  `rolling(14).sum()` 的语义就是「窗口内累计净流入」，不是「日均」。★ **分母保护**：负向资金流之和精确为 0（14 日全是上涨）时，  参考库的 `+1e-10` 会给出 1e10 量级的假比值 -> 引擎值域告警；  本文件用 `ctx.safe_div` 给 NaN，MFI 保持 [0,100] 的值域。★ 方向：参考库正向排名（资金流入推动排前），本因子同样 True。
- **`mfx_dc_amount`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_dc_pct_change`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_dc_pressure`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_dc_range`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_dc_turnover_rate`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_dc_volume`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_index_amount`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_index_gap`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_index_pressure`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_index_range`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_index_volume`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_minute_amplitude`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_minute_volume_concentration`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_minute_weighted_pressure`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_tdx_amount`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_tdx_volume`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_ths_gap`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_ths_premium`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`mfx_ths_volume`**：本地候选定义，至少45个有效配对日。只用截至D日行情，供D+1交易。不读取当前板块成分，不直接把所有股票相同的市场序列输出为因子。每日等权中位数对应当日源库覆盖集合，供应商回填/指数集合历史变化的局限仍存在；相关性不代表收益方向。
- **`momentum_10`**：★ 后复权（hfq），不是前复权：`adj` 在参考库里叫 daily_adj.parquet（qfq），照抄会破坏 PIT。停牌日 hfq_close 被前向填充 → 停牌段收益为 0、复牌跳空计入累计收益（口径正确，不要再用 ffill 去「修」）。★ 清洗口径：走 `_ret_k`（= Π(1+ret(1)) − 1），单日 |收益|>60% 的复权脏数据毒化整个窗口；**不用 `ctx.ret(k)`** 是因为它拿「k 日累计 >60%」当脏数据判据，实测 k=250 时误杀 48% 的有效格且全是强势股（详见 `_ret_k` 的 ★ 段）。介于短周期反转与月度动量之间，实测常是三者里最弱的一档。
- **`momentum_120`**：★ 后复权（hfq），不是前复权：`adj` 在参考库里叫 daily_adj.parquet（qfq），照抄会破坏 PIT。停牌日 hfq_close 被前向填充 → 停牌段收益为 0、复牌跳空计入累计收益（口径正确，不要再用 ffill 去「修」）。★ 清洗口径：走 `_ret_k`（= Π(1+ret(1)) − 1），单日 |收益|>60% 的复权脏数据毒化整个窗口；**不用 `ctx.ret(k)`** 是因为它拿「k 日累计 >60%」当脏数据判据，实测 k=250 时误杀 48% 的有效格且全是强势股（详见 `_ret_k` 的 ★ 段）。对应参考库的 alpha_125d / return_126d（125/126 交易日 ≈ 120），窗口对齐半年。
- **`momentum_20`**：★ 后复权（hfq），不是前复权：`adj` 在参考库里叫 daily_adj.parquet（qfq），照抄会破坏 PIT。停牌日 hfq_close 被前向填充 → 停牌段收益为 0、复牌跳空计入累计收益（口径正确，不要再用 ffill 去「修」）。★ 清洗口径：走 `_ret_k`（= Π(1+ret(1)) − 1），单日 |收益|>60% 的复权脏数据毒化整个窗口；**不用 `ctx.ret(k)`** 是因为它拿「k 日累计 >60%」当脏数据判据，实测 k=250 时误杀 48% 的有效格且全是强势股（详见 `_ret_k` 的 ★ 段）。A 股月度动量效应显著；与 reversal_2d / short_term_reversal_5 方向相反，建模时不要同时用（会互相抵消）。
- **`momentum_250`**：★ 后复权（hfq），不是前复权：`adj` 在参考库里叫 daily_adj.parquet（qfq），照抄会破坏 PIT。停牌日 hfq_close 被前向填充 → 停牌段收益为 0、复牌跳空计入累计收益（口径正确，不要再用 ffill 去「修」）。★ 清洗口径：走 `_ret_k`（= Π(1+ret(1)) − 1），单日 |收益|>60% 的复权脏数据毒化整个窗口；**不用 `ctx.ret(k)`** 是因为它拿「k 日累计 >60%」当脏数据判据，实测 k=250 时误杀 48% 的有效格且全是强势股（详见 `_ret_k` 的 ★ 段）。★ warmup 必须 ≥480：250 交易日 ≈ 365 日历天，给少了每年分区的头 100 多天会静默错。
- **`net_margin_ttm`**：参考库 #35 npm_ttm 的公式未指明净利口径，这里统一归母（见文件头口径总纲）。营收用 revenue（营业收入），不是 total_revenue（营业总收入）。净利率可以为负（亏损），是正常的截面读数。地板 100 万元营收：TTM 营收低于此的主板公司等于空壳，比率无意义。
- **`new_high_60_event`**：`new_low_60_event` 的镜像（把 low/min 换成 high/max），偏离与理由逐条相同。★ 与 event2.py 的 `new_high_frequency_60` **互补不重复**：那个是「60 日内创新高的**天数占比**」（频率，无时间结构），本因子是「新高事件的**衰减加权和**」（越近期的新高权重越大，多次新高累积）。两者在**持续创新高**的股票上相关，但本因子对「最近才突破」的股票给更高分。窗口/半衰期同 `new_low_60_event`（N=10, H=5）。★ 实测零值占比 82.35%、|value|max = 5.7938（同样是 10 天连续新高）、非空率 100.00%。与 `new_low_60_event` 的截面秩相关 −0.238（互为镜像，符号相反是预期行为）。
- **`new_high_frequency_60`**：★ 复权基座必须是**后复权** `ctx.hfq("close")`：参考库用的 daily_adj.parquet 是前复权（历史值会随未来分红重算，PIT 红线）。后复权锚定序列起点，除权日不产生假新高。停牌日 hfq_close 被前向填充（= 上一日价），除非整窗横盘否则不会误判为新高。`min_count=30` 对齐参考库 min_periods=30（窗口内至少半年数据）。值域 [0,1]；次新股上市满 30 个交易日后才有值 → 上市初期为 NaN（正常）。
- **`new_low_60_event`**：★ 与参考库的三处偏离：① 参考库用**复权收盘价**判定新高/新低（`adj.eq(rolling(60).min())`），本因子按任务书用**当日最低价**（`hfq(low) == 60 日 hfq(low) 的最低`）——「新低」在技术分析里指**盘中**创出的低点，用 low 更贴定义；副作用是触发更频繁（盘中破位后收回也算），故零值占比反而更低（见下）。② 复权基座必须是**后复权** `ctx.hfq`：参考库的 `daily_adj.parquet` 是前复权（历史值随未来分红重算，违反 PIT 红线）。后复权锚定序列起点，除权日不产生假新低。③ **显式挡停牌日**（`ctx.traded()`）：`hfq(low)` 在停牌日被前向填充，若停牌前一天正好创了新低，停牌期间每一天都会重复触发一次事件、把 10 日衰减窗口灌满（模块 docstring 三.1）。参考库没有这个问题是因为它对每只股票独立滚动、且没有停牌行的概念。★ 窗口/半衰期：参考库 `event_decay(half_life=5)` 是**无限记忆**（衰减到下一次事件为止），本实现必须有限窗口，取 N=10 = 2 个半衰期（尾部权重 0.25，截断误差 ≤ 25%，且事件在本窗口内多次触发会累积）。取 half_life=5 而不是 _5 系列的 3：60 日新高/新低是**结构性**突破，信息比单日涨跌停持久（参考库对这两个事件也用 5）。`min_count=30` 对齐参考库 `min_periods=30`（次新股上市满 30 个交易日后才有值）。★ 零膨胀：**实测零值占比 71.31%**（8 个因子里最低）—— 因为 10 日窗口里只要盘中破过一次 60 日低点就有值，震荡市里这很常见；|value|max = 5.7938 = 10 天连续新低的完整和 ∑_{k=0}^{9}0.5^(k/5)（说明确实有股票连续 10 天创 60 日新低，长尾是真实的）。★ 该因子与 8 个兄弟因子的截面秩相关最高只有 −0.238（`new_high_60_event`），与涨跌停/极端波动的相关性都 < 0.15 —— **信息独立度高**，建议下游重点看。
- **`np_to_deferred_tax_yoy`**：参考库 `因子库.md` 5、Quality #28。★ 同族偏离（单季→TTM），见 `np_to_inventory_yoy` 的 note。**为什么递延所得税资产这一支有独立信息**：递延所得税资产主要是**可抵扣暂时性差异与可结转亏损**的累积 —— 它是「税务当局尚未认可的会计利润」最干净的单科目代理。企业只有在**预期未来能盈利**时才会确认这笔资产（否则要计提减值），所以它的相对规模变化携带了管理层对自身盈利前景的判断。此前**零个因子**用过这个字段。
- **`np_to_fixed_assets_yoy`**：参考库 `因子库.md` 5、Quality #41。★ 与 `np_to_inventory_yoy` 同族、同偏离（单季→TTM），两处**共用同一个地板（1e6 元）**，保证两者的量级可比。**为什么固定资产这一支值得单独发**：重资产行业的产能利用率变化是盈利周期最直接的度量 —— 单位固定资产创利上升 = 产能被更充分利用（或刚做完减值、分母变干净）。与已删的 `fixed_asset_turnover`（收入/固定资产的**水平**）不同：那个是效率水平，本因子是**效率的同比变化**（分子换成净利、取同比）。
- **`np_to_inventory_yoy`**：参考库 `因子库.md` 5、Quality #8（per-unit 比率同比族）。★ **刻意偏离**：参考库分子用**单季**净利（`NetProfit_Q`），本项目的 `fea/deriv.py` **只暴露 TTM**（没有单季访问器，见 `factors/DEVELOPING.md` §3.2 的字段表），故用 `n_income_attr_p_TTM`。语义从「单季的存货创利效率」变成「**滚动一年的**存货创利效率」，少了季节性、也更平滑 —— 在单年 IC 检验里这**是好事**。★ 分母 `inventories` **不在** `POSITIVE_ONLY` 里（`fea/deriv.py`）——银行/券商没有存货，该字段是精确 0 或 NaN ⇒ 由 `safe_div` 的 1e6 地板给 NaN，**这正是期望行为**（金融股不该有这个因子）。★ 为什么这是 growth 而不是 quality：它测的是**效率的变化率**，而 `inventory_turnover`（在册）测的是**水平**。
- **`np_to_opex_yoy`**：参考库 `因子库.md` 5、Quality #27 的 per-unit 族。★★ **刻意偏离：分母不含 `fin_exp`（财务费用）**（这是与前一轮作者的明确决定一致）。参考库的「三费」= 销售 + 管理 + **财务**费用，但本项目实测`fin_exp` 对 **20%~34% 的公司为负**（利息净收入大于利息支出），含它会让分母**跨零**⇒ 比值的符号静默翻转（`quality.py` 的排除清单里写明了这一条）。本实现改用 **销售 + 管理 + 研发**（三项恒为非负），语义从「三费」变成「**经营性费用**」（研发本就该算进经营费用，参考库把研发另计是它那一版的口径）。★ 与 `opm_ttm` / `opm_npm_spread` 的关系：那两个是**收入为分母**的利润率（测定价能力），本因子是**费用为分母**的产出率（测费用效率），分母的物理量不同、且本因子取同比。
- **`np_to_salary_yoy`**：参考库 `因子库.md` 5、Quality #42。★ **本因子无需任何偏离**：参考库原文就是 `NetProfit_TTM / StaffBehalfPaid_TTM`（**TTM/TTM**），与 `np_to_inventory_yoy` / `np_to_fixed_assets_yoy`那两条「单季→TTM」的偏离不同。**经济含义**：每元薪酬产出多少利润 = **人力投入的运营杠杆**。这个比率上升可以由两头驱动：收入增长摊薄了固定人力成本（经营杠杆释放），或裁员降本。两者的后续走势完全不同，但作为「效率改善」的信号方向一致。★ `c_paid_to_for_empl` 由现金流量表提供，实测 2012 起 100% 非零。
- **`ocf_to_profit`**：★ 分母用**带符号**的归母净利润并设 100 万元地板：净利为负的公司得到**负值**（经营现金流覆盖不了亏损），符合语义。**刻意不用 |NetProfit|** —— 那会把「亏损但现金流为正」的公司排到截面最顶端（方向完全反了）；同文件的 cash_profit_ratio 用 |NP| 是因为它构造的是「超额现金流的相对量」，两者的分母保护动机不同。|NP| < 100 万元视为盈亏平衡、比率无意义 → NaN。
- **`ocf_to_revenue`**：参考库未单列（其 ind 表里的 ocf_to_or 是**累计 YTD** 口径，被契约 §5 禁用，故这里用 ctx.ttm 从现金流量表重算）。**银行业例外**：经营现金流被存款/同业资金进出主导，该比值对银行没有「收入含金量」的含义 —— 银行会散布到截面两端，下游按行业中性化时会自动处理。
- **`one_word_limit_down_freq_20`**：★ 与 event2.py 的 `one_word_limit_up_freq_20` **镜像**（那边是一字**涨停**），唯一的结构差异是跌停判定：那边用 `pct_chg >= 9.8`，本因子用 **LL 优先（`limit == 'D'`）+ 价格近似回退**（与同文件 `limit_down_event_5` 共用`_limit_grid`），因为一字跌停的样本比一字涨停少一个量级，供应商标记带来的判定差更容易影响分布。`high == low` 是一字板的**定义**（全天只成交在一个价位），除权日不产生假值（high/low 是同日同尺度量）。停牌日 `pct_chg` 为 NaN → 该日按「无观测」记 NaN（不参与均值），故 `min_count=5`（对齐参考库 min_periods=5，与 event2 的同款因子一致）。值域严格 [0,1]；**零膨胀（固有）**：从没打过一字跌停的股票恒为 0 —— 这是正确的 0，不是缺失（event2 对一字涨停同款因子给了同样的结论）。**实测**：零值占比 94.31%、|value|max = 0.30（20 天里 6 个一字跌停）、非空率 100.00% —— 刚好压在「>95% 说明口径太稀」的红线之内。注意 20 日窗口内有效观测不足 5 天时是 NaN（次新股/长期停牌），沙箱实测有 167 行 NaN（518,591 / 518,758 行）。与 `consecutive_limit_down` 的区别：那个数「连续几个跌停」（不要求一字），本因子数「跌停里有多少是一字」（流动性冻结的指纹）。
- **`open5_amt_log`**：★ 执行/容量口径，不是 alpha 声称。与 share 版的区别：share 是**相对**结构（这只票自己开盘占全天的比重），log 版是**绝对**池子大小 —— 执行层要估「这笔单子最多下多少钱」用的是后者（`单笔上限 ≈ 开盘 5 分钟成交额 × 参与率`），而 share 版回答的是「同规模下谁的开盘更厚」。两者不可互相替代。取对数是因为原始金额跨 4~5 个数量级，截面 rank 虽然能吃掉量级、但下游若要做线性回归或分层，对数尺度更稳。停牌 / 未落格 -> NaN（不填 0；填 0 的 log 是 −inf）。
- **`open5_amt_share`**：★ 执行/容量口径：高 = 开盘时段流动性池厚，按开盘价成交的冲击更小、可容纳的资金更多。**不是**收益方向的声称。口径见 fea/open5.py：首 5 分钟 = 当日第一根 bar（厂商时间戳 09:35，区间右端）。实测全池中位 ≈ 8.1%（2026Q1）、且随年份单调上行（2010 约 2.6% → 2025 约 8.0%）——做时序比较请用本文件的 _pct20 版本，或只依赖逐日截面 rank。另：项目原有执行层只用「信号日全日成交额的 1%」限容量（V63/analysis.py:272-273），本因子是给它换成**开盘时段**口径的原料。
- **`open5_amt_share_pct20`**：★ 执行/容量口径，不是 alpha 声称。为什么要这一版：占比本身有**长期漂移**（2010 中位 2.6% → 2025 中位 8.0%，见 fea/open5.py 与 README 的交付记录），直接跨年比较会把「全市场都在变」误读成「这只票变了」。时序分位把每只票自己的 20 日历史当基准，漂移与个股规模一起约掉。`ctx.roll_rank` 是 WorldQuant 的 Ts_Rank（窗口最后一个值在窗口内的百分位，返回 [0,1]），min_count=10 = 20//2，与参考库 `_roll_sum(x,20,10)` 的 `min_periods` 约定一致。★ 与 open5_amt_share 高度相关（同一分子分母，只差一次时序排名）——下游若做冗余筛除，这两个应当**视作一簇**。
- **`opm_npm_spread`**：**本文件新造**。★ 为什么用一个**价差**而不是两个水平：`营业利润率` 与 `净利率` 两个水平各自都是慢变量，但它们的**差**在截面上的离散度大得多、时序也更快 ——差刻画的是「营业利润在非经常性损益、税、少数股东损益这一路上损耗了多少」，而这一路的构成**每年都在变**（减值、投资收益、税率优惠），所以价差不是慢变量的线性重标定。被删的 `gross_margin_change` / `net_margin_change` 是**时序差分**（还是慢变量），本因子是**同日截面内的两个水平之差**，机制不同。方向取负：损耗越大，盈利质量越差。
- **`order_size_concentration`**：逐字复刻参考库（`fund_flow_deep.py`），只把 `cross_sectional_rank` 交给引擎。★ **符号是必需的，不是装饰**：实测「无符号 HHI」与 `mf_order_concentration`（大单毛额占比，判定当时在册）的截面 rank 相关 **−0.940** —— 四档 HHI 与「大单占比」在数学上几乎互为镜像（HHI 高 ⟺ 某一档独大 ⟺ 四档越不平均），去掉符号就是又一个重复因子。乘上 `sign(smart_direction)` 后与 `mf_big_order_ratio` 相关降到 **+0.650**、与 `mf_smart_dumb_divergence` **+0.623**（2026 年逐日截面 rank，上游表直接复算）——这正是参考库 thesis 想做的「区分两种高集中情形」。取值域：HHI ∈ [0.25, 1]（四档的下界），乘符号后 ∈ [−1,−0.25] ∪ [0.25,1]，以 0 为界分两侧；`sign(0)=0` 的格子（净额恰好为 0，测度零）给出 0，与参考库一致。分子分母同为**万元**（毛额口径，不受双记影响），比值无量纲、无需换算。
- **`order_size_ratio_change`**：逐字复刻参考库（`fund_flow_deep.py`）。二阶变化抓机构参与的**拐点**（占比从 10% 升到 20% = 刚介入；已在 40% 高位 = 可能出货），与同族其它因子互补：实测与 `mf_order_concentration`（占比**水平**）相关 +0.308、与 `big_vs_small_divergence_5d`（净额口径的 5 日变化）相关 **+0.086** —— 「毛额占比的变化」与「净额方向的变化」是两件事，不冗余。毛额口径本身不受双记影响（Σ4 毛额 = 八列总额）。★ 与 `mf_tier_net_spread_20` 的区别：后者是四档净额占比的 20 日极差（分歧度），本因子是单边占比的 5 日变化（参与度），实测相关仅 +0.16。
- **`outside_bar_count_20`**：参考库 Class1 `outside_bar_count_20`（structure_patterns.py）用 `_roll_sum(outside, 20, 5)` （计数）→ 本实现按任务书做成占比（`roll_mean(..., 20, 10)`），满窗时 rank 等价，且对停牌天然免疫（同 `inside_bar_count_20`）。★ 判据是**收盘价**与昨日全天区间比（不是今日高低点吞没昨日高低点）——参考库的 `outside` 就是 `(adj.gt(prev_high) | adj.lt(prev_low))`，本实现照抄。`gt/lt` 是**严格**比较，相等不算（照抄）。★ 掩码 `traded(T) & traded(T-1)`。值域 [0,1]。
- **`overnight_gap_vol_20`**：参考库 Class1 `overnight_gap_vol_20`（price.py）返回 `cross_sectional_rank(-gap_vol)`。契约 §2.3 不许在因子里 rank，故本因子返回**原始值 `+gap_vol`** 并置 `higher_is_better=False`（与 `factors/intraday.py: idt_overnight_gap` 同样的处置）。★ 与同族 `overnight_std_5d` 的差别只有窗口（20 vs 5）与参考库的 min_periods（10 vs 1，本实现统一 N//2 → 10 vs 2）；两者高度相关但窗口不同，不是重复因子。★ 分母的 `+1e-8` 由 `safe_div(min_abs_den=1e-8)` 承担（语义相同：只挡 0 价脏行）。ddof=1 与 pandas 对齐（实测 1.4e-17）。隔夜腿掩码 `traded(T) & traded(T-1)`。
- **`overnight_intraday_ratio_20d`**：参考库 Class1 `overnight_intraday_ratio_20d`（fac_new_daily.py）逐字为 `oma / (ima.abs() + 1e-6)`，`roll` 默认 min_periods=1 → 本实现两条腿都 min_count=10。★ 分母是 **|日内均值的绝对值|**（不是 |日内| 的均值）—— 这是参考库的写法，本实现照抄：它会在大样本上接近 0，所以比值可以很大（不是有界量）。保留 `+1e-6` 地板（照抄参考库），**没有**把分母换成「接近 0 就 NaN」——只加了一层 `safe_div(min_abs_den=0.0)` 挡精确 0，口径不变。★ 实测（2026 全年 170 个交易日、1,553 万行）：|value| 最大值 1.005e5，全样本只有 12 个格子 > 1e5、0 个 > 1e8 —— 尾部确实很重（分母小的时候），但离体检红线远；这种重尾交给下游 winsor/rank（契约 §1 不许在因子里 winsor）。★ 两条腿同用 `traded(T) & traded(T-1)` 掩码（隔夜腿需要 T-1，日内腿只需 T，取并集让「隔夜 vs 日内」的对比落在同一批交易日上）。
- **`overnight_ma_20d`**：参考库 Class1 `overnight_ma_20d`（fac_new_daily.py），口径同 `overnight_ma_5d`：min_periods=1 → 本实现 min_count=10（N//2）；隔夜腿掩码 `traded(T) & traded(T-1)`。
- **`overnight_ma_5d`**：参考库 Class1 `overnight_ma_5d`（fac_new_daily.py）的 `roll()` 默认 **min_periods=1** —— 那等于「窗口里只有 1 天有数据」也产出「5 日均值」，对长期停牌股是纯噪声。本实现 min_count=2（= N//2，模块 docstring 第 5 条）。★ 隔夜腿掩码 `traded(T) & traded(T-1)`（模块 docstring 第 1 条）。停牌日 NaN 由 min_count 跳过，**不补 0**：补 0 等于把「没交易」记成「隔夜涨跌为 0」。
- **`overnight_ma_60d`**：参考库 Class1 `overnight_ma_60d`（fac_new_daily.py），口径同 `overnight_ma_5d`：min_periods=1 → 本实现 min_count=30（N//2）；隔夜腿掩码 `traded(T) & traded(T-1)`。★ 60 个交易日里撞上停牌的累积概率很高，min_count=30 保证「至少一半有行情」才出值。
- **`overnight_sign_consistency_20d`**：参考库 Class1 `overnight_sign_consistency_20d`（fac_new_daily.py）逐字为 `df['overnight_pos'] = (df['overnight']>0).astype(float)` + `roll(...,20,'mean')`，`roll` 默认 min_periods=1 → 本实现 min_count=10（N//2）。★ `overnight == 0` 精确相等时**不算**正方向（参考库用严格 `> 0`，本实现照抄）。★ 判据不可得（停牌 / 前一日停牌）给 NaN 而不是 0：给 0 会让停牌多的股票被系统性拉向「方向不一致」，而这个偏差与停牌频率相关，不是随机噪声（第 4 条）。值域 [0,1]。
- **`overnight_skewness_20d`**：参考库 Class1 `overnight_skewness_20d`（price_deep.py）返回 `cross_sectional_rank(-skew.abs())`。契约 §2.3 不许在因子里 rank，所以本因子返回**原始值 `-|skew|`**（rank 由引擎统一做），方向语义不变：值越大（越接近 0）= 隔夜分布越对称 = 信息冲击越不极端。★ 偏度做 `clip(-5, 5)` 保留（参考库原样），避免 3 阶矩在退化窗口里爆掉。★ **必须用本文件的 `_roll_skew_eff`**：`ctx.roll_skew` 在 `min_count < n` 时三阶矩分母固定取 n，实测偏差高达 0.2008（模块 docstring 第 6 条）；本实现按有效值个数做矩，与 pandas `rolling(20, min_periods=10).skew()` 一致到 3.3e-15。★ 隔夜腿掩码 `traded(T) & traded(T-1)`。窗口内有效值 < 3 时三阶矩无定义 → NaN。
- **`overnight_std_5d`**：参考库 Class1 `overnight_std_5d`（fac_short_term.py）走 `roll(df,'overnight',5,'std')`，pandas `.rolling().std()` 是 **ddof=1**；`mathx.roll_std` 默认 ddof=0，差一个随窗口内有效天数变化的 `sqrt(k/(k-1))`（不是常数，会改变截面排序），故本因子显式 `ddof=1`。实测与 pandas `rolling(5,min_periods=2).std(ddof=1)` 一致到 **1.4e-17**（模块 docstring 第 6 条）。min_periods=1 → 本实现 min_count=2（N//2）；隔夜腿掩码 `traded(T) & traded(T-1)`。★ 方向取负（`higher_is_better=False`）：粒度同族的 `overnight_gap_vol_20` 在参考库里就是 `rank(-gap_vol)`（高波动排后）。
- **`panic_selling_ratio_60`**：逐字抄自 Class1 risk / panic_selling_ratio_60。嵌套窗口：20 日均量 + 60 日统计 = 80 个交易日，故 warmup=80×1.8+20≈164。min_count：均量 10（=参考 min_periods）、两个 sum 用 1（参考的 min_periods=1：只要有 ≥1 个放量日就能估计占比，0 个放量日 → 0/0 → NaN）。★ 用 `ctx.px("vol")`（流量，停牌日为 NaN）—— 补 0 会把停牌当成「缩量日」。放量日 = vol > 1.5×20 日均量，停牌日与均量为 NaN 的头部都不算放量日（与 pandas 的 NaN.gt() → False 一致）。
- **`pegh5`**：参考库 §9 Value #2 逐字，两处口径实现说明：① **EPS 序列自算**（白名单无 basic_eps）：EPS_TTM = 归母净利TTM / 当期股本；5 年前的 EPS 用 `lag_ttm(NP, 20)`（20 个**报告期** = 5 年）配同一套股本口径 ——**增速因此等于归母净利 TTM 的 5 年复合增速**（股本是同一个乘数、在比值里相消）。这样做的理由：参考库的 `BasicEPS_Y` 序列不做复权，A 股高送转会把 EPS 名义值砍到 1/10，伪造出 −90% 的「EPS 崩塌」；用净利口径则送转/拆股完全无影响。**代价**：增发摊薄不体现在增速里（下游若要每股口径需另建因子）。② **符号与定义域**：增长率两端都 > 0 才算，且要求 **g5 > 0**（负增长时 PEG 无意义，参考库取 −rank(PEGH5) 会把「负增速 = 负 PEG」顶到「最便宜」的一端，方向完全反了）。分母（g5 × EPS）加 1e-3 地板，防 g5→0+ 炸出 1e11 量级的假值。③ **量纲**：按参考库字面公式，PEGH5 = PE_TTM / g5（g5 为小数），即比常见 PEG（PE / 增速百分数）**大 100 倍**，与同族 `peg_252d` 不同量纲 ——但两者都只出 rank，排序不受影响（见两个 note 的对照）。④ **实测口径代价**：要求 g5 > 0 会丢掉「5 年零增长/负增长」的样本（估值上它们是「贵」的一端，不是「便宜」的一端）——缺失与「成长性差」强相关，下游注意。⑤ 起点 = 实测首个有效日 **2016-01-18**（见 `FIN_START_DEEP20`）：20 个报告期的回看要等到 2015Q4 的报告期才凑得齐，2012~2015 四个整年必然全空。
- **`price_distance_from_52w_low`**：与 price_to_52w_high 一起构成 George-Hwang 效应的两端：贴着一年的低点 = 持续阴跌，远离低点 = 趋势健康。值域 [0, +∞)。★ 同样偏离参考库两处：窗口 252→250 交易日、min_periods=120→要求满 250 日（理由见 price_to_52w_high）。复权基座保证除权日不会造出假新低。
- **`price_to_52w_high`**：George-Hwang 52 周高点效应在 A 股的实现。★ warmup=480（250 交易日窗口）。★ 偏离两处：(1) 参考库窗口是 252 交易日，本实现用 250（52 周 ≈ 250 个交易日，差 2 日对「年内高点」的位置无实质影响），公式串保留参考库原文以利溯源；(2) 参考库用 min_periods=120（上市不足半年也给值），本实现要求**满 250 日**，否则次新股的「52 周高点」其实是上市以来的高点，口径不同。值域 (−1, 0]，越接近 0 = 越贴近年内高点（上方套牢盘最少）。
- **`qf_cash_margin_floor_4q`**：本轮自定义单季度质量候选，原始字段来自上游接口字典，沿用财务版本表的公告时点。字段百分数除以100转比例；同比差为比例差，非同比增长率。lag按报告期且取当时已知版本，非日频位移；四季统计要求连续四个报告期全部有效，std为总体标准差。2026Q2发现52只池内股票五个核心字段同时为0，保守将该报告期的整组值标为缺失；单个字段真实零值及负值保留，非有限值为NaN；不回退到旧报告来掩盖缺失。仅用于当晚生成、次日交易；方向为经济解释，未经收益检验。源表若已覆盖旧版本或追溯修订公告日，无法凭现存快照还原真实历史到达时间。
- **`qf_cash_margin_vol_4q`**：本轮自定义单季度质量候选，原始字段来自上游接口字典，沿用财务版本表的公告时点。字段百分数除以100转比例；同比差为比例差，非同比增长率。lag按报告期且取当时已知版本，非日频位移；四季统计要求连续四个报告期全部有效，std为总体标准差。2026Q2发现52只池内股票五个核心字段同时为0，保守将该报告期的整组值标为缺失；单个字段真实零值及负值保留，非有限值为NaN；不回退到旧报告来掩盖缺失。仅用于当晚生成、次日交易；方向为经济解释，未经收益检验。源表若已覆盖旧版本或追溯修订公告日，无法凭现存快照还原真实历史到达时间。
- **`qf_cash_margin_yoy_change`**：本轮自定义单季度质量候选，原始字段来自上游接口字典，沿用财务版本表的公告时点。字段百分数除以100转比例；同比差为比例差，非同比增长率。lag按报告期且取当时已知版本，非日频位移；四季统计要求连续四个报告期全部有效，std为总体标准差。2026Q2发现52只池内股票五个核心字段同时为0，保守将该报告期的整组值标为缺失；单个字段真实零值及负值保留，非有限值为NaN；不回退到旧报告来掩盖缺失。仅用于当晚生成、次日交易；方向为经济解释，未经收益检验。源表若已覆盖旧版本或追溯修订公告日，无法凭现存快照还原真实历史到达时间。
- **`qf_core_roe_floor_4q`**：本轮自定义单季度质量候选，原始字段来自上游接口字典，沿用财务版本表的公告时点。字段百分数除以100转比例；同比差为比例差，非同比增长率。lag按报告期且取当时已知版本，非日频位移；四季统计要求连续四个报告期全部有效，std为总体标准差。2026Q2发现52只池内股票五个核心字段同时为0，保守将该报告期的整组值标为缺失；单个字段真实零值及负值保留，非有限值为NaN；不回退到旧报告来掩盖缺失。仅用于当晚生成、次日交易；方向为经济解释，未经收益检验。源表若已覆盖旧版本或追溯修订公告日，无法凭现存快照还原真实历史到达时间。
- **`qf_core_roe_vol_4q`**：本轮自定义单季度质量候选，原始字段来自上游接口字典，沿用财务版本表的公告时点。字段百分数除以100转比例；同比差为比例差，非同比增长率。lag按报告期且取当时已知版本，非日频位移；四季统计要求连续四个报告期全部有效，std为总体标准差。2026Q2发现52只池内股票五个核心字段同时为0，保守将该报告期的整组值标为缺失；单个字段真实零值及负值保留，非有限值为NaN；不回退到旧报告来掩盖缺失。仅用于当晚生成、次日交易；方向为经济解释，未经收益检验。源表若已覆盖旧版本或追溯修订公告日，无法凭现存快照还原真实历史到达时间。
- **`qf_core_roe_yoy_change`**：本轮自定义单季度质量候选，原始字段来自上游接口字典，沿用财务版本表的公告时点。字段百分数除以100转比例；同比差为比例差，非同比增长率。lag按报告期且取当时已知版本，非日频位移；四季统计要求连续四个报告期全部有效，std为总体标准差。2026Q2发现52只池内股票五个核心字段同时为0，保守将该报告期的整组值标为缺失；单个字段真实零值及负值保留，非有限值为NaN；不回退到旧报告来掩盖缺失。仅用于当晚生成、次日交易；方向为经济解释，未经收益检验。源表若已覆盖旧版本或追溯修订公告日，无法凭现存快照还原真实历史到达时间。
- **`qf_noncore_roe_gap`**：本轮自定义单季度质量候选，原始字段来自上游接口字典，沿用财务版本表的公告时点。字段百分数除以100转比例；同比差为比例差，非同比增长率。lag按报告期且取当时已知版本，非日频位移；四季统计要求连续四个报告期全部有效，std为总体标准差。2026Q2发现52只池内股票五个核心字段同时为0，保守将该报告期的整组值标为缺失；单个字段真实零值及负值保留，非有限值为NaN；不回退到旧报告来掩盖缺失。仅用于当晚生成、次日交易；方向为经济解释，未经收益检验。源表若已覆盖旧版本或追溯修订公告日，无法凭现存快照还原真实历史到达时间。
- **`qf_roa_yoy_change`**：本轮自定义单季度质量候选，原始字段来自上游接口字典，沿用财务版本表的公告时点。字段百分数除以100转比例；同比差为比例差，非同比增长率。lag按报告期且取当时已知版本，非日频位移；四季统计要求连续四个报告期全部有效，std为总体标准差。2026Q2发现52只池内股票五个核心字段同时为0，保守将该报告期的整组值标为缺失；单个字段真实零值及负值保留，非有限值为NaN；不回退到旧报告来掩盖缺失。仅用于当晚生成、次日交易；方向为经济解释，未经收益检验。源表若已覆盖旧版本或追溯修订公告日，无法凭现存快照还原真实历史到达时间。
- **`qf_roe_yoy_change`**：本轮自定义单季度质量候选，原始字段来自上游接口字典，沿用财务版本表的公告时点。字段百分数除以100转比例；同比差为比例差，非同比增长率。lag按报告期且取当时已知版本，非日频位移；四季统计要求连续四个报告期全部有效，std为总体标准差。2026Q2发现52只池内股票五个核心字段同时为0，保守将该报告期的整组值标为缺失；单个字段真实零值及负值保留，非有限值为NaN；不回退到旧报告来掩盖缺失。仅用于当晚生成、次日交易；方向为经济解释，未经收益检验。源表若已覆盖旧版本或追溯修订公告日，无法凭现存快照还原真实历史到达时间。
- **`qf_sales_growth_accel`**：本轮自定义单季度质量候选，原始字段来自上游接口字典，沿用财务版本表的公告时点。字段百分数除以100转比例；同比差为比例差，非同比增长率。lag按报告期且取当时已知版本，非日频位移；四季统计要求连续四个报告期全部有效，std为总体标准差。2026Q2发现52只池内股票五个核心字段同时为0，保守将该报告期的整组值标为缺失；单个字段真实零值及负值保留，非有限值为NaN；不回退到旧报告来掩盖缺失。仅用于当晚生成、次日交易；方向为经济解释，未经收益检验。源表若已覆盖旧版本或追溯修订公告日，无法凭现存快照还原真实历史到达时间。
- **`qf_sales_growth_floor_4q`**：本轮自定义单季度质量候选，原始字段来自上游接口字典，沿用财务版本表的公告时点。字段百分数除以100转比例；同比差为比例差，非同比增长率。lag按报告期且取当时已知版本，非日频位移；四季统计要求连续四个报告期全部有效，std为总体标准差。2026Q2发现52只池内股票五个核心字段同时为0，保守将该报告期的整组值标为缺失；单个字段真实零值及负值保留，非有限值为NaN；不回退到旧报告来掩盖缺失。仅用于当晚生成、次日交易；方向为经济解释，未经收益检验。源表若已覆盖旧版本或追溯修订公告日，无法凭现存快照还原真实历史到达时间。
- **`qf_sales_growth_vol_4q`**：本轮自定义单季度质量候选，原始字段来自上游接口字典，沿用财务版本表的公告时点。字段百分数除以100转比例；同比差为比例差，非同比增长率。lag按报告期且取当时已知版本，非日频位移；四季统计要求连续四个报告期全部有效，std为总体标准差。2026Q2发现52只池内股票五个核心字段同时为0，保守将该报告期的整组值标为缺失；单个字段真实零值及负值保留，非有限值为NaN；不回退到旧报告来掩盖缺失。仅用于当晚生成、次日交易；方向为经济解释，未经收益检验。源表若已覆盖旧版本或追溯修订公告日，无法凭现存快照还原真实历史到达时间。
- **`quality_composite`**：★ **与参考库的有意偏离**：参考库 §5 Quality #10 是 AQR QMJ 的「**6 项比率直接相加**」，本实现是「**3 项比率截面标准化后等权**」。两条理由：① 直接相加**量纲不可比**（ROE ~0.1、毛利率 ~0.3、(OCF+ICF)/总资产 ~0.05，相加等于给毛利率 3 倍权重）；② 参考库第 6 项 `NetProfit / (OpCashInflow + InvCashInflow)` 的分母可以为负/近零（参考库自己要用 `filter=True` 挡 ±inf），在契约 §3.4「除法一律带保护」下会被 NaN 掉大半样本。★ **合成口径**：每项先 `ctx.cs_zscore(x, mask=ctx.universe)`（**逐交易日的截面**标准化，只用当日主板股票，**绝不使用全样本统计量** —— 那会引入未来信息），再等权平均。三项是：**盈利能力**（ROE_TTM）、**定价能力**（毛利率 TTM）、**现金创造**（经营现金流 TTM / 期末总资产）—— 分别对应利润表、利润表的毛利率结构、现金流量表，三张表各取一个，刻意不重叠。★ **缺失处理**：允许**至少 2 项有效**（`nanmean`），此时用有效项的均值。理由：银行/券商没有可比的 `oper_cost`（毛利率恒为 NaN），若要求 3 项全有，整个金融板块会被排除；要求 2 项则它们由 ROE + 现金流质量代表。只有 1 项有效的格子返回 NaN（**不给单因子冒充复合因子**）。★ **与既有因子的关系（重要）**：本因子是三个已落盘因子（`roe_ttm`、`gpm_ttm`、`ocf_to_asset` 的等价物）的线性组合，**与 `roe_ttm` 的相关性最高**（ROE 的截面分散度最大）。它的价值是「一次拿到一个已经中性化的质量分」，下游做因子筛选时**不应**再同时保留全部成分因子（共线）。
- **`rd_intensity`**：★ 参考库未收录（研发投入维度）。**起点 2019-05-01（偏离 start=None）**：上游 `rd_exp` 在 2012~2017 的年报行里 **97%~100% 是精确的 0.0**（实测：2012-2014 正值占比 0.03%，2015 起 3%，2017 才 25%），因为 2018 年新准则（财会[2018]15号）之前研发费用普遍混在管理费用里、**不单独披露** —— 那是「没披露」而不是「没研发」。2018 年报起披露率跳到 87%~88%。不设起点的话 2012-2017 的因子值是「一大片 0 + 零星正值」，虽然不是常数，但截面完全没有区分度。取 2019-05-01 = 全部主板公司 2018 年报都已公告，TTM 窗口口径统一。口径说明：用**费用化**的 `rd_exp`；`r_and_d`（资产负债表时点字段）实测99% 是 0（只有资本化的开发支出），不值得用。
- **`rel_mom_ind_10d`**：逐字抄参考库 cat-sector-c1 的同名因子定义（`个股 k 日收益 − 行业等权 k 日收益`）。★★ **行业归属是本项目自建的动态行业**（见模块 docstring §二），不是参考库那份无日期的快照成分股表 —— 参考库自己就是因为快照非 PIT 才禁用了 6 个 sector 因子。★ 与参考库的实现差异：行业收益用**指数点位**（市值加权），参考库用**等权**行业均值；理由见模块 docstring §三.3。经济含义：剥离板块 β 之后的个股超额 —— 同一行业里跑赢同伴的股票延续性更强，绝对动量受行业轮动干扰大。
- **`rel_mom_ind_20d`**：逐字抄参考库 cat-sector-c1 的同名因子定义（`个股 k 日收益 − 行业等权 k 日收益`）。★★ **行业归属是本项目自建的动态行业**（见模块 docstring §二），不是参考库那份无日期的快照成分股表 —— 参考库自己就是因为快照非 PIT 才禁用了 6 个 sector 因子。★ 与参考库的实现差异：行业收益用**指数点位**（市值加权），参考库用**等权**行业均值；理由见模块 docstring §三.3。经济含义：剥离板块 β 之后的个股超额 —— 同一行业里跑赢同伴的股票延续性更强，绝对动量受行业轮动干扰大。
- **`rel_mom_ind_250d`**：逐字抄参考库 cat-sector-c1 的同名因子定义（`个股 k 日收益 − 行业等权 k 日收益`）。★★ **行业归属是本项目自建的动态行业**（见模块 docstring §二），不是参考库那份无日期的快照成分股表 —— 参考库自己就是因为快照非 PIT 才禁用了 6 个 sector 因子。★ 与参考库的实现差异：行业收益用**指数点位**（市值加权），参考库用**等权**行业均值；理由见模块 docstring §三.3。经济含义：剥离板块 β 之后的个股超额 —— 同一行业里跑赢同伴的股票延续性更强，绝对动量受行业轮动干扰大。
- **`rel_mom_ind_3d`**：逐字抄参考库 cat-sector-c1 的同名因子定义（`个股 k 日收益 − 行业等权 k 日收益`）。★★ **行业归属是本项目自建的动态行业**（见模块 docstring §二），不是参考库那份无日期的快照成分股表 —— 参考库自己就是因为快照非 PIT 才禁用了 6 个 sector 因子。★ 与参考库的实现差异：行业收益用**指数点位**（市值加权），参考库用**等权**行业均值；理由见模块 docstring §三.3。经济含义：剥离板块 β 之后的个股超额 —— 同一行业里跑赢同伴的股票延续性更强，绝对动量受行业轮动干扰大。
- **`rel_mom_ind_5d`**：逐字抄参考库 cat-sector-c1 的同名因子定义（`个股 k 日收益 − 行业等权 k 日收益`）。★★ **行业归属是本项目自建的动态行业**（见模块 docstring §二），不是参考库那份无日期的快照成分股表 —— 参考库自己就是因为快照非 PIT 才禁用了 6 个 sector 因子。★ 与参考库的实现差异：行业收益用**指数点位**（市值加权），参考库用**等权**行业均值；理由见模块 docstring §三.3。经济含义：剥离板块 β 之后的个股超额 —— 同一行业里跑赢同伴的股票延续性更强，绝对动量受行业轮动干扰大。
- **`rel_mom_ind_60d`**：逐字抄参考库 cat-sector-c1 的同名因子定义（`个股 k 日收益 − 行业等权 k 日收益`）。★★ **行业归属是本项目自建的动态行业**（见模块 docstring §二），不是参考库那份无日期的快照成分股表 —— 参考库自己就是因为快照非 PIT 才禁用了 6 个 sector 因子。★ 与参考库的实现差异：行业收益用**指数点位**（市值加权），参考库用**等权**行业均值；理由见模块 docstring §三.3。经济含义：剥离板块 β 之后的个股超额 —— 同一行业里跑赢同伴的股票延续性更强，绝对动量受行业轮动干扰大。
- **`rel_turnover_ind_20d`**：抄参考库 cat-sector-c1 `rel_turnover_ind`（原式是**当日**换手 − 行业均值；本因子取 20 日均值后再做差 —— 单日换手极噪（涨跌停、大额协议转让），20 日均值是参考库自己在 `rel_turnover_ind_ma20` 里用的口径）。★ 数据源 `stock_finance.turnover_rate`（供应商口径），不是自算 `vol/股本` —— 与已注册的 `turnover_f_20`（自由流通换手）的股本口径不同，两者**不是**同一个量的重标定。★ 剔除的是「板块整体交投活跃度」：高换手 = 资金关注但筹码交换剧烈，低换手 = 惜售/锁筹 —— 只有在**同一行业内部**比较才有意义。
- **`rel_vol_ind_20d`**：抄参考库 cat-sector-c1 `rel_vol_ind_20d`（参考库用等权行业内均值，本文件同口径 ——指数点位没有「波动」这一个量，只能用成员等权，见模块 docstring §三.4）。★ 成员均值是 **NaN 感知**的（分母 = 该日行业内有效值的个数，不是行业成员数），否则停牌 / 未上市 / 退市会把行业均值系统性拉低。与已注册的 `vol_120` / `idio_vol_60` 的区别：那两个是**绝对波动**（`idio_vol_60` 是相对沪深300 的残差波动），本因子是**行业内的相对**波动 ——剔除的是板块整体波动水平（某些行业天然波动大）。
- **`ret_autocorr_1d_20`**：直接度量「趋势的稳固程度」，与动量的**水平**正交，是反转强度的标准化度量。★ 偏离：参考库 min_periods=10，本实现要求满 20 个有效观测（缺失毒化）。停牌日 ret(1)=0（价格层前向填充的副作用），会给自相关灌进少量 0 观测 → 轻微向 0 收缩；停牌多的股票解释时要小心。值域 [−1, 1]。★★ 实测坑（2070 格 |ρ|>1、最大 14.36，全部是 002145.SZ / 600193.SH 一类连续停牌股）：退化窗口里 num 与 den 都是浮点噪声，`mathx.roll_corr` 的 `den > 0` 守卫挡不住。故本因子显式做两步**定义域**处理（不是 winsor、不是 rank，只是把「无定义」判成 NaN）：① 窗口内真实成交日 < 10（= 一半）→ NaN；② |ρ| > 1 → NaN。处理后值域严格 [−1,1]（实测 max 0.975），代价是 2015 年丢掉约 5% 的格子（那些格子本来就没有信息）。
- **`ret_efficiency_20`**：★ **偏离参考库的 ret_efficiency_20**：参考库那个 factor 叫『价格效率』，实现是 `rolling_sum(pct_chg, 20) / rolling_sum(amount, 20)`（收益/成交额）。本实现按任务书括号里的『路径效率』做成 Kaufman 效率比率（净位移 / 路径长度）：ER 高 = 单边趋势（路径短、位移大），ER 低 = 反复震荡（路径长、位移小）。选它的理由：对成交额的单位（元/千元）与量纲不敏感、值域天然是 [0,1]、且与 amihud 类流动性因子不重复。要改回参考库口径只需换成 safe_div(roll_sum(ctx.ret(1),20), roll_sum(ctx.amount,20))。值域 [0,1]，恒不缺失。
- **`ret_ind_rel_1d`**：逐字抄参考库 cat-sector-c1 `ret_ind_rel_1d`（唯一一个 1 日窗口的行业相对因子）。★★ 行业归属是自建动态行业（模块 docstring §二），非快照成分股。与 `rel_mom_ind_3d` 的关系：那是 3 日累计，本因子是**当日单日**超额 ——在 1 日预测期上信息衰减最快，短窗口是必要的（参考库把它单独放在`fac_cand_daily.py` 而不是常规族里，也是这个理由）。
- **`ret_kurt_20`**：逐字抄自 Class1 risk / ret_kurt_20。min_count=15 = 参考 min_periods。★ 返回的是**超额**峰度（减 3 之后，正态 ≈ 0），与 pandas `.kurt()` 同口径；不是原始的四阶矩比。★ 数值守卫 |kurt| > 1e4 → NaN：同 ret_skew_20 的退化窗口（m2→0 → m4/m2² 爆炸），实测垃圾值全部 >1e8（含 ±inf），真实值 <1e3，中间同样是空的。
- **`ret_skew_20`**：逐字抄自 Class1 risk / ret_skew_20。min_count=15 = 参考 min_periods。用 `ctx.roll_skew`（与 pandas `.rolling(n).skew()` 同口径：有偏偏度再乘 sqrt(n(n-1))/(n-2) 去偏）。★ 数值守卫 |skew| > 100 → NaN：一字板/长期停牌后价格不动会让 m2→0、m3/m2^1.5 爆炸，实测截面双峰（真实值 <10、垃圾值 >1000、中间 0 个格子）。
- **`revenue_cagr_3y`**：★★ 起点显式写 2014-01-01（不是 default_start）：本因子要 lag_ttm(REV, 12)，上游财报最早只到 2010Q1、TTM 最早算到第 83 期（2010FY）→ lag 12 最早在当期报告期 >= 95（2013Q4）时可达，实测首个有效日 2014-01-22、2014-04 起覆盖稳定。2012/2013 两整年会是全 NaN，而引擎目前在整年全 NaN 时会崩（见文件头 DEEP_LAG_START 的说明）。★★ 符号问题（本家族第二个大坑）：`(x_t/x_{t−k})^(1/k) − 1` 在 x 为负时无定义。本实现要求**两端都 > 0**，否则置 NaN。理由：① 分母为负时比值的符号会反转（−1亿 → +0.5亿 会算出 +2 倍的「增长」）；② 严格 > 0 而不是 ≥ 0：营收近零的壳公司一旦有了一点收入，`(x/0)^(1/3)` 会炸成任意大的值。★ 亏损转盈的样本**置 NaN 而非保留**：CAGR 的定义是「几何平均」，它对符号翻转型样本没有可解释的含义；保留它们（例如取 |分母|）会把「扭亏」这个事件伪装成一个巨大的正增长，在截面上系统性高估困境反转股。代价是 CAGR 因子在亏损股上缺失（截面偏向盈利样本），这是**有意的**。★ 口径：12 个**报告期**（3 年），不是日频 t-756；12 期需要更深的版本表，由引擎的 `years_for_window(lag_years=3)` 覆盖。早期年份（数据起点后的头 3 年）会有一段 NaN，属正常。
- **`reversal_2d`**：参考库 fac_short_term.py 的 reversal_2d **不做取负**（值就是 2 日收益），方向由下游处理 —— 本实现照抄该口径，用 higher_is_better=False 标注方向。★ 与 short_term_reversal_5 不同：那个参考库里显式取了负号。反转因子理论上不需要复权（2 日内除权概率低），这里仍走 `_ret_k`（后复权），一致性优先，且能挡掉复权因子脏行造的假暴跌。
- **`roa_ttm`**：★ 分子用**归母**净利润，与既有 roe_ttm / yoy_net_profit 同口径 —— 这样 ROE = ROA × 权益乘数 的杜邦恒等式在本家族内逐格成立（权益乘数同样用归母权益）。参考库只写 NetProfit，未指明是否含少数股东。分母用**期末**总资产（参考库同）；平均资产口径要取前一期报表，PIT 更脆。银行的总资产收益率天然很低（~0.8%），是行业属性不是异常。
- **`roc_12`**：参考库出处：factors.md `类别 price` / technical_daily.py 的 roc_12。等价于 `ctx.pct_change(hfq_close, 12)`，即 12 个**交易日**的变化率（面板本身是交易日历，位移 = 交易日）。★ 参考库的 `fill_method=None` 是 pandas 的显式声明（不对停牌做填充）；  本框架的价格层对水平量做前向填充，停牌日收益为 0，窗口照常推进。
- **`roe_ttm`**：用期末权益；灵启文档提「平均权益」，平均口径需前一期行，PIT 更脆
- **`roe_ttm_lag63d`**：参考库 §5 Quality #1 `roe_ttm_lag63d`（其 `lag` 参数是**交易日**数）。★ **为什么这里可以、也必须用日频 shift**（与文件头第 4 条不矛盾）：本因子返回的是「63 个交易日前的 ROE **水平量**」，不是用日频位移去算**增长率**。ROE(TTM) 是按 `ann_date` 前向填充的**阶梯函数**（一年只跳 4 次），所以 `shift(63)` 取到的就是「约一个季度前那一版财报算出的 ROE」。★ 用途（下游怎么用）：它与 `roe_ttm` 构成一对 ——**`roe_ttm − roe_ttm_lag63d` 就是「一份新财报带来的 ROE 漂移」**，是个事件式的边际量；直接做季度环比差分在这里是做不到的，因为新的季度值出现的确切日期（ann_date）不定，而 shift(63) 是固定窗口。★ **与 `roe_ttm` 高度相关**（同一条阶梯序列平移 63 个交易日，一年里约一半时间取到的是同一个报告期的值）—— 下游建模时**不要**把它当独立因子与 `roe_ttm` 并列，它的价值在「做差」。★ 口径：与 `roe_ttm` 完全一致（归母净利 TTM / 期末归母权益），多一层 100 万元净资产地板（近零净资产会把 ROE 炸到 ±1e5）。
- **`rsi_14`**：参考库出处：因子库.md「6、Reversal」第 2 条 rsi（同一段公式也出现在 factors.md 的 rsi_spread_6_14 的 `_rsi` 私有函数里，口径一致）。★ 偏离：参考库 rsi_spread_6_14 的 `_rsi` 用 `delta = pct_chg/100`，本文件用 `diff(hfq_close)` —— 两者在正常交易日恒等，但 pct_chg 未经复权因子还原（交易所给的 pct_chg 已在除权日调整过，实际也等价），统一走 hfq 更省心。★ 分母保护：AvgLoss 精确为 0（连续上涨或长期停牌导致涨跌幅全为 0）时 safe_div 给 NaN，理论上该点 RSI=100 —— 宁可缺一格也不给假极值。★ 停牌日 hfq_close 被前向填充 -> 涨跌幅为 0（不是 NaN），RSI 会向中位漂移。★ 方向：参考库 rsi_14_excess 取负向排名（高 RSI = 超买排后），本因子标 higher_is_better=False，与之一致。warmup 给 300 而不是 14×1.8+20=46：Wilder EMA 的种子残差要 ~200 个交易日才衰减到 float32 精度以下，给少了增量与全量的值会在窗口头部不一致。
- **`rsi_spread_6_14`**：参考库出处：factors.md `类别 price` / technical_daily.py 的 rsi_spread_6_14。★ 偏离：参考库分母写 `avg_loss + 1e-10`，本文件用 `ctx.safe_div`（分母为 0 给 NaN）。1e-10 在 RSI 的 0~1 量纲上等价于「分母为 0 时返回 1e10 量级的假值」，契约 §2.3 明令禁止哨兵/巨值，故改成 NaN。★ 复权：参考库用 pct_chg（交易所口径），本文件用 diff(hfq_close)，正常交易日等价。
- **`rsrs_beta_18`**：★★ v2（2026-09-17）：`mathx.roll_cov` 修了「分母用固定 n」的 bug（见该函数注释）——本因子是它的直接下游，历史值整体重算过。参考库出处：factors.md `类别 price` / technical_pattern.py 的 rsrs_beta_18，以及 因子库.md「4、Momentum」第 16 条 `rsrs`（`Slope_t = Beta from OLS(Low_{t-N+1:t} ~ High_{t-N+1:t}), N=18`）。★ 实现：滚动 OLS 斜率 = cov(low, high)/var(high)，用 `ctx.roll_cov` + `ctx.roll_var` 向量化（都是 O(T·C) 的 cumsum 差分），**没有 Python 循环**。★ 偏离（数值）：参考库把 high/low 乘 `scale = adj/close` 折算到复权空间；  本框架的 `ctx.hfq("high"/"low")` 已经是复权价（少一次折算），  再各自除以当日 hfq 收盘价只为一个目的：把数值压到 O(1)。  除以同一个正常数不改变斜率（beta 对分子分母同尺度缩放不变）。★ beta 是**无量纲**的（cov(low,high)/var(high) 的单位是 low/high），  所以不需要额外归一化；但它的绝对值会随「high 与 low 的相对波动」漂移。★ 退化保护：18 日 high 完全不动（长期停牌）-> var == 0 -> NaN。
- **`rsrs_zscore_18`**：★★ v2（2026-09-17）：随 `mathx.roll_cov` 的分母 bug 修复整体重算（z-score 里的 beta 正是该函数的输出）。参考库出处：factors.md `类别 price` / technical_pattern.py 的 rsrs_zscore_18；参数口径以 因子库.md「4、Momentum」第 16 条 `rsrs` 为准 ——`RSRS_t = Z-Score(Slope_{t-M+1:t}), M=200`，即**200 个交易日**。★ **偏离（窗口长度）**：factors.md 的代码注释写「400-day rolling window」，  与 因子库.md 的 M=200（交易日）冲突。两个数字单位不同：  200 个**交易日** ≈ 280 个日历天，仍不等于 400。  本文件以 因子库.md 的显式参数 M=200（交易日）为准，  因为它是带参数名的规范定义，而 factors.md 那句只是行内注释。  如需改为 400 个交易日，只改 `rsrs_zscore_18` 里的 200 与它的 warmup。★ `warmup_days=400`（任务指定）：要覆盖 200 个交易日的 z-score 窗  + 18 日回归窗，400 个日历天 ≈ 274 个交易日，够用且有余量。★ `min_count=100`（= 参考库的 `min_periods=100`，但参考库窗口 400、  本文件窗口 200，所以本文件的放松**相对更严格**）。  为什么必须放松：`high`/`low` 属于价格层的 LEVELS，停牌日**前向填充**，  所以**连续停牌 >= 18 个交易日**的股票，其 18 日窗口内 high 完全不动 ->  `roll_var(high,18) == 0` -> `safe_div` 给 NaN -> beta 缺失。  而 beta 一旦缺一天，200 日 z-score 窗就被 NaN 毒化 **200 个交易日**。  实测：2012~2014 年截面勉强够用（1600~1900），但 2015 年千股停牌后  截面从年初 1596 一路衰减到年末 1305（中位数 1385），  低于契约 §7「日均截面 1500~3300」的下限。给 min_count=100 后回升。★ 不给 min_count 的代价（原实现）：长期停牌股会被整段踢出截面，  而停牌恰恰与「重组/重大事项」强相关 -> 缺失是**非随机**的，  等于在截面上系统性剔除了事件股。这是比放松窗口更坏的性质。
- **`rv_term_structure`**：参考库的 `rv_term_structure_slope` 用的是 5min 已实现波动；本实现用**日频收益的滚动标准差**做同一件事（日频口径更稳、且不受 5min 数据起点限制）。值 >1 = 短期波动放大（不安定期），<1 = 波动回落。⚠️ 与已删除的 `vol_20`/`vol_60` 不同：那两个是**水平**，这个是**比率**（不含水平信息）。
- **`seal_float_strength_20`**：参考事件驱动因子方法的字段扩展。按日散点后取 20 交易日内有效涨停事件均值；没有事件为 NaN，不把空值当作没有封单。至少 1 个事件才有效，因此覆盖率天然稀疏；百分比按供应商原单位保留，负值无效。2018 年实测这些字段已可用；当晚生成供次日使用。
- **`seal_reopen_pressure_20`**：参考事件驱动因子方法的字段扩展。按日散点后取 20 交易日内有效涨停事件均值；没有事件为 NaN，不把空值当作没有封单。至少 1 个事件才有效，因此覆盖率天然稀疏；百分比按供应商原单位保留，负值无效。2018 年实测这些字段已可用；当晚生成供次日使用。
- **`seal_turnover_strength_20`**：参考事件驱动因子方法的字段扩展。按日散点后取 20 交易日内有效涨停事件均值；没有事件为 NaN，不把空值当作没有封单。至少 1 个事件才有效，因此覆盖率天然稀疏；百分比按供应商原单位保留，负值无效。2018 年实测这些字段已可用；当晚生成供次日使用。
- **`share_issuance_yoy`**：**本文件新造**。用 `total_share`（资产负债表时点科目，在 `fea/deriv.py` 的 `POSITIVE_ONLY` 里 ⇒ 恒为正，比值天然安全）。**为什么必须做**：全库此前**没有任何因子**刻画股本的**扩张** ——`log_mv` / `free_share_ratio` 是水平、`bps_yoy` 是**扣除**股本扩张之后的结果，而本因子正是那个被扣掉的东西。两者构成一组完整的分解：`净资产同比 ≈ 每股净资产同比 + 股本同比`。经济含义：股本同比高 ⇒ 增发摊薄（老股东权益被稀释）或高送转；低/负 ⇒ 回购注销（近年才常见）。⚠ **送转是机械事件不是估值信号**：10 送 10 会让股本翻倍但价值不变，本因子会把它记成 +100% 的「扩张」。这是**已知的口径局限**，写在 note 里供下游取舍 —— 若要剔除送转，需要除权事件的复权因子，那属于另一族（`stock_adj_factor_changes`）的范畴。★ `total_share` 是**真实日频序列**（实测每股每年 5~15 个不同取值，变动日与 `adj_factor` 变动日同步），不是一年四次的季报值 ——所以本因子比同族 TTM 比率的活跃度高。方向取负（摊薄越大越差）。
- **`short_balance_ratio_change_20d`**：与 `margin_leverage_change_20d` 完全同口径（同一个分母、同一个 diff(20)、同一天下移），只把分子从融资余额换成**融券**余额，所以两个因子可以直接对比「多头杠杆 vs 空头杠杆」的边际变化。占比为 0 的股票（约 6% 的行 `rqye = 0`）变化也是 0 —— 参考库明确指出这是**有效信息**（空头仓位本来就是 0），不置 NaN。★ 融券余额占流通市值实测只有万分之几（`rqye/rzye` 中位数 0.0021），所以本因子的绝对量级很小，靠截面排名取信息。★ 「多空杠杆变化的对比」这个用法实测是成立的：与 `margin_leverage_change_20d` 的日内 rank 相关中位数 **−0.073**（沙箱 2012~2015），几乎正交 —— 融资盘的加杠杆与融券盘的加仓在截面上是两件独立的事。
- **`short_interest_volatility_20d`**：与 `margin_balance_volatility_20d` 同结构，只把余额换成**融券余量**。实测 2026-09-10 CV p1/p50/p99 = 0 / 0.220 / 1.835 —— 参考库 clip(0, 2) 也在 p99 附近（手工缩尾）。★ 融券余量大量为 0（约 6% 的格子 `rqyl = 0`）：窗口内全 0 时 std 与 mean 都是 0，`safe_div` 会给出 NaN（而不是 0/0 的 inf）——这是对的，「从来没有融券」不构成「空头仓位稳定」的证据。
- **`short_sell_volume_ratio`**：★ 单位已实测核对：`rqmcl`（融券卖出量）与 `ctx.px("vol")` **都是股**，可以直接相除。实测 2026-09-10 中位数 5.2e-5、p99 7.0e-3、约 48% 的两融标的当日融券卖出量为 0 —— 与「A 股融券占成交比重极小」一致。★ 偏离参考库的一句话（口径级）：参考库 note 写「Date=T 使用当时可获得的 T-1 融券卖出量和 T 日总成交量」，即**跨日配对**；本实现按框架 §4 的强制口径，在**同一交易日**网格上算 rqmcl/vol、再把整个网格下移一格 —— 这样 T 日的因子用的是 T-1 的融券卖出量 **和 T-1 的成交量**，分子分母严格同日（跨日配对会把两天的信息混在一起，反而更难解释）。停牌日 `vol` 是 NaN（流量语义）→ 因子 NaN，不会把停牌当成「零做空」。
- **`short_squeeze_risk`**：★ 参考库原式是 `rqyl(股) / rzye(元)`，**量纲混合**（股/元），这里**照抄参考库**保持可比性：它仍然是一个有效的截面排序量，但会带上一点股价水平的倾斜（同样经济含义下，高价股的比值更小）。实测 2026-09-10 该比值 ×100 后 p1/p50/p99 = 0 / 0.0104 / 10.04 —— 参考库 clip(0, 10) 恰好压在 p99 上（再次说明它的 clip 就是手工缩尾）。★ 若下游希望去掉价格水平的影响，等价的「金额口径」是 `rqye / rzye`（两列都是元，实测 ×100 后 p50 = 0.213、p99 = 22.6）—— 本文件未改口径，因为参考库与任务书都写的是 `rqyl / rzye`；要换口径请改 version 并在 FACTORS.md 里登记。
- **`short_term_reversal_5`**：参考库 momentum_rebuilt.py 的 short_term_reversal_5 是最新 5 日动量的**负向**，这里 keep 负号并标 higher_is_better=True（值大 = 超跌更多 = 反转预期更强）。★ 它是 momentum_5 的精确相反数（−1 ×），两者只能留一个。短窗口除权概率低、本可不复权，仍统一走 `_ret_k`（后复权 + 单日脏数据毒化）。
- **`small_order_crowding`**：★★ **偏离参考库的字面公式（必须看）**：参考库返回 `rank(-小单毛量占比)`，实测它与本项目的 `mf_retail_dominance`（小单毛**额**占比）截面 rank 相关 **0.99999** —— 是同一个因子的两种写法，不是「量 vs 额」两条信息：由 双记口径：本表每笔成交同时进买方桶与卖方桶，Σ4buy ≡ Σ4sell（见 fundflow.py docstring ①）。 及 `netA/netV` 与 `totA/totV` 同为一个「100 元/手」的换算比，小单的量占比与额占比在全样本上几乎逐格相等（实测 2026 年 891,494 行 rank 相关 0.99999）。照抄会得到一个与既有因子精确重复的因子。本实现改为**相对自身 20 日中枢的偏离**（拥挤度的**边际**变化，而不是水平）：「今天散户占比比它自己最近 20 天的常态高多少」。理由：(1) 水平量已被 `mf_retail_dominance` + `mf_order_size_entropy` 覆盖；(2) 参考库 thesis 的诉求是「**拥挤**」这个状态量，而自归一化后才是可比的跨股「异常度」（大盘股天然小单占比低，直接比水平等于在比市值）；(3) 沙箱 2026 实测（vs 全部 259 个既有生产因子的逐日截面 rank 相关）：max|corr| 从 0.99999 降到 **−0.431**（`amount_ratio_20`；与 `mf_retail_dominance` 仅 **+0.353**、与 `volume_ratio` −0.348）。★ 不做参考库的 `rank(-small_pct)`（不取负）：方向用 `higher_is_better=False` 表达，`value` 列保持「偏离幅度」的可解释量纲（0 = 与自身中枢持平）。取值域 [−1, +∞)，沙箱 2026 实测中位数 +0.006、|value|max = 18.5（重尾来自20 日中枢极小的格子，靠引擎截面缩尾+rank 处理）。分母 `safe_div(min_abs_den=1e-6)` 挡掉零成交；`min_count=10` = 参考库 min_periods。
- **`sortino_ratio_60`**：逐字抄自 Class1 risk / sortino_ratio_60。★ 偏离任务书的一句话描述：任务书写「60 日**超额**收益均值」，参考库用的是**原始**均收益（不减无风险利率，A 股日频也没有合适的无风险日利率口径），照抄参考库。分母 = 负收益子样本的 60 日 roll_std，min_count=5（必须的放松：60 日里负收益约 28 天）；分子 min_count=30 = 参考 min_periods。
- **`sp_ttm`**：★ 偏离参考库实现：参考库取供应商 ps_ttm 的倒数，我们直接用 PIT 对齐的营业收入TTM / 自算市值（等价于 1/PS_TTM，但分子可审计）。选 TTM 不用单季：营收有季节性，单季口径会在四个季度间系统性起伏。revenue 在 POSITIVE_ONLY 里（<=0 置 NaN）→ sp_ttm 恒正，无 ±1e6 风险。
- **`super_large_order_intensity`**：★★ **偏离参考库的字面公式（必须看）**：参考库是「单日 超大单净额 / 八列毛额」，而八列毛额正是 `mf_elg_order_ratio` 的分母、分子也是同一个单档净额 —— 实测两者截面 rank 相关 **1.0000000**（2026 年 517,349 行逐格核对，最大相对差 4e-8），是**精确重复**。本实现把分子分母都换成 **5 日累计**再相除（ratio-of-sums，不是 5 日比率的均值），既保留参考库「净买入 / 总成交额」的口径与量纲（无量纲），又抓住 thesis 里「**持续**收集筹码」的时间维度。实测与既有因子的最大 |相关| 降到 **0.362**（沙箱 2026，vs 全部 259 个既有生产因子；最近邻 `net_turnover_rate_20`，与 `short_term_reversal_5` −0.359、与 `mf_big_order_ratio` +0.333、与 `elg_net_60d_to_mv` +0.278）。★ 与同族 `mf_large_order_net_5d` **不重复**：那条只算 lg 档、且是「5 日**比率**的均值」，本因子含 elg 档、用「5 日**累计额**之比」—— 参考库把 elg 与 lg 分成两个因子正是因为超大单更脉冲（每笔 >100 万 vs >20 万），实测两者相关仅约 +0.4。`min_count=3` = 参考库 min_periods；停牌/无记录 -> NaN（不补 0）。
- **`tail_risk_pct_60`**：逐字抄自 Class1 risk / tail_risk_pct_60。嵌套窗口：60 日标准化 + 60 日频率 = 120 个交易日，故 warmup=240（同 N=120 的口径）。min_count=30 = 参考 min_periods。★ 偏离：参考库把 |z| 算不出来的日子（停牌）当 False 计进分母（pandas 的 NaN.gt() → False），会把长期停牌股的尾风险稀释掉；本实现额外要求窗口内 ≥30 个真实交易日，否则置 NaN。
- **`td_setup_count`**：★ **口径偏离（唯一一处公式级偏离，务必看）**：参考库 `td_setup_count`（technical_daily.py）返回的是**连续段长度**（`cond` 的连续 True 计数，段被 False 打断后归零），不是窗口统计。本实现按任务书对形态族的定义做成「**20 日窗口内满足形态的交易日占比**」（`roll_mean(cond, 20, 10)`）。理由：(a) 任务书明确要求形态族用窗口占比；(b) 连续段计数在**有缺失的日线上不成立** —— 停牌日被掩码后会打断连续段，于是「停牌多的股票 setup 计数被系统性重置」，这不是我们要的信号；(c) 窗口占比与前 6 个形态因子同构，可比较、可解释。若要换回参考库口径，只需把函数体换成按段累计（本文件不提供，避免两套口径并存）。★ 比较基准是 `shift(hfq_close, 4)`：后复权口径（参考库自己也注明「跨日比较必须走复权基座，未复权 close 在除权日跳变会伪造连跌 setup」）。`hfq_close` 在停牌日是前向填充值 —— 与参考库在 adj 序列上的行为一致，本实现不再额外要求 T-4 有成交（那会把复牌股整段抹掉），只要求 T 有成交。值域 [0,1]。
- **`three_black_crows`**：★ 口径偏离同 `td_setup_count`（参考库 event_dynamics.py 是**连续段计数**，本实现按任务书做成 20 日窗口占比）。经典三只黑鸦是「连续三根阴线」，参考库把它实现为「连续阴跌天数（不限于 3 天）」，本实现是「20 日里的阴跌天数占比」—— 三者在单边阴跌行情里同向，区别只在横盘夹杂时的权重。★ `pre_close` 用 `hfq_close(T-1)` 代替（数学等价，PIT 安全）。★ 掩码 `traded(T) & traded(T-1)`：这个判据**同时**用到当日 OHLC 与前收，前一日停牌时前收是陈旧的，属于第 1 条要挡的复牌跳空。值域 [0,1]。
- **`time_since_52w_high`**：★★ **v3（2026-09-17）**：`warmup_days` 480 → 960。本因子的实现是`np.maximum.accumulate(行号)`（求「最近一次创 250 日新高的行号」），而 `at_high` 依赖 `roll_max` 先有效（面板前 249 行是 NaN）⇒ 累加器只能从第 250 行起有状态 ⇒**面板必须有两倍窗口（≈500 交易日）才能在年初给出正确值**。给 480 天时年初计数被钉在 249（假「250 日没创过新高」），实测 `000001.SZ@2016-01-04` 真值 141 而旧版给 249（960 天版给 141 ✔）。这是面板锚点从「全局」改成「按年」之后**才暴露**的 —— 全局锚定下面板天然够长。
★ 偏离参考库三处：(0) 窗口 252→250 交易日（与另两个 52 周因子一致）；(1) 参考库用「全历史 ffill 最后一次新高时间」→ 无界（可以到几千天），但我们的面板只从 warmup 起算，无界口径会让**长期阴跌股静默变成 NaN**（恰恰是因子最该识别的尾部）；本实现改用 250 日窗内的高点，值域 [0, 249]，永不缺值。(2) 单位用**交易日**而不是参考库的日历天 —— 交易日数不受假期/周末错位影响，更稳。实现上用 np.maximum.accumulate(行号) 取「最近一次创 250 日新高的行号」，所以今天平前高（>=）记 0；这与 roll_argmax 不同（后者并列时取**最早**一次）。
- **`top_list_net_rate_20`**：★ **不再做滞后位移**（2026-09-15 晚，v2→v3）：`stock_top_list` 实测**当天可得** ——日更工程 T=2026-09-15 21:14 的逐日观测 `delay_obs = 0`（服务端当晚就有 T 日上榜记录，取数窗口已按当天收敛）。用户口径：「服务端当天有数据就必须拿当天」。所以去掉了 v2 加的 `ctx.lag_grid(grid, 1)`：本因子 T 日直接用 ≤ T 日的上榜记录，比 v2 多一天信息（整条序列相对 v2 前移一个交易日）。⚠️ 前提：因子在 D 日收盘后到当晚算，下游按 D+1 及以后交易 —— 这样用 D 当晚发布的数据不构成未来函数。（更正史：早期曾列为『别误判成滞后』→ 2026-09-15 白天改成滞后表并位移 → 当晚实测证伪、又去掉了位移。）★ 先汇总再算比率：`stock_top_list` 的键是 `(trade_date, stock_code, reason)`，同一股票同日可因**多个原因**上榜（实测 17,909 组重复），直接逐行取 net_rate 会丢掉其余上榜原因的量。用 `event_grid` 散点求和天然完成聚合。实测 `net_amount / amount × 100` 与表内 `net_rate` 相关 0.9999999，所以本口径与供应商的「净买率」同源，只是做了 20 日累计（更稳、且能覆盖零星上榜）。★ 覆盖率天然低且必须说明：没上榜的日子净买率**无定义**（不是 0）→ NaN，所以非空率 ≈ 「20 日内上过榜的股票占比」（约 10%~20%）。值域约 [−1, 1]（量纲自约，净额与成交额同为元）。★★ 精度（实测的框架缺陷，已在本因子内规避）：输入是**元**级大额（1e9~1e12），而 `ctx.panel.roll_sum` 的 cumsum 在 float32 网格上按 **float32** 累加，长面板下 cs 到 1e10 量级 → 相邻两期相减的绝对误差可达数百元，在**分母小**的票上会放大成可见的因子噪声（实测 002061.SZ 2015-12-31：float32 路径 0.80180428、float64 路径 0.80188098、逐行精确复算 0.80188100 ——同一格子两次运行（面板长度不同）分别落到两个值上，0.38% 的格子有 1e-4 相对偏差）。本因子在滚动前 `.astype(np.float64)` 消除该误差（误差降到 1e-4 元量级）。已作为框架缺口报给主 Agent。
- **`total_leverage_ratio`**：★ 分母是**流通市值**（用户口径，参考库用总市值）。本因子的值就是「两融余额占流通市值的百分比」原值（不取反），方向由 metadata 标注。实测 2026-09-10 p1/p50/p99 = 0.0031 / **0.0377** / 0.111 —— 与「小百分比（0~0.2）」的预期一致。参考库的 clip(0, 0.5) 在本平台是空操作（p99 只有 0.11），未实现。停牌日：分子是状态量、分母的 close 与 float_share 也都是状态量（前向填充），所以停牌日仍有值 —— 这是有意的（停牌期间杠杆并没有消失）。
- **`trend_strength_60`**：★ 两份参考文档里没有同名因子，这是**自建口径**：任务书要求『R² 或 t 值类』。时序回归 R² = 相关系数平方，所以用 roll_corr(行号, 收盘价, 60)² 免去 OLS 的斜率/截距。取**带符号**：纯 R² ∈ [0,1] 对上涨和下跌一视同仁，在动量家族里没有方向；sign(ρ)·R² 让「强势单边上涨」最大、「强势单边下跌」最小（−1）。值域 [−1, 1]。R² 对 y 的正仿射变换不变，所以用不用后复权价、用不用对数价，结果完全一样。与 ret_autocorr_1d_20 同样的退化窗口兜底：|ρ|>1 只可能来自数值退化 → NaN（本因子的 x 是行号、方差大，实测 0 格越界；加它是为了不让同一个坑再咬一次）。
- **`trix_12_20`**：参考库出处：factors.md `类别 price` / technical_daily.py 的 trix_12_20（`_trix_wide` 是私有函数，文档只给了「三重指数平滑(12)的20日变化率」一句）。★ 实现：`E = EMA(EMA(EMA(close, 12), 12), 12)`，`trix = pct_change(E, 20)`。  这与**经典 TRIX 不同**：经典口径是三重 EMA 的**1 日**变化率 ×100，  本因子按文档/因子名的 `_12_20`（N=12, M=20）取 20 日变化率。  两者是同一趋势的不同平滑度，方向语义一致。★ 无量纲（分母是同一条 EMA），不需要额外归一化。★ warmup 400：三重 EMA 的脉冲响应是 Γ(3, 1/α) 形状，拖尾比单层 EMA 长得多，  再叠 20 日变化率；300 不够，给 400。
- **`trix_signal_gap`**：参考库出处：factors.md `类别 price` / technical_daily.py 的 trix_signal_gap。★ 分母保护：TRIX 会穿过 0（无趋势时三重 EMA 走平），参考库写 `|signal|+1e-10`，本文件用 `safe_div(..., min_abs_den=1e-8)`：TRIX 的量级是 ~1e-2，20 日均值 |signal| < 1e-8（= 十万分之一）已经等价于「完全无趋势」，给 NaN；这样比值上界 ~4e6，不会触发引擎的值域告警。★ 与 trix_12_20 共用同一条 TRIX 链，两者会高度相关，下游注意共线性。
- **`turnover_anomaly_20`**：参考库没写 min_periods，按本框架的 n//2 约定取 min_count=10 / 30。60 日窗要 128 个日历天 warmup。分母 to_60 是换手率、恒正，但停牌多的股票 to_60 可能为 NaN → 直接 NaN，不补值。
- **`turnover_f_20`**：价格层没有自由流通换手率字段，故自算 vol/free_share×100（vol 与 free_share 都是**股**；实测与供应商 turnover_rate_f 的中位相对偏差 8.3e-4 —— 差的是股本快照版本，可忽略）。自由流通口径剔除大股东锁定股份，比 turnover_20 更贴近真实交易活跃度。min_count=10 的理由同 turnover_20（对齐参考库 min_periods=10）。
- **`turnover_f_delta_5`**：换手率持续下降 = 浮动筹码被逐步吸收、筹码趋于集中，常伴随筑底。自由流通换手率自算（口径同 turnover_f_20）。★ 偏离：不加参考库的 clip(-1, 3)，极端值交给引擎的 1%/99% winsorize。用 ctx.pct_change（分母取绝对值），所以分母为正时与参考库一致；实测自由流通换手率恒 >= 0，两种写法等价。★ 分母保护**没有加**，依据：全样本 |变化率| > 50 的 131 格（0.0092%）里，分母（5 日前 turnover_f_20）全部落在 [0.096, 15.2]，中位数 1.31 ——不是近零分母，而是**复牌首日的真实流动性暴增**（如停牌前 5 日均换手 0.1%、复牌当日 60%）。加分母地板会把这些真事件一并 NaN 掉，所以只交给引擎 winsorize。|value| max 3440 << 1e8，不触发值域红线。
- **`turnover_std_20`**：★ 用 _roll_std 而不是 ctx.roll_std：后者在 min_count=10<n=20 时把缺失当 0、分母固定取 20，会把波动率算小约 1/3（实测 1.125 vs 1.219），而缺多少格与停牌频率相关 → 偏差带截面结构。ddof=0（总体），参考库 pandas .std() 是 ddof=1；两者只差 20/19 的常数，在有停牌的窗口上 ddof=0 对「有效值个数」才自洽。口径为流通股本换手率。
- **`ulcer_index_20`**：逐字抄自 Class1 risk / ulcer_index_20。回撤口径同 max_drawdown_*（对 20 日滚动前高）。嵌套窗口：前高 20 日 + 回撤均值 20 日 = 40 个交易日，故 warmup=40×1.8+20≈92（比单窗口的大）。min_count=10 = 参考 min_periods。
- **`var_95_20`**：逐字抄自 Class1 risk / var_95_20。★ 用**历史分位数**（线性插值，与 pandas rolling.quantile 同口径），不假设正态。min_count=10 = 参考 min_periods。实现走 `fea.mathx.roll_quantile`（mathx 里有、但 FactorContext 没转发，见模块 docstring）；它对窗口内的 NaN 是**忽略**而不是毒化，与 pandas 的 min_periods 语义一致。
- **`vol_120`**：参考库 risk 类只有 21/42/63/126/252 日的 return_std_*（因子库.md 7、Risk），本因子按家族文档取 120 个交易日窗口，公式形式逐字同 return_std_126d。min_count=60 = n//2。★ 窗口内缺 60 天时框架的方差分母仍是 120，会把长期停牌股的波动率压低约 30%，这批股票同时是低波因子会挑出来的——下游用它做风险控制时需注意。
- **`vol_clustering_20`**：逐字抄自 Class1 risk / vol_clustering_20。min_count=10 = 参考 min_periods。shift(1) 在面板上就是「上一交易日」（面板本身即交易日历），不会跨股错位；warmup 比 W20 多 4 天是为了覆盖这个 lag。★ 数值守卫 |corr| > 1 → NaN：相关系数在数学上不可能越界，越界只可能是「窗口内 |收益| 几乎恒定 → 分母方差 ≈ 0」的退化窗口（实测 296/497093 个格子，最大 4.0），此时真实相关系未定义，置 NaN 而不是 clip（clip 会造出假的 ±1）。
- **`vol_of_rv`**：表达「波动率本身有多不稳定」。★ 与已删除的 `vol_of_vol_20` 不同源：那个是「波动水平的窗口比较」（与 vol_20 ρ=0.968，已删）；本口径是**嵌套滚动**（先算 20 日 RV，再取它 60 日的标准差），衡量的是波动率的**变化节奏**。
- **`vol_of_vol_20`**：逐字抄自 Class1 risk / vol_of_vol_20d。口径就是参考库的「|日收益| 的 20 日 std」（不是「波动率序列的波动」——那个在参考库里叫 vol_of_vol_60，本家族未选）。min_count=5 = 参考 min_periods。
- **`vol_ratio_ma5_ma20`**：对齐 factors.md 的 volume_momentum_5（成交量短长均线比）。参考库的 `vol_ratio_ma5` / `vol_ratio_ma20` 是 finance.volume_ratio 的两条均线，而本框架的价格层读不到 stock_finance 的 volume_ratio 列，所以按同样的「5 日 vs 20 日」构造直接用 vol 自算，比参考的「量比之均线」更平滑、不含当日突刺（当日突刺由 volume 相关的 amount_ratio_20 / turnover 系列覆盖）。min_count=2 / 10 对应 n//2。
- **`volume_dry_up`**：值越小越「干」= 交投意愿冰点（参考库取 -dryness 排名，方向已用 higher_is_better=False 表达）。roll_min / roll_mean 都带 min_count=10，停牌日不会伪装成地量（NaN 不进 min，也不进分母）。
- **`volume_price_divergence_score`**：参考库 Class1 `volume_price_divergence_score`（volume_price_dynamics.py）逐字如上。★ **偏离**：与 `obv_divergence_20` 同一处置 —— 两次 `cross_sectional_rank` 换成两次 `ctx.cs_zscore(..., mask=ctx.universe)`（契约 §2.3 不许在因子里 rank）。★ 与 `obv_divergence_20` 的区别：那个用 **OBV 的 20 日变化 / 均量**（带方向符号的量能净额），这个用**成交量的 20 日变化率**（无量纲、无方向）。两者相关但不重复。★ 价动量用 `ctx.ret(20)`（后复权累计收益，已挡单日 |r|>60% 的复权脏数据），不用 `pct_change(hfq_close, 20)`，与 `factors/momentum.py` 的口径一致。★ 量动量的分母地板给 **1.0**（原始成交量单位）：`vol` 近 0 的格子（停牌前后、极冷门股）会产出 ±1e6 级的假变化率，默认的 1e-12 挡不住。★ 方向：值高 = 价强量弱的可疑上涨，`higher_is_better=False`。
- **`volume_ratio`**：★ 偏离参考库实现：参考库直取供应商 volume_ratio，我们**用价格层自算**（vol / 过去 5 日均量，不含当日）。两条理由：① 停牌日 stock_finance 没有行，as-of 前向填充会把**陈旧的量比**带到停牌日；自算版本的分子 vol 是流量、停牌日为 NaN，语义更干净。② 与 ctx.px('turnover') 同源，口径可自证。实测两者中位相对偏差 2.5e-3（定义一致，差异来自供应商的均量窗口取整）。窗口内出现停牌（NaN）会按契约毒化 → 停牌前后 5 个交易日为 NaN。
- **`volume_tilt_20`**：★ 任务清单里的括号注（「上涨日与下跌日的量比」）描述的是另两个参考因子（up_day_volume_ratio_20 / volume_tilt 的另一种读法），这里按 factors.md 的 volume_tilt_20 原文实现：量加权收益 − 等权收益。收益用小数（ctx.ret），参考库用 pct_chg 原值（百分数）——只差 100 倍常数，截面 rank 完全相同。停牌日在 _ret 里是 NaN，乘积为 NaN 不进窗口。
- **`vwap_daily_deviation`**：★ 全部用**未复权**同日字段：amount/vol = 当日成交均价（元/股），close 取 `ctx.px('close')`（未复权，状态量会前向填充）——**不能**换成 ctx.hfq('close')，那会和未复权的 vwap 不同量纲，在除权日产生 ±30% 的假偏离。停牌日 amount/vol 同为 NaN → NaN。vol 是**股**、amount 是**元**，比值才是价格；两者若不同源会静默错，这里保证同源同日（都来自 stock_daily 的同一行）。
- **`winner_rate`**：★ 直接取摘要表的 below_close —— 它就是「用未复权 close 精确算出的 F(close)」（逐档比较 p <= close 后按归一化权重求和），与参考库 cyq_perf.winner_rate 同义。参考库 rank 取负（高获利盘=获利了结压力=反转信号）；本框架只出原始值，方向看此标注。已 clip 到 [0,1]（仅挡 1e-15 量级的浮点噪声）。
- **`winner_rate_acceleration`**：★ 名称冲突已定调：参考库有两个同名含义——Class1 的 `winner_rate_acceleration`是**一阶** 5 日变化（chip_deep.py），Class2 的 `chip_winner_rate_acceleration` 是**二阶**（「winner_rate_change_5d 的 5 日差分（二阶导数）」）。这里取**二阶**语义：一阶 5 日变化已在 winner_rate_change_20d 的同族里、且与Class1 的 winner_rate_change_5d 只是排序符号相反；二阶导才是独立信息（上涨加速=短期赶顶）。参考库 rank 取负（加速获利=赶顶信号）。要回看 10 个交易日，故 warmup 给 40。
- **`winner_rate_reversal_signal`**：参考库 chip_deep.py 的原始形态，逐字保留。★ 这是本家族唯一**非单调**于获利盘的因子（V 形）：两端（>90% 极端获利 / <10% 极端深套）方向确定但反转压力最大，50% 附近是方向不确定的「不确定溢价」。值域 [−0.5, 0]。
- **`zero_return_fraction_20`**：★ 口径写清（任务要求）：分子 = 「交易所口径涨跌幅 |pct_chg| < 0.1%」，**不是**「没有成交」。停牌日的计入方式：`ctx.px('pct_chg')` 在停牌日是 NaN，本因子把 NaN **同时排除出分子和分母**（np.where(isfinite…) 而不是 `abs(NaN)<0.1 → False`），所以 a) 停牌不会把占比压低，b) 分母是「窗口内有行情的交易日数」，min_count=10 要求至少 10 个交易日。参考库的 `pct_chg.abs().lt(0.1).astype(float)` 会把停牌算成 0（不缩尾），这里显式纠正。值域 ∈[0,1]（末尾 `np.clip(frac,0,1)` 只吃掉 roll_mean 的 ±1e-16 舍入尾巴，实测未夹之前 min = -1.28e-16）。收益用**未复权** pct_chg（这是任务指定，因为要的就是「报价是否没动」，除权日按实际涨跌幅计）。
<!-- FEA:DOC:factor-catalog:END -->
<a id="dev-handbook"></a>
## 运维手册（`CLAUDE.md` 正文 · 现行）

> 本节是原 `CLAUDE.md` 的**现行版本**，2026-09-20 从折叠块里取出、直接展开。
> 内容涵盖硬约束、PIT 红线、本模块速查、全量 vs 增量、性能实测、架构图、已知的坑。
> 其中的「与其它文档的关系」一节写的 `FACTORS.md` / `docs/FACTOR_TRIAGE.md` /
> `factors.md` / `因子库.md` 等**旧文件名在 2026-09-19 的合并中就已经不存在了**，
> 对应内容现在的去向是：
>
> | 旧引用 | 现在在哪 |
> |:--|:--|
> | `FACTORS.md` | 本文上方的[自动因子字典](#factor-catalog) |
> | `docs/FACTOR_TRIAGE.md` | [`HISTORY.md`](HISTORY.md) |
> | `factors.md` / `因子库.md`（数据商参考库） | [`REFERENCE_lingqi.md`](REFERENCE_lingqi.md) |
> | `docs/2026-09-1x_*.md` 各期维护记录 | [`HISTORY.md`](HISTORY.md) |

<a id="source-881144153522"></a>
<!-- SOURCE:88114415352273535be51210a061520f733792faa687671cbdbfe597d1c772df:BEGIN -->
<a id="source-881144153522-featureengineering--量化平台--模块②-因子工程"></a>
# featureengineering —— 量化平台 · 模块② 因子工程

> **2026-09-19 当前运行口径（优先于下方历史记录）：**输出下界统一为 **2018-01-01**；固定股票池仍为 **2115 只**；注册对象 **337 因子 + 5 标签**。实际输出起点取全局下界和因子可得起点的较大值，保留更早上游数据做预热。
> 日常手动运行 `python main.py`；全量或多年修复使用 `python main.py rebuild`，默认 1 个进程，最多 2 个。不要照旧记录使用 8 个进程。
> 本轮代码变更、验证结果、内存约束、新字段和剩余边界统一见
> **[统一日期与数据对齐（`HISTORY.md`）](HISTORY.md#source-8f3d6d26d029)**。
> 公式与完整注册清单见上方的[自动因子字典](#factor-catalog)。
> 下方带旧日期的数量、起点、测试结果属于当时快照；不能作为当前全库已通过验证的证据。

> ★★★ **2026-09-18 深夜：股票池已冻结为固定 2115 只 + 全量重建完成。**
> 池子口径（主板 ∩ 至今存续 ∩ 从未ST ∩ 上市≤2018-01-01）、三个已接受的代价
> （**幸存者偏差**等）、以及一个「13 个因子永远重建不了」的真 bug，
> **权威记录在 `QUANT_PLATFORM.md` 第二十二部分（§81~§88）**；
> 操作口径见本文件坑清单 **#18**，名单本体 `conf/universe_frozen.tsv`（文件头写了口径与代价）。
> ⚠️ **结论（IC / 净值）不可跨这次变更前后比较；做 A/B 必须两侧同池。**

> 🔍 **前视（未来数据）审计记录：`docs/2026-09-18_前视审计.md`** —— 2026-09-18 对 **230 个注册对象
> （225 因子 + 5 标签）逐个评估**「有没有用 > T 的数据」，含：6 种审计手段的实测结果、16 个族的小结、
> **12 条已记录问题（F1~F12，按用户要求只记录不修改）**、6 项上游数据真实性抽查、逐因子判定附录。
> 冷启动若关心 PIT / 前视，先读它 —— 它把「已堵死的入口」与「剩下的运行侧残留风险」分了类。

> 📌 **当日专题记录**：`docs/2026-09-15.md` —— 2026-09-15 那一轮的**改动清单 / 注意事项 /
> 讨论纪要**（delay 反转、NaN 落盘、停牌矩阵、复权体检、10 项改动与各自验证方式）。
> 冷启动建议顺序：本文件（硬约束 + 坑）→ `docs/2026-09-15.md`（最近的变更与契约）。

> ### ⚠️ 上游数据变更提示（2026-09-17）：三张表已彻底删除，**不要再依赖**
>
> 用户拍板，以下三个上游数据集**连同数据一起删除**，日更工程与全量工程都**不再生成、不再更新**：
>
> | 数据集 | 原用途 | 删除原因（实测） |
> |:--|:--|:--|
> | `index_weight` | 指数成分与权重（月频） | 厂商月频延迟 1~2 个月、200 个指数里 11 个服务端无数据 |
> | `stock_report_rc` | 券商研报 / 分析师预期 | 发布日天然稀疏，"最新报告日"在 T−0~T−4 之间跳，delay 不是常数 |
> | `stock_main_fund_flow_overview` | 主力资金四档净流入总览 | 与 `stock_main_fund_flow` 口径不兼容、且厂商会重算历史值 |
>
> **对因子侧的影响**：
> 1. **当前 259 个因子没有一个依赖它们**（已全仓核对），所以**无需改任何因子代码**；
> 2. 将来若要开发"分析师预期 / 指数成分 / 资金流分档"类因子，**这三张表已经不在了**
>    （`stock_main_fund_flow`（2010 起、买卖分列）与 `index_daily` 仍在，可作替代起点）；
> 3. `fea/delay.py` 的滞后表名单随之缩短 —— **生效名单只剩 `stock_margin_detail: 1`**
>    （`_BUILTIN` 里那行 `stock_report_rc: 2` 成了死条目、无因子引用；按用户要求
>    「只写 tips、不动因子侧代码」**未清理**，将来顺手删即可）；
> 4. 备份仅留档、**勿恢复**：`../everyday_tasks/state/backtest/red3_purge_20260916_235239/`。

> ### 📌 第四批因子（2026-09-18）：新增 5 个家族 / 81 个因子
>
> 完整记录：**`docs/2026-09-18_新一轮因子开发.md`**（本轮唯一权威文件）。
> `intraday2`(24) · `breadth`(12) · `sector`(17) · `fundflow3`(12) · `fundamental3`(16)。
> 五项验证全绿（`check` / `audit-pit` / `eval` / `dedup` / 增量==全量 1.87 亿格 0 不一致）。
>
> **三条本轮新踩到、下一轮动这些数据前必须先知道的坑**（细节见各家族 docstring）：
> 1. ★★ **`stock_market_distribution_history` 的 `limit_up_count` / `limit_down_count`
>    在 2010–2019 恒等于 0**（十年，不是缺测）⇒ 用它做因子会得到恒 0 常数。
>    改用涨跌幅分桶 `up_5_to_7+up_7_to_10+up_over_10`（全历史可用）。
> 2. ★★ **「自参照滚动分位」做条件化在趋势行情里会整条截面变 NaN** ——
>    「今天相对自己过去很弱」按定义不成立，可以连续几十天不命中 ⇒ `check` 报 `✘CONST`。
>    改用**绝对阈值**（并先验证「最长连续不命中 < 窗口长度」）。
> 3. ★ **`tdx_blocks` 的板块代码前缀与 `block_type` 不一一对应**
>    （8805/8806/8807/8808/8809 里 0/1/2 型混编）⇒ 必须查字典，不能靠前缀。
>
> **本轮的行业口径**：`factors/sector.py` 用**自建滚动相关性动态行业**
> （每年初按过去 250 日收益相关性定行业归属，只用历史数据），
> **完全不碰** `tdx_block_stocks` / `index_ths_constituent_stocks` 这两张
> 无日期的快照成分股表（参考库自己就因快照非 PIT 禁用了 6 个 sector 因子）。

<a id="source-881144153522--因子开发的硬性约束每次开发前必须确认别凭印象"></a>
## ★★★ 因子开发的硬性约束（每次开发前必须确认，别凭印象）

下游模型任务是 **A 股日横断面回归排序**，由此推出四条不能违反的规则：

| # | 约束 | 具体含义 |
|:--|:--|:--|
| 1 | **全部因子必须日频** | 对齐到 `(trade_date, stock_code)` 日频面板后落盘，不允许周频/月频/稀疏事件频率 |
| 2 | **固定主板股票池 2115 只，名单为准** | 沪 `600/601/603/605` + 深 `000/001/002/003`（002 原中小板，2021-04 并入主板）<br>**排除** 创业板 `300/301/302`、科创板 `688/689`、北交所 `832/833/920` |
| 3 | **输出起点由 `conf/config.yaml` 的 `default_start` 控制** | 当前 **2018-01-01**（2026-09-19 统一输出下界；该值进 recipe 指纹，改它会触发全量重建）。`FactorSpec.start` 留 `None` 即跟随配置；实际起点取显式起点和全局下界的较大值。以下较早显式日期仅保留历史来源，不能绕过 2018 下界：受上游起点限制的因子显式写死（筹码→2018-01-02、两融→2011、股东户数→2016-01-04、涨跌停→2015-01-05、limit_list→2020-01-02、**深回看 `pegh5`→2016-01-18 / `etp5`→2015-06-16** 见下方「按年锚定」）<br>★ **全历史重建不要直接 `main.py run --rebuild`**：15 年会挤在**一个进程**里连着算，价格层/派生层窗口只扩不缩 → 内存爆。用 `main.py rebuild`（按年分块，`--resume` 断点续跑） |
| 4 | **★ 历史因子值不得因未来的分红事件而变化** | 见下方「PIT 红线」 |
| 5 | **最终产物必须日频** | 由 5min 派生的日内因子也**先聚合成日频再落盘**，绝不落分钟级产物 |
| 6 | **格式统一，只有日期允许长短不一样** | 全部因子 4 列 `trade_date/stock_code/value/rank`，同列序、同 dtype（`string/string/float32/float32`）、同分区方式。由 `fea/store.py` 强制 + `main.py check` 常驻校验。<br>★ **2026-09-15 晚用户拍板「NaN 还是落盘的好」**：整块面板都落盘（`value`/`rank` 全 NaN 的格子也写），**行数 = 面板大小（≈3484 只 × 交易日）**。理由：只写非空行会让「不在股票池 / 上游没数据 / 算了是 NaN」在产物里长得一模一样，覆盖率无法审计。配套：`check` 的"值域异常"不再把 NaN 当错（只查 `±inf` 与量级），"日均截面"列 = 每日**非空**格子数；实测增量仍幂等（跑完逐字节一致） |

<a id="source-881144153522--pit-红线前复权qfq会破坏历史的可复现性"></a>
### ★ PIT 红线：前复权（qfq）会破坏历史的可复现性

前复权把整条价格序列按**最新**的复权因子缩放：
`price_adj(t) = price_raw(t) × adj_factor(t) / adj_factor(T_latest)`。
一旦发生新的分红/送转，`adj_factor(T_latest)` 改变 → **全部历史价格被一起重算** →
同一段历史今天算的因子值 ≠ 昨天算的。回测里这就是前视偏差：模型「知道」了未来的分红。

- **禁止**把 `stock_kline_adj` / `stock_daily_adj`（前复权）直接喂给因子函数。
  `fea/spec.py` 的 `register()` 会**直接抛错拒绝**；确需使用必须显式 `allow_qfq=True` 并写明理由。
- 价格**水平**类（市值、book-to-market）→ 用**未复权**价 × 当期已披露股本。
- 收益**比率**类（动量、波动）→ 未复权价 + `stock_adj_factor` 取**截至当日**的累计复权因子
  （等价于「截至当日的后复权」；后复权锚定序列起点，历史值稳定）。
- 价格一律走 `ctx.hfq(...)` / `ctx.ret(...)`（价格层已把复权、停牌、脏行都处理好了）。

<a id="source-881144153522--已知问题stock_adj_factor-有-545-格历史异常2026-09-15-实测建档每月复检一次"></a>
### ★★ 已知问题：`stock_adj_factor` 有 545 格历史异常（2026-09-15 实测建档，**每月复检一次**）

用户 2026-09-15 交办：「确认一下目前的复权 adj_factor 是合理正确的…将这个问题记录一下，
我们可以隔一个月的时候检查一次」。

**结论：整体正确，但有 545 格（全历史 0.0038%）局部异常。** 判据 = 用**独立字段对账**：
`hfq_ret = (close×adj_factor)_t / (close×adj_factor)_{t−1} − 1` 必须等于 `pct_chg/100`
（供应商的 `pct_chg` 按除权后基准价 `pre_close` 计算，代表真实收益）。

| 判据 | 全历史实测（2010-01-04 ~ 2026-09-15，14.37M 格） |
|:--|:--|
| 结构完整性（主键/缺失/非正值） | 0 问题 ✔ |
| 逐股单调性（累计因子不该减） | **92 格 / 18 只股**违反（相对跌幅 >0.1%） |
| **hfq 收益 vs 交易所口径** | 红线 1e-3 下 **453 格**；2e-4 / 5e-4 档（12,651 / 1,848 格）是**舍入噪声** —— `pct_chg` 只 2 位小数、`adj_factor` 只 4 位小数 |
| 除权日一致性（`r_fac × r_ref ≈ 1`） | 47,560 个除权日里 99.4% 通过；257 个不符（多在 2020-2023） |

**2026 年（与我们的因子直接相关）**：935,089 格里 **28 格 / 22 只股**异常
（**主板只有 17 格 / 13 只股**，其余是创业板/科创板，不在股票池内）。三类机制（已逐格定性）：
- **A 因子跳了、价格没跳（全历史 102 格）**：hfq 序列凭空多一根假涨 —— 2026 年 15 只集中在 07-06；
- **B 价格跳了、因子没跳（205 格）**：除权日因子**晚一天**才补（`688689.SH` 06-29 除权、06-30 补跳，
  比例严丝合缝 ✔）或**整年都没补**；
- **C 两边都跳但比例不符（146 格）**、**D 因子相对前值下跌（92 格）**。

**影响**：这些 (股,日) 上吃 hfq 的因子（动量/波动/`label_ret_*`）会有 0.1%~48% 的假收益。
量级极小但**不该静默**，所以：
- 基线清单：`state/known_adj_anomalies.json`（545 条，含 2026 年的人工定性结果）；
- **每月复检命令**：
  ```bash
  PY=/autodl-fs/data/miniconda3/bin/python
  $PY scripts/check_adj_factor.py                  # 全历史；只报"新增"，有新增则退出码 1
  $PY scripts/check_adj_factor.py --years 2026     # 只看当年
  ```
  报告落在 `state/adj_check_<日期>.md`。**新增 = 厂商回溯改写了复权因子**（或又出现新批次），
  要按 A/B/C/D 逐条定性后再决定是否并入基线。
- ⚠️ **教训（免得再踩）**：除权日两侧的正确关系是 **`r_fac × r_ref ≈ 1`**
  （因子跳变比例 = 1/参考价比例）。我第一版脚本写成 `r_fac ≈ r_ref`，
  一次性报了 **50,233 个假异常** —— **判据写错比没有判据更危险**。

<a id="source-881144153522--第二个-pit-陷阱数据到达晚同样会改写历史stock_margin_detail-等"></a>
### ★ 第二个 PIT 陷阱：数据**到达晚**同样会改写历史（`stock_margin_detail` 等）

上面那条说的是"数据被**重算**"（前复权），这一条说的是"数据**来晚了**"。
后果一样——同一个 T 的因子值前后不同——但更隐蔽，因为**它当天看起来完全正确**。

**实测清单（2026-09-13 首次穷举，2026-09-15 晚由日更工程逐日复核后定稿）**：

| 数据集 | 滞后 | 备注 |
|:--|:--|:--|
| **`stock_margin_detail`** | **1 个交易日** | 每交易日 ~4440 行；用户口径「一定是 1」→ 让日更闸门**等它** |
| ~~**`stock_report_rc`**~~ | ~~2 个交易日~~ | ⚠️ **该数据集 2026-09-17 已彻底删除**（见文件开头的提示），此行仅作沿革 |

> ★★ **2026-09-15 深夜定稿（同晚第二次反转）**：`ths_hot` / `stock_st_info` / `index_ths_daily`
> **由 1 改回 0** —— 它们不是"次日才发布"，而是"**当天深夜才发布**"。证据是同晚 7 次采样：
> 滞后序列（新→旧）`0,1,1,1,0,0,0`，**21:22 还观测到滞后 1，23:32 服务端就有当天数据了**。
> 连同更早的 `dc_daily` / `stock_top_list` / `stock_dragon_tiger` / `stock_adj_factor_changes`
> （也改成 0），**现在的滞后表只剩 2 张**：`stock_margin_detail:1`、`stock_report_rc:2`。
>
> ⚠️ **教训（写下来免得再犯）**：「发布晚」不等于「不发布」。我当晚一度把 `fea/delay.py`
> 改成"**按最近一次观测取值**"，结果同一张表在**同一个晚上**从 1 翻成 0 —— 因为
> **采样时间**（早/晚）会系统性影响观测值。现在改成：
> **① 声明常数（`frequency.yaml`，与日更侧 `registry._DELAY` 逐表对齐）为准；
> ② 观测只做安全网**（连续 ≥3 次观测一致地更差才临时上调，只上不下）。

> ✅ **框架已有自动守卫（2026-09-15 实施，当晚两度重写取值规则）**：`fea/spec.py` 的
> `register()` 会在注册期**直接抛错**拦下"依赖了滞后表却没声明 `lagged_ok`"的因子。
> 名单**不硬编码**，由 `fea/delay.py` 取值：
> **① 声明常数为准**（`datadownload/conf/frequency.yaml` 的 `delay_days`，与日更侧
> `registry._DELAY` 逐表对齐）；**② 观测只做安全网**（`delay_history.json` 里
> 连续 ≥3 次观测一致地更差才临时上调，只上不下）；③ `_BUILTIN` 兜底。
> 最后统一过滤掉 `<=0` 的表。
> ⚠️ 演化史（别再走弯路）：最早是"三路取最大值 + 丢掉 0" → **放宽永远传播不了**；
> 当晚我一度改成"按最近一次观测取值" → **被早采样带偏**（同表同晚 1→0，见上）。
> **正确做法**：`FactorSpec(..., lagged_ok=("表名",))` + 函数体**最后一步**
> `return ctx.lag_grid(grid, n)`（行位移 = 把 (T,C) 网格下移 n 个交易日，n ≥ 上表的天数）。
>
> ⚠️ **别误判成滞后**（行数本来就波动、最后一天为 0 属正常）：
> `stock_limit_up` / `stock_limit_list`（涨停家数 40~95 波动）、
> 全部季频/月频/不定期表。

> ★★ **2026-09-16：`stock_kline` 已从上游彻底删除，本模块不得引用。**
> 它是全库唯一「**每次抓取都改写历史值**」的表 —— 周/月线实为「抓取请求范围 ∩ 该周期」
> 内日线的聚合（**不是日历周期聚合**），抓取窗口下界一滑动，**已结束周期的值也会变**。
> 实测（5,550 只全市场对账）：周线 `09-11` == 日线[09-09~09-11] **100% 吻合、中位相对差 0**；
> 而 vs 完整一周 [09-07~09-11] 只有 **0.1% 吻合、中位相对差 41.5%**。
> 另外它 14 个数据列与 `stock_daily` **完全相同**（只多 `period` 标签）、覆盖区间还更短 ⇒ **零额外信息**。
> **需要周/月频特征：用 `stock_daily` + `ctx.cal` 现算**（自己切周期比接口那份准）。
> 守卫：它已从上游注册表删除，`register()` 的 qfq/存在性检查会拒绝依赖它的因子。

爬取本身都**成功跑完**了（不是模块① 没爬到），且是**永久性质**——不是"明天就补上"。

**★ 别把这几个误判成滞后**（它们每交易日行数本来就波动，最后一天为 0 或偏少是正常的）：

| 数据集 | 为什么不是滞后 |
|:--|:--|
| `stock_report_rc` | 券商研报**不定期**：实测 `…88 0 0 0 67 0`，连续 3 个交易日为 0 |
| `stock_dragon_tiger` / `stock_top_list` | 龙虎榜**上榜家数每天不同**（17~20 / 58~67），要看连续性而不是最后一天 |
| `stock_limit_up` / `stock_limit_list` | 涨停家数本身波动大（实测 40~95），少不代表缺 |
| `stock_kline` / `stock_kline_adj` | **周线/月线**，根本不该每天有 |
| 财报五张表 / `stock_pledge_stat` / `index_weight` / `stock_forecast` | 季频 / 月频 / 不定期，按 `ann_date`（或 `end_date`）对齐，不适用"日频滞后" |

**判定方法（照做，别凭印象）**：取某数据集**最近 12 个交易日**的行数分布，
若「每个交易日都有稳定行数」而「最后一天为 0」，就是滞后；若行数本身就忽高忽低甚至连续为 0，
那是不定期数据集，**不适用**这条规则。
（交叉验证：在**周六/周日**跑一次增量，若 max_date 仍停在周四而不是周五，即坐实滞后——
因为周末爬取时周五的数据一定已经发布了。）

**为什么这构成前视**：`panel.asof` 是"取 `src_day <= d` 的最近一条"（前向填充）。
不处理的话，同一个 T 在不同日子跑出来不一样：

| 跑的时刻 | 上游 margin 有到 | T=09-11 取到的值 | 看起来 |
|:--|:--|:--|:--|
| 今天 | 09-10 | 回退到 **09-10** 的值 | ✔ 正确 |
| 明天 | 09-11 | 取 **09-11 自己**的值 | ✘ 同一个 T 的值变了 |

**规则**：`stock_margin_detail` 里 `trade_date = d` 的行，其**可得日**是 `d` 的
**下一个交易日**。所以 T 日的因子只能用到 `trade_date <= T−1（上一个交易日）` 的记录。
其他日频数据源（`stock_daily` / `stock_limit_up` 等）仍然直接用 T。

**实现**（两条路，都必须真的做——不做的话因子当天能跑对，明天静默变错）：

1. **行位移**（推荐，零框架改动）：把 margin 的 `(T, C)` 网格整体下移一行 ——
   `out[1:] = grid[:-1]`。面板本身就是交易日历，移一行 = 移一个交易日，
   语义即 `value(T) := margin(T−1)`。首行变 NaN 无害：它在 warmup 区，进不了输出。
2. **日期位移**（更显式）：给 `asof_daily` / `event_grid` 传
   `next_trading_day(trade_date)` 而不是 `trade_date`。目前 `FactorContext`
   拿不到交易日历，走这条路要先给 ctx 加个日历引用。

**加 margin 因子时必须同时改的两处**：

- `conf/config.yaml` 的 `dep_backfill_days` **没有** `stock_margin_detail` 条目 →
  落到 `default: 400`。同类日频的 `stock_limit_up` 用的是 **60**。margin 的水位
  每天都变，不配的话**每次增量都会回算 400 天**。
- `engine.DEP_PIT_COL` 也没有该条目 → 回退到 `trade_date`
  （`upstream.py:124` 的 `pit_col or "trade_date"`）。对这张表恰好正确，
  但属于"碰巧对"，顺手显式写上。

**这份清单会变长**：`stock_daily` 是**基准**（它永远跟到最后一个交易日），
拿别的表和它比就行。发现新的一例就加进上表。

**另一条不同的问题（`stock_st_info` 特有）**：它参与因子生成，却**不在任何因子的
`deps` 里**，而 `universe_fp` 只随**配置**变、不随数据变 —— 所以上游新增一条 ST 记录
**不会触发任何重算**。后果不是主动改写，而是**静默不一致**：下次全量重建
（改配置/改 version）会把那些日期的股票池重算一遍，重算前后的 T 日成分股不一样。

<a id="source-881144153522-本模块速查"></a>
## 本模块速查

```bash
PY=/autodl-fs/data/miniconda3/bin/python      # ★ 必须全路径！非登录 shell 的 python 没有 pandas
$PY main.py run                     # 每日增量（用户自己跑这条）
$PY main.py run --rebuild           # 全量重建（当前 = 全历史 15 年，★ 请改用下面的分块脚本）
$PY main.py rebuild --jobs 1       # ★ 全历史/多年重建：按年分块 + 内存采样 + 断点续跑
$PY scripts/backfill_history.py --factors a b  # ★ 只补某几个因子的历史（块内只有 1~2 个任务，峰值 6~12 GB）
$PY scripts/backfill_history.py --dry-run      # 只列计划（哪些年已覆盖；全部覆盖 = 打印「共 0 块待跑」）
$PY main.py run roe_ttm yoy_revenue # 只跑指定因子
$PY main.py run --jobs 1            # 强制串行（排查用）
$PY main.py list                    # 因子清单与本地进度
$PY main.py status                  # 已落地统计
$PY main.py check                   # 基础体检：universe / 值域 / 复权 / 格式；历史因果性另用 audit-pit
$PY main.py eval --years 2026 2026  # 有效性：IC / RankIC / 覆盖 / 分层单调性
$PY main.py dedup                   # 冗余检测：|ρ|≥0.95 的重复簇 + 完全重复对
$PY main.py audit-pit               # ★ 前视审计：静态扫描 + 抽样**截断复算**
$PY main.py dayhash --date 2026-09-14 --verify   # ★ 逐截面 MD5 台账：历史有没有被改动
$PY main.py docs                    # 重新生成 FACTORS.md
```

<a id="source-881144153522--全量生成-vs-增量生成2026-09-15-实测两者必须逐格一致"></a>
## ★★ 全量生成 vs 增量生成（2026-09-15 实测，两者必须逐格一致）

**两条命令，同一套代码，结果必须完全相同**：

```bash
$PY main.py run --rebuild            # 全量重建（275 个因子 / 2026 一年 ≈ 9 分钟）
$PY main.py run                      # 每日增量（默认只重算缺口 + 回刷窗口 ≈ 8 分钟）
```

**实测证据**（两次，均为 0 不一致）：

- 2026-09-15：`scripts/test_incremental.py`（备份全量结果 → 删掉最新交易日 → 跑增量 →
  逐格比对）在 275 个因子 / **1.355 亿个格子**上比对，**不一致 0 个**。
- **2026-09-17（T=09-17）**：流程同上但起点是「全量重建 2026 → 留 MD5 基准 → 裁掉 09-17 →
  跑增量」—— 230 个因子 / **1.386 亿个格子逐格比对不一致 0 个**，另加逐截面 MD5
  **1610 条变化 0 条**（两个独立验证）。
  **★ 本次性能实测（用户点名要记录）**：增量总用时 **507.1 s（8.45 min）**，`test_incremental`
  记的墙钟 **517.4 s**，`--jobs 4`；**内存峰值 32.75 GiB**（anon 口径、1 s 采样）；
  各因子 `task_seconds` 合计 1,947 s ÷ 墙钟 ≈ **3.76 ≈ 并行度**（调度饱满）；
  每因子平均重算 **9.6 个交易日**（9~44，由回刷窗口 `revision_days` + L2 水位决定，不是只算 1 天）。
  ⚠️ 内存峰值**漏采了前 6 分钟**（前段人工观测 32.2 GiB）→ 真实值 ≥ 32.75 GiB。
  逐字记录见 `docs/2026-09-17_增量与全量一致性实测.md`。

**为什么能一致**（三条设计，动其中任何一条都会破坏它）：

1. **面板顶锚在因子自己的起点**（`Engine.run_year`）。增量只输出缺口那几天，
   但**面板按同一个顶边构造** —— 否则 `mathx` 的滚动原语（用"列首第一个有效值"做
   浮点中心）会随窗口漂移，实测 49 个因子出现 0.01%~3% 的格子差最后一个 ULP。
2. **财务字段集是 run 级统一**（`Engine._deriv_fields_run`）。按因子各传各的会让
   "要窄表"和"要全表"的因子互相踢缓存 → 实测增量跑重建衍生层 **73 次**（每次 6~25s）。
3. **warmup 必须给足**（否则窗口头部会 NaN，而那是**算错**不是"有点噪"）。

**性能红线（都是实测踩过的）**：

- `main.py` 在 `import numpy` **之前**把 `OMP/MKL_NUM_THREADS` 压到 1
  （服务器默认 32，8 进程 × 32 线程 = 250+ 线程抢 32 核，load 冲到 60，单任务慢几倍）。
- `series_to_int` 是**全引擎最热函数**：原先每个字段落格都要重转一次日期列
  （单任务 47 次 × 2.4s = 占 84% 时间）。现已改 numpy 快路径 + 价格层缓存 `_day_i`。
- 单因子成本由「读+重写 50 万行的年度分区 + 载入上游窗口」主导，**不是**算几天。
  想把日增量压到 1~2 分钟需要改存储布局（例如"最近 N 天"单独存一个文件），尚未做。

**逐因子耗时怎么看**：每次 `run` 都会把 `task_seconds`（本因子各任务实际耗时之和）
与 `task_days`（重算了多少个交易日）写进 `state/<因子>.json` 的 `last_run`；
控制台每行末尾也直接打印该任务的耗时。

**逐截面 MD5 台账**（`main.py dayhash`）：把每个因子的**最近 7 个交易日的横截面**
各算一个 MD5（位级比较，float32 原始位），存到 `log<MMDD>/dayhash.tsv`。
**用途**：上游每天更新后重跑一次，若历史上某天的哈希变了 → 说明上游回溯改了历史，
或增量没有严格复现全量。这是"历史永不改变"这条红线的**常驻证据**，别只在出问题时才跑。

> **`audit-pit` 的截断复算是抓前视唯一可靠的办法**：静态扫描只能看"写出来的"
> 未来引用（`shift(-1)`），抓着不着"全样本标准化""用整段面板算极值"这类隐性泄漏。
> 判据是那条定义本身 —— **T 日能算出的值只能用到 ≤T 的数据**：
> 把输出范围截到 T 重算（面板右端就是 T），与生产落盘值逐格比对，不一致就是有未来函数。
> 复算在沙箱目录里做，不碰生产 `data/` 与 `state/`。
> **输出起点附近的时间点最重要**：L2 水位驱动的重算窗口是"尾部"，所以泄漏通常表现为
> **同一个 T 在不同日子跑出不同值** —— 这正是它要抓的。

> ⚠️ **跑每日增量前，先确认模块① 的上游数据到齐了**：上游没到齐就算，会产出基于
> **不完整上游**的因子值。虽然 L2 水位机制会在上游补齐后自动重算，但**重算之前下游模型
> 可能已经用了**。（目前 `main.py run` **没有这道闸门**，得靠人确认。）

<a id="source-881144153522-性能设计与实测2026-09-13-优化"></a>
## 性能设计与实测（2026-09-13 优化）

| 操作 | 优化前 | 优化后 |
|:--|--:|--:|
| 每日增量 | 44s | **11s** |
| 全量重建（8518 万行） | 10.5 min | **76s** |

优化点（按收益排序，都在 `fea/` 里）：

1. **`listed_mask` 向量化 + 缓存**（最大的一项）。原来是 `for c in range(3484):
   mask[a:b,c]=True` 的 Python 循环，单次 1.86s，而它只是
   (日期范围 × 股票池) 的纯函数，却在 10 因子 × 15 年里被重算 150 次 → ~280s。
   改成差分数组+前缀和（`np.bincount`），并在 `Engine.universe_for` 上按日期范围缓存。
2. **串行/并行自适应**：按 **(因子, 年)** 切任务（143 个），而不是按因子切——
   各因子耗时差一倍以上（49s vs 87s），按因子切会被最慢的那个拖住。
   用 `fork` + `ProcessPoolExecutor`，父进程预建共享状态（衍生层、ST 事件），
   worker 写时复制继承。**manifest 只由父进程写**，worker 绝不碰状态文件。
3. **`st_mask` 预处理**：原来每次对 32.7 万行做 `isin(dict)`+`map(dict)`，单次 917ms。
   抽成 `st_events()`（与面板日期无关，只算一次），并用 `reindex` 代替 `map(dict)`。
4. **上游水位缓存**：`watermark` 要逐分区读 `ann_date` 列（stock_income 有 37 个分区），
   而它在每个因子收尾时都被调用 → 按数据集缓存。

**并行正确性已验证**：串行与并行重建的结果**逐因子行数完全一致**
（如 `asset_growth_qoq` 9,329,813 行），`check` 全绿。

- 上游数据：`../datadownload/data`（模块① 的产出，本模块**只读**）
- 产物：`data/factors/<因子名>/year=YYYY/data.parquet`（统一 4 列，年分区）
- 状态：`state/<因子名>.json`（**一因子一文件**，增量与断点续传的依据）

> ### ★★ 2026-09-17：本模块已放开**全历史**（2012-01-01 起）
> 用户要求「历史因子全部补全」，`conf/config.yaml` 的 `default_start` 已改为
> **`2012-01-01`**（进 recipe 指纹 → 自动全量重建）。配套改动见
> **`docs/2026-09-17_全历史补全.md`**，要点：
> 1. **面板锚点从「因子全局起点」收窄为「本任务所在年的 1 月 1 日」**
>    （`fea/engine.py::_anchor_for`）。面板 = `[本年 1/1 − warmup, 本年]`：
>    15 年总计算量从 ≈120 个「单年」降到 ≈15 个，单进程内存从 7~8 GB 降到 2~3 GB，
>    **每日增量顺带从 ~6 分钟降到 ~1 分钟**。落盘仍按年分区，时间轴靠 warmup 保证连续。
> 2. **全历史重建走 `scripts/backfill_history.py`**（按年分块 + 每年独立进程 + 内存峰值采样）。
> 3. **深回看因子必须显式写起点**：`pegh5` → 2016-01-18、`etp5` → 2015-06-16
>    （实测首个有效日；再往前是整年全 NaN 的垃圾分区）。
> 4. `data/derived/` 的日内层/筹码层会按年重建（自愈，`scripts/prune_derived.py` 可裁）。

<a id="source-881144153522-架构一图"></a>
## 架构（一图）

```
main.py            薄入口：run / list / status / check / docs
fea/
  spec.py          ★ FactorSpec + @register 声明式注册表 + PIT 红线守卫
  deriv.py         ★ 财务衍生层：单季拆分 / 三行式TTM / 追溯修正 / PIT 对齐（坑都在这）
  engine.py        ★ 三层失效判定 + warmup + 按年写（每年只写一次）
  panel.py         (T,C) 网格 + asof/scatter + 缩尾截面排名
  context.py       FactorContext —— 因子作者唯一要打交道的对象
  universe.py      主板过滤 + 上市窗口
  upstream.py / store.py / manifest.py / dates.py / config.py
factors/           因子实现：一个家族一个文件，import 即注册
```

**当前规模**（2026-09-18，第四批已并入）：**306 个因子 + 5 个标签** /
**36.88 亿行 / 16 GB**（按年分区）/ **2012-01-04 → 2026-09-17**（15 年）。
`conf/config.yaml` 的 `default_start = 2012-01-01` 同时是计算下界。
全历史重建走 `scripts/backfill_history.py`（按年分块）；
每日增量分钟级（当日缺口 + 最近 10 天回刷，实测 311 个对象 `--jobs 4` ≈ 12 min / 峰值 48 GB）。

新增因子 = 在 `factors/` 写一个 `@register(FactorSpec(...))` 函数 + 在
`factors/__init__.py` 加一行 import。**不需要改框架代码。**

<a id="source-881144153522-三条最容易被忽略的约定"></a>
## 三条最容易被忽略的约定

1. **因子函数只吃 `(ctx)`、只吐 `(T, C)` float32 数组**，不接收也不返回 DataFrame。
   这保证了内存有界、滚动窗口可向量化、增量 warmup 可证明。
2. **一切「跨期」比较都在报告期粒度上做**（TTM / 同比 / 环比），再按 `ann_date`
   前向填充到日频。**绝不**对日频序列做 `.shift(252)` ——那会横跨报告期边界。
3. **warmup_days 是日历天，必须给足**。财务类 700 天；给少了，每次增量都会在
   窗口头部产出一年左右的 NaN，而且看起来只像「因子有点噪」。

<a id="source-881144153522-已知的坑实测别重踩"></a>
## 已知的坑（实测，别重踩）

0. **★★ 「状态型」因子的 warmup 必须给「两倍窗口」（2026-09-17 实测，全历史回填时抓到）**：
   凡是把状态**跨行累积**的因子（`np.maximum.accumulate` —— days-since / 连续段那一类），
   状态只能从**滚动窗口有效之后**才开始积累。所以要在第 T 天拿到正确值，面板必须覆盖
   **窗口 + 窗口** 个交易日（前一半让 `roll_*` 有效，后一半让累加器有状态）。
   - 实测标本 `time_since_52w_high`：状态 = 「最近一次创 250 日新高的行号」，warmup 原来给 480 天
     （≈320 交易日 < 需要的 500）⇒ **每年年初的计数被钉在 249**（假的"250 日没创过新高"）。
     手工真值：`000001.SZ@2016-01-04` 真值 **141**，旧版 **249**，warmup=960 版 **141** ✔。
   - 症状曲线很典型：**1 月差异最大（91.5% 的股票）、到 12 月归零** —— 因为当年真创一次新高后状态就恢复。
   - ⚠️ **按年锚定之后才暴露**（全局锚定下面板天然够长）。凡「按年锚定 + 状态型因子」都要按这条复查。
   - 判定工具：`scripts/check_year_boundary.py`（年初 churn 离群，先看**绝对值**大不大，别只看倍数）
     → `scripts/check_anchor_warmup.py <因子> --years YYYY`（A/B）或 `--scaled-warmup 2`
     → **手工真值核对**（按定义从 `close × adj_factor` 自己算一遍，这是唯一能定案的）。
   - 同族排查过、**不受影响**的：`consecutive_limit_up/down`、`limit_board_streak_mean_60`、
     `_run_length`/`_run_len`（状态随 flags 翻转归零，连续段通常 < 30 天）；
     `dividend_yield_3y_avg`/`etp5` 的 float64 前缀和（窗口差分，只依赖 t−n..t）。

1. **★ 产物行不得早于声明起点：合并语义会留残留**（2026-09-17 实测，`check` 抓到）：
   `store.upsert_year` 的语义是「用新数据替换同键旧行、其余原样保留」⇒
   **因子起点后移**（如 `dividend_yield_3y_avg` 从默认 2012 改成实测的 2013-01-11）之后重建，
   分区里**起点之前的老行永久残留**（实测 2013 分区有 5 天 / 17,420 行 NaN）。
   已修：`upsert_year` 增加 `prune_before`（与 `prune_after` 对称），`Engine.run_year` 传
   `max(spec.start_int, default_start)` —— **新因子/改起点后重跑一次即自愈**，不必手工删行。

2. **★★ `main.py run --rebuild --start` 会吃掉历史 `coverage`（2026-09-17 实测，走
   「全量重建 2026 → 删最新一天 → 增量」流程时踩到）**：`Engine.plan` 里
   ```python
   if rebuild or man.recipe != recipe:
       man.reset(recipe)          # ← --rebuild 本身就会清空 coverage 与 partitions
       spans = [(start_i, end_i)] # ← start_i 被 --start 限制成那一年
   ```
   于是 manifest 里**只剩本次跑的那一段**。后果：`Engine.plan` 以为更早的历史全缺 ⇒ 跨 ≥3 年
   ⇒ `main.py` 的**多年守卫把全部因子摘出**（原话：「本次要跑的因子全被摘出，没有可执行的活」）。
   ⚠️ **数据一行不会丢**（`year=20xx` 分区都在磁盘上），坏的只是 `state/*.json` 的
   `coverage` / `partitions` 两项（`recipe` / `input_watermark` / `last_run` 都保留）。
   - **修法**：`scripts/rebuild_manifest.py` —— 扫 `data/factors/<因子>/year=*/data.parquet`
     按分区实际内容重建那两项。实测 230 个因子全部恢复成 15 年 / 12,444,848 行，
     与出问题前的 `main.py status` **逐字相同**（约 7.6 min）。
   - **要「只重建某一年」就别用 `--rebuild --start`**：改用
     `scripts/backfill_history.py --from-year Y --to-year Y`（靠 `missing_ranges` 逐段补，
     **不 reset**）；或接受 reset、跑完立刻 `rebuild_manifest.py` 修回。


1. 财报按 `end_date` 分区，但 `ann_date` 可滞后 **15 个月以上**（实测 2024-12-31 的报告
   2026-03-20 才公告）→ 读分区要开 `[年份-3, 年份]` 窗口。
2. 财报是**累计 YTD**，不是单季。TTM 用三行式 `cum_q(y) + cum_FY(y-1) - cum_q(y-1)`，
   **Q1 必须单独处理**（否则 `cum(Q1) - cum(上一年FY)` 会算出负数）。
3. 银行/券商的 `oper_cost` 是**精确的 0.0 而不是 NaN** → 不设防的话毛利率恒为 100%，
   所有金融股会挤在截面顶端。已在 deriv 层 `<=0 → NaN`。
4. `revenue` ≠ `total_revenue`（实测茅台 FY2025：1688.4亿 vs 1720.5亿）。统一用 `revenue`。
5. `merge_asof` **拒绝字符串日期**；且复合键 as-of 必须**校验同股同期**，否则会静默串值。
6. `Series.to_numpy()` 可能是**只读视图**，就地赋值会抛
   `assignment destination is read-only`；同理 DataFrame 的 rank 结果。
7. 衍生层的面板缓存必须**带上面板日期范围**做键：不同因子 warmup 不同、增量窗口更短，
   同一个 Derivative 会被不同形状的 Panel 复用。
8. **绝不用 `pkill -f`**（会杀掉执行它的 shell 自己，实测 exit 144）；用
   `ps -eo pid,args | grep '[m]ain\.py'` 找 PID 再 `kill`。
11. **并行前必须 `up.clear_cache()`**。`fork` 是写时复制的，但 Python 的引用计数会在
    **读取**对象时改头部字段，从而触发整页复制。上游财报缓存有好几百 MB（6 年 × 3 张表），
    8 个 worker 各复制一遍，光 COW 开销就比省下的计算还贵——实测曾让并行比串行还慢
    （29s vs 17s）。衍生层建好后原始表就不再需要，直接清掉。
12. **并行只提速「计算」，不提速「固定成本」**。引擎初始化 + 衍生层构建（约 5~9s）
    是每个 run 都要付的，且不可并行（单进程内）。所以任务很少时并行的边际收益有限，
    `--jobs` 默认取 `min(8, 核数-1)`，可用 `--jobs 1` 退回串行。
9. 单实例保护的匹配要限定到**本模块目录**：模块① 也叫 `main.py` 且常年有 `run` 在跑。
10. 改 `conf/config.yaml` 里的 universe/winsor 会改变历史因子值 → 已并入 recipe 指纹，
    会自动触发全量重建（这是有意的，别绕过）。
13. **★ 滞后表名单（2026-09-15 深夜定稿：**只剩 2 张**，由 `fea/delay.py` 运行期装填）**：
    **`stock_margin_detail`**（晚 1 个交易日）、**`stock_report_rc`**（晚 2 个交易日）。
    用它们做因子必须 `lagged_ok=(...)` 声明 + **结果整体下移**（`ctx.lag_grid(grid, n)` 放最后一步），
    否则注册期直接抛错（详见上面「第二个 PIT 陷阱」）。
    ⚠️ 名单**以运行期为准**：`python -c "from fea.delay import load; print(load())"` ——
    文档这张表会随上游变化而过期（2026-09-15 一晚就有 7 张表被证伪移出）。
    ⚠️ `stock_st_info` 还额外有个坑：它参与 `universe` 的 ST 剔除，却不在任何因子的
    `deps` 里 → 上游新增 ST 记录**不会触发重算**（下次全量重建才会改，属静默不一致）。
15. ~~**★ `holder_number_chg` 违反 PIT 红线（2026-09-16 实测确认，待修）**~~ →
    **✅ 2026-09-17 已修**（原记录保留在下方，作为"这类 bug 长什么样"的标本）：
    改为 **as-of 状态机**（逐条公告推进「最新两期」，每条公告只看得到它自己那天之前的信息），
    `version` 1→2。**验证**：① 带阳性对照的 PIT 测试 —— 用截断数据（`ann ≤ 09-10`）
    重跑，旧实现有 2 处值不同 + 24 处缺失，新实现 **0 处不同 / 0 处缺失**；
    ② 人工逐日手算核对 5 个变化点全部一致；③ 连续两次重建逐位相同（幂等）。
    ⚠️ 历史值因此**整体重算过一次**（6.06% 的格子与修复前不同 —— 那正是旧实现里
    "用未来才披露的修正值改写历史"的部分）。
    原记录：

    **★ `holder_number_chg` 违反 PIT 红线（2026-09-16 实测确认，待修）**：
    它的 `drop_duplicates(["stock_code","end_date"], keep="last")` 是**对全表**做的，
    而 `stock_holder_number` 的**同一个 `end_date` 要跨越约一周、由不同公司陆续披露**
    （实测 `end_date=2026-09-10` 有 **8 个不同 `ann_date`**，09-09~09-16，每天约 150 条）。
    于是新公告一到就挤掉旧公告、并把自己的 `ann_date` 盖上去；若该 `ann_date > T`，
    记录被 `asof_daily` 排除，T 日**回退到上一个报告期的值** —— 同一 T 在不同日子跑出来不同。
    **实测**：沙箱重算（含 09-16 公告）vs 生产（19:37 构建、数据止于 09-15）
    → **72 格 / 9 天不一致**（09-15 有 36 格，09-14 16 格），差异最大到正负号相反
    （`000514.SZ` 09-15：`0.0032` ↔ `-0.0901`）。占 59.6 万格的 0.012%。
    ⚠️ **无下游因子依赖它**，所以影响面目前限于它自己。
    ⚠️ **修好之前每次重建都会改写这批历史值** —— 要冻结就先从重建清单里摘出来。
    **修法方向**：`drop_duplicates` 必须**按 T 做 as-of 去重**（每个 T 只用 `ann_date ≤ T`
    的最新公告），而不是对全表取最新；等价做法是保留全部公告让 `asof_daily` 自然对齐，
    再把"上一期"的链接建立在 as-of 结果上。**建议单开一轮 + 用 `audit-pit` 验证。**
    （上游表现：公告滞后中位数 16 天、90 分位 83 天、最长 1098 天；24.65% 超 30 天。）
16. **★ 耦合因子（`ctx.load_factor`）的两条约定**：
    ① 父因子名必须写进 `deps` —— 引擎靠它做**失效传播**
    （`fea/engine.py::_factor_watermark` 把父因子的 manifest 当成它的水位）；
    ② `main.py run` 会**分两趟执行**：第一趟全部非耦合因子，等它们落盘后第二趟才跑耦合因子
    （`FactorIO.load` 读的是产物文件，跑在父因子前面会**静默拿到全 NaN**，只 warning 不报错）。
    ③ 若父因子本身是滞后表做的（如 `margin_velocity`），**不要重复 `lag_grid`** —— 父因子的
    滞后处置已经被继承，再移一次等于白丢一天信息。
15. **★ 内存是硬约束（用户 2026-09-17 再次点名「注意别超内存」）**：
    - **上限以 `/sys/fs/cgroup/memory.max` 实测为准**：2026-09-17 实测 **60 GiB**
      （旧文档写的 120 GB 是上一台实例的值，**已过期**；无 swap，超了直接 OOM-Kill）。
    - 两个放大点：① `Derivative._panel_cols` 的面板缓存
      （每条 (T,C) float32 ≈ 10 MB × 字段数 × 任务数，`fea/deriv.py::trim_cache` 是唯一闸门）；
      ② **价格层**（全历史窗口实测 7.1~8.1 GB/进程 —— 2026-09-17 换了面板锚点后降到 2~3 GB）。
    - 实测**16 个 worker 会顶到 108 GB 被 OOM**（表现是 `BrokenProcessPool`，不是 MemoryError）。
      **当前默认 `--jobs 1`，最多 2；同时观察共享服务器的实际内存。**
    - ⚠️ **进程级缓存只扩不缩**（`PriceLayer._ensure_loaded` / `DerivedCache._read_all`）：
      一次请求跨多年时，每个 worker 手里会攒下全历史。`main.py run` 现在会在请求跨度 ≥2 年时
      打印内存提示；**全历史重建请用 `scripts/backfill_history.py` 按年分块**。
    - ★★ **真正决定内存的是「计划里跨几年」，不是「请求的日期范围」**（2026-09-17 亲历 OOM）：
      实测「**2 个因子 × 14 年**、jobs 4」在**第 27 个任务上被 OOM-Kill 两次** ——
      同一个进程从 2013 一路算到 2026，价格层窗口跟着年份**只扩不缩**，
      每个 worker 手里攒下全历史。**现在的防线有三条**：
      1. `main.py run` 的**多年守卫**：计划里跨 ≥3 年的因子会**从本次运行摘出**并打印
         `scripts/backfill_history.py --factors <它们>`（不会连累其它因子的每日增量）；
         确实要单进程硬跑：`--allow-multiyear`（内存自负）。
      2. **按年分块 = 每块一个新进程**：`scripts/backfill_history.py [--factors A B]`
         —— 补历史（含"新因子缺历史"）**一律走它**，别直接 `main.py run A B`。
      3. 单块并行度按年份给：全量 230 因子的块在 jobs=5 下，2012 块 36.6 GB、
         **2023/2024 块 58.0 GB（离上限只剩 2 GB）** ⇒ 后段年份 **jobs ≤ 4**。
      ⚠️ 爆内存的表现是 **`BrokenProcessPool`**（worker 被内核杀掉），**不是** `MemoryError`
         —— 别当成"偶发故障"重试，重试只会再爆一次。
      ⚠️ `main.py dayhash` 的并行度走的是 `resources.safe_jobs()`：**只能 1 或 2**
         （2026-09-22 核对；文档此前写的"默认 fork min(16, 核数-1)"是 09-16 的旧行为，已不成立）。
         即便只有 1~2 个 worker，它仍要读全库因子的尾部窗口，**别和别的重活同时跑**
         （2026-09-17 那次 OOM 就发生在 dayhash 与 2 因子重算并行时）。
    新因子尽量声明 `fin_fields`（不声明 = 该因子触发"全字段"衍生层）。
16. **去糟粕是常规动作，不是一次性清理**：`main.py dedup` 出冗余簇、
    `main.py eval` 出弱因子，两者交叉验证后再决定删/留。
    删除必须**三层一起**（源码注册 + `data/factors/<名>` + `state/<名>.json`），
    只删数据不删注册 = 下次 run 原样算回来。工具：`scripts/prune_factors.py`
    （默认 dry-run，先写 `docs/FACTOR_TRIAGE.md` 存档再删）。

17. **★★★ `stock_daily.stock_name` 不是历史名称 —— 绝不用它判 ST / 风险 / 分类**
    （2026-09-18 实测，差点据此生成错误的股票池名单）。
    厂商把**整段历史**的 `stock_name` 都改写成**当前**名称。实例 `000004.SZ`：
    `stock_st_info` 说它 **2022-05-06** 起才是 `ST国华`，而 `stock_daily` 里
    **全部 3724 行的名称都是 `国华退`**（当前处于退市整理期的名字）。
    - 后果：用名称判"曾经 ST"只找到 **284** 只，官方口径 `stock_st_info` 有 **784** 只，
      **漏 502 只** ⇒ 池子会多算 308 只。
    - **正确来源**：判 ST 用 `stock_st_info`（`type='ST'` / `type_name='风险警示板'`）；
      判板块用**代码前缀**（不依赖名称）；判上市/退市用 `stock_list` 的
      `list_date` / `delist_date`。
      ⚠️ `stock_st_info` 覆盖**从 2016-08-09 起**，更早的 ST 无法判定（已知盲区）。
    - 这也解释了单日 MD5 台账里反复出现的 `变化列: stock_name` —— 是厂商在改写历史名，
      **不是我们的数据出错**。
    - 全库已复查：20 个家族文件**无一处读名称列**（`stock_name`/`name`/`act_name` 全空），
      当前无落点；此条是"将来别用"的约定。

18. **★★ 股票池已冻结为固定 2115 只（2026-09-18 用户拍板）—— 别再按"动态池"理解**
    - 口径：主板 ∩ **至今存续** ∩ **从未ST(2016-08-09起)** ∩ 上市日 ≤ 2018-01-01。
      名单 `conf/universe_frozen.tsv`，生成器 `scripts/build_frozen_universe.py`
      （`--check` 可复核名单是否仍成立）。
    - **动因**：动态池下**一只新股上市就让全库历史截面全红** —— 2026-09-18
      `601091.SH` 入池 → 每个历史截面 +1 行 → 面板大小与 rank/cs_zscore 平移
      → 单日 MD5 台账 2177 条全变。
    - ⚠️ **用户明确接受的代价**（别当 bug 报）：① 要求"至今存续" ⇒ **幸存者偏差**，
      回测收益被系统性高估；②"从未ST"是事后判据；③ ST 历史最早只覆盖到 2016-08-09，之前无法判定。
      ⇒ **结论（IC / 净值）不可跨这次变更前后比较**，做 A/B 必须两侧同池。
    - 名单内容哈希进 `universe_fp`（`fx=<sha1>:n=2115`）⇒ 改名单即触发全量重建。
      置空 `conf/config.yaml:universe.frozen_list` 可回到动态池（`fx=off`）。
    - `fea/universe.py::code_master` 对名单外代码**直接报错**，不静默。

<a id="source-881144153522-与其它文档的关系"></a>
## 与其它文档的关系

- 全平台主记录：`/root/autodl-fs/QUANT_PLATFORM.md`（第 0~2 节是给冷启动接手写的）
- 因子字典：本目录 `FACTORS.md`（由 `main.py docs` 生成）
- 分诊存档：本目录 `docs/FACTOR_TRIAGE.md`（2026-09-15 删掉的 100 个因子及其理由与公式）
- 参考因子库（数据商提供，859 个因子的定义与公式）：本目录 `factors.md`、`因子库.md`
<!-- SOURCE:88114415352273535be51210a061520f733792faa687671cbdbfe597d1c772df:END -->
<a id="dev-contract"></a>
## 因子开发契约（`factors/DEVELOPING.md` 正文 · 现行）

> 本节是原 `factors/DEVELOPING.md` 的**现行版本**，2026-09-20 从折叠块里取出、直接展开。
> **写新因子前必读**：六条硬约束、因子函数模子、`ctx` 的全部能力清单、
> 上游 delay、`ctx.ind()` 字段范围、`note=` 要求、提交前自检、汇报格式。
>
> ⚠️ 文中 §4 的标题写「4 张表晚一个交易日」，**与现行不符** —— 现行只有
> `stock_margin_detail` 一张表滞后 1 个交易日（见 `datadownload/conf/frequency.yaml`）。
> 旧版原文（确实写 4 张表的那份）已移到 [`HISTORY.md`](HISTORY.md)，
> 本条差异也已记入体检报告的 P2-4。

<a id="source-4dac550d90a4"></a>
<!-- SOURCE:4dac550d90a4656e9198ba2481636ac4f8897f1699f4240528864f5d7767866d:BEGIN -->
<a id="source-4dac550d90a4-因子开发契约写因子前必读逐条照做"></a>
# 因子开发契约（写因子前必读，逐条照做）

> 本文件是**多 Agent 并行开发**的规范。每个 Agent 拿到一个家族文件 + 本契约，
> 按契约写完自检、自测、汇报。**违反契约的代码会在收口时被打回。**

---

<a id="source-4dac550d90a4-0-你只拥有一个文件"></a>
## 0. 你只拥有一个文件

**只创建/修改 `factors/<你的家族>.py` 这一个文件。**

禁止改：`fea/**`、`conf/**`、`main.py`、`factors/__init__.py`、其它家族的 `.py`。
（`factors/__init__.py` 的 import 由主 Agent 统一加，避免并发写冲突。）

---

<a id="source-4dac550d90a4-1-六条硬约束用户拍板违反即废"></a>
## 1. 六条硬约束（用户拍板，违反即废）

| # | 约束 |
|:--|:--|
| 1 | **全部因子日频** —— 对齐 `(trade_date, stock_code)` 后落盘，不允许周频/月频/稀疏频率 |
| 2 | **固定主板股票池 2115 只** —— 按 `conf/universe_frozen.tsv`，不自行增删股票 |
| 3 | **输出起点** 由 `conf/config.yaml: default_start` 控制（当前 `2018-01-01`；实际取全局下界与显式起点的较大值）；<br>受上游起点限制的因子**显式写** `start=` |
| 4 | **禁止前复权(qfq)** —— 一律用 `ctx.hfq(...)`（= 未复权价 × 截至当日的 `adj_factor`） |
| 5 | **注意上游 delay** —— 当前仅两融明细晚 1 个交易日，见 §4 |
| 6 | **统一格式** —— 引擎强制 4 列 `trade_date/stock_code/value/rank`；**不许自定义输出列** |

---

<a id="source-4dac550d90a4-2-因子函数的形式照抄这个模子"></a>
## 2. 因子函数的形式（照抄这个模子）

```python
from __future__ import annotations

import numpy as np
from fea.spec import FactorSpec, register


@register(FactorSpec(
    name="momentum_20",                 # 全局唯一，snake_case
    group="momentum",                   # 家族名
    deps=("stock_daily", "stock_adj_factor"),   # 用到的上游数据集（水位失效判定用）
    desc="20 个交易日的后复权收益率",       # 一句话
    formula="Mom20 = hfq_close(T) / hfq_close(T-20) - 1",   # ★ 逐字抄参考库
    start=None,                         # ★ None = 跟随 default_start
    warmup_days=60,                     # 「计算窗口要往前多读多少**日历天**」
    higher_is_better=True,              # 只影响文档方向标注，不影响计算
    note="★ 用后复权收盘价；停牌日持有收益允许为 0；需成交日收益时使用 ctx.ret_clean",
))
def momentum_20(ctx):
    return ctx.ret(20)
```

<a id="source-4dac550d90a4-三条铁律"></a>
### 三条铁律

1. **签名必须是 `def name(ctx):`**，**返回值必须是 `(T, C)` 的数组**
   （`T = ctx.panel.T`，`C = ctx.panel.C`）。
   ⚠️ **返回 `(C, T)` 是最危险的错误**：元素数相同，以前会被引擎静默转置、产出
   「看起来完全正常」的错面板。现在引擎会直接报错，但请你自己别写错。
2. **缺失值一律用 `np.nan`**。不许用 `0` / `-1` / 任意哨兵值。
   补 0 会让滚动均值、截面排名全部失真。
3. **不许在因子里做 winsor / rank** —— 引擎统一做，返回原始值即可。
   **不许返回 DataFrame**。

<a id="source-4dac550d90a4-warmup_days-怎么给给少了每次增量都会在窗口头部产出一年左右的-nan"></a>
### `warmup_days` 怎么给（给少了每次增量都会在窗口头部产出一年左右的 NaN，
而且**看起来只像「因子有点噪」**）

| 因子类型 | warmup_days |
|:--|:--|
| 财务 / TTM / 同比 | **700**（4 季 TTM + 4 季同比 + ann_date 最长滞后 15 个月） |
| 纯日频滚动，窗口 N 个交易日 | `N × 1.8 + 20`（交易日→日历天要乘 7/5，再留冗余） |
| 250 日动量 | **480 以上** |
| 日内派生（日频聚合） | 60 |

---

<a id="source-4dac550d90a4-3-ctx-的全部能力不在这个列表里的-api-都不存在别编"></a>
## 3. `ctx` 的全部能力（**不在这个列表里的 API 都不存在，别编**）

<a id="source-4dac550d90a4-31-价格与收益"></a>
### 3.1 价格与收益

| 方法 | 说明 |
|:--|:--|
| `ctx.px(field)` | 未复权价量：`open/high/low/close/pre_close/vol/amount/pct_chg`。<br>停牌日 `vol/amount` 是 NaN（不是 0）；`close` 这类水平量会自动前向填充 |
| `ctx.hfq(field="close")` | **后复权**价 = 未复权价 × 截至当日 `adj_factor`。**价格一律用它** |
| `ctx.ret(k=1)` | k 个交易日的后复权持有收益；停牌期间价格可沿用。成交日统计用 `ctx.ret_clean(k)`；单日异常复权收益有独立过滤 |
| `ctx.traded()` | `(T,C) bool`，当日真有成交 |

<a id="source-4dac550d90a4-32-财务"></a>
### 3.2 财务

| 方法 | 说明 |
|:--|:--|
| `ctx.ttm(field)` | 累计制科目 → **TTM**，已按 `ann_date` 前向填充到日频 |
| `ctx.lag_ttm(field, k=4)` | TTM 的「k 个报告期前」的值（同比用） |
| `ctx.point(field, lag=0)` | 时点科目（资产负债表）→ 日频 |
| `ctx.ind(field, lag=0)` | `stock_financial_indicator` 的**时点比率**（见 §5 的字段范围） |

**可用的 `ttm()` 字段**（其余字段不在版本表里，用会 `KeyError`）：
`revenue, oper_cost, n_income_attr_p, operate_profit, total_profit, n_income,
sell_exp, admin_exp, fin_exp, rd_exp, assets_impair_loss, invest_income,
non_oper_income, non_oper_exp, income_tax, minority_gain, total_cogs,
fin_exp_int_exp, fv_value_chg_gain, ebit, n_cashflow_act, n_cashflow_inv_act,
n_cash_flows_fnc_act, c_fr_sale_sg, c_paid_goods_s, c_paid_to_for_empl,
c_paid_for_taxes, c_pay_acq_const_fiolta, depr_fa_coga_dpba, amort_intang_assets,
lt_amort_deferred_exp, credit_impa_loss, prov_depr_assets, free_cashflow,
c_pay_dist_dpcp_int_exp, decr_inventories, invest_loss, n_incr_cash_cash_equ,
recp_tax_rends, c_recp_borrow, c_prepay_amt_borr`

**可用的 `point()` 字段**：
`total_assets, total_liab, total_hldr_eqy_exc_min_int, total_hldr_eqy_inc_min_int,
defer_tax_assets, total_share, minority_int, money_cap, accounts_receiv,
notes_receiv, accounts_receiv_bill, inventories, prepayment, oth_receiv,
contract_assets, total_cur_assets, total_cur_liab, total_nca, total_ncl,
fix_assets, cip, cip_total, goodwill, intan_assets, r_and_d, lt_borr, st_borr,
non_cur_liab_due_1y, bond_payable, acct_payable, notes_payable, accounts_pay,
adv_receipts, contract_liab, defer_tax_liab, defer_inc_non_cur_liab,
treasury_share, trad_asset`

> ⚠️ 若你需要一个**不在这两个列表里**的字段 → **不要自己读 parquet**，
> 在汇报里说明「需要引擎扩字段」，由主 Agent 统一加进 `fea/deriv.py`。

<a id="source-4dac550d90a4-33-通用数据接入"></a>
### 3.3 通用数据接入

| 方法 | 说明 |
|:--|:--|
| `ctx.dataset(name, columns, years)` | 读任意上游表 → **原始 DataFrame**（不在网格上） |
| `ctx.asof_daily(codes, days, values)` | **状态量**：稀疏披露值 → 日频前向填充 |
| `ctx.event_grid(codes, days, weights)` | **事件**：稀疏事件 → 日频散点累加（缺失补 0） |
| `ctx.code_index(codes)` | 股票代码 → 列下标（不在池子里返回 -1） |
| `ctx.date_col(series)` | `"YYYY-MM-DD"` 字符串列 → int32 YYYYMMDD |

> **状态量 vs 事件量的选择**（选错会静默毁掉因子）：
> 净资产、总股本这类**状态量**用 `asof_daily`；
> 今天涨停了没有这类**事件量**用 `event_grid`。
> 用 asof 处理事件 → 退化成「0 或者永远的陈旧值」。

<a id="source-4dac550d90a4-34-数学全部-tc-进-tc-出"></a>
### 3.4 数学（全部 `(T,C)` 进 `(T,C)` 出）

```
位移：  ctx.shift(mat,k)  ctx.diff(mat,k)  ctx.pct_change(mat,k)
        （k>0 取过去，k<0 取未来 —— 只有标签会用到 k<0）
滚动：  ctx.roll_sum/mean/std/var/max/min/count/skew/kurt(mat, n, min_count=None)
        ctx.roll_rank(mat,n)          Ts_Rank：窗口最后一个值的百分位
        ctx.roll_argmax/argmin(mat,n)  极值距今天多少个 bar
        ctx.roll_corr(x,y,n)  ctx.roll_cov(x,y,n)
        ctx.ewm_mean(mat, span=k)     ctx.decay_linear(mat,n)
截面：  ctx.cs_demean(mat)  ctx.cs_zscore(mat)  ctx.cs_winsor(mat)
其它：  ctx.signed_power(mat,p)  ctx.safe_log(mat)  ctx.safe_sqrt(mat)
        ctx.safe_div(num, den, min_abs_den=0)   ← 除法**一律**用它
```

**NaN 策略**：窗口内只要出现过 NaN，结果就是 NaN（与 `roll_sum` 一致）。
`min_count=k` 是**显式放松**（窗口内至少 k 个有效值）—— 用了要在 `note` 里写理由。

<a id="source-4dac550d90a4-35-只属于特定家族的能力"></a>
### 3.5 只属于特定家族的能力

| 方法 | 用在 | 说明 |
|:--|:--|:--|
| `ctx.intraday_field(f)` | `intraday.py` | 5min 预聚合出的日频字段（见 `fea/intraday.py` 的字段表） |
| `ctx.chip(f)` | `chips.py` | 筹码峰预聚合出的日频摘要（见 `fea/chips.py` 的字段表） |
| `ctx.load_factor(name)` | `coupling.py` | 读另一个已落盘因子的 value，切到当前面板 |
| `ctx.lag_grid(mat, 1)` | 滞后表家族 | 见 §4 |
| `ctx.next_trading_day(days)` | 滞后表家族 | 日期位移法 |

---

<a id="source-4dac550d90a4-4-上游-delay两融明细晚一个交易日"></a>
## 4. 上游 delay：两融明细晚一个交易日

当前生效名单只有 `stock_margin_detail: 1`。已删接口不应重新加入；其他表不能照抄旧名单。
`stock_margin_detail` 的 `trade_date = d` 的数据，**要到 d 的下一个交易日才拿得到**。

不处理的话：**T 日的因子值今天算对、明天静默变成另一个值**，而且**没有任何报错**。

**处置**（二选一，推荐第 1 个）：

```python
# 1) 行位移（推荐）：面板本身就是交易日历，移一行 = 移一个交易日
@register(FactorSpec(
    name="margin_buy_pressure", deps=("stock_margin_detail",),
    lagged_ok=("stock_margin_detail",),    # ★ 必须显式声明，否则 register() 直接抛错
    ...))
def margin_buy_pressure(ctx):
    ...
    return ctx.lag_grid(grid, 1)           # ★ 整体下移一行

# 2) 日期位移
days_ok = ctx.next_trading_day(days)
```

⚠️ **别把这些误判成滞后**：`stock_report_rc`（研报不定期）、`stock_dragon_tiger` /
`stock_top_list`（上榜家数每天不同）、`stock_limit_up`（涨停家数 40~95 波动）、
`stock_kline`（周/月线）、全部季频/月频/不定期表。
判定方法：取最近 12 个交易日的行数，**「每天都有稳定行数」+「最后一天为 0」** 才是滞后。

---

<a id="source-4dac550d90a4-5-ctxind-的字段范围--只许用这些"></a>
## 5. `ctx.ind()` 的字段范围 —— **只许用这些**

`stock_financial_indicator` 里有 163 个比率，但**大多数是累计 YTD，不能当日频用**
（会得到跨季锯齿，振幅 4 倍）。只有下面这些是安全的：

> ★ 2026-09-18 起，**这份白名单已经被真正用起来了** ——
> `factors/fundamental3.py` 首次消费了 `currentdebt_to_debt` / `int_to_talcap` /
> `dt_netprofit_yoy` / `bps_yoy` 四个字段（该文件 docstring §一 有「为什么之前零消费」的复盘）。
> 白名单本身**没有变**，用别的字段仍然会 `KeyError`。

**2026-09-21 白名单增量**：经上游接口字典核对，`q_roe, q_dt_roe, q_npta, q_ocf_to_sales, q_sales_yoy` 作为明确的单季度字段加入 `IND_QUARTERLY`，可以通过 `ctx.ind(field, lag=k)` 使用。累计 YTD 字段仍禁止直接使用；季度统计按报告期滞后，不能按日频平移。五字段同时为零的源表占位保护在 `quarterly_quality.py` 内执行，不改变其他家族。

**时点比率**（`end_date` 上的瞬时量，可直接用）：
`debt_to_assets, current_ratio, quick_ratio, cash_ratio, assets_to_eqt, ca_to_assets,
nca_to_assets, tbassets_to_totalassets, int_to_talcap, currentdebt_to_debt,
longdeb_to_debt, debt_to_eqt, tangibleasset_to_debt, ebitda_to_debt, turn_days`

**同期同比**（分子分母都是同期 YTD，季节性自动抵消）：
`netprofit_yoy, or_yoy, tr_yoy, op_yoy, ebt_yoy, roe_yoy, ocf_yoy, assets_yoy,
eqt_yoy, bps_yoy, dt_netprofit_yoy`

**绝对不许直接用的**（必须用 `ctx.ttm()` 从原始三表重算，或干脆不做）：
`roe, roe_waa, roa, roic, npta, grossprofit_margin, netprofit_margin, op_of_gr,
ocf_to_or, inv_turn, ar_turn, ca_turn, fa_turn, assets_turn, rd_exp, ...`

---

<a id="source-4dac550d90a4-6-每个因子必须写-note"></a>
## 6. 每个因子必须写 `note=`

记录：**实测坑**、**与参考库文档的偏离**、**口径选择及其理由**。
这份 note 会进 `FACTORS.md`，是下游建模时判断「这个因子能不能信」的唯一依据。
参考现有 `factors/fundamental.py` 的写法（例如「银行/券商的 oper_cost 是精确 0.0 而不是 NaN」）。

---

<a id="source-4dac550d90a4-7-自检提交前必须全过"></a>
## 7. 自检（提交前必须全过）

```bash
PY=/autodl-fs/data/miniconda3/bin/python
cd /root/autodl-fs/featureengineering

# ① 注册通过（会跑 PIT 守卫：qfq 依赖 / 滞后表未声明 都会在这里抛错）
$PY -c "import factors; from fea.spec import REGISTRY; print(len(REGISTRY))"

# ② 沙箱冒烟（★ 必须带 --sandbox，否则会写生产 data/ 与 state/，和主 Agent 打架）
$PY main.py run --sandbox /tmp/sbx_<你的家族> --jobs 4 --end 2014-12-31 \
      <你的因子名1> <你的因子名2> ...

# ③ 逐因子看产出是否合理（空值率、值域、截面分散度）
$PY - <<'EOF'
import pandas as pd, glob
for f in sorted(glob.glob('/tmp/sbx_<你的家族>/factors/*/year=*/*.parquet')):
    d = pd.read_parquet(f)
    print(f.split('/')[-3], d['value'].describe().to_dict(),
          '非空%.1f%%' % (d['value'].notna().mean()*100))
EOF
```

**必须自查的四个数**（不达标就回炉）：
1. **非空率**：正常应在 60%~100%（受停牌/上市时间影响）。`< 30%` 说明口径错了。
2. **截面分散度**：每日有效股票数中位数应在 1500~3300（主板有 3484 只）。
3. **值域**：`|value| > 1e8` 的格子应当为 0（引擎 `check` 会报「值域异常」）。
4. **不是常数**：每日截面唯一值 >> 1。若恒为常数，说明依赖的字段是空的。

---

<a id="source-4dac550d90a4-8-参考库怎么用"></a>
## 8. 参考库怎么用

`学习资料/factors.md`（859 个因子，每个都有 `**定义** / **公式（计算逻辑）** / **意义** / **依赖数据**`）
和 `学习资料/因子库.md`（202 个因子，带数学表达式与参数）。

**做法**：按家族名去这两份文档里找对应因子 →
把 `公式（计算逻辑）` 里的代码块**逐字抄进 `FactorSpec.formula`** →
用本契约的 API 重新实现它 → 在 `note` 里写清与参考实现的偏离。

⚠️ 参考库的 `context.load("daily.parquet")`、`cross_sectional_rank(...)`、
`groupby(level="Code")` 这些**在本框架里都不存在**，要翻译成 §3 的 API。
⚠️ 参考库用的是 `daily_adj.parquet`（前复权）—— **我们不能照抄**，
必须换成 `ctx.hfq(...)`（后复权），这才是 PIT 安全的。

---

<a id="source-4dac550d90a4-9-汇报格式交给主-agent"></a>
## 9. 汇报格式（交给主 Agent）

```markdown
## 家族：<name>.py   因子数：N

### 因子清单
| 因子 | group | 公式 | deps | warmup | start | 参考库出处 |
|---|---|---|---|---|---|---|

### 自检结果
- 注册通过：✓（REGISTRY 共 XX 个）
- 沙箱冒烟：`main.py run --sandbox ... --end 2014-12-31` ✓ / 各因子行数
- 非空率 / 截面分散度 / 值域：
  | 因子 | 非空率 | 日均截面 | \|value\|max |

### 与参考库的偏离（逐条给理由）
### 需要引擎扩展的地方（若有）
```


<a id="source-4dac550d90a4-2026-09-19-维护要求"></a>
## 2026-09-19 维护要求

新增因子通过注册表和 `main.py` 统一运行。日常 `python main.py` 默认串行；历史构建使用 `python main.py rebuild --jobs 1`，最多 2 个计算进程。详细操作和本轮变更见 `../docs/2026-09-19_统一日期与数据对齐.md`。

字段扩展必须先核对单位、公告可得日、历史覆盖和异常值。价格与供应商未复权成本比较时使用同口径原始价格；不要对缺失的事件或公告随意补零。新公式应明确标注参考公式或本地扩展，不将未经验证的经济方向当成收益结论。新增指标必须做历史截断重算、旧日期数据修订传播和稀疏覆盖检查。财报必须使用财务衍生层，按各报表独立的公告版本取值，不能事先只留最终版本。PIT 检验需要过滤未来公告源记录；只缩短价格面板不能检出所有财报修订问题。
<!-- SOURCE:4dac550d90a4656e9198ba2481636ac4f8897f1699f4240528864f5d7767866d:END -->

<a id="automatic-reports"></a>
## 后续自动检查报告

辅助检查默认追加到此区；分诊记录是执行前计划，不能作为已删除或删除成功的凭据。

<!-- FEA:REPORTS:BEGIN -->
# 因子分诊报告（去糟粕）· 2026-09-22 本轮（人工合并成一节）

> 依据 = `main.py dedup` 的全量实测（**656 因子 × 2018~2026、|ρ|≥0.95**，2026-09-22 09:14 完成）：
> **37 簇 / 58 个候选**。★ 实际删除 **57 个**，保留 1 个（父依赖，见下）。
> · 完整簇清单 / 代表清单：`artifacts/audits/dedup_20260922/{REPORT.md,representatives.json}`
> · 逐因子理由与摘除路径：`scripts/prune_factors.py` 的 `DELETE`（2026-09-22 段）
> · 删除前全量备份：`artifacts/backups/prune_20260922/`（58 产物目录 + 58 状态文件 + 源码/配置）
> · 删除后：注册对象 661 → **604**（因子 656 → **599**，标签 5 不变）；`main.py docs` 已刷新本手册的字典

## 删除结果（按摘除路径 —— 四种注册机制各走各的）

| 摘除路径 | 数量 | 哪些 |
|:--|--:|:--|
| `conf/field_expansion.json` 删条目 | 39 | `afx_*`（该 JSON 就是这族的注册源） |
| 并入文件自带的 `REJECTED_CANDIDATES` | 12 | `efx_*` 3 个（field_events.py）、`mfx_*` 9 个（field_markets.py） |
| AST 摘 `@register` 块 | 6 | chips.py 2、cyq_perf.py 2、valuation.py 1、volatility.py 1 |
| 删产物目录 + 状态文件 | 57 + 57 | 全部候选（保留的 1 个已还原） |

★ `scripts/prune_factors.py` 本轮补齐了两件事：**按机制分派摘除**（老实现只认 `@register(FactorSpec(name=...))`，
对 conf 目录驱动与 `REJECTED_CANDIDATES` 两族会「未定位到」却照样删产物 ⇒ 下次 `main.py run` 全部复活），
以及**依赖守卫**（见下节）。

## 保留 1 个：`momentum_60`（不是漏删）

`momentum_60` 与 `sortino_ratio_60` 的 |ρ|=0.953，但它同时是**两个存活耦合因子的父依赖**：
`cp_momentum_highvol_60`（`z(momentum_60)×z(vol_120)`）、`cp_quality_momentum`（`z(roe_ttm)×z(momentum_60)`）。
删掉父因子后子因子的 `ctx.load_factor` 会**静默**返回全 NaN（只 warning）⇒ 表现为「非空率 0%」而不报错。
同 2026-09-17 轮保留 `mf_big_order_ratio` / `short_term_reversal_5` 的道理 —— 本轮已把这条判据固化成
`prune_factors.py` 的**依赖守卫**（拦下即退出码 2，未改任何代码或数据）。

## 分簇对照（每簇留 1 个代表；`→` 左边是被删的）

| 簇 | 代表（保留） | |ρ| 范围 | 删除 |
|--:|:--|:--|:--|
| 10 | `afx_fi_npta` | [0.908, 1.000] | `afx_fi_roa`、`afx_fi_roa2_yearly`、`afx_fi_roa_dp`、`afx_fi_roa_yearly`、`afx_fi_roe`、`afx_fi_roe_waa`、`afx_fi_roe_yearly`、`afx_fi_roic`、`afx_fi_roic_yearly` |
| 6 | `afx_bs_oth_assets` | [0.957, 1.000] | `afx_bs_oth_liab`、`afx_is_n_commis_income`、`afx_is_n_oth_income`、`afx_is_oper_exp`、`afx_is_oth_b_income` |
| 5 | `afx_fi_ebit_of_gr` | [0.924, 0.998] | `afx_fi_netprofit_margin`、`afx_fi_op_of_gr`、`afx_fi_profit_to_gr`、`afx_fi_profit_to_op` |
| 4 | `mfx_dc_pressure` | [0.883, 0.988] | `mfx_minute_pressure`、`mfx_tdx_pressure`、`mfx_ths_pressure` |
| 4 | `mfx_dc_range` | [0.963, 0.998] | `mfx_dc_swing`、`mfx_tdx_range`、`mfx_ths_range` |
| 3 | `afx_fi_diluted2_eps` | [0.961, 0.997] | `afx_fi_dt_eps`、`afx_fi_eps` |
| 3 | `avg_cost_premium` | [0.943, 0.992] | `chip_resistance_distance`、`cyqp_average_cost_premium` |
| 2 | `afx_cf_beg_bal_cash` | [0.976, 0.976] | `afx_cf_c_cash_equ_beg_period` |
| 2 | `afx_cf_c_cash_equ_end_period` | [0.976, 0.976] | `afx_cf_end_bal_cash` |
| 2 | `afx_cf_im_net_cashflow_oper_act` | [1.000, 1.000] | `afx_cf_st_cash_out_act` |
| 2 | `afx_fi_basic_eps_yoy` | [0.985, 0.985] | `afx_fi_dt_eps_yoy` |
| 2 | `afx_fi_cogs_of_sales` | [0.998, 0.998] | `afx_fi_grossprofit_margin` |
| 2 | `afx_fi_ebit` | [0.989, 0.989] | `afx_fi_profit_prefin_exp` |
| 2 | `afx_fi_eqt_to_interestdebt` | [0.978, 0.978] | `afx_fi_tangasset_to_intdebt` |
| 2 | `afx_fi_fcfe` | [0.963, 0.963] | `afx_fi_fcfe_ps` |
| 2 | `afx_fi_fcff` | [0.974, 0.974] | `afx_fi_fcff_ps` |
| 2 | `afx_fi_n_op_profit_of_ebt` | [1.000, 1.000] | `afx_fi_nop_to_ebt` |
| 2 | `afx_fi_ocf_to_debt` | [0.952, 0.952] | `afx_fi_ocf_to_shortdebt` |
| 2 | `afx_fi_q_gr_qoq` | [0.997, 0.997] | `afx_fi_q_sales_qoq` |
| 2 | `afx_fi_q_netprofit_margin` | [0.998, 0.998] | `afx_fi_q_profit_to_gr` |
| 2 | `afx_fi_q_netprofit_yoy` | [0.951, 0.951] | `afx_fi_q_profit_yoy` |
| 2 | `afx_fi_retainedps` | [0.976, 0.976] | `afx_fi_undist_profit_ps` |
| 2 | `afx_fi_revenue_ps` | [1.000, 1.000] | `afx_fi_total_revenue_ps` |
| 2 | `afx_fi_roe_avg` | [0.966, 0.966] | `afx_fi_roe_dt` |
| 2 | `afx_is_basic_eps` | [0.982, 0.982] | `afx_is_diluted_eps` |
| 2 | `afx_is_compr_inc_attr_p` | [0.987, 0.987] | `afx_is_t_compr_income` |
| 2 | `bollinger_width_20` | [0.981, 0.981] | `market_cap_concentration_20d` |
| 2 | `chip_median_distance` | [0.957, 0.957] | `chip_support_distance` |
| 2 | `chip_position` | [0.957, 0.957] | `cyqp_price_cost_position` |
| 2 | `downside_upside_vol_60` | [0.952, 0.952] | `ret_skew_60` |
| 2 | `sp_ttm` | [0.974, 0.974] | `efx_annual_sales_yield` |
| 2 | `efx_top_amount_rate` | [1.000, 1.000] | `efx_top_participation` |
| 2 | `efx_top_imbalance` | [0.968, 0.968] | `efx_top_net_rate` |
| 2 | `mfx_dc_pct_change` | [0.988, 0.988] | `mfx_ths_pct_change` |
| 2 | `mfx_dc_turnover_rate` | [0.965, 0.965] | `mfx_ths_turnover_rate` |
| 2 | `mfx_minute_amplitude` | [0.959, 0.959] | `mfx_minute_realized_vol` |
| 2 | `sortino_ratio_60` | [0.953, 0.953] | `momentum_60` |

> ★ 覆盖率校核：`keep` 的排序依据（`state/eval/summary.json`）只覆盖 306 个旧因子 ⇒
> 新因子同簇时 `keep` 与质量无关；交付物 `REPORT.md` 单列了 9 簇「覆盖率最高成员」的差异（Δ ≤ 2.8pp）。

## 保留但标注低置信

- `ac1 > 0.995` 的财务/慢变量：单年只有约 4 个独立样本（季报），IC 统计上不可信；
  同时它们换手≈0，无法独立产生交易信号 —— 作为**风格暴露/控制变量**保留。
- `ac1 < 0.10` 的日内/资金流瞬时因子：日频全换手，必须按**成本后**收益复核。
<!-- FEA:REPORTS:END -->
