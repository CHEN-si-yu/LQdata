"""隔夜/日内收益分解 + K 线形态 + 缺口行为 因子（30 个）。

全部日频产出、只主板、跟随 `conf/config.yaml: default_start`，group = ``pattern``。

数据源只有**价格层**（`ctx.hfq` / `ctx.px` / `ctx.traded` / `ctx.ret` + `fea/mathx.py`）。
本文件不碰任何滞后表（§4 的 8 张表一张没用），所以**不需要** `lagged_ok`。

参考库：`学习资料/factors.md` 的同名条目（`fac_new_daily.py` / `price.py` /
`price_deep.py` / `fac_short_term.py` / `structure_patterns.py` / `trend_pattern.py` /
`technical_daily.py` / `event_dynamics.py` / `volume_price_dynamics.py` /
`momentum_structure.py`）。

## 写这一族必须记住的七件事（实测，不是推理）

1. **隔夜腿必须用「连续两个交易日都有成交」的掩码**。
   隔夜 = `hfq_open(T)/hfq_close(T-1) - 1`，两条腿都在**价格层的前向填充语义**之下
   （`fea/prices.py` 对 open/high/low/close 这些**状态量** ffill）。不掩码的话，
   停牌日的「隔夜收益」算出来是**上一个成交日自己**的日内涨跌 —— 一个凭空的假跳空。
   更重要的是**复牌日**：`hfq_close(T-1)` 是停牌前最后一个收盘价，于是复牌当天
   `overnight` = **整段停牌期间的累计涨跌**（重组复牌 +44% 那种），它不是隔夜溢价。
   本文件的隔夜腿统一用 `ok2 = traded(T) & traded(T-1)` 掩码，实测代价很小：
   主板「前一日无成交」的行占比 2013 = 0.68%、2015 = 1.08%、2020 = 0.49%、
   2026 = 0.64%（48.4 万 / 47.5 万 / 71.2 万 / 54.2 万行）。
   日内腿只看当日，用 `traded(T)` 就够（它不需要 T-1）。

2. **`ctx.px("pre_close")` 一律不用**。后复权比值 `hfq_open(T)/hfq_close(T-1)`
   与交易所前收（昨收 × 除权调整）**数学等价**，但走的是本项目唯一允许的
   「未复权价 × 累计复权因子」路径，不依赖供应商的除权口径（与 `factors/intraday.py`
   第 1 条同样的理由）。复权只影响**跨日**比较；形态族的日内比值（影线/振幅）
   在同一天内 adj 因子整体约掉，用不用 hfq 都一样 —— 但本文件统一走 hfq。

3. **日线 OHLC 的一致性实测是干净的**（与 5min 层相反，见 `factors/intraday.py` 第 9 条）。
   2013 / 2015 / 2020 / 2026 四年共 **3,037,791** 行：
   `high < max(open, close)` **0 行**、`low > min(open, close)` **0 行**。
   所以本文件**不做**区间扩张修复（那是 5min 层才需要的）。
   真正要挡的是另一件事：**`high == low` 的一字板**（振幅 0）——
   2013 = 1,024 / 564,208 行（0.18%）、2015 = 10,408 / 574,572（1.81%）、
   2020 = 10,812 / 963,888（1.12%）、2026 = 1,883 / 935,123（0.20%）。
   振幅 0 时 `|实体| ≤ 10%×振幅`、`(上影+下影)/振幅 < 0.1` 这些判据是 **0/0 型无定义**，
   参考库用 `.replace(0, np.nan)` 把它判成 False（NaN < 0.1 → False），本文件用
   `振幅 > 0` 显式把该日判成「不是该形态」（NaN 而不是 0，见第 4 条）。

4. **判据不成立 → `0.0`，判据不可得 → `NaN`，两者不许混**。
   停牌 / 未上市 / 前一日停牌这些格子**不是「没有出现该形态」**，是「当天没有数据」。
   混成 0 会把停牌多的股票的系统性拉低/拉高（`hammer_ratio_20d` 越低、
   `gap_up_fade_freq_20d` 越低），而这个偏差恰好与停牌频率相关，不是随机噪声。
   本文件统一用 `_ind(ok, cond)`：`ok` 为 False 处给 NaN，让 `min_count` 去跳过它。

5. **`min_count` 取 `N//2`，与参考库自己的 `min_periods` 一致**。
   参考库 `fac_new_daily.py` 的 `roll()` 默认 `min_periods=1`（几乎不设限），
   但那等于允许「窗口里只有 1 天有数据」也产出「20 日均值」——
   对长期停牌的股票会吐出纯噪声。参考库自己在 `structure_patterns.py` 的
   `inside/outside_bar_count_20` 用 `_roll_sum(x, 20, 5)`、在 `momentum_structure.py`
   的 `overnight_return_share_20` 用 `_roll_sum(x, 20, 10)`（= N//2）。
   本族统一取 **N//2**（5→2、10→5、14→7、20→10、60→30），
   语义是「窗口内至少一半的交易日有数据」，已经在各 note 里逐条标注。

6. **三个滚动原语在本文件里要绕开或改写**（都做了数值验证）：

   * `ctx.roll_skew(x, n, min_count=k)` **在 k < n 时有系统偏差** ——
     `mathx._cum_moments` 的三阶矩分母固定取 **n**（不是有效值个数），
     而 `_poison` 又按 `cnt >= k` 放行，于是缺 3 个值的 20 日窗会算出错的偏度。
     实测（T=60, C=5, n=20, 缺失率 15%, 与 pandas `rolling(20, min_periods=10).skew()`
     逐格比）：**最大偏差 0.2008**。→ 本文件用 `_roll_skew_eff()` 重写，
     改用「除以有效值个数」的矩，与 pandas **逐格一致到 3.3e-15**。
     （完整性：`mathx.roll_skew(x, n, None)` 在**满窗**时是对的，误差 3.2e-15 ——
     问题只出在 `min_count < n` 这条路径上。）
   * `ctx.roll_std(x, n, min_count=k, ddof=1)` **是对的**（`mathx.roll_var` 已按
     有效值个数做分母），实测与 pandas `rolling(n, min_periods=k).std(ddof=1)`
     一致到 **1.4e-17**。参考库的 `rolling(...).std()` 就是 ddof=1，
     本文件照抄 ddof=1（`mathx` 默认 ddof=0，差一个 `sqrt(k/(k-1))`，
     会随窗口内有效天数变化，不是常数）。
   * `ctx.roll_prod` 把 NaN 当**乘法单位元 1.0**（不复权语义下等于「跳过缺失」），
     会把「窗口里有 5 天没数据」静默算成「乘了 5 次 1」。
     复利类因子（`*_cum_20d`）本文件改用 `expm1(roll_sum(log1p(x)))`：
     停牌日是 NaN → 不进 sum 也不进 `min_count` 的分母，语义与不掩码的累乘一致。

7. **`ctx.roll_sum` 现在是有 `min_count` 参数的**（`fea/context.py` 里只剩一处定义，
   `liquidity.py` docstring 里记的那个「两处同名、后者覆盖前者」的坑已经修掉）。
   本文件直接用 `ctx.roll_sum(x, n, min_count=k)`，并已实测通过。

8. **形态族的「判据恰好等于阈值」是浮点边界，别拿它做逐格回归**（实测，供后人省一次排查）。
   `marubozu_ratio_10d` 的 `(上影+下影)/振幅 < 0.1` 在真实行情里会**恰好命中 0.1**
   （例：600095.SH 2026-01-06，`0.04/0.40`、`11.95−11.91=0.04`、`11.95−11.25=0.70`）。
   此时答案取决于价格×复权因子那几步乘法的最后几位：
   引擎算出的比值**恰好等于 float64 的 0.1** → `0.1 < 0.1` = False（数学上正确的一侧）；
   而把同样的公式用 pandas 直接从 parquet 复算会落在 0.1 下方 ~2e-15 → True。
   实测（探针因子 `(ratio-0.1)*1e9` 落盘）：2026 年有 **877 格** |ratio−0.1| < 1e-7，
   逐格比对时 `marubozu_ratio_10d` 因此有 2,104 格（0.41%）与独立复算差 ±1/窗口 ≤ 0.2。
   **不是未来函数、也不是掩码问题** —— 它同时出现在修复前后、也与 `high/low/open/close`
   的 ffill 无关；`min_count`、掩码、窗口全都一致。要改只能改判据（例如 `< 0.1 - 1e-9`
   或先量化价格），但那会偏离参考库的逐字公式，所以**保留原样并在此记录**。

## 与参考库的偏离（逐条给理由，一共 6 类）

| # | 偏离 | 理由 |
|:--|:--|:--|
| 1 | `daily_adj`（**前复权 qfq**）→ `ctx.hfq(...)`（后复权） | 契约硬约束 4 + `fea/spec.py` 的 PIT 红线：前复权会被未来的分红整体重算，历史因子值随未来事件变化 |
| 2 | `open/pre_close-1` → `hfq_open(T)/hfq_close(T-1)-1` | 数学等价（交易所前收 ≡ 昨收 × 除权调整），但不依赖供应商的 `pre_close` 列 |
| 3 | 参考库 `min_periods=1`（`fac_new_daily` 的 `roll` 默认）→ **N//2** | 见第 5 条；与参考库自己在 `structure_patterns` / `momentum_structure` 里的取值一致 |
| 4 | **形态族全部改成「20 日窗口内满足形态的交易日占比」** | 任务书对形态族的定义就是占比；`_count_` 两个名字在**满窗**时与参考库的计数只差常数 20（rank 完全相同），但占比对停牌日天然免疫；`td_setup_count` / `three_black_crows` 参考库是**连续段计数**（consecutive run length），不是窗口统计 —— 见各自 note |
| 5 | `cross_sectional_rank(...)` → `ctx.cs_zscore(..., mask=ctx.universe)` | 契约 §2.3「不许在因子里做 rank」；参考库的 `obv_divergence_20` / `volume_price_divergence_score` 是「两次截面 rank 相减再 rank」，本文件保留「两次截面标准化相减」的**结构**，只换掉标准化方式（z-score 与 pct-rank 逐行单调、同量纲） |
| 6 | `obv` 的方向用 `ctx.ret(1)` 的符号，不用 `pct_chg` | 契约硬约束 4；两者只在除权日差符号，而除权日正是最需要复权口径的地方（参考库自己也写了「方向用复权收益符号(规避未复权 close diff 符号错乱)」） |

### 一处**不**偏离、但值得记的：任务书里「跳空在 N 日内被回补」按字面实现是**未来数据**
`gap_fill_tendency_10d` 若按「跳空在之后 N 日内被回补」实现，必须在 T 日读 `T+1..T+N`
的价格 —— 那是**未来数据泄露**（面板右端就是 T）。参考库的公式是**当日口径**：
`gap_filled = (gap>1% & close<pre_close) | (gap<-1% & close>pre_close)`，
再对**过去** 10 日滚动求和。本文件照参考库实现，窗口只向过去看。

## 已知的量纲/上游事实（实测，供下游解读）

* `ctx.px("vol")` 是**股**，且**没有**单位翻转：`amount/vol ≈ close` 在 2013
  （000001.SZ 前 3 日：16.167/16.195/16.045 vs close 15.99/16.30/16.00）与 2025
  （11.90/11.75/…/11.44）都成立。（`factors/intraday.py` 第 4 条记的「手→股」翻转
  只发生在 **5min 层**，日线层没有。）
* `eom_14` 的量纲是 **元²/股**（`mid` 位移 × 振幅 ÷ 成交量），数量级 ~1e-9~1e-5，
  **不要**与参考库的绝对水平对齐（参考库的 `scale` 除法与成交量单位都可能不同），
  截面 rank 不受常数因子影响。
* `kama_efficiency_20` 与 `factors/momentum.py` 的 `ret_efficiency_20` 是
  **同一个量的两种写法**（`|ΔP_20|/Σ|ΔP|` ≡ `|Σr|/Σ|r|`，差一个 log1p 近似）。
  实测（2026 全年、与 `main.py dedup` 同口径：落盘 rank 逐日截面 z 后长向量 Pearson）
  **ρ = 0.9876**（另口径：逐日截面 Spearman 均值 0.9901，170 天里最低 0.9679）
  —— 超过 0.95 的重线，**收口时应删掉一个**（本文件建议留 `ret_efficiency_20`，
  因为它已在生产面板里、且名字更通用）。
* 本族内部的重复簇只有两对，且都是「同一因子的两个写法」，属**任务书点名要的**：
  `overnight_cum_20d ~ overnight_ma_20d` ρ=0.9991、`intraday_cum_20d ~ intraday_ma_20d`
  ρ=0.9983（复利累计 ≡ 20×均值，只差 min_count 口径）。除此之外全族 435 对里
  只有 2 对 ≥0.90、4 对 ≥0.80。
* 与**生产面板**的跨家族 ρ ≥0.90 只有两对（2026）：上面那对 `kama_efficiency_20`、
  以及 `intraday_ret_momentum ~ idt_overnight_minus_intraday` ρ=0.9078
  （同一件事的两种切法：单日 (close-open)/open vs 隔夜/日内分解之差）——
  低于阈值线，保留但下游降权/正交化时要注意。形态族与生产面板**几乎正交**：
  全族每只股票的最大跨家族 |ρ| ≤ 0.35，最大的三个也只有
  `three_black_crows ~ close_location_20d` 0.714、`td_setup_count ~ close_location_20d` 0.851、
  `eom_14 ~ bias_20` 0.750。
"""

from __future__ import annotations

import warnings

import numpy as np

from fea.spec import FactorSpec, register

# 跟随 conf/config.yaml 的 default_start（当前 2026-01-01）
PAT_START = None

GROUP = "pattern"
PRICE_DEPS = ("stock_daily", "stock_adj_factor")

# warmup_days = 「计算窗口要往前多读多少**日历天**」，纯日频滚动按 N×1.8+20 给（契约 §2）。
W5 = 30        # 5 交易日窗
W10 = 40       # 10 交易日窗
W14 = 46       # 14 交易日窗（eom_14）
W20 = 60       # 20 交易日窗（20×1.8+20 = 56）
W60 = 128      # 60 交易日窗
WT = 70        # td_setup：20 日窗 **再往前 shift 4 日** → 有效 24 个交易日


# ══════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════

def _hfq(ctx, field: str) -> np.ndarray:
    """后复权价（本项目唯一允许的价格口径）。停牌日是**前向填充**的旧值 —— 谁用谁负责掩码。"""
    return np.asarray(ctx.hfq(field), dtype=np.float64)


def _tr(ctx) -> np.ndarray:
    """`(T, C) bool`：当日真有成交（停牌 / 未上市 / 已退市为 False）。"""
    return np.asarray(ctx.traded(), dtype=bool)


def _tr_lag(ctx, k: int) -> np.ndarray:
    """**k 个交易日之前**是否真有成交（面板本身就是交易日历，移一行 = 移一天）。"""
    return np.asarray(ctx.shift(_tr(ctx).astype(np.float64), k), dtype=np.float64) > 0.5


def _ok2(ctx) -> np.ndarray:
    """`traded(T) & traded(T-1)` —— 所有**跨日**比较的统一步调掩码（模块 docstring 第 1 条）。"""
    return _tr(ctx) & _tr_lag(ctx, 1)


def _ind(ok: np.ndarray, cond: np.ndarray) -> np.ndarray:
    """布尔判据 → `{0.0, 1.0}`，**`ok` 为 False 处给 NaN 而不是 0**（模块 docstring 第 4 条）。

    0 = 「当天有行情，但没出现该形态」；NaN = 「当天根本没有数据」。
    混起来的后果是把停牌频率混进因子值里，而且不报错。
    """
    return np.where(np.asarray(ok, dtype=bool), np.asarray(cond, dtype=bool).astype(np.float64),
                    np.nan)


def _overnight(ctx, ok: np.ndarray) -> np.ndarray:
    """隔夜收益 = `hfq_open(T) / hfq_close(T-1) - 1`，`ok` 之外的格子是 NaN。

    `hfq_close(T-1)` 与交易所前收（昨收 × 除权调整）**数学等价**，但走的是
    「未复权价 × 累计复权因子」这条 PIT 安全路径（模块 docstring 第 2 条）。
    `min_abs_den=1e-8` 挡掉价格层的 0 价脏行（后复权价最小量级 0.01，这个地板不会误伤）。
    """
    op = _hfq(ctx, "open")
    pc = ctx.shift(_hfq(ctx, "close"), 1)
    g = ctx.safe_div(op, pc, min_abs_den=1e-8) - 1.0
    return np.where(ok, g, np.nan)


def _intraday(ctx, ok: np.ndarray) -> np.ndarray:
    """日内收益 = `hfq_close(T) / hfq_open(T) - 1`，`ok` 之外的格子是 NaN。

    只吃当日 OHLC（同一天内 adj 整体约掉），所以停牌日只需要 `traded(T)` 一层掩码。
    """
    op = _hfq(ctx, "open")
    cl = _hfq(ctx, "close")
    r = ctx.safe_div(cl, op, min_abs_den=1e-8) - 1.0
    return np.where(ok, r, np.nan)






def _roll_skew_eff(ctx, x: np.ndarray, n: int, min_count: int) -> np.ndarray:
    """滚动**无偏**偏度，**按窗口内有效值个数**做矩的分母（= pandas `rolling(n, min_periods=k).skew()`）。

    ★ 为什么不能用 `ctx.roll_skew`：`mathx._cum_moments` 的一/二/三阶矩分母固定取
      **n**，而 `_poison` 按 `cnt >= k` 放行 —— `k < n` 时缺 k..n-1 个值的窗口会算出
      系统性偏掉的偏度。实测（n=20, min_count=10, 缺失率 15%）最大偏差 **0.2008**
      （偏度本身量级才 ~1，这不是小误差）。本实现改分母为有效值个数：

          m1 = E[x]  m2 = E[x²]-E[x]²  m3c = E[x³] - 3·E[x]·E[x²] + 2·E[x]³
          g1 = m3c / m2^1.5 ;  skew = sqrt(k(k-1))/(k-2) · g1      （k = 有效值个数）

      与 pandas 逐格一致到 **3.3e-15**（实测，同一组随机面板）。
      三个期望都用 `ctx.roll_mean`（它在**有效值**上取均值），所以缺哪几天不影响口径。
    """
    m1 = ctx.roll_mean(x, n, min_count)
    m2 = ctx.roll_mean(x * x, n, min_count)
    m3 = ctx.roll_mean(x * x * x, n, min_count)
    k = ctx.roll_count(x, n)                       # 有效值个数（不毒化）
    var = np.maximum(m2 - m1 * m1, 0.0)            # 浮点误差导致的极小负值
    m3c = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    with np.errstate(all="ignore"):
        g1 = m3c / np.power(var, 1.5)
        out = np.sqrt(k * (k - 1.0)) / (k - 2.0) * g1
    return np.where(k >= 3.0, out, np.nan)         # k < 3 时三阶矩无定义


def _bars(ctx):
    """当日后复权 `(open, high, low, close)` —— 形态族共用（同一天内 adj 整体约掉）。"""
    return _hfq(ctx, "open"), _hfq(ctx, "high"), _hfq(ctx, "low"), _hfq(ctx, "close")






# ══════════════════════════════════════════════════════════════════════
# ① 隔夜族（9 个）
# ══════════════════════════════════════════════════════════════════════



def _overnight_ma(ctx, n: int, min_count: int) -> np.ndarray:
    return ctx.roll_mean(_overnight(ctx, _ok2(ctx)), n, min_count)


@register(FactorSpec(
    name="overnight_ma_5d", version=2, group=GROUP, deps=PRICE_DEPS,
    desc="隔夜收益均值（5 个交易日）",
    formula="overnight = open/pre_close - 1; ma5 = rolling(5).mean()",
    start=PAT_START, warmup_days=W5, higher_is_better=True,
    note="参考库 Class1 `overnight_ma_5d`（fac_new_daily.py）的 `roll()` 默认 **min_periods=1** —— "
         "那等于「窗口里只有 1 天有数据」也产出「5 日均值」，对长期停牌股是纯噪声。"
         "本实现 min_count=2（= N//2，模块 docstring 第 5 条）。"
         "★ 隔夜腿掩码 `traded(T) & traded(T-1)`（模块 docstring 第 1 条）。"
         "停牌日 NaN 由 min_count 跳过，**不补 0**：补 0 等于把「没交易」记成「隔夜涨跌为 0」。",
))
def overnight_ma_5d(ctx):
    return _overnight_ma(ctx, 5, 2)


@register(FactorSpec(
    name="overnight_ma_20d", version=2, group=GROUP, deps=PRICE_DEPS,
    desc="隔夜收益均值（20 个交易日）",
    formula="overnight = open/pre_close - 1; ma20 = rolling(20).mean()",
    start=PAT_START, warmup_days=W20, higher_is_better=True,
    note="参考库 Class1 `overnight_ma_20d`（fac_new_daily.py），口径同 `overnight_ma_5d`："
         "min_periods=1 → 本实现 min_count=10（N//2）；隔夜腿掩码 `traded(T) & traded(T-1)`。",
))
def overnight_ma_20d(ctx):
    return _overnight_ma(ctx, 20, 10)


@register(FactorSpec(
    name="overnight_ma_60d", version=2, group=GROUP, deps=PRICE_DEPS,
    desc="隔夜收益均值（60 个交易日）",
    formula="overnight = open/pre_close - 1; ma60 = rolling(60).mean()",
    start=PAT_START, warmup_days=W60, higher_is_better=True,
    note="参考库 Class1 `overnight_ma_60d`（fac_new_daily.py），口径同 `overnight_ma_5d`："
         "min_periods=1 → 本实现 min_count=30（N//2）；隔夜腿掩码 `traded(T) & traded(T-1)`。"
         "★ 60 个交易日里撞上停牌的累积概率很高，min_count=30 保证「至少一半有行情」才出值。",
))
def overnight_ma_60d(ctx):
    return _overnight_ma(ctx, 60, 30)






@register(FactorSpec(
    name="overnight_std_5d", version=2, group=GROUP, deps=PRICE_DEPS,
    desc="隔夜收益标准差（5 个交易日，样本口径 ddof=1）",
    formula="overnight = open/pre_close - 1; std5 = rolling(5).std()",
    start=PAT_START, warmup_days=W5, higher_is_better=False,
    note="参考库 Class1 `overnight_std_5d`（fac_short_term.py）走 `roll(df,'overnight',5,'std')`，"
         "pandas `.rolling().std()` 是 **ddof=1**；`mathx.roll_std` 默认 ddof=0，"
         "差一个随窗口内有效天数变化的 `sqrt(k/(k-1))`（不是常数，会改变截面排序），"
         "故本因子显式 `ddof=1`。实测与 pandas `rolling(5,min_periods=2).std(ddof=1)` "
         "一致到 **1.4e-17**（模块 docstring 第 6 条）。"
         "min_periods=1 → 本实现 min_count=2（N//2）；隔夜腿掩码 `traded(T) & traded(T-1)`。"
         "★ 方向取负（`higher_is_better=False`）：粒度同族的 `overnight_gap_vol_20` 在参考库里"
         "就是 `rank(-gap_vol)`（高波动排后）。",
))
def overnight_std_5d(ctx):
    return ctx.roll_std(_overnight(ctx, _ok2(ctx)), 5, min_count=2, ddof=1)






# ══════════════════════════════════════════════════════════════════════
# ② 日内族（6 个）
# ══════════════════════════════════════════════════════════════════════















@register(FactorSpec(
    name="intraday_vol_ratio_5d", version=2, group=GROUP, deps=PRICE_DEPS,
    desc="日内/隔夜波动比：5 日 |日内收益| 均值 ÷ 5 日 |隔夜收益| 均值",
    formula="_ia = |intraday|; _oa = |overnight|; m_ia = rolling(5).mean(_ia); "
            "m_oa = rolling(5).mean(_oa); x = m_ia / (m_oa + 1e-8)",
    start=PAT_START, warmup_days=W5, higher_is_better=True,
    note="参考库 Class1 `intraday_vol_ratio_5d`（fac_short_term.py）逐字为 "
         "`m_ia / (m_oa + 1e-8)`，`roll` 默认 min_periods=1 → 本实现两条腿 min_count=2（N//2）。"
         "★ 注意分母是 **|隔夜| 的均值**（与 `overnight_intraday_ratio_20d` 不同 ——"
         "那个的分母是 |隔夜均值的绝对值|）。分子分母都 ≥0，比值 ≥0，"
         "**不是有界量**（隔夜全为 0 时溢出），保留 `+1e-8` 地板即参考库口径。"
         "★ 两条腿同用 `traded(T) & traded(T-1)`（模块 docstring 第 1 条）。",
))
def intraday_vol_ratio_5d(ctx):
    ok = _ok2(ctx)
    g = _overnight(ctx, ok)
    ia = _intraday(ctx, ok)
    m_ia = ctx.roll_mean(np.abs(ia), 5, 2)
    m_oa = ctx.roll_mean(np.abs(g), 5, 2)
    return ctx.safe_div(m_ia, m_oa + 1e-8, min_abs_den=0.0)


# ══════════════════════════════════════════════════════════════════════
# ③ 形态族（7 个）—— 一律「20 日窗口内满足该形态的交易日占比」（任务书口径）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="td_setup_count", version=2, group=GROUP, deps=PRICE_DEPS,
    desc="TD setup：20 日内 close ≤ 4 日前 close 的交易日**占比**（0~1，超卖衰竭程度）",
    formula="cond = (adj_close <= adj_close.shift(4)).astype(int); "
            "seg = (~cond).groupby(Code).cumsum(); count = cond.groupby([Code, seg]).cumsum(); "
            "参考库: cross_sectional_rank(count)   # 连续段计数",
    start=PAT_START, warmup_days=WT, higher_is_better=True,
    note="★ **口径偏离（唯一一处公式级偏离，务必看）**：参考库 "
         "`td_setup_count`（technical_daily.py）返回的是**连续段长度**"
         "（`cond` 的连续 True 计数，段被 False 打断后归零），不是窗口统计。"
         "本实现按任务书对形态族的定义做成「**20 日窗口内满足形态的交易日占比**」"
         "（`roll_mean(cond, 20, 10)`）。理由：(a) 任务书明确要求形态族用窗口占比；"
         "(b) 连续段计数在**有缺失的日线上不成立** —— 停牌日被掩码后会打断连续段，"
         "于是「停牌多的股票 setup 计数被系统性重置」，这不是我们要的信号；"
         "(c) 窗口占比与前 6 个形态因子同构，可比较、可解释。"
         "若要换回参考库口径，只需把函数体换成按段累计（本文件不提供，避免两套口径并存）。"
         "★ 比较基准是 `shift(hfq_close, 4)`：后复权口径（参考库自己也注明"
         "「跨日比较必须走复权基座，未复权 close 在除权日跳变会伪造连跌 setup」）。"
         "`hfq_close` 在停牌日是前向填充值 —— 与参考库在 adj 序列上的行为一致，"
         "本实现不再额外要求 T-4 有成交（那会把复牌股整段抹掉），只要求 T 有成交。"
         "值域 [0,1]。",
))
def td_setup_count(ctx):
    c = _hfq(ctx, "close")
    ref = ctx.shift(c, 4)
    ok = _tr(ctx) & np.isfinite(ref)
    return ctx.roll_mean(_ind(ok, c <= ref), 20, 10)


@register(FactorSpec(
    name="three_black_crows", version=2, group=GROUP, deps=PRICE_DEPS,
    desc="三只黑鸦：20 日内「收盘<开盘 且 收盘<前收」的交易日**占比**（0~1，连续阴跌强度）",
    formula="cond = (close < open) & (close < pre_close); "
            "seg = (~cond).groupby(Code).cumsum(); count = cond.groupby([Code, seg]).cumsum(); "
            "参考库: cross_sectional_rank(count)   # 连续段计数",
    start=PAT_START, warmup_days=W20, higher_is_better=True,
    note="★ 口径偏离同 `td_setup_count`（参考库 event_dynamics.py 是**连续段计数**，"
         "本实现按任务书做成 20 日窗口占比）。经典三只黑鸦是「连续三根阴线」，"
         "参考库把它实现为「连续阴跌天数（不限于 3 天）」，本实现是「20 日里的阴跌天数占比」"
         "—— 三者在单边阴跌行情里同向，区别只在横盘夹杂时的权重。"
         "★ `pre_close` 用 `hfq_close(T-1)` 代替（数学等价，PIT 安全）。"
         "★ 掩码 `traded(T) & traded(T-1)`：这个判据**同时**用到当日 OHLC 与前收，"
         "前一日停牌时前收是陈旧的，属于第 1 条要挡的复牌跳空。"
         "值域 [0,1]。",
))
def three_black_crows(ctx):
    o, _, _, c = _bars(ctx)
    ok = _ok2(ctx)
    cond = (c < o) & (c < ctx.shift(c, 1))
    return ctx.roll_mean(_ind(ok, cond), 20, 10)












# ══════════════════════════════════════════════════════════════════════
# ④ 缺口族（4 个）—— 跳空 = open(T) vs pre_close(T-1)
# ══════════════════════════════════════════════════════════════════════









# ══════════════════════════════════════════════════════════════════════
# ⑤ 其它（4 个）
# ══════════════════════════════════════════════════════════════════════







