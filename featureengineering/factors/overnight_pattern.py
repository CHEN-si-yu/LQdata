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


def _log1p_pos(x: np.ndarray) -> np.ndarray:
    """`log1p(x)`，`x <= -1`（价格比非正，只可能是上游脏数据）处给 NaN。

    ★ 不能直接 `np.log1p`：`x = -1` 会产出 **-inf**，而 `mathx` 的滚动把 ±inf
      当**缺失**（`_finite` = isfinite），于是那一天被静默从窗口里剔除 ——
      累乘结果看起来完全正常，只是少乘了一天。
    """
    ok = np.isfinite(x) & (x > -1.0)
    return np.where(ok, np.log1p(np.where(ok, x, 0.0)), np.nan)


def _cum_prod(ctx, x: np.ndarray, n: int, min_count: int) -> np.ndarray:
    """`Π(1+x) - 1` 的滚动版（参考库 `_cum_prod` = `(1+s).rolling(n).apply(np.prod) - 1`）。

    ★ 不用 `ctx.roll_prod`：它把 NaN 当**乘法单位元 1.0**，等于「跳过缺失」，
      与「缺失毒化 + min_count」的框架语义不一致（模块 docstring 第 6 条）。
      `log1p` + `roll_sum` 是 O(T·C)，且 `expm1(log1p(x)) = x` 逐位可逆。
    """
    return np.expm1(ctx.roll_sum(_log1p_pos(x), n, min_count))


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


def _rng(ctx, hi: np.ndarray, lo: np.ndarray):
    """振幅与「振幅是否为正」掩码（一字板 `high == low` → 形态判据是 0/0 型无定义）。"""
    r = hi - lo
    return r, np.isfinite(r) & (r > 0)


def _cs_z(ctx, x: np.ndarray) -> np.ndarray:
    """截面 z-score（`mask=ctx.universe`），**本地吞掉全 NaN 行的 RuntimeWarning**。

    ★ `fea/mathx.py: _row_reduce` 直接调 `np.nanmean/np.nanstd`，而面板**开头的 warmup 行**
      在 `ret(20)` 这类因子上是整行 NaN（所有股票都还没满 20 个交易日）——
      这是预期行为（返回 NaN 是对的），但 numpy 会刷 "Mean of empty slice" /
      "Degrees of freedom <= 0 for slice" 两条 RuntimeWarning。实测跑一次会各出 2 条。
      这里只在**本因子自己的调用**上抑制，不改 `fea/**`（契约 §0 禁止）。
      （同样的问题 valuation.py 的 `pe_pb_zscore` 也会撞到 —— 已写进汇报的「需要引擎扩展」。）
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return ctx.cs_zscore(x, mask=ctx.universe)


# ══════════════════════════════════════════════════════════════════════
# ① 隔夜族（9 个）
# ══════════════════════════════════════════════════════════════════════



def _overnight_ma(ctx, n: int, min_count: int) -> np.ndarray:
    return ctx.roll_mean(_overnight(ctx, _ok2(ctx)), n, min_count)


@register(FactorSpec(
    name="overnight_ma_5d", group=GROUP, deps=PRICE_DEPS,
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
    name="overnight_ma_20d", group=GROUP, deps=PRICE_DEPS,
    desc="隔夜收益均值（20 个交易日）",
    formula="overnight = open/pre_close - 1; ma20 = rolling(20).mean()",
    start=PAT_START, warmup_days=W20, higher_is_better=True,
    note="参考库 Class1 `overnight_ma_20d`（fac_new_daily.py），口径同 `overnight_ma_5d`："
         "min_periods=1 → 本实现 min_count=10（N//2）；隔夜腿掩码 `traded(T) & traded(T-1)`。",
))
def overnight_ma_20d(ctx):
    return _overnight_ma(ctx, 20, 10)


@register(FactorSpec(
    name="overnight_ma_60d", group=GROUP, deps=PRICE_DEPS,
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
    name="overnight_sign_consistency_20d", group=GROUP, deps=PRICE_DEPS,
    desc="隔夜方向一致性：20 日内 overnight > 0 的交易日占比（0~1）",
    formula="overnight_pos = (overnight > 0).astype(float); x = rolling(20).mean()",
    start=PAT_START, warmup_days=W20, higher_is_better=True,
    note="参考库 Class1 `overnight_sign_consistency_20d`（fac_new_daily.py）逐字为 "
         "`df['overnight_pos'] = (df['overnight']>0).astype(float)` + `roll(...,20,'mean')`，"
         "`roll` 默认 min_periods=1 → 本实现 min_count=10（N//2）。"
         "★ `overnight == 0` 精确相等时**不算**正方向（参考库用严格 `> 0`，本实现照抄）。"
         "★ 判据不可得（停牌 / 前一日停牌）给 NaN 而不是 0：给 0 会让停牌多的股票"
         "被系统性拉向「方向不一致」，而这个偏差与停牌频率相关，不是随机噪声（第 4 条）。"
         "值域 [0,1]。",
))
def overnight_sign_consistency_20d(ctx):
    g = _overnight(ctx, _ok2(ctx))
    return ctx.roll_mean(_ind(np.isfinite(g), g > 0.0), 20, 10)


@register(FactorSpec(
    name="overnight_skewness_20d", group=GROUP, deps=PRICE_DEPS,
    desc="20 日隔夜收益偏度的**绝对值取反**（−|skew|，越接近 0 = 信息冲击越不极端）",
    formula="skew = rolling(20, min_periods=10).skew(); skew = skew.clip(-5, 5); "
            "factor = -skew.abs()   # 参考库: cross_sectional_rank(-skew.abs())",
    start=PAT_START, warmup_days=W20, higher_is_better=True,
    note="参考库 Class1 `overnight_skewness_20d`（price_deep.py）返回 "
         "`cross_sectional_rank(-skew.abs())`。契约 §2.3 不许在因子里 rank，"
         "所以本因子返回**原始值 `-|skew|`**（rank 由引擎统一做），方向语义不变："
         "值越大（越接近 0）= 隔夜分布越对称 = 信息冲击越不极端。"
         "★ 偏度做 `clip(-5, 5)` 保留（参考库原样），避免 3 阶矩在退化窗口里爆掉。"
         "★ **必须用本文件的 `_roll_skew_eff`**：`ctx.roll_skew` 在 `min_count < n` 时"
         "三阶矩分母固定取 n，实测偏差高达 0.2008（模块 docstring 第 6 条）；"
         "本实现按有效值个数做矩，与 pandas `rolling(20, min_periods=10).skew()` 一致到 3.3e-15。"
         "★ 隔夜腿掩码 `traded(T) & traded(T-1)`。窗口内有效值 < 3 时三阶矩无定义 → NaN。",
))
def overnight_skewness_20d(ctx):
    g = _overnight(ctx, _ok2(ctx))
    s = _roll_skew_eff(ctx, g, 20, 10)
    return -np.abs(np.clip(s, -5.0, 5.0))


@register(FactorSpec(
    name="overnight_std_5d", group=GROUP, deps=PRICE_DEPS,
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


@register(FactorSpec(
    name="overnight_gap_vol_20", group=GROUP, deps=PRICE_DEPS,
    desc="20 日隔夜跳空波动率（样本标准差 ddof=1，高波动=信息不确定性高）",
    formula="overnight_ret = (open - pre_close)/(pre_close + 1e-8); "
            "gap_vol = rolling(20, min_periods=10).std()   # 参考库 rank(-gap_vol)",
    start=PAT_START, warmup_days=W20, higher_is_better=False,
    note="参考库 Class1 `overnight_gap_vol_20`（price.py）返回 `cross_sectional_rank(-gap_vol)`。"
         "契约 §2.3 不许在因子里 rank，故本因子返回**原始值 `+gap_vol`** 并置 "
         "`higher_is_better=False`（与 `factors/intraday.py: idt_overnight_gap` 同样的处置）。"
         "★ 与同族 `overnight_std_5d` 的差别只有窗口（20 vs 5）与参考库的 min_periods"
         "（10 vs 1，本实现统一 N//2 → 10 vs 2）；两者高度相关但窗口不同，不是重复因子。"
         "★ 分母的 `+1e-8` 由 `safe_div(min_abs_den=1e-8)` 承担（语义相同：只挡 0 价脏行）。"
         "ddof=1 与 pandas 对齐（实测 1.4e-17）。隔夜腿掩码 `traded(T) & traded(T-1)`。",
))
def overnight_gap_vol_20(ctx):
    return ctx.roll_std(_overnight(ctx, _ok2(ctx)), 20, min_count=10, ddof=1)


@register(FactorSpec(
    name="overnight_intraday_ratio_20d", group=GROUP, deps=PRICE_DEPS,
    desc="隔夜/日内收益强度比：20 日隔夜均值 ÷ |20 日日内均值|",
    formula="oma = rolling(20).mean(overnight); ima = rolling(20).mean(intraday); "
            "x = oma / (ima.abs() + 1e-6)",
    start=PAT_START, warmup_days=W20, higher_is_better=True,
    note="参考库 Class1 `overnight_intraday_ratio_20d`（fac_new_daily.py）逐字为 "
         "`oma / (ima.abs() + 1e-6)`，`roll` 默认 min_periods=1 → 本实现两条腿都 min_count=10。"
         "★ 分母是 **|日内均值的绝对值|**（不是 |日内| 的均值）—— 这是参考库的写法，"
         "本实现照抄：它会在大样本上接近 0，所以比值可以很大（不是有界量）。"
         "保留 `+1e-6` 地板（照抄参考库），**没有**把分母换成「接近 0 就 NaN」——"
         "只加了一层 `safe_div(min_abs_den=0.0)` 挡精确 0，口径不变。"
         "★ 实测（2026 全年 170 个交易日、1,553 万行）：|value| 最大值 1.005e5，"
         "全样本只有 12 个格子 > 1e5、0 个 > 1e8 —— 尾部确实很重（分母小的时候），"
         "但离体检红线远；这种重尾交给下游 winsor/rank（契约 §1 不许在因子里 winsor）。"
         "★ 两条腿同用 `traded(T) & traded(T-1)` 掩码（隔夜腿需要 T-1，日内腿只需 T，"
         "取并集让「隔夜 vs 日内」的对比落在同一批交易日上）。",
))
def overnight_intraday_ratio_20d(ctx):
    ok = _ok2(ctx)
    oma = ctx.roll_mean(_overnight(ctx, ok), 20, 10)
    ima = ctx.roll_mean(_intraday(ctx, ok), 20, 10)
    return ctx.safe_div(oma, np.abs(ima) + 1e-6, min_abs_den=0.0)


# ══════════════════════════════════════════════════════════════════════
# ② 日内族（6 个）
# ══════════════════════════════════════════════════════════════════════

def _intraday_1d(ctx) -> np.ndarray:
    """单日日内收益，只按 `traded(T)` 掩码（当日量，不需要 T-1）。"""
    return _intraday(ctx, _tr(ctx))




def _intraday_ma(ctx, n: int, min_count: int) -> np.ndarray:
    return ctx.roll_mean(_intraday_1d(ctx), n, min_count)


@register(FactorSpec(
    name="intraday_ma_5d", group=GROUP, deps=PRICE_DEPS,
    desc="日内收益均值（5 个交易日）",
    formula="intraday = close/open - 1; ma5 = rolling(5).mean()",
    start=PAT_START, warmup_days=W5, higher_is_better=True,
    note="参考库 Class1 `intraday_ma_5d`（fac_new_daily.py），`roll` 默认 min_periods=1 → "
         "本实现 min_count=2（N//2）。掩码只用 `traded(T)`（当日量）。",
))
def intraday_ma_5d(ctx):
    return _intraday_ma(ctx, 5, 2)


@register(FactorSpec(
    name="intraday_ma_20d", group=GROUP, deps=PRICE_DEPS,
    desc="日内收益均值（20 个交易日）",
    formula="intraday = close/open - 1; ma20 = rolling(20).mean()",
    start=PAT_START, warmup_days=W20, higher_is_better=True,
    note="参考库 Class1 `intraday_ma_20d`（fac_new_daily.py），min_periods=1 → 本实现 min_count=10。"
         "掩码只用 `traded(T)`（当日量）。",
))
def intraday_ma_20d(ctx):
    return _intraday_ma(ctx, 20, 10)


@register(FactorSpec(
    name="intraday_ma_60d", group=GROUP, deps=PRICE_DEPS,
    desc="日内收益均值（60 个交易日）",
    formula="intraday = close/open - 1; ma60 = rolling(60).mean()",
    start=PAT_START, warmup_days=W60, higher_is_better=True,
    note="参考库 Class1 `intraday_ma_60d`（fac_new_daily.py），min_periods=1 → 本实现 min_count=30。"
         "掩码只用 `traded(T)`（当日量）。",
))
def intraday_ma_60d(ctx):
    return _intraday_ma(ctx, 60, 30)


@register(FactorSpec(
    name="intraday_ret_momentum", group=GROUP, deps=PRICE_DEPS,
    desc="日内收益因子：(close − open) / open，单日口径",
    formula="intraday = (close - open) / open   # 参考库 cross_sectional_rank(intraday)",
    start=PAT_START, warmup_days=W5, higher_is_better=True,
    note="参考库 Class1 `intraday_ret_momentum`（price_deep.py）逐字为 "
         "`(close - open)/open` 再取截面排名 —— 名字里有 momentum，但**定义就是单日日内收益**，"
         "没有任何滚动窗口。本因子返回原始值（rank 由引擎做），方向不变。"
         "★ 与同族 `intraday_ma_5d` 的区别：那个是 5 日均值，这个是**当日**值"
         "（所以 warmup 只给 W5 冗余，面板右端 T 就是 T）。"
         "★ 掩码只用 `traded(T)`。参考库用 `open.replace(0, np.nan)` 挡 0 价，"
         "本实现用 `safe_div(min_abs_den=1e-8)`（后复权价最小量级 0.01，不误伤）。",
))
def intraday_ret_momentum(ctx):
    return _intraday_1d(ctx)


@register(FactorSpec(
    name="intraday_vol_ratio_5d", group=GROUP, deps=PRICE_DEPS,
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
    name="td_setup_count", group=GROUP, deps=PRICE_DEPS,
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
    name="three_black_crows", group=GROUP, deps=PRICE_DEPS,
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


@register(FactorSpec(
    name="hammer_ratio_20d", group=GROUP, deps=PRICE_DEPS,
    desc="锤子线频率：20 日内（下影 > 2×实体 且 上影 < 0.3×振幅 且 实体>0）的占比",
    formula="body=|close-open|; lower=min(open,close)-low; upper=high-max(open,close); "
            "rng=high-low; is_hammer=(lower>2*body)&(upper<0.3*rng)&(body>0); "
            "ratio=rolling(20, min_periods=10).mean(is_hammer)",
    start=PAT_START, warmup_days=W20, higher_is_better=True,
    note="参考库 Class1 `hammer_ratio_20d`（trend_pattern.py）逐字如上，`min_periods=10` 与"
         "本实现的 min_count=10（N//2）**完全一致**（这一族里唯一一个原生就对齐的）。"
         "参考库返回 `cross_sectional_rank(ratio)`，本因子返回原始占比（rank 由引擎做）。"
         "★ 一字板（`high == low`）时 `upper < 0.3*rng` 是 0 < 0 型无定义，"
         "参考库靠 `.replace(0, nan)` 判成 False；本实现用 `rng > 0` 显式判 `ok` → 给 **NaN**"
         "（判据不可得 ≠ 形态不成立，第 4 条）。实测一字板占 0.20%~1.81%（按年，模块 docstring 第 3 条）。"
         "★ 掩码只用 `traded(T)`：判据全是**当日** OHLC，不跨日。"
         "值域 [0,1]。",
))
def hammer_ratio_20d(ctx):
    o, h, l, c = _bars(ctx)
    body = np.abs(c - o)
    lower = np.minimum(o, c) - l
    upper = h - np.maximum(o, c)
    rng, ok_r = _rng(ctx, h, l)
    ok = _tr(ctx) & ok_r
    cond = (lower > 2.0 * body) & (upper < 0.3 * rng) & (body > 0.0)
    return ctx.roll_mean(_ind(ok, cond), 20, 10)


@register(FactorSpec(
    name="inside_bar_count_20", group=GROUP, deps=PRICE_DEPS,
    desc="孕线（内包线）频率：20 日内「今高 ≤ 昨高 且 今低 ≥ 昨低」的交易日占比",
    formula="prev_high=shift(high,1); prev_low=shift(low,1); "
            "inside=(high<=prev_high)&(low>=prev_low); count=_roll_sum(inside,20,5); "
            "参考库: cross_sectional_rank(count)",
    start=PAT_START, warmup_days=W20, higher_is_better=True,
    note="参考库 Class1 `inside_bar_count_20`（structure_patterns.py）用 `_roll_sum(inside, 20, 5)` "
         "= **20 日窗口内满足形态的交易日**（计数，不是比率）。"
         "本实现按任务书做成**占比**（`roll_mean(inside, 20, 10)`）：满窗 20 天时"
         "`count/20` 与 `count` 只差常数 20 → **截面 rank 完全相同**；"
         "但占比对停牌日天然免疫（计数会因为窗口里少了 3 天而系统性偏小，"
         "而且偏小的幅度与停牌频率相关）。min_periods 由 5 提到 10（N//2，模块 docstring 第 5 条）。"
         "★ 参考库注明「跨日比较用复权口径高低点」——本实现用 `ctx.hfq('high'/'low')`。"
         "★ 掩码 `traded(T) & traded(T-1)`（要跟昨日高低比）。值域 [0,1]。",
))
def inside_bar_count_20(ctx):
    _, h, l, _ = _bars(ctx)
    ok = _ok2(ctx)
    ph = ctx.shift(h, 1)
    pl = ctx.shift(l, 1)
    cond = (h <= ph) & (l >= pl)
    return ctx.roll_mean(_ind(ok & np.isfinite(ph) & np.isfinite(pl), cond), 20, 10)


@register(FactorSpec(
    name="outside_bar_count_20", group=GROUP, deps=PRICE_DEPS,
    desc="吞没/突破频率：20 日内「收盘 > 昨高 或 收盘 < 昨低」的交易日占比",
    formula="prev_high=shift(high,1); prev_low=shift(low,1); "
            "outside=(close>prev_high)|(close<prev_low); count=_roll_sum(outside,20,5); "
            "参考库: cross_sectional_rank(count)",
    start=PAT_START, warmup_days=W20, higher_is_better=True,
    note="参考库 Class1 `outside_bar_count_20`（structure_patterns.py）用 `_roll_sum(outside, 20, 5)` "
         "（计数）→ 本实现按任务书做成占比（`roll_mean(..., 20, 10)`），满窗时 rank 等价，"
         "且对停牌天然免疫（同 `inside_bar_count_20`）。"
         "★ 判据是**收盘价**与昨日全天区间比（不是今日高低点吞没昨日高低点）——"
         "参考库的 `outside` 就是 `(adj.gt(prev_high) | adj.lt(prev_low))`，本实现照抄。"
         "`gt/lt` 是**严格**比较，相等不算（照抄）。"
         "★ 掩码 `traded(T) & traded(T-1)`。值域 [0,1]。",
))
def outside_bar_count_20(ctx):
    _, h, l, c = _bars(ctx)
    ok = _ok2(ctx)
    ph = ctx.shift(h, 1)
    pl = ctx.shift(l, 1)
    cond = (c > ph) | (c < pl)
    return ctx.roll_mean(_ind(ok & np.isfinite(ph) & np.isfinite(pl), cond), 20, 10)




@register(FactorSpec(
    name="marubozu_ratio_10d", group=GROUP, deps=PRICE_DEPS,
    desc="光头光脚阳线频率：10 日内（上下影合计 < 10%×振幅 且 收>开）的交易日占比",
    formula="body=|close-open|; upper=high-max(open,close); lower=min(open,close)-low; "
            "rng=high-low; is_marubozu=((upper+lower)/rng < 0.1); is_green=(close>open); "
            "signal=is_marubozu*is_green; ratio=rolling(10, min_periods=5).mean(signal)",
    start=PAT_START, warmup_days=W10, higher_is_better=True,
    note="参考库 Class1 `marubozu_ratio_10d`（trend_pattern.py）逐字如上，`min_periods=5` = "
         "10//2，与本实现的 min_count=5 **完全一致**。窗口是 **10 日**（不是 20 日）——"
         "任务书里唯一一个 10 日窗的形态因子，本实现照抄。"
         "★ `is_marubozu` 用 `is_green` **相乘**而不是与（参考库原样）："
         "`(upper+lower)/rng` 无定义（rng=0）时参考库靠 `.replace(0, nan)` 让"
         "`nan < 0.1` 为 False → 0；本实现用 `ok = rng > 0` 显式判 NaN。"
         "★ 掩码只用 `traded(T)`。值域 [0,1]。",
))
def marubozu_ratio_10d(ctx):
    o, h, l, c = _bars(ctx)
    rng, ok_r = _rng(ctx, h, l)
    ok = _tr(ctx) & ok_r
    shadow = (h - np.maximum(o, c)) + (np.minimum(o, c) - l)
    # ★ 除法走 safe_div：一字板 rng=0 时给 NaN（并顺带消掉 numpy 的 0/0 RuntimeWarning）。
    #   参考库是 `(upper+lower)/total_range.replace(0, np.nan)`，语义相同。
    cond = (ctx.safe_div(shadow, rng, min_abs_den=0.0) < 0.1) & (c > o)
    return ctx.roll_mean(_ind(ok, cond), 10, 5)


# ══════════════════════════════════════════════════════════════════════
# ④ 缺口族（4 个）—— 跳空 = open(T) vs pre_close(T-1)
# ══════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="gap_fill_tendency_10d", group=GROUP, deps=PRICE_DEPS,
    desc="缺口回补倾向：10 日内跳空（>1%）后被**当日收盘**回补的比例",
    formula="gap=(open-pre_close)/pre_close; is_gap=|gap|>0.01; "
            "filled=((gap>0.01)&(close<pre_close))|((gap<-0.01)&(close>pre_close)); "
            "rate=_roll_sum(filled,10,5)/(_roll_sum(is_gap,10,5)+0.01)",
    start=PAT_START, warmup_days=W10, higher_is_better=False,
    note="参考库 Class1 `gap_fill_tendency_10d`（price_deep.py）逐字如上，`min_periods=5` 照抄。"
         "★ **回补 = 当日收盘回到前收之上/之下**（完全回补），不是「盘中触及缺口边缘」，"
         "也不是「之后 N 日内回补」—— 后者按字面实现要读 `T+1..T+N`，是**未来数据泄露**"
         "（面板右端就是 T），本文件**不**做。窗口只向过去看（模块 docstring 的专门一节）。"
         "★ 分母 `+0.01` 照抄参考库（避免除 0）；没有跳空日的股票 → 分母 0.01 → 值 0.0"
         "（不是 NaN），这一点与 `gap_open_follow_ratio_20` 不同，是参考库的原样行为。"
         "★ 跳空阈值 **1%** 照抄参考库（比 `gap_open_follow_ratio_20` 的 0.5% 严）。"
         "★ 掩码 `traded(T) & traded(T-1)`。值域 [0,1]（分子 ≤ 分母）。",
))
def gap_fill_tendency_10d(ctx):
    ok = _ok2(ctx)
    g = _overnight(ctx, ok)
    c = _hfq(ctx, "close")
    pc = ctx.shift(c, 1)
    is_gap = np.abs(g) > 0.01
    filled = ((g > 0.01) & (c < pc)) | ((g < -0.01) & (c > pc))
    n_gap = ctx.roll_sum(_ind(ok, is_gap), 10, 5)
    n_fill = ctx.roll_sum(_ind(ok, filled), 10, 5)
    return ctx.safe_div(n_fill, n_gap + 0.01, min_abs_den=0.0)


@register(FactorSpec(
    name="gap_up_fade_freq_20d", group=GROUP, deps=PRICE_DEPS,
    desc="高开低走频率：20 日内（隔夜>0 且 日内<0）的交易日占比（0~1）",
    formula="gap_up_fade = ((overnight>0) & (intraday<0)).astype(float); "
            "freq = rolling(20).mean()",
    start=PAT_START, warmup_days=W20, higher_is_better=False,
    note="参考库 Class1 `gap_up_fade_freq_20d`（fac_new_daily.py）逐字如上，"
         "`roll` 默认 min_periods=1 → 本实现 min_count=10（N//2）。"
         "★ 方向：值高 = 冲高抛压重、上方套牢盘多（参考库 `意义` 一栏），故 "
         "`higher_is_better=False`。"
         "★ 判据**不用**阈值（隔夜 > 0 就算高开，哪怕只高开 0.001%）—— 参考库原样。"
         "★ 停牌 / 前一日停牌一律 NaN，且**不补 0**：补 0 会把停牌多的股票推成"
         "「从不低走」（第 4 条）。值域 [0,1]。",
))
def gap_up_fade_freq_20d(ctx):
    ok = _ok2(ctx)
    g = _overnight(ctx, ok)
    ia = _intraday(ctx, ok)
    return ctx.roll_mean(_ind(ok, (g > 0) & (ia < 0)), 20, 10)


@register(FactorSpec(
    name="gap_down_recover_freq_20d", group=GROUP, deps=PRICE_DEPS,
    desc="低开高走频率：20 日内（隔夜<0 且 日内>0）的交易日占比（0~1）",
    formula="gap_down_rec = ((overnight<0) & (intraday>0)).astype(float); "
            "freq = rolling(20).mean()",
    start=PAT_START, warmup_days=W20, higher_is_better=True,
    note="参考库 Class1 `gap_down_recover_freq_20d`（fac_new_daily.py）逐字如上，"
         "`roll` 默认 min_periods=1 → 本实现 min_count=10（N//2）。"
         "★ 与 `gap_up_fade_freq_20d` 是**镜像**关系但**不是** 1 − x："
         "「隔夜>0 且 日内<0」与「隔夜<0 且 日内>0」互不覆盖（隔夜/日内同号的日子两边都不计），"
         "所以两个都必须独立保留。"
         "★ 方向：值高 = 恐慌被消化、买盘韧性好（参考库 `意义` 一栏），`higher_is_better=True`。"
         "★ 停牌一律 NaN，不补 0。值域 [0,1]。",
))
def gap_down_recover_freq_20d(ctx):
    ok = _ok2(ctx)
    g = _overnight(ctx, ok)
    ia = _intraday(ctx, ok)
    return ctx.roll_mean(_ind(ok, (g < 0) & (ia > 0)), 20, 10)


# ══════════════════════════════════════════════════════════════════════
# ⑤ 其它（4 个）
# ══════════════════════════════════════════════════════════════════════



@register(FactorSpec(
    name="eom_14", group=GROUP, deps=PRICE_DEPS,
    desc="14 日 Ease of Movement：价格中点位移 ÷ (成交量/振幅) 的 14 日均值",
    formula="mid=(adj_high+adj_low)/2; box_ratio=vol/(adj_high-adj_low); "
            "distance=mid-mid.shift(1); eom=distance/box_ratio; eom_avg=rolling(14, min_periods=7).mean()",
    start=PAT_START, warmup_days=W14, higher_is_better=True,
    note="参考库 Class1 `eom_14`（technical_daily.py）逐字如上，`min_periods=7` = 14//2，"
         "与本实现的 min_count=7（N//2）**一致**。参考库返回 `cross_sectional_rank(eom_avg)`，"
         "本因子返回原始值。"
         "★ 参考库注释明确写了「**除** box_ratio 而不是乘」（「Multiplication would reward "
         "high-volume/narrow-range days and is the inverse of the named indicator」），"
         "本实现照抄除法。"
         "★ 中点位移走**后复权**口径（参考库用 `scale = adj/close` 折算 high/low，"
         "数学上就是复权价），避免除权日伪位移。"
         "★ 振幅 0（一字板）→ `box_ratio` 分母 0 → `eom` NaN（参考库 `.replace(0, np.nan)` 同义）。"
         "★ 掩码 `traded(T) & traded(T-1)`：`distance` 是跨日量；停牌日 `vol` 本来就是 NaN，"
         "所以这层掩码只额外挡住「复牌日」（mid 从停牌前的陈旧值跳到复牌价）。"
         "★ 量纲是 **元²/股**（数量级 1e-9~1e-5），不要与参考库的绝对水平对齐；"
         "截面 rank 不受常数因子影响。值域无界但极小。",
))
def eom_14(ctx):
    _, h, l, _ = _bars(ctx)
    mid = (h + l) / 2.0
    dist = mid - ctx.shift(mid, 1)
    rng = h - l
    box = ctx.safe_div(ctx.px("vol"), rng, min_abs_den=0.0)     # 振幅 0 → NaN
    eom = ctx.safe_div(dist, box, min_abs_den=0.0)
    return ctx.roll_mean(np.where(_ok2(ctx), eom, np.nan), 14, 7)




@register(FactorSpec(
    name="volume_price_divergence_score", group=GROUP, deps=PRICE_DEPS,
    desc="量价背离得分：20 日价动量截面 z − 20 日量动量截面 z（值高 = 价升量缩）",
    formula="price_mom=adj_close.pct_change(20); vol_mom=vol.pct_change(20); "
            "pr=cs_rank(price_mom); vr=cs_rank(vol_mom); "
            "参考库: cross_sectional_rank(pr - vr)",
    start=PAT_START, warmup_days=W20, higher_is_better=False,
    note="参考库 Class1 `volume_price_divergence_score`（volume_price_dynamics.py）逐字如上。"
         "★ **偏离**：与 `obv_divergence_20` 同一处置 —— 两次 `cross_sectional_rank` "
         "换成两次 `ctx.cs_zscore(..., mask=ctx.universe)`（契约 §2.3 不许在因子里 rank）。"
         "★ 与 `obv_divergence_20` 的区别：那个用 **OBV 的 20 日变化 / 均量**（带方向符号的"
         "量能净额），这个用**成交量的 20 日变化率**（无量纲、无方向）。两者相关但不重复。"
         "★ 价动量用 `ctx.ret(20)`（后复权累计收益，已挡单日 |r|>60% 的复权脏数据），"
         "不用 `pct_change(hfq_close, 20)`，与 `factors/momentum.py` 的口径一致。"
         "★ 量动量的分母地板给 **1.0**（原始成交量单位）：`vol` 近 0 的格子"
         "（停牌前后、极冷门股）会产出 ±1e6 级的假变化率，默认的 1e-12 挡不住。"
         "★ 方向：值高 = 价强量弱的可疑上涨，`higher_is_better=False`。",
))
def volume_price_divergence_score(ctx):
    vol = np.asarray(ctx.px("vol"), dtype=np.float64)
    price_mom = np.asarray(ctx.ret(20), dtype=np.float64)
    vol_mom = ctx.pct_change(vol, 20, min_abs_den=1.0)
    return _cs_z(ctx, price_mom) - _cs_z(ctx, vol_mom)
