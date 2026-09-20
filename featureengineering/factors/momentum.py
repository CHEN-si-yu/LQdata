"""动量 / 反转 / 相对强度因子（24 个）—— 只依赖价格层与指数日线。

## 口径（与契约 §1 的六条硬约束对齐）

- **一律走 `ctx.hfq(...)` / `ctx.ret(k)`**，绝不自己读 `stock_daily`。
  `ctx.ret(k) = hfq_close(T)/hfq_close(T−k) − 1`，其中的 `hfq = 未复权价 × 截至当日的累计
  `adj_factor``（后复权，PIT 安全）。手工写 `close/close.shift(k)-1` 会在**每个除权日造出
  一个假暴跌**，而且不会有任何报错 —— 这是本家族最容易犯的错。
- `ctx.ret(k)` 还顺手挡掉了 |单日收益| > 60% 的复权因子脏数据（`fea/prices.py` 实测有 223 格）。
  ⚠️ 但 k > 1 时这条阈值会**误杀强势股**，所以 k 日收益统一走 `_ret_k`（见下文专节）。

## ★ 停牌在本家族的语义（实测，与任务书的假设不同，务必照此理解）

任务书的假设是「`ctx.ret()` 在停牌日是 NaN」，**实测不是**：
`fea/prices.py` 把价格**水平量**（含 `hfq_close`）做了前向填充，所以停牌日的
`hfq_close(t) == hfq_close(t−1)` → `ctx.ret(1)` 在该格子是 **0.0**。

实测（2014-06 ~ 2015-12 面板）：9.7 万个「有价但无成交」的格子里，
`ret(1)` 为 0 的有 9.70 万个、为 NaN 的只有 53 个（那 53 个是 |r|>60% 被清洗掉的）。

后果与处置：

1. **长窗口动量不会因停牌而缺值**（这是对的，而且是更好的口径）：跨停牌段的累计收益
   把复牌跳空算进去了，等价于「停牌期间持有到复牌」的真实收益。
2. **不要用 ffill 去「补」停牌**（价格层已经补过水平量，再补流量会把停牌算成 0 成交）。
3. 代价：停牌日在 `ret(1)` 里贡献一个 **0 收益**观测，会让 `ret_autocorr_1d_20`、
   `up_down_vol_ratio_60`、`ret_efficiency_20` 这类**逐日收益**因子轻微低估波动、
   高估路径效率。这是价格层的既定口径（停牌=价格不动），不是本文件的 bug。
4. **连续停牌 ⇒ 相关系数退化**：`mathx.roll_corr` 只挡 `den == 0`，而长期停牌股的
   窗口里 `num`/`den` 都是 ~1e-16 的浮点噪声 → 实测吐出 |ρ| 高达 14.4 的垃圾值
   （2070 格，002145.SZ / 600193.SH 等）。`ret_autocorr_1d_20` / `trend_strength_60`
   因此在因子内做**定义域**判断（真实成交日不足 → NaN；|ρ|>1 → NaN），
   见各自 `note`。这是「无定义判缺失」，不是 winsor。
5. **窗口内零成交 ⇒ NaN**（本家族统一守卫 `_zero_trade_mask`）：停牌日 `ret(1)=0`，
   于是「整个窗口都没成交」的股票会拿到一个**凭空造出来的 0 动量**（实测 000023.SZ
   在 2015-12-21 的 momentum_20 = 0.000000），这个 0 会混进截面排名；而且引擎
   **全量重建**路径给 0.0、**增量重算**路径给 NaN（前向填充没有种子），同输入不同输出。
   注意：**部分停牌不挡**（窗口内成交过就按价格层的前向填充语义给值）。

## ★ k 日收益：为什么本家族不直接用 `ctx.ret(k)`（重要，实测）

`ctx.ret(k)` 的清洗是 `|r| > 60% → NaN`。`fea/prices.py` 自己的注释写的是
「**单日** |收益| > 60% 认定是复权因子脏数据」（实测脏数据确实是 223 个**日频**格子），
但这条阈值被直接套在了 **k 日累计收益** 上 → k 越大误杀越狠（实测 561×3484 面板）：

| k | 被误杀格数 | 误杀格子的 k 日收益中位 |
|--:|--:|--:|
| 5 | 2,019 | +61% |
| 20 | 15,592 | +74% |
| 60 | 94,562 | +81% |
| 120 | 167,413 | +89% |
| 250 | **317,223（有效格的 48%）** | **+109%（最大 +2367%）** |

被误杀的全是**涨得最多的股票**。对动量因子这是最坏方向的截尾：2015-06 泡沫顶点那几天，
`momentum_250` 的日均截面从 ~2100 掉到 **76**（只剩银行、石化这些没涨的）。

所以本家族统一用 `_ret_k(ctx, k) = Π(1 + ctx.ret(1)) − 1`（log1p + roll_sum 实现）：
无单日异常时与 `ctx.ret(k)` **逐格相等**（最大差 3.6e-15），单日异常毒化整个窗口
（比累计阈值更贴合 prices.py 的注释意图），停牌语义完全不变。
**待办（已报给主 Agent）**：把 `fea/prices.py:ret()` 的阈值改成 k 相关或按单日判定，
之后 `_ret_k` 可以直接换回 `ctx.ret(k)`。

`ret(1)` 的因子（`ret_autocorr_1d_20` / `ret_efficiency_20` / `up_down_vol_ratio_60` /
`residual_momentum_20`）不受影响：日频下 ±60% 恰好就是设计口径。

## 相对强度 `rs_*` 与残差动量的市场基准

用 `index_daily` 里的 **`000300.SH`（沪深300）** 日收益（`pct_chg/100`），
按**面板日期精确对齐**（实测 2010-01-04 起面板日期 100% 命中，无缺日；
真遇到缺日按「市场无变动 0」处理并计数告警）。基准写在 `BENCH` 常量里。
`index_daily` **不在** `fea/spec.py` 的 `LAGGED_DATASETS` 里（那是 `index_ths_daily`），
且实测它的最新日期就是最新交易日，不需要行位移。

## warmup

纯日频滚动窗口 N 个交易日 → `N × 1.8 + 20` 日历天（契约 §2）。
`momentum_250` / `price_to_52w_high` / `price_distance_from_52w_low` /
`time_since_52w_high` / `rs_250` 的 250 日窗口一律用 **480**（契约要求 ≥480）。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fea.spec import FactorSpec, register

log = logging.getLogger("fea.factors.momentum")

GROUP = "momentum"
PRICE_DEPS = ("stock_daily", "stock_adj_factor")
INDEX_DEPS = ("stock_daily", "stock_adj_factor", "index_daily")

#: 市场基准 —— 沪深300。换基准只改这一处（note 里写的是它）
BENCH = "000300.SH"

#: 250 日窗口的 warmup：契约 §2 明确要求 ≥480（250 交易日 ≈ 365 日历天 + 冗余）
W250 = 480

#: ★ `time_since_52w_high` 专用：**两倍窗口** 的 warmup。
#:
#: 为什么不是 W250（2026-09-17 实测，锚点改按年之后暴露）：该因子的状态是
#: 「最近一次『创 250 日新高』的行号」（`np.maximum.accumulate`），而
#: `at_high = px >= roll_max(px, 250)` —— 面板**前 249 行**的 `roll_max` 是 NaN，
#: 于是 `at_high` 恒 False、累加器只能从第 250 行起才有状态。
#: ⇒ 要在第 T 天得到正确值，面板至少要有 **250（让 roll_max 有效）+ 250（让累加器有状态）**
#:    = 500 个交易日 ≈ 760 日历天。给 480 天（≈320 交易日）时，年初只有 ~70 行有效状态，
#:   「最近一次新高」若发生在这 70 行之外 ⇒ 计数被**钉在 249**（假值），直到当年真的创一次新高。
#: 实测（2016-01-04）：`000001.SZ` 真值 141 → 480 天给出 **249**、960 天给出 **141** ✔；
#: `600519.SH` 真值 151 → 249 / **151** ✔。按年锚定下这会让**每年年初的一批格子失真**，
#: 全局锚定（旧口径）因为面板足够长而看不出问题 —— 属于「锚点一改就暴露」的老 bug。
W250_STATE = 960


def _warm(n_days: int) -> int:
    """N 个交易日的滚动窗口 → warmup 日历天（契约 §2：N×1.8+20）。"""
    return int(n_days * 1.8 + 20)


def _ret_k(ctx, k: int) -> np.ndarray:
    """k 个交易日的后复权收益 —— 与 `ctx.ret(k)` 等价，但把「脏数据」判定改回**单日**口径。

    ★ 为什么本家族不能直接用 `ctx.ret(k)`（实测，2013-09~2015-12 面板 561×3484）：

      `fea/prices.py:ret()` 的清洗是 `|r| > 60% → NaN`，它**自己的注释**写的是
      「**单日** |收益| > 60% 认定是复权因子脏数据」（实测脏数据确实是 223 个**日频**格子）。
      但这条阈值被直接套在了 **k 日累计收益** 上，于是 k 越大误杀越狠：

        k=5    误杀   2,019 格（被误杀者的 k 日收益中位 +61%）
        k=20   误杀  15,592 格（+74%）
        k=60   误杀  94,562 格（+81%）
        k=120  误杀 167,413 格（+89%）
        k=250  误杀 317,223 格 = 该面板有效格的 48%（中位 +109%，最大 +2367%）

      被误杀的全是**涨得最多的股票**（2014~2015 牛市）。对动量因子这是最坏方向的截尾：
      2015-06 泡沫顶点那些天，momentum_250 的日均截面从 ~2100 掉到 76（只剩银行、石化）。

    本实现的构造：`r_k = Π(1 + ret(1)_i) − 1`，用 `log1p` + `roll_sum`（NaN 毒化）。

      * **无单日异常时与 `ctx.ret(k)` 逐格相等**（实测最大差 3.6e-15，就是浮点误差）；
      * 任何**单日**异常毒化整个窗口 —— 这才是 prices.py 注释想表达的口径；
      * 停牌语义完全不变（停牌日 `ret(1)=0`，即 1 倍，复牌跳空自然计入）；
      * `ret(1)` 已把 `|r|>60%` 置 NaN，所以 `log1p` 的入参恒在 [0.4, 1.6]，不会出 ±inf。

    ★★ 另一处必须自己挡的坑：**窗口内一次都没成交**的格子。

      `ctx.ret(1)` 对停牌日返回的是 **0.0 而不是 NaN**（实测：`fea/prices.py:ret()` 走
      `hfq_close`，而价格水平是**前向填充**量 → 停牌期间 `px(T) == px(T−1)` → 收益恰好 0）。
      于是一只**整个窗口都没交易**的股票会得到「动量 = 0」这个**凭空造出来的值**，
      而不是「不知道」：

        实测 000023.SZ / 000033.SZ 在 2015-12-21：20 日内成交天数 = 0，
        20 日 hfq 收盘 80.4466 → 80.4466（全常数），momentum_20 = 0.000000。

      这个 0.0 会**混进截面**（正好落在动量分布的正中间，排名既不前也不后），
      比缺失更危险 —— 契约 §2 的「补 0 会让滚动均值、截面排名全部失真」说的就是它。
      更麻烦的是**引擎两条路径会给出不同结果**（全量重建 = 0.0；增量尾部重算时面板起点
      在最后成交日之后，前向填充没有种子 → NaN → 整行被丢掉），同一格因子值取决于
      走哪条路径，这破坏了「同输入同输出」。

      处置：窗口内成交天数 == 0 ⇒ NaN（本函数与家族内其它价格窗口因子统一用
      `_zero_trade_mask`）。**部分停牌不挡**（如 000048.SZ 20 日内成交 10 天，
      仍按价格层的前向填充语义给值）—— 那只影响新鲜度，不影响可算性。

    价格层把阈值改成 k 相关（或按单日判定）、并让停牌日的 `ret()` 返回 NaN 之后，
    本函数可直接换回 `ctx.ret(k)`（`_zero_trade_mask` 届时也可一并去掉）。
    """
    r1 = ctx.ret(1)
    r = np.expm1(ctx.roll_sum(np.log1p(r1), int(k)))
    return np.where(_zero_trade_mask(ctx, k), np.nan, r)


def _zero_trade_mask(ctx, n: int) -> np.ndarray:
    """窗口 `n` 个交易日内**一次都没成交**的格子 → True。

    这类格子上的价格全是前向填充出来的陈旧值，任何由它算出的收益/位置都是
    「0 或常数」的假值（见 `_ret_k` 的 ★★）。`traded()` 是 bool（`vol` 有限），
    没有 NaN，所以 `roll_sum` 不会毒化。
    """
    tr = np.asarray(ctx.traded(), dtype=np.float64)
    return ctx.roll_sum(tr, int(n)) <= 0.0


# --------------------------------------------------------------------------- 基准
def _market_ret(ctx) -> np.ndarray:
    """沪深300 的日收益 → `(T,)`，与面板日期逐日对齐；缺日按 0（无变动）处理。

    ★ 指数**不在股票网格里**（`ctx.code_index` 会给 −1），所以不能走
      `asof_daily` / `panel.place`（那两个都要 code 下标）。
      这里直接在日期轴上 `searchsorted` 精确对齐 —— 指数是日频的，
      不需要 asof 语义，也就不存在「前向填充陈旧值」的问题。
    """
    df = ctx.dataset("index_daily", columns=["ts_code", "trade_date", "pct_chg"])
    T = ctx.panel.T
    if df.empty:
        log.warning("index_daily 为空，%s 的因子将全是 NaN", ctx.panel.dates[0])
        return np.full(T, np.nan)
    df = df[df["ts_code"] == BENCH]
    days = ctx.date_col(df["trade_date"])
    vals = pd.to_numeric(df["pct_chg"], errors="coerce").to_numpy(dtype=np.float64) / 100.0
    order = np.argsort(days, kind="stable")
    days, vals = days[order], vals[order]

    pos = np.clip(np.searchsorted(days, ctx.panel.dates), 0, days.size - 1)
    ok = (days[pos] == ctx.panel.dates) & np.isfinite(vals[pos])
    n_miss = int((~ok).sum())
    if n_miss:
        # 面板早于指数起点（index_daily 实测从 2010-01-04 开始）时会走到这里。
        # 补 0 = 「市场当日无变动」，比 ffill 一个陈旧收益安全（后者会重复计入涨跌）。
        log.warning("面板有 %d 个交易日在 %s 的 index_daily 里找不到，已按 0 收益处理"
                    "（首个：%s）", n_miss, BENCH, ctx.panel.dates[0])
    return np.where(ok, vals[pos], 0.0)


def _index_level(ctx) -> np.ndarray:
    """沪深300 的累计净值 `(T, 1)`：`L(t) = Π(1 + r_i)`，用于算指数的 k 日收益。"""
    m = _market_ret(ctx)
    return np.cumprod(1.0 + np.nan_to_num(m, nan=0.0))[:, None]


def _relative_strength(ctx, k: int) -> np.ndarray:
    """祖鲁法则的相对强度：`RS = (r_s − r_m) / (1 + r_m)`（参考库同式）。

    `r_m` 是指数**复利** k 日收益（不是把日收益乘 k），所以要用累计净值之比。
    """
    rs_ = _ret_k(ctx, k)                              # (T, C)
    lv = _index_level(ctx)                            # (T, 1)
    rm = lv / ctx.shift(lv, k) - 1.0                  # (T, 1) 指数 k 日复利收益
    return ctx.safe_div(rs_ - rm, 1.0 + rm, min_abs_den=1e-6)


# --------------------------------------------------------------------------- 动量
_MOM_DESC = {
    5: "5 个交易日的后复权动量（1 周趋势动能）",
    10: "10 个交易日的后复权动量（2 周趋势动能）",
    20: "20 个交易日的后复权动量（Jegadeesh-Titman 月度动量区间）",
    60: "60 个交易日的后复权动量（季度趋势延续）",
    120: "120 个交易日的后复权动量（半年趋势质量）",
    250: "250 个交易日的后复权动量（年度动量，最慢最稳）",
}
_MOM_NOTE = {
    5: "短周期，A 股 1~2 周以反转为主，本因子信号弱且与 short_term_reversal_5 完全反号。",
    10: "介于短周期反转与月度动量之间，实测常是三者里最弱的一档。",
    20: "A 股月度动量效应显著；与 reversal_2d / short_term_reversal_5 方向相反，"
        "建模时不要同时用（会互相抵消）。",
    60: "与 momentum_20 相关性高（同方向），只差一个尺度，适合二选一或做正交化。",
    120: "对应参考库的 alpha_125d / return_126d（125/126 交易日 ≈ 120），窗口对齐半年。",
    250: "★ warmup 必须 ≥480：250 交易日 ≈ 365 日历天，给少了每年分区的头 100 多天会静默错。",
}


def _make_momentum(k: int):
    def _fn(ctx):
        # 后复权 + 停牌 ffill + 脏数据清洗都在里面（见 _ret_k 的 ★ 说明：
        # 不用 ctx.ret(k) 是因为它对 k 日累计收益套了「单日 ±60%」的阈值，
        # 会把 2015 年涨得最多的股票整片抹成缺失）
        return _ret_k(ctx, k)
    _fn.__name__ = f"momentum_{k}"
    return _fn


for _k in (10, 20, 60, 120, 250):
    # 绑定成模块属性（= 注册函数名），方便调试时直接 from factors.momentum import momentum_20
    globals()[f"momentum_{_k}"] = _make_momentum(_k)
    register(FactorSpec(
        name=f"momentum_{_k}", group=GROUP, deps=PRICE_DEPS,
        desc=_MOM_DESC[_k],
        formula=f"Mom{_k} = hfq_close(T) / hfq_close(T-{_k}) - 1  "
                f"# 参考库: adj.groupby(Code).pct_change({_k}, fill_method=None)",
        start=None,
        warmup_days=W250 if _k >= 250 else _warm(_k),
        higher_is_better=True,
        version=3,          # v3：加「窗口内零成交 ⇒ NaN」守卫（★ 见 _ret_k 的 ★★），口径已变 → 必须重建
        note="★ 后复权（hfq），不是前复权：`adj` 在参考库里叫 daily_adj.parquet（qfq），"
             "照抄会破坏 PIT。停牌日 hfq_close 被前向填充 → 停牌段收益为 0、"
             "复牌跳空计入累计收益（口径正确，不要再用 ffill 去「修」）。"
             "★ 清洗口径：走 `_ret_k`（= Π(1+ret(1)) − 1），单日 |收益|>60% 的复权脏数据"
             "毒化整个窗口；**不用 `ctx.ret(k)`** 是因为它拿「k 日累计 >60%」当脏数据判据，"
             "实测 k=250 时误杀 48% 的有效格且全是强势股（详见 `_ret_k` 的 ★ 段）。"
             + _MOM_NOTE[_k],
    ))(globals()[f"momentum_{_k}"])








# --------------------------------------------------------------------------- 反转
@register(FactorSpec(
    name="reversal_2d", group=GROUP, deps=PRICE_DEPS,
    desc="2 日收益（超短周期反转的原始信号，方向在因子外部用）",
    formula="Reversal2 = hfq_close(T) / hfq_close(T-2) - 1  "
            "# 参考库: close.groupby(Code).pct_change(2)",
    start=None, warmup_days=_warm(2), higher_is_better=False,
    version=3,          # v3：加「窗口内零成交 ⇒ NaN」守卫（★ 见 _ret_k 的 ★★）
    note="参考库 fac_short_term.py 的 reversal_2d **不做取负**（值就是 2 日收益），"
         "方向由下游处理 —— 本实现照抄该口径，用 higher_is_better=False 标注方向。"
         "★ 与 short_term_reversal_5 不同：那个参考库里显式取了负号。"
         "反转因子理论上不需要复权（2 日内除权概率低），这里仍走 `_ret_k`（后复权），"
         "一致性优先，且能挡掉复权因子脏行造的假暴跌。",
))
def reversal_2d(ctx):
    return _ret_k(ctx, 2)


@register(FactorSpec(
    name="short_term_reversal_5", group=GROUP, deps=PRICE_DEPS,
    desc="5 日短周期反转 = −(5 日动量)（近 5 日超买者排前）",
    formula="ShortReversal5 = -Mom5 = -(hfq_close(T) / hfq_close(T-5) - 1)  "
            "# 参考库: cross_sectional_rank(-mom5)",
    start=None, warmup_days=_warm(5), higher_is_better=True,
    version=3,          # v3：加「窗口内零成交 ⇒ NaN」守卫（★ 见 _ret_k 的 ★★）
    note="参考库 momentum_rebuilt.py 的 short_term_reversal_5 是最新 5 日动量的**负向**，"
         "这里 keep 负号并标 higher_is_better=True（值大 = 超跌更多 = 反转预期更强）。"
         "★ 它是 momentum_5 的精确相反数（−1 ×），两者只能留一个。"
         "短窗口除权概率低、本可不复权，仍统一走 `_ret_k`（后复权 + 单日脏数据毒化）。",
))
def short_term_reversal_5(ctx):
    return -_ret_k(ctx, 5)










# --------------------------------------------------------------------------- 52 周 / 区间位置
@register(FactorSpec(
    name="price_to_52w_high", group=GROUP, deps=PRICE_DEPS,
    desc="52 周高点接近度 = 现价 / 250 日最高收盘价 − 1（越接近 0 越靠近年内高点）",
    formula="Proximity = hfq_close / roll_max(hfq_close, 252) - 1  "
            "# 参考库: adj / adj.rolling(252, min_periods=120).max() - 1",
    start=None, warmup_days=W250, higher_is_better=True,
    version=2,          # v2：加「窗口内零成交 ⇒ NaN」守卫（★ 见 _ret_k 的 ★★）
    note="George-Hwang 52 周高点效应在 A 股的实现。★ warmup=480（250 交易日窗口）。"
         "★ 偏离两处：(1) 参考库窗口是 252 交易日，本实现用 250（52 周 ≈ 250 个交易日，"
         "差 2 日对「年内高点」的位置无实质影响），公式串保留参考库原文以利溯源；"
         "(2) 参考库用 min_periods=120（上市不足半年也给值），本实现要求**满 250 日**，"
         "否则次新股的「52 周高点」其实是上市以来的高点，口径不同。"
         "值域 (−1, 0]，越接近 0 = 越贴近年内高点（上方套牢盘最少）。",
))
def price_to_52w_high(ctx):
    px = ctx.hfq("close")
    hi = ctx.roll_max(px, 250)
    out = ctx.safe_div(px - hi, np.abs(hi), min_abs_den=1e-12)
    # 零成交窗口 ⇒ NaN（否则 px 被前向填充成常数、恰好等于「年内高点」→ 假值 0.0）
    return np.where(_zero_trade_mask(ctx, 250), np.nan, out)


@register(FactorSpec(
    name="price_distance_from_52w_low", group=GROUP, deps=PRICE_DEPS,
    desc="距 52 周低点的距离 = (现价 − 250 日最低收盘) / 250 日最低收盘",
    formula="Dist = (hfq_close - roll_min(hfq_close, 252)) / roll_min(hfq_close, 252)  "
            "# 参考库: (adj - low_252) / low_252",
    start=None, warmup_days=W250, higher_is_better=True,
    version=2,          # v2：加「窗口内零成交 ⇒ NaN」守卫（★ 见 _ret_k 的 ★★）
    note="与 price_to_52w_high 一起构成 George-Hwang 效应的两端：贴着一年的低点 = 持续阴跌，"
         "远离低点 = 趋势健康。值域 [0, +∞)。★ 同样偏离参考库两处：窗口 252→250 交易日、"
         "min_periods=120→要求满 250 日（理由见 price_to_52w_high）。"
         "复权基座保证除权日不会造出假新低。",
))
def price_distance_from_52w_low(ctx):
    px = ctx.hfq("close")
    lo = ctx.roll_min(px, 250)
    out = ctx.safe_div(px - lo, np.abs(lo), min_abs_den=1e-12)
    # 零成交窗口 ⇒ NaN（否则 px 被前向填充成常数、恰好等于「年内低点」→ 假值 0.0）
    return np.where(_zero_trade_mask(ctx, 250), np.nan, out)


@register(FactorSpec(
    name="time_since_52w_high", group=GROUP, deps=PRICE_DEPS,
    desc="距最近一次 250 日新高的**交易日**数（0 = 今天创的年内新高）",
    formula="at_high = (hfq_close >= roll_max(hfq_close, 252));  "
            "DaysSince = bars since the last at_high（截断到 249）",
    start=None, warmup_days=W250_STATE, higher_is_better=False,
    version=3,          # v3：warmup 480→960（累加器状态需要**两倍窗口**，见 W250_STATE 的说明）
                        # v2：加「窗口内零成交 ⇒ NaN」守卫（★ 见 _ret_k 的 ★★）
    note="★★ **v3（2026-09-17）**：`warmup_days` 480 → 960。本因子的实现是"
         "`np.maximum.accumulate(行号)`（求「最近一次创 250 日新高的行号」），"
         "而 `at_high` 依赖 `roll_max` 先有效（面板前 249 行是 NaN）⇒ 累加器只能从第 250 行起有状态 ⇒"
         "**面板必须有两倍窗口（≈500 交易日）才能在年初给出正确值**。"
         "给 480 天时年初计数被钉在 249（假「250 日没创过新高」），"
         "实测 `000001.SZ@2016-01-04` 真值 141 而旧版给 249（960 天版给 141 ✔）。"
         "这是面板锚点从「全局」改成「按年」之后**才暴露**的 —— 全局锚定下面板天然够长。\n"
         "★ 偏离参考库三处："
         "(0) 窗口 252→250 交易日（与另两个 52 周因子一致）；"
         "(1) 参考库用「全历史 ffill 最后一次新高时间」→ 无界（可以到几千天），"
         "但我们的面板只从 warmup 起算，无界口径会让**长期阴跌股静默变成 NaN**"
         "（恰恰是因子最该识别的尾部）；本实现改用 250 日窗内的高点，值域 [0, 249]，永不缺值。"
         "(2) 单位用**交易日**而不是参考库的日历天 —— 交易日数不受假期/周末错位影响，更稳。"
         "实现上用 np.maximum.accumulate(行号) 取「最近一次创 250 日新高的行号」，"
         "所以今天平前高（>=）记 0；这与 roll_argmax 不同（后者并列时取**最早**一次）。",
))
def time_since_52w_high(ctx):
    px = ctx.hfq("close")
    hi = ctx.roll_max(px, 250)
    at_high = np.isfinite(hi) & (px >= hi)
    rows = np.arange(px.shape[0], dtype=np.float64)[:, None]
    # 最近一次「创 250 日新高」的行号；从未出现过是 −inf → 下面截断成 249
    last = np.maximum.accumulate(np.where(at_high, rows, -np.inf), axis=0)
    out = np.minimum(rows - last, 249.0)
    # 零成交窗口 ⇒ NaN（否则 px 被前向填充成常数 → 「天天都是年内新高」→ 恒为 0）
    return np.where(_zero_trade_mask(ctx, 250), np.nan, out)




# --------------------------------------------------------------------------- 收益结构
@register(FactorSpec(
    name="ret_autocorr_1d_20", group=GROUP, deps=PRICE_DEPS,
    desc="日收益一阶自相关（20 日窗口）：正 = 涨后跟涨（动量），负 = 均值回归（反转）",
    formula="AC = Corr(ret, shift(ret, 1), 20)  "
            "# 参考库: ret_w.rolling(20).corr(ret_w.shift(1))",
    start=None, warmup_days=_warm(20), higher_is_better=True,
    version=2,          # v2：加退化窗口定义域守卫（见 note 的 ★★），口径已变 → 必须重建
    note="直接度量「趋势的稳固程度」，与动量的**水平**正交，是反转强度的标准化度量。"
         "★ 偏离：参考库 min_periods=10，本实现要求满 20 个有效观测（缺失毒化）。"
         "停牌日 ret(1)=0（价格层前向填充的副作用），会给自相关灌进少量 0 观测 → "
         "轻微向 0 收缩；停牌多的股票解释时要小心。值域 [−1, 1]。"
         "★★ 实测坑（2070 格 |ρ|>1、最大 14.36，全部是 002145.SZ / 600193.SH 一类"
         "连续停牌股）：退化窗口里 num 与 den 都是浮点噪声，`mathx.roll_corr` 的 "
         "`den > 0` 守卫挡不住。故本因子显式做两步**定义域**处理（不是 winsor、"
         "不是 rank，只是把「无定义」判成 NaN）：① 窗口内真实成交日 < 10（= 一半）→ NaN；"
         "② |ρ| > 1 → NaN。处理后值域严格 [−1,1]（实测 max 0.975），"
         "代价是 2015 年丢掉约 5% 的格子（那些格子本来就没有信息）。",
))
def ret_autocorr_1d_20(ctx):
    r = ctx.ret(1)
    c = ctx.roll_corr(r, ctx.shift(r, 1), 20)
    # ① 定义域：窗口里真实成交的交易日太少 ⇒ 收益序列≈常数 ⇒ 相关系数无定义
    #    （不是 0，是「没有信息」）。停牌日 ret(1)=0 会把这些格子伪装成「有观测」。
    n_traded = ctx.roll_sum(np.asarray(ctx.traded(), dtype=np.float64), 20)
    c = np.where(n_traded >= 10.0, c, np.nan)
    # ② 数学值域兜底：|ρ|≤1 恒成立，越界只可能是①没盖住的残余数值退化
    return np.where(np.abs(c) <= 1.0, c, np.nan)


@register(FactorSpec(
    name="ret_efficiency_20", group=GROUP, deps=PRICE_DEPS,
    desc="20 日路径效率 = |20 日净收益| / 20 日收益的绝对值和（Kaufman ER）",
    formula="ER = |Σ_{20} r| / Σ_{20} |r|, r = 日收益  "
            "# 参考库同名侧写: kama_efficiency_20（|adj.diff(20)| / rolling_sum(|adj.diff()|)）",
    start=None, warmup_days=_warm(20), higher_is_better=True,
    note="★ **偏离参考库的 ret_efficiency_20**：参考库那个 factor 叫『价格效率』，"
         "实现是 `rolling_sum(pct_chg, 20) / rolling_sum(amount, 20)`（收益/成交额）。"
         "本实现按任务书括号里的『路径效率』做成 Kaufman 效率比率（净位移 / 路径长度）："
         "ER 高 = 单边趋势（路径短、位移大），ER 低 = 反复震荡（路径长、位移小）。"
         "选它的理由：对成交额的单位（元/千元）与量纲不敏感、值域天然是 [0,1]、"
         "且与 amihud 类流动性因子不重复。要改回参考库口径只需换成 "
         "safe_div(roll_sum(ctx.ret(1),20), roll_sum(ctx.amount,20))。"
         "值域 [0,1]，恒不缺失。",
))
def ret_efficiency_20(ctx):
    r = ctx.ret(1)
    net = np.abs(ctx.roll_sum(r, 20))
    gross = ctx.roll_sum(np.abs(r), 20)
    return ctx.safe_div(net, gross, min_abs_den=1e-12)






@register(FactorSpec(
    name="trend_strength_60", group=GROUP, deps=PRICE_DEPS,
    desc="60 日趋势强度 = 带符号的线性回归 R²（价格对时间回归）",
    formula="R2 = Corr(t, hfq_close, 60)^2 （= OLS 的 R²）;  Factor = sign(Corr) * R2  "
            "# 参考库无同名因子；同类见 rsrs_r2_18 / macd_trend_strength",
    start=None, warmup_days=_warm(60), higher_is_better=True,
    version=2,          # v2：加 |ρ|>1 兜底（见 note 末句），口径已变 → 必须重建
    note="★ 两份参考文档里没有同名因子，这是**自建口径**：任务书要求『R² 或 t 值类』。"
         "时序回归 R² = 相关系数平方，所以用 roll_corr(行号, 收盘价, 60)² 免去 OLS 的斜率/截距。"
         "取**带符号**：纯 R² ∈ [0,1] 对上涨和下跌一视同仁，在动量家族里没有方向；"
         "sign(ρ)·R² 让「强势单边上涨」最大、「强势单边下跌」最小（−1）。值域 [−1, 1]。"
         "R² 对 y 的正仿射变换不变，所以用不用后复权价、用不用对数价，结果完全一样。"
         "与 ret_autocorr_1d_20 同样的退化窗口兜底：|ρ|>1 只可能来自数值退化 → NaN"
         "（本因子的 x 是行号、方差大，实测 0 格越界；加它是为了不让同一个坑再咬一次）。",
))
def trend_strength_60(ctx):
    px = ctx.hfq("close")
    T, C = px.shape
    tn = np.broadcast_to(np.arange(T, dtype=np.float64)[:, None], (T, C)).copy()
    c = ctx.roll_corr(tn, px, 60)
    return np.where(np.abs(c) <= 1.0, np.sign(c) * c * c, np.nan)
