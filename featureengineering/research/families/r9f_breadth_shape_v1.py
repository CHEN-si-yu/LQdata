"""Round 9 · 单元 F · 市场宽度**分布形态**状态 × 个股特征的交互候选
（family `r9f_breadth_shape_v1`）。

前置事实（决定了本单元的形状）
══════════════════════════════════════════════════════════════════════
1. `research/evaluation.py` 对 `spec.is_market=True` 的对象有硬分支：只输出
   `evaluation_type="market_context_quality_only"`、`rankic=None, ic=None`
   （实测源码 L45–L55 / L68–L73），**市场标量拿不到任何 RankIC**，也就无法有 G4 裁决。
   所以本单元的交付物是**股票形状的 (T, C) 交互候选**（`is_market` 保持默认 False），
   市场状态序列只作为**条件变量**在 `compute(ctx)` 内部构造。
2. 项目内市场标量的预测性通道已被系统性检验过：`doc/市场因子清理记录_20261004.md`
   记录 27 个构造型市场因子 **M5 预测检验 0/27**，全数删除。⇒ 本单元不再登记任何
   独立市场标量，只登记交互候选。

三个预登记机制（每个恰一个候选；方向全部先于任何结果统计冻结为 False）
══════════════════════════════════════════════════════════════════════
state 三种（都来自 `stock_market_distribution_history` 的**整市场**分钟快照，
口径是**全市场**（实测每日 241 行、up+down+flat 中位数 4,770 只），
**不是** 2,115 只冻结池 —— 与 `mkt_full_flat_ratio` 同族口径，
名字/描述里不得与池内口径混用）：

* `tail_asym(T)`  = (up_over_10 + up_7_to_10 − down_over_10 − down_7_to_10) / total
                    取 15:00 那一行。**±7% 以上的极端尾桶的净不对称**。
* `slope(T)`      = up_share(15:00) − mean(up_share(09:30..09:59))，
                    up_share = up_count / (up_count+down_count+flat_count)。
                    **当日宽度路径的净漂移**（日内路径的一阶矩），**日内信息无法从日线还原**。
* `hhi(T)`        = Σ_b p_b²，p_b = 第 b 个收益桶 / total，取 15:00 那一行（11 个收益桶）。
                    **当日整市场截面收益分布的集中度**（形状，非尺度）。

三者都做 **z252**：`(x − roll_mean252(x)) / roll_std252(x, ddof=0)`，`min_count=120`
（沿用 `r4n_market_b2_v1` 已冻结的市场腿 z252 口径，禁用全样本标准化）。z252 是**有符号**的，
这是交互能成立的前提：状态为正时按个股腿排序、状态为负时反向，平均 IC 才是
「状态是否调制了该腿与未来收益的关系」的检验；若用恒正的状态（原始水平），
交互因子在**每一天**的截面排序都恒等于个股腿本身，那是结构性同构（§8.6），不登记。

三个机制的机制陈述与腿
----------------------
1. `breadth_tailskew_x_turnover`（方向 False）：整市场收益分布**上尾占优**（极端上涨桶
   相对极端下跌桶更重）＝全市场投机情绪高涨期。此时高换手波动（注意力/投机强度高）的
   股票最容易被推离基本面（Miller 1977 分歧 + 卖空约束 + 零售注意力），随后回吐，
   故交互项对未来 1/5 日收益**偏负**。腿 = 换手率 20 日标准差（正号版本，
   与生产 `turnover_std_20` 同口径、仅去掉其外置负号）。
2. `breadth_intraday_slope_x_beta`（方向 False）：宽度在**尾盘方向性抬升**
   （收盘 up_share 高于早盘均值）＝全市场买压集中在后半场。尾盘的聚合需求
   （指数/ETF 调仓、收盘集合竞价、羊群）会把**市场敏感度最高**的股票推过均衡价，
   次日部分回吐，故交互项偏负。腿 = 120 日市场 beta（对池内等权收益回归，
   `roll_cov/roll_var`，min_count=60）。
3. `breadth_concentration_x_reversal`（方向 False）：HHI 高＝整市场截面收益**紧聚在
   零附近**（低离散/安静盘面），个股 5 日涨跌更可能是流动性/噪声驱动，反转更强；
   HHI 低＝离散度大，价格移动更可能由信息驱动，倾向延续。故交互项
   （HHI 状态 × 5 日动量）对未来收益**偏负**。腿 = 5 日动量 `ctx.ret(5)`
   （其相反数是生产 `short_term_reversal_5`）。

★ 结构性披露（必须与结论一起读）：任何「(T,) 标量状态 × 单一 (T,C) 腿」的交互，
在**单个交易日内**的截面秩都恒等于该腿的秩（只差一个当日符号），因此
「与腿的当日截面 |Spearman| ≡ 1」是**构造的必然属性，不是发现**。候选的全部增量
信息在**符号/尺度的时间序列**上：RankIC 的日均值检验的是「状态是否调制了该腿与
未来收益的关系」。G5 应按**面板合并口径**（pooled）读数，本报告同时给出两列。

§8.8 强制重蹈检查（`doc/因子清理记录_20261003.md`，374 个已删股票因子）
--------------------------------------------------------------------
* 已删 `breadth` 族 10 个（`bw_beta_60` / `bw_intraday_beta_20` / `bw_session_follow_20` /
  `bw_capture_asym_20` / `bw_breadthvol_response_20` / `bw_tail_comove_60` /
  `bw_overnight_lag_beta_20` / `bw_beta_asym_20` / `bw_beta_change_20` /
  `bw_weak_breadth_ret_20`）与在库 `bw_strength_sensitivity_20` 都是**同一个构造**：
  用 **20–60 日滚动窗口**估计**个股自身**对「上涨家数份额 / 其变化」的 β、相关或条件均值，
  即**每只股票一个敏感度**。本单元不是敏感度：状态是**当日整市场分布的形状矩**
  （尾桶不对称 / 日内路径漂移 / 桶集中度），个股腿是**该股自身的特征**（换手波动 / beta /
  5 日动量），两者**相乘**得到状态条件化的排序。
* 已删 `field_markets` 族 8 个（`mfx_minute_amplitude` / `mfx_minute_volume_concentration` /
  `mfx_minute_weighted_pressure` / `mfx_index_pressure` 等）用的是
  **分钟线的振幅/量集中/加权压力与指数层**，**没有一个**用本表的 11 个收益桶分布形状。
* 已删 `sector` 族 15 个是行业层构造，与整市场桶分布无关。
* 在库触及本表的只有 2 个：`mkt_full_flat_ratio`（平盘占比，市场标量、无方向）与
  `bw_strength_sensitivity_20`（个股对「涨超 5% 占比」的 20 日相关，属上面的敏感度族）。
  **11 个收益桶的形状（尾桶不对称 / 桶集中度）与 241 点日内路径在此之前完全未被使用。**
* ⚠️ 实测：`limit_up_count` / `limit_down_count` 在 **2018、2019 全年恒为 0**
  （2018-01-02~2019-12-31 逐行 max = 0），2020 起才有值 ⇒ 任务书 F4
  （涨停/跌停不对称）**在声明的全窗口上不可运行**，本单元**不登记**该机制，
  以实测事实在 report.md 记录为 not-run。

PIT（可得性）合同
-----------------
* `stock_market_distribution_history` 的日期列 `trade_time`（`frequency.yaml` delay_days=0），
  行自 `trade_time` 起可见；15:00 那一行是**收盘快照**，生产 `mkt_full_flat_ratio`
  正是「取每日 15:00 那一行」，即 T 日收盘后可用 ⇒ 因子在 T 日取值、用于 T+1 开盘起的
  标签，无前视。早起腿（09:30–09:59）更早可得。
* `trade_time` 是**完整 datetime**（实测 241 个时间点/日，09:30…15:00 连续、无缺行），
  因此日内路径腿必须**先在日频聚合**再进入滚动窗口；本模块不做 (T, C, bucket) 立方体。
* 截断自证：`artifacts/research/round9_20261005/f_market_breadth_shape/probe_truncation.txt`。

实测数据事实（2026-10-05，本单元 probe）
----------------------------------------
* 全表 **980,147 行**（17 个年分区），每日**恰好 241 行**（09:30–15:00），
  17 个数据列**全部非空**（nonnull = 1.0000）；`bucket_sum == up+down+flat` 的比例 = 1.0000。
* 窗口内 15:00 行 **2,123/2,123** 全有（2018-01-02~2026-09-30）。
* `tail_asym` 分位 p1/p25/p50/p75/p99 = −0.1797 / 0.0038 / 0.0134 / 0.0247 / 0.1018。
* `slope`（15:00 − 早盘均值）分位 = −0.4582 / −0.0525 / 0.0932 / 0.2497 / 0.6788；
  与收盘 up_share 的 Spearman = +0.7101（与早盘水平 −0.339），日内 std 与 |slope| 的
  Spearman = +0.4681 ⇒ 斜率不是水平量的线性重标定。
* `hhi` 分位 = 0.2067 / 0.3259 / 0.3629 / 0.4024 / 0.5481；与当日上市家数的
  Spearman = **−0.0788** ⇒ 集中度不是「股票变多」的伪影。
* 在库市场因子 `mkt_strong_up_ratio` 用 **≥5%** 阈值 + **2,115 只冻结池**；
  本单元的尾桶腿用 **≥7% 分桶（7–10 与 >10）+ 全市场约 4,770 只**，两者口径不同。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PARAMETERS = {"signal": "breadth_tailskew_x_turnover"}

SIGNALS = (
    "breadth_tailskew_x_turnover",
    "breadth_intraday_slope_x_beta",
    "breadth_concentration_x_reversal",
)

STATE_WINDOW = 252        # z252（与 r4n 市场腿冻结口径一致）
STATE_MIN_COUNT = 120
VOL_WINDOW = 20           # 换手率波动窗（与生产 turnover_std_20 同口径）
VOL_MIN_COUNT = 10
BETA_WINDOW = 120
BETA_MIN_COUNT = 60
MOM_WINDOW = 5

BUCKETS = ("up_over_10", "up_7_to_10", "up_5_to_7", "up_3_to_5", "up_0_to_3", "zero",
           "down_0_to_3", "down_3_to_5", "down_5_to_7", "down_7_to_10", "down_over_10")
DIST_COLS = ["trade_time", "up_count", "down_count", "flat_count", *BUCKETS]
CLOSE_MINUTE = 15 * 60            # 15:00
MORNING_FIRST = 9 * 60 + 30       # 09:30
MORNING_LAST = 9 * 60 + 59        # 09:59


# ────────────────────────────────────────────────────────────────────
# 市场状态（全市场口径，日频聚合后再滚动）
# ────────────────────────────────────────────────────────────────────
def _market_shape(ctx):
    """返回 (T,) 的三个原始状态序列：tail_asym / slope / hhi（按面板交易日轴对齐）。"""
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    y0, y1 = int(dates[0]) // 10000, int(dates[-1]) // 10000
    frame = ctx.dataset("stock_market_distribution_history",
                        columns=list(DIST_COLS), years=(y0 - 1, y1))
    if frame is None or frame.empty:
        return None
    tt = pd.to_datetime(frame["trade_time"].astype(str), errors="coerce")
    day = (tt.dt.year.to_numpy(dtype=np.float64) * 10000
           + tt.dt.month.to_numpy(dtype=np.float64) * 100
           + tt.dt.day.to_numpy(dtype=np.float64))
    minute = (tt.dt.hour.to_numpy(dtype=np.float64) * 60 + tt.dt.minute.to_numpy(dtype=np.float64))
    ok = np.isfinite(day) & np.isfinite(minute)
    day = day[ok].astype(np.int64)
    minute = minute[ok].astype(np.int64)
    up = frame["up_count"].to_numpy(dtype=np.float64)[ok]
    dn = frame["down_count"].to_numpy(dtype=np.float64)[ok]
    fl = frame["flat_count"].to_numpy(dtype=np.float64)[ok]
    total = up + dn + fl
    with np.errstate(all="ignore"):
        up_share = np.where(total > 0, up / total, np.nan)
        probs = np.stack([frame[c].to_numpy(dtype=np.float64)[ok] for c in BUCKETS], axis=1)
        probs = probs / total[:, None]
        hhi = np.nansum(probs * probs, axis=1)
        tail = (frame["up_over_10"].to_numpy(dtype=np.float64)[ok]
                + frame["up_7_to_10"].to_numpy(dtype=np.float64)[ok]
                - frame["down_over_10"].to_numpy(dtype=np.float64)[ok]
                - frame["down_7_to_10"].to_numpy(dtype=np.float64)[ok])
        tail = np.where(total > 0, tail / total, np.nan)

    close_m = minute == CLOSE_MINUTE
    close = pd.DataFrame({"day": day[close_m], "up_share": up_share[close_m],
                          "hhi": hhi[close_m], "tail": tail[close_m]})
    close = close.drop_duplicates("day", keep="first").set_index("day")
    morning_m = (minute >= MORNING_FIRST) & (minute <= MORNING_LAST)
    morning = (pd.DataFrame({"day": day[morning_m], "up_share": up_share[morning_m]})
               .groupby("day")["up_share"].mean())
    shape = close.join(morning.rename("up_share_am"), how="left").reindex(dates)
    return {"tail_asym": shape["tail"].to_numpy(dtype=np.float64),
            "slope": (shape["up_share"] - shape["up_share_am"]).to_numpy(dtype=np.float64),
            "hhi": shape["hhi"].to_numpy(dtype=np.float64)}


def _z252(ctx, series):
    """252 日滚动 z（min_count=120，ddof=0）——与 r4n 市场腿冻结口径逐位一致。

    sd 用 `mean(x²) − mean(x)²` 自算：`ctx.roll_mean` 的分母是窗口内**有效值个数**，
    两个窗口的有效集合相同，故该式在缺失下仍严格等于 ddof=0 的总体标准差；
    绕开 `ctx.roll_std` 在 min_count<n 时的已知口径坑（见 factors/liquidity.py 注释）。
    """
    col = np.asarray(series, dtype=np.float64)[:, None]
    mean = ctx.roll_mean(col, STATE_WINDOW, min_count=STATE_MIN_COUNT)
    mean_sq = ctx.roll_mean(col * col, STATE_WINDOW, min_count=STATE_MIN_COUNT)
    var = mean_sq - mean * mean
    sd = np.sqrt(np.where(var > 0, var, np.nan))
    with np.errstate(all="ignore"):
        z = (col - mean) / sd
    return np.where(np.isfinite(z[:, 0]), z[:, 0], np.nan)


# ────────────────────────────────────────────────────────────────────
# 个股腿
# ────────────────────────────────────────────────────────────────────
def _valid_pool(ctx):
    return (np.asarray(ctx.universe, dtype=bool) & np.asarray(ctx.traded(), dtype=bool))


def _turnover_vol(ctx):
    """换手率 20 日标准差（ddof=0，min_count=10；生产 turnover_std_20 去掉外置负号的同口径）。

    流通股本口径换手率（`ctx.px('turnover')`，% 量纲），停牌/缺测不计入。
    """
    t = np.asarray(ctx.px("turnover"), dtype=np.float64)
    valid = _valid_pool(ctx) & np.isfinite(t)
    tv = np.where(valid, t, np.nan)
    mean = ctx.roll_mean(tv, VOL_WINDOW, min_count=VOL_MIN_COUNT)
    mean_sq = ctx.roll_mean(tv * tv, VOL_WINDOW, min_count=VOL_MIN_COUNT)
    var = mean_sq - mean * mean
    sd = np.sqrt(np.where(var > 0, var, np.nan))
    return np.where(valid, sd, np.nan)


def _beta120(ctx):
    """120 日市场 beta：对池内等权已成交收益回归（roll_cov/roll_var，min_count=60）。"""
    r = np.asarray(ctx.ret_clean(1), dtype=np.float64)
    valid = _valid_pool(ctx) & np.isfinite(r)
    count = valid.sum(axis=1)
    pool_ret = np.where(valid, r, 0.0).sum(axis=1) / np.maximum(count, 1)
    pool_ret = np.where(count >= int(ctx.cfg.min_cross_section), pool_ret, np.nan)
    market = np.repeat(pool_ret[:, None], r.shape[1], axis=1)
    cov = ctx.roll_cov(r, market, BETA_WINDOW, BETA_MIN_COUNT)
    var = ctx.roll_var(market, BETA_WINDOW, BETA_MIN_COUNT)
    with np.errstate(all="ignore"):
        beta = np.where(var > 0, cov / var, np.nan)
    return np.where(valid, beta, np.nan)


def _mom5(ctx):
    """5 日动量（`ctx.ret(5)`；其相反数是生产 `short_term_reversal_5`）。"""
    r5 = np.asarray(ctx.ret(5), dtype=np.float64)
    valid = _valid_pool(ctx) & np.isfinite(r5)
    return np.where(valid, r5, np.nan)


def _interaction(ctx, state_key, leg_fn):
    shape = _market_shape(ctx)
    if shape is None:
        return None
    state = _z252(ctx, shape[state_key])
    leg = leg_fn(ctx)
    with np.errstate(all="ignore"):
        grid = state[:, None] * leg
    return np.where(np.isfinite(state)[:, None] & np.isfinite(leg), grid, np.nan)


_SIGNALS = {
    "breadth_tailskew_x_turnover": lambda ctx: _interaction(ctx, "tail_asym", _turnover_vol),
    "breadth_intraday_slope_x_beta": lambda ctx: _interaction(ctx, "slope", _beta120),
    "breadth_concentration_x_reversal": lambda ctx: _interaction(ctx, "hhi", _mom5),
}


def build(parameters):
    """builder：参数只允许预登记的 signal 名（family allowed_variants 逐项冻结）。"""
    params = dict(parameters)
    signal = params.get("signal")
    if signal not in SIGNALS:
        raise ValueError(f"Family permits only preregistered signals: {sorted(SIGNALS)}")
    if set(params) != {"signal"}:
        raise ValueError(f"Unregistered parameters: {sorted(set(params) - {'signal'})}")
    fn = _SIGNALS[str(signal)]

    def compute(ctx):
        value = fn(ctx)
        if value is None:
            value = ctx.panel.empty(dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    compute.__name__ = "r9f_" + str(signal)
    return compute
