"""Round 4 · 批 3 · V6 市场批 3（family `r4o_market_b3_v1`）。

任务书（唯一权威）：`artifacts/research/round4_20261004/RESEARCH_BRIEF_B3.md` §V6
（通用纪律 = B1 §0–§3、§5–§8 原样适用）。
机制集合、通道、方向均为 Lead 事先固定；本模块只做实现，**不得**在看过任何结果后
修改方向、增删机制或扩大参数（登记后不可改判通道）。

通道预登记（5 个机制，每个恰一个候选，无参数变体）：
- 严格通道（1 个）：`mkt_amount_price_divergence_5` —— z252(5日均成交额同比变化)
  − z252(等权池 5 日收益)；**方向 −1**（放量滞涨 → 未来偏弱；量价背离的成交额侧表达，
  与 b2 的宽度侧背离机制独立，独立性论证见 freeze b3 / report.md）。
  须过 M0–M7 全部（M1 = research verify 实跑）。
- 绿卡通道（4 个，无方向）：`mkt_cs_q90_q10_spread`（截面 90–10 分位差 20 日均）/
  `mkt_overnight_vol20`（隔夜收益 20 日 std）/ `mkt_amount_median`（池内成交额中位数）/
  `mkt_cs_absret_median`（|日收益| 截面中位数）；只过 M0∧M2∧M3（M4 只记录）。

实现口径对齐已验收的生产市场族（`factors/market.py` 范式，勿偏离）：
- 固定 2115 只主板池；有效样本分母（缺测/停牌不计入）；有效样本 < 100（cfg.min_cross_section）
  的交易日记 NaN；市场标量不做截面 rank。
- 成交额按生产 `_base` 口径：`amount = where(universe & finite(ret), px('amount'), NaN)`
  （停牌日 ret 为 NaN → 不计入）；合计需 ≥ min_cross_section 个有效样本。
- 分位数与中位数用 `np.nanquantile`（线性插值，生产 `_quantile` 同口径）。
- 缺失保持 NaN，不用未来值回填、不零填充、不前向填充伪造成交。
- T 日值只用 T 日及以前的日线数据；`stock_daily` / `stock_adj_factor` 均不在
  `fea/delay.py` 的可得性滞后名单，无需 `lag_grid`。

**冻结口径（先于任何结果统计成文，写入 freeze b3）**：
- z252 = 252 个交易日滚动 z——`(x−roll_mean252(x))/roll_std252(x,ddof=0)`，min_count=120
  （与批 2/批 3 任务书的 z252 冻结口径一致；禁用全样本标准化）；任一腿 NaN → 结果 NaN。
- 严格候选「5日均成交额同比变化」的唯一冻结解读：`amt5(T) = MA5(池内成交额合计, min_count=5)`；
  `yoy(T) = amt5(T) / amt5(T−252) − 1`（**同比 = 252 交易日 ≈ 1 年**，与 z252 同一交易日单位）；
  另一可读解读（逐日同比再 5 日平均）在 freeze b3 edge_notes 披露为事前已记录备选、不参与本单元判定。
- 20 日滚动窗口 min_count=15、ddof=0（与批 2 绿卡同口径）。

本模块**不注册任何正式因子**（无 `register` 调用）：只作为 research 隔离候选的实现，
通过 `build(parameters)` 返回 `callable(ctx) -> (T,) float64`。
"""
from __future__ import annotations

import warnings

import numpy as np


# ────────────────────────────────────────────────────────────────────
# 与 factors/market.py 同口径的纯函数（分母保护 / 均值 / 中位数 / 分位数）
# ────────────────────────────────────────────────────────────────────
def _div(a, b):
    """带分母保护的除法：分母非正或 NaN → NaN（与生产市场族 `_div` 逐行为一致）。"""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=np.isfinite(b) & (b > 0))
    return out


def _mean(x):
    """逐行有限值均值：nansum / 有限计数（全 NaN 行 → NaN）。"""
    return _div(np.nansum(x, axis=1), np.isfinite(x).sum(axis=1))


def _median(x):
    """逐行有限值中位数（全 NaN 行 → NaN，RuntimeWarning 静默）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.asarray(x, dtype=float), axis=1)


def _quantile(x, q):
    """逐行有限值分位数（生产 `_quantile` 同口径：np.nanquantile 线性插值）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanquantile(np.asarray(x, dtype=float), q, axis=1)


def _sufficient(ctx, valid):
    """有效样本数达到 min_cross_section 的交易日掩码。"""
    return valid.sum(axis=1) >= ctx.cfg.min_cross_section


def _ret_matrix(ctx):
    """停牌日置 NaN 的日收益网格（与生产市场族 `_base` 的 r 口径一致）。"""
    ret = ctx.ret_clean(1)
    valid = ctx.universe & np.isfinite(ret)
    return np.where(valid, ret, np.nan), valid


def _ew_return(ctx, r, valid):
    """等权池日收益（有效样本不足日 → NaN）。与生产日线等权收益同口径。"""
    return np.where(_sufficient(ctx, valid), _mean(r), np.nan)


def _roll_z252(ctx, x):
    """252 日滚动 z（min_count=120，ddof=0 总体标准差；sd<=0 或 x NaN → NaN）。

    冻结口径：与批 2/批 3 任务书的 z252(min120) 一致；禁用全样本标准化。
    """
    mean = ctx.roll_mean(x[:, None], 252, min_count=120)[:, 0]
    sd = ctx.roll_std(x[:, None], 252, min_count=120, ddof=0)[:, 0]
    out = np.full(x.shape, np.nan)
    good = np.isfinite(x) & np.isfinite(mean) & np.isfinite(sd) & (sd > 0)
    np.divide(x - mean, sd, out=out, where=good)
    return out


# ────────────────────────────────────────────────────────────────────
# 五个预登记机制的实现
# ────────────────────────────────────────────────────────────────────
def _sig_amount_price_divergence_5(ctx):
    """mkt_amount_price_divergence_5（严格，方向 −1）：
    z252(5日均成交额同比变化) − z252(等权池 5 日累计收益)。

    冻结口径：amt5 = MA5(池内成交额合计, min_count=5)；
    yoy(T) = amt5(T)/amt5(T−252) − 1（同比 = 252 交易日）；
    值 = z252(yoy) − z252(ew5)，ew5 = 复利 5 日等权池收益。
    机制：放量滞涨（成交额腿显著为正、价格腿不涨）→ 未来 1/5 日市场收益偏弱；
    反向（缩量上涨）偏强。两腿各自 252 日滚动 z（min120）再相减。
    """
    r, valid = _ret_matrix(ctx)
    a = np.where(valid, ctx.px("amount"), np.nan)
    totals = np.nansum(a, axis=1)
    totals = np.where(np.isfinite(a).sum(axis=1) >= ctx.cfg.min_cross_section,
                      totals, np.nan)
    amt5 = ctx.roll_mean(totals[:, None], 5, min_count=5)[:, 0]
    yoy = np.full(amt5.shape, np.nan)
    good = (np.isfinite(amt5[252:]) & np.isfinite(amt5[:-252])
            & (amt5[252:] > 0) & (amt5[:-252] > 0))
    yoy[252:] = np.where(good, amt5[252:] / amt5[:-252] - 1.0, np.nan)
    ew = _ew_return(ctx, r, valid)
    grown = np.where(np.isfinite(ew), 1.0 + ew, np.nan)
    ew5 = ctx.roll_prod(grown[:, None], 5, min_count=5)[:, 0] - 1.0
    return _roll_z252(ctx, yoy) - _roll_z252(ctx, ew5)


def _sig_cs_q90_q10_spread(ctx):
    """mkt_cs_q90_q10_spread（绿卡）：池内日收益截面 90–10 分位差的 20 日均值。

    分位数口径 = 生产 `_quantile`（np.nanquantile 线性插值）；spread = q90(r) − q10(r)；
    有效票 < 100 的日记 NaN；20 日均 min_count=15。
    """
    r, valid = _ret_matrix(ctx)
    spread = np.where(_sufficient(ctx, valid),
                      _quantile(r, 0.90) - _quantile(r, 0.10), np.nan)
    return ctx.roll_mean(spread[:, None], 20, min_count=15)[:, 0]


def _sig_overnight_vol20(ctx):
    """mkt_overnight_vol20（绿卡）：池内等权隔夜收益的 20 日滚动标准差（ddof=0，min15）。

    隔夜收益逐股 = open/pre_close − 1（当日原始 open 与厂商已除权调整的 pre_close，
    与 r4m `mkt_overnight_ret` 同口径）；池序列 = 有效票均值；
    有效票 < 100 的日记 NaN；滚动窗口 ≥15 个有效值。
    """
    o = ctx.px_raw("open")
    pc = ctx.px_raw("pre_close")
    valid = ctx.universe & ctx.traded() & np.isfinite(o) & np.isfinite(pc) & (pc > 0)
    ov = np.where(valid, _div(o, pc) - 1.0, np.nan)
    nightly = np.where(_sufficient(ctx, valid), _mean(ov), np.nan)
    return ctx.roll_std(nightly[:, None], 20, min_count=15, ddof=0)[:, 0]


def _sig_amount_median(ctx):
    """mkt_amount_median（绿卡）：池内成交额中位数（日频原值）。

    成交额按生产 `_base` 口径掩码（universe & finite(ret)，停牌日不计）；
    有效票 < 100 的日记 NaN。
    """
    r, valid = _ret_matrix(ctx)
    a = np.where(valid, ctx.px("amount"), np.nan)
    return np.where(_sufficient(ctx, valid), _median(a), np.nan)


def _sig_cs_absret_median(ctx):
    """mkt_cs_absret_median（绿卡）：|日收益| 截面中位数（日频原值）。

    r 为停牌置 NaN 的日收益；有效票 < 100 的日记 NaN。
    """
    r, valid = _ret_matrix(ctx)
    absr = np.where(valid, np.abs(r), np.nan)
    return np.where(_sufficient(ctx, valid), _median(absr), np.nan)


_SIGNALS = {
    "mkt_amount_price_divergence_5": _sig_amount_price_divergence_5,
    "mkt_cs_q90_q10_spread": _sig_cs_q90_q10_spread,
    "mkt_overnight_vol20": _sig_overnight_vol20,
    "mkt_amount_median": _sig_amount_median,
    "mkt_cs_absret_median": _sig_cs_absret_median,
}


def build(parameters):
    """builder：parameters 只允许预登记的 signal 名（family allowed_variants 逐项冻结）。"""
    signal = str(parameters["signal"])
    if signal not in _SIGNALS:
        raise ValueError(f"未预登记的机制：{signal}")
    fn = _SIGNALS[signal]

    def compute(ctx):
        return fn(ctx)

    compute.__name__ = "r4o_" + signal
    return compute
