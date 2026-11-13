"""Round 4 · 批 1 · M1 市场状态补充（family `r4m_market_state_v1`）。

任务书（唯一权威）：`artifacts/research/round4_20261004/RESEARCH_BRIEF_B1.md` §4-M1。
机制集合、通道、方向均为 Lead 事先固定；本模块只做实现，**不得**在看过任何结果后
修改方向、增删机制或扩大参数（登记后不可改判通道）。

通道预登记（6 个机制，每个恰一个候选，无参数变体）：
- 严格通道（1 个）：`mkt_ew_reversal_5` —— 等权池过去 5 日累计收益（方向 −1）；
  须过 M0–M7 全部（方向与先验强度封存于 round4 freeze 方向表）。
- 绿卡通道（5 个，无方向）：`mkt_cs_skew_20` / `mkt_cs_kurtosis_20` /
  `mkt_turnover_median` / `mkt_amount_total_log`（边界案例，判定理由见 freeze） /
  `mkt_overnight_ret`；只过 M0∧M2∧M3（M4 只记录）。

实现口径对齐已验收的生产市场族（`factors/market.py` 范式，勿偏离）：
- 固定 2115 只主板池；有效样本分母（缺测/停牌不计入）；有效样本 < 100（cfg.min_cross_section）
  的交易日记 NaN；市场标量不做截面 rank。
- 缺失保持 NaN，不用未来值回填、不零填充、不前向填充伪造成交。
- T 日值只用 T 日及以前的日线数据；`stock_daily` / `stock_adj_factor` / `stock_finance`
  均不在 `fea/delay.py` 的可得性滞后名单，无需 `lag_grid`。
- 本模块**不注册任何正式因子**（无 `register` 调用）：只作为 research 隔离候选的实现，
  通过 `build(parameters)` 返回 `callable(ctx) -> (T,) float64`。
"""
from __future__ import annotations

import warnings

import numpy as np


# ────────────────────────────────────────────────────────────────────
# 与 factors/market.py 同口径的纯函数（分母保护 / 均值 / 中位数）
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


def _sufficient(ctx, valid):
    """有效样本数达到 min_cross_section 的交易日掩码。"""
    return valid.sum(axis=1) >= ctx.cfg.min_cross_section


def _ret_matrix(ctx):
    """停牌日置 NaN 的日收益网格（与生产市场族 `_base` 的 r 口径一致）。"""
    ret = ctx.ret_clean(1)
    valid = ctx.universe & np.isfinite(ret)
    return np.where(valid, ret, np.nan), valid


def _cs_stats(ctx):
    """日频截面矩：返回 (r, valid, mean, skew(非超额), kurt(非超额))。

    口径与 `factors/market.py::_stats` 一致：sd 为总体标准差（ddof=0）、
    skew=mean(z^3)/sd^3；kurt 同法取四阶（**非超额峰度**，正态下约 3）。
    有效样本不足日 → skew/kurt 记 NaN。
    """
    r, valid = _ret_matrix(ctx)
    mean = _mean(r)
    z = r - mean[:, None]
    sd = np.sqrt(_mean(z * z))
    skew = _div(_mean(z ** 3), sd ** 3)
    kurt = _div(_mean(z ** 4), sd ** 4)
    ok = _sufficient(ctx, valid)
    return r, valid, mean, np.where(ok, skew, np.nan), np.where(ok, kurt, np.nan)


def _ew_return(ctx):
    """等权池日收益（有效样本不足日 → NaN）。与生产日线等权收益同口径。"""
    r, valid = _ret_matrix(ctx)
    return np.where(_sufficient(ctx, valid), _mean(r), np.nan)


# ────────────────────────────────────────────────────────────────────
# 六个预登记机制的实现
# ────────────────────────────────────────────────────────────────────
def _sig_ew_reversal_5(ctx):
    """mkt_ew_reversal_5：等权池过去 5 日累计收益（复利），方向登记 −1。

    cum5(T) = ∏_{k=0..4} (1 + ew(T−k)) − 1；窗口内任一 NaN（含样本不足日）→ NaN。
    """
    ew = _ew_return(ctx)
    grown = np.where(np.isfinite(ew), 1.0 + ew, np.nan)
    prod5 = ctx.roll_prod(grown[:, None], 5, min_count=5)[:, 0]
    return prod5 - 1.0


def _sig_cs_skew_20(ctx):
    """mkt_cs_skew_20：池内日收益截面偏度的 20 日均值（min_count=15）。"""
    _, _, _, skew, _ = _cs_stats(ctx)
    return ctx.roll_mean(skew[:, None], 20, min_count=15)[:, 0]


def _sig_cs_kurtosis_20(ctx):
    """mkt_cs_kurtosis_20：池内日收益截面（非超额）峰度的 20 日均值（min_count=15）。"""
    _, _, _, _, kurt = _cs_stats(ctx)
    return ctx.roll_mean(kurt[:, None], 20, min_count=15)[:, 0]


def _sig_turnover_median(ctx):
    """mkt_turnover_median：池内换手率中位数（日频原值；`ctx.px('turnover')` 百分数口径）。"""
    t = ctx.px("turnover")
    valid = ctx.universe & ctx.traded() & np.isfinite(t)
    tv = np.where(valid, t, np.nan)
    return np.where(_sufficient(ctx, valid), _median(tv), np.nan)


def _sig_amount_total_log(ctx):
    """mkt_amount_total_log：log(池内成交额合计)；边界案例（见 freeze 判定理由）。"""
    a = ctx.px("amount")
    valid = ctx.universe & np.isfinite(a)
    av = np.where(valid, a, np.nan)
    total = np.nansum(av, axis=1)
    total = np.where(_sufficient(ctx, valid), total, np.nan)
    out = np.full(total.shape, np.nan)
    good = np.isfinite(total) & (total > 0)
    np.log(total, out=out, where=good)
    return out


def _sig_overnight_ret(ctx):
    """mkt_overnight_ret：池内 open/pre_close−1 的均值（等权隔夜收益，日频原值）。"""
    o = ctx.px_raw("open")
    pc = ctx.px_raw("pre_close")
    valid = ctx.universe & ctx.traded() & np.isfinite(o) & np.isfinite(pc) & (pc > 0)
    r = np.where(valid, _div(o, pc) - 1.0, np.nan)
    return np.where(_sufficient(ctx, valid), _mean(r), np.nan)


_SIGNALS = {
    "mkt_ew_reversal_5": _sig_ew_reversal_5,
    "mkt_cs_skew_20": _sig_cs_skew_20,
    "mkt_cs_kurtosis_20": _sig_cs_kurtosis_20,
    "mkt_turnover_median": _sig_turnover_median,
    "mkt_amount_total_log": _sig_amount_total_log,
    "mkt_overnight_ret": _sig_overnight_ret,
}


def build(parameters):
    """builder：parameters 只允许预登记的 signal 名（family allowed_variants 逐项冻结）。"""
    signal = str(parameters["signal"])
    if signal not in _SIGNALS:
        raise ValueError(f"未预登记的机制：{signal}")
    fn = _SIGNALS[signal]

    def compute(ctx):
        return fn(ctx)

    compute.__name__ = "r4m_" + signal
    return compute
