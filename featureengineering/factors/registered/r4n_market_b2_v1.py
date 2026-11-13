"""Round 4 · 批 2 · U11 市场批 2（family `r4n_market_b2_v1`）。

任务书（唯一权威）：`artifacts/research/round4_20261004/RESEARCH_BRIEF_B2.md` §6-U11
（通用纪律 = B1 §0–§3、§5–§8 原样适用）。
机制集合、通道、方向均为 Lead 事先固定；本模块只做实现，**不得**在看过任何结果后
修改方向、增删机制或扩大参数（登记后不可改判通道）。

通道预登记（5 个机制，每个恰一个候选，无参数变体）：
- 严格通道（1 个）：`mkt_breadth_price_divergence_5` —— z252(池内上涨家数占比的 5 日均)
  − z252(等权池 5 日收益)；**方向 +1**（宽度改善领先价格：背离时以宽度为方向）。
  须过 M0–M7 全部（M1 = research verify 实跑）。
- 绿卡通道（4 个，无方向）：`mkt_cs_median_ret1` / `mkt_turnover_median_std20` /
  `mkt_idx_csi300_amount_log` / `mkt_ew_amplitude`；只过 M0∧M2∧M3（M4 只记录）。

实现口径对齐已验收的生产市场族（`factors/market.py` 范式，勿偏离）：
- 固定 2115 只主板池；有效样本分母（缺测/停牌不计入）；有效样本 < 100（cfg.min_cross_section）
  的交易日记 NaN；市场标量不做截面 rank。
- `up_ratio` 逐位复用生产口径：count(r>0)/count(valid)，比较含 1e-12 级容差（r>1e-12）。
- 缺失保持 NaN，不用未来值回填、不零填充、不前向填充伪造成交。
- T 日值只用 T 日及以前的日线数据；`stock_daily` / `stock_adj_factor` / `stock_finance` /
  `index_daily` 均不在 `fea/delay.py` 的可得性滞后名单，无需 `lag_grid`。
- **冻结口径（先于任何结果统计成文，写入 freeze b2）**：严格候选两个腿的 z252 =
  252 个交易日滚动 z——`(x−roll_mean252(x))/roll_std252(x,ddof=0)`，min_count=120
  （与批 2 任务书 U4/U8 的 z252 冻结口径一致；禁用全样本标准化）；任一腿 NaN → 结果 NaN。
- 本模块**不注册任何正式因子**（无 `register` 调用）：只作为 research 隔离候选的实现，
  通过 `build(parameters)` 返回 `callable(ctx) -> (T,) float64`。
"""
from __future__ import annotations

import warnings

import numpy as np


# ────────────────────────────────────────────────────────────────────
# 与 factors/market.py 同口径的纯函数（分母保护 / 容差比较 / 均值 / 中位数）
# ────────────────────────────────────────────────────────────────────
def _div(a, b):
    """带分母保护的除法：分母非正或 NaN → NaN（与生产市场族 `_div` 逐行为一致）。"""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=np.isfinite(b) & (b > 0))
    return out


def _gt(a, b):
    """严格大于（生产 `_gt` 逐位一致）：含 1e-12+1e-12*abs(b) 容差。"""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return np.isfinite(a) & np.isfinite(b) & ((a - b) > (1e-12 + 1e-12 * np.abs(b)))


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


def _up_ratio(ctx, r, valid):
    """池内上涨家数占比（生产 `mkt_up_ratio` 逐位口径：count(r>0)/count(valid)，容差 1e-12）。"""
    n = valid.sum(axis=1)
    hit = (_gt(r, 0.0) & valid).sum(axis=1)
    return np.where(n >= ctx.cfg.min_cross_section, _div(hit, n), np.nan)


def _ew_return(ctx, r, valid):
    """等权池日收益（有效样本不足日 → NaN）。与生产日线等权收益同口径。"""
    return np.where(_sufficient(ctx, valid), _mean(r), np.nan)


def _roll_z252(ctx, x):
    """252 日滚动 z（min_count=120，ddof=0 总体标准差；sd<=0 或 x NaN → NaN）。

    冻结口径：与批 2 任务书 U4/U8 的 z252(min120) 一致；禁用全样本标准化。
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
def _sig_breadth_price_divergence_5(ctx):
    """mkt_breadth_price_divergence_5（严格，方向 +1）：
    z252(上涨家数占比的 5 日均) − z252(等权池 5 日累计收益)。

    机制：宽度改善领先价格——价格腿短期落后于宽度腿时（背离为正），未来市场收益偏正；
    背离为负（价格超前于宽度）时偏负。两腿均先做 252 日滚动 z（min120）再相减。
    """
    r, valid = _ret_matrix(ctx)
    up = _up_ratio(ctx, r, valid)
    ew = _ew_return(ctx, r, valid)
    up5 = ctx.roll_mean(up[:, None], 5, min_count=5)[:, 0]
    grown = np.where(np.isfinite(ew), 1.0 + ew, np.nan)
    ew5 = ctx.roll_prod(grown[:, None], 5, min_count=5)[:, 0] - 1.0
    return _roll_z252(ctx, up5) - _roll_z252(ctx, ew5)


def _sig_cs_median_ret1(ctx):
    """mkt_cs_median_ret1（绿卡）：池内单日收益中位数（日频原值；有效票 < 100 → NaN）。"""
    r, valid = _ret_matrix(ctx)
    med = _median(r)
    return np.where(_sufficient(ctx, valid), med, np.nan)


def _sig_turnover_median_std20(ctx):
    """mkt_turnover_median_std20（绿卡）：池内换手率中位数的 20 日滚动标准差（ddof=0，min15）。

    换手率 = ctx.px('turnover')（vol/float_share*100 百分数口径，与生产/r4m 同）；
    有效票 < 100 的日中位数记 NaN；滚动窗口至少 15 个有效值，否则 NaN。
    """
    t = ctx.px("turnover")
    valid = ctx.universe & ctx.traded() & np.isfinite(t)
    tv = np.where(valid, t, np.nan)
    med = np.where(_sufficient(ctx, valid), _median(tv), np.nan)
    return ctx.roll_std(med[:, None], 20, min_count=15, ddof=0)[:, 0]


def _sig_idx_csi300_amount_log(ctx):
    """mkt_idx_csi300_amount_log（绿卡）：沪深300 成交额 log（直接指数行情的值域压缩变换）。

    读取方式与生产指数族 `_index` 一致：`ctx.dataset('index_daily', ...)` 按面板年范围
    读取、重复主键检查、reindex 到面板交易日轴；amount<=0 或缺失 → NaN。
    """
    import pandas as pd
    from fea.dates import series_to_int
    dates = ctx.panel.dates
    years = (int(dates[0]) // 10000, int(dates[-1]) // 10000)
    df = ctx.dataset("index_daily", columns=["ts_code", "trade_date", "amount"], years=years)
    df = df[df.ts_code == "000300.SH"].copy()
    if df.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("指数日线含重复主键（000300.SH）")
    df["day"] = series_to_int(df.trade_date)
    amt = df.set_index("day")["amount"].reindex(dates).to_numpy(dtype=float)
    out = np.full(amt.shape, np.nan)
    good = np.isfinite(amt) & (amt > 0)
    np.log(amt, out=out, where=good)
    return out


def _sig_ew_amplitude(ctx):
    """mkt_ew_amplitude（绿卡）：池内 (high−low)/pre_close 的均值（等权平均振幅，日频原值）。

    用 ctx.px_raw 取当日原始 high/low/pre_close（pre_close 为厂商已除权调整的前收，
    A 股振幅标准口径）；停牌/缺价/ pre_close<=0 不计入分母；有效票 < 100 → NaN。
    """
    hi = ctx.px_raw("high")
    lo = ctx.px_raw("low")
    pc = ctx.px_raw("pre_close")
    valid = (ctx.universe & ctx.traded() & np.isfinite(hi) & np.isfinite(lo)
             & np.isfinite(pc) & (pc > 0))
    amp = np.where(valid, _div(hi - lo, pc), np.nan)
    return np.where(_sufficient(ctx, valid), _mean(amp), np.nan)


_SIGNALS = {
    "mkt_breadth_price_divergence_5": _sig_breadth_price_divergence_5,
    "mkt_cs_median_ret1": _sig_cs_median_ret1,
    "mkt_turnover_median_std20": _sig_turnover_median_std20,
    "mkt_idx_csi300_amount_log": _sig_idx_csi300_amount_log,
    "mkt_ew_amplitude": _sig_ew_amplitude,
}


def build(parameters):
    """builder：parameters 只允许预登记的 signal 名（family allowed_variants 逐项冻结）。"""
    signal = str(parameters["signal"])
    if signal not in _SIGNALS:
        raise ValueError(f"未预登记的机制：{signal}")
    fn = _SIGNALS[signal]

    def compute(ctx):
        return fn(ctx)

    compute.__name__ = "r4n_" + signal
    return compute
