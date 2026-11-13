"""Daily intraday pressure statistics, exposed through explicit production builders."""
from __future__ import annotations

import numpy as np

SIGNALS = (
    "turnover_weighted_intraday_return_path",
    "range_benchmarked_semivariance_excess",
    "tail30_signed_flow_pressure",
)

def _f(ctx, field):
    return np.asarray(ctx.intraday_field(field), dtype=np.float64)


def _finish(ctx, value):
    n_bars = _f(ctx, "n_bars")
    valid = (np.asarray(ctx.universe, dtype=bool) &
             np.asarray(ctx.traded(), dtype=bool) & np.isfinite(value) &
             (n_bars == 48))
    return np.where(valid, value, np.nan)


def _fixed_window_mean(values, width, min_count=None):
    """Same ordered fixed-window sum used by the frozen source implementation."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"fixed_window_mean expects (T,C), got {x.shape}")
    width = int(width)
    if width <= 0:
        raise ValueError("width must be positive")
    need = width if min_count is None else int(min_count)
    if need < 0 or need > width:
        raise ValueError("min_count must be in [0,width]")
    t, c = x.shape
    sums = np.zeros((t, c), dtype=np.float64)
    count = np.zeros((t, c), dtype=np.int16)
    for lag in range(width):
        shifted = np.full((t, c), np.nan, dtype=np.float64)
        if lag == 0:
            shifted[:] = x
        elif lag < t:
            shifted[lag:] = x[:-lag]
        finite = np.isfinite(shifted)
        sums += np.where(finite, shifted, 0.0)
        count += finite.astype(np.int16)
    out = np.full((t, c), np.nan, dtype=np.float64)
    good = count >= need
    np.divide(sums, count, out=out, where=good & (count > 0))
    if need == 0:
        out[count == 0] = 0.0
    return out


def _mean5(ctx, x):
    x = np.asarray(x, dtype=np.float64)
    good_day = _f(ctx, "n_bars") == 48
    x = np.where(good_day, x, np.nan)
    out = _fixed_window_mean(x, 5, min_count=4)
    return np.where(np.isfinite(x), out, np.nan)


def _turnover_weighted_intraday_return_path(ctx):
    traded = np.asarray(ctx.traded(), dtype=bool)
    op, cl = _f(ctx, "open5"), _f(ctx, "close5")
    ret = np.full(op.shape, np.nan, dtype=np.float64)
    valid_price = (traded & (_f(ctx, "n_bars") == 48) & np.isfinite(op) & (op > 0) &
                   np.isfinite(cl) & (cl > 0))
    ret[valid_price] = np.log(cl[valid_price] / op[valid_price])
    turn = np.asarray(ctx.px("turnover"), dtype=np.float64)
    ok = np.isfinite(ret) & np.isfinite(turn) & (turn >= 0)
    r = np.where(ok, ret, np.nan)
    w = np.where(ok, np.log1p(turn), np.nan)
    wr = np.where(ok, r * w, np.nan)
    n = ctx.roll_count(r, 20, min_count=15)
    sum_r = ctx.roll_sum(r, 20, min_count=15)
    sum_w = ctx.roll_sum(w, 20, min_count=15)
    sum_wr = ctx.roll_sum(wr, 20, min_count=15)
    equal = sum_r / np.maximum(n, 1.0)
    weighted = np.full(ret.shape, np.nan)
    np.divide(sum_wr, sum_w, out=weighted,
              where=np.isfinite(sum_wr) & np.isfinite(sum_w) & (sum_w > 1e-10))
    return _finish(ctx, np.where(ok, weighted - equal, np.nan))


def _range_benchmarked_semivariance_excess(ctx):
    pos = _f(ctx, "posret2_sum")
    neg = _f(ctx, "negret2_sum")
    hi = np.maximum(_f(ctx, "hi5"), np.fmax(_f(ctx, "open5"), _f(ctx, "close5")))
    lo = np.minimum(_f(ctx, "lo5"), np.fmin(_f(ctx, "open5"), _f(ctx, "close5")))
    ok = (np.isfinite(pos) & (pos >= 0) & np.isfinite(neg) & (neg >= 0) &
          np.isfinite(hi) & (hi > 0) & np.isfinite(lo) & (lo > 0) & (hi >= lo))
    cont = np.full(pos.shape, np.nan)
    cont[ok] = np.log(hi[ok] / lo[ok]) ** 2 / (4.0 * np.log(2.0))
    jpos = np.maximum(pos - 0.5 * cont, 0.0)
    jneg = np.maximum(neg - 0.5 * cont, 0.0)
    excess = jpos + jneg
    daily = np.full(pos.shape, np.nan)
    daily[ok] = np.log1p(1e4 * excess[ok] / (cont[ok] + 1e-8))
    return _finish(ctx, _mean5(ctx, daily))


def _tail30_signed_flow_pressure(ctx):
    op = np.asarray(ctx.closing30_field("open_tail30"), dtype=np.float64)
    cl = np.asarray(ctx.closing30_field("close_tail30"), dtype=np.float64)
    tail_amt = np.asarray(ctx.closing30_field("amt_tail30"), dtype=np.float64)
    day_amt = np.asarray(ctx.closing30_field("amt_day"), dtype=np.float64)
    ok = (np.isfinite(op) & (op > 0) & np.isfinite(cl) & (cl > 0) &
          np.isfinite(tail_amt) & (tail_amt >= 0) & np.isfinite(day_amt) & (day_amt > 0))
    daily = np.full(op.shape, np.nan)
    daily[ok] = np.log(cl[ok] / op[ok]) * tail_amt[ok] / day_amt[ok]
    return _finish(ctx, _mean5(ctx, daily))


_FUNCS = {
    "turnover_weighted_intraday_return_path": _turnover_weighted_intraday_return_path,
    "range_benchmarked_semivariance_excess": _range_benchmarked_semivariance_excess,
    "tail30_signed_flow_pressure": _tail30_signed_flow_pressure,
}


def build(parameters):
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError(f"signal must be one of {SIGNALS}")
    return _FUNCS[parameters["signal"]]
