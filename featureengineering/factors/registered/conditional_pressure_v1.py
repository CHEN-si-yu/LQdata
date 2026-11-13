"""Fixed label-free cross-sectional projections of embedded pressure sources."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

FLOW = "stock_main_fund_flow"
TIERS = ("sm", "md", "lg", "elg")
SOURCE_IDS = {"range_excess", "tail_pressure", "buy_sell_impact"}


def _f(ctx, field):
    return np.asarray(ctx.intraday_field(field), dtype=np.float64)


def _fixed_window_mean(values, width, min_count=None):
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


def _finish(ctx, value):
    n_bars = _f(ctx, "n_bars")
    valid = (np.asarray(ctx.universe, dtype=bool) &
             np.asarray(ctx.traded(), dtype=bool) & np.isfinite(value) & (n_bars == 48))
    return np.where(valid, value, np.nan)


def _range_excess(ctx):
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


def _tail_pressure(ctx):
    op = np.asarray(ctx.closing30_field("open_tail30"), dtype=np.float64)
    cl = np.asarray(ctx.closing30_field("close_tail30"), dtype=np.float64)
    tail_amt = np.asarray(ctx.closing30_field("amt_tail30"), dtype=np.float64)
    day_amt = np.asarray(ctx.closing30_field("amt_day"), dtype=np.float64)
    ok = (np.isfinite(op) & (op > 0) & np.isfinite(cl) & (cl > 0) &
          np.isfinite(tail_amt) & (tail_amt >= 0) & np.isfinite(day_amt) & (day_amt > 0))
    daily = np.full(op.shape, np.nan)
    daily[ok] = np.log(cl[ok] / op[ok]) * tail_amt[ok] / day_amt[ok]
    return _finish(ctx, _mean5(ctx, daily))


def _flow_tiers(ctx):
    d0, d1 = int(ctx.panel.dates[0]) // 10000, int(ctx.panel.dates[-1]) // 10000
    cols = ["stock_code", "trade_date"]
    for tier in TIERS:
        cols.extend((f"buy_{tier}_amount", f"sell_{tier}_amount"))
    df = ctx.dataset(FLOW, columns=cols, years=(d0, d1))
    if df is None or df.empty:
        nan = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        return {tier: (nan.copy(), nan.copy()) for tier in TIERS}
    if df.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("stock_main_fund_flow contains duplicate (stock_code, trade_date) keys")
    codes = df["stock_code"].to_numpy()
    days = ctx.date_col(df["trade_date"])
    traded = np.asarray(ctx.traded(), dtype=bool)
    hit = ctx.event_grid(codes, days, np.ones(len(df), dtype=np.float64))
    out = {}
    for tier in TIERS:
        buy = df[f"buy_{tier}_amount"].to_numpy(np.float64)
        sell = df[f"sell_{tier}_amount"].to_numpy(np.float64)
        gross = buy + sell
        net = buy - sell

        def place(values):
            grid = ctx.safe_div(ctx.event_grid(codes, days, values), hit, min_abs_den=0.5)
            return np.where(traded, grid, np.nan)

        out[tier] = (place(net), place(gross))
    return out


def _conditional_mean(ctx, y, mask, n=60, min_obs=10):
    ok = np.asarray(mask, dtype=bool) & np.isfinite(y)
    count = ctx.roll_sum(ok.astype(np.float64), n, n)
    total = ctx.roll_sum(np.where(ok, y, 0.0), n, n)
    mean = np.where(count >= min_obs, ctx.safe_div(total, count, min_abs_den=0.0), np.nan)
    return mean, count


def _buy_sell_impact(ctx):
    legs = _flow_tiers(ctx)
    # The source definition uses the large tier only; extra-large is excluded.
    large_net, large_gross = legs["lg"]
    big = ctx.safe_div(large_net, large_gross, min_abs_den=0.0)
    traded = np.asarray(ctx.traded(), dtype=bool)
    prior_traded = ctx.shift(traded.astype(np.float64), 1) == 1.0
    ret = np.asarray(ctx.ret(1), dtype=np.float64)
    ret = np.where(traded & prior_traded, ret, np.nan)
    turn = np.asarray(ctx.px("turnover"), dtype=np.float64)
    impact = ctx.safe_div(np.abs(ret), turn, min_abs_den=1e-8)
    buy = big > 0
    sell = big < 0
    buy_cost, buy_n = _conditional_mean(ctx, impact, buy & np.isfinite(impact), 60, 10)
    sell_cost, sell_n = _conditional_mean(ctx, impact, sell & np.isfinite(impact), 60, 10)
    return np.where((buy_n >= 10) & (sell_n >= 10), sell_cost - buy_cost, np.nan)


_SOURCES = {
    "range_excess": _range_excess,
    "tail_pressure": _tail_pressure,
    "buy_sell_impact": _buy_sell_impact,
}


def _source_id(parameters):
    if "source" in parameters:
        source = parameters["source"]
    else:
        # Compatibility with the previous exploratory candidate dictionaries.
        source_params = parameters.get("source_parameters", {})
        implementation = parameters.get("source_implementation", "")
        if source_params.get("signal") == "range_benchmarked_semivariance_excess":
            source = "range_excess"
        elif source_params.get("signal") == "tail30_signed_flow_pressure":
            source = "tail_pressure"
        elif source_params.get("variant") == "lf_sell_buy_impact_asym_60":
            source = "buy_sell_impact"
        else:
            raise ValueError(f"unsupported source parameters for implementation {implementation!r}")
    if source not in SOURCE_IDS:
        raise ValueError(f"source must be one of {sorted(SOURCE_IDS)}")
    return source


def _rank(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.isfinite(x).sum(axis=1, keepdims=True)
    return (rankdata(x, axis=1, method="average", nan_policy="omit") - (n + 1) / 2) / np.maximum(n, 1)


def build(parameters):
    source_id = _source_id(parameters)
    controls = tuple(parameters["controls"])
    if not controls:
        raise ValueError("controls must be a nonempty fixed sequence")
    source_sign = int(parameters["source_sign"])
    if source_sign not in (-1, 1):
        raise ValueError("source_sign must be -1 or +1")
    source = _SOURCES[source_id]

    def compute(ctx):
        value = np.asarray(source(ctx), dtype=np.float64)
        x = _rank(value) * source_sign
        c = [_rank(ctx.load_factor(name)) for name in controls]
        output = np.full(x.shape, np.nan, dtype=np.float64)
        for t in range(ctx.panel.T):
            good = np.isfinite(x[t]) & np.asarray(ctx.universe[t], dtype=bool)
            if good.sum() < 100:
                continue
            Z = np.column_stack([a[t, good] for a in c])
            finite = np.isfinite(Z)
            counts = finite.sum(axis=0)
            if np.any(counts < 100):
                continue
            means = np.nansum(Z, axis=0) / counts
            Z = np.where(finite, Z - means, 0.0)
            scales = np.sqrt((Z * Z).sum(axis=0) / counts)
            identified = scales > 1e-10
            Z = Z[:, identified] / scales[identified]
            flags = finite[:, (counts > 0) & (counts < len(finite))].astype(np.float64)
            if flags.shape[1]:
                flags -= flags.mean(axis=0)
                flags /= np.maximum(flags.std(axis=0), 1e-12)
                Z = np.column_stack((Z, flags))
            C = np.column_stack((np.ones(good.sum()), Z))
            if good.sum() < max(100, 3 * C.shape[1]):
                continue
            beta, _, _, _ = np.linalg.lstsq(C, x[t, good], rcond=1e-8)
            residual = x[t, good] - C @ beta
            if np.var(residual) < 1e-12:
                continue
            output[t, good] = residual
        return output

    return compute
