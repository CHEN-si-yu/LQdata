"""R8 B: local Amihud shock conditional-beta proxy, frozen weak negative prior.

Not the Pastor/Stambaugh monthly liquidity factor. Primary runtime is independent
of every other research module. Only realized same-day raw price/volume sources.
Historical innovations retain their own one-sided AR1, never a later AR fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fea.universe import frozen_codes

PARAMETERS = {"signal": "local_amihud_shock_beta", "ar_window": 126,
              "beta_window": 126, "min_pairs": 100}
AR_WINDOW = BETA_WINDOW = 126
MIN_PAIRS = MARKET_MIN = 100
RET_MAX = 0.60
AR_VAR_FLOOR = U_VAR_FLOOR = 1e-12
M_VAR_FLOOR = 1e-10
CORR_DENOM_FLOOR = 1e-6
NESTED_PREFIX = 253
DAILY_COLUMNS = ["stock_code", "trade_date", "close", "vol", "amount"]
ADJ_COLUMNS = ["stock_code", "trade_date", "adj_factor"]


def _date_int(series):
    text = series.astype("string").str.slice(0, 10)
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    compact = text.str.fullmatch(r"\d{8}", na=False)
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return (parsed.dt.year * 10000 + parsed.dt.month * 100
            + parsed.dt.day).fillna(0).to_numpy(dtype=np.int64)


def _geometry(ctx):
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    calendar = np.asarray(ctx.cal.days, dtype=np.int64)
    if days.ndim != 1 or not len(days) or np.any(np.diff(days) <= 0):
        raise ValueError("panel dates must be nonempty unique increasing")
    if calendar.ndim != 1 or not len(calendar) or np.any(np.diff(calendar) <= 0):
        raise ValueError("exchange calendar must be nonempty unique increasing")
    pos = np.searchsorted(calendar, days)
    if np.any(pos >= len(calendar)) or not np.array_equal(calendar[pos], days):
        raise ValueError("panel contains non-exchange dates")
    if len(pos) > 1 and np.any(np.diff(pos) != 1):
        raise ValueError("panel must retain every consecutive exchange-date slot")
    expected = frozen_codes(ctx.cfg)
    if expected is None or len(expected) != 2115:
        raise ValueError("requires fixed2115 universe")
    if not np.array_equal(np.asarray(ctx.panel.codes).astype(str), expected):
        raise ValueError("market statistics require the full ordered frozen pool")
    universe = np.asarray(ctx.universe)
    if universe.dtype.kind != "b" or universe.shape != ctx.panel.shape:
        raise ValueError("universe must be boolean with identical panel axes")
    return days, universe


def _exact_fields(ctx, frame, fields, days):
    if frame is None or frame.empty:
        return {f: np.full(ctx.panel.shape, np.nan, dtype=np.float64) for f in fields}
    code = np.asarray(ctx.code_index(frame.stock_code.to_numpy()), dtype=np.int64)
    date = _date_int(frame.trade_date)
    keep = (code >= 0) & np.isin(date, days)
    code, date = code[keep], date[keep]
    if len(code) and len(np.unique(code * 100_000_000 + date)) != len(code):
        raise ValueError("duplicate pool(date,stock) source key")
    result = {}
    for field in fields:
        value = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)[keep]
        result[field] = ctx.panel.place(code, date, value, dtype=np.float64)
    return result


def strict_returns(close, volume, adjustment, universe):
    close, volume, adjustment = [np.asarray(a, dtype=np.float64)
                                 for a in (close, volume, adjustment)]
    universe = np.asarray(universe, dtype=bool)
    if close.ndim != 2 or any(a.shape != close.shape for a in (volume, adjustment, universe)):
        raise ValueError("source arrays must share date/stock axes")
    endpoint = (universe & np.isfinite(close) & (close > 0)
                & np.isfinite(volume) & (volume > 0)
                & np.isfinite(adjustment) & (adjustment > 0))
    result = np.full(close.shape, np.nan, dtype=np.float64)
    with np.errstate(all="ignore"):
        gross = close[1:] * adjustment[1:] / (close[:-1] * adjustment[:-1])
        r = gross - 1.0
    # Gross bounds retain exactly60% instead of subtract-one roundoff rejection.
    valid = (endpoint[1:] & endpoint[:-1] & np.isfinite(r)
             & (gross >= 1.0 - RET_MAX) & (gross <= 1.0 + RET_MAX))
    result[1:] = np.where(valid, r, np.nan)
    return result


def market_state(r, amount, min_names=100):
    r, amount = np.asarray(r, dtype=np.float64), np.asarray(amount, dtype=np.float64)
    if r.ndim != 2 or amount.shape != r.shape:
        raise ValueError("r/amount shape mismatch")
    valid = np.isfinite(r) & np.isfinite(amount) & (amount > 0)
    count = valid.sum(axis=1)
    market = np.full(len(r), np.nan)
    illiquidity = np.full(len(r), np.nan)
    for t in np.flatnonzero(count >= max(MARKET_MIN, int(min_names))):
        values = r[t, valid[t]]
        market[t] = values.mean()
        illiquidity[t] = np.quantile(np.abs(values) / amount[t, valid[t]], .5, method="linear")
    q = np.full(len(r), np.nan)
    hit = np.isfinite(illiquidity) & (illiquidity > 0)
    q[hit] = np.log(1e8 * illiquidity[hit])
    # Return the same E used for market mean/median, not a larger stock sample.
    return np.where(valid, r, np.nan), market, illiquidity, q, count


def one_sided_innovations(q):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError("q must be one time series")
    u = np.full(len(q), np.nan)
    alpha = np.full(len(q), np.nan)
    phi = np.full(len(q), np.nan)
    count = np.full(len(q), np.nan)
    for d in range(AR_WINDOW + 1, len(q)):
        # s=d-127..d-2, target last index d-1. Current q[d] is not in training.
        x = q[d - AR_WINDOW - 1:d - 1]
        y = q[d - AR_WINDOW:d]
        valid = np.isfinite(x) & np.isfinite(y)
        count[d] = valid.sum()
        if count[d] < MIN_PAIRS or not np.isfinite(q[d - 1:d + 1]).all():
            continue
        x, y = x[valid], y[valid]
        xm, ym = x.mean(), y.mean()
        xc, yc = x - xm, y - ym
        var = np.mean(xc * xc)
        if not var > AR_VAR_FLOOR:
            continue
        phi[d] = np.mean(xc * yc) / var
        alpha[d] = ym - phi[d] * xm
        value = q[d] - alpha[d] - phi[d] * q[d - 1]
        if np.isfinite(value):
            u[d] = value
    return u, alpha, phi, count


def conditional_betas(r, market, u):
    r = np.asarray(r, dtype=np.float64)
    market, u = np.asarray(market, dtype=np.float64), np.asarray(u, dtype=np.float64)
    if r.ndim != 2 or market.shape != (len(r),) or u.shape != market.shape:
        raise ValueError("r/market/u axes mismatch")
    theta, b, nobs, u_residual_var = [np.full(r.shape, np.nan) for _ in range(4)]
    for t in range(BETA_WINDOW - 1, len(r)):
        sl = slice(t - BETA_WINDOW + 1, t + 1)
        x, z, y = market[sl, None], u[sl, None], r[sl]
        valid = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
        count = valid.sum(axis=0)
        nobs[t] = count
        current = np.isfinite(r[t]) & np.isfinite(market[t]) & np.isfinite(u[t])
        cols = np.flatnonzero((count >= MIN_PAIRS) & current)
        if not len(cols):
            continue
        valid, y = valid[:, cols], y[:, cols]
        den = count[cols].astype(np.float64)
        mx = np.where(valid, x, 0.0).sum(axis=0) / den
        mz = np.where(valid, z, 0.0).sum(axis=0) / den
        my = np.where(valid, y, 0.0).sum(axis=0) / den
        xc, zc, yc = [np.where(valid, v - mu, 0.0)
                      for v, mu in ((x, mx), (z, mz), (y, my))]
        vx, vz = (xc * xc).sum(axis=0) / den, (zc * zc).sum(axis=0) / den
        with np.errstate(all="ignore"):
            sx, sz = np.sqrt(vx), np.sqrt(vz)
            rho = (xc * zc).sum(axis=0) / den / sx / sz
            diff = 1.0 - rho * rho
            xy = (xc * yc).sum(axis=0) / den / sx
            zy = (zc * yc).sum(axis=0) / den / sz
            bt = (xy - rho * zy) / diff / sx
            th = (zy - rho * xy) / diff / sz
        good = ((vx > M_VAR_FLOOR) & (vz > U_VAR_FLOOR)
                & (diff > CORR_DENOM_FLOOR) & np.isfinite(bt) & np.isfinite(th))
        ids = cols[good]
        theta[t, ids], b[t, ids] = th[good], bt[good]
        u_residual_var[t, ids] = vz[good] * diff[good]
    return theta, b, nobs, u_residual_var


def analyze_arrays(close, volume, amount, adjustment, universe, min_names=100):
    r = strict_returns(close, volume, adjustment, universe)
    r, market, illiquidity, q, market_count = market_state(r, amount, min_names)
    u, alpha, phi, ar_count = one_sided_innovations(q)
    raw, b, count, uvar = conditional_betas(r, market, u)
    # q[t-252] needs close[t-253]: 254 price slots including current date.
    # Structural history before those full nested price slots is unknown,
    # rather than letting min100 bootstrap an insufficient panel prefix.
    raw[:NESTED_PREFIX] = np.nan
    b[:NESTED_PREFIX] = np.nan
    uvar[:NESTED_PREFIX] = np.nan
    return raw, {"conditional_market_beta": b, "regression_nobs": count,
                 "u_residual_var": uvar, "current_return": r,
                 "market_return": market, "illiquidity_median": illiquidity,
                 "log_illiquidity": q, "innovation": u, "ar_alpha": alpha,
                 "ar_phi": phi, "ar_nobs": ar_count, "market_nobs": market_count}


def _compute(ctx):
    days, universe = _geometry(ctx)
    years = (int(days[0]) // 10000, int(days[-1]) // 10000)
    daily = ctx.dataset("stock_daily", columns=DAILY_COLUMNS, years=years)
    d = _exact_fields(ctx, daily, ["close", "vol", "amount"], days)
    del daily
    adj = ctx.dataset("stock_adj_factor", columns=ADJ_COLUMNS, years=years)
    a = _exact_fields(ctx, adj, ["adj_factor"], days)
    del adj
    return analyze_arrays(d["close"], d["vol"], d["amount"], a["adj_factor"],
                          universe, max(MARKET_MIN, int(ctx.cfg.min_cross_section)))


def control_components(ctx):
    """Diagnostics only; common time series are never stock G7 regressors."""
    return _compute(ctx)[1]


def additional_local_controls(ctx):
    """Separately requested NEW diagnostics using approved current risk kernels.
    These do not tighten primary eligibility or become candidate dependencies.
    Their hashes/numerics/window/min_count/annualization are frozen separately.
    Registered R4/R7 controls are central exact-module reproductions, not here.
    """
    from factors.liquidity import _amihud, _turnover
    from factors.volatility import _ret
    r = _ret(ctx)
    return {"amihud_mean20": ctx.roll_mean(_amihud(ctx), 20, min_count=10),
            "turnover_mean20": ctx.roll_mean(_turnover(ctx), 20, min_count=10),
            "vol20": ctx.roll_std(r, 20, min_count=10),
            "vol60": ctx.roll_std(r, 60, min_count=30)}


def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only the single frozen R8 B definition is permitted")
    def compute(ctx):
        return _compute(ctx)[0]
    return compute
