"""R7 E: context-dependent return salience; frozen weak negative prior.

Raw exact-date close/adjustment and real turnover at BOTH consecutive exchange
dates. No ffill, no elapsed-valid-observation window, no fitted parameters.
Cosemans/Frehen2021 motivates the mechanism; local1/5day alpha is unvalidated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fea.universe import frozen_codes

PARAMETERS = {"signal": "context_salience20", "window": 20, "min_valid": 15,
              "theta": 0.1, "delta": 0.7, "market_min": 100}
DAILY_COLUMNS = ["stock_code", "trade_date", "close", "vol"]
ADJ_COLUMNS = ["stock_code", "trade_date", "adj_factor"]
WINDOW = 20
MIN_VALID = 15
MARKET_MIN = 100
THETA = 0.10
DELTA = 0.70
RET_MAX = 0.60
DSV_FLOOR = 1e-12


def valid_dates(series):
    text = series.astype("string").str.slice(0, 10)
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    compact = text.str.fullmatch(r"\d{8}", na=False)
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(
            text.loc[compact], format="%Y%m%d", errors="coerce")
    return (parsed.dt.year * 10000 + parsed.dt.month * 100
            + parsed.dt.day).fillna(0).to_numpy(dtype=np.int64)


def _axis(values, name):
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim != 1 or not len(arr) or np.any(np.diff(arr) <= 0):
        raise ValueError(f"{name} must be nonempty, unique and increasing")
    pd.to_datetime(arr.astype(str), format="%Y%m%d", errors="raise")
    return arr


def _geometry(ctx):
    days = _axis(ctx.panel.dates, "panel dates")
    cal = _axis(ctx.cal.days, "exchange calendar")
    pos = np.searchsorted(cal, days)
    if np.any(pos >= len(cal)) or not np.array_equal(cal[pos], days):
        raise ValueError("panel contains a non-exchange date")
    if len(pos) > 1 and np.any(np.diff(pos) != 1):
        raise ValueError("panel must retain every consecutive exchange-date slot")
    expected = frozen_codes(ctx.cfg)
    if expected is None or len(expected) != 2115:
        raise ValueError("R7 E requires the fixed2115 universe contract")
    if not np.array_equal(np.asarray(ctx.panel.codes).astype(str), expected):
        raise ValueError("market benchmark requires the full ordered frozen pool")
    universe = np.asarray(ctx.universe)
    if universe.dtype.kind != "b" or universe.shape != ctx.panel.shape:
        raise ValueError("universe must be a boolean panel with identical axes")
    return days, universe


def _exact_fields(ctx, frame, names, days):
    if frame is None or frame.empty:
        return {name: np.full(ctx.panel.shape, np.nan, dtype=np.float64)
                for name in names}
    dates = valid_dates(frame["trade_date"])
    codes = np.asarray(ctx.code_index(frame["stock_code"].to_numpy()),
                       dtype=np.int64)
    keep = ((codes >= 0) & (dates >= days[0]) & (dates <= days[-1])
            & np.isin(dates, days))
    c, d = codes[keep], dates[keep]
    if len(c):
        pairs = c * 100_000_000 + d
        if len(np.unique(pairs)) != len(pairs):
            raise ValueError("duplicate pool(date,stock) source key")
    out = {}
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(
            dtype=np.float64)[keep]
        out[name] = ctx.panel.place(c, d, values, dtype=np.float64)
    return out


def strict_returns(close, volume, adjustment, universe):
    """Float64 one-day returns with both exact endpoints; first slot is unknown."""
    close, volume, adjustment = [np.asarray(x, dtype=np.float64)
                                 for x in (close, volume, adjustment)]
    universe = np.asarray(universe, dtype=bool)
    if close.ndim != 2 or any(x.shape != close.shape
                             for x in (volume, adjustment, universe)):
        raise ValueError("all source arrays must share date/stock shape")
    endpoint = (universe & np.isfinite(close) & (close > 0)
                & np.isfinite(volume) & (volume > 0)
                & np.isfinite(adjustment) & (adjustment > 0))
    out = np.full(close.shape, np.nan, dtype=np.float64)
    with np.errstate(all="ignore"):
        gross = (close[1:] * adjustment[1:]) / (
            close[:-1] * adjustment[:-1])
        r = gross - 1.0
    # Compare gross bounds to keep the exact60% endpoint without subtract-one
    # rounding (e.g.160/100-1 is0.6000000000000001). No epsilon or clipping.
    ok = (endpoint[1:] & endpoint[:-1] & np.isfinite(r)
          & (gross >= 1.0 - RET_MAX) & (gross <= 1.0 + RET_MAX))
    out[1:] = np.where(ok, r, np.nan)
    return out


def source_returns(ctx):
    days, universe = _geometry(ctx)
    years = (int(days[0]) // 10000, int(days[-1]) // 10000)
    daily = ctx.dataset("stock_daily", columns=DAILY_COLUMNS, years=years)
    adj = ctx.dataset("stock_adj_factor", columns=ADJ_COLUMNS, years=years)
    d = _exact_fields(ctx, daily, ["close", "vol"], days)
    a = _exact_fields(ctx, adj, ["adj_factor"], days)
    return strict_returns(d["close"], d["vol"], a["adj_factor"], universe)


def _rank_weights(sigma):
    """Rows are stocks, columns20 historical slots; NaNs never enter ties/ranks."""
    valid = np.isfinite(sigma)
    # Pairwise comparison is bounded at20x20 per stock, never full-history cubic.
    higher = (sigma[:, None, :] > sigma[:, :, None]).sum(axis=2)
    equal = (sigma[:, None, :] == sigma[:, :, None]).sum(axis=2)
    rank = 1.0 + higher + (equal - 1.0) / 2.0
    powers = np.where(valid, DELTA ** (rank - 1.0), 0.0)
    total = powers.sum(axis=1, keepdims=True)
    weights = np.divide(powers, total, out=np.zeros_like(powers), where=total > 0)
    return weights


def rolling_statistics(returns):
    """Joint factor and audit controls on exactly the same observable state set."""
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("returns must be date x stock")
    shape = r.shape
    factor = np.full(shape, np.nan, dtype=np.float64)
    controls = {key: np.full(shape, np.nan, dtype=np.float64)
                for key in ("strict_MAX20", "strict_MIN20", "DSV20")}
    valid = np.isfinite(r)
    count = valid.sum(axis=1)
    market = np.full(shape[0], np.nan, dtype=np.float64)
    np.divide(np.where(valid, r, 0.0).sum(axis=1), count, out=market,
              where=count >= MARKET_MIN)
    paired = valid & np.isfinite(market[:, None])
    with np.errstate(all="ignore"):
        sigma = np.where(paired, np.abs(r - market[:, None])
                         / (np.abs(r) + np.abs(market[:, None]) + THETA), np.nan)
    for t in range(WINDOW - 1, shape[0]):
        sl = slice(t - WINDOW + 1, t + 1)
        v = paired[sl].T
        n = v.sum(axis=1)
        cols = np.flatnonzero((n >= MIN_VALID) & paired[t])
        if not len(cols):
            continue
        v = v[cols]
        x = r[sl].T[cols]
        sig = sigma[sl].T[cols]
        w = _rank_weights(sig)
        safe = np.where(v, x, 0.0)
        mean = safe.sum(axis=1) / n[cols]
        # Centered equivalent of sum(w*r)-mean(r); avoids large-term cancellation.
        centered = np.where(v, x - mean[:, None], 0.0)
        value = (w * centered).sum(axis=1)
        same = (np.where(v, sig, np.inf).min(axis=1)
                == np.where(v, sig, -np.inf).max(axis=1))
        value[same] = 0.0  # Uniform salience is an exact economic zero, not noise.
        factor[t, cols] = value
        controls["strict_MAX20"][t, cols] = np.where(v, x, -np.inf).max(axis=1)
        controls["strict_MIN20"][t, cols] = np.where(v, x, np.inf).min(axis=1)
        denominator = (safe * safe).sum(axis=1)
        numerator = (np.minimum(safe, 0.0) ** 2).sum(axis=1)
        dsv = np.full(len(cols), np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=dsv, where=denominator > DSV_FLOOR)
        controls["DSV20"][t, cols] = dsv
    return factor, controls


def control_components(ctx):
    """Only audit controls, not extra candidates; no changed sample/return mask."""
    return rolling_statistics(source_returns(ctx))[1]


def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only the single frozen R7 E definition is permitted")

    def compute(ctx):
        return rolling_statistics(source_returns(ctx))[0]

    return compute
