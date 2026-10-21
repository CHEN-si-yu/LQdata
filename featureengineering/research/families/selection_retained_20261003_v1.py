"""保留迁移（清理复审，2026-10-03）：仅 upside_impact_fragility 与 buy_sell_concentration_gap。

来源：research/families/selection_20261001_v1.py（FILE-SHA 封存，不改动原件）。
保留实现的公共辅助按闭包逐字复制；被拒信号的判定分支删去。参数与方向不变；
新身份须重新走原生 G1 与旧生产值精确 parity 后才批准。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea import mathx as mx


SIGNALS = ("upside_impact_fragility", "buy_sell_concentration_gap")
KEPT_SIGNALS = SIGNALS

TIERS = ("sm", "md", "lg", "elg")


LOOKBACK, MIN_HISTORY = 60, 48


def _divide(a, b, floor=1e-12):
    with np.errstate(all="ignore"):
        result = np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64)
    return np.where(np.isfinite(result) & (np.abs(b) > floor), result, np.nan)


def _past_mean(x, n=LOOKBACK, minimum=MIN_HISTORY):
    return mx.shift(mx.roll_mean(x, n, min_count=minimum), 1)


def _past_std(x, n=LOOKBACK, minimum=MIN_HISTORY):
    return mx.shift(mx.roll_std(x, n, min_count=minimum), 1)


def _standardize(x, mean, sigma, floor=1e-6):
    return np.clip(_divide(x - mean, np.maximum(sigma, floor), floor=0), -8., 8.)


def _innovation(x, n=LOOKBACK, minimum=MIN_HISTORY):
    return _standardize(x, _past_mean(x, n, minimum), _past_std(x, n, minimum))


def _prediction(y, x, second=None):
    """Past-only per-stock OLS (small fixed ridge), then predict current x.

    All fitting moments end at t-1. Complete paired historical observations
    are required. Constant x falls back to an intercept, not invented data.
    """
    valid = np.isfinite(y) & np.isfinite(x)
    if second is not None:
        valid &= np.isfinite(second)
    y, x = np.where(valid, y, np.nan), np.where(valid, x, np.nan)
    my, m1 = _past_mean(y), _past_mean(x)
    v1 = np.maximum(_past_mean(x * x) - m1 * m1, 0.)
    vy = np.maximum(_past_mean(y * y) - my * my, 0.)
    c1 = _past_mean(x * y) - m1 * my
    if second is None:
        beta1 = _divide(c1, v1 * (1. + 1e-6) + 1e-12, floor=0)
        prediction = my + beta1 * (x - m1)
        residual_var = vy - beta1 * c1
    else:
        z = np.where(valid, second, np.nan)
        m2 = _past_mean(z)
        v2 = np.maximum(_past_mean(z * z) - m2 * m2, 0.)
        c2 = _past_mean(z * y) - m2 * my
        cross = _past_mean(x * z) - m1 * m2
        ridge = 1e-6 * (v1 + v2) + 1e-12
        a, b = v1 + ridge, v2 + ridge
        determinant = a * b - cross * cross
        beta1 = _divide(c1 * b - c2 * cross, determinant, floor=0)
        beta2 = _divide(c2 * a - c1 * cross, determinant, floor=0)
        prediction = my + beta1 * (x - m1) + beta2 * (z - m2)
        residual_var = vy - beta1 * c1 - beta2 * c2
    sigma = np.sqrt(np.maximum(residual_var, 0.))
    return prediction, sigma


def _pool_relative(ctx, values):
    """Demean using the current fixed pool, never future label eligibility."""
    valid = np.asarray(ctx.universe, dtype=bool) & np.isfinite(values)
    count = valid.sum(axis=1)
    minimum = max(3, int(ctx.cfg.min_cross_section))
    mean = _divide(np.where(valid, values, 0.).sum(axis=1), count, floor=0)
    mean[count < minimum] = np.nan
    return np.where(valid, values - mean[:, None], np.nan)


def _prices(ctx):
    traded = np.asarray(ctx.traded(), dtype=bool)
    raw = {field: np.asarray(ctx.px_raw(field), dtype=np.float64)
           for field in ("open", "high", "low", "close", "amount", "vol")}
    valid = traded.copy()
    for value in raw.values():
        valid &= np.isfinite(value) & (value > 0)
    valid &= (raw["high"] >= np.maximum(raw["open"], raw["close"]))
    valid &= (raw["low"] <= np.minimum(raw["open"], raw["close"]))
    valid &= raw["high"] >= raw["low"]
    with np.errstate(all="ignore"):
        log_amount = np.where(valid, np.log(raw["amount"]), np.nan)
        day_range = np.where(valid, np.log(raw["high"] / raw["low"]), np.nan)
        intraday = np.where(valid, np.log(raw["close"] / raw["open"]), np.nan)
    clv = _divide(2. * raw["close"] - raw["high"] - raw["low"], raw["high"] - raw["low"], floor=0)
    clv = np.where(valid & (raw["high"] == raw["low"]), 0., clv)
    clv = np.where(valid, np.clip(clv, -1., 1.), np.nan)
    # PriceLayer owns HFQ and return clipping. Candidate validity additionally
    # rejects a positive stale price on an invalid/nontrading source day before
    # it can enter either a lag or the next day's return control.
    close = np.where(valid, np.asarray(ctx.hfq("close"), dtype=np.float64), np.nan)
    opening = np.where(valid, np.asarray(ctx.hfq("open"), dtype=np.float64), np.nan)
    prior_close = mx.shift(close, 1)
    returns = np.where(valid & np.isfinite(prior_close), np.asarray(ctx.ret_clean(1), dtype=np.float64), np.nan)
    relative = _pool_relative(ctx, returns)
    sigma = _past_std(relative)
    with np.errstate(all="ignore"):
        gap = np.where(valid, np.log(opening / prior_close), np.nan)
    return {**raw, "amount": np.where(valid, raw["amount"], np.nan),
            "price_valid": valid, "valid": valid & np.isfinite(returns), "relative": relative,
            "sigma": sigma, "log_amount": log_amount, "range": day_range,
            "intraday": _pool_relative(ctx, intraday), "clv": clv,
            "hfq_close": close, "gap": gap}


def _impact(p, side):
    relative_amount = _divide(p["amount"], _past_mean(p["amount"]))
    with np.errstate(all="ignore"):
        impact = np.log(np.abs(p["relative"]) + 1e-6) - np.log(relative_amount)
    active = p["relative"] < 0 if side == "down" else p["relative"] > 0
    history = np.where(active, impact, np.nan)
    surprise = _standardize(impact, _past_mean(history, 60, 12), _past_std(history, 60, 12))
    complete = np.isfinite(_past_mean(p["relative"])) & np.isfinite(relative_amount)
    return np.where(complete, np.where(active, surprise, 0.), np.nan)


def _flows(ctx, need_vol=False):
    columns = ["stock_code", "trade_date"] + [f"{side}_{tier}_amount" for side in ("buy", "sell") for tier in TIERS]
    if need_vol:
        columns += [f"{side}_{tier}_vol" for side in ("buy", "sell") for tier in ("lg", "elg")]
    days = ctx.panel.dates
    frame = ctx.dataset("stock_main_fund_flow", columns=columns,
                        years=(int(days[0]) // 10000, int(days[-1]) // 10000))
    if frame.empty:
        return {"valid": np.zeros(ctx.panel.shape, dtype=bool),
                "buy": np.full((*ctx.panel.shape, 4), np.nan),
                "sell": np.full((*ctx.panel.shape, 4), np.nan),
                "rates": np.full((*ctx.panel.shape, 4), np.nan),
                "main": ctx.panel.empty(dtype=np.float64)}
    frame = frame[frame.stock_code.isin(ctx.panel.codes)].copy()
    if frame.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate stock/day in four-bucket flow source")
    positions, dates = ctx.code_index(frame.stock_code.to_numpy()), ctx.date_col(frame.trade_date)
    def grid(column):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        values = np.where(np.isfinite(values) & (values >= 0), values, np.nan)
        return ctx.panel.place(positions, dates, values)
    buy = np.stack([grid(f"buy_{tier}_amount") for tier in TIERS], axis=-1)
    sell = np.stack([grid(f"sell_{tier}_amount") for tier in TIERS], axis=-1)
    total = np.sum(buy + sell, axis=-1)
    rates = _divide(buy - sell, total[..., None])
    valid = np.asarray(ctx.traded(), dtype=bool) & np.isfinite(buy).all(axis=-1) & np.isfinite(sell).all(axis=-1) & (total > 0)
    # Final output masking is too late for innovations and fitting histories.
    # A nontrading/stale source row must never be a historical zero or shock.
    rates = np.where(valid[..., None], rates, np.nan)
    result = {"buy": buy, "sell": sell, "rates": rates, "main": rates[..., 2] + rates[..., 3], "valid": valid}
    if need_vol:
        result["buy_big_vol"] = grid("buy_lg_vol") + grid("buy_elg_vol")
        result["sell_big_vol"] = grid("sell_lg_vol") + grid("sell_elg_vol")
    return result


def build(parameters):
    """保留迁移：仅 upside_impact_fragility 与 buy_sell_concentration_gap（参数与方向不变）。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in KEPT_SIGNALS:
        raise ValueError("retained family permits only the preserved preregistered signals")
    signal = parameters["signal"]

    def compute(ctx):
        p = _prices(ctx) if signal == "upside_impact_fragility" else None
        if signal == "upside_impact_fragility":
            value = _impact(p, "up")
        else:  # buy_sell_concentration_gap
            f = _flows(ctx, need_vol=False)
            bshare = _divide(f["buy"], f["buy"].sum(axis=-1)[..., None])
            sshare = _divide(f["sell"], f["sell"].sum(axis=-1)[..., None])
            value = np.sum(bshare ** 2 - sshare ** 2, axis=-1)
            value = np.where(f["valid"], value, np.nan)
        if p is not None:
            value = np.where(p["valid"], value, np.nan)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
