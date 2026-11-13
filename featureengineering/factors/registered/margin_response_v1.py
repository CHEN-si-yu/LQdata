"""Lagged margin demand response."""
from __future__ import annotations

import numpy as np

MARGIN = "stock_margin_detail"


def _unique(df):
    if df.duplicated(["stock_code", "trade_date"]).any():
        dup = df.loc[df.duplicated(["stock_code", "trade_date"], keep=False),
                     ["stock_code", "trade_date"]]
        raise ValueError(f"{MARGIN} has duplicate keys: {dup.head(5).to_dict('records')}")


def _exact_grid(ctx, df, col):
    """Scatter one observed value per source key; absent source rows stay NaN."""
    codes = df["stock_code"].to_numpy()
    days = ctx.date_col(df["trade_date"])
    values = df[col].to_numpy(dtype=np.float64)
    present = ctx.event_grid(codes, days, np.ones(values.size, dtype=np.float64))
    total = ctx.event_grid(codes, days, values)
    # Duplicate keys are rejected before event_grid. This preserves real zero
    # flow while leaving absent keys and null source values missing.
    return ctx.safe_div(total, present, min_abs_den=0.5)


def _pairwise_corr60(ctx, x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    a = np.where(valid, x, np.nan)
    b = np.where(valid, y, np.nan)
    ax = ctx.roll_mean(a, 60, 45)
    by = ctx.roll_mean(b, 60, 45)
    cov = ctx.roll_mean(a * b, 60, 45) - ax * by
    vx = ctx.roll_mean(a * a, 60, 45) - ax * ax
    vy = ctx.roll_mean(b * b, 60, 45) - by * by
    # Independently reject zero/negative variances. Product-only checks could
    # accept two negative round-off artifacts and emit a spurious correlation.
    min_var = 1e-24  # equivalent to a 1e-12 standard-deviation floor per leg
    good = (np.isfinite(cov) & np.isfinite(vx) & np.isfinite(vy) &
            (vx > min_var) & (vy > min_var))
    denom = np.sqrt(np.where(good, vx, 0.0) * np.where(good, vy, 0.0))
    out = np.full(x.shape, np.nan, dtype=np.float64)
    np.divide(cov, denom, out=out, where=good & (denom > 0.0))
    return out


def build(parameters):
    if parameters.get("variant") != "lf_margin_dip_demand_corr_60":
        raise ValueError(f"unknown margin variant: {parameters.get('variant')}")

    def compute(ctx):
        d0 = int(ctx.panel.dates[0]) // 10000
        d1 = int(ctx.panel.dates[-1]) // 10000
        df = ctx.dataset(MARGIN, columns=["stock_code", "trade_date", "rzmre", "rzye"],
                         years=(d0, d1))
        if df is None or df.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        _unique(df)
        source_flow = _exact_grid(ctx, df, "rzmre")
        source_balance = _exact_grid(ctx, df, "rzye")

        # The denominator must be an actually observed positive balance on the
        # preceding exchange session; no forward fill or lower-bound rescue.
        prior_balance = ctx.shift(source_balance, 1)
        buy_rate = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        valid_balance = (np.isfinite(source_flow) & (source_flow >= 0) &
                         np.isfinite(prior_balance) & (prior_balance > 0))
        np.divide(source_flow, prior_balance, out=buy_rate, where=valid_balance)

        traded = np.asarray(ctx.traded(), dtype=bool)
        prior_traded = ctx.shift(traded.astype(np.float64), 1) == 1.0
        ret = np.asarray(ctx.ret(1), dtype=np.float64)
        ret = np.where(traded & prior_traded & np.isfinite(ret), ret, np.nan)

        # Source-date correlation; stock_margin_detail is released one session
        # later, so shift the final statistic only after all source calculations.
        corr = _pairwise_corr60(ctx, ret, buy_rate)
        signal = -corr
        signal = ctx.lag_grid(signal, 1)
        final_mask = np.asarray(ctx.universe, dtype=bool) & np.asarray(ctx.traded(), dtype=bool)
        return np.where(final_mask, signal, np.nan)

    return compute
