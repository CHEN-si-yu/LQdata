"""R38 market state wave 2: small-cap turnover-share rotation.

``mkt_small_large_index_amount_share_shift_5_20_v1``
    At each session d, the CSI1000 (000852.SH) share of CSI300+CSI1000
    turnover amount is s_d.  The factor is mean(s, last 5 sessions) minus
    mean(s, last 20 sessions); all 20 daily values must be finite positive.
    A broadening small-cap liquidity share is expected to persist into
    positive equal-weight market h5 returns.

Only daily post-close index data are used.  The factor returns a market
``(T,)`` series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "mkt_small_large_index_amount_share_shift_5_20_v1"
LARGE = "000300.SH"
SMALL = "000852.SH"
WINDOW = 20
SHORT = 5


def build(parameters):
    if parameters:
        raise ValueError(f"{NAME} takes no parameters")

    def compute(ctx):
        dates = np.asarray(ctx.panel.dates, dtype=np.int64)
        years = (int(dates[0]) // 10000, int(dates[-1]) // 10000)
        frame = ctx.dataset(
            "index_daily",
            columns=["ts_code", "trade_date", "amount"],
            years=years,
        )
        if frame is None or frame.empty:
            return np.full(len(dates), np.nan, dtype=np.float64)
        frame = frame.copy()
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["trade_date"] = np.asarray(ctx.date_col(frame["trade_date"]), dtype=np.int64)
        frame = frame[frame["ts_code"].isin((LARGE, SMALL))]
        if frame.empty:
            return np.full(len(dates), np.nan, dtype=np.float64)
        frame["amount"] = np.asarray(
            pd.to_numeric(frame["amount"], errors="coerce"), dtype=np.float64
        )
        frame = frame[np.isfinite(frame["amount"]) & (frame["amount"] > 0.0)]

        # Unique index-day amounts; if the vendor repeats a day, the last row wins.
        pivot = frame.drop_duplicates(["ts_code", "trade_date"], keep="last").pivot(
            index="trade_date", columns="ts_code", values="amount"
        )
        pivot = pivot.reindex(dates)
        if LARGE not in pivot.columns or SMALL not in pivot.columns:
            return np.full(len(dates), np.nan, dtype=np.float64)
        large = pivot[LARGE].to_numpy(np.float64)
        small = pivot[SMALL].to_numpy(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(
                np.isfinite(large) & np.isfinite(small) & (large > 0.0) & (small > 0.0),
                small / (large + small),
                np.nan,
            )

        out = np.full(len(dates), np.nan, dtype=np.float64)
        for i in range(WINDOW - 1, len(dates)):
            w = share[i - WINDOW + 1:i + 1]
            if np.isfinite(w).all():
                out[i] = float(np.mean(w[-SHORT:]) - np.mean(w))
        return out

    return compute
