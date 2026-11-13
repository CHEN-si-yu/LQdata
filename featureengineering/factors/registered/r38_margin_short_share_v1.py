"""R38 margin short-balance share: 20-session mean of rqye / rzrqye, one-session lag."""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "mg_short_balance_share_20"
WINDOW = 20
MIN_VALID = 15


def build(parameters):
    if parameters:
        raise ValueError(f"{NAME} has no tunable parameters")

    def compute(ctx):
        end_year = int(ctx.panel.dates[-1]) // 10000
        frame = ctx.dataset("stock_margin_detail", columns=[
            "trade_date", "stock_code", "rqye", "rzrqye",
        ], years=(2010, end_year))
        if frame is None or frame.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        frame = frame.copy()
        frame["stock_code"] = frame["stock_code"].astype(str)
        dates_i = np.asarray(ctx.panel.dates, dtype=np.int64)
        codes = np.asarray(ctx.panel.codes).astype(str)
        cpos = {c: i for i, c in enumerate(codes)}
        frame["code_idx"] = frame["stock_code"].map(cpos)
        frame = frame[frame["code_idx"].notna()]
        frame["day_i"] = np.asarray(ctx.date_col(frame["trade_date"]), dtype=np.int64)
        frame = frame[(frame["day_i"] >= dates_i[0]) & (frame["day_i"] <= dates_i[-1])]
        if frame.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        frame["code_idx"] = frame["code_idx"].astype(np.int64)
        frame["rqye"] = pd.to_numeric(frame["rqye"], errors="coerce")
        frame["rzrqye"] = pd.to_numeric(frame["rzrqye"], errors="coerce")
        g = frame.groupby(["code_idx", "day_i"], as_index=False)[["rqye", "rzrqye"]].sum()
        # Must use the full source coverage to know which sessions were actually
        # reported; a missing whole-source day makes every stock's value NaN.
        source_days = np.unique(frame["day_i"].to_numpy(np.int64))
        dpos = {int(d): i for i, d in enumerate(dates_i)}
        raw = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        for row in g.itertuples(index=False):
            ti = dpos.get(int(row.day_i))
            if ti is None:
                continue
            if not (np.isfinite(row.rqye) and np.isfinite(row.rzrqye)):
                continue
            if row.rzrqye > 0 and row.rqye >= 0:
                raw[ti, int(row.code_idx)] = row.rqye / row.rzrqye
        complete = np.isin(dates_i, source_days)
        raw[~complete, :] = np.nan
        lagged = ctx.shift(raw, 1)
        return ctx.roll_mean(lagged, WINDOW, min_count=MIN_VALID)

    return compute
