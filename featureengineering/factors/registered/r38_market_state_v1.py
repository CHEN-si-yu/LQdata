"""R38 market state: 20-day persistence of Eastmoney L1 industry return ranks."""
from __future__ import annotations

import numpy as np
import pandas as pd

INDUSTRIES = (
    "BK0427.DC", "BK0433.DC", "BK0436.DC", "BK0437.DC", "BK0438.DC",
    "BK0456.DC", "BK0464.DC", "BK0478.DC", "BK0479.DC", "BK0486.DC",
    "BK0728.DC", "BK1200.DC", "BK1201.DC", "BK1202.DC", "BK1203.DC",
    "BK1204.DC", "BK1205.DC", "BK1206.DC", "BK1207.DC", "BK1208.DC",
    "BK1209.DC", "BK1210.DC", "BK1211.DC", "BK1212.DC", "BK1213.DC",
    "BK1214.DC", "BK1215.DC", "BK1216.DC", "BK1217.DC", "BK1283.DC",
)
WINDOW = 20
MIN_VALID = 15
MIN_PAIRS = 20


def _rank_corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < MIN_PAIRS:
        return np.nan
    x = a[m]; y = b[m]
    if np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return np.nan
    from scipy.stats import rankdata
    xr = rankdata(x); yr = rankdata(y)
    xr = xr - xr.mean(); yr = yr - yr.mean()
    den = np.sqrt((xr @ xr) * (yr @ yr))
    return float(xr @ yr / den) if den > 0 else np.nan


def build(parameters):
    if parameters:
        raise ValueError("market state factor takes no parameters")

    def compute(ctx):
        dates = np.asarray(ctx.panel.dates, dtype=np.int32)
        end_year = int(dates[-1]) // 10000
        start_year = max(2017, int(dates[0]) // 10000 - 1)
        frames = []
        for y in range(start_year, end_year + 1):
            p = ctx.up.root / "dc_daily" / f"year={y}" / "data.parquet"
            if not p.exists():
                continue
            fr = ctx.dataset("dc_daily", columns=["block_code", "trade_date", "close"],
                             years=(y, y))
            if fr is not None and not fr.empty:
                frames.append(fr)
        if not frames:
            return np.full(len(dates), np.nan)
        fr = pd.concat(frames, ignore_index=True)
        fr = fr[fr["block_code"].astype(str).isin(INDUSTRIES)].copy()
        if fr.empty:
            return np.full(len(dates), np.nan)
        fr["trade_date"] = pd.to_datetime(fr["trade_date"]).dt.strftime("%Y%m%d").astype(np.int32)
        close = fr.pivot(index="trade_date", columns="block_code", values="close")
        close = close.reindex(dates).astype(np.float64)
        ret = close / close.shift(1) - 1.0
        corr = np.full(len(dates), np.nan)
        for i in range(1, len(dates)):
            corr[i] = _rank_corr(ret.iloc[i].to_numpy(), ret.iloc[i - 1].to_numpy())
        out = np.full(len(dates), np.nan)
        for i in range(len(dates)):
            w = corr[max(0, i - WINDOW + 1):i + 1]
            n = np.isfinite(w).sum()
            if n >= MIN_VALID:
                out[i] = np.nanmean(w)
        return out

    return compute
