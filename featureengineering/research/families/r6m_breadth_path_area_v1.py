"""Round6 strict market signed PM breadth area, lagged one complete trading day.
Supplier all-market counts, not the fixed stock pool. Weak negative prior.
Lag1 is a conservative arrival assumption, not proved historical vendor availability.
Single definition: no variants, filling, smoothing or direction selection.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SIGNAL = 'full_pm_breadth_path_area_lag1'
SOURCE = 'stock_market_distribution_history'
COLUMNS = ['trade_time', 'up_count', 'down_count', 'flat_count']
PM_TIMES = tuple(f'{m//60:02d}:{m%60:02d}:00' for m in range(13*60+1, 15*60+1))
REQUIRED_TIMES = ('11:30:00',) + PM_TIMES
U = np.arange(1, 121, dtype=np.float64) / 120.0


def signed_area(counts):
    """Input rows: exact 11:30 anchor then 120 ordered PM minutes; columns up/down/flat."""
    x = np.asarray(counts, dtype=np.float64)
    if x.shape != (121, 3):
        return np.nan
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x != np.floor(x)):
        return np.nan
    total = x.sum(axis=1)
    if not np.isfinite(total).all() or np.any(total <= 0) or np.any(total != total[0]):
        return np.nan
    breadth = x[:, 0] / total
    chord = (1.0 - U) * breadth[0] + U * breadth[-1]
    return float(np.mean(breadth[1:] - chord))


def area_path(frame, panel_dates):
    """Return (lag1 output, raw daily area) on the exact supplied trading-date axis.

    Only required minutes from dates on this axis affect values. Duplicate required
    date/time keys invalidate that day; missing days are not skipped when lagging.
    Raw future rows cannot affect earlier output values. This helper computes no returns.
    """
    dates = np.asarray(panel_dates, dtype=np.int64)
    if dates.ndim != 1 or (dates.size > 1 and np.any(dates[1:] <= dates[:-1])):
        raise ValueError('R6M requires unique ascending trading dates')
    parsed_dates = pd.to_datetime(dates.astype(str), format='%Y%m%d', errors='coerce')
    if parsed_dates.isna().any():
        raise ValueError('R6M invalid trading date axis')
    raw = np.full(len(dates), np.nan, dtype=np.float64)
    if frame is None or frame.empty or len(dates) == 0:
        return raw.copy(), raw
    if not set(COLUMNS).issubset(frame.columns):
        raise ValueError('R6M missing source fields')
    text = frame['trade_time'].astype('string')
    source_dates = pd.to_datetime(text.str.slice(0, 10), format='%Y-%m-%d', errors='coerce')
    if source_dates.isna().any():
        raise ValueError('R6M source has invalid date key')
    day = (source_dates.dt.year * 10000 + source_dates.dt.month * 100 + source_dates.dt.day).to_numpy(np.int64)
    times = text.str.slice(11, 19)
    exact = text.str.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:00', na=False).to_numpy(bool)
    keep = np.isin(day, dates) & times.isin(REQUIRED_TIMES).to_numpy(bool) & exact
    selected = frame.loc[keep, COLUMNS[1:]].copy()
    selected['day'] = day[keep]
    selected['minute'] = times.loc[keep].to_numpy()
    for d, group in selected.groupby('day', sort=False):
        if len(group) != 121 or group['minute'].duplicated().any():
            continue
        if set(group['minute']) != set(REQUIRED_TIMES):
            continue
        numeric = group.set_index('minute').reindex(REQUIRED_TIMES)[COLUMNS[1:]].apply(pd.to_numeric, errors='coerce')
        raw[int(np.searchsorted(dates, d))] = signed_area(numeric.to_numpy(np.float64))
    out = np.full_like(raw, np.nan)
    out[1:] = raw[:-1]
    return out, raw


def build(parameters):
    if parameters != {'signal': SIGNAL}:
        raise ValueError('R6M has exactly one frozen signed-area definition')
    def compute(ctx):
        dates = np.asarray(ctx.panel.dates, dtype=np.int64)
        if dates.size == 0:
            return np.empty(0, dtype=np.float64)
        frame = ctx.dataset(SOURCE, columns=COLUMNS, years=(int(dates[0])//10000, int(dates[-1])//10000))
        out, _ = area_path(frame, dates)
        return out
    return compute
