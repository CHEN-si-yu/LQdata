"""R38 pledge factors: weekly pledge-stat snapshots with a conservative lag.

The upstream ``stock_pledge_stat`` table has only an ``end_date`` (weekly
statistic date) and no announcement date.  To avoid look-ahead we use the first
exchange session at least 10 calendar days after ``end_date`` as the effective
date, and carry a snapshot for at most 21 calendar days.  This is the frozen
research convention, not a production convenience.

Variants
--------
* ``pl_ratio_level_v1``          -- pledged shares / total shares (%).
* ``pl_ratio_over_freefloat_v1`` -- pledge ratio re-based to the free-float
  share count at the effective date; claims on the actually tradable supply.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = ("pl_ratio_level_v1", "pl_ratio_over_freefloat_v1")

PLEDGE_LAG_DAYS = 10
MAX_STALE_DAYS = 21


def _date_int(series):
    return (pd.to_datetime(series, errors="coerce").dt.strftime("%Y%m%d")
            .fillna("0").astype(np.int64).to_numpy())


def _place_finance(ctx, columns):
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    frame = ctx.dataset(
        "stock_finance",
        columns=["trade_date", "stock_code", *columns],
        years=(int(dates[0]) // 10000, int(dates[-1]) // 10000),
    )
    out = {}
    if frame is None or frame.empty:
        for c in columns:
            out[c] = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        return out
    idx = np.asarray(ctx.code_index(frame["stock_code"].to_numpy()), dtype=np.int64)
    day = np.asarray(ctx.date_col(frame["trade_date"]), dtype=np.int64)
    keep = idx >= 0
    for c in columns:
        vals = pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
        out[c] = ctx.panel.place(idx[keep], day[keep], vals[keep])
    return out


def _pledge_snapshots(ctx):
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    frame = ctx.dataset(
        "stock_pledge_stat",
        columns=["stock_code", "end_date", "pledge_ratio"],
        years=(int(dates[0]) // 10000, int(dates[-1]) // 10000),
    )
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame["ed"] = _date_int(frame["end_date"])
    frame = frame[(frame["ed"] > 0) & np.isfinite(frame["pledge_ratio"].to_numpy(np.float64))]
    if frame.empty:
        return None

    cal = np.asarray(ctx.cal.days, dtype=np.int64)
    target = (pd.to_datetime(frame["ed"].astype(str), format="%Y%m%d")
              + pd.Timedelta(days=PLEDGE_LAG_DAYS))
    target_i = target.dt.strftime("%Y%m%d").astype(np.int64).to_numpy()
    pos = np.searchsorted(cal, target_i, side="left")
    ok = pos < len(cal)
    E = np.zeros(len(frame), dtype=np.int64)
    E[ok] = cal[pos[ok]]
    frame["E"] = E
    frame = frame[frame["E"] > 0]
    if frame.empty:
        return None
    frame = frame.sort_values(["stock_code", "ed"], kind="stable").reset_index(drop=True)
    # Group positions must equal labels: _map_snapshots indexes the per-snapshot
    # value array by g.index (positions), not by the original parquet row label.
    frame = frame.reset_index(drop=True)
    return frame


def _map_snapshots(ctx, snap, values, mode):
    """Map per-snapshot values to the panel using latest E <= T within 21 days."""
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    codes = np.asarray(ctx.panel.codes).astype(str)
    cpos = {c: i for i, c in enumerate(codes)}
    out = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    # End dates span calendar years: integer YYYYMMDD subtraction is not an age.
    date_d = pd.to_datetime(dates.astype(str), format="%Y%m%d").to_numpy().astype("datetime64[D]")

    for code, g in snap.groupby("stock_code", sort=False):
        col = cpos.get(code)
        if col is None:
            continue
        E = g["E"].to_numpy(np.int64)
        v = vals[g.index.to_numpy()]
        idx = np.searchsorted(E, dates, side="right") - 1
        ok = idx >= 0
        age = np.full(len(dates), 10**9, dtype=np.int64)
        if ok.any():
            e_d = pd.to_datetime(E[idx[ok]].astype(str), format="%Y%m%d").to_numpy().astype("datetime64[D]")
            age[ok] = (date_d[ok] - e_d).astype(np.int64)
        valid = ok & (age >= 0) & (age <= MAX_STALE_DAYS) & np.isfinite(v[idx.clip(min=0)])
        if valid.any():
            out[valid, col] = v[idx[valid]]
    return out


def _level(ctx, snap):
    ratio = pd.to_numeric(snap["pledge_ratio"], errors="coerce").to_numpy(np.float64)
    values = np.where(np.isfinite(ratio) & (ratio >= 0.0) & (ratio <= 100.0), ratio, np.nan)
    return _map_snapshots(ctx, snap, values, "level")


def _over_freefloat(ctx, snap):
    ratio = pd.to_numeric(snap["pledge_ratio"], errors="coerce").to_numpy(np.float64)
    fin = _place_finance(ctx, ["total_share", "float_share", "free_share"])
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    codes = np.asarray(ctx.panel.codes).astype(str)
    cpos = {c: i for i, c in enumerate(codes)}
    total = fin["total_share"]
    flt = fin["float_share"]
    free = fin["free_share"]

    values = np.full(len(snap), np.nan, dtype=np.float64)
    for code, g in snap.groupby("stock_code", sort=False):
        col = cpos.get(code)
        if col is None:
            continue
        E = g["E"].to_numpy(np.int64)
        posE = np.searchsorted(dates, E, side="left")
        exact = (posE < len(dates)) & (dates[np.clip(posE, 0, len(dates) - 1)] == E)
        posE = np.clip(posE, 0, len(dates) - 1)
        totE = total[posE, col]
        fltE = flt[posE, col]
        freE = free[posE, col]
        r = ratio[g.index.to_numpy()]
        good = exact & np.isfinite(r) & (totE > 0) & (freE > 0) & (fltE > 0)
        good &= freE <= fltE * (1.0 + 1e-9)
        good &= fltE <= totE * (1.0 + 1e-9)
        with np.errstate(divide="ignore", invalid="ignore"):
            values[g.index.to_numpy()] = np.where(good, r * totE / freE, np.nan)
    return _map_snapshots(ctx, snap, values, "over_freefloat")


def build(parameters):
    variant = parameters.get("variant")
    if variant not in NAMES:
        raise KeyError(f"unknown R38 pledge variant: {variant!r}")

    def compute(ctx):
        snap = _pledge_snapshots(ctx)
        if snap is None:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        if variant == "pl_ratio_level_v1":
            return _level(ctx, snap)
        return _over_freefloat(ctx, snap)

    return compute
