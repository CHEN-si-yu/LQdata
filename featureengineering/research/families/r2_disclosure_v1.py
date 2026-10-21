"""R2-I4 disclosure timing / information rhythm (announcement-date-only).

Only date columns are consumed:
  stock_income / stock_balancesheet / stock_cashflow : ann_date, f_ann_date, end_date
  stock_financial_indicator                          : ann_date, end_date
  stock_forecast                                     : ann_date, first_ann_date, end_date, type

No accounting value is read. end_date is the *report period* and is never used
as an availability time. stock_forecast.end_date may point at a future period;
it is used only as a pairing key, never as an input value.

Availability contract (identical to fea/upstream.py::_audit_columns/_audit_frame,
so that the isolated truncation audit and the production read reveal the same rows):
  * financial statements -> a row is visible from max(ann_date, f_ann_date, end_date)
    and requires ann_date or f_ann_date >= 1990-01-01;
  * stock_forecast       -> a row is visible from ann_date and requires
    ann_date >= 1990-01-01.
'avail' below is exactly that day, and every grid placement uses 'avail', so no
value can appear before the row that carries it is readable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = (
    "earliness_last",
    "earliness_mean4",
    "peer_rank_last",
    "disclosure_gap",
    "baseline_forecast_type",
)

# (table, actual-announcement column or None) -- exactly the columns the engine
# audit treats as announcements for that table.
FIN_TABLES = (
    ("stock_income", "f_ann_date"),
    ("stock_balancesheet", "f_ann_date"),
    ("stock_cashflow", "f_ann_date"),
    ("stock_financial_indicator", None),
)
FIN_DEPS = tuple(name for name, _ in FIN_TABLES)

# Pre-registered plausibility guard: a periodic report first readable more than
# 60 calendar days after its statutory deadline is a supplier back-fill /
# restatement artefact, not a disclosure-timing observation.
EARLINESS_FLOOR = -60
# Pre-registered plausibility guard on forecast -> formal-report spacing.
GAP_MAX_DAYS = 400
# Look-back used by the persistence variant (number of announcements, not days).
MEAN_PERIODS = 4

# Pre-registered monotone "news goodness" ordering of stock_forecast.type.
# Fixed before any evaluation; not tuned on IC.
TYPE_SCORE = {
    "预增": 4.0, "扭亏": 3.0, "略增": 3.0, "续盈": 1.0,
    "不确定": 0.0, "其他": 0.0,
    "减亏": -1.0, "略减": -2.0, "预减": -3.0,
    "首亏": -4.0, "续亏": -4.0, "增亏": -4.0,
}


def _ordinal(days) -> np.ndarray:
    """YYYYMMDD int -> proleptic day number, so differences are calendar days.

    numpy's datetime64[D] treats an integer as a day count, not as YYYYMMDD, so
    the conversion has to go through an explicit %Y%m%d parse.
    """
    values = np.asarray(days, dtype=np.int64)
    stamps = pd.to_datetime(pd.Series(values).astype(str), format="%Y%m%d", errors="coerce")
    return stamps.to_numpy(dtype="datetime64[D]").astype(np.int64)


def _quarter_ends(end_date: np.ndarray) -> np.ndarray:
    month = (end_date // 100) % 100
    day = end_date % 100
    return (((month == 3) & (day == 31)) | ((month == 6) & (day == 30))
            | ((month == 9) & (day == 30)) | ((month == 12) & (day == 31)))


def _statutory_deadline(end_date: np.ndarray) -> np.ndarray:
    """CSRC/SSE periodic-report deadlines: Q1 04-30, H1 08-31, Q3 10-31, FY 04-30."""
    year = end_date // 10000
    month = (end_date // 100) % 100
    return np.where(
        month == 12, (year + 1) * 10000 + 430,
        np.where(month == 6, year * 10000 + 831,
                 np.where(month == 9, year * 10000 + 1031, year * 10000 + 430)))


def _periodic_pairs(ctx):
    """(stock, report period) -> earliest day any of the four statements is readable."""
    parts = []
    for name, actual_col in FIN_TABLES:
        columns = ["stock_code", "ann_date", "end_date"]
        if actual_col:
            columns.append(actual_col)
        frame = ctx.dataset(name, columns=columns)
        if frame is None or frame.empty:
            continue
        frame = frame[frame["stock_code"].isin(ctx.panel.codes)]
        if frame.empty:
            continue
        announced = ctx.date_col(frame["ann_date"]).astype(np.int64)
        period_end = ctx.date_col(frame["end_date"]).astype(np.int64)
        if actual_col:
            actual = ctx.date_col(frame[actual_col]).astype(np.int64)
            known = (announced >= 19900101) | (actual >= 19900101)
            avail = np.maximum(np.maximum(announced, actual), period_end)
        else:
            known = announced >= 19900101
            avail = np.maximum(announced, period_end)
        parts.append(pd.DataFrame({
            "stock_code": frame["stock_code"].to_numpy()[known],
            "end_date": period_end[known],
            "avail": avail[known]}))
    if not parts:
        return None
    rows = pd.concat(parts, ignore_index=True)
    rows = rows[_quarter_ends(rows["end_date"].to_numpy(dtype=np.int64))]
    if rows.empty:
        return None
    pairs = rows.groupby(["stock_code", "end_date"], as_index=False)["avail"].min()
    end_date = pairs["end_date"].to_numpy(dtype=np.int64)
    avail = pairs["avail"].to_numpy(dtype=np.int64)
    earliness = _ordinal(_statutory_deadline(end_date)) - _ordinal(avail)
    # The statutory window bounds earliness from above by construction; the floor
    # removes supplier back-fills of long-expired report periods.
    keep = earliness >= EARLINESS_FLOOR
    pairs = pairs.loc[keep].copy()
    pairs["earliness"] = earliness[keep].astype(np.float64)
    return pairs


def _forecast_rows(ctx):
    frame = ctx.dataset("stock_forecast", columns=[
        "stock_code", "ann_date", "first_ann_date", "end_date", "type"])
    if frame is None or frame.empty:
        return None
    frame = frame[frame["stock_code"].isin(ctx.panel.codes)]
    if frame.empty:
        return None
    announced = ctx.date_col(frame["ann_date"]).astype(np.int64)
    keep = announced >= 19900101
    if not keep.any():
        return None
    return pd.DataFrame({
        "stock_code": frame["stock_code"].to_numpy()[keep],
        "end_date": ctx.date_col(frame["end_date"]).astype(np.int64)[keep],
        "ann_date": announced[keep],
        "first_ann_date": ctx.date_col(frame["first_ann_date"]).astype(np.int64)[keep],
        "type": frame["type"].to_numpy()[keep]})


def _asof(ctx, stock_code, avail, values):
    grid = ctx.asof_daily(stock_code.to_numpy(), avail.to_numpy(dtype=np.int64),
                          np.asarray(values, dtype=np.float64))
    grid = np.asarray(grid, dtype=np.float64)
    return np.where(np.isfinite(grid), grid, np.nan)


def _earliness_last(ctx, pairs):
    return _asof(ctx, pairs["stock_code"], pairs["avail"], pairs["earliness"])


def _earliness_mean4(ctx, pairs):
    ordered = pairs.sort_values(["stock_code", "avail", "end_date"], kind="mergesort")
    smoothed = ordered.groupby("stock_code")["earliness"].transform(
        lambda s: s.rolling(MEAN_PERIODS, min_periods=1).mean())
    return _asof(ctx, ordered["stock_code"], ordered["avail"], smoothed)


def _peer_rank_last(ctx, pairs):
    """Pool percentile of a firm's own announcement day inside its report period.

    rank(method='max') counts the pool members whose announcement is already
    readable on that day (ties included), and the denominator is the frozen pool
    size, a constant. The value is therefore fully known on the announcement day
    itself and never refers to a peer that discloses later.
    """
    ordered = pairs.sort_values(["end_date", "avail", "stock_code"], kind="mergesort")
    count = ordered.groupby("end_date")["avail"].rank(method="max").to_numpy(dtype=np.float64)
    return _asof(ctx, ordered["stock_code"], ordered["avail"],
                 count / float(ctx.panel.C))


def _disclosure_gap(ctx, pairs, forecasts):
    """Calendar days between the first forecast announcement and the formal report."""
    first = forecasts.sort_values(
        ["stock_code", "end_date", "ann_date", "first_ann_date"], kind="mergesort")
    first = first.groupby(["stock_code", "end_date"], as_index=False).agg(
        forecast_day=("ann_date", "first"), first_announce=("first_ann_date", "first"))
    report = pairs[["stock_code", "end_date", "avail"]].rename(columns={"avail": "report_day"})
    both = first.merge(report, on=["stock_code", "end_date"], how="inner")
    if both.empty:
        return None
    gap = _ordinal(both["report_day"]) - _ordinal(both["first_announce"])
    keep = (gap >= 0) & (gap <= GAP_MAX_DAYS)
    if not keep.any():
        return None
    both = both.loc[keep]
    # Known only once the formal report itself is readable (forecast_day <= report_day
    # for every surviving pair), so placement at report_day is not a forward look.
    return _asof(ctx, both["stock_code"], both["report_day"], gap[keep].astype(np.float64))


def _forecast_level(ctx, forecasts):
    score = forecasts["type"].map(TYPE_SCORE)
    keep = score.notna().to_numpy()
    if not keep.any():
        return None
    rows = forecasts.loc[keep]
    return _asof(ctx, rows["stock_code"], rows["ann_date"],
                 score.to_numpy(dtype=np.float64)[keep])


def disclosure_timing(parameters):
    """One of the pre-registered R2-I4 mechanisms; no searched windows or directions."""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("Family permits only the pre-registered R2-I4 signal identifiers")
    signal = parameters["signal"]

    def compute(ctx):
        forecasts = _forecast_rows(ctx) if signal in ("disclosure_gap", "baseline_forecast_type") else None
        pairs = _periodic_pairs(ctx) if signal != "baseline_forecast_type" else None

        if signal == "earliness_last":
            value = None if pairs is None else _earliness_last(ctx, pairs)
        elif signal == "earliness_mean4":
            value = None if pairs is None else _earliness_mean4(ctx, pairs)
        elif signal == "peer_rank_last":
            value = None if pairs is None else _peer_rank_last(ctx, pairs)
        elif signal == "disclosure_gap":
            value = None if (pairs is None or forecasts is None) else _disclosure_gap(ctx, pairs, forecasts)
        else:
            value = None if forecasts is None else _forecast_level(ctx, forecasts)

        if value is None:
            value = ctx.panel.empty(dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute