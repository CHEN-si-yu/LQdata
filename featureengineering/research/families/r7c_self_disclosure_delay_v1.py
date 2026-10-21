"""R7 C: completed-report delay relative to own prior three same-season reports.

Weak negative prior. Dates only; no unresolved waiting-pressure signal.
All date-only history is read so yearly/prefix engine windows cannot truncate
same-season baselines. Source announcement fields are not captured arrival vintages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL = "self_relative_report_delay"
PARAMETERS = {"signal": SIGNAL, "same_season_years": 3, "active_trading_days": 60}
COLUMNS = ["stock_code", "ann_date", "f_ann_date", "end_date"]
ACTIVE_TD = 60


def valid_dates(series):
    """Parse actual calendar dates; missing/malformed values become zero."""
    text = series.astype("string").str.slice(0, 10)
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    compact = text.str.fullmatch(r"\d{8}", na=False)
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(
            text.loc[compact], format="%Y%m%d", errors="coerce")
    return (parsed.dt.year * 10000 + parsed.dt.month * 100
            + parsed.dt.day).fillna(0).to_numpy(dtype=np.int64)


def _ordinals(days):
    parsed = pd.to_datetime(np.asarray(days).astype(str), format="%Y%m%d",
                            errors="raise")
    return parsed.to_numpy(dtype="datetime64[D]").astype(np.int64)


def _axis(days, label):
    values = np.asarray(days, dtype=np.int64)
    if values.ndim != 1 or not len(values) or np.any(np.diff(values) <= 0):
        raise ValueError(f"{label} must be nonempty, unique, increasing")
    _ordinals(values)
    return values


def first_reports(frame, code_index, cutoff):
    """First *valid* announcement per code/quarter, under the output cutoff."""
    empty = pd.DataFrame(columns=["code", "end", "pit", "L"])
    if frame is None or frame.empty:
        return empty
    end = valid_dates(frame["end_date"])
    ann = valid_dates(frame["ann_date"])
    fann = valid_dates(frame["f_ann_date"])
    pit = np.maximum(ann, fann)
    keep = (end > 0) & np.isin(end % 10000, [331, 630, 930, 1231])
    # Never max with end: no real announcement => no fabricated event.
    keep &= (pit > 0) & (pit >= end) & (pit <= int(cutoff))
    codes = np.asarray(code_index(frame["stock_code"].to_numpy()[keep]),
                       dtype=np.int64)
    rows = pd.DataFrame({"code": codes, "end": end[keep], "pit": pit[keep]})
    rows = rows.loc[rows.code >= 0].sort_values(
        ["code", "end", "pit"], kind="stable").drop_duplicates(
            ["code", "end"], keep="first").reset_index(drop=True)
    if rows.empty:
        return empty
    rows["L"] = _ordinals(rows.pit) - _ordinals(rows.end)
    return rows


def disclosure_events(first, calendar):
    """Expose frozen L/B/D on monotonically advancing report-period frontiers."""
    columns = ["code", "end", "pit", "L", "B", "value", "visible", "release_pos"]
    if first.empty:
        return pd.DataFrame(columns=columns)
    rows = first.copy()
    baseline = np.full(len(rows), np.nan, dtype=np.float64)
    for _, group in rows.groupby("code", sort=False):
        history = {int(e): (int(a), int(l)) for e, a, l in
                   zip(group.end, group.pit, group.L)}
        for idx, ev in zip(group.index, group.itertuples(index=False)):
            previous = [history.get(int(ev.end) - 10000 * year)
                        for year in (1, 2, 3)]
            if all(item is not None and item[0] < int(ev.pit)
                   for item in previous):
                baseline[int(idx)] = np.median([item[1] for item in previous])
    rows["B"] = baseline
    rows["value"] = rows.L.to_numpy(dtype=np.float64) - baseline
    positions = np.searchsorted(calendar, rows.pit.to_numpy(), side="right")
    # Announcements before the available calendar can inform old baselines, but
    # their release cannot be inferred as the first calendar day.
    known = (rows.pit.to_numpy() >= calendar[0]) & (positions < len(calendar))
    rows = rows.loc[known].copy()
    rows["release_pos"] = positions[known]
    rows["visible"] = calendar[positions[known]]
    rows = rows.sort_values(["code", "visible", "end"], kind="stable")
    rows = rows.drop_duplicates(["code", "visible"], keep="last")
    selected = []
    for _, group in rows.groupby("code", sort=False):
        frontier = -1
        for idx, ev in zip(group.index, group.itertuples(index=False)):
            if int(ev.end) > frontier:
                selected.append(idx)
                frontier = int(ev.end)
    return rows.loc[selected, columns].reset_index(drop=True)


def _map(ctx, events, field):
    if events.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    return ctx.panel.asof(events.code.to_numpy(dtype=np.int64),
                          events.visible.to_numpy(dtype=np.int64),
                          events[field].to_numpy(dtype=np.float64),
                          dtype=np.float64)


def event_state(ctx):
    """Reusable date-state table for fixed L/B/Age/report-season audit controls."""
    days = _axis(ctx.panel.dates, "panel dates")
    calendar = _axis(ctx.cal.days, "exchange calendar")
    positions = np.searchsorted(calendar, days)
    if np.any(positions >= len(calendar)) or not np.array_equal(calendar[positions], days):
        raise ValueError("every panel date must belong to the exchange calendar")
    if len(set(ctx.panel.codes.tolist())) != len(ctx.panel.codes):
        raise ValueError("panel stock keys must be unique")
    # No years/window restriction: only four small date columns over full source
    # history. Financial numerical values (including compatibility revenue) unused.
    frame = ctx.dataset("stock_income", columns=COLUMNS, years=None)
    first = first_reports(frame, ctx.code_index, int(days[-1]))
    events = disclosure_events(first, calendar)
    age = positions[:, None] - _map(ctx, events, "release_pos")
    return events, age


def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only the single frozen R7 C definition is permitted")

    def compute(ctx):
        events, age = event_state(ctx)
        values = _map(ctx, events, "value")
        traded = np.asarray(ctx.traded(), dtype=bool)
        if traded.shape != ctx.panel.shape:
            raise ValueError("traded mask must match the panel stock/date axes")
        valid = traded & (age >= 0) & (age < ACTIVE_TD) & np.isfinite(values)
        return np.where(valid, values, np.nan).astype(np.float64, copy=False)

    return compute
