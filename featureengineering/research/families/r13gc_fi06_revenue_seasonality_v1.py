"""Artifact-only R13 FI06: historical calendar-quarter share of annual sales.

This is a GC accounting descriptor proposal, not an alpha claim. It uses only
standard consolidated stock_income revenue rows and their retained PIT
vintages. The output for a current report-quarter q is the equal-weighted
average of q's share in each of the three immediately preceding fiscal years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


DATASET = "stock_income"
FIELD = "revenue"
COMPANY_TYPE = 1
REPORT_TYPE = 1
PERIOD_BASE = 1990
PERIOD_W = 100_000
KEY_W = 100_000_000
MIN_ANNUAL_REVENUE = 1_000_000.0  # denominator must be strictly greater
MAX_REPORT_AGE_DAYS = 550
WARMUP_DAYS = 1800

VARIANT = "FI06_revenue_calendar_seasonal_share"
PARAMETERS = {"variant": VARIANT, "definition_version": 1}
DEPENDENCIES = ("stock_income", "stock_daily")


def _valid_dates(ctx, values):
    raw = np.asarray(ctx.date_col(values), dtype=np.int64)
    parsed = pd.to_datetime(raw.astype(str), format="%Y%m%d", errors="coerce")
    valid = (raw >= 19900101) & ~np.asarray(pd.isna(parsed))
    return np.where(valid, raw, 0)


def _period_of(end_date):
    end_date = np.asarray(end_date, dtype=np.int64)
    year = end_date // 10000
    month = (end_date // 100) % 100
    return ((year - PERIOD_BASE) * 4 + (month // 3 - 1)).astype(np.int64)


def _period_end(period):
    period = np.asarray(period, dtype=np.int64)
    month_day = np.array([331, 630, 930, 1231], dtype=np.int64)[period % 4]
    year = PERIOD_BASE + period // 4
    return year * 10000 + month_day


def _yyyymmdd_to_days(values):
    values = np.asarray(values, dtype=np.int64)
    return pd.to_datetime(values.astype(str), format="%Y%m%d", errors="raise") \
        .to_numpy(dtype="datetime64[D]").astype(np.int64)


def _empty_table():
    return pd.DataFrame(columns=["code", "period", "pit", FIELD])


def _report_table(ctx):
    """Read standard report rows and preserve whole-row PIT versions.

    Type filtering precedes same-key/PIT de-duplication. A physical last row
    wins for a tied standard key even if revenue is nonfinite; that row clears
    the old amount at that PIT rather than falling back to an earlier row.
    """
    columns = [
        "stock_code", "ann_date", "f_ann_date", "end_date",
        "comp_type", "report_type", FIELD,
    ]
    frame = ctx.dataset(DATASET, columns=columns, years=None)
    if frame is None or frame.empty:
        return _empty_table()

    end = _valid_dates(ctx, frame["end_date"])
    ann = _valid_dates(ctx, frame["ann_date"])
    fann = _valid_dates(ctx, frame["f_ann_date"])
    pit = np.maximum(np.maximum(ann, fann), end)
    comp = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    report = pd.to_numeric(frame["report_type"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    codes_all = np.asarray(ctx.code_index(frame["stock_code"].to_numpy()),
                           dtype=np.int64)

    keep = codes_all >= 0
    keep &= np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= end >= 19900101
    keep &= ((ann >= 19900101) | (fann >= 19900101))
    keep &= pit >= end
    keep &= (comp == COMPANY_TYPE) & (report == REPORT_TYPE)
    if ctx.panel.T:
        keep &= pit <= int(ctx.panel.dates[-1])
    else:
        keep &= False
    if not keep.any():
        return _empty_table()

    values = pd.to_numeric(frame[FIELD], errors="coerce").to_numpy(
        dtype=np.float64
    )[keep]
    data = pd.DataFrame({
        "code": codes_all[keep],
        "period": _period_of(end[keep]),
        "pit": pit[keep],
        FIELD: values,
    })
    return data.drop_duplicates(["code", "period", "pit"], keep="last")


def _frontier_events(data):
    """Make one event per code/PIT, with prefix-max standard report quarter."""
    if data.empty:
        return pd.DataFrame(columns=["code", "pit", "period"])
    events = data[["code", "period", "pit"]].copy()
    events = events.sort_values(["code", "pit", "period"], kind="stable")
    events["frontier"] = events.groupby("code", sort=False).period.cummax()
    events = events.drop_duplicates(["code", "pit"], keep="last")
    return events[["code", "pit", "frontier"]].rename(
        columns={"frontier": "period"}
    ).reset_index(drop=True)


def _sorted_versions(data):
    data = data.sort_values(["code", "period", "pit"], kind="stable")
    code = data.code.to_numpy(dtype=np.int64)
    period = data.period.to_numpy(dtype=np.int64)
    pit = data.pit.to_numpy(dtype=np.int64)
    return {
        "keys": (code * PERIOD_W + period) * KEY_W + pit,
        "code": code,
        "period": period,
        FIELD: data[FIELD].to_numpy(dtype=np.float64),
    }


def _lookup(vt, code, period, pit):
    code, period, pit = np.broadcast_arrays(code, period, pit)
    if not len(vt["keys"]):
        return np.full(code.shape, np.nan, dtype=np.float64)
    query = (code * PERIOD_W + period) * KEY_W + pit
    pos = np.searchsorted(vt["keys"], query, side="right") - 1
    safe = np.maximum(pos, 0)
    valid = pos >= 0
    valid &= vt["code"][safe] == code
    valid &= vt["period"][safe] == period
    # Latest as-of vintage wins, including nonfinite amounts.
    return np.where(valid, vt[FIELD][safe], np.nan).astype(np.float64,
                                                          copy=False)


def _seasonal_share_at(vt, code, frontier_period, pit):
    """Evaluate FI06 from 12 exact historical YTD rows plus current frontier.

    The three previous fiscal years are fixed by the frontier year. All four
    YTD rows from each year must be finite. No slot compression or zero fill.
    """
    code, frontier_period, pit = np.broadcast_arrays(
        np.asarray(code, dtype=np.int64),
        np.asarray(frontier_period, dtype=np.int64),
        np.asarray(pit, dtype=np.int64),
    )
    out = np.full(code.shape, np.nan, dtype=np.float64)
    if not code.size:
        return out

    current_ytd = _lookup(vt, code, frontier_period, pit)
    quarter_index = frontier_period % 4
    current_ok = np.isfinite(current_ytd)

    # Fixed absolute fiscal slots: for each of Y-3, Y-2, Y-1 take Q1..Q4.
    fiscal_year_start = (frontier_period // 4) * 4
    offsets = np.asarray([-12, -8, -4], dtype=np.int64)
    slots = fiscal_year_start[:, None, None] \
        + offsets[None, :, None] + np.arange(4, dtype=np.int64)[None, None, :]
    values = _lookup(vt, code[:, None, None], slots, pit[:, None, None])
    complete = np.isfinite(values).all(axis=(1, 2))

    # Select current-quarter YTD for each FY and subtract only its exact
    # same-FY previous quarter. Q1 is already a single quarter.
    row = np.arange(len(code))
    q = quarter_index
    this_ytd = values[row[:, None], np.arange(3)[None, :], q[:, None]]
    if np.any(q == 0):
        prev_ytd = np.zeros_like(this_ytd)
        non_q1 = q > 0
        prev_ytd[non_q1] = values[
            row[non_q1, None],
            np.arange(3)[None, :],
            (q[non_q1] - 1)[:, None],
        ]
    else:
        prev_ytd = values[
            row[:, None], np.arange(3)[None, :], (q - 1)[:, None]
        ]
    single_quarter = this_ytd - prev_ytd
    annual = values[:, :, 3]
    denominator_ok = np.isfinite(annual).all(axis=1)
    denominator_ok &= (annual > MIN_ANNUAL_REVENUE).all(axis=1)

    eligible = current_ok & complete & denominator_ok
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        shares = single_quarter / annual
        value = shares.mean(axis=1)
    eligible &= np.isfinite(single_quarter).all(axis=1)
    eligible &= np.isfinite(value)
    out[eligible] = value[eligible]
    return out


def _compute_events(events, vt):
    if events.empty:
        return np.empty(0, dtype=np.float64)
    return _seasonal_share_at(
        vt,
        events.code.to_numpy(dtype=np.int64),
        events.period.to_numpy(dtype=np.int64),
        events.pit.to_numpy(dtype=np.int64),
    )


def _map_daily(ctx, events, values):
    empty = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    if events.empty or not ctx.panel.T:
        return empty

    pit = events.pit.to_numpy(dtype=np.int64)
    effective = np.asarray(ctx.next_trading_day(pit), dtype=np.int64)
    mapped = pd.DataFrame({
        "code": events.code.to_numpy(dtype=np.int64),
        "pit": pit,
        "effective": effective,
        "period": events.period.to_numpy(dtype=np.int64),
        "value": np.asarray(values, dtype=np.float64),
    })
    mapped = mapped[(mapped.effective > mapped.pit)
                    & (mapped.effective <= int(ctx.panel.dates[-1]))]
    if mapped.empty:
        return empty
    # If multiple PIT events map to one session, the later PIT state wins;
    # this also preserves an invalid revision as a clearing NaN event.
    mapped = mapped.sort_values(
        ["code", "effective", "pit"], kind="stable"
    ).drop_duplicates(["code", "effective"], keep="last")
    code = mapped.code.to_numpy(dtype=np.int64)
    effective = mapped.effective.to_numpy(dtype=np.int64)
    state = ctx.panel.asof(
        code, effective, mapped.value.to_numpy(dtype=np.float64), dtype=np.float64
    )
    report_end = _yyyymmdd_to_days(
        _period_end(mapped.period.to_numpy(dtype=np.int64))
    )
    end_state = ctx.panel.asof(
        code, effective, report_end.astype(np.float64), dtype=np.float64
    )
    panel_days = _yyyymmdd_to_days(ctx.panel.dates)
    age = panel_days[:, None] - end_state
    valid = np.isfinite(state) & (age >= 0) & (age <= MAX_REPORT_AGE_DAYS)
    valid &= np.asarray(ctx.universe, dtype=bool)
    valid &= np.asarray(ctx.traded(), dtype=bool)
    return np.where(valid, state, np.nan)


def _evaluate(ctx):
    data = _report_table(ctx)
    events = _frontier_events(data)
    values = _compute_events(events, _sorted_versions(data))
    return _map_daily(ctx, events, values)


def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only the exact frozen FI06 definition is allowed")
    return _evaluate
