"""F001 v2 research builder: PIT CCC seasonal/trend innovation.

Numeric implementation copied from the screened isolated v2 draft. The
separate two-stock PIT fixture remains under artifacts/research/round14_20261007.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from fea.deriv import KEY_W, PERIOD_BASE, PERIOD_W, period_of

WINDOW = 12
FULL_RANK = 5
RESIDUAL_DF = WINDOW - FULL_RANK
RMSE_FLOOR_DAYS = 1e-4
MAX_REPORT_AGE_DAYS = 550
EVENT_BATCH = 2048
AMOUNT_FLOOR = 1_000_000.0
COMPANY_TYPE = 1
REPORT_TYPE = 1
PARAMETERS = {
    "signal": "quarterly_ccc_seasonal_trend_innovation",
    "window_quarters": WINDOW,
    "full_rank": FULL_RANK,
    "residual_df": RESIDUAL_DF,
    "rmse_floor_days": RMSE_FLOOR_DAYS,
    "direction": "lower_raw_is_better",
}
INCOME_FIELDS = ("revenue", "oper_cost")
BALANCE_FIELDS = ("inventories", "accounts_receiv", "acct_payable")


def _valid_dates(ctx, values):
    raw = np.asarray(ctx.date_col(values), dtype=np.int64)
    parsed = pd.to_datetime(raw.astype(str), format="%Y%m%d", errors="coerce")
    return np.where((raw >= 19900101) & ~np.asarray(pd.isna(parsed)), raw, 0)


def _source_years(ctx):
    """Read exact report-year partitions needed for q-16 through panel end."""
    if not ctx.panel.T:
        return (PERIOD_BASE, PERIOD_BASE)
    lo = int(ctx.panel.dates[0]) // 10000
    hi = int(ctx.panel.dates[-1]) // 10000
    return max(PERIOD_BASE, lo - 5), hi


def _report_table(ctx, dataset, fields):
    columns = ["stock_code", "ann_date", "f_ann_date", "end_date",
               "comp_type", "report_type", *fields]
    frame = ctx.dataset(dataset, columns=columns, years=_source_years(ctx))
    out_columns = ["code", "period", "pit", *fields]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=out_columns)

    end = _valid_dates(ctx, frame["end_date"])
    ann = _valid_dates(ctx, frame["ann_date"])
    fann = _valid_dates(ctx, frame["f_ann_date"])
    pit = np.maximum(np.maximum(ann, fann), end)
    comp = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy(dtype=np.float64)
    report = pd.to_numeric(frame["report_type"], errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= (end >= 19900101) & (pit >= end) & (pit > 0)
    keep &= (ann >= 19900101) | (fann >= 19900101)
    # Restrict scope before same-PIT whole-row selection and frontier creation.
    keep &= (comp == COMPANY_TYPE) & (report == REPORT_TYPE)
    if ctx.panel.T:
        keep &= pit <= int(ctx.panel.dates[-1])
    codes = ctx.code_index(frame["stock_code"].to_numpy()[keep])
    data = pd.DataFrame({"code": codes, "period": period_of(end[keep]),
                         "pit": pit[keep]})
    for field in fields:
        # Keep a whole new row even when its amount is NaN; it clears older state.
        data[field] = pd.to_numeric(frame[field], errors="coerce").to_numpy(
            dtype=np.float64)[keep]
    data = data.loc[data.code >= 0]
    return data.drop_duplicates(["code", "period", "pit"], keep="last")


def _sorted_versions(data, fields):
    if data.empty:
        return {"keys": np.empty(0, dtype=np.int64),
                "code": np.empty(0, dtype=np.int64),
                "period": np.empty(0, dtype=np.int64),
                **{field: np.empty(0, dtype=np.float64) for field in fields}}
    data = data.sort_values(["code", "period", "pit"], kind="stable")
    code = data.code.to_numpy(dtype=np.int64)
    period = data.period.to_numpy(dtype=np.int64)
    pit = data.pit.to_numpy(dtype=np.int64)
    return {"keys": (code * PERIOD_W + period) * KEY_W + pit,
            "code": code, "period": period,
            **{field: data[field].to_numpy(dtype=np.float64) for field in fields}}


def _lookup(vt, code, period, pit, field):
    code, period, pit = np.broadcast_arrays(code, period, pit)
    if not len(vt["keys"]):
        return np.full(code.shape, np.nan, dtype=np.float64)
    query = (code * PERIOD_W + period) * KEY_W + pit
    pos = np.searchsorted(vt["keys"], query, side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (vt["code"][safe] == code) & (vt["period"][safe] == period)
    return np.where(ok, vt[field][safe], np.nan).astype(np.float64, copy=False)


def _strict_ttm(vt, code, period, pit, field):
    """Strict TTM from exact YTD slots as they were visible at `pit`."""
    code, period, pit = np.broadcast_arrays(code, period, pit)
    current = _lookup(vt, code, period, pit, field)
    phase = period % 4
    prior_fy_q4 = period - (phase + 1)
    prior_same_q = period - 4
    prior_q4 = _lookup(vt, code, prior_fy_q4, pit, field)
    prior_same = _lookup(vt, code, prior_same_q, pit, field)
    result = np.where(phase == 3, current, current + prior_q4 - prior_same)
    needed = np.isfinite(current) & np.where(phase == 3, True,
                                              np.isfinite(prior_q4) & np.isfinite(prior_same))
    return np.where(needed & np.isfinite(result), result, np.nan)


def _frontier_events(data):
    """Per-table prefix-max report-period state at each publication PIT."""
    if data.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    state = data[["code", "period", "pit"]].drop_duplicates()
    state = state.sort_values(["code", "pit", "period"], kind="stable")
    state["q"] = state.groupby("code", sort=False).period.cummax()
    return state.drop_duplicates(["code", "pit"], keep="last")[
        ["code", "pit", "q"]].reset_index(drop=True)


def _frontier_at(frontiers, code, pit):
    if frontiers.empty:
        return np.full(len(code), -1, dtype=np.int64)
    code = np.asarray(code, dtype=np.int64)
    pit = np.asarray(pit, dtype=np.int64)
    keys = frontiers.code.to_numpy(dtype=np.int64) * KEY_W + frontiers.pit.to_numpy(dtype=np.int64)
    query = code * KEY_W + pit
    pos = np.searchsorted(keys, query, side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (frontiers.code.to_numpy(dtype=np.int64)[safe] == code)
    return np.where(ok, frontiers.q.to_numpy(dtype=np.int64)[safe], -1)


def _event_table(income, balance):
    pieces = [data[["code", "pit"]] for data in (income, balance) if not data.empty]
    if not pieces:
        return pd.DataFrame(columns=["code", "pit", "q"])
    events = pd.concat(pieces, ignore_index=True).drop_duplicates().sort_values(
                           ["code", "pit"], kind="stable").reset_index(drop=True)
    code = events.code.to_numpy(dtype=np.int64)
    pit = events.pit.to_numpy(dtype=np.int64)
    iq = _frontier_at(_frontier_events(income), code, pit)
    bq = _frontier_at(_frontier_events(balance), code, pit)
    common = (iq >= 0) & (iq == bq)
    # Keep mismatched publication events. q=-1 maps to NaN and deliberately
    # clears a previously valid state until both table frontiers match again.
    events["q"] = np.where(common, iq, -1)
    return events


def _ccc_batch(events, income_vt, balance_vt):
    """Compute current and 12 historical CCCs with each lookup PIT-capped."""
    result = np.full(len(events), np.nan, dtype=np.float64)
    event_rows = np.flatnonzero(events.q.to_numpy(dtype=np.int64) >= 0)
    matched = events.iloc[event_rows].reset_index(drop=True)
    for lo in range(0, len(matched), EVENT_BATCH):
        hi = min(lo + EVENT_BATCH, len(matched))
        block = matched.iloc[lo:hi]
        code = block.code.to_numpy(dtype=np.int64)
        pit = block.pit.to_numpy(dtype=np.int64)
        q = block.q.to_numpy(dtype=np.int64)
        periods = q[:, None] - np.arange(WINDOW, -1, -1, dtype=np.int64)[None, :]
        code_grid = np.broadcast_to(code[:, None], periods.shape)
        pit_grid = np.broadcast_to(pit[:, None], periods.shape)

        revenue = _strict_ttm(income_vt, code_grid, periods, pit_grid, "revenue")
        cost = _strict_ttm(income_vt, code_grid, periods, pit_grid, "oper_cost")
        inv = _lookup(balance_vt, code_grid, periods, pit_grid, "inventories")
        receiv = _lookup(balance_vt, code_grid, periods, pit_grid, "accounts_receiv")
        payable = _lookup(balance_vt, code_grid, periods, pit_grid, "acct_payable")

        valid = np.isfinite(revenue) & np.isfinite(cost)
        valid &= (revenue > AMOUNT_FLOOR) & (cost > AMOUNT_FLOOR)
        valid &= np.isfinite(inv) & np.isfinite(receiv) & np.isfinite(payable)
        valid &= (inv >= 0) & (receiv >= 0) & (payable >= 0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ccc = 365.0 * (inv / cost + receiv / revenue - payable / cost)
        valid &= np.isfinite(ccc)
        complete = valid.all(axis=1)
        if not complete.any():
            continue

        # Every training observation is consecutive and complete by construction.
        train = ccc[:, :WINDOW]
        current = ccc[:, WINDOW]
        for phase in range(4):
            rows = np.flatnonzero(complete & (q % 4 == phase))
            if not len(rows):
                continue
            train_periods = periods[rows[0], :WINDOW]
            t = np.arange(WINDOW, dtype=np.float64)
            X = np.column_stack([
                np.ones(WINDOW, dtype=np.float64), t,
                (train_periods % 4 == 1).astype(np.float64),
                (train_periods % 4 == 2).astype(np.float64),
                (train_periods % 4 == 3).astype(np.float64),
            ])
            if np.linalg.matrix_rank(X) != FULL_RANK:
                continue
            try:
                beta, _, rank, _ = np.linalg.lstsq(X, train[rows].T, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if rank != FULL_RANK:
                continue
            residual = train[rows].T - X @ beta
            rmse = np.sqrt(np.sum(residual * residual, axis=0) / RESIDUAL_DF)
            forecast_row = np.array([1.0, 12.0,
                                     float(phase == 1), float(phase == 2),
                                     float(phase == 3)], dtype=np.float64)
            forecast = forecast_row @ beta
            value = (current[rows] - forecast) / rmse
            ok = np.isfinite(rmse) & (rmse >= RMSE_FLOOR_DAYS)
            ok &= np.isfinite(forecast) & np.isfinite(current[rows]) & np.isfinite(value)
            result[event_rows[lo + rows[ok]]] = value[ok]
    return result


def _period_end(period):
    period = np.asarray(period, dtype=np.int64)
    md = np.array([331, 630, 930, 1231], dtype=np.int64)[period % 4]
    return (PERIOD_BASE + period // 4) * 10000 + md


def _epoch_days(yyyymmdd):
    return pd.to_datetime(np.asarray(yyyymmdd).astype(str), format="%Y%m%d",
                          errors="raise").to_numpy(dtype="datetime64[D]").astype(np.int64)


def _map_fresh(ctx, events, values):
    empty = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    if events.empty or not ctx.panel.T:
        return empty
    pit = events.pit.to_numpy(dtype=np.int64)
    effective = np.asarray(ctx.next_trading_day(pit), dtype=np.int64)
    end = _period_end(events.q.to_numpy(dtype=np.int64))
    usable = (effective > pit) & (effective <= int(ctx.panel.dates[-1]))
    if not usable.any():
        return empty
    mapped = pd.DataFrame({"code": events.code.to_numpy(dtype=np.int64)[usable],
                           "effective": effective[usable], "pit": pit[usable],
                           "end_epoch": _epoch_days(end[usable]),
                           "value": np.asarray(values, dtype=np.float64)[usable]})
    mapped = mapped.sort_values(["code", "effective", "pit"], kind="stable")
    # Multiple weekend/holiday releases mapped to one next session resolve to
    # the last visible source PIT for that stock/session. A NaN is state-clearing.
    mapped = mapped.drop_duplicates(["code", "effective"], keep="last")
    code = mapped.code.to_numpy(dtype=np.int64)
    effective = mapped.effective.to_numpy(dtype=np.int64)
    state = ctx.panel.asof(code, effective, mapped.value.to_numpy(dtype=np.float64),
                           dtype=np.float64)
    report_end = ctx.panel.asof(code, effective, mapped.end_epoch.to_numpy(dtype=np.float64),
                                dtype=np.float64)
    age = _epoch_days(ctx.panel.dates)[:, None] - report_end
    valid = np.isfinite(state) & np.isfinite(age) & (age >= 0) & (age <= MAX_REPORT_AGE_DAYS)
    valid &= np.asarray(ctx.universe, dtype=bool) & np.asarray(ctx.traded(), dtype=bool)
    return np.where(valid, state, np.nan)


def build(parameters=PARAMETERS):
    if parameters != PARAMETERS:
        raise ValueError("Only the frozen F001 definition is supported")

    def compute(ctx):
        income = _report_table(ctx, "stock_income", INCOME_FIELDS)
        balance = _report_table(ctx, "stock_balancesheet", BALANCE_FIELDS)
        events = _event_table(income, balance)
        if events.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        income_vt = _sorted_versions(income, INCOME_FIELDS)
        balance_vt = _sorted_versions(balance, BALANCE_FIELDS)
        values = _ccc_batch(events, income_vt, balance_vt)
        return _map_fresh(ctx, events, values)

    return compute


