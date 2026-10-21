"""R13-F draft: one-step, seasonal/trend-adjusted quarterly CFO innovation.

This module is intentionally isolated under artifacts and is not registered.
It uses only the source rows visible through the project PIT timestamp contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea.deriv import KEY_W, PERIOD_BASE, PERIOD_W, period_of

WINDOW = 12
FULL_RANK = 5
RESIDUAL_DF = WINDOW - FULL_RANK
ASSET_FLOOR = 1e6
RMSE_FLOOR = 1e-4
MAX_REPORT_AGE_DAYS = 550
EVENT_BATCH = 4096
PARAMETERS = {
    "signal": "ocf_seasonal_trend_innovation",
    "window": WINDOW,
    "full_rank": FULL_RANK,
    "residual_df": RESIDUAL_DF,
    "rmse_floor": RMSE_FLOOR,
}
COMPANY_TYPE = 1
REPORT_TYPE = 1
DIAGNOSTIC_NAMES = ("current_x", "seasonal_trend_forecast", "training_rmse")


def _valid_dates(ctx, values):
    raw = np.asarray(ctx.date_col(values), dtype=np.int64)
    parsed = pd.to_datetime(raw.astype(str), format="%Y%m%d", errors="coerce")
    return np.where((raw >= 19900101) & ~pd.isna(parsed), raw, 0)


def _report_table(ctx, dataset, field):
    columns = ["stock_code", "ann_date", "f_ann_date", "end_date",
               "comp_type", "report_type", field]
    frame = ctx.dataset(dataset, columns=columns)
    out_columns = ["code", "period", "pit", "comp_type", "report_type", field]
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
    # The candidate is scoped to ordinary commercial consolidated statements.
    # Filter source type before tied-PIT whole-row dedup and before frontier build.
    keep &= (comp == COMPANY_TYPE) & (report == REPORT_TYPE)
    if ctx.panel.T:
        keep &= pit <= int(ctx.panel.dates[-1])
    else:
        keep &= False

    codes = ctx.code_index(frame["stock_code"].to_numpy()[keep])
    data = pd.DataFrame({
        "code": codes,
        "period": period_of(end[keep]),
        "pit": pit[keep],
        "comp_type": comp[keep],
        "report_type": report[keep],
        field: pd.to_numeric(frame[field], errors="coerce").to_numpy(
            dtype=np.float64)[keep],
    })
    data = data.loc[data.code >= 0]
    # Whole-row physical-last wins at tied PIT, even if type/amount is invalid.
    return data.drop_duplicates(["code", "period", "pit"], keep="last")


def _sorted_versions(data, field):
    data = data.sort_values(["code", "period", "pit"], kind="stable")
    code = data.code.to_numpy(dtype=np.int64)
    period = data.period.to_numpy(dtype=np.int64)
    pit = data.pit.to_numpy(dtype=np.int64)
    return {
        "keys": (code * PERIOD_W + period) * KEY_W + pit,
        "code": code,
        "period": period,
        "comp_type": data.comp_type.to_numpy(dtype=np.float64),
        "report_type": data.report_type.to_numpy(dtype=np.float64),
        field: data[field].to_numpy(dtype=np.float64),
    }


def _lookup(vt, code, period, pit, field):
    code, period, pit = np.broadcast_arrays(code, period, pit)
    shape = code.shape
    if not len(vt["keys"]):
        return np.full(shape, np.nan, dtype=np.float64)
    query = (code * PERIOD_W + period) * KEY_W + pit
    pos = np.searchsorted(vt["keys"], query, side="right") - 1
    safe = np.maximum(pos, 0)
    valid = (pos >= 0) & (vt["code"][safe] == code) & (vt["period"][safe] == period)
    # Eligibility is tested only after the report-period frontier is established.
    valid &= (vt["comp_type"][safe] == COMPANY_TYPE)
    valid &= (vt["report_type"][safe] == REPORT_TYPE)
    return np.where(valid, vt[field][safe], np.nan).astype(np.float64, copy=False)


def _frontier_events(cash, balance):
    """Union frontier: presence in either table advances q before eligibility."""
    pieces = []
    for data in (cash, balance):
        if not data.empty:
            pieces.append(data[["code", "period", "pit"]])
    if not pieces:
        return pd.DataFrame(columns=["code", "pit", "q"])
    present = pd.concat(pieces, ignore_index=True).drop_duplicates()
    present = present.sort_values(["code", "pit", "period"], kind="stable")
    present["q"] = present.groupby("code", sort=False).period.cummax()
    # With simultaneous reports, sort-by-period makes the physical-last row the max.
    return present.drop_duplicates(["code", "pit"], keep="last")[
        ["code", "pit", "q"]].reset_index(drop=True)


def _period_end(period):
    period = np.asarray(period, dtype=np.int64)
    year = PERIOD_BASE + period // 4
    month_day = np.array([331, 630, 930, 1231], dtype=np.int64)[period % 4]
    return year * 10000 + month_day


def _epoch_days(yyyymmdd):
    return pd.to_datetime(np.asarray(yyyymmdd).astype(str), format="%Y%m%d",
                          errors="raise").to_numpy(dtype="datetime64[D]").astype(np.int64)


def _compute_batch(events, cash_vt, balance_vt):
    """Return signal and its three fixed same-support diagnostic components."""
    out = np.full((len(events), 4), np.nan, dtype=np.float64)
    if events.empty:
        return out

    for lo in range(0, len(events), EVENT_BATCH):
        hi = min(lo + EVENT_BATCH, len(events))
        block = events.iloc[lo:hi]
        code = block.code.to_numpy(dtype=np.int64)
        q = block.q.to_numpy(dtype=np.int64)
        pit = block.pit.to_numpy(dtype=np.int64)

        # 14 source quarter points q-13..q supply 13 singles q-12..q.
        offsets = np.arange(13, -1, -1, dtype=np.int64)
        periods = q[:, None] - offsets[None, :]
        code_grid = np.broadcast_to(code[:, None], periods.shape)
        pit_grid = np.broadcast_to(pit[:, None], periods.shape)
        ytd = _lookup(cash_vt, code_grid, periods, pit_grid, "n_cashflow_act")
        assets = _lookup(balance_vt, code_grid, periods, pit_grid, "total_assets")

        quarter_periods = periods[:, 1:]
        cfo_single = np.where(quarter_periods % 4 == 0,
                              ytd[:, 1:], ytd[:, 1:] - ytd[:, :-1])
        avg_assets = (assets[:, 1:] + assets[:, :-1]) / 2.0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            x = cfo_single / avg_assets
        # All 13 current YTD values are needed. A prior YTD is needed only when
        # the corresponding single quarter is Q2-Q4; a Q1 single is its own YTD.
        needed_previous = quarter_periods % 4 != 0
        valid_ytd = np.isfinite(ytd[:, 1:]).all(axis=1)
        valid_ytd &= np.where(needed_previous, np.isfinite(ytd[:, :-1]), True).all(axis=1)
        valid = valid_ytd & np.isfinite(assets).all(axis=1)
        valid &= (assets > ASSET_FLOOR).all(axis=1)
        valid &= np.isfinite(cfo_single).all(axis=1) & np.isfinite(avg_assets).all(axis=1)
        valid &= (avg_assets > ASSET_FLOOR).all(axis=1) & np.isfinite(x).all(axis=1)
        if not valid.any():
            continue

        # x columns are q-12,...,q. Training excludes current q by construction.
        train_periods = quarter_periods[:, :WINDOW]
        train_values = x[:, :WINDOW]
        current_x = x[:, WINDOW]
        for phase in range(4):
            rows = np.flatnonzero(valid & (q % 4 == phase))
            if not len(rows):
                continue
            train_p = train_periods[rows[0]]
            t = np.arange(WINDOW, dtype=np.float64)
            X = np.column_stack([
                np.ones(WINDOW, dtype=np.float64), t,
                (train_p % 4 == 1).astype(np.float64),
                (train_p % 4 == 2).astype(np.float64),
                (train_p % 4 == 3).astype(np.float64),
            ])
            if np.linalg.matrix_rank(X) != FULL_RANK:
                continue
            try:
                beta, _, rank, _ = np.linalg.lstsq(
                    X, train_values[rows].T, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if rank != FULL_RANK:
                continue
            residual = train_values[rows].T - X @ beta
            with np.errstate(invalid="ignore", over="ignore"):
                rmse = np.sqrt(np.sum(residual * residual, axis=0) / RESIDUAL_DF)
            q_design = np.array([1.0, 12.0,
                                 float(phase == 1), float(phase == 2),
                                 float(phase == 3)], dtype=np.float64)
            forecast = q_design @ beta
            now = current_x[rows]
            signal = (now - forecast) / rmse
            ok = np.isfinite(rmse) & (rmse >= RMSE_FLOOR)
            ok &= np.isfinite(forecast) & np.isfinite(now) & np.isfinite(signal)
            chosen = rows[ok]
            out[lo + chosen, 0] = signal[ok]
            out[lo + chosen, 1] = now[ok]
            out[lo + chosen, 2] = forecast[ok]
            out[lo + chosen, 3] = rmse[ok]
    return out


def _event_components(ctx):
    cash = _report_table(ctx, "stock_cashflow", "n_cashflow_act")
    balance = _report_table(ctx, "stock_balancesheet", "total_assets")
    events = _frontier_events(cash, balance)
    if events.empty:
        return events, np.full((0, 4), np.nan, dtype=np.float64)
    values = _compute_batch(events, _sorted_versions(cash, "n_cashflow_act"),
                            _sorted_versions(balance, "total_assets"))
    return events, values


def _map_fresh(ctx, events, values):
    empty = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    if events.empty or not ctx.panel.T:
        return empty
    pit = events.pit.to_numpy(dtype=np.int64)
    # Project financial-factor effective-date rule is applied exactly once.
    effective = np.asarray(ctx.next_trading_day(pit), dtype=np.int64)
    keep = (effective > pit) & (effective <= int(ctx.panel.dates[-1]))
    if not keep.any():
        return empty
    code = events.code.to_numpy(dtype=np.int64)[keep]
    q = events.q.to_numpy(dtype=np.int64)[keep]
    state = ctx.panel.asof(code, effective[keep], np.asarray(values)[keep], dtype=np.float64)
    end_state = ctx.panel.asof(code, effective[keep], _epoch_days(_period_end(q)),
                               dtype=np.float64)
    age = _epoch_days(ctx.panel.dates)[:, None] - end_state
    valid = (age >= 0) & (age <= MAX_REPORT_AGE_DAYS) & np.isfinite(state)
    valid &= np.asarray(ctx.universe, dtype=bool) & np.asarray(ctx.traded(), dtype=bool)
    return np.where(valid, state, np.nan)


def control_components(ctx):
    """Three frozen current-period diagnostics on the signal's exact support."""
    events, values = _event_components(ctx)
    return {name: _map_fresh(ctx, events, values[:, i + 1])
            for i, name in enumerate(DIAGNOSTIC_NAMES)}


def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only the single frozen R13-F definition is permitted")

    def compute(ctx):
        events, values = _event_components(ctx)
        return _map_fresh(ctx, events, values[:, 0])

    return compute
