"""R8 E: approximate book working-capital cash-realization noise.
One frozen weak negative prior; neither literal DD nor manipulation detection.
Twenty contiguous historical rows use cashflow through the current disclosed q.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fea.deriv import PERIOD_BASE, PERIOD_W, KEY_W, period_of

BALANCE_FIELDS = ("total_cur_assets", "money_cap", "total_cur_liab", "st_borr",
                  "non_cur_liab_due_1y", "total_assets")
CASH_FIELDS = ("n_cashflow_act",)
TYPES = ("comp_type", "report_type")
CONTROL_NAMES = ("cfo_sigma20q", "dw_sigma20q", "due_delta_mean20q",
                 "due_delta_sigma20q", "cfo_mean20q", "log_avg_assets20q")
PARAMETERS = {"signal": "book_wc_cash_realization_noise", "window": 20,
              "full_rank": 7, "residual_df": 13}
AMOUNT_FLOOR = 1e6
MAX_REPORT_AGE_DAYS = 550
EVENT_BATCH = 4096


def _valid_dates(ctx, values):
    values = np.asarray(ctx.date_col(values), dtype=np.int64)
    parsed = pd.to_datetime(values.astype(str), format="%Y%m%d", errors="coerce")
    return np.where((values >= 19900101) & ~pd.isna(parsed), values, 0)


def _report_table(ctx, dataset, fields):
    frame = ctx.dataset(dataset, columns=["stock_code", "ann_date", "f_ann_date",
                                         "end_date", *TYPES, *fields])
    columns = ["code", "period", "pit", *TYPES, *fields]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    end = _valid_dates(ctx, frame["end_date"])
    ann = _valid_dates(ctx, frame["ann_date"])
    fann = _valid_dates(ctx, frame["f_ann_date"])
    pit = np.maximum(ann, fann)
    # Never manufacture an announcement by taking max(announcement, report end).
    keep = np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= (end >= 19900101) & (pit >= end) & (pit > 0)
    if ctx.panel.T:
        keep &= pit <= int(ctx.panel.dates[-1])
    else:
        keep &= False
    data = pd.DataFrame({"code": ctx.code_index(frame["stock_code"].to_numpy()[keep]),
                         "period": period_of(end[keep]), "pit": pit[keep]})
    for field in (*TYPES, *fields):
        data[field] = pd.to_numeric(frame[field], errors="coerce").to_numpy(
            dtype=np.float64)[keep]
    data = data.loc[data.code >= 0]
    # Physical last WHOLE row wins at a tied PIT, including missing/bad types.
    # No update_flag exists in the probed source. Source versions/order are sealed.
    return data.drop_duplicates(["code", "period", "pit"], keep="last")


def _sorted_versions(data, fields):
    data = data.sort_values(["code", "period", "pit"], kind="stable")
    code = data.code.to_numpy(dtype=np.int64)
    period = data.period.to_numpy(dtype=np.int64)
    pit = data.pit.to_numpy(dtype=np.int64)
    return {"keys": (code * PERIOD_W + period) * KEY_W + pit,
            "code": code, "period": period,
            **{f: data[f].to_numpy(dtype=np.float64) for f in (*TYPES, *fields)}}


def _joint_events(balance, cash):
    if balance.empty or cash.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    bp = balance.groupby(["code", "period"], sort=False).pit.min().rename("bp")
    cp = cash.groupby(["code", "period"], sort=False).pit.min().rename("cp")
    common = pd.concat([bp, cp], axis=1, join="inner").reset_index()
    if common.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    common["pit"] = np.maximum(common.bp, common.cp).astype(np.int64)
    common = common.sort_values(["code", "pit", "period"], kind="stable")
    # Presence establishes the frontier BEFORE type/amount eligibility.
    common["q"] = common.groupby("code", sort=False).period.cummax()
    common = common.drop_duplicates(["code", "pit"], keep="last")
    events = pd.concat([balance[["code", "pit"]], cash[["code", "pit"]]],
                       ignore_index=True).drop_duplicates()
    events = events.sort_values(["code", "pit"], kind="stable")
    source_code = common.code.to_numpy(dtype=np.int64)
    keys = source_code * KEY_W + common.pit.to_numpy(dtype=np.int64)
    event_code = events.code.to_numpy(dtype=np.int64)
    query = event_code * KEY_W + events.pit.to_numpy(dtype=np.int64)
    pos = np.searchsorted(keys, query, side="right") - 1
    safe = np.maximum(pos, 0)
    keep = (pos >= 0) & (source_code[safe] == event_code)
    events = events.loc[keep].copy()
    events["q"] = common.q.to_numpy(dtype=np.int64)[safe[keep]]
    return events.reset_index(drop=True)


def _points(vt, events, offsets, fields):
    code = events.code.to_numpy(dtype=np.int64)[:, None]
    period = events.q.to_numpy(dtype=np.int64)[:, None] - np.asarray(offsets)[None, :]
    pit = events.pit.to_numpy(dtype=np.int64)[:, None]
    if not len(vt["keys"]):
        return {f: np.full(period.shape, np.nan) for f in fields}, np.zeros(period.shape, bool)
    pos = np.searchsorted(vt["keys"], (code * PERIOD_W + period) * KEY_W + pit,
                          side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (vt["code"][safe] == code) & (vt["period"][safe] == period)
    ok &= (vt["comp_type"][safe] == 1) & (vt["report_type"][safe] == 1)
    return {f: np.where(ok, vt[f][safe], np.nan) for f in fields}, ok


def _single_cfo(cumulative, q):
    """23 oldest-first YTD positions q-22..q -> 22 singles q-21..q.
    Only Q1 ignores its preceding cumulative position; no calendar compression.
    """
    periods = np.asarray(q)[:, None] - np.arange(21, -1, -1)[None, :]
    return np.where(periods % 4 == 0, cumulative[:, 1:],
                    cumulative[:, 1:] - cumulative[:, :-1])


def _fit_components(y, X, due_delta, assets):
    """All inputs are one complete chronological 20-row design."""
    empty = np.full(1 + len(CONTROL_NAMES), np.nan, dtype=np.float64)
    if y.shape != (20,) or X.shape != (20, 7):
        raise ValueError("Frozen regression requires exactly20 rows and7 columns")
    if not (np.isfinite(y).all() and np.isfinite(X).all()
            and np.isfinite(due_delta).all() and np.isfinite(assets).all()
            and (assets > AMOUNT_FLOOR).all()):
        return empty
    try:
        if np.linalg.matrix_rank(X) != 7:
            return empty
        # Rank is checked FIRST; a degenerate design never becomes false zero.
        if np.all(y == y[0]):
            raw = 0.0
        else:
            beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
            if rank != 7:
                return empty
            residual = y - X @ beta
            sse = np.dot(residual, residual)
            raw = np.sqrt(sse / 13.0)
    except np.linalg.LinAlgError:
        return empty
    if not np.isfinite(raw):
        return empty
    with np.errstate(over="ignore", invalid="ignore"):
        controls = np.array([np.std(X[:, 5], ddof=1), np.std(y, ddof=1),
                             np.mean(due_delta), np.std(due_delta, ddof=1),
                             np.mean(X[:, 5]), np.mean(np.log(assets))], dtype=np.float64)
    # Raw support is never tightened by a diagnostic overflow.
    return np.r_[raw, np.where(np.isfinite(controls), controls, np.nan)]


def _event_components(ctx):
    balance = _report_table(ctx, "stock_balancesheet", BALANCE_FIELDS)
    cash = _report_table(ctx, "stock_cashflow", CASH_FIELDS)
    events = _joint_events(balance, cash)
    result = np.full((len(events), 1 + len(CONTROL_NAMES)), np.nan, dtype=np.float64)
    if events.empty:
        return events, result
    bv = _sorted_versions(balance, BALANCE_FIELDS)
    cv = _sorted_versions(cash, CASH_FIELDS)
    # Bounded blocks avoid materializing20-quarter arrays for all report events.
    for lo in range(0, len(events), EVENT_BATCH):
        ev = events.iloc[lo:lo + EVENT_BATCH]
        # q supplies ONLY BS frontier/type; W/TA use q-21..q-1.
        b, bok = _points(bv, ev, np.r_[0, np.arange(21, 0, -1)], BALANCE_FIELDS)
        c, cok = _points(cv, ev, np.arange(22, -1, -1), CASH_FIELDS)
        bs = {f: x[:, 1:] for f, x in b.items()}
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            W = (bs["total_cur_assets"] - bs["money_cap"] - bs["total_cur_liab"]
                 + bs["st_borr"] + bs["non_cur_liab_due_1y"])
            assets = (bs["total_assets"][:, 1:] + bs["total_assets"][:, :-1]) / 2.0
            y = np.diff(W, axis=1) / assets
            due = np.diff(bs["non_cur_liab_due_1y"], axis=1) / assets
            cfo = _single_cfo(c["n_cashflow_act"], ev.q.to_numpy(dtype=np.int64))
            cflegs = np.stack([cfo[:, :20], cfo[:, 1:21], cfo[:, 2:22]], axis=2)
            cflegs = cflegs / assets[:, :, None]
        complete = bok.all(axis=1) & cok[:, -1]
        complete &= np.isfinite(cfo).all(axis=1)
        for arr in bs.values():
            complete &= np.isfinite(arr).all(axis=1)
        complete &= np.isfinite(assets).all(axis=1) & (assets > AMOUNT_FLOOR).all(axis=1)
        quarters = (ev.q.to_numpy(dtype=np.int64)[:, None] -
                    np.arange(20, 0, -1)[None, :]) % 4
        for i in np.flatnonzero(complete):
            X = np.column_stack([np.ones(20), quarters[i] == 1,
                                 quarters[i] == 2, quarters[i] == 3, cflegs[i]])
            result[lo + i] = _fit_components(y[i], X, due[i], assets[i])
    return events, result


def _epoch_days(yyyymmdd):
    return pd.to_datetime(np.asarray(yyyymmdd).astype(str), format="%Y%m%d",
                          errors="raise").to_numpy(dtype="datetime64[D]").astype(np.int64)


def _map_fresh(ctx, events, values):
    empty = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    if events.empty or not ctx.panel.T:
        return empty
    pit = events.pit.to_numpy(dtype=np.int64)
    effective = np.asarray(ctx.next_trading_day(pit), dtype=np.int64)
    keep = effective > pit
    if not keep.any():
        return empty
    code = events.code.to_numpy(dtype=np.int64)[keep]
    q = events.q.to_numpy(dtype=np.int64)[keep]
    end = (PERIOD_BASE + q // 4) * 10000 + np.array([331, 630, 930, 1231])[q % 4]
    # Sorted code/PIT events stay sorted after nextTD; later weekend events win.
    state = ctx.panel.asof(code, effective[keep], np.asarray(values)[keep], dtype=np.float64)
    report_state = ctx.panel.asof(code, effective[keep], _epoch_days(end), dtype=np.float64)
    age = _epoch_days(ctx.panel.dates)[:, None] - report_state
    valid = (age >= 0) & (age <= MAX_REPORT_AGE_DAYS) & np.isfinite(state)
    valid &= np.asarray(ctx.universe, dtype=bool) & np.asarray(ctx.traded(), dtype=bool)
    return np.where(valid, state, np.nan)


def control_components(ctx):
    """Six preregistered controls, raw values, same event/quarter/support/mask.
    No direction transform and no choice among control subsets.
    """
    events, values = _event_components(ctx)
    return {name: _map_fresh(ctx, events, values[:, i + 1])
            for i, name in enumerate(CONTROL_NAMES)}


def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only the unique frozen R8E definition is permitted")
    def compute(ctx):
        events, values = _event_components(ctx)
        return _map_fresh(ctx, events, values[:, 0])
    return compute
