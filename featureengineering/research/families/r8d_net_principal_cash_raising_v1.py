"""R8 D: explicit principal cash raised net of debt repayment / lag4 assets.
Weak negative prior; excludes repurchases/dividends/interest, not complete XFIN.
Gross single quarters nonnegative; net disbursement retained; strict nextTD PIT.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from fea.deriv import PERIOD_BASE, PERIOD_W, KEY_W, period_of

AMOUNT_FLOOR = 1e6
MAX_REPORT_AGE_DAYS = 550
CASH_FIELDS = ("c_recp_cap_contrib", "c_recp_borrow", "proc_issue_bonds", "c_prepay_amt_borr")
CONTROL_CASH_FIELDS = ("n_cash_flows_fnc_act", "c_pay_dist_dpcp_int_exp", "oth_cashpay_ral_fnc_act", "n_cashflow_act", "c_pay_acq_const_fiolta")
CONTROL_BALANCE_FIELDS = ("money_cap",)
BALANCE_FIELDS = ("total_assets",)
PARAMETERS = {"signal": "net_principal_cash_raising", "window": 4, "asset_lag_quarters": 4}

def _report_table(ctx, dataset, fields):
    frame = ctx.dataset(dataset, columns=["stock_code", "ann_date", "f_ann_date", "end_date", "comp_type", *fields])
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["code", "period", "pit", "comp_type", *fields])
    end = ctx.date_col(frame["end_date"])
    ann = ctx.date_col(frame["ann_date"])
    fann = ctx.date_col(frame["f_ann_date"])
    ann = np.where(ann >= 19900101, ann, 0)
    fann = np.where(fann >= 19900101, fann, 0)
    pit = np.maximum(ann, fann)
    keep = np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= (end >= 19900101) & (pit >= end) & (pit > 0)
    data = pd.DataFrame({"code": ctx.code_index(frame["stock_code"].to_numpy()[keep]),
                         "period": period_of(end[keep]), "pit": pit[keep]})
    data["comp_type"] = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy(dtype=np.float64)[keep]
    for field in fields:
        data[field] = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)[keep]
    data = data.loc[data.code >= 0]
    # Whole report row wins: a newly missing component must not reuse an older value.
    return data.drop_duplicates(["code", "period", "pit"], keep="last")

def _sorted_versions(data, fields):
    data = data.sort_values(["code", "period", "pit"], kind="stable")
    code = data.code.to_numpy(dtype=np.int64)
    period = data.period.to_numpy(dtype=np.int64)
    pit = data.pit.to_numpy(dtype=np.int64)
    return {"keys": (code * PERIOD_W + period) * KEY_W + pit, "code": code,
            "period": period, **{f: data[f].to_numpy(dtype=np.float64) for f in (*fields, "comp_type")}}

def _lookup(vt, code, period, pit, field):
    if len(vt["keys"]) == 0:
        return np.full(len(code), np.nan, dtype=np.float64)
    query = (code * PERIOD_W + period) * KEY_W + pit
    pos = np.searchsorted(vt["keys"], query, side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (vt["code"][safe] == code) & (vt["period"][safe] == period)
    return np.where(ok, vt[field][safe], np.nan)

def _point_window(vt, events, field, width=5, offset=0):
    code = events.code.to_numpy(dtype=np.int64)
    q = events.q.to_numpy(dtype=np.int64)
    pit = events.pit.to_numpy(dtype=np.int64)
    return np.column_stack([_lookup(vt, code, q-offset-j, pit, field) for j in range(width)])

def _joint_events(cash, balance):
    if cash.empty or balance.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    ia = cash.groupby(["code", "period"], sort=False).pit.min().rename("cash_pit")
    ca = balance.groupby(["code", "period"], sort=False).pit.min().rename("balance_pit")
    common = pd.concat([ia, ca], axis=1, join="inner").reset_index()
    if common.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    common["pit"] = np.maximum(common.cash_pit, common.balance_pit).astype(np.int64)
    common = common.sort_values(["code", "pit", "period"], kind="stable")
    common["q"] = common.groupby("code", sort=False).period.cummax()
    common = common.drop_duplicates(["code", "pit"], keep="last")
    events = pd.concat([cash[["code", "pit"]], balance[["code", "pit"]]], ignore_index=True)
    events = events.drop_duplicates().sort_values(["code", "pit"], kind="stable")
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


def _epoch_days(yyyymmdd):
    parsed = pd.to_datetime(np.asarray(yyyymmdd).astype(str), format="%Y%m%d", errors="raise")
    return parsed.to_numpy(dtype="datetime64[D]").astype(np.int64)

def _map_fresh(ctx, events, values):
    if len(events) == 0:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    code = events.code.to_numpy(dtype=np.int64)
    pit = events.pit.to_numpy(dtype=np.int64)
    q = events.q.to_numpy(dtype=np.int64)
    end = (PERIOD_BASE + q // 4) * 10000 + np.array([331,630,930,1231])[q % 4]
    report_days = _epoch_days(end).astype(np.float64)
    effective = np.asarray(ctx.next_trading_day(pit), dtype=np.int64)
    keep = effective > pit
    state = ctx.panel.asof(code[keep], effective[keep], np.asarray(values, dtype=np.float64)[keep], dtype=np.float64)
    report_state = ctx.panel.asof(code[keep], effective[keep], report_days[keep], dtype=np.float64)
    age = _epoch_days(ctx.panel.dates)[:, None] - report_state
    # Calendar age advances even when no new publication event arrives.
    return np.where((age >= 0) & (age <= MAX_REPORT_AGE_DAYS) & np.isfinite(state), state, np.nan)



def _single_window(vt, events, field, width, offset=0):
    cumulative = _point_window(vt, events, field, width=width+1, offset=offset)
    q = events.q.to_numpy(dtype=np.int64)
    first_quarter = ((q[:, None] - offset - np.arange(width)[None, :]) % 4) == 0
    # Q1 does not require previous fiscal-year Q4; other quarters require both.
    return np.where(first_quarter, cumulative[:, :-1],
                    cumulative[:, :-1] - cumulative[:, 1:])


def _financing(gross_single_quarters, lag_assets, general):
    """Shape (events, four gross flow fields, four fixed calendar quarters)."""
    valid = np.isfinite(gross_single_quarters).all(axis=(1, 2))
    valid &= (gross_single_quarters >= 0).all(axis=(1, 2))
    valid &= np.isfinite(lag_assets) & (lag_assets > AMOUNT_FLOOR) & general
    with np.errstate(all="ignore"):
        gross = np.sum(gross_single_quarters, axis=2)
        raw = (gross[:, 0] + gross[:, 1] + gross[:, 2] - gross[:, 3]) / lag_assets
        equity_leg = gross[:, 0] / lag_assets
        debt_leg = (gross[:, 1] + gross[:, 2] - gross[:, 3]) / lag_assets
    valid &= np.isfinite(raw) & np.isfinite(equity_leg) & np.isfinite(debt_leg)
    return tuple(np.where(valid, v, np.nan) for v in (raw, equity_leg, debt_leg))

def _event_components(ctx, include_controls=False):
    cash_fields = CASH_FIELDS + (CONTROL_CASH_FIELDS if include_controls else ())
    balance_fields = BALANCE_FIELDS + (CONTROL_BALANCE_FIELDS if include_controls else ())
    cash = _report_table(ctx, "stock_cashflow", cash_fields)
    balance = _report_table(ctx, "stock_balancesheet", balance_fields)
    events = _joint_events(cash, balance)
    if events.empty:
        empty = np.empty(0, dtype=np.float64)
        return events, (empty, empty, empty), {}
    cv = _sorted_versions(cash, cash_fields)
    bv = _sorted_versions(balance, balance_fields)
    general = (_point_window(cv, events, "comp_type", width=1)[:, 0] == 1)
    general &= (_point_window(bv, events, "comp_type", width=1)[:, 0] == 1)
    assets = _point_window(bv, events, "total_assets", width=1, offset=4)[:, 0]
    gross = np.stack([_single_window(cv, events, f, width=4) for f in CASH_FIELDS], axis=1)
    components = _financing(gross, assets, general)
    controls = {}
    if include_controls:
        raw = components[0]
        for f in CONTROL_CASH_FIELDS:
            singles = _single_window(cv, events, f, width=4)
            with np.errstate(all="ignore"):
                v = np.sum(singles, axis=1) / assets
            controls[f+"_ttm_per_lag_assets"] = np.where(
                np.isfinite(raw) & np.isfinite(singles).all(axis=1) & np.isfinite(v), v, np.nan)
        cash_now = _point_window(bv, events, "money_cap", width=1)[:, 0]
        cash_prev = _point_window(bv, events, "money_cap", width=1, offset=4)[:, 0]
        with np.errstate(all="ignore"):
            change = (cash_now - cash_prev) / assets
        controls["money_cap_change_per_lag_assets"] = np.where(
            np.isfinite(raw) & np.isfinite(cash_now) & np.isfinite(cash_prev)
            & np.isfinite(change), change, np.nan)
    return events, components, controls

def control_components(ctx):
    """Return separately diagnosable equity E and debt D plus current controls.
    raw=E+D; a joint original-value model with both legs and raw is unidentifiable.
    Optional-control missingness is reported; it never tightens raw eligibility.
    Old net-borrowing/share-issuance/R6 overlap require their original frozen
    definitions outside this module, then intersection with this raw mask.
    """
    events, (_, equity, debt), extra = _event_components(ctx, include_controls=True)
    panels = {"E": _map_fresh(ctx, events, equity), "D": _map_fresh(ctx, events, debt)}
    for name, values in extra.items():
        panels[name] = _map_fresh(ctx, events, values)
    return panels

def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only the single frozen R8D definition is permitted")
    def compute(ctx):
        events, (raw, _, _), _ = _event_components(ctx)
        return _map_fresh(ctx, events, raw)
    return compute
