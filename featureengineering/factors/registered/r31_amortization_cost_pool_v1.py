"""Annual amortization share of the disclosed long-asset noncash charge pool."""
from __future__ import annotations

import numpy as np
import pandas as pd

FIELDS = ("depr_fa_coga_dpba", "amort_intang_assets", "lt_amort_deferred_exp")
REPORT_AGE_DAYS = 550
MIN_TOTAL = 1_000_000.0
PERIOD_BASE = 200001
PERIOD_W = 1000
KEY_W = 100000000


def _date_values(ctx, series):
    raw = np.asarray(ctx.date_col(series), dtype=np.int64)
    parsed = pd.to_datetime(raw.astype(str), format="%Y%m%d", errors="coerce")
    return np.where((raw >= 19900101) & ~np.asarray(pd.isna(parsed)), raw, 0)


def _load(ctx):
    if not ctx.panel.T:
        return pd.DataFrame(columns=["code", "period", "pit", *FIELDS])
    lo = max(2000, int(ctx.panel.dates[0]) // 10000 - 2)
    hi = int(ctx.panel.dates[-1]) // 10000
    frame = ctx.dataset("stock_cashflow", columns=[
        "stock_code", "ann_date", "f_ann_date", "end_date", "comp_type", "report_type", *FIELDS
    ], years=(lo, hi))
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["code", "period", "pit", *FIELDS])
    end = _date_values(ctx, frame.end_date)
    ann = _date_values(ctx, frame.ann_date)
    fann = _date_values(ctx, frame.f_ann_date)
    pit = np.maximum(np.maximum(ann, fann), end)
    comp = pd.to_numeric(frame.comp_type, errors="coerce").to_numpy(dtype=np.float64)
    report = pd.to_numeric(frame.report_type, errors="coerce").to_numpy(dtype=np.float64)
    keep = (end >= 19900101) & (end % 10000 == 1231)
    keep &= ((ann >= 19900101) | (fann >= 19900101)) & (pit >= end)
    keep &= (comp == 1) & (report == 1)
    keep &= pit <= int(ctx.panel.dates[-1])
    codes = ctx.code_index(frame.stock_code.to_numpy()[keep])
    data = pd.DataFrame({"code": codes, "period": end[keep] // 10000,
                         "pit": pit[keep]})
    for field in FIELDS:
        data[field] = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)[keep]
    data = data.loc[data.code >= 0]
    # PIT cutoff is already applied. First resolve same-PIT physical-last rows,
    # then retain the earliest whole row per stock/FY. A later restatement is
    # not a new factor event and may neither overwrite nor repair a first row.
    return _first_vintages(data)


def _first_vintages(data):
    if data.empty:
        return data.copy()
    physical_last = data.drop_duplicates(["code", "period", "pit"], keep="last")
    ordered = physical_last.sort_values(["code", "period", "pit"], kind="stable")
    return ordered.drop_duplicates(["code", "period"], keep="first").reset_index(drop=True)


def _frontier(data):
    if data.empty:
        return pd.DataFrame(columns=["code", "pit", "fy"])
    state = data[["code", "period", "pit"]].sort_values(
        ["code", "pit", "period"], kind="stable").copy()
    state["fy"] = state.groupby("code", sort=False).period.cummax()
    return state.drop_duplicates(["code", "pit"], keep="last")[
        ["code", "pit", "fy"]].reset_index(drop=True)


def _pit_semantics_selfcheck():
    sample = pd.DataFrame([
        {"code":0,"period":2020,"pit":20210301,"depr_fa_coga_dpba":np.nan,
         "amort_intang_assets":2.,"lt_amort_deferred_exp":3.},
        {"code":0,"period":2020,"pit":20210401,"depr_fa_coga_dpba":20.,
         "amort_intang_assets":2.,"lt_amort_deferred_exp":3.},
        {"code":1,"period":2020,"pit":20210301,"depr_fa_coga_dpba":20.,
         "amort_intang_assets":2.,"lt_amort_deferred_exp":3.},
        {"code":1,"period":2020,"pit":20210401,"depr_fa_coga_dpba":100.,
         "amort_intang_assets":2.,"lt_amort_deferred_exp":3.},
        {"code":2,"period":2021,"pit":20220301,"depr_fa_coga_dpba":20.,
         "amort_intang_assets":2.,"lt_amort_deferred_exp":3.},
        {"code":2,"period":2020,"pit":20220401,"depr_fa_coga_dpba":10.,
         "amort_intang_assets":1.,"lt_amort_deferred_exp":1.},
        {"code":3,"period":2020,"pit":20210301,"depr_fa_coga_dpba":20.,
         "amort_intang_assets":2.,"lt_amort_deferred_exp":3.},
        {"code":3,"period":2020,"pit":20210301,"depr_fa_coga_dpba":30.,
         "amort_intang_assets":2.,"lt_amort_deferred_exp":3.},
    ])
    first = _first_vintages(sample)
    r0=first[(first.code==0)&(first.period==2020)].iloc[0]
    r1=first[(first.code==1)&(first.period==2020)].iloc[0]
    r3=first[(first.code==3)&(first.period==2020)].iloc[0]
    assert not np.isfinite(r0.depr_fa_coga_dpba)
    assert r1.depr_fa_coga_dpba == 20.
    assert r3.depr_fa_coga_dpba == 30.
    ev=_frontier(first[first.code==2])
    assert ev.fy.tolist()==[2021,2021]
    return {"invalid_first_not_repaired":True,"valid_first_not_revised":True,
            "late_old_fy_does_not_regress_frontier":True,
            "same_pit_physical_last_whole_row":True}


def _calculate(data, events):
    if data.empty or events.empty:
        return np.full(len(events), np.nan, dtype=np.float64)
    versions = data.sort_values(["code", "period", "pit"], kind="stable")
    code = versions.code.to_numpy(dtype=np.int64)
    period = versions.period.to_numpy(dtype=np.int64)
    pit = versions.pit.to_numpy(dtype=np.int64)
    keys = (code * PERIOD_W + period) * KEY_W + pit
    qcode = events.code.to_numpy(dtype=np.int64)
    qperiod = events.fy.to_numpy(dtype=np.int64)
    qpit = events.pit.to_numpy(dtype=np.int64)
    query = (qcode * PERIOD_W + qperiod) * KEY_W + qpit
    pos = np.searchsorted(keys, query, side="right") - 1
    valid = pos >= 0
    safe = np.maximum(pos, 0)
    valid &= (code[safe] == qcode) & (period[safe] == qperiod)
    legs = np.column_stack([versions[f].to_numpy(dtype=np.float64)[safe] for f in FIELDS])
    valid &= np.isfinite(legs).all(axis=1) & (legs >= 0).all(axis=1)
    total = legs.sum(axis=1)
    valid &= total > MIN_TOTAL
    out = np.full(len(events), np.nan, dtype=np.float64)
    out[valid] = (legs[valid, 1] + legs[valid, 2]) / total[valid]
    return out


def _map(ctx, events, values):
    out = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    if events.empty or not ctx.panel.T:
        return out
    pit = events.pit.to_numpy(dtype=np.int64)
    effective = np.asarray(ctx.next_trading_day(pit), dtype=np.int64)
    fy = events.fy.to_numpy(dtype=np.int64)
    report_end = fy * 10000 + 1231
    use = (effective > pit) & (effective <= int(ctx.panel.dates[-1]))
    if not use.any():
        return out
    mapped = pd.DataFrame({"code": events.code.to_numpy(dtype=np.int64)[use],
                           "effective": effective[use], "pit": pit[use],
                           "end": report_end[use], "value": np.asarray(values)[use]})
    mapped = mapped.sort_values(["code", "effective", "pit"], kind="stable")
    mapped = mapped.drop_duplicates(["code", "effective"], keep="last")
    code = mapped.code.to_numpy(dtype=np.int64)
    effective = mapped.effective.to_numpy(dtype=np.int64)
    state = ctx.panel.asof(code, effective, mapped.value.to_numpy(dtype=np.float64), dtype=np.float64)
    end = ctx.panel.asof(code, effective, mapped.end.to_numpy(dtype=np.float64), dtype=np.float64)
    dates = pd.to_datetime(ctx.panel.dates.astype(str), format="%Y%m%d").to_numpy(dtype="datetime64[D]")
    end_int = np.where(np.isfinite(end), end, 0).astype(np.int64)
    ends = pd.to_datetime(end_int.ravel().astype(str), format="%Y%m%d",
                           errors="coerce").to_numpy(dtype="datetime64[D]").reshape(end.shape)
    age = (dates[:, None] - ends).astype("timedelta64[D]").astype(np.float64)
    valid = np.isfinite(state) & np.isfinite(age) & (age >= 0) & (age <= REPORT_AGE_DAYS)
    # Engine production universe is listing/delisting window plus ST; no traded mask here.
    valid &= np.asarray(ctx.universe, dtype=bool)
    return np.where(valid, state, np.nan)


def build(parameters=None):
    if parameters not in (None, {}):
        raise ValueError("Frozen draft takes no parameters")

    def compute(ctx):
        data = _load(ctx)
        if data.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        events = _frontier(data)
        values = _calculate(data, events)
        return _map(ctx, events, values)

    return compute
