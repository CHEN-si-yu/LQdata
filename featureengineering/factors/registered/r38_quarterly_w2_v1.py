"""R38 quarterly wave2: working-capital and inventory-turnover seasonal deviations.

The PIT/quarter helpers below are a production-local copy of the frozen
research contract (no dependency on ``work/``): first disclosed whole row,
comp_type=1/report_type=1, PIT=max(ann,f_ann,end), strict next exchange
session, exact single-quarter YTD differences, and a 550-calendar-day report
age cap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fea.deriv import KEY_W, PERIOD_BASE, PERIOD_W, period_of

NAMES = (
    "qw2_net_working_capital_seasonal_dev_v1",
    "qw2_inventory_turnover_seasonal_dev_v1",
)
MIN = 1_000_000.0
MAX_AGE_DAYS = 550
COMPANY_TYPE = 1
REPORT_TYPE = 1


@dataclass
class Table:
    dataset: str
    fields: tuple[str, ...]
    code: np.ndarray
    period: np.ndarray
    pit: np.ndarray
    values: dict[str, np.ndarray]
    keys: np.ndarray


def _valid_dates(values) -> np.ndarray:
    raw = pd.to_datetime(pd.Series(values), errors="coerce")
    ints = raw.dt.strftime("%Y%m%d").astype("Int64")
    out = ints.to_numpy(dtype=np.float64, na_value=np.nan)
    ok = np.isfinite(out) & (out >= 19900101)
    return np.where(ok, out, 0).astype(np.int64)


def _load(ctx, dataset: str, fields: tuple[str, ...]) -> Table:
    empty_i = np.empty(0, dtype=np.int64)
    empty_t = Table(dataset, fields, empty_i, empty_i, empty_i,
                    {f: np.empty(0, dtype=np.float64) for f in fields}, empty_i)
    if not ctx.panel.T:
        return empty_t
    lo = max(PERIOD_BASE, int(ctx.panel.dates[0]) // 10000 - 8)
    hi = int(ctx.panel.dates[-1]) // 10000
    columns = ["stock_code", "ann_date", "end_date",
               "f_ann_date", "comp_type", "report_type", *fields]
    frame = ctx.dataset(dataset, columns=columns, years=(lo, hi))
    if frame is None or frame.empty:
        return empty_t
    end = _valid_dates(frame["end_date"])
    ann = _valid_dates(frame["ann_date"])
    fann = _valid_dates(frame["f_ann_date"])
    comp = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy(np.float64)
    report = pd.to_numeric(frame["report_type"], errors="coerce").to_numpy(np.float64)
    pit = np.maximum(np.maximum(ann, fann), end)
    known = (ann >= 19900101) | (fann >= 19900101)
    keep = np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= (end >= 19900101) & (pit >= end) & (pit > 0) & known
    keep &= (comp == COMPANY_TYPE) & (report == REPORT_TYPE)
    keep &= pit <= int(ctx.panel.dates[-1])
    if not keep.any():
        return empty_t
    codes = ctx.code_index(frame["stock_code"].to_numpy()[keep])
    data = pd.DataFrame({"code": codes, "period": period_of(end[keep]), "pit": pit[keep]})
    for field in fields:
        data[field] = pd.to_numeric(frame[field], errors="coerce").to_numpy(np.float64)[keep]
    data = data.loc[data.code >= 0]
    if data.empty:
        return empty_t
    data = data.drop_duplicates(["code", "period", "pit"], keep="last")
    data = data.sort_values(["code", "period", "pit"], kind="stable")
    data = data.drop_duplicates(["code", "period"], keep="first").reset_index(drop=True)
    code = data.code.to_numpy(np.int64)
    period = data.period.to_numpy(np.int64)
    pit = data.pit.to_numpy(np.int64)
    keys = (code * PERIOD_W + period) * KEY_W + pit
    values = {f: data[f].to_numpy(np.float64) for f in fields}
    return Table(dataset, fields, code, period, pit, values, keys)


def _frontier(table: Table) -> pd.DataFrame:
    if not len(table.code):
        return pd.DataFrame(columns=["code", "pit", "q"])
    state = pd.DataFrame({"code": table.code, "period": table.period, "pit": table.pit})
    state = state.sort_values(["code", "pit", "period"], kind="stable")
    state["q"] = state.groupby("code", sort=False).period.cummax()
    return state.drop_duplicates(["code", "pit"], keep="last")[
        ["code", "pit", "q"]].reset_index(drop=True)


def _frontier_at(frontiers: pd.DataFrame, code: np.ndarray, pit: np.ndarray) -> np.ndarray:
    if frontiers.empty:
        return np.full(len(code), -1, dtype=np.int64)
    keys = frontiers.code.to_numpy(np.int64) * KEY_W + frontiers.pit.to_numpy(np.int64)
    query = np.asarray(code, np.int64) * KEY_W + np.asarray(pit, np.int64)
    pos = np.searchsorted(keys, query, side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (frontiers.code.to_numpy(np.int64)[safe] == np.asarray(code))
    return np.where(ok, frontiers.q.to_numpy(np.int64)[safe], -1)


def _joint_events(tables: list[Table]) -> pd.DataFrame:
    pieces = [pd.DataFrame({"code": t.code, "pit": t.pit}) for t in tables if len(t.code)]
    if not pieces:
        return pd.DataFrame(columns=["code", "pit", "q"])
    events = pd.concat(pieces, ignore_index=True).drop_duplicates()
    events = events.sort_values(["code", "pit"], kind="stable").reset_index(drop=True)
    code = events.code.to_numpy(np.int64)
    pit = events.pit.to_numpy(np.int64)
    q = None
    ok = np.ones(len(events), dtype=bool)
    for table in tables:
        qt = _frontier_at(_frontier(table), code, pit)
        if q is None:
            q = qt
        else:
            ok &= qt == q
        ok &= qt >= 0
    events["q"] = np.where(ok & (q >= 0), q, -1)
    return events


def _lookup(table: Table, code, period, cap, field: str) -> np.ndarray:
    code, period, cap = np.broadcast_arrays(
        np.asarray(code, np.int64), np.asarray(period, np.int64), np.asarray(cap, np.int64))
    if not len(table.keys):
        return np.full(code.shape, np.nan, dtype=np.float64)
    query = (code * PERIOD_W + period) * KEY_W + cap
    pos = np.searchsorted(table.keys, query, side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (table.code[safe] == code) & (table.period[safe] == period)
    return np.where(ok, table.values[field][safe], np.nan)


def _sq(table: Table, code, period, cap, field: str) -> np.ndarray:
    code, period, cap = np.broadcast_arrays(
        np.asarray(code, np.int64), np.asarray(period, np.int64), np.asarray(cap, np.int64))
    current = _lookup(table, code, period, cap, field)
    phase = period % 4
    previous = _lookup(table, code, period - 1, cap, field)
    needed = np.where(phase == 0, True, np.isfinite(previous))
    result = np.where(phase == 0, current, current - previous)
    return np.where(np.isfinite(current) & needed & np.isfinite(result), result, np.nan)


def _period_end(period: np.ndarray) -> np.ndarray:
    period = np.asarray(period, np.int64)
    md = np.array([331, 630, 930, 1231], dtype=np.int64)[period % 4]
    return (PERIOD_BASE + period // 4) * 10000 + md


def _epoch_days(yyyymmdd: np.ndarray) -> np.ndarray:
    return pd.to_datetime(
        np.asarray(yyyymmdd).astype(str), format="%Y%m%d", errors="raise"
    ).to_numpy(dtype="datetime64[D]").astype(np.int64)


def _map(ctx, events: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    out = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    if events.empty or not ctx.panel.T:
        return out
    pit = events.pit.to_numpy(np.int64)
    q = events.q.to_numpy(np.int64)
    effective = np.asarray(ctx.next_trading_day(pit), dtype=np.int64)
    end = _period_end(np.maximum(q, 0))
    end = np.where(q >= 0, end, effective)
    usable = (effective > pit) & (effective <= int(ctx.panel.dates[-1]))
    if not usable.any():
        return out
    mapped = pd.DataFrame({
        "code": events.code.to_numpy(np.int64)[usable],
        "effective": effective[usable],
        "pit": pit[usable],
        "end_epoch": _epoch_days(end[usable]),
        "value": np.asarray(values, np.float64)[usable],
    })
    mapped = mapped.sort_values(["code", "effective", "pit"], kind="stable")
    mapped = mapped.drop_duplicates(["code", "effective"], keep="last")
    code = mapped.code.to_numpy(np.int64)
    effective = mapped.effective.to_numpy(np.int64)
    state = ctx.panel.asof(code, effective, mapped.value.to_numpy(np.float64), dtype=np.float64)
    report_end = ctx.panel.asof(code, effective, mapped.end_epoch.to_numpy(np.float64),
                                dtype=np.float64)
    dates = _epoch_days(ctx.panel.dates)
    age = dates[:, None] - report_end
    valid = np.isfinite(state) & np.isfinite(age) & (age >= 0) & (age <= MAX_AGE_DAYS)
    valid &= np.asarray(ctx.universe, dtype=bool)
    return np.where(valid, state, np.nan)


def _nwc(ctx):
    bs = _load(ctx, "stock_balancesheet", (
        "inventories", "accounts_receiv", "acct_payable", "notes_payable",
        "total_assets",
    ))
    ev = _joint_events([bs])
    if ev.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    code = ev.code.to_numpy(np.int64)
    q = ev.q.to_numpy(np.int64)
    cap = ev.pit.to_numpy(np.int64)

    def ratio(lag):
        qq = q - lag
        inv = _lookup(bs, code, qq, cap, "inventories")
        ar = _lookup(bs, code, qq, cap, "accounts_receiv")
        ap = _lookup(bs, code, qq, cap, "acct_payable")
        npay = _lookup(bs, code, qq, cap, "notes_payable")
        ta = _lookup(bs, code, qq, cap, "total_assets")
        tp = _lookup(bs, code, qq - 1, cap, "total_assets")
        avg = (ta + tp) / 2.0
        good = (np.isfinite(inv) & np.isfinite(ar) & np.isfinite(ap)
                & np.isfinite(npay) & np.isfinite(avg) & (avg > MIN))
        nwc = inv + ar - ap - npay
        return np.where(good, nwc / np.where(good, avg, 1.0), np.nan)

    r0, r1, r2 = ratio(0), ratio(4), ratio(8)
    mean = (r1 + r2) / 2.0
    val = np.where(np.isfinite(r0) & np.isfinite(mean), r0 - mean, np.nan)
    return _map(ctx, ev, val)


def _inventory_turnover(ctx):
    inc = _load(ctx, "stock_income", ("oper_cost",))
    bs = _load(ctx, "stock_balancesheet", ("inventories",))
    ev = _joint_events([inc, bs])
    if ev.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    code = ev.code.to_numpy(np.int64)
    q = ev.q.to_numpy(np.int64)
    cap = ev.pit.to_numpy(np.int64)

    def ratio(lag):
        qq = q - lag
        cost = _sq(inc, code, qq, cap, "oper_cost")
        inv0 = _lookup(bs, code, qq, cap, "inventories")
        inv1 = _lookup(bs, code, qq - 1, cap, "inventories")
        avg = (inv0 + inv1) / 2.0
        # Card condition: both inventory balances must clear the floor in
        # every same-quarter slot (not only their average).
        good = (np.isfinite(cost) & (cost > MIN) & np.isfinite(inv0)
                & (inv0 > MIN) & np.isfinite(inv1) & (inv1 > MIN) & (avg > MIN))
        with np.errstate(divide="ignore", invalid="ignore"):
            turn = np.where(good, cost / np.where(good, avg, 1.0), np.nan)
        return np.log(np.where(turn > 0, turn, np.nan))

    r0, r1, r2 = ratio(0), ratio(4), ratio(8)
    mean = (r1 + r2) / 2.0
    val = np.where(np.isfinite(r0) & np.isfinite(mean), r0 - mean, np.nan)
    return _map(ctx, ev, val)


def build(parameters):
    variant = (parameters or {}).get("variant")
    if variant not in NAMES:
        raise KeyError(f"unknown R38 quarterly wave2 variant: {variant!r}")

    def compute(ctx):
        if variant == "qw2_net_working_capital_seasonal_dev_v1":
            return _nwc(ctx)
        return _inventory_turnover(ctx)

    return compute
