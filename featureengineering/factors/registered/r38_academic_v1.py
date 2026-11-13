"""R38 academic fundamental factors with first-vintage A-share PIT.

Production-local copy of the frozen research contract: comp_type=1 /
report_type=1 whole rows, PIT=max(valid ann_date, f_ann_date, end_date),
earliest PIT per stock/period retained, prefix-max fiscal frontier, strict
next-session mapping, 550-calendar-day age cap, and exact three-slot TTM.
No ``work/`` import and no return/label input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fea.deriv import KEY_W, PERIOD_BASE, PERIOD_W, period_of

NAMES = (
    "acf_roe_stability_5y_v1",
    "acf_rd_capital_assets_fy_v1",
    "acf_cash_operating_profit_assets_ttm_v1",
)
FLOOR = 1_000_000.0
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
    columns = ["stock_code", "ann_date", "end_date", *fields]
    if dataset != "stock_financial_indicator":
        columns += ["f_ann_date", "comp_type", "report_type"]
    frame = ctx.dataset(dataset, columns=columns, years=(lo, hi))
    if frame is None or frame.empty:
        return empty_t
    end = _valid_dates(frame["end_date"])
    ann = _valid_dates(frame["ann_date"])
    if dataset == "stock_financial_indicator":
        pit = np.maximum(ann, end)
        known = ann >= 19900101
        comp_ok = np.ones(len(frame), dtype=bool)
        report_ok = np.ones(len(frame), dtype=bool)
    else:
        fann = _valid_dates(frame["f_ann_date"])
        pit = np.maximum(np.maximum(ann, fann), end)
        known = (ann >= 19900101) | (fann >= 19900101)
        comp = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy(np.float64)
        report = pd.to_numeric(frame["report_type"], errors="coerce").to_numpy(np.float64)
        comp_ok = comp == COMPANY_TYPE
        report_ok = report == REPORT_TYPE
    keep = np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= (end >= 19900101) & (pit >= end) & (pit > 0) & known
    keep &= comp_ok & report_ok
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


def _frontier(table: Table, fiscal_only: bool = False) -> pd.DataFrame:
    if not len(table.code):
        return pd.DataFrame(columns=["code", "pit", "q"])
    mask = np.ones(len(table.code), dtype=bool)
    if fiscal_only:
        mask = (table.period % 4) == 3
    if not mask.any():
        return pd.DataFrame(columns=["code", "pit", "q"])
    state = pd.DataFrame({"code": table.code[mask], "period": table.period[mask],
                          "pit": table.pit[mask]})
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


def _events(tables: list[Table], fiscal_only: bool) -> pd.DataFrame:
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
        qt = _frontier_at(_frontier(table, fiscal_only), code, pit)
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


def _ttm(table: Table, code, period, cap, field: str) -> np.ndarray:
    code, period, cap = np.broadcast_arrays(
        np.asarray(code, np.int64), np.asarray(period, np.int64), np.asarray(cap, np.int64))
    current = _lookup(table, code, period, cap, field)
    phase = period % 4
    prior_fy_q4 = period - (phase + 1)
    prior_same = period - 4
    prior_q4 = _lookup(table, code, prior_fy_q4, cap, field)
    prior_same_v = _lookup(table, code, prior_same, cap, field)
    result = np.where(phase == 3, current, current + prior_q4 - prior_same_v)
    needed = np.isfinite(current)
    needed &= np.where(phase == 3, True,
                       np.isfinite(prior_q4) & np.isfinite(prior_same_v))
    return np.where(needed & np.isfinite(result), result, np.nan)


def _nz(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x), x, 0.0)


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
    state = ctx.panel.asof(code, effective, mapped.value.to_numpy(np.float64))
    report_end = ctx.panel.asof(code, effective,
                                mapped.end_epoch.to_numpy(np.float64))
    dates = _epoch_days(ctx.panel.dates)
    age = dates[:, None] - report_end
    valid = np.isfinite(state) & np.isfinite(age) & (age >= 0) & (age <= MAX_AGE_DAYS)
    valid &= np.asarray(ctx.universe, dtype=bool)
    return np.where(valid, state, np.nan)


def _roe_stability(ctx):
    inc = _load(ctx, "stock_income", ("n_income_attr_p",))
    bs = _load(ctx, "stock_balancesheet", ("total_hldr_eqy_exc_min_int",))
    ev = _events([inc, bs], fiscal_only=True)
    if ev.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    ev = ev.sort_values(["code", "pit"], kind="stable").reset_index(drop=True)
    prev = ev.groupby("code", sort=False).q.shift()
    keep = ev.q.to_numpy(np.int64) >= 0
    keep &= prev.isna().to_numpy() | (ev.q.to_numpy(np.int64) != prev.to_numpy())
    ev = ev.loc[keep].reset_index(drop=True)
    code = ev.code.to_numpy(np.int64)
    q = ev.q.to_numpy(np.int64)
    cap = ev.pit.to_numpy(np.int64)
    roes = []
    good = np.ones(len(ev), dtype=bool)
    for j in range(5):
        per = q - 4 * j
        ni = _lookup(inc, code, per, cap, "n_income_attr_p")
        eq = _lookup(bs, code, per, cap, "total_hldr_eqy_exc_min_int")
        ok = np.isfinite(ni) & np.isfinite(eq) & (eq > FLOOR)
        good &= ok
        roes.append(np.where(ok, ni / np.where(ok, eq, 1.0), np.nan))
    sd = np.std(np.stack(roes, axis=0), axis=0, ddof=0)
    values = np.where(good & np.isfinite(sd), -sd, np.nan)
    return _map(ctx, ev, values)


def _rd_capital(ctx):
    ind = _load(ctx, "stock_financial_indicator", ("rd_exp",))
    bs = _load(ctx, "stock_balancesheet", ("total_assets",))
    ev = _events([ind, bs], fiscal_only=True)
    if ev.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    ev = ev.sort_values(["code", "pit"], kind="stable").reset_index(drop=True)
    prev = ev.groupby("code", sort=False).q.shift()
    keep = ev.q.to_numpy(np.int64) >= 0
    keep &= prev.isna().to_numpy() | (ev.q.to_numpy(np.int64) != prev.to_numpy())
    ev = ev.loc[keep].reset_index(drop=True)
    code = ev.code.to_numpy(np.int64)
    q = ev.q.to_numpy(np.int64)
    cap = ev.pit.to_numpy(np.int64)
    rd = []
    for j in range(5):
        rd.append(_nz(_lookup(ind, code, q - 4 * j, cap, "rd_exp")))
    ta = _lookup(bs, code, q, cap, "total_assets")
    num = rd[0] + 0.8 * rd[1] + 0.6 * rd[2] + 0.4 * rd[3] + 0.2 * rd[4]
    ok = np.isfinite(ta) & (ta > FLOOR)
    values = np.where(ok, num / np.where(ok, ta, 1.0), np.nan)
    return _map(ctx, ev, values)


def _cash_operating_profit(ctx):
    inc = _load(ctx, "stock_income", (
        "revenue", "oper_cost", "sell_exp", "admin_exp", "rd_exp",
    ))
    bs = _load(ctx, "stock_balancesheet", (
        "accounts_receiv", "inventories", "prepayment", "contract_liab",
        "adv_receipts", "defer_inc_non_cur_liab", "accounts_pay",
        "payroll_payable", "taxes_payable", "total_assets",
    ))
    ev = _events([inc, bs], fiscal_only=False)
    if ev.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    mask = ev.q.to_numpy(np.int64) >= 0
    if not mask.any():
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    sub = ev.loc[mask].reset_index(drop=True)
    code = sub.code.to_numpy(np.int64)
    q = sub.q.to_numpy(np.int64)
    cap = sub.pit.to_numpy(np.int64)
    rev = _ttm(inc, code, q, cap, "revenue")
    cost = _ttm(inc, code, q, cap, "oper_cost")
    sell = _nz(_ttm(inc, code, q, cap, "sell_exp"))
    admin = _nz(_ttm(inc, code, q, cap, "admin_exp"))
    rd = _nz(_ttm(inc, code, q, cap, "rd_exp"))

    def d(field):
        cur = _lookup(bs, code, q, cap, field)
        prev = _lookup(bs, code, q - 4, cap, field)
        return _nz(cur) - _nz(prev)

    d_ar = d("accounts_receiv")
    d_inv = d("inventories")
    d_pre = d("prepayment")
    d_defcur = d("contract_liab") + d("adv_receipts")
    d_deflt = d("defer_inc_non_cur_liab")
    d_ap = d("accounts_pay")
    d_acc = d("payroll_payable") + d("taxes_payable")
    ta = _lookup(bs, code, q, cap, "total_assets")
    num = (rev - cost - (sell + admin - rd) - d_ar - d_inv - d_pre
           + d_defcur + d_deflt + d_ap + d_acc)
    ok = np.isfinite(rev) & np.isfinite(cost) & np.isfinite(ta)
    ok &= (rev > FLOOR) & (cost > FLOOR) & (ta > FLOOR)
    values = np.where(ok, num / np.where(ok, ta, 1.0), np.nan)
    values_all = np.full(len(ev), np.nan, dtype=np.float64)
    values_all[mask] = values
    return _map(ctx, ev, values_all)


def build(parameters):
    variant = (parameters or {}).get("variant")
    if variant not in NAMES:
        raise KeyError(f"unknown R38 academic variant: {variant!r}")

    def compute(ctx):
        if variant == "acf_roe_stability_5y_v1":
            return _roe_stability(ctx)
        if variant == "acf_rd_capital_assets_fy_v1":
            return _rd_capital(ctx)
        return _cash_operating_profit(ctx)

    return compute
