"""Frozen R22 first-vintage disposal proceeds intensity candidate.

This builder intentionally reads the two raw statement tables. The R22 contract
selects each report period's earliest PIT-valid standard row and never lets a
later restatement replace that first vintage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PARAMETERS = {"signal": "disposal_proceeds_intensity_yoy_v1"}
SOURCE_START_YEAR = 2010
CF_FIELD = "n_recp_disp_fiolta"
INCOME_FIELD = "revenue"


def _period(end_date: np.ndarray) -> np.ndarray:
    md = end_date % 10000
    quarter = np.select((md == 331, md == 630, md == 930, md == 1231),
                        (0, 1, 2, 3), default=-1)
    return np.where(quarter >= 0, (end_date // 10000) * 4 + quarter, -1)


def _first_vintage(ctx, dataset: str, field: str, cutoff: int):
    """Return per-code period maps of (effective day, source value)."""
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    codes = np.asarray(ctx.panel.codes)
    empty = [dict() for _ in codes]
    if not dates.size or not codes.size:
        return empty

    frame = ctx.dataset(
        dataset,
        columns=["stock_code", "ann_date", "f_ann_date", "end_date",
                 "comp_type", "report_type", field],
        years=(SOURCE_START_YEAR, int(cutoff) // 10000),
    )
    if frame is None or frame.empty:
        return empty

    end = ctx.date_col(frame["end_date"]).astype(np.int64)
    ann = ctx.date_col(frame["ann_date"]).astype(np.int64)
    fann = ctx.date_col(frame["f_ann_date"]).astype(np.int64)
    valid_ann, valid_fann = ann >= 19900101, fann >= 19900101
    pit = np.maximum(np.maximum(np.where(valid_ann, ann, 0),
                                np.where(valid_fann, fann, 0)), end)
    period = _period(end)
    comp = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy()
    report = pd.to_numeric(frame["report_type"], errors="coerce").to_numpy()
    keep = (period >= 0) & (end >= 19900101) & (pit >= end)
    keep &= (valid_ann | valid_fann) & (pit <= int(cutoff))
    keep &= (comp == 1) & (report == 1)
    if not keep.any():
        return empty

    codes_raw = frame["stock_code"].astype(str).to_numpy()[keep]
    code_idx = ctx.code_index(codes_raw)
    selected = code_idx >= 0
    if not selected.any():
        return empty
    vals = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)[keep]
    sub = pd.DataFrame({
        "code": code_idx[selected].astype(np.int32),
        "period": period[keep][selected].astype(np.int64),
        "pit": pit[keep][selected].astype(np.int64),
        "value": vals[selected],
    })
    # R22 concatenated source-year partitions in ascending order, stable-sorted
    # by code/period/PIT, then retained the first PIT-valid row per period.
    sub = sub.sort_values(["code", "period", "pit"], kind="stable")
    sub = sub.drop_duplicates(["code", "period"], keep="first")
    visible_day = np.asarray(ctx.next_trading_day(sub["pit"].to_numpy(np.int32)), dtype=np.int32)
    sub["effective"] = visible_day
    for row in sub.itertuples(index=False):
        empty[int(row.code)][int(row.period)] = (int(row.effective), float(row.value))
    return empty


def _visible(table, period: int, day: int) -> float:
    item = table.get(int(period))
    return np.nan if item is None or item[0] > day else item[1]


def _latest_common_period(cash, income, day: int):
    common = cash.keys() & income.keys()
    for q in sorted(common, reverse=True):
        if cash[q][0] <= day and income[q][0] <= day:
            return int(q)
    return None


def _ttm(table, field_period: int, day: int) -> float:
    """Strict TTM from YTD values; `field_period` is an integer quarter index."""
    q = int(field_period)
    current = _visible(table, q, day)
    if not np.isfinite(current):
        return np.nan
    phase = q % 4
    if phase == 3:
        return float(current)
    prior_fy = _visible(table, q - phase - 1, day)
    prior_same = _visible(table, q - 4, day)
    if not np.isfinite(prior_fy) or not np.isfinite(prior_same):
        return np.nan
    return float(current + prior_fy - prior_same)


def _candidate_value(cash, income, q: int, day: int) -> float:
    cash_now = _ttm(cash, q, day)
    cash_old = _ttm(cash, q - 4, day)
    revenue_now = _ttm(income, q, day)
    revenue_old = _ttm(income, q - 4, day)
    if not all(np.isfinite(x) for x in (cash_now, cash_old, revenue_now, revenue_old)):
        return np.nan
    if revenue_now <= 1e6 or revenue_old <= 1e6:
        return np.nan
    return float(cash_now / revenue_now - cash_old / revenue_old)


def build(parameters):
    """Build the fixed candidate; metadata and production identity live in JSON."""
    if parameters != PARAMETERS:
        raise ValueError("Frozen R22 disposal_proceeds_intensity_yoy_v1 only")

    def compute(ctx):
        dates = np.asarray(ctx.panel.dates, dtype=np.int32)
        codes = np.asarray(ctx.panel.codes)
        out = np.full((dates.size, codes.size), np.nan, dtype=np.float64)
        if not dates.size or not codes.size:
            return out

        cutoff = int(dates[-1])
        cash = _first_vintage(ctx, "stock_cashflow", CF_FIELD, cutoff)
        income = _first_vintage(ctx, "stock_income", INCOME_FIELD, cutoff)

        for ci, (cash_table, income_table) in enumerate(zip(cash, income)):
            events = {int(dates[0])}
            events.update(eff for eff, _ in cash_table.values()
                          if dates[0] < eff <= dates[-1])
            events.update(eff for eff, _ in income_table.values()
                          if dates[0] < eff <= dates[-1])
            event_days = sorted(events)
            for ei, day in enumerate(event_days):
                start = int(np.searchsorted(dates, day, side="left"))
                stop = (int(np.searchsorted(dates, event_days[ei + 1], side="left"))
                        if ei + 1 < len(event_days) else dates.size)
                if start >= stop:
                    continue
                q = _latest_common_period(cash_table, income_table, day)
                value = np.nan if q is None else _candidate_value(cash_table, income_table, q, day)
                out[start:stop, ci] = value

        return np.where(np.asarray(ctx.traded(), dtype=bool), out, np.nan)

    return compute

