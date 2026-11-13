"""Independent PIT builder for the admitted R25 credit-impairment candidate.

No research-work files are imported. Source rows are read directly through the
production context and frozen to the first PIT-valid standard vintage per
report period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


NAME = "annual_credit_impairment_adjustment_assets_v1"
SOURCE_START_YEAR = 2010
MAX_REPORT_AGE_DAYS = 550
MIN_CNY = 1_000_000.0
CF_FIELDS = ("credit_impa_loss",)
BS_FIELDS = ("total_assets",)


def _periods(end: np.ndarray) -> np.ndarray:
    md = end % 10000
    quarter = np.select((md == 331, md == 630, md == 930, md == 1231),
                        (0, 1, 2, 3), default=-1)
    return np.where(quarter >= 0, end // 10000 * 4 + quarter, -1).astype(np.int64)


def _report_table(ctx, dataset: str, fields: tuple[str, ...], cutoff: int):
    """First PIT-valid standard row per code/report-period, with strict next-session visibility."""
    codes = np.asarray(ctx.panel.codes)
    empty = [dict() for _ in codes]
    if not codes.size:
        return empty
    frame = ctx.dataset(
        dataset,
        columns=["stock_code", "ann_date", "f_ann_date", "end_date",
                 "comp_type", "report_type", *fields],
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
    period = _periods(end)
    comp = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy()
    report = pd.to_numeric(frame["report_type"], errors="coerce").to_numpy()
    keep = (period >= 0) & (end >= 19900101) & (pit >= end) & (pit <= int(cutoff))
    keep &= (valid_ann | valid_fann) & (comp == 1) & (report == 1)
    if not keep.any():
        return empty

    code_idx = ctx.code_index(frame["stock_code"].astype(str).to_numpy()[keep])
    chosen = code_idx >= 0
    if not chosen.any():
        return empty
    values = {f: pd.to_numeric(frame[f], errors="coerce").to_numpy(np.float64)[keep][chosen]
              for f in fields}
    rows = pd.DataFrame({
        "code": code_idx[chosen].astype(np.int32),
        "period": period[keep][chosen],
        "pit": pit[keep][chosen],
    })
    for f in fields:
        rows[f] = values[f]
    # The cutoff is applied above before this stable first-vintage selection.
    rows = rows.sort_values(["code", "period", "pit"], kind="stable")
    rows = rows.drop_duplicates(["code", "period"], keep="first")

    # Match the R25 adapter: effective_i = searchsorted(all exchange sessions,
    # PIT, side='right'); rows with no next exchange session are not effective.
    sessions = np.asarray(ctx.cal.days, dtype=np.int32)
    pits = rows["pit"].to_numpy(dtype=np.int32)
    next_i = np.searchsorted(sessions, pits, side="right")
    has_next = next_i < sessions.size
    rows = rows.loc[has_next].copy()
    rows["effective"] = sessions[next_i[has_next]]
    for row in rows.itertuples(index=False):
        vals = {f: float(getattr(row, f)) for f in fields}
        empty[int(row.code)][int(row.period)] = (int(row.effective), int(row.pit), vals)
    return empty


def _latest_annual(table, day: int):
    visible = [q for q, (effective, _, _) in table.items()
               if q % 4 == 3 and effective <= day]
    return max(visible) if visible else None


def _period_end(q: int) -> pd.Timestamp:
    year, quarter = divmod(int(q), 4)
    month, day = ((3, 31), (6, 30), (9, 30), (12, 31))[quarter]
    return pd.Timestamp(year=year, month=month, day=day)


def _age_stop(dates: np.ndarray, q: int) -> int:
    cutoff = int((_period_end(q) + pd.Timedelta(days=MAX_REPORT_AGE_DAYS)).strftime("%Y%m%d"))
    return int(np.searchsorted(dates, cutoff, side="right"))


def _credit_impairment_value(cash, balance, q: int, day: int) -> float:
    cf = cash.get(int(q))
    assets = balance.get(int(q))
    prior_assets = balance.get(int(q) - 4)
    if cf is None or assets is None or prior_assets is None:
        return np.nan
    if max(cf[0], assets[0], prior_assets[0]) > day:
        return np.nan
    credit = cf[2].get("credit_impa_loss", np.nan)
    current = assets[2].get("total_assets", np.nan)
    prior = prior_assets[2].get("total_assets", np.nan)
    if not all(np.isfinite(v) for v in (credit, current, prior)):
        return np.nan
    if current <= MIN_CNY or prior <= MIN_CNY:
        return np.nan
    return float(credit / ((current + prior) / 2.0))


def build(parameters):
    """Build the fixed R25 candidate using {'name': factor_name}."""
    if parameters != {"name": NAME}:
        raise ValueError(f"Expected {{'name': {NAME!r}}}")

    def compute(ctx):
        dates = np.asarray(ctx.panel.dates, dtype=np.int32)
        codes = np.asarray(ctx.panel.codes)
        out = np.full((dates.size, codes.size), np.nan, dtype=np.float64)
        if not dates.size or not codes.size:
            return out
        cutoff = int(dates[-1])
        cash = _report_table(ctx, "stock_cashflow", CF_FIELDS, cutoff)
        balance = _report_table(ctx, "stock_balancesheet", BS_FIELDS, cutoff)

        first_day, last_day = int(dates[0]), int(dates[-1])
        for ci, cash_table in enumerate(cash):
            balance_table = balance[ci]
            relevant = (cash_table, balance_table)
            events = {first_day}
            events.update(effective for table in relevant for effective, _, _ in table.values()
                          if first_day < effective <= last_day)
            event_days = sorted(events)
            for ei, day in enumerate(event_days):
                start = int(np.searchsorted(dates, day, side="left"))
                stop = (int(np.searchsorted(dates, event_days[ei + 1], side="left"))
                        if ei + 1 < len(event_days) else dates.size)
                if start >= stop:
                    continue
                q = _latest_annual(cash_table, day)
                if q is None:
                    continue
                value = _credit_impairment_value(cash_table, balance_table, q, day)
                stop = min(stop, _age_stop(dates, q))
                if start < stop:
                    out[start:stop, ci] = value

        return np.where(np.asarray(ctx.traded(), dtype=bool), out, np.nan)

    return compute

