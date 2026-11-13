"""Staging implementation of the frozen R39 FHP investment-CFO slope.

This module has no research-directory or NPZ dependency. ``build(parameters)``
returns the normal FactorContext ``compute(ctx) -> (T,C)`` function. The public
``compute_with_aux(ctx)`` helper returns the frozen main and ten source
diagnostics for parity checks; production's main factor is exactly the first
array from that helper.

The source loader always starts at raw partition year 2010. It filters each
raw row by a real announcement PIT strictly before the as-of date before
first-vintage selection. The 3000-day Engine warmup is natural calendar days;
it does not replace the absolute 2010 financial-source initialization floor.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd


NAME = "fhp_investment_cfo_sensitivity_v1"
AUX_NAMES = (
    "fhp_aux_mean_investment_rate_v1",
    "fhp_aux_sd_investment_rate_v1",
    "fhp_aux_sales_growth_slope_v1",
    "fhp_aux_fit_n_v1",
    "fhp_aux_mean_cfo_rate_v1",
    "fhp_aux_mean_sales_growth_rate_v1",
    "fhp_aux_latest_fy_v1",
    "fhp_aux_report_age_days_v1",
    "fhp_aux_frontier_cf_pit_v1",
    "fhp_aux_frontier_valid_v1",
)
BASE_YEAR = 2010
WARMUP_DAYS = 3000
MAX_REPORT_AGE_DAYS = 550
WINDOW_FYS = 8
MIN_VALID_SLOTS = 6
MIN_RANK = 3
ASSET_FLOOR_CNY = 1_000_000.0

_TABLE_FIELDS = {
    "stock_cashflow": ("c_pay_acq_const_fiolta", "n_cashflow_act"),
    "stock_income": ("revenue",),
    "stock_balancesheet": ("total_assets",),
}
_REQUIRED = ("stock_code", "ann_date", "f_ann_date", "end_date",
            "report_type", "comp_type")


def _dates_to_int(values: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(values, errors="coerce")
    return ((dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day)
            .fillna(0).to_numpy(np.int32))


def _date(v: int) -> _dt.date:
    v = int(v)
    return _dt.date(v // 10000, (v // 100) % 100, v % 100)


def _num(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(np.float64)


def _read_first_vintages(ctx, table: str, fields: tuple[str, ...],
                         codes: np.ndarray, final_date: int) -> dict:
    """Read 2010..end-year partitions, strict PIT<T, then whole-row first vintage.

    At least one actual announcement date is required. A valid announcement
    earlier than period end invalidates the row. ``pit`` is the maximum valid
    announcement date and report end. Sorting/dedup matches the frozen source
    builder: same-PIT duplicate keeps last physical partition/row, then the
    earliest PIT whole row wins for each code/report-period.
    """
    ctx.accessed.add(table)
    wanted = set(map(str, codes.tolist()))
    end_year = int(final_date) // 10000
    columns = [*_REQUIRED, *fields]
    chunks: list[pd.DataFrame] = []
    for part_year in range(BASE_YEAR, end_year + 1):
        frame = ctx.up.read(table, columns=columns,
                            years=(part_year, part_year), use_cache=False)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["stock_code"] = frame["stock_code"].astype(str)
        frame = frame.loc[frame.stock_code.isin(wanted)]
        if frame.empty:
            continue
        ann = _dates_to_int(frame.ann_date)
        fann = _dates_to_int(frame.f_ann_date)
        period = _dates_to_int(frame.end_date)
        valid_ann = (ann >= 19900101) & (ann <= 21001231)
        valid_fann = (fann >= 19900101) & (fann <= 21001231)
        has_real_announcement = valid_ann | valid_fann
        invalid_early_announcement = ((valid_ann & (ann < period)) |
                                      (valid_fann & (fann < period)))
        pit = np.maximum.reduce((np.where(valid_ann, ann, 0),
                                 np.where(valid_fann, fann, 0), period))
        report_type = pd.to_numeric(frame.report_type, errors="coerce").to_numpy()
        comp_type = pd.to_numeric(frame.comp_type, errors="coerce").to_numpy()
        keep = (period > 0) & (period % 10000 == 1231)
        keep &= (report_type == 1) & (comp_type == 1)
        keep &= has_real_announcement & ~invalid_early_announcement & (pit >= period)
        # Raw strict prefix is applied before any first-vintage selection.
        keep &= pit < int(final_date)
        if not keep.any():
            continue
        frame = frame.loc[keep].copy()
        frame["_period"] = period[keep] // 10000
        frame["_end"] = period[keep]
        frame["_pit"] = pit[keep]
        frame["_partition"] = part_year
        frame["_row"] = np.flatnonzero(keep)
        for field in fields:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
        chunks.append(frame[["stock_code", "_period", "_end", "_pit",
                             "_partition", "_row", *fields]])
    if not chunks:
        return {}
    rows = pd.concat(chunks, ignore_index=True)
    rows = rows.sort_values(["stock_code", "_period", "_pit", "_partition", "_row"],
                            kind="stable")
    rows = rows.drop_duplicates(["stock_code", "_period", "_pit"], keep="last")
    rows = rows.drop_duplicates(["stock_code", "_period"], keep="first")
    result = {}
    for row in rows.to_dict("records"):
        item = {field: (float(row[field]) if pd.notna(row[field]) else np.nan)
                for field in fields}
        item.update(pit=int(row["_pit"]), period=int(row["_end"]))
        result[(str(row["stock_code"]), int(row["_period"]))] = item
    return result


def _fit_one(day: int, fy: int, cf_by_fy: dict, income_by_fy: dict,
             bs_by_fy: dict) -> tuple[float, ...]:
    """Calculate one as-of frontier value and ten diagnostics in frozen order."""
    aux = [np.nan] * len(AUX_NAMES)
    cf_current = cf_by_fy[fy]
    age = (_date(day) - _date(cf_current["period"])).days
    aux[6] = float(fy)
    aux[7] = float(age)
    aux[8] = float(cf_current["pit"])
    if age < 0 or age > MAX_REPORT_AGE_DAYS:
        aux[9] = 0.0
        return np.nan, *aux

    i_current = income_by_fy.get(fy)
    i_prior = income_by_fy.get(fy - 1)
    bs_prior = bs_by_fy.get(fy - 1)

    def visible(row):
        return (row is not None and int(row["pit"]) < day and int(row["period"]) < day)

    current_visible = all(visible(r) for r in (cf_current, i_current, i_prior, bs_prior))
    current_values = (
        cf_current["c_pay_acq_const_fiolta"], cf_current["n_cashflow_act"],
        np.nan if i_current is None else i_current["revenue"],
        np.nan if i_prior is None else i_prior["revenue"],
        np.nan if bs_prior is None else bs_prior["total_assets"],
    )
    current_valid = (current_visible and np.isfinite(current_values).all()
                     and current_values[0] >= 0.0
                     and current_values[4] > ASSET_FLOOR_CNY)
    aux[9] = float(bool(current_valid))
    if not current_valid:
        return np.nan, *aux

    investments, cfo_rates, sales_growth = [], [], []
    for slot_fy in range(fy - WINDOW_FYS + 1, fy + 1):
        cf = cf_by_fy.get(slot_fy)
        inc_cur = income_by_fy.get(slot_fy)
        inc_lag = income_by_fy.get(slot_fy - 1)
        bs_lag = bs_by_fy.get(slot_fy - 1)
        if not all(visible(r) for r in (cf, inc_cur, inc_lag, bs_lag)):
            continue
        assets = bs_lag["total_assets"]
        capex = cf["c_pay_acq_const_fiolta"]
        cfo = cf["n_cashflow_act"]
        revenue = inc_cur["revenue"]
        prior_revenue = inc_lag["revenue"]
        values = (assets, capex, cfo, revenue, prior_revenue)
        if not np.isfinite(values).all() or assets <= ASSET_FLOOR_CNY or capex < 0.0:
            continue
        investments.append(capex / assets)
        cfo_rates.append(cfo / assets)
        sales_growth.append((revenue - prior_revenue) / assets)

    n = len(investments)
    aux[3] = float(n)
    if n < MIN_VALID_SLOTS:
        return np.nan, *aux
    y = np.asarray(investments, dtype=np.float64)
    c = np.asarray(cfo_rates, dtype=np.float64)
    g = np.asarray(sales_growth, dtype=np.float64)
    x = np.column_stack((np.ones(n, dtype=np.float64), c, g))
    if np.linalg.matrix_rank(x) < MIN_RANK:
        return np.nan, *aux
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    aux[0] = float(y.mean(dtype=np.float64))
    aux[1] = float(y.std(ddof=0, dtype=np.float64))
    aux[2] = float(beta[2])
    aux[4] = float(c.mean(dtype=np.float64))
    aux[5] = float(g.mean(dtype=np.float64))
    return float(beta[1]), *aux


def _visibility_key(day: int, fy: int, cf_by_fy: dict, income_by_fy: dict,
                    bs_by_fy: dict) -> tuple[bool, ...]:
    """Exact report visibility state for the fixed eight-FY regression window."""
    flags = []
    for slot_fy in range(fy - WINDOW_FYS + 1, fy + 1):
        for row in (cf_by_fy.get(slot_fy), income_by_fy.get(slot_fy),
                    income_by_fy.get(slot_fy - 1), bs_by_fy.get(slot_fy - 1)):
            flags.append(row is not None and int(row["pit"]) < day
                         and int(row["period"]) < day)
    return tuple(flags)


def compute_with_aux(ctx) -> dict[str, np.ndarray]:
    """Return main and ten diagnostics as float32 (T,C) arrays."""
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    codes = np.asarray(ctx.panel.codes).astype(str)
    universe = np.asarray(ctx.universe, dtype=bool)
    if universe.shape != (len(dates), len(codes)):
        raise ValueError("FactorContext universe does not match panel axes")
    if not len(dates):
        return {name: np.empty((0, len(codes)), dtype=np.float32)
                for name in (NAME, *AUX_NAMES)}
    if np.any(dates[1:] <= dates[:-1]):
        raise ValueError("FactorContext dates must be strictly increasing")

    last_day = int(dates[-1])
    income = _read_first_vintages(ctx, "stock_income", _TABLE_FIELDS["stock_income"],
                                  codes, last_day)
    cashflow = _read_first_vintages(ctx, "stock_cashflow", _TABLE_FIELDS["stock_cashflow"],
                                    codes, last_day)
    balance = _read_first_vintages(ctx, "stock_balancesheet", _TABLE_FIELDS["stock_balancesheet"],
                                   codes, last_day)
    income_by_code: dict[str, dict[int, dict]] = {}
    cf_by_code: dict[str, dict[int, dict]] = {}
    bs_by_code: dict[str, dict[int, dict]] = {}
    for (code, fy), row in income.items():
        income_by_code.setdefault(code, {})[fy] = row
    for (code, fy), row in cashflow.items():
        cf_by_code.setdefault(code, {})[fy] = row
    for (code, fy), row in balance.items():
        bs_by_code.setdefault(code, {})[fy] = row

    shape = universe.shape
    out = {name: np.full(shape, np.nan, dtype=np.float32) for name in (NAME, *AUX_NAMES)}
    col = {str(code): i for i, code in enumerate(codes)}
    day_ord = np.asarray([_date(int(d)).toordinal() for d in dates], dtype=np.int32)
    for code, cf_history in cf_by_code.items():
        ci = col.get(code)
        if ci is None:
            continue
        i_history = income_by_code.get(code, {})
        b_history = bs_by_code.get(code, {})
        # Every possible input visibility change and every 550-day expiry is
        # converted to the first trading-panel index at which its state changes.
        # The regression is therefore fitted once per actual report state, not
        # once per stock-day. Daily report age is still emitted on every row.
        event_pits = []
        expiry_ordinals = []
        for history in (cf_history, i_history, b_history):
            event_pits.extend(int(row["pit"]) for row in history.values())
        for row in cf_history.values():
            expiry_ordinals.append((_date(row["period"]) + _dt.timedelta(
                days=MAX_REPORT_AGE_DAYS)).toordinal())
        boundaries = {0, len(dates)}
        for pit in event_pits:
            ix = int(np.searchsorted(dates, pit, side="right"))
            if 0 < ix < len(dates):
                boundaries.add(ix)
        for expiry in expiry_ordinals:
            ix = int(np.searchsorted(day_ord, expiry, side="right"))
            if 0 < ix < len(dates):
                boundaries.add(ix)
        cuts = sorted(boundaries)
        fyears = np.asarray(sorted(cf_history), dtype=np.int32)
        fpits = np.asarray([cf_history[int(y)]["pit"] for y in fyears], dtype=np.int32)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if lo >= hi:
                continue
            visible_universe = universe[lo:hi, ci]
            if not visible_universe.any():
                continue
            day = int(dates[lo])
            visible_fy = fyears[fpits < day]
            if not len(visible_fy):
                continue
            fy = int(visible_fy[-1])
            cf_current = cf_history[fy]
            age_vec = day_ord[lo:hi] - _date(cf_current["period"]).toordinal()
            age0 = int(age_vec[0])
            if age0 < 0 or age0 > MAX_REPORT_AGE_DAYS:
                fitted = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                          np.nan, float(fy), float(age0), float(cf_current["pit"]), 0.0)
            else:
                fitted = _fit_one(day, fy, cf_history, i_history, b_history)
            out[NAME][lo:hi, ci] = np.float32(fitted[0])
            for j, aux_name in enumerate(AUX_NAMES):
                out[aux_name][lo:hi, ci] = np.float32(fitted[j + 1])
            out[AUX_NAMES[7]][lo:hi, ci] = age_vec.astype(np.float32)
    for array in out.values():
        array[~universe] = np.nan
    return out


def build(parameters):
    """Return the formal Engine compute function for the frozen sole variant."""
    if parameters not in ({}, None, {"variant": NAME}):
        raise ValueError(f"only frozen variant {NAME!r} is supported")

    def compute(ctx):
        return compute_with_aux(ctx)[NAME]

    return compute
