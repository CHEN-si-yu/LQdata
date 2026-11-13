"""Artifact-only, parent-frozen descriptive GC drafts for PT01/PT05/PT06.

No registration or project-data execution is performed by this module.  It
accepts only one of the three exact signal names below; horizons and gates
are constants, not tuning parameters.
"""
from __future__ import annotations

import numpy as np


SIGNALS = frozenset({
    "same_calendar_month_return_profile",
    "turnover_to_nextday_abs_return_corr_126",
    "same_calendar_month_turnover_profile",
})
MONTH_SLOTS = 5
MIN_VALID_MONTH_SLOTS = 3
LEADLAG_PAIRS = 126
LEADLAG_MIN_PAIRS = 95
LEADLAG_CHUNK_COLS = 128
ANNUAL_MIN_VALID_TRADES = 150
MONTH_MIN_VALID_TRADES = 10


def _panel(x, name, dtype=np.float64):
    a = np.asarray(x, dtype=dtype)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2D (date, stock) array")
    return a


def _dates(ctx):
    p = np.asarray(ctx.panel.dates, dtype=np.int64)
    c = np.asarray(ctx.cal.days, dtype=np.int64)
    if p.ndim != 1 or c.ndim != 1 or p.size == 0 or c.size == 0:
        raise ValueError("panel/calendar dates must be nonempty one-dimensional arrays")
    if np.any(p[1:] <= p[:-1]) or np.any(c[1:] <= c[:-1]):
        raise ValueError("panel/calendar dates must be strictly increasing")
    cp = np.searchsorted(c, p)
    if np.any(cp >= c.size) or np.any(c[np.minimum(cp, c.size - 1)] != p):
        raise ValueError("every panel date must occur on the exchange calendar")
    c_to_p = np.full(c.size, -1, dtype=np.int64)
    c_to_p[cp] = np.arange(p.size, dtype=np.int64)
    return p, c, c_to_p


def _calendar_groups(cal_days):
    year = cal_days // 10000
    month = (cal_days // 100) % 100
    groups = {}
    for i, (y, m) in enumerate(zip(year, month)):
        key = (int(y), int(m))
        groups.setdefault(key, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}


def _full_panel_rows(cal_rows, c_to_p):
    p_rows = c_to_p[cal_rows]
    if p_rows.size == 0 or np.any(p_rows < 0):
        return None
    return p_rows


def same_calendar_month_return_profile(ctx):
    """Mean completed same-month returns over the fixed y-1..y-5 slots.

    A source month is all-or-nothing: it must be calendar-complete, have a
    traded/valid adjusted-close anchor in the immediately preceding calendar
    month, and every actually-traded row must have a finite clean return and
    positive observed raw/adjusted close.  True zero returns are retained.
    """
    ret = _panel(ctx.ret_clean(1), "ret_clean(1)")
    traded = _panel(ctx.traded(), "traded", dtype=bool)
    raw_close = _panel(ctx.px_raw("close"), "raw close")
    hfq_close = _panel(ctx.hfq("close"), "PIT adjusted close")
    universe = _panel(ctx.universe, "universe", dtype=bool)
    if any(x.shape != ret.shape for x in (traded, raw_close, hfq_close, universe)):
        raise ValueError("PT01 inputs must have matching (date, stock) shapes")
    panel_days, cal_days, c_to_p = _dates(ctx)
    if ret.shape[0] != panel_days.size:
        raise ValueError("PT01 panel date axis does not match arrays")
    groups = _calendar_groups(cal_days)

    month_returns = {}
    for (year, month), cal_rows in groups.items():
        p_rows = _full_panel_rows(cal_rows, c_to_p)
        prev_ordinal = year * 12 + month - 2
        prev_key = (prev_ordinal // 12, prev_ordinal % 12 + 1)
        prev_cal_rows = groups.get(prev_key)
        prev_p_rows = None if prev_cal_rows is None else _full_panel_rows(prev_cal_rows, c_to_p)
        if p_rows is None or prev_p_rows is None:
            month_returns[(year, month)] = np.full(ret.shape[1], np.nan, dtype=np.float64)
            continue

        month_traded = traded[p_rows]
        month_ret = ret[p_rows]
        month_raw = raw_close[p_rows]
        month_hfq = hfq_close[p_rows]
        n_traded = month_traded.sum(axis=0)
        good_return = np.isfinite(month_ret) & (month_ret > -1.0)
        good_endpoint = (np.isfinite(month_raw) & (month_raw > 0.0) &
                         np.isfinite(month_hfq) & (month_hfq > 0.0))
        all_current_returns_ok = np.all(~month_traded | good_return, axis=0)
        all_current_endpoints_ok = np.all(~month_traded | good_endpoint, axis=0)

        prev_traded = traded[prev_p_rows]
        has_prev = np.any(prev_traded, axis=0)
        # The last actually-traded row in the prior month is the close anchor.
        rev_last = prev_traded.shape[0] - 1 - np.argmax(prev_traded[::-1], axis=0)
        cols = np.arange(ret.shape[1])
        prev_raw_anchor = raw_close[prev_p_rows[rev_last], cols]
        prev_hfq_anchor = hfq_close[prev_p_rows[rev_last], cols]
        valid_anchor = (has_prev & np.isfinite(prev_raw_anchor) & (prev_raw_anchor > 0.0) &
                        np.isfinite(prev_hfq_anchor) & (prev_hfq_anchor > 0.0))

        factors = np.where(month_traded, 1.0 + month_ret, 1.0)
        with np.errstate(invalid="ignore", over="ignore"):
            compounded = np.prod(factors, axis=0, dtype=np.float64) - 1.0
        valid_month = ((n_traded > 0) & all_current_returns_ok & all_current_endpoints_ok &
                       valid_anchor & np.isfinite(compounded))
        month_returns[(year, month)] = np.where(valid_month, compounded, np.nan)

    out = np.full(ret.shape, np.nan, dtype=np.float64)
    for t, day in enumerate(panel_days):
        year = int(day // 10000)
        month = int((day // 100) % 100)
        values = []
        for j in range(1, MONTH_SLOTS + 1):
            prior_year = year - j
            v = month_returns.get((prior_year, month))
            if v is not None:
                values.append(v)
        if not values:
            continue
        stacked = np.stack(values, axis=0)
        count = np.isfinite(stacked).sum(axis=0)
        with np.errstate(invalid="ignore"):
            mean = np.nansum(stacked, axis=0) / np.maximum(count, 1)
        valid = (count >= MIN_VALID_MONTH_SLOTS) & universe[t] & traded[t] & np.isfinite(mean)
        out[t] = np.where(valid, mean, np.nan)
    out[~np.isfinite(out)] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)


def same_calendar_month_turnover_profile(ctx):
    """Year-normalized same-month share-turnover profile over y-1..y-5.

    Profiles use actually-traded valid rows only.  They require a full panel
    calendar year, >=150 valid traded observations in that year, >=10 in the
    same month, and a positive annual mean turnover.  Suspensions are omitted,
    never filled with zero; an observed zero flow remains an observed zero.
    """
    volume = _panel(ctx.px("vol"), "volume")
    float_share = _panel(ctx.px("float_share"), "float_share")
    traded = _panel(ctx.traded(), "traded", dtype=bool)
    universe = _panel(ctx.universe, "universe", dtype=bool)
    if any(x.shape != volume.shape for x in (float_share, traded, universe)):
        raise ValueError("PT06 inputs must have matching (date, stock) shapes")
    panel_days, cal_days, c_to_p = _dates(ctx)
    if volume.shape[0] != panel_days.size:
        raise ValueError("PT06 panel date axis does not match arrays")
    groups = _calendar_groups(cal_days)

    profiles = {}
    for year in sorted({k[0] for k in groups}):
        cal_year_rows = np.concatenate([rows for (y, _), rows in groups.items() if y == year])
        year_p_rows = _full_panel_rows(cal_year_rows, c_to_p)
        if year_p_rows is None:
            continue
        valid_year = (traded[year_p_rows] & np.isfinite(volume[year_p_rows]) &
                      (volume[year_p_rows] >= 0.0) &
                      np.isfinite(float_share[year_p_rows]) &
                      (float_share[year_p_rows] > 0.0))
        year_to = np.full(valid_year.shape, np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(volume[year_p_rows], float_share[year_p_rows], out=year_to,
                      where=valid_year)
        valid_year &= np.isfinite(year_to) & (year_to >= 0.0)
        year_count = valid_year.sum(axis=0)
        annual_sum = np.where(valid_year, year_to, 0.0).sum(axis=0, dtype=np.float64)
        annual_mean = annual_sum / np.maximum(year_count, 1)
        annual_ok = ((year_count >= ANNUAL_MIN_VALID_TRADES) & np.isfinite(annual_mean) &
                     (annual_mean > 0.0))

        for (y, month), cal_month_rows in groups.items():
            if y != year:
                continue
            month_p_rows = _full_panel_rows(cal_month_rows, c_to_p)
            if month_p_rows is None:
                profiles[(year, month)] = np.full(volume.shape[1], np.nan, dtype=np.float64)
                continue
            # Convert calendar indices to their positions in the ordered year panel.
            year_row_lookup = {int(row): i for i, row in enumerate(cal_year_rows)}
            month_positions = np.asarray([year_row_lookup[int(row)] for row in cal_month_rows],
                                         dtype=np.int64)
            month_valid = valid_year[month_positions]
            month_values = year_to[month_positions]
            month_count = month_valid.sum(axis=0)
            month_sum = np.where(month_valid, month_values, 0.0).sum(axis=0, dtype=np.float64)
            month_mean = month_sum / np.maximum(month_count, 1)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                q = month_mean / annual_mean
            ok = ((month_count >= MONTH_MIN_VALID_TRADES) & annual_ok &
                  np.isfinite(month_mean) & (month_mean >= 0.0) & np.isfinite(q))
            profiles[(year, month)] = np.where(ok, q, np.nan)

    out = np.full(volume.shape, np.nan, dtype=np.float64)
    for t, day in enumerate(panel_days):
        year = int(day // 10000)
        month = int((day // 100) % 100)
        values = [profiles.get((year - j, month), np.full(volume.shape[1], np.nan,
                                                           dtype=np.float64))
                  for j in range(1, MONTH_SLOTS + 1)]
        stacked = np.stack(values, axis=0)
        count = np.isfinite(stacked).sum(axis=0)
        mean = np.nansum(stacked, axis=0) / np.maximum(count, 1)
        valid = (count >= MIN_VALID_MONTH_SLOTS) & universe[t] & traded[t] & np.isfinite(mean)
        out[t] = np.where(valid, mean, np.nan)
    out[~np.isfinite(out)] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)


def _rolling_sum_block(x, window):
    cs = np.empty((x.shape[0] + 1, x.shape[1]), dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(x, axis=0, dtype=np.float64, out=cs[1:])
    out = np.zeros_like(x, dtype=np.float64)
    if x.shape[0] >= window:
        out[window - 1:] = cs[window:] - cs[:-window]
    return out


def turnover_to_nextday_abs_return_corr_126(ctx):
    """Corr(log1p(percent turnover_s), |ret_clean(s+1)|), 126 pairs/min 95."""
    turnover_pct = _panel(ctx.px("turnover"), "percent turnover")
    ret = _panel(ctx.ret_clean(1), "ret_clean(1)")
    traded = _panel(ctx.traded(), "traded", dtype=bool)
    universe = _panel(ctx.universe, "universe", dtype=bool)
    if any(x.shape != turnover_pct.shape for x in (ret, traded, universe)):
        raise ValueError("PT05 inputs must have matching (date, stock) shapes")

    T, C = turnover_pct.shape
    out = np.full((T, C), np.nan, dtype=np.float64)
    mature = np.arange(T) >= LEADLAG_PAIRS
    for left in range(0, C, LEADLAG_CHUNK_COLS):
        right = min(left + LEADLAG_CHUNK_COLS, C)
        to = turnover_pct[:, left:right]
        rr = ret[:, left:right]
        tr = traded[:, left:right]
        xlevel = np.full(to.shape, np.nan, dtype=np.float64)
        valid_x = np.isfinite(to) & (to >= 0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            xlevel[valid_x] = np.log1p(to[valid_x])
        x = np.zeros(to.shape, dtype=np.float64)
        y = np.zeros(to.shape, dtype=np.float64)
        pair = np.zeros(to.shape, dtype=bool)
        if T > 1:
            pair[1:] = (tr[:-1] & tr[1:] & np.isfinite(xlevel[:-1]) & np.isfinite(rr[1:]))
            x[1:] = np.where(pair[1:], xlevel[:-1], 0.0)
            y[1:] = np.where(pair[1:], np.abs(rr[1:]), 0.0)
        w = pair.astype(np.float64)
        n = _rolling_sum_block(w, LEADLAG_PAIRS)
        sx = _rolling_sum_block(x, LEADLAG_PAIRS)
        sy = _rolling_sum_block(y, LEADLAG_PAIRS)
        sxx = _rolling_sum_block(x * x, LEADLAG_PAIRS)
        syy = _rolling_sum_block(y * y, LEADLAG_PAIRS)
        sxy = _rolling_sum_block(x * y, LEADLAG_PAIRS)
        n_safe = np.maximum(n, 1.0)
        vx = sxx - sx * sx / n_safe
        vy = syy - sy * sy / n_safe
        cov = sxy - sx * sy / n_safe
        tol_x = 64.0 * np.finfo(np.float64).eps * np.maximum.reduce(
            [np.abs(sxx), np.abs(sx * sx / n_safe), np.ones_like(sxx)])
        tol_y = 64.0 * np.finfo(np.float64).eps * np.maximum.reduce(
            [np.abs(syy), np.abs(sy * sy / n_safe), np.ones_like(syy)])
        good = (mature[:, None] & (n >= LEADLAG_MIN_PAIRS) &
                np.isfinite(vx) & np.isfinite(vy) & np.isfinite(cov) &
                (vx > tol_x) & (vy > tol_y))
        rho = np.full(to.shape, np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            rho[good] = cov[good] / np.sqrt(vx[good] * vy[good])
        rho = np.where(np.isfinite(rho), np.clip(rho, -1.0, 1.0), np.nan)
        current = universe[:, left:right] & tr
        out[:, left:right] = np.where(current & good & np.isfinite(rho), rho, np.nan)
    out[~np.isfinite(out)] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)


def build(parameters):
    """Accept exactly one fixed descriptive signal; there are no tunable knobs."""
    if not isinstance(parameters, dict) or set(parameters) != {"signal"}:
        raise ValueError("parameters must be exactly {'signal': <frozen signal name>}")
    signal = parameters["signal"]
    if not isinstance(signal, str) or signal not in SIGNALS:
        raise ValueError(f"signal must be one of {sorted(SIGNALS)}")

    def compute(ctx):
        if signal == "same_calendar_month_return_profile":
            return same_calendar_month_return_profile(ctx)
        if signal == "turnover_to_nextday_abs_return_corr_126":
            return turnover_to_nextday_abs_return_corr_126(ctx)
        return same_calendar_month_turnover_profile(ctx)

    return compute
