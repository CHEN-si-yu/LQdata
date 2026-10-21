"""R17 seat_cross batch 3: calendar / seasonality exposures (C + GC).

All calendar factors are stock-level cross-sectional objects; market-level
calendar dummies have zero cross-sectional dispersion and are not registered.

Completed-period discipline
---------------------------
Month-level objects use only months that are strictly BEFORE the output month.
For a day in month s, the value is a step function of monthly aggregates over
s-k..s-1.  Year-level objects use only calendar years strictly before the
output year.  Daily rolling objects (weekday, pre/post holiday) use only the
trailing n sessions ending at t.  No current-month or future month return is
ever used, and no future month-end is assumed: month boundaries come from the
panel trading calendar and a month segment is complete by construction once
its rows exist.

Returns
-------
Monthly returns use ctx.ret(1) (the price layer's documented suspension=0
semantics) and follow the r4j rule: a month is NaN if it has no traded day or
if any daily value in the month is NaN.  Daily conditional means use
ret_clean(1) (traded days only), so a suspension day is missing, not zero.

Candidates
----------
C: month-start spread, month-end spread, mid-month spread, pre-holiday spread,
post-holiday spread, earnings-season spread/consistency, January spread,
Q4-to-Q1 reversal.  GC: Monday/Friday spreads, weekday range, weekday-profile
stability, monthly seasonality dispersion.
"""
from __future__ import annotations

import warnings

import numpy as np

WARMUP_DAYS = 1400
HOLIDAY_GAP_DAYS = 4       # Fri->Mon is 3; require a real holiday gap

C_VARIANTS = (
    "cal_month_start_spread_12m",
    "cal_month_end_spread_12m",
    "cal_midmonth_spread_12m",
    "cal_preholiday_spread_3y",
    "cal_postholiday_spread_3y",
    "cal_earnings_season_spread_3y",
    "cal_earnings_season_consistency_3y",
    "cal_january_spread_3y",
    "cal_q4_q1_reversal_3y",
)
GC_VARIANTS = (
    "cal_dow_monday_spread_252",
    "cal_dow_friday_spread_252",
    "cal_weekday_range_252",
    "cal_dow_profile_stability_252",
    "cal_monthly_seasonality_dispersion_3y",
)
PARAMETERS = {name: {"variant": name} for name in C_VARIANTS + GC_VARIANTS}
ALLOWED_VARIANTS = tuple(PARAMETERS[name] for name in PARAMETERS)

_CACHE_KEY = "_r17_cross_calendar_v1"


def _cs(a):
    return np.vstack([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])


def _roll_sum(a, n, min_count):
    T = a.shape[0]
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if T < n:
        return out
    fin = np.isfinite(a)
    x = np.where(fin, a, 0.0)
    s = _cs(x)
    c = _cs(fin.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        out[n - 1:] = np.where((c[n:] - c[:-n]) >= max(int(min_count), 1),
                               s[n:] - s[:-n], np.nan)
    return out


def _roll_mean_ind(y, ind, n, min_count):
    yy = np.where(ind, y, np.nan)
    ii = np.where(ind, 1.0, np.nan)
    s = _roll_sum(yy, n, min_count)
    c = _roll_sum(ii, n, min_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / c


def _weekday(days: np.ndarray) -> np.ndarray:
    """0=Monday..6=Sunday via Zeller's congruence (vectorised)."""
    y = days // 10000
    m = (days // 100) % 100
    d = days % 100
    adj = m < 3
    m = np.where(adj, m + 12, m)
    y = np.where(adj, y - 1, y)
    K = y % 100
    J = y // 100
    h = (d + (13 * (m + 1)) // 5 + K + K // 4 + J // 4 + 5 * J) % 7
    return (h + 5) % 7


def _month_runs(days):
    tags = np.asarray(days, dtype=np.int64) // 100
    starts = np.flatnonzero(np.r_[True, tags[1:] != tags[:-1]])
    ends = np.r_[starts[1:], len(tags)]
    return [int(tags[s]) for s in starts], starts, ends


def _monthly_returns(r_ff, tr, starts):
    """(S,N) product returns; NaN month if no traded day or any NaN daily."""
    x = 1.0 + r_ff
    gross = np.multiply.reduceat(x, starts, axis=0)
    cnt = np.add.reduceat(tr.astype(np.float64), starts, axis=0)
    out = gross - 1.0
    out[cnt < 1] = np.nan
    return out


def _month_spread(r_clean, starts, ends, ind_fn, min_hi, min_lo):
    """Per-month mean return on indicator days minus other days."""
    S, N = len(starts), r_clean.shape[1]
    out = np.full((S, N), np.nan)
    for s in range(S):
        d0, d1 = starts[s], ends[s]
        ind = ind_fn(d1 - d0)
        if np.asarray(ind).ndim == 1:
            ind = np.asarray(ind)[:, None]
        sl = r_clean[d0:d1]
        hi = ind & np.isfinite(sl)
        lo = (~ind) & np.isfinite(sl)
        nh = hi.sum(axis=0)
        nl = lo.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mh = np.where(hi, sl, 0.0).sum(axis=0) / np.maximum(nh, 1)
            ml = np.where(lo, sl, 0.0).sum(axis=0) / np.maximum(nl, 1)
        out[s] = np.where((nh >= min_hi) & (nl >= min_lo), mh - ml, np.nan)
    return out


def _prior_month_mean(arr, starts, k, min_valid):
    """For each month s, mean of arr over [s-k, s-1]; NaN if fewer valid."""
    S, N = arr.shape
    val = np.full((S, N), np.nan)
    for s in range(S):
        lo = max(0, s - k)
        if s - lo < min_valid:
            continue
        block = arr[lo:s]
        cnt = np.isfinite(block).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.nansum(block, axis=0) / np.maximum(cnt, 1)
        val[s] = np.where(cnt >= min_valid, v, np.nan)
    return val


def _broadcast_month(val, starts, ends, T):
    out = np.full((T, val.shape[1]), np.nan)
    for s in range(len(starts)):
        out[starts[s]:ends[s]] = val[s][None, :]
    return out


def _year_agg(tags, mret, month_sel):
    """Per calendar year mean monthly return over selected months (S,N)->(Y,N)."""
    years = np.asarray([t // 100 for t in tags])
    uniq = np.unique(years)
    out = {}
    for y in uniq:
        rows = np.flatnonzero((years == y) & month_sel)
        if len(rows) == 0:
            continue
        block = mret[rows]
        cnt = np.isfinite(block).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.nansum(block, axis=0) / np.maximum(cnt, 1)
        out[int(y)] = np.where(cnt >= max(1, len(rows) - 1), v, np.nan)
    return out


class _Cal:
    def __init__(self, ctx):
        cache = ctx.__dict__.setdefault(_CACHE_KEY, {})
        if cache.get("ready"):
            self.__dict__.update(cache["obj"])
            self.cache = cache
            return
        self.ctx = ctx
        days = np.asarray(ctx.panel.dates, dtype=np.int64)
        T, N = len(days), ctx.panel.C
        self.days = days
        self.T = T
        self.cache = {}
        r_clean = np.array(ctx.ret_clean(1), dtype=np.float64, copy=True)
        tr = np.asarray(ctx.traded(), dtype=bool)
        univ = np.asarray(ctx.universe, dtype=bool)
        adj = tr.copy()
        adj[0] = False
        adj[1:] &= tr[:-1]
        r_clean[~adj] = np.nan
        self.r_clean = r_clean
        self.tr = tr
        self.univ = univ
        self.r_ff = ctx.ret(1)

        tags, starts, ends = _month_runs(days)
        self.tags = tags
        self.starts = starts
        self.ends = ends
        self.mret = _monthly_returns(self.r_ff, tr, starts)
        self.midx = np.searchsorted(starts, np.arange(T), side="right") - 1
        self.wd = _weekday(days)

        # holiday indicators from the full trading calendar
        cal = np.asarray(ctx.cal.days, dtype=np.int64)
        gap = np.diff(cal)
        pre_days = set(cal[np.flatnonzero(gap >= HOLIDAY_GAP_DAYS)].tolist())
        post_days = set(cal[np.flatnonzero(gap >= HOLIDAY_GAP_DAYS) + 1].tolist())
        self.pre_hol = np.array([d in pre_days for d in days])
        self.post_hol = np.array([d in post_days for d in days])

        # monthly conditional spreads
        self.m_start = _month_spread(
            r_clean, starts, ends, lambda L: np.arange(L) < 3, 2, 5)
        self.m_end = _month_spread(
            r_clean, starts, ends, lambda L: np.arange(L) >= L - 3, 2, 5)
        self.m_mid = _month_spread(
            r_clean, starts, ends, lambda L: (np.arange(L) >= 6) & (np.arange(L) <= 15),
            4, 5)

        # prior-window monthly averages (step functions)
        self.v_start = _prior_month_mean(self.m_start, starts, 12, 8)
        self.v_end = _prior_month_mean(self.m_end, starts, 12, 8)
        self.v_mid = _prior_month_mean(self.m_mid, starts, 12, 8)
        self.v_start_panel = _broadcast_month(self.v_start, starts, ends, T)
        self.v_end_panel = _broadcast_month(self.v_end, starts, ends, T)
        self.v_mid_panel = _broadcast_month(self.v_mid, starts, ends, T)

        # earnings season / January / Q4-Q1 (year-level with prior 3 years)
        month_of = np.asarray([t % 100 for t in tags])
        years = np.asarray([t // 100 for t in tags])
        es = _year_agg(tags, self.mret, np.isin(month_of, (4, 8, 10)))
        non_es = _year_agg(tags, self.mret, ~np.isin(month_of, (4, 8, 10)))
        jan = _year_agg(tags, self.mret, month_of == 1)
        non_jan = _year_agg(tags, self.mret, month_of != 1)
        q4 = _year_agg(tags, self.mret, month_of == 12)
        q123 = _year_agg(tags, self.mret, np.isin(month_of, (1, 2, 3, 4, 5, 6, 7, 8, 9)))
        pos_ind = np.where(np.isfinite(self.mret),
                           (self.mret > 0).astype(np.float64), np.nan)
        pos_es = _year_agg(tags, pos_ind, np.isin(month_of, (4, 8, 10)))
        pos_non = _year_agg(tags, pos_ind, ~np.isin(month_of, (4, 8, 10)))

        def prior3(mapping, y):
            vals = [mapping.get(y - k) for k in (1, 2, 3)]
            vals = [v for v in vals if v is not None]
            if len(vals) < 2:
                return np.full(N, np.nan)
            arr = np.vstack(vals)
            cnt = np.isfinite(arr).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.nansum(arr, axis=0) / np.maximum(cnt, 1)
            return np.where(cnt >= 2, out, np.nan)

        es_val = np.full((len(starts), N), np.nan)
        es_con = np.full((len(starts), N), np.nan)
        jan_val = np.full((len(starts), N), np.nan)
        q_val = np.full((len(starts), N), np.nan)
        sea_m = np.full((12,), np.nan)
        for y in np.unique(years):
            rows = np.flatnonzero(years == y)
            ev = prior3(es, y) - prior3(non_es, y)
            ec = prior3(pos_es, y) - prior3(pos_non, y)
            jv = prior3(jan, y) - prior3(non_jan, y)
            # Q4-Q1 uses only the last completed year (Y-1)
            if (y - 1) in q4 and (y - 1) in q123:
                qv = q4[y - 1] - q123[y - 1]
            else:
                qv = np.full(N, np.nan)
            es_val[rows] = ev
            es_con[rows] = ec
            jan_val[rows] = jv
            q_val[rows] = qv

        # dispersion across the 12 calendar-month means over prior 3 years
        sea_val = np.full((len(starts), N), np.nan)
        for s in range(len(starts)):
            y = tags[s] // 100
            means = []
            for mm in range(1, 13):
                vals = []
                for k in (1, 2, 3):
                    rows = np.flatnonzero((years == y - k) & (month_of == mm))
                    if len(rows):
                        vals.append(self.mret[rows[0]])
                if len(vals) >= 2:
                    arr = np.vstack(vals)
                    cnt = np.isfinite(arr).sum(axis=0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mu = np.nansum(arr, axis=0) / np.maximum(cnt, 1)
                    means.append(np.where(cnt >= 2, mu, np.nan))
            if len(means) >= 8:
                arr = np.vstack(means)
                cnt = np.isfinite(arr).sum(axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mu = np.nansum(arr, axis=0) / np.maximum(cnt, 1)
                    var = np.nansum((arr - mu[None, :]) ** 2, axis=0) / np.maximum(cnt, 1)
                sea_val[s] = np.where(cnt >= 8, np.sqrt(np.maximum(var, 0.0)), np.nan)

        self.es_val = _broadcast_month(es_val, starts, ends, T)
        self.es_con = _broadcast_month(es_con, starts, ends, T)
        self.jan_val = _broadcast_month(jan_val, starts, ends, T)
        self.q_val = _broadcast_month(q_val, starts, ends, T)
        self.sea_val = _broadcast_month(sea_val, starts, ends, T)

        # weekday means over trailing 252 sessions
        wd_means = []
        for w in range(5):
            ind = self.wd == w
            wd_means.append(_roll_mean_ind(r_clean, ind[:, None], 252, 20))
        self.wd_means = wd_means
        mon = wd_means[0]
        other = np.nanmean(np.stack(wd_means[1:], axis=0), axis=0)
        self.mon_spread = mon - other
        fri = wd_means[4]
        other2 = np.nanmean(np.stack(wd_means[:4], axis=0), axis=0)
        self.fri_spread = fri - other2
        stack = np.stack(wd_means, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.wd_range = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
        # weekday-profile stability across two adjacent 252-session blocks
        self.profile_stab = _profile_stability(r_clean, self.wd)

        cache["obj"] = {k: v for k, v in self.__dict__.items() if k != "cache"}
        cache["ready"] = True

    def out(self, values):
        return np.where(self.tr & self.univ, values, np.nan)


def _profile_stability(r_clean, wd):
    """Corr across weekdays between mean returns in [t-503,t-252] and [t-251,t]."""
    T, N = r_clean.shape
    sumA = np.zeros((T, N)); sumB = np.zeros((T, N))
    sumAA = np.zeros((T, N)); sumBB = np.zeros((T, N)); sumAB = np.zeros((T, N))
    cnt = np.zeros((T, N))
    for w in range(5):
        m = _roll_mean_ind(r_clean, (wd == w)[:, None], 252, 10)
        a = np.full((T, N), np.nan); a[252:] = m[:-252]
        ok = np.isfinite(a) & np.isfinite(m)
        sumA += np.where(ok, a, 0.0); sumB += np.where(ok, m, 0.0)
        sumAA += np.where(ok, a * a, 0.0); sumBB += np.where(ok, m * m, 0.0)
        sumAB += np.where(ok, a * m, 0.0)
        cnt += ok.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.maximum(cnt, 1.0)
        ma = sumA / n; mb = sumB / n
        cov = sumAB / n - ma * mb
        va = sumAA / n - ma * ma
        vb = sumBB / n - mb * mb
        out = cov / np.sqrt(va * vb)
    out[(cnt < 5) | ~(va > 0) | ~(vb > 0)] = np.nan
    return out


def build(parameters):
    if parameters not in ALLOWED_VARIANTS:
        raise ValueError("Only frozen batch-3 calendar variants are allowed")
    variant = parameters["variant"]

    def compute(ctx):
        c = _Cal(ctx)
        if variant == "cal_month_start_spread_12m":
            out = c.v_start_panel
        elif variant == "cal_month_end_spread_12m":
            out = c.v_end_panel
        elif variant == "cal_midmonth_spread_12m":
            out = c.v_mid_panel
        elif variant == "cal_preholiday_spread_3y":
            ind = c.pre_hol[:, None]
            hi = _roll_mean_ind(c.r_clean, ind, 756, 12)
            lo = _roll_mean_ind(c.r_clean, ~ind, 756, 200)
            out = hi - lo
        elif variant == "cal_postholiday_spread_3y":
            ind = c.post_hol[:, None]
            hi = _roll_mean_ind(c.r_clean, ind, 756, 12)
            lo = _roll_mean_ind(c.r_clean, ~ind, 756, 200)
            out = hi - lo
        elif variant == "cal_earnings_season_spread_3y":
            out = c.es_val
        elif variant == "cal_earnings_season_consistency_3y":
            out = c.es_con
        elif variant == "cal_january_spread_3y":
            out = c.jan_val
        elif variant == "cal_q4_q1_reversal_3y":
            out = c.q_val
        elif variant == "cal_dow_monday_spread_252":
            out = c.mon_spread
        elif variant == "cal_dow_friday_spread_252":
            out = c.fri_spread
        elif variant == "cal_weekday_range_252":
            out = c.wd_range
        elif variant == "cal_dow_profile_stability_252":
            out = c.profile_stab
        elif variant == "cal_monthly_seasonality_dispersion_3y":
            out = c.sea_val
        else:
            raise KeyError(variant)
        if np.isinf(out).any():
            raise ValueError("Infinite calendar factor value")
        return c.out(out)

    return compute


