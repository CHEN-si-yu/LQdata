"""R15 T3 family builder: session allocation of movement (overnight vs intraday).

One parameterized pure builder for the r15t3_session_allocation_v1 family.
All objects read the same two legs on actual traded days of a fixed window:

    r_on(d) = log(hfq_open(d)) - log(hfq_close(d-1))     (overnight / non-trading)
    r_id(d) = log(hfq_close(d)) - log(hfq_open(d))       (intraday / trading hours)

both in the project's continuous hfq close/open basis (adjacent hfq-close
differences, consistent with P013/P014 path returns). Strict endpoint rule:
the overnight leg needs d and d-1 actual trades plus finite positive hfq_open,
hfq_close(d-1), raw_open, raw_close; the intraday leg needs d traded with
finite positive hfq_open, hfq_close, raw_open, raw_close. No ffill inside a
leg; a failing endpoint excludes that day without time compression.
log(hfq_close(d))-log(hfq_close(d-1)) = r_on(d)+r_id(d) exactly.

Declared objects (one candidate each, lookup table - no window sweeps, no
tuned thresholds; economically independent measurement objects, complementary
identity legs are NOT separately registered):

    overnight_variance_share      sum(r_on^2)/(sum(r_on^2)+sum(r_id^2))
    leg_correlation               corr(r_on(d), r_id(d))
    overnight_to_intraday_spillover   corr(r_id(d-1), r_on(d))
    gap_fill_ratio                share of nonzero-gap days with the gap at
                                  least a quarter retraced intraday
    overnight_tail_share          share of overnight squared movement in the
                                  top-quartile |r_on| days by rank per window
                                  (m=ceil((1-q)*valid); ties by day order;
                                  deterministic in degenerate windows)

min_valid_days=40 observations entering the statistic per window (pairs for
the spillover object). Output is NaN unless traded and in universe at t.
screening only; physical descriptors, no alpha direction claim.
"""
import numpy as np

WINDOW = 60
MIN_VALID_DAYS = 40
ENDPOINT_RULE = "both_legs_actual_trades_strict"
GAP_RETRACE_FRACTION = 0.25
TAIL_QUANTILE = 0.75

OBJECTS = (
    "overnight_variance_share",
    "leg_correlation",
    "overnight_to_intraday_spillover",
    "gap_fill_ratio",
    "overnight_tail_share",
)

PARAMETER_TEMPLATE = {
    "object": OBJECTS[0],
    "window": WINDOW,
    "min_valid_days": MIN_VALID_DAYS,
    "endpoint_rule": ENDPOINT_RULE,
    "gap_retrace_fraction": GAP_RETRACE_FRACTION,
    "tail_quantile": TAIL_QUANTILE,
}

VARIANT_OVERRIDES = {}  # object is the only free key per candidate (kept for declaration clarity)


def resolve_parameters(parameters):
    p = dict(PARAMETER_TEMPLATE)
    p.update(parameters)
    if set(p) != set(PARAMETER_TEMPLATE) or set(parameters) - set(PARAMETER_TEMPLATE):
        raise ValueError("Unknown parameter keys")
    if p["object"] not in OBJECTS:
        raise ValueError("Unknown object")
    for key, expect in (("window", WINDOW), ("min_valid_days", MIN_VALID_DAYS),
                        ("endpoint_rule", ENDPOINT_RULE),
                        ("gap_retrace_fraction", GAP_RETRACE_FRACTION),
                        ("tail_quantile", TAIL_QUANTILE)):
        if p[key] != expect:
            raise ValueError("Only the declared object may vary")
    return p


def session_legs(hfq_open, hfq_close, raw_open, raw_close, traded):
    """Strict overnight and intraday legs; validity masks per day."""
    n, k = hfq_open.shape
    fin_pos = lambda a: np.isfinite(a) & (a > 0)
    on_ok = np.zeros((n, k), bool)
    id_ok = np.zeros((n, k), bool)
    r_on = np.full((n, k), np.nan)
    r_id = np.full((n, k), np.nan)
    if n < 2:
        return r_on, r_id, on_ok, id_ok
    id_ok[1:] = (traded[1:] & fin_pos(hfq_open[1:]) & fin_pos(hfq_close[1:])
                 & fin_pos(raw_open[1:]) & fin_pos(raw_close[1:]))
    on_ok[1:] = (traded[1:] & traded[:-1] & fin_pos(hfq_open[1:])
                 & fin_pos(hfq_close[:-1]) & fin_pos(raw_open[1:])
                 & fin_pos(raw_close[:-1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.log(hfq_open[1:])
        lc = np.log(hfq_close[1:])
        lc_prev = np.log(hfq_close[:-1])
        r_id[1:] = np.where(id_ok[1:], lc - lo, np.nan)
        r_on[1:] = np.where(on_ok[1:], lo - lc_prev, np.nan)
    return r_on, r_id, on_ok, id_ok


def _corr(x, y, ok):
    """Column Pearson correlation over ok entries; NaN unless >=2 and both sd>0."""
    x = np.where(ok, x, 0.0)
    y = np.where(ok, y, 0.0)
    cnt = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = x.sum(0) / cnt
        my = y.sum(0) / cnt
        cxx = (x * x).sum(0) / cnt - mx * mx
        cyy = (y * y).sum(0) / cnt - my * my
        cxy = (x * y).sum(0) / cnt - mx * my
        r = cxy / np.sqrt(cxx * cyy)
    good = (cnt >= 2) & (cxx > 0) & (cyy > 0)
    return np.where(good, r, np.nan)


def _window_slice(t, window):
    return slice(t - window + 1, t + 1)


def object_series(name, r_on, r_id, day_ok, window, min_valid_days):
    n, k = r_on.shape
    out = np.full((n, k), np.nan)
    zero = np.zeros((1, k))
    for t in range(window, n):
        sl = _window_slice(t, window)
        ok = day_ok[sl]
        cnt = ok.sum(0)
        if name == "overnight_variance_share":
            ro = np.where(ok, r_on[sl], 0.0)
            ri = np.where(ok, r_id[sl], 0.0)
            s_on = (ro * ro).sum(0)
            s_id = (ri * ri).sum(0)
            den = s_on + s_id
            with np.errstate(invalid="ignore", divide="ignore"):
                v = s_on / den
            v = np.where((cnt >= min_valid_days) & (den > 0), v, np.nan)
        elif name == "leg_correlation":
            v = _corr(r_on[sl], r_id[sl], ok)
            v = np.where(cnt >= min_valid_days, v, np.nan)
        elif name == "overnight_to_intraday_spillover":
            pair_ok = np.vstack([zero, ok[:-1] & ok[1:]])
            v = _corr(np.vstack([zero, r_id[sl][:-1]]), r_on[sl], pair_ok)
            v = np.where(pair_ok.sum(0) >= min_valid_days, v, np.nan)
        elif name == "gap_fill_ratio":
            ro = np.where(ok, r_on[sl], np.nan)
            ri = np.where(ok, r_id[sl], np.nan)
            gap = ok & np.isfinite(ro) & (ro != 0)
            fill = gap & (np.sign(ri) == -np.sign(ro)) & (np.abs(ri) >= GAP_RETRACE_FRACTION * np.abs(ro))
            den = gap.sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = fill.sum(0) / den
            v = np.where((cnt >= min_valid_days) & (den > 0), v, np.nan)
        elif name == "overnight_tail_share":
            ro = np.where(ok, r_on[sl], np.nan)
            ro2 = np.where(ok, r_on[sl] * r_on[sl], np.nan)
            a = np.abs(ro)
            # rank-based top quartile: m = ceil((1-q)*valid_count) days with the
            # largest |r_on| per column (ties by day order); deterministic in
            # tied/degenerate windows. Invalid days sort last and are excluded
            # because m <= count.
            m = np.ceil((1.0 - TAIL_QUANTILE) * cnt).astype(int)
            order = np.argsort(-a, axis=0)
            ro2_sorted = np.take_along_axis(np.nan_to_num(ro2, nan=0.0), order, axis=0)
            in_tail = np.arange(ro.shape[0])[:, None] < m[None, :]
            tail = (ro2_sorted * in_tail).sum(0)
            tot = np.nan_to_num(ro2, nan=0.0).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = tail / tot
            v = np.where((cnt >= min_valid_days) & (tot > 0), v, np.nan)
        else:
            raise ValueError(name)
        out[t] = v
    return out


def build(parameters):
    p = resolve_parameters(parameters)

    def compute(ctx):
        hfq_open = np.asarray(ctx.hfq("open"), dtype=np.float64)
        hfq_close = np.asarray(ctx.hfq("close"), dtype=np.float64)
        raw_open = np.asarray(ctx.px_raw("open"), dtype=np.float64)
        raw_close = np.asarray(ctx.px_raw("close"), dtype=np.float64)
        traded = np.asarray(ctx.traded(), dtype=bool)
        universe = np.asarray(ctx.universe, dtype=bool)
        r_on, r_id, on_ok, id_ok = session_legs(hfq_open, hfq_close, raw_open, raw_close, traded)
        day_ok = on_ok & id_ok
        out = object_series(p["object"], r_on, r_id, day_ok, p["window"], p["min_valid_days"])
        out[~(universe & traded)] = np.nan
        return out

    return compute
