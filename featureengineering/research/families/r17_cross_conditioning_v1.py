"""R17 seat_cross batch 4: cross-sectional state-conditioned responses (C + GC).

Economic object
---------------
A stock's *conditional response coefficient* is the slope of its next-day
return on a lagged signal (5d reversal or 20d momentum), estimated over the
trailing 252 sessions separately in high and low states of a stock-level
characteristic.  The factor is the response in the economically relevant state
(negated for reversal legs), not a product of two existing factors: the value
is an estimated time-series regression coefficient for each stock.

States are cross-sectional, observable at t-1: turnover, Amihud illiquidity,
realised volatility, market cap, drawdown from the 52-week high, market beta
and earnings yield.  Signals are trailing 5d/20d mean returns ending at t-1;
the outcome is the return at t.  All rolling estimates use only rows <= t.

Mechanisms
----------
Reversal legs: high-turnover / illiquid / high-vol / small-cap / drawdown /
high-beta / high-EP states proxy for liquidity provision, inventory risk or
overreaction; a stronger estimated reversal in that state is a compensated
liquidity/overreaction premium (declared positive after the slope negates the
reversal).  Momentum legs: low-vol / low-turnover / large-cap / near-high /
low-illiquidity / high-EP states proxy for gradual information diffusion; a
stronger momentum slope in that state is declared positive.  High-turnover
momentum is expected to reverse (attention crowding), so the negated slope is
declared positive.

GC legs are the unconditional reversal and momentum response coefficients
(descriptive: how strongly the stock's own past return predicts its next
return), with no predictive claim.
"""
from __future__ import annotations

import warnings

import numpy as np

WARMUP_DAYS = 1200
WIN = 252
MIN_STATE_OBS = 40
MIN_OBS = 80
MIN_STOCKS = 100
INDEX = "000300.SH"

C_VARIANTS = (
    "cx_rev_turnover_high",
    "cx_rev_illiq_high",
    "cx_rev_vol_high",
    "cx_rev_smallcap",
    "cx_rev_drawdown",
    "cx_rev_beta_high",
    "cx_mom_lowvol",
    "cx_mom_lowturnover",
    "cx_mom_largecap",
    "cx_mom_nearhigh",
    "cx_mom_lowilliq",
    "cx_value_rev",
    "cx_value_mom",
    "cx_small_liq_rev",
    "cx_high_turnover_mom",
)
GC_VARIANTS = (
    "cx_rev_slope_uncond",
    "cx_mom_slope_uncond",
)
PARAMETERS = {name: {"variant": name} for name in C_VARIANTS + GC_VARIANTS}
ALLOWED_VARIANTS = tuple(PARAMETERS[name] for name in PARAMETERS)

_CACHE_KEY = "_r17_cross_conditioning_v1"


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


def _shift1(a):
    out = np.full_like(a, np.nan, dtype=np.float64)
    out[1:] = a[:-1]
    return out


def _cs_state(Z, above=True, min_stocks=MIN_STOCKS):
    """Cross-sectional median split, NaN where the row is too sparse."""
    fin = np.isfinite(Z)
    cnt = fin.sum(axis=1)
    med = np.full(Z.shape[0], np.nan)
    good = cnt >= min_stocks
    if good.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med[good] = np.nanmedian(Z[good], axis=1)
    with np.errstate(invalid="ignore"):
        cmp = (Z > med[:, None]) if above else (Z < med[:, None])
    return np.where(fin & good[:, None], cmp.astype(np.float64), np.nan)


def _cond_slope(X, Y, state, n=WIN, min_obs=MIN_STATE_OBS):
    """Slope of Y on X on state-high and state-low days, trailing n rows."""
    ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(state)
    hi = ok & (state > 0.5)
    lo = ok & (state < 0.5)

    def slope(mask):
        I = np.where(mask, 1.0, np.nan)
        Xm = np.where(mask, X, np.nan)
        Ym = np.where(mask, Y, np.nan)
        cnt = _roll_sum(I, n, min_obs)
        sx = _roll_sum(Xm, n, min_obs)
        sy = _roll_sum(Ym, n, min_obs)
        sxx = _roll_sum(Xm * Xm, n, min_obs)
        sxy = _roll_sum(Xm * Ym, n, min_obs)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = sx / cnt
            my = sy / cnt
            cov = sxy / cnt - mx * my
            var = sxx / cnt - mx * mx
            out = cov / var
        out[~(np.isfinite(out)) | (var <= 1e-10)] = np.nan
        return out

    return slope(hi), slope(lo)


def _uncond_slope(X, Y, n=WIN, min_obs=MIN_OBS):
    ok = np.isfinite(X) & np.isfinite(Y)
    I = np.where(ok, 1.0, np.nan)
    Xm = np.where(ok, X, np.nan)
    Ym = np.where(ok, Y, np.nan)
    cnt = _roll_sum(I, n, min_obs)
    sx = _roll_sum(Xm, n, min_obs)
    sy = _roll_sum(Ym, n, min_obs)
    sxx = _roll_sum(Xm * Xm, n, min_obs)
    sxy = _roll_sum(Xm * Ym, n, min_obs)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = sx / cnt
        my = sy / cnt
        cov = sxy / cnt - mx * my
        var = sxx / cnt - mx * mx
        out = cov / var
    out[~(np.isfinite(out)) | (var <= 1e-10)] = np.nan
    return out


def _roll_beta_mkt(y, x1d, n, min_count):
    x = np.asarray(x1d, dtype=np.float64)[:, None]
    ok = np.isfinite(y) & np.isfinite(x)
    Y = np.where(ok, y, 0.0)
    X = np.where(ok, x, 0.0)
    cnt = _roll_sum(ok.astype(np.float64), n, min_count)
    sy = _roll_sum(Y, n, min_count)
    sx = _roll_sum(X, n, min_count)
    syy = _roll_sum(Y * Y, n, min_count)
    sxx = _roll_sum(X * X, n, min_count)
    sxy = _roll_sum(Y * X, n, min_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        my = sy / cnt
        mx = sx / cnt
        cov = sxy / cnt - mx * my
        var = sxx / cnt - mx * mx
        out = cov / var
    out[~(np.isfinite(out)) | (var <= 1e-12)] = np.nan
    return out


def _index_ret(ctx, days):
    out = np.full(len(days), np.nan)
    y0, y1 = int(days[0]) // 10000, int(days[-1]) // 10000
    df = ctx.dataset("index_daily", columns=["ts_code", "trade_date", "pct_chg"],
                     years=(y0, y1))
    if df is None or df.empty:
        return out
    df = df.loc[df["ts_code"] == INDEX]
    if df.empty:
        return out
    d = ctx.date_col(df["trade_date"])
    v = np.asarray(df["pct_chg"], dtype=np.float64) / 100.0
    pos = np.searchsorted(days, d)
    ok = (pos < len(days)) & (days[np.minimum(pos, len(days) - 1)] == d)
    out[pos[ok]] = v[ok]
    return out


class _Cx:
    def __init__(self, ctx):
        cache = ctx.__dict__.setdefault(_CACHE_KEY, {})
        if cache.get("ready"):
            self.__dict__.update(cache["obj"])
            self.states = cache.setdefault("states", {})
            return
        self.ctx = ctx
        days = np.asarray(ctx.panel.dates, dtype=np.int64)
        r = np.array(ctx.ret_clean(1), dtype=np.float64, copy=True)
        tr = np.asarray(ctx.traded(), dtype=bool)
        univ = np.asarray(ctx.universe, dtype=bool)
        adj = tr.copy()
        adj[0] = False
        adj[1:] &= tr[:-1]
        r[~adj] = np.nan
        self.r = r
        self.tr = tr
        self.univ = univ
        self.states = cache.setdefault("states", {})

        self.R5 = _shift1(ctx.roll_mean(r, 5, 4))
        self.R20 = _shift1(ctx.roll_mean(r, 20, 16))
        self.market = _index_ret(ctx, days)

        turn = np.array(ctx.px("turnover"), dtype=np.float64, copy=True)
        turn[~tr] = np.nan
        self.turn = turn
        amount = np.array(ctx.px("amount"), dtype=np.float64, copy=True)
        amount[~tr] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            illiq_d = np.where(np.isfinite(amount) & (amount > 0),
                               np.abs(r) / amount, np.nan)
        self.illiq = ctx.roll_mean(illiq_d, 20, 10)
        self.vol = ctx.roll_std(r, 60, 40)
        close = np.asarray(ctx.px("close"), dtype=np.float64)
        tshare = np.asarray(ctx.px("total_share"), dtype=np.float64)
        mv = close * tshare
        logmv = np.log(np.where(mv > 0, mv, np.nan))
        self.logmv = np.where(tr, logmv, np.nan)
        np_ = np.asarray(ctx.ttm("n_income_attr_p"), dtype=np.float64)
        self.ep = np.where(tr, ctx.safe_div(np_, mv, min_abs_den=1e6), np.nan)
        hfq = np.asarray(ctx.hfq("close"), dtype=np.float64)
        ratio52 = ctx.safe_div(hfq, ctx.roll_max(hfq, 250, 200),
                               min_abs_den=1e-12)
        self.ratio52 = np.where(tr, ratio52, np.nan)
        self.beta = _roll_beta_mkt(r, self.market, 60, 40)

        cache["obj"] = {k: v for k, v in self.__dict__.items()
                        if k != "states"}
        cache["ready"] = True

    def state(self, name):
        st = self.states
        if name in st:
            return st[name]
        if name == "turn_high":
            v = _cs_state(self.turn, True)
        elif name == "illiq_high":
            v = _cs_state(self.illiq, True)
        elif name == "illiq_low":
            v = _cs_state(self.illiq, False)
        elif name == "vol_high":
            v = _cs_state(self.vol, True)
        elif name == "vol_low":
            v = _cs_state(self.vol, False)
        elif name == "smallcap":
            v = _cs_state(self.logmv, False)
        elif name == "largecap":
            v = _cs_state(self.logmv, True)
        elif name == "drawdown":
            v = np.where(np.isfinite(self.ratio52),
                         (self.ratio52 < 0.80).astype(np.float64), np.nan)
        elif name == "nearhigh":
            v = np.where(np.isfinite(self.ratio52),
                         (self.ratio52 > 0.90).astype(np.float64), np.nan)
        elif name == "beta_high":
            v = _cs_state(self.beta, True)
        elif name == "ep_high":
            v = _cs_state(self.ep, True)
        elif name == "turn_low":
            v = _cs_state(self.turn, False)
        else:
            raise KeyError(name)
        st[name] = _shift1(v)
        return st[name]

    def pair_state(self, a, b):
        key = a + "__" + b
        st = self.states
        if key in st:
            return st[key]
        va, vb = self.state(a), self.state(b)
        st[key] = np.where(np.isfinite(va) & np.isfinite(vb),
                           ((va > 0.5) & (vb > 0.5)).astype(np.float64), np.nan)
        return st[key]

    def out(self, values):
        return np.where(self.tr & self.univ, values, np.nan)


def _compute(ctx, variant):
    c = _Cx(ctx)
    R5, R20, r = c.R5, c.R20, c.r

    def rev(state):
        hi, _lo = _cond_slope(R5, r, state)
        return -hi

    def mom(state):
        _hi, lo = _cond_slope(R20, r, state)
        return lo

    if variant == "cx_rev_turnover_high":
        out = rev(c.state("turn_high"))
    elif variant == "cx_rev_illiq_high":
        out = rev(c.state("illiq_high"))
    elif variant == "cx_rev_vol_high":
        out = rev(c.state("vol_high"))
    elif variant == "cx_rev_smallcap":
        out = rev(c.state("smallcap"))
    elif variant == "cx_rev_drawdown":
        out = rev(c.state("drawdown"))
    elif variant == "cx_rev_beta_high":
        out = rev(c.state("beta_high"))
    elif variant == "cx_mom_lowvol":
        out = mom(c.state("vol_low"))
    elif variant == "cx_mom_lowturnover":
        out = mom(c.state("turn_low"))
    elif variant == "cx_mom_largecap":
        out = mom(c.state("largecap"))
    elif variant == "cx_mom_nearhigh":
        out = mom(c.state("nearhigh"))
    elif variant == "cx_mom_lowilliq":
        out = mom(c.state("illiq_low"))
    elif variant == "cx_value_rev":
        out = rev(c.state("ep_high"))
    elif variant == "cx_value_mom":
        out = mom(c.state("ep_high"))
    elif variant == "cx_small_liq_rev":
        out = rev(c.pair_state("smallcap", "turn_high"))
    elif variant == "cx_high_turnover_mom":
        hi, _lo = _cond_slope(R20, r, c.state("turn_high"))
        out = -hi
    elif variant == "cx_rev_slope_uncond":
        out = _uncond_slope(R5, r)
    elif variant == "cx_mom_slope_uncond":
        out = _uncond_slope(R20, r)
    else:
        raise KeyError(variant)
    if np.isinf(out).any():
        raise ValueError("Infinite conditioning factor value")
    return c.out(out)


def build(parameters):
    if parameters not in ALLOWED_VARIANTS:
        raise ValueError("Only frozen batch-4 conditioning variants are allowed")
    variant = parameters["variant"]

    def compute(ctx):
        return _compute(ctx, variant)

    return compute
