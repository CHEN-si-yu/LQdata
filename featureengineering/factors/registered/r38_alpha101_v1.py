"""R38 Alpha101/191 factors on the frozen A-share daily panel.

Four price/volume factors whose source formulas are dimensionless or are
adapted only through a documented return/ratio transform.  All prices are
hfq = raw * cumulative adj_factor; both endpoints of every return must be
real traded bars.  Cross-sectional ranks are average-tie percentiles within
each date's finite set, matching the frozen research implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = (
    "af_a101_009_monotone_trend_5",
    "af_a101_012_volume_signed_reversal_1",
    "af_a101_038_tsrank_body_overbought_10",
    "af_gtja_090_vwap_volume_rank_corr_5",
)


def _cs_rank(x: np.ndarray) -> np.ndarray:
    return pd.DataFrame(x).rank(axis=1, pct=True, method="average").to_numpy(np.float64)


def _bars(ctx):
    """Research-equivalent daily panels: valid OHLCV/amount/adj bars only."""
    o = ctx.px_raw("open")
    h = ctx.px_raw("high")
    l = ctx.px_raw("low")
    c = ctx.px_raw("close")
    v = ctx.px_raw("vol")
    a = ctx.px_raw("amount")
    adj = ctx.px_raw("adj_factor")
    finite = (np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
              & np.isfinite(v) & np.isfinite(a) & np.isfinite(adj))
    traded = finite & (o > 0.0) & (h > 0.0) & (l > 0.0) & (c > 0.0) \
        & (v > 0.0) & (a > 0.0) & (adj > 0.0)
    ok_bar = traded & (h >= np.maximum(o, c) - 1e-12) \
        & (l <= np.minimum(o, c) + 1e-12)
    safe_adj = np.where(traded, adj, np.nan)
    nan = np.full(c.shape, np.nan, dtype=np.float64)
    return {
        "O": np.where(ok_bar, o, nan),
        "H": np.where(ok_bar, h, nan),
        "L": np.where(ok_bar, l, nan),
        "C": np.where(ok_bar, c, nan),
        "V": np.where(traded, v, nan),
        "A": np.where(traded, a, nan),
        "adj": safe_adj,
        "Cs": np.where(ok_bar, c * safe_adj, nan),
        "Hs": np.where(ok_bar, h * safe_adj, nan),
        "Ls": np.where(ok_bar, l * safe_adj, nan),
        "Qs": np.where(traded & (v > 0.0), (a / v) * safe_adj, nan),
        "traded": traded,
        "ok_bar": ok_bar,
    }


def _returns(p):
    prev = np.roll(p["Cs"], 1, axis=0)
    prev[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        r = p["Cs"] / prev - 1.0
    return np.where(np.isfinite(p["Cs"]) & np.isfinite(prev), r, np.nan)


def _ts_rank_last(x: np.ndarray, n: int) -> np.ndarray:
    """Pandas ``rolling(n, min_periods=n).rank(pct=True).iloc[-1]`` equivalence.

    The percentile is the average 1-based rank of the window's last value
    divided by the number of observations (pandas' tie convention).
    """
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] < n:
        return out
    for t in range(n - 1, x.shape[0]):
        w = x[t - n + 1:t + 1]
        last = w[-1]
        ok = np.isfinite(w)
        cnt = ok.sum(axis=0)
        less = np.sum(ok & (w < last), axis=0)
        equal = np.sum(ok & (w == last), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            val = (less + (equal + 1.0) / 2.0) / cnt
        out[t] = np.where((cnt == n) & np.isfinite(last), val, np.nan)
    return out


def _monotone(ctx):
    p = _bars(ctx)
    r = _returns(p)
    nvalid = ctx.roll_count(np.where(np.isfinite(r), 1.0, np.nan), 5)
    lo = ctx.roll_min(r, 5, min_count=5)
    hi = ctx.roll_max(r, 5, min_count=5)
    up = np.isfinite(nvalid) & (nvalid >= 5.0) & np.isfinite(lo) & (lo > 0.0)
    dn = np.isfinite(nvalid) & (nvalid >= 5.0) & np.isfinite(hi) & (hi < 0.0)
    val = np.where(up | dn, r, -r)
    return np.where(np.isfinite(nvalid) & (nvalid >= 5.0) & np.isfinite(r), val, np.nan)


def _volume_signed(ctx):
    p = _bars(ctx)
    r = _returns(p)
    v = p["V"]
    prev_v = np.roll(v, 1, axis=0)
    prev_v[0] = np.nan
    with np.errstate(invalid="ignore"):
        dv = np.sign(v - prev_v)
    dv = np.where(np.isfinite(v) & np.isfinite(prev_v), dv, np.nan)
    return np.where(np.isfinite(dv) & np.isfinite(r), dv * (-r), np.nan)


def _tsrank_body(ctx):
    p = _bars(ctx)
    a = _ts_rank_last(p["Cs"], 10)
    body = np.where(p["ok_bar"] & (p["C"] > 0.0) & (p["O"] > 0.0),
                    p["C"] / p["O"], np.nan)
    ra = _cs_rank(a)
    rb = _cs_rank(body)
    return -(ra * rb)


def _vwap_rank_corr(ctx):
    p = _bars(ctx)
    qrank = _cs_rank(p["Qs"])
    vrank = _cs_rank(p["V"])
    # Pandas pairwise-complete rolling correlation is the frozen research
    # definition (including its exact treatment of tied/constant windows).
    corr = pd.DataFrame(qrank).rolling(5, min_periods=4).corr(
        pd.DataFrame(vrank)).to_numpy(np.float64)
    return -_cs_rank(corr)


def build(parameters):
    variant = (parameters or {}).get("variant")
    if variant not in NAMES:
        raise KeyError(f"unknown R38 alpha variant: {variant!r}")

    def compute(ctx):
        if variant == "af_a101_009_monotone_trend_5":
            return _monotone(ctx)
        if variant == "af_a101_012_volume_signed_reversal_1":
            return _volume_signed(ctx)
        if variant == "af_a101_038_tsrank_body_overbought_10":
            return _tsrank_body(ctx)
        return _vwap_rank_corr(ctx)

    return compute
