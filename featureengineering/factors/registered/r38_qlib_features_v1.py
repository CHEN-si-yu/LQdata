"""R38 Qlib-derived factor: daily return / log-volume-change correlation (CORD60).

CORD60 = corr(r_t, log(V_t/V_{t-1}+1)) over 60 trailing slots, pairwise
complete, at least 48 valid pairs.  Returns use hfq close and require real
trades at both endpoints; volume is raw and both endpoints must be positive.
"""
from __future__ import annotations

import numpy as np

NAMES = (
    "ql_cord60_v1",
    "ql_max20_v1",
    "ql_imxd60_v1",
)
WINDOW = 60
MIN_PAIRS = 48


def _cord(ctx):
    """CORD60: corr(return, log1p(volume ratio)) over 60 trailing slots."""
    hfq = ctx.hfq("close")
    vol = ctx.px_raw("vol")
    traded = ctx.traded()
    prev_tr = ctx.shift(traded.astype(np.float64), 1) > 0.5
    both = traded & prev_tr
    prev_hfq = ctx.shift(hfq, 1)
    prev_vol = ctx.shift(vol, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = np.where(
            both & np.isfinite(hfq) & np.isfinite(prev_hfq) & (prev_hfq > 0.0),
            hfq / np.where(prev_hfq > 0.0, prev_hfq, 1.0) - 1.0,
            np.nan,
        )
        vr = np.where(
            both & np.isfinite(vol) & np.isfinite(prev_vol)
            & (vol > 0.0) & (prev_vol > 0.0),
            vol / np.where(prev_vol > 0.0, prev_vol, 1.0),
            np.nan,
        )
    y = np.log1p(vr)
    out = ctx.roll_corr(r1, y, WINDOW, min_count=MIN_PAIRS)
    return np.where(np.isfinite(out), np.clip(out, -1.0, 1.0), np.nan)


def _hfq_product(ctx, field):
    """Research panel value: (raw field * adj_factor) rounded to float32."""
    raw = np.asarray(ctx.px_raw(field), dtype=np.float64)
    adj = np.asarray(ctx.px_raw("adj_factor"), dtype=np.float64)
    prod = np.where(np.isfinite(raw) & np.isfinite(adj) & (raw > 0.0) & (adj > 0.0),
                    raw * adj, np.nan)
    return prod.astype(np.float32).astype(np.float64)


def _max20(ctx):
    h = _hfq_product(ctx, "high")
    c = _hfq_product(ctx, "close")
    T, C = h.shape
    mx = np.full((T, C), np.nan, dtype=np.float64)
    cnt = np.zeros((T, C), dtype=np.float64)
    for k in range(20):
        v = np.full((T, C), np.nan, dtype=np.float64)
        if k < T:
            v[k:] = h[:T - k]
        valid = np.isfinite(v)
        cnt += valid.astype(np.float64)
        mx = np.where(valid & (~np.isfinite(mx) | (v > mx)), v, mx)
    out = np.where((cnt >= 16.0) & np.isfinite(c) & (c != 0.0),
                   mx / np.where(np.isfinite(c) & (c != 0.0), c, 1.0) - 1.0, np.nan)
    return out


def _imxd60(ctx):
    h = _hfq_product(ctx, "high")
    l = _hfq_product(ctx, "low")
    T, C = h.shape
    best_h = np.full((T, C), -np.inf, dtype=np.float64)
    best_l = np.full((T, C), np.inf, dtype=np.float64)
    lag_h = np.full((T, C), np.nan, dtype=np.float64)
    lag_l = np.full((T, C), np.nan, dtype=np.float64)
    cnt = np.zeros((T, C), dtype=np.float64)
    for k in range(60):
        hh = np.full((T, C), np.nan, dtype=np.float64)
        ll = np.full((T, C), np.nan, dtype=np.float64)
        if k < T:
            hh[k:] = h[:T - k]
            ll[k:] = l[:T - k]
        valid = np.isfinite(hh) & np.isfinite(ll)
        cnt += valid.astype(np.float64)
        up = valid & (hh > best_h)
        best_h = np.where(up, hh, best_h)
        lag_h = np.where(up, float(k), lag_h)
        dn = valid & (ll < best_l)
        best_l = np.where(dn, ll, best_l)
        lag_l = np.where(dn, float(k), lag_l)
    ok = cnt >= 48.0
    imax = (60.0 - lag_h) / 60.0
    imin = (60.0 - lag_l) / 60.0
    return np.where(ok, imax - imin, np.nan)


def build(parameters):
    variant = (parameters or {}).get("variant") or "ql_cord60_v1"
    if variant not in NAMES:
        raise KeyError(f"unknown R38 qlib variant: {variant!r}")

    def compute(ctx):
        if variant == "ql_cord60_v1":
            return _cord(ctx)
        if variant == "ql_max20_v1":
            return _max20(ctx)
        return _imxd60(ctx)

    return compute
