"""R38 ownership and limit-breadth factors.

The holder-number state machine follows the frozen research contract:
only standard quarter-ends, ``ann_date`` is the visibility date, a
restatement takes effect only from its own announcement, and adjacent
standard periods with 0 < gap <= 200 days form a change pair.  The limit-up
breadth is the 20-session mean of the source-day industry event share,
shifted one session and mapped through the static stock_list industry
crosswalk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = (
    "own_holder_chg_vol_8q",
    "own_holder_level_pct_12q",
    "lu_industry_breadth_20",
)


def _dates(ctx):
    return np.asarray(ctx.panel.dates, dtype=np.int64)


def _holder_state(ctx):
    dates = _dates(ctx)
    S, C = ctx.panel.shape
    lo = max(2010, int(dates[0]) // 10000 - 8)
    hi = int(dates[-1]) // 10000
    frame = ctx.dataset(
        "stock_holder_number",
        columns=["stock_code", "ann_date", "end_date", "holder_num"],
        years=(lo, hi),
    )
    if frame is None or frame.empty:
        return np.full(ctx.panel.shape, np.nan), np.full(ctx.panel.shape, np.nan)
    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    ci = ctx.code_index(frame["stock_code"].to_numpy())
    ann = ctx.date_col(frame["ann_date"]).astype(np.int64)
    end = ctx.date_col(frame["end_date"]).astype(np.int64)
    num = pd.to_numeric(frame["holder_num"], errors="coerce").to_numpy(np.float64)
    md = end % 10000
    keep = (ci >= 0) & (ann >= 19900101) & (end >= 19900101) & (num > 0)
    keep &= np.isin(md, (331, 630, 930, 1231))
    if not keep.any():
        return np.full(ctx.panel.shape, np.nan), np.full(ctx.panel.shape, np.nan)
    ci = ci[keep]
    ann = ann[keep]
    end = end[keep]
    num = num[keep]
    eff = np.searchsorted(dates, ann, side="right")
    usable = eff < S
    ci, ann, end, num, eff = ci[usable], ann[usable], end[usable], num[usable], eff[usable]
    # period gaps are real calendar days, not YYYYMMDD integer differences
    end_epoch = pd.to_datetime(end.astype(str), format="%Y%m%d", errors="raise").to_numpy(
        dtype="datetime64[D]").astype(np.int64)
    order = np.lexsort((end_epoch, ann, eff, ci))
    ci, ann, end_epoch, num, eff = ci[order], ann[order], end_epoch[order], num[order], eff[order]

    h1 = np.full((S, C), np.nan, dtype=np.float64)
    h3 = np.full((S, C), np.nan, dtype=np.float64)
    start = 0
    while start < len(ci):
        stop = start + 1
        while stop < len(ci) and ci[stop] == ci[start]:
            stop += 1
        c = int(ci[start])
        state: dict[int, tuple[float, int]] = {}
        prev_k = None
        prev_vals = None
        kpos = start
        # research groups all rows by effective session and carries the state
        while kpos < stop:
            k = int(eff[kpos])
            while kpos < stop and int(eff[kpos]) == k:
                state[int(end_epoch[kpos])] = (float(num[kpos]), int(ann[kpos]))
                kpos += 1
            levels = sorted(state.items(), key=lambda kv: kv[0])
            changes = []
            for (e0, (n0, _a0)), (e1, (n1, a1)) in zip(levels[:-1], levels[1:]):
                gap = e1 - e0
                if 0 < gap <= 200 and n0 > 0 and n1 > 0:
                    changes.append((float(np.log(n1 / n0)), int(a1)))
            v1 = np.nan
            if len(changes) >= 4:
                ds = np.asarray([x[0] for x in changes[-8:]], dtype=np.float64)
                if ds.size >= 2:
                    v1 = float(np.std(ds, ddof=1))
            v3 = np.nan
            if len(levels) >= 6:
                lev = np.asarray([x[1][0] for x in levels[-12:]], dtype=np.float64)
                n0 = lev[-1]
                v3 = float(((lev < n0).sum() + 0.5 * (lev == n0).sum()) / lev.size)
            if prev_k is not None:
                h1[prev_k:k, c] = prev_vals[0]
                h3[prev_k:k, c] = prev_vals[1]
            prev_k = k
            prev_vals = (v1, v3)
        if prev_k is not None:
            h1[prev_k:, c] = prev_vals[0]
            h3[prev_k:, c] = prev_vals[1]
        start = stop
    return h1, h3


def _holder_chg_vol(ctx):
    return _holder_state(ctx)[0]


def _holder_level_pct(ctx):
    return _holder_state(ctx)[1]


def _industry_groups(ctx):
    listing = ctx.dataset("stock_list", columns=["stock_code", "industry"])
    if listing is None or listing.empty:
        raise RuntimeError("stock_list industry is empty")
    listing = listing.copy()
    listing["stock_code"] = listing["stock_code"].astype(str)
    listing = listing.drop_duplicates("stock_code")
    mp = dict(zip(listing["stock_code"], listing["industry"]))
    labels = []
    for code in np.asarray(ctx.panel.codes).astype(str):
        v = mp.get(code)
        labels.append("" if v is None or not isinstance(v, str) else v.strip())
    if any(v == "" for v in labels):
        raise RuntimeError("unmapped frozen-pool industry")
    names = sorted(set(labels))
    gid = np.asarray([names.index(v) for v in labels], dtype=np.int64)
    return gid, len(names)


def _limit_breadth(ctx):
    dates = _dates(ctx)
    S, C = ctx.panel.shape
    lo = max(2010, int(dates[0]) // 10000 - 2)
    hi = int(dates[-1]) // 10000
    frame = ctx.dataset(
        "stock_limit_up",
        columns=["trade_date", "stock_code", "is_limit_up"],
        years=(lo, hi),
    )
    if frame is None or frame.empty:
        return np.full(ctx.panel.shape, np.nan)
    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    ci = ctx.code_index(frame["stock_code"].to_numpy())
    day = ctx.date_col(frame["trade_date"]).astype(np.int64)
    flag = pd.to_numeric(frame["is_limit_up"], errors="coerce").to_numpy(np.float64)
    keep = (ci >= 0) & np.isfinite(flag) & (flag == 1.0)
    if not keep.any():
        return np.full(ctx.panel.shape, np.nan)
    ci = ci[keep]
    day = day[keep]
    si = np.searchsorted(dates, day, side="left")
    ok = (si < S) & (dates[np.clip(si, 0, S - 1)] == day)
    si, ci = si[ok], ci[ok]
    gid, ng = _industry_groups(ctx)
    counts = np.zeros((S, ng), dtype=np.float64)
    for s, c in zip(si, ci):
        counts[int(s), int(gid[int(c)])] += 1.0
    denom = np.bincount(gid, minlength=ng).astype(np.float64)
    share = counts / denom[None, :]
    shifted = np.full((S, ng), np.nan, dtype=np.float64)
    if S >= 2:
        shifted[1:] = share[:-1]
    roll = pd.DataFrame(shifted).rolling(20, min_periods=20).mean().to_numpy(np.float64)
    return roll[:, gid]


def build(parameters):
    variant = (parameters or {}).get("variant")
    if variant not in NAMES:
        raise KeyError(f"unknown R38 ownership variant: {variant!r}")

    def compute(ctx):
        if variant == "own_holder_chg_vol_8q":
            return _holder_chg_vol(ctx)
        if variant == "own_holder_level_pct_12q":
            return _holder_level_pct(ctx)
        return _limit_breadth(ctx)

    return compute
