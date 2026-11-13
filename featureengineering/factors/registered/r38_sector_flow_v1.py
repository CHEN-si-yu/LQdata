"""R38 sector leader/follower fund-flow factor.

Within each static ``stock_list.industry`` group, the leaders are the top
ceil(10%) of members by 60-day mean amount.  Leader fund-flow intensity
(5-day net main inflow / gross two-way amount) minus the non-leader median,
multiplied by each member's 20-day return gap to its group median, measures
the leader-led catch-up state.  All windows end at T and use only source data
available at T close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "sec_leader_catchup_20"
FLOW_COLS = (
    "buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount",
    "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount",
    "net_mf_amount",
)


def _place(ctx, frame, col):
    idx = np.asarray(ctx.code_index(frame["stock_code"].astype(str).to_numpy()), dtype=np.int64)
    day = np.asarray(ctx.date_col(frame["trade_date"]), dtype=np.int64)
    val = pd.to_numeric(frame[col], errors="coerce").to_numpy(np.float64)
    keep = idx >= 0
    return ctx.panel.place(idx[keep], day[keep], val[keep])


def _groups(ctx):
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
    if len(names) != 110:
        raise RuntimeError(f"expected 110 stock_list industries, got {len(names)}")
    gid = np.asarray([names.index(v) for v in labels], dtype=np.int64)
    return gid, len(names)


def _rmean(a, w, m):
    return pd.DataFrame(a).rolling(w, min_periods=m).mean().to_numpy(np.float64)


def _rsum_masked(a, valid, w, m):
    x = pd.DataFrame(np.where(valid, a, np.nan))
    return x.rolling(w, min_periods=m).sum().to_numpy(np.float64)


def _rcount(valid, w, m):
    return pd.DataFrame(valid.astype(np.float64)).rolling(w, min_periods=m).sum().to_numpy(np.float64)


def _compute(ctx):
    dates = np.asarray(ctx.panel.dates, dtype=np.int64)
    S, C = ctx.panel.shape
    hi = int(dates[-1]) // 10000
    lo = max(2010, int(dates[0]) // 10000 - 1)
    frame = ctx.dataset("stock_main_fund_flow",
                        columns=["trade_date", "stock_code", *FLOW_COLS],
                        years=(lo, hi))
    if frame is None or frame.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    flow = {c: _place(ctx, frame, c) for c in FLOW_COLS}
    amount = np.asarray(ctx.px_raw("amount"), dtype=np.float64)
    close = np.asarray(ctx.px_raw("close"), dtype=np.float64)
    adj = np.asarray(ctx.px_raw("adj_factor"), dtype=np.float64)
    hc = np.where(np.isfinite(close) & np.isfinite(adj) & (close > 0) & (adj > 0),
                  close * adj, np.nan)
    vol = np.asarray(ctx.px_raw("vol"), dtype=np.float64)
    traded = ctx.traded() & np.isfinite(close) & (close > 0) & (vol > 0)

    bsum = sum(flow[f"buy_{t}_amount"] for t in ("sm", "md", "lg", "elg"))
    ssum = sum(flow[f"sell_{t}_amount"] for t in ("sm", "md", "lg", "elg"))
    gross = bsum + ssum
    valid = np.isfinite(gross) & (gross > 0)
    net = np.asarray(flow["net_mf_amount"], dtype=np.float64)
    x_net = np.where(valid & np.isfinite(net), net / np.where(valid, gross, np.nan), np.nan)

    amount60 = _rmean(amount, 60, 40)
    flow5_num = _rsum_masked(net, valid & np.isfinite(net), 5, 4)
    flow5_den = _rsum_masked(gross, valid, 5, 4)
    flow5_cnt = _rcount(valid & np.isfinite(net), 5, 4)
    flow5 = np.where((flow5_den > 0) & (flow5_cnt >= 4),
                     flow5_num / np.where(flow5_den > 0, flow5_den, np.nan),
                     np.nan)
    prev20 = np.full((S, C), np.nan, dtype=np.float64)
    if S > 20:
        prev20[20:] = hc[:-20]
    r20 = np.where(np.isfinite(hc) & np.isfinite(prev20) & (prev20 != 0),
                   hc / np.where(prev20 != 0, prev20, np.nan) - 1.0, np.nan)

    gid, G = _groups(ctx)
    gap = np.full((S, G), np.nan, dtype=np.float64)
    lag = np.full((S, C), np.nan, dtype=np.float64)
    for g in range(G):
        idx = np.where(gid == g)[0]
        if idx.size == 0:
            continue
        a = amount60[:, idx]
        av = np.isfinite(a)
        nvalid = av.sum(axis=1)
        for t in np.where(nvalid >= 8)[0]:
            ii = idx[av[t]]
            if ii.size < 8:
                continue
            k = max(1, int(np.ceil(0.1 * ii.size)))
            order = np.argsort(-a[t, av[t]], kind="stable")
            lead = ii[order[:k]]
            non = ii[order[k:]]
            fl = flow5[t, lead]
            fn = flow5[t, non]
            fl = fl[np.isfinite(fl)]
            fn = fn[np.isfinite(fn)]
            if fl.size == 0 or fn.size == 0:
                continue
            gap[t, g] = np.median(fl) - np.median(fn)
        rr = r20[:, idx]
        with np.errstate(all="ignore"):
            med = np.nanmedian(rr, axis=1)
        ok = np.isfinite(med)
        lag[:, idx] = np.where(ok[:, None], med[:, None] - rr, np.nan)

    prod = gap[:, gid] * lag
    out = _rmean(prod, 20, 15)
    return np.where(traded, out, np.nan)


def build(parameters):
    if parameters and parameters.get("variant") not in (None, NAME):
        raise KeyError(f"unknown R38 sector-flow variant: {parameters!r}")

    def compute(ctx):
        return _compute(ctx)

    return compute
