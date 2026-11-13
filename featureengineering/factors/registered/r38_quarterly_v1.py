"""R38 quarterly seasonal inventory/sales deviation."""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN = 1_000_000.0
MAX_AGE = 550
BASE_YEAR = 2010
PERIOD_W = 100000
KEY_W = 100000000


def _date_vec(ctx, s):
    raw = np.asarray(ctx.date_col(s), dtype=np.int64)
    return np.where((raw >= 19900101) & (raw <= 21001231), raw, 0)


def _quarter(end):
    md = end % 10000
    q = np.select((md == 331, md == 630, md == 930, md == 1231), (0, 1, 2, 3), default=-1)
    return np.where(q >= 0, end // 10000 * 4 + q, -1)


def _load(ctx, dataset, fields):
    end_year = int(ctx.panel.dates[-1]) // 10000
    cols = ["stock_code", "ann_date", "f_ann_date", "end_date",
            "comp_type", "report_type", *fields]
    fr = ctx.dataset(dataset, columns=cols, years=(BASE_YEAR, end_year))
    if fr is None or fr.empty:
        return pd.DataFrame(columns=["code", "period", "pit", "end", *fields])
    ann = _date_vec(ctx, fr["ann_date"])
    fann = _date_vec(ctx, fr["f_ann_date"])
    end = _date_vec(ctx, fr["end_date"])
    pit = np.maximum(np.maximum(ann, fann), end)
    comp = pd.to_numeric(fr["comp_type"], errors="coerce").to_numpy()
    rep = pd.to_numeric(fr["report_type"], errors="coerce").to_numpy()
    period = _quarter(end)
    keep = (period >= 0) & ((ann >= 19900101) | (fann >= 19900101))
    keep &= (pit >= end) & (pit <= int(ctx.panel.dates[-1])) & (comp == 1) & (rep == 1)
    if not keep.any():
        return pd.DataFrame(columns=["code", "period", "pit", "end", *fields])
    code = ctx.code_index(fr["stock_code"].astype(str).to_numpy()[keep])
    df = pd.DataFrame({"code": code, "period": period[keep],
                       "pit": pit[keep], "end": end[keep]})
    for f in fields:
        df[f] = pd.to_numeric(fr[f], errors="coerce").to_numpy(np.float64)[keep]
    df = df[df.code >= 0]
    df = df.sort_values(["code", "period", "pit"], kind="stable")
    df = df.drop_duplicates(["code", "period"], keep="first").reset_index(drop=True)
    return df


def _frontier(df):
    if df.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    s = df[["code", "period", "pit"]].sort_values(["code", "pit", "period"], kind="stable")
    s["q"] = s.groupby("code", sort=False).period.cummax()
    return s.drop_duplicates(["code", "pit"], keep="last")[["code", "pit", "q"]].reset_index(drop=True)


def _frontier_at(front, code, pit):
    if front.empty:
        return np.full(len(code), -1, dtype=np.int64)
    keys = front.code.to_numpy(np.int64) * KEY_W + front.pit.to_numpy(np.int64)
    q = np.asarray(code, np.int64) * KEY_W + np.asarray(pit, np.int64)
    pos = np.searchsorted(keys, q, side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (front.code.to_numpy(np.int64)[safe] == np.asarray(code))
    return np.where(ok, front.q.to_numpy(np.int64)[safe], -1)


def _lookup(df, code, period, cap, field):
    if df.empty:
        return np.full(len(code), np.nan)
    keys = ((df.code.to_numpy(np.int64) * PERIOD_W + df.period.to_numpy(np.int64))
            * KEY_W + df.pit.to_numpy(np.int64))
    q = ((np.asarray(code, np.int64) * PERIOD_W + np.asarray(period, np.int64))
         * KEY_W + np.asarray(cap, np.int64))
    pos = np.searchsorted(keys, q, side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (df.code.to_numpy(np.int64)[safe] == np.asarray(code))
    ok &= (df.period.to_numpy(np.int64)[safe] == np.asarray(period))
    return np.where(ok, df[field].to_numpy(np.float64)[safe], np.nan)


def _period_end(period):
    p = np.asarray(period, np.int64)
    md = np.array([331, 630, 930, 1231])[p % 4]
    return (p // 4) * 10000 + md


def _map(ctx, code, pit, period, values):
    out = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    if not len(code):
        return out
    sessions = np.asarray(ctx.cal.days, dtype=np.int32)
    nxt = np.searchsorted(sessions, np.asarray(pit, np.int64) - 1, side="left")
    ok = nxt < len(sessions)
    if not ok.any():
        return out
    eff = sessions[nxt[ok]]
    c = np.asarray(code, np.int64)[ok]
    v = np.asarray(values, np.float64)[ok]
    age_end = (pd.to_datetime(_period_end(np.asarray(period, np.int64)[ok]).astype(str),
                              format="%Y%m%d") + pd.Timedelta(days=MAX_AGE)
               ).strftime("%Y%m%d").astype(np.int64).to_numpy()
    panel = ctx.panel.asof(c, eff, v)
    age = ctx.panel.asof(c, eff, age_end)
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    return np.where(np.isfinite(panel) & (dates[:, None] <= age), panel, np.nan)


def build(parameters):
    params = dict(parameters or {})
    if params.get("variant") != "q_inventory_revenue_seasonal_dev_v1" or params.get("version", 1) != 1:
        raise ValueError("Unknown R38 quarterly variant")

    def compute(ctx):
        bs = _load(ctx, "stock_balancesheet", ("inventories",))
        inc = _load(ctx, "stock_income", ("revenue",))
        if bs.empty or inc.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        pieces = [pd.DataFrame({"code": t.code, "pit": t.pit}) for t in (bs, inc)]
        ev = pd.concat(pieces, ignore_index=True).drop_duplicates()
        ev = ev.sort_values(["code", "pit"], kind="stable").reset_index(drop=True)
        code = ev.code.to_numpy(np.int64)
        pit = ev.pit.to_numpy(np.int64)
        qb = _frontier_at(_frontier(bs), code, pit)
        qi = _frontier_at(_frontier(inc), code, pit)
        ok = (qb >= 0) & (qi >= 0) & (qb == qi)
        code, pit, q = code[ok], pit[ok], qb[ok]
        if not len(q):
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        # balance-sheet inventories at q, q-4, q-8, q-12
        inv = [_lookup(bs, code, q - k, pit, "inventories") for k in (0, 4, 8, 12)]
        # exact single-quarter revenue at q and the three prior same-quarter slots
        def sq_rev(period):
            cur = _lookup(inc, code, period, pit, "revenue")
            prev = _lookup(inc, code, period - 1, pit, "revenue")
            return np.where((period % 4 == 0) | (period % 4 == 3), cur, cur - prev)
        rev = [sq_rev(q - k) for k in (0, 4, 8, 12)]
        xs = []
        for iv, rv in zip(inv, rev):
            good = np.isfinite(iv) & np.isfinite(rv) & (iv > MIN) & (rv > MIN)
            xs.append(np.where(good, np.log(iv / np.where(good, rv, 1.0)), np.nan))
        okv = np.isfinite(xs[0]) & np.isfinite(xs[1]) & np.isfinite(xs[2]) & np.isfinite(xs[3])
        val = np.where(okv, xs[0] - (xs[1] + xs[2] + xs[3]) / 3.0, np.nan)
        return _map(ctx, code, pit, q, val)

    return compute
