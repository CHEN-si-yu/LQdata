"""R38 cross-statement factor: OCI flow scaled by average equity."""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_AGE = 550
MIN_EQ = 1_000_000.0
BASE_YEAR = 2010


def _date_vec(ctx, s):
    raw = np.asarray(ctx.date_col(s), dtype=np.int64)
    return np.where((raw >= 19900101) & (raw <= 21001231), raw, 0)


def _annual_rows(ctx, dataset, fields):
    end_year = int(ctx.panel.dates[-1]) // 10000
    cols = ["stock_code", "ann_date", "f_ann_date", "end_date",
            "comp_type", "report_type", *fields]
    frame = ctx.dataset(dataset, columns=cols, years=(BASE_YEAR, end_year))
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["code", "year", "pit", "end", *fields])
    ann = _date_vec(ctx, frame["ann_date"])
    fann = _date_vec(ctx, frame["f_ann_date"])
    end = _date_vec(ctx, frame["end_date"])
    pit = np.maximum(np.maximum(ann, fann), end)
    comp = pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy()
    rep = pd.to_numeric(frame["report_type"], errors="coerce").to_numpy()
    keep = ((ann >= 19900101) | (fann >= 19900101)) & (end > 0) & (pit >= end)
    keep &= (end % 10000 == 1231) & (comp == 1) & (rep == 1)
    keep &= pit <= int(ctx.panel.dates[-1])
    if not keep.any():
        return pd.DataFrame(columns=["code", "year", "pit", "end", *fields])
    code = ctx.code_index(frame["stock_code"].astype(str).to_numpy()[keep])
    rows = pd.DataFrame({"code": code, "year": end[keep] // 10000,
                         "pit": pit[keep], "end": end[keep]})
    for f in fields:
        rows[f] = pd.to_numeric(frame[f], errors="coerce").to_numpy(np.float64)[keep]
    rows = rows[rows.code >= 0]
    # first disclosed whole row per stock/FY
    rows = rows.sort_values(["code", "year", "pit"], kind="stable")
    rows = rows.drop_duplicates(["code", "year"], keep="first").reset_index(drop=True)
    return rows


def _lookup(table, code, year, cap, field):
    """Latest visible first-vintage value for (code, year) at cap."""
    if table.empty:
        return np.full(len(code), np.nan)
    keys = ((table.code.to_numpy(np.int64) * 10000 + table.year.to_numpy(np.int64))
            * 100000000 + table.pit.to_numpy(np.int64))
    q = (np.asarray(code, np.int64) * 10000 + np.asarray(year, np.int64)) * 100000000 + np.asarray(cap, np.int64)
    pos = np.searchsorted(keys, q, side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (table.code.to_numpy(np.int64)[safe] == np.asarray(code))
    ok &= (table.year.to_numpy(np.int64)[safe] == np.asarray(year))
    return np.where(ok, table[field].to_numpy(np.float64)[safe], np.nan)


def _frontier(df):
    if df.empty:
        return pd.DataFrame(columns=["code", "pit", "q"])
    s = df[["code", "year", "pit"]].sort_values(["code", "pit", "year"], kind="stable")
    s["q"] = s.groupby("code", sort=False).year.cummax()
    return s.drop_duplicates(["code", "pit"], keep="last")[["code", "pit", "q"]].reset_index(drop=True)


def _frontier_at(front, code, pit):
    if front.empty:
        return np.full(len(code), -1, dtype=np.int64)
    keys = front.code.to_numpy(np.int64) * 100000000 + front.pit.to_numpy(np.int64)
    q = np.asarray(code, np.int64) * 100000000 + np.asarray(pit, np.int64)
    pos = np.searchsorted(keys, q, side="right") - 1
    safe = np.maximum(pos, 0)
    ok = (pos >= 0) & (front.code.to_numpy(np.int64)[safe] == np.asarray(code))
    return np.where(ok, front.q.to_numpy(np.int64)[safe], -1)


def _joint_events(a, b):
    pieces = [pd.DataFrame({"code": t.code, "pit": t.pit}) for t in (a, b)]
    ev = pd.concat(pieces, ignore_index=True).drop_duplicates()
    ev = ev.sort_values(["code", "pit"], kind="stable").reset_index(drop=True)
    code = ev.code.to_numpy(np.int64)
    pit = ev.pit.to_numpy(np.int64)
    qa = _frontier_at(_frontier(a), code, pit)
    qb = _frontier_at(_frontier(b), code, pit)
    ok = (qa >= 0) & (qa == qb)
    return code[ok], pit[ok], qa[ok]


def _year_end(y):
    return np.asarray(y, np.int64) * 10000 + 1231


def _map(ctx, code, pit, end, values):
    out = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    if len(code) == 0:
        return out
    sessions = np.asarray(ctx.cal.days, dtype=np.int32)
    nxt = np.searchsorted(sessions, np.asarray(pit, np.int64) - 1, side="left")
    ok = nxt < len(sessions)
    if not ok.any():
        return out
    eff = sessions[nxt[ok]]
    c = np.asarray(code, np.int64)[ok]
    v = np.asarray(values, np.float64)[ok]
    age_end = (pd.to_datetime(np.asarray(end, np.int64)[ok].astype(str), format="%Y%m%d")
               + pd.Timedelta(days=MAX_AGE)).strftime("%Y%m%d").astype(np.int64).to_numpy()
    panel = ctx.panel.asof(c, eff, v)
    age = ctx.panel.asof(c, eff, age_end)
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    return np.where(np.isfinite(panel) & (dates[:, None] <= age), panel, np.nan)


def build(parameters):
    params = dict(parameters or {})
    variant = params.get("variant")
    if params.get("version", 1) != 1:
        raise ValueError("Unknown R38 cross-field version")

    def compute_oci(ctx):
        inc = _annual_rows(ctx, "stock_income", ("oth_compr_income",))
        bs = _annual_rows(ctx, "stock_balancesheet", ("total_hldr_eqy_inc_min_int",))
        if inc.empty or bs.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        eq_cur = _lookup(bs, inc.code, inc.year, inc.pit, "total_hldr_eqy_inc_min_int")
        eq_prev = _lookup(bs, inc.code, inc.year - 1, inc.pit, "total_hldr_eqy_inc_min_int")
        oci = inc["oth_compr_income"].to_numpy(np.float64)
        avg = (eq_cur + eq_prev) / 2.0
        ok = np.isfinite(oci) & np.isfinite(eq_cur) & np.isfinite(eq_prev) & (avg > MIN_EQ)
        vals = np.where(ok, np.arcsinh(oci / np.where(ok, avg, 1.0)), np.nan)
        return _map(ctx, inc.code.to_numpy(np.int64), inc.pit.to_numpy(np.int64),
                    inc.end.to_numpy(np.int64), vals)

    def compute_payout(ctx):
        inc = _annual_rows(ctx, "stock_income", ("fin_exp_int_exp", "n_income_attr_p"))
        cf = _annual_rows(ctx, "stock_cashflow", ("c_pay_dist_dpcp_int_exp",))
        if inc.empty or cf.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        code, pit, q = _joint_events(inc, cf)
        dist = _lookup(cf, code, q, pit, "c_pay_dist_dpcp_int_exp")
        interest = _lookup(inc, code, q, pit, "fin_exp_int_exp")
        ni = _lookup(inc, code, q, pit, "n_income_attr_p")
        proxy = dist - interest
        ok = np.isfinite(proxy) & np.isfinite(ni) & (proxy >= 0.0) & (ni > MIN_EQ)
        vals = np.where(ok, np.arcsinh(proxy / np.where(ok, ni, 1.0)), np.nan)
        return _map(ctx, code, pit, _year_end(q), vals)

    def compute_rd(ctx):
        inc = _annual_rows(ctx, "stock_income", ("rd_exp",))
        bs = _annual_rows(ctx, "stock_balancesheet", ("r_and_d",))
        if inc.empty or bs.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        code, pit, q = _joint_events(inc, bs)
        exp = _lookup(inc, code, q, pit, "rd_exp")
        rd_t = _lookup(bs, code, q, pit, "r_and_d")
        rd_p = _lookup(bs, code, q - 1, pit, "r_and_d")
        ok = np.isfinite(exp) & np.isfinite(rd_t) & np.isfinite(rd_p) & (exp > MIN_EQ)
        vals = np.where(ok, np.arcsinh((rd_t - rd_p) / np.where(ok, exp, 1.0)), np.nan)
        return _map(ctx, code, pit, _year_end(q), vals)

    if variant == "xf_oci_flow_to_equity_fy_v1":
        return compute_oci
    if variant == "xf_payout_ratio_fy_v1":
        return compute_payout
    if variant == "xf_rd_capitalization_fy_v1":
        return compute_rd
    raise ValueError(f"Unknown R38 cross-field variant: {variant!r}")
