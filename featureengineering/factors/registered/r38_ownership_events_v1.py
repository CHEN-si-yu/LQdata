"""R38 event-conditioned dragon-tiger and top-list factors.

All source rows dated d become effective on the first trading session
strictly after d.  Dragon-tiger multi-day reason rows are removed, exact
seat rows are deduplicated, and mirrored reason groups are cleaned with the
same frozen rule as the research build.  These factors are event-conditioned:
a stock-day with no recent event is NaN, never a fabricated zero.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

NAMES = (
    "dt_institution_row_share_20",
    "dt_ticket_size_20",
    "dt_northbound_share_20",
    "tl_attention_reason_share_20",
    "tl_listing_frequency_60",
)

_GENERIC_DT_REASONS = frozenset({
    "涨幅偏离值达7%的证券",
    "跌幅偏离值达7%的证券",
    "换手率达20%的证券",
    "振幅值达15%的证券",
    "单只标的证券的当日融资买入数量达到当日该证券总交易量的50%以上",
})


def _dates(ctx):
    return np.asarray(ctx.panel.dates, dtype=np.int64)


def _date_strings(s):
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


def _map_exact(dates, src):
    src = np.asarray(src, dtype=np.int64)
    pos = np.searchsorted(dates, src, side="left")
    safe = np.clip(pos, 0, len(dates) - 1)
    ok = (pos < len(dates)) & (dates[safe] == src)
    return pos, ok


def _roll_sum(a, w):
    # research semantics: NaN contributes zero and only the first w-1 rows are NaN
    x = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    cs = np.vstack([np.zeros((1, x.shape[1]), dtype=np.float64), np.cumsum(x, axis=0)])
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape[0] >= w:
        out[w - 1:] = cs[w:] - cs[:-w]
    return out


def _event_mean(val, occ, valid, w=20):
    s = _roll_sum(val, w)
    nv = _roll_sum(valid, w)
    no = _roll_sum(occ, w)
    with np.errstate(all="ignore"):
        out = s / nv
    cond = (no >= 1.0) & (nv >= 1.0) & (np.abs(no - nv) < 1e-9)
    return np.where(cond, out, np.nan)


def _dt_nonsym(rows):
    return tuple(r for r in rows if r[1] != r[2])


def _dt_official_prefix(reason):
    s = str(reason).strip()
    return s.startswith("日") or s.startswith("有价格涨跌幅限制")


def _pick_mirror_group(a, b, td, sc, top):
    def matches(x):
        for lb, ls in top.get((td, sc, x["reason"]), ()):
            if (np.isfinite(lb) and np.isfinite(ls)
                    and np.isclose(x["buy"], lb, rtol=1e-6, atol=1.0)
                    and np.isclose(x["sell"], ls, rtol=1e-6, atol=1.0)):
                return True
        return False

    ma, mb = matches(a), matches(b)
    if ma and not mb:
        return 0
    if mb and not ma:
        return 1
    ga = a["reason"].strip() in _GENERIC_DT_REASONS
    gb = b["reason"].strip() in _GENERIC_DT_REASONS
    if ga != gb:
        return 1 if ga else 0
    fa, fb = "％" in a["reason"], "％" in b["reason"]
    if fa != fb:
        return 0 if fa else 1
    oa, ob = _dt_official_prefix(a["reason"]), _dt_official_prefix(b["reason"])
    if oa != ob:
        return 0 if oa else 1
    return None


def _clean_mirrored(d, top):
    drop = []
    for (td, sc), g in d.groupby(["trade_date", "stock_code"], sort=False):
        items = []
        for reason, gg in g.groupby("reason", sort=True):
            rows = []
            for org, bv, sv in zip(gg["org"].tolist(), gg["buy"].tolist(), gg["sell"].tolist()):
                bv = 0.0 if not np.isfinite(bv) else float(bv)
                sv = 0.0 if not np.isfinite(sv) else float(sv)
                rows.append((str(org), round(bv, 2), round(sv, 2)))
            items.append({
                "reason": str(reason), "rows": tuple(rows),
                "buy": float(np.sum([x[1] for x in rows])),
                "sell": float(np.sum([x[2] for x in rows])),
                "idx": gg.index.to_numpy(),
            })
        used = set()
        for i in range(len(items)):
            if i in used:
                continue
            for j in range(i + 1, len(items)):
                if j in used:
                    continue
                a = _dt_nonsym(items[i]["rows"])
                b = _dt_nonsym(items[j]["rows"])
                if len(a) == 0 or len(a) != len(b):
                    continue
                swapped = Counter((o, sv, bv) for o, bv, sv in b)
                if Counter(a) != swapped:
                    continue
                used.add(i)
                used.add(j)
                pick = _pick_mirror_group(items[i], items[j], td, sc, top)
                if pick is None:
                    drop.extend(items[i]["idx"].tolist())
                    drop.extend(items[j]["idx"].tolist())
                else:
                    drop.extend((items[j] if pick == 0 else items[i])["idx"].tolist())
                break
    if drop:
        d = d.drop(index=drop).reset_index(drop=True)
    return d


def _load_frame(ctx, table, columns):
    dates = _dates(ctx)
    lo = max(2010, int(dates[0]) // 10000 - 2)
    hi = int(dates[-1]) // 10000
    return ctx.dataset(table, columns=columns, years=(lo, hi))


def _dragon(ctx):
    dates = _dates(ctx)
    S, C = ctx.panel.shape
    frame = _load_frame(ctx, "stock_dragon_tiger",
                        ["trade_date", "stock_code", "org_name", "buy_amount",
                         "sell_amount", "reason"])
    if frame is None or frame.empty:
        empty = np.full(ctx.panel.shape, np.nan)
        return empty, empty, empty
    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame["trade_date"] = _date_strings(frame["trade_date"])
    frame["reason"] = frame["reason"].fillna("").astype(str)
    frame = frame[~frame["reason"].str.contains("连续|期间|个交易日", regex=True)]
    frame = frame.drop_duplicates(
        ["trade_date", "stock_code", "org_name", "buy_amount", "sell_amount"])
    frame["buy"] = pd.to_numeric(frame["buy_amount"], errors="coerce")
    frame["sell"] = pd.to_numeric(frame["sell_amount"], errors="coerce")
    frame["org"] = frame["org_name"].fillna("").astype(str)
    frame["north"] = frame["org"].str.contains("沪股通|深股通", regex=True)
    frame["inst"] = frame["org"] == "机构专用"
    frame["missorg"] = frame["org"].str.strip() == ""
    frame = frame.reset_index(drop=True)
    top_frame = _load_frame(ctx, "stock_top_list",
                            ["trade_date", "stock_code", "reason", "l_buy", "l_sell"])
    top = {}
    if top_frame is not None and not top_frame.empty:
        tf = top_frame.copy()
        tf["stock_code"] = tf["stock_code"].astype(str)
        tf["trade_date"] = _date_strings(tf["trade_date"])
        tf["reason"] = tf["reason"].fillna("").astype(str)
        use = set(frame["stock_code"].unique())
        tf = tf[tf["stock_code"].isin(use)]
        for row in tf.itertuples(index=False):
            top.setdefault((row.trade_date, row.stock_code, row.reason), []).append(
                (pd.to_numeric(row.l_buy, errors="coerce"),
                 pd.to_numeric(row.l_sell, errors="coerce")))
    frame = _clean_mirrored(frame, top)

    row = frame.groupby(["trade_date", "stock_code"], sort=False).agg(
        n_rows=("org", "size"), n_inst=("inst", "sum"), missorg=("missorg", "max"))
    org = frame.groupby(["trade_date", "stock_code", "org"], sort=False, as_index=False).agg(
        buy=("buy", "sum"), sell=("sell", "sum"), north=("north", "max"))
    org["gross"] = org["buy"] + org["sell"]
    org["posbuy"] = np.where(org["buy"] > 0, org["buy"], 0.0)
    org["isbuy"] = (org["buy"] > 0).astype(np.float64)
    agg = org.groupby(["trade_date", "stock_code"], sort=False).agg(
        gross=("gross", "sum"), buy_sum=("posbuy", "sum"), buy_seats=("isbuy", "sum"))
    north = org[org["north"]].groupby(["trade_date", "stock_code"], sort=False)["gross"].sum()
    agg["north_gross"] = north.reindex(agg.index).fillna(0.0)
    agg = agg.join(row, how="left").reset_index()
    agg["buy_seats"] = agg["buy_seats"].fillna(0).astype(np.int64)
    # event row -> source panel row, then next session
    src = agg["trade_date"].str.replace("-", "", regex=False).astype(np.int64).to_numpy()
    si, ok = _map_exact(dates, src)
    ci = ctx.code_index(agg["stock_code"].to_numpy())
    keep = ok & (ci >= 0)
    agg = agg.iloc[np.where(keep)[0]].reset_index(drop=True)
    si, ci = si[keep], ci[keep]
    amt = np.asarray(ctx.px_raw("amount"), dtype=np.float64)
    v2 = np.full((S, C), np.nan)
    v3 = np.full((S, C), np.nan)
    v4 = np.full((S, C), np.nan)
    o2 = np.zeros((S, C)); o3 = np.zeros((S, C)); o4 = np.zeros((S, C))
    a2 = np.zeros((S, C)); a3 = np.zeros((S, C)); a4 = np.zeros((S, C))
    for j in range(len(agg)):
        s = int(si[j]); c = int(ci[j]); e = s + 1
        if e >= S:
            continue
        gross = float(agg.at[j, "gross"])
        buy = float(agg.at[j, "buy_sum"])
        bs = int(agg.at[j, "buy_seats"])
        ng = float(agg.at[j, "north_gross"])
        nrows = float(agg.at[j, "n_rows"])
        ninst = float(agg.at[j, "n_inst"])
        miss = bool(agg.at[j, "missorg"])
        o2[e, c] = 1.0
        if nrows > 0 and not miss and np.isfinite(ninst):
            v2[e, c] = ninst / nrows
            a2[e, c] = 1.0
        o3[e, c] = 1.0
        if bs >= 1 and np.isfinite(amt[s, c]) and amt[s, c] > 0 and np.isfinite(buy):
            v3[e, c] = (buy / bs) / amt[s, c]
            a3[e, c] = 1.0
        o4[e, c] = 1.0
        if np.isfinite(gross) and gross > 0 and np.isfinite(ng):
            v4[e, c] = ng / gross
            a4[e, c] = 1.0
    return (_event_mean(v2, o2, a2), _event_mean(v3, o3, a3),
            _event_mean(v4, o4, a4))


def _toplist(ctx):
    dates = _dates(ctx)
    S, C = ctx.panel.shape
    frame = _load_frame(ctx, "stock_top_list", ["trade_date", "stock_code", "reason"])
    if frame is None or frame.empty:
        empty = np.full(ctx.panel.shape, np.nan)
        return empty, empty
    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame["trade_date"] = _date_strings(frame["trade_date"])
    frame["reason"] = frame["reason"].fillna("").astype(str)
    grp = frame.groupby(["trade_date", "stock_code"], sort=False)["reason"].agg(list).reset_index()
    src = grp["trade_date"].str.replace("-", "", regex=False).astype(np.int64).to_numpy()
    si, ok = _map_exact(dates, src)
    ci = ctx.code_index(grp["stock_code"].to_numpy())
    keep = ok & (ci >= 0)
    grp = grp.iloc[np.where(keep)[0]].reset_index(drop=True)
    si, ci = si[keep], ci[keep]
    v = np.full((S, C), np.nan)
    o = np.zeros((S, C)); a = np.zeros((S, C))
    ev = np.zeros((S, C))
    for j in range(len(grp)):
        s = int(si[j]); c = int(ci[j]); e = s + 1
        if e >= S:
            continue
        reasons = grp.at[j, "reason"]
        ev[e, c] = 1.0
        o[e, c] = 1.0
        if any(not isinstance(x, str) or x.strip() == "" for x in reasons):
            v[e, c] = np.nan
            a[e, c] = 0.0
            continue
        att = any(("换手率" in x or "振幅" in x) for x in reasons)
        price = any(("涨幅偏离" in x or "跌幅偏离" in x) for x in reasons)
        multi = any(("连续" in x or "期间" in x or "个交易日" in x) for x in reasons)
        v[e, c] = 1.0 if (att and not price and not multi) else 0.0
        a[e, c] = 1.0
    t1 = _event_mean(v, o, a)
    cnt = _roll_sum(ev, 60)
    t2 = cnt / 60.0
    t2 = np.where((cnt >= 1.0) & np.isfinite(cnt), t2, np.nan)
    return t1, t2


def build(parameters):
    variant = (parameters or {}).get("variant")
    if variant not in NAMES:
        raise KeyError(f"unknown R38 ownership event variant: {variant!r}")

    def compute(ctx):
        if variant.startswith("dt_"):
            d2, d3, d4 = _dragon(ctx)
            if variant == "dt_institution_row_share_20":
                return d2
            if variant == "dt_ticket_size_20":
                return d3
            return d4
        t1, t2 = _toplist(ctx)
        if variant == "tl_attention_reason_share_20":
            return t1
        return t2

    return compute
