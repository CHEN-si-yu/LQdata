"""Shared production PIT helper for the R38 raw-field annual factors.

The helper reads arbitrary annual report fields through the production Context,
keeps the latest visible version of the current report-period frontier, maps it
to the first exchange session on/after the announcement, and applies a
550-calendar-day report-age cap.  Cross-statement ratios require the same
fiscal year on both legs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_AGE = 550
MIN_DEN = 1_000_000.0
BASE_YEAR = 2010


def _date_vec(ctx, series):
    raw = np.asarray(ctx.date_col(series), dtype=np.int64)
    return np.where((raw >= 19900101) & (raw <= 21001231), raw, 0)


def annual_panels(ctx, dataset, fields, max_age=MAX_AGE):
    fields = tuple(dict.fromkeys(fields))
    end_year = int(ctx.panel.dates[-1]) // 10000
    columns = ["stock_code", "ann_date", "f_ann_date", "end_date",
               "comp_type", "report_type", *fields]
    frame = ctx.dataset(dataset, columns=columns, years=(BASE_YEAR, end_year))
    shape = ctx.panel.shape
    out = {f: np.full(shape, np.nan, dtype=np.float64) for f in fields}
    years = np.full(shape, np.nan, dtype=np.float64)
    if frame is None or frame.empty:
        return out, years
    ann = _date_vec(ctx, frame["ann_date"]) if "ann_date" in frame else np.zeros(len(frame), np.int64)
    fann = _date_vec(ctx, frame["f_ann_date"]) if "f_ann_date" in frame else np.zeros(len(frame), np.int64)
    end = _date_vec(ctx, frame["end_date"])
    pit = np.maximum(np.maximum(ann, fann), end)
    rep = (pd.to_numeric(frame["report_type"], errors="coerce").to_numpy()
           if "report_type" in frame.columns else np.ones(len(frame)))
    comp = (pd.to_numeric(frame["comp_type"], errors="coerce").to_numpy()
            if "comp_type" in frame.columns else np.ones(len(frame)))
    keep = ((ann >= 19900101) | (fann >= 19900101)) & (end > 0) & (pit >= end)
    keep &= (end % 10000 == 1231) & (rep == 1)
    # Non-financial and financial issuers are both retained; the raw field
    # semantics are the same annual-report line.  comp is currently unused.
    if not keep.any():
        return out, years
    code = ctx.code_index(frame["stock_code"].astype(str).to_numpy()[keep])
    rows = pd.DataFrame({"code": code, "year": end[keep] // 10000,
                         "end": end[keep], "pit": pit[keep],
                         "_row": np.flatnonzero(keep)})
    for f in fields:
        rows[f] = pd.to_numeric(frame[f], errors="coerce").to_numpy(np.float64)[keep]
    rows = rows[rows.code >= 0].copy()
    if rows.empty:
        return out, years
    # latest visible version for the current annual-report period frontier
    rows = rows.sort_values(["code", "year", "pit", "_row"], kind="stable")
    rows = rows.drop_duplicates(["code", "year", "pit"], keep="last")
    rows = rows.sort_values(["code", "pit", "year", "_row"], kind="stable")
    frontier = rows.groupby("code", sort=False)["year"].cummax().to_numpy()
    rows = rows.loc[rows["year"].to_numpy() == frontier]
    rows = rows.sort_values(["code", "pit", "_row"], kind="stable")
    rows = rows.drop_duplicates(["code", "pit"], keep="last")
    sessions = np.asarray(ctx.cal.days, dtype=np.int32)
    # first session >= pit; pit-1 avoids skipping a trading-day announcement
    nxt = np.searchsorted(sessions, rows["pit"].to_numpy(np.int64) - 1, side="left")
    ok = nxt < len(sessions)
    rows = rows.loc[ok].copy()
    rows["eff"] = sessions[nxt[ok]]
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    code_all = rows["code"].to_numpy(np.int64)
    eff_all = rows["eff"].to_numpy(np.int64)
    end_all = rows["end"].to_numpy(np.int64)
    year_all = rows["year"].to_numpy(np.float64)
    age_end_all = (pd.to_datetime(end_all.astype(str), format="%Y%m%d")
                   + pd.Timedelta(days=max_age)).strftime("%Y%m%d").astype(np.int64)
    age_panel = ctx.panel.asof(code_all, eff_all, age_end_all)
    year_panel = ctx.panel.asof(code_all, eff_all, year_all)
    for f in fields:
        panel = ctx.panel.asof(code_all, eff_all, rows[f].to_numpy(np.float64))
        valid = np.isfinite(panel) & (dates[:, None] <= age_panel)
        out[f] = np.where(valid, panel, np.nan)
    years = np.where(np.isfinite(year_panel), year_panel, np.nan)
    return out, years


def ratio(num, den, min_den=MIN_DEN):
    ok = np.isfinite(num) & np.isfinite(den) & (den > min_den)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    out[ok] = num[ok] / den[ok]
    return out
