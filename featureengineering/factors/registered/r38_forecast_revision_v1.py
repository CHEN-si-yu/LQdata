"""R38 forecast revision: midpoint change of the latest visible forecast vs its first version."""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "fc_revision_120d"
WINDOW_DAYS = 120
SOURCE_START_YEAR = 2015


def _date_vec(ctx, series):
    raw = np.asarray(ctx.date_col(series), dtype=np.int64)
    return np.where((raw >= 19900101) & (raw <= 21001231), raw, 0)


def build(parameters):
    if parameters:
        raise ValueError(f"{NAME} has no tunable parameters")

    def compute(ctx):
        end_year = int(ctx.panel.dates[-1]) // 10000
        frame = ctx.dataset("stock_forecast", columns=[
            "stock_code", "ann_date", "end_date", "p_change_min", "p_change_max",
        ], years=(SOURCE_START_YEAR, end_year))
        out = np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        if frame is None or frame.empty:
            return out
        codes = np.asarray(ctx.panel.codes).astype(str)
        cpos = {c: i for i, c in enumerate(codes)}
        frame = frame.copy()
        frame["_code"] = frame["stock_code"].astype(str)
        frame = frame[frame["_code"].isin(cpos)]
        ann = _date_vec(ctx, frame["ann_date"])
        end = _date_vec(ctx, frame["end_date"])
        lo = pd.to_numeric(frame["p_change_min"], errors="coerce").to_numpy(np.float64)
        hi = pd.to_numeric(frame["p_change_max"], errors="coerce").to_numpy(np.float64)
        mid = (lo + hi) / 2.0
        keep = (ann > 0) & (end > 0)
        if not keep.any():
            return out
        rows = pd.DataFrame({
            "code": frame["_code"].to_numpy()[keep],
            "ann": ann[keep], "end": end[keep], "mid": mid[keep],
            "_row": np.flatnonzero(keep),
        })
        rows = rows.sort_values(["code", "ann", "end", "_row"], kind="stable")
        # first announced version per (stock, report period); a NaN first
        # midpoint stays NaN and later revisions are not promoted.
        first = rows.drop_duplicates(["code", "end"], keep="first").set_index(
            ["code", "end"])["mid"]
        rows["first_mid"] = rows.set_index(["code", "end"]).index.map(first)
        rows["first_mid"] = pd.to_numeric(rows["first_mid"], errors="coerce")
        rows["rev"] = (rows["mid"] - rows["first_mid"]) / 100.0
        dates_i = np.asarray(ctx.panel.dates, dtype=np.int64)
        for code, sub in rows.groupby("code", sort=False):
            ci = cpos[code]
            sub = sub.sort_values(["ann", "end"], kind="stable")
            events = []
            for row in sub.itertuples(index=False):
                if not np.isfinite(row.rev) or row.ann >= dates_i[-1]:
                    continue
                eff = int(np.searchsorted(dates_i, int(row.ann), side="right"))
                if eff >= len(dates_i):
                    continue
                expiry = int((pd.Timestamp(str(int(row.ann))) + pd.Timedelta(days=WINDOW_DAYS))
                             .strftime("%Y%m%d"))
                last = int(np.searchsorted(dates_i, expiry, side="right")) - 1
                if last < eff:
                    continue
                events.append((eff, min(last, len(dates_i) - 1), int(row.end), float(row.rev)))
            events.sort(key=lambda r: (r[0], r[2]))
            for k, (eff, last, _end, val) in enumerate(events):
                nxt = events[k + 1][0] if k + 1 < len(events) else len(dates_i)
                stop = min(last, nxt - 1) if nxt > eff else last
                if stop >= eff:
                    out[eff:stop + 1, ci] = val
        return out

    return compute
