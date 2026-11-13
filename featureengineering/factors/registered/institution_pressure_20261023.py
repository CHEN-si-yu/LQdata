"""Institution-only Dragon Tiger net pressure from the frozen R21 definition."""
from __future__ import annotations

import numpy as np
import pandas as pd

def _compute(ctx):
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    codes = np.asarray(ctx.panel.codes).astype(str)
    shape = (len(dates), len(codes))
    self_code_pos = {str(code): i for i, code in enumerate(codes)}
    if not len(dates) or not len(codes):
        return np.full(shape, np.nan, dtype=np.float64)

    # Read all available event partitions in the panel's warmup span.
    y0, y1 = int(dates[0]) // 10000, int(dates[-1]) // 10000
    dragon = ctx.dataset(
        "stock_dragon_tiger",
        columns=["stock_code", "trade_date", "org_name", "buy_amount", "sell_amount", "reason"],
        years=(y0, y1),
    )
    source_daily = np.full(shape, np.nan, dtype=np.float64)
    if dragon is not None and not dragon.empty:
        d = ctx.date_col(dragon["trade_date"])
        cidx = ctx.code_index(dragon["stock_code"].astype(str).to_numpy())
        in_dates = (d >= dates[0]) & (d <= dates[-1])
        # Any source row is the R21 observable source-date coverage certificate.
        covered = np.unique(d[in_dates])
        date_pos = {int(day): i for i, day in enumerate(dates)}
        for day in covered:
            i = date_pos.get(int(day))
            if i is not None:
                source_daily[i, :] = 0.0

        in_panel = in_dates & (cidx >= 0)
        f = dragon.loc[in_panel].copy()
        f["date_int"] = d[in_panel]
        f["code_idx"] = cidx[in_panel]
        f = f[f["org_name"].eq("机构专用")]
        f = f[~f["reason"].fillna("").astype(str).str.contains(r"连续|期间|个交易日", regex=True)]
        f["buy_amount"] = pd.to_numeric(f["buy_amount"], errors="coerce")
        f["sell_amount"] = pd.to_numeric(f["sell_amount"], errors="coerce")

        bad = f[~np.isfinite(f["buy_amount"]) | ~np.isfinite(f["sell_amount"])]
        bad_keys = set(zip(bad["code_idx"].astype(int), bad["date_int"].astype(int)))
        f = f[np.isfinite(f["buy_amount"]) & np.isfinite(f["sell_amount"])].copy()
        f = f.drop_duplicates(
            ["stock_code", "date_int", "org_name", "buy_amount", "sell_amount"], keep="first"
        )
        f["net"] = f["buy_amount"] - f["sell_amount"]
        f = f[np.isfinite(f["net"].to_numpy(dtype=np.float64))]
        # Match the frozen R21 key/group order exactly; changing key dtype/order
        # can move the last few float64 bits after summing multiple disclosures.
        f = f.groupby(["stock_code", "date_int"], as_index=False, sort=False)["net"].sum()
        for row in f.itertuples(index=False):
            ti = date_pos.get(int(row.date_int))
            ci = self_code_pos.get(str(row.stock_code))
            if ci is not None and ti is not None and (ci, int(row.date_int)) not in bad_keys:
                source_daily[ti, ci] = float(row.net)
        for ci, day in bad_keys:
            ti = date_pos.get(day)
            if ti is not None:
                source_daily[ti, ci] = np.nan

    amount = np.asarray(ctx.px_raw("amount"), dtype=np.float64)
    valid = np.isfinite(source_daily) & np.isfinite(amount) & (amount > 0)
    numerator = np.where(valid, source_daily, np.nan)
    denominator = np.where(valid, amount, np.nan)

    def rolling_sum(x, width=20, minimum=15):
        # Preserve frozen candidates.py's per-date np.nansum(window, axis=0)
        # reduction order, including its partial leading windows.
        out = np.full_like(x, np.nan, dtype=np.float64)
        for t in range(x.shape[0]):
            window = x[max(0, t - width + 1):t + 1]
            count = np.isfinite(window).sum(axis=0)
            total = np.nansum(window, axis=0)
            out[t] = np.where(count >= minimum, total, np.nan)
        return out

    num = rolling_sum(numerator)
    den = rolling_sum(denominator)
    ratio = np.full(shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = num / den
    ratio[:] = np.where(np.isfinite(quotient) & (den > 0), quotient, np.nan)
    out = np.full(shape, np.nan, dtype=np.float64)
    out[1:] = ratio[:-1]
    return np.where(np.asarray(ctx.universe, dtype=bool), out, np.nan)


def build(parameters):
    """Production entry point; the frozen candidate has no tunable parameters."""
    if parameters:
        raise ValueError("dragon_tiger_institution_net_pressure_20_v1 takes no parameters")
    return _compute
