"""R9-A / unit A-2: stock exposure to its OWN sector's intraday realised volatility.

Frontier: `tdx_minute` (5-minute bars of TDX *board* indices, zero usage in the 177
factor catalog).  One deleted construct family is nearby and is named here explicitly
rather than hidden -- see `note` in the candidate spec for the differentiation argument.

Data facts measured 2026-10-05 (evidence: artifacts/research/round9_20261005/
a_chip_distribution/a_report.md):
  * tdx_minute: year partitions 2011..2026, 1121 distinct board_code, 48 bars/day
    (09:35..15:00), so a "day" is a complete 5-minute bar set.
  * `board_name` is used only as a join key, never as a numeric signal --
    conf/field_exclusions.json says exactly that for tdx_minute.board_name.
  * There is NO block->stock constituent table locally.  `tdx_blocks` (1121 rows:
    block_code/block_name/block_type) and `dc_blocks` (1031 rows) are classification
    dictionaries with no members, and BOTH fail `Upstream._audit_columns` inside a
    research run (no valid date column -> ValueError).  The live market factor
    `mkt_sector_dispersion` documents the same absence ("local has no historical
    constituent table").  Therefore the sector map is built from `stock_list.industry`,
    which IS audit-safe because `Upstream._audit_columns` whitelists `stock_list` as
    mode="contract" (a calendar/axis contract, not an information snapshot).
  * Measured map coverage: 2114/2114 frozen-pool stocks carry a non-empty industry and
    110/110 distinct industries resolve to a board_name present in tdx_minute.

Memory note: tdx_minute is read one/two year partitions at a time and rows are filtered
to the ~110 needed boards before any per-group work; the 60M-row table is never held
whole.  Arrow-backed string columns are kept as-is (no `.astype(str)` on large frames).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea import mathx as mx
from fea.prices import day_ints

SIGNALS = ("sec_own_rv_beta",)

# ---- constants frozen before the first run; never tuned on IC or returns ----
WINDOW = 60          # rolling beta window, trading days
MIN_COUNT = 54       # 90% of WINDOW must be present, else NaN
MIN_RET = 40         # a sector-day needs >= 40 valid 5-min returns out of 47
VAR_FLOOR = 1e-12    # guard against a degenerate regressor

PARAMETERS = {"signal": "sec_own_rv_beta", "window": WINDOW, "min_count": MIN_COUNT}


def _sector_rv(tm: pd.DataFrame) -> pd.DataFrame:
    """(day x board_code) intraday realised vol = sqrt(sum of squared 5-min log returns).

    Bars are ordered by the ISO `trade_time` string, which sorts chronologically.
    The first bar of each day has no previous close and is dropped by `shift(1)`.
    """
    frame = tm.loc[:, ["board_code", "trade_time", "close"]].copy()
    frame["day"] = day_ints(frame["trade_time"].str.slice(0, 10))
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    frame["close"] = close
    frame = frame.loc[np.isfinite(close) & (close > 0)].copy()
    frame = frame.sort_values(["board_code", "day", "trade_time"], kind="stable")
    grouped = frame.groupby(["board_code", "day"], sort=False)
    prev = grouped["close"].shift(1).to_numpy(dtype=np.float64)
    cur = frame["close"].to_numpy(dtype=np.float64)
    with np.errstate(all="ignore"):
        logret = np.log(cur / prev)
    frame["r2"] = np.where(np.isfinite(logret), logret * logret, np.nan)
    agg = frame.groupby(["board_code", "day"], sort=False)["r2"].agg(["sum", "count"])
    rv = np.sqrt(agg["sum"].to_numpy(dtype=np.float64))
    ok = agg["count"].to_numpy(dtype=np.float64) >= MIN_RET
    out = pd.DataFrame({"board_code": agg.index.get_level_values(0),
                        "day": agg.index.get_level_values(1).astype(np.int64),
                        "rv": np.where(ok, rv, np.nan)})
    return out.pivot(index="day", columns="board_code", values="rv")


def _board_lookup(tm: pd.DataFrame) -> dict:
    """board_name -> board_code, deterministic when a name matches several codes."""
    pairs = tm.loc[:, ["board_name", "board_code"]].dropna().drop_duplicates()
    pairs = pairs.assign(board_name=pairs["board_name"].astype(str).str.strip(),
                         board_code=pairs["board_code"].astype(str))
    weight = pairs.groupby(["board_name", "board_code"]).size().rename("n").reset_index()
    # most rows wins; ties broken by the lexicographically smallest board_code
    weight = weight.sort_values(["board_name", "n", "board_code"],
                                ascending=[True, False, True], kind="stable")
    best = weight.drop_duplicates("board_name", keep="first")
    return dict(zip(best["board_name"], best["board_code"]))


def _stock_board(ctx, board_of_name: dict, codes: np.ndarray) -> np.ndarray:
    """stock column -> board_code, via stock_list.industry == tdx_minute.board_name."""
    listing = ctx.dataset("stock_list", columns=["stock_code", "industry"])
    listing = listing.dropna(subset=["industry"])
    listing = listing.assign(stock_code=listing["stock_code"].astype(str),
                             industry=listing["industry"].astype(str).str.strip())
    listing = listing.loc[listing["industry"] != ""].drop_duplicates("stock_code", keep="last")
    industry = dict(zip(listing["stock_code"], listing["industry"]))
    return np.array([board_of_name.get(industry.get(str(code), ""), "") for code in codes],
                    dtype=object)


def build(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Only the single frozen R9-A sector-intraday definition is permitted")

    def compute(ctx):
        days = np.asarray(ctx.panel.dates, dtype=np.int64)
        codes = np.asarray(ctx.panel.codes, dtype=object)
        shape = (days.size, codes.size)
        y0, y1 = int(days[0]) // 10000, int(days[-1]) // 10000
        tm = ctx.dataset("tdx_minute",
                         columns=["board_code", "board_name", "trade_time", "close"],
                         years=(y0, y1))
        if tm is None or tm.empty:
            return np.full(shape, np.nan, dtype=np.float64)

        stock_board = _stock_board(ctx, _board_lookup(tm), codes)
        needed = {b for b in stock_board if b}
        if not needed:
            return np.full(shape, np.nan, dtype=np.float64)
        tm = tm.loc[tm["board_code"].isin(needed)]
        table = _sector_rv(tm)
        del tm
        if table.empty:
            return np.full(shape, np.nan, dtype=np.float64)

        table = table.reindex(index=days)
        pos = {c: i for i, c in enumerate(table.columns)}
        idx = np.array([pos.get(b, -1) for b in stock_board], dtype=np.int64)
        values = table.to_numpy(dtype=np.float64)
        # -1 must stay NaN: indexing with -1 would silently take the last column.
        regressor = np.full(shape, np.nan, dtype=np.float64)
        ok = idx >= 0
        if ok.any():
            regressor[:, ok] = values[:, idx[ok]]

        returns = np.asarray(ctx.ret(1), dtype=np.float64)
        with np.errstate(all="ignore"):
            covariance = mx.roll_cov(returns, regressor, WINDOW, min_count=MIN_COUNT)
            variance = mx.roll_var(regressor, WINDOW, min_count=MIN_COUNT)
            beta = np.where(variance > VAR_FLOOR, covariance / variance, np.nan)
        return np.where(np.isfinite(beta), beta, np.nan)

    return compute
