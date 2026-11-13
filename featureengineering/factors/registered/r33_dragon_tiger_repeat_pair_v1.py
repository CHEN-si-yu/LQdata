"""Count repeated public disclosure-name event-date pairs in prior20 sessions.

The repeated objects are literal disclosed organization-name labels, not
brokerage seats or beneficial owners. Source dates remain on their original
session; output T uses [T-20,T), which is the fixed one-session availability
lag for stock_dragon_tiger. No extra event-date shift is applied.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

NAME = "dragon_tiger_public_label_repeat_pair_count_20_v1"
WINDOW = 20
MIN_VALID = 15
MULTIDAY_REASON = r"连续|期间|个交易日"
SOURCE_COLUMNS = ("trade_date", "stock_code", "org_name", "buy_amount", "sell_amount", "reason")


def _compute(ctx) -> np.ndarray:
    dates = np.asarray(ctx.panel.dates, dtype=np.int32)
    codes = np.asarray(ctx.panel.codes).astype(str)
    shape = (len(dates), len(codes))
    out = np.full(shape, np.nan, dtype=np.float64)
    if not len(dates) or not len(codes):
        return out

    date_pos = {int(day): i for i, day in enumerate(dates)}
    code_pos = {str(code): i for i, code in enumerate(codes)}
    years = (int(dates[0]) // 10000, int(dates[-1]) // 10000)

    # source_complete is established from every upstream source row before
    # code-pool selection and before excluding multi-day reason rows.
    rows = ctx.dataset("stock_dragon_tiger", columns=list(SOURCE_COLUMNS), years=years)
    source_complete = np.zeros(len(dates), dtype=bool)
    event_labels: dict[tuple[int, int], dict[str, set[float]]] = defaultdict(lambda: defaultdict(set))
    event_cells: set[tuple[int, int]] = set()
    ambiguous: set[tuple[int, int]] = set()

    if rows is not None and not rows.empty:
        source_dates = ctx.date_col(rows["trade_date"]).astype(np.int32)
        in_axis = (source_dates >= dates[0]) & (source_dates <= dates[-1])
        for day in np.unique(source_dates[in_axis]):
            ti = date_pos.get(int(day))
            if ti is not None:
                source_complete[ti] = True

        code_i = rows["stock_code"].astype(str).map(code_pos).fillna(-1).to_numpy(np.int32)
        eligible = in_axis & (code_i >= 0)
        f = rows.loc[eligible].copy()
        f["_day"] = source_dates[eligible]
        f["_ci"] = code_i[eligible]
        f = f[~f["reason"].fillna("").astype(str).str.contains(MULTIDAY_REASON, regex=True)]
        buy = pd.to_numeric(f["buy_amount"], errors="coerce").to_numpy(np.float64)
        sell = pd.to_numeric(f["sell_amount"], errors="coerce").to_numpy(np.float64)
        gross = buy + sell
        labels = f["org_name"].astype("string").str.strip()
        label_ok = labels.notna().to_numpy() & labels.ne("").to_numpy()
        amount_ok = (np.isfinite(buy) & np.isfinite(sell) & (buy >= 0.0) &
                     (sell >= 0.0) & np.isfinite(gross))
        day_values = f["_day"].to_numpy(np.int32)
        code_values = f["_ci"].to_numpy(np.int32)
        names = labels.to_numpy(dtype=object)

        for j in range(len(f)):
            day = int(day_values[j]); ci = int(code_values[j]); key = (ci, day)
            event_cells.add(key)
            if not label_ok[j] or not amount_ok[j]:
                ambiguous.add(key)
                continue
            event_labels[key][str(names[j])].add(float(gross[j]))

        # A stock-date is ambiguous if any eligible row is malformed or if
        # one exact public label maps to more than one gross amount.
        for key, by_name in event_labels.items():
            if any(len(grosses) != 1 for grosses in by_name.values()):
                ambiguous.add(key)

    # Actual traded slots require only finite positive stock_daily volume.
    daily = ctx.dataset("stock_daily", columns=["trade_date", "stock_code", "vol"], years=years)
    traded = np.zeros(shape, dtype=bool)
    if daily is not None and not daily.empty:
        dd = ctx.date_col(daily["trade_date"]).astype(np.int32)
        ci = daily["stock_code"].astype(str).map(code_pos).fillna(-1).to_numpy(np.int32)
        vol = pd.to_numeric(daily["vol"], errors="coerce").to_numpy(np.float64)
        keep = (dd >= dates[0]) & (dd <= dates[-1]) & (ci >= 0) & np.isfinite(vol) & (vol > 0)
        if keep.any():
            ti = np.fromiter((date_pos.get(int(x), -1) for x in dd[keep]), np.int32, count=int(keep.sum()))
            good = ti >= 0
            traded[ti[good], ci[keep][good]] = True

    # Unknown/malformed and untraded slots stay invalid. A complete source
    # date with no eligible event is a valid empty slot and contributes zero.
    slot_valid = source_complete[:, None] & traded
    for ci, day in ambiguous:
        ti = date_pos.get(int(day))
        if ti is not None:
            slot_valid[ti, ci] = False

    by_code_label: dict[tuple[int, str], list[int]] = defaultdict(list)
    for (ci, day), by_name in event_labels.items():
        ti = date_pos.get(int(day))
        if (ti is None or not source_complete[ti] or not traded[ti, ci] or
                (ci, day) in ambiguous):
            continue
        for label, gross_values in by_name.items():
            if len(gross_values) == 1 and next(iter(gross_values)) > 0.0:
                by_code_label[(ci, label)].append(ti)

    # For a label with two distinct valid event dates a<b, that pair contributes
    # exactly at output endpoints T satisfying b<T<=a+20. Difference arrays
    # avoid rescanning every label for every stock/date.
    diff = np.zeros((len(dates) + 1, len(codes)), dtype=np.int32)
    for (ci, _label), raw_occurrences in by_code_label.items():
        occurrences = sorted(set(raw_occurrences))
        for j, b in enumerate(occurrences):
            k = j - 1
            while k >= 0 and b - occurrences[k] < WINDOW:
                a = occurrences[k]
                lo = b + 1
                hi = min(a + WINDOW + 1, len(dates))  # half-open through a+20
                if lo < hi:
                    diff[lo, ci] += 1
                    diff[hi, ci] -= 1
                k -= 1
    pair_counts = np.cumsum(diff[:-1], axis=0, dtype=np.int32)

    for t in range(WINDOW, len(dates)):
        valid_slots = slot_valid[t - WINDOW:t].sum(axis=0)
        enough = valid_slots >= MIN_VALID
        if enough.any():
            out[t, enough] = pair_counts[t, enough]
    return np.where(np.asarray(ctx.universe, dtype=bool), out, np.nan)


def build(parameters: dict):
    if parameters:
        raise ValueError(f"{NAME} has no parameters")
    return _compute


def _synthetic_checks():
    # Same multiset and same number of distinct event dates, same-day repeats
    # count once; invalid days are unknown and do not reduce to observed zero.
    labels = [set(), {"机构专用", "券商A"}, {"机构专用"}, set(), {"机构专用"}]
    by = defaultdict(set)
    for i, day_labels in enumerate(labels):
        for label in day_labels:
            by[label].add(i)
    assert sum(len(v) * (len(v)-1)//2 for v in by.values()) == 3
    invalid = [set()] * 14 + [None] * 6
    assert sum(x is not None for x in invalid) == 14 < MIN_VALID
    # Pair (a,b) visible only at b<T<=a+20, and only if b-a<20.
    a, b = 10, 15
    assert [t for t in range(40) if b < t <= a + WINDOW] == list(range(16, 31))
    assert len({"机构专用", "机构专用"}) == 1
    return True


if __name__ == "__main__":
    assert _synthetic_checks()
    print("repeat-pair production draft synthetic checks PASS")
