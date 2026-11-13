"""Basic descriptive chip factors under the 2026-11-10 green-card mandate.

These are economic state measurements, without a claimed alpha direction or
predictive admission result. Registered parameters select one documented
measurement; every raw histogram is aggregated by the common source-aware
cache, never by individual factor implementations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea.dates import series_to_int

from .chip_distribution_cache_v1 import field, _VIEW

_TEMPORAL = {}


def _column_reader(ctx, seed, frame):
    """Place all columns with one exact key mapping for this immutable frame.

    The caller has just validated disk/source identities through ``field``.
    Reusing the row mapping avoids parsing half a million string dates and
    looking up the same stock codes for every requested temporal input.
    """
    dates = ctx.panel.dates
    days = series_to_int(frame.trade_date)
    ci = pd.Index(ctx.panel.codes).get_indexer(frame.stock_code)
    ti = np.searchsorted(dates, days)
    valid = (ci >= 0) & (ti < len(dates))
    valid &= dates[np.minimum(ti, len(dates)-1)] == days
    cutoff = getattr(ctx.up, "audit_cutoff", None)
    if cutoff is not None:
        valid &= days <= cutoff
    flat = ti[valid] * ctx.panel.C + ci[valid]
    data = {"log_mean": seed}

    def getter(context, name):
        if context is not ctx:
            raise ValueError("a chip column view belongs to its context")
        if name not in data:
            out = np.full(ctx.panel.size, np.nan, dtype=np.float64)
            out[flat] = frame[name].to_numpy(np.float64)[valid]
            data[name] = out.reshape(ctx.panel.shape)
        return data[name]

    return getter


def build(parameters):
    from .chip_gc_geometry_v1 import NAMES as geometry
    from .chip_gc_close_v1 import NAMES as close
    from .chip_gc_temporal_v1 import NAMES as temporal, _impl
    variant = parameters["variant"]
    if variant not in (*geometry, *close, *temporal):
        raise KeyError(variant)

    def compute(ctx):
        if variant in temporal:
            # Validate the current disk/source identity before reusing a compact
            # temporal pack. Axis bytes also distinguish equal-shaped panels.
            seed = field(ctx, "log_mean")
            identity = (_VIEW.get("key"), ctx.panel.dates.tobytes(),
                        tuple(map(str, ctx.panel.codes)))
            adj_sources = []
            for year in range(int(ctx.panel.dates[0])//10000, int(ctx.panel.dates[-1])//10000+1):
                p = ctx.cfg.upstream / "stock_adj_factor" / f"year={year}" / "data.parquet"
                st = p.stat() if p.exists() else None
                adj_sources.append((str(p), st.st_size, st.st_mtime_ns) if st else (str(p), None))
            identity += (tuple(adj_sources),)
            if _TEMPORAL.get("identity") != identity:
                getter = _column_reader(ctx, seed, _VIEW["frame"])
                pack = _impl(ctx, getter)
                _TEMPORAL.clear()
                _TEMPORAL.update(identity=identity, pack=pack)
            # The cache hit still declares every direct source it consumes.
            ctx.accessed.add("stock_adj_factor")
            out = _TEMPORAL["pack"][variant]
        else:
            out = field(ctx, variant)
        cl = np.asarray(ctx.px_raw("close"), dtype=np.float64)
        mask = ctx.traded() & ctx.universe & np.isfinite(cl) & (cl > 0)
        return np.where(mask & np.isfinite(out), out, np.nan)

    return compute
