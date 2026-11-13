"""Shared, source-identified daily chip distributions for descriptive factors.

Raw price bins are scanned once per year in monthly batches. A bad bin
invalidates its whole stock-day; duplicate prices are combined before measuring
shape. No labels or future prices enter this layer. Cache files are derived
data, never factor output or production completion records.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from fea.dates import series_to_int, int_to_str, int_to_str_vec
from fea.manifest import _atomic_json
from fea.store import _atomic_write

BASE_FIELDS = ("log_mean", "log_std", "log_skew", "log_kurt", "hhi",
               "entropy", "max_mass", "n_peaks", "n_positive",
               "log_q10", "log_q25", "log_q50", "log_q75", "log_q90",
               *tuple(f"q{i:02d}" for i in range(5, 100, 5)))
LEGACY_FIELDS = ("legacy_W", "legacy_over", "legacy_above", "legacy_near",
                 "legacy_far", "legacy_local")
_VIEW = {}


def _implementations():
    from .chip_gc_geometry_v1 import NAMES as gn, summarize as gs
    from .chip_gc_close_v1 import NAMES as cn, summarize as cs
    return gn + cn, gs, cs


@njit(cache=True)
def _base(p, w):
    z = np.log(p)
    mu = np.sum(w * z)
    v = np.sum(w * (z - mu)**2)
    sd = np.sqrt(max(v, 0.0))
    sk = np.nan
    ku = np.nan
    if sd > 1e-14:
        sk = np.sum(w * (z - mu)**3) / sd**3
        ku = np.sum(w * (z - mu)**4) / sd**4 - 3.0
    ent = 0.0
    for wi in w:
        if wi > 0:
            ent -= wi * np.log(wi)
    peaks = 0
    for j in range(1, len(w)-1):
        if w[j] > w[j-1] and w[j] > w[j+1]:
            peaks += 1
    cw = np.cumsum(w)
    cw /= cw[-1]
    # Interpolate in PRICE space, then take the log; same-day cost quantiles.
    q = np.log(np.interp(np.array([.10, .25, .50, .75, .90]), cw, p))
    qgrid = np.log(np.interp(np.arange(1, 20) * .05, cw, p))
    out = np.empty(33)
    out[:9] = np.array([mu, sd, sk, ku, np.sum(w*w), ent,
                         np.max(w), float(peaks), float(np.sum(w > 0))])
    out[9:14] = q
    out[14:] = qgrid
    return out


def _kernel():
    names, geometry, close_metrics = _implementations()

    @njit(cache=False)
    def aggregate(price, weight, starts, ends, closes):
        out = np.full((len(starts), len(BASE_FIELDS) + len(names)), np.nan)
        for i in range(len(starts)):
            a, b = starts[i], ends[i]
            p0, w0 = price[a:b], weight[a:b]
            good = np.all(np.isfinite(p0)) and np.all(p0 > 0)
            good = good and np.all(np.isfinite(w0)) and np.all(w0 >= 0)
            total = np.sum(w0)
            if not good or not np.isfinite(total) or total <= 0:
                continue
            order = np.argsort(p0)
            ps, ws = p0[order], w0[order]
            p, w = np.empty(len(ps)), np.empty(len(ps))
            n = 0
            for k in range(len(ps)):
                if n and ps[k] == p[n-1]:
                    w[n-1] += ws[k]
                else:
                    p[n], w[n] = ps[k], ws[k]
                    n += 1
            p, w = p[:n], w[:n] / total
            out[i, :len(BASE_FIELDS)] = _base(p, w)
            geo = geometry(p, w, closes[i])
            ng = len(geo)
            out[i, len(BASE_FIELDS):len(BASE_FIELDS)+ng] = geo
            out[i, len(BASE_FIELDS)+ng:] = close_metrics(p, w, closes[i])
        return out
    return aggregate


_AGGREGATE = None


def _source_identity(ctx, year):
    import pyarrow.parquet as pq
    files = {}
    for ds in ("stock_cyq_chips", "stock_daily"):
        p = ctx.cfg.upstream / ds / f"year={year}" / "data.parquet"
        st = p.stat() if p.exists() else None
        files[ds] = [str(p.resolve()), st.st_size, st.st_mtime_ns,
                     pq.ParquetFile(p).metadata.num_rows] if st else None
    code = {}
    for f in (Path(__file__), Path(__file__).with_name("chip_gc_geometry_v1.py"),
              Path(__file__).with_name("chip_gc_close_v1.py")):
        code[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return {"version": 1, "year": year, "sources": files, "code": code,
            "codes": hashlib.sha256("\n".join(map(str, ctx.panel.codes)).encode()).hexdigest(),
            "calendar": hashlib.sha256(np.asarray(ctx.cal.days, dtype="int32").tobytes()).hexdigest(),
            "cutoff": getattr(ctx.up, "audit_cutoff", None)}


def _root(ctx):
    if getattr(ctx.up, "audit_cutoff", None) is not None and not ctx.cfg.raw.get("_sandbox"):
        raise ValueError("raw-cutoff chip audits require an isolated sandbox")
    value = os.environ.get("FEA_CACHE_DIR") or ctx.cfg.raw.get("paths", {}).get("derived", "data/derived")
    base = Path(value)
    base = (base if base.is_absolute() else ctx.cfg.root / base).resolve()
    if not base.is_relative_to(ctx.cfg.root.resolve()):
        raise ValueError("chip cache must stay inside the execution namespace")
    return base / "chip_distribution_v1"


def _build_year(ctx, year):
    global _AGGREGATE
    names, _, _ = _implementations()
    fields = (*BASE_FIELDS, *names, *LEGACY_FIELDS)
    src = ctx.cfg.upstream / "stock_cyq_chips" / f"year={year}" / "data.parquet"
    if not src.exists():
        return pd.DataFrame(columns=["trade_date", "stock_code", *fields])
    cutoff = getattr(ctx.up, "audit_cutoff", None)
    close = ctx.up.read("stock_daily", columns=["trade_date", "stock_code", "close"],
                        years=(year, year), use_cache=False)
    codes = pd.Index(ctx.panel.codes)
    ci = codes.get_indexer(close.stock_code)
    day = series_to_int(close.trade_date)
    ok = ci >= 0
    keys = day[ok].astype(np.int64) * len(codes) + ci[ok]
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    values = pd.to_numeric(close.close, errors="coerce").to_numpy(np.float64)[ok][order]
    if len(np.unique(keys)) != len(keys):
        raise ValueError("duplicate close stock-day keys")
    frames = []
    if _AGGREGATE is None:
        _AGGREGATE = _kernel()
    for month in range(1, 13):
        lo = f"{year}-{month:02d}-01"
        hi = f"{year+1}-01-01" if month == 12 else f"{year}-{month+1:02d}-01"
        if cutoff is not None and int(lo.replace("-", "")) > cutoff:
            break
        filters = [("trade_date", ">=", lo), ("trade_date", "<", hi),
                   ("stock_code", "in", codes.tolist())]
        if cutoff is not None:
            filters.append(("trade_date", "<=", int_to_str(cutoff)))
        raw = pd.read_parquet(src, columns=["trade_date", "stock_code", "price", "percent"], filters=filters)
        if raw.empty:
            continue
        day = series_to_int(raw.trade_date)
        ci = codes.get_indexer(raw.stock_code)
        keep = (ci >= 0) & (day > 0)
        if not keep.all():
            raw = raw.loc[keep].reset_index(drop=True)
            day, ci = day[keep], ci[keep]
        if raw.empty:
            continue
        key = day.astype(np.int64) * len(codes) + ci
        order = np.argsort(key, kind="stable")
        key, day, ci = key[order], day[order], ci[order]
        price = pd.to_numeric(raw.price, errors="coerce").to_numpy(np.float64)[order]
        pct = pd.to_numeric(raw.percent, errors="coerce").to_numpy(np.float64)[order]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        ends = np.r_[starts[1:], len(key)]
        group = key[starts]
        cl = np.full(len(group), np.nan)
        if len(keys):
            ix = np.searchsorted(keys, group)
            hit = (ix < len(keys)) & (keys[np.minimum(ix, len(keys)-1)] == group)
            cl[hit] = values[ix[hit]]
        summary = _AGGREGATE(price, pct, starts, ends, cl)
        # Preserve the R38 NumPy log operation and the original within-day
        # summation order exactly. These legacy fields drop individual bad
        # bins, while the new descriptors intentionally reject an invalid day.
        group_id = np.repeat(np.arange(len(starts)), ends-starts)
        close_rows = cl[group_id]
        good = (np.isfinite(price) & (price > 0) & np.isfinite(pct) & (pct >= 0)
                & np.isfinite(close_rows) & (close_rows > 0))
        x = np.log(price[good] / close_rows[good])
        mass, group_id = pct[good], group_id[good]
        positive = x > 0
        selections = (np.ones(len(x), bool), positive, positive,
                      positive & (x <= np.log(1.10)), x > np.log(1.30),
                      np.abs(x) <= np.log(1.02))
        legacy = []
        for k, take in enumerate(selections):
            weights = mass[take] * x[take] if k == 1 else mass[take]
            legacy.append(np.bincount(group_id[take], weights=weights, minlength=len(starts)))
        # Groups without any valid legacy bin were absent in the old sparse
        # histogram, hence NaN rather than a fabricated zero observation.
        legacy = np.column_stack(legacy)
        count = np.bincount(group_id, minlength=len(starts))
        legacy[count == 0] = np.nan
        summary = np.column_stack((summary, legacy))
        frames.append(pd.DataFrame({"trade_date": int_to_str_vec(day[starts]),
                                    "stock_code": codes.to_numpy()[ci[starts]],
                                    **{f: summary[:, k] for k, f in enumerate(fields)}}))
        del raw, summary, price, pct
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["trade_date", "stock_code", *fields])


def _ensure(ctx, year):
    root = _root(ctx) / f"year={year}"
    root.mkdir(parents=True, exist_ok=True)
    path, manifest = root / "data.parquet", root / "identity.json"
    identity = _source_identity(ctx, year)
    # Audit namespaces are isolated, and cutoff is part of every disk identity.
    with (root / ".build.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            prior = json.loads(manifest.read_text())
        except (OSError, ValueError):
            prior = None
        if prior != identity or not path.exists():
            frame = _build_year(ctx, year)
            if _source_identity(ctx, year) != identity:
                raise RuntimeError("chip source changed during cache construction")
            _atomic_write(frame, path, ctx.cfg.compression)
            _atomic_json(identity, manifest)
    return path, identity


def field(ctx, name):
    """Exact daily placement; a bounded wide frame serves every factor worker."""
    ctx.accessed.update(("stock_cyq_chips", "stock_daily"))
    names, _, _ = _implementations()
    if name not in (*BASE_FIELDS, *names, *LEGACY_FIELDS):
        raise KeyError(name)
    years = (int(ctx.panel.dates[0]) // 10000, int(ctx.panel.dates[-1]) // 10000)
    paths, identities = [], []
    for year in range(years[0], years[1]+1):
        p, identity = _ensure(ctx, year)
        paths.append(p)
        identities.append(identity)
    key = json.dumps([str(_root(ctx)), identities], sort_keys=True)
    if _VIEW.get("key") != key:
        frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        _VIEW.clear()
        _VIEW.update(key=key, frame=frame)
    frame = _VIEW["frame"]
    ci = pd.Index(ctx.panel.codes).get_indexer(frame.stock_code)
    days = series_to_int(frame.trade_date)
    valid = ci >= 0
    cutoff = getattr(ctx.up, "audit_cutoff", None)
    if cutoff is not None:
        valid &= days <= cutoff
    return ctx.panel.place(ci[valid], days[valid], frame[name].to_numpy(np.float64)[valid])
