"""fourth batch: raw 5-minute internal-structure statistics.

One quarter-chunked scan builds one row per stock-day. All candidate compute
functions reuse that cache; no candidate independently scans minute bars.
"""
from __future__ import annotations

import hashlib
import logging
import fcntl
from pathlib import Path

import numpy as np
import pandas as pd

from fea.derived import DerivedCache
from fea.dates import int_to_str_vec, series_to_int
from fea.upstream import Upstream
from fea.closing30 import EXPECTED_HM

log = logging.getLogger("factor.minute_structure")


def fixed_window_mean(values, width, min_count=None):
    """Ordered short-window mean, embedded to keep this module self-contained."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"fixed_window_mean expects (T,C), got {x.shape}")
    width = int(width)
    if width <= 0:
        raise ValueError("width must be positive")
    need = width if min_count is None else int(min_count)
    if need < 0 or need > width:
        raise ValueError("min_count must be in [0,width]")
    t, c = x.shape
    sums = np.zeros((t, c), dtype=np.float64)
    count = np.zeros((t, c), dtype=np.int16)
    for lag in range(width):
        shifted = np.full((t, c), np.nan, dtype=np.float64)
        if lag == 0:
            shifted[:] = x
        elif lag < t:
            shifted[lag:] = x[:-lag]
        finite = np.isfinite(shifted)
        sums += np.where(finite, shifted, 0.0)
        count += finite.astype(np.int16)
    out = np.full((t, c), np.nan, dtype=np.float64)
    good = count >= need
    np.divide(sums, count, out=out, where=good & (count > 0))
    if need == 0:
        out[count == 0] = 0.0
    return out

SIGNALS = (
    "vwap_vs_twap",
    "efficient_bar_smart_q",
    "return_volume_lead_lag",
    "minute_price_volume_corr",
    "minute_return_volume_corr",
    "unexpected_clock_volume_return",
)
N_BARS = 48
SLOT_FIELDS = tuple(f"ret_{i:02d}" for i in range(N_BARS)) + tuple(
    f"amt_share_{i:02d}" for i in range(N_BARS)
)
SCALAR_FIELDS = (
    "n_bars", "grid_ok", "vwap_twap_q", "efficient_bar_smart_q",
    "return_volume_lead_lag", "minute_price_volume_corr",
    "minute_return_volume_corr",
)
FIELDS = [*SCALAR_FIELDS, *SLOT_FIELDS]


def _corr(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 12:
        return np.nan
    x, y = x[ok], y[ok]
    x, y = x - x.mean(), y - y.mean()
    den = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / den) if den > 1e-15 else np.nan


def _row_corr(x, y, min_count=12):
    """Vectorized row-wise Pearson correlation with pairwise finite values."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(axis=1)
    xx = np.where(ok, x, 0.0)
    yy = np.where(ok, y, 0.0)
    den_n = np.maximum(n, 1)
    mx = xx.sum(axis=1) / den_n
    my = yy.sum(axis=1) / den_n
    xc = np.where(ok, x - mx[:, None], 0.0)
    yc = np.where(ok, y - my[:, None], 0.0)
    xy = np.sum(xc * yc, axis=1)
    xx2 = np.sum(xc * xc, axis=1)
    yy2 = np.sum(yc * yc, axis=1)
    den = np.sqrt(xx2 * yy2)
    out = np.full(n.shape, np.nan, dtype=np.float64)
    valid = (n >= min_count) & (den > 1e-15)
    out[valid] = xy[valid] / den[valid]
    return out


class MinuteStructureLayer(DerivedCache):
    """Full-history daily cache; each aggregate uses only one stock and day."""
    name = "minute_structure_v1"
    version = 1

    def _fp(self, year):
        # This cache intentionally covers complete years. A full per-stock-day
        # aggregate is safe to reuse at any cutoff because no day reads another
        # day; _read_all applies the active audit cutoff before panel placement.
        cache = self.__dict__.setdefault("_fp_cache", {})
        if year not in cache:
            fp = self.source_fingerprint(year)
            fp["universe_sha256"] = hashlib.sha256(
                "\n".join(map(str, self.codes)).encode()).hexdigest()
            cache[year] = fp
        return cache[year]

    def source_fingerprint(self, year):
        src = self.cfg.upstream / "stock_history_5min" / f"year={year}" / "data.parquet"
        if not src.exists():
            return {"exists": False}
        import pyarrow.parquet as pq
        st = src.stat()
        return {"exists": True, "rows": pq.ParquetFile(src).metadata.num_rows,
                "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}

    def ensure(self, y0: int, y1: int) -> None:
        """Build each year once under a year lock; merge manifest under a short lock.

        Adjacent production tasks have overlapping warmup years. The base
        DerivedCache.ensure is intentionally single-writer; this family can be
        requested concurrently, so it uses per-year build locks and a separate
        manifest read/merge/write lock. Different years remain parallel.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_lock = self.root / ".manifest.lock"
        for year in range(int(y0), int(y1) + 1):
            fp = self._fp(year)
            if not fp.get("exists"):
                continue
            target = self.root / f"year={year}" / "data.parquet"
            year_lock = self.root / f".build-year-{year}.lock"
            with year_lock.open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                with manifest_lock.open("a+") as mlock:
                    fcntl.flock(mlock, fcntl.LOCK_EX)
                    manifest = self._load_man()
                    if manifest.get("version") != self.version:
                        manifest = {"version": self.version, "years": {}}
                    current = (manifest.get("years", {}).get(str(year)) == fp and
                               target.exists())
                    fcntl.flock(mlock, fcntl.LOCK_UN)
                if current:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    continue
                log.info("%s: building year=%d", self.name, year)
                frame = self.build_year(year, self.up)
                fp_after = self.source_fingerprint(year)
                fp_after["universe_sha256"] = hashlib.sha256(
                    "\n".join(map(str, self.codes)).encode()).hexdigest()
                if fp_after != fp:
                    raise RuntimeError(f"{self.name} year={year}: source changed during aggregation")
                if frame is None or not len(frame.columns):
                    frame = pd.DataFrame(columns=["trade_date", "stock_code", *FIELDS])
                from fea import store
                store._atomic_write(frame, target, self.cfg.compression)
                with manifest_lock.open("a+") as mlock:
                    fcntl.flock(mlock, fcntl.LOCK_EX)
                    manifest = self._load_man()
                    if manifest.get("version") != self.version:
                        manifest = {"version": self.version, "years": {}}
                    manifest.setdefault("years", {})[str(year)] = fp_after
                    self._save_man(manifest)
                    fcntl.flock(mlock, fcntl.LOCK_UN)
                self.clear_raw()
                self._cache.clear()
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read_all(self, y0: int, y1: int) -> pd.DataFrame:
        # Base DerivedCache intentionally keeps a window frame in memory. Its
        # frame is already cropped by _audit_frame, so switching cutoffs on the
        # same layer must invalidate it before a wider cutoff can be requested.
        cutoff = getattr(self.up, "audit_cutoff", None)
        current = None if cutoff is None else int(cutoff)
        prior = self.__dict__.get("_luna_audit_cutoff", object())
        if prior != current:
            self._frame = None
            self._frame_window = None
            self._cache.clear()
            self.__dict__["_luna_audit_cutoff"] = current
        return super()._read_all(y0, y1)

    def build_year(self, year, up):
        src = self.cfg.upstream / "stock_history_5min" / f"year={year}" / "data.parquet"
        if not src.exists():
            return pd.DataFrame()
        frames = []
        for month in (1, 4, 7, 10):
            lo = f"{year}-{month:02d}-01"
            hi = f"{year + 1}-01-01" if month == 10 else f"{year}-{month + 3:02d}-01"
            df = pd.read_parquet(
                src, columns=["stock_code", "trade_time", "close", "vol", "amount"],
                filters=[("trade_time", ">=", lo), ("trade_time", "<", hi),
                         ("stock_code", "in", self.codes.tolist())])
            frame = self._aggregate(df)
            if not frame.empty:
                frames.append(frame)
            del df
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _aggregate(self, df):
        if df.empty:
            return pd.DataFrame()
        cpos = pd.Series(np.arange(len(self.codes), dtype=np.int32), index=self.codes)
        ci = cpos.reindex(df.stock_code.to_numpy()).to_numpy()
        keep = np.isfinite(ci)
        df = df.loc[keep]
        ci = ci[keep].astype(np.int32)
        tt = df.trade_time.astype("string")
        day = series_to_int(tt.str.slice(0, 10))
        hm = (tt.str.slice(11, 13).astype("int32") * 100 +
              tt.str.slice(14, 16).astype("int32")).to_numpy(dtype=np.int32)
        close = pd.to_numeric(df.close, errors="coerce").to_numpy(np.float64)
        vol = pd.to_numeric(df.vol, errors="coerce").to_numpy(np.float64)
        amt = pd.to_numeric(df.amount, errors="coerce").to_numpy(np.float64)

        order = np.lexsort((hm, day, ci))
        ci, day, hm = ci[order], day[order], hm[order]
        close, vol, amt = close[order], vol[order], amt[order]
        key = ci.astype(np.int64) * 100_000_000 + day.astype(np.int64)
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        ends = np.r_[starts[1:], len(key)]
        count = ends - starts
        ix = np.flatnonzero(count == N_BARS)
        if not ix.size:
            return pd.DataFrame()
        grid_ix = starts[ix, None] + np.arange(N_BARS, dtype=np.int64)[None, :]
        grid_ok = np.all(hm[grid_ix] == EXPECTED_HM[None, :], axis=1)
        # Reject incomplete/bad price-volume rows as a whole stock-day. A
        # repeated or missing timestamp can never become a 48-slot signal.
        valid = (grid_ok & np.all(np.isfinite(close[grid_ix]) & (close[grid_ix] > 0), axis=1) &
                 np.all(np.isfinite(vol[grid_ix]) & (vol[grid_ix] > 0), axis=1) &
                 np.all(np.isfinite(amt[grid_ix]) & (amt[grid_ix] > 0), axis=1))
        ix, grid_ix = ix[valid], grid_ix[valid]
        if not ix.size:
            return pd.DataFrame()
        c, v, a = close[grid_ix], vol[grid_ix], amt[grid_ix]
        # Match IntradayLayer's hand-vs-share normalization per stock-day.
        raw_day_vwap = a.sum(axis=1) / v.sum(axis=1)
        mult = np.where(raw_day_vwap / c[:, -1] > 10.0, 100.0, 1.0)
        v = v * mult[:, None]
        bar_vwap = a / v
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.log(c[:, 1:] / c[:, :-1])
            logv = np.log(v)
            vwap = a.sum(axis=1) / v.sum(axis=1)
            twap = np.mean(bar_vwap, axis=1)
            q_twap = np.log(vwap / twap)

        # Rank bars by absolute return per square-root share volume. Select 10
        # of the 48 bars, with the first bar unrankable due to no prior close.
        score = np.full((len(ix), N_BARS), -np.inf, dtype=np.float64)
        score[:, 1:] = np.abs(ret) / np.sqrt(v[:, 1:])
        top = np.argpartition(score, -10, axis=1)[:, -10:]
        rows = np.arange(len(ix))[:, None]
        smart_vwap = a[rows, top].sum(axis=1) / v[rows, top].sum(axis=1)
        q_smart = np.log(smart_vwap / vwap)

        # Return at t against next- versus previous-bar volume. Interior bars
        # only so both lag directions use the same observations.
        # ret[:,0:46] is the return ending at bars 1..46. Compare it with
        # volume at the next bar (2..47) and previous bar (0..45), respectively.
        lead_lag = (_row_corr(ret[:, 0:46], logv[:, 2:48]) -
                    _row_corr(ret[:, 0:46], logv[:, 0:46]))
        price_corr = _row_corr(np.log(c), logv)
        ret_corr = _row_corr(ret, logv[:, 1:])
        share = a / a.sum(axis=1, keepdims=True)
        slot_ret = np.column_stack([np.full(len(ix), np.nan), ret])

        day0 = day[starts[ix]]
        code0 = self.codes[ci[starts[ix]]]
        data = {
            "trade_date": int_to_str_vec(day0), "stock_code": code0,
            "n_bars": np.full(len(ix), N_BARS, dtype=np.float64),
            "grid_ok": np.ones(len(ix), dtype=np.float64),
            "vwap_twap_q": q_twap, "efficient_bar_smart_q": q_smart,
            "return_volume_lead_lag": lead_lag,
            "minute_price_volume_corr": price_corr,
            "minute_return_volume_corr": ret_corr,
        }
        data.update({f"ret_{j:02d}": slot_ret[:, j] for j in range(N_BARS)})
        data.update({f"amt_share_{j:02d}": share[:, j] for j in range(N_BARS)})
        return pd.DataFrame(data)

    def panel_all(self, panel):
        cutoff = getattr(self.up, "audit_cutoff", None)
        key = (int(panel.dates[0]), int(panel.dates[-1]), panel.C,
               None if cutoff is None else int(cutoff))
        hit = self._cache.get((key, "all"))
        if hit is not None:
            return hit
        frame = self._read_all(int(panel.dates[0]) // 10000,
                               int(panel.dates[-1]) // 10000)
        if frame.empty:
            base = {name: np.full(panel.shape, np.nan, dtype=np.float64)
                    for name in SCALAR_FIELDS}
            slots = [np.full((*panel.shape, N_BARS), np.nan, dtype=np.float64),
                     np.full((*panel.shape, N_BARS), np.nan, dtype=np.float64)]
        else:
            pos = pd.Series(np.arange(panel.C, dtype=np.int32), index=panel.codes)
            code_ix = pos.reindex(frame.stock_code.to_numpy()).to_numpy()
            keep = np.isfinite(code_ix)
            frame = frame.loc[keep]
            code_ix = code_ix[keep].astype(np.int32)
            days = Upstream._audit_dates(frame.trade_date)
            scalars = {
                name: panel.place(code_ix, days, frame[name].to_numpy(np.float64))
                for name in SCALAR_FIELDS
            }
            slot_arrays = []
            for prefix in ("ret", "amt_share"):
                cols = [f"{prefix}_{j:02d}" for j in range(N_BARS)]
                slot_arrays.append(np.stack([
                    panel.place(code_ix, days, frame[col].to_numpy(np.float64)) for col in cols
                ], axis=2))
            base = scalars
            slots = slot_arrays
        result = (base, slots)
        self._cache[(key, "all")] = result
        return result


def _layer(ctx):
    ctx.accessed.add("stock_history_5min")
    # One in-memory accessor per context; the persisted cache is shared by all
    # candidates and is keyed by source identity plus the 50/production universe.
    layer = ctx.__dict__.get("_minute_structure_v1")
    if layer is None:
        layer = MinuteStructureLayer(ctx.up, ctx.cfg, ctx.panel.codes)
        ctx.__dict__["_minute_structure_v1"] = layer
    layer.ensure(int(ctx.panel.dates[0]) // 10000, int(ctx.panel.dates[-1]) // 10000)
    return layer


def _mean20(ctx, daily):
    daily = np.asarray(daily, dtype=np.float64)
    n = _layer(ctx).panel_all(ctx.panel)[0]["n_bars"]
    good = n == 48
    x = np.where(good, daily, np.nan)
    out = fixed_window_mean(x, 20, min_count=16)
    return np.where(np.isfinite(x), out, np.nan)


def _compute(ctx, signal):
    layer = _layer(ctx)
    scalar, slots = layer.panel_all(ctx.panel)
    if signal == "unexpected_clock_volume_return":
        rets, shares = slots
        surprise = np.zeros(ctx.panel.shape, dtype=np.float64)
        ok = np.ones(ctx.panel.shape, dtype=bool)
        # Bar 0 has no preceding within-day close, so it has no return. Use
        # only the 47 observable adjacent-bar returns (slots 1..47).
        for j in range(1, N_BARS):
            expected = fixed_window_mean(ctx.shift(shares[:, :, j], 1), 20, min_count=16)
            ok &= np.isfinite(expected) & np.isfinite(rets[:, :, j])
            surprise += (shares[:, :, j] - expected) * rets[:, :, j]
        daily = np.where(ok, surprise, np.nan)
    else:
        scalar_field = "vwap_twap_q" if signal == "vwap_vs_twap" else signal
        daily = scalar[scalar_field]
    value = _mean20(ctx, daily)
    current_good = scalar["grid_ok"] == 1
    traded = np.asarray(ctx.traded(), dtype=bool)
    valid = (np.asarray(ctx.universe, dtype=bool) & traded & current_good & np.isfinite(value))
    return np.where(valid, value, np.nan)


def build(parameters):
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError(f"signal must be one of {SIGNALS}")
    signal = parameters["signal"]
    return lambda ctx: _compute(ctx, signal)
