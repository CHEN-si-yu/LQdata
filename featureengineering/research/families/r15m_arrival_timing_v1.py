"""Frozen R15 seat_micro arrival-timing GC descriptors; source visibility exclusively ctx.dataset.
No disk/cache reads. Each annual frame is released and upstream cache cleared.
Eight freshly-parameterized objects share one exact-48-grid stock-day pass:
  amt_t50 (amount-weighted median arrival), amt_span (amount IQR span),
  vol_arrival_skew (volume-weighted arrival-time third standardized moment),
  rv_t50 (variance-mass median arrival), each as level20 (incl. today) and
  dev20 (minus prior-20 mean). The daily-statistic functions below are copied
  verbatim from probe_arrival_batch1.py (sha 7f4e211190f8de2fea623425bf8382f1e8ffaccacfbfe450a6c5fa3ce61386a8).
"""
from collections import deque
import gc
import numpy as np

SIGNALS = ('amt_t50_frac_20', 'amt_t50_dev_20', 'amt_span_frac_20', 'amt_span_dev_20',
           'vol_arrival_skew_20', 'vol_arrival_skew_dev_20', 'rv_t50_frac_20', 'rv_t50_dev_20')
COLUMNS = ['stock_code', 'trade_time', 'open', 'close', 'vol', 'amount']
GRID = tuple([f'{h:02d}:{m:02d}:00' for h in range(9, 12) for m in range(0, 60, 5) if '09:35' <= f'{h:02d}:{m:02d}' <= '11:30'] + [f'{h:02d}:{m:02d}:00' for h in range(13, 16) for m in range(0, 60, 5) if '13:05' <= f'{h:02d}:{m:02d}' <= '15:00'])
N = 48
CENTERS = np.arange(N, dtype=np.float64) + 0.5


def arrival_pos(mass, target):
    """Interpolated arrival position in slot units [0, 48] for cumulative
    fraction `target` in (0, 1). Slot k (0-based) spans [k, k+1]; its mass is
    uniform inside the slot. Requires nonnegative finite mass, total > 0."""
    m = np.asarray(mass, dtype=np.float64)
    total = m.sum()
    t = float(target) * total
    cum = np.cumsum(m)
    k = int(np.searchsorted(cum, t, side='left'))
    prev = cum[k - 1] if k > 0 else 0.0
    return float(k) + (t - prev) / m[k]


def day_stats(open_, close, vol, amount):
    """Four daily statistics from one exact-48 full-grid stock-day.

    Returns dict with keys t50_amt, span_amt, skew_vol, t50_rv (None when the
    specific requirement is not met). Arrays must already be aligned to SLOTS.
    """
    out = {'t50_amt': None, 'span_amt': None, 'skew_vol': None, 't50_rv': None}
    a = np.asarray(amount, dtype=np.float64)
    v = np.asarray(vol, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    o = np.asarray(open_, dtype=np.float64)
    if a.shape == (N,) and np.isfinite(a).all() and (a >= 0).all() and a.sum() > 0:
        out['t50_amt'] = arrival_pos(a, 0.5) / N
        out['span_amt'] = (arrival_pos(a, 0.75) - arrival_pos(a, 0.25)) / N
    if v.shape == (N,) and np.isfinite(v).all() and (v >= 0).all() and v.sum() > 0:
        s = v / v.sum()
        mu = float(np.sum(s * CENTERS))
        var = float(np.sum(s * (CENTERS - mu) ** 2))
        if var > 0:
            out['skew_vol'] = float(np.sum(s * (CENTERS - mu) ** 3) / var ** 1.5)
    if (c.shape == (N,) and np.isfinite(c).all() and (c > 0).all()
            and np.isfinite(o[0]) and o[0] > 0):
        m = np.zeros(N)
        m[0] = (c[0] / o[0] - 1.0) ** 2
        r = c[1:] / c[:-1] - 1.0
        m[1:] = r ** 2
        m[24] = 0.0  # lunch-crossing return excluded (r3b/r4b convention)
        if m.sum() > 0:
            out['t50_rv'] = arrival_pos(m, 0.5) / N
    return out


def daily_stats(open_, close, vol, amount):
    """Ordered 4-tuple view of day_stats: (amt_t50, amt_span, vol_skew, rv_t50)."""
    d = day_stats(open_, close, vol, amount)
    return (d['t50_amt'], d['span_amt'], d['skew_vol'], d['t50_rv'])


class Accumulator:
    """Same shape and operation order as the frozen r14m13_m14 Accumulator.
    values[2*s] = level20 of statistic s; values[2*s+1] = dev20 of statistic s."""

    def __init__(self, days, codes):
        self.days = np.asarray(days, dtype=np.int64); self.codes = np.asarray(codes).astype(str)
        if len(self.days) < 21 or np.any(np.diff(self.days) <= 0): raise ValueError('Need ordered calendar with20 prior rows')
        if len(set(self.codes)) != len(self.codes): raise ValueError('Duplicate stock axis')
        self.di = {str(int(d))[:4] + '-' + str(int(d))[4:6] + '-' + str(int(d))[6:8]: k for k, d in enumerate(self.days)}
        self.ci = {c: j for j, c in enumerate(self.codes)}; self.slot = {s: j for j, s in enumerate(GRID)}
        self.hist = {c: deque() for c in self.codes}; self.last = {}; self.key = None; self.rows = []
        shape = (len(days) - 20, len(codes)); self.values = [np.full(shape, np.nan) for _ in SIGNALS]

    def add(self, c, tt, open_, close, vol, amount):
        if c not in self.ci or not isinstance(tt, str) or tt[:10] not in self.di: return
        key = (c, tt[:10])
        if key != self.key: self.flush(); self.key = key; self.rows = []
        self.rows.append((self.slot.get(tt[11:], -1), open_, close, vol, amount))

    def flush(self):
        if self.key is None: return
        c, d = self.key; t = self.di[d]; j = self.ci[c]; rows = self.rows
        if c in self.last and t <= self.last[c]: raise ValueError('Repeated or nonmonotone stock-day')
        self.last[c] = t
        valid = len(rows) == 48 and sorted(z[0] for z in rows) == list(range(48))
        stats = (None, None, None, None)
        if valid:
            rows.sort()
            stats = daily_stats([z[1] for z in rows], [z[2] for z in rows], [z[3] for z in rows], [z[4] for z in rows])
        h = self.hist[c]
        while h and h[0][0] < t - 20: h.popleft()
        if t >= 20:
            o = t - 20
            for s in range(4):
                today = stats[s]
                if today is None: continue
                counts = [z[1][s] for z in h if z[0] >= t - 19 and z[1][s] is not None]
                counts.append(today)
                if len(counts) >= 15:
                    self.values[2 * s][o, j] = sum(counts) / len(counts)
                prior = [z[1][s] for z in h if z[1][s] is not None]
                if len(prior) >= 15:
                    self.values[2 * s + 1][o, j] = today - sum(prior) / len(prior)
        h.append((t, stats)); self.key = None; self.rows = []


def compute_all(ctx):
    dates = np.asarray(ctx.panel.dates, dtype=np.int64); calendar = np.asarray(ctx.cal.days, dtype=np.int64)
    if not len(dates) or np.any(np.diff(dates) <= 0): raise ValueError('Invalid panel dates')
    lo = np.searchsorted(calendar, dates[0]); hi = np.searchsorted(calendar, dates[-1])
    if lo < 20 or hi >= len(calendar) or not np.array_equal(calendar[lo:hi + 1], dates): raise ValueError('Full official calendar and20 prior rows required')
    axis = calendar[lo - 20:hi + 1]; acc = Accumulator(axis, ctx.panel.codes)
    for year in range(int(axis[0]) // 10000, int(axis[-1]) // 10000 + 1):
        frame = ctx.dataset('stock_history_5min', columns=COLUMNS, years=(year, year))
        try:
            if not frame.empty:
                if list(frame.columns) != COLUMNS: frame = frame[COLUMNS]
                for row in frame.itertuples(index=False, name=None): acc.add(*row)
                acc.flush()
        finally:
            del frame
            ctx.up.clear_cache()
            gc.collect()
    u = np.asarray(ctx.universe, dtype=bool)
    if u.shape != acc.values[0].shape: raise ValueError('Universe shape mismatch')
    return tuple(np.where(u, x, np.nan) for x in acc.values)


def build(parameters):
    if set(parameters) != {'signal'} or parameters['signal'] not in SIGNALS: raise ValueError('Frozen signals only')
    index = SIGNALS.index(parameters['signal'])

    def compute(ctx):
        return compute_all(ctx)[index]
    return compute
