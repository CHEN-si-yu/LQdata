"""Sole R7 market: disclosed earnings-change dispersion; weak true prior.

Quarterly operating differentiation is not belief disagreement or daily news.
Annual contemporaneous/price-leading literature is not 1/5-day proof. Frozen
review requires HAC19/63 and report-period contributions, not another signal.
Independent source implementation: never imports another research family.
"""
from __future__ import annotations
import datetime as _dt
import numpy as np
import pandas as pd

SIGNAL = 'disclosed_earnings_change_dispersion'
MIN_COUNT = 1692
STALE_DAYS = 550
AMOUNT_FLOOR = 1e6
PERIOD_W = 1_000_000
KEY_W = 100_000_000
COLUMNS = ['stock_code', 'ann_date', 'f_ann_date', 'end_date', 'n_income', 'revenue']
STATISTICS = ('primary', 'raw_change_median', 'asinh_change_median',
              'improvement_breadth', 'sales_growth_median', 'sales_growth_iqr',
              'net_margin_median', 'net_margin_iqr')
COMPONENTS = (*STATISTICS, 'report_period', 'report_quarter', 'report_age_days',
              'frontier_age_days', 'frontier_threshold_pit', 'disclosure_count',
              'valid_count', 'disclosure_fraction', 'valid_fraction',
              'sales_growth_valid_count', 'net_margin_valid_count')


def _ordinals(days):
    return np.array([_dt.date(int(d)//10000, int(d)//100 % 100, int(d) % 100).toordinal()
                     for d in days], dtype=np.int64)


def _table(ctx):
    frame = ctx.dataset('stock_income', columns=COLUMNS)
    names = ['code', 'period', 'pit', 'n_income', 'revenue']
    if frame is None or frame.empty:
        return pd.DataFrame(columns=names)
    end = np.asarray(ctx.date_col(frame.end_date), dtype=np.int64)
    ann = np.asarray(ctx.date_col(frame.ann_date), dtype=np.int64)
    fann = np.asarray(ctx.date_col(frame.f_ann_date), dtype=np.int64)
    pit = np.maximum(np.maximum(ann, fann), end)
    keep = (end >= 19900101) & np.isin(end % 10000, [331, 630, 930, 1231])
    keep &= (ann >= 19900101) | (fann >= 19900101)
    code = ctx.code_index(frame.stock_code.to_numpy()[keep])
    year = end[keep] // 10000
    quarter = np.searchsorted(np.array([331, 630, 930, 1231]), end[keep] % 10000)
    out = pd.DataFrame({'code': code, 'period': year*4+quarter, 'pit': pit[keep]})
    for field in ['n_income', 'revenue']:
        values = pd.to_numeric(frame[field], errors='coerce').to_numpy(np.float64)[keep]
        out[field] = np.where(np.isfinite(values), values, np.nan)
    # Last duplicate is a whole row, including missing legs. Never fieldwise coalesce.
    return out.loc[out.code >= 0].drop_duplicates(['code', 'period', 'pit'], keep='last')


def _versions(data):
    d = data.sort_values(['code', 'period', 'pit'], kind='stable')
    c = d.code.to_numpy(np.int64); q = d.period.to_numpy(np.int64)
    p = d.pit.to_numpy(np.int64)
    return {'keys': (c*PERIOD_W+q)*KEY_W+p, 'code': c, 'period': q,
            'n_income': d.n_income.to_numpy(np.float64),
            'revenue': d.revenue.to_numpy(np.float64)}


def _lookup(v, field, code, q, day):
    if len(v['keys']) == 0:
        return np.full(len(code), np.nan, dtype=np.float64)
    query = (code*PERIOD_W+q)*KEY_W+day
    # Strict PIT<current trading date; same-date reports/revisions never enter.
    pos = np.searchsorted(v['keys'], query, side='left')-1
    safe = np.maximum(pos, 0)
    good = (pos >= 0) & (v['code'][safe] == code) & (v['period'][safe] == q)
    return np.where(good, v[field][safe], np.nan)


def _single(v, field, code, q, day):
    current = _lookup(v, field, code, q, day)
    if q % 4 == 0:
        return current
    with np.errstate(all='ignore'):
        result = current-_lookup(v, field, code, q-1, day)
    return np.where(np.isfinite(result), result, np.nan)


def _iqr(x):
    q = np.quantile(x, [.25, .75], method='linear')
    return float(q[1]-q[0])


def _snapshot(v, q, day, code_count, minimum):
    code = np.arange(code_count, dtype=np.int64)
    profit = _single(v, 'n_income', code, q, day)
    profit_lag = _single(v, 'n_income', code, q-4, day)
    revenue_lag = _single(v, 'revenue', code, q-4, day)
    valid = np.isfinite(profit) & np.isfinite(profit_lag) & np.isfinite(revenue_lag)
    valid &= revenue_lag > AMOUNT_FLOOR
    with np.errstate(all='ignore'):
        raw = (profit-profit_lag)/np.where(valid, revenue_lag, np.nan)
        transformed = np.arcsinh(raw)  # dimensionless scale exactly 1
    valid &= np.isfinite(raw) & np.isfinite(transformed)
    result = {k: np.nan for k in STATISTICS}
    result.update(valid_count=float(valid.sum()), sales_growth_valid_count=0.,
                  net_margin_valid_count=0.)
    if valid.sum() < minimum:
        return result
    z = raw[valid]; x = transformed[valid]
    result.update(primary=_iqr(x), raw_change_median=float(np.median(z)),
                  asinh_change_median=float(np.median(x)),
                  improvement_breadth=float(np.mean(z > 0)))
    # Diagnostics only: current revenue is deliberately absent from primary validity.
    revenue = _single(v, 'revenue', code, q, day)
    good = valid & np.isfinite(revenue) & (revenue > AMOUNT_FLOOR)
    with np.errstate(all='ignore'):
        growth = (revenue-revenue_lag)/revenue_lag
        margin = profit/revenue
    sg = good & np.isfinite(growth); mg = good & np.isfinite(margin)
    result['sales_growth_valid_count'] = float(sg.sum())
    result['net_margin_valid_count'] = float(mg.sum())
    if sg.sum() >= minimum:
        result.update(sales_growth_median=float(np.median(growth[sg])),
                      sales_growth_iqr=_iqr(growth[sg]))
    if mg.sum() >= minimum:
        result.update(net_margin_median=float(np.median(margin[mg])),
                      net_margin_iqr=_iqr(margin[mg]))
    return result


def _path(data, dates, code_count, minimum=MIN_COUNT):
    """Pure-memory state path. Nondefault minimum is solely for tiny unit fixtures.

    Registered builders always pass fixed 1692, with the official 2115 code axis.
    All components use identical source PIT/frontier. No return data is accessed.
    """
    dates = np.asarray(dates, dtype=np.int64)
    if dates.ndim != 1 or (len(dates)>1 and np.any(dates[1:] <= dates[:-1])):
        raise ValueError('Unique ascending trading dates required')
    if not 1 <= minimum <= code_count:
        raise ValueError('Invalid fixture/code-axis threshold')
    out = {k: np.full(len(dates), np.nan, dtype=np.float64) for k in COMPONENTS}
    if data.empty or len(dates) == 0:
        return out
    # Unknown stocks were excluded in _table; fixture helper also guards index scope.
    data = data.loc[(data.code >= 0) & (data.code < code_count)]
    if data.empty:
        return out
    first = data.groupby(['code', 'period'], sort=False).pit.min().reset_index()
    first_by_q = {int(q): np.sort(g.pit.to_numpy(np.int64))
                  for q, g in first.groupby('period', sort=False)}
    frontier = np.full(len(dates), -1, dtype=np.int64)
    threshold_pit = {}
    for q, p in first_by_q.items():
        if len(p) < minimum:
            continue
        gate = int(p[minimum-1]); idx = int(np.searchsorted(dates, gate, side='right'))
        threshold_pit[q] = gate
        if idx < len(dates):
            frontier[idx] = max(frontier[idx], q)
    frontier = np.maximum.accumulate(frontier)  # no report fallback, even after NaN revisions
    releases = set(np.searchsorted(dates, data.pit.to_numpy(np.int64), side='right').tolist())
    v = _versions(data); ords = _ordinals(dates)
    cached = None; prior_q = -1
    for t, q in enumerate(frontier):
        if q < 0:
            continue
        if q != prior_q or t in releases or cached is None:
            cached = _snapshot(v, int(q), int(dates[t]), code_count, minimum)
        end = int(q//4*10000+np.array([331,630,930,1231])[q % 4])
        age = int(ords[t]-_ordinals([end])[0])
        # Age of the observable kth-first-PIT anchor, invariant to panel start.
        pit = threshold_pit[int(q)]
        pit_age = int(ords[t]-_ordinals([pit])[0])
        disclosed = int(np.searchsorted(first_by_q[int(q)], dates[t], side='left'))
        for k in STATISTICS:
            out[k][t] = cached[k] if 0 <= age <= STALE_DAYS else np.nan
        for k in ['valid_count','sales_growth_valid_count','net_margin_valid_count']:
            out[k][t] = cached[k]
        out['report_period'][t] = end
        out['report_quarter'][t] = q % 4+1
        out['report_age_days'][t] = age
        out['frontier_age_days'][t] = pit_age
        out['frontier_threshold_pit'][t] = pit
        out['disclosure_count'][t] = disclosed
        out['disclosure_fraction'][t] = disclosed/code_count
        out['valid_fraction'][t] = cached['valid_count']/code_count
        prior_q = int(q)
    return out


def diagnostic_components(ctx):
    """T-length primary/state/center/financial controls; SUE controls remain external.

    Main center control is raw_change_median. asinh_change_median is a substitute
    diagnostic, never an independent additional control. Counts are retained on
    invalid/stale dates to explain coverage, while primary/statistics remain NaN.
    frontier_age_days is calendar time since kth-first-retained PIT, not a claim
    of actual vendor arrival; availability still strictly waits until next TD.
    """
    if len(ctx.panel.codes) != 2115:
        raise ValueError('Registered R7M requires the official fixed 2115-code axis')
    return _path(_table(ctx), ctx.panel.dates, 2115, minimum=MIN_COUNT)


def build(parameters):
    if parameters != {'signal': SIGNAL}:
        raise ValueError('R7M permits exactly the single frozen definition')
    def compute(ctx):
        return diagnostic_components(ctx)['primary']
    return compute
