"""T3 GC family: balance-sheet pool composition shares (one parameterized builder).

POOLS is the lookup table: pool -> member legs. Each candidate's parameters
select {'pool': ..., 'numerator': ..., 'version': 1}; the computed series is the
numerator's point-balance share of the declared same-class pool sum.
"""
import numpy as np
from research.families import r14f1_ccc_innovation_v2 as pit

POOLS = {
    'operating_assets': ('inventories', 'accounts_receiv', 'prepayment'),
    'longterm_productive': ('fix_assets', 'cip', 'intan_assets'),
}
VERSION = 1


def member_parameters(pool, numerator):
    if pool not in POOLS or numerator not in POOLS[pool]:
        raise ValueError('Unknown pool-composition member')
    return {'pool': pool, 'numerator': numerator, 'version': VERSION}


def event_values(data, fields, numerator):
    e = pit._frontier_events(data)
    v = pit._sorted_versions(data, fields)
    c = e.code.to_numpy(dtype=np.int64)
    q = e.q.to_numpy(dtype=np.int64)
    cap = e.pit.to_numpy(dtype=np.int64)
    total = np.zeros(len(c), dtype=np.float64)
    ok = np.ones(len(c), dtype=bool)
    for field in fields:
        leg = pit._lookup(v, c, q, cap, field)
        ok &= np.isfinite(leg) & (leg > 0)
        total = total + np.where(np.isfinite(leg), leg, 0.0)
    ok &= total > 1e6
    num = pit._lookup(v, c, q, cap, numerator)
    with np.errstate(divide='ignore', invalid='ignore'):
        values = np.where(ok, num / total, np.nan)
    return e, values


def build(parameters):
    params = dict(parameters)
    pool = params.get('pool')
    numerator = params.get('numerator')
    if pool not in POOLS or numerator not in POOLS[pool] or params.get('version') != VERSION:
        raise ValueError('Unknown frozen pool-composition member')
    fields = tuple(POOLS[pool])

    def compute(ctx):
        data = pit._report_table(ctx, 'stock_balancesheet', fields)
        e, v = event_values(data, fields, numerator)
        return pit._map_fresh(ctx, e, v)
    return compute
