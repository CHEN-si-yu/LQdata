"""T3-C2 GC family: liability/equity-side pool composition shares (one parameterized builder).

POOLS is the lookup table: pool -> (legs, policy). A leg is a tuple of field
names summed into one leg (composite leg, e.g. 预收款项+合同负债 counted as one
economic leg). Each candidate's parameters select {'pool': ..., 'numerator':
..., 'version': 1}; the computed series is the numerator leg's point-balance
share of the declared same-class pool sum.

Finiteness requires every component of every leg (a missing component makes the
whole state NaN and clears); the value condition applies to the leg SUM
(policy 'strict': every leg > 0, zero treated as undisclosed; policy 'nonneg':
every leg >= 0, genuine zeros kept, e.g. non-bond issuers). Pool sum must
exceed 1e6 RMB. Frozen by seat_finance/t3c2_design_freeze_v1.json +
amendment1 (equity pair swapped to (surplus_rese, undistr_porfit), cap_rese
closed by identity).
"""
import numpy as np
from research.families import r14f1_ccc_innovation_v2 as pit

VERSION = 1
AMOUNT_FLOOR = 1_000_000.0
POOLS = {
    'current_liab_core': {
        'legs': (('accounts_pay',), ('adv_receipts', 'contract_liab'), ('oth_pay_total',)),
        'policy': 'strict',
    },
    'noncurr_liab_core': {
        'legs': (('lt_borr',), ('bond_payable',), ('defer_tax_liab',)),
        'policy': 'nonneg',
    },
    'equity_reserve': {
        'legs': (('cap_rese',), ('surplus_rese',), ('undistr_porfit',)),
        'policy': 'strict',
    },
}


def leg_id(leg):
    return '+'.join(leg)


def pool_fields(pool):
    return tuple(sorted({field for leg in POOLS[pool]['legs'] for field in leg}))


def member_parameters(pool, numerator):
    if pool not in POOLS or numerator not in [leg_id(leg) for leg in POOLS[pool]['legs']]:
        raise ValueError('Unknown pool-composition member')
    return {'pool': pool, 'numerator': numerator, 'version': VERSION}


def event_values(data, pool, numerator):
    spec = POOLS[pool]
    e = pit._frontier_events(data)
    v = pit._sorted_versions(data, pool_fields(pool))
    c = e.code.to_numpy(dtype=np.int64)
    q = e.q.to_numpy(dtype=np.int64)
    cap = e.pit.to_numpy(dtype=np.int64)
    total = np.zeros(len(c), dtype=np.float64)
    num = np.full(len(c), np.nan, dtype=np.float64)
    ok = np.ones(len(c), dtype=bool)
    for leg in spec['legs']:
        val = np.zeros(len(c), dtype=np.float64)
        finite = np.ones(len(c), dtype=bool)
        for field in leg:
            x = pit._lookup(v, c, q, cap, field)
            finite &= np.isfinite(x)
            val = val + np.where(np.isfinite(x), x, 0.0)
        if spec['policy'] == 'strict':
            ok &= finite & (val > 0)
        else:
            ok &= finite & (val >= 0)
        total = total + val
        if leg_id(leg) == numerator:
            num = val
    ok &= total > AMOUNT_FLOOR
    with np.errstate(divide='ignore', invalid='ignore'):
        values = np.where(ok, num / total, np.nan)
    return e, values


def build(parameters):
    params = dict(parameters)
    pool = params.get('pool')
    numerator = params.get('numerator')
    if pool not in POOLS or params.get('version') != VERSION:
        raise ValueError('Unknown frozen pool-composition member')
    if numerator not in [leg_id(leg) for leg in POOLS[pool]['legs']]:
        raise ValueError('Unknown frozen pool-composition member')

    def compute(ctx):
        data = pit._report_table(ctx, 'stock_balancesheet', pool_fields(pool))
        e, v = event_values(data, pool, numerator)
        return pit._map_fresh(ctx, e, v)
    return compute
