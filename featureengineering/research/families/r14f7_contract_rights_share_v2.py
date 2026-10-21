"""Isolated F007 conservative v2 GC draft. No production registration. Reuses frozen PIT helpers."""
import numpy as np
from research.families import r14f1_ccc_innovation_v2 as pit
PARAMETERS = {'signal': 'positive_net_conditional_customer_rights_share_v2', 'start': 20200101, 'minimum_report_end': 20200331}
FIELDS = ('contract_assets', 'accounts_receiv')

def event_values(data):
    events = pit._frontier_events(data)
    vt = pit._sorted_versions(data, FIELDS)
    code = events.code.to_numpy(dtype=np.int64)
    q = events.q.to_numpy(dtype=np.int64)
    cap = events.pit.to_numpy(dtype=np.int64)
    ca = pit._lookup(vt, code, q, cap, FIELDS[0])
    ar = pit._lookup(vt, code, q, cap, FIELDS[1])
    den = ca + ar
    valid = (pit._period_end(q) >= 20200331) & np.isfinite(ca) & np.isfinite(ar) & (ca > 0) & (ar > 0) & (den > 1e6)
    with np.errstate(divide='ignore', invalid='ignore'):
        value = np.where(valid, ca / den, np.nan)
    return events, value

def build(parameters=PARAMETERS):
    if parameters != PARAMETERS:
        raise ValueError('Frozen F007 draft only')
    def compute(ctx):
        data = pit._report_table(ctx, 'stock_balancesheet', FIELDS)
        events, value = event_values(data)
        mapped = pit._map_fresh(ctx, events, value)
        mapped[ctx.panel.dates < 20200101] = np.nan
        return mapped
    return compute
