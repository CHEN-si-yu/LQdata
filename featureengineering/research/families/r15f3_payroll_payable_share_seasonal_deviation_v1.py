"""T3-C2 GC innovation family: 12-quarter seasonal/trend deviation of the
payroll-payable share of current liabilities (F001 estimator clone, new object).

ratio_q = point(payroll_payable_q) / point(total_cur_liab_q) (payroll > 0,
current liabilities > 1e6 RMB). The estimator is the frozen F001 clone:
X=[1,t,phase1..3] over 12 consecutive training quarters, rank 5 required,
residual df 7, rmse floor 1e-4, forecast row [1,12,phase]; value =
(ratio_q - forecast_q) / training_rmse. Every lookup is PIT-capped at the
event PIT (visibility discipline identical to F001). GC raw order only.
Object (payroll_payable, total_cur_liab) is disjoint from F001/CF06/IS05.
"""
import numpy as np

from research.families import r14f1_ccc_innovation_v2 as pit

WINDOW = 12
FULL_RANK = 5
RESIDUAL_DF = WINDOW - FULL_RANK
RMSE_FLOOR = 1e-4
NUMERATOR = 'payroll_payable'
DENOMINATOR = 'total_cur_liab'
FIELDS = (NUMERATOR, DENOMINATOR)
PARAMETERS = {
    'signal': 'payroll_payable_share_seasonal_trend_deviation',
    'numerator': NUMERATOR,
    'denominator': DENOMINATOR,
    'window_quarters': WINDOW,
    'full_rank': FULL_RANK,
    'residual_df': RESIDUAL_DF,
    'rmse_floor': RMSE_FLOOR,
    'version': 1,
}


def _ratio_lookup(vt, code, period, pitch):
    num = pit._lookup(vt, code, period, pitch, NUMERATOR)
    den = pit._lookup(vt, code, period, pitch, DENOMINATOR)
    valid = np.isfinite(num) & (num > 0) & np.isfinite(den) & (den > pit.AMOUNT_FLOOR)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = num / den
    valid &= np.isfinite(ratio)
    return ratio, valid


def _deviation_batch(events, vt):
    result = np.full(len(events), np.nan, dtype=np.float64)
    event_rows = np.flatnonzero(events.q.to_numpy(dtype=np.int64) >= 0)
    matched = events.iloc[event_rows].reset_index(drop=True)
    for lo in range(0, len(matched), pit.EVENT_BATCH):
        hi = min(lo + pit.EVENT_BATCH, len(matched))
        block = matched.iloc[lo:hi]
        code = block.code.to_numpy(dtype=np.int64)
        pitch = block.pit.to_numpy(dtype=np.int64)
        q = block.q.to_numpy(dtype=np.int64)
        periods = q[:, None] - np.arange(WINDOW, -1, -1, dtype=np.int64)[None, :]
        code_grid = np.broadcast_to(code[:, None], periods.shape)
        pit_grid = np.broadcast_to(pitch[:, None], periods.shape)

        ratio, valid = _ratio_lookup(vt, code_grid, periods, pit_grid)
        complete = valid.all(axis=1)
        if not complete.any():
            continue

        train = ratio[:, :WINDOW]
        current = ratio[:, WINDOW]
        for phase in range(4):
            rows = np.flatnonzero(complete & (q % 4 == phase))
            if not len(rows):
                continue
            train_periods = periods[rows[0], :WINDOW]
            t = np.arange(WINDOW, dtype=np.float64)
            X = np.column_stack([
                np.ones(WINDOW, dtype=np.float64), t,
                (train_periods % 4 == 1).astype(np.float64),
                (train_periods % 4 == 2).astype(np.float64),
                (train_periods % 4 == 3).astype(np.float64),
            ])
            if np.linalg.matrix_rank(X) != FULL_RANK:
                continue
            try:
                beta, _, rank, _ = np.linalg.lstsq(X, train[rows].T, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if rank != FULL_RANK:
                continue
            residual = train[rows].T - X @ beta
            rmse = np.sqrt(np.sum(residual * residual, axis=0) / RESIDUAL_DF)
            forecast_row = np.array([1.0, 12.0,
                                     float(phase == 1), float(phase == 2),
                                     float(phase == 3)], dtype=np.float64)
            forecast = forecast_row @ beta
            value = (current[rows] - forecast) / rmse
            ok = np.isfinite(rmse) & (rmse >= RMSE_FLOOR)
            ok &= np.isfinite(forecast) & np.isfinite(current[rows]) & np.isfinite(value)
            result[event_rows[lo + rows[ok]]] = value[ok]
    return result


def build(parameters=PARAMETERS):
    if parameters != PARAMETERS:
        raise ValueError('Only the frozen r15f3 definition is supported')

    def compute(ctx):
        data = pit._report_table(ctx, 'stock_balancesheet', FIELDS)
        if data.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        events = pit._frontier_events(data)
        if events.empty:
            return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
        vt = pit._sorted_versions(data, FIELDS)
        values = _deviation_batch(events, vt)
        return pit._map_fresh(ctx, events, values)

    return compute
