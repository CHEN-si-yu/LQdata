"""M002 preregistered GC descriptor: AM-minus-PM amount-scaled price movement.
Higher values are only a sorting convention, with no return direction claim.
Same-day unadjusted prices, amount in RMB, available after close. No date cap.
"""
from __future__ import annotations
import numpy as np

SCALE_RMB = 100_000_000.0


def build_m002(am_open, am_close, am_amount, pm_open, pm_close, pm_amount):
    """Return AM-minus-PM amount-scaled absolute log-return, aligned to inputs.

    Positive values mean the AM session moved more per traded RMB than the PM
    session. Inputs are same-day, unadjusted session prices and amount in RMB.
    Missing/zero amounts or invalid prices remain NaN.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (am_open, am_close, am_amount, pm_open, pm_close, pm_amount)]
    shape = arrays[0].shape
    if any(x.shape != shape for x in arrays):
        raise ValueError("all six inputs must have the same [day, stock] shape")
    ao, ac, aa, po, pc, pa = arrays
    valid = (np.isfinite(ao) & (ao > 0) & np.isfinite(ac) & (ac > 0) &
             np.isfinite(aa) & (aa > 0) & np.isfinite(po) & (po > 0) &
             np.isfinite(pc) & (pc > 0) & np.isfinite(pa) & (pa > 0))
    with np.errstate(all="ignore"):
        am_illiq = np.log1p(SCALE_RMB * np.abs(np.log(ac / ao)) / aa)
        pm_illiq = np.log1p(SCALE_RMB * np.abs(np.log(pc / po)) / pa)
        value = am_illiq - pm_illiq
    return np.where(valid & np.isfinite(value), value, np.nan)


PARAMETERS = {"signal": "session_illiquidity_asymmetry"}
FIELDS = ("am_open5", "am_close5", "am_amt", "pm_open5", "pm_close5", "pm_amt")

def build(parameters):
    if dict(parameters) != PARAMETERS:
        raise ValueError("Only the frozen M002 definition is permitted")
    def compute(ctx):
        value = build_m002(*(ctx.intraday_field(field) for field in FIELDS))
        universe = np.asarray(ctx.universe, dtype=bool)
        if universe.shape != value.shape:
            raise ValueError("universe/value shape mismatch")
        return np.where(universe, value, np.nan)
    return compute
