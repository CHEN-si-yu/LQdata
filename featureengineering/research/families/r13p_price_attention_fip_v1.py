"""Frozen R13-P design module: one Frog-in-the-Pan interaction candidate.

This remains inside the research artifact directory for independent review. It
is not registered and must not be moved into shared factor source before the
lead's implementation audit.

The original definition is ``-PRET * ID``, where
``ID = sign(PRET) * (p_neg - p_pos)``. Therefore the exact identity is
``-PRET * ID = abs(PRET) * (p_pos - p_neg)``. ID summarizes the same daily
return signs; it is not an independent source of information.
"""
from __future__ import annotations

import numpy as np


PARAMETERS = {
    "signal": "fip_return_discreteness_interaction",
    "formation_slots": 252,
    "formation_lag": 21,
    "skip_slots": 21,
    "min_formation_valid": 200,
    "min_skip_valid": 16,
    "warmup_days": 520,
}


def build(parameters):
    """Build only the lead-frozen R13-P definition; no window search is allowed."""
    if parameters != PARAMETERS:
        raise ValueError("Only the single frozen R13-P FIP definition is permitted")

    def compute(ctx):
        # All source returns stay aligned to the full calendar panel. Nonfinite
        # and impossible (< -100%) input returns are missing; an actual -100%
        # return remains valid and makes the compounded formation return -100%.
        r = np.asarray(ctx.ret_clean(1), dtype=np.float64)
        traded = np.asarray(ctx.traded(), dtype=bool)
        univ = np.asarray(ctx.universe, dtype=bool)
        if r.ndim != 2 or traded.shape != r.shape or univ.shape != r.shape:
            raise ValueError("R13-P requires matching date-by-stock panels")
        valid_input = np.isfinite(r) & traded & (r >= -1.0)
        r = np.where(valid_input, r, np.nan)

        formation_slots = PARAMETERS["formation_slots"]
        formation_lag = PARAMETERS["formation_lag"]
        skip_slots = PARAMETERS["skip_slots"]

        # At signal date T, shifting by 21 then rolling 252 calendar rows uses
        # exactly [T-272, T-21]. The skip count uses [T-20, T]. Neither window
        # compresses around NaNs or suspensions.
        r_form = ctx.shift(r, formation_lag)
        valid_form = np.isfinite(r_form)
        n_form = ctx.roll_count(r_form, formation_slots, min_count=0)

        # Neutral values are used only inside reductions. The accompanying
        # valid-count gate preserves the distinction between a true zero return
        # (valid denominator observation) and a missing/suspended slot.
        is_total_loss = valid_form & (r_form == -1.0)
        log_terms = np.zeros(r_form.shape, dtype=np.float64)
        positive_gross = valid_form & (r_form > -1.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_terms[positive_gross] = np.log1p(r_form[positive_gross])
        sum_log = ctx.roll_sum(log_terms, formation_slots, min_count=0)
        loss_count = ctx.roll_sum(is_total_loss.astype(np.float64),
                                  formation_slots, min_count=0)
        with np.errstate(all="ignore"):
            pret = np.expm1(sum_log)
        pret = np.where(loss_count > 0.0, -1.0, pret)
        pret = np.where(np.isfinite(pret), pret, np.nan)

        positive = (valid_form & (r_form > 0.0)).astype(np.float64)
        negative = (valid_form & (r_form < 0.0)).astype(np.float64)
        n_pos = ctx.roll_sum(positive, formation_slots, min_count=0)
        n_neg = ctx.roll_sum(negative, formation_slots, min_count=0)
        denominator_ok = np.isfinite(n_form) & (n_form > 0.0)
        p_pos = np.divide(n_pos, n_form, out=np.full_like(n_form, np.nan),
                          where=denominator_ok)
        p_neg = np.divide(n_neg, n_form, out=np.full_like(n_form, np.nan),
                          where=denominator_ok)
        id_value = np.sign(pret) * (p_neg - p_pos)
        raw = -pret * id_value

        n_skip = ctx.roll_count(r, skip_slots, min_count=0)
        ok = univ & traded
        ok &= n_form >= float(PARAMETERS["min_formation_valid"])
        ok &= n_skip >= float(PARAMETERS["min_skip_valid"])
        # Minimum valid-count gates must not let early panel rows stand in for
        # calendar slots that do not exist yet. The first complete signal path
        # is the 273rd row (zero-based index 272): 252 formation + 21 skip.
        calendar_ready = (np.arange(r.shape[0])[:, None] >=
                          formation_slots + formation_lag - 1)
        ok &= calendar_ready
        ok &= np.isfinite(pret) & np.isfinite(id_value) & np.isfinite(raw)
        result = np.where(ok, raw, np.nan)
        result = np.asarray(result, dtype=np.float64)
        result[~np.isfinite(result)] = np.nan
        return np.ascontiguousarray(result)

    return compute
