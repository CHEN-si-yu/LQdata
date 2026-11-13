"""Same-day cost-distribution geometry descriptors.

Input contract: p is ascending, unique, finite positive float64 prices; w is
same-order finite nonnegative float64 mass normalized to one (zero mass bins
are allowed); close is the same-day raw close (unused; retained in the shared
family interface). No pandas, labels, state, or cross-day data are accessed.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit
try:
    from numba.extending import register_jitable
except ImportError:  # Keep the standalone reference usable without Numba.
    def register_jitable(fn):
        return fn

NAMES = (
    "cg_log_iqr_v1",
    "cg_log_p90_p10_v1",
    "cg_log_mad_v1",
    "cg_log_std_v1",
    "cg_price_gini_v1",
    "cg_interval_mass_iqr_v1",
    "cg_shortest50_width_v1",
    "cg_shortest50_offset_v1",
    "cg_lmoment_scale_v1",
    "cg_lmoment_skew_v1",
    "cg_lmoment_kurt_v1",
    "cg_trimmed_scale_10_90_v1",
    "cg_trimmed_skew_10_90_v1",
    "cg_trimmed_kurt_10_90_v1",
    "cg_log_skewness_v1",
    "cg_log_excess_kurtosis_v1",
    "cg_tail_mass_balance_10_v1",
    "cg_upper_tail_mean_excess_90_v1",
    "cg_lower_tail_mean_excess_10_v1",
    "cg_tail_distance_balance_v1",
    "cg_reflection_wasserstein_v1",
    "cg_max_quantile_reflection_mismatch_v1",
    "cg_peak_valley_barrier_v1",
    "cg_peak_separation_iqr_v1",
    "cg_secondary_basin_mass_v1",
    "cg_basin_mass_imbalance_v1",
    "cg_median_valley_depth_v1",
    "cg_modal_basin_width_v1",
    "cg_log_gap_cv_v1",
    "cg_mass_gap_correlation_v1",
    "cg_effective_support_span_v1",
    "cg_outer_fence_mass_15iqr_v1",
    "cg_max_occupied_gap_iqr_v1",
    "cg_support_midpoint_lower_mass_v1",
    "cg_median_support_position_v1",
    "cg_log_mean_median_gap_iqr_v1",
    "cg_mean_support_position_v1",
    "cg_semivariance_balance_v1",
    "cg_equal_width4_entropy_v1",
    "cg_outer_fence_mass_balance_v1",
)


@register_jitable
def _quantile(x, w, q):
    total = 0.0
    for z in range(w.size):
        total += w[z]
    if total <= 0.0:
        return math.nan
    target = q * total
    c = 0.0
    for z in range(w.size):
        c += w[z]
        if c >= target - 1e-14 * total:
            return x[z]
    return x[w.size - 1]


@register_jitable
def _moment(x, w, center, order):
    s = 0.0
    for i in range(x.size):
        d = x[i] - center
        if order == 2:
            s += w[i] * d * d
        elif order == 3:
            s += w[i] * d * d * d
        else:
            s += w[i] * d * d * d * d
    return s


@register_jitable
def _trimmed_moments(x, w, lo, hi):
    mass = 0.0
    mean = 0.0
    for i in range(x.size):
        if x[i] >= lo and x[i] <= hi:
            mass += w[i]
            mean += w[i] * x[i]
    if mass <= 0.0:
        return math.nan, math.nan, math.nan
    mean /= mass
    v2 = 0.0
    v3 = 0.0
    v4 = 0.0
    for i in range(x.size):
        if x[i] >= lo and x[i] <= hi:
            d = x[i] - mean
            v2 += w[i] * d * d / mass
            v3 += w[i] * d * d * d / mass
            v4 += w[i] * d * d * d * d / mass
    if v2 <= 0.0:
        return math.nan, math.nan, math.nan
    sd = math.sqrt(v2)
    return sd, v3 / (sd * sd * sd), v4 / (v2 * v2) - 3.0


@register_jitable
def _l_moments(x, w, trim_lo=0.0, trim_hi=1.0):
    """L-moments from the weighted empirical quantile function.

    Intersect each atom's CDF interval with [trim_lo, trim_hi] and integrate
    the shifted Legendre polynomials exactly over that interval.
    """
    mass = trim_hi - trim_lo
    if mass <= 0.0:
        return math.nan, math.nan, math.nan
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    b3 = 0.0
    cdf = 0.0
    for i in range(x.size):
        nxt = cdf + w[i]
        a = max(cdf, trim_lo)
        b = min(nxt, trim_hi)
        if b > a:
            # Integral of u^r on the quantile interval, normalized by mass.
            b0 += x[i] * (b - a) / mass
            b1 += x[i] * (b * b - a * a) / (2.0 * mass)
            b2 += x[i] * (b * b * b - a * a * a) / (3.0 * mass)
            b3 += x[i] * (b ** 4 - a ** 4) / (4.0 * mass)
        cdf = nxt
    l2 = 2.0 * b1 - b0
    if abs(l2) <= 1e-15:
        return l2, math.nan, math.nan
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    l4 = 20.0 * b3 - 30.0 * b2 + 12.0 * b1 - b0
    return l2, l3 / l2, l4 / l2


@register_jitable
def _shortest_interval(x, w, target):
    n = x.size
    best_width = math.inf
    best_lo = math.nan
    i = 0
    running = 0.0
    for j in range(n):
        running += w[j]
        while i <= j and running - w[i] >= target - 1e-14:
            running -= w[i]
            i += 1
        if running >= target - 1e-14 and x[j] - x[i] < best_width:
            best_width = x[j] - x[i]
            best_lo = x[i]
    return best_width, best_lo


@register_jitable
def _peaks(w):
    # Collapse equal-height plateaus to one midpoint peak, Numba-compatible.
    ids = np.empty(w.size, dtype=np.int64)
    count = 0
    i = 0
    while i < w.size:
        j = i
        while j + 1 < w.size and w[j + 1] == w[i]:
            j += 1
        left = w[i - 1] if i > 0 else -1.0
        right = w[j + 1] if j + 1 < w.size else -1.0
        if w[i] > 0.0 and w[i] > left and w[i] > right:
            ids[count] = (i + j) // 2
            count += 1
        i = j + 1
    return ids, count


@register_jitable
def _w1_reflection(x, w, median, scale):
    # W1(P, reflected P) = integral |F_P-F_reflected| dx over merged supports.
    if scale <= 0.0:
        return math.nan
    z = np.empty(2 * x.size, dtype=np.float64)
    z[:x.size] = x
    for i in range(x.size):
        z[x.size + i] = 2.0 * median - x[x.size - 1 - i]
    z.sort()
    acc = 0.0
    fp = 0.0
    fr = 0.0
    ip = 0
    ir = x.size - 1
    prev = z[0]
    for k in range(z.size):
        point = z[k]
        acc += abs(fp - fr) * (point - prev)
        while ip < x.size and x[ip] <= point:
            fp += w[ip]
            ip += 1
        while ir >= 0 and 2.0 * median - x[ir] <= point:
            fr += w[ir]
            ir -= 1
        prev = point
    return acc / scale


@njit(cache=True)
def summarize(p, w, close):
    """Return all NAMES as float64; invalid/degenerate distributions yield NaNs."""
    out = np.full(len(NAMES), np.nan, dtype=np.float64)
    if p is None or w is None:
        return out
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if p.ndim != 1 or w.ndim != 1 or p.size != w.size or p.size < 2:
        return out
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(w)):
        return out
    if np.any(p <= 0.0) or np.any(w < 0.0):
        return out
    total = float(np.sum(w))
    if total <= 0.0:
        return out
    if np.any(p[1:] <= p[:-1]):
        return out
    w = w / total
    xall = np.log(p)
    wall = w.copy()
    keep = w > 0.0
    x = xall[keep]
    ww = w[keep]
    if x.size < 2:
        return out
    ww = ww / float(np.sum(ww))
    q05 = _quantile(x, ww, 0.05)
    q10 = _quantile(x, ww, 0.10)
    q25 = _quantile(x, ww, 0.25)
    q50 = _quantile(x, ww, 0.50)
    q75 = _quantile(x, ww, 0.75)
    q90 = _quantile(x, ww, 0.90)
    q95 = _quantile(x, ww, 0.95)
    iqr = q75 - q25
    central = _quantile(x, ww, 0.90) - _quantile(x, ww, 0.10)
    total_span = x[-1] - x[0]
    if iqr > 0.0:
        out[0] = iqr
    if central > 0.0:
        out[1] = central
    mad = 0.0
    mean = 0.0
    for i in range(x.size):
        mean += ww[i] * x[i]
        mad += ww[i] * abs(x[i] - q50)
    out[2] = mad
    v2 = _moment(x, ww, mean, 2)
    if v2 > 0.0:
        sd = math.sqrt(v2)
        out[3] = sd
        out[14] = _moment(x, ww, mean, 3) / (sd ** 3)
        out[15] = _moment(x, ww, mean, 4) / (v2 * v2) - 3.0
    # Price-axis Gini, distinct from rank-by-level weight Gini.
    meanp = 0.0
    cdf = 0.0
    g = 0.0
    for i in range(x.size):
        pi = math.exp(x[i])
        meanp += ww[i] * pi
        cdf += ww[i]
        g += ww[i] * pi * (2.0 * cdf - ww[i] - 1.0)
    if meanp > 0.0:
        out[4] = g / meanp
    # Maximal interval mass for width equal to the central quartile spread.
    if iqr >= 0.0:
        best = 0.0
        j = 0
        running = 0.0
        for i in range(x.size):
            while j < x.size and x[j] - x[i] <= iqr:
                running += ww[j]
                j += 1
            if running > best:
                best = running
            if j > i:
                running -= ww[i]
        out[5] = best
    sw, sl = _shortest_interval(x, ww, 0.50)
    if math.isfinite(sw):
        out[6] = sw
        out[7] = ((sl + sw * 0.5) - q50) / iqr if iqr > 0.0 else 0.0
    lm2, lm3, lm4 = _l_moments(x, ww)
    out[8], out[9], out[10] = lm2, lm3, lm4
    ts, tsk, tku = _trimmed_moments(x, ww, q10, q90)
    out[11], out[12], out[13] = ts, tsk, tku
    # Tail mass balance compares probability outside symmetric central deciles.
    lowmass = 0.0
    highmass = 0.0
    lowex = 0.0
    highex = 0.0
    lowrad = 0.0
    hirad = 0.0
    for i in range(x.size):
        if x[i] <= q10:
            lowmass += ww[i]
            lowex += ww[i] * (q10 - x[i])
            lowrad += ww[i] * abs(x[i] - q50)
        if x[i] >= q90:
            highmass += ww[i]
            highex += ww[i] * (x[i] - q90)
            hirad += ww[i] * abs(x[i] - q50)
    if lowmass + highmass > 0.0:
        out[16] = (highmass - lowmass) / (highmass + lowmass)
    if highmass > 0.0:
        out[17] = highex / highmass
    if lowmass > 0.0:
        out[18] = lowex / lowmass
    if lowrad + hirad > 0.0:
        out[19] = (hirad - lowrad) / (hirad + lowrad)
    out[20] = _w1_reflection(x, ww, q50, iqr)
    if iqr > 0.0:
        biggest = 0.0
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            mismatch = abs(_quantile(x, ww, q) + _quantile(x, ww, 1.0 - q) - 2.0 * q50)
            biggest = max(biggest, mismatch)
        out[21] = biggest / iqr
    # Discrete peak topology: basin mass assigned to the nearest local maximum.
    peaks, n_peaks = _peaks(wall)
    if n_peaks >= 2:
        a = 0
        b = 1
        if wall[peaks[b]] > wall[peaks[a]]:
            a, b = b, a
        for k in range(2, n_peaks):
            if wall[peaks[k]] > wall[peaks[a]]:
                b = a
                a = k
            elif wall[peaks[k]] > wall[peaks[b]]:
                b = k
        a = peaks[a]
        b = peaks[b]
        left, right = (a, b) if a < b else (b, a)
        valley = wall[left]
        for i in range(left + 1, right):
            if wall[i] < valley:
                valley = wall[i]
        highpeak = max(wall[a], wall[b])
        if highpeak > 0.0:
            out[22] = (highpeak - valley) / highpeak
        if iqr > 0.0:
            out[23] = abs(xall[a] - xall[b]) / iqr
        basin = np.zeros(n_peaks, dtype=np.float64)
        for i in range(xall.size):
            nearest = 0
            bestd = abs(xall[i] - xall[peaks[0]])
            for k in range(1, n_peaks):
                d = abs(xall[i] - xall[peaks[k]])
                if d < bestd:
                    bestd = d
                    nearest = k
            basin[nearest] += wall[i]
        basin.sort()
        out[24] = basin[-2]
        out[25] = basin[-1] - basin[-2]
        # Median valley deficit: how far the median-bin mass sits below the
        # tallest of the two dominant peaks. A median on a peak gives zero.
        midbin = 0
        for k in range(1, x.size):
            if abs(x[k] - q50) < abs(x[midbin] - q50):
                midbin = k
        peak_height = max(wall[a], wall[b])
        out[26] = max(0.0, peak_height - ww[midbin]) / peak_height
        # Width of the dominant peak's nearest-peak Voronoi basin.
        dominant = peaks[0]
        for k in range(1, n_peaks):
            if wall[peaks[k]] > wall[dominant]:
                dominant = peaks[k]
        pkpos = 0
        while pkpos < n_peaks and peaks[pkpos] != dominant:
            pkpos += 1
        basin_left = xall[0] if pkpos == 0 else 0.5 * (xall[peaks[pkpos - 1]] + xall[dominant])
        basin_right = xall[-1] if pkpos + 1 == n_peaks else 0.5 * (xall[dominant] + xall[peaks[pkpos + 1]])
        out[27] = (basin_right - basin_left) / total_span if total_span > 0.0 else 0.0
    # Irregular spacing of positive-mass cost levels and its coupling to mass.
    # These describe geometry of occupied cost support, not zero-weight rows.
    if x.size >= 3:
        gaps = np.empty(x.size - 1, dtype=np.float64)
        gmean = 0.0
        for i in range(x.size - 1):
            gaps[i] = x[i + 1] - x[i]
            gmean += gaps[i]
        gmean /= gaps.size
        if gmean > 0.0:
            gvar = 0.0
            for i in range(gaps.size):
                d = gaps[i] - gmean
                gvar += d * d
            out[28] = math.sqrt(gvar / gaps.size) / gmean
        localgap = np.empty(x.size, dtype=np.float64)
        localgap[0] = gaps[0]
        localgap[x.size - 1] = gaps[gaps.size - 1]
        for i in range(1, x.size - 1):
            localgap[i] = 0.5 * (gaps[i - 1] + gaps[i])
        meanw = 1.0 / x.size
        meang = 0.0
        for i in range(x.size):
            meang += localgap[i] / x.size
        cov = 0.0
        vw = 0.0
        vg = 0.0
        for i in range(x.size):
            dw = ww[i] - meanw
            dg = localgap[i] - meang
            cov += dw * dg
            vw += dw * dw
            vg += dg * dg
        if vw > 0.0 and vg > 0.0:
            out[29] = cov / math.sqrt(vw * vg)
    if total_span > 0.0:
        out[30] = central / total_span
    fence_lo = q25 - 1.5 * iqr
    fence_hi = q75 + 1.5 * iqr
    outer_mass = 0.0
    for i in range(x.size):
        if x[i] < fence_lo or x[i] > fence_hi:
            outer_mass += ww[i]
    out[31] = outer_mass
    # Largest gap between occupied cost levels, scaled by central spread.
    maxgap = 0.0
    for i in range(1, x.size):
        maxgap = max(maxgap, x[i] - x[i - 1])
    if iqr > 0.0:
        out[32] = maxgap / iqr
    # Probability mass below the geometric midpoint of observed support.
    support_mid = 0.5 * (x[0] + x[-1])
    out[33] = 0.0
    for i in range(x.size):
        if x[i] <= support_mid:
            out[33] += ww[i]
    if total_span > 0.0:
        out[34] = (q50 - x[0]) / total_span
        out[35] = (mean - q50) / iqr if iqr > 0.0 else 0.0
        out[36] = (mean - 0.5 * (x[0] + x[-1])) / total_span
    lowersemi = 0.0
    uppersemi = 0.0
    for i in range(x.size):
        d = x[i] - q50
        if d < 0.0:
            lowersemi += ww[i] * d * d
        elif d > 0.0:
            uppersemi += ww[i] * d * d
    if lowersemi + uppersemi > 0.0:
        out[37] = (uppersemi - lowersemi) / (uppersemi + lowersemi)
    # Coarse log-price mass entropy on four equal-width central-support bands.
    if q95 > q05:
        bands = np.zeros(4, dtype=np.float64)
        width = (q95 - q05) / 4.0
        for i in range(x.size):
            if x[i] >= q05 and x[i] <= q95:
                bidx = int((x[i] - q05) / width)
                if bidx > 3:
                    bidx = 3
                bands[bidx] += ww[i]
        retained = 0.0
        for k in range(4):
            retained += bands[k]
        if retained > 0.0:
            ent = 0.0
            for k in range(4):
                if bands[k] > 0.0:
                    z = bands[k] / retained
                    ent -= z * math.log(z)
            out[38] = ent / math.log(4.0)
    upperouter = 0.0
    lowerouter = 0.0
    for i in range(x.size):
        if x[i] < fence_lo:
            lowerouter += ww[i]
        elif x[i] > fence_hi:
            upperouter += ww[i]
    if upperouter + lowerouter > 0.0:
        out[39] = (upperouter - lowerouter) / (upperouter + lowerouter)
    return out
