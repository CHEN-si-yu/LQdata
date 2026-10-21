"""Lead-reviewed R13 minute successor. Positive direction and 5/20-session
smoothers are unchanged. Independent source/engineering evidence precedes outcomes.
The bar-price proxy does not identify actual informed traders or trade aggressors.
Admission uses frozen v3 alongside preserved v2 diagnostics; no automatic alpha claim.
SMF selects descending abs(close/open-1)/sqrt(vol_raw), using cumulative
volume BEFORE each bar <20% of daily usable volume. Selected close VWAP
and full amount VWAP both use a stock-day share/hand multiplier, accepted
only for amount/(vol*close) in [0.5,2] or [50,200], otherwise daily NaN.
At least 40 usable bars; rolling min counts 4/15. Missing daily values
are NaN before smoothing. Current universe mask applies after smoothing;
a suspended session can retain a finite smoothed prior-day feature.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

log = logging.getLogger("research.families.r13c_smf_v2")

SOURCE = "stock_history_5min"
MIN_BARS = 40            # 当日可用棒下限（与 R3-B / r4b 同口径）
FRAC = 0.20              # 高信息量成交子集的量份额（原文 20%）
AUDIT_SEGMENTS = 2000
# 事前冻结：(kind, window) -> 候选名；窗口只登记 5 / 20 两个期限（期限结构式，不扫网格）
FROZEN = {
    ("smf", 5): "r13c_smf_5_v2",
    ("smf", 20): "r13c_smf_20_v2",
}
MIN_VALID = {5: 4, 20: 15}


def _frozen_key(parameters):
    if not isinstance(parameters, dict):
        return None
    if parameters.get("kind") != "smf":
        return None
    return ("smf", parameters.get("window"))


def build(parameters):
    """只允许预登记的两个期限变体；不搜索窗口、不搜索方向。"""
    key = _frozen_key(parameters)
    if key not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))
    window = key[1]
    min_valid = MIN_VALID[window]

    def compute(ctx):
        daily = _smf_daily(ctx)
        value = ctx.roll_mean(daily, window, min_valid)
        value = np.where(np.isfinite(value), value, np.nan)
        grid = np.where(np.asarray(ctx.universe, dtype=bool), value, np.nan)
        log.info("r13c_smf_v2 window=%d panel_finite=%d", window, int(np.isfinite(grid).sum()))
        return np.ascontiguousarray(grid, dtype=np.float64)

    return compute


# ---------------------------------------------------------------- 分段工具
# 与 r12c_net_turnover_v1 / r4b_intraday_shape_v1 同一套（上游 = r3_intraday_moments_v2）。
def _arrow(series):
    arr = getattr(series, "array", None)
    pa_arr = getattr(arr, "_pa_array", None)
    if pa_arr is None:
        pa_arr = pa.array(series)
    return pa_arr


def _segments(frame):
    """切 (股票, 日) 段；返回 (starts, lengths, lunch_mask)。"""
    tt = _arrow(frame["trade_time"])
    n = len(tt)
    if n < 2:
        raise ValueError("stock_history_5min 年分区行数 < 2")
    t16 = pc.utf8_slice_codeunits(tt, 11, 16)
    back = pc.fill_null(pc.less(t16.slice(1, n - 1), t16.slice(0, n - 1)), False)
    back = np.asarray(back.to_numpy(zero_copy_only=False), dtype=bool)
    d10 = pc.utf8_slice_codeunits(tt, 0, 10)
    daychg = pc.not_equal(d10.slice(1, n - 1), d10.slice(0, n - 1))
    daychg = np.asarray(pc.fill_null(daychg, False).to_numpy(zero_copy_only=False), dtype=bool)
    boundary = back | daychg

    lunch = np.zeros(n, dtype=bool)
    at1130 = pc.equal(t16.slice(0, n - 1), pa.scalar("11:30"))
    at1130 = np.asarray(pc.fill_null(at1130, False).to_numpy(zero_copy_only=False), dtype=bool)
    cand = np.flatnonzero(at1130 & (~boundary))
    if cand.size:
        nxt = pc.take(t16, pa.array((cand + 1).astype(np.int64)))
        is_pm = np.asarray(pc.or_(pc.equal(nxt, pa.scalar("13:00")),
                                  pc.equal(nxt, pa.scalar("13:05"))).to_numpy(zero_copy_only=False),
                           dtype=bool)
        lunch[cand[is_pm] + 1] = True

    starts = np.flatnonzero(np.concatenate((np.array([True]), boundary)))
    lengths = np.diff(np.concatenate((starts, np.array([n], dtype=starts.dtype))))
    return starts.astype(np.int64), lengths.astype(np.int64), lunch


def _audit_segments(ctx, frame, starts, lengths, sample=AUDIT_SEGMENTS):
    if starts.size == 0:
        return
    sel = np.unique(np.linspace(0, starts.size - 1, min(sample, starts.size)).astype(np.int64))
    picks, gids = [], []
    for g, k in enumerate(sel):
        s, ln = int(starts[k]), int(lengths[k])
        take = np.unique(np.linspace(s, s + ln - 1, min(ln, 8)).astype(np.int64))
        picks.append(take)
        gids.append(np.full(take.size, g, dtype=np.int64))
    rows = np.concatenate(picks)
    gid = np.concatenate(gids)
    codes = frame["stock_code"].iloc[rows].to_numpy()
    days = np.asarray(ctx.date_col(frame["trade_time"].iloc[rows].str.slice(0, 10)), dtype=np.int64)
    chk = pd.DataFrame({"g": gid, "c": codes, "d": days}).groupby("g", sort=False).nunique()
    bad = chk[(chk["c"] != 1) | (chk["d"] != 1)]
    if len(bad):
        raise ValueError("分段完整性复核失败：%d 个抽样段内出现多个股票代码或日期，"
                         "说明段边界被漏切（拒绝继续）" % len(bad))


def _year_facts(ctx, frame, starts, lengths):
    _audit_segments(ctx, frame, starts, lengths)
    code_seg = frame["stock_code"].iloc[starts].to_numpy()
    day_seg = np.asarray(ctx.date_col(frame["trade_time"].iloc[starts].str.slice(0, 10)),
                         dtype=np.int64)
    if not np.all(day_seg > 0):
        raise ValueError("trade_time 不能全部解析为合法日期")
    pos_seg = np.asarray(ctx.code_index(code_seg), dtype=np.int64)
    keep = np.flatnonzero((pos_seg >= 0) & (lengths >= MIN_BARS))
    if keep.size == 0:
        return None
    key = pos_seg[keep] * np.int64(100_000_000) + day_seg[keep]
    if np.unique(key).size != key.size:
        raise ValueError("(股票, 日) 段不唯一：年分区存在重复股票日，拒绝继续")
    return pos_seg, day_seg, keep


def _gather(frame, starts, lengths, keep, cols):
    lens = lengths[keep]
    total = int(lens.sum())
    cum = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lens)))
    rows = np.repeat(starts[keep], lens) + (np.arange(total, dtype=np.int64)
                                            - np.repeat(cum[:-1], lens))
    out = {}
    for c in cols:
        allv = frame[c].to_numpy(np.float64)
        out[c] = allv[rows]
        del allv
    return out, rows, lens


# ---------------------------------------------------------------- C2 日频值
def _smf_daily(ctx):
    """逐 (股票, 日) 的当日 SMF（未滚动）铺进面板；池外与不足 MIN_BARS 记 NaN。"""
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    y0, y1 = int(days[0]) // 10000, int(days[-1]) // 10000
    cols = ["stock_code", "trade_time", "open", "close", "vol", "amount"]
    daily = None
    for year in range(y0, y1 + 1):
        frame = ctx.dataset(SOURCE, columns=cols, years=(year, year))
        if frame is None or frame.empty:
            continue
        starts, lengths, _lunch = _segments(frame)
        facts = _year_facts(ctx, frame, starts, lengths)
        if facts is None:
            continue
        pos_seg, day_seg, keep = facts
        data, rows, lens = _gather(frame, starts, lengths, keep,
                                   ["open", "close", "vol", "amount"])
        seg_id = np.repeat(np.arange(lens.size, dtype=np.int64), lens)
        seg_starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lens)[:-1]))
        value = _daily_value(data, seg_starts, lens, seg_id)
        g = ctx.panel.place(pos_seg[keep], day_seg[keep], value)
        daily = g if daily is None else np.where(np.isfinite(g), g, daily)
        log.info("r13c_smf_v2 year=%d segs=%d kept=%d bars=%d finite_day=%d",
                 year, int(starts.size), int(keep.size), int(lens.sum()),
                 int(np.isfinite(value).sum()))
        del frame, data, rows, seg_id
    if daily is None:
        return ctx.panel.empty(dtype=np.float64)
    return daily


def _daily_value(data, seg_starts, lens, seg_id, return_diagnostics=False):
    """Stock-day SMF with a fail-closed per-day volume-unit guard.

    `unit_ratio = sum(amount) / sum(vol * close)` is near 1 for share-count bars
    and near 100 for lot-count bars. A single multiplier per stock-day normalizes
    both selected and full-day volume, preserving the frozen close-weighted
    selected VWAP and the 20% selection rule. Ambiguous ratios yield NaN.
    """
    o, c, v, a = data["open"], data["close"], data["vol"], data["amount"]
    usable = (np.isfinite(o) & (o > 0.0) & np.isfinite(c) & (c > 0.0)
              & np.isfinite(v) & (v > 0.0) & np.isfinite(a) & (a >= 0.0))

    # Unit classification uses the same usable bars and price basis for numerator
    # and denominator. A daily scalar is held constant through each segment.
    raw_amount = np.add.reduceat(np.where(usable, a, 0.0), seg_starts)
    raw_price_volume = np.add.reduceat(np.where(usable, v * c, 0.0), seg_starts)
    with np.errstate(all="ignore"):
        unit_ratio = raw_amount / np.where(raw_price_volume > 0.0, raw_price_volume, np.nan)
    share_unit = np.isfinite(unit_ratio) & (unit_ratio >= 0.5) & (unit_ratio <= 2.0)
    lot_unit = np.isfinite(unit_ratio) & (unit_ratio >= 50.0) & (unit_ratio <= 200.0)
    unit_ok = share_unit | lot_unit
    unit_multiplier = np.where(lot_unit, 100.0, 1.0)
    mult_bar = np.repeat(unit_multiplier, lens)
    v_normalized = v * mult_bar

    with np.errstate(all="ignore"):
        zdf = np.abs(c / np.where(usable, o, np.nan) - 1.0)
        smart = zdf / np.sqrt(np.where(usable, v, np.nan))
    neg = np.where(usable, -smart, np.inf)          # 不可用棒排到段尾
    order = np.lexsort((neg, seg_id.astype(np.int32)))

    usable_s = usable[order]
    v_s = np.where(usable_s, v_normalized[order], 0.0)
    vc_s = np.where(usable_s, v_s * c[order], 0.0)
    a_s = np.where(usable_s, a[order], 0.0)

    cs = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(v_s)))
    base = cs[seg_starts]
    cum_prev = cs[:-1] - np.repeat(base, lens)          # 段内、当前棒之前的累计量
    total = cs[seg_starts + lens] - base
    keep = cum_prev < FRAC * np.repeat(total, lens)

    num = np.add.reduceat(np.where(keep, vc_s, 0.0), seg_starts)
    den = np.add.reduceat(np.where(keep, v_s, 0.0), seg_starts)
    day_v = np.add.reduceat(v_s, seg_starts)
    day_a = np.add.reduceat(a_s, seg_starts)
    n_ok = np.add.reduceat(usable_s.astype(np.float64), seg_starts)
    selected_bars = np.add.reduceat((keep & usable_s).astype(np.float64), seg_starts)
    selected_volume = np.add.reduceat(np.where(keep, v_s, 0.0), seg_starts)

    vwap_sel = np.where(den > 0.0, num / np.where(den > 0.0, den, np.nan), np.nan)
    vwap_day = np.where(day_v > 0.0, day_a / np.where(day_v > 0.0, day_v, np.nan), np.nan)
    with np.errstate(all="ignore"):
        smf = vwap_sel / vwap_day - 1.0
        selected_fraction = selected_volume / np.where(day_v > 0.0, day_v, np.nan)
    good = unit_ok & (n_ok >= MIN_BARS) & np.isfinite(smf) & (vwap_day > 0.0)
    value = np.where(good, smf, np.nan)
    if not return_diagnostics:
        return value
    return value, {
        "unit_ratio": unit_ratio,
        "unit_multiplier": unit_multiplier,
        "unit_ok": unit_ok,
        "usable_bars": n_ok,
        "selected_bars": selected_bars,
        "selected_volume_fraction": selected_fraction,
        "vwap_selected": vwap_sel,
        "vwap_day": vwap_day,
    }
