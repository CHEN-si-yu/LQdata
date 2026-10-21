# ══════════════════════════════════════════════════════════════════════
# R3-E · Round 2 反向稳定信号的独立确认
#
# ⚠️ 这是**新的、单独预登记的假设**，不是把 R2 候选反号。
#    R2 的两个信号 declared 分年 9/9 与**预登记方向相反**（见 §背景），
#    按 R3 任务书 0.3 的规则属于"样本内发现，不作准入证据"。
#    R3-E 的任务是用独立设计去**证伪或确认**这个反向关系。
#    本模块把反向关系**直接构造成因子值本身**（值 = 反向信号量），
#    并把六个候选全部登记为 higher_is_better = true，
#    于是 evaluation.json 的 declared RankIC 就等于"反向假设"的 RankIC，
#    raw 与 declared 恒等，不存在方向翻转的歧义。
#    本模块**没有**改任何已登记候选的 higher_is_better，也没有删除任何 R2 产物。
#
# 背景数字（来自 R2 的 evaluation.json，本模块不重新计算）：
#   · r2own_v2_announcement_drift5（higher_is_better=true，故 declared=raw）
#     declared h1 = −0.02127（95% CI [−0.02790,−0.01465]），分年 0 正 / 9 负，
#     h5/h10/h20 分别为 −0.0212 / −0.0214 / −0.0253，全部 9/9 为负。
#     → 预登记方向（公告后相对漂移越大、后续收益越高）**被证伪**；
#       稳定成立的是它的反向：**公告后相对漂移越大，后续收益越低**。
#   · r2ca_corpaction_frequency_w500（higher_is_better=false，故 declared=−raw）
#     declared h1 = −0.00691，分年 0 正 / 9 负 → raw = +0.0069，9/9 为正。
#     → 预登记方向（除权除息越频繁、后续收益越低）**被证伪**；
#       稳定成立的是它的反向：**过去 N 日除权除息次数越多，后续收益越高**。
#
# ═══════════════ 机制陈述（在看任何本轮 IC 之前写死，不依赖数据观测） ═══════════════
#
# D1 · 公告后过度反应反转（announcement-window overreaction reversal）
#   股东户数公告是一个**离散的注意力事件**。公告披露的是滞后 1–3 个月的持仓结构，
#   因此公告后窗口内的相对价格变动主要由**注意力驱动的过度反应**构成，而不是新的基本面；
#   过度反应在随后数个交易日回吐。该机制与"无条件短期反转"不同：
#   它只在**公告生效日起的 N 个交易日内**这一个子集上定义（R2 实测该子集约占全池 13%），
#   以公告日为锚，而不是每天对所有人的过去收益取反。
#   反向假设（本模块登记的方向）：**公告窗口内相对涨幅越大 → 随后收益越低**。
#
# D2 · 股东回报活跃度溢价（shareholder-return activity）
#   除权除息事件（现金分红 + 送转）的**频次**刻画公司在股东回报上的**持续活跃度**与
#   盈利/现金流的兑现能力。持续分红的公司信息披露与治理通常更规范、代理成本更低，
#   在有限套利环境下享有"股利溢价"；而一次性高股息率并不代表持续回报能力。
#   该机制测的是**事件频率**（公司行为记录），**不是**股息率水平，也**不含**价格信息，
#   因此与 dividend_yield / bp / 动量在概念上不同。
#   反向假设（本模块登记的方向）：**事件频率越高 → 随后收益越高**。
#
# 两项机制都必须靠 (b) 时间分块、(c) h=10/20 外推、(d) 三套参数化、(e) 正交残差来确认，
# 任一不过即判"不可确认"。
# ══════════════════════════════════════════════════════════════════════
"""R3-E 独立确认：6 个事前登记参数化（2 个方向 × 3 套参数化）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fea import mathx as mx

SIGNALS = (
    "rev_drift_w3", "rev_drift_w20", "rev_drift_decay10",
    "ca_freq_w120", "ca_rate_w250", "ca_ewma60",
)
REV_DAYS = {"rev_drift_w3": 3, "rev_drift_w20": 20, "rev_drift_decay10": 10}
DECAY_DAYS = 10            # 前沿加权窗口（P3）
CA_WINDOW = {"ca_freq_w120": 120, "ca_rate_w250": 250}
CA_HISTORY_GATE = 250      # EWMA 的"历史足够"闸门窗口（交易日）
EWMA_HALFLIFE = 60.0       # P3 的半衰期（交易日）
NOISE_EPS = 1e-3           # 复权因子跳变的噪声底（R2 用 1e-4，本模块更高、更保守）
MIN_HISTORY_FRAC = 0.8     # 窗口内原始 adj_factor 有效天数占比下限
HOLDER_WARMUP = 500        # rev_* 的面板预热（日历天）
CA_WARMUP = 700            # ca_* 的面板预热（日历天）


def _divide(a, b, floor=1e-12):
    with np.errstate(all="ignore"):
        out = np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64)
    return np.where(np.isfinite(out) & (np.abs(b) > floor), out, np.nan)


def _pool_relative(ctx, values):
    """用当前固定池的当日有效样本等权去均值（与 R2 一致，不是未来标签资格）。"""
    valid = np.asarray(ctx.universe, dtype=bool) & np.isfinite(values)
    count = valid.sum(axis=1)
    minimum = max(3, int(ctx.cfg.min_cross_section))
    mean = _divide(np.where(valid, values, 0.0).sum(axis=1), count, floor=0)
    mean = np.where(count >= minimum, mean, np.nan)
    return np.where(valid, values - mean[:, None], np.nan)


# ────────────────────────────── D1：公告窗口反转 ──────────────────────────────
def _holder_events(ctx):
    """每个不同的 (stock_code, ann_date) 是一个公告事件；生效日 = 首个 >= ann_date 的交易日。"""
    frame = ctx.dataset("stock_holder_number",
                        columns=["stock_code", "ann_date", "end_date", "holder_num"],
                        years=None)
    if frame is None or frame.empty:
        return np.array([], dtype=object), np.zeros(0, dtype=np.int64)
    code = frame["stock_code"].astype(str).to_numpy()
    ann = np.asarray(ctx.date_col(frame["ann_date"]), dtype=np.int64)
    num = pd.to_numeric(frame["holder_num"], errors="coerce").to_numpy(dtype=np.float64)
    keep = (ann >= 19900101) & np.isfinite(num) & (num >= 0)
    if not keep.any():
        return np.array([], dtype=object), np.zeros(0, dtype=np.int64)
    code, ann = code[keep], ann[keep]
    key = np.array([f"{c}|{d}" for c, d in zip(code, ann)], dtype=object)
    _, first = np.unique(key, return_index=True)
    first = np.sort(first)
    return code[first], ann[first]


def _since_last_event(ctx, eff_codes, eff_days):
    """(T,C)：距最近一次公告生效日的交易日数（当天公告 = 0）。"""
    panel = ctx.panel
    grid = ctx.event_grid(eff_codes, eff_days)
    index = np.arange(panel.T, dtype=np.int64)[:, None]
    last = np.maximum.accumulate(np.where(grid > 0, index, -1), axis=0)
    return np.where(last >= 0, index - last, -1)


def _rev_flat(ctx):
    """D1-P1/P2 的公共部分：池内相对漂移（相对公告生效日收盘）与 since。"""
    panel = ctx.panel
    codes, ann = _holder_events(ctx)
    if ann.size == 0:
        return None, None
    pos = np.searchsorted(panel.dates, ann)
    inside = (pos < panel.T) & (ann >= int(panel.dates[0]))
    pos = np.clip(pos, 0, panel.T - 1)
    eff = panel.dates[pos]
    col = np.asarray(ctx.code_index(codes), dtype=np.int64)
    keep = inside & (col >= 0)
    eff, col, pos = eff[keep], col[keep], pos[keep]
    if eff.size == 0:
        return None, None
    key = col.astype(np.int64) * 100000000 + eff.astype(np.int64)
    _, uniq = np.unique(key, return_index=True)
    uniq = np.sort(uniq)
    eff, col, pos = eff[uniq], col[uniq], pos[uniq]
    close = np.asarray(ctx.hfq("close"), dtype=np.float64)
    anchor = ctx.asof_daily(codes[keep][uniq], eff, close[pos, col])
    with np.errstate(all="ignore"):
        drift = _divide(close, anchor) - 1.0
    since = _since_last_event(ctx, codes[keep][uniq], eff)
    return _pool_relative(ctx, drift), since


def _rev_decay(ctx):
    """D1-P3：公告后**前沿加权**的池内相对日均收益（权重在公告后第 1 天最大）。"""
    panel = ctx.panel
    codes, ann = _holder_events(ctx)
    if ann.size == 0:
        return None
    pos = np.searchsorted(panel.dates, ann)
    inside = (pos < panel.T) & (ann >= int(panel.dates[0]))
    pos = np.clip(pos, 0, panel.T - 1)
    eff = panel.dates[pos]
    col = np.asarray(ctx.code_index(codes), dtype=np.int64)
    keep = inside & (col >= 0)
    eff, col = eff[keep], col[keep]
    if eff.size == 0:
        return None
    key = col.astype(np.int64) * 100000000 + eff.astype(np.int64)
    _, uniq = np.unique(key, return_index=True)
    uniq = np.sort(uniq)
    eff = eff[uniq]
    since = _since_last_event(ctx, codes[keep][uniq], eff)
    traded = np.asarray(ctx.traded(), dtype=bool)
    one = np.asarray(ctx.ret_clean(1), dtype=np.float64)
    one = np.where(traded & np.isfinite(one) & (one > -1.0), one, np.nan)
    with np.errstate(all="ignore"):
        logret = np.log1p(one)
    daily = _pool_relative(ctx, logret)
    acc = np.zeros(panel.shape, dtype=np.float64)
    wsum = np.zeros(panel.shape, dtype=np.float64)
    for a in range(1, DECAY_DAYS + 1):
        w = float(DECAY_DAYS - a + 1)                      # a=1（公告当天）权重最大
        da = mx.shift(daily, a - 1)
        ok = (since >= a) & np.isfinite(da)
        acc = acc + np.where(ok, w * da, 0.0)
        wsum = wsum + np.where(ok, w, 0.0)
    value = -_divide(acc, np.where(wsum > 0, wsum, np.nan), floor=0)
    valid = (since >= 1) & (since <= DECAY_DAYS) & (wsum > 0)
    return np.where(valid, value, np.nan)


# ────────────────────── D2：除权除息事件频次（股东回报活跃度） ──────────────────────
def _ca_state(ctx):
    """复权因子跳变比、事件掩码、有效上市日掩码。事件定义与 R2 同源（数据语义所迫）。"""
    raw = np.asarray(ctx.px("adj_factor"), dtype=np.float64)
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, np.nan)
    af = mx.nan_fill_ffill(raw)
    prev = mx.shift(af, 1)
    with np.errstate(all="ignore"):
        ratio = af / prev
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
    days = ctx.panel.dates
    frame = ctx.dataset("stock_adj_factor_changes", columns=["stock_code", "date"],
                        years=(int(days[0]) // 10000, int(days[-1]) // 10000))
    flag = np.zeros(ctx.panel.shape, dtype=bool)
    if frame is not None and not frame.empty:
        grid = ctx.event_grid(frame["stock_code"].to_numpy(),
                              ctx.date_col(frame["date"]),
                              np.ones(len(frame), dtype=np.float64))
        flag = grid > 0
    jump = ratio - 1.0
    event = flag & np.isfinite(jump) & (jump > NOISE_EPS)
    listed = np.isfinite(af)
    return event, listed


def _ewma(events, halflife):
    """前向递推 EWMA 强度（单位：事件数）。不读未来。"""
    decay = float(np.exp(-np.log(2.0) / halflife))
    T, C = events.shape
    out = np.empty((T, C), dtype=np.float64)
    state = np.zeros(C, dtype=np.float64)
    e = events.astype(np.float64)
    for t in range(T):
        state = e[t] + decay * state
        out[t] = state
    return out


def build(parameters):
    """R3-E 六个事前登记参数化之一；不搜索窗口、不看结果调参。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("R3-E 只允许六个事前登记的 signal 标识")
    signal = parameters["signal"]

    def compute(ctx):
        universe = np.asarray(ctx.universe, dtype=bool)
        traded = np.asarray(ctx.traded(), dtype=bool)
        if signal in REV_DAYS:
            if signal == "rev_drift_decay10":
                value = _rev_decay(ctx)
            else:
                drift, since = _rev_flat(ctx)
                if drift is None:
                    return ctx.panel.empty(dtype=np.float64)
                w = REV_DAYS[signal]
                value = np.where((since >= 1) & (since <= w), -drift, np.nan)
            if value is None:
                return ctx.panel.empty(dtype=np.float64)
            value = np.where(universe & traded, value, np.nan)
            return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

        event, listed = _ca_state(ctx)
        enough = mx.roll_count(listed.astype(np.float64),
                               CA_HISTORY_GATE) >= int(MIN_HISTORY_FRAC * CA_HISTORY_GATE)
        if signal in CA_WINDOW:
            w = CA_WINDOW[signal]
            hist = mx.roll_count(listed.astype(np.float64), w)
            gate = np.isfinite(hist) & (hist >= max(1, int(np.ceil(MIN_HISTORY_FRAC * w))))
            count = mx.roll_sum(np.where(event, 1.0, 0.0), w, min_count=w)
            if signal == "ca_rate_w250":
                value = _divide(count, hist, floor=0)
            else:
                value = count
            value = np.where(gate, value, np.nan)
        else:
            value = _ewma(event, EWMA_HALFLIFE)
            value = np.where(enough, value, np.nan)
        value = np.where(traded, value, np.nan)
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
