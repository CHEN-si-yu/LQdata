"""R2-I2 除权除息与送转事件：四个预登记机制（复权因子跳变比）。

只读 stock_adj_factor_changes / stock_adj_factor / stock_daily。所有阈值在
查看任何 IC/收益之前固定，见下方常量与注释。

关键实现约定（均已实测，证据见 artifacts 与交付报告）
---------------------------------------------------
1. 事件强度用**跳变比** ratio(T) = adj(T)/adj(T-1)，不用差分。
   池内 2018-01-02…2026-09-30 共 16446 个 changes 命中日上，
   corr(ratio, close(T-1)/pre_close(T)) = 0.9999993，比值中位数 1.000000，
   即 ratio 正是除权除息参考价隐含的比例。差分不具备这个性质。
2. 事件日 = changes 表命中 AND (ratio-1) > NOISE_EPS。
   只用 ratio 会把厂商 8 个「批量刷新日」写下的假跳变当成事件：实测
   2018-08-16 / 2018-08-28 / 2020-01-02 / 2021-01-07 / 2023-06-01 /
   2024-06-25 / 2024-06-26 / 2022-05-30 共 4457 行未命中 changes 表，
   最大 ratio 1.0647。反向不受影响：未命中行里没有一行 ratio >= 1.10。
3. ratio >= SPLIT_RATIO_MIN(1.10) 记为「送转/转增」，1e-4 < ratio-1 < 1.10
   记为「现金分红」。1.10 的依据：A 股最小送转单位是每 10 股送转 1 股(=1.10)；
   纯现金要达到 1.10 需要单次股息率 >= 9.09%；池内实测事件密度谷在
   [1.10,1.20)（8.75 年共 109 个事件，而 [1.00,1.02) 有 10191 个、
   [1.30,1.32) 有 228 个）。
4. 「窗口内无事件」记为 0，**不是缺失**：池内 2115 只的每个 stock_daily 行
   都有原始 adj_factor（实测缺 0 / 6053645），所以「无事件」是被观测到的
   状态而非记录缺失。窗口内原始 adj_factor 有效天数不足 80% 时输出 NaN。
5. 非 A 股代码（00025.HK 等）与池外股票由 ctx.code_index 在 event_grid 内
   直接丢弃，不进入任何网格。
6. 全程 float64；只在最终返回 float64，由 Store 转 float32。
   时间方向：全部为前向递推（只用到 <= T 的行），无向后填充、无未来位移。
   递推状态在窗口长度 W 处截断，所以每个年度面板只要预热 >= W 个交易日，
   结果就与全量重算逐格一致。
"""
from __future__ import annotations

import numpy as np

from fea import mathx as mx

SIGNALS = ("split_magnitude", "ex_days_since", "corpaction_frequency", "dividend_stability")

# ---- 事前固定的阈值（不得在看到 IC 后调整） ----
NOISE_EPS = 1e-4          # 厂商 4 位小数舍入抖动上限；<= 它不算事件
SPLIT_RATIO_MIN = 1.10    # 送转 / 现金分红分界
MIN_HISTORY_FRAC = 0.8    # 窗口内原始 adj_factor 有效天数下限（占比）


def _divide(a, b, floor=1e-12):
    with np.errstate(all="ignore"):
        result = np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64)
    return np.where(np.isfinite(result) & (np.abs(b) > floor), result, np.nan)


def _pool_relative(ctx, values):
    """用当前固定池的当日有效样本等权去均值；不足 min_cross_section 记缺失。"""
    valid = np.asarray(ctx.universe, dtype=bool) & np.isfinite(values)
    count = valid.sum(axis=1)
    minimum = max(3, int(ctx.cfg.min_cross_section))
    mean = _divide(np.where(valid, values, 0.0).sum(axis=1), count, floor=0)
    mean = np.where(count >= minimum, mean, np.nan)
    return np.where(valid, values - mean[:, None], np.nan)


def _ratio(ctx):
    """(T,C) 复权因子跳变比。adj_factor 是状态量：按交易日前向沿用后再取比。"""
    raw = np.asarray(ctx.px("adj_factor"), dtype=np.float64)
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, np.nan)
    af = mx.nan_fill_ffill(raw)
    prev = mx.shift(af, 1)
    with np.errstate(all="ignore"):
        ratio = af / prev
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
    return af, ratio


def _events(ctx, ratio):
    """事件掩码：changes 表命中 AND 跳变幅度超过噪声底。"""
    days = ctx.panel.dates
    frame = ctx.dataset("stock_adj_factor_changes", columns=["stock_code", "date"],
                        years=(int(days[0]) // 10000, int(days[-1]) // 10000))
    flag = np.zeros(ctx.panel.shape, dtype=bool)
    if not frame.empty:
        grid = ctx.event_grid(frame["stock_code"].to_numpy(),
                              ctx.date_col(frame["date"]),
                              np.ones(len(frame), dtype=np.float64))
        flag = grid > 0
    jump = ratio - 1.0
    event = flag & np.isfinite(jump) & (jump > NOISE_EPS)
    split = event & (ratio >= SPLIT_RATIO_MIN)
    cash = event & (ratio < SPLIT_RATIO_MIN)
    return event, split, cash


def _last_event_scan(events, magnitudes, cap):
    """前向递推：距最近事件的天数（上限 cap）与最近事件的幅度。"""
    T, C = events.shape
    cap = float(cap)
    age = np.full(C, cap)
    mag = np.zeros(C)
    out_age = np.empty((T, C), dtype=np.float64)
    out_mag = np.empty((T, C), dtype=np.float64)
    for t in range(T):
        ev = events[t]
        age = np.where(ev, 0.0, np.minimum(age + 1.0, cap))
        mag = np.where(ev, magnitudes[t], mag)
        out_age[t] = age
        out_mag[t] = mag
    return out_age, out_mag


def _since_event_scan(events, daily, cap):
    """前向递推：距最近事件的天数与事件后的累计值（事件当日归零）。"""
    T, C = events.shape
    cap = float(cap)
    age = np.full(C, cap)
    acc = np.zeros(C)
    out_age = np.empty((T, C), dtype=np.float64)
    out_acc = np.empty((T, C), dtype=np.float64)
    for t in range(T):
        ev = events[t]
        age = np.where(ev, 0.0, np.minimum(age + 1.0, cap))
        acc = np.where(ev, 0.0, acc + daily[t])
        acc = np.where(age >= cap, 0.0, acc)
        out_age[t] = age
        out_acc[t] = acc
    return out_age, out_acc


def build(parameters):
    """四个预登记机制之一；窗口在 family 里事前登记，不做搜索。"""
    if set(parameters) != {"signal", "window"} or parameters["signal"] not in SIGNALS:
        raise ValueError("R2-I2 只允许四个预登记机制与已登记窗口")
    signal = parameters["signal"]
    window = int(parameters["window"])
    if window <= 0:
        raise ValueError("window 必须为正")

    def compute(ctx):
        af, ratio = _ratio(ctx)
        event, split, cash = _events(ctx, ratio)
        logjump = np.where(event, np.log(np.maximum(ratio, 1.0)), 0.0)
        listed = np.isfinite(af).astype(np.float64)
        history = mx.roll_count(listed, window)
        need = max(1, int(np.ceil(MIN_HISTORY_FRAC * window)))
        enough = np.isfinite(history) & (history >= need)
        traded = np.asarray(ctx.traded(), dtype=bool)

        if signal == "split_magnitude":
            # 最近一次「送转」事件的幅度，按事件新鲜度线性衰减；窗口内无送转 = 0。
            age, mag = _last_event_scan(split, logjump, window)
            value = mag * np.maximum(0.0, 1.0 - age / float(window))
        elif signal == "ex_days_since":
            # 距上次除权除息的天数（新鲜度）× 除权后累计的池相对表现。
            one = np.asarray(ctx.ret_clean(1), dtype=np.float64)
            one = np.where(traded & np.isfinite(one) & (one > -1.0), one, np.nan)
            with np.errstate(all="ignore"):
                logret = np.log1p(one)
            rel = _pool_relative(ctx, logret)
            rel = np.where(np.isfinite(rel), rel, 0.0)
            age, acc = _since_event_scan(event, rel, window)
            value = acc * np.maximum(0.0, 1.0 - age / float(window))
        elif signal == "corpaction_frequency":
            # 过去 window 个交易日的除权除息事件次数。
            value = mx.roll_sum(np.where(event, 1.0, 0.0), window, min_count=window)
        else:
            # 过去 window 个交易日「小额跳变」（现金分红）的累计对数复权幅度。
            value = mx.roll_sum(np.where(cash, logjump, 0.0), window, min_count=window)

        value = np.where(np.isfinite(value), value, np.nan)
        value = np.where(traded & enough, value, np.nan)
        return value.astype(np.float64, copy=False)

    return compute
