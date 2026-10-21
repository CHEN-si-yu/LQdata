"""R12 批 C · C1 tick-rule 净换手率（family `r12c_net_turnover_v1`；**alpha_v2（B）通道**）。

════════════════════════════════════════════════════════════════════════
一、机制与方向预注册（H2）
════════════════════════════════════════════════════════════════════════
用 5min 棒的**价格方向**（tick rule / Lee-Ready 的 K 线近似）对成交额签名：

    sign_t = +1  若 close_t > open_t（主动买占优的棒）
    sign_t = −1  若 close_t < open_t（主动卖占优的棒）
    sign_t = 上一非零方向棒的方向（“沿用上棒符号”）
    段首（当日首棒）用**隔夜跳空方向** sign(close_first − pre_close) 定符号；
    跳空为 0 时退化为 sign(close_first − open_first)；仍不可判定 ⇒ 该棒记 0 权重。

    signed_amt_t = sign_t × amount_t          （amount 单位恒为元，见 §三 单位口径）
    NT(d)   = Σ_{t∈d} signed_amt_t / (close_d × free_share_d)     （自由流通市值，元）
    factor  = 过去 N 个交易日的 NT 均值（N = 5 / 20，两个期限结构变体）

**方向先验（事前冻结，不得反号）**：NT 高（净主动买入占比高）→ 未来 h=1/h=5 收益正向
（`higher_is_better=True`，declared = raw）。
**证伪点**：若 h=1 或 h=5 的 declared RankIC HAC95 下界 ≤ 0，或正向年份 < 6/9（政策 v2 G4），
该先验即被证伪，按 H2 不反号、直接否决。

与在产厂商资金流（`mf_*` 五项：按**单笔金额分档**汇总）不同，本对象按**价格路径**
分类（tick rule），是同一构念的**不同估计量**（idea_bank r12u2_001；对照已删 374 项中
24 项资金流均为厂商分档口径、26 项 intraday 均为量/价占比型，无签名成交量构造）。

════════════════════════════════════════════════════════════════════════
二、数据与 PIT（H5）
════════════════════════════════════════════════════════════════════════
· 只经 `ctx.dataset("stock_history_5min", ...)` 读**原始 5min 年分区**（截断审计管得到），
  逐 (股票, 日) 段聚合成日频值后走 `ctx.panel.place`（**精确落格、缺格 NaN**——停牌日
  必须是 NaN 而不是陈旧值，这与 `asof` 的语义区分见 fea/panel.py::place 文档）。
· 自由流通市值 = **未复权当日 close**（`ctx.px_raw("close")`，当日缺行即 NaN）×
  `free_share`（`ctx.px`，状态量前向填充）——与在产 `log_mv` 的自算市值约定同源，
  只用 T 日及以前信息；5min 价格与 stock_daily 同为**未复权**，分子分母口径一致。
· `stock_history_5min` / `stock_daily` / `stock_finance` 在 `conf/frequency.yaml` 中
  `delay_days: 0`（T 日收盘后可得），**不需要 `lag_grid` 位移**。
· 模块内**没有** shift / 前向填充 / 全样本统计量；全部为因果滚动窗口。
· 分段 = 「日期变化 | 时段回跳」（沿用 r3/r4b 的既有范式；不依赖 48 网格，兼容
  47/49 根 bar 的不规整股票日），段内 (股票, 日) 唯一性与日期解析做抽样复核，
  不通过直接抛错。午休跨段收益与本机制无关（本机制只用**棒内** open/close 与
  棒间方向传播，不做收益累乘），故不剔除午休棒。

════════════════════════════════════════════════════════════════════════
三、单位口径（vol 手/股翻转对本机制**完全不适用**）
════════════════════════════════════════════════════════════════════════
`stock_history_5min.vol` 的单位在 2025-12-01 前后发生 手→股 翻转（fea/intraday.py
实测；本模块 2026-09-30 抽样复核 000001.SZ：amount/vol ≈ 99.7 × close ⇒ 当期又回到「手」，
即单位**分段且不稳定**）。本机制**只使用 amount（元，全程一致）**：
签名对象是金额、分母是元口径市值，**不读 vol**，故对该翻转严格免疫
（与 r4b 的金额机制同一处置思路）。C2 读 vol 但只做**当日**排序与比值，亦然。

════════════════════════════════════════════════════════════════════════
四、缺失与闸门约定
════════════════════════════════════════════════════════════════════════
· 当日可用棒数（open/close 可判、amount 有限且 >0 的棒）< MIN_BARS(40) ⇒ 当日 NT 记 NaN
  （与 r4b/R3-B 的 MIN_BARS=40 同口径）。
· 池外 / 未上市 / ST（`ctx.universe` 为 False）⇒ NaN。停牌日（无 5min 行）⇒ NaN。
· 5min 源当前覆盖 2018-01-02 ~ 2026-09-30（逐日 ≥100 池内股票，预检见 gcC/README），
  源此后的尾部为 NaN（不是 0）。
· 窗口均值 `ctx.roll_mean(daily, N, min_count)`，min_count = 5→4 / 20→15（75% 有效日，
  r4b 口径）；窗内有效日不足 ⇒ NaN，不做 partial 均值。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

log = logging.getLogger("research.families.r12c_net_turnover_v1")

SOURCE = "stock_history_5min"
MIN_BARS = 40            # 当日可用棒下限（与 R3-B / r4b 同口径）
AUDIT_SEGMENTS = 2000    # 分段完整性抽样复核段数
# 事前冻结：(kind, window) -> 候选名；窗口只登记 5 / 20 两个期限（期限结构式，不扫网格）
FROZEN = {
    ("nt", 5): "r12c_net_turnover_5_v1",
    ("nt", 20): "r12c_net_turnover_20_v1",
}
MIN_VALID = {5: 4, 20: 15}     # 窗口内有效日下限（75%）


def _frozen_key(parameters):
    if not isinstance(parameters, dict):
        return None
    if parameters.get("kind") != "nt":
        return None
    return ("nt", parameters.get("window"))


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
        daily = _nt_daily(ctx)
        value = ctx.roll_mean(daily, window, min_valid)
        value = np.where(np.isfinite(value), value, np.nan)
        grid = np.where(np.asarray(ctx.universe, dtype=bool), value, np.nan)
        log.info("r12c_nt window=%d panel_finite=%d", window, int(np.isfinite(grid).sum()))
        return np.ascontiguousarray(grid, dtype=np.float64)

    return compute


# ---------------------------------------------------------------- 分段工具
# 以下五个函数逐字沿用 `research/families/r4b_intraday_shape_v1.py`（上游 = r3 范式）：
# 段边界、午休标记、抽样复核、段头事实、段内聚拢。改这里之前先读 r4b 的文档。
def _arrow(series):
    """pandas 的 arrow-backed 字符串列 -> pyarrow 数组（零拷贝优先）。"""
    arr = getattr(series, "array", None)
    pa_arr = getattr(arr, "_pa_array", None)
    if pa_arr is None:
        pa_arr = pa.array(series)
    return pa_arr


def _segments(frame):
    """切 (股票, 日) 段 + 标出午休跨段棒；返回 (starts, lengths, lunch_mask)。"""
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
    """抽样复核：每个段内的股票代码与日期必须唯一（能发现被误合并的段）。"""
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
    """段头取股票位与日期；唯一性守卫（同 r4b）。"""
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
    """把保留段的行按段连续聚拢；逐列取出（控制峰值内存）。"""
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


# ---------------------------------------------------------------- C1 日频值
def _nt_daily(ctx):
    """逐 (股票, 日) 的当日净换手率（未滚动）铺进面板；池外与不足 MIN_BARS 记 NaN。"""
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    y0, y1 = int(days[0]) // 10000, int(days[-1]) // 10000
    pre_close = np.asarray(ctx.px_raw("pre_close"), dtype=np.float64)
    close_raw = np.asarray(ctx.px_raw("close"), dtype=np.float64)
    free_share = np.asarray(ctx.px("free_share"), dtype=np.float64)
    cols = ["stock_code", "trade_time", "open", "close", "amount"]
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
                                   ["open", "close", "amount"])
        # 保留段在面板上的日行号（段头日必须精确命中面板日历；否则该段丢弃）
        drow = ctx.panel.rows_of(day_seg[keep])
        T = ctx.panel.T
        ok_day = (drow < T) & (ctx.panel.dates[np.clip(drow, 0, T - 1)] == day_seg[keep])
        # 每段的自由流通市值（元）：未复权 close × free_share；缺一即 NaN（整段 NaN）
        pc_seg = np.where(ok_day, pre_close[np.clip(drow, 0, T - 1), np.clip(pos_seg[keep], 0, ctx.panel.C - 1)], np.nan)
        cl_seg = np.where(ok_day, close_raw[np.clip(drow, 0, T - 1), np.clip(pos_seg[keep], 0, ctx.panel.C - 1)], np.nan)
        fs_seg = np.where(ok_day, free_share[np.clip(drow, 0, T - 1), np.clip(pos_seg[keep], 0, ctx.panel.C - 1)], np.nan)
        mktcap_seg = cl_seg * fs_seg
        seg_starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lens)[:-1]))
        value = _daily_value(data, seg_starts, lens, pc_seg, mktcap_seg)
        g = ctx.panel.place(pos_seg[keep], day_seg[keep], value)
        daily = g if daily is None else np.where(np.isfinite(g), g, daily)
        log.info("r12c_nt year=%d segs=%d kept=%d bars=%d finite_day=%d",
                 year, int(starts.size), int(keep.size), int(lens.sum()),
                 int(np.isfinite(value).sum()))
        del frame, data, rows
    if daily is None:
        return ctx.panel.empty(dtype=np.float64)
    return daily


def _daily_value(data, seg_starts, lens, pre_close_seg, mktcap_seg):
    """每个保留段的当日净换手率 = Σ(sign × amount) / 自由流通市值。

    sign 解析（段内一次完成，无 Python 循环）：
      · 逐棒 s = sign(close − open)（不可用棒记 0）；
      · 段内「最近一个非零 s」的下标用 `np.maximum.accumulate` 求出；
        非零棒取自身 s，其后为零的棒沿用之（“沿用上棒符号”）；
      · 首个非零棒之前的棒（含首棒本身若为平棒）取**段首隔夜跳空**方向
        sign(close_first − pre_close)；跳空不可判（pre_close 缺失或相等）⇒ 0 权重。
    """
    o, c, a = data["open"], data["close"], data["amount"]
    usable = np.isfinite(o) & (o > 0.0) & np.isfinite(c) & (c > 0.0) & np.isfinite(a)
    diff = c - o
    s = np.where(usable & (diff > 0.0), 1.0,
                 np.where(usable & (diff < 0.0), -1.0, 0.0))
    idx = np.arange(s.size, dtype=np.int64)
    last_nz = np.maximum.accumulate(np.where(s != 0.0, idx, -1))

    # 段首隔夜跳空符号：sign(首棒 close − 当日 pre_close)；首棒不可用或跳空为 0 ⇒ 0
    first = seg_starts
    first_ok = usable[first] & np.isfinite(pre_close_seg) & (pre_close_seg > 0.0)
    gap = np.zeros(first.size, dtype=np.float64)
    gap[first_ok] = np.sign(c[first][first_ok] - pre_close_seg[first_ok])
    init = np.repeat(gap, lens)          # 段首符号铺满段（仅对段首前的零方向棒生效）

    sign = np.where(last_nz >= 0, s[np.maximum(last_nz, 0)], init)

    pos_amount = usable & (a > 0.0)
    contrib = np.where(pos_amount, sign * np.where(np.isfinite(a), a, 0.0), 0.0)
    net = np.add.reduceat(contrib, seg_starts)
    n_ok = np.add.reduceat(pos_amount.astype(np.float64), seg_starts)
    with np.errstate(all="ignore"):
        val = net / np.where(np.isfinite(mktcap_seg) & (mktcap_seg > 0.0), mktcap_seg, np.nan)
    good = (n_ok >= MIN_BARS) & np.isfinite(val)
    return np.where(good, val, np.nan)
