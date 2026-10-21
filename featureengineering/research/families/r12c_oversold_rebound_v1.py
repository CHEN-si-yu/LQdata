"""R12 批 C · C7 相对超跌反弹事件（family `r12c_oversold_rebound_v1`；**alpha_v2（B）通道**）。

════════════════════════════════════════════════════════════════════════
一、机制与方向预注册（H2）
════════════════════════════════════════════════════════════════════════
A 股显著反转特征下，个股**相对指数超跌**后大概率短线反弹、收益右偏（idea_bank r12u2_057）。
T 日收盘出信号、T+1 开盘执行、持有 1 周（本模块只产出 T 日截面信号；持有期由 eval 的
h=1/h=5 RankIC 口径表达）：

    个股 20 日累计收益 r20 = Π_{k=0..19}(1 + r_daily(t−k)) − 1（后复权，停牌日贡献 0）
    指数 20 日累计收益 i20 = 同面板行上的沪深300（000300.SH）同口径累计
    相对收益 excess = (1 + r20) / (1 + i20) − 1
    事件条件：excess ≤ −threshold（threshold ∈ {10%, 15%}）**且 T 日非跌停**且 universe
    factor(t) = −excess（= **超跌深度**，事件日取正；非事件日 **NaN**）

**方向先验（事前冻结，不得反号）**：超跌越深（−excess 越大）→ 未来 h=1/h=5 收益正向
（`higher_is_better=True`，declared = raw；极端阈值化 + 相对指数定义）。
**证伪点**：h=1 或 h=5 的 declared RankIC HAC95 下界 ≤ 0、或正向年份 < 6/9（政策 v2 G4）
⇒ 先验被证伪，按 H2 不反号、直接否决。

两个预登记变体 = **阈值结构**（−10% / −15%），窗口恒 20 日（期限结构式，不扫网格）。

差异论证（H3/H4）：与在产反转族 `short_term_reversal_5` / `reversal_2d` 同族但构造不同 ——
增量 = **极端阈值化**（−10%/−15%）+ **相对指数**定义（去市场 β 与市场整体回撤）+
**剔除跌停**（不可成交的假信号）。已删 374 项无「相对指数超跌事件」构造。
须报告与 `short_term_reversal_5`、`momentum_20`、`limit_down_event_5` 的 pooled |ρ|
及**增量检验**（G5；>0.85 需论证）。

**H6/H7（稀疏/事件，报告侧必做）**：非事件日 NaN 占比高（2026 主板跌停约 0.4%/日，
本事件更稀）；须**并报零填充口径**（非事件日 = 0，事件日 = 深度）；
报告 unique/day（连续深度值，预计不触发 H7 的上限问题，但零占比高须留痕）。

**与 idea 原文的偏离（README 同步）**：idea r12u2_057 另要求「剔除一周内重复信号」。
本模块**不实现**该去重：本对象是**截面信号因子**（供 h=1/h=5 RankIC 评估），
持有期重叠的去重属交易执行层；去重会进一步稀释本就稀疏的事件（G2 覆盖风险），
且与「深度连续化」的排序语义冲突。若后续做事件组合策略，须另立模块并重登记。

════════════════════════════════════════════════════════════════════════
二、跌停判定（与 factors/event3.py 同一口径：LL 优先、2020 前价格近似）
════════════════════════════════════════════════════════════════════════
· `stock_limit_list.limit == 'D'` 为供应商跌停（该表 **2020-01-02 起**）；
· 2020 年前（及 LL 表无记录格）用价格近似：`pct_chg ≤ −9.5 且 close == low`
  （封死 = 收在跌停价；pct_chg 是**百分数**，停牌日 NaN ⇒ 不会误判）；
· 逐格选择与 event3 相同：该 (date, code) 在 LL 表有任意行 ⇒ 用供应商判定；
  无记行 ⇒ 用价格近似。实测两口径在 2020+ 的一致性见 event3 文档（D 标记 23,562 行中
  23,555 行 close == low）。

════════════════════════════════════════════════════════════════════════
三、PIT 与口径
════════════════════════════════════════════════════════════════════════
· 个股 20 日收益走 `ctx.ret(20)`（单日后复权收益的 log1p 滚动和；单日 ±60% 清洗只在
  单日层面做，k 日累计不套阈值）；**额外守卫**：20 日窗内成交日 ≥ 15（75%）才出值
  （`ctx.roll_sum(traded, 20)`），避免「停牌十几天后复牌」的伪 20 日累计。
· 指数走 `ctx.dataset("index_daily", ...)` 自算（000300.SH，`delay_days: 0`）；
  指数日收益 = close/prev_close − 1（同日历行），20 日累计用 log1p + `roll_sum(20, 20)`。
· `stock_limit_list` / `stock_daily` / `index_daily` 均 `delay_days: 0`，无 `lag_grid`。
· 事件当日若停牌（traded False）⇒ NaN；池外/未上市/ST ⇒ NaN。
· warmup_days=56（20 交易日 × 1.8 + 20 ≈ 56，契约 §2）。
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("research.families.r12c_oversold_rebound_v1")

INDEX_CODE = "000300.SH"      # 沪深300
INDEX_DS = "index_daily"
LL = "stock_limit_list"       # 涨/跌/炸板（2020-01-02 起）
LIMIT_PCT = 9.5               # 价格近似的涨跌停幅度（百分数；与 factors/event3.py 一致）
WIN = 20                      # 相对超跌窗（交易日）
MIN_TRADED = 15               # 窗内成交日下限（75%）
# 事前冻结：(kind, threshold) -> 候选名；只登记 −10% / −15% 两个阈值
FROZEN = {
    ("orb", 0.10): "r12c_orb_dd10_v1",
    ("orb", 0.15): "r12c_orb_dd15_v1",
}


def _frozen_key(parameters):
    if not isinstance(parameters, dict):
        return None
    if parameters.get("kind") != "orb":
        return None
    return ("orb", parameters.get("threshold"))


def _index_cum20(ctx):
    """沪深300 的 20 日累计收益（对齐面板行；缺口行记 NaN 不补）。"""
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    y0, y1 = int(days[0]) // 10000, int(days[-1]) // 10000
    frame = ctx.dataset(INDEX_DS, columns=["ts_code", "trade_date", "close"],
                        years=(y0, y1))
    T = ctx.panel.T
    close = np.full(T, np.nan, dtype=np.float64)
    if frame is not None and not frame.empty:
        sel = (frame["ts_code"] == INDEX_CODE).to_numpy()
        if sel.any():
            d = ctx.date_col(frame["trade_date"])[sel]
            v = frame["close"].to_numpy(np.float64)[sel]
            rows = ctx.panel.rows_of(d)
            ok = (rows < T) & (ctx.panel.dates[np.clip(rows, 0, T - 1)] == d)
            close[rows[ok]] = v[ok]
    # ★ ctx.shift 转发到 fea/mathx.shift，而 mathx 只接受 (T, C) 二维（`_as_f64` 会
    #   对 1-D 抛 ValueError）。指数收盘是一维序列，先加一列轴再取回，语义不变。
    prev = np.asarray(ctx.shift(close[:, None], 1), dtype=np.float64)[:, 0]
    with np.errstate(all="ignore"):
        r1 = close / prev - 1.0
    lg = np.where(np.isfinite(r1), np.log1p(r1), np.nan)
    cum = np.expm1(ctx.roll_sum(lg[:, None], WIN, min_count=WIN))[:, 0]
    return cum


def _limit_down_grid(ctx):
    """跌停 0/1 网格：`stock_limit_list.limit == 'D'` 优先，无记录格用价格近似
    （与 factors/event3.py::_limit_grid 逐格选择一致）。"""
    pc = np.asarray(ctx.px("pct_chg"), dtype=np.float64)
    cl = np.asarray(ctx.px("close"), dtype=np.float64)
    lo = np.asarray(ctx.px("low"), dtype=np.float64)
    approx = np.isfinite(pc) & (pc <= -LIMIT_PCT) & (np.abs(cl - lo) < 1e-6)

    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    y0, y1 = int(days[0]) // 10000, int(days[-1]) // 10000
    frame = ctx.dataset(LL, columns=["stock_code", "trade_date", "limit"], years=(y0, y1))
    if frame is None or frame.empty:
        return approx
    # NA-safe：`limit` 若含缺失，直接用 object 数组比较会抛
    # 「boolean value of NA is ambiguous」（实测 pandas 3.0 object 数组行为）；
    # 用 StringArray.eq + fillna(False) 规避，语义与 event3 的逐格选择一致。
    is_d = (frame["limit"].astype("string").eq("D").fillna(False)
            .to_numpy(dtype=bool).astype(np.float64))
    codes = frame["stock_code"].to_numpy()
    d = ctx.date_col(frame["trade_date"])
    any_ll = ctx.event_grid(codes, d) > 0.0
    ev_ll = ctx.event_grid(codes, d, is_d) > 0.0
    return np.where(any_ll, ev_ll, approx)


def build(parameters):
    """只允许预登记的两个阈值变体；不搜索窗口、不搜索方向。"""
    key = _frozen_key(parameters)
    if key not in FROZEN:
        raise ValueError(
            "Only the frozen preregistered variants are permitted; got %r. Allowed: %s"
            % (parameters, sorted(FROZEN)))
    threshold = float(key[1])

    def compute(ctx):
        univ = np.asarray(ctx.universe, dtype=bool)
        r20 = np.asarray(ctx.ret(WIN), dtype=np.float64)
        traded = np.asarray(ctx.traded(), dtype=bool)
        n_tr = ctx.roll_sum(traded.astype(np.float64), WIN, min_count=0)
        i20 = _index_cum20(ctx)[:, None]
        with np.errstate(all="ignore"):
            excess = (1.0 + r20) / (1.0 + i20) - 1.0
        ld = _limit_down_grid(ctx)
        event = (np.isfinite(excess) & (excess <= -threshold) & (ld <= 0.0)
                 & traded & (n_tr >= MIN_TRADED) & univ)
        depth = np.where(event, -excess, np.nan)
        grid = np.where(univ & traded, depth, np.nan)
        log.info("r12c_orb thr=%.2f events=%d", threshold, int(np.isfinite(grid).sum()))
        return np.ascontiguousarray(grid, dtype=np.float64)

    return compute
