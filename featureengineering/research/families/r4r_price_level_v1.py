"""Round 4 批3 · V3 股价水平与颗粒度（family `r4r_price_level_v1`）。

预登记机制集合（4 个候选，恰好用满预算 ≤4；见
`conf/research/round4_20261004/price_level/{idea,family,cand_*}.yaml`）：

  1. `log_price_level`        = ln(当日原始收盘价)。方向 true（预登记）。
  2. `price_granularity`      = 最小报价单位占价比（tick=0.01 元）：100·tick/P = 1/P。
                                方向 true（预登记，与机制 1 同轴的两侧对照）。
  3. `log_price_level_ma60`   = ln(过去 60 个交易日原始收盘价均值)。方向 true（变体 1）。
  4. `price_granularity_ma60` = 100·tick/过去60日均价。方向 true（变体 2）。

★ 价格水平口径（**事前定死并披露**，在任何结果出来之前写下）：
  一律用 **原始名义收盘价**（未复权，`ctx.px_raw("close")`），**不用后复权价**。理由：
  (a) 构念是投资者实际面对的名义价格水平，以及最小报价单位（0.01 元，主板 A 股全区间
      不变）相对名义价的占比——tick 规则作用在名义价上，颗粒度只能用名义价；
  (b) 跨股票可比性：原始价全部落在同一名义 CNY 尺度上，两只报相同名义价的股票就是
      同一价格水平；后复权价 = 原始价 × 截至当日累计复权因子，会把各股上市以来不同的
      分红/送转史混进「价格水平」，截面排名不再可比（这正是任务书要求注意的口径一致性）；
  (c) `fea/spec.py` 顶部 PIT 红线的官方口径指引即「价格**水平**类（市值、book-to-market）
      → 用**未复权**价」；
  (d) 原始价天然 PIT 不变：任一历史日的原始收盘价不依赖任何时变的复权因子版本。
  已知代价（事前披露）：ma60 变体的 60 日窗口内若发生送转/拆股，原始价序列会出现一次
  台阶（个股自身、影响 ≤60 个交易日）；为保持与日频口径严格一致，接受该噪声而不改用
  后复权（后复权会引入 (b) 的截面不可比问题）。

实现约定（全部前台可复核）：
  - 有效格 = **当日真实成交**（`ctx.traded()`）且价格有限且 >0；停牌/无行情一律 NaN（四个
    候选统一，不用停牌前的旧价出值）。`px_raw` 是严格当日原始值（不外推），停牌日无行。
  - ma60 = 过去 60 个交易日（含 T）的滚动均值，NaN 感知，窗口内有效价 <48（80%）→ NaN；
    窗口内停牌日不贡献价格（skip），且 T 当日必须真实成交才出值。
  - 输出 float64；不做任何截面变换（下游面板负责缩尾/zscore/方向）。
  - 无未来函数：只读 T 及更早的行；无 shift(-k)、无 ffill 外推。

本模块不接受预登记之外的参数（严格 `{"signal": ...}` 精确匹配）。
"""
from __future__ import annotations

import numpy as np

from fea import mathx as mx

SIGNALS = ("log_price_level", "price_granularity",
           "log_price_level_ma60", "price_granularity_ma60")
WINDOW, MIN_COUNT = 60, 48          # 60 个交易日窗口，≥80% 有效价（项目惯例 0.8）
TICK = 0.01                         # 主板 A 股最小报价单位（元），全区间不变


def _close(ctx):
    """当日原始名义收盘价（未复权）；非真实成交/无效价为 NaN。"""
    traded = np.asarray(ctx.traded(), dtype=bool)
    close = np.asarray(ctx.px_raw("close"), dtype=np.float64)
    ok = traded & np.isfinite(close) & (close > 0)
    return np.where(ok, close, np.nan)


def build(parameters):
    """四个固定信号之一；参数只允许预登记的精确 {signal} 字典。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError("Batch permits only the four preregistered signal identifiers")
    signal = parameters["signal"]

    def compute(ctx):
        traded = np.asarray(ctx.traded(), dtype=bool)
        close = _close(ctx)
        if signal.endswith("_ma60"):
            close = np.asarray(mx.roll_mean(close, WINDOW, min_count=MIN_COUNT),
                               dtype=np.float64)
            close = np.where(np.isfinite(close) & (close > 0), close, np.nan)
        if signal.startswith("log_price_level"):
            with np.errstate(all="ignore"):
                value = np.where(np.isfinite(close), np.log(close), np.nan)
        else:
            # 最小报价单位占价比：100·(tick/close)（百分数表达）≡ 1/close
            value = np.where(np.isfinite(close), 100.0 * TICK / close, np.nan)
        value = np.where(traded, value, np.nan)      # 当日未真实成交 → NaN（统一口径）
        return np.where(np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
