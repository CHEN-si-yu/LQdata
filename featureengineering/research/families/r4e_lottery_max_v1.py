"""Round 4 · 单元 U5：彩票/极端收益结构（MAX 族）—— 只读价格层，方向全部预先登记为 false。

机制集合（任务书 §4-U5 事前冻结，4 个，方向不可改）：
  1. `max_ret_20`           过去 20 个交易日（含 T）最大单日收益（后复权）
  2. `idio_max_20`          市场调整残差 (ret − mkt_ew_ret) 的 20 日最大值
  3. `idio_max_60`          同上 60 日版本
  4. `extreme_up_share_60`  60 日内单日收益 > 7% 的交易日占比

口径（事前冻结，报告披露）：
  · 日收益 r = `ctx.ret_clean(1)` = 后复权（未复权价 × 截至当日累计复权因子）日收益，
    停牌/未上市/当日无成交 → NaN（**不按 0 收益参与窗口**），|r|>60% 的复权脏数据
    由价格层置 NaN。等价于本库 volatility 家族 `np.where(ctx.traded(), ctx.ret(1), nan)`。
  · 市场调整 `mkt_ew_ret[T]` = 冻结池内 **当日** 等权平均日收益：
    对 `ctx.universe & isfinite(r)` 的格子逐日求均值（要求 ≥100 只，池 2115 只下从不触发），
    只用同日已实现收益，无未来信息。注意与正式因子 `idio_vol_60` 的
    「60 日滚动 beta × 沪深300」口径不同：本单元按任务书固定为等权池均值差。
  · 窗口完整性阈值统一取 min_count = ceil(0.75 × n)（n=20→15，n=60→45），
    与 `ret_skew_20`（20 窗口 min_count=15）同族口径；不足 → NaN。
  · `extreme_up_share_60`：分子 = 窗口内 1[r > 7.0%] 之和（有效交易日口径），
    分母 = 窗口内有效交易日数；分母 < 45 → NaN。阈值严格大于（不取等号）。
    与 `extreme_move_event` 的差异：后者用供应商 pct_chg 的 |·|≥7%（含跌），
    本因子用后复权日收益的严格上穿 7%，不测下跌侧。
"""
from __future__ import annotations

import numpy as np

SIGNALS = ("max_ret_20", "idio_max_20", "idio_max_60", "extreme_up_share_60")

# 事前冻结的窗口参数（不随结果调整）
_WINDOW = {"max_ret_20": 20, "idio_max_20": 20, "idio_max_60": 60, "extreme_up_share_60": 60}
_MIN_COUNT = {20: 15, 60: 45}          # ceil(0.75 × n)
_UP_THRESHOLD = 0.07                    # 严格大于 7%
_MKT_MIN_NAMES = 100                    # 池内等权均值的健康度下限（池 2115 只，从不触发）

__all__ = ["SIGNALS", "build"]


def _market_ew_return(ctx, r: np.ndarray) -> np.ndarray:
    """冻结池内当日等权平均日收益 (T,)，只用同日已实现且池内的收益。"""
    valid = np.asarray(ctx.universe, dtype=bool) & np.isfinite(r)
    count = valid.sum(axis=1)
    total = np.where(valid, r, 0.0).sum(axis=1)
    mean = np.full(r.shape[0], np.nan, dtype=np.float64)
    ok = count >= max(_MKT_MIN_NAMES, int(ctx.cfg.min_cross_section))
    mean[ok] = total[ok] / count[ok]
    return mean


def build(parameters):
    """四个预登记机制之一；不搜索窗口、不搜索方向。"""
    if set(parameters) != {"signal"} or parameters["signal"] not in SIGNALS:
        raise ValueError(f"Batch permits only the four preregistered signal identifiers: {SIGNALS}")
    signal = parameters["signal"]
    n = _WINDOW[signal]
    mc = _MIN_COUNT[n]

    def compute(ctx):
        universe = np.asarray(ctx.universe, dtype=bool)
        r = np.asarray(ctx.ret_clean(1), dtype=np.float64)
        r = np.where(universe, r, np.nan)
        if signal == "max_ret_20":
            value = ctx.roll_max(r, 20, min_count=mc)
        elif signal in ("idio_max_20", "idio_max_60"):
            market = _market_ew_return(ctx, r)
            idio = r - market[:, None]
            value = ctx.roll_max(idio, n, min_count=mc)
        else:  # extreme_up_share_60
            event = np.where(np.isfinite(r), (r > _UP_THRESHOLD).astype(np.float64), np.nan)
            numerator = ctx.roll_sum(event, 60, min_count=mc)
            denominator = ctx.roll_count(r, 60, min_count=mc)
            with np.errstate(all="ignore"):
                value = numerator / denominator
            value = np.where(np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0),
                             value, np.nan)
        # 输出仍按池掩码与有限性收口；NaN 语义 = 不知道（不填 0）。
        return np.where(universe & np.isfinite(value), value, np.nan).astype(np.float64, copy=False)

    return compute
