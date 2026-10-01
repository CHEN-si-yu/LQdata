"""尾盘 30 分钟结构：T 日 14:30–15:00，T 收盘后可用。

使用 `Closing30Layer` 严格校验的 48 根 5 分钟棒。因子定义表达
"尾盘推升/冲高回落"的交易机制；方向按历史目标相关性复核，
尚未完成模型增益或交易后收益检验。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register


def _field(ctx, name: str) -> np.ndarray:
    return np.asarray(ctx.closing30_field(name), dtype=np.float64)


def _share(ctx) -> np.ndarray:
    tail = _field(ctx, "amt_tail30")
    day = _field(ctx, "amt_day")
    out = np.full_like(day, np.nan, dtype=np.float64)
    # 全天不足 1000 元的股日对 Top5 策略没有可用容量。
    ok = np.isfinite(tail) & np.isfinite(day) & (day >= 1_000.0) & (tail >= 0)
    np.divide(tail, day, out=out, where=ok)
    return np.where(ok, np.clip(out, 0.0, 1.0), np.nan)


def _ret(ctx) -> np.ndarray:
    op = _field(ctx, "open_tail30")  # 14:35 棒的 open = 14:30 价格
    cl = _field(ctx, "close_tail30")  # 15:00 棒的 close
    out = np.full_like(op, np.nan, dtype=np.float64)
    ok = np.isfinite(op) & (op > 0) & np.isfinite(cl) & (cl > 0)
    np.divide(cl, op, out=out, where=ok)
    return np.where(ok, out - 1.0, np.nan)


def _giveback(ctx) -> np.ndarray:
    hi = _field(ctx, "high_tail30")
    cl = _field(ctx, "close_tail30")
    out = np.full_like(hi, np.nan, dtype=np.float64)
    ok = np.isfinite(hi) & (hi > 0) & np.isfinite(cl) & (cl > 0)
    np.divide(cl, hi, out=out, where=ok)
    return np.where(ok, np.maximum(0.0, 1.0 - out), np.nan)


@register(FactorSpec(
    name="close30_amt_share", group="intraday",
    desc="收盘前 30 分钟成交额占全日比例",
    formula="Σamount[14:35..15:00] / Σamount[09:35..15:00]",
    deps=("stock_history_5min",), warmup_days=0, version=1,
    higher_is_better=True,
    note="48 根时钟完整、全天成交额≥1000 元；尾盘零成交是真实的 0。金额单位为元。方向未检验。",
))
def close30_amt_share(ctx):
    return _share(ctx)


@register(FactorSpec(
    name="close30_ret", group="intraday",
    desc="尾盘 14:30 至 15:00 的未复权同日收益",
    formula="close[15:00] / open[14:35] - 1",
    deps=("stock_history_5min",), warmup_days=0, version=2,
    higher_is_better=False,
    note="14:35 棒右端时间戳，其 open 是 14:30 价格；同日比值不需跨日复权。尾盘无成交为 NaN。历史 1d/5d RankIC 同为负，按反转方向设定；模型增益待验证。",
))
def close30_ret(ctx):
    return _ret(ctx)




@register(FactorSpec(
    name="close30_giveback", group="intraday",
    desc="尾盘最高价至 15:00 收盘的回落幅度",
    formula="1 - close[15:00] / max(open, high, close)[14:35..15:00]",
    deps=("stock_history_5min",), warmup_days=0, version=1,
    higher_is_better=False,
    note="逐棒扩张异常 high，回落幅度非负；尾盘无成交或价格缺测为 NaN。回落越大可能代表供给压力，尚未验证。",
))
def close30_giveback(ctx):
    return _giveback(ctx)



@register(FactorSpec(
    name="close30_crowded_fade", group="intraday",
    desc="尾盘成交集中伴随冲高回落的强度",
    formula="(Σamount[14:35..15:00]/Σamount[全天]) × max(0,1-close[15:00]/max_tail_high)",
    deps=("stock_history_5min",), warmup_days=0, version=2,
    higher_is_better=True,
    note="把尾盘放量与价格回落交互；历史 1d RankIC 为正，按隔日反转方向设定。与 giveback 的增量区别须单独验证。",
))
def close30_crowded_fade(ctx):
    return _share(ctx) * _giveback(ctx)
