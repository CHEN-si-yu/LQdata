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




def _ret(ctx) -> np.ndarray:
    op = _field(ctx, "open_tail30")  # 14:35 棒的 open = 14:30 价格
    cl = _field(ctx, "close_tail30")  # 15:00 棒的 close
    out = np.full_like(op, np.nan, dtype=np.float64)
    ok = np.isfinite(op) & (op > 0) & np.isfinite(cl) & (cl > 0)
    np.divide(cl, op, out=out, where=ok)
    return np.where(ok, out - 1.0, np.nan)






@register(FactorSpec(
    name="close30_ret", group="intraday",
    desc="尾盘 14:30 至 15:00 的未复权同日收益",
    formula="close[15:00] / open[14:35] - 1",
    deps=("stock_history_5min",), warmup_days=0, version=3,
    higher_is_better=False,
    note="14:35 棒右端时间戳，其 open 是 14:30 价格；同日比值不需跨日复权。尾盘无成交为 NaN。历史 1d/5d RankIC 同为负，按反转方向设定；模型增益待验证。",
))
def close30_ret(ctx):
    return _ret(ctx)







