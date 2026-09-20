"""目标标签（label）—— 给模块③ 模型训练用的**预测目标**，不是因子。

参考库的定义（`学习资料/factors.md` §四.5 标签语义）：
    T 日收盘出信号 → **T+1 开盘买入** → 持有 N 个交易日 → T+1+N 开盘卖出
    label = open_hfq(T+1+N) / open_hfq(T+1) − 1

**为什么用开盘价而不是收盘价**：T 日的因子值只有收盘后才知道，最早能成交的时点
就是 T+1 开盘。用收盘价当买入价会引入一天的不可实现收益。

**为什么用后复权**：`hfq_open(t) = 未复权开盘价(t) × adj_factor(t)`。
除权日当天开盘价会因为除权而"跳水"，不做复权的话标签里会混进一堆假的负收益。

**格式**：与所有因子**完全相同的 4 列**（用户要求格式统一），
但 `rank` 列恒为 NaN —— 对未来收益做截面排名没有意义。
标签也**不参与** `main.py eval` 的 IC 统计（它是被解释变量），但**参与**格式校验。
"""

from __future__ import annotations

import numpy as np

from fea import mathx as mx
from fea.spec import FactorSpec, register

# 复用已有的 label 能力：market 的横截面标签是最常用的
HORIZONS = (1, 3, 5, 10, 20)


def _make_label(h: int):
    """生成一个持有 h 个交易日的标签因子函数。

    行 T 上要取的是：
        buy  = o[T+1]     -> shift(o, -1)[T]        （下一个交易日开盘买）
        sell = o[T+1+h]   -> shift(o, -(h+1))[T]    （再持有 h 个交易日）
    注意 k<0 是「取未来」，全框架只有标签会用到；面板已经向后延伸了 h+1 个交易日，
    所以行 T 落在输出窗口末尾时也拿得到未来值。
    """
    def _fn(ctx):
        o = ctx.hfq("open")
        buy = mx.shift(o, -1)
        sell = mx.shift(o, -(h + 1))
        raw = mx.safe_div(sell, buy, min_abs_den=1e-12) - 1.0
        # ★★ 卖出日必须落在**真有行情**的范围内（2026-09-15 修的真 bug）。
        #   面板会为标签向后延伸 `forward_days` 个交易日，但延伸出去的那几天
        #   **没有真实行情**：价格层对"水平量"（open 等）做前向填充，
        #   于是 `o[T+21]` 会取到"最后一个真实交易日"的陈旧价 ——
        #   结果是一条 4~6 天的短收益被冒充成 20 天标签，**而且不是 NaN，查不出来**。
        #   实测：`label_ret_20d` 在 2026-09-04/07/08 还有值，而它们的 T+21
        #   落在 2026-10 中旬（上游只到 09-15）。
        #   判据：`ctx.traded()` 对"超出数据范围"的行恒为 False（vol 是流量，不做 ffill），
        #   所以取"最后一个还有任一股票成交的行"，卖出行号超过它的格子一律置 NaN。
        traded = np.asarray(ctx.traded())
        idx = np.flatnonzero(traded.any(axis=1))
        if idx.size == 0:
            return np.full(raw.shape, np.nan)
        last_row = int(idx[-1])
        rows = np.arange(raw.shape[0]) + (h + 1)          # 卖出行号
        return np.where((rows <= last_row)[:, None], raw, np.nan)
    return _fn


for _h in HORIZONS:
    register(FactorSpec(
        name=f"label_ret_{_h}d",
        group="label",
        deps=("stock_daily", "stock_adj_factor"),
        desc=f"T+1 开盘买入、T+{_h + 1} 开盘卖出，持有 {_h} 个交易日的收益",
        formula=f"label = open_hfq(T+1+{_h}) / open_hfq(T+1) - 1",
        start=None,
        warmup_days=5,
        forward_days=_h + 1,
        is_label=True,
        higher_is_better=True,
        version=2,
        note="★ 这是**标签不是因子**：引擎跳过了 winsor 与截面 rank（rank 列恒为 NaN），"
             "value 即未来收益。复权开盘价 = 未复权开盘价 × 当日累计 adj_factor"
             "（后复权锚定，PIT 安全）。"
             "★★ v2 修了一个**静默算错**的 bug（2026-09-15）：面板为标签向后延伸了 "
             "`forward_days` 天，但延伸段**没有真实行情**，而价格层对开盘价做了前向填充 →"
             "`o[T+21]` 取到「最后一个真实交易日」的陈旧价，一条 4~6 天的短收益被冒充成 "
             "20 天标签（实测 `label_ret_20d` 在 09-04/07/08 仍有值，而它们的 T+21 在 10 月中旬）。"
             "现在按「卖出行必须落在还有成交的范围内」截断，超出的一律 NaN。",
    ))(_make_label(_h))
