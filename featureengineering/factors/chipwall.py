"""收盘价附近的筹码供给因子。

原始档位由 `fea.chipwall.ChipWallLayer` 按年聚合并缓存；因子函数只读
日频摘要，避免因子 worker 重复扫描原始筹码表。
"""

from __future__ import annotations

from fea.spec import FactorSpec, register


@register(FactorSpec(
    name="chip_near_overhang_5pct",
    group="chip",
    deps=("stock_cyq_chips", "stock_daily"),
    desc="收盘价上方 0–5% 的筹码供给占比；高值表示近端待解套供给较重",
    formula=(
        "sum(percent[(close < price) & (price <= 1.05 * close)]) / "
        "sum(percent[valid chip levels]); close is same-day unadjusted stock_daily.close"
    ),
    start="2018-01-02",
    warmup_days=20,
    higher_is_better=False,
    note=(
        "筹码 price 与当日未复权 close 同口径；有效档位要求 price>0、percent>=0，"
        "且两者有限。分母是当日有效档位的实际 Σpercent，不假设总和为 100。"
        "无当日筹码行、无当日 close、Σpercent<=0 或少于两个有效档位时为 NaN；"
        "有有效分布但区间内无筹码时为 0，不对缺失做前向填充。"
        "T 日收盘后可得，用于 T+1 开盘下单；higher_is_better=False 是供给压力的"
        "研究假设，预测有效性须通过后续样本外检验。"
    ),
))
def chip_near_overhang_5pct(ctx):
    return ctx.chipwall_field("near_overhang_5pct")
