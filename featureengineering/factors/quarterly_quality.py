"""单季度盈利、增长和现金流的改善与稳定性，供次日横截面模型使用。

公式为本轮自定义研究候选，不宣称预测有效，也不照搬累计 YTD 比率。
字段口径见 datadownload/灵启数据API有权限接口文档，财报披露版本沿用 Derivative。
只用报告期 lag；截至当时未披露或缺失的季度不以零替代。
"""
from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register


FIELDS = ("q_roe", "q_dt_roe", "q_npta", "q_ocf_to_sales", "q_sales_yoy")
NOTE = (
    "本轮自定义单季度质量候选，原始字段来自上游接口字典，沿用财务版本表的公告时点。"
    "字段百分数除以100转比例；同比差为比例差，非同比增长率。"
    "lag按报告期且取当时已知版本，非日频位移；四季统计要求连续四个报告期全部有效，std为总体标准差。"
    "2026Q2发现52只池内股票五个核心字段同时为0，保守将该报告期的整组值标为缺失；"
    "单个字段真实零值及负值保留，非有限值为NaN；不回退到旧报告来掩盖缺失。"
    "仅用于当晚生成、次日交易；方向为经济解释，未经收益检验。"
    "源表若已覆盖旧版本或追溯修订公告日，无法凭现存快照还原真实历史到达时间。"
)


def _spec(name, desc, formula, high=True):
    return FactorSpec(
        name=name, group="quarterly_quality", desc=desc, formula=formula,
        deps=("stock_financial_indicator",), fin_fields=FIELDS,
        warmup_days=1100, higher_is_better=high, note=NOTE,
    )


def _quarter(ctx, field, lag=0):
    # 联合全零是源表整组占位的迹象；不把每个零都删掉，以保留真实盈亏平衡。
    values = {f: np.asarray(ctx.ind(f, lag=lag), dtype=np.float64) for f in FIELDS}
    placeholder = np.logical_and.reduce([values[f] == 0 for f in FIELDS])
    value = values[field]
    return np.where(np.isfinite(value) & ~placeholder, value / 100.0, np.nan)


def _change(ctx, field, lag):
    return _quarter(ctx, field) - _quarter(ctx, field, lag)


def _four(ctx, field, mode):
    values = np.stack([_quarter(ctx, field, lag) for lag in range(4)])
    # 不用 nanmin/nanstd：缺季时应保持缺失，不能把两季伪装成四季稳定性。
    return np.min(values, axis=0) if mode == "floor" else np.std(values, axis=0, ddof=0)


@register(_spec("qf_roe_yoy_change", "单季度ROE同比改善", "(q_roe(P)-q_roe(P-4))/100"))
def qf_roe_yoy_change(ctx):
    return _change(ctx, "q_roe", 4)


@register(_spec("qf_core_roe_yoy_change", "单季度扣非ROE同比改善", "(q_dt_roe(P)-q_dt_roe(P-4))/100"))
def qf_core_roe_yoy_change(ctx):
    return _change(ctx, "q_dt_roe", 4)


@register(_spec("qf_roa_yoy_change", "单季度资产净利率同比改善", "(q_npta(P)-q_npta(P-4))/100"))
def qf_roa_yoy_change(ctx):
    return _change(ctx, "q_npta", 4)


@register(_spec("qf_sales_growth_accel", "单季度收入同比增速的环比变化", "(q_sales_yoy(P)-q_sales_yoy(P-1))/100"))
def qf_sales_growth_accel(ctx):
    return _change(ctx, "q_sales_yoy", 1)


@register(_spec("qf_sales_growth_floor_4q", "最近四季收入同比增长的最低值", "min(q_sales_yoy(P-k),k=0..3)/100"))
def qf_sales_growth_floor_4q(ctx):
    return _four(ctx, "q_sales_yoy", "floor")


@register(_spec("qf_sales_growth_vol_4q", "最近四季收入同比增长波动", "std_population(q_sales_yoy(P-k),k=0..3)/100", high=False))
def qf_sales_growth_vol_4q(ctx):
    return _four(ctx, "q_sales_yoy", "vol")


@register(_spec("qf_core_roe_floor_4q", "最近四季扣非ROE最低值", "min(q_dt_roe(P-k),k=0..3)/100"))
def qf_core_roe_floor_4q(ctx):
    return _four(ctx, "q_dt_roe", "floor")


@register(_spec("qf_core_roe_vol_4q", "最近四季扣非ROE波动", "std_population(q_dt_roe(P-k),k=0..3)/100", high=False))
def qf_core_roe_vol_4q(ctx):
    return _four(ctx, "q_dt_roe", "vol")


@register(_spec("qf_noncore_roe_gap", "单季度ROE中的非经常损益贡献差", "(q_roe(P)-q_dt_roe(P))/100", high=False))
def qf_noncore_roe_gap(ctx):
    return _quarter(ctx, "q_roe") - _quarter(ctx, "q_dt_roe")


@register(_spec("qf_cash_margin_yoy_change", "单季度经营现金收入比同比改善", "(q_ocf_to_sales(P)-q_ocf_to_sales(P-4))/100"))
def qf_cash_margin_yoy_change(ctx):
    return _change(ctx, "q_ocf_to_sales", 4)


@register(_spec("qf_cash_margin_floor_4q", "最近四季经营现金收入比最低值", "min(q_ocf_to_sales(P-k),k=0..3)/100"))
def qf_cash_margin_floor_4q(ctx):
    return _four(ctx, "q_ocf_to_sales", "floor")


@register(_spec("qf_cash_margin_vol_4q", "最近四季经营现金收入比波动", "std_population(q_ocf_to_sales(P-k),k=0..3)/100", high=False))
def qf_cash_margin_vol_4q(ctx):
    return _four(ctx, "q_ocf_to_sales", "vol")
