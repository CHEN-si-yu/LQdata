"""财务未开发字段的年度结构/年度变化因子。

刻意使用已披露年报，避开上游Q1/Q3的结构字段占位零值，以及累计YTD比率的季节跳变。
按日可用不等于每日有新财务信息。它们是慢变量候选，不代表经过收益筛选。
可解释定义逐项放在 conf/field_expansion.json，而不是从任意数值列盲目自动注册。
"""
import numpy as np

from fea.field_expansion import CATALOG, annual_alias
from fea.spec import FactorSpec, register

NOTE = (
    '本地扩展公式；仅用截至当晚已披露的12月31日年度报告，按源表独立保留公告修订版本。'
    '年度结构比率分母必须为正且超过阈值，缺失不补零，单独零值保留。'
    '年度同比按报告期滞后4季，缺年保持NaN；不将累计YTD比率直接当成日频值。'
    'asinh仅为单调压缩极端值，不做因子内截面缩尾/排名。'
    'ratio为同币种或每股同单位比值，delta按供应商原单位相减；year-end单季度字段代表Q4。'
    '正负方向仅代表字段暴露的高低，不宣称收益方向；固定池和源表历史修订局限沿用项目约定。'
)


def _build(entry):
    ds, field = entry['dataset'], entry['field']
    dds, den = entry['den_dataset'], entry['den_field']
    mode = entry['mode']
    first = f'annual({ds}.{field})'
    formula = {'ratio':f'asinh({first}/annual({dds}.{den}))',
               'delta':f'asinh({first}-annual_lag1y({ds}.{field}))',
               'growth':f'asinh(({first}-annual_lag1y({ds}.{field}))/abs(annual_lag1y({ds}.{field})))'}[mode]
    fields = tuple(annual_alias(src, f) for src, fs in entry['source_fields'].items() for f in fs)

    @register(FactorSpec(name=entry['name'], group=entry['group'], desc=entry['desc'],
        formula=formula, deps=tuple(entry['source_fields']), fin_fields=fields,
        warmup_days=1100, start=entry.get('start'), note=NOTE+entry['note']))
    def compute(ctx):
        value = np.asarray(ctx.annual(ds, field), dtype=np.float64)
        if mode == 'ratio':
            denominator = np.asarray(ctx.annual(dds, den), dtype=np.float64)
            out = ctx.safe_div(value, np.where(denominator > entry['min_den'], denominator, np.nan), entry['min_den'])
        else:
            previous = np.asarray(ctx.annual(ds, field, lag_years=1), dtype=np.float64)
            out = value - previous
            if mode == 'growth':
                out = ctx.safe_div(out, np.abs(previous), entry['min_den'])
        return np.arcsinh(np.where(np.isfinite(out), out, np.nan))

    compute.__name__ = entry['name']
    return compute


for _entry in CATALOG:
    globals()[_entry['name']] = _build(_entry)
