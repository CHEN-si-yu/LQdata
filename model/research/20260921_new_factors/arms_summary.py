"""全部实验臂的总表：一臂一行，主判据与诊断指标并列。

用法：python arms_summary.py V62 V63 V64 V65 V66 V67
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = Path('/autodl-fs/data/model')
OUT = MODEL / 'research/20260921_new_factors'

DESC = {
    'V32': '旧快照 337 列（基线）', 'V62': '337 列（对照，=V32）', 'V63': '全部 653 列',
    'V64': '337+mfx28+efx32=397', 'V65': '337+按|IC|前40=371', 'V66': '337+随机40=377',
    'V67': '只用新因子 316 列', 'V34': '旧快照 337 列 seed3254', 'V36': '旧快照 337 列 seed3255',
}


def row(ver):
    d = MODEL / ver
    main = pd.read_csv(d / 'model_pred/tables/main_metrics_1d.csv')
    tot = main[main['quarter'] == '总计'].iloc[0]
    strat = pd.read_csv(d / 'model_pred/tables/strategy_summary.csv')
    e = strat[strat.source == 'ensemble']
    g = e[e.strategy == 'top1_1d'].iloc[0]
    g5 = e[e.strategy == 'top5_1d'].iloc[0]
    curve = d / 'model_pred/ensemble/cash_top1_1d.csv'
    H2 = np.nan
    if curve.exists():
        eq = pd.read_csv(curve)['equity'].to_numpy()
        H2 = eq[-1] / eq[len(eq) // 2] - 1
    res = dict(ver=ver, 说明=DESC.get(ver, ''), IC=tot.IC, ICIR=tot.ICIR, Σtop1=tot.top_return,
               top1净收益=g.return_value, top1回撤=g.max_drawdown, top1后半程=H2,
               top1费用=g.total_fees, top1笔数=g.trades,
               top5净收益=g5.return_value, top5费用=g5.total_fees)
    gp = d / 'market_gate_new' / ver / 'results.csv'
    if gp.exists():
        gg = pd.read_csv(gp)
        r1 = gg[(gg.source == 'ensemble') & (gg.topn == 1)]
        r5 = gg[(gg.source == 'ensemble') & (gg.topn == 5)]
        if len(r1):
            res['门槛top1'] = float(r1.iloc[0].return_value)
            res['门槛top1回撤'] = float(r1.iloc[0].max_drawdown)
            res['门槛top1剔5日'] = float(r1.iloc[0].remove_best5)
            res['门槛top1后半程'] = float(r1.iloc[0].H2)
        if len(r5):
            res['门槛top5'] = float(r5.iloc[0].return_value)
    return res


def main():
    vers = sys.argv[1:] or ['V62', 'V63', 'V64']
    df = pd.DataFrame([row(v) for v in vers])
    df.to_csv(OUT / 'arms_summary.csv', index=False)
    pd.set_option('display.width', 250)
    show = ['ver', '说明', 'IC', 'ICIR', 'Σtop1', 'top1净收益', 'top1回撤', 'top1后半程', 'top1笔数',
            '门槛top1', '门槛top1回撤', '门槛top1后半程', '门槛top1剔5日', 'top5净收益', '门槛top5']
    print(df[[c for c in show if c in df.columns]].to_string(index=False, float_format=lambda x: f'{x:,.4f}'))


if __name__ == '__main__':
    main()
