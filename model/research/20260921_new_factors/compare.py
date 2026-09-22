"""配对比较：同一份新快照上，旧 337 列（V62）vs 全部 653 列（V63）。

只读两版已生成的产物（`model_pred/tables/*`、`market_gate_new/<版>/`），不改任何东西。
判据见同目录 `PLAN.md` §4（事前写死，不在这里挑好的报）。

    python compare.py V62 V63
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path('/autodl-fs/data/model')


CASH = {'top1_1d': 'cash_top1_1d.csv', 'top5_1d': 'cash_top5_1d.csv',
        'top1_exit2_1d': 'cash_buffer_r2_t1.csv', 'top1_exit10_1d': 'cash_buffer_r10_t1.csv'}


def load(ver):
    d = ROOT / ver
    rep = json.loads((d / 'model_pred/tables/report.json').read_text())
    main = pd.read_csv(d / 'model_pred/tables/main_metrics_1d.csv')
    strat = pd.read_csv(d / 'model_pred/tables/strategy_summary.csv')
    qs = pd.read_csv(d / 'model_pred/tables/quarterly_summary.csv')
    gate = None
    gp = d / 'market_gate_new' / ver / 'results.csv'
    if gp.exists():
        gate = pd.read_csv(gp)
    return dict(ver=ver, report=rep, main=main, strat=strat, quarterly=qs, gate=gate,
                n_folds=len(list((d / 'model_train').glob('*/fold*/complete.json'))))


def halves(d, fname):
    """从现金曲线算后半程收益（H2）—— strategy_summary 里没有这一列。"""
    p = d / 'model_pred/ensemble' / fname
    if not p.exists():
        return float('nan')
    c = pd.read_csv(p)
    eq = c['equity'].to_numpy()
    mid = len(eq) // 2
    return float(eq[-1] / eq[mid] - 1)


def ens(strat, name):
    row = strat[(strat['source'] == 'ensemble') & (strat['strategy'] == name)]
    return None if row.empty else row.iloc[0]


def main():
    a, b = sys.argv[1], sys.argv[2]
    A, B = load(a), load(b)
    print(f'# 配对比较 {a}（对照） vs {b}（处置）\n')
    for D in (A, B):
        print(f'{D["ver"]}: 完成折 {D["n_folds"]}/16, panel_digest={D["report"].get("panel_digest")}')
    print()

    print('## 1. 主评估指标（1d 口径）\n')
    print(f'| 季度 | {a} IC | {b} IC | ΔIC | {a} Σtop1 | {b} Σtop1 | ΔΣtop1 |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    ma, mb = A['main'].set_index('quarter'), B['main'].set_index('quarter')
    for q in ma.index:
        if q not in mb.index:
            continue
        print(f'| {q} | {ma.loc[q,"IC"]:+.4f} | {mb.loc[q,"IC"]:+.4f} | '
              f'{mb.loc[q,"IC"]-ma.loc[q,"IC"]:+.4f} | {ma.loc[q,"top_return"]:+.4f} | '
              f'{mb.loc[q,"top_return"]:+.4f} | {mb.loc[q,"top_return"]-ma.loc[q,"top_return"]:+.4f} |')

    print('\n## 2. 集成分数上的策略净收益（单边 3bp 含费现金账户）\n')
    print(f'| 策略 | {a} 净收益 | 回撤 | 后半程 | {b} 净收益 | 回撤 | 后半程 | Δ收益(pp) |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    names = ['top1_1d', 'top5_1d', 'top1_exit2_1d', 'top1_exit10_1d']
    for n in names:
        ra, rb = ens(A['strat'], n), ens(B['strat'], n)
        if ra is None or rb is None:
            continue
        ha, hb = halves(ROOT / a, CASH[n]), halves(ROOT / b, CASH[n])
        print(f'| {n} | {ra["return_value"]:+.2%} | {ra["max_drawdown"]:.2%} | {ha:+.2%} | '
              f'{rb["return_value"]:+.2%} | {rb["max_drawdown"]:.2%} | {hb:+.2%} | '
              f'{(rb["return_value"]-ra["return_value"])*100:+.2f} |')

    if A['gate'] is not None or B['gate'] is not None:
        print('\n## 3. 固定 60 日市场门槛（单边 3bp）\n')
        for D in (A, B):
            if D['gate'] is None:
                print(f'{D["ver"]}: 门槛结果缺失')
                continue
            print(f'--- {D["ver"]} ---')
            print(D['gate'].to_string(index=False))
            print()
    print(f'\nanon/时间/资源采样：{ROOT}/{a}/logs/resource_summary.json, {ROOT}/{b}/logs/resource_summary.json')


if __name__ == '__main__':
    main()
