"""种子维的集成规模曲线：k 份打分平均（逐日截面标准差对齐后线性相加）→ 含费净收益。

R30 测过的是**折**维（k=1→4 单调变好且未饱和），**种子**维只在资金层面测过（1/3 资金各跑一份），
而资金层面的三模型组合（+130.67%）反而低于单种子最好那份（+160.57%）。本脚本把种子维挪到
**打分层面**重测，并枚举所有 k 子集（同 R30 的表：报均值与区间，不挑最好的那一组）。

用法：python seed_ensemble_size.py --versions V62 V34 V36 --model-args "V62=337,V34=337,V36=337"
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluation_core as E
from combo_eval import QUARTERS, MODEL, gate_signal, scale_to


def ens_of(version):
    chunks, dates = [], []
    for q in QUARTERS:
        pf = []
        for f in range(1, 5):
            fo = MODEL / version / 'model_train' / q / f'fold{f}'
            d = json.loads((fo / 'complete.json').read_text())
            pf.append(np.load(fo / 'test_predictions.npy'))
            if f == 1:
                dates.extend(d['test_dates'])
        chunks.append(sum(pf).astype(np.float32))
    assert len(dates) == 242
    return np.concatenate(chunks), dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--versions', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    panel = E.Panel(load_x=False)
    px = E.Prices(panel)
    _, RISK_ON = gate_signal(px.close * px.adj)

    raw = {}
    for v in args.versions:
        raw[v], dates = ens_of(v)
    days = np.searchsorted(panel.days, dates)

    ref = raw[args.versions[0]]
    scaled = {v: (ref if i == 0 else scale_to(raw[v], ref)) for i, v in enumerate(args.versions)}

    rows = []
    for k in range(1, len(args.versions) + 1):
        combos = list(itertools.combinations(args.versions, k))
        nets, gates, mdds = [], [], []
        for combo in combos:
            ens = (sum(scaled[v] for v in combo) / k).astype(np.float32)
            for tag, pred in (('plain', ens), ('gate', np.where(RISK_ON[days][:, None], ens, np.nan))):
                stat, curve, _ = E.stress_cash_backtest(pred, panel, px, days, n=1, slippage=0.0003)
                if tag == 'plain':
                    nets.append(stat['return_value'])
                    mdds.append(stat['max_drawdown'])
                else:
                    gates.append(stat['return_value'])
        rows.append(dict(k=k, 组合数=len(combos), 单模型净收益均值=np.mean(nets),
                         门槛净收益均值=np.mean(gates), 门槛净收益最小=np.min(gates),
                         门槛净收益最大=np.max(gates), 回撤均值=np.mean(mdds)))
    df = pd.DataFrame(rows)
    out = MODEL / 'research/20260921_new_factors' / args.out
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'seed_ensemble_size.csv', index=False)
    print(df.to_string(index=False, float_format=lambda x: f'{x:,.4f}'))


if __name__ == '__main__':
    main()
