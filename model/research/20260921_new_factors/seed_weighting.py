"""种子加权：等权 vs 按验证窗信息加权（**不看测试窗**）。

动机：k 曲线显示"多个种子的平均"比单种子稳。但种子之间质量差很多（门槛净收益 +10%~+161%）。
能不能只用**验证窗**的信息给它们加权，把差的种子压低？

★ 只用 `complete.json` 的 `best_validation`（**验证窗**指标，不碰测试窗）。
  这是与 R29/R33 同一条纪律：验证窗信息可以用来选/加权，测试窗不行。

用法：python seed_weighting.py --seeds V62 V34 V36 V50 V51 V52
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import evaluation_core as E
from combo_eval import gate_signal, scale_to
from seed_ensemble_size import ens_of

MODEL = Path('/autodl-fs/data/model')
QS = ['2025Q3', '2025Q4', '2026Q1', '2026Q2']


def val_stats(version):
    """四个季度 × 四折的验证窗统计（只读验证窗指标）。"""
    out = []
    for q in QS:
        for f in range(1, 5):
            d = json.loads((MODEL/version/'model_train'/q/f'fold{f}'/'complete.json').read_text())
            out.append(d['best_validation'])
    return dict(val_wei=np.mean([x['val_wei'] for x in out]),
                val_ret=np.mean([x['val_return_proxy'] for x in out]),
                val_rankic=np.mean([x['RankIC'] for x in out]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', nargs='+', required=True)
    a = ap.parse_args()
    panel = E.Panel(load_x=False); px = E.Prices(panel)
    _, RISK = gate_signal(px.close * px.adj)
    raw = {v: ens_of(v)[0] for v in a.seeds}
    dates = ens_of(a.seeds[0])[1]; days = np.searchsorted(panel.days, dates)
    ref = raw[a.seeds[0]]
    sc = {v: (ref if v == a.seeds[0] else scale_to(raw[v], ref)) for v in a.seeds}
    stats = {v: val_stats(v) for v in a.seeds}
    print(pd.DataFrame(stats).T.to_string(float_format=lambda x: f'{x:.4f}'))

    def run(w, tag):
        ens = sum(sc[v]*w[v] for v in a.seeds if w[v] > 0).astype(np.float32)
        ens = ens / sum(w.values())
        g = np.where(RISK[days][:, None], ens, np.nan)
        s, curve, _ = E.stress_cash_backtest(g, panel, px, days, n=1, slippage=0.0003)
        eq = np.r_[100000., [x['equity'] for x in curve]]; r = eq[1:]/eq[:-1]-1
        print(f'{tag:28s} 门槛净收益 {s["return_value"]:+7.2%} 回撤 {s["max_drawdown"]:7.2%} '
              f'剔5日 {np.prod(1+np.sort(r)[:-5])-1:+7.2%} 笔数 {s["trades"]}')

    n = len(a.seeds)
    run({v: 1.0 for v in a.seeds}, f'等权（k={n}）')
    for key in ('val_wei', 'val_ret', 'val_rankic'):
        z = np.array([stats[v][key] for v in a.seeds])
        run({v: float(x) for v, x in zip(a.seeds, z)}, f'按 {key} 线性加权')
        run({v: float(np.exp(3*(x-z.mean())/max(z.std(ddof=1), 1e-9))) for v, x in zip(a.seeds, z)},
            f'按 {key} softmax(3σ) 加权')
        run({v: float(max(x, 0.0)) for v, x in zip(a.seeds, z)}, f'按 {key} 截断到非负')


if __name__ == '__main__':
    main()
