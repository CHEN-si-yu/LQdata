"""名次剖面：打分排序里第 1..10 名的逐日 1d 收益（套"次日开盘买得进"掩码）。

R28/R31 的验收工具：一个排序模型的名次剖面**本该在头部单调**；
"第 1 名不如第 5 名"就是打分口径有问题的直接证据，而网格扫描与 IC 都可能被路径骗过去。

用法：python rank_profile.py V62 V63 [V64 ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluation_core as E  # noqa: E402

MODEL = Path('/autodl-fs/data/model')
QUARTERS = ['2025Q3', '2025Q4', '2026Q1', '2026Q2']
TOPK = 10


def ensemble(version):
    """四折打分直接相加（与官方集成口径一致），并对齐到评价窗 242 天。"""
    chunks, dates = [], []
    for q in QUARTERS:
        per_fold = []
        for f in range(1, 5):
            folder = MODEL / version / 'model_train' / q / f'fold{f}'
            d = json.loads((folder / 'complete.json').read_text())
            p = np.load(folder / 'test_predictions.npy')
            assert p.shape == (len(d['test_dates']), 2115)
            per_fold.append(p)
            if f == 1:
                dates.extend(d['test_dates'])
        chunks.append(sum(per_fold).astype(np.float32))
    pred = np.concatenate(chunks)
    assert len(dates) == 242 and dates[0] == '2025-07-01'
    return pred, dates


def profile(pred, panel, px, days):
    acc = {k: [] for k in range(1, TOPK + 1)}
    for i, d in enumerate(days):
        p = pred[i]
        y = panel.Y['label_ret_1d'][d]
        can = np.flatnonzero(np.isfinite(p) & px.next_entry[d] & np.isfinite(y))
        if can.size < TOPK + 5:
            continue
        order = can[np.argsort(-p[can], kind='stable')]
        for k in range(1, TOPK + 1):
            acc[k].append(float(y[order[k - 1]]))
    rows = []
    for k, v in acc.items():
        v = np.asarray(v)
        rows.append(dict(rank=k, mean_pct=100 * v.mean(), t=float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))),
                         win_rate=float((v > 0).mean()), n=len(v)))
    return pd.DataFrame(rows)


def main():
    panel = E.Panel(load_x=False)
    px = E.Prices(panel)
    out = MODEL / 'research/20260921_new_factors'
    frames = {}
    for v in sys.argv[1:]:
        pred, dates = ensemble(v)
        days = np.searchsorted(panel.days, dates)
        t = profile(pred, panel, px, days)
        frames[v] = t
        print(f'\n=== {v} 名次剖面（第 1..10 名，日均 1d 收益，已套次日开盘可买掩码）===')
        print(t.to_string(index=False, float_format=lambda x: f'{x:.4f}'), flush=True)
    if len(frames) > 1:
        m = pd.DataFrame({v: t.set_index('rank')['mean_pct'] for v, t in frames.items()})
        m.index.name = 'rank'
        print('\n=== 并列（日均 1d 收益 %）===')
        print(m.to_string(float_format=lambda x: f'{x:+.4f}'))
    pd.concat(frames, names=['version']).to_csv(out / 'rank_profile.csv')


if __name__ == '__main__':
    main()
