"""两份打分在**头部**的分歧诊断：第 1 名是不是同一只？分歧时谁对？

口径与名次剖面一致：只在"次日开盘买得进、当日有 1d 标签"的票里排名次。
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluation_core as E

MODEL = Path('/autodl-fs/data/model'); QS = ['2025Q3','2025Q4','2026Q1','2026Q2']

def ensemble(v):
    chunks, dates = [], []
    for q in QS:
        pf = []
        for f in range(1,5):
            fo = MODEL/v/'model_train'/q/f'fold{f}'
            d = json.loads((fo/'complete.json').read_text())
            pf.append(np.load(fo/'test_predictions.npy'))
            if f==1: dates.extend(d['test_dates'])
        chunks.append(sum(pf).astype(np.float32))
    return np.concatenate(chunks), dates

def main():
    v1, v2 = sys.argv[1], sys.argv[2]
    panel = E.Panel(load_x=False); px = E.Prices(panel)
    a, dates = ensemble(v1); b, _ = ensemble(v2)
    days = np.searchsorted(panel.days, dates)
    r1, r2 = [], []
    agree = 0; n = 0
    for i, d in enumerate(days):
        y = panel.Y['label_ret_1d'][d]
        can = np.isfinite(a[i]) & np.isfinite(b[i]) & px.next_entry[d] & np.isfinite(y)
        if can.sum() < 10: continue
        idx = np.flatnonzero(can)
        o1 = idx[np.argsort(-a[i][idx], kind='stable')]
        o2 = idx[np.argsort(-b[i][idx], kind='stable')]
        n += 1
        agree += int(o1[0] == o2[0])
        r1.append((y[o1[0]], y[o2[0]]))   # (v1 的第1名收益, v2 在该票上的… ) 见下
        r2.append(y[o2[0]])
    r1 = np.asarray([x[0] for x in r1]); r2 = np.asarray(r2)
    print(f'{v1} 第1名 日均 {100*r1.mean():+.4f}%  t={r1.mean()/r1.std(ddof=1)*np.sqrt(len(r1)):.2f}')
    print(f'{v2} 第1名 日均 {100*r2.mean():+.4f}%  t={r2.mean()/r2.std(ddof=1)*np.sqrt(len(r2)):.2f}')
    print(f'两者第1名相同：{agree}/{n} = {agree/n:.1%}')
    same = r1 == r2
    print(f'  相同日：{v1} {100*r1[same].mean():+.4f}% (n={same.sum()})')
    if (~same).sum():
        print(f'  分歧日：{v1} {100*r1[~same].mean():+.4f}% | {v2} {100*r2[~same].mean():+.4f}% (n={(~same).sum()})')

main()
