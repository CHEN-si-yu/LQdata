"""新因子的单因子体检：只用**严格早于评价窗**的数据算，不构成收益结论。

对上游新交付的 316 个因子，逐年算：
  ① 逐日截面 RankIC（因子 rank 列 vs `label_ret_5d`）的年度均值与 ICIR；
  ② 与旧 337 列的最大 |秩相关|（冗余度）—— 同一天截面内两两秩相关。

窗口：**2018-01-02 ~ 2025-06-30**（评价窗 2025-07-01 起，严格隔开）。
用法：python newfactor_ic.py [--out 文件名.csv]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

MODEL = Path('/autodl-fs/data/model')
SNAP = MODEL / 'trainingdata'
OLD337 = json.loads((MODEL / 'V62/features_old337.json').read_text())
META = json.loads((SNAP / 'meta.json').read_text())
ALL = META['columns']['features']
NEW = [f for f in ALL if f not in set(OLD337)]
LABEL = 'label_ret_5d'
CUTOFF = '2025-06-30'


def year_ic(year, cols):
    f = pq.read_table(SNAP / f'factors/year={year}/data.parquet',
                      columns=['trade_date', 'stock_code'] + cols).to_pandas()
    t = pq.read_table(SNAP / f'target/year={year}/data.parquet',
                      columns=['trade_date', 'stock_code', LABEL]).to_pandas()
    df = f.merge(t, on=['trade_date', 'stock_code'], how='left')
    df = df[df['trade_date'] <= CUTOFF]
    out = {c: [] for c in cols}
    for day, g in df.groupby('trade_date', sort=True):
        y = g[LABEL].to_numpy(np.float64)
        ok_y = np.isfinite(y)
        if ok_y.sum() < 50:
            continue
        for c in cols:
            x = g[c].to_numpy(np.float64)
            m = ok_y & np.isfinite(x)
            if m.sum() < 50:
                continue
            xr = x[m]
            if np.all(xr == xr[0]):
                out[c].append(0.0)
                continue
            out[c].append(float(np.corrcoef(rankdata(xr), rankdata(y[m]))[0, 1]))
    del f, t, df
    return {c: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0., len(v))
            for c, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='newfactor_ic.csv')
    ap.add_argument('--redundancy-sample-days', type=int, default=250)
    args = ap.parse_args()

    rows = []
    for year in range(2018, 2026):
        r = year_ic(year, NEW)
        for c, (m, s, n) in r.items():
            rows.append(dict(factor=c, year=year, RankIC=m, ICIR=(m / s if s else 0.0), days=n))
        print(f'{year} 完成：{len(r)} 个因子', flush=True)
    per_year = pd.DataFrame(rows)
    agg = per_year.groupby('factor').apply(
        lambda g: pd.Series(dict(
            ic_mean=g['RankIC'].mean(),
            icir=(g['RankIC'].mean() / g['RankIC'].std(ddof=1)) if len(g) > 1 and g['RankIC'].std(ddof=1) else 0.0,
            ic_pos_years=int((g['RankIC'] > 0).sum()), years=len(g))), include_groups=False).reset_index()

    # 冗余：与旧 337 列在同一天截面内的最大 |秩相关|（抽若干天）
    days_ix = []
    for year in range(2020, 2025):
        d = pq.read_table(SNAP / f'factors/year={year}/data.parquet', columns=['trade_date'])
        days_ix.extend(sorted(set(d.column(0).to_pylist()))[::40])
    days_ix = days_ix[:args.redundancy_sample_days]
    by_year = {}
    for d in days_ix:
        by_year.setdefault(d[:4], []).append(d)
    max_rho = {c: [] for c in NEW}
    for year, days in by_year.items():
        f = pq.read_table(SNAP / f'factors/year={year}/data.parquet',
                          columns=['trade_date'] + OLD337 + NEW).to_pandas()
        f = f[f['trade_date'].isin(days)]
        for day, g in f.groupby('trade_date'):
            old = g[OLD337].to_numpy(np.float64)
            new = g[NEW].to_numpy(np.float64)
            # 先按列秩化，再算相关（缺失填中位秩）
            def rk(a):
                a = np.where(np.isfinite(a), a, np.nan)
                out = np.empty_like(a)
                for j in range(a.shape[1]):
                    col = a[:, j]
                    m = np.isfinite(col)
                    out[:, j] = 0.5
                    if m.sum() > 10 and not np.all(col[m] == col[m][0]):
                        out[m, j] = (rankdata(col[m]) - .5) / m.sum()
                return out
            ro = rk(old); rn = rk(new)
            ro = ro - ro.mean(0); rn = rn - rn.mean(0)
            den = np.sqrt((ro ** 2).sum(0))[:, None] * np.sqrt((rn ** 2).sum(0))[None, :]
            with np.errstate(invalid='ignore', divide='ignore'):
                rho = np.abs((ro.T @ rn) / den)
            rho = np.nan_to_num(rho)
            for j, c in enumerate(NEW):
                max_rho[c].append(float(rho[j].max()))
        del f
        print(f'冗余 {year} 完成', flush=True)
    agg['max_abs_rho_vs_old'] = agg['factor'].map({c: float(np.mean(v)) for c, v in max_rho.items()})
    agg = agg.sort_values('ic_mean', key=lambda s: -s.abs())
    out = MODEL / 'research/20260921_new_factors' / args.out
    agg.to_csv(out, index=False)
    print(f'\n已写 {out}（{len(agg)} 个新因子）')
    print(agg.head(25).to_string(index=False))
    print('\n--- 与旧列冗余最高的一批 ---')
    print(agg.sort_values('max_abs_rho_vs_old', ascending=False).head(10).to_string(index=False))


if __name__ == '__main__':
    main()
