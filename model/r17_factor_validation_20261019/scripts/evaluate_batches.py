#!/usr/bin/env python3
"""Aggregate the A/B factor-batch experiment.

Outputs (under results/):
  quarter_model.csv  - 4 models x 2 batches x 4 quarters test table (from picks.md)
  fold_valid.csv     - 4 models x 2 batches x 4 quarters x 4 folds validation metrics
  fold_test.csv      - fold-level test IC/RankIC recomputed from test_predictions.npy
  comparison.json    - paired A/B deltas and summaries
  summary.md         - human-readable report
"""
from __future__ import annotations
import json, re, statistics
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent.parent
MODEL = WORK.parent
UNITS = WORK / 'units'
SNAP = WORK / 'snapshots'
RES = WORK / 'results'
RES.mkdir(exist_ok=True)
QUARTERS = ['2025Q3','2025Q4','2026Q1','2026Q2']
METRICS = ['return','IC_1d','IC_5d','RankIC_1d','RankIC_5d']

units = json.loads((WORK/'scripts/unit_manifest.json').read_text())
UNIT = {(u['batch'], u['model']): u for u in units}
MODELS = ['V220','V230','V240','V252']


def parse_picks(path: Path):
    text = path.read_text(encoding='utf-8')
    rows = {}
    in_test = False
    for line in text.splitlines():
        if 'test 评价窗' in line:
            in_test = True
            continue
        if in_test and 'valid 训练验证集' in line:
            break
        if not in_test:
            continue
        parts = line.split()
        if parts and parts[0] in QUARTERS + ['总计'] and len(parts) >= 6:
            try:
                vals = [float(x) for x in parts[1:6]]
            except ValueError:
                continue
            rows[parts[0]] = dict(zip(METRICS, vals))
    return rows


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def valid_rows(batch, model):
    out = []
    for q in QUARTERS:
        for f in range(1,5):
            p = UNITS/f'{batch}_{model}'/'model_train'/q/f'fold{f}'/'complete.json'
            if not p.exists():
                continue
            d = load_json(p)
            bv = d.get('best_validation', {})
            row = dict(batch=batch, model=model, quarter=q, fold=f,
                       seconds=d.get('seconds'), epochs=d.get('epochs'),
                       best_epoch=d.get('best_epoch'), peak_rss_gib=d.get('peak_rss_gib'))
            for k in ('val_wei','IC_1d','IC_3d','IC_5d','IC_10d','IC_20d','RankIC_1d','RankIC_3d','RankIC_5d','RankIC_10d','RankIC_20d'):
                row[k] = bv.get(k)
            out.append(row)
    return out


class LabelCache:
    def __init__(self, snap):
        self.snap = snap
        self.meta = load_json(snap/'meta.json')
        self.codes = list(self.meta['axis']['codes'])
        self._years = {}
    def year(self, y):
        if y not in self._years:
            p = self.snap/'target'/f'year={y}'/'data.parquet'
            df = pd.read_parquet(p, columns=['trade_date','stock_code','label_ret_1d','label_ret_5d'])
            df['trade_date'] = df['trade_date'].astype(str).str[:10]
            self._years[y] = {c: df.pivot(index='trade_date', columns='stock_code', values=c).reindex(columns=self.codes)
                              for c in ('label_ret_1d','label_ret_5d')}
        return self._years[y]


def daily_ic(pred, lab, min_n=100):
    ok = np.isfinite(pred) & np.isfinite(lab)
    n = int(ok.sum())
    if n < min_n:
        return np.nan, np.nan
    p = pred[ok].astype(np.float64); y = lab[ok].astype(np.float64)
    if p.std() == 0 or y.std() == 0:
        return np.nan, np.nan
    ic = float(np.corrcoef(p, y)[0,1])
    pr = pd.Series(p).rank(method='average').to_numpy()
    yr = pd.Series(y).rank(method='average').to_numpy()
    ric = float(np.corrcoef(pr, yr)[0,1])
    return ic, ric


def fold_test_rows(batch, model, cache):
    out = []
    for q in QUARTERS:
        for f in range(1,5):
            folder = UNITS/f'{batch}_{model}'/'model_train'/q/f'fold{f}'
            cp = folder/'complete.json'
            if not cp.exists():
                continue
            d = load_json(cp)
            dates = [str(x)[:10] for x in d['test_dates']]
            pred = np.load(folder/'test_predictions.npy')
            if pred.shape[0] != len(dates):
                raise ValueError(f'{folder}: prediction rows {pred.shape} != dates {len(dates)}')
            ic1=[];ic5=[];ric1=[];ric5=[]
            for i, day in enumerate(dates):
                y = cache.year(int(day[:4]))
                p = pred[i]
                a,b = daily_ic(p, y['label_ret_1d'].loc[day].to_numpy())
                ic1.append(a); ric1.append(b)
                a,b = daily_ic(p, y['label_ret_5d'].loc[day].to_numpy())
                ic5.append(a); ric5.append(b)
            out.append(dict(batch=batch, model=model, quarter=q, fold=f, days=len(dates),
                            IC_1d=np.nanmean(ic1), IC_5d=np.nanmean(ic5),
                            RankIC_1d=np.nanmean(ric1), RankIC_5d=np.nanmean(ric5),
                            IC_1d_sd=np.nanstd(ic1), IC_5d_sd=np.nanstd(ic5)))
    return out


def desc(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals: return dict(n=0)
    return dict(n=len(vals), mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0,
                min=float(np.min(vals)), max=float(np.max(vals)),
                positive=int(sum(v>0 for v in vals)), negative=int(sum(v<0 for v in vals)))


def main():
    qrows = []
    for (batch, model), u in UNIT.items():
        picks = UNITS/f'{batch}_{model}'/'model_pred'/'picks.md'
        fa = UNITS/f'{batch}_{model}'/'model_info'/'final_audit.json'
        if not picks.exists():
            print('MISSING picks', batch, model); continue
        parsed = parse_picks(picks)
        total = load_json(fa).get('metrics_total', {}) if fa.exists() else {}
        for q, metrics in parsed.items():
            qrows.append(dict(batch=batch, model=model, quarter=q, **metrics))
        # totals in picks already include 总计 row; add final_audit metrics as separate fields
        if '总计' in parsed:
            for k,v in (total or {}).items():
                if isinstance(v,(int,float)) and k not in ('days',):
                    qrows[-1][f'audit_{k}'] = v
    qdf = pd.DataFrame(qrows)
    qdf.to_csv(RES/'quarter_model.csv', index=False)

    vrows = []
    for b in ('A','B'):
        for m in MODELS:
            vrows += valid_rows(b,m)
    vdf = pd.DataFrame(vrows)
    vdf.to_csv(RES/'fold_valid.csv', index=False)

    caches = {b: LabelCache(SNAP/('A_260' if b=='A' else 'B_310')) for b in ('A','B')}
    trows = []
    for b in ('A','B'):
        for m in MODELS:
            trows += fold_test_rows(b,m,caches[b])
    tdf = pd.DataFrame(trows)
    tdf.to_csv(RES/'fold_test.csv', index=False)

    comp = {}
    # quarter-level paired deltas
    for m in MODELS:
        comp[m] = {}
        for metric in METRICS:
            d = []
            for q in QUARTERS:
                a = qdf[(qdf.batch=='A')&(qdf.model==m)&(qdf.quarter==q)]
                b = qdf[(qdf.batch=='B')&(qdf.model==m)&(qdf.quarter==q)]
                if len(a) and len(b):
                    d.append(float(b.iloc[0][metric]-a.iloc[0][metric]))
            comp[m][metric] = desc(d)
        comp[m]['_per_quarter'] = {}
        for q in QUARTERS:
            a = qdf[(qdf.batch=='A')&(qdf.model==m)&(qdf.quarter==q)]
            b = qdf[(qdf.batch=='B')&(qdf.model==m)&(qdf.quarter==q)]
            if len(a) and len(b):
                comp[m]['_per_quarter'][q] = {k: float(b.iloc[0][k]-a.iloc[0][k]) for k in METRICS}
    # overall across model x quarter
    comp['_overall_quarter'] = {}
    for metric in METRICS:
        vals = [comp[m][metric].get('mean') for m in MODELS if comp[m][metric].get('n')]
        comp['_overall_quarter'][metric] = desc([comp[m][metric]['mean'] for m in MODELS if comp[m][metric].get('n')])
    # fold-level valid and test paired deltas
    for dfname, df in (('valid',vdf),('test',tdf)):
        key = f'_overall_fold_{dfname}'
        comp[key] = {}
        for metric in ('IC_1d','IC_5d','RankIC_1d','RankIC_5d'):
            vals=[]
            for m in MODELS:
                for q in QUARTERS:
                    a = df[(df.batch=='A')&(df.model==m)&(df.quarter==q)].set_index('fold')[metric]
                    b = df[(df.batch=='B')&(df.model==m)&(df.quarter==q)].set_index('fold')[metric]
                    for f in sorted(set(a.index)&set(b.index)):
                        if np.isfinite(a.loc[f]) and np.isfinite(b.loc[f]): vals.append(float(b.loc[f]-a.loc[f]))
            comp[key][metric] = desc(vals)
    # model means by batch (quarter ensemble)
    means = {}
    for b in ('A','B'):
        means[b] = {}
        for m in MODELS:
            sub = qdf[(qdf.batch==b)&(qdf.model==m)&(qdf.quarter.isin(QUARTERS))]
            means[b][m] = {k: float(sub[k].mean()) for k in METRICS} if len(sub) else {}
    comp['_batch_means'] = means
    (RES/'comparison.json').write_text(json.dumps(comp, ensure_ascii=False, indent=1, default=float))

    # summary markdown
    lines = ['# R17 新因子批次验证：260 → 310（25Q3–26Q2，4 模型 × 4 季度 × 4 折）','',
             '批次 A = 当前四个 release 模型训练所用的股票因子集（V230/V240/V252: 260；V220: 146）。',
             '批次 B = A + R17 正式交付的 50 个股票因子（V230/V240/V252: 310；V220: 196）。',
             '同一快照日期 2026-10-16、同一切分、同一随机种子 35253；仅特征集不同。','']
    lines.append('## 1. 四季度 test 综合评价（模型级，四折集成）')
    for m in MODELS:
        lines += ['', f'### {m}', '', '| 批次 | 收益% | IC_1d | IC_5d | RankIC_1d | RankIC_5d |', '|---|---:|---:|---:|---:|---:|']
        for b in ('A','B'):
            sub = qdf[(qdf.batch==b)&(qdf.model==m)&(qdf.quarter.isin(QUARTERS))]
            if len(sub):
                lines.append(f'| {b} | {sub['return'].mean():.4f} | {sub.IC_1d.mean():.4f} | {sub.IC_5d.mean():.4f} | {sub.RankIC_1d.mean():.4f} | {sub.RankIC_5d.mean():.4f} |')
        d = comp[m]
        lines.append(f'| Δ(B−A) | {d["return"].get("mean",float("nan")):+.4f} | {d["IC_1d"].get("mean",float("nan")):+.4f} | {d["IC_5d"].get("mean",float("nan")):+.4f} | {d["RankIC_1d"].get("mean",float("nan")):+.4f} | {d["RankIC_5d"].get("mean",float("nan")):+.4f} |')
        lines.append(f'| 季度为正 | {d["return"].get("positive",0)}/4 | {d["IC_1d"].get("positive",0)}/4 | {d["IC_5d"].get("positive",0)}/4 | {d["RankIC_1d"].get("positive",0)}/4 | {d["RankIC_5d"].get("positive",0)}/4 |')
    lines += ['', '## 2. 分季度 Δ(B−A)', '', '| 模型 | 季度 | Δ收益% | ΔIC_1d | ΔIC_5d | ΔRankIC_1d | ΔRankIC_5d |', '|---|---|---:|---:|---:|---:|---:|']
    for m in MODELS:
        for q in QUARTERS:
            d = comp[m]['_per_quarter'].get(q)
            if d: lines.append(f'| {m} | {q} | {d["return"]:+.4f} | {d["IC_1d"]:+.4f} | {d["IC_5d"]:+.4f} | {d["RankIC_1d"]:+.4f} | {d["RankIC_5d"]:+.4f} |')
    lines += ['', '## 3. 折级配对（16 折/模型，test 重算）', '', '| 指标 | n | Δ均值 | Δ标准差 | 正/负 |', '|---|---:|---:|---:|---:|']
    for k,v in comp['_overall_fold_test'].items():
        lines.append(f'| {k} | {v.get("n",0)} | {v.get("mean",float("nan")):+.5f} | {v.get("sd",float("nan")):.5f} | {v.get("positive",0)}/{v.get("negative",0)} |')
    lines += ['', '## 4. 折级 valid（best epoch）配对', '', '| 指标 | n | Δ均值 | Δ标准差 | 正/负 |', '|---|---:|---:|---:|---:|']
    for k,v in comp['_overall_fold_valid'].items():
        lines.append(f'| {k} | {v.get("n",0)} | {v.get("mean",float("nan")):+.5f} | {v.get("sd",float("nan")):.5f} | {v.get("positive",0)}/{v.get("negative",0)} |')
    (RES/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('wrote', RES/'summary.md')
    print('\n'.join(lines[:60]))

if __name__ == '__main__':
    main()
