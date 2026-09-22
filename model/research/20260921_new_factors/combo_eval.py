"""打分组合实验：把不同特征集/不同种子的模型分数**逐日截面秩标准化**后线性组合。

动机：V63（653 列）IC 更高但 top1 崩了（Σtop1 −1.115），而 R30 的结论是
「头部质量随模型份数单调改善」—— 那把两份打分合起来，能不能拿到「IC 高 + top1 不塌」？

口径：
  · 每一份打分**逐日**做截面秩标准化（`(rank−.5)/n − .5`）⇒ 两份尺度可比；
  · 组合 = 各份的加权平均；权重网格在跑之前写死（见 --weights），不按结果挑；
  · 评价 = 同一套 `evaluation_core.cash_backtest`（含费、T+1、整手、1% 容量）+ 固定 60 日市场门槛版；
  · 不训练任何东西，只读已有 `test_predictions.npy`。

用法：python combo_eval.py --versions V62 V63 --out combo_v62_v63
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import evaluation_core as E
from strategy_research import quarter_cash

MODEL = Path('/autodl-fs/data/model')
QUARTERS = ['2025Q3', '2025Q4', '2026Q1', '2026Q2']


def gate_signal(adjusted):
    """固定 60 日市场趋势门槛：等权复权收盘指数 ≥ 过去 60 交易日均值。与 market_gate_research 同口径。"""
    ret = adjusted[1:] / adjusted[:-1] - 1
    good = np.isfinite(ret)
    count = good.sum(axis=1)
    mr = np.r_[0., np.where(good, ret, 0.).sum(axis=1) / count]
    idx = 100 * np.cumprod(1 + mr)
    ma = pd.Series(idx).rolling(60, min_periods=60).mean().to_numpy()
    return idx, idx >= ma


def folds_of(version):
    """四折 × 四季度 → {fold: pred}；★ 每折的 `test_dates` 只是**该季度**的切片，
    把四个季度按顺序接起来才是 242 天的评价窗。"""
    out = {}
    for f in range(1, 5):
        chunks, dates = [], []
        for q in QUARTERS:
            folder = MODEL / version / 'model_train' / q / f'fold{f}'
            done = json.loads((folder / 'complete.json').read_text())
            p = np.load(folder / 'test_predictions.npy')
            assert p.shape == (len(done['test_dates']), 2115), (version, q, f, p.shape)
            chunks.append(p)
            dates.extend(done['test_dates'])
        assert len(dates) == 242 and dates[0] == '2025-07-01' and dates[-1] == '2026-06-30', (version, f, len(dates))
        out[f] = (dates, np.concatenate(chunks))
    return out


def scale_to(x, ref):
    """把 x 逐日截面标准差缩放到与 ref 相同 —— **线性**缩放，不改 x 当天的排名。

    ★ 为什么不是"秩标准化后再平均"：秩化会把每个源的分布压成同一个形状，
      于是 w=0 的端点**不再等于该版本自己的集成打分**（实测 +207% vs 官方 +160.57%，
      见 2026-09-21 记录）。改用"标准差对齐后线性相加"后，w=0 逐位复现官方口径、
      w=1 与 B 的排名逐位相同 —— 两个端点都可对账，中间才是真正的插值。
    """
    out = np.full(x.shape, np.nan, np.float32)
    for i in range(x.shape[0]):
        a, b = ref[i], x[i]
        ma, mb = np.isfinite(a), np.isfinite(b)
        if ma.sum() < 3 or mb.sum() < 3:
            continue
        sa, sb = np.std(a[ma]), np.std(b[mb])
        if sb <= 0:
            continue
        out[i, mb] = b[mb] * (sa / sb)
    return out


def ics(score, Y, days):
    out = []
    for i, d in enumerate(days):
        y = Y[d]
        m = np.isfinite(score[i]) & np.isfinite(y)
        out.append(np.corrcoef(rankdata(score[i][m]), rankdata(y[m]))[0, 1] if m.sum() > 50 else np.nan)
    return float(np.nanmean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--versions', nargs='+', required=True, help='第一份是基准（权重 1−w），其余等权分摊 w')
    ap.add_argument('--weights', nargs='+', type=float, default=[0., .25, .5, .75, 1.])
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--equal', action='store_true',
                    help='★ 不做 w 网格：把列出的**所有**版本等权（每天各自按截面标准差对齐后平均）。'
                         '用途：R30 的"加模型"曲线在**种子维**上从没在打分层面测过 —— 那里只测过资金层面的 3 模型组合。')
    args = ap.parse_args()

    out = MODEL / 'research/20260921_new_factors' / args.out
    out.mkdir(parents=True, exist_ok=True)

    panel = E.Panel(load_x=False)
    px = E.Prices(panel)
    _, RISK_ON = gate_signal(px.close * px.adj)

    A = args.versions[0]
    fa = folds_of(A)
    dates = fa[1][0]
    days = np.searchsorted(panel.days, dates)
    assert dates[0] == '2025-07-01' and dates[-1] == '2026-06-30' and len(dates) == 242

    raw = {A: {f: p for f, (_, p) in fa.items()}}
    for v in args.versions[1:]:
        fv = folds_of(v)
        assert fv[1][0] == dates, f'{v} 的日期与 {A} 不一致'
        raw[v] = {f: p for f, (_, p) in fv.items()}

    # 各版本自己的集成分数（四折相加）只算一次
    ens_of = {v: sum(raw[v].values()).astype(np.float32) for v in args.versions}
    base = ens_of[A]
    # 其余版本逐日缩放到 A 的截面离散度（线性缩放，不改当天排名）
    scaled = {v: (base if v == A else scale_to(ens_of[v], base)) for v in args.versions}
    others = np.zeros_like(base)
    for v in args.versions[1:]:
        others = others + scaled[v] / (len(args.versions) - 1)

    grid = [1.0] if args.equal else args.weights
    rows = []
    for w in grid:
        ens = (sum(scaled.values()) / len(args.versions)).astype(np.float32) if args.equal \
            else ((1 - w) * base + w * others).astype(np.float32)
        for tag, mixed in (('plain', False), ('gate', True)):
            pred = ens.copy()
            if mixed:
                pred[~RISK_ON[days]] = np.nan
            for n in (1, 5):
                stat, curve, trades = E.stress_cash_backtest(pred, panel, px, days, n=n, slippage=0.0003)
                eq = np.r_[100000., [x['equity'] for x in curve]]
                ret = eq[1:] / eq[:-1] - 1
                qs = quarter_cash(curve, days, panel)
                rows.append(dict(w=w, mode=tag, topn=n,
                                 ic_1d=ics(ens, panel.Y['label_ret_1d'], days),
                                 ic_5d=ics(ens, panel.Y['label_ret_5d'], days),
                                 net=stat['return_value'], maxdd=stat['max_drawdown'], sharpe=stat['sharpe'],
                                 fees=stat['total_fees'], trades=stat['trades'],
                                 remove_best5=float(np.prod(1 + np.sort(ret)[:-5]) - 1),
                                 H1=float(np.prod([1 + q['net'] for q in qs[:2]]) - 1),
                                 H2=float(np.prod([1 + q['net'] for q in qs[2:]]) - 1)))
                print(f'w={w:<5} {tag:<5} top{n}: 净收益 {stat["return_value"]:+.2%} '
                      f'回撤 {stat["max_drawdown"]:.2%} 费用 {stat["total_fees"]:,.0f}', flush=True)
                if w in (0., 1.) and n == 1:
                    pd.DataFrame(curve).to_csv(out / f'curve_w{w}_{tag}_top{n}.csv', index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'combo_results.csv', index=False)
    E.atomic_json(out / 'plan.json', dict(
        versions=args.versions, weights=args.weights, tag=args.tag,
        standardize='per-day cross-sectional rank standardization per source, then weighted mean',
        note='权重网格在结果产生前写死；w=0.5 为主口径，其余只作单调性诊断（同 R34 的读法）。',
        window=['2025-07-01', '2026-06-30'], panel_digest=panel.meta['panel_digest']))
    pd.set_option('display.width', 200)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
