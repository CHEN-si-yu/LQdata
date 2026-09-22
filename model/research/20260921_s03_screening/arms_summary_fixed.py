"""全部实验臂的总表（**H2 口径修正版**，评审建议 S-03）。

## 与 `../20260921_new_factors/arms_summary.py` 的唯一区别

原脚本第 33-34 行用**现金曲线的行中点**当"后半程"：

    eq = pd.read_csv(curve)['equity'].to_numpy()
    H2 = eq[-1] / eq[len(eq) // 2] - 1

V63 的曲线 243 行、首 2025-07-02、末 2026-07-02 ⇒ `len//2 = 121`，除数是 **2025-12-26**，
分子还含 2026-07-01/02（2026Q2 的信号跨季执行 + 末尾清仓）。
而 REPORT 自己定义的口径是"**按发出交易指令的信号日归属**"，
`market_gate_research.py:73` 也是这么算的。

⇒ 同一张 `arms_summary.csv` 里"top1 后半程"（行中点法）与"门槛 H2"（信号季度法）
**窗口不同**，实例 V62 两列为 +9.25% vs +30.77%，差 **21.5pp**。

## 本版怎么做

复用平台**现有的**季度归属函数 `strategy_research.quarter_cash()` ——
它把现金曲线的每个点按"信号所属季度"归组，并自带两条断言
（`cursor==len(curve)`、各季度净值的连乘 == 全期净值/10 万）。
**不重跑任何回测**，只读已落盘的 `cash_<策略>.csv`。

    H1 = ∏(1+q.net) over quarters[:2]
    H2 = ∏(1+q.net) over quarters[2:]      ← 与 REPORT / market_gate 口径一致

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY arms_summary_fixed.py --check V62 V63 V66     # 只打印两种口径的差异
    $PY arms_summary_fixed.py V62 V63 V64 V65 V66      # 写 arms_summary_fixed.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parents[1]          # = /autodl-fs/data/model
sys.path.insert(0, str(ROOT))    # 让 `import evaluation_core` 生效（本目录已放副本）
import evaluation_core as E       # noqa: E402
from strategy_research import quarter_cash  # noqa: E402

EVAL_START, EVAL_END = '2025-07-01', '2026-06-30'
STRATS = ('top1_1d', 'top5_1d')

DESC = {
    'V32': '旧快照 337 列（基线）', 'V62': '337 列（对照，=V32）', 'V63': '全部 653 列',
    'V64': '337+mfx28+efx32=397', 'V65': '337+按|IC|前34=371', 'V66': '337+随机40=377',
    'V67': '只用新因子 316 列', 'V34': '旧快照 337 列 seed3254', 'V36': '旧快照 337 列 seed3255',
    'V69': 'V66 换种子 3254', 'V70': 'V66 换种子 3255', 'V72': '种子扩容 3301',
    'V73': '种子扩容 3302',
}

_PANEL = None


def panel():
    """只加载坐标轴（`load_x=False`），不读特征面板 —— 本脚本不需要 X。"""
    global _PANEL
    if _PANEL is None:
        _PANEL = E.Panel(load_x=False)
    return _PANEL


def eval_signal_days(p) -> list[int]:
    return [i for i, d in enumerate(p.days) if EVAL_START <= d <= EVAL_END]


def midpoint_H2(curve_df: pd.DataFrame) -> float:
    """原脚本的口径（行中点）—— 保留它，只用于对照。"""
    eq = curve_df['equity'].to_numpy()
    return float(eq[-1] / eq[len(eq) // 2] - 1)


def quarter_H2(curve_df: pd.DataFrame, p, days) -> tuple[float, list[dict]]:
    """修正口径：按**信号季度**归属，返回 (H2, 各季度明细)。

    `quarter_cash` 自带断言，所以这里的"对齐"不是靠我口算的日期 —— 它会在
    季度净值连乘 != 全期净值/10 万时直接抛错。
    """
    curve = curve_df.to_dict('records')
    qs = quarter_cash(curve, days, p)
    h1 = float(np.prod([1 + x['net'] for x in qs[:2]]) - 1)
    h2 = float(np.prod([1 + x['net'] for x in qs[2:]]) - 1)
    return h2, qs


def row(ver: str, p, days, check: bool = False) -> dict:
    d = MODEL / ver
    out: dict = {'ver': ver, '说明': DESC.get(ver, '')}
    strat = d / 'model_pred/tables/strategy_summary.csv'
    if strat.exists():
        e = pd.read_csv(strat)
        e = e[e.source == 'ensemble']
        for s in STRATS:
            m = e[e.strategy == s]
            if len(m):
                g = m.iloc[0]
                out[f'{s}_净收益'] = float(g.return_value)
                out[f'{s}_回撤'] = float(g.max_drawdown)
    main = d / 'model_pred/tables/main_metrics_1d.csv'
    if main.exists():
        tot = pd.read_csv(main)
        tot = tot[tot['quarter'] == '总计']
        if len(tot):
            out['IC'] = float(tot.iloc[0].IC)
            out['Σtop1'] = float(tot.iloc[0].top_return)
    for s in STRATS:
        cv = d / 'model_pred/ensemble' / f'cash_{s}.csv'
        if not cv.exists():
            continue
        cdf = pd.read_csv(cv)
        out[f'{s}_后半程_行中点'] = midpoint_H2(cdf)
        h2, qs = quarter_H2(cdf, p, days)
        out[f'{s}_后半程_信号季度'] = h2
        if check:
            print(f'  {ver} {s}: 行中点={out[f"{s}_后半程_行中点"]:+.4f}  '
                  f'信号季度={h2:+.4f}  Δ={h2 - out[f"{s}_后半程_行中点"]:+.4f}')
            for q in qs:
                print(f'      {q["quarter"]}: net={q["net"]:+.4f} maxdd={q["maxdd"]:+.4f}')
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('vers', nargs='*', default=['V62', 'V63', 'V64', 'V65', 'V66'])
    ap.add_argument('--check', action='store_true', help='只打印两种口径差异，不写 CSV')
    ap.add_argument('--out', default='arms_summary_fixed.csv')
    a = ap.parse_args()

    p = panel()
    days = eval_signal_days(p)
    print(f'评价窗信号日 {len(days)} 天（{p.days[days[0]]} ~ {p.days[days[-1]]}）')
    rows = [row(v, p, days, check=a.check) for v in a.vers]
    df = pd.DataFrame(rows)
    if a.check:
        print('\n（--check：未写文件）')
        return 0
    out = ROOT / a.out
    df.to_csv(out, index=False)
    (ROOT / 'arms_summary_fixed_meta.json').write_text(json.dumps(
        dict(eval_window=[EVAL_START, EVAL_END], n_signal_days=len(days),
             h2_semantics='按信号季度归属（quarter_cash），非现金曲线行中点',
             source='fixed copy of ../20260921_new_factors/arms_summary.py'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    pd.set_option('display.width', 250)
    cols = [c for c in df.columns if 'net' in c or 'IC' in c or '回撤' in c or '后半程' in c or c in ('ver', '说明')]
    print(df[cols].to_string(index=False, float_format=lambda x: f'{x:,.4f}'))
    print(f'\n已写 {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
