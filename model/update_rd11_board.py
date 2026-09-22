"""重新生成 `LEADERBOARD_RD11.md` 的主榜（2026-09-22）。

用户 2026-09-22：「以后每次完成训练后需要更新相关的文档记录」。
手工抄数会出错，所以主榜**由本脚本从各单元的 `strategy_summary.csv` 生成**，
只替换 `<!-- RD11_MAIN_TABLE_START -->` / `<!-- RD11_MAIN_TABLE_END -->` 之间的内容，
**不碰文件里的散文**。

判定规则（预登记，写死在脚本里，不允许按观察结果改）：
  · 进主榜 = **三粒种子跑齐 且 三粒全部为正**；按**最差净收益**降序。
  · 单种子格不进主榜，只在末尾单列。
  · 取 `source == 'ensemble'`（★ 用户 2026-09-20：4Fold 集成才是这个模型）。

用法：
    $PY update_rd11_board.py            # 重写主榜表格
    $PY update_rd11_board.py --check    # 只打印，不写文件
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path('/autodl-fs/data/model')
BOARD = ROOT / 'LEADERBOARD_RD11.md'
START, END = '<!-- RD11_MAIN_TABLE_START', '<!-- RD11_MAIN_TABLE_END -->'

#: 注册的 RD11 单元：单元名 -> 臂标签。种子与特征列从各自的 model.py 读，不写第二遍。
UNITS = ['V76', 'V77', 'V78', 'V79', 'V80', 'V81']
ARM = {'V76': '旧核', 'V78': '旧核', 'V79': '旧核',
       'V77': '全量', 'V80': '全量', 'V81': '全量'}

#: 主榜只收这些策略（预登记格）；其余策略留在各单元自己的 strategy_summary.csv 里
BOARD_STRATEGIES = [
    'top1_1d',
    're400_t5_h5', 're500_t5_h5', 're600_t5_h5', 're700_t5_h5',
    're800_t5_h5', 're1000_t5_h5',
    'g_re400_t5_h5', 'g_re500_t5_h5', 'g_re600_t5_h5', 'g_re700_t5_h5',
    'g_re800_t5_h5', 'g_re1000_t5_h5',
    're500_t5_h5_tp20', 're500_t5_h5_sl8',
    'g_re500_t5_h5_tp20', 'g_re500_t5_h5_sl8', 'g_re500_t5_h5_sl8_tp20',
    'g_re700_t5_h5_sl8_tp20',
]


def unit_meta(unit: str):
    """从单元自己的 model.py 读 (seed, 特征列数) —— 不从别处抄第二遍（LESSONS §11-26）。"""
    src = (ROOT / unit / 'model.py').read_text(encoding='utf-8')
    seed = int(re.search(r'^\s*seed=(\d+),', src, re.M).group(1))
    feat_file = re.search(r"features_arm[AB]_\w+\.json", src).group(0)
    ncol = len(json.loads((ROOT / unit / feat_file).read_text(encoding='utf-8')))
    return seed, ncol


def load():
    rows = []
    for u in UNITS:
        p = ROOT / u / 'model_pred' / 'tables' / 'strategy_summary.csv'
        if not p.exists():
            continue
        seed, ncol = unit_meta(u)
        d = pd.read_csv(p)
        d = d[(d.source == 'ensemble') & (d.strategy.isin(BOARD_STRATEGIES))]
        for _, r in d.iterrows():
            rows.append(dict(unit=u, arm=ARM.get(u, '?'), ncol=ncol, seed=seed,
                             strategy=r.strategy, ret=float(r.return_value),
                             dd=float(r.max_drawdown), fee=float(r.total_fees),
                             trades=int(r.trades)))
    return pd.DataFrame(rows)


def render(df: pd.DataFrame) -> str:
    if df.empty:
        return '| — | — | — | — | *（还没有任何单元产出）* | — | — | — | — | — | — | — | — | 0/3 |\n'

    piv = df.pivot_table(index=['arm', 'ncol', 'strategy'], columns='seed', values='ret')
    ddp = df.pivot_table(index=['arm', 'ncol', 'strategy'], columns='seed', values='dd')
    fep = df.pivot_table(index=['arm', 'ncol', 'strategy'], columns='seed', values='fee')
    trp = df.pivot_table(index=['arm', 'ncol', 'strategy'], columns='seed', values='trades')
    seed_cols = sorted(piv.columns)

    agg = pd.DataFrame({
        'seeds': piv.notna().sum(axis=1),
        'worst': piv.min(axis=1),
        'med': piv.median(axis=1),
        'spread': (piv.max(axis=1) - piv.min(axis=1)) * 100,
        'allpos': (piv > 0).all(axis=1),
        'worst_dd': ddp.max(axis=1),
        'trades': trp.median(axis=1),
        'fee': fep.median(axis=1),
    })

    # 进榜 = 三粒跑齐 且 三粒全正；其余单列
    full = agg[(agg.seeds == 3) & agg.allpos].sort_values('worst', ascending=False)
    partial = agg[(agg.seeds != 3) | ~agg.allpos]

    def name(arm, ncol, strat):
        gate = '✔' if strat.startswith('g_') else '✘'
        risk = []
        m = re.search(r'_sl(\d+)', strat)
        if m:
            risk.append(f"止损{m.group(1)}%")
        m = re.search(r'_tp(\d+)', strat)
        if m:
            risk.append(f"止盈{m.group(1)}%")
        return arm, ncol, strat, gate, '+'.join(risk) if risk else '—'

    lines = ['| 排名 | 臂 | 特征列 | 策略 | 门槛 | 止损/止盈 | **最差净收益** | 中位 | 极差 | 最差回撤 | 笔数 | 费用/元 | 种子数 |',
             '|--:|:--|--:|:--|:--:|:--|--:|--:|--:|--:|--:|--:|--:|']
    for i, (key, r) in enumerate(full.iterrows(), 1):
        arm, ncol, strat, gate, risk = name(*key)
        lines.append(f'| **{i}** | {arm} | {ncol} | `{strat}` | {gate} | {risk} | '
                     f'**{r.worst * 100:+.2f}%** | {r.med * 100:+.2f}% | {r.spread:.1f}pp | '
                     f'{r.worst_dd * 100:.2f}% | {int(r.trades)} | {int(r.fee):,} | {int(r.seeds)}/3 |')
    for key, r in partial.iterrows():
        arm, ncol, strat, gate, risk = name(*key)
        note = '单种子待补' if r.seeds != 3 else '**有种子为负 ⇒ 不进榜**'
        lines.append(f'| — | {arm} | {ncol} | `{strat}` | {gate} | {risk} | '
                     f'{r.worst * 100:+.2f}% | {r.med * 100:+.2f}% | {r.spread:.1f}pp | '
                     f'{r.worst_dd * 100:.2f}% | {int(r.trades)} | {int(r.fee):,} | {int(r.seeds)}/3 · {note} |')

    lines += ['', f'*（本表由 `update_rd11_board.py` 生成 · 单元 {sorted(df.unit.unique())} · '
                  f'种子 {seed_cols} · ensemble 打分）*']
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='只打印，不写文件')
    a = ap.parse_args()

    df = load()
    table = render(df)

    if a.check:
        print(table)
        return 0

    txt = BOARD.read_text(encoding='utf-8')
    i, j = txt.find(START), txt.find(END)
    if i < 0 or j < 0:
        print(f'✘ 找不到标记 {START} / {END}', file=sys.stderr)
        return 1
    head = txt[:txt.index('\n', i) + 1]           # 保留 START 那一整行
    BOARD.write_text(head + table + txt[j:], encoding='utf-8')
    print(f'✔ 已重写 {BOARD.name} 主榜（{len(df)} 条读数 · {df.unit.nunique()} 个单元）')
    print(table)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
