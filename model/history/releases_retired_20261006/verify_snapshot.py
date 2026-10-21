# -*- coding: utf-8 -*-
"""快照增量后的**一致性判据**：对齐上游、只许前向扩一天、列集一字不许动。

为什么单列一条判据：`releases/` 三个单元的特征列集是**在加载时**由
`meta['columns']['features']` 减去各自的排除表算出来的（`model.py:feature_columns`），
不是写在配置里的常量。所以**上游增删一列，模型的输入维度就跟着变**：

  · 列数变了 → `load_state_dict` 形状不符，报错（好）；
  · 列数没变但**名字或顺序变了** → 不报错，静默错位（坏，真危险）。

这条判据把上面两种都变成一句话可读的结论，并且证明增量是**纯前向追加**：
旧日期的行不许动，`axis.end` 只许往前走，`n_codes` 必须不变
（股票轴一变，落盘的 (天, 股票) 打分矩阵就对不上了）。

用法：python verify_snapshot.py <meta_before.json> <meta_after.json>
"""
import json
import sys


def load(p):
    return json.load(open(p, encoding='utf-8'))


def main(before_path, after_path):
    b, a = load(before_path), load(after_path)
    rows, bad = [], 0

    def rec(ok, name, detail):
        nonlocal bad
        bad += 0 if ok else 1
        rows.append((ok, name, detail))

    fa, fb = list(a['columns']['features']), list(b['columns']['features'])
    rec(fb == fa, '特征列集逐位相同',
        f'{len(fb)} → {len(fa)}' + ('' if fb == fa else
                                    f'；仅旧有={sorted(set(fb) - set(fa))[:6]} 仅新有={sorted(set(fa) - set(fb))[:6]}'))
    rec(list(a['columns']['direction']) == list(b['columns']['direction']), '方向元数据键序相同',
        f'{len(b["columns"]["direction"])} → {len(a["columns"]["direction"])}')
    # ★ 只比键序是不够的：值从 +1 翻成 −1 意味着**这一列的值被整体取反**，
    #   而列名和列序都没变 ⇒ load_state_dict 不报错、静默吃错列。
    #   2026-09-28 实测 `close30_giveback` 就是 1 → −1（上游改了 higher_is_better）。
    db, da = b['columns']['direction'], a['columns']['direction']
    flips = [k for k in db if k in da and db[k] != da[k]]
    rec(not flips, '方向**取值**无翻转',
        '；'.join(f'{k}: {db[k]}→{da[k]}' for k in flips[:5]) or 'ok')

    bx, ax = b['axis'], a['axis']
    rec(ax['n_codes'] == bx['n_codes'], '股票轴长度不变', f'{bx["n_codes"]} → {ax["n_codes"]}')
    rec(ax['codes'][:bx['n_codes']] == bx['codes'], '股票轴逐位不变（前 n 只）',
        'ok' if ax['codes'][:bx['n_codes']] == bx['codes'] else '★ 顺序变了，落盘打分矩阵不可比')

    nb, na = int(bx['n_days']), int(ax['n_days'])
    rec(na >= nb, '样本轴只增不减', f'{bx["end"]} → {ax["end"]}（{nb} → {na} 天）')
    rec(bx['start'] == ax['start'], '样本轴起点不变', f'{bx["start"]} → {ax["start"]}')

    # 旧年份的分区统计必须一个字节都不变 —— 增量只该重算尾部窗口。
    # ★ 只豁免**最末一年**：新的一天只会落进它，它的行数/字节本就该长。
    #   往前任何一年动一个字节，都说明增量回写了历史分区（那是真事故，不是噪声）。
    tail = str(max(b.get('built_years', [0])))
    moved = []
    for y in sorted(set(b.get('years', {})) & set(a.get('years', {}))):
        if y == tail:
            continue
        for blk, v in b['years'][y].get('files', {}).items():
            va = a['years'][y].get('files', {}).get(blk)
            if va is None:
                moved.append(f'{y}/{blk} 消失')
            elif (va['rows'], va['bytes']) != (v['rows'], v['bytes']):
                moved.append(f'{y}/{blk} 行/字节变了 {v["rows"]}→{va["rows"]} / {v["bytes"]}→{va["bytes"]}')
    rec(not moved, f'{tail} 以前的年份行数/字节未变（未回写旧分区）',
        '；'.join(moved[:4]) or 'ok')

    rec(len(a['years']) >= len(b['years']), '年份分区未减少',
        f'{len(b["years"])} → {len(a["years"])}')
    rec(a['semantics'] == b['semantics'], '值口径语义标识不变',
        f'{b["semantics"]} → {a["semantics"]}')

    print(f'{"判据":<34}{"结果":<6}明细')
    print('-' * 100)
    for ok, name, detail in rows:
        print(f'{name:<34}{"✔" if ok else "✘":<6}{detail}')
    print('-' * 100)
    print(f'共 {len(rows)} 条，失败 {bad} 条')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
