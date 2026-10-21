# -*- coding: utf-8 -*-
"""增量后对 `releases/` 打分的**一致性判据**：新打分必须是旧打分的**纯前向追加**。

为什么这条判据成立（不是"大概应该一致"，是可证的）：
  `predict()` 的四步 —— `predict_heads`（逐日前向）、`score_of`（**逐日**截面秩）、
  `blend_linear`（**逐日**仿射定标）、`hot_filter`（逐日热门池）——
  **每一步都只在当日截面上运算**（`model.py:1150` / `1179` / `1238` / `1321`），
  而 BN 统计量冻结在 `best.pt` 里、重推演时不重估。
  ⇒ 第 i 天的打分只是第 i 天截面与权重的函数，**与推演窗有多长无关**。
  ⇒ 追加新的一天只能**多出一行**，不能改动已算出的任何一行。
  这正是"增量只扩窗、不必重训"的依据；本脚本把它变成可核对的数字。

判据（任一不过就返回 1）：
  ① 行数 = 旧行数 + 新增交易日数，列数不变（股票轴没动）；
  ② 旧行**逐格**一致（阈值 2e-4，与 `analysis.py --live` 的重载对拍同一量级）；
  ③ 可打分图样（NaN 位置）逐格一致 —— NaN 错位会让"哪些票能买"静默改变；
  ④ `best.pt` 字节未变 —— 本次是重推演，不是重训。

用法：python verify_extension.py [新增交易日数=1]
"""
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

RELEASES = pathlib.Path('/root/autodl-fs/model/releases')
#: 基线目录（含 release_artifacts_before.json 与 old_scores/），由命令行第 2 个参数给出。
#: 它属于**某一次增量**的档案，不是脚本的固定依赖 —— 放在那次增量的工作目录里。
BASE = pathlib.Path(os.environ.get('MX_BASELINE', '.'))
QUARTER = '2026Q3'
TOL = 2e-4


def sha16(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main(new_days=1):
    before = json.loads((BASE / 'release_artifacts_before.json').read_text(encoding='utf-8'))
    bad = 0
    print(f'{"单元/折":<12}{"形状 旧→新":<20}{"旧行最大差":<14}{"NaN 错位":<10}{"best.pt":<10}')
    print('-' * 76)
    for u in ('V31', 'V36', 'V38'):
        for k in (1, 2, 3, 4):
            d = RELEASES / u / 'model_train' / QUARTER / f'fold{k}'
            old = np.load(BASE / 'old_scores' / f'{u}_fold{k}.npy')
            new = np.load(d / 'score_predictions.npy')
            shape_ok = (new.shape[0] == old.shape[0] + new_days and new.shape[1] == old.shape[1])
            n = old.shape[0]
            a, b = new[:n], old
            both = np.isfinite(a) & np.isfinite(b)
            with np.errstate(invalid='ignore'):
                diff = np.abs(a[both] - b[both])
            dmax = float(diff.max()) if diff.size else 0.0
            nan_shift = int((np.isfinite(a) != np.isfinite(b)).sum())
            bp_ok = sha16(d / 'best.pt') == before[f'{u}/fold{k}/best.pt']['sha16']
            ok = shape_ok and dmax <= TOL and nan_shift == 0 and bp_ok
            bad += 0 if ok else 1
            print(f'{u}/fold{k:<6}{f"{old.shape} → {new.shape}":<20}{dmax:<14.3e}{nan_shift:<10}'
                  f'{"✔ 未变" if bp_ok else "✘ 变了":<10}{"" if ok else "  ★ 不符"}')
    print('-' * 76)
    print(f'共 12 折，不符 {bad} 折；逐格阈值 {TOL:g}')
    return 1 if bad else 0


if __name__ == '__main__':
    # 用法：verify_extension.py [新增交易日数] [基线目录]
    if len(sys.argv) > 2:
        BASE = pathlib.Path(sys.argv[2])
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1))
