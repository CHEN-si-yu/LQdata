# -*- coding: utf-8 -*-
"""只读对拍：用 release 单元里的 best.pt 重新前向，与落盘的 score_predictions.npy 逐格比。

为什么不直接跑 `model.py --rescore`：
  rescore 会**原地覆盖** releases/<V>/model_train/2026Q3/fold*/ 下的正式产物。
  在做正式增量之前，先证明"重载权重 → 重新前向"这条路是**逐位可复现**的，
  否则一旦新数据进来、对拍不一致，就分不清是"新数据导致的"还是"重载本身不重现"。
  本脚本**只读**，不写任何单元内路径。

为什么要写这个：`README.md §4` 明说权重装载必须 `model/ridge/features` 三件套一起恢复，
  而 `predict()` 是纯函数（对任意日期索引成立）—— 这条正是"增量只扩窗、不重训"的依据。
  这里把它变成可核对的数字。

用法：python verify_rescore.py [V38 ...]
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault('MX_LIVE', '1')          # 必须与产出打分时同一模式，否则切分不同
os.environ.setdefault('MX_THREADS', '4')

RELEASES = Path('/root/autodl-fs/model/releases')


def check(unit):
    root = RELEASES / unit
    sys.path.insert(0, str(root))
    for mod in ('model', 'analysis'):
        if mod in sys.modules:
            del sys.modules[mod]
    M = importlib.import_module('model')
    ck = __import__('torch').load(root / 'model_train' / M.LIVE_QUARTER / 'fold1' / 'best.pt',
                                  map_location='cpu', weights_only=False)
    panel = M.Panel(M.RECIPE['features'])       # 与训练同一条列集口径
    prices = M.Prices(panel)
    ev = M.splits(panel.days, quarter=M.LIVE_QUARTER)[0]['score']
    model = M.PredictModel(len(panel.features), panel.market_dim)
    model.load_state_dict(ck['model'])
    model.ridge = ck.get('ridge')               # ★ 线性分量（20%），漏了会静默给错分
    model.eval()
    cols_ok = list(ck['features']) == list(panel.features)
    pred = M.predict(model, panel, prices, ev, __import__('torch').device('cpu'))
    stored = np.load(root / 'model_train' / M.LIVE_QUARTER / 'fold1' / 'score_predictions.npy')
    both = np.isfinite(pred) & np.isfinite(stored)
    d = np.abs(pred[both] - stored[both])
    nan_mismatch = int((np.isfinite(pred) != np.isfinite(stored)).sum())
    sys.path.remove(str(root))
    print(f'[{unit}] 列名顺序与 best.pt 一致 = {cols_ok}；推演窗 {pred.shape} vs 落盘 {stored.shape}')
    print(f'        逐格最大绝对差 = {d.max():.3e}；非有限值错配 = {nan_mismatch}')
    return dict(unit=unit, cols_ok=bool(cols_ok), shape=list(map(int, pred.shape)),
                max_abs_diff=float(d.max()), nan_mismatch=nan_mismatch)


if __name__ == '__main__':
    units = sys.argv[1:] or ['V31', 'V36', 'V38']
    for u in units:
        check(u)
