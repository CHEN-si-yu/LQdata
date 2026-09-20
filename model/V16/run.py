#!/usr/bin/env python3
"""单元训练入口 —— **每折一进程**（折 = 一次「种子 × 划分」的独立训练）。

    python run.py 1              # 训第 1 折
    python run.py 1 --smoke      # 冒烟（小模型、放宽最小块天数），产物落 smoke/ 树

`train.sh` 负责分批并发地调它；单跑也完全可以（断点续跑就是靠 `logs/fold<N>.log`
里的 `EXIT:0` 标记 + 产物存在性）。

★ `ROOT = parents[0]`（本目录），**不是 `parents[1]`**。

V12/V13/V14 写的是 `parents[1]`（= 模块根）。那样 `import mx` 会命中**顶层的**引擎副本，
而顶层引擎的防分叉闸门（`mx/unit.py:load_unit`）会拒绝从模块根跑自包含单元 ⇒
`bash train.sh` 与 `python run.py 1` 在那三个单元里**全部失败**，只有
`python main.py run <单元>`（走 `.parent`）能跑。上一轮自包含化的验收先把顶层 `mx/`
改名藏起来才通过的，**恰好绕过了这个 bug**。

本目录是单元自己、也是引擎所在，所以这两种写法在这里是**同一个目录** ——
`parents[0]` 在仓内原位跑与打包发出去跑两条路径下都正确，不依赖任何巧合。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # ★ = 单元目录 = 引擎 mx/ 所在
sys.path.insert(0, str(ROOT))

from mx import train, unit as U                      # noqa: E402
from mx.config import load                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="单元 · 单折训练")
    ap.add_argument("fold", type=int, help="折号（从 1 开始；对应 SEEDS 里的第几个种子）")
    ap.add_argument("--smoke", action="store_true", help="冒烟：小模型 + 放宽最小块天数")
    ap.add_argument("--unit", default=None, help="单元名（默认取本目录名）")
    a = ap.parse_args()

    cfg = load()
    name = a.unit or Path(__file__).resolve().parent.name
    u = U.load_unit(cfg, name)
    train.train_fold(cfg, u, a.fold, smoke=a.smoke)
    print(f"EXIT:0 fold{a.fold}")                    # ★ train.sh 靠这一行判定成功
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
