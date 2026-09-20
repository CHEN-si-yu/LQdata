#!/usr/bin/env python3
"""单元分析入口 —— 推演 → 榜单 → 回测 → 落盘单元打分。

    python analysis.py                 # 全部：装配 + 评价 + 榜单 + 回测 + 落盘
    python analysis.py --no-backtest   # 只出榜单
    python analysis.py --smoke         # 分析冒烟产物（smoke/ 树）

产物：
    pred/score__<单元>__<标签>/        本单元的组合打分（4 列契约，best 晋级时取它）
    state/leaderboard.json             全指标 + 榜单 + 回测结果

★ `ROOT = parents[0]`（本目录），不是 `parents[1]` —— 理由见 `run.py` 顶部那段。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # ★ = 单元目录 = 引擎 mx/ 所在
sys.path.insert(0, str(ROOT))

from mx import analyze, backtest, unit as U            # noqa: E402
from mx.config import load                             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="单元 · 分析与回测")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--top", type=int, default=None, help="回测持仓数（默认 conf 的 backtest.top_n）")
    ap.add_argument("--unit", default=None)
    a = ap.parse_args()

    cfg = load()
    u = U.load_unit(cfg, a.unit or Path(__file__).resolve().parent.name)
    s = analyze.run(cfg, u, smoke=a.smoke, top_n=a.top, do_backtest=not a.no_backtest)
    print(analyze.render(s))
    for k, bt in (s.get("backtest") or {}).items():
        print(f"\n  【回测 · {k} · 头={bt.get('head')}】")
        print(backtest.render(bt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
