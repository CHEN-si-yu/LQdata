#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""545 格复权因子异常 **与冻结池 × 输出窗口求交**（S-01，2026-09-21）。

## 为什么要有这个脚本

`state/known_adj_anomalies.json` 里 545 格的定性结论是**全历史、全市场**口径的
（含 2010~2017、含创业板/科创板）。但因子产物只覆盖
**冻结池 2115 只 × 2018-01-01 起**，所以"545 格"里真正能影响下游结论的只是一小撮。
`HISTORY.md` 里写的"主板只有 17 格 / 13 只股"**用的是主板前缀口径、且没有限定输出窗口** ——
与本文的冻结池口径对不上。这个脚本把口径固定下来，避免以后再各说各话。

## 三个口径别混（这是最容易出错的地方）

    main_board（基线文件里的字段）  主板前缀（600/601/603/605/000/001/002/003）
                                     —— 宽，包含已退市/曾 ST 的票
    冻结池                          conf/universe_frozen.tsv，2115 只
                                     —— 窄：主板 ∩ 至今存续 ∩ 从未ST ∩ 上市≤2018-01-01
    输出窗口                        >= conf/config.yaml 的 default_start（2018-01-01）

**只有「冻结池 ∩ 输出窗口」这一格会真正进入下场结论。**

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/adj_anomaly_pool_intersect.py            # 打印
    $PY scripts/adj_anomaly_pool_intersect.py --out-dir artifacts/audits/adj_anomalies_<日期>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_START = "2018-01-01"


def load_pool() -> list[str]:
    tsv = ROOT / "conf" / "universe_frozen.tsv"
    return [ln.split("\t")[0].strip() for ln in tsv.read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(ROOT / "state" / "known_adj_anomalies.json"))
    ap.add_argument("--start", default=DEFAULT_START, help="输出窗口下界（默认 2018-01-01）")
    ap.add_argument("--out-dir", default="", help="可选：把这些结果写成报告目录")
    a = ap.parse_args()

    d = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
    an = d["anomalies"]
    pool = set(load_pool())

    in_pool = [x for x in an if x["stock_code"] in pool]
    in_win = [x for x in in_pool if x["date"] >= a.start]

    lines = []
    add = lines.append
    add("# 545 格复权因子异常 × 冻结池 × 输出窗口（求交结果）\n")
    add(f"- 生成时间：{d.get('created_at')}（基线）· 本报告现算")
    add(f"- 基线文件：`{a.baseline}`（共 {len(an)} 格，全历史 {d.get('scope')}）")
    add(f"- 冻结池：{len(pool)} 只 · 输出窗口：>= {a.start}\n")
    add("## 三个口径的计数（别混用）\n")
    add("| 口径 | 格数 | 股票数 | 说明 |")
    add("|:--|--:|--:|:--|")
    add(f"| 全历史全市场 | {len(an)} | {len({x['stock_code'] for x in an})} | 基线文件本身 |")
    add(f"| 主板前缀（基线 `main_board` 字段） | "
        f"{sum(1 for x in an if x.get('main_board'))} | "
        f"{len({x['stock_code'] for x in an if x.get('main_board')})} | 宽口径，含已退市/曾ST |")
    add(f"| ∩ 冻结池 | {len(in_pool)} | {len({x['stock_code'] for x in in_pool})} | 2115 只池内 |")
    add(f"| **∩ 冻结池 ∩ 输出窗口** | **{len(in_win)}** | "
        f"**{len({x['stock_code'] for x in in_win})}** | ★ **真正影响下游结论的** |")
    add("")
    add("> `HISTORY.md` 里「主板只有 17 格 / 13 只股」与上表任一格都对不上 —— "
        "它用的是主板前缀、且未限定输出窗口。以本表为准。\n")
    add("## ∩ 冻结池 ∩ 输出窗口 的逐格明细\n")
    add("| 股票 | 日期 | 机制 | hfq − exch |")
    add("|:--|:--|:--|--:|")
    for x in sorted(in_win, key=lambda v: (v["date"], v["stock_code"])):
        hv = x.get("hfq_minus_exchange")
        hv = "nan" if hv is None or hv != hv else f"{hv:.6f}"
        add(f"| {x['stock_code']} | {x['date']} | {x['kind']} | {hv} |")
    add("")
    add("## 日期分布（∩ 冻结池）\n")
    add("```")
    add(str(dict(sorted(Counter(x["date"][:4] for x in in_pool).items()))))
    add("```")
    add("")
    add("★ 注意这些日期在**多只股票上重复出现**（2018-08-16/23/28、2020-01-02、2021-01-07、"
        "2023-06-01）—— 说明它不是个股的公司行为处理错误，而是**厂商侧的批量事件**。\n")
    add("## 结论与处置\n")
    add("1. 真正进入产物的只有上表那几格，量级远小于「545」给人的印象；")
    add("2. **不屏蔽、不回改**（与 `HISTORY.md:372-382` 的现行处置一致）：逐格定性后并入基线，"
        "每月 `scripts/check_adj_factor.py` **只报新增**；")
    add("3. 影响面已可用本脚本随时复算 —— 以后引用「545」必须同时说明是哪一格口径。")

    text = "\n".join(lines)
    if a.out_dir:
        out = ROOT / a.out_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "REPORT.md").write_text(text, encoding="utf-8")
        rows = [{"stock_code": x["stock_code"], "date": x["date"], "kind": x["kind"],
                 "hfq_minus_exchange": x.get("hfq_minus_exchange")} for x in in_win]
        (out / "in_pool_in_window.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {out}/REPORT.md 与 in_pool_in_window.json")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
