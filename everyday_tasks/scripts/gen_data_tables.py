#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""现读注册表，打印启用表的「编号对照」，并核对冻结编号是否与现状一致。

用法：
    /autodl-fs/data/miniconda3/bin/python scripts/gen_data_tables.py

★ 编号纪律（见 DATA_TABLES.md）：
    本文档编号已于 2026-09-19 冻结，**永久绑定数据集，不再变动**。
    · 本脚本按 `(频率档, 名字母序)` 排序 —— 这是**当年分配编号时的顺序**，
      所以现状若与 FROZEN 完全一致，打印出来的编号就等于文档编号。
    · **新增**数据 = 追加到末尾（32、33…），**绝不打乱已有编号**。
    · **删除**数据 = 编号作废、不复用。
    · 因此 `--check` 报出差异时：不要重排，改 DATA_TABLES.md 末尾追加即可。
"""
from __future__ import annotations

import datetime
import json
import os
import sys

sys.path.insert(0, "/autodl-fs/data/datadownload")
sys.path.insert(0, "/autodl-fs/data/everyday_tasks")

from scripts import catalog as CAT  # noqa: E402
from data_incremental import registry as R  # noqa: E402

STATE = "/autodl-fs/data/datadownload/state"

# 2026-09-19 冻结的编号 → 数据集名。**这是权威，不要重排。**
FROZEN: dict[int, str] = {
    1: "dc_daily", 2: "index_daily", 3: "index_ths_daily", 4: "stock_adj_factor",
    5: "stock_cyq_chips", 6: "stock_daily", 7: "stock_finance",
    8: "stock_main_fund_flow", 9: "stock_margin_detail", 10: "tdx_daily",
    11: "stock_adj_factor_changes", 12: "stock_dragon_tiger", 13: "stock_limit_list",
    14: "stock_limit_up", 15: "stock_st_info", 16: "stock_suspension",
    17: "stock_top_list", 18: "stock_history_5min",
    19: "stock_market_distribution_history", 20: "tdx_minute",
    21: "stock_balancesheet", 22: "stock_cashflow", 23: "stock_financial_indicator",
    24: "stock_income", 25: "stock_pledge_stat",
    26: "stock_forecast", 27: "stock_holder_number",
    28: "basic_calendar", 29: "dc_blocks", 30: "stock_list", 31: "tdx_blocks",
    32: "stock_cyq_perf",
}

# 展示层频率归并：分钟级并入日频（注册表里的 freq 不动，见 DATA_TABLES.md）
GROUP = {"daily_full": "日频", "daily_sparse": "日频", "minute": "日频",
         "quarterly": "季频", "irregular": "不定期", "snapshot": "快照"}
ORDER = ["daily_full", "daily_sparse", "minute", "quarterly", "irregular", "snapshot"]


def years(a: str | None, b: str | None) -> str:
    if not a or not b or not a[:1].isdigit():
        return "—"
    return f"{(datetime.date.fromisoformat(b) - datetime.date.fromisoformat(a)).days / 365.25:.1f}"


def coverage(d) -> str:
    """从 state/<name>.json 的 partitions 现读覆盖范围。"""
    if d.mode == "snapshot":
        return "未来日历（到 T+30）" if d.name == "basic_calendar" else "无日期轴"
    try:
        meta = json.load(open(os.path.join(STATE, d.name + ".json")))
        parts = meta.get("partitions") or {}
        lo = min((v.get("min_date") for v in parts.values()
                  if isinstance(v, dict) and v.get("min_date")), default=None)
        hi = max((v.get("max_date") for v in parts.values()
                  if isinstance(v, dict) and v.get("max_date")), default=None)
        return f"{lo} ~ {hi}（{years(lo, hi)} 年）"
    except Exception:
        return "—"


def live() -> list:
    """当前启用的接口，按分配编号时的顺序。"""
    out = []
    for freq in ORDER:
        out += sorted((x for x in R.all_ds() if x.enabled and x.freq == freq),
                      key=lambda x: x.name)
    # 永久编号决定输出顺序；新增接口不得挤走历史编号。
    number = {name: i for i, name in FROZEN.items()}
    return sorted(out, key=lambda d: (number.get(d.name, 10**9), d.name))


def main() -> int:
    force_check = "--check" in sys.argv
    rows = live()
    names = [d.name for d in rows]

    print(f"# 全局数据表（现读 · {datetime.date.today()}）\n")
    print("| # | 数据集 | 接口路径 | 频率 | 覆盖范围 | 主键（一行 = 什么） | 说明 |")
    print("|--:|:--|:--|:--|:--|:--|:--|")
    dist: dict[str, int] = {}
    number = {name: i for i, name in FROZEN.items()}
    for offset, d in enumerate(rows, 1):
        i = number.get(d.name, f"新增:{offset}")
        g = GROUP[d.freq]
        dist[g] = dist.get(g, 0) + 1
        print(f"| {i} | `{d.name}` | `{d.path}` | {g} | {coverage(d)} | "
              f"{' + '.join(d.keys)} | {CAT._USAGE.get(d.name, '')} |")
    total = len(rows)
    print(f"\n合计 **{total}** 个启用接口 | 频率分布："
          + " · ".join(f"{k} **{v}**" for k, v in dist.items()))

    # ---- 与冻结编号核对 ----
    frozen_names = [FROZEN[k] for k in sorted(FROZEN)]
    added = [n for n in names if n not in FROZEN.values()]
    removed = [n for n in frozen_names if n not in names]
    frozen_on = [n for n in frozen_names if n in names]

    print("\n" + "=" * 60)
    if not added and not removed and frozen_on == [n for n in names if n in FROZEN.values()]:
        print(f"✅ 与冻结编号一致（{total} 个，编号 1–{max(FROZEN)} 全部对得上）")
        return 0

    print("⚠️ 与冻结编号有差异 —— **不要重排编号**，按下面处理 DATA_TABLES.md：")
    if added:
        print(f"\n新增（追加到末尾，编号从 {max(FROZEN) + 1} 起）：")
        for j, n in enumerate(added):
            print(f"  {max(FROZEN) + 1 + j:>3}  {n}")
    if removed:
        print("\n已删除（编号作废、不复用）：")
        for n in removed:
            print(f"  {[k for k, v in FROZEN.items() if v == n][0]:>3}  {n}")
    print("\n若只是顺序变了（增删导致的重排），**忽略本脚本的序号，以 DATA_TABLES.md 为准**。")
    return 1 if force_check else 0


if __name__ == "__main__":
    raise SystemExit(main())
