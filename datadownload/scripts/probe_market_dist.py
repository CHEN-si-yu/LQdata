#!/usr/bin/env python
"""探测 /api/stock/market_distribution_history 的数据可用区间。

背景：2026-09-13 的 `main.py doctor` 对该接口用「最近一个交易日」探测，返回 0 行。
需要判断这是「当天数据未发布」还是「接口系统性为空」。

★ 本脚本从 APIKey.txt 读取密钥（经 lingqi.client），**不要**像 scripts/probe1-9.py
  那样把 key 明文硬编码在源码里。

用法：
    /autodl-fs/data/miniconda3/bin/python scripts/probe_market_dist.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import yaml
from lingqi.client import LingqiClient, extract_list

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = yaml.safe_load(open(os.path.join(ROOT, "conf", "config.yaml"), encoding="utf-8"))
client = LingqiClient(cfg, None)

# 覆盖：最新交易日 / 往前若干日 / 早年 / 极早年
DATES = [
    "2026-09-11", "2026-09-10", "2026-09-09", "2026-09-04", "2026-09-03",
    "2026-08-28",
    "2025-06-16", "2024-01-02",
    "2020-01-02", "2015-01-05", "2010-01-04",
]

print(f"{'日期':<12} {'行数':>6}  {'耗时':>6}  样本")
print("-" * 78)
for d in DATES:
    t0 = time.perf_counter()
    try:
        data = client.call("/stock/market_distribution_history", {"date": d},
                           method="POST", expect_rows=False)
        rows = extract_list(data)
        el = time.perf_counter() - t0
        # 信封形态：期望 data.date / data.count / data.list
        env = ""
        if isinstance(data, dict):
            env = f"env.date={data.get('date')} env.count={data.get('count')}"
        sample = json.dumps(rows[0], ensure_ascii=False)[:60] if rows else ""
        print(f"{d:<12} {len(rows):>6}  {el:>5.1f}s  {sample}")
        if env and len(rows) == 0:
            print(f"{'':12} └─ {env}")
    except Exception as exc:  # noqa: BLE001
        el = time.perf_counter() - t0
        print(f"{d:<12} {'ERR':>6}  {el:>5.1f}s  {str(exc)[:50]}")

print()
print("统计:", client.stats.snapshot())
