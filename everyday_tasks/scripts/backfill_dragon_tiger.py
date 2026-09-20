"""`stock_dragon_tiger` 全量回填 —— 主键 4 列 → 7 列（2026-09-15 晚，用户拍板）之后必须重抓一遍。

为什么必须重抓：旧主键（trade_date, stock_code, org_name, direction）把**同名的多个机构席位
合并成一条**（`机构专用` 是匿名化的多个机构席位，同名不同金额 = 不同记录）。被合并掉的行
**本地根本没有**，所以只能回服务端重抓；好在重抓是幂等的（新主键下 upsert 会把缺失的行补上、
已有的行原样覆盖）。

做法：按交易日逐天请求 `/stock/dragon_tiger?date=YYYY-MM-DD`（每天 1 个请求，约 3,300 天），
用本工程自己的 `_Buf` 落盘（列对齐 / 主键去重 / manifest 分区元数据 / chmod 全部沿用线上路径），
**按年 flush**（避免同一个年分区被反复整文件重写 = O(n²)）。

用法（脱离会话）：
    setsid nohup /autodl-fs/data/miniconda3/bin/python -u scripts/backfill_dragon_tiger.py \
        > logs/backfill_dragon_tiger.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd                                        # noqa: E402

from data_incremental import config as C, paths, registry as R   # noqa: E402
from data_incremental.core import lock as lockmod, state          # noqa: E402
from data_incremental.core.client import Client, extract_list     # noqa: E402
from data_incremental.pipeline.calendar import trading_days_between  # noqa: E402
from data_incremental.pipeline.strategies import _Buf             # noqa: E402

DS_NAME = "stock_dragon_tiger"
START = "2013-01-01"          # 与 spec 的 start 一致
END = datetime.now().strftime("%Y-%m-%d")


def main() -> int:
    cfg = C.load()
    ds = R.get(DS_NAME)
    cal = sorted(str(x)[:10] for x in json.loads(
        (paths.STATE / "calendar_ext.json").read_text(encoding="utf-8")))
    days = trading_days_between(cal, START, END)
    print(f"[{datetime.now():%H:%M:%S}] 回填 {DS_NAME}：{len(days)} 个交易日 "
          f"({days[0]} ~ {days[-1]})，主键 = {ds.keys}", flush=True)

    with lockmod.acquire():          # force 会抢活进程的锁，这里不用
        cli = Client(cfg)
        man = state.Manifest.load(DS_NAME)
        buf = _Buf(ds, man, flush_rows=200_000, flush_batches=400)
        t0 = time.monotonic()
        done = empty = rows = 0
        cur_year = None
        for d in days:
            if cur_year is not None and d[:4] != cur_year:
                buf.flush()                     # 跨年 → 落盘一次（一个年分区只重写一次）
            cur_year = d[:4]
            try:
                data = cli.call(ds.path, {**ds.params, ds.date_param: d,
                                          "page": 0, "page_size": 2000},
                                method=ds.method, expect_rows=False)
                recs = extract_list(data)
            except Exception as exc:            # noqa: BLE001
                print(f"  ⚠️ {d} 抓取失败：{str(exc)[:90]}", flush=True)
                continue
            if recs:
                buf.add(pd.DataFrame(recs))
                man.mark_done("dates", d)       # 与线上 per_date 策略一致
                rows += len(recs)
                done += 1
            else:
                empty += 1
            if (done + empty) % 200 == 0:
                el = time.monotonic() - t0
                n = done + empty
                print(f"  [{datetime.now():%H:%M:%S}] {n}/{len(days)} 天"
                      f"（有数据 {done} / 空 {empty}），累计 {rows:,} 行，"
                      f"用时 {el/60:.1f} 分，预计还需 {el/n*(len(days)-n)/60:.1f} 分", flush=True)
        buf.flush()
        man.save()
        print(f"[{datetime.now():%H:%M:%S}] ✔ 回填完成：{done} 天有数据 / {empty} 天空，"
              f"共写入 {rows:,} 行，新增 {buf.rows:,} 行，用时 {(time.monotonic()-t0)/60:.1f} 分",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
