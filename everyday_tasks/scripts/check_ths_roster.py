"""THS 指数名单巡检 —— 发现厂商**新上**的板块指数，提醒把它加进固化名单。

背景：`index_ths_daily` 的抓取对象是 `conf/ths_index_codes.txt` 这份**固化名单**
（1,676 只）。固化是有意的取舍 —— 换来覆盖范围固定、可复现，代价是
**厂商新上的指数不会自动进来**。2026-09-19 实测过一次漏抓：
名单外/新上的 480 只，其 2026-01-05 起的 34 个交易日一直没抓，而它正落在
研究 test 窗里（见 `scripts/backfill_index_ths_daily.py` 的文件头）。

所以需要一个**把"厂商有新的"变成"要人做的一件事"**的检查点 —— 就是本脚本。

★ 只查不存：`index_ths_sector_categories` 作为**数据表**已于 2026-09-19 删除
  （快照无历史 ⇒ 回测前视）。这里只把它当**权威名单源**问一次，不落盘、不建表。

★ 复用工程自带 `Client`，走全局限速台账。

用法：
    python scripts/check_ths_roster.py            # 只看，不动名单
    python scripts/check_ths_roster.py --update   # 把新增的追加进名单（保留原顺序）
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_incremental import config as C                       # noqa: E402
from data_incremental.core.client import Client                # noqa: E402
from data_incremental.pipeline import roster as roster_mod      # noqa: E402

# ★ 名单的读写只有一份实现（`roster.py`）：日更主流程用的是它，本脚本也用同一份，
#   免得"线上自动同步"和"手动巡检"两套逻辑哪天对不上。
ROSTER_PATH = roster_mod.ROSTER_PATH
SRC_PATH = roster_mod.SRC_PATH


def read_roster() -> list[str]:
    return roster_mod._load_roster()


def main() -> int:
    update = "--update" in sys.argv
    known = set(read_roster())

    cli = Client(C.load())
    try:
        rows = cli.fetch_all(SRC_PATH, {}, page_size=1000, expect_rows=False)
    finally:
        cli.close()

    vendor = {str(r.get("index_code")) for r in rows if r.get("index_code")}
    if not vendor:
        # 空返回**绝不当成"厂商清空了"** —— 那会把名单误删成空表。
        print("❌ 厂商返回 0 条名单，无法比对（不修改名单）。请稍后重试或检查接口。")
        return 2

    added = sorted(vendor - known)
    gone = sorted(known - vendor)

    print(f"[{datetime.now():%H:%M:%S}] THS 名单巡检")
    print(f"  固化名单 {len(known):,} 只   /   厂商当前 {len(vendor):,} 只")
    print()
    if not added and not gone:
        print("  ✅ 完全一致，无需处理。")
        return 0

    if added:
        print(f"  🆕 厂商有、名单没有：{len(added)} 只 —— **这些不会被抓，需要你决定**")
        for c in added[:20]:
            print(f"       {c}")
        if len(added) > 20:
            print(f"       … 其余 {len(added)-20} 只见日志外")
    if gone:
        print(f"  ⚠️ 名单有、厂商没有：{len(gone)} 只")
        print("       ★ 多半正常：厂商停供后名单会一直留着它们（如 2026-09-14 的 291 只")
        print("         R/B 变体）。留着的代价只是每次多问一个空请求，**不会报错**；")
        print("         删掉则失去'厂商哪天恢复供应'的自动发现。默认不动。")
        for c in gone[:10]:
            print(f"       {c}")

    if update and added:
        # 只追加，不重排 —— 保持文件 diff 最小、可读
        with ROSTER_PATH.open("a", encoding="utf-8") as f:
            f.write(f"# ---- {datetime.now():%Y-%m-%d} 巡检新增 {len(added)} 只 ----\n")
            for c in added:
                f.write(c + "\n")
        print(f"\n  ✔ 已追加 {len(added)} 只到 {ROSTER_PATH}")
        print("  ⚠️ 追加后请补跑一次回填，把它们的**历史**也拉回来（新指数往往不是当天才有的）：")
        print("       python scripts/backfill_index_ths_daily.py")
    elif added:
        print(f"\n  → 要加进名单：`python {Path(__file__).name} --update`，然后补跑回填脚本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
