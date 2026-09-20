"""`index_ths_daily` 名单完整性 sweep —— **手动强制**入口（薄封装）。

★ 2026-09-19 起逻辑已移进 `data_incremental/pipeline/roster.py`，并**挂进日更主流程**：
  跑 `python main.py` 就会自动同步名单、每 30 天自动 sweep 一次。
  本脚本只是「不等周期、立刻跑一次」的手动入口，实现与线上**同一份**，不会各改各的。

什么时候用它：
  · 刚往 `conf/ths_index_codes.txt` 里加了新代码（虽然日更也会自动回填，但想立刻要）
  · 怀疑有洞 / 刚发生过中断 / 换过机器
  · 每月的自动 sweep 之外想多跑一次

用法：
    python scripts/backfill_index_ths_daily.py              # 立刻全量刷新
    python main.py run --sweep-now                          # 等价（且顺带跑当日增量）
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_incremental import config as C                       # noqa: E402
from data_incremental.core import lock as lockmod              # noqa: E402
from data_incremental.core.client import Client                # noqa: E402
from data_incremental.pipeline import roster as roster_mod      # noqa: E402


def main() -> int:
    cfg = C.load()
    print(f"[{datetime.now():%H:%M:%S}] 名单完整性 sweep（全历史刷新，幂等可重跑）",
          flush=True)
    with lockmod.acquire():          # 与日更互斥，避免两个写者
        cli = Client(cfg)
        try:
            t0 = time.monotonic()
            res = roster_mod.sweep(cfg, cli)
        finally:
            cli.close()
    if not res.get("ok"):
        print(f"❌ {res.get('note')}")
        return 2
    print(f"[{datetime.now():%H:%M:%S}] ✔ 完成：{res['codes']:,} 只 / "
          f"取回 {res['fetched']:,} 行 / **补齐 {res['added']:,} 行** / "
          f"失败批 {res['failed']}，用时 {(time.monotonic()-t0)/60:.1f} 分", flush=True)
    if res["failed"]:
        print("⚠️ 有批次失败 —— 直接重跑本脚本即可补齐（幂等）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
