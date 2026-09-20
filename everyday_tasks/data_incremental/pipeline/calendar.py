"""交易日历维护。

两个职责：
  1. **更新到未来 T+30**（用户 2026-09-14 要求）—— 提供前瞻窗口，
     日历本身也决定"哪些日子该有数据"。
  2. **自愈** —— 服务端在探针里返回过的交易日都并进 `calendar_ext.json`。
     这是零额外请求的兜底：即使 `/basic/calendar` 拿不到未来日期，
     探针每天遇到的交易日也会被记下来。

⚠️ `basic_calendar` 是**快照**表，但**绝不能用整表覆盖写**：
   旧工程实测过一次短响应就把 2010 年以来的历史日历整表删掉的风险场景。
   这里改用 upsert（keys=("date",)）—— 只增不改，最坏情况是什么都不加。
"""
from __future__ import annotations

from datetime import date as _date
from datetime import timedelta

import pandas as pd

from .. import registry as R
from ..core import state, store
from ..core.client import Client


def extend_days(cfg: dict) -> int:
    return int(cfg.get("calendar", {}).get("extend_days", 30))


def update_calendar(client: Client, cfg: dict, write: bool = True) -> dict:
    """把 basic_calendar 更新到「今天 + extend_days」。返回统计信息。

    `write=False`（dry-run）时只取回来看一眼，**不落盘**。
    """
    ds = R.get("basic_calendar")
    end = (_date.today() + timedelta(days=extend_days(cfg))).isoformat()
    man = state.Manifest.load("basic_calendar")
    before_max = man.max_partition_date()

    rows: list[dict] = []
    err: str | None = None
    try:
        rows = client.fetch_all(
            ds.path, {**ds.params, ds.start_param: ds.start, ds.end_param: end},
            page_size=ds.page_size, method=ds.method, expect_rows=False)
    except Exception as exc:  # noqa: BLE001
        err = str(exc)[:200]

    if not rows:
        return {"ok": False, "rows": 0, "before_max": before_max, "after_max": before_max,
                "note": err or "服务端未返回日历数据（不动本地，退回自愈模式）"}

    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str[:10]
    today_s = _date.today().isoformat()
    if not write:
        dsx = sorted(set(df["date"].astype(str).str[:10])) if "date" in df.columns else []
        return {"ok": True, "rows": len(df), "total": len(dsx),
                "before_max": before_max, "after_max": dsx[-1] if dsx else None,
                "future_days": len([d for d in dsx if d > today_s]),
                "note": "dry-run，未落盘"}
    path = store.flat_path("basic_calendar")
    merged = store.upsert(path, df, keys=("date",), sort_by=("date",))
    dsx = sorted(set(merged["date"].astype(str).str[:10]))
    man.mark_partition(0, len(merged), dsx[0] if dsx else None, dsx[-1] if dsx else None)
    man.columns = list(merged.columns)
    man.add_coverage(dsx[0], dsx[-1]) if dsx else None
    man.save()

    # 顺手把日历里到今天为止的交易日并进自愈日历（零额外请求）
    today = _date.today().isoformat()
    opens = []
    if "is_open" in merged.columns:
        opens = [d for d, o in zip(merged["date"].astype(str).str[:10], merged["is_open"])
                 if str(o) in ("1", "1.0", "True")]
    state.calendar_ext_add([d for d in opens if d <= today])

    return {"ok": True, "rows": len(df), "total": len(merged),
            "before_max": before_max, "after_max": dsx[-1] if dsx else None,
            "future_days": len([d for d in dsx if d > today]), "note": err or ""}


def trading_days_between(cal: list[str], start: str, end: str) -> list[str]:
    return [d for d in cal if start <= d <= end]


def calendar_days_between(start: str, end: str) -> list[str]:
    """`[start, end]` 之间的**全部日历日**（含周末）。

    ★★ 2026-09-16 新增。为什么需要：`stock_holder_number` 的台账键是 `ann_date`
      （公告日），而**公告日含周末** —— 实测 2026-09-12（周六）有 4 条、09-13（周日）有 1 条。
      沿用 `trading_days_between` 会让那几格**永远不被指纹化** → 台账漏天。
      只在 `DS.ledger_calendar_days=True` 的表上使用，其它表行为完全不变。
    """
    from datetime import date, timedelta
    try:
        a = date.fromisoformat(str(start)[:10])
        b = date.fromisoformat(str(end)[:10])
    except ValueError:
        return []
    if b < a:
        return []
    return [(a + timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def tail_window(cal: list[str], T: str, n: int) -> tuple[str, str]:
    """尾部窗口 `[T-n, T]`（都是交易日）。

    ★ 用户 2026-09-14 要求：「T 时刻增量更新，不仅要取 T，还要取 T-1…T-5 这五个
      交易日的数据，冗余 5 天」。语义是 **[T - n, T]**，n=5 → 6 个交易日。
      改一处 `conf/daily.yaml` 的 `redundancy_days` 即可。
    """
    if T not in cal:
        cand = [d for d in cal if d <= T]
        if not cand:
            return T, T
        T = cand[-1]
    i = cal.index(T)
    j = max(0, i - int(n))
    return cal[j], T
