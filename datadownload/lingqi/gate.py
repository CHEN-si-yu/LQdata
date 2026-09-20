"""增量闸门 —— "上游到齐了没有？" 在开跑之前先问一遍。

用户 2026-09-13 的要求（原话）：

> 每次增量需要 check 所有的日频数据是否都更新到最新日期（注意这里的最新日期是
> **包含 delay 了的**，比如 stock_daily 的最新日期是 0911，而 stock_margin_detail
> 的最新日期是 0910），如果某一个日频数据还没有到达最新日期，则等待 5min，
> 再检查一次。总的等待时间是 4h。如果确认所有的日频数据都已经更新了，
> 那才开始一次对所有的日频数据进行更新。

**为什么需要它**：现在的 `main.py run` 是**无条件跑**的 —— 不管上游到没到齐，
缺什么补什么。问题是上游发布有先后：
- 大部分日频表当天收盘后就有（`stock_daily` 09-11 当天到）
- 但有 4 张表要**晚一个交易日**（`stock_margin_detail` / `index_ths_daily` /
  `stock_st_info` / `stock_adj_factor_changes`，最新只到 09-10）

如果不等，跑出来的就是**基于不完整上游**的结果；而且对模块② 有连带影响：
因子会基于不完整上游算出值，虽然后续会重算，但**重算之前下游可能已经用了**。

**判据（核心）**：
1. 取基准表（`stock_daily`，`frequency.yaml` 的 `baseline`）的最新日期 `B`；
2. 对每张日频表 `d`，它"应该到"的日期 = 交易日历里 `B` 往前数 `d.delay_days` 个交易日；
3. 该表实际 `max_date` ≥ 该日期 → 就绪；否则记为未就绪。

⚠️ 全程**零 API 请求** —— 只读 `state/*.json` 的 manifest 元数据。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# 只看这两类频率（"日频数据"= 每个交易日都该更新的）
DAILY_FREQS = ("daily_full", "daily_sparse")


def load_calendar(data_root: Path) -> list[str]:
    """交易日历（升序）。basic_calendar 存的是全部自然日，靠 is_open 区分。"""
    import pyarrow.parquet as pq

    f = Path(data_root) / "basic_calendar" / "data.parquet"
    if not f.exists():
        return []
    t = pq.read_table(f, columns=["date", "is_open"])
    days, opens = t.column("date").to_pylist(), t.column("is_open").to_pylist()
    return sorted({str(dv)[:10] for dv, op in zip(days, opens) if dv is not None and op == 1})


def dataset_max_date(state_root: Path, name: str) -> str | None:
    """从 manifest 的 partitions 里取该数据集的最大日期（零 IO，不读 parquet）。"""
    p = Path(state_root) / f"{name}.json"
    if not p.exists():
        return None
    try:
        man = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    ds = [v.get("max_date") for v in (man.get("partitions") or {}).values() if v.get("max_date")]
    return max(ds) if ds else None


def shift_back(calendar: list[str], anchor: str, n: int) -> str | None:
    """从 anchor 往回数 n 个交易日。anchor 不在日历里就取它之前最近的一个交易日。"""
    idx = None
    for i, x in enumerate(calendar):
        if x <= anchor:
            idx = i
        else:
            break
    if idx is None:
        return None
    j = idx - n
    return calendar[j] if j >= 0 else None


def check(specs, freq_cfg: dict, state_root: Path, data_root: Path) -> dict:
    """检查所有日频表是否已更新到"含 delay 的最新日期"。

    返回 {ready: bool, baseline: str|None, rows: [{name, freq, delay, actual, expected, ok}]}
    """
    ds_cfg = freq_cfg.get("datasets") or {}
    baseline_name = freq_cfg.get("baseline", "stock_daily")
    cal = load_calendar(data_root)

    base_max = dataset_max_date(state_root, baseline_name)
    rows: list[dict] = []
    if not cal or not base_max:
        return {"ready": False, "baseline": base_max, "rows": rows,
                "reason": "交易日历或基准表为空"}

    for s in specs:
        cfg = ds_cfg.get(s.name) or {}
        if cfg.get("freq") not in DAILY_FREQS:
            continue
        delay = int(cfg.get("delay_days", 0) or 0)
        expected = shift_back(cal, base_max, delay)
        actual = dataset_max_date(state_root, s.name)
        ok = bool(actual and expected and actual >= expected)
        rows.append({"name": s.name, "freq": cfg.get("freq"), "delay": delay,
                     "actual": actual, "expected": expected, "ok": ok})

    return {"ready": all(r["ok"] for r in rows) and bool(rows),
            "baseline": base_max, "baseline_name": baseline_name, "rows": rows}


def wait_until_ready(specs, freq_cfg: dict, state_root: Path, data_root: Path,
                     interval: float = 300, max_wait: float = 4 * 3600,
                     log=print) -> dict:
    """反复检查直到就绪或超时。默认每 5 分钟一次，总预算 4 小时。"""
    t0 = time.time()
    attempt = 0
    while True:
        attempt += 1
        res = check(specs, freq_cfg, state_root, data_root)
        elapsed = time.time() - t0

        if res["ready"]:
            log(f"✔ 日频数据已全部到齐（基准 {res['baseline_name']} = {res['baseline']}，"
                f"检查 {attempt} 次 / 用时 {elapsed:.0f}s）")
            return res

        bad = [r for r in res["rows"] if not r["ok"]]
        log(f"⏳ 第 {attempt} 次检查：{len(bad)}/{len(res['rows'])} 张日频表还没到齐"
            f"（已等 {elapsed / 60:.1f} 分钟，预算 {max_wait / 3600:.1f} 小时）")
        for r in bad[:12]:
            log(f"     {r['name']:32s} 实际 {r['actual']} < 应到 {r['expected']}"
                f"（delay {r['delay']}）")
        if len(bad) > 12:
            log(f"     … 另有 {len(bad) - 12} 张")

        if elapsed + interval > max_wait:
            log(f"✖ 等待超时（{max_wait / 3600:.1f} 小时）：仍有 {len(bad)} 张日频表未到齐，"
                f"**不开始增量**（用户要求『确认到齐才跑』）。")
            res["timed_out"] = True
            return res

        log(f"     等 {interval / 60:.0f} 分钟后再查一次…")
        time.sleep(interval)
