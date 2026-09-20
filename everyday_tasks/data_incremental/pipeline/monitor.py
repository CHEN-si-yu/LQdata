"""数据更新监测 —— **只读**，每晚跑一次，把观测攒成可决策的证据。

为什么需要它（用户 2026-09-15 的原话）：
    「很多东西只能在动态更新的时候才能发现，比如说 delay 到底是多少等问题。
      我需要连续跟踪几天才能够完全确认目前的配置。」

它回答三个问题：
  1. **服务端到底哪天有数据** → 每张日频表的**真实 delay**（声明值 vs 观测值）
  2. **本地到底更新到位没有** → 每张表本地水位 vs 应有的水位
  3. **几天下来稳不稳** → 逐日起伏、是否需要改配置

★ 三个设计要点：
  · **只读**：不发任何写请求、不改数据、不改 manifest。任何时刻都能安全跑。
  · **区分稠密表与稀疏表**：稠密表逐日往前探（T, T-1, T-2…）；稀疏表（停牌、
    涨跌停这类"某天本来就该为空"的表）单日探测必然大量为空，改用 10 天窗口
    一次拿到"服务端最新日期"—— 与闸门 `probe_table_ready` 同一套判据。
  · **攒历史**：每次一条 append 到 `state/monitor/history.jsonl`，
    报告窗口里的众数/最大值/波动都从它算。样本 <3 天时明确说"证据不足"。
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from datetime import datetime

from .. import paths, registry as R
from ..core import state
from ..core.client import Client
from . import probe as probe_mod

# 稠密表往前探几个交易日就放弃（探测成本 = 表数 × 天数）
MAX_BACK = 3
# 至少要看几天才敢下"改 delay"的结论
MIN_SAMPLES = 3

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# 只收**真正的日历轴**（交易日 / 发布日 / 时刻）；`end_date` / `ann_date` 是
# "报告期 / 公告日"，问"某天服务端有没有"没有意义，会把观测变成噪声。
_MONITOR_DATE_FIELDS = ("trade_date", "date", "report_date", "trade_time")


def monitored_tables() -> list["R.DS"]:
    """要跟踪的表 = 日频表（闸门管的）+ 其他**有日历轴**的表。

    ★ 2026-09-15 扩展。原实现只跟踪 `R.daily_tables()`，于是 `stock_report_rc`
      （freq=irregular）这种"发布日稀疏 + 有发布延迟"的表**完全不进观测** ——
      它的 delay 只能靠人肉猜，最后连报三天假告警才被发现（见 registry 的
      `_EXPECT_ROWS` / `_DELAY` 注释）。用户要的"持续跟踪几天把 delay 定下来"
      必须覆盖到这类表，否则同一类坑还会再踩。
    """
    out = list(R.daily_tables())
    seen = {d.name for d in out}
    for d in R.all_ds():
        if not d.enabled or d.name in seen or d.mode in ("snapshot", "dump"):
            continue
        if d.date_field in _MONITOR_DATE_FIELDS:
            out.append(d)
            seen.add(d.name)
    return out


def history_path():
    return paths.STATE / "monitor" / "history.jsonl"


def _ensure_dir():
    history_path().parent.mkdir(parents=True, exist_ok=True)


def load_history(limit: int | None = None) -> list[dict]:
    p = history_path()
    if not p.exists():
        return []
    out: list[dict] = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out[-limit:] if limit else out


def append_history(rec: dict) -> None:
    _ensure_dir()
    with open(history_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ================================================================ 观测
def _local_max(ds: R.DS) -> str | None:
    return state.Manifest.load(ds.name).max_partition_date()


def observe_table(client: Client, ds: R.DS, cal: list[str], T: str) -> dict:
    """探出这张表**服务端最新有数据的交易日**，据此算观测 delay。"""
    declared = int(ds.delay_days or 0)
    local_max = _local_max(ds)
    local_lag = None
    if local_max and T in cal and local_max in cal:
        local_lag = cal.index(T) - cal.index(local_max)

    ratio = probe_mod.trailing_nonzero_ratio(ds.name, ds.date_field)
    server_max: str | None = None
    probes = 0

    if ds.mode == "snapshot":
        return {"declared": declared, "server_max": None, "lag": None,
                "local_max": local_max, "local_lag": local_lag,
                "sparse": None, "probes": 0, "note": "快照表，无日期轴"}

    if ratio < 0.8:
        # 稀疏表：单日探测会把"合法空日"误判成"没有数据"，用 10 天窗口一次问出来
        tp = probe_mod.probe_table_ready(client, ds, T, cal, sparse_ratio=ratio)
        probes = 1
        if _DATE_RE.match(str(tp.actual or "")):
            server_max = str(tp.actual)[:10]
    else:
        for i in range(MAX_BACK + 1):
            d = state.shift_back(cal, T, i)
            if d is None:
                break
            probes += 1
            n = probe_mod.probe_rows(client, ds, d, cal, page_size=3)
            if n:
                server_max = d
                break

    lag = None
    if server_max and T in cal and server_max in cal:
        lag = cal.index(T) - cal.index(server_max)

    return {"declared": declared, "server_max": server_max, "lag": lag,
            "local_max": local_max, "local_lag": local_lag,
            "sparse": round(ratio, 3), "probes": probes, "note": ""}


def observe(client: Client, cfg: dict, cal: list[str], T: str, log=print) -> dict:
    """对全部被跟踪的表观测一遍。只读。"""
    tables: dict[str, dict] = {}
    for ds in monitored_tables():
        tables[ds.name] = observe_table(client, ds, cal, T)
        log(f"   {ds.name:34} 服务端最新 {str(tables[ds.name]['server_max'] or '—'):11} "
            f"声明 delay={tables[ds.name]['declared']} "
            f"观测={tables[ds.name]['lag'] if tables[ds.name]['lag'] is not None else '—'}")
    return {"observed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "T": T, "tables": tables}


# ================================================================ 汇总
def summarize(history: list[dict], cal: list[str]) -> list[dict]:
    """把历史观测按表汇总，给出建议 delay 与证据强度。"""
    names: list[str] = []
    for rec in history:
        for n in (rec.get("tables") or {}):
            if n not in names:
                names.append(n)

    # ★ 2026-09-17：只汇总**当前仍启用**的数据集。
    #   `history.jsonl` 是 append-only 的，里面留着已删除数据集（index_weight /
    #   stock_report_rc / main_fund_flow_overview）的历史观测 —— 不过滤的话，
    #   `monitor` 会一直把它们列出来并报"落后 N 天"（实测 index_weight 报 55 天），
    #   是纯噪声，还会误导每晚读这张表的人。
    live = {d.name for d in R.enabled()}

    out: list[dict] = []
    for name in names:
        if name not in live:
            continue
        lags, missing = [], 0
        for rec in history:
            t = (rec.get("tables") or {}).get(name) or {}
            v = t.get("lag")
            if isinstance(v, int):
                lags.append(v)
            else:
                missing += 1
        declared = None
        latest: dict = {}
        for rec in reversed(history):          # 最近一条观测里的本地水位
            t = (rec.get("tables") or {}).get(name) or {}
            if not latest and t:
                latest = t
            if declared is None and t.get("declared") is not None:
                declared = int(t["declared"])
            if declared is not None and latest:
                break
        if declared is None:
            continue

        n = len(lags)
        # ★ 「波动」= 同一张表的滞后在几次观测里**不一致**（例如 dc_daily 的 1,1,0,1）。
        #   用户 2026-09-15 的原则：「以稳定存在的为主……不要去冒险提取最新的数据」。
        #   所以建议值永远取**最大滞后**（最保守的那一档），并把波动明说出来 ——
        #   它解释了"为什么明明有时能拿到 T，却还是按 delay=1 跑"。
        volatile = n >= 1 and len(set(lags)) > 1
        hit_t = sum(1 for v in lags if v == 0)      # 观测到"T 当天就有数据"的次数
        rec_delay, why = declared, "样本不足"
        if n >= MIN_SAMPLES:
            mx, mode = max(lags), Counter(lags).most_common(1)[0][0]
            if mx > declared:
                rec_delay = mx
                why = f"最近 {n} 天里最大滞后 {mx} 天（众数 {mode}）"
                if volatile and declared > 0:
                    why += (f"；⚡ 波动：{hit_t}/{n} 次看到 T 当日就有数据 "
                            f"→ 按最保守的 {mx} 天用（只取稳定日）")
            elif mode < declared and mx < declared:
                rec_delay = mx
                why = f"最近 {n} 天最大滞后只有 {mx} 天，声明 {declared} 偏保守"
            else:
                why = (f"最近 {n} 天滞后 {lags}，与声明一致"
                       + (f"；⚡ {hit_t}/{n} 次看到 T 当日就有数据（波动，仍按 {declared} 跑）"
                          if volatile and declared > 0 else ""))
        # ★ 「本地落后服务端多少」才是**我们的**问题。
        #   本地落后 T 不一定是问题 —— 服务端自己就可能没有更新的数据
        #   （实测 `index_weight`：上游 2026-08/09 直接返回 0 行，服务端最新也是
        #    2026-07-01）。按"本地 vs T"判会天天报"落后 53 天"，纯告警疲劳。
        _sl, _ll = latest.get("lag"), latest.get("local_lag")
        behind = (_ll - _sl) if isinstance(_sl, int) and isinstance(_ll, int) else None

        out.append({
            "name": name, "declared": declared, "samples": n, "missing": missing,
            "server_lag": _sl, "behind": behind,
            "local_max": latest.get("local_max"), "local_lag": latest.get("local_lag"),
            "lags": lags, "max": max(lags) if lags else None,
            "mode": Counter(lags).most_common(1)[0][0] if lags else None,
            "rec": rec_delay, "why": why,
            "change": rec_delay != declared,
            "volatile": volatile, "hit_t": hit_t,
            "stable": (n >= MIN_SAMPLES and len(set(lags)) == 1),
        })
    return out


def render(rows: list[dict], history: list[dict]) -> str:
    out = []
    out.append("=" * 92)
    out.append("  Delay 标定追踪")
    out.append("=" * 92)
    out.append(f"  样本天数 {len(history)}"
               + ("" if len(history) >= MIN_SAMPLES
                  else f"  ⏳ 还不足 {MIN_SAMPLES} 天，下面的建议只是初步观测"))
    out.append("")
    out.append(f"  {'数据集':32}{'声明':>5}{'观测滞后(新→旧)':>22}{'建议':>6}  {'稳':>2} 说明")
    out.append("  " + "-" * 90)
    for r in sorted(rows, key=lambda x: (-(x["max"] or 0), x["name"])):
        seq = ",".join(str(x) for x in reversed(r["lags"])) or "—"
        mark = "  ★改" if r["change"] else ""
        # ⚡ = 观测值有起伏（该表在 T 当天时有时无）；✔ = 观测值完全一致
        stab = "⚡" if r.get("volatile") else ("✔" if r.get("stable") else "·")
        out.append(f"  {r['name']:32}{r['declared']:>5}{seq:>22}{r['rec']:>6}  {stab:>2} {r['why']}{mark}")
    out.append("")
    out.append("  ⚡ = 观测值有起伏（T 当天时有时无）→ 建议值取**最保守**的那一档，只取稳定日")
    out.append("  ✔ = 连续观测完全一致，可以定下来")
    return "\n".join(out)


def render_status(rows: list[dict], T: str) -> str:
    """本地水位 vs **服务端水位** —— 一眼看出"今晚到底更新到位没有"。

    ★ 判据是"本地比服务端还落后多少"，不是"本地落后 T 多少"（2026-09-15 改）。
      服务端自己就可能没有更新的数据（`index_weight` 上游 2026-08/09 直接空），
      按 T 判会天天报"落后 53 天" —— 那不是我们的问题，只会淹没真信号。
      没探到服务端水位时（`--no-probe` / 探测失败）退回"比声明 delay"的旧判据。
    """
    out = ["", "=" * 92, "  本地水位 vs 服务端水位", "=" * 92,
           f"  T = {T}（服务端已发布的最新交易日）", ""]
    out.append(f"  {'数据集':32}{'本地最新':>12}{'声明delay':>10}{'落后T':>6}{'本地-服务端':>11}  判定")
    out.append("  " + "-" * 94)
    bad = 0
    for r in sorted(rows, key=lambda x: -(x["local_lag"] if x["local_lag"] is not None else -99)):
        lag, behind, decl = r["local_lag"], r.get("behind"), r["declared"]
        loc = r["local_max"] or "—"
        srv_lag = r.get("server_lag")
        if lag is None:
            verdict = "—  无数据/快照"
        elif behind is not None:
            if behind <= decl:
                verdict = ("✔ 到位" if behind <= 0
                           else f"✔ 到位（服务端也只到 T-{srv_lag}）")
            else:
                verdict = f"⚠️ 本地比服务端还落后 {behind} 天（服务端已到 T-{srv_lag}）"
                bad += 1
        elif lag <= decl:
            verdict = "✔ 到位"
        else:
            verdict = f"⚠️ 比声明落后 {lag - decl} 天（未探到服务端水位，按声明判）"
            bad += 1
        out.append(f"  {r['name']:32}{str(loc):>12}{decl:>10}"
                   f"{(lag if lag is not None else '—'):>6}"
                   f"{(str(behind) if behind is not None else '—'):>11}  {verdict}")
    out.append("")
    out.append(f"  落后于服务端的表：{bad} 张　（本地-服务端 ≤ 声明 delay 就算到位）")
    return "\n".join(out)
