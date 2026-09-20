"""延迟观测与自动标定。

背景（用户 2026-09-14 明确）：
    「有的数据存在 delay，目前我能确认的是日频数据 stock_margin_detail 就存在一天的
     delay，其余的数据我还不确定。如果是确定了 delay 窗口的，在更新 T 时间的时候，
     普通数据是 check 服务器 T 时间的数据是否存在，而这一类数据是 check T-delay
     这个数据是否存在。」

所以 delay 是**闸门判据的参数**，必须准确。但它会变（发布节奏、节假日），
靠人肉维护不现实，所以这里做**连续观测 + 自动标定**。

标定规则（保守方向）：
    resolved = max(declared, 近 N 次观测的众数)
**只上不下** —— 宁可多等一个交易日，也不能提前跑（提前跑拿到的是空数据，
会白记一堆 suspect）。

⚠️ 不写回 `datadownload/conf/frequency.yaml`：那份配置是模块② 因子工程
   「可得日锚定」的依据，悄悄改它会让历史因子的语义漂移。本工程只在自己的
   state 里覆盖，并在报告里提示"建议人工更新 frequency.yaml"。
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime

from .. import paths
from ..core.state import shift_back

MAX_HISTORY = 60          # 保留最近多少次观测
OBSERVE_WINDOW = 10       # 用最近多少次观测投票


def _load(path, default):
    try:
        v = json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        return v if isinstance(v, type(default)) else default
    except (json.JSONDecodeError, OSError):
        return default


def _save(path, obj) -> None:
    from ..core.state import _atomic_json
    _atomic_json(path, obj)


def _is_date(s) -> bool:
    """是不是 `YYYY-MM-DD`。用来把 `"有"` / `"未知"` / `"不适用"` 这类**状态字符串**
    挡在日期比较之外（见 `delay_from_server_max` 的注释）。"""
    return isinstance(s, str) and len(s) == 10 and s[4] == "-" and s[7] == "-"


def record_observation(T: str, probes: list[dict], mode: str = "ready") -> None:
    """记一次观测：`{日期: {T, mode, 各表服务端最新日期, 观测到的 delay}}`。

    probes: [{"name":..., "server_max":..., "delay_obs":..., "ready":bool,
              "probe_ok":bool}, ...]

    `mode`：`ready`（闸门真正通过）/**`timeout`**（预算耗尽转 partial）/
           **`dry_run`**（干跑，预算被压成 0，必然"超时"）/ `strict`。
    ★★ 2026-09-17：为什么要记 mode —— 旧实现按 T 覆盖写、且干跑也会写一条
       `ready=False` 的观测。实测 2026-09-16 23:56 的一次 dry-run 把 22:21
       闸门**真正通过**的那条记录覆盖掉了，于是 `absent_streak` 被
       "人为制造的失败"喂大 → 触发 delay 自动上调 → 取数窗口上界后退
       → 最新几天永远不抓（而报告还显示 ✔）。现在 `absent_streak` 只认
       `mode == "ready"` 且 `probe_ok` 的记录。
    """
    hist = _load(paths.DELAY_HISTORY, {})
    hist[T] = {
        "observed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "T": T,
        "mode": mode,
        "tables": {p["name"]: {"server_max": p.get("server_max"),
                               "server_max_raw": p.get("server_max_raw"),
                               "probe_ok": p.get("probe_ok", True),
                               "delay_obs": p.get("delay_obs"),
                               "ready": p.get("ready")} for p in probes},
    }
    keys = sorted(hist)[-MAX_HISTORY:]
    _save(paths.DELAY_HISTORY, {k: hist[k] for k in keys})


def observed_delay(name: str) -> int | None:
    """近 N 次观测里该表 delay 的众数。样本不足返回 None。"""
    hist = _load(paths.DELAY_HISTORY, {})
    vals = []
    for rec in list(hist.values())[-OBSERVE_WINDOW:]:
        v = (rec.get("tables") or {}).get(name) or {}
        d = v.get("delay_obs")
        if isinstance(d, int) and d >= 0:
            vals.append(d)
    if len(vals) < 3:            # 样本太少不下结论
        return None
    return Counter(vals).most_common(1)[0][0]


def absent_streak(name: str) -> int:
    """该表**连续多少轮**在 `expected` 那天被判"没到齐"（ready=False）。

    ★ 这是稠密表唯一可用的 delay 信号。稠密表单日探测只回 "有"/"无"，
    回不出"服务端最新日期"，所以"它到底晚几天"是观测不到的；但
    "它连着好几天都没到"完全可观测 —— 这恰恰就是 delay 需要上调的场景。
    旧实现把 `"有"` 当日期做字符串比较（`"有" >= "2026-09-14"` 恒真）→
    恒返回 0，自动标定永远不会触发。

    ★★ 2026-09-17 收紧两个口径（都会**误推高 delay**，进而让取数窗口上界后退、
       最新几天永远不抓）：
         · `mode == "dry_run"` 的记录**不计** —— 干跑把闸门预算压成 0，
           必然"超时"，那是人为制造的 ready=False；
         · `probe_ok is False`（探测失败/超时/取不到实体）**不计** ——
           它只说明"没看清"，不说明"服务端没有"。实测 `stock_cyq_chips`
           就是因为代表实体取到退市股而每一轮 probe 失败。
       只有"闸门真正跑过 + 探测成功 + 确认没数据"才算缺席。
    """
    hist = _load(paths.DELAY_HISTORY, {})
    n = 0
    for rec in reversed(list(hist.values())):
        if str(rec.get("mode") or "ready") == "dry_run":
            break
        v = (rec.get("tables") or {}).get(name) or {}
        if v.get("probe_ok") is False:
            break
        if v.get("ready") is False:
            n += 1
        else:
            break
    return n


def resolve(name: str, declared: int, cal: list[str], T: str | None,
            autobump_after: int = 3, max_extra: int = 2) -> tuple[int, str]:
    """最终用于闸门判据的 delay。返回 (delay, 说明)。

    两条独立证据，都只上不下（宁可多等一个交易日）：
      ① `delay_obs` 的众数 > 声明值（需要"服务端最新日期"这种强证据）；
      ② **连续 N 轮 ready=False** —— 稠密表也能拿到的弱证据，
         用来终结"某张表开始恒滞后 → 闸门每天白等满 4 小时"的死循环。
    """
    if T is None:
        return declared, "无 T，用声明值"
    obs = observed_delay(name)
    if obs is not None and obs > declared:
        # 连续多少次观测到更大的 delay？够就采用（保守方向，只上不下）
        hist = _load(paths.DELAY_HISTORY, {})
        streak = 0
        for rec in reversed(list(hist.values())):
            v = (rec.get("tables") or {}).get(name) or {}
            d = v.get("delay_obs")
            if isinstance(d, int) and d > declared:
                streak += 1
            else:
                break
        if streak >= autobump_after:
            return obs, (f"★ 连续 {streak} 天观测到 delay={obs} > 声明 {declared}，"
                         f"已自动采用（建议人工更新 frequency.yaml）")
        return declared, (f"观测到 delay={obs}（连续 {streak} 天，≥{autobump_after} 天才采用），"
                          f"暂用声明值")

    streak = absent_streak(name)
    if streak >= autobump_after:
        # ★★ 2026-09-17 加**上限**：只上不下 + 无上限 = 一旦探针侧出问题
        #   （如代表实体取错），delay 会每晚 +1，而 `data_hi = T - delay` 是取数
        #   窗口上界 → **最新的 N 天永远不会被请求**，且报告仍显示 ✔（静默）。
        #   封顶后，"一直探测不到"会变成一个**要人看**的告警，而不是继续自伤。
        if declared >= max_extra:
            return declared, (f"✖ 连续 {streak} 轮在 expected 那天探测不到该表，"
                              f"但 delay 已达自动上调上限（声明 {declared}，"
                              f"最多 +{max_extra}）—— **不再自动上调**。"
                              f"这更像探针/接口层面的问题，不是上游晚发布："
                              f"查 `main.py gate` 与 `main.py monitor`")
        return declared + 1, (f"★ 连续 {streak} 轮在 expected 那天探测不到该表，"
                              f"按 delay={declared + 1} 处理（上限 +{max_extra}，"
                              f"下轮再观察；建议人工更新 frequency.yaml）")
    if obs is not None and obs < declared:
        return declared, f"观测 {obs} < 声明 {declared}（声明偏保守，无害，不自动下调）"
    return declared, "与声明一致"


def delay_from_server_max(cal: list[str], T: str, server_max: str | None) -> int | None:
    """由"服务端该表最新日期"反推它落后 T 几个交易日。"""
    if not server_max or not T:
        return None
    # ★★ `server_max` 来自 `TableProbe.actual`，而它在**稠密表探测成功**时是
    #    字面量 `"有"`（probe.py），失败时是 `"未知"` / `"不适用"`。
    #    旧实现只挡了 `(None, "无", "—")`，于是 `"有"` 落到下一行的字符串比较：
    #    `"有" >= "2026-09-14"` —— `ord("有")=26377 ≫ ord("2")=50` → **恒为真**
    #    → 对所有稠密表恒返回 0 → `delay_history.json` 里 19/20 张表
    #    `server_max="有" / delay_obs=0`，自动标定形同虚设（实测）。
    if not _is_date(server_max):
        return None
    if server_max >= T:
        return 0
    if T not in cal or server_max not in cal:
        return None
    return cal.index(T) - cal.index(server_max)
