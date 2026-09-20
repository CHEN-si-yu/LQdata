"""闸门 —— 「上游到齐了没有？」在开跑之前先问一遍。

用户 2026-09-14 的要求（原话要点）：
    每天执行 main.py，它会检查所有的日频数据是否都更新到了最新的数据，
    如果存在一个还没有，那么进行等待 5min 再次重新申请，直至所有的日频数据
    都更新，才正式启动当日的数据更新逻辑。
    「如果是确定了 delay 窗口的，在更新 T 时间的时候，普通数据是 check 服务器
     T 时间的数据是否存在，而这一类数据是 check T-delay 这个数据是否存在。」

判据（核心）：
    T        = **服务端**已发布的最新交易日（探测得来，不看本地）
    expected = 交易日历里 T 往前数 delay_days 个交易日
    ready    = **服务端**在 expected 那天有这张表的数据
                （sparse 表退化为 10 天窗口判据，见 probe.probe_table_ready）

★ 设计原则：闸门只决定「何时开始」，正确性由 planner 保证。
   planner **永远无条件重抓尾部窗口**，所以：
     闸门假阴性（该跑没跑）→ 白等 5 分钟，无损失
     闸门假阳性（不该跑跑了）→ 拿回空、记 suspect、下轮重试，无损失
   这让闸门可以便宜、粗略、有超时 —— 这是与旧工程闸门最重要的区别
   （旧闸门是正确性关键路径，一旦永久 exit 3 整个日更就废了）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .. import paths, registry as R
from ..core import progress as P
from ..core import state
from ..core.client import Client
from . import delay as delay_mod
from . import probe as probe_mod
from . import viz


@dataclass
class GateResult:
    ready: bool
    T: str | None
    rows: list[dict] = field(default_factory=list)
    timed_out: bool = False
    elapsed: float = 0.0
    observations: list[dict] = field(default_factory=list)
    note: str = ""

    @property
    def blocking(self) -> list[dict]:
        return [r for r in self.rows if not r["ready"]]


def _plan_targets(cfg: dict, cal: list[str], T: str) -> list[tuple[R.DS, str, str, int]]:
    """闸门要看的每一张表 → (ds, tier, expected, resolved_delay)。

    每张日频/分钟级表今天应该到哪天 = `T - 解析后的 delay`（阻塞判据）；
    `watch` 档没有"该到哪天"的确定含义（月频/季频/不定期/周月线），expected 只填 T 占位。

    ★★ 2026-09-15 深夜扩表（用户拍板「闸门要等所有数据都达到预期状态才启动」）：
      改造前只遍历 `R.daily_tables()`（20 张），其余 22 张 enabled 表**完全不进闸门** ——
      于是"跑批开始时非日频还没发布"这件事没人等，只能事后补一轮（当晚就是 21:14 一轮
      + 23:41 补一轮）。现在按 `R.gate_tier` 分档，**只有 `wait` 档参与阻塞**。
    """
    autobump = int(cfg.get("gate", {}).get("autobump_after_days", 3))
    out: list[tuple[R.DS, str, str, int]] = []
    for ds in R.gate_tables():
        tier = R.gate_tier(ds)
        if tier == "watch":
            out.append((ds, tier, T, int(ds.delay_days or 0)))
            continue
        d, _why = delay_mod.resolve(ds.name, ds.delay_days, cal, T, autobump,
                                    max_extra=int(cfg.get("gate", {}).get(
                                        "delay_autobump_max", 2)))
        exp = state.shift_back(cal, T, d) or T
        out.append((ds, tier, exp, d))
    return out


def run_gate(client: Client, cfg: dict, cal: list[str], runner: P.Runner,
             log=print) -> GateResult:
    """跑闸门：探测 → 判据 → 未到齐就等 5 分钟重来，直到到齐或超时。"""
    g = cfg.get("gate", {})
    interval = float(g.get("interval_seconds", 300))
    budget = float(g.get("max_wait_hours", 4.0)) * 3600
    publish_ratio = float(g.get("publish_ratio", 0.7))
    t0 = time.monotonic()
    attempt = 0
    # 观测档（T2）的探测结果要跨轮沿用 —— 它不阻塞，没必要每 5 分钟重探一遍
    watch_rows: dict[str, dict] = {}

    while True:
        attempt += 1
        elapsed = time.monotonic() - t0
        # T2 只在第 1 轮 + 每 5 轮探一次（省请求；它们本来就不阻塞）
        probe_watch_now = (attempt == 1) or (attempt % 5 == 1)

        # ---- 1) 定 T
        ev = probe_mod.resolve_T(client, cal, cfg)
        T = ev.T
        if T is None:
            rows = [{"name": ds.name, "tier": R.gate_tier(ds), "delay": ds.delay_days,
                     "expected": "—", "actual": "—",
                     "ready": R.gate_tier(ds) != "wait", "note": "T 未确定"}
                    for ds in R.gate_tables()]
            viz.print_gate(attempt, elapsed, budget, None, rows, interval, ev.note)
            if elapsed + interval > budget:
                return GateResult(False, None, rows, True, elapsed, note=ev.note)
            time.sleep(interval)
            continue

        # ---- 2) 逐表判据
        rows: list[dict] = []
        observations: list[dict] = []
        for ds, tier, exp, d in _plan_targets(cfg, cal, T):
            # ---- 观测档（T2）：只探测与展示，**永不阻塞**（死锁论证见 registry.gate_tier）
            if tier == "watch":
                if probe_watch_now or ds.name not in watch_rows:
                    wtp = probe_mod.probe_table_watch(client, ds, cal, T)
                    watch_rows[ds.name] = {"name": ds.name, "tier": tier, "delay": d,
                                           "expected": T, "actual": wtp.actual,
                                           "ready": True, "note": wtp.note}
                rows.append(watch_rows[ds.name])
                continue

            # ---- 等待档（T1）：阻塞判据
            #   分钟级表用"完整性比例"判据 —— 盘中抓到的半份不该算就绪。
            if ds.freq == "minute":
                tp = probe_mod.probe_ready_by_ratio(client, ds, exp, cal, publish_ratio)
            else:
                tp = probe_mod.probe_table_ready(client, ds, exp, cal)
            # ★ 已知厂商洞（用户 2026-09-15 拍板）：期望日期正好落在"上游撤回过、
            #   迟迟不补"的那一天 → **不阻塞闸门**（否则整轮要白等到预算耗尽，
            #   实测 dc_daily@09-14 会把 20:33 起跑的那轮拖到 00:33），但必须**看得见**：
            #   闸门表里标 ⏭ 且带 note，尾部窗口每天照常重试它（发布即自动补回）。
            #   ⚠️ 只在"服务端**确实没有**这天"时才算洞 —— 上游若把数据补回来了，
            #      这一行要照常显示 ✔ 并正常抓取（否则洞被填上了我们也不知道）。
            hole = (not tp.ready) and str(exp)[:10] in (getattr(ds, "known_holes", ()) or ())
            rows.append({"name": ds.name, "tier": tier, "delay": d, "expected": exp,
                         "actual": tp.actual,
                         "ready": tp.ready or hole,
                         "hole": hole,
                         # ★ 2026-09-17：探测失败（unknown）不阻塞，但要在闸门表里
                         #   一眼看出来 —— 它是"看不清"，不是"已到齐"。
                         "unknown": bool(getattr(tp, "unknown", False)),
                         "note": ("⏭ 已知厂商洞：上游撤回过这一天（registry._KNOWN_HOLES），"
                                  "不阻塞闸门；尾部窗口每天照常重试，发布即自动补回"
                                  if hole else tp.note)})
            # 顺手记 delay 观测（零额外请求：用已探到的 actual）
            # ★ `tp.actual` 对稠密表是 `"有"/"无"` 这种**状态字符串**，不是日期
            #   （稀疏表分支才回真实日期）。只有真是 YYYY-MM-DD 时才拿它反推 delay；
            #   同时把 `ready` 记下来 —— 那是稠密表唯一可用的滞后信号
            #   （见 delay.absent_streak）。
            import re as _re
            is_date = bool(_re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(tp.actual or "")))
            if is_date:
                actual_date = str(tp.actual)[:10]
            elif tp.actual in ("有", "无"):
                # 探测到的是"有没有"，反推服务端最新日期 = expected（有）或不可知（无）
                actual_date = exp if tp.actual == "有" else None
            else:
                actual_date = None
            obs = delay_mod.delay_from_server_max(cal, T, actual_date)
            observations.append({"name": ds.name,
                                 "server_max": actual_date,
                                 "server_max_raw": tp.actual,
                                 # ★★ 2026-09-17：把"探测本身是否成功"记下来。
                                 #   `absent_streak` 只该统计"探测成功且确认无数据"，
                                 #   否则探针坏掉（unknown）会一路推高 delay（见 delay.py）。
                                 "probe_ok": not bool(getattr(tp, "unknown", False)),
                                 "delay_obs": obs, "ready": tp.ready})

        # ★ 只有 wait 档参与阻塞 —— watch 档恒 ready=True（见 registry.gate_tier 的死锁论证）
        wait_rows = [r for r in rows if r.get("tier") == "wait"]
        ready = all(r["ready"] for r in wait_rows)
        note = ev.note
        if ev.cross_ok is False:
            note = (note + f"  ⚠️ 交叉验证不一致：/ths/hot 最新 {ev.cross_check} ≠ T={T}").strip()
        viz.print_gate(attempt, elapsed, budget, T, rows, interval, note)

        if ready:
            # 记录本次 delay 观测（mode="ready" = 闸门真正通过，见 delay.record_observation）
            delay_mod.record_observation(T, observations, mode="ready")
            behind = [r["name"] for r in rows
                      if r.get("tier") == "watch" and "落后于服务端" in (r.get("note") or "")]
            msg = (f"✔ 等待档 {len(wait_rows)} 张已全部到齐"
                   f"（T={T}，检查 {attempt} 次 / 用时 {P.fmt_duration(elapsed)}）")
            if behind:
                msg += f"；⚠️ 另有 {len(behind)} 张观测档表落后于服务端（不阻塞）：{', '.join(behind)}"
            runner.note(msg)
            return GateResult(True, T, rows, False, elapsed, observations)

        bad = [r for r in wait_rows if not r["ready"]]
        log(f"⏳ 第 {attempt} 次检查：{len(bad)}/{len(wait_rows)} 张等待档表还没到齐"
            f"（已等 {elapsed/60:.1f} 分钟，预算 {budget/3600:.1f} 小时）")

        if elapsed + interval > budget:
            mode = str(g.get("on_timeout", "partial"))
            # ★ 2026-09-17：干跑（--dry-run 把预算压成 0）必然走到这里，
            #   它的 ready=False 是**人为制造**的，不能拿去喂 delay 自动标定。
            obs_mode = ("dry_run" if g.get("dry_run")
                        else "strict" if mode == "strict" else "timeout")
            delay_mod.record_observation(T, observations, mode=obs_mode)
            if mode == "strict":
                runner.note(f"✖ 闸门等待超时：仍有 {len(bad)} 张表未到齐，**不开始增量**（on_timeout=strict）")
                return GateResult(False, T, rows, True, elapsed, observations, note="strict 超时")
            runner.note(f"⚠️ 闸门等待超时（{budget/3600:.1f} 小时）：{len(bad)} 张表未到齐，"
                        f"按 on_timeout=partial **只跑已到齐的表**，未到齐的下轮自动补：")
            for r in bad:
                runner.note(f"     ⊘ 跳过 {r['name']:32} 应到 {r['expected']} 实到 {r['actual']}")
            return GateResult(False, T, rows, True, elapsed, observations,
                              note=f"partial 超时，跳过 {len(bad)} 张")

        log(f"     等 {interval/60:.0f} 分钟后再查一次…")
        time.sleep(interval)


def ready_names(result: GateResult) -> set[str] | None:
    """已就绪的数据集名集合。None = 全都可跑（闸门通过）。"""
    if result.ready:
        return None
    return {r["name"] for r in result.rows if r["ready"]}
