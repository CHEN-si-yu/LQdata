"""运行报告 —— 汇总 + 异常判定 + 计时。

用户 2026-09-14 要求：「记录好每一个数据接口的更新时间用时，最终的总用时等」。
所以每个数据集都记 4 个量：新增行数 / 请求数 / 重试数 / **该数据集耗时**，
外加全局总耗时与各阶段耗时。

异常判定（报告的核心价值，对应旧工程 TASKS.md §六(c) 的"有效性"要求）：
    · 日频表新增 0 行 → 报警（稀疏表按 nonzero_ratio 豁免）
    · 重试率 > 2% → 警告（旧工程实测基线口径）
    · 单日 MD5 与台账不符 → 🔴 上游修改了历史数据
"""
from __future__ import annotations

import json
from datetime import datetime

from .. import paths
from ..core import state
from .strategies import Result


def build(results: list[Result], ctx, phases: dict | None = None,
          revisions: list[dict] | None = None,
          gate_info: dict | None = None, dry_run: bool = False,
          test_mode: bool = False) -> dict:
    """组装报告 dict（供 viz / HTML / run_log 共用）。"""
    groups: dict[str, list[dict]] = {}
    freq_of = {}
    for ds in [r for r in _all_ds()]:
        freq_of[ds.name] = ds.freq
    for r in results:
        f = freq_of.get(r.name, "other")
        g = ("日频" if f in ("daily_full", "daily_sparse") else
             "分钟/周月" if f in ("minute", "weekly_monthly", "monthly") else
             "季频/不定期/快照")
        groups.setdefault(g, []).append({
            "name": r.name, "rows": r.rows, "requests": r.requests,
            "retries": r.retries, "empty_retries": getattr(r, "empty_retries", 0),
            "seconds": round(r.seconds, 1),
            "status": r.status, "note": r.note})

    # --as-of 是测试模式：那些日期本地本来就有，新增 0 行是**预期**，不该报警。
    # ★ 但只豁免"0 行"这一档 —— 部分失败（⚠️）与上游改历史（🔴）必须照报，
    #   否则官方推荐的验证口径会变成"什么都看不到"的静默模式。
    alerts = [] if dry_run else _judge(results, revisions or [],
                                       zero_rows_noise=not test_mode,
                                       T=getattr(ctx, "T", None),
                                       cal=getattr(ctx, "cal", None))
    if not dry_run and gate_info and gate_info.get("timed_out") and gate_info.get("blocking"):
        alerts.append("⚠️ 闸门超时，以下接口本轮未维护：" + ", ".join(gate_info["blocking"]))
    # ★★ 2026-09-17：把「要人处理的」和「只是记录」的分开 —— **退出码只看前者**。
    #   ℹ️ 档是记录性的（上游撤数 / 已知厂商洞 / 厂商晚间重算后我们自己核对过），
    #   一条都不需要动手；✘/⚠️/🔴 才需要。
    #   旧实现 `cmd_run` 用 `bool(alerts)` 定退出码 → 良性记录也让 rc=1，
    #   watchdog 与 `&&` 链无法区分"要处理"和"只是记录"（实测一次 62 条假 ℹ️）。
    actionable = [a for a in alerts if not a.startswith("ℹ️")]
    plans = [{"name": r.name, "note": r.note} for r in results if dry_run and r.note]
    total_sec = sum(r.seconds for r in results)
    req = sum(r.requests for r in results)
    ret = sum(r.retries for r in results)
    emp = sum(getattr(r, "empty_retries", 0) for r in results)

    daily = [r for r in results if freq_of.get(r.name) in ("daily_full", "daily_sparse")]
    ok_daily = [r for r in daily if r.rows > 0]
    if dry_run:
        verdict = f"🧪 dry-run：{len(daily)} 张日频表的执行计划已列出（未写任何数据）"
    elif test_mode:
        verdict = (f"🧪 测试模式（--as-of {getattr(ctx,'T','?')}）：{len(daily)} 张日频表已重跑，"
                   f"新增 {sum(r.rows for r in daily):,} 行（幂等预期为 0）")
    else:
        verdict = (f"✅ {len(ok_daily)}/{len(daily)} 张日频表有新数据"
                   + (f"；{len(alerts)} 条需要关注" if alerts else "；无异常"))
    if not daily and not dry_run:
        verdict = "ℹ️ 本次没有跑日频表"

    return {
        "schema_version": 1,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "T": getattr(ctx, "T", None),
        "dry_run": dry_run,
        "seconds": round(total_sec, 1),
        "requests": req,
        "retries": ret,
        "empty_retries": emp,
        "groups": groups,
        "alerts": alerts,
        "alerts_actionable": actionable,     # ★ 退出码只看它（见上面注释）
        "plans": plans,
        "revisions": revisions or [],
        "gate": gate_info or {},
        "phases": phases or {},
        "verdict": verdict,
    }


def _all_ds():
    from .. import registry as R
    return R.enabled()


def _vanished_notes(r: Result) -> list[str]:
    """「历史数据消失」报告：本地有数据、服务端本次却给了 0 行的那些交易日。

    判据（**两层都要满足**，避免把"本地本来就没有"误报成消失）：
      ① 本次请求覆盖到了这天（`Result.server_counts` 的日期跨度之内）；
      ② 单日台账 `state/dayhash/<表>.parquet` 里这天**本地有行**，而服务端给了 0 行。

    用户 2026-09-15 口径：**本地数据一律保留，不删**；只打报告。
    """
    counts = getattr(r, "server_counts", None) or {}
    if not counts:
        return []
    from .. import registry as R
    try:
        ds = R.get(r.name)
    except KeyError:
        return []
    # ★★ 2026-09-17 修：**台账键 ≠ 抓取键时必须整个跳过**。
    #   `server_counts` 的键是抓取键（`ds.date_field`，见 strategies.run_range），
    #   而台账里存的是 `ds.ledger_date_field()`。两者是**完全不同的日期轴**时，
    #   `counts.get(台账日)` 几乎永远取不到 → 台账里每一天都被判成"服务端返回 0 行"。
    #   实测（`stock_holder_number`：抓 `end_date`、台账 `ann_date`，400 天窗口）：
    #   一次会刷 **62 条**假"上游撤数"，而 `viz` 只显示 alerts 前 20 条、
    #   🔴"上游改历史"又排在最后 → **真信号被假信号挤出屏幕**。
    #   跨轴的等价实现需要按 (抓取键→台账键) 做映射，收益不抵复杂度；
    #   这里的正确做法是"看不清就不说"。
    if ds.ledger_date_field() != ds.date_field:
        return []
    try:
        from . import dayhash as DH
        led = DH.load_ledger(r.name)
        if not len(led):
            return []
        days = [str(d)[:10] for d in led["date"].tolist()]
        rows = [int(x) for x in led["rows"].tolist()]
    except Exception:  # noqa: BLE001 —— 报告增强，绝不因此中断
        return []
    lo, hi = min(counts), max(counts)
    out: list[str] = []
    for d, n in zip(days, rows):
        if not (lo <= d <= hi) or n <= 0:
            continue
        if int(counts.get(d, 0)) == 0:
            hole = d in (getattr(ds, "known_holes", ()) or ())
            out.append(f"ℹ️ {r.name} {d}：服务端该日已返回 0 行（上游撤数）"
                       f"，本地保留 {n:,} 行**未删除**"
                       + ("（已在 registry._KNOWN_HOLES 登记）" if hole else "")
                       + "；上游恢复即自动更新")
    # ★ 同表同类提示最多 10 条 —— 上游一次性撤掉一整段时，
    #   几十条同类 ℹ️ 会把别的表的告警挤出控制台（见 viz.render_summary）。
    if len(out) > 10:
        extra = len(out) - 10
        out = out[:10] + [f"ℹ️ {r.name}：另有 {extra} 天同样是「服务端已返回 0 行」，"
                          f"本地数据均保留未删（详见 state/dayhash/）"]
    return out


def _judge(results: list[Result], revisions: list[dict],
           *, zero_rows_noise: bool = True, T: str | None = None,
           cal: list[str] | None = None) -> list[str]:
    """把「需要人关注的东西」变成 alerts —— **`cmd_run` 的退出码由它决定**。

    ★ 2026-09-15 补上 `⚠️` 这一档。此前只有 `✘` 会进 alerts，而"部分失败"
       （区间请求失败 / 批量实体部分失败 / dump 配额用尽 / 稀疏表空响应）
       全部记成 `⚠️` —— 于是 alerts 为空、verdict 写"无异常"、退出码 **0**。
       一个 dump 配额用尽、当天没补上的失败，机器可读的出口全是"成功"。
    """
    from . import probe as probe_mod

    out: list[str] = []
    for r in results:
        if r.status == "✘":
            out.append(f"✘ {r.name} 执行失败：{r.note}")
            continue
        if r.status == "⚠️":
            out.append(f"⚠️ {r.name} 部分失败（未完成，下轮应自动重试）：{r.note or '有请求失败，详见日志'}")
        # ★★ 2026-09-15 修：重试率里必须**剔除"空响应自旋"**。
        #    `expect_rows=True` 的接口返回空时，`client.call` 会自己重试最多 4 次
        #    （`stats["empty_retries"]`）—— 那是我们主动的探测行为，**不是**服务端抖动。
        #    实测：`stock_report_rc` 一轮 8 个请求 / 4 次重试 = "50% 重试率"，
        #    其中 4 次全是空响应自旋（那天（09-14）本来就没有研报，服务端返回空），
        #    一次真抖动都没有 —— 这条告警 100% 是假阳性，还把真信号淹了。
        net_retries = max(0, r.retries - getattr(r, "empty_retries", 0))
        if r.requests and net_retries / r.requests > 0.02:
            spin = getattr(r, "empty_retries", 0)
            out.append(f"⚠️ {r.name} 重试率 {net_retries/r.requests*100:.1f}% 超 2% 基线（服务端抖动？）"
                       + (f"　[已剔除 {spin} 次空响应自旋]" if spin else ""))
        if zero_rows_noise and r.rows == 0 and r.status == "✔":
            # ★ 「新增 0 行」本身**不是**异常：同一天跑第二遍、或闸门 partial
            #   跳过之后的重跑，数据本来就已经在了。旧实现只看行数，
            #   于是一次幂等重跑会刷出 20 条"有效性"告警（实测），
            #   真正的信号被淹没。
            #   改成**看数据到没到应到的日期**：本地水位 < T-delay 才是真问题。
            from .. import registry as R
            from ..core import state as _state
            try:
                ds = R.get(r.name)
            except KeyError:
                continue
            if ds.freq not in ("daily_full", "daily_sparse") or not ds.date_field:
                continue
            exp = None
            if T and cal:
                exp = _state.shift_back(cal, T, int(ds.delay_days or 0))
            mx = _state.Manifest.load(ds.name).max_partition_date()
            # ★ 2026-09-15 晚（用户拍板）：期望日期落在**已知厂商洞**里时，
            #   这条不再是"要人去查上游/策略"的 ⚠️ —— 洞是厂商撤回造成的、
            #   代码里已登记（registry._KNOWN_HOLES），尾部窗口每天照常重试。
            #   保留一行 ℹ️ 让它**看得见**，但不制造排查噪音。
            if exp and str(exp)[:10] in (getattr(ds, "known_holes", ()) or ()) \
                    and (not mx or str(mx)[:10] < exp):
                out.append(f"ℹ️ {r.name} 本地水位 {mx or '—'}（按 T={T} / delay={ds.delay_days} "
                           f"应到 {exp}）—— 该日期是**已知厂商洞**（上游撤回未补），"
                           f"非我方问题；尾部窗口每天自动重试")
            elif exp and (not mx or str(mx)[:10] < exp):
                out.append(f"⚠️ {r.name} 本轮新增 0 行，且本地只到 {mx or '—'}"
                           f"（按 T={T} / delay={ds.delay_days} 应到 {exp}）—— 检查上游/策略")
            elif not exp:
                ratio = probe_mod.trailing_nonzero_ratio(ds.name, ds.date_field)
                if ds.freq == "daily_full" or ratio >= 0.8:
                    out.append(f"⚠️ {r.name} 是日频表但本次新增 0 行（有效性问题，检查上游/策略）")

        # ★★ 2026-09-15 晚（用户交办）：「如果遇上了历史数据消失的情况，但是今天的 target
        #    数据反而存在，那么只需要把新数据合入；**不需要特意删除以前的数据**，
        #    但需要**给这个消失的数据打一个报告**。」
        #    这里就是那份报告：本地有、服务端这次却给了 0 行 = 上游撤数。
        #    ⚠️ 注意：数据**一律保留**（写盘路径只有 upsert，没有任何删除逻辑；
        #       要删只能人工跑 `main.py delete-day`）。
        out.extend(_vanished_notes(r))
    for rev in revisions:
        out.append(f"🔴 {rev['dataset']} {rev['date']} 的单日 MD5 与台账不符 —— "
                   f"上游修改了历史数据（变化列: {', '.join(rev.get('changed_cols') or []) or '整行'}）")
    return out


def save(report: dict, append_log: bool = True) -> None:
    paths.ensure_dirs()
    try:
        # ★ 2026-09-17：改原子写（tmp + os.replace）。旧实现是 `write_text` ——
        #   报告是"机器可读的出口"，写到一半被 kill 会留下半截 JSON，
        #   而 `load_last()` 只能返回空（等于这次运行的结论全丢）。
        state._atomic_json(paths.LAST_REPORT, report)
    except OSError:
        pass
    if append_log:
        try:
            slim = {k: report[k] for k in
                    ("finished_at", "T", "seconds", "requests", "retries",
                     "empty_retries", "verdict")
                    if k in report}
            # ★ 2026-09-17：趋势图与日报的重试率口径要对齐 —— 记下两档告警条数，
            #   免得"趋势图打 ⚠、日报说无异常"（见 viz.render_trend）。
            slim["alerts"] = len(report.get("alerts") or [])
            slim["alerts_actionable"] = len(report.get("alerts_actionable") or [])
            with open(paths.RUN_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(slim, ensure_ascii=False) + "\n")
        except OSError:
            pass


def load_last() -> dict:
    try:
        v = json.loads(paths.LAST_REPORT.read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}
