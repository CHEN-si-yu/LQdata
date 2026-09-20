"""编排 —— 把整个每日流程串起来。

流程（用户要求的顺序）：
    ① 抢跨项目锁（两边写同一份 data/state，共享盘没有文件锁）
    ② 更新交易日历到 T+30（**必须第一步** —— 闸门的候选日枚举依赖它）
    ③ 闸门：探测服务端，未到齐就每 5 分钟重查，直到全到或超时
    ④ 跑**日频表**（20 张）
    ⑤ 跑**其他频率**（周月/月频/季频/不定期/快照/分钟）
    ⑥ 单日 MD5 台账：对尾部窗口重算指纹并比对 → 揪出"上游改了历史"
       ★ 2026-09-17 起还按 `dayhash.sample_old_days` 从**台账已有日期**里
         轮换抽查若干历史日（此前该配置是死的，历史那部分完全没有保护）
    ⑦ 报告 + 落日志（含每个接口的耗时与总耗时）
"""
from __future__ import annotations

import time
from datetime import datetime

from .. import registry as R
from ..core import progress as P
from ..core import state
from ..core.client import Client
from . import calendar as cal_mod
from . import dayhash as dayhash_mod
from . import gate as gate_mod
from . import report as report_mod
from . import strategies as strat
from . import viz

# 执行顺序：先小后大，先快后慢（让早期失败早暴露）
ORDER = [
    # ---- 快照/基础（很小，先跑，后续都要用）
    "stock_list", "tdx_blocks", "dc_blocks",
    # index_ths_sector_categories 已于 2026-09-19 删除（名单已固化进 conf/ths_index_codes.txt）
    # ---- range（date 轴，每表 1~3 请求）
    "stock_daily", "stock_cyq_perf", "stock_adj_factor", "stock_finance", "index_daily",
    "stock_main_fund_flow", "stock_margin_detail",
    "stock_limit_up", "stock_limit_list", "stock_st_info",
    # stock_dc_block_fund_flow 已于 2026-09-19 彻底删除（厂商可供数据到 2025-02-26，
    # 本地只覆盖到 2026-01-05 起，用户判定时间覆盖太短 → 不再更新）
    # stock_ths_block_fund_flow 已于 2026-09-19 彻底删除（同上；本地 2025-01-02 起仅 1.7 年）
    "stock_holder_number", "stock_pledge_stat",
    # stock_kline 已于 2026-09-16 彻底删除（周/月线语义错误 + 零额外信息）
    # ---- per_date（date 轴，每天 1 请求）
    "dc_daily", "stock_suspension", "stock_adj_factor_changes",
    "stock_dragon_tiger", "stock_top_list",
    # ths_hot 已于 2026-09-19 彻底删除（厂商无更早数据，本地仅 2.6 年，用户判定停更）
    "stock_market_distribution_history",
    # ---- per_stock（entity 轴，全量重拉但批量）
    "stock_financial_indicator", "stock_income", "stock_balancesheet",
    "stock_cashflow", "stock_forecast",
    # ---- per_entity（both 轴；窗口/批量见 registry 的调度元数据）
    "tdx_daily", "index_ths_daily", "tdx_minute",
    # tdx_block_stocks / index_ths_constituent_stocks 已于 2026-09-19 删除（快照表）
    "stock_cyq_chips",
    # ---- dump（最大的一张，靠 1 个请求吃饱）
    "stock_daily_dump", "stock_history_5min",
]


def _ordered(ds_list: list[R.DS]) -> list[R.DS]:
    idx = {n: i for i, n in enumerate(ORDER)}
    return sorted(ds_list, key=lambda d: (idx.get(d.name, 999), d.name))


def run(cfg: dict, runner: P.Runner, client: Client, *,
        dry_run: bool = False, only: set[str] | None = None,
        skip_gate: bool = False, ready: set[str] | None = None,
        as_of: str | None = None, report_backfill: bool = False,
        persist_report: bool = True) -> dict:
    """跑一次每日增量。

    `as_of`：把这一天当作 T（而不是探测出来的最新交易日）。
    ★ 用途（用户 2026-09-14 的建议）：用**已经确定完整发布**的日线做测试 ——
      今天（09-14）的数据还在波动（实测 dc_daily 从 1031 行变 0 行），
      而 09-11 的数据必然齐全。用 `--as-of 2026-09-11` 跑，尾部窗口会落在
      [09-04, 09-11]，那些数据本地都有，upsert 幂等 → **零新增**，
      可以在不引入新数据的前提下把整条流水线验证一遍。
    """
    phases: dict[str, float] = {}
    maintenance_alerts: list[str] = []
    t_all = time.monotonic()

    # ---- ② 日历（必须第一步）
    t0 = time.monotonic()
    runner.note("① 更新交易日历（T+30）…")
    cinfo = cal_mod.update_calendar(client, cfg, write=not dry_run)
    if cinfo.get("ok"):
        runner.note(f"   ✔ 日历 {cinfo.get('before_max')} → {cinfo.get('after_max')}"
                    f"（未来 {cinfo.get('future_days')} 天，共 {cinfo.get('total')} 行）")
    else:
        runner.note(f"   ⚠️ 日历更新无返回：{cinfo.get('note')}（改用自愈日历）")
        maintenance_alerts.append(f"⚠️ basic_calendar 更新失败：{cinfo.get('note')}")
    cal = state.effective_calendar()
    phases["日历"] = time.monotonic() - t0

    # ---- ③ 闸门
    gate_info: dict = {}
    T: str | None = None
    t0 = time.monotonic()
    if skip_gate or as_of:
        # --as-of 是测试模式：直接用它当 T，不跑闸门（否则会为当天未发布的表白等 4 小时）
        from . import probe as probe_mod
        ev = probe_mod.resolve_T(client, cal, cfg)
        T = ev.T
        runner.note(f"② 跳过闸门（{'--as-of 测试模式' if as_of else '--no-wait'}），"
                    f"探测到的 T = {T}")
    else:
        runner.note("② 闸门：探测服务端…")
        gres = gate_mod.run_gate(client, cfg, cal, runner, log=runner.note)
        T = gres.T
        gate_info = {"ready": gres.ready, "T": gres.T,
                     "blocking": [r["name"] for r in gres.blocking],
                     "timed_out": gres.timed_out, "seconds": round(gres.elapsed, 1),
                     "note": gres.note}
        # ★ on_timeout=strict 必须真的"一个表都不跑"。
        #   旧实现里 gate 只是把 note 换成 "strict 超时"，返回结构不变，
        #   而 runner 从不消费 `gres.ready`（只消费 ready_names），于是 strict
        #   被静默降级成 partial —— 使用者以为今天不会更新，实际更新了一半。
        if gres.timed_out and str(cfg.get("gate", {}).get("on_timeout", "partial")) == "strict":
            runner.note("✖ 闸门等待超时（on_timeout=strict）—— 按配置**不开始任何增量**。")
            rep = report_mod.build([], None, phases, gate_info=gate_info)
            rep["verdict"] = "✖ 闸门 strict 超时：未做任何更新"
            rep["alerts"] = [f"✖ 闸门 strict 超时：仍有 {len(gres.blocking)} 张日频表未到齐，未做任何更新"]
            if persist_report:
                report_mod.save(rep)
            return rep
        if ready is None:
            ready = gate_mod.ready_names(gres)
        if gres.timed_out:
            runner.note(f"   ⚠️ partial 超时：只跑已到齐的 {len(ready or [])} 张**日频**表"
                        f"（非日频表不受闸门限制，照常跑）")
    phases["闸门"] = time.monotonic() - t0

    if T is None:
        runner.note("✖ 拿不到上游最新交易日（T），无法开跑。")
        rep = report_mod.build([], None, phases, gate_info=gate_info)
        rep["verdict"] = "✖ T 未确定，未做任何更新"
        # ★ 这一行必须进 alerts：`cmd_run` 用 `bool(rep["alerts"])` 决定退出码。
        #   旧实现只有一个 verdict 字符串，alerts 仍是 [] → **一行数据都没更新的
        #   一轮，退出码却是 0**，watchdog / `&&` 链会当成成功。
        rep["alerts"] = ["✖ 拿不到上游最新交易日（T），本轮未做任何更新"]
        if persist_report:
            report_mod.save(rep)
        return rep

    if as_of:
        # ★ 用户建议的测试模式：用已确认完整发布的日期当 T（零新增、可反复跑）
        if as_of not in cal:
            runner.note(f"   ⚠️ --as-of {as_of} 不在交易日历里，仍按给定日期使用")
        runner.note(f"   ★ --as-of {as_of}：强制把 T 设为该日（探测到的实际 T={T}）")
        T = as_of
        if ready is not None:
            ready = None        # 测试模式不做闸门过滤
        gate_info["as_of"] = as_of

    ctx = strat.Ctx(client=client, cfg=cfg, cal=cal, T=T,
                    runner=runner, dry_run=dry_run, only=only, ready=ready,
                    report_backfill=report_backfill)

    # ---- ④ 日频表
    #   ⚠️ DS 是带 dict/list 字段的 dataclass → 不可哈希，按**名字**比对
    daily_names = {d.name for d in R.daily_tables()}
    daily = _ordered([d for d in R.daily_tables() if ctx.selected(d)])
    # ★ 非日频表**不受闸门 ready 过滤**：闸门只探日频表，`ready` 里永远不会有
    #   非日频表的名字。用同一个判据过滤，等于"闸门一 partial，非日频全灭"。
    other = _ordered([d for d in R.enabled()
                      if d.name not in daily_names and ctx.selected(d, ignore_ready=True)])
    # Consumers such as chips and board prices must see this run's fresh lists.
    foundations = [d for d in other if d.mode == "snapshot" and d.name != "basic_calendar"]
    other = [d for d in other if d not in foundations]
    results: list[strat.Result] = []
    t0 = time.monotonic()
    runner.note(f"③ 基础名单 {len(foundations)} 张（先于依赖它们的日频表）")
    results += _run_group(foundations, ctx, runner, dry_run)
    phases["基础名单"] = time.monotonic() - t0
    runner.note(f"③ 日频表 {len(daily)} 张"
                + (f"（闸门 partial：只跑已到齐的）" if ready is not None else "")
                + ("  [dry-run]" if dry_run else ""))

    t0 = time.monotonic()
    results += _run_group(daily, ctx, runner, dry_run)
    phases["日频"] = time.monotonic() - t0

    # ---- ⑤ 其他频率
    runner.note(f"④ 其他频率 {len(other)} 张")
    t0 = time.monotonic()
    results += _run_group(other, ctx, runner, dry_run)
    phases["其他频率"] = time.monotonic() - t0

    # ---- ⑥ 单日 MD5 台账
    revisions: list[dict] = []
    t0 = time.monotonic()
    if cfg.get("dayhash", {}).get("enabled", True) and not dry_run:
        runner.note("⑤ 单日 MD5 台账：对尾部窗口重算指纹并与台账比对…")
        lo, hi = ctx.tail()
        days_trading = cal_mod.trading_days_between(cal, lo, hi)
        # ★★ 2026-09-16：窗口**按表决定**，不再是全局一份 —— 见 DS.ledger_calendar_days /
        #   ledger_lag_days。多数表仍走原来的「交易日窗口、滞后 0」，行为逐格不变；
        #   只有声明了的表（如 stock_holder_number）用日历日并让比对滞后 N 天。
        def _days_for(ds) -> list[str]:
            if ds.ledger_calendar_days:
                d = cal_mod.calendar_days_between(lo, hi)   # 含周末（公告日有周末）
            else:
                d = list(days_trading)
            return d[: len(d) - ds.ledger_lag_days] if ds.ledger_lag_days > 0 else d
        n_chk = 0
        n_sample = 0
        # ★★ 2026-09-17 三处改动：
        #   ① `dayhash.sample_old_days` 之前是**死配置**（全工程无人读）——
        #      用户的核心诉求就是"防止厂商偷改历史"，而每轮只重算尾窗（~6 天），
        #      历史部分完全没有保护。现在按配置从**台账已有日期**里轮换抽查。
        #   ② 本轮 ✘/⚠️ 的表**只比对、不写台账**：它的数据可能只落了一半，
        #      拿半成品当基线，等补齐那轮会误报"上游修改了历史数据"（归因错位）。
        #   ③ 逐表 try/except：台账读坏不该让整轮跑批在"抓取全做完之后"崩掉 ——
        #      那会让报告与 run_log 双双缺失（最难排查的一种失败）。
        sample_n = int(cfg.get("dayhash", {}).get("sample_old_days", 0) or 0)
        status_of = {r.name: r.status for r in results}
        for ds in R.enabled():
            if not dayhash_mod.trackable(ds):      # ★ 判据统一在 dayhash.trackable（原先三处各写一遍）
                continue
            if only and ds.name not in only:
                continue
            days = _days_for(ds)
            extra = dayhash_mod.sample_old_days(ds, sample_n, exclude=set(days))
            if extra:
                days = days + extra
                n_sample += len(extra)
            st = status_of.get(ds.name, "✔")
            try:
                ch, n = dayhash_mod.update_and_compare(ds.name, ds, days, write=(st == "✔"))
            except Exception as exc:  # noqa: BLE001
                runner.note(f"   ⚠️ {ds.name} 台账更新失败（**不影响已抓到的数据**）："
                            f"{str(exc)[:90]}")
                maintenance_alerts.append(f"⚠️ {ds.name} 台账校验失败：{str(exc)[:90]}")
                continue
            revisions += [c.as_dict() for c in ch]
            n_chk += n
        sample_tag = f"，另抽查历史 {n_sample} 天" if n_sample else ""
        if revisions:
            runner.note(f"   🔴 检测到 {len(revisions)} 处上游修改历史（详见报告）")
        else:
            runner.note(f"   ✔ {n_chk} 个「数据集×日」指纹全部一致"
                        f"（尾部窗口 {lo}~{hi}{sample_tag}）")
    phases["MD5台账"] = time.monotonic() - t0

    # ---- ⑦ 报告
    rep = report_mod.build(results, ctx, phases, revisions, gate_info,
                           dry_run=dry_run, test_mode=bool(as_of))
    rep["seconds"] = round(time.monotonic() - t_all, 1)
    if not dry_run:
        rep.setdefault("alerts", []).extend(maintenance_alerts)
        rep.setdefault("alerts_actionable", []).extend(maintenance_alerts)
    if persist_report:
        report_mod.save(rep, append_log=not dry_run)
    return rep


def _run_group(ds_list: list[R.DS], ctx: strat.Ctx, runner: P.Runner,
               dry_run: bool) -> list[strat.Result]:
    out: list[strat.Result] = []
    total = len(ds_list) or 1
    for i, ds in enumerate(ds_list, 1):
        man = state.Manifest.load(ds.name)
        runner.note(f"[{i}/{total}] {ds.name}  ({ds.mode}/{ds.pull_axis})  {man.summary()}")
        res = strat.run_one(ds, man, ctx)
        if not dry_run:
            if res.status == "⊘":
                # ★★ 「跳过」不等于「跑过」，**绝不能刷新 finished_at**。
                #    `_period_skip` 用 `last_run.finished_at` 距今多少天来决定
                #    "快照类表到期了没"。旧实现在这里无条件 finish_run，
                #    于是哪怕本次是 ⊘ 跳过，时间戳也被刷成"刚刚" →
                #    age ≈ 0 < snapshot_every_days 恒成立 → **该表永远跳过、再也不刷新**。
                #    实测受害：`index_weight`（30 天）冻在 2026-07-01、
                #    `tdx_block_stocks` / `index_ths_constituent_stocks`（7 天）同样。
                #    现在只记一笔"跳过了"，finished_at 保持上一次**真正跑完**的时刻。
                man.last_run = {**(man.last_run or {}),
                                "skipped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "skip_note": res.note}
            else:
                man.finish_run(**{k: v for k, v in res.as_dict().items()
                                  if k in ("rows", "requests", "retries", "status")},
                               seconds=round(res.seconds, 1))
            man.save()
        out.append(res)
        runner.note(f"     {res.status} {ds.name:34} 新增 {res.rows:>10,} 行 · "
                    f"请求 {res.requests:>4} · 用时 {P.fmt_duration(res.seconds)}"
                    + (f"\n        → {res.note}" if (dry_run or res.status != "✔") and res.note else ""))
    return out
