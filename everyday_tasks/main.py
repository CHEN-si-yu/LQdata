#!/usr/bin/env python
"""每日增量更新 · 唯一入口

    python /root/autodl-fs/everyday_tasks/main.py            # 全流程（默认）
    python .../main.py gate [--watch N]                      # 只跑闸门（零写入）
    python .../main.py run [--dry-run] [--only a,b] [--no-wait]
    python .../main.py doctor                                # 体检：注册表 / 接口连通性 / 批量能力
    python .../main.py status                                # 各数据集本地水位
    python .../main.py trend [--days 30]                     # 运行趋势
    python .../main.py backtest ...                          # 破坏性回测（转发 tools/）

设计：**独立于 `datadownload` 的爬虫框架** —— 不 import 它的任何模块，
只对齐它的数据格式（parquet 分区 + state JSON）。接口配置照抄在 registry.py。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_incremental import config as C  # noqa: E402

# Apply limits before imports that initialize numerical libraries.
if __name__ == "__main__":
    from data_incremental.core.resources import configure
    configure(C.load())

from data_incremental import paths, registry as R  # noqa: E402
from data_incremental.core import lock as lockmod  # noqa: E402
from data_incremental.core import progress as P  # noqa: E402
from data_incremental.core import state  # noqa: E402
from data_incremental.core.client import Client  # noqa: E402
from data_incremental.pipeline import calendar as cal_mod  # noqa: E402
from data_incremental.pipeline import gate as gate_mod  # noqa: E402
from data_incremental.pipeline import viz  # noqa: E402

log = logging.getLogger("daily")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def _new_client(cfg: dict) -> Client:
    return Client(cfg, log)


# ================================================================ doctor
def cmd_doctor(args, cfg: dict) -> int:
    """体检：注册表 vs 磁盘双向覆盖 + 拉取轴 + 接口连通性 + 批量能力实测。"""
    print("=" * 88)
    print("  每日增量工程 · 体检")
    print("=" * 88)

    disk = {p.name for p in paths.DATA_ROOT.iterdir() if p.is_dir()} if paths.DATA_ROOT.exists() else set()
    # ★ 只比对**启用**的数据集：`enabled=False` 的（index_history / stock_daily_adj /
    #   stock_daily_dump）本来就不落盘，永远报"磁盘无数据"只会制造恒定假告警。
    reg = {d.name for d in R.enabled()}
    print(f"\n【注册表 vs 磁盘】启用 {len(reg)} / 磁盘 {len(disk)} 个目录")
    only_reg, only_disk = sorted(reg - disk), sorted(disk - reg)
    if only_reg:
        print(f"  ⚠️ 注册了但磁盘无数据（{len(only_reg)}）：{', '.join(only_reg[:10])}")
    # ★ 已知的"磁盘有目录但不在启用清单里"的例外：`stock_daily_dump` 是 **dump 通道**
    #   （数据经 dump_bridge 落进 stock_history_5min，它自己的目录现在放**按日缓存**），
    #   报成"未注册"是假告警。
    _KNOWN_OFF = {d.name for d in R.all_ds() if not d.enabled}
    unknown = [n for n in only_disk if n not in _KNOWN_OFF]
    known_off = [n for n in only_disk if n in _KNOWN_OFF]
    if known_off:
        print(f"  ℹ️ 磁盘有目录但未启用（{len(known_off)}）：{', '.join(known_off[:10])}"
              f"　（如 stock_daily_dump = daily_dump 的按日缓存，见 `main.py dump-cache`）")
    if unknown:
        print(f"  ⚠️ 磁盘有数据但未注册（{len(unknown)}）：{', '.join(unknown[:10])}")
    if not only_reg and not only_disk:
        print("  ✔ 双向完全覆盖")

    from collections import Counter
    print("\n【拉取轴分布】（★ 按接口天然能力选策略的依据）")
    for ax, n in sorted(Counter(d.pull_axis for d in R.enabled()).items()):
        names = [d.name for d in R.by_axis(ax)]
        print(f"  {ax:9} {n:>2} 个: {', '.join(names[:6])}{' …' if len(names) > 6 else ''}")
    print(f"\n【日频表（闸门管）】{len(R.daily_tables())} 张")

    print("\n【本地水位】latest = manifest 的 max_date")
    print(f"  {'数据集':32} {'freq':13} {'delay':>5} {'latest':11} {'mode':11} {'axis':9} {'冗余':>5}")
    print("  " + "-" * 92)
    for d in R.enabled():
        man = state.Manifest.load(d.name)
        mx = man.max_partition_date() or "—"
        print(f"  {d.name:32} {d.freq:13} {d.delay_days:>5} {str(mx):11} "
              f"{d.mode:11} {d.pull_axis:9} {d.window(cfg['download']['redundancy_days']):>5}")

    if not args.no_api:
        print("\n【接口实测】真实请求（消耗限速额度）")
        client = _new_client(cfg)
        cal = state.effective_calendar()
        import datetime as _dt
        today = _dt.date.today().isoformat()
        cand = [x for x in cal if x <= today]
        probe_day = cand[-1] if cand else today
        from data_incremental.core.client import extract_list
        tests = [
            ("stock_cyq_chips 数组批量", "/stock/cyq_chips", "POST",
             {"stock_code": ["000001.SZ", "600519.SH"], "start_time": probe_day, "end_time": probe_day}),
            ("tdx_daily 按日查全板块", "/tdx/daily", "GET", {"trade_date": probe_day}),
            ("stock_forecast ann_date 区间", "/stock/forecast", "POST",
             {"start_date": probe_day, "finish_date": probe_day}),
        ]
        for label, path, method, payload in tests:
            try:
                data = client.call(path, {**payload, "page": 0, "page_size": 5},
                                   method=method, expect_rows=False)
                n = len(extract_list(data))
                print(f"  {label:30} → {n:>6} 行  "
                      f"{'✅ 可用（可批量/可区间，按此优化）' if n else '⚠️ 0 行（该日可能本就无数据，换个日期再试）'}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {label:30} → ✘ {str(exc)[:70]}")
        print(f"  请求统计: {client.stats}")
        client.close()
    return 0


# ================================================================ gate
def cmd_gate(args, cfg: dict) -> int:
    """只跑闸门 —— 零写入，随时可跑。"""
    paths.ensure_dirs()
    if getattr(args, "watch", None):
        cfg["gate"]["interval_seconds"] = args.watch
    if getattr(args, "max_wait", None) is not None:
        cfg["gate"]["max_wait_hours"] = args.max_wait
    if getattr(args, "on_timeout", None):
        cfg["gate"]["on_timeout"] = args.on_timeout
    runner = P.Runner()
    client = _new_client(cfg)

    runner.note("① 更新交易日历（闸门要按日历回推 expected）…")
    cinfo = cal_mod.update_calendar(client, cfg)
    if cinfo.get("ok"):
        runner.note(f"   日历 {cinfo['before_max']} → {cinfo['after_max']}"
                    f"（含未来 {cinfo['future_days']} 天，共 {cinfo.get('total')} 行）")
    else:
        runner.note(f"   ⚠️ 日历更新无返回：{cinfo.get('note')}（改用自愈日历）")

    cal = state.effective_calendar()
    runner.note(f"   有效交易日历 {len(cal)} 天，最后一天 {cal[-1] if cal else '—'}")

    runner.note("② 闸门开始（探测服务端）…")
    res = gate_mod.run_gate(client, cfg, cal, runner, log=runner.note)
    client.close()

    print()
    wait_rows = [r for r in res.rows if r.get("tier", "wait") == "wait"]
    watch_rows = [r for r in res.rows if r.get("tier") == "watch"]
    if res.ready:
        print(f"✅ 闸门通过：T = {res.T}，{len(wait_rows)} 张等待档表全部到齐，"
              f"用时 {P.fmt_duration(res.elapsed)}")
        if watch_rows:
            behind = [r["name"] for r in watch_rows if "落后于服务端" in (r.get("note") or "")]
            print(f"   （另有 {len(watch_rows)} 张观测档表已探测，不阻塞"
                  + (f"；⚠️ 其中落后于服务端：{', '.join(behind)}" if behind else "；均无落后") + "）")
        print(f"   下一步：python {ROOT}/main.py run")
    else:
        bad = res.blocking
        print(f"{'⏱' if res.timed_out else '⏳'} 闸门未通过：T = {res.T}，"
              f"{len(bad)}/{len(wait_rows)} 张等待档表未到齐（用时 {P.fmt_duration(res.elapsed)}）")
        for r in bad:
            print(f"   ⊘ {r['name']:32} 应到 {r['expected']} 实到 {r['actual']}  {r.get('note','')}")
        if res.timed_out:
            print(f"   → on_timeout={cfg['gate']['on_timeout']}")
    return 0 if res.ready else 1


# ================================================================ status / trend
def cmd_status(args, cfg: dict) -> int:
    print(f"  {'数据集':32} {'freq':13} {'模式':11} {'行数':>14}  {'最新日期':11} delay")
    print("  " + "-" * 90)
    total = 0
    for d in R.enabled():
        man = state.Manifest.load(d.name)
        rows = man.partition_rows()
        total += rows
        print(f"  {d.name:32} {d.freq:13} {d.mode:11} {rows:>14,}  "
              f"{str(man.max_partition_date() or '—'):11} {d.delay_days}")
    print(f"  {'合计':43} {total:>14,}")
    return 0


def cmd_trend(args, cfg: dict) -> int:
    print(viz.render_trend(days=args.days))
    return 0


def cmd_run(args, cfg: dict) -> int:
    """跑当日增量：锁 → 日历 → 闸门 → 日频 → 其他频率 → MD5 台账 → 报告。"""
    from data_incremental.pipeline import report as report_mod
    from data_incremental.pipeline import runner as runner_mod
    run_started = __import__("time").monotonic()

    runner = P.Runner()
    client = _new_client(cfg)
    only = {s.strip() for s in args.only.split(",")} if getattr(args, "only", None) else None

    # dry-run 不该等 4 小时：把闸门预算压到 0 —— 只探一轮（照发探测请求、不写数据、不 sleep），
    # 既能看到"上游到齐没有"，又能立刻拿到执行计划。
    if args.dry_run:
        cfg.setdefault("gate", {})["max_wait_hours"] = 0
        # ★ 2026-09-17：标记干跑 —— 它必然在闸门"超时"，那条 ready=False 的观测
        #   不能用于 delay 自动标定（delay.absent_streak 会跳过 mode=="dry_run"）。
        cfg["gate"]["dry_run"] = True

    logfile = None
    try:
        stamp = __import__("time").strftime("%m%d_%H%M")
        logfile = paths.LOGS / f"run_daily_{stamp}.log"
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)
        runner.note(f"日志: {logfile}")
    except OSError:
        pass

    # ★★ 2026-09-19：名单同步放在**主体之前**，这样厂商新上的指数当轮就能被抓到 ——
    #   新增的代码不在 `done.entities` 里，`run_per_entity` 会走 `todo_new` 路径
    #   给它**全量回填历史**（不是只抓当天）。放主体之后就得等第二天。
    from data_incremental.pipeline import roster as roster_mod
    maintain_roster = not only or roster_mod.THS_DS in only
    rinfo = (roster_mod.sync(cfg, client, write=not args.dry_run) if maintain_roster
             else {"ok": True, "skipped": True, "added": 0})
    if not rinfo.get("ok"):
        runner.note(f"⓪ 名单同步跳过：{rinfo.get('note')}")
    elif rinfo.get("added"):
        runner.note(f"⓪ 名单同步：厂商新增 {rinfo['added']} 只 THS 指数"
                    f"（{', '.join(rinfo.get('sample') or [])} …）"
                    f"，已写进固化名单，本轮起自动抓取并全量回填历史")
    else:
        runner.note(f"⓪ 名单同步：厂商 {rinfo.get('vendor')} 只 / 固化名单 "
                    f"{rinfo.get('roster')} 只，无新增")
    if rinfo.get("warn"):
        runner.note(f"   ⚠️ {rinfo['warn']}")

    sweep_info = None
    try:
        rep = runner_mod.run(cfg, runner, client,
                             dry_run=args.dry_run,
                             only=only,
                             skip_gate=args.no_wait,
                             as_of=getattr(args, "as_of", None),
                             report_backfill=getattr(args, "report_backfill", False),
                             persist_report=False)
        # ★ 名单完整性 sweep：日更只补尾部窗口（约 6 个交易日），**更早的洞永不自愈**。
        #   故每 SWEEP_EVERY_DAYS 天做一次全历史刷新兜底。失败不影响当日增量（已入库）。
        ths_ran = any(row.get("name") == roster_mod.THS_DS
                      for group in rep.get("groups", {}).values() for row in group)
        if not args.dry_run and maintain_roster and ths_ran:
            try:
                if getattr(args, "sweep_now", False):
                    runner.note("   --sweep-now：强制执行名单完整性 sweep（忽略周期）")
                    sweep_info = roster_mod.sweep(cfg, client, runner)
                else:
                    sweep_info = roster_mod.sweep_if_due(cfg, client, runner)
            except Exception as exc:              # noqa: BLE001
                runner.note(f"   ⚠️ 名单 sweep 失败（不影响当日增量）：{str(exc)[:90]}")
                sweep_info = {"ok": False, "note": str(exc)[:90]}
    finally:
        client.close()

    rep["roster_sync"] = rinfo
    rep["roster_sweep"] = sweep_info
    extra_alerts = []
    if not rinfo.get("ok"):
        extra_alerts.append(f"⚠️ 名单同步未完成：{rinfo.get('note')}")
    if rinfo.get("warn"):
        extra_alerts.append(f"⚠️ 名单同步：{rinfo['warn']}")
    if sweep_info and not sweep_info.get("ok"):
        extra_alerts.append(f"⚠️ 名单全历史补查未完成：{sweep_info.get('note')}")
    if not args.dry_run:
        rep.setdefault("alerts", []).extend(extra_alerts)
        rep.setdefault("alerts_actionable", []).extend(extra_alerts)
    # Include calendar/gate/roster requests and maintenance work in the final report.
    rep["requests"] = client.stats.get("requests", rep.get("requests", 0))
    rep["retries"] = client.stats.get("retries", rep.get("retries", 0))
    rep["empty_retries"] = client.stats.get("empty_retries", rep.get("empty_retries", 0))
    rep["seconds"] = round(__import__("time").monotonic() - run_started, 1)
    if rep.get("alerts_actionable"):
        rep["verdict"] = f"⚠️ 本轮有 {len(rep['alerts_actionable'])} 条需要处理的告警"
    report_mod.save(rep, append_log=not args.dry_run)

    if cfg.get("viz", {}).get("summary", True):
        print(viz.render_summary(rep))
    # 名单相关的结果独立打印：它们不属于某张表的增量，混进汇总表反而看不清
    if rinfo.get("ok") and rinfo.get("added"):
        print(f"  ⓪ 名单同步：新增 {rinfo['added']} 只 THS 指数已进固化名单"
              f"（本轮已自动回填其历史）")
    if sweep_info:
        if sweep_info.get("ok"):
            print(f"  ⓪ 名单 sweep（全历史刷新）：{sweep_info['codes']} 只 / "
                  f"取回 {sweep_info['fetched']:,} 行 / **补齐 {sweep_info['added']:,} 行**"
                  + (f" / {sweep_info['failed']} 批失败（下轮自动重试）"
                     if sweep_info.get("failed") else ""))
        else:
            print(f"  ⚠️ 名单 sweep 未完成：{sweep_info.get('note')}")
    if getattr(args, "html", False) and not args.dry_run:
        p = viz.write_html(rep)
        print(f"  HTML 报告：{p}")
    if args.dry_run:
        print("  [dry-run] 没有写入任何数据")
    # ★★ 2026-09-17：退出码只看「**要处理**的告警」。
    #   ℹ️ 档（上游撤数 / 已知厂商洞 / 厂商晚间重算）是记录性的，不该让 rc=1 ——
    #   否则 watchdog / `&&` 链会把"数据一行没丢的一轮"当成失败（实测一次 62 条假 ℹ️）。
    actionable = rep.get("alerts_actionable")
    if actionable is None:                      # 兼容旧报告结构
        actionable = rep.get("alerts") or []
    if actionable:
        print(f"  → 退出码 1：{len(actionable)} 条**要处理**的告警"
              f"（另 {len(rep.get('alerts') or []) - len(actionable)} 条为记录性 ℹ️）")
    return 0 if not actionable else 1


def cmd_calibrate_dump(args, cfg: dict) -> int:
    """标定 daily_dump 的 vol/amount 单位（一次性，必须先做）。"""
    from data_incremental.pipeline import dump_bridge

    client = _new_client(cfg)
    try:
        r = dump_bridge.calibrate(client, cfg, log=print)
    finally:
        client.close()
    print("\n" + "=" * 72)
    for k in ("verdict", "day", "dump_rows", "local_rows", "matched", "n_stocks",
              "n_bars_per_stock", "price_mismatch", "vol_ratio_median",
              "amount_ratio_median", "vol_factor", "amount_factor", "note"):
        if k in r:
            print(f"  {k:22} {r[k]}")
    print("=" * 72)
    verdict = r.get("verdict")
    if verdict == "ok":
        print("✅ 标定通过，dump 通道可用。")
        print(f"   标定文件：{paths.DUMP_CALIB}")
    else:
        print(f"❌ 标定未通过（{verdict}）—— dump 通道**不会**启用。")
    return 0 if verdict == "ok" else 1


def cmd_delete_day(args, cfg: dict) -> int:
    """删除某一天（或各数据集最新一天）的数据 —— 演练 / 修复 / 回测用。"""
    from data_incremental.tools import delete_day, backtest_increment as BT

    only = {s.strip() for s in args.only.split(",")} if getattr(args, "only", None) else None
    keep_cov = bool(getattr(args, "keep_coverage", False))
    print(f"删除 {'各数据集最新一天' if not args.date else args.date}"
          f"{'  [dry-run]' if args.dry_run else ''}")
    # ★★ 2026-09-17：删除台账改成"**开始清空 + 每删一张落一次盘**"。
    #   旧实现是全部删完之后才写一次（40 张表实测约 2 小时）—— 中途任何中断都会让
    #   `last_deleted.json` 缺失，于是回测把每一张都判成"未参与删除" → bad=0
    #   → 一次真的删遍了全平台的演练报告"通过"（向不安全方向失败）。
    if not args.dry_run:
        BT.reset_deleted()
    gone = delete_day.delete_all(args.date, only=only, dry_run=args.dry_run,
                                 trim_coverage=not keep_cov,
                                 on_each=BT.append_deleted, log=print)
    tot = sum(g.rows for g in gone)
    print(f"\n{'将删除' if args.dry_run else '已删除'} {len(gone)} 个数据集 / {tot:,} 行")
    if args.dry_run:
        print("（dry-run：没有写入任何数据，也没有写备份文件）")
    elif tot:
        if keep_cov:
            print("⚠️ --keep-coverage：manifest 的 coverage/done **没动** —— 这正是回测要检验的：")
            print("   如果增量逻辑只看标记不看实际数据，它就不会去补。")
        else:
            print(f"✔ 已同步把 manifest 的 coverage 从该日期裁掉 → 下一轮增量会把它当缺口补回。")
            print(f"   验证：python {ROOT}/main.py backtest verify")
    return 0


def cmd_monitor(args, cfg: dict) -> int:
    """数据更新监测 —— **只读**。

    观测每张日频表"服务端实际上哪天有数据"，据此反推**真实 delay**，
    并与本地水位对照；每次一条 append 到 `state/monitor/history.jsonl`。
    连续几天之后，就能用证据（而不是猜）确认 `registry.py` 里的 delay 配置。
    """
    from data_incremental.pipeline import monitor as mon
    from data_incremental.pipeline import probe as probe_mod

    cal = state.effective_calendar()
    T = getattr(args, "as_of", None)

    if not args.no_probe:
        client = _new_client(cfg)
        try:
            if T is None:
                ev = probe_mod.resolve_T(client, cal, cfg)
                T = ev.T
                if T is None:
                    print("✖ 拿不到 T（服务端没有已发布的交易日）—— 无法观测 delay")
                    return 1
            print(f"观测中（只读）…  T = {T}")
            rec = mon.observe(client, cfg, cal, T, log=lambda s: print(s))
            mon.append_history(rec)
            print(f"  ✔ 观测已记入 {mon.history_path()}")
        finally:
            client.close()

    hist = mon.load_history(limit=int(getattr(args, "days", 14) or 14))
    if not hist:
        print("还没有任何观测记录。先跑一次 `main.py monitor`。")
        return 0
    if T is None:
        T = hist[-1].get("T")

    rows = mon.summarize(hist, cal)
    print(mon.render_status(rows, T))
    print()
    print(mon.render(rows, hist))

    todo = [r for r in rows if r["change"] and r["samples"] >= 3]
    if todo:
        print()
        print("★ 建议改 registry.py 的 _DELAY（只上不下的保守方向）：")
        for r in todo:
            print(f'     "{r["name"]}": {r["rec"]},   # 原 {r["declared"]} —— {r["why"]}')
    return 0


def cmd_dump_cache(args, cfg: dict) -> int:
    """看 daily_dump 的**本地按日缓存**（零请求、零写入）。

    用户 2026-09-15 要求：daily_dump 落盘到 `datadownload/data`，「以单个日期为单位
    进行保存，以后优先读取本地是否有相关的数据」。这里就是那份缓存：
    `DATA_ROOT/stock_daily_dump/date=YYYY-MM-DD/{data.parquet,meta.json}`。

    每天跑增量时，**完整交易日**直接从缓存读（0 请求、0 配额）；只有缓存缺失、
    不完整、或坏了的那些天才去问服务端。
    """
    from data_incremental.pipeline import dump_bridge as DB

    rows = DB.cache_days()
    if not rows:
        print(f"缓存还是空的：{paths.DUMP_CACHE_ROOT}")
        print("  下一次跑 `main.py`（或 `main.py run --only stock_history_5min`）就会填上。")
        return 0
    rows.sort(key=lambda r: r["date"], reverse=bool(getattr(args, "latest", False)))
    show = rows[-int(args.limit):] if getattr(args, "limit", 0) else rows
    total_bytes = sum(r["bytes"] or 0 for r in rows)
    print("=" * 92)
    print("  daily_dump 本地缓存")
    print("=" * 92)
    print(f"  落点：{paths.DUMP_CACHE_ROOT}")
    print(f"  共 {len(rows)} 天 / {total_bytes/1e6:,.1f} MB"
          f"　（完整交易日 {sum(1 for r in rows if r.get('final'))} 天）")
    print()
    print(f"  {'日期':12}{'行数':>10}{'股票':>7}{'完整':>5}{'来源':>6}  {'写入时间':20}")
    print("  " + "-" * 88)
    for r in show:
        print(f"  {r['date']:12}{str(r['rows'] or '?'):>10}{str(r['stocks'] or '?'):>7}"
              f"{'✔' if r.get('final') else '✗':>5}{str(r['source'] or '-'):>6}  {str(r['fetched_at'] or '-'):20}")
    if len(show) < len(rows):
        print(f"  …（只显示 {len(show)} 天，用 --limit 0 看全部）")
    print()
    print("  完整 = bar 覆盖率 ≥98%（股票数 × 48 根）**且**行数 ≥0.9×正常一天；")
    print("  ✗ 的不会被采信（下次跑照常问服务端，拿到更全的就覆盖）。")
    return 0


def cmd_dump_quota(args, cfg: dict) -> int:
    """看 daily_dump 的配额台账（默认只读；`--reset <日期|all>` 才写）。

    服务端口径是「**每个数据日期每天**最多 10 次」，超了封该日期 3 天。
    本台账按日历日分桶记录（2026-09-17 修）——旧实现把历史累计次数当判据，
    于是"累计取过 10 次"就**永久**拒绝该日期，而耗配额的只有坏日子。
    """
    q = state.DumpQuota()
    reset = getattr(args, "reset", None)
    if reset:
        n = q.reset(None if reset == "all" else reset)
        print(f"✔ 已清掉 {n} 条配额记录"
              + ("（全部日期）" if reset == "all" else f"（数据日期 {reset}）"))
        return 0
    if not q.data:
        print("配额台账是空的（还没下载过 daily_dump）")
        return 0
    print("=" * 78)
    print("  daily_dump 配额台账（服务端：每个数据日期每天 10 次，超了封 3 天）")
    print("=" * 78)
    print(f"  {'数据日期':12} {'今天已用':>8} {'历史累计':>8}  最后请求时间")
    print("  " + "-" * 70)
    for k in sorted(q.data):
        date = k.split("|")[0]
        used, tot = q.count_on(date), q.count_by_date(date)
        flag = "  ⚠️ 今天已用满" if used >= state.DumpQuota.LIMIT_PER_DATE else ""
        print(f"  {date:12} {used:>8} {tot:>8}  "
              f"{str((q.data[k] or {}).get('last_at') or '-'):20}{flag}")
    print()
    print("  「今天已用」按**日历日**计（服务端口径）；「历史累计」只作展示。")
    print("  确有把握不会再超时可显式清：main.py dump-quota --reset 2026-09-14")
    return 0


def cmd_init_hashes(args, cfg: dict) -> int:
    """一次性：给全部历史补单日 MD5 台账。"""
    from data_incremental.pipeline import dayhash

    only = {s.strip() for s in args.only.split(",")} if getattr(args, "only", None) else None
    print("给全部历史建单日 MD5 台账（读全库，约 5~10 分钟）…")
    st = dayhash.init_hashes(only=only, log=print)
    days = sum(v.get("days", 0) for v in st.values())
    print(f"\n✔ {len(st)} 个数据集 / {days:,} 个「数据集×日」指纹已建")
    print(f"  台账目录：{paths.DAYHASH_DIR}")
    return 0


def cmd_hash_audit(args, cfg: dict) -> int:
    """单日 MD5 台账完整性自检（只读）。退出码 1 = 有该修的（无台账/陈旧/漏天/复算不符/主键漂移）。"""
    from data_incremental.pipeline import dayhash

    only = {s.strip() for s in args.only.split(",")} if getattr(args, "only", None) else None
    res = dayhash.audit(only=only, deep=args.deep)
    rows = res["tables"]

    tracked = [r for r in rows if r["trackable"]]
    no_led = [r for r in tracked if not r["ledger_days"]]
    stale = [r for r in tracked if r["stale"]]
    drift = [r for r in tracked if r["keys_drift"] and r["keys_drift"] != "未记录（旧台账）"]
    orphans = [r for r in tracked if r.get("orphans")]
    missing = [r for r in tracked if r.get("missing")]
    mism = [r for r in tracked if r.get("mismatch")]

    print("=" * 96)
    print(f"  单日 MD5 台账 · 完整性自检{'（--deep）' if res['deep'] else ''}"
          f"    该记账 {len(tracked)} 张 / 不记账 {len(rows) - len(tracked)} 张")
    print("=" * 96)
    print(f"  {'数据集':34} {'台账天':>7} {'范围':26} {'台账更新于':20} 判定")
    print("  " + "-" * 94)
    for r in sorted(rows, key=lambda x: (not x["trackable"], not x["ledger_days"], x["name"])):
        if not r["trackable"]:
            print(f"  {r['name']:34} {'—':>7} {'':26} {'':20} ⊘ 不记账（{r['reason']}）")
            continue
        rng = f"{r['ledger_min'] or '—'} ~ {r['ledger_max'] or '—'}"
        days = (f"{r['ledger_days']}/{r['data_days']}"
                if r.get("data_days") is not None else f"{r['ledger_days']}")
        flags = []
        if not r["ledger_days"]:
            flags.append("⚠️ 无台账")
        if r["stale"]:
            flags.append("⚠️ 台账早于跑批")
        if r.get("orphans"):
            flags.append(f"⚠️ 陈旧残留 {len(r['orphans'])}")
        if r.get("missing"):
            flags.append(f"⚠️ 漏天 {len(r['missing'])}")
        if r.get("mismatch"):
            flags.append(f"🔴 复算不符 {len(r['mismatch'])}")
        if r["keys_drift"] and r["keys_drift"] != "未记录（旧台账）":
            flags.append("🔴 主键漂移")
        print(f"  {r['name']:34} {days:>9} {rng:26} "
              f"{str(r['updated_at'] or '—'):20} " + ("　".join(flags) if flags else "✔"))

    print("=" * 96)
    n_nokeys = sum(1 for r in tracked if r["keys_drift"] == "未记录（旧台账）")
    print(f"  无台账 {len(no_led)} 张 · 台账早于跑批 {len(stale)} 张 · 主键漂移 {len(drift)} 张"
          + (f" · 主键未记录 {n_nokeys} 张（旧台账，下次重建后自动补上）" if n_nokeys else ""))
    if res["deep"]:
        print(f"  陈旧残留 {len(orphans)} 张 · 漏天 {len(missing)} 张 · 复算不符 {len(mism)} 张")
        for r in orphans:
            print(f"    ⚠️ {r['name']} 台账里有但数据里没有：{r['orphans'][:8]}")
            print(f"       → 补法：`main.py init-hashes --only {r['name']}` 或清掉台账里那几天")
        for r in missing:
            print(f"    ⚠️ {r['name']} 数据里有但台账没有（**台账自身范围内**的缺口）：{r['missing'][:8]}")
        # ℹ️ 台账只覆盖最近 N 天是**已知状态**（只有 4 张表建了全历史台账），不是缺陷 ——
        #    这里单独说明，免得下次又被当成问题查（第一版就误报过 25 张"漏天"）。
        partial = [r for r in tracked
                   if r.get("data_days") and r["ledger_days"] < r["data_days"]]
        if partial:
            full = [r["name"] for r in tracked if r.get("data_days") and r["ledger_days"] >= r["data_days"]]
            print(f"  ℹ️ 台账未覆盖全历史 {len(partial)} 张（**已知状态，不是缺陷**）——"
                  f"每轮只重算尾部窗口；全历史台账只有 {len(full)} 张：{', '.join(full) or '（无）'}")
            print(f"     要补某张：`main.py init-hashes --only <表名>`（台账天/数据天 见上表第 2 列）")
    else:
        print("  （陈旧残留 / 漏天 / 复算 需要 --deep 才查 —— 要读数据，慢）")
    bad = len(no_led) + len(stale) + len(drift) + len(orphans) + len(missing) + len(mism)
    if bad:
        print(f"\n  ⚠️ 共 {bad} 项待处理")
        return 1
    print("\n  ✔ 全部通过")
    return 0


def cmd_cleanup_backups(args, cfg: dict) -> int:
    """归档数据备份校验值，再按需释放磁盘；与日更共用锁。"""
    from data_incremental.tools.cleanup_backups import run
    run(apply=args.apply)
    return 0


def cmd_backtest(args, cfg: dict) -> int:
    """破坏性回测：**基线 → 删除 → 重跑增量 → 指纹比对**。

    回答一个问题：**删掉某一天的数据，下一轮增量能不能原样补回来、要多久？**
    """
    from data_incremental.tools import backtest_increment as BT

    step = getattr(args, "step", "help")
    if step == "baseline":
        BT.take_baseline_latest()
        return 0
    if step == "verify":
        rows = BT.compare("latest", deleted=BT.load_deleted())
        print()
        print(BT.render(rows))
        return 0 if not BT.summary(rows)["bad"] else 1
    if step == "deleted":
        d = BT.load_deleted()
        for k, v in sorted(d.items()):
            print(f"   {k:34} {v['date']} 删了 {v['rows']:>9,} 行  备份 {v['backup']}")
        return 0

    print("破坏性回测 —— 三个步骤（中间那步会真的删数据，但有备份）：")
    print()
    print(f"  1) python {ROOT}/main.py backtest baseline        # 记录各表最新一天的指纹")
    print(f"  2) python {ROOT}/main.py delete-day               # 删各表最新一天（自动备份）")
    print(f"  3) python {ROOT}/main.py run --no-wait            # 跑一轮增量，看补不补得回")
    print(f"  4) python {ROOT}/main.py backtest verify          # 逐表比对指纹")
    print()
    print("  想检验得更严格一点（连 coverage 标记一起保留，逼增量「只看数据」）：")
    print(f"      python {ROOT}/main.py delete-day --keep-coverage")
    print()
    print("  备份落在 state/backtest/deleted_<表>__<日期>.parquet，可直接读回核对。")
    return 0


# ================================================================ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description="每日增量更新")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("gate", help="只跑闸门（零写入）")
    g.add_argument("--watch", type=float, default=None, help="覆盖轮询间隔（秒）")
    g.add_argument("--max-wait", type=float, default=None, help="覆盖最长等待（小时）")
    g.add_argument("--on-timeout", choices=["partial", "strict"], default=None,
                   help="超时行为：partial=跳过未到齐的照跑 / strict=不跑")
    g.set_defaults(fn=cmd_gate)

    d = sub.add_parser("doctor", help="体检")
    d.add_argument("--no-api", action="store_true", help="跳过接口实测（零请求）")
    d.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("status", help="本地水位")
    s.set_defaults(fn=cmd_status)

    t = sub.add_parser("trend", help="运行趋势")
    t.add_argument("--days", type=int, default=30)
    t.set_defaults(fn=cmd_trend)

    mo = sub.add_parser("monitor", help="数据更新监测：观测真实 delay + 本地水位（只读）")
    mo.add_argument("--days", type=int, default=14, help="回看最近几次观测（默认 14）")
    mo.add_argument("--no-probe", action="store_true",
                    help="跳过服务端探测，只看历史与本地水位（零请求）")
    mo.add_argument("--as-of", default=None, metavar="YYYY-MM-DD", help="把这天当作 T（测试用）")
    mo.set_defaults(fn=cmd_monitor)

    r = sub.add_parser("run", help="跑当日增量（默认命令）")
    r.add_argument("--dry-run", action="store_true", help="只打印计划，不写任何数据")
    r.add_argument("--only", default=None, help="只跑指定数据集（逗号分隔）")
    r.add_argument("--no-wait", action="store_true", help="跳过闸门等待，直接开跑")
    r.add_argument("--as-of", default=None, metavar="YYYY-MM-DD",
                   help="★ 把这一天当作 T（测试用；建议 --as-of 2026-09-11，那天数据必然齐全）")
    r.add_argument("--html", action="store_true", help="额外输出 HTML 报告")
    r.add_argument("--report-backfill", action="store_true",
                   help="一次性存量回填：把 4 张财报表(21-24)的**全部**历史报告期抓一遍"
                        "（常规轮次只抓最近 3 期 + ann_date 窗口；见 README §9.11）")
    r.add_argument("--sweep-now", action="store_true",
                   help="强制做一次名单完整性 sweep（忽略 30 天周期；约 40 个请求 / 10 分钟）")
    r.add_argument("--force", action="store_true", help="忽略单实例保护")
    r.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    r.set_defaults(fn=cmd_run)

    cd = sub.add_parser("calibrate-dump", help="标定 daily_dump 的 vol/amount 单位（一次性）")
    cd.set_defaults(fn=cmd_calibrate_dump)

    dq = sub.add_parser("dump-quota", help="看 daily_dump 配额台账（只读；--reset 才写）")
    dq.add_argument("--reset", default=None, metavar="DATE|all",
                    help="清掉某个数据日期的配额记录（或 all）")
    dq.set_defaults(fn=cmd_dump_quota)

    dc = sub.add_parser("dump-cache", help="看 daily_dump 的本地按日缓存（只读，零请求）")
    dc.add_argument("--limit", type=int, default=30, help="显示最近 N 天（0 = 全部）")
    dc.add_argument("--latest", action="store_true", help="按日期倒序（默认正序）")
    dc.set_defaults(fn=cmd_dump_cache)

    dd = sub.add_parser("delete-day", help="删除某一天的数据（演练/修复/回测）")
    dd.add_argument("--date", default=None, help="YYYY-MM-DD；不给则删各数据集自己的最新一天")
    dd.add_argument("--only", default=None, help="只处理这些数据集（逗号分隔）")
    dd.add_argument("--dry-run", action="store_true",
                    help="只打印会删什么；零副作用（不写数据、不写备份）")
    dd.add_argument("--keep-coverage", action="store_true",
                    help="保留 manifest 的 coverage 标记（回测用：检验「不看标记也能自愈」）")
    dd.set_defaults(fn=cmd_delete_day)

    ih = sub.add_parser("init-hashes", help="一次性建全部历史的单日 MD5 台账")
    ih.add_argument("--only", default=None)
    ih.set_defaults(fn=cmd_init_hashes)

    ha = sub.add_parser("hash-audit", help="单日 MD5 台账完整性自检（只读；--deep 才读数据）")
    ha.add_argument("--only", default=None, help="只看指定数据集（逗号分隔）")
    ha.add_argument("--deep", action="store_true",
                    help="额外查陈旧残留/漏天/抽样复算（要读数据，慢）")
    ha.set_defaults(fn=cmd_hash_audit)

    cb = sub.add_parser("cleanup-backups", help="备份数据转 MD5/SHA-256 清单（默认预览）")
    cb.add_argument("--apply", action="store_true", help="先保存完整校验清单，再删除备份数据")
    cb.set_defaults(fn=cmd_cleanup_backups)

    b = sub.add_parser("backtest", help="破坏性回测：基线/删除/重跑/比对")
    b.add_argument("step", nargs="?", default="help",
                   choices=["baseline", "verify", "deleted", "help"],
                   help="baseline=记录基线；verify=跑完增量后比对指纹；deleted=看本轮回测删了什么")
    b.set_defaults(fn=cmd_backtest)

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    # ★★ 裸命令 = 全流程。文件头的用法说明一直写着
    #    `python main.py  # 全流程（默认）`，但旧实现把默认命令设成了 `cmd_gate`
    #    —— 而 gate **零写入**。于是"每天执行 main.py"实际上**一行数据都不会更新**，
    #    只在终端刷一遍闸门结果。用户要的"每天只跑一条命令"必须真的跑增量。
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["run"] + argv        # 裸命令 / 只带全局开关 → 走 run（闸门 + 全量增量）

    args = build_parser().parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    cfg = C.load()
    paths.ensure_dirs()

    if not getattr(args, "fn", None):
        args.fn = cmd_run

    # ★ 只读命令不抢锁：`gate` / `monitor` / `status` / `trend` / `doctor` / `dump-cache` / `hash-audit`
    #   必须在"另一轮增量正在跑"时也能执行（否则连看进度都做不到）。
    #   ⚠️ `hash-audit` 是纯只读（`update_and_compare(..., write=False)`），
    #      所以必须在白名单里 —— `init-hashes`/`delete-day` 那类写命令抢锁是**故意**的。
    if args.fn in (cmd_gate, cmd_monitor, cmd_status, cmd_trend, cmd_doctor,
                   cmd_dump_cache, cmd_dump_quota, cmd_hash_audit):
        return args.fn(args, cfg)

    try:
        with lockmod.acquire(force=getattr(args, "force", False)):
            return args.fn(args, cfg)
    except lockmod.AlreadyRunning as exc:
        print(f"✖ {exc}")
        return 2
    except KeyboardInterrupt:
        print("\n✖ 被中断")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
