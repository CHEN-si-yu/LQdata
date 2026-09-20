#!/usr/bin/env python3
"""灵启数据下载器 —— 主入口。

日常增量更新（你每天手动跑这一条即可）：
    python main.py run

首次全量：
    python main.py run --full

单独跑某几个数据集：
    python main.py run stock_daily stock_adj_factor

其它：
    python main.py list            查看所有数据集与进度
    python main.py status          汇总本地已落地情况
    python main.py doctor          体检：API 连通性 + 服务端数据量对比
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lingqi import spec as spec_mod                      # noqa: E402
from lingqi.engine import Engine, today_str              # noqa: E402
from lingqi.manifest import Manifest                     # noqa: E402
from lingqi.progress import fmt_duration, fmt_rows       # noqa: E402
from lingqi import store                                 # noqa: E402

# 下载顺序：依赖优先 + 先小后大。
#   第 0 层 基础：日历/股票列表 —— 其它数据集的增量判断都依赖它们
#   第 1 层 小表：秒级到几分钟
#   第 2 层 中表：逐股/逐日的遍历类
#   第 3 层 大表：千万行级
#   第 4 层 慢/重：单次极慢或体量巨大，放在最后，中断也不影响主干
DEFAULT_ORDER = [
    # 0 基础
    "basic_calendar", "stock_list",
    # 1 小表 / 快照
    # index_ths_sector_categories / index_ths_constituent_stocks / tdx_block_stocks
    # 已于 2026-09-19 彻底删除（快照表，无历史版本，回测必然前视）
    "tdx_blocks", "dc_blocks",
    "stock_pledge_stat",
    "stock_forecast", "stock_holder_number",
    "stock_limit_up", "stock_limit_list",
    # stock_dc_block_fund_flow / stock_ths_block_fund_flow / ths_hot
    # 均已于 2026-09-19 彻底删除（时间覆盖不足）
    "dc_daily", "tdx_daily",
    "stock_financial_indicator", "stock_income", "stock_balancesheet", "stock_cashflow",
    # 2 中表
    "stock_st_info", "stock_suspension", "stock_adj_factor_changes",
    "stock_dragon_tiger", "stock_top_list",
    "index_daily", "index_ths_daily", "tdx_minute",
    # 3 大表
    "stock_adj_factor", "stock_daily", "stock_daily_adj",
    "stock_main_fund_flow", "stock_margin_detail",
    "stock_finance",
    # 4 慢/重
    # 2026-09-13 用户拍板重新启用后重排：时间窗口敏感的先跑，13 小时的重活放最后。
    #   stock_daily_dump  —— 只有最近 90 天窗口，越晚跑丢得越多（且不能回补）
    #   stock_market_distribution_history —— 最快（约 1~2 小时）
    #   stock_history_5min —— 中等
    #   stock_cyq_chips    —— 最重（约 13 小时 / 12.7 亿行），放最后，
    #                          被中断时前面的活已经干完
    "index_history", "stock_daily_dump",
    "stock_market_distribution_history", "stock_history_5min", "stock_cyq_chips",
]


def setup_logging(verbose: bool) -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    handlers = [logging.FileHandler(ROOT / "logs" / "download.log", encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=handlers,
    )


def load_cfg() -> dict:
    with open(ROOT / "conf" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ordered_specs(names: list[str] | None) -> list:
    if names:
        out = []
        for n in names:
            if n not in spec_mod.REGISTRY:
                raise SystemExit(f"未知数据集: {n}\n用 `python main.py list` 查看可用名称")
            out.append(spec_mod.get(n))
        return out
    known = {s.name: s for s in spec_mod.all_specs()}
    ordered = [known[n] for n in DEFAULT_ORDER if n in known]
    ordered += [s for s in spec_mod.all_specs() if s.name not in DEFAULT_ORDER]
    return ordered


# ------------------------------------------------------------------ 命令
def _run_pidfile(state_root: Path) -> Path:
    return state_root / "run.pid"


def _running_run_pids(state_root: Path) -> list[int]:
    """找出**本项目**正在运行的下载实例（排除自己）。

    为什么需要：共享盘是网络文件系统，**没有文件锁保证**，两个实例同时写
    data/ 与 state/ 会互相覆盖。全量任务要跑十几小时，很容易撞上手动跑的每日增量。

    ⚠️ **为什么不用 `ps | grep 'main.py run'`**（2026-09-13 实测踩过两次）：
      1. 模块② `featureengineering/main.py` 也是同名入口，argv 一模一样 → 误判成
         "已有下载在跑"，会让本项目的下载被错误拒绝、看门狗空等；
      2. 本容器里 `/proc/<pid>/cwd` **连 root 都读不到**（Permission denied），
         没法用工作目录区分。
    所以改用标准的 **pidfile + 存活探测**：cmd_run 启动时把 PID 写进 state/run.pid，
    退出时删掉；这里只认"文件里记的 PID 还活着"。
    """
    out: list[int] = []
    p = _run_pidfile(state_root)
    try:
        pid = int(p.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return out                              # 没有 pidfile / 内容坏了
    if pid == os.getpid():
        return out
    try:
        os.kill(pid, 0)                         # 0 号信号只探测存活，不真的发信号
    except ProcessLookupError:
        return out                              # 进程已退出 → 陈旧的 pidfile
    except PermissionError:
        pass                                    # 进程存在但不属于我 → 仍然算冲突
    out.append(pid)
    return out


def cmd_run(args, cfg: dict) -> int:
    state_root = ROOT / cfg["paths"]["state"]
    state_root.mkdir(parents=True, exist_ok=True)
    others = _running_run_pids(state_root)
    if others and not args.force:
        pids = " ".join(str(p) for p in others)
        print(f"\n✘ 已有下载进程在跑（PID {pids}），拒绝启动。\n"
              f"  两个实例同时写共享盘会互相覆盖，务必只跑一个。\n"
              f"  查看进度：cat state/status.json\n"
              f"  确认要停：kill {pids}   （不要用 pkill -f，会误杀自己）\n"
              f"  确实要并行（例如只补某个数据集）：--force")
        return 2

    pid_file = _run_pidfile(state_root)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    try:
        return _cmd_run_body(args, cfg)
    finally:
        # 只删自己的 pidfile，避免误删后来者的
        try:
            if pid_file.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pid_file.unlink()
        except OSError:
            pass


def _cmd_run_body(args, cfg: dict) -> int:
    specs = ordered_specs(args.datasets)
    if args.group:
        specs = [s for s in specs if s.group in args.group]
    specs = [s for s in specs if s.enabled]
    if not specs:
        print("没有匹配的数据集")
        return 1

    state_root = ROOT / cfg["paths"]["state"]
    data_root = ROOT / cfg["paths"]["data"]

    # ---------------------------------------------------------- 增量闸门
    # 用户 2026-09-13 要求：日频数据全部到齐（含各自 delay）才开始增量，否则每 5 分钟
    # 复查一次、最多等 4 小时。只在"跑全量清单的普通增量"时生效 —— 指定单个数据集
    # （补数/除错）或 --full 不该被别人的进度卡住。
    gated = (not args.datasets) and (not args.group) and (not args.full) and (not args.no_wait)
    if gated:
        from lingqi import gate as gate_mod
        import yaml as _yaml
        with open(ROOT / "conf" / "frequency.yaml", encoding="utf-8") as _f:
            _fq = _yaml.safe_load(_f)
        print("\n检查日频数据是否都已更新到最新（含各自 delay）…")
        res = gate_mod.wait_until_ready(
            specs, _fq, state_root, data_root,
            interval=args.wait_interval, max_wait=args.wait_max * 3600)
        if not res.get("ready"):
            print("\n✘ 日频数据未到齐，**不开始增量**（这是用户要求的闸门）。")
            print("  确实要跳过闸门：加 --no-wait")
            return 3

    # 记录本次运行前各数据集的最新日期，跑完后用来报告"谁真的进新数据了"
    from lingqi import gate as _g
    pre_max = {s.name: _g.dataset_max_date(state_root, s.name) for s in specs}

    end = args.end or today_str()
    engine = Engine(cfg, ROOT)
    mode = "全量" if args.full else "增量"
    print(f"\n灵启数据下载 · {mode}更新 · 截止 {end} · 共 {len(specs)} 个数据集")
    print(f"限速 {cfg['api']['rate_limit_per_min']}/分钟 · 并发 {cfg['api']['concurrency']}")
    print("-" * 92)

    t0 = time.time()
    overall = engine.board.overall(len(specs), "总进度")
    failed: list[tuple[str, str]] = []
    for i, s in enumerate(specs, 1):
        overall.set_description(f"总进度 [{i}/{len(specs)}] {s.name}")
        try:
            engine.run(s, full=args.full, end=end, levels=args.levels)
        except KeyboardInterrupt:
            print("\n收到中断，已保存进度，可重跑续传。")
            overall.close()
            return 130
        except Exception as exc:  # noqa: BLE001
            logging.exception("数据集 %s 失败", s.name)
            failed.append((s.name, str(exc)[:120]))
            engine.board.note(f"✘ {s.name} 失败: {str(exc)[:120]}")
        overall.update(1)
    overall.close()

    el = time.time() - t0
    st = engine.client.stats.snapshot()
    print("-" * 92)
    print(f"完成：{len(specs) - len(failed)}/{len(specs)} 个数据集，用时 {fmt_duration(el)}")
    print(f"请求 {st['requests']:,} 次 · 重试 {st['retries']:,} · 空响应重试 {st['empty_retries']:,} "
          f"· 下行 {st['bytes_in'] / 1e9:.2f} GB · 入库 {fmt_rows(st['rows'])}行")
    if failed:
        print("\n以下数据集失败（可单独重跑，会自动续传）：")
        for n, e in failed:
            print(f"  - {n}: {e}")
    _report_progress(pre_max, specs, state_root, data_root)
    print(f"\n配额台账：{engine.dump_quota.stats()}")
    return 1 if failed else 0


def _report_progress(pre_max: dict, specs, state_root: Path, data_root: Path) -> None:
    """跑完后的"谁真的进了新数据"报告 —— 用户要求「更新完日频数据后，
    检查其他频率的数据是否存在最新的结果」。

    分两组看：日频（闸门已保证到齐，这里看有没有真的推进）和其他频率
    （季频/月频/不定期 —— 它们的"有没有新结果"**不能只盯 max_date**，
    本报告只客观列出变化，判读交给下游）。
    """
    import yaml as _yaml
    from lingqi import gate as _g

    try:
        with open(ROOT / "conf" / "frequency.yaml", encoding="utf-8") as f:
            fq = _yaml.safe_load(f)
    except OSError:
        return
    ds_cfg = fq.get("datasets") or {}
    daily, other, nochange = [], [], 0
    for s in specs:
        after = _g.dataset_max_date(state_root, s.name)
        before = pre_max.get(s.name)
        if after is None:
            continue
        if after != before:
            item = (s.name, before, after, ds_cfg.get(s.name, {}).get("freq", "?"))
            (daily if item[3] in _g.DAILY_FREQS else other).append(item)
        else:
            nochange += 1

    print("-" * 92)
    print("本次增量的推进情况：")
    if daily:
        print(f"  【日频】{len(daily)} 个数据集有新数据：")
        for n, b, a, _ in daily[:15]:
            print(f"      {n:34s} {b} → {a}")
        if len(daily) > 15:
            print(f"      … 另有 {len(daily) - 15} 个")
    if other:
        print(f"  【其他频率】{len(other)} 个数据集有新结果（季频/月频/不定期，判读请结合各自发布节奏）：")
        for n, b, a, f in other[:15]:
            print(f"      {n:34s} [{f}] {b} → {a}")
        if len(other) > 15:
            print(f"      … 另有 {len(other) - 15} 个")
    if not daily and not other:
        print("  （全部数据集的最新日期都没有变化）")
    print(f"  未变化：{nochange} 个")


def _orphan_datasets(state_root: Path, data_root: Path, known: set[str]) -> list[str]:
    """找出有数据、但没有对应 spec 的产出目录。

    dump 通道就是这种情况：spec 名是 `stock_daily_dump`，实际按级别落到了
    `stock_dump_5min` / `stock_dump_1min`。不列出来的话，用户会以为这份数据不存在。
    """
    out = []
    if not data_root.is_dir():
        return out
    for d in sorted(data_root.iterdir()):
        if not d.is_dir() or d.name in known:
            continue
        # 有条目的看 manifest；没有的（异常情况）看目录里是否真的落了 parquet
        if Manifest.load(state_root, d.name).partition_rows() or store.dataset_size(data_root, d.name):
            out.append(d.name)
    return out


def cmd_list(args, cfg: dict) -> int:
    state_root = ROOT / cfg["paths"]["state"]
    data_root = ROOT / cfg["paths"]["data"]
    print(f"\n{'数据集':<34}{'模式':<11}{'tier':<8}{'本地状态':<34}{'描述'}")
    print("-" * 130)
    total = 0
    specs = ordered_specs(None)
    for s in specs:
        man = Manifest.load(state_root, s.name)
        desc = man.summary() if man.partitions else "—"
        total += man.partition_rows()
        print(f"{s.name:<34}{s.mode:<11}{s.tier:<8}{desc:<34}{s.desc}")
    for n in _orphan_datasets(state_root, data_root, {s.name for s in specs}):
        man = Manifest.load(state_root, n)
        desc = man.summary() if man.partitions else "—"
        total += man.partition_rows()
        print(f"{n:<34}{'dump产出':<11}{'—':<8}{desc:<34}★ 非 spec 产出（见 run_dump）")
    print("-" * 130)
    print(f"本地总行数：{total:,}")
    return 0


def cmd_status(args, cfg: dict) -> int:
    state_root = ROOT / cfg["paths"]["state"]
    data_root = ROOT / cfg["paths"]["data"]
    print(f"\n{'数据集':<34}{'行数':>14}{'分区':>6}{'占用':>11}  {'最早':<11}{'最晚':<11}")
    print("-" * 96)
    grand = 0
    specs = ordered_specs(None)
    names = [s.name for s in specs]
    names += _orphan_datasets(state_root, data_root, set(names))
    for name in names:
        man = Manifest.load(state_root, name)
        rows = man.partition_rows()
        if not rows:
            continue
        grand += rows
        size = store.dataset_size(data_root, name)
        print(f"{name:<34}{rows:>14,}{len(man.partitions):>6}{size / 1e6:>10.1f}M  "
              f"{str(man.min_partition_date()):<11}{str(man.max_partition_date()):<11}")
    print("-" * 96)
    print(f"合计 {grand:,} 行")
    st = state_root / "status.json"
    if st.exists():
        print("\n上次运行状态：")
        print(st.read_text(encoding="utf-8"))
    return 0


def cmd_doctor(args, cfg: dict) -> int:
    """体检：逐个数据集发起一次最小请求，确认接口可用。"""
    from lingqi.client import LingqiClient, extract_list
    client = LingqiClient(cfg, logging.getLogger("doctor"))
    engine = Engine(cfg, ROOT)
    specs = ordered_specs(args.datasets)
    print(f"\n体检 {len(specs)} 个数据集（各发一次最小请求）\n")
    ok = bad = 0
    for s in specs:
        payload = dict(s.params)
        try:
            if s.mode == "per_date":
                days = engine.trading_days_between("2026-01-01", today_str())
                payload[s.date_param] = days[-1] if days else today_str()
            elif s.mode in ("range",):
                payload[s.start_param] = "2026-08-01"
                payload[s.end_param] = "2026-09-10"
            elif s.mode == "per_stock":
                payload[s.code_param] = "600000.SH"
            elif s.mode == "per_entity":
                codes = engine._entity_codes(s)
                if not codes:
                    print(f"  ⊘ {s.name:<34} 依赖未就绪（缺实体列表）")
                    continue
                payload[s.code_param] = codes[0]
                payload[s.start_param] = "2026-08-01"
                payload[s.end_param] = "2026-09-10"
            elif s.mode == "dump":
                print(f"  ⊘ {s.name:<34} 跳过（配额宝贵，不做体检请求）")
                continue
            data = client.call(s.path, payload, method=s.method, expect_rows=False)
            n = len(extract_list(data))
            print(f"  ✓ {s.name:<34} {n:>7} 行")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ✘ {s.name:<34} {str(exc)[:70]}")
            bad += 1
    print(f"\n可用 {ok} · 异常 {bad}")
    return 0 if bad == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(
        prog="main.py", description="灵启数据下载器",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("-v", "--verbose", action="store_true", help="同时输出到屏幕")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="下载/增量更新（默认全部数据集）")
    r.add_argument("datasets", nargs="*", help="只跑指定数据集")
    r.add_argument("--full", action="store_true", help="全量重下（忽略已有进度）")
    r.add_argument("--end", default=None, help="截止日期 YYYY-MM-DD，默认今天")
    r.add_argument("--group", nargs="*", help="按分组过滤 basic/stock/index/tdx/dc/ths/dump")
    r.add_argument("--levels", nargs="*", help="dump 模式的分钟级别，如 5min 1min")
    r.add_argument("--force", action="store_true", help="忽略『已有实例在跑』的保护，强制启动")
    r.add_argument("--no-wait", action="store_true",
                   help="跳过『等所有日频数据到齐』的闸门，立即开始（默认会先等）")
    r.add_argument("--wait-max", type=float, default=4.0,
                   help="闸门最长等待小时数，默认 4.0")
    r.add_argument("--wait-interval", type=float, default=300,
                   help="闸门重查间隔秒数，默认 300（5 分钟）")

    sub.add_parser("list", help="列出数据集与本地进度")
    sub.add_parser("status", help="查看已落地数据统计")
    d = sub.add_parser("doctor", help="接口体检")
    d.add_argument("datasets", nargs="*")

    args = p.parse_args()
    setup_logging(args.verbose)
    cfg = load_cfg()
    return {"run": cmd_run, "list": cmd_list, "status": cmd_status, "doctor": cmd_doctor}[args.cmd](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
