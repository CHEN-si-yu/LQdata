#!/usr/bin/env python3
"""直接生成因子：无参数增量，rebuild 全量，按因子对齐进度条。"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def _mutating_cli(argv):
    """Compatibility helper used by project maintenance scripts."""
    if "--dry-run" in argv or "--sandbox" in argv:
        return False
    for i, arg in enumerate(argv):
        base = Path(arg).name
        if base == "backfill_history.py":
            return True
        if base == "trim_recent_days.py":
            return "--apply" in argv
        if base == "main.py" and (i == len(argv)-1 or argv[i+1] in {"run", "incremental", "rebuild", "full"}):
            return True
    return False


def _date(value):
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式必须是 YYYY-MM-DD") from exc


def parser():
    p = argparse.ArgumentParser(description="因子生产：全量、增量、自动跳过、并行、进度条")
    p.add_argument("-v", "--verbose", action="store_true", help="记录详细计算日志")
    sub = p.add_subparsers(dest="cmd")
    for name, aliases, help_text in (("run", ["incremental"], "增量：补缺/新日期，自动跳过已有结果"),
                                     ("rebuild", ["full"], "全量：重算选中因子的请求范围")):
        sp = sub.add_parser(name, aliases=aliases, help=help_text)
        sp.add_argument("names", nargs="*", help="指定因子；默认全部已启用注册因子")
        sp.add_argument("--factors", nargs="+", default=None, help="指定因子（兼容旧用法）")
        sp.add_argument("--group", nargs="+", default=None, help="按类别选择")
        sp.add_argument("--jobs", type=int, default=0,
                        help="并行进程数；正整数豁免项目并行度和内存限制；0 自动选择并保留资源保护，配置默认 12")
        sp.add_argument("--start", type=_date)
        sp.add_argument("--end", type=_date, help="截止日；默认上游 stock_daily 最新交易日")
        sp.add_argument("--from-year", type=int)
        sp.add_argument("--to-year", type=int)
        sp.add_argument("--sandbox", metavar="DIR", help="输出到项目内的隔离目录")
        sp.add_argument("--progress", choices=("auto", "bars", "plain", "off"), default="auto",
                        help="终端默认多行 tqdm；重定向时默认对齐文本")
        sp.add_argument("--dry-run", action="store_true", help="只显示待算日期")
        sp.add_argument("--refresh", action="store_true", help="重算请求范围，保留范围外结果")
        sp.add_argument("--rebuild", action="store_true", help=argparse.SUPPRESS)
        sp.add_argument("--resume", action="store_true", help="全量中断后只补尚未完成的部分")
    for name in ("list", "status"):
        sp = sub.add_parser(name, help="列出注册因子与本地覆盖进度")
        sp.add_argument("--sandbox", metavar="DIR")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Options alone, like --jobs 8, are ordinary incremental production.
    commands = {"run", "incremental", "rebuild", "full", "list", "status"}
    if not argv:
        argv = ["run"]
    elif not any(a in commands for a in argv) and not any(a in ("-h", "--help") for a in argv):
        argv.insert(1 if argv[0] in ("-v", "--verbose") else 0, "run")
    args = parser().parse_args(argv)
    from fea import config
    from fea.resources import configure
    cfg = config.load()
    if args.sandbox:
        cfg = cfg.sandbox(Path(args.sandbox))
    configure(cfg.raw.get("resources"), enforce_limits=getattr(args, "jobs", 0) <= 0)
    os.environ.setdefault("FEA_LOG_NAME", f"run_factors_{dt.datetime.now():%m%d_%H%M%S}_{os.getpid()}")
    config.setup_logging(args.verbose, cfg)
    from fea.progress import announce
    announce(f"日志: {cfg.logs_dir / (os.environ['FEA_LOG_NAME']+'.log')}")
    from fea.production import run, registered_specs
    if args.cmd in ("list", "status"):
        from fea.manifest import Manifest
        specs = registered_specs(cfg)
        width = max(map(lambda s: len(s.name), specs), default=6)
        for s in specs:
            man = Manifest.load(cfg.state_dir, s.name)
            print(f"{s.name:<{width}} {s.group:<12} {man.summary()}")
        print(f"已启用注册因子 {len(specs)}")
        return 0
    args.factors = args.factors if args.factors is not None else args.names
    args.rebuild = (args.rebuild or args.cmd in ("rebuild", "full")) and not args.resume
    if args.from_year:
        args.start = _date(f"{args.from_year:04d}-01-01")
    if args.to_year:
        args.end = _date(f"{args.to_year:04d}-12-31")
    try:
        return run(args, cfg)
    except KeyboardInterrupt:
        print("已停止；已完成分区已保存，再执行 main.py 自动补齐剩余任务", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, MemoryError, OSError) as exc:
        import logging
        logging.exception("生产失败")
        print(f"计算失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    shared = Path(os.environ.get("FEA_PYTHON", "/autodl-fs/data/miniconda3/bin/python"))
    if shared.exists() and Path(sys.executable).resolve() != shared.resolve():
        os.execv(str(shared), [str(shared), str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.dont_write_bytecode = True
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "ARROW_NUM_THREADS"):
        os.environ[key] = "1"
    temporary = ROOT / "artifacts/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ[key] = str(temporary)
    raise SystemExit(main())
