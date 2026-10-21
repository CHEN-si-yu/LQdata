"""CLI hooks for main.py; all writes remain in the factor project."""
from __future__ import annotations

import json
from pathlib import Path

from .catalog import Catalog, clean_json, load_definition, project_path
from .protocol import load_protocol, walk_forward_splits


def add_parser(subparsers):
    parser = subparsers.add_parser("research", help="隔离候选研究、评价、验证和显式准入")
    commands = parser.add_subparsers(dest="research_command", required=True)
    for command in ("idea", "family", "candidate"):
        child = commands.add_parser(command, help="冻结登记定义；改变定义必须使用新版本")
        child.add_argument("--file", type=Path, required=True)
    commands.add_parser("init-example", help="登记六个未验证的量价候选，不计算或进入生产")
    listing = commands.add_parser("list", help="读取定义与当前生命周期")
    listing.add_argument("--kind", choices=("ideas", "families", "candidates"), default="candidates")
    run = commands.add_parser("run", help="隔离计算候选、声明父依赖与正式标签")
    run.add_argument("candidate_id")
    run.add_argument("--start")
    run.add_argument("--end")
    run.add_argument("--jobs", type=int, default=0)
    evaluate = commands.add_parser("evaluate", help="按全部期限评价已完成隔离运行")
    evaluate.add_argument("run_id")
    verify = commands.add_parser("verify", help="实际截断及两阶段增量复算，验证逐值一致")
    verify.add_argument("run_id")
    verify.add_argument("--cutoff")
    verify.add_argument("--jobs", type=int, default=0)
    promote = commands.add_parser("promote", help="链接实际证据后显式改变生命周期")
    promote.add_argument("candidate_id")
    promote.add_argument("--to", choices=("qualified", "shadow", "production", "paused", "retired"), required=True)
    promote.add_argument("--reason", required=True)
    promote.add_argument("--evaluation")
    promote.add_argument("--verification")
    promote.add_argument("--external-evidence", type=Path)
    splits = commands.add_parser("splits", help="输出日期级 purged walk-forward 计划，并非已完成模型测试")
    splits.add_argument("--start", default="2018-01-01")
    splits.add_argument("--end")
    from .selection import add_arguments as selection_arguments
    from .batch import add_arguments as batch_arguments
    selection_arguments(commands.add_parser("selection", help="按固定任务网格审查正式库；保存证据，不自动删除"))
    batch_arguments(commands.add_parser("batch", help="批量登记和隔离计算新候选；统一资源与可恢复账本"))
    return parser


def _definition_path(cfg, path):
    path = Path(path)
    path = (path if path.is_absolute() else cfg.root / path).resolve()
    if not path.is_relative_to(cfg.root.resolve()):
        raise ValueError("Keep research definitions and optional evidence inside this project")
    return path


def init_example(cfg):
    catalog = Catalog(cfg)
    base = cfg.root / "conf" / "research" / "examples"
    idea = catalog.register_idea(load_definition(base / "pv_idea.yaml"))
    family = catalog.register_family(load_definition(base / "pv_family.yaml"))
    template = load_definition(base / "pv_candidate.yaml")
    records = []
    for parameters in family["definition"]["allowed_variants"]:
        data = {**template, "parameters": parameters,
                "name": f"pv_exhaustion_w{parameters['window']}_{parameters['expression']}_v1"}
        spec = dict(data["spec"])
        spec["warmup_days"] = 4 * int(parameters["window"]) + 30
        if parameters["expression"] == "event":
            spec["formula"] = "1[logret_A>0 and logret_B-logret_A>0 and logvol_mean_A-logvol_mean_B>0]"
            spec["desc"] = "未验证事件候选：上涨中的放量与价格增长减速同时成立为1，完整输入但未成立为0。"
        data["spec"] = spec
        records.append(catalog.register_candidate(data))
    return {"idea_id": idea["definition"]["idea_id"], "family_id": family["definition"]["family_id"],
            "candidates": [{"candidate_id": r["candidate_id"], "validated_alpha": False} for r in records]}


def dispatch(args, cfg):
    command = args.research_command
    catalog = Catalog(cfg)
    if command in ("idea", "family", "candidate"):
        data = load_definition(_definition_path(cfg, args.file))
        result = getattr(catalog, "register_" + command)(data)
    elif command == "init-example":
        result = init_example(cfg)
    elif command == "list":
        records = catalog.records(args.kind)
        if args.kind == "candidates":
            result = [{**r, "current_state": catalog.state(r["candidate_id"])} for r in records]
        else:
            result = records
    elif command == "run":
        from .runner import run_candidate
        result = run_candidate(cfg, args.candidate_id, start=args.start, end=args.end, jobs=args.jobs)
    elif command == "evaluate":
        from .evaluation import evaluate_run
        result = evaluate_run(cfg, args.run_id)
    elif command == "verify":
        from .runner import verify_candidate
        result = verify_candidate(cfg, args.run_id, jobs=args.jobs, cutoff=args.cutoff)
    elif command == "promote":
        from .lifecycle import promote
        extra = load_definition(_definition_path(cfg, args.external_evidence)) if args.external_evidence else None
        result = promote(cfg, args.candidate_id, args.to, args.reason, args.evaluation, args.verification, extra)
    elif command == "splits":
        from fea.engine import Engine
        from .catalog import immutable_json
        import uuid
        folder = project_path(cfg, "artifacts/research/split_plans")
        temporary = cfg.sandbox(folder / ("s_" + uuid.uuid4().hex))
        engine = Engine(temporary)
        end = min(int(args.end.replace("-", "")), engine.baseline_last_day()) if args.end else engine.baseline_last_day()
        p = load_protocol(cfg)
        days = engine.cal.between(int(args.start.replace("-", "")), end)
        result = {"protocol": p.to_dict(), "protocol_hash": p.identity, "folds": walk_forward_splits(days, p)}
        immutable_json(temporary.root / "split_plan.json", result)
    elif command == "selection":
        from .selection import run_selection
        result = run_selection(args, cfg)
        return int(result["status"] != "complete")
    elif command == "batch":
        from .batch import cli
        return cli(args, cfg)
    else:
        raise ValueError(command)
    print(json.dumps(clean_json(result), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0
