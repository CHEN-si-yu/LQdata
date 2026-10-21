#!/usr/bin/env python
"""从 `state/rebuild_attempts/<stamp>.json` 的 `coverage_before` 证据恢复 manifest coverage。

背景（2026-10-05 全量重跑事故的根因之一）：
    `fea.executor.run_rebuild(force=True)`（main.py rebuild 默认）会先把区间内的
    coverage **清空**（防止 --resume 把"未刷新完成的年"误判成已完成），并把清空前
    的 coverage 存进 `state/rebuild_attempts/<stamp>.json` 的 `coverage_before`。
    若该重建被中断且只回补了一部分年份，manifest 的 coverage 就残留成缺口状态 ⇒
    之后的**任何**日常运行都会把"其实已经算好、文件完好"的年份重新排进计划
    （写盘层会跳过相同内容，但 CPU 已经花掉 —— 即"总是全量生成"）。

本工具做的事（只读证据 + 核验后回写 manifest，不做任何计算）：
    1. 读 `coverage_before`（或 --release 发布清单）里的日期区间；
    2. 对每个 (因子, 年份) 分区做**核验**，三条通道按序：
       a) 发布清单 sha256（--release 提供时最强证据）
       b) file_identity（文件字节级未动过）与 manifest 分区记录一致
       c) 内容核对 `Engine._partition_matches_record`（行数/日期界/内容 sha256 全同）
       全部通道都不通过的年份：**不恢复**，记入报告由正常修复机制处理。
    3. 核验通过的区间 `add_coverage()` 回写 manifest（仅并集，不删除已有覆盖）。

默认 **dry-run**（只预览）；加 `--apply` 才写。报告写 `artifacts/audits/coverage_restore_<stamp>.json`。

用法：
    python scripts/restore_coverage_from_attempt.py --attempt state/rebuild_attempts/20261005_225036_419285.json
    python scripts/restore_coverage_from_attempt.py --attempt <file> --release artifacts/releases/<id>.json --apply
    python scripts/restore_coverage_from_attempt.py --attempt <file> --factors momentum_20 roe_ttm --apply
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from fea import config as cfg_mod  # noqa: E402
from fea.engine import Engine, partition_content_sha256  # noqa: E402
from fea.manifest import Manifest  # noqa: E402


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_release(path: Path) -> dict:
    """发布清单：{(factor, year): {'sha256':..., 'bytes':..., 'mtime_ns':...}}"""
    rel = json.loads(path.read_text())
    out = {}
    for obj in rel.get("objects", []):
        for part in obj.get("partitions", []):
            year = None
            p = str(part.get("path", ""))
            for seg in p.split("/"):
                if seg.startswith("year="):
                    year = int(seg.split("=")[1])
            if year is not None:
                out[(obj["name"], year)] = part
    return out


def spec_dir(cfg, name: str) -> Path:
    from fea.spec import REGISTRY
    spec = REGISTRY[name]
    return cfg.factor_root(spec) / name


def verify_partition(cfg, eng: Engine, spec_name: str, year: int, man: Manifest,
                     release: dict) -> tuple[bool, str]:
    """返回 (是否可信, 通道说明)。"""
    path = spec_dir(cfg, spec_name) / f"year={year}" / "data.parquet"
    if not path.exists():
        return False, "file_missing"
    part = release.get((spec_name, year))
    if part is not None and part.get("sha256"):
        return (_file_sha256(path) == part["sha256"]), "release_sha256"
    meta = (man.partitions or {}).get(str(year))
    if not meta:
        return False, "no_partition_record"
    ident = meta.get("file_identity")
    if ident:
        st = path.stat()
        if list(ident) == [st.st_size, st.st_mtime_ns]:
            return True, "file_identity"
    try:
        if eng._partition_matches_record(path, meta):
            return True, "content_sha256"
    except Exception as exc:  # noqa: BLE001
        return False, f"content_check_error:{exc!r}"
    return False, "identity_and_content_mismatch"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--attempt", required=True, help="state/rebuild_attempts/<stamp>.json")
    ap.add_argument("--release", default=None, help="（可选）发布清单 JSON，作最强内容核验")
    ap.add_argument("--factors", nargs="*", default=None, help="只处理这些因子")
    ap.add_argument("--apply", action="store_true", help="真正回写 manifest（默认 dry-run）")
    args = ap.parse_args()

    cfg = cfg_mod.load()
    import factors as _f  # noqa: F401  注册 REGISTRY
    from fea.release import production_specs  # 研究来源因子由它在 REGISTRY 中补齐
    production_specs(cfg)
    eng = Engine(cfg)
    attempt = json.loads(Path(args.attempt).read_text())
    coverage_before: dict = attempt.get("coverage_before", {})
    release = _load_release(Path(args.release)) if args.release else {}

    report = {"attempt": args.attempt, "release": args.release, "apply": bool(args.apply),
              "at": datetime.datetime.now().isoformat(timespec="seconds"),
              "factors": {}}
    applied = skipped = 0
    for name, spans in sorted(coverage_before.items()):
        if args.factors and name not in set(args.factors):
            continue
        if not spans:
            continue
        man = Manifest.load(cfg.state_dir, name)
        have = {tuple(s) for s in man.coverage}
        add, rejects = [], []
        for a, b in spans:
            if tuple([a, b]) in have:
                continue
            years = sorted({int(d[:4]) for d in (a, b)})
            years = list(range(years[0], years[-1] + 1))
            ok_all, why = True, []
            for y in years:
                ok, channel = verify_partition(cfg, eng, name, y, man, release)
                why.append(f"{y}:{channel}")
                if not ok:
                    ok_all = False
            if ok_all:
                add.append([a, b])
            else:
                rejects.append({"span": [a, b], "channels": why})
        if add or rejects:
            report["factors"][name] = {"add": add, "rejects": rejects}
        if add and args.apply:
            for a, b in add:
                man.add_coverage(a, b)
            man.save()
            applied += 1
        elif add:
            applied += 1
        if rejects:
            skipped += 1

    out_dir = cfg.project_path("artifacts", "artifacts") / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"coverage_restore_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[{'APPLY' if args.apply else 'DRY-RUN'}] 因子改动 {applied} 个 · 有未通过项 {skipped} 个 · 报告 {out}")
    for name, item in list(report["factors"].items())[:5]:
        print("  ", name, "add=", item["add"], "rejects=", item["rejects"][:2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
