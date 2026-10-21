#!/usr/bin/env python3
"""破坏性删除工具：去掉每个对象「最近 N 个交易日」的数据行 + 同步 manifest。

用途（用户 2026-10-03 交办）：全量重构完成后，对全部对象删除最近 5 个交易日数据，
再跑无参数 `main.py` 增量补齐，验证「增量生成结果 == 全量生成结果」。

## 为什么必须同步 manifest（而不是只删行）

引擎恢复缺口走两条通道（`fea/engine.py plan()`）：
  · **coverage 缺口通道**：manifest.coverage 说「这些天没算过」→ 只重算缺口 + 回刷窗
    （~10 个日历日）。这才是「增量补齐」的语义，也是本工具要模拟的场景；
  · **文件核对通道**：文件 size/mtime 与 manifest 记录不符、内容 sha 也不符 → 判「输出
    损坏」→ 重算**整年**（合法但不是本对拍要测的路径）。
只删行不同步 manifest 会走第二条（整年重算）；本工具把 partitions[year] 的
rows/nonnull/min_date/max_date/file_identity/content_sha256 全部重算到「删除后文件」
的真实状态，并把 coverage 截断到删除日之前 —— 于是引擎看到的是一个**自洽的、这些天
从未算过**的状态，走第一条通道。

## 安全设计（破坏性操作，逐条对应理由）

  · **默认 dry-run**：只有 `--apply` 才写盘；dry-run 输出逐对象删除行数供先审；
  · **先备份再删**：被删行片段（parquet）+ 原分区整文件 + 原 manifest 三样先落
    `artifacts/backups/trim_<stamp>/`；失败可 `--restore` 按主键 upsert 回滚；
  · **原子写**：复用 `fea/store._atomic_write`（PID 化 tmp + os.replace）；
  · **写后回读校验**：`partition_content_sha256(回读) == 预期`，错则不落 manifest；
  · **幂等**：天已不在（重跑）→ deleted_rows=0，不写盘、不动 manifest；
  · **互斥**：启动前 `main._running_pids()` + `daily_inflight` 检查；同时
    `main._mutating_cli` 已把本脚本 `--apply` 认作改写者，跑批期间 `main.py run` 会拒绝启动。

    用法：
    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/trim_recent_days.py                      # dry-run 预览（默认 5 日）
    $PY scripts/trim_recent_days.py --apply              # 真删（含前置备份）
    $PY scripts/trim_recent_days.py --restore <backup_dir>   # 从备份回滚
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fea import config as cfg_mod            # noqa: E402
from fea import store                        # noqa: E402  # 只 import 复用，不改（在数值内核清单里）
from fea.dates import int_to_str, str_to_int  # noqa: E402
from fea.manifest import Manifest, _atomic_json  # noqa: E402
from fea.release import mark_dirty, production_specs  # noqa: E402
from fea.spec import all_specs               # noqa: E402

import factors                               # noqa: F401,E402


def _guard_no_writers(cfg) -> None:
    """与 `main.py prune-history` 同款前置：有别的改写者/未完成批次就拒绝开工。"""
    from main import _running_pids
    if _running_pids():
        raise RuntimeError("已有因子更新或重建在运行，拒绝同时删除数据。")
    if (cfg.state_dir / "daily_inflight.json").exists():
        raise RuntimeError("存在未完成的批次日记（daily_inflight.json）；先跑 main.py 恢复，再动数据。")


def _pick_specs(cfg, factors_arg: list[str] | None):
    prod = {s.name for s in production_specs(cfg)}
    specs = [s for s in all_specs() if s.enabled and s.name in prod]
    if factors_arg:
        want = set(factors_arg)
        unknown = want - {s.name for s in specs}
        if unknown:
            raise ValueError(f"未知或不在生产的对象：{sorted(unknown)}")
        specs = [s for s in specs if s.name in want]
    return specs


def _sync_manifest(man: Manifest, year: int, frame: pd.DataFrame, deleted_rows: int,
                   file_path: Path, sha256: str, by: str = "trim_recent_days") -> None:
    """把 manifest 重写到「与删除后文件自洽」的状态（字段口径照 engine._finalize）。"""
    if len(frame):
        nonnull = int(np.isfinite(frame["value"].to_numpy(dtype="float32")).sum())
        min_d, max_d = str(frame["trade_date"].min()), str(frame["trade_date"].max())
    else:
        nonnull, min_d, max_d = 0, None, None
    man.mark_partition(year, len(frame), nonnull, min_d, max_d)
    part = man.partitions[str(year)]
    # 诚实记录：这是删除/回滚操作，不是引擎写盘；identical=False 说明分区内容确实变了。
    part["last_write"] = {"factor": man.name, "year": year, "identical": False,
                          "deleted_rows": int(deleted_rows), "by": by}
    st = file_path.stat()
    part["file_identity"] = [st.st_size, st.st_mtime_ns]
    part["content_sha256"] = sha256


def _trim_coverage(man: Manifest, first_deleted: str, prev_day: str) -> None:
    """coverage 截断：被删天（尾部连续段）从覆盖里去掉。

    语义：删除集合是「截至 cutoff 的最近 N 个交易日」，它们构成日历尾部的连续段。
    对每个覆盖区间 [a,b]：b 早于 first_deleted 的整段保留；否则保留 [a, prev_day]
    （a 晚于 prev_day 时该段整体落在被删区，丢弃）。
    """
    kept = []
    for a, b in man.coverage:
        if b < first_deleted:
            kept.append([a, b])
        elif a <= prev_day:
            kept.append([a, prev_day])
    man.coverage = kept


def _prev_trading_day(cfg, day_s: str) -> str:
    """`day_s` 的前一个交易日（用上游日历，避免用自然日差 1 天的坑）。"""
    from fea.engine import Engine
    cal = Engine(cfg).cal
    d = str_to_int(day_s)
    pos = int(np.searchsorted(cal.days, d, side="left"))
    if pos == 0:
        raise ValueError(f"{day_s} 之前没有交易日")
    return int_to_str(int(cal.days[pos - 1]))


def run_trim(args, cfg) -> int:
    from fea.engine import Engine, partition_content_sha256
    eng = Engine(cfg)
    end_i = str_to_int(args.cutoff) if args.cutoff else eng.baseline_last_day()
    cutoff_s = int_to_str(end_i)
    # 最近 N 个交易日（截至 cutoff 含；与 dayhash 同口径）
    from fea.dayhash import recent_trading_days
    days = [int_to_str(int(d)) for d in recent_trading_days(cfg, end_i, args.days)]
    days_set = sorted(set(days))
    first_deleted = days_set[0]
    prev_day = _prev_trading_day(cfg, first_deleted)

    specs = _pick_specs(cfg, args.factors)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = (Path(args.backup_dir) if args.backup_dir
                  else ROOT / "artifacts" / "backups" / f"trim_{stamp}")
    backup_dir = backup_dir if backup_dir.is_absolute() else (ROOT / backup_dir)
    backup_dir = backup_dir.resolve()
    if not backup_dir.is_relative_to(ROOT):
        raise ValueError("备份目录必须在项目内")

    print(f"目标：{len(specs)} 个对象 · 删除 {len(days_set)} 个交易日 "
          f"{days_set[0]}..{days_set[-1]} · cutoff={cutoff_s} · "
          f"{'APPLY（将写盘）' if args.apply else 'dry-run（只看）'}")
    print(f"备份目录：{backup_dir}")

    _guard_no_writers(cfg)
    if args.apply:
        # 一开始就标脏：删除期间下游 read_release 必须被拒绝。
        mark_dirty(cfg, "trim_recent_days")
        (backup_dir / "fragments").mkdir(parents=True, exist_ok=True)
        (backup_dir / "partitions").mkdir(parents=True, exist_ok=True)
        (backup_dir / "state_manifests").mkdir(parents=True, exist_ok=True)

    ledger = {"tool": "trim_recent_days", "stamp": stamp, "apply": bool(args.apply),
              "cutoff": cutoff_s, "days": days_set, "objects": [], "missing": []}
    # 按年分组（本场景全在 2026；通用化处理跨年）
    by_year: dict[int, list[str]] = {}
    for d in days_set:
        by_year.setdefault(int(d[:4]), []).append(d)

    for si, spec in enumerate(specs, 1):
        root = cfg.factor_root(spec)
        rec = {"name": spec.name, "parts": []}
        any_del = 0
        for year, ydays in sorted(by_year.items()):
            path = store.year_path(root, spec.name, year)
            if not path.exists():
                rec["parts"].append({"year": year, "deleted_rows": 0, "note": "分区不存在"})
                continue
            df = store.read_year(root, spec.name, year)
            mask = df["trade_date"].isin(ydays)
            frag = df[mask].copy()
            keep = df[~mask].copy()
            n_del = int(len(frag))
            if n_del == 0:
                rec["parts"].append({"year": year, "deleted_rows": 0, "note": "本分区无该日数据"})
                continue
            part_rec = {"year": year, "deleted_rows": n_del, "rows_before": int(len(df)),
                        "rows_after": int(len(keep)),
                        "sha256_before": partition_content_sha256(df, already_sorted=False)}
            if not args.apply:
                rec["parts"].append(part_rec)
                any_del += n_del
                continue
            # —— 1) 备份：片段 + 原分区整文件 + 原 manifest ——
            frag_path = backup_dir / "fragments" / f"{spec.name}.year={year}.parquet"
            frag.to_parquet(frag_path, compression=cfg.compression)
            shutil.copy2(path, backup_dir / "partitions" / f"{spec.name}.year={year}.parquet")
            shutil.copy2(cfg.state_dir / f"{spec.name}.json",
                         backup_dir / "state_manifests" / f"{spec.name}.json")
            # —— 2) 原子写回（去掉被删天）——
            store._atomic_write(keep, path, compression=cfg.compression)
            # —— 3) 回读校验：内容 sha 必须等于对 keep 的预期 ——
            back = store.read_year(root, spec.name, year)
            sha_now = partition_content_sha256(back, already_sorted=False)
            sha_want = partition_content_sha256(keep, already_sorted=False)
            if sha_now != sha_want:
                raise RuntimeError(f"{spec.name} year={year} 写后回读不一致；已中止（备份在 {backup_dir}）")
            # —— 4) 同步 manifest（自洽化：这些天变成『从未算过』）——
            man = Manifest.load(cfg.state_dir, spec.name)
            _sync_manifest(man, year, back, n_del, path, sha_now)
            _trim_coverage(man, first_deleted, prev_day)
            man.save()
            part_rec.update({"sha256_after": sha_now,
                             "size_after": path.stat().st_size,
                             "fragment": str(frag_path.relative_to(ROOT))})
            rec["parts"].append(part_rec)
            any_del += n_del
        ledger["objects"].append(rec)
        if not any_del:
            ledger["missing"].append(spec.name)
        if si % 30 == 0 or si == len(specs):
            print(f"  [{si}/{len(specs)}] 已处理", flush=True)

    total_del = sum(p["deleted_rows"] for o in ledger["objects"] for p in o["parts"])
    ledger["total_deleted_rows"] = int(total_del)
    ledger["objects_with_delete"] = int(sum(
        1 for o in ledger["objects"] if any(p["deleted_rows"] for p in o["parts"])))
    if args.apply:
        _atomic_json(ledger, backup_dir / "trim_ledger.json")
    else:
        out = ROOT / "artifacts" / "audits" / f"trim_dryrun_{stamp}.json"
        _atomic_json(ledger, out)
        print(f"dry-run 清单：{out}")
    print(f"共删除 {total_del:,} 行 · 涉及 {ledger['objects_with_delete']}/{len(specs)} 个对象"
          + (f" · 备份 {backup_dir}" if args.apply else "（未写盘）"))
    if ledger["missing"]:
        print(f"⚠ {len(ledger['missing'])} 个对象在这几天没有数据：{ledger['missing'][:8]}"
              + (" …" if len(ledger["missing"]) > 8 else ""))
    return 0


def run_restore(args, cfg) -> int:
    """从备份目录回滚：片段按主键 upsert 回当前分区（不整文件覆盖，
    容忍删除之后其它天的合法重算），再按实际文件重写 manifest 分区字段，
    coverage 采用备份中的原值。"""
    from fea.engine import partition_content_sha256
    backup_dir = Path(args.restore)
    backup_dir = backup_dir if backup_dir.is_absolute() else (ROOT / backup_dir)
    backup_dir = backup_dir.resolve()
    ledger_path = backup_dir / "trim_ledger.json"
    if not ledger_path.exists():
        raise FileNotFoundError(f"备份目录缺少 trim_ledger.json：{backup_dir}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    _guard_no_writers(cfg)

    n_restored = 0
    for rec in ledger["objects"]:
        name = rec["name"]
        for part in rec["parts"]:
            if not part.get("deleted_rows"):
                continue
            year = int(part["year"])
            frag_path = ROOT / part["fragment"]
            frag = pd.read_parquet(frag_path)
            # 找到对象的 factor_root（市场/股票）
            spec = next(s for s in all_specs() if s.name == name)
            root = cfg.factor_root(spec)
            path = store.year_path(root, name, year)
            cur = store.read_year(root, name, year)
            keep = cur[~cur["trade_date"].isin(set(frag["trade_date"]))]
            merged = pd.concat([keep, frag], ignore_index=True)
            merged = merged.sort_values(["trade_date", "stock_code"] if "stock_code" in merged.columns
                                        else ["trade_date"], kind="stable").reset_index(drop=True)
            for c, t in (store.MARKET_DTYPES if list(merged.columns) == store.MARKET_COLUMNS
                         else store.DTYPES).items():
                merged[c] = merged[c].astype(t)
            store._atomic_write(merged, path, compression=cfg.compression)
            back = store.read_year(root, name, year)
            sha = partition_content_sha256(back, already_sorted=False)
            man = Manifest.load(cfg.state_dir, name)
            bak_man_path = backup_dir / "state_manifests" / f"{name}.json"
            if bak_man_path.exists():
                bak = json.loads(bak_man_path.read_text(encoding="utf-8"))
                man.coverage = bak.get("coverage", man.coverage)
            _sync_manifest(man, year, back, int(part["deleted_rows"]), path, sha, by="trim_restore")
            man.save()
            n_restored += 1
    mark_dirty(cfg, "trim_restore")
    print(f"已回滚 {n_restored} 个 (对象,年) 分区；coverage 采用备份原值。"
          f"建议随后跑 main.py check 核对。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="破坏性删除最近 N 个交易日数据 + 同步 manifest（默认 dry-run）")
    ap.add_argument("--days", type=int, default=5, help="删除最近几个交易日（默认 5）")
    ap.add_argument("--cutoff", default=None, help="截止日 YYYY-MM-DD（默认 = 上游最后一天）")
    ap.add_argument("--factors", nargs="*", default=None, help="只处理这些对象（默认 = 全部生产对象）")
    ap.add_argument("--apply", action="store_true", help="★ 真正写盘（默认只预览）")
    ap.add_argument("--dry-run", action="store_true", help="显式 dry-run（与默认同义）")
    ap.add_argument("--backup-dir", default=None, help="备份目录（默认 artifacts/backups/trim_<stamp>）")
    ap.add_argument("--restore", default=None, metavar="DIR", help="从备份目录回滚")
    args = ap.parse_args()
    if args.days < 1:
        raise ValueError("--days 必须大于 0")
    cfg = cfg_mod.load()
    if args.restore:
        return run_restore(args, cfg)
    if args.dry_run:
        args.apply = False
    return run_trim(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
