#!/usr/bin/env python3
"""冷启动：从 `conf/research/` 的 YAML 恢复研究定义目录（`artifacts/research/catalog/`）。

## 为什么需要它（2026-10-03 净室测试暴露）

正式生产清单 `conf/production.json` 里有 **4 个研究来源对象**（`s26_*` / `r2own_*` /
`r3b_*`），它们按 `candidate_id` 引用 `artifacts/research/catalog/` 里的定义。
把 `data/`、`state/`、`artifacts/` 全部清空后，`python main.py` 会在 3 秒内以
`KeyError: Unknown candidates: ...` 崩溃 —— 这是「清数据重来」唯一需要手工补的一步。

定义的**源头**是 `conf/research/` 下的 YAML（属于配置、随代码保留）；`candidate_id`
是定义内容的确定性哈希，所以按 `idea → family → candidate` 顺序重放注册即可恢复，
且恢复出的 id 与 `production.json` 的引用逐字一致（本脚本会核对）。

## 用法（清空数据之后、`python main.py` 之前跑一次；幂等）

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/restore_research_catalog.py           # 重放注册
    $PY scripts/restore_research_catalog.py --check   # 只核对：生产清单里的研究对象是否都已就绪
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml                                              # noqa: E402


def _kind(raw: dict) -> str | None:
    """按**字段组合**判定定义类型（顺序敏感）：

    · candidate：有 `name` 且有 `spec`；
    · family：有 `family_id`、**没有** `name`（family 也会带 `idea_id`——曾因此被
      误判成 idea，导致 family 从未注册、candidate 全部 `Unknown families` 失败）；
    · idea：有 `idea_id` 且没有 `family_id`。
    """
    if "name" in raw and "spec" in raw:
        return "candidate"
    if "family_id" in raw and "name" not in raw:
        return "family"
    if "idea_id" in raw and "family_id" not in raw:
        return "idea"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="从 conf/research 恢复研究定义目录")
    ap.add_argument("--check", action="store_true", help="只核对生产清单引用的研究对象是否就绪")
    ap.add_argument("--all", action="store_true",
                    help="重放 conf/research 下**全部**定义（默认只恢复正式清单引用的那几条链，"
                         "研究轮次里未随 conf/ 保留 family 的孤儿子集会注册失败，属预期）")
    args = ap.parse_args()

    import factors                                        # noqa: F401
    from fea import config as cfg_mod
    from fea.release import production_specs
    from research.catalog import Catalog

    cfg = cfg_mod.load()

    if args.check:
        try:
            specs = production_specs(cfg)
        except Exception as exc:                          # noqa: BLE001
            print(f"✘ 正式清单解析失败：{exc}")
            return 1
        print(f"✔ 正式清单解析通过：{len(specs)} 个对象（含研究来源）")
        return 0

    files = sorted((ROOT / "conf" / "research").rglob("*.yaml"))
    parsed: dict[str, tuple[Path, dict]] = {}
    for p in files:
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:                                 # noqa: BLE001
            continue
        if isinstance(raw, dict) and (k := _kind(raw)):
            parsed[str(p)] = (p, {"kind": k, **raw})

    wanted: set[str] | None = None
    if not args.all:
        # 只恢复正式清单引用到的链：对象名形如 `<candidate.name>_<candidate_id前16位>`。
        import json as _json
        prod = _json.loads((ROOT / "conf" / "production.json").read_text(encoding="utf-8"))
        names = [o["name"] for o in prod["objects"] if o.get("candidate_id")]
        wanted = set()
        for path, (p, raw) in parsed.items():
            if raw["kind"] != "candidate":
                continue
            cand_name = str(raw.get("name", ""))
            if cand_name and any(n == cand_name or n.startswith(cand_name + "_") for n in names):
                wanted.add(path)
                fam = str(raw.get("family_id", ""))
                for p2, r2 in ((pp, rr) for pp, rr in parsed.values()):
                    if r2["kind"] == "family" and str(r2.get("family_id")) == fam:
                        wanted.add(str(p2))
                        idea = str(r2.get("idea_id", ""))
                        for p3, r3 in parsed.values():
                            if r3["kind"] == "idea" and str(r3.get("idea_id")) == idea:
                                wanted.add(str(p3))
        print(f"生产清单引用 {len(names)} 个研究来源对象 → 需恢复 {len(wanted)} 个定义文件")

    buckets: dict[str, list[Path]] = {"idea": [], "family": [], "candidate": []}
    for path, (p, raw) in parsed.items():
        if wanted is not None and path not in wanted:
            continue
        buckets[raw["kind"]].append(p)
    total = sum(len(v) for v in buckets.values())
    print(f"待注册 {total} 个定义文件（idea {len(buckets['idea'])} · family "
          f"{len(buckets['family'])} · candidate {len(buckets['candidate'])}）")
    registered = skipped = failed = 0
    for kind in ("idea", "family", "candidate"):
        for p in buckets[kind]:
            cmd = [sys.executable, str(ROOT / "main.py"), "research", kind,
                   "--file", str(p.relative_to(ROOT))]
            out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            if out.returncode == 0:
                registered += 1
            elif "已登记" in (out.stdout + out.stderr) or "已存在" in (out.stdout + out.stderr):
                skipped += 1
            else:
                failed += 1
                print(f"  ✘ {kind}: {p.name} —— {(out.stderr or out.stdout).strip().splitlines()[-1:]}")
    print(f"完成：新登记 {registered} · 已存在 {skipped} · 失败 {failed}")
    if failed:
        # 失败多为「研究轮次里 family 定义未随 conf/ 保留」的**非生产**候选 ——
        # 它们不影响正式生产清单；以生产清单能否解析为准（见下）。
        print(f"  ℹ 其中 {failed} 个失败若不在正式清单引用之列，可忽略（研究轮次的孤儿子集）")
    # 最后用生产清单自证 —— 这才是冷启动是否可跑的判据
    try:
        specs = production_specs(cfg)
        print(f"✔ 正式清单解析通过：{len(specs)} 个对象（含研究来源）")
    except Exception as exc:                              # noqa: BLE001
        print(f"✘ 正式清单仍解析失败：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
