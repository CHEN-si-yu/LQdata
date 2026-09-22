#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 `main.py dedup` 的机读报告转成**可交付的保守代表清单**（S-02，2026-09-21）。

## 为什么要有这一步

`main.py dedup` 的产物是 `state/dedup/report.json`：一份 653×653 的相关系数矩阵
加簇清单（约 3.5 MB）。它是**机读**的，但下游要的是两样东西：

  1. **一份能看懂的簇清单**（哪些因子在同一个簇里、|ρ| 范围多少）；
  2. **一份能直接用的代表清单**（每个簇留谁，其余是"候选删除"）。

## 为什么是"候选"而不是"删除"

平台纪律有两条卡在这里：
  · 断言失败也不删产物（本轮审计只是**去冗余的证据**，不是删除指令）；
  · 语义错 ≠ 高相关 —— 两个因子 |ρ|=0.98 只说明**在这段历史上**高度共线，
    不代表其中一个没有信息。删不删由模型侧在多种子配对里决定（S-02 原话即是如此）。

所以本脚本只产出**建议**，并且**明确不算 IC/收益**（那是模型侧的事）。

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/dedup_representatives.py                    # 打印 + 写 artifacts/audits/dedup_<日期>/
    $PY scripts/dedup_representatives.py --threshold 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fea.manifest import Manifest                              # noqa: E402


def _coverage(names: list[str]) -> dict[str, float | None]:
    """每个因子的覆盖率 = 它自己落地清单里 `nonnull` 合计占比（不读因子数据）。

    ★ 为什么要外挂这一项：dedup 的选代表口径是「覆盖率最高、其次 |RankIC| 最大」，
    但它的覆盖率来自 `state/eval/summary.json`（2026-09-18 版，只覆盖 306 个旧因子）。
    2026-09-21 之后新交付的因子（`afx_*`/`efx_*`/`mfx_*`/`qf_*`/`open5_*`）在里面
    没有这一项 ⇒ 同簇里全是新因子时，`keep` 是按注册顺序取的、与质量无关。
    这里用**本工程自己的落地清单**把覆盖率补齐，只用于核验 keep 是否保守，不改机器报告。
    """
    out: dict[str, float | None] = {}
    for nm in names:
        try:
            man = Manifest.load(ROOT / "state", nm)
            out[nm] = man.nonnull_ratio() if man.partition_rows() else None
        except Exception:                                      # noqa: BLE001
            out[nm] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(ROOT / "state" / "dedup" / "report.json"))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--date", default="20260921")
    a = ap.parse_args()

    d = json.loads(Path(a.report).read_text(encoding="utf-8"))
    clusters = d.get("clusters", [])
    names = list(d.get("corr", {}))
    rows_dropped = sum(c["size"] - 1 for c in clusters)

    # 覆盖率校核：keep 是不是该簇覆盖率最高的成员？不是的挑出来单列（见 _coverage 的说明）。
    cov = _coverage(sorted({n for c in clusters for n in [c["keep"], *c["drop"]]}))
    devi: list[tuple[dict, str, dict]] = []
    for c in clusters:
        vals = {m: cov[m] for m in [c["keep"], *c["drop"]] if cov.get(m) is not None}
        if not vals:
            continue
        best = max(vals, key=lambda k: vals[k])
        if best != c["keep"] and vals[best] - (vals.get(c["keep"]) or 0.0) > 1e-9:
            devi.append((c, best, vals))
    devi.sort(key=lambda t: -(t[2][t[1]] - (t[2].get(t[0]["keep"]) or 0.0)))

    lines: list[str] = []
    add = lines.append
    add(f"# {len(names)} 因子冗余审计：保守代表清单（S-02）\n")
    add(f"- 机读来源：`{a.report}`")
    add(f"- 覆盖：**{len(names)} 个因子** · 年份 {d.get('years')} · |ρ| ≥ {d.get('threshold')} 判为同簇")
    add(f"- 结果：**{len(clusters)} 个簇** · 建议保留 {len(clusters)} 个代表 · "
        f"**候选删除 {rows_dropped} 个**\n")
    add("> ★ 这是**去冗余的证据**，不是删除指令。平台纪律：断言失败也不删产物；")
    add("> 「高相关」只说明在这段历史上共线，删不删由模型侧在多种子配对里决定（S-02 原话）。")
    add("> ★ 本清单**不含任何 IC / 收益判断** —— 那不属于这一步。\n")

    if not clusters:
        add("## 结论\n")
        add(f"在 |ρ| ≥ {d.get('threshold')} 下**没有发现重复簇**。")
    else:
        add("## 簇清单（按簇大小降序）\n")
        add("| # | 大小 | 保留代表 | 同簇成员 | |ρ| 范围 | 完全重复对 |")
        add("|--:|--:|:--|:--|:--|:--|")
        for i, c in enumerate(sorted(clusters, key=lambda v: -v["size"]), 1):
            ident = c.get("identical_pairs") or []
            ident_s = "、".join(f"{x[0]}={x[1]}({x[2]})" for x in ident[:3]) or "—"
            add(f"| {i} | {c['size']} | `{c['keep']}` | {'、'.join('`'+x+'`' for x in c['drop'])} | "
                f"[{c['rho_min']:.3f}, {c['rho_max']:.3f}] | {ident_s} |")
        add("")
        add("## 覆盖率校核（保守口径）\n")
        add("> dedup 的选代表口径是「覆盖率最高、其次 |RankIC| 最大」，但它的覆盖率取自")
        add("> `state/eval/summary.json` —— 那是 **2026-09-18 版、只覆盖 306 个旧因子**。")
        add("> 2026-09-21 之后交付的新因子（`afx_*`/`efx_*`/`mfx_*`/`qf_*`/`open5_*`）在里面")
        add("> 没有覆盖率 ⇒ 同簇里全是新因子时，`keep` 是按注册顺序取的、与质量无关。")
        add("> 下表用**因子自己的落地清单**（`state/<因子>.json` 的 `nonnull` 合计占比）")
        add("> 把这一项补齐，只用于核验、不改机器报告。\n")
        if not devi:
            add(f"- {len(clusters)} 簇的 `keep` **都是**该簇覆盖率最高的成员。")
        else:
            add(f"- **{len(devi)}/{len(clusters)} 簇**的 `keep` 不是该簇覆盖率最高的成员；")
            add("  差值都很小，但若要严格按「覆盖率最高」取代表，把这几个换成右列：\n")
            add("| 簇大小 | 现 keep | 覆盖率 | 覆盖率最高成员 | 覆盖率 | Δ |")
            add("|--:|:--|--:|:--|--:|--:|")
            for c, best, vals in devi:
                k = c["keep"]
                add(f"| {c['size']} | `{k}` | {vals[k]:.1%} | `{best}` | {vals[best]:.1%} | "
                    f"{vals[best] - vals[k]:.1%} |")
        add("")
        add("## 规模分布\n")
        add("```")
        add(str(dict(sorted(Counter(c["size"] for c in clusters).items(), reverse=True))))
        add("```")
        add("")
        add("## 候选删除清单（每簇除代表外全部）\n")
        add("```")
        for x in sorted(n for c in clusters for n in c["drop"]):
            add(x)
        add("```")
        add("")
        add("## 代表清单（每簇留 1 个）\n")
        add("```")
        for x in sorted(c["keep"] for c in clusters):
            add(x)
        add("```")

    text = "\n".join(lines)
    if a.out_dir:
        out = ROOT / a.out_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "REPORT.md").write_text(text, encoding="utf-8")
        payload = {
            "source": str(a.report), "years": d.get("years"), "threshold": d.get("threshold"),
            "n_factors": len(names), "n_clusters": len(clusters),
            "keep": sorted(c["keep"] for c in clusters),
            "drop": sorted(n for c in clusters for n in c["drop"]),
            "clusters": clusters,
            "coverage": {k: (round(v, 6) if v is not None else None) for k, v in cov.items()},
            "keep_best_coverage": {c["keep"]: best for c, best, _ in devi},
            "note": "候选清单，不是删除指令；不含 IC/收益判断。"
                    "keep_best_coverage：覆盖率校核认为更该保留的成员（见 REPORT.md 同名节）。",
        }
        (out / "representatives.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {out}/REPORT.md 与 representatives.json")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
