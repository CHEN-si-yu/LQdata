#!/usr/bin/env python
"""接口台账生成器 —— 给「所有还在更新的接口」建一份可天天刷新的台账。

    python scripts/api_ledger.py                 # 写到 logs/ledger/LOG<MMDD>（MMDD = T）
    python scripts/api_ledger.py --name LOG0916  # 指定目录名
    python scripts/api_ledger.py --out /path/to  # 指定落点

产出（与 `featureengineering/logMMDD/` 的约定对齐：tsv + meta.json，另加一份人读的 README）：

    logs/ledger/LOG0916/
      ├── interfaces.tsv   一行一个接口：路径 / 数据集 / 模式 / 频率 / delay /
      │                    行数 / 水位 / 主键 / 列数 / 每轮请求 …
      ├── columns.tsv      一行一列：接口 → 列序 + 列名（数据字典）
      ├── meta.json        机器可读的汇总（同 factor 侧的 meta.json 风格）
      └── README.md        人读版（分组表格 + 口径说明 + 怎么刷新）

★ 2026-09-19：落点由 `everyday_tasks/LOG<MMDD>/` 改为 **`logs/ledger/LOG<MMDD>/`**
  （用户要求"LOG 文件也整理到一起"；顶层只留 README.md）。三份旧台账已搬入，内容未动。

**零 API 请求**：数据全部来自 `datadownload/state/*.json`（manifest）与注册表；
「每轮请求」取自最近一次**完整跑批**的日志（脚本会记录来源文件名）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_incremental import paths, registry as R        # noqa: E402
from data_incremental.core import state                  # noqa: E402

# 频率分组（人读版的顺序）
GROUPS = [("daily_full", "日频 · 全量发布"), ("daily_sparse", "日频 · 稀疏发布"),
          ("minute", "分钟级"), ("quarterly", "季频"), ("irregular", "不定期"),
          ("snapshot", "快照 · 周期刷新")]

_SUMMARY_RE = re.compile(
    r"^\s{2,}(\S+)\s+([\d,]+)\s+(\d+)\s+(\S+)\s+(✔|⚠️|✘|⊘)\s*$")


def _requests_from_logs() -> tuple[dict[str, int], str]:
    """从最近一次**完整跑批**的日志里取每表请求数（找不到就返回空）。"""
    logs = sorted(paths.LOGS.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in logs[:12]:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        got: dict[str, int] = {}
        for ln in text.splitlines():
            m = _SUMMARY_RE.match(ln)
            if m and m.group(1) in R.REGISTRY:
                got[m.group(1)] = int(m.group(3))
        if len(got) >= 25:                     # 一次完整跑批应覆盖 30+ 张表
            return got, p.name
    return {}, "（未找到完整跑批日志）"


def _latest_T() -> str:
    """台账命名用的 **T = 数据日**（不是日历上的"今天"）。

    ★ 口径：取各表 manifest 水位（`max_partition_date`）的最大值、且不超过今天。
      为什么不用"日历里今天及以前最后一个交易日"：那样跑在 09-17 上午会得到
      **09-17**（今天的行情还没发布），而实际最新数据日是 **09-16** ——
      台账目录名就对不上了（`basic_calendar` 的水位是 T+30 的**未来**日历，
      由 `<= today` 一并挡掉）。
    零请求：只读本地 manifest。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    best = None
    for ds in R.enabled():
        try:
            mx = state.Manifest.load(ds.name).max_partition_date()
        except Exception:  # noqa: BLE001
            continue
        if mx and str(mx)[:10] <= today and (best is None or str(mx)[:10] > best):
            best = str(mx)[:10]
    if best:
        return best
    cal = state.effective_calendar()
    cand = [d for d in cal if d <= today]
    return cand[-1] if cand else today


def collect() -> dict:
    req, src = _requests_from_logs()
    rows, cols = [], []
    for ds in R.all_ds():
        man = state.Manifest.load(ds.name) if ds.enabled else None
        part = man.partitions if man else {}
        rows.append({
            "path": ds.path,
            "dataset": ds.name,
            "enabled": int(bool(ds.enabled)),
            "mode": ds.mode,
            "pull_axis": ds.pull_axis,
            "freq": ds.freq or "-",
            "delay_days": int(ds.delay_days or 0),
            "tail_window": ds.window(5),
            "revision_days": (ds.revision_days if ds.revision_days is not None else 0),
            "date_field": ds.date_field or "-",
            "ledger_field": ds.ledger_date_field() or "-",
            "keys": ",".join(ds.keys),
            "rows": sum(int(v.get("rows") or 0) for v in part.values()) if man else 0,
            "partitions": len(part),
            "data_min": (man.min_partition_date() if man else None) or "-",
            "data_max": (man.max_partition_date() if man else None) or "-",
            "n_cols": len(man.columns) if man else 0,
            "req_per_round": req.get(ds.name, 0),
            "snapshot_every_days": int(getattr(ds, "snapshot_every_days", 1) or 1),
        })
        if man and man.columns:
            for i, c in enumerate(man.columns, 1):
                cols.append((ds.name, ds.path, i, c))
    return {"interfaces": rows, "columns": cols, "req_source": src,
            "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


TSV_HEAD = ("# 接口台账 · 数据侧（一个数据集 = 一个接口）\n"
            "# computed_at={at}  T={T}\n"
            "# 行数/水位取自各表 manifest（datadownload/state/<数据集>.json）；"
            "req_per_round 取自最近一次完整跑批日志：{src}\n"
            "# enabled=0 = 已停更（不再参与日更）\n"
            "path\tdataset\tenabled\tmode\tpull_axis\tfreq\tdelay_days\ttail_window"
            "\trevision_days\tdate_field\tledger_field\tkeys\trows\tpartitions"
            "\tdata_min\tdata_max\tn_cols\treq_per_round\tsnapshot_every_days\n")


def write_ledger(data: dict, outdir: Path, T: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    ints = data["interfaces"]

    # ---- interfaces.tsv
    lines = [TSV_HEAD.format(at=data["computed_at"], T=T, src=data["req_source"])]
    for r in sorted(ints, key=lambda x: (not x["enabled"], x["path"])):
        lines.append("\t".join(str(r[k]) for k in (
            "path", "dataset", "enabled", "mode", "pull_axis", "freq", "delay_days",
            "tail_window", "revision_days", "date_field", "ledger_field", "keys",
            "rows", "partitions", "data_min", "data_max", "n_cols",
            "req_per_round", "snapshot_every_days")) + "\n")
    (outdir / "interfaces.tsv").write_text("".join(lines), encoding="utf-8")

    # ---- columns.tsv（数据字典：接口 → 列）
    cl = ["# 列台账（数据字典）：dataset / path / 序号 / 列名\n"
          f"# computed_at={data['computed_at']}  共 {len(data['columns'])} 列\n"
          "dataset\tpath\tord\tcolumn\n"]
    cl += [f"{a}\t{b}\t{c}\t{d}\n" for a, b, c, d in data["columns"]]
    (outdir / "columns.tsv").write_text("".join(cl), encoding="utf-8")

    # ---- meta.json
    en = [r for r in ints if r["enabled"]]
    meta = {
        "computed_at": data["computed_at"],
        "T": T,
        "interfaces_enabled": len(en),
        "interfaces_stopped": len(ints) - len(en),
        "daily_gated": len([r for r in en if r["freq"] in ("daily_full", "daily_sparse")]),
        "rows_total": sum(r["rows"] for r in en),
        "requests_per_round": sum(r["req_per_round"] for r in en),
        "columns_total": len(data["columns"]),
        "requests_source": data["req_source"],
        "by_group": {k: {"n": len([r for r in en if r["freq"] == k]),
                         "rows": sum(r["rows"] for r in en if r["freq"] == k),
                         "requests": sum(r["req_per_round"] for r in en if r["freq"] == k)}
                     for k, _t in GROUPS},
        "stopped": [{"path": r["path"], "dataset": r["dataset"]} for r in ints if not r["enabled"]],
        "note": "数据侧接口台账；与 featureengineering/logMMDD（因子横截面台账）同约定，互不重叠",
    }
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")

    # ---- README.md（人读）
    md = [f"# 接口台账 · T = {T}", "",
          f"生成时间：{data['computed_at']}　|　生成脚本：`everyday_tasks/scripts/api_ledger.py`"
          f"（**零 API 请求**，随时可重跑）", "",
          f"- **仍在更新：{meta['interfaces_enabled']} 个接口 / {meta['rows_total']:,} 行**"
          f"（其中闸门管着 {meta['daily_gated']} 张日频表）",
          f"- 一轮完整更新约 **{meta['requests_per_round']} 次请求**"
          f"（请求数取自 `{data['req_source']}`）",
          f"- 已停更：{meta['interfaces_stopped']} 个接口（见文末）",
          "- 机器可读：`interfaces.tsv`（一行一接口）· `columns.tsv`（一行一列）· `meta.json`",
          ""]
    for key, title in GROUPS:
        items = sorted([r for r in en if r["freq"] == key], key=lambda x: -x["rows"])
        if not items:
            continue
        md += [f"## {title}（{len(items)} 个接口 / {sum(r['rows'] for r in items):,} 行）", "",
               "| 接口 | 数据集 | 行数 | 最早 | 最新 | 模式 | delay | 每轮请求 |",
               "|:--|:--|--:|:--|:--|:--|--:|--:|"]
        for r in items:
            every = f"（每 {r['snapshot_every_days']} 天刷新）" if r["snapshot_every_days"] > 1 else ""
            req = r["req_per_round"]
            req_s = f"{req}~1*" if r["dataset"] == "stock_history_5min" and req == 0 else str(req)
            md.append(f"| `{r['path']}` | {r['dataset']} | {r['rows']:,} | {r['data_min']} | "
                      f"{r['data_max']} | {r['mode']}{every} | {r['delay_days']} | "
                      f"{req_s} |")
        md.append("")
    md += ["> `*` = **daily_dump 通道**：本地按日缓存命中就 **0 请求**，出现新的完整交易日才 1 个请求"
           "（这正是它一行顶 5,901 个请求的原因）。",
           "> `basic_calendar` 的「最新」是 **T+30 的未来日历**（刻意更新到未来，供闸门回推 expected），"
           "不是行情水位。",
           "> 标「每 N 天刷新」的快照表，未到周期时每轮请求数为 0（自动跳过，见 `strategies._period_skip`）。",
           ""]
    md += ["## 已停更（不参与日更）", ""]
    for r in ints:
        if not r["enabled"]:
            md.append(f"- `{r['path']}` → {r['dataset']}（本地 {r['rows']:,} 行）")
    md += ["", "## 口径说明", "",
           "- **行数 / 水位**取自各表 manifest（`datadownload/state/<数据集>.json`），"
           "与磁盘实测一致（`main.py hash-audit` 可复核）。",
           "- **每轮请求**是最近一次完整跑批的真实值；表多、窗口小是设计目标"
           "（`batch`/`按日查`/`报告期查` 等优化见 `README.md` §2.4）。",
           "- **tail_window** = 每轮无条件重抓的交易日数（自愈纵深）；"
           "**revision_days** = 低频表的额外回刷窗口（日历天）。",
           "- **delay_days** = 该表比最后交易日晚上市几天（闸门等它的依据）。",
           "- 刷新这本台账：`python scripts/api_ledger.py`（默认写到 `logs/ledger/LOG<T 的 MMDD>/`）。",
           ""]
    (outdir / "README.md").write_text("\n".join(md), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="接口台账生成（零 API 请求）")
    ap.add_argument("--name", default=None, help="目录名（默认 LOG<MMDD>，MMDD 取 T）")
    ap.add_argument("--out", default=None, help="落点目录（默认 everyday_tasks/logs/ledger/<name>）")
    ap.add_argument("--date", default=None, help="指定 T（默认按本地日历推）")
    a = ap.parse_args(argv)

    T = a.date or _latest_T()
    name = a.name or ("LOG" + T[5:7] + T[8:10])
    outdir = Path(a.out) if a.out else (ROOT / "logs" / "ledger" / name)

    data = collect()
    write_ledger(data, outdir, T)
    meta = json.loads((outdir / "meta.json").read_text(encoding="utf-8"))
    print(f"✔ 接口台账已写入 {outdir}")
    print(f"   T={T} · 在更新 {meta['interfaces_enabled']} 个接口 / {meta['rows_total']:,} 行"
          f" · 一轮 {meta['requests_per_round']} 请求 · 列 {meta['columns_total']}")
    print(f"   文件：interfaces.tsv / columns.tsv / meta.json / README.md"
          f"（请求数来源：{meta['requests_source']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
